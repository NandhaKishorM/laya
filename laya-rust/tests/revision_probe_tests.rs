//! Replays `tests/golden/routing/revision_probe.json`, recorded from Python's own
//! `laya.revisions.resolve_revision` / `verify_digests` and `laya.router._digests_from_env` /
//! `_merge_expected_digests` (corpus: `tools/revision_cases.py`, fixture files in
//! `tools/revision_fixtures/checkpoint`), against this crate. No model, no network.
//!
//! Every entry is replayed: the result must be the recorded value, or an error of the recorded
//! kind with the recorded text. A `ValueError` is any error but [`LayaError::DigestFileNotFound`],
//! which is the `FileNotFoundError`. The pinned table is compared with the recorded one, so it
//! cannot drift from `laya/revisions.py` without this file failing.

mod common;

use common::routing_golden;
use indexmap::IndexMap;
use laya::error::{LayaError, Result};
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{digests_from_env, merge_expected_digests};
use laya::revisions::{DigestGraph, PINNED_REVISIONS, REVIEWED, resolve_revision, verify_digests};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Sets (or clears, for `None`) the two variables the probe drives and restores the originals.
struct Env {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: [(&'static str, Option<String>); 2],
}

impl Env {
    fn new(revision: Option<&str>, digests: Option<&str>) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let names = ["LAYA_REVISION", "LAYA_SHA256_DIGESTS"];
        let saved = names.map(|n| (n, std::env::var(n).ok()));
        for (name, value) in names.iter().zip([revision, digests]) {
            // SAFETY: `ENV_LOCK` is held for this guard's whole life.
            match value {
                Some(v) => unsafe { std::env::set_var(name, v) },
                None => unsafe { std::env::remove_var(name) },
            }
        }
        Self { _lock: lock, saved }
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        for (name, value) in &self.saved {
            match value {
                Some(v) => unsafe { std::env::set_var(name, v) },
                None => unsafe { std::env::remove_var(name) },
            }
        }
    }
}

fn probe() -> Value {
    routing_golden::load("revision_probe.json")
        .expect("tests/golden/routing/revision_probe.json is part of the source tree")
}

fn entries<'a>(probe: &'a Value, section: &str) -> &'a Vec<Value> {
    let list = probe[section].as_array().expect("section is an array");
    assert!(!list.is_empty(), "{section} is empty");
    list
}

fn str_map(value: &Value) -> IndexMap<String, String> {
    value
        .as_object()
        .expect("object")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().expect("string value").to_string()))
        .collect()
}

/// Checks `outcome` against a recorded `{"ok": ..}` / `{"error_type": .., "error_message": ..}`.
/// On success `check_ok` compares the value; `dir` is the text Python's `{dir}` stands for.
fn check<T>(
    label: &str,
    recorded: &Value,
    outcome: Result<T>,
    dir: Option<&str>,
    check_ok: impl FnOnce(&T, &Value),
) {
    if let Some(error_type) = recorded.get("error_type") {
        let error = match outcome {
            Ok(_) => panic!("{label}: expected {error_type}, got Ok"),
            Err(e) => e,
        };
        let want_type = error_type.as_str().unwrap();
        let is_not_found = matches!(error, LayaError::DigestFileNotFound(_));
        assert_eq!(
            is_not_found,
            want_type == "FileNotFoundError",
            "{label}: error kind for {want_type}: {error}"
        );
        let mut text = error.to_string();
        if let Some(dir) = dir {
            text = text.replace(dir, "{dir}");
        }
        assert_eq!(
            text,
            recorded["error_message"].as_str().unwrap(),
            "{label}: error text"
        );
    } else {
        match outcome {
            Ok(value) => check_ok(&value, &recorded["ok"]),
            Err(e) => panic!("{label}: expected ok, got error: {e}"),
        }
    }
}

#[test]
fn pinned_table_and_reviewed_word_equal_the_recorded_ones() {
    let probe = probe();
    let recorded: Vec<(&str, &str)> = probe["pinned_revisions"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str().unwrap()))
        .collect();
    assert_eq!(PINNED_REVISIONS, recorded.as_slice());
    assert_eq!(REVIEWED, probe["reviewed"].as_str().unwrap());
}

#[test]
fn resolve_revision_replays_every_recorded_case() {
    let probe = probe();
    for entry in entries(&probe, "resolve") {
        let label = entry["label"].as_str().unwrap();
        let _env = Env::new(entry["env"].as_str(), None);
        let outcome =
            resolve_revision(entry["model"].as_str().unwrap(), entry["revision"].as_str());
        check(label, &entry["result"], outcome, None, |got, want| {
            assert_eq!(got.as_deref(), want.as_str(), "{label}");
        });
    }
}

#[test]
fn verify_digests_replays_every_recorded_case() {
    let probe = probe();
    for entry in entries(&probe, "verify") {
        let label = entry["label"].as_str().unwrap();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools")
            .join(
                entry["dir"]
                    .as_str()
                    .unwrap()
                    .replace('/', std::path::MAIN_SEPARATOR_STR),
            );
        let expected = entry["expected"]
            .is_object()
            .then(|| str_map(&entry["expected"]));
        let graph_path = entry["onnx_path"].as_str().map(|rel| dir.join(rel));
        let graph = graph_path.as_deref().map(DigestGraph::Fused);
        let _env = Env::new(None, entry["env"].as_str());
        let outcome = verify_digests(&dir, expected.as_ref(), graph);
        let dir_text = dir.display().to_string();
        check(
            label,
            &entry["result"],
            outcome,
            Some(&dir_text),
            |_, want| {
                assert!(want.is_null(), "{label}: verify returns nothing");
            },
        );
    }
}

#[test]
fn digests_from_env_replays_every_recorded_case() {
    let probe = probe();
    for entry in entries(&probe, "digests_from_env") {
        let label = entry["label"].as_str().unwrap();
        let models: Vec<LayaCheckpoint> = entry["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| laya::LayaRouter::normalise_name(m.as_str().unwrap()).unwrap())
            .collect();
        let _env = Env::new(None, entry["env"].as_str());
        let outcome = digests_from_env(&models);
        check(label, &entry["result"], outcome, None, |got, want| {
            let want = want.as_object().expect("recorded dict");
            assert_eq!(got.len(), want.len(), "{label}: checkpoints");
            for (name, inner) in want {
                let key = laya::LayaRouter::normalise_name(name).unwrap();
                let got_inner = got.get(&key).unwrap_or_else(|| panic!("{label}: {name}"));
                // Same files, same digests, same order.
                let got_pairs: Vec<(&String, &String)> = got_inner.iter().collect();
                let want_map = str_map(inner);
                let want_pairs: Vec<(&String, &String)> = want_map.iter().collect();
                assert_eq!(got_pairs, want_pairs, "{label}: {name}");
            }
        });
    }
}

#[test]
fn merge_expected_digests_replays_every_recorded_case() {
    let probe = probe();
    for entry in entries(&probe, "merge") {
        let label = entry["label"].as_str().unwrap();
        let shared = entry["shared"]
            .is_object()
            .then(|| str_map(&entry["shared"]));
        let per_model = entry["per_model"]
            .is_object()
            .then(|| str_map(&entry["per_model"]));
        let merged = merge_expected_digests(shared.as_ref(), per_model.as_ref());
        check(
            label,
            &entry["result"],
            Ok(merged),
            None,
            |got, want| match (got, want.is_object()) {
                (None, false) => {}
                (Some(got), true) => {
                    let want_map = str_map(want);
                    let got_pairs: Vec<_> = got.iter().collect();
                    let want_pairs: Vec<_> = want_map.iter().collect();
                    assert_eq!(got_pairs, want_pairs, "{label}");
                }
                (got, _) => panic!("{label}: got {got:?}, recorded {want}"),
            },
        );
    }
}
