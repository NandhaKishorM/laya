//! Integration tests for [`laya::ModelArtifacts`], covering the resolution order, error
//! reporting, and cache/download behaviour implemented in `src/model_artifacts.rs`.
//!
//! Every test that reads or writes `LAYA_ONNX_DIR` / `LAYA_ONNX_ROOT` / `HF_TOKEN` holds
//! [`ENV_LOCK`] for its whole body: `cargo test` runs tests on multiple threads by default, and
//! these are process-wide environment variables, so two such tests running concurrently would
//! stomp on each other's state. Tests that never touch an env var don't need the lock.
//!
//! Network-touching behaviour (404 / 401 / 403 handling, the `.part`-then-rename write) is covered
//! by unit tests inside `src/model_artifacts.rs` against a local `TcpListener`, not here — those
//! tests need access to the private `fetch_to_file` seam, which isn't visible from an integration
//! test crate.
//!
//! Tests build `LayaOptions` with [`plain_options`] plus direct field assignment rather than
//! `LayaOptions { ..., ..Default::default() }`: `LayaOptions` has a private field
//! (`hugging_face_subfolder`, deliberately not `pub` because its default is computed from
//! `checkpoint`), and Rust's `..base` struct-update syntax requires every field of the struct type
//! to be visible at the call site — even ones the update doesn't mention — so it does not compile
//! from outside the defining crate. Plain field assignment on an already-constructed value has no
//! such restriction, since each assignment only needs that one field to be `pub`.

use laya::{LayaCheckpoint, LayaError, LayaOptions, ModelArtifacts, ModelLayout, ModelPaths};
use std::path::Path;
use std::sync::Mutex;

/// Serializes every test in this file that reads or writes `LAYA_ONNX_DIR`, `LAYA_ONNX_ROOT`, or
/// `HF_TOKEN`.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Restores (or removes) one environment variable when dropped, so a test can freely mutate
/// process-wide state without leaking it into whichever test happens to run next.
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: callers hold `ENV_LOCK` for the guard's whole lifetime, so no other thread in
        // this test binary observes or mutates the environment concurrently.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Clears `LAYA_ONNX_DIR` and `LAYA_ONNX_ROOT` for the duration of the returned guards, so a test
/// starts from a clean slate regardless of what the outer environment happens to have set (a
/// developer running the suite locally with either variable exported would otherwise see
/// unrelated failures).
fn clear_onnx_env_vars() -> (EnvVarGuard, EnvVarGuard) {
    (
        EnvVarGuard::unset("LAYA_ONNX_DIR"),
        EnvVarGuard::unset("LAYA_ONNX_ROOT"),
    )
}

/// A fresh, all-defaults `LayaOptions`. See the module docs for why this — plus direct field
/// assignment — is used instead of `LayaOptions { .., ..Default::default() }`.
fn plain_options() -> LayaOptions {
    LayaOptions::new()
}

fn write_complete_artifact(dir: &Path) {
    std::fs::create_dir_all(dir.join("tokenizer")).expect("create tokenizer dir");
    std::fs::write(dir.join("model.onnx"), b"graph").expect("write model.onnx");
    std::fs::write(dir.join("model.onnx.data"), b"weights").expect("write model.onnx.data");
    std::fs::write(dir.join("rl_agent_config.json"), b"{}").expect("write config");
    std::fs::write(dir.join("tokenizer").join("tokenizer.json"), b"{}")
        .expect("write tokenizer.json");
}

// ── is_complete ───────────────────────────────────────────────────────────────

#[test]
fn is_complete_true_for_a_fully_populated_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    assert!(ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_false_for_a_directory_that_does_not_exist() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(!ModelArtifacts::is_complete(dir.path().join("nope")));
}

#[test]
fn is_complete_false_when_model_onnx_is_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::remove_file(dir.path().join("model.onnx")).unwrap();
    assert!(!ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_false_when_model_onnx_data_is_missing() {
    // The most common real-world case: the 1.29 GB sidecar was left behind when copying.
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::remove_file(dir.path().join("model.onnx.data")).unwrap();
    assert!(!ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_false_when_config_is_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::remove_file(dir.path().join("rl_agent_config.json")).unwrap();
    assert!(!ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_false_when_tokenizer_json_is_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::remove_file(dir.path().join("tokenizer").join("tokenizer.json")).unwrap();
    assert!(!ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_ignores_a_leftover_part_file() {
    // An interrupted download leaves `model.onnx.data.part`, never `model.onnx.data` itself.
    // `is_complete` must not be fooled by the `.part` file's presence.
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::remove_file(dir.path().join("model.onnx.data")).unwrap();
    std::fs::write(dir.path().join("model.onnx.data.part"), b"partial").unwrap();
    assert!(!ModelArtifacts::is_complete(dir.path()));
}

// ── layout detection ──────────────────────────────────────────────────────────

fn write_split_pair(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create dir");
    std::fs::write(dir.join("encoder.onnx"), b"encoder-graph").expect("write encoder.onnx");
    std::fs::write(dir.join("head.onnx"), b"head-graph").expect("write head.onnx");
    std::fs::write(dir.join("rl_agent_config.json"), b"{}").expect("write config");
    std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer.json");
}

#[test]
fn is_complete_true_for_a_fused_only_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    assert!(ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_true_for_a_split_only_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_split_pair(dir.path());
    assert!(ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_true_when_both_layouts_present_fused_wins() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    // A stray split pair alongside a complete fused layout must not affect completeness.
    std::fs::write(dir.path().join("encoder.onnx"), b"encoder-graph").unwrap();
    std::fs::write(dir.path().join("head.onnx"), b"head-graph").unwrap();
    assert!(ModelArtifacts::is_complete(dir.path()));

    let artifacts = ModelArtifacts::from_directory(dir.path(), None).expect("must resolve");
    assert_eq!(artifacts.layout(), ModelLayout::Fused);
}

#[test]
fn is_complete_false_for_neither_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("rl_agent_config.json"), b"{}").unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    assert!(!ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn is_complete_false_for_half_a_split_pair() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_split_pair(dir.path());
    std::fs::remove_file(dir.path().join("head.onnx")).unwrap();
    assert!(!ModelArtifacts::is_complete(dir.path()));
}

#[test]
fn from_directory_resolves_the_split_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_split_pair(dir.path());

    let artifacts = ModelArtifacts::from_directory(dir.path(), None).expect("must resolve");
    assert_eq!(artifacts.layout(), ModelLayout::Split);
    match &artifacts.model {
        ModelPaths::Split {
            encoder_path,
            head_path,
        } => {
            assert_eq!(encoder_path, &dir.path().join("encoder.onnx"));
            assert_eq!(head_path, &dir.path().join("head.onnx"));
        }
        other => panic!("expected ModelPaths::Split, got {other:?}"),
    }
    // The split layout never ships tokenizer_config.json.
    assert_eq!(artifacts.tokenizer_config_path, None);
}

#[test]
fn from_directory_prefers_fused_when_both_layouts_are_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::write(dir.path().join("encoder.onnx"), b"encoder-graph").unwrap();
    std::fs::write(dir.path().join("head.onnx"), b"head-graph").unwrap();

    let artifacts = ModelArtifacts::from_directory(dir.path(), None).expect("must resolve");
    assert_eq!(artifacts.layout(), ModelLayout::Fused);
}

#[test]
fn from_directory_reports_neither_layout_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("rl_agent_config.json"), b"{}").unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();

    let err = ModelArtifacts::from_directory(dir.path(), None).expect_err("must fail");
    match err {
        LayaError::MissingFile { path, hint } => {
            assert_eq!(path, dir.path().join("model.onnx"));
            let hint = hint.expect("should explain both layouts");
            assert!(hint.contains("encoder.onnx") && hint.contains("head.onnx"));
        }
        other => panic!("expected MissingFile, got {other:?}"),
    }
}

#[test]
fn from_directory_reports_a_half_written_split_export_by_the_missing_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_split_pair(dir.path());
    std::fs::remove_file(dir.path().join("head.onnx")).unwrap();

    let err = ModelArtifacts::from_directory(dir.path(), None).expect_err("must fail");
    match err {
        LayaError::MissingFile { path, .. } => {
            assert_eq!(path, dir.path().join("head.onnx"));
        }
        other => panic!("expected MissingFile naming head.onnx, got {other:?}"),
    }
}

// ── tokenizer lookup order ────────────────────────────────────────────────────

#[test]
fn from_directory_finds_tokenizer_in_the_nested_tokenizer_directory_first() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    // A root-level tokenizer.json should be ignored in favor of the nested one.
    std::fs::write(dir.path().join("tokenizer.json"), b"root-copy").unwrap();

    let artifacts = ModelArtifacts::from_directory(dir.path(), None).expect("must resolve");
    assert_eq!(
        artifacts.tokenizer_path,
        dir.path().join("tokenizer").join("tokenizer.json")
    );
}

#[test]
fn from_directory_falls_back_to_the_root_tokenizer_when_no_nested_one_exists() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_split_pair(dir.path());

    let artifacts = ModelArtifacts::from_directory(dir.path(), None).expect("must resolve");
    assert_eq!(artifacts.tokenizer_path, dir.path().join("tokenizer.json"));
}

// ── from_directory ────────────────────────────────────────────────────────────

#[test]
fn from_directory_succeeds_and_resolves_paths_for_a_complete_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());

    let artifacts = ModelArtifacts::from_directory(dir.path(), Some(LayaCheckpoint::Multilingual))
        .expect("resolve should succeed");

    assert_eq!(artifacts.layout(), ModelLayout::Fused);
    match &artifacts.model {
        ModelPaths::Fused { model_path } => {
            assert_eq!(model_path, &dir.path().join("model.onnx"));
        }
        other => panic!("expected ModelPaths::Fused, got {other:?}"),
    }
    assert_eq!(
        artifacts.config_path,
        dir.path().join("rl_agent_config.json")
    );
    assert_eq!(
        artifacts.tokenizer_path,
        dir.path().join("tokenizer").join("tokenizer.json")
    );
    assert_eq!(artifacts.checkpoint, Some(LayaCheckpoint::Multilingual));
}

#[test]
fn from_directory_reports_missing_file_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::remove_file(dir.path().join("rl_agent_config.json")).unwrap();

    let err = ModelArtifacts::from_directory(dir.path(), None).expect_err("must fail");
    match err {
        LayaError::MissingFile { path, .. } => {
            assert_eq!(path, dir.path().join("rl_agent_config.json"));
        }
        other => panic!("expected MissingFile, got {other:?}"),
    }
}

#[test]
fn from_directory_reports_directory_not_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("does-not-exist");

    let err = ModelArtifacts::from_directory(&missing, None).expect_err("must fail");
    assert!(matches!(err, LayaError::DirectoryNotFound { .. }));
}

// ── resolve: precedence ───────────────────────────────────────────────────────

#[test]
fn resolve_prefers_model_directory_over_everything_else() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let option_dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(option_dir.path());

    let env_dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(env_dir.path());
    let _env_dir_set = EnvVarGuard::set("LAYA_ONNX_DIR", env_dir.path().to_str().unwrap());

    let cache_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&cache_root.path().join("multilingual"));

    let mut options = plain_options();
    options.model_directory = Some(option_dir.path().to_path_buf());
    options.cache_directory = Some(cache_root.path().to_path_buf());

    let artifacts = ModelArtifacts::resolve(&options).expect("resolve should succeed");
    assert_eq!(artifacts.directory, option_dir.path());
    // Explicit directories carry no inferred checkpoint.
    assert_eq!(artifacts.checkpoint, None);
}

#[test]
fn resolve_prefers_env_dir_over_onnx_root_and_cache() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let env_dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(env_dir.path());
    let _env_dir_set = EnvVarGuard::set("LAYA_ONNX_DIR", env_dir.path().to_str().unwrap());

    let onnx_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&onnx_root.path().join("multilingual"));
    let _root_set = EnvVarGuard::set("LAYA_ONNX_ROOT", onnx_root.path().to_str().unwrap());

    let cache_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&cache_root.path().join("multilingual"));

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());

    let artifacts = ModelArtifacts::resolve(&options).expect("resolve should succeed");
    assert_eq!(artifacts.directory, env_dir.path());
    assert_eq!(artifacts.checkpoint, None);
}

#[test]
fn resolve_prefers_onnx_root_over_cache() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let onnx_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&onnx_root.path().join("multilingual"));
    let _root_set = EnvVarGuard::set("LAYA_ONNX_ROOT", onnx_root.path().to_str().unwrap());

    let cache_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&cache_root.path().join("multilingual"));

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());

    let artifacts = ModelArtifacts::resolve(&options).expect("resolve should succeed");
    assert_eq!(artifacts.directory, onnx_root.path().join("multilingual"));
    assert_eq!(artifacts.checkpoint, Some(LayaCheckpoint::Multilingual));
}

#[test]
fn resolve_falls_through_a_stale_onnx_root_to_the_cache() {
    // LAYA_ONNX_ROOT is set but this checkpoint's subdirectory under it is incomplete: resolution
    // must fall through to the cache rather than failing immediately.
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let onnx_root = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(onnx_root.path().join("multilingual")).unwrap();
    // Incomplete: no model.onnx.data.
    std::fs::write(
        onnx_root.path().join("multilingual").join("model.onnx"),
        b"x",
    )
    .unwrap();
    let _root_set = EnvVarGuard::set("LAYA_ONNX_ROOT", onnx_root.path().to_str().unwrap());

    let cache_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&cache_root.path().join("multilingual"));

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());

    let artifacts = ModelArtifacts::resolve(&options).expect("resolve should succeed");
    assert_eq!(artifacts.directory, cache_root.path().join("multilingual"));
}

#[test]
fn resolve_uses_the_cache_directory_when_nothing_else_is_set() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let cache_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&cache_root.path().join("english"));

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());
    options.checkpoint = LayaCheckpoint::English;

    let artifacts = ModelArtifacts::resolve(&options).expect("resolve should succeed");
    assert_eq!(artifacts.directory, cache_root.path().join("english"));
    assert_eq!(artifacts.checkpoint, Some(LayaCheckpoint::English));
}

#[test]
fn resolve_respects_checkpoint_subdirectory_naming() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let cache_root = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(&cache_root.path().join("typed-decisions"));

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());
    options.checkpoint = LayaCheckpoint::TypedDecisions;

    let artifacts = ModelArtifacts::resolve(&options).expect("resolve should succeed");
    assert_eq!(
        artifacts.directory,
        cache_root.path().join("typed-decisions")
    );
}

// ── resolve: not found ────────────────────────────────────────────────────────

#[test]
fn resolve_reports_every_location_tried_when_nothing_is_found() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let cache_root = tempfile::tempdir().expect("tempdir");
    // Deliberately left empty: no complete artifact anywhere.

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());
    options.allow_download = false;

    let err = ModelArtifacts::resolve(&options).expect_err("must fail: nothing is present");
    let message = err.to_string();

    assert!(matches!(err, LayaError::ArtifactsNotFound { .. }));
    assert!(message.contains("not found"), "message was: {message}");
    let expected_cache_dir = cache_root.path().join("multilingual");
    assert!(
        message.contains(expected_cache_dir.to_str().unwrap()),
        "message should name the cache directory it tried: {message}"
    );
    assert!(
        message.contains("LayaOptions::model_directory"),
        "message should list the ModelDirectory step: {message}"
    );
    assert!(
        message.contains("LAYA_ONNX_DIR"),
        "message should list the LAYA_ONNX_DIR step: {message}"
    );
    assert!(
        message.contains("LAYA_ONNX_ROOT"),
        "message should list the LAYA_ONNX_ROOT step: {message}"
    );
}

#[test]
fn resolve_not_found_message_names_the_onnx_root_path_when_the_env_var_is_set() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let onnx_root = tempfile::tempdir().expect("tempdir");
    // Root is set but has nothing under it for this checkpoint.
    let _root_set = EnvVarGuard::set("LAYA_ONNX_ROOT", onnx_root.path().to_str().unwrap());

    let cache_root = tempfile::tempdir().expect("tempdir");

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());

    let err = ModelArtifacts::resolve(&options).expect_err("must fail");
    let message = err.to_string();
    let expected_root_dir = onnx_root.path().join("multilingual");
    assert!(
        message.contains(expected_root_dir.to_str().unwrap()),
        "message should name the LAYA_ONNX_ROOT path it tried: {message}"
    );
}

#[test]
fn resolve_with_allow_download_false_never_touches_the_network() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let cache_root = tempfile::tempdir().expect("tempdir");

    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());
    options.allow_download = false;
    // A repo name that does not exist; if resolve() attempted a network call despite
    // allow_download being false, this would surface as a very different (slow, DNS-error)
    // failure rather than the immediate ArtifactsNotFound below.
    options.hugging_face_repo = "this-repo-does-not-exist/laya-nope".to_string();

    let started = std::time::Instant::now();
    let err = ModelArtifacts::resolve(&options).expect_err("must fail without downloading");
    assert!(matches!(err, LayaError::ArtifactsNotFound { .. }));
    // Resolution is pure filesystem/env work when allow_download is false; it should complete
    // near-instantly. A generous bound avoids flakiness on a loaded CI box while still catching
    // an accidental network attempt (which would take seconds, not milliseconds).
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "resolve() with allow_download=false took {:?}; it may have attempted a network call",
        started.elapsed()
    );
}

// ── resolve: explicit-path validation errors propagate ────────────────────────

#[test]
fn resolve_propagates_missing_file_error_from_an_explicit_model_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_complete_artifact(dir.path());
    std::fs::remove_file(dir.path().join("model.onnx")).unwrap();

    let mut options = plain_options();
    options.model_directory = Some(dir.path().to_path_buf());

    let err = ModelArtifacts::resolve(&options).expect_err("must fail");
    assert!(matches!(err, LayaError::MissingFile { .. }));
}

#[test]
fn resolve_propagates_directory_not_found_from_laya_onnx_dir() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let missing = tempfile::tempdir()
        .expect("tempdir")
        .path()
        .join("does-not-exist");
    let _env_dir_set = EnvVarGuard::set("LAYA_ONNX_DIR", missing.to_str().unwrap());

    let options = plain_options();
    let err = ModelArtifacts::resolve(&options).expect_err("must fail");
    assert!(matches!(err, LayaError::DirectoryNotFound { .. }));
}

// ── download feature gate ──────────────────────────────────────────────────────

#[test]
#[cfg(not(feature = "download"))]
fn resolve_with_allow_download_true_and_no_download_feature_gives_a_clear_error() {
    let _env = lock_env();
    let (_dir_guard, _root_guard) = clear_onnx_env_vars();

    let cache_root = tempfile::tempdir().expect("tempdir");
    let mut options = plain_options();
    options.cache_directory = Some(cache_root.path().to_path_buf());
    options.allow_download = true;

    let err = ModelArtifacts::resolve(&options).expect_err("must fail: no download feature");
    let message = err.to_string();
    assert!(
        message.contains("download") && message.contains("feature"),
        "message should explain the missing `download` feature: {message}"
    );
}
