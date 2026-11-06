//! Revision pinning and SHA-256 digest checks, model-free. Ports the behaviour of
//! `laya/revisions.py` (`resolve_revision`, `verify_digests`) and the router's
//! `_digests_from_env` / `_merge_expected_digests` / `_revision_pin`, with Python's error texts.
//!
//! Every test here holds [`ENV_LOCK`] and restores the variables it touches (`LAYA_REVISION`,
//! `LAYA_SHA256_DIGESTS`), because they are process-wide.

use indexmap::IndexMap;
use laya::error::{LayaError, Result};
use laya::laya_options::{LayaCheckpoint, LayaOptions};
use laya::laya_router::{
    LayaPredictor, LayaRouter, LayaRouterOptions, digests_from_env, merge_expected_digests,
    revision_pin,
};
use laya::revisions::{
    DigestGraph, PINNED_REVISIONS, REVIEWED, encode_revision_segment, resolve_revision,
    reviewed_pin, verify_digests,
};
use laya::{LayaEngine, ModelArtifacts};
use serde_json::Value;
use std::path::Path;
use std::sync::{Arc, Mutex};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: every test holds `ENV_LOCK` for the guard's whole lifetime.
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

/// Takes the lock and clears every variable these tests read, so a developer's own exports never
/// reach them. The guards restore the originals on drop.
fn clean_env() -> (std::sync::MutexGuard<'static, ()>, EnvVarGuard, EnvVarGuard) {
    let lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    (
        lock,
        EnvVarGuard::unset("LAYA_REVISION"),
        EnvVarGuard::unset("LAYA_SHA256_DIGESTS"),
    )
}

fn map(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn message(result: Result<impl std::fmt::Debug>) -> String {
    result.expect_err("expected an error").to_string()
}

// SHA-256 of the bytes written by `fixture_dir`.
const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

fn fixture_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("abc.txt"), b"abc").unwrap();
    std::fs::create_dir_all(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub").join("abc.txt"), b"abc").unwrap();
    std::fs::write(dir.path().join("model.onnx"), b"abc").unwrap();
    dir
}

// ---------------------------------------------------------------------- the table

#[test]
fn pinned_table_mirrors_python() {
    assert_eq!(REVIEWED, "reviewed");
    assert_eq!(
        PINNED_REVISIONS,
        &[
            (
                "convaiinnovations/laya",
                "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
            ),
            (
                "convaiinnovations/laya-multilingual",
                "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
            ),
            (
                "convaiinnovations/laya-typed-decisions",
                "1a793eb568e6718f15941d08f85432581df534e3"
            ),
        ]
    );
}

#[test]
fn reviewed_pin_is_exact_then_case_insensitive() {
    assert_eq!(
        reviewed_pin("ConvaiInnovations/Laya"),
        Some("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
    );
    assert_eq!(reviewed_pin("someone/else"), None);
}

// ---------------------------------------------------------------------- resolve_revision

#[test]
fn resolve_revision_argument_beats_environment_and_both_are_stripped() {
    let (_lock, _rev, _dig) = clean_env();
    let _env = EnvVarGuard::set("LAYA_REVISION", " envsha ");
    assert_eq!(
        resolve_revision("x/y", Some("  argsha "))
            .unwrap()
            .as_deref(),
        Some("argsha")
    );
    assert_eq!(
        resolve_revision("x/y", None).unwrap().as_deref(),
        Some("envsha")
    );
    // An empty argument is "not asked for", so the variable applies.
    assert_eq!(
        resolve_revision("x/y", Some("")).unwrap().as_deref(),
        Some("envsha")
    );
}

#[test]
fn resolve_revision_blank_argument_suppresses_the_environment() {
    let (_lock, _rev, _dig) = clean_env();
    let _env = EnvVarGuard::set("LAYA_REVISION", "envsha");
    assert_eq!(resolve_revision("x/y", Some("   ")).unwrap(), None);
}

#[test]
fn resolve_revision_unset_or_blank_environment_is_no_pin() {
    let (_lock, _rev, _dig) = clean_env();
    assert_eq!(resolve_revision("x/y", None).unwrap(), None);
    let _env = EnvVarGuard::set("LAYA_REVISION", "   ");
    assert_eq!(resolve_revision("x/y", None).unwrap(), None);
}

#[test]
fn resolve_revision_reviewed_looks_up_the_pin_case_insensitively() {
    let (_lock, _rev, _dig) = clean_env();
    for repo in ["convaiinnovations/laya", "ConvaiInnovations/LAYA"] {
        assert_eq!(
            resolve_revision(repo, Some("reviewed")).unwrap().as_deref(),
            Some("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
        );
    }
    let _env = EnvVarGuard::set("LAYA_REVISION", " reviewed ");
    assert_eq!(
        resolve_revision("convaiinnovations/laya-multilingual", None)
            .unwrap()
            .as_deref(),
        Some("e4e9ddf21a7b1903b7acffd8814ad4307bf63a67")
    );
}

#[test]
fn resolve_revision_reviewed_without_a_pin_names_who_asked() {
    let (_lock, _rev, _dig) = clean_env();
    let from_argument = message(resolve_revision("a/b", Some("reviewed")));
    assert_eq!(
        from_argument,
        "laya: revision= asked for 'reviewed', but 'a/b' has no reviewed SHA in PINNED_REVISIONS; \
         pass a commit, branch or tag in revision= instead"
    );
    let _env = EnvVarGuard::set("LAYA_REVISION", "reviewed");
    let from_env = message(resolve_revision("a/b", None));
    assert_eq!(
        from_env,
        "laya: LAYA_REVISION=reviewed, but 'a/b' has no reviewed SHA in PINNED_REVISIONS; set \
         LAYA_REVISION to a commit or unset it"
    );
    assert!(matches!(
        resolve_revision("a/b", None),
        Err(LayaError::InvalidRevision(_))
    ));
}

#[test]
fn revision_segments_encode_a_branch_as_one_path_segment() {
    assert_eq!(encode_revision_segment("refs/pr/1"), "refs%2Fpr%2F1");
    assert_eq!(encode_revision_segment("v1.0"), "v1.0");
    assert_eq!(encode_revision_segment(".."), "%2E%2E");
    assert_eq!(encode_revision_segment("a b"), "a%20b");
}

// ---------------------------------------------------------------------- verify_digests

#[test]
fn verify_digests_accepts_a_matching_digest_ignoring_case_and_whitespace() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    let upper = format!("  {}  ", ABC_SHA.to_uppercase());
    let expected = map(&[("abc.txt", &upper), ("sub\\abc.txt", ABC_SHA)]);
    verify_digests(dir.path(), Some(&expected), None).unwrap();
}

#[test]
fn verify_digests_mismatch_text() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    let expected = map(&[("abc.txt", "00")]);
    let got = message(verify_digests(dir.path(), Some(&expected), None));
    assert_eq!(
        got,
        format!(
            "laya: SHA-256 mismatch for abc.txt: expected 00, got {ABC_SHA}. The artifact does \
             not match the reviewed digest; refusing to load it."
        )
    );
}

#[test]
fn verify_digests_missing_file_text() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    let expected = map(&[("nope.bin", ABC_SHA)]);
    let err = verify_digests(dir.path(), Some(&expected), None).unwrap_err();
    assert!(matches!(err, LayaError::DigestFileNotFound(_)));
    assert_eq!(
        err.to_string(),
        format!(
            "laya: cannot verify 'nope.bin': no such file under {}",
            dir.path().display()
        )
    );
}

#[test]
fn verify_digests_refuses_unsafe_paths_with_pythons_texts() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    for rel in ["/etc/passwd", "\\\\srv\\share\\f", "C:\\x", "C:/x"] {
        let expected = map(&[(rel, ABC_SHA)]);
        let got = message(verify_digests(dir.path(), Some(&expected), None));
        assert!(
            got.starts_with("laya: unsafe absolute path in expected digests: '"),
            "{rel}: {got}"
        );
    }
    for rel in ["..", "../x", "a/../b", "..\\x", ""] {
        let expected = map(&[(rel, ABC_SHA)]);
        let got = message(verify_digests(dir.path(), Some(&expected), None));
        assert!(
            got.starts_with("laya: unsafe path in expected digests: '"),
            "{rel:?}: {got}"
        );
    }
    // Only the first four spellings above are refused as absolute; `C:x` is a relative name.
    let expected = map(&[("C:x", ABC_SHA)]);
    assert!(matches!(
        verify_digests(dir.path(), Some(&expected), None),
        Err(LayaError::DigestFileNotFound(_))
    ));
}

#[test]
fn verify_digests_reads_the_environment_only_when_no_map_is_given() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    verify_digests(dir.path(), None, None).unwrap(); // unset: nothing to check
    let _env = EnvVarGuard::set(
        "LAYA_SHA256_DIGESTS",
        &format!(r#" {{"abc.txt": "{ABC_SHA}"}} "#),
    );
    verify_digests(dir.path(), None, None).unwrap();
    // An explicit empty map masks the variable.
    let _bad = EnvVarGuard::set("LAYA_SHA256_DIGESTS", r#"{"abc.txt": "00"}"#);
    assert!(verify_digests(dir.path(), None, None).is_err());
    verify_digests(dir.path(), Some(&IndexMap::new()), None).unwrap();
}

#[test]
fn verify_digests_malformed_environment_texts() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    let _env = EnvVarGuard::set("LAYA_SHA256_DIGESTS", "not json");
    assert_eq!(
        message(verify_digests(dir.path(), None, None)),
        "LAYA_SHA256_DIGESTS must be a JSON object of artifact->sha256"
    );
    let _list = EnvVarGuard::set("LAYA_SHA256_DIGESTS", "[1]");
    assert_eq!(
        message(verify_digests(dir.path(), None, None)),
        "expected_sha256 must be a mapping of artifact paths to digests"
    );
}

#[test]
fn verify_digests_stringifies_non_string_environment_values_as_python_does() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    let _env = EnvVarGuard::set("LAYA_SHA256_DIGESTS", r#"{"abc.txt": 12}"#);
    let got = message(verify_digests(dir.path(), None, None));
    assert!(got.contains("expected 12, got "), "{got}");
}

#[test]
fn verify_digests_hashes_a_file_larger_than_one_chunk() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = tempfile::tempdir().unwrap();
    // 2.5 MiB of zeros (more than two 1 MiB chunks); sha256 computed by Python's hashlib.
    std::fs::write(dir.path().join("big.bin"), vec![0u8; 5 * 512 * 1024]).unwrap();
    let expected = map(&[(
        "big.bin",
        "6de7493c5c90f643357c268fbaaf461c1567e0334e4948023ce17268403aa37a",
    )]);
    let result = verify_digests(dir.path(), Some(&expected), None);
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
}

#[test]
fn onnx_key_names_the_fused_graph_and_is_refused_on_a_split_checkpoint() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = fixture_dir();
    let graph = dir.path().join("model.onnx");
    for key in ["onnx", "onnx_path"] {
        let expected = map(&[(key, ABC_SHA)]);
        verify_digests(
            dir.path(),
            Some(&expected),
            Some(DigestGraph::Fused(&graph)),
        )
        .unwrap();
        let split = message(verify_digests(
            dir.path(),
            Some(&expected),
            Some(DigestGraph::Split),
        ));
        assert_eq!(
            split,
            "laya: the 'onnx' / 'onnx_path' digest key names the fused model.onnx, \
             but this checkpoint is the split layout (encoder.onnx + head.onnx); \
             list encoder.onnx and head.onnx by relative path instead"
        );
        // Without a graph the key is an ordinary relative path, as Python without onnx_path.
        assert!(matches!(
            verify_digests(dir.path(), Some(&expected), None),
            Err(LayaError::DigestFileNotFound(_))
        ));
    }
}

// ---------------------------------------------------------------------- engine ordering

fn write_split_dir(dir: &Path, config: &[u8]) {
    std::fs::write(dir.join("encoder.onnx"), b"enc").unwrap();
    std::fs::write(dir.join("head.onnx"), b"head").unwrap();
    std::fs::write(dir.join("rl_agent_config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
}

fn engine_options(dir: &Path, expected: Option<IndexMap<String, String>>) -> LayaOptions {
    let mut options = LayaOptions::new();
    options.model_directory = Some(dir.to_path_buf());
    options.expected_sha256 = expected;
    options
}

#[test]
fn engine_verifies_digests_before_it_reads_the_config() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = tempfile::tempdir().unwrap();
    // The config is not JSON: a load that parsed it first would fail with a JSON error instead.
    write_split_dir(dir.path(), b"this is not json");
    let bad = map(&[("encoder.onnx", "00")]);
    let err = LayaEngine::create(engine_options(dir.path(), Some(bad)))
        .err()
        .expect("must fail");
    assert!(matches!(err, LayaError::InvalidDigests(_)), "{err}");
    assert!(
        err.to_string()
            .starts_with("laya: SHA-256 mismatch for encoder.onnx")
    );

    // With a matching digest the load goes on and only then trips over the config.
    let good = map(&[("head.onnx", &sha_of(b"head"))]);
    let err = LayaEngine::create(engine_options(dir.path(), Some(good)))
        .err()
        .expect("must fail");
    assert!(!matches!(err, LayaError::InvalidDigests(_)), "{err}");
}

#[test]
fn digests_are_checked_before_the_directory_is_checked_for_completeness() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = tempfile::tempdir().unwrap();
    // Incomplete on purpose: no head.onnx, no rl_agent_config.json, no tokenizer.
    std::fs::write(dir.path().join("encoder.onnx"), b"enc").unwrap();
    let bad = map(&[("encoder.onnx", "00")]);
    let err = LayaEngine::create(engine_options(dir.path(), Some(bad.clone())))
        .err()
        .expect("must fail");
    assert!(matches!(err, LayaError::InvalidDigests(_)), "{err}");
    let err =
        ModelArtifacts::resolve(&engine_options(dir.path(), Some(bad))).expect_err("must fail");
    assert!(matches!(err, LayaError::InvalidDigests(_)), "{err}");

    // Once the digest matches, the incomplete directory is reported as before.
    let good = map(&[("encoder.onnx", &sha_of(b"enc"))]);
    let err =
        ModelArtifacts::resolve(&engine_options(dir.path(), Some(good))).expect_err("must fail");
    assert!(matches!(err, LayaError::MissingFile { .. }), "{err}");

    // A directory that does not exist is still reported as missing, not as an unverifiable file.
    let missing = dir.path().join("nope");
    let err = ModelArtifacts::resolve(&engine_options(&missing, Some(map(&[("a", "00")]))))
        .expect_err("must fail");
    assert!(matches!(err, LayaError::DirectoryNotFound { .. }), "{err}");
}

#[test]
fn engine_refuses_the_onnx_key_on_a_split_checkpoint() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = tempfile::tempdir().unwrap();
    write_split_dir(dir.path(), b"{}");
    let err = LayaEngine::create(engine_options(dir.path(), Some(map(&[("onnx", "00")]))))
        .err()
        .expect("must fail");
    assert!(err.to_string().contains("split layout"), "{err}");
}

#[test]
fn engine_applies_the_environment_map_to_a_local_directory() {
    let (_lock, _rev, _dig) = clean_env();
    let dir = tempfile::tempdir().unwrap();
    write_split_dir(dir.path(), b"{}");
    let _env = EnvVarGuard::set("LAYA_SHA256_DIGESTS", r#"{"head.onnx": "00"}"#);
    let err = LayaEngine::create(engine_options(dir.path(), None))
        .err()
        .expect("must fail");
    assert!(matches!(err, LayaError::InvalidDigests(_)), "{err}");
    // `Some(empty)` masks the variable, so the load gets past verification.
    let err = LayaEngine::create(engine_options(dir.path(), Some(IndexMap::new())))
        .err()
        .expect("must fail later");
    assert!(!matches!(err, LayaError::InvalidDigests(_)), "{err}");
}

fn sha_of(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn model_artifacts_reports_no_revision_for_a_local_directory() {
    let dir = tempfile::tempdir().unwrap();
    write_split_dir(dir.path(), b"{}");
    assert_eq!(
        ModelArtifacts::from_directory(dir.path(), None)
            .unwrap()
            .revision,
        None
    );
}

// ---------------------------------------------------------------------- router helpers

#[test]
fn revision_pin_treats_blank_as_unset() {
    assert_eq!(revision_pin(None), None);
    assert_eq!(revision_pin(Some("")), None);
    assert_eq!(revision_pin(Some("  \t")), None);
    assert_eq!(revision_pin(Some(" abc ")), Some(" abc "));
}

#[test]
fn merge_is_file_by_file_and_the_checkpoint_wins() {
    let shared = map(&[("a", "1"), ("b", "2")]);
    let own = map(&[("b", "9"), ("c", "3")]);
    let merged = merge_expected_digests(Some(&shared), Some(&own)).unwrap();
    assert_eq!(merged, map(&[("a", "1"), ("b", "9"), ("c", "3")]));
    assert_eq!(
        merged.keys().collect::<Vec<_>>(),
        ["a", "b", "c"],
        "shared order first, new keys appended, as a Python dict update"
    );
    assert_eq!(merge_expected_digests(None, None), None);
    assert_eq!(merge_expected_digests(Some(&shared), None).unwrap(), shared);
    assert_eq!(
        merge_expected_digests(None, Some(&IndexMap::new())),
        Some(IndexMap::new()),
        "an empty entry is not silence: it masks the environment"
    );
}

const ALL: [LayaCheckpoint; 3] = [
    LayaCheckpoint::English,
    LayaCheckpoint::Multilingual,
    LayaCheckpoint::TypedDecisions,
];

#[test]
fn digests_from_env_flat_unset_and_unparseable_name_nothing() {
    let (_lock, _rev, _dig) = clean_env();
    assert!(digests_from_env(&ALL).unwrap().is_empty());
    for raw in ["", "  ", "not json", "[]", "{}", r#"{"a": "b"}"#, "7"] {
        let _env = EnvVarGuard::set("LAYA_SHA256_DIGESTS", raw);
        assert!(digests_from_env(&ALL).unwrap().is_empty(), "{raw:?}");
    }
}

#[test]
fn digests_from_env_nested_names_models_and_defaults_the_rest_to_empty() {
    let (_lock, _rev, _dig) = clean_env();
    let _env = EnvVarGuard::set(
        "LAYA_SHA256_DIGESTS",
        r#"{"en": {"model.onnx": "AA"}, "Multilingual": {"x": 5}}"#,
    );
    let got = digests_from_env(&ALL).unwrap();
    assert_eq!(got[&LayaCheckpoint::English], map(&[("model.onnx", "AA")]));
    assert_eq!(got[&LayaCheckpoint::Multilingual], map(&[("x", "5")]));
    assert!(got[&LayaCheckpoint::TypedDecisions].is_empty());
}

#[test]
fn digests_from_env_mixed_shape_and_unknown_model_are_refused() {
    let (_lock, _rev, _dig) = clean_env();
    let _env = EnvVarGuard::set(
        "LAYA_SHA256_DIGESTS",
        r#"{"zeta": "aa", "alpha": {"f": "b"}}"#,
    );
    let err = digests_from_env(&ALL).unwrap_err();
    assert!(matches!(err, LayaError::InvalidDigests(_)));
    assert_eq!(
        err.to_string(),
        "LAYA_SHA256_DIGESTS must be either {artifact: digest} for every checkpoint or {model: \
         {artifact: digest}} per checkpoint; ['alpha', 'zeta'] mixes the two or holds a value \
         that is neither"
    );
    let _unknown = EnvVarGuard::set("LAYA_SHA256_DIGESTS", r#"{"nope": {"f": "b"}}"#);
    assert!(
        digests_from_env(&ALL)
            .unwrap_err()
            .to_string()
            .starts_with("unknown model 'nope'")
    );
}

// ---------------------------------------------------------------------- router wiring

struct Fake(Option<&'static str>);

impl LayaPredictor for Fake {
    fn predict(
        &self,
        _state: Value,
        _questions: &laya::questions::QuestionSet,
    ) -> Result<laya::answers::LayaResult> {
        unreachable!("not called")
    }

    fn revision(&self) -> Option<String> {
        self.0.map(str::to_string)
    }
}

type Seen = Arc<
    Mutex<
        Vec<(
            LayaCheckpoint,
            Option<String>,
            Option<IndexMap<String, String>>,
        )>,
    >,
>;

fn recording_router(options: LayaRouterOptions) -> Result<(LayaRouter, Seen)> {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let router = LayaRouter::with_factory(options, move |checkpoint, opts| {
        sink.lock().unwrap().push((
            checkpoint,
            opts.revision.clone(),
            opts.expected_sha256.clone(),
        ));
        Ok(Arc::new(Fake(Some("built"))) as Arc<dyn LayaPredictor>)
    })?;
    Ok((router, seen))
}

fn seen_for(
    seen: &Seen,
    key: LayaCheckpoint,
) -> (Option<String>, Option<IndexMap<String, String>>) {
    let log = seen.lock().unwrap();
    let (_, rev, exp) = log.iter().find(|(k, _, _)| *k == key).expect("built");
    (rev.clone(), exp.clone())
}

#[test]
fn router_revision_precedence_is_per_checkpoint_then_shared_then_none() {
    let (_lock, _rev, _dig) = clean_env();
    let mut options = LayaRouterOptions::default();
    options.engine_options.revision = Some("shared".to_string());
    options
        .revisions
        .insert("en".to_string(), Some("own".to_string()));
    options
        .revisions
        .insert("multilingual".to_string(), Some("  ".to_string()));
    options.revisions.insert("typed".to_string(), None);
    let (router, seen) = recording_router(options).unwrap();
    for name in ["english", "multilingual", "typed-decisions"] {
        router.load(name).unwrap();
    }
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::English).0.as_deref(),
        Some("own")
    );
    // Blank and None are "no pin of its own": they inherit the shared one.
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::Multilingual).0.as_deref(),
        Some("shared")
    );
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::TypedDecisions).0.as_deref(),
        Some("shared")
    );
}

#[test]
fn router_without_any_pin_passes_none_so_the_loader_reads_the_environment() {
    let (_lock, _rev, _dig) = clean_env();
    let mut options = LayaRouterOptions::default();
    options.engine_options.revision = Some("   ".to_string());
    let (router, seen) = recording_router(options).unwrap();
    router.load("english").unwrap();
    assert_eq!(seen_for(&seen, LayaCheckpoint::English), (None, None));
}

#[test]
fn router_merges_shared_and_per_checkpoint_digests_with_the_checkpoint_winning() {
    let (_lock, _rev, _dig) = clean_env();
    let mut options = LayaRouterOptions::default();
    options.engine_options.expected_sha256 =
        Some(map(&[("tokenizer.json", "t"), ("model.onnx", "s")]));
    options
        .sha256_digests
        .insert("english".to_string(), Some(map(&[("model.onnx", "e")])));
    options
        .sha256_digests
        .insert("multilingual".to_string(), None);
    let (router, seen) = recording_router(options).unwrap();
    for name in ["english", "multilingual", "typed-decisions"] {
        router.load(name).unwrap();
    }
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::English).1.unwrap(),
        map(&[("tokenizer.json", "t"), ("model.onnx", "e")])
    );
    // A `None` entry names no files: the shared map alone applies.
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::Multilingual).1.unwrap(),
        map(&[("tokenizer.json", "t"), ("model.onnx", "s")])
    );
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::TypedDecisions).1.unwrap(),
        map(&[("tokenizer.json", "t"), ("model.onnx", "s")])
    );
}

#[test]
fn router_none_entry_masks_a_flat_environment_map_and_no_entry_leaves_it() {
    let (_lock, _rev, _dig) = clean_env();
    let _env = EnvVarGuard::set("LAYA_SHA256_DIGESTS", r#"{"f": "aa"}"#);
    let mut options = LayaRouterOptions::default();
    options.sha256_digests.insert("english".to_string(), None);
    let (router, seen) = recording_router(options).unwrap();
    router.load("english").unwrap();
    router.load("multilingual").unwrap();
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::English).1,
        Some(IndexMap::new())
    );
    assert_eq!(seen_for(&seen, LayaCheckpoint::Multilingual).1, None);
}

#[test]
fn router_seeds_a_nested_environment_map_and_the_argument_wins_per_checkpoint() {
    let (_lock, _rev, _dig) = clean_env();
    let _env = EnvVarGuard::set(
        "LAYA_SHA256_DIGESTS",
        r#"{"english": {"a": "1"}, "multilingual": {"b": "2"}}"#,
    );
    let mut options = LayaRouterOptions::default();
    options.engine_options.expected_sha256 = Some(map(&[("a", "shared")]));
    options
        .sha256_digests
        .insert("multilingual".to_string(), Some(map(&[("b", "arg")])));
    let (router, seen) = recording_router(options).unwrap();
    for name in ["english", "multilingual", "typed-decisions"] {
        router.load(name).unwrap();
    }
    // The environment's per-checkpoint entry beats the shared map, file by file.
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::English).1.unwrap(),
        map(&[("a", "1")])
    );
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::Multilingual).1.unwrap(),
        map(&[("a", "shared"), ("b", "arg")])
    );
    // A checkpoint the nested variable does not name gets an empty entry: shared only.
    assert_eq!(
        seen_for(&seen, LayaCheckpoint::TypedDecisions).1.unwrap(),
        map(&[("a", "shared")])
    );
}

#[test]
fn router_construction_refuses_a_mixed_shape_environment_and_unknown_names() {
    let (_lock, _rev, _dig) = clean_env();
    {
        let _env = EnvVarGuard::set("LAYA_SHA256_DIGESTS", r#"{"a": "x", "b": {"c": "d"}}"#);
        let err = recording_router(LayaRouterOptions::default())
            .err()
            .unwrap();
        assert!(matches!(err, LayaError::InvalidDigests(_)));
    }
    let mut options = LayaRouterOptions::default();
    options.revisions.insert("nope".to_string(), None);
    assert!(recording_router(options).is_err());
    let mut options = LayaRouterOptions::default();
    options.sha256_digests.insert("nope".to_string(), None);
    assert!(recording_router(options).is_err());
}

#[test]
fn loaded_revisions_reports_what_each_resident_predictor_says() {
    let (_lock, _rev, _dig) = clean_env();
    let (router, _seen) = recording_router(LayaRouterOptions::default()).unwrap();
    router.load("english").unwrap();
    router
        .attach(
            "multilingual",
            Arc::new(Fake(None)) as Arc<dyn LayaPredictor>,
        )
        .unwrap();
    assert_eq!(
        router.loaded_revisions(),
        vec![
            (LayaCheckpoint::English, Some("built".to_string())),
            (LayaCheckpoint::Multilingual, None),
        ]
    );
}
