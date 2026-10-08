//! Loads a [`LayaEngine`] for one checkpoint at a time.
//!
//! Model loads are ~1.3-1.7 GB each, so the Tier 2 test files deliberately do **not** cache all
//! three engines for the lifetime of the test binary. Instead, every test here loads the one
//! engine it needs into a local variable and lets it drop at the end of that loop iteration (or
//! that test function) before moving on to the next checkpoint, so at most one ~1.3-1.7 GB engine is ever
//! resident at a time *within a single test*. Cargo still runs different `#[test]` functions in
//! the same binary concurrently by default, so two tests can each be holding one engine at once;
//! run the heavy suites with `cargo test --release -- --test-threads=1` (noted again on each Tier
//! 2 test file) to guarantee only one engine total is ever loaded.

use laya::LayaEngine;
use laya::laya_options::LayaCheckpoint;

use super::test_artifacts;

/// Every checkpoint, in the order golden data and shape-sweep tables list them.
pub const CHECKPOINTS: [LayaCheckpoint; 3] = [
    LayaCheckpoint::Multilingual,
    LayaCheckpoint::English,
    LayaCheckpoint::TypedDecisions,
];

/// Loads the engine for `checkpoint`, or prints a skip line and returns `None` when its ONNX
/// artifacts are not available locally. Panics (rather than returning `None`) if the artifacts
/// are present but fail to load — that is a real bug, not a missing-fixture skip.
pub fn load(checkpoint: LayaCheckpoint) -> Option<LayaEngine> {
    if !(test_artifacts::has_model(checkpoint) && test_artifacts::has_tokenizer(checkpoint)) {
        eprintln!(
            "skip: no ONNX artifacts for '{}'; set LAYA_ONNX_ROOT to the parent of all \
             checkpoints, or LAYA_ONNX_DIR for multilingual",
            checkpoint.subdir()
        );
        return None;
    }
    let dir = test_artifacts::locate(checkpoint).unwrap_or_else(|| {
        panic!(
            "has_model/has_tokenizer were true but locate() found nothing for '{}'",
            checkpoint.subdir()
        )
    });
    Some(LayaEngine::from_directory(&dir).unwrap_or_else(|e| {
        panic!(
            "failed to load engine for '{}' at {}: {e}",
            checkpoint.subdir(),
            dir.display()
        )
    }))
}

/// Loads the engine for `checkpoint` from its split-layout export (`<repo>/onnx-split/<ckpt>`,
/// the layout laya-ts's exporter produces: `encoder.onnx` + `head.onnx`, no sibling
/// `tokenizer_config.json`), or prints a skip line and returns `None` when that directory is not
/// present locally — this is expected on most machines, so callers should skip rather than fail.
/// [`LayaEngine::from_directory`] auto-detects the layout (see
/// [`laya::model_artifacts::ModelArtifacts`]), so no separate split-aware constructor is needed
/// here.
pub fn load_split(checkpoint: LayaCheckpoint) -> Option<LayaEngine> {
    if !(test_artifacts::has_split_model(checkpoint)
        && test_artifacts::has_split_tokenizer(checkpoint))
    {
        eprintln!(
            "skip: no split-layout ONNX artifacts for '{}' under <repo>/onnx-split/{}",
            checkpoint.subdir(),
            checkpoint.subdir()
        );
        return None;
    }
    let dir = test_artifacts::locate_split(checkpoint).unwrap_or_else(|| {
        panic!(
            "has_split_model/has_split_tokenizer were true but locate_split() found nothing for \
             '{}'",
            checkpoint.subdir()
        )
    });
    Some(LayaEngine::from_directory(&dir).unwrap_or_else(|e| {
        panic!(
            "failed to load split-layout engine for '{}' at {}: {e}",
            checkpoint.subdir(),
            dir.display()
        )
    }))
}
