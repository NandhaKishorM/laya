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
use laya::laya_options::{LayaCheckpoint, LayaOptions};
use serde_json::Value;

use super::golden_data::CheckpointGoldenData;
use super::test_artifacts;

/// The engine options for an artifact directory: the directory, plus the `lang_temperatures`
/// table the checkpoint's goldens were recorded with, so the per-language cases run on an engine
/// built as Python's agent was.
fn options_for(checkpoint: LayaCheckpoint, dir: &std::path::Path) -> LayaOptions {
    LayaOptions {
        model_directory: Some(dir.to_path_buf()),
        lang_temperatures: CheckpointGoldenData::for_checkpoint(checkpoint.subdir()).lang_table(),
        ..Default::default()
    }
}

/// The hand-made calibration file the `calibration_*` golden cases were recorded with
/// (`tools/calibration_fixture.json`), which an engine under test loads through
/// `LayaOptions::calibration`.
pub fn calibration_fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/calibration_fixture.json")
}

/// Whether a recorded case was made with the calibration file.
pub fn recorded_with_calibration(root: &Value) -> bool {
    root.get("calibration").is_some_and(Value::is_object)
}

/// For a case recorded with a calibration file: the file it names is the fixture, and what it held
/// then is what it holds now, so an engine loaded from the fixture is the engine that was recorded.
pub fn assert_fixture_is_the_recorded_file(root: &Value) {
    if !recorded_with_calibration(root) {
        return;
    }
    let recorded = &root["calibration"];
    assert_eq!(
        Some("calibration_fixture.json"),
        recorded["file"].as_str(),
        "the file a calibration case names"
    );
    let path = calibration_fixture();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let current: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        recorded["payload"], current,
        "calibration_fixture.json has changed since the golden was recorded; run tools/regen_golden.py"
    );
}

/// The engines one checkpoint's recorded cases run on: the plain engine, and for a case recorded
/// with the calibration file one loaded with it (a file replaces the checkpoint's temperatures for
/// every question, so those cases cannot share the plain engine). At most one engine is resident:
/// the calibration cases are the last in each index, so the plain one is released when the first of
/// them arrives.
pub struct CaseEngines {
    checkpoint: LayaCheckpoint,
    split: bool,
    current: Option<(bool, LayaEngine)>,
}

impl CaseEngines {
    /// The engines of `checkpoint`'s fused export, or `None` (with a skip line) without artifacts.
    pub fn fused(checkpoint: LayaCheckpoint) -> Option<Self> {
        load(checkpoint).map(|engine| Self {
            checkpoint,
            split: false,
            current: Some((false, engine)),
        })
    }

    /// The engines of `checkpoint`'s split export, or `None` (with a skip line) without artifacts.
    pub fn split(checkpoint: LayaCheckpoint) -> Option<Self> {
        load_split(checkpoint).map(|engine| Self {
            checkpoint,
            split: true,
            current: Some((false, engine)),
        })
    }

    /// The engine that answers the recorded case `root`.
    pub fn for_case(&mut self, root: &Value) -> &LayaEngine {
        let calibrated = recorded_with_calibration(root);
        assert_fixture_is_the_recorded_file(root);
        if self.current.as_ref().map(|(was, _)| *was) != Some(calibrated) {
            self.current = None;
            let engine = if self.split {
                load_split_with(self.checkpoint, calibrated)
            } else {
                load_with(self.checkpoint, calibrated)
            };
            self.current = engine.map(|engine| (calibrated, engine));
        }
        &self
            .current
            .as_ref()
            .expect("the artifacts were present when the engines were first loaded")
            .1
    }
}

/// Every checkpoint, in the order golden data and shape-sweep tables list them.
pub const CHECKPOINTS: [LayaCheckpoint; 3] = [
    LayaCheckpoint::Multilingual,
    LayaCheckpoint::English,
    LayaCheckpoint::TypedDecisions,
];

/// The checkpoints this run covers: every one, or those `LAYA_TEST_CHECKPOINTS` selects (see
/// [`test_artifacts::selected_subdirs`]). Loops over checkpoints use this, never [`CHECKPOINTS`]
/// directly, so a parity job only touches the checkpoints it owns.
pub fn checkpoints() -> Vec<LayaCheckpoint> {
    CHECKPOINTS
        .into_iter()
        .filter(|c| test_artifacts::is_selected(*c))
        .collect()
}

/// Like [`load`], for a test pinned to one checkpoint: returns `None` without a skip line when
/// that checkpoint is not selected, because another parity job owns it.
pub fn load_selected(checkpoint: LayaCheckpoint) -> Option<LayaEngine> {
    if !test_artifacts::is_selected(checkpoint) {
        return None;
    }
    load(checkpoint)
}

/// Loads the engine for `checkpoint`, or prints a skip line and returns `None` when its ONNX
/// artifacts are not available locally. Panics (rather than returning `None`) if the artifacts
/// are present but fail to load — that is a real bug, not a missing-fixture skip.
pub fn load(checkpoint: LayaCheckpoint) -> Option<LayaEngine> {
    load_with(checkpoint, false)
}

/// [`load`], with the calibration fixture when `calibrated`.
fn load_with(checkpoint: LayaCheckpoint, calibrated: bool) -> Option<LayaEngine> {
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
    let mut options = options_for(checkpoint, &dir);
    if calibrated {
        options.calibration = Some(calibration_fixture());
    }
    Some(LayaEngine::create(options).unwrap_or_else(|e| {
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
    load_split_with(checkpoint, false)
}

/// [`load_split`], with the calibration fixture when `calibrated`.
fn load_split_with(checkpoint: LayaCheckpoint, calibrated: bool) -> Option<LayaEngine> {
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
    let mut options = options_for(checkpoint, &dir);
    if calibrated {
        options.calibration = Some(calibration_fixture());
    }
    Some(LayaEngine::create(options).unwrap_or_else(|e| {
        panic!(
            "failed to load split-layout engine for '{}' at {}: {e}",
            checkpoint.subdir(),
            dir.display()
        )
    }))
}
