//! Finds ONNX artifacts for each checkpoint, so model-backed tests can skip rather than fail on a
//! machine that has not downloaded the checkpoints.

use laya::laya_options::LayaCheckpoint;
use std::path::{Path, PathBuf};

/// Every checkpoint subdirectory name `LAYA_TEST_CHECKPOINTS` may select.
const KNOWN_SUBDIRS: [&str; 3] = ["multilingual", "english", "typed-decisions"];

/// The checkpoint subdirectories a parity job selected with `LAYA_TEST_CHECKPOINTS` (names
/// separated by spaces or commas), or `None` when the variable is unset or blank, which selects
/// every checkpoint, so ordinary local runs are unchanged.
///
/// A deselected checkpoint is skipped silently, without a `skip:` line, because another job owns
/// it. A selected checkpoint whose artifacts or goldens are missing still prints `skip:`, which
/// `tools/check_test_skips.py` rejects. An unknown name panics rather than selecting nothing, so
/// a typo cannot turn a job into one that tests no checkpoint.
pub fn selected_subdirs() -> Option<Vec<String>> {
    let raw = std::env::var("LAYA_TEST_CHECKPOINTS").unwrap_or_default();
    let names: Vec<String> = raw
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect();
    if names.is_empty() {
        return None;
    }
    if let Some(unknown) = names.iter().find(|n| !KNOWN_SUBDIRS.contains(&n.as_str())) {
        panic!("unknown LAYA_TEST_CHECKPOINTS entry '{unknown}' in '{raw}'");
    }
    Some(names)
}

/// Whether the checkpoint subdirectory `subdir` is selected for this run.
pub fn is_selected_subdir(subdir: &str) -> bool {
    selected_subdirs().is_none_or(|names| names.iter().any(|n| n == subdir))
}

/// Whether `checkpoint` is selected for this run (see [`selected_subdirs`]).
pub fn is_selected(checkpoint: LayaCheckpoint) -> bool {
    is_selected_subdir(checkpoint.subdir())
}

/// Resolve the artifact directory for `checkpoint`, or `None` if it cannot be found.
///
/// Resolution order: `LAYA_ONNX_DIR` (multilingual only, backward-compat single-checkpoint env
/// var), `LAYA_ONNX_ROOT` (parent of every checkpoint's named subdirectory), then walking up from
/// `CARGO_MANIFEST_DIR` looking for `onnx/<checkpoint>/`.
pub fn locate(checkpoint: LayaCheckpoint) -> Option<PathBuf> {
    let subfolder = checkpoint.subdir();

    if checkpoint == LayaCheckpoint::Multilingual
        && let Ok(env_dir) = std::env::var("LAYA_ONNX_DIR")
        && !env_dir.trim().is_empty()
    {
        // A set variable wins outright, even when it points nowhere: falling back would silently
        // test a different directory than the one asked for.
        let path = PathBuf::from(env_dir);
        return if path.is_dir() { Some(path) } else { None };
    }

    if let Ok(root) = std::env::var("LAYA_ONNX_ROOT")
        && !root.trim().is_empty()
    {
        let candidate = PathBuf::from(root).join(subfolder);
        if candidate.join("rl_agent_config.json").is_file() {
            return Some(candidate);
        }
        // Falls through to the walk-up so a single-checkpoint root doesn't block the rest.
    }

    let mut dir = Some(Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf());
    while let Some(d) = dir {
        let candidate = d.join("onnx").join(subfolder);
        if candidate.join("rl_agent_config.json").is_file() {
            return Some(candidate);
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    None
}

/// Whether `tokenizer/tokenizer.json` is present for `checkpoint`.
pub fn has_tokenizer(checkpoint: LayaCheckpoint) -> bool {
    locate(checkpoint).is_some_and(|dir| dir.join("tokenizer").join("tokenizer.json").is_file())
}

/// Whether the graph and its external-data sidecar are both present for `checkpoint`.
pub fn has_model(checkpoint: LayaCheckpoint) -> bool {
    locate(checkpoint).is_some_and(|dir| {
        dir.join("model.onnx").is_file() && dir.join("model.onnx.data").is_file()
    })
}

/// Resolve the split-layout artifact directory for `checkpoint`, or `None` if it cannot be
/// found.
///
/// `LAYA_ONNX_SPLIT_ROOT` selects the parent of the checkpoint directories when exports live
/// outside the checkout. Otherwise walk up from `CARGO_MANIFEST_DIR` for `onnx-split/<checkpoint>`.
/// Missing exports skip locally; the CI output guard rejects every such skip.
pub fn locate_split(checkpoint: LayaCheckpoint) -> Option<PathBuf> {
    let subfolder = checkpoint.subdir();
    if let Ok(root) = std::env::var("LAYA_ONNX_SPLIT_ROOT")
        && !root.trim().is_empty()
    {
        let candidate = PathBuf::from(root).join(subfolder);
        return candidate
            .join("rl_agent_config.json")
            .is_file()
            .then_some(candidate);
    }
    let mut dir = Some(Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf());
    while let Some(d) = dir {
        let candidate = d.join("onnx-split").join(subfolder);
        if candidate.join("rl_agent_config.json").is_file() {
            return Some(candidate);
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    None
}

/// Whether the split layout's root-level `tokenizer.json` is present for `checkpoint`.
pub fn has_split_tokenizer(checkpoint: LayaCheckpoint) -> bool {
    locate_split(checkpoint).is_some_and(|dir| dir.join("tokenizer.json").is_file())
}

/// Whether both `encoder.onnx` and `head.onnx` are present for `checkpoint`'s split-layout
/// export.
pub fn has_split_model(checkpoint: LayaCheckpoint) -> bool {
    locate_split(checkpoint)
        .is_some_and(|dir| dir.join("encoder.onnx").is_file() && dir.join("head.onnx").is_file())
}
