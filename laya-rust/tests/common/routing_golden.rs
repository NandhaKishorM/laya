//! Access to the checkpoint-independent routing golden vectors recorded by
//! `tools/dump_routing_golden.py`: `tests/golden/routing/*.json`.
//!
//! Mirrors [`super::golden_data`]'s "read from the source tree, don't copy" approach and its
//! `LAYA_GOLDEN_DIR` override (the routing directory is `LAYA_GOLDEN_DIR/routing` when that
//! variable is set, else the same default base directory `golden_data::golden_base_dir` uses).

use serde_json::Value;
use std::path::PathBuf;

/// The `golden/routing/` directory.
pub fn routing_dir() -> PathBuf {
    super::golden_data::golden_base_dir().join("routing")
}

/// Parse one file under `golden/routing/`, or `None` if the routing golden tree is not present
/// at all (so Tier 1 tests can print a skip line instead of failing on a checkout that has not
/// run Phase A's dump script).
pub fn load(file_name: &str) -> Option<Value> {
    let path = routing_dir().join(file_name);
    if !path.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read routing golden file {}: {e}", path.display()));
    Some(serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "failed to parse routing golden file {}: {e}",
            path.display()
        )
    }))
}

/// [`load`], expecting a top-level JSON array.
pub fn load_array(file_name: &str) -> Option<Vec<Value>> {
    load(file_name).map(|v| {
        v.as_array()
            .unwrap_or_else(|| panic!("{file_name} must be a JSON array"))
            .clone()
    })
}
