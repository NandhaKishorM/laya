//! Supply-chain integrity for downloaded checkpoints: opt-in revision pins and SHA-256 digest
//! checks. Ports `laya/revisions.py`.
//!
//! Two halves, both reachable the same way. [`resolve_revision`] picks the Hub revision to
//! download (an explicit argument, else `LAYA_REVISION`, or the word `reviewed` for the pinned
//! commit in [`PINNED_REVISIONS`]); [`verify_digests`] checks the SHA-256 of the files in an
//! artifact directory (an explicit map, else `LAYA_SHA256_DIGESTS`) before any of them is parsed.
//!
//! Differences from Python, all forced by the language or by this SDK's layout:
//! - Python's `snapshot_revision` (the commit a Hub snapshot directory points at) has no
//!   equivalent here: this SDK caches a download under `<cache>/<revision>/<checkpoint>`, so the
//!   revision an engine reports is the pin it was downloaded with, and `None` otherwise.
//! - The `"onnx"` / `"onnx_path"` digest key names the fused `model.onnx`. On a split checkpoint
//!   (`encoder.onnx` + `head.onnx`) there is no single graph file, so the key is refused; list
//!   `encoder.onnx` and `head.onnx` by path instead.
//! - Python raises `ValueError` / `FileNotFoundError`; those become [`LayaError::InvalidRevision`],
//!   [`LayaError::InvalidDigests`] and [`LayaError::DigestFileNotFound`], with Python's text.

use crate::error::{LayaError, Result};
use crate::language_detection::py_strip;
use crate::laya_router::python_repr_str;
use crate::python_json::PythonJson;
use crate::python_json_reader::PyJson;
use indexmap::IndexMap;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Opt-in reviewed commit SHAs of the published checkpoints, mirroring
/// `laya.revisions.PINNED_REVISIONS`. They are never applied implicitly, so existing caches keep
/// working; ask for one with `revision = Some("reviewed")` or `LAYA_REVISION=reviewed`.
pub const PINNED_REVISIONS: &[(&str, &str)] = &[
    (
        "convaiinnovations/laya",
        "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    ),
    (
        "convaiinnovations/laya-multilingual",
        "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    ),
    (
        "convaiinnovations/laya-typed-decisions",
        "1a793eb568e6718f15941d08f85432581df534e3",
    ),
];

/// The opt-in `LAYA_REVISION` value meaning "the reviewed SHA of the repository being loaded".
pub const REVIEWED: &str = "reviewed";

/// The reviewed SHA for `model_id`, matching repo ids case-insensitively (exact match first), as
/// Python's `_reviewed_pin` does.
pub fn reviewed_pin(model_id: &str) -> Option<&'static str> {
    if let Some((_, sha)) = PINNED_REVISIONS.iter().find(|(repo, _)| *repo == model_id) {
        return Some(sha);
    }
    let wanted = model_id.to_lowercase();
    PINNED_REVISIONS
        .iter()
        .find(|(repo, _)| repo.to_lowercase() == wanted)
        .map(|(_, sha)| *sha)
}

/// Picks the revision to download for `model_id`. Ports `resolve_revision`.
///
/// Both the argument and `LAYA_REVISION` are stripped, and the argument wins when it is non-empty.
/// A non-empty but blank argument (`"   "`) therefore wins and then strips away to `None`: it
/// suppresses the environment. An empty argument (or `None`) means "not asked for", so the
/// variable applies. The value `reviewed` looks `model_id` up in [`PINNED_REVISIONS`] and is an
/// error for a repository the table has no entry for, naming whichever of the argument and the
/// variable asked for it. Anything else is returned as is.
pub fn resolve_revision(model_id: &str, revision: Option<&str>) -> Result<Option<String>> {
    let argument = revision.filter(|r| !r.is_empty());
    let from_env;
    let chosen = match argument {
        Some(r) => r,
        None => {
            from_env = std::env::var("LAYA_REVISION").unwrap_or_default();
            &from_env
        }
    };
    let value = py_strip(chosen);
    if value.is_empty() {
        return Ok(None);
    }
    if value == REVIEWED {
        return match reviewed_pin(model_id) {
            Some(pinned) => Ok(Some(pinned.to_string())),
            None if argument.is_some() => Err(LayaError::InvalidRevision(format!(
                "laya: revision= asked for 'reviewed', but {} has no reviewed SHA in \
                 PINNED_REVISIONS; pass a commit, branch or tag in revision= instead",
                python_repr_str(model_id)
            ))),
            None => Err(LayaError::InvalidRevision(format!(
                "laya: LAYA_REVISION=reviewed, but {} has no reviewed SHA in \
                 PINNED_REVISIONS; set LAYA_REVISION to a commit or unset it",
                python_repr_str(model_id)
            ))),
        };
    }
    Ok(Some(value.to_string()))
}

/// Percent-encodes a revision for use as one URL path segment (`huggingface_hub` quotes it with
/// `safe=""`) and, identically, as one cache directory name: everything but `A-Z a-z 0-9 . _ - ~`
/// becomes `%XX`, so a branch such as `refs/pr/1` stays a single segment, and a revision that is
/// only dots cannot climb out of the cache. The mapping is injective, so two revisions never
/// share a folder.
pub fn encode_revision_segment(revision: &str) -> String {
    let mut out = String::with_capacity(revision.len());
    for byte in revision.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-' | b'~' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    if out == "." || out == ".." {
        out = out.replace('.', "%2E");
    }
    out
}

/// What the `"onnx"` / `"onnx_path"` digest key names.
#[derive(Debug, Clone, Copy)]
pub enum DigestGraph<'a> {
    /// The fused layout's `model.onnx`.
    Fused(&'a Path),
    /// A split layout: there is no single graph file, so the key is refused.
    Split,
}

/// Python's `ntpath.isabs` on a string whose backslashes are already forward slashes.
fn nt_isabs(s: &str) -> bool {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some('/'), _) => true,
        (Some(_), Some(':')) => chars.next() == Some('/'),
        _ => false,
    }
}

/// Python's `str(value)` for a parsed JSON value.
fn py_str(value: &PyJson) -> String {
    match value {
        PyJson::Str(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python's `repr(value)` for a parsed JSON value.
fn py_repr(value: &PyJson) -> String {
    match value {
        PyJson::Null => "None".to_string(),
        PyJson::Bool(true) => "True".to_string(),
        PyJson::Bool(false) => "False".to_string(),
        PyJson::Int(literal) => {
            if literal.trim_start_matches('-').bytes().all(|b| b == b'0') {
                "0".to_string()
            } else {
                literal.clone()
            }
        }
        PyJson::Float(f) if f.is_nan() => "nan".to_string(),
        PyJson::Float(f) if f.is_infinite() => if *f > 0.0 { "inf" } else { "-inf" }.to_string(),
        PyJson::Float(f) => PythonJson::repr(*f),
        PyJson::Str(s) => python_repr_str(s),
        PyJson::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        PyJson::Object(entries) => {
            let parts: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", python_repr_str(k), py_repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// A JSON object as the string map the digest checks use: values are `str()`-ed, as Python's
/// `str(want)` does for a digest that is not a string.
pub(crate) fn digest_map_from_entries(entries: &[(String, PyJson)]) -> IndexMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| (k.clone(), py_str(v)))
        .collect()
}

/// Parses `LAYA_SHA256_DIGESTS`: `Ok(None)` when it is unset or blank, the parsed JSON otherwise,
/// and Python's own message when it is not JSON.
pub(crate) fn digests_env_json() -> Result<Option<PyJson>> {
    let raw = std::env::var("LAYA_SHA256_DIGESTS").unwrap_or_default();
    let raw = py_strip(&raw);
    if raw.is_empty() {
        return Ok(None);
    }
    PyJson::parse(raw).map(Some).map_err(|_| {
        LayaError::InvalidDigests(
            "LAYA_SHA256_DIGESTS must be a JSON object of artifact->sha256".to_string(),
        )
    })
}

/// Verifies the SHA-256 digests of files under `model_dir` against `{relpath: hexdigest}`. Ports
/// `verify_digests`.
///
/// `expected = None` reads `LAYA_SHA256_DIGESTS` (a JSON object of artifact -> digest; unset or
/// blank checks nothing); `Some` of an empty map masks that variable. Absolute and `..` paths are
/// refused, a listed file that is absent is [`LayaError::DigestFileNotFound`], and a digest
/// mismatch is [`LayaError::InvalidDigests`]. The comparison ignores case and surrounding
/// whitespace in the expected digest. `graph` says what the `"onnx"` / `"onnx_path"` key means:
/// `Some(Fused(path))` verifies that file, `Some(Split)` refuses the key, and `None` reads the key
/// as an ordinary relative path (Python without an `onnx_path`).
pub fn verify_digests(
    model_dir: &Path,
    expected: Option<&IndexMap<String, String>>,
    graph: Option<DigestGraph<'_>>,
) -> Result<()> {
    let from_env;
    let expected = match expected {
        Some(map) => map,
        None => match digests_env_json()? {
            None => return Ok(()),
            Some(PyJson::Object(entries)) => {
                from_env = digest_map_from_entries(&entries);
                &from_env
            }
            Some(_) => {
                return Err(LayaError::InvalidDigests(
                    "expected_sha256 must be a mapping of artifact paths to digests".to_string(),
                ));
            }
        },
    };
    for (rel, want) in expected {
        let raw_rel = rel.replace('\\', "/");
        let path = match (rel.as_str(), graph) {
            ("onnx" | "onnx_path", Some(DigestGraph::Fused(path))) => path.to_path_buf(),
            ("onnx" | "onnx_path", Some(DigestGraph::Split)) => {
                return Err(LayaError::InvalidDigests(
                    "laya: the 'onnx' / 'onnx_path' digest key names the fused model.onnx, \
                     but this checkpoint is the split layout (encoder.onnx + head.onnx); \
                     list encoder.onnx and head.onnx by relative path instead"
                        .to_string(),
                ));
            }
            _ => {
                if raw_rel.starts_with('/') || nt_isabs(&raw_rel) {
                    return Err(LayaError::InvalidDigests(format!(
                        "laya: unsafe absolute path in expected digests: {}",
                        python_repr_str(rel)
                    )));
                }
                let rel_norm = raw_rel.trim_start_matches('/');
                if rel_norm.is_empty()
                    || rel_norm == ".."
                    || rel_norm.starts_with("../")
                    || rel_norm.contains("/../")
                {
                    return Err(LayaError::InvalidDigests(format!(
                        "laya: unsafe path in expected digests: {}",
                        python_repr_str(rel)
                    )));
                }
                // Joined as text, not component by component, so a name such as `C:x` stays a
                // file name under `model_dir` instead of re-rooting the path on Windows.
                let mut joined = model_dir.as_os_str().to_os_string();
                joined.push(std::path::MAIN_SEPARATOR_STR);
                joined.push(rel_norm);
                std::path::PathBuf::from(joined)
            }
        };
        if !path.is_file() {
            return Err(LayaError::DigestFileNotFound(format!(
                "laya: cannot verify {}: no such file under {}",
                python_repr_str(rel),
                model_dir.display()
            )));
        }
        let got = sha256_hex(&path)?;
        if got != py_strip(want).to_lowercase() {
            return Err(LayaError::InvalidDigests(format!(
                "laya: SHA-256 mismatch for {rel}: expected {want}, got {got}. The artifact does \
                 not match the reviewed digest; refusing to load it."
            )));
        }
    }
    Ok(())
}

/// Lower-case hex SHA-256 of a file, read in 1 MiB chunks.
fn sha256_hex(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_a_known_file_is_the_standard_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.bin");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_hex(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn revision_segments_are_injective_and_cannot_climb() {
        assert_eq!(encode_revision_segment("refs/pr/1"), "refs%2Fpr%2F1");
        assert_eq!(encode_revision_segment(".."), "%2E%2E");
        assert_eq!(encode_revision_segment("v1.2"), "v1.2");
        assert_ne!(
            encode_revision_segment("a%2Fb"),
            encode_revision_segment("a/b")
        );
    }
}
