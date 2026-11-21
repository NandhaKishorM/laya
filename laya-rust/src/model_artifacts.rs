//! Resolved, validated paths to the files a Laya engine needs at startup.
//!
//! [`ModelArtifacts::resolve`] implements the full five-step resolution order (explicit directory -> `LAYA_ONNX_DIR` -> `LAYA_ONNX_ROOT`
//! -> cache -> opt-in download), and [`ModelArtifacts::from_directory`] validates a directory the
//! caller already knows about.
//!
//! Checkpoint validation (comparing the loaded `rl_agent_config.json`'s `encoder` field against the
//! checkpoint the caller asked for) is **not** done here: this module only records `Checkpoint`,
//! and `LayaEngine::create`'s validation step does the actual check once the config is loaded.
//! That check belongs in `laya_engine.rs`.

use crate::error::{LayaError, Result};
use crate::laya_options::{LayaCheckpoint, LayaOptions};
use crate::revisions::{DigestGraph, encode_revision_segment, resolve_revision, verify_digests};
use std::path::{Component, Path, PathBuf};

#[cfg(feature = "download")]
use crate::laya_options::ArtifactDownloadProgress;
#[cfg(feature = "download")]
use std::io::{Read, Write};
#[cfg(feature = "download")]
use std::sync::Arc;

/// Which of the two supported ONNX export layouts an artifact directory uses.
///
/// - `Fused`: a single graph, `model.onnx` + `model.onnx.data`, exposing `input_ids`,
///   `attention_mask`, `marker_pos`, `marker_mask`, `qtype` -> `logits`, `act_logits` directly.
/// - `Split`: two graphs, `encoder.onnx` + `head.onnx` — the layout laya-ts's exporter
///   (`scripts/export_onnx.py`) produces. `encoder.onnx` maps `(input_ids, attention_mask)` to
///   `last_hidden_state`; `head.onnx` maps `(hidden_states, marker_pos, marker_mask, qtype,
///   attention_mask)` to `(logits, act_logits)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLayout {
    /// `model.onnx` + `model.onnx.data`.
    Fused,
    /// `encoder.onnx` + `head.onnx`.
    Split,
}

/// The resolved graph file path(s) for whichever [`ModelLayout`] an artifact directory uses.
#[derive(Debug, Clone)]
pub enum ModelPaths {
    /// The fused layout's single graph file.
    Fused {
        /// Absolute path to `model.onnx`.
        model_path: PathBuf,
    },
    /// The split layout's two graph files.
    Split {
        /// Absolute path to `encoder.onnx`.
        encoder_path: PathBuf,
        /// Absolute path to `head.onnx`.
        head_path: PathBuf,
    },
}

impl ModelPaths {
    /// Which layout these paths belong to.
    pub fn layout(&self) -> ModelLayout {
        match self {
            ModelPaths::Fused { .. } => ModelLayout::Fused,
            ModelPaths::Split { .. } => ModelLayout::Split,
        }
    }
}

/// Resolved, validated paths to every file the Laya engine needs at startup.
#[derive(Debug, Clone)]
pub struct ModelArtifacts {
    /// The directory that contains the graph file(s).
    pub directory: PathBuf,
    /// The resolved graph file path(s); see [`ModelPaths::layout`] for which layout was found.
    pub model: ModelPaths,
    /// Absolute path to `rl_agent_config.json`.
    pub config_path: PathBuf,
    /// Absolute path to `tokenizer.json`. Looked up in `<directory>/tokenizer/` first, then
    /// directly in `<directory>` — the fused layout ships the former, the split layout the
    /// latter (see the module docs).
    pub tokenizer_path: PathBuf,
    /// Absolute path to `tokenizer_config.json`, when present, using the same `<directory>/
    /// tokenizer/` then `<directory>` lookup order as [`Self::tokenizer_path`]. The split layout
    /// never has this file by design; when absent, [`crate::tokenization::HfTokenizer`] falls
    /// back to resolving special tokens from `tokenizer.json`'s own `added_tokens`.
    pub tokenizer_config_path: Option<PathBuf>,
    /// The checkpoint this artifact directory corresponds to, or `None` when loaded from an
    /// explicit path (`LayaOptions::model_directory` or `LAYA_ONNX_DIR`), in which case no encoder
    /// validation is performed by later phases.
    pub checkpoint: Option<LayaCheckpoint>,
    /// The Hub commit this directory holds, when it came from the cache or a download: the
    /// `X-Repo-Commit` the Hub sent with every file (kept in a `.laya_revision` stamp file beside
    /// them, so a later cache hit reports it too), else the pin in force (see
    /// [`LayaOptions::revision`]), else `None`. The header is the stand-in for Python's
    /// `snapshot_revision` (the commit a Hub snapshot points at). It is dropped when a file lacks
    /// it or the files disagree. `None` for an explicit directory, `LAYA_ONNX_DIR` and
    /// `LAYA_ONNX_ROOT`.
    pub revision: Option<String>,
}

impl ModelArtifacts {
    /// Which ONNX export layout this artifact directory uses.
    pub fn layout(&self) -> ModelLayout {
        self.model.layout()
    }

    /// Whether `dir` contains a complete artifact for either supported layout: the graph file(s)
    /// for one layout (fused: `model.onnx` + `model.onnx.data`; split: `encoder.onnx` +
    /// `head.onnx`), plus `rl_agent_config.json` and a `tokenizer.json` resolvable via the
    /// `<dir>/tokenizer/` then `<dir>` lookup order. When `model.onnx` is present the fused
    /// layout is what gets checked, even if a split pair also happens to sit alongside it —
    /// matching [`Self::from_directory`]'s "fused wins" detection order. A `.part` file left over
    /// from an interrupted download does not count: the check looks for the final file name only.
    pub fn is_complete(dir: impl AsRef<Path>) -> bool {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return false;
        }
        let layout_complete = if dir.join("model.onnx").is_file() {
            dir.join("model.onnx.data").is_file()
        } else {
            dir.join("encoder.onnx").is_file() && dir.join("head.onnx").is_file()
        };
        layout_complete
            && dir.join("rl_agent_config.json").is_file()
            && find_tokenizer_json(dir).is_some()
    }

    /// Validates `dir` and resolves it to an absolute path, checking each required file
    /// individually so a missing one is named rather than surfacing as a generic "not found".
    ///
    /// Layout detection: if `model.onnx` exists, the fused layout is used (and `model.onnx.data`
    /// is then required next to it). Otherwise, if both `encoder.onnx` and `head.onnx` exist, the
    /// split layout is used. Otherwise this returns [`LayaError::MissingFile`] describing both
    /// layouts. A directory with only one of `encoder.onnx` / `head.onnx` (a half-written split
    /// export) also fails, naming the specific file that is missing.
    ///
    /// `checkpoint` is recorded as-is; no encoder-family validation against it happens here (see
    /// the module docs).
    pub fn from_directory(
        dir: impl AsRef<Path>,
        checkpoint: Option<LayaCheckpoint>,
    ) -> Result<Self> {
        let requested = dir.as_ref().to_path_buf();
        let resolved = absolute_path(&requested);

        // Reported before the per-file checks, and naming the requested path beside the resolved
        // one, because the two diverge in ways that are invisible otherwise: a POSIX shell eats
        // unquoted backslashes, and Windows then reads what is left of "D:\a\b" as the
        // drive-relative "D:ab", resolving it against the current directory. Naming only the
        // resolved path would turn that into a baffling "missing model.onnx" in a directory the
        // user never typed.
        if !resolved.is_dir() {
            return Err(LayaError::DirectoryNotFound {
                requested: if requested == resolved {
                    None
                } else {
                    Some(requested)
                },
                resolved,
            });
        }

        let model = resolve_model_paths(&resolved)?;

        let config_path = resolved.join("rl_agent_config.json");
        require_file(&config_path, None)?;

        let tokenizer_path =
            find_tokenizer_json(&resolved).ok_or_else(|| LayaError::MissingFile {
                path: resolved.join("tokenizer").join("tokenizer.json"),
                hint: Some(
                    "Looked for tokenizer.json in <directory>/tokenizer/ and then directly in \
                 <directory>; neither had it."
                        .to_string(),
                ),
            })?;
        let tokenizer_config_path = find_tokenizer_config_json(&resolved);

        Ok(ModelArtifacts {
            directory: resolved,
            model,
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            checkpoint,
            revision: None,
        })
    }

    /// [`Self::from_directory`] after the digest check: Python verifies
    /// `expected_sha256` (else `LAYA_SHA256_DIGESTS`) as soon as the directory is known and
    /// before it looks for `rl_agent_config.json` or the tokenizer, so a directory that is both
    /// incomplete and mismatched reports the digest. A directory that does not exist is reported
    /// by `from_directory` first, as Python reports a missing path first.
    fn verified(
        options: &LayaOptions,
        dir: &Path,
        checkpoint: Option<LayaCheckpoint>,
    ) -> Result<Self> {
        let resolved = absolute_path(dir);
        if resolved.is_dir() {
            let model = resolved.join("model.onnx");
            // The layout is told from the graph files alone, "fused wins" as in
            // `resolve_model_paths`; with neither present the `onnx` key is an ordinary path.
            let graph = if model.is_file() {
                Some(DigestGraph::Fused(&model))
            } else if resolved.join("encoder.onnx").is_file()
                || resolved.join("head.onnx").is_file()
            {
                Some(DigestGraph::Split)
            } else {
                None
            };
            verify_digests(&resolved, options.expected_sha256.as_ref(), graph)?;
        }
        Self::from_directory(dir, checkpoint)
    }

    /// Resolves and validates the artifact directory, downloading it when
    /// `options.allow_download` is `true` and it is not already present.
    ///
    /// Resolution order:
    ///
    /// 1. `options.model_directory`, if set — validated and used directly. No further steps.
    /// 2. The `LAYA_ONNX_DIR` environment variable, if set — backward-compat; always treated as
    ///    pointing directly at the artifact directory, independent of the checkpoint. No further
    ///    steps.
    /// 3. `LAYA_ONNX_ROOT/<checkpoint-subdir>`, if the env var is set and that directory contains
    ///    a complete artifact. Otherwise falls through to the remaining steps rather than failing
    ///    immediately, so a partially-downloaded cache still works.
    /// 4. `<cache root>/<checkpoint-subdir>`, if it already contains a complete artifact.
    /// 5. A download into the cache directory, if `options.allow_download` is `true`.
    /// 6. [`LayaError::ArtifactsNotFound`], listing every location tried, if none of the above
    ///    worked.
    ///
    /// Returns [`LayaError::DirectoryNotFound`] or [`LayaError::MissingFile`] when an explicit
    /// path (steps 1 or 2) does not check out, and [`LayaError::Download`] if a download is
    /// requested but the crate was built with `--no-default-features` (the `download` feature is
    /// on by default).
    ///
    /// Step 4 and 5 honour [`LayaOptions::revision`] / `LAYA_REVISION` (see
    /// [`crate::revisions::resolve_revision`]): with a pin, the cache folder is
    /// `<cache root>/<revision>/<checkpoint-subdir>` and a download comes from
    /// `resolve/<revision>/`; without one, both are exactly as before. The pin is resolved only
    /// once steps 1 to 3 have not answered, as Python resolves it only for a Hub id.
    pub fn resolve(options: &LayaOptions) -> Result<Self> {
        Self::resolve_with_endpoint(options, HF_ENDPOINT)
    }

    /// [`Self::resolve`] with the Hub host as a parameter, so tests can use a local server.
    fn resolve_with_endpoint(options: &LayaOptions, endpoint: &str) -> Result<Self> {
        #[cfg(not(feature = "download"))]
        let _ = endpoint;
        // 1. Explicit directory. No checkpoint inference: the caller knows what they loaded.
        if let Some(dir) = &options.model_directory
            && !dir.as_os_str().is_empty()
        {
            return Self::verified(options, dir, None);
        }

        // 2. LAYA_ONNX_DIR: backward-compat env var that points directly at the artifact dir.
        //    Treated as an explicit path for the same reason: it may point at any checkpoint.
        if let Ok(env_dir) = std::env::var("LAYA_ONNX_DIR")
            && !env_dir.trim().is_empty()
        {
            return Self::verified(options, Path::new(&env_dir), None);
        }

        let subfolder = options.checkpoint.subdir();

        // 3. LAYA_ONNX_ROOT: new env var for a tree that holds every checkpoint in named
        //    subdirectories (english/, multilingual/, typed-decisions/).
        let onnx_root = std::env::var("LAYA_ONNX_ROOT")
            .ok()
            .filter(|s| !s.trim().is_empty());
        if let Some(root) = &onnx_root {
            let root_dir = Path::new(root).join(subfolder);
            if Self::is_complete(&root_dir) {
                return Self::verified(options, &root_dir, Some(options.checkpoint));
            }
            // Not complete under LAYA_ONNX_ROOT — fall through to cache and download rather than
            // failing immediately.
        }

        // 4. Cache hit. A pinned load looks in its own per-revision folder.
        let revision = resolve_revision(&options.hugging_face_repo, options.revision.as_deref())?;
        let cache_root = resolve_cache_root(options);
        let artifact_dir = match &revision {
            Some(rev) => cache_root
                .join(encode_revision_segment(rev))
                .join(subfolder),
            None => cache_root.join(subfolder),
        };
        if Self::is_complete(&artifact_dir) {
            let mut artifacts = Self::verified(options, &artifact_dir, Some(options.checkpoint))?;
            artifacts.revision = read_revision_stamp(&artifact_dir).or(revision);
            return Ok(artifacts);
        }

        // 5. Download, if allowed.
        if options.allow_download {
            #[cfg(feature = "download")]
            {
                let commit =
                    download_artifact(options, &artifact_dir, endpoint, revision.as_deref())?;
                write_revision_stamp(&artifact_dir, commit.as_deref());
                let mut artifacts =
                    Self::verified(options, &artifact_dir, Some(options.checkpoint))?;
                artifacts.revision = commit.or(revision);
                return Ok(artifacts);
            }
            #[cfg(not(feature = "download"))]
            {
                return Err(LayaError::Download(
                    "LayaOptions::allow_download is true, but this crate was built without the \
                     `download` cargo feature (it is on by default; it was likely disabled with \
                     --no-default-features). Rebuild with the `download` feature enabled, or \
                     fetch the artifact manually and point LayaOptions::model_directory, \
                     LAYA_ONNX_DIR, or LAYA_ONNX_ROOT at it."
                        .to_string(),
                ));
            }
        }

        // 6. Nothing worked — report every location tried.
        Err(LayaError::ArtifactsNotFound {
            tried: build_tried_list(&onnx_root, subfolder, &artifact_dir),
        })
    }
}

/// Detects which layout `dir` uses and resolves + validates its graph file path(s).
///
/// `model.onnx` present wins outright (fused), even if a split pair also happens to sit alongside
/// it. Otherwise both `encoder.onnx` and `head.onnx` must be present (split). A directory with
/// only one of the split pair is treated as a validation error naming the missing file, not as
/// "neither layout" — the caller very likely meant to export the split layout and just failed
/// partway through.
fn resolve_model_paths(dir: &Path) -> Result<ModelPaths> {
    let fused_path = dir.join("model.onnx");
    if fused_path.is_file() {
        let data_path = dir.join("model.onnx.data");
        require_file(
            &data_path,
            Some(
                "The external weights file must sit next to model.onnx — ONNX Runtime resolves \
                 it relative to the graph file, and loading the graph on its own produces a \
                 baffling 'missing initializer' error.",
            ),
        )?;
        return Ok(ModelPaths::Fused {
            model_path: fused_path,
        });
    }

    let encoder_path = dir.join("encoder.onnx");
    let head_path = dir.join("head.onnx");
    let has_encoder = encoder_path.is_file();
    let has_head = head_path.is_file();

    if has_encoder || has_head {
        // A half-written split export: name the specific missing file rather than the generic
        // "neither layout found" message below.
        require_file(&encoder_path, None)?;
        require_file(&head_path, None)?;
        return Ok(ModelPaths::Split {
            encoder_path,
            head_path,
        });
    }

    Err(LayaError::MissingFile {
        path: fused_path,
        hint: Some(
            "Expected either the fused layout (model.onnx + model.onnx.data) or the split \
             layout (encoder.onnx + head.onnx, the layout laya-ts's exporter produces) in this \
             directory; found neither."
                .to_string(),
        ),
    })
}

/// Looks up `tokenizer.json` using the fused-then-split lookup order: `<dir>/tokenizer/` first
/// (the fused layout's location), then directly in `<dir>` (where the split layout's exporter
/// puts it).
fn find_tokenizer_json(dir: &Path) -> Option<PathBuf> {
    let nested = dir.join("tokenizer").join("tokenizer.json");
    if nested.is_file() {
        return Some(nested);
    }
    let root = dir.join("tokenizer.json");
    if root.is_file() {
        return Some(root);
    }
    None
}

/// Looks up `tokenizer_config.json` using the same lookup order as [`find_tokenizer_json`]. The
/// split layout never ships this file; callers must treat its absence as expected, not an error.
fn find_tokenizer_config_json(dir: &Path) -> Option<PathBuf> {
    let nested = dir.join("tokenizer").join("tokenizer_config.json");
    if nested.is_file() {
        return Some(nested);
    }
    let root = dir.join("tokenizer_config.json");
    if root.is_file() {
        return Some(root);
    }
    None
}

fn require_file(path: &Path, hint: Option<&str>) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(LayaError::MissingFile {
            path: path.to_path_buf(),
            hint: hint.map(str::to_string),
        })
    }
}

/// An absolute path: joins onto the current directory if relative, then lexically collapses `.`
/// and `..` components. Deliberately does not resolve symlinks and does not require the path to
/// exist (unlike `std::fs::canonicalize`, which fails outright on a path that isn't there yet —
/// exactly the case this function's caller needs to report clearly).
fn absolute_path(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };

    let mut result = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

/// The cache root: `options.cache_directory` if set, otherwise `%LOCALAPPDATA%\laya\onnx` on
/// Windows or `~/.cache/laya/onnx` elsewhere.
fn resolve_cache_root(options: &LayaOptions) -> PathBuf {
    if let Some(dir) = &options.cache_directory
        && !dir.as_os_str().is_empty()
    {
        return dir.clone();
    }
    default_cache_root()
}

#[cfg(target_os = "windows")]
fn default_cache_root() -> PathBuf {
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA")
        && !local_app_data.trim().is_empty()
    {
        return PathBuf::from(local_app_data).join("laya").join("onnx");
    }
    // %LOCALAPPDATA% is unset on some minimal/service accounts; fall back to a `.cache`
    // directory under the user profile.
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    PathBuf::from(home).join(".cache").join("laya").join("onnx")
}

#[cfg(not(target_os = "windows"))]
fn default_cache_root() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache").join("laya").join("onnx")
}

/// Builds the "Tried:" list for [`LayaError::ArtifactsNotFound`]. By the time this runs,
/// `LayaOptions::model_directory` and `LAYA_ONNX_DIR` are known to be unset (otherwise
/// [`ModelArtifacts::resolve`] would already have returned via one of those two steps), so those
/// two lines hardcode "(not set)" for the same reason.
fn build_tried_list(
    onnx_root: &Option<String>,
    subfolder: &str,
    artifact_dir: &Path,
) -> Vec<String> {
    let onnx_root_line = match onnx_root {
        Some(root) => format!(
            "  - LAYA_ONNX_ROOT env var        {}",
            Path::new(root).join(subfolder).display()
        ),
        None => "  - LAYA_ONNX_ROOT env var        (not set)".to_string(),
    };
    vec![
        "  - LayaOptions::model_directory  (not set)".to_string(),
        "  - LAYA_ONNX_DIR env var         (not set)".to_string(),
        onnx_root_line,
        format!(
            "  - Cache directory               {}",
            artifact_dir.display()
        ),
        String::new(),
        "To fix this, do one of:".to_string(),
        "  a) Download the artifact and point LayaOptions::model_directory or LAYA_ONNX_DIR at \
         its directory, or set LAYA_ONNX_ROOT to the parent of all checkpoints."
            .to_string(),
        "  b) Set LayaOptions::allow_download = true to download automatically.".to_string(),
        "Note: both model.onnx and model.onnx.data must reside in the same directory — ORT \
         resolves the external weights file relative to the graph file."
            .to_string(),
    ]
}

// ── download ───────────────────────────────────────────────────────────────

/// File in a cache folder that records the commit the Hub reported for its files.
const REVISION_STAMP: &str = ".laya_revision";

/// The commit recorded by [`write_revision_stamp`], if the stamp exists and is not blank.
fn read_revision_stamp(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(REVISION_STAMP)).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Records `commit` in the folder's stamp, or removes a stale stamp when there is none. Best
/// effort: a folder that cannot be stamped just reports the pin (or nothing) next time.
#[cfg_attr(not(feature = "download"), allow(dead_code))]
fn write_revision_stamp(dir: &Path, commit: Option<&str>) {
    let path = dir.join(REVISION_STAMP);
    match commit {
        Some(commit) => {
            let _ = std::fs::write(path, commit);
        }
        None => {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// The Hub host downloads come from. A parameter of the internal resolve path only so tests can
/// point it at a local server.
#[cfg_attr(not(feature = "download"), allow(dead_code))]
const HF_ENDPOINT: &str = "https://huggingface.co";

// Files to fetch for the fused layout, as (remote sub-path relative to the HF subfolder, local
// path segments relative to the destination directory).
#[cfg(feature = "download")]
const FUSED_ARTIFACT_FILES: &[(&str, &[&str])] = &[
    ("model.onnx", &["model.onnx"]),
    ("model.onnx.data", &["model.onnx.data"]),
    ("rl_agent_config.json", &["rl_agent_config.json"]),
    ("tokenizer/tokenizer.json", &["tokenizer", "tokenizer.json"]),
    (
        "tokenizer/tokenizer_config.json",
        &["tokenizer", "tokenizer_config.json"],
    ),
];

// Files to fetch for the split layout (laya-ts's exporter output): no `tokenizer/` subdirectory,
// no `tokenizer_config.json`.
#[cfg(feature = "download")]
const SPLIT_ARTIFACT_FILES: &[(&str, &[&str])] = &[
    ("encoder.onnx", &["encoder.onnx"]),
    ("encoder.onnx.data", &["encoder.onnx.data"]),
    ("head.onnx", &["head.onnx"]),
    ("head.onnx.data", &["head.onnx.data"]),
    ("rl_agent_config.json", &["rl_agent_config.json"]),
    ("tokenizer.json", &["tokenizer.json"]),
];

// External-data sidecars are only written when a graph's weights exceed the protobuf limit, so a
// split export may ship without them; these are skipped when absent instead of failing.
#[cfg(feature = "download")]
const OPTIONAL_SPLIT_FILES: &[&str] = &["encoder.onnx.data", "head.onnx.data"];

#[cfg(feature = "download")]
fn local_path(dest_dir: &Path, segments: &[&str]) -> PathBuf {
    segments
        .iter()
        .fold(dest_dir.to_path_buf(), |path, segment| path.join(segment))
}

/// Probes the remote repository to find out which layout it publishes, mirroring the local
/// detection order in [`resolve_model_paths`]: `model.onnx` wins if present, otherwise
/// `encoder.onnx` + `head.onnx`. Uses a `HEAD` request per candidate file so no bytes are
/// downloaded before the layout is known.
#[cfg(feature = "download")]
fn probe_remote_layout(
    agent: &ureq::Agent,
    base_url: &str,
    token: Option<&str>,
) -> Result<ModelLayout> {
    if remote_file_exists(agent, &format!("{base_url}model.onnx"), token) {
        return Ok(ModelLayout::Fused);
    }
    let has_encoder = remote_file_exists(agent, &format!("{base_url}encoder.onnx"), token);
    let has_head = remote_file_exists(agent, &format!("{base_url}head.onnx"), token);
    if has_encoder && has_head {
        return Ok(ModelLayout::Split);
    }
    Err(LayaError::Download(format!(
        "Hugging Face has neither the fused layout (model.onnx) nor the split layout \
         (encoder.onnx + head.onnx) at {base_url}.\n\n\
         The Laya repository publishes PyTorch weights (model.safetensors), which this SDK \
         cannot load: it needs an ONNX export. Unless that export has been uploaded to the \
         repository, allow_download cannot succeed.\n\n\
         Export the model once with the Python package (or laya-ts's exporter), then point the \
         SDK at the output directory via LAYA_ONNX_DIR or LayaOptions::model_directory."
    )))
}

/// `true` if a `HEAD` request against `url` succeeds. Any failure (404, network error, TLS error,
/// ...) is treated the same way — as "not present" — since the caller only needs a yes/no signal
/// to pick a file list, not a diagnostic; a real download attempt further down produces the
/// detailed 404/401/403 errors.
#[cfg(feature = "download")]
fn remote_file_exists(agent: &ureq::Agent, url: &str, token: Option<&str>) -> bool {
    let mut request = agent.head(url);
    if let Some(t) = token {
        request = request.header("Authorization", format!("Bearer {t}"));
    }
    request.call().is_ok()
}

/// Downloads every file in `files` into `dest_dir`, creating parent directories as needed (the
/// fused layout nests two files under `tokenizer/`; the split layout does not nest anything). One
/// `ureq::Agent` is built and reused for all files rather than a fresh client per file.
#[cfg(feature = "download")]
fn download_file_list(
    options: &LayaOptions,
    dest_dir: &Path,
    agent: &ureq::Agent,
    base_url: &str,
    token: Option<&str>,
    files: &[(&str, &[&str])],
    get_agent: &ureq::Agent,
) -> Result<Option<String>> {
    // The `X-Repo-Commit` of every file fetched; one missing or differing value voids the lot.
    let mut commits: Vec<Option<String>> = Vec::new();
    for (remote, local) in files {
        let url = format!("{base_url}{remote}");
        if OPTIONAL_SPLIT_FILES.contains(remote) && !remote_file_exists(agent, &url, token) {
            continue;
        }
        let dest_path = local_path(dest_dir, local);
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut part_os = dest_path.clone().into_os_string();
        part_os.push(".part");
        let part_path = PathBuf::from(part_os);
        let display_name = local.last().copied().unwrap_or(*remote);

        commits.push(fetch_to_file(
            get_agent,
            &url,
            display_name,
            &dest_path,
            &part_path,
            token,
            options.download_progress.as_ref(),
        )?);
    }
    Ok(match commits.split_first() {
        Some((Some(first), rest)) if rest.iter().all(|c| c.as_ref() == Some(first)) => {
            Some(first.clone())
        }
        _ => None,
    })
}

/// Detects which layout the remote repository publishes, then downloads that layout's artifact
/// files into `dest_dir`.
#[cfg(feature = "download")]
fn download_artifact(
    options: &LayaOptions,
    dest_dir: &Path,
    endpoint: &str,
    revision: Option<&str>,
) -> Result<Option<String>> {
    std::fs::create_dir_all(dest_dir)?;

    let token = options
        .hugging_face_token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .filter(|t| !t.is_empty());

    let agent = ureq::Agent::new_with_defaults();
    let get_agent = no_redirect_agent();

    // The English checkpoint lives at the bundle root; all others use a named subfolder.
    // hugging_face_download_subfolder() returns "" for English, so the URL has no extra segment.
    let hf_sub = options.hugging_face_download_subfolder();
    // Unpinned keeps `main`; a pin is one percent-encoded path segment, as `huggingface_hub`
    // quotes it.
    let rev_segment = revision.map_or_else(|| "main".to_string(), encode_revision_segment);
    let base_url = if hf_sub.is_empty() {
        format!(
            "{endpoint}/{}/resolve/{rev_segment}/",
            options.hugging_face_repo
        )
    } else {
        format!(
            "{endpoint}/{}/resolve/{rev_segment}/{}/",
            options.hugging_face_repo, hf_sub
        )
    };

    let layout = probe_remote_layout(&agent, &base_url, token.as_deref())?;
    let files = match layout {
        ModelLayout::Fused => FUSED_ARTIFACT_FILES,
        ModelLayout::Split => SPLIT_ARTIFACT_FILES,
    };
    download_file_list(
        options,
        dest_dir,
        &agent,
        &base_url,
        token.as_deref(),
        files,
        &get_agent,
    )
}

/// An agent that hands a 3xx response back instead of following it, so [`fetch_to_file`] can read
/// `X-Repo-Commit` off the first hop (the Hub sends it there, not on the CDN's final response) and
/// follow the redirects itself.
#[cfg(feature = "download")]
fn no_redirect_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .max_redirects(0)
        .max_redirects_will_error(false)
        .build()
        .into()
}

/// Most redirects [`fetch_to_file`] follows for one file.
#[cfg(feature = "download")]
const MAX_REDIRECTS: usize = 10;

/// `scheme://host[:port]` of an absolute URL (the whole string when it has no path).
#[cfg(feature = "download")]
fn url_origin(url: &str) -> &str {
    let after_scheme = url.find("://").map_or(0, |i| i + 3);
    match url[after_scheme..].find(['/', '?', '#']) {
        Some(i) => &url[..after_scheme + i],
        None => url,
    }
}

/// Resolves a `Location` value against the URL that returned it: absolute, scheme-relative,
/// root-relative or path-relative.
#[cfg(feature = "download")]
fn resolve_location(base: &str, location: &str) -> String {
    if location.contains("://") {
        return location.to_string();
    }
    let origin = url_origin(base);
    if let Some(rest) = location.strip_prefix("//") {
        let scheme = origin.split("://").next().unwrap_or("https");
        return format!("{scheme}://{rest}");
    }
    if location.starts_with('/') {
        return format!("{origin}{location}");
    }
    let path = &base[origin.len()..];
    let path = path.split(['?', '#']).next().unwrap_or("");
    let dir = path.rfind('/').map_or("/", |i| &path[..=i]);
    format!("{origin}{dir}{location}")
}

/// The download seam: fetches one URL to `part_path`, then atomically promotes it to
/// `dest_path`, returning the `X-Repo-Commit` of the first response when it has one (the Hub sends
/// it on the redirect, not on the CDN's final response; see [`no_redirect_agent`]). Kept as a
/// standalone function (rather than inlined into
/// [`download_artifact`]) so tests can point it at a local `TcpListener` instead of a real
/// HuggingFace URL and exercise the 404 / 401 / success paths without any network access.
#[cfg(feature = "download")]
fn fetch_to_file(
    agent: &ureq::Agent,
    url: &str,
    display_name: &str,
    dest_path: &Path,
    part_path: &Path,
    token: Option<&str>,
    progress: Option<&Arc<dyn Fn(ArtifactDownloadProgress) + Send + Sync>>,
) -> Result<Option<String>> {
    // Redirects are followed here, not by the agent (see `no_redirect_agent`). The token goes to
    // the host the caller named and to no other: the CDN a file redirects to must not see it.
    let mut current = url.to_string();
    let mut hops = 0usize;
    let mut commit: Option<String> = None;

    // A bare status-code error here would surface as "404 (Not Found)" with no hint as to which
    // file was missing or what to do about it. The two failures worth naming:
    //   404 — the repository does not publish the ONNX export at all (it ships safetensors, which
    //         this SDK cannot load), or the subfolder is wrong.
    //   401/403 — a gated or private repository that needs a token.
    let mut response = loop {
        let mut request = agent.get(&current);
        if let Some(t) = token
            && url_origin(&current) == url_origin(url)
        {
            request = request.header("Authorization", format!("Bearer {t}"));
        }
        match request.call() {
            Ok(response) => {
                if hops == 0 {
                    commit = response
                        .headers()
                        .get("x-repo-commit")
                        .and_then(|v| v.to_str().ok())
                        .map(|v| v.trim().to_string())
                        .filter(|v| !v.is_empty());
                }
                if response.status().is_redirection() {
                    hops += 1;
                    if hops > MAX_REDIRECTS {
                        return Err(LayaError::Download(format!(
                            "downloading '{display_name}' from {url}: more than {MAX_REDIRECTS} redirects"
                        )));
                    }
                    let Some(location) = response
                        .headers()
                        .get("location")
                        .and_then(|v| v.to_str().ok())
                    else {
                        return Err(LayaError::Download(format!(
                            "downloading '{display_name}' from {url}: redirect without a Location"
                        )));
                    };
                    current = resolve_location(&current, location);
                    continue;
                }
                break response;
            }
            Err(ureq::Error::StatusCode(404)) => {
                return Err(LayaError::Download(format!(
                    "Hugging Face has no '{display_name}' at {url} (404).\n\n\
                 The Laya repository publishes PyTorch weights (model.safetensors), which this \
                 SDK cannot load: it needs an ONNX export (model.onnx + model.onnx.data). Unless \
                 that export has been uploaded to the repository, allow_download cannot \
                 succeed.\n\n\
                 Export the model once with the Python package, then point the SDK at the output \
                 directory via LAYA_ONNX_DIR or LayaOptions::model_directory."
                )));
            }
            Err(ureq::Error::StatusCode(code)) if code == 401 || code == 403 => {
                return Err(LayaError::Download(format!(
                    "Hugging Face refused access to {url} ({code}).\n\
                 The repository is gated or private. Set LayaOptions::hugging_face_token, or the \
                 HF_TOKEN environment variable, to a token that can read it."
                )));
            }
            Err(e) => {
                return Err(LayaError::Download(format!(
                    "downloading '{display_name}' from {url}: {e}"
                )));
            }
        }
    };

    let total = response.body().content_length();

    let mut file = std::fs::File::create(part_path)?;
    {
        let mut reader = response.body_mut().as_reader();
        let mut buffer = [0u8; 81_920];
        let mut received: u64 = 0;
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])?;
            received += read as u64;
            if let Some(callback) = progress {
                callback(ArtifactDownloadProgress {
                    file_name: display_name.to_string(),
                    bytes_received: received,
                    total_bytes: total,
                });
            }
        }
    }
    drop(file);

    // Atomic promotion: a .part file that survived means a previous run was interrupted, not
    // that the destination is valid. Windows can't rename onto an existing file, so remove it
    // first.
    if dest_path.is_file() {
        std::fs::remove_file(dest_path)?;
    }
    std::fs::rename(part_path, dest_path)?;
    Ok(commit)
}

#[cfg(all(test, feature = "download"))]
mod download_tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Mutex;

    /// Starts a one-shot HTTP server on an ephemeral port, replies with `response` to the single
    /// connection it accepts, and returns the base URL (`http://127.0.0.1:PORT`). The accept loop
    /// runs on a background thread and this function does not join it — the test process exits
    /// (or the listener is dropped) when the test ends, which is enough to clean it up.
    fn spawn_once_server(response: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                // Read (and discard) whatever the client sent; we don't need to parse it, just
                // drain enough that the client isn't blocked writing its request.
                let _ = stream.read(&mut buf);
                let _ = std::io::Write::write_all(&mut stream, &response);
                let _ = std::io::Write::flush(&mut stream);
            }
        });
        format!("http://{addr}")
    }

    fn http_response(status_line: &str, body: &[u8]) -> Vec<u8> {
        let mut head = format!(
            "{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        head.extend_from_slice(body);
        head
    }

    #[test]
    fn fetch_to_file_success_streams_body_and_reports_progress() {
        let body = b"hello world, this is the fake artifact body";
        let base = spawn_once_server(http_response("HTTP/1.1 200 OK", body));
        let url = format!("{base}/model.onnx");

        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("model.onnx");
        let part = dir.path().join("model.onnx.part");

        let agent = ureq::Agent::new_with_defaults();
        let received_totals: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = received_totals.clone();
        let callback: Arc<dyn Fn(ArtifactDownloadProgress) + Send + Sync> =
            Arc::new(move |p: ArtifactDownloadProgress| {
                sink.lock().unwrap().push(p.bytes_received);
                assert_eq!(p.total_bytes, Some(body.len() as u64));
            });

        fetch_to_file(
            &agent,
            &url,
            "model.onnx",
            &dest,
            &part,
            None,
            Some(&callback),
        )
        .expect("download should succeed");

        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(
            !part.exists(),
            ".part file must be renamed away, not left behind"
        );
        assert!(
            !received_totals.lock().unwrap().is_empty(),
            "progress callback should have fired at least once"
        );
    }

    #[test]
    fn fetch_to_file_404_gives_descriptive_error() {
        let base = spawn_once_server(http_response("HTTP/1.1 404 Not Found", b""));
        let url = format!("{base}/model.onnx");

        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("model.onnx");
        let part = dir.path().join("model.onnx.part");

        let agent = ureq::Agent::new_with_defaults();
        let err = fetch_to_file(&agent, &url, "model.onnx", &dest, &part, None, None)
            .expect_err("404 must be an error");
        let message = err.to_string();
        assert!(message.contains("404"), "message was: {message}");
        assert!(
            message.contains("safetensors") || message.contains("ONNX export"),
            "message should explain the safetensors-vs-ONNX gap: {message}"
        );
        assert!(!dest.exists());
        assert!(!part.exists());
    }

    #[test]
    fn fetch_to_file_401_gives_descriptive_error() {
        let base = spawn_once_server(http_response("HTTP/1.1 401 Unauthorized", b""));
        let url = format!("{base}/model.onnx");

        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("model.onnx");
        let part = dir.path().join("model.onnx.part");

        let agent = ureq::Agent::new_with_defaults();
        let err = fetch_to_file(&agent, &url, "model.onnx", &dest, &part, None, None)
            .expect_err("401 must be an error");
        let message = err.to_string();
        assert!(message.contains("401"), "message was: {message}");
        assert!(
            message.contains("HF_TOKEN") || message.contains("hugging_face_token"),
            "message should point at the token settings: {message}"
        );
    }

    #[test]
    fn fetch_to_file_403_gives_descriptive_error() {
        let base = spawn_once_server(http_response("HTTP/1.1 403 Forbidden", b""));
        let url = format!("{base}/model.onnx");

        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("model.onnx");
        let part = dir.path().join("model.onnx.part");

        let agent = ureq::Agent::new_with_defaults();
        let err = fetch_to_file(&agent, &url, "model.onnx", &dest, &part, None, None)
            .expect_err("403 must be an error");
        assert!(err.to_string().contains("403"));
    }

    #[test]
    fn fetch_to_file_sends_bearer_token_when_provided() {
        // The server echoes back whether it saw an Authorization header, as the response body,
        // so the test can assert on it without a full HTTP request parser.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request_text = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
                let saw_bearer = request_text.contains("authorization: bearer secret-token");
                let body = if saw_bearer {
                    b"ok".as_slice()
                } else {
                    b"missing".as_slice()
                };
                let response = http_response("HTTP/1.1 200 OK", body);
                let _ = std::io::Write::write_all(&mut stream, &response);
                let _ = std::io::Write::flush(&mut stream);
            }
        });
        let url = format!("http://{addr}/model.onnx");

        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("model.onnx");
        let part = dir.path().join("model.onnx.part");
        let agent = ureq::Agent::new_with_defaults();

        fetch_to_file(
            &agent,
            &url,
            "model.onnx",
            &dest,
            &part,
            Some("secret-token"),
            None,
        )
        .expect("download should succeed");

        assert_eq!(std::fs::read(&dest).unwrap(), b"ok");
    }

    // ── pinned downloads and the per-revision cache ────────────────────────────

    /// Serves every connection until the process ends: `HEAD` gets the headers of a 4-byte body,
    /// `GET` gets the body. Returns the base URL and the `"METHOD /path"` of every request seen.
    fn spawn_recording_server() -> (String, Arc<Mutex<Vec<String>>>) {
        spawn_server_with_commits(|_| None)
    }

    /// As [`spawn_recording_server`], but every `GET` response carries `X-Repo-Commit:
    /// <commit_for(path)>` when that returns `Some`.
    fn spawn_server_with_commits(
        commit_for: fn(&str) -> Option<&'static str>,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let mut line = text.lines().next().unwrap_or("").split_whitespace();
                let method = line.next().unwrap_or("").to_string();
                let path = line.next().unwrap_or("").to_string();
                log.lock().unwrap().push(format!("{method} {path}"));
                let body: &[u8] = if method == "HEAD" { b"" } else { b"data" };
                let commit = match commit_for(&path) {
                    Some(c) if method == "GET" => format!("X-Repo-Commit: {c}\r\n"),
                    _ => String::new(),
                };
                let mut response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: 4\r\n{commit}Connection: close\r\n\r\n"
                )
                .into_bytes();
                response.extend_from_slice(body);
                let _ = std::io::Write::write_all(&mut stream, &response);
                let _ = std::io::Write::flush(&mut stream);
            }
        });
        (format!("http://{addr}"), seen)
    }

    /// Serializes the tests below, which set process-wide environment variables.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Holds [`ENV_LOCK`] and removes the variables that steer artifact resolution (the test run
    /// points them at paths that do not exist), restoring them on drop.
    struct CleanEnv {
        _lock: std::sync::MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl CleanEnv {
        const KEYS: [&'static str; 4] = [
            "LAYA_ONNX_DIR",
            "LAYA_ONNX_ROOT",
            "LAYA_REVISION",
            "HF_TOKEN",
        ];

        fn new() -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let saved = Self::KEYS
                .iter()
                .map(|k| (*k, std::env::var(k).ok()))
                .collect();
            for key in Self::KEYS {
                // SAFETY: `ENV_LOCK` is held for this guard's whole life.
                unsafe { std::env::remove_var(key) };
            }
            Self { _lock: lock, saved }
        }

        fn set(&self, key: &str, value: &str) {
            // SAFETY: as above.
            unsafe { std::env::set_var(key, value) };
        }
    }

    impl Drop for CleanEnv {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                match value {
                    Some(v) => unsafe { std::env::set_var(key, v) },
                    None => unsafe { std::env::remove_var(key) },
                }
            }
        }
    }

    fn download_options(cache: &Path, repo: &str) -> LayaOptions {
        let mut options = LayaOptions::new();
        options.allow_download = true;
        options.cache_directory = Some(cache.to_path_buf());
        options.hugging_face_repo = repo.to_string();
        options.checkpoint = LayaCheckpoint::Multilingual;
        options
    }

    #[test]
    fn pinned_download_uses_the_revision_url_and_a_per_revision_cache() {
        let env = CleanEnv::new();
        let (base, seen) = spawn_recording_server();
        let cache = tempfile::tempdir().expect("tempdir");
        let mut options = download_options(cache.path(), "owner/repo");
        options.revision = Some(" refs/pr/1 ".to_string());

        let artifacts = ModelArtifacts::resolve_with_endpoint(&options, &base)
            .expect("pinned download should succeed");

        let folder = cache.path().join("refs%2Fpr%2F1").join("multilingual");
        assert_eq!(artifacts.directory, absolute_path(&folder));
        assert_eq!(artifacts.revision.as_deref(), Some("refs/pr/1"));
        assert_eq!(std::fs::read(folder.join("model.onnx")).unwrap(), b"data");
        assert!(
            !cache.path().join("multilingual").exists(),
            "a pinned load must not write the unpinned layout"
        );
        let requests = seen.lock().unwrap().clone();
        assert!(!requests.is_empty());
        for request in &requests {
            assert!(
                request.contains("/owner/repo/resolve/refs%2Fpr%2F1/multilingual/"),
                "{request}"
            );
        }
        assert!(requests.contains(
            &"GET /owner/repo/resolve/refs%2Fpr%2F1/multilingual/model.onnx.data".to_string()
        ));

        // The second load is a cache hit: nothing listens at this endpoint any more.
        let again = ModelArtifacts::resolve_with_endpoint(&options, "http://127.0.0.1:1")
            .expect("per-revision cache hit");
        assert_eq!(again.directory, artifacts.directory);
        assert_eq!(again.revision.as_deref(), Some("refs/pr/1"));

        // Another revision does not see that folder, and neither does an unpinned load.
        options.revision = Some("other".to_string());
        assert!(ModelArtifacts::resolve_with_endpoint(&options, "http://127.0.0.1:1").is_err());
        options.revision = None;
        assert!(ModelArtifacts::resolve_with_endpoint(&options, "http://127.0.0.1:1").is_err());
        drop(env);
    }

    #[test]
    fn reviewed_from_the_environment_downloads_the_pinned_commit() {
        let env = CleanEnv::new();
        env.set("LAYA_REVISION", "reviewed");
        let (base, seen) = spawn_recording_server();
        let cache = tempfile::tempdir().expect("tempdir");
        let options = download_options(cache.path(), "convaiinnovations/laya-multilingual");

        let artifacts = ModelArtifacts::resolve_with_endpoint(&options, &base).expect("download");

        let sha = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67";
        assert_eq!(artifacts.revision.as_deref(), Some(sha));
        assert!(
            artifacts
                .directory
                .ends_with(Path::new(sha).join("multilingual"))
        );
        assert!(
            seen.lock()
                .unwrap()
                .iter()
                .all(|r| r.contains(&format!("/resolve/{sha}/multilingual/")))
        );
    }

    #[test]
    fn unpinned_download_is_unchanged() {
        let env = CleanEnv::new();
        let (base, seen) = spawn_recording_server();
        let cache = tempfile::tempdir().expect("tempdir");
        let options = download_options(cache.path(), "owner/repo");

        let artifacts = ModelArtifacts::resolve_with_endpoint(&options, &base).expect("download");

        assert_eq!(artifacts.revision, None);
        assert_eq!(
            artifacts.directory,
            absolute_path(&cache.path().join("multilingual"))
        );
        assert!(
            cache
                .path()
                .join("multilingual")
                .join("model.onnx")
                .is_file()
        );
        for request in seen.lock().unwrap().iter() {
            assert!(
                request.contains("/owner/repo/resolve/main/multilingual/"),
                "{request}"
            );
        }
        drop(env);
    }

    #[test]
    fn unpinned_english_download_keeps_the_bundle_root_url() {
        let _env = CleanEnv::new();
        let (base, seen) = spawn_recording_server();
        let cache = tempfile::tempdir().expect("tempdir");
        let mut options = download_options(cache.path(), "owner/repo");
        options.checkpoint = LayaCheckpoint::English;

        ModelArtifacts::resolve_with_endpoint(&options, &base).expect("download");

        assert!(
            seen.lock()
                .unwrap()
                .contains(&"GET /owner/repo/resolve/main/model.onnx".to_string())
        );
        assert!(cache.path().join("english").join("model.onnx").is_file());
    }

    #[test]
    fn reviewed_without_a_pin_fails_before_any_download() {
        let _env = CleanEnv::new();
        let (base, seen) = spawn_recording_server();
        let cache = tempfile::tempdir().expect("tempdir");
        let mut options = download_options(cache.path(), "owner/repo");
        options.revision = Some("reviewed".to_string());

        let err = ModelArtifacts::resolve_with_endpoint(&options, &base).expect_err("no pin");

        assert!(matches!(err, LayaError::InvalidRevision(_)), "{err}");
        assert!(seen.lock().unwrap().is_empty());
    }

    // ── the X-Repo-Commit stamp ────────────────────────────────────────────────

    const DEAD: &str = "http://127.0.0.1:1";

    #[test]
    fn unpinned_download_reports_the_hub_commit_and_a_cache_hit_still_does() {
        let _env = CleanEnv::new();
        let (base, _seen) = spawn_server_with_commits(|_| Some("c0ffee"));
        let cache = tempfile::tempdir().expect("tempdir");
        let options = download_options(cache.path(), "owner/repo");

        let first = ModelArtifacts::resolve_with_endpoint(&options, &base).expect("download");
        assert_eq!(first.revision.as_deref(), Some("c0ffee"));
        let stamp = cache.path().join("multilingual").join(REVISION_STAMP);
        assert_eq!(std::fs::read_to_string(stamp).unwrap(), "c0ffee");

        let again = ModelArtifacts::resolve_with_endpoint(&options, DEAD).expect("cache hit");
        assert_eq!(again.revision.as_deref(), Some("c0ffee"));
    }

    #[test]
    fn the_hub_commit_wins_over_the_pin_it_resolved_from() {
        let _env = CleanEnv::new();
        let (base, _seen) = spawn_server_with_commits(|_| Some("deadbeef"));
        let cache = tempfile::tempdir().expect("tempdir");
        let mut options = download_options(cache.path(), "owner/repo");
        options.revision = Some("main".to_string());

        let first = ModelArtifacts::resolve_with_endpoint(&options, &base).expect("download");
        assert_eq!(first.revision.as_deref(), Some("deadbeef"));
        let again = ModelArtifacts::resolve_with_endpoint(&options, DEAD).expect("cache hit");
        assert_eq!(again.revision.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn without_the_header_a_download_reports_the_pin_else_nothing_and_writes_no_stamp() {
        let _env = CleanEnv::new();
        let (base, _seen) = spawn_recording_server();
        let cache = tempfile::tempdir().expect("tempdir");
        let options = download_options(cache.path(), "owner/repo");
        let unpinned = ModelArtifacts::resolve_with_endpoint(&options, &base).expect("download");
        assert_eq!(unpinned.revision, None);
        assert!(
            !cache
                .path()
                .join("multilingual")
                .join(REVISION_STAMP)
                .exists()
        );
        let hit = ModelArtifacts::resolve_with_endpoint(&options, DEAD).expect("cache hit");
        assert_eq!(hit.revision, None);

        let mut pinned = download_options(cache.path(), "owner/repo");
        pinned.revision = Some("v1".to_string());
        let first = ModelArtifacts::resolve_with_endpoint(&pinned, &base).expect("download");
        assert_eq!(first.revision.as_deref(), Some("v1"));
        let hit = ModelArtifacts::resolve_with_endpoint(&pinned, DEAD).expect("cache hit");
        assert_eq!(hit.revision.as_deref(), Some("v1"));
    }

    #[test]
    fn files_that_disagree_or_lack_the_header_fall_back_to_the_pin() {
        let _env = CleanEnv::new();
        for commit_for in [
            (|p: &str| {
                if p.ends_with("model.onnx.data") {
                    Some("bbb")
                } else {
                    Some("aaa")
                }
            }) as fn(&str) -> Option<&'static str>,
            |p: &str| {
                if p.ends_with("rl_agent_config.json") {
                    None
                } else {
                    Some("aaa")
                }
            },
        ] {
            let (base, _seen) = spawn_server_with_commits(commit_for);
            let cache = tempfile::tempdir().expect("tempdir");
            let mut options = download_options(cache.path(), "owner/repo");
            options.revision = Some("v1".to_string());
            let got = ModelArtifacts::resolve_with_endpoint(&options, &base).expect("download");
            assert_eq!(got.revision.as_deref(), Some("v1"));
            let folder = cache.path().join("v1").join("multilingual");
            assert!(!folder.join(REVISION_STAMP).exists());
        }
    }

    #[test]
    fn a_local_directory_reports_no_revision_even_beside_a_stamp() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(REVISION_STAMP), "abc").unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"g").unwrap();
        std::fs::write(dir.path().join("model.onnx.data"), b"d").unwrap();
        std::fs::write(dir.path().join("rl_agent_config.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        let mut options = LayaOptions::new();
        options.model_directory = Some(dir.path().to_path_buf());
        let _env = CleanEnv::new();
        assert_eq!(ModelArtifacts::resolve(&options).unwrap().revision, None);
    }

    // ── redirects: X-Repo-Commit is on the first hop only ─────────────────────

    type Handler = Box<dyn Fn(&str) -> String + Send>;

    /// A server that answers each request with whatever `handler(path)` returns (a full raw
    /// response). Returns the base URL and a log of `"GET /path auth=<bool>"` lines.
    fn spawn_handler_server(handler: Handler) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let mut line = text.lines().next().unwrap_or("").split_whitespace();
                let method = line.next().unwrap_or("").to_string();
                let path = line.next().unwrap_or("").to_string();
                let auth = text
                    .to_ascii_lowercase()
                    .contains("authorization: bearer tok");
                log.lock()
                    .unwrap()
                    .push(format!("{method} {path} auth={auth}"));
                let _ = std::io::Write::write_all(&mut stream, handler(&path).as_bytes());
                let _ = std::io::Write::flush(&mut stream);
            }
        });
        (format!("http://{addr}"), seen)
    }

    fn redirect(status: &str, location: &str, extra: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nLocation: {location}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
    }

    fn ok_body(extra: &str) -> String {
        format!("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n{extra}Connection: close\r\n\r\nbytes")
    }

    fn fetch(
        url: &str,
    ) -> (
        Result<Option<String>>,
        std::path::PathBuf,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("f.bin");
        let part = dir.path().join("f.bin.part");
        let result = fetch_to_file(
            &no_redirect_agent(),
            url,
            "f.bin",
            &dest,
            &part,
            Some("tok"),
            None,
        );
        (result, dest, dir)
    }

    #[test]
    fn commit_is_read_from_the_first_hop_and_a_relative_location_is_followed() {
        let (base, seen) = spawn_handler_server(Box::new(|path| match path {
            "/f" => redirect(
                "307 Temporary Redirect",
                "/cdn/f",
                "X-Repo-Commit: abc123\r\n",
            ),
            "/cdn/f" => ok_body(""),
            _ => redirect("404 Not Found", "/", ""),
        }));
        let (result, dest, _dir) = fetch(&format!("{base}/f"));
        assert_eq!(result.expect("download").as_deref(), Some("abc123"));
        assert_eq!(std::fs::read(dest).unwrap(), b"bytes");
        assert_eq!(
            *seen.lock().unwrap(),
            ["GET /f auth=true", "GET /cdn/f auth=true"],
            "the same host still gets the token"
        );
    }

    #[test]
    fn an_absolute_location_on_another_host_does_not_get_the_token() {
        let (cdn, cdn_seen) = spawn_handler_server(Box::new(|_| ok_body("")));
        let target = format!("{cdn}/blob");
        let (base, seen) = spawn_handler_server(Box::new(move |_| {
            redirect("302 Found", &target, "X-Repo-Commit: def456\r\n")
        }));
        let (result, dest, _dir) = fetch(&format!("{base}/f"));
        assert_eq!(result.expect("download").as_deref(), Some("def456"));
        assert_eq!(std::fs::read(dest).unwrap(), b"bytes");
        assert_eq!(*seen.lock().unwrap(), ["GET /f auth=true"]);
        assert_eq!(*cdn_seen.lock().unwrap(), ["GET /blob auth=false"]);
    }

    #[test]
    fn a_header_only_on_the_final_response_is_not_the_commit() {
        let (base, _seen) = spawn_handler_server(Box::new(|path| match path {
            "/f" => redirect("302 Found", "cdn/f", ""),
            _ => ok_body("X-Repo-Commit: late\r\n"),
        }));
        let (result, dest, _dir) = fetch(&format!("{base}/f"));
        assert_eq!(result.expect("download"), None);
        assert!(dest.is_file());
    }

    #[test]
    fn a_redirect_loop_is_an_error() {
        let (base, seen) = spawn_handler_server(Box::new(|_| redirect("302 Found", "/f", "")));
        let (result, _dest, _dir) = fetch(&format!("{base}/f"));
        let message = result.expect_err("loop").to_string();
        assert!(message.contains("redirects"), "{message}");
        assert_eq!(seen.lock().unwrap().len(), MAX_REDIRECTS + 1);
    }

    #[test]
    fn resolve_location_handles_every_form() {
        let base = "https://h.example:8443/a/b/c.bin?x=1";
        assert_eq!(
            resolve_location(base, "https://o.example/z"),
            "https://o.example/z"
        );
        assert_eq!(
            resolve_location(base, "//o.example/z"),
            "https://o.example/z"
        );
        assert_eq!(resolve_location(base, "/z"), "https://h.example:8443/z");
        assert_eq!(resolve_location(base, "z"), "https://h.example:8443/a/b/z");
        assert_eq!(
            resolve_location("http://h.example", "z"),
            "http://h.example/z"
        );
    }
}
