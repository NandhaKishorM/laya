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
        })
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
    pub fn resolve(options: &LayaOptions) -> Result<Self> {
        // 1. Explicit directory. No checkpoint inference: the caller knows what they loaded.
        if let Some(dir) = &options.model_directory
            && !dir.as_os_str().is_empty()
        {
            return Self::from_directory(dir, None);
        }

        // 2. LAYA_ONNX_DIR: backward-compat env var that points directly at the artifact dir.
        //    Treated as an explicit path for the same reason: it may point at any checkpoint.
        if let Ok(env_dir) = std::env::var("LAYA_ONNX_DIR")
            && !env_dir.trim().is_empty()
        {
            return Self::from_directory(&env_dir, None);
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
                return Self::from_directory(&root_dir, Some(options.checkpoint));
            }
            // Not complete under LAYA_ONNX_ROOT — fall through to cache and download rather than
            // failing immediately.
        }

        // 4. Cache hit.
        let cache_root = resolve_cache_root(options);
        let artifact_dir = cache_root.join(subfolder);
        if Self::is_complete(&artifact_dir) {
            return Self::from_directory(&artifact_dir, Some(options.checkpoint));
        }

        // 5. Download, if allowed.
        if options.allow_download {
            #[cfg(feature = "download")]
            {
                download_artifact(options, &artifact_dir)?;
                return Self::from_directory(&artifact_dir, Some(options.checkpoint));
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
) -> Result<()> {
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

        fetch_to_file(
            agent,
            &url,
            display_name,
            &dest_path,
            &part_path,
            token,
            options.download_progress.as_ref(),
        )?;
    }
    Ok(())
}

/// Detects which layout the remote repository publishes, then downloads that layout's artifact
/// files into `dest_dir`.
#[cfg(feature = "download")]
fn download_artifact(options: &LayaOptions, dest_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dest_dir)?;

    let token = options
        .hugging_face_token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .filter(|t| !t.is_empty());

    let agent = ureq::Agent::new_with_defaults();

    // The English checkpoint lives at the bundle root; all others use a named subfolder.
    // hugging_face_download_subfolder() returns "" for English, so the URL has no extra segment.
    let hf_sub = options.hugging_face_download_subfolder();
    let base_url = if hf_sub.is_empty() {
        format!(
            "https://huggingface.co/{}/resolve/main/",
            options.hugging_face_repo
        )
    } else {
        format!(
            "https://huggingface.co/{}/resolve/main/{}/",
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
    )
}

/// The download seam: fetches one URL to `part_path`, then atomically promotes it to
/// `dest_path`. Kept as a standalone function (rather than inlined into
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
) -> Result<()> {
    let mut request = agent.get(url);
    if let Some(t) = token {
        request = request.header("Authorization", format!("Bearer {t}"));
    }

    // A bare status-code error here would surface as "404 (Not Found)" with no hint as to which
    // file was missing or what to do about it. The two failures worth naming:
    //   404 — the repository does not publish the ONNX export at all (it ships safetensors, which
    //         this SDK cannot load), or the subfolder is wrong.
    //   401/403 — a gated or private repository that needs a token.
    let mut response = match request.call() {
        Ok(response) => response,
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
    Ok(())
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
}
