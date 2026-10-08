//! Checkpoint selection and engine configuration. Ports the options `laya.load` / `laya.Agent`
//! accept in `laya/agent.py`.

use std::path::PathBuf;
use std::sync::Arc;

/// Selects which Laya checkpoint to load. Each maps to a different encoder and context length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LayaCheckpoint {
    /// mmBERT-base (322M), 1024-token context, 100+ languages. The safest default for
    /// mixed-language traffic; covers English too, so there is no router to configure.
    #[default]
    Multilingual,
    /// ModernBERT-large (421M), 512-token context, English-optimised. Higher accuracy on
    /// English-only workloads; accuracy collapses on non-Latin scripts.
    English,
    /// ModernBERT-large (421M), 1024-token context, fine-tuned on four typed-decisions
    /// workflows. Use only for the specific workflows it was trained on.
    TypedDecisions,
}

impl LayaCheckpoint {
    /// The local artifact subdirectory name. Used inside the cache root and inside an `onnx/`
    /// tree (e.g. `onnx/multilingual/`).
    pub fn subdir(self) -> &'static str {
        match self {
            LayaCheckpoint::Multilingual => "multilingual",
            LayaCheckpoint::English => "english",
            LayaCheckpoint::TypedDecisions => "typed-decisions",
        }
    }

    /// The subfolder path used in HuggingFace download URLs. For English the checkpoint lives at
    /// the bundle root (empty string), which differs from the local directory name `"english"`.
    pub fn hf_download_subfolder(self) -> &'static str {
        match self {
            LayaCheckpoint::Multilingual => "multilingual",
            LayaCheckpoint::English => "",
            LayaCheckpoint::TypedDecisions => "typed-decisions",
        }
    }
}

/// ONNX execution provider. `Cpu` is always available; the GPU providers need the matching cargo
/// feature, and requesting one without it returns a clear error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayaExecutionProvider {
    /// CPU execution (default).
    #[default]
    Cpu,
    /// NVIDIA CUDA. Requires the crate's `cuda` feature.
    Cuda,
    /// Windows DirectML. Requires the crate's `directml` feature.
    DirectMl,
}

/// Progress report for a single file being downloaded from HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct ArtifactDownloadProgress {
    /// The file name being fetched, e.g. `model.onnx.data`.
    pub file_name: String,
    /// Bytes written to disk so far.
    pub bytes_received: u64,
    /// Total file size in bytes, or `None` when the server omits `Content-Length`.
    pub total_bytes: Option<u64>,
}

/// Controls how the Laya engine locates or downloads its ONNX artifact and how it configures
/// ONNX Runtime.
///
/// `Clone`: `laya_router.rs` keeps one `LayaOptions` template shared across every checkpoint it
/// may load and clones it per load, overriding only `checkpoint`. Every field here is itself
/// cheaply cloneable (`Arc` for the progress callback, owned `String`/`PathBuf` elsewhere).
#[derive(Clone)]
pub struct LayaOptions {
    /// The directory that contains `model.onnx`. When set, skips all other resolution steps.
    pub model_directory: Option<PathBuf>,
    /// Which checkpoint to load. The subdirectory, HuggingFace subfolder, and cache path all
    /// derive from this value unless overridden by `hugging_face_subfolder` or `model_directory`.
    pub checkpoint: LayaCheckpoint,
    /// Which hardware back-end to use.
    pub execution_provider: LayaExecutionProvider,
    /// Number of threads used within a single ONNX operator. `None` lets ONNX Runtime choose.
    pub intra_op_threads: Option<u32>,
    /// Number of threads used to run independent ONNX operators in parallel. `None` lets ONNX
    /// Runtime choose.
    pub inter_op_threads: Option<u32>,
    /// Whether the engine may fetch the artifact from HuggingFace Hub when it is not found
    /// locally. Downloading is always opt-in.
    pub allow_download: bool,
    /// HuggingFace repository that hosts the artifact.
    pub hugging_face_repo: String,
    /// Subfolder override for non-standard bundle layouts. `None` derives it from `checkpoint`;
    /// public so `LayaOptions { .., ..Default::default() }` compiles outside the crate.
    pub hugging_face_subfolder: Option<String>,
    /// HuggingFace access token used in the `Authorization: Bearer` header. When `None`, falls
    /// back to the `HF_TOKEN` environment variable. Set to `Some(String::new())` to suppress all
    /// auth (public repos only).
    pub hugging_face_token: Option<String>,
    /// Root directory under which the downloaded artifact is stored. `None` resolves to
    /// `%LOCALAPPDATA%\laya\onnx` on Windows and `~/.cache/laya/onnx` elsewhere.
    pub cache_directory: Option<PathBuf>,
    /// Optional sink for per-file download progress.
    pub download_progress: Option<Arc<dyn Fn(ArtifactDownloadProgress) + Send + Sync>>,
}

impl std::fmt::Debug for LayaOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayaOptions")
            .field("model_directory", &self.model_directory)
            .field("checkpoint", &self.checkpoint)
            .field("execution_provider", &self.execution_provider)
            .field("intra_op_threads", &self.intra_op_threads)
            .field("inter_op_threads", &self.inter_op_threads)
            .field("allow_download", &self.allow_download)
            .field("hugging_face_repo", &self.hugging_face_repo)
            .field("hugging_face_subfolder", &self.hugging_face_subfolder)
            .field(
                "hugging_face_token",
                &self.hugging_face_token.as_ref().map(|_| "<redacted>"),
            )
            .field("cache_directory", &self.cache_directory)
            .field("download_progress", &self.download_progress.is_some())
            .finish()
    }
}

impl Default for LayaOptions {
    fn default() -> Self {
        Self {
            model_directory: None,
            checkpoint: LayaCheckpoint::default(),
            execution_provider: LayaExecutionProvider::default(),
            intra_op_threads: None,
            inter_op_threads: None,
            allow_download: false,
            hugging_face_repo: "convaiinnovations/laya".to_string(),
            hugging_face_subfolder: None,
            hugging_face_token: None,
            cache_directory: None,
            download_progress: None,
        }
    }
}

impl LayaOptions {
    /// The default options: multilingual checkpoint, CPU, no download.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subfolder within the repository and local cache. Derived from `checkpoint` unless set
    /// explicitly with [`Self::set_hugging_face_subfolder`].
    ///
    /// Note that the HuggingFace bundle subfolder for the English checkpoint is the bundle root
    /// (empty string), not `"english"`. When this is left at its default, download URLs are
    /// computed correctly; if you override it for English you must use the empty string for the
    /// download path, while `"english"` remains the conventional local directory name.
    pub fn hugging_face_subfolder(&self) -> String {
        self.hugging_face_subfolder
            .clone()
            .unwrap_or_else(|| self.checkpoint.subdir().to_string())
    }

    /// Override the HuggingFace subfolder. Prefer setting `checkpoint` instead unless loading a
    /// custom export that does not follow the standard bundle layout.
    pub fn set_hugging_face_subfolder(&mut self, value: impl Into<String>) {
        self.hugging_face_subfolder = Some(value.into());
    }

    /// The subfolder path to use in HuggingFace download URLs specifically. Differs from
    /// [`Self::hugging_face_subfolder`] only when the subfolder was never overridden and the
    /// checkpoint is English (see [`LayaCheckpoint::hf_download_subfolder`]).
    pub fn hugging_face_download_subfolder(&self) -> String {
        self.hugging_face_subfolder
            .clone()
            .unwrap_or_else(|| self.checkpoint.hf_download_subfolder().to_string())
    }
}
