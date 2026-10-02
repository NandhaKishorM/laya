//! [`LayaError`], the one error type every fallible operation in this crate returns.

use std::path::PathBuf;

/// The crate-wide result alias.
pub type Result<T> = std::result::Result<T, LayaError>;

/// Everything that can go wrong building, configuring, or running a Laya engine.
///
/// Kept as one flat enum (rather than one type per module) so a caller can match on it without
/// chasing error types across `questions`, `config`, `artifacts`, and so on.
#[derive(Debug, thiserror::Error)]
pub enum LayaError {
    /// A filesystem operation failed (reading a config, a tokenizer file, an artifact, ...).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A JSON document could not be parsed or did not have the expected shape.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// No artifact directory was found after trying every resolution step.
    #[error("Laya model artifact (~1.3 GB) not found. Tried:\n{}", .tried.join("\n"))]
    ArtifactsNotFound {
        /// One line per location that was tried, in resolution order.
        tried: Vec<String>,
    },

    /// The resolved artifact directory does not exist.
    #[error(
        "Laya artifact directory does not exist.{}\nResolved  : {}\n\n\
         If the resolved path does not look like what you typed, quote it: an unquoted \
         Windows path loses its backslashes in a POSIX shell. Forward slashes also work.",
        .requested.as_ref().map(|r| format!("\nRequested : {}", r.display())).unwrap_or_default(),
        .resolved.display()
    )]
    DirectoryNotFound {
        /// The path exactly as given, when it differs from `resolved`.
        requested: Option<PathBuf>,
        /// The requested path after normalization to an absolute path.
        resolved: PathBuf,
    },

    /// A required artifact file (`model.onnx`, `model.onnx.data`, ...) is missing.
    #[error(
        "Missing artifact file: {}{}",
        .path.display(),
        .hint.as_deref().map(|h| format!("\n{h}")).unwrap_or_default()
    )]
    MissingFile {
        /// The absolute path that was checked.
        path: PathBuf,
        /// An optional extra hint about why this file matters.
        hint: Option<String>,
    },

    /// Fetching an artifact from HuggingFace Hub failed.
    #[error("{0}")]
    Download(String),

    /// The tokenizer could not be loaded or used.
    #[error("{0}")]
    Tokenizer(String),

    /// The ONNX Runtime session could not be built or run.
    #[error("{0}")]
    Onnx(String),

    /// A `Question` or `QuestionSet` was constructed with invalid data (duplicate ids or
    /// labels, zero options, ...).
    #[error("{0}")]
    InvalidQuestion(String),

    /// `rl_agent_config.json` could not be interpreted.
    #[error("{0}")]
    InvalidConfig(String),

    /// The loaded checkpoint's encoder family does not match what the caller asked for.
    #[error("{0}")]
    CheckpointMismatch(String),

    /// One or more of a question's option markers were truncated away by `max_len`, so the
    /// question cannot be answered without silently scoring fewer options than it was given.
    /// Ports the "silently answering a truncated question" failure mode called out in
    /// `sequence-construction.md`, and matches Python's wording exactly (parity tests compare
    /// this text against golden `error_message` values byte-for-byte).
    #[error("question '{id}' options exceed head_max_len={head_max_len}")]
    NoMarkers {
        /// The question id that lost one or more markers.
        id: String,
        /// The configured head budget the question's options did not fit within.
        head_max_len: usize,
    },
}
