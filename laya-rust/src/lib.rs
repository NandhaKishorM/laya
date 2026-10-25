//! Rust port of the Laya Python SDK (the `laya` package on PyPI): typed-decision inference over a
//! local ONNX model, run on ONNX Runtime instead of PyTorch.
//!
//! The crate covers the full engine (tokenization, sequence building, batching, the ONNX forward
//! pass and calibration) plus the router, language detection, email cleaning and the embedding
//! shortlist. Its answers are checked against recorded Python output by the golden-vector tests.
//!
//! Module names follow the Python SDK's modules where a direct counterpart exists
//! (`laya_router.rs` <-> `laya/router.py`, `laya_shortlist.rs` <-> `laya/shortlist.py`,
//! `sequence_builder.rs` <-> `build_sequence` in `laya/common.py`, ...) so a reader who knows the Python SDK can find the
//! equivalent Rust code by name. `error.rs` is the one Rust-only module — Python surfaces failures
//! as ordinary exceptions, which Rust has no equivalent of, so every fallible operation here
//! returns [`LayaError`] instead.
//!
//! ## Two ONNX export layouts
//!
//! [`model_artifacts::ModelArtifacts::from_directory`] (and therefore [`LayaEngine::create`] /
//! [`LayaEngine::from_directory`]) auto-detects which of two layouts a checkpoint directory holds:
//! **fused** (a single `model.onnx` + `model.onnx.data`, the layout `tools/export_onnx.py`
//! produces) or **split** (`encoder.onnx` + `head.onnx`, the layout laya-ts's exporter produces).
//! [`model_artifacts::ModelLayout`] and [`model_artifacts::ModelPaths`] report which one was
//! found and resolved. See the crate [README](https://github.com/NandhaKishorM/laya/blob/main/laya-rust/README.md#getting-the-model)
//! and [MODELS.md](https://github.com/NandhaKishorM/laya/blob/main/laya-rust/MODELS.md#4-output-layout)
//! for the full directory layouts.
//!
//! ## Design notes
//!
//! - **No `tokenizer.nopost.json` strip-and-cache step.** The `tokenizers` Rust crate exposes
//!   `encode(text, add_special_tokens: false)` directly — exactly what Python's `laya/common.py`
//!   does — so there is no need to rewrite `tokenizer.json` with `post_processor: null` and cache
//!   the result just to get an un-post-processed encode.
//! - **`PythonJson::criterion`'s `default=str` fallback is not reachable.** The three dialect
//!   functions in [`python_json`] take `&serde_json::Value`, which can only ever hold
//!   JSON-representable data; there is no "opaque object with no JSON representation" case for a
//!   fallback-to-`ToString()` path to handle. The blank-criterion rule and the float/escaping
//!   rules are otherwise byte-for-byte identical to Python's `json.dumps`.
//! - **Warnings go through the `log` facade.** Python's `warnings.warn` calls that this crate
//!   ports (a checkpoint whose temperatures were clamped, and `predict_long` capping the window or
//!   reducing the stride) are sent to the [`log`](https://docs.rs/log) facade at `warn` level with
//!   target `laya` and Python's text. [`laya_config::LayaConfig::clamped_temperatures`] still
//!   records the clamped entries.

pub mod answers;
pub mod batch_prediction;
pub mod calibration;
pub mod collator;
pub mod confidence;
pub mod error;
pub mod language_detection;
pub mod laya_config;
pub mod laya_email;
pub mod laya_engine;
pub mod laya_options;
pub mod laya_presets;
pub mod laya_router;
pub mod laya_shortlist;
pub mod long_prediction;
pub mod model_artifacts;
pub mod predict_options;
pub mod python_json;
mod python_json_reader;
pub mod questions;
pub mod sequence_builder;
pub mod tokenization;

pub use answers::{
    ActionInfo, Answer, ChoiceAnswer, LayaResult, NoulAnswer, ScanUsage, ScoreAnswer, StateUsage,
    Usage, WindowInfo,
};
pub use batch_prediction::BatchOptions;
pub use calibration::Calibration;
pub use collator::{CollatedBatch, Collator, SequenceItem};
pub use confidence::{GATE_STATES, MinConfidence, bucket_keys};
pub use error::{LayaError, Result};
pub use language_detection::{LanguageAnalysis, LanguageDetection, LatinProfile};
pub use laya_config::LayaConfig;
pub use laya_email::LayaEmail;
pub use laya_engine::LayaEngine;
pub use laya_options::{
    ArtifactDownloadProgress, LayaCheckpoint, LayaExecutionProvider, LayaOptions,
};
pub use laya_presets::LayaPresets;
pub use laya_router::{
    BatchRequest, GuessInput, LangGuess, LayaBatchPredictor, LayaLongPredictor, LayaPredictor,
    LayaRouter, LayaRouterOptions, RouteDecision,
};
pub use laya_shortlist::{DEFAULT_SHORTLIST_K, LayaShortlist, ShortlistInfo};
pub use long_prediction::LongOptions;
pub use model_artifacts::{ModelArtifacts, ModelLayout, ModelPaths};
pub use predict_options::PredictOptions;
pub use python_json::PythonJson;
pub use questions::{
    ChoiceQuestion, NoulQuestion, Question, QuestionSet, QuestionType, ScoreQuestion,
};
pub use sequence_builder::SequenceBuilder;
pub use tokenization::{HfTokenizer, LayaTokenizer, SpecialTokens};
