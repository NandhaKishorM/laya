//! Answers typed questions about a piece of state in one forward pass. Ports the inference path
//! of Python's `laya.Agent` (`laya/agent.py`).
//!
//! Loading the model is expensive and the session is reusable, so create one engine and keep it:
//! it is safe to share across threads (`Arc<LayaEngine>`) and works well as a singleton. Every
//! question in a [`QuestionSet`] is answered by a single batched inference call, matching
//! Python's single forward pass — asking five questions costs roughly what asking one does.
//!
//! ## `Session::run` needs `&mut self`
//!
//! `ort` 2.x's `Session::run` takes `&mut self`, so the ONNX session is held in a [`Mutex`] here
//! so [`LayaEngine`] itself can stay lock-free everywhere else and still be `Send + Sync`.
//! Tokenization needs no such lock: `tokenizers::Tokenizer::encode` takes `&self`.

use crate::answers::{
    ActionInfo, Answer, ChoiceAnswer, LayaResult, NoulAnswer, ScoreAnswer, Usage,
};
use crate::calibration::Calibration;
use crate::collator::{CollatedBatch, Collator, SequenceItem};
use crate::error::{LayaError, Result};
use crate::laya_config::LayaConfig;
use crate::laya_options::{LayaCheckpoint, LayaExecutionProvider, LayaOptions};
use crate::model_artifacts::{ModelArtifacts, ModelPaths};
use crate::questions::{Question, QuestionSet};
use crate::sequence_builder::SequenceBuilder;
use crate::tokenization::{HfTokenizer, LayaTokenizer};
use indexmap::IndexMap;
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

const MODEL_NAME: &str = "laya-rl-agent";

/// The raw graph outputs of one forward pass, before calibration: the logits and action logits
/// tuple the engine's internal run step produces.
///
/// Public (rather than crate-private) only so the integration test suite — a separate crate that
/// links against this one — can drive the ONNX graph directly for the shape sweep and the raw
/// logits comparison, without loading a second copy of the ~1.3 GB weights just to reach a
/// crate-private API. `#[doc(hidden)]` keeps it out of the published API surface; Rust has no
/// visibility scoped to "this crate's own tests", so a hidden `pub` item is the closest match
/// without a whole extra `test-internals` feature.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct LayaOutput {
    /// Flattened `[batch.count, batch.marker_count]` raw scores, one per option marker.
    pub logits: Vec<f32>,
    /// Flattened `[batch.count, n_act]` raw action-head scores.
    pub act_logits: Vec<f32>,
}

/// The loaded ONNX session(s) for whichever [`crate::model_artifacts::ModelLayout`] the artifact
/// directory used. `Fused` holds the single graph; `Split` holds the encoder and head graphs that
/// [`LayaEngine::run`] chains together, feeding the encoder's `last_hidden_state` output into the
/// head as its `hidden_states` input.
enum ModelSessions {
    Fused(Session),
    Split { encoder: Session, head: Session },
}

/// Answers typed questions about a piece of state in one forward pass. See the module docs.
pub struct LayaEngine {
    sessions: Mutex<ModelSessions>,
    tokenizer: Box<dyn LayaTokenizer>,
    config: LayaConfig,
    artifacts: ModelArtifacts,
}

impl LayaEngine {
    /// Loads an engine, resolving the artifact directory from `options` (see
    /// [`ModelArtifacts::resolve`] for the full resolution order: an explicit directory,
    /// `LAYA_ONNX_DIR`, `LAYA_ONNX_ROOT`, the local cache, then an opt-in download).
    ///
    /// Returns [`LayaError::CheckpointMismatch`] when the loaded config's encoder contradicts the
    /// requested [`LayaOptions::checkpoint`] — this only happens when the artifact was resolved
    /// through a checkpoint-aware path (`LAYA_ONNX_ROOT`, cache, or download); an explicit
    /// `model_directory` is trusted as-is.
    pub fn create(options: LayaOptions) -> Result<Self> {
        let artifacts = ModelArtifacts::resolve(&options)?;
        let config = LayaConfig::load(&artifacts.config_path)?;

        if let Some(expected_checkpoint) = artifacts.checkpoint {
            validate_encoder(
                config.encoder.as_deref(),
                expected_checkpoint,
                &artifacts.directory,
            )?;
        }

        let sessions = match &artifacts.model {
            ModelPaths::Fused { model_path } => {
                ModelSessions::Fused(build_session(&options, model_path)?)
            }
            ModelPaths::Split {
                encoder_path,
                head_path,
            } => ModelSessions::Split {
                encoder: build_session(&options, encoder_path)?,
                head: build_session(&options, head_path)?,
            },
        };
        let tokenizer = HfTokenizer::from_file_with_config(
            &artifacts.tokenizer_path,
            artifacts.tokenizer_config_path.as_deref(),
        )?;

        Ok(LayaEngine {
            sessions: Mutex::new(sessions),
            tokenizer: Box::new(tokenizer),
            config,
            artifacts,
        })
    }

    /// Loads an engine from an artifact directory containing `model.onnx` directly, skipping
    /// every other resolution step.
    pub fn from_directory(directory: impl AsRef<Path>) -> Result<Self> {
        Self::create(LayaOptions {
            model_directory: Some(directory.as_ref().to_path_buf()),
            ..Default::default()
        })
    }

    /// The loaded checkpoint's configuration.
    pub fn config(&self) -> &LayaConfig {
        &self.config
    }

    /// Where the loaded artifacts came from.
    pub fn artifacts(&self) -> &ModelArtifacts {
        &self.artifacts
    }

    /// Which checkpoint variant was loaded, or `None` when loaded from an explicit directory
    /// where no checkpoint was inferred.
    pub fn checkpoint(&self) -> Option<LayaCheckpoint> {
        self.artifacts.checkpoint
    }

    /// Answer every question in `questions` about `state`.
    ///
    /// `state` accepts anything convertible to a [`serde_json::Value`] — a `&str`/`String` is used
    /// verbatim, anything else is serialized the way Python's `serialize_state` does (see
    /// [`crate::python_json::PythonJson::state`]). For a type that only implements
    /// [`serde::Serialize`] (rather than `Into<Value>`), use [`Self::predict_serialize`] instead.
    ///
    /// Returns [`LayaError::NoMarkers`] if a question's options cannot fit within
    /// `Config::head_max_len`, rather than silently scoring fewer options than it was asked about.
    pub fn predict(&self, state: impl Into<Value>, questions: &QuestionSet) -> Result<LayaResult> {
        self.predict_value(state.into(), questions)
    }

    /// [`Self::predict`], for a state that implements [`serde::Serialize`] but not
    /// `Into<serde_json::Value>` (e.g. a `#[derive(Serialize)]` struct).
    pub fn predict_serialize(
        &self,
        state: &impl Serialize,
        questions: &QuestionSet,
    ) -> Result<LayaResult> {
        let value = serde_json::to_value(state)?;
        self.predict_value(value, questions)
    }

    fn predict_value(&self, state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        if questions.is_empty() {
            return Err(LayaError::InvalidQuestion(
                "no questions to answer".to_string(),
            ));
        }

        let ids: Vec<String> = questions.ids().map(str::to_string).collect();
        let mut items = Vec::with_capacity(ids.len());

        // A conversation-turn list is serialized newest-last (see `PythonJson::state`), so the
        // default right-truncation of an over-long state would silently drop the newest turn.
        // Truncate from the left for lists so the most recent turn survives instead. Ports
        // `Agent._encode_state`'s `truncate_left = isinstance(state, list)` (laya/agent.py, laya
        // 0.3.21) — 0.3.6 never set this, so every state was right-truncated regardless of shape.
        let truncate_left = matches!(state, Value::Array(_));

        // The tokenizer here needs no lock: tokenizers::Tokenizer::encode takes &self.
        for id in &ids {
            let question = &questions[id.as_str()];
            let (seq, markers) = SequenceBuilder::build(
                self.tokenizer.as_ref(),
                &state,
                question,
                self.config.max_len,
                self.config.head_max_len,
                truncate_left,
            )?;

            // A marker dropped for landing past max_len means an option is not in the sequence at
            // all, so its logit would score whatever token happens to sit at position 0.
            if markers.len() != SequenceBuilder::render_options(question).len() {
                return Err(LayaError::NoMarkers {
                    id: id.clone(),
                    head_max_len: self.config.head_max_len,
                });
            }

            items.push(SequenceItem::new(seq, markers, question.question_type()));
        }

        let batch = Collator::collate(&items, self.tokenizer.special().pad_id)?;
        let output = self.run(&batch)?;

        // n_act is the act_logits column count, derived from the flat output length rather than
        // hardcoded so this stays correct if the graph is re-exported with a wider action head.
        let n_act = output.act_logits.len() / batch.count;

        let mut answers = HashMap::with_capacity(ids.len());
        for (row, id) in ids.iter().enumerate() {
            let question = &questions[id.as_str()];
            let k = batch.marker_counts[row];
            let temperature =
                Calibration::resolve_temperature(&self.config, question.question_type(), k);
            let row_start = row * batch.marker_count;
            let row_logits = &output.logits[row_start..row_start + batch.marker_count];
            let p = Calibration::probabilities(row_logits, k, temperature);

            let act_start = row * n_act;
            let act_row = &output.act_logits[act_start..act_start + n_act];
            let action = ActionInfo {
                act_probability: Calibration::round4(Calibration::softmax(act_row)[0]),
            };

            answers.insert(id.clone(), build_answer(question, &p, action));
        }

        Ok(LayaResult::new(
            MODEL_NAME,
            ids,
            answers,
            Usage {
                input_tokens: batch.input_tokens as u32,
                output_tokens: 0,
            },
        ))
    }

    /// One forward pass over a collated batch, returning the raw graph outputs. See the
    /// [`LayaOutput`] docs for why this is `pub` rather than crate-private.
    ///
    /// For the fused layout this is a single `session.run`. For the split layout, the encoder
    /// runs first over `input_ids`/`attention_mask`, and its `last_hidden_state` output is fed
    /// into the head session's `hidden_states` input alongside `marker_pos`/`marker_mask`/
    /// `qtype`/`attention_mask` — one copy of the hidden-state tensor is unavoidable (it must
    /// outlive the encoder's `SessionOutputs` borrow, which is tied to the encoder session's
    /// exclusive borrow, itself released before the head session can be borrowed mutably from the
    /// same lock), but no other tensor is copied more than once.
    #[doc(hidden)]
    pub fn run(&self, batch: &CollatedBatch) -> Result<LayaOutput> {
        // Session::run needs &mut self (see the module docs), so the lock is held for both the
        // run(s) below and the output extraction; tensor construction and calibration in the
        // caller happen outside it.
        let mut sessions = self.sessions.lock().expect("onnx session mutex poisoned");
        match &mut *sessions {
            ModelSessions::Fused(session) => {
                let inputs = build_inputs(session, batch, None)?;
                let outputs = session.run(inputs).map_err(onnx_err)?;
                Ok(LayaOutput {
                    logits: extract_f32(&outputs, "logits")?,
                    act_logits: extract_f32(&outputs, "act_logits")?,
                })
            }
            ModelSessions::Split { encoder, head } => {
                let encoder_inputs = build_inputs(encoder, batch, None)?;
                let encoder_outputs = encoder.run(encoder_inputs).map_err(onnx_err)?;
                let hidden_states = extract_hidden_states(&encoder_outputs, "last_hidden_state")?;
                drop(encoder_outputs);

                let head_inputs = build_inputs(head, batch, Some(hidden_states))?;
                let head_outputs = head.run(head_inputs).map_err(onnx_err)?;
                Ok(LayaOutput {
                    logits: extract_f32(&head_outputs, "logits")?,
                    act_logits: extract_f32(&head_outputs, "act_logits")?,
                })
            }
        }
    }
}

// ── generic, metadata-driven input binding (shared by both layouts) ─────────────────────────

/// The encoder's `last_hidden_state` output, extracted to owned data so it can outlive the
/// encoder's `SessionOutputs` borrow and be fed into the head session as its `hidden_states`
/// input. One copy (the `try_extract_tensor` call in [`extract_hidden_states`]); building the
/// head's input tensor from it below does not copy again.
struct HiddenStates {
    shape: Vec<i64>,
    data: Vec<f32>,
}

impl HiddenStates {
    fn into_dyn_value(self, ty: TensorElementType) -> Result<DynValue> {
        match ty {
            TensorElementType::Float32 => Ok(Tensor::from_array((self.shape, self.data))
                .map_err(onnx_err)?
                .into_dyn()),
            other => Err(LayaError::Onnx(format!(
                "the head model declares 'hidden_states' as {other:?}, but the encoder produces \
                 Float32"
            ))),
        }
    }
}

fn extract_hidden_states(
    outputs: &ort::session::SessionOutputs,
    name: &str,
) -> Result<HiddenStates> {
    let value = outputs
        .get(name)
        .ok_or_else(|| LayaError::Onnx(format!("the encoder produced no '{name}' output")))?;
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(onnx_err)?;
    Ok(HiddenStates {
        shape: shape.to_vec(),
        data: data.to_vec(),
    })
}

/// Builds an integer-sourced tensor (`input_ids`, `attention_mask`, `marker_pos`, `qtype`),
/// casting `data` to whichever element type the session declared.
fn int_tensor(shape: Vec<i64>, data: &[i64], ty: TensorElementType) -> Result<DynValue> {
    match ty {
        TensorElementType::Int64 => Ok(Tensor::from_array((shape, data.to_vec()))
            .map_err(onnx_err)?
            .into_dyn()),
        TensorElementType::Int32 => {
            let data: Vec<i32> = data.iter().map(|&v| v as i32).collect();
            Ok(Tensor::from_array((shape, data))
                .map_err(onnx_err)?
                .into_dyn())
        }
        TensorElementType::Float32 => {
            let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            Ok(Tensor::from_array((shape, data))
                .map_err(onnx_err)?
                .into_dyn())
        }
        TensorElementType::Bool => {
            let data: Vec<bool> = data.iter().map(|&v| v != 0).collect();
            Ok(Tensor::from_array((shape, data))
                .map_err(onnx_err)?
                .into_dyn())
        }
        other => Err(LayaError::Onnx(format!(
            "unsupported ONNX element type {other:?} for an integer-valued input"
        ))),
    }
}

/// Builds a boolean-sourced tensor (`marker_mask`), casting to whichever element type the session
/// declared.
fn bool_tensor(shape: Vec<i64>, data: &[bool], ty: TensorElementType) -> Result<DynValue> {
    match ty {
        TensorElementType::Bool => Ok(Tensor::from_array((shape, data.to_vec()))
            .map_err(onnx_err)?
            .into_dyn()),
        TensorElementType::Int64 => {
            let data: Vec<i64> = data.iter().map(|&b| b as i64).collect();
            Ok(Tensor::from_array((shape, data))
                .map_err(onnx_err)?
                .into_dyn())
        }
        TensorElementType::Int32 => {
            let data: Vec<i32> = data.iter().map(|&b| b as i32).collect();
            Ok(Tensor::from_array((shape, data))
                .map_err(onnx_err)?
                .into_dyn())
        }
        TensorElementType::Float32 => {
            let data: Vec<f32> = data.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            Ok(Tensor::from_array((shape, data))
                .map_err(onnx_err)?
                .into_dyn())
        }
        other => Err(LayaError::Onnx(format!(
            "unsupported ONNX element type {other:?} for a boolean-valued input"
        ))),
    }
}

/// Produces the value for one declared input by name, or `None` if `name` is not one this engine
/// knows how to provide (which [`build_inputs`] turns into a clear error rather than silently
/// skipping it). `qtype`'s shape follows the session's own declared rank: `[batch, 1]` when the
/// graph declares it rank 2 (the split layout's head, whose exporter's dummy input is `[[0]] *
/// batch`), `[batch]` otherwise (the fused layout) — decided from metadata, not hardcoded per
/// layout, so either graph shape is honored even if a future export changes it.
fn value_for_input(
    name: &str,
    ty: TensorElementType,
    rank: usize,
    batch: &CollatedBatch,
    hidden_states: &mut Option<HiddenStates>,
) -> Result<Option<DynValue>> {
    let count = batch.count as i64;
    match name {
        "input_ids" => Ok(Some(int_tensor(
            vec![count, batch.sequence_length as i64],
            &batch.input_ids,
            ty,
        )?)),
        "attention_mask" => Ok(Some(int_tensor(
            vec![count, batch.sequence_length as i64],
            &batch.attention_mask,
            ty,
        )?)),
        "marker_pos" => Ok(Some(int_tensor(
            vec![count, batch.marker_count as i64],
            &batch.marker_pos,
            ty,
        )?)),
        "marker_mask" => Ok(Some(bool_tensor(
            vec![count, batch.marker_count as i64],
            &batch.marker_mask,
            ty,
        )?)),
        "qtype" => {
            let shape = if rank >= 2 {
                vec![count, 1]
            } else {
                vec![count]
            };
            Ok(Some(int_tensor(shape, &batch.qtype, ty)?))
        }
        "hidden_states" => {
            let hidden = hidden_states.take().ok_or_else(|| {
                LayaError::Onnx(
                    "the head model declares a 'hidden_states' input, but no encoder output was \
                     available to provide it"
                        .to_string(),
                )
            })?;
            Ok(Some(hidden.into_dyn_value(ty)?))
        }
        _ => Ok(None),
    }
}

/// Binds only the inputs `session` actually declares (per item 3 of the design: never a fixed,
/// hardcoded list), reading each one's declared dtype and rank from session metadata so the
/// tensor built for it matches exactly what the graph expects. `hidden_states` is consumed (via
/// [`Option::take`]) the one time a declared input is actually named `hidden_states`, so the
/// extracted encoder output is moved into its tensor rather than copied again.
fn build_inputs(
    session: &Session,
    batch: &CollatedBatch,
    hidden_states: Option<HiddenStates>,
) -> Result<Vec<(String, DynValue)>> {
    let mut hidden_states = hidden_states;
    let mut inputs = Vec::with_capacity(session.inputs().len());
    for outlet in session.inputs() {
        let name = outlet.name();
        let (ty, rank) = match outlet.dtype() {
            ValueType::Tensor { ty, shape, .. } => (*ty, shape.len()),
            other => {
                return Err(LayaError::Onnx(format!(
                    "input '{name}' has unsupported (non-tensor) type {other:?}"
                )));
            }
        };
        match value_for_input(name, ty, rank, batch, &mut hidden_states)? {
            Some(value) => inputs.push((name.to_string(), value)),
            None => {
                return Err(LayaError::Onnx(format!(
                    "the model declares an input '{name}' that this engine does not know how to \
                     provide"
                )));
            }
        }
    }
    Ok(inputs)
}

// SAFETY-free: every field is itself Send + Sync (Mutex<ModelSessions> — Session is Send + Sync
// and so is the enum wrapping one or two of them —, Box<dyn LayaTokenizer> whose trait requires
// Send + Sync, and plain owned data in LayaConfig / ModelArtifacts), so this is
// the ordinary auto-derived impl, not an unsafe opt-in. Asserted here so a future field addition
// that breaks it fails to compile at this obvious spot rather than at some faraway `Arc<LayaEngine>`
// call site.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LayaEngine>();
};

fn build_answer(question: &Question, p: &[f64], action: ActionInfo) -> Answer {
    let k = p.len();
    let confidence = Calibration::round4(Calibration::confidence_from_probs(p, k));
    // Calibrated on every question type alike (`max(p)`), unlike `confidence` above. Ports
    // `answer_confidence` from `laya/common.py` (laya 0.3.21); see `Answer::answer_confidence`.
    let answer_confidence = Calibration::round4(Calibration::answer_confidence(p, k));

    match question {
        Question::Choice(choice) => {
            let labels: Vec<&str> = choice.labels().collect();
            // Python zips labels against probabilities, so a mismatch would silently drop the
            // tail rather than misalign; take the shorter for the same reason.
            let n = k.min(labels.len());
            let mut probabilities = IndexMap::with_capacity(n);
            for (label, &prob) in labels.iter().zip(p.iter()).take(n) {
                probabilities.insert((*label).to_string(), Calibration::round4(prob));
            }
            let chosen = labels
                .get(Calibration::arg_max(p))
                .copied()
                .unwrap_or_default()
                .to_string();
            Answer::Choice(ChoiceAnswer::new(
                chosen,
                probabilities,
                confidence,
                answer_confidence,
                action,
            ))
        }

        Question::Score(score) => {
            let probabilities: Vec<f64> = p.iter().copied().map(Calibration::round4).collect();
            let expected = Calibration::round4(Calibration::expected_score(p));
            Answer::Score(ScoreAnswer::new(
                expected,
                score.levels().to_vec(),
                probabilities,
                confidence,
                answer_confidence,
                action,
            ))
        }

        Question::Noul(_) => {
            // Noul reports the true-branch probability, and its confidence is the distance from a
            // coin flip rather than the entropy measure the other two types use. Over two options
            // max(p_true, 1 - p_true) equals max(p), so `answer_confidence` agrees with
            // `confidence` here even though the two differ for choice and score.
            let p_true = p[1];
            Answer::Noul(NoulAnswer::new(
                Calibration::round4(p_true),
                Calibration::round4(p_true.max(1.0 - p_true)),
                answer_confidence,
                action,
            ))
        }
    }
}

fn validate_encoder(
    actual_encoder: Option<&str>,
    checkpoint: LayaCheckpoint,
    dir: &Path,
) -> Result<()> {
    // Multilingual uses mmBERT; English and TypedDecisions both use ModernBERT-large. This can
    // only detect a family mismatch (mmBERT vs. ModernBERT), not distinguish English from
    // TypedDecisions, since both share the same encoder string.
    let expect_modern_bert = checkpoint != LayaCheckpoint::Multilingual;
    let got_modern_bert = actual_encoder
        .map(|e| e.to_ascii_lowercase().contains("modernbert"))
        .unwrap_or(false);

    if expect_modern_bert != got_modern_bert {
        let expected = if checkpoint == LayaCheckpoint::Multilingual {
            "jhu-clsp/mmBERT-base"
        } else {
            "answerdotai/ModernBERT-large"
        };
        return Err(LayaError::CheckpointMismatch(format!(
            "Checkpoint '{checkpoint:?}' expects encoder '{expected}' but the artifact at '{}' \
             declares encoder '{}'. The artifact directory may contain the wrong checkpoint.",
            dir.display(),
            actual_encoder.unwrap_or("(null)")
        )));
    }
    Ok(())
}

fn build_session(options: &LayaOptions, model_path: &Path) -> Result<Session> {
    let mut builder = Session::builder().map_err(onnx_err)?;
    if let Some(intra) = options.intra_op_threads {
        builder = builder
            .with_intra_threads(intra as usize)
            .map_err(onnx_err)?;
    }
    if let Some(inter) = options.inter_op_threads {
        builder = builder
            .with_inter_threads(inter as usize)
            .map_err(onnx_err)?;
    }
    let mut builder = apply_execution_provider(builder, options.execution_provider)?;

    // The path must point at model.onnx inside its own directory: this checkpoint is in ONNX
    // external-data format and the runtime resolves the ~1.29 GB model.onnx.data sidecar relative
    // to the graph file.
    builder.commit_from_file(model_path).map_err(onnx_err)
}

fn apply_execution_provider(
    builder: SessionBuilder,
    provider: LayaExecutionProvider,
) -> Result<SessionBuilder> {
    match provider {
        LayaExecutionProvider::Cpu => Ok(builder),

        LayaExecutionProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                builder
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .map_err(onnx_err)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = builder;
                Err(LayaError::Onnx(
                    "LayaExecutionProvider::Cuda was requested, but this crate was built \
                     without the `cuda` feature."
                        .to_string(),
                ))
            }
        }

        LayaExecutionProvider::DirectMl => {
            #[cfg(feature = "directml")]
            {
                builder
                    .with_execution_providers([ort::ep::DirectML::default().build()])
                    .map_err(onnx_err)
            }
            #[cfg(not(feature = "directml"))]
            {
                let _ = builder;
                Err(LayaError::Onnx(
                    "LayaExecutionProvider::DirectMl was requested, but this crate was built \
                     without the `directml` feature."
                        .to_string(),
                ))
            }
        }
    }
}

fn extract_f32(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Vec<f32>> {
    let value = outputs
        .get(name)
        .ok_or_else(|| LayaError::Onnx(format!("the model produced no '{name}' output")))?;
    let (_, data) = value.try_extract_tensor::<f32>().map_err(onnx_err)?;
    Ok(data.to_vec())
}

fn onnx_err(e: impl std::fmt::Display) -> LayaError {
    LayaError::Onnx(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A tiny two-row batch, just enough shape/data to exercise `value_for_input` without a real
    /// ONNX session.
    fn tiny_batch() -> CollatedBatch {
        CollatedBatch {
            count: 2,
            sequence_length: 3,
            marker_count: 2,
            input_ids: vec![1, 2, 3, 4, 5, 6],
            attention_mask: vec![1, 1, 1, 1, 1, 0],
            marker_pos: vec![0, 1, 0, 1],
            marker_mask: vec![true, true, true, false],
            qtype: vec![7, 9],
            marker_counts: vec![2, 1],
            input_tokens: 5,
        }
    }

    fn tensor_shape(value: &DynValue) -> Vec<i64> {
        value
            .try_extract_tensor::<i64>()
            .map(|(shape, _)| shape.to_vec())
            .or_else(|_| {
                value
                    .try_extract_tensor::<i32>()
                    .map(|(shape, _)| shape.to_vec())
            })
            .expect("qtype should extract as an integer tensor")
    }

    #[test]
    fn qtype_rank_one_declared_yields_a_one_dimensional_shape() {
        let batch = tiny_batch();
        let mut hidden = None;
        let value = value_for_input("qtype", TensorElementType::Int64, 1, &batch, &mut hidden)
            .expect("qtype should be handled")
            .expect("qtype should produce a value");
        assert_eq!(tensor_shape(&value), vec![2]);
    }

    #[test]
    fn qtype_rank_two_declared_yields_a_batch_by_one_shape() {
        // The split layout's head declares qtype as rank 2 ([B, 1]), matching the exporter's
        // dummy input `[[0]] * batch`.
        let batch = tiny_batch();
        let mut hidden = None;
        let value = value_for_input("qtype", TensorElementType::Int64, 2, &batch, &mut hidden)
            .expect("qtype should be handled")
            .expect("qtype should produce a value");
        assert_eq!(tensor_shape(&value), vec![2, 1]);
    }

    #[test]
    fn qtype_rank_selection_is_independent_of_element_type() {
        // Rank drives the shape regardless of which integer dtype the graph declares.
        let batch = tiny_batch();
        let mut hidden = None;
        let value = value_for_input("qtype", TensorElementType::Int32, 2, &batch, &mut hidden)
            .expect("qtype should be handled")
            .expect("qtype should produce a value");
        assert_eq!(tensor_shape(&value), vec![2, 1]);
    }

    #[test]
    fn hidden_states_input_is_taken_exactly_once() {
        let batch = tiny_batch();
        let mut hidden = Some(HiddenStates {
            shape: vec![2, 3, 4],
            data: vec![0.0; 24],
        });
        let value = value_for_input(
            "hidden_states",
            TensorElementType::Float32,
            3,
            &batch,
            &mut hidden,
        )
        .expect("hidden_states should be handled")
        .expect("hidden_states should produce a value");
        assert!(
            hidden.is_none(),
            "hidden_states should be consumed by take()"
        );
        let (shape, _) = value
            .try_extract_tensor::<f32>()
            .expect("should extract as f32");
        assert_eq!(shape.to_vec(), vec![2, 3, 4]);
    }

    #[test]
    fn hidden_states_input_without_an_encoder_output_is_a_clear_error() {
        let batch = tiny_batch();
        let mut hidden = None;
        let err = value_for_input(
            "hidden_states",
            TensorElementType::Float32,
            3,
            &batch,
            &mut hidden,
        )
        .expect_err("must fail: no encoder output was ever provided");
        assert!(matches!(err, LayaError::Onnx(_)));
    }

    #[test]
    fn build_answer_reports_answer_confidence_alongside_confidence_for_choice() {
        // A skewed three-option distribution: answer_confidence is just the top probability,
        // while confidence is the normalized-entropy measure, so the two must differ here.
        let question = Question::choice(
            "h",
            [("a", Value::Null), ("b", Value::Null), ("c", Value::Null)],
        )
        .unwrap();
        let p = [0.7, 0.2, 0.1];
        let action = ActionInfo {
            act_probability: 1.0,
        };
        let answer = build_answer(&question, &p, action);

        assert_eq!(0.7, answer.answer_confidence());
        assert_ne!(answer.answer_confidence(), answer.confidence());
    }

    #[test]
    fn build_answer_answer_confidence_agrees_with_confidence_for_noul() {
        // Over two options max(p_true, 1 - p_true) equals max(p), so the two quantities coincide
        // for noul even though they differ for choice and score.
        let question = Question::noul("h", Value::Null, Value::Null);
        let p = [0.3, 0.7];
        let action = ActionInfo {
            act_probability: 1.0,
        };
        let answer = build_answer(&question, &p, action);

        assert_eq!(answer.answer_confidence(), answer.confidence());
        assert_eq!(0.7, answer.answer_confidence());
    }

    #[test]
    fn truncate_left_is_derived_from_the_state_shape() {
        // Ports `Agent._encode_state`'s `truncate_left = isinstance(state, list)` (laya 0.3.21):
        // a JSON array state truncates from the left, anything else from the right.
        assert!(matches!(json!([1, 2, 3]), Value::Array(_)));
        assert!(!matches!(json!({"a": 1}), Value::Array(_)));
        assert!(!matches!(json!("state text"), Value::Array(_)));
        assert!(!matches!(Value::Null, Value::Array(_)));
    }

    #[test]
    fn an_unknown_input_name_is_reported_as_none_not_an_error() {
        // `build_inputs` is the one that turns a `None` into a clear error; `value_for_input`
        // itself just reports "I don't know this name".
        let batch = tiny_batch();
        let mut hidden = None;
        let result = value_for_input(
            "some_future_input",
            TensorElementType::Int64,
            1,
            &batch,
            &mut hidden,
        )
        .expect("unknown names are not themselves an error");
        assert!(result.is_none());
    }
}
