//! The plan behind `LayaEngine::predict_batch`: how many states go through one forward pass, when
//! states are reordered to pad less, and how results come back in input order. Ports
//! `ONNXAgent._infer_batch` (`laya/onnx_agent.py`), whose `batch_size` and `sort_by_length`
//! handling is the same contract as `Agent.predict_batch`.
//!
//! The driver here is generic over what an encoded state is, so the chunking, the sort windows
//! and the order restoration are testable without a model: the engine supplies the tokenizing and
//! the forward pass.
//!
//! Python's per-call hooks, `lang`, `max_len` and `head_max_len` are not part of this port yet.
//! [`BatchOptions`] is `#[non_exhaustive]` so they can be added as fields without a signature
//! change.

use crate::confidence::MinConfidence;
use crate::error::{LayaError, Result};
use std::ops::Range;

/// How many batches of states the `sort_by_length` lookahead spans. Bounds the tokenized
/// lookahead independently of the input size, as Python does.
const SORT_WINDOW_BATCHES: usize = 8;

/// Options for `LayaEngine::predict_batch` and `LayaRouter::predict_batch`. Every field is
/// optional; `Default` gives Python's defaults. Build one with the chained setters.
///
/// Per-call hooks, `lang`, `max_len` and `head_max_len`, which Python's `predict_batch` also
/// accepts, are not part of this port yet.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct BatchOptions {
    /// Most states per forward pass. `None` or `0` sends them all in one pass. Python treats a
    /// `batch_size` of `None`, `0` or any negative number as unset; a negative size cannot be
    /// written as a `usize`, so `0` stands for all of them here. Set it to bound peak memory when
    /// batching many or long states.
    pub batch_size: Option<usize>,
    /// Group similarly sized encoded states within windows of eight batches, so each batch pads
    /// to a shorter maximum. Takes effect only with a `batch_size` greater than one and smaller
    /// than the number of states; otherwise it does nothing. Results keep input order. Changing
    /// the batch shapes can slightly change floating-point predictions near decision thresholds.
    pub sort_by_length: bool,
    /// Opt-in abstention threshold on `answer_confidence` (#361), applied batch-wide to every
    /// answer of every result; it plays no part in how states are grouped or batched. See
    /// [`crate::PredictOptions::min_confidence`] for what it reports.
    pub min_confidence: Option<MinConfidence>,
}

impl BatchOptions {
    /// Python's defaults: one pass for every state, input order.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`Self::batch_size`].
    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set [`Self::sort_by_length`].
    #[must_use]
    pub fn sort_by_length(mut self, sort_by_length: bool) -> Self {
        self.sort_by_length = sort_by_length;
        self
    }

    /// Set [`Self::min_confidence`]. Build the value with [`MinConfidence::scalar`] or
    /// [`MinConfidence::map`].
    #[must_use]
    pub fn min_confidence(mut self, min_confidence: MinConfidence) -> Self {
        self.min_confidence = Some(min_confidence);
        self
    }

    /// States per forward pass for `count` states: `batch_size` when it is positive, else all.
    fn chunk(&self, count: usize) -> usize {
        match self.batch_size {
            Some(size) if size > 0 => size,
            _ => count,
        }
    }

    /// Whether `sort_by_length` reorders `count` states: `1 < chunk < count`.
    pub(crate) fn reorders(&self, count: usize) -> bool {
        let chunk = self.chunk(count);
        self.sort_by_length && 1 < chunk && chunk < count
    }
}

/// Encodes, orders, chunks and runs `count` states, returning one result per state in input order.
///
/// * `encode` turns a range of input indices into their encoded states (all of a window at once,
///   so a tokenizing error surfaces before any forward pass of that window, as in Python);
/// * `row_len` is a state's longest encoded question row, the sort key;
/// * `infer` runs one chunk of encoded states and returns one result per state of the chunk.
///
/// Without a reordering `sort_by_length` the windows are the chunks themselves. With one, each
/// window spans eight chunks, its states are stably sorted by `row_len`, and the sorted states are
/// cut into chunks.
pub(crate) fn run<E, R>(
    count: usize,
    options: &BatchOptions,
    mut encode: impl FnMut(Range<usize>) -> Result<Vec<E>>,
    row_len: impl Fn(&E) -> usize,
    mut infer: impl FnMut(Vec<E>) -> Result<Vec<R>>,
) -> Result<Vec<R>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let chunk = options.chunk(count);
    let reorder = options.reorders(count);
    let window = if reorder {
        chunk.saturating_mul(SORT_WINDOW_BATCHES)
    } else {
        chunk
    };

    let mut results = Vec::with_capacity(count);
    for start in (0..count).step_by(window) {
        let end = start.saturating_add(window).min(count);
        let mut slots: Vec<Option<E>> = encode(start..end)?.into_iter().map(Some).collect();
        let mut order: Vec<usize> = (0..slots.len()).collect();
        if reorder {
            order.sort_by_key(|&i| slots[i].as_ref().map_or(0, &row_len));
        }
        let mut window_results: Vec<Option<R>> = Vec::with_capacity(slots.len());
        window_results.resize_with(slots.len(), || None);
        for indices in order.chunks(chunk) {
            let batch: Vec<E> = indices.iter().filter_map(|&i| slots[i].take()).collect();
            let outputs = infer(batch)?;
            if outputs.len() != indices.len() {
                return Err(internal(format!(
                    "a forward pass returned {} results for {} states",
                    outputs.len(),
                    indices.len()
                )));
            }
            for (&index, output) in indices.iter().zip(outputs) {
                window_results[index] = Some(output);
            }
        }
        for result in window_results {
            results.push(result.ok_or_else(|| internal("a state was left unanswered".into()))?);
        }
    }
    Ok(results)
}

fn internal(message: String) -> LayaError {
    LayaError::Onnx(format!("internal error: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An encoded state: its input index and its longest row.
    type Fake = (usize, usize);

    /// Runs the driver over states whose row lengths are `lens`, recording each forward pass as
    /// the input indices it carried. The result for a state is its own index.
    fn drive(lens: &[usize], options: &BatchOptions) -> (Vec<usize>, Vec<Vec<usize>>) {
        let mut passes = Vec::new();
        let results = run(
            lens.len(),
            options,
            |range| Ok(range.map(|i| (i, lens[i])).collect::<Vec<Fake>>()),
            |state| state.1,
            |batch| {
                passes.push(batch.iter().map(|s| s.0).collect::<Vec<_>>());
                Ok(batch.into_iter().map(|s| s.0).collect())
            },
        )
        .unwrap();
        (results, passes)
    }

    fn sized(batch_size: usize, sort: bool) -> BatchOptions {
        BatchOptions::new()
            .batch_size(batch_size)
            .sort_by_length(sort)
    }

    #[test]
    fn no_batch_size_sends_every_state_in_one_pass() {
        let (results, passes) = drive(&[5, 4, 3], &BatchOptions::new());
        assert_eq!(results, [0, 1, 2]);
        assert_eq!(passes, [vec![0, 1, 2]]);
    }

    #[test]
    fn zero_batch_size_is_unset_as_python_treats_it() {
        let (_, passes) = drive(&[5, 4, 3], &sized(0, true));
        assert_eq!(passes, [vec![0, 1, 2]]);
    }

    #[test]
    fn batch_size_cuts_chunks_in_input_order() {
        let (results, passes) = drive(&[1, 2, 3, 4, 5], &sized(2, false));
        assert_eq!(results, [0, 1, 2, 3, 4]);
        assert_eq!(passes, [vec![0, 1], vec![2, 3], vec![4]]);
    }

    #[test]
    fn sorting_groups_similar_lengths_and_restores_input_order() {
        let (results, passes) = drive(&[9, 1, 8, 2, 7], &sized(2, true));
        assert_eq!(results, [0, 1, 2, 3, 4]);
        assert_eq!(passes, [vec![1, 3], vec![4, 2], vec![0]]);
    }

    #[test]
    fn sorting_is_stable_for_equal_lengths() {
        let (_, passes) = drive(&[3, 3, 3, 1, 3], &sized(2, true));
        assert_eq!(passes, [vec![3, 0], vec![1, 2], vec![4]]);
    }

    #[test]
    fn sorting_needs_a_batch_size_above_one_and_below_the_state_count() {
        let lens = [9, 1, 8, 2];
        // No batch size: one pass, unsorted.
        assert_eq!(drive(&lens, &sized(0, true)).1, [vec![0, 1, 2, 3]]);
        // A batch of one: nothing to group.
        assert_eq!(
            drive(&lens, &sized(1, true)).1,
            [vec![0], vec![1], vec![2], vec![3]]
        );
        // A batch as large as the input, or larger: one pass, unsorted.
        assert_eq!(drive(&lens, &sized(4, true)).1, [vec![0, 1, 2, 3]]);
        assert_eq!(drive(&lens, &sized(10, true)).1, [vec![0, 1, 2, 3]]);
        // Just under the count sorts.
        assert_eq!(drive(&lens, &sized(3, true)).1, [vec![1, 3, 2], vec![0]]);
    }

    #[test]
    fn reorders_follows_the_python_gate() {
        assert!(sized(2, true).reorders(3));
        assert!(!sized(2, false).reorders(3));
        assert!(!sized(2, true).reorders(2));
        assert!(!sized(1, true).reorders(3));
        assert!(!BatchOptions::new().sort_by_length(true).reorders(3));
    }

    #[test]
    fn sort_windows_span_eight_batches() {
        // Batch size 2, so a window is 16 states. Descending lengths: the first window sorts only
        // among its own 16 states, the next one among the remaining 4.
        let lens: Vec<usize> = (0..20).rev().collect();
        let (results, passes) = drive(&lens, &sized(2, true));
        assert_eq!(results, (0..20).collect::<Vec<_>>());
        assert_eq!(passes[0], [15, 14]);
        assert_eq!(passes[7], [1, 0]);
        assert_eq!(passes[8], [19, 18]);
        assert_eq!(passes[9], [17, 16]);
        assert_eq!(passes.len(), 10);
    }

    #[test]
    fn unsorted_windows_are_the_chunks() {
        let (_, passes) = drive(&[3, 2, 1], &sized(2, false));
        assert_eq!(passes, [vec![0, 1], vec![2]]);
    }

    #[test]
    fn no_states_run_nothing() {
        let (results, passes) = drive(&[], &sized(2, true));
        assert!(results.is_empty());
        assert!(passes.is_empty());
    }

    #[test]
    fn an_encode_error_stops_before_any_pass_of_its_window() {
        let mut passes = 0;
        let outcome: Result<Vec<usize>> = run(
            3,
            &BatchOptions::new(),
            |_| Err(LayaError::InvalidQuestion("boom".into())),
            |_: &usize| 0,
            |batch: Vec<usize>| {
                passes += 1;
                Ok(batch)
            },
        );
        assert!(matches!(outcome, Err(LayaError::InvalidQuestion(_))));
        assert_eq!(passes, 0);
    }

    #[test]
    fn a_pass_returning_the_wrong_count_is_an_error_not_a_panic() {
        let outcome: Result<Vec<usize>> = run(
            2,
            &BatchOptions::new(),
            |range| Ok(range.collect::<Vec<usize>>()),
            |_: &usize| 0,
            |_| Ok(vec![0usize]),
        );
        assert!(outcome.is_err());
    }

    #[test]
    fn options_take_chained_setters() {
        let options = BatchOptions::new().batch_size(4).sort_by_length(true);
        assert_eq!(options.batch_size, Some(4));
        assert!(options.sort_by_length);
        assert_eq!(BatchOptions::default(), BatchOptions::new());
    }

    // ------------------------------------------------------------------ recorded session runs

    /// Where the per-checkpoint goldens live: `LAYA_GOLDEN_DIR`, else `tests/golden`.
    fn golden_root() -> std::path::PathBuf {
        match std::env::var("LAYA_GOLDEN_DIR") {
            Ok(dir) if !dir.trim().is_empty() => dir.into(),
            _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden"),
        }
    }

    /// The fused artifact directory for a checkpoint, found as the integration tests find it:
    /// `LAYA_ONNX_DIR` (multilingual only), `LAYA_ONNX_ROOT`, then `onnx/<checkpoint>` up the tree.
    fn artifact_dir(checkpoint: crate::LayaCheckpoint) -> Option<std::path::PathBuf> {
        use std::path::{Path, PathBuf};
        let complete = |dir: &Path| {
            dir.join("rl_agent_config.json").is_file()
                && dir.join("model.onnx").is_file()
                && dir.join("model.onnx.data").is_file()
                && dir.join("tokenizer").join("tokenizer.json").is_file()
        };
        let name = checkpoint.subdir();
        if checkpoint == crate::LayaCheckpoint::Multilingual
            && let Ok(dir) = std::env::var("LAYA_ONNX_DIR")
            && !dir.trim().is_empty()
        {
            let dir = PathBuf::from(dir);
            return complete(&dir).then_some(dir);
        }
        if let Ok(root) = std::env::var("LAYA_ONNX_ROOT")
            && !root.trim().is_empty()
        {
            let dir = PathBuf::from(root).join(name);
            if complete(&dir) {
                return Some(dir);
            }
        }
        let mut dir = Some(Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf());
        while let Some(d) = dir {
            let candidate = d.join("onnx").join(name);
            if complete(&candidate) {
                return Some(candidate);
            }
            dir = d.parent().map(Path::to_path_buf);
        }
        None
    }

    /// The questions of a recorded case, from its `questions` definition (choice, score, noul).
    fn recorded_questions(definition: &serde_json::Value) -> crate::QuestionSet {
        use crate::{Question, QuestionSet};
        use serde_json::Value;
        let mut set = QuestionSet::new();
        for (id, def) in definition.as_object().expect("questions object") {
            let instructions = def["instructions"].clone();
            let criteria = def.get("criteria");
            let question = match def["type"].as_str().expect("question type") {
                "choice" => {
                    let options: Vec<(String, Value)> = match criteria.expect("criteria") {
                        Value::Array(labels) => labels
                            .iter()
                            .map(|l| (l.as_str().expect("label").to_string(), Value::Null))
                            .collect(),
                        Value::Object(map) => {
                            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                        }
                        other => panic!("{id}: unusable criteria {other}"),
                    };
                    Question::choice(instructions, options).expect("choice question")
                }
                "score" => {
                    let levels = criteria.and_then(Value::as_array).expect("levels").clone();
                    Question::score(instructions, levels).expect("score question")
                }
                "noul" => {
                    let side = |key: &str| {
                        criteria
                            .and_then(|c| c.get(key))
                            .cloned()
                            .unwrap_or(Value::Null)
                    };
                    Question::noul(instructions, side("false"), side("true"))
                }
                other => panic!("{id}: unknown question type {other}"),
            };
            set.insert(id.clone(), question).expect("unique ids");
        }
        set
    }

    fn recorded_options(options: &serde_json::Value) -> BatchOptions {
        BatchOptions {
            // Python treats a batch_size that is zero or negative as unset.
            batch_size: options
                .get("batch_size")
                .and_then(serde_json::Value::as_i64)
                .and_then(|size| usize::try_from(size).ok())
                .filter(|&size| size > 0),
            sort_by_length: options
                .get("sort_by_length")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            min_confidence: None,
        }
    }

    /// Python's own session-run shapes against this port's chunk planning, over this port's own
    /// token counts: each state's width (its longest question row) must equal Python's
    /// `state_widths`, and the planned runs, `[rows, width]` per forward pass, must be the ones
    /// Python made. Needs the tokenizer and the batch goldens, not a forward pass; skips like the
    /// integration tests when either is missing.
    #[test]
    fn plans_the_recorded_session_runs() {
        use crate::laya_config::LayaConfig;
        use crate::model_artifacts::ModelArtifacts;
        use crate::sequence_builder::SequenceBuilder;
        use crate::tokenization::HfTokenizer;
        use serde_json::Value;

        let mut checked_any = false;
        for checkpoint in [
            crate::LayaCheckpoint::Multilingual,
            crate::LayaCheckpoint::English,
            crate::LayaCheckpoint::TypedDecisions,
        ] {
            let Some(dir) = artifact_dir(checkpoint) else {
                continue;
            };
            let golden_dir = golden_root().join(checkpoint.subdir());
            let Ok(index) = std::fs::read_to_string(golden_dir.join("index.json")) else {
                eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
                continue;
            };
            let index: Value = serde_json::from_str(&index).expect("index.json");
            let artifacts =
                ModelArtifacts::from_directory(&dir, Some(checkpoint)).expect("artifacts");
            let config = LayaConfig::load(&artifacts.config_path).expect("config");
            let tokenizer = HfTokenizer::from_file_with_config(
                &artifacts.tokenizer_path,
                artifacts.tokenizer_config_path.as_deref(),
            )
            .expect("tokenizer");

            for entry in index.as_array().expect("index array") {
                let is_batch = entry["kind"].as_str() == Some("batch");
                if !is_batch || entry["expects_error"].as_bool() == Some(true) {
                    continue;
                }
                let text =
                    std::fs::read_to_string(golden_dir.join(entry["file"].as_str().unwrap()))
                        .expect("case file");
                let root: Value = serde_json::from_str(&text).expect("case json");
                let case = format!(
                    "{}/{}",
                    checkpoint.subdir(),
                    entry["name"].as_str().unwrap()
                );
                let states = root["states"].as_array().expect("states");
                let questions = recorded_questions(&root["questions"]);
                if states.is_empty() || questions.is_empty() {
                    continue;
                }
                checked_any = true;

                let widths: Vec<usize> = states
                    .iter()
                    .map(|state| {
                        questions
                            .iter()
                            .map(|(_, question)| {
                                SequenceBuilder::build_with_stats(
                                    &tokenizer,
                                    state,
                                    question,
                                    config.max_len,
                                    config.head_max_len,
                                    matches!(state, Value::Array(_)),
                                )
                                .expect("sequence")
                                .0
                                .len()
                            })
                            .max()
                            .unwrap_or(0)
                    })
                    .collect();
                let recorded_widths: Vec<usize> = root["state_widths"]
                    .as_array()
                    .expect("state_widths")
                    .iter()
                    .map(|w| w.as_u64().expect("width") as usize)
                    .collect();
                assert_eq!(recorded_widths, widths, "{case}: state widths");

                let mut runs: Vec<[usize; 2]> = Vec::new();
                run(
                    states.len(),
                    &recorded_options(&root["options"]),
                    |range| Ok(range.map(|i| widths[i]).collect::<Vec<usize>>()),
                    |width: &usize| *width,
                    |chunk: Vec<usize>| {
                        runs.push([
                            chunk.len() * questions.len(),
                            chunk.iter().copied().max().unwrap_or(0),
                        ]);
                        Ok(chunk)
                    },
                )
                .expect("plan");
                let recorded_runs: Vec<[usize; 2]> = root["runs"]
                    .as_array()
                    .expect("runs")
                    .iter()
                    .map(|r| {
                        let r = r.as_array().expect("run");
                        [
                            r[0].as_u64().unwrap() as usize,
                            r[1].as_u64().unwrap() as usize,
                        ]
                    })
                    .collect();
                assert_eq!(recorded_runs, runs, "{case}: planned session runs");
            }
        }
        if !checked_any {
            eprintln!("skip: no ONNX tokenizer + batch golden data found for any checkpoint");
        }
    }
}
