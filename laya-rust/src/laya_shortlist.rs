//! Opt-in embedding shortlist for high-cardinality choice questions. Ports `laya/shortlist.py`.
//!
//! Choice options share one `head_max_len` token budget, so a large label set leaves only a few
//! tokens per label. [`LayaShortlist::predict`] embeds the state and each option with a
//! caller-supplied `embed_fn`, keeps the top `k`, and runs a single predict on that reduced
//! criteria set. A non-choice question is forwarded unchanged, and a choice whose label count is
//! already `<= k` is forwarded unchanged too — `embed_fn` is never called for it.
//!
//! `embed_fn_from_agent` (Python: mean-pools the checkpoint encoder already loaded on an `Agent`)
//! is deliberately not ported: this port only accepts a caller-supplied embedding function (a
//! real deployment should bring a dedicated bi-encoder), matching the plan for this port.
//!
//! ## Why `embed_fn` is a generic `Fn` bound rather than a boxed trait object
//!
//! `shortlist_choice`/`predict` take `embed_fn: F where F: Fn(&[String]) -> Result<Vec<Vec<f64>>>`
//! instead of `embed_fn: &dyn Fn(...)` or `Box<dyn Fn(...)>`. A shortlist call embeds a handful of
//! short strings once per choice question — not a hot loop — so the extra dynamic-dispatch
//! indirection a trait object would add buys nothing; a generic parameter still accepts an
//! ordinary closure, a `fn` item, or a `&F`/`Box<F>` coerced by the caller, and it lets the
//! compiler monomorphize and inline the common "one closure, called a few times" case. The
//! function signature itself (`&[String] -> Result<Vec<Vec<f64>>>`) mirrors Python's
//! `Callable[[Sequence[str]], Any]`: a list of texts in, one embedding vector per text out, in
//! the same order, fallible (a real embedder can hit a network or model error, unlike this
//! crate's own deterministic test embedder).

use crate::answers::LayaResult;
use crate::error::{LayaError, Result};
use crate::laya_router::LayaPredictor;
use crate::python_json::PythonJson;
use crate::questions::{Question, QuestionSet};
use crate::sequence_builder::SequenceBuilder;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::HashSet;

/// Default shortlist size, matching Python's `DEFAULT_SHORTLIST_K`.
pub const DEFAULT_SHORTLIST_K: usize = 20;

/// What [`LayaShortlist::predict`] recorded for one shortlisted choice question.
#[derive(Debug, Clone, PartialEq)]
pub struct ShortlistInfo {
    /// Kept labels, in rank order (highest cosine similarity first).
    pub labels: Vec<String>,
    /// Cosine similarity per kept label, in the same order as `labels`. `None` when this
    /// question passed through unchanged (`k` was already `>=` the label count, so nothing was
    /// ranked and `embed_fn` was never called).
    pub scores: Option<Vec<f64>>,
    /// The `k` this question was shortlisted with.
    pub k: usize,
    /// How many labels the question had before shortlisting.
    pub n: usize,
    /// Whether this question passed through unchanged.
    pub passthrough: bool,
}

/// Namespace for the embedding shortlist. See the module docs for the source this ports.
pub struct LayaShortlist;

impl LayaShortlist {
    /// Return the top-`k` choice labels for `state`.
    ///
    /// `embed_fn` is called once, with the query text first and then one string per option in
    /// criteria order; option strings match [`SequenceBuilder::render_options`] for a choice
    /// question. When `k` is at least the number of labels, every label is returned in its
    /// original order and `embed_fn` is not called.
    ///
    /// Ties keep the earlier label. A zero vector scores 0 and does not outrank a label that came
    /// before it.
    ///
    /// `criteria` is an ordered `(label, description)` sequence — the same shape
    /// [`Question::choice`] takes, covering both Python's dict criteria (label -> description)
    /// and list criteria (bare labels, each with a `Value::Null` description) uniformly. Rejects
    /// an empty or duplicate-labelled criteria set the same way `Question::choice` does.
    pub fn shortlist_choice<F, L, D>(
        state: &Value,
        criteria: impl IntoIterator<Item = (L, D)>,
        embed_fn: F,
        k: usize,
        instructions: Option<&Value>,
    ) -> Result<Vec<String>>
    where
        F: Fn(&[String]) -> Result<Vec<Vec<f64>>>,
        L: Into<String>,
        D: Into<Value>,
    {
        let items: Vec<(String, Value)> = criteria
            .into_iter()
            .map(|(l, d)| (l.into(), d.into()))
            .collect();
        check_k(k)?;
        check_criteria(&items)?;
        let query = query_text(state, instructions);
        Ok(rank(query, &items, &embed_fn, k)?.labels)
    }

    /// Shortlist each choice question in `questions`, then call `predictor.predict` once on the
    /// reduced set. Non-choice questions are forwarded unchanged; a choice question whose label
    /// count is already `<= k` is forwarded unchanged and does not call `embed_fn`.
    ///
    /// The returned [`LayaResult`] carries [`LayaResult::shortlist`] recording, per shortlisted
    /// choice question id, the kept labels (rank order), their cosine scores (`None` for a
    /// passthrough), `k`, `n` and whether it passed through. `predictor` may be a bare
    /// [`crate::LayaEngine`] or a [`crate::laya_router::LayaRouter`] — they compose, since a
    /// router's own [`LayaPredictor::predict`] already attaches its routing decision to the
    /// result this then adds shortlist metadata to.
    pub fn predict<F>(
        predictor: &dyn LayaPredictor,
        state: impl Into<Value>,
        questions: &QuestionSet,
        embed_fn: F,
        k: usize,
    ) -> Result<LayaResult>
    where
        F: Fn(&[String]) -> Result<Vec<Vec<f64>>>,
    {
        check_k(k)?;
        let state = state.into();
        let mut reduced = QuestionSet::new();
        let mut meta: IndexMap<String, ShortlistInfo> = IndexMap::new();

        for (id, question) in questions.iter() {
            match question {
                Question::Choice(choice) => {
                    let items = choice.options().to_vec();
                    let query = query_text(&state, Some(choice.instructions()));
                    let outcome = rank(query, &items, &embed_fn, k)?;

                    meta.insert(
                        id.to_string(),
                        ShortlistInfo {
                            labels: outcome.labels.clone(),
                            scores: outcome.scores,
                            k,
                            n: outcome.n,
                            passthrough: outcome.passthrough,
                        },
                    );

                    if outcome.passthrough {
                        reduced.insert(id, question.clone())?;
                    } else {
                        // Rebuild with only the kept labels, in rank order: Python's
                        // `_subset_criteria` builds `{label: criteria[label] for label in
                        // labels}`, which is exactly a re-keyed, re-ordered option list here.
                        let kept: Vec<(String, Value)> = outcome
                            .labels
                            .into_iter()
                            .map(|label| {
                                let description = items
                                    .iter()
                                    .find(|(l, _)| *l == label)
                                    .map(|(_, d)| d.clone())
                                    .expect("a ranked label always came from `items`");
                                (label, description)
                            })
                            .collect();
                        let reduced_choice = Question::choice(choice.instructions().clone(), kept)?;
                        reduced.insert(id, reduced_choice)?;
                    }
                }
                other => {
                    reduced.insert(id, other.clone())?;
                }
            }
        }

        let result = predictor.predict(state, &reduced)?;
        Ok(result.with_shortlist(meta))
    }
}

fn check_k(k: usize) -> Result<()> {
    if k == 0 {
        return Err(LayaError::InvalidQuestion(
            "k must be a positive integer, got 0".to_string(),
        ));
    }
    Ok(())
}

fn check_criteria(items: &[(String, Value)]) -> Result<()> {
    if items.is_empty() {
        return Err(LayaError::InvalidQuestion(
            "choice criteria must contain at least one option".to_string(),
        ));
    }
    let mut seen = HashSet::with_capacity(items.len());
    for (label, _) in items {
        if !seen.insert(label.as_str()) {
            return Err(LayaError::InvalidQuestion(format!(
                "choice criteria label '{label}' is duplicated"
            )));
        }
    }
    Ok(())
}

/// The shortlist ranking's query text: `serialize_state(state)`, or `"instructions\n" +
/// serialize_state(state)` when non-blank instructions are given. Blank is `None` or the empty
/// string, matching Python's `instructions is None or instructions == ""` (a non-string
/// instructions value is never equal to `""`, so only an actual empty *string* counts).
fn query_text(state: &Value, instructions: Option<&Value>) -> String {
    let body = PythonJson::state(state);
    match instructions {
        Some(v) if !is_blank_instructions(v) => {
            format!("{}\n{body}", PythonJson::shortlist_instructions(v))
        }
        _ => body,
    }
}

fn is_blank_instructions(value: &Value) -> bool {
    matches!(value, Value::Null) || matches!(value, Value::String(s) if s.is_empty())
}

/// What [`rank`] found: the same four values [`ShortlistInfo`] eventually carries for one
/// question, plus the bare label list [`LayaShortlist::shortlist_choice`] returns.
struct RankOutcome {
    labels: Vec<String>,
    scores: Option<Vec<f64>>,
    passthrough: bool,
    n: usize,
}

/// Rank `items` against `query` with `embed_fn`. `query` is the already-built query text (not the
/// raw state), so both [`LayaShortlist::shortlist_choice`] and [`LayaShortlist::predict`] can
/// share this after building their own query text (a bare `instructions` kwarg for the former, a
/// question's own `instructions` field for the latter).
fn rank<F>(query: String, items: &[(String, Value)], embed_fn: &F, k: usize) -> Result<RankOutcome>
where
    F: Fn(&[String]) -> Result<Vec<Vec<f64>>>,
{
    let n = items.len();
    let keys: Vec<String> = items.iter().map(|(label, _)| label.clone()).collect();
    if k >= n {
        return Ok(RankOutcome {
            labels: keys,
            scores: None,
            passthrough: true,
            n,
        });
    }

    // A throwaway choice question purely to reuse `SequenceBuilder::render_options`'s exact
    // rendering rule (bare label when the description is blank, `"label: description"`
    // otherwise) rather than duplicating it here.
    let temp = Question::choice(Value::Null, items.iter().cloned())?;
    let option_texts = SequenceBuilder::render_options(&temp);

    let mut texts = Vec::with_capacity(1 + option_texts.len());
    texts.push(query);
    texts.extend(option_texts);

    let matrix = embed_matrix(embed_fn, &texts)?;
    let sims = cosine(&matrix[0], &matrix[1..]);

    let mut order: Vec<usize> = (0..sims.len()).collect();
    // Stable sort descending by score: comparing `(b, a)` instead of `(a, b)` reverses the sense
    // of the comparison while stability still preserves the original (ascending-index) order
    // among ties, matching `np.argsort(-sims, kind="mergesort")`.
    order.sort_by(|&a, &b| sims[b].total_cmp(&sims[a]));
    order.truncate(k);

    let labels: Vec<String> = order.iter().map(|&i| keys[i].clone()).collect();
    let scores: Vec<f64> = order.iter().map(|&i| sims[i]).collect();
    Ok(RankOutcome {
        labels,
        scores: Some(scores),
        passthrough: false,
        n,
    })
}

/// Calls `embed_fn`, checks the returned matrix has one row per text and a consistent, non-zero
/// row width, and zeroes non-finite components (`np.nan_to_num`: only the offending components,
/// not the whole row — `[NaN, 1.0]` becomes `[0.0, 1.0]`, a real vector, not a zero vector).
fn embed_matrix<F>(embed_fn: &F, texts: &[String]) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[String]) -> Result<Vec<Vec<f64>>>,
{
    let raw = embed_fn(texts)?;
    if raw.len() != texts.len() {
        return Err(LayaError::InvalidQuestion(format!(
            "embed_fn must return one row per text: got {} rows for {} texts",
            raw.len(),
            texts.len()
        )));
    }
    let dim = raw.first().map(Vec::len).unwrap_or(0);
    if dim == 0 || raw.iter().any(|row| row.len() != dim) {
        return Err(LayaError::InvalidQuestion(
            "embed_fn must return rows of equal, non-zero length".to_string(),
        ));
    }
    Ok(raw
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|v| if v.is_finite() { v } else { 0.0 })
                .collect()
        })
        .collect())
}

/// Cosine similarity of `query` against every row of `docs`. A zero-norm query scores every
/// document 0 (matching Python: the whole `sims` array is returned as-is, all zero, without even
/// looking at the documents). A zero-norm document scores 0 against a non-zero query.
fn cosine(query: &[f64], docs: &[Vec<f64>]) -> Vec<f64> {
    let query_norm = query.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut sims = vec![0.0; docs.len()];
    if query_norm == 0.0 {
        return sims;
    }
    for (i, doc) in docs.iter().enumerate() {
        let doc_norm = doc.iter().map(|v| v * v).sum::<f64>().sqrt();
        let denom = doc_norm * query_norm;
        if denom > 0.0 {
            let dot: f64 = doc.iter().zip(query).map(|(a, b)| a * b).sum();
            sims[i] = dot / denom;
        }
    }
    sims
}
