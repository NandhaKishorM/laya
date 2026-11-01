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
use crate::laya_router::{LayaPredictor, python_repr_str};
use crate::long_prediction::emit_warnings;
use crate::python_json::PythonJson;
use crate::questions::{Question, QuestionSet};
use crate::sequence_builder::SequenceBuilder;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

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
                        let kept_count = kept.len();
                        let mut reduced_choice =
                            Question::choice(choice.instructions().clone(), kept)?;
                        // Python re-keys only the criteria and keeps every other key, so a
                        // `labels` entry (refused later for a choice) survives the narrowing.
                        if let Some(labels) = question.labels() {
                            reduced_choice = reduced_choice.with_labels(labels.clone());
                        }
                        // The order has one slot per option, so one that fitted the caller's
                        // labels does not fit the kept ones; it is dropped, not remapped (which
                        // would move options between slots). Ports `_drop_stale_order`.
                        if let Some(order) = question.option_order() {
                            emit_warnings(
                                &[format!(
                                    "laya.shortlist: question {}: 'option_order' ({} slots) was \
                                     dropped because the label set was narrowed to {}; the order \
                                     no longer fits the kept options and remapping it onto them \
                                     would move options between slots.",
                                    python_repr_str(id),
                                    order.len(),
                                    kept_count
                                )],
                                stale_order_site(),
                            );
                        }
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

    /// Narrow each large choice question by elimination, then call `predictor.predict` once more.
    /// Ports `predict_tournament` (`laya/shortlist.py`); needs no embedder.
    ///
    /// A choice with more than `group_size` labels is cut, in criteria order, into groups of
    /// near-equal size and at most `group_size`. One predict call answers every group of every
    /// such question (the groups go in as questions `"0"`, `"1"`, ..., so they share one forward
    /// pass), and each group's answer goes through to the next round. Rounds repeat until no
    /// choice has more than `group_size` labels left; at [`DEFAULT_TOURNAMENT_GROUP`] (16), up to
    /// 256 labels take one round. The final call answers every question of the request, each
    /// choice that played a round cut to its finalists; everything else goes to it unchanged, so
    /// when nothing needs a round it is the only call. The caller's `questions` are not changed.
    ///
    /// A choice that played a round loses an `option_order` it carried, with Python's
    /// `RuntimeWarning` text through the `log` facade (target `laya`), once per question per call
    /// and, as for any identical warning, once per process. A choice that needed no round keeps
    /// its order.
    ///
    /// The result is the final call's result, so its answers and [`LayaResult::usage`] are that
    /// call's, plus [`LayaResult::tournament`]: an entry for each choice question (`labels`, the
    /// finalists in criteria order; `n`, the count before the first round; `rounds`).
    /// Probabilities are over a played choice's finalists only.
    ///
    /// Errors: a `group_size` below 2, and the replies Python reads by key: a round reply without
    /// the answer to group `i` is `LayaError::InvalidQuestion("'<i>'")` (Python's `KeyError`),
    /// and an answer that is not a choice answer, `InvalidQuestion("'choice'")`.
    ///
    /// To forward per-call options to every round and the final call (Python's
    /// `**predict_kwargs`), use [`Self::predict_tournament_with`].
    pub fn predict_tournament(
        predictor: &dyn LayaPredictor,
        state: impl Into<Value>,
        questions: &QuestionSet,
        group_size: usize,
    ) -> Result<LayaResult> {
        Self::predict_tournament_with(
            |state, questions| predictor.predict(state, questions),
            state,
            questions,
            group_size,
        )
    }

    /// [`Self::predict_tournament`] with the predict call given as a closure, which is run for
    /// every round and for the final call. A caller captures its per-call options in it, as
    /// Python forwards `**predict_kwargs`.
    pub fn predict_tournament_with<P>(
        predict: P,
        state: impl Into<Value>,
        questions: &QuestionSet,
        group_size: usize,
    ) -> Result<LayaResult>
    where
        P: Fn(Value, &QuestionSet) -> Result<LayaResult>,
    {
        if group_size < 2 {
            return Err(LayaError::InvalidQuestion(format!(
                "group_size must be an integer of at least 2, got {group_size}"
            )));
        }
        let state = state.into();
        let mut meta: IndexMap<String, TournamentInfo> = IndexMap::new();
        let mut warned: HashSet<String> = HashSet::new();
        for (id, question) in questions.iter() {
            if let Question::Choice(choice) = question {
                let labels: Vec<String> = choice.options().iter().map(|(l, _)| l.clone()).collect();
                let n = labels.len();
                meta.insert(
                    id.to_string(),
                    TournamentInfo {
                        labels,
                        n,
                        rounds: 0,
                    },
                );
            }
        }

        loop {
            let mut groups: Vec<(String, Vec<String>)> = Vec::new();
            for (id, entry) in &meta {
                let len = entry.labels.len();
                let parts = len.div_ceil(group_size);
                if parts > 1 {
                    for i in 0..parts {
                        groups.push((
                            id.clone(),
                            entry.labels[i * len / parts..(i + 1) * len / parts].to_vec(),
                        ));
                    }
                }
            }
            if groups.is_empty() {
                break;
            }
            let mut round = QuestionSet::new();
            for (i, (id, labels)) in groups.iter().enumerate() {
                round.insert(
                    i.to_string(),
                    tournament_cut(questions, id, labels, &mut warned)?,
                )?;
            }
            let reply = predict(state.clone(), &round)?;
            let mut winners: IndexMap<String, Vec<String>> = IndexMap::new();
            for (i, (id, _)) in groups.iter().enumerate() {
                let answer = reply
                    .get(&i.to_string())
                    .ok_or_else(|| LayaError::InvalidQuestion(python_repr_str(&i.to_string())))?;
                let choice = answer
                    .as_choice()
                    .ok_or_else(|| LayaError::InvalidQuestion(python_repr_str("choice")))?;
                winners
                    .entry(id.clone())
                    .or_default()
                    .push(choice.choice.clone());
            }
            for (id, labels) in winners {
                let entry = meta.get_mut(&id).expect("a group came from a meta entry");
                entry.labels = labels;
                entry.rounds += 1;
            }
        }

        let mut last = QuestionSet::new();
        for (id, question) in questions.iter() {
            match meta.get(id) {
                Some(entry) if entry.rounds > 0 => {
                    last.insert(
                        id,
                        tournament_cut(questions, id, &entry.labels, &mut warned)?,
                    )?;
                }
                _ => last.insert(id, question.clone())?,
            }
        }
        let result = predict(state, &last)?;
        Ok(result.with_tournament(meta))
    }

    /// Wrap `embed_fn` in an LRU cache of `maxsize` strings. Ports `cached_embed_fn`; see
    /// [`CachedEmbed`]. Python's default is [`DEFAULT_CACHE_MAXSIZE`].
    pub fn cached_embed_fn<F>(embed_fn: F, maxsize: usize) -> Result<CachedEmbed<F>>
    where
        F: Fn(&[String]) -> Result<Vec<Vec<f64>>>,
    {
        CachedEmbed::with_maxsize(embed_fn, maxsize)
    }
}

/// Where Python attributes the stale-order warning: the fixed line of `predict_shortlist` that
/// calls `_drop_stale_order`, whoever called it, so the same text is shown once per process.
fn stale_order_site() -> &'static std::panic::Location<'static> {
    std::panic::Location::caller()
}

/// Where Python attributes the stale-order warning of `predict_tournament`: the fixed line in its
/// `cut` helper that calls `_drop_stale_order`, for every round and for the final call.
fn tournament_stale_order_site() -> &'static std::panic::Location<'static> {
    std::panic::Location::caller()
}

/// Python's `cut` in `predict_tournament`: the question `id` narrowed to `labels` (criteria
/// `{label: criteria[label] for label in labels}`, every other key kept), without an
/// `option_order`, which is dropped with a warning once per question per call.
fn tournament_cut(
    questions: &QuestionSet,
    id: &str,
    labels: &[String],
    warned: &mut HashSet<String>,
) -> Result<Question> {
    let question = &questions[id];
    let choice = question
        .as_choice()
        .expect("a tournament entry is a choice question");
    let mut kept: Vec<(String, Value)> = Vec::with_capacity(labels.len());
    for label in labels {
        let description = choice
            .options()
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, d)| d.clone())
            .ok_or_else(|| LayaError::InvalidQuestion(python_repr_str(label)))?;
        // A dict comprehension collapses a repeated key into its first slot.
        if !kept.iter().any(|(l, _)| l == label) {
            kept.push((label.clone(), description));
        }
    }
    let mut narrowed = Question::choice(choice.instructions().clone(), kept)?;
    if let Some(labels_value) = question.labels() {
        narrowed = narrowed.with_labels(labels_value.clone());
    }
    if let Some(order) = question.option_order()
        && warned.insert(id.to_string())
    {
        emit_warnings(
            &[format!(
                "laya.shortlist: question {}: 'option_order' ({} slots) was dropped because the \
                 label set was narrowed to {}; the order no longer fits the kept options and \
                 remapping it onto them would move options between slots.",
                python_repr_str(id),
                order.len(),
                labels.len()
            )],
            tournament_stale_order_site(),
        );
    }
    Ok(narrowed)
}

/// Default group size of [`LayaShortlist::predict_tournament`], Python's
/// `DEFAULT_TOURNAMENT_GROUP`.
pub const DEFAULT_TOURNAMENT_GROUP: usize = 16;

/// Default capacity of [`CachedEmbed`], Python's `cached_embed_fn(maxsize=4096)`.
pub const DEFAULT_CACHE_MAXSIZE: usize = 4096;

/// What [`LayaShortlist::predict_tournament`] recorded for one choice question (Python's
/// `result["tournament"][id]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TournamentInfo {
    /// The finalists, in criteria order (all the labels when no round was played).
    pub labels: Vec<String>,
    /// How many labels the question had before the first round: how wide the field was, not how
    /// many finalists came out of it.
    pub n: usize,
    /// How many elimination rounds the question played.
    pub rounds: usize,
}

/// A snapshot of a [`CachedEmbed`]'s counters (Python's `cache_info()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheInfo {
    /// Texts currently cached.
    pub size: usize,
    /// The capacity.
    pub maxsize: usize,
    /// Requested texts found in the cache, counted per occurrence.
    pub hits: u64,
    /// Requested texts not found in the cache, counted per occurrence.
    pub misses: u64,
}

#[derive(Default)]
struct CacheState {
    /// text -> (recency tick, row).
    rows: HashMap<String, (u64, Arc<Vec<f32>>)>,
    /// recency tick -> text; the first entry is the least recently used.
    recency: BTreeMap<u64, String>,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl CacheState {
    /// Python's `move_to_end`.
    fn touch(&mut self, key: &str) {
        self.tick += 1;
        let tick = self.tick;
        if let Some((old, _)) = self.rows.get_mut(key) {
            self.recency.remove(old);
            *old = tick;
            self.recency.insert(tick, key.to_string());
        }
    }
}

/// An LRU cache around an embedding function. Ports `cached_embed_fn` (`laya/shortlist.py`).
///
/// Lookups are exact string matches. Texts missing from the cache are de-duplicated and embedded
/// in a single `embed_fn` call. Rows are stored as `f32` and handed back widened to `f64`; at most
/// `maxsize` strings are kept, the least recently used evicted first. Nothing is cached when
/// `embed_fn` fails or returns a bad shape. A non-finite value is not a bad shape: NaN and
/// infinities become `0.0` on the way in (after the `f32` cast), and the repaired row is cached,
/// so one transient NaN keeps that component at zero until [`Self::cache_clear`].
///
/// Safe to share between threads: the lock covers only cache reads and writes, never the
/// `embed_fn` call. Use it as an embedder with `|texts| cache.embed(texts)`.
pub struct CachedEmbed<F> {
    embed_fn: F,
    maxsize: usize,
    state: Mutex<CacheState>,
}

impl<F> CachedEmbed<F>
where
    F: Fn(&[String]) -> Result<Vec<Vec<f64>>>,
{
    /// A cache of [`DEFAULT_CACHE_MAXSIZE`] strings.
    pub fn new(embed_fn: F) -> Self {
        Self {
            embed_fn,
            maxsize: DEFAULT_CACHE_MAXSIZE,
            state: Mutex::new(CacheState::default()),
        }
    }

    /// A cache of `maxsize` strings; `maxsize` must be at least 1.
    pub fn with_maxsize(embed_fn: F, maxsize: usize) -> Result<Self> {
        if maxsize < 1 {
            return Err(LayaError::InvalidQuestion(format!(
                "maxsize must be a positive integer, got {maxsize}"
            )));
        }
        Ok(Self {
            embed_fn,
            maxsize,
            state: Mutex::new(CacheState::default()),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One row per text, in order. An empty `texts` returns an empty matrix without touching the
    /// counters or `embed_fn`.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut found: HashMap<String, Arc<Vec<f32>>> = HashMap::new();
        let missing: Vec<String> = {
            let mut state = self.lock();
            let mut hits = 0u64;
            for key in texts {
                let row = state.rows.get(key.as_str()).map(|(_, row)| row.clone());
                if let Some(row) = row {
                    state.touch(key);
                    found.insert(key.clone(), row);
                    hits += 1;
                }
            }
            state.hits += hits;
            state.misses += texts.len() as u64 - hits;
            let mut seen: HashSet<&str> = HashSet::new();
            texts
                .iter()
                .filter(|key| !found.contains_key(key.as_str()) && seen.insert(key.as_str()))
                .cloned()
                .collect()
        };
        if !missing.is_empty() {
            let raw = (self.embed_fn)(&missing)?;
            let width = raw.first().map(Vec::len).unwrap_or(0);
            if raw.len() != missing.len() || width < 1 {
                let shape = if raw.is_empty() {
                    "(0,)".to_string()
                } else {
                    format!("({}, {})", raw.len(), width)
                };
                return Err(LayaError::InvalidQuestion(format!(
                    "embed_fn must return an array of shape ({}, dim), got {shape}",
                    missing.len()
                )));
            }
            if raw.iter().any(|row| row.len() != width) {
                return Err(LayaError::InvalidQuestion(
                    "embed_fn must return rows of equal, non-zero length".to_string(),
                ));
            }
            let fresh: Vec<Arc<Vec<f32>>> = raw
                .iter()
                .map(|row| {
                    Arc::new(
                        row.iter()
                            .map(|&v| {
                                let v = v as f32;
                                if v.is_finite() { v } else { 0.0 }
                            })
                            .collect(),
                    )
                })
                .collect();
            let mut state = self.lock();
            if let Some((_, key)) = state.recency.iter().next() {
                let want = state.rows[key].1.len();
                if width != want {
                    return Err(LayaError::InvalidQuestion(format!(
                        "embed_fn returned dim {width}, but the cache holds dim {want}; \
                         call cache_clear() if the model behind embed_fn changed"
                    )));
                }
            }
            for (key, row) in missing.iter().zip(&fresh) {
                if let Some((old, _)) = state.rows.remove(key) {
                    state.recency.remove(&old);
                }
                state.tick += 1;
                let tick = state.tick;
                state.rows.insert(key.clone(), (tick, row.clone()));
                state.recency.insert(tick, key.clone());
                while state.rows.len() > self.maxsize {
                    let (_, oldest) = state.recency.pop_first().expect("a non-empty cache");
                    state.rows.remove(&oldest);
                }
                found.insert(key.clone(), row.clone());
            }
        }
        Ok(texts
            .iter()
            .map(|key| found[key.as_str()].iter().map(|&v| f64::from(v)).collect())
            .collect())
    }

    /// The current size, capacity and counters.
    pub fn cache_info(&self) -> CacheInfo {
        let state = self.lock();
        CacheInfo {
            size: state.rows.len(),
            maxsize: self.maxsize,
            hits: state.hits,
            misses: state.misses,
        }
    }

    /// Empty the cache and reset both counters.
    pub fn cache_clear(&self) {
        let mut state = self.lock();
        state.rows.clear();
        state.recency.clear();
        state.hits = 0;
        state.misses = 0;
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
            // Python clips (`np.clip(dot / denom, -1.0, 1.0)`, `laya/shortlist.py::_cosine`) and
            // laya-go ports that with `TestCosineIsClipped` ("the clip is the reference's
            // contract"). The quotient rounds OUTSIDE [-1, 1] for vectors that are parallel but
            // not exactly representable -- a self-cosine of [34, 15.285714285714286,
            // 70.85714285714286] is 1.0000000000000002 -- so an unclipped value breaks the range
            // this API documents for `ShortlistInfo::scores`. `clamp` propagates NaN unchanged,
            // matching laya-go, which also leaves NaN alone.
            sims[i] = (dot / denom).clamp(-1.0, 1.0);
        }
    }
    sims
}
