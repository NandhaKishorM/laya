//! The typed result of one `predict` call. Pure data — nothing here runs the model; `engine.rs`
//! (a later phase) is what builds these from calibrated probabilities.

use crate::confidence::{self, Gate};
use crate::laya_router::RouteDecision;
use crate::laya_shortlist::ShortlistInfo;
use crate::questions::QuestionType;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::HashMap;

/// Token accounting for one `predict` call. Laya emits no tokens, so output is always 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Non-padding tokens across every question in the batch.
    pub input_tokens: u32,
    /// Always 0: a System 1 pass generates nothing. Kept for API symmetry with token-based SDKs.
    pub output_tokens: u32,
}

/// The action head's read on whether the answer can be acted on directly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ActionInfo {
    /// Probability of acting rather than escalating: softmax over the action logits, first
    /// component.
    pub act_probability: f64,
}

/// The scan window that decided an answer, from `LayaEngine::predict_long`. Offsets index the
/// tokenized state; the end is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowInfo {
    /// Zero-based index of the deciding window.
    pub index: usize,
    /// First state token in the window.
    pub token_start: usize,
    /// One past the last state token in the window.
    pub token_end: usize,
    /// Total windows scored.
    pub count: usize,
}

/// The answer to a [`QuestionType::Choice`] question.
#[derive(Debug, Clone)]
pub struct ChoiceAnswer {
    /// The highest-probability label.
    pub choice: String,
    /// Probability per label, in the order the options were declared. Rounded to 4 decimals.
    pub probabilities: IndexMap<String, f64>,
    confidence: f64,
    answer_confidence: f64,
    action: ActionInfo,
    window: Option<WindowInfo>,
    gate: Gate,
}

impl ChoiceAnswer {
    /// Build a choice answer. `probabilities` must be non-empty and in label order.
    pub fn new(
        choice: impl Into<String>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        answer_confidence: f64,
        action: ActionInfo,
    ) -> Self {
        Self {
            choice: choice.into(),
            probabilities,
            confidence,
            answer_confidence,
            action,
            window: None,
            gate: Gate::default(),
        }
    }

    /// The probability assigned to one label, if any option carries that label.
    pub fn probability(&self, label: &str) -> Option<f64> {
        self.probabilities.get(label).copied()
    }
}

/// The answer to a [`QuestionType::Score`] question.
#[derive(Debug, Clone)]
pub struct ScoreAnswer {
    /// Expected level: the probability-weighted mean of the level indices, a fractional value
    /// between 0 and `legend.len() - 1`. Rounded to 4 decimals.
    pub score: f64,
    /// What each level index means. Engine-produced legends render each criterion as a string,
    /// matching Python 0.4.0, including structured dict/list levels.
    pub legend: Vec<Value>,
    /// Probability per level index. Rounded to 4 decimals.
    pub probabilities: Vec<f64>,
    confidence: f64,
    answer_confidence: f64,
    action: ActionInfo,
    window: Option<WindowInfo>,
    gate: Gate,
}

impl ScoreAnswer {
    /// Build a score answer.
    pub fn new(
        score: f64,
        legend: Vec<Value>,
        probabilities: Vec<f64>,
        confidence: f64,
        answer_confidence: f64,
        action: ActionInfo,
    ) -> Self {
        Self {
            score,
            legend,
            probabilities,
            confidence,
            answer_confidence,
            action,
            window: None,
            gate: Gate::default(),
        }
    }

    /// The single most likely level index, as opposed to the expected value. Ties go to the
    /// lowest index.
    pub fn most_likely_level(&self) -> usize {
        let mut best = 0;
        for (i, &p) in self.probabilities.iter().enumerate().skip(1) {
            if p > self.probabilities[best] {
                best = i;
            }
        }
        best
    }
}

/// The answer to a [`QuestionType::Noul`] question.
#[derive(Debug, Clone, Copy)]
pub struct NoulAnswer {
    /// Probability that the statement holds, in `[0, 1]`. Rounded to 4 decimals.
    pub probability: f64,
    confidence: f64,
    answer_confidence: f64,
    action: ActionInfo,
    window: Option<WindowInfo>,
    gate: Gate,
}

impl NoulAnswer {
    /// Build a noul answer.
    pub fn new(
        probability: f64,
        confidence: f64,
        answer_confidence: f64,
        action: ActionInfo,
    ) -> Self {
        Self {
            probability,
            confidence,
            answer_confidence,
            action,
            window: None,
            gate: Gate::default(),
        }
    }

    /// Whether the statement more likely holds than not.
    pub fn value(&self) -> bool {
        self.probability >= 0.5
    }
}

/// One answer. Cast with [`Answer::as_choice`], [`Answer::as_score`] or [`Answer::as_noul`].
#[derive(Debug, Clone)]
pub enum Answer {
    /// See [`ChoiceAnswer`].
    Choice(ChoiceAnswer),
    /// See [`ScoreAnswer`].
    Score(ScoreAnswer),
    /// See [`NoulAnswer`].
    Noul(NoulAnswer),
}

impl Answer {
    /// Which question type produced this answer.
    pub fn question_type(&self) -> QuestionType {
        match self {
            Answer::Choice(_) => QuestionType::Choice,
            Answer::Score(_) => QuestionType::Score,
            Answer::Noul(_) => QuestionType::Noul,
        }
    }

    /// Calibrated confidence in `[0, 1]`. For choice and score this is normalized Shannon entropy
    /// (`1 - H(p)/log k`); for noul it is `max(p, 1 - p)`. Rounded to 4 decimals.
    pub fn confidence(&self) -> f64 {
        match self {
            Answer::Choice(c) => c.confidence,
            Answer::Score(s) => s.confidence,
            Answer::Noul(n) => n.confidence,
        }
    }

    /// Calibrated confidence in `[0, 1]`, on the scale temperature scaling actually fits: `max(p)`
    /// over the option probabilities, clipped to `[0, 1]` and rounded to 4 decimals. Unlike
    /// [`Self::confidence`] (normalized entropy for choice/score, `max(p, 1-p)` for noul), this is
    /// the same quantity for every question type, so a caller can gate across types on one number.
    /// Ports `answer_confidence` from `laya/common.py` (laya 0.3.21); every ECE figure in the
    /// Python repository is computed on this quantity, not [`Self::confidence`].
    pub fn answer_confidence(&self) -> f64 {
        match self {
            Answer::Choice(c) => c.answer_confidence,
            Answer::Score(s) => s.answer_confidence,
            Answer::Noul(n) => n.answer_confidence,
        }
    }

    /// The scan window that decided this answer, or `None` for an ordinary `predict` and for a
    /// `predict_long` state short enough to be scored in one pass. The probabilities describe that
    /// window, not a calibrated whole-document probability.
    pub fn window(&self) -> Option<WindowInfo> {
        match self {
            Answer::Choice(c) => c.window,
            Answer::Score(s) => s.window,
            Answer::Noul(n) => n.window,
        }
    }

    pub(crate) fn with_window(mut self, window: WindowInfo) -> Self {
        match &mut self {
            Answer::Choice(c) => c.window = Some(window),
            Answer::Score(s) => s.window = Some(window),
            Answer::Noul(n) => n.window = Some(window),
        }
        self
    }

    pub(crate) fn set_gate(&mut self, gate: Gate) {
        match self {
            Answer::Choice(c) => c.gate = gate,
            Answer::Score(s) => s.gate = gate,
            Answer::Noul(n) => n.gate = gate,
        }
    }

    fn gate(&self) -> Gate {
        match self {
            Answer::Choice(c) => c.gate,
            Answer::Score(s) => s.gate,
            Answer::Noul(n) => n.gate,
        }
    }

    /// Whether the `min_confidence` gate flagged this answer: its confidence (`answer_confidence`,
    /// else `confidence`) fell below the threshold. `false` when no gate ran, and `false` for an
    /// answer with no usable confidence (that one reads `"unevaluated"`). Ports Python's
    /// `low_confidence: True`, which is present only when flagged.
    pub fn low_confidence(&self) -> bool {
        self.gate().low_confidence
    }

    /// The gate's state for this answer: one of [`crate::confidence::GATE_STATES`]
    /// (`"passed"`, `"abstained"`, `"unevaluated"`), or `None` when no `min_confidence` was set
    /// for the call that produced it. Ports Python's `abstention`.
    pub fn abstention(&self) -> Option<&'static str> {
        self.gate().abstention
    }

    /// The threshold this answer was gated at: the scalar, or the map entry for its option-count
    /// bucket (else the map's `"default"`, else `0.0`). `None` when no gate ran. Ports Python's
    /// `abstention_threshold`.
    pub fn abstention_threshold(&self) -> Option<f64> {
        self.gate().threshold
    }

    /// The `confidence` Jev would report for this answer's (rounded) probabilities, which is not
    /// Laya's normalized entropy: choice `(max(p) - 1/n) / (1 - 1/n)`, score `max(0, 1 -
    /// sum(p_i * |i - mode|) / D)`. `None` for a noul answer and for probabilities that are not
    /// finite and non-negative. Ports `jev_confidence` from `laya/confidence.py` (#302).
    pub fn jev_confidence(&self) -> Option<f64> {
        confidence::jev_confidence(self)
    }

    /// The `qtype:size` bucket a per-bucket `min_confidence` map gates this answer under, e.g.
    /// `"choice:3-5"` (see [`crate::confidence::bucket_keys`]). `None` for a choice or score
    /// answer with no options.
    pub fn option_bucket(&self) -> Option<String> {
        confidence::option_bucket(self)
    }

    /// The action head's output.
    pub fn action(&self) -> ActionInfo {
        match self {
            Answer::Choice(c) => c.action,
            Answer::Score(s) => s.action,
            Answer::Noul(n) => n.action,
        }
    }

    /// This answer as a choice answer.
    pub fn as_choice(&self) -> Option<&ChoiceAnswer> {
        match self {
            Answer::Choice(c) => Some(c),
            _ => None,
        }
    }

    /// This answer as a score answer.
    pub fn as_score(&self) -> Option<&ScoreAnswer> {
        match self {
            Answer::Score(s) => Some(s),
            _ => None,
        }
    }

    /// This answer as a noul answer.
    pub fn as_noul(&self) -> Option<&NoulAnswer> {
        match self {
            Answer::Noul(n) => Some(n),
            _ => None,
        }
    }
}

/// Option-span collision diagnostics reported only when clipping makes options indistinguishable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionUsage {
    /// Options requested by the question.
    pub total: usize,
    /// Distinct token spans remaining after per-option clipping.
    pub distinct: usize,
    /// Applied token cap, including the marker, or `None` when no cap was applied.
    pub tokens_per_option: Option<usize>,
}

/// State-budget diagnostics reported by Python's `usage` since 0.4.0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateUsage {
    /// Tokens in the untruncated state, counted once rather than once per question.
    pub state_tokens: usize,
    /// The largest state-token loss across all question heads.
    pub state_tokens_dropped: usize,
    /// Whether at least one question lost state tokens.
    pub truncated: bool,
    /// Ids of questions that lost state tokens, in question order.
    pub truncated_questions: Vec<String>,
    /// Questions whose options collapsed to identical token spans.
    pub options: IndexMap<String, OptionUsage>,
}

/// How many scan windows `LayaEngine::predict_long` scored, and how many of them lost state
/// tokens. Kept beside [`Usage`] rather than in it so `Usage` struct literals keep compiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanUsage {
    /// Windows scored; 1 when the state fit one ordinary pass.
    pub windows: usize,
    /// Windows in which at least one question lost state tokens (Python's `usage["truncated"]`
    /// for a scan; [`StateUsage::truncated`] is this being non-zero).
    pub truncated_windows: usize,
}

/// The result of one `predict` call: one answer per question, plus token usage.
#[derive(Debug, Clone)]
pub struct LayaResult {
    model: String,
    usage: Usage,
    order: Vec<String>,
    by_id: HashMap<String, Answer>,
    routing: Option<RouteDecision>,
    shortlist: Option<IndexMap<String, ShortlistInfo>>,
    state_usage: Option<StateUsage>,
    scan_usage: Option<ScanUsage>,
}

impl LayaResult {
    /// Build a result from the model identifier, question order, and per-id answers. `routing`
    /// and `shortlist` start unset; see [`Self::with_routing`] and [`Self::with_shortlist`].
    /// Ports Python's plain result dict, which `router.py`/`shortlist.py` add a `"routing"` /
    /// `"shortlist"` key to after the fact — the same shape here as two optional fields, since
    /// `LayaResult` is a typed struct rather than a dict.
    pub fn new(
        model: impl Into<String>,
        order: Vec<String>,
        by_id: HashMap<String, Answer>,
        usage: Usage,
    ) -> Self {
        Self {
            model: model.into(),
            usage,
            order,
            by_id,
            routing: None,
            shortlist: None,
            state_usage: None,
            scan_usage: None,
        }
    }

    /// Attach a routing decision (`LayaRouter::predict` calls this on its own result before
    /// returning it).
    pub fn with_routing(mut self, decision: RouteDecision) -> Self {
        self.routing = Some(decision);
        self
    }

    /// Attach shortlist metadata, one entry per shortlisted choice question (`LayaShortlist::
    /// predict` calls this on its own result before returning it).
    pub fn with_shortlist(mut self, shortlist: IndexMap<String, ShortlistInfo>) -> Self {
        self.shortlist = Some(shortlist);
        self
    }

    /// The routing decision that selected this result's checkpoint, when this result came from
    /// [`crate::laya_router::LayaRouter::predict`].
    pub fn routing(&self) -> Option<&RouteDecision> {
        self.routing.as_ref()
    }

    /// Per-question shortlist metadata, when this result came from
    /// [`crate::laya_shortlist::LayaShortlist::predict`]. They compose: a shortlisted predict run
    /// through a router carries both this and [`Self::routing`].
    pub fn shortlist(&self) -> Option<&IndexMap<String, ShortlistInfo>> {
        self.shortlist.as_ref()
    }

    /// The model identifier, matching Python's `"laya-rl-agent"`-style `model_name`.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Token accounting for the call.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// State-budget diagnostics for engine-produced answers. Manually constructed results
    /// retain `None`, preserving the existing constructor and the `Copy` token-total `Usage`.
    pub fn state_usage(&self) -> Option<&StateUsage> {
        self.state_usage.as_ref()
    }

    /// Window counts for a result from `LayaEngine::predict_long`; `None` for an ordinary
    /// `predict`. When present, [`Self::state_usage`] holds the sums across windows.
    pub fn scan_usage(&self) -> Option<ScanUsage> {
        self.scan_usage
    }

    pub(crate) fn with_scan_usage(mut self, usage: ScanUsage) -> Self {
        self.scan_usage = Some(usage);
        self
    }

    pub(crate) fn with_state_usage(mut self, usage: StateUsage) -> Self {
        self.state_usage = Some(usage);
        self
    }

    /// Applies `f` to every answer, in no particular order (the gate writes each independently).
    pub(crate) fn for_each_answer_mut(&mut self, mut f: impl FnMut(&mut Answer)) {
        self.by_id.values_mut().for_each(&mut f);
    }

    /// The question ids, in the order they were asked.
    pub fn ids(&self) -> &[String] {
        &self.order
    }

    /// How many answers there are.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether there are no answers.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The answer to one question, if it was asked.
    pub fn get(&self, id: &str) -> Option<&Answer> {
        self.by_id.get(id)
    }

    /// Every `(id, answer)` pair, in the order the questions were asked.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Answer)> {
        self.order
            .iter()
            .map(move |id| (id.as_str(), &self.by_id[id]))
    }
}

impl std::ops::Index<&str> for LayaResult {
    type Output = Answer;

    /// Panics if no question carried this id; use [`LayaResult::get`] for a fallible lookup.
    fn index(&self, id: &str) -> &Answer {
        self.by_id
            .get(id)
            .unwrap_or_else(|| panic!("no answer for question '{id}'"))
    }
}
