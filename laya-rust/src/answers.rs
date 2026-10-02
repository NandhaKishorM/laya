//! The typed result of one `predict` call. Pure data — nothing here runs the model; `engine.rs`
//! (a later phase) is what builds these from calibrated probabilities.

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
    /// What each level index means, as supplied on the question.
    pub legend: Vec<Value>,
    /// Probability per level index. Rounded to 4 decimals.
    pub probabilities: Vec<f64>,
    confidence: f64,
    answer_confidence: f64,
    action: ActionInfo,
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

/// The result of one `predict` call: one answer per question, plus token usage.
#[derive(Debug, Clone)]
pub struct LayaResult {
    model: String,
    usage: Usage,
    order: Vec<String>,
    by_id: HashMap<String, Answer>,
    routing: Option<RouteDecision>,
    shortlist: Option<IndexMap<String, ShortlistInfo>>,
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
