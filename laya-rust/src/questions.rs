//! Typed questions and the insertion-ordered set they are asked in.
//!
//! Ports `laya/presets.py`'s question shape. Order is load-bearing throughout this module: batch
//! row order follows `QuestionSet` order, and choice label order decides which logit each
//! label's probability comes from (see `sequence-construction.md`).

use crate::error::{LayaError, Result};
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::HashSet;

/// The three question types Laya answers. Values match Python's `QTYPES` and the `qtype` tensor
/// encoding (`model-contract.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionType {
    /// Pick one label from a set of named options.
    Choice = 0,
    /// Place the state on an ordered scale; the answer is an expected level.
    Score = 1,
    /// Does the statement hold? The answer is a probability.
    Noul = 2,
}

// The Python type name (`SequenceBuilder::type_name`) and the `qtype` tensor's integer encoding
// (a plain `qtype as i64` cast, since the discriminants above already match Python's `QTYPES`)
// live in `sequence_builder.rs` alongside the rest of that logic, rather than here, keeping this
// enum bare.

impl TryFrom<i64> for QuestionType {
    type Error = LayaError;

    /// The question type for a `qtype` tensor value. Rust has no implicit enum-from-integer cast,
    /// so this is the explicit equivalent.
    fn try_from(index: i64) -> Result<Self> {
        match index {
            0 => Ok(QuestionType::Choice),
            1 => Ok(QuestionType::Score),
            2 => Ok(QuestionType::Noul),
            other => Err(LayaError::InvalidQuestion(format!(
                "unknown qtype index {other}"
            ))),
        }
    }
}

/// A [`QuestionType::Choice`] question: ordered, named options.
#[derive(Debug, Clone)]
pub struct ChoiceQuestion {
    instructions: Value,
    options: Vec<(String, Value)>,
    option_order: Option<Vec<i64>>,
    labels: Option<Value>,
}

impl ChoiceQuestion {
    /// The prompt: a string verbatim, or anything else rendered through
    /// [`crate::python_json::instructions`].
    pub fn instructions(&self) -> &Value {
        &self.instructions
    }

    /// The options in label order. Index N here is the label for logit N.
    pub fn options(&self) -> &[(String, Value)] {
        &self.options
    }

    /// The labels, in order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.options.iter().map(|(label, _)| label.as_str())
    }
}

/// A [`QuestionType::Score`] question: an ordered scale, lowest level first.
#[derive(Debug, Clone)]
pub struct ScoreQuestion {
    instructions: Value,
    levels: Vec<Value>,
    option_order: Option<Vec<i64>>,
    labels: Option<Value>,
}

impl ScoreQuestion {
    /// The prompt. See [`ChoiceQuestion::instructions`].
    pub fn instructions(&self) -> &Value {
        &self.instructions
    }

    /// The levels, lowest first. The answer is an expectation over these indices.
    pub fn levels(&self) -> &[Value] {
        &self.levels
    }
}

/// A [`QuestionType::Noul`] question: does the statement hold?
#[derive(Debug, Clone)]
pub struct NoulQuestion {
    instructions: Value,
    if_false: Value,
    if_true: Value,
    option_order: Option<Vec<i64>>,
    labels: Option<Value>,
}

impl NoulQuestion {
    /// The prompt. See [`ChoiceQuestion::instructions`].
    pub fn instructions(&self) -> &Value {
        &self.instructions
    }

    /// Wording for the false option, or `Value::Null` for the default.
    pub fn if_false(&self) -> &Value {
        &self.if_false
    }

    /// Wording for the true option, or `Value::Null` for the default.
    pub fn if_true(&self) -> &Value {
        &self.if_true
    }
}

/// One typed question. Construct with [`Question::choice`], [`Question::choice_labels`],
/// [`Question::score`] or [`Question::noul`].
#[derive(Debug, Clone)]
pub enum Question {
    /// See [`ChoiceQuestion`].
    Choice(ChoiceQuestion),
    /// See [`ScoreQuestion`].
    Score(ScoreQuestion),
    /// See [`NoulQuestion`].
    Noul(NoulQuestion),
}

impl Question {
    /// Which of the three question types this is.
    pub fn question_type(&self) -> QuestionType {
        match self {
            Question::Choice(_) => QuestionType::Choice,
            Question::Score(_) => QuestionType::Score,
            Question::Noul(_) => QuestionType::Noul,
        }
    }

    /// The prompt. A string is used verbatim; anything else is serialized with
    /// [`crate::python_json::instructions`], matching Python's `_to_internal`.
    pub fn instructions(&self) -> &Value {
        match self {
            Question::Choice(c) => c.instructions(),
            Question::Score(s) => s.instructions(),
            Question::Noul(n) => n.instructions(),
        }
    }

    /// This question as a choice question.
    pub fn as_choice(&self) -> Option<&ChoiceQuestion> {
        match self {
            Question::Choice(c) => Some(c),
            _ => None,
        }
    }

    /// This question as a score question.
    pub fn as_score(&self) -> Option<&ScoreQuestion> {
        match self {
            Question::Score(s) => Some(s),
            _ => None,
        }
    }

    /// This question as a noul question.
    pub fn as_noul(&self) -> Option<&NoulQuestion> {
        match self {
            Question::Noul(n) => Some(n),
            _ => None,
        }
    }

    /// A choice question over ordered `(label, description)` options. A `Value::Null` description
    /// means "no description"; note that `0` and `false` are real descriptions and are rendered
    /// (see [`crate::python_json::is_blank_criterion`]).
    ///
    /// Rejects duplicate labels and an empty option list: Python would silently collapse
    /// duplicate keys into one option, changing the option count and so every marker position, so
    /// this refuses rather than silently answering a different question.
    pub fn choice<I, L, D>(instructions: impl Into<Value>, options: I) -> Result<Question>
    where
        I: IntoIterator<Item = (L, D)>,
        L: Into<String>,
        D: Into<Value>,
    {
        let mut list = Vec::new();
        let mut seen = HashSet::new();
        for (label, description) in options {
            let label = label.into();
            if !seen.insert(label.clone()) {
                return Err(LayaError::InvalidQuestion(format!(
                    "duplicate choice label '{label}'"
                )));
            }
            list.push((label, description.into()));
        }
        if list.is_empty() {
            return Err(LayaError::InvalidQuestion(
                "a choice question needs at least one option".to_string(),
            ));
        }
        Ok(Question::Choice(ChoiceQuestion {
            instructions: instructions.into(),
            options: list,
            option_order: None,
            labels: None,
        }))
    }

    /// A choice question over bare labels, with no descriptions.
    pub fn choice_labels<I, L>(instructions: impl Into<Value>, labels: I) -> Result<Question>
    where
        I: IntoIterator<Item = L>,
        L: Into<String>,
    {
        Question::choice(
            instructions,
            labels.into_iter().map(|label| (label, Value::Null)),
        )
    }

    /// A score question over ordered levels, lowest first. Rejects an empty level list.
    pub fn score<I, D>(instructions: impl Into<Value>, levels: I) -> Result<Question>
    where
        I: IntoIterator<Item = D>,
        D: Into<Value>,
    {
        let list: Vec<Value> = levels.into_iter().map(Into::into).collect();
        if list.is_empty() {
            return Err(LayaError::InvalidQuestion(
                "a score question needs at least one level".to_string(),
            ));
        }
        Ok(Question::Score(ScoreQuestion {
            instructions: instructions.into(),
            levels: list,
            option_order: None,
            labels: None,
        }))
    }

    /// A yes/no question. Supplying `if_false` / `if_true` (e.g. `Some("on fire")`, or `None::<Value>`
    /// for the default) replaces the default "no, the statement does not hold" / "yes, the
    /// statement holds" wording.
    pub fn noul(
        instructions: impl Into<Value>,
        if_false: impl Into<Value>,
        if_true: impl Into<Value>,
    ) -> Question {
        Question::Noul(NoulQuestion {
            instructions: instructions.into(),
            if_false: if_false.into(),
            if_true: if_true.into(),
            option_order: None,
            labels: None,
        })
    }

    /// Show the options to the model in a different order. Slot `s` shows option `order[s]`; the
    /// answer is still reported by option, so the probabilities come back unpermuted. `order`
    /// must be a permutation of `0..n`, one slot per option (`n` is the number of options of a
    /// choice, the number of levels of a score, and 2 for a noul). Ports a question's
    /// `option_order` (`laya/agent.py` `_check_question`).
    ///
    /// Like Python, an order that is not such a permutation is not rejected here: the question
    /// is refused when it is predicted, with [`LayaError::InvalidQuestion`].
    pub fn with_option_order(mut self, order: impl IntoIterator<Item = i64>) -> Question {
        let order = Some(order.into_iter().collect());
        match &mut self {
            Question::Choice(q) => q.option_order = order,
            Question::Score(q) => q.option_order = order,
            Question::Noul(q) => q.option_order = order,
        }
        self
    }

    /// Name the two options of a noul question, instead of "false" / "true". `labels` is the
    /// JSON object Python takes: `{"false": ..., "true": ...}`, exactly those two keys, each a
    /// distinct non-empty string once stripped of whitespace. The model reads
    /// `"<label>: <criterion or default text>"`; the answer is unchanged (the probability of
    /// true). Ports a question's `labels` (`laya/common.py` `_resolve_noul_labels`).
    ///
    /// Like Python, labels that are not valid, or that sit on a question that is not a noul,
    /// are not rejected here: the question is refused when it is predicted, with
    /// [`LayaError::InvalidQuestion`]. `Value::Null` means the defaults.
    pub fn with_labels(mut self, labels: impl Into<Value>) -> Question {
        let labels = Some(labels.into());
        match &mut self {
            Question::Choice(q) => q.labels = labels,
            Question::Score(q) => q.labels = labels,
            Question::Noul(q) => q.labels = labels,
        }
        self
    }

    /// [`Question::with_labels`] for the usual case: the false label and the true label.
    pub fn with_noul_labels(self, false_label: &str, true_label: &str) -> Question {
        self.with_labels(serde_json::json!({ "false": false_label, "true": true_label }))
    }

    /// The labels set with [`Question::with_labels`], if any.
    pub fn labels(&self) -> Option<&Value> {
        match self {
            Question::Choice(q) => q.labels.as_ref(),
            Question::Score(q) => q.labels.as_ref(),
            Question::Noul(q) => q.labels.as_ref(),
        }
    }

    /// The option order set with [`Question::with_option_order`], if any.
    pub fn option_order(&self) -> Option<&[i64]> {
        match self {
            Question::Choice(q) => q.option_order.as_deref(),
            Question::Score(q) => q.option_order.as_deref(),
            Question::Noul(q) => q.option_order.as_deref(),
        }
    }

    /// Python's `_option_count`: how many options this question renders to.
    fn option_count(&self) -> usize {
        match self {
            Question::Choice(q) => q.options.len(),
            Question::Score(q) => q.levels.len(),
            Question::Noul(_) => 2,
        }
    }

    /// Ports the rules of `Agent._check_question` that can fail after construction:
    /// `option_order`, then `labels`, with Python's texts.
    pub(crate) fn check_question(&self, id: &str) -> Result<()> {
        self.check_option_order(id)?;
        if self.labels().is_none() {
            return Ok(());
        }
        let prefix = crate::laya_router::python_repr_str(id);
        if !matches!(self, Question::Noul(_)) {
            return Err(LayaError::InvalidQuestion(format!(
                "question {prefix}: 'labels' is only supported for noul questions"
            )));
        }
        resolve_noul_labels(self.labels())
            .map(|_| ())
            .map_err(|e| LayaError::InvalidQuestion(format!("question {prefix}: {e}")))
    }

    /// Ports the `option_order` rule of `Agent._check_question`, with its text.
    pub(crate) fn check_option_order(&self, id: &str) -> Result<()> {
        let Some(order) = self.option_order() else {
            return Ok(());
        };
        let n = self.option_count();
        let mut seen = vec![false; n];
        let permutation = order.len() == n
            && order.iter().all(|&i| {
                usize::try_from(i)
                    .ok()
                    .filter(|&i| i < n && !std::mem::replace(&mut seen[i], true))
                    .is_some()
            });
        if permutation {
            return Ok(());
        }
        let shown: Vec<String> = order.iter().map(i64::to_string).collect();
        Err(LayaError::InvalidQuestion(format!(
            "question {}: 'option_order' must be a permutation of range({n}) -- one slot per \
             option, each option once -- got [{}]",
            crate::laya_router::python_repr_str(id),
            shown.join(", ")
        )))
    }
}

const NOUL_LABELS_MESSAGE: &str =
    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";

/// Python's `str.strip()`: `char::is_whitespace` plus U+001C..U+001F, which Python also counts.
fn python_strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// Ports `_resolve_noul_labels` (`laya/common.py`): the (false, true) labels, stripped. `None` and
/// `Value::Null` give the defaults. The error is Python's `ValueError` text.
pub(crate) fn resolve_noul_labels(
    labels: Option<&Value>,
) -> std::result::Result<(String, String), &'static str> {
    let labels = match labels {
        None | Some(Value::Null) => return Ok(("false".to_string(), "true".to_string())),
        Some(labels) => labels,
    };
    let Value::Object(map) = labels else {
        return Err(NOUL_LABELS_MESSAGE);
    };
    if map.len() != 2 || !map.contains_key("false") || !map.contains_key("true") {
        return Err(NOUL_LABELS_MESSAGE);
    }
    let (Value::String(f), Value::String(t)) = (&map["false"], &map["true"]) else {
        return Err(NOUL_LABELS_MESSAGE);
    };
    let (f, t) = (python_strip(f), python_strip(t));
    if f.is_empty() || t.is_empty() || f == t {
        return Err(NOUL_LABELS_MESSAGE);
    }
    Ok((f.to_string(), t.to_string()))
}

/// An insertion-ordered set of question id to [`Question`], answered in one batch.
///
/// Deliberately backed by [`IndexMap`] rather than a hash map: the batch row order must match the
/// order the caller wrote the questions in, because that is what Python's `dict` guarantees and
/// what the answer order is compared against.
#[derive(Debug, Clone, Default)]
pub struct QuestionSet {
    map: IndexMap<String, Question>,
}

impl QuestionSet {
    /// An empty set.
    pub fn new() -> Self {
        Self {
            map: IndexMap::new(),
        }
    }

    /// How many questions are in the set.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set has no questions.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Insert a question by id. Returns an error if the id is already present, rather than
    /// silently overwriting it and shifting every subsequent row's meaning.
    pub fn insert(&mut self, id: impl Into<String>, question: Question) -> Result<()> {
        let id = id.into();
        if self.map.contains_key(&id) {
            return Err(LayaError::InvalidQuestion(format!(
                "duplicate question id '{id}'"
            )));
        }
        self.map.insert(id, question);
        Ok(())
    }

    /// Builder-style [`insert`](Self::insert): `QuestionSet::new().with("a", q1)?.with("b", q2)?`.
    pub fn with(mut self, id: impl Into<String>, question: Question) -> Result<Self> {
        self.insert(id, question)?;
        Ok(self)
    }

    /// The question with this id, if any.
    pub fn get(&self, id: &str) -> Option<&Question> {
        self.map.get(id)
    }

    /// Ports the per-question validation `predict_batch` and `predict_long` run first
    /// (`Agent._check_question`), in question order; only the `option_order` and `labels` rules can fail here,
    /// the rest being refused when the question is built.
    pub(crate) fn check(&self) -> Result<()> {
        self.map
            .iter()
            .try_for_each(|(id, question)| question.check_question(id))
    }

    /// Whether the set contains this id.
    pub fn contains_id(&self, id: &str) -> bool {
        self.map.contains_key(id)
    }

    /// The question ids, in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    /// Every `(id, question)` pair, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Question)> {
        self.map.iter().map(|(id, q)| (id.as_str(), q))
    }
}

impl QuestionSet {
    /// An order-sensitive signature of the set: two sets share it only when they have the same
    /// ids, types, instructions and options in the same order. The router groups requests by it
    /// so equal schemas share a forward pass. Ports `_question_schema` (`laya/router.py`), which
    /// is `json.dumps(questions, sort_keys=False)`: option and key order are positional.
    pub(crate) fn schema_key(&self) -> String {
        let rows: Vec<Value> = self
            .map
            .iter()
            .map(|(id, question)| {
                let body = match question {
                    Question::Choice(q) => Value::Array(vec![
                        Value::from("choice"),
                        q.instructions.clone(),
                        Value::Array(
                            q.options
                                .iter()
                                .map(|(label, text)| {
                                    Value::Array(vec![Value::from(label.as_str()), text.clone()])
                                })
                                .collect(),
                        ),
                    ]),
                    Question::Score(q) => Value::Array(vec![
                        Value::from("score"),
                        q.instructions.clone(),
                        Value::Array(q.levels.clone()),
                    ]),
                    Question::Noul(q) => Value::Array(vec![
                        Value::from("noul"),
                        q.instructions.clone(),
                        q.if_false.clone(),
                        q.if_true.clone(),
                    ]),
                };
                let body = match (body, question.option_order()) {
                    (Value::Array(mut parts), Some(order)) => {
                        parts.push(Value::from(order.to_vec()));
                        Value::Array(parts)
                    }
                    (body, _) => body,
                };
                let body = match (body, question.labels()) {
                    (Value::Array(mut parts), Some(labels)) => {
                        parts.push(labels.clone());
                        Value::Array(parts)
                    }
                    (body, _) => body,
                };
                Value::Array(vec![Value::from(id.as_str()), body])
            })
            .collect();
        Value::Array(rows).to_string()
    }
}

impl std::ops::Index<&str> for QuestionSet {
    type Output = Question;

    /// Panics if no question carries this id; use [`QuestionSet::get`] for a fallible lookup.
    fn index(&self, id: &str) -> &Question {
        self.map
            .get(id)
            .unwrap_or_else(|| panic!("no question with id '{id}'"))
    }
}

#[cfg(test)]
mod option_order_tests {
    use super::*;

    fn three() -> Question {
        Question::choice_labels("h", ["a", "b", "c"]).unwrap()
    }

    fn message(result: Result<()>) -> String {
        match result {
            Err(LayaError::InvalidQuestion(text)) => text,
            other => panic!("expected InvalidQuestion, got {other:?}"),
        }
    }

    #[test]
    fn a_permutation_passes_for_each_question_type() {
        assert!(
            three()
                .with_option_order([2, 0, 1])
                .check_option_order("q")
                .is_ok()
        );
        assert!(
            Question::score("h", ["x", "y"])
                .unwrap()
                .with_option_order([1, 0])
                .check_option_order("q")
                .is_ok()
        );
        assert!(
            Question::noul("h", Value::Null, Value::Null)
                .with_option_order([1, 0])
                .check_option_order("q")
                .is_ok()
        );
        assert!(three().check_option_order("q").is_ok());
    }

    #[test]
    fn a_non_permutation_is_refused_with_pythons_text_and_the_question_id() {
        let text = message(
            three()
                .with_option_order([0, 0, 1])
                .check_option_order("it's"),
        );
        assert_eq!(
            "question \"it's\": 'option_order' must be a permutation of range(3) -- one slot per \
             option, each option once -- got [0, 0, 1]",
            text
        );
        let text = message(
            Question::noul("h", Value::Null, Value::Null)
                .with_option_order([0])
                .check_option_order("q"),
        );
        assert!(text.ends_with("range(2) -- one slot per option, each option once -- got [0]"));
        let text = message(
            Question::score("h", ["x", "y"])
                .unwrap()
                .with_option_order([-1, 0])
                .check_option_order("q"),
        );
        assert!(text.ends_with("range(2) -- one slot per option, each option once -- got [-1, 0]"));
        let text = message(three().with_option_order([]).check_option_order("q"));
        assert!(text.ends_with("range(3) -- one slot per option, each option once -- got []"));
        let text = message(three().with_option_order([0, 1, 3]).check_option_order("q"));
        assert!(text.ends_with("got [0, 1, 3]"));
    }

    #[test]
    fn the_set_check_reports_the_first_bad_question_in_order() {
        let set = QuestionSet::new()
            .with("ok", three().with_option_order([1, 2, 0]))
            .unwrap()
            .with("bad1", three().with_option_order([0, 1]))
            .unwrap()
            .with("bad2", three().with_option_order([9, 9, 9]))
            .unwrap();
        assert!(message(set.check()).starts_with("question 'bad1':"));
        assert!(
            QuestionSet::new()
                .with("ok", three())
                .unwrap()
                .check()
                .is_ok()
        );
    }

    #[test]
    fn the_schema_key_tells_two_orders_and_no_order_apart() {
        let key = |q: Question| QuestionSet::new().with("q", q).unwrap().schema_key();
        let plain = key(three());
        let first = key(three().with_option_order([0, 1, 2]));
        let second = key(three().with_option_order([2, 1, 0]));
        assert_ne!(plain, first);
        assert_ne!(first, second);
        assert_eq!(first, key(three().with_option_order([0, 1, 2])));
        let noul = Question::noul("h", Value::Null, Value::Null);
        assert_ne!(key(noul.clone()), key(noul.with_option_order([1, 0])));
    }
}

#[cfg(test)]
mod labels_tests {
    use super::*;
    use serde_json::json;

    const TEXT: &str =
        "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";

    fn noul() -> Question {
        Question::noul("h", Value::Null, Value::Null)
    }

    fn message(result: Result<()>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn valid_labels_are_stripped_and_the_defaults_apply_without_labels() {
        assert_eq!(
            ("no".to_string(), "yes".to_string()),
            resolve_noul_labels(Some(&json!({"false": " no\t", "true": "\nyes\u{1f}"}))).unwrap()
        );
        assert_eq!(
            ("false".to_string(), "true".to_string()),
            resolve_noul_labels(None).unwrap()
        );
        assert_eq!(
            ("false".to_string(), "true".to_string()),
            resolve_noul_labels(Some(&Value::Null)).unwrap()
        );
        assert!(
            noul()
                .with_noul_labels("no", "yes")
                .check_question("q")
                .is_ok()
        );
        // A null value in the labels is the default, as in Python.
        assert!(noul().with_labels(Value::Null).check_question("q").is_ok());
    }

    #[test]
    fn invalid_labels_are_refused_with_pythons_text_and_the_question_id() {
        for labels in [
            json!({"false": "no"}),
            json!({"false": "no", "true": "yes", "maybe": "m"}),
            json!({"False": "no", "true": "yes"}),
            json!(["no", "yes"]),
            json!("no/yes"),
            json!({"false": 0, "true": "yes"}),
            json!({"false": "no", "true": null}),
            json!({"false": "  ", "true": "yes"}),
            json!({"false": "no", "true": ""}),
            json!({"false": "same", "true": " same "}),
        ] {
            assert_eq!(
                format!("question \"it's\": {TEXT}"),
                message(noul().with_labels(labels.clone()).check_question("it's")),
                "{labels}"
            );
        }
    }

    #[test]
    fn labels_on_a_non_noul_question_are_refused() {
        let choice = Question::choice_labels("h", ["a", "b"]).unwrap();
        for question in [
            choice.clone().with_noul_labels("no", "yes"),
            choice.with_labels(Value::Null),
            Question::score("h", ["x", "y"])
                .unwrap()
                .with_noul_labels("no", "yes"),
        ] {
            assert_eq!(
                "question 'q': 'labels' is only supported for noul questions",
                message(question.check_question("q"))
            );
        }
    }

    #[test]
    fn option_order_is_checked_before_labels_and_the_set_reports_the_first_bad_one() {
        let both = noul().with_option_order([0]).with_labels(json!([]));
        assert!(message(both.check_question("q")).contains("'option_order'"));
        let set = QuestionSet::new()
            .with("ok", noul().with_noul_labels("no", "yes"))
            .unwrap()
            .with("bad", noul().with_noul_labels("x", "x"))
            .unwrap();
        assert_eq!(format!("question 'bad': {TEXT}"), message(set.check()));
    }

    #[test]
    fn the_schema_key_tells_labels_apart_in_value_and_order() {
        let key = |q: Question| QuestionSet::new().with("q", q).unwrap().schema_key();
        let plain = key(noul());
        let yes_no = key(noul().with_noul_labels("no", "yes"));
        assert_ne!(plain, yes_no);
        assert_ne!(yes_no, key(noul().with_noul_labels("nope", "yes")));
        assert_eq!(yes_no, key(noul().with_noul_labels("no", "yes")));
        // Python dumps the dict as written, so key order separates groups too.
        assert_ne!(
            yes_no,
            key(noul().with_labels(json!({"true": "yes", "false": "no"})))
        );
    }
}
