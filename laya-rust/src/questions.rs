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
        })
    }
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

impl std::ops::Index<&str> for QuestionSet {
    type Output = Question;

    /// Panics if no question carries this id; use [`QuestionSet::get`] for a fallible lookup.
    fn index(&self, id: &str) -> &Question {
        self.map
            .get(id)
            .unwrap_or_else(|| panic!("no question with id '{id}'"))
    }
}
