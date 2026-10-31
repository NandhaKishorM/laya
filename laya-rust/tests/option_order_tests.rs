//! Tier 1: a question's `option_order` without a model. Sequence slots follow the order, the
//! decode scatters probabilities back by option, and the shortlist drops a stale order with
//! Python's warning. Ports `tests/test_option_order.py`; the answers a model gives for an order
//! are the golden cases in `golden_questions_tests.rs`.

mod common;

use common::stub_tokenizer::StubTokenizer;
use indexmap::IndexMap;
use laya::answers::{ActionInfo, Answer, ChoiceAnswer, LayaResult, Usage};
use laya::calibration::Calibration;
use laya::error::Result;
use laya::laya_router::LayaPredictor;
use laya::laya_shortlist::LayaShortlist;
use laya::questions::{Question, QuestionSet};
use laya::sequence_builder::SequenceBuilder;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;

fn abc() -> Question {
    Question::choice_labels("h", ["a", "b", "c"]).unwrap()
}

fn texts_for(question: &Question) -> Vec<String> {
    let tok = StubTokenizer::new();
    SequenceBuilder::build(&tok, &json!("s"), question, 64, 32, false).unwrap();
    tok.encoded()
}

#[test]
fn choice_options_are_encoded_in_slot_order() {
    assert_eq!(
        vec!["choice question: h", " c", " a", " b", "s"],
        texts_for(&abc().with_option_order([2, 0, 1]))
    );
}

#[test]
fn score_options_are_encoded_in_slot_order() {
    let q = Question::score("h", ["x", "y", "z"])
        .unwrap()
        .with_option_order([1, 2, 0]);
    assert_eq!(
        vec![
            "score question: h",
            " level 1: y",
            " level 2: z",
            " level 0: x",
            "s"
        ],
        texts_for(&q)
    );
}

#[test]
fn noul_options_are_encoded_in_slot_order() {
    let q = Question::noul("h", Value::Null, Value::Null).with_option_order([1, 0]);
    assert_eq!(
        vec![
            "noul question: h",
            " true: yes, the statement holds",
            " false: no, the statement does not hold",
            "s"
        ],
        texts_for(&q)
    );
}

#[test]
fn the_identity_order_builds_the_same_sequence_as_none() {
    let tok = StubTokenizer::new();
    let build = |q: &Question| SequenceBuilder::build(&tok, &json!("s"), q, 64, 32, false).unwrap();
    assert_eq!(build(&abc()), build(&abc().with_option_order([0, 1, 2])));
}

#[test]
fn a_reordered_question_keeps_its_marker_count() {
    let tok = StubTokenizer::new();
    let (_, markers) = SequenceBuilder::build(
        &tok,
        &json!("s"),
        &abc().with_option_order([2, 1, 0]),
        64,
        32,
        false,
    )
    .unwrap();
    assert_eq!(3, markers.len());
}

#[test]
fn unpermute_scatters_canonical_by_option() {
    // canonical[order[s]] = p[s]
    let p = vec![0.5, 0.3, 0.2];
    assert_eq!(
        vec![0.3, 0.2, 0.5],
        Calibration::unpermute_probs(p.clone(), Some(&[2, 0, 1]))
    );
    assert_eq!(p, Calibration::unpermute_probs(p.clone(), Some(&[0, 1, 2])));
    assert_eq!(p, Calibration::unpermute_probs(p.clone(), None));
    // A length mismatch is returned untouched, as in Python's `unpermute_probs`.
    assert_eq!(p, Calibration::unpermute_probs(p.clone(), Some(&[1, 0])));
}

#[test]
fn softmax_in_slot_order_then_scatter_follows_the_order() {
    let logits = [2.0f32, 0.0, 1.0];
    let p = Calibration::probabilities(&logits, 3, 1.0);
    let out = Calibration::unpermute_probs(p.clone(), Some(&[1, 2, 0]));
    assert_eq!(p[0], out[1]);
    assert_eq!(p[1], out[2]);
    assert_eq!(p[2], out[0]);
}

// ---------------------------------------------------------------------------- shortlist

struct Capture;
static SEEN: Mutex<Vec<(String, log::Level, String)>> = Mutex::new(Vec::new());
impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        SEEN.lock().unwrap().push((
            record.target().to_string(),
            record.level(),
            record.args().to_string(),
        ));
    }
    fn flush(&self) {}
}
static LOGGER: Capture = Capture;

struct Echo(Mutex<Vec<QuestionSet>>);

impl LayaPredictor for Echo {
    fn predict(&self, _state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        self.0.lock().unwrap().push(questions.clone());
        let order: Vec<String> = questions.ids().map(String::from).collect();
        let mut by_id = HashMap::new();
        for id in &order {
            if let Question::Choice(c) = &questions[id.as_str()] {
                let first = c.labels().next().unwrap().to_string();
                let mut probabilities = IndexMap::new();
                probabilities.insert(first.clone(), 1.0);
                by_id.insert(
                    id.clone(),
                    Answer::Choice(ChoiceAnswer::new(
                        first,
                        probabilities,
                        1.0,
                        1.0,
                        ActionInfo::default(),
                    )),
                );
            }
        }
        Ok(LayaResult::new("mock", order, by_id, Usage::default()))
    }
}

fn embed(texts: &[String]) -> Result<Vec<Vec<f64>>> {
    // The query first, then one row per option: option i points further from the query.
    Ok(texts
        .iter()
        .enumerate()
        .map(|(i, _)| vec![1.0, i as f64])
        .collect())
}

#[test]
fn the_shortlist_drops_a_stale_order_with_pythons_warning_and_keeps_a_passthrough_one() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);
    let questions = QuestionSet::new()
        .with(
            "wide",
            Question::choice_labels("q", ["a", "b", "c", "d"])
                .unwrap()
                .with_option_order([3, 2, 1, 0]),
        )
        .unwrap()
        .with(
            "narrow",
            Question::choice_labels("q", ["a", "b"])
                .unwrap()
                .with_option_order([1, 0]),
        )
        .unwrap()
        .with(
            "score",
            Question::score("q", ["x", "y", "z"])
                .unwrap()
                .with_option_order([2, 1, 0]),
        )
        .unwrap();
    let echo = Echo(Mutex::new(Vec::new()));
    LayaShortlist::predict(&echo, "state", &questions, embed, 2).unwrap();

    let seen = echo.0.lock().unwrap();
    let forwarded = &seen[0];
    assert_eq!(None, forwarded["wide"].option_order());
    assert_eq!(Some(&[1, 0][..]), forwarded["narrow"].option_order());
    assert_eq!(Some(&[2, 1, 0][..]), forwarded["score"].option_order());
    // The caller's set is not mutated.
    assert_eq!(Some(&[3, 2, 1, 0][..]), questions["wide"].option_order());

    let log = SEEN.lock().unwrap();
    assert_eq!(1, log.len());
    assert_eq!("laya", log[0].0);
    assert_eq!(log::Level::Warn, log[0].1);
    assert_eq!(
        "laya.shortlist: question 'wide': 'option_order' (4 slots) was dropped because the label \
         set was narrowed to 2; the order no longer fits the kept options and remapping it onto \
         them would move options between slots.",
        log[0].2
    );
}
