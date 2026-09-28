//! Tier 1: the embedding shortlist. Ports the non-torch checks from `tests/test_shortlist.py`
//! (the torch mean-pool block — `embed_fn_from_agent` — is out of scope: this port only accepts a
//! caller-supplied `embed_fn`) plus every entry of the routing golden's `shortlist_probe.json`,
//! and the hashing embedder vs its own probe (see `hashing_embedder_tests.rs`).

mod common;

use common::routing_golden;
use indexmap::IndexMap;
use laya::answers::{ActionInfo, Answer, ChoiceAnswer, LayaResult, Usage};
use laya::error::{LayaError, Result};
use laya::laya_router::LayaPredictor;
use laya::laya_shortlist::{LayaShortlist, ShortlistInfo};
use laya::questions::{Question, QuestionSet};
use serde_json::{Value, json};
use std::collections::HashMap;

/// A predictor that never inspects `state`: it just answers every choice question with its first
/// (post-shortlist) label, so `LayaShortlist::predict` tests can check the shortlist metadata and
/// the reduced question set without needing a real engine.
struct FirstLabelPredictor;

impl LayaPredictor for FirstLabelPredictor {
    fn predict(&self, _state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        let order: Vec<String> = questions.ids().map(String::from).collect();
        let mut by_id = HashMap::new();
        for id in &order {
            if let Question::Choice(c) = questions.get(id).unwrap() {
                let first = c.labels().next().unwrap_or("").to_string();
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

fn table_embed(table: &[(&str, [f64; 2])]) -> impl Fn(&[String]) -> Result<Vec<Vec<f64>>> {
    let table: Vec<(String, Vec<f64>)> = table
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_vec()))
        .collect();
    move |texts: &[String]| {
        texts
            .iter()
            .map(|t| {
                table
                    .iter()
                    .find(|(k, _)| k == t)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| LayaError::InvalidQuestion(format!("unexpected text {t:?}")))
            })
            .collect()
    }
}

fn boom_embed(_texts: &[String]) -> Result<Vec<Vec<f64>>> {
    panic!("embed_fn should not run when k >= n")
}

// ---------------------------------------------------------------------- named cases (test_shortlist.py)

#[test]
fn deterministic_topk_and_ties() {
    // Vectors: query [1, 0]. alpha and delta tie at cosine 1; gamma is 0.6; beta is 0.
    // Stable order must keep alpha ahead of delta.
    let embed = table_embed(&[
        ("pay me", [1.0, 0.0]),
        ("alpha", [1.0, 0.0]),
        ("beta", [0.0, 1.0]),
        ("gamma: mid", [0.6, 0.8]),
        ("delta: same", [1.0, 0.0]),
    ]);
    let criteria = [
        ("alpha", Value::Null),
        ("beta", json!("")),
        ("gamma", json!("mid")),
        ("delta", json!("same")),
    ];
    let state = json!("pay me");

    assert_eq!(
        LayaShortlist::shortlist_choice(&state, criteria.clone(), &embed, 2, None).unwrap(),
        vec!["alpha", "delta"],
        "topk/k=2 keeps the cosine tie in input order"
    );
    assert_eq!(
        LayaShortlist::shortlist_choice(&state, criteria.clone(), &embed, 1, None).unwrap(),
        vec!["alpha"],
        "topk/k=1 is the earliest max"
    );
    assert_eq!(
        LayaShortlist::shortlist_choice(&state, criteria.clone(), &embed, 3, None).unwrap(),
        vec!["alpha", "delta", "gamma"],
        "topk/k=3 appends the next cosine"
    );

    // All-zero query: every cosine is 0, so the earliest labels win.
    let zero_q = table_embed(&[
        ("pay me", [0.0, 0.0]),
        ("alpha", [1.0, 0.0]),
        ("beta", [0.0, 1.0]),
        ("gamma: mid", [0.6, 0.8]),
        ("delta: same", [3.0, 4.0]),
    ]);
    assert_eq!(
        LayaShortlist::shortlist_choice(&state, criteria.clone(), &zero_q, 2, None).unwrap(),
        vec!["alpha", "beta"],
        "topk/zero query keeps original order"
    );

    // A non-finite option vector is treated as 0 and loses to a real match.
    let nan_embed = |texts: &[String]| -> Result<Vec<Vec<f64>>> {
        Ok(texts
            .iter()
            .map(|t| match t.as_str() {
                "pay me" => vec![1.0, 0.0],
                "alpha" => vec![f64::NAN, f64::NAN],
                "beta" => vec![1.0, 0.0],
                other => panic!("unexpected text {other:?}"),
            })
            .collect())
    };
    assert_eq!(
        LayaShortlist::shortlist_choice(
            &state,
            [("alpha", Value::Null), ("beta", Value::Null)],
            nan_embed,
            1,
            None
        )
        .unwrap(),
        vec!["beta"],
        "topk/nan vector sorts behind a finite match"
    );
}

#[test]
fn list_criteria_and_instructions_change_the_query() {
    let embed = table_embed(&[
        ("Classify\npay me", [0.0, 1.0]),
        ("alpha", [1.0, 0.0]),
        ("beta", [0.0, 1.0]),
        ("gamma", [0.0, 0.2]),
    ]);
    let state = json!("pay me");
    let instructions = json!("Classify");
    let got = LayaShortlist::shortlist_choice(
        &state,
        [
            ("alpha", Value::Null),
            ("beta", Value::Null),
            ("gamma", Value::Null),
        ],
        &embed,
        2,
        Some(&instructions),
    )
    .unwrap();
    assert_eq!(
        got,
        vec!["beta", "gamma"],
        "list/instructions change the winner"
    );
}

#[test]
fn dict_state_is_serialized_into_the_query() {
    let embed = table_embed(&[
        ("Classify\n{\"text\": \"hi\"}", [1.0, 0.0]),
        ("alpha", [1.0, 0.0]),
        ("beta", [0.0, 1.0]),
    ]);
    let state = json!({"text": "hi"});
    let instructions = json!("Classify");
    let got = LayaShortlist::shortlist_choice(
        &state,
        [("alpha", Value::Null), ("beta", Value::Null)],
        &embed,
        1,
        Some(&instructions),
    )
    .unwrap();
    assert_eq!(got, vec!["alpha"], "query/dict state is serialized");
}

#[test]
fn k_ge_n_passes_through_without_calling_embed_fn() {
    let criteria = [
        ("alpha", Value::Null),
        ("beta", json!("")),
        ("gamma", json!("mid")),
        ("delta", json!("same")),
    ];
    let state = json!("pay me");
    assert_eq!(
        LayaShortlist::shortlist_choice(&state, criteria.clone(), boom_embed, 4, None).unwrap(),
        vec!["alpha", "beta", "gamma", "delta"],
        "pass/k == n returns every label in order"
    );
    assert_eq!(
        LayaShortlist::shortlist_choice(&state, criteria, boom_embed, 20, None).unwrap(),
        vec!["alpha", "beta", "gamma", "delta"],
        "pass/k > n returns every label in order"
    );
}

#[test]
fn predict_reduces_choice_and_forwards_other_questions_unchanged() {
    // cosine vs [1, 0]: tech=1, sales≈0.707, billing=0, other=0. k=2 -> tech, sales.
    let embed = table_embed(&[
        ("Which desk?\nI was charged twice", [1.0, 0.0]),
        ("billing: {\"desc\": \"payments\"}", [0.0, 1.0]),
        ("tech: bugs", [1.0, 0.0]),
        ("sales", [0.2, 0.2]),
        ("other: misc", [0.0, 1.0]),
    ]);
    let full_criteria = [
        ("billing", json!({"desc": "payments"})),
        ("tech", json!("bugs")),
        ("sales", Value::Null),
        ("other", json!("misc")),
    ];
    let intent = Question::choice("Which desk?", full_criteria).unwrap();
    let urgency = Question::score("How urgent?", ["low", "mid", "high", "now"]).unwrap();
    let refund = Question::noul("Is a refund requested?", Value::Null, Value::Null);
    let mut questions = QuestionSet::new();
    questions = questions.with("intent", intent).unwrap();
    questions = questions.with("urgency", urgency).unwrap();
    questions = questions.with("refund", refund).unwrap();

    let state = json!("I was charged twice");
    let result = LayaShortlist::predict(&FirstLabelPredictor, state, &questions, embed, 2).unwrap();

    let shortlist = result.shortlist().expect("shortlist metadata");
    let intent_meta = &shortlist["intent"];
    assert_eq!(
        intent_meta.labels,
        vec!["tech", "sales"],
        "kept top 2, rank order"
    );
    assert_eq!(intent_meta.k, 2);
    assert_eq!(intent_meta.n, 4);
    assert!(!intent_meta.passthrough);
    let scores = intent_meta.scores.as_ref().unwrap();
    assert!(
        scores[0] > scores[1] && scores[1] > 0.0,
        "result/scores descend"
    );
    assert!(
        !shortlist.contains_key("urgency"),
        "non-choice questions absent from shortlist meta"
    );

    assert_eq!(
        result.get("intent").unwrap().as_choice().unwrap().choice,
        "tech",
        "mock's first shortlisted label wins"
    );
}

#[test]
fn predict_passthrough_forwards_the_original_question() {
    let full_criteria = [
        ("billing", json!({"desc": "payments"})),
        ("tech", json!("bugs")),
        ("sales", Value::Null),
        ("other", json!("misc")),
    ];
    let intent = Question::choice("Which desk?", full_criteria).unwrap();
    let mut questions = QuestionSet::new();
    questions = questions.with("intent", intent).unwrap();

    let result = LayaShortlist::predict(
        &FirstLabelPredictor,
        json!("I was charged twice"),
        &questions,
        boom_embed,
        4,
    )
    .unwrap();
    let meta = &result.shortlist().unwrap()["intent"];
    assert!(meta.passthrough, "pass/flag");
    assert_eq!(meta.scores, None, "pass/scores omitted");
    assert_eq!(meta.labels, vec!["billing", "tech", "sales", "other"]);
}

#[test]
fn zero_and_false_criteria_values_are_real_descriptions() {
    let rich = [
        ("zero", json!(0)),
        ("no", json!(false)),
        ("bare", Value::Null),
        ("named", json!({"desc": "payments"})),
    ];
    let state = json!("pay me");
    // Passthrough (k >= n): does not call embed_fn, so this only exercises validation + ordering.
    let got = LayaShortlist::shortlist_choice(&state, rich, boom_embed, 4, None).unwrap();
    assert_eq!(got, vec!["zero", "no", "bare", "named"]);
}

#[test]
fn errors_are_rejected() {
    let state = json!("pay me");
    let embed = |_: &[String]| -> Result<Vec<Vec<f64>>> { Ok(vec![]) };

    assert!(matches!(
        LayaShortlist::shortlist_choice(&state, [("a", Value::Null)], embed, 0, None),
        Err(LayaError::InvalidQuestion(_))
    ));
    assert!(matches!(
        LayaShortlist::shortlist_choice(&state, Vec::<(&str, Value)>::new(), embed, 1, None),
        Err(LayaError::InvalidQuestion(_))
    ));
    assert!(matches!(
        LayaShortlist::shortlist_choice(
            &state,
            [("alpha", Value::Null), ("alpha", Value::Null)],
            embed,
            1,
            None
        ),
        Err(LayaError::InvalidQuestion(_))
    ));

    // A shape mismatch from embed_fn must fail, and must not have called `predict`.
    let bad_shape = |_: &[String]| -> Result<Vec<Vec<f64>>> { Ok(vec![vec![0.0; 4]]) };
    struct Counting(std::sync::atomic::AtomicU32);
    impl LayaPredictor for Counting {
        fn predict(&self, _state: Value, questions: &QuestionSet) -> Result<LayaResult> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(LayaResult::new(
                "mock",
                questions.ids().map(String::from).collect(),
                HashMap::new(),
                Usage::default(),
            ))
        }
    }
    let counting = Counting(std::sync::atomic::AtomicU32::new(0));
    let original = Question::choice(
        "Which desk?",
        [("billing", Value::Null), ("tech", Value::Null)],
    )
    .unwrap();
    let mut questions = QuestionSet::new();
    questions = questions.with("intent", original).unwrap();
    let err = LayaShortlist::predict(&counting, json!("pay me"), &questions, bad_shape, 1);
    assert!(err.is_err(), "err/bad embed shape");
    assert_eq!(
        counting.0.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "err/bad shape does not call predict"
    );
}

// ---------------------------------------------------------------------- shortlist_probe.json

fn parse_component(v: &Value) -> f64 {
    match v {
        Value::String(s) => match s.as_str() {
            "nan" => f64::NAN,
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            other => other
                .parse()
                .unwrap_or_else(|_| panic!("unexpected string component {other:?}")),
        },
        _ => v
            .as_f64()
            .expect("expected a numeric (or nan/inf-string) component"),
    }
}

fn parse_criteria(v: &Value) -> Vec<(String, Value)> {
    match v {
        Value::Object(map) => map
            .iter()
            .map(|(k, val)| (k.clone(), val.clone()))
            .collect(),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                (
                    item.as_str()
                        .expect("list criteria item must be a string")
                        .to_string(),
                    Value::Null,
                )
            })
            .collect(),
        other => panic!("unexpected criteria shape: {other:?}"),
    }
}

fn parse_vectors(v: &Value) -> Option<Vec<Vec<f64>>> {
    let arr = v.as_array()?;
    Some(
        arr.iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(parse_component)
                    .collect()
            })
            .collect(),
    )
}

#[test]
fn shortlist_probe_matches_every_entry() {
    let Some(cases) = routing_golden::load_array("shortlist_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };

    for case in &cases {
        let label = case["label"].as_str().unwrap();
        let state = case["state"].clone();
        let criteria = parse_criteria(&case["criteria"]);
        let k = case["k"].as_u64().unwrap() as usize;
        let instructions = &case["instructions"];
        let vectors = parse_vectors(&case["vectors"]);
        let expected_labels: Vec<String> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let expected_scores: Option<Vec<f64>> = case["scores"]
            .as_array()
            .map(|arr| arr.iter().map(|v| v.as_f64().unwrap()).collect());
        let expected_passthrough = case["passthrough"].as_bool().unwrap();
        let expected_n = case["n"].as_u64().unwrap() as usize;

        // shortlist_choice: the direct, label-only entry point.
        let vectors_for_choice = vectors.clone();
        let choice_embed = move |_texts: &[String]| -> Result<Vec<Vec<f64>>> {
            match &vectors_for_choice {
                Some(v) => Ok(v.clone()),
                None => panic!("{label}: embed_fn should not run when k >= n (shortlist_choice)"),
            }
        };
        let ins_opt = if instructions.is_null() {
            None
        } else {
            Some(instructions)
        };
        let got_labels =
            LayaShortlist::shortlist_choice(&state, criteria.clone(), choice_embed, k, ins_opt)
                .unwrap_or_else(|e| panic!("{label}: shortlist_choice failed: {e}"));
        assert_eq!(
            got_labels, expected_labels,
            "{label}: shortlist_choice labels"
        );

        // predict: the full ShortlistInfo (labels + scores + passthrough + n), reached by putting
        // the same criteria/instructions on one choice question.
        let question = Question::choice(instructions.clone(), criteria.clone()).unwrap();
        let mut questions = QuestionSet::new();
        questions = questions.with("q", question).unwrap();

        let vectors_for_predict = vectors.clone();
        let predict_embed = move |_texts: &[String]| -> Result<Vec<Vec<f64>>> {
            match &vectors_for_predict {
                Some(v) => Ok(v.clone()),
                None => panic!("{label}: embed_fn should not run when k >= n (predict)"),
            }
        };
        let result = LayaShortlist::predict(
            &FirstLabelPredictor,
            state.clone(),
            &questions,
            predict_embed,
            k,
        )
        .unwrap_or_else(|e| panic!("{label}: predict failed: {e}"));
        let info: &ShortlistInfo = &result.shortlist().unwrap()["q"];
        assert_eq!(info.labels, expected_labels, "{label}: predict labels");
        assert_eq!(
            info.passthrough, expected_passthrough,
            "{label}: passthrough"
        );
        assert_eq!(info.n, expected_n, "{label}: n");
        assert_eq!(info.k, k, "{label}: k");
        match (&info.scores, &expected_scores) {
            (None, None) => {}
            (Some(got), Some(want)) => {
                assert_eq!(got.len(), want.len(), "{label}: scores length");
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-9,
                        "{label}: scores[{i}] got {g}, want {w}"
                    );
                }
            }
            (got, want) => panic!("{label}: scores mismatch, got {got:?}, want {want:?}"),
        }
    }
    eprintln!(
        "shortlist_probe_matches_every_entry: checked {} cases",
        cases.len()
    );
}
