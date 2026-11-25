//! The abstention gate (`min_confidence`, #361/#394): the router's side of it with fake predictors
//! (validation order, batch-wide application, nothing loaded on a bad threshold), the exact error
//! texts, and the recorded Python results for `confidence_probe.json` and the model cases.
//!
//! The probe is model-free: it holds the output of Python's own `confidence.py` functions. The
//! model-backed cases skip, with a `skip:` line, until the golden files carry them or without
//! `LAYA_ONNX_ROOT`.

mod common;

use common::routing_golden;
use indexmap::IndexMap;
use laya::answers::{ActionInfo, Answer, ChoiceAnswer, LayaResult, NoulAnswer, ScoreAnswer, Usage};
use laya::batch_prediction::BatchOptions;
use laya::confidence::{GATE_STATES, MinConfidence, bucket_keys};
use laya::error::{LayaError, Result};
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{
    BatchRequest, LayaBatchPredictor, LayaPredictor, LayaRouter, LayaRouterOptions,
};
use laya::predict_options::PredictOptions;
use laya::questions::{Question, QuestionSet};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------- model-free

type Builds = Arc<Mutex<Vec<LayaCheckpoint>>>;
type Seen = Arc<Mutex<Vec<BatchOptions>>>;

/// A predictor whose answers carry the confidence its state names: the state `"0.8"` gives a
/// choice answer with `answer_confidence` 0.8 and a noul answer with the same.
struct Fake {
    seen: Seen,
}

fn answers_for(state: &Value) -> LayaResult {
    let confidence: f64 = state.as_str().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let mut probabilities = IndexMap::new();
    probabilities.insert("a".to_string(), confidence);
    probabilities.insert("b".to_string(), 1.0 - confidence);
    let choice = Answer::Choice(ChoiceAnswer::new(
        "a",
        probabilities,
        0.1,
        confidence,
        ActionInfo::default(),
    ));
    let noul = Answer::Noul(NoulAnswer::new(
        confidence,
        0.1,
        confidence,
        ActionInfo::default(),
    ));
    let mut by_id = HashMap::new();
    by_id.insert("pick".to_string(), choice);
    by_id.insert("flag".to_string(), noul);
    LayaResult::new(
        "fake",
        vec!["pick".to_string(), "flag".to_string()],
        by_id,
        Usage::default(),
    )
}

impl LayaPredictor for Fake {
    fn predict(&self, state: Value, _questions: &QuestionSet) -> Result<LayaResult> {
        Ok(answers_for(&state))
    }

    fn as_batch_predictor(&self) -> Option<&dyn LayaBatchPredictor> {
        Some(self)
    }
}

impl LayaBatchPredictor for Fake {
    fn predict_batch(
        &self,
        states: &[Value],
        _questions: &QuestionSet,
        options: &BatchOptions,
    ) -> Result<Vec<LayaResult>> {
        self.seen.lock().unwrap().push(options.clone());
        Ok(states.iter().map(answers_for).collect())
    }
}

fn fixture() -> (LayaRouter, Builds, Seen) {
    let builds: Builds = Arc::default();
    let seen: Seen = Arc::default();
    let (factory_builds, factory_seen) = (Arc::clone(&builds), Arc::clone(&seen));
    let router = LayaRouter::with_factory(LayaRouterOptions::default(), move |checkpoint, _| {
        factory_builds.lock().unwrap().push(checkpoint);
        Ok(Arc::new(Fake {
            seen: Arc::clone(&factory_seen),
        }) as Arc<dyn LayaPredictor>)
    })
    .expect("a fake router never fails to construct");
    (router, builds, seen)
}

fn questions() -> QuestionSet {
    QuestionSet::new()
        .with(
            "pick",
            Question::choice_labels("Which?", ["a", "b"]).unwrap(),
        )
        .unwrap()
}

fn scalar(threshold: f64) -> PredictOptions {
    PredictOptions::new().min_confidence(MinConfidence::scalar(threshold).unwrap())
}

#[test]
fn router_predict_gates_with_the_options_threshold() {
    let (router, _, _) = fixture();
    let result = router
        .predict_with_options(
            "0.8",
            &questions(),
            Some("en"),
            None,
            None,
            None,
            &scalar(0.9),
        )
        .unwrap();
    assert_eq!(result["pick"].abstention(), Some("abstained"));
    assert!(result["pick"].low_confidence());
    assert_eq!(result["pick"].abstention_threshold(), Some(0.9));
    let passed = router
        .predict_with_options(
            "0.8",
            &questions(),
            Some("en"),
            None,
            None,
            None,
            &scalar(0.5),
        )
        .unwrap();
    assert_eq!(passed["pick"].abstention(), Some("passed"));
    assert!(passed.routing().is_some(), "the routing decision is kept");
}

#[test]
fn router_predict_without_a_threshold_reports_no_gate() {
    let (router, _, _) = fixture();
    for result in [
        router
            .predict("0.8", &questions(), Some("en"), None, None)
            .unwrap(),
        router
            .predict_with_options(
                "0.8",
                &questions(),
                Some("en"),
                None,
                None,
                None,
                &PredictOptions::default(),
            )
            .unwrap(),
    ] {
        for (_, answer) in result.iter() {
            assert!(!answer.low_confidence());
            assert_eq!(answer.abstention(), None);
            assert_eq!(answer.abstention_threshold(), None);
        }
    }
}

#[test]
fn router_predict_refuses_a_bad_threshold_before_routing_or_loading() {
    let (router, builds, _) = fixture();
    let bad = PredictOptions::new().min_confidence(MinConfidence::Scalar(1.5));
    let error = router
        .predict_with_options("0.8", &questions(), Some("en"), None, None, None, &bad)
        .unwrap_err();
    assert!(matches!(error, LayaError::InvalidMinConfidence(_)));
    assert_eq!(
        error.to_string(),
        "min_confidence must be a float in [0.0, 1.0], got 1.5"
    );
    assert!(builds.lock().unwrap().is_empty(), "nothing was loaded");

    // Router.predict checks the state first (router.py:1336), then the threshold (:1338).
    let error = router
        .predict_with_options(
            Value::Null,
            &questions(),
            Some("en"),
            None,
            None,
            None,
            &bad,
        )
        .unwrap_err();
    assert!(matches!(error, LayaError::InvalidState(_)), "{error}");
}

#[test]
fn router_predict_batch_gates_every_result_with_one_threshold() {
    let (router, _, seen) = fixture();
    let requests = [
        BatchRequest::new("0.95", questions()).model("en"),
        BatchRequest::new("0.2", questions()).model("multi"),
        BatchRequest::new("0.8", questions()).model("en"),
    ];
    let options = BatchOptions::new().min_confidence(MinConfidence::scalar(0.9).unwrap());
    let results = router.predict_batch(&requests, &options).unwrap();
    let states: Vec<_> = results
        .iter()
        .map(|r| (r["pick"].abstention(), r["flag"].abstention()))
        .collect();
    assert_eq!(
        states,
        [
            (Some("passed"), Some("passed")),
            (Some("abstained"), Some("abstained")),
            (Some("abstained"), Some("abstained")),
        ]
    );
    for result in &results {
        assert_eq!(result["pick"].abstention_threshold(), Some(0.9));
    }
    // The threshold is the router's to apply; it is no part of the grouping or of what the
    // predictors are asked.
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|o| o.min_confidence.is_none())
    );
    assert_eq!(seen.lock().unwrap().len(), 2, "one call per checkpoint");
}

#[test]
fn router_predict_batch_refuses_a_bad_threshold_before_routing_or_loading() {
    let (router, builds, _) = fixture();
    // Even the routing error is not reached: the request names a checkpoint that does not exist.
    let requests = [BatchRequest::new("0.8", questions()).model("no-such-model")];
    let mut map = IndexMap::new();
    map.insert("choice:1".to_string(), 0.5);
    let options = BatchOptions::new().min_confidence(MinConfidence::Map(map));
    let error = router.predict_batch(&requests, &options).unwrap_err();
    assert!(
        matches!(error, LayaError::InvalidMinConfidence(_)),
        "{error}"
    );
    assert!(builds.lock().unwrap().is_empty());
}

#[test]
fn router_predict_batch_with_no_requests_still_validates() {
    let (router, _, _) = fixture();
    let options = BatchOptions::new().min_confidence(MinConfidence::Scalar(-1.0));
    assert!(router.predict_batch(&[], &options).is_err());
    assert!(
        router
            .predict_batch(&[], &BatchOptions::new())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn the_public_vocabulary_is_python_s() {
    assert_eq!(GATE_STATES, ["passed", "abstained", "unevaluated"]);
    let keys = bucket_keys();
    assert_eq!(keys.len(), 12);
    for qtype in ["choice", "score", "noul"] {
        for size in ["2", "3-5", "6-10", "11+"] {
            assert!(keys.contains(&format!("{qtype}:{size}")));
        }
    }
}

// ---------------------------------------------------------------------- confidence_probe.json

/// Python's own `laya/confidence.py` over the shared synthetic inputs of
/// `tools/confidence_cases.py`. A float field is a JSON number, or "nan"/"inf"/"-inf".
fn probe() -> Option<Value> {
    let probe = routing_golden::load("confidence_probe.json");
    if probe.is_none() {
        eprintln!("skip: routing golden not found (tests/golden/routing/confidence_probe.json)");
    }
    probe
}

fn try_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "nan" => Some(f64::NAN),
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

fn float(value: &Value) -> f64 {
    try_float(value).unwrap_or_else(|| panic!("not a float: {value}"))
}

/// A float field the answer may not have reported: a typed answer reads it as NaN.
fn reported(answer: &Value, key: &str) -> f64 {
    answer.get(key).map_or(f64::NAN, float)
}

/// A typed answer for an entry of the probe, or `None` for the shapes a typed answer cannot hold
/// (`"typed": false`: a boolean or text "probability", a non-integer score level).
fn build_answer(entry: &Value) -> Option<Answer> {
    let confidence = reported(entry, "confidence");
    let answer_confidence = reported(entry, "answer_confidence");
    let action = ActionInfo::default();
    let probabilities = entry.get("probabilities").and_then(Value::as_object);
    match entry["type"].as_str()? {
        "choice" => {
            let mut map = IndexMap::new();
            for (label, value) in probabilities? {
                map.insert(label.clone(), try_float(value)?);
            }
            let first = map.keys().next().cloned().unwrap_or_default();
            Some(Answer::Choice(ChoiceAnswer::new(
                first,
                map,
                confidence,
                answer_confidence,
                action,
            )))
        }
        "score" => {
            let mut levels: Vec<(i64, f64)> = Vec::new();
            for (level, value) in probabilities? {
                levels.push((level.parse().ok()?, try_float(value)?));
            }
            levels.sort_by_key(|(level, _)| *level);
            Some(Answer::Score(ScoreAnswer::new(
                0.0,
                Vec::new(),
                levels.into_iter().map(|(_, p)| p).collect(),
                confidence,
                answer_confidence,
                action,
            )))
        }
        "noul" => Some(Answer::Noul(NoulAnswer::new(
            float(&entry["noul"]),
            confidence,
            answer_confidence,
            action,
        ))),
        _ => None,
    }
}

fn result_of(answers: Vec<(String, Answer)>) -> LayaResult {
    LayaResult::new(
        "probe",
        answers.iter().map(|(id, _)| id.clone()).collect(),
        answers.into_iter().collect(),
        Usage::default(),
    )
}

/// A threshold as the probe spells it: null (unset), a float, or an ordered bucket map.
fn threshold(value: &Value) -> Option<MinConfidence> {
    match value {
        Value::Null => None,
        Value::Object(map) => Some(
            MinConfidence::map(map.iter().map(|(k, v)| (k.clone(), float(v)))).expect("valid map"),
        ),
        other => Some(MinConfidence::scalar(float(other)).expect("valid scalar")),
    }
}

/// A router whose predictor hands back whatever result is in the slot, so a test can feed a
/// gated result back in, as a reused payload is.
struct Slot(Mutex<LayaResult>);

impl LayaPredictor for Slot {
    fn predict(&self, _state: Value, _questions: &QuestionSet) -> Result<LayaResult> {
        Ok(self.0.lock().unwrap().clone())
    }
}

fn slot_router(result: LayaResult) -> (LayaRouter, Arc<Slot>) {
    let slot = Arc::new(Slot(Mutex::new(result)));
    let shared = Arc::clone(&slot);
    let router = LayaRouter::with_factory(LayaRouterOptions::default(), move |_, _| {
        Ok(Arc::clone(&shared) as Arc<dyn LayaPredictor>)
    })
    .expect("a fake router never fails to construct");
    (router, slot)
}

/// Gate the slot's result at `min_confidence` through `Router::predict_with_options`, and store the
/// gated result back, so the next call gates a reused one.
fn gate_slot(
    router: &LayaRouter,
    slot: &Slot,
    min_confidence: Option<MinConfidence>,
) -> LayaResult {
    let mut options = PredictOptions::new();
    options.min_confidence = min_confidence;
    let result = router
        .predict_with_options(
            "state",
            &questions(),
            Some("en"),
            None,
            None,
            None,
            &options,
        )
        .expect("the gate does not fail on a valid threshold");
    *slot.0.lock().unwrap() = result.clone();
    result
}

fn assert_same_float(got: f64, want: f64, context: &str) {
    assert!(
        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
        "{context}: got {got:?}, python gave {want:?}"
    );
}

/// What the gate wrote on each answer must be exactly what Python wrote: `low_confidence` only
/// when flagged, `abstention` and `abstention_threshold` only when a threshold was set.
fn assert_verdicts(result: &LayaResult, expected: &Value, context: &str) {
    let expected = expected.as_object().expect("expected verdicts");
    assert_eq!(result.len(), expected.len(), "{context}: answer count");
    for (id, want) in expected {
        let answer = &result[id.as_str()];
        let where_answer = format!("{context} / {id}");
        assert_eq!(
            want.get("low_confidence").is_some(),
            answer.low_confidence(),
            "{where_answer}: low_confidence"
        );
        if let Some(flag) = want.get("low_confidence") {
            assert_eq!(flag, &Value::Bool(true), "{where_answer}: only ever true");
        }
        assert_eq!(
            want.get("abstention").and_then(Value::as_str),
            answer.abstention(),
            "{where_answer}: abstention"
        );
        match (
            want.get("abstention_threshold"),
            answer.abstention_threshold(),
        ) {
            (None, None) => {}
            (Some(want), Some(got)) => {
                assert_same_float(got, float(want), &format!("{where_answer}: threshold"));
            }
            (want, got) => panic!("{where_answer}: threshold {got:?} vs python {want:?}"),
        }
    }
}

fn gate_answers(probe: &Value) -> Vec<(String, Answer)> {
    probe["gate_answers"]
        .as_object()
        .expect("gate_answers")
        .iter()
        .map(|(id, entry)| {
            (
                id.clone(),
                build_answer(entry).expect("every gate answer is typed"),
            )
        })
        .collect()
}

#[test]
fn probe_vocabulary_matches_python() {
    let Some(probe) = probe() else { return };
    let mut keys = bucket_keys();
    keys.sort();
    let expected: Vec<&str> = probe["bucket_keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap())
        .collect();
    assert_eq!(keys, expected);
    let states: Vec<&str> = probe["gate_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap())
        .collect();
    assert_eq!(GATE_STATES.to_vec(), states);
}

#[test]
fn probe_scalar_thresholds_validate_as_python_does() {
    let Some(probe) = probe() else { return };
    let cases = probe["scalars"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let value = float(&case["value"]);
        let got = MinConfidence::scalar(value);
        if case["valid"].as_bool().unwrap() {
            let want = float(&case["result"]);
            match got {
                Ok(MinConfidence::Scalar(kept)) => assert_same_float(kept, want, "scalar"),
                other => panic!("{value:?}: python accepted it, got {other:?}"),
            }
        } else {
            let error = got.expect_err(&format!("{value:?}: python rejected it"));
            assert!(matches!(error, LayaError::InvalidMinConfidence(_)));
            assert_eq!(
                error.to_string(),
                case["error_message"].as_str().unwrap(),
                "{value:?}"
            );
        }
    }
}

#[test]
fn probe_threshold_maps_validate_as_python_does() {
    let Some(probe) = probe() else { return };
    let cases = probe["maps"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let pairs: Vec<(String, f64)> = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (pair[0].as_str().unwrap().to_string(), float(&pair[1])))
            .collect();
        let got = MinConfidence::map(pairs.clone());
        if case["valid"].as_bool().unwrap() {
            let Ok(MinConfidence::Map(kept)) = got else {
                panic!("{pairs:?}: python accepted it, got {got:?}");
            };
            // A repeated key keeps its first position and its last value, as a dict does.
            let want: Vec<(String, f64)> = case["result"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| (pair[0].as_str().unwrap().to_string(), float(&pair[1])))
                .collect();
            let got: Vec<(String, f64)> = kept.into_iter().collect();
            assert_eq!(got.len(), want.len(), "{pairs:?}");
            for ((got_key, got_value), (want_key, want_value)) in got.iter().zip(&want) {
                assert_eq!(got_key, want_key, "{pairs:?}");
                assert_same_float(*got_value, *want_value, &format!("{pairs:?}"));
            }
        } else {
            let error = got.expect_err(&format!("{pairs:?}: python rejected it"));
            assert!(matches!(error, LayaError::InvalidMinConfidence(_)));
            assert_eq!(
                error.to_string(),
                case["error_message"].as_str().unwrap(),
                "{pairs:?}"
            );
        }
    }
}

#[test]
fn probe_option_buckets_match_python() {
    let Some(probe) = probe() else { return };
    let cases = probe["buckets"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let answer = build_answer(&case["answer"]).expect("a typed answer");
        assert_eq!(
            answer.option_bucket().as_deref(),
            case["bucket"].as_str(),
            "{}",
            case["answer"]
        );
    }
}

#[test]
fn probe_resolved_thresholds_match_python() {
    let Some(probe) = probe() else { return };
    let cases = probe["resolve"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let answer = build_answer(&case["answer"]).expect("a typed answer");
        let (router, slot) = slot_router(result_of(vec![("q".to_string(), answer)]));
        let gated = gate_slot(&router, &slot, threshold(&case["thresholds"]));
        assert_same_float(
            gated["q"].abstention_threshold().expect("a gate ran"),
            float(&case["result"]),
            &format!("{} under {}", case["answer"], case["thresholds"]),
        );
    }
}

#[test]
fn probe_gate_verdicts_match_python() {
    let Some(probe) = probe() else { return };
    let cases = probe["gate"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let (router, slot) = slot_router(result_of(gate_answers(&probe)));
        let gated = gate_slot(&router, &slot, threshold(&case["threshold"]));
        assert_verdicts(
            &gated,
            &case["expected"],
            &format!("threshold {}", case["threshold"]),
        );
    }
}

/// #910: python's reused payload carried a stale flag across thresholds. A result here is a
/// value, but one gated and gated again must still end up as the last threshold alone decides.
#[test]
fn probe_reused_results_never_keep_an_earlier_verdict() {
    let Some(probe) = probe() else { return };
    let sequences = probe["reuse"].as_array().unwrap();
    assert!(!sequences.is_empty());
    for (index, sequence) in sequences.iter().enumerate() {
        let (router, slot) = slot_router(result_of(gate_answers(&probe)));
        for (step, case) in sequence["steps"].as_array().unwrap().iter().enumerate() {
            let gated = gate_slot(&router, &slot, threshold(&case["threshold"]));
            assert_verdicts(
                &gated,
                &case["expected"],
                &format!("reuse {index} step {step} at {}", case["threshold"]),
            );
        }
    }
}

#[test]
fn probe_jev_confidence_matches_python() {
    let Some(probe) = probe() else { return };
    let cases = probe["jev"].as_array().unwrap();
    assert!(!cases.is_empty());
    let (mut typed, mut untyped) = (0, 0);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let want = match &case["result"] {
            Value::Null => None,
            value => Some(float(value)),
        };
        // The shapes a typed answer cannot hold are the ones python refuses: no result.
        let Some(answer) = build_answer(&case["answer"]) else {
            assert!(
                !case["typed"].as_bool().unwrap(),
                "{name}: should be buildable"
            );
            assert_eq!(
                want, None,
                "{name}: python refuses what a typed answer cannot hold"
            );
            untyped += 1;
            continue;
        };
        assert!(
            case["typed"].as_bool().unwrap(),
            "{name}: typed answer built from an untyped case"
        );
        typed += 1;
        match (answer.jev_confidence(), want) {
            (None, None) => {}
            (Some(got), Some(want)) => assert_same_float(got, want, name),
            (got, want) => panic!("{name}: got {got:?}, python gave {want:?}"),
        }
    }
    assert!(
        typed > 60 && untyped >= 4,
        "typed {typed}, untyped {untyped}"
    );
}

// ---------------------------------------------------------------------- model-backed

/// The `min_confidence` cases of the checkpoint goldens (`min_confidence_*` for `system_one`,
/// `batch_min_confidence_*` for `predict_batch`) against the real model. The parity tests
/// (`predict_parity_tests`, `predict_batch_tests`) run every recorded case with its options; this
/// is what says the cases exist at all, since goldens recorded before the gate hold none and
/// would otherwise pass those tests without ever exercising it.
#[test]
fn the_recorded_min_confidence_cases_reproduce() {
    let mut checked_any = false;
    for checkpoint in common::engines::checkpoints() {
        let Some(engine) = common::engines::load(checkpoint) else {
            continue;
        };
        let golden = common::golden_data::CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }
        let single: Vec<_> = golden
            .success_cases()
            .into_iter()
            .filter(|case| case.name.starts_with("min_confidence_"))
            .collect();
        let batch: Vec<_> = golden
            .batch_cases()
            .into_iter()
            .filter(|case| case.name.starts_with("batch_min_confidence_"))
            .collect();
        if single.is_empty() || batch.is_empty() {
            eprintln!(
                "skip: the '{}' goldens hold no min_confidence cases; regenerate them with \
                 tools/regen_golden.py",
                checkpoint.subdir()
            );
            continue;
        }
        checked_any = true;
        let mut gated_answers = 0;
        for case in single {
            let root = golden.load_case(&case);
            let state = root["state"].clone();
            let questions = common::golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{}/{case}: {e}", checkpoint.subdir()));
            let options = common::golden_data::recorded_predict_options(&root);
            assert!(
                options.min_confidence.is_some(),
                "{case}: records a threshold"
            );
            let result = engine
                .predict_with_options(state, &questions, &options)
                .unwrap_or_else(|e| panic!("{}/{case}: {e}", checkpoint.subdir()));
            common::result_compare::assert_result_matches(
                &format!("{}/{case}", checkpoint.subdir()),
                &root["result"],
                &result,
            );
            gated_answers += result
                .iter()
                .filter(|(_, answer)| answer.abstention().is_some())
                .count();
        }
        assert!(gated_answers > 0, "no recorded case reported a gate state");
    }
    if !checked_any {
        eprintln!("skip: no ONNX artifacts + min_confidence golden cases found for any checkpoint");
    }
}
