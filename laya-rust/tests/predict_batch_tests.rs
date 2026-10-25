//! `predict_batch`: the model-free pieces (router grouping, order restoration, `route_batch`
//! loading nothing, option forwarding) with fake predictors, and the model-backed parity of every
//! recorded `"kind": "batch"` golden case.
//!
//! The goldens come from the genuine `ONNXAgent.predict_batch` over the same ONNX graph, so they pin
//! the chunking, the `sort_by_length` windows and the order restoration along with the numbers.
//!
//! Run with `LAYA_ONNX_ROOT` (and `LAYA_ONNX_SPLIT_ROOT` for the split layout) set and
//! `--test-threads=1`, as for the other model-backed tests.

mod common;

use common::engines::{self, CHECKPOINTS};
use common::golden_data::{self, CheckpointGoldenData};
use common::result_compare::{assert_result_matches, assert_state_usage_matches};
use laya::answers::{LayaResult, Usage};
use laya::batch_prediction::BatchOptions;
use laya::error::{LayaError, Result};
use laya::laya_engine::LayaEngine;
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{
    BatchRequest, LangGuess, LayaBatchPredictor, LayaPredictor, LayaRouter, LayaRouterOptions,
};
use laya::questions::{Question, QuestionSet};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------- model-free

/// One call a fake predictor received.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    checkpoint: LayaCheckpoint,
    states: Vec<Value>,
    ids: Vec<String>,
    options: BatchOptions,
}

/// Unmistakably English text, so detection alone routes it to the English checkpoint.
const ENGLISH: &str = "Hello there, this is plain English text from a customer.";

type CallLog = Arc<Mutex<Vec<Call>>>;

/// A predictor with no ONNX weights that records each `predict_batch` call. A state's result
/// carries the state's text length as `input_tokens`, so a test can tell results apart.
struct FakeBatch {
    checkpoint: LayaCheckpoint,
    calls: CallLog,
}

impl LayaPredictor for FakeBatch {
    fn predict(&self, _state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        Ok(fake_result(self.checkpoint, 0, questions))
    }

    fn as_batch_predictor(&self) -> Option<&dyn LayaBatchPredictor> {
        Some(self)
    }
}

impl LayaBatchPredictor for FakeBatch {
    fn predict_batch(
        &self,
        states: &[Value],
        questions: &QuestionSet,
        options: &BatchOptions,
    ) -> Result<Vec<LayaResult>> {
        self.calls.lock().unwrap().push(Call {
            checkpoint: self.checkpoint,
            states: states.to_vec(),
            ids: questions.ids().map(String::from).collect(),
            options: options.clone(),
        });
        Ok(states
            .iter()
            .map(|state| {
                let size = state.as_str().map_or(0, str::len);
                fake_result(self.checkpoint, size as u32, questions)
            })
            .collect())
    }
}

fn fake_result(checkpoint: LayaCheckpoint, tag: u32, questions: &QuestionSet) -> LayaResult {
    LayaResult::new(
        checkpoint.subdir(),
        questions.ids().map(String::from).collect(),
        HashMap::new(),
        Usage {
            input_tokens: tag,
            output_tokens: 0,
        },
    )
}

struct Fixture {
    router: LayaRouter,
    calls: CallLog,
    builds: Arc<Mutex<Vec<LayaCheckpoint>>>,
}

fn fixture() -> Fixture {
    let calls = CallLog::default();
    let builds = Arc::new(Mutex::new(Vec::new()));
    let (factory_calls, factory_builds) = (Arc::clone(&calls), Arc::clone(&builds));
    let options = LayaRouterOptions {
        max_loaded: 3,
        ..LayaRouterOptions::default()
    };
    let router = LayaRouter::with_factory(options, move |checkpoint, _| {
        factory_builds.lock().unwrap().push(checkpoint);
        Ok(Arc::new(FakeBatch {
            checkpoint,
            calls: Arc::clone(&factory_calls),
        }) as Arc<dyn LayaPredictor>)
    })
    .expect("a fake router never fails to construct");
    Fixture {
        router,
        calls,
        builds,
    }
}

fn noul(id: &str) -> QuestionSet {
    QuestionSet::new()
        .with(id, Question::noul("Is it?", Value::Null, Value::Null))
        .unwrap()
}

fn choice(options: &[&str]) -> QuestionSet {
    QuestionSet::new()
        .with(
            "pick",
            Question::choice_labels("Which?", options.iter().copied()).unwrap(),
        )
        .unwrap()
}

#[test]
fn route_batch_loads_nothing_and_keeps_input_order() {
    let f = fixture();
    let questions = Arc::new(noul("flag"));
    let requests = [
        BatchRequest::new(ENGLISH, Arc::clone(&questions)),
        BatchRequest::new("texte", Arc::clone(&questions)).model("multi"),
        BatchRequest::new("anything", Arc::clone(&questions)).lang("en-US"),
        BatchRequest::new("anything", Arc::clone(&questions)).lang("fr"),
        BatchRequest::new("anything", Arc::clone(&questions)).lang_guess("de"),
        BatchRequest::new("anything", Arc::clone(&questions)).task("typed_decisions"),
    ];
    let decisions = f.router.route_batch(&requests).unwrap();
    let models: Vec<_> = decisions.iter().map(|d| d.model).collect();
    assert_eq!(
        models,
        [
            LayaCheckpoint::English,
            LayaCheckpoint::Multilingual,
            LayaCheckpoint::English,
            LayaCheckpoint::Multilingual,
            LayaCheckpoint::Multilingual,
            LayaCheckpoint::TypedDecisions,
        ]
    );
    assert!(
        f.builds.lock().unwrap().is_empty(),
        "route_batch must not load"
    );
    assert!(f.router.loaded().is_empty());
}

#[test]
fn route_batch_honours_each_requests_lang_guess_and_the_installed_one() {
    let questions = Arc::new(noul("flag"));
    let f = fixture();
    let requests = [
        BatchRequest::new("anything", Arc::clone(&questions)).lang_guess("fr"),
        BatchRequest::new("anything", Arc::clone(&questions))
            .lang_guess(LangGuess::callback(|_| Some("en".to_string()))),
        // A hint that names no language abstains; detection then decides.
        BatchRequest::new(ENGLISH, Arc::clone(&questions)).lang_guess("und"),
    ];
    let models: Vec<_> = f
        .router
        .route_batch(&requests)
        .unwrap()
        .iter()
        .map(|d| d.model)
        .collect();
    assert_eq!(
        models,
        [
            LayaCheckpoint::Multilingual,
            LayaCheckpoint::English,
            LayaCheckpoint::English
        ]
    );
}

#[test]
fn route_batch_refuses_a_null_state_by_index_like_python() {
    let f = fixture();
    let questions = Arc::new(noul("flag"));
    let requests = [
        BatchRequest::new("fine", Arc::clone(&questions)),
        BatchRequest::new(Value::Null, Arc::clone(&questions)),
    ];
    let error = f.router.route_batch(&requests).unwrap_err();
    assert!(matches!(error, LayaError::InvalidState(_)), "{error:?}");
    assert_eq!(
        error.to_string(),
        "request 1 'state' must not be None; pass a string, dict, or list"
    );
    let error = f
        .router
        .predict_batch(&requests, &BatchOptions::new())
        .unwrap_err();
    assert!(matches!(error, LayaError::InvalidState(_)), "{error:?}");
    assert!(
        f.builds.lock().unwrap().is_empty(),
        "nothing loads when routing fails"
    );
}

#[test]
fn predict_batch_without_requests_answers_nothing_and_loads_nothing() {
    let f = fixture();
    assert!(
        f.router
            .predict_batch(&[], &BatchOptions::new())
            .unwrap()
            .is_empty()
    );
    assert!(f.builds.lock().unwrap().is_empty());
}

#[test]
fn predict_batch_groups_by_checkpoint_and_restores_input_order() {
    let f = fixture();
    let questions = Arc::new(noul("flag"));
    // Interleaved checkpoints; each state's text length tags its result.
    let requests = [
        BatchRequest::new("a", Arc::clone(&questions)).model("en"),
        BatchRequest::new("bb", Arc::clone(&questions)).model("multi"),
        BatchRequest::new("ccc", Arc::clone(&questions)).model("en"),
        BatchRequest::new("dddd", Arc::clone(&questions)).model("multi"),
        BatchRequest::new("eeeee", Arc::clone(&questions)).model("en"),
    ];
    let results = f
        .router
        .predict_batch(&requests, &BatchOptions::new())
        .unwrap();

    let tags: Vec<u32> = results.iter().map(|r| r.usage().input_tokens).collect();
    assert_eq!(tags, [1, 2, 3, 4, 5], "results follow the request order");
    let models: Vec<&str> = results.iter().map(LayaResult::model).collect();
    assert_eq!(
        models,
        [
            "english",
            "multilingual",
            "english",
            "multilingual",
            "english"
        ]
    );
    for (result, request) in results.iter().zip(&requests) {
        let routing = result.routing().expect("every result carries its routing");
        assert_eq!(
            routing.reason,
            format!(
                "explicit model={}",
                python_repr(request.model.as_deref().unwrap())
            )
        );
    }

    // One load and one forward-pass group per checkpoint, in order of first appearance.
    assert_eq!(
        *f.builds.lock().unwrap(),
        [LayaCheckpoint::English, LayaCheckpoint::Multilingual]
    );
    let calls = f.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].checkpoint, LayaCheckpoint::English);
    assert_eq!(calls[0].states, [json!("a"), json!("ccc"), json!("eeeee")]);
    assert_eq!(calls[1].checkpoint, LayaCheckpoint::Multilingual);
    assert_eq!(calls[1].states, [json!("bb"), json!("dddd")]);
}

fn python_repr(text: &str) -> String {
    format!("'{text}'")
}

#[test]
fn predict_batch_splits_a_checkpoint_by_question_schema() {
    let f = fixture();
    let a = Arc::new(noul("flag"));
    let b = Arc::new(noul("other"));
    let ab = Arc::new(choice(&["x", "y"]));
    let ba = Arc::new(choice(&["y", "x"]));
    let requests = [
        BatchRequest::new("1", Arc::clone(&a)).model("en"),
        BatchRequest::new("22", Arc::clone(&b)).model("en"),
        // Separately built but equal to `a`: shares a pass with request 0 and 4.
        BatchRequest::new("333", noul("flag")).model("en"),
        BatchRequest::new("4444", Arc::clone(&ab)).model("en"),
        BatchRequest::new("55555", Arc::clone(&a)).model("en"),
        // Same ids and labels as `ab` in the other order: options are positional.
        BatchRequest::new("666666", Arc::clone(&ba)).model("en"),
    ];
    let results = f
        .router
        .predict_batch(&requests, &BatchOptions::new())
        .unwrap();

    let tags: Vec<u32> = results.iter().map(|r| r.usage().input_tokens).collect();
    assert_eq!(tags, [1, 2, 3, 4, 5, 6]);
    let calls = f.calls.lock().unwrap();
    let shapes: Vec<(Vec<Value>, Vec<String>)> = calls
        .iter()
        .map(|c| (c.states.clone(), c.ids.clone()))
        .collect();
    assert_eq!(
        shapes,
        [
            (
                vec![json!("1"), json!("333"), json!("55555")],
                vec!["flag".to_string()]
            ),
            (vec![json!("22")], vec!["other".to_string()]),
            (vec![json!("4444")], vec!["pick".to_string()]),
            (vec![json!("666666")], vec!["pick".to_string()]),
        ]
    );
    assert_eq!(
        f.builds.lock().unwrap().len(),
        1,
        "one load for one checkpoint"
    );
}

#[test]
fn predict_batch_forwards_the_options_to_every_group() {
    let f = fixture();
    let questions = Arc::new(noul("flag"));
    let requests = [
        BatchRequest::new("a", Arc::clone(&questions)).model("en"),
        BatchRequest::new("b", Arc::clone(&questions)).model("multi"),
    ];
    let options = BatchOptions::new().batch_size(2).sort_by_length(true);
    f.router.predict_batch(&requests, &options).unwrap();
    let calls = f.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|call| call.options == options));
}

#[test]
fn predict_batch_through_the_predictor_trait_routes_each_state() {
    let f = fixture();
    let questions = noul("flag");
    let batcher = LayaPredictor::as_batch_predictor(&f.router).expect("a router can batch");
    let results = batcher
        .predict_batch(
            &[json!(ENGLISH), json!("Привет, как дела у тебя сегодня?")],
            &questions,
            &BatchOptions::new(),
        )
        .unwrap();
    let models: Vec<&str> = results.iter().map(LayaResult::model).collect();
    assert_eq!(models, ["english", "multilingual"]);
}

struct SingleOnly;

impl LayaPredictor for SingleOnly {
    fn predict(&self, _state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        Ok(fake_result(LayaCheckpoint::English, 0, questions))
    }
}

#[test]
fn a_predictor_without_predict_batch_is_reported_not_emulated() {
    let router = LayaRouter::with_factory(LayaRouterOptions::default(), |_, _| {
        Ok(Arc::new(SingleOnly) as Arc<dyn LayaPredictor>)
    })
    .unwrap();
    let requests = [BatchRequest::new("state", noul("flag")).model("en")];
    let error = router
        .predict_batch(&requests, &BatchOptions::new())
        .unwrap_err();
    assert!(matches!(error, LayaError::NotSupported(_)), "{error:?}");
    assert!(error.to_string().contains("no predict_batch"), "{error}");
}

#[test]
fn a_failing_group_fails_the_call() {
    struct Failing;
    impl LayaPredictor for Failing {
        fn predict(&self, _: Value, _: &QuestionSet) -> Result<LayaResult> {
            Err(LayaError::InvalidConfig("nope".into()))
        }
        fn as_batch_predictor(&self) -> Option<&dyn LayaBatchPredictor> {
            Some(self)
        }
    }
    impl LayaBatchPredictor for Failing {
        fn predict_batch(
            &self,
            _: &[Value],
            _: &QuestionSet,
            _: &BatchOptions,
        ) -> Result<Vec<LayaResult>> {
            Err(LayaError::InvalidConfig("nope".into()))
        }
    }
    let router = LayaRouter::with_factory(LayaRouterOptions::default(), |_, _| {
        Ok(Arc::new(Failing) as Arc<dyn LayaPredictor>)
    })
    .unwrap();
    let requests = [BatchRequest::new("state", noul("flag")).model("en")];
    let error = router
        .predict_batch(&requests, &BatchOptions::new())
        .unwrap_err();
    assert_eq!(error.to_string(), "nope");
}

// ---------------------------------------------------------------------- model-backed

fn batch_options(options: &Value) -> BatchOptions {
    let mut built = BatchOptions::new();
    // Python treats a batch_size that is zero or negative as unset.
    built.batch_size = options
        .get("batch_size")
        .and_then(Value::as_i64)
        .and_then(|size| usize::try_from(size).ok())
        .filter(|&size| size > 0);
    built.sort_by_length = options
        .get("sort_by_length")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    built.min_confidence = golden_data::recorded_min_confidence(options);
    built
}

fn check_batch_case(engine: &LayaEngine, where_case: &str, root: &Value) {
    let states = root["states"].as_array().expect("states");
    let questions = golden_data::build_questions(&root["questions"])
        .unwrap_or_else(|e| panic!("{where_case}: {e}"));
    let options = batch_options(&root["options"]);
    let got = engine
        .predict_batch(states, &questions, &options)
        .unwrap_or_else(|e| panic!("{where_case}: predict_batch failed: {e}"));
    let expected = root["results"].as_array().expect("results");
    assert_eq!(expected.len(), got.len(), "{where_case}: result count");

    for (index, (want, result)) in expected.iter().zip(&got).enumerate() {
        let where_state = format!("{where_case}[{index}]");
        // The SDK keeps its own model identifier instead of the ONNX agent's suffix.
        assert_eq!("laya-rl-agent-onnx", want["model"].as_str().unwrap());
        if questions.is_empty() {
            assert_eq!("laya-rl-agent", result.model(), "{where_state}: model");
            assert!(result.ids().is_empty(), "{where_state}: no answers");
            assert!(
                result.state_usage().is_none(),
                "{where_state}: no state usage"
            );
            assert_eq!(
                0,
                result.usage().input_tokens,
                "{where_state}: input_tokens"
            );
            assert_eq!(
                0,
                result.usage().output_tokens,
                "{where_state}: output_tokens"
            );
            assert_eq!(
                want["usage"].as_object().unwrap().len(),
                2,
                "{where_state}: usage keys"
            );
        } else {
            // The batch goldens come from Python's ONNXAgent, whose family tag is
            // "laya-rl-agent-onnx"; this crate keeps the tag of Python's Agent (and of the
            // plain goldens), "laya-rl-agent", for predict and predict_batch alike.
            assert_eq!(
                "laya-rl-agent-onnx", want["model"],
                "{where_state}: golden model"
            );
            assert_eq!("laya-rl-agent", result.model(), "{where_state}: model");
            let mut want = want.clone();
            want["model"] = json!("laya-rl-agent");
            assert_result_matches(&where_state, &want, result);
            assert_state_usage_matches(&where_state, &want, result);
        }
    }
}

fn run_success_cases(
    load: fn(LayaCheckpoint) -> Option<LayaEngine>,
    label: &str,
    unavailable: &str,
) {
    let mut checked_any = false;
    for checkpoint in CHECKPOINTS {
        let Some(engine) = load(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }
        for case in golden.batch_cases() {
            checked_any = true;
            let root = golden.load_case(&case);
            check_batch_case(
                &engine,
                &format!("{}/{case}{label}", checkpoint.subdir()),
                &root,
            );
        }
    }
    if !checked_any {
        eprintln!("skip: {unavailable}");
    }
}

#[test]
fn reproduces_the_recorded_batch_predictions() {
    run_success_cases(
        engines::load,
        "",
        "no batch golden data + artifacts found for any checkpoint",
    );
}

#[test]
fn reproduces_the_recorded_batch_predictions_for_the_split_layout() {
    run_success_cases(
        engines::load_split,
        " (split layout)",
        "no split-layout ONNX artifacts + batch golden data found for any checkpoint",
    );
}

#[test]
fn rejects_the_same_batch_requests_python_rejects() {
    let mut checked_any = false;
    for checkpoint in CHECKPOINTS {
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            continue;
        }
        for case in golden.batch_error_cases() {
            checked_any = true;
            let root = golden.load_case(&case);
            let questions = golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{}/{case}: {e}", checkpoint.subdir()));
            let states = root["states"].as_array().expect("states");
            let error = engine
                .predict_batch(states, &questions, &batch_options(&root["options"]))
                .expect_err(&format!(
                    "{}/{case}: expected predict_batch to fail",
                    checkpoint.subdir()
                ));
            // Python's TypeError is the refused null state; its ValueError is a question whose
            // options do not fit, which Rust reports as NoMarkers with the same leading text.
            let message = root["error_message"].as_str().unwrap();
            match root["error_type"].as_str().unwrap() {
                "TypeError" => {
                    assert!(
                        matches!(error, LayaError::InvalidState(_)),
                        "{}/{case}: wrong error variant: {error:?}",
                        checkpoint.subdir()
                    );
                    assert_eq!(message, error.to_string(), "{}/{case}", checkpoint.subdir());
                }
                "ValueError" if root["options"].get("min_confidence").is_some() => {
                    // A rejected threshold: Python's text, whole, checked before any state.
                    assert!(
                        matches!(error, LayaError::InvalidMinConfidence(_)),
                        "{}/{case}: wrong error variant: {error:?}",
                        checkpoint.subdir()
                    );
                    assert_eq!(message, error.to_string(), "{}/{case}", checkpoint.subdir());
                }
                "ValueError" => {
                    assert!(
                        matches!(error, LayaError::NoMarkers { .. }),
                        "{}/{case}: wrong error variant: {error:?}",
                        checkpoint.subdir()
                    );
                    assert!(
                        error.to_string().starts_with(message),
                        "{}/{case}: python: {message}
  rust:   {error}",
                        checkpoint.subdir()
                    );
                }
                other => panic!(
                    "{}/{case}: unexpected error type {other}",
                    checkpoint.subdir()
                ),
            }
        }
    }
    if !checked_any {
        eprintln!("skip: no batch error-case golden data + artifacts found for any checkpoint");
    }
}

/// `predict` is a one-state batch in Python (`system_one` is `predict_batch([state], ...)`).
#[test]
fn predict_is_a_one_state_batch_for_null_states_and_empty_questions() {
    let mut checked_any = false;
    for checkpoint in CHECKPOINTS {
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };
        checked_any = true;
        let questions = golden_data::build_questions(&json!({
            "flag": {"type": "noul", "instructions": "Is it?"}
        }))
        .unwrap();
        let error = engine.predict(Value::Null, &questions).unwrap_err();
        assert!(matches!(error, LayaError::InvalidState(_)), "{error:?}");
        assert_eq!(
            error.to_string(),
            "state must not be None; pass a string, dict, or list"
        );

        let empty = engine.predict("some state", &QuestionSet::new()).unwrap();
        assert!(empty.ids().is_empty());
        assert!(empty.state_usage().is_none());
        assert_eq!(0, empty.usage().input_tokens);
        assert_eq!(0, empty.usage().output_tokens);
    }
    if !checked_any {
        eprintln!("skip: no ONNX artifacts found for any checkpoint");
    }
}
