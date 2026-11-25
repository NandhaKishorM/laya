//! The Python recordings for per-language temperatures and the per-call `lang`, `max_len` and
//! `head_max_len`: the model-free probes `routing/lang_temperatures_probe.json` (Python's own
//! `resolve_lang_temperatures` over 84 inputs) and `routing/router_forwarding_probe.json` (what
//! Python's `Router` hands the agent it routed to), and a guard that the model-backed cases are
//! recorded. The model-backed cases themselves run in the ordinary parity suites
//! (`predict_parity_tests`, `predict_batch_tests`, `predict_long_tests`), on engines built with the
//! table the goldens were recorded with (`common::engines`).

mod common;

use common::engines::CHECKPOINTS;
use common::golden_data::{self, CheckpointGoldenData};
use common::routing_golden;
use laya::answers::{LayaResult, Usage};
use laya::batch_prediction::BatchOptions;
use laya::error::{LayaError, Result};
use laya::lang_temperatures::resolve_lang_temperatures_probe;
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{
    BatchRequest, LangGuess, LayaBatchPredictor, LayaLongPredictor, LayaPredictor, LayaRouter,
    LayaRouterOptions,
};
use laya::long_prediction::LongOptions;
use laya::predict_options::PredictOptions;
use laya::questions::QuestionSet;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------- lang_temperatures_probe

#[test]
fn the_resolver_matches_pythons_on_every_typed_case() {
    let Some(probe) = routing_golden::load("lang_temperatures_probe.json") else {
        eprintln!("skip: routing golden not found (lang_temperatures_probe.json)");
        return;
    };
    let mut checked = 0;
    for case in probe["cases"].as_array().unwrap() {
        // A case a typed table cannot state (ints, bools, strings, a non-mapping entry, a key
        // that is no string) pins Python's own spellings and refusals; the typed API cannot
        // express it, so it is not run.
        if !case["typed"].as_bool().unwrap() {
            continue;
        }
        let label = case["label"].as_str().unwrap();
        let table = golden_data::recorded_lang_table(&case["pairs"]);
        let (warnings, resolved) = resolve_lang_temperatures_probe(&table, &case["base"]);
        checked += 1;

        let expected: Vec<&str> = case["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| {
                assert_eq!("RuntimeWarning", w["category"].as_str().unwrap());
                w["message"].as_str().unwrap()
            })
            .collect();
        assert_eq!(expected, warnings, "{label}: warnings");

        if let Some(message) = case.get("error_message") {
            assert_eq!(
                "ValueError",
                case["error_type"].as_str().unwrap(),
                "{label}"
            );
            match resolved {
                Err(LayaError::InvalidConfig(text)) => {
                    assert_eq!(message.as_str().unwrap(), text, "{label}: error")
                }
                other => panic!("{label}: expected the ValueError, got {other:?}"),
            }
            continue;
        }
        let resolved = resolved.unwrap_or_else(|e| panic!("{label}: {e}"));
        let want = case["result"].as_object().unwrap();
        let mut languages: Vec<&str> = want.keys().map(String::as_str).collect();
        languages.sort_unstable();
        assert_eq!(languages, resolved.languages(), "{label}: languages");
        for (language, entry) in want {
            let got = resolved.get(language).unwrap();
            let temperature: Vec<f64> = entry["temperature"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            assert_eq!(
                temperature, got.temperature,
                "{label}/{language}: temperature"
            );
            let buckets = entry["temperature_by_options"].as_object().unwrap();
            assert_eq!(
                buckets.len(),
                got.temperature_by_options.len(),
                "{label}/{language}: bucket count"
            );
            for (bucket, value) in buckets {
                assert_eq!(
                    Some(&value.as_f64().unwrap()),
                    got.temperature_by_options.get(bucket),
                    "{label}/{language}/{bucket}"
                );
            }
        }
    }
    assert!(checked >= 50, "only {checked} typed cases ran");
}

// ---------------------------------------------------------------------- router_forwarding_probe

#[derive(Debug, Clone)]
enum Seen {
    Predict(PredictOptions),
    Long(LongOptions),
    Batch(&'static str, Vec<Value>, BatchOptions),
}

struct Recorder {
    name: &'static str,
    lang_temperatures: bool,
    seen: Arc<Mutex<Vec<Seen>>>,
}

fn answered(name: &str, questions: &QuestionSet) -> LayaResult {
    LayaResult::new(
        name,
        questions.ids().map(String::from).collect(),
        HashMap::new(),
        Usage::default(),
    )
}

impl LayaPredictor for Recorder {
    fn predict(&self, _: Value, questions: &QuestionSet) -> Result<LayaResult> {
        Ok(answered(self.name, questions))
    }
    fn predict_with_options(
        &self,
        _: Value,
        questions: &QuestionSet,
        options: &PredictOptions,
    ) -> Result<LayaResult> {
        self.seen
            .lock()
            .unwrap()
            .push(Seen::Predict(options.clone()));
        Ok(answered(self.name, questions))
    }
    fn has_lang_temperatures(&self) -> bool {
        self.lang_temperatures
    }
    fn as_batch_predictor(&self) -> Option<&dyn LayaBatchPredictor> {
        Some(self)
    }
    fn as_long_predictor(&self) -> Option<&dyn LayaLongPredictor> {
        Some(self)
    }
}

impl LayaBatchPredictor for Recorder {
    fn predict_batch(
        &self,
        states: &[Value],
        questions: &QuestionSet,
        options: &BatchOptions,
    ) -> Result<Vec<LayaResult>> {
        self.seen
            .lock()
            .unwrap()
            .push(Seen::Batch(self.name, states.to_vec(), options.clone()));
        Ok(states
            .iter()
            .map(|_| answered(self.name, questions))
            .collect())
    }
}

impl LayaLongPredictor for Recorder {
    fn predict_long(
        &self,
        _: Value,
        questions: &QuestionSet,
        options: &LongOptions,
    ) -> Result<LayaResult> {
        self.seen.lock().unwrap().push(Seen::Long(options.clone()));
        Ok(answered(self.name, questions))
    }
}

fn router(has: &HashMap<&str, bool>) -> (LayaRouter, Arc<Mutex<Vec<Seen>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handle = Arc::clone(&seen);
    let has: HashMap<String, bool> = has.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    let router = LayaRouter::with_factory(LayaRouterOptions::default(), move |checkpoint, _| {
        let name = checkpoint.subdir();
        Ok(Arc::new(Recorder {
            name,
            lang_temperatures: has.get(name).copied().unwrap_or(false),
            seen: Arc::clone(&handle),
        }) as Arc<dyn LayaPredictor>)
    })
    .unwrap();
    (router, seen)
}

fn string_arg(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

fn probe() -> Option<Value> {
    let probe = routing_golden::load("router_forwarding_probe.json");
    if probe.is_none() {
        eprintln!("skip: routing golden not found (router_forwarding_probe.json)");
    }
    probe
}

fn routed_model(result: &LayaResult) -> &'static str {
    result.routing().expect("a routed result").model.subdir()
}

#[test]
fn router_predict_forwards_what_pythons_router_forwards() {
    let Some(probe) = probe() else { return };
    let questions = golden_data::build_questions(&serde_json::json!({
        "q": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    }))
    .unwrap();
    for case in probe["predict"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let args = &case["args"];
        let (router, seen) = router(&HashMap::new());
        let guess = string_arg(args, "lang_guess").map(LangGuess::from);
        let mut options = PredictOptions::new();
        options.max_len = golden_data::recorded_budget(args, "max_len");
        options.head_max_len = golden_data::recorded_budget(args, "head_max_len");
        let result = router
            .predict_with_options(
                args["state"].as_str().unwrap(),
                &questions,
                args.get("model").and_then(Value::as_str),
                None,
                args.get("lang").and_then(Value::as_str),
                guess.as_ref(),
                &options,
            )
            .unwrap();
        assert_eq!(
            case["model"].as_str().unwrap(),
            routed_model(&result),
            "{label}"
        );
        let seen = seen.lock().unwrap();
        let [Seen::Predict(got)] = &seen[..] else {
            panic!("{label}: one predict call expected, got {seen:?}");
        };
        let received = &case["received"];
        assert_eq!(string_arg(received, "lang"), got.lang, "{label}: lang");
        assert_eq!(
            golden_data::recorded_budget(received, "max_len"),
            got.max_len,
            "{label}: max_len"
        );
        assert_eq!(
            golden_data::recorded_budget(received, "head_max_len"),
            got.head_max_len,
            "{label}: head_max_len"
        );
    }
}

#[test]
fn router_predict_long_forwards_what_pythons_router_forwards() {
    let Some(probe) = probe() else { return };
    let questions = golden_data::build_questions(&serde_json::json!({
        "q": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    }))
    .unwrap();
    for case in probe["long"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let args = &case["args"];
        let (router, seen) = router(&HashMap::new());
        let guess = string_arg(args, "lang_guess").map(LangGuess::from);
        let result = router
            .predict_long_with_lang_guess(
                args["state"].as_str().unwrap(),
                &questions,
                &LongOptions::new(),
                None,
                None,
                args.get("lang").and_then(Value::as_str),
                guess.as_ref(),
            )
            .unwrap();
        assert_eq!(
            case["model"].as_str().unwrap(),
            routed_model(&result),
            "{label}"
        );
        // A scan whose `predict_long` has no `lang` parameter (`stub_takes_lang: false`) runs
        // without it and Python warns. A Rust predictor always receives `LongOptions`, so that
        // shape cannot arise; the call and the routing are still checked.
        if args.get("stub_takes_lang").and_then(Value::as_bool) == Some(false) {
            continue;
        }
        let seen = seen.lock().unwrap();
        let [Seen::Long(got)] = &seen[..] else {
            panic!("{label}: one scan expected, got {seen:?}");
        };
        assert_eq!(
            string_arg(&case["received"], "lang"),
            got.lang,
            "{label}: lang"
        );
    }
}

#[test]
fn router_predict_batch_groups_and_forwards_what_pythons_router_does() {
    let Some(probe) = probe() else { return };
    for case in probe["batch"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let has: HashMap<&str, bool> = case["has_lang_temperatures"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_bool().unwrap()))
            .collect();
        let (router, seen) = router(&has);
        let requests: Vec<BatchRequest> = case["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|request| {
                let questions = golden_data::build_questions(&request["questions"]).unwrap();
                let mut built =
                    BatchRequest::new(request["state"].as_str().unwrap(), Arc::new(questions));
                built.model = string_arg(request, "model");
                built.lang = string_arg(request, "lang");
                built.lang_guess = string_arg(request, "lang_guess").map(LangGuess::from);
                built.max_len = golden_data::recorded_budget(request, "max_len");
                built.head_max_len = golden_data::recorded_budget(request, "head_max_len");
                built
            })
            .collect();
        let results = router
            .predict_batch(&requests, &BatchOptions::new())
            .unwrap();
        let models: Vec<&str> = results.iter().map(routed_model).collect();
        let want_models: Vec<&str> = case["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap())
            .collect();
        assert_eq!(want_models, models, "{label}: models");

        let seen = seen.lock().unwrap();
        let want_calls = case["calls"].as_array().unwrap();
        assert_eq!(want_calls.len(), seen.len(), "{label}: call count");
        for (index, (want, got)) in want_calls.iter().zip(seen.iter()).enumerate() {
            let Seen::Batch(model, states, options) = got else {
                panic!("{label}: expected a batch call, got {got:?}");
            };
            let where_call = format!("{label}[{index}]");
            assert_eq!(
                want["model"].as_str().unwrap(),
                *model,
                "{where_call}: model"
            );
            assert_eq!(
                want["states"].as_array().unwrap(),
                states,
                "{where_call}: states"
            );
            let kwargs = &want["kwargs"];
            assert_eq!(
                string_arg(kwargs, "lang"),
                options.lang,
                "{where_call}: lang"
            );
            assert_eq!(
                golden_data::recorded_budget(kwargs, "max_len"),
                options.max_len,
                "{where_call}: max_len"
            );
            assert_eq!(
                golden_data::recorded_budget(kwargs, "head_max_len"),
                options.head_max_len,
                "{where_call}: head_max_len"
            );
        }
    }
}

// ---------------------------------------------------------------------- model-backed cases

/// The per-language and budget cases are in each checkpoint's golden index once it is regenerated
/// (`tools/regen_golden.py`), and from then on the ordinary parity suites run them. Until then
/// there is nothing to run for a checkpoint, which this says rather than passes silently.
#[test]
fn the_model_backed_cases_are_recorded_for_every_checkpoint() {
    const NAMES: [&str; 8] = [
        "lang_replace_de",
        "lang_region_tag",
        "lang_clamped",
        "budget_max_len",
        "budget_head_negative",
        "budget_max_len_negative",
        "batch_lang_region_tag",
        "long_lang_scan",
    ];
    for checkpoint in CHECKPOINTS {
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }
        let recorded: Vec<String> = golden.index().into_iter().map(|c| c.name).collect();
        if !recorded.iter().any(|name| name == "lang_replace_de") {
            eprintln!(
                "skip: '{}' goldens predate the per-language cases; regenerate with \
                 tools/regen_golden.py --checkpoint {}",
                checkpoint.subdir(),
                checkpoint.subdir()
            );
            continue;
        }
        for name in NAMES {
            assert!(
                recorded.iter().any(|recorded| recorded == name),
                "{}: golden case {name} is missing",
                checkpoint.subdir()
            );
        }
        assert!(
            !golden.lang_table().is_empty(),
            "{}: the recorded table",
            checkpoint.subdir()
        );
        let _ = LayaCheckpoint::default();
    }
}
