//! Tier 1: the checkpoint router. Ports the routing/LRU/attach/preload/unload/thread-safety
//! checks from `tests/test_router.py` (everything except the `repo`-carrying assertions: this
//! port's `RouteDecision` intentionally drops that field — see `laya_router.rs`'s doc comment)
//! plus every entry of the routing golden's `route_probe.json`.

mod common;

use common::golden_data;
use common::routing_golden;
use laya::answers::{LayaResult, Usage};
use laya::error::Result;
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{
    BatchRequest, GuessInput, LangGuess, LayaPredictor, LayaRouter, LayaRouterOptions,
};
use laya::questions::QuestionSet;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A predictor with no ONNX weights: records its own checkpoint name and, optionally, sleeps
/// before answering (to widen a race window in the concurrency tests). Ports Python's `_Stub`.
struct FakePredictor {
    name: &'static str,
    delay: std::time::Duration,
}

impl LayaPredictor for FakePredictor {
    fn predict(&self, _state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        Ok(LayaResult::new(
            self.name,
            questions.ids().map(String::from).collect(),
            HashMap::new(),
            Usage::default(),
        ))
    }
}

fn checkpoint_name(c: LayaCheckpoint) -> &'static str {
    c.subdir()
}

/// A router whose checkpoints are [`FakePredictor`]s, so LRU/attach/preload/unload tests need no
/// ONNX weights. `build_log` records the checkpoint (and build order) each factory call built, so
/// tests can assert exactly how many times — and for which checkpoints — the factory ran.
fn fake_router(max_loaded: usize, build_log: Arc<Mutex<Vec<LayaCheckpoint>>>) -> LayaRouter {
    fake_router_with_delay(max_loaded, build_log, std::time::Duration::ZERO)
}

fn fake_router_with_delay(
    max_loaded: usize,
    build_log: Arc<Mutex<Vec<LayaCheckpoint>>>,
    delay: std::time::Duration,
) -> LayaRouter {
    let options = LayaRouterOptions {
        max_loaded,
        ..LayaRouterOptions::default()
    };
    LayaRouter::with_factory(options, move |checkpoint, _opts| {
        build_log.lock().unwrap().push(checkpoint);
        Ok(Arc::new(FakePredictor {
            name: checkpoint_name(checkpoint),
            delay,
        }) as Arc<dyn LayaPredictor>)
    })
    .expect("fake router construction never fails (preload is off)")
}

// ---------------------------------------------------------------------- name normalisation

#[test]
fn normalise_name_resolves_aliases() {
    let cases = [
        ("en", LayaCheckpoint::English),
        ("laya", LayaCheckpoint::English),
        ("multi", LayaCheckpoint::Multilingual),
        ("ML", LayaCheckpoint::Multilingual),
        ("typed", LayaCheckpoint::TypedDecisions),
        ("typed_decisions", LayaCheckpoint::TypedDecisions),
        ("English", LayaCheckpoint::English),
        ("laya-typed-decisions", LayaCheckpoint::TypedDecisions),
        ("laya-multilingual", LayaCheckpoint::Multilingual),
        ("default", LayaCheckpoint::English),
        ("decisions", LayaCheckpoint::TypedDecisions),
    ];
    for (alias, want) in cases {
        assert_eq!(
            LayaRouter::normalise_name(alias).unwrap(),
            want,
            "alias/{alias}"
        );
    }
    assert!(
        LayaRouter::normalise_name("nope").is_err(),
        "alias/unknown raises"
    );
}

// ---------------------------------------------------------------------- workflow signatures

#[test]
fn workflow_signatures_match_exactly() {
    let td: [(&str, &[&str]); 4] = [
        (
            "agent_trace_observability",
            &["action", "needs_review", "outcome", "risk", "urgency"],
        ),
        (
            "customer_service",
            &["action", "category", "churn_risk", "needs_human", "urgency"],
        ),
        (
            "invoice_processing",
            &[
                "discrepancy_severity",
                "disposition",
                "duplicate",
                "matches_order",
                "urgency",
            ],
        ),
        (
            "security_incidents",
            &[
                "credential_compromise",
                "disposition",
                "severity",
                "true_positive",
                "urgency",
            ],
        ),
    ];
    for (workflow, ids) in td {
        let questions = noul_question_set(ids);
        assert_eq!(
            LayaRouter::match_typed_decisions_workflow(&questions).as_deref(),
            Some(workflow),
            "workflow/{workflow}"
        );
    }

    let partial = noul_question_set(&["urgency", "category"]);
    assert_eq!(
        LayaRouter::match_typed_decisions_workflow(&partial),
        None,
        "workflow/partial overlap"
    );

    let superset = noul_question_set(&[
        "action",
        "category",
        "churn_risk",
        "needs_human",
        "urgency",
        "extra",
    ]);
    assert_eq!(
        LayaRouter::match_typed_decisions_workflow(&superset),
        None,
        "workflow/superset"
    );

    let empty = QuestionSet::new();
    assert_eq!(
        LayaRouter::match_typed_decisions_workflow(&empty),
        None,
        "workflow/empty"
    );
}

fn noul_question_set(ids: &[&str]) -> QuestionSet {
    let mut set = QuestionSet::new();
    for id in ids {
        set = set
            .with(
                *id,
                laya::questions::Question::noul("x", Value::Null, Value::Null),
            )
            .unwrap();
    }
    set
}

// ---------------------------------------------------------------------- LRU bookkeeping

#[test]
fn lru_cap_one_keeps_newest() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(1, log);
    router.load("english").unwrap();
    router.load("multilingual").unwrap();
    assert_eq!(
        router.loaded(),
        vec![LayaCheckpoint::Multilingual],
        "lru/cap 1 keeps newest"
    );
}

#[test]
fn default_keeps_two_checkpoints_like_python() {
    // Python's Router(max_loaded=2) since #180: alternating two languages must not reload.
    assert_eq!(LayaRouterOptions::default().max_loaded, 2);
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(LayaRouterOptions::default().max_loaded, log.clone());
    router.load("english").unwrap();
    router.load("multilingual").unwrap();
    router.load("english").unwrap();
    assert_eq!(router.loaded().len(), 2, "both checkpoints stay resident");
    assert_eq!(
        log.lock().unwrap().len(),
        2,
        "switching back does not rebuild english"
    );
}

#[test]
fn lru_cap_two_evicts_oldest() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(2, log);
    router.load("english").unwrap();
    router.load("multilingual").unwrap();
    router.load("typed-decisions").unwrap();
    assert_eq!(
        router.loaded(),
        vec![LayaCheckpoint::Multilingual, LayaCheckpoint::TypedDecisions],
        "lru/cap 2 evicts oldest"
    );
}

#[test]
fn lru_touch_protects_from_eviction() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(2, log);
    router.load("english").unwrap();
    router.load("multilingual").unwrap();
    router.load("english").unwrap(); // touch english
    router.load("typed-decisions").unwrap();
    let mut loaded = router.loaded();
    loaded.sort_by_key(|c| checkpoint_name(*c));
    assert_eq!(
        loaded,
        vec![LayaCheckpoint::English, LayaCheckpoint::TypedDecisions],
        "lru/touch protects"
    );
}

#[test]
fn unload_one_and_all() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(2, log);
    router.load("english").unwrap();
    router.load("multilingual").unwrap();
    router.unload(Some("english")).unwrap();
    assert!(
        !router.loaded().contains(&LayaCheckpoint::English),
        "lru/unload one"
    );
    router.unload(None).unwrap();
    assert!(router.loaded().is_empty(), "lru/unload all");
}

// ---------------------------------------------------------------------- attach

#[test]
fn attach_registers_and_protects_from_eviction() {
    let log = Arc::new(Mutex::new(Vec::new()));
    // Python's equivalent test starts from `max_loaded=1` and then explicitly bumps
    // `max_loaded` to 2 before loading a second checkpoint (`attach` alone only guarantees room
    // for what is already attached, not for a load that has not happened yet). This router is
    // simply constructed with room for two from the start, since this port has no public
    // `max_loaded` setter to mutate after construction.
    let router = fake_router(2, log);
    let sentinel: Arc<dyn LayaPredictor> = Arc::new(FakePredictor {
        name: "already-built",
        delay: std::time::Duration::ZERO,
    });
    let attached = router.attach("english", Arc::clone(&sentinel)).unwrap();
    assert!(Arc::ptr_eq(&attached, &sentinel));
    assert!(
        router.loaded().contains(&LayaCheckpoint::English),
        "attach/counts as resident"
    );

    // A later load of a second checkpoint must not evict the attached one.
    router.load("multilingual").unwrap();
    let mut loaded = router.loaded();
    loaded.sort_by_key(|c| checkpoint_name(*c));
    assert_eq!(
        loaded,
        vec![LayaCheckpoint::English, LayaCheckpoint::Multilingual],
        "attach/survives a later load"
    );

    // Still the exact same object attach() was given, not a rebuilt one.
    let engine = router.load("english").unwrap();
    assert!(
        Arc::ptr_eq(&engine, &sentinel),
        "attach/still the same object"
    );
}

#[test]
fn attach_accepts_aliases() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(1, log);
    let sentinel: Arc<dyn LayaPredictor> = Arc::new(FakePredictor {
        name: "x",
        delay: std::time::Duration::ZERO,
    });
    assert!(
        router.attach("en", sentinel).is_ok(),
        "attach/accepts aliases"
    );
}

// ---------------------------------------------------------------------- preload

#[test]
fn preload_all_three_stay_resident_and_raises_max_loaded() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(1, log);
    router.preload(None).unwrap();
    let mut loaded = router.loaded();
    loaded.sort_by_key(|c| checkpoint_name(*c));
    assert_eq!(
        loaded,
        vec![
            LayaCheckpoint::English,
            LayaCheckpoint::Multilingual,
            LayaCheckpoint::TypedDecisions
        ],
        "preload/all three stay resident"
    );
}

#[test]
fn preload_subset_stays_resident_and_touch_does_not_evict() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(1, log.clone());
    router.preload(Some(&["english", "multilingual"])).unwrap();
    let mut loaded = router.loaded();
    loaded.sort_by_key(|c| checkpoint_name(*c));
    assert_eq!(
        loaded,
        vec![LayaCheckpoint::English, LayaCheckpoint::Multilingual],
        "preload/subset stays resident"
    );
    let builds_before = log.lock().unwrap().len();
    // Routing to an already-resident checkpoint must not evict anything, and must not rebuild.
    router.load("english").unwrap();
    assert_eq!(
        log.lock().unwrap().len(),
        builds_before,
        "preload/touch does not evict or rebuild"
    );
    let mut loaded = router.loaded();
    loaded.sort_by_key(|c| checkpoint_name(*c));
    assert_eq!(
        loaded,
        vec![LayaCheckpoint::English, LayaCheckpoint::Multilingual]
    );
}

// ---------------------------------------------------------------------- thread safety

#[test]
fn concurrent_load_of_the_same_checkpoint_builds_once() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = Arc::new(fake_router_with_delay(
        1,
        log.clone(),
        std::time::Duration::from_millis(20),
    ));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let router = Arc::clone(&router);
            std::thread::spawn(move || router.load("english").unwrap())
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Every caller got the same object.
    for r in &results[1..] {
        assert!(
            Arc::ptr_eq(&results[0], r),
            "threads/8 concurrent loads share one predictor"
        );
    }
    assert_eq!(
        log.lock().unwrap().len(),
        1,
        "threads/predictor constructed exactly once"
    );
    assert_eq!(
        router.loaded(),
        vec![LayaCheckpoint::English],
        "threads/LRU view stays consistent"
    );
}

#[test]
fn predict_survives_eviction_of_its_own_checkpoint() {
    // max_loaded=1 with a slow predict on "english": while it is running, another thread loads
    // "multilingual", which evicts "english" from the router's map. The in-flight predict must
    // still complete successfully against the engine it already captured a clone of.
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = Arc::new(fake_router_with_delay(
        1,
        log,
        std::time::Duration::from_millis(100),
    ));
    router.load("english").unwrap();

    let predicting = {
        let router = Arc::clone(&router);
        std::thread::spawn(move || {
            let mut qs = QuestionSet::new();
            qs = qs
                .with(
                    "q",
                    laya::questions::Question::noul("x", Value::Null, Value::Null),
                )
                .unwrap();
            router.predict(json!("hello"), &qs, Some("english"), None, None)
        })
    };

    // Give the predict a moment to start, then evict "english" by loading a second checkpoint.
    std::thread::sleep(std::time::Duration::from_millis(20));
    router.load("multilingual").unwrap();
    assert!(
        !router.loaded().contains(&LayaCheckpoint::English),
        "english should have been evicted by the load above"
    );

    let result = predicting
        .join()
        .unwrap()
        .expect("in-flight predict must still succeed");
    assert_eq!(result.model(), "english");
}

// ---------------------------------------------------------------------- route_probe.json

fn router_from_kwargs(kwargs: &Value) -> LayaRouter {
    let mut options = LayaRouterOptions::default();
    if let Some(default) = kwargs.get("default").and_then(Value::as_str) {
        options.default = LayaRouter::normalise_name(default).unwrap();
    }
    if let Some(auto) = kwargs.get("auto_task_detection").and_then(Value::as_bool) {
        options.auto_task_detection = auto;
    }
    // `models` / `standalone_repos` router_kwargs only affect the `repo` field Python's
    // RouteDecision carries; this port's RouteDecision has no `repo` field (see laya_router.rs),
    // so those two knobs have no observable effect on `model`/`reason`/`detection`/`workflow` and
    // are intentionally not threaded through here.
    let router =
        LayaRouter::new(options).expect("router construction never fails (preload is off)");
    match kwargs.get("lang_guess").and_then(Value::as_str) {
        Some(code) => router.with_lang_guess(code),
        None => router,
    }
}

#[test]
fn route_probe_matches_every_entry() {
    let Some(cases) = routing_golden::load_array("route_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };

    for case in &cases {
        let label = case["label"].as_str().unwrap();
        let router = router_from_kwargs(&case["router_kwargs"]);
        let route_kwargs = &case["route_kwargs"];
        let state = &route_kwargs["state"];
        let questions = route_kwargs
            .get("questions")
            .filter(|q| !q.is_null())
            .map(|q| golden_data::build_questions(q).unwrap());
        let model = route_kwargs.get("model").and_then(Value::as_str);
        let task = route_kwargs.get("task").and_then(Value::as_str);
        let lang = route_kwargs.get("lang").and_then(Value::as_str);

        let lang_guess = route_kwargs
            .get("lang_guess")
            .and_then(Value::as_str)
            .map(LangGuess::from);

        let decision = router
            .route_with_lang_guess(
                state,
                questions.as_ref(),
                model,
                task,
                lang,
                lang_guess.as_ref(),
            )
            .unwrap_or_else(|e| panic!("{label}: route failed: {e}"));

        let expected = &case["decision"];
        assert_eq!(
            checkpoint_name(decision.model),
            expected["model"].as_str().unwrap(),
            "{label}: model"
        );
        assert_eq!(
            decision.reason,
            expected["reason"].as_str().unwrap(),
            "{label}: reason"
        );
        assert_eq!(
            decision.workflow.as_deref(),
            expected["workflow"].as_str(),
            "{label}: workflow"
        );

        match (&decision.detection, expected.get("detection")) {
            (None, Some(Value::Null)) | (None, None) => {}
            (Some(got), Some(want)) if !want.is_null() => {
                assert_eq!(
                    got.script,
                    want["script"].as_str().unwrap(),
                    "{label}: detection.script"
                );
                assert_eq!(
                    got.language.as_deref(),
                    want["language"].as_str(),
                    "{label}: detection.language"
                );
                assert_eq!(
                    got.is_english,
                    want["is_english"].as_bool().unwrap(),
                    "{label}: detection.is_english"
                );
                assert_eq!(
                    got.mixed_segment.as_deref(),
                    want["mixed_segment"].as_str(),
                    "{label}: detection.mixed_segment"
                );
                assert_eq!(
                    got.language_undecided,
                    want["language_undecided"].as_bool().unwrap(),
                    "{label}: detection.language_undecided"
                );
            }
            (got, want) => {
                panic!("{label}: detection presence mismatch: got {got:?}, want {want:?}")
            }
        }
    }
    eprintln!(
        "route_probe_matches_every_entry: checked {} cases",
        cases.len()
    );
}

// ---------------------------------------------------------------------- route_batch_probe.json

fn batch_request(request: &Value) -> BatchRequest {
    let questions = golden_data::build_questions(&request["questions"]).unwrap();
    let mut built = BatchRequest::new(request["state"].clone(), questions);
    built.model = request
        .get("model")
        .and_then(Value::as_str)
        .map(String::from);
    built.task = request
        .get("task")
        .and_then(Value::as_str)
        .map(String::from);
    built.lang = request
        .get("lang")
        .and_then(Value::as_str)
        .map(String::from);
    built.lang_guess = request
        .get("lang_guess")
        .and_then(Value::as_str)
        .map(LangGuess::from);
    built
}

fn batch_requests(case: &Value) -> Vec<BatchRequest> {
    case["requests"]
        .as_array()
        .unwrap()
        .iter()
        .map(batch_request)
        .collect()
}

#[test]
fn route_batch_probe_matches_every_entry() {
    let Some(probe) = routing_golden::load("route_batch_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };

    let cases = probe["cases"].as_array().unwrap();
    for case in cases {
        let label = case["label"].as_str().unwrap();
        let router = router_from_kwargs(&case["router_kwargs"]);
        let decisions = router
            .route_batch(&batch_requests(case))
            .unwrap_or_else(|e| panic!("{label}: route_batch failed: {e}"));
        let expected = case["decisions"].as_array().unwrap();
        assert_eq!(expected.len(), decisions.len(), "{label}: decision count");
        for (index, (decision, want)) in decisions.iter().zip(expected).enumerate() {
            assert_eq!(
                checkpoint_name(decision.model),
                want["model"].as_str().unwrap(),
                "{label}[{index}]: model"
            );
            assert_eq!(
                decision.reason,
                want["reason"].as_str().unwrap(),
                "{label}[{index}]: reason"
            );
            assert_eq!(
                decision.workflow.as_deref(),
                want["workflow"].as_str(),
                "{label}[{index}]: workflow"
            );
            assert_eq!(
                decision.detection.as_ref().map(|d| d.script.as_str()),
                want["detection"]["script"].as_str(),
                "{label}[{index}]: detection.script"
            );
        }
    }
    eprintln!(
        "route_batch_probe_matches_every_entry: checked {} cases",
        cases.len()
    );
}

/// Python's request validation errors. A missing `state` or `questions`, a `questions` that is not
/// a mapping and a request that is not a mapping cannot be written with `BatchRequest`, so those
/// entries are unrepresentable in Rust and are checked to be exactly that set.
#[test]
fn route_batch_probe_errors_match_python_or_cannot_be_expressed() {
    let Some(probe) = routing_golden::load("route_batch_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };

    let mut compared = Vec::new();
    for case in probe["errors"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let requests = case["requests"].as_array().unwrap();
        let expressible = requests.iter().all(|request| {
            request.is_object()
                && request.get("state").is_some()
                && request.get("questions").is_some_and(Value::is_object)
        });
        if !expressible {
            continue;
        }
        let router = router_from_kwargs(&case["router_kwargs"]);
        let error = router.route_batch(&batch_requests(case)).expect_err(label);
        assert_eq!(
            case["error_message"].as_str().unwrap(),
            error.to_string(),
            "{label}: error message"
        );
        compared.push(label.to_string());
    }
    assert_eq!(["state_none", "unknown_model"], compared.as_slice());
}

/// Python's `Router.predict` raises `TypeError` for a `None` state before it routes or loads, and
/// `Router.predict_long` goes through `predict`, so neither may build a checkpoint for one.
#[test]
fn predict_and_predict_long_refuse_a_null_state_before_routing() {
    let message = "state must not be None; pass a string, dict, or list";
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(2, log.clone());
    let questions = noul_question_set(&["q"]);

    let error = router
        .predict(Value::Null, &questions, None, None, None)
        .expect_err("predict refuses null");
    assert_eq!(message, error.to_string());
    let error = router
        .predict(Value::Null, &questions, Some("bogus"), None, None)
        .expect_err("the state is checked before the model name");
    assert_eq!(message, error.to_string());
    let error = router
        .predict_long(
            Value::Null,
            &questions,
            &laya::long_prediction::LongOptions::new(),
            None,
            None,
            None,
        )
        .expect_err("predict_long refuses null");
    assert_eq!(message, error.to_string());
    assert!(log.lock().unwrap().is_empty(), "no checkpoint is loaded");
}

// ---------------------------------------------------------------------- lang_guess (callbacks)
//
// The golden can only carry string hints; callbacks (a language-identification model) are pinned
// here. Mirrors the callable cases of `tests/test_lang_guess.py`.

const ROMANIAN: &str = "Care este ora in Tokyo?";

fn route_model(router: &LayaRouter, state: &str, guess: Option<&LangGuess>) -> LayaCheckpoint {
    router
        .route_with_lang_guess(&json!(state), None, None, None, None, guess)
        .unwrap()
        .model
}

#[test]
fn lang_guess_callback_code_is_used_and_receives_the_state() {
    let router = LayaRouter::new(LayaRouterOptions::default()).unwrap();
    let to_romanian = LangGuess::callback(|_| Some("ro".to_string()));
    assert_eq!(
        route_model(&router, ROMANIAN, Some(&to_romanian)),
        LayaCheckpoint::Multilingual
    );
    let by_content = LangGuess::callback(|state| {
        Some(
            if matches!(state, GuessInput::Value(v) if v.as_str().is_some_and(|s| s.contains("Tokyo"))) {
                "en"
            } else {
                "ro"
            }
            .to_string(),
        )
    });
    assert_eq!(
        route_model(&router, ROMANIAN, Some(&by_content)),
        LayaCheckpoint::English
    );
}

#[test]
fn lang_guess_callback_that_abstains_falls_through_to_detection() {
    let router = LayaRouter::new(LayaRouterOptions::default()).unwrap();
    let baseline = router
        .route(&json!(ROMANIAN), None, None, None, None)
        .unwrap();
    for abstain in [
        LangGuess::callback(|_| None),
        LangGuess::callback(|_| Some(String::new())),
        LangGuess::callback(|_| Some("C".to_string())),
    ] {
        let decision = router
            .route_with_lang_guess(&json!(ROMANIAN), None, None, None, None, Some(&abstain))
            .unwrap();
        assert_eq!(decision.model, baseline.model);
        assert_eq!(decision.reason, baseline.reason);
        assert!(
            decision.detection.is_some(),
            "an abstaining hint must leave detection intact"
        );
    }
}

#[test]
fn installed_lang_guess_applies_to_every_request_and_per_call_overrides_it() {
    let router = LayaRouter::new(LayaRouterOptions::default())
        .unwrap()
        .with_lang_guess("ro");
    assert_eq!(
        route_model(&router, ROMANIAN, None),
        LayaCheckpoint::Multilingual
    );
    let english = LangGuess::from("en");
    assert_eq!(
        route_model(&router, ROMANIAN, Some(&english)),
        LayaCheckpoint::English
    );
    // a per-call hint that abstains leaves the installed one in force
    let abstain = LangGuess::from("POSIX");
    assert_eq!(
        route_model(&router, ROMANIAN, Some(&abstain)),
        LayaCheckpoint::Multilingual
    );
    let decision = router
        .route(&json!(ROMANIAN), None, None, None, None)
        .unwrap();
    assert_eq!(
        decision.reason,
        "Router(lang_guess=...): the caller identified this as non-English text"
    );
    assert!(decision.detection.is_none());
}

#[test]
fn installed_callback_works_and_lang_guess_is_off_by_default() {
    let plain = LayaRouter::new(LayaRouterOptions::default()).unwrap();
    assert_eq!(route_model(&plain, ROMANIAN, None), LayaCheckpoint::English);
    let router = plain.with_lang_guess(LangGuess::callback(|_| Some("ro".into())));
    assert_eq!(
        route_model(&router, ROMANIAN, None),
        LayaCheckpoint::Multilingual
    );
}

#[test]
fn explicit_model_task_and_lang_beat_lang_guess() {
    let router = LayaRouter::new(LayaRouterOptions::default()).unwrap();
    let ro = LangGuess::from("ro");
    let by_model = router
        .route_with_lang_guess(
            &json!(ROMANIAN),
            None,
            Some("english"),
            None,
            None,
            Some(&ro),
        )
        .unwrap();
    assert_eq!(by_model.model, LayaCheckpoint::English);
    let by_task = router
        .route_with_lang_guess(
            &json!(ROMANIAN),
            None,
            None,
            Some("typed_decisions"),
            None,
            Some(&ro),
        )
        .unwrap();
    assert_eq!(by_task.model, LayaCheckpoint::TypedDecisions);
    let by_lang = router
        .route_with_lang_guess(&json!(ROMANIAN), None, None, None, Some("en"), Some(&ro))
        .unwrap();
    assert_eq!(by_lang.model, LayaCheckpoint::English);
    assert_eq!(by_lang.reason, "explicit lang='en'");
}

#[test]
fn lang_guess_is_forwarded_by_predict() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = fake_router(2, log).with_lang_guess("ro");
    let questions = noul_question_set(&["a"]);
    let result = router
        .predict(json!(ROMANIAN), &questions, None, None, None)
        .unwrap();
    assert_eq!(result.model(), "multilingual");
    let english = LangGuess::from("en");
    let result = router
        .predict_with_lang_guess(
            json!(ROMANIAN),
            &questions,
            None,
            None,
            None,
            Some(&english),
        )
        .unwrap();
    assert_eq!(result.model(), "english");
}

#[test]
fn blank_and_language_agnostic_lang_codes_fall_through_to_detection() {
    let router = LayaRouter::new(LayaRouterOptions::default()).unwrap();
    for lang in [
        "", "   ", "C", "c.utf-8", "POSIX", "und", "zxx", "mul", ".", ".UTF-8",
    ] {
        let decision = router
            .route(
                &json!("Hello there my friend"),
                None,
                None,
                None,
                Some(lang),
            )
            .unwrap();
        assert!(
            decision.detection.is_some(),
            "lang {lang:?} names no language, so detection must decide"
        );
    }
    for (lang, model) in [
        ("en_US.UTF-8", LayaCheckpoint::English),
        ("EN", LayaCheckpoint::English),
        (" english ", LayaCheckpoint::English),
        ("pt-BR", LayaCheckpoint::Multilingual),
        ("qq", LayaCheckpoint::Multilingual),
    ] {
        let decision = router
            .route(&json!("hola"), None, None, None, Some(lang))
            .unwrap();
        assert_eq!(decision.model, model, "lang {lang:?}");
        assert!(decision.detection.is_none());
    }
}

#[test]
fn mixed_segment_reason_truncates_to_sixty_characters() {
    let router = LayaRouter::new(LayaRouterOptions::default()).unwrap();
    // enough English that the whole-state read stays English and the segment scan decides
    let english =
        "Please review the error and send me the details of this and what you would like to do.\n"
            .repeat(6);
    let state = json!(format!(
        "{english}Eu nao consigo entrar na minha conta e depois eu nao sei mais o que fazer."
    ));
    let decision = router.route(&state, None, None, None, None).unwrap();
    assert_eq!(decision.model, LayaCheckpoint::Multilingual);
    let mixed = decision.detection.unwrap().mixed_segment.unwrap();
    assert!(mixed.chars().count() > 60);
    let expected: String = mixed.chars().take(60).collect();
    assert_eq!(
        decision.reason,
        format!(
            "Latin script, mostly English, but a line or field reads as 'pt' ('{expected}'); \
             the English checkpoint cannot read it"
        )
    );
}
