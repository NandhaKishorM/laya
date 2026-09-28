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
use laya::laya_router::{LayaPredictor, LayaRouter, LayaRouterOptions};
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
    LayaRouter::new(options).expect("router construction never fails (preload is off)")
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

        let decision = router
            .route(state, questions.as_ref(), model, task, lang)
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
