//! Tier 2 (needs ONNX artifacts): the router driving real engines end to end.
//!
//! Run the full suite with:
//! `LAYA_ONNX_ROOT=<parent-of-checkpoints> cargo test --release -- --test-threads=1`
//! `--test-threads=1` matters here even more than in `predict_parity_tests.rs`: the router itself
//! can hold up to two ~1.3-1.7 GB engines resident (English + Multilingual, per the plan's
//! `max_loaded=2` router-predict section), so two of *this* file's tests running at once could
//! hold four engines between them.

mod common;

use common::{golden_data::CheckpointGoldenData, result_compare, routing_golden, test_artifacts};
use laya::laya_engine::LayaEngine;
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{LayaPredictor, LayaRouter, LayaRouterOptions};
use std::sync::Arc;

fn have_english_and_multilingual() -> bool {
    [LayaCheckpoint::English, LayaCheckpoint::Multilingual]
        .into_iter()
        .all(|c| test_artifacts::has_model(c) && test_artifacts::has_tokenizer(c))
}

/// Builds each engine from the directory `test_artifacts` found (the repo walk-up included), not the
/// router's default resolution, which only checks `LAYA_ONNX_ROOT` and the cache: otherwise the
/// skip check above can pass while the router still fails to find the artifacts.
fn new_router(max_loaded: usize) -> LayaRouter {
    LayaRouter::with_factory(
        LayaRouterOptions {
            max_loaded,
            ..LayaRouterOptions::default()
        },
        |checkpoint, opts| {
            let mut opts = opts.clone();
            opts.checkpoint = checkpoint;
            opts.model_directory = test_artifacts::locate(checkpoint);
            Ok(Arc::new(LayaEngine::create(opts)?) as Arc<dyn LayaPredictor>)
        },
    )
    .expect("router construction")
}

#[test]
fn router_predict_routes_and_matches_the_recorded_checkpoint_result() {
    if !have_english_and_multilingual() {
        eprintln!("skip: need both the english and multilingual ONNX artifacts");
        return;
    }

    let router = new_router(2);

    // English state -> english checkpoint, compared against english's own recorded golden.
    let english_golden = CheckpointGoldenData::for_checkpoint(LayaCheckpoint::English.subdir());
    if !english_golden.available() {
        eprintln!("skip: no golden data for 'english'");
        return;
    }
    let english_case = english_golden.load("case_sample_app_english.json");
    let state = english_case["state"].clone();
    let questions = common::golden_data::build_questions(&english_case["questions"])
        .expect("build english questions");

    let result = router
        .predict(state, &questions, None, None, None)
        .expect("router predict (english state)");
    let routing = result.routing().expect("routing must be populated");
    assert_eq!(
        routing.model,
        LayaCheckpoint::English,
        "english state routes to english"
    );
    result_compare::assert_result_matches(
        "router/sample_app_english",
        &english_case["result"],
        &result,
    );

    // Hindi state -> multilingual checkpoint, compared against multilingual's own recorded golden.
    let multilingual_golden =
        CheckpointGoldenData::for_checkpoint(LayaCheckpoint::Multilingual.subdir());
    if !multilingual_golden.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }
    let hindi_case = multilingual_golden.load("case_sample_app_hindi.json");
    let state = hindi_case["state"].clone();
    let questions = common::golden_data::build_questions(&hindi_case["questions"])
        .expect("build hindi questions");

    let result = router
        .predict(state, &questions, None, None, None)
        .expect("router predict (hindi state)");
    let routing = result.routing().expect("routing must be populated");
    assert_eq!(
        routing.model,
        LayaCheckpoint::Multilingual,
        "hindi state routes to multilingual"
    );
    result_compare::assert_result_matches(
        "router/sample_app_hindi",
        &hindi_case["result"],
        &result,
    );
}

#[test]
fn sample_inputs_match_recorded_model_case_inputs() {
    let sample_inputs = routing_golden::load("sample_inputs.json")
        .expect("the routing sample fixture must be present");
    let support = &sample_inputs["support_email"];
    let english_case = CheckpointGoldenData::for_checkpoint("english")
        .load("case_sample_app_english.json");
    let hindi_case = CheckpointGoldenData::for_checkpoint("multilingual")
        .load("case_sample_app_hindi.json");

    assert_eq!(support["english_state"], english_case["state"], "English sample state");
    assert_eq!(support["questions"], english_case["questions"], "English sample questions");
    assert_eq!(support["questions"], hindi_case["questions"], "Hindi sample questions");

    // The routing sample includes Hindi sender/subject headers; the quickstart
    // recording deliberately uses only its body. Compare that exact projection.
    assert!(support["hindi_state"]["body"].is_string(), "Hindi sample body must exist");
    assert_eq!(
        serde_json::json!({"body": support["hindi_state"]["body"]}),
        hindi_case["state"],
        "Hindi sample body-only state"
    );
}

#[test]
fn router_predict_is_correct_under_eight_thread_concurrency_with_max_loaded_one() {
    if !have_english_and_multilingual() {
        eprintln!("skip: need both the english and multilingual ONNX artifacts");
        return;
    }
    let english_golden = CheckpointGoldenData::for_checkpoint(LayaCheckpoint::English.subdir());
    let multilingual_golden =
        CheckpointGoldenData::for_checkpoint(LayaCheckpoint::Multilingual.subdir());
    if !english_golden.available() || !multilingual_golden.available() {
        eprintln!("skip: golden data missing for english or multilingual");
        return;
    }

    let english_case = english_golden.load("case_sample_app_english.json");
    let hindi_case = multilingual_golden.load("case_sample_app_hindi.json");

    let router = Arc::new(new_router(1)); // max_loaded 1 forces an eviction on every alternation

    // 4 threads on the English state, 4 on the Hindi state, all launched together so `max_loaded
    // = 1` forces genuine evictions under load rather than settling onto one checkpoint.
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let router = Arc::clone(&router);
            let (case, expected_model) = if i % 2 == 0 {
                (english_case.clone(), LayaCheckpoint::English)
            } else {
                (hindi_case.clone(), LayaCheckpoint::Multilingual)
            };
            std::thread::spawn(move || {
                let state = case["state"].clone();
                let questions = common::golden_data::build_questions(&case["questions"])
                    .expect("build questions");
                let result = router
                    .predict(state, &questions, None, None, None)
                    .expect("concurrent router predict");
                let routing = result.routing().expect("routing must be populated");
                assert_eq!(
                    routing.model, expected_model,
                    "thread {i}: routed to the expected checkpoint"
                );
                (case, result)
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        let (case, result) = handle.join().unwrap_or_else(|e| {
            panic!("thread {i} panicked (a disposed-engine use would panic or corrupt here): {e:?}")
        });
        result_compare::assert_result_matches(
            &format!("router/concurrent[{i}]"),
            &case["result"],
            &result,
        );
    }
}
