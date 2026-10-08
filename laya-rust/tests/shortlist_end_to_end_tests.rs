//! Tier 2 (needs ONNX artifacts): the shortlist driving a real engine end to end, for all three
//! checkpoints. Ports the `shortlist_many_options` golden case (`"kind": "shortlist"` in each
//! checkpoint's `index.json`, kept out of the ordinary predict-parity cases by the `golden_data`
//! fix described in the routing-port notes).
//!
//! Run the full suite with:
//! `LAYA_ONNX_ROOT=<parent-of-checkpoints> cargo test --release -- --test-threads=1`

mod common;

use common::{engines, golden_data::CheckpointGoldenData, hashing_embedder, result_compare};
use laya::laya_options::LayaCheckpoint;
use laya::laya_shortlist::LayaShortlist;

#[test]
fn shortlist_predict_matches_the_recorded_result_for_every_checkpoint() {
    let mut checked_any = false;

    for checkpoint in engines::CHECKPOINTS {
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }

        for case_info in golden.shortlist_cases() {
            checked_any = true;
            let root = golden.load_case(&case_info);
            let where_case = format!("{}/{case_info}", checkpoint.subdir());

            let state = root["state"].clone();
            let full_questions = common::golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{where_case}: build full questions: {e}"));
            let k = root["shortlist_k"]
                .as_u64()
                .unwrap_or_else(|| panic!("{where_case}: missing 'shortlist_k'"))
                as usize;

            let result =
                LayaShortlist::predict(&engine, state, &full_questions, hashing_embedder::embed, k)
                    .unwrap_or_else(|e| panic!("{where_case}: LayaShortlist::predict failed: {e}"));

            // The kept, rank-ordered labels for "intent" must match Python's recorded reduction
            // exactly (`reduced_questions.intent.criteria`'s key order).
            let expected_labels: Vec<&str> = root["reduced_questions"]["intent"]["criteria"]
                .as_object()
                .unwrap_or_else(|| {
                    panic!("{where_case}: 'reduced_questions.intent.criteria' must be an object")
                })
                .keys()
                .map(String::as_str)
                .collect();
            let shortlist_meta = result.shortlist().unwrap_or_else(|| {
                panic!("{where_case}: LayaResult::shortlist() must be populated")
            });
            let intent_meta = shortlist_meta
                .get("intent")
                .unwrap_or_else(|| panic!("{where_case}: no shortlist metadata for 'intent'"));
            assert_eq!(
                intent_meta.labels, expected_labels,
                "{where_case}: shortlisted labels vs reduced_questions"
            );
            assert!(
                !intent_meta.passthrough,
                "{where_case}: 40 options with k={k} must not pass through"
            );
            assert_eq!(intent_meta.n, 40, "{where_case}: option count");

            // The answer itself (kept labels' probabilities, chosen answer, confidence) must
            // match the recorded predict-on-the-reduced-question result.
            result_compare::assert_result_matches(&where_case, &root["result"], &result);
        }
    }

    if !checked_any {
        eprintln!("skip: no shortlist golden data + artifacts found for any checkpoint");
    }
}

/// Sanity check that every checkpoint's shortlist case actually reduced from 40 to 20 options
/// (the number the plan's combined-sample section names), not some other pair that happens to
/// still pass the assertions above.
#[test]
fn shortlist_many_options_case_shape_is_forty_reduced_to_twenty() {
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        eprintln!("skip: no multilingual artifacts");
        return;
    };
    let golden = CheckpointGoldenData::for_checkpoint(LayaCheckpoint::Multilingual.subdir());
    if !golden.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }
    let Some(case_info) = golden.shortlist_cases().into_iter().next() else {
        eprintln!("skip: no shortlist case recorded");
        return;
    };
    let root = golden.load_case(&case_info);
    let full_questions =
        common::golden_data::build_questions(&root["questions"]).expect("build questions");
    assert_eq!(root["shortlist_k"].as_u64().unwrap(), 20);
    let result = LayaShortlist::predict(
        &engine,
        root["state"].clone(),
        &full_questions,
        hashing_embedder::embed,
        20,
    )
    .expect("predict");
    let meta = &result.shortlist().unwrap()["intent"];
    assert_eq!(meta.n, 40);
    assert_eq!(meta.k, 20);
    assert_eq!(meta.labels.len(), 20);
}
