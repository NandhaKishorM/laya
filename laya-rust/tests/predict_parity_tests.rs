//! Tier 2: full answers against the ones Python recorded, which needs the model weights.
//!
//! The goldens come from the real `laya.Agent.system_one` running over the same ONNX graph, so any
//! gap here is this port's, not the model's. Recorded values are rounded to four decimals on both
//! sides, which is why the tolerances sit a little above 1e-4 rather than at zero.
//!
//! Run the full suite (all three checkpoints) with:
//! `LAYA_ONNX_ROOT=<parent-of-checkpoints> cargo test --release -- --test-threads=1`
//! `--test-threads=1` bounds memory: each test below loads one ~1.3-1.7 GB engine per checkpoint
//! into a local variable and drops it before moving to the next, so at most one is ever resident
//! *within* a test, but cargo's default parallel test scheduling could otherwise still run two of
//! these `#[test]` functions — each holding its own engine — at the same time.

mod common;

use common::engines::{self, CHECKPOINTS};
use common::golden_data::{self, CheckpointGoldenData};
use laya::answers::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
use laya::error::LayaError;
use laya::laya_options::LayaCheckpoint;
use laya::python_json::PythonJson;
use laya::questions::QuestionSet;
use laya::sequence_builder::SequenceBuilder;
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Tracks the worst absolute diff seen across every [`assert_close`] call, purely for the final
/// report's "max observed diff vs golden" figure — not a correctness check by itself.
static MAX_DIFF: Mutex<f64> = Mutex::new(0.0);

/// Probabilities and confidences: 4-dp recorded values, so a rounding boundary is 1e-4 wide.
const PROB_TOLERANCE: f64 = 2e-4;
/// A score is a weighted sum over level indices, so it carries several of those errors.
const SCORE_TOLERANCE: f64 = 2e-3;

fn assert_close(expected: f64, actual: f64, tolerance: f64, what: &str) {
    let diff = (expected - actual).abs();
    {
        let mut max_diff = MAX_DIFF.lock().unwrap();
        if diff > *max_diff {
            *max_diff = diff;
        }
    }
    assert!(
        diff <= tolerance,
        "{what}: python {expected}, rust {actual} (diff {diff}, tolerance {tolerance})"
    );
}

#[test]
fn reproduces_the_recorded_answers() {
    let mut checked_any = false;

    for checkpoint in CHECKPOINTS {
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }

        for case in golden.success_cases() {
            checked_any = true;
            let root = golden.load_case(&case);
            let state = root["state"].clone();
            let questions = golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{}/{case}: {e}", checkpoint.subdir()));

            let result = engine
                .predict(state, &questions)
                .unwrap_or_else(|e| panic!("{}/{case}: predict failed: {e}", checkpoint.subdir()));
            let expected = &root["result"];
            let where_case = format!("{}/{case}", checkpoint.subdir());

            assert_eq!(
                expected["model"].as_str().unwrap(),
                result.model(),
                "{where_case}: model"
            );
            assert_eq!(
                expected["usage"]["input_tokens"].as_u64().unwrap() as u32,
                result.usage().input_tokens,
                "{where_case}: usage.input_tokens"
            );
            assert_eq!(
                expected["usage"]["output_tokens"].as_u64().unwrap() as u32,
                result.usage().output_tokens,
                "{where_case}: usage.output_tokens"
            );

            let answers = expected["answers"].as_object().unwrap();
            // Ids in order: a result that answered the right questions in the wrong order would
            // attach every answer to the wrong question, and per-id comparison alone would not
            // notice.
            let expected_ids: Vec<&str> = answers.keys().map(String::as_str).collect();
            assert_eq!(
                expected_ids,
                result.ids(),
                "{where_case}: answer ids in order"
            );

            for (id, want) in answers {
                let got = result
                    .get(id)
                    .unwrap_or_else(|| panic!("{where_case}/{id}: missing answer"));
                let where_q = format!("{where_case}/{id}");

                assert_eq!(
                    want["type"].as_str().unwrap(),
                    SequenceBuilder::type_name(got.question_type()),
                    "{where_q}: type"
                );
                assert_close(
                    want["confidence"].as_f64().unwrap(),
                    got.confidence(),
                    PROB_TOLERANCE,
                    &format!("{where_q}.confidence"),
                );
                assert_close(
                    want["action"]["act_probability"].as_f64().unwrap(),
                    got.action().act_probability,
                    PROB_TOLERANCE,
                    &format!("{where_q}.action.act_probability"),
                );

                match got {
                    Answer::Choice(choice) => assert_choice(want, choice, &where_q),
                    Answer::Score(score) => assert_score(want, score, &where_q),
                    Answer::Noul(noul) => assert_noul(want, noul, &where_q),
                }
            }
        }
    }

    if !checked_any {
        eprintln!("skip: no golden data + artifacts found for any checkpoint");
    } else {
        eprintln!(
            "reproduces_the_recorded_answers: max abs diff across probability/confidence/action/score assertions = {}",
            *MAX_DIFF.lock().unwrap()
        );
    }
}

#[test]
fn reproduces_the_recorded_answers_for_the_split_layout() {
    // Same golden data, same assertions as `reproduces_the_recorded_answers`, but against
    // `<repo>/onnx-split/<ckpt>` (the layout laya-ts's exporter produces: `encoder.onnx` +
    // `head.onnx`) instead of the fused `model.onnx`. Skips per checkpoint exactly like the fused
    // test does when that checkpoint's split-layout export is not present locally — most
    // machines will not have one, and that is expected, not a failure.
    let mut checked_any = false;

    for checkpoint in CHECKPOINTS {
        let Some(engine) = engines::load_split(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }

        for case in golden.success_cases() {
            checked_any = true;
            let root = golden.load_case(&case);
            let state = root["state"].clone();
            let questions = golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{}/{case}: {e}", checkpoint.subdir()));

            let result = engine
                .predict(state, &questions)
                .unwrap_or_else(|e| panic!("{}/{case}: predict failed: {e}", checkpoint.subdir()));
            let expected = &root["result"];
            let where_case = format!("{}/{case} (split layout)", checkpoint.subdir());

            assert_eq!(
                expected["model"].as_str().unwrap(),
                result.model(),
                "{where_case}: model"
            );

            let answers = expected["answers"].as_object().unwrap();
            let expected_ids: Vec<&str> = answers.keys().map(String::as_str).collect();
            assert_eq!(
                expected_ids,
                result.ids(),
                "{where_case}: answer ids in order"
            );

            for (id, want) in answers {
                let got = result
                    .get(id)
                    .unwrap_or_else(|| panic!("{where_case}/{id}: missing answer"));
                let where_q = format!("{where_case}/{id}");

                assert_close(
                    want["confidence"].as_f64().unwrap(),
                    got.confidence(),
                    PROB_TOLERANCE,
                    &format!("{where_q}.confidence"),
                );

                match got {
                    Answer::Choice(choice) => assert_choice(want, choice, &where_q),
                    Answer::Score(score) => assert_score(want, score, &where_q),
                    Answer::Noul(noul) => assert_noul(want, noul, &where_q),
                }
            }
        }
    }

    if !checked_any {
        eprintln!(
            "skip: no split-layout ONNX artifacts + golden data found for any checkpoint under \
             <repo>/onnx-split/"
        );
    }
}

#[test]
fn rejects_the_same_questions_python_rejects() {
    let mut checked_any = false;

    for checkpoint in CHECKPOINTS {
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            continue;
        }

        for case in golden.error_cases() {
            checked_any = true;
            let root = golden.load_case(&case);
            let state = root["state"].clone();
            let questions = golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{}/{case}: {e}", checkpoint.subdir()));

            let error = engine.predict(state, &questions).expect_err(&format!(
                "{}/{case}: expected predict to fail",
                checkpoint.subdir()
            ));
            let expected_message = root["error_message"].as_str().unwrap();
            let actual_message = error.to_string();
            assert!(
                actual_message.starts_with(expected_message),
                "{}/{case}: error message mismatch\n  python: {expected_message}\n  rust:   {actual_message}",
                checkpoint.subdir()
            );
            // The specific variant matters too: this failure mode is always a marker/head-budget
            // problem, never e.g. a missing question id.
            assert!(
                matches!(error, LayaError::NoMarkers { .. }),
                "{}/{case}: expected LayaError::NoMarkers, got {error:?}",
                checkpoint.subdir()
            );
        }
    }

    if !checked_any {
        eprintln!("skip: no error-case golden data + artifacts found for any checkpoint");
    }
}

#[test]
fn answers_the_same_way_when_questions_are_asked_one_at_a_time() {
    // Every question rides in one batch, so a padding or masking mistake could let a question be
    // influenced by its neighbours. Asking them singly has to give the same answers.
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        return;
    };
    let golden = CheckpointGoldenData::for_checkpoint(LayaCheckpoint::Multilingual.subdir());
    if !golden.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }
    let root = golden.load("case_quickstart.json");
    let state = root["state"].clone();
    let questions = golden_data::build_questions(&root["questions"]).expect("build_questions");

    let batched = engine
        .predict(state.clone(), &questions)
        .expect("batched predict");

    for id in questions.ids() {
        let single_set = QuestionSet::new()
            .with(id, questions.get(id).unwrap().clone())
            .expect("single-question set");
        let single = engine
            .predict(state.clone(), &single_set)
            .expect("single predict");
        assert_close(
            batched[id].confidence(),
            single[id].confidence(),
            PROB_TOLERANCE,
            &format!("{id}.confidence"),
        );
    }
}

#[test]
fn is_safe_to_call_from_several_threads_at_once() {
    // The engine is documented as shareable and a caller will hold it as a singleton, so the ONNX
    // session mutex and the tokenizer both have to tolerate concurrent use.
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        return;
    };
    let golden = CheckpointGoldenData::for_checkpoint(LayaCheckpoint::Multilingual.subdir());
    if !golden.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }
    let root = golden.load("case_quickstart.json");
    let state = root["state"].clone();
    let questions = golden_data::build_questions(&root["questions"]).expect("build_questions");

    let engine = Arc::new(engine);
    let want = engine
        .predict(state.clone(), &questions)
        .expect("warm-up predict")
        .get("department")
        .unwrap()
        .as_choice()
        .unwrap()
        .choice
        .clone();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let state = state.clone();
            let questions = questions.clone();
            std::thread::spawn(move || {
                engine
                    .predict(state, &questions)
                    .expect("concurrent predict")
                    .get("department")
                    .unwrap()
                    .as_choice()
                    .unwrap()
                    .choice
                    .clone()
            })
        })
        .collect();

    for handle in handles {
        let choice = handle.join().expect("thread panicked");
        assert_eq!(want, choice);
    }
}

#[test]
fn warm_predict_latency_is_reported() {
    // Not a parity check: reports the warm (post-warm-up) predict latency for the final report,
    // the way samples/Laya.Sample's Program.cs does with a Stopwatch. Run with `--nocapture` to
    // see the printed line.
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        return;
    };
    let golden = CheckpointGoldenData::for_checkpoint(LayaCheckpoint::Multilingual.subdir());
    if !golden.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }
    let root = golden.load("case_quickstart.json");
    let state = root["state"].clone();
    let questions = golden_data::build_questions(&root["questions"]).expect("build_questions");

    // First call pays for ONNX Runtime session warm-up.
    let cold_start = Instant::now();
    engine
        .predict(state.clone(), &questions)
        .expect("cold predict");
    let cold = cold_start.elapsed();

    let warm_start = Instant::now();
    engine.predict(state, &questions).expect("warm predict");
    let warm = warm_start.elapsed();

    eprintln!(
        "warm_predict_latency_is_reported: multilingual cold={cold:?} warm={warm:?} (quickstart, {} questions)",
        questions.len()
    );
}

// ── helpers ───────────────────────────────────────────────────────────────

fn assert_choice(want: &Value, got: &ChoiceAnswer, where_q: &str) {
    let probabilities = want["probabilities"].as_object().unwrap();
    // Labels in order, because the entire marker layout rests on labels being positional.
    let expected_labels: Vec<&str> = probabilities.keys().map(String::as_str).collect();
    let actual_labels: Vec<&str> = got.probabilities.keys().map(String::as_str).collect();
    assert_eq!(
        expected_labels, actual_labels,
        "{where_q}: probability label order"
    );

    for (label, value) in probabilities {
        assert_close(
            value.as_f64().unwrap(),
            got.probability(label).unwrap(),
            PROB_TOLERANCE,
            &format!("{where_q}.probabilities[{label}]"),
        );
    }

    // Checked after the distribution: when the top two are close the argmax is the fragile part,
    // and the distribution mismatch is the more informative failure to report.
    assert_eq!(
        want["choice"].as_str().unwrap(),
        got.choice,
        "{where_q}: choice"
    );
    let sum: f64 = got.probabilities.values().sum();
    assert_close(1.0, sum, 1e-3, &format!("{where_q}: probabilities sum"));
}

fn assert_score(want: &Value, got: &ScoreAnswer, where_q: &str) {
    // Python's JSON keys both the legend and the distribution by the stringified level index.
    let legend = want["legend"].as_object().unwrap();
    assert_eq!(legend.len(), got.legend.len(), "{where_q}: legend length");
    for i in 0..got.legend.len() {
        let expected_level = &legend[&i.to_string()];
        assert_eq!(
            PythonJson::criterion(expected_level),
            PythonJson::criterion(&got.legend[i]),
            "{where_q}: legend[{i}]"
        );
    }

    let probabilities = want["probabilities"].as_object().unwrap();
    assert_eq!(
        probabilities.len(),
        got.probabilities.len(),
        "{where_q}: probabilities length"
    );
    for i in 0..got.probabilities.len() {
        assert_close(
            probabilities[&i.to_string()].as_f64().unwrap(),
            got.probabilities[i],
            PROB_TOLERANCE,
            &format!("{where_q}.probabilities[{i}]"),
        );
    }

    assert_close(
        want["score"].as_f64().unwrap(),
        got.score,
        SCORE_TOLERANCE,
        &format!("{where_q}.score"),
    );
}

fn assert_noul(want: &Value, got: &NoulAnswer, where_q: &str) {
    assert_close(
        want["noul"].as_f64().unwrap(),
        got.probability,
        PROB_TOLERANCE,
        &format!("{where_q}.noul"),
    );
}
