//! `predict_long`: the model-free pieces (router capability check, tokenizer trait default) and the
//! model-backed parity of every recorded `"kind": "long"` golden case.
//!
//! The goldens come from the genuine `ONNXAgent.predict_long` over the same ONNX graph. Besides the
//! answers, window attribution and usage, the cases record the tokenized state length, the
//! effective window/stride/room and each window's decoded text, which pin the token offsets, the
//! decoder's special-token retention and the budget arithmetic independently of rounded scores.
//!
//! Run with `LAYA_ONNX_ROOT` (and `LAYA_ONNX_SPLIT_ROOT` for the split layout) set and
//! `--test-threads=1`, as for the other model-backed tests.

mod common;

use common::engines;
use common::golden_data::{self, CheckpointGoldenData};
use common::result_compare::assert_answers_match;
use common::stub_tokenizer::StubTokenizer;
use laya::answers::{LayaResult, Usage, WindowInfo};
use laya::error::LayaError;
use laya::laya_engine::LayaEngine;
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{LayaPredictor, LayaRouter, LayaRouterOptions};
use laya::long_prediction::{LongOptions, window_budget};
use laya::python_json::PythonJson;
use laya::questions::{Question, QuestionSet};
use laya::tokenization::{HfTokenizer, LayaTokenizer};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------- model-free

#[test]
fn a_tokenizer_without_decode_reports_it_instead_of_breaking_the_trait() {
    // `StubTokenizer` implements only the required methods; `decode` is the trait's default.
    let error = StubTokenizer::new().decode(&[1, 2, 3]).unwrap_err();
    assert!(matches!(error, LayaError::Tokenizer(_)), "got {error:?}");
}

struct EncodeOnlyPredictor;

impl LayaPredictor for EncodeOnlyPredictor {
    fn predict(&self, _state: Value, questions: &QuestionSet) -> laya::Result<LayaResult> {
        Ok(LayaResult::new(
            "fake",
            questions.ids().map(String::from).collect(),
            HashMap::new(),
            Usage::default(),
        ))
    }
}

#[test]
fn a_router_never_silently_truncates_through_a_predictor_without_predict_long() {
    let router = LayaRouter::with_factory(LayaRouterOptions::default(), |_, _| {
        Ok(Arc::new(EncodeOnlyPredictor) as Arc<dyn LayaPredictor>)
    })
    .expect("router with a fake factory");
    let questions = QuestionSet::new()
        .with("flag", Question::noul("Is it?", Value::Null, Value::Null))
        .unwrap();

    let error = router
        .predict_long(
            "some state",
            &questions,
            &LongOptions::new(),
            Some("en"),
            None,
            None,
        )
        .unwrap_err();
    assert!(
        matches!(error, LayaError::NotSupported(_)),
        "expected NotSupported, got {error:?}"
    );
    assert!(error.to_string().contains("no predict_long"), "{error}");
}

#[test]
fn long_options_treat_zero_like_unset_and_take_chained_setters() {
    let options = LongOptions::new().window(64).stride(32).batch_size(2);
    assert_eq!(options.window, Some(64));
    assert_eq!(options.stride, Some(32));
    assert_eq!(options.batch_size, Some(2));
    assert_eq!(options.aggregate, None);
    assert_eq!(LongOptions::default(), LongOptions::new());
}

// ---------------------------------------------------------------------- model-backed parity

fn option_usize(options: &Value, name: &str) -> Option<usize> {
    options
        .get(name)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
}

fn long_options(options: &Value) -> LongOptions {
    let mut built = LongOptions::new();
    built.window = option_usize(options, "window");
    built.stride = option_usize(options, "stride");
    built.batch_size = option_usize(options, "batch_size");
    built.aggregate = options
        .get("aggregate")
        .and_then(Value::as_str)
        .map(str::to_string);
    built.lang = golden_data::recorded_lang(options);
    built
}

fn expect_usize(value: &Value, what: &str) -> usize {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{what}: expected a non-negative integer, got {value}"))
        as usize
}

fn check_long_usage(where_case: &str, want: &Value, got: &LayaResult) {
    assert_eq!(
        expect_usize(&want["input_tokens"], "input_tokens") as u32,
        got.usage().input_tokens,
        "{where_case}: usage.input_tokens"
    );
    assert_eq!(
        expect_usize(&want["output_tokens"], "output_tokens") as u32,
        got.usage().output_tokens,
        "{where_case}: usage.output_tokens"
    );
    let scan = got
        .scan_usage()
        .unwrap_or_else(|| panic!("{where_case}: predict_long must report scan usage"));
    assert_eq!(
        expect_usize(&want["windows"], "windows"),
        scan.windows,
        "{where_case}: usage.windows"
    );

    let Some(state) = got.state_usage() else {
        // Only a call with no questions has no per-state diagnostics, and Python then omits them.
        for key in [
            "state_tokens",
            "state_tokens_dropped",
            "truncated",
            "truncated_questions",
        ] {
            assert!(
                want.get(key).is_none(),
                "{where_case}: usage.{key} missing from result"
            );
        }
        return;
    };
    assert_eq!(
        expect_usize(&want["state_tokens"], "state_tokens"),
        state.state_tokens,
        "{where_case}: usage.state_tokens"
    );
    assert_eq!(
        expect_usize(&want["state_tokens_dropped"], "state_tokens_dropped"),
        state.state_tokens_dropped,
        "{where_case}: usage.state_tokens_dropped"
    );
    // Python reports a bool for one pass and a count of truncated windows for a scan.
    let truncated = match &want["truncated"] {
        Value::Bool(b) => usize::from(*b),
        other => expect_usize(other, "truncated"),
    };
    assert_eq!(
        truncated, scan.truncated_windows,
        "{where_case}: usage.truncated"
    );
    assert_eq!(
        truncated > 0,
        state.truncated,
        "{where_case}: truncated flag"
    );
    let ids: Vec<&str> = want["truncated_questions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap())
        .collect();
    let got_ids: Vec<&str> = state
        .truncated_questions
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(ids, got_ids, "{where_case}: usage.truncated_questions");

    let want_options = want.get("options").and_then(Value::as_object);
    assert_eq!(
        want_options.map_or(0, |o| o.len()),
        state.options.len(),
        "{where_case}: usage.options count"
    );
    if let Some(want_options) = want_options {
        let want_ids: Vec<&str> = want_options.keys().map(String::as_str).collect();
        let got_ids: Vec<&str> = state.options.keys().map(String::as_str).collect();
        assert_eq!(want_ids, got_ids, "{where_case}: usage.options ids");
        for (id, expected) in want_options {
            let got = &state.options[id];
            assert_eq!(
                expect_usize(&expected["total"], "total"),
                got.total,
                "{where_case}/{id}: options.total"
            );
            assert_eq!(
                expect_usize(&expected["distinct"], "distinct"),
                got.distinct,
                "{where_case}/{id}: options.distinct"
            );
            assert_eq!(
                expected["tokens_per_option"].as_u64().map(|v| v as usize),
                got.tokens_per_option,
                "{where_case}/{id}: options.tokens_per_option"
            );
        }
    }
}

fn check_windows(where_case: &str, want: &Value, got: &LayaResult) {
    let answers = want["answers"].as_object().unwrap();
    for (id, expected) in answers {
        let answer = got
            .get(id)
            .unwrap_or_else(|| panic!("{where_case}/{id}: missing answer"));
        match expected.get("window") {
            Some(window) => assert_eq!(
                Some(WindowInfo {
                    index: expect_usize(&window["index"], "index"),
                    token_start: expect_usize(&window["token_start"], "token_start"),
                    token_end: expect_usize(&window["token_end"], "token_end"),
                    count: expect_usize(&window["count"], "count"),
                }),
                answer.window(),
                "{where_case}/{id}: window attribution"
            ),
            None => assert_eq!(
                None,
                answer.window(),
                "{where_case}/{id}: single-pass answer must carry no window"
            ),
        }
    }
}

fn check_long_case(engine: &LayaEngine, where_case: &str, root: &Value) {
    let state = root["state"].clone();
    let questions = golden_data::build_questions(&root["questions"])
        .unwrap_or_else(|e| panic!("{where_case}: {e}"));
    let options = &root["options"];
    let got = engine
        .predict_long(state.clone(), &questions, &long_options(options))
        .unwrap_or_else(|e| panic!("{where_case}: predict_long failed: {e}"));
    let expected = &root["result"];

    // The SDK keeps its own model identifier instead of the ONNX agent's suffix.
    assert_eq!("laya-rl-agent-onnx", expected["model"].as_str().unwrap());
    assert_eq!("laya-rl-agent", got.model(), "{where_case}: model");
    check_long_usage(where_case, &expected["usage"], &got);
    assert_answers_match(where_case, expected, &got);
    check_windows(where_case, expected, &got);

    // These comparisons pin token offsets, decoder cleanup and special-token retention
    // independently of model probabilities (a dropped token could otherwise round away).
    let artifacts = engine.artifacts();
    let tokenizer = HfTokenizer::from_file_with_config(
        &artifacts.tokenizer_path,
        artifacts.tokenizer_config_path.as_deref(),
    )
    .unwrap_or_else(|e| panic!("{where_case}: tokenizer: {e}"));
    let mask_token = tokenizer.special().mask_token.clone();
    let state_ids = tokenizer
        .encode(&PythonJson::state(&state).replace(&mask_token, " "))
        .unwrap();
    assert_eq!(
        expect_usize(&root["state_tokens"], "state_tokens"),
        state_ids.len(),
        "{where_case}: state token count"
    );
    let config = engine.config();
    let budget = window_budget(
        &tokenizer,
        &questions,
        config.max_len,
        config.head_max_len,
        option_usize(options, "window").unwrap_or(0),
        option_usize(options, "stride").unwrap_or(0),
    )
    .unwrap_or_else(|e| panic!("{where_case}: window_budget: {e}"));
    assert_eq!(
        expect_usize(&root["effective_window"], "effective_window"),
        budget.window,
        "{where_case}: effective window"
    );
    assert_eq!(
        expect_usize(&root["effective_stride"], "effective_stride"),
        budget.stride,
        "{where_case}: effective stride"
    );
    assert_eq!(
        expect_usize(&root["effective_room"], "effective_room"),
        budget.room,
        "{where_case}: effective room"
    );
    for window in root["windows"].as_array().unwrap() {
        let start = expect_usize(&window["token_start"], "token_start");
        let end = expect_usize(&window["token_end"], "token_end");
        assert_eq!(
            window["text"].as_str().unwrap(),
            tokenizer.decode(&state_ids[start..end]).unwrap(),
            "{where_case}: decoded window {start}..{end}"
        );
    }

    let windows = got.scan_usage().unwrap().windows;
    match root["name"].as_str().unwrap() {
        "long_exact_limit" => {
            assert_eq!(
                budget.window,
                state_ids.len(),
                "{where_case}: state at limit"
            );
            assert_eq!(1, windows, "{where_case}: exact-limit state is one pass");
        }
        "long_room_fast_path" => {
            assert!(
                state_ids.len() > budget.window,
                "{where_case}: state must exceed the default window"
            );
            assert_eq!(budget.room, state_ids.len(), "{where_case}: state at room");
            assert_eq!(1, windows, "{where_case}: room-sized state is one pass");
        }
        _ => {}
    }

    if windows == 1 && !questions.is_empty() {
        // The one-pass shortcut is the ordinary predict path, so it must agree with it exactly,
        // with the same language (Python's predict_long passes lang to system_one).
        let mut one_pass = laya::predict_options::PredictOptions::new();
        if let Some(lang) = root["options"]["lang"].as_str() {
            one_pass = one_pass.lang(lang);
        }
        let plain = engine
            .predict_with_options(state, &questions, &one_pass)
            .expect("plain predict");
        assert_eq!(plain.usage(), got.usage(), "{where_case}: one-pass usage");
        for id in questions.ids() {
            assert_eq!(
                plain[id].confidence(),
                got[id].confidence(),
                "{where_case}/{id}: one-pass confidence"
            );
            assert_eq!(
                plain[id].answer_confidence(),
                got[id].answer_confidence(),
                "{where_case}/{id}: one-pass answer_confidence"
            );
            assert_eq!(
                plain[id].action(),
                got[id].action(),
                "{where_case}/{id}: one-pass action"
            );
        }
    }
}

fn run_success_cases(
    load: fn(LayaCheckpoint) -> Option<LayaEngine>,
    label: &str,
    unavailable: &str,
) {
    let mut checked_any = false;
    for checkpoint in engines::checkpoints() {
        let Some(engine) = load(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }
        for case in golden.long_cases() {
            checked_any = true;
            let root = golden.load_case(&case);
            check_long_case(
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
fn reproduces_the_recorded_long_predictions() {
    run_success_cases(
        engines::load,
        "",
        "no long golden data + artifacts found for any checkpoint",
    );
}

#[test]
fn reproduces_the_recorded_long_predictions_for_the_split_layout() {
    run_success_cases(
        engines::load_split,
        " (split layout)",
        "no split-layout ONNX artifacts + long golden data found for any checkpoint",
    );
}

#[test]
fn rejects_the_same_long_requests_python_rejects() {
    let mut checked_any = false;
    for checkpoint in engines::checkpoints() {
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            continue;
        }
        for case in golden.long_error_cases() {
            checked_any = true;
            let root = golden.load_case(&case);
            let questions = golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{}/{case}: {e}", checkpoint.subdir()));
            let error = engine
                .predict_long(
                    root["state"].clone(),
                    &questions,
                    &long_options(&root["options"]),
                )
                .expect_err(&format!(
                    "{}/{case}: expected predict_long to fail",
                    checkpoint.subdir()
                ));
            let expected_message = root["error_message"].as_str().unwrap();
            let actual_message = error.to_string();
            assert!(
                actual_message.starts_with(expected_message),
                "{}/{case}: error message mismatch\n  python: {expected_message}\n  rust:   {actual_message}",
                checkpoint.subdir()
            );
            let name = root["name"].as_str().unwrap();
            let right_variant = match name {
                "long_invalid_aggregate" => matches!(error, LayaError::UnsupportedAggregate(_)),
                _ => matches!(error, LayaError::InvalidWindow(_)),
            };
            assert!(
                right_variant,
                "{}/{case}: wrong error variant: {error:?}",
                checkpoint.subdir()
            );
        }
    }
    if !checked_any {
        eprintln!("skip: no long error-case golden data + artifacts found for any checkpoint");
    }
}

// ---------------------------------------------------------------------- deferred config types

/// Hard-links every file under `from` into `to` (the weights are over a gigabyte), copying only
/// files small enough that a copy is harmless when linking is not possible.
fn link_tree(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let (source, target) = (entry.path(), to.join(entry.file_name()));
        if source.is_dir() {
            link_tree(&source, &target);
        } else if std::fs::hard_link(&source, &target).is_err() {
            let size = std::fs::metadata(&source).unwrap().len();
            assert!(
                size < 64 * 1024 * 1024,
                "cannot hard-link {} ({size} bytes) into the scratch checkpoint",
                source.display()
            );
            std::fs::copy(&source, &target).unwrap();
        }
    }
}

struct ScratchDir(std::path::PathBuf);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Checkpoint configs whose `max_len` / `head_max_len` / `temperature` Python reads loosely. A
/// wrongly typed or negative length is refused when the engine is created, with the `TypeError`
/// text Python raises at the first prediction; a `temperature` list shorter than three loads and
/// fails with Python's `IndexError` for a question type whose index it lacks; a longer one is
/// fine. The messages were produced by Python's `common.build_sequence` and `_decode_answers`.
#[test]
fn config_lengths_and_temperatures_are_refused_or_deferred_as_agreed() {
    if !common::test_artifacts::is_selected(LayaCheckpoint::English) {
        return;
    }
    let Some(source) = common::test_artifacts::locate(LayaCheckpoint::English).filter(|dir| {
        dir.join("model.onnx").is_file() && dir.join("tokenizer").join("tokenizer.json").is_file()
    }) else {
        eprintln!("skip: no ONNX artifacts for 'english'; set LAYA_ONNX_ROOT");
        return;
    };
    let config_text = std::fs::read_to_string(source.join("rl_agent_config.json")).unwrap();
    let questions = QuestionSet::new()
        .with(
            "flag",
            Question::noul("Is it urgent?", Value::Null, Value::Null),
        )
        .unwrap();
    let scratch = ScratchDir(
        std::env::temp_dir().join(format!("laya-deferred-length-{}", std::process::id())),
    );
    let operand =
        |kind: &str| format!("TypeError: unsupported operand type(s) for -: '{kind}' and 'int'");
    let make = |change: Option<(&str, Value)>| {
        let _ = std::fs::remove_dir_all(&scratch.0);
        link_tree(&source, &scratch.0);
        let config_path = scratch.0.join("rl_agent_config.json");
        std::fs::remove_file(&config_path).unwrap();
        let mut config: Value = serde_json::from_str(&config_text).unwrap();
        if let Some((key, value)) = change {
            config[key] = value;
        }
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();
        LayaEngine::from_directory(&scratch.0)
    };

    // Refused at create / from_directory.
    let refused = [
        (("max_len", Value::from("512")), operand("str")),
        (("head_max_len", Value::Null), operand("NoneType")),
        (
            ("max_len", Value::from(300.5)),
            "TypeError: slice indices must be integers or None or have an __index__ method"
                .to_string(),
        ),
        (
            ("max_len", Value::from(-5)),
            "max_len=-5: must be a non-negative integer".to_string(),
        ),
    ];
    for (change, expected) in refused {
        match make(Some(change)) {
            Err(LayaError::InvalidConfig(got)) => assert_eq!(expected, got),
            Err(other) => panic!("expected InvalidConfig({expected}), got {other:?}"),
            Ok(_) => panic!("expected InvalidConfig({expected}), but the engine loaded"),
        }
    }

    // Loads; the short temperature list fails predict and predict_long (a noul question needs
    // index 2); a longer list and the shipped config predict normally.
    let cases = [
        (
            Some(("temperature", serde_json::json!([1.0]))),
            Some("IndexError: list index out of range"),
        ),
        (
            Some(("temperature", serde_json::json!([1.0, 1.0, 1.0, 9.0]))),
            None,
        ),
        (None, None),
    ];
    for (change, expected) in cases {
        let engine = make(change).expect("these configs load");
        let predicted = engine.predict("Please refund my order.", &questions);
        let scanned =
            engine.predict_long("Please refund my order.", &questions, &LongOptions::new());
        match expected {
            Some(message) => {
                for outcome in [predicted, scanned] {
                    match outcome {
                        Err(LayaError::InvalidConfig(got)) => assert_eq!(message, got),
                        other => panic!("expected InvalidConfig({message}), got {other:?}"),
                    }
                }
            }
            None => {
                predicted.expect("predicts normally");
                scanned.expect("scans normally");
            }
        }
        drop(engine);
    }
}
