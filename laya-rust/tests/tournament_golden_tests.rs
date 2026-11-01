//! Tier 1 (probes) and Tier 2 (model cases): `LayaShortlist::predict_tournament` and the cached
//! embedder against what Python recorded.
//!
//! - `routing/tournament_probe.json`: Python's own `predict_tournament` over a scripted predictor,
//!   replayed here through `predict_tournament_with` with a closure that answers the same way.
//!   Every call's questions and kwargs, the warnings and the result or error are checked.
//! - `routing/cached_embed_probe.json`: Python's own `cached_embed_fn` over the hashing embedder
//!   divided by 3.0, replayed step by step.
//! - `tournament_*` cases of each checkpoint's `index.json` (`"kind": "tournament"`): the real
//!   model through `predict_tournament_with`, call by call. Skipped until the checkpoint's golden
//!   data and ONNX artifacts are present.
//!
//! Not expressible in Rust and so not replayed (counted and printed): a predictor that returns
//! `None` (`predict_returns_none`), a negative `group_size` or `maxsize`, and a `None` text
//! (which Python turns into `""`; it is replayed as `""`).

mod common;

use common::golden_data::{CheckpointGoldenData, build_questions, recorded_predict_options};
use common::{engines, hashing_embedder, result_compare, routing_golden};
use indexmap::IndexMap;
use laya::answers::{ActionInfo, Answer, ChoiceAnswer, LayaResult, NoulAnswer, ScoreAnswer, Usage};
use laya::error::{LayaError, Result};
use laya::laya_shortlist::{CachedEmbed, LayaShortlist};
use laya::questions::{Question, QuestionSet};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

// ------------------------------------------------------------------------------- log capture

struct Capture;
static LOGGED: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Warning texts already shown in this process: an identical one is shown once, as in Python.
static SHOWN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
/// The tests share one logger and one set of shown warnings, so they run one at a time.
static GATE: Mutex<()> = Mutex::new(());

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        if record.target() == "laya" && record.level() == log::Level::Warn {
            LOGGED.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}
static LOGGER: Capture = Capture;

fn init_logger() {
    // The first test to get here installs it; the others find it in place.
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
}

fn take_logged() -> Vec<String> {
    std::mem::take(&mut *LOGGED.lock().unwrap())
}

/// Assert the warnings a case raised are the recorded ones, minus those already shown.
fn assert_warnings(where_case: &str, recorded: &Value, logged: Vec<String>) {
    let mut shown = SHOWN.lock().unwrap();
    let shown = shown.get_or_insert_with(HashSet::new);
    let mut expected = Vec::new();
    for w in recorded.as_array().unwrap() {
        assert_eq!(
            "RuntimeWarning",
            w["category"].as_str().unwrap(),
            "{where_case}: category"
        );
        let message = w["message"].as_str().unwrap().to_string();
        if shown.insert(message.clone()) {
            expected.push(message);
        }
    }
    assert_eq!(expected, logged, "{where_case}: warnings");
}

// ------------------------------------------------------------------ question normal forms

/// The recorded definition of a question, reduced to what the replay compares.
fn norm_def(def: &Value) -> Value {
    let kind = def["type"].as_str().unwrap();
    let body = match (kind, &def["criteria"]) {
        ("choice", Value::Object(map)) => {
            Value::Array(map.iter().map(|(k, v)| json!([k, v])).collect())
        }
        ("choice", Value::Array(labels)) => {
            Value::Array(labels.iter().map(|l| json!([l, null])).collect())
        }
        ("score", levels) => levels.clone(),
        _ => Value::Null,
    };
    json!({
        "type": kind,
        "instructions": def["instructions"],
        "option_order": def.get("option_order").cloned().unwrap_or(Value::Null),
        "body": body,
    })
}

fn norm_question(question: &Question) -> Value {
    let (kind, body) = match question {
        Question::Choice(c) => (
            "choice",
            Value::Array(c.options().iter().map(|(l, d)| json!([l, d])).collect()),
        ),
        Question::Score(s) => ("score", Value::Array(s.levels().to_vec())),
        Question::Noul(_) => ("noul", Value::Null),
    };
    json!({
        "type": kind,
        "instructions": question.instructions(),
        "option_order": question.option_order().map(|o| json!(o)).unwrap_or(Value::Null),
        "body": body,
    })
}

fn assert_request(where_call: &str, recorded: &Value, got: &QuestionSet) {
    let want_ids: Vec<&str> = recorded
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(want_ids, got.ids().collect::<Vec<_>>(), "{where_call}: ids");
    for (id, def) in recorded.as_object().unwrap() {
        assert_eq!(
            norm_def(def),
            norm_question(&got[id.as_str()]),
            "{where_call}: question {id}"
        );
    }
}

fn group_size_of(case: &Value) -> Option<usize> {
    match case.get("group_size") {
        None => Some(laya::laya_shortlist::DEFAULT_TOURNAMENT_GROUP),
        Some(v) => v.as_u64().map(|n| n as usize),
    }
}

// ------------------------------------------------------------------------ tournament probe

/// Answers like the probe's `_ScriptedAgent`.
struct Scripted {
    mode: String,
    calls: Mutex<Vec<QuestionSet>>,
}

impl Scripted {
    fn predict(&self, questions: &QuestionSet) -> Result<LayaResult> {
        let number = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(questions.clone());
            calls.len()
        };
        let order: Vec<String> = questions.ids().map(String::from).collect();
        let mut by_id = HashMap::new();
        for (id, question) in questions.iter() {
            let action = ActionInfo::default();
            let answer = match question {
                Question::Choice(c) => {
                    let labels: Vec<&str> = c.labels().collect();
                    let mut pick = labels[labels.len() * 2 / 3].to_string();
                    if self.mode == "bad_winner" && number == 1 {
                        pick = "no-such-label".to_string();
                    }
                    let mut probabilities = IndexMap::new();
                    probabilities.insert(pick.clone(), 1.0);
                    Answer::Choice(ChoiceAnswer::new(pick, probabilities, 1.0, 1.0, action))
                }
                Question::Score(_) => {
                    Answer::Score(ScoreAnswer::new(1.5, vec![], vec![], 1.0, 1.0, action))
                }
                Question::Noul(_) => Answer::Noul(NoulAnswer::new(0.7, 1.0, 1.0, action)),
            };
            by_id.insert(id.to_string(), answer);
        }
        if self.mode == "omit_group" && number == 1 {
            by_id.remove("1");
        }
        if self.mode == "score_group" && number == 1 {
            by_id.insert(
                "1".to_string(),
                Answer::Score(ScoreAnswer::new(
                    1.5,
                    vec![],
                    vec![],
                    1.0,
                    1.0,
                    ActionInfo::default(),
                )),
            );
        }
        let usage = Usage {
            input_tokens: (100 * number + questions.len()) as u32,
            output_tokens: 0,
        };
        Ok(LayaResult::new("scripted", order, by_id, usage))
    }
}

#[test]
fn tournament_probe_replays_pythons_calls_warnings_results_and_errors() {
    let _gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(cases) = routing_golden::load_array("tournament_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };
    init_logger();
    take_logged();
    let (mut replayed, mut skipped) = (0, 0);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let mode = case["mode"].as_str().unwrap();
        let Some(group_size) = group_size_of(case) else {
            skipped += 1; // a negative group_size has no usize
            eprintln!("{name}: negative group_size is not expressible, not replayed");
            continue;
        };
        if mode == "none" {
            skipped += 1; // a predictor that returns None has no Rust form
            eprintln!("{name}: a predictor returning None is not expressible, not replayed");
            continue;
        }
        let questions = build_questions(&case["questions"]).unwrap();
        let agent = Scripted {
            mode: mode.to_string(),
            calls: Mutex::new(Vec::new()),
        };
        let kwargs = case.get("kwargs").cloned().unwrap_or_else(|| json!({}));
        let seen_kwargs = Mutex::new(Vec::new());
        take_logged();
        let outcome = LayaShortlist::predict_tournament_with(
            |_state, set| {
                seen_kwargs.lock().unwrap().push(kwargs.clone());
                agent.predict(set)
            },
            case["state"].clone(),
            &questions,
            group_size,
        );
        assert_warnings(name, &case["warnings"], take_logged());

        let calls = agent.calls.lock().unwrap();
        let recorded_calls = case["calls"].as_array().unwrap();
        assert_eq!(recorded_calls.len(), calls.len(), "{name}: number of calls");
        for (i, (rec, got)) in recorded_calls.iter().zip(calls.iter()).enumerate() {
            assert_request(&format!("{name} call {i}"), &rec["questions"], got);
            assert_eq!(
                &rec["kwargs"],
                &seen_kwargs.lock().unwrap()[i],
                "{name} call {i}: kwargs"
            );
        }

        match (&case["error"], outcome) {
            (Value::Null, Ok(result)) => {
                let want = &case["result"];
                assert_eq!(want["model"].as_str().unwrap(), result.model(), "{name}");
                assert_eq!(
                    want["usage"]["input_tokens"].as_u64().unwrap() as u32,
                    result.usage().input_tokens,
                    "{name}: usage is the final call's"
                );
                let want_ids: Vec<&str> = want["answers"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect();
                assert_eq!(want_ids, result.ids().to_vec(), "{name}: answer ids");
                for (id, answer) in want["answers"].as_object().unwrap() {
                    if let Some(choice) = answer.get("choice") {
                        assert_eq!(
                            choice.as_str().unwrap(),
                            result[id.as_str()].as_choice().unwrap().choice,
                            "{name}: answer {id}"
                        );
                    }
                }
                let got_meta = result.tournament().expect("tournament metadata");
                let want_meta = want["tournament"].as_object().unwrap();
                assert_eq!(
                    want_meta.keys().collect::<Vec<_>>(),
                    got_meta.keys().collect::<Vec<_>>(),
                    "{name}: tournament ids"
                );
                for (id, entry) in want_meta {
                    let got = &got_meta[id.as_str()];
                    let labels: Vec<&str> = entry["labels"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|l| l.as_str().unwrap())
                        .collect();
                    assert_eq!(labels, got.labels, "{name}: {id} labels");
                    assert_eq!(
                        entry["n"].as_u64().unwrap() as usize,
                        got.n,
                        "{name}: {id} n"
                    );
                    assert_eq!(
                        entry["rounds"].as_u64().unwrap() as usize,
                        got.rounds,
                        "{name}: {id} rounds"
                    );
                }
            }
            (Value::Null, Err(e)) => panic!("{name}: Python succeeded, Rust failed: {e}"),
            (error, Ok(_)) => panic!("{name}: Python raised {error}, Rust succeeded"),
            (error, Err(e)) => {
                let kind = error["type"].as_str().unwrap();
                assert!(
                    matches!(e, LayaError::InvalidQuestion(_)),
                    "{name}: {kind} maps to InvalidQuestion, got {e:?}"
                );
                assert_eq!(error["message"].as_str().unwrap(), e.to_string(), "{name}");
            }
        }
        replayed += 1;
    }
    eprintln!("tournament probe: replayed {replayed}, not expressible {skipped}");
    assert!(replayed >= 18, "most of the probe must replay");
}

// -------------------------------------------------------------------------- cached embed probe

/// `hashing_embedder.embed / 3.0` with the misbehaviours a step can ask for.
struct Switchable {
    calls: Mutex<Vec<Vec<String>>>,
    mode: Mutex<String>,
    bad: Mutex<Vec<String>>,
}

impl Switchable {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        self.calls.lock().unwrap().push(texts.to_vec());
        let mode = self.mode.lock().unwrap().clone();
        if mode == "raise" {
            return Err(LayaError::Onnx("boom".to_string()));
        }
        let mut rows: Vec<Vec<f64>> = hashing_embedder::embed(texts)?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v / 3.0).collect())
            .collect();
        if mode == "short" {
            rows.pop();
            return Ok(rows);
        }
        if mode == "dim32" {
            for row in &mut rows {
                row.truncate(32);
            }
        }
        for (row, text) in rows.iter_mut().zip(texts) {
            if self.bad.lock().unwrap().contains(text) {
                row[0] = f64::NAN;
                row[1] = f64::INFINITY;
                row[2] = f64::NEG_INFINITY;
            }
        }
        Ok(rows)
    }
}

fn assert_info(where_step: &str, want: &Value, got: laya::laya_shortlist::CacheInfo) {
    assert_eq!(
        (
            want["size"].as_u64().unwrap() as usize,
            want["maxsize"].as_u64().unwrap() as usize,
            want["hits"].as_u64().unwrap(),
            want["misses"].as_u64().unwrap()
        ),
        (got.size, got.maxsize, got.hits, got.misses),
        "{where_step}: cache_info"
    );
}

#[test]
fn cached_embed_probe_replays_pythons_cache_step_by_step() {
    let Some(probe) = routing_golden::load("cached_embed_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };
    let mut steps_checked = 0;
    for scenario in probe["scenarios"].as_array().unwrap() {
        let name = scenario["name"].as_str().unwrap();
        let embedder = Switchable {
            calls: Mutex::new(Vec::new()),
            mode: Mutex::new("normal".to_string()),
            bad: Mutex::new(Vec::new()),
        };
        let cache = LayaShortlist::cached_embed_fn(
            |texts: &[String]| embedder.embed(texts),
            scenario["maxsize"].as_u64().unwrap() as usize,
        )
        .unwrap();
        for (i, step) in scenario["steps"].as_array().unwrap().iter().enumerate() {
            let where_step = format!("{name} step {i}");
            embedder.calls.lock().unwrap().clear();
            if step["op"] == "clear" {
                cache.cache_clear();
                assert_info(&where_step, &step["info"], cache.cache_info());
                steps_checked += 1;
                continue;
            }
            *embedder.mode.lock().unwrap() = step["mode"].as_str().unwrap().to_string();
            *embedder.bad.lock().unwrap() = step["bad"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_str().unwrap().to_string())
                .collect();
            // Python turns a None text into "".
            let texts: Vec<String> = step["texts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_str().unwrap_or("").to_string())
                .collect();
            let outcome = cache.embed(&texts);
            let seen: Vec<Vec<String>> = embedder.calls.lock().unwrap().clone();
            let want_calls: Vec<Vec<String>> = step["embed_calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    c.as_array()
                        .unwrap()
                        .iter()
                        .map(|t| t.as_str().unwrap().to_string())
                        .collect()
                })
                .collect();
            assert_eq!(want_calls, seen, "{where_step}: texts the embedder saw");
            match (&step["error"], outcome) {
                (Value::Null, Ok(rows)) => {
                    let want: Vec<Vec<f64>> = step["rows"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|r| {
                            r.as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_f64().unwrap())
                                .collect()
                        })
                        .collect();
                    assert_eq!(want, rows, "{where_step}: rows (float32 widened)");
                    let shape = (
                        step["shape"][0].as_u64().unwrap() as usize,
                        step["shape"][1].as_u64().unwrap() as usize,
                    );
                    assert_eq!(
                        shape,
                        (rows.len(), rows.first().map_or(0, Vec::len)),
                        "{where_step}: shape"
                    );
                }
                (Value::Null, Err(e)) => panic!("{where_step}: Python succeeded, Rust failed: {e}"),
                (error, Ok(_)) => panic!("{where_step}: Python raised {error}, Rust succeeded"),
                (error, Err(e)) => {
                    assert_eq!(
                        error["message"].as_str().unwrap(),
                        e.to_string(),
                        "{where_step}"
                    );
                }
            }
            assert_info(&where_step, &step["info"], cache.cache_info());
            steps_checked += 1;
        }
    }
    assert!(steps_checked >= 25);

    let mut refused = 0;
    for bad in probe["bad_maxsize"].as_array().unwrap() {
        let Some(maxsize) = bad["maxsize"].as_u64() else {
            eprintln!(
                "maxsize {} is negative, not expressible as usize",
                bad["maxsize"]
            );
            continue;
        };
        let error = CachedEmbed::with_maxsize(hashing_embedder::embed, maxsize as usize)
            .err()
            .expect("maxsize 0 is refused");
        assert!(matches!(error, LayaError::InvalidQuestion(_)));
        assert_eq!(bad["message"].as_str().unwrap(), error.to_string());
        refused += 1;
    }
    assert_eq!(1, refused);
    eprintln!("cached_embed probe: {steps_checked} steps replayed");
}

// ------------------------------------------------------------------- model cases (Tier 2)

#[test]
fn tournament_cases_match_the_real_model_for_every_checkpoint() {
    let _gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
    init_logger();
    let mut checked_any = false;
    for checkpoint in engines::checkpoints() {
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }
        let cases: Vec<_> = golden
            .index()
            .into_iter()
            .filter(|c| c.kind.as_deref() == Some("tournament"))
            .collect();
        if cases.is_empty() {
            eprintln!(
                "skip: no tournament cases recorded for '{}' (regenerate the checkpoint goldens)",
                checkpoint.subdir()
            );
            continue;
        }
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };
        for info in cases {
            checked_any = true;
            let where_case = format!("{}/{info}", checkpoint.subdir());
            let root = golden.load_case(&info);
            run_model_case(&where_case, &engine, &root, info.expects_error);
        }
    }
    if !checked_any {
        eprintln!("skip: no tournament golden data + artifacts found for any checkpoint");
    }
}

/// One call to the model: the request, the choice it gave each choice question, the input tokens.
type ModelCall = (QuestionSet, Vec<(String, String)>, u32);

fn run_model_case(where_case: &str, engine: &laya::LayaEngine, root: &Value, expects_error: bool) {
    let questions = build_questions(&root["questions"]).unwrap();
    let Some(group_size) = group_size_of(root) else {
        eprintln!("{where_case}: negative group_size is not expressible, not replayed");
        return;
    };
    let options = recorded_predict_options(root);
    let calls: Mutex<Vec<ModelCall>> = Mutex::new(Vec::new());
    take_logged();
    let outcome = LayaShortlist::predict_tournament_with(
        |state, set| {
            let result = engine.predict_with_options(state, set, &options)?;
            let choices = result
                .iter()
                .filter_map(|(id, a)| a.as_choice().map(|c| (id.to_string(), c.choice.clone())))
                .collect();
            calls
                .lock()
                .unwrap()
                .push((set.clone(), choices, result.usage().input_tokens));
            Ok(result)
        },
        root["state"].clone(),
        &questions,
        group_size,
    );
    let logged = take_logged();

    if expects_error {
        let error = outcome
            .err()
            .unwrap_or_else(|| panic!("{where_case}: expected an error"));
        assert!(
            matches!(error, LayaError::InvalidQuestion(_)),
            "{where_case}: {error:?}"
        );
        assert_eq!(
            root["error_message"].as_str().unwrap(),
            error.to_string(),
            "{where_case}: message"
        );
        assert!(
            calls.lock().unwrap().is_empty(),
            "{where_case}: the model was asked"
        );
        return;
    }
    let result = outcome.unwrap_or_else(|e| panic!("{where_case}: predict_tournament failed: {e}"));
    assert_warnings(where_case, &root["warnings"], logged);

    let calls = calls.lock().unwrap();
    let recorded = root["calls"].as_array().unwrap();
    assert_eq!(recorded.len(), calls.len(), "{where_case}: number of calls");
    for (i, (rec, (set, choices, tokens))) in recorded.iter().zip(calls.iter()).enumerate() {
        let where_call = format!("{where_case} call {i}");
        assert_request(&where_call, &rec["questions"], set);
        let want: Vec<(String, String)> = rec["choices"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();
        assert_eq!(want, *choices, "{where_call}: choices");
        assert_eq!(
            rec["usage"]["input_tokens"].as_u64().unwrap() as u32,
            *tokens,
            "{where_call}: input tokens"
        );
    }
    let meta = result.tournament().expect("tournament metadata");
    for (id, entry) in root["result"]["tournament"].as_object().unwrap() {
        let got = &meta[id.as_str()];
        let labels: Vec<&str> = entry["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_str().unwrap())
            .collect();
        assert_eq!(labels, got.labels, "{where_case}: {id} labels");
        assert_eq!(
            entry["n"].as_u64().unwrap() as usize,
            got.n,
            "{where_case}: {id} n"
        );
        assert_eq!(
            entry["rounds"].as_u64().unwrap() as usize,
            got.rounds,
            "{where_case}: {id} rounds"
        );
    }
    assert_eq!(
        root["result"]["tournament"].as_object().unwrap().len(),
        meta.len(),
        "{where_case}: tournament ids"
    );
    result_compare::assert_result_matches(where_case, &root["result"], &result);
}
