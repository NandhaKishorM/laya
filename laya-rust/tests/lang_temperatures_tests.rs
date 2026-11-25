//! Per-language temperatures (`lang_temperatures`), the per-call `lang`, and the per-call
//! `max_len` / `head_max_len`, without a model. Expected values come from running Python 0.4.15
//! (`common.resolve_lang_temperatures`, `common.build_sequence` with a stub tokenizer) on the same
//! inputs. The model-backed half (scores under a table) is in `lang_temperatures_golden_tests.rs`.

mod common;

use common::stub_tokenizer::StubTokenizer;
use laya::answers::{LayaResult, Usage};
use laya::batch_prediction::BatchOptions;
use laya::error::{LayaError, Result};
use laya::lang_temperatures::{
    LangTemperature, LangTemperatures, TemperatureValue, resolve_lang_temperatures,
};
use laya::laya_options::{LayaCheckpoint, LayaOptions};
use laya::laya_router::{
    BatchRequest, LangGuess, LayaBatchPredictor, LayaLongPredictor, LayaPredictor, LayaRouter,
    LayaRouterOptions,
};
use laya::long_prediction::LongOptions;
use laya::predict_options::PredictOptions;
use laya::questions::{Question, QuestionSet, QuestionType};
use laya::sequence_builder::SequenceBuilder;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------- the table

fn tv(values: [f64; 3]) -> Vec<TemperatureValue> {
    values.iter().map(|&v| TemperatureValue::Float(v)).collect()
}

fn base() -> Vec<TemperatureValue> {
    tv([1.0, 2.0, 3.0])
}

fn entry(temperature: [f64; 3]) -> Option<LangTemperature> {
    Some(LangTemperature::new().temperature(temperature))
}

fn error_text(table: &[(String, Option<LangTemperature>)], base: &[TemperatureValue]) -> String {
    match resolve_lang_temperatures(table, base) {
        Err(LayaError::InvalidConfig(message)) => message,
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

static SEEN: Mutex<Vec<(String, log::Level, String)>> = Mutex::new(Vec::new());
struct Capture;
impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        SEEN.lock().unwrap().push((
            record.target().to_string(),
            record.level(),
            record.args().to_string(),
        ));
    }
    fn flush(&self) {}
}
static LOGGER: Capture = Capture;

/// The warnings a closure sends to the `log` facade. Serialised by the one test using it, because
/// the logger is process-wide.
fn warnings_of(run: impl FnOnce()) -> Vec<String> {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    SEEN.lock().unwrap().clear();
    run();
    let seen = SEEN.lock().unwrap();
    assert!(
        seen.iter()
            .all(|(target, level, _)| target == "laya" && *level == log::Level::Warn)
    );
    seen.iter().map(|(_, _, text)| text.clone()).collect()
}

#[test]
fn the_table_resolves_and_warns_as_python_does() {
    // Every row: Python's answer for `resolve_lang_temperatures(table, [1.0, 2.0, 3.0])`.
    let mut warned = Vec::new();
    warned.extend(warnings_of(|| {
        // key normalised (region tag, upper case), the last duplicate wins
        let table = vec![
            ("DE-AT".to_string(), entry([1.5, 2.0, 2.5])),
            ("de".to_string(), entry([3.0, 3.0, 3.0])),
        ];
        let resolved = resolve_lang_temperatures(&table, &base()).unwrap();
        assert_eq!([3.0, 3.0, 3.0], resolved.get("de").unwrap().temperature);
        assert!(resolved.get("DE-AT").is_none());
        assert!(resolved.get("fr").is_none());
        // a null entry and a missing temperature inherit the checkpoint's list
        let table = vec![
            ("fr".to_string(), None),
            ("es".to_string(), Some(LangTemperature::new())),
        ];
        let resolved = resolve_lang_temperatures(&table, &base()).unwrap();
        for lang in ["fr", "es"] {
            let language = resolved.get(lang).unwrap();
            assert_eq!([1.0, 2.0, 3.0], language.temperature);
            assert!(language.temperature_by_options.is_empty());
        }
        // nothing to say about accepted values
        let table = vec![(
            "de".to_string(),
            Some(
                LangTemperature::new()
                    .temperature([1.5, 2.0, 5.0])
                    .temperature_by_options([("choice:2", 0.5)]),
            ),
        )];
        resolve_lang_temperatures(&table, &base()).unwrap();
    }));
    assert!(
        warned.is_empty(),
        "accepted values must not warn: {warned:?}"
    );

    let prefix = "laya: language override";
    let suffix = "Treat confidence from the affected entries as uncalibrated.";

    // buckets are clamped and named; the inherited list is not reported
    let table = vec![(
        "de".to_string(),
        Some(LangTemperature::new().temperature_by_options([
            ("choice:2", TemperatureValue::Float(2.5)),
            ("noul:2", TemperatureValue::Int(9)),
        ])),
    )];
    let mut resolved = None;
    let texts = warnings_of(|| resolved = Some(resolve_lang_temperatures(&table, &base())))
        .into_iter()
        .collect::<Vec<_>>();
    let resolved = resolved.unwrap().unwrap();
    let de = resolved.get("de").unwrap();
    assert_eq!([1.0, 2.0, 3.0], de.temperature);
    assert_eq!(Some(&2.5), de.temperature_by_options.get("choice:2"));
    assert_eq!(Some(&5.0), de.temperature_by_options.get("noul:2"));
    assert_eq!(
        vec![format!(
            "{prefix} 'de' ships invalid temperatures or values outside [0.5, 5]; using \
             noul:2=9 -> 5. {suffix}"
        )],
        texts
    );

    // temperature entries first, then buckets, each quoted with %r
    let table = vec![(
        "de".to_string(),
        Some(
            LangTemperature::new()
                .temperature(vec![
                    TemperatureValue::Bool(true),
                    TemperatureValue::from("2.5"),
                    TemperatureValue::Null,
                ])
                .temperature_by_options([
                    ("a", TemperatureValue::from("x")),
                    ("b", TemperatureValue::Bool(false)),
                    ("c", TemperatureValue::Float(f64::NAN)),
                ]),
        ),
    )];
    let texts = warnings_of(|| {
        let resolved = resolve_lang_temperatures(&table, &base()).unwrap();
        let de = resolved.get("de").unwrap();
        assert_eq!([1.0, 2.5, 1.0], de.temperature);
        assert_eq!(Some(&1.0), de.temperature_by_options.get("c"));
    });
    assert_eq!(
        vec![format!(
            "{prefix} 'de' ships invalid temperatures or values outside [0.5, 5]; using \
             temperature[0]=True -> 1, temperature[2]=None -> 1, a='x' -> 1, b=False -> 1, \
             c=nan -> 1. {suffix}"
        )],
        texts
    );

    // clamped numbers; the language is quoted with Python's repr
    let table = vec![(
        "it'a".to_string(),
        Some(LangTemperature::new().temperature(vec![
            TemperatureValue::Int(99),
            TemperatureValue::Float(0.01),
            TemperatureValue::Int(1),
        ])),
    )];
    let texts = warnings_of(|| {
        let resolved = resolve_lang_temperatures(&table, &base()).unwrap();
        assert_eq!([5.0, 0.5, 1.0], resolved.get("it'a").unwrap().temperature);
    });
    assert_eq!(
        vec![format!(
            "{prefix} \"it'a\" ships invalid temperatures or values outside [0.5, 5]; using \
             temperature[0]=99 -> 5, temperature[1]=0.01 -> 0.5. {suffix}"
        )],
        texts
    );

    // a repeated bucket key keeps its last value, as a dict does
    let table = vec![(
        "de".to_string(),
        Some(LangTemperature::new().temperature_by_options([("choice:2", 1.5), ("choice:2", 9.0)])),
    )];
    let texts = warnings_of(|| {
        let resolved = resolve_lang_temperatures(&table, &base()).unwrap();
        assert_eq!(
            Some(&5.0),
            resolved
                .get("de")
                .unwrap()
                .temperature_by_options
                .get("choice:2")
        );
    });
    assert_eq!(
        vec![format!(
            "{prefix} 'de' ships invalid temperatures or values outside [0.5, 5]; using \
             choice:2=9.0 -> 5. {suffix}"
        )],
        texts
    );
}

#[test]
fn a_temperature_list_that_is_not_three_long_is_refused_with_pythons_text() {
    let table = |temperature: Vec<TemperatureValue>| {
        vec![(
            "de".to_string(),
            Some(LangTemperature::new().temperature(temperature)),
        )]
    };
    let got = |shown: &str| {
        format!("Language override 'de' temperature must be a list of 3 floats, got {shown}")
    };
    assert_eq!(
        got("[1.0, 2.0]"),
        error_text(&table(tv([1.0, 2.0, 3.0])[..2].to_vec()), &base())
    );
    assert_eq!(
        got("[1, 2, 3, 4]"),
        error_text(
            &table((1..=4).map(TemperatureValue::Int).collect()),
            &base()
        )
    );
    assert_eq!(got("[]"), error_text(&table(Vec::new()), &base()));
    // an entry that inherits a base that is not three long fails the same way
    let inheriting = vec![("de".to_string(), None)];
    assert_eq!(
        got("[1.0, 2.0]"),
        error_text(&inheriting, &tv([1.0, 2.0, 3.0])[..2])
    );
}

#[test]
fn a_language_replaces_both_checkpoint_maps_and_never_falls_back_to_the_checkpoint_bucket() {
    let table = vec![
        (
            "DE-at".to_string(),
            Some(
                LangTemperature::new()
                    .temperature([1.5, 2.0, 2.5])
                    .temperature_by_options([("choice:3-5", 4.0)]),
            ),
        ),
        ("fr".to_string(), None),
    ];
    let resolved: LangTemperatures = resolve_lang_temperatures(&table, &base()).unwrap();
    // bucket hit
    assert_eq!(
        Some(4.0),
        resolved.temperature_for(Some("de"), QuestionType::Choice, 4)
    );
    // bucket miss: the language's per-type temperature, by question type
    assert_eq!(
        Some(1.5),
        resolved.temperature_for(Some("de-CH"), QuestionType::Choice, 2)
    );
    assert_eq!(
        Some(2.0),
        resolved.temperature_for(Some("DE"), QuestionType::Score, 4)
    );
    assert_eq!(
        Some(2.5),
        resolved.temperature_for(Some("de"), QuestionType::Noul, 2)
    );
    // inherits the checkpoint's own list
    assert_eq!(
        Some(3.0),
        resolved.temperature_for(Some("fr"), QuestionType::Noul, 2)
    );
    // unknown, empty and absent languages change nothing
    for lang in [Some("xx"), Some(""), None] {
        assert_eq!(None, resolved.temperature_for(lang, QuestionType::Noul, 2));
    }
}

#[test]
fn options_carry_the_table_and_default_to_none() {
    assert!(LayaOptions::default().lang_temperatures.is_empty());
    let options = LayaOptions {
        lang_temperatures: vec![("de".to_string(), entry([1.0, 1.0, 1.0]))],
        ..Default::default()
    };
    assert_eq!(1, options.clone().lang_temperatures.len());
}

// ---------------------------------------------------------------------- signed budgets

#[test]
fn per_call_budgets_follow_pythons_arithmetic_for_zero_and_negative_values() {
    // (max_len, head_max_len, sequence length, markers, tokens per option): Python 0.4.15's
    // `build_sequence` on the same question and state with a one-id-per-word tokenizer.
    let cases: [(i64, i64, usize, &[usize]); 13] = [
        (512, 192, 57, &[9, 11, 13]),
        (64, 32, 57, &[9, 11, 13]),
        (0, 32, 0, &[]),
        (-1, 32, 16, &[]),
        (-5, 32, 12, &[]),
        (-100, 32, 0, &[]),
        (1, 32, 1, &[]),
        (64, 0, 57, &[9, 11, 13]),
        (64, -7, 57, &[9, 11, 13]),
        (64, -1000, 57, &[9, 11, 13]),
        (64, 1, 57, &[9, 11, 13]),
        (10, 32, 10, &[9]),
        (24, 32, 24, &[9, 11, 13]),
    ];
    let tok = StubTokenizer::new();
    let question = Question::choice(
        "pick one of these three",
        [
            ("alpha", Value::Null),
            ("beta", Value::Null),
            ("gamma", Value::Null),
        ],
    )
    .unwrap();
    let state = (0..40)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    for (max_len, head_max_len, length, markers) in cases {
        let (ids, found) = SequenceBuilder::build_signed(
            &tok,
            &json!(state),
            &question,
            max_len,
            head_max_len,
            false,
        )
        .unwrap();
        assert_eq!(
            length,
            ids.len(),
            "max_len={max_len} head_max_len={head_max_len}"
        );
        assert_eq!(
            markers,
            &found[..],
            "max_len={max_len} head_max_len={head_max_len}"
        );
    }
    // A bound beyond the platform's range is as good as unbounded.
    let (ids, markers) =
        SequenceBuilder::build_signed(&tok, &json!(state), &question, i64::MAX, i64::MIN, false)
            .unwrap();
    assert_eq!(57, ids.len());
    assert_eq!(3, markers.len());
}

#[test]
fn a_lost_marker_names_the_override_as_written() {
    let error = LayaError::NoMarkers {
        id: "pick".to_string(),
        head_max_len: -7,
        actual_markers: 0,
        expected_markers: 3,
        max_len: -5,
    };
    // Python: "question 'pick': only 0 of its 3 option markers fit in max_len=-5 with
    // head_max_len=-7 spent on the question; lower head_max_len, raise max_len, or use fewer options"
    assert_eq!(
        "question 'pick': only 0 of its 3 option markers fit in max_len=-5 with \
         head_max_len=-7 spent on the question; lower head_max_len, raise max_len, or use fewer \
         options",
        error.to_string()
    );
}

// ---------------------------------------------------------------------- the router

/// What a fake predictor was handed.
#[derive(Debug, Clone, PartialEq)]
enum Seen {
    Predict(PredictOptions),
    Batch(Vec<Value>, BatchOptions),
    Long(LongOptions),
}

struct Recorder {
    lang_temperatures: bool,
    seen: Arc<Mutex<Vec<Seen>>>,
}

fn result(questions: &QuestionSet) -> LayaResult {
    LayaResult::new(
        "fake",
        questions.ids().map(String::from).collect(),
        HashMap::new(),
        Usage::default(),
    )
}

impl LayaPredictor for Recorder {
    fn predict(&self, _: Value, questions: &QuestionSet) -> Result<LayaResult> {
        Ok(result(questions))
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
        Ok(result(questions))
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
            .push(Seen::Batch(states.to_vec(), options.clone()));
        Ok(states.iter().map(|_| result(questions)).collect())
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
        Ok(result(questions))
    }
}

fn recording_router(lang_temperatures: bool) -> (LayaRouter, Arc<Mutex<Vec<Seen>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handle = Arc::clone(&seen);
    let router = LayaRouter::with_factory(LayaRouterOptions::default(), move |_, _| {
        Ok(Arc::new(Recorder {
            lang_temperatures,
            seen: Arc::clone(&handle),
        }) as Arc<dyn LayaPredictor>)
    })
    .unwrap();
    (router, seen)
}

fn noul(id: &str) -> Arc<QuestionSet> {
    Arc::new(
        QuestionSet::new()
            .with(id, Question::noul("Is it?", Value::Null, Value::Null))
            .unwrap(),
    )
}

/// Latin text with enough German in it that detection names a language.
const GERMAN: &str = "Ich habe meine Bestellung nicht erhalten und das Paket ist seit zwei \
                      Wochen unterwegs, bitte melden Sie sich.";

fn detected(router: &LayaRouter, state: &str, questions: &QuestionSet) -> Option<String> {
    router
        .route(&json!(state), Some(questions), None, None, None)
        .unwrap()
        .detection
        .and_then(|detection| detection.language)
}

#[test]
fn the_router_forwards_the_explicit_lang_else_the_detected_one() {
    let (router, seen) = recording_router(true);
    let questions = noul("flag");
    let language = detected(&router, GERMAN, &questions);
    assert!(language.is_some(), "the German sample must be identified");

    // explicit lang wins, raw and unnormalised (and it also routes)
    router
        .predict_with_options(
            GERMAN,
            &questions,
            None,
            None,
            Some("De-AT"),
            None,
            &PredictOptions::new().max_len(300).head_max_len(100),
        )
        .unwrap();
    // no lang: the language the detection found
    router
        .predict_with_options(
            GERMAN,
            &questions,
            None,
            None,
            None,
            None,
            &PredictOptions::default(),
        )
        .unwrap();
    // a lang_guess hit leaves the detection empty, so there is no language to forward
    router
        .predict_with_options(
            GERMAN,
            &questions,
            None,
            None,
            None,
            Some(&LangGuess::from("de")),
            &PredictOptions::default(),
        )
        .unwrap();

    let seen = seen.lock().unwrap();
    let options = |index: usize| match &seen[index] {
        Seen::Predict(options) => options.clone(),
        other => panic!("expected a predict call, got {other:?}"),
    };
    assert_eq!(Some("De-AT"), options(0).lang.as_deref());
    assert_eq!(
        (Some(300), Some(100)),
        (options(0).max_len, options(0).head_max_len)
    );
    assert_eq!(language, options(1).lang);
    assert_eq!((None, None), (options(1).max_len, options(1).head_max_len));
    assert_eq!(None, options(2).lang);
}

#[test]
fn predict_long_forwards_the_explicit_lang_else_the_detected_one() {
    let (router, seen) = recording_router(true);
    let questions = noul("flag");
    let language = detected(&router, GERMAN, &questions);
    router
        .predict_long(
            GERMAN,
            &questions,
            &LongOptions::new().window(100),
            None,
            None,
            Some("fr"),
        )
        .unwrap();
    router
        .predict_long(GERMAN, &questions, &LongOptions::new(), None, None, None)
        .unwrap();
    let seen = seen.lock().unwrap();
    match (&seen[0], &seen[1]) {
        (Seen::Long(first), Seen::Long(second)) => {
            assert_eq!(Some("fr"), first.lang.as_deref());
            assert_eq!(Some(100), first.window);
            assert_eq!(language, second.lang);
        }
        other => panic!("expected two scans, got {other:?}"),
    }
}

fn batch_calls(seen: &Arc<Mutex<Vec<Seen>>>) -> Vec<(usize, BatchOptions)> {
    seen.lock()
        .unwrap()
        .iter()
        .map(|call| match call {
            Seen::Batch(states, options) => (states.len(), options.clone()),
            other => panic!("expected a batch call, got {other:?}"),
        })
        .collect()
}

#[test]
fn a_batch_splits_on_language_only_when_the_engine_has_language_temperatures() {
    let questions = noul("flag");
    let requests = || {
        vec![
            BatchRequest::new("one", Arc::clone(&questions))
                .model("en")
                .lang("de"),
            BatchRequest::new("two", Arc::clone(&questions))
                .model("en")
                .lang("de-AT"),
            BatchRequest::new("three", Arc::clone(&questions))
                .model("en")
                .lang("de"),
            BatchRequest::new("four", Arc::clone(&questions)).model("en"),
        ]
    };

    // with a table: the raw language joins the key, so "de" and "de-AT" do not share a pass
    let (router, seen) = recording_router(true);
    router
        .predict_batch(&requests(), &BatchOptions::new())
        .unwrap();
    let calls = batch_calls(&seen);
    let langs: Vec<_> = calls.iter().map(|(n, o)| (*n, o.lang.clone())).collect();
    assert_eq!(
        vec![
            (2, Some("de".to_string())),
            (1, Some("de-AT".to_string())),
            (1, None),
        ],
        langs
    );

    // without one: the language is unused and the group shares a single pass
    let (router, seen) = recording_router(false);
    router
        .predict_batch(&requests(), &BatchOptions::new())
        .unwrap();
    let calls = batch_calls(&seen);
    assert_eq!(1, calls.len());
    assert_eq!((4, None), (calls[0].0, calls[0].1.lang.clone()));
}

#[test]
fn a_batch_without_a_request_lang_forwards_the_detected_one_to_an_engine_with_a_table() {
    let questions = noul("flag");
    let (router, seen) = recording_router(true);
    let language = detected(&router, GERMAN, &questions);
    router
        .predict_batch(
            &[BatchRequest::new(GERMAN, Arc::clone(&questions))],
            &BatchOptions::new(),
        )
        .unwrap();
    assert_eq!(language, batch_calls(&seen)[0].1.lang);
}

#[test]
fn a_batch_splits_on_the_requests_own_budgets() {
    let questions = noul("flag");
    let (router, seen) = recording_router(false);
    let requests = vec![
        BatchRequest::new("a", Arc::clone(&questions)).model("en"),
        BatchRequest::new("b", Arc::clone(&questions))
            .model("en")
            .max_len(0),
        BatchRequest::new("c", Arc::clone(&questions)).model("en"),
        BatchRequest::new("d", Arc::clone(&questions))
            .model("en")
            .max_len(0)
            .head_max_len(-3),
        BatchRequest::new("e", Arc::clone(&questions))
            .model("en")
            .max_len(0),
    ];
    // Python's Router.predict_batch has no batch-wide language or budget: a TypeError
    for (options, name) in [
        (BatchOptions::new().lang("fr"), "lang"),
        (BatchOptions::new().max_len(7), "max_len"),
        (BatchOptions::new().head_max_len(7), "head_max_len"),
    ] {
        let error = router.predict_batch(&requests, &options).unwrap_err();
        assert_eq!(
            format!("Router.predict_batch() got an unexpected keyword argument '{name}'"),
            error.to_string()
        );
    }
    assert!(seen.lock().unwrap().is_empty());
    router
        .predict_batch(&requests, &BatchOptions::new())
        .unwrap();
    let calls: Vec<_> = batch_calls(&seen)
        .into_iter()
        .map(|(n, o)| (n, o.max_len, o.head_max_len, o.lang))
        .collect();
    assert_eq!(
        vec![
            (2, None, None, None),
            (2, Some(0), None, None),
            (1, Some(0), Some(-3), None),
        ],
        calls
    );
}

#[test]
fn the_default_predictor_drops_lang_but_refuses_a_token_budget() {
    struct Plain;
    impl LayaPredictor for Plain {
        fn predict(&self, _: Value, questions: &QuestionSet) -> Result<LayaResult> {
            Ok(result(questions))
        }
    }
    let questions = noul("flag");
    let with_lang = PredictOptions::new().lang("de");
    assert!(
        Plain
            .predict_with_options(json!("s"), &questions, &with_lang)
            .is_ok()
    );
    assert!(matches!(
        Plain.predict_with_options(json!("s"), &questions, &PredictOptions::new().max_len(5)),
        Err(LayaError::NotSupported(_))
    ));
}

#[test]
fn the_router_hands_its_table_to_the_engines_it_builds() {
    let seen_tables = Arc::new(Mutex::new(Vec::new()));
    let handle = Arc::clone(&seen_tables);
    let engine_options = LayaOptions {
        lang_temperatures: vec![("de".to_string(), entry([1.0, 1.0, 1.0]))],
        ..Default::default()
    };
    let router = LayaRouter::with_factory(
        LayaRouterOptions {
            engine_options,
            ..Default::default()
        },
        move |checkpoint: LayaCheckpoint, options: &LayaOptions| {
            handle
                .lock()
                .unwrap()
                .push((checkpoint, options.lang_temperatures.len()));
            Ok(Arc::new(Recorder {
                lang_temperatures: true,
                seen: Arc::new(Mutex::new(Vec::new())),
            }) as Arc<dyn LayaPredictor>)
        },
    )
    .unwrap();
    router.load("en").unwrap();
    router.load("ml").unwrap();
    assert_eq!(
        vec![
            (LayaCheckpoint::English, 1),
            (LayaCheckpoint::Multilingual, 1)
        ],
        *seen_tables.lock().unwrap()
    );
}
