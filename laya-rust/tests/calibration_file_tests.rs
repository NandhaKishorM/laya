//! Calibration files and histogram binning, model-free: the loader against Python's own
//! `apply_calibration_payload` and `apply_binning_map` (the probe `routing/calibration_probe.json`,
//! recorded over the synthetic inputs of `tools/calibration_cases.py`), the shipped fixture, reading
//! the file, installing it on a config, the clamp warning through the `log` facade, and the router
//! handing the path to every engine. The model-backed cases (`calibration_*`) run in the ordinary
//! parity suites on engines built with the fixture (`common::engines`).

mod common;

use common::routing_golden;
use laya::calibration_file::{BinningMap, CalibrationFile};
use laya::error::LayaError;
use laya::laya_config::LayaConfig;
use laya::laya_options::{LayaCheckpoint, LayaOptions};
use laya::laya_router::{LayaPredictor, LayaRouter, LayaRouterOptions};
use laya::questions::QuestionSet;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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

/// A recorded number: JSON's, or the strings the dumper writes for what JSON has no literal for.
fn recorded_f64(value: &Value) -> f64 {
    match value {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) => match s.as_str() {
            "nan" => f64::NAN,
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            other => panic!("unexpected recorded number {other:?}"),
        },
        other => panic!("unexpected recorded number {other}"),
    }
}

fn the_error_text(error: &LayaError) -> String {
    match error {
        LayaError::InvalidConfig(text) => text.clone(),
        other => panic!("expected the ValueError as InvalidConfig, got {other:?}"),
    }
}

/// Python's `type: message` for an error it raised, as the crate spells it: a `ValueError` is the
/// bare message and any other type carries its name.
fn python_error(error_type: &str, message: &str) -> String {
    if error_type == "ValueError" {
        message.to_string()
    } else {
        format!("{error_type}: {message}")
    }
}

fn binning_map_of(file: &CalibrationFile) -> Value {
    match &file.binning_map {
        None => Value::Null,
        Some(map) => Value::Object(
            map.buckets()
                .map(|bucket| {
                    let entry = map.get(bucket).unwrap();
                    (
                        bucket.to_string(),
                        serde_json::json!({"bins": entry.bins(), "values": entry.values()}),
                    )
                })
                .collect(),
        ),
    }
}

/// Numbers equal as Python's `==` reads them (`0` is `0.0`), objects without regard to order.
fn same_json(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_json(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, x)| b.get(k).is_some_and(|y| same_json(x, y)))
        }
        _ => expected == actual,
    }
}

#[test]
fn the_loader_matches_pythons_apply_calibration_payload_on_every_recorded_case() {
    let Some(probe) = routing_golden::load("calibration_probe.json") else {
        eprintln!("skip: routing golden not found (calibration_probe.json)");
        return;
    };
    let mut checked = 0;
    let mut identity_warnings = 0;
    for case in probe["payloads"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let (warnings, file) = CalibrationFile::parse_probe(case["json"].as_str().unwrap());
        checked += 1;

        // Python also warns, as a UserWarning, when a version 2 file records a model, subfolder or
        // config other than the agent's. This crate has no `model_id_or_path` to compare (the engine
        // loads an exported artifact directory), so it never emits that warning: the probe's identity
        // warnings are the only ones filtered, and each is checked to be exactly that kind. Every
        // other warning is compared word for word.
        let mut expected = Vec::new();
        for warning in case["warnings"].as_array().unwrap() {
            let (category, message) = (
                warning["category"].as_str().unwrap(),
                warning["message"].as_str().unwrap(),
            );
            if category == "UserWarning" {
                assert!(
                    message.starts_with("calibration was fitted for model_id_or_path=")
                        && message.ends_with("Loading the temperatures anyway."),
                    "{label}: an unexpected UserWarning: {message}"
                );
                identity_warnings += 1;
            } else {
                assert_eq!("RuntimeWarning", category, "{label}");
                expected.push(message);
            }
        }
        assert_eq!(expected, warnings, "{label}: warnings");

        if let Some(message) = case.get("error_message") {
            let want = python_error(
                case["error_type"].as_str().unwrap(),
                message.as_str().unwrap(),
            );
            let error = file.expect_err(label);
            assert_eq!(want, the_error_text(&error), "{label}: error");
            continue;
        }
        let file = file.unwrap_or_else(|e| panic!("{label}: {e}"));
        let want = &case["result"];
        let temperature: Vec<f64> = want["temperature"]
            .as_array()
            .unwrap()
            .iter()
            .map(recorded_f64)
            .collect();
        assert_eq!(temperature, file.temperature, "{label}: temperature");
        let buckets: Vec<(String, f64)> = want["temperature_by_options"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), recorded_f64(v)))
            .collect();
        assert_eq!(
            buckets, file.temperature_by_options,
            "{label}: temperature_by_options"
        );
        assert!(
            same_json(&want["binning_map"], &binning_map_of(&file)),
            "{label}: binning_map {} vs {}",
            want["binning_map"],
            binning_map_of(&file)
        );
    }
    assert_eq!(152, checked, "every recorded payload ran");
    assert!(
        identity_warnings >= 15,
        "{identity_warnings} identity warnings seen"
    );
}

#[test]
fn binning_matches_pythons_apply_binning_map_on_every_recorded_lookup() {
    let Some(probe) = routing_golden::load("calibration_probe.json") else {
        eprintln!("skip: routing golden not found (calibration_probe.json)");
        return;
    };
    let maps: std::collections::HashMap<&str, BinningMap> = probe["maps"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, map)| {
            let payload = serde_json::json!({"temperature": [1.0, 1.0, 1.0], "binning_map": map});
            let (_, file) = CalibrationFile::parse_probe(&payload.to_string());
            (name.as_str(), file.unwrap().binning_map.unwrap())
        })
        .collect();
    let mut checked = 0;
    for case in probe["lookups"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let map = &maps[case["map"].as_str().unwrap()];
        let got = map.apply(
            recorded_f64(&case["confidence"]),
            case["bucket"].as_str().unwrap(),
        );
        checked += 1;
        match case.get("error_message") {
            Some(message) => {
                let want = python_error(
                    case["error_type"].as_str().unwrap(),
                    message.as_str().unwrap(),
                );
                assert_eq!(want, the_error_text(&got.unwrap_err()), "{label}");
            }
            None => {
                let want = recorded_f64(&case["result"]);
                let got = got.unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(
                    got == want || (got.is_nan() && want.is_nan()),
                    "{label}: python {want}, rust {got}"
                );
            }
        }
    }
    assert_eq!(62, checked, "every recorded lookup ran");
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/calibration_fixture.json")
}

#[test]
fn the_shipped_fixture_loads_with_its_temperatures_and_binning_map() {
    let file = CalibrationFile::load(fixture_path()).unwrap();
    assert_eq!([1.8, 2.2, 1.3], file.temperature);
    assert_eq!(
        vec![
            ("choice:3-5".to_string(), 2.4),
            ("noul:2".to_string(), 0.8),
            // 0.6 and 3.0 are inside the clamp range; nothing in the fixture is clamped.
            ("choice:6-10".to_string(), 0.6),
            ("score:6-10".to_string(), 3.0),
        ],
        file.temperature_by_options
    );
    let map = file.binning_map.unwrap();
    assert_eq!(
        vec![
            "choice:3-5",
            "noul:2",
            "score:6-10",
            "choice:11+",
            "score:2"
        ],
        map.buckets().collect::<Vec<_>>()
    );
    // A hit takes its bin, a bucket the map lacks (choice:2) keeps the temperature-scaled value.
    assert_eq!(0.31, map.apply(0.3, "choice:3-5").unwrap());
    assert_eq!(0.3, map.apply(0.3, "choice:2").unwrap());
    // The last bin is the top edge: 1.0 does not run past the map.
    assert_eq!(0.68, map.apply(1.0, "choice:3-5").unwrap());
    assert_eq!(0.5, map.apply(0.0, "score:2").unwrap());
}

fn scratch(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("laya_calibration_tests_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn a_file_is_read_as_strict_utf8_json_in_text_mode() {
    // `open()` turns CRLF and a lone CR into LF, so a line count in a decode error matches Python's.
    let crlf = scratch(
        "crlf.json",
        b"{\r\n \"temperature\": [1.5,\r 2.0, 1.25]\r\n}",
    );
    assert_eq!(
        [1.5, 2.0, 1.25],
        CalibrationFile::load(&crlf).unwrap().temperature
    );

    // `json.load` accepts NaN, which `clamp_temperature` replaces with 1.0 (a warning, not an error).
    let nan = scratch("nan.json", b"{\"temperature\": [NaN, 2.0, 1.25]}");
    assert_eq!(
        [1.0, 2.0, 1.25],
        CalibrationFile::load(&nan).unwrap().temperature
    );

    let bom = scratch("bom.json", b"\xef\xbb\xbf{\"temperature\": [1, 1, 1]}");
    let text = CalibrationFile::load(&bom).unwrap_err().to_string();
    assert!(
        text.starts_with("Unexpected UTF-8 BOM (decode using utf-8-sig)"),
        "{text}"
    );

    let broken = scratch("broken.json", b"{\"temperature\": [1, 1,\r\n");
    let text = CalibrationFile::load(&broken).unwrap_err().to_string();
    assert!(text.contains("line 2 column 1"), "{text}");

    let not_utf8 = scratch(
        "latin1.json",
        b"{\"temperature\": [1, 1, 1], \"x\": \"\xe9\"}",
    );
    assert!(matches!(
        CalibrationFile::load(&not_utf8),
        Err(LayaError::Io(_))
    ));

    let missing = std::env::temp_dir().join("laya_calibration_tests_no_such_file.json");
    assert!(matches!(
        CalibrationFile::load(&missing),
        Err(LayaError::Io(_))
    ));
}

#[test]
fn installing_a_file_replaces_both_of_the_checkpoints_temperature_maps() {
    let mut config = LayaConfig::parse(
        r#"{"temperature": [1.2, 1.3, 1.4], "temperature_by_options": {"choice:2": 1.1, "noul:2": 4.0}}"#,
    )
    .unwrap();
    let file = CalibrationFile::parse(
        r#"{"temperature": [1.5, "2.5", 9], "temperature_by_options": {"choice:3-5": 2.5}}"#,
    )
    .unwrap();
    file.install_temperatures(&mut config);
    assert_eq!(vec![1.5, 2.5, 5.0], config.temperature);
    assert_eq!(vec![1.5, 2.5, 9.0], config.temperature_raw);
    assert_eq!(1, config.temperature_by_options.len());
    assert_eq!(Some(&2.5), config.temperature_by_options.get("choice:3-5"));
    assert_eq!(None, config.temperature_by_options.get("choice:2"));
    assert_eq!(
        Some(&2.5),
        config.temperature_by_options_raw.get("choice:3-5")
    );
}

#[test]
fn the_clamp_warning_reaches_the_log_once_per_text_and_a_version_2_file_adds_no_other() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);

    // Accepted temperatures, and a version 2 file recorded for another checkpoint, say nothing:
    // the identity warning is not ported.
    CalibrationFile::parse(r#"{"temperature": [1.5, 2.0, 1.25]}"#).unwrap();
    CalibrationFile::parse(
        r#"{"version": 2, "temperature": [1.5, 2.0, 1.25], "model_id_or_path": "elsewhere",
            "subfolder": "x", "config": {"encoder": "other"}}"#,
    )
    .unwrap();
    assert!(
        SEEN.lock().unwrap().is_empty(),
        "{:?}",
        SEEN.lock().unwrap()
    );

    let clamped =
        r#"{"temperature": [9.0, true, 1.0], "temperature_by_options": {"noul:2": 0.01}}"#;
    CalibrationFile::parse(clamped).unwrap();
    CalibrationFile::parse(clamped).unwrap();
    let seen = SEEN.lock().unwrap();
    assert_eq!(1, seen.len(), "once per text: {seen:?}");
    assert_eq!("laya", seen[0].0);
    assert_eq!(log::Level::Warn, seen[0].1);
    assert_eq!(
        "laya: calibration temperatures are invalid or outside [0.5, 5]; using temperature[0]=9.0 \
         -> 5, temperature[1]=True -> 1, noul:2=0.01 -> 0.5. Treat confidence from the affected \
         entries as uncalibrated.",
        seen[0].2
    );
}

#[test]
fn the_option_defaults_to_no_file_and_a_router_hands_it_to_every_engine() {
    assert_eq!(None, LayaOptions::default().calibration);
    let mut options = LayaOptions::new();
    options.set_calibration("some/calibration.json");
    assert_eq!(
        Some(PathBuf::from("some/calibration.json")),
        options.calibration
    );
    assert!(format!("{options:?}").contains("calibration: Some(\"some/calibration.json\")"));

    struct Idle;
    impl LayaPredictor for Idle {
        fn predict(
            &self,
            _: Value,
            _: &QuestionSet,
        ) -> laya::error::Result<laya::answers::LayaResult> {
            Err(LayaError::NotSupported("idle".to_string()))
        }
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handle = Arc::clone(&seen);
    let router = LayaRouter::with_factory(
        LayaRouterOptions {
            engine_options: options,
            ..Default::default()
        },
        move |checkpoint: LayaCheckpoint, options: &LayaOptions| {
            handle
                .lock()
                .unwrap()
                .push((checkpoint, options.calibration.clone()));
            Ok(Arc::new(Idle) as Arc<dyn LayaPredictor>)
        },
    )
    .unwrap();
    router.load("en").unwrap();
    router.load("ml").unwrap();
    let path = Some(PathBuf::from("some/calibration.json"));
    assert_eq!(
        vec![
            (LayaCheckpoint::English, path.clone()),
            (LayaCheckpoint::Multilingual, path)
        ],
        *seen.lock().unwrap()
    );
}

/// The calibration cases are in each checkpoint's golden index once it is regenerated
/// (`tools/regen_golden.py`), and from then on the ordinary parity suites run them on an engine
/// loaded with the fixture. Until then there is nothing to run for a checkpoint, which this says
/// rather than passes silently.
#[test]
fn the_model_backed_cases_are_recorded_for_every_checkpoint() {
    const NAMES: [&str; 17] = [
        "calibration_plain",
        "calibration_buckets",
        "calibration_lang_override_skips_binning",
        "calibration_lang_miss_bins",
        "calibration_lang_null_entry_inherits_checkpoint",
        "calibration_lang_buckets_skip_binning",
        "calibration_table_without_lang",
        "calibration_min_confidence",
        "calibration_min_confidence_map",
        "calibration_min_confidence_buckets",
        "calibration_min_confidence_lang_override",
        "calibration_batch",
        "calibration_batch_lang_override",
        "calibration_batch_min_confidence",
        "calibration_long_scan",
        "calibration_long_single_window",
        "calibration_long_lang_override",
    ];
    for checkpoint in common::engines::checkpoints() {
        let golden = common::golden_data::CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!("skip: no golden data for '{}'", checkpoint.subdir());
            continue;
        }
        let index = golden.index();
        if !index.iter().any(|case| case.name == "calibration_plain") {
            eprintln!(
                "skip: '{}' goldens predate the calibration cases; regenerate with \
                 tools/regen_golden.py --checkpoint {}",
                checkpoint.subdir(),
                checkpoint.subdir()
            );
            continue;
        }
        for name in NAMES {
            let case = index
                .iter()
                .find(|case| case.name == name)
                .unwrap_or_else(|| {
                    panic!("{}: golden case {name} is missing", checkpoint.subdir())
                });
            let root = golden.load_case(case);
            assert!(
                common::engines::recorded_with_calibration(&root),
                "{}/{name}: not recorded with the calibration file",
                checkpoint.subdir()
            );
            common::engines::assert_fixture_is_the_recorded_file(&root);
        }
    }
}
