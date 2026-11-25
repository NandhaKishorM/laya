//! `float(str)` for string temperatures, as Python's `clamp_temperature` reads them. The applied
//! values and the warning text were produced by Python's `clamp_temperature`,
//! `temperature_rejected` and the loader's format string on the same 37 strings.

use laya::laya_config::LayaConfig;
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());
static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
struct Capture;
impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        SEEN.lock().unwrap().push(record.args().to_string());
    }
    fn flush(&self) {}
}
static LOGGER: Capture = Capture;

const CONFIG: &str = "{\"temperature\": [\"2.5\", \" 2.5 \", \"\\t2.5\\n\", \"+2.5\", \"-1\", \"1e1\", \"1E-1\", \".5\", \"5.\", \"2_5\", \"1_000\", \"_1\", \"1_\", \"1__0\", \"1_.5\", \"1e_1\", \"0x1\", \"abc\", \"\", \"nan\", \"-inf\", \"Infinity\", \"+INF\", \"\\uff12.\\uff15\", \"\\u0662\", \"\\u00a02\", \"\\u30002.5\\u3000\", \"\\u00852\", \"1e500\", \"1_0e1_0\", \"\\ud835\\udfd0\", \".\", \"e5\", \"1e\", \"1.5.2\", \"--1\", \"2 5\"]}";
const APPLIED: [f64; 37] = [
    2.5, 2.5, 2.5, 2.5, 0.5, 5.0, 0.5, 0.5, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 2.5, 2.0, 2.0, 2.5, 2.0, 1.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const WARNING: &str = "laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort confidence; clamping temperature[4]='-1' -> 0.5, temperature[5]='1e1' -> 5, temperature[6]='1E-1' -> 0.5, temperature[9]='2_5' -> 5, temperature[10]='1_000' -> 5, temperature[11]='_1' -> 1, temperature[12]='1_' -> 1, temperature[13]='1__0' -> 1, temperature[14]='1_.5' -> 1, temperature[15]='1e_1' -> 1, temperature[16]='0x1' -> 1, temperature[17]='abc' -> 1, temperature[18]='' -> 1, temperature[19]='nan' -> 1, temperature[20]='-inf' -> 1, temperature[21]='Infinity' -> 1, temperature[22]='+INF' -> 1, temperature[28]='1e500' -> 1, temperature[29]='1_0e1_0' -> 5, temperature[31]='.' -> 1, temperature[32]='e5' -> 1, temperature[33]='1e' -> 1, temperature[34]='1.5.2' -> 1, temperature[35]='--1' -> 1, temperature[36]='2 5' -> 1. Treat confidence from the affected buckets as uncalibrated.";

#[test]
fn string_temperatures_follow_python_float_parsing() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    SEEN.lock().unwrap().clear();

    let config = LayaConfig::parse(CONFIG).unwrap();
    assert_eq!(APPLIED.to_vec(), config.temperature);
    let seen = SEEN.lock().unwrap();
    assert_eq!(vec![WARNING.to_string()], *seen);
}

// Python 0.4.20: json.loads gives an `int` for a 400-digit literal; `float()` raises OverflowError,
// which `clamp_temperature` and `temperature_rejected` catch, so the entry becomes the neutral 1.0
// and the loader's warning names it digit for digit. `1e400` is a float (inf -> 1.0); 10**308 is
// still convertible (-> 5.0); 2**1024 is the smallest power of two that is not.
#[test]
fn huge_integer_temperatures_fall_back_to_neutral_like_float_literals() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    let list = |entry: &str| format!(r#"{{"temperature": [1.0, {entry}, 1.0]}}"#);
    let by_options =
        |entry: &str| format!(r#"{{"temperature_by_options": {{"choice:2": {entry}}}}}"#);
    let huge = format!("1{}", "0".repeat(400));
    let two_1024 = "179769313486231590772930519078902473361797697894230657273430081157732675805500963132708477322407536021120113879871393357658789768814416622492847430639474124377767893424865485276302219601246094119453082952085005768838150682342462881473913110540827237163350510684586298239947245938479716304835356329624224137216";
    for literal in [huge.clone(), format!("-{huge}"), two_1024.to_string()] {
        SEEN.lock().unwrap().clear();
        let config = LayaConfig::parse(&list(&literal)).unwrap();
        assert_eq!(vec![1.0, 1.0, 1.0], config.temperature, "{literal}");
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(
            vec![format!(
                "laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort \
                 confidence; clamping temperature[1]={literal} -> 1. Treat confidence from the \
                 affected buckets as uncalibrated."
            )],
            seen
        );

        SEEN.lock().unwrap().clear();
        let config = LayaConfig::parse(&by_options(&literal)).unwrap();
        assert_eq!(Some(&1.0), config.temperature_by_options.get("choice:2"));
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(1, seen.len(), "{seen:?}");
        assert!(
            seen[0].contains(&format!("clamping choice:2={literal} -> 1. Treat")),
            "{}",
            seen[0]
        );
    }
    SEEN.lock().unwrap().clear();
    for (entry, applied) in [
        ("1e400".to_string(), 1.0),
        ("-1e400".to_string(), 1.0),
        (format!("1.{}", "0".repeat(400)), 1.0),
        (format!("1{}", "0".repeat(308)), 5.0),
    ] {
        let config = LayaConfig::parse(&list(&entry)).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(vec![1.0, applied, 1.0], config.temperature, "{entry}");
    }
    // Python: temperature[1]=inf -> 1 (a float, so repr is `inf`, not a string)
    let seen = SEEN.lock().unwrap();
    assert!(
        seen[0].contains("clamping temperature[1]=inf -> 1. Treat"),
        "{}",
        seen[0]
    );
    assert!(
        seen[1].contains("clamping temperature[1]=-inf -> 1. Treat"),
        "{}",
        seen[1]
    );
}

// Python's json.loads reads the bare tokens NaN, Infinity and -Infinity as floats (nothing else:
// `nan`, `+Infinity`, `Inf` and `01` are errors there). Expected text from Python's own
// clamp_temperature / temperature_rejected on the loaded values.
#[test]
fn bare_nan_and_infinity_tokens_load_as_in_python() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    SEEN.lock().unwrap().clear();

    let config = LayaConfig::parse(
        r#"{"temperature": [NaN, 1.2, Infinity, -Infinity],
            "temperature_by_options": {"choice:2": NaN, "note": "NaN"}}"#,
    )
    .unwrap();
    assert_eq!(vec![1.0, 1.2, 1.0, 1.0], config.temperature);
    assert_eq!(Some(&1.0), config.temperature_by_options.get("choice:2"));
    assert_eq!(
        vec![
            "laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort \
             confidence; clamping choice:2=nan -> 1, note='NaN' -> 1, temperature[0]=nan -> 1, \
             temperature[2]=inf -> 1, temperature[3]=-inf -> 1. Treat confidence from the \
             affected buckets as uncalibrated."
                .to_string()
        ],
        *SEEN.lock().unwrap()
    );
    for bad in ["nan", "+Infinity", "Inf", "-NaN", "Infinityx", "01", "-"] {
        let text = format!(r#"{{"temperature": [{bad}, 1.0, 1.0]}}"#);
        assert!(
            LayaConfig::parse(&text).is_err(),
            "{bad} must stay a JSON error"
        );
    }
}

fn warnings_for(config: &str) -> Vec<String> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    SEEN.lock().unwrap().clear();
    LayaConfig::parse(config).unwrap();
    SEEN.lock().unwrap().clone()
}

fn onnx_warning(clamping: &str) -> String {
    format!(
        "laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort \
         confidence; clamping {clamping}. Treat confidence from the affected buckets as \
         uncalibrated."
    )
}

// B1: Python keeps `-0` and integers above 2**64 as ints, so the warning prints them as ints.
#[test]
fn warning_prints_json_integers_as_python_ints() {
    assert_eq!(
        vec![onnx_warning("temperature[0]=0 -> 0.5")],
        warnings_for(r#"{"temperature":[-0,1,1]}"#)
    );
    assert_eq!(
        vec![onnx_warning(
            "temperature[0]=123456789012345678901234567890 -> 5"
        )],
        warnings_for(r#"{"temperature":[123456789012345678901234567890,1,1]}"#)
    );
    assert_eq!(
        vec![onnx_warning(
            "temperature[0]=-123456789012345678901234567890 -> 0.5, \
             temperature[1]=18446744073709551616 -> 5, temperature[2]=1e+22 -> 5, \
             temperature[3]=1.5e+300 -> 5, temperature[4]=0.0 -> 0.5, \
             temperature[5]=-0.0 -> 0.5, temperature[6]=100.0 -> 5"
        )],
        warnings_for(
            r#"{"temperature":[-123456789012345678901234567890, 18446744073709551616, 1e22,
                1.5e300, 0.0, -0.0, 1E2]}"#
        )
    );
}

// B3: a real string that spells an internal marker is just a string.
#[test]
fn strings_that_look_like_internal_markers_stay_strings() {
    assert_eq!(
        vec![onnx_warning(
            "temperature[0]='__laya_nan__' -> 1, temperature[1]='__laya_inf__' -> 1, \
             temperature[2]='__laya_int_too_large_to_convert_to_float__' -> 1"
        )],
        warnings_for(
            r#"{"temperature":["__laya_nan__","__laya_inf__",
                "__laya_int_too_large_to_convert_to_float__",1,1]}"#
        )
    );
}

// B2: Python's json.loads rejects all of these (checked with Python 3.12).
#[test]
fn malformed_json_stays_rejected() {
    for bad in [
        "{NaN: 1}",
        "{Infinity: 1}",
        r#"{"a": 1, -Infinity: 2}"#,
        "[NaN,]",
        r#"{"a":NaN,}"#,
        "[Infinity Infinity]",
        r#"{"a" NaN}"#,
        "NaN NaN",
        r#"{"a":-}"#,
        "[1.]",
        "[.5]",
        "[+1]",
        "[01]",
        "[nan]",
        r#"{"a":tru}"#,
        "[1e]",
    ] {
        assert!(LayaConfig::parse(bad).is_err(), "{bad} must be rejected");
    }
}

// N2: Python's default filter shows a warning once per (text, line of the attributed frame).
// The attributed line is the caller's (`stacklevel`), which `#[track_caller]` reproduces.
fn capture(run: impl FnOnce()) -> Vec<String> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    SEEN.lock().unwrap().clear();
    run();
    SEEN.lock().unwrap().clone()
}

#[test]
fn a_warning_repeated_from_one_line_is_emitted_once() {
    let config = r#"{"temperature":[0.25,1,1]}"#;
    let seen = capture(|| {
        for _ in 0..3 {
            LayaConfig::parse(config).unwrap();
        }
    });
    assert_eq!(1, seen.len());
}

#[test]
fn the_same_warning_from_two_lines_is_emitted_twice() {
    let config = r#"{"temperature":[0.27,1,1]}"#;
    let seen = capture(|| {
        LayaConfig::parse(config).unwrap();
        LayaConfig::parse(config).unwrap();
    });
    assert_eq!(2, seen.len());
    assert_eq!(seen[0], seen[1]);
}

// The router builds engines from one fixed line inside the library (it has no `#[track_caller]`),
// so loading through it warns once per process however many lines call it, as in Python.
fn like_the_router(config: &str) {
    LayaConfig::parse(config).unwrap();
}

#[test]
fn a_warning_through_a_fixed_library_line_is_emitted_once() {
    let config = r#"{"temperature":[0.29,1,1]}"#;
    let seen = capture(|| {
        like_the_router(config);
        like_the_router(config);
    });
    assert_eq!(1, seen.len());
    assert_eq!(
        1,
        capture(|| like_the_router(r#"{"temperature":[0.31,1,1]}"#)).len()
    );
}

// `ONNXAgent.__init__` iterates `cfg.get("temperature", ...)` and calls `.items()` on
// `cfg.get("temperature_by_options", {})` while the checkpoint loads, whatever JSON type they are.
// Messages and warnings below come from running those lines in Python 3.12.
#[test]
fn wrongly_typed_temperature_containers_follow_python() {
    let error_of = |config: &str| match LayaConfig::parse(config) {
        Err(laya::error::LayaError::InvalidConfig(message)) => message,
        other => panic!("expected InvalidConfig for {config}, got {other:?}"),
    };
    for (value, kind) in [
        ("5", "int"),
        ("2.5", "float"),
        ("true", "bool"),
        ("null", "NoneType"),
    ] {
        assert_eq!(
            format!("TypeError: '{kind}' object is not iterable"),
            error_of(&format!(r#"{{"temperature": {value}}}"#))
        );
    }
    for (value, kind) in [
        ("[1]", "list"),
        (r#""x""#, "str"),
        ("null", "NoneType"),
        ("5", "int"),
        ("1.5", "float"),
        ("true", "bool"),
    ] {
        assert_eq!(
            format!("AttributeError: '{kind}' object has no attribute 'items'"),
            error_of(&format!(r#"{{"temperature_by_options": {value}}}"#))
        );
    }
    // The list is walked first, so its error wins.
    assert_eq!(
        "TypeError: 'int' object is not iterable",
        error_of(r#"{"temperature": 5, "temperature_by_options": [1]}"#)
    );

    // A string iterates its characters and a dict its keys, each a one-entry temperature.
    assert_eq!(
        vec![onnx_warning(
            "temperature[0]='a' -> 1, temperature[1]='b' -> 1, temperature[2]='c' -> 1"
        )],
        warnings_for(r#"{"temperature": "abc"}"#)
    );
    assert_eq!(
        vec![onnx_warning(
            "temperature[0]='a' -> 1, temperature[1]='9' -> 5"
        )],
        warnings_for(r#"{"temperature": {"a": 1, "9": 2}}"#)
    );
    assert_eq!(
        vec![onnx_warning(
            "k='b' -> 1, j=9 -> 5, temperature[0]='a' -> 1"
        )],
        warnings_for(r#"{"temperature": "a", "temperature_by_options": {"k": "b", "j": 9}}"#)
    );
    assert_eq!(
        vec![onnx_warning(
            "temperature[0]='0' -> 0.5, temperature[1]='.' -> 1"
        )],
        warnings_for(r#"{"temperature": "0.5"}"#)
    );
    // "123" is three valid temperatures and "" and {} are none: no warning.
    for config in [
        r#"{"temperature": "123"}"#,
        r#"{"temperature": ""}"#,
        r#"{"temperature": {}}"#,
    ] {
        assert!(warnings_for(config).is_empty(), "{config}");
    }
    assert_eq!(
        vec![1.0, 2.0, 3.0],
        LayaConfig::parse(r#"{"temperature": "123"}"#)
            .unwrap()
            .temperature
    );
}
