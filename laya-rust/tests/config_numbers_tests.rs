//! Checkpoint config fields that Python reads with `cfg.get(...)`, and the `json.loads` limits
//! that apply to them. Expected messages were produced by Python 3.12 (`ONNXAgent.__init__`,
//! `json.loads`, and `common.build_sequence` / `state_room` with a stub tokenizer).

use laya::error::LayaError;
use laya::laya_config::LayaConfig;

fn error_of(config: &str) -> String {
    match LayaConfig::parse(config) {
        Err(LayaError::InvalidConfig(message)) => message,
        other => panic!("expected InvalidConfig for {config}, got {other:?}"),
    }
}

#[test]
fn option_layout_other_than_sequential_is_refused_as_in_python() {
    for (layout, shown) in [
        (r#""parallel""#, "'parallel'"),
        ("null", "None"),
        ("5", "5"),
        ("true", "True"),
        (r#""Sequential""#, "'Sequential'"),
    ] {
        assert_eq!(
            format!(
                "option_layout={shown} is not supported by the ONNX runtime; load this \
                 checkpoint with laya.Agent"
            ),
            error_of(&format!(r#"{{"option_layout": {layout}}}"#))
        );
    }
    assert!(LayaConfig::parse(r#"{"option_layout": "sequential"}"#).is_ok());
    assert!(LayaConfig::parse("{}").is_ok());
}

#[test]
fn integers_of_more_than_4300_digits_are_refused_as_in_python() {
    let message = |n: usize| {
        format!(
            "Exceeds the limit (4300 digits) for integer string conversion: value has {n} digits; \
             use sys.set_int_max_str_digits() to increase the limit"
        )
    };
    let digits = |n: usize| "1".repeat(n);
    assert!(LayaConfig::parse(&format!(r#"{{"x": {}}}"#, digits(4300))).is_ok());
    assert!(LayaConfig::parse(&format!(r#"{{"x": -{}}}"#, digits(4300))).is_ok());
    assert_eq!(
        message(4301),
        error_of(&format!(r#"{{"x": {}}}"#, digits(4301)))
    );
    assert_eq!(
        message(4301),
        error_of(&format!(r#"{{"x": [-{}]}}"#, digits(4301)))
    );
    // Floats have no such limit.
    assert!(LayaConfig::parse(&format!(r#"{{"x": 1.{}}}"#, "0".repeat(5000))).is_ok());
}

#[test]
fn max_len_and_head_max_len_follow_python() {
    let read = |config: &str| {
        let c = LayaConfig::parse(config).unwrap();
        (c.max_len, c.head_max_len)
    };
    assert_eq!((512, 192), read("{}"));
    assert_eq!((1, 0), read(r#"{"max_len": true, "head_max_len": false}"#));
    assert_eq!((0, 0), read(r#"{"max_len": 0, "head_max_len": -0}"#));
    assert_eq!((1024, 64), read(r#"{"max_len": 1024, "head_max_len": 64}"#));
    assert_eq!(
        (usize::MAX, 192),
        read(r#"{"max_len": 1000000000000000000000000000000}"#)
    );

    let operand =
        |kind: &str| format!("TypeError: unsupported operand type(s) for -: '{kind}' and 'int'");
    let slice = "TypeError: slice indices must be integers or None or have an __index__ method";
    for key in ["max_len", "head_max_len"] {
        assert_eq!(operand("str"), error_of(&format!(r#"{{"{key}": "512"}}"#)));
        assert_eq!(
            operand("NoneType"),
            error_of(&format!(r#"{{"{key}": null}}"#))
        );
        assert_eq!(operand("list"), error_of(&format!(r#"{{"{key}": []}}"#)));
        assert_eq!(operand("dict"), error_of(&format!(r#"{{"{key}": {{}}}}"#)));
        assert_eq!(slice, error_of(&format!(r#"{{"{key}": 512.0}}"#)));
        assert_eq!(
            format!("{key}=-5: must be a non-negative integer"),
            error_of(&format!(r#"{{"{key}": -5}}"#))
        );
    }
}

#[test]
fn a_two_megabyte_string_heavy_document_parses() {
    let mut text = String::from(r#"{"encoder": ""#);
    text.push_str(&"abc\u{e9}\u{4e2d} ".repeat(150_000)); // ~1.2 MB in one string
    text.push_str(r#"", "notes": ["#);
    for i in 0..20_000 {
        if i > 0 {
            text.push(',');
        }
        text.push_str(&format!(r#""line {i} with some text to copy""#));
    }
    text.push_str("]}");
    assert!(text.len() > 1_900_000);
    let config = LayaConfig::parse(&text).unwrap();
    assert!(config.encoder.as_deref().unwrap().len() > 1_000_000);
}

// Python opens the file in text mode (`\r\n` and `\r` become `\n`) before `json.load`, so the
// line, column and offset of a decode error count the translated text. Expected texts from
// `json.load(open(path))` on the same bytes.
#[test]
fn decode_errors_from_a_file_count_text_mode_newlines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rl_agent_config.json");
    for (bytes, expected) in [
        (
            &b"{\r\n  \"a\": 1,\r\n  }"[..],
            "Expecting property name enclosed in double quotes: line 3 column 3 (char 14)",
        ),
        (
            &b"{\r  \"a\": 1,\r  \"b\" 2}"[..],
            "Expecting ':' delimiter: line 3 column 7 (char 18)",
        ),
    ] {
        std::fs::write(&path, bytes).unwrap();
        let error = LayaConfig::load(&path).unwrap_err();
        assert_eq!(expected, error.to_string());
    }
}
