//! Tier 1: verifies that [`PythonJson`] reproduces Python's `json.dumps` byte-for-byte.

mod common;

use common::golden_data::CheckpointGoldenData;
use laya::python_json::PythonJson;
use serde_json::{Value, json};

/// Every checkpoint's `json_probe.json` is checked: the three dialects do
/// not depend on the checkpoint, but this also verifies all three golden directories are
/// discoverable and mutually consistent.
const CHECKPOINTS: [&str; 3] = ["multilingual", "english", "typed-decisions"];

#[test]
fn all_dialects_match_golden_for_every_checkpoint() {
    let mut checked_any = false;

    for checkpoint in CHECKPOINTS {
        let data = CheckpointGoldenData::for_checkpoint(checkpoint);
        if !data.available() {
            eprintln!(
                "skip: no golden data for '{checkpoint}' (set LAYA_GOLDEN_DIR to override the default location)"
            );
            continue;
        }
        checked_any = true;

        let probe = data.load("json_probe.json");
        let entries = probe
            .as_array()
            .expect("json_probe.json must be a JSON array");
        assert!(
            !entries.is_empty(),
            "{checkpoint}'s json_probe.json is empty"
        );

        for (i, entry) in entries.iter().enumerate() {
            let value = entry.get("value").cloned().unwrap_or(Value::Null);

            assert_dialect(
                checkpoint,
                i,
                "criterion",
                entry.get("criterion_dialect").and_then(Value::as_str),
                &PythonJson::criterion(&value),
            );
            assert_dialect(
                checkpoint,
                i,
                "instructions",
                entry.get("instructions_dialect").and_then(Value::as_str),
                &PythonJson::instructions(&value),
            );
            assert_dialect(
                checkpoint,
                i,
                "state",
                entry.get("state_dialect").and_then(Value::as_str),
                &PythonJson::state(&value),
            );
        }
    }

    if !checked_any {
        eprintln!("skip: no golden data found for any checkpoint");
    }
}

fn assert_dialect(
    checkpoint: &str,
    index: usize,
    dialect: &str,
    expected: Option<&str>,
    actual: &str,
) {
    assert_eq!(
        expected,
        Some(actual),
        "{checkpoint} json_probe[{index}] {dialect} dialect mismatch"
    );
}

// ── State(null) ─────────────────────────────────────────────────────────────

#[test]
fn state_with_null_returns_json_null() {
    // Python: serialize_state(None) -> json.dumps(None) -> "null".
    assert_eq!("null", PythonJson::state(&Value::Null));
}

// ── Repr-specific unit tests ──────────────────────────────────────────────────

#[test]
fn repr_matches_python() {
    let cases: &[(f64, &str)] = &[
        (f64::NAN, "NaN"),
        (f64::INFINITY, "Infinity"),
        (f64::NEG_INFINITY, "-Infinity"),
        (0.0, "0.0"),
        (1.0, "1.0"),
        (2.0, "2.0"),
        (-0.125, "-0.125"),
        (0.5, "0.5"),
        (1e21, "1e+21"),
        (1e-7, "1e-07"),
        (0.3333333333333333, "0.3333333333333333"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (1e17, "1e+17"),
        (0.0001, "0.0001"),
        (1e-5, "1e-05"),
        (123456789012345.6, "123456789012345.6"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (5e-324, "5e-324"),
    ];
    for &(input, expected) in cases {
        assert_eq!(expected, PythonJson::repr(input), "repr({input})");
    }
}

#[test]
fn repr_negative_zero() {
    // -0.0 needs a separate test because a float literal cannot distinguish it from 0.0 in a table.
    assert_eq!("-0.0", PythonJson::repr(-0.0));
}

// ── Escaping unit tests ────────────────────────────────────────────────────────

#[test]
fn ensure_ascii_false_keeps_literal_unicode() {
    let value = json!({"a": "naïve"});
    assert_eq!("{\"a\": \"naïve\"}", PythonJson::criterion(&value));
}

#[test]
fn instructions_no_longer_escapes_non_ascii_since_0_3_21() {
    // laya 0.3.21 changed `_to_internal`'s instructions path from `ensure_ascii=True` to
    // `ensure_ascii=False` (see `PythonJson::instructions`'s doc comment), matching `criterion`
    // and `state`. ü = U+00FC; 😀 = U+1F600 must now come through literal, not as `\uXXXX`.
    let value = json!({"a": "naïve", "b": "😀"});
    assert_eq!(
        "{\"a\": \"naïve\", \"b\": \"😀\"}",
        PythonJson::instructions(&value)
    );
}

#[test]
fn html_chars_not_escaped() {
    // Strings pass through verbatim.
    let value = Value::String("<tag> & 'quoted'".to_string());
    assert_eq!("<tag> & 'quoted'", PythonJson::criterion(&value));
}

#[test]
fn slash_not_escaped() {
    let value = json!({"url": "a/b"});
    assert_eq!("{\"url\": \"a/b\"}", PythonJson::criterion(&value));
}

#[test]
fn control_chars_escaped() {
    let value = json!({"x": "\u{1}\u{1f}"});
    assert_eq!("{\"x\": \"\\u0001\\u001f\"}", PythonJson::criterion(&value));
}

// ── Integer preservation ───────────────────────────────────────────────────────

#[test]
fn integers_never_gain_a_decimal_point() {
    // A JSON literal `3` must render as "3", not "3.0" — the whole reason serde_json's
    // int/float-preserving `Value::Number` was chosen over routing everything through f64.
    let value = json!({"open_tickets": 3});
    assert_eq!("{\"open_tickets\": 3}", PythonJson::state(&value));

    let value = json!({"open_tickets": 3.0});
    assert_eq!("{\"open_tickets\": 3.0}", PythonJson::state(&value));
}

#[test]
fn large_integer_within_i64_stays_exact() {
    let value = json!(1234567890123456789i64);
    assert_eq!("1234567890123456789", PythonJson::state(&value));
}
