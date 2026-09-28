//! A hand-written serializer that byte-for-byte reproduces Python's `json.dumps` output for the
//! three call-site dialects Laya uses, including Python's float `repr` rules.
//!
//! `serde_json`'s own writer cannot be reused here: it uses compact separators, HTML-escapes
//! nothing (good) but also formats floats differently from Python (`2.0` -> `2.0` is fine, but the
//! exponent thresholds and digit counts differ), and has no `ensure_ascii` knob.

use serde_json::Value;

/// Namespace for the three `json.dumps` dialects Laya uses. A unit struct rather than free
/// functions so the Rust API reads the same way the ported `PythonJson` static class does.
pub struct PythonJson;

impl PythonJson {
    /// Matches Python's `serialize_state`: a string passes through verbatim; everything else
    /// (including `null`, which becomes `"null"`) is serialized with `ensure_ascii=False`.
    pub fn state(value: &Value) -> String {
        if let Value::String(s) = value {
            return s.clone();
        }
        Self::dumps(value, false)
    }

    /// Matches Python's `render_criterion`: a string passes through verbatim; anything else is
    /// serialized with `ensure_ascii=False`. (`default=str` has no Rust equivalent because
    /// `Value` only ever holds JSON-representable data to begin with; see the crate README.)
    pub fn criterion(value: &Value) -> String {
        if let Value::String(s) = value {
            return s.clone();
        }
        Self::dumps(value, false)
    }

    /// Matches Python's `_to_internal` instructions path: a string passes through verbatim;
    /// anything else is serialized with `ensure_ascii=False`, the same escaping as [`Self::state`]
    /// and [`Self::criterion`]. Laya 0.3.6 escaped non-ASCII here (`ensure_ascii=True`); 0.3.21
    /// switched to `ensure_ascii=False` because the tokenizer then read the escaped `\uXXXX` text
    /// back as literal characters instead of the real ones — on the English checkpoint one German
    /// question answered differently as a dict than as the identical plain string.
    pub fn instructions(value: &Value) -> String {
        if let Value::String(s) = value {
            return s.clone();
        }
        Self::dumps(value, false)
    }

    /// The fourth dialect: `laya/shortlist.py`'s `_query_text` builds the shortlist's embedding
    /// query as `"%s\n%s" % (instructions, serialize_state(state))`, where a non-string
    /// `instructions` value is first replaced by `json.dumps(instructions, ensure_ascii=False)`.
    /// That is byte-for-byte the same rule as [`Self::state`] (string passes through verbatim,
    /// everything else becomes `ensure_ascii=False` JSON) applied to the instructions value
    /// instead of the state value — a separate named entry point because it is a distinct call
    /// site in `laya_shortlist.rs`, not because the formatting differs.
    pub fn shortlist_instructions(value: &Value) -> String {
        Self::state(value)
    }

    /// The underlying serializer: `json.dumps(value, ensure_ascii=escape_non_ascii,
    /// separators=(", ", ": "))`. Strings are still quoted and escaped here (unlike the three
    /// dialect entry points above, which special-case a top-level string to pass through
    /// unquoted).
    fn dumps(value: &Value, escape_non_ascii: bool) -> String {
        let mut out = String::new();
        Self::write_value(&mut out, value, escape_non_ascii);
        out
    }

    fn write_value(out: &mut String, value: &Value, escape_non_ascii: bool) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::String(s) => Self::write_string(out, s, escape_non_ascii),
            Value::Number(n) => Self::write_number(out, n),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    Self::write_value(out, item, escape_non_ascii);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    Self::write_string(out, k, escape_non_ascii);
                    out.push_str(": ");
                    Self::write_value(out, v, escape_non_ascii);
                }
                out.push('}');
            }
        }
    }

    fn write_number(out: &mut String, n: &serde_json::Number) {
        if n.is_f64() {
            // A JSON literal with a `.` or exponent always parses to Number::Float, so this is
            // exactly Python's int/float distinction (see json-dialects.md "Integer preservation").
            out.push_str(&Self::repr(
                n.as_f64().expect("is_f64 implies as_f64 succeeds"),
            ));
        } else {
            // Integers (positive or negative) render as their plain decimal digits, same as Python.
            out.push_str(&n.to_string());
        }
    }

    fn write_string(out: &mut String, s: &str, escape_non_ascii: bool) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c if escape_non_ascii && (c as u32) > 0x7f => {
                    let cp = c as u32;
                    if cp > 0xFFFF {
                        // Astral character: emit as a UTF-16 surrogate pair, matching Python.
                        let v = cp - 0x10000;
                        let high = 0xD800 + (v >> 10);
                        let low = 0xDC00 + (v & 0x3FF);
                        out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                    } else {
                        out.push_str(&format!("\\u{cp:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    /// Formats an `f64` exactly as Python's `repr(float)`: shortest round-trip decimal digits,
    /// always containing a `.` or `e`, exponent notation when the decimal exponent is `>= 16` or
    /// `<= -5`. Ports `PythonJson.Repr(double)`.
    pub fn repr(value: f64) -> String {
        if value.is_nan() {
            return "NaN".to_string();
        }
        if value.is_infinite() {
            return if value > 0.0 {
                "Infinity".to_string()
            } else {
                "-Infinity".to_string()
            };
        }
        if value == 0.0 {
            return if value.is_sign_negative() {
                "-0.0".to_string()
            } else {
                "0.0".to_string()
            };
        }

        let neg = value.is_sign_negative();
        let abs = value.abs();

        // Rust's `{:e}` for f64 already gives the shortest decimal that round-trips (same
        // algorithm family Display uses), normalized to exactly one digit before the point — e.g.
        // "3.333e-1", "1e0", "5e-324". That single-leading-digit form is exactly what the
        // reformatting below assumes, so there is no need to search for the decimal point or
        // exponent marker location separately.
        let raw = format!("{abs:e}");
        let e_pos = raw.find('e').expect("LowerExp output always contains 'e'");
        let mantissa = &raw[..e_pos];
        let exp10: i32 = raw[e_pos + 1..]
            .parse()
            .expect("LowerExp exponent is always a valid integer");

        let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
        while digits.len() > 1 && digits.ends_with('0') {
            digits.pop();
        }

        let prefix = if neg { "-" } else { "" };

        if exp10 >= 16 || exp10 <= -5 {
            let mantissa_str = if digits.len() == 1 {
                digits.clone()
            } else {
                format!("{}.{}", &digits[..1], &digits[1..])
            };
            let exp_str = if exp10 >= 0 {
                format!("e+{exp10:02}")
            } else {
                format!("e-{:02}", -exp10)
            };
            format!("{prefix}{mantissa_str}{exp_str}")
        } else if exp10 >= 0 {
            // Fixed notation, |value| >= 1.
            let int_digit_count = (exp10 + 1) as usize;
            if digits.len() <= int_digit_count {
                let mut s = digits.clone();
                while s.len() < int_digit_count {
                    s.push('0');
                }
                format!("{prefix}{s}.0")
            } else {
                format!(
                    "{prefix}{}.{}",
                    &digits[..int_digit_count],
                    &digits[int_digit_count..]
                )
            }
        } else {
            // Fixed notation, 0 < |value| < 1.
            let leading_zeros = (-exp10 - 1) as usize;
            format!("{prefix}0.{}{digits}", "0".repeat(leading_zeros))
        }
    }
}
