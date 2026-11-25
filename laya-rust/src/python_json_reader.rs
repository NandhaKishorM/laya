//! A JSON reader that reads text the way Python's `json.loads` does, for the checkpoint files
//! Python parses with `json.load` (`rl_agent_config.json`, `tokenizer_config.json`).
//!
//! `serde_json` differs from Python in ways the checkpoint loaders can observe:
//! - it refuses the bare constants `NaN`, `Infinity` and `-Infinity`, which Python reads as floats;
//! - it refuses a float literal that overflows (`1e400`, which Python reads as `inf`);
//! - it turns an integer above `u64::MAX` (or `-0`) into a float, where Python keeps an `int`;
//! - its error messages differ from Python's `JSONDecodeError`.
//!
//! [`PyJson`] keeps all of those apart from strings and from each other, so nothing is encoded
//! in-band: an integer keeps its literal text, a float is a correctly rounded `f64` (possibly
//! non-finite), and a string is always a string. Anything Python rejects is rejected here with
//! Python's message, line, column and character offset (`Expecting ',' delimiter: line 1 column 7
//! (char 6)`); the constants are values only (never object keys), numbers follow the JSON grammar
//! as Python's scanner matches it, and trailing commas or trailing data are errors. One known
//! difference remains: a lone surrogate escape (`"\ud800"`) loads in Python and is an error here,
//! since a Rust `String` cannot hold one.

use crate::error::{LayaError, Result};
use serde::de::Error as _;
use serde_json::Value;

/// Python's recursion limit makes `json.loads` fail on arrays nested about a thousand deep.
const MAX_DEPTH: usize = 1000;

/// Python 3.11+ refuses to turn an integer literal of more than this many digits into an `int`
/// (`sys.get_int_max_str_digits()`), so `json.loads` raises `ValueError` for it. The sign does
/// not count.
const MAX_INT_DIGITS: usize = 4300;

/// A parsed JSON value with Python's number kinds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PyJson {
    Null,
    Bool(bool),
    /// A JSON integer literal (no fraction or exponent), exactly as written: Python's `int`.
    Int(String),
    /// A JSON float, including `NaN`, `Infinity`, `-Infinity` and literals that overflow to `inf`.
    Float(f64),
    Str(String),
    Array(Vec<PyJson>),
    /// Entries in file order; a repeated key keeps its first position and its last value, as a
    /// Python `dict` does.
    Object(Vec<(String, PyJson)>),
}

/// Why a read stopped: a `JSONDecodeError` (message and byte offset of the fault) or the plain
/// `ValueError` of the integer digit limit.
enum ReadError {
    Decode { message: &'static str, at: usize },
    IntegerLimit(usize),
}

type Read<T> = std::result::Result<T, ReadError>;

fn decode<T>(message: &'static str, at: usize) -> Read<T> {
    Err(ReadError::Decode { message, at })
}

impl PyJson {
    /// Parse `text` as `json.loads` does.
    pub(crate) fn parse(text: &str) -> Result<PyJson> {
        let mut reader = Reader {
            text,
            bytes: text.as_bytes(),
            pos: 0,
        };
        let outcome = if text.starts_with('\u{feff}') {
            decode("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0)
        } else {
            reader.skip_ws();
            reader.value(0).and_then(|value| {
                reader.skip_ws();
                if reader.pos == reader.bytes.len() {
                    Ok(value)
                } else {
                    decode("Extra data", reader.pos)
                }
            })
        };
        match outcome {
            Ok(value) => Ok(value),
            Err(ReadError::IntegerLimit(digits)) => Err(LayaError::InvalidConfig(format!(
                "Exceeds the limit ({MAX_INT_DIGITS} digits) for integer string conversion: value \
                 has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
            ))),
            Err(ReadError::Decode { message, at }) => {
                // `JSONDecodeError`: offsets count characters, columns start at 1.
                let before = &text[..at];
                let chars = before.chars().count();
                let line = before.matches('\n').count() + 1;
                let column = match before.rfind('\n') {
                    Some(newline) => before[newline + 1..].chars().count() + 1,
                    None => chars + 1,
                };
                Err(LayaError::Json(serde_json::Error::custom(format!(
                    "{message}: line {line} column {column} (char {chars})"
                ))))
            }
        }
    }

    /// The value of `key` when `self` is an object.
    pub(crate) fn get(&self, key: &str) -> Option<&PyJson> {
        match self {
            PyJson::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            PyJson::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Python's `float(int)` for an integer literal: correctly rounded, `None` on overflow.
    pub(crate) fn int_as_f64(literal: &str) -> Option<f64> {
        literal.parse::<f64>().ok().filter(|f| f.is_finite())
    }

    /// A `serde_json::Value` view for code that only reads strings and objects. A non-finite
    /// number has no `Value` form and becomes `null`; either way it is not a string.
    pub(crate) fn to_value(&self) -> Value {
        match self {
            PyJson::Null => Value::Null,
            PyJson::Bool(b) => Value::Bool(*b),
            PyJson::Int(literal) => literal
                .parse::<i64>()
                .map(Value::from)
                .or_else(|_| literal.parse::<u64>().map(Value::from))
                .ok()
                .or_else(|| {
                    Self::int_as_f64(literal)
                        .and_then(serde_json::Number::from_f64)
                        .map(Value::Number)
                })
                .unwrap_or(Value::Null),
            PyJson::Float(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            PyJson::Str(s) => Value::String(s.clone()),
            PyJson::Array(items) => Value::Array(items.iter().map(PyJson::to_value).collect()),
            PyJson::Object(entries) => Value::Object(
                entries
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_value()))
                    .collect(),
            ),
        }
    }
}

struct Reader<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn skip_ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn starts_with(&self, word: &str) -> bool {
        self.bytes[self.pos..].starts_with(word.as_bytes())
    }

    fn value(&mut self, depth: usize) -> Read<PyJson> {
        if depth > MAX_DEPTH {
            return decode("Expecting value", self.pos);
        }
        match self.bytes.get(self.pos) {
            Some(b'"') => self.string().map(PyJson::Str),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'n') if self.starts_with("null") => self.word("null", PyJson::Null),
            Some(b't') if self.starts_with("true") => self.word("true", PyJson::Bool(true)),
            Some(b'f') if self.starts_with("false") => self.word("false", PyJson::Bool(false)),
            Some(b'N') if self.starts_with("NaN") => self.word("NaN", PyJson::Float(f64::NAN)),
            Some(b'I') if self.starts_with("Infinity") => {
                self.word("Infinity", PyJson::Float(f64::INFINITY))
            }
            Some(b'-') if self.starts_with("-Infinity") => {
                self.word("-Infinity", PyJson::Float(f64::NEG_INFINITY))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => decode("Expecting value", self.pos),
        }
    }

    fn word(&mut self, word: &str, value: PyJson) -> Read<PyJson> {
        self.pos += word.len();
        Ok(value)
    }

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`, the longest such prefix. A
    /// literal with a fraction or an exponent is a float; otherwise it is an int.
    fn number(&mut self) -> Read<PyJson> {
        let start = self.pos;
        let digits = |reader: &mut Self| {
            let from = reader.pos;
            while matches!(reader.bytes.get(reader.pos), Some(b'0'..=b'9')) {
                reader.pos += 1;
            }
            reader.pos > from
        };
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        match self.bytes.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return decode("Expecting value", start),
        }
        let mut is_float = false;
        if self.bytes.get(self.pos) == Some(&b'.')
            && matches!(self.bytes.get(self.pos + 1), Some(b'0'..=b'9'))
        {
            self.pos += 1;
            digits(self);
            is_float = true;
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            let mark = self.pos;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if digits(self) {
                is_float = true;
            } else {
                self.pos = mark;
            }
        }
        let literal = &self.text[start..self.pos];
        if is_float {
            Ok(PyJson::Float(literal.parse::<f64>().unwrap_or(f64::NAN)))
        } else {
            let count = literal.trim_start_matches('-').len();
            if count > MAX_INT_DIGITS {
                return Err(ReadError::IntegerLimit(count));
            }
            Ok(PyJson::Int(literal.to_string()))
        }
    }

    /// Four hex digits starting at `self.pos`; `u_at` is the offset of the `u` for the error.
    /// CPython's scanner needs at least one more character after the digits (`next + 4 < len`),
    /// so a text ending right after `\u0041` has an invalid escape, not an unterminated string.
    fn hex4(&mut self, u_at: usize) -> Read<u32> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .filter(|_| self.pos + 4 < self.bytes.len())
            .and_then(|b| std::str::from_utf8(b).ok())
            .filter(|s| s.bytes().all(|b| b.is_ascii_hexdigit()));
        let Some(digits) = digits else {
            return decode("Invalid \\uXXXX escape", u_at);
        };
        self.pos += 4;
        Ok(u32::from_str_radix(digits, 16).unwrap_or(0))
    }

    fn string(&mut self) -> Read<String> {
        let quote = self.pos;
        self.pos += 1;
        let mut out = String::new();
        loop {
            let Some(&byte) = self.bytes.get(self.pos) else {
                return decode("Unterminated string starting at", quote);
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    let backslash = self.pos;
                    self.pos += 1;
                    let Some(&escape) = self.bytes.get(self.pos) else {
                        return decode("Unterminated string starting at", quote);
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let u_at = self.pos - 1;
                            let unit = self.hex4(u_at)?;
                            let is_high = (0xD800..0xDC00).contains(&unit);
                            // A pair needs room for one character after the second escape
                            // (`end + 6 < len`); without it the high surrogate stands alone and the
                            // second `\u` is scanned, and judged, as an escape of its own.
                            if is_high
                                && self.starts_with("\\u")
                                && self.pos + 6 >= self.bytes.len()
                            {
                                let second_u = self.pos + 1;
                                self.pos += 2;
                                self.hex4(second_u)?;
                            }
                            let code = if is_high && self.starts_with("\\u") {
                                let second_u = self.pos + 1;
                                self.pos += 2;
                                let low = self.hex4(second_u)?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return decode("Invalid \\uXXXX escape", second_u);
                                }
                                0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                unit
                            };
                            match char::from_u32(code) {
                                Some(ch) => out.push(ch),
                                // A lone surrogate loads in Python; a Rust string cannot hold one.
                                None => return decode("Invalid \\uXXXX escape", u_at),
                            }
                        }
                        _ => return decode("Invalid \\escape", backslash),
                    }
                }
                0..=0x1f => return decode("Invalid control character at", self.pos),
                _ => {
                    // Copy the whole run up to the next quote, backslash or control byte; these
                    // are ASCII, so the run ends on a character boundary.
                    let end = self.bytes[self.pos..]
                        .iter()
                        .position(|b| matches!(b, b'"' | b'\\' | 0..=0x1f))
                        .map_or(self.bytes.len(), |n| self.pos + n);
                    out.push_str(&self.text[self.pos..end]);
                    self.pos = end;
                }
            }
        }
    }

    fn array(&mut self, depth: usize) -> Read<PyJson> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(PyJson::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(PyJson::Array(items));
                }
                _ => return decode("Expecting ',' delimiter", self.pos),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Read<PyJson> {
        self.pos += 1;
        let mut entries: Vec<(String, PyJson)> = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(PyJson::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return decode(
                    "Expecting property name enclosed in double quotes",
                    self.pos,
                );
            }
            let key = self.string()?;
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b':') {
                return decode("Expecting ':' delimiter", self.pos);
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.value(depth + 1)?;
            match entries.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = value,
                None => entries.push((key, value)),
            }
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(PyJson::Object(entries));
                }
                _ => return decode("Expecting ',' delimiter", self.pos),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PyJson;

    // `tokenizer_config.json` is read through this reader: Python's `json.load` accepts the bare
    // constants in unrelated keys (e.g. `model_max_length`), and they are numbers, not strings.
    #[test]
    fn constants_are_floats_and_strings_stay_strings() {
        let value = PyJson::parse(
            r#"{"model_max_length": Infinity, "x": [NaN, -Infinity], "cls_token": "[CLS] NaN",
                "t": "Infinity", "big": 123456789012345678901234567890, "z": -0, "f": 1e400}"#,
        )
        .unwrap();
        assert_eq!(
            Some(&PyJson::Float(f64::INFINITY)),
            value.get("model_max_length")
        );
        assert_eq!(Some(&PyJson::Str("Infinity".into())), value.get("t"));
        assert_eq!(
            Some("[CLS] NaN"),
            value.get("cls_token").and_then(PyJson::as_str)
        );
        assert_eq!(
            Some(&PyJson::Int("123456789012345678901234567890".into())),
            value.get("big")
        );
        assert_eq!(Some(&PyJson::Int("-0".into())), value.get("z"));
        assert_eq!(Some(&PyJson::Float(f64::INFINITY)), value.get("f"));
        let as_value = value.to_value();
        assert!(!as_value["model_max_length"].is_string());
        assert!(as_value["t"].is_string());
    }

    #[test]
    fn duplicate_keys_keep_first_position_and_last_value_as_a_dict_does() {
        let value = PyJson::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(
            PyJson::Object(vec![
                ("a".into(), PyJson::Int("3".into())),
                ("b".into(), PyJson::Int("2".into()))
            ]),
            value
        );
    }

    #[test]
    fn strings_decode_escapes_and_surrogate_pairs() {
        let value = PyJson::parse(r#"["é\n𝐀\/"]"#).unwrap();
        assert_eq!(
            PyJson::Array(vec![PyJson::Str("\u{e9}\n\u{1d400}/".into())]),
            value
        );
    }

    // CPython wants one character after the four hex digits of a `\u` escape, and room for one
    // after a surrogate pair's second escape; otherwise the escape is invalid at its `u`.
    // Messages from Python 3.12's `json.loads`.
    #[test]
    fn unicode_escapes_at_the_end_of_the_text_report_like_cpython() {
        let invalid = |chars: usize| {
            format!(
                "Invalid \\uXXXX escape: line 1 column {} (char {chars})",
                chars + 1
            )
        };
        let unterminated = "Unterminated string starting at: line 1 column 1 (char 0)".to_string();
        let cases: [(&str, String); 10] = [
            (r#""a\u"#, invalid(3)),
            (r#""a\u004"#, invalid(3)),
            (r#""a\u0041"#, invalid(3)),
            (r#""a\u00411"#, unterminated.clone()),
            (r#""\ud835\udc00"#, invalid(8)),
            (r#""\ud835\udc00x"#, unterminated),
            (r#""\ud835\udc0"#, invalid(8)),
            (r#""\ud835\u00"#, invalid(8)),
            (r#""\ud835"#, invalid(2)),
            (r#"{"temperature": "A\u0041"#, invalid(19)),
        ];
        for (text, expected) in cases {
            let error = PyJson::parse(text).unwrap_err();
            assert_eq!(expected, error.to_string(), "{text}");
        }
        assert!(PyJson::parse(r#""\ud835\udc00x""#).is_ok());
        assert!(PyJson::parse(r#""a\u0041""#).is_ok());
    }
}
