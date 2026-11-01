//! The contents of a checkpoint's `rl_agent_config.json`. Ports the config shape `laya/agent.py`'s
//! `Agent.system_one` reads via `cfg.get(...)`.
//!
//! Defaults match Python's `cfg.get(...)` fallbacks in `Agent.system_one`, not the shipped
//! multilingual values, so a config missing a key behaves the same in both runtimes.

use crate::calibration::Calibration;
use crate::error::{LayaError, Result};
use crate::lang_temperatures::{self, LangTemperature, LangTemperatures};
use crate::language_detection::py_strip;
use crate::laya_router::python_repr_str;
use crate::long_prediction::emit_warnings;
use crate::python_json::PythonJson;
use crate::python_json_reader::PyJson;
use std::collections::HashMap;
use std::path::Path;
use unicode_general_category::{GeneralCategory, get_general_category};

/// The parsed and clamped contents of `rl_agent_config.json`.
#[derive(Debug, Clone)]
pub struct LayaConfig {
    /// The encoder the checkpoint was trained on, for information only.
    pub encoder: Option<String>,
    /// Maximum total sequence length.
    pub max_len: usize,
    /// Token budget for the question head: instructions plus all option fragments.
    pub head_max_len: usize,
    /// Per-question-type temperature, clamped to the usable range. Always 3 entries.
    pub temperature: Vec<f64>,
    /// Per `(type, option-count)` bucket temperature, clamped. Keys as
    /// [`Calibration::temp_bucket`] builds them.
    pub temperature_by_options: HashMap<String, f64>,
    /// What the checkpoint shipped, before clamping.
    pub temperature_raw: Vec<f64>,
    /// What the checkpoint shipped, before clamping.
    pub temperature_by_options_raw: HashMap<String, f64>,
    /// Descriptions of every temperature that had to be clamped, empty when none were. A
    /// non-empty list means confidence from those buckets is uncalibrated. [`Self::parse`] also
    /// sends Python's warning for it to the `log` facade (`warn`, target `laya`).
    pub clamped_temperatures: Vec<String>,
}

/// The value of a Unicode decimal digit (`Nd`), as Python's `int()`/`float()` read it. The `Nd`
/// characters come in runs of ten starting at zero, and runs can touch (the mathematical digits),
/// so the value is the distance from the start of the unbroken `Nd` stretch, modulo ten.
pub(crate) fn decimal_digit_value(ch: char) -> Option<u32> {
    let is_nd = |c: char| get_general_category(c) == GeneralCategory::DecimalNumber;
    if !is_nd(ch) {
        return None;
    }
    let mut back = 0u32;
    let mut code = ch as u32;
    while code > 0 && char::from_u32(code - 1).is_some_and(is_nd) {
        back += 1;
        code -= 1;
    }
    Some(back % 10)
}

/// Python's `float(str)`: surrounding whitespace is stripped, then an optional sign and either
/// `inf`/`infinity`/`nan` (any case) or decimal digits (any script) with at most one `_` between
/// two digits, an optional `.` and fraction, and an optional exponent. Anything else is `None`
/// (Python's `ValueError`).
pub(crate) fn py_float(text: &str) -> Option<f64> {
    let text = py_strip(text);
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let lower = body.to_ascii_lowercase();
    let magnitude = match lower.as_str() {
        "inf" | "infinity" => f64::INFINITY,
        "nan" => f64::NAN,
        _ => {
            let ascii = ascii_float_text(body)?;
            ascii.parse::<f64>().ok()?
        }
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// `digitpart ["." [digitpart]] | "." digitpart`, then `[eE] [sign] digitpart`, rewritten with
/// ASCII digits and no underscores; `None` when `body` does not have that shape.
fn ascii_float_text(body: &str) -> Option<String> {
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    // A run of digits with single underscores between digits; returns how many digits it read.
    let digits = |i: &mut usize, out: &mut String| -> usize {
        let mut count = 0;
        while *i < chars.len() {
            if let Some(d) = decimal_digit_value(chars[*i]) {
                out.push(char::from_digit(d, 10).unwrap());
                count += 1;
                *i += 1;
            } else if chars[*i] == '_'
                && count > 0
                && chars
                    .get(*i + 1)
                    .is_some_and(|c| decimal_digit_value(*c).is_some())
            {
                *i += 1;
            } else {
                break;
            }
        }
        count
    };
    let whole = digits(&mut i, &mut out);
    let mut fraction = 0;
    if chars.get(i) == Some(&'.') {
        out.push('.');
        i += 1;
        fraction = digits(&mut i, &mut out);
    }
    if whole + fraction == 0 {
        return None;
    }
    if matches!(chars.get(i), Some('e' | 'E')) {
        out.push('e');
        i += 1;
        if let Some(&sign @ ('+' | '-')) = chars.get(i) {
            out.push(sign);
            i += 1;
        }
        if digits(&mut i, &mut out) == 0 {
            return None;
        }
    }
    (i == chars.len()).then_some(out)
}

/// A JSON temperature entry as `clamp_temperature` reads it: a number as itself, a string through
/// `float(str)`, and a bool or anything else not at all. An integer too wide for a double does not
/// convert either (Python's `OverflowError`, which `clamp_temperature` and `temperature_rejected`
/// catch since 0.4.20), so it is `None` here and becomes the neutral 1.0, named in the warning.
pub(crate) fn temperature_number(raw: &PyJson) -> Option<f64> {
    match raw {
        PyJson::Int(literal) => PyJson::int_as_f64(literal),
        PyJson::Float(f) => Some(*f),
        PyJson::Str(s) => py_float(s),
        _ => None,
    }
}

/// Python's name for the type of a JSON value, as it appears in `TypeError` messages.
pub(crate) fn python_type_name(value: &PyJson) -> &'static str {
    match value {
        PyJson::Null => "NoneType",
        PyJson::Bool(_) => "bool",
        PyJson::Int(_) => "int",
        PyJson::Float(_) => "float",
        PyJson::Str(_) => "str",
        PyJson::Array(_) => "list",
        PyJson::Object(_) => "dict",
    }
}

/// What `for t in cfg.get("temperature", [1.0, 1.0, 1.0])` visits: a list's items, the
/// characters of a string, the keys of a dict (each a string), and for anything else
/// (`5`, `2.5`, `true`, `null`) Python's `TypeError`, raised while the checkpoint loads.
fn temperature_elements(root: &PyJson) -> Result<Vec<PyJson>> {
    match root.get("temperature") {
        None => Ok(vec![PyJson::Float(1.0); 3]),
        Some(PyJson::Array(items)) => Ok(items.clone()),
        Some(PyJson::Str(text)) => Ok(text.chars().map(|c| PyJson::Str(c.to_string())).collect()),
        Some(PyJson::Object(entries)) => Ok(entries
            .iter()
            .map(|(k, _)| PyJson::Str(k.clone()))
            .collect()),
        Some(other) => Err(LayaError::InvalidConfig(format!(
            "TypeError: '{}' object is not iterable",
            python_type_name(other)
        ))),
    }
}

/// What `cfg.get("temperature_by_options", {}).items()` yields: a dict's entries, and for
/// anything else Python's `AttributeError`, raised while the checkpoint loads.
fn temperature_options(root: &PyJson) -> Result<Vec<(String, PyJson)>> {
    match root.get("temperature_by_options") {
        None => Ok(Vec::new()),
        Some(PyJson::Object(entries)) => Ok(entries.clone()),
        Some(other) => Err(LayaError::InvalidConfig(format!(
            "AttributeError: '{}' object has no attribute 'items'",
            python_type_name(other)
        ))),
    }
}

/// `%g` of a clamped temperature (six significant digits, trailing zeros dropped); the value is
/// always inside `[0.5, 5]` or the neutral `1`, so no exponent form is needed.
pub(crate) fn percent_g(value: f64) -> String {
    let exponent: i32 = format!("{value:.5e}")
        .split('e')
        .nth(1)
        .and_then(|e| e.parse().ok())
        .unwrap_or(0);
    let decimals = (5 - exponent).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// Python's `repr()` of a JSON value as `json.load` would have produced it.
pub(crate) fn py_repr(value: &PyJson) -> String {
    match value {
        PyJson::Null => "None".to_string(),
        PyJson::Bool(true) => "True".to_string(),
        PyJson::Bool(false) => "False".to_string(),
        PyJson::Int(literal) => match literal.strip_prefix('-') {
            // Python's `int('-0')` is `0`.
            Some("0") => "0".to_string(),
            _ => literal.clone(),
        },
        PyJson::Float(f) if f.is_nan() => "nan".to_string(),
        PyJson::Float(f) if f.is_infinite() => if *f > 0.0 { "inf" } else { "-inf" }.to_string(),
        PyJson::Float(f) => PythonJson::repr(*f),
        PyJson::Str(s) => python_repr_str(s),
        PyJson::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        PyJson::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", python_repr_str(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The text of the `warnings.warn` in `ONNXAgent.__init__` for a checkpoint whose temperatures
/// had to be replaced, or `None` when every one was accepted. Entries are named, in Python's
/// order (`temperature_by_options` in file order, then `temperature[i]`), with the value as
/// shipped (`%r`) and the one applied (`%g`).
fn temperature_warning(options: &[(String, PyJson)], elements: &[PyJson]) -> Option<String> {
    let mut entries: Vec<(String, &PyJson)> = Vec::new();
    entries.extend(options.iter().map(|(k, v)| (k.clone(), v)));
    entries.extend(
        elements
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("temperature[{i}]"), v)),
    );
    let rejected: Vec<String> = entries
        .into_iter()
        .filter_map(|(name, raw)| {
            // A bool or anything that is not a number is replaced by the neutral 1.0.
            let number = temperature_number(raw);
            let applied = Calibration::clamp_temperature(number.unwrap_or(f64::NAN));
            (number != Some(applied))
                .then(|| format!("{name}={} -> {}", py_repr(raw), percent_g(applied)))
        })
        .collect();
    if rejected.is_empty() {
        return None;
    }
    Some(format!(
        "laya ONNX: this checkpoint ships temperatures outside [{}, {}] which would distort \
         confidence; clamping {}. Treat confidence from the affected buckets as uncalibrated.",
        percent_g(Calibration::TEMP_MIN),
        percent_g(Calibration::TEMP_MAX),
        rejected.join(", ")
    ))
}

/// `ONNXAgent.__init__`: the exported graph takes the sequential layout only.
fn check_option_layout(root: &PyJson) -> Result<()> {
    match root.get("option_layout") {
        None => Ok(()),
        Some(PyJson::Str(layout)) if layout == "sequential" => Ok(()),
        Some(other) => Err(LayaError::InvalidConfig(format!(
            "option_layout={} is not supported by the ONNX runtime; load this checkpoint with \
             laya.Agent",
            py_repr(other)
        ))),
    }
}

/// `cfg.get(key, default)` for `max_len` / `head_max_len`, as Python's arithmetic and slicing
/// read it. Accepted: a non-negative integer, a bool (`true` is 1, `false` is 0), `-0` (0), and
/// an integer beyond the platform's range (as good as unbounded). Refused at load:
/// - a float, string, `null`, list or object: Python's `TypeError`, which it raises the first
///   time a prediction builds a sequence;
/// - a negative integer: Python slices from the end for it, which this port does not emulate.
fn config_length(root: &PyJson, key: &str, default: usize) -> Result<usize> {
    let refused = |kind: &str| LayaError::InvalidConfig(length_type_error(kind));
    match root.get(key) {
        None => Ok(default),
        Some(PyJson::Bool(b)) => Ok(usize::from(*b)),
        Some(PyJson::Int(literal)) => {
            let digits = literal.trim_start_matches('-');
            if digits.bytes().all(|b| b == b'0') {
                Ok(0)
            } else if literal.starts_with('-') {
                Err(LayaError::InvalidConfig(format!(
                    "{key}={literal}: must be a non-negative integer"
                )))
            } else {
                Ok(digits
                    .parse::<u64>()
                    .map_or(usize::MAX, |v| usize::try_from(v).unwrap_or(usize::MAX)))
            }
        }
        Some(PyJson::Float(_)) => Err(refused("float")),
        Some(PyJson::Str(_)) => Err(refused("str")),
        Some(PyJson::Null) => Err(refused("NoneType")),
        Some(PyJson::Array(_)) => Err(refused("list")),
        Some(PyJson::Object(_)) => Err(refused("dict")),
    }
}

/// Python's `TypeError` text when `kind` is used as a length in `x - len(...)` or a slice.
fn length_type_error(kind: &str) -> String {
    if kind == "float" {
        "TypeError: slice indices must be integers or None or have an __index__ method".to_string()
    } else {
        format!("TypeError: unsupported operand type(s) for -: '{kind}' and 'int'")
    }
}

/// What Python only discovers when a prediction needs it, kept by the engine (not part of the
/// public [`LayaConfig`]).
#[derive(Debug, Clone)]
pub(crate) struct Deferred {
    /// The real length of a `temperature` list shorter than three: Python indexes
    /// `self.temperature[qtype]` for every question, so a question type at or past it raises
    /// `IndexError`, however its bucket resolves.
    pub(crate) short_temperature: Option<usize>,
    /// The resolved `lang_temperatures` option (Python's `self.lang_temperatures`).
    pub(crate) lang_temperatures: LangTemperatures,
}

/// Python opens the file in text mode, which turns `\r\n` and a lone `\r` into `\n` before
/// `json.load` sees the text; the line, column and offset in a decode error count the result.
pub(crate) fn python_text_mode(text: String) -> String {
    if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text
    }
}

impl LayaConfig {
    /// Read a config from a `rl_agent_config.json` file.
    #[track_caller]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&python_text_mode(std::fs::read_to_string(path)?))
    }

    /// [`Self::parse`] with a `lang_temperatures` table resolved the way the engine resolves it:
    /// right after the checkpoint's temperatures are clamped and before it warns about them. Hidden
    /// from the docs: it exists so a test can check that order without a model.
    #[doc(hidden)]
    #[track_caller]
    pub fn parse_with_lang_temperatures(
        json: &str,
        table: &[(String, Option<LangTemperature>)],
    ) -> Result<Self> {
        Self::parse_with(json, table, std::panic::Location::caller()).map(|(config, _)| config)
    }

    /// [`Self::load`] for the engine, which also keeps what Python only discovers at prediction
    /// time.
    #[track_caller]
    pub(crate) fn load_deferred(
        path: impl AsRef<Path>,
        table: &[(String, Option<LangTemperature>)],
    ) -> Result<(Self, Deferred)> {
        Self::parse_with(
            &python_text_mode(std::fs::read_to_string(path)?),
            table,
            std::panic::Location::caller(),
        )
    }

    /// Parse a config from JSON text.
    #[track_caller]
    pub fn parse(json: &str) -> Result<Self> {
        Self::parse_with(json, &[], std::panic::Location::caller()).map(|(config, _)| config)
    }

    fn parse_with(
        json: &str,
        table: &[(String, Option<LangTemperature>)],
        at: &std::panic::Location<'_>,
    ) -> Result<(Self, Deferred)> {
        let root = PyJson::parse(json)?;
        check_option_layout(&root)?;
        // Python builds `[clamp_temperature(t) for t in temperature]` and then the by-options dict,
        // so each container's error comes in that order.
        let elements = temperature_elements(&root)?;
        let options = temperature_options(&root)?;

        // `ONNXAgent.__init__` resolves `lang_temperatures` right after clamping the checkpoint's
        // own, and only then warns about what it clamped: an invalid table fails the load before
        // that warning, and a table's warnings come first.
        let temperature_base = root
            .get("temperature")
            .cloned()
            .unwrap_or_else(|| PyJson::Array(vec![PyJson::Float(1.0); 3]));
        let lang_temperatures = lang_temperatures::resolve_json(table, &temperature_base, at)?;

        let mut temperature_raw: Vec<f64> = elements
            .iter()
            .map(|v| temperature_number(v).unwrap_or(f64::NAN))
            .collect();
        // Python indexes self.temperature[qtype] for qtype 0..2, so a short list would be an
        // IndexError there; pad to 3 with the neutral value rather than failing later per-question.
        while temperature_raw.len() < 3 {
            temperature_raw.push(1.0);
        }

        let temperature_by_options_raw: HashMap<String, f64> = options
            .iter()
            .map(|(k, v)| (k.clone(), temperature_number(v).unwrap_or(f64::NAN)))
            .collect();

        let encoder = root
            .get("encoder")
            .and_then(PyJson::as_str)
            .map(str::to_string);
        let temperature: Vec<f64> = temperature_raw
            .iter()
            .copied()
            .map(Calibration::clamp_temperature)
            .collect();
        let temperature_by_options: HashMap<String, f64> = temperature_by_options_raw
            .iter()
            .map(|(k, &v)| (k.clone(), Calibration::clamp_temperature(v)))
            .collect();

        let mut clamped_temperatures = Vec::new();
        for (bucket, &raw) in &temperature_by_options_raw {
            if Calibration::clamp_temperature(raw) != raw {
                clamped_temperatures.push(format!("{bucket}={raw}"));
            }
        }
        for (i, &raw) in temperature_raw.iter().enumerate() {
            if Calibration::clamp_temperature(raw) != raw {
                clamped_temperatures.push(format!("temperature[{i}]={raw}"));
            }
        }

        // Refused at load: Python's `build_head` meets `head_max_len` before `max_len`.
        let head_max_len = config_length(&root, "head_max_len", 192)?;
        let max_len = config_length(&root, "max_len", 512)?;

        if let Some(warning) = temperature_warning(&options, &elements) {
            emit_warnings(&[warning], at);
        }

        let deferred = Deferred {
            short_temperature: (elements.len() < 3).then_some(elements.len()),
            lang_temperatures,
        };

        Ok((
            LayaConfig {
                encoder,
                max_len,
                head_max_len,
                temperature,
                temperature_by_options,
                temperature_raw,
                temperature_by_options_raw,
                clamped_temperatures,
            },
            deferred,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::LayaConfig;

    // Python 3.12: the `TypeError` the first prediction with questions raises (`build_head` meets
    // `head_max_len` first, then `max_len`); a wrongly typed or negative length is refused at load.
    #[test]
    fn a_wrongly_typed_or_negative_length_is_refused_at_load_with_pythons_text() {
        let operand = |kind: &str| {
            format!("TypeError: unsupported operand type(s) for -: '{kind}' and 'int'")
        };
        let slice = "TypeError: slice indices must be integers or None or have an __index__ method";
        let cases: [(&str, String); 9] = [
            (r#"{"max_len": "512"}"#, operand("str")),
            (r#"{"max_len": null}"#, operand("NoneType")),
            (r#"{"head_max_len": "x"}"#, operand("str")),
            (r#"{"head_max_len": null}"#, operand("NoneType")),
            (r#"{"max_len": 512.0}"#, slice.to_string()),
            (r#"{"head_max_len": [1]}"#, operand("list")),
            // both wrong: head_max_len is met first
            (
                r#"{"max_len": "x", "head_max_len": null}"#,
                operand("NoneType"),
            ),
            (r#"{"max_len": 512.0, "head_max_len": "x"}"#, operand("str")),
            (
                r#"{"max_len": "x", "head_max_len": 192.0}"#,
                slice.to_string(),
            ),
        ];
        for (config, expected) in cases {
            match LayaConfig::parse(config) {
                Err(crate::error::LayaError::InvalidConfig(got)) => {
                    assert_eq!(expected, got, "{config}");
                }
                other => panic!("{config}: expected InvalidConfig, got {other:?}"),
            }
        }
        for (config, expected) in [
            (
                r#"{"max_len": -5}"#,
                "max_len=-5: must be a non-negative integer",
            ),
            (
                r#"{"head_max_len": -1}"#,
                "head_max_len=-1: must be a non-negative integer",
            ),
        ] {
            match LayaConfig::parse(config) {
                Err(crate::error::LayaError::InvalidConfig(got)) => assert_eq!(expected, got),
                other => panic!("{config}: expected InvalidConfig, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_short_temperature_list_is_kept_for_the_engine_and_a_longer_one_is_ignored() {
        // A `temperature` list shorter than three loads; the engine fails the question types whose
        // index it lacks (checked against `_decode_answers`: choice needs 1 entry, score 2, noul
        // 3, with or without a matching `temperature_by_options` bucket). Longer lists are fine.
        for (config, short) in [
            (r#"{"temperature": []}"#, Some(0)),
            (r#"{"temperature": [1.0]}"#, Some(1)),
            (r#"{"temperature": "ab"}"#, Some(2)),
            (r#"{"temperature": {"a": 1, "b": 2}}"#, Some(2)),
            (r#"{"temperature": [1.0, 1.0, 1.0]}"#, None),
            (r#"{"temperature": [1.0, 2.0, 3.0, 4.0]}"#, None),
            (r#"{}"#, None),
        ] {
            let (loaded, deferred) =
                LayaConfig::parse_with(config, &[], std::panic::Location::caller()).unwrap();
            assert_eq!(short, deferred.short_temperature, "{config}");
            assert_eq!(3, loaded.temperature.len().min(3), "{config}");
        }
        let longer = LayaConfig::parse(r#"{"temperature": [1.0, 2.0, 3.0, 4.0]}"#).unwrap();
        let noul = crate::questions::QuestionType::Noul;
        assert_eq!(
            3.0,
            crate::calibration::Calibration::resolve_temperature(&longer, noul, 2)
        );
    }
}
