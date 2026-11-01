//! Calibration files and histogram binning. Ports `ONNXAgent.load_calibration`,
//! `calibrate.apply_calibration_payload` and `calibrate.apply_binning_map`.
//!
//! A calibration file is the JSON map Python's `Agent.save_calibration` writes: per-type
//! `temperature`, per-bucket `temperature_by_options` and an optional `binning_map`. It is an engine
//! construction option ([`crate::LayaOptions::calibration`]), read after the checkpoint's own
//! temperatures and the `lang_temperatures` table, so its temperatures replace the checkpoint's
//! (both maps, wholesale) and its binning map recalibrates `answer_confidence`. This crate only
//! reads and applies the file; fitting and saving one is Python's job.
//!
//! The file is read as Python does (`json.load`: `NaN`/`Infinity` accepted, a duplicate key keeps
//! its last value) with the crate's Python-json reader, as strict UTF-8 like the checkpoint's config.
//! A malformed payload is refused with [`LayaError::InvalidConfig`] carrying Python's `ValueError`
//! text word for word; an error Python raises as another type is prefixed with that type's name
//! (`OverflowError: ...`). A temperature that is an integer too wide for a double is not an error:
//! it is replaced by 1.0 and named in the clamp warning, as Python 0.4.20 does.
//!
//! ## Not ported: the checkpoint-identity warning
//!
//! Python warns when a version 2 file records a `model_id_or_path`, `subfolder` or `config` that
//! differs from the agent's. An ONNX agent's `model_id_or_path` is the string its caller passed to
//! name the PyTorch checkpoint (a Hub id or a path); this crate loads an exported artifact
//! directory and has no such string, so there is nothing faithful to compare. A version 2 file
//! therefore loads exactly as Python loads it, with the same temperatures, binning, errors and
//! clamp warning, but never emits that warning.

use crate::calibration::Calibration;
use crate::error::{LayaError, Result};
use crate::language_detection::py_strip;
use crate::laya_config::{
    LayaConfig, decimal_digit_value, percent_g, py_float, py_repr, python_text_mode,
    python_type_name, temperature_number,
};
use crate::laya_router::python_repr_str;
use crate::long_prediction::emit_warnings;
use crate::python_json_reader::PyJson;
use std::collections::HashMap;
use std::path::Path;

/// Python's `ValueError` from `apply_calibration_payload`.
fn value_error(text: String) -> LayaError {
    LayaError::InvalidConfig(text)
}

/// One bucket's histogram: `values.len()` equal-width bins over `[0, 1]`, each holding the
/// recalibrated confidence for a raw confidence that falls in it.
#[derive(Debug, Clone, PartialEq)]
pub struct BinningEntry {
    values: Vec<f64>,
}

impl BinningEntry {
    /// The number of bins (Python's `bins`).
    pub fn bins(&self) -> usize {
        self.values.len()
    }

    /// The recalibrated confidence of each bin.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// A fitted histogram-binning map, keyed by calibration bucket ([`Calibration::temp_bucket`]).
/// Ports the map `apply_binning_map` reads; a bucket without an entry passes through unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BinningMap {
    entries: Vec<(String, BinningEntry)>,
}

impl BinningMap {
    /// Whether no bucket has an entry (Python's falsy `{}`, which skips binning altogether).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry for `bucket`, if the map was fitted for it.
    pub fn get(&self, bucket: &str) -> Option<&BinningEntry> {
        self.entries
            .iter()
            .find(|(name, _)| name == bucket)
            .map(|(_, entry)| entry)
    }

    /// The buckets with an entry, in file order.
    pub fn buckets(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(name, _)| name.as_str())
    }

    /// Recalibrate one `answer_confidence` for its option-count `bucket`. Ports
    /// `calibrate.apply_binning_map`: the confidence is returned unchanged when the map has no
    /// entry for the bucket, else `values[min(bins - 1, max(0, int(confidence * bins)))]`.
    ///
    /// A NaN or infinite confidence that reaches an entry is Python's `int()` error (the
    /// `ValueError` "cannot convert float NaN to integer", the `OverflowError` "cannot convert
    /// float infinity to integer"), returned as [`LayaError::InvalidConfig`] spelled as the loader
    /// spells them: the `ValueError` bare, the `OverflowError` prefixed with its name.
    pub fn apply(&self, confidence: f64, bucket: &str) -> Result<f64> {
        let Some(entry) = self.get(bucket) else {
            return Ok(confidence);
        };
        let bins = entry.values.len();
        let scaled = confidence * bins as f64;
        if scaled.is_nan() {
            return Err(value_error(
                "cannot convert float NaN to integer".to_string(),
            ));
        }
        if scaled.is_infinite() {
            return Err(value_error(
                "OverflowError: cannot convert float infinity to integer".to_string(),
            ));
        }
        // `int()` truncates toward zero; a negative product lands in bin 0, a large one in the last.
        let index = if scaled <= 0.0 {
            0
        } else {
            (scaled.trunc() as usize).min(bins - 1)
        };
        Ok(entry.values[index])
    }
}

/// What a calibration file installs on an engine. Ports the state `apply_calibration_payload`
/// writes: the temperatures as shipped and as clamped, and the binning map.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationFile {
    /// The per-type temperatures as shipped (`NaN` for an entry that is not a number).
    pub temperature_raw: [f64; 3],
    /// The per-type temperatures after `clamp_temperature`: what inference reads.
    pub temperature: [f64; 3],
    /// The per-bucket temperatures as shipped (`NaN` for an entry that is not a number), in file order.
    pub temperature_by_options_raw: Vec<(String, f64)>,
    /// The per-bucket temperatures after `clamp_temperature`, in file order.
    pub temperature_by_options: Vec<(String, f64)>,
    /// The binning map; `None` when the file has none (missing or `null`), which clears any map.
    pub binning_map: Option<BinningMap>,
}

/// `calibration JSON version must be an integer`: a missing or `null` version is 1; an integer, a
/// whole float or a string `int()` reads is accepted (its value only matters to the identity
/// check, which this crate does not port); anything else is refused.
fn check_version(version: Option<&PyJson>) -> Result<()> {
    let version = match version {
        None | Some(PyJson::Null) => return Ok(()),
        Some(version) => version,
    };
    let accepted = match version {
        PyJson::Int(_) => true,
        PyJson::Float(f) => f.is_finite() && f.fract() == 0.0,
        PyJson::Str(text) => python_int_accepts(text),
        _ => false,
    };
    if accepted {
        Ok(())
    } else {
        Err(value_error(format!(
            "calibration JSON version must be an integer, got {}",
            py_repr(version)
        )))
    }
}

/// Whether Python's `int(text)` reads `text`: surrounding whitespace, an optional sign, then decimal
/// digits of any script with a single `_` allowed between two digits, at most 4300 of them.
fn python_int_accepts(text: &str) -> bool {
    let text = py_strip(text);
    let body = text
        .strip_prefix(['+', '-'])
        .unwrap_or(text)
        .chars()
        .collect::<Vec<_>>();
    let mut digits = 0usize;
    let mut previous_digit = false;
    for (i, &ch) in body.iter().enumerate() {
        if decimal_digit_value(ch).is_some() {
            digits += 1;
            previous_digit = true;
        } else if ch == '_'
            && previous_digit
            && body
                .get(i + 1)
                .is_some_and(|c| decimal_digit_value(*c).is_some())
        {
            previous_digit = false;
        } else {
            return false;
        }
    }
    digits > 0 && digits <= 4300
}

/// The `binning_map` shape checks of `apply_calibration_payload`, in Python's order.
fn parse_binning_map(binning: &PyJson) -> Result<BinningMap> {
    let PyJson::Object(entries) = binning else {
        return Err(value_error(format!(
            "calibration JSON binning_map must be an object of bucket -> {{bins, values}}, got {}",
            python_type_name(binning)
        )));
    };
    let mut map = BinningMap::default();
    for (name, entry) in entries {
        let name_repr = python_repr_str(name);
        if !matches!(entry, PyJson::Object(_)) {
            return Err(value_error(format!(
                "calibration JSON binning_map[{name_repr}] must be an object with \"bins\" and \
                 \"values\", got {}",
                python_type_name(entry)
            )));
        }
        let bins = entry.get("bins");
        let values = entry.get("values");
        let bins_literal = match bins {
            Some(PyJson::Int(literal))
                if !literal.starts_with('-') && literal.bytes().any(|b| b != b'0') =>
            {
                literal.as_str()
            }
            other => {
                return Err(value_error(format!(
                    "calibration JSON binning_map[{name_repr}] must have an integer \"bins\" >= 1, \
                     got {}",
                    other.map_or_else(|| "None".to_string(), py_repr)
                )));
            }
        };
        let values = match values {
            Some(PyJson::Array(values)) if values.len().to_string() == bins_literal => values,
            _ => {
                return Err(value_error(format!(
                    "calibration JSON binning_map[{name_repr}] must have \"values\" of length \
                     \"bins\" ({bins_literal})"
                )));
            }
        };
        let not_numbers = || {
            value_error(format!(
                "calibration JSON binning_map[{name_repr}] values must be numbers, got {}",
                py_repr(&PyJson::Array(values.clone()))
            ))
        };
        // `[float(v) for v in values]` catches `(TypeError, ValueError, OverflowError)` since
        // Python 0.4.21: a huge int, a `None`, list or object or a string `float()` rejects is
        // the ValueError above.
        let mut parsed = Vec::with_capacity(values.len());
        for value in values {
            parsed.push(match value {
                PyJson::Int(literal) => PyJson::int_as_f64(literal).ok_or_else(not_numbers)?,
                PyJson::Float(f) => *f,
                PyJson::Bool(b) => f64::from(u8::from(*b)),
                PyJson::Str(text) => py_float(text).ok_or_else(not_numbers)?,
                _ => return Err(not_numbers()),
            });
        }
        // Then the strict pass: a bool or a string parsed above is still not a number.
        if values
            .iter()
            .any(|v| matches!(v, PyJson::Bool(_) | PyJson::Str(_)))
        {
            return Err(not_numbers());
        }
        for &v in &parsed {
            if !v.is_finite() {
                return Err(value_error(format!(
                    "calibration JSON binning_map[{name_repr}] values must be finite, got {}",
                    py_repr(&PyJson::Array(values.clone()))
                )));
            }
            if !(0.0..=1.0).contains(&v) {
                return Err(value_error(format!(
                    "calibration JSON binning_map[{name_repr}] values must be in [0, 1], got {}",
                    py_repr(&PyJson::Array(values.clone()))
                )));
            }
        }
        map.entries
            .push((name.clone(), BinningEntry { values: parsed }));
    }
    Ok(map)
}

/// The `(name, raw value)` of each entry `clamp_temperature` had to replace, as the warning names it.
fn rejected_temperatures(entries: &[(String, &PyJson)]) -> Vec<String> {
    entries
        .iter()
        .filter_map(|(name, raw)| {
            let number = temperature_number(raw);
            let applied = Calibration::clamp_temperature(number.unwrap_or(f64::NAN));
            (number != Some(applied))
                .then(|| format!("{name}={} -> {}", py_repr(raw), percent_g(applied)))
        })
        .collect()
}

/// The place Python attributes the clamp warning to (`stacklevel=3`: `load_calibration` in
/// `onnx_agent.py`). It is the same line for every load, so a text is shown once per process.
fn warning_site() -> &'static std::panic::Location<'static> {
    std::panic::Location::caller()
}

impl CalibrationFile {
    /// Read a calibration file: `ONNXAgent.load_calibration`'s `open` + `json.load`, then
    /// [`Self::parse`]. The file is read as strict UTF-8, with `\r\n` and `\r` turned into `\n`
    /// first, as the checkpoint's config is.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&python_text_mode(std::fs::read_to_string(path)?))
    }

    /// Parse and validate a calibration payload (`json.loads` then `apply_calibration_payload`).
    /// The warning Python raises when a temperature had to be clamped goes to the `log` facade
    /// (`warn`, target `laya`), once per text.
    pub fn parse(json: &str) -> Result<Self> {
        let (warnings, file) = Self::parse_probe(json);
        emit_warnings(&warnings, warning_site());
        file
    }

    /// [`Self::parse`] returning the text of the warnings instead of sending them to the log.
    /// Hidden from the docs: it exists so the probe test can compare each case with Python's own
    /// warnings, which the once-per-text log filter would otherwise swallow.
    #[doc(hidden)]
    pub fn parse_probe(json: &str) -> (Vec<String>, Result<Self>) {
        let mut warnings = Vec::new();
        let file =
            PyJson::parse(json).and_then(|payload| Self::from_payload(&payload, &mut warnings));
        (warnings, file)
    }

    /// `apply_calibration_payload`, minus the identity check this crate does not port.
    fn from_payload(payload: &PyJson, warnings: &mut Vec<String>) -> Result<Self> {
        if !matches!(payload, PyJson::Object(_)) {
            return Err(value_error(format!(
                "calibration JSON must be an object, got {}",
                python_type_name(payload)
            )));
        }
        check_version(payload.get("version"))?;
        let temps = match payload.get("temperature") {
            Some(PyJson::Array(temps)) if temps.len() == 3 => temps,
            _ => {
                return Err(value_error(
                    "calibration JSON must contain temperature: [3 floats]".to_string(),
                ));
            }
        };
        let by_options: &[(String, PyJson)] = match payload.get("temperature_by_options") {
            None | Some(PyJson::Null) => &[],
            Some(PyJson::Object(entries)) => entries,
            Some(other) => {
                return Err(value_error(format!(
                    "calibration JSON temperature_by_options must be an object of bucket -> \
                     float, got {}",
                    python_type_name(other)
                )));
            }
        };
        let binning_map = match payload.get("binning_map") {
            None | Some(PyJson::Null) => None,
            Some(binning) => Some(parse_binning_map(binning)?),
        };

        // `_install_temperatures`: the list is clamped first, then the dict. An integer too wide
        // for a double is no number to `clamp_temperature` and `temperature_rejected` (they catch
        // the `OverflowError` since Python 0.4.20): it becomes the neutral 1.0 and the warning
        // names it digit for digit. A bin value refuses it too (`parse_binning_map`).
        let raw_of = |v: &PyJson| temperature_number(v).unwrap_or(f64::NAN);
        let mut temperature_raw = [0.0; 3];
        let mut temperature = [0.0; 3];
        for (i, value) in temps.iter().enumerate() {
            temperature_raw[i] = raw_of(value);
            temperature[i] = Calibration::clamp_temperature(temperature_raw[i]);
        }
        let temperature_by_options_raw: Vec<(String, f64)> = by_options
            .iter()
            .map(|(k, v)| (k.clone(), raw_of(v)))
            .collect();
        let temperature_by_options: Vec<(String, f64)> = temperature_by_options_raw
            .iter()
            .map(|(k, v)| (k.clone(), Calibration::clamp_temperature(*v)))
            .collect();

        let mut entries: Vec<(String, &PyJson)> = temps
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("temperature[{i}]"), v))
            .collect();
        entries.extend(by_options.iter().map(|(k, v)| (k.clone(), v)));
        let rejected = rejected_temperatures(&entries);
        if !rejected.is_empty() {
            warnings.push(format!(
                "laya: calibration temperatures are invalid or outside [{}, {}]; using {}. Treat \
                 confidence from the affected entries as uncalibrated.",
                percent_g(Calibration::TEMP_MIN),
                percent_g(Calibration::TEMP_MAX),
                rejected.join(", ")
            ));
        }

        Ok(CalibrationFile {
            temperature_raw,
            temperature,
            temperature_by_options_raw,
            temperature_by_options,
            binning_map,
        })
    }

    /// Install the file's temperatures on `config`, replacing both of the checkpoint's maps
    /// (Python's `_install_temperatures`). The binning map is the engine's to keep
    /// ([`crate::LayaEngine::binning_map`]).
    pub fn install_temperatures(&self, config: &mut LayaConfig) {
        config.temperature_raw = self.temperature_raw.to_vec();
        config.temperature = self.temperature.to_vec();
        config.temperature_by_options_raw = self
            .temperature_by_options_raw
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
        config.temperature_by_options = self
            .temperature_by_options
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, &[f64])]) -> BinningMap {
        BinningMap {
            entries: entries
                .iter()
                .map(|(name, values)| {
                    (
                        (*name).to_string(),
                        BinningEntry {
                            values: values.to_vec(),
                        },
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn a_confidence_lands_in_its_bin_and_the_edges_stay_inside_the_map() {
        let m = map(&[("choice:3-5", &[0.1, 0.4, 0.6, 0.9])]);
        for (confidence, expected) in [
            (0.0, 0.1),
            (0.2499999999999999, 0.1),
            (0.25, 0.4),
            (0.5, 0.6),
            (0.75, 0.9),
            (1.0, 0.9),
            (-0.1, 0.1),
            (7.0, 0.9),
            (1e300, 0.9),
        ] {
            assert_eq!(
                expected,
                m.apply(confidence, "choice:3-5").unwrap(),
                "{confidence}"
            );
        }
    }

    #[test]
    fn a_bucket_the_map_lacks_passes_the_confidence_through_even_when_it_is_not_finite() {
        let m = map(&[("choice:3-5", &[0.5])]);
        assert_eq!(1.5, m.apply(1.5, "score:2").unwrap());
        assert!(m.apply(f64::NAN, "score:2").unwrap().is_nan());
        assert_eq!(f64::INFINITY, m.apply(f64::INFINITY, "score:2").unwrap());
    }

    #[test]
    fn a_non_finite_confidence_in_a_binned_bucket_is_pythons_int_error() {
        let m = map(&[("choice:3-5", &[0.5])]);
        let text = |c: f64| m.apply(c, "choice:3-5").unwrap_err().to_string();
        assert_eq!("cannot convert float NaN to integer", text(f64::NAN));
        let overflow = "OverflowError: cannot convert float infinity to integer";
        assert_eq!(overflow, text(f64::INFINITY));
        assert_eq!(overflow, text(f64::NEG_INFINITY));
    }

    #[test]
    fn python_int_reads_strings_the_way_the_version_check_needs() {
        for ok in ["2", " 2 ", "+2", "-2", "0_2", "\u{ff12}", "00"] {
            assert!(python_int_accepts(ok), "{ok:?}");
        }
        for bad in ["", "abc", "1.5", "1__2", "_1", "1_", "+", " ", "it's"] {
            assert!(!python_int_accepts(bad), "{bad:?}");
        }
    }
}
