//! Per-language temperature overrides. Ports `resolve_lang_temperatures` from `laya/common.py`
//! and the language branch of `ONNXAgent._decode_answers`.
//!
//! A table maps a language code to the temperatures that replace the checkpoint's own for answers
//! given in that language; it is an engine construction option
//! ([`crate::LayaOptions::lang_temperatures`]) and a call selects an entry with `lang`. Python's
//! helper takes a dict of dicts of whatever the caller wrote; here the same shapes are typed, so
//! the `ValueError`s for a non-mapping entry, a non-string key or a non-mapping
//! `temperature_by_options` cannot occur. The one a typed table can still provoke (a temperature
//! list that is not three long) is Python's, word for word.

use crate::calibration::Calibration;
use crate::error::{LayaError, Result};
use crate::laya_config::{percent_g, py_repr, temperature_number};
use crate::laya_router::python_repr_str;
use crate::long_prediction::emit_warnings;
use crate::python_json_reader::PyJson;
use crate::questions::QuestionType;
use std::collections::HashMap;

/// One temperature as a caller can write it. Python's `clamp_temperature` accepts a number, reads
/// a string through `float()` and replaces a bool, `None` or anything else with the neutral 1.0,
/// so the table carries the value as written and the warning can quote it (`%r`).
#[derive(Debug, Clone, PartialEq)]
pub enum TemperatureValue {
    /// A float, including `NaN` and the infinities (replaced by 1.0).
    Float(f64),
    /// An integer.
    Int(i64),
    /// A bool: never a temperature, replaced by 1.0.
    Bool(bool),
    /// A string, read as Python's `float(str)` does.
    Text(String),
    /// Python's `None`, replaced by 1.0.
    Null,
}

impl TemperatureValue {
    fn to_py(&self) -> PyJson {
        match self {
            Self::Float(f) => PyJson::Float(*f),
            Self::Int(i) => PyJson::Int(i.to_string()),
            Self::Bool(b) => PyJson::Bool(*b),
            Self::Text(s) => PyJson::Str(s.clone()),
            Self::Null => PyJson::Null,
        }
    }
}

impl From<f64> for TemperatureValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<i64> for TemperatureValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<bool> for TemperatureValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<&str> for TemperatureValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for TemperatureValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// One language's entry in the table. `Default` is Python's `{}`: inherit everything.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct LangTemperature {
    /// The per-question-type temperatures. `None` (Python's missing or `null` `temperature`)
    /// inherits the checkpoint's own list, and then the entry is not reported by the
    /// invalid-value warning. Anything but three values is a [`LayaError::InvalidConfig`].
    pub temperature: Option<Vec<TemperatureValue>>,
    /// Per `(type, option-count)` bucket temperatures, keys as
    /// [`Calibration::temp_bucket`] builds them. A repeated key keeps its first position and its
    /// last value, as a Python dict does.
    pub temperature_by_options: Vec<(String, TemperatureValue)>,
}

impl LangTemperature {
    /// An entry that inherits everything; set what it overrides with the chained setters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`Self::temperature`].
    #[must_use]
    pub fn temperature<V: Into<TemperatureValue>>(
        mut self,
        temperature: impl IntoIterator<Item = V>,
    ) -> Self {
        self.temperature = Some(temperature.into_iter().map(Into::into).collect());
        self
    }

    /// Set [`Self::temperature_by_options`].
    #[must_use]
    pub fn temperature_by_options<K: Into<String>, V: Into<TemperatureValue>>(
        mut self,
        buckets: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.temperature_by_options = buckets
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        self
    }
}

/// The table as a caller gives it: language code to entry, in order. An entry of `None` is
/// Python's `{"de": None}` and inherits everything. Empty is Python's `None` (no overrides).
pub type LangTemperatureTable = Vec<(String, Option<LangTemperature>)>;

/// One language's temperatures after clamping.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedLangTemperature {
    /// Per question type, clamped to `[TEMP_MIN, TEMP_MAX]`.
    pub temperature: [f64; 3],
    /// Per bucket, clamped.
    pub temperature_by_options: HashMap<String, f64>,
}

/// The resolved table, keyed by `lang.split("-")[0].lower()`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LangTemperatures {
    entries: HashMap<String, ResolvedLangTemperature>,
}

/// The key a language code is looked up under: its part before the first `-`, lowercased.
pub fn normalise_lang(lang: &str) -> String {
    lang.split('-').next().unwrap_or("").to_lowercase()
}

impl LangTemperatures {
    /// Whether no language has an override (Python's falsy `lang_temperatures`).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The normalised language keys, sorted.
    pub fn languages(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.entries.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    /// The entry for a normalised language key.
    pub fn get(&self, normalised: &str) -> Option<&ResolvedLangTemperature> {
        self.entries.get(normalised)
    }

    /// The entry a call's `lang` selects: `None` for no language or an empty one (Python's
    /// `lang and ...`), else the entry for [`normalise_lang`] of it if the table has one.
    pub fn for_lang(&self, lang: Option<&str>) -> Option<&ResolvedLangTemperature> {
        match lang {
            Some(lang) if !lang.is_empty() => self.entries.get(&normalise_lang(lang)),
            _ => None,
        }
    }

    /// The temperature a call's `lang` imposes on a question of `qtype` with `option_count`
    /// options, or `None` when the language has no entry and the checkpoint's own applies. A
    /// language that has one replaces both of the checkpoint's maps: its bucket map, else its
    /// per-type temperature, never the checkpoint's bucket.
    pub fn temperature_for(
        &self,
        lang: Option<&str>,
        qtype: QuestionType,
        option_count: usize,
    ) -> Option<f64> {
        let language = self.for_lang(lang)?;
        Some(
            language
                .temperature_by_options
                .get(&Calibration::temp_bucket(qtype, option_count))
                .copied()
                .unwrap_or(language.temperature[qtype as usize]),
        )
    }
}

/// Parse a `lang_temperatures` table. Ports `common.resolve_lang_temperatures`.
///
/// `base_temperature` is the checkpoint's own `temperature` list, which an entry without a
/// `temperature` of its own inherits (it must then be three long, as Python checks). Every value
/// is clamped, and an entry the clamp had to replace is named in one warning per language, sent
/// to the `log` facade (`warn`, target `laya`) with Python's text. A later key that normalises to
/// the same language replaces an earlier one.
#[track_caller]
pub fn resolve_lang_temperatures(
    table: &[(String, Option<LangTemperature>)],
    base_temperature: &[TemperatureValue],
) -> Result<LangTemperatures> {
    let base = PyJson::Array(
        base_temperature
            .iter()
            .map(TemperatureValue::to_py)
            .collect(),
    );
    resolve_json(table, &base, std::panic::Location::caller())
}

pub(crate) fn resolve_json(
    table: &[(String, Option<LangTemperature>)],
    base: &PyJson,
    at: &std::panic::Location<'_>,
) -> Result<LangTemperatures> {
    let mut warnings = Vec::new();
    let resolved = resolve_into(table, base, &mut warnings);
    // Python warns as it goes, so the warnings of the languages before a refused one are shown.
    emit_warnings(&warnings, at);
    resolved
}

/// [`resolve_lang_temperatures`] for a checkpoint `temperature` of any JSON shape, returning the
/// text of every warning (in order, before any error) instead of sending it to the log. Hidden
/// from the docs: it exists so the probe test can compare each case with Python's own warnings,
/// which the once-per-text log filter would otherwise swallow.
#[doc(hidden)]
pub fn resolve_lang_temperatures_probe(
    table: &[(String, Option<LangTemperature>)],
    base: &serde_json::Value,
) -> (Vec<String>, Result<LangTemperatures>) {
    fn py(value: &serde_json::Value) -> PyJson {
        match value {
            serde_json::Value::Null => PyJson::Null,
            serde_json::Value::Bool(b) => PyJson::Bool(*b),
            serde_json::Value::Number(n) if n.is_f64() => PyJson::Float(n.as_f64().unwrap()),
            serde_json::Value::Number(n) => PyJson::Int(n.to_string()),
            serde_json::Value::String(s) => PyJson::Str(s.clone()),
            serde_json::Value::Array(items) => PyJson::Array(items.iter().map(py).collect()),
            serde_json::Value::Object(map) => {
                PyJson::Object(map.iter().map(|(k, v)| (k.clone(), py(v))).collect())
            }
        }
    }
    let mut warnings = Vec::new();
    let resolved = resolve_into(table, &py(base), &mut warnings);
    (warnings, resolved)
}

fn resolve_into(
    table: &[(String, Option<LangTemperature>)],
    base: &PyJson,
    warnings: &mut Vec<String>,
) -> Result<LangTemperatures> {
    // A Python dict holds a key once: the first position, the last value.
    let mut keyed: Vec<(&str, &Option<LangTemperature>)> = Vec::new();
    for (lang, entry) in table {
        match keyed.iter_mut().find(|(known, _)| known == lang) {
            Some(slot) => slot.1 = entry,
            None => keyed.push((lang, entry)),
        }
    }

    let empty = LangTemperature::default();
    let mut resolved = HashMap::new();
    for (lang, entry) in keyed {
        let norm = normalise_lang(lang);
        let cfg = entry.as_ref().unwrap_or(&empty);
        let inherited = cfg.temperature.is_none();
        let t_raw = match &cfg.temperature {
            Some(values) => PyJson::Array(values.iter().map(TemperatureValue::to_py).collect()),
            None => base.clone(),
        };
        let items = match &t_raw {
            PyJson::Array(items) if items.len() == 3 => items,
            other => {
                return Err(LayaError::InvalidConfig(format!(
                    "Language override {} temperature must be a list of 3 floats, got {}",
                    python_repr_str(lang),
                    py_repr(other)
                )));
            }
        };

        let mut tbo_raw: Vec<(&str, PyJson)> = Vec::new();
        for (key, value) in &cfg.temperature_by_options {
            let value = value.to_py();
            match tbo_raw.iter_mut().find(|(known, _)| known == key) {
                Some(slot) => slot.1 = value,
                None => tbo_raw.push((key, value)),
            }
        }

        let clamp = |raw: &PyJson| {
            Calibration::clamp_temperature(temperature_number(raw).unwrap_or(f64::NAN))
        };
        let temperature: [f64; 3] = [clamp(&items[0]), clamp(&items[1]), clamp(&items[2])];
        let by_options: HashMap<String, f64> = tbo_raw
            .iter()
            .map(|(key, raw)| ((*key).to_string(), clamp(raw)))
            .collect();

        // An entry that inherits the checkpoint's list is not reported again: the loader that got
        // here has already named what it replaced in it.
        let mut entries: Vec<(String, &PyJson, f64)> = Vec::new();
        if !inherited {
            entries.extend(
                items
                    .iter()
                    .zip(temperature)
                    .enumerate()
                    .map(|(i, (raw, applied))| (format!("temperature[{i}]"), raw, applied)),
            );
        }
        entries.extend(
            tbo_raw
                .iter()
                .map(|(key, raw)| ((*key).to_string(), raw, by_options[*key])),
        );
        let rejected: Vec<String> = entries
            .into_iter()
            // A bool or anything that is not a number is replaced, as `temperature_rejected`.
            .filter(|(_, raw, applied)| temperature_number(raw) != Some(*applied))
            .map(|(name, raw, applied)| {
                format!("{name}={} -> {}", py_repr(raw), percent_g(applied))
            })
            .collect();
        if !rejected.is_empty() {
            warnings.push(format!(
                "laya: language override {} ships invalid temperatures or values outside \
                 [{}, {}]; using {}. Treat confidence from the affected entries as \
                 uncalibrated.",
                python_repr_str(lang),
                percent_g(Calibration::TEMP_MIN),
                percent_g(Calibration::TEMP_MAX),
                rejected.join(", ")
            ));
        }
        resolved.insert(
            norm,
            ResolvedLangTemperature {
                temperature,
                temperature_by_options: by_options,
            },
        );
    }
    Ok(LangTemperatures { entries: resolved })
}
