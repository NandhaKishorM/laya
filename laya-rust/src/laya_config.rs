//! The contents of a checkpoint's `rl_agent_config.json`. Ports the config shape `laya/agent.py`'s
//! `Agent.system_one` reads via `cfg.get(...)`.
//!
//! Defaults match Python's `cfg.get(...)` fallbacks in `Agent.system_one`, not the shipped
//! multilingual values, so a config missing a key behaves the same in both runtimes.

use crate::calibration::Calibration;
use crate::error::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

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
    /// non-empty list means confidence from those buckets is uncalibrated and worth surfacing to
    /// the caller (this crate does not log; the caller decides how to report it).
    pub clamped_temperatures: Vec<String>,
}

impl LayaConfig {
    /// Read a config from a `rl_agent_config.json` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Parse a config from JSON text.
    pub fn parse(json: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(json)?;

        let mut temperature_raw: Vec<f64> = root
            .get("temperature")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect())
            .unwrap_or_default();
        // Python indexes self.temperature[qtype] for qtype 0..2, so a short list would be an
        // IndexError there; pad to 3 with the neutral value rather than failing later per-question.
        while temperature_raw.len() < 3 {
            temperature_raw.push(1.0);
        }

        let temperature_by_options_raw: HashMap<String, f64> = root
            .get("temperature_by_options")
            .and_then(Value::as_object)
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(f64::NAN)))
                    .collect()
            })
            .unwrap_or_default();

        let encoder = root
            .get("encoder")
            .and_then(Value::as_str)
            .map(str::to_string);
        let max_len = root
            .get("max_len")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(512);
        let head_max_len = root
            .get("head_max_len")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(192);

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

        Ok(LayaConfig {
            encoder,
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
            temperature_raw,
            temperature_by_options_raw,
            clamped_temperatures,
        })
    }
}
