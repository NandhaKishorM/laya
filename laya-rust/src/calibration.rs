//! Turns raw marker logits into calibrated probabilities and confidence. Ports `temp_bucket`,
//! `clamp_temperature` and `confidence_from_probs` from `laya/common.py`, plus the readout
//! arithmetic in `Agent.system_one`.
//!
//! Grouped as associated functions on a `Calibration` type rather than free functions, so the
//! call sites read as `Calibration::temp_bucket(...)` etc.

use crate::laya_config::LayaConfig;
use crate::questions::QuestionType;
use crate::sequence_builder::SequenceBuilder;

/// Namespace for calibration math. See the module docs for the source this ports.
pub struct Calibration;

impl Calibration {
    /// Lowest temperature that will be applied. A fitted temperature below 1 sharpens logits
    /// instead of softening them; the shipped `choice:11+` bucket of 0.1006 multiplies them
    /// roughly tenfold, publishing a 0.24 top probability as 0.99. A caller gating on confidence
    /// would be told a coin flip is a certainty, so such a value is refused rather than applied.
    pub const TEMP_MIN: f64 = 0.5;

    /// Highest temperature that will be applied.
    pub const TEMP_MAX: f64 = 5.0;

    /// A usable temperature: confined to [`Self::TEMP_MIN`]..[`Self::TEMP_MAX`], with anything
    /// that is not a finite number falling back to 1.0.
    pub fn clamp_temperature(t: f64) -> f64 {
        if !t.is_finite() {
            1.0
        } else {
            t.clamp(Self::TEMP_MIN, Self::TEMP_MAX)
        }
    }

    /// The calibration bucket key for a question type and option count, e.g. `"choice:3-5"`.
    pub fn temp_bucket(qtype: QuestionType, option_count: usize) -> String {
        let size = if option_count <= 2 {
            "2"
        } else if option_count <= 5 {
            "3-5"
        } else if option_count <= 10 {
            "6-10"
        } else {
            "11+"
        };
        format!("{}:{size}", SequenceBuilder::type_name(qtype))
    }

    /// The temperature to divide logits by for one question.
    pub fn resolve_temperature(
        config: &LayaConfig,
        qtype: QuestionType,
        option_count: usize,
    ) -> f64 {
        let bucket = Self::temp_bucket(qtype, option_count);
        config
            .temperature_by_options
            .get(&bucket)
            .copied()
            .unwrap_or_else(|| config.temperature[qtype as usize])
    }

    /// Tempered softmax over the first `option_count` logits of one row.
    ///
    /// Only the real markers are read. Padding columns carry the `-1e4` the model's `masked_fill`
    /// wrote, which would underflow to zero anyway, but slicing first keeps the normalisation
    /// identical to Python's `logits[r, :k]`. Returns an empty vector for `option_count == 0`
    /// rather than panicking — library code never panics on caller-controlled sizes.
    pub fn probabilities(logits: &[f32], option_count: usize, temperature: f64) -> Vec<f64> {
        let k = option_count.min(logits.len());
        if k == 0 {
            return Vec::new();
        }

        let mut p: Vec<f64> = logits[..k]
            .iter()
            .map(|&z| f64::from(z) / temperature)
            .collect();
        let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut sum = 0.0;
        for v in &mut p {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in &mut p {
            *v /= sum;
        }
        p
    }

    /// Softmax over a whole row, used for the action head. No temperature scaling.
    pub fn softmax(logits: &[f32]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f64> = logits.iter().map(|&z| f64::from(z - max).exp()).collect();
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }
        p
    }

    /// Normalized Shannon entropy confidence, `1 - H(p) / log k`, clipped to `[0, 1]`. A
    /// single-option question is fully decided by construction, so it returns 1.0.
    pub fn confidence_from_probs(p: &[f64], option_count: usize) -> f64 {
        if option_count < 2 {
            return 1.0;
        }
        let k = option_count.min(p.len());
        let mut entropy = 0.0;
        for &pi in &p[..k] {
            let clipped = pi.clamp(1e-12, 1.0);
            entropy -= pi * clipped.ln();
        }
        (1.0 - entropy / (k as f64).ln()).clamp(0.0, 1.0)
    }

    /// Probability mass on the answer being reported: `max(p[:option_count])`, clipped to
    /// `[0, 1]`. This is the quantity temperature scaling fits and every ECE figure in the Python
    /// repository is computed on, unlike [`Self::confidence_from_probs`], which reports a
    /// differently-scaled quantity with no such calibration guarantee. Ports `answer_confidence`
    /// from `laya/common.py` (laya 0.3.21). Returns `1.0` when `option_count < 1`, matching
    /// Python's `k < 1` guard.
    pub fn answer_confidence(p: &[f64], option_count: usize) -> f64 {
        if option_count < 1 {
            return 1.0;
        }
        let k = option_count.min(p.len());
        let max = p[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max.clamp(0.0, 1.0)
    }

    /// The expected level of a score answer: the probability-weighted mean of the level indices, a
    /// fractional value rather than the most likely level.
    pub fn expected_score(p: &[f64]) -> f64 {
        p.iter().enumerate().map(|(i, &pi)| i as f64 * pi).sum()
    }

    /// The index of the largest probability, ties going to the lowest index as NumPy's `argmax`
    /// does.
    pub fn arg_max(p: &[f64]) -> usize {
        let mut best = 0;
        for (i, &v) in p.iter().enumerate().skip(1) {
            if v > p[best] {
                best = i;
            }
        }
        best
    }

    /// Python's `round(x, 4)`: round-half-to-even at 4 decimal places. Both runtimes round half
    /// to even, so small binary-vs-decimal tie discrepancies (at most ~5e-5) are well within the
    /// parity tolerances used everywhere this is applied.
    pub fn round4(value: f64) -> f64 {
        if !value.is_finite() {
            return value;
        }
        (value * 10_000.0).round_ties_even() / 10_000.0
    }
}
