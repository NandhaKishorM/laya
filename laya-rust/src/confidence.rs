//! The abstention gate: per-call `min_confidence`, the states it reports, and Jev's `confidence`.
//! Ports `laya/confidence.py` (#361, #394), function for function.
//!
//! Python hands the gate a plain `float` or a `dict` and validates it with
//! `check_min_confidence`. Here a threshold is a [`MinConfidence`], built through
//! [`MinConfidence::scalar`] or [`MinConfidence::map`], which raise Python's own `ValueError`
//! text as [`LayaError::InvalidMinConfidence`]. The variants are public, so an unvalidated value
//! can be written by hand; [`MinConfidence::validate`] therefore runs again where Python runs
//! `check_min_confidence` (the start of `predict_batch`, and the router's `predict` and
//! `predict_batch`), so a hand-built value cannot reach the gate.
//!
//! What cannot be reached from Rust is the type check (`bool`, `str`, `None` in place of a float)
//! and the `isinstance(m, dict)` check: the types already rule them out.

use crate::answers::{Answer, LayaResult};
use crate::error::{LayaError, Result};
use crate::laya_router::python_repr_str;
use crate::python_json::PythonJson;
use indexmap::IndexMap;

/// The qtype half of a bucket key, in `common.temp_bucket`'s spelling.
pub const BUCKET_QTYPES: [&str; 3] = ["choice", "score", "noul"];
/// The option-count half of a bucket key.
pub const BUCKET_SIZES: [&str; 4] = ["2", "3-5", "6-10", "11+"];
/// The one non-bucket key a map may carry: the threshold for every bucket the map does not name.
pub const DEFAULT_BUCKET_KEY: &str = "default";

/// A gate ran and this answer's confidence cleared it.
pub const GATE_PASSED: &str = "passed";
/// A gate ran and this answer's confidence fell below it.
pub const GATE_ABSTAINED: &str = "abstained";
/// A gate ran and this answer carried no usable confidence.
pub const GATE_UNEVALUATED: &str = "unevaluated";

/// The states [`Answer::abstention`] reports, and the only ones. There is deliberately no member
/// for "no gate was configured": with no `min_confidence` an answer reports nothing at all
/// ([`Answer::abstention`] is `None`).
pub const GATE_STATES: [&str; 3] = [GATE_PASSED, GATE_ABSTAINED, GATE_UNEVALUATED];

/// Every option-count bucket a `min_confidence` map can name, e.g. `"choice:3-5"`. Ports
/// `bucket_keys`. Python returns a set; this is the same twelve keys in qtype-major order.
pub fn bucket_keys() -> Vec<String> {
    BUCKET_QTYPES
        .iter()
        .flat_map(|qtype| {
            BUCKET_SIZES
                .iter()
                .map(move |size| format!("{qtype}:{size}"))
        })
        .collect()
}

/// Python's `repr(float)`, which spells the non-finite values `nan`, `inf` and `-inf` (unlike
/// the JSON spelling [`PythonJson::repr`] gives them).
fn repr_float(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        PythonJson::repr(value)
    }
}

/// Python's `repr(dict)` for a `str -> float` map: `{'choice:2': 1.5, 'default': 0.5}`.
fn repr_map(map: &IndexMap<String, f64>) -> String {
    let items: Vec<String> = map
        .iter()
        .map(|(key, value)| format!("{}: {}", python_repr_str(key), repr_float(*value)))
        .collect();
    format!("{{{}}}", items.join(", "))
}

/// Ports `_check_one_threshold`.
fn check_one_threshold(value: f64) -> Result<f64> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(LayaError::InvalidMinConfidence(format!(
            "min_confidence must be a float in [0.0, 1.0], got {}",
            repr_float(value)
        )));
    }
    Ok(value)
}

/// An opt-in abstention threshold on `answer_confidence`: one number for every answer, or a
/// per-bucket map so the threshold can differ by option count (#394).
///
/// Build one with [`Self::scalar`] or [`Self::map`], which validate as Python's
/// `check_min_confidence` does.
#[derive(Debug, Clone, PartialEq)]
pub enum MinConfidence {
    /// One threshold for every answer, in `[0.0, 1.0]`.
    Scalar(f64),
    /// A threshold per option-count bucket (every [`bucket_keys`] member) plus the optional
    /// `"default"` for any bucket the map does not name.
    Map(IndexMap<String, f64>),
}

impl MinConfidence {
    /// A validated scalar threshold. Errors with Python's text, e.g.
    /// `min_confidence must be a float in [0.0, 1.0], got 1.5`.
    pub fn scalar(threshold: f64) -> Result<Self> {
        let value = Self::Scalar(threshold);
        value.validate()?;
        Ok(value)
    }

    /// A validated per-bucket map, in the order given (which decides which entry an error names
    /// first). Keys are [`bucket_keys`] members or `"default"`; values are floats in `[0.0, 1.0]`.
    pub fn map<K: Into<String>>(thresholds: impl IntoIterator<Item = (K, f64)>) -> Result<Self> {
        let value = Self::Map(
            thresholds
                .into_iter()
                .map(|(key, threshold)| (key.into(), threshold))
                .collect(),
        );
        value.validate()?;
        Ok(value)
    }

    /// Ports `check_min_confidence` / `check_min_confidence_map`. Run again wherever Python runs
    /// it, because the variants are public.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Scalar(threshold) => check_one_threshold(*threshold).map(|_| ()),
            Self::Map(map) => {
                if map.is_empty() {
                    return Err(LayaError::InvalidMinConfidence(format!(
                        "a min_confidence map must be a non-empty dict of bucket -> float, got {}",
                        repr_map(map)
                    )));
                }
                let allowed = bucket_keys();
                for (key, threshold) in map {
                    if key != DEFAULT_BUCKET_KEY && !allowed.contains(key) {
                        return Err(LayaError::InvalidMinConfidence(format!(
                            "min_confidence map keys must be a bucket like 'choice:3-5' or \
                             'default', got {}",
                            python_repr_str(key)
                        )));
                    }
                    check_one_threshold(*threshold)?;
                }
                Ok(())
            }
        }
    }

    /// Ports `resolve_min_confidence` (a scalar passes straight through): the threshold this
    /// answer is gated at.
    fn resolve(&self, answer: &Answer) -> f64 {
        match self {
            Self::Scalar(threshold) => *threshold,
            Self::Map(map) => answer
                .option_bucket()
                .and_then(|key| map.get(&key))
                .or_else(|| map.get(DEFAULT_BUCKET_KEY))
                .copied()
                .unwrap_or(0.0),
        }
    }
}

/// The gate's verdict on one answer, kept on the answer.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Gate {
    pub(crate) low_confidence: bool,
    pub(crate) abstention: Option<&'static str>,
    pub(crate) threshold: Option<f64>,
}

/// Ports `_gate_confidence`: `answer_confidence` when usable, else `confidence`, else nothing.
fn gate_confidence(answer: &Answer) -> Option<f64> {
    [answer.answer_confidence(), answer.confidence()]
        .into_iter()
        .find(|value| value.is_finite())
}

/// Ports `flag_low_confidence` and `apply_confidence_gate` for one answer. The verdict replaces
/// whatever an earlier gate left, so a reused answer cannot keep a stale flag (#910).
fn gate_answer(answer: &mut Answer, min_confidence: &MinConfidence) {
    let threshold = min_confidence.resolve(answer);
    let confidence = gate_confidence(answer);
    let low_confidence = confidence.is_some_and(|c| c < threshold);
    let abstention = if low_confidence {
        GATE_ABSTAINED
    } else if confidence.is_none() {
        GATE_UNEVALUATED
    } else {
        GATE_PASSED
    };
    answer.set_gate(Gate {
        low_confidence,
        abstention: Some(abstention),
        threshold: Some(threshold),
    });
}

/// Ports `apply_confidence_gate`: with a threshold, reports the gate's state on every answer of
/// every result; with `None`, writes nothing.
pub(crate) fn apply_confidence_gate(
    results: &mut [LayaResult],
    min_confidence: Option<&MinConfidence>,
) {
    let Some(min_confidence) = min_confidence else {
        return;
    };
    for result in results {
        result.for_each_answer_mut(|answer| gate_answer(answer, min_confidence));
    }
}

/// Python's `sum()` over floats. Since CPython 3.12 it is not a plain left-to-right addition: it
/// carries a Neumaier compensation, so `sum([0.7, 0.2, 0.1])` is `1.0` where the naive loop gives
/// `0.9999999999999999`. This is that algorithm (`builtin_sum_impl`, float branch), so the totals
/// and spreads below round as Python 3.12+ rounds them.
fn python_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter();
    // The start value is the int 0; adding the first float gives that float.
    let mut sum = match values.next() {
        Some(first) => 0.0 + first,
        None => return 0.0,
    };
    let mut compensation = 0.0_f64;
    for value in values {
        let total = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - total) + value;
        } else {
            compensation += (value - total) + sum;
        }
        sum = total;
    }
    // Do not let the compensation turn an infinite or overflowed sum into NaN.
    if compensation != 0.0 && compensation.is_finite() {
        sum += compensation;
    }
    sum
}

/// Ports `jev_confidence` over an answer's (already rounded) probabilities: the `confidence` Jev
/// would report, or `None` for a noul answer and for a malformed document.
pub(crate) fn jev_confidence(answer: &Answer) -> Option<f64> {
    let probabilities: Vec<f64> = match answer {
        Answer::Choice(choice) => choice.probabilities.values().copied().collect(),
        Answer::Score(score) => score.probabilities.clone(),
        Answer::Noul(_) => return None,
    };
    if probabilities.is_empty() || !probabilities.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return None;
    }
    let n = probabilities.len();
    if n == 1 {
        return Some(1.0);
    }
    let count = n as f64;
    let total = python_sum(probabilities.iter().copied());
    let p: Vec<f64> = if total != 0.0 {
        probabilities.iter().map(|v| v / total).collect()
    } else {
        vec![1.0 / count; n]
    };
    // `max` and `max(range(n), key=...)` keep the first of equal values.
    let mut mode = 0;
    for (i, value) in p.iter().enumerate().skip(1) {
        if *value > p[mode] {
            mode = i;
        }
    }
    if matches!(answer, Answer::Choice(_)) {
        return Some((p[mode] - 1.0 / count) / (1.0 - 1.0 / count));
    }
    let spread = python_sum(
        p.iter()
            .enumerate()
            .map(|(i, v)| v * (i as f64 - mode as f64).abs()),
    );
    let center = (n - 1) as f64 / 2.0;
    let uniform_spread = python_sum((0..n).map(|i| (i as f64 - center).abs())) / count;
    // Python's `max(0.0, x)`: zero unless `x` is greater.
    let jev = 1.0 - spread / uniform_spread;
    Some(if jev > 0.0 { jev } else { 0.0 })
}

/// Ports `_option_bucket`: the `qtype:size` key an answer is gated under.
pub(crate) fn option_bucket(answer: &Answer) -> Option<String> {
    let (qtype, options) = match answer {
        Answer::Choice(choice) => ("choice", choice.probabilities.len()),
        Answer::Score(score) => ("score", score.probabilities.len()),
        // A noul answer carries no probabilities dict, so Python reads it as two options.
        Answer::Noul(_) => ("noul", 2),
    };
    if options == 0 {
        return None;
    }
    let size = match options {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    Some(format!("{qtype}:{size}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answers::{ActionInfo, ChoiceAnswer, NoulAnswer, ScoreAnswer};
    use std::collections::HashMap;

    fn choice(probabilities: &[f64], confidence: f64, answer_confidence: f64) -> Answer {
        let map: IndexMap<String, f64> = probabilities
            .iter()
            .enumerate()
            .map(|(i, p)| (format!("o{i}"), *p))
            .collect();
        Answer::Choice(ChoiceAnswer::new(
            "o0",
            map,
            confidence,
            answer_confidence,
            ActionInfo::default(),
        ))
    }

    fn score(probabilities: &[f64]) -> Answer {
        Answer::Score(ScoreAnswer::new(
            0.0,
            Vec::new(),
            probabilities.to_vec(),
            0.5,
            0.5,
            ActionInfo::default(),
        ))
    }

    fn noul(confidence: f64, answer_confidence: f64) -> Answer {
        Answer::Noul(NoulAnswer::new(
            0.7,
            confidence,
            answer_confidence,
            ActionInfo::default(),
        ))
    }

    fn result(answers: Vec<(&str, Answer)>) -> LayaResult {
        let order = answers.iter().map(|(id, _)| id.to_string()).collect();
        let by_id: HashMap<String, Answer> = answers
            .into_iter()
            .map(|(id, a)| (id.to_string(), a))
            .collect();
        LayaResult::new("m", order, by_id, crate::answers::Usage::default())
    }

    fn gated(answer: Answer, min_confidence: &MinConfidence) -> Answer {
        let mut results = [result(vec![("q", answer)])];
        apply_confidence_gate(&mut results, Some(min_confidence));
        results[0]["q"].clone()
    }

    #[test]
    fn an_ungated_answer_reports_nothing() {
        let answer = choice(&[0.6, 0.4], 0.3, 0.6);
        assert!(!answer.low_confidence());
        assert_eq!(answer.abstention(), None);
        assert_eq!(answer.abstention_threshold(), None);
        let mut results = [result(vec![("q", answer)])];
        apply_confidence_gate(&mut results, None);
        assert_eq!(results[0]["q"].abstention(), None);
    }

    #[test]
    fn scalar_gate_reports_passed_abstained_and_the_threshold() {
        let threshold = MinConfidence::scalar(0.7).unwrap();
        let low = gated(choice(&[0.6, 0.4], 0.9, 0.6), &threshold);
        assert!(low.low_confidence());
        assert_eq!(low.abstention(), Some("abstained"));
        assert_eq!(low.abstention_threshold(), Some(0.7));
        let high = gated(choice(&[0.8, 0.2], 0.1, 0.8), &threshold);
        assert!(!high.low_confidence());
        assert_eq!(high.abstention(), Some("passed"));
        // Equal to the threshold clears it: the test is strictly below.
        let equal = gated(choice(&[0.7, 0.3], 0.1, 0.7), &threshold);
        assert_eq!(equal.abstention(), Some("passed"));
    }

    #[test]
    fn the_gate_reads_answer_confidence_first_then_confidence() {
        let threshold = MinConfidence::scalar(0.5).unwrap();
        // answer_confidence passes although confidence would not.
        let a = gated(choice(&[0.6, 0.4], 0.1, 0.6), &threshold);
        assert_eq!(a.abstention(), Some("passed"));
        // A NaN answer_confidence falls back to confidence (confidence.py:86).
        let b = gated(choice(&[0.6, 0.4], 0.1, f64::NAN), &threshold);
        assert_eq!(b.abstention(), Some("abstained"));
        let c = gated(choice(&[0.6, 0.4], 0.9, f64::INFINITY), &threshold);
        assert_eq!(c.abstention(), Some("passed"));
    }

    #[test]
    fn an_answer_with_no_usable_confidence_is_unevaluated_not_flagged() {
        let threshold = MinConfidence::scalar(0.9).unwrap();
        let answer = gated(noul(f64::NAN, f64::NAN), &threshold);
        assert!(!answer.low_confidence());
        assert_eq!(answer.abstention(), Some("unevaluated"));
        assert_eq!(answer.abstention_threshold(), Some(0.9));
    }

    #[test]
    fn zero_is_a_gate_that_passes_everything_usable_and_echoes_zero() {
        let threshold = MinConfidence::scalar(0.0).unwrap();
        let answer = gated(choice(&[0.5, 0.5], 0.0, 0.0), &threshold);
        assert_eq!(answer.abstention(), Some("passed"));
        assert_eq!(answer.abstention_threshold(), Some(0.0));
        assert!(!answer.low_confidence());
        let unusable = gated(noul(f64::NAN, f64::NAN), &threshold);
        assert_eq!(unusable.abstention(), Some("unevaluated"));
    }

    #[test]
    fn a_map_gates_each_answer_at_its_own_bucket_then_default_then_zero() {
        let map =
            MinConfidence::map([("choice:2", 0.9), ("score:3-5", 0.8), ("default", 0.55)]).unwrap();
        let two = gated(choice(&[0.6, 0.4], 0.5, 0.6), &map);
        assert_eq!(two.abstention_threshold(), Some(0.9));
        assert_eq!(two.abstention(), Some("abstained"));
        let three_scores = gated(score(&[0.2, 0.7, 0.1]), &map);
        assert_eq!(three_scores.abstention_threshold(), Some(0.8));
        // noul:2 is not named, so it falls to the default.
        let unnamed = gated(noul(0.7, 0.7), &map);
        assert_eq!(unnamed.abstention_threshold(), Some(0.55));
        assert_eq!(unnamed.abstention(), Some("passed"));
        // No default either: gate nothing.
        let no_default = MinConfidence::map([("choice:2", 0.9)]).unwrap();
        let ungated = gated(noul(0.1, 0.1), &no_default);
        assert_eq!(ungated.abstention_threshold(), Some(0.0));
        assert_eq!(ungated.abstention(), Some("passed"));
    }

    #[test]
    fn option_buckets_follow_the_size_bands() {
        let sizes = [
            (1, "2"),
            (2, "2"),
            (3, "3-5"),
            (5, "3-5"),
            (6, "6-10"),
            (10, "6-10"),
            (11, "11+"),
        ];
        for (count, size) in sizes {
            let p = vec![1.0 / count as f64; count];
            assert_eq!(
                choice(&p, 0.5, 0.5).option_bucket(),
                Some(format!("choice:{size}"))
            );
            assert_eq!(score(&p).option_bucket(), Some(format!("score:{size}")));
        }
        assert_eq!(noul(0.5, 0.5).option_bucket().as_deref(), Some("noul:2"));
        assert_eq!(choice(&[], 0.5, 0.5).option_bucket(), None);
    }

    /// #910: a result gated at a strict threshold and then at a lenient one must not keep the
    /// flag. Results are values here, so the verdict is rewritten on every gate.
    #[test]
    fn regating_a_reused_result_clears_a_stale_flag() {
        let mut results = [result(vec![("q", choice(&[0.6, 0.4], 0.5, 0.6))])];
        apply_confidence_gate(&mut results, Some(&MinConfidence::scalar(0.9).unwrap()));
        assert!(results[0]["q"].low_confidence());
        apply_confidence_gate(&mut results, Some(&MinConfidence::scalar(0.0).unwrap()));
        assert!(!results[0]["q"].low_confidence());
        assert_eq!(results[0]["q"].abstention(), Some("passed"));
        assert_eq!(results[0]["q"].abstention_threshold(), Some(0.0));
        // A later call with no threshold writes nothing, so the last verdict is untouched.
        apply_confidence_gate(&mut results, None);
        assert_eq!(results[0]["q"].abstention(), Some("passed"));
    }

    fn approx(a: Option<f64>, b: f64) {
        let a = a.expect("a jev confidence");
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn jev_confidence_for_choice_and_score() {
        approx(choice(&[0.5, 0.5], 0.0, 0.5).jev_confidence(), 0.0);
        approx(choice(&[1.0, 0.0], 0.0, 1.0).jev_confidence(), 1.0);
        approx(
            choice(&[0.6, 0.3, 0.1], 0.0, 0.6).jev_confidence(),
            (0.6 - 1.0 / 3.0) / (2.0 / 3.0),
        );
        // A single option is 1.0, and a zero total reads as uniform.
        approx(choice(&[0.0], 0.0, 0.0).jev_confidence(), 1.0);
        approx(choice(&[0.0, 0.0], 0.0, 0.0).jev_confidence(), 0.0);
        // Unnormalised input is normalised by its sum.
        approx(choice(&[3.0, 1.0], 0.0, 0.0).jev_confidence(), 0.5);
        // Score: a point mass has zero spread; a uniform distribution is 0.
        approx(score(&[0.0, 1.0, 0.0]).jev_confidence(), 1.0);
        approx(score(&[0.25, 0.25, 0.25, 0.25]).jev_confidence(), 0.0);
    }

    #[test]
    fn jev_confidence_is_none_for_noul_and_malformed_probabilities() {
        assert_eq!(noul(0.5, 0.5).jev_confidence(), None);
        assert_eq!(choice(&[], 0.5, 0.5).jev_confidence(), None);
        assert_eq!(choice(&[0.5, f64::NAN], 0.5, 0.5).jev_confidence(), None);
        assert_eq!(
            choice(&[0.5, f64::INFINITY], 0.5, 0.5).jev_confidence(),
            None
        );
        assert_eq!(choice(&[0.5, -0.1], 0.5, 0.5).jev_confidence(), None);
        assert_eq!(score(&[0.5, -0.1]).jev_confidence(), None);
        assert_eq!(score(&[]).jev_confidence(), None);
    }

    #[test]
    fn scalar_validation_uses_pythons_text_and_repr() {
        let text = |v: f64| MinConfidence::scalar(v).unwrap_err().to_string();
        let head = "min_confidence must be a float in [0.0, 1.0], got ";
        assert_eq!(text(1.5), format!("{head}1.5"));
        assert_eq!(text(2.0), format!("{head}2.0"));
        assert_eq!(text(-0.1), format!("{head}-0.1"));
        assert_eq!(text(f64::NAN), format!("{head}nan"));
        assert_eq!(text(f64::INFINITY), format!("{head}inf"));
        assert_eq!(text(f64::NEG_INFINITY), format!("{head}-inf"));
        assert_eq!(text(1e16), format!("{head}1e+16"));
        assert!(MinConfidence::scalar(0.0).is_ok());
        assert!(MinConfidence::scalar(1.0).is_ok());
    }

    #[test]
    fn map_validation_uses_pythons_text_in_python_order() {
        let err = |m: Vec<(&str, f64)>| MinConfidence::map(m).unwrap_err().to_string();
        let keys = "min_confidence map keys must be a bucket like 'choice:3-5' or 'default', got ";
        let value = "min_confidence must be a float in [0.0, 1.0], got 1.5";
        assert_eq!(
            err(vec![]),
            "a min_confidence map must be a non-empty dict of bucket -> float, got {}"
        );
        assert_eq!(err(vec![("choice:1", 0.5)]), format!("{keys}'choice:1'"));
        assert_eq!(err(vec![("it's", 0.5)]), format!("{keys}\"it's\""));
        assert_eq!(err(vec![("choice:2", 1.5)]), value);
        // The first bad entry in insertion order decides which error is raised.
        assert_eq!(err(vec![("choice:2", 1.5), ("nope", 0.5)]), value);
        assert_eq!(
            err(vec![("nope", 0.5), ("choice:2", 1.5)]),
            format!("{keys}'nope'")
        );
        assert!(MinConfidence::map([("default", 0.5), ("noul:11+", 0.0)]).is_ok());
    }

    #[test]
    fn hand_built_variants_are_validated_again() {
        assert!(MinConfidence::Scalar(2.0).validate().is_err());
        assert!(MinConfidence::Map(IndexMap::new()).validate().is_err());
    }

    /// CPython 3.12 compensates `sum()`: the naive loop would give 0.9999999999999999.
    #[test]
    fn python_sum_is_compensated_like_cpython_312() {
        assert_eq!(python_sum([0.7, 0.2, 0.1]), 1.0);
        assert_eq!(python_sum([0.1; 10]), 1.0);
        assert_eq!(python_sum([]), 0.0);
        assert_eq!(python_sum([f64::INFINITY, 1.0]), f64::INFINITY);
    }

    #[test]
    fn bucket_keys_are_the_twelve_buckets() {
        let keys = bucket_keys();
        assert_eq!(keys.len(), 12);
        assert_eq!(keys[0], "choice:2");
        assert_eq!(keys[11], "noul:11+");
        assert_eq!(GATE_STATES, ["passed", "abstained", "unevaluated"]);
    }
}
