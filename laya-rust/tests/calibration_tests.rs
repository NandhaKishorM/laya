//! Tier 1: the calibration maths, which needs no artifacts.

mod common;

use common::golden_data::CheckpointGoldenData;
use laya::calibration::Calibration;
use laya::laya_config::LayaConfig;
use laya::questions::QuestionType;

#[test]
fn temp_bucket_matches_python_boundaries() {
    let cases = [
        (QuestionType::Choice, 1, "choice:2"),
        (QuestionType::Choice, 2, "choice:2"),
        (QuestionType::Choice, 3, "choice:3-5"),
        (QuestionType::Choice, 5, "choice:3-5"),
        (QuestionType::Choice, 6, "choice:6-10"),
        (QuestionType::Choice, 10, "choice:6-10"),
        (QuestionType::Choice, 11, "choice:11+"),
        (QuestionType::Score, 4, "score:3-5"),
        (QuestionType::Noul, 2, "noul:2"),
    ];
    for (qtype, k, expected) in cases {
        assert_eq!(
            expected,
            Calibration::temp_bucket(qtype, k),
            "temp_bucket({qtype:?}, {k})"
        );
    }
}

#[test]
fn clamp_temperature_matches_python() {
    let cases = [
        (1.0, 1.0),
        (2.5, 2.5),
        (0.1006, 0.5), // Python's range, not the implementation constants being tested
        (0.0, 0.5),
        (-3.0, 0.5),
        (99.0, 5.0),
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (f64::NEG_INFINITY, 1.0),
    ];
    for (input, expected) in cases {
        assert_eq!(
            expected,
            Calibration::clamp_temperature(input),
            "clamp_temperature({input})"
        );
    }
}

#[test]
fn probabilities_read_only_the_real_markers() {
    // The padded marker column carries the -1e4 the model's masked_fill wrote. A single-option
    // question must still come out as a certainty rather than leaking the pad column in.
    let p = Calibration::probabilities(&[-0.912_778, -10000.0], 1, 1.0);
    assert_eq!(1, p.len());
    assert!((p[0] - 1.0).abs() < 1e-12);
}

#[test]
fn probabilities_sum_to_one_and_respect_temperature() {
    let logits = [2.0f32, 1.0, 0.0];
    let sharp = Calibration::probabilities(&logits, 3, 0.5);
    let soft = Calibration::probabilities(&logits, 3, 5.0);

    assert!((sharp.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((soft.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // A lower temperature concentrates mass on the top option; a higher one flattens it.
    assert!(sharp[0] > soft[0]);
    assert!(soft[2] > sharp[2]);
}

#[test]
fn probabilities_survive_large_logits() {
    // act_logits in the goldens reach +/-1600; a naive exp() would overflow to NaN.
    let p = Calibration::softmax(&[1366.2135, -1624.665]);
    assert!((p[0] - 1.0).abs() < 1e-12);
    assert!((p[1] - 0.0).abs() < 1e-12);
}

#[test]
fn confidence_is_one_for_a_single_option() {
    assert_eq!(1.0, Calibration::confidence_from_probs(&[1.0], 1));
}

#[test]
fn confidence_is_zero_for_a_uniform_distribution() {
    assert!((Calibration::confidence_from_probs(&[0.5, 0.5], 2) - 0.0).abs() < 1e-12);
    assert!((Calibration::confidence_from_probs(&[0.25, 0.25, 0.25, 0.25], 4) - 0.0).abs() < 1e-12);
}

#[test]
fn confidence_is_one_for_a_certainty() {
    assert!((Calibration::confidence_from_probs(&[1.0, 0.0], 2) - 1.0).abs() < 1e-12);
}

#[test]
fn confidence_matches_the_recorded_quickstart_values() {
    // From golden/multilingual/case_quickstart.json: urgency's four-level distribution and confidence.
    let urgency = [0.0179, 0.465, 0.3208, 0.1964];
    let confidence = Calibration::round4(Calibration::confidence_from_probs(&urgency, 4));
    assert!(
        (confidence - 0.1976).abs() < 1e-3,
        "confidence = {confidence}"
    );
}

#[test]
fn confidence_ignores_padding_beyond_the_option_count() {
    let padded = [0.5, 0.5, 0.0, 0.0];
    assert_eq!(
        Calibration::confidence_from_probs(&[0.5, 0.5], 2),
        Calibration::confidence_from_probs(&padded, 2)
    );
}

#[test]
fn answer_confidence_is_the_top_probability_not_the_entropy_gap() {
    // Ports laya/common.py's `answer_confidence` (laya 0.3.21): unlike `confidence_from_probs`
    // (normalized entropy), it is just `max(p[:k])`, clipped to [0, 1].
    assert_eq!(1.0, Calibration::answer_confidence(&[1.0], 1));
    assert_eq!(0.5, Calibration::answer_confidence(&[0.5, 0.5], 2));
    assert_eq!(0.7, Calibration::answer_confidence(&[0.7, 0.2, 0.1], 3));
}

#[test]
fn answer_confidence_ignores_padding_beyond_the_option_count() {
    let padded = [0.5, 0.5, 0.9, 0.9];
    assert_eq!(
        Calibration::answer_confidence(&[0.5, 0.5], 2),
        Calibration::answer_confidence(&padded, 2)
    );
}

#[test]
fn answer_confidence_is_one_when_option_count_is_less_than_one() {
    // Mirrors Python's `if k < 1: return 1.0` guard.
    assert_eq!(1.0, Calibration::answer_confidence(&[0.5, 0.5], 0));
}

#[test]
fn answer_confidence_matches_the_recorded_quickstart_values() {
    // Same distribution as confidence_matches_the_recorded_quickstart_values, but the top
    // probability rather than the normalized-entropy confidence.
    let urgency = [0.0179, 0.465, 0.3208, 0.1964];
    let answer_confidence = Calibration::round4(Calibration::answer_confidence(&urgency, 4));
    assert!(
        (answer_confidence - 0.465).abs() < 1e-3,
        "answer_confidence = {answer_confidence}"
    );
}

#[test]
fn expected_score_is_a_probability_weighted_mean() {
    let urgency = [0.0179, 0.465, 0.3208, 0.1964];
    let expected = Calibration::round4(Calibration::expected_score(&urgency));
    assert!(
        (expected - 1.6956).abs() < 1e-3,
        "expected_score = {expected}"
    );
    assert!((Calibration::expected_score(&[1.0, 0.0, 0.0]) - 0.0).abs() < 1e-12);
    assert!((Calibration::expected_score(&[0.0, 0.0, 1.0]) - 2.0).abs() < 1e-12);
}

#[test]
fn arg_max_breaks_ties_toward_the_lowest_index_as_numpy_does() {
    assert_eq!(0, Calibration::arg_max(&[0.5, 0.5, 0.0]));
}

#[test]
fn resolve_temperature_prefers_the_bucket_then_the_type_default() {
    let config = LayaConfig::parse(
        r#"{"max_len": 1024, "head_max_len": 256,
            "temperature": [1.5, 2.0, 2.5],
            "temperature_by_options": {"choice:3-5": 3.0, "noul:2": 0.2}}"#,
    )
    .expect("valid config");

    assert_eq!(
        3.0,
        Calibration::resolve_temperature(&config, QuestionType::Choice, 4)
    );
    assert_eq!(
        1.5,
        Calibration::resolve_temperature(&config, QuestionType::Choice, 8)
    );
    assert_eq!(
        2.0,
        Calibration::resolve_temperature(&config, QuestionType::Score, 4)
    );
    // 0.2 is below TEMP_MIN, so the bucket is clamped rather than applied as recorded.
    assert_eq!(
        Calibration::TEMP_MIN,
        Calibration::resolve_temperature(&config, QuestionType::Noul, 2)
    );
}

#[test]
fn config_falls_back_to_pythons_in_code_defaults() {
    let config = LayaConfig::parse("{}").expect("empty config parses");
    assert_eq!(512, config.max_len);
    assert_eq!(192, config.head_max_len);
    assert_eq!(vec![1.0, 1.0, 1.0], config.temperature);
}

#[test]
fn shipped_config_parses_as_recorded() {
    let data = CheckpointGoldenData::for_checkpoint("multilingual");
    if !data.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }
    let meta = data.meta();
    let config = LayaConfig::parse(&meta["config"].to_string()).expect("shipped config parses");

    assert_eq!(meta["max_len"].as_u64().unwrap() as usize, config.max_len);
    assert_eq!(
        meta["head_max_len"].as_u64().unwrap() as usize,
        config.head_max_len
    );
    let expected_temperature: Vec<f64> = meta["temperature"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(expected_temperature, config.temperature);
    assert!(config.temperature_by_options.is_empty());
}
