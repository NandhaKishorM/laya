//! Shared "does this `LayaResult` match the recorded golden `result`" comparison, factored out of
//! `predict_parity_tests.rs`'s per-case assertions so the Tier 2 routing/shortlist end-to-end
//! tests (which record a plain `result` the same way every other golden case does) can reuse it
//! instead of duplicating the tolerance logic.

use laya::LayaResult;
use laya::answers::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
use laya::python_json::PythonJson;
use laya::sequence_builder::SequenceBuilder;
use serde_json::Value;

/// Probabilities and confidences: 4-dp recorded values, so a rounding boundary is 1e-4 wide.
pub const PROB_TOLERANCE: f64 = 2e-4;
/// A score is a weighted sum over level indices, so it carries several of those errors.
pub const SCORE_TOLERANCE: f64 = 2e-3;

fn assert_close(expected: f64, actual: f64, tolerance: f64, what: &str) {
    let diff = (expected - actual).abs();
    assert!(
        diff <= tolerance,
        "{what}: python {expected}, rust {actual} (diff {diff}, tolerance {tolerance})"
    );
}

/// Assert every field of `result` (model, usage, answer ids/types/probabilities/confidence/
/// action/choice/score) matches the recorded golden `expected` (a case's `"result"` field) within
/// the standard parity tolerances. `where_case` prefixes every assertion message.
pub fn assert_result_matches(where_case: &str, expected: &Value, result: &LayaResult) {
    assert_eq!(
        expected["model"].as_str().unwrap(),
        result.model(),
        "{where_case}: model"
    );
    assert_eq!(
        expected["usage"]["input_tokens"].as_u64().unwrap() as u32,
        result.usage().input_tokens,
        "{where_case}: usage.input_tokens"
    );

    let answers = expected["answers"].as_object().unwrap();
    let expected_ids: Vec<&str> = answers.keys().map(String::as_str).collect();
    assert_eq!(
        expected_ids,
        result.ids(),
        "{where_case}: answer ids in order"
    );

    for (id, want) in answers {
        let got = result
            .get(id)
            .unwrap_or_else(|| panic!("{where_case}/{id}: missing answer"));
        let where_q = format!("{where_case}/{id}");

        assert_eq!(
            want["type"].as_str().unwrap(),
            SequenceBuilder::type_name(got.question_type()),
            "{where_q}: type"
        );
        assert_close(
            want["confidence"].as_f64().unwrap(),
            got.confidence(),
            PROB_TOLERANCE,
            &format!("{where_q}.confidence"),
        );
        assert_close(
            want["action"]["act_probability"].as_f64().unwrap(),
            got.action().act_probability,
            PROB_TOLERANCE,
            &format!("{where_q}.action.act_probability"),
        );

        match got {
            Answer::Choice(choice) => assert_choice(want, choice, &where_q),
            Answer::Score(score) => assert_score(want, score, &where_q),
            Answer::Noul(noul) => assert_noul(want, noul, &where_q),
        }
    }
}

fn assert_choice(want: &Value, got: &ChoiceAnswer, where_q: &str) {
    let probabilities = want["probabilities"].as_object().unwrap();
    let expected_labels: Vec<&str> = probabilities.keys().map(String::as_str).collect();
    let actual_labels: Vec<&str> = got.probabilities.keys().map(String::as_str).collect();
    assert_eq!(
        expected_labels, actual_labels,
        "{where_q}: probability label order"
    );
    for (label, value) in probabilities {
        assert_close(
            value.as_f64().unwrap(),
            got.probability(label).unwrap(),
            PROB_TOLERANCE,
            &format!("{where_q}.probabilities[{label}]"),
        );
    }
    assert_eq!(
        want["choice"].as_str().unwrap(),
        got.choice,
        "{where_q}: choice"
    );
}

fn assert_score(want: &Value, got: &ScoreAnswer, where_q: &str) {
    let legend = want["legend"].as_object().unwrap();
    assert_eq!(legend.len(), got.legend.len(), "{where_q}: legend length");
    for i in 0..got.legend.len() {
        let expected_level = &legend[&i.to_string()];
        assert_eq!(
            PythonJson::criterion(expected_level),
            PythonJson::criterion(&got.legend[i]),
            "{where_q}: legend[{i}]"
        );
    }
    let probabilities = want["probabilities"].as_object().unwrap();
    for i in 0..got.probabilities.len() {
        assert_close(
            probabilities[&i.to_string()].as_f64().unwrap(),
            got.probabilities[i],
            PROB_TOLERANCE,
            &format!("{where_q}.probabilities[{i}]"),
        );
    }
    assert_close(
        want["score"].as_f64().unwrap(),
        got.score,
        SCORE_TOLERANCE,
        &format!("{where_q}.score"),
    );
}

fn assert_noul(want: &Value, got: &NoulAnswer, where_q: &str) {
    assert_close(
        want["noul"].as_f64().unwrap(),
        got.probability,
        PROB_TOLERANCE,
        &format!("{where_q}.noul"),
    );
}
