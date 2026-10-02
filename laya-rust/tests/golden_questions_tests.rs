//! Verifies that rebuilding a golden case's `questions` into a [`QuestionSet`] and rendering it
//! with [`SequenceBuilder`] reproduces exactly what the recorded `per_question` entries show —
//! i.e. that [`Question`] construction, [`SequenceBuilder::render_options`] and
//! [`SequenceBuilder::render_instructions`] agree with the real Python `_to_internal` /
//! `render_options` pipeline, not just with each other. Also checks `serialized_state` against
//! [`PythonJson::state`]. This is a Rust-specific addition.

mod common;

use common::golden_data::{self, CheckpointGoldenData};
use laya::python_json::PythonJson;
use laya::sequence_builder::SequenceBuilder;
use serde_json::Value;

const CHECKPOINTS: [&str; 3] = ["multilingual", "english", "typed-decisions"];

#[test]
fn golden_questions_render_exactly_as_recorded() {
    let mut checked_any = false;

    for checkpoint in CHECKPOINTS {
        let data = CheckpointGoldenData::for_checkpoint(checkpoint);
        if !data.available() {
            eprintln!("skip: no golden data for '{checkpoint}'");
            continue;
        }

        for case in data.success_cases() {
            checked_any = true;
            let root = data.load_case(&case);

            let question_set = golden_data::build_questions(&root["questions"])
                .unwrap_or_else(|e| panic!("{checkpoint}/{case}: failed to build questions: {e}"));

            for per_question in root["per_question"].as_array().unwrap() {
                let id = per_question["id"].as_str().unwrap();
                let question = question_set
                    .get(id)
                    .unwrap_or_else(|| panic!("{checkpoint}/{case}: missing question '{id}'"));

                let expected_options: Vec<&str> = per_question["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                let actual_options = SequenceBuilder::render_options(question);
                assert_eq!(
                    expected_options, actual_options,
                    "{checkpoint}/{case}: question '{id}' options"
                );

                let expected_instructions = per_question["instructions"].as_str().unwrap();
                let actual_instructions = SequenceBuilder::render_instructions(question);
                assert_eq!(
                    expected_instructions, actual_instructions,
                    "{checkpoint}/{case}: question '{id}' instructions"
                );
            }

            if let Some(expected_state) = root.get("serialized_state").and_then(Value::as_str) {
                let actual_state = PythonJson::state(&root["state"]);
                assert_eq!(
                    expected_state, actual_state,
                    "{checkpoint}/{case}: serialized_state vs PythonJson::state"
                );
            }
        }
    }

    if !checked_any {
        eprintln!("skip: no golden data found for any checkpoint");
    }
}
