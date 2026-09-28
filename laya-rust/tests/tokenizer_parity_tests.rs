//! Token-level parity against Python for every checkpoint, needing `tokenizer/tokenizer.json` but
//! not the model weights.
//!
//! This is where a port like this most plausibly goes wrong: a divergence of one token shifts
//! every marker after it, and the model then answers a different question without anything
//! failing. These tests compare against ids recorded from the real HuggingFace tokenizer, so they
//! fail loudly instead. Each test loops over all three checkpoints and loads the tokenizer inline,
//! so missing artifacts for one checkpoint skip only that checkpoint's cases — the tokenizer alone
//! is a few hundred KB, so (unlike the Tier 2 tests) there is no memory reason to avoid loading
//! all three at once.

mod common;

use common::golden_data::{self, CheckpointGoldenData};
use common::test_artifacts;
use laya::laya_options::LayaCheckpoint;
use laya::python_json::PythonJson;
use laya::sequence_builder::SequenceBuilder;
use laya::tokenization::{HfTokenizer, LayaTokenizer};
use serde_json::Value;

const CHECKPOINTS: [LayaCheckpoint; 3] = [
    LayaCheckpoint::Multilingual,
    LayaCheckpoint::English,
    LayaCheckpoint::TypedDecisions,
];

/// Loads the tokenizer for `checkpoint`, or prints a skip line and returns `None` when
/// `tokenizer/tokenizer.json` is not available locally.
fn load_tokenizer(checkpoint: LayaCheckpoint) -> Option<HfTokenizer> {
    if !test_artifacts::has_tokenizer(checkpoint) {
        eprintln!(
            "skip: no tokenizer.json for '{}'; set LAYA_ONNX_ROOT or LAYA_ONNX_DIR",
            checkpoint.subdir()
        );
        return None;
    }
    let dir = test_artifacts::locate(checkpoint).unwrap_or_else(|| {
        panic!(
            "has_tokenizer was true but locate() found nothing for '{}'",
            checkpoint.subdir()
        )
    });
    let path = dir.join("tokenizer").join("tokenizer.json");
    Some(HfTokenizer::from_file(&path).unwrap_or_else(|e| {
        panic!(
            "failed to load tokenizer for '{}' at {}: {e}",
            checkpoint.subdir(),
            path.display()
        )
    }))
}

fn to_u32(values: Vec<i64>) -> Vec<u32> {
    values.into_iter().map(|v| v as u32).collect()
}

#[test]
fn special_token_ids_match_the_recorded_values() {
    for checkpoint in CHECKPOINTS {
        let Some(tok) = load_tokenizer(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!(
                "skip: no golden data for '{}'; run tools/dump_golden.py",
                checkpoint.subdir()
            );
            continue;
        }

        let meta = golden.meta();
        let expected = &meta["special_tokens"];
        let special = tok.special();
        let name = checkpoint.subdir();

        assert_eq!(
            expected["pad_id"].as_u64().unwrap() as u32,
            special.pad_id,
            "{name}: pad_id"
        );
        assert_eq!(
            expected["cls_id"].as_u64().unwrap() as u32,
            special.cls_id,
            "{name}: cls_id"
        );
        assert_eq!(
            expected["sep_id"].as_u64().unwrap() as u32,
            special.sep_id,
            "{name}: sep_id"
        );
        assert_eq!(
            expected["mask_id"].as_u64().unwrap() as u32,
            special.mask_id,
            "{name}: mask_id"
        );
        assert_eq!(
            expected["unk_id"].as_u64().unwrap() as u32,
            special.unk_id,
            "{name}: unk_id"
        );
        assert_eq!(
            expected["mask_token"].as_str().unwrap(),
            special.mask_token,
            "{name}: mask_token"
        );
    }
}

#[test]
fn encodes_every_probe_string_exactly_as_python_does() {
    for checkpoint in CHECKPOINTS {
        let Some(tok) = load_tokenizer(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            eprintln!(
                "skip: no golden data for '{}'; run tools/dump_golden.py",
                checkpoint.subdir()
            );
            continue;
        }

        let probes = golden.load("tokenizer_probe.json");
        let mut failures = Vec::new();
        for probe in probes
            .as_array()
            .expect("tokenizer_probe.json must be an array")
        {
            let text = probe["text"].as_str().expect("probe entry needs 'text'");
            let expected = to_u32(golden_data::ints(&probe["input_ids"]));
            let actual = tok
                .encode(text)
                .unwrap_or_else(|e| panic!("{}: encode {text:?} failed: {e}", checkpoint.subdir()));
            if expected != actual {
                failures.push(format!(
                    "  {text:?}\n    python: {expected:?}\n    rust:   {actual:?}"
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "{}: {} probe string(s) tokenized differently:\n{}",
            checkpoint.subdir(),
            failures.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn adds_no_special_tokens_of_its_own() {
    // The shipped tokenizer.json has a TemplateProcessing post-processor that wraps every encode.
    // Left in place it would inject the checkpoint's CLS and SEP ids into every fragment, shifting
    // all marker positions. HfTokenizer must never invoke it.
    for checkpoint in CHECKPOINTS {
        let Some(tok) = load_tokenizer(checkpoint) else {
            continue;
        };
        let ids = tok.encode("hello").unwrap();
        assert_ne!(
            tok.special().cls_id,
            ids[0],
            "{}: leading id looks like an injected CLS",
            checkpoint.subdir()
        );
        assert_ne!(
            tok.special().sep_id,
            *ids.last().unwrap(),
            "{}: trailing id looks like an injected SEP",
            checkpoint.subdir()
        );
    }
}

#[test]
fn encodes_the_empty_string_to_nothing() {
    for checkpoint in CHECKPOINTS {
        let Some(tok) = load_tokenizer(checkpoint) else {
            continue;
        };
        assert!(
            tok.encode("").unwrap().is_empty(),
            "{}: empty string should encode to nothing",
            checkpoint.subdir()
        );
    }
}

#[test]
fn builds_the_recorded_sequence_for_every_question() {
    for checkpoint in CHECKPOINTS {
        let Some(tok) = load_tokenizer(checkpoint) else {
            continue;
        };
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            continue;
        }

        let meta = golden.meta();
        let max_len = meta["max_len"].as_u64().unwrap() as usize;
        let head_max_len = meta["head_max_len"].as_u64().unwrap() as usize;

        for case in golden.success_cases() {
            let root = golden.load_case(&case);
            let state = root["state"].clone();
            let questions = golden_data::build_questions(&root["questions"]).unwrap_or_else(|e| {
                panic!(
                    "{}/{case}: failed to build questions: {e}",
                    checkpoint.subdir()
                )
            });

            for recorded in root["per_question"]
                .as_array()
                .expect("case must have 'per_question'")
            {
                let id = recorded["id"]
                    .as_str()
                    .expect("per_question entry needs 'id'");
                let question = questions.get(id).unwrap_or_else(|| {
                    panic!("{}/{case}: missing question '{id}'", checkpoint.subdir())
                });

                // The rendered option strings come first: they are the readable form of a
                // divergence, so a mismatch here explains a mismatch in the ids below.
                let expected_options: Vec<&str> = recorded["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                assert_eq!(
                    expected_options,
                    SequenceBuilder::render_options(question),
                    "{}/{case}/{id}: options",
                    checkpoint.subdir()
                );
                assert_eq!(
                    recorded["instructions"].as_str().unwrap(),
                    SequenceBuilder::render_instructions(question),
                    "{}/{case}/{id}: instructions",
                    checkpoint.subdir()
                );

                let (ids, markers) =
                    SequenceBuilder::build(&tok, &state, question, max_len, head_max_len, false)
                        .unwrap_or_else(|e| {
                            panic!("{}/{case}/{id}: build failed: {e}", checkpoint.subdir())
                        });

                assert_eq!(
                    golden_data::indices(&recorded["markers"]),
                    markers,
                    "{}/{case}/{id}: markers",
                    checkpoint.subdir()
                );
                assert_eq!(
                    to_u32(golden_data::ints(&recorded["input_ids"])),
                    ids,
                    "{}/{case}/{id}: input_ids",
                    checkpoint.subdir()
                );
            }
        }
    }
}

#[test]
fn serializes_state_exactly_as_python_does() {
    // State serialization is checkpoint-independent (it is the same JSON serializer for all), but
    // exercising it across checkpoints gives coverage over diverse state shapes in the golden
    // corpus of every checkpoint.
    for checkpoint in CHECKPOINTS {
        let golden = CheckpointGoldenData::for_checkpoint(checkpoint.subdir());
        if !golden.available() {
            continue;
        }
        for case in golden.index() {
            let root = golden.load_case(&case);
            if let Some(expected) = root.get("serialized_state").and_then(Value::as_str) {
                let actual = PythonJson::state(&root["state"]);
                assert_eq!(
                    expected,
                    actual,
                    "{}/{case}: serialized_state vs PythonJson::state",
                    checkpoint.subdir()
                );
            }
        }
    }
}
