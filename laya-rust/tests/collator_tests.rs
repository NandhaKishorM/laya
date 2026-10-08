//! Tier 1: batch padding, which needs no artifacts.

mod common;

use common::golden_data::{self, CheckpointGoldenData};
use laya::collator::{Collator, SequenceItem};
use laya::questions::QuestionType;

fn item(length: usize, markers: usize, qtype: QuestionType) -> SequenceItem {
    SequenceItem::new(
        (10..10 + length as u32).collect(),
        (1..=markers).collect(),
        qtype,
    )
}

#[test]
fn pads_to_the_longest_row_and_marks_attention_accordingly() {
    let batch = Collator::collate(
        &[
            item(5, 2, QuestionType::Choice),
            item(3, 2, QuestionType::Choice),
        ],
        0,
    )
    .unwrap();

    assert_eq!(2, batch.count);
    assert_eq!(5, batch.sequence_length);
    assert_eq!(vec![1, 1, 1, 1, 1, 1, 1, 1, 0, 0], batch.attention_mask);
    // The short row's tail is the pad id, and attention is 0 there.
    assert_eq!(0, batch.input_ids[8]);
    assert_eq!(0, batch.input_ids[9]);
}

#[test]
fn fills_with_the_given_pad_id_rather_than_zero() {
    let batch = Collator::collate(
        &[
            item(3, 2, QuestionType::Choice),
            item(1, 2, QuestionType::Choice),
        ],
        77,
    )
    .unwrap();
    assert_eq!(77, batch.input_ids[4]);
    assert_eq!(77, batch.input_ids[5]);
    assert_eq!(0, batch.attention_mask[4]);
}

#[test]
fn counts_only_real_tokens_as_input_tokens() {
    let batch = Collator::collate(
        &[
            item(5, 2, QuestionType::Choice),
            item(3, 2, QuestionType::Choice),
        ],
        0,
    )
    .unwrap();
    assert_eq!(8, batch.input_tokens);
}

#[test]
fn widens_the_marker_axis_to_two_for_single_option_batches() {
    // The exported graph has TopK k=2 baked in: a one-column marker axis makes it fail outright.
    let batch = Collator::collate(&[item(6, 1, QuestionType::Choice)], 0).unwrap();

    assert_eq!(Collator::MIN_MARKERS, batch.marker_count);
    assert_eq!(vec![1, 0], batch.marker_pos);
    assert_eq!(vec![true, false], batch.marker_mask);
    // The real option count is kept separately, so the pad column never reaches an answer.
    assert_eq!(vec![1], batch.marker_counts);
}

#[test]
fn pads_the_marker_axis_to_the_widest_question() {
    let batch = Collator::collate(
        &[
            item(6, 4, QuestionType::Choice),
            item(6, 2, QuestionType::Choice),
        ],
        0,
    )
    .unwrap();

    assert_eq!(4, batch.marker_count);
    assert_eq!(vec![1, 2, 3, 4, 1, 2, 0, 0], batch.marker_pos);
    assert_eq!(
        vec![true, true, true, true, true, true, false, false],
        batch.marker_mask
    );
    assert_eq!(vec![4, 2], batch.marker_counts);
}

#[test]
fn carries_the_question_type_per_row_using_pythons_numbering() {
    let batch = Collator::collate(
        &[
            item(4, 2, QuestionType::Noul),
            item(4, 2, QuestionType::Choice),
            item(4, 2, QuestionType::Score),
        ],
        0,
    )
    .unwrap();

    assert_eq!(vec![2, 0, 1], batch.qtype);
}

#[test]
fn refuses_an_empty_batch() {
    assert!(Collator::collate(&[], 0).is_err());
}

#[test]
fn reproduces_the_recorded_batch_tensors() {
    let data = CheckpointGoldenData::for_checkpoint("multilingual");
    if !data.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }

    let pad_id = data.meta()["special_tokens"]["pad_id"].as_u64().unwrap() as u32;

    for case in data.success_cases() {
        let root = data.load_case(&case);
        let expected = &root["batch"];

        // Rebuild the batch from the recorded per-question ids, so this checks the collator alone
        // and stays meaningful without a tokenizer present.
        let items: Vec<SequenceItem> = root["per_question"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                let ids: Vec<u32> = golden_data::ints(&q["input_ids"])
                    .into_iter()
                    .map(|v| v as u32)
                    .collect();
                let markers = golden_data::indices(&q["markers"]);
                let qtype = QuestionType::try_from(q["qtype"].as_i64().unwrap())
                    .expect("valid qtype in golden data");
                SequenceItem::new(ids, markers, qtype)
            })
            .collect();

        let widest_markers = items.iter().map(|it| it.markers.len()).max().unwrap_or(0);
        let batch = Collator::collate(&items, pad_id).unwrap_or_else(|e| {
            panic!("case '{case}': collate failed: {e}");
        });

        assert_matrix(
            &expected["input_ids"],
            &batch.input_ids,
            batch.sequence_length,
            &case,
        );
        assert_matrix(
            &expected["attention_mask"],
            &batch.attention_mask,
            batch.sequence_length,
            &case,
        );
        assert_matrix(
            &expected["marker_pos"],
            &batch.marker_pos,
            batch.marker_count,
            &case,
        );

        let expected_qtype = golden_data::ints(&expected["qtype"]);
        assert_eq!(expected_qtype, batch.qtype, "case '{case}': qtype mismatch");

        let expected_mask = expected["marker_mask"].as_array().unwrap();
        assert_eq!(
            expected_mask.len(),
            batch.count,
            "case '{case}': marker_mask row count"
        );
        for (r, row) in expected_mask.iter().enumerate() {
            let row = row.as_array().unwrap();
            for (c, expected) in row.iter().enumerate().take(batch.marker_count) {
                assert_eq!(
                    expected.as_bool().unwrap(),
                    batch.marker_mask[r * batch.marker_count + c],
                    "case '{case}': marker_mask[{r}][{c}]"
                );
            }
        }

        let expected_input_tokens =
            root["result"]["usage"]["input_tokens"].as_u64().unwrap() as usize;
        assert_eq!(
            expected_input_tokens, batch.input_tokens,
            "case '{case}': input_tokens"
        );

        // marker_pos above already pinned the padded width; this checks the flag the dump script
        // set, so a future golden that stops exercising the TopK edge case is noticed.
        let expected_padded = root["marker_axis_padded"].as_bool().unwrap();
        assert_eq!(
            expected_padded,
            batch.marker_count > widest_markers,
            "case '{case}': marker_axis_padded flag"
        );
    }
}

fn assert_matrix(
    expected: &serde_json::Value,
    actual: &[i64],
    cols: usize,
    case: &golden_data::GoldenCaseInfo,
) {
    let rows = expected.as_array().unwrap();
    assert_eq!(
        rows.len() * cols,
        actual.len(),
        "case '{case}': matrix size"
    );
    for (r, row) in rows.iter().enumerate() {
        let row = row.as_array().unwrap();
        assert_eq!(cols, row.len(), "case '{case}': row {r} width");
        for (c, value) in row.iter().enumerate() {
            assert_eq!(
                value.as_i64().unwrap(),
                actual[r * cols + c],
                "case '{case}': [{r}][{c}]"
            );
        }
    }
}
