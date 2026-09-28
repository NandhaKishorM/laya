//! Tier 2: the exported graph's dynamic axes and its raw outputs, driven directly through
//! [`laya::LayaEngine::run`].
//!
//! The export was traced at one shape, so every axis being genuinely dynamic is a property of the
//! artifact rather than something the exporter guarantees. This sweep re-runs the 72 configurations
//! the Python probe covered, then checks the graph outputs against recorded ones, so a bad artifact
//! is caught here rather than as a puzzling answer mismatch elsewhere.
//!
//! Run the full suite (all three checkpoints) with:
//! `LAYA_ONNX_ROOT=<parent-of-checkpoints> cargo test --release -- --test-threads=1`
//! (see `predict_parity_tests.rs`'s module docs for why `--test-threads=1` matters here).

mod common;

use common::engines;
use common::golden_data::{self, CheckpointGoldenData};
use laya::collator::{CollatedBatch, Collator, SequenceItem};
use laya::laya_config::LayaConfig;
use laya::laya_options::LayaCheckpoint;
use laya::questions::QuestionType;
use serde_json::Value;

/// Candidate sequence lengths, straddling the multilingual traced length of 93. Each checkpoint's
/// actual sweep is this list filtered to `<= config.max_len`, with `max_len` always included.
const SEQUENCE_LENGTHS: [usize; 12] = [16, 32, 55, 59, 80, 92, 93, 94, 128, 256, 512, 1024];
/// Batch sizes: one row, and more than one.
const BATCH_SIZES: [usize; 2] = [1, 3];
/// Marker counts, from the graph's TopK floor of 2 upwards.
const MARKER_COUNTS: [usize; 3] = [2, 5, 9];

fn shapes() -> impl Iterator<Item = (usize, usize, usize)> {
    SEQUENCE_LENGTHS.into_iter().flat_map(|seq| {
        BATCH_SIZES.into_iter().flat_map(move |batch| {
            MARKER_COUNTS
                .into_iter()
                .map(move |markers| (seq, batch, markers))
        })
    })
}

/// The sequence lengths to sweep for one checkpoint: [`SEQUENCE_LENGTHS`] filtered to
/// `<= max_len`, with `max_len` itself always included.
fn checkpoint_sequence_lengths(checkpoint: LayaCheckpoint) -> Vec<usize> {
    // Derive max_len from the artifact config without loading the ONNX graph. Default to 1024
    // (the largest across all checkpoints) when the config cannot be read; the caller's
    // `engines::load` will then skip the whole test anyway if artifacts are absent.
    let max_len = common::test_artifacts::locate(checkpoint)
        .and_then(|dir| LayaConfig::load(dir.join("rl_agent_config.json")).ok())
        .map(|cfg| cfg.max_len)
        .unwrap_or(1024);

    let mut lens: Vec<usize> = SEQUENCE_LENGTHS
        .into_iter()
        .filter(|&s| s <= max_len)
        .collect();
    if !lens.contains(&max_len) {
        lens.push(max_len);
    }
    lens.sort_unstable();
    lens
}

/// Same as [`checkpoint_sequence_lengths`], but reads `rl_agent_config.json` from the split-layout
/// export directory (`<repo>/onnx-split/<ckpt>`) instead of the fused one — the two exports for the
/// same checkpoint are expected to share the same `max_len`, but this reads it from the directory
/// actually under test rather than assuming that.
fn checkpoint_sequence_lengths_split(checkpoint: LayaCheckpoint) -> Vec<usize> {
    let max_len = common::test_artifacts::locate_split(checkpoint)
        .and_then(|dir| LayaConfig::load(dir.join("rl_agent_config.json")).ok())
        .map(|cfg| cfg.max_len)
        .unwrap_or(1024);

    let mut lens: Vec<usize> = SEQUENCE_LENGTHS
        .into_iter()
        .filter(|&s| s <= max_len)
        .collect();
    if !lens.contains(&max_len) {
        lens.push(max_len);
    }
    lens.sort_unstable();
    lens
}

fn item(seq: usize, markers: usize, qtype: QuestionType) -> SequenceItem {
    // A plausible sequence: CLS, ordinary vocabulary ids, SEP. The ids only have to be in range —
    // this exercises shapes, not meaning.
    let mut ids = vec![0u32; seq];
    if seq > 0 {
        ids[0] = 2;
    }
    for (i, id) in ids
        .iter_mut()
        .enumerate()
        .take(seq.saturating_sub(1))
        .skip(1)
    {
        *id = 100 + (i % 5000) as u32;
    }
    if seq > 0 {
        ids[seq - 1] = 1;
    }
    // Markers sit just after CLS and stay inside the sequence.
    let positions: Vec<usize> = (0..markers).map(|i| 1 + i).collect();
    SequenceItem::new(ids, positions, qtype)
}

fn synthesize(seq: usize, batch: usize, markers: usize) -> CollatedBatch {
    let types = [
        QuestionType::Choice,
        QuestionType::Score,
        QuestionType::Noul,
    ];
    let items: Vec<SequenceItem> = (0..batch)
        .map(|r| item(seq, markers, types[r % types.len()]))
        .collect();
    Collator::collate(&items, 0).expect("collate should not fail for a non-empty batch")
}

/// Reconstructs items from a golden case's recorded `batch` field, dropping the padding the
/// collator will re-add, so this isolates the ONNX run from tokenization: a mismatch is the graph
/// or the tensor marshalling, nothing else.
fn from_recorded(recorded: &Value, pad_id: u32) -> CollatedBatch {
    let attention = recorded["attention_mask"]
        .as_array()
        .expect("attention_mask");
    let input_ids = recorded["input_ids"].as_array().expect("input_ids");
    let marker_pos = recorded["marker_pos"].as_array().expect("marker_pos");
    let marker_mask = recorded["marker_mask"].as_array().expect("marker_mask");
    let qtype = recorded["qtype"].as_array().expect("qtype");

    let mut items = Vec::with_capacity(input_ids.len());
    for r in 0..input_ids.len() {
        let length = attention[r]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v.as_i64().unwrap() != 0)
            .count();
        let ids: Vec<u32> = golden_data::ints(&input_ids[r])
            .into_iter()
            .take(length)
            .map(|v| v as u32)
            .collect();

        let positions = golden_data::indices(&marker_pos[r]);
        let mask_row = marker_mask[r].as_array().unwrap();
        let markers: Vec<usize> = positions
            .into_iter()
            .enumerate()
            .filter(|(i, _)| mask_row[*i].as_bool().unwrap())
            .map(|(_, pos)| pos)
            .collect();

        let qt = QuestionType::try_from(qtype[r].as_i64().unwrap()).expect("valid qtype");
        items.push(SequenceItem::new(ids, markers, qt));
    }

    Collator::collate(&items, pad_id).expect("collate")
}

fn assert_close_slices(expected: &[f32], actual: &[f32], tolerance: f64, what: &str) {
    assert_eq!(expected.len(), actual.len(), "{what}: length mismatch");
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (f64::from(e) - f64::from(a)).abs();
        if diff > worst {
            worst = diff;
            at = i;
        }
    }
    eprintln!("{what}: worst diff {worst} at index {at} (tolerance {tolerance})");
    assert!(
        worst <= tolerance,
        "{what}: worst difference {worst} at index {at} (python {}, rust {}), tolerance {tolerance}",
        expected.get(at).copied().unwrap_or(f32::NAN),
        actual.get(at).copied().unwrap_or(f32::NAN)
    );
}

// ── multilingual-only shape tests ────────────────────────────────────────────

#[test]
fn runs_at_every_shape() {
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        return;
    };

    for (seq, batch, markers) in shapes() {
        let collated = synthesize(seq, batch, markers);
        let output = engine
            .run(&collated)
            .unwrap_or_else(|e| panic!("seq={seq} batch={batch} markers={markers}: {e}"));

        assert_eq!(
            batch * markers,
            output.logits.len(),
            "seq={seq} batch={batch} markers={markers}: logits length"
        );
        assert_eq!(
            batch * 2,
            output.act_logits.len(),
            "seq={seq} batch={batch} markers={markers}: act_logits length"
        );
        for &v in &output.logits {
            assert!(
                v.is_finite(),
                "non-finite logit at seq={seq}, markers={markers}"
            );
        }
        for &v in &output.act_logits {
            assert!(v.is_finite(), "non-finite act logit at seq={seq}");
        }
    }
}

#[test]
fn reproduces_the_recorded_graph_outputs() {
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        return;
    };
    let golden = CheckpointGoldenData::for_checkpoint(LayaCheckpoint::Multilingual.subdir());
    if !golden.available() {
        eprintln!("skip: no golden data for 'multilingual'");
        return;
    }

    let pad_id = golden.meta()["special_tokens"]["pad_id"].as_u64().unwrap() as u32;

    for case in golden.success_cases() {
        let root = golden.load_case(&case);
        // Feed the tensors Python recorded rather than ones rebuilt here, so this isolates the
        // ONNX run from tokenization: a mismatch is the graph or the tensor marshalling, nothing
        // else.
        let batch = from_recorded(&root["batch"], pad_id);
        let output = engine
            .run(&batch)
            .unwrap_or_else(|e| panic!("{case}: run failed: {e}"));

        let mut rows = 0;
        let mut cols = 0;
        let expected_logits = golden_data::matrix(&root["logits"], &mut rows, &mut cols);
        assert_eq!(batch.count, rows, "{case}: logits row count");
        assert_eq!(batch.marker_count, cols, "{case}: logits col count");
        // Logits live around +/-10 (and exactly -1e4 in padded columns), so this is a tight bound.
        assert_close_slices(
            &expected_logits,
            &output.logits,
            2e-3,
            &format!("{case}: logits"),
        );

        // act_logits reach +/-1600, where float32 spacing alone is ~1e-4; 0.05 is ~3e-5 relative.
        let expected_act = golden_data::matrix(&root["act_logits"], &mut rows, &mut cols);
        assert_close_slices(
            &expected_act,
            &output.act_logits,
            0.05,
            &format!("{case}: act_logits"),
        );
    }
}

#[test]
fn keeps_padded_marker_columns_at_the_masked_fill_value() {
    // The model writes -1e4 into columns marker_mask says are padding. Calibration relies on that
    // to keep a padded column out of a distribution, so it is worth pinning on the graph itself.
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        return;
    };

    let items = vec![
        item(32, 4, QuestionType::Choice),
        item(32, 2, QuestionType::Choice),
    ];
    let batch = Collator::collate(&items, 0).expect("collate");
    let output = engine.run(&batch).expect("run");

    assert_eq!(4, batch.marker_count);
    let row1 = batch.marker_count; // second row's offset into the flat logits vec
    assert!(
        output.logits[row1 + 2] < -1e3,
        "padded column carried {}",
        output.logits[row1 + 2]
    );
    assert!(
        output.logits[row1 + 3] < -1e3,
        "padded column carried {}",
        output.logits[row1 + 3]
    );
    assert!(output.logits[row1 + 1].is_finite());
}

#[test]
fn runs_a_single_option_question_that_topk_would_otherwise_reject() {
    // The graph has TopK k=2 baked in, so a one-column marker axis fails outright. Collator widens
    // it to two; this is the test that the widening is what actually keeps the graph runnable.
    let Some(engine) = engines::load(LayaCheckpoint::Multilingual) else {
        return;
    };

    let items = vec![item(24, 1, QuestionType::Choice)];
    let batch = Collator::collate(&items, 0).expect("collate");
    assert_eq!(2, batch.marker_count);

    let output = engine.run(&batch).expect("run");
    assert!(output.logits[0].is_finite());
}

// ── all-checkpoint shape test ────────────────────────────────────────────────

#[test]
fn runs_at_every_shape_for_all_checkpoints() {
    for checkpoint in engines::CHECKPOINTS {
        let Some(engine) = engines::load(checkpoint) else {
            continue;
        };

        for seq in checkpoint_sequence_lengths(checkpoint) {
            for batch in BATCH_SIZES {
                for markers in MARKER_COUNTS {
                    let collated = synthesize(seq, batch, markers);
                    let output = engine.run(&collated).unwrap_or_else(|e| {
                        panic!(
                            "{}: seq={seq} batch={batch} markers={markers}: {e}",
                            checkpoint.subdir()
                        )
                    });

                    assert_eq!(
                        batch * markers,
                        output.logits.len(),
                        "{}: seq={seq} batch={batch} markers={markers}: logits length",
                        checkpoint.subdir()
                    );
                    assert_eq!(
                        batch * 2,
                        output.act_logits.len(),
                        "{}: seq={seq} batch={batch} markers={markers}: act_logits length",
                        checkpoint.subdir()
                    );
                    for &v in &output.logits {
                        assert!(
                            v.is_finite(),
                            "{}: non-finite logit: seq={seq} markers={markers}",
                            checkpoint.subdir()
                        );
                    }
                    for &v in &output.act_logits {
                        assert!(
                            v.is_finite(),
                            "{}: non-finite act logit: seq={seq}",
                            checkpoint.subdir()
                        );
                    }
                }
            }
        }
        // `engine` drops here, before the next checkpoint's is loaded.
    }
}

/// Same sweep as [`runs_at_every_shape_for_all_checkpoints`], but against each checkpoint's
/// split-layout export (`<repo>/onnx-split/<ckpt>`, the layout laya-ts's exporter produces:
/// `encoder.onnx` + `head.onnx`) instead of the fused `model.onnx`. Skips per checkpoint exactly
/// like the fused sweep does when that checkpoint's split-layout export is not present locally.
#[test]
fn runs_at_every_shape_for_all_checkpoints_split_layout() {
    for checkpoint in engines::CHECKPOINTS {
        let Some(engine) = engines::load_split(checkpoint) else {
            continue;
        };

        for seq in checkpoint_sequence_lengths_split(checkpoint) {
            for batch in BATCH_SIZES {
                for markers in MARKER_COUNTS {
                    let collated = synthesize(seq, batch, markers);
                    let output = engine.run(&collated).unwrap_or_else(|e| {
                        panic!(
                            "{} (split layout): seq={seq} batch={batch} markers={markers}: {e}",
                            checkpoint.subdir()
                        )
                    });

                    assert_eq!(
                        batch * markers,
                        output.logits.len(),
                        "{} (split layout): seq={seq} batch={batch} markers={markers}: logits length",
                        checkpoint.subdir()
                    );
                    assert_eq!(
                        batch * 2,
                        output.act_logits.len(),
                        "{} (split layout): seq={seq} batch={batch} markers={markers}: act_logits length",
                        checkpoint.subdir()
                    );
                    for &v in &output.logits {
                        assert!(
                            v.is_finite(),
                            "{} (split layout): non-finite logit: seq={seq} markers={markers}",
                            checkpoint.subdir()
                        );
                    }
                    for &v in &output.act_logits {
                        assert!(
                            v.is_finite(),
                            "{} (split layout): non-finite act logit: seq={seq}",
                            checkpoint.subdir()
                        );
                    }
                }
            }
        }
        // `engine` drops here, before the next checkpoint's is loaded.
    }
}
