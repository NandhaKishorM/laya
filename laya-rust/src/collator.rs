//! Pads tokenized questions into one batch. Ports `collate_items` from `laya/common.py`. The
//! padded tensors match the input shapes the ONNX graph expects.

use crate::error::{LayaError, Result};
use crate::questions::QuestionType;

/// One question, tokenized and ready to be batched.
#[derive(Debug, Clone)]
pub struct SequenceItem {
    /// The token sequence.
    pub ids: Vec<u32>,
    /// One position per option, pointing at that option's `[MASK]`.
    pub markers: Vec<usize>,
    /// Which question type, fed to the model's type embedding.
    pub qtype: QuestionType,
}

impl SequenceItem {
    /// Build a sequence item from its parts.
    pub fn new(ids: Vec<u32>, markers: Vec<usize>, qtype: QuestionType) -> Self {
        Self {
            ids,
            markers,
            qtype,
        }
    }
}

/// A padded batch, laid out as the row-major tensors the ONNX graph takes.
#[derive(Debug, Clone)]
pub struct CollatedBatch {
    /// Rows in the batch: one per question.
    pub count: usize,
    /// Padded sequence length, the longest row.
    pub sequence_length: usize,
    /// Padded marker-axis width.
    pub marker_count: usize,
    /// `[count, sequence_length]`, padded with the pad id.
    pub input_ids: Vec<i64>,
    /// `[count, sequence_length]`, 1 on real tokens and 0 on padding.
    pub attention_mask: Vec<i64>,
    /// `[count, marker_count]`, 0 in padding columns.
    pub marker_pos: Vec<i64>,
    /// `[count, marker_count]`, false in padding columns.
    pub marker_mask: Vec<bool>,
    /// `[count]`, the question type per row.
    pub qtype: Vec<i64>,
    /// The real (unpadded) option count per row — how many logits to read from that row.
    pub marker_counts: Vec<usize>,
    /// Non-padding tokens across the batch, which is what Python reports as input tokens.
    pub input_tokens: usize,
}

/// Namespace for batch collation. See the module docs for the source this ports.
pub struct Collator;

impl Collator {
    /// The marker axis is never narrower than this.
    ///
    /// The tracer baked `TopK k=2` into the exported graph: Python chooses the top-2 branch at
    /// runtime and the export only ever saw two or more markers, so a batch whose questions all
    /// have a single option makes TopK fail outright. Padding to two columns with `marker_mask =
    /// false` is exactly equivalent, not merely close: the model's `masked_fill` puts `-1e4` in
    /// the pad column, which underflows softmax to 0.0, so `p == [1.0, 0.0]` — precisely the
    /// `[top1, 0.0]` Python substitutes. Reading only [`CollatedBatch::marker_counts`] logits per
    /// row keeps the pad column out of the answer.
    pub const MIN_MARKERS: usize = 2;

    /// Pad a batch of tokenized questions. Returns an error for an empty batch rather than
    /// producing a zero-row tensor the ONNX graph would reject anyway.
    pub fn collate(items: &[SequenceItem], pad_id: u32) -> Result<CollatedBatch> {
        if items.is_empty() {
            return Err(LayaError::InvalidQuestion("nothing to collate".to_string()));
        }

        let n = items.len();
        let length = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
        let marker_count =
            Self::MIN_MARKERS.max(items.iter().map(|it| it.markers.len()).max().unwrap_or(0));

        let mut input_ids = vec![i64::from(pad_id); n * length];
        let mut attention_mask = vec![0i64; n * length];
        let mut marker_pos = vec![0i64; n * marker_count];
        let mut marker_mask = vec![false; n * marker_count];
        let mut qtype = vec![0i64; n];
        let mut marker_counts = vec![0usize; n];
        let mut input_tokens = 0usize;

        for (i, item) in items.iter().enumerate() {
            let row = i * length;
            for (j, &id) in item.ids.iter().enumerate() {
                input_ids[row + j] = i64::from(id);
                attention_mask[row + j] = 1;
            }
            input_tokens += item.ids.len();

            let mrow = i * marker_count;
            for (j, &pos) in item.markers.iter().enumerate() {
                marker_pos[mrow + j] = pos as i64;
                marker_mask[mrow + j] = true;
            }

            qtype[i] = item.qtype as i64;
            marker_counts[i] = item.markers.len();
        }

        Ok(CollatedBatch {
            count: n,
            sequence_length: length,
            marker_count,
            input_ids,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
            marker_counts,
            input_tokens,
        })
    }
}
