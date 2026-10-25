//! Tier 1: the forward-pass memo used by the batch parity tests answers only byte-identical inputs
//! from its store. Needs no artifacts.

mod common;

use common::forward_memo::{ForwardMemo, ForwardMemos};
use laya::collator::{CollatedBatch, Collator, SequenceItem};
use laya::laya_engine::LayaOutput;
use laya::laya_options::LayaCheckpoint;
use laya::questions::QuestionType;
use std::cell::Cell;

fn batch(items: &[SequenceItem]) -> CollatedBatch {
    Collator::collate(items, 0).unwrap()
}

fn item(ids: &[u32], markers: &[usize], qtype: QuestionType) -> SequenceItem {
    SequenceItem::new(ids.to_vec(), markers.to_vec(), qtype)
}

fn base() -> SequenceItem {
    item(&[5, 6, 7, 8], &[1, 2], QuestionType::Choice)
}

/// Counts the real runs and returns outputs derived from the call count.
struct Graph {
    calls: Cell<usize>,
}

impl Graph {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
        }
    }

    fn run(&self, batch: &CollatedBatch) -> laya::error::Result<LayaOutput> {
        self.calls.set(self.calls.get() + 1);
        Ok(LayaOutput {
            logits: vec![
                batch.input_ids.iter().sum::<i64>() as f32 + self.calls.get() as f32,
                1.0,
            ],
            act_logits: vec![batch.marker_pos.iter().sum::<i64>() as f32, 2.0],
        })
    }
}

fn forward(memo: &ForwardMemo, graph: &Graph, b: &CollatedBatch) -> LayaOutput {
    memo.forward(b, &|x| graph.run(x)).unwrap()
}

#[test]
fn an_identical_batch_runs_the_graph_once() {
    let memo = ForwardMemo::new();
    let graph = Graph::new();
    let first = forward(&memo, &graph, &batch(&[base()]));
    let second = forward(&memo, &graph, &batch(&[base()]));

    assert_eq!(1, graph.calls.get());
    assert_eq!((1, 1), (memo.runs(), memo.hits()));
    assert_eq!(first.logits, second.logits);
    assert_eq!(first.act_logits, second.act_logits);
}

#[test]
fn a_hit_returns_a_copy_so_a_caller_cannot_alter_the_store() {
    let memo = ForwardMemo::new();
    let graph = Graph::new();
    let mut first = forward(&memo, &graph, &batch(&[base()]));
    let expected = first.logits.clone();
    first.logits[0] = -1.0;
    let mut second = forward(&memo, &graph, &batch(&[base()]));
    second.logits[0] = -2.0;
    let third = forward(&memo, &graph, &batch(&[base()]));

    assert_eq!(expected, third.logits);
}

#[test]
fn any_differing_input_runs_the_graph_again() {
    let differing = [
        // one token id
        batch(&[item(&[5, 6, 7, 9], &[1, 2], QuestionType::Choice)]),
        // one marker position
        batch(&[item(&[5, 6, 7, 8], &[1, 3], QuestionType::Choice)]),
        // question type
        batch(&[item(&[5, 6, 7, 8], &[1, 2], QuestionType::Noul)]),
        // marker count (shape and mask)
        batch(&[item(&[5, 6, 7, 8], &[1, 2, 3], QuestionType::Choice)]),
        // sequence length
        batch(&[item(&[5, 6, 7, 8, 8], &[1, 2], QuestionType::Choice)]),
        // row count
        batch(&[base(), base()]),
        // padding and attention mask
        batch(&[base(), item(&[5, 6], &[1, 0], QuestionType::Choice)]),
    ];
    for (index, other) in differing.iter().enumerate() {
        let memo = ForwardMemo::new();
        let graph = Graph::new();
        forward(&memo, &graph, &batch(&[base()]));
        forward(&memo, &graph, other);
        assert_eq!(2, graph.calls.get(), "differing batch {index}");
        assert_eq!(0, memo.hits(), "differing batch {index}");
    }
}

#[test]
fn a_differing_attention_mask_is_a_miss_even_when_the_shape_matches() {
    // Same shape: the second row is shorter, so its padding and attention mask differ.
    let memo = ForwardMemo::new();
    let graph = Graph::new();
    let full = batch(&[base(), base()]);
    let shorter = batch(&[base(), item(&[5, 6, 7], &[1, 2], QuestionType::Choice)]);
    assert_eq!(full.input_ids.len(), shorter.input_ids.len());
    forward(&memo, &graph, &full);
    forward(&memo, &graph, &batch(&[base(), base()]));
    assert_eq!(1, graph.calls.get());
    forward(&memo, &graph, &shorter);
    assert_eq!(2, graph.calls.get());
}

#[test]
fn an_error_from_the_graph_is_not_stored() {
    let memo = ForwardMemo::new();
    let b = batch(&[base()]);
    let failed = memo.forward(&b, &|_| {
        Err(laya::error::LayaError::InvalidConfig("x".into()))
    });
    assert!(failed.is_err());
    let graph = Graph::new();
    forward(&memo, &graph, &b);
    assert_eq!((1, 0), (memo.runs(), memo.hits()));
    assert_eq!(1, graph.calls.get());
}

#[test]
fn checkpoints_and_layouts_keep_separate_memos() {
    let b = batch(&[base()]);
    let graph = Graph::new();
    let checkpoints = [
        LayaCheckpoint::Multilingual,
        LayaCheckpoint::English,
        LayaCheckpoint::TypedDecisions,
    ];
    let mut memos = Vec::new();
    for checkpoint in checkpoints {
        for split in [false, true] {
            memos.push(ForwardMemos::for_layout(checkpoint, split));
        }
    }
    for (i, a) in memos.iter().enumerate() {
        for other in &memos[i + 1..] {
            assert!(!std::sync::Arc::ptr_eq(a, other));
        }
    }
    assert!(std::sync::Arc::ptr_eq(
        &memos[0],
        &ForwardMemos::for_layout(LayaCheckpoint::Multilingual, false)
    ));
    // The same batch in every memo reaches the graph every time: nothing leaks across memos.
    for memo in &memos {
        forward(memo, &graph, &b);
    }
    assert_eq!(memos.len(), graph.calls.get());
}
