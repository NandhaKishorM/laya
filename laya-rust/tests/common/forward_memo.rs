//! Runs byte-identical forward passes once. Test-only: it sits behind
//! [`LayaEngine::set_forward_hook`] and wraps the graph run only, so every case still tokenizes,
//! collates, calibrates, gates and decodes on its own and is compared with its own golden.
//!
//! The key is the whole collated batch: its three dimensions and the exact contents of every
//! tensor the graph is fed (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`),
//! compared element by element by the map, never through a hash alone. A memo belongs to one
//! checkpoint and one layout (see [`ForwardMemos`]), which fixes the graph and so the set of
//! input names. A hit returns a copy, and a lookup does not depend on which case ran first.

use laya::collator::CollatedBatch;
use laya::error::Result;
use laya::laya_engine::{ForwardHook, LayaEngine, LayaOutput};
use laya::laya_options::LayaCheckpoint;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    shape: [usize; 3],
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    marker_pos: Vec<i64>,
    marker_mask: Vec<bool>,
    qtype: Vec<i64>,
}

impl Key {
    fn of(batch: &CollatedBatch) -> Self {
        Self {
            shape: [batch.count, batch.sequence_length, batch.marker_count],
            input_ids: batch.input_ids.clone(),
            attention_mask: batch.attention_mask.clone(),
            marker_pos: batch.marker_pos.clone(),
            marker_mask: batch.marker_mask.clone(),
            qtype: batch.qtype.clone(),
        }
    }
}

#[derive(Default)]
struct State {
    stored: HashMap<Key, LayaOutput>,
    runs: usize,
    hits: usize,
}

/// The stored outputs of one checkpoint and layout.
#[derive(Default)]
pub struct ForwardMemo {
    state: Mutex<State>,
}

impl ForwardMemo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forward passes that reached the graph.
    pub fn runs(&self) -> usize {
        self.state.lock().unwrap().runs
    }

    /// Forward passes answered from an identical earlier one.
    pub fn hits(&self) -> usize {
        self.state.lock().unwrap().hits
    }

    /// The outputs for `batch`, running `run` only when no byte-identical batch was seen.
    pub fn forward(
        &self,
        batch: &CollatedBatch,
        run: &dyn Fn(&CollatedBatch) -> Result<LayaOutput>,
    ) -> Result<LayaOutput> {
        let key = Key::of(batch);
        // Held across the run: tests here are sequential, and a second caller with the same
        // input then waits for the answer instead of running it again.
        let mut state = self.state.lock().unwrap();
        if let Some(hit) = state.stored.get(&key) {
            let copy = hit.clone();
            state.hits += 1;
            return Ok(copy);
        }
        let output = run(batch)?;
        state.runs += 1;
        state.stored.insert(key, output.clone());
        Ok(output)
    }
}

/// One [`ForwardMemo`] per checkpoint and layout.
pub struct ForwardMemos;

impl ForwardMemos {
    /// The memo for one checkpoint and layout, created on first use.
    pub fn for_layout(checkpoint: LayaCheckpoint, split: bool) -> Arc<ForwardMemo> {
        type Registry = Mutex<HashMap<(&'static str, bool), Arc<ForwardMemo>>>;
        static MEMOS: OnceLock<Registry> = OnceLock::new();
        let mut memos = MEMOS.get_or_init(Default::default).lock().unwrap();
        Arc::clone(memos.entry((checkpoint.subdir(), split)).or_default())
    }

    /// Routes `engine`'s forward passes through the memo of its checkpoint and layout until the
    /// returned guard drops.
    pub fn install<'a>(
        engine: &'a LayaEngine,
        checkpoint: LayaCheckpoint,
        split: bool,
    ) -> HookGuard<'a> {
        let memo = Self::for_layout(checkpoint, split);
        let hook: ForwardHook = Arc::new(move |batch, run| memo.forward(batch, run));
        engine.set_forward_hook(Some(hook));
        HookGuard { engine }
    }
}

/// Removes the engine's forward hook on drop.
pub struct HookGuard<'a> {
    engine: &'a LayaEngine,
}

impl Drop for HookGuard<'_> {
    fn drop(&mut self) {
        self.engine.set_forward_hook(None);
    }
}
