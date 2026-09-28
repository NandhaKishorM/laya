//! A deterministic tokenizer with no model behind it, for testing sequence structure.
//!
//! It emits one id per whitespace-separated word, so a sequence's shape — marker positions, the
//! order of head, options and state, where truncation lands — can be asserted by counting words.
//! It says nothing about real token ids; that is what the tokenizer-parity tests are for.

use laya::error::Result;
use laya::tokenization::{LayaTokenizer, SpecialTokens};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

/// Ids start here, above every special id, so a word is never mistaken for one.
pub const FIRST_WORD_ID: u32 = 100;

/// See the module docs.
pub struct StubTokenizer {
    special: SpecialTokens,
    encoded: Mutex<Vec<String>>,
}

impl StubTokenizer {
    /// A stub tokenizer with the mmBERT-style special token ids (0-4) `SequenceBuilder` tests are
    /// written against.
    pub fn new() -> Self {
        Self {
            special: SpecialTokens {
                pad_id: 0,
                sep_id: 1,
                cls_id: 2,
                unk_id: 3,
                mask_id: 4,
                mask_token: "<mask>".to_string(),
            },
            encoded: Mutex::new(Vec::new()),
        }
    }

    /// Every text passed to [`LayaTokenizer::encode`], in call order.
    pub fn encoded(&self) -> Vec<String> {
        self.encoded
            .lock()
            .expect("encoded log mutex poisoned")
            .clone()
    }

    /// How many times [`LayaTokenizer::encode`] has been called.
    pub fn encode_calls(&self) -> usize {
        self.encoded().len()
    }
}

impl Default for StubTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl LayaTokenizer for StubTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encoded
            .lock()
            .expect("encoded log mutex poisoned")
            .push(text.to_string());
        // A stable, collision-tolerant id per word: the tests only need determinism, and hashing
        // keeps identical words mapping to identical ids across calls.
        Ok(text
            .split_whitespace()
            .map(|word| {
                let mut hasher = DefaultHasher::new();
                word.hash(&mut hasher);
                FIRST_WORD_ID + (hasher.finish() % 10_000) as u32
            })
            .collect())
    }

    fn special(&self) -> &SpecialTokens {
        &self.special
    }
}
