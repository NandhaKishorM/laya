//! The tokenizer trait the engine depends on, decoupling it from any one backend.
//! `hf_tokenizer.rs` (a real HuggingFace-backed implementation) is the one the engine uses.

use crate::error::Result;

/// The special token ids a checkpoint's tokenizer resolves to, plus the literal mask token string
/// used to scrub forged markers out of user input (see `sequence-construction.md`'s "Mask
/// injection defense").
///
/// Ids and the mask token string differ by checkpoint (mmBERT uses ids 0-4, ModernBERT uses ids
/// 50280-50284) and must always be read from the checkpoint's own config files rather than
/// hard-coded.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Token id of the padding token, used to right-pad sequences to a uniform length.
    pub pad_id: u32,
    /// Token id of the CLS token, prepended to every sequence before the head fragment.
    pub cls_id: u32,
    /// Token id of the SEP token, inserted between the head, the option fragments, and the state.
    pub sep_id: u32,
    /// Token id of the mask token. Each option fragment starts with one; its position in the
    /// sequence is recorded as a marker and later read by the scoring head.
    pub mask_id: u32,
    /// Token id of the unknown-token.
    pub unk_id: u32,
    /// The literal mask token string (e.g. `"<mask>"` for mmBERT or `"[MASK]"` for ModernBERT).
    pub mask_token: String,
}

/// Tokenizes text for the Laya ONNX model, matching the behaviour of the Python `tokenizers`
/// library called with `add_special_tokens=False`. Ports `ILayaTokenizer`.
pub trait LayaTokenizer: Send + Sync {
    /// Encodes `text` to token ids with **no** special tokens prepended or appended. Equivalent to
    /// Python's `tok(text, add_special_tokens=False).input_ids`.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// The special token ids and mask token string this tokenizer resolved at load time.
    fn special(&self) -> &SpecialTokens;
}
