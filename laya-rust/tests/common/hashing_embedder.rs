//! Deterministic, dependency-free demo embedder for the shortlist golden vectors and end-to-end
//! tests. Ports `tools/hashing_embedder.py` bit-for-bit — see that file's docstring for the exact
//! algorithm spec this follows step by step. **Not** a real embedding model: a hashed
//! character-trigram bag-of-features vector, chosen only because it is a pure function of the
//! input text that is trivial to reproduce identically in every language.
//!
//! Every step below is precision-sensitive:
//! - Lowercasing uses full Unicode case mapping (`str::to_lowercase`), matching Python's
//!   `str.lower()` — not a per-UTF-16-unit simple mapping.
//! - The 3-item sliding window is over Unicode scalar values (`char`, which in Rust already *is*
//!   a Unicode scalar value), matching Python's per-code-point iteration; an astral character
//!   (outside the BMP) is one window position, not two.
//! - The window is re-encoded to UTF-8 on its own each time (not sliced out of one pre-encoded
//!   byte buffer), because a multi-byte character's encoding does not change with position, but
//!   slicing a shared buffer at the wrong byte offset would corrupt it.
//! - The FNV-1a multiply wraps at 32 bits (`u32::wrapping_mul`), matching Python's explicit
//!   `& 0xFFFFFFFF`.

use laya::error::Result as LayaResult;

/// Output vector width.
pub const DIM: usize = 64;

const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
const FNV_PRIME: u32 = 0x0100_0193;

/// 32-bit FNV-1a over raw bytes, wrapping like an unsigned 32-bit integer.
fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn embed_one(text: &str) -> Vec<f64> {
    let mut vector = vec![0.0f64; DIM];
    let padded: Vec<char> = format!("  {}  ", text.to_lowercase()).chars().collect();
    let n = padded.len();
    // Padding alone is 4 characters, so `n >= 4` always and this subtraction never underflows;
    // the guard is defensive documentation, not a reachable branch.
    if n < 3 {
        return vector;
    }
    for start in 0..=(n - 3) {
        let window: String = padded[start..start + 3].iter().collect();
        let hash = fnv1a_32(window.as_bytes());
        let bucket = (hash % DIM as u32) as usize;
        let sign = if (hash >> 31) & 1 == 0 { 1.0 } else { -1.0 };
        vector[bucket] += sign;
    }
    vector
}

/// Embed a list of strings into one `(len(texts), 64)` matrix, in input order. See the module
/// docs for the exact algorithm; `None` inputs have no Rust equivalent (the signature already
/// requires `&str`), matching Python's `"" if t is None else str(t)` only in that an empty string
/// embeds the same way an absent one would have.
///
/// Returns [`laya::error::Result`] so it can be used directly wherever `LayaShortlist` expects an
/// `embed_fn` closure, even though this particular embedder never fails.
pub fn embed(texts: &[String]) -> LayaResult<Vec<Vec<f64>>> {
    Ok(texts.iter().map(|t| embed_one(t)).collect())
}
