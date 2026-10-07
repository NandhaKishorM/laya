"""Deterministic, dependency-free demo embedder for the shortlist golden vectors and samples.

This is **not** a real embedding model: it is a hashed character-trigram bag-of-features
vector, chosen only because it is a pure function of the input text and is trivial to
reimplement bit-for-bit in C# with nothing but a string, UTF-8 encoding and 32-bit
integer arithmetic. It is deliberately named and documented as a demo: any real deployment
should pass `embed_fn_from_agent` (Python) or a real bi-encoder to the shortlist, not this.

The .NET port MUST reproduce this exact algorithm, because
`hashing_embedder_probe.json` and `case_shortlist_many_options.json` in the golden vectors
were both generated with it, and the .NET sample uses it too, so its
stdout can be diffed against the Python-recorded golden line for line.

Algorithm (this docstring is the spec -- read it precisely before porting):

1. Lowercase the input with a full Unicode case mapping (Python's `str.lower()`; C# must use
   `.ToLowerInvariant()` on the *string*, which uses full Unicode case folding rules too --
   NOT a per-UTF-16-unit simple mapping. This matters for characters whose lowercase form is
   more than one code point, e.g. `"İ".lower() == "i̇"` (Turkish capital I with dot
   above lowercases to 2 code points: 'i' + combining dot above).
2. Pad the lowercased text with two ASCII space characters (U+0020) on each side:
   `"  " + text + "  "`. Padding is added even when `text` is empty, so the padded text is
   never shorter than 4 code points and there are always at least 2 windows (see step 3):
   this is what lets short inputs like `""` or `"a"` still produce a nonzero, well-defined
   vector instead of an empty one.
3. Slide a window of exactly 3 **Unicode code points** (not UTF-16 code units, not grapheme
   clusters, not bytes) across the padded text, one code point at a time. For padded text of
   length `n` code points there are `n - 2` windows, at start offsets `0, 1, ..., n - 3`
   inclusive. A code point outside the Basic Multilingual Plane (an "astral" character, e.g.
   an emoji) counts as exactly one code point in this count -- a C# implementation that
   iterates `char` (UTF-16 code units) instead of Unicode scalar values will split a
   surrogate pair into two units and get a different window count and different window
   contents; it must enumerate Unicode scalar values instead (e.g. via `StringInfo` /
   `System.Globalization.StringInfo.GetTextElementEnumerator` is NOT correct either, since
   that walks grapheme clusters -- use rune/code-point enumeration, e.g. .NET's
   `System.Text.Rune` via `str.EnumerateRunes()`).
4. Encode each 3-code-point window as UTF-8 bytes (the window's own bytes, not the whole
   padded text's bytes -- re-encode the 3-code-point substring on its own each time).
5. Hash those UTF-8 bytes with 32-bit FNV-1a:

       hash = 0x811c9dc5                      # FNV offset basis
       for byte in utf8_bytes:
           hash ^= byte
           hash = (hash * 0x01000193) & 0xFFFFFFFF   # FNV prime, wrapped to 32 bits

   `hash` is an unsigned 32-bit integer throughout; the multiply must wrap (mask to
   0xFFFFFFFF / use unchecked arithmetic in C#) exactly like Python's explicit
   `& 0xFFFFFFFF` above.
6. `bucket = hash % 64` (0..63, unsigned).
7. `sign = +1.0 if (hash >> 31) & 1 == 0 else -1.0` (the hash's top bit, as an unsigned
   32-bit value -- this is a sign-hashing trick to reduce feature collisions, not a
   statement about the hash being negative).
8. Accumulate `sign` into `vector[bucket]` for every window (i.e. `vector[bucket] += sign`;
   multiple windows can and do land in the same bucket and their signed contributions add).
   The accumulator is float64 throughout.
9. No normalisation is applied. The shortlist's cosine similarity handles scale on its own,
   and leaving the raw counts in the golden vectors makes them easier to hand-check.

`embed(texts)` returns one such vector per input string, as a `(len(texts), 64)` float64
NumPy array, in input order. It does not batch, cache, or special-case any input; every text
is processed independently and only via the steps above.
"""
from typing import Sequence

import numpy as np

DIM = 64
_FNV_OFFSET_BASIS = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def _fnv1a_32(data: bytes) -> int:
    """32-bit FNV-1a over raw bytes, wrapping like an unsigned 32-bit integer."""
    h = _FNV_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _MASK32
    return h


def _embed_one(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float64)
    padded = "  " + text.lower() + "  "
    n = len(padded)
    for i in range(n - 2):
        window = padded[i : i + 3]
        h = _fnv1a_32(window.encode("utf-8"))
        bucket = h % DIM
        sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
        vec[bucket] += sign
    return vec


def embed(texts: Sequence[str]) -> np.ndarray:
    """Embed a list of strings into a `(len(texts), 64)` float64 array. See the module docstring
    for the exact algorithm; every other-language port must match it bit-for-bit."""
    rows = [_embed_one("" if t is None else str(t)) for t in texts]
    if not rows:
        return np.zeros((0, DIM), dtype=np.float64)
    return np.stack(rows, axis=0)
