# Decimal digit/place features

`laya.numeric` provides optional, standard-library-only helpers for representing
numeric text in local model extensions. Importing or using them does not change
LAYA predictions, model weights, tokenization, or training objectives. No model
download, GPU, hosted service, or external API is needed.

## Preserve decimal significance

```python
from laya.numeric import decimal_slots

slots = decimal_slots("12.30e2")
assert [(s.digit, s.power) for s in slots] == [(1, 3), (2, 2), (3, 1), (0, 0)]
```

Each immutable `DigitSlot` contains `digit`, `power`, `character`, and `negative`.
The digit contributes `digit * 10**power` to the literal's magnitude. `negative`
records the literal's sign, including negative zero; it is not a per-digit sign.
`character` is a Python character index, with an optional nonnegative `start`
offset. Exponent digits change powers and are not emitted as mantissa digits.
Leading and trailing zeros remain represented.

Parsing never converts the literal to a floating-point number. Large integers
therefore retain all their digits. A large exponent changes integer powers, not
the number of returned slots; no zero-filled magnitude-sized array is allocated.

Accepted syntax is an optional sign, ASCII integer digits, an optional decimal
point followed by digits, and an optional signed scientific exponent. Unsupported
inputs raise `ValueError`: examples include whitespace, comma-separated numbers,
NaN, infinity, hexadecimal, `.5`, and `1.`. Normalize other formats explicitly in
the caller rather than guessing locale or numeric conventions. Exponent parsing
uses Python integers and is subject to the interpreter's integer-string limits.

## Align complete quantities to tokens

```python
from laya.numeric import align_slots

text = "x=123.40"
spans = [(2, 8)]  # Caller-supplied complete numeric literals.
offsets = [(0, 0), (0, 2), (2, 5), (5, 6), (6, 8), (0, 0)]
features = align_slots(text, spans, offsets)
assert [f.token for f in features] == [2, 2, 2, 4, 4]
assert [f.power for f in features] == [2, 1, 0, -1, -2]
```

Use a tokenizer's character `offset_mapping` for the exact supplied text.
Ranges are half-open Python character offsets, not byte positions. Offsets from
separately tokenized substrings need conversion to the original text coordinates.
Zero-length special-token offsets are ignored but retain their token indices.
Multiple digits inside one token remain distinct features.

Each immutable `AlignedDigit` contains `quantity`, `token`, `digit`, `power`, and
`negative`. Quantity indices follow the caller's span order. Overlapping token
ranges or quantity spans are rejected. Every character of a selected literal,
including signs, decimal points, and exponents, must be covered. Truncated input
raises instead of silently producing a different number. The caller remains
responsible for supplying offsets and complete literal spans from the same text.

These are representation helpers, not a comparison engine: they do not identify
fields, infer units, convert quantities, calculate verdicts, or establish model
accuracy. A trained extension must define how to consume the features. Existing
checkpoints and prediction APIs remain unchanged.

Run the offline checks with `python tests/test_numeric.py`.
