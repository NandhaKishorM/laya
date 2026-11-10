"""Lossless decimal digit/place features for optional local model extensions.

Callers provide complete numeric-literal spans and tokenizer character offsets.
These helpers do not infer operands, units, comparisons, or decision labels and
do not change the tokenizer or any model's prediction path.
"""
from dataclasses import dataclass
import re
from typing import Sequence


__all__ = ["DigitSlot", "AlignedDigit", "decimal_slots", "align_slots"]

_DECIMAL_LITERAL = re.compile(
    r"(?P<sign>[+-]?)(?P<integer>[0-9]+)(?:\.(?P<fraction>[0-9]+))?"
    r"(?:[eE](?P<exponent>[+-]?[0-9]+))?\Z"
)


@dataclass(frozen=True)
class DigitSlot:
    """One mantissa digit, its base-ten power, text position, and literal sign."""

    digit: int
    power: int
    character: int
    negative: bool


@dataclass(frozen=True)
class AlignedDigit:
    """A digit/place feature bound to a caller's quantity index and token index."""

    quantity: int
    token: int
    digit: int
    power: int
    negative: bool


def decimal_slots(literal: str, start: int = 0) -> tuple[DigitSlot, ...]:
    """Represent a complete decimal literal without conversion to floating point.

    Accept ASCII digits, an optional sign, an optional fractional part, and an
    optional signed scientific exponent. At least one integer digit is required.
    Exponent digits shift powers; they are not emitted as mantissa digits.
    Leading/trailing zeros and negative-zero signs are retained. ``start`` is a
    nonnegative Python character offset into the original text.

    Raises ValueError for invalid literals or offsets. Whitespace, commas, NaN,
    infinity, hexadecimal, ``.5`` and ``1.`` are deliberately unsupported.
    """
    if not isinstance(literal, str) or type(start) is not int or start < 0:
        raise ValueError("a string literal and nonnegative integer start are required")
    match = _DECIMAL_LITERAL.fullmatch(literal)
    if match is None:
        raise ValueError("unsupported decimal literal")
    exponent = int(match['exponent'] or '0')
    negative = match['sign'] == '-'
    slots = []
    integer = match['integer']
    for index, digit in enumerate(integer):
        slots.append(DigitSlot(int(digit), exponent + len(integer) - index - 1,
                               start + match.start('integer') + index, negative))
    for index, digit in enumerate(match['fraction'] or ''):
        slots.append(DigitSlot(int(digit), exponent - index - 1,
                               start + match.start('fraction') + index, negative))
    return tuple(slots)


def _span(value, length, *, empty):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("spans must be pairs of integer character offsets")
    left, right = value
    if type(left) is not int or type(right) is not int:
        raise ValueError("spans must be pairs of integer character offsets")
    if not 0 <= left <= right <= length or (left == right and not empty):
        raise ValueError("span is empty or outside text")
    return left, right


def align_slots(text: str, spans: Sequence[tuple[int, int]],
                offsets: Sequence[tuple[int, int]]) -> tuple[AlignedDigit, ...]:
    """Bind decimal digits to unmodified tokenizer character-offset spans.

    ``spans`` selects complete literals in quantity order. ``offsets`` contains
    half-open Python character ranges in token order for this exact ``text``;
    byte offsets and offsets relative to separately tokenized substrings must
    first be converted by the caller. Zero-length token spans are ignored but
    still count toward token indices. Multiple digits per token remain separate.

    Raises ValueError for malformed or overlapping ranges, unsupported literals,
    or uncovered literal characters (including signs, points, and exponents).
    Empty quantity/token sequences are allowed; an uncovered quantity is not.
    No text outside the supplied quantity spans is interpreted as a number.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if (not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)) or
            not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes))):
        raise ValueError("quantity spans and token offsets must be sequences of pairs")
    owner = {}
    for token, value in enumerate(offsets):
        left, right = _span(value, len(text), empty=True)
        for char in range(left, right):
            if char in owner:
                raise ValueError("overlapping tokenizer offsets")
            owner[char] = token
    seen = set()
    result = []
    for quantity, value in enumerate(spans):
        left, right = _span(value, len(text), empty=False)
        chars = set(range(left, right))
        if seen.intersection(chars):
            raise ValueError("overlapping quantity spans")
        seen.update(chars)
        if not chars.issubset(owner):
            raise ValueError("quantity is truncated or absent from token offsets")
        for slot in decimal_slots(text[left:right], start=left):
            result.append(AlignedDigit(quantity, owner[slot.character], slot.digit,
                                       slot.power, slot.negative))
    return tuple(result)
