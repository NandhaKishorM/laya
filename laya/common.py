"""Core model architecture, token sequence construction, and confidence estimation for laya."""
import json
import math
import os
import threading
import warnings
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}

# A fast tokenizer is not read-only: `truncation=True` / `padding=True` make it call
# `enable_truncation` / `enable_padding`, which mutates the shared Rust object. One tokenizer is
# parsed per checkpoint directory and shared by every Agent that wants it, so concurrent
# `predict()` calls -- on one Agent or on two sharing the cache -- raced and raised
# `RuntimeError: Already borrowed`. Serialise encoding instead: it is a small fraction of a call
# next to the forward pass, and this keeps the cache's single parse.
_TOKENIZE_LOCK = threading.RLock()
_QUESTION_TOKEN_CACHE = ContextVar("laya_question_token_cache", default=None)


def encode_text(tok, text, **kwargs):
    """Tokenize `text` while holding the lock a shared fast tokenizer needs."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)


def _reuse_question_tokens(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # One cache per prediction call: nested calls get their own scope, and exceptions
        # restore the outer scope. Nothing is retained on an Agent or shared across threads.
        scope = {"thread": threading.get_ident(), "tokens": {}}
        token = _QUESTION_TOKEN_CACHE.set(scope)
        try:
            return fn(*args, **kwargs)
        finally:
            # A timed hook can copy this context to a worker that outlives the call.
            scope["tokens"] = None
            _QUESTION_TOKEN_CACHE.reset(token)
    return wrapped


def _disable_question_token_reuse():
    scope = _QUESTION_TOKEN_CACHE.get()
    if scope is not None:
        scope["tokens"] = None


def _encode_question_text(tok, text, **kwargs):
    scope = _QUESTION_TOKEN_CACHE.get()
    cache = scope["tokens"] if scope is not None and scope["thread"] == threading.get_ident() else None
    if cache is None:
        return encode_text(tok, text, **kwargs)["input_ids"]
    # Key the rendered, mask-sanitized text and encoding settings, not a JSON question:
    # option order, structured criteria and custom noul labels must keep their meaning.
    key = (id(tok), text, tuple(kwargs.items()))
    if key not in cache:
        # Keep the tokenizer alive so its identity cannot be reused within this scope.
        cache[key] = (tok, tuple(encode_text(tok, text, **kwargs)["input_ids"]))
    # Sequence assembly must not mutate token lists retained for later states.
    return list(cache[key][1])


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


# First estimate of how many characters of serialized state buy one token of sequence budget
# (`encode_state_head`). This is a STARTING POINT, not an upper bound: if the prefix it selects
# comes up short, the next attempt is sized from the density that attempt just measured, and the
# `len(head) >= need` check below is what makes any under-estimate safe. So this wants to be near
# real text, not above all of it.
#
# Measured densities (chars per token):
#
#   English prose, english tokenizer ........  5.3
#   Russian prose, multilingual tokenizer ...  3.8
#   flat JSON dict of short fields ..........  1.5-2.2 either way
#
# (Re-measured 2026-09-28 on the shipped tokenizers with varied prose. An earlier note here
# claimed 7.4 and 16.0; neither reproduced. The lower real densities only make 8 safer.)
#
# 8 therefore clears the budget on the first attempt for all of these, which is the common case,
# and costs a quarter of what a bound of 32 did. Dense text takes one more attempt.
#
# An upper bound is not available anyway: the most source characters one token can stand for is 31
# on the multilingual tokenizer (16 for any token bearing a letter or digit) but 512 on the english
# one -- a 512-space token, a 257-character dash token -- so no multiplier is a proof there. That is
# why the retry and the length check exist rather than a bigger constant.
STATE_HEAD_CHARS_PER_TOKEN = 8

# How many prefix attempts before giving up and tokenizing the whole state. Each attempt costs one
# tokenization of a prefix, so this trades a bounded amount of repeated work for a much smaller
# first guess. Three is measured to be enough: prose and JSON land on attempt 1, the densest real
# text on attempt 2, and pathological states (a 256-character dash run per token) exhaust it and
# take the full tokenization -- correct, just not faster.
_STATE_HEAD_MAX_ATTEMPTS = 3

# How much bigger than the first budget a state must be before a prefix is worth taking at all.
# Below this the "prefix" is nearly the whole state, so the head tokenizes almost the same text and
# the cut scan plus the three soundness gates are pure overhead. Measured just above the budget
# without this margin: 0.94x on english at 1.01x the budget and 0.88x on multilingual -- a real,
# if narrow, regression against simply tokenizing the state. At 1.25x the budget the head is
# already ahead (1.11x english, 1.06x multilingual), so that is where the gate sits.
_STATE_HEAD_MIN_RATIO = 1.25

# Smallest density probe, in characters. An eighth of the budget is used when that is larger. Small
# enough that a wasted probe is noise against the state it is measuring, large enough to contain a
# cut point and several tokens in ordinary text.
_STATE_HEAD_PROBE_CHARS = 512

# Floor for the probe, so it still spans a cut point and a few tokens on a small state. 64
# characters is enough to measure density on every shape checked: it reaches ~6 tokens on a
# 16-character dash run and ~12 on English prose.
_STATE_HEAD_PROBE_MIN = 64

# How many candidate cut points to probe before giving up and tokenizing the whole state. One
# probe answers ordinary text. Each extra one steps back one character, not one run -- inside a
# contiguous space run `str.rfind(" ", 1, i)` returns `i - 1` -- so this tolerates a budget
# landing at most this many characters past a whitespace run's first character. Measured: a
# 100-space run is still cut when the budget sits 63 characters into it and not at 64 -- the first
# probe consumes one of them -- where
# `_state_cut` returns 0 and the state is tokenized in full (correct, just not faster). Ordinary
# indented or padded JSON is far inside that; do not lower it on the belief that one probe skips
# a whole run.
_STATE_CUT_PROBES = 64


def _state_cut(text: str, budget: int) -> int:
    r"""The largest `i <= budget` at which slicing `text` cannot change how it tokenizes.

    A cut is safe when `text[i]` is a plain space and `text[i - 1]` is not whitespace: `text[:i]`
    then tokenizes to a prefix of `text`'s ids. Three things have to hold for that, and all three
    were checked against the shipped tokenizers rather than assumed:

      * No added token can straddle the cut. Both checkpoints carry added tokens that are runs of
        one whitespace character (2-24 spaces on english, 1-31 newlines or tabs on multilingual),
        and the trie matches them greedily before anything else -- which is what makes a cut in
        the middle of a run unsafe, and why `text[i - 1]` must not be whitespace. Every
        space-bearing added token in both checkpoints is a run of a single whitespace character
        (23 of english's 116 added tokens hold a space, and each is 2-24 spaces and nothing
        else), so a token covering `i` would have to begin inside the run -- which the
        `text[i - 1]` test excludes. `_added_tokens_permit_cut` checks that property rather than
        trusting it, because a checkpoint may add tokens of its own.
      * No pre-token can straddle the cut. Metaspace (multilingual) rewrites each space to U+2581
        and splits on it. The ByteLevel regex (english) only ever puts a space first in a piece
        (` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`, `\s+`). Either way the space at `i` opens
        a piece, and BPE merges never cross a piece. `_pre_tokenizer_permits_cut` checks this too,
        because a T5-shaped tokenizer puts Metaspace in the normalizer and splits nowhere.
      * Normalization must not reach across the cut. The english tokenizer normalizes NFC; a
        space is a starter with combining class 0, so it never composes with what precedes it.
        The multilingual normalizer only rewrites single spaces.

    Returns 0 when there is no such point in reach -- a state with no spaces, or one whose only
    spaces sit inside a long run -- and the caller must then tokenize the whole state.
    """
    i = text.rfind(" ", 1, budget + 1)
    for _ in range(_STATE_CUT_PROBES):
        if i < 1:
            break
        if not text[i - 1].isspace():
            return i
        # Inside a whitespace run: the run's own tokenization depends on how long it is, so only
        # its first character is a cut, and that one is what the next probe looks for.
        i = text.rfind(" ", 1, i)
    return 0


# Characters that can carry a cut. U+0020 is the one `_state_cut` slices at; U+2581 is what a
# Metaspace-style normalizer rewrites it to, which is how a sentencepiece checkpoint spells the same
# boundary inside an added token.
_CUT_CHARS = (" ", "\u2581")


def _added_tokens_permit_cut(tok) -> bool:
    r"""Whether `_state_cut`'s rule is sound for the added tokens *this* tokenizer carries.

    The rule cuts at a space whose predecessor is not whitespace, and that is safe only if no
    added token can cover that space. An added token can only do so by holding a space somewhere
    after its first character whose predecessor *inside the token* is not whitespace -- which is
    exactly what this tests, mirroring `_state_cut`'s own `text[i - 1].isspace()`.

    Both shipped checkpoints pass: every space-bearing added token is a run of one whitespace
    character. A checkpoint that adds a domain phrase does not -- `add_tokens(["New York"])`, or a
    sentencepiece user-defined symbol -- and `Agent` loads any directory a training run wrote, so
    this is checked rather than assumed. Without the check such a tokenizer silently returns ids
    that are not a prefix of the full tokenization, which the `len(head) >= need` test below
    cannot catch: the head is long enough, just wrong.

    Deliberately not cached: this costs 53 us on english and 91 us on multilingual against the
    tokenization it guards, and a cache keyed on the tokenizer would go stale the moment a caller
    added a token to it. The head path is only entered for a state well past the budget
    (`_STATE_HEAD_MIN_RATIO`), where that tokenization is milliseconds.
    """
    get = getattr(tok, "get_added_vocab", None)
    if get is None:
        # Cannot establish the property, so do not rely on it -- the same answer the `except`
        # below gives, and the same as the other two gates give for an uninspectable tokenizer.
        return False
    try:
        added = get()
    except Exception:
        # Cannot establish the property, so do not rely on it.
        return False
    for piece in added:
        # Only a piece containing a space can cover a cut, and `in` is a C-level scan, so this
        # skips almost every added token before the character loop below runs: 93 of english's 116
        # and all 249 of multilingual's. Measured, it takes this function from 69 to 53 us on
        # english and 147 to 91 us on multilingual. What is left is `get_added_vocab()` rebuilding
        # its dict (72 us on multilingual), not the scan -- so this is not cached, on the grounds
        # that a cache keyed on the tokenizer goes stale the moment a caller adds a token, and the
        # remaining cost is under 6% of the one tokenization the gate protects.
        #
        # U+2581 counts as a space here, not only U+0020. A sentencepiece-derived checkpoint spells
        # its added tokens against the *normalized* text, where a Metaspace-style normalizer has
        # already rewritten " " to "\u2581" -- so the phrase this gate exists to catch arrives as
        # `\u2581New\u2581York` and contains no U+0020 at all. Checking only U+0020 let exactly
        # that tokenizer through: measured, the gate returned True and the head stopped being a
        # prefix of the full tokenization.
        if not any(c in piece for c in _CUT_CHARS):
            continue
        for k in range(1, len(piece)):
            if piece[k] in _CUT_CHARS and not (piece[k - 1].isspace() or piece[k - 1] in _CUT_CHARS):
                return False
    return True


def _splits_on_whitespace(pt) -> bool:
    """Whether this pre-tokenizer guarantees a space always opens a piece."""
    name = type(pt).__name__
    if name == "ByteLevel":
        # Its regex is what puts a space first in every piece; without it nothing splits.
        return bool(getattr(pt, "use_regex", False))
    if name == "Metaspace":
        # `split=False` rewrites spaces to U+2581 but leaves one piece, so merges cross it.
        return bool(getattr(pt, "split", False))
    if name in ("Whitespace", "WhitespaceSplit", "BertPreTokenizer"):
        return True
    if name == "Sequence":
        try:
            members = list(pt)
        except Exception:
            return False
        return any(_splits_on_whitespace(m) for m in members)
    return False


def _normalizer_permits_cut(tok) -> bool:
    r"""Whether `_state_cut`'s rule survives this tokenizer's *normalization*.

    The third of the three properties `_state_cut` documents, and the last one still assumed. The
    rule cuts at a plain space, so normalization must leave that space as a single boundary
    character the pre-tokenizer can still split on. Both shipped checkpoints do, for different
    reasons: english normalizes NFC and leaves the space alone, multilingual rewrites it to U+2581
    one-for-one, which is exactly what its Metaspace pre-tokenizer then splits on.

    This is a behavioural probe rather than a list of allowed normalizer classes, because the class
    does not decide it: `Replace` is the multilingual checkpoint's own normalizer and is safe, while
    `Replace(Regex(r"\s+"), "")` -- a plausible CJK-oriented finetune -- deletes the cut entirely.
    With the space gone the pre-tokenizer has no boundary there and a single BPE piece spans it, so
    the head stops being a prefix while staying long enough for `len(head) >= need` to wave it
    through. Measured on a tokenizer built that way: divergence at token 0.

    A normalizer that cannot be probed, or one that does anything to the probe other than keep the
    space as one whitespace or U+2581 character in the same position, is treated as unverifiable.
    """
    bt = getattr(tok, "backend_tokenizer", None)
    if bt is None:
        return False
    try:
        normalizer = bt.normalizer
    except Exception:
        return False
    if normalizer is None:
        return True                      # nothing runs, so nothing can move the cut
    try:
        for head, rest in (("ab", " cd"), ("x1", " y2 z3"), ("ąę", " żź")):
            probe = head + rest
            out = normalizer.normalize_str(probe)
            cut_at = len(normalizer.normalize_str(head))
            # The space has to survive, in place, as exactly one character the pre-tokenizer can
            # split on -- and the text before it must still normalize to a prefix of the whole.
            if not out.startswith(normalizer.normalize_str(head)):
                return False
            if cut_at >= len(out):
                return False
            if not (out[cut_at].isspace() or out[cut_at] == "▁"):
                return False
    except Exception:
        return False
    return True


def _pre_tokenizer_permits_cut(tok) -> bool:
    r"""Whether `_state_cut`'s rule survives this tokenizer's *pre-tokenization*.

    The second of the three properties `_state_cut` documents: no pre-token may straddle the cut.
    Both shipped checkpoints give it -- ByteLevel's regex only ever puts a space first in a piece,
    and Metaspace with `split=True` splits on U+2581 -- but it is a property of those tokenizers,
    not of tokenizers in general, and `Agent` loads any directory a training run wrote.

    A T5- or Llama-shaped tokenizer is the case that breaks it: Metaspace sits in the *normalizer*
    as `Prepend` + `Replace`, `pre_tokenizer` is null, so BPE merges run over the whole string and
    the vocabulary holds pieces that span what used to be a space. Cutting inside one changes the
    ids *before* the cut -- measured: divergence at token 0 -- and `len(head) >= need` cannot catch
    it, because the head is long enough, merely wrong.

    A tokenizer whose pre-tokenizer cannot be inspected (a slow, pure-Python one has no
    `backend_tokenizer`) is treated as unverifiable and gets the full tokenization.
    """
    bt = getattr(tok, "backend_tokenizer", None)
    if bt is None:
        return False
    try:
        return _splits_on_whitespace(bt.pre_tokenizer)
    except Exception:
        return False


def encode_state_head(tok, text: str, need: int) -> List[int]:
    """Tokenize `text` far enough to fill `need` tokens of state budget, and no further.

    Returns ids that agree with `encode_text(tok, text)` on their first `need` entries -- either
    the whole state's ids, or a prefix of them. `build_sequence` slices at most `max_len` state
    tokens off the front, so tokenizing the rest of a large document was work whose only use was
    to be thrown away: on the english checkpoint a 50000-character state (the cap `laya.serve`
    enforces) tokenized in 10.30 ms and its first 16384 characters in 3.24 ms.

    Only valid where the slice `build_sequence` takes is a prefix. A conversation list is
    truncated from the left, so `truncate_left=True` callers must keep tokenizing in full -- the
    tokens they keep are at the end of the document, not the start.
    """
    full = lambda: encode_text(tok, text, add_special_tokens=False)["input_ids"]
    if need <= 0 or len(text) <= need * STATE_HEAD_CHARS_PER_TOKEN * _STATE_HEAD_MIN_RATIO:
        return full()

    # Measure the state's density on a SMALL probe before committing to a full-size prefix. A first
    # attempt sized at `need * 8` characters is wasted work when the state turns out denser than
    # that, and the waste is proportional to the attempt: a 256-character dash run per token
    # tokenized 3 854 characters, came up short, and then the whole 6 000-character state anyway --
    # 1.64x the characters for a measured 0.57x slowdown. The probe bounds that waste to its own
    # size, a sixty-fourth of the state at most. Measured, shrinking it also made the wins bigger,
    # because the probe is pure overhead on states that go on to succeed: English prose at 6 000
    # characters went 1.50x -> 1.64x and at 50 000, 11.72x -> 12.15x.
    budget = need * STATE_HEAD_CHARS_PER_TOKEN
    probe_budget = max(_STATE_HEAD_PROBE_MIN, min(_STATE_HEAD_PROBE_CHARS, budget // 8,
                                                 len(text) // 64))
    cut = _state_cut(text, probe_budget)
    if not cut:
        return full()                      # no safe cut point even near the start
    probe = encode_text(tok, text[:cut], add_special_tokens=False)["input_ids"]
    if not probe:
        return full()                      # nothing measurable, so nothing to extrapolate from
    density = cut / len(probe)

    # How much text a prefix covering the budget would need. 1.25x margin because density is not
    # uniform across a document, and a floor of 64 so a tiny budget still moves.
    want = need * density * 1.25 + 64
    # A prefix that large is not worth taking: paying for it and then the whole state would cost
    # more than the state alone. Decided BEFORE the soundness gates, because they cost 53-91 us and
    # a state that will not be cut does not need them answered. A dense state tokenizes few ids from
    # many characters, so its own tokenization is cheap and that fixed cost showed up as a 0.82x
    # regression when the gates ran first.
    if len(probe) < need and want * _STATE_HEAD_MIN_RATIO >= len(text):
        return full()

    # Only now, with a cut in hand and a prefix worth taking, are the three soundness gates asked.
    if not (_added_tokens_permit_cut(tok)
            and _pre_tokenizer_permits_cut(tok)
            and _normalizer_permits_cut(tok)):
        return full()
    if len(probe) >= need:
        return probe                       # the probe alone already covers the budget

    last_cut = cut
    for _ in range(_STATE_HEAD_MAX_ATTEMPTS):
        if want * _STATE_HEAD_MIN_RATIO >= len(text):
            break
        cut = _state_cut(text, int(want))
        # `_state_cut` clamps to the last space at or before `want`, so a larger `want` can select
        # the same cut -- if the text past it holds no further space, every attempt would tokenize
        # an identical prefix. Stop instead of paying for it two more times.
        if not cut or cut <= last_cut:
            break
        last_cut = cut
        head = encode_text(tok, text[:cut], add_special_tokens=False)["input_ids"]
        if len(head) >= need:
            return head
        if not head:
            break
        density = max(density, cut / len(head))
        want = need * density * 1.25 + 64
    return full()


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
    rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return false_label, true_label


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values.
        # `str(k)` unconditionally: a label with no description is rendered as itself, so an int
        # label used to come back as an int from a function annotated `-> List[str]` and then
        # reached `build_sequence`, which calls `.replace` on it and raised an AttributeError
        # naming neither the question nor the label. With a description the same label already
        # went through `"%s: %s" %` and was a str, which is why only the undescribed form broke.
        # `structured._enum_field` stringifies labels the same way; the returned answer still
        # carries the caller's original label, which is unchanged.
        return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": "
        + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": "
        + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    state_ids: Optional[List[int]] = None,
    return_stats: bool = False,
    return_truncation_stats: bool = False,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `state_ids` lets a caller tokenize the shared state once and reuse it across every question,
    instead of re-serializing and re-tokenizing the same document per question.

    `return_stats` adds a third return value describing what the head budget did to the options:
    `options` (how many the question defines), `options_distinct` (how many still have a token
    span of their own) and `tokens_per_option` (the cap applied to each, or None when none was).

    The state is clamped to whatever room is left after the head, so a long state loses tokens
    here silently. `return_truncation_stats=True` adds one more return value, after the option
    stats when both are asked for, reporting that clamp:

        {"state_tokens": int, "state_tokens_used": int, "state_tokens_dropped": int,
         "truncated": bool}

    Callers cannot reconstruct this from the outside. The budget is in tokens, not characters,
    and the room left for the state depends on `max_len`, `head_max_len`, the instruction and
    the rendered options - so it moves per checkpoint and per question. A caller guessing with a
    fixed character threshold is wrong in both directions: it reports truncation that did not
    happen, and stays silent while evidence is being dropped (issue #174).

    The question half is `build_head`; `state_room` reports how much of `max_len` is left for the
    state after it, which is what a caller must size a window against.
    """
    ids, markers, stats = build_head(tok, q, head_max_len, option_order=option_order)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = encode_text(tok, serialize_state(state).replace(tok.mask_token, " "),
                                add_special_tokens=False)["input_ids"]
    # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    extra = ()
    if return_truncation_stats:
        # `room` leaves space for the closing [SEP], so every token in `st` survives the [:max_len] clamp
        extra = ({
            "state_tokens": len(state_ids),
            "state_tokens_used": len(st),
            "state_tokens_dropped": len(state_ids) - len(st),
            "truncated": len(st) < len(state_ids),
        },)
    if not return_stats:
        return (ids, markers) + extra
    return (ids, markers, stats) + extra


def build_head(tok, q: Dict, head_max_len: int = 192, option_order: Optional[List[int]] = None):
    """The question half of a sequence: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]`.

    `build_sequence` appends the state to this and `state_room` measures what is left over for it,
    so neither can disagree with the other about what the head costs.

    Returns `(ids, markers, stats)`: the head token ids, the `[MASK]` position of each option in
    `option_order`, and the stats dict `build_sequence` returns for `return_stats`.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = _encode_question_text(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)
    opt_ids = []
    for i in order:
        # Cap at the tokenizer, not after the fact: `[:48]` still makes the tokenizer process the
        # whole (possibly long) description. truncation=True, max_length=48 keeps the first 48
        # tokens, which is exactly what the previous slice produced.
        opt_tokens = _encode_question_text(
            tok,
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per_option = None
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        per_option = per
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    # Two options that share a prefix can come out of the cut as the same token span: the marker
    # count still matches the option count, so the guard in `Agent._encode_state` passes and
    # nothing downstream can tell that the question lost the ability to name them apart. Counted
    # on the capped option ids, before assembly: re-slicing the finished sequence cannot close
    # the last option's span -- it runs on into the serialized state, which differs per request,
    # so the last option always looks distinguishable however it collided (#538).
    return ids, markers, {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }


def state_room(tok, q: Dict, max_len: int = 512, head_max_len: int = 192) -> int:
    """How many state tokens `q` leaves inside `max_len`, which is what `build_sequence` keeps.

    The head is the question's own -- its instructions plus one `[MASK]`-prefixed span per option --
    so a question with many options leaves less room for the state than one with two, and two
    questions in the same request do not have to leave the same amount. Anything past the return
    value is cut off (the start is kept, or the end for a conversation turn list).
    """
    head, _, _ = build_head(tok, q, head_max_len)
    return max(0, max_len - len(head) - 1)          # -1 for the [SEP] that closes the state


#: How far the DEFAULT window may be cut before `window_budget` says so. A small clamp is ordinary
#: and warning about it would be noise; a large one multiplies the number of forward passes by the
#: same factor, which a caller needs to be told about because nothing in their code implies it. 2x
#: is the point where the scan costs at least twice what the caller would estimate from `max_len`.
_WINDOW_CLAMP_WARN_RATIO = 2

# How much capping the window may multiply the window count before `window_batch_cap` starts
# chunking. Under this, chunking would cost the single-shared-pass property for no real protection.
_WINDOW_BATCH_BLOWUP = 2

def window_budget(tok, questions, max_len: int = 512, head_max_len: int = 192,
                  window: Optional[int] = None, stride: Optional[int] = None):
    """Window size and stride for scanning a state that is longer than one sequence.

    `predict_long` decodes each token window back to text and scores it as an ordinary state, so a
    window wider than the room the questions leave is re-truncated by `build_sequence` on the way
    in: the tail of every window reaches no model, while the reported span says it did. The window
    is therefore capped at `state_room`, and at the *smallest* room of `questions`, because the
    windows are one list of states scored for every question in shared forward passes -- a window
    sized for the roomiest question would be cut short for the tightest one, and the offsets
    reported on its answers would mean something different per question. With no questions there is
    nothing to fit, so the caller's window (or the checkpoint default) stands.

    `questions` are internal question dicts, as `Agent._to_internal` returns them.

    An explicit `window` wider than the room is clamped to it with a `RuntimeWarning`, since a
    scan at the requested size cannot read what it claims to. A `stride` past the effective window
    is refused: the tokens between two windows would be read by neither, which is the failure
    `predict_long` exists to prevent. The default stride keeps its 50% overlap of the *effective*
    window, so a span near a boundary still lands whole inside some window; an explicit stride
    equal to the window still reads every token, with no overlap to catch a span that straddles a
    boundary.

    Returns `(window, stride, room)`.
    """
    rooms = [state_room(tok, q, max_len, head_max_len) for q in questions]
    requested = window if (window and window > 0) else max(64, max_len - head_max_len - 8)
    size = requested
    room = min(rooms) if rooms else size
    if room <= 0:
        raise ValueError(
            "predict_long: the questions' options fill the whole sequence (max_len=%d,"
            " head_max_len=%d), leaving no room for the state; no window can carry any of it."
            " A label set this large is what laya.shortlist.predict_shortlist is for"
            % (max_len, head_max_len))
    if size > room:
        if window and window > 0:
            warnings.warn(
                "laya: predict_long: window=%d is wider than the %d state tokens these questions"
                " leave inside max_len=%d, so every window would be truncated to %d on the way to"
                " the model; scanning with window=%d instead" % (size, room, max_len, room, room),
                RuntimeWarning, stacklevel=3)
        elif size >= room * _WINDOW_CLAMP_WARN_RATIO:
            # The DEFAULT window was cut, and cut hard. Capping it is what stops the tail of every
            # window reaching no model, but it is not free and it must not be silent: the scan now
            # needs about `size / room` times as many windows, each one a full forward pass, and
            # nothing in the caller's code says why. Measured on the English checkpoint with 100
            # four-word options: room 102 of max_len 512, so the question head alone is 409 tokens,
            # the window falls 312 -> 102 and the scan goes 11 windows -> 36, a 3.1x wall-clock
            # increase (1902 ms -> 5858 ms) for the same document.
            #
            # The cost is not the capping, it is the shape of the request: 80% of every sequence is
            # the question, and because the encoder is bidirectional the head cannot be computed
            # once and reused -- its representations depend on the state it is paired with. So the
            # warning names the real remedy rather than only reporting the clamp.
            warnings.warn(
                "laya: predict_long: these questions leave only %d of max_len=%d for the state"
                " (their heads take the rest), so the scan window is capped %d -> %d and roughly"
                " %.1fx as many windows -- each a full forward pass -- are needed to read the"
                " document. Fewer or shorter options, a larger max_len, or"
                " laya.shortlist.predict_shortlist for a large label set will all cost less than"
                " scanning at this width" % (room, max_len, size, room, size / room),
                RuntimeWarning, stacklevel=3)
        size = room
    step = stride if (stride and stride > 0) else max(1, size // 2)
    if step > size:
        if size < requested and stride and stride <= requested:
            # The window the caller asked for was reduced above, and their stride was valid for the
            # window they asked for -- so this is the library's clamp, not their mistake. Reducing
            # the stride to match keeps the no-gap guarantee without making a self-consistent pair
            # of arguments an error. A stride that overshot the *requested* window is still refused
            # below, because that one really is the caller's.
            warnings.warn(
                "laya: predict_long: stride=%d was a 50%% step for the window=%d you asked for, but"
                " the window was reduced to %d to fit the room these questions leave; scanning with"
                " stride=%d instead" % (step, requested, size, max(1, size // 2)),
                RuntimeWarning, stacklevel=3)
            step = max(1, size // 2)
        else:
            raise ValueError(
                "predict_long: stride=%d steps past the %d-token window%s, so %d tokens between"
                " every pair of windows would be read by no window at all; pass stride <= %d"
                % (step, size, " these questions leave room for" if size < requested else "",
                   step - size, size))
    return size, step, room



def window_batch_cap(n_windows: int, window: int, config_budget: int,
                     batch_size: Optional[int] = None) -> Optional[int]:
    """Keep one forward pass no wider than the un-capped scan's would have been -- but only when
    capping the window has multiplied the window count enough to matter.

    Capping the window at the room the questions leave multiplies the number of windows on exactly
    the inputs it targets: measured, a 4 561-token document at 120 options goes from 29 windows of
    312 tokens to roughly 413 of 23. `predict_batch` with `batch_size=None` puts every state in one
    forward pass, so a caller who passed no `batch_size` would go from a 29-row pass to a 413-row one
    at `max_len` width -- a plausible out-of-memory on an input that used to fit.

    Sending every window in one run is also a deliberate property (`ONNXAgent` asserts it), and a mild
    cap -- a 64-token budget reduced to 43, say -- multiplies the count by well under two. Chunking
    those would trade a real property for no real protection. So the cap only applies past
    `_WINDOW_BATCH_BLOWUP`x, and then bounds the pass at the count the un-capped budget would have
    produced: peak memory stays at parity with the behaviour before the cap, and the scan still reads
    the whole document, just in more passes. An explicit `batch_size` is always honoured.
    """
    if batch_size and batch_size > 0:
        return batch_size
    if window >= config_budget:
        return None                       # not capped: one pass, exactly as before
    unclamped = max(1, -(-n_windows * max(1, window // 2) // max(1, config_budget // 2)))
    if n_windows <= unclamped * _WINDOW_BATCH_BLOWUP:
        return None                       # a mild cap: keep the single shared pass
    return unclamped

def collapsed_options(qids, items) -> Dict[str, Dict[str, Optional[int]]]:
    """The questions whose options no longer have a token span each, from per-item stats.

    `total` is the number of options the question defines, not the number of markers that
    reached the sequence: a report counted from the markers would say "43/58" about a request
    where 28 options never made it into the input at all.
    """
    out = {}
    for qid, item in zip(qids, items):
        stats = item.get("options")
        if stats and stats["options_distinct"] < stats["options"]:
            out[qid] = {"total": stats["options"], "distinct": stats["options_distinct"],
                        "tokens_per_option": stats["tokens_per_option"]}
    return out


class _DynamicMultiheadAttention(nn.MultiheadAttention):
    """`nn.MultiheadAttention` that keeps the shapes it traces.

    The stock module reshapes the packed projection with sizes captured while tracing, so under
    the legacy ONNX exporter the traced sequence length becomes a constant and a model exported
    from a short dummy input only runs at that length. The parameters and the maths are the same
    here; the reshape uses only constant shape arguments (`chunk` / `unflatten` / `flatten`) and
    `scaled_dot_product_attention`, the kernel the stock path already uses when weights are not
    requested.
    """

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if (need_weights or self.in_proj_weight is None or self.bias_k is not None
                or self.bias_v is not None
                or (attn_mask is not None and attn_mask.dtype != torch.bool)):
            # Weight averaging, additive masks and the optional k/v bias are not on the traced
            # path; the stock implementation keeps them correct.
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                                   need_weights=need_weights, attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights, is_causal=is_causal)
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        # (T, B, E) from here, matching the stock module's internals; attention runs on (B, H, T, D).
        if query is key is value:
            q, k, v = (part.unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                       for part in F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1))
        else:
            embed_dim = query.shape[-1]
            wq, wk, wv = self.in_proj_weight.split(embed_dim, dim=0)
            bq, bk, bv = ((None, None, None) if self.in_proj_bias is None
                          else self.in_proj_bias.split(embed_dim, dim=0))
            q, k, v = (
                F.linear(t, w, b).unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                for t, w, b in ((query, wq, bq), (key, wk, bk), (value, wv, bv))
            )
        mask = None
        if attn_mask is not None:
            mask = ~attn_mask
        if key_padding_mask is not None:
            # `== 0` keeps this correct for a bool mask and for the 0 / -inf float mask the encoder
            # layer hands over (`F._canonical_mask`), where `~` would not be defined.
            keep = key_padding_mask[:, None, None, :] == 0
            mask = keep if mask is None else mask & keep
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal and mask is None,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = attn.permute(2, 0, 1, 3).flatten(-2)
        if self.batch_first:
            attn = attn.transpose(0, 1)
        return self.out_proj(attn), None


def unpermute_probs(p: np.ndarray, option_order: Optional[List[int]]) -> np.ndarray:
    """Put a slot-ordered probability row back into the caller's option order.

    `build_sequence` puts option `option_order[s]` in slot `s`, so a model row comes back
    indexed by slot. Everything downstream indexes by option -- `zip(keys, p)` for a choice,
    `arange(k) * p` for a score level, `p[1]` for noul-true -- so the row has to be inverted
    first or the probabilities end up attached to the wrong options, which is silent.

    A missing or mismatched order returns `p` untouched, so the canonical path is unaffected.
    """
    if option_order is None or len(option_order) != len(p):
        return p
    canonical = np.empty_like(p)
    canonical[np.asarray(option_order, dtype=int)] = p
    return canonical


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1,
                 no_init: bool = False):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        # `no_init` means the caller is about to load every parameter from a checkpoint, so the
        # head's initial values are pure overhead -- and not just time. transformers'
        # `no_init_weights()` patches the torch.nn.init functions, but something inside
        # nn.TransformerEncoderLayer draws from the RNG outside them, so building it still
        # advanced the global generator and made `load()` a visible side effect. On the meta
        # device no initialisation kernel runs at all; the layers are materialised empty and the
        # load fills them.
        with torch.device("meta") if no_init else nullcontext():
            nhead = max(1, d // 64)
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
            # The stock attention bakes the traced length into an exported graph; see
            # _DynamicMultiheadAttention. Same parameters, same maths, traceable shapes.
            layer.self_attn = _DynamicMultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))
        if no_init:
            # Only the modules created above are on the meta device; the encoder is already real
            # and may hold non-persistent buffers (RoPE frequencies) that to_empty would wipe.
            for module in (self.head, self.type_emb, self.scorer, self.act_head):
                if module is not None:
                    module.to_empty(device="cpu")
            self.temperature = torch.empty_like(self.temperature, device="cpu")
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    # Non-reentrant checkpointing also trains the head when its input
                    # is frozen. Default RNG preservation keeps dropout consistent.
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...)
            # has nothing to select for the second slot and raises. The answer
            # is still well-defined: softmax over one logit is 1.0 regardless of
            # its value, so pad the missing second entry with 0.0 - that gives
            # the act head top1 - top2 == 1.0, the same "fully decided" signal
            # it would see for any other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def _apply_rope_config(ecfg) -> None:
    """Carry transformers>=5 per-layer RoPE settings over to the attributes 4.x reads.

    A checkpoint re-saved by transformers 5 stores RoPE as
    `rope_parameters = {"full_attention": {"rope_theta": ...}, "sliding_attention": {...}}`.
    transformers 4.x does not know that key, so it keeps its own defaults (global 160000,
    local 10000) and any checkpoint whose sliding-attention theta differs silently runs the
    wrong RoPE base -- mmBERT is exactly that case, both of its thetas are 160000. Map the
    values onto `global_rope_theta` / `local_rope_theta`, which 4.x does read. On
    transformers 5 this is a no-op beyond re-setting the same numbers.
    """
    rope = getattr(ecfg, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention", "global_rope_theta"),
                             ("sliding_attention", "local_rope_theta")):
        params = rope.get(layer_type)
        theta = params.get("rope_theta") if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def _no_init_weights():
    """`no_init_weights` lives in different modules across transformers versions."""
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights
    return no_init_weights()


def build_model(cfg: Dict, encoder_dir: Optional[str] = None, pretrained: bool = True,
                revision: Optional[str] = None) -> DecisionModel:
    """Build the decision model described by `cfg`.

    With `pretrained=False`, or when `encoder_dir` holds a saved encoder config, nothing is
    downloaded and **no parameter is initialised**: the caller is expected to load a checkpoint
    into the result with `load_state_dict(..., strict=True)` immediately. Skipping initialisation
    keeps `load()` from spending time on, or consuming RNG for, weights it is about to overwrite.
    """
    from transformers import AutoConfig, AutoModel

    head_layers, n_act = cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1
    if not pretrained or (encoder_dir and os.path.exists(encoder_dir)):
        ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])
        _apply_rope_config(ecfg)
        with _no_init_weights():
            enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        return DecisionModel(enc, head_layers, n_act, no_init=True)
    # Training-time Hub load of the base encoder; allow pinning it like the checkpoints.
    kw = {"attn_implementation": "sdpa"}
    if revision:
        kw["revision"] = revision
    enc = AutoModel.from_pretrained(cfg["encoder"], **kw)
    return DecisionModel(enc, head_layers, n_act)


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Probability mass on the answer being reported: max(p).

    This is the quantity temperature scaling fits, and the quantity every calibration figure in
    this repository is computed on -- both benchmark harnesses take `conf = max(probs)` before
    calling `ece_score`. The README's gating section relies on the property that goes with it:
    of the answers returned at confidence c, about c of them are right. That property is
    conditional, and the condition is not met by default -- it holds only after the temperatures
    have been fitted and validated on held-out data for this checkpoint and this option count.
    The shipped checkpoints are over-confident: `choice:11+` is a ~10x sharpener that returns a
    point mass at 1.0, so a threshold applied to them selects below model accuracy (issue #394).

    `confidence_from_probs` below reports a different quantity on a different scale and carries
    no such guarantee, so the two must not be compared against the same threshold.
    """
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k).

    How concentrated the whole distribution is. Useful, but not calibrated: it is not what
    temperature scaling fits and not what the reported ECE measures. See `answer_confidence`.
    """
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published as
# 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest calibration
# needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number.

    A bool is not a number either: `True`/`False` used to float to 1.0/0.0 here and read as
    fitted/sharpening temperatures, the same class of quiet acceptance `check_min_confidence`
    already refuses. A custom `lo`/`hi` still bounds either way.
    """
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):    # NaN / inf
        return 1.0
    return min(hi, max(lo, t))


def resolve_lang_temperatures(raw: Optional[Dict[str, Any]],
                              base_temperature: Sequence[float]) -> Dict[str, Dict[str, Any]]:
    """Parse the `lang_temperatures` option into `{language: {temperature, temperature_by_options}}`.

    One implementation, because `Agent` and `ONNXAgent` both accept this option and both promise
    the same confidences for it. Reading it with `cfg.get(...)` and `len(...)` before checking the
    shape of either raised `AttributeError` and `TypeError` for exactly the inputs the
    `ValueError` below is written for, after the whole checkpoint had loaded:

        {"de": {"temperature": 2}}       -> TypeError: object of type 'int' has no len()
        {"de": {"temperature": None}}    -> TypeError: object of type 'NoneType' has no len()
        {"de": None}                     -> AttributeError: 'NoneType' object has no attribute 'get'

    A `null` entry or a `null` temperature both mean "inherit the checkpoint's own", which is how
    the `laya-ts` port reads the same option (`agent.ts:322-327`).
    """
    resolved: Dict[str, Dict[str, Any]] = {}
    for lang, cfg in (raw or {}).items():
        if not isinstance(lang, str):
            raise ValueError("Language override keys must be strings, got %r" % (lang,))
        norm = lang.split("-")[0].lower()
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ValueError("Language override %r must be a mapping, got %s"
                             % (lang, type(cfg).__name__))
        t_raw = cfg.get("temperature")
        if t_raw is None:
            t_raw = base_temperature
        if not isinstance(t_raw, (list, tuple)) or len(t_raw) != 3:
            raise ValueError("Language override %r temperature must be a list of 3 floats, got %r"
                             % (lang, t_raw))
        tbo_raw = cfg.get("temperature_by_options") or {}
        if not isinstance(tbo_raw, dict):
            raise ValueError("Language override %r temperature_by_options must be a mapping of "
                             "bucket -> float, got %s" % (lang, type(tbo_raw).__name__))
        resolved[norm] = {
            "temperature": [clamp_temperature(t) for t in t_raw],
            "temperature_by_options": {k: clamp_temperature(v) for k, v in tbo_raw.items()},
        }
    return resolved


def amp_dtype(name: Optional[str]) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    has_target = any("target" in it for it in items)
    target = torch.zeros((n, kmax), dtype=torch.float32) if has_target else None

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            if len(it["target"]) > k:
                # Otherwise this lands as "The expanded size of the tensor (k) must match the
                # existing size (kmax)" from inside the assignment, which says nothing about the
                # actual mistake: a target with more entries than the item has options. The limit
                # is this item's own marker count, not the batch-wide kmax: in a mixed-width
                # batch a longer sibling row must not legitimise extra entries (#311).
                raise ValueError(
                    "collate_items: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (i, len(it["target"]), k))
            target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target")} for it in items],
    }
    if target is not None:
        res["target"] = target
    return res
