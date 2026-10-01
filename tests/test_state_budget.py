"""Regression: the state budget is sized from the head actually built, not from `head_max_len`.

`head_max_len` is a cap. `build_sequence` fills the option prompt budget only as far as the
question and its option descriptions need, then sizes the state as `max_len - head_len - 1`
from the head it actually built.

`max_len - head_max_len` is wrong in *both* directions, which is what the README now says:

* a question that does not fill the cap -> the cap-based figure **understates** the room
  (conservative, but it discards state the model would have read);
* a high-cardinality question -> it **overstates** it. `per = max(4, (head_max_len - 16) // k)`
  floors at 4 tokens per option and `head_ids` keeps a floor of 8, so past roughly
  `head_max_len / 4` options the head grows beyond the cap, and a state sized to the cap-based
  figure is **truncated** rather than merely conservative.

This suite pins the relationship, and — when the checkpoints are present — reads the figures
back out of `README.md` and checks them against a live measurement, so the renderer, the
documentation and this file cannot drift apart silently.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import build_sequence  # noqa: E402

PASS, FAIL, SKIPPED = [], [], []

# The same 4-option question the README's token-budget section walks through.
Q = {"t": "choice", "ins": "Which department should handle this request?",
     "crit": {"billing": "invoices, payments, refunds",
              "technical": "bugs, outages, system errors",
              "sales": "pricing, new contracts, plan upgrades",
              "other": "everything else"}}


class _Skip(Exception):
    pass


def skip(reason):
    """A skip that is visible in both runners.

    Under pytest this raises `pytest.skip`, so `-rs` reports it. As a plain script it raises
    `_Skip`, which the runner prints and counts -- a bare `return` passes vacuously and no
    runner flag reveals it.
    """
    print("  SKIP: " + reason)
    SKIPPED.append(reason)
    if __name__ != "__main__":
        import pytest
        pytest.skip(reason)
    raise _Skip(reason)


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


class _FakeTok:
    """Deterministic stand-in: one id per word, so head length is predictable."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def id_of(self, word):
        return self(" " + word)["input_ids"][0]


def _head_len(tok, q, head_max_len):
    """Length of the built prompt the state is appended to, excluding the trailing [SEP]."""
    seq, _ = build_sequence(tok, "", q, max_len=10 ** 6, head_max_len=head_max_len)
    return len(seq) - 1


def _kept_state(tok, state, q, max_len, head_max_len):
    head = _head_len(tok, q, head_max_len)
    seq, _ = build_sequence(tok, state, q, max_len=max_len, head_max_len=head_max_len)
    return seq[head:-1]


def _many_options(k, desc="a short description of this option"):
    return {"t": "choice", "ins": "Which intent is this?",
            "crit": {"intent_%d" % i: desc for i in range(k)}}


def _models_dir():
    return os.environ.get("LAYA_LAB_MODELS", os.path.expanduser("~/laya_models"))


def _real_tok(name):
    """The checkpoint's tokenizer, or None. Callers decide whether absence is a skip."""
    path = os.path.join(_models_dir(), name, "tokenizer")
    if not os.path.isdir(path):
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


# --------------------------------------------------------------------------- invariant checks

def test_state_budget_is_sized_from_the_actual_head():
    """A state longer than the room is cut to exactly `max_len - head_len - 1`."""
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    head = _head_len(tok, Q, head_max_len)
    assert head < head_max_len, "fixture assumes a question that does not fill the cap"

    room = max_len - head - 1
    state = " ".join("w%d" % i for i in range(room + 200))
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    check("kept state length is max_len - head_len - 1", len(kept), room)


def test_dropping_the_minus_one_is_caught():
    """The `-1` is load-bearing: one token more than the room must lose its tail."""
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    head = _head_len(tok, Q, head_max_len)
    marker = "TAILMARKER"
    exact = max_len - head - 1                    # the true room

    state = " ".join("w%d" % i for i in range(exact - 1)) + " " + marker
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    check("a state of exactly the room is kept whole", len(kept), exact)
    assert tok.id_of(marker) in kept, "the tail marker must survive an exactly-sized state"

    over = " ".join("w%d" % i for i in range(exact)) + " " + marker
    kept_over = _kept_state(tok, over, Q, max_len, head_max_len)
    assert tok.id_of(marker) not in kept_over, (
        "a state one token over the room must lose its tail, so the room is max_len-head_len-1 "
        "and not max_len-head_len")


def test_cap_based_figure_understates_a_short_question():
    """A state of exactly `max_len - head_max_len` tokens survives for a short question."""
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    documented = max_len - head_max_len
    head = _head_len(tok, Q, head_max_len)
    assert max_len - head - 1 > documented, "fixture assumes the cap-based figure understates"

    marker = "UNIQUETAILMARKER"
    state = " ".join("w%d" % i for i in range(documented - 3)) + " " + marker
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    assert tok.id_of(marker) in kept, (
        "a %d-token state must survive a %d budget for this question" % (documented, max_len))


def test_cap_based_figure_overstates_a_high_cardinality_question():
    """Past ~`head_max_len / 4` options the head exceeds the cap, so the cap-based figure lies.

    This is the direction that matters: a caller sizing a state to `max_len - head_max_len`
    is truncated, not merely conservative.
    """
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    head = _head_len(tok, _many_options(100), head_max_len)
    assert head > head_max_len, (
        "a 100-option question must build a head beyond the %d cap, got %d" % (head_max_len, head))
    check("cap-based figure overstates the room at k=100",
          max_len - head_max_len > max_len - head - 1, True)


# --------------------------------------------------------------------- documentation agreement

def _readme_quotes():
    """The figures README.md states, parsed out of the token-budget bullet."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    heads = {int(m) for m in re.findall(r"\*\*(\d+)-token\*\* head", text)}
    rooms = {int(m) for m in re.findall(r"\*\*(\d+) tokens for state\*\*", text)}
    return heads, rooms


def test_readme_quotes_the_measured_figures():
    """README's head/room figures equal a live measurement, for each shipped checkpoint.

    This is what makes the suite stop the README and the renderer desynchronising; without it
    the file only guards the renderer.
    """
    heads, rooms = _readme_quotes()
    assert heads and rooms, "could not parse the head/room figures out of README.md"

    checked = 0
    for name, max_len, head_max_len in (("english", 512, 192), ("multilingual", 1024, 256)):
        try:
            tok = _real_tok(name)
        except Exception as e:                      # transformers absent or unloadable
            skip("%s: tokenizer unavailable (%s)" % (name, e))
            continue
        if tok is None:
            skip("%s: no tokenizer under %s" % (name, _models_dir()))
            continue
        head = _head_len(tok, Q, head_max_len)
        room = max_len - head - 1
        assert head in heads, (
            "README does not quote a %d-token head for %s (quotes %s)" % (head, name, sorted(heads)))
        assert room in rooms, (
            "README does not quote a %d-token state room for %s (quotes %s)"
            % (room, name, sorted(rooms)))
        checked += 1
    if not checked:
        skip("no checkpoints under %s" % _models_dir())


if __name__ == "__main__":
    for fn in (test_state_budget_is_sized_from_the_actual_head,
               test_dropping_the_minus_one_is_caught,
               test_cap_based_figure_understates_a_short_question,
               test_cap_based_figure_overstates_a_high_cardinality_question,
               test_readme_quotes_the_measured_figures):
        try:
            fn()
        except AssertionError as e:
            FAIL.append("%s: %s" % (fn.__name__, e))
        except _Skip:
            pass
        else:
            PASS.append(fn.__name__)

    print("\n%d passed, %d failed, %d skipped" % (len(PASS), len(FAIL), len(SKIPPED)))
    for f in FAIL:
        print("  FAIL " + f)
    for s in SKIPPED:
        print("  SKIP " + s)
    sys.exit(1 if FAIL else 0)
