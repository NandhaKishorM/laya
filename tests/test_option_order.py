"""`option_order` must reach the encoder and be undone on the way back.

`build_sequence` has always taken an `option_order` -- slot `s` shows option `order[s]` -- and
training used it to shuffle. Nothing in the inference path ever passed it, so a question dict
carrying one was silently ignored: `_check_question` does not reject unknown keys and
`_to_internal` dropped it, so every call showed the canonical order.

That matters because where an option sits changes the answer. `research/results/
presentation_checks_shipped.json` measures the English checkpoint on five options whose text is
*identical*: the per-slot logits centre at [+1.68, +1.47, +0.11, -1.35, -1.91], a 3.59 spread
decided by position alone. `research/eval/metamorphic.py` and the `option_order_flip_rate` in
`research/scripts/bench_apps.py` exist for the same reason. Until now a caller had no way to
present the options any differently and see what changed.

The half that is easy to get wrong is the way back. The logit row comes out in *slot* order,
while everything downstream indexes by the caller's option order -- `zip(keys, p)` for a
choice, `arange(k) * p` for a score, `p[1]` for noul-true. Without the inverse permutation the
probabilities come back attached to the wrong labels, which is silent and worse than ignoring
the argument. These tests pin both halves.

No weights are loaded: the encoder is driven with a stub tokenizer and the decoder through
`Agent.__new__`, as in tests/test_batch.py.

Run: python tests/test_option_order.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent, _option_count  # noqa: E402
from laya.common import build_sequence  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
        return
    except Exception as e:                       # noqa: BLE001 - the point is the wrong type
        FAIL.append("%s: raised %s" % (name, type(e).__name__))
        return
    FAIL.append("%s: did not raise" % name)


class StubTok:
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id, pad_token_id = 3, 1, 2, 0

    def __init__(self):
        self._vocab = {}

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [self._vocab.setdefault(w, len(self._vocab) + 10) for w in str(text).split()]
        return {"input_ids": ids[:max_length] if (truncation and max_length) else ids}


# ------------------------------------------------------------------ option count
check("count/choice dict", _option_count({"type": "choice", "criteria": {"a": None, "b": None}}), 2)
check("count/choice list", _option_count({"type": "choice", "criteria": ["a", "b", "c"]}), 3)
check("count/score levels", _option_count({"type": "score", "criteria": ["lo", "mid", "hi"]}), 3)
check("count/noul is always two", _option_count({"type": "noul"}), 2)

# ------------------------------------------------------------------ validation
def _q(**extra):
    q = {"type": "choice", "instructions": "which team",
         "criteria": {"refund": "money", "tech": "a bug", "sales": "pricing"}}
    q.update(extra)
    return q


Agent._check_question("q", _q())                                   # no option_order is fine
Agent._check_question("q", _q(option_order=[2, 0, 1]))             # a permutation is fine
for bad, why in (([0, 1], "too short"), ([0, 1, 2, 0], "too long"), ([0, 1, 1], "repeats"),
                 ([0, 1, 3], "out of range"), ("abc", "not a list"), ([0, 1, True], "bool")):
    check_raises("validate/rejects %s" % why, ValueError,
                 lambda b=bad: Agent._check_question("q", _q(option_order=b)))

# ------------------------------------------------------------------ carried into the internal form
check("internal/absent stays absent", "option_order" in Agent._to_internal(_q()), False)
check("internal/carried through", Agent._to_internal(_q(option_order=[2, 0, 1]))["option_order"], [2, 0, 1])

# ------------------------------------------------------------------ reaches the encoder
tok = StubTok()
qi = {"t": "choice", "ins": "which team",
      "crit": {"refund": "money back", "tech": "a bug", "sales": "pricing questions"}}
canonical, m_canon = build_sequence(tok, {"x": "hello"}, qi, 512, 192)
reversed_, m_rev = build_sequence(tok, {"x": "hello"}, qi, 512, 192, option_order=[2, 1, 0])
check_true("encode/a different order encodes differently", canonical != reversed_)
check("encode/same option count either way", (len(m_canon), len(m_rev)), (3, 3))
check_true("encode/identity order matches no order at all",
           build_sequence(tok, {"x": "hello"}, qi, 512, 192, option_order=[0, 1, 2])[0] == canonical)

# ------------------------------------------------------------------ undone on the way back
# One logit row, deliberately lopsided, decoded under several presentations. Whatever order the
# slots were in, the probabilities must come back keyed in the caller's own option order.
decoder = Agent.__new__(Agent)
decoder.temperature = {0: 1.0, 1: 1.0, 2: 1.0}
decoder.temperature_by_options = {}
decoder.lang_temperatures = {}

CHOICE = {"t": "choice", "crit": {"a": None, "b": None, "c": None}}
SLOT_P = [0.7, 0.2, 0.1]                       # slot 0 wins, by construction
logits = np.log([SLOT_P])
act = np.array([[0.25, 0.75]])
items = [{"markers": [0, 1, 2]}]

for order, expect_winner in (([0, 1, 2], "a"), ([2, 1, 0], "c"), ([1, 2, 0], "b")):
    q = dict(CHOICE, option_order=order)
    ans = decoder._decode_answers(logits, act, items, ["q"], {"q": q}, 0)["q"]
    check("decode/keys stay in the caller's order %s" % (order,), list(ans["probabilities"]), ["a", "b", "c"])
    # slot 0 held option `order[0]`, so that option must carry slot 0's mass
    check("decode/slot 0 mass lands on option %s" % expect_winner, ans["choice"], expect_winner)
    check("decode/mass follows the slot for %s" % (order,),
          round(ans["probabilities"][expect_winner], 4), round(SLOT_P[0], 4))
    check("decode/probabilities still sum to one %s" % (order,),
          round(sum(ans["probabilities"].values()), 4), 1.0)

no_order = decoder._decode_answers(logits, act, items, ["q"], {"q": dict(CHOICE)}, 0)["q"]
identity = decoder._decode_answers(logits, act, items, ["q"], {"q": dict(CHOICE, option_order=[0, 1, 2])}, 0)["q"]
check("decode/identity order changes nothing", identity["probabilities"], no_order["probabilities"])

# A score level is an ordinal index, so the expected value has to be computed after un-permuting.
SCORE = {"t": "score", "crit": ["low", "mid", "high"]}
plain = decoder._decode_answers(logits, act, items, ["s"], {"s": dict(SCORE)}, 0)["s"]
flipped = decoder._decode_answers(logits, act, items, ["s"], {"s": dict(SCORE, option_order=[2, 1, 0])}, 0)["s"]
check("decode/score keeps its level legend", list(flipped["probabilities"]), ["0", "1", "2"])
check("decode/score mass moves to the level that held slot 0",
      round(flipped["probabilities"]["2"], 4), round(SLOT_P[0], 4))
check_true("decode/score expected value follows the levels, not the slots",
           flipped["score"] > plain["score"], "%.4f vs %.4f" % (flipped["score"], plain["score"]))

# noul: `p[1]` is P(true) in option order, so a flipped presentation must not invert the answer.
NOUL = {"t": "noul", "crit": None}
n_items = [{"markers": [0, 1]}]
n_logits = np.log([[0.7, 0.3]])
straight = decoder._decode_answers(n_logits, act, n_items, ["n"], {"n": dict(NOUL)}, 0)["n"]
swapped = decoder._decode_answers(n_logits, act, n_items, ["n"], {"n": dict(NOUL, option_order=[1, 0])}, 0)["n"]
check("decode/noul canonical P(true)", round(straight["noul"], 4), 0.3)
check("decode/noul swapped slots report P(true), not P(slot 1)", round(swapped["noul"], 4), 0.7)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all option-order tests passed")
sys.exit(1 if FAIL else 0)
