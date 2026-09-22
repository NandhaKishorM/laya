"""State truncation must be reported from the token budget that actually applies (issue #174).

`build_sequence` clamps the state to whatever room the head leaves, and before this the clamp was
silent: `system_one` returned a token count and nothing about what it had dropped. A caller could
only guess, and the guess reported in #174 - a fixed character threshold - is wrong in both
directions, because the budget is in tokens and moves with `max_len`, `head_max_len` and the
rendered head of each question.

These use a stub tokenizer (one token per whitespace word) so they run offline and the expected
token counts are exact, the same approach as tests/test_shortlist.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import build_sequence  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


class StubTok:
    """One token per whitespace-separated word, so token counts are exact and readable."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [10 + (len(w) % 50) for w in text.split()]}


TOK = StubTok()
Q = {"t": "noul", "ins": "Does the user ask for a refund?", "crit": None}


def stats_for(state, max_len=512, head_max_len=192, truncate_left=False, q=Q):
    _, _, st = build_sequence(
        TOK, state, q, max_len, head_max_len, truncate_left=truncate_left, return_stats=True)
    return st


# --------------------------------------------------------------- the reported gap: a silent clamp
# A state that fits loses nothing, and says so.
short = stats_for("the customer was billed twice")
check("short/not truncated", short["truncated"], False)
check("short/nothing dropped", short["state_tokens_dropped"], 0)
check("short/all tokens used", short["state_tokens_used"], short["state_tokens"])

# A state that does not fit reports the exact shortfall rather than staying silent.
long_state = " ".join("word%d" % i for i in range(4000))
clamped = stats_for(long_state, max_len=512)
check("long/truncated", clamped["truncated"], True)
check("long/counts the whole state", clamped["state_tokens"], 4000)
check_true("long/used less than sent", clamped["state_tokens_used"] < 4000)
check("long/dropped is the remainder",
      clamped["state_tokens_dropped"], 4000 - clamped["state_tokens_used"])
check_true("long/used fits the window", clamped["state_tokens_used"] <= 512)


# --------------------------------------------------------------- the flag tracks the real window
# #174: the same state was reported identically on a 512-token and a 1024-token checkpoint, while
# the actual clamp differed by more than 2x. The reported numbers must move with the window.
w512 = stats_for(long_state, max_len=512)
w1024 = stats_for(long_state, max_len=1024)
check_true("window/1024 keeps strictly more than 512",
           w1024["state_tokens_used"] > w512["state_tokens_used"],
           "(512 -> %d, 1024 -> %d)" % (w512["state_tokens_used"], w1024["state_tokens_used"]))
check_true("window/1024 drops strictly less than 512",
           w1024["state_tokens_dropped"] < w512["state_tokens_dropped"])

# A state sized to fit the larger window only: truncated on one checkpoint, intact on the other.
mid = " ".join("word%d" % i for i in range(600))
check("window/512 truncates the mid state", stats_for(mid, max_len=512)["truncated"], True)
check("window/1024 does not", stats_for(mid, max_len=1024)["truncated"], False)

# The head is part of the budget too, so two questions over one state can disagree. A longer
# instruction and richer options leave less room for the state.
wide = {
    "t": "choice",
    "ins": "Which team should own this request, given the account tier and the contract terms?",
    "crit": {"billing": "invoices, payments, refunds, chargebacks and duplicate charges",
             "technical": "bugs, outages, degraded performance and system errors",
             "sales": "pricing, quotes, renewals and new contracts",
             "other": "everything that does not belong to the three above"},
}
narrow = {"t": "noul", "ins": "Refund?", "crit": None}
check_true("head/a wider head leaves less room for the state",
           stats_for(long_state, q=wide)["state_tokens_used"]
           < stats_for(long_state, q=narrow)["state_tokens_used"])


# --------------------------------------------------------------- left truncation
left = stats_for(long_state, max_len=512, truncate_left=True)
check("left/truncated", left["truncated"], True)
check_true("left/keeps as much as the right-hand clamp",
           left["state_tokens_used"] == w512["state_tokens_used"])

# With no room at all, `st[-0:]` is the whole state and only the final [:max_len] clamp drops it.
# Counting `kept` would report nothing dropped while every state token was in fact discarded.
no_room = stats_for(long_state, max_len=8, head_max_len=8, truncate_left=True)
check("no-room/reports the drop", no_room["truncated"], True)
check_true("no-room/never claims more than the window",
           no_room["state_tokens_used"] <= 8,
           "(used %d)" % no_room["state_tokens_used"])


# --------------------------------------------------------------- reported counts match the sequence
# The accounting must describe the sequence that is actually returned, not an estimate beside it.
for name, ml, hml in [("512", 512, 192), ("1024", 1024, 512), ("tight", 64, 32)]:
    seq, markers, st = build_sequence(TOK, long_state, Q, ml, hml, return_stats=True)
    check_true("accounting/%s sequence fits max_len" % name, len(seq) <= ml,
               "(len %d > %d)" % (len(seq), ml))
    check_true("accounting/%s used tokens fit the sequence" % name, st["state_tokens_used"] <= len(seq))
    check("accounting/%s used + dropped == sent" % name,
          st["state_tokens_used"] + st["state_tokens_dropped"], st["state_tokens"])
    check("accounting/%s truncated agrees with the count" % name,
          st["truncated"], st["state_tokens_dropped"] > 0)


# --------------------------------------------------------------- the stats are opt-in
# Existing callers unpack two values (research/scripts/bench_local.py, the fine-tuning notebook).
default = build_sequence(TOK, "a short state", Q)
check("compat/default returns a pair", len(default), 2)
seq_a, mk_a = build_sequence(TOK, long_state, Q, 512, 192)
seq_b, mk_b, _ = build_sequence(TOK, long_state, Q, 512, 192, return_stats=True)
check("compat/return_stats does not change the sequence", seq_a, seq_b)
check("compat/return_stats does not change the markers", mk_a, mk_b)


# --------------------------------------------------------------- system_one reports it in `usage`
# Checked on the source: exercising it for real needs a checkpoint on disk (tests/test_local_e2e.py).
import inspect  # noqa: E402

from laya import agent as _agent  # noqa: E402

_src = inspect.getsource(_agent.Agent.system_one)
check_true("usage/asks build_sequence for the stats", "return_stats=True" in _src)
check_true("usage/publishes the flag", '"truncated": dropped > 0' in _src)
check_true("usage/publishes the dropped count", '"state_tokens_dropped": dropped' in _src)
check_true("usage/names the questions that truncated", '"truncated_questions"' in _src)
check_true("usage/keeps the existing token counts",
           '"input_tokens": n_tokens' in _src and '"output_tokens": 0' in _src)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
