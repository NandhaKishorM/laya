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

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [10 + (len(w) % 50) for w in text.split()]
        # build_sequence caps each option at the tokenizer (truncation=True, max_length=48)
        return {"input_ids": ids[:max_length] if truncation and max_length is not None else ids}


TOK = StubTok()
Q = {"t": "noul", "ins": "Does the user ask for a refund?", "crit": None}


def stats_for(state, max_len=512, head_max_len=192, truncate_left=False, q=Q):
    _, _, st = build_sequence(
        TOK, state, q, max_len, head_max_len, truncate_left=truncate_left, return_truncation_stats=True)
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
    seq, markers, st = build_sequence(TOK, long_state, Q, ml, hml, return_truncation_stats=True)
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
seq_b, mk_b, _ = build_sequence(TOK, long_state, Q, 512, 192, return_truncation_stats=True)
check("compat/return_truncation_stats does not change the sequence", seq_a, seq_b)
check("compat/return_truncation_stats does not change the markers", mk_a, mk_b)


# --------------------------------------------------------------- system_one reports it in `usage`
# Checked on the source: exercising it for real needs a checkpoint on disk (tests/test_local_e2e.py).
import inspect  # noqa: E402

from laya import agent as _agent  # noqa: E402

_src = inspect.getsource(_agent.Agent._encode_state) + inspect.getsource(_agent.Agent.predict_batch)
check_true("usage/asks build_sequence for the stats", "return_truncation_stats=True" in _src)
check_true("usage/publishes the flag", '"truncated": dropped > 0' in _src)
check_true("usage/publishes the dropped count", 'usage["state_tokens_dropped"] = dropped' in _src)
# The two counts are exact only when the whole state was tokenized, so they are opt-in: asking
# restores the full tokenization for that call, and not asking leaves them out rather than
# reporting a floor. Both halves have to be in the source, or one of them has gone missing.
check_true("usage/the counts are behind the opt-in", "if state_token_counts:" in _src)
check_true("usage/asking restores the full tokenization",
           "if truncate_left or state_token_counts:" in _src)
check_true("usage/names the questions that truncated", '"truncated_questions"' in _src)
check_true("usage/keeps the existing token counts",
           '"input_tokens": n_tokens' in _src and '"output_tokens": 0' in _src)


# --------------------------------------------------------- the truncation examples teach this dict
# examples/15 and examples/37 are the two pages that teach the report: each prints it off a call that
# really truncated. tests/test_structured_docs.py already holds examples/03's drawn shape to the same
# literal, so this covers only what that gate cannot see -- that each example reads and names every
# key the agents publish, and that the prose about which end of the state survives matches the code
# that decides it. The key set is read out of `laya/agent.py` by AST rather than repeated here, so a
# key added to the dict has to be taught and one renamed fails the example still pointing at it.
import ast  # noqa: E402
import re  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX15 = "examples/15_truncation_basics.py"
EX37 = "examples/37_long_document.py"
EXAMPLES = (("15", EX15), ("37", EX37))


def _tree(rel):
    with open(os.path.join(ROOT, rel)) as fh:
        return ast.parse(fh.read(), filename=rel)


def _published(rel):
    """Every usage key the module can publish for an answered call, literal and conditional.

    The `usage = {..}` literal is the always-on part. Keys added afterwards as `usage["k"] = ...` are
    published too, just not on every call -- `options` only when a question's options collapsed
    (#538), `state_tokens` / `state_tokens_dropped` only when the caller asked for them (#687). An
    example that reads one of those is reading something the agent really does publish, so both
    sources count here; whether the example asks for it is its own business.
    """
    always, conditional = (), ()
    for fn in ast.walk(_tree(rel)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        literal, keys = None, set()
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Assign) and node.targets):
                continue
            target = node.targets[0]
            if (isinstance(target, ast.Name) and target.id == "usage"
                    and isinstance(node.value, ast.Dict)):
                literal = tuple(k.value for k in node.value.keys if isinstance(k, ast.Constant))
            elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "usage" and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)):
                keys.add(target.slice.value)
        if literal is not None:
            # Scoped to the one function that builds it. `predict_long` also assigns into a `usage`
            # name while aggregating windows, and reading its `usage["windows"]` here would hold the
            # examples to a key no single-window answer carries.
            always, conditional = literal, tuple(sorted(keys - set(literal)))
            break
    check_true("report/a usage dict is built in %s" % rel, bool(always), "(found none)")
    return always, conditional


def _strings(rel):
    """Every string the example carries: the module docstring, the banner, each printed line."""
    return [node.value for node in ast.walk(_tree(rel))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def _usage_reads(rel):
    """The usage keys the example reads, as code rather than prose: `usage["k"]`, `x["usage"]["k"]`."""
    found = set()
    for node in ast.walk(_tree(rel)):
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
            continue
        key = node.slice.value
        if not isinstance(key, str):
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id == "usage":
            found.add(key)
        if (isinstance(base, ast.Subscript) and isinstance(base.slice, ast.Constant)
                and base.slice.value == "usage"):
            found.add(key)
    return tuple(sorted(found))


published, conditional = _published("laya/agent.py")
# Named rather than inferred by subtraction. "Everything in the dict that is not a token total" was
# this report while the dict held exactly those keys, and it no longer is: `options` sits in the same
# block but belongs to #538, and `state_tokens` / `state_tokens_dropped` are published only when the
# caller asks for them (#687). So the four keys are named, and then checked to be published at all --
# which is the half of the old check that had teeth.
report = ("state_tokens", "state_tokens_dropped", "truncated", "truncated_questions")
check_true("report/the report is more than the two token totals",
           len(report) >= 4, "(%r)" % (report,))
check_true("report/every key of the report is published by the agent",
           all(k in published + conditional for k in report),
           "(always %r, conditional %r)" % (published, conditional))

UNREPORTED = re.compile(r"nothing warns|is silent|silent in|no warning|not reported"
                        r"|nothing says|never says|cannot be seen", re.I)

for tag, rel in EXAMPLES:
    missing = tuple(k for k in report if k not in _usage_reads(rel))
    check("%s/prints every key of the report from a live call" % tag, missing, ())
    extra = tuple(k for k in _usage_reads(rel) if k not in published + conditional)
    check("%s/reads no key the agents do not publish" % tag, extra, ())
    prose = " ".join(_strings(rel))
    for key in report:
        check_true("%s/names `%s` so a reader can grep for it" % (tag, key), "`%s`" % key in prose)

    # The claim that made 15 wrong when it was written: it told the reader truncation is silent and
    # "nothing warns you", one line after printing `truncated`. A sentence may still say the cut is
    # invisible -- the two answers really are indistinguishable -- but only if it names the key that
    # says otherwise.
    for text in _strings(rel):
        for sentence in [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]:
            if UNREPORTED.search(sentence):
                check_true("%s/a claim that the cut is invisible must name the key that reports it" % tag,
                           "`truncated`" in sentence, "(%r)" % sentence[:120])
            # Which end of the state survives is a property of the state's type, not of a parameter a
            # caller passes. 37 said `predict` "does not expose" `truncate_left`, which is only true
            # of the keyword: the rule is reachable, and it is `list`. Deliberately per sentence -- a
            # page that names `list` somewhere else still has to say which type the clamp reads in
            # the same breath as the flag, because that pairing is the thing a reader takes away.
            if "`truncate_left`" in sentence or "truncate_left=" in sentence:
                check_true("%s/a sentence about `truncate_left` names the type that sets it" % tag,
                           "list" in sentence, "(%r)" % sentence[:120])

# ------------------------------------------------- and the rule the examples teach is the one shipped
# Both agents pick the clamp side the same way, from the state's type, because a conversation list
# serializes newest-last and a left-keeping clamp is the only one that preserves the newest turn.
# Read as a pattern rather than a string: #687 moves the same expression inline into the
# `build_sequence` call, and the examples' claim has to survive that reformatting or keep its meaning.
CLAMP = re.compile(r"truncate_left[^\n]*?isinstance\(state, ?list\)")
for rel in ("laya/agent.py", "laya/onnx_agent.py"):
    with open(os.path.join(ROOT, rel)) as fh:
        src = fh.read()
    check_true("clamp/%s clamps from the state's type" % rel, CLAMP.search(src) is not None,
               "no `truncate_left = isinstance(state, list)`")
# ...and 37 is the page that teaches it, so the rule above has something to police: a page that
# dropped the name would pass every sentence check by never raising one.
check_true("37/teaches the clamp by its real name",
           any("truncate_left" in text for text in _strings(EX37)), "never mentions `truncate_left`")


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)