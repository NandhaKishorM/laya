"""ONNXAgent.predict_batch: the batch path the PyTorch Agent has and ONNX did not (#323).

`Router.predict_batch` calls `agent.predict_batch` unconditionally, so an ONNX-backed Router raised
`AttributeError` on the batched entry point while the single-state path worked. Beyond the missing
method, one session run per state is the slow way to score many states: ONNX Runtime already
parallelises inside a run, so the rows of several states belong in one call.

No onnxruntime and no checkpoint: onnxruntime is imported lazily inside `__init__`, so the class
imports, and `predict_batch` is driven with a stub session and a fake tokenizer (the technique
`tests/test_onnx_lang_parity.py` uses for `_infer`). The stub derives each row's logits from that
row's own tokens, so a row decoded for the wrong state changes the answer and fails these checks.

Run: python tests/test_onnx_batch.py
"""
import inspect
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import clamp_temperature  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402
from laya.router import Router  # noqa: E402

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


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: raised %r, expected %s" % (name, e, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))


# ---------------------------------------------------------------- a fake tokenizer + stub session
class _FakeTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(c) % 40) for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _StubSession:
    """Row logits derived from the row's own tokens: a mis-mapped row changes the answer."""

    def __init__(self):
        self.calls = []          # rows per session run, in order

    def run(self, names, inputs):
        ids = inputs["input_ids"]
        self.calls.append(len(ids))
        # Every row for a state carries that state's tokens last, so this weight is constant
        # within a state and distinct across the states used below.
        weight = 1.0 + (ids.sum(axis=1) % 4)
        logits = np.stack([weight, np.ones_like(weight)], axis=1).astype(np.float32)
        act = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (len(ids), 1))
        return [logits, act]


def _bare_onnx(lang_temperatures=None):
    a = ONNXAgent.__new__(ONNXAgent)         # skip __init__: no onnxruntime, no checkpoint
    a.model_id = "stub"
    a.cfg = {"max_len": 64, "head_max_len": 32}
    a.tok = _FakeTok()
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    a.session = _StubSession()
    a.hooks = ()
    a.hooks_raise = True
    a.hooks_concurrent = True
    a.hooks_timeout = None
    a._hooks_lock = None
    a._hooks_mutex = threading.Lock()
    a.lang_temperatures = {}
    for l, cfg in (lang_temperatures or {}).items():
        norm = l.split("-")[0].lower()
        a.lang_temperatures[norm] = {
            "temperature": [clamp_temperature(t) for t in cfg["temperature"]],
            "temperature_by_options": {k: clamp_temperature(v)
                                       for k, v in cfg.get("temperature_by_options", {}).items()},
        }
    return a


QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "support": "help", "sales": "buy"}},
    "flag": {"type": "noul", "instructions": "Urgent?"},
}
STATES = ["aa", "bb", "c", "e"]   # state token sums 54/56/29/31 -> four distinct mod-4 weights

# ---------------------------------------------------------------- contract
check_true("contract/predict_batch is defined", callable(getattr(ONNXAgent, "predict_batch", None)))
check_true("contract/system_one delegates to predict_batch",
           "self.predict_batch(" in inspect.getsource(ONNXAgent.system_one))
check_true("contract/predict alias preserved", ONNXAgent.predict is ONNXAgent.system_one)

_onnx_so = inspect.signature(ONNXAgent.predict_batch).parameters
_agent_so = inspect.signature(Agent.predict_batch).parameters
_shared = ["batch_size", "lang", "hooks", "on_predict_start", "on_predict_end", "hooks_raise",
           "max_len", "head_max_len", "min_confidence"]
check("contract/parameter names match Agent.predict_batch",
      [p for p in _shared if p not in _onnx_so], [])
check("contract/the Agent's full knob set is now ported",
      sorted(set(_agent_so) - set(_onnx_so)), [])


# ---------------------------------------------------------------- batched == sequential
sequential_agent = _bare_onnx()
sequential = [sequential_agent.system_one(s, QUESTIONS) for s in STATES]

agent = _bare_onnx()
batched = agent.predict_batch(STATES, QUESTIONS)

check("batch/one result per state", len(batched), len(STATES))
check("batch/answers match system_one exactly", batched, sequential)
check("batch/four states cost one session run", agent.session.calls, [2 * len(STATES)])
check_true("batch/result shape matches system_one",
           all(set(r) == {"model", "answers", "usage"} for r in batched))
check("batch/usage matches system_one", [r["usage"] for r in batched],
      [r["usage"] for r in sequential])
check_true("batch/states are distinguishable (so the parity check is not vacuous)",
           len({r["answers"]["dept"]["probabilities"]["billing"] for r in batched}) == len(STATES))

chunked = _bare_onnx()
chunked_results = chunked.predict_batch(STATES, QUESTIONS, batch_size=2)
check("batch/batch_size=2 chunks into two runs of two states each",
      chunked.session.calls, [2 * 2, 2 * 2])
check("batch/chunked results keep input order", chunked_results, sequential)
for batch_size in (0, -1, None):
    whole = _bare_onnx()
    check("batch/batch_size=%r still matches system_one" % batch_size,
          whole.predict_batch(STATES, QUESTIONS, batch_size=batch_size), sequential)
    check("batch/batch_size=%r does not chunk" % batch_size, whole.session.calls, [2 * len(STATES)])

# ---------------------------------------------------------------- empty and invalid input
empty = _bare_onnx()
check("empty/no states -> empty list", empty.predict_batch([], QUESTIONS), [])
check("empty/no states -> no session run", empty.session.calls, [])
no_q = _bare_onnx()
check("empty/no questions -> one zero-usage result per state",
      no_q.predict_batch(STATES, {}),
      [{"model": "laya-rl-agent-onnx", "answers": {},
        "usage": {"input_tokens": 0, "output_tokens": 0}}] * len(STATES))
check("empty/no questions -> no session run", no_q.session.calls, [])
check_raises("empty/bare string rejected", TypeError,
             lambda: _bare_onnx().predict_batch("just a string", QUESTIONS))
check_raises("empty/bare dict rejected", TypeError,
             lambda: _bare_onnx().predict_batch({"body": "x"}, QUESTIONS))
check_raises("empty/invalid question rejected before any run", ValueError,
             lambda: _bare_onnx().predict_batch(STATES, {"bad": {"type": "nope", "instructions": "?"}}))
for bad_type in ([], {}):
    invalid = _bare_onnx()
    name = "invalid/question type %r" % (bad_type,)
    try:
        invalid.predict_batch(STATES, {"refund": {"type": bad_type, "instructions": "Refund requested?"}})
        FAIL.append("%s: no error raised" % name)
    except ValueError as e:
        check(name, str(e), "question 'refund': unknown type %r; use one of ['choice', 'noul', 'score']"
              % (bad_type,))
    except Exception as e:
        FAIL.append("%s: %s instead of ValueError: %s" % (name, type(e).__name__, e))
    check("%s does not run the session" % name, invalid.session.calls, [])
for bad_ins in (None, "", "   ", []):
    check_raises("invalid/instructions %r rejected" % (bad_ins,), ValueError,
                 lambda bi=bad_ins: _bare_onnx().predict_batch(STATES, {"q": {"type": "noul", "instructions": bi}}))
check_raises("invalid/questions not a dict", TypeError,
             lambda: _bare_onnx().predict_batch(STATES, ["not", "a", "dict"]))
check_raises("invalid/state is None in batch", TypeError,
             lambda: _bare_onnx().predict_batch([None], QUESTIONS))
check("budget/max_len and head_max_len forward through the batch path",
      _bare_onnx().predict_batch(STATES, QUESTIONS, max_len=32, head_max_len=16),
      [_bare_onnx().system_one(s, QUESTIONS, max_len=32, head_max_len=16) for s in STATES])

# ---------------------------------------------------------------- hooks
seen, events = [], []


def _end_hook(ctx):
    events.append(ctx)
    seen.append([sorted(r["answers"]) for r in ctx.results])


hooked = _bare_onnx()
hooked_results = hooked.predict_batch(STATES[:2], QUESTIONS, on_predict_end=_end_hook)
check("hooks/end hook sees the batched results", seen[-1], [["dept", "flag"]] * 2)
check("hooks/end hook usage aggregates every state", events[-1].usage["input_tokens"],
      sum(r["usage"]["input_tokens"] for r in hooked_results))
check_true("hooks/end hook sees elapsed_ms", isinstance(events[-1].elapsed_ms, float))

rewritten = _bare_onnx()
check("hooks/start hook can rewrite the states",
      rewritten.predict_batch(STATES[:1], QUESTIONS,
                              on_predict_start=lambda ctx: setattr(ctx, "states", ["fff"])),
      [_bare_onnx().system_one("fff", QUESTIONS)])

skipped = _bare_onnx()
sentinel = [{"model": "cached", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}]
check("hooks/start hook can serve cached results",
      skipped.predict_batch(STATES, QUESTIONS, on_predict_start=lambda ctx: ctx.skip(sentinel)),
      sentinel)
check("hooks/skip runs no session", skipped.session.calls, [])


class _ErrorHook:
    def __init__(self, sink):
        self.sink = sink

    def on_error(self, ctx):
        self.sink.append(ctx.error)


class _Boom:
    def run(self, names, inputs):
        raise RuntimeError("session failed")


errors = []
failing = _bare_onnx()
failing.session = _Boom()
check_raises("hooks/session failure propagates", RuntimeError,
             lambda: failing.predict_batch(STATES, QUESTIONS, hooks=[_ErrorHook(errors)]))
check("hooks/error hook receives the failure once", [str(e) for e in errors], ["session failed"])


# ---------------------------------------------------------------- min_confidence
gated = _bare_onnx()
results = gated.predict_batch(STATES, QUESTIONS, min_confidence=0.9)
flagged = [r["answers"]["dept"].get("low_confidence", False) for r in results]
check("min_confidence/flags exactly the answers below the threshold", flagged,
      [r["answers"]["dept"]["answer_confidence"] < 0.9 for r in results])
check_true("min_confidence/covers both branches", any(flagged) and not all(flagged), flagged)
check("min_confidence/raw answers stay intact",
      [r["answers"]["dept"]["probabilities"] for r in results],
      [r["answers"]["dept"]["probabilities"] for r in sequential])

flag_seen = []
_bare_onnx().predict_batch(
    STATES, QUESTIONS, min_confidence=0.9,
    on_predict_end=lambda ctx: flag_seen.append(
        [r["answers"]["dept"].get("low_confidence", False) for r in ctx.results]))
check("min_confidence/end hook sees the flags", flag_seen[-1], flagged)

check("min_confidence/unset adds no flag",
      any("low_confidence" in a for r in _bare_onnx().predict_batch(STATES, QUESTIONS)
          for a in r["answers"].values()), False)
check("min_confidence/0.0 is a no-op",
      any("low_confidence" in a
          for r in _bare_onnx().predict_batch(STATES, QUESTIONS, min_confidence=0.0)
          for a in r["answers"].values()), False)
check_raises("min_confidence/rejects out-of-range", ValueError,
             lambda: _bare_onnx().predict_batch(STATES, QUESTIONS, min_confidence=1.5))
check_raises("min_confidence/rejects a bool", ValueError,
             lambda: _bare_onnx().predict_batch(STATES, QUESTIONS, min_confidence=True))

# ---------------------------------------------------------------- lang on the batch path
lang_agent = _bare_onnx({"de": {"temperature": [3.0, 3.0, 3.0]}})
base_lang = lang_agent.predict_batch(STATES, QUESTIONS)
de_lang = lang_agent.predict_batch(STATES, QUESTIONS, lang="de")
check("lang/batch equals system_one with the same lang",
      de_lang, [lang_agent.system_one(s, QUESTIONS, lang="de") for s in STATES])
check_true("lang/override changes the batch probabilities",
           [r["answers"]["dept"]["probabilities"] for r in de_lang]
           != [r["answers"]["dept"]["probabilities"] for r in base_lang])
check("lang/hyphen subtag resolves on the batch path",
      lang_agent.predict_batch(STATES, QUESTIONS, lang="de-DE")[0]["answers"]["dept"]["probabilities"],
      de_lang[0]["answers"]["dept"]["probabilities"])


# ---------------------------------------------------------------- Router integration
# The motivating failure: Router.predict_batch calls agent.predict_batch unconditionally, so an
# attached ONNX agent raised AttributeError on the batched entry point while predict worked.
router_agent = _bare_onnx()
router = Router(max_loaded=1, default="english")
router.attach("english", router_agent)
requests = [{"state": s, "questions": QUESTIONS, "model": "english"} for s in STATES[:2]]
routed = router.predict_batch(requests)

check("router/one routed result per request", len(routed), 2)
check("router/results carry the routing key", [sorted(r) for r in routed],
      [["answers", "model", "routing", "usage"]] * 2)
check("router/two states share one session run", router_agent.session.calls, [2 * 2])


# ---------------------------------------------------------------- single-option questions
# The exported graph traces the decision head's `topk(2)` (the eager forward's one-option branch is
# a Python `if` on the marker count, which the trace bakes in), so ONNX Runtime refused a run whose
# questions all had one option -- "k argument [2] should not be greater than specified axis dim
# value [1]" -- while the PyTorch Agent answered it. This stub enforces the same rule, and scores a
# masked slot -1e4 the way the graph's `masked_fill` does.
class _ExportedGraphSession:
    def __init__(self):
        self.marker_widths = []
        self.masks = []

    def run(self, names, inputs):
        mask = inputs["marker_mask"]
        self.marker_widths.append(mask.shape[1])
        self.masks.append(mask.tolist())
        if mask.shape[1] < 2:
            raise RuntimeError("[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while "
                               "running TopK node. Status Message: k argument [2] should not be "
                               "greater than specified axis dim value [%d]" % mask.shape[1])
        logits = np.where(mask, 1.0 + np.arange(mask.shape[1], dtype=np.float32), -1e4).astype(np.float32)
        act = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (len(mask), 1))
        return [logits, act]


ONE_OPTION = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money"}}}
ONE_LEVEL = {"level": {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent"]}}

for label, questions, key, want in (("choice", ONE_OPTION, "dept", {"billing": 1.0}),
                                    ("score", ONE_LEVEL, "level", {"0": 1.0})):
    single = _bare_onnx()
    single.session = _ExportedGraphSession()
    try:
        got = single.predict("aa", questions)["answers"][key]["probabilities"]
    except RuntimeError as e:
        got = str(e)[:80]
    check("single option/a one-option %s is answered, not refused by the graph" % label, got, want)
    check("single option/%s: the extra slot is padded and masked" % label, single.session.masks, [[[True, False]]])

single = _bare_onnx()
single.session = _ExportedGraphSession()
try:
    got = [r["answers"]["dept"]["probabilities"] for r in single.predict_batch(STATES, ONE_OPTION)]
except RuntimeError as e:
    got = str(e)[:80]
check("single option/the batch path answers too, in one run", (got, single.session.marker_widths),
      ([{"billing": 1.0}] * len(STATES), [2]))

wide = _bare_onnx()
wide.session = _ExportedGraphSession()
wide.predict("aa", QUESTIONS)
check("single option/a run that already has two or more slots is sent as before",
      wide.session.marker_widths, [3])


# ---------------------------------------------------------------- batch_size is a shape, not a sign
# `ONNXAgent.predict_batch` spends `batch_size` as a `range()` step, so a value `range()` cannot
# spend used to leak the interpreter's own error from the middle of the batch loop. Measured on the
# stub session above at 0.4.22: `2.5` raised `TypeError: 'float' object cannot be interpreted as an
# integer`, and `"2"` raised `TypeError: '>' not supported between instances of 'str' and 'int'` --
# the second from the `> 0` test, before a single state was encoded. `laya/mcp/tools.py` and
# `laya/serve.py` both refused those shapes before the call reached here, so only the Python route
# -- the documented one, and the one with no schema in front of it -- met the raw `TypeError`.
# `common.check_batch_size` now refuses them by name, on the same line of `predict_batch` that
# `Agent` uses, so the two agents answer a bad cap the same way.
for bad in (2.5, 2.0, "2", "", True, False, [2], {"n": 2}, float("nan"), float("inf")):
    check_raises("chunk/batch_size %r is refused by name" % (bad,), ValueError,
                 lambda b=bad: _bare_onnx().predict_batch(STATES, QUESTIONS, batch_size=b))

refused = _bare_onnx()
try:
    refused.predict_batch(STATES, QUESTIONS, batch_size=2.5)
except ValueError as exc:
    check("chunk/the refusal names the value it cannot use", str(exc),
          "batch_size must be an integer count of states, got 2.5")
else:
    check_true("chunk/the refusal names the value it cannot use", False, "no ValueError raised")
check("chunk/a refused cap runs no session", refused.session.calls, [])

# A `bool` was the worse failure because it does not crash: `True` *is* an `int` to `range()`, so it
# capped every pass at one state -- measured `[2, 2, 2, 2]` where the un-capped path is one run of
# eight rows -- and it switched `sort_by_length` off, because that knob's own gate is `1 < chunk`.
# `False` was not a cap of zero but "unset", because the gate is `batch_size > 0`. Both were silent.

# The "no cap" spellings stay legal, and stay one run over every state:
# `laya-java/fixtures/window_plan.json`'s `batch_cap` grid replays `(None, 0, -1, 1, 8)` through the
# Java, Go, C#, C++ and TS ports, whose typed `int` treats zero and negative as "unset". The guard
# stops at the shape, never at the sign.
for unset in (None, 0, -1):
    one = _bare_onnx()
    out = one.predict_batch(STATES, QUESTIONS, batch_size=unset)
    check("chunk/batch_size=%r is one run over every state" % (unset,),
          (len(out), one.session.calls), (len(STATES), [2 * len(STATES)]))

# `Integral`, not `int`, is the test -- `numpy.int64` spends in `range()` exactly as well, and a
# caller who derives the cap from an array shape is the common case. `ece_score` in the same module
# gives the same reason for the same reason.
np_cap = _bare_onnx()
np_cap.predict_batch(STATES, QUESTIONS, batch_size=np.int64(2))
check("chunk/a numpy integer cap is spent, not refused", np_cap.session.calls, [4, 4])


# ---------------------------------------------------------------- question token reuse
# Agent.predict_batch caches question tokenizations across states (@_reuse_question_tokens);
# ONNXAgent.predict_batch shares that cache, so question instructions and options are encoded once
# per batch rather than once per state document.
tok_records = []


class _TrackingTok(_FakeTok):
    def __call__(self, text, **kw):
        tok_records.append(text)
        return super().__call__(text, **kw)


tracking_agent = _bare_onnx()
tracking_agent.tok = _TrackingTok()
tracking_agent.predict_batch(STATES, QUESTIONS)
dept_instruction = "choice question: Which team?"
check("reuse/question instruction encoded once across the batch",
      tok_records.count(dept_instruction), 1)
check("reuse/total tokenizations capped by reuse across states",
      len(tok_records), 7 + len(STATES))



# --------------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)


