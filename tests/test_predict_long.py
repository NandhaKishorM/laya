"""predict_long: scan a state longer than the window and aggregate per question.

Weight-free. The real forward path is stubbed (predict_batch returns canned per-window answers),
so this checks only predict_long's own logic: the fits-in-one-window short-circuit, the overlapping
window split, the per-type aggregation (noul = strongest window, choice/score = most-confident
window), and the per-call hook controls it forwards to whichever of those two calls runs. Numerical
behaviour on real weights is exercised in tests/test_local_e2e.py.
"""
import functools
import inspect
import os
import re
import sys
import threading
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import laya.agent as agent_mod
from laya.agent import Agent, _check_scan_budget, _start_evidence

from laya.onnx_agent import ONNXAgent  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name if got == want else "%s: got %r want %r" % (name, got, want))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: raised %r not %s" % (name, e, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s: %s" % (name, detail))


class _Tok:
    # The special-token ids and `truncation` are what `laya.common.build_head` reads: predict_long
    # assembles each question's head to measure the room it leaves for the state before it sizes a
    # window, so a fake that cannot build a head cannot reach the scan at all.
    mask_token = "[M]"
    cls_token_id, sep_token_id, mask_token_id = 101, 102, 103

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        # token count == character count, so the test controls windowing by string length
        ids = list(range(len(text)))
        return {"input_ids": ids[:max_length] if (truncation and max_length) else ids}

    def decode(self, ids):
        return "w%d_%d" % (ids[0], ids[-1]) if ids else "w"


def make_agent(batch_result_fn):
    a = Agent.__new__(Agent)
    a.cfg = {"max_len": 100, "head_max_len": 20}   # budget = max(64, 100-20-8) = 72
    a.tok = _Tok()
    a._to_internal = staticmethod(Agent._to_internal).__func__
    a._calls = {"system_one": 0, "batch_states": None, "system_one_kwargs": None,
                "batch_kwargs": None, "batch_size": "unset"}

    def _system_one(state, questions, lang=None, **controls):
        a._calls["system_one"] += 1
        a._calls["system_one_kwargs"] = controls
        return {"model": "laya-rl-agent", "answers": {"_via": "system_one"}, "usage": {"input_tokens": 1}}

    def _predict_batch(states, questions, batch_size=None, lang=None, **controls):
        a._calls["batch_states"] = list(states)
        a._calls["batch_kwargs"] = controls
        a._calls["batch_size"] = batch_size
        return batch_result_fn(list(states), questions)

    a.system_one = _system_one
    a.predict_batch = _predict_batch
    return a


# `_encode_state` items carry the state truncation counts that `predict_batch` reports in `usage` (#174)
NO_STATE_STATS = {"state_tokens": 0, "state_tokens_used": 0, "state_tokens_dropped": 0, "truncated": False}


def make_real_agent():
    """`predict_long` on the real `predict_batch`/`system_one`, with only the three composed
    helpers stubbed -- the harness `tests/test_hooks.py` uses, so no weights are involved.

    Why both agents: the stubbed `predict_batch` above is the right instrument for what
    `predict_long` *forwards*, but it never runs a hook, so anything it says about a hook answering
    rests on a result count alone. Here the hook chain really runs and every forward pass is
    counted, which is the difference between "a hook answered the document" and "a hook replaced
    the window list" -- both change the count.
    """
    a = Agent.__new__(Agent)
    a.cfg = {"max_len": 100, "head_max_len": 20}       # budget = max(64, 100-20-8) = 72
    a.tok = type("Tok", (_Tok,), {"pad_token_id": 0})()
    a._to_internal = staticmethod(Agent._to_internal).__func__
    a.model_id = "convaiinnovations/laya"
    a.hooks = []
    a.hooks_raise = True
    a.hooks_timeout = None
    a._hooks_lock = threading.Lock()
    a._forward_calls, a._encoded = [], []

    def _encode_state(state, ids, internal, **overrides):
        a._encoded.append(state)
        return [{"ids": [1, 2, 3], "markers": [0, 1], "qtype": 2, "state_stats": NO_STATE_STATS} for _ in ids]

    def _forward(b):
        n = b["input_ids"].shape[0]
        a._forward_calls.append(n)
        return np.zeros((n, 2), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode_answers(logits, act, items, ids, internal, row, **kw):
        conf = 0.4 + 0.05 * row                        # so the last window decides
        return {"dept": {"type": "choice", "choice": "a", "confidence": conf,
                         "answer_confidence": conf},
                "flag": {"type": "noul", "noul": conf, "confidence": conf,
                         "answer_confidence": conf}}

    a._encode_state = _encode_state
    a._forward = _forward
    a._decode_answers = _decode_answers
    return a


Q = {"dept": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
     "flag": {"type": "noul", "instructions": "?"}}

# 1. fits in one window -> delegates to system_one, no windowing
a = make_agent(lambda s, q: [])
short = a.predict_long({"body": "x" * 50}, Q)   # 50 tokens <= budget 72
check("short/delegates to system_one", short["answers"], {"_via": "system_one"})
check("short/no predict_batch call", a._calls["batch_states"], None)
check("short/one window is reported", short["usage"].get("windows", "<absent>"), 1)
check("short/system_one's own usage is kept", short["usage"].get("input_tokens", "<absent>"), 1)

# 1b. the count is added to a copy, because a start hook that skips hands back the caller's own
# payload dict and that object may be cached and reused
payload = {"model": "laya-rl-agent", "answers": {"_via": "system_one"}, "usage": {"input_tokens": 1}}
a = make_agent(lambda s, q: [])
a.system_one = lambda state, questions, lang=None, **controls: payload
via_hook = a.predict_long({"body": "x" * 50}, Q)
check("short/a new result dict comes back", via_hook is payload, False)
check("short/the caller's payload is not written to", payload["usage"], {"input_tokens": 1})

# 2. long state -> overlapping windows, aggregated per question
def canned(states, q):
    # one canned answer per window; the 3rd window is the confident/positive one
    out = []
    for i, _ in enumerate(states):
        conf = 0.9 if i == 2 else 0.4
        ptrue = 0.95 if i == 2 else 0.1
        out.append({"answers": {
            "dept": {"type": "choice", "choice": "b" if i == 2 else "a",
                     "probabilities": {"a": 1 - conf, "b": conf}, "confidence": conf,
                     "answer_confidence": conf, "action": {"act_probability": 1.0}},
            "flag": {"type": "noul", "noul": ptrue, "confidence": max(ptrue, 1 - ptrue),
                     "answer_confidence": max(ptrue, 1 - ptrue), "action": {"act_probability": 1.0}},
        }, "usage": {"input_tokens": 10}})
    return out


a = make_agent(canned)
# 300 tokens, budget 72, stride 36 -> several overlapping windows, last covers the tail
res = a.predict_long({"body": "y" * 300}, Q)
nwin = len(a._calls["batch_states"])
check("long/windows recorded in usage", res["usage"]["windows"], nwin)
check("long/more than one window", nwin > 1, True)
check("long/overlap: stride is half the budget", a._calls["batch_states"][1], "w36_107")
check("long/choice = most-confident window", res["answers"]["dept"]["choice"], "b")
check("long/noul = strongest window", res["answers"]["flag"]["noul"], 0.95)
check("long/usage sums window tokens", res["usage"]["input_tokens"], 10 * nwin)
# the deciding window is named on each answer (window index 2 is the confident/positive one)
check("long/choice names the deciding window", res["answers"]["dept"]["window"]["index"], 2)
check("long/noul names the deciding window", res["answers"]["flag"]["window"]["index"], 2)
check("long/window start is the 3rd overlap offset", res["answers"]["dept"]["window"]["token_start"], 72)
check("long/window carries the count", res["answers"]["flag"]["window"]["count"], nwin)


# 2b. a per-question usage field is merged across windows, not replaced
# `usage["options"]` is a dict keyed by question id, set only on the windows where option
# spans actually collapsed. The deciding window is the most confident one, so it is not
# necessarily the last window that collapsed: replacing instead of merging reported a
# collapse for a window that did not decide and dropped the deciding window's own record.
def collapsing(states, q):
    # window 0 collapses dept and is the most confident; every later window collapses flag,
    # so the last collapsing window is not the deciding one
    out = []
    for i, _ in enumerate(states):
        conf = 0.9 if i == 0 else 0.4
        which = "dept" if i == 0 else "flag"
        total, distinct = (58, 42) if which == "dept" else (30, 12)
        opt = {which: {"total": total, "distinct": distinct, "tokens_per_option": 0.5}}
        out.append({"answers": {
            "dept": {"type": "choice", "choice": "b" if i == 0 else "a",
                     "probabilities": {"a": 1 - conf, "b": conf}, "confidence": conf,
                     "answer_confidence": conf, "action": {"act_probability": 1.0}},
            "flag": {"type": "noul", "noul": 0.9 if i == 0 else 0.1, "confidence": conf,
                     "answer_confidence": conf, "action": {"act_probability": 1.0}},
        }, "usage": {"input_tokens": 10, "options": opt}})
    return out


_seen_windows = []


def collapsing_capturing(states, q, **kw):
    _seen_windows.extend(states)
    return collapsing(states, q, **kw)


a = make_agent(collapsing_capturing)
res = a.predict_long({"body": "y" * 300}, Q)
opts = res["usage"].get("options") or {}
nwin = len(_seen_windows)
decided = res["answers"]["dept"]["window"]["index"]
check("collapse/records from more than one window survive", sorted(opts), ["dept", "flag"])
check("collapse/the deciding window is the confident first one", decided, 0)
check_true("collapse/which is not the last window scanned", decided != nwin - 1, [decided, nwin])
check("collapse/so the deciding window's own record is reported", opts.get("dept"), {
    "total": 58, "distinct": 42, "tokens_per_option": 0.5})
check("collapse/a later non-deciding window is reported too", opts.get("flag"), {
    "total": 30, "distinct": 12, "tokens_per_option": 0.5})
check("collapse/numerics still sum across every window", res["usage"]["input_tokens"], 10 * nwin)

# 3. only aggregate="auto" is supported
a = make_agent(canned)
check_raises("aggregate/rejects unknown mode", ValueError,
             lambda: a.predict_long({"body": "y" * 300}, Q, aggregate="mean"))

# ------------------------------------------------------------------- Router.predict_long
# The Router is the entry point the README leads with, and it had no windowed scan: a state past
# max_len was answered from its first window however it was routed. Weight-free, like the rest of
# this file -- the routed agent is a stub, attached so no checkpoint is ever built.
from laya.router import Router  # noqa: E402


class _LongStub:
    def __init__(self):
        self.calls = []

    def predict_long(self, state, questions, window=None, stride=None, aggregate="auto",
                     batch_size=None, lang=None):
        self.calls.append({"state": state, "questions": questions, "window": window,
                           "stride": stride, "aggregate": aggregate,
                           "batch_size": batch_size, "lang": lang})
        return {"model": "stub", "answers": {"scanned": {"noul": 0.5}},
                "usage": {"windows": 3, "input_tokens": 30}}


def _router(stub, hooks=None):
    r = Router(hooks=hooks) if hooks else Router()
    r.attach("english", stub)
    return r


# 4. routes first, then scans on the routed agent, with the caller's options forwarded
stub = _LongStub()
out = _router(stub).predict_long({"body": "y" * 300}, Q, model="english",
                                 window=64, stride=32, batch_size=8, lang="de")
check("router/routing key attached", out["routing"]["model"], "english")
# `predict_long` takes no `min_confidence` -- a window scan has no single confidence to gate on --
# so the gate writes nothing and the scan's answers come back exactly as produced. A long-document
# result is not a gated decision, and now it does not claim to be one.
check("router/answers come from the scan", out["answers"],
      {"scanned": {"noul": 0.5}})
check("router/window forwarded", stub.calls[0]["window"], 64)
check("router/stride forwarded", stub.calls[0]["stride"], 32)
check("router/batch_size forwarded", stub.calls[0]["batch_size"], 8)
check("router/explicit lang forwarded", stub.calls[0]["lang"], "de")
check("router/usage carried through", out["usage"]["windows"], 3)

# 5. an installed start hook runs before the scan and may answer instead of it
class _Skipper:
    def __init__(self):
        self.started = 0

    def on_predict_start(self, ctx):
        self.started += 1
        ctx.skip([{"model": "hook", "answers": {"cached": {"noul": 0.9}}, "usage": {}}])


skipper = _Skipper()
skipped_stub = _LongStub()
out = _router(skipped_stub, hooks=[skipper]).predict_long(
    {"body": "y" * 300}, Q, model="english")
check("router/installed start hook ran", skipper.started, 1)
check("router/hook answer wins over the scan", skipped_stub.calls, [])
check("router/skipped answer still routed", out["routing"]["model"], "english")

# 6. a start hook that rewrites the state: the scan reads what it left behind
class _Rewriter:
    def on_predict_start(self, ctx):
        ctx.states[0] = "rewritten by the start hook"


rewritten_stub = _LongStub()
_router(rewritten_stub, hooks=[_Rewriter()]).predict_long(
    {"body": "y" * 300}, Q, model="english")
check("router/scan uses the rewritten state", rewritten_stub.calls[0]["state"],
      "rewritten by the start hook")

# 7. a per-call end hook sees the scanned result
ended = {}


class _Recorder:
    def on_predict_end(self, ctx):
        ended["answers"] = ctx.results[0]["answers"]


recorder_stub = _LongStub()
_router(recorder_stub).predict_long({"body": "y" * 300}, Q, model="english",
                                    on_predict_end=_Recorder().on_predict_end)
check("router/per-call end hook sees the scan", ended["answers"],
      {"scanned": {"noul": 0.5}})

# 8. an agent with no predict_long is a named caller error, not a bare AttributeError
class _NoScan:
    def system_one(self, state, questions):
        return {"model": "noscan", "answers": {}, "usage": {}}


r_noscan = Router()
r_noscan.attach("english", _NoScan())
check_raises("router/agent without predict_long", TypeError,
             lambda: r_noscan.predict_long({"body": "y" * 300}, Q, model="english"))

# 4. the per-call hook controls `predict`/`system_one`/`predict_batch` take reach the scan
HOOK_KEYS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")
LONG = {"body": "y" * 300}
sentinel = object()


def sentinel_start(ctx):
    pass


a = make_agent(canned)
res = a.predict_long(LONG, Q, hooks=[sentinel], on_predict_start=sentinel_start,
                     hooks_raise=False, hooks_timeout=2.5)
got = a._calls["batch_kwargs"]
check("hooks/forwarded to the scan", sorted(got), sorted(HOOK_KEYS))
check("hooks/the hook list reaches it", got.get("hooks"), [sentinel])
check("hooks/hooks_raise reaches it", got.get("hooks_raise"), False)
check("hooks/hooks_timeout reaches it", got.get("hooks_timeout"), 2.5)
started = got.get("on_predict_start")
started = started if isinstance(started, list) else ([] if started is None else [started])
check("hooks/the caller's start hook still runs first",
      bool(started) and started[0] is sentinel_start, True)
check("hooks/end stays absent when the caller sets none", got.get("on_predict_end"), None)

# 5. the same controls reach the fits-in-one-window path
a = make_agent(lambda s, q: [])
a.predict_long({"body": "x" * 50}, Q, hooks=[sentinel], hooks_timeout=1.25)
got = a._calls["system_one_kwargs"]
check("hooks/forwarded to system_one", sorted(got), sorted(HOOK_KEYS))
check("hooks/the hook list reaches it there", got.get("hooks"), [sentinel])
check("hooks/hooks_timeout reaches it there", got.get("hooks_timeout"), 1.25)

# 6. what the scan forwards is the window texts, in scan order, not the caller's state object
a = make_agent(canned)
a.predict_long(LONG, Q)
states = a._calls["batch_states"]
check("states/several windows", len(states) > 1, True)
check("states/decoded text, not the caller's dict",
      all(isinstance(s, str) for s in states) and states[0] == "w0_71", True)
check("states/they overlap", a._calls["batch_states"][1], "w36_107")

# 7. forwarding the controls must not move a decision
plain = make_agent(canned).predict_long(LONG, Q)
with_hooks = make_agent(canned).predict_long(LONG, Q, hooks=[sentinel], hooks_timeout=9)
check("hooks/forwarding is decision-neutral", with_hooks, plain)

# 8. a start hook that answers the document, on the engine's own evidence
DOC = {"model": "laya-rl-agent",
       "answers": {"dept": {"type": "choice", "choice": "b", "answer_confidence": 0.9},
                   "flag": {"type": "noul", "noul": 0.2, "answer_confidence": 0.8}},
       "usage": {"input_tokens": 3, "output_tokens": 0}}

# The instrument for sections 8-8e: a predict_long call is captured rather than let propagate, so
# a case that ends in the wrong shape -- an answer that should have come back raising, or a
# rejection that should not have happened at all -- fails a named check instead of aborting the run
# and hiding every check after it.
def _attempt(fn):
    """Call `fn`, returning (value, exception)."""
    try:
        return fn(), None
    except BaseException as exc:  # noqa: BLE001
        return None, exc


def _kind(exc):
    return exc.__class__.__name__ if exc else None


# The control first: how many windows this state really splits into, measured from the forward
# passes it costs rather than from the key under test. Two questions, one row each per window.
scan = make_real_agent()
base, base_exc = _attempt(lambda: scan.predict_long(LONG, Q))
check("scan/an unhooked scan returns", _kind(base_exc), None)
nwin = sum(scan._forward_calls) // 2
check("scan/more than one window", nwin > 1, True)
check("scan/the scan reports the windows it read",
      ((base or {}).get("usage") or {}).get("windows", "<absent>"), nwin)
check("scan/the deciding window is the last",
      ((base or {}).get("answers") or {}).get("dept", {}).get("window", {}).get("index"), nwin - 1)


def _narrow(keep):
    """A start hook that truncates the window list and lets inference run on what is left."""
    def hook(ctx):
        ctx.states = list(ctx.states)[:keep]
    return hook


a = make_real_agent()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res, skip_exc = _attempt(
        lambda: a.predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.skip([DOC])))
check("skip/one payload for the document is never refused as a count mismatch", _kind(skip_exc), None)
check("skip/no forward pass ran", a._forward_calls, [])
usage, answers = (res or {}).get("usage", {}), (res or {}).get("answers", {})
check("skip/no window was scored", usage.get("windows", "<absent>"), 0)
check("skip/the payload tokens are kept", usage.get("input_tokens", "<absent>"), 3)
check("skip/the hook's answer is returned", answers.get("dept", {}).get("choice"), "b")
check("skip/no deciding window is claimed",
      sorted(k for v in answers.values() for k in v if k == "window"), [])
check("skip/the caller is told", [w.category.__name__ for w in caught], ["RuntimeWarning"])

# 8b. a start hook that replaces the scan is the middle row: inference ran on the hook's states, so
# the answers aggregate, the count is theirs, and no span of the caller's document is named -- the
# offsets computed above describe windows that were not scored.
a = make_real_agent()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    narrow, narrow_exc = _attempt(lambda: a.predict_long(LONG, Q, on_predict_start=_narrow(1)))
check("narrow/one state left behind is a scan, not a rejected count", _kind(narrow_exc), None)
# Why the forward sits next to the count: it is the proof that tokens were spent, which is exactly
# what made this case read as a hook answer before.
check("narrow/inference ran on the state that was left", a._forward_calls, [2])
check("narrow/the count is the state scored, not the split made",
      ((narrow or {}).get("usage") or {}).get("windows", "<absent>"), 1)
check("narrow/the tokens that forward booked are kept",
      ((narrow or {}).get("usage") or {}).get("input_tokens", 0) > 0, True)
check("narrow/no span of the caller's document is named",
      sorted(k for v in (narrow or {}).get("answers", {}).values() for k in v if k == "window"), [])
check("narrow/the answer is still aggregated",
      (narrow or {}).get("answers", {}).get("dept", {}).get("type"), "choice")
check("narrow/a rewrite is not warned about as a hook answer",
      [w.category.__name__ for w in caught], [])
a = make_real_agent()
two, exc = _attempt(lambda: a.predict_long(LONG, Q, on_predict_start=_narrow(2)))
check("narrow/two states left behind are counted as two",
      ((two or {}).get("usage") or {}).get("windows", "<absent>"), 2)
check("narrow/and two states is what the forward saw", a._forward_calls, [4])
# the other direction, and in place: `predict_long` hands the hook a copy of its split, so the
# state the hook invented is scored and counted without moving the windows computed above.
a = make_real_agent()
grown, exc = _attempt(
    lambda: a.predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.states.append("invented")))
check("grow/scoring one invented state is not an error", _kind(exc), None)
check("grow/the hook's added state is scored",
      ((grown or {}).get("usage") or {}).get("windows", "<absent>"), nwin + 1)
check("grow/the caller's split is the one that was extended", a._forward_calls, [2 * (nwin + 1)])
check("grow/and no span is named",
      sorted(k for v in (grown or {}).get("answers", {}).values() for k in v if k == "window"), [])

# 8c. an unchanged scan keeps the attribution; any rewrite drops it, even one of the same length
a = make_real_agent()
noop, exc = _attempt(
    lambda: a.predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.states.extend(())))
check("rewrite/a no-op on the list changes nothing", _kind(exc), None)
check("rewrite/a no-op keeps the count", ((noop or {}).get("usage") or {}).get("windows", "<absent>"), nwin)
check("rewrite/a no-op keeps the deciding window",
      ((noop or {}).get("answers") or {}).get("dept", {}).get("window", {}).get("index"), nwin - 1)
a = make_real_agent()
upper, exc = _attempt(lambda: a.predict_long(
    LONG, Q, on_predict_start=lambda ctx: setattr(ctx, "states", [s.upper() for s in ctx.states])))
check("rewrite/the hook's text is what was scored", (a._encoded or [""])[0].isupper(), True)
check("rewrite/the same length is still the same count",
      ((upper or {}).get("usage") or {}).get("windows", "<absent>"), nwin)
check("rewrite/a text rewrite names no span",
      sorted(k for v in (upper or {}).get("answers", {}).values() for k in v if k == "window"), [])

# A shared start hook may add a review question or replace the request's schema. Aggregate the
# questions that actually ran, just as a short state does, including a changed type under one id.
def _question_agent():
    a = make_real_agent()
    a.temperature, a.temperature_by_options, a.lang_temperatures = [1.0] * 3, {}, {}
    del a._decode_answers                 # exercise the real typed answer decoder

    def forward(b):
        n = b["input_ids"].shape[0]
        logits = np.tile(np.array([[0.0, 1.4]], dtype=np.float32), (n, 1))
        logits[0] = [3.0, 0.0]           # most confident differs from strongest P(true)
        return logits, np.full((n, 2), 0.5, dtype=np.float32)

    a._forward = forward
    return a


def _rewrite_agrees(name, actual, expected):
    """A question rewrite must reproduce the direct call, EXCEPT for the pass count.

    #692 added this comparison so a start hook's questions drive aggregation: an added decision must
    not disappear, a rename must not `KeyError`, a type change must apply the right rule. All of that
    lives in `answers`, and all of it is asserted exactly as before.

    What is exempted is `usage.windows`, and only in the safe direction. Windowing happens BEFORE the
    hook chain -- it has to, because a start hook is documented to see and rewrite `ctx.states`, i.e.
    the windows themselves -- so the scan is sized from the questions the caller passed. When a hook
    then LEAVES MORE room than the scan was sized for, the scan is finer than it needed to be: every
    token is still covered, the answers are identical, and the only difference is that more passes
    were made. Measured on the ONNX fixture for the `clear` rewrite: 14 windows against 6, with
    identical `answers`. The torch fixture happens to produce 8 either way, so it passed the
    whole-dict comparison by luck rather than by construction.

    The other direction is not exempted and is not silent: a hook that leaves LESS room than the scan
    was sized for is refused outright by `_check_scan_budget`, because then the windows really would be
    re-truncated and part of the document would reach no model. That refusal has its own checks, and it
    is why this exemption is one-sided in practice rather than by assertion here -- an earlier revision
    of this helper also asserted `hooked >= direct`, which reads like a guarantee and cannot fail:
    the only route to a coarser hooked scan is a shrinking rewrite, and that is refused before it can
    be observed. Removed rather than kept as decoration.
    """
    # `actual` is None when the call raised: the torch side routes it through `_attempt`, which
    # swallows the exception and returns None. Report that as a failure rather than raising out of
    # the helper -- the whole-dict `check` this replaced compared None against a dict and failed
    # cleanly, and losing that cost a crash instead of a diagnosis the first time a mutation
    # reintroduced #692's KeyError.
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        FAIL.append("questions/%s did not return a result to compare (actual=%r, expected=%r)"
                    % (name, type(actual).__name__, type(expected).__name__))
        return
    a_usage = dict(actual.get("usage") or {})
    e_usage = dict(expected.get("usage") or {})
    a_windows, e_windows = a_usage.pop("windows", None), e_usage.pop("windows", None)
    check("questions/%s reproduces the answers of the direct call" % name,
          actual.get("answers"), expected.get("answers"))
    check("questions/%s reproduces everything but the pass count" % name,
          ({k: v for k, v in actual.items() if k != "usage"}, a_usage),
          ({k: v for k, v in expected.items() if k != "usage"}, e_usage))
    # `usage.windows` deliberately not compared; both values are read above only to strip them.
    del a_windows, e_windows


question_rewrites = [
    ("append", {**Q, "review": Q["flag"]}),
    ("replace", {"review": Q["flag"]}),
    ("delete", {"flag": Q["flag"]}),
    ("clear", {}),
    ("choice to noul", {**Q, "dept": Q["flag"]}),
    ("noul to choice", {**Q, "flag": Q["dept"]}),
]
for name, rewritten_questions in question_rewrites:
    expected = _question_agent().predict_long(LONG, rewritten_questions)
    actual, exc = _attempt(lambda: _question_agent().predict_long(
        LONG, Q, on_predict_start=lambda ctx: setattr(ctx, "questions", rewritten_questions)))
    check("questions/%s returns without error" % name, _kind(exc), None)
    _rewrite_agrees(name, actual, expected)

check("questions/no-op preserves the unhooked result",
      _question_agent().predict_long(LONG, Q, on_predict_start=lambda ctx: None),
      _question_agent().predict_long(LONG, Q))

# An end hook may annotate only one window. Extra answers are not questions in the scan.
def _annotate_first_window(ctx):
    ctx.results[0]["answers"]["review"] = {"type": "noul", "noul": 0.9, "answer_confidence": 0.9}


expected = _question_agent().predict_long(LONG, Q)
actual, exc = _attempt(lambda: _question_agent().predict_long(LONG, Q, on_predict_end=_annotate_first_window))
check("questions/a first-window end annotation preserves the scan's answers", actual, expected)
check("questions/a first-window end annotation does not require other windows to match", _kind(exc), None)

# The recorder must leave invalid input to predict_batch's existing question validation.
for malformed in (None, [], {"bad": None}, {"bad": {}}, {"bad": {"type": "unknown"}}):
    _, direct_exc = _attempt(lambda: _question_agent().system_one(LONG, malformed))
    _, long_exc = _attempt(lambda: _question_agent().predict_long(LONG, malformed))
    check("questions/invalid input keeps the validator's error: %r" % malformed,
          (_kind(long_exc), str(long_exc)), (_kind(direct_exc), str(direct_exc)))

# 8e. a hook that leaves nothing scores nothing: 0 windows and no answers, not a max() over []
a = make_real_agent()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    empty, exc = _attempt(
        lambda: a.predict_long(LONG, Q, on_predict_start=lambda ctx: setattr(ctx, "states", [])))
check("empty/nothing scored returns instead of raising", _kind(exc), None)
check("empty/no answer is invented", (empty or {"answers": "?"}).get("answers", "?"), {})
check("empty/the count says so", ((empty or {}).get("usage") or {}).get("windows", "<absent>"), 0)
check("empty/the caller is told", [w.category.__name__ for w in caught], ["RuntimeWarning"])

# 8d. the fits-in-one-window path answers to a hook too, so 0 keeps meaning "no window scored this"
a = make_real_agent()
short_hooked, exc = _attempt(
    lambda: a.predict_long({"body": "x" * 50}, Q, on_predict_start=lambda ctx: ctx.skip([DOC])))
check("short skip/no forward pass", a._forward_calls, [])
check("short skip/a hook answer is not counted as a window read",
      ((short_hooked or {}).get("usage") or {}).get("windows", "<absent>"), 0)
a = make_real_agent()
short_plain, exc = _attempt(lambda: a.predict_long({"body": "x" * 50}, Q))
check("short skip/the same state without a hook is one window",
      ((short_plain or {}).get("usage") or {}).get("windows", "<absent>"), 1)

# 9. a batch call that returns the wrong count for its own states is an error, not a guess
a = make_agent(lambda s, q: [{"answers": {}, "usage": {}}] * 2)
check_raises("count/rejects a result count matching nothing", ValueError,
             lambda: a.predict_long(LONG, Q))

# 10. the count is total, so one question works on every result predict_long can return
short_all = make_agent(lambda s, q: []).predict_long({"body": "x" * 50}, Q)
scan_agent = make_agent(canned)
scanned_all = scan_agent.predict_long(LONG, Q)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hook_all, hook_exc = _attempt(
        lambda: make_real_agent().predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.skip([DOC])))
check("total/a hook answer returns rather than being refused", _kind(hook_exc), None)
check("total/no path leaves the key out",
      [(r or {}).get("usage", {}).get("windows", "<absent>")
       for r in (short_all, scanned_all, hook_all)],
      [1, len(scan_agent._calls["batch_states"]), 0])
check("total/0 separates a hook answer from a single-window one",
      ((hook_all or {}).get("usage", {}).get("windows", "<absent>"),
       short_all["usage"].get("windows", "<absent>")),
      (0, 1))


# 11. the page that teaches this key teaches the value the code actually writes
README = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "README.md"), encoding="utf-8").read()
bullet = README[README.index("A state that already fits one window"):][:600]
check("docs/README pins the single-window value", 'usage["windows"] = 1' in bullet, True)
check("docs/README states the key is total", "The key is total" in bullet, True)
check("docs/README documents all three counts", sorted(set(re.findall(r"`([0-9N])`", bullet))),
      ["0", "1", "N"])
# The two rules this branch added, pinned where they are taught rather than in the code comment
hooks_bullet = README[README.index("Hooks wrap the inference that answers the state"):][:900]
check("docs/README says attribution survives only an unchanged scan",
      "reported when the scan reached inference unchanged" in hooks_bullet, True)
check("docs/README names ctx.skip as the way to answer", "ctx.skip(...)" in hooks_bullet, True)
API = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs", "hooks", "api.md"), encoding="utf-8").read()
api_para = API[API.index("On `predict_long` the hooks wrap"):][:1600]
check("docs/api.md tabulates the rewritten scan as its own case",
      "| replaced the scan, in any way |" in api_para, True)
check("docs/api.md tabulates a hook answer as zero windows",
      "| answered with `ctx.skip([result])` | `0` |" in api_para, True)
check("docs/predict_long's docstring says the key is total",
      "always present" in (Agent.predict_long.__doc__ or ""), True)

# 12. the window is sized by the room the questions leave, not by the config alone
#
# Every window is decoded and scored as an ordinary state, so `build_sequence` cuts one that is
# wider than the room the question's own head leaves inside `max_len`. Sized from the config alone
# the window was cut short on the way to the model while `answer["window"]["token_end"]` still
# reported the whole span, and once the room fell under the stride, spans of the document were
# read by no window at all.
from laya import common as common_mod  # noqa: E402
from laya.common import build_sequence, serialize_state  # noqa: E402

TOK = _Tok()
MAX_LEN, HEAD_MAX_LEN, CONFIG_BUDGET = 100, 20, 72      # make_agent's cfg, and its budget


def head_len(qdef):
    """The head `build_sequence` puts in front of the state, measured with no state at all."""
    return len(build_sequence(TOK, "", Agent._to_internal(qdef), 10 ** 6, HEAD_MAX_LEN)[0]) - 1


def room_for(qdef):
    """The state tokens `qdef` leaves inside `max_len`, re-derived from `build_sequence`.

    Deliberately not `laya.common.state_room`: the window has to fit what the sequence builder
    really keeps, so this measures that the long way round instead of trusting the same helper the
    code under test uses.
    """
    return max(0, MAX_LEN - head_len(qdef) - 1)


def q_many(n):
    """A choice question with `n` options, each long enough to be worth capping."""
    return {"type": "choice", "instructions": "which one?",
            "criteria": {"opt%02d" % i: "d" * 20 for i in range(n)}}


def canned_for(questions, i, n):
    """One canned answer per window, the last window the confident one."""
    conf = 0.9 if i == n - 1 else 0.4
    return {"answers": {qid: {"type": "choice", "choice": "opt00", "noul": conf,
                              "confidence": conf, "answer_confidence": conf}
                        for qid in questions},
            "usage": {"input_tokens": 10}}


def scan(questions, **kw):
    """Scan LONG and return (result, the token spans of the windows handed to predict_batch)."""
    agent = make_agent(lambda sts, q: [canned_for(q, i, len(sts)) for i in range(len(sts))])
    result = agent.predict_long(LONG, questions, **kw)
    handed = []
    for text in agent._calls["batch_states"]:
        first, last = text[1:].split("_")           # _Tok.decode renders a span as "wA_B"
        handed.append((int(first), int(last) + 1))
    return result, handed


def read_spans(handed, room):
    """What the model reads of each window: `build_sequence` keeps its first `room` tokens."""
    return [(start, min(end, start + room)) for start, end in handed]


def uncovered(spans, total):
    """Token indices of `total` that no span in `spans` contains."""
    seen = set()
    for start, end in spans:
        seen.update(range(start, end))
    return sorted(set(range(total)) - seen)


STATE_TOKENS = len(TOK(serialize_state(LONG))["input_ids"])

# The premise, checked against build_sequence rather than assumed: a state is cut to the room, and
# for a 12-option question on this config that room is under the budget the scan sized windows by.
ROOM_12 = room_for(q_many(12))
kept = len(build_sequence(TOK, "x" * (MAX_LEN + 50), Agent._to_internal(q_many(12)),
                          MAX_LEN, HEAD_MAX_LEN)[0]) - head_len(q_many(12)) - 1
check("room/build_sequence keeps exactly the room, whatever the state's length", kept, ROOM_12)
check("room/12 options leave less room than the config budget", ROOM_12 < CONFIG_BUDGET, True)
check("room/the questions the sections above scan with leave more room than the budget",
      [room_for(q) >= CONFIG_BUDGET for q in Q.values()], [True, True])

# 12a. a window is never wider than the room, so no window is truncated on the way in
tight, handed = scan({"a": q_many(12)})
check("tight/no window is wider than the room the question leaves",
      max(end - start for start, end in handed) <= ROOM_12, True)
check("tight/the windows still overlap",
      all(b[0] < a[1] for a, b in zip(handed, handed[1:])), True)
check("tight/every token of the state is read by some window",
      uncovered(read_spans(handed, ROOM_12), STATE_TOKENS), [])

# 12b. and the span reported is the span read, not the span asked for
decided = tight["answers"]["a"]["window"]
check("tight/token_end does not overstate what the model read",
      decided["token_end"] - decided["token_start"] <= ROOM_12, True)
check("tight/token_start/token_end are the deciding window's own span",
      (decided["token_start"], decided["token_end"]), handed[decided["index"]])
check("tight/the last window stops at the end of the state", handed[-1][1], STATE_TOKENS)

# 12c. the failure that needed no wide window at all: a room under the stride left whole spans of
# the document unread, because consecutive windows no longer touched.
ROOM_20 = room_for(q_many(20))
check("gap/20 options leave less room than the default stride",
      ROOM_20 < CONFIG_BUDGET // 2, True)
gappy, handed = scan({"a": q_many(20)})
check("gap/the scan covers the whole state even so",
      uncovered(read_spans(handed, ROOM_20), STATE_TOKENS), [])
check("gap/which costs windows, and reports them", gappy["usage"]["windows"], len(handed))

# 12d. two questions with different rooms share one list of windows, so the tightest one sets it
mixed, handed = scan({"wide": q_many(2), "tight": q_many(12)})
check("mixed/the window fits the tightest question",
      max(end - start for start, end in handed) <= min(ROOM_12, room_for(q_many(2))), True)
check("mixed/both questions name the same span",
      mixed["answers"]["wide"]["window"], mixed["answers"]["tight"]["window"])

# 12e. an explicit window past the room is clamped, and the caller is told
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    asked, handed = scan(Q, window=200)
check("explicit/a window past the room is clamped to it",
      max(end - start for start, end in handed) <= room_for(Q["dept"]), True)
check("explicit/and the caller is warned about it",
      [w.category.__name__ for w in caught], ["RuntimeWarning"])
check("explicit/the warning names the room it clamped to",
      "leave inside max_len" in str(caught[0].message) if caught else False, True)

# Clamping the *default* is not the caller's doing, so it is not warned about.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    scan({"a": q_many(12)})
check("explicit/clamping the default window is not warned about",
      [w.category.__name__ for w in caught], [])

# 12f. a stride past the effective window is refused: the tokens between two windows would be read
# by nothing, which is the failure this method exists to prevent.
check_raises("stride/past the effective window is an error", ValueError,
             lambda: scan(Q, stride=room_for(Q["dept"]) + 1))
# A stride the caller paired with a window they asked for is NOT their error when the library then
# reduces that window: window=90 with stride=90 is a self-consistent "no overlap, read everything".
# Refusing it made the library's own clamp look like the caller's mistake, and forced every such
# caller -- including `tests/test_onnx_long.py`'s window=96 stride=96 -- to be edited. It is clamped
# to half the effective window under the same RuntimeWarning, and the scan still covers the state.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _clamped = scan({"a": q_many(12)}, window=90, stride=90)
check_true("stride/a stride valid for the requested window is clamped, not refused",
           _clamped[0]["usage"]["windows"] > 1, _clamped[0]["usage"])
check_true("stride/and the caller is told the stride moved too",
           any("stride=90" in str(w.message) and "reduced" in str(w.message) for w in caught),
           [str(w.message)[:90] for w in caught])
_room12 = room_for(q_many(12))
# `read_spans` clamps every span to `start + room`, so `b - a <= room` is true of ANY input --
# executed against deliberate garbage it still passed. What the clamped scan has to guarantee is
# that nothing is left unread, which `uncovered` measures and which goes red when the stride clamp
# is mutated to `size * 2`.
check("stride/the clamped scan leaves no gap",
      uncovered(read_spans(_clamped[1], _room12), STATE_TOKENS), [])
# A stride past the window the caller actually asked for is still their error.
check_raises("stride/past the window the caller asked for is still refused", ValueError,
             lambda: scan({"a": q_many(12)}, window=40, stride=80))
check("stride/a stride inside the window is accepted",
      scan(Q, window=40, stride=40)[0]["usage"]["windows"] > 1, True)

# 12f-bis. the ONNX path honours the same cap, because README and Router promise it for both
try:                                                                            # noqa: E402
    from laya.common import window_batch_cap, window_budget
except ImportError:      # source reverted: let the checks below go red rather than abort the suite
    def window_budget(*_a, **_k):        # type: ignore[misc]
        return (0, 0, 0)

    def window_batch_cap(*_a, **_k):     # type: ignore[misc]
        return -1

_onnx = ONNXAgent.__new__(ONNXAgent)
_onnx.tok = _Tok()
_onnx.cfg = {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}
_q_onnx = {"a": q_many(12)}   # 100 options fills this tiny harness entirely
_eff_onnx, _step_onnx, _room_onnx = window_budget(
    _onnx.tok, [_onnx._to_internal(_q_onnx["a"])], MAX_LEN, HEAD_MAX_LEN)
check_true("onnx/predict_long caps the window at the room too",
           _eff_onnx <= _room_onnx and _eff_onnx < max(64, MAX_LEN - HEAD_MAX_LEN - 8),
           (_eff_onnx, _room_onnx))
check_true("onnx/and its stride stays inside that window", _step_onnx <= _eff_onnx,
           (_step_onnx, _eff_onnx))
check_true("onnx/predict_long actually calls window_budget",
           "window_budget(" in inspect.getsource(ONNXAgent.predict_long),
           "the README and Router docstring promise the cap for both agents")

# 12f-ter. capping the window must not turn one forward pass into an out-of-memory
check("batch/a mild cap keeps the single shared pass", window_batch_cap(10, 43, 64), None)
check("batch/no cap at all keeps it too", window_batch_cap(16, 312, 312), None)
check_true("batch/a blow-up is bounded to the un-capped pass width",
           0 < (window_batch_cap(413, 23, 312) or 0) < 413, window_batch_cap(413, 23, 312))
check("batch/an explicit batch_size is always honoured", window_batch_cap(413, 23, 312, 8), 8)

# 12g. the page that teaches the default teaches the cap as well
window_para = README[README.index("A smaller `window` isolates"):][:900]
check("docs/README says the window is capped at the room the questions leave",
      "capped at the room the questions leave" in window_para, True)
check("docs/predict_long's docstring documents the cap",
      "capped at the room the" in (Agent.predict_long.__doc__ or ""), True)

# 12h. "fits in one window" is the room the questions leave, not the default window. A plain call
# reads a state up to that room whole, so windowing one between the two only re-read it in pieces,
# and the max over windows moved answers `predict` had already given, at twice the forward passes.
ROOM_Q = min(room_for(q) for q in Q.values())
check("fits/these questions leave more room than the default window", ROOM_Q > CONFIG_BUDGET, True)
_pad = len(serialize_state({"body": ""}))
AT_ROOM, PAST_ROOM = {"body": "x" * (ROOM_Q - _pad)}, {"body": "x" * (ROOM_Q - _pad + 1)}
check("fits/the state under test is exactly the room",
      len(TOK(serialize_state(AT_ROOM))["input_ids"]), ROOM_Q)
a = make_agent(canned)
fits = a.predict_long(AT_ROOM, Q)
check("fits/a state at the room goes to system_one", fits["answers"], {"_via": "system_one"})
check("fits/and is not windowed", a._calls["batch_states"], None)
check("fits/one window is reported", fits["usage"].get("windows", "<absent>"), 1)
a = make_agent(canned)
a.predict_long(PAST_ROOM, Q)
check_true("fits/one token past the room is still scanned",
           len(a._calls["batch_states"] or []) > 1, a._calls["batch_states"])
a = make_agent(canned)
a.predict_long(AT_ROOM, Q, window=CONFIG_BUDGET)
check_true("fits/an explicit window still scans a state wider than it",
           len(a._calls["batch_states"] or []) > 1, a._calls["batch_states"])
# The one-pass branch still refuses a start hook that narrows the room under the state: sized at the
# default window, the check would pass and the tail of the state would be cut without a word.
a = make_real_agent()
_, narrowed = _attempt(lambda: a.predict_long(
    AT_ROOM, Q, on_predict_start=lambda ctx: setattr(ctx, "max_len", MAX_LEN - 1)))
check("fits/a hook that narrows the room under the state is refused", _kind(narrowed), "ValueError")
a = make_real_agent()
whole, exc = _attempt(lambda: a.predict_long(AT_ROOM, Q))
check("fits/unhooked, the real path reads it in one pass",
      (_kind(exc), a._forward_calls, ((whole or {}).get("usage") or {}).get("windows")),
      (None, [2], 1))


# --- findings from an adversarial review -----------------------------------------------------

# The batch cap is wired into BOTH agents and nothing pinned the wiring: replacing `cap` with
# `batch_size` at either call site, or swapping window_budget's two arguments so the cap can never
# fire, left every suite green. The cap is the OOM protection the change exists to keep.
_blow = make_agent(lambda states, questions: [
    {"model": "m", "answers": {"a": {"choice": "x", "answer_confidence": 0.5}},
     "usage": {"input_tokens": 1}} for _ in states])
_blow.predict_long(LONG, {"a": q_many(16)})   # room 24 vs a 72-token config budget: 3x blow-up
# The exact value, not "some int": returning a quarter of it, or moving _WINDOW_BATCH_BLOWUP,
# both left a plausible-looking number that an in-range assertion accepted.
check("batch cap/a blown-up scan is chunked to the un-capped scan's pass size",
      _blow._calls["batch_size"], 9)

_mild = make_agent(lambda states, questions: [
    {"model": "m", "answers": {"a": {"choice": "x", "answer_confidence": 0.5}},
     "usage": {"input_tokens": 1}} for _ in states])
_mild.predict_long(LONG, {"a": q_many(2)})
check("batch cap/a mild scan keeps the single shared pass", _mild._calls["batch_size"], None)

_explicit = make_agent(lambda states, questions: [
    {"model": "m", "answers": {"a": {"choice": "x", "answer_confidence": 0.5}},
     "usage": {"input_tokens": 1}} for _ in states])
_explicit.predict_long(LONG, {"a": q_many(16)}, batch_size=3)
check("batch cap/an explicit batch_size is honoured untouched",
      _explicit._calls["batch_size"], 3)

# Both agents must USE the cap, and the torch scan must install the budget guard. The fake agent
# replaces `predict_batch` wholesale, so no hook chain runs through it and behaviour cannot see the
# wiring; the source is what distinguishes "computed" from "computed and passed on".
_torch_src = inspect.getsource(Agent.predict_long)
_onnx_src = inspect.getsource(ONNXAgent.predict_long)
check_true("batch cap/the torch scan passes the cap to predict_batch",
           "batch_size=cap" in _torch_src, "")
check_true("batch cap/the ONNX scan passes the cap to predict_batch",
           "window_batch_cap(" in _onnx_src and "batch_size=cap" in _onnx_src, "")
# Compared on the statements, not on any mention: the comment above the call names `_to_internal`.
check_true("questions/the ONNX scan validates before _to_internal",
           _onnx_src.index("_Agent._check_question(qid")
           < _onnx_src.index("internal = {qid: self._to_internal"), "")


# A start hook may set ctx.max_len / ctx.head_max_len, or rewrite ctx.questions -- both documented
# powers -- and `build_sequence` uses whatever it finds, while the scan was sized from the config
# before any hook ran. `widen_for_high_cardinality`, which docs/hooks/patterns.md ships for exactly
# these questions, widens the head faster than max_len, so the real room SHRINKS: 303 sized against
# 253 real at 50 options on the English checkpoint, and 43.4% of a document unread at 100.
#
# The first version of this was a start hook that RAISED, and three ways of defeating it all left
# the suite green: `hooks_raise=False` -- what docs/hooks/tracing.md recommends -- downgraded the
# refusal to a RuntimeWarning and the scan proceeded; a hook that rewrote ctx.questions instead of
# the budget was invisible; and `sized=0` at the call site made it unable to fire while the
# source-text assertions still matched. It is now a plain function `predict_long` calls itself, so
# no hook policy governs it, and both halves are checked below: what it decides, and that it is
# actually called with the budget the scan used.
_CFG = (100, 20)              # make_agent's config; budget = max(64, 100-20-8) = 72


def _evidence(max_len=None, head_max_len=None, questions=None, answered=False):
    return {"answered": answered, "states": None, "max_len": max_len,
            "head_max_len": head_max_len,
            "questions": {"a": q_many(2)} if questions is None else questions}


_probe_agent = make_agent(lambda states, qs: [])


def _scan_raw(questions=None, **kw):
    a = make_agent(lambda states, qs: [])
    return a.predict_long(LONG, questions or {"a": q_many(2)}, **kw)


def _budget_message():
    try:
        _check_scan_budget(_probe_agent, _evidence(head_max_len=60), 72, *_CFG)
    except ValueError as exc:
        return str(exc)
    return ""


check_raises("hook budget/a hook that shrinks the room is refused", ValueError,
             lambda: _check_scan_budget(_probe_agent, _evidence(head_max_len=60), 72, *_CFG))
# `asked` is what predict_long was called with; the recorded questions are what the chain left.
check_raises("hook budget/rewriting ctx.questions is caught too", ValueError,
             lambda: _check_scan_budget(_probe_agent, _evidence(questions={"a": q_many(16)}),
                                        72, *_CFG, asked={"a": q_many(2)}))
# The boundary is `>=`, not `>`: a re-budget that leaves the room EXACTLY the size the scan used
# is harmless and must still answer. At cfg 100/20 a 4-option question leaves exactly 72, which is
# what a 2-option scan sized to, so this is the one shape that separates the two comparisons.
check_true("hook budget/a room exactly equal to the window is allowed",
           _check_scan_budget(_probe_agent, _evidence(questions={"a": q_many(4)}), 72, *_CFG,
                              asked={"a": q_many(2)}) is None, "room == sized")

check_true("hook budget/questions untouched skips the recompute entirely",
           _check_scan_budget(_probe_agent, _evidence(questions={"a": q_many(16)}), 72, *_CFG,
                              asked={"a": q_many(16)}) is None, "nothing moved")
check_true("hook budget/an unchanged budget is allowed",
           _check_scan_budget(_probe_agent, _evidence(), 72, *_CFG) is None, "no hook")
check_true("hook budget/a budget widened together is allowed",
           _check_scan_budget(_probe_agent, _evidence(400, 60), 72, *_CFG) is None, "room grows")
check_true("hook budget/a hook that answered the document is not refused",
           _check_scan_budget(_probe_agent, _evidence(head_max_len=60, answered=True),
                              72, *_CFG) is None, "cache hit")
check_true("hook budget/no questions is not a crash",
           _check_scan_budget(_probe_agent, _evidence(head_max_len=60, questions={}),
                              72, *_CFG) is None, "empty questions")
check_true("hook budget/the refusal names both budgets and both rooms",
           all(t in _budget_message() for t in ("max_len=100", "head_max_len=60", "sized for 72")),
           _budget_message())

# ... and that predict_long actually calls it, with the budget it sized the windows to. This is what
# `sized=0` walked past: the symbol was present and the call site spelled right, but the value made
# the check inert.
_seen_budget = []
_real_check = agent_mod._check_scan_budget
try:
    agent_mod._check_scan_budget = (
        lambda ag, ev, sized, ml, hm, asked=None: _seen_budget.append((sized, ml, hm)))
    _wired = make_agent(lambda states, qs: [
        {"model": "m", "answers": {"a": {"choice": "x", "answer_confidence": 0.5}},
         "usage": {"input_tokens": 1}} for _ in states])
    _wired.predict_long(LONG, {"a": q_many(2)})
finally:
    agent_mod._check_scan_budget = _real_check
check("hook budget/the torch scan checks its budget exactly once", len(_seen_budget), 1)
# The room is the SMALLEST any question leaves, which is why `window_budget` takes a min. No test
# put two questions of different cardinality through the checker, so `min` -> `max` was green.
# At the unchanged budget these two leave rooms of 76 and 24 against a scan sized for 72, so `min`
# refuses and `max` does not -- which is the whole reason `window_budget` takes a min. Collapsing
# them under a widened head (both 36) makes the mutant survive, which is how it slipped through.
check_raises("hook budget/the smallest room wins across questions", ValueError,
             lambda: _check_scan_budget(
                 _probe_agent, _evidence(questions={"a": q_many(2), "b": q_many(16)}), 72, *_CFG,
                 asked={"a": q_many(2)}))


# A hook that widens a question IN PLACE is the case the guard was written for, and the one a
# shallow snapshot missed. `predict_long` has to compare against the questions as they were when it
# sized the scan, not against the caller's mapping, which the hook mutates too: with the same object
# on both sides `questions == asked` stayed True, the checker returned early, and the scan proceeded
# with windows `build_sequence` would re-truncate -- silently worse than not windowing at all.
# `widen_for_high_cardinality` (docs/hooks/patterns.md) is exactly this shape. The questions are a
# fresh mapping because the hook mutates what it is handed.
_inplace_questions = {"dept": {"type": "choice", "instructions": "?",
                               "criteria": {"a": "x", "b": "y"}},
                      "flag": dict(Q["flag"])}


def _widen_in_place(ctx):
    ctx.questions["dept"]["criteria"].update(q_many(16)["criteria"])


_widened, _widen_exc = _attempt(
    lambda: make_real_agent().predict_long(LONG, _inplace_questions,
                                           on_predict_start=_widen_in_place))
check("hook budget/an in-place question rewrite is refused", _kind(_widen_exc), "ValueError")

# What the probe RECORDS is half the check: synthesising an evidence dict in the tests above leaves
# the recording itself unpinned, and dropping either field made the checker silently inert.
_rec_probe, _rec = _start_evidence()


class _RecCtx:
    results = None
    states = ["s"]
    max_len = 400
    head_max_len = 60
    questions = {"a": q_many(2)}


_rec_probe(_RecCtx())
check("probe/records the budget in force", (_rec["max_len"], _rec["head_max_len"]), (400, 60))
check("probe/records the questions in force", list(_rec["questions"]), ["a"])
check_true("probe/copies the questions rather than aliasing them",
           _rec["questions"] is not _RecCtx.questions, "")

# Validation before `_to_internal` on the torch agent too. The ONNX side is pinned by a source-order
# check; deleting the torch loop was green.
check_raises("questions/the torch scan validates before _to_internal", ValueError,
             lambda: _scan_raw(questions={"a": {"type": "choice", "instructions": "?",
                                                "criteria": None}}))

# Both agents, not one. The ONNX scan had no check at all while the torch one refused the identical
# input -- measured, 34.6% of a document reaching no model -- and the source comment two screens up
# says the contract binds both agents.
_onnx_src = inspect.getsource(ONNXAgent.predict_long)
check_true("hook budget/the ONNX scan checks its budget too",
           "_check_scan_budget(self, evidence, budget, max_len, head_max_len, asked)" in _onnx_src,
           "")
check("hook budget/both agents check it on the single-window path too",
      (inspect.getsource(Agent.predict_long).count("_check_scan_budget("),
       _onnx_src.count("_check_scan_budget(")), (2, 2))

check("hook budget/it is handed the window size and the config it sized from",
      _seen_budget[0] if _seen_budget else None,
      (window_budget(TOK, [Agent._to_internal(q_many(2))], 100, 20,
                     window=None, stride=None)[0], 100, 20))



# A question whose options fill the whole sequence leaves no room for any state, so no window can
# carry a single token of the document. `window_budget` refuses rather than scanning with window=0,
# and deleting that raise left all 149 checks green -- the scan then ran with a window of 0 and
# reported spans for text no model had seen.
_no_room = {"a": q_many(400)}
check_true("no room/the fixture really does fill the sequence",
           room_for(q_many(400)) <= 0,
           "room=%d" % room_for(q_many(400)))
check_raises("no room/a scan with no room for the state is refused", ValueError,
             lambda: window_budget(TOK, [Agent._to_internal(_no_room["a"])], MAX_LEN, HEAD_MAX_LEN,
                                   window=None, stride=None))
check_raises("no room/and predict_long refuses it rather than scanning a zero-width window",
             ValueError, lambda: scan(_no_room))
_no_room_msg = _attempt(lambda: window_budget(TOK, [Agent._to_internal(_no_room["a"])],
                                              MAX_LEN, HEAD_MAX_LEN, window=None, stride=None))
check_true("no room/the refusal names the budget that caused it",
           "max_len=%d" % MAX_LEN in str(_no_room_msg) and "no room for the state" in str(_no_room_msg),
           str(_no_room_msg)[:120])
# And one option fewer still scans, so this pins the boundary rather than "big questions fail".
check_true("no room/a question that leaves room is still scanned",
           room_for(q_many(2)) > 0 and len(scan({"a": q_many(2)})[1]) > 0, "")


# ------------------------------------------- a hard clamp of the DEFAULT window must not be silent
# Capping the window at the room is what stops the tail of every window reaching no model, but it is
# not free: the scan needs about `requested / room` times as many windows, each a full forward pass.
# Measured on the English checkpoint with 100 four-word options -- room 102 of max_len 512, window
# 312 -> 102, 11 windows -> 36, 1902 ms -> 5858 ms. Nothing in the caller's code implies that, so
# `window_budget` says so. Only on a HARD clamp: warning about every small one would be noise, and
# noise is how a warning that matters gets filtered out.
check_true("clamp warning/the threshold is a ratio above 1 and not absurd",
           1 < common_mod._WINDOW_CLAMP_WARN_RATIO <= 4, common_mod._WINDOW_CLAMP_WARN_RATIO)

_default_window = max(64, MAX_LEN - HEAD_MAX_LEN - 8)


def _clamp_warnings(qdef, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        window_budget(TOK, [Agent._to_internal(qdef)], MAX_LEN, HEAD_MAX_LEN,
                      window=kw.get("window"), stride=kw.get("stride"))
    return [str(c.message) for c in caught]


# A question whose head leaves less than half the default window: found by search rather than by
# hard-coding an option count, so a change to the fake tokenizer cannot quietly stop this reaching
# the branch.
_hard = next((n for n in range(2, 400)
              if 0 < room_for(q_many(n)) * common_mod._WINDOW_CLAMP_WARN_RATIO <= _default_window),
             None)
check_true("clamp warning/a hard-clamp fixture exists", _hard is not None, "")
# ... and one whose clamp is mild or absent, which must stay silent.
_mild = next((n for n in range(2, 400) if room_for(q_many(n)) >= _default_window), None)
check_true("clamp warning/a mild fixture exists", _mild is not None, "")

if _hard is not None and _mild is not None:
    _hard_msgs = _clamp_warnings(q_many(_hard))
    check("clamp warning/fires once on a hard clamp of the default window", len(_hard_msgs), 1)
    _m = _hard_msgs[0] if _hard_msgs else ""
    check_true("clamp warning/names the room it was cut to", str(room_for(q_many(_hard))) in _m, _m)
    check_true("clamp warning/names max_len", "max_len=%d" % MAX_LEN in _m, _m)
    check_true("clamp warning/names how much more scanning it costs", "as many windows" in _m, _m)
    check_true("clamp warning/points at the tool for a large label set",
               "predict_shortlist" in _m, _m)

    check("clamp warning/stays silent when the clamp is mild or absent",
          _clamp_warnings(q_many(_mild)), [])

    # An explicit `window=` keeps its own message and does not also get this one -- the caller who
    # named a width is told their width was reduced, which is a different statement.
    _explicit = _clamp_warnings(q_many(_hard), window=_default_window)
    check("clamp warning/an explicit window gets exactly one message", len(_explicit), 1)
    check_true("clamp warning/and it is the explicit-window one",
               _explicit and "is wider than the" in _explicit[0], _explicit)

# The refusal for a question with no room at all points at the same remedy.
_no_room_msg = _attempt(lambda: window_budget(TOK, [Agent._to_internal(q_many(400))],
                                              MAX_LEN, HEAD_MAX_LEN, window=None, stride=None))
check_true("clamp warning/the no-room refusal names predict_shortlist too",
           "predict_shortlist" in str(_no_room_msg), str(_no_room_msg)[:140])



# ---------------------------------------------------------------------------------------------
# The scan is laya's own forward pass, not a caller's hook
#
# `Router.predict_long` used to append `_ScanLong` to the start-hook chain, so the scan executed
# inside `dispatch()` and inherited the caller's hook machinery. Four consequences, all pinned here,
# none of which any check in this repo caught before: the scan was bounded by `hooks_timeout`, it
# held the hooks lock for its whole duration, it ran before `_SKIP_DEFAULTS` was entered so every
# process-wide default hook fired twice, and an error from it was swallowed
# under `hooks_raise=False`.
import time as _time  # noqa: E402

from laya.hooks import PredictContext, compose_hooks, dispatch, set_default_hooks  # noqa: E402


class _SlowScan:
    """Both entry points, so a test can tell which one answered, and a scan that takes real time."""

    def __init__(self, seconds=0.30):
        self.seconds = seconds
        self.scans = 0
        self.singles = 0

    def system_one(self, state, questions, **controls):
        self.singles += 1
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {"input_tokens": 1}}

    def predict_long(self, state, questions, **controls):
        self.scans += 1
        _time.sleep(self.seconds)
        return {"model": "stub", "answers": {"via": "predict_long"},
                "usage": {"input_tokens": 7, "windows": 3}}


# 1. a hook timeout must not abort the scan -- it bounds the caller's hooks, not laya's inference
_slow = _SlowScan(0.30)
_timed = Router(hooks_timeout=0.05)
_timed.attach("english", _slow)
_out, _exc = _attempt(lambda: _timed.predict_long(LONG, Q, model="english"))
check_true("scan/a hook timeout does not abort it",
           _exc is None and _out is not None and _out["answers"]["via"] == "predict_long",
           "exc=%r answers=%r" % (_exc, (_out or {}).get("answers")))
check("scan/the agent really scanned once", (_slow.scans, _slow.singles), (1, 0))


# 2. the hooks lock must not be held across the scan: `hooks_concurrent=False` serialises each hook,
#    not whole calls (docs/hooks/lifecycle.md). Measured as the longest single hold, so no timing
#    threshold is involved -- a lock held across a 0.30s scan cannot report a hold near zero.
class _TimedLock:
    def __init__(self):
        self._lock = threading.RLock()
        self.longest = 0.0

    def __enter__(self):
        self._entered = _time.perf_counter()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self.longest = max(self.longest, _time.perf_counter() - self._entered)
        self._lock.release()
        return False


_lock_stub = _SlowScan(0.30)
_locked = _router(_lock_stub)
_locked._hooks_lock = _TimedLock()
_t0 = _time.perf_counter()
_locked.predict_long(LONG, Q, model="english")
_wall = _time.perf_counter() - _t0
check_true("scan/the hooks lock is not held across it",
           _lock_stub.scans == 1 and _locked._hooks_lock.longest < _wall / 2,
           "longest hold %.3fs of a %.3fs call" % (_locked._hooks_lock.longest, _wall))


# 3. process-wide default hooks fire once. The scan reached the agent before `_SKIP_DEFAULTS` was
#    set, so `compose_hooks` re-added every `set_default_hooks` hook and the chain ran
#    twice. Measured
#    on real weights, `usage` did NOT double -- `input_tokens` is identical on both trees -- so what
#    this pins is the hook firing, and nothing else.
_fired = {"start": 0, "end": 0, "shapes": []}


class _CountDefaults:
    def on_predict_start(self, ctx):
        _fired["start"] += 1
        _fired["shapes"].append(len(ctx.states))     # 1 = the document, >1 = the windows

    def on_predict_end(self, ctx):
        _fired["end"] += 1


class _ComposingScan:
    """Composes its own hook list and dispatches, the way `Agent.predict_long` does.

    A plain stub cannot pin this. `_SKIP_DEFAULTS` is read in exactly one place --
    `laya.hooks.compose_hooks` -- so an agent that never composes fires the process-wide defaults
    once however the flag is set, and the check below would pass just as happily with the fix
    reverted. Verified: with a non-composing stub, setting `_SKIP_DEFAULTS` to False instead of True
    left this check green, which is the whole reason it is written this way.
    """

    def _answer(self, states, questions):
        active = compose_hooks([])
        ctx = PredictContext(states=list(states), questions=questions)
        dispatch(active, "on_predict_start", ctx, raise_errors=True)
        result = {"model": "stub", "answers": {}, "usage": {"input_tokens": 5}}
        ctx.results = [result]
        dispatch(active, "on_predict_end", ctx, raise_errors=True)
        return result

    def system_one(self, state, questions, **controls):
        return self._answer([state], questions)

    def predict_long(self, state, questions, **controls):
        # `Agent.predict_long` composes and dispatches over the WINDOWS it cut, not over the
        # document, so the stub does the same: three windows here. A dict or list state is passed
        # through unsliced -- the window COUNT is what the check reads.
        text = state if isinstance(state, str) else str(state)
        windows = [text[:40], text[30:70], text[60:]]
        return self._answer(windows, questions)


set_default_hooks([_CountDefaults()])
try:
    _usage = _router(_ComposingScan()).predict_long(LONG, Q, model="english")["usage"]
finally:
    set_default_hooks([])
check("scan/a default hook fires once, not twice", (_fired["start"], _fired["end"]), (1, 1))
# The count alone does not say WHICH dispatch was dropped. `main` fired for the document and then
# again for the agent's own windows; the one that survives must be the document, because that is
# the request the caller made -- and because a hook that mutates state would otherwise be applied
# per window as well, which measurably moves a scan's answers on real weights.
check("scan/the surviving dispatch is the document, not the agent's windows",
      _fired["shapes"], [1])
# Not a pin for this change -- it passes with the scan back inside the hook chain --
# but a guard that
# the double dispatch never starts double-counting, which is what the first version of this comment
# wrongly claimed it already did.
check("scan/usage is still counted once (regression guard, not a fix pin)",
      _usage["input_tokens"], 5)


# 4. an agent that cannot scan is refused whatever `hooks_raise` says. As a hook this was swallowed
#    under `hooks_raise=False` and `system_one` answered one window instead -- a different question
#    than the caller asked, reported as success.
class _NoScanBoth:
    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {}}


for _policy in (True, False):
    _r = Router(hooks_raise=_policy)
    _r.attach("english", _NoScanBoth())
    _out, _exc = _attempt(lambda r=_r: r.predict_long(LONG, Q, model="english"))
    # The MESSAGE, not just the type: `getattr(agent, "predict_long", None)` is None for this
    # agent, so a missing guard would call None and raise its own `TypeError` ('NoneType' object is
    # not callable). Asserting the class alone let that mutant live.
    check("scan/no predict_long is refused with hooks_raise=%s" % _policy,
          (_kind(_exc), "has no predict_long" in str(_exc), "_NoScanBoth" in str(_exc)),
          ("TypeError", True, True))


# 5. everything the hook ordering used to give must still hold: the scan runs last, so a caller's
#    hook that answered wins and one that rewrote the request is what gets scanned.
class _Skipper:
    def on_predict_start(self, ctx):
        ctx.skip([{"model": "cached", "answers": {"via": "hook"}, "usage": {}}])


_skipped = _SlowScan(0.0)
_out = _router(_skipped, hooks=[_Skipper()]).predict_long(LONG, Q, model="english")
check("scan/a hook that answered still wins", _out["answers"], {"via": "hook"})
check("scan/the agent was not scanned at all", (_skipped.scans, _skipped.singles), (0, 0))


class _RewriteQuestions:
    def on_predict_start(self, ctx):
        ctx.questions = {"rewritten": {"type": "noul", "instructions": "?"}}


_rewritten = _LongStub()
_router(_rewritten, hooks=[_RewriteQuestions()]).predict_long(LONG, Q, model="english")
check("scan/a hook's rewritten questions are what is scanned",
      sorted(_rewritten.calls[0]["questions"]), ["rewritten"])


# 6. the scan must not leak into a `predict` that a hook calls back into. It travels
#    on a contextvar,
#    and `predict` clears it on read, so the nested call answers with `system_one`.
_nested = {"answers": None}


class _ReenterPredict:
    def __init__(self):
        self.entered = False

    def on_predict_start(self, ctx):
        # Guard BEFORE the nested call: this hook is installed on the router, so the inner `predict`
        # fires it again, and a guard set afterwards recurses until the stack runs out.
        if self.entered:
            return
        self.entered = True
        _nested["answers"] = ctx.router.predict("short", Q, model="english")["answers"]


_leak = _SlowScan(0.0)
_outer = _router(_leak, hooks=[_ReenterPredict()]).predict_long(LONG, Q, model="english")
check("scan/does not leak into a nested predict from a hook", _nested["answers"],
      {"via": "system_one"})
check("scan/the outer call still scanned", _outer["answers"], {"via": "predict_long"})
check("scan/one scan and one single-window call", (_leak.scans, _leak.singles), (1, 1))


# 6b. and an early failure inside `predict` must not leave the scan set for the next call.
#     `predict` validates `state`/`questions` before it consumes the scan, so without the reset in
#     `predict_long` the contextvar survives the raise and the NEXT plain `predict` in this context
#     would scan a document nobody asked it to scan.
_after_raise = _SlowScan(0.0)
_raiser = _router(_after_raise)
_, _early_exc = _attempt(lambda: _raiser.predict_long(None, Q, model="english"))
check_true("scan/an early failure still raises", isinstance(_early_exc, TypeError),
           "got %r" % (_early_exc,))
_raiser.predict("short state", Q, model="english")
check("scan/an early failure leaves no scan behind for the next call",
      (_after_raise.scans, _after_raise.singles), (0, 1))


# 6c. every window option reaches the agent, including `aggregate` -- which nothing in this repo
#     asserted through the Router, so forwarding a constant instead of it was a mutation the whole
#     suite missed. The value asserted here is deliberately NOT "auto": "auto" is both `_ScanLong`'s
#     default and the stub's own, so asserting it could not tell forwarding from a hardcoded
#     constant. The stub takes any string; `Agent.predict_long` refuses anything but "auto", and
#     that refusal is the second half of the evidence.
_agg = _LongStub()
_router(_agg).predict_long(LONG, Q, model="english", aggregate="mean")
check("scan/aggregate is forwarded, not defaulted", _agg.calls[0]["aggregate"], "mean")
check_raises("scan/an unsupported aggregate is refused by the agent, not swallowed", ValueError,
             lambda: _router(make_real_agent()).predict_long(LONG, Q, model="english",
                                                             aggregate="mean"))


# 6d. the scan reads `ctx` at the moment it runs, which is what makes it equivalent to the hook it
#     replaced. A start hook can still swap the agent, rewrite the detected language,
#     and have its own
#     `routing` survive -- three things that broke when the scan read `predict`'s locals instead.
class _SwapAgent:
    def __init__(self, replacement):
        self.replacement = replacement

    def on_predict_start(self, ctx):
        ctx.agent = self.replacement


_routed_agent, _swapped_agent = _LongStub(), _LongStub()
_router(_routed_agent, hooks=[_SwapAgent(_swapped_agent)]).predict_long(LONG, Q, model="english")
check("scan/a hook that swaps ctx.agent is what gets scanned",
      (len(_routed_agent.calls), len(_swapped_agent.calls)), (0, 1))


class _SetDetectedLanguage:
    def on_predict_start(self, ctx):
        ctx.decision["detection"] = {"language": "ja"}


_relang = _LongStub()
_router(_relang, hooks=[_SetDetectedLanguage()]).predict_long(LONG, Q, model="english")
check("scan/a hook's detected language reaches the scan", _relang.calls[0]["lang"], "ja")


class _OwnRouting:
    def predict_long(self, state, questions, **controls):
        return {"model": "stub", "answers": {}, "usage": {},
                "routing": {"model": "the agent's own"}}

    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {}, "usage": {}}


_own = _router(_OwnRouting()).predict_long(LONG, Q, model="english")
check("scan/an agent's own routing key is preserved", _own["routing"], {"model": "the agent's own"})


# 6e. an agent whose `predict_long` has no `lang` parameter. Whether to pass `lang` is read from
#     the signature, never from catching a `TypeError`: on `main` this agent raised under the
#     default `hooks_raise=True` and was answered by `system_one` on one window under
#     `hooks_raise=False` -- symptom 4 -- so there is no prior behaviour to preserve by retrying,
#     and a retry cannot tell this case from the one below.
# Resolved defensively so reverting the fix fails these checks rather than aborting the suite on
# an ImportError, which would report nothing at all.
import laya.router as _router_mod  # noqa: E402

_takes_lang = getattr(_router_mod, "_takes_lang", None)


def _lang_check(fn):
    """`_attempt`-style, because one of these inputs raises from its own `__signature__`: a
    narrowed `except` in `_takes_lang` must turn these checks red, not abort the module and take
    the sixteen checks after them with it."""
    if _takes_lang is None:
        return "laya.router has no _takes_lang"
    out, exc = _attempt(lambda: _takes_lang(fn))
    return out if exc is None else "raised %s" % _kind(exc)


class _NoLangKwarg:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {}}

    def predict_long(self, state, questions, window=None, stride=None, aggregate="auto",
                     batch_size=None):
        self.calls += 1
        return {"model": "stub", "answers": {"via": "predict_long"}, "usage": {}}


for _policy in (True, False):
    _old = _NoLangKwarg()
    _r = Router(hooks_raise=_policy)
    _r.attach("english", _old)
    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always")
        _out, _exc = _attempt(lambda r=_r: r.predict_long(LONG, Q, model="english", lang="de"))
    check("scan/an agent without a lang parameter is scanned once, not retried (hooks_raise=%s)"
          % _policy,
          (_exc is None, (_out or {}).get("answers"), _old.calls),
          (True, {"via": "predict_long"}, 1))
    check("scan/dropping the caller's lang is reported (hooks_raise=%s)" % _policy,
          [w.category.__name__ for w in _caught], ["RuntimeWarning"])
    # `_msg` rather than `_caught[0]`: with the fix reverted nothing warns, and indexing an empty
    # list would abort the suite instead of failing this check.
    _msg = str(_caught[0].message) if _caught else ""
    check("scan/the warning names the language that did not reach the agent (hooks_raise=%s)"
          % _policy,
          ("requested" in _msg, "'de'" in _msg), (True, True))

# and with no language to pass, there is nothing to report
_quiet = _NoLangKwarg()
_qr = Router()
_qr.attach("english", _quiet)
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    # `_attempt`, because with the fix reverted this call raises (`main` passes `lang=None`
    # positionally into an agent that has no such parameter) and an abort reports nothing.
    _, _quiet_exc = _attempt(lambda: _qr.predict_long(LONG, Q, model="english"))
check("scan/no warning when there was no language to pass",
      (_kind(_quiet_exc), [w.category.__name__ for w in _caught], _quiet.calls), (None, [], 1))


# 6f. the case a `TypeError` retry would have swallowed: an agent that DOES take `lang` and raises
#     `TypeError: ... unexpected keyword argument 'lang'` from somewhere deeper inside its own scan.
#     Matching on the message could not tell this from 6e, so it re-ran the whole scan with `lang`
#     dropped and -- when the inner failure was conditional on `lang` -- answered as if the caller
#     had never asked for a language. It must raise, and the scan must run exactly once.
class _InnerLangFailure:
    def __init__(self):
        self.langs = []

    @staticmethod
    def _helper(state):                     # the helper the lang-aware path forgot to update
        return state

    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {}}

    def predict_long(self, state, questions, window=None, stride=None, aggregate="auto",
                     batch_size=None, lang=None):
        self.langs.append(lang)
        if lang is not None:
            self._helper(state, lang=lang)  # raises TypeError naming 'lang'
        return {"model": "stub", "answers": {"via": "predict_long", "lang": lang}, "usage": {}}


_inner = _InnerLangFailure()
_ir = Router()
_ir.attach("english", _inner)
_out, _exc = _attempt(lambda: _ir.predict_long(LONG, Q, model="english", lang="de"))
check("scan/an error from inside the agent's own scan is not retried away",
      (_kind(_exc), (_out or {}).get("answers"), _inner.langs),
      ("TypeError", None, ["de"]))

# the signature check itself: a `**kwargs` forwarder takes `lang`, and an entry point `inspect`
# cannot read is given it rather than silently losing it (`min` stands in for any C callable).
check("scan/_takes_lang: an explicit parameter",
      _lang_check(lambda state, questions, lang=None: None), True)
check("scan/_takes_lang: a **kwargs forwarder counts",
      _lang_check(lambda state, questions, **kw: None), True)
check("scan/_takes_lang: no lang parameter", _lang_check(lambda state, questions: None), False)
# The KIND matters, not just the name: neither of these can be given `lang=` as a keyword, so
# claiming they take one would pass an argument that raises `TypeError` on arrival.
def _posonly(state, questions, lang, /):       # PEP 570: `lang` cannot be passed by keyword
    return None


check("scan/_takes_lang: a positional-only lang is not a keyword", _lang_check(_posonly), False)
check("scan/_takes_lang: a *lang var-positional is not a keyword",
      _lang_check(lambda state, questions, *lang: None), False)
check("scan/_takes_lang: an unintrospectable callable is given the argument",
      _lang_check(min), True)


class _HostileSignature:
    """`inspect.signature` reads `__signature__`, which is arbitrary code. A deployer's agent must
    not become uncallable because that code raises something other than TypeError/ValueError --
    `main` never introspected at all, so the no-change answer is to pass the argument."""

    @property
    def __signature__(self):
        raise KeyError("/srv/laya/secrets/key.pem")

    def __call__(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "predict_long"}, "usage": {}}


check("scan/_takes_lang: a __signature__ that raises anything is not fatal",
      _lang_check(_HostileSignature()), True)


class _InterruptingSignature:
    """`except Exception`, deliberately not `except BaseException`: a deployer's `__signature__`
    must not be able to swallow Ctrl-C or a `SystemExit` on its way past."""

    @property
    def __signature__(self):
        raise KeyboardInterrupt()

    def __call__(self, state, questions, **controls):
        return {"model": "stub", "answers": {}, "usage": {}}


check("scan/_takes_lang: a BaseException from __signature__ still propagates",
      _lang_check(_InterruptingSignature()), "raised KeyboardInterrupt")


# The third cell of the warning table, and the one that was missing: a DETECTED language, with no
# explicit `lang=` anywhere. Two mutants lived in the gap -- warning only for an explicit `lang`
# (so a detected one was dropped in silence, which is the whole failure the warning exists for),
# and hardcoding the word "requested" (so the message would name a language the caller never
# asked for). Routing has to run for real here, which is why there is no `model=`.
class _NoLangMultilingual:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {}}

    def predict_long(self, state, questions, window=None, stride=None, aggregate="auto",
                     batch_size=None):
        self.calls += 1
        return {"model": "stub", "answers": {"via": "predict_long"}, "usage": {}}


# Its own sample rather than `_german` below, which is defined further down the file.
_german_doc = "Wir wurden zweimal belastet und moechten eine Rueckerstattung erhalten. " * 8
_detected_drop = _NoLangMultilingual()
_dd_router = Router()
_dd_router.attach("multilingual", _detected_drop)
_dd_router.attach("english", _LongStub())
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    # `_attempt` here and below: with the fix reverted these calls raise (`main` passes `lang=`
    # unconditionally), and an abort would report nothing at all.
    _dd_out, _dd_exc = _attempt(lambda: _dd_router.predict_long(_german_doc, Q))
_dd_out = _dd_out or {"answers": None, "routing": {}}
_dd_msg = str(_caught[0].message) if _caught else ""
check("scan/a DETECTED language that cannot be passed is reported too",
      ([w.category.__name__ for w in _caught], _detected_drop.calls), (["RuntimeWarning"], 1))
check("scan/the warning calls a detected language detected, not requested",
      ("detected" in _dd_msg, "requested" in _dd_msg, "'de'" in _dd_msg), (True, False, True))
check("scan/the warning names the agent that could not take it",
      "_NoLangMultilingual" in _dd_msg, True)
check("scan/a short language is not marked as truncated", "..." in _dd_msg, False)
for _n, _want_ellipsis in ((31, False), (32, False), (33, True)):
    _edge = _NoLangKwarg()
    _er = Router()
    _er.attach("english", _edge)
    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always")
        _attempt(lambda n=_n: _er.predict_long(LONG, Q, model="english", lang="x" * n))
    _e_msg = str(_caught[0].message) if _caught else ""
    # At exactly 32 every character is present, so claiming a truncation would be the message
    # lying about its own handling -- the boundary `>` guards and `>=` would get wrong.
    check("scan/a %d-character language claims truncation: %s" % (_n, _want_ellipsis),
          ("..." in _e_msg, _edge.calls), (_want_ellipsis, 1))

# The language goes into a log line, and `serve.py` caps the body size but not this field, so the
# message truncates it: `%r` of a megabyte of `lang` would otherwise be six megabytes of warning.
_long_lang = _NoLangKwarg()
_llr = Router()
_llr.attach("english", _long_lang)
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    _attempt(lambda: _llr.predict_long(LONG, Q, model="english", lang="de" + "A" * 5000))
_ll_msg = str(_caught[0].message) if _caught else ""
check("scan/a hostile language value cannot amplify the warning",
      (len(_ll_msg) < 200, "..." in _ll_msg, "A" * 64 in _ll_msg, _long_lang.calls),
      (True, True, False, 1))
# and the cap is exactly 32 characters of the value, not merely "some cap": a band would let 33 or
# 40 through, and the sliced prefix is what says where the cut fell.
check("scan/the warning carries exactly the first 32 characters",
      repr("de" + "A" * 30) + "..." in _ll_msg, True)


# The cap slices the VALUE, not the rendered text, and names a non-`str` by its type instead of
# rendering it: a `str` subclass can report `len() == 2` and a megabyte of `__repr__`, and a
# non-`str` object's `__repr__` is caller code laya has no business running at all.
class _SneakyStr(str):
    def __repr__(self):
        return "'" + "A" * 100000 + "'"


class _SneakySlice(str):
    """`type(lang) is str`, not `isinstance`: slicing a subclass runs the subclass's own
    `__getitem__`, so an `isinstance` test would hand the slice straight back to caller code."""

    def __getitem__(self, key):
        return _WatchedRepr()


class _WatchedRepr:
    rendered = 0

    def __repr__(self):
        type(self).rendered += 1
        return "<" + "B" * 100000 + ">"


for _label, _hostile in (("str subclass with a huge repr", _SneakyStr("de")),
                         ("str subclass with its own __getitem__", _SneakySlice("de")),
                         ("a non-str list", ["C" * 20000]),
                         ("a non-str object", _WatchedRepr())):
    _hl = _NoLangKwarg()
    _hr = Router()
    _hr.attach("english", _hl)
    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always")
        _attempt(lambda h=_hostile: _hr.predict_long(LONG, Q, model="english", lang=h))
    _h_msg = str(_caught[0].message) if _caught else ""
    check("scan/%s cannot amplify the warning either" % _label,
          (len(_h_msg) < 200, _hl.calls), (True, 1))
check("scan/a non-str language is a constant, never rendered",
      (_WatchedRepr.rendered, "<not a string>" in _h_msg), (0, True))


# `type(lang).__name__` was the next thing along, and it is caller-reachable too: a writable slot
# on any heap type, and a metaclass can make it a property that returns a megabyte, returns an
# object whose `__str__` then runs, or raises and fails the whole call. The branch is a constant
# for exactly that reason, so none of these four can reach the message.
_name_ran = []


class _StringyName:
    def __str__(self):
        _name_ran.append(1)
        return "C" * 20000


class _MetaBig(type):
    @property
    def __name__(cls):
        return "B" * 20000


class _MetaObj(type):
    @property
    def __name__(cls):
        return _StringyName()


class _MetaRaise(type):
    @property
    def __name__(cls):
        raise RuntimeError("boom from __name__")


class _BigNameSlot:
    pass


_BigNameSlot.__name__ = "A" * 20000


class _PropName(metaclass=_MetaBig):
    pass


class _ObjName(metaclass=_MetaObj):
    pass


class _RaiseName(metaclass=_MetaRaise):
    pass


for _label, _hostile in (("a writable 1MB __name__", _BigNameSlot()),
                         ("a metaclass __name__ property", _PropName()),
                         ("a metaclass __name__ returning an object", _ObjName()),
                         ("a metaclass __name__ that raises", _RaiseName())):
    _nl = _NoLangKwarg()
    _nr = Router()
    _nr.attach("english", _nl)
    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always")
        _, _n_exc = _attempt(lambda h=_hostile: _nr.predict_long(LONG, Q, model="english", lang=h))
    _n_msg = str(_caught[0].message) if _caught else ""
    # `"<not a string>" in _n_msg` as well as the length: `len("") < 200` is true, so a mutant
    # that stops warning for a non-`str` at all would otherwise leave every one of these green.
    check("scan/%s cannot reach the warning" % _label,
          (_kind(_n_exc), len(_n_msg) < 200, "<not a string>" in _n_msg, _nl.calls),
          (None, True, True, 1))
check("scan/no __name__ code ran while building the warning", _name_ran, [])
check("scan/the document was still scanned, on the routed checkpoint",
      (_kind(_dd_exc), _dd_out["answers"], _dd_out.get("routing", {}).get("model")),
      (None, {"via": "predict_long"}, "multilingual"))


# The warning has to reach the CALLER, not just be emitted: `warnings.warn` is called three frames
# below `predict_long`, and the default "once per location" filter keys its registry on the frame
# `stacklevel` selects. Attributed to this file, two call sites warn twice; attributed to
# `router.py`, they collapse to one and every later call site in the process is silent.
_attr = _NoLangKwarg()
_ar = Router()
_ar.attach("english", _attr)


def _warn_site():
    """Calls `predict_long` DIRECTLY, so the frame distance is the real one.

    Not `_attempt(lambda: ...)`: that inserts the lambda and `_attempt` between this line and
    `predict_long`, which leaves a `stacklevel` two too large still landing in this file on a
    distinct line -- which is why the check below asserts the LINE, and why asserting only the file
    and the dedup behaviour was not enough. The `try` keeps a revert from aborting the module,
    exactly as `_attempt` would.
    """
    try:
        return _ar.predict_long(LONG, Q, model="english", lang="de")
    except TypeError:
        return None


# Located by reading the function's own source, not by a hand-counted offset that editing the
# docstring above would silently break.
_site_lines, _site_start = inspect.getsourcelines(_warn_site)
_want_line = _site_start + next((i for i, line in enumerate(_site_lines)
                                 if "_ar.predict_long(" in line), -1)
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    _warn_site()
check("scan/the warning points at the predict_long call's own line",
      (os.path.basename(_caught[0].filename), _caught[0].lineno) if _caught else None,
      (os.path.basename(__file__), _want_line))
check("scan/the warning is attributed to the caller, not to laya",
      os.path.basename(_caught[0].filename) if _caught else "no warning",
      os.path.basename(__file__))
with warnings.catch_warnings(record=True) as _caught:
    warnings.resetwarnings()
    warnings.simplefilter("default")                  # the interpreter's own default, not "always"
    _attempt(lambda: _ar.predict_long(LONG, Q, model="english", lang="de"))
    _attempt(lambda: _ar.predict_long(LONG, Q, model="english", lang="de"))
check("scan/two call sites each warn under the default filter, rather than collapsing to one",
      len(_caught), 2)


# A decorator built with `functools.wraps` must be followed through to the entry point that really
# runs: that is `inspect.signature`'s default, and it is how a real agent's `predict_long` most
# often ends up wrapped. Reading the wrapper's own `(*a, **kw)` instead would pass `lang` to an
# inner function that cannot take it.
class _WrappedScan:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {}}

    def _inner(self, state, questions, window=None, stride=None, aggregate="auto",
               batch_size=None):
        self.calls += 1
        return {"model": "stub", "answers": {"via": "predict_long"}, "usage": {}}

    @functools.wraps(_inner)
    def predict_long(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


_wrapped = _WrappedScan()
_wr = Router()
_wr.attach("english", _wrapped)
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    _wout, _wexc = _attempt(lambda: _wr.predict_long(LONG, Q, model="english", lang="de"))
check("scan/a functools.wraps wrapper is followed to the signature that really runs",
      (_kind(_wexc), (_wout or {}).get("answers"), _wrapped.calls,
       [w.category.__name__ for w in _caught]),
      (None, {"via": "predict_long"}, 1, ["RuntimeWarning"]))


class _KwargsScan:
    def __init__(self):
        self.kwargs = []

    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {}}

    def predict_long(self, state, questions, **controls):
        self.kwargs.append(controls)
        return {"model": "stub", "answers": {"via": "predict_long"}, "usage": {}}


_kw = _KwargsScan()
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    _router(_kw).predict_long(LONG, Q, model="english", lang="de")
check("scan/a **kwargs agent receives lang",
      (_kw.kwargs[0].get("lang"), [w.category.__name__ for w in _caught]), ("de", []))


# 6g. the other half of the `routing` line this change split in two. The scan path uses
#     `setdefault` so an agent's own `routing` survives (6d above); the plain `predict` path still
#     OVERWRITES, because `predict` promises a `routing` key that records the router's decision and
#     an agent must not be able to forge it. Nothing in the repo held that branch.
class _ForgesRouting:
    def system_one(self, state, questions, **controls):
        return {"model": "stub", "answers": {"via": "system_one"}, "usage": {},
                "routing": {"model": "forged by the agent"}}


_forged = _router(_ForgesRouting()).predict("a short state", Q, model="english")
check("predict/an agent cannot forge the routing key", _forged["routing"]["model"], "english")


# 7. `lang` still follows `predict`'s rule -- an explicit `lang=` wins, otherwise the language the
#    routing detected. The rule used to be written twice; it is computed once now and passed in.
_lang_explicit = _LongStub()
_router(_lang_explicit).predict_long(LONG, Q, model="english", lang="de")
check("scan/an explicit lang reaches the agent", _lang_explicit.calls[0]["lang"], "de")

# Both halves set at once, which is what pins the PRECEDENCE rather than each branch separately:
# an explicit `lang=` and a hook that writes a different detected language. Checked with the hook,
# because `model=` short-circuits detection, so this is the only way to have both.
_lang_both = _LongStub()
_router(_lang_both, hooks=[_SetDetectedLanguage()]).predict_long(
    LONG, Q, model="english", lang="de")
check("scan/an explicit lang outranks a detected one", _lang_both.calls[0]["lang"], "de")

# And with no explicit `lang`, the language the routing detected. This needs routing to actually run
# -- an explicit `model=` short-circuits it and leaves `detection` empty -- so it is the one check
# here that routes for real, and it is what fails if the scan goes back to reading its own `lang`
# attribute (None) instead of the value `predict` computed.
_lang_detected = _LongStub()
_detect_router = Router()
_detect_router.attach("multilingual", _lang_detected)
_detect_router.attach("english", _LongStub())
_german = "Wir wurden zweimal belastet und moechten eine Rueckerstattung erhalten. " * 8
_detected = _detect_router.predict_long(_german, Q)
check("scan/routing detected German", _detected["routing"]["model"], "multilingual")
check("scan/the detected language reaches the agent", _lang_detected.calls[0]["lang"], "de")


# --- the default window's floor ---------------------------------------------------------------

# `window_budget` has guarded the default with `max(64, ...)` since the feature landed (9acde82),
# and all three `predict_long` docstrings taught the budget on its own. The floor is the whole
# answer on a widened head: at `max_len=256, head_max_len=200` -- the shape docs/hooks/patterns.md
# tells you to build for a 60-option question -- the taught formula is 48 and the scan runs 64-token
# windows, so the prose sized every scan it touched 25% short. The number and the expression are
# read out of the code rather than typed here, so a docstring that drifts to a different constant
# fails as loudly as one that drops the guard. Scoped to these four docstrings on purpose: the bare
# formula is a true statement about the STATE room elsewhere (`state_room` measures the head that
# was built), and this gate is not in a position to renegotiate that.
import ast as _ast  # noqa: E402

_WB_SRC = inspect.getsource(window_budget)
_WB_TREE = _ast.parse(_WB_SRC)
_REQUESTED = [n for n in _WB_TREE.body[0].body
              if isinstance(n, _ast.Assign)
              and any(getattr(t, "id", "") == "requested" for t in n.targets)]
_MAX_CALL = None
for _node in _ast.walk(_REQUESTED[0] if _REQUESTED else _ast.Constant(None)):
    if (isinstance(_node, _ast.Call) and getattr(_node.func, "id", "") == "max"
            and len(_node.args) == 2 and isinstance(_node.args[0], _ast.Constant)
            and isinstance(_node.args[0].value, int)):
        _MAX_CALL = _node
        break
_FLOOR = _MAX_CALL.args[0].value if _MAX_CALL is not None else None
_DEFAULT_EXPR = (_ast.get_source_segment(_WB_SRC, _MAX_CALL)
                 if _MAX_CALL is not None else None)

check_true("floor/window_budget guards the default with max(<constant>, the budget)",
           _MAX_CALL is not None, "no such call in " + repr(_DEFAULT_EXPR))
check_true("floor/and the guarded expression is the docstrings' text verbatim",
           bool(_DEFAULT_EXPR) and _DEFAULT_EXPR.startswith("max("), _DEFAULT_EXPR)

# The behaviour the prose has to match: a config whose budget falls under the floor, with room to
# spare, so it is the floor that decides the window and not the room clamp.
_FLOOR_MAX_LEN, _FLOOR_HEAD_MAX_LEN = 256, 200
_FLOOR_Q = [Agent._to_internal(q_many(2))]
_FLOOR_EFF, _FLOOR_STEP, _FLOOR_ROOM = window_budget(
    TOK, _FLOOR_Q, _FLOOR_MAX_LEN, _FLOOR_HEAD_MAX_LEN)
check_true("floor/the taught budget is under the floor on this config",
           _FLOOR is not None and _FLOOR_MAX_LEN - _FLOOR_HEAD_MAX_LEN - 8 < _FLOOR,
           (_FLOOR_MAX_LEN - _FLOOR_HEAD_MAX_LEN - 8, _FLOOR))
check_true("floor/and the question leaves room for the floor, so the clamp is not what decides",
           _FLOOR is not None and _FLOOR_ROOM > _FLOOR, (_FLOOR_ROOM, _FLOOR))
check_true("floor/the default window is the floor, not the budget",
           _FLOOR is not None and _FLOOR_EFF == _FLOOR, (_FLOOR_EFF, _FLOOR))
check_true("floor/the stride still halves the window the floor produced",
           _FLOOR is not None and _FLOOR_STEP == _FLOOR // 2, (_FLOOR_STEP, _FLOOR))

_WINDOW_DOCSTRINGS = [
    ("laya/common.py window_budget", window_budget.__doc__, None),
    ("laya/agent.py Agent.predict_long", Agent.predict_long.__doc__, "stride:"),
    ("laya/router.py Router.predict_long", Router.predict_long.__doc__, "stride:"),
    ("laya/onnx_agent.py ONNXAgent.predict_long", ONNXAgent.predict_long.__doc__, "stride:"),
]
for _doc_name, _doc, _stop in _WINDOW_DOCSTRINGS:
    _para = (_doc or "")
    if _stop:
        _start = _para.find("window:")
        _stop_at = _para.find(_stop, _start)
        _para = _para[_start:_stop_at if _stop_at > -1 else None]
    check_true("floor/%s teaches the floored default" % _doc_name,
               bool(_DEFAULT_EXPR) and _DEFAULT_EXPR in _para, _para[:160])
    check_true("floor/%s states the budget nowhere without the floor" % _doc_name,
               "max_len - head_max_len - 8" not in _para.replace(_DEFAULT_EXPR or "", ""),
               _para[:160])
    # Taught as the DEFAULT, not merely mentioned: the expression has to sit inside the sentence
    # that names it, or a docstring could carry it as a footnote and still teach the bare budget.
    _at = _para.find(_DEFAULT_EXPR or "")
    check_true("floor/%s teaches it as the default, not beside it" % _doc_name,
               _at > -1 and "default" in _para[max(0, _at - 220):_at].lower(), _para[:160])


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
