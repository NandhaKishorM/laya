"""predict_long: scan a state longer than the window and aggregate per question.

Weight-free. The real forward path is stubbed (predict_batch returns canned per-window answers),
so this checks only predict_long's own logic: the fits-in-one-window short-circuit, the overlapping
window split, the per-type aggregation (noul = strongest window, choice/score = most-confident
window), and the per-call hook controls it forwards to whichever of those two calls runs. Numerical
behaviour on real weights is exercised in tests/test_local_e2e.py.
"""
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
           "_check_scan_budget(self, evidence, budget, max_len, head_max_len, questions)" in _onnx_src,
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


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
