"""Prediction hooks: observe or shape a decision without a model or a download.

The Agent path uses the same weight-free fake as `test_batch.py` (the real `predict_batch`
with `_encode_state` / `_forward` / `_decode_answers` stubbed). The Router path patches
`laya.agent.Agent` so a model build never touches the Hub.

Run: python tests/test_hooks.py
"""
import os
import sys
import threading
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya.agent as _agent_mod  # noqa: E402
from laya.agent import Agent  # noqa: E402
from laya.hooks import normalise_hooks  # noqa: E402
from laya.router import RouteDecision, Router  # noqa: E402

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
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
        return
    except BaseException as other:  # noqa: BLE001
        FAIL.append("%s: raised %r, want %r" % (name, other, exc))
        return
    FAIL.append("%s: did not raise %r" % (name, exc))


NQ = 2
QUESTIONS = {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "noul", "instructions": "?"}}


def make_fake():
    """A real `predict_batch` with the three composed helpers stubbed out."""
    fake = Agent.__new__(Agent)
    fake.tok = type("Tok", (), {"pad_token_id": 0})()
    fake._to_internal = staticmethod(Agent._to_internal).__func__
    fake._encode_states = []
    fake._forward_calls = []

    def _encode_state(state, ids, internal):
        fake._encode_states.append(state)
        return [{"ids": [1, 2, 3], "markers": [0, 1], "qtype": 2} for _ in ids]

    def _forward(b):
        n = b["input_ids"].shape[0]
        fake._forward_calls.append(n)
        return np.zeros((n, 2), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode_answers(logits, act, items, ids, internal, offset):
        return {"_offset": offset}

    fake._encode_state = _encode_state
    fake._forward = _forward
    fake._decode_answers = _decode_answers
    return fake


class FakeAgent:
    def system_one(self, state, questions):
        return {"model": "fake", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


# --------------------------------------------------------------- order and context
events = []
results_at_start = []
results_at_end = []
run_id_start = []
run_id_end = []


def start(ctx):
    events.append(("start", ctx))
    results_at_start.append(ctx.results)
    run_id_start.append(ctx.run_id)


def end(ctx):
    events.append(("end", ctx))
    results_at_end.append(ctx.results)
    run_id_end.append(ctx.run_id)


f = make_fake()
res = f.predict_batch(["s0", "s1"], QUESTIONS, on_predict_start=start, on_predict_end=end)
check("order/start then end", [e[0] for e in events], ["start", "end"])
check("order/once each", len(events), 2)
check("context/start sees states", events[0][1].states, ["s0", "s1"])
check("context/start sees questions", events[0][1].questions, QUESTIONS)
check("context/start has no results", results_at_start, [None])
check_true("context/end results is the returned object", results_at_end[0] is res)
check("context/end result count", len(results_at_end[0]), 2)
check_true("context/elapsed_ms set", events[1][1].elapsed_ms is not None)
check("context/usage aggregated", events[1][1].usage, {"input_tokens": 12, "output_tokens": 0})
check_true("context/agent set", events[1][1].agent is f)
check_true("context/run_id is a non-empty string",
           isinstance(run_id_start[0], str) and bool(run_id_start[0]))
check("context/run_id is shared by start and end", run_id_end[0], run_id_start[0])

_rids = []
_f2 = make_fake()
_f2.predict_batch(["s0"], QUESTIONS, on_predict_end=lambda c: _rids.append(c.run_id))
_f2.predict_batch(["s0"], QUESTIONS, on_predict_end=lambda c: _rids.append(c.run_id))
check("context/run_id differs per call", len(set(_rids)), 2)


# --------------------------------------------------------------- mutation
def rewrite(ctx):
    ctx.states = ["rewritten"]
    ctx.questions = {"a": {"type": "noul", "instructions": "?"}}


f = make_fake()
f.predict_batch(["orig"], QUESTIONS, on_predict_start=rewrite)
check("mutation/start rewrite is what gets encoded", f._encode_states, ["rewritten"])


def replace_results(ctx):
    ctx.results = [{"model": "replaced"}]


f = make_fake()
res = f.predict_batch(["s0"], QUESTIONS, on_predict_end=replace_results)
check("mutation/end rewrite is returned", res, [{"model": "replaced"}])


# --------------------------------------------------------------- caching short-circuit
end_calls = {"n": 0}


def cache_hit(ctx):
    ctx.skip([{"model": "cached"}])


def count_end(ctx):
    end_calls["n"] += 1


f = make_fake()
res = f.predict_batch(["s0"], QUESTIONS, on_predict_start=cache_hit, on_predict_end=count_end)
check("skip/returns cached results", res, [{"model": "cached"}])
check("skip/no forward pass", f._forward_calls, [])
check("skip/end still runs", end_calls["n"], 1)


# --------------------------------------------------------------- installed vs per-call
order = []
f = make_fake()
f.hooks = normalise_hooks(on_predict_start=lambda ctx: order.append("installed"))
f.predict_batch(["s0"], QUESTIONS, on_predict_start=lambda ctx: order.append("percall"))
check("installed/per-call additive and ordered", order, ["installed", "percall"])

seq = []
f = make_fake()
f.predict_batch(["s0"], QUESTIONS,
                on_predict_start=[lambda c: seq.append(1), lambda c: seq.append(2)])
check("sequence/runs in order", seq, [1, 2])


# --------------------------------------------------------------- system_one inherits
seen = []
f = make_fake()
f.hooks = normalise_hooks(on_predict_start=lambda ctx: seen.append(ctx.states))
out = f.system_one("state", QUESTIONS)
check("system_one/hook fired", seen, [["state"]])
check_true("system_one/returns a dict", isinstance(out, dict))


# --------------------------------------------------------------- error policy
def boom(ctx):
    raise ValueError("hook boom")


f = make_fake()
check_raises("hooks_raise=True/propagates", ValueError,
             lambda: f.predict_batch(["s0"], QUESTIONS, on_predict_start=boom))

f = make_fake()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = f.predict_batch(["s0"], QUESTIONS, on_predict_start=boom, hooks_raise=False)
check("hooks_raise=False/continues", len(res), 1)
check_true("hooks_raise=False/warns",
           any(issubclass(w.category, RuntimeWarning) for w in caught))


class ErrHook:
    def __init__(self):
        self.errors = []

    def on_error(self, ctx):
        self.errors.append(ctx.error)


fail_seen = {}


def end_on_fail(ctx):
    fail_seen["error"] = ctx.error


def bad_forward(b):
    raise RuntimeError("infer boom")


rec = ErrHook()
f = make_fake()
f._forward = bad_forward
raised = None
try:
    f.predict_batch(["s0"], QUESTIONS, hooks=[rec], on_predict_end=end_on_fail)
except RuntimeError as exc:
    raised = exc
check_true("on_error/propagates original", isinstance(raised, RuntimeError))
check("on_error/fired once", len(rec.errors), 1)
check("on_error/saw the error", str(rec.errors[0]), "infer boom")
check("on_error/end still ran", str(fail_seen.get("error")), "infer boom")


class BadErrHook:
    def on_error(self, ctx):
        raise ValueError("hook error")


f = make_fake()
f._forward = bad_forward
raised = None
try:
    f.predict_batch(["s0"], QUESTIONS, hooks=[BadErrHook()])
except BaseException as exc:  # noqa: BLE001
    raised = exc
check_true("on_error/failing hook does not mask the original", isinstance(raised, RuntimeError))
check_true("on_error/failing hook is chained",
           isinstance(getattr(raised, "__context__", None), ValueError))


# a start hook that raises must still run on_error and the end hooks
class StartErrRec:
    def __init__(self):
        self.errors = []

    def on_error(self, ctx):
        self.errors.append(ctx.error)


start_fail_seen = []


def raising_start(ctx):
    raise ValueError("start boom")


def end_after_start_fail(ctx):
    start_fail_seen.append(ctx.error)


rec2 = StartErrRec()
f = make_fake()
raised = None
try:
    f.predict_batch(["s0"], QUESTIONS, hooks=[rec2],
                    on_predict_start=raising_start, on_predict_end=end_after_start_fail)
except ValueError as exc:
    raised = exc
check_true("start failure/propagates", isinstance(raised, ValueError))
check("start failure/on_error fired", len(rec2.errors), 1)
check("start failure/end fired with the error", len(start_fail_seen), 1)
check("start failure/end saw the error", str(start_fail_seen[0]), "start boom")


# ctx.model carries the agent's model id
f = make_fake()
f.model_id = "convaiinnovations/laya"
models = []
f.predict_batch(["s0"], QUESTIONS, on_predict_end=lambda c: models.append(c.model))
check("context/model is the agent model_id", models, ["convaiinnovations/laya"])


# --------------------------------------------------------------- empty inputs
empty = []
f = make_fake()
res = f.predict_batch([], QUESTIONS,
                      on_predict_start=lambda c: empty.append(("s", c.results)),
                      on_predict_end=lambda c: empty.append(("e", c.results)))
check("empty states/returns []", res, [])
check("empty states/start results None", empty[0][1], None)
check("empty states/end results []", empty[1][1], [])

empty = []
f = make_fake()
f.predict_batch(["s0"], {}, on_predict_end=lambda c: empty.append(c.results))
check("empty questions/end sees empty results",
      empty[0],
      [{"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}])


# --------------------------------------------------------------- input guards and rescue
f = make_fake()
check_raises("guard/bare string raises", TypeError, lambda: f.predict_batch("nope", QUESTIONS))
f = make_fake()
check_raises("guard/bare dict raises", TypeError, lambda: f.predict_batch({"body": "x"}, QUESTIONS))


def normalise_state(ctx):
    ctx.states = [ctx.states]


f = make_fake()
f.predict_batch("nope", QUESTIONS, on_predict_start=normalise_state)
check("guard/a hook can normalise a bare string", f._encode_states, ["nope"])


# --------------------------------------------------------------- hand-built instance
f = make_fake()
res = f.predict_batch(["s0", "s1", "s2"], QUESTIONS)
check("defaults/unset hooks are a no-op", len(res), 3)
check("defaults/offsets unchanged", [r["answers"]["_offset"] for r in res], [0, NQ, 2 * NQ])


# --------------------------------------------------------------- dynamic registration
class Tag:
    def __init__(self, log, tag):
        self.log = log
        self.tag = tag

    def on_predict_end(self, ctx):
        self.log.append(self.tag)


log = []
f = make_fake()
f.add_hook(Tag(log, "a"))
f.predict_batch(["s0"], QUESTIONS)
check("add_hook/fires", log, ["a"])
check("remove_hook/returns True", f.remove_hook(f.hooks[-1]), True)
log.clear()
f.predict_batch(["s0"], QUESTIONS)
check("remove_hook/no longer fires", log, [])
check("remove_hook/unknown returns False", f.remove_hook(object()), False)

log = []
f = make_fake()
check_true("add_hook/returns self", f.add_hook([Tag(log, "x"), Tag(log, "y")]) is f)
f.predict_batch(["s0"], QUESTIONS)
check("add_hook/sequence order", log, ["x", "y"])

log = []
f = make_fake()
f.add_hook(Tag(log, "installed"))
with f.hooks_installed(Tag(log, "temp")):
    f.predict_batch(["s0"], QUESTIONS)
f.predict_batch(["s0"], QUESTIONS)
check("hooks_installed/only during the block", log, ["installed", "temp", "installed"])

log = []
r = Router()
r.add_hook(Tag(log, "router"))
r.attach("english", FakeAgent())
r.predict("hello", QUESTIONS)
check("router/add_hook fires", log, ["router"])


# --------------------------------------------------------------- per-call token budget
def make_len_fake():
    fake = make_fake()
    fake._seen = []

    def _encode(state, ids, internal, max_len=None, head_max_len=None):
        fake._seen.append((max_len, head_max_len))
        return [{"ids": [1, 2, 3], "markers": [0, 1], "qtype": 2} for _ in ids]

    fake._encode_state = _encode
    return fake


f = make_len_fake()
f.predict_batch(["s"], QUESTIONS, max_len=128, head_max_len=64)
check("budget/per-call kwargs reach _encode_state", f._seen, [(128, 64)])

f = make_len_fake()
f.predict_batch(["s"], QUESTIONS, on_predict_start=lambda c: setattr(c, "max_len", 200))
check("budget/hook-set ctx reaches _encode_state", f._seen, [(200, None)])

f = make_len_fake()
f.predict_batch(["s"], QUESTIONS)
check("budget/default passes no override", f._seen, [(None, None)])


class LenFake:
    def __init__(self):
        self.seen = []

    def system_one(self, state, questions, max_len=None, head_max_len=None):
        self.seen.append((max_len, head_max_len))
        return {"model": "x", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


lf = LenFake()
r = Router()
r.attach("english", lf)
r.predict("hello", QUESTIONS, max_len=256, head_max_len=128)
check("budget/router per-call reaches the agent", lf.seen, [(256, 128)])

lf = LenFake()
r = Router()
r.attach("english", lf)
r.predict("hello", QUESTIONS, on_predict_start=lambda c: setattr(c, "head_max_len", 96))
check("budget/router hook-set reaches the agent", lf.seen, [(None, 96)])


class StrictFake:
    """An agent-like object that does not accept the override kwargs."""

    def system_one(self, state, questions):
        return {"model": "x", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


r = Router()
r.attach("english", StrictFake())
r.predict("hello", QUESTIONS)
check("budget/default does not pass override kwargs", True, True)


# --------------------------------------------------------------- context semantics
from laya.hooks import PredictContext  # noqa: E402

_c1 = PredictContext(states=[], questions={})
_c2 = PredictContext(states=[], questions={})
check_true("context/hashable by identity", isinstance(hash(_c1), int))
check("context/two contexts are never equal", _c1 == _c2, False)


# --------------------------------------------------------------- validation
class NotAHook:
    pass


check_raises("validation/hooks= rejects a class", TypeError, lambda: normalise_hooks(hooks=[NotAHook]))
check_raises("validation/hooks= rejects a plain callable", TypeError,
             lambda: normalise_hooks(hooks=[lambda ctx: None]))
check_raises("validation/on_predict_start must be callable", TypeError,
             lambda: normalise_hooks(on_predict_start=123))


class BadMethod:
    on_predict_start = 5


check_raises("validation/hook method must be callable", TypeError,
             lambda: normalise_hooks(hooks=[BadMethod()]))


# --------------------------------------------------------------- Router
class RouteHook:
    def __init__(self):
        self.decisions = []

    def on_route(self, ctx):
        self.decisions.append(dict(ctx.decision))
        ctx.decision = RouteDecision(model="multilingual", repo="convaiinnovations/laya/multilingual",
                                     reason="pinned by hook", detection=None, workflow=None)


rh = RouteHook()
r = Router(hooks=[rh])
decision = r.route("hello", QUESTIONS)
check("router/on_route fired", len(rh.decisions), 1)
check("router/on_route saw the original", rh.decisions[0]["model"], "english")
check("router/on_route can replace the decision", decision["model"], "multilingual")


class LoadHook:
    def __init__(self):
        self.loads = []
        self.evicts = []

    def on_load(self, ctx):
        self.loads.append(ctx.model)

    def on_evict(self, ctx):
        self.evicts.append(ctx.model)


class BuiltAgent(FakeAgent):
    def __init__(self, *args, **kwargs):
        pass


lh = LoadHook()
real_agent = _agent_mod.Agent
_agent_mod.Agent = BuiltAgent
try:
    r = Router(max_loaded=1, hooks=[lh])
    r.load("english")
    r.load("multilingual")  # evicts english
finally:
    _agent_mod.Agent = real_agent
check("router/on_load fired per build", lh.loads, ["english", "multilingual"])
check("router/on_evict fired on eviction", lh.evicts, ["english"])


predict_seen = {}
r = Router(on_predict_start=lambda ctx: predict_seen.update(decision=dict(ctx.decision)),
           on_predict_end=lambda ctx: predict_seen.update(results=ctx.results))
r.attach("english", FakeAgent())
out = r.predict("hello", QUESTIONS)
check("router/predict start sees the decision", predict_seen["decision"]["model"], "english")
check("router/predict end sees results", len(predict_seen["results"]), 1)
check("router/predict keeps routing", out["routing"]["model"], "english")

cached = [{"model": "cached", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}]
r = Router()
r.attach("english", FakeAgent())
skipped = r.predict("hello", QUESTIONS, on_predict_start=lambda c: c.skip(cached))
check("router/skip returns the cached payload", skipped, cached[0])
check("router/skip still adds routing", skipped.get("routing", {}).get("model"), "english")


class PerCallRoute:
    def __init__(self):
        self.decisions = []

    def on_route(self, ctx):
        self.decisions.append(dict(ctx.decision))


pcr = PerCallRoute()
r = Router()
r.attach("english", FakeAgent())
r.predict("hello", QUESTIONS, hooks=[pcr])
check("router/per-call hooks apply to on_route", len(pcr.decisions), 1)


# --------------------------------------------------------------- Router.predict_batch
# `predict_batch` promises each result is what `predict` returns for that request, so Router-level
# predict hooks must run per request there too. It used to skip them: a redaction hook never ran,
# and the raw state reached the model.
class BatchFake:
    """Records every `predict_batch` call and answers with the state it was given."""

    def __init__(self):
        self.calls = []

    def predict_batch(self, states, questions, batch_size=None, **overrides):
        self.calls.append((list(states), questions, overrides))
        if "boom" in states:
            raise RuntimeError("inference failed")
        return [{"model": "fake", "answers": {"seen": s},
                 "usage": {"input_tokens": len(s), "output_tokens": 0}} for s in states]

    def system_one(self, state, questions, **overrides):
        return self.predict_batch([state], questions, **overrides)[0]


def batch_router(**kwargs):
    r = Router(**kwargs)
    en, ml = BatchFake(), BatchFake()
    r.attach("english", en)
    r.attach("multilingual", ml)
    return r, en, ml


def req(state, model="english", questions=QUESTIONS):
    return {"state": state, "questions": questions, "model": model}


def redact_start(ctx):
    ctx.states = [s.replace("secret", "[redacted]") for s in ctx.states]


r, en, ml = batch_router(on_predict_start=redact_start)
out = r.predict_batch([req("a secret"), req("b secret", "multilingual"), req("c secret")])
check("router_batch/start hook rewrite reaches the agent",
      [c[0] for c in en.calls + ml.calls], [["a [redacted]", "c [redacted]"], ["b [redacted]"]])
check("router_batch/results follow the rewrite", [o["answers"]["seen"] for o in out],
      ["a [redacted]", "b [redacted]", "c [redacted]"])

batch_events = []
batch_results_at_start = {}


class BatchTrace:
    def on_predict_start(self, ctx):
        batch_events.append(("start", ctx.states[0], ctx))
        batch_results_at_start[ctx.states[0]] = ctx.results

    def on_predict_end(self, ctx):
        batch_events.append(("end", ctx.states[0], ctx))

    def on_error(self, ctx):
        batch_events.append(("error", ctx.states[0], ctx))


r, en, ml = batch_router(hooks=[BatchTrace()])
out = r.predict_batch([req("one"), req("two", "multilingual"), req("three")])
check("router_batch/one start and one end per request, per checkpoint group",
      [(e[0], e[1]) for e in batch_events],
      [("start", "one"), ("start", "three"), ("end", "one"), ("end", "three"),
       ("start", "two"), ("end", "two")])
starts = {e[1]: e[2] for e in batch_events if e[0] == "start"}
ends = {e[1]: e[2] for e in batch_events if e[0] == "end"}
check("router_batch/every request starts and ends", (sorted(starts), sorted(ends)),
      (["one", "three", "two"], ["one", "three", "two"]))
if len(starts) == len(ends) == 3:
    check("router_batch/start sees its own state", starts["two"].states, ["two"])
    check("router_batch/start sees its own questions", starts["two"].questions, QUESTIONS)
    check("router_batch/start sees its own decision", starts["two"].decision["model"], "multilingual")
    check("router_batch/context model is the routed checkpoint", starts["two"].model, "multilingual")
    check_true("router_batch/context agent is the routed agent", starts["two"].agent is ml)
    check_true("router_batch/context router is set", starts["two"].router is r)
    check("router_batch/start has no results", batch_results_at_start, {"one": None, "two": None, "three": None})
    check_true("router_batch/start and end share one context per request", ends["two"] is starts["two"])
    check("router_batch/end sees its own result", ends["two"].results[0]["answers"]["seen"], "two")
    check("router_batch/end result carries routing", ends["two"].results[0]["routing"]["model"], "multilingual")
    check("router_batch/usage is per request", ends["three"].usage, {"input_tokens": 5, "output_tokens": 0})
    check_true("router_batch/elapsed_ms set", ends["one"].elapsed_ms is not None)
    check("router_batch/run_id differs per request", len({c.run_id for c in starts.values()}), 3)
check("router_batch/still one agent call per checkpoint", [c[0] for c in en.calls + ml.calls],
      [["one", "three"], ["two"]])


class PlainDictPin:
    """An on_route hook replacing the decision with a plain dict, which `predict` accepts."""

    def on_route(self, ctx):
        ctx.decision = {**ctx.decision, "model": "multilingual"}


r, en, ml = batch_router(hooks=[PlainDictPin()])
try:
    out = r.predict_batch([req("pinned")])
    got = ([c[0] for c in ml.calls], out[0]["routing"]["model"])
except AttributeError as e:
    got = repr(e)
check("router_batch/an on_route hook may replace the decision with a plain dict", got, ([["pinned"]], "multilingual"))

group_ctxs = []
elapsed_when_first_end_ran = []


def remember_start(ctx):
    group_ctxs.append(ctx)


def first_end(ctx):
    if not elapsed_when_first_end_ran:
        elapsed_when_first_end_ran.append([c.elapsed_ms is not None for c in group_ctxs])


r, en, ml = batch_router(on_predict_start=remember_start, on_predict_end=first_end)
r.predict_batch([req("a"), req("b"), req("c")])
check("router_batch/elapsed_ms is set for the whole group before any end hook runs",
      elapsed_when_first_end_ran, [[True, True, True]])


def replace_result(ctx):
    ctx.results = [{"replaced": ctx.states[0]}]


r, en, ml = batch_router(on_predict_end=replace_result)
check("router_batch/an end hook's replacement is what is returned",
      r.predict_batch([req("x"), req("y", "multilingual")]), [{"replaced": "x"}, {"replaced": "y"}])

cached_x = {"model": "cached", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
r, en, ml = batch_router(on_predict_start=lambda c: c.skip([dict(cached_x)]) if c.states == ["x"] else None)
out = r.predict_batch([req("x"), req("y")])
check("router_batch/skip keeps that state from the agent", [c[0] for c in en.calls], [["y"]])
check("router_batch/skip returns the cached payload", out[0]["model"], "cached")
check("router_batch/skip still adds routing", out[0].get("routing", {}).get("model"), "english")
check("router_batch/the other request is inferred", out[1]["answers"]["seen"], "y")


def annotate_end(ctx):
    ctx.results[0]["audited"] = ctx.decision["model"]


mixed = [req("a secret"), req("b secret", "multilingual"), req("c", "english")]
r, en, ml = batch_router(on_predict_start=redact_start, on_predict_end=annotate_end)
batched = r.predict_batch(mixed)
one_by_one = [r.predict(m["state"], m["questions"], model=m["model"]) for m in mixed]
check("router_batch/identical to one predict call per request", batched, one_by_one)


def budget_start(ctx):
    if ctx.states == ["long"]:
        ctx.max_len = 1024


r, en, ml = batch_router(on_predict_start=budget_start)
r.predict_batch([req("short"), req("long"), req("short again")])
check("router_batch/hook-set token budget reaches the agent for that request only",
      [(c[0], c[2]) for c in en.calls], [(["short", "short again"], {}), (["long"], {"max_len": 1024})])

Q_OTHER = {"other": {"type": "noul", "instructions": "Other?"}}
r, en, ml = batch_router(on_predict_start=lambda c: setattr(c, "questions", Q_OTHER) if c.states == ["b"] else None)
r.predict_batch([req("a"), req("b"), req("c")])
check("router_batch/rewritten questions are batched on their own",
      [(c[0], c[1]) for c in en.calls], [(["a", "c"], QUESTIONS), (["b"], Q_OTHER)])


class StrictBatchFake(BatchFake):
    """An agent-like object whose predict_batch takes no token-budget kwargs."""

    def predict_batch(self, states, questions, batch_size=None):
        return BatchFake.predict_batch(self, states, questions, batch_size)


r = Router(on_predict_start=lambda c: None)
strict = StrictBatchFake()
r.attach("english", strict)
r.predict_batch([req("a"), req("b")], batch_size=8)
check("router_batch/default passes no override kwargs", [c[0] for c in strict.calls], [["a", "b"]])

batch_events = []
r, en, ml = batch_router(hooks=[BatchTrace()])
check_raises("router_batch/inference failure propagates", RuntimeError,
             lambda: r.predict_batch([req("ok", "multilingual"), req("boom"), req("also")]))
check("router_batch/failure pairs every started request with on_error then on_predict_end",
      [(e[0], e[1]) for e in batch_events],
      [("start", "ok"), ("end", "ok"), ("start", "boom"), ("start", "also"),
       ("error", "boom"), ("error", "also"), ("end", "boom"), ("end", "also")])
ends = {e[1]: e[2] for e in batch_events if e[0] == "end"}
check_true("router_batch/end of a failed request sees the error",
           all(isinstance(getattr(ends.get(s), "error", None), RuntimeError) for s in ("boom", "also")))
check_true("router_batch/an earlier group that finished ends without an error",
           "ok" in ends and ends["ok"].error is None)

batch_events = []
r, en, ml = batch_router(hooks=[BatchTrace()],
                         on_predict_start=lambda c: c.skip([dict(cached_x)]) if c.states == ["cached"] else None)
check_raises("router_batch/inference failure with a cache hit in the group propagates", RuntimeError,
             lambda: r.predict_batch([req("cached"), req("boom")]))
ends = {e[1]: e[2] for e in batch_events if e[0] == "end"}
check("router_batch/only requests without a result get on_error",
      [e[1] for e in batch_events if e[0] == "error"], ["boom"])
check_true("router_batch/a cache hit in a failed group keeps its result and no error",
           "cached" in ends and ends["cached"].error is None and ends["cached"].results is not None)

batch_events = []


def fail_end_on_a(ctx):
    if ctx.states == ["a"]:
        raise ValueError("end hook failed")


r, en, ml = batch_router(on_predict_end=fail_end_on_a, hooks=[BatchTrace()])
check_raises("router_batch/end hook failure propagates", ValueError,
             lambda: r.predict_batch([req("a"), req("b")]))
check("router_batch/end hook failure still ends the other requests",
      [(e[0], e[1]) for e in batch_events if e[0] == "end"], [("end", "a"), ("end", "b")])

batch_events = []
r, en, ml = batch_router(hooks=[BatchTrace()])
check_raises("router_batch/inference failure propagates before a later group", RuntimeError,
             lambda: r.predict_batch([req("boom"), req("later", "multilingual")]))
check("router_batch/a later group never starts", ("start", "later") in [(e[0], e[1]) for e in batch_events], False)
check("router_batch/a later group runs no inference", ml.calls, [])

batch_events = []


def fail_on_b(ctx):
    if ctx.states == ["b"]:
        raise ValueError("start hook failed")


r, en, ml = batch_router(hooks=[BatchTrace()], on_predict_start=fail_on_b)
check_raises("router_batch/start hook failure propagates", ValueError,
             lambda: r.predict_batch([req("a"), req("b"), req("c")]))
check("router_batch/start hook failure still ends every started request",
      [(e[0], e[1]) for e in batch_events],
      [("start", "a"), ("start", "b"), ("error", "a"), ("error", "b"), ("end", "a"), ("end", "b")])
check("router_batch/start hook failure runs no inference", en.calls, [])

r, en, ml = batch_router(on_predict_start=fail_on_b, hooks_raise=False)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = r.predict_batch([req("a"), req("b")])
check("router_batch/hooks_raise=False keeps the batch going", [o["answers"]["seen"] for o in out], ["a", "b"])
check_true("router_batch/hooks_raise=False warns", any("start hook failed" in str(w.message) for w in caught))


# --------------------------------------------------------------- hooks_concurrent storage
r = Router()
check("router/hooks_concurrent default True", r.hooks_concurrent, True)
check_true("router/hooks_concurrent default has no lock", r._hooks_lock is None)
r = Router(hooks_concurrent=False)
check("router/hooks_concurrent=False stored", r.hooks_concurrent, False)
check_true("router/hooks_concurrent=False installs a lock", r._hooks_lock is not None)


# --------------------------------------------------------------- concurrency
contexts = []


def slow_start(ctx):
    time.sleep(0.01)
    contexts.append(id(ctx))


f = make_fake()
threads = [threading.Thread(target=lambda: f.predict_batch(["s0"], QUESTIONS,
                                                           on_predict_start=slow_start))
           for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("concurrency/one independent context per call", len(set(contexts)), 5)


class ConcHook:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def on_predict_start(self, ctx):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1


conc = ConcHook()
f = make_fake()
f.hooks = [conc]
f._hooks_lock = threading.RLock()  # what hooks_concurrent=False installs
threads = [threading.Thread(target=lambda: f.predict_batch(["s0"], QUESTIONS)) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("concurrency/hooks_concurrent=False serialises", conc.max_active, 1)


# --------------------------------------------------------------- ONNXAgent parity
from laya.onnx_agent import ONNXAgent  # noqa: E402

o = ONNXAgent.__new__(ONNXAgent)
o.model_id = "convaiinnovations/laya-onnx"
o._infer = lambda state, questions, **kwargs: {"model": "onnx", "answers": {},
                                               "usage": {"input_tokens": 0, "output_tokens": 0}}
onnx_seen = []
onnx_models = []
onnx_out = o.system_one("s", QUESTIONS,
                        on_predict_start=lambda c: onnx_seen.append("start"),
                        on_predict_end=lambda c: (onnx_seen.append("end"), onnx_models.append(c.model)))
check("onnx/hooks fire", onnx_seen, ["start", "end"])
check("onnx/returns the inference result", onnx_out["model"], "onnx")
check("onnx/context model is the agent model_id", onnx_models, ["convaiinnovations/laya-onnx"])

o = ONNXAgent.__new__(ONNXAgent)


def _never(*args, **kwargs):
    raise AssertionError("inference should have been skipped")


o._infer = _never
check("onnx/skip short-circuits",
      o.system_one("s", QUESTIONS, on_predict_start=lambda c: c.skip([{"model": "cached"}])),
      {"model": "cached"})


# --------------------------------------------------------------- BaseHook
from laya import BaseHook  # noqa: E402


class OnlyEnd(BaseHook):
    def __init__(self, log, tag):
        self.log = log
        self.tag = tag

    def on_predict_end(self, ctx):
        self.log.append(self.tag)


log = []
f = make_fake()
f.add_hook(OnlyEnd(log, "end"))
f.predict_batch(["s0"], QUESTIONS)
check("BaseHook/overridden event fires", log, ["end"])

f = make_fake()
f.add_hook(BaseHook())  # every method is a no-op
f.predict_batch(["s0"], QUESTIONS)
check("BaseHook/no-op instance is harmless", f._forward_calls, [2])


# --------------------------------------------------------------- process-wide defaults
from laya import hooks as _hooks  # noqa: E402

log = []
_hooks.set_default_hooks(on_predict_end=lambda ctx: log.append("default"))
try:
    f = make_fake()
    f.add_hook(Tag(log, "installed"))
    f.predict_batch(["s0"], QUESTIONS, on_predict_end=lambda ctx: log.append("percall"))
    check("defaults/run before installed and per-call", log, ["default", "installed", "percall"])
finally:
    _hooks.clear_default_hooks()
check("defaults/clear empties the registry", _hooks.default_hooks(), [])

log = []
_hooks.add_default_hook(Tag(log, "a"))
_hooks.add_default_hook(Tag(log, "b"))
try:
    f = make_fake()
    f.predict_batch(["s0"], QUESTIONS)
    check("defaults/add in order", log, ["a", "b"])
finally:
    _hooks.clear_default_hooks()

# A default hook has to reach the Router as well as the Agent: `set_default_hooks` documents
# "every Agent, Router and ONNXAgent in the process", and `predict_batch` promises each request
# its own PredictContext. It read `list(self.hooks)` at its dispatch site instead of composing
# with the registry, so a process-wide hook saw the Agent-level events and none of the
# Router-level ones -- invisible unless the hook records which level it ran at, which is why
# the checks above never caught it.
events = []


class LevelTag:
    def __init__(self, tag):
        self.tag = tag

    def on_predict_start(self, ctx):
        events.append((self.tag, "start", "router" if getattr(ctx, "router", None) is not None
                                         else "agent"))

    def on_predict_end(self, ctx):
        events.append((self.tag, "end", "router" if getattr(ctx, "router", None) is not None
                                       else "agent"))


_hooks.set_default_hooks([LevelTag("default")])
try:
    r, en, ml = batch_router()
    events.clear()
    r.predict_batch([req("a"), req("b")])
    # `BatchFake` replaces the agent wholesale, so the only events that can come from this
    # Router are its own. That is exactly what the defect removed.
    check("defaults/reach the Router on predict_batch",
          events, [("default", "start", "router"), ("default", "start", "router"),
                   ("default", "end", "router"), ("default", "end", "router")])

    # `predict` was already correct; assert the two entry points agree, which is the property
    # that was actually violated.
    events.clear()
    r.predict("a", QUESTIONS, model="english")
    via_predict = list(events)
    events.clear()
    r.predict_batch([req("a")])
    check("defaults/predict and predict_batch give the same event shape",
          events, via_predict)

    # An instance hook is composed the same way and must not be affected either way.
    events.clear()
    r2, _, _ = batch_router(hooks=[LevelTag("instance")])
    r2.predict_batch([req("a")])
    check("defaults/instance hooks still fire once alongside defaults",
          [e for e in events if e[0] == "instance"],
          [("instance", "start", "router"), ("instance", "end", "router")])
finally:
    _hooks.clear_default_hooks()


class LifeDefaults(BaseHook):
    def __init__(self):
        self.events = []

    def on_load(self, ctx):
        self.events.append(("load", ctx.model))

    def on_evict(self, ctx):
        self.events.append(("evict", ctx.model))


ld = LifeDefaults()
_hooks.set_default_hooks(hooks=[ld])
try:
    _agent_mod.Agent = BuiltAgent
    r = Router(max_loaded=1)
    r.load("english")
    r.load("multilingual")  # evicts english
finally:
    _agent_mod.Agent = real_agent
    _hooks.clear_default_hooks()
check("defaults/cover router lifecycle", ld.events,
      [("load", "english"), ("evict", "english"), ("load", "multilingual")])

events = []
_hooks.set_default_hooks([LevelTag("default")])
r = Router()
r.attach("english", make_fake())
r.predict("a", QUESTIONS, model="english")
r.predict_batch([req("b")])
_hooks.clear_default_hooks()
check("defaults/fire once per request", events, [("default", "start", "router"), ("default", "end", "router")] * 2)


# --------------------------------------------------------------- async hooks
import asyncio  # noqa: E402

from laya import AsyncHook  # noqa: E402
from laya.hooks import run_coroutine_sync  # noqa: E402

calls = []


class AsyncAudit:
    async def on_predict_end(self, ctx):
        await asyncio.sleep(0)
        calls.append("wrapped")


f = make_fake()
f.add_hook(AsyncHook(AsyncAudit()))
f.predict_batch(["s0"], QUESTIONS)
check("async/wrapped hook awaited", calls, ["wrapped"])

calls.clear()


async def async_end(ctx):
    await asyncio.sleep(0)
    calls.append("plain")


f = make_fake()
f.predict_batch(["s0"], QUESTIONS, on_predict_end=async_end)
check("async/plain callable awaited", calls, ["plain"])

calls.clear()


async def async_start(ctx):
    calls.append("loop")


f = make_fake()


async def _in_loop():
    f.predict_batch(["s0"], QUESTIONS, on_predict_start=async_start)


asyncio.run(_in_loop())
check("async/works inside a running loop", calls, ["loop"])


async def _seven():
    return 7


check("async/run_coroutine_sync returns", run_coroutine_sync(_seven()), 7)
check_raises("async/AsyncHook rejects a class", TypeError, lambda: AsyncHook(AsyncAudit))


# --------------------------------------------------------------- hook timeout
def slow_hook(ctx):
    time.sleep(0.3)


f = make_fake()
raised = None
try:
    f.predict_batch(["s0"], QUESTIONS, on_predict_start=slow_hook, hooks_timeout=0.05)
except TimeoutError as exc:
    raised = str(exc)
check_true("timeout/raises TimeoutError", isinstance(raised, str) and "exceeded" in raised)

f = make_fake()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = f.predict_batch(["s0"], QUESTIONS, on_predict_start=slow_hook,
                          hooks_timeout=0.05, hooks_raise=False)
check("timeout/hooks_raise=False continues", len(res), 1)
check_true("timeout/hooks_raise=False warns",
           any(issubclass(w.category, RuntimeWarning) for w in caught))

f = make_fake()
f.hooks_timeout = 0.05
raised = None
try:
    f.predict_batch(["s0"], QUESTIONS, on_predict_start=slow_hook)
except TimeoutError:
    raised = True
check("timeout/instance-level applies", raised, True)

f = make_fake()
f.hooks_timeout = 0.05
res = f.predict_batch(["s0"], QUESTIONS, on_predict_start=slow_hook, hooks_timeout=5.0)
check("timeout/per-call override wins", len(res), 1)

f = make_fake()
res = f.predict_batch(["s0"], QUESTIONS, on_predict_end=lambda ctx: None, hooks_timeout=1.0)
check("timeout/fast hook unaffected", len(res), 1)


# --------------------------------------------------------------- input validation and context
import contextvars  # noqa: E402

from laya.hooks import validate_timeout  # noqa: E402

check_raises("timeout/zero is rejected", ValueError, lambda: validate_timeout(0))
check_raises("timeout/negative is rejected", ValueError, lambda: validate_timeout(-0.5))
check("timeout/positive passes through", validate_timeout(1.5), 1.5)
check("timeout/None means no limit", validate_timeout(None), None)

f = make_fake()
check_raises("timeout/zero per call is rejected", ValueError,
             lambda: f.predict_batch(["s0"], QUESTIONS, hooks_timeout=0))

not_running = asyncio.new_event_loop()
try:
    check_raises("async/a non-running loop is rejected", ValueError,
                 lambda: run_coroutine_sync(_seven(), loop=not_running))
finally:
    not_running.close()


async def _own_loop():
    own = asyncio.get_running_loop()
    try:
        run_coroutine_sync(_seven(), loop=own)
    except ValueError:
        return "raised"
    return "no"


check("async/the calling thread's own loop is rejected", asyncio.run(_own_loop()), "raised")
check_raises("async/AsyncHook rejects an object with no events", TypeError,
             lambda: AsyncHook(object()))

_cv_seen = contextvars.ContextVar("cv_seen", default=None)
_cv_calls = []


class CvHook:
    def on_predict_start(self, ctx):
        _cv_calls.append(_cv_seen.get())


f = make_fake()
_token = _cv_seen.set("request-1")
try:
    f.predict_batch(["s0"], QUESTIONS, hooks=[CvHook()], hooks_timeout=1.0)
finally:
    _cv_seen.reset(_token)
check("timeout/a timed hook sees the caller's contextvars", _cv_calls, ["request-1"])


# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f_ in FAIL:
    print("  FAIL " + f_)
if not FAIL:
    print("all hook tests passed")
sys.exit(1 if FAIL else 0)
