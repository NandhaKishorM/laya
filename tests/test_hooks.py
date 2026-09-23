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
o._infer = lambda state, questions: {"model": "onnx", "answers": {},
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


def _never(*args):
    raise AssertionError("inference should have been skipped")


o._infer = _never
check("onnx/skip short-circuits",
      o.system_one("s", QUESTIONS, on_predict_start=lambda c: c.skip([{"model": "cached"}])),
      {"model": "cached"})


# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f_ in FAIL:
    print("  FAIL " + f_)
if not FAIL:
    print("all hook tests passed")
sys.exit(1 if FAIL else 0)
