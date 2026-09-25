"""predict_long: scan a state longer than the window and aggregate per question.

Weight-free. The real forward path is stubbed (predict_batch returns canned per-window answers),
so this checks only predict_long's own logic: the fits-in-one-window short-circuit, the overlapping
window split, and the per-type aggregation (noul = strongest window, choice/score = most-confident
window). Numerical behaviour on real weights is exercised in tests/test_local_e2e.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent  # noqa: E402

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


class _Tok:
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False):
        # token count == character count, so the test controls windowing by string length
        return {"input_ids": list(range(len(text)))}

    def decode(self, ids):
        return "w%d_%d" % (ids[0], ids[-1]) if ids else "w"


def make_agent(batch_result_fn):
    a = Agent.__new__(Agent)
    a.cfg = {"max_len": 100, "head_max_len": 20}   # budget = max(64, 100-20-8) = 72
    a.tok = _Tok()
    a._to_internal = staticmethod(Agent._to_internal).__func__
    a._calls = {"system_one": 0, "batch_states": None}

    def _system_one(state, questions, lang=None):
        a._calls["system_one"] += 1
        return {"model": "laya-rl-agent", "answers": {"_via": "system_one"}, "usage": {"input_tokens": 1}}

    def _predict_batch(states, questions, batch_size=None, lang=None):
        a._calls["batch_states"] = list(states)
        return batch_result_fn(list(states), questions)

    a.system_one = _system_one
    a.predict_batch = _predict_batch
    return a


Q = {"dept": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
     "flag": {"type": "noul", "instructions": "?"}}

# 1. fits in one window -> delegates to system_one, no windowing
a = make_agent(lambda s, q: [])
short = a.predict_long({"body": "x" * 50}, Q)   # 50 tokens <= budget 72
check("short/delegates to system_one", short["answers"], {"_via": "system_one"})
check("short/no predict_batch call", a._calls["batch_states"], None)

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
check("router/answers come from the scan", out["answers"], {"scanned": {"noul": 0.5}})
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
check("router/per-call end hook sees the scan", ended["answers"], {"scanned": {"noul": 0.5}})

# 8. an agent with no predict_long is a named caller error, not a bare AttributeError
class _NoScan:
    def system_one(self, state, questions):
        return {"model": "noscan", "answers": {}, "usage": {}}


r_noscan = Router()
r_noscan.attach("english", _NoScan())
check_raises("router/agent without predict_long", TypeError,
             lambda: r_noscan.predict_long({"body": "y" * 300}, Q, model="english"))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
