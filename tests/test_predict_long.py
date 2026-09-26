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

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
