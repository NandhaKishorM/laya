"""predict_long: scan a state longer than the window and aggregate per question.

Weight-free. The real forward path is stubbed (predict_batch returns canned per-window answers),
so this checks only predict_long's own logic: the fits-in-one-window short-circuit, the overlapping
window split, the per-type aggregation (noul = strongest window, choice/score = most-confident
window), and the per-call hook controls it forwards to whichever of those two calls runs. Numerical
behaviour on real weights is exercised in tests/test_local_e2e.py.
"""
import os
import sys
import warnings

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
    a._calls = {"system_one": 0, "batch_states": None, "system_one_kwargs": None,
                "batch_kwargs": None}

    def _system_one(state, questions, lang=None, **controls):
        a._calls["system_one"] += 1
        a._calls["system_one_kwargs"] = controls
        return {"model": "laya-rl-agent", "answers": {"_via": "system_one"}, "usage": {"input_tokens": 1}}

    def _predict_batch(states, questions, batch_size=None, lang=None, **controls):
        a._calls["batch_states"] = list(states)
        a._calls["batch_kwargs"] = controls
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

# 4. the per-call hook controls `predict`/`system_one`/`predict_batch` take reach the scan
HOOK_KEYS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")
LONG = {"body": "y" * 300}
sentinel = object()
a = make_agent(canned)
res = a.predict_long(LONG, Q, hooks=[sentinel], hooks_raise=False, hooks_timeout=2.5)
got = a._calls["batch_kwargs"]
check("hooks/forwarded to the scan", sorted(got), sorted(HOOK_KEYS))
check("hooks/the hook list reaches it", got.get("hooks"), [sentinel])
check("hooks/hooks_raise reaches it", got.get("hooks_raise"), False)
check("hooks/hooks_timeout reaches it", got.get("hooks_timeout"), 2.5)
check("hooks/start/end default to None",
      [got.get("on_predict_start"), got.get("on_predict_end")], [None, None])

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

# 8. a start hook that answers the document is not attributed to a window that was never scored
def one_document_answer(states, q):
    return [{"answers": {"dept": {"type": "choice", "choice": "a", "answer_confidence": 0.7},
                         "flag": {"type": "noul", "noul": 0.2, "answer_confidence": 0.8}},
             "usage": {"input_tokens": 3, "output_tokens": 0}}]


a = make_agent(one_document_answer)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = a.predict_long(LONG, Q)
nwin = len(a._calls["batch_states"])
check("skip/the state was split into several windows", nwin > 1, True)
check("skip/no window was scored", res["usage"]["windows"], 0)
check("skip/the payload tokens are kept", res["usage"]["input_tokens"], 3)
check("skip/no deciding window is claimed",
      sorted(k for v in res["answers"].values() for k in v if k == "window"), [])
check("skip/the hook's answer is returned", res["answers"]["dept"]["choice"], "a")
check("skip/the caller is told", [w.category.__name__ for w in caught], ["RuntimeWarning"])

# 9. a payload that matches neither the document nor the windows is an error, not a guess
a = make_agent(lambda s, q: [{"answers": {}, "usage": {}}] * 2)
check_raises("skip/rejects a count matching nothing", ValueError, lambda: a.predict_long(LONG, Q))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
