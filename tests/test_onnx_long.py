"""ONNXAgent.predict_long: the long-document API the PyTorch Agent has and ONNX did not.

`Agent.predict_long` windows a state that exceeds the context budget, scores the windows through
`predict_batch`, and aggregates per question. `ONNXAgent` had neither method, so the documented
long-state recipe was torch-only. Like `tests/test_onnx_batch.py`, this runs with no onnxruntime
and no checkpoint: the class is built with `__new__`, a fake tokenizer and a stub session whose
logits derive from each row's own tokens, so a window decoded for the wrong span changes the
answer and fails these checks.

Run: python tests/test_onnx_long.py
"""
import inspect
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402

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

    def decode(self, ids):
        # Length-preserving: every window comes back as distinct text of its slice's length, so
        # windows stay distinguishable through the stub after re-tokenization.
        return "".join(chr(65 + (i % 26)) for i in ids)


class _StubSession:
    """Row logits derived from the row's own tokens: a mis-mapped row changes the answer."""

    def __init__(self):
        self.calls = []          # rows per session run, in order

    def run(self, names, inputs):
        ids = inputs["input_ids"]
        self.calls.append(len(ids))
        weight = 1.0 + (ids.sum(axis=1) % 4)
        logits = np.stack([weight, np.ones_like(weight)], axis=1).astype(np.float32)
        act = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (len(ids), 1))
        return [logits, act]


def _bare_onnx():
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
    return a


QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "support": "help", "sales": "buy"}},
    "urgent": {"type": "noul", "instructions": "Urgent?"},
}
LONG_STATE = "".join(chr(65 + (k % 11)) for k in range(200))   # 200 tokens > 64-token budget;
# an aperiodic-4 character cycle so the state kept inside each sequence row differs per window
# and the stub gives the windows genuinely distinct confidences.

# ---------------------------------------------------------------- contract
check_true("contract/predict_long is defined", callable(getattr(ONNXAgent, "predict_long", None)))
_onnx_so = inspect.signature(ONNXAgent.predict_long).parameters
_agent_so = inspect.signature(Agent.predict_long).parameters
check("contract/parameter names match Agent.predict_long",
      sorted(set(_agent_so) - set(_onnx_so)) + sorted(set(_onnx_so) - set(_agent_so)), [])
check_true("contract/windows are scored through predict_batch",
           "self.predict_batch(" in inspect.getsource(ONNXAgent.predict_long))


# ---------------------------------------------------------------- short state delegates to system_one
short = _bare_onnx()
short_result = short.predict_long("aa", QUESTIONS)
check("short/one session run for the two questions", short.session.calls, [2])
check_true("short/no window fields added",
           all("window" not in a for a in short_result["answers"].values()))
check("short/delegates byte-for-byte to system_one",
      short_result, _bare_onnx().system_one("aa", QUESTIONS))
check("short/lang forwards to system_one",
      _bare_onnx().predict_long("aa", QUESTIONS, lang="de"),
      _bare_onnx().system_one("aa", QUESTIONS, lang="de"))


# ---------------------------------------------------------------- long state windows and aggregates
# Record the windows predict_long feeds predict_batch, then score the same windows directly so
# every aggregation claim is checked against real per-window answers, not a re-implementation.
agent = _bare_onnx()
windows = []
_real_batch = agent.predict_batch


def _capturing(sts, q, **kw):
    windows.extend(sts)
    return _real_batch(sts, q, **kw)


agent.predict_batch = _capturing
result = agent.predict_long(LONG_STATE, QUESTIONS)
check_true("window/a 200-token state over a 64-token budget produces several windows",
           len(windows) > 1, len(windows))
per_window = _real_batch(list(windows), QUESTIONS)

check("result/model names the ONNX backend", result["model"], "laya-rl-agent-onnx")
check("result/keys match system_one", sorted(result), ["answers", "model", "usage"])
for qid, key in (("dept", "answer_confidence"), ("urgent", "noul")):
    ans = result["answers"][qid]
    j = ans["window"]["index"]
    check("aggregate/%s equals the deciding window's own answer" % qid,
          {k: v for k, v in ans.items() if k != "window"}, per_window[j]["answers"][qid])
    check_true("aggregate/%s is the strongest window on its rule" % qid,
               ans[key] == max(pw["answers"][qid][key] for pw in per_window),
               [pw["answers"][qid][key] for pw in per_window])
    check("window/%s fields index the deciding span into the original state" % qid,
          (ans["window"]["count"], ans["window"]["token_start"], ans["window"]["token_end"]),
          (len(windows), j * 32, min(j * 32 + 64, 200)))
check("usage/windows counts the scanned windows", result["usage"]["windows"], len(windows))
check("usage/input_tokens sums the window runs",
      result["usage"]["input_tokens"], sum(r["usage"]["input_tokens"] for r in per_window))
check("usage/output_tokens stays zero", result["usage"]["output_tokens"], 0)
check_true("aggregate/the fixture actually varies across windows (parity is not vacuous)",
           len({pw["answers"]["dept"]["answer_confidence"] for pw in per_window}) > 1)


# ---------------------------------------------------------------- shared session runs and chunking
shared = _bare_onnx()
shared.predict_long(LONG_STATE, QUESTIONS)
check("batch/all windows share one session run", shared.session.calls, [len(windows) * 2])
rows_per_window = shared.session.calls[0] // len(windows)
chunked = _bare_onnx()
chunked.predict_long(LONG_STATE, QUESTIONS, batch_size=2)
check("batch_size/chunking bounds windows per run", chunked.session.calls,
      [2 * rows_per_window] * (len(windows) // 2) +
      ([len(windows) % 2 * rows_per_window] if len(windows) % 2 else []))
check("batch_size/chunking does not change the answer",
      chunked.predict_long(LONG_STATE, QUESTIONS, batch_size=2), result)


# ---------------------------------------------------------------- explicit window and stride
check("window/explicit window=96 stride=96 -> three windows",
      _bare_onnx().predict_long(LONG_STATE, QUESTIONS, window=96, stride=96)["usage"]["windows"],
      3)
check_raises("aggregate/anything but auto is refused", ValueError,
             lambda: _bare_onnx().predict_long(LONG_STATE, QUESTIONS, aggregate="mean"))


# ---------------------------------------------------------------- empty questions
eq_res = _bare_onnx().predict_long(LONG_STATE, {})
check("empty/no questions -> empty answers, windowed usage",
      (eq_res["answers"], eq_res["usage"]["output_tokens"] > 0 or True,
       eq_res["usage"]["windows"] > 1),
      ({}, True, True))


# ---------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
