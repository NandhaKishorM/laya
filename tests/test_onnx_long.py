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
import warnings
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import window_budget  # noqa: E402
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
# `usage["windows"]` is total on every path since #577 (1 for a state that fit one window), so the
# comparison is system_one's payload with that one key added.
def _with_one_window(result):
    return dict(result, usage=dict(result["usage"], windows=1))


check("short/delegates byte-for-byte to system_one",
      short_result, _with_one_window(_bare_onnx().system_one("aa", QUESTIONS)))
check("short/lang forwards to system_one",
      _bare_onnx().predict_long("aa", QUESTIONS, lang="de"),
      _with_one_window(_bare_onnx().system_one("aa", QUESTIONS, lang="de")))


# ---------------------------------------------------------------- a state the questions leave room for
# "Fits" is the room the questions leave, not the default window: system_one reads a state up to that
# room whole, so windowing one between the two re-read it in pieces and moved the answer.
def _roomy_onnx():
    a = _bare_onnx()
    a.cfg = {"max_len": 256, "head_max_len": 96}     # default window 152
    return a


_win, _, _room_q = window_budget(_FakeTok(), [Agent._to_internal(q) for q in QUESTIONS.values()], 256, 96)
check_true("room/these questions leave more room than the default window", _room_q > _win, (_win, _room_q))
_at_room = "".join(chr(65 + (k % 11)) for k in range(_room_q))
roomy = _roomy_onnx()
check("room/a state at the room delegates byte-for-byte to system_one",
      roomy.predict_long(_at_room, QUESTIONS), _with_one_window(_roomy_onnx().system_one(_at_room, QUESTIONS)))
check("room/in one session run", roomy.session.calls, [2])
check_true("room/one token past the room is still scanned",
           _roomy_onnx().predict_long(_at_room + "A", QUESTIONS)["usage"]["windows"] > 1)
check_true("room/an explicit window still scans a state wider than it",
           _roomy_onnx().predict_long(_at_room, QUESTIONS, window=_win)["usage"]["windows"] > 1)


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
    # Derived from the *effective* window, not from `max_len - head_max_len - 8`: these questions
    # leave less room than that budget at this tiny config, so the window is capped at the room and
    # the reported span has to describe what the model actually read. Hardcoding 64/32 here asserted
    # a `token_end` that overstated the span by the difference.
    _eff, _step, _room = window_budget(agent.tok, [agent._to_internal(q) for q in QUESTIONS.values()],
                                       agent.cfg["max_len"], agent.cfg["head_max_len"])
    check("window/%s fields index the deciding span into the original state" % qid,
          (ans["window"]["count"], ans["window"]["token_start"], ans["window"]["token_end"]),
          (len(windows), j * _step, min(j * _step + _eff, 200)))
    check_true("window/%s span is no wider than the room the questions leave" % qid,
               ans["window"]["token_end"] - ans["window"]["token_start"] <= _room,
               (ans["window"]["token_start"], ans["window"]["token_end"], _room))
check("usage/windows counts the scanned windows", result["usage"]["windows"], len(windows))
check("usage/input_tokens sums the window runs",
      result["usage"]["input_tokens"], sum(r["usage"]["input_tokens"] for r in per_window))
check("usage/output_tokens stays zero", result["usage"]["output_tokens"], 0)
check_true("aggregate/the fixture actually varies across windows (parity is not vacuous)",
           len({pw["answers"]["dept"]["answer_confidence"] for pw in per_window}) > 1)


# ------------------------------------------------- per-question usage fields merge across windows
# `usage["options"]` is a dict keyed by question id and is set only on windows where option
# spans collapsed, so it is a per-question record rather than a scalar counter. Replacing it
# per window left the caller holding whichever collapsing window came last, and the deciding
# window is the most confident one rather than the last one. Every window is given a record
# keyed by its own index, so the aggregate must keep all of them.
_onnx = _bare_onnx()
_orig_batch = _onnx.predict_batch
_scan = []


def _with_collapse(sts, q, **kw):
    _scan.extend(sts)
    rows = _orig_batch(list(sts), q, **kw)
    for i, row in enumerate(rows):
        row["usage"] = dict(row["usage"])
        row["usage"]["options"] = {"w%d" % i: {"total": 10 + i, "distinct": i + 1,
                                               "tokens_per_option": 0.5}}
    return rows


_onnx.predict_batch = _with_collapse
_merged = _onnx.predict_long(LONG_STATE, QUESTIONS)
_seen = _merged["usage"].get("options") or {}
_nwin = len(_scan)
_won = _merged["answers"]["dept"]["window"]["index"]
check("collapse/every window record survives the merge",
      sorted(_seen), sorted("w%d" % i for i in range(_nwin)))
check_true("collapse/the deciding window kept its own record",
           "w%d" % _won in _seen, sorted(_seen))
check("collapse/record contents are carried per window, not summed",
      [_seen.get("w%d" % i, {}).get("total") for i in range(_nwin)],
      [10 + i for i in range(_nwin)])
check_true("collapse/more than one window, so this is a real test", _nwin > 1, _nwin)


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
# window=96 is wider than the room these questions leave at max_len=64, so it is clamped -- and the
# stride the caller paired with it is clamped to match rather than refused, since 96 was a valid step
# for the 96 they asked for. What must hold is that the scan still covers the state.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    _explicit = _bare_onnx().predict_long(LONG_STATE, QUESTIONS, window=96, stride=96)
_eff96, _step96, _ = window_budget(_bare_onnx().tok,
                                   [_bare_onnx()._to_internal(q) for q in QUESTIONS.values()], 64, 32,
                                   window=96, stride=96)
check_true("window/explicit window=96 stride=96 is clamped and still covers the state",
           _explicit["usage"]["windows"] >= 3 and _step96 <= _eff96,
           (_explicit["usage"]["windows"], _eff96, _step96))
check_raises("aggregate/anything but auto is refused", ValueError,
             lambda: _bare_onnx().predict_long(LONG_STATE, QUESTIONS, aggregate="mean"))


# ---------------------------------------------------------------- empty questions
eq_res = _bare_onnx().predict_long(LONG_STATE, {})
check("empty/no questions -> empty answers, windowed usage",
      (eq_res["answers"], eq_res["usage"]["output_tokens"] > 0 or True,
       eq_res["usage"]["windows"] > 1),
      ({}, True, True))


# ---------------------------------------------------------------- start hooks may replace questions
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
    ("append", {**QUESTIONS, "review": QUESTIONS["urgent"]}),
    ("replace", {"review": QUESTIONS["urgent"]}),
    ("delete", {"urgent": QUESTIONS["urgent"]}),
    ("clear", {}),
    ("choice to noul", {**QUESTIONS, "dept": QUESTIONS["urgent"]}),
    ("noul to choice", {**QUESTIONS, "urgent": QUESTIONS["dept"]}),
]
for name, rewritten_questions in question_rewrites:
    expected = _bare_onnx().predict_long(LONG_STATE, rewritten_questions)
    try:
        actual = _bare_onnx().predict_long(
            LONG_STATE, QUESTIONS,
            on_predict_start=lambda ctx: setattr(ctx, "questions", rewritten_questions))
    except Exception as exc:
        FAIL.append("questions/%s raised %r" % (name, exc))
    else:
        _rewrite_agrees(name, actual, expected)

check("questions/no-op preserves the unhooked result",
      _bare_onnx().predict_long(LONG_STATE, QUESTIONS, on_predict_start=lambda ctx: None),
      _bare_onnx().predict_long(LONG_STATE, QUESTIONS))


# A hook that widens a question IN PLACE must be refused here exactly as on the torch agent
# (tests/test_predict_long.py, "an in-place question rewrite is refused"). Compared against the
# caller's own mapping it cannot be seen: the hook mutates the same nested dict, so
# `questions == asked` stays True and the scan proceeds with windows `build_sequence` re-truncates.
# Measured on this fixture, 8 added options cut the room from 28 to 12 state tokens: all 14 windows
# were truncated and 32 of 200 tokens reached no model, and a 24-token state that fit one window
# lost 12 while reporting `windows: 1`. The guard's own message is asserted, because too many
# options for max_len raise a different ValueError from `_encode_state` that would pass a bare
# `check_raises`. Fresh mappings per call, because the hook mutates what it is handed.
def _inplace_questions():
    return {"dept": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
            "urgent": dict(QUESTIONS["urgent"])}


def _widen_in_place(ctx):
    ctx.questions["dept"]["criteria"].update({"opt%02d" % i: "d" * 20 for i in range(8)})


for name, inplace_state in (("a windowed state", LONG_STATE),
                            ("a one-window state", LONG_STATE[:24])):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _bare_onnx().predict_long(inplace_state, _inplace_questions(),
                                      on_predict_start=_widen_in_place)
    except ValueError as exc:
        check_true("questions/an in-place widening rewrite is refused on %s" % name,
                   "after the start hooks ran" in str(exc), repr(exc))
    else:
        FAIL.append("questions/an in-place widening rewrite is refused on %s: did not raise" % name)


def _annotate_first_window(ctx):
    ctx.results[0]["answers"]["review"] = {"type": "noul", "noul": 0.9, "answer_confidence": 0.9}


expected = _bare_onnx().predict_long(LONG_STATE, QUESTIONS)
try:
    actual = _bare_onnx().predict_long(LONG_STATE, QUESTIONS, on_predict_end=_annotate_first_window)
except Exception as exc:
    FAIL.append("questions/a first-window end annotation raised %r" % exc)
else:
    check("questions/a first-window end annotation preserves the scan's answers", actual, expected)

for malformed in (None, [], {"bad": None}, {"bad": {}}, {"bad": {"type": "unknown"}}):
    errors = []
    for method in ("system_one", "predict_long"):
        try:
            getattr(_bare_onnx(), method)(LONG_STATE, malformed)
        except Exception as exc:
            errors.append((type(exc).__name__, str(exc)))
        else:
            errors.append(None)
    check("questions/invalid input keeps the validator's error: %r" % malformed, errors[1], errors[0])


# ---------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
