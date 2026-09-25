"""ONNXAgent.predict_batch(sort_by_length=True): the length-grouping knob, ported from the Agent.

`Agent.predict_batch` sorts encoded states inside windows of eight batches so each shared run
pads to a shorter maximum (#294). The port must keep three properties: results stay aligned with
the input order, the knob is a no-op outside `1 < batch_size < len(states)`, and grouping really
does reduce padded rows x width. No onnxruntime and no checkpoint: the same stub-session
technique as tests/test_onnx_batch.py, extended to record each run's padded width.

Run: python tests/test_onnx_sort.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

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


class _FakeTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(c) % 40) for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _StubSession:
    """Records (rows, padded width) per run; row logits derive from the row's own tokens."""

    def __init__(self):
        self.calls = []  # [(rows, width), ...]

    def run(self, names, inputs):
        ids = inputs["input_ids"]
        self.calls.append((len(ids), int(ids.shape[1])))
        weight = 1.0 + (ids.sum(axis=1) % 4)
        logits = np.stack([weight, np.ones_like(weight)], axis=1).astype(np.float32)
        act = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (len(ids), 1))
        return [logits, act]

    @property
    def padded_cells(self):
        return sum(rows * width for rows, width in self.calls)


def _bare_onnx():
    a = ONNXAgent.__new__(ONNXAgent)         # skip __init__: no onnxruntime, no checkpoint
    a.model_id = "stub"
    a.cfg = {"max_len": 128, "head_max_len": 32}
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
    "flag": {"type": "noul", "instructions": "Urgent?"},
}
ROWS_PER_STATE = 2  # two questions -> two head rows per state

# 12 states alternating short/long, characters cycling so adjacent states differ in mod-4
# weight: a row decoded for the wrong state changes the answer and fails the order checks.
_LETTERS = "abcd"
STATES = [(_LETTERS[i % 4]) * (4 if i % 2 == 0 else 40) for i in range(12)]


def results_of(sort, batch_size=4, states=STATES):
    agent = _bare_onnx()
    res = agent.predict_batch(states, QUESTIONS, batch_size=batch_size, sort_by_length=sort)
    return res, agent.session


# ---------------------------------------------------------------- no-op conditions
base, base_sess = results_of(False)
for bs, runs in ((None, 1), (1, len(STATES)), (len(STATES), 1)):
    agent = _bare_onnx()
    got = agent.predict_batch(STATES, QUESTIONS, batch_size=bs, sort_by_length=True)
    check("noop/batch_size=%r: sort_by_length changes nothing" % bs, got, base)
    check("noop/batch_size=%r: session runs unchanged by the knob" % bs,
          len(agent.session.calls), runs)

# default is off
agent = _bare_onnx()
check("default/sort_by_length defaults to the unsorted behaviour",
      agent.predict_batch(STATES, QUESTIONS, batch_size=4), base)

# ---------------------------------------------------------------- order and parity
sorted_res, sorted_sess = results_of(True)
check("order/results stay aligned with the input order", sorted_res, base)
check("shape/same number of runs sorted vs unsorted",
      [r for r, _ in sorted_sess.calls], [r for r, _ in base_sess.calls])

# ---------------------------------------------------------------- padding actually shrinks
check_true("padding/mixed lengths are really present unsorted (every batch pads to the long width)",
           all(w == base_sess.calls[0][1] > 40 for _, w in base_sess.calls),
           base_sess.calls)
check_true("padding/sorting narrows batches without widening any",
           min(w for _, w in sorted_sess.calls) < min(w for _, w in base_sess.calls)
           and max(w for _, w in sorted_sess.calls) <= max(w for _, w in base_sess.calls)
           and sum(w for _, w in sorted_sess.calls) < sum(w for _, w in base_sess.calls),
           (base_sess.calls, sorted_sess.calls))
check_true("padding/total padded cells drop with sorting",
           sorted_sess.padded_cells < base_sess.padded_cells,
           (sorted_sess.padded_cells, base_sess.padded_cells))
# shortest states first: the first sorted run pads to the short length only
check("padding/first sorted batch holds the short group (4 states x 2 rows)",
      sorted_sess.calls[0], (ROWS_PER_STATE * 4, sorted_sess.calls[0][1]))
check_true("padding/first sorted batch is narrower than the first unsorted one",
           sorted_sess.calls[0][1] < base_sess.calls[0][1],
           (sorted_sess.calls[0], base_sess.calls[0]))

# ---------------------------------------------------------------- lookahead window
# 40 states at batch_size=4 sorts within windows of 4*8=32, then a second window of 8.
many = [(_LETTERS[i % 4]) * (4 + 4 * (i % 5)) for i in range(40)]
agent = _bare_onnx()
res = agent.predict_batch(many, QUESTIONS, batch_size=4, sort_by_length=True)
check("window/40 states at batch_size=4 -> 10 runs", len(agent.session.calls), 10)
sequential = [_bare_onnx().system_one(s, QUESTIONS) for s in many]
check("window/results still match per-state calls across window boundaries", res, sequential)
check("window/within a window, lengths are non-decreasing across runs",
      [w for _, w in agent.session.calls[:8]], sorted(w for _, w in agent.session.calls[:8]))

# ---------------------------------------------------------------- hooks still see input order
seen = []


def _end(ctx):
    seen.append([r["answers"]["dept"]["probabilities"]["billing"] for r in ctx.results])


agent = _bare_onnx()
agent.predict_batch(STATES, QUESTIONS, batch_size=4, sort_by_length=True, on_predict_end=_end)
check("hooks/end hook sees results in input order",
      seen[0], [r["answers"]["dept"]["probabilities"]["billing"] for r in base])

# ---------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
