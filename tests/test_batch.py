"""Batch inference: predict_batch packs many states into shared forward passes.

These tests need no model weights. They cover:
  1. Contract — predict_batch exists and system_one is defined in terms of it, so the single-state
     and batched paths can never numerically drift apart.
  2. Orchestration — empty input, bad input, per-state row mapping, and batch_size chunking, all
     exercised against a fake whose forward is stubbed. The numerical equivalence of the real
     model path (predict_batch(states) == [system_one(s) for s in states]) is checked with weights
     in tests/test_local_e2e.py.
  3. Decoding — calibrated answer values and confidence, including mixed question types.
"""
import inspect
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya import agent as _agent  # noqa: E402
from laya.agent import Agent  # noqa: E402

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


# --------------------------------------------------------------- contract (source inspection)
check_true("contract/predict_batch is defined", callable(getattr(Agent, "predict_batch", None)))

_so = inspect.getsource(Agent.system_one)
check_true("contract/system_one delegates to predict_batch", "self.predict_batch(" in _so)
check_true("contract/predict alias preserved", Agent.predict is Agent.system_one)

_sig = inspect.signature(Agent.predict_batch)
check_true("contract/predict_batch has batch_size kwarg", "batch_size" in _sig.parameters)


# --------------------------------------------------------------- orchestration (weight-free fake)
# We drive the real predict_batch method but stub the three helpers it composes, so no encoder,
# tokenizer vocabulary, or weights are needed. collate_items runs for real on the fake items.
NQ = 2
QUESTIONS = {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "noul", "instructions": "?"}}


def make_fake():
    fake = _agent.Agent.__new__(_agent.Agent)
    fake.tok = type("Tok", (), {"pad_token_id": 0})()
    fake._to_internal = staticmethod(Agent._to_internal).__func__  # reuse the real normalizer
    fake._forward_calls = []

    def _encode_state(state, ids, internal):
        # one 3-token, 2-marker item per question; content is irrelevant to the mapping test
        return [{"ids": [1, 2, 3], "markers": [0, 1], "qtype": 2} for _ in ids]

    def _forward(b):
        n = b["input_ids"].shape[0]
        fake._forward_calls.append(n)
        # deterministic logits so decode is stable; act pre-softmaxed as _forward would return it
        return np.zeros((n, 2), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode_answers(logits, act, items, ids, internal, offset):
        # record the row offset this state was decoded from, to verify per-state alignment
        return {"_offset": offset}

    fake._encode_state = _encode_state
    fake._forward = _forward
    fake._decode_answers = _decode_answers
    return fake


# --------------------------------------------------------------- length grouping and restoration
def make_length_fake():
    fake = make_fake()
    fake.encoded = []
    fake.batch_rows = []
    fake.slots = 0
    fake.encoded_at_forward = []

    def encode(state, ids, internal, **overrides):
        fake.encoded.append(state)
        limit = overrides.get("max_len", 100)
        # Question rows have different lengths; sorting must use the longest.
        return [{"ids": [state["id"] + 1] * min(limit, size), "markers": [0, 1], "qtype": 2}
                for size in (3, state["length"])]

    def forward(batch):
        fake.slots += batch["input_ids"].numel()
        fake.batch_rows.append(batch["input_ids"][:, 0].tolist())
        fake.encoded_at_forward.append(len(fake.encoded))
        return batch["input_ids"][:, :1].numpy(), None

    def decode(logits, act, items, ids, internal, offset):
        return {qid: int(logits[offset + i, 0]) - 1 for i, qid in enumerate(ids)}

    fake._encode_state = encode
    fake._forward = forward
    fake._decode_answers = decode
    return fake


states = [{"id": i, "length": [40, 5, 25, 8][i % 4]} for i in range(37)]
plain, grouped = make_length_fake(), make_length_fake()
expected = plain.predict_batch(states, QUESTIONS, batch_size=2)
events = []
actual = grouped.predict_batch(states, QUESTIONS, batch_size=2, sort_by_length=True,
                               on_predict_end=lambda ctx: events.append((ctx.results, ctx.usage)))
check("length/answers and usage retain input order across windows", actual, expected)
check("length/each state encoded once in input order", grouped.encoded, states)
check_true("length/grouping reduces padded slots", grouped.slots < plain.slots)
check("length/same number of forward passes", len(grouped.batch_rows), len(plain.batch_rows))
check_true("length/batch size caps all forward passes", all(len(rows) <= 4 for rows in grouped.batch_rows))
check("length/partial final batch", len(grouped.batch_rows[-1]), 2)
check("length/first forward buffers at most eight batches", grouped.encoded_at_forward[0], 16)
check("length/end hook sees restored results", events[0][0], expected)
check("length/end hook usage excludes padding", events[0][1]["input_tokens"],
      sum(3 + st["length"] for st in states))
check("length/does not mutate caller states", [st["id"] for st in states], list(range(37)))

# A hook may replace both inputs and the length budget before grouping occurs.
def rewrite_length_inputs(ctx):
    ctx.states = states
    ctx.max_len = 10


rewritten = make_length_fake().predict_batch(["unused"], QUESTIONS, batch_size=2,
                                            sort_by_length=True, on_predict_start=rewrite_length_inputs)
reference = make_length_fake().predict_batch(states, QUESTIONS, batch_size=2, max_len=10)
check("length/hook rewrites and truncation budgets precede sorting", rewritten, reference)

for batch_size in (None, 0, -1, 1, 100):
    plain, grouped = make_length_fake(), make_length_fake()
    check("length/no-op results for batch_size=%r" % batch_size,
          grouped.predict_batch(states, QUESTIONS, batch_size=batch_size, sort_by_length=True),
          plain.predict_batch(states, QUESTIONS, batch_size=batch_size))
    check("length/no-op grouping for batch_size=%r" % batch_size, grouped.batch_rows, plain.batch_rows)

f = make_length_fake()
check("length/empty inputs", f.predict_batch([], QUESTIONS, sort_by_length=True), [])
check("length/empty questions", f.predict_batch(states[:1], {}, sort_by_length=True)[0]["answers"], {})
check("length/empty cases do not encode", f.encoded, [])
f.predict_batch(states, QUESTIONS, sort_by_length=True, on_predict_start=lambda ctx: ctx.skip([]))
check("length/cache skip does not encode", f.encoded, [])

plain, grouped = make_length_fake(), make_length_fake()
equal_lengths = [{"id": i, "length": 10} for i in range(20)]
plain.predict_batch(equal_lengths, QUESTIONS, batch_size=3)
grouped.predict_batch(equal_lengths, QUESTIONS, batch_size=3, sort_by_length=True)
check("length/equal length ties preserve row order", grouped.batch_rows, plain.batch_rows)

errors = []
class RecordFailure:
    def on_error(self, ctx):
        errors.append(ctx.error)


def fail_forward(batch):
    raise RuntimeError("forward failed")


f = make_length_fake()
f._forward = fail_forward
check_raises("length/forward error propagates", RuntimeError,
             lambda: f.predict_batch(states, QUESTIONS, batch_size=2,
                                     sort_by_length=True, hooks=[RecordFailure()]))
check("length/error hook receives failure once", [str(e) for e in errors], ["forward failed"])


f = make_fake()
check("orchestration/empty states -> empty list", f.predict_batch([], QUESTIONS), [])

f = make_fake()
res = f.predict_batch(["s0", "s1", "s2"], QUESTIONS)
check("orchestration/one result per state", len(res), 3)
check("orchestration/state 0 decoded from row 0", res[0]["answers"]["_offset"], 0)
check("orchestration/state 1 decoded from row NQ", res[1]["answers"]["_offset"], NQ)
check("orchestration/state 2 decoded from row 2*NQ", res[2]["answers"]["_offset"], 2 * NQ)
check_true("orchestration/result shape matches system_one",
           all(set(r) == {"model", "answers", "usage"} for r in res))
check("orchestration/single forward pass by default", f._forward_calls, [6])  # 3 states x 2 questions

f = make_fake()
f.predict_batch(["s0", "s1", "s2"], QUESTIONS, batch_size=1)
check("orchestration/batch_size=1 chunks into three passes", f._forward_calls, [2, 2, 2])

f = make_fake()
f.predict_batch(["s0", "s1", "s2", "s3", "s4"], QUESTIONS, batch_size=2)
check("orchestration/batch_size=2 chunks 5 states as 2+2+1", f._forward_calls, [4, 4, 2])

f = make_fake()
check_raises("orchestration/bare string rejected", TypeError, lambda: f.predict_batch("just a string", QUESTIONS))
f = make_fake()
check_raises("orchestration/bare dict rejected", TypeError, lambda: f.predict_batch({"body": "x"}, QUESTIONS))


# --------------------------------------------------------------- real answer decoding (no weights)
# Exercise the real decoder with a row offset and distinct per-type/bucket temperatures.
# Noul uses top probability for confidence; its unused entropy calculation can be skipped.
decoder = Agent.__new__(Agent)
decoder.temperature = {0: 1.0, 1: 2.0, 2: 3.0}
decoder.temperature_by_options = {"choice:2": 0.5}
decode_ids = ["pick", "level", "flag"]
decode_internal = {
    "pick": {"t": "choice", "crit": {"left": "left", "right": "right"}},
    "level": {"t": "score", "crit": ["low", "high"]},
    "flag": {"t": "noul", "crit": None},
}
decode_items = [{"markers": [0, 1]} for _ in decode_ids]
decode_logits = np.array([
    [99.0, -99.0],  # preceding state: must not be decoded
    np.log([0.25, 0.75]) * 0.5,
    np.log([0.25, 0.75]) * 2.0,
    np.log([0.8, 0.2]) * 3.0,
])
decode_act = np.array([[0.0, 1.0], [0.125, 0.875], [0.25, 0.75], [0.75, 0.25]])
with patch.object(_agent, "confidence_from_probs", wraps=_agent.confidence_from_probs) as entropy:
    decoded = decoder._decode_answers(
        decode_logits, decode_act, decode_items, decode_ids, decode_internal, 1
    )
    check("decode/mixed answers retain calibrated values", decoded, {
        "pick": {"type": "choice", "choice": "right",
                 "probabilities": {"left": 0.25, "right": 0.75}, "confidence": 0.1887,
                 "answer_confidence": 0.75,
                 "action": {"act_probability": 0.125}},
        "level": {"type": "score", "score": 0.75, "legend": {"0": "low", "1": "high"},
                  "probabilities": {"0": 0.25, "1": 0.75}, "confidence": 0.1887,
                  "answer_confidence": 0.75,
                  "action": {"act_probability": 0.25}},
        "flag": {"type": "noul", "noul": 0.2, "confidence": 0.8,
                 # max(p) over two options is max(p_true, 1 - p_true): the same number
                 "answer_confidence": 0.8,
                 "action": {"act_probability": 0.75}},
    })
    check("decode/entropy only computed for choice and score", entropy.call_count, 2)
    entropy.reset_mock()
    for true_probability in (0.1, 0.5, 0.9):
        logits = np.log([[1.0 - true_probability, true_probability]]) * 3.0
        result = decoder._decode_answers(
            logits, decode_act[-1:], decode_items[-1:], ["flag"], decode_internal, 0
        )
        check("decode/noul confidence at p=%s" % true_probability, result["flag"], {
            "type": "noul", "noul": true_probability,
            "confidence": max(true_probability, 1.0 - true_probability),
            "answer_confidence": max(true_probability, 1.0 - true_probability),
            "action": {"act_probability": 0.75},
        })
    check("decode/noul-only answers skip entropy", entropy.call_count, 0)


# --------------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f_ in FAIL:
    print("  FAIL " + f_)
sys.exit(1 if FAIL else 0)
