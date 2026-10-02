"""`state_token_counts`: the two state-token counts are opt-in, and asking makes them exact (#687).

`usage["state_tokens"]` and `usage["state_tokens_dropped"]` count the WHOLE serialized state. The
head-budget optimization tokenizes only as far as a prefix-truncated sequence can hold, so on that
path those counts would be floors (`>= max_len`) rather than counts. Rather than publish a number
that is exact on one path and a floor on another, they are reported only when the caller asks --
and asking restores the full tokenization for that call, so what comes back is exact.

`truncated` / `truncated_questions` stay always-on and free, and that is not a compromise:
`encode_state_head` returns at least `max_len` ids and every question's room is strictly below
`max_len`, so on the prefix path the clamp really did drop something and saying so is correct.

Weight-free. What is pinned here is the plumbing -- which tokenizer call runs, which `usage` keys
come back, and that every entry point forwards the flag. Numerical exactness against a real
tokenizer cannot be reached from a stub (all three of `_state_cut`'s soundness gates inspect
tokenizer internals a stub does not have, so a stub always takes the full path); that is measured
on the shipped checkpoints in the PR's end-to-end harness.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import laya.agent as agent_mod  # noqa: E402
from laya.agent import Agent  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402
from laya.router import Router, _ScanLong  # noqa: E402

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


class _Tok:
    """One token per character, so a state's token count is its length."""

    mask_token = "[M]"
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 101, 102, 103, 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = list(range(len(text)))
        return {"input_ids": ids[:max_length] if (truncation and max_length) else ids}

    def decode(self, ids):
        return "w"


Q = {"dept": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
     "flag": {"type": "noul", "instructions": "?"}}


def make_agent():
    """A real `predict_batch` / `_encode_state` with only the forward and the decode stubbed."""
    a = Agent.__new__(Agent)
    a.cfg = {"max_len": 100, "head_max_len": 20}
    a.tok = _Tok()
    a.model_id = "convaiinnovations/laya"
    a.hooks, a.hooks_raise, a.hooks_timeout = [], True, None
    a._hooks_lock = threading.Lock()

    def _forward(b):
        n = b["input_ids"].shape[0]
        return np.zeros((n, 2), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode_answers(logits, act, items, ids, internal, row, **kw):
        return {qid: {"type": "noul", "noul": 0.5, "confidence": 0.5, "answer_confidence": 0.5}
                for qid in ids}

    a._forward = _forward
    a._decode_answers = _decode_answers
    return a


STATE = "word " * 400          # 2000 characters == 2000 tokens, far past max_len 100


def encoders_used(call):
    """Which of the two tokenizer entry points ran during `call`.

    Patched on `laya.agent`, not on `laya.common`: `agent.py` does
    `from .common import encode_state_head, encode_text`, so it holds its own names and patching
    the defining module would leave this test measuring nothing.
    """
    used = {"head": 0, "full": 0}
    real_head, real_full = agent_mod.encode_state_head, agent_mod.encode_text

    def head(tok, text, need):
        used["head"] += 1
        return real_head(tok, text, need)

    def full(tok, text, **kw):
        used["full"] += 1
        return real_full(tok, text, **kw)

    agent_mod.encode_state_head, agent_mod.encode_text = head, full
    try:
        call()
    finally:
        agent_mod.encode_state_head, agent_mod.encode_text = real_head, real_full
    return used


# --------------------------------------------------------------- 1. the default omits the counts
a = make_agent()
default = a.predict_batch([STATE], Q)[0]["usage"]
# `options` rides along because this stub's 20-token head budget collapses the two option spans
# into one (#538) -- correct, and unrelated to the counts, so it is named rather than filtered.
check("default/usage keys", sorted(default),
      ["input_tokens", "options", "output_tokens", "truncated", "truncated_questions"])
check_true("default/state_tokens is absent", "state_tokens" not in default, default)
check_true("default/state_tokens_dropped is absent", "state_tokens_dropped" not in default, default)
check("default/truncated is still reported", default["truncated"], True)
check("default/truncated_questions is still reported", sorted(default["truncated_questions"]),
      ["dept", "flag"])

# --------------------------------------------------------------- 2. asking reports them, exactly
asked = a.predict_batch([STATE], Q, state_token_counts=True)[0]["usage"]
check("asked/usage keys", sorted(asked),
      ["input_tokens", "options", "output_tokens", "state_tokens", "state_tokens_dropped",
       "truncated", "truncated_questions"])
check("asked/state_tokens is the whole state, not a floor", asked["state_tokens"], len(STATE))
check_true("asked/something was dropped", asked["state_tokens_dropped"] > 0, asked)
check("asked/truncated agrees with the default", asked["truncated"], default["truncated"])
check("asked/truncated_questions agrees with the default",
      asked["truncated_questions"], default["truncated_questions"])
check("asked/input_tokens is unchanged", asked["input_tokens"], default["input_tokens"])

# --------------------------------------------------------------- 3. which tokenization ran
off = encoders_used(lambda: make_agent().predict_batch([STATE], Q))
on = encoders_used(lambda: make_agent().predict_batch([STATE], Q, state_token_counts=True))
check_true("path/the default goes through encode_state_head", off["head"] == 1 and off["full"] == 0,
           "default used %r" % (off,))
check_true("path/asking goes through the full tokenization instead",
           on["full"] == 1 and on["head"] == 0, "asking used %r" % (on,))

# A conversation list is truncated from the left, so it has never had a prefix to take: asking
# changes nothing about how it is tokenized, only whether the counts come back.
convo = [{"role": "user", "content": STATE}]
list_off = encoders_used(lambda: make_agent().predict_batch([convo], Q))
list_on = encoders_used(lambda: make_agent().predict_batch([convo], Q, state_token_counts=True))
check("path/a conversation list tokenizes in full either way",
      (list_off["head"], list_off["full"], list_on["head"], list_on["full"]), (0, 1, 0, 1))
check_true("list/asking still reports the counts",
           "state_tokens" in make_agent().predict_batch([convo], Q,
                                                        state_token_counts=True)[0]["usage"])

# --------------------------------------------------------------- 4. a state that fits
short = make_agent().predict_batch(["hi"], Q, state_token_counts=True)[0]["usage"]
check("fits/nothing dropped", short["state_tokens_dropped"], 0)
check("fits/not truncated", short["truncated"], False)
check("fits/state_tokens is the state", short["state_tokens"], 2)

# --------------------------------------------------------------- 5. system_one forwards it
one_off = make_agent().system_one(STATE, Q)["usage"]
one_on = make_agent().system_one(STATE, Q, state_token_counts=True)["usage"]
check_true("system_one/default omits the counts", "state_tokens" not in one_off, one_off)
check("system_one/asking reports the whole state", one_on["state_tokens"], len(STATE))

# --------------------------------------------------------------- 6. predict_long forwards it
class _LongAgent(Agent):
    """Records what `predict_long` hands to `predict_batch`."""

    def __init__(self):
        self.cfg = {"max_len": 100, "head_max_len": 20}
        self.tok = _Tok()
        self.hooks, self.hooks_raise, self.hooks_timeout = [], True, None
        self._hooks_lock = threading.Lock()
        self.seen = {}

    def predict_batch(self, states, questions, batch_size=None, lang=None, **controls):
        self.seen = dict(controls)
        return [{"model": "m", "answers": {"dept": {"type": "noul", "noul": 0.5, "confidence": 0.5,
                                                    "answer_confidence": 0.5}},
                 "usage": {"input_tokens": 1}} for _ in states]


long_off = _LongAgent()
long_off.predict_long(STATE, {"dept": Q["dept"]})
check_true("predict_long/default does not forward it",
           "state_token_counts" not in long_off.seen, long_off.seen)
long_on = _LongAgent()
long_on.predict_long(STATE, {"dept": Q["dept"]}, state_token_counts=True)
check("predict_long/asking forwards it", long_on.seen.get("state_token_counts"), True)


# --------------------------------------------------------- 6b. the ONNX predict_long forwards it
class _OnnxLongAgent(ONNXAgent):
    """Records what the ONNX `predict_long` hands to its own `predict_batch`."""

    def __init__(self):
        self.cfg = {"max_len": 100, "head_max_len": 20}
        self.tok = _Tok()
        self.hooks, self.hooks_raise, self.hooks_timeout = [], True, None
        self._hooks_lock = threading.Lock()
        self.seen = {}

    def predict_batch(self, states, questions, batch_size=None, lang=None, **controls):
        self.seen = dict(controls)
        return [{"model": "m", "answers": {"dept": {"type": "noul", "noul": 0.5, "confidence": 0.5,
                                                    "answer_confidence": 0.5}},
                 "usage": {"input_tokens": 1}} for _ in states]


_onnx_off = _OnnxLongAgent()
_onnx_off.predict_long(STATE, {"dept": Q["dept"]})
check_true("onnx predict_long/default does not forward it",
           "state_token_counts" not in _onnx_off.seen, _onnx_off.seen)
_onnx_on = _OnnxLongAgent()
_onnx_on.predict_long(STATE, {"dept": Q["dept"]}, state_token_counts=True)
check("onnx predict_long/asking forwards it", _onnx_on.seen.get("state_token_counts"), True)


# --------------------------------------------------------------- 7. every Router entry point
class _RecordingAgent:
    """An Agent-like object that records the controls each Router path hands it."""

    def __init__(self):
        self.calls = []

    def _result(self):
        return {"model": "m", "answers": {}, "usage": {"input_tokens": 1}}

    def system_one(self, state, questions, **controls):
        self.calls.append(("system_one", controls))
        return self._result()

    def predict_batch(self, states, questions, batch_size=None, **controls):
        self.calls.append(("predict_batch", controls))
        return [self._result() for _ in states]

    def predict_long(self, state, questions, **controls):
        self.calls.append(("predict_long", controls))
        return self._result()


def routed(method, asked, **extra):
    stub = _RecordingAgent()
    r = Router()
    r.attach("english", stub)
    kwargs = dict(extra)
    if asked:
        kwargs["state_token_counts"] = True
    getattr(r, method)(**kwargs)
    return stub.calls[-1][1]


check("router.predict/asking reaches the agent",
      routed("predict", True, state="hi", questions=Q, model="english").get("state_token_counts"),
      True)
check_true("router.predict/the default does not pass it",
           "state_token_counts" not in routed("predict", False, state="hi", questions=Q,
                                              model="english"))
check("router.predict_long/asking reaches the agent",
      routed("predict_long", True, state="hi", questions=Q,
             model="english").get("state_token_counts"), True)
check_true("router.predict_long/the default does not pass it",
           "state_token_counts" not in routed("predict_long", False, state="hi", questions=Q,
                                              model="english"))
check("router.predict_batch/asking reaches the agent",
      routed("predict_batch", True,
             requests=[{"state": "hi", "questions": Q, "model": "english"}]).get(
                 "state_token_counts"), True)
check_true("router.predict_batch/the default does not pass it",
           "state_token_counts" not in routed("predict_batch", False,
                                              requests=[{"state": "hi", "questions": Q,
                                                         "model": "english"}]))

# `_ScanLong` is what calls the agent's `predict_long`, so the flag has to survive the hop through
# it -- `Router.predict_long` cannot pass it to `predict`, which never reaches `system_one` here.
check("router/_ScanLong carries it", _ScanLong(window=None, stride=None, aggregate="auto",
                                               batch_size=None, lang=None,
                                               state_token_counts=True).state_token_counts, True)
check("router/_ScanLong defaults to off", _ScanLong(window=None, stride=None, aggregate="auto",
                                                    batch_size=None, lang=None).state_token_counts,
      False)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
