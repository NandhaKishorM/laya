"""Regression tests for the state-reuse and autocast changes. No weights are downloaded.

Covers:
  * the state is serialized/tokenized once per call, not once per question
  * `build_sequence(..., state_ids=...)` is equivalent to the tokenizing path
  * the autocast path works and degrades to full precision instead of failing
"""
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import MPS_AMP_MIN_ROWS_DEFAULT, Agent, _amp_context, _mps_amp_min_rows  # noqa: E402
from laya.common import DecisionModel, build_sequence, serialize_state  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


# ------------------------------------------------------------------ fake tokenizer
class FakeTok:
    """Just enough of a tokenizer for `build_sequence` / `system_one`."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        self.calls.append(text)
        n = max(1, len(text) // 4)
        if truncation and max_length:
            n = min(n, max_length)
        return {"input_ids": [5] * n}


class FakeModel:
    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits = torch.where(
            marker_mask,
            torch.ones_like(marker_mask, dtype=torch.float32),
            torch.full_like(marker_mask, -1e4, dtype=torch.float32),
        )
        return logits, torch.zeros((input_ids.shape[0], 2))


def _bare_agent(model, dtype=torch.float32, amp=False, tok=None):
    a = object.__new__(Agent)
    a.device = torch.device("cpu")
    a.dtype = dtype
    a.amp_enabled = amp
    a.cfg = {"max_len": 64, "head_max_len": 32}
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    a.tok = tok or FakeTok()
    a.model = model
    return a


# ------------------------------------------------------------------ state reused once
STATE = {"subject": "Duplicate charge", "body": "x" * 400}
QUESTION = {"t": "choice", "ins": "pick", "crit": {"a": "x", "b": "y"}}

tok_ref = FakeTok()
seq_ref, markers_ref = build_sequence(tok_ref, STATE, QUESTION, 64, 32)

tok_shared = FakeTok()
state_ids = tok_shared(serialize_state(STATE), add_special_tokens=False)["input_ids"]
seq_shared, markers_shared = build_sequence(tok_shared, STATE, QUESTION, 64, 32, state_ids=state_ids)
check("build_sequence/state_ids identical ids", seq_shared, seq_ref)
check("build_sequence/state_ids identical markers", markers_shared, markers_ref)
check("build_sequence/state_ids does not re-tokenize state", tok_shared.calls.count(serialize_state(STATE)), 1)

agent = _bare_agent(FakeModel())
out = agent.system_one("the customer was charged twice", {
    "department": {"type": "choice", "instructions": "which?", "criteria": {"billing": "x", "technical": "y"}},
    "urgent": {"type": "noul", "instructions": "is it urgent?"},
})
state_text = serialize_state("the customer was charged twice").replace(agent.tok.mask_token, " ")
check("system_one/state tokenized once for two questions", agent.tok.calls.count(state_text), 1)
check("system_one/answers present", sorted(out["answers"]), ["department", "urgent"])


class RecordingTok(FakeTok):
    def __init__(self):
        super().__init__()
        self.caps = []

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        self.caps.append((truncation, max_length))
        return super().__call__(text, add_special_tokens=add_special_tokens,
                                truncation=truncation, max_length=max_length)


long_q = {"t": "choice", "ins": "pick", "crit": {"a": "x" * 400, "b": "y" * 400}}
rtok = RecordingTok()
long_seq, long_markers = build_sequence(rtok, "state", long_q, 300, 200)
# options are capped at the tokenizer (truncation=True, max_length=48), not sliced afterwards
check("build_sequence/options capped at the tokenizer",
      len([c for c in rtok.caps if c == (True, 48)]), len(long_markers))
check("build_sequence/option segments stay <= 49 tokens",
      all(long_markers[i + 1] - long_markers[i] <= 49 for i in range(len(long_markers) - 1)), True)


# ------------------------------------------------------------------ autocast
class DummyEnc(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d, num_attention_heads=1)
        self.emb = nn.Embedding(16, d)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


model = DecisionModel(DummyEnc(), head_layers=1, n_act=2)
iid = torch.tensor([[1, 2, 3, 4, 5]])
am = torch.ones_like(iid)
mp = torch.tensor([[1, 3]])
mm = torch.ones_like(mp, dtype=torch.bool)
qt = torch.tensor([0])
with torch.no_grad():
    logits32, act32 = model(iid, am, mp, mm, qt)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        logits16, act16 = model(iid, am, mp, mm, qt)
check("autocast/shapes match", tuple(logits16.shape), tuple(logits32.shape))
check("autocast/finite", bool(torch.isfinite(logits16).all() and torch.isfinite(act16).all()), True)
check("autocast/close to fp32", float((logits32 - logits16).abs().max()) < 0.5, True)

calls = {"n": 0}


class Flaky:
    def __call__(self, *args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("autocast not supported on this build")
        return torch.zeros((1, 2)), torch.zeros((1, 2))


batch = {
    "input_ids": torch.tensor([[1]]),
    "attention_mask": torch.ones((1, 1), dtype=torch.long),
    "marker_pos": torch.zeros((1, 1), dtype=torch.long),
    "marker_mask": torch.ones((1, 1), dtype=torch.bool),
    "qtype": torch.tensor([0]),
}
flaky = _bare_agent(Flaky(), dtype=torch.bfloat16, amp=True)
flaky._infer(batch)
check("infer/falls back and disables amp", flaky.amp_enabled, False)
check("infer/retried once", calls["n"], 2)


class Boom:
    def __call__(self, *args):
        raise RuntimeError("genuine failure")


boom = _bare_agent(Boom())
raised = False
try:
    boom._infer(batch)
except RuntimeError:
    raised = True
check("infer/non-autocast error propagates", raised, True)


# ------------------------------------------------------------------ MPS autocast gating
# MPS fp16 is slower than fp32 on one small row and only wins once the batch grows, so it is
# gated by row count. These check the gate without an MPS device.
def _mps_agent(amp=True, min_rows=None):
    a = _bare_agent(FakeModel(), dtype=torch.float16, amp=amp)
    a.device = torch.device("mps")
    if min_rows is not None:
        a.mps_amp_min_rows = min_rows
    return a


a = _mps_agent()
check("mps-gate/one row stays fp32", a._amp_enabled_for(1), False)
check("mps-gate/below threshold stays fp32", a._amp_enabled_for(a.mps_amp_min_rows - 1), False)
check("mps-gate/at threshold enables fp16", a._amp_enabled_for(a.mps_amp_min_rows), True)
check("mps-gate/above threshold enables fp16", a._amp_enabled_for(a.mps_amp_min_rows + 3), True)

check("mps-gate/threshold override", _mps_agent(min_rows=2)._amp_enabled_for(2), True)
check("mps-gate/amp disabled stays off", _mps_agent(amp=False)._amp_enabled_for(100), False)

cpu = _bare_agent(FakeModel(), dtype=torch.bfloat16, amp=True)
check("cpu-gate/not gated by rows", cpu._amp_enabled_for(1), True)

os.environ["LAYA_MPS_AMP_MIN_ROWS"] = "2"
check("mps-gate/env override", _mps_amp_min_rows(), 2)
os.environ["LAYA_MPS_AMP_MIN_ROWS"] = "nonsense"
check("mps-gate/env invalid falls back", _mps_amp_min_rows(), MPS_AMP_MIN_ROWS_DEFAULT)
del os.environ["LAYA_MPS_AMP_MIN_ROWS"]
check("mps-gate/env default", _mps_amp_min_rows(), MPS_AMP_MIN_ROWS_DEFAULT)


# ------------------------------------------------------------------ amp context shape
# A disabled gate must never construct torch.autocast: on torch builds without an MPS
# autocast backend, entering it raises even with enabled=False, which broke MPS predict().
mps = torch.device("mps")
check("amp-context/disabled mps is a no-op", isinstance(_amp_context(mps, torch.float16, False), nullcontext), True)
with _amp_context(mps, torch.float16, False):  # must not raise
    pass
check("amp-context/disabled mps enters cleanly", True, True)

check("amp-context/disabled cpu is a no-op",
      isinstance(_amp_context(torch.device("cpu"), torch.float32, False), nullcontext), True)
check("amp-context/enabled cpu is autocast",
      not isinstance(_amp_context(torch.device("cpu"), torch.bfloat16, True), nullcontext), True)

# the disabled path through _infer completes on a plain CPU agent
cpu_disabled = _bare_agent(FakeModel())  # amp=False
cpu_disabled._infer(batch)
check("amp-context/_infer disabled completes", True, True)


# ------------------------------------------------------------------ report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all runtime-fix tests passed")
sys.exit(1 if FAIL else 0)
