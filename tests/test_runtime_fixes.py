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
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import (  # noqa: E402
    MPS_AMP_MIN_ROWS_DEFAULT, Agent, _BATCH_AUTOCAST_CACHE, _amp_context, _cpu_amp_dtype,
    _cuda_amp_dtype, _mps_amp_min_rows,
)
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
    """Fail the autocast attempt of the first three requests; the fp32 retry succeeds."""

    def __call__(self, *args):
        calls["n"] += 1
        if calls["n"] in (1, 3, 5):
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
check("infer/one miss keeps amp", flaky.amp_enabled, True)
check("infer/one miss keeps dtype", flaky.dtype, torch.bfloat16)
check("infer/retried once", calls["n"], 2)
flaky._infer(batch)
check("infer/two misses keep amp", flaky.amp_enabled, True)
flaky._infer(batch)
check("infer/third miss disables amp", flaky.amp_enabled, False)
check("infer/third miss drops dtype", flaky.dtype, torch.float32)
check("infer/three misses retried", calls["n"], 6)
flaky._infer(batch)
check("infer/later request is one forward", calls["n"], 7)


class OneMiss:
    def __init__(self):
        self.n = 0

    def __call__(self, *args):
        self.n += 1
        if self.n == 1:
            raise RuntimeError("autocast not supported on this build")
        return torch.zeros((1, 2)), torch.zeros((1, 2))


streak = _bare_agent(OneMiss(), dtype=torch.bfloat16, amp=True)
streak._infer(batch)
check("infer/streak is one after a miss", streak._amp_failures, 1)
streak._infer(batch)
check("infer/a clean forward clears the streak", streak._amp_failures, 0)
check("infer/a clean forward keeps amp", streak.amp_enabled, True)


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

# `dtype` is the autocast target; `dtype_for(rows)` is the precision a forward with `rows` rows
# runs in (#621). Below the MPS gate that is fp32, even though `dtype` still says fp16.
a = _mps_agent()
check("dtype_for/mps below threshold is fp32", a.dtype_for(a.mps_amp_min_rows - 1), torch.float32)
check("dtype_for/mps at threshold is the target", a.dtype_for(a.mps_amp_min_rows), torch.float16)
check("dtype_for/mps huge threshold stays fp32", _mps_agent(min_rows=10 ** 9).dtype_for(10), torch.float32)
check("dtype_for/target unchanged", a.dtype, torch.float16)
check("dtype_for/amp disabled is fp32", _mps_agent(amp=False).dtype_for(100), torch.float32)
check("dtype_for/cpu bf16 not gated by rows", cpu.dtype_for(1), torch.bfloat16)
check("dtype_for/plain cpu is fp32", _bare_agent(FakeModel()).dtype_for(1), torch.float32)

os.environ["LAYA_MPS_AMP_MIN_ROWS"] = "2"
check("mps-gate/env override", _mps_amp_min_rows(), 2)
os.environ["LAYA_MPS_AMP_MIN_ROWS"] = "nonsense"
check("mps-gate/env invalid falls back", _mps_amp_min_rows(), MPS_AMP_MIN_ROWS_DEFAULT)
del os.environ["LAYA_MPS_AMP_MIN_ROWS"]
check("mps-gate/env default", _mps_amp_min_rows(), MPS_AMP_MIN_ROWS_DEFAULT)

os.environ.pop("LAYA_CUDA_AMP", None)
check("cuda-amp/checkpoint default wins when unset", _cuda_amp_dtype("bf16"), torch.bfloat16)
check("cuda-amp/no checkpoint value means fp16", _cuda_amp_dtype(None), torch.float16)
os.environ["LAYA_CUDA_AMP"] = "fp16"
check("cuda-amp/env fp16 overrides a bf16 checkpoint", _cuda_amp_dtype("bf16"), torch.float16)
os.environ["LAYA_CUDA_AMP"] = "BF16"
check("cuda-amp/env bf16 overrides an fp16 checkpoint", _cuda_amp_dtype("fp16"), torch.bfloat16)
os.environ["LAYA_CUDA_AMP"] = "int8"
check("cuda-amp/env invalid falls back to the checkpoint", _cuda_amp_dtype("bf16"), torch.bfloat16)
# `agent.dtype` reports float16 and bfloat16, so those are the spellings a caller can read off one
# response and ask for on the next. laya/agent.py has accepted both from the start; both prose
# sites that listed the vocabulary described them as inert, and tests/test_env_docs.py now holds
# every page that names a dtype to the tuples above.
os.environ["LAYA_CUDA_AMP"] = "float16"
check("cuda-amp/env float16 is the same ask as fp16", _cuda_amp_dtype("bf16"), torch.float16)
os.environ["LAYA_CUDA_AMP"] = "BFloat16"
check("cuda-amp/env bfloat16 is the same ask as bf16", _cuda_amp_dtype("fp16"), torch.bfloat16)
del os.environ["LAYA_CUDA_AMP"]

# CPU has its own vocabulary and it is the narrower one: bf16 only. No arm reached this comparison
# before -- the agents in this file hand-set `dtype` and `amp_enabled` through `_bare_agent`, so
# the value the documentation promises was never the value anything checked.
os.environ.pop("LAYA_CPU_AMP", None)
check("cpu-amp/unset leaves the forward fp32", _cpu_amp_dtype(), None)
os.environ["LAYA_CPU_AMP"] = "bf16"
check("cpu-amp/env bf16 opts in", _cpu_amp_dtype(), torch.bfloat16)
os.environ["LAYA_CPU_AMP"] = "BFloat16"
check("cpu-amp/env bfloat16 opts in", _cpu_amp_dtype(), torch.bfloat16)
os.environ["LAYA_CPU_AMP"] = "fp16"
check("cpu-amp/env fp16 is not offered on this device", _cpu_amp_dtype(), None)
os.environ["LAYA_CPU_AMP"] = "int8"
check("cpu-amp/env invalid leaves the forward fp32", _cpu_amp_dtype(), None)
del os.environ["LAYA_CPU_AMP"]


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


# ------------------------------------------------------------------ OOM fallback observability (#351)
class FakeCUDAInput:
    """An input_ids that fails the way a real CUDA OOM does when moved off CPU.

    Only `input_ids` needs this: it is the first tensor `_infer` moves, so the
    failure fires before the other (real, CPU-safe) tensors are touched, and on
    the CPU retry `.to('cpu')` passes it straight through.
    """

    shape = (1, 8)

    def __init__(self):
        self.moves = 0

    def to(self, device):
        self.moves += 1
        if str(device) != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return self


class ImmovableModel:
    """Accepts model.to() without moving anything (there is no real tensor to move)."""

    def to(self, device):
        self.placed = str(device)
        return self

    def __call__(self, *args):
        return torch.zeros((1, 2)), torch.zeros((1, 2))


oom_batch = dict(batch, input_ids=FakeCUDAInput())
oom = _bare_agent(ImmovableModel(), dtype=torch.float16)
oom.device = torch.device("cuda")   # the OOM branch only reads .type
check("oom-fallback/count starts at 0", oom.cpu_fallback_count, 0)
check("oom-fallback/reason starts None", oom.last_fallback_reason, None)

fallback_events = []
original_to = oom.model.to


def record_move(device):
    fallback_events.append("move-%s" % device)
    return original_to(device)


oom.model.to = record_move
with patch.object(torch, "clear_autocast_cache", side_effect=lambda: fallback_events.append("clear")):
    out = oom._infer(oom_batch)      # first forward raises OOM -> scoped CPU retry
check("oom-fallback/retry answered", isinstance(out, tuple), True)
check("oom-fallback/no batch scope leaves other caches alone", fallback_events[:1], ["move-cpu"])
check("oom-fallback/count recorded", oom.cpu_fallback_count, 1)
check("oom-fallback/reason recorded",
      "out of memory" in (oom.last_fallback_reason or ""), True)
check("oom-fallback/scoped: device restored", oom.device.type, "cuda")

fallback_events.clear()
token = _BATCH_AUTOCAST_CACHE.set(True)
try:
    with patch.object(torch, "clear_autocast_cache", side_effect=lambda: fallback_events.append("clear")):
        oom._infer(oom_batch)        # a second OOM inside the batch scope clears before CPU move
finally:
    _BATCH_AUTOCAST_CACHE.reset(token)
check("oom-fallback/batch copies cleared before CPU move", fallback_events[:2], ["clear", "move-cpu"])
check("oom-fallback/second OOM counts too", oom.cpu_fallback_count, 2)

# a plain forward never touches the counters
check("oom-fallback/plain CPU infer stays 0", cpu_disabled.cpu_fallback_count, 0)
check("oom-fallback/plain CPU reason stays None", cpu_disabled.last_fallback_reason, None)


# ------------------------------------------------------------------ example 35 must teach this policy
# examples/35 is the page a reader opens to ask "does my machine get mixed precision?", and on main
# it answered no twice: "CPU and MPS run **fp32**; `torch.autocast` has no MPS backend here, so Laya
# only wraps the forward pass on CUDA" in the banner, then, printed under the timings, "both legs
# used fp32 (device.type in ('cpu','mps') -> torch.float32); mixed precision would only be enabled
# on CUDA." The MPS branch of `Agent.__init__` sets `amp_enabled = True` at `torch.float16`, and
# `_amp_enabled_for` decides per call by row count. The second sentence was half true, though -- that
# 3-question call really did run fp32 -- so the gate does not ban the words. It reads the policy out
# of `Agent.__init__` by AST, so a rename or a reformat moves the gate and not the test, and it asks
# any sentence that pairs MPS with fp32 to name the key that decided it (`mps_amp_min_rows`, or the
# `dtype_for(rows)` that consults it). The page also had no more right to the number 5 than to the
# rule, so that is pinned to `MPS_AMP_MIN_ROWS_DEFAULT` rather than repeated here. This is the same
# shape as the example gates in tests/test_truncation.py: prose policed per sentence, contract read
# from the source.
import ast  # noqa: E402
import inspect  # noqa: E402
import re  # noqa: E402
import textwrap  # noqa: E402

import laya.agent as laya_agent  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX35 = "examples/35_device_selection_and_latency.py"
DEVICES = ("cuda", "mps", "xpu", "cpu")
BRANCH_TEST = re.compile(r"self\.device\.type == ['\"]([a-z]+)['\"]")


def _branch(device):
    """`Agent.__init__`'s `if/elif self.device.type == "<device>"` branch, statement by statement.

    Unparsed from the AST, which normalizes indentation, spacing and line breaks: a branch that is
    only reformatted still reads the same, so this cannot fail for a reason that is not a change of
    policy. That matters here because the branch is exactly the kind of line a reformat touches.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(Agent.__init__)))
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            match = BRANCH_TEST.search(ast.unparse(node.test))
            if match and match.group(1) == device:
                return "\n".join(ast.unparse(stmt) for stmt in node.body)
    return ""


def _strings(rel):
    """Every string the example carries: the module docstring, the banner, each printed line.

    UTF-8 explicitly, for the reason recorded at the `EX35` read below.
    """
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=rel)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    nodes.sort(key=lambda n: (n.lineno, n.col_offset))
    return [n.value for n in nodes]


# What the runtime promises, read rather than asserted.
BODIES = {device: _branch(device) for device in DEVICES}
check("amp-policy/every device branch is found",
      [d for d, body in BODIES.items() if not body], [])
ON = tuple(d for d in DEVICES if "self.amp_enabled = True" in BODIES[d])
check("amp-policy/mixed precision is not confined to CUDA", ON, DEVICES)
check("amp-policy/mps autocasts at float16", "self.dtype = torch.float16" in BODIES["mps"], True)
check("amp-policy/xpu autocasts at bfloat16", "self.dtype = torch.bfloat16" in BODIES["xpu"], True)
GATE_SRC = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(Agent._amp_enabled_for))))
check("amp-policy/mps is gated per call by row count",
      "rows < self.mps_amp_min_rows" in GATE_SRC, True)
READS = set(re.findall(r'"(LAYA_[A-Z0-9_]+)"', inspect.getsource(laya_agent)))
check("amp-policy/the runtime reads an env var per AMP knob",
      {"LAYA_MPS_AMP_MIN_ROWS", "LAYA_CUDA_AMP", "LAYA_CPU_AMP"} <= READS, True)

strings = _strings(EX35)
document = " ".join(strings)
sentences = [s for text in strings for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

MPS = re.compile(r"\bMPS\b", re.I)
FP32 = re.compile(r"\bfp32\b|\bfloat32\b", re.I)
GATE_KEY = re.compile(r"mps_amp_min_rows|dtype_for")
OTHER_DEVICE = re.compile(r"\b(MPS|XPU|CPU)\b", re.I)
ONLY_CUDA = re.compile(r"\bonly\b[^.]{0,60}\bcuda\b|\bcuda\b[^.]{0,40}\bonly\b", re.I)
NO_BACKEND = re.compile(r"(?:torch\.autocast|autocast) has no|has no (?:MPS|Apple)[^.]{0,24}backend"
                        r"|no MPS backend", re.I)


def _lie(sentence):
    """Why this sentence would mislead a reader about the autocast policy, or None if it does not."""
    if NO_BACKEND.search(sentence):
        return "asserts the runtime has no MPS autocast backend"
    if MPS.search(sentence) and FP32.search(sentence) and not GATE_KEY.search(sentence):
        return "says an MPS call runs fp32 without naming the key that decides it"
    if ONLY_CUDA.search(sentence) and not OTHER_DEVICE.search(sentence):
        return "confines mixed precision to CUDA, naming no other device that enables it"
    return None


flagged = ["%s  (%s)" % (_lie(s), s[:80]) for s in sentences if _lie(s)]
check("35/carries no sentence that misstates the autocast policy", flagged, [])

# The two sentences the gate exists for, kept verbatim so the rules above are shown to bite: a rule
# that matches nothing would let the page go back to being wrong without anyone noticing.
for historical in (
        "Precision follows the device -- mixed precision (bf16/fp16) is a CUDA win, while CPU and MPS"
        " run **fp32**; `torch.autocast` has no MPS backend here, so Laya only wraps the forward pass"
        " on CUDA.",
        "both legs used fp32 (device.type in ('cpu','mps') -> torch.float32); mixed precision would"
        " only be enabled on CUDA."):
    check("35/the rule still catches %r" % historical[:44], _lie(historical) is not None, True)

# The mechanism has to be taught by name, and the number by source rather than by hand.
for name in ("amp_enabled", "dtype_for", "`mps_amp_min_rows`", "LAYA_MPS_AMP_MIN_ROWS"):
    check("35/names %s so a reader can grep for it" % name, name in document, True)
MPS_TARGET = re.search(r"self\.dtype = (?:torch\.(\w+)|(\w+\(\)))", BODIES["mps"])
MPS_TARGET = MPS_TARGET.group(1) or MPS_TARGET.group(2) if MPS_TARGET else ""
# The window stops at the next device's name: a page may satisfy "MPS ... <dtype>" by going on to
# describe XPU, and that is not the same claim.
MPS_CLAIM = r"\bMPS\b(?:(?!\bXPU\b|\bCPU\b|\bCUDA\b)[^.]){0,200}\b%s\b" % re.escape(MPS_TARGET)
check("35/states MPS's target as the branch sets it (%s)" % MPS_TARGET,
      re.search(MPS_CLAIM, document, re.I) is not None, True)
check("35/the MPS threshold it states is the module default",
      [int(n) for n in re.findall(r"(\d+)\s+rows", document)], [MPS_AMP_MIN_ROWS_DEFAULT])

# Read from the agent, and observed in a real forward: the page may not hand-copy the threshold, nor
# relabel `dtype` as though it were the precision a call ran in. That relabeling is how the old
# sentence got printed with a straight face -- `dtype` said float16 while the call ran fp32.
# `encoding="utf-8"` because `open()` otherwise takes the runner's locale codec, which is cp1252 on
# `tests (windows)` and cannot decode the UTF-8 these pages carry (tests/test_portability.py gates it).
with open(os.path.join(ROOT, EX35), encoding="utf-8") as fh:
    example = ast.parse(fh.read(), filename=EX35)
attrs = {n.attr for n in ast.walk(example) if isinstance(n, ast.Attribute)}
calls = {n.func.attr for n in ast.walk(example)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
# Read twice, not once: the page has to report the agent's threshold in the per-leg line and drive
# the arms from it. Assigning to the attribute would not count, which is why this asks for `Load`.
threshold_reads = [n for n in ast.walk(example) if isinstance(n, ast.Attribute)
                   and n.attr == "mps_amp_min_rows" and isinstance(n.ctx, ast.Load)]
check("35/reads the agent's threshold rather than hand-copying it", len(threshold_reads) >= 2, True)
check("35/asks the agent `dtype_for(rows)`", "dtype_for" in calls, True)
check("35/observes the precision inside a real forward",
      sorted({"register_forward_hook", "is_autocast_enabled"} - calls), [])

# Every LAYA_* the page presents as a knob has to be one the runtime reads. `LAYA_DEVICE` is the
# exception the page itself flags, so it is allowed only in a sentence saying Laya does not read it.
invented = sorted(name for name in set(re.findall(r"\bLAYA_[A-Z0-9_]+\b", document))
                  if name not in READS and not any(name in s and re.search(r"does not read|not read", s)
                                                   for s in sentences))
check("35/names no env var the runtime does not read, or says so", invented, [])


# ------------------------------------------------------------------ report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all runtime-fix tests passed")
sys.exit(1 if FAIL else 0)
