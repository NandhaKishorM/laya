"""Portability regressions: checkpoints saved by transformers 5, and non-CUDA devices.

Two failures found while getting Laya to run on a macOS/Intel box, both silent or fatal depending
on the machine:

1. transformers 5 records ModernBERT's RoPE bases as
   `rope_parameters = {"full_attention": {...}, "sliding_attention": {...}}`. transformers 4.x
   (the only line that runs on macOS Intel, where PyTorch stops at 2.2.2) does not read that key
   and falls back to its own defaults -- global 160000, local 10000. English and typed-decisions
   happen to match those defaults; mmBERT does not (both of its bases are 160000), so it ran with
   the wrong RoPE base instead of failing loudly.

2. `torch.autocast(device_type=self.device.type, enabled=False)` raises on devices torch has no
   autocast backend for, even when disabled: 'User specified an unsupported autocast device_type
   mps'. Laya only ever enables autocast on CUDA, but it still entered the context on every call,
   so `predict()` died outright on the MPS device torch selects on Apple/AMD machines.
"""
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from laya import agent as _agent  # noqa: E402
from laya.common import _apply_rope_config  # noqa: E402

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


# ------------------------------------------------ 1. transformers-5 RoPE config -> transformers 4.x
def cfg_with(rope_parameters, **kwargs):
    """A stand-in for a ModernBertConfig carrying both the 5.x and the 4.x view of RoPE."""
    base = dict(global_rope_theta=160000.0, local_rope_theta=10000.0)   # 4.x defaults
    base.update(kwargs)
    return SimpleNamespace(rope_parameters=rope_parameters, **base)


# the mmBERT case: both bases are 160000, so the 4.x default local theta (10000) is wrong
mmbert = cfg_with({"full_attention": {"rope_theta": 160000, "rope_type": "default"},
                   "sliding_attention": {"rope_theta": 160000, "rope_type": "default"}})
_apply_rope_config(mmbert)
check("rope/mmbert global theta", mmbert.global_rope_theta, 160000.0)
check("rope/mmbert local theta corrected", mmbert.local_rope_theta, 160000.0)

# ModernBERT-large / typed-decisions: 160000 / 10000 -- already the 4.x defaults, must not move
modernbert = cfg_with({"full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                       "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})
_apply_rope_config(modernbert)
check("rope/modernbert global theta unchanged", modernbert.global_rope_theta, 160000.0)
check("rope/modernbert local theta unchanged", modernbert.local_rope_theta, 10000.0)

# a config that carries no rope_parameters at all (every 4.x-era checkpoint) must be left alone
native = cfg_with(None, global_rope_theta=1234.0, local_rope_theta=5678.0)
_apply_rope_config(native)
check("rope/absent rope_parameters is a no-op", (native.global_rope_theta, native.local_rope_theta),
      (1234.0, 5678.0))

# transformers 5 keeps the thetas only inside rope_parameters; the old attributes are gone, so
# there is nothing to write and the call must not raise
v5_only = SimpleNamespace(rope_parameters={"full_attention": {"rope_theta": 160000},
                                           "sliding_attention": {"rope_theta": 160000}})
try:
    _apply_rope_config(v5_only)
    check_true("rope/transformers-5 config is a no-op", not hasattr(v5_only, "global_rope_theta"))
except Exception as e:
    FAIL.append("rope/transformers-5 config raised: %r" % e)

# a flat rope_parameters mapping is also accepted
flat = cfg_with({"rope_theta": 10000.0, "rope_type": "default"})
_apply_rope_config(flat)
check("rope/flat rope_parameters", (flat.global_rope_theta, flat.local_rope_theta), (10000.0, 10000.0))


# ------------------------------------------------ 2. autocast is never entered off CUDA
def _autocast_that_rejects_non_cuda(entered):
    def fake(*args, **kwargs):
        entered.append(kwargs.get("device_type"))
        device_type = kwargs.get("device_type")
        if device_type != "cuda":
            raise RuntimeError("User specified an unsupported autocast device_type %r" % device_type)
        return nullcontext()
    return fake


for name in ("cpu", "mps", "xpu"):
    entered = []
    with mock.patch.object(_agent.torch, "autocast", _autocast_that_rejects_non_cuda(entered)):
        try:
            ctx = _agent._amp_context(SimpleNamespace(type=name), torch.float32)
            with ctx:
                pass
            check_true("amp/%s never enters autocast" % name, entered == [], "entered %s" % entered)
        except RuntimeError as e:
            FAIL.append("amp/%s raised %s" % (name, e))

# CUDA still gets mixed precision, with the dtype the model was configured for
entered = []
with mock.patch.object(_agent.torch, "autocast", _autocast_that_rejects_non_cuda(entered)):
    with _agent._amp_context(SimpleNamespace(type="cuda"), torch.float16):
        pass
check("amp/cuda uses autocast", entered, ["cuda"])

# the old unconditional call is what broke MPS -- make sure it cannot come back
import inspect  # noqa: E402

_src = inspect.getsource(_agent.Agent.system_one)
check_true("amp/forward pass goes through _amp_context", "with _amp_context(self.device, self.dtype):" in _src)
check_true("amp/no unconditional autocast in the forward pass",
           "torch.autocast(device_type=self.device.type" not in _src)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
