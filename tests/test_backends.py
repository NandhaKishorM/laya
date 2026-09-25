"""The inference backend layer (`laya.backends`), checked without a GPU or a checkpoint.

What is pinned here is the contract every backend is held to and the parts the Agent relies on:
the names and aliases `load(backend=...)` accepts, the `auto` policy off CUDA, the shared padding
policy both graph backends use, and that a backend which cannot run here falls back to eager with
exactly one warning (or raises under `strict=True`) and leaves the model's forward untouched.

Run: python tests/test_backends.py   (or under pytest)
"""
import os
import sys
import warnings
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from laya import backends  # noqa: E402
from laya.backends import BackendUnavailable, bucket_shape, pad_batch  # noqa: E402
from laya.backends.base import Backend, choose_dtype  # noqa: E402
from laya.backends.eager import EagerBackend  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name if got == want else "%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s: %s" % (name, detail))


class TinyModel(torch.nn.Module):
    """A forward with the DecisionModel signature, so a backend can be installed on it."""

    def __init__(self):
        super().__init__()
        self.encoder = SimpleNamespace(config=SimpleNamespace(model_type="bert", reference_compile=None))
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        n, k = marker_mask.shape
        return torch.zeros(n, k) + attention_mask.sum(1, keepdim=True).float(), torch.zeros(n, 2)


def fake_agent(device="cpu", dtype=torch.float32):
    return SimpleNamespace(model=TinyModel(), device=torch.device(device), dtype=dtype, amp_enabled=False,
                           cfg={"max_len": 64}, tok=SimpleNamespace(pad_token_id=0, cls_token_id=1), _backend=None)


# ------------------------------------------------------------------ names and aliases
for raw, want in (("eager", "eager"), ("Eager", "eager"), ("fast", "tilelang"), ("TileLang", "tilelang"),
                  ("torch.compile", "compile"), ("inductor", "compile"), ("auto", "auto"), ("onnx", "onnx"), (None, "eager")):
    check("normalise(%r)" % (raw,), backends.normalise(raw), want)
try:
    backends.normalise("cuda-graphs")
    FAIL.append("normalise/unknown name accepted")
except ValueError as e:
    check_true("normalise/unknown name names the choices", "eager" in str(e) and "tilelang" in str(e), str(e))

# ------------------------------------------------------------------ auto policy off CUDA
check("auto_policy/cpu -> eager", backends.auto_policy(fake_agent("cpu")), "eager")
check("make_backend(auto)/cpu is eager", backends.make_backend("auto", fake_agent("cpu")).name, "eager")

# ------------------------------------------------------------------ dtype policy
check("choose_dtype/cpu default", choose_dtype(torch.device("cpu"), {}), (torch.float32, False))
os.environ["LAYA_CPU_AMP"] = "bf16"
check("choose_dtype/cpu LAYA_CPU_AMP=bf16", choose_dtype(torch.device("cpu"), {}), (torch.bfloat16, True))
del os.environ["LAYA_CPU_AMP"]
if torch.cuda.is_available():
    dt, amp = choose_dtype(torch.device("cuda"), {"amp_dtype": "bf16"})
    check_true("choose_dtype/cuda is 16-bit autocast", amp and dt in (torch.bfloat16, torch.float16), (dt, amp))

# ------------------------------------------------------------------ padding policy
for (n, L, max_len), want in (((1, 5, 512), (1, 16)), ((3, 70, 512), (4, 80)), ((5, 256, 512), (8, 256)),
                              ((2, 257, 512), (2, 320)), ((1, 1000, 512), (1, 512)), ((16, 16, 512), (16, 16))):
    check("bucket_shape(%d, %d, %d)" % (n, L, max_len), bucket_shape(n, L, max_len), want)

ids = torch.tensor([[7, 8, 9], [7, 8, 0]])
att = torch.tensor([[1, 1, 1], [1, 1, 0]])
mpos = torch.tensor([[1, 2], [1, 0]])
mmask = torch.tensor([[True, True], [True, False]])
qt = torch.tensor([0, 2])
p_ids, p_att, p_mpos, p_mmask, p_qt, N, L0 = pad_batch(ids, att, mpos, mmask, qt, max_len=64, pad_id=5)
check("pad_batch/shape", tuple(p_ids.shape), (2, 16))
check("pad_batch/(N, L0)", (N, L0), (2, 3))
check_true("pad_batch/request kept", torch.equal(p_ids[:2, :3], ids) and torch.equal(p_att[:2, :3], att)
           and torch.equal(p_qt[:2], qt) and torch.equal(p_mmask[:2], mmask))
check("pad_batch/pad id fills", int(p_ids[0, 3]), 5)
check("pad_batch/padding tokens masked", int(p_att[:, 3:].sum()), 0)
p3 = pad_batch(ids[:1], att[:1], mpos[:1], mmask[:1], qt[:1], max_len=64)
check("pad_batch/one row stays one row", tuple(p3[0].shape), (1, 16))
p5 = pad_batch(ids.repeat(3, 1)[:5], att.repeat(3, 1)[:5], mpos.repeat(3, 1)[:5], mmask.repeat(3, 1)[:5], qt.repeat(3)[:5], 64,
               keep_one_token=True)
check("pad_batch/five rows pad to eight", tuple(p5[0].shape), (8, 16))
check("pad_batch/padding rows have empty masks", int(p5[1][5:].sum()), 3)      # keep_one_token: token 0 valid
check("pad_batch/padding rows keep one marker", int(p5[3][5:].sum()), 3)
check("pad_batch/padding rows have qtype 0", int(p5[4][5:].sum()), 0)

# ------------------------------------------------------------------ install / uninstall / fallback
agent = fake_agent()


def untouched(model):
    """The class forward is in use and no instance attribute shadows it."""
    return "forward" not in model.__dict__ and model.forward.__func__ is TinyModel.forward


be = backends.install(agent, "eager")
check("install(eager)/name", be.name, "eager")
check_true("install(eager)/forward untouched", untouched(agent.model))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    be = backends.install(agent, "compile")               # CPU: compile declines
check("install(compile)/cpu falls back to eager", be.name, "eager")
rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
check("install(compile)/cpu warns exactly once", len(rw), 1)
check_true("install(compile)/warning names the backend and the fallback",
           rw and "'compile'" in str(rw[0].message) and "eager" in str(rw[0].message), [str(x.message) for x in rw])
check_true("install(compile)/forward untouched after fallback", untouched(agent.model))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    be = backends.install(agent, "tilelang")              # CPU and/or no tilelang: declines
check("install(tilelang)/cpu falls back to eager", be.name, "eager")
check("install(tilelang)/cpu warns exactly once", len([x for x in w if issubclass(x.category, RuntimeWarning)]), 1)

try:
    backends.install(agent, "compile", strict=True)
    FAIL.append("install(compile, strict)/cpu did not raise")
except BackendUnavailable:
    PASS.append("install(compile, strict)/cpu raises BackendUnavailable")
check_true("install(strict)/forward untouched after raise", untouched(agent.model))


class Doubling(Backend):
    """A backend that runs the stock forward and doubles the logits: proves install swaps the forward."""
    name = "doubling"

    def forward(self, *args, **kw):
        logits, act = self.stock_forward(*args, **kw)
        return logits * 2, act


b = {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "qtype": qt}
ref = agent.model(**b)[0]
d = Doubling(agent)
d.install()
check_true("custom backend/installed forward is used", torch.equal(agent.model(**b)[0], ref * 2))
check_true("custom backend/stock_forward still reaches the model", torch.equal(d.stock_forward(**b)[0], ref))
d.uninstall()
check_true("custom backend/uninstall restores the forward", untouched(agent.model) and torch.equal(agent.model(**b)[0], ref))
d.install(); d.install()
check_true("custom backend/install is idempotent", torch.equal(agent.model(**b)[0], ref * 2))
d.uninstall(); d.uninstall()
check_true("custom backend/uninstall is idempotent", untouched(agent.model))

e = EagerBackend(agent)
check("eager/max_len is unbounded", e.max_len, None)
check("eager/warmup is free", e.warmup(), 0.0)

# ------------------------------------------------------------------ the Agent surface
import inspect  # noqa: E402
from laya.agent import Agent, load  # noqa: E402

check("Agent.__init__/backend default None", inspect.signature(Agent.__init__).parameters["backend"].default, None)
check("load/backend default None", inspect.signature(load).parameters["backend"].default, None)
check("Agent/backend property default", Agent.__new__(Agent).backend, "eager")
check_true("Agent/set_backend, accelerate, deaccelerate exist",
           all(callable(getattr(Agent, n, None)) for n in ("set_backend", "accelerate", "deaccelerate")))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)


def test_backends():
    assert not FAIL, FAIL
