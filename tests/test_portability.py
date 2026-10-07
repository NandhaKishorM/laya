"""Portability regressions: checkpoints saved by transformers 5, and non-CUDA devices.

Three failures found while getting Laya to run on a machine that is not the one it was written
on, all silent or fatal depending on the platform:

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

3. `open(path)` and `Path.read_text()` with no `encoding=` take `locale.getpreferredencoding()`,
   which is cp1252 on the `tests (windows)` runner. The pages the suites police -- README.md, the
   docs, the examples -- are UTF-8 and carry characters cp1252 has no mapping for, so a test that
   reads one of them dies at the read with `UnicodeDecodeError: 'charmap' codec can't decode byte
   0x81` instead of failing one assertion. It has cost the runner twice: once at a metadata gate
   that read README.md, and once before that at the batch-shape gate, whose fix is recorded in
   tests/test_mcp.py's comment at its README read. Section 4 below gates the convention.
"""
import ast
import glob
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


# A disabled context must never enter torch.autocast: on a torch build with no MPS autocast
# backend, entering it raises. The gate (Agent._amp_enabled_for) decides `enabled`.
for name in ("cpu", "mps", "xpu"):
    entered = []
    with mock.patch.object(_agent.torch, "autocast", _autocast_that_rejects_non_cuda(entered)):
        try:
            ctx = _agent._amp_context(SimpleNamespace(type=name), torch.float32, False)
            with ctx:
                pass
            check_true("amp/%s never enters autocast when disabled" % name, entered == [],
                       "entered %s" % entered)
        except RuntimeError as e:
            FAIL.append("amp/%s raised %s" % (name, e))

check_true("amp/disabled is a no-op",
           isinstance(_agent._amp_context(SimpleNamespace(type="mps"), torch.float16, False), nullcontext))

# CUDA still gets mixed precision, with the dtype the model was configured for
entered = []
with mock.patch.object(_agent.torch, "autocast", _autocast_that_rejects_non_cuda(entered)):
    with _agent._amp_context(SimpleNamespace(type="cuda"), torch.float16, True):
        pass
check("amp/cuda uses autocast", entered, ["cuda"])

# the old unconditional call is what broke MPS -- make sure it cannot come back
import inspect  # noqa: E402

_src = inspect.getsource(_agent.Agent._infer)
check_true("amp/infer goes through _amp_context", "with _amp_context(self.device, self.dtype, enabled)" in _src)
check_true("amp/no unconditional autocast in the forward pass",
           "torch.autocast(device_type=self.device.type" not in _src)


# ------------------------------------------------ 3. a failed GPU placement falls back to CPU
# `system_one` promises to survive a device that cannot hold the work: on a memory error it moves
# to CPU and re-runs. That path is easy to break and never fires on a machine with no GPU, so it
# is driven here with a stand-in model that fails the first forward pass the way CUDA does.
class _FakeTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(c) % 40) for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _FailsOnce(torch.nn.Module):
    """Raises a CUDA-style OOM on the first call, then answers."""

    def __init__(self, message):
        super().__init__()
        self.message, self.calls = message, 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(self.message)
        logits = torch.zeros((input_ids.shape[0], marker_mask.shape[1]))
        logits[:, 0] = 1.0
        return logits, torch.tensor([[1.0, 0.0]])


class _RestoreFails(_FailsOnce):
    """OOMs once like _FailsOnce, but also refuses to move back to the accelerator afterwards,
    so `_restore_runtime`'s `model.to(device)` fails and the agent must stay on CPU."""
    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        dt = getattr(target, "type", None) or (target if isinstance(target, str) else None)
        if dt == "mps":
            raise RuntimeError("CUDA out of memory: model no longer fits after the retry")
        return super().to(*args, **kwargs)


def _bare_agent(model):
    agent = _agent.Agent.__new__(_agent.Agent)     # no weights: exercise system_one only
    agent.device = torch.device("mps")             # any non-CPU device enters the fallback branch
    agent.dtype = torch.float32
    agent.cfg = {"max_len": 64, "head_max_len": 32}
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.tok = _FakeTok()
    agent.model = model
    return agent


QUESTIONS = {"q": {"type": "choice", "instructions": "Pick one",
                   "criteria": {"a": "first option", "b": "second option"}}}

# Placement on the stand-in device is faked so the scenario also runs where torch has no
# such backend compiled in (the CPU-only wheels CI installs): batch `.to(device)` calls
# targeting the fake device become no-ops and the stand-in model supplies the failure,
# exactly as the real device would.
_real_tensor_to = torch.Tensor.to


def _to_that_ignores_mps(self, *args, **kwargs):
    target = args[0] if args else kwargs.get("device")
    if (isinstance(target, torch.device) and target.type == "mps") or target == "mps":
        return self
    return _real_tensor_to(self, *args, **kwargs)


with mock.patch.object(torch.Tensor, "to", _to_that_ignores_mps):
    agent = _bare_agent(_FailsOnce("CUDA out of memory. Tried to allocate 2.00 GiB"))
    try:
        result = agent.predict({"body": "some state"}, QUESTIONS)
        check("fallback/answers after a memory failure", result["answers"]["q"]["choice"], "a")
        check("fallback/forward pass retried once", agent.model.calls, 2)
        # the demotion is scoped to the failed request (#344): one oversized call must not
        # leave every later call on a ~10-15x slower CPU path for the life of the process
        check("fallback/device restored after the retry", agent.device.type, "mps")
        check("fallback/a later request is answered without a new demotion",
              agent.predict({"body": "another state"}, QUESTIONS)["answers"]["q"]["choice"], "a")
        check("fallback/later request needs no extra forward", agent.model.calls, 3)
        check("fallback/device still the original after a later call", agent.device.type, "mps")
    except Exception as e:  # noqa: BLE001
        FAIL.append("fallback/memory failure was not survived: %s: %s" % (type(e).__name__, e))

    # a non-memory RuntimeError must still propagate: silently swallowing real bugs is worse
    # than the crash it would hide
    agent = _bare_agent(_FailsOnce("shape mismatch in attention"))
    try:
        agent.predict({"body": "some state"}, QUESTIONS)
        FAIL.append("fallback/non-memory error propagates (nothing raised)")
    except RuntimeError:
        PASS.append("fallback/non-memory error propagates")
    except Exception as e:  # noqa: BLE001
        FAIL.append("fallback/non-memory error raised %s instead of RuntimeError" % type(e).__name__)

    # a RuntimeError that merely mentions cuda is not an OOM (#344): it used to demote the
    # agent to CPU permanently and return silently-CPU results
    agent = _bare_agent(_FailsOnce("CUDA error: device-side assert triggered"))
    try:
        agent.predict({"body": "some state"}, QUESTIONS)
        FAIL.append("fallback/cuda-worded non-memory error propagates (nothing raised)")
    except RuntimeError:
        PASS.append("fallback/cuda-worded non-memory error propagates")
    except Exception as e:  # noqa: BLE001
        FAIL.append("fallback/cuda-worded non-memory error raised %s instead of RuntimeError"
                    % type(e).__name__)

    # restore fails: the model no longer fits the accelerator after the CPU retry, so it stays
    # demoted rather than crashing a request that already succeeded
    agent = _bare_agent(_RestoreFails("CUDA out of memory. Tried to allocate 2.00 GiB"))
    try:
        result = agent.predict({"body": "some state"}, QUESTIONS)
        check("fallback/restore-failure still answers the request", result["answers"]["q"]["choice"], "a")
        check("fallback/restore-failure stays on cpu", agent.device.type, "cpu")
        check("fallback/restore-failure: later call runs on cpu without crashing",
              agent.predict({"body": "another state"}, QUESTIONS)["answers"]["q"]["choice"], "a")
        check("fallback/restore-failure: no extra forward beyond retry + later call", agent.model.calls, 3)
    except Exception as e:  # noqa: BLE001
        FAIL.append("fallback/restore-failure not survived: %s: %s" % (type(e).__name__, e))

    # one unsupported autocast op retries that request and leaves AMP on. Three misses in a
    # row disable it, so a build without the op does not pay for two forwards forever (#351).
    # CPU so the MPS row gate does not hide the branch.
    class _AutocastMisses(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            self.calls += 1
            if self.calls in (1, 3, 5):
                raise RuntimeError("User specified an unsupported autocast device_type cpu")
            logits = torch.zeros((input_ids.shape[0], marker_mask.shape[1]))
            logits[:, 0] = 1.0
            return logits, torch.tensor([[1.0, 0.0]])

    agent = _bare_agent(_AutocastMisses())
    agent.device = torch.device("cpu")
    agent.amp_enabled = True
    agent.dtype = torch.float16
    try:
        result = agent.predict({"body": "some state"}, QUESTIONS)
        check("autocast/answers after one miss", result["answers"]["q"]["choice"], "a")
        check("autocast/one miss retries once", agent.model.calls, 2)
        check("autocast/one miss keeps amp", agent.amp_enabled, True)
        check("autocast/one miss keeps dtype", agent.dtype, torch.float16)
        agent.predict({"body": "another state"}, QUESTIONS)
        check("autocast/two misses keep amp", agent.amp_enabled, True)
        agent.predict({"body": "third state"}, QUESTIONS)
        check("autocast/third miss disables amp", agent.amp_enabled, False)
        check("autocast/third miss drops dtype", agent.dtype, torch.float32)
        check("autocast/three misses are six forwards", agent.model.calls, 6)
        agent.predict({"body": "later state"}, QUESTIONS)
        check("autocast/later request is one forward", agent.model.calls, 7)
    except Exception as e:  # noqa: BLE001
        FAIL.append("autocast/streak was not survived: %s: %s" % (type(e).__name__, e))


# ------------------------------------------------ 4. a repo read must pin its encoding, not take the
# locale's. See the module docstring: on `tests (windows)` the locale codec is cp1252, so reading a
# UTF-8 page raises at the read and the whole suite exits 1 rather than failing one assertion.
TEXT_SUFFIXES = (".md", ".py", ".nix", ".txt", ".yml", ".yaml", ".json", ".toml")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_reads(src):
    """The reads of a repo file `src` carries, as `(lineno, call name, path expression, pinned)`.

    Writes are out of scope: this is about decoding, which only a read does. A read is in scope when
    its path expression names a file that exists in the repo, or builds from the module's own repo
    root (`ROOT`, `parents[`, `dirname(`, `__file__`). The second arm is deliberately blunt --
    `open(os.path.join(ROOT, rel))` is how both crashes were written, and `rel` cannot be resolved
    without running the module, so a read of that shape is asked to pin its encoding rather than
    proven dangerous.
    """
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
        if name == "open" and node.args:
            target = node.args[0]
        elif name == "read_text" and isinstance(fn, ast.Attribute):
            # `Path.read_text()` carries its path on the object, not in `args`.
            target = fn.value
        else:
            continue
        modes = [ast.unparse(k.value) for k in node.keywords if k.arg == "mode"]
        if name == "open" and len(node.args) > 1:
            modes.append(ast.unparse(node.args[1]))
        if any("w" in m or "a" in m or "b" in m for m in modes):
            continue
        arg = ast.unparse(target)
        literals = [n.value for n in ast.walk(target)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        names_repo_file = any(l.endswith(TEXT_SUFFIXES) and os.path.exists(os.path.join(REPO, l))
                              for l in literals)
        from_root = any(tok in arg for tok in ("ROOT", "parents[", "dirname(", "__file__"))
        if names_repo_file or from_root:
            out.append((node.lineno, name, arg, any(k.arg == "encoding" for k in node.keywords)))
    return out


reads_by_suite = {}
for _path in sorted(glob.glob(os.path.join(REPO, "tests", "*.py"))):
    with open(_path, encoding="utf-8") as _fh:
        reads_by_suite["tests/" + os.path.basename(_path)] = _repo_reads(_fh.read())

unpinned = ["%s:%d %s" % (_rel, _lineno, _arg)
            for _rel, _reads in sorted(reads_by_suite.items())
            for _lineno, _name, _arg, _pinned in _reads if not _pinned]
check("encoding/no repo read in tests/ leaves the codec to the locale", unpinned, [])

# The rule must not pass by finding nothing: it has to see the reads that already pin the encoding.
pinned_seen = sum(1 for _reads in reads_by_suite.values() for _, _, _, _p in _reads if _p)
check_true("encoding/the scan sees the reads that already pin it", pinned_seen >= 10,
           "only %d pinned repo reads found, so the scan is not reaching the suites" % pinned_seen)

# And it has to see both call shapes. The first version of this scanner read `node.args`, which a
# `Path.read_text()` does not use -- it carries its path on the object -- so it reported a clean repo
# while tests/test_evals.py still decoded docs/evals.md with the locale codec. A scan that reaches one
# shape only passes the half of the convention it can still see.
shapes_seen = {_name for _reads in reads_by_suite.values() for _, _name, _, _ in _reads}
check_true("encoding/the scan sees both open() and Path.read_text()",
           shapes_seen >= {"open", "read_text"},
           "the scan reached %s only" % sorted(shapes_seen))

# And the premise has to hold: pages the suites read really are UTF-8 that cp1252 cannot decode.
# Without one such page the rule is a style preference, not a crash guard.
_repo_text = []
for _pattern in ("README.md", os.path.join("docs", "**", "*.md"),
                 os.path.join("examples", "*.py"), os.path.join("laya", "*.py")):
    _repo_text.extend(glob.glob(os.path.join(REPO, _pattern), recursive=True))
_undecodable = []
for _page in _repo_text:
    try:
        with open(_page, encoding="cp1252") as _fh:
            _fh.read()
    except UnicodeDecodeError:
        _undecodable.append(os.path.relpath(_page, REPO).replace(os.sep, "/"))
check_true("encoding/some repo page is UTF-8 the Windows locale cannot decode", bool(_undecodable),
           "every repo page decodes as cp1252, so the crash this gate guards has nothing to crash on")
print("encoding: %d repo pages cp1252 cannot decode, %d repo reads in tests/ all pinned"
      % (len(_undecodable), pinned_seen))


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
