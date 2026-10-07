"""The `compile=True` path: `torch.compile` over the whole decision model, without per-shape recompiles.

Laya's forward sees a new (rows, tokens, markers) shape on almost every call. Under
`torch.compile(model)` with its defaults that costs, on an RTX 4070 (measured in #472):

* one recompile at the second distinct shape, because the first compile is static and only the
  recompile switches to dynamic shapes ("automatic dynamic"): ~18 s;
* one more for every request that breaks a "duck sizing" guard. Dynamo gives two dimensions
  that happen to be equal at trace time one symbol and then guards on them staying equal, so a
  first call with four questions of four options (rows == markers == 4) recompiles, ~25 s, as
  soon as they differ.

`compile_model` compiles with `dynamic=True`, which removes the first, and `independent_dims()`
switches duck sizing off while a compiled forward runs, which removes the second: every dimension
of Laya's inputs is independent, so each gets its own symbol. What is left is torch's own 0/1
specialisation, one extra graph the first time a single-row batch arrives.

`use_duck_shape` is read when a graph is traced, which happens lazily inside a call, so it cannot be
set once at load and restored. It is set for the duration of each compiled forward instead. In older
torch (2.11, for one) the setting is process-global, so it is put back to whatever it was when the
last call returns (refcounted, so concurrent calls on several threads do not restore it early); newer
torch (2.14) keeps config overrides per thread (a `ContextVar` per entry), so each call sets and
restores its own thread's value with the config's `patch`. Nothing is left changed between calls.

The default Inductor mode is unchanged. `compile_mode="reduce-overhead"` opts into CUDA graphs;
Laya serializes those forwards and copies their outputs before a later replay can overwrite them.
Compilation still runs on whatever device the agent uses, CPU included.
"""
import threading
import os
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_lock = threading.Lock()
_depth = 0
_saved = None
_cudagraph_lock = threading.Lock()


def compile_model(model, **kwargs):
    """`torch.compile(model, dynamic=True)`; `kwargs` go to `torch.compile` (tests pass `backend=`)."""
    return torch.compile(model, dynamic=True, **kwargs)


def configure_cache():
    """Respect an existing Inductor directory; otherwise use the user's Laya cache."""
    root = os.environ.get("XDG_CACHE_HOME", "")
    if not os.path.isabs(root):
        root = os.path.expanduser("~/.cache")
    directory = os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(root, "laya", "torchinductor"))
    os.makedirs(directory, exist_ok=True)
    return directory


@contextmanager
def cuda_graph_step():
    """Serialize Laya CUDA graph forwards until their outputs have been copied."""
    mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if mark_step is None:
        raise RuntimeError("reduce-overhead on CUDA requires torch.compiler.cudagraph_mark_step_begin")
    with _cudagraph_lock:
        mark_step()
        yield


def _fx_config():
    try:
        from torch.fx.experimental import _config
    except ImportError:  # a torch without symbolic shapes has nothing to switch
        return None
    return _config if hasattr(_config, "use_duck_shape") else None


def _per_thread(cfg):
    """True where a config override only applies to the thread that set it (torch 2.14; 2.11 is global)."""
    entry = getattr(cfg, "_config", {}).get("use_duck_shape")
    return isinstance(getattr(entry, "user_override", None), ContextVar)


@contextmanager
def independent_dims():
    """Trace with `use_duck_shape = False` inside the block, restoring the prior value afterwards."""
    global _depth, _saved
    cfg = _fx_config()
    if cfg is None:
        yield
        return
    if _per_thread(cfg):
        # each thread traces with its own settings, so each call sets and restores its own
        with cfg.patch(use_duck_shape=False):
            yield
        return
    with _lock:
        if _depth == 0:
            _saved = cfg.use_duck_shape
            cfg.use_duck_shape = False
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                cfg.use_duck_shape = _saved
                _saved = None
