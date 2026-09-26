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

`use_duck_shape` is a process-global torch setting read when a graph is traced, which happens lazily
inside a call, so it cannot be set once at load and restored. It is set for the duration of each
compiled forward instead and put back to whatever it was when the last one returns (refcounted, so
concurrent calls on several threads do not restore it early). Nothing is left changed between calls.

Mode and device are unchanged from what `compile=True` has always done: the default inductor mode,
on whatever device the agent runs on, CPU included.
"""
import threading
from contextlib import contextmanager

import torch

_lock = threading.Lock()
_depth = 0
_saved = None


def compile_model(model, **kwargs):
    """`torch.compile(model, dynamic=True)`; `kwargs` go to `torch.compile` (tests pass `backend=`)."""
    return torch.compile(model, dynamic=True, **kwargs)


def _fx_config():
    try:
        from torch.fx.experimental import _config
    except ImportError:  # a torch without symbolic shapes has nothing to switch
        return None
    return _config if hasattr(_config, "use_duck_shape") else None


@contextmanager
def independent_dims():
    """Trace with `use_duck_shape = False` inside the block, restoring the prior value afterwards."""
    global _depth, _saved
    cfg = _fx_config()
    if cfg is None:
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
