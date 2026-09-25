"""The `torch.compile` backend: inductor kernels + CUDA graphs, no extra dependency.

    agent = laya.load("convaiinnovations/laya", backend="compile")     # compile=True is the alias

`torch.compile(model.forward, dynamic=True, mode="reduce-overhead")`: one compile serves every
shape (no recompile per sequence length, which `dynamic=False` pays ~18 s for), and CUDA graphs
remove the ~200 Python-side kernel launches a small request is bound by. Requests are padded to
the shared bucket policy (`laya.backends.base.bucket_shape`) so the CUDA-graph tree records a
bounded set of shapes, and the outputs are copied out of the graph's static buffers before the
lock is released.

Cold start is the cost: the first compile takes tens of seconds. Two things cut it:

* the inductor FX-graph cache, persisted in a stable directory. Set `LAYA_INDUCTOR_CACHE_DIR`
  (default `~/.cache/laya/inductor`); it is applied to `TORCHINDUCTOR_CACHE_DIR` unless that is
  already set, so a `TORCHINDUCTOR_*` variable the operator exports still wins. A second process
  loads the compiled graphs from there instead of compiling them again.
* warm-up at load (`warmup=True`, or `LAYA_COMPILE_WARMUP=0` to skip): a few representative
  shapes are run at load time so the first request pays nothing. `agent.backend_object.warmup_s`
  reports what it cost.

`torch.compile` is only used on CUDA: on CPU the inductor C++ path needs a compiler and buys
little at Laya's batch sizes, and MPS is unsupported, so those raise `BackendUnavailable`.
"""
import os
import threading
import time
from typing import Optional

import torch

from .base import Backend, BackendUnavailable, pad_batch

CACHE_DIR_ENV = "LAYA_INDUCTOR_CACHE_DIR"
WARMUP_ENV = "LAYA_COMPILE_WARMUP"
DEFAULT_CACHE_DIR = os.path.join("~", ".cache", "laya", "inductor")
# rows x tokens x markers run at load: one short call, a few questions at a medium length, and a
# bigger batch; the three sizes differ within each shape so no two dimensions get tied together
WARMUP_SHAPES = ((1, 64, 3), (4, 128, 5), (8, 256, 6))


def configure_inductor_cache() -> str:
    """Point inductor's on-disk caches at a stable directory and switch the FX-graph cache on.

    Returns the directory in use. Inductor reads `TORCHINDUCTOR_CACHE_DIR` when it opens the
    cache, so this must run before the first compile; an explicit `TORCHINDUCTOR_CACHE_DIR` from
    the environment is left alone.
    """
    cache_dir = os.path.expanduser(os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)
    current = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    # torch writes its own default (/tmp/torchinductor_<user>) into the environment the first
    # time anything asks for the cache dir, so "already set" only counts when it is not that
    # default: an operator's explicit TORCHINDUCTOR_CACHE_DIR is kept, torch's is replaced.
    try:
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
        torch_default = default_cache_dir()
    except Exception:
        torch_default = None
    if current is None or current == torch_default:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    try:
        import torch._inductor.config as icfg
        icfg.fx_graph_cache = True
    except Exception:
        pass
    return os.environ["TORCHINDUCTOR_CACHE_DIR"]


def warmup_enabled(default: bool = True) -> bool:
    raw = os.environ.get(WARMUP_ENV)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


class CompileBackend(Backend):
    name = "compile"

    def __init__(self, agent, mode: str = "reduce-overhead", dynamic: bool = True,
                 warmup: Optional[bool] = None):
        super().__init__(agent)
        self.mode = mode
        self.dynamic = dynamic
        self.do_warmup = warmup_enabled(True) if warmup is None else bool(warmup)
        self.compiled = None
        self.cache_dir = None
        self.warmup_s = 0.0
        # CUDA graphs replay into static buffers: keep one forward in flight at a time.
        self._lock = threading.RLock()

    def _install(self):
        agent = self.agent
        if agent.device.type != "cuda":
            raise BackendUnavailable("torch.compile is used on CUDA only here; %s runs eager" % agent.device.type)
        if not hasattr(torch, "compile"):
            raise BackendUnavailable("this torch has no torch.compile")
        self.cache_dir = configure_inductor_cache()
        self.max_len = int(agent.cfg.get("max_len", 512))
        try:
            # ModernBERT's own reference_compile would wrap the encoder a second time.
            self.model.encoder.config.reference_compile = False
        except Exception:
            pass
        try:
            # Dynamo's "duck sizing" gives two dimensions that happen to be equal at the first
            # trace one symbol, and then guards on them staying equal: with four questions of four
            # options, rows == markers, and every request where they differ recompiled (~25 s).
            # Every dimension here is independent, so give each its own symbol.
            torch.fx.experimental._config.use_duck_shape = False
        except Exception:
            pass
        try:
            self.compiled = torch.compile(self._stock_forward, dynamic=self.dynamic, mode=self.mode)
        except Exception as e:
            raise BackendUnavailable(e)

    def _uninstall(self):
        self.compiled = None
        try:
            torch._dynamo.reset()
        except Exception:
            pass

    def install(self):
        super().install()
        if self.do_warmup:
            try:
                self.warmup_s = self.warmup()
            except Exception as e:
                # A compile failure surfaces here rather than on the first request.
                self.uninstall()
                raise BackendUnavailable(e)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        ids, att, mpos, mmask, qt, N, _ = pad_batch(
            input_ids, attention_mask, marker_pos, marker_mask, qtype, self.max_len,
            pad_id=self.agent.tok.pad_token_id or 0, keep_one_token=True)
        with self._lock:
            logits, act = self.compiled(ids, att, mpos, mmask, qt, detach_encoder)
            # copy out of the CUDA graph's static output buffers before the next replay
            return logits[:N].clone(), act[:N].clone()

    def warmup(self, shapes=None):
        agent = self.agent
        dev = agent.device
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=agent.dtype, enabled=agent.amp_enabled):
            for rows, tokens, markers in shapes or WARMUP_SHAPES:
                ids = torch.full((rows, tokens), agent.tok.pad_token_id or 0, dtype=torch.long, device=dev)
                ids[:, 0] = agent.tok.cls_token_id if agent.tok.cls_token_id is not None else 0
                att = torch.ones((rows, tokens), dtype=torch.long, device=dev)
                mpos = torch.arange(1, markers + 1, device=dev).repeat(rows, 1)
                mmask = torch.ones((rows, markers), dtype=torch.bool, device=dev)
                qt = torch.zeros((rows,), dtype=torch.long, device=dev)
                # cudagraph trees record on the second call of a shape and replay from the third
                for _ in range(3):
                    self.forward(ids, att, mpos, mmask, qt)
        torch.cuda.synchronize()
        return time.perf_counter() - t0
