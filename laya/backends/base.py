"""The inference backend contract, and the two policies every backend shares.

A backend owns one thing: how `DecisionModel.forward` is executed. The Agent keeps tokenization,
collation, temperature scaling and decoding; a backend is handed the collated batch on the agent's
device and returns `(logits, act_logits)` with the same shapes and semantics the stock forward has.

    backend = make_backend("compile", agent)     # laya.backends
    backend.install()                            # swaps agent.model.forward; may raise
    ...
    backend.uninstall()                          # restores the stock forward

Two policies live here rather than in the backends so every backend sees the same inputs:

* dtype: `choose_dtype(device, cfg)` is the autocast policy `Agent.__init__` applies, and
  `agent.dtype` is what a backend must honour afterwards. There is one dtype per agent; a
  backend that cannot run in it declines (raises `BackendUnavailable`) rather than picking
  another one silently.
* padding: `bucket_shape(n_rows, n_tokens, max_len)` is the (rows, tokens) bucket a request is
  padded to before a shape-specialised backend (CUDA graphs under TileLang or `torch.compile`)
  sees it, and `pad_batch` applies it. Both graph backends use it, so they replay the same
  bounded set of shapes; the eager backend pads nothing.
"""
import os
import warnings
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch

from ..common import amp_dtype

# ------------------------------------------------------------------------------------- dtype policy
CPU_AMP_ENV = "LAYA_CPU_AMP"


def choose_dtype(device: torch.device, cfg: Dict[str, Any]) -> Tuple[torch.dtype, bool]:
    """The autocast policy: `(dtype, amp_enabled)` for a checkpoint config on a device.

    CUDA and MPS both support fp16/bf16 autocast and the shipped checkpoints are trained in
    reduced precision. On CUDA the checkpoint's `amp_dtype` (bf16 for the shipped ones) is
    used on compute capability >= 8; older GPUs (T4, V100) have no bf16 tensor cores and get
    fp16. CPU bf16 is only a win on hardware with native BF16, so it stays opt-in via
    LAYA_CPU_AMP=bf16. MPS is fp16, gated per call by `Agent.mps_amp_min_rows`.
    """
    if device.type == "cuda":
        if torch.cuda.get_device_capability(device)[0] < 8:
            return torch.float16, True
        return amp_dtype(cfg.get("amp_dtype", "fp16")), True
    if device.type == "mps":
        return torch.float16, True
    if device.type == "cpu" and os.environ.get(CPU_AMP_ENV, "").lower() in ("bf16", "bfloat16"):
        return torch.bfloat16, True
    return torch.float32, False


def amp_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    """Autocast context for the forward pass, or a no-op when mixed precision is not in use.

    Entering `torch.autocast` on a device torch has no autocast backend for raises even with
    `enabled=False` ("User specified an unsupported autocast device_type mps"), which broke
    every `predict()` on Apple Silicon on some torch builds. Only enter it when we use it.
    """
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# ---------------------------------------------------------------------------------- padding policy
DYNAMIC_MAX_L = 256   # up to here a dynamic-shape kernel is as fast as a static one: 16-token buckets
LONG_BUCKET = 64      # beyond it, 64-token buckets


def bucket_rows(n: int) -> int:
    """Rows are padded to the next power of two (1, 2, 4, ... questions per call)."""
    return 1 << max(0, (n - 1).bit_length())


def bucket_tokens(n_tokens: int, max_len: int) -> int:
    """Tokens are padded to a 16-token bucket up to `DYNAMIC_MAX_L`, 64-token buckets beyond, capped
    at `max_len` (the checkpoint's context, which the tokenizer never exceeds)."""
    g = 16 if n_tokens <= DYNAMIC_MAX_L else LONG_BUCKET
    return min(max_len, ((n_tokens + g - 1) // g) * g)


def bucket_shape(n_rows: int, n_tokens: int, max_len: int) -> Tuple[int, int]:
    return bucket_rows(n_rows), bucket_tokens(n_tokens, max_len)


def pad_batch(input_ids, attention_mask, marker_pos, marker_mask, qtype, max_len: int, pad_id: int = 0,
              keep_one_token: bool = False):
    """Pad a collated batch to its `bucket_shape`. Returns the padded tensors and `(N, L0)` to slice back.

    Padding rows carry `pad_id` tokens, an empty attention mask, markers at position 0 and qtype 0,
    so a backend that reads only the first `N` rows of its output sees exactly the request.
    `keep_one_token=True` marks token 0 of a padding row as valid instead: SDPA/eager attention
    turns an all-masked row into NaN, and although those rows are sliced away, a NaN in the batch
    is enough to poison a CUDA-graph replay under `torch.compile`; the TileLang attention kernel
    keeps fully masked rows finite by construction and does not need it.
    """
    N, L0 = input_ids.shape
    B, L = bucket_shape(N, L0, max_len)
    dev = input_ids.device
    ids = torch.full((B, L), pad_id, dtype=input_ids.dtype, device=dev)
    ids[:N, :L0] = input_ids
    att = torch.zeros((B, L), dtype=attention_mask.dtype, device=dev)
    att[:N, :L0] = attention_mask
    if keep_one_token and B > N:
        att[N:, 0] = 1
    mpos = torch.zeros((B, marker_pos.shape[1]), dtype=marker_pos.dtype, device=dev)
    mpos[:N] = marker_pos
    mmask = torch.zeros((B, marker_mask.shape[1]), dtype=marker_mask.dtype, device=dev)
    mmask[:N] = marker_mask
    # a padding row keeps one live marker so its softmax/entropy stay finite (the stock head
    # clamps k to 2 and masks the rest to -1e4; an all-masked row is a uniform softmax, fine too)
    if B > N:
        mmask[N:, 0] = True
    qt = torch.zeros((B,), dtype=qtype.dtype, device=dev)
    qt[:N] = qtype
    return ids, att, mpos, mmask, qt, N, L0


# ------------------------------------------------------------------------------------- the contract
class BackendUnavailable(RuntimeError):
    """The backend cannot run here (no CUDA, dependency missing, dtype unsupported): fall back."""


class Backend:
    """Base class. Subclasses set `name`, and override `_install`/`_uninstall`/`forward`.

    `max_len` is the longest sequence the backend accepts once installed, or None for no limit
    beyond the checkpoint's; `Agent._infer` reports a request past it with a clear message.
    """

    name = "eager"
    max_len: Optional[int] = None

    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model
        self.installed = False
        self._stock_forward = None
        self._prior_forward = None

    # -- lifecycle
    def install(self) -> None:
        """Swap `model.forward` for this backend's. Raises `BackendUnavailable` when it cannot run
        here (the caller then falls back to eager with one warning) and leaves the model untouched."""
        if self.installed:
            return
        # `forward` normally comes from the class; another replacement (an instance attribute) is
        # restored as it was, so backends never stack and uninstall leaves no trace.
        self._prior_forward = self.model.__dict__.get("forward")
        self._stock_forward = self.model.forward
        self._install()
        self.model.forward = self.forward
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        if self._prior_forward is None:
            self.model.__dict__.pop("forward", None)
        else:
            self.model.forward = self._prior_forward
        self._stock_forward = self._prior_forward = None
        self._uninstall()
        self.installed = False

    def _install(self) -> None:
        pass

    def _uninstall(self) -> None:
        pass

    def warmup(self, shapes=None) -> float:
        """Run representative shapes through the backend so the first real call is not the slow one.
        Returns seconds spent. The default backend has nothing to warm."""
        return 0.0

    # -- the forward itself
    def stock_forward(self, *args, **kwargs):
        """The model's own forward, whatever backend is installed."""
        fn = self._stock_forward if self._stock_forward is not None else type(self.model).forward.__get__(self.model)
        return fn(*args, **kwargs)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        return self.stock_forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)

    def __repr__(self):
        return "%s(installed=%s)" % (type(self).__name__, self.installed)


def warn_fallback(name: str, why: Any) -> None:
    """The one warning a caller sees when a requested backend cannot be used."""
    warnings.warn(
        "laya: backend %r is unavailable (%s); using the eager forward instead." % (name, why),
        RuntimeWarning, stacklevel=3)
