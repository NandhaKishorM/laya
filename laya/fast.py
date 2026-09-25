"""GPU fast path for laya: TileLang fused kernels + bf16 resident weights + CUDA graphs.

    agent = laya.load("convaiinnovations/laya", fast=True)      # or agent.accelerate()

Requires CUDA and `pip install laya[fast]` (tilelang).  Falls back to the stock forward otherwise.

The implementation moved to `laya.backends.tilelang` when the backends were given one contract
(`laya.backends`); this module keeps the old import path working.
"""
from .backends.tilelang import BF, FastLaya, TileLangBackend, _bucket_n, _top_two  # noqa: F401
