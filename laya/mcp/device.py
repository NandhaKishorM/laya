"""Device resolution: LAYA_DEVICE has the same meaning as in laya.serve.

The environment variable is passed straight to torch (``Router(device=...)``);
``resolve_device`` is a best-effort label of the *configured preference*
(LAYA_DEVICE or torch auto-detection), not of the device a loaded checkpoint
actually runs on. For the real device read ``Agent.device`` via
``agent_device`` / ``router_agent``: the Agent falls back to CPU silently when
the requested GPU is unavailable or runs out of memory, and ``Agent.device``
reflects the fallback.
"""

from __future__ import annotations

import os
from typing import Any

_ENV_KEY = "LAYA_DEVICE"


def env_device() -> str | None:
    """LAYA_DEVICE verbatim (empty or unset = None), ready for Router(device=...)."""
    value = os.environ.get(_ENV_KEY)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_device(force: str | None = None) -> str:
    """Best-effort label of the configured torch device preference.

    Priority:
      1. explicit ``force`` argument (tests)
      2. ``LAYA_DEVICE`` environment variable (the same value serve passes to torch)
      3. ``torch.cuda.is_available()``
      4. CPU

    This is what the Router is asked to build on, not what a loaded agent
    computes on: use ``agent_device`` for the real device (a GPU -> CPU
    fallback happens silently at agent build time).
    """
    if force:
        return force
    value = env_device()
    if value is not None:
        return value.lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def device_report() -> dict:
    """Structured device info for laya_status.

    ``device`` is the configured preference (see ``resolve_device``); it is
    the top-level answer only before any checkpoint is loaded.
    """
    torch_cuda = False
    torch_version = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        torch_cuda = bool(torch.cuda.is_available())
    except Exception:
        pass

    return {
        "device": resolve_device(),
        "torch_cuda": torch_cuda,
        "torch_version": torch_version,
    }


def agent_device(agent: Any) -> str | None:
    """The real device a loaded agent computes on (str), or None if unreadable.

    Reads ``Agent.device``, a ``torch.device`` that already reflects the
    silent GPU -> CPU fallback done at build time. Accepts a plain string too,
    so tests can fake an agent without torch.
    """
    if agent is None:
        return None
    device = getattr(agent, "device", None)
    if device is None:
        return None
    kind = getattr(device, "type", None)
    if isinstance(kind, str) and kind:
        return kind
    if isinstance(device, str) and device:
        return device
    return None


def router_agent(router: Any, name: str) -> Any | None:
    """The agent a router currently holds for checkpoint ``name``, or None.

    Read-only on purpose: it reads the private ``_agents`` mapping (the same
    dictionary ``Router.load`` populates, keyed by the normalised name) and
    never calls ``load()``, because ``load`` has side effects: it reorders the
    LRU for a resident checkpoint and rebuilds an evicted one (hundreds of MB)
    -- unacceptable for a device-label read.
    """
    if router is None:
        return None
    agents = getattr(router, "_agents", None)
    if not isinstance(agents, dict):
        return None
    if name in agents:
        return agents[name]
    # Routers key _agents by the normalised name (Router.load normalises the
    # same way); try the core normaliser for aliased inputs, without heavy
    # imports at module level.
    try:
        from laya.router import normalise_name

        key = normalise_name(name)
    except Exception:
        return None
    return agents.get(key) if key != name else None
