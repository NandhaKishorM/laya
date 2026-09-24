"""Pluggable shared state for the serving layer: per-client admission and circuit breakers.

``LocalState`` is the default and keeps everything in-process, so behavior without a
``LAYA_STATE_URL`` is exactly what it was before this module existed. A shared backend makes
admission and (optionally) the breaker cluster-wide; see ``laya.redis_state``.

Every method is async so a network-backed store fits without a second code path. Store errors
are fail-open: the request proceeds and ``laya_shared_state_errors_total`` is incremented,
because a state-store outage must not stop inference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Protocol

# A vanished client's breaker entry is reaped after at least this long, even if the configured
# cooldown is shorter; below it, a short cooldown would churn breaker state for no reason.
DEFAULT_ENTRY_TTL_FLOOR_S = 60.0
# Backstop: sweep immediately once the local breaker map passes this many keys, even before the
# TTL elapses, so an identity-churning client cannot grow it without bound.
BREAKER_SWEEP_LIMIT = 4096


class StateStore(Protocol):
    """Admission and breaker state, in-process or shared across replicas."""

    async def acquire(self, key: str, limit: int) -> bool:
        """Reserve one in-flight slot for `key`. `limit <= 0` means unlimited."""
        ...

    async def release(self, key: str) -> None:
        """Return one in-flight slot for `key`."""
        ...

    async def breaker_allow(self, key: str, threshold: int, cooldown_s: float,
                            probe_successes: int) -> "BreakerDecision":
        """Decide whether `key` may dispatch, marking a half-open probe if allowed."""
        ...

    async def breaker_success(self, key: str, probe_successes: int) -> None:
        """Record a successful decision for `key`, closing after enough half-open probes."""
        ...

    async def breaker_failure(self, key: str, threshold: int, cooldown_s: float) -> None:
        """Record a failed decision for `key`, opening or re-opening at the threshold."""
        ...

    async def snapshot(self) -> Dict[str, float]:
        """Gauges for the metrics surface. A shared store reports only a local view."""
        ...

    async def close(self) -> None:
        """Release any backing client."""
        ...


@dataclass
class BreakerDecision:
    """The breaker's answer for one submission."""

    allowed: bool
    retry_after: int = 1
    probe: bool = False


class LocalState:
    """In-process admission and breaker state. The default, behavior-preserving backend."""

    def __init__(self, config: Any):
        self._config = config
        self._inflight: Dict[str, int] = {}
        self._breakers: Dict[str, Dict[str, Any]] = {}
        self._last_sweep = 0.0

    # ------------------------------------------------------------------ admission
    async def acquire(self, key: str, limit: int) -> bool:
        if limit <= 0:
            return True
        current = self._inflight.get(key, 0)
        if current >= limit:
            return False
        self._inflight[key] = current + 1
        return True

    async def release(self, key: str) -> None:
        current = self._inflight.get(key, 0) - 1
        if current <= 0:
            self._inflight.pop(key, None)
        else:
            self._inflight[key] = current

    # ------------------------------------------------------------------ breaker
    async def breaker_allow(self, key: str, threshold: int, cooldown_s: float,
                            probe_successes: int) -> BreakerDecision:
        if not threshold:
            return BreakerDecision(True)
        state = self._breakers.get(key)
        if not state or not state["open_until"]:
            return BreakerDecision(True)
        now = time.perf_counter()
        if now < state["open_until"]:
            return BreakerDecision(False, retry_after=max(1, int(state["open_until"] - now + 0.999)))
        if state["probe"]:
            # Half-open: a probe is already in flight; do not flood a recovering engine.
            return BreakerDecision(False, retry_after=1)
        state["probe"] = True
        state["seen"] = now
        return BreakerDecision(True, probe=True)

    async def breaker_success(self, key: str, probe_successes: int) -> None:
        if not self._config.breaker_threshold:
            return
        state = self._breakers.get(key)
        if state is None:
            return
        state["seen"] = time.perf_counter()
        if state["open_until"]:
            state["probe"] = False
            state["probe_successes"] += 1
            if state["probe_successes"] >= probe_successes:
                self._breakers.pop(key, None)
        else:
            # A clean request in the closed state resets the streak and bounds memory.
            self._breakers.pop(key, None)

    async def breaker_failure(self, key: str, threshold: int, cooldown_s: float) -> None:
        if not threshold:
            return
        now = time.perf_counter()
        state = self._breakers.get(key)
        if state is None:
            self._maybe_sweep(now)
            state = {"failures": 0, "open_until": 0.0, "probe": False,
                     "probe_successes": 0, "seen": now}
            self._breakers[key] = state
        state["seen"] = now
        state["failures"] += 1
        if state["probe"] or state["failures"] >= threshold:
            state["open_until"] = now + cooldown_s
            state["probe"] = False
            state["probe_successes"] = 0

    def _effective_ttl(self) -> float:
        raw = getattr(self._config, "breaker_entry_ttl_s", None)
        if raw is None:
            return max(self._config.breaker_cooldown_s, DEFAULT_ENTRY_TTL_FLOOR_S)
        return float(raw)                       # 0 disables the TTL

    def _maybe_sweep(self, now: float) -> None:
        ttl = self._effective_ttl()
        if ttl <= 0:
            return
        if now - self._last_sweep < ttl and len(self._breakers) <= BREAKER_SWEEP_LIMIT:
            return
        self._last_sweep = now
        cutoff = now - ttl
        for stale in [k for k, v in self._breakers.items() if v.get("seen", now) < cutoff]:
            self._breakers.pop(stale, None)

    async def snapshot(self) -> Dict[str, float]:
        now = time.perf_counter()
        self._maybe_sweep(now)                 # a scrape also reaps expired breaker entries
        return {
            "clients_active": float(len(self._inflight)),
            "circuits_open": float(sum(1 for v in self._breakers.values()
                                       if v["open_until"] and v["open_until"] > now)),
            "circuits_tracked": float(len(self._breakers)),
            "inflight_total": float(sum(self._inflight.values())),
        }

    async def close(self) -> None:
        return None


def build_state(config: Any, metrics: Any = None) -> StateStore:
    """LocalState, or a shared backend when ``LAYA_STATE_URL`` is set.

    Raises ``ImportError`` with an install hint when a shared URL is set but the optional
    Redis support is not installed; the caller turns that into a ConfigError.
    """
    url = getattr(config, "state_url", None)
    if not url:
        return LocalState(config)
    try:
        from .redis_state import RedisState
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "LAYA_STATE_URL is set but Redis support is not installed; "
            "install 'laya[serve-redis]'") from exc

    return RedisState(config, metrics)
