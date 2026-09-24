"""Redis-backed shared state, so admission and (optionally) the breaker span replicas.

Requires ``pip install 'laya[serve-redis]'``. Every counter and breaker transition is a single
Lua script, so an ``N``-replica deployment enforces one cluster-wide budget instead of ``N``
per-process budgets. Cluster-wide gauges come from a counter and two expiry-sorted sets, so a
scrape is O(log n) instead of a keyspace ``SCAN``.

A store error follows ``state_failure_policy``: ``local`` degrades to in-process enforcement
(still capped, per replica), ``open`` lets the request through. Either way the error is counted
and ``laya_state_degraded`` is set, because the datapath must survive the state store.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import redis.asyncio as redis

from .state import DEFAULT_ENTRY_TTL_FLOOR_S, BreakerDecision, LocalState

log = logging.getLogger("laya.serve")

KEY_PREFIX = "laya"
ADM_KEY = KEY_PREFIX + ":adm:"
INFLIGHT_KEY = KEY_PREFIX + ":inflight"
BRK_KEY = KEY_PREFIX + ":brk:"
BRK_OPEN_KEY = KEY_PREFIX + ":brk:open"
BRK_TRACKED_KEY = KEY_PREFIX + ":brk:tracked"
# When the breaker TTL is disabled, keys still get this safety expiry so a crashed process or a
# vanished client cannot leave state in Redis forever.
SAFETY_TTL_MS = 24 * 60 * 60 * 1000

# -- admission ---------------------------------------------------------------------------------
# KEYS[1]=per-client counter  KEYS[2]=global inflight  ARGV[1]=ttl ms  ARGV[2]=limit.
_ACQUIRE = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[1]) end
if n > tonumber(ARGV[2]) then redis.call('DECR', KEYS[1]); return 0 end
redis.call('INCR', KEYS[2])
redis.call('PEXPIRE', KEYS[2], ARGV[1])
return 1
"""
_RELEASE = """
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
local n = redis.call('DECR', KEYS[1])
if n <= 0 then redis.call('DEL', KEYS[1]); n = 0 end
local g = redis.call('DECR', KEYS[2])
if g < 0 then redis.call('SET', KEYS[2], 0) end
redis.call('PEXPIRE', KEYS[2], ARGV[1])
return n
"""

# -- breaker -----------------------------------------------------------------------------------
# A hash per client: failures, open_until (epoch ms), probe, probe_successes.
# ARGV[1]=threshold  ARGV[2]=cooldown ms  ARGV[3]=ttl ms  ARGV[4]=client
_BREAKER_ALLOW = """
local threshold = tonumber(ARGV[1])
if threshold == 0 then return {1, 0, 0} end
local now = tonumber(redis.call('TIME')[1]) * 1000 + math.floor(tonumber(redis.call('TIME')[2]) / 1000)
local open_until = tonumber(redis.call('HGET', KEYS[1], 'open_until') or '0')
if open_until > 0 and now < open_until then
  return {0, math.max(1, math.floor((open_until - now) / 1000 + 0.999)), 0}
end
if open_until > 0 then
  if tonumber(redis.call('HGET', KEYS[1], 'probe') or '0') == 1 then return {0, 1, 0} end
  redis.call('HSET', KEYS[1], 'probe', 1, 'seen', now)
  redis.call('PEXPIRE', KEYS[1], ARGV[3])
  redis.call('ZADD', KEYS[2], now + tonumber(ARGV[3]), ARGV[4])
  return {1, 0, 1}
end
return {1, 0, 0}
"""
# ARGV[1]=probe_successes needed  ARGV[2]=ttl ms  ARGV[3]=client
_BREAKER_SUCCESS = """
if redis.call('EXISTS', KEYS[1]) == 0 then return end
local now = tonumber(redis.call('TIME')[1]) * 1000 + math.floor(tonumber(redis.call('TIME')[2]) / 1000)
local open_until = tonumber(redis.call('HGET', KEYS[1], 'open_until') or '0')
if open_until > 0 then
  local ps = tonumber(redis.call('HGET', KEYS[1], 'probe_successes') or '0') + 1
  if ps < tonumber(ARGV[1]) then
    redis.call('HSET', KEYS[1], 'probe', 0, 'probe_successes', ps, 'seen', now)
    redis.call('PEXPIRE', KEYS[1], ARGV[2])
    redis.call('ZADD', KEYS[3], now + tonumber(ARGV[2]), ARGV[3])
    return
  end
end
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], ARGV[3])
redis.call('ZREM', KEYS[3], ARGV[3])
"""
# ARGV[1]=threshold  ARGV[2]=cooldown ms  ARGV[3]=ttl ms  ARGV[4]=client
_BREAKER_FAILURE = """
local threshold = tonumber(ARGV[1])
if threshold == 0 then return end
local now = tonumber(redis.call('TIME')[1]) * 1000 + math.floor(tonumber(redis.call('TIME')[2]) / 1000)
local f = tonumber(redis.call('HGET', KEYS[1], 'failures') or '0') + 1
local probe = tonumber(redis.call('HGET', KEYS[1], 'probe') or '0')
if probe == 1 or f >= threshold then
  redis.call('HSET', KEYS[1], 'failures', f, 'open_until', now + tonumber(ARGV[2]),
             'probe', 0, 'probe_successes', 0, 'seen', now)
  redis.call('ZADD', KEYS[2], now + tonumber(ARGV[2]), ARGV[4])
else
  redis.call('HSET', KEYS[1], 'failures', f, 'seen', now)
end
redis.call('PEXPIRE', KEYS[1], ARGV[3])
redis.call('ZADD', KEYS[3], now + tonumber(ARGV[3]), ARGV[4])
"""
# KEYS[1]=open zset  KEYS[2]=tracked zset  KEYS[3]=inflight counter. Returns {open, tracked, inflight}.
_SNAPSHOT = """
local now = tonumber(redis.call('TIME')[1]) * 1000 + math.floor(tonumber(redis.call('TIME')[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
return {redis.call('ZCARD', KEYS[1]), redis.call('ZCARD', KEYS[2]),
        tonumber(redis.call('GET', KEYS[3]) or '0')}
"""


class RedisState:
    """Shared ``StateStore`` over Redis, with an in-process fallback for store errors."""

    def __init__(self, config: Any, metrics: Any = None):
        self._config = config
        self._metrics = metrics
        self._client = redis.Redis.from_url(config.state_url, decode_responses=True)
        self._acquire = self._client.register_script(_ACQUIRE)
        self._release = self._client.register_script(_RELEASE)
        self._breaker_allow = self._client.register_script(_BREAKER_ALLOW)
        self._breaker_success = self._client.register_script(_BREAKER_SUCCESS)
        self._breaker_failure = self._client.register_script(_BREAKER_FAILURE)
        self._snapshot = self._client.register_script(_SNAPSHOT)
        # The in-process shadow: the primary breaker when shared_breaker is off, and the
        # degradation target on a store error when state_failure_policy is "local".
        self._fallback = LocalState(config)
        self._shared_breaker = bool(getattr(config, "shared_breaker", True))
        self._policy = getattr(config, "state_failure_policy", "local")

    # ------------------------------------------------------------------ helpers
    def _fail(self, op: str, exc: Exception) -> None:
        if self._metrics is not None:
            self._metrics.shared_state_error(op)
        log.warning("shared state error in %s: %s: %s", op, type(exc).__name__, exc)

    def _recover(self) -> None:
        if self._metrics is not None:
            self._metrics.state_recovered()

    def _admission_ttl_ms(self) -> int:
        return max(1000, int((self._config.request_timeout_s + 5.0) * 1000))

    def _breaker_ttl_ms(self) -> int:
        raw = getattr(self._config, "breaker_entry_ttl_s", None)
        ttl_s = max(self._config.breaker_cooldown_s, DEFAULT_ENTRY_TTL_FLOOR_S) if raw is None else float(raw)
        return int(ttl_s * 1000) if ttl_s > 0 else SAFETY_TTL_MS

    # ------------------------------------------------------------------ admission
    async def acquire(self, key: str, limit: int) -> bool:
        if limit <= 0:
            return True
        try:
            allowed = await self._acquire(keys=[ADM_KEY + key, INFLIGHT_KEY],
                                          args=[self._admission_ttl_ms(), limit])
            self._recover()
            return bool(allowed)
        except Exception as exc:  # noqa: BLE001
            self._fail("acquire", exc)
            if self._policy == "open":
                return True
            return await self._fallback.acquire(key, limit)   # degrade to per-replica

    async def release(self, key: str) -> None:
        try:
            await self._release(keys=[ADM_KEY + key, INFLIGHT_KEY], args=[self._admission_ttl_ms()])
            self._recover()
        except Exception as exc:  # noqa: BLE001
            self._fail("release", exc)
        await self._fallback.release(key)                     # release any degraded slot

    # ------------------------------------------------------------------ breaker
    async def breaker_allow(self, key: str, threshold: int, cooldown_s: float,
                            probe_successes: int) -> BreakerDecision:
        if not self._shared_breaker:
            return await self._fallback.breaker_allow(key, threshold, cooldown_s, probe_successes)
        if not threshold:
            return BreakerDecision(True)
        try:
            allowed, retry_after, probe = await self._breaker_allow(
                keys=[BRK_KEY + key, BRK_TRACKED_KEY],
                args=[threshold, int(cooldown_s * 1000), self._breaker_ttl_ms(), key])
            self._recover()
            return BreakerDecision(bool(allowed), retry_after=int(retry_after), probe=bool(probe))
        except Exception as exc:  # noqa: BLE001
            self._fail("breaker_allow", exc)
            if self._policy == "open":
                return BreakerDecision(True)
            return await self._fallback.breaker_allow(key, threshold, cooldown_s, probe_successes)

    async def breaker_success(self, key: str, probe_successes: int) -> None:
        if not self._shared_breaker:
            return await self._fallback.breaker_success(key, probe_successes)
        if self._policy == "local":
            await self._fallback.breaker_success(key, probe_successes)   # keep the shadow warm
        try:
            await self._breaker_success(
                keys=[BRK_KEY + key, BRK_OPEN_KEY, BRK_TRACKED_KEY],
                args=[probe_successes, self._breaker_ttl_ms(), key])
            self._recover()
        except Exception as exc:  # noqa: BLE001
            self._fail("breaker_success", exc)

    async def breaker_failure(self, key: str, threshold: int, cooldown_s: float) -> None:
        if not self._shared_breaker:
            return await self._fallback.breaker_failure(key, threshold, cooldown_s)
        if self._policy == "local":
            await self._fallback.breaker_failure(key, threshold, cooldown_s)
        try:
            await self._breaker_failure(
                keys=[BRK_KEY + key, BRK_OPEN_KEY, BRK_TRACKED_KEY],
                args=[threshold, int(cooldown_s * 1000), self._breaker_ttl_ms(), key])
            self._recover()
        except Exception as exc:  # noqa: BLE001
            self._fail("breaker_failure", exc)

    async def snapshot(self) -> Dict[str, float]:
        try:
            open_circuits, tracked, inflight = await self._snapshot(
                keys=[BRK_OPEN_KEY, BRK_TRACKED_KEY, INFLIGHT_KEY])
            self._recover()
            local = await self._fallback.snapshot()          # distinct local clients only
            return {"clients_active": local["clients_active"],
                    "circuits_open": float(open_circuits),
                    "circuits_tracked": float(tracked),
                    "inflight_total": float(inflight)}
        except Exception as exc:  # noqa: BLE001
            self._fail("snapshot", exc)
            return await self._fallback.snapshot()           # degraded: local view

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass
