"""Integration tests for the Redis shared-state backend.

Skipped unless a Redis is reachable via ``LAYA_TEST_REDIS_URL`` (e.g.
``redis://127.0.0.1:6379/15``). CI installs ``.[serve]`` only, so this stays skipped there.

Run: LAYA_TEST_REDIS_URL=redis://127.0.0.1:6379/15 python -m pytest tests/test_redis_state.py -q
"""
import asyncio
import os
import uuid

import pytest

pytest.importorskip("redis")

URL = os.environ.get("LAYA_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set LAYA_TEST_REDIS_URL to run Redis tests")

from laya.serving import ServingConfig  # noqa: E402
from laya.state import BreakerDecision  # noqa: E402


def _config(**kwargs):
    base = dict(request_timeout_s=5.0, breaker_threshold=2, breaker_cooldown_s=0.1,
                breaker_probe_successes=2, state_url=URL)
    base.update(kwargs)
    return ServingConfig(**base)


def _state(config):
    from laya.redis_state import RedisState

    return RedisState(config)


def _run(coro):
    return asyncio.run(coro)


def test_shared_admission_enforces_one_cluster_budget():
    async def scenario():
        first, second = _state(_config()), _state(_config())
        key = "t-" + uuid.uuid4().hex
        assert await first.acquire(key, 2) is True
        assert await second.acquire(key, 2) is True      # the second replica sees the first
        assert await second.acquire(key, 2) is False
        await first.release(key)
        assert await second.acquire(key, 2) is True
        await first.close()
        await second.close()
        return True

    assert _run(scenario())


def test_shared_breaker_across_replicas():
    async def scenario():
        first, second = _state(_config()), _state(_config())
        key = "t-" + uuid.uuid4().hex
        await first.breaker_failure(key, 2, 0.1)
        await second.breaker_failure(key, 2, 0.1)        # total failures shared -> open
        denied = await second.breaker_allow(key, 2, 0.1, 2)
        assert not denied.allowed and denied.retry_after >= 1
        await asyncio.sleep(0.15)
        probe = await first.breaker_allow(key, 2, 0.1, 2)
        assert probe.allowed and probe.probe
        await first.breaker_success(key, 2)
        probe2 = await first.breaker_allow(key, 2, 0.1, 2)
        assert probe2.allowed and probe2.probe
        await first.breaker_success(key, 2)              # streak met -> deleted
        assert (await second.breaker_allow(key, 2, 0.1, 2)).allowed
        await first.close()
        await second.close()
        return True

    assert _run(scenario())


def test_local_breaker_toggle_keeps_breaker_per_replica():
    async def scenario():
        config = _config(shared_breaker=False)
        first, second = _state(config), _state(config)
        key = "t-" + uuid.uuid4().hex
        await first.breaker_failure(key, 2, 0.1)         # only replica 1's local view fails
        assert (await second.breaker_allow(key, 2, 0.1, 2)).allowed
        await first.close()
        await second.close()
        return True

    assert _run(scenario())


def test_shared_gauges_are_cluster_wide():
    async def scenario():
        config = _config(breaker_cooldown_s=5.0)
        first, second = _state(config), _state(config)
        key = "t-" + uuid.uuid4().hex
        await first.breaker_failure(key, 2, 5.0)
        await first.breaker_failure(key, 2, 5.0)          # open on replica 1
        snap = await second.snapshot()                    # seen from replica 2
        assert snap["circuits_open"] >= 1.0
        assert snap["circuits_tracked"] >= 1.0

        adm = "t-" + uuid.uuid4().hex
        assert await first.acquire(adm, 1) is True
        snap2 = await second.snapshot()
        assert snap2["inflight_total"] >= 1.0
        await first.release(adm)
        await first.close()
        await second.close()
        return True

    assert _run(scenario())


def test_failure_policy_local_degrades_to_per_replica():
    async def scenario():
        config = _config(state_url="redis://127.0.0.1:63999/0", state_failure_policy="local")
        state = _state(config)
        assert await state.acquire("k", 1) is True          # store down, local fallback allows
        assert await state.acquire("k", 1) is False         # and still enforces the cap
        await state.release("k")
        assert await state.acquire("k", 1) is True
        await state.close()
        return True

    assert _run(scenario())


def test_failure_policy_open_lets_through():
    async def scenario():
        config = _config(state_url="redis://127.0.0.1:63999/0", state_failure_policy="open")
        state = _state(config)
        for _ in range(3):                                  # no enforcement during the outage
            assert await state.acquire("k", 1) is True
        await state.close()
        return True

    assert _run(scenario())


def test_fail_open_when_redis_is_unreachable():
    async def scenario():
        config = _config(state_url="redis://127.0.0.1:63999/0")  # nothing listening
        state = _state(config)
        assert await state.acquire("k", 1) is True       # fail open
        decision = await state.breaker_allow("k", 2, 0.1, 2)
        assert decision == BreakerDecision(True)
        await state.close()
        return True

    assert _run(scenario())
