"""Async client and retry policy for laya.

Laya inference is local CPU/GPU compute, not a network call, so there is no I/O to await.
:class:`AsyncAgent` therefore runs the forward pass in a thread executor: it keeps the
event loop responsive and lets callers fan out with ``asyncio.gather`` without blocking.
"""
import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

from .agent import Agent
from .errors import InferenceError, LayaError

logger = logging.getLogger("laya")

__all__ = ["RetryPolicy", "AsyncAgent", "with_retries"]


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter.

    Only transient failures are retried. Question/config errors are deterministic -- a
    retry would fail identically -- so by default only :class:`InferenceError` is retried.
    """

    max_attempts: int = 3
    initial_backoff: float = 0.1
    max_backoff: float = 5.0
    multiplier: float = 2.0
    jitter: bool = True
    retry_on: Tuple[Type[BaseException], ...] = (InferenceError,)

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1, got %r" % self.max_attempts)
        if self.initial_backoff < 0 or self.max_backoff < 0:
            raise ValueError("backoff values must be non-negative")

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        return attempt < self.max_attempts and isinstance(exc, self.retry_on)

    def backoff_for(self, attempt: int) -> float:
        """Delay in seconds before ``attempt`` + 1 (attempt is 1-based)."""
        delay = min(self.max_backoff, self.initial_backoff * (self.multiplier ** (attempt - 1)))
        if self.jitter:
            delay *= 0.5 + random.random() / 2.0
        return delay


NO_RETRY = RetryPolicy(max_attempts=1)


def with_retries(fn, policy: RetryPolicy, *args, **kwargs):
    """Call ``fn`` under a retry policy (synchronous)."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not policy.should_retry(exc, attempt):
                raise
            delay = policy.backoff_for(attempt)
            logger.warning(
                "laya: attempt %d/%d failed (%s: %s); retrying in %.2fs",
                attempt, policy.max_attempts, type(exc).__name__, exc, delay,
            )
            time.sleep(delay)


async def _await_retries(fn, policy: RetryPolicy, *args, **kwargs):
    attempt = 0
    loop = asyncio.get_event_loop()
    while True:
        attempt += 1
        try:
            return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
        except Exception as exc:
            if not policy.should_retry(exc, attempt):
                raise
            delay = policy.backoff_for(attempt)
            logger.warning(
                "laya: attempt %d/%d failed (%s: %s); retrying in %.2fs",
                attempt, policy.max_attempts, type(exc).__name__, exc, delay,
            )
            await asyncio.sleep(delay)


class AsyncAgent:
    """Async wrapper around :class:`laya.Agent`.

    Example::

        agent = await AsyncAgent.load("convaiinnovations/laya")
        result = await agent.system_one(state, questions)
        batch = await agent.batch([state_a, state_b], questions)
    """

    def __init__(self, agent: Agent, retry: Optional[RetryPolicy] = None):
        self._agent = agent
        self.retry = retry or RetryPolicy()

    @classmethod
    async def load(
        cls,
        model_id_or_path: str = "convaiinnovations/laya",
        retry: Optional[RetryPolicy] = None,
        **kwargs,
    ) -> "AsyncAgent":
        """Load the model off the event loop."""
        loop = asyncio.get_event_loop()
        agent = await loop.run_in_executor(
            None, functools.partial(Agent, model_id_or_path, **kwargs)
        )
        return cls(agent, retry=retry)

    @property
    def agent(self) -> Agent:
        """The wrapped synchronous agent."""
        return self._agent

    async def system_one(self, state, questions: Dict[str, Dict[str, Any]], retry=None):
        """Evaluate typed questions without blocking the event loop."""
        return await _await_retries(
            self._agent.system_one, retry or self.retry, state, questions
        )

    predict = system_one

    async def batch(
        self,
        states: Sequence[Any],
        questions: Dict[str, Dict[str, Any]],
        max_concurrency: int = 8,
        return_exceptions: bool = False,
    ):
        """Evaluate the same questions over many states concurrently."""
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1, got %r" % max_concurrency)
        sem = asyncio.Semaphore(max_concurrency)

        async def one(st):
            async with sem:
                return await self.system_one(st, questions)

        return await asyncio.gather(
            *(one(s) for s in states), return_exceptions=return_exceptions
        )
