import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    def __init__(self, name: str, remaining_seconds: float):
        self.name = name
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Circuit breaker '{name}' is OPEN (recovery in {remaining_seconds:.1f}s)"
        )


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_rate_threshold: float = 0.5
    window_size: float = 60.0
    min_calls_in_window: int = 20
    recovery_timeout: float = 15.0
    half_open_max_calls: int = 10
    excluded_exceptions: tuple[type[BaseException], ...] = ()


@dataclass
class CircuitBreakerMetrics:
    state: CircuitState
    total_calls: int
    failure_count: int
    failure_rate: float


class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._outcomes: deque[tuple[float, bool]] = deque()
        self._opened_at: float = 0.0
        self._half_open_calls: int = 0
        self._half_open_completed: int = 0
        self._half_open_successes: int = 0
        self._lock = asyncio.Lock()

    def get_metrics(self) -> CircuitBreakerMetrics:
        self._evict_old()
        total = len(self._outcomes)
        failures = sum(1 for _, s in self._outcomes if not s)
        return CircuitBreakerMetrics(
            state=self._state,
            total_calls=total,
            failure_count=failures,
            failure_rate=failures / total if total > 0 else 0.0,
        )

    @property
    def recovery_due(self) -> bool:
        return (
            self._state == CircuitState.OPEN
            and (time.monotonic() - self._opened_at) >= self.config.recovery_timeout
        )

    @asynccontextmanager
    async def guard(self):
        async with self._lock:
            self._pre_call_check()

        try:
            yield
        except BaseException as exc:
            if isinstance(exc, CircuitBreakerOpen):
                raise
            if self._is_excluded_exception(exc):
                await self._record_success()
            else:
                await self._record_failure()
            raise
        else:
            await self._record_success()

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        async with self.guard():
            return await func(*args, **kwargs)

    async def reset(self) -> None:
        async with self._lock:
            self._outcomes.clear()
            self._half_open_calls = 0
            self._half_open_completed = 0
            self._half_open_successes = 0
            self._transition_to(CircuitState.CLOSED)

    def _pre_call_check(self) -> None:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.config.recovery_timeout:
                self._half_open_calls = 0
                self._half_open_completed = 0
                self._half_open_successes = 0
                self._transition_to(CircuitState.HALF_OPEN)
            else:
                remaining = self.config.recovery_timeout - elapsed
                raise CircuitBreakerOpen(self.name, remaining)

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpen(self.name, 0)
            self._half_open_calls += 1

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                self._half_open_completed += 1
                self._evaluate_half_open()
                return

            self._outcomes.append((time.monotonic(), True))

            if self._state == CircuitState.CLOSED:
                self._evaluate_thresholds()

    async def _record_failure(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_completed += 1
                self._evaluate_half_open()
                return

            self._outcomes.append((time.monotonic(), False))

            if self._state == CircuitState.CLOSED:
                self._evaluate_thresholds()

    def _evaluate_half_open(self) -> None:
        if self._half_open_completed < self.config.half_open_max_calls:
            return

        failures = self._half_open_completed - self._half_open_successes
        failure_rate = failures / self._half_open_completed

        if failure_rate < self.config.failure_rate_threshold:
            self._outcomes.clear()
            self._transition_to(CircuitState.CLOSED)
            logger.info(
                f"Circuit breaker '{self.name}': half-open test calls passed "
                f"({self._half_open_successes}/{self._half_open_completed} succeeded, "
                f"failure_rate={failure_rate:.2%} < threshold={self.config.failure_rate_threshold:.2%}), closing circuit"
            )
        else:
            self._open(
                f"half-open test failure_rate={failure_rate:.2%} ({failures}/{self._half_open_completed}) "
                f">= threshold={self.config.failure_rate_threshold:.2%}"
            )

    def _evaluate_thresholds(self) -> None:
        self._evict_old()
        total = len(self._outcomes)

        if total < self.config.min_calls_in_window:
            return

        failures = sum(1 for _, s in self._outcomes if not s)
        failure_rate = failures / total

        if failure_rate >= self.config.failure_rate_threshold:
            self._open(
                f"failure_rate={failure_rate:.2%} ({failures}/{total}) >= threshold={self.config.failure_rate_threshold:.2%}"
            )

    def _open(self, reason: str) -> None:
        self._opened_at = time.monotonic()
        self._half_open_calls = 0
        self._half_open_completed = 0
        self._half_open_successes = 0
        logger.warning(f"Circuit breaker '{self.name}' OPENING: {reason}")
        self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        logger.warning(
            f"Circuit breaker '{self.name}': {old_state.value} → {new_state.value}"
        )

    def _evict_old(self) -> None:
        cutoff = time.monotonic() - self.config.window_size
        while self._outcomes and self._outcomes[0][0] < cutoff:
            self._outcomes.popleft()

    def _is_excluded_exception(self, exc: BaseException) -> bool:
        return isinstance(exc, self.config.excluded_exceptions)

    def __repr__(self) -> str:
        m = self.get_metrics()
        return f"CircuitBreaker(name={self.name!r}, state={m.state.value}, failure_rate={m.failure_rate:.2%}, calls={m.total_calls})"
