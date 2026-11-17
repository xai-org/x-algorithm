import asyncio
import logging
import random
import traceback

from abc import ABC, abstractmethod
from grox.config.config import TaskGeneratorConfig
from grox.core.schedules.types import TaskResult, TaskPayload
from limits import RateLimitItemPerSecond, storage, strategies
from typing import AsyncGenerator

logger = logging.getLogger(__name__)
limiter = strategies.FixedWindowRateLimiter(storage.MemoryStorage())


class TaskGenerator(ABC):
    TASK_GENERATOR_TYPE: str | None = None

    def __init__(self, max_qps: int | None):
        self._shutdown_event = asyncio.Event()
        self._limiter_key = self.__class__.__name__
        self._limit = RateLimitItemPerSecond(max_qps, 1) if max_qps else None

    @classmethod
    def from_config(cls, cfg: TaskGeneratorConfig) -> "TaskGenerator":
        return cls(cfg.max_qps)

    def is_shutdown(self) -> bool:
        try:
            return self._shutdown_event.is_set()
        except Exception:
            logger.error(
                f"Error checking if task generator is shutdown: {traceback.format_exc()}"
            )
            return True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        logger.info(f"Stopping task generator {self.__class__.__name__}")
        self._shutdown_event.set()

    async def poll(self) -> AsyncGenerator[TaskPayload | None, None]:
        async for payload in self._poll():
            if not payload:
                yield None
                continue
            if self._limit:
                while not limiter.test(self._limit, self._limiter_key):
                    yield None
                    await asyncio.sleep(0.01)
                limiter.hit(self._limit, self._limiter_key)
            yield payload

    @abstractmethod
    def _poll(self) -> AsyncGenerator[TaskPayload | None, None]:
        pass

    async def ack(self, result: TaskResult):
        pass

    def identify_task_origin(self, result: TaskResult) -> str | None:
        return self.TASK_GENERATOR_TYPE

    def on_terminal_failure(self, result: TaskResult) -> None:
        pass


class PriorityTaskGenerator(TaskGenerator):
    def __init__(self, generators: list[tuple[TaskGenerator, int]]):
        if not generators:
            raise ValueError("No generators provided")
        if any(weight <= 0 for _, weight in generators):
            raise ValueError("All weights must be positive")
        super().__init__(None)
        self._generators: dict[str, TaskGenerator] = {}
        self._weights: dict[str, int] = {}
        for i, (gen, weight) in enumerate(generators):
            label = f"GEN_{i}"
            self._generators[label] = gen
            self._weights[label] = weight
        self._result_cache: dict[str, str] = {}
        logger.info(
            f"Initialized priority task generator with {list(zip(self._generators.keys(), [gen.__class__.__name__ for gen in self._generators.values()], self._weights.values(), strict=True))}"
        )

    async def start(self) -> None:
        logger.info("Starting priority task generators")
        await asyncio.gather(*[gen.start() for gen in self._generators.values()])
        self._streams = {label: gen.poll() for label, gen in self._generators.items()}
        logger.info("Priority task generators started")

    async def stop(self) -> None:
        logger.warning("Stopping priority task generators")
        await asyncio.gather(*[gen.stop() for gen in self._generators.values()])
        await super().stop()
        logger.warning("Priority task generators stopped")

    async def _poll(self) -> AsyncGenerator[TaskPayload | None, None]:
        if not self._streams:
            raise RuntimeError("Task generators not started")
        while self._weights:
            _weights = self._weights.copy()
            polled = False
            while _weights:
                labels = list(_weights.keys())
                weights = list(_weights.values())
                labels = random.choices(labels, weights, k=1)
                label = labels[0]
                stream = self._streams[label]
                try:
                    payload = await anext(stream)
                    if payload:
                        self._result_cache[payload.payload_id] = label
                        yield payload
                        polled = True
                        break
                    else:
                        del _weights[label]
                except StopAsyncIteration:
                    logger.warning(
                        f"Task generator {label} exhausted, removing from pool"
                    )
                    if label in _weights:
                        del _weights[label]
                    if label in self._weights:
                        del self._weights[label]
                    continue
                except Exception:
                    logger.error(
                        f"Error polling task generator {label}: {traceback.format_exc()}"
                    )
                    if label in _weights:
                        del _weights[label]
                    continue
            if not polled:
                yield None

    async def ack(self, result: TaskResult):
        logger.debug(f"Acknowledging task {result.task.payload_id}")
        label = self._result_cache.pop(result.task.payload_id, None)
        if not label:
            logger.warning(
                f"No label found for task {result.task.payload_id}, skipping ack"
            )
            return
        gen = self._generators[label]
        await gen.ack(result)

    def identify_task_origin(self, result: TaskResult) -> str | None:
        logger.debug(f"Identifying task origin for {result.task.payload_id}")
        label = self._result_cache.get(result.task.payload_id)
        if not label:
            logger.warning(
                f"No label found for task {result.task.payload_id}, cannot identify origin"
            )
            return None
        gen = self._generators[label]
        return gen.identify_task_origin(result)

    def on_terminal_failure(self, result: TaskResult) -> None:
        label = self._result_cache.get(result.task.payload_id)
        if not label:
            return
        self._generators[label].on_terminal_failure(result)
