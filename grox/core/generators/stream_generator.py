import abc
import logging
from collections.abc import AsyncGenerator

from grox.core.data_loaders.message_queue_loader import MessageQueueLoader
from grox.core.generators.task_generator import TaskGenerator
from grox.core.schedules.types import TaskPayload, TaskResult

logger = logging.getLogger(__name__)


class StreamTaskGenerator(TaskGenerator, metaclass=abc.ABCMeta):
    PLANS_TO_INJECT: set[str]

    def __init__(self, max_qps: int | None):
        super().__init__(max_qps)
        self._loader = self._get_loader()

    @abc.abstractmethod
    def _get_loader(self) -> MessageQueueLoader:
        pass

    async def start(self) -> None:
        logger.info("Starting StreamTaskGenerator")
        await self._loader.start()

    async def stop(self) -> None:
        logger.info("Stopping StreamTaskGenerator")
        await super().stop()
        await self._loader.stop()

    async def _poll(self) -> AsyncGenerator[TaskPayload | None, None]:
        async for payload in self._loader.poll():
            if not payload:
                yield None
                continue
            yield TaskPayload(
                payload_id=payload.mid,
                post=payload.post,
                user=payload.user,
                task_type=self.TASK_GENERATOR_TYPE,
                plans=self.PLANS_TO_INJECT.copy(),
                embedding=payload.embedding,
                extensions=payload.extensions,
            )

    async def ack(self, result: TaskResult):
        await self._loader.ack(result.task.payload_id, result.success)
