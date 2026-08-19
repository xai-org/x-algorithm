import logging
from abc import abstractmethod
from typing import override

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskStopExecution

logger = logging.getLogger(__name__)


class TaskFilter(Task):
    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        if not await cls._eligible(ctx):
            raise TaskStopExecution()

    @classmethod
    @abstractmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        raise NotImplementedError()


class TaskFilterWithPost(TaskFilter):
    @override
    @classmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        if not ctx.payload.post:
            return False
        return await cls._eligible_with_post(ctx.payload.post, ctx)

    @classmethod
    @abstractmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        raise NotImplementedError()
