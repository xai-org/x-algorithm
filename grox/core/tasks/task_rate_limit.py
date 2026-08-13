import logging
from abc import abstractmethod
from collections.abc import Awaitable
from typing import override

from cachetools import TTLCache
from monitor.metrics import Metrics

from grox.core.data_loaders.data_types import Post
from grox.core.tasks.task import Task, TaskContext, TaskStopExecution

logger = logging.getLogger(__name__)


class TaskRateLimit(Task):
    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        eligible = await cls._eligible(ctx)
        Metrics.counter("task.rate_limit.count").add(
            1, attributes={"task_name": cls.get_name(), "passed": eligible}
        )
        if not eligible:
            raise TaskStopExecution()

    @classmethod
    @abstractmethod
    def _eligible(cls, ctx: TaskContext) -> Awaitable[bool]:
        raise NotImplementedError()


class TaskRateLimitWithPost(TaskRateLimit):
    @override
    @classmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        if not ctx.payload.post:
            return False
        return await cls._eligible_with_post(ctx.payload.post, ctx)

    @classmethod
    @abstractmethod
    def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> Awaitable[bool]:
        raise NotImplementedError()


class TaskTTLDedupeWithPost(TaskRateLimitWithPost):
    DEDUPE_CACHE: TTLCache
    DEDUPE_NAME = ""

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        return cls._dedupe(post.id, cls.DEDUPE_CACHE, cls.DEDUPE_NAME)

    @classmethod
    def _dedupe(cls, post_id: str, cache: TTLCache, name: str) -> bool:
        if post_id not in cache:
            cache[post_id] = True
            return True
        logger.info(f"Post {post_id} seen in the last {cache.ttl}s, deduped ({name})")
        return False
