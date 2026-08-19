import logging
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from functools import cache

from monitor.metrics import Metrics
from tenacity import retry, stop_after_attempt, wait_fixed

from grox.core.data_loaders.data_types import Post, User
from grox.core.lib.utils import camel_to_snake
from grox.core.schedules.types import TaskContext
from grox.core.tasks.disable_rules import DisableTaskRule

logger = logging.getLogger(__name__)


class TaskResultCategory(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"


class TaskStopExecution(Exception):
    pass


class Task(ABC):
    DISABLE_RULES: list[type[DisableTaskRule]] = []

    @classmethod
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        metrics_attributes = {"task_name": cls.get_name()}
        Metrics.counter("task.exec.count").add(1, attributes=metrics_attributes)
        logger.debug(f"[{cls.get_name()}] starting task")
        if cls.should_skip(ctx):
            Metrics.counter("task.exec.skipped.count").add(
                1, attributes=metrics_attributes
            )
            logger.info(f"[{cls.get_name()}] skipping task")
            return TaskResultCategory.SKIPPED
        Metrics.counter("task.exec.intaken.count").add(1, attributes=metrics_attributes)
        try:
            await cls._exec(ctx)
        except TaskStopExecution:
            Metrics.counter("task.exec.skipped.count").add(
                1, attributes=metrics_attributes
            )
            logger.info(f"[{cls.get_name()}] skipping task")
            return TaskResultCategory.SKIPPED
        except Exception:
            Metrics.counter("task.exec.failed.count").add(
                1, attributes=metrics_attributes
            )
            logger.error(
                f"[{cls.get_name()}] failed to execute task with error: {traceback.format_exc()}"
            )
            raise
        Metrics.counter("task.exec.success.count").add(1, attributes=metrics_attributes)
        return TaskResultCategory.SUCCESS

    @classmethod
    @abstractmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        raise NotImplementedError()

    @classmethod
    def should_skip(cls, ctx: TaskContext) -> bool:
        if cls.should_disable(ctx):
            return True
        return cls._should_skip(ctx)

    @classmethod
    def _should_skip(cls, ctx: TaskContext) -> bool:
        return False

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        for rule in cls.DISABLE_RULES:
            if rule.should_disable(ctx):
                logger.debug(
                    f"[{cls.get_name()}] skipping task because {rule.disable_reason()}"
                )
                return True
        return False

    @classmethod
    @cache
    def get_name(cls) -> str:
        return camel_to_snake(cls.__name__)


class TaskWithUser(Task):
    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        user = ctx.payload.user
        if not user:
            raise TaskStopExecution("No user for task")
        await cls._exec_with_user(ctx, user)

    @classmethod
    @abstractmethod
    async def _exec_with_user(cls, ctx: TaskContext, user: User) -> None:
        raise NotImplementedError()


class TaskWithPost(Task):
    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        post = ctx.payload.post
        if not post:
            raise TaskStopExecution("No post for task")
        await cls._exec_with_post(ctx, post)

    @classmethod
    @abstractmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        raise NotImplementedError()
