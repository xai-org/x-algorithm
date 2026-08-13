import logging

from monitor.metrics import Metrics

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.upa.classifier_post_safety_screen_deluxe import (
    PostSafetyDeluxeClassifier,
)
from grox.flows.upa.state_post_safety import PostSafetyState

logger = logging.getLogger(__name__)


class TaskPostSafetyScreenDeluxe(TaskWithPost):
    classifier = PostSafetyDeluxeClassifier()

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.post_safety_screen_deluxe.total.count").add(1)
        ctx.state(PostSafetyState).result = await cls.classifier.classify(post)
