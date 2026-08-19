from monitor.metrics import Metrics

from grox.core.data_loaders.data_types import Post
from grox.core.data_loaders.media_processor import MediaProcessor
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import TaskWithPost


class TaskMediaHydration(TaskWithPost):
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.media_hydration.total.count").add(1)
        await MediaProcessor.process(post)
        Metrics.counter("task.media_hydration.passed.count").add(1)
