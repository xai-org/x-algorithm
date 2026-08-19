import logging
from typing import override

from monitor.metrics import Metrics

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task_filters import TaskFilterWithPost

logger = logging.getLogger(__name__)


class TaskInitialBangerFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        Metrics.counter("task.initial_banger_filter.count").add(1)
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "content_understanding", "reason": "no_user"}
            )
            return False
        if post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "content_understanding", "reason": "reply"}
            )
            logger.info(f"Skipping post {post.id} because it is a reply")
            return False
        filter_reason = cls._get_hardcoded_filter_reason(post)
        if filter_reason:
            logger.info(
                f"Skipping post {post.id} because it is hit by hardcoded filters, reason: {filter_reason}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={"filter": "content_understanding", "reason": filter_reason},
            )
            return False
        logger.info(f"Post {post.id} is eligible for initial banger")
        Metrics.counter("task.initial_banger_filter.eligible.count").add(1)
        return True

    @classmethod
    def _get_hardcoded_filter_reason(cls, post: Post) -> str | None:
        if not post.user:
            return None
        if post.user.is_protected:
            return "private_account"
        return None


class TaskPostSafetyDeluxeFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "post_safety_deluxe", "reason": "no_user"}
            )
            return False

        if post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "post_safety_deluxe", "reason": "reply"}
            )
            logger.info(f"Skipping post {post.id} because it is a reply")
            return False

        filter_reason = cls._get_hardcoded_filter_reason(post)
        if filter_reason:
            logger.info(
                f"Skipping upa deluxe {post.id} because it is hit by hardcoded filters, reason: {filter_reason}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "post_safety_deluxe", "reason": filter_reason}
            )
            return False

        Metrics.counter("task.post_safety_deluxe.eligible.count").add(1)
        return True

    @classmethod
    def _get_hardcoded_filter_reason(cls, post: Post) -> str | None:
        if not post.user:
            return None
        if post.user.is_protected:
            return "private_account"
        return None
