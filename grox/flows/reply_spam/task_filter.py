import asyncio
import logging
from typing import override

from monitor.metrics import Metrics

from grox.core.constants import GORK_USER_ID, GROK_USER_ID
from grox.core.data_loaders.data_types import Post
from grox.core.data_loaders.strato_loader import UserStratoLoader
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task_filters import TaskFilterWithPost

logger = logging.getLogger(__name__)


class TaskSpamFilter(TaskFilterWithPost):
    FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION = 60000

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "not_reply"}
            )
            return False
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "no_user"}
            )
            return False
        if post.user.id == GROK_USER_ID:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "is_grok_reply"}
            )
            return False
        if post.user.id == GORK_USER_ID:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "is_gork_reply"}
            )
            return False
        if not post.ancestors[-1].user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "spam_detection",
                    "reason": "previous_post_no_user",
                },
            )
            return False
        if post.user.id == post.ancestors[-1].user.id:
            logger.info(
                f"Skipping reply spam since the replier is same as reply target post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "same_user_reply"}
            )
            return False
        if not post.ancestors[0].user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={"filter": "spam_detection", "reason": "root_post_no_user"},
            )
            return False
        if post.user.id == post.ancestors[0].user.id:
            logger.info(
                f"Skipping reply spam since the replier is same as reply root post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "spam_detection",
                    "reason": "same_user_reply_as_root",
                },
            )
            return False
        in_reply_user_follower_count = post.ancestors[-1].user.follower_count or 0
        root_user_follower_count = post.ancestors[0].user.follower_count or 0
        if (
            in_reply_user_follower_count
            > cls.FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION
            or root_user_follower_count
            > cls.FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION
        ):
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "spam_detection",
                    "reason": "reply_ranking_target",
                },
            )
            return False
        return True


class TaskCoordinatedSpamFilter(TaskFilterWithPost):
    FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION = 1000
    FILTER_NAME = "coordinated_spam"

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": cls.FILTER_NAME, "reason": "not_reply"}
            )
            return False
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": cls.FILTER_NAME, "reason": "no_user"}
            )
            return False
        if post.user.id == GROK_USER_ID:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": cls.FILTER_NAME, "reason": "is_grok_reply"}
            )
            return False
        if post.user.id == GORK_USER_ID:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": cls.FILTER_NAME, "reason": "is_gork_reply"}
            )
            return False
        if not post.ancestors[-1].user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": cls.FILTER_NAME,
                    "reason": "previous_post_no_user",
                },
            )
            return False
        if not post.ancestors[0].user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": cls.FILTER_NAME, "reason": "root_post_no_user"}
            )
            return False
        if post.user.id == post.ancestors[0].user.id:
            logger.info(
                f"Skipping coordinated spam since the replier is same as reply root post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": cls.FILTER_NAME,
                    "reason": "same_user_reply_as_root",
                },
            )
            return False
        if len(post.ancestors) < 2:
            logger.info(
                f"Skipping coordinated spam since the reply thread is not more than two level deep {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": cls.FILTER_NAME, "reason": "one_level_deep"}
            )
            return False
        root_user_follower_count = post.ancestors[0].user.follower_count or 0
        if root_user_follower_count < cls.FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": cls.FILTER_NAME, "reason": "low_blast_radius"}
            )
            return False
        is_high_page_rank, is_grey_badge = await asyncio.gather(
            UserStratoLoader.is_high_page_rank_v2_user(post.user.id),
            UserStratoLoader.is_grey_badge_user(post.user.id),
        )
        if is_high_page_rank or is_grey_badge:
            logger.info(
                f"Skipping coordinated spam for post {post.id} user {post.user.id} "
                f"(high_page_rank_v2={is_high_page_rank}, grey_badge={is_grey_badge})"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": cls.FILTER_NAME,
                    "reason": "high_page_rank_or_grey_badge",
                },
            )
            return False
        Metrics.counter("task.coordinated_spam_filter.eligible.count").add(1)
        return True


class TaskReplyRankingFilter(TaskFilterWithPost):
    FOLLOWER_COUNT_THRESHOLD_FOR_REPLY_RANKING = 60000

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "not_reply"}
            )
            return False
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "no_user"}
            )
            return False
        if not post.ancestors[-1].user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "reply_ranking",
                    "reason": "previous_post_no_user",
                },
            )
            return False
        if not post.ancestors[0].user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "root_post_no_user"}
            )
            return False
        in_reply_user_follower_count = post.ancestors[-1].user.follower_count or 0
        root_user_follower_count = post.ancestors[0].user.follower_count or 0
        if (
            in_reply_user_follower_count
            <= cls.FOLLOWER_COUNT_THRESHOLD_FOR_REPLY_RANKING
            and root_user_follower_count
            <= cls.FOLLOWER_COUNT_THRESHOLD_FOR_REPLY_RANKING
        ):
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "low_blast_radius"}
            )
            return False
        if post.user.id == post.ancestors[-1].user.id:
            logger.info(
                f"Skipping reply ranking since the replier is same as reply target post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "same_user_reply"}
            )
            return False
        if post.user.id == post.ancestors[0].user.id:
            logger.info(
                f"Skipping reply ranking since the replier is same as reply root post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "reply_ranking",
                    "reason": "same_user_reply_as_root",
                },
            )
            return False

        Metrics.counter("task.reply_ranking.eligible.count").add(1)
        return True
