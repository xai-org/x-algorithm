import asyncio
import logging
from typing import override

from monitor.metrics import Metrics
from strato_http.queries.safety_label import StratoSafetyLabel

from grox.core.data_loaders.data_types import Post
from grox.core.data_loaders.strato_loader import UserStratoLoader
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task_filters import TaskFilterWithPost
from grox.flows.ptos.constants import SAFETY_PTOS_DELUXE

logger = logging.getLogger(__name__)


class TaskSafetyPtosFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        is_deluxe = ctx.payload.task_type == SAFETY_PTOS_DELUXE
        filter_name = (
            "safety_ptos_deluxe_detection" if is_deluxe else "safety_ptos_detection"
        )

        Metrics.counter(f"task.{filter_name}.request.count").add(1)
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": filter_name, "reason": "no_user"}
            )
            return False

        Metrics.counter(f"task.{filter_name}.eligible.count").add(1)
        return True


class TaskSafetyPtosRealtimeWithSignalsFilter(TaskFilterWithPost):
    MIN_MENTION_COUNT = 2
    DELAY_SECS = 10

    safety_label = StratoSafetyLabel()

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "ptos_realtime", "reason": "no_user"}
            )
            return False
        text_handles, tag_handles = post.text_mention_handles, post.media_tag_handles
        handle_count = len(text_handles | tag_handles)
        for kind, count in (
            ("text", len(text_handles)),
            ("media_tags", len(tag_handles)),
            ("total", handle_count),
        ):
            Metrics.counter(
                "task.safety_ptos_realtime_with_signals_filter.mentions.count"
            ).add(
                1,
                attributes={
                    "kind": kind,
                    "handles": "3+" if count >= 3 else str(count),
                },
            )
        if handle_count < cls.MIN_MENTION_COUNT:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={"filter": "ptos_realtime", "reason": "not_enough_mention"},
            )
            return False
        is_high_page_rank, is_grey_badge = await asyncio.gather(
            UserStratoLoader.is_high_page_rank_v2_user(post.user.id),
            UserStratoLoader.is_grey_badge_user(post.user.id),
        )
        if is_high_page_rank or is_grey_badge:
            logger.info(
                f"Skipping ptos realtime for post {post.id} user {post.user.id} "
                f"(high_page_rank_v2={is_high_page_rank}, grey_badge={is_grey_badge})"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "ptos_realtime",
                    "reason": "high_page_rank_or_grey_badge",
                },
            )
            return False
        await asyncio.sleep(cls.DELAY_SECS)
        safety_labels = await cls.safety_label.scan(int(post.id))
        if "SpamHighRecall" in safety_labels:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "ptos_realtime", "reason": "already_spam"}
            )
            return False
        Metrics.counter(
            "task.safety_ptos_realtime_with_signals_filter.passed.count"
        ).add(1)
        return True
