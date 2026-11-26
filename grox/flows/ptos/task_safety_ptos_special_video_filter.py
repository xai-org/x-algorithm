from typing import override

from monitor.metrics import Metrics

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post, Video
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task_filters import TaskFilterWithPost

_METRIC_PREFIX = "task.safety_ptos_special_video_filter"


class TaskSafetyPtosSpecialVideoFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        videos = cls._videos(post)
        reason = cls._skip_reason(post, videos)
        if reason is not None:
            Metrics.counter(f"{_METRIC_PREFIX}.skipped.count").add(
                1, attributes={"reason": reason}
            )
            return False
        for video in videos:
            video.motion_reveal = True
        Metrics.counter(f"{_METRIC_PREFIX}.eligible.count").add(1)
        return True

    @classmethod
    def _videos(cls, post: Post) -> list[Video]:
        return [
            m
            for p in (post, post.quoted_post)
            if p is not None
            for m in p.media or []
            if isinstance(m, Video)
        ]

    @classmethod
    def _skip_reason(cls, post: Post, videos: list[Video]) -> str | None:
        if not post.user:
            return "no_user"
        if not videos:
            return "no_video"
        if (
            post.get_fav_count()
            < grox_config.media_hydration.deluxe_fav_count_threshold
        ):
            return "not_high_fav"
        return None
