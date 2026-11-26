from typing import override

from cachetools import TTLCache

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task_rate_limit import TaskTTLDedupeWithPost
from grox.flows.ptos.constants import DELUXE_TIER_TASK_TYPES


class TaskRateLimitSafetyPtosAnnotationWithPost(TaskTTLDedupeWithPost):
    POST_CACHE_FOR_SAFETY_PTOS = TTLCache(maxsize=10_000, ttl=60)
    POST_CACHE_FOR_SAFETY_PTOS_DELUXE = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        is_deluxe = ctx.payload.task_type in DELUXE_TIER_TASK_TYPES
        cache = (
            cls.POST_CACHE_FOR_SAFETY_PTOS_DELUXE
            if is_deluxe
            else cls.POST_CACHE_FOR_SAFETY_PTOS
        )
        name = "safety ptos deluxe" if is_deluxe else "safety ptos"
        return cls._dedupe(post.id, cache, name)


class TaskRateLimitSafetyPtosAdultContentLeadingFrames(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "safety ptos adult content leading frames"


class TaskRateLimitSafetyPtosKeyFrames(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "safety ptos key frames"


class TaskRateLimitSafetyPtosManyMedia(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "safety ptos many media"


class TaskRateLimitSafetyPtosMediaInjectedAdultInfraredVideoSpamDetection(
    TaskTTLDedupeWithPost
):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "safety ptos media injected adult infrared video spam detection"


class TaskRateLimitSafetyPtosRealtimeWithSignals(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "safety ptos realtime with signals"


class TaskRateLimitSafetyPtosLiveClusterAnchors(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "safety ptos live cluster anchors"
