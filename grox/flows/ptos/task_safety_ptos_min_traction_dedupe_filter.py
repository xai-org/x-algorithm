from typing import override

from monitor.metrics import Metrics
from strato_http.queries.grok_ptos_in_progress_dedup_cache import (
    StratoGrokPtosInProgressDedupCache,
    StratoGrokPtosInProgressDedupCacheAtla,
    StratoGrokPtosInProgressDedupCachePdxa,
)
from strato_http.queries.safety_post_annotations_result import (
    StratoSafetyPostAnnotationsResultDirectMh,
)

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task_filters import TaskFilterWithPost
from grox.flows.ptos.constants import (
    POST_MIN_IMPRESSION_STREAM_FOR_GROX_PTOS,
    POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS,
)


class TaskSafetyPtosMinTractionDedupeFilter(TaskFilterWithPost):
    _MIN_TRAFFIC_PTOS_TYPES: set[str] = {
        POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS,
        POST_MIN_IMPRESSION_STREAM_FOR_GROX_PTOS,
    }

    _in_progress_cache_default = StratoGrokPtosInProgressDedupCache()
    _in_progress_cache_atla = StratoGrokPtosInProgressDedupCacheAtla()
    _in_progress_cache_pdxa = StratoGrokPtosInProgressDedupCachePdxa()
    _result_direct_mh = StratoSafetyPostAnnotationsResultDirectMh()

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        Metrics.counter("task.safety_ptos_min_traction_filter.request.count").add(1)

        task_type = ctx.payload.task_type
        tweet_id = int(post.id)

        if task_type not in cls._MIN_TRAFFIC_PTOS_TYPES:
            Metrics.counter(
                "task.safety_ptos_min_traction_filter.skipped.ineligible_task_type.count"
            ).add(1)
            return True

        stream_name = task_type

        in_progress = await cls._in_progress_cache_default.fetch(tweet_id)
        if in_progress and len(in_progress) > 0:
            Metrics.counter(
                "task.safety_ptos_min_traction_filter.skipped.already_in_progress"
            ).add(1)
            return False

        existing_result = await cls._result_direct_mh.fetch(tweet_id)
        if existing_result:
            Metrics.counter(
                "task.safety_ptos_min_traction_filter.skipped.already_annotated"
            ).add(1)
            return False

        await cls._in_progress_cache_atla.put(tweet_id, [stream_name])
        await cls._in_progress_cache_pdxa.put(tweet_id, [stream_name])
        Metrics.counter("task.safety_ptos_min_traction_filter.eligible.count").add(1)
        return True
