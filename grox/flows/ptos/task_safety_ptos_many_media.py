import asyncio
import logging
import math
from typing import override

from monitor.metrics import Metrics
from strato_http.queries.data_types import SafetyPolicyCategory
from strato_http.queries.safety_post_annotations_result import (
    StratoSafetyPostAnnotationsResultDirectMh,
)

from grox.core.data_loaders.data_types import Post
from grox.core.plans.plan import Plan
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.core.tasks.task_filters import TaskFilterWithPost
from grox.flows.ptos.constants import (
    MANY_MEDIA_CHUNK_SIZE,
    MANY_MEDIA_UNCONDITIONAL_FAV,
)
from grox.flows.ptos.task_safety_ptos_adult_content_cross_validation import (
    TaskSafetyPtosAdultContentCrossValidation,
)
from grox.flows.ptos.task_safety_ptos_category import TaskSafetyPtosCategoryDetection
from grox.flows.ptos.task_safety_ptos_policy import TaskSafetyPtosPolicyDetection
from grox.flows.ptos.task_safety_ptos_safemodel_sex_nudity import (
    TaskSafetyPtosSafemodelSexNudity,
)
from grox.flows.ptos.task_write_safety_post_annotations_result_sink import (
    TaskWriteSafetyPostAnnotationsResultSink,
)

logger = logging.getLogger(__name__)

_FILTER_METRIC_PREFIX = "task.safety_ptos_many_media_filter"
_CHUNKS_METRIC_PREFIX = "task.safety_ptos_many_media_chunks"
MANY_MEDIA_MAX_CONCURRENT_CHUNKS = 5


class TaskSafetyPtosManyMediaFilter(TaskFilterWithPost):
    _result_direct_mh = StratoSafetyPostAnnotationsResultDirectMh()

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.user:
            return cls._skip("no_user")
        own, quoted = _media_counts(post)
        if max(own, quoted) <= MANY_MEDIA_CHUNK_SIZE:
            return cls._skip("few_media")
        favs = post.get_fav_count()
        forced = favs >= MANY_MEDIA_UNCONDITIONAL_FAV
        prior_adult = False
        if not forced:
            prior = await cls._result_direct_mh.fetch(int(post.id))
            if (
                prior is not None
                and prior.safetyBoolMetadata
                and prior.safetyBoolMetadata.isNsfw
            ):
                return cls._skip("already_nsfw")
            prior_adult = prior is not None and any(
                v.category == SafetyPolicyCategory.AdultContent
                for a in prior.safetyPostAnnotations or []
                for v in a.violatedPolicies or []
            )
            if not prior_adult:
                return cls._skip("no_prior_ptos" if prior is None else "no_adult_prior")
        chunks = math.ceil(max(own, quoted) / MANY_MEDIA_CHUNK_SIZE)
        Metrics.counter(f"{_FILTER_METRIC_PREFIX}.eligible.count").add(
            1,
            attributes={
                "chunks": str(chunks),
                "source": "post" if own >= quoted else "quoted",
                "forced": str(forced).lower(),
            },
        )
        logger.info(
            f"Post {post.id}: {own} media, {quoted} quoted media, favs={favs}, "
            f"forced={forced}, prior_adult={prior_adult}, eligible for {chunks} PTOS rounds"
        )
        return True

    @classmethod
    def _skip(cls, reason: str) -> bool:
        Metrics.counter(f"{_FILTER_METRIC_PREFIX}.skipped.count").add(
            1, attributes={"reason": reason}
        )
        return False


class PlanSafetyPtosManyMediaChunk(Plan):
    KEY = "safety_ptos_many_media_chunk"

    TASKS = {
        "task_safety_ptos_category_detection": TaskSafetyPtosCategoryDetection,
        "task_safety_ptos_policy_detection": TaskSafetyPtosPolicyDetection,
        "task_safety_ptos_safemodel_sex_nudity": TaskSafetyPtosSafemodelSexNudity,
        "task_safety_ptos_adult_content_cross_validation": TaskSafetyPtosAdultContentCrossValidation,
        "task_write_safety_post_annotations_result_sink": TaskWriteSafetyPostAnnotationsResultSink,
    }

    TASK_DEPENDENCIES = {
        "task_safety_ptos_category_detection": {},
        "task_safety_ptos_policy_detection": {"task_safety_ptos_category_detection"},
        "task_safety_ptos_safemodel_sex_nudity": {"task_safety_ptos_policy_detection"},
        "task_safety_ptos_adult_content_cross_validation": {
            "task_safety_ptos_policy_detection",
            "task_safety_ptos_safemodel_sex_nudity",
        },
        "task_write_safety_post_annotations_result_sink": {
            "task_safety_ptos_adult_content_cross_validation"
        },
    }


def _media_counts(post: Post) -> tuple[int, int]:
    return len(post.media or []), len(
        post.quoted_post.media or []
    ) if post.quoted_post else 0


def _media_window(post: Post, i: int) -> list:
    return (post.media or [])[
        i * MANY_MEDIA_CHUNK_SIZE : (i + 1) * MANY_MEDIA_CHUNK_SIZE
    ]


def chunk_posts(post: Post) -> list[Post]:
    chunks = []
    for i in range(math.ceil(max(_media_counts(post)) / MANY_MEDIA_CHUNK_SIZE)):
        update = {"media": _media_window(post, i)}
        if i > 0:
            update["screenshot"] = None
            update["combined_view"] = None
        if post.quoted_post:
            quoted_update = {"media": _media_window(post.quoted_post, i)}
            if i > 0:
                quoted_update["combined_view"] = None
            update["quoted_post"] = post.quoted_post.model_copy(update=quoted_update)
        chunks.append(post.model_copy(update=update))
    return chunks


class TaskSafetyPtosManyMediaChunks(TaskWithPost):
    _chunk_plan = PlanSafetyPtosManyMediaChunk()

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        chunks = chunk_posts(post)
        semaphore = asyncio.Semaphore(MANY_MEDIA_MAX_CONCURRENT_CHUNKS)

        async def _run_chunk(chunk: Post):
            async with semaphore:
                payload = ctx.payload.model_copy(
                    update={"post": chunk, "plans": {PlanSafetyPtosManyMediaChunk.KEY}}
                )
                return await cls._chunk_plan.execute(payload)

        results = await asyncio.gather(*(_run_chunk(chunk) for chunk in chunks))
        failed = 0
        for i, result in enumerate(results):
            ok = result is not None and result.success
            Metrics.counter(f"{_CHUNKS_METRIC_PREFIX}.round.count").add(
                1, attributes={"outcome": "success" if ok else "failed"}
            )
            if not ok:
                failed += 1
                logger.warning(
                    f"Post {post.id}: PTOS round {i + 1}/{len(chunks)} failed: {result.error if result else 'not run'}"
                )
        own, quoted = _media_counts(post)
        logger.info(
            f"Post {post.id}: {len(chunks)} PTOS rounds over {own} media and {quoted} quoted media, {failed} failed"
        )
        if failed == len(chunks):
            raise RuntimeError(f"every PTOS round failed for post {post.id}")

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
