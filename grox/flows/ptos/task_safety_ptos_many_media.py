import logging
import math
from typing import override

from monitor.metrics import Metrics

from grox.core.data_loaders.data_types import Post
from grox.core.plans.plan import Plan
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.core.tasks.task_filters import TaskFilterWithPost
from grox.flows.ptos.constants import MANY_MEDIA_CHUNK_SIZE
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


class TaskSafetyPtosManyMediaFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.user:
            return cls._skip("no_user")
        own, quoted = _media_counts(post)
        if max(own, quoted) <= MANY_MEDIA_CHUNK_SIZE:
            return cls._skip("few_media")
        chunks = math.ceil(max(own, quoted) / MANY_MEDIA_CHUNK_SIZE)
        Metrics.counter(f"{_FILTER_METRIC_PREFIX}.eligible.count").add(
            1,
            attributes={
                "chunks": str(chunks),
                "source": "post" if own >= quoted else "quoted",
            },
        )
        logger.info(
            f"Post {post.id}: {own} media, {quoted} quoted media, eligible for {chunks} PTOS rounds"
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
        if post.quoted_post:
            update["quoted_post"] = post.quoted_post.model_copy(
                update={"media": _media_window(post.quoted_post, i)}
            )
        chunks.append(post.model_copy(update=update))
    return chunks


class TaskSafetyPtosManyMediaChunks(TaskWithPost):
    _chunk_plan = PlanSafetyPtosManyMediaChunk()

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        chunks = chunk_posts(post)
        failed = 0
        for i, chunk in enumerate(chunks):
            payload = ctx.payload.model_copy(
                update={"post": chunk, "plans": {PlanSafetyPtosManyMediaChunk.KEY}}
            )
            result = await cls._chunk_plan.execute(payload)
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
