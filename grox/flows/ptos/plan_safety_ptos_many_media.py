from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.ptos.task_rate_limit import TaskRateLimitSafetyPtosManyMedia
from grox.flows.ptos.task_safety_ptos_many_media import (
    TaskSafetyPtosManyMediaChunks,
    TaskSafetyPtosManyMediaFilter,
)


@register
class PlanSafetyPtosManyMedia(Plan):
    KEY = "safety_ptos_many_media"

    TASKS = {
        "task_safety_ptos_annotation_rate_limit": TaskRateLimitSafetyPtosManyMedia,
        "task_safety_ptos_many_media_filter": TaskSafetyPtosManyMediaFilter,
        "task_media_hydration": TaskMediaHydration,
        "task_safety_ptos_many_media_chunks": TaskSafetyPtosManyMediaChunks,
    }

    TASK_DEPENDENCIES = {
        "task_safety_ptos_annotation_rate_limit": {},
        "task_safety_ptos_many_media_filter": {
            "task_safety_ptos_annotation_rate_limit"
        },
        "task_media_hydration": {"task_safety_ptos_many_media_filter"},
        "task_safety_ptos_many_media_chunks": {"task_media_hydration"},
    }
