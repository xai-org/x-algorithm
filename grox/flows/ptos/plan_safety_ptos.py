from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.ptos.task_filter import TaskSafetyPtosFilter
from grox.flows.ptos.task_safety_ptos_category import TaskSafetyPtosCategoryDetection
from grox.flows.ptos.task_safety_ptos_adult_content_cross_validation import (
    TaskSafetyPtosAdultContentCrossValidation,
)
from grox.flows.ptos.task_safety_ptos_policy import TaskSafetyPtosPolicyDetection
from grox.flows.ptos.task_safety_ptos_safemodel_sex_nudity import (
    TaskSafetyPtosSafemodelSexNudity,
)
from grox.flows.ptos.task_rate_limit import TaskRateLimitSafetyPtosAnnotationWithPost
from grox.flows.ptos.task_safety_ptos_min_traction_dedupe_filter import (
    TaskSafetyPtosMinTractionDedupeFilter,
)
from grox.flows.ptos.task_write_safety_post_annotations_result_sink import (
    TaskWriteSafetyPostAnnotationsResultSink,
)


@register
class PlanSafetyPtos(Plan):
    KEY = "safety_ptos"

    TASKS = {
        "task_safety_ptos_filter": TaskSafetyPtosFilter,
        "task_safety_ptos_annotation_rate_limit": TaskRateLimitSafetyPtosAnnotationWithPost,
        "task_safety_ptos_min_traction_dedupe_filter": TaskSafetyPtosMinTractionDedupeFilter,
        "task_media_hydration": TaskMediaHydration,
        "task_safety_ptos_category_detection": TaskSafetyPtosCategoryDetection,
        "task_safety_ptos_policy_detection": TaskSafetyPtosPolicyDetection,
        "task_safety_ptos_safemodel_sex_nudity": TaskSafetyPtosSafemodelSexNudity,
        "task_safety_ptos_adult_content_cross_validation": TaskSafetyPtosAdultContentCrossValidation,
        "task_write_safety_post_annotations_result_sink": TaskWriteSafetyPostAnnotationsResultSink,
    }

    TASK_DEPENDENCIES = {
        "task_safety_ptos_filter": {},
        "task_safety_ptos_annotation_rate_limit": {"task_safety_ptos_filter"},
        "task_safety_ptos_min_traction_dedupe_filter": {
            "task_safety_ptos_annotation_rate_limit"
        },
        "task_media_hydration": {"task_safety_ptos_min_traction_dedupe_filter"},
        "task_safety_ptos_category_detection": {"task_media_hydration"},
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
