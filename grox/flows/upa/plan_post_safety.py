from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.upa.task_filter import TaskPostSafetyDeluxeFilter
from grox.flows.upa.task_grok_upa_action_with_labels import TaskGrokUpaActionWithLabels
from grox.flows.upa.task_post_safety_screen_deluxe import TaskPostSafetyScreenDeluxe
from grox.flows.upa.task_rate_limit import TaskRateLimitPostSafetyAnnotationWithPost
from grox.flows.upa.task_write import TaskUpsertTweetBoolMetadataToUnifiedPostAnnotation


@register
class PlanPostSafety(Plan):
    KEY = "post_safety"

    TASKS = {
        "task_post_safety_deluxe_filter": TaskPostSafetyDeluxeFilter,
        "task_post_safety_annotation_rate_limit": TaskRateLimitPostSafetyAnnotationWithPost,
        "task_media_hydration": TaskMediaHydration,
        "task_post_safety_screen_deluxe": TaskPostSafetyScreenDeluxe,
        "task_grok_upa_action_with_labels": TaskGrokUpaActionWithLabels,
        "task_upsert_tweet_bool_metadata_to_unified_post_annotations_manhattan": TaskUpsertTweetBoolMetadataToUnifiedPostAnnotation,
    }

    TASK_DEPENDENCIES = {
        "task_post_safety_deluxe_filter": set(),
        "task_post_safety_annotation_rate_limit": {"task_post_safety_deluxe_filter"},
        "task_media_hydration": {"task_post_safety_annotation_rate_limit"},
        "task_post_safety_screen_deluxe": {"task_media_hydration"},
        "task_grok_upa_action_with_labels": {"task_post_safety_screen_deluxe"},
        "task_upsert_tweet_bool_metadata_to_unified_post_annotations_manhattan": {
            "task_post_safety_screen_deluxe"
        },
    }
