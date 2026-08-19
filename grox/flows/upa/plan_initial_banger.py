from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.upa.task_banger_screen import TaskBangerScreen
from grox.flows.upa.task_filter import TaskInitialBangerFilter
from grox.flows.upa.task_grok_upa_action_with_labels import TaskGrokUpaActionWithLabels
from grox.flows.upa.task_rate_limit import TaskRateLimitBangerAnnotationWithPost
from grox.flows.upa.task_write import TaskPublishUnifiedPostAnnotationsManhattan


@register
class PlanInitialBanger(Plan):
    KEY = "banger_initial_screen"

    TASKS = {
        "task_initial_banger_filter": TaskInitialBangerFilter,
        "task_banger_annotation_rate_limit": TaskRateLimitBangerAnnotationWithPost,
        "task_media_hydration": TaskMediaHydration,
        "task_banger_screen_initial": TaskBangerScreen,
        "task_grok_upa_action_with_labels": TaskGrokUpaActionWithLabels,
        "task_publish_unified_post_annotations_manhattan": TaskPublishUnifiedPostAnnotationsManhattan,
    }

    TASK_DEPENDENCIES = {
        "task_initial_banger_filter": set(),
        "task_banger_annotation_rate_limit": {"task_initial_banger_filter"},
        "task_media_hydration": {"task_banger_annotation_rate_limit"},
        "task_banger_screen_initial": {"task_media_hydration"},
        "task_grok_upa_action_with_labels": {"task_banger_screen_initial"},
        "task_publish_unified_post_annotations_manhattan": {
            "task_banger_screen_initial"
        },
    }
