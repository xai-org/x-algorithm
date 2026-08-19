from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.ptos.task_filter import TaskSafetyPtosRealtimeWithSignalsFilter
from grox.flows.ptos.task_ptos_spam_detection import TaskPtosSpamDetection
from grox.flows.ptos.task_rate_limit import TaskRateLimitSafetyPtosRealtimeWithSignals
from grox.flows.ptos.task_write_safety_post_annotations_result_sink import (
    TaskWriteSafetyPostAnnotationsResultSink,
)


@register
class PlanSafetyPtosWithRealTimeSignals(Plan):
    KEY = "safety_ptos_realtime_with_signals"

    TASKS = {
        "task_safety_ptos_realtime_with_signals_filter": TaskSafetyPtosRealtimeWithSignalsFilter,
        "task_safety_ptos_realtime_with_signals_rate_limit": TaskRateLimitSafetyPtosRealtimeWithSignals,
        "task_media_hydration": TaskMediaHydration,
        "task_ptos_spam_detection": TaskPtosSpamDetection,
        "task_write_safety_post_annotations_result_sink": TaskWriteSafetyPostAnnotationsResultSink,
    }

    TASK_DEPENDENCIES = {
        "task_safety_ptos_realtime_with_signals_filter": set(),
        "task_safety_ptos_realtime_with_signals_rate_limit": {
            "task_safety_ptos_realtime_with_signals_filter"
        },
        "task_media_hydration": {"task_safety_ptos_realtime_with_signals_rate_limit"},
        "task_ptos_spam_detection": {"task_media_hydration"},
        "task_write_safety_post_annotations_result_sink": {"task_ptos_spam_detection"},
    }
