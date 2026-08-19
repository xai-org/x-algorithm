from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.reply_spam.task_coordinated_spam import TaskCoordinatedSpamDetection
from grox.flows.reply_spam.task_filter import TaskCoordinatedSpamFilter
from grox.flows.reply_spam.task_rate_limit import (
    TaskRateLimitCoordinatedSpamAnnotationWithPost,
)
from grox.flows.reply_spam.task_write import TaskWriteCoordinatedSpamReplyRanking


@register
class PlanCoordinatedSpam(Plan):
    KEY = "coordinated_spam"

    TASKS = {
        "task_coordinated_spam_filter": TaskCoordinatedSpamFilter,
        "task_coordinated_spam_annotation_rate_limit": TaskRateLimitCoordinatedSpamAnnotationWithPost,
        "task_media_hydration": TaskMediaHydration,
        "task_coordinated_spam_detection": TaskCoordinatedSpamDetection,
        "task_write_coordinated_spam_reply_ranking": TaskWriteCoordinatedSpamReplyRanking,
    }

    TASK_DEPENDENCIES = {
        "task_coordinated_spam_filter": set(),
        "task_coordinated_spam_annotation_rate_limit": {"task_coordinated_spam_filter"},
        "task_media_hydration": {"task_coordinated_spam_annotation_rate_limit"},
        "task_coordinated_spam_detection": {"task_media_hydration"},
        "task_write_coordinated_spam_reply_ranking": {
            "task_coordinated_spam_detection"
        },
    }
