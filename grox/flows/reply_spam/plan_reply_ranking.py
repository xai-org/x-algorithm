from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.reply_spam.task_filter import TaskReplyRankingFilter
from grox.flows.reply_spam.task_rank_replies import TaskRankReplies
from grox.flows.reply_spam.task_rate_limit import (
    TaskRateLimitReplyRankingAnnotationWithPost,
)
from grox.flows.reply_spam.task_write import TaskWriteReplyRankingManhattan


@register
class PlanReplyRanking(Plan):
    KEY = "reply_ranking"

    TASKS = {
        "task_reply_ranking_filter": TaskReplyRankingFilter,
        "task_reply_ranking_annotation_rate_limit": TaskRateLimitReplyRankingAnnotationWithPost,
        "task_media_hydration": TaskMediaHydration,
        "task_rank_replies": TaskRankReplies,
        "task_write_reply_ranking_manhattan": TaskWriteReplyRankingManhattan,
    }

    TASK_DEPENDENCIES = {
        "task_reply_ranking_filter": set(),
        "task_reply_ranking_annotation_rate_limit": {"task_reply_ranking_filter"},
        "task_media_hydration": {"task_reply_ranking_annotation_rate_limit"},
        "task_rank_replies": {"task_media_hydration"},
        "task_write_reply_ranking_manhattan": {"task_rank_replies"},
    }
