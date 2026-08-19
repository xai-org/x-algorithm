import logging

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.reply_spam.classifier_reply_ranking import ReplyScorer
from grox.flows.reply_spam.state_reply_ranking import ReplyRankingState

logger = logging.getLogger(__name__)


class TaskRankReplies(TaskWithPost):
    scorer = ReplyScorer()

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        user = post.user
        logger.info(
            f"[task_rank_replies] {post.id=} "
            f"is_pasted={post.is_pasted} "
            f"user_agent={post.user_agent!r} "
            f"composition_source={post.composition_source!r} "
            f"app_attestation_status={post.app_attestation_status!r} "
            f"has_risky_user_safety_label={user.has_risky_user_safety_label if user else None} "
            f"num_legit_blocks_received_last_24hrs={user.num_legit_blocks_received_last_24hrs if user else None}"
        )
        ctx.state(ReplyRankingState).result = await cls.scorer.score(post)
