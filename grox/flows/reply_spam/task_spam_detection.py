import logging
from typing import override

from monitor.metrics import Metrics

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.reply_spam.classifier_simple_reply_scorer import SimpleReplyScorer
from grox.flows.reply_spam.state_reply_ranking import ReplyRankingState

logger = logging.getLogger(__name__)


class TaskSpamDetection(TaskWithPost):
    scorer = SimpleReplyScorer()

    @classmethod
    def get_follower_bucket_string(cls, post: Post) -> str:
        if not post.ancestors:
            return "invalid"
        in_reply_user_follower_count = post.ancestors[-1].user.follower_count or 0
        root_user_follower_count = post.ancestors[0].user.follower_count or 0
        if in_reply_user_follower_count <= 100 and root_user_follower_count <= 100:
            return "lte_100"
        elif in_reply_user_follower_count <= 500 and root_user_follower_count <= 500:
            return "lte_500"
        elif in_reply_user_follower_count <= 1000 and root_user_follower_count <= 1000:
            return "lte_1000"
        else:
            return "gt_1000"

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @override
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        result = await cls.scorer.score(post)
        ctx.state(ReplyRankingState).result = result

        score = result.score
        follower_bucket_string = cls.get_follower_bucket_string(post)
        if score == 0.0:
            logger.info(
                f"Reply Spam Found for lower than 1000 follower bucket. The post_id is {post.id} and the follower bucket is {follower_bucket_string}"
            )
            Metrics.counter("task.spam_comment_detection.positive.count").add(
                1, attributes={"reason": follower_bucket_string}
            )
        else:
            Metrics.counter("task.spam_comment_detection.negative.count").add(
                1, attributes={"reason": follower_bucket_string}
            )
