import logging

from monitor.metrics import Metrics

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.reply_spam.classifier_coordinated_spam import CoordinatedSpamScorer
from grox.flows.reply_spam.state_coordinated_spam import CoordinatedSpamState

logger = logging.getLogger(__name__)


class TaskCoordinatedSpamDetection(TaskWithPost):
    scorer = CoordinatedSpamScorer()

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        result = await cls.scorer.score(post)
        ctx.state(CoordinatedSpamState).result = result
        if result.spam_post_ids:
            author_case = cls._author_case(post, result.spam_post_ids)
            logger.info(
                f"Coordinated spam found for post {post.id}: spam_post_ids={result.spam_post_ids} "
                f"author_case={author_case} reason={result.reason!r}"
            )
            Metrics.counter("task.coordinated_spam_detection.positive.count").add(
                1, attributes={"author_case": author_case}
            )
        else:
            Metrics.counter("task.coordinated_spam_detection.negative.count").add(1)

    @staticmethod
    def _author_case(post: Post, spam_post_ids: list[str]) -> str:
        reply_author = post.user.id if post.user else None
        author_by_id = {
            p.id: (p.user.id if p.user else None)
            for p in (post.ancestors or []) + [post]
        }
        spam_authors = {author_by_id.get(pid) for pid in spam_post_ids}
        if reply_author is None or None in spam_authors:
            return "unknown"
        if spam_authors == {reply_author}:
            return "same_author"
        return "mixed"
