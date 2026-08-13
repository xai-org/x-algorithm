import asyncio
import logging
import traceback

from monitor.metrics import Metrics
from strato_http.queries.apply_label_from_grox import (
    SafetyLabelType,
    StratoApplyLabelFromGrox,
)
from strato_http.queries.data_types import (
    ReplyRankingScore,
    ReplyRankingScoreKafka,
)
from strato_http.queries.is_test_user import StratoIsTestUser

from grox.core.data_loaders.data_types import (
    Post,
)
from grox.core.data_loaders.strato_loader import UserStratoLoader
from grox.core.schedules.types import TaskContext
from grox.core.tasks.disable_rules import DisableTaskForNonProd
from grox.core.tasks.task import Task
from grox.flows.reply_spam.state_coordinated_spam import CoordinatedSpamState
from grox.flows.reply_spam.state_reply_ranking import (
    ReplyRankingState,
    ReplyScoreResult,
)
from grox.flows.reply_spam.strato_loader import ReplyRankingScoreStratoLoader

logger = logging.getLogger(__name__)


_strato_apply_label_from_grox = StratoApplyLabelFromGrox()
_strato_is_test_user = StratoIsTestUser()


async def _apply_reply_spam_label(post_id: str, author_id: int | None) -> None:
    if author_id is None:
        Metrics.counter("task.apply_label_from_grox.skipped.count").add(
            1, attributes={"reason": "no_author"}
        )
        return
    is_high_page_rank, is_grey_badge = await asyncio.gather(
        UserStratoLoader.is_high_page_rank_v2_user(author_id),
        UserStratoLoader.is_grey_badge_user(author_id),
    )
    if is_high_page_rank or is_grey_badge:
        logger.info(
            f"Skipping applyLabelFromGrox for post {post_id} user {author_id} "
            f"(high_page_rank_v2={is_high_page_rank}, grey_badge={is_grey_badge})"
        )
        Metrics.counter("task.apply_label_from_grox.skipped.count").add(
            1, attributes={"reason": "high_page_rank_or_grey_badge"}
        )
        return
    if await _strato_is_test_user.fetch(author_id):
        Metrics.counter("task.apply_label_from_grox.skipped.count").add(
            1, attributes={"reason": "test_user"}
        )
        return
    await _strato_apply_label_from_grox.apply(
        int(post_id), SafetyLabelType.RiskyHighVizReply
    )
    Metrics.counter("task.apply_label_from_grox.applied.count").add(
        1, attributes={"label": SafetyLabelType.RiskyHighVizReply.value}
    )
    logger.info(
        f"applyLabelFromGrox applied {SafetyLabelType.RiskyHighVizReply.value} for post {post_id}"
    )


class TaskWriteReplyRankingManhattan(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        post = ctx.payload.post
        result = ctx.state(ReplyRankingState).result
        if not post:
            return
        if result is None:
            Metrics.counter("task.write_reply_ranking_manhattan.skipped.count").add(
                1, attributes={"reason": "no_results"}
            )
            return
        Metrics.counter("task.write_reply_ranking_manhattan.intaken.count").add(1)
        try:
            await cls._publish_to_reply_ranking_manhattan(post, result)
            logger.info(
                f"Published reply ranking post to manhattan: {post.id=} user_id={post.user.id if post.user else None}"
            )
        except Exception:
            Metrics.counter("task.write_reply_ranking_manhattan.failed.count").add(1)
            logger.error(
                f"Failed to write reply ranking score to manhattan: {traceback.format_exc()}"
            )
            raise

    @classmethod
    async def _publish_to_reply_ranking_manhattan(
        cls, post: Post, result: ReplyScoreResult
    ):
        logger.info(f"[_publish_to_reply_ranking_manhattan] checking result: {result}")
        score = result.score
        reasoning = result.reason

        if post.user:
            logger.info(
                f"[_publish_to_reply_ranking_manhattan] {reasoning=} {post.id=} {post.user.id=} {score=}"
            )
        else:
            logger.info(
                f"Missing user id [_publish_to_reply_ranking_manhattan] {reasoning=} {post.id=} {score=}"
            )

        if score == 0.0:
            await _apply_reply_spam_label(post.id, post.user.id if post.user else None)

        await ReplyRankingScoreStratoLoader.save_reply_ranking_score(
            post_id=post.id,
            reply_ranking_score=ReplyRankingScore(
                score=score, reasoning=reasoning[-500:]
            ),
        )

        await ReplyRankingScoreStratoLoader.save_reply_ranking_kafka_v2(
            post_id=post.id,
            reply_ranking_score_kafka=ReplyRankingScoreKafka(
                postId=int(post.id), score=score, reasoning=reasoning[-500:]
            ),
        )

        Metrics.counter("task.write_reply_ranking_manhattan.success.count").add(
            1, attributes={"column": "reply_ranking"}
        )


class TaskWriteCoordinatedSpamReplyRanking(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        post = ctx.payload.post
        if not post:
            return
        result = ctx.state(CoordinatedSpamState).result
        if result is None:
            Metrics.counter("task.write_coordinated_spam.skipped.count").add(
                1, attributes={"reason": "no_results"}
            )
            return
        if not result.spam_post_ids:
            Metrics.counter("task.write_coordinated_spam.skipped.count").add(
                1, attributes={"reason": "no_spam"}
            )
            return

        thread_posts = (post.ancestors or []) + [post]
        spam_ids = set(result.spam_post_ids)
        reasoning = (result.reason or "")[-500:]

        Metrics.counter("task.write_coordinated_spam.intaken.count").add(1)
        for p in thread_posts:
            if p.id not in spam_ids:
                continue
            if not p.user:
                Metrics.counter("task.write_coordinated_spam.skipped_post.count").add(
                    1, attributes={"reason": "no_author"}
                )
                continue
            await cls._mark_spam(p.id, p.user.id, reasoning)
            Metrics.counter("task.write_coordinated_spam.success.count").add(1)

    @classmethod
    async def _mark_spam(cls, post_id: str, author_id: int, reasoning: str) -> None:
        await _apply_reply_spam_label(post_id, author_id)

        await ReplyRankingScoreStratoLoader.save_reply_ranking_score(
            post_id=post_id,
            reply_ranking_score=ReplyRankingScore(
                score=0.0, reasoning=reasoning[-500:]
            ),
        )

        await ReplyRankingScoreStratoLoader.save_reply_ranking_kafka_v2(
            post_id=post_id,
            reply_ranking_score_kafka=ReplyRankingScoreKafka(
                postId=int(post_id), score=0.0, reasoning=reasoning
            ),
        )
        logger.info(
            f"Published coordinated spam reply ranking score 0 for post {post_id}"
        )
