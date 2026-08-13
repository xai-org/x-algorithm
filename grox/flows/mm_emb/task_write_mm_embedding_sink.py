import logging

from monitor.metrics import Metrics
from strato_http.queries.post_multimodal_embedding_mh_searchai import (
    StratoPostMultimodalEmbeddingMhSearchAiNoCache,
    TweetEmbedding,
)
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.mm_emb.disable_rules import DisableTaskForNonMmEmbProd
from grox.flows.mm_emb.state import MultimodalPostEmbeddingState
from grox.flows.mm_emb.task_embedding_pub import (
    TaskPublishEmbeddingV5AllKafka,
    TaskPublishEmbeddingV5Kafka,
    TaskPublishEmbeddingV82Kafka,
)

logger = logging.getLogger(__name__)


class TaskWriteMMEmbeddingSinkBase(TaskWithPost):
    model_version: str

    DISABLE_RULES = [DisableTaskForNonMmEmbProd]

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_chain(wait_fixed(1), wait_fixed(2)))
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)


class TaskWriteMMEmbeddingSinkV5(TaskWriteMMEmbeddingSinkBase):
    model_version = "v5_1"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        embedding = ctx.state(MultimodalPostEmbeddingState).embeddings[
            cls.model_version
        ]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAiNoCache()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        await TaskPublishEmbeddingV5Kafka._publish_to_kafka(post, embedding)

        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version})"
        )
        Metrics.counter("task.write_post_embedding_sink_v5.count").add(1)


class TaskWriteMMEmbeddingSinkV5SkipKafkaForReplies(TaskWriteMMEmbeddingSinkBase):
    model_version = "v5_1"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        embedding = ctx.state(MultimodalPostEmbeddingState).embeddings[
            cls.model_version
        ]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAiNoCache()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        is_reply = bool(post.ancestors)
        if not is_reply:
            await TaskPublishEmbeddingV5Kafka._publish_to_kafka(post, embedding)
        else:
            Metrics.counter(
                "task.write_post_embedding_sink_v5.kafka_skipped_reply.count"
            ).add(1)
            logger.info(
                f"Skipping Kafka publish for reply post {post.id} (written to Manhattan only)"
            )

        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version}, kafka={'yes' if not is_reply else 'no'})"
        )
        Metrics.counter("task.write_post_embedding_sink_v5.count").add(1)


class TaskWriteMMEmbeddingSinkV82SkipKafkaForReplies(TaskWriteMMEmbeddingSinkBase):
    model_version = "v8_2"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        embedding = ctx.state(MultimodalPostEmbeddingState).embeddings[
            cls.model_version
        ]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAiNoCache()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        is_reply = bool(post.ancestors)
        if not is_reply:
            await TaskPublishEmbeddingV82Kafka._publish_to_kafka(post, embedding)
        else:
            Metrics.counter(
                "task.write_post_embedding_sink_v82.kafka_skipped_reply.count"
            ).add(1)
            logger.info(
                f"Skipping Kafka publish for reply post {post.id} (written to Manhattan only)"
            )

        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version}, kafka={'yes' if not is_reply else 'no'})"
        )
        Metrics.counter("task.write_post_embedding_sink_v82.count").add(1)


class TaskWriteMMEmbeddingSinkV82(TaskWriteMMEmbeddingSinkBase):
    model_version = "v8_2"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        embedding = ctx.state(MultimodalPostEmbeddingState).embeddings[
            cls.model_version
        ]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAiNoCache()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        await TaskPublishEmbeddingV82Kafka._publish_to_kafka(post, embedding)

        logger.info(
            f"wrote post embedding to strato sink + kafka for post {post.id} (model: {cls.model_version})"
        )
        Metrics.counter("task.write_post_embedding_sink_v82.count").add(1)


class TaskWriteMMEmbeddingV5ToAllTopic(TaskWriteMMEmbeddingSinkBase):
    model_version = "v5_1"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        embedding = ctx.state(MultimodalPostEmbeddingState).embeddings[
            cls.model_version
        ]
        assert embedding is not None

        await TaskPublishEmbeddingV5AllKafka._publish_to_kafka(post, embedding)

        Metrics.counter("task.write_post_embedding_v5_all_kafka.count").add(1)
