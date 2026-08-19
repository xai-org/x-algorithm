import logging
import os
from functools import cache

from kafka_cli.producer import ScramKafkaProducer
from thrifts.gen.twitter.strato.columns.content_understanding.content_understanding.ttypes import (
    SimpleTweetEmbedding,
)
from thrifts.serdes import Serializer

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post
from grox.flows.mm_emb.constants import (
    TOPIC_EMBEDDING_V5,
    TOPIC_EMBEDDING_V5_ALL,
    TOPIC_EMBEDDING_V8_2,
)

logger = logging.getLogger(__name__)


class TaskPublishEmbeddingKafka:
    KAFKA_TOPIC_NAME: str

    @classmethod
    async def _publish_to_kafka(cls, post: Post, embedding: list[float]) -> None:
        tweet_embedding = SimpleTweetEmbedding(
            tweetId=int(post.id),
            embedding1=embedding,
        )
        serialized_bytes = Serializer.serialize(tweet_embedding)
        await cls._get_kafka_producer().send(id=post.id, value=serialized_bytes)
        logger.info(f"Published embedding for post {post.id} to {cls.KAFKA_TOPIC_NAME}")

    @classmethod
    @cache
    def _get_kafka_producer(cls) -> ScramKafkaProducer:
        producer_config = grox_config.get_kafka_producer_topic(cls.KAFKA_TOPIC_NAME)
        password = os.environ.get("KAFKA_RECSYS_PASSWORD")
        if not password:
            raise RuntimeError(
                "KAFKA_RECSYS_PASSWORD env var is required for SCRAM auth but is not set."
            )
        producer_config.ssl.sasl_plain_password = password
        return ScramKafkaProducer(producer_config)


class TaskPublishEmbeddingV5Kafka(TaskPublishEmbeddingKafka):
    KAFKA_TOPIC_NAME = TOPIC_EMBEDDING_V5


class TaskPublishEmbeddingV5AllKafka(TaskPublishEmbeddingKafka):
    KAFKA_TOPIC_NAME = TOPIC_EMBEDDING_V5_ALL


class TaskPublishEmbeddingV82Kafka(TaskPublishEmbeddingKafka):
    KAFKA_TOPIC_NAME = TOPIC_EMBEDDING_V8_2
