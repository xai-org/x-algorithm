import os
import struct
import uuid
import asyncio
import logging
import traceback
from abc import abstractmethod
from typing import override
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from kafka_cli.config import KafkaMessage
from grox.config.config import grox_config
from kafka_cli.consumer import KafkaConsumer
from kafka_cli.multi_region_consumer import MultiRegionKafkaConsumer
from limits import RateLimitItemPerSecond, storage, strategies
from grox.core.data_loaders.data_types import Post
from grox.core.data_loaders.message_queue_loader import (
    MessageQueueLoader,
    MessageQueuePayload,
)
from monitor.metrics import Metrics
from thrifts.serdes import Deserializer
from thrifts.gen.twitter.strato.columns.content_understanding.content_understanding.ttypes import (
    SimpleTweetEmbedding,
)


logger = logging.getLogger(__name__)
MAX_WORKING_THREADS = 12
_limiter = strategies.FixedWindowRateLimiter(storage.MemoryStorage())


def parse_native_encoding(data: bytes) -> int:
    if not data or len(data) != 8:
        return 0
    buf = bytearray(data)
    buf[0] ^= 0x80
    return struct.unpack(">q", bytes(buf))[0]


class KafkaLoader(MessageQueueLoader):
    def __init__(self, topic_name: str):
        super().__init__()
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self.topic_name = topic_name
        self.loader_config = grox_config.get_kafka_loader_topic(topic_name)
        if topic_name in grox_config.kafka_consumers:
            self.consumer_config = grox_config.kafka_consumers[topic_name].model_copy(
                deep=True
            )
            self.consumer = MultiRegionKafkaConsumer(self.consumer_config)
        else:
            self.consumer_config = grox_config.get_kafka_consumer_topic(topic_name)
            self._maybe_inject_scram_password(self.consumer_config)
            self.consumer = KafkaConsumer(self.consumer_config)
        self.queue: asyncio.Queue[MessageQueuePayload] = asyncio.Queue()
        self._prefetcher_task: asyncio.Task | None = None
        self._max_qps_per_partition = self.loader_config.max_qps_per_partition
        self._limit_item: RateLimitItemPerSecond | None = None

    @staticmethod
    def _maybe_inject_scram_password(consumer_config) -> None:
        ssl_conf = consumer_config.ssl
        if ssl_conf is None or not ssl_conf.sasl_mechanism.startswith("SCRAM"):
            return
        if ssl_conf.sasl_plain_password:
            return
        password = os.environ.get("KAFKA_RECSYS_PASSWORD")
        if not password:
            raise RuntimeError(
                "KAFKA_RECSYS_PASSWORD env var is required for SCRAM auth but is not set."
            )
        ssl_conf.sasl_plain_password = password

    def _is_shutdown(self) -> bool:
        try:
            return self._shutdown_event.is_set()
        except Exception:
            logger.error(
                f"Error checking if KafkaLoader is shutdown: {traceback.format_exc()}"
            )
            return True

    async def start(self):
        logger.info(f"Initializing KafkaLoader, topic: {self.topic_name}")
        self._initialized = True
        await self.consumer.start()
        self._prefetcher_task = asyncio.create_task(self._prefetcher())
        self._initialized = True
        logger.info(f"KafkaLoader initialized, topic: {self.topic_name}")

    async def stop(self):
        logger.warning(f"Stopping KafkaLoader, topic: {self.topic_name}")
        self._shutdown_event.set()
        try:
            if self._prefetcher_task:
                await asyncio.wait_for(self._prefetcher_task, 5)
        except asyncio.TimeoutError:
            logger.warning(
                f"Waiting prefetcher to stop timed out, topic: {self.topic_name}"
            )
        await self.consumer.stop()
        logger.warning(f"KafkaLoader stopped, topic: {self.topic_name}")

    def _current_limit(self) -> RateLimitItemPerSecond | None:
        if not self._max_qps_per_partition:
            return None
        partitions = self.consumer.assigned_partitions
        if partitions <= 0:
            return self._limit_item
        amount = self._max_qps_per_partition * partitions
        if self._limit_item is None or self._limit_item.amount != amount:
            self._limit_item = RateLimitItemPerSecond(amount, 1)
        return self._limit_item

    async def poll(self) -> AsyncGenerator[MessageQueuePayload | None, None]:
        while not self._shutdown_event.is_set() or not self.queue.empty():
            limit = self._current_limit()
            if (
                limit
                and not self._shutdown_event.is_set()
                and not _limiter.test(limit, self.topic_name)
            ):
                yield None
                continue
            try:
                payload = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                logger.debug(
                    f"Queue is empty, waiting for prefetcher to fill, topic: {self.topic_name}"
                )
                yield None
                continue
            except Exception:
                logger.error(
                    f"Error polling from kafka, topic: {self.topic_name}, error: {traceback.format_exc()}"
                )
                yield None
                continue
            if limit:
                _limiter.hit(limit, self.topic_name)
            yield payload

    async def ack(self, mid: str, success: bool = True):
        pass

    @abstractmethod
    def _messages_to_payloads(
        self, messages: list[KafkaMessage]
    ) -> list[MessageQueuePayload]:
        pass

    def _process_messages(
        self, messages: list[KafkaMessage]
    ) -> list[MessageQueuePayload]:
        group_size = max(1, len(messages) // MAX_WORKING_THREADS)
        message_groups = [
            messages[i : i + group_size] for i in range(0, len(messages), group_size)
        ]

        with ThreadPoolExecutor(max_workers=MAX_WORKING_THREADS) as executor:
            payloads = []
            for result in executor.map(self._messages_to_payloads, message_groups):
                payloads.extend(result)
        return payloads

    async def _prefetcher(self) -> None:
        logger.info(f"Starting prefetcher, topic: {self.topic_name}")
        prefetching_threshold = self.loader_config.prefetching_threshold
        prefetching_batch_size = self.loader_config.prefetching_batch_size
        while not self._is_shutdown():
            if self.queue.qsize() < prefetching_threshold:
                logger.debug(
                    f"Inventory low at {self.queue.qsize()}, prefetching {prefetching_batch_size} messages, topic: {self.topic_name}"
                )
                try:
                    messages = await self.consumer.poll(prefetching_batch_size)
                    payloads = self._process_messages(messages)
                    await asyncio.gather(
                        *[self.queue.put(payload) for payload in payloads]
                    )
                    logger.debug(
                        f"Prefetched {prefetching_batch_size} messages, inventory now at {self.queue.qsize()}, topic: {self.topic_name}"
                    )
                except Exception:
                    logger.error(
                        f"Error prefetching messages, error: {traceback.format_exc()}"
                    )
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)
        logger.warning("Prefetcher stopped")


class KafkaPostLoader(KafkaLoader):
    def __init__(self, topic_name: str):
        super().__init__(topic_name)

    @override
    def _messages_to_payloads(
        self, messages: list[KafkaMessage]
    ) -> list[MessageQueuePayload]:
        payloads: list[MessageQueuePayload] = []
        for message in messages:
            try:
                post = Post.from_thrift_content_understanding_metadata(message.value)
            except Exception:
                Metrics.counter("kafka_loader.message_dropped.count").add(
                    1, attributes={"topic": self.topic_name}
                )
                logger.error(
                    f"Dropping un-deserializable message, topic: {self.topic_name}, error: {traceback.format_exc()}"
                )
                continue
            payloads.append(MessageQueuePayload(mid=uuid.uuid4().hex, post=post))
        return payloads


class KafkaTweetEmbeddingLoader(KafkaLoader):
    def __init__(self, topic_name: str):
        super().__init__(topic_name)

    @override
    def _messages_to_payloads(
        self, messages: list[KafkaMessage]
    ) -> list[MessageQueuePayload]:
        payloads: list[MessageQueuePayload] = []
        for message in messages:
            try:
                tweet_embedding = Deserializer.deserialize(
                    message.value, SimpleTweetEmbedding
                )
            except Exception:
                Metrics.counter("kafka_loader.message_dropped.count").add(
                    1, attributes={"topic": self.topic_name}
                )
                logger.error(
                    f"Dropping un-deserializable message, topic: {self.topic_name}, error: {traceback.format_exc()}"
                )
                continue
            if tweet_embedding.tweetId is None:
                Metrics.counter("kafka_tweet_embedding_loader.skipped.count").add(
                    1, attributes={"reason": "no_tweet_id"}
                )
                continue
            payloads.append(
                MessageQueuePayload(
                    mid=uuid.uuid4().hex,
                    post=Post(id=str(tweet_embedding.tweetId)),
                    embedding=tweet_embedding.embedding1,
                )
            )
        return payloads
