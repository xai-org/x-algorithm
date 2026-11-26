import asyncio
import contextlib
import logging
import time
import traceback
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import override

from kafka_cli.config import KafkaMessage
from kafka_cli.consumer import ConsumerRebalanceListener, KafkaConsumer, TopicPartition
from monitor.metrics import Metrics

from grox.config.config import grox_config
from grox.core.data_loaders.kafka_loader import (
    MAX_WORKING_THREADS,
    KafkaLoader,
    KafkaPostLoader,
    KafkaTweetEmbeddingLoader,
)
from grox.core.data_loaders.message_queue_loader import MessageQueuePayload

logger = logging.getLogger(__name__)
_DRAIN_POLL_INTERVAL_BUFFER_SEC = 60


class _CommitOnRebalance(ConsumerRebalanceListener):
    def __init__(self, loader: "ManualOffsetKafkaLoader"):
        self._loader = loader

    async def on_partitions_revoked(self, revoked: set[TopicPartition]) -> None:
        self._loader._rebalancing = True
        self._loader._drop_queued()
        await self._loader._commit()

    async def on_partitions_assigned(self, assigned: set[TopicPartition]) -> None:
        self._loader._forget_partitions_except(set(assigned))
        self._loader._rebalancing = False


class ManualOffsetKafkaLoader(KafkaLoader):
    BATCH_MESSAGES: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.consumer, KafkaConsumer):
            raise ValueError(
                f"{type(self).__name__} does not support multi-region kafka_consumers "
                f"(only single-cluster kafka_consumer_topics is supported), topic: {self.topic_name}"
            )
        if self.consumer_config.auto_commit:
            raise ValueError(
                f"{type(self).__name__} requires auto_commit: false on kafka_consumer_topics, topic: {self.topic_name}"
            )

        min_poll_interval_ms = int(
            (grox_config.shutdown_drain_timeout + _DRAIN_POLL_INTERVAL_BUFFER_SEC)
            * 1000
        )
        if self.consumer_config.max_poll_interval_ms < min_poll_interval_ms:
            logger.info(
                f"Raising max_poll_interval_ms from {self.consumer_config.max_poll_interval_ms} to {min_poll_interval_ms} "
                f"to exceed shutdown_drain_timeout ({grox_config.shutdown_drain_timeout}s), topic: {self.topic_name}"
            )
            self.consumer_config = self.consumer_config.model_copy(
                update={"max_poll_interval_ms": min_poll_interval_ms}
            )

        self.consumer = KafkaConsumer(
            self.consumer_config, listener=_CommitOnRebalance(self)
        )
        self._unfinished: dict[str, list[tuple[TopicPartition, int]]] = {}
        self._unfinished_since: dict[str, float] = {}
        self._next_offsets: dict[TopicPartition, int] = {}
        self._last_committed: dict[TopicPartition, int] = {}
        self._handed_out: set[str] = set()
        self._rebalancing = False
        self._commit_task: asyncio.Task | None = None
        self._commit_lock = asyncio.Lock()

    @override
    async def start(self):
        await super().start()
        self._commit_task = asyncio.create_task(self._commit_loop())

    @override
    async def stop(self):
        self._shutdown_event.set()
        await self._commit_after_in_flight()
        await super().stop()

    @override
    async def poll(self) -> AsyncGenerator[MessageQueuePayload | None, None]:
        payloads = super().poll()
        try:
            while not self._shutdown_event.is_set():
                if self._rebalancing:
                    await asyncio.sleep(0.01)
                    yield None
                    continue
                payload = await anext(payloads)
                if payload is None:
                    await asyncio.sleep(0)
                elif payload.mid not in self._unfinished:
                    continue
                else:
                    self._handed_out.add(payload.mid)
                yield payload
        finally:
            await payloads.aclose()

    @override
    async def ack(self, mid: str, success: bool = True):
        self._handed_out.discard(mid)
        if success or not self._shutdown_event.is_set():
            self._unfinished.pop(mid, None)
            self._unfinished_since.pop(mid, None)

    def _convert_single_message(
        self, message: KafkaMessage
    ) -> list[MessageQueuePayload]:
        try:
            return self._messages_to_payloads([message])
        except Exception:
            Metrics.counter("kafka_loader.message_dropped.count").add(
                1, attributes={"topic": self.topic_name}
            )
            logger.error(
                f"Skipping message {message.tp} offset {message.offset}, error: {traceback.format_exc()}"
            )
            return []

    def _convert_message_group(
        self, group: list[KafkaMessage]
    ) -> list[MessageQueuePayload]:
        try:
            return self._messages_to_payloads(group)
        except Exception:
            Metrics.counter("kafka_loader.message_dropped.count").add(
                len(group), attributes={"topic": self.topic_name}
            )
            logger.error(
                f"Skipping message group of size {len(group)}, topic: {self.topic_name}, error: {traceback.format_exc()}"
            )
            return []

    @override
    def _process_messages(
        self, messages: list[KafkaMessage]
    ) -> list[MessageQueuePayload]:
        eligible = [m for m in messages if m.offset >= self._next_offsets.get(m.tp, 0)]
        if not eligible:
            return []

        now = time.monotonic()
        payloads: list[MessageQueuePayload] = []
        if not self.BATCH_MESSAGES:
            with ThreadPoolExecutor(
                max_workers=min(MAX_WORKING_THREADS, len(eligible))
            ) as executor:
                for message, message_payloads in zip(
                    eligible,
                    executor.map(self._convert_single_message, eligible),
                    strict=True,
                ):
                    self._next_offsets[message.tp] = message.offset + 1
                    for payload in message_payloads:
                        self._unfinished[payload.mid] = [(message.tp, message.offset)]
                        self._unfinished_since[payload.mid] = now
                    payloads.extend(message_payloads)
            return payloads

        group_size = max(1, len(eligible) // MAX_WORKING_THREADS)
        eligible_groups = [
            eligible[i : i + group_size] for i in range(0, len(eligible), group_size)
        ]
        with ThreadPoolExecutor(
            max_workers=min(MAX_WORKING_THREADS, len(eligible_groups))
        ) as executor:
            for group, group_payloads in zip(
                eligible_groups,
                executor.map(self._convert_message_group, eligible_groups),
                strict=True,
            ):
                sources = [(m.tp, m.offset) for m in group]
                for m in group:
                    self._next_offsets[m.tp] = m.offset + 1
                for payload in group_payloads:
                    self._unfinished[payload.mid] = sources
                    self._unfinished_since[payload.mid] = now
                payloads.extend(group_payloads)
        return payloads

    def _record_unfinished_metrics(self) -> None:
        attributes = {"topic": self.topic_name}
        Metrics.gauge("kafka_loader.unfinished.count").set(
            len(self._unfinished), attributes=attributes
        )
        oldest_ts = next(iter(self._unfinished_since.values()), None)
        oldest_age = (
            max(0.0, time.monotonic() - oldest_ts) if oldest_ts is not None else 0.0
        )
        Metrics.gauge("kafka_loader.oldest_unfinished_age_sec").set(
            oldest_age, attributes=attributes
        )

    async def _commit_after_in_flight(self) -> None:
        deadline = time.monotonic() + grox_config.shutdown_drain_timeout
        while self._handed_out and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if self._commit_task is not None:
            self._commit_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._commit_task
            self._commit_task = None
        await self._commit()
        self._record_unfinished_metrics()

    async def _commit_loop(self) -> None:
        interval = max(0.01, self.loader_config.commit_interval_sec)
        while True:
            await asyncio.sleep(interval)
            try:
                self._record_unfinished_metrics()
                if not self._rebalancing:
                    await self._commit()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(
                    f"Unexpected error in manual commit loop, topic: {self.topic_name}, error: {traceback.format_exc()}"
                )

    async def _commit(self) -> None:
        async with self._commit_lock:
            offsets = self._commit_offsets()
            if not offsets:
                return
            try:
                await asyncio.wait_for(
                    self.consumer.commit(offsets),
                    timeout=self.consumer_config.commit_timeout_sec,
                )
                for tp, (offset, _) in offsets.items():
                    self._last_committed[tp] = offset
                Metrics.counter("kafka_loader.commit_succeeded.count").add(
                    1, attributes={"topic": self.topic_name}
                )
            except Exception:
                Metrics.counter("kafka_loader.commit_failed.count").add(
                    1, attributes={"topic": self.topic_name}
                )
                logger.warning(
                    f"Committing Kafka offsets failed, topic: {self.topic_name}, error: {traceback.format_exc()}"
                )

    def _commit_offsets(self) -> dict[TopicPartition, tuple[int, str]]:
        offsets = dict(self._next_offsets)
        for pairs in self._unfinished.values():
            for tp, offset in pairs:
                offsets[tp] = min(offsets[tp], offset)
        return {
            tp: (offset, "")
            for tp, offset in offsets.items()
            if self._last_committed.get(tp) != offset
        }

    def _drop_queued(self) -> None:
        while not self.queue.empty():
            payload = self.queue.get_nowait()
            self._unfinished_since.pop(payload.mid, None)
            for tp, offset in self._unfinished.pop(payload.mid, ()):
                self._next_offsets[tp] = min(self._next_offsets[tp], offset)

    def _forget_partitions_except(self, assigned: set[TopicPartition]) -> None:
        self._next_offsets = {
            tp: offset for tp, offset in self._next_offsets.items() if tp in assigned
        }
        self._last_committed = {
            tp: offset for tp, offset in self._last_committed.items() if tp in assigned
        }
        self._unfinished = {
            mid: kept
            for mid, pairs in self._unfinished.items()
            if (kept := [(tp, offset) for tp, offset in pairs if tp in assigned])
        }
        self._unfinished_since = {
            mid: ts
            for mid, ts in self._unfinished_since.items()
            if mid in self._unfinished
        }


class ManualOffsetKafkaPostLoader(ManualOffsetKafkaLoader, KafkaPostLoader):
    pass


class ManualOffsetKafkaTweetEmbeddingLoader(
    ManualOffsetKafkaLoader, KafkaTweetEmbeddingLoader
):
    pass
