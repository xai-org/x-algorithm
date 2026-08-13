import asyncio
import logging
import time

from aiokafka import AIOKafkaConsumer
from aiokafka.coordinator.assignors.sticky.sticky_assignor import (
    StickyPartitionAssignor,
)
from monitor.metrics import Metrics
from pydantic import BaseModel, Field, model_validator

from kafka_cli.config import KafkaMessage
from kafka_cli.mtls import create_mtls_ssl_context

logger = logging.getLogger(__name__)


class MultiRegionKafkaConsumerConfig(BaseModel):
    topic: str
    group_id: str
    clusters: dict[str, list[str]]
    auto_offset_reset: str = Field(default="latest")
    fetch_max_bytes: int = Field(default=50 * 1024 * 1024)
    fetch_min_bytes: int = Field(default=1024 * 128)
    max_poll_records: int = Field(default=500)
    request_timeout_ms: int = Field(default=30000)

    @model_validator(mode="after")
    def _validate_clusters(self) -> "MultiRegionKafkaConsumerConfig":
        if not self.clusters:
            raise ValueError("`clusters` must contain at least one region")
        for region, brokers in self.clusters.items():
            if not brokers:
                raise ValueError(f"Region {region!r} must list at least one broker")
        return self


RETRY_FAILED_REGION_INTERVAL_SEC = 60


class MultiRegionKafkaConsumer:
    def __init__(self, config: MultiRegionKafkaConsumerConfig):
        self.config = config
        self.group_id: str = config.group_id
        self._consumers: dict[str, AIOKafkaConsumer] = {}
        self._region_retry_task: asyncio.Task | None = None

    async def start(self):
        clusters = list(self.config.clusters.items())
        results = await asyncio.gather(
            *[self._start_region(region, brokers) for region, brokers in clusters],
            return_exceptions=True,
        )
        failed: dict[str, list[str]] = {}
        for (region, brokers), result in zip(clusters, results):
            if isinstance(result, BaseException):
                logger.error(f"Failed to start region {region!r}: {result!r}")
                Metrics.counter("kafka_consumer.start_errors.count").add(
                    1, attributes={"group_id": self.group_id, "region": region}
                )
                failed[region] = brokers
            else:
                self._consumers[region] = result
        if not self._consumers:
            raise next(r for r in results if isinstance(r, BaseException))
        if failed:
            self._region_retry_task = asyncio.create_task(
                self._retry_failed_regions(failed)
            )

    async def _retry_failed_regions(self, failed: dict[str, list[str]]):
        while failed:
            await asyncio.sleep(RETRY_FAILED_REGION_INTERVAL_SEC)
            for region in list(failed):
                try:
                    self._consumers[region] = await self._start_region(
                        region, failed[region]
                    )
                    del failed[region]
                    logger.info(f"Region {region!r} recovered and rejoined consumption")
                except Exception as e:
                    logger.error(f"Retry failed for region {region!r}: {e!r}")
                    Metrics.counter("kafka_consumer.start_errors.count").add(
                        1, attributes={"group_id": self.group_id, "region": region}
                    )

    async def _start_region(self, region: str, brokers: list[str]) -> AIOKafkaConsumer:
        config = self.config
        consumer = AIOKafkaConsumer(
            config.topic,
            bootstrap_servers=",".join(brokers),
            group_id=config.group_id,
            auto_offset_reset=config.auto_offset_reset,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            security_protocol="SSL",
            ssl_context=create_mtls_ssl_context(),
            request_timeout_ms=config.request_timeout_ms,
            fetch_max_bytes=config.fetch_max_bytes,
            fetch_min_bytes=config.fetch_min_bytes,
            max_poll_records=config.max_poll_records,
            partition_assignment_strategy=[StickyPartitionAssignor],
        )
        try:
            await consumer.start()
        except BaseException:
            try:
                await consumer.stop()
            except Exception:
                logger.exception(
                    f"Failed to clean up consumer after failed start for region {region!r}"
                )
            raise
        logger.info(
            f"MultiRegionKafkaConsumer started region {region!r}: topic={config.topic} "
            f"group_id={config.group_id} partitions={len(consumer.assignment() or ())}"
        )
        return consumer

    async def stop(self):
        if self._region_retry_task is not None:
            self._region_retry_task.cancel()
            try:
                await self._region_retry_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Region retry task failed during shutdown")
            self._region_retry_task = None
        for region, consumer in self._consumers.items():
            try:
                await consumer.stop()
            except Exception:
                logger.exception(f"Failed to stop consumer for region {region!r}")
        self._consumers = {}

    async def poll(self, num: int) -> list[KafkaMessage]:
        if not self._consumers:
            raise RuntimeError("Consumer not started")
        Metrics.counter("kafka_consumer.fetching.count").add(
            num, attributes={"group_id": self.group_id}
        )
        start = time.perf_counter()
        consumers = list(self._consumers.items())
        max_records = max(1, num // len(consumers))
        results = await asyncio.gather(
            *[
                consumer.getmany(timeout_ms=1000, max_records=max_records)
                for _, consumer in consumers
            ],
            return_exceptions=True,
        )
        duration = time.perf_counter() - start
        current_time = int(time.time())
        Metrics.histogram("kafka_consumer.fetch_duration").record(
            duration, attributes={"group_id": self.group_id}
        )

        fanin: list[KafkaMessage] = []
        assigned_total = 0
        errors: list[BaseException] = []
        for (region, consumer), messages in zip(consumers, results):
            attributes = {"group_id": self.group_id, "region": region}
            if isinstance(messages, BaseException):
                errors.append(messages)
                logger.error(f"Poll failed on region {region!r}: {messages!r}")
                Metrics.counter("kafka_consumer.poll_errors.count").add(
                    1, attributes=attributes
                )
                continue
            num_fetched = sum(len(msg_list) for msg_list in messages.values())
            Metrics.counter("kafka_consumer.fetched.count").add(
                num_fetched, attributes=attributes
            )
            for tp, msg_list in messages.items():
                for msg in msg_list:
                    timestamp_type = (
                        "creation_time"
                        if msg.timestamp_type == 0
                        else "log_append_time"
                    )
                    Metrics.histogram(
                        f"kafka_consumer.fetch_delay.{timestamp_type}"
                    ).record(
                        current_time - msg.timestamp // 1000, attributes=attributes
                    )
                    fanin.append(
                        KafkaMessage(
                            tp,
                            offset=msg.offset,
                            value=msg.value,
                            key=msg.key,
                            timestamp=msg.timestamp,
                        )
                    )
            assigned_partitions = consumer.assignment()
            assigned_total += len(assigned_partitions) if assigned_partitions else 0
        if errors and len(errors) == len(consumers):
            raise errors[0]
        Metrics.gauge("kafka_consumer.assigned_partitions").set(
            assigned_total, attributes={"group_id": self.group_id}
        )
        logger.debug(f"Polled {len(fanin)} messages in {duration:.2f} seconds")
        return fanin
