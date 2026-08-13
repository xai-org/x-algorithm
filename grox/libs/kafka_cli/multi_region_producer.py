import asyncio
import logging
import random
import time

from aiokafka import AIOKafkaProducer
from monitor.metrics import Metrics
from pydantic import BaseModel, Field, model_validator

from kafka_cli.mtls import create_mtls_ssl_context

logger = logging.getLogger(__name__)

RETRY_FAILED_REGION_INTERVAL_SEC = 60


class MultiRegionKafkaProducerConfig(BaseModel):
    topic: str
    clusters: dict[str, list[str]]
    acks: int | str = Field(default=1)
    request_timeout_ms: int = Field(default=30000)

    @model_validator(mode="after")
    def _validate(self) -> "MultiRegionKafkaProducerConfig":
        if not self.clusters:
            raise ValueError("`clusters` must contain at least one region")
        for region, brokers in self.clusters.items():
            if not brokers:
                raise ValueError(f"Region {region!r} must list at least one broker")
        if self.acks not in (0, 1, "all"):
            raise ValueError(f"acks must be 0, 1, or 'all', got {self.acks!r}")
        return self


class MultiRegionKafkaProducer:
    def __init__(self, config: MultiRegionKafkaProducerConfig):
        self.config = config
        self.topic: str = config.topic
        self._producers: dict[str, AIOKafkaProducer] = {}
        self._region_retry_task: asyncio.Task | None = None

    async def start(self):
        regions = list(self.config.clusters.items())
        try:
            results = await asyncio.gather(
                *[self._start_region(region, brokers) for region, brokers in regions],
                return_exceptions=True,
            )
        except BaseException:
            await self.stop()
            raise
        failed: dict[str, list[str]] = {}
        for (region, brokers), result in zip(regions, results):
            if isinstance(result, BaseException):
                logger.error(
                    f"Failed to start producer for region {region!r}: {result!r}"
                )
                Metrics.counter("kafka_producer.start_errors.count").add(
                    1, attributes={"topic": self.topic, "region": region}
                )
                failed[region] = brokers
        if not self._producers:
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
                    await self._start_region(region, failed[region])
                    del failed[region]
                    logger.info(f"Producer region {region!r} recovered")
                except Exception as e:
                    logger.error(f"Producer retry failed for region {region!r}: {e!r}")
                    Metrics.counter("kafka_producer.start_errors.count").add(
                        1, attributes={"topic": self.topic, "region": region}
                    )

    async def _start_region(self, region: str, brokers: list[str]) -> None:
        producer = AIOKafkaProducer(
            bootstrap_servers=",".join(brokers),
            security_protocol="SSL",
            ssl_context=create_mtls_ssl_context(),
            acks=self.config.acks,
            request_timeout_ms=self.config.request_timeout_ms,
        )
        try:
            await producer.start()
        except BaseException:
            try:
                await producer.stop()
            except Exception:
                logger.exception(
                    f"Failed to clean up producer after failed start for region {region!r}"
                )
            raise
        self._producers[region] = producer
        logger.info(
            f"MultiRegionKafkaProducer started region {region!r}: topic={self.topic}"
        )

    async def stop(self):
        if self._region_retry_task is not None:
            self._region_retry_task.cancel()
            try:
                await self._region_retry_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Producer region retry task failed during shutdown")
            self._region_retry_task = None
        for region, producer in self._producers.items():
            try:
                await producer.stop()
            except Exception:
                logger.exception(f"Failed to stop producer for region {region!r}")
        self._producers = {}

    def _candidate_regions(self) -> list[str]:
        candidates = [
            region for region in self.config.clusters if region in self._producers
        ]
        random.shuffle(candidates)
        return candidates

    async def send(self, id: str, value: bytes):
        if not self._producers:
            raise RuntimeError("Producer not started")
        candidates = self._candidate_regions()
        if not candidates:
            raise RuntimeError(
                f"No healthy Kafka region available for topic {self.topic!r}"
            )
        start = time.perf_counter()
        errors: list[BaseException] = []
        for region in candidates:
            attributes = {"topic": self.topic, "region": region}
            try:
                await self._producers[region].send_and_wait(
                    self.topic, key=id.encode(), value=value
                )
                Metrics.counter("kafka_producer.sent.count").add(
                    1, attributes=attributes
                )
                Metrics.histogram("kafka_producer.send_duration").record(
                    time.perf_counter() - start, attributes=attributes
                )
                return
            except Exception as e:
                errors.append(e)
                logger.error(
                    f"Send failed for id={id} topic={self.topic} region={region!r}: {e!r}"
                )
                Metrics.counter("kafka_producer.send_errors.count").add(
                    1, attributes=attributes
                )
        raise errors[-1]
