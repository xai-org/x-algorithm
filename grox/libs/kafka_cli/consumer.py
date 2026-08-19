import logging
import os
import ssl
import time
from datetime import datetime

from aiokafka import AIOKafkaConsumer
from aiokafka.coordinator.assignors.sticky.sticky_assignor import (
    StickyPartitionAssignor,
)
from aiokafka.structs import TopicPartition
from monitor.metrics import Metrics
from tenacity import retry, stop_after_attempt, wait_incrementing
from wily_cli.config import WilyConfig
from wily_cli.wily_client import WilyNs

from kafka_cli.config import KafkaConsumerConfig, KafkaMessage

logger = logging.getLogger(__name__)


class KafkaConsumer:
    def __init__(self, config: KafkaConsumerConfig):
        self.config: KafkaConsumerConfig = config
        self.group_id: str = config.group_id
        self._consumer: AIOKafkaConsumer | None = None

    @property
    def assigned_partitions(self) -> int:
        if self._consumer is None:
            return 0
        return len(self._consumer.assignment() or ())

    def _auth_mode(self) -> str:
        if not self.config.ssl:
            return "PLAINTEXT"

        proto = (self.config.ssl.security_protocol or "").upper()
        if proto == "SSL":
            return "MTLS"
        if "SASL" in proto:
            mech = (self.config.ssl.sasl_mechanism or "").upper()
            return "SCRAM" if "SCRAM" in mech else "GSSAPI"
        return "PLAINTEXT"

    def _create_ssl_context(self) -> ssl.SSLContext:
        ssl_ctx = ssl.create_default_context()
        ca_file = os.getenv("SSL_CA_FILE", "/etc/ssl/internal-ca/ca-bundle.crt")

        if ca_file and os.path.exists(ca_file):
            ssl_ctx.load_verify_locations(ca_file)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            logger.info(
                f"[{datetime.now()}] SSL verification disabled (CERT_NONE) with CA loaded: {ca_file}"
            )
        else:
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            logger.info(f"[{datetime.now()}] SSL verification disabled")

        cert_file = os.getenv("SSL_CERT_FILE", "/etc/ssl/s2s/client/tls.crt")
        key_file = os.getenv("SSL_KEY_FILE", "/etc/ssl/s2s/client/tls.key")

        if not os.path.exists(cert_file) and os.path.exists("/certs/client.fullchain"):
            cert_file = "/certs/client.fullchain"
        if not os.path.exists(key_file) and os.path.exists("/certs/client.key"):
            key_file = "/certs/client.key"

        logger.info(f"[{datetime.now()}] Using CERT={cert_file} KEY={key_file}")

        if os.path.exists(cert_file) and os.path.exists(key_file):
            ssl_ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
            logger.info(f"[{datetime.now()}] mTLS client certificate loaded")
        else:
            logger.warning(f"[{datetime.now()}] WARNING: client cert files not found")

        return ssl_ctx

    @retry(
        stop=stop_after_attempt(3), wait=wait_incrementing(start=1, increment=3, max=9)
    )
    async def start(self):
        config = self.config

        if isinstance(config.brokers, WilyConfig):
            wily = WilyNs(config.brokers)
            instances = await wily.resolve(config.dest)
            brokers = ",".join(
                f"{instance.address}:{instance.port}" for instance in instances
            )
        else:
            brokers = ",".join(config.brokers)

        auth_mode = self._auth_mode()

        ssl_context = None
        if auth_mode == "MTLS":
            ssl_context = self._create_ssl_context()
            consumer = AIOKafkaConsumer(
                config.topic,
                bootstrap_servers=brokers,
                group_id=config.group_id,
                auto_offset_reset=config.auto_offset_reset,
                enable_auto_commit=config.auto_commit,
                auto_commit_interval_ms=5000,
                security_protocol="SSL",
                ssl_context=ssl_context,
                request_timeout_ms=config.request_timeout_ms,
                fetch_max_bytes=config.fetch_max_bytes,
                fetch_min_bytes=config.fetch_min_bytes,
                max_poll_records=config.max_poll_records,
                partition_assignment_strategy=[StickyPartitionAssignor],
            )
        elif auth_mode in ("GSSAPI", "SCRAM"):
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            consumer = AIOKafkaConsumer(
                config.topic,
                bootstrap_servers=brokers,
                group_id=config.group_id,
                auto_offset_reset=config.auto_offset_reset,
                enable_auto_commit=config.auto_commit,
                auto_commit_interval_ms=5000,
                security_protocol=config.ssl.security_protocol,
                sasl_kerberos_domain_name=config.ssl.sasl_kerberos_domain_name,
                sasl_kerberos_service_name=config.ssl.sasl_kerberos_service_name,
                sasl_mechanism=config.ssl.sasl_mechanism,
                ssl_context=ssl_context,
                sasl_plain_username=config.ssl.sasl_plain_username,
                sasl_plain_password=config.ssl.sasl_plain_password,
                partition_assignment_strategy=[StickyPartitionAssignor],
                fetch_max_bytes=config.fetch_max_bytes,
                fetch_min_bytes=config.fetch_min_bytes,
                max_poll_records=config.max_poll_records,
            )
        else:
            consumer = AIOKafkaConsumer(
                config.topic,
                bootstrap_servers=brokers,
                group_id=config.group_id,
                auto_offset_reset=config.auto_offset_reset,
                enable_auto_commit=config.auto_commit,
                auto_commit_interval_ms=5000,
                security_protocol="PLAINTEXT",
                partition_assignment_strategy=[StickyPartitionAssignor],
                fetch_max_bytes=config.fetch_max_bytes,
                fetch_min_bytes=config.fetch_min_bytes,
                max_poll_records=config.max_poll_records,
                request_timeout_ms=config.request_timeout_ms,
            )
        await consumer.start()
        self._consumer = consumer

        past_committed: dict[TopicPartition, int] = {}
        for tp in consumer.assignment():
            offset = await consumer.committed(tp)
            past_committed[tp] = offset or 0
        logger.info(
            f"Consumer started listening to {len(past_committed)} partitions topic={config.topic}"
        )

    async def seek_to_committed(self):
        return await self.consumer.seek_to_committed()

    async def get_lag(self) -> dict[int, int]:
        try:
            lag_dict = {}
            assigned = self.consumer.assignment()

            for tp in assigned:
                highwater = self.consumer.highwater(tp)
                committed = await self.consumer.committed(tp)

                if highwater is not None and committed is not None:
                    lag = highwater - committed
                    if lag > 0:
                        lag_dict[tp.partition] = lag
                elif highwater is not None and committed is None:
                    if highwater > 0:
                        lag_dict[tp.partition] = highwater

            return lag_dict
        except Exception as e:
            logger.warning(f"Failed to get lag: {e}")
            return {}

    async def stop(self):
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

    async def poll(self, num: int) -> list[KafkaMessage]:
        logger.debug(f"Polling {num} messages")
        Metrics.counter("kafka_consumer.fetching.count").add(
            num, attributes={"group_id": self.group_id}
        )
        start = time.perf_counter()
        messages = await self.consumer.getmany(timeout_ms=1000, max_records=num)
        num_fetched = sum(len(msg_list) for msg_list in messages.values())
        Metrics.counter("kafka_consumer.fetched.count").add(
            num_fetched, attributes={"group_id": self.group_id}
        )
        end = time.perf_counter()
        current_time = int(time.time())
        duration = end - start
        Metrics.histogram("kafka_consumer.fetch_duration").record(
            duration, attributes={"group_id": self.group_id}
        )
        for _, msg_list in messages.items():
            for msg in msg_list:
                timestamp_type = (
                    "creation_time" if msg.timestamp_type == 0 else "log_append_time"
                )
                Metrics.histogram(
                    f"kafka_consumer.fetch_delay.{timestamp_type}"
                ).record(
                    current_time - msg.timestamp // 1000,
                    attributes={"group_id": self.group_id},
                )
        logger.debug(f"Polled {num_fetched} messages in {duration:.2f} seconds")
        assigned_partitions = self.consumer.assignment()
        Metrics.gauge("kafka_consumer.assigned_partitions").set(
            len(assigned_partitions) if assigned_partitions else 0,
            attributes={"group_id": self.group_id},
        )
        return [
            KafkaMessage(
                tp,
                offset=msg.offset,
                value=msg.value,
                key=msg.key,
                timestamp=msg.timestamp,
            )
            for tp, msg_list in messages.items()
            for msg in msg_list
        ]

    async def commit(self, offsets: dict[TopicPartition, tuple[int, str]]):
        await self.consumer.commit(offsets)

    @property
    def consumer(self) -> AIOKafkaConsumer:
        assert self._consumer is not None
        return self._consumer
