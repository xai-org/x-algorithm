from dataclasses import dataclass
from typing import Any

from aiokafka.structs import TopicPartition
from pydantic import BaseModel, Field
from wily_cli.config import WilyConfig


@dataclass
class KafkaMessage:
    tp: TopicPartition
    offset: int
    value: Any | None
    key: Any | None = None
    retry_count: int = 0
    timestamp: int | None = None


class KafkaException(Exception):
    pass


class NonRetryableKafkaException(KafkaException):
    pass


class RetryableKafkaException(KafkaException):
    pass


class KafkaSslConfig(BaseModel):
    sasl_kerberos_service_name: str = Field(default="kafka")
    sasl_kerberos_domain_name: str = Field(default="kafka")
    sasl_mechanism: str = Field(default="GSSAPI")
    security_protocol: str = Field(default="SASL_SSL")
    sasl_plain_username: str | None = None
    sasl_plain_password: str | None = None


class KafkaConfig(BaseModel):
    dest: str
    topic: str
    timeout_sec: int = Field(default=10)
    brokers: list[str] | WilyConfig = Field(default_factory=list)
    ssl: KafkaSslConfig | None = None


class KafkaProducerConfig(KafkaConfig):
    pass


class KafkaConsumerConfig(KafkaConfig):
    group_id: str
    auto_commit: bool = Field(default=True)
    auto_offset_reset: str = Field(default="earliest")
    commit_timeout_sec: int = Field(default=10)
    fetch_max_bytes: int = Field(default=50 * 1024 * 1024)
    fetch_min_bytes: int = Field(default=1024 * 128)
    max_poll_records: int = Field(default=500)
    prefetching_threshold: int = 256
    request_timeout_ms: int = Field(default=30000)
