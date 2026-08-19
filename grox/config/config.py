import logging
import os
from pathlib import Path

from embed.embed_http import EmbeddingModelConfig
from grok_sampler.config import OaiModelConfig
from grox.config.env import grox_env
from kafka_cli.config import KafkaConsumerConfig, KafkaProducerConfig
from kafka_cli.multi_region_consumer import MultiRegionKafkaConsumerConfig
from kafka_cli.multi_region_producer import MultiRegionKafkaProducerConfig
from monitor.config import LoggingConfig, MetricsConfig
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

_FROZEN = ConfigDict(frozen=True)


def _get_config_file_paths() -> list[str]:
    base_path = Path(__file__).parent / "yaml"
    base_config = base_path / "config.base.yaml"
    variant_config = base_path / f"config.{grox_env.value}.yaml"
    if not variant_config.exists():
        raise FileNotFoundError(f"Config file not found: {variant_config}")
    paths: list[str] = []
    if base_config.exists():
        paths.append(base_config.as_posix())
        logger.info(f"Using base config file: {base_config}")
    paths.append(variant_config.as_posix())
    logger.info(f"Using variant config file: {variant_config}")
    return paths


def _read_bearer_token() -> str:
    try:
        return (
            Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
            .read_text()
            .strip()
        )
    except (FileNotFoundError, PermissionError):
        return ""


def _read_xai_api_key() -> str:
    return os.getenv("XAI_API_KEY") or ""


class ModelConfig(BaseModel):
    model_config = _FROZEN

    model_name: str
    endpoint: str
    timeout: int = 60
    max_resp_len: int = 8092
    temperature: float = 0.7
    bearer_token: str = Field(default_factory=_read_bearer_token)
    priority_header: str = ""
    priority: int = 15


class EapiModelConfig(BaseModel):
    model_config = _FROZEN

    api_key: str = Field(default_factory=_read_xai_api_key)
    model: str
    timeout: int = 300
    temperature: float = 0.0001
    priority: int = 15
    api_host: str = ""


class ModelName:
    GROK_4_MINI = "grok-4-mini"
    GROK_4_1_FAST_HEDGEHOG = "grok-4-1-fast-hedgehog"
    GROK_4_1_FAST_HEDGEHOG_CRITICAL = "grok-4-1-fast-hedgehog-critical"
    GROK_X_ALGO_EXTRA = "grok-x-algo-extra"
    GROK_4_1_FAST_X_ALGO_EXTRA = "grok-4-1-fast-x-algo-extra"
    GROK_4_MINI_CRITICAL = "grok-4-mini-critical"
    GROK_4_MINI_CRITICAL_SAFETY = "critical-safety"
    EAPI_GROK_420_REASONING_X_ALGO = "eapi-grok-420-reasoning-x-algo"
    EAPI_GROK_420_REASONING_INTERNAL = "eapi-grok-420-reasoning-internal"
    EAPI_GROK_4_3_INTERNAL = "eapi-grok-4-3-internal"
    EAPI_GROK_4_3_X_ALGO = "eapi-grok-4-3-x-algo"
    EAPI_GROK_4_5_INTERNAL = "eapi-grok-4-5-internal"
    EAPI_GROK_4_5_X_ALGO = "eapi-grok-4-5-x-algo"


class NightOwlConfig(BaseModel):
    model_config = _FROZEN

    endpoint: str = "localhost:50051"
    timeout: float = 10.0


class MediaHydrationConfig(BaseModel):
    model_config = _FROZEN

    max_workers: int = 4
    max_post_count: int = 9
    video_max_frames: int = 20
    video_max_frames_light: int = 2
    video_tile_size: int = 448
    image_tile_size: int = 448
    enable_light_dark_enhancement: bool = False
    enable_clahe_enhancement: bool = False
    deluxe_fav_count_threshold: int = 64
    deluxe_video_max_frames: int = 30
    deluxe_video_tile_size: int = 600
    deluxe_image_tile_size: int = 600


class GroxKafkaLoaderConfig(BaseModel):
    model_config = _FROZEN

    prefetching_threshold: int = 256
    prefetching_batch_size: int = 1024
    max_qps_per_partition: int | None = None


class GrpcServerConfig(BaseModel):
    model_config = _FROZEN

    port: int = 9988
    graceful_shutdown_timeout: float = 120


class EngineConfig(BaseModel):
    model_config = _FROZEN

    graceful_shutdown_timeout: float = 120
    task_timeout: float = 900


class TaskGeneratorConfig(BaseModel):
    model_config = _FROZEN

    type: str
    weight: int
    max_qps: int | None = None


class DispatcherConfig(BaseModel):
    model_config = _FROZEN

    graceful_shutdown_timeout: float = 120
    max_in_flight: int = 3
    task_generators: list[TaskGeneratorConfig] = []


class KerberosConfig(BaseModel):
    model_config = _FROZEN

    keytab_path: str | None = None
    principal: str | None = None


class ASRConfig(BaseModel):
    model_config = _FROZEN

    endpoint: str = ""
    max_tokens: int = 8192
    max_audio_duration_s: float = 600.0
    timeout: float = 300.0
    temperature: float = 0.0
    max_workers: int = 0


class ProductMediaConfig(BaseModel):
    model_config = _FROZEN

    max_workers: int = 0


class PeriodicGcConfig(BaseModel):
    model_config = _FROZEN

    interval: float = 28800
    jitter: float = 900


class PromptTokensConfig(BaseModel):
    model_config = _FROZEN

    thinking_control_start: str = ""
    thinking_control_end: str = ""
    no_thinking_prompt: str = ""
    separator: str = ""
    image_pad: str = ""


class GroxConfig(BaseSettings):
    metrics: MetricsConfig = MetricsConfig()
    logging: LoggingConfig = LoggingConfig()
    periodic_gc: PeriodicGcConfig = PeriodicGcConfig()
    shutdown_drain_timeout: float = 300
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    embedding_models: dict[str, EmbeddingModelConfig] = Field(default_factory=dict)
    eapi_models: dict[str, EapiModelConfig] = Field(default_factory=dict)
    oai_models: dict[str, OaiModelConfig] = Field(default_factory=dict)
    media_hydration: MediaHydrationConfig = MediaHydrationConfig()
    product_media: ProductMediaConfig = ProductMediaConfig()
    nightowl: NightOwlConfig | None = NightOwlConfig()
    kafka_producer_topics: dict[str, KafkaProducerConfig] = Field(default_factory=dict)
    kafka_consumer_topics: dict[str, KafkaConsumerConfig] = Field(default_factory=dict)
    kafka_consumers: dict[str, MultiRegionKafkaConsumerConfig] = Field(
        default_factory=dict
    )
    kafka_producers: dict[str, MultiRegionKafkaProducerConfig] = Field(
        default_factory=dict
    )
    kafka_loader_topics: dict[str, GroxKafkaLoaderConfig] = Field(default_factory=dict)
    grpc_server: GrpcServerConfig = GrpcServerConfig()
    engine: EngineConfig = EngineConfig()
    dispatcher: DispatcherConfig = DispatcherConfig()
    kerberos: KerberosConfig = KerberosConfig()
    asr: ASRConfig = ASRConfig()
    prompt_tokens: PromptTokensConfig = PromptTokensConfig()

    model_config = SettingsConfigDict(yaml_file=_get_config_file_paths(), frozen=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls, deep_merge=True),
        )

    def get_model(self, model_name: str) -> ModelConfig:
        return self.models[model_name]

    def get_embedding_model(self, model_name: str) -> EmbeddingModelConfig:
        return self.embedding_models[model_name].model_copy(deep=True)

    def get_eapi_model(self, model_name: str) -> EapiModelConfig:
        return self.eapi_models[model_name]

    def get_oai_model(self, model_name: str) -> OaiModelConfig:
        return self.oai_models[model_name].model_copy(deep=True)

    def get_kafka_producer_topic(self, topic_name: str) -> KafkaProducerConfig:
        return self.kafka_producer_topics[topic_name].model_copy(deep=True)

    def get_kafka_producer(self, name: str) -> MultiRegionKafkaProducerConfig:
        return self.kafka_producers[name].model_copy(deep=True)

    def get_kafka_consumer_topic(self, topic_name: str) -> KafkaConsumerConfig:
        return self.kafka_consumer_topics[topic_name].model_copy(deep=True)

    def get_kafka_loader_topic(self, topic_name: str) -> GroxKafkaLoaderConfig:
        return self.kafka_loader_topics[topic_name]


grox_config = GroxConfig()
