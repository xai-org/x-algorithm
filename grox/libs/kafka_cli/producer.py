import asyncio
import logging
import os
import ssl
import traceback
from datetime import datetime

from aiokafka import AIOKafkaProducer
from wily_cli.config import WilyConfig
from wily_cli.wily_client import WilyNs

from kafka_cli.config import KafkaProducerConfig

logger = logging.getLogger(__name__)


def _create_mtls_ssl_context() -> ssl.SSLContext:
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


class _KafkaProducerClient:
    _cached_clients: dict[str, AIOKafkaProducer] = {}

    @classmethod
    async def _build_client(cls, config: KafkaProducerConfig) -> AIOKafkaProducer:
        if isinstance(config.brokers, WilyConfig):
            wily = WilyNs(config.brokers)
            instances = await wily.resolve(config.dest)
            brokers = ",".join(
                f"{instance.address}:{instance.port}" for instance in instances
            )
        else:
            brokers = ",".join(config.brokers)
        ssl_conf = config.ssl
        if ssl_conf is not None:
            proto = (ssl_conf.security_protocol or "").upper()
            if proto == "SSL":
                ssl_context = _create_mtls_ssl_context()
                producer = AIOKafkaProducer(
                    bootstrap_servers=brokers,
                    security_protocol="SSL",
                    ssl_context=ssl_context,
                    request_timeout_ms=config.timeout_sec * 1000,
                )
            else:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                producer = AIOKafkaProducer(
                    bootstrap_servers=brokers,
                    sasl_kerberos_service_name=ssl_conf.sasl_kerberos_service_name,
                    sasl_kerberos_domain_name=ssl_conf.sasl_kerberos_domain_name,
                    sasl_mechanism=ssl_conf.sasl_mechanism,
                    security_protocol=ssl_conf.security_protocol,
                    ssl_context=ssl_context,
                    request_timeout_ms=config.timeout_sec * 1000,
                )
        else:
            producer = AIOKafkaProducer(
                bootstrap_servers=brokers,
                security_protocol="PLAINTEXT",
                request_timeout_ms=config.timeout_sec * 1000,
            )
        await producer.start()
        return producer

    @classmethod
    async def get(cls, config: KafkaProducerConfig) -> AIOKafkaProducer:
        loop_id = id(asyncio.get_running_loop())
        key = str(loop_id) + "_" + config.topic + "_" + config.dest
        if key not in cls._cached_clients:
            cls._cached_clients[key] = await cls._build_client(config)
        return cls._cached_clients[key]


class KafkaProducer:
    config: KafkaProducerConfig
    _cli: AIOKafkaProducer | None = None

    def __init__(self, config: KafkaProducerConfig):
        logger.info(f"Initializing kafka producer {config.topic=}, {config.dest=}")
        self.config = config

    async def send(self, id: str, value: bytes):
        if self._cli is None:
            self._cli = await _KafkaProducerClient.get(self.config)
        try:
            await self._cli.send_and_wait(
                self.config.topic,
                key=id.encode(),
                value=value,
            )
            logger.info(f"Sent message for id: {id}")
        except Exception:
            logger.error(
                f"Error sending message for id: {id}, error: {traceback.format_exc()}"
            )
            raise


class _ScramKafkaProducerClient:
    _cached_clients: dict[str, AIOKafkaProducer] = {}

    @classmethod
    async def _build_client(cls, config: KafkaProducerConfig) -> AIOKafkaProducer:
        if isinstance(config.brokers, WilyConfig):
            wily = WilyNs(config.brokers)
            instances = await wily.resolve(config.dest)
            brokers = ",".join(
                f"{instance.address}:{instance.port}" for instance in instances
            )
        else:
            brokers = ",".join(config.brokers)
        ssl_conf = config.ssl
        if ssl_conf is None:
            raise ValueError("ScramKafkaProducer requires an ssl config")

        proto = (ssl_conf.security_protocol or "").upper()
        if proto == "SSL":
            ssl_context = _create_mtls_ssl_context()
            producer = AIOKafkaProducer(
                bootstrap_servers=brokers,
                security_protocol="SSL",
                ssl_context=ssl_context,
                request_timeout_ms=config.timeout_sec * 1000,
            )
        else:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            producer = AIOKafkaProducer(
                bootstrap_servers=brokers,
                sasl_mechanism=ssl_conf.sasl_mechanism,
                security_protocol=ssl_conf.security_protocol,
                ssl_context=ssl_context,
                request_timeout_ms=config.timeout_sec * 1000,
                sasl_plain_username=ssl_conf.sasl_plain_username,
                sasl_plain_password=ssl_conf.sasl_plain_password,
            )
        await producer.start()
        return producer

    @classmethod
    async def get(cls, config: KafkaProducerConfig) -> AIOKafkaProducer:
        loop_id = id(asyncio.get_running_loop())
        key = str(loop_id) + "_" + config.topic + "_" + config.dest
        if key not in cls._cached_clients:
            cls._cached_clients[key] = await cls._build_client(config)
        return cls._cached_clients[key]


class ScramKafkaProducer:
    config: KafkaProducerConfig
    _cli: AIOKafkaProducer | None = None

    def __init__(self, config: KafkaProducerConfig):
        logger.info(f"Initializing kafka producer {config.topic=}, {config.dest=}")
        self.config = config

    async def send(self, id: str, value: bytes):
        if self._cli is None:
            self._cli = await _ScramKafkaProducerClient.get(self.config)
        try:
            await self._cli.send_and_wait(
                self.config.topic,
                key=id.encode(),
                value=value,
            )
            logger.info(f"Sent message for id: {id}")
        except Exception:
            logger.error(
                f"Error sending message for id: {id}, error: {traceback.format_exc()}"
            )
            raise
