import logging
from asyncio import get_running_loop
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path

import grpc
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GrpcCliConfig(BaseModel):
    ca_root_dir: str = "/etc/certs/grpc-root-cas"
    default_ca_name: str = "prod.internal.x.ai"
    keepalive_timeout_ms: int = 12000
    keepalive_period_ms: int = 20000
    min_time_between_pings_ms: int = 10000
    max_message_length: int = 128 * 1024 * 1024
    client_cert_path: str | None = None
    client_key_path: str | None = None


def get_loop_id() -> int:
    return id(get_running_loop())


def maybe_add_port(base_url: str, secured: bool) -> str:
    if ":" in base_url:
        return base_url
    default_port = 443 if secured else 9988
    return f"{base_url}:{default_port}"


@asynccontextmanager
async def aclosing_grpc(call):
    try:
        yield call
    finally:
        call.cancel()


class GrpcClient:
    _cached_clients: dict[tuple[int, str, bool], grpc.aio.Channel] = {}

    def __init__(self, **kwargs):
        self.config = GrpcCliConfig(**kwargs)

    def make(self, base_url: str) -> grpc.aio.Channel:
        base_url = base_url.strip().lower()
        if base_url.startswith("grpcs://") or base_url.startswith("https://"):
            secured = True
            base_url = base_url.replace("grpcs://", "").replace("https://", "")
        else:
            secured = False
            base_url = base_url.replace("grpc://", "").replace("http://", "")
        base_url = maybe_add_port(base_url, secured)
        return self._get_client(base_url, secured)

    def _get_client(self, base_url: str, secured: bool) -> grpc.aio.Channel:
        loop_id = get_loop_id()
        key = (loop_id, base_url, secured)

        if key in GrpcClient._cached_clients:
            channel = GrpcClient._cached_clients[key]
            try:
                state = channel.get_state(try_to_connect=False)
                if state in [
                    grpc.ChannelConnectivity.TRANSIENT_FAILURE,
                    grpc.ChannelConnectivity.SHUTDOWN,
                ]:
                    logger.warning(
                        f"Removing stale channel for {base_url}, state={state}, loop_id={loop_id}"
                    )
                    del GrpcClient._cached_clients[key]
                elif state == grpc.ChannelConnectivity.IDLE:
                    logger.info(
                        f"Channel for {base_url} is IDLE, will reconnect on use, loop_id={loop_id}"
                    )
                else:
                    logger.debug(
                        f"Reusing cached channel for {base_url}, state={state}, loop_id={loop_id}"
                    )
            except Exception as e:
                logger.error(
                    f"Error checking channel state for {base_url}: {e}, removing from cache"
                )
                del GrpcClient._cached_clients[key]

        if key not in GrpcClient._cached_clients:
            logger.info(
                f"Creating {'secured ' if secured else ''}client with loop_id={loop_id}, base_url={base_url}"
            )
            options = [
                ("grpc.max_send_message_length", self.config.max_message_length),
                ("grpc.max_receive_message_length", self.config.max_message_length),
                ("grpc.keepalive_timeout_ms", self.config.keepalive_timeout_ms),
                ("grpc.keepalive_time_ms", self.config.keepalive_period_ms),
                ("grpc.keepalive_permit_without_calls", 1),
                (
                    "grpc.http2.min_time_between_pings_ms",
                    self.config.min_time_between_pings_ms,
                ),
            ]

            if secured:
                host_name = base_url.split(":")[0]
                credentials = self._get_credentials(host_name)
                channel = grpc.aio.secure_channel(
                    base_url,
                    credentials,
                    options=options,
                )
            else:
                channel = grpc.aio.insecure_channel(
                    base_url,
                    options=options,
                )
            GrpcClient._cached_clients[key] = channel
            logger.info(
                f"Created and cached new channel for {base_url}, loop_id={loop_id}"
            )
        return GrpcClient._cached_clients[key]

    def _get_best_ca_file(self, host_name: str) -> Path:
        ca_root_dir = Path(self.config.ca_root_dir)
        if (ca_root_dir / host_name).is_file():
            return ca_root_dir / host_name
        return ca_root_dir / self.config.default_ca_name

    @cache
    def _get_credentials(self, host_name: str) -> grpc.ChannelCredentials:
        ca_file = self._get_best_ca_file(host_name)
        with open(ca_file, "rb") as f:
            root_ca = f.read()
        client_cert = None
        client_key = None
        if self.config.client_cert_path and self.config.client_key_path:
            with open(self.config.client_cert_path, "rb") as f:
                client_cert = f.read()
            with open(self.config.client_key_path, "rb") as f:
                client_key = f.read()
            return grpc.ssl_channel_credentials(
                root_certificates=root_ca,
                private_key=client_key,
                certificate_chain=client_cert,
            )
        return grpc.ssl_channel_credentials(
            root_certificates=root_ca,
        )
