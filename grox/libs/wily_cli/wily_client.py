import asyncio
import logging
import random
import socket
from dataclasses import dataclass
from time import time
from typing import Any
from urllib.parse import quote

import httpx

from wily_cli.config import WilyConfig

logger = logging.getLogger(__name__)

DEFAULT_ZONE = "atla"
WILYNS_ENDPOINTS = {
    "atla": "wilyns-atla.twitter.biz",
    "pdxa": "wilyns-pdxa.twitter.biz",
}


@dataclass
class Instance:
    address: str
    hostname: str
    port: int
    weight: float = 1.0
    shard_id: str | None = None

    def __str__(self):
        return f"WilyNS.Instance({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"


class _WilyClient:
    _cached_clients: dict[int, httpx.AsyncClient] = {}

    @classmethod
    def _build_client(cls, config: WilyConfig) -> httpx.AsyncClient:
        logger.info(f"Building wily client with config: {config.model_dump()}")
        limits = httpx.Limits(
            max_connections=10,
            max_keepalive_connections=2,
            keepalive_expiry=10,
        )
        timeout = httpx.Timeout(config.timeout)
        return httpx.AsyncClient(
            verify=config.verify,
            limits=limits,
            timeout=timeout,
        )

    @classmethod
    def get(cls, config: WilyConfig) -> httpx.AsyncClient:
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in cls._cached_clients:
            cls._cached_clients[loop_id] = cls._build_client(config)
        return cls._cached_clients[loop_id]


class WilyNs:
    class ClientError(Exception):
        pass

    def __init__(self, wily_config: WilyConfig, **kwargs: Any) -> None:
        self.config = wily_config
        self.endpoint = WILYNS_ENDPOINTS[self.config.zone]
        self.context = self._process_context(wily_config)

    @property
    def client(self) -> httpx.AsyncClient:
        return _WilyClient.get(self.config)

    def _ensure_startswith(self, s: str, prefix: str) -> str:
        return f"{prefix}{s}" if s and not s.startswith(prefix) else s

    def _entry_to_instance(self, entry: dict[str, Any]) -> Instance:
        return Instance(
            address=entry["addr"],
            hostname=entry["hostname"],
            port=entry["port"],
            weight=entry.get("weight", 1.0),
            shard_id=entry.get("zkMetadata", {}).get("shardId"),
        )

    def _process_context(self, wily_config: WilyConfig) -> str:
        return f"/p/static/{wily_config.zone}/{wily_config.role}/{wily_config.client_name}/{socket.gethostname()}/{int(time())}"

    def _make_url(
        self, path: str, dtab: str = "", endpoint: str | None = None
    ) -> str:
        final_endpoint = endpoint or self.endpoint
        path = self._ensure_startswith(self._ensure_startswith(path, "/"), "/lookups")
        return f"https://{final_endpoint}{quote(path)}?context={self.context}&dtab={quote(dtab)}"

    async def _resolve(
        self, wily_path: str, dtab: str = "", endpoint: str | None = None
    ) -> list[Instance]:
        if self.config.jitter:
            await asyncio.sleep(random.random() * self.config.jitter)

        response = await self.client.get(self._make_url(wily_path, dtab, endpoint))
        response.raise_for_status()
        return [
            self._entry_to_instance(entry)
            for entry in response.json().get("entries", [])
        ]

    async def resolve(self, wily_path: str, dtab: str = "") -> list[Instance]:
        if self.config.allow_alternative_zones:
            other_endpoints = list(set(WILYNS_ENDPOINTS.values()) - {self.endpoint})
            ordered_endpoints = [self.endpoint] + other_endpoints
            logger.debug(f"Possible endpoints: {ordered_endpoints}")

            for endpoint in ordered_endpoints:
                try:
                    logger.debug(f"Attempting to resolve against {endpoint}")
                    return await self._resolve(wily_path, dtab, endpoint)
                except httpx.RequestError:
                    logger.debug(
                        f"Could not connect to {endpoint}. Trying the next one."
                    )
        else:
            return await self._resolve(wily_path, dtab, self.endpoint)

        raise self.ClientError("Could not reach any wilyns endpoints")

    async def close(self):
        await self.client.aclose()
