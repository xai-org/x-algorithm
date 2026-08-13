import asyncio
import logging
import traceback
from abc import ABC
from enum import Enum
from urllib.parse import quote

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ContentType(Enum):
    TEXT_PLAIN = "text/plain"
    TEXT_HTML = "text/html"
    TEXT_CSS = "text/css"
    TEXT_JAVASCRIPT = "text/javascript"
    APPLICATION_JSON = "application/json"
    APPLICATION_XML = "application/xml"
    APPLICATION_PDF = "application/pdf"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_GIF = "image/gif"
    IMAGE_SVG_XML = "image/svg+xml"
    AUDIO_MPEG = "audio/mpeg"
    AUDIO_WAV = "audio/wav"
    VIDEO_MP4 = "video/mp4"
    VIDEO_MPEG = "video/mpeg"
    APPLICATION_OCTET_STREAM = "application/octet-stream"
    MULTIPART_FORM_DATA = "multipart/form-data"
    APPLICATION_X_WWW_FORM_URLENCODED = "application/x-www-form-urlencoded"


class BlobstoreClientConfig(BaseModel):
    endpoint: str = "https://ton.atla.twitter.com"
    max_connections: int = 50
    max_keepalive_connections: int = 20
    keepalive_expiry_sec: int = 15
    connection_timeout_sec: int | None = 30
    timeout_sec: int = 60
    auth_token: str | None = None


class _BlobstoreClient:
    _cached_clients: dict[int, httpx.AsyncClient] = {}

    @classmethod
    def _build_client(cls, config: BlobstoreClientConfig) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_keepalive_connections,
            keepalive_expiry=config.keepalive_expiry_sec,
        )
        timeout = httpx.Timeout(
            config.timeout_sec,
            connect=config.connection_timeout_sec,
        )
        return httpx.AsyncClient(limits=limits, timeout=timeout)

    @classmethod
    def get(cls, config: BlobstoreClientConfig) -> httpx.AsyncClient:
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in cls._cached_clients:
            cls._cached_clients[loop_id] = cls._build_client(config)
        return cls._cached_clients[loop_id]


class Blobstore(ABC):
    namespace: str
    subpath: str | None
    config: BlobstoreClientConfig

    def __init__(
        self,
        namespace: str,
        subpath: str | None = None,
        auth_token: str | None = None,
        **kwargs,
    ):
        namespace = namespace.strip("/")
        subpath = subpath.strip("/") if subpath else None
        logger.info(f"Initializing blobstore {namespace=}, {subpath=}")
        self.namespace = quote(namespace)
        self.subpath = quote(subpath) if subpath else None
        self.config = BlobstoreClientConfig(auth_token=auth_token, **kwargs)
        self.config.endpoint = self.config.endpoint.rstrip("/")

    @property
    def _cli(self) -> httpx.AsyncClient:
        return _BlobstoreClient.get(self.config)

    def _get_url(self, key: str) -> str:
        key = f"{self.subpath}/{quote(key)}" if self.subpath else quote(key)
        return f"{self.config.endpoint}/{self.namespace}/{quote(key)}"

    def _get_headers(
        self, additional_headers: dict[str, str] | None = None
    ) -> dict[str, str]:
        headers = additional_headers.copy() if additional_headers else {}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return headers

    async def put(
        self,
        key: str,
        data: bytes,
        content_type: ContentType = ContentType.APPLICATION_OCTET_STREAM,
    ) -> str:
        logger.info(
            f"Writing to blobstore {self.namespace}, key: {key}, content_type: {content_type}, data-len: {len(data)}"
        )
        url = self._get_url(key)
        headers = self._get_headers(
            {
                "Content-Type": content_type.value,
                "Content-Length": str(len(data)),
            }
        )
        try:
            resp = await self._cli.post(url, headers=headers, content=data)
            resp.raise_for_status()
            return url
        except Exception as e:
            logger.error(
                f"Failed writing to blobstore {self.namespace}, error: {e}, {traceback.format_exc()}"
            )
            raise e

    async def get(self, key: str) -> bytes | None:
        logger.info(f"Reading from blobstore {self.namespace}, key: {key}")
        url = self._get_url(key)
        headers = self._get_headers()
        try:
            resp = await self._cli.get(url, headers=headers)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(
                    f"Key not found in blobstore {self.namespace}, key: {key}"
                )
                return None
            elif e.response.status_code == 403:
                logger.warning(
                    f"Key got 403 error in blobstore {self.namespace}, key: {key}"
                )
                return None
            raise e
        except Exception:
            logger.warning(
                f"Failed to read from blobstore {self.namespace}, error: {traceback.format_exc()}"
            )
            return None
