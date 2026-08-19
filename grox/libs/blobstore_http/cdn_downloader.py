import logging
import os
import time

import httpx
from monitor.metrics import Metrics

from blobstore_http.blobstore import BlobstoreClientConfig, _BlobstoreClient

logger = logging.getLogger(__name__)


def _count_unavailable(status_code: int) -> None:
    Metrics.counter("cdn_downloader.unavailable.count").add(
        1, attributes={"status": str(status_code), "pid": str(os.getpid())}
    )


class CDNDownloader:
    def __init__(self):
        self._blobstore_config = BlobstoreClientConfig(endpoint="https://pbs.twimg.com")

    @property
    def _cli(self) -> httpx.AsyncClient:
        return _BlobstoreClient.get(self._blobstore_config)

    async def get(self, url: str) -> bytes | None:
        logger.info(f"Downloading {url}")
        start = time.perf_counter()
        try:
            response = await self._cli.get(url, follow_redirects=True)
            response.raise_for_status()
            logger.info(
                f"Downloaded {url} in {time.perf_counter() - start:.2f} seconds"
            )
            return response.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"CDN url not found: {url}")
                _count_unavailable(404)
                return None
            elif e.response.status_code in (401, 403):
                logger.warning(f"CDN url {e.response.status_code}: {url}")
                _count_unavailable(e.response.status_code)
                return None
            elif e.response.status_code == 500:
                logger.warning(f"CDN url 500: {url}")
                return None
            raise e

    async def getNonGraceful(self, url: str) -> bytes | None:
        logger.info(f"Downloading {url}")
        start = time.perf_counter()
        try:
            response = await self._cli.get(url, follow_redirects=True)
            response.raise_for_status()
            logger.info(
                f"Downloaded {url} in {time.perf_counter() - start:.2f} seconds"
            )
            return response.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"CDN url not found: {url}")
                _count_unavailable(404)
                return None
            elif e.response.status_code in (401, 403):
                logger.warning(f"CDN url {e.response.status_code}: {url}")
                _count_unavailable(e.response.status_code)
                return None
            raise e
