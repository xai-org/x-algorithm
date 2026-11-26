import logging
import time

from monitor.metrics import Metrics
from protos.night_owl_search.night_owl_search_pb2 import (
    SEARCH_TARGET_REALTIME_AUTHOR,
    Pagination,
    Ranking,
    RecencyRanking,
    SearchAfter,
    SearchRequest,
)

from grox.config.config import grox_config
from grox.core.clients.nightowl_client import NightOwlClient

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "user_replies"
_MAX_PAGES = 10


class UserRepliesLoader:
    _nightowl: NightOwlClient | None = None

    @classmethod
    async def fetch_reply_ids(
        cls,
        user_id: int,
        *,
        max_age_sec: int,
        max_ids: int,
        pinned_post_id: int | None = None,
        enable_archive: bool = True,
    ) -> list[int]:
        client = cls._client()
        ids = (
            await cls._search(client, user_id, max_age_sec, max_ids, enable_archive)
            if client is not None
            else []
        )
        if pinned_post_id is not None and pinned_post_id not in ids:
            ids = [pinned_post_id, *ids]
        return ids

    @classmethod
    async def _search(
        cls,
        client: NightOwlClient,
        user_id: int,
        max_age_sec: int,
        max_ids: int,
        enable_archive: bool,
    ) -> list[int]:
        since_sec = int(time.time()) - max_age_sec
        post_ids: set[int] = set()
        cursor = b""
        for _ in range(_MAX_PAGES):
            request = SearchRequest(
                old_query=f"include:spam include:protected filter:replies since_time:{since_sec}",
                target=SEARCH_TARGET_REALTIME_AUTHOR,
                enable_archive=enable_archive,
                from_user_ids=[user_id],
                ranking=Ranking(recency=RecencyRanking()),
                pagination=Pagination(
                    search_after=SearchAfter(size=max_ids, cursor=cursor)
                ),
            )
            try:
                response = await client.search(request)
            except Exception:
                Metrics.counter(f"{_METRIC_PREFIX}.nightowl_failed.count").add(1)
                raise
            if response.timed_out:
                Metrics.counter(f"{_METRIC_PREFIX}.partial_page.count").add(1)
            post_ids.update(hit.doc_id for hit in response.hits)
            if len(post_ids) >= max_ids:
                break
            next_cursor = (
                response.next_cursor if response.HasField("next_cursor") else b""
            )
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return sorted(post_ids, reverse=True)[:max_ids]

    @classmethod
    def _client(cls) -> NightOwlClient | None:
        if cls._nightowl is None and grox_config.nightowl is not None:
            cfg = grox_config.nightowl
            cls._nightowl = NightOwlClient(endpoint=cfg.endpoint, timeout=cfg.timeout)
        return cls._nightowl
