import asyncio
import logging
import traceback

from strato_http.queries.content_understanding_post_quote_metadata import (
    StratoContentUnderstandingPostQuoteMetadata,
)

from grox.core.data_loaders.data_types import Post
from grox.core.data_loaders.post_mapper import PostMapper

logger = logging.getLogger(__name__)


class DescendantHydrator:
    def __init__(self):
        self._strato_hydrator = StratoContentUnderstandingPostQuoteMetadata()

    async def hydrate(self, post_ids: list[int]) -> list[Post]:
        if not post_ids:
            return []

        tasks = [self._strato_hydrator.fetch(post_id) for post_id in post_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        posts: list[Post] = []
        for post_id, result in zip(post_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to hydrate post {post_id}: {result}")
                continue
            if result is None:
                logger.warning(f"No metadata found for post {post_id}")
                continue
            try:
                post = PostMapper.from_strato_post_with_quote_metadata(result)
                posts.append(post)
            except Exception:
                logger.warning(
                    f"Failed to map post {post_id}: {traceback.format_exc()}"
                )

        return posts
