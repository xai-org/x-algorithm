import logging
import time

import numpy as np
from embed.embed_http import ChatTemplate, XaiEmbeddingClientHttp
from monitor.metrics import Metrics

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post, Video
from grox.flows.mm_emb.constants import RECSYS_V5_EMBED
from grox.flows.mm_emb.renderer import V5EmbedPostRenderer

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "Represent this X post for retrieving relevant X posts by content, topic, sentiment, or interest groups"
TRUNCATE_DIM = 1024


class MultimodalPostEmbedderV5:
    @staticmethod
    def has_video(post: Post) -> bool:
        if post.media:
            for m in post.media:
                if isinstance(m, Video):
                    return True
        return False

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_images: int | None = None,
    ):
        self.truncate_dim = TRUNCATE_DIM
        self.max_images = max_images
        embed_config = grox_config.get_embedding_model(RECSYS_V5_EMBED)
        if embed_config.timeout_seconds is None:
            embed_config = embed_config.model_copy(update={"timeout_seconds": 60.0})
        chat_template = ChatTemplate(system_prompt=system_prompt)
        self._client = XaiEmbeddingClientHttp(
            config=embed_config, chat_template=chat_template
        )

    def _maybe_truncate(self, embedding: np.ndarray) -> list[float]:
        if self.truncate_dim > 0 and len(embedding) > self.truncate_dim:
            emb = embedding[: self.truncate_dim]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.tolist()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    async def embed(
        self,
        post: Post,
        transcript: str | None = None,
        is_query: bool = False,
        **kwargs,
    ) -> tuple[list[tuple[str, str | bytes]], list[float]]:
        total_start = time.perf_counter()

        render_start = time.perf_counter()
        text_with_pads, images = V5EmbedPostRenderer.render_for_embedding(
            post, max_images=self.max_images
        )
        render_duration_ms = (time.perf_counter() - render_start) * 1000
        Metrics.histogram("post_embedding_v5.render_duration_ms").record(
            render_duration_ms
        )

        if transcript:
            text_with_pads += f"\nTranscript: {transcript}"

        document: list[tuple[str, str | bytes]] = [("text", text_with_pads)]
        for img in images:
            document.append(("image", img))

        if not text_with_pads and not images:
            logger.warning(f"Post {post.id} has no text or media content")
            return document, []

        encode_start = time.perf_counter()
        embedding = await self._client.encode_with_embedded_pads_async(
            text_with_pads, images if images else None
        )
        encode_duration_ms = (time.perf_counter() - encode_start) * 1000
        Metrics.histogram("post_embedding_v5.encode_duration_ms").record(
            encode_duration_ms
        )

        truncate_start = time.perf_counter()
        result = self._maybe_truncate(embedding)
        truncate_duration_ms = (time.perf_counter() - truncate_start) * 1000
        Metrics.histogram("post_embedding_v5.truncate_duration_ms").record(
            truncate_duration_ms
        )

        total_duration_ms = (time.perf_counter() - total_start) * 1000
        Metrics.histogram("post_embedding_v5.total_duration_ms").record(
            total_duration_ms
        )
        Metrics.counter("post_embedding_v5.image_count").add(len(images))

        total_image_bytes = sum(len(img) for img in images) if images else 0
        Metrics.histogram("post_embedding_v5.image_payload_bytes").record(
            total_image_bytes
        )

        logger.info(
            f"Embedding V5 post={post.id}: total={total_duration_ms:.1f}ms "
            f"(render={render_duration_ms:.1f}ms, encode={encode_duration_ms:.1f}ms, truncate={truncate_duration_ms:.1f}ms), "
            f"images={len(images)}, image_bytes={total_image_bytes:,}, text_len={len(text_with_pads)}, has_transcript={transcript is not None}"
        )

        return document, result
