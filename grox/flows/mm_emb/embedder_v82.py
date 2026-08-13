import logging
import time
from typing import Any

import numpy as np
from embed.embed_http import XaiEmbeddingClientHttp
from monitor.metrics import Metrics
from strato_http.queries.user_core import StratoUserCore

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post, User, Video
from grox.flows.mm_emb.constants import RECSYS_V82_MM_EMBED
from grox.flows.mm_emb.renderer_v82 import V82EmbedPostRenderer

logger = logging.getLogger(__name__)

MAX_FRAMES = 4


class MultimodalPostEmbedderV82:
    def __init__(self) -> None:
        embed_config = grox_config.get_embedding_model(RECSYS_V82_MM_EMBED)
        if embed_config.timeout_seconds is None:
            embed_config = embed_config.model_copy(update={"timeout_seconds": 60.0})
        self._client = XaiEmbeddingClientHttp(config=embed_config)
        self._user_core = StratoUserCore()

    @staticmethod
    def _renormalize(embedding: np.ndarray) -> list[float]:
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    async def _fetch_core_user(self, user: User | None) -> dict[str, Any] | None:
        if user is None or user.id is None:
            return None
        try:
            return await self._user_core.fetch(int(user.id))
        except Exception as e:
            Metrics.counter("post_embedding_v82.core_user_fetch_error").add(1)
            logger.warning(f"core.User fetch failed for {user.id}: {e}")
            return None

    @staticmethod
    def _collect_asr(post: Post) -> str | None:
        if not post.media:
            return None
        transcripts = [
            m.convo_video.asr_transcript
            for m in post.media
            if isinstance(m, Video) and m.convo_video and m.convo_video.asr_transcript
        ]
        return "\n".join(transcripts) if transcripts else None

    async def embed(
        self,
        post: Post,
        **kwargs,
    ) -> tuple[list[tuple[str, str | bytes]], list[float]]:
        total_start = time.perf_counter()

        core_user = await self._fetch_core_user(post.user)
        quoted_core_user = None
        if post.quoted_post is not None:
            quoted_core_user = await self._fetch_core_user(post.quoted_post.user)
        asr_transcript = self._collect_asr(post)

        render_start = time.perf_counter()
        text, images = V82EmbedPostRenderer.render_for_embedding(
            post,
            core_user=core_user,
            quoted_core_user=quoted_core_user,
            asr_transcript=asr_transcript,
            max_frames=MAX_FRAMES,
        )
        render_duration_ms = (time.perf_counter() - render_start) * 1000
        Metrics.histogram("post_embedding_v82.render_duration_ms").record(
            render_duration_ms
        )

        document: list[tuple[str, str | bytes]] = [("text", text)]
        for img in images:
            document.append(("image", img))

        if not text and not images:
            logger.warning(f"Post {post.id} has no text or media content")
            return document, []

        encode_start = time.perf_counter()
        embedding = await self._client.embed_openai_async(
            text=text, images=images or None
        )
        encode_duration_ms = (time.perf_counter() - encode_start) * 1000
        Metrics.histogram("post_embedding_v82.encode_duration_ms").record(
            encode_duration_ms
        )

        result = self._renormalize(embedding)

        total_duration_ms = (time.perf_counter() - total_start) * 1000
        Metrics.histogram("post_embedding_v82.total_duration_ms").record(
            total_duration_ms
        )
        Metrics.counter("post_embedding_v82.image_count").add(len(images))
        Metrics.counter("post_embedding_v82.core_user_hit").add(1 if core_user else 0)

        logger.info(
            f"Embedding V8.2 post={post.id}: total={total_duration_ms:.1f}ms "
            f"(render={render_duration_ms:.1f}ms, encode={encode_duration_ms:.1f}ms), "
            f"n_img={len(images)}, has_core_user={core_user is not None}, "
            f"has_asr={asr_transcript is not None}, text_len={len(text)}, dim={len(result)}"
        )

        return document, result
