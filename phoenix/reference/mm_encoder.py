#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

IMAGE_PAD = "<|vision_start|><|image_pad|><|vision_end|>"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

DEFAULT_SYSTEM_PROMPT = (
    "Represent this X post for retrieving relevant X posts by content, topic, "
    "sentiment, or interest groups"
)

TRUNCATE_DIM = 1024

QWEN_MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"

ENDPOINT_ENV = "MM_ENCODER_ENDPOINT"


@dataclass
class Author:
    name: str | None = None
    handle: str | None = None


@dataclass
class Image:
    content: bytes | None = None


@dataclass
class Video:
    frames: list[bytes] = field(default_factory=list)


@dataclass
class PollCard:
    choice_label: str | None = None
    choice_image: Image | None = None


@dataclass
class LegacyCard:
    title: str | None = None
    description: str | None = None
    thumbnail_image: Image | None = None
    poll_cards: list[PollCard] | None = None


@dataclass
class UnifiedCard:
    title: str | None = None
    description: str | None = None
    media: Image | Video | None = None


@dataclass
class CardV2:
    legacy_card: LegacyCard | None = None
    unified_cards: list[UnifiedCard] | None = None


@dataclass
class ArticleMetadata:
    title: str | None = None
    cover_media: Image | None = None
    media: list[Image | Video] | None = None


@dataclass
class Post:
    id: int | None = None
    text: str | None = None
    author: Author | None = None
    media: list[Image | Video] | None = None
    cards: list[CardV2] | None = None
    article: ArticleMetadata | None = None
    quoted_post: Post | None = None


def remove_tco_links(text: str) -> str:
    import re

    return re.sub(r"https?://t\.co/\S+", "", text).strip()


def render_post(
    post: Post,
    max_images: int | None = None,
    include_quoted: bool = True,
) -> tuple[str, list[bytes]]:
    num_posts = 1
    if include_quoted and post.quoted_post:
        num_posts = 2

    per_post_budget = None
    if max_images is not None:
        per_post_budget = max_images // num_posts

    text_parts: list[str] = []
    images: list[bytes] = []

    _render_single_post(post, text_parts, images, per_post_budget)

    if include_quoted and post.quoted_post:
        text_parts.append("\n\nQuoted post:")
        quoted_text_parts: list[str] = []
        quoted_images: list[bytes] = []
        _render_single_post(post.quoted_post, quoted_text_parts, quoted_images, per_post_budget)
        if quoted_text_parts:
            text_parts.extend(quoted_text_parts)
        images.extend(quoted_images)

    return "\n".join(filter(None, text_parts)), images


def _render_single_post(
    post,
    text_parts: list[str],
    images: list[bytes],
    max_images: int | None,
) -> None:
    author = getattr(post, "author", None)
    if author:
        user_parts: list[str] = []
        if author.name:
            user_parts.append(author.name)
        if author.handle:
            user_parts.append(f"@{author.handle}")
        if user_parts:
            text_parts.append(" ".join(user_parts))

    article = getattr(post, "article", None)
    if article:
        if article.title:
            text_parts.append(f"Article: {article.title}")
        if article.cover_media:
            _process_image(article.cover_media, text_parts, images, max_images)

    if getattr(post, "text", None):
        text_parts.append(remove_tco_links(post.text))

    if getattr(post, "media", None):
        for medium in post.media:
            if max_images is not None and len(images) >= max_images:
                break
            _process_medium(medium, text_parts, images, max_images)

    if getattr(post, "cards", None):
        for card in post.cards:
            if max_images is not None and len(images) >= max_images:
                break
            _process_cardv2(card, text_parts, images, max_images)

    if article and article.media:
        for medium in article.media:
            if max_images is not None and len(images) >= max_images:
                break
            _process_medium(medium, text_parts, images, max_images)


def _process_medium(
    medium, text_parts: list[str], images: list[bytes], max_images: int | None
) -> None:
    if isinstance(medium, Image):
        _process_image(medium, text_parts, images, max_images)
    elif isinstance(medium, Video):
        _process_video(medium, text_parts, images, max_images)


def _process_image(
    image, text_parts: list[str], images: list[bytes], max_images: int | None
) -> None:
    if max_images is not None and len(images) >= max_images:
        return
    if getattr(image, "content", None):
        text_parts.append(IMAGE_PAD)
        images.append(image.content)


def _process_video(
    video, text_parts: list[str], images: list[bytes], max_images: int | None
) -> None:
    frames = [f for f in (getattr(video, "frames", None) or []) if f]
    if not frames:
        return

    remaining_budget = None
    if max_images is not None:
        remaining_budget = max_images - len(images)
        if remaining_budget <= 0:
            return

    if remaining_budget is not None and len(frames) > remaining_budget:
        indices = np.linspace(0, len(frames) - 1, remaining_budget, dtype=int).tolist()
        frames = [frames[i] for i in indices]

    for frame in frames:
        if max_images is not None and len(images) >= max_images:
            break
        text_parts.append(IMAGE_PAD)
        images.append(frame)


def _process_cardv2(
    card, text_parts: list[str], images: list[bytes], max_images: int | None
) -> None:
    lc = getattr(card, "legacy_card", None)
    if lc:
        card_text = []
        if lc.title:
            card_text.append(f"Card: {lc.title}")
        if lc.description:
            card_text.append(lc.description)
        if card_text:
            text_parts.append("\n" + "\n".join(card_text))

        if lc.thumbnail_image:
            _process_image(lc.thumbnail_image, text_parts, images, max_images)

        if lc.poll_cards:
            for poll_card in lc.poll_cards:
                if poll_card.choice_label:
                    text_parts.append(f"Poll choice: {poll_card.choice_label}")
                if poll_card.choice_image:
                    _process_image(poll_card.choice_image, text_parts, images, max_images)

    if getattr(card, "unified_cards", None):
        for uc in card.unified_cards:
            card_text = []
            if uc.title:
                card_text.append(f"Card: {uc.title}")
            if uc.description:
                card_text.append(uc.description)
            if card_text:
                text_parts.append("\n" + "\n".join(card_text))

            if uc.media:
                _process_medium(uc.media, text_parts, images, max_images)


def format_chatml(text_with_pads: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return (
        f"{IM_START}system\n{system_prompt}{IM_END}\n"
        f"{IM_START}user\n{text_with_pads}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def truncate_l2(emb: np.ndarray, dim: int = TRUNCATE_DIM) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    if dim > 0 and emb.shape[0] > dim:
        emb = emb[:dim]
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def encode(
    text: str,
    images: Sequence[bytes] | None = None,
    *,
    backend: str = "http",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    dim: int = TRUNCATE_DIM,
    endpoint: str | None = None,
    timeout: float = 60.0,
    model_id: str = QWEN_MODEL_ID,
    device: str | None = None,
) -> np.ndarray:
    img_list: list[bytes] = list(images) if images else []
    if backend == "http":
        raw = _encode_http(
            text, img_list, system_prompt=system_prompt, endpoint=endpoint, timeout=timeout
        )
    elif backend == "transformers":
        raw = _encode_transformers(
            text, img_list, system_prompt=system_prompt, model_id=model_id, device=device
        )
    else:
        raise ValueError(f"unknown backend {backend!r} (expected 'http' or 'transformers')")
    return truncate_l2(raw, dim)


def embed_post(
    post: Post,
    *,
    transcript: str | None = None,
    max_images: int | None = None,
    backend: str = "http",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    dim: int = TRUNCATE_DIM,
    **encode_kwargs,
) -> np.ndarray:
    text, imgs = render_post(post, max_images=max_images)
    if transcript:
        text += f"\nTranscript: {transcript}"
    return encode(
        text, imgs, backend=backend, system_prompt=system_prompt, dim=dim, **encode_kwargs
    )


def _encode_http(
    text: str,
    images: list[bytes],
    *,
    system_prompt: str,
    endpoint: str | None,
    timeout: float,
) -> np.ndarray:
    import base64

    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(
            "BLOCKED: the 'http' backend needs httpx. Install the mm-encoder extra "
            "(`uv sync --extra mm-encoder`) and run an SGLang server on the stock "
            f"{QWEN_MODEL_ID} weights (--is-embedding)."
        ) from e

    endpoint = endpoint or os.environ.get(ENDPOINT_ENV)
    if not endpoint:
        raise RuntimeError(
            "BLOCKED: no SGLang endpoint for the 'http' backend. Pass endpoint='host:port' "
            f"or set ${ENDPOINT_ENV} to a server you run on the stock {QWEN_MODEL_ID} "
            "weights (--is-embedding). Refusing to fabricate an embedding."
        )

    prompt = format_chatml(text, system_prompt)
    payload: dict = {"text": prompt, "sampling_params": {"max_new_tokens": 1}}
    if images:
        payload["image_data"] = [
            f"data:image/png;base64,{base64.b64encode(img).decode()}" for img in images
        ]

    base = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
    resp = httpx.post(f"{base}/encode", json=payload, timeout=timeout)
    resp.raise_for_status()
    return np.asarray(_parse_encode_response(resp.json()), dtype=np.float32)


def _parse_encode_response(result) -> list:
    if isinstance(result, dict) and "embedding" in result:
        return list(result["embedding"])
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "embedding" in first:
            return list(first["embedding"])
        return list(first)
    if isinstance(result, dict):
        if "embeddings" in result:
            emb = result["embeddings"]
            if isinstance(emb, list) and emb:
                return list(emb[0]) if isinstance(emb[0], list) else list(emb)
        raise ValueError(f"Cannot parse /encode response: dict keys={list(result.keys())}")
    raise ValueError(f"Cannot parse /encode response: {type(result)}")


_ST_MODEL_CACHE: dict[tuple[str, str | None], object] = {}


def _load_st_model(model_id: str, device: str | None):
    key = (model_id, device)
    model = _ST_MODEL_CACHE.get(key)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
        _ST_MODEL_CACHE[key] = model
    return model


def _encode_transformers(
    text: str,
    images: list[bytes],
    *,
    system_prompt: str,
    model_id: str,
    device: str | None,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "BLOCKED: the 'transformers' backend needs sentence-transformers + torch "
            "(+ pillow for images). Install the mm-encoder extra "
            f"(`uv sync --extra mm-encoder`); the stock {QWEN_MODEL_ID} weights are "
            "pulled from HF on first use."
        ) from e

    model = _load_st_model(model_id, device)

    clean_text = text.replace(IMAGE_PAD, " ")
    clean_text = " ".join(clean_text.split()).strip()

    if not images:
        out = model.encode([clean_text], prompt=system_prompt)
        return np.asarray(out[0], dtype=np.float32)

    try:
        import io

        from PIL import Image as PILImage
    except ImportError as e:
        raise RuntimeError(
            "BLOCKED: the 'transformers' backend needs pillow to decode image bytes. "
            "Install the mm-encoder extra (`uv sync --extra mm-encoder`)."
        ) from e

    pil_images = [PILImage.open(io.BytesIO(b)).convert("RGB") for b in images]
    content: list[dict] = [{"type": "image", "image": im} for im in pil_images]
    if clean_text:
        content.append({"type": "text", "text": clean_text})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    out = model.encode([conversation])
    return np.asarray(out[0], dtype=np.float32)


def _self_check() -> None:
    import base64

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
    )

    p_text = Post(
        id=1, text="hello https://t.co/abc world", author=Author(name="Ada", handle="ada")
    )
    t0, i0 = render_post(p_text)
    assert i0 == [], i0
    assert IMAGE_PAD not in t0, t0
    assert "Ada @ada" in t0, t0
    assert "https://t.co/" not in t0, "t.co link not stripped"
    assert "hello" in t0 and "world" in t0

    p_img = Post(
        id=2,
        text="pic post",
        author=Author(name="Bo", handle="bo"),
        media=[Image(content=png), Image(content=png + b"x")],
    )
    t1, i1 = render_post(p_img)
    assert t1.count(IMAGE_PAD) == len(i1) == 2, (t1.count(IMAGE_PAD), len(i1))
    assert i1 == [png, png + b"x"], "image bytes not collected in order"

    p_quote = Post(
        id=3,
        text="main",
        author=Author(handle="main"),
        media=[Image(content=png)],
        quoted_post=Post(
            id=4, text="quoted", author=Author(handle="qt"), media=[Image(content=png)]
        ),
    )
    t2, i2 = render_post(p_quote, max_images=2)
    assert "\n\nQuoted post:" in t2, t2
    assert t2.count(IMAGE_PAD) == len(i2) == 2, (t2.count(IMAGE_PAD), len(i2))
    p_quote_budget = Post(
        id=5,
        text="m",
        media=[Image(content=png), Image(content=png)],
        quoted_post=Post(id=6, text="q", media=[Image(content=png), Image(content=png)]),
    )
    t3, i3 = render_post(p_quote_budget, max_images=2)
    assert len(i3) == 2 and t3.count(IMAGE_PAD) == 2, (len(i3), t3.count(IMAGE_PAD))

    p_vid = Post(id=7, text="vid", media=[Video(frames=[png, png, png, png])])
    t4, i4 = render_post(p_vid, max_images=2)
    assert len(i4) == 2 and t4.count(IMAGE_PAD) == 2, (len(i4), t4.count(IMAGE_PAD))

    cm = format_chatml("BODY")
    assert cm == (
        f"{IM_START}system\n{DEFAULT_SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\nBODY{IM_END}\n{IM_START}assistant\n"
    ), repr(cm)
    assert DEFAULT_SYSTEM_PROMPT in cm and cm.endswith("assistant\n")

    rng = np.random.default_rng(0)
    v4096 = rng.standard_normal(4096).astype(np.float32) * 7.0
    u = truncate_l2(v4096, 1024)
    assert u.shape == (1024,) and u.dtype == np.float32, (u.shape, u.dtype)
    assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-5, np.linalg.norm(u)
    assert np.allclose(u, v4096[:1024] / np.linalg.norm(v4096[:1024]), atol=1e-6)
    short = truncate_l2(rng.standard_normal(1024).astype(np.float32), 1024)
    assert short.shape == (1024,) and abs(float(np.linalg.norm(short)) - 1.0) < 1e-5
    assert np.array_equal(truncate_l2(np.zeros(4096, np.float32), 1024), np.zeros(1024, np.float32))

    print(
        "mm_encoder self-check PASS  "
        "render(pad==images 1:1, author emitted, t.co stripped, 50/50 quoted budget, "
        "video-frame subsample), format_chatml(v5 system+ChatML), "
        "truncate_l2(4096->1024 unit-norm)\n"
        "  (⚠ ships CODE not weights; v5 = STOCK Qwen/Qwen3-VL-Embedding-8B, NOT any "
        "fine-tuned/soup checkpoint)"
    )

    try:
        encode("x", None, backend="http", endpoint=None)
        raise AssertionError("http backend did not raise BLOCKED without an endpoint")
    except RuntimeError as e:
        assert "BLOCKED" in str(e), e
        print(f"  encode[http]: correctly BLOCKED without endpoint ({str(e).split('.')[0]})")

    try:
        import sentence_transformers
    except ImportError:
        print(
            "  encode[transformers]: SKIP (BLOCKED) — sentence-transformers/torch not "
            "installed. Install the mm-encoder extra + supply Qwen OSS weights to run it."
        )
    else:
        print(
            "  encode[transformers]: deps present; not pulling the 8B weights in the "
            "self-check. Construct via embed_post(..., backend='transformers') to run."
        )


if __name__ == "__main__":
    _self_check()
