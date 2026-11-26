# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

TWITTER_EPOCH_MS = 1288834974657
DEFAULT_COLD_START_MAX_AGE_SECONDS = 2 * 60 * 60


def default_cold_pool_metadata_path() -> Path:
    return (
        Path(os.environ.get("PHOENIX_INDEX_BASE", "/data/phoenix_index/prod"))
        / "post_author_pairs_metadata/metadata_1day.parquet"
    )


def cold_pool_mask(
    post_ids: np.ndarray,
    metadata_path: str | Path | None,
    *,
    max_favs: int | None = 8,
    max_views: int = 500,
) -> np.ndarray | None:
    if not metadata_path or not os.path.exists(metadata_path):
        if metadata_path:
            logger.warning("Cold-pool metadata not found, skipping filter: %s", metadata_path)
        return None

    start = time.time()
    meta = pq.read_table(str(metadata_path), columns=["post_id", "fav_count", "view_count"])
    m_pid = np.asarray(meta.column("post_id").to_numpy(zero_copy_only=False), dtype=np.uint64)
    m_fav = np.asarray(meta.column("fav_count").to_numpy(zero_copy_only=False))
    m_view = np.asarray(meta.column("view_count").to_numpy(zero_copy_only=False))

    ok = m_view < max_views
    if max_favs is not None:
        ok &= m_fav < max_favs

    post_ids_u64 = np.asarray(post_ids).astype(np.uint64, copy=False)
    in_meta = np.isin(post_ids_u64, m_pid, assume_unique=True)
    in_ok = np.isin(post_ids_u64, m_pid[ok], assume_unique=True)
    mask = ~in_meta | in_ok
    logger.info(
        "Cold-pool filter (fav%s, view<%d): kept %d/%d posts "
        "(%.1f%%; %d missing-meta kept) in %.1fs",
        f"<{max_favs}" if max_favs is not None else " unrestricted",
        max_views,
        int(mask.sum()),
        len(mask),
        100.0 * mask.sum() / max(len(mask), 1),
        int((~in_meta).sum()),
        time.time() - start,
    )
    return mask


def fresh_post_mask(
    post_ids: np.ndarray, max_age_seconds: float, *, now_s: float | None = None
) -> np.ndarray:
    ids = np.asarray(post_ids).reshape(-1)
    if max_age_seconds <= 0:
        return np.ones(ids.shape[0], dtype=bool)
    now_ms = int((time.time() if now_s is None else now_s) * 1000.0)
    cutoff_ms = now_ms - int(max_age_seconds * 1000.0) - TWITTER_EPOCH_MS
    if cutoff_ms <= 0:
        return np.ones(ids.shape[0], dtype=bool)
    cutoff_id = np.int64(cutoff_ms) << np.int64(22)
    return ids.astype(np.int64, copy=False) >= cutoff_id
