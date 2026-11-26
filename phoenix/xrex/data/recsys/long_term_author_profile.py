# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ID_COLUMN = "longTermAuthorIds"
POST_COUNT_COLUMN = "longTermAuthorPostCounts"
STRONG_COUNT_COLUMN = "longTermAuthorStrongCounts"
LAST_ENGAGED_MS_COLUMN = "longTermAuthorLastEngagedMs"
IN_WINDOW_COLUMN = "longTermAuthorInWindow"
DWELL_POST_COUNT_COLUMN = "longTermAuthorDwellPostCounts"
AVG_DWELL_MS_COLUMN = "longTermAuthorAvgDwellMs"
COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    POST_COUNT_COLUMN,
    STRONG_COUNT_COLUMN,
    LAST_ENGAGED_MS_COLUMN,
    IN_WINDOW_COLUMN,
    DWELL_POST_COUNT_COLUMN,
    AVG_DWELL_MS_COLUMN,
)

HASHES = "user_long_term_author_hashes"
MASK = "user_long_term_author_mask"
POST_COUNTS = "user_long_term_author_post_counts"
STRONG_COUNTS = "user_long_term_author_strong_counts"
AGE_SEC = "user_long_term_author_age_sec"
IN_WINDOW = "user_long_term_author_in_window"
DWELL_POST_COUNTS = "user_long_term_author_dwell_post_counts"
AVG_DWELL_MS = "user_long_term_author_avg_dwell_ms"
BATCH_KEYS: tuple[str, ...] = (
    HASHES,
    MASK,
    POST_COUNTS,
    STRONG_COUNTS,
    AGE_SEC,
    IN_WINDOW,
    DWELL_POST_COUNTS,
    AVG_DWELL_MS,
)

_INT32_MAX = np.iinfo(np.int32).max


def _column(
    record_batch: pa.RecordBatch, name: str, batch_size: int, size: int, dtype: type
) -> np.ndarray:
    out = np.zeros((batch_size, size), dtype=dtype)
    if name not in record_batch.schema.names:
        return out
    col = record_batch.column(name)
    if not pa.types.is_fixed_size_list(col.type):
        raise ValueError(f"{name} must be a FixedSizeList column, got {col.type}")
    width = col.type.list_size
    values = col.values.slice(col.offset * width, len(col) * width)
    if values.null_count:
        values = pc.fill_null(values, False if pa.types.is_boolean(values.type) else 0)
    rows = values.to_numpy(zero_copy_only=False).reshape(batch_size, width)
    kept = min(width, size)
    out[:, :kept] = rows[:, :kept]
    if col.null_count:
        out[col.is_null().to_numpy(zero_copy_only=False)] = 0
    return out


def read(
    record_batch: pa.RecordBatch,
    batch_size: int,
    size: int,
    request_ms: np.ndarray,
    author_hash: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    ids = _column(record_batch, ID_COLUMN, batch_size, size, np.int64)
    mask = ids > 0
    ids = np.where(mask, ids, 0)
    last_engaged_ms = _column(record_batch, LAST_ENGAGED_MS_COLUMN, batch_size, size, np.int64)
    age_sec = np.clip(
        (request_ms.astype(np.int64)[:, None] - last_engaged_ms) // 1000, 0, _INT32_MAX
    )

    def counts(name: str) -> np.ndarray:
        return np.where(mask, _column(record_batch, name, batch_size, size, np.int32), 0)

    dwell_post_counts = counts(DWELL_POST_COUNT_COLUMN)
    return {
        HASHES: np.asarray(author_hash(ids), dtype=np.int32),
        MASK: mask,
        POST_COUNTS: counts(POST_COUNT_COLUMN),
        STRONG_COUNTS: counts(STRONG_COUNT_COLUMN),
        AGE_SEC: np.where(mask, age_sec, 0).astype(np.int32),
        IN_WINDOW: _column(record_batch, IN_WINDOW_COLUMN, batch_size, size, np.bool_) & mask,
        DWELL_POST_COUNTS: dwell_post_counts,
        AVG_DWELL_MS: np.where(dwell_post_counts > 0, counts(AVG_DWELL_MS_COLUMN), 0),
    }


def present(batch: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {key: cast(np.ndarray, batch[key]) for key in BATCH_KEYS if batch.get(key) is not None}


def empty(
    batch_size: int, size: int, author_hash: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    ids = np.zeros((batch_size, size), dtype=np.int64)
    return {
        HASHES: np.asarray(author_hash(ids), dtype=np.int32),
        MASK: np.zeros((batch_size, size), dtype=np.bool_),
        POST_COUNTS: np.zeros((batch_size, size), dtype=np.int32),
        STRONG_COUNTS: np.zeros((batch_size, size), dtype=np.int32),
        AGE_SEC: np.zeros((batch_size, size), dtype=np.int32),
        IN_WINDOW: np.zeros((batch_size, size), dtype=np.bool_),
        DWELL_POST_COUNTS: np.zeros((batch_size, size), dtype=np.int32),
        AVG_DWELL_MS: np.zeros((batch_size, size), dtype=np.int32),
    }
