# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

AD_SIDE_ID_BUCKETS = 1 << 19
AD_SIDE_CARD_TYPES = 48
AD_SIDE_MEDIA_TYPES = 8
MEDIA_CONTENT_TYPE_UNKNOWN = 4
AD_SIDE_CREATIVE_UNKNOWN = AD_SIDE_CARD_TYPES + MEDIA_CONTENT_TYPE_UNKNOWN
AD_SIDE_OBJECTIVES = 64

AD_SIDE_FIELD_VOCAB = {
    "campaign": AD_SIDE_ID_BUCKETS,
    "line_item": AD_SIDE_ID_BUCKETS,
    "creative": AD_SIDE_CARD_TYPES + AD_SIDE_MEDIA_TYPES,
    "objective": AD_SIDE_OBJECTIVES,
    "url_domain": AD_SIDE_ID_BUCKETS,
}
AD_SIDE_DEFAULT_FIELDS = ("campaign", "line_item", "creative")
AD_SIDE_FIELD_COLUMNS = {
    "campaign": ("campaignIdSeq",),
    "line_item": ("lineItemIdSeq",),
    "creative": ("unifiedCardTypeSeq", "mediaContentTypeSeq"),
    "objective": ("lineItemObjectiveSeq",),
    "url_domain": ("urlDomainIdSeq",),
}

CREATIVE_UNLOGGED_START_MS = 1_787_788_800_000
CREATIVE_UNLOGGED_END_MS = 1_789_833_600_000


def parse_ad_side_fields(spec: str) -> tuple[str, ...]:
    fields = tuple(s.strip() for s in spec.split(",") if s.strip())
    unknown = [f for f in fields if f not in AD_SIDE_FIELD_VOCAB]
    if not fields or unknown or len(set(fields)) != len(fields):
        raise ValueError(f"ad_side_fields {spec!r}: expected nonempty, unique known fields")
    return fields


def hash_ad_side_ids(ids: npt.ArrayLike, buckets: int = AD_SIDE_ID_BUCKETS) -> np.ndarray:
    ids = np.asarray(ids)
    x = ids.astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    rows = (x % np.uint64(buckets - 1)).astype(np.int32) + 1
    return np.where(ids == 0, 0, rows).astype(np.int32)


def creative_format(
    card_type: npt.ArrayLike,
    media_type: npt.ArrayLike,
    impressed_time_ms: npt.ArrayLike,
) -> np.ndarray:
    card = np.clip(np.asarray(card_type).astype(np.int32), 0, AD_SIDE_CARD_TYPES - 1)
    media = np.clip(np.asarray(media_type).astype(np.int32), 0, AD_SIDE_MEDIA_TYPES - 1)
    out = np.where(card > 0, card, AD_SIDE_CARD_TYPES + media).astype(np.int32)
    t = np.asarray(impressed_time_ms)
    unlogged = (t >= CREATIVE_UNLOGGED_START_MS) & (t < CREATIVE_UNLOGGED_END_MS)
    out[unlogged & (card == 0) & (media == 0)] = AD_SIDE_CREATIVE_UNKNOWN
    return out


def ad_side_ids(
    fields: Sequence[str],
    columns: Mapping[str, np.ndarray],
    *,
    impressed_time_ms: np.ndarray,
) -> np.ndarray:
    out = np.empty((*impressed_time_ms.shape, len(fields)), dtype=np.int32)
    for k, field in enumerate(fields):
        if field in ("campaign", "line_item", "url_domain"):
            out[..., k] = hash_ad_side_ids(columns[AD_SIDE_FIELD_COLUMNS[field][0]])
        elif field == "creative":
            out[..., k] = creative_format(
                columns["unifiedCardTypeSeq"], columns["mediaContentTypeSeq"], impressed_time_ms
            )
        elif field == "objective":
            out[..., k] = np.clip(columns["lineItemObjectiveSeq"], 0, AD_SIDE_OBJECTIVES - 1)
        else:
            raise ValueError(f"unknown ad-side field {field!r}")
    return out
