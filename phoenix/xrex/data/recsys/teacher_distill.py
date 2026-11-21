# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from typing import cast

import numpy as np
import pyarrow as pa

from xai_proto import recsys_pb2

COLUMN = "teacherScores"
DEFAULT_ACTIONS = "CLIENT_TWEET_OPEN_LINK,ADS_MMP_CLICK,CLIENT_TWEET_VIDEO_QUALITY_VIEW"


def parse_actions(value: str) -> tuple[int, ...]:
    actions = tuple(
        int(v) if v.strip().isdigit() else recsys_pb2.ActionName.Value(v.strip())
        for v in value.split(",")
        if v.strip()
    )
    if not actions:
        raise ValueError("distill_actions needs at least one action")
    return actions


def parse_spans_ms(value: str) -> tuple[tuple[int, int], ...]:
    spans = []
    for item in (x.strip() for x in value.split(",") if x.strip()):
        start, sep, end = item.partition("-")
        if not sep or int(start) >= int(end):
            raise ValueError(f"distill_bad_spans_ms: {item!r} must be START_MS-END_MS")
        spans.append((int(start), int(end)))
    return tuple(spans)


def read_teacher_slots(
    record_batch: pa.RecordBatch, teacher: str, actions: tuple[int, ...]
) -> np.ndarray | None:
    if COLUMN not in record_batch.schema.names:
        return None
    column = cast(pa.MapArray, record_batch.column(COLUMN))
    if not pa.types.is_map(column.type):
        raise ValueError(f"{COLUMN} must be a Map, got {column.type}")
    rows = record_batch.num_rows
    offsets = np.asarray(column.offsets, dtype=np.int64)
    first, last = int(offsets[0]), int(offsets[-1])
    keys = column.keys.slice(first, last - first).to_numpy(zero_copy_only=False)
    items = cast(pa.StructArray, column.items.slice(first, last - first))
    entry_rows = np.repeat(np.arange(rows), np.diff(offsets))
    valid_rows = column.is_valid().to_numpy(zero_copy_only=False)
    entry_actions = cast(pa.ListArray, items.field("actions"))
    action_offsets = np.asarray(entry_actions.offsets, dtype=np.int64)
    action_ids = entry_actions.values.to_numpy(zero_copy_only=False)
    slots = cast(pa.FixedSizeListArray, items.field("scores"))
    width = slots.type.list_size
    per_slot = cast(pa.ListArray, slots.values)
    slot_offsets = np.asarray(per_slot.offsets, dtype=np.int64)
    scores = per_slot.values.to_numpy(zero_copy_only=False)
    out = np.full((rows, width, len(actions)), np.nan, dtype=np.float32)
    for entry in np.flatnonzero(keys == teacher):
        row = entry_rows[entry]
        heads = action_ids[action_offsets[entry] : action_offsets[entry + 1]]
        if not valid_rows[row] or len(heads) == 0:
            continue
        base = (slots.offset + entry) * width
        start, end = slot_offsets[base], slot_offsets[base + width]
        if end - start != width * len(heads):
            raise ValueError(
                f"{COLUMN}[{teacher}]: {end - start} scores for {width} slots x {len(heads)} actions"
            )
        values = scores[start:end].reshape(width, len(heads))
        for k, action in enumerate(actions):
            found = np.flatnonzero(heads == action)
            if found.size:
                out[row, :, k] = values[:, found[0]]
    return out


def mask_bad_spans(
    probs: np.ndarray, impressed_ms: np.ndarray, spans_ms: tuple[tuple[int, int], ...]
) -> np.ndarray:
    bad = np.zeros(impressed_ms.shape, dtype=np.bool_)
    for start, end in spans_ms:
        bad |= (impressed_ms >= start) & (impressed_ms < end)
    if not bad.any():
        return probs
    return np.where(bad[..., None], np.float32(np.nan), probs)
