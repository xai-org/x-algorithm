# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import bisect
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import NotRequired, TypedDict

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("rank")

DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_TIME_FORMATS = (DATE_TIME_FORMAT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")

BATCHES_PER_SUBDIR = 2000

BATCH_MANIFEST_FILENAME = ".batch_manifest.json"
_MANIFEST_KEY = re.compile(r"^partition=(\d+)/\d+/batch_(\d+)\.parquet$")


class ValidBatchesMetadata(TypedDict):
    min_valid_batch: int
    max_valid_batch: int
    num_partitions: int


class DataPosition(TypedDict):
    last_batch_id: int
    rows_read_in_batch: int
    batch_size: int | None
    block_shuffle_window_start: NotRequired[int]
    block_shuffle_segments: NotRequired[list[list[int]]]
    block_shuffle_num_shards: NotRequired[int]
    block_shuffle_fingerprint: NotRequired[str]
    block_shuffle_rows_in_file: NotRequired[int]
    block_shuffle_rows_in_next_file: NotRequired[int]
    rows_in_file: NotRequired[int]


def batch_path(topic_dir: str, partition_id: int, batch_id: int) -> str:
    sub_dir = str(batch_id // BATCHES_PER_SUBDIR)
    return os.path.join(
        topic_dir, f"partition={partition_id}", sub_dir, f"batch_{batch_id}.parquet"
    )


def load_valid_batches_metadata(path: str) -> ValidBatchesMetadata | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return ValidBatchesMetadata(
            min_valid_batch=int(data["min_valid_batch"]),
            max_valid_batch=int(data["max_valid_batch"]),
            num_partitions=int(data["num_partitions"]),
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to load valid batches metadata from %s: %s", path, e)
        return None


def parse_date_time(s: str) -> datetime:
    s = s.strip()
    for fmt in DATE_TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date {s!r}: expected 'YYYY-MM-DD HH:MM:SS', "
        "'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD'"
    )


def parse_date_bound(s: str) -> int | None:
    s = s.strip().strip("()")
    if s.lower() == "none":
        return None
    return int(parse_date_time(s).timestamp() * 1000)


class _FooterTimestamps:
    def __init__(self, topic_dir: str, min_batch: int, max_batch: int):
        self._topic_dir = topic_dir
        self._min_batch = min_batch
        self._max_batch = max_batch

    def __len__(self) -> int:
        return self._max_batch - self._min_batch + 1

    def __getitem__(self, idx: int) -> int:
        bid = self._min_batch + idx
        path = batch_path(self._topic_dir, 0, bid)
        try:
            pf = pq.ParquetFile(path)
            meta = pf.metadata.metadata or {}
            return int(meta[b"min_kafka_timestamp_ms"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Parquet file {path} does not contain min_kafka_timestamp_ms footer metadata. "
                f"Time-range filtering requires this footer key on "
                f"Hive-partitioned files. Error: {e}"
            ) from e
        except Exception as e:
            raise ValueError(f"Cannot read footer metadata from {path}: {e}") from e


@dataclass(frozen=True)
class BatchManifest:
    first_batch: int
    rows: np.ndarray
    min_ts: np.ndarray

    def _index(self, batch_id: int) -> int | None:
        i = batch_id - self.first_batch
        return i if 0 <= i < self.rows.shape[0] else None

    def rows_matrix(
        self, topic_dir: str, start_batch_id: int, end_batch_id: int, num_partitions: int
    ) -> np.ndarray:
        out = np.full((end_batch_id - start_batch_id, num_partitions), -1, dtype=np.int64)
        lo = max(start_batch_id, self.first_batch)
        hi = min(end_batch_id, self.first_batch + self.rows.shape[0])
        cols = min(num_partitions, self.rows.shape[1])
        if lo < hi and cols > 0:
            out[lo - start_batch_id : hi - start_batch_id, :cols] = self.rows[
                lo - self.first_batch : hi - self.first_batch, :cols
            ]
        for i, p in zip(*np.nonzero(out < 0)):
            path = batch_path(topic_dir, int(p), start_batch_id + int(i))
            out[i, p] = pq.ParquetFile(path).metadata.num_rows
        return out

    def batch_timestamps(self, min_batch: int, max_batch: int) -> list[int]:
        out: list[int] = []
        running: int | None = None
        for bid in range(min_batch, max_batch + 1):
            i = self._index(bid)
            ts: int | None = None
            if i is not None:
                row = self.min_ts[i]
                if row.shape[0] > 0 and row[0] >= 0:
                    ts = int(row[0])
                elif (row >= 0).any():
                    ts = int(row[row >= 0].min())
            if ts is not None:
                running = ts if running is None else max(running, ts)
            out.append(-1 if running is None else running)
        if running is None:
            raise ValueError(
                f"{BATCH_MANIFEST_FILENAME} lists no file for batch ids [{min_batch}, {max_batch}]"
            )
        first_known = next(t for t in out if t >= 0)
        return [first_known if t < 0 else t for t in out]


_manifest_cache: dict[str, tuple[tuple[int, int], BatchManifest | None]] = {}


def load_batch_manifest(topic_dir: str) -> BatchManifest | None:
    path = os.path.join(topic_dir, BATCH_MANIFEST_FILENAME)
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    cached = _manifest_cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    manifest: BatchManifest | None = None
    try:
        with open(path) as f:
            raw = json.load(f)
        entries: list[tuple[int, int, int, int]] = []
        for key, stats in raw.items():
            m = _MANIFEST_KEY.match(key)
            if m is None:
                continue
            entries.append(
                (
                    int(m.group(2)),
                    int(m.group(1)),
                    int(stats["rows"]),
                    int(stats["min_impressed_time_ms"]),
                )
            )
        if entries:
            bids = np.fromiter((e[0] for e in entries), dtype=np.int64, count=len(entries))
            parts = np.fromiter((e[1] for e in entries), dtype=np.int64, count=len(entries))
            first = int(bids.min())
            shape = (int(bids.max()) - first + 1, int(parts.max()) + 1)
            rows = np.full(shape, -1, dtype=np.int64)
            min_ts = np.full(shape, -1, dtype=np.int64)
            idx = (bids - first, parts)
            rows[idx] = np.fromiter((e[2] for e in entries), dtype=np.int64, count=len(entries))
            min_ts[idx] = np.fromiter((e[3] for e in entries), dtype=np.int64, count=len(entries))
            manifest = BatchManifest(first_batch=first, rows=rows, min_ts=min_ts)
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        logger.warning("Ignoring unreadable %s: %s", path, e)
        manifest = None
    _manifest_cache[path] = (stamp, manifest)
    return manifest


def steps_in_file(num_rows: int, batch_size: int) -> int:
    full, tail = divmod(num_rows, batch_size)
    return full + (1 if tail > 0 and tail * 2 >= batch_size else 0)


def remaining_steps_from_manifest(
    topic_dir: str,
    manifest: BatchManifest,
    num_partitions: int,
    num_shards: int,
    batch_size: int,
    start_batch_id: int,
    end_batch_id: int,
    resume_batch_id: int | None = None,
    resume_reads_in_batch: int = 0,
    shards_per_partition: int = 1,
) -> int:
    rows = manifest.rows_matrix(topic_dir, start_batch_id, end_batch_id, num_partitions)
    full, tail = np.divmod(rows, batch_size * shards_per_partition)
    kept_tail = tail // shards_per_partition
    steps = full + ((kept_tail > 0) & (kept_tail * 2 >= batch_size))
    shard_of = np.arange(num_partitions) % num_shards
    per_batch_shard = np.stack(
        [steps[:, shard_of == s].sum(axis=1) for s in range(num_shards)], axis=1
    )
    total = per_batch_shard.sum(axis=0)
    consumed = np.zeros(num_shards, dtype=np.int64)
    if resume_batch_id is not None and resume_batch_id >= start_batch_id:
        done = min(resume_batch_id, end_batch_id) - start_batch_id
        consumed = per_batch_shard[:done].sum(axis=0) + resume_reads_in_batch
    return max(int((total - consumed).min()), 0)


def _batch_timestamps(
    topic_dir: str, min_batch: int, max_batch: int
) -> _FooterTimestamps | list[int]:
    footer = _FooterTimestamps(topic_dir, min_batch, max_batch)
    if len(footer) == 0:
        return footer
    try:
        footer[0]
        return footer
    except ValueError as footer_error:
        manifest = load_batch_manifest(topic_dir)
        if manifest is None:
            raise footer_error
        logger.info(
            "No min_kafka_timestamp_ms footer under %s; resolving the time range from %s",
            topic_dir,
            BATCH_MANIFEST_FILENAME,
        )
        return manifest.batch_timestamps(min_batch, max_batch)


def batch_start_time_ms(topic_dir: str, batch_id: int) -> int:
    return int(_batch_timestamps(topic_dir, batch_id, batch_id)[0])


def resolve_time_range(
    topic_dir: str,
    min_batch: int,
    max_batch: int,
    min_timestamp_ms: int | None,
    max_timestamp_ms: int | None,
) -> tuple[int, int]:
    ts = _batch_timestamps(topic_dir, min_batch, max_batch)
    n = len(ts)

    if n == 0:
        raise ValueError("No valid batches — cannot resolve time range")
    data_min_ts = ts[0]
    data_max_ts = ts[n - 1]
    if min_timestamp_ms is not None and min_timestamp_ms > data_max_ts:
        raise ValueError(
            f"min_timestamp_ms={min_timestamp_ms} is after all data "
            f"(last batch min_ts={data_max_ts})"
        )
    if max_timestamp_ms is not None and max_timestamp_ms <= data_min_ts:
        raise ValueError(
            f"max_timestamp_ms={max_timestamp_ms} is before all data "
            f"(first batch min_ts={data_min_ts})"
        )

    start_idx = bisect.bisect_left(ts, min_timestamp_ms) if min_timestamp_ms is not None else 0
    end_idx = bisect.bisect_left(ts, max_timestamp_ms) if max_timestamp_ms is not None else n

    start_batch_id = min_batch + start_idx
    end_batch_id = min_batch + end_idx

    if start_batch_id >= end_batch_id:
        raise ValueError(
            f"Time range [{min_timestamp_ms}, {max_timestamp_ms}) "
            f"resolves to empty batch range [{start_batch_id}, {end_batch_id})"
        )

    logger.info(
        "Time range [%s, %s) → batch_ids [%d, %d) (%d batches)",
        min_timestamp_ms,
        max_timestamp_ms,
        start_batch_id,
        end_batch_id,
        end_batch_id - start_batch_id,
    )
    return start_batch_id, end_batch_id
