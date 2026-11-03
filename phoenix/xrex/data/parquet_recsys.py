# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import dataclasses
import enum
import functools
import hashlib
import logging
import os
import re
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Iterator, NamedTuple, cast, final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

from xai_configlib import configclass
from xrex.configs.config import Dataset
from xrex.data import conversion_labels
from xrex.data.parquet_recsys_metadata import (
    DataPosition,
    load_batch_manifest,
    parse_date_bound,
    parse_date_time,
    remaining_steps_from_manifest,
)
from xrex.data.parquet_recsys_metadata import (
    batch_path as _batch_path,
)
from xrex.data.parquet_recsys_metadata import (
    load_valid_batches_metadata as _load_valid_batches_metadata,
)
from xrex.data.parquet_recsys_metadata import (
    resolve_time_range as _resolve_time_range,
)
from xrex.data.recsys import id_list_features as id_lists
from xrex.data.recsys import reply_features
from xrex.data.recsys.array_allocator import ArrayAllocator
from xrex.data.recsys.constants import CONVERSION_DELAY_NONE
from xrex.data.recsys.recsys_batch import (
    EMBEDDING_CONFIG,
    NUM_ADS_COUNT_FEATURES,
    NUM_USER_INSTALLED_APPS,
    PAGE_RECORD_COLUMNS,
    POST_INT64_FEATURE_SIZE,
    USER_INT64_FEATURE_SIZE,
    VALUE_LABEL_DTYPES,
    CandidateNegativeFilter,
    CandidateNegativeMode,
    EmbeddingType,
    PostEmbeddingTable,
    PostSeq,
    RecsysFeaturesBatch,
    empty_conversion_delays,
    empty_feature_arrays,
    empty_history_url_embedding,
    empty_page_record_arrays,
    empty_user_feature_arrays,
    from_record_batch,
    value_label_dtypes,
)
from xrex.data.retrieval_dataset import PHOENIX_INDEX_BASE
from xrex.models.recsys_embedding import HashTable

rank_logger = logging.getLogger("rank")

DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def extract_datetime_from_file_name(file_name: str) -> datetime:
    match = re.search(r"year=(\d{4})/month=(\d{2})/day=(\d{2})/hour=(\d{2})", file_name)
    assert match is not None
    year, month, day, hour = match.groups()
    date_str = f"{year}-{month}-{day} {hour}:00:00"
    return datetime.strptime(date_str, DATE_TIME_FORMAT)


def load_global_ids_from_parquet_file(
    file_path: Path,
    read_creation_datetime: bool = False,
    read_post_sid: bool = False,
    sid_num_levels: int = 6,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if not file_path or not os.path.exists(file_path):
        rank_logger.error(f"Global ids file not found: {file_path}")
        return None, None, None, None

    start = time.time()
    rank_logger.info(f"Loading global ids from {file_path}")

    columns = ["post_id", "author_id"]
    if read_creation_datetime:
        columns.append("created_at")
    if read_post_sid:
        columns.append("post_sid")
    table = pq.read_table(str(file_path), columns=columns)
    post_ids = np.asarray(table.column("post_id").to_numpy(zero_copy_only=False), dtype=np.uint64)
    author_ids = np.asarray(
        table.column("author_id").to_numpy(zero_copy_only=False), dtype=np.uint64
    )
    post_creation_datetimes = None
    if read_creation_datetime:
        created_at = table.column("created_at").to_numpy(zero_copy_only=False)
        post_creation_datetimes = np.asarray(created_at, dtype="datetime64[ms]")

    post_sids: np.ndarray | None = None
    if read_post_sid:
        n = len(table)
        sid_col = table.column("post_sid").combine_chunks()
        offsets = sid_col.offsets.to_numpy(zero_copy_only=False)
        flat_values = sid_col.values.to_numpy(zero_copy_only=False)
        expected_total = n * sid_num_levels
        if n > 0 and (
            flat_values.size != expected_total or offsets[-1] - offsets[0] != expected_total
        ):
            raise ValueError(
                f"post_sid schema invariant violated in {file_path}: "
                f"expected {n} rows × {sid_num_levels} codes = {expected_total} flat ints, "
                f"got flat_values.size={flat_values.size}, offsets span "
                f"{int(offsets[-1] - offsets[0]) if offsets.size else 0}"
            )
        post_sids = np.ascontiguousarray(flat_values.reshape(-1, sid_num_levels), dtype=np.int32)
        n_with = int((post_sids[:, 0] != -1).sum()) if n > 0 else 0
        rank_logger.info(
            f"Packed post_sid for {n_with:,}/{n:,} rows ({n_with / max(n, 1):.1%}) into [{n}, {sid_num_levels}] int32"
        )

    if len(post_ids) == 0 or len(author_ids) == 0 or len(post_ids) != len(author_ids):
        rank_logger.info(
            f"Global ids file is empty or has mismatched post and author ids or creation datetimes: {file_path}"
        )
        return None, None, None, None

    rank_logger.info(
        f"Loaded {len(post_ids):,} global ids from {file_path} in {time.time() - start:.3f} seconds"
    )
    return post_ids, author_ids, post_creation_datetimes, post_sids


class _IdListSourceCounter:
    def __init__(self, every: int = 1000):
        self._every: int = every
        self._counts: dict[str, int] = {"sidecar": 0, "native": 0, "none": 0}
        self._seen: set[str] = set()

    def note(self, fname: str, source: str) -> None:
        self._counts[source] += 1
        total = sum(self._counts.values())
        if source not in self._seen:
            self._seen.add(source)
            rank_logger.info(f"id-list features: first file with source={source}: {fname}")
        if total % self._every == 0:
            rank_logger.info(f"id-list features: sources after {total} files: {self._counts}")

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)


class LazyRecordBatchIterator:
    def __init__(
        self,
        pf: pq.ParquetFile,
        batch_size: int,
        fname: str,
        conversion_delay_columns: list[str] | None = None,
        include_action_delay_columns: bool = False,
        id_list_sidecar_root: str | None = None,
        id_list_prefer_sidecar: bool = True,
        id_list_sidecar_strict: bool = False,
        id_list_sources: _IdListSourceCounter | None = None,
    ):
        self.pf: pq.ParquetFile = pf
        self.iter: Iterator[pa.RecordBatch] | None = None
        self.num_rows: int = pf.metadata.num_rows
        self.rows_to_skip: int = 0
        self.batch_size: int = batch_size
        self.fname: str = fname
        self.conversion_delay_columns: list[str] | None = conversion_delay_columns
        self.include_action_delay_columns: bool = include_action_delay_columns
        self._sidecar_delays: dict[str, np.ndarray] | None = None
        self.id_list_sidecar_root: str | None = id_list_sidecar_root
        self.id_list_prefer_sidecar: bool = id_list_prefer_sidecar
        self.id_list_sidecar_strict: bool = id_list_sidecar_strict
        self._id_list_sources: _IdListSourceCounter | None = id_list_sources
        self._id_list_sidecar: pa.Array | None = None
        self._id_list_source: str | None = None
        self._row_pos: int = 0

    def seek(self):
        arrow_schema = self.pf.schema_arrow
        excluded_columns = ["firstPageSeq"]
        valid_columns = [name for name in arrow_schema.names if name not in excluded_columns]
        self.iter = self.pf.iter_batches(self.batch_size, columns=valid_columns)
        assert self.iter is not None

        if self.conversion_delay_columns:
            sidecar = conversion_labels.sidecar_path_for(self.fname)
            columns = list(self.conversion_delay_columns)
            if self.include_action_delay_columns:
                columns += conversion_labels.action_delay_columns(sidecar)
            self._sidecar_delays = conversion_labels.load_sidecar_delays(sidecar, columns)
            for name, mat in self._sidecar_delays.items():
                if mat.shape[0] != self.num_rows:
                    raise ValueError(
                        f"sidecar {sidecar} column {name} has {mat.shape[0]} rows, "
                        f"batch file has {self.num_rows}"
                    )

        if self.id_list_sidecar_root is not None:
            has_native = id_lists.COLUMN in arrow_schema.names
            sidecar = id_lists.sidecar_path_for(self.fname, self.id_list_sidecar_root)
            has_sidecar = os.path.exists(sidecar)
            if self.id_list_sidecar_strict and not has_sidecar:
                raise FileNotFoundError(f"id-list sidecar missing for {self.fname}: {sidecar}")
            if has_sidecar and (self.id_list_prefer_sidecar or not has_native):
                self._id_list_sidecar = id_lists.canonical_column(
                    id_lists.load_sidecar_column(sidecar, expected_rows=self.num_rows)
                )
                self._id_list_source = "sidecar"
            else:
                self._id_list_source = "native" if has_native else "none"
            if self._id_list_sources is not None:
                self._id_list_sources.note(self.fname, self._id_list_source)

        cnt = 0
        while cnt < self.rows_to_skip:
            skipped = next(self.iter)
            cnt += self.batch_size
            self._row_pos += skipped.num_rows

    def read(self) -> pa.RecordBatch:
        if self.rows_to_skip >= self.num_rows:
            raise StopIteration

        if self.iter is None:
            self.seek()

        assert self.iter is not None
        batch = next(self.iter)
        if self._sidecar_delays is not None:
            window = {
                name: mat[self._row_pos : self._row_pos + batch.num_rows]
                for name, mat in self._sidecar_delays.items()
            }
            batch = conversion_labels.attach_delays(batch, window)
        if self._id_list_source == "sidecar":
            assert self._id_list_sidecar is not None
            batch = id_lists.set_column(
                batch, self._id_list_sidecar.slice(self._row_pos, batch.num_rows)
            )
        elif self._id_list_source == "native":
            batch = id_lists.set_column(
                batch, id_lists.canonical_column(batch.column(id_lists.COLUMN))
            )
        elif self._id_list_source == "none":
            batch = id_lists.set_column(batch, id_lists.null_column(batch.num_rows))
        self._row_pos += batch.num_rows
        return batch

    def skip_batch(self):
        if self.rows_to_skip < self.num_rows:
            self.rows_to_skip += self.batch_size
            return True
        return False


class PrereadRecordBatchIterator:
    def __init__(self, batch_iterator, batches: list[pa.RecordBatch]) -> None:
        self._batches = deque(batches)
        self.fname: str = batch_iterator.fname

    def read(self) -> pa.RecordBatch:
        if not self._batches:
            raise StopIteration
        return self._batches.popleft()

    def skip_batch(self) -> bool:
        if not self._batches:
            return False
        self._batches.popleft()
        return True


class RowSplitRecordBatchIterator:
    def __init__(
        self,
        partition_batch_iterator: LazyRecordBatchIterator,
        shards_per_partition: int,
        rank_in_partition: int,
    ):
        assert 0 <= rank_in_partition < shards_per_partition
        self._partition_batch_iterator = partition_batch_iterator
        self._shards_per_partition = shards_per_partition
        self._rank_in_partition = rank_in_partition
        self.fname: str = partition_batch_iterator.fname

    def read(self) -> pa.RecordBatch:
        batch = self._partition_batch_iterator.read()
        usable = batch.num_rows - batch.num_rows % self._shards_per_partition
        rows = np.arange(self._rank_in_partition, usable, self._shards_per_partition)
        return batch.take(pa.array(rows))

    def skip_batch(self) -> bool:
        return self._partition_batch_iterator.skip_batch()


def _resolve_file_path(base_path: str, file_entry: str) -> str:
    if os.path.isabs(file_entry):
        return file_entry
    return os.path.join(base_path, file_entry)


_PARTITION_BATCH_HIER_RE = re.compile(r"partition=(\d+)/\d+/batch_(\d+)\.parquet$")
_PARTITION_BATCH_FLAT_RE = re.compile(r"partition=(\d+)/batch_(\d+)\.parquet$")
_PARTITION_DATA_RE = re.compile(r"partition=(\d+)/data(\d+)\.parquet$")


def _match_partition_file(file_path: str) -> re.Match[str] | None:
    return (
        _PARTITION_BATCH_HIER_RE.search(file_path)
        or _PARTITION_BATCH_FLAT_RE.search(file_path)
        or _PARTITION_DATA_RE.search(file_path)
    )


def _extract_batch_id(file_path: str) -> int | None:
    m = _match_partition_file(file_path)
    if m:
        return int(m.group(2))
    return None


def _extract_partition_id(file_path: str) -> int | None:
    m = _match_partition_file(file_path)
    if m:
        return int(m.group(1))
    return None


_SHUFFLE_READ_CHUNK_ROWS = 1 << 16
_SHUFFLE_PREFETCH_FILES = 2


def _shuffle_window_end(window_start: int, window: int) -> int:
    return (window_start // window + 1) * window


def _shuffled_remaining_batches(
    *,
    start_bid: int,
    end_bid: int,
    window: int,
    rows_per_bid: int | np.ndarray,
    batch_size: int,
    resume_bid: int | None = None,
    resume_rows: int = 0,
) -> int:
    window_start, skip = start_bid, 0
    if resume_bid is not None and resume_bid >= start_bid:
        window_start, skip = resume_bid, resume_rows
    total = 0
    while window_start < end_bid:
        window_end = min(_shuffle_window_end(window_start, window), end_bid)
        if isinstance(rows_per_bid, np.ndarray):
            window_rows = int(rows_per_bid[window_start - start_bid : window_end - start_bid].sum())
        else:
            window_rows = (window_end - window_start) * rows_per_bid
        rows = max(0, window_rows - skip)
        skip = 0
        full, rem = divmod(rows, batch_size)
        total += full + (1 if rem * 2 >= batch_size and rem > 0 else 0)
        window_start = window_end
    return total


class ShuffleMode(str, enum.Enum):
    description: str

    def __new__(cls, value: str, description: str) -> "ShuffleMode":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    NONE = (
        "none",
        "Chronological. For compatibility, shuffle_window_time_slices > 0 still selects "
        "FILE_SHUFFLE.",
    )
    FILE_SHUFFLE = (
        "file_shuffle",
        "Each tumbling window of shuffle_window_time_slices batch_ids is read in a random "
        "file order per shard, with rows optionally mixed through "
        "shuffle_in_memory_buffer_rows.",
    )
    BLOCK_SHUFFLE = (
        "block_shuffle",
        "Each tumbling window of shuffle_window_time_slices batch_ids is cut into blocks of "
        "block_shuffle_block_time_slices (e.g. 480 = 24 blocks of 20, ~1 day of ~1-hour blocks on "
        "ads records); blocks are visited in a random order, the same on every shard unless "
        "block_shuffle_order_per_shard, and each block is read in time order.",
    )


def _block_shuffle_order(
    seed: int, window_start: int, window_blocks: int, shard: int | None
) -> list[int]:
    tag = f"{seed}:{window_start}:{'-' if shard is None else shard}"
    return sorted(
        range(window_blocks),
        key=lambda j: (hashlib.blake2b(f"{tag}:{j}".encode(), digest_size=8).digest(), j),
    )


_BLOCK_LOOKAHEAD_FILES = 4


def _block_lookahead_plan(files_per_batch_id: int) -> list[int]:
    files_per_batch_id = max(1, files_per_batch_id)
    plan, budget = [], max(0, _BLOCK_LOOKAHEAD_FILES)
    while budget > 0:
        plan.append(min(files_per_batch_id, budget))
        budget -= plan[-1]
    return plan


def _block_shuffle_fingerprint(
    *,
    window_start: int,
    block: int,
    window_blocks: int,
    seed: int,
    per_shard: bool,
    num_shards: int,
) -> str:
    first = _block_shuffle_order(seed, window_start, window_blocks, 0 if per_shard else None)
    key = f"{block}:{window_blocks}:{num_shards if per_shard else '-'}:{first}"
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()


def _block_shuffle_ranges(
    *,
    window_start: int,
    block: int,
    window_blocks: int,
    seed: int,
    lo: int,
    hi: int,
    shard: int | None = None,
) -> list[tuple[int, int]]:
    order = _block_shuffle_order(seed, window_start, window_blocks, shard)
    blocks = [(window_start + j * block, window_start + (j + 1) * block) for j in order]
    return [(max(b0, lo), min(b1, hi)) for b0, b1 in blocks if max(b0, lo) < min(b1, hi)]


def _block_shuffle_sequence(
    *,
    window_start: int,
    block: int,
    window_blocks: int,
    seed: int,
    shard: int | None,
    lo: int,
    hi: int,
) -> list[int]:
    ranges = _block_shuffle_ranges(
        window_start=window_start,
        block=block,
        window_blocks=window_blocks,
        seed=seed,
        lo=lo,
        hi=hi,
        shard=shard,
    )
    return [bid for b0, b1 in ranges for bid in range(b0, b1)]


def _replay_block_segments(
    *,
    window_start: int,
    block: int,
    window_blocks: int,
    seed: int,
    shard: int | None,
    segments: list[list[int]],
) -> tuple[set[int], list[int]]:
    done: set[int] = set()
    left: list[int] = []
    for lo, hi, n in segments:
        seq = _block_shuffle_sequence(
            window_start=window_start,
            block=block,
            window_blocks=window_blocks,
            seed=seed,
            shard=shard,
            lo=lo,
            hi=hi,
        )
        seq = [bid for bid in seq if bid not in done]
        done.update(seq[:n])
        left = seq[n:]
    return done, left


def _block_window_grew(metadata_path: str, end_batch_id: int | None, hi: int) -> bool:
    meta = _load_valid_batches_metadata(metadata_path)
    if meta is None:
        return False
    end = meta["max_valid_batch"] + 1
    return (end if end_batch_id is None else min(end, end_batch_id)) > hi


@final
class _BlockLog:
    def __init__(
        self,
        *,
        shard_index: int,
        window_start: int,
        window: int,
        block: int,
        order: list[tuple[int, int]],
        segments: list[list[int]],
    ):
        self._shard, self._block, self._num_blocks = shard_index, block, len(order)
        self._starts = {b0 // block: (i, b0, b1) for i, (b0, b1) in enumerate(order, 1)}
        self._reading: int | None = None
        rank_logger.info(
            "Shard %d: block-shuffle window [%d, %d), segments %s, block order %s",
            shard_index,
            window_start,
            window_start + window,
            segments,
            order,
        )

    def batch_id(self, bid: int) -> None:
        key = bid // self._block
        if key != self._reading and key in self._starts:
            self._reading = key
            i, b0, b1 = self._starts[key]
            rank_logger.info(
                "Shard %d: reading block %d/%d, batch_ids [%d, %d)",
                self._shard,
                i,
                self._num_blocks,
                b0,
                b1,
            )


class _BlockResume(NamedTuple):
    window_start: int
    segments: list[list[int]]
    reads: int
    drop_in_progress: bool
    rows_in_file: int | None = None
    next_file_rows: int = 0


def _block_shuffle_resume(
    position: DataPosition,
    *,
    batch_size: int,
    num_shards: int,
    block: int,
    window_blocks: int,
    seed: int,
    per_shard: bool,
    start_batch_id: int,
    end_batch_id: int | None,
) -> _BlockResume | None:
    window = block * window_blocks
    reads = position["rows_read_in_batch"]
    saved_bs = position.get("batch_size")
    drop = saved_bs is not None and saved_bs != batch_size
    saved_ws = position.get("block_shuffle_window_start")
    saved_segments = position.get("block_shuffle_segments")
    if saved_ws is not None and saved_segments:
        segments = [list(seg) for seg in saved_segments]
        saved_shards = position.get("block_shuffle_num_shards")
        shards_changed = saved_shards is not None and saved_shards != num_shards
        saved_fp = position.get("block_shuffle_fingerprint")
        fp = _block_shuffle_fingerprint(
            window_start=saved_ws,
            block=block,
            window_blocks=window_blocks,
            seed=seed,
            per_shard=per_shard,
            num_shards=num_shards,
        )
        if (saved_fp is not None and saved_fp != fp) or (per_shard and shards_changed):
            end_read = max(hi for _, hi, _ in segments)
            rank_logger.warning(
                "Block order changed since the checkpoint; treating batch_ids before %d as read",
                end_read,
            )
            ws = (end_read // window) * window
            return _BlockResume(ws, [[ws, end_read, end_read - ws]], 0, False)
        return _BlockResume(
            saved_ws,
            segments,
            reads,
            drop or shards_changed,
            position.get("block_shuffle_rows_in_file"),
            position.get("block_shuffle_rows_in_next_file", 0),
        )
    bid = position["last_batch_id"]
    if bid < start_batch_id:
        return None
    if end_batch_id is not None and bid >= end_batch_id:
        ws = (end_batch_id // window) * window
        return _BlockResume(ws, [[ws, end_batch_id, end_batch_id - ws]], 0, False)
    ws = (bid // window) * window
    if saved_bs == 1:
        return _BlockResume(ws, [[ws, ws + window, window]], 0, False)
    return _BlockResume(ws, [[ws, bid, bid - ws], [bid, bid + 1, 0]], reads, drop)


def _block_resume_window(
    *,
    segments: list[list[int]],
    in_progress: bool,
    drop_in_progress: bool,
    window_start: int,
    lo: int,
    hi: int,
    block: int,
    window_blocks: int,
    seed: int,
    shard: int | None,
    num_shards: int,
    next_file_rows: int = 0,
) -> tuple[list[list[int]], set[int], int | None, int]:
    replay = functools.partial(
        _replay_block_segments,
        window_start=window_start,
        block=block,
        window_blocks=window_blocks,
        seed=seed,
        segments=segments,
    )
    done, left = replay(shard=shard)
    current: int | None = None
    if in_progress and left:
        drop = drop_in_progress or not lo <= left[0] < hi
        if not drop and shard is not None:
            for s in range(num_shards):
                s_left = replay(shard=s)[1]
                if s_left and not lo <= s_left[0] < hi:
                    drop = True
                    break
        if drop:
            rank_logger.warning("Dropping the rest of in-progress batch_id %d", left[0])
            segments[-1][2] += 1
            done.add(left[0])
        else:
            current = left[0]
    if shard is not None and lo > window_start:
        left_to_read = {
            sum(b not in replay(shard=s)[0] for b in range(lo, hi)) for s in range(num_shards)
        }
        if len(left_to_read) > 1:
            window = block * window_blocks
            rank_logger.warning(
                "TTL removed unequal unread data per shard in window [%d, %d); "
                "treating the rest of it as read",
                window_start,
                window_start + window,
            )
            segments.append([window_start, window_start + window, window])
            done = set(range(window_start, window_start + window))
            current = None
    if next_file_rows and len(left) > 1 and left[1] not in done:
        overflow = left[1]
        sequence = _block_shuffle_sequence(
            window_start=window_start,
            block=block,
            window_blocks=window_blocks,
            seed=seed,
            shard=shard,
            lo=lo,
            hi=hi,
        )
        upcoming = next((b for b in sequence if b not in done and b != left[0]), None)
        if upcoming != overflow:
            rank_logger.warning(
                "Window grew before batch_id %d, whose first rows were read; dropping it",
                overflow,
            )
            if lo <= overflow < hi:
                segments.insert(0, [overflow, overflow + 1, 1])
                done.add(overflow)
            next_file_rows = 0
    else:
        next_file_rows = 0
    return segments, done, current, next_file_rows


def _same_schema(a: pa.RecordBatch, b: pa.RecordBatch) -> bool:
    return a.schema.equals(b.schema, check_metadata=False)


@final
class _RowShuffleBuffer:
    def __init__(self, capacity_rows: int, rng: np.random.Generator):
        self._capacity = capacity_rows
        self._rng = rng
        self._batches: list[pa.RecordBatch] = []
        self._rows = 0

    def add(self, batch: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
        if self._capacity == 0:
            yield batch
            return
        if batch.num_rows == 0:
            return
        if self._batches and not _same_schema(batch, self._batches[0]):
            yield from self.flush()
        self._batches.append(batch)
        self._rows += batch.num_rows
        if self._rows >= self._capacity:
            yield from self._emit(keep=0)

    def flush(self) -> Iterator[pa.RecordBatch]:
        if self._rows > 0:
            yield from self._emit(keep=0)

    def _emit(self, keep: int) -> Iterator[pa.RecordBatch]:
        merged = pa.concat_batches(self._batches)
        merged = merged.take(pa.array(self._rng.permutation(merged.num_rows)))
        n_emit = merged.num_rows - keep
        rest = merged.slice(n_emit)
        self._batches = [rest] if rest.num_rows else []
        self._rows = rest.num_rows
        yield merged.slice(0, n_emit)


@final
class _Rebatcher:
    def __init__(self, batch_size: int):
        self._batch_size = batch_size
        self._pending: list[pa.RecordBatch] = []
        self._rows = 0

    def add(self, chunk: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
        if chunk.num_rows == 0:
            return
        if self._pending and not _same_schema(chunk, self._pending[0]):
            yield from self.flush()
        self._pending.append(chunk)
        self._rows += chunk.num_rows
        if self._rows < self._batch_size:
            return
        merged = pa.concat_batches(self._pending)
        n_full = (merged.num_rows // self._batch_size) * self._batch_size
        rest = merged.slice(n_full)
        self._pending = [rest] if rest.num_rows else []
        self._rows = rest.num_rows
        for i in range(0, n_full, self._batch_size):
            yield pa.concat_batches([merged.slice(i, self._batch_size)])

    def flush(self) -> Iterator[pa.RecordBatch]:
        if self._rows > 0:
            merged = pa.concat_batches(self._pending)
            self._pending = []
            self._rows = 0
            yield merged


@final
class InterleavingRecordBatchProvider:
    def __init__(
        self,
        *,
        index_path: str | None = None,
        metadata_path: str | None = None,
        topic_dir: str | None = None,
        batch_size: int,
        num_shards: int,
        shard_index: int,
        interleave_k: int,
        num_kafka_partitions: int,
        skip_rows: int = 0,
        date_range: tuple[str, str] | None = None,
        continuous: bool = False,
        poll_interval_s: float = 60.0,
        resume_position: DataPosition | None = None,
        min_timestamp_ms: int | None = None,
        max_timestamp_ms: int | None = None,
        conversion_delay_columns: list[str] | None = None,
        include_action_delay_columns: bool = False,
        shuffle_window_time_slices: int = 0,
        shuffle_in_memory_buffer_rows: int = 0,
        shuffle_seed: int = 0,
        shuffle_mode: ShuffleMode | str = ShuffleMode.NONE,
        block_shuffle_block_time_slices: int = 0,
        block_shuffle_order_per_shard: bool = False,
        id_list_sidecar_root: str | None = None,
        id_list_prefer_sidecar: bool = True,
        id_list_sidecar_strict: bool = False,
        end_batch_id: int | None = None,
    ):
        self._conversion_delay_columns = conversion_delay_columns
        self._include_action_delay_columns = include_action_delay_columns
        self._id_list_sidecar_root = id_list_sidecar_root
        self._id_list_prefer_sidecar = id_list_prefer_sidecar
        self._id_list_sidecar_strict = id_list_sidecar_strict
        self._id_list_sources = _IdListSourceCounter() if id_list_sidecar_root is not None else None
        if metadata_path is None and index_path is None:
            raise ValueError("Either metadata_path or index_path must be provided")

        if shuffle_window_time_slices < 0 or shuffle_in_memory_buffer_rows < 0:
            raise ValueError(
                f"shuffle_window_time_slices ({shuffle_window_time_slices}) and "
                f"shuffle_in_memory_buffer_rows ({shuffle_in_memory_buffer_rows}) must be >= 0"
            )
        if shuffle_in_memory_buffer_rows > 0 and shuffle_window_time_slices == 0:
            raise ValueError(
                "shuffle_in_memory_buffer_rows requires shuffle_window_time_slices > 0"
            )
        if shuffle_window_time_slices > 0:
            if metadata_path is None:
                raise ValueError("shuffle requires metadata mode (.valid_batches.json)")
            if continuous:
                raise ValueError("shuffle is not supported with continuous=True")
        try:
            shuffle_mode = ShuffleMode(shuffle_mode)
        except ValueError:
            raise ValueError(
                f"shuffle_mode must be one of {[m.value for m in ShuffleMode]}, "
                f"got {shuffle_mode!r}"
            ) from None
        if shuffle_mode == ShuffleMode.FILE_SHUFFLE and shuffle_window_time_slices <= 0:
            raise ValueError("file_shuffle needs shuffle_window_time_slices > 0")
        if shuffle_mode == ShuffleMode.BLOCK_SHUFFLE:
            if block_shuffle_block_time_slices <= 0 or shuffle_window_time_slices <= 0:
                raise ValueError(
                    f"block_shuffle needs block_shuffle_block_time_slices ({block_shuffle_block_time_slices}) "
                    f"and shuffle_window_time_slices ({shuffle_window_time_slices}) > 0"
                )
            if shuffle_window_time_slices % block_shuffle_block_time_slices:
                raise ValueError(
                    f"shuffle_window_time_slices ({shuffle_window_time_slices}) must be a "
                    f"multiple of block_shuffle_block_time_slices ({block_shuffle_block_time_slices})"
                )
            if shuffle_in_memory_buffer_rows:
                raise ValueError(
                    "block_shuffle reads each block in time order; leave "
                    "shuffle_in_memory_buffer_rows at 0"
                )
            if metadata_path is None:
                raise ValueError("block_shuffle requires metadata mode (.valid_batches.json)")
            if continuous:
                raise ValueError("block_shuffle is not supported with continuous=True")
            if num_kafka_partitions % num_shards and num_shards % num_kafka_partitions:
                rank_logger.warning(
                    "block_shuffle: %d partitions over %d shards gives shards unequal file "
                    "counts, so they drift apart and resume from one checkpoint is approximate",
                    num_kafka_partitions,
                    num_shards,
                )
        elif block_shuffle_order_per_shard:
            raise ValueError("block_shuffle_order_per_shard requires shuffle_mode=block_shuffle")
        self._shuffle_window_time_slices = shuffle_window_time_slices
        self._shuffle_in_memory_buffer_rows = shuffle_in_memory_buffer_rows
        self._shuffle_seed = shuffle_seed
        self._shuffle_mode = shuffle_mode
        self._block_shuffle_block_time_slices = block_shuffle_block_time_slices
        self._shuffle_window_blocks = (
            shuffle_window_time_slices // block_shuffle_block_time_slices
            if shuffle_mode == ShuffleMode.BLOCK_SHUFFLE
            else 0
        )
        self._block_shuffle_order_per_shard = block_shuffle_order_per_shard

        if resume_position is not None and metadata_path is None:
            raise ValueError(
                "resume_position is only supported in metadata mode (.valid_batches.json)"
            )

        has_time_range = min_timestamp_ms is not None or max_timestamp_ms is not None
        if has_time_range and metadata_path is None:
            raise ValueError(
                "min_timestamp_ms/max_timestamp_ms require metadata mode (.valid_batches.json)"
            )

        self._index_path = index_path
        self._metadata_path = metadata_path
        if metadata_path is not None:
            if topic_dir is None:
                topic_dir = str(Path(metadata_path).parent)
            topic_dir = os.path.abspath(topic_dir)
            self._topic_dir = topic_dir
        self._path = str(Path(index_path).parent) if index_path else topic_dir or ""
        self._batch_size = batch_size
        self._num_shards = num_shards
        self._shard_index = shard_index
        self._interleave_k = interleave_k
        self._shards_per_partition = 1
        if num_kafka_partitions and num_shards > num_kafka_partitions:
            if num_shards % num_kafka_partitions:
                raise ValueError(
                    f"{num_shards} shards over {num_kafka_partitions} partitions: the shard "
                    "count must be a multiple of the partition count"
                )
            if shuffle_mode != ShuffleMode.BLOCK_SHUFFLE:
                raise ValueError(
                    f"{num_shards} shards over {num_kafka_partitions} partitions splits rows, "
                    "which only shuffle_mode=block_shuffle supports"
                )
            self._shards_per_partition = num_shards // num_kafka_partitions
        self._num_partition_owners = num_shards // self._shards_per_partition
        self._partition_owner_index = shard_index // self._shards_per_partition
        self._rank_in_partition = shard_index % self._shards_per_partition
        self._date_range = date_range
        self._continuous = continuous
        self._poll_interval_s = poll_interval_s
        self._num_kafka_partitions = num_kafka_partitions

        self._end_batch_id: int | None = None
        start_batch_id = 0

        self._remaining_skips: int = 0

        if has_time_range:
            assert metadata_path is not None
            meta = _load_valid_batches_metadata(metadata_path)
            if meta is None:
                raise ValueError(f"Cannot load metadata from {metadata_path}")
            start_batch_id, range_end_batch_id = _resolve_time_range(
                self._topic_dir,
                meta["min_valid_batch"],
                meta["max_valid_batch"],
                min_timestamp_ms,
                max_timestamp_ms,
            )
            if max_timestamp_ms is not None:
                self._end_batch_id = range_end_batch_id
        if end_batch_id is not None:
            self._end_batch_id = (
                end_batch_id
                if self._end_batch_id is None
                else min(self._end_batch_id, end_batch_id)
            )

        self._shuffle_skip_rows: int = 0
        self._shuffle_skip_spans_windows: bool = False
        self._rows_in_window: int = 0
        self._range_start_batch_id: int = start_batch_id
        self._block_resume: tuple[int, list[list[int]]] | None = None
        self._block_drop_in_progress: bool = False
        self._block_next_file_rows: int = 0
        self._block_in_file: bool = False
        self._block_state: tuple[int, list[list[int]], str] | None = None

        block_resume = (
            _block_shuffle_resume(
                resume_position,
                batch_size=batch_size,
                num_shards=num_shards,
                block=block_shuffle_block_time_slices,
                window_blocks=self._shuffle_window_blocks,
                seed=shuffle_seed,
                per_shard=block_shuffle_order_per_shard,
                start_batch_id=start_batch_id,
                end_batch_id=self._end_batch_id,
            )
            if resume_position is not None and shuffle_mode == ShuffleMode.BLOCK_SHUFFLE
            else None
        )
        if resume_position is not None and block_resume is not None:
            self._block_resume = (block_resume.window_start, block_resume.segments)
            self._next_batch_id = block_resume.window_start
            self._record_batches_to_skip = block_resume.reads
            rows_in_file = block_resume.rows_in_file
            saved_bs = resume_position.get("batch_size") or batch_size
            self._block_drop_in_progress = block_resume.drop_in_progress or (
                rows_in_file is not None and rows_in_file % saved_bs != 0
            )
            self._block_next_file_rows = block_resume.next_file_rows
            self._block_in_file = bool(rows_in_file)
        elif resume_position is not None:
            resume_bid = resume_position["last_batch_id"]
            resume_in_range = resume_bid >= start_batch_id and (
                self._end_batch_id is None or resume_bid < self._end_batch_id
            )
            if resume_in_range:
                self._next_batch_id = resume_bid
                saved_reads = resume_position["rows_read_in_batch"]
                saved_batch_size = resume_position.get("batch_size")

                if saved_batch_size is not None and saved_batch_size != batch_size:
                    assert saved_batch_size > 0, (
                        f"saved_batch_size must be positive, got {saved_batch_size}"
                    )
                    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
                    total_rows = saved_reads * saved_batch_size
                    adjusted_reads = total_rows // batch_size
                    rank_logger.info(
                        "Adjusting data resume skip count for batch_size change: "
                        "saved_reads=%d * saved_batch_size=%d = %d rows, "
                        "new skip = %d reads * batch_size=%d = %d rows "
                        "(remainder %d rows will be re-read)",
                        saved_reads,
                        saved_batch_size,
                        total_rows,
                        adjusted_reads,
                        batch_size,
                        adjusted_reads * batch_size,
                        total_rows - adjusted_reads * batch_size,
                    )
                    self._record_batches_to_skip = adjusted_reads
                else:
                    self._record_batches_to_skip = saved_reads
                self._shuffle_skip_rows = saved_reads * (saved_batch_size or batch_size)

                rank_logger.info(
                    "Resuming from DataPosition: batch_id=%d, rows_read_in_batch=%d, "
                    "saved_batch_size=%s, current_batch_size=%d, "
                    "num_shards=%d, shard_index=%d, interleave_k=%d, "
                    "effective_skip=%d reads (%d rows)",
                    resume_bid,
                    saved_reads,
                    saved_batch_size,
                    batch_size,
                    num_shards,
                    shard_index,
                    self._interleave_k,
                    self._record_batches_to_skip,
                    self._record_batches_to_skip * batch_size,
                )
            elif resume_bid < start_batch_id:
                self._next_batch_id = start_batch_id
                self._record_batches_to_skip = 0
                rank_logger.warning(
                    "Resume batch_id %d is before start_batch_id %d; "
                    "starting from range start (ignoring rows_read_in_batch)",
                    resume_bid,
                    start_batch_id,
                )
            else:
                self._next_batch_id = self._end_batch_id or resume_bid
                self._record_batches_to_skip = 0
                rank_logger.warning(
                    "Resume batch_id %d is past end_batch_id %s; range already exhausted",
                    resume_bid,
                    self._end_batch_id,
                )
        else:
            self._next_batch_id = start_batch_id
            self._record_batches_to_skip = skip_rows // (batch_size * num_shards)
            self._shuffle_skip_rows = self._record_batches_to_skip * batch_size
            self._shuffle_skip_spans_windows = True
            rank_logger.info(
                f"Skipping {self._record_batches_to_skip=}: {skip_rows=} {batch_size=} {num_shards=}"
            )

        self._current_drain_batch_id: int = self._next_batch_id
        self._reads_in_current_batch: int = 0

    def _get_ready_batches(self) -> list[list[str]]:
        if self._metadata_path is not None:
            return self._get_ready_batches_from_metadata()
        return self._get_ready_batches_from_index()

    def _get_ready_batches_from_metadata(self) -> list[list[str]]:
        assert self._metadata_path is not None
        meta = _load_valid_batches_metadata(self._metadata_path)
        if meta is None:
            return []

        min_batch = meta["min_valid_batch"]
        max_batch = meta["max_valid_batch"]
        num_partitions = meta["num_partitions"]

        if self._end_batch_id is not None:
            max_batch = min(max_batch, self._end_batch_id - 1)

        start = max(min_batch, self._next_batch_id)
        if start > max_batch:
            return []

        ready: list[list[str]] = []
        for bid in range(start, max_batch + 1):
            my_files = [
                _batch_path(self._topic_dir, p, bid)
                for p in range(num_partitions)
                if p % self._num_partition_owners == self._partition_owner_index
            ]
            ready.append(my_files)
            self._next_batch_id = bid + 1

        return ready

    def _get_ready_batches_from_index(self) -> list[list[str]]:
        index_path = self._index_path
        if index_path is None or not os.path.isfile(index_path):
            raise ValueError(f"Index file {index_path} not found")

        with open(index_path) as f:
            all_files = [line.strip() for line in f if line.strip()]

        if self._date_range is not None:
            start_str, end_str = self._date_range
            start_date = parse_date_time(start_str) if start_str.lower() != "none" else None
            end_date = parse_date_time(end_str) if end_str.lower() != "none" else None
            if start_date is not None or end_date is not None:
                filtered: list[str] = []
                for file in all_files:
                    try:
                        file_date = extract_datetime_from_file_name(file)
                        if start_date is not None and file_date < start_date:
                            continue
                        if end_date is not None and file_date > end_date:
                            continue
                        filtered.append(file)
                    except (AssertionError, ValueError):
                        filtered.append(file)
                all_files = filtered

        batch_to_files: dict[int, list[str]] = {}
        for f in all_files:
            bid = _extract_batch_id(f)
            if bid is None:
                continue
            batch_to_files.setdefault(bid, []).append(f)

        ready: list[list[str]] = []
        for bid in sorted(batch_to_files.keys()):
            if bid < self._next_batch_id:
                continue
            partition_ids = {_extract_partition_id(f) for f in batch_to_files[bid]}
            if len(partition_ids) >= self._num_kafka_partitions:
                files_for_batch = batch_to_files[bid]
                my_files = [
                    f
                    for f in files_for_batch
                    if (_extract_partition_id(f) or 0) % self._num_partition_owners
                    == self._partition_owner_index
                ]
                ready.append(my_files)
                self._next_batch_id = bid + 1
            else:
                break

        return ready

    def _open_file(
        self, file: str, pool: ThreadPoolExecutor, active: deque, read_all: bool = False
    ) -> None:
        rank_logger.info(f"Worker {self._shard_index}/{self._num_shards} opening file {file}")
        path = _resolve_file_path(self._path, file)

        def _impl():
            try:
                pf = ParquetFile(path)
                holder = LazyRecordBatchIterator(
                    pf,
                    self._batch_size * self._shards_per_partition,
                    path,
                    self._conversion_delay_columns,
                    self._include_action_delay_columns,
                    id_list_sidecar_root=self._id_list_sidecar_root,
                    id_list_prefer_sidecar=self._id_list_prefer_sidecar,
                    id_list_sidecar_strict=self._id_list_sidecar_strict,
                    id_list_sources=self._id_list_sources,
                )
                if self._shards_per_partition > 1:
                    holder = RowSplitRecordBatchIterator(
                        holder, self._shards_per_partition, self._rank_in_partition
                    )
                if read_all:
                    batches = []
                    while (batch := self._safe_read(holder)) is not None:
                        batches.append(batch)
                    return PrereadRecordBatchIterator(holder, batches)
                return holder
            except Exception as e:
                if "No such file or directory" in str(e):
                    rank_logger.warning(f"Skipping missing file {file}")
                    return None
                raise ValueError(f"Error processing file {file}: {e}") from e

        active.append(pool.submit(_impl))

    @staticmethod
    def _safe_read(
        holder: "LazyRecordBatchIterator",
    ) -> pa.RecordBatch | None:
        try:
            return holder.read()
        except StopIteration:
            return None

    def _drain_files(
        self,
        files: list[str],
        *,
        pool: ThreadPoolExecutor,
        prefetched: deque[Future[LazyRecordBatchIterator | None]] | None = None,
        prefetched_batches: list[pa.RecordBatch] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        pending: deque[Future[LazyRecordBatchIterator | None]] = deque()
        ready: deque[LazyRecordBatchIterator] = deque()
        active: deque[LazyRecordBatchIterator] = deque()
        read_ahead: dict[int, Future[pa.RecordBatch | None]] = {}
        file_idx = 0

        def _submit_opens() -> None:
            nonlocal file_idx
            total_in_flight = len(pending) + len(ready) + len(active)
            while total_in_flight < self._interleave_k and file_idx < len(files):
                self._open_file(files[file_idx], pool, pending)
                file_idx += 1
                total_in_flight += 1

        def _harvest_ready() -> None:
            while pending and pending[0].done():
                h = pending.popleft().result()
                if h is not None:
                    ready.append(h)

        def _fill_active() -> None:
            while len(active) < self._interleave_k and ready:
                active.append(ready.popleft())

        def _fill_active_blocking() -> None:
            _fill_active()
            while len(active) < self._interleave_k and pending:
                h = pending.popleft().result()
                if h is not None:
                    active.append(h)

        if prefetched:
            pending.extend(prefetched)
            file_idx = len(prefetched)
            _submit_opens()
            _fill_active_blocking()

            if prefetched_batches:
                for batch in prefetched_batches:
                    self._reads_in_current_batch += 1
                    yield batch
        else:
            _submit_opens()
            _fill_active_blocking()

        while active:
            _submit_opens()
            _harvest_ready()

            if self._remaining_skips > 0:
                next_active: deque[LazyRecordBatchIterator] = deque()
                for holder in active:
                    if self._remaining_skips <= 0:
                        next_active.append(holder)
                        continue
                    if holder.skip_batch():
                        self._remaining_skips -= 1
                        self._reads_in_current_batch += 1
                        rank_logger.info(
                            f"Skipping batch: {holder.fname} remaining={self._remaining_skips}"
                        )
                        next_active.append(holder)
                active = next_active
                _fill_active()
                continue

            holders = list(active)
            futures = [
                read_ahead.pop(id(h), None) or pool.submit(self._safe_read, h) for h in holders
            ]
            active.clear()

            for holder, fut in zip(holders, futures):
                batch = fut.result()
                if batch is not None:
                    self._reads_in_current_batch += 1
                    active.append(holder)
                    read_ahead[id(holder)] = pool.submit(self._safe_read, holder)
                    yield batch

            _submit_opens()
            _harvest_ready()

            _fill_active()

            if len(active) < self._interleave_k and pending:
                _fill_active_blocking()

    def get_record_batches(self) -> Iterator[pa.RecordBatch]:
        if self._shuffle_mode == ShuffleMode.BLOCK_SHUFFLE:
            yield from self._get_record_batches_block_shuffle()
        elif self._shuffle_window_time_slices > 0:
            yield from self._get_record_batches_shuffled()
        else:
            yield from self._get_record_batches_synced()

    def _read_whole_file(self, file: str) -> list[pa.RecordBatch]:
        path = _resolve_file_path(self._path, file)
        try:
            holder = LazyRecordBatchIterator(
                ParquetFile(path),
                _SHUFFLE_READ_CHUNK_ROWS,
                path,
                self._conversion_delay_columns,
                self._include_action_delay_columns,
                id_list_sidecar_root=self._id_list_sidecar_root,
                id_list_prefer_sidecar=self._id_list_prefer_sidecar,
                id_list_sidecar_strict=self._id_list_sidecar_strict,
                id_list_sources=self._id_list_sources,
            )
        except Exception as e:
            if "No such file or directory" in str(e):
                rank_logger.warning(f"Skipping missing file {file}")
                return []
            raise ValueError(f"Error processing file {file}: {e}") from e
        chunks = []
        while (chunk := self._safe_read(holder)) is not None:
            chunks.append(chunk)
        return chunks

    def _shuffled_window_outputs(
        self,
        files: list[str],
        buffer: "_RowShuffleBuffer",
        pool: ThreadPoolExecutor,
        prefetch: int,
    ) -> Iterator[pa.RecordBatch]:
        rebatcher = _Rebatcher(self._batch_size)

        def _skip_then_rebatch(rows: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
            if self._shuffle_skip_rows > 0:
                n = min(self._shuffle_skip_rows, rows.num_rows)
                self._shuffle_skip_rows -= n
                self._rows_in_window += n
                rows = rows.slice(n)
            for out in rebatcher.add(rows):
                self._rows_in_window += out.num_rows
                yield out

        in_flight: deque[Future[list[pa.RecordBatch]]] = deque()
        file_iter = iter(files)
        for f in file_iter:
            in_flight.append(pool.submit(self._read_whole_file, f))
            if len(in_flight) >= prefetch:
                break
        while in_flight:
            chunks = in_flight.popleft().result()
            next_file = next(file_iter, None)
            if next_file is not None:
                in_flight.append(pool.submit(self._read_whole_file, next_file))
            for chunk in chunks:
                for rows in buffer.add(chunk):
                    yield from _skip_then_rebatch(rows)
        for rows in buffer.flush():
            yield from _skip_then_rebatch(rows)
        for out in rebatcher.flush():
            self._rows_in_window += out.num_rows
            yield out

    def _get_record_batches_shuffled(self) -> Iterator[pa.RecordBatch]:
        assert self._metadata_path is not None
        self._record_batches_to_skip = 0
        window_start = self._next_batch_id
        prefetch = min(_SHUFFLE_PREFETCH_FILES, max(1, self._interleave_k))

        pool = ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="shuffle_read_parquet")
        try:
            while True:
                meta = _load_valid_batches_metadata(self._metadata_path)
                if meta is None:
                    return
                end = meta["max_valid_batch"] + 1
                if self._end_batch_id is not None:
                    end = min(end, self._end_batch_id)
                min_valid = meta["min_valid_batch"]
                if _shuffle_window_end(window_start, self._shuffle_window_time_slices) <= min_valid:
                    if self._shuffle_skip_rows > 0:
                        rank_logger.warning(
                            "TTL advanced past window start %d (first available: %d); "
                            "clearing %d stale skip rows",
                            window_start,
                            min_valid,
                            self._shuffle_skip_rows,
                        )
                        self._shuffle_skip_rows = 0
                    window_start = max(window_start, min_valid)
                if window_start >= end:
                    return

                window_end = min(
                    _shuffle_window_end(window_start, self._shuffle_window_time_slices), end
                )
                entries = [
                    (bid, _batch_path(self._topic_dir, p, bid))
                    for bid in range(window_start, window_end)
                    for p in range(meta["num_partitions"])
                    if p % self._num_partition_owners == self._partition_owner_index
                ]
                rng = np.random.default_rng((self._shuffle_seed, self._shard_index, window_start))
                order = rng.permutation(len(entries))
                files = [entries[i][1] for i in order if entries[i][0] >= min_valid]

                self._current_drain_batch_id = window_start
                self._rows_in_window = 0
                rank_logger.info(
                    f"Shard {self._shard_index}: shuffled window [{window_start}, {window_end}), "
                    f"{len(files)} files, buffer_rows={self._shuffle_in_memory_buffer_rows}, "
                    f"skip_rows={self._shuffle_skip_rows}"
                )

                buffer = _RowShuffleBuffer(self._shuffle_in_memory_buffer_rows, rng)
                yield from self._shuffled_window_outputs(files, buffer, pool, prefetch)

                if self._shuffle_skip_rows > 0 and not self._shuffle_skip_spans_windows:
                    rank_logger.warning(
                        "Window [%d, %d) ended with %d resume skip rows left; not carrying "
                        "them into the next window",
                        window_start,
                        window_end,
                        self._shuffle_skip_rows,
                    )
                    self._shuffle_skip_rows = 0

                window_start = window_end
                self._next_batch_id = window_end
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _get_record_batches_block_shuffle(self) -> Iterator[pa.RecordBatch]:
        assert self._metadata_path is not None
        block, window_blocks = self._block_shuffle_block_time_slices, self._shuffle_window_blocks
        window = block * window_blocks
        shard = self._shard_index if self._block_shuffle_order_per_shard else None
        self._remaining_skips = self._record_batches_to_skip
        self._record_batches_to_skip = 0
        resume = self._block_resume
        window_start = resume[0] if resume else (self._next_batch_id // window) * window

        files_per_batch_id = max(1, self._interleave_k)
        pool = ThreadPoolExecutor(
            max_workers=files_per_batch_id, thread_name_prefix="block_shuffle_parquet"
        )
        lookahead_plan = _block_lookahead_plan(files_per_batch_id)
        lookahead_pool = ThreadPoolExecutor(
            max_workers=max(1, sum(lookahead_plan)),
            thread_name_prefix="block_shuffle_lookahead",
        )
        try:
            while True:
                meta = _load_valid_batches_metadata(self._metadata_path)
                if meta is None:
                    return
                end = meta["max_valid_batch"] + 1
                if self._end_batch_id is not None:
                    end = min(end, self._end_batch_id)
                if window_start >= end:
                    return
                lo = max(meta["min_valid_batch"], self._range_start_batch_id)
                hi = min(end, window_start + window)
                num_partitions = meta["num_partitions"]
                done: set[int] = set()
                segments: list[list[int]] = [[lo, hi, 0]]
                current: int | None = None
                carry = 0
                if resume is not None and resume[0] == window_start:
                    segments, done, current, next_file_rows = _block_resume_window(
                        segments=resume[1],
                        in_progress=self._remaining_skips > 0 or self._block_in_file,
                        drop_in_progress=self._block_drop_in_progress,
                        window_start=window_start,
                        lo=lo,
                        hi=hi,
                        block=block,
                        window_blocks=window_blocks,
                        seed=self._shuffle_seed,
                        shard=shard,
                        num_shards=self._num_shards,
                        next_file_rows=self._block_next_file_rows,
                    )
                    pps = max(1, num_partitions // self._num_shards)
                    carry = -(-next_file_rows // self._batch_size) * pps
                    self._remaining_skips = 0 if current is None else self._remaining_skips
                    self._block_in_file = self._block_drop_in_progress = False
                    self._block_next_file_rows = 0
                resume = None
                fingerprint = _block_shuffle_fingerprint(
                    window_start=window_start,
                    block=block,
                    window_blocks=window_blocks,
                    seed=self._shuffle_seed,
                    per_shard=shard is not None,
                    num_shards=self._num_shards,
                )
                self._block_state = (window_start, segments, fingerprint)
                block_log = _BlockLog(
                    shard_index=self._shard_index,
                    window_start=window_start,
                    window=window,
                    block=block,
                    order=_block_shuffle_ranges(
                        window_start=window_start,
                        block=block,
                        window_blocks=window_blocks,
                        seed=self._shuffle_seed,
                        lo=lo,
                        hi=hi,
                        shard=shard,
                    ),
                    segments=segments,
                )
                if current is not None:
                    block_log.batch_id(current)
                    yield from self._drain_batch_id(current, num_partitions, pool)
                    self._remaining_skips = 0
                    segments[-1][2] += 1
                    self._reads_in_current_batch = 0
                    done.add(current)
                if segments[-1][:2] != [lo, hi]:
                    segments.append([lo, hi, 0])
                sequence = _block_shuffle_sequence(
                    window_start=window_start,
                    block=block,
                    window_blocks=window_blocks,
                    seed=self._shuffle_seed,
                    shard=shard,
                    lo=lo,
                    hi=hi,
                )
                pending_bids = [bid for bid in sequence if bid not in done]
                prefetched_by_position: dict[
                    int, deque[Future[LazyRecordBatchIterator | None]]
                ] = {}
                for position, bid in enumerate(pending_bids):
                    if carry:
                        self._remaining_skips, carry = carry, 0
                    for offset, files_to_read in enumerate(lookahead_plan, start=1):
                        ahead = position + offset
                        if ahead >= len(pending_bids):
                            break
                        if ahead not in prefetched_by_position:
                            opened: deque[Future[LazyRecordBatchIterator | None]] = deque()
                            files = self._files_for_batch_id(pending_bids[ahead], num_partitions)
                            for file in files[:files_to_read]:
                                self._open_file(file, lookahead_pool, opened, read_all=True)
                            prefetched_by_position[ahead] = opened
                    prefetched = prefetched_by_position.pop(position, None)
                    block_log.batch_id(bid)
                    yield from self._drain_batch_id(
                        bid,
                        num_partitions,
                        pool,
                        prefetched=prefetched if self._remaining_skips == 0 else None,
                    )
                    segments[-1][2] += 1
                    self._reads_in_current_batch = 0
                if hi < window_start + window and _block_window_grew(
                    self._metadata_path, self._end_batch_id, hi
                ):
                    resume = (window_start, segments)
                    continue
                window_start += window
                self._next_batch_id = window_start
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            lookahead_pool.shutdown(wait=False, cancel_futures=True)

    def _files_for_batch_id(self, bid: int, num_partitions: int) -> list[str]:
        return [
            _batch_path(self._topic_dir, p, bid)
            for p in range(num_partitions)
            if p % self._num_partition_owners == self._partition_owner_index
        ]

    def _drain_batch_id(
        self,
        bid: int,
        num_partitions: int,
        pool: ThreadPoolExecutor,
        prefetched: deque[Future[LazyRecordBatchIterator | None]] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        files = self._files_for_batch_id(bid, num_partitions)
        self._current_drain_batch_id = bid
        self._reads_in_current_batch = 0
        yield from self._drain_files(files, pool=pool, prefetched=prefetched)

    def set_end_batch_id(self, end_batch_id: int) -> None:
        if self._end_batch_id is not None and end_batch_id < self._end_batch_id:
            raise ValueError(
                f"end_batch_id can only move forward: {self._end_batch_id} -> {end_batch_id}"
            )
        self._end_batch_id = end_batch_id

    def block_shuffle_window_start(self) -> int | None:
        state = self._block_state
        return None if state is None else state[0]

    def get_position(self) -> DataPosition:
        if self._shuffle_window_time_slices > 0 and self._shuffle_mode != ShuffleMode.BLOCK_SHUFFLE:
            return DataPosition(
                last_batch_id=self._current_drain_batch_id,
                rows_read_in_batch=self._rows_in_window,
                batch_size=1,
            )
        rank_logger.debug(
            "Saving data position: batch_id=%d, rows_read=%d, batch_size=%d, "
            "total_rows=%d, shard_index=%d, num_shards=%d",
            self._current_drain_batch_id,
            self._reads_in_current_batch,
            self._batch_size,
            self._reads_in_current_batch * self._batch_size,
            self._shard_index,
            self._num_shards,
        )
        position = DataPosition(
            last_batch_id=self._current_drain_batch_id,
            rows_read_in_batch=self._reads_in_current_batch,
            batch_size=self._batch_size,
        )
        state = self._block_state
        if self._shuffle_mode == ShuffleMode.BLOCK_SHUFFLE and state is not None:
            position["block_shuffle_window_start"] = state[0]
            position["block_shuffle_segments"] = [list(seg) for seg in state[1]]
            position["block_shuffle_num_shards"] = self._num_shards
            position["block_shuffle_fingerprint"] = state[2]
        return position

    def _get_record_batches_synced(self) -> Iterator[pa.RecordBatch]:
        @contextmanager
        def safe_thread_pool():
            pool = ThreadPoolExecutor(
                max_workers=self._interleave_k,
                thread_name_prefix="open_parquet_files",
            )
            yield pool
            pool.shutdown(cancel_futures=True)

        with safe_thread_pool() as pool:
            self._remaining_skips = self._record_batches_to_skip
            self._record_batches_to_skip = 0
            _resume_batch_id = self._next_batch_id

            _prefetched_files: deque[Future[LazyRecordBatchIterator | None]] = deque()
            _prefetched_batches: list[pa.RecordBatch] = []
            _prefetch_batch_id: int | None = None

            _prefetch_futures: list[Future] = []

            def _start_prefetch_async(next_files: list[str]) -> None:
                nonlocal _prefetch_futures
                _prefetch_futures.clear()

                if not next_files:
                    return

                def _open_and_read_first(
                    file_path: str,
                ) -> tuple[LazyRecordBatchIterator | None, pa.RecordBatch | None]:
                    try:
                        pf = pq.ParquetFile(file_path)
                        holder = LazyRecordBatchIterator(
                            pf,
                            self._batch_size,
                            file_path,
                            self._conversion_delay_columns,
                            self._include_action_delay_columns,
                            id_list_sidecar_root=self._id_list_sidecar_root,
                            id_list_prefer_sidecar=self._id_list_prefer_sidecar,
                            id_list_sidecar_strict=self._id_list_sidecar_strict,
                            id_list_sources=self._id_list_sources,
                        )
                        batch = holder.read()
                        return holder, batch
                    except StopIteration:
                        return holder, None
                    except Exception as e:
                        if "No such file" in str(e):
                            return None, None
                        raise

                for f in next_files[: self._interleave_k]:
                    _prefetch_futures.append(pool.submit(_open_and_read_first, f))

            def _collect_prefetch() -> tuple[deque, list]:
                nonlocal _prefetch_futures

                prefetched_files: deque[Future[LazyRecordBatchIterator | None]] = deque()
                prefetched_batches: list[pa.RecordBatch] = []

                for fut in _prefetch_futures:
                    holder, batch = fut.result()
                    if holder is not None:
                        done_fut: Future[LazyRecordBatchIterator | None] = Future()
                        done_fut.set_result(holder)
                        prefetched_files.append(done_fut)
                    if batch is not None:
                        prefetched_batches.append(batch)

                _prefetch_futures.clear()
                return prefetched_files, prefetched_batches

            while True:
                ready_batches = self._get_ready_batches()

                if ready_batches:
                    first_ready_bid = self._next_batch_id - len(ready_batches)
                    if first_ready_bid > _resume_batch_id and self._remaining_skips > 0:
                        rank_logger.warning(
                            "TTL advanced past resume batch_id %d "
                            "(first available: %d); clearing %d stale skips",
                            _resume_batch_id,
                            first_ready_bid,
                            self._remaining_skips,
                        )
                        self._remaining_skips = 0
                    _resume_batch_id = first_ready_bid

                    for i, files in enumerate(ready_batches):
                        if not files:
                            continue
                        self._current_drain_batch_id = self._next_batch_id - (
                            len(ready_batches) - ready_batches.index(files)
                        )
                        self._reads_in_current_batch = 0

                        prefetched = None
                        prefetched_batches = None
                        if _prefetch_futures and _prefetch_batch_id == self._current_drain_batch_id:
                            prefetched, prefetched_batches = _collect_prefetch()
                            _prefetch_batch_id = None

                        next_files = ready_batches[i + 1] if i + 1 < len(ready_batches) else None
                        if next_files:
                            _prefetch_batch_id = self._current_drain_batch_id + 1
                            _start_prefetch_async(next_files)

                        rank_logger.info(
                            f"Shard {self._shard_index}: draining batch "
                            f"(next_batch_id={self._next_batch_id}, "
                            f"drain_batch_id={self._current_drain_batch_id}, "
                            f"{len(files)} files for this shard, "
                            f"remaining_skips={self._remaining_skips}, "
                            f"prefetched={len(prefetched) if prefetched else 0})"
                        )
                        yield from self._drain_files(
                            files,
                            pool=pool,
                            prefetched=prefetched,
                            prefetched_batches=prefetched_batches,
                        )
                else:
                    if not self._continuous:
                        return
                    rank_logger.info(
                        f"Shard {self._shard_index}: waiting for batch "
                        f"{self._next_batch_id} to be ready across all "
                        f"{self._num_kafka_partitions} partitions, "
                        f"polling in {self._poll_interval_s}s..."
                    )
                    time.sleep(self._poll_interval_s)


def pad_batch(batch_unpadded: RecsysFeaturesBatch, batch_size: int) -> RecsysFeaturesBatch:
    num_rows = batch_unpadded["user_hashes"].shape[0]

    def pad_array(arr: np.ndarray) -> np.ndarray:
        return np.pad(
            arr,
            ((0, batch_size - num_rows),) + ((0, 0),) * (arr.ndim - 1),
        )

    def pad_post_seq(post_seq: PostSeq) -> PostSeq:
        padded = _pad_post_seq_fields(post_seq)
        if (_tcm := post_seq.get("trained_candidate_mask")) is not None:
            padded["trained_candidate_mask"] = np.pad(
                _tcm, ((0, batch_size - num_rows), (0, 0), (0, 0)), constant_values=True
            )
        if (_hem := post_seq.get("head_eligible_mask")) is not None:
            padded["head_eligible_mask"] = np.pad(
                _hem, ((0, batch_size - num_rows), (0, 0), (0, 0)), constant_values=True
            )
        for key in VALUE_LABEL_DTYPES:
            if (value_labels := post_seq.get(key)) is not None:
                padded[key] = pad_array(value_labels)
        if (ads_counts := post_seq.get("ads_count_buckets")) is not None:
            padded["ads_count_buckets"] = pad_array(ads_counts)
        if (delays := post_seq.get("conversion_delay_ms")) is not None:
            padded["conversion_delay_ms"] = np.pad(
                delays,
                ((0, batch_size - num_rows), (0, 0), (0, 0)),
                constant_values=CONVERSION_DELAY_NONE,
            )
        for key, (_, _, fill) in PAGE_RECORD_COLUMNS.items():
            if (values := post_seq.get(key)) is not None:
                padded[key] = np.pad(
                    values, ((0, batch_size - num_rows), (0, 0)), constant_values=fill
                )
        if (impressed := post_seq.get("rl_impressed")) is not None:
            padded["rl_impressed"] = pad_array(impressed)
        return padded

    def _pad_post_seq_fields(post_seq: PostSeq) -> PostSeq:
        return PostSeq(
            impr_ts=pad_array(post_seq["impr_ts"]) if post_seq["impr_ts"] is not None else None,
            actions=pad_array(post_seq["actions"]) if post_seq["actions"] is not None else None,
            continuous_actions=pad_array(post_seq["continuous_actions"]),
            post_hashes=pad_array(post_seq["post_hashes"]),
            auth_hashes=pad_array(post_seq["auth_hashes"]),
            ip_hashes=pad_array(post_seq["ip_hashes"]),
            product_surface=pad_array(post_seq["product_surface"]),
            client_app_id=pad_array(post_seq["client_app_id"]),
            post_ids=pad_array(post_seq["post_ids"]) if post_seq["post_ids"] is not None else None,
            promoted_ids=pad_array(post_seq["promoted_ids"])
            if post_seq["promoted_ids"] is not None
            else None,
            line_item_objective=pad_array(post_seq["line_item_objective"])
            if post_seq["line_item_objective"] is not None
            else None,
            safety_label_mask=pad_array(post_seq["safety_label_mask"])
            if post_seq["safety_label_mask"] is not None
            else None,
            embedding=pad_array(cast(np.ndarray, post_seq["embedding"]))
            if post_seq["embedding"] is not None
            else None,
            search_query_embeddings=pad_array(post_seq["search_query_embeddings"])
            if post_seq["search_query_embeddings"] is not None
            else None,
            categorical_features=pad_array(post_seq["categorical_features"]),
            bool_features=pad_array(post_seq["bool_features"]),
            float_features=pad_array(post_seq["float_features"]),
            int64_features=pad_array(post_seq["int64_features"]),
            post_creation_ts_sec=pad_array(post_seq["post_creation_ts_sec"]),
            post_sids=pad_array(_psid)
            if (_psid := post_seq.get("post_sids")) is not None
            else None,
        )

    padded: RecsysFeaturesBatch = {
        "user_hashes": pad_array(batch_unpadded["user_hashes"]),
        "user_ip_hashes": pad_array(batch_unpadded["user_ip_hashes"]),
        "history_seq": pad_post_seq(batch_unpadded["history_seq"]),
        "candidate_seq": pad_post_seq(batch_unpadded["candidate_seq"]),
        "user_categorical_features": pad_array(batch_unpadded["user_categorical_features"]),
        "user_bool_features": pad_array(batch_unpadded["user_bool_features"]),
        "user_float_features": pad_array(batch_unpadded["user_float_features"]),
        "user_int64_features": pad_array(batch_unpadded["user_int64_features"]),
        "user_installed_apps_multihot": pad_array(batch_unpadded["user_installed_apps_multihot"]),
        "num_positive_candidates": pad_array(npc)
        if (npc := batch_unpadded.get("num_positive_candidates")) is not None
        else None,
        "sample_weights": pad_array(sw)
        if (sw := batch_unpadded.get("sample_weights")) is not None
        else None,
        "sample_source": pad_array(ss)
        if (ss := batch_unpadded.get("sample_source")) is not None
        else None,
    }
    if (cp := batch_unpadded.get("candidate_positions")) is not None:
        padded["candidate_positions"] = pad_array(cp)
    if (il := batch_unpadded.get("user_id_list_hashes")) is not None and (
        ilm := batch_unpadded.get("user_id_list_mask")
    ) is not None:
        padded["user_id_list_hashes"] = pad_array(il)
        padded["user_id_list_mask"] = pad_array(ilm)

    extras = cast(dict[str, np.ndarray], batch_unpadded)
    padded_dict = cast(dict[str, np.ndarray], padded)
    for key, arr in extras.items():
        if key.startswith("conversion_delay_ms_seq"):
            pad_rows = batch_size - num_rows
            padded_dict[key] = np.concatenate(
                [arr, np.full((pad_rows, *arr.shape[1:]), -1, dtype=arr.dtype)]
            )
        elif key.startswith("conversion_label_seq"):
            padded_dict[key] = pad_array(arr)

    return padded


@configclass
class PhoenixDataset(Dataset):
    hash_table: HashTable
    path: str | None = None
    pad_token: int = 0
    input_vocab_size: int = 100_000
    hash_vocab_size: int = 0
    output_vocab_size: int = 64
    num_continuous_actions: int = 2
    history_seq_len: int = 1024
    candidate_seq_len: int = 128
    is_eval: bool = False
    num_negatives_per_example: int = 1
    num_kafka_partitions: int | None = None
    include_candidate_post_ids: bool = False
    date_range: tuple[str, str] | None = None
    search_query_embedding_dim: int = 0
    aux_value_events: tuple[str, ...] = ()

    candidate_negative_filter: CandidateNegativeFilter | None = None
    candidate_negative_mode: CandidateNegativeMode | None = None

    num_global_negatives_per_example: int = 0
    global_post_ids: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_author_ids: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_post_creation_datetimes: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_post_sids: np.ndarray | None = dataclasses.field(init=False, default=None)
    _alloc: ArrayAllocator = dataclasses.field(
        init=False, default_factory=ArrayAllocator, repr=False, compare=False
    )
    global_ids_file_path: Path = (
        PHOENIX_INDEX_BASE / "post_creation_snapshots/post_creation_1day.parquet"
    )

    use_post_sid: bool = False
    sid_num_levels: int = 0

    compute_post_unexplored_label: bool = False
    enable_stale_post: bool = False
    enable_stale_post_30d: bool = False

    ads_head_masking: bool = False
    dump_keep_rate_per_10000: int | None = None
    ads_pixel_head_masking: bool = False
    ads_late_window_heads: bool = False
    page_records: bool = False
    read_ads_engagement_counts: bool = False

    multimodal_embedding_type: EmbeddingType | None = None

    history_url_embedding_dim: int = 0

    use_conversion_labels: bool = False
    conversion_label_window_ms: int = 7 * 24 * 60 * 60 * 1000
    conversion_label_types: tuple[str, ...] = ()
    fold_conversion_actions_into_multihot: bool = True
    emit_conversion_label_keys: bool = False

    id_list_features: str = ""
    id_list_sidecar_root: str | None = None
    id_list_prefer_sidecar: bool = True
    id_list_sidecar_strict: bool = False

    @property
    def id_list_specs(self) -> tuple[id_lists.IdListFeatureSpec, ...]:
        return id_lists.parse_specs(self.id_list_features)

    @property
    def multimodal_embedding_dim(self) -> int:
        if self.multimodal_embedding_type is None:
            return 0
        return EMBEDDING_CONFIG[self.multimodal_embedding_type][1]

    offline_embedding_table_dir: str | None = None

    filter_candidates_require_embedding: bool = False

    continuous: bool = False

    shuffle_window_time_slices: int = 0
    shuffle_in_memory_buffer_rows: int = 0
    shuffle_seed: int = 0
    shuffle_mode: ShuffleMode = ShuffleMode.NONE
    block_shuffle_block_time_slices: int = 20
    block_shuffle_order_per_shard: bool = False

    reply_parent_mode: bool = False

    def int64_feature_sizes(self) -> tuple[int, int]:
        if self.reply_parent_mode:
            return (
                reply_features.REPLY_POST_INT64_FEATURE_SIZE,
                reply_features.REPLY_USER_INT64_FEATURE_SIZE,
            )
        return (POST_INT64_FEATURE_SIZE, USER_INT64_FEATURE_SIZE)

    @staticmethod
    def _parse_date_bound(s: str) -> int | None:
        return parse_date_bound(s)

    def _counts_steps_from_manifest(self) -> bool:
        return True

    def compute_max_steps(
        self,
        num_shards: int,
        batch_size: int,
        current_step: int,
        resume_position: DataPosition | None = None,
        end_batch_id: int | None = None,
        start_batch_id: int | None = None,
    ) -> int | None:
        if self.path is None:
            return None
        min_ts = max_ts = None
        if self.date_range is not None:
            min_ts = self._parse_date_bound(self.date_range[0])
            max_ts = self._parse_date_bound(self.date_range[1])
        if end_batch_id is None and max_ts is None:
            return None

        topic_dir = self.path
        metadata_path = os.path.join(topic_dir, ".valid_batches.json")
        meta = _load_valid_batches_metadata(metadata_path)
        if meta is None:
            return None

        if end_batch_id is None:
            start_bid, end_bid = _resolve_time_range(
                topic_dir,
                meta["min_valid_batch"],
                meta["max_valid_batch"],
                min_ts,
                max_ts,
            )
        else:
            start_bid = self.start_batch_id() if start_batch_id is None else start_batch_id
            end_bid = min(end_batch_id, meta["max_valid_batch"] + 1)
        resume_bid: int | None = None
        saved_reads = 0
        saved_bs: int | None = None
        if resume_position is not None:
            resume_bid = resume_position["last_batch_id"]
            saved_reads = resume_position["rows_read_in_batch"]
            saved_bs = resume_position.get("batch_size")

        manifest = load_batch_manifest(topic_dir) if self._counts_steps_from_manifest() else None
        num_partitions = meta["num_partitions"]
        partitions = self.num_kafka_partitions or num_partitions
        shards_per_partition = 1
        if partitions and num_shards > partitions and num_shards % partitions == 0:
            shards_per_partition = num_shards // partitions
        num_partition_owners = num_shards // shards_per_partition
        rows_per_step = batch_size * shards_per_partition
        files_per_shard = (self.num_kafka_partitions or 0) // num_partition_owners

        if (
            self.shuffle_window_time_slices > 0
            and self.shuffle_mode != ShuffleMode.BLOCK_SHUFFLE
            and not self.is_eval
        ):
            count = functools.partial(
                _shuffled_remaining_batches,
                start_bid=start_bid,
                end_bid=end_bid,
                window=self.shuffle_window_time_slices,
                batch_size=batch_size,
            )
            resume = (
                {}
                if resume_bid is None
                else {
                    "resume_bid": resume_bid,
                    "resume_rows": saved_reads * (saved_bs or batch_size),
                }
            )
            if manifest is not None:
                rows = manifest.rows_matrix(topic_dir, start_bid, end_bid, num_partitions)
                shard_of = np.arange(num_partitions) % num_shards
                remaining = min(
                    count(rows_per_bid=rows[:, shard_of == s].sum(axis=1), **resume)
                    for s in range(num_shards)
                )
                source = "batch manifest, shuffled windows"
            else:
                dump_rows = pq.ParquetFile(_batch_path(topic_dir, 0, start_bid)).metadata.num_rows
                remaining = count(rows_per_bid=files_per_shard * dump_rows, **resume)
                source = f"{dump_rows}-row sample file, shuffled windows"
        elif (
            self.shuffle_mode == ShuffleMode.BLOCK_SHUFFLE
            and not self.is_eval
            and resume_position is not None
            and (
                block_resume := _block_shuffle_resume(
                    resume_position,
                    batch_size=batch_size,
                    num_shards=num_shards,
                    block=self.block_shuffle_block_time_slices,
                    window_blocks=self.shuffle_window_time_slices
                    // max(1, self.block_shuffle_block_time_slices),
                    seed=self.shuffle_seed,
                    per_shard=self.block_shuffle_order_per_shard,
                    start_batch_id=start_bid,
                    end_batch_id=end_bid,
                )
            )
            is not None
        ):
            if manifest is not None:
                rows = manifest.rows_matrix(topic_dir, start_bid, end_bid, num_partitions)
                full, tail = np.divmod(rows, rows_per_step)
                kept_tail = tail // shards_per_partition
                steps = full + ((kept_tail > 0) & (kept_tail * 2 >= batch_size))
                owner_of = np.arange(num_partitions) % num_partition_owners
                per_bid = [steps[:, owner_of == o].sum(axis=1) for o in range(num_partition_owners)]
                source = "batch manifest, block shuffle"
            else:
                dump_rows = pq.ParquetFile(_batch_path(topic_dir, 0, start_bid)).metadata.num_rows
                uniform = np.full(
                    end_bid - start_bid, files_per_shard * (dump_rows // rows_per_step)
                )
                per_bid = [uniform] * num_partition_owners
                source = f"{dump_rows}-row sample file, block shuffle"
            block = self.block_shuffle_block_time_slices
            remaining = None
            for shard in range(num_shards):
                done, left = _replay_block_segments(
                    window_start=block_resume.window_start,
                    block=block,
                    window_blocks=self.shuffle_window_time_slices // max(1, block),
                    seed=self.shuffle_seed,
                    shard=shard if self.block_shuffle_order_per_shard else None,
                    segments=block_resume.segments,
                )
                counts = per_bid[shard // shards_per_partition]
                left_steps = sum(
                    int(counts[b - start_bid])
                    for b in range(max(start_bid, block_resume.window_start), end_bid)
                    if b not in done
                )
                if left and start_bid <= left[0] < end_bid:
                    in_bid = int(counts[left[0] - start_bid])
                    left_steps -= (
                        in_bid if block_resume.drop_in_progress else min(block_resume.reads, in_bid)
                    )
                remaining = left_steps if remaining is None else min(remaining, left_steps)
            remaining = max(remaining or 0, 0)
        else:
            adjusted_reads = saved_reads
            if saved_bs is not None and saved_bs != batch_size:
                adjusted_reads = (saved_reads * saved_bs) // batch_size
            if manifest is not None:
                remaining = remaining_steps_from_manifest(
                    topic_dir,
                    manifest,
                    num_partitions=num_partitions,
                    num_shards=num_partition_owners,
                    batch_size=batch_size,
                    start_batch_id=start_bid,
                    end_batch_id=end_bid,
                    resume_batch_id=resume_bid,
                    resume_reads_in_batch=adjusted_reads,
                    shards_per_partition=shards_per_partition,
                )
                source = "batch manifest"
            else:
                dump_rows = pq.ParquetFile(_batch_path(topic_dir, 0, start_bid)).metadata.num_rows
                batches_per_bid = files_per_shard * (dump_rows // rows_per_step)
                total_batches = (end_bid - start_bid) * batches_per_bid
                consumed = 0
                if resume_bid is not None and resume_bid >= start_bid:
                    consumed = (resume_bid - start_bid) * batches_per_bid + adjusted_reads
                remaining = total_batches - consumed
                source = f"{dump_rows}-row sample file"

        data_end_step = current_step + remaining - 1

        rank_logger.info(
            "compute_max_steps: %d (current_step=%d + %d remaining over batch_ids [%d, %d), "
            "counted from the %s)",
            data_end_step,
            current_step,
            remaining,
            start_bid,
            end_bid,
            source,
        )
        return data_end_step

    def supports_follow_data_end(self) -> bool:
        return True

    @property
    def _topic_dir(self) -> str:
        assert self.path is not None, "follow_data_end requires dataset.path"
        return self.path

    def start_batch_id(self) -> int:
        meta = _load_valid_batches_metadata(os.path.join(self._topic_dir, ".valid_batches.json"))
        if meta is None:
            raise RuntimeError(f"could not read the dataset index under {self._topic_dir}")
        min_ts = None if self.date_range is None else self._parse_date_bound(self.date_range[0])
        if min_ts is None:
            return meta["min_valid_batch"]
        start_bid, _ = _resolve_time_range(
            self._topic_dir, meta["min_valid_batch"], meta["max_valid_batch"], min_ts, None
        )
        return start_bid

    def available_end_batch_id(self, after_end_batch_id: int = -1) -> int | None:
        meta = _load_valid_batches_metadata(os.path.join(self._topic_dir, ".valid_batches.json"))
        if meta is None:
            return None
        end = meta["max_valid_batch"] + 1
        if end <= after_end_batch_id:
            return None
        manifest = (
            load_batch_manifest(self._topic_dir) if self._counts_steps_from_manifest() else None
        )
        if manifest is not None:
            end = min(end, manifest.first_batch + manifest.rows.shape[0])
        return end if end > after_end_batch_id else None

    def set_end_batch_id(self, end_batch_id: int) -> None:
        self._end_batch_id = end_batch_id
        if self._rb_provider is not None:
            self._rb_provider.set_end_batch_id(end_batch_id)

    def reader_window_start(self) -> int | None:
        provider = self._rb_provider
        return None if provider is None else provider.block_shuffle_window_start()

    def shutdown(self):
        stop_event = self._producer_stop
        queue = self._producer_queue
        if stop_event is None or queue is None:
            return
        stop_event.set()
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break
        thread = self._producer_thread
        if thread is not None:
            thread.join(timeout=10)
            if thread.is_alive():
                rank_logger.warning("parquet producer thread did not retire within 10s")

    _rb_provider: InterleavingRecordBatchProvider | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _producer_queue: Queue | None = dataclasses.field(init=False, default=None, repr=False)
    _producer_stop: Event | None = dataclasses.field(init=False, default=None, repr=False)
    _producer_thread: Thread | None = dataclasses.field(init=False, default=None, repr=False)
    _end_batch_id: int | None = dataclasses.field(init=False, default=None, repr=False)

    def get_data_position(self) -> DataPosition | None:
        if self._rb_provider is not None:
            return self._rb_provider.get_position()
        return None

    def make(
        self,
        *,
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        server_hosts: list[str],
        server_port: int = 8898,
        skip_rows: int = 0,
        keep_and_pad_partial_batch: bool | None = None,
        prefetch_factor: int = 2,
        resume_position: DataPosition | None = None,
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        queue = Queue(maxsize=prefetch_factor)
        stop_event = Event()
        self._producer_queue = queue
        self._producer_stop = stop_event

        SENTINEL = object()
        del run_server, server_hosts, server_port, keep_and_pad_partial_batch

        def producer() -> None:
            try:
                assert self.path is not None, (
                    f"Called make() on {self.__class__} but self.path was None"
                )

                offline_emb_table: PostEmbeddingTable | None = None
                if self.offline_embedding_table_dir is not None:
                    offline_emb_table = PostEmbeddingTable(self.offline_embedding_table_dir)

                if self.num_global_negatives_per_example > 0:
                    (
                        self.global_post_ids,
                        self.global_author_ids,
                        self.global_post_creation_datetimes,
                        self.global_post_sids,
                    ) = load_global_ids_from_parquet_file(
                        self.global_ids_file_path,
                        read_creation_datetime=True,
                        read_post_sid=self.use_post_sid,
                        sid_num_levels=self.sid_num_levels,
                    )

                assert self.num_kafka_partitions is not None
                assert (
                    self.num_kafka_partitions % num_shards == 0
                    or num_shards % self.num_kafka_partitions == 0
                ), (
                    self.num_kafka_partitions,
                    num_shards,
                )
                interleave_k = max(1, self.num_kafka_partitions // num_shards)

                topic_dir = self.path
                metadata_path = str(Path(topic_dir) / ".valid_batches.json")
                index_path = str(Path(topic_dir) / ".index")

                conversion_delay_columns: list[str] | None = None
                if self.use_conversion_labels:
                    conversion_delay_columns = [conversion_labels.DELAY_COLUMN] + [
                        conversion_labels.type_delay_column(t) for t in self.conversion_label_types
                    ]
                id_list_specs = self.id_list_specs or None
                if id_list_specs:
                    rank_logger.info(
                        f"id-list features: {[f'{x.name}:{x.k}' for x in id_list_specs]} "
                        f"(sidecar root {self.id_list_sidecar_root!r}, "
                        f"prefer sidecar {self.id_list_prefer_sidecar}, "
                        f"strict {self.id_list_sidecar_strict})"
                    )
                if self.id_list_sidecar_root is not None and not os.path.isdir(
                    self.id_list_sidecar_root
                ):
                    raise FileNotFoundError(
                        f"id_list_sidecar_root does not exist: {self.id_list_sidecar_root!r}"
                    )
                id_list_fill_logged = False

                min_timestamp_ms: int | None = None
                max_timestamp_ms: int | None = None
                if self.date_range is not None:
                    min_timestamp_ms = self._parse_date_bound(self.date_range[0])
                    max_timestamp_ms = self._parse_date_bound(self.date_range[1])

                shuffle_window = 0 if self.is_eval else self.shuffle_window_time_slices
                shuffle_mode = ShuffleMode.NONE if self.is_eval else self.shuffle_mode
                shuffle_buffer = 0 if self.is_eval else self.shuffle_in_memory_buffer_rows

                if os.path.isfile(metadata_path):
                    rank_logger.info(f"Using metadata mode: {metadata_path}")
                    rb_provider = InterleavingRecordBatchProvider(
                        metadata_path=metadata_path,
                        topic_dir=topic_dir,
                        batch_size=batch_size,
                        num_shards=num_shards,
                        shard_index=shard_index,
                        interleave_k=interleave_k,
                        skip_rows=skip_rows if not self.is_eval else 0,
                        date_range=self.date_range,
                        continuous=self.continuous,
                        num_kafka_partitions=self.num_kafka_partitions,
                        resume_position=resume_position,
                        min_timestamp_ms=min_timestamp_ms,
                        max_timestamp_ms=max_timestamp_ms,
                        conversion_delay_columns=conversion_delay_columns,
                        include_action_delay_columns=self.use_conversion_labels
                        and self.fold_conversion_actions_into_multihot,
                        id_list_sidecar_root=self.id_list_sidecar_root,
                        id_list_prefer_sidecar=self.id_list_prefer_sidecar,
                        id_list_sidecar_strict=self.id_list_sidecar_strict,
                        shuffle_window_time_slices=shuffle_window,
                        shuffle_in_memory_buffer_rows=shuffle_buffer,
                        shuffle_seed=self.shuffle_seed,
                        shuffle_mode=shuffle_mode,
                        block_shuffle_block_time_slices=self.block_shuffle_block_time_slices,
                        block_shuffle_order_per_shard=(
                            self.block_shuffle_order_per_shard and not self.is_eval
                        ),
                        end_batch_id=self._end_batch_id,
                    )
                else:
                    if shuffle_window > 0 or shuffle_mode != ShuffleMode.NONE:
                        raise ValueError(
                            f"read-time shuffle needs metadata mode; {metadata_path} does not exist"
                        )
                    if resume_position is not None:
                        rank_logger.warning(
                            "resume_position was provided but dataset is in index mode; "
                            "ignoring resume_position and falling back to skip_rows."
                        )
                    rank_logger.info(f"Using index mode: {index_path}")
                    rb_provider = InterleavingRecordBatchProvider(
                        index_path=index_path,
                        batch_size=batch_size,
                        num_shards=num_shards,
                        shard_index=shard_index,
                        interleave_k=interleave_k,
                        skip_rows=skip_rows if not self.is_eval else 0,
                        date_range=self.date_range,
                        continuous=self.continuous,
                        num_kafka_partitions=self.num_kafka_partitions,
                        conversion_delay_columns=conversion_delay_columns,
                        include_action_delay_columns=self.use_conversion_labels
                        and self.fold_conversion_actions_into_multihot,
                        id_list_sidecar_root=self.id_list_sidecar_root,
                        id_list_prefer_sidecar=self.id_list_prefer_sidecar,
                        id_list_sidecar_strict=self.id_list_sidecar_strict,
                    )
                self._rb_provider = rb_provider
                data_iter = self._rb_provider.get_record_batches()

                for record_batch in data_iter:
                    if record_batch.num_rows * 2 < batch_size:
                        rank_logger.warning(
                            "Skipping record batch that doesn't contain enough rows"
                        )
                        continue

                    if self.global_post_creation_datetimes is not None:
                        assert self.global_post_ids is not None
                        assert self.global_author_ids is not None
                        data_datetime = self.get_latest_datetime(record_batch["impressedTimeMsSeq"])
                        earlist_datetime = pd.to_datetime(
                            data_datetime - timedelta(hours=24)
                        ).to_numpy()
                        latest_datetime = pd.to_datetime(data_datetime).to_numpy()
                        qualified_indices = np.where(
                            (earlist_datetime <= self.global_post_creation_datetimes)
                            & (self.global_post_creation_datetimes <= latest_datetime)
                        )[0]
                        rank_logger.info(
                            f"Only select {len(qualified_indices)}/{len(self.global_post_ids)} global posts created within 24 hours of the data datetime: {data_datetime}"
                        )
                        global_post_ids = self.global_post_ids[qualified_indices]
                        global_author_ids = self.global_author_ids[qualified_indices]
                        global_post_sids = (
                            self.global_post_sids[qualified_indices]
                            if self.global_post_sids is not None
                            else None
                        )
                    else:
                        global_post_ids = self.global_post_ids
                        global_author_ids = self.global_author_ids
                        global_post_sids = self.global_post_sids

                    if self.use_conversion_labels and self.fold_conversion_actions_into_multihot:
                        record_batch = conversion_labels.fold_action_delays_into_multihot(
                            record_batch, self.conversion_label_window_ms
                        )

                    batch = from_record_batch(
                        record_batch,
                        self.history_seq_len,
                        self.candidate_seq_len,
                        self.num_negatives_per_example,
                        self.output_vocab_size,
                        self.num_continuous_actions,
                        self.hash_table,
                        self.include_candidate_post_ids,
                        self.num_global_negatives_per_example,
                        global_post_ids,
                        global_author_ids,
                        embedding_type=self.multimodal_embedding_type,
                        offline_embedding_table=offline_emb_table,
                        filter_candidates_require_embedding=self.filter_candidates_require_embedding,
                        search_query_embedding_dim=self.search_query_embedding_dim,
                        global_post_sids=global_post_sids,
                        sid_num_levels=self.sid_num_levels if self.use_post_sid else 0,
                        compute_post_unexplored_label=self.compute_post_unexplored_label,
                        zero_stale_post_14d_candidate_counts=self.enable_stale_post,
                        stale_post_30d=self.enable_stale_post_30d,
                        ads_head_masking=self.ads_head_masking,
                        dump_keep_rate_per_10000=self.dump_keep_rate_per_10000,
                        ads_pixel_head_masking=self.ads_pixel_head_masking,
                        ads_late_window_heads=self.ads_late_window_heads,
                        reply_parent_mode=self.reply_parent_mode,
                        page_records=self.page_records,
                        id_list_specs=id_list_specs,
                        history_url_embedding_dim=self.history_url_embedding_dim,
                        read_ads_engagement_counts=self.read_ads_engagement_counts,
                        alloc=self._alloc,
                        aux_value_events=self.aux_value_events,
                    )

                    if self.use_conversion_labels and self.emit_conversion_label_keys:
                        assert conversion_delay_columns is not None
                        for col_name, ctype in zip(
                            conversion_delay_columns,
                            (None, *self.conversion_label_types),
                        ):
                            delays_col = record_batch.column(col_name)
                            seq_len = delays_col.type.list_size
                            delays = (
                                delays_col.flatten()
                                .to_numpy(zero_copy_only=False)
                                .astype(np.int64)
                                .reshape(record_batch.num_rows, seq_len)
                            )
                            suffix = "" if ctype is None else f"_{ctype}"
                            batch[f"conversion_delay_ms_seq{suffix}"] = delays
                            batch[f"conversion_label_seq{suffix}"] = (
                                conversion_labels.delays_to_labels(
                                    delays, self.conversion_label_window_ms
                                )
                            )

                    if (
                        id_list_specs
                        and not id_list_fill_logged
                        and (id_list_mask := batch.get("user_id_list_mask")) is not None
                    ):
                        id_list_fill_logged = True
                        stats = id_lists.fill_stats(id_list_mask, id_list_specs)
                        rank_logger.info(
                            "id-list features in the first batch (users>=1 / full at k / "
                            "mean fill): "
                            + ", ".join(
                                f"{n}={a:.3f}/{b:.3f}/{c:.3f}" for n, (a, b, c) in stats.items()
                            )
                        )
                        for warning in id_lists.fill_warnings(stats):
                            rank_logger.warning(f"id-list features: {warning}")

                    if record_batch.num_rows < batch_size:
                        rank_logger.warning(
                            f"Padding batch of size {record_batch.num_rows} to {batch_size}"
                        )
                        batch = pad_batch(batch, batch_size)

                    while not stop_event.is_set():
                        try:
                            queue.put(batch, timeout=0.5)
                            break
                        except Full:
                            continue
                    if stop_event.is_set():
                        return
            except Exception as e:
                rank_logger.error(traceback.print_exc())
                queue.put(e)
            finally:
                if not stop_event.is_set():
                    queue.put(SENTINEL)

        thread = Thread(target=producer, daemon=True)
        self._producer_thread = thread
        thread.start()

        def generator():
            while True:
                item = queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item, None

            thread.join()

        return generator()

    def example_data_shape(self, batch_size: int) -> Any:
        import jax
        import numpy as np

        example_data: RecsysFeaturesBatch = self.example_data(batch_size)
        user_hashes = example_data["user_hashes"]
        history_seq = example_data["history_seq"]
        candidate_seq = example_data["candidate_seq"]

        history_seq_shape = {}
        for k, v in history_seq.items():
            if v is not None and isinstance(v, np.ndarray):
                history_seq_shape[k] = jax.ShapeDtypeStruct(v.shape, v.dtype)
            else:
                history_seq_shape[k] = None

        candidate_seq_shape = {}
        for k, v in candidate_seq.items():
            if v is not None and isinstance(v, np.ndarray):
                candidate_seq_shape[k] = jax.ShapeDtypeStruct(v.shape, v.dtype)
            else:
                candidate_seq_shape[k] = None

        user_hashes_shape = jax.ShapeDtypeStruct(user_hashes.shape, user_hashes.dtype)
        user_ip_hashes = example_data["user_ip_hashes"]
        user_ip_hashes_shape = jax.ShapeDtypeStruct(user_ip_hashes.shape, user_ip_hashes.dtype)

        batch_shape = {
            "user_hashes": user_hashes_shape,
            "user_ip_hashes": user_ip_hashes_shape,
            "history_seq": history_seq_shape,
            "candidate_seq": candidate_seq_shape,
            "user_categorical_features": jax.ShapeDtypeStruct(
                example_data["user_categorical_features"].shape,
                example_data["user_categorical_features"].dtype,
            ),
            "user_bool_features": jax.ShapeDtypeStruct(
                example_data["user_bool_features"].shape,
                example_data["user_bool_features"].dtype,
            ),
            "user_float_features": jax.ShapeDtypeStruct(
                example_data["user_float_features"].shape,
                example_data["user_float_features"].dtype,
            ),
            "user_int64_features": jax.ShapeDtypeStruct(
                example_data["user_int64_features"].shape,
                example_data["user_int64_features"].dtype,
            ),
            "user_installed_apps_multihot": jax.ShapeDtypeStruct(
                example_data["user_installed_apps_multihot"].shape,
                example_data["user_installed_apps_multihot"].dtype,
            ),
            "num_positive_candidates": jax.ShapeDtypeStruct(npc.shape, npc.dtype)
            if (npc := example_data.get("num_positive_candidates")) is not None
            else None,
            "sample_weights": jax.ShapeDtypeStruct(sw.shape, sw.dtype)
            if (sw := example_data.get("sample_weights")) is not None
            else None,
            "sample_source": jax.ShapeDtypeStruct(ss.shape, ss.dtype)
            if (ss := example_data.get("sample_source")) is not None
            else None,
        }
        if self.reply_parent_mode:
            batch_shape["candidate_positions"] = jax.ShapeDtypeStruct(
                candidate_seq["impr_ts"].shape, np.int32
            )
        if (il := example_data.get("user_id_list_hashes")) is not None and (
            ilm := example_data.get("user_id_list_mask")
        ) is not None:
            batch_shape["user_id_list_hashes"] = jax.ShapeDtypeStruct(il.shape, il.dtype)
            batch_shape["user_id_list_mask"] = jax.ShapeDtypeStruct(ilm.shape, ilm.dtype)

        return batch_shape

    def _example_head_eligible_mask(
        self, batch_size: int, candidate_seq_len: int
    ) -> dict[str, np.ndarray]:
        if not self.ads_pixel_head_masking:
            return {}
        return {
            "head_eligible_mask": np.ones(
                (batch_size, candidate_seq_len, self.output_vocab_size), dtype=np.bool_
            )
        }

    def example_data(
        self,
        batch_size: int,
    ) -> RecsysFeaturesBatch:
        post_int64_size, user_int64_size = self.int64_feature_sizes()
        history_seq_len = self.history_seq_len
        num_negatives_per_example = self.num_negatives_per_example
        num_global_negatives_per_example = self.num_global_negatives_per_example
        num_neg_blocks = 2 if self.search_query_embedding_dim > 0 else 1
        candidate_seq_len = (
            self.candidate_seq_len * (1 + num_neg_blocks * num_negatives_per_example)
            + num_global_negatives_per_example
        )
        batch = RecsysFeaturesBatch(
            user_hashes=np.zeros((batch_size, self.hash_table.num_user_hashes), dtype=np.int32),
            user_ip_hashes=np.zeros((batch_size, self.hash_table.num_ip_hashes), dtype=np.int32),
            history_seq=PostSeq(
                impr_ts=np.zeros((batch_size, history_seq_len), dtype=np.int32),
                actions=np.zeros(
                    (batch_size, history_seq_len, self.output_vocab_size), dtype=np.bool_
                ),
                continuous_actions=np.zeros(
                    (batch_size, history_seq_len, self.num_continuous_actions), dtype=np.float32
                ),
                post_hashes=np.zeros(
                    (batch_size, history_seq_len, self.hash_table.num_item_hashes), dtype=np.int32
                ),
                auth_hashes=np.zeros(
                    (batch_size, history_seq_len, self.hash_table.num_author_hashes), dtype=np.int32
                ),
                product_surface=np.zeros((batch_size, history_seq_len), dtype=np.int32),
                ip_hashes=np.zeros(
                    (batch_size, history_seq_len, self.hash_table.num_ip_hashes), dtype=np.int32
                ),
                client_app_id=np.zeros((batch_size, history_seq_len), dtype=np.int32),
                post_ids=None,
                promoted_ids=None,
                line_item_objective=None,
                safety_label_mask=np.zeros((batch_size, history_seq_len), dtype=np.int64),
                embedding=empty_history_url_embedding(
                    batch_size, history_seq_len, self.history_url_embedding_dim
                ),
                search_query_embeddings=None,
                post_creation_ts_sec=np.zeros((batch_size, history_seq_len), dtype=np.int32),
                post_sids=np.zeros(
                    (batch_size, history_seq_len, self.sid_num_levels), dtype=np.uint16
                )
                if self.use_post_sid
                else None,
                **empty_feature_arrays(batch_size, history_seq_len, post_int64_size),
            ),
            candidate_seq=PostSeq(
                impr_ts=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
                actions=np.zeros(
                    (batch_size, candidate_seq_len, self.output_vocab_size), dtype=np.bool_
                ),
                continuous_actions=np.zeros(
                    (batch_size, candidate_seq_len, self.num_continuous_actions), dtype=np.float32
                ),
                post_hashes=np.zeros(
                    (batch_size, candidate_seq_len, self.hash_table.num_item_hashes), dtype=np.int32
                ),
                auth_hashes=np.zeros(
                    (batch_size, candidate_seq_len, self.hash_table.num_author_hashes),
                    dtype=np.int32,
                ),
                ip_hashes=np.zeros(
                    (batch_size, candidate_seq_len, self.hash_table.num_ip_hashes), dtype=np.int32
                ),
                product_surface=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
                client_app_id=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
                trained_candidate_mask=np.ones(
                    (batch_size, candidate_seq_len, self.output_vocab_size), dtype=np.bool_
                ),
                **self._example_head_eligible_mask(batch_size, candidate_seq_len),
                **{
                    key: np.zeros((batch_size, candidate_seq_len), dtype=dtype)
                    for key, dtype in value_label_dtypes(self.aux_value_events).items()
                },
                conversion_delay_ms=empty_conversion_delays(batch_size, candidate_seq_len),
                post_ids=np.zeros((batch_size, candidate_seq_len), dtype=np.int64)
                if self.include_candidate_post_ids
                else None,
                promoted_ids=np.zeros((batch_size, candidate_seq_len), dtype=np.int64),
                line_item_objective=np.zeros((batch_size, candidate_seq_len), dtype=np.int16),
                safety_label_mask=np.zeros((batch_size, candidate_seq_len), dtype=np.int64),
                embedding=np.zeros(
                    (batch_size, candidate_seq_len, self.multimodal_embedding_dim), dtype=np.float32
                )
                if self.multimodal_embedding_dim > 0
                else None,
                search_query_embeddings=np.zeros(
                    (batch_size, candidate_seq_len, self.search_query_embedding_dim),
                    dtype=np.float32,
                )
                if self.search_query_embedding_dim > 0
                else None,
                post_creation_ts_sec=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
                post_sids=np.zeros(
                    (batch_size, candidate_seq_len, self.sid_num_levels), dtype=np.uint16
                )
                if self.use_post_sid
                else None,
                **empty_feature_arrays(batch_size, candidate_seq_len, post_int64_size),
                **(
                    empty_page_record_arrays(batch_size, candidate_seq_len)
                    if self.page_records
                    else {}
                ),
            ),
            **empty_user_feature_arrays(batch_size, user_int64_size),
            user_installed_apps_multihot=np.zeros(
                (batch_size, NUM_USER_INSTALLED_APPS), dtype=np.bool_
            ),
            num_positive_candidates=np.full((batch_size, 1), candidate_seq_len, dtype=np.int32)
            if self.candidate_negative_filter is not None
            and self.candidate_negative_filter != CandidateNegativeFilter.NONE
            else None,
            sample_weights=np.ones((batch_size, 1), dtype=np.float32),
            sample_source=np.zeros((batch_size, 1), dtype=np.int8),
        )
        if self.read_ads_engagement_counts:
            batch["candidate_seq"]["ads_count_buckets"] = np.zeros(
                (batch_size, candidate_seq_len, NUM_ADS_COUNT_FEATURES), dtype=np.uint8
            )
        if specs := self.id_list_specs:
            ids, batch["user_id_list_mask"] = id_lists.empty_arrays(batch_size, specs)
            batch["user_id_list_hashes"] = id_lists.hash_id_lists(
                ids, specs, self.hash_table.get_item_hash, self.hash_table.get_author_hash
            )
        return batch

    def get_latest_datetime(self, time_ms_seq) -> datetime:
        time_ms_seq = time_ms_seq.to_numpy(zero_copy_only=False)
        if time_ms_seq.size > 0:
            max_time = np.max(np.concatenate(time_ms_seq))
        else:
            max_time = 0
        return pd.to_datetime(max_time, unit="ms")


@configclass
class PhoenixToyDataset(PhoenixDataset):
    def tweet_id_to_action_id(self, tweet_ids: np.ndarray) -> np.ndarray:
        action_global_inv_probs = np.arange(1, self.output_vocab_size + 1)
        return (tweet_ids[:, None] % action_global_inv_probs[None, :]) == 0

    def make_recsys_features_batch(self, batch_size: int) -> RecsysFeaturesBatch:
        num_neg_blocks = 2 if self.search_query_embedding_dim > 0 else 1
        candidate_seq_len = (
            self.candidate_seq_len * (1 + num_neg_blocks * self.num_negatives_per_example)
            + self.num_global_negatives_per_example
        )

        user_ids = np.random.randint(1, 100, size=(batch_size,), dtype=int)
        history_lengths = np.random.randint(1, self.history_seq_len, size=(batch_size,))
        candidate_lengths = np.random.randint(1, candidate_seq_len, size=(batch_size,))
        history_tweet_ids = np.zeros((batch_size, self.history_seq_len), dtype=np.int32)
        history_author_ids = np.zeros((batch_size, self.history_seq_len), dtype=np.int32)
        candidate_tweet_ids = np.zeros((batch_size, candidate_seq_len), dtype=np.int32)
        candidate_author_ids = np.zeros((batch_size, candidate_seq_len), dtype=np.int32)
        history_impression_timestamps = np.zeros((batch_size, self.history_seq_len), dtype=np.int32)
        candidate_impression_timestamps = np.zeros((batch_size, candidate_seq_len), dtype=np.int32)
        history_product_surface = np.zeros((batch_size, self.history_seq_len), dtype=np.int32)
        candidate_product_surface = np.zeros((batch_size, candidate_seq_len), dtype=np.int32)
        history_actions = np.zeros(
            (batch_size, self.history_seq_len, self.output_vocab_size), dtype=np.bool_
        )
        candidate_actions = np.zeros(
            (batch_size, candidate_seq_len, self.output_vocab_size), dtype=np.bool_
        )

        for idx, (history_length, candidate_length) in enumerate(
            zip(history_lengths, candidate_lengths)
        ):
            tweet_and_author_ids_single_row_history = np.random.randint(1, 100, size=history_length)
            tweet_and_author_ids_single_row_candidate = np.random.randint(
                1, 100, size=candidate_length
            )
            history_tweet_ids[idx, :history_length] = tweet_and_author_ids_single_row_history
            history_author_ids[idx, :history_length] = tweet_and_author_ids_single_row_history
            candidate_tweet_ids[idx, :candidate_length] = tweet_and_author_ids_single_row_candidate
            candidate_author_ids[idx, :candidate_length] = tweet_and_author_ids_single_row_candidate
            history_actions[idx, :history_length, :] = self.tweet_id_to_action_id(
                tweet_and_author_ids_single_row_history
            )
            candidate_actions[idx, :candidate_length, :] = self.tweet_id_to_action_id(
                tweet_and_author_ids_single_row_candidate
            )

        return RecsysFeaturesBatch(
            user_hashes=self.hash_table.get_user_hash(user_ids),
            user_ip_hashes=np.zeros((batch_size, self.hash_table.num_ip_hashes), dtype=np.int32),
            history_seq=PostSeq(
                impr_ts=history_impression_timestamps,
                actions=history_actions,
                continuous_actions=np.zeros(
                    (batch_size, self.history_seq_len, self.num_continuous_actions),
                    dtype=np.float32,
                ),
                post_hashes=self.hash_table.get_item_hash(history_tweet_ids),
                auth_hashes=self.hash_table.get_author_hash(history_author_ids),
                product_surface=history_product_surface,
                client_app_id=np.zeros((batch_size, self.history_seq_len), dtype=np.int32),
                post_ids=None,
                promoted_ids=None,
                line_item_objective=None,
                safety_label_mask=np.zeros((batch_size, self.history_seq_len), dtype=np.int64),
                embedding=empty_history_url_embedding(
                    batch_size, self.history_seq_len, self.history_url_embedding_dim
                ),
                search_query_embeddings=None,
                post_creation_ts_sec=np.zeros((batch_size, self.history_seq_len), dtype=np.int32),
                **empty_feature_arrays(batch_size, self.history_seq_len),
            ),
            candidate_seq=PostSeq(
                impr_ts=candidate_impression_timestamps,
                actions=candidate_actions,
                continuous_actions=np.zeros(
                    (batch_size, candidate_seq_len, self.num_continuous_actions),
                    dtype=np.float32,
                ),
                post_hashes=self.hash_table.get_item_hash(candidate_tweet_ids),
                auth_hashes=self.hash_table.get_author_hash(candidate_author_ids),
                product_surface=candidate_product_surface,
                client_app_id=np.zeros((batch_size, self.candidate_seq_len), dtype=np.int32),
                trained_candidate_mask=np.ones(
                    (batch_size, candidate_seq_len, self.output_vocab_size), dtype=np.bool_
                ),
                **self._example_head_eligible_mask(batch_size, candidate_seq_len),
                **{
                    key: np.zeros((batch_size, candidate_seq_len), dtype=dtype)
                    for key, dtype in value_label_dtypes(self.aux_value_events).items()
                },
                conversion_delay_ms=empty_conversion_delays(batch_size, candidate_seq_len),
                post_ids=candidate_tweet_ids.astype(np.int64)
                if self.include_candidate_post_ids
                else None,
                promoted_ids=np.zeros((batch_size, candidate_seq_len), dtype=np.int64),
                line_item_objective=np.zeros((batch_size, candidate_seq_len), dtype=np.int16),
                safety_label_mask=np.zeros((batch_size, candidate_seq_len), dtype=np.int64),
                embedding=None,
                search_query_embeddings=None,
                post_creation_ts_sec=np.zeros((batch_size, candidate_seq_len), dtype=np.int32),
                **empty_feature_arrays(batch_size, candidate_seq_len),
                **(
                    empty_page_record_arrays(batch_size, candidate_seq_len)
                    if self.page_records
                    else {}
                ),
            ),
            **empty_user_feature_arrays(batch_size),
            user_installed_apps_multihot=np.zeros(
                (batch_size, NUM_USER_INSTALLED_APPS), dtype=np.bool_
            ),
            num_positive_candidates=None,
        )

    def make(
        self,
        *,
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        server_hosts: list[str],
        server_port: int = 8898,
        skip_rows: int = 0,
        keep_and_pad_partial_batch: bool | None = None,
        prefetch_factor: int = 2,
        resume_position: DataPosition | None = None,
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        del (
            shard_index,
            num_shards,
            run_server,
            server_hosts,
            server_port,
            skip_rows,
            keep_and_pad_partial_batch,
            prefetch_factor,
            resume_position,
        )
        np.random.seed(42)
        while True:
            yield self.make_recsys_features_batch(batch_size), None
