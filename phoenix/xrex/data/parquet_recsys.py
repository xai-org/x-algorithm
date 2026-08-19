# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import dataclasses
import logging
import os
import re
import time
import traceback
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, cast, final

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
    parse_date_bound,
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
from xrex.data.recsys.recsys_batch import (
    EMBEDDING_CONFIG,
    NUM_USER_INSTALLED_APPS,
    CandidateNegativeFilter,
    CandidateNegativeMode,
    EmbeddingType,
    PostEmbeddingTable,
    PostSeq,
    RecsysFeaturesBatch,
    empty_feature_arrays,
    empty_user_feature_arrays,
    from_record_batch,
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


class LazyRecordBatchIterator:
    def __init__(
        self,
        pf: pq.ParquetFile,
        batch_size: int,
        fname: str,
        conversion_delay_columns: list[str] | None = None,
        include_action_delay_columns: bool = False,
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
        self._row_pos += batch.num_rows
        return batch

    def skip_batch(self):
        if self.rows_to_skip < self.num_rows:
            num_rows = min(self.num_rows - self.rows_to_skip, self.batch_size)
            self.rows_to_skip += self.batch_size
            if num_rows < int(0.5 * self.batch_size):
                return False
            return True
        return False


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
    ):
        self._conversion_delay_columns = conversion_delay_columns
        self._include_action_delay_columns = include_action_delay_columns
        if metadata_path is None and index_path is None:
            raise ValueError("Either metadata_path or index_path must be provided")

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
            start_batch_id, end_batch_id = _resolve_time_range(
                self._topic_dir,
                meta["min_valid_batch"],
                meta["max_valid_batch"],
                min_timestamp_ms,
                max_timestamp_ms,
            )
            if max_timestamp_ms is not None:
                self._end_batch_id = end_batch_id

        if resume_position is not None:
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
                if p % self._num_shards == self._shard_index
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
            start_date = (
                datetime.strptime(start_str, DATE_TIME_FORMAT)
                if start_str.lower() != "none"
                else None
            )
            end_date = (
                datetime.strptime(end_str, DATE_TIME_FORMAT) if end_str.lower() != "none" else None
            )
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
                    if (_extract_partition_id(f) or 0) % self._num_shards == self._shard_index
                ]
                ready.append(my_files)
                self._next_batch_id = bid + 1
            else:
                break

        return ready

    def _open_file(self, file: str, pool: ThreadPoolExecutor, active: deque) -> None:
        rank_logger.info(f"Worker {self._shard_index}/{self._num_shards} opening file {file}")
        path = _resolve_file_path(self._path, file)

        def _impl():
            try:
                pf = ParquetFile(path)
                return LazyRecordBatchIterator(
                    pf,
                    self._batch_size,
                    path,
                    self._conversion_delay_columns,
                    self._include_action_delay_columns,
                )
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
            futures = [pool.submit(self._safe_read, h) for h in holders]
            active.clear()

            for holder, fut in zip(holders, futures):
                batch = fut.result()
                if batch is not None:
                    self._reads_in_current_batch += 1
                    active.append(holder)
                    yield batch

            _submit_opens()
            _harvest_ready()

            _fill_active()

            if len(active) < self._interleave_k and pending:
                _fill_active_blocking()

    def get_record_batches(self) -> Iterator[pa.RecordBatch]:
        yield from self._get_record_batches_synced()

    def get_position(self) -> DataPosition:
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
        return DataPosition(
            last_batch_id=self._current_drain_batch_id,
            rows_read_in_batch=self._reads_in_current_batch,
            batch_size=self._batch_size,
        )

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
    }

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

    candidate_negative_filter: CandidateNegativeFilter | None = None
    candidate_negative_mode: CandidateNegativeMode | None = None

    num_global_negatives_per_example: int = 0
    global_post_ids: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_author_ids: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_post_creation_datetimes: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_post_sids: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_ids_file_path: Path = (
        PHOENIX_INDEX_BASE / "post_creation_snapshots/post_creation_1day.parquet"
    )

    use_post_sid: bool = False
    sid_num_levels: int = 0

    compute_post_unexplored_label: bool = False

    multimodal_embedding_type: EmbeddingType | None = None

    use_conversion_labels: bool = False
    conversion_label_window_ms: int = 7 * 24 * 60 * 60 * 1000
    conversion_label_types: tuple[str, ...] = ()
    fold_conversion_actions_into_multihot: bool = True
    emit_conversion_label_keys: bool = False

    @property
    def multimodal_embedding_dim(self) -> int:
        if self.multimodal_embedding_type is None:
            return 0
        return EMBEDDING_CONFIG[self.multimodal_embedding_type][1]

    offline_embedding_table_dir: str | None = None

    filter_candidates_require_embedding: bool = False

    continuous: bool = False

    @staticmethod
    def _parse_date_bound(s: str) -> int | None:
        return parse_date_bound(s)

    def compute_max_steps(
        self,
        num_shards: int,
        batch_size: int,
        current_step: int,
        resume_position: DataPosition | None = None,
    ) -> int | None:
        if self.date_range is None or self.path is None:
            return None
        max_ts = self._parse_date_bound(self.date_range[1])
        if max_ts is None:
            return None

        topic_dir = self.path
        metadata_path = os.path.join(topic_dir, ".valid_batches.json")
        meta = _load_valid_batches_metadata(metadata_path)
        if meta is None:
            return None

        min_ts = self._parse_date_bound(self.date_range[0])
        start_bid, end_bid = _resolve_time_range(
            topic_dir,
            meta["min_valid_batch"],
            meta["max_valid_batch"],
            min_ts,
            max_ts,
        )
        files_per_shard = (self.num_kafka_partitions or 0) // num_shards

        sample_path = _batch_path(topic_dir, 0, start_bid)
        dump_rows = pq.ParquetFile(sample_path).metadata.num_rows
        chunks_per_file = dump_rows // batch_size
        batches_per_bid = files_per_shard * chunks_per_file
        total_batches = (end_bid - start_bid) * batches_per_bid

        consumed = 0
        if resume_position is not None:
            resume_bid = resume_position["last_batch_id"]
            if resume_bid >= start_bid:
                saved_reads = resume_position["rows_read_in_batch"]
                saved_bs = resume_position.get("batch_size")
                if saved_bs is not None and saved_bs != batch_size:
                    adjusted_reads = (saved_reads * saved_bs) // batch_size
                else:
                    adjusted_reads = saved_reads
                consumed = (resume_bid - start_bid) * batches_per_bid + adjusted_reads
        remaining = total_batches - consumed

        data_end_step = current_step + remaining - 1

        rank_logger.info(
            "compute_max_steps: %d (current_step=%d + %d remaining "
            "of %d total batches, consumed=%d)",
            data_end_step,
            current_step,
            remaining,
            total_batches,
            consumed,
        )
        return data_end_step

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
                assert self.num_kafka_partitions % num_shards == 0, (
                    self.num_kafka_partitions,
                    num_shards,
                )
                interleave_k = self.num_kafka_partitions // num_shards

                topic_dir = self.path
                metadata_path = str(Path(topic_dir) / ".valid_batches.json")
                index_path = str(Path(topic_dir) / ".index")

                conversion_delay_columns: list[str] | None = None
                if self.use_conversion_labels:
                    conversion_delay_columns = [conversion_labels.DELAY_COLUMN] + [
                        conversion_labels.type_delay_column(t) for t in self.conversion_label_types
                    ]

                min_timestamp_ms: int | None = None
                max_timestamp_ms: int | None = None
                if self.date_range is not None:
                    min_timestamp_ms = self._parse_date_bound(self.date_range[0])
                    max_timestamp_ms = self._parse_date_bound(self.date_range[1])

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
                    )
                else:
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
        }

        return batch_shape

    def example_data(
        self,
        batch_size: int,
    ) -> RecsysFeaturesBatch:
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
                embedding=None,
                search_query_embeddings=None,
                post_creation_ts_sec=np.zeros((batch_size, history_seq_len), dtype=np.int32),
                post_sids=np.zeros(
                    (batch_size, history_seq_len, self.sid_num_levels), dtype=np.uint16
                )
                if self.use_post_sid
                else None,
                **empty_feature_arrays(batch_size, history_seq_len),
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
                **empty_feature_arrays(batch_size, candidate_seq_len),
            ),
            **empty_user_feature_arrays(batch_size),
            user_installed_apps_multihot=np.zeros(
                (batch_size, NUM_USER_INSTALLED_APPS), dtype=np.bool_
            ),
            num_positive_candidates=np.full((batch_size, 1), candidate_seq_len, dtype=np.int32)
            if self.candidate_negative_filter is not None
            and self.candidate_negative_filter != CandidateNegativeFilter.NONE
            else None,
            sample_weights=np.ones((batch_size, 1), dtype=np.float32),
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
                embedding=None,
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
