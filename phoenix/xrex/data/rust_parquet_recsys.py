# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import functools
import importlib.util
import logging
import os
import sys
import threading
from typing import Callable, Iterable, Iterator, Sequence, cast

import numpy as np
import pyarrow as pa

from xai_configlib import configclass
from xrex import settings
from xrex.data import rust_ext
from xrex.data.parquet_recsys import (
    PhoenixDataset,
    ShuffleMode,
    _block_resume_window,
    _block_shuffle_fingerprint,
    _block_shuffle_order,
    _block_shuffle_ranges,
    _block_shuffle_resume,
    _block_shuffle_sequence,
    _block_window_grew,
    _BlockLog,
    _replay_block_segments,
)
from xrex.data.parquet_recsys_metadata import (
    DataPosition,
    batch_path,
    is_file,
    is_object_store_path,
    load_batch_manifest,
    load_valid_batches_metadata,
    parquet_file,
    parse_date_bound,
    resolve_time_range,
)

rank_logger = logging.getLogger("rank")
_COPY_ROOT_BOUNDS_LOCK = threading.Lock()


def _block_shuffle_record_batches(
    make_provider: Callable[..., Iterable[pa.RecordBatch]],
    *,
    topic_dir: str,
    num_partitions: int,
    batch_size: int,
    shard_index: int,
    num_shards: int,
    window: int,
    block: int,
    seed: int,
    per_shard: bool,
    start_batch_id: int | None,
    end_batch_id: int | None,
    skip_rows: int,
    resume_position: DataPosition | None,
    end_batch_id_fn: Callable[[], int | None] | None = None,
    on_window: Callable[[int], None] | None = None,
    shards_per_partition: int = 1,
) -> Iterator[tuple[pa.RecordBatch, DataPosition]]:
    def end_cap() -> int | None:
        return end_batch_id if end_batch_id_fn is None else end_batch_id_fn()

    window_blocks = window // block
    owners = num_shards // shards_per_partition
    read_size = batch_size * shards_per_partition
    pps = num_partitions // owners
    metadata_path = os.path.join(topic_dir, ".valid_batches.json")
    range_start = start_batch_id or 0
    shard = shard_index if per_shard else None

    resume: tuple[int, list[list[int]]] | None = None
    resume_reads = 0
    resume_drop = False
    resume_rows_in_file: int | None = None
    resume_next_file_rows = 0
    pending_skip_reads = 0
    window_start: int | None = None
    if resume_position is not None:
        block_resume = _block_shuffle_resume(
            resume_position,
            batch_size=batch_size,
            num_shards=num_shards,
            block=block,
            window_blocks=window_blocks,
            seed=seed,
            per_shard=per_shard,
            start_batch_id=range_start,
            end_batch_id=end_cap(),
        )
        if block_resume is not None:
            resume = (block_resume.window_start, block_resume.segments)
            resume_reads = block_resume.reads
            resume_drop = block_resume.drop_in_progress
            resume_rows_in_file = block_resume.rows_in_file
            resume_next_file_rows = block_resume.next_file_rows
            window_start = block_resume.window_start
    else:
        pending_skip_reads = skip_rows // (read_size * owners)

    row_counts = _row_counter(topic_dir, shard_index // shards_per_partition, owners, pps)

    while True:
        meta = load_valid_batches_metadata(metadata_path)
        if meta is None:
            return
        end = meta["max_valid_batch"] + 1
        cap = end_cap()
        if cap is not None:
            end = min(end, cap)
        lo = max(meta["min_valid_batch"], range_start)
        if window_start is None:
            window_start = (lo // window) * window
        if window_start >= end:
            return
        hi = min(end, window_start + window)
        if on_window is not None:
            on_window(window_start)
        done: set[int] = set()
        segments: list[list[int]] = [[lo, hi, 0]]
        current: int | None = None
        skip_reads = 0
        next_file_rows = 0
        if resume is not None and resume[0] == window_start:
            segments, done, current, overflow_rows = _block_resume_window(
                segments=resume[1],
                in_progress=resume_reads > 0 or bool(resume_rows_in_file),
                drop_in_progress=resume_drop,
                window_start=window_start,
                lo=lo,
                hi=hi,
                block=block,
                window_blocks=window_blocks,
                seed=seed,
                shard=shard,
                num_shards=num_shards,
                next_file_rows=resume_next_file_rows,
            )
            if current is not None:
                skip_reads = resume_reads
                done.add(current)
            else:
                next_file_rows = overflow_rows
        resume = None
        fingerprint = _block_shuffle_fingerprint(
            window_start=window_start,
            block=block,
            window_blocks=window_blocks,
            seed=seed,
            per_shard=per_shard,
            num_shards=num_shards,
        )
        before = [list(seg) for seg in segments]
        if current is not None:
            segments[-1][2] += 1
        if segments[-1][:2] != [lo, hi]:
            segments.append([lo, hi, 0])
        after = segments
        sequence = _block_shuffle_sequence(
            window_start=window_start,
            block=block,
            window_blocks=window_blocks,
            seed=seed,
            shard=shard,
            lo=lo,
            hi=hi,
        )
        bids = ([current] if current is not None else []) + [b for b in sequence if b not in done]
        first_rest = 1 if current is not None else 0

        block_log = _BlockLog(
            shard_index=shard_index,
            window_start=window_start,
            window=window,
            block=block,
            order=_block_shuffle_ranges(
                window_start=window_start,
                block=block,
                window_blocks=window_blocks,
                seed=seed,
                lo=lo,
                hi=hi,
                shard=shard,
            ),
            segments=before,
        )
        if bids:
            if pending_skip_reads > 0:
                window_reads = sum(
                    -(-sum(row_counts(bid)[p] for bid in bids) // read_size) for p in range(pps)
                )
                if pending_skip_reads >= window_reads:
                    pending_skip_reads -= window_reads
                    window_start += window
                    continue
                skip_reads, pending_skip_reads = pending_skip_reads, 0
                bulk, discard = (skip_reads // pps) * read_size, skip_reads % pps
            elif current is not None:
                rows_in = (
                    resume_rows_in_file
                    if resume_rows_in_file is not None
                    else (skip_reads // pps) * read_size
                )
                bulk, discard = min(rows_in, _max_rows(row_counts, bids[0])), skip_reads % pps
            elif next_file_rows:
                bulk, discard = next_file_rows, 0
            else:
                bulk, discard = 0, 0
            first = 0
            while first < len(bids) and bulk >= _max_rows(row_counts, bids[first]):
                bulk -= _max_rows(row_counts, bids[first])
                first += 1
            if first < len(bids):
                yield from _block_shuffle_window(
                    make_provider(skip_rows=bulk, batch_ranges=_to_ranges(bids[first:])),
                    bids=bids[first:],
                    row_counts=row_counts,
                    bulk=bulk,
                    discard=discard,
                    pps=pps,
                    batch_size=read_size,
                    shards_per_partition=shards_per_partition,
                    window_start=window_start,
                    num_shards=num_shards,
                    fingerprint=fingerprint,
                    on_batch_id=block_log.batch_id,
                    segments_at=functools.partial(
                        _segments_while_reading, before, after, first_rest - first
                    ),
                )
        if hi < window_start + window and _block_window_grew(metadata_path, end_cap(), hi):
            finished = [list(seg) for seg in after]
            finished[-1][2] += len(bids) - first_rest
            resume = (window_start, finished)
            resume_reads, resume_rows_in_file, resume_drop, resume_next_file_rows = (
                0,
                None,
                False,
                0,
            )
            continue
        window_start += window


def _segments_while_reading(
    before: list[list[int]], after: list[list[int]], shift: int, f: int
) -> list[list[int]]:
    if f < shift:
        return [list(seg) for seg in before]
    out = [list(seg) for seg in after]
    out[-1][2] += f - shift
    return out


def _to_ranges(bids: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for bid in bids:
        if ranges and ranges[-1][1] == bid:
            ranges[-1] = (ranges[-1][0], bid + 1)
        else:
            ranges.append((bid, bid + 1))
    return ranges


def _row_counter(
    topic_dir: str, shard_index: int, num_shards: int, pps: int
) -> Callable[[int], tuple[int, ...]]:
    parts = [shard_index + k * num_shards for k in range(pps)]

    @functools.cache
    def row_counts(bid: int) -> tuple[int, ...]:
        return tuple(parquet_file(batch_path(topic_dir, p, bid)).metadata.num_rows for p in parts)

    return row_counts


def _max_rows(row_counts: Callable[[int], tuple[int, ...]], bid: int) -> int:
    return max(row_counts(bid))


def _sequence_reads(
    provider: Iterable[pa.RecordBatch],
    *,
    bids: Sequence[int],
    row_counts: Callable[[int], tuple[int, ...]],
    bulk: int,
    discard: int,
    pps: int,
    batch_size: int,
    shards_per_partition: int = 1,
) -> Iterator[tuple[pa.RecordBatch, int, int, int, bool]]:
    it = iter(provider)
    n = 0
    for _ in range(discard):
        if next(it, None) is None:
            break
        n += 1
    starts: list[list[int]] = [[0] for _ in range(pps)]
    f_of = [0] * pps
    columns: list[list[str] | None] = [None] * pps
    for record_batch in it:
        n += 1
        p_read = (n - 1) % pps
        names, previous = record_batch.schema.names, columns[p_read]
        if previous is not None and names != previous:
            before, after = set(previous), set(names)
            rank_logger.warning(
                "Rust reader: partition %d of this shard changed columns around batch_id %d "
                "(added %s, dropped %s); a read across a schema change keeps only the columns "
                "both files have",
                p_read,
                bids[f_of[p_read]],
                sorted(after - before),
                sorted(before - after),
            )
        columns[p_read] = names
        rows = bulk + (n // pps) * batch_size
        for p in range(pps):
            while f_of[p] + 1 < len(bids):
                end = starts[p][f_of[p]] + row_counts(bids[f_of[p]])[p]
                if rows < end:
                    break
                starts[p].append(end)
                f_of[p] += 1
        if record_batch.num_rows * 2 < batch_size // shards_per_partition:
            rank_logger.warning("Skipping record batch that doesn't contain enough rows")
            continue
        f = min(f_of)
        in_files = [rows - starts[p][f] for p in range(pps)]
        exact = (
            all(fp == f for fp in f_of)
            and len(set(in_files)) == 1
            and len(set(row_counts(bids[f]))) == 1
        )
        yield record_batch, f, min(in_files), n, exact
    expected = sum(
        -(-max(0, sum(row_counts(bid)[p] for bid in bids) - bulk) // batch_size) for p in range(pps)
    )
    if n < expected:
        raise RuntimeError(
            f"The Rust reader returned {n} of {expected} record batches for batch_ids "
            f"{bids[0]}..{bids[-1]}: a partition stopped early (a missing or unreadable file, "
            "or files that share no column). Read local data with python_reader=True; on O2, "
            "which only the Rust reader reads, an object is missing or a GET failed after "
            "object_store's retries."
        )


def _position_reads(
    in_file: int, file_rows: int, n: int, pps: int, batch_size: int, exact: bool
) -> tuple[int, int | None, int]:
    if not exact:
        return (in_file // batch_size) * pps, in_file, 0
    reads = _reads_in_batch_id(in_file, file_rows, n, pps, batch_size)
    if in_file >= file_rows:
        return reads, None, 0
    next_rows = (
        in_file + batch_size - file_rows if n % pps and in_file + batch_size > file_rows else 0
    )
    return reads, in_file, next_rows


def _reads_in_batch_id(in_file: int, file_rows: int, n: int, pps: int, batch_size: int) -> int:
    if in_file >= file_rows:
        return -(-file_rows // batch_size) * pps
    return (in_file // batch_size) * pps + n % pps


def _block_shuffle_window(
    provider: Iterable[pa.RecordBatch],
    *,
    bids: list[int],
    row_counts: Callable[[int], tuple[int, ...]],
    bulk: int,
    discard: int,
    pps: int,
    batch_size: int,
    shards_per_partition: int,
    window_start: int,
    num_shards: int,
    fingerprint: str,
    on_batch_id: Callable[[int], None],
    segments_at: Callable[[int], list[list[int]]],
) -> Iterator[tuple[pa.RecordBatch, DataPosition]]:
    for record_batch, f, in_file, n, exact in _sequence_reads(
        provider,
        bids=bids,
        row_counts=row_counts,
        bulk=bulk,
        discard=discard,
        pps=pps,
        batch_size=batch_size,
        shards_per_partition=shards_per_partition,
    ):
        on_batch_id(bids[f])
        reads, rows_in_file, next_rows = _position_reads(
            in_file, _max_rows(row_counts, bids[f]), n, pps, batch_size, exact
        )
        position = DataPosition(
            last_batch_id=bids[f],
            rows_read_in_batch=reads,
            batch_size=batch_size // shards_per_partition,
        )
        position["block_shuffle_window_start"] = window_start
        position["block_shuffle_segments"] = segments_at(f)
        position["block_shuffle_num_shards"] = num_shards
        position["block_shuffle_fingerprint"] = fingerprint
        if rows_in_file is not None:
            position["block_shuffle_rows_in_file"] = rows_in_file
        if next_rows:
            position["block_shuffle_rows_in_next_file"] = next_rows
        yield record_batch, position


def _time_order_record_batches(
    make_provider: Callable[..., Iterable[pa.RecordBatch]],
    *,
    topic_dir: str,
    num_partitions: int,
    batch_size: int,
    shard_index: int,
    num_shards: int,
    start_batch_id: int | None,
    end_batch_id: int | None,
    skip_rows: int,
    resume_position: DataPosition | None,
) -> Iterator[tuple[pa.RecordBatch, DataPosition]]:
    pps = num_partitions // num_shards
    meta = load_valid_batches_metadata(os.path.join(topic_dir, ".valid_batches.json"))
    if meta is None:
        raise ValueError(f"Cannot load metadata from {topic_dir}/.valid_batches.json")
    lo = max(meta["min_valid_batch"], start_batch_id or 0)
    end = meta["max_valid_batch"] + 1
    if end_batch_id is not None:
        end = min(end, end_batch_id)
    row_counts = _row_counter(topic_dir, shard_index, num_shards, pps)
    first, bulk, discard = _time_order_start(
        lo=lo,
        end=end,
        resume_position=resume_position,
        skip_rows=skip_rows,
        batch_size=batch_size,
        num_shards=num_shards,
        pps=pps,
        row_counts=row_counts,
    )
    if first >= end:
        return
    rank_logger.info(
        "Rust reader: batch_ids [%d, %d), skipping %d rows per partition and %d reads",
        first,
        end,
        bulk,
        discard,
    )
    bids = range(first, end)
    for record_batch, f, in_file, n, exact in _sequence_reads(
        make_provider(skip_rows=bulk, start_batch_id=first, end_batch_id=end),
        bids=bids,
        row_counts=row_counts,
        bulk=bulk,
        discard=discard,
        pps=pps,
        batch_size=batch_size,
    ):
        reads, rows_in_file, _ = _position_reads(
            in_file, _max_rows(row_counts, bids[f]), n, pps, batch_size, exact
        )
        position = DataPosition(
            last_batch_id=bids[f], rows_read_in_batch=reads, batch_size=batch_size
        )
        if rows_in_file is not None:
            position["rows_in_file"] = rows_in_file
        yield record_batch, position


def _time_order_start(
    *,
    lo: int,
    end: int,
    resume_position: DataPosition | None,
    skip_rows: int,
    batch_size: int,
    num_shards: int,
    pps: int,
    row_counts: Callable[[int], tuple[int, ...]],
) -> tuple[int, int, int]:
    first, bulk, discard = lo, 0, 0
    if resume_position is not None:
        bid = resume_position["last_batch_id"]
        reads = resume_position["rows_read_in_batch"]
        saved_bs = resume_position.get("batch_size") or batch_size
        exact = resume_position.get("rows_in_file")
        if bid < lo:
            rank_logger.warning(
                "Resume batch_id %d is before the first readable batch_id %d (TTL or "
                "date_range); starting there",
                bid,
                lo,
            )
        elif exact is not None and saved_bs == batch_size:
            first, bulk, discard = bid, exact, reads % pps
        elif exact is not None:
            first, bulk = bid, exact + (saved_bs if reads % pps else 0)
        else:
            if saved_bs != batch_size:
                reads = reads * saved_bs // batch_size
            first, bulk, discard = bid, (reads // pps) * batch_size, reads % pps
            if bulk >= _max_rows(row_counts, bid):
                bulk, discard = _max_rows(row_counts, bid), 0
    else:
        reads = skip_rows // (batch_size * num_shards)
        bulk, discard = (reads // pps) * batch_size, reads % pps
    while first < end and bulk >= _max_rows(row_counts, first):
        bulk -= _max_rows(row_counts, first)
        first += 1
    return first, bulk, discard


def _rust_batches(rows: int, batch_size: int, shards_per_partition: int = 1) -> int:
    full, tail = divmod(max(rows, 0), batch_size * shards_per_partition)
    kept = tail // shards_per_partition
    return full + (1 if kept and kept * 2 >= batch_size else 0)


def _rust_remaining_batches(
    *,
    rows: Callable[[int], np.ndarray],
    parts: list[int],
    segments: list[tuple[list[int], int, int]],
    batch_size: int,
    shards_per_partition: int = 1,
) -> int:
    total = 0
    for bids, skip, discard in segments:
        if not bids:
            continue
        per_partition = np.sum([rows(bid)[parts] for bid in bids], axis=0)
        batches = sum(
            _rust_batches(int(r) - skip, batch_size, shards_per_partition) for r in per_partition
        )
        total += max(batches - discard, 0)
    return total


def _merge_by_visit(
    roots: list[Iterator[tuple[pa.RecordBatch, DataPosition]]],
    visit_key: Callable[[DataPosition], tuple[int, ...]],
) -> Iterator[tuple[int, pa.RecordBatch, DataPosition]]:
    heads = []
    for i, it in enumerate(roots):
        item = next(it, None)
        if item is not None:
            heads.append((visit_key(item[1]), i, item, it))
    while heads:
        j = min(range(len(heads)), key=lambda k: heads[k][:2])
        _, i, (record_batch, position), it = heads[j]
        yield i, record_batch, position
        item = next(it, None)
        if item is None:
            heads.pop(j)
        else:
            heads[j] = (visit_key(item[1]), i, item, it)


def _shards_per_partition(num_partitions: int, num_shards: int) -> int:
    if num_partitions and num_shards > num_partitions and num_shards % num_partitions == 0:
        return num_shards // num_partitions
    return 1


def _rust_reader_installed() -> bool:
    name = "xai_recsys_parquet_reader"
    return name in sys.modules or importlib.util.find_spec(name) is not None


@configclass
class RustParquetDataset(PhoenixDataset):
    queue_size: int = 4
    lookahead_files: int = 4
    python_reader: bool = False
    copy_roots: str = ""

    _position: DataPosition | None = dataclasses.field(init=False, default=None, repr=False)
    _rust_window_start: int | None = dataclasses.field(init=False, default=None, repr=False)
    _copy_root_bounds: dict[str, tuple[int, int]] | None = dataclasses.field(
        init=False, default=None, repr=False
    )

    def _copy_root_paths(self) -> list[str]:
        return [r.strip().rstrip("/") for r in self.copy_roots.split(",") if r.strip()]

    def _check_copy_roots(self) -> None:
        reason = self.python_reader_reason()
        if reason is not None:
            raise ValueError(f"copy_roots needs the Rust reader, not the Python one ({reason})")
        if self.dump_keep_rate_per_10000 is not None:
            raise ValueError(
                "copy_roots needs the stored sampleWeight: dump_keep_rate_per_10000 recomputes "
                "the keep rule from the actions, which relabeled copies change"
            )
        if not self.ads_head_masking:
            raise ValueError(
                "copy_roots needs ads_head_masking=True: copies train by sample_source"
            )

    def _copy_root_bounds_of(self, root: str, num_partitions: int) -> tuple[int, int]:
        with _COPY_ROOT_BOUNDS_LOCK:
            if self._copy_root_bounds is None:
                self._copy_root_bounds = {}
            if root not in self._copy_root_bounds:
                meta = load_valid_batches_metadata(os.path.join(root, ".valid_batches.json"))
                if meta is None:
                    raise ValueError(f"copy root {root} has no readable .valid_batches.json")
                if meta["num_partitions"] != num_partitions:
                    raise ValueError(
                        f"copy root {root} has {meta['num_partitions']} partitions, {self.path} "
                        f"{num_partitions}"
                    )
                self._copy_root_bounds[root] = (
                    meta["min_valid_batch"],
                    meta["max_valid_batch"] + 1,
                )
            return self._copy_root_bounds[root]

    def python_reader_reason(self) -> str | None:
        if self.python_reader:
            return "python_reader=True"
        if not _rust_reader_installed():
            return "xai_recsys_parquet_reader is not installed"
        if self.use_conversion_labels:
            return "use_conversion_labels (the label sidecar join)"
        if self.id_list_sidecar_root is not None:
            return "id_list_sidecar_root (the id-list sidecar join)"
        if self.ad_fields_sidecar_root is not None:
            return "ad_fields_sidecar_root (the ad-field sidecar join)"
        if self.continuous:
            return "continuous=True"
        if self.path is not None and not is_file(os.path.join(self.path, ".valid_batches.json")):
            return "index mode (no .valid_batches.json)"
        if not self.is_eval and (
            self.shuffle_in_memory_buffer_rows > 0
            or self.shuffle_mode == ShuffleMode.FILE_SHUFFLE
            or (self.shuffle_mode == ShuffleMode.NONE and self.shuffle_window_time_slices > 0)
        ):
            return "file_shuffle or shuffle_in_memory_buffer_rows"
        return None

    def _check_python_readable(self, reason: str) -> None:
        if self.path is not None and is_object_store_path(self.path):
            raise ValueError(
                f"{self.path} is on O2, which only the Rust reader reads, but this config needs "
                f"the Python reader: {reason}"
            )

    def get_data_position(self) -> DataPosition | None:
        if self._rb_provider is not None:
            return super().get_data_position()
        return self._position

    def _counts_steps_from_manifest(self) -> bool:
        return self.python_reader_reason() is not None and super()._counts_steps_from_manifest()

    def supports_follow_data_end(self) -> bool:
        return True

    def reader_window_start(self) -> int | None:
        if self._rb_provider is not None:
            return super().reader_window_start()
        return self._rust_window_start

    def available_end_batch_id(self, after_end_batch_id: int = -1) -> int | None:
        end = super().available_end_batch_id(after_end_batch_id)
        if end is None or self.python_reader_reason() is not None or self.path is None:
            return end
        manifest = load_batch_manifest(self.path)
        if manifest is not None:
            end = min(end, manifest.first_batch + manifest.rows.shape[0])
        return end if end > after_end_batch_id else None

    def compute_max_steps(
        self,
        num_shards: int,
        batch_size: int,
        current_step: int,
        resume_position: DataPosition | None = None,
        end_batch_id: int | None = None,
        start_batch_id: int | None = None,
    ) -> int | None:
        if self._copy_root_paths():
            self._check_copy_roots()
        reason = self.python_reader_reason()
        if reason is not None:
            self._check_python_readable(reason)
            return super().compute_max_steps(
                num_shards=num_shards,
                batch_size=batch_size,
                current_step=current_step,
                resume_position=resume_position,
                end_batch_id=end_batch_id,
                start_batch_id=start_batch_id,
            )
        if self.path is None:
            return None
        topic_dir = self.path
        meta = load_valid_batches_metadata(os.path.join(topic_dir, ".valid_batches.json"))
        if meta is None:
            return None
        min_ts = max_ts = None
        if self.date_range is not None:
            min_ts = parse_date_bound(self.date_range[0])
            max_ts = parse_date_bound(self.date_range[1])
        if end_batch_id is None:
            if max_ts is None:
                return None
            start_bid, end_bid = resolve_time_range(
                topic_dir, meta["min_valid_batch"], meta["max_valid_batch"], min_ts, max_ts
            )
        else:
            start_bid = self.start_batch_id() if start_batch_id is None else start_batch_id
            end_bid = min(end_batch_id, meta["max_valid_batch"] + 1)
        start_bid = max(start_bid, meta["min_valid_batch"])
        num_partitions = meta["num_partitions"]
        shards_per_partition = _shards_per_partition(num_partitions, num_shards)
        owners = num_shards // shards_per_partition
        pps = num_partitions // owners
        if end_bid <= start_bid or pps <= 0:
            return current_step - 1

        left, source = self._rust_shard_batches(
            topic_dir, start_bid, end_bid, num_partitions, num_shards, batch_size, resume_position
        )
        for root in self._copy_root_paths():
            root_min, root_end = self._copy_root_bounds_of(root, num_partitions)
            root_left, _ = self._rust_shard_batches(
                root,
                max(start_bid, root_min),
                min(end_bid, root_end),
                num_partitions,
                num_shards,
                batch_size,
                (resume_position or {}).get("copy_roots", {}).get(root),
            )
            left = {shard: n + root_left.get(shard, 0) for shard, n in left.items()}
        remaining = min(left.values(), default=0)
        data_end_step = current_step + remaining - 1
        rank_logger.info(
            "Data end (Rust reader, %s): batch_ids [%d, %d), %d batches left -> step %d",
            source,
            start_bid,
            end_bid,
            remaining,
            data_end_step,
        )
        return data_end_step

    def _rust_shard_batches(
        self,
        topic_dir: str,
        start_bid: int,
        end_bid: int,
        num_partitions: int,
        num_shards: int,
        batch_size: int,
        resume_position: DataPosition | None,
    ) -> tuple[dict[int, int], str]:
        shards_per_partition = _shards_per_partition(num_partitions, num_shards)
        owners = num_shards // shards_per_partition
        pps = num_partitions // owners
        if end_bid <= start_bid:
            return {}, "empty range"
        manifest = load_batch_manifest(topic_dir)
        if manifest is not None:
            matrix = manifest.rows_matrix(topic_dir, start_bid, end_bid, num_partitions)
            source = "batch manifest"

            def rows(bid: int) -> np.ndarray:
                return matrix[bid - start_bid]
        elif topic_dir != self.path:
            raise ValueError(f"copy root {topic_dir} has no readable .batch_manifest.json")
        else:
            sample = parquet_file(batch_path(topic_dir, 0, start_bid)).metadata.num_rows
            uniform = np.full(num_partitions, sample, dtype=np.int64)
            source = f"{sample}-row sample file"

            def rows(bid: int) -> np.ndarray:
                return uniform

        block_shuffle = not self.is_eval and self.shuffle_mode == ShuffleMode.BLOCK_SHUFFLE
        left: dict[int, int] = {}
        for shard in range(num_shards):
            if shard % shards_per_partition and not self.block_shuffle_order_per_shard:
                continue
            owner = shard // shards_per_partition
            parts = [owner + k * owners for k in range(pps)]

            def shard_counts(bid: int, parts: list[int] = parts) -> tuple[int, ...]:
                return tuple(int(r) for r in rows(bid)[parts])

            if block_shuffle:
                segments = self._rust_block_segments(
                    start_bid=start_bid,
                    end_bid=end_bid,
                    shard=shard,
                    num_shards=num_shards,
                    pps=pps,
                    batch_size=batch_size,
                    shards_per_partition=shards_per_partition,
                    resume_position=resume_position,
                    row_counts=shard_counts,
                )
            else:
                first, bulk, discard = _time_order_start(
                    lo=start_bid,
                    end=end_bid,
                    resume_position=resume_position,
                    skip_rows=0,
                    batch_size=batch_size,
                    num_shards=num_shards,
                    pps=pps,
                    row_counts=shard_counts,
                )
                segments = [(list(range(first, end_bid)), bulk, discard)]
            left[shard] = _rust_remaining_batches(
                rows=rows,
                parts=parts,
                segments=segments,
                batch_size=batch_size,
                shards_per_partition=shards_per_partition,
            )
        return left, source

    def _rust_block_segments(
        self,
        *,
        start_bid: int,
        end_bid: int,
        shard: int,
        num_shards: int,
        pps: int,
        batch_size: int,
        resume_position: DataPosition | None,
        row_counts: Callable[[int], tuple[int, ...]],
        shards_per_partition: int = 1,
    ) -> list[tuple[list[int], int, int]]:
        window = self.shuffle_window_time_slices
        block = self.block_shuffle_block_time_slices
        window_blocks = window // max(1, block)
        shard_arg = shard if self.block_shuffle_order_per_shard else None
        segments: list[tuple[list[int], int, int]] = []
        window_start = (start_bid // window) * window
        resume = None
        if resume_position is not None:
            resume = _block_shuffle_resume(
                resume_position,
                batch_size=batch_size,
                num_shards=num_shards,
                block=block,
                window_blocks=window_blocks,
                seed=self.shuffle_seed,
                per_shard=self.block_shuffle_order_per_shard,
                start_batch_id=start_bid,
                end_batch_id=end_bid,
            )
        if resume is not None:
            window_start = resume.window_start
            done, left = _replay_block_segments(
                window_start=window_start,
                block=block,
                window_blocks=window_blocks,
                seed=self.shuffle_seed,
                shard=shard_arg,
                segments=resume.segments,
            )
            lo, hi = max(start_bid, window_start), min(end_bid, window_start + window)
            bids = [b for b in range(lo, hi) if b not in done]
            skip, discard = 0, 0
            current = left[0] if left and lo <= left[0] < hi and left[0] not in done else None
            if current is not None and resume.drop_in_progress:
                bids.remove(current)
                skip = resume.next_file_rows
            elif current is not None:
                rows_in = (
                    resume.rows_in_file
                    if resume.rows_in_file is not None
                    else (resume.reads // pps) * batch_size * shards_per_partition
                )
                skip = min(rows_in, max(row_counts(current)))
                discard = resume.reads % pps
            segments.append((bids, skip, discard))
            window_start += window
        while window_start < end_bid:
            lo, hi = max(start_bid, window_start), min(end_bid, window_start + window)
            segments.append((list(range(lo, hi)), 0, 0))
            window_start += window
        return segments

    def _record_batches(
        self,
        *,
        batch_size: int,
        shard_index: int,
        num_shards: int,
        skip_rows: int,
        resume_position: DataPosition | None,
    ) -> Iterator[pa.RecordBatch]:
        copy_roots = self._copy_root_paths()
        if copy_roots:
            self._check_copy_roots()
        reason = self.python_reader_reason()
        if reason is not None:
            self._check_python_readable(reason)
            rank_logger.warning("RustParquetDataset reads through the Python reader: %s", reason)
            return super()._record_batches(
                batch_size=batch_size,
                shard_index=shard_index,
                num_shards=num_shards,
                skip_rows=skip_rows,
                resume_position=resume_position,
            )
        block_shuffle = not self.is_eval and self.shuffle_mode == ShuffleMode.BLOCK_SHUFFLE
        if block_shuffle:
            window, block = self.shuffle_window_time_slices, self.block_shuffle_block_time_slices
            if block <= 0 or window <= 0:
                raise ValueError(
                    f"block_shuffle needs block_shuffle_block_time_slices ({block}) and "
                    f"shuffle_window_time_slices ({window}) > 0"
                )
            if window % block:
                raise ValueError(
                    f"shuffle_window_time_slices ({window}) must be a multiple of "
                    f"block_shuffle_block_time_slices ({block})"
                )
        RecordBatchProvider = rust_ext.load("xai_recsys_parquet_reader").RecordBatchProvider

        assert self.path is not None, f"Called make() on {self.__class__} but self.path was None"
        topic_dir = self.path
        meta = load_valid_batches_metadata(os.path.join(topic_dir, ".valid_batches.json"))
        if meta is None:
            raise ValueError(f"Cannot load metadata from {topic_dir}/.valid_batches.json")
        num_partitions = meta["num_partitions"]
        if self.num_kafka_partitions and self.num_kafka_partitions != num_partitions:
            rank_logger.warning(
                "num_kafka_partitions=%d but %s has %d partitions; reading all %d",
                self.num_kafka_partitions,
                topic_dir,
                num_partitions,
                num_partitions,
            )
        if num_partitions <= 0:
            raise ValueError("num_partitions is 0, no data to read")
        if num_shards > num_partitions:
            if num_shards % num_partitions:
                raise ValueError(
                    f"{num_shards} shards over {num_partitions} partitions: the shard count must "
                    "be a multiple of the partition count"
                )
            if not block_shuffle:
                raise ValueError(
                    f"{num_shards} shards over {num_partitions} partitions splits rows, which "
                    "only shuffle_mode=block_shuffle supports"
                )
        elif num_partitions % num_shards != 0:
            remedy = (
                "on O2, which only the Rust reader reads, pick a num_shards that divides it"
                if is_object_store_path(topic_dir)
                else "the Python reader (dataset.python_reader=True) splits partitions across shards"
            )
            raise ValueError(
                f"num_partitions ({num_partitions}) must be divisible by num_shards ({num_shards}); "
                f"{remedy}"
            )
        shards_per_partition = _shards_per_partition(num_partitions, num_shards)
        owners = num_shards // shards_per_partition
        rank_in_partition = shard_index % shards_per_partition
        pps = num_partitions // owners
        lookahead_files = max(1, self.lookahead_files // pps) if self.lookahead_files > 0 else 0

        start_batch_id: int | None = None
        range_end: int | None = None
        if self.date_range is not None:
            min_ts = parse_date_bound(self.date_range[0])
            max_ts = parse_date_bound(self.date_range[1])
            if min_ts is not None or max_ts is not None:
                start_batch_id, range_end = resolve_time_range(
                    topic_dir, meta["min_valid_batch"], meta["max_valid_batch"], min_ts, max_ts
                )
                if max_ts is None:
                    range_end = None

        def end_batch_id() -> int | None:
            ends = [e for e in (range_end, self._end_batch_id) if e is not None]
            return min(ends) if ends else None

        self._rust_window_start = None
        self._position = None
        self._rb_provider = None
        for root in [topic_dir, *copy_roots]:
            if is_object_store_path(root):
                rank_logger.info(
                    "Rust reader: %s through O2 at %s",
                    root,
                    settings.DATASET_OBJECT_STORE_ENDPOINT,
                )

        def make_provider(root: str, **kw: object) -> Iterable[pa.RecordBatch]:
            o2: dict = (
                dict(o2_endpoint=settings.DATASET_OBJECT_STORE_ENDPOINT)
                if is_object_store_path(root)
                else {}
            )
            return RecordBatchProvider(
                topic_dir=root,
                batch_size=batch_size * shards_per_partition,
                num_shards=owners,
                shard_index=shard_index // shards_per_partition,
                queue_size=self.queue_size,
                num_partitions=num_partitions,
                lookahead_files=lookahead_files,
                row_split=(
                    (shards_per_partition, rank_in_partition) if shards_per_partition > 1 else None
                ),
                **o2,
                **kw,
            )

        def root_reads(
            root: str,
            start: int | None,
            end_fn: Callable[[], int | None],
            skip: int,
            resume: DataPosition | None,
            on_window: Callable[[int], None] | None = None,
        ) -> Iterator[tuple[pa.RecordBatch, DataPosition]]:
            common: dict = dict(
                topic_dir=root,
                num_partitions=num_partitions,
                batch_size=batch_size,
                shard_index=shard_index,
                num_shards=num_shards,
                start_batch_id=start,
                end_batch_id=end_fn(),
                skip_rows=skip,
                resume_position=resume,
            )
            provider = functools.partial(make_provider, root)
            if block_shuffle:
                return _block_shuffle_record_batches(
                    provider,
                    window=self.shuffle_window_time_slices,
                    block=self.block_shuffle_block_time_slices,
                    seed=self.shuffle_seed,
                    per_shard=self.block_shuffle_order_per_shard,
                    end_batch_id_fn=end_fn,
                    on_window=on_window,
                    shards_per_partition=shards_per_partition,
                    **common,
                )
            return _time_order_record_batches(provider, **common)

        reads = root_reads(
            topic_dir,
            start_batch_id,
            end_batch_id,
            0 if self.is_eval else skip_rows,
            resume_position,
            self._set_rust_window_start,
        )
        if not copy_roots:
            return self._tracking_position(reads)
        if resume_position is None and skip_rows > 0:
            raise ValueError(
                "copy_roots cannot resume from skip_rows without a data position: turn it on "
                "at the start of a run"
            )
        resumed = (resume_position or {}).get("copy_roots", {})
        missing = [
            root for root in copy_roots if resume_position is not None and root not in resumed
        ]
        if missing:
            raise ValueError(
                f"The checkpoint's data position has no copy_roots {missing}: turn copy_roots on "
                "at the start of a run (or load= this checkpoint with reset_data_position=True)"
            )

        def capped(end: int) -> Callable[[], int | None]:
            return lambda: end if end_batch_id() is None else min(end_batch_id(), end)

        start = max(start_batch_id or 0, meta["min_valid_batch"])
        lanes = [
            root_reads(
                root,
                start,
                capped(self._copy_root_bounds_of(root, num_partitions)[1]),
                0,
                resumed.get(root),
            )
            for root in copy_roots
        ]
        return self._tracking_position(
            self._merged_positions(
                reads,
                lanes,
                copy_roots,
                resume_position,
                self._visit_key(block_shuffle, shard_index),
            )
        )

    def _visit_key(
        self, block_shuffle: bool, shard_index: int
    ) -> Callable[[DataPosition], tuple[int, ...]]:
        if not block_shuffle:
            return lambda position: (position["last_batch_id"],)
        block = self.block_shuffle_block_time_slices
        window_blocks = self.shuffle_window_time_slices // block
        shard = shard_index if self.block_shuffle_order_per_shard else None

        @functools.cache
        def ranks(window_start: int) -> dict[int, int]:
            order = _block_shuffle_order(self.shuffle_seed, window_start, window_blocks, shard)
            return {j: rank for rank, j in enumerate(order)}

        def key(position: DataPosition) -> tuple[int, ...]:
            window_start = position["block_shuffle_window_start"]
            bid = position["last_batch_id"]
            return (window_start, ranks(window_start)[(bid - window_start) // block], bid)

        return key

    @staticmethod
    def _merged_positions(
        reads: Iterator[tuple[pa.RecordBatch, DataPosition]],
        lanes: list[Iterator[tuple[pa.RecordBatch, DataPosition]]],
        copy_roots: list[str],
        resume_position: DataPosition | None,
        visit_key: Callable[[DataPosition], tuple[int, ...]],
    ) -> Iterator[tuple[pa.RecordBatch, DataPosition | None]]:
        resumed = dict(resume_position or {})
        last_lanes = resumed.pop("copy_roots", {})
        last: list[DataPosition | None] = [
            cast(DataPosition, resumed) if resume_position else None,
            *(last_lanes.get(root) for root in copy_roots),
        ]
        for i, record_batch, position in _merge_by_visit([reads, *lanes], visit_key):
            last[i] = position
            fresh = last[0]
            if fresh is None:
                yield record_batch, None
                continue
            copies = dict(zip(copy_roots, last[1:]))
            yield record_batch, cast(DataPosition, {**fresh, "copy_roots": copies})

    def _set_rust_window_start(self, window_start: int) -> None:
        self._rust_window_start = window_start

    def _tracking_position(
        self, reads: Iterator[tuple[pa.RecordBatch, DataPosition | None]]
    ) -> Iterator[pa.RecordBatch]:
        self._rb_provider = None
        for record_batch, position in reads:
            self._position = position
            yield record_batch
