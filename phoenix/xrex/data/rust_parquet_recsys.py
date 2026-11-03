# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import functools
import logging
import traceback
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from xai_configlib import configclass
from xrex.data import rust_ext
from xrex.data.parquet_recsys import (
    PhoenixDataset,
    ShuffleMode,
    _block_resume_window,
    _block_shuffle_fingerprint,
    _block_shuffle_ranges,
    _block_shuffle_resume,
    _block_shuffle_sequence,
    _block_window_grew,
    _BlockLog,
    pad_batch,
)
from xrex.data.parquet_recsys_metadata import (
    DataPosition,
    batch_path,
    load_valid_batches_metadata,
    parse_date_bound,
    resolve_time_range,
)
from xrex.data.recsys.recsys_batch import (
    RecsysFeaturesBatch,
    from_record_batch,
)

rank_logger = logging.getLogger("rank")


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
) -> Iterator[tuple[pa.RecordBatch, DataPosition]]:
    window_blocks = window // block
    pps = num_partitions // num_shards
    metadata_path = str(Path(topic_dir) / ".valid_batches.json")
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
            end_batch_id=end_batch_id,
        )
        if block_resume is not None:
            resume = (block_resume.window_start, block_resume.segments)
            resume_reads = block_resume.reads
            resume_drop = block_resume.drop_in_progress
            resume_rows_in_file = block_resume.rows_in_file
            resume_next_file_rows = block_resume.next_file_rows
            window_start = block_resume.window_start
    else:
        pending_skip_reads = skip_rows // (batch_size * num_shards)

    @functools.cache
    def row_count(bid: int) -> int:
        rows = pq.ParquetFile(batch_path(topic_dir, shard_index, bid)).metadata.num_rows
        if rows <= 0:
            raise ValueError(f"Empty parquet file for batch_id {bid}")
        return rows

    while True:
        meta = load_valid_batches_metadata(metadata_path)
        if meta is None:
            return
        end = meta["max_valid_batch"] + 1
        if end_batch_id is not None:
            end = min(end, end_batch_id)
        lo = max(meta["min_valid_batch"], range_start)
        if window_start is None:
            window_start = (lo // window) * window
        if window_start >= end:
            return
        hi = min(end, window_start + window)
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
                window_reads = pps * -(-sum(map(row_count, bids)) // batch_size)
                if pending_skip_reads >= window_reads:
                    pending_skip_reads -= window_reads
                    window_start += window
                    continue
                skip_reads, pending_skip_reads = pending_skip_reads, 0
                bulk, discard = (skip_reads // pps) * batch_size, skip_reads % pps
            elif current is not None:
                rows_in = (
                    resume_rows_in_file
                    if resume_rows_in_file is not None
                    else (skip_reads // pps) * batch_size
                )
                bulk, discard = min(rows_in, row_count(bids[0])), skip_reads % pps
            elif next_file_rows:
                bulk, discard = next_file_rows, 0
            else:
                bulk, discard = 0, 0
            first = 0
            while first < len(bids) and bulk >= row_count(bids[first]):
                bulk -= row_count(bids[first])
                first += 1
            if first < len(bids):
                yield from _block_shuffle_window(
                    make_provider(skip_rows=bulk, batch_ranges=_to_ranges(bids[first:])),
                    bids=bids[first:],
                    row_count=row_count,
                    bulk=bulk,
                    discard=discard,
                    pps=pps,
                    batch_size=batch_size,
                    window_start=window_start,
                    num_shards=num_shards,
                    fingerprint=fingerprint,
                    on_batch_id=block_log.batch_id,
                    segments_at=functools.partial(
                        _segments_while_reading, before, after, first_rest - first
                    ),
                )
        if hi < window_start + window and _block_window_grew(metadata_path, end_batch_id, hi):
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


def _block_shuffle_window(
    provider: Iterable[pa.RecordBatch],
    *,
    bids: list[int],
    row_count: Callable[[int], int],
    bulk: int,
    discard: int,
    pps: int,
    batch_size: int,
    window_start: int,
    num_shards: int,
    fingerprint: str,
    on_batch_id: Callable[[int], None],
    segments_at: Callable[[int], list[list[int]]],
) -> Iterator[tuple[pa.RecordBatch, DataPosition]]:
    it = iter(provider)
    for _ in range(discard):
        if next(it, None) is None:
            return
    n = discard
    f, file_start = 0, 0
    for record_batch in it:
        n += 1
        if record_batch.num_rows * 2 < batch_size:
            rank_logger.warning("Skipping record batch that doesn't contain enough rows")
            continue
        rows = bulk + (n // pps) * batch_size
        while f + 1 < len(bids) and rows >= file_start + row_count(bids[f]):
            file_start += row_count(bids[f])
            f += 1
        in_file, file_rows = rows - file_start, row_count(bids[f])
        on_batch_id(bids[f])
        position = DataPosition(
            last_batch_id=bids[f],
            rows_read_in_batch=(
                -(-file_rows // batch_size) * pps
                if in_file >= file_rows
                else (in_file // batch_size) * pps + n % pps
            ),
            batch_size=batch_size,
        )
        position["block_shuffle_window_start"] = window_start
        position["block_shuffle_segments"] = segments_at(f)
        position["block_shuffle_num_shards"] = num_shards
        position["block_shuffle_fingerprint"] = fingerprint
        if in_file < file_rows:
            position["block_shuffle_rows_in_file"] = in_file
            if n % pps and in_file + batch_size > file_rows:
                position["block_shuffle_rows_in_next_file"] = in_file + batch_size - file_rows
        yield record_batch, position


@configclass
class RustParquetDataset(PhoenixDataset):
    queue_size: int = 4

    _position: DataPosition | None = dataclasses.field(init=False, default=None, repr=False)

    def get_data_position(self) -> DataPosition | None:
        return self._position

    def _counts_steps_from_manifest(self) -> bool:
        return False

    def supports_follow_data_end(self) -> bool:
        return False

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
        block_shuffle = not self.is_eval and self.shuffle_mode == ShuffleMode.BLOCK_SHUFFLE
        file_shuffle = not self.is_eval and (
            self.shuffle_mode == ShuffleMode.FILE_SHUFFLE
            or self.shuffle_in_memory_buffer_rows > 0
            or (self.shuffle_mode == ShuffleMode.NONE and self.shuffle_window_time_slices > 0)
        )
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
            if self.shuffle_in_memory_buffer_rows:
                raise ValueError(
                    "block_shuffle reads each block in time order; leave "
                    "shuffle_in_memory_buffer_rows at 0"
                )
        elif file_shuffle:
            raise NotImplementedError(
                "file_shuffle (shuffle_window_time_slices, shuffle_in_memory_buffer_rows) is not "
                "supported by RustParquetDataset. Use PhoenixDataset "
                "(dataset_type=offline_kafka_dump) or shuffle_mode=block_shuffle."
            )

        RecordBatchProvider = rust_ext.load("xai_recsys_parquet_reader").RecordBatchProvider

        if self.use_conversion_labels:
            raise NotImplementedError(
                "use_conversion_labels is not supported by RustParquetDataset: the "
                "conversion-label sidecar join is implemented in "
                "InterleavingRecordBatchProvider only. Use PhoenixDataset."
            )
        if self.id_list_sidecar_root is not None:
            raise NotImplementedError(
                "id_list_sidecar_root is not supported by RustParquetDataset: the id-list "
                "sidecar join is implemented in InterleavingRecordBatchProvider only. "
                "Use PhoenixDataset, or unset id_list_sidecar_root to read the native column."
            )

        queue: Queue = Queue(maxsize=prefetch_factor)
        stop_event = Event()
        SENTINEL = object()
        del run_server, server_hosts, server_port, keep_and_pad_partial_batch

        if self.continuous:
            raise ValueError(
                "RustParquetDataset does not support continuous=True. "
                "Use PhoenixDataset for continuous/tailing mode."
            )

        dataset = self

        def emit(record_batch: pa.RecordBatch, position: DataPosition) -> bool:
            batch = from_record_batch(
                record_batch,
                dataset.history_seq_len,
                dataset.candidate_seq_len,
                dataset.num_negatives_per_example,
                dataset.output_vocab_size,
                dataset.num_continuous_actions,
                dataset.hash_table,
                dataset.include_candidate_post_ids,
                dataset.num_global_negatives_per_example,
                dataset.global_post_ids,
                dataset.global_author_ids,
                embedding_type=dataset.multimodal_embedding_type,
                search_query_embedding_dim=dataset.search_query_embedding_dim,
                aux_value_events=dataset.aux_value_events,
                sid_num_levels=(dataset.sid_num_levels if dataset.use_post_sid else 0),
                compute_post_unexplored_label=dataset.compute_post_unexplored_label,
                zero_stale_post_14d_candidate_counts=dataset.enable_stale_post,
                stale_post_30d=dataset.enable_stale_post_30d,
                ads_head_masking=dataset.ads_head_masking,
                dump_keep_rate_per_10000=dataset.dump_keep_rate_per_10000,
                ads_pixel_head_masking=dataset.ads_pixel_head_masking,
                ads_late_window_heads=dataset.ads_late_window_heads,
                reply_parent_mode=dataset.reply_parent_mode,
                page_records=dataset.page_records,
                id_list_specs=dataset.id_list_specs or None,
                history_url_embedding_dim=dataset.history_url_embedding_dim,
                alloc=dataset._alloc,
            )

            if record_batch.num_rows < batch_size:
                rank_logger.warning(
                    f"Padding batch of size {record_batch.num_rows} to {batch_size}"
                )
                batch = pad_batch(batch, batch_size)

            dataset._position = position

            while not stop_event.is_set():
                try:
                    queue.put(batch, timeout=1.0)
                    return True
                except Full:
                    continue
            return False

        def producer() -> None:
            try:
                assert dataset.path is not None, (
                    f"Called make() on {dataset.__class__} but self.path was None"
                )
                topic_dir = dataset.path
                metadata_path = str(Path(topic_dir) / ".valid_batches.json")
                meta = load_valid_batches_metadata(metadata_path)
                if meta is None:
                    raise ValueError(f"Cannot load metadata from {metadata_path}")

                num_partitions = dataset.num_kafka_partitions or meta["num_partitions"]
                if num_partitions <= 0:
                    raise ValueError("num_partitions is 0, no data to read")
                if num_partitions % num_shards != 0:
                    raise ValueError(
                        f"num_partitions ({num_partitions}) must be divisible by "
                        f"num_shards ({num_shards})"
                    )

                start_batch_id: int | None = None
                end_batch_id: int | None = None
                if dataset.date_range is not None:
                    min_ts = parse_date_bound(dataset.date_range[0])
                    max_ts = parse_date_bound(dataset.date_range[1])
                    if min_ts is not None or max_ts is not None:
                        if meta is None:
                            raise ValueError(f"Cannot load metadata from {metadata_path}")
                        start_batch_id, end_batch_id = resolve_time_range(
                            topic_dir,
                            meta["min_valid_batch"],
                            meta["max_valid_batch"],
                            min_ts,
                            max_ts,
                        )

                if block_shuffle:
                    for record_batch, position in _block_shuffle_record_batches(
                        lambda **kw: RecordBatchProvider(
                            topic_dir=topic_dir,
                            batch_size=batch_size,
                            num_shards=num_shards,
                            shard_index=shard_index,
                            queue_size=dataset.queue_size,
                            num_partitions=num_partitions,
                            **kw,
                        ),
                        topic_dir=topic_dir,
                        num_partitions=num_partitions,
                        batch_size=batch_size,
                        shard_index=shard_index,
                        num_shards=num_shards,
                        window=dataset.shuffle_window_time_slices,
                        block=dataset.block_shuffle_block_time_slices,
                        seed=dataset.shuffle_seed,
                        per_shard=dataset.block_shuffle_order_per_shard,
                        start_batch_id=start_batch_id,
                        end_batch_id=end_batch_id,
                        skip_rows=skip_rows,
                        resume_position=resume_position,
                    ):
                        if not emit(record_batch, position):
                            break
                    return

                partitions_per_shard = num_partitions // num_shards
                effective_start = start_batch_id or (meta["min_valid_batch"] if meta else 0)
                discard_count = 0

                if resume_position is not None:
                    resume_bid = resume_position["last_batch_id"]
                    saved_reads = resume_position["rows_read_in_batch"]
                    saved_bs = resume_position.get("batch_size")

                    if saved_bs is not None and saved_bs != batch_size:
                        total_rows = saved_reads * saved_bs
                        saved_reads = total_rows // batch_size
                        rank_logger.info(
                            "Adjusting resume skip for batch_size change: "
                            "saved_reads=%d * saved_bs=%d = %d rows → %d reads * bs=%d",
                            resume_position["rows_read_in_batch"],
                            saved_bs,
                            total_rows,
                            saved_reads,
                            batch_size,
                        )

                    start_batch_id = resume_bid
                    bulk_per_partition = (saved_reads // partitions_per_shard) * batch_size
                    discard_count = saved_reads % partitions_per_shard

                    rank_logger.info(
                        "Resuming from DataPosition: start_batch_id=%d, "
                        "rust_skip_rows=%d (per partition), "
                        "discard_count=%d interleaved reads",
                        start_batch_id,
                        bulk_per_partition,
                        discard_count,
                    )
                else:
                    raw_skip_rows = skip_rows if not dataset.is_eval else 0
                    bulk_per_partition = 0
                    if raw_skip_rows > 0:
                        sample_path = batch_path(topic_dir, shard_index, effective_start)
                        rows_per_file = pq.ParquetFile(sample_path).metadata.num_rows
                        total_rows_per_batch_id = rows_per_file * num_partitions
                        if total_rows_per_batch_id > 0:
                            skip_batch_ids = raw_skip_rows // total_rows_per_batch_id
                            start_batch_id = effective_start + skip_batch_ids
                            remaining_rows = raw_skip_rows % total_rows_per_batch_id
                            per_partition_rows = remaining_rows // num_partitions
                            bulk_per_partition = per_partition_rows
                            leftover_rows = remaining_rows % num_partitions
                            discard_count = leftover_rows // batch_size

                provider = RecordBatchProvider(
                    topic_dir=topic_dir,
                    batch_size=batch_size,
                    num_shards=num_shards,
                    shard_index=shard_index,
                    skip_rows=bulk_per_partition,
                    queue_size=dataset.queue_size,
                    start_batch_id=start_batch_id,
                    end_batch_id=end_batch_id,
                    num_partitions=num_partitions,
                )

                if discard_count > 0:
                    rank_logger.info(
                        "Discarding %d batches to reach resume position", discard_count
                    )
                    provider_iter = iter(provider)
                    for _ in range(discard_count):
                        next(provider_iter)
                    provider = provider_iter

                actual_start = start_batch_id or (meta["min_valid_batch"] if meta else 0)
                sample_path = batch_path(topic_dir, shard_index, actual_start)
                rows_per_file = pq.ParquetFile(sample_path).metadata.num_rows
                reads_per_file = rows_per_file // batch_size
                reads_per_batch_id = partitions_per_shard * reads_per_file

                bulk_reads = (bulk_per_partition // batch_size) * partitions_per_shard
                total_reads = bulk_reads + discard_count
                current_batch_id = actual_start + total_reads // reads_per_batch_id
                reads_in_batch = total_reads % reads_per_batch_id

                for record_batch in provider:
                    if record_batch.num_rows * 2 < batch_size:
                        rank_logger.warning(
                            "Skipping record batch that doesn't contain enough rows"
                        )
                        continue

                    reads_in_batch += 1
                    total_reads += 1
                    new_batch_id = actual_start + total_reads // reads_per_batch_id
                    if new_batch_id != current_batch_id:
                        current_batch_id = new_batch_id
                        reads_in_batch = 0

                    position = DataPosition(
                        last_batch_id=current_batch_id,
                        rows_read_in_batch=reads_in_batch,
                        batch_size=batch_size,
                    )
                    if not emit(record_batch, position):
                        break

            except Exception as e:
                rank_logger.error("Producer thread failed: %s", traceback.format_exc())
                try:
                    queue.put(e, timeout=1.0)
                except Full:
                    pass
            finally:
                try:
                    queue.put(SENTINEL, timeout=1.0)
                except Full:
                    pass

        thread = Thread(target=producer, daemon=True)
        thread.start()

        def generator():
            try:
                while True:
                    item = queue.get()
                    if item is SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item, None
            finally:
                stop_event.set()

        return generator()
