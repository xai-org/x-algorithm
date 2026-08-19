# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import dataclasses
import logging
import traceback
from collections.abc import Iterator
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread

import pyarrow.parquet as pq
from xai_configlib import configclass

from xrex.data import rust_ext
from xrex.data.parquet_recsys import PhoenixDataset, pad_batch
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


@configclass
class RustParquetDataset(PhoenixDataset):
    queue_size: int = 4

    _position: DataPosition | None = dataclasses.field(init=False, default=None, repr=False)

    def get_data_position(self) -> DataPosition | None:
        return self._position

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
        RecordBatchProvider = rust_ext.load("xai_recsys_parquet_reader").RecordBatchProvider

        if self.use_conversion_labels:
            raise NotImplementedError(
                "use_conversion_labels is not supported by RustParquetDataset: the "
                "conversion-label sidecar join is implemented in "
                "InterleavingRecordBatchProvider only. Use PhoenixDataset."
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
                        sid_num_levels=(dataset.sid_num_levels if dataset.use_post_sid else 0),
                        compute_post_unexplored_label=dataset.compute_post_unexplored_label,
                    )

                    if record_batch.num_rows < batch_size:
                        rank_logger.warning(
                            f"Padding batch of size {record_batch.num_rows} to {batch_size}"
                        )
                        batch = pad_batch(batch, batch_size)

                    reads_in_batch += 1
                    total_reads += 1
                    new_batch_id = actual_start + total_reads // reads_per_batch_id
                    if new_batch_id != current_batch_id:
                        current_batch_id = new_batch_id
                        reads_in_batch = 0

                    dataset._position = DataPosition(
                        last_batch_id=current_batch_id,
                        rows_read_in_batch=reads_in_batch,
                        batch_size=batch_size,
                    )

                    while not stop_event.is_set():
                        try:
                            queue.put(batch, timeout=1.0)
                            break
                        except Full:
                            continue
                    else:
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
