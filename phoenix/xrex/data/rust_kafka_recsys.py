# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator

from xai_configlib import configclass

from xrex.data import rust_ext
from xrex.data.parquet_recsys import load_global_ids_from_parquet_file
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.data.streaming.kafkaconsumer import ConsumerMode, _check_reset_sentinel
from xrex.data.streaming.kafkaloader import (
    PhoenixKafkaDataset,
    _resolve_sasl_password,
)

rank_logger = logging.getLogger("rank")


_QSIZE_LOG_INTERVAL_SECS = 1.0


@configclass
class RustKafkaDataset(PhoenixKafkaDataset):
    queue_size: int = 1000

    keep_per_factor: float = 1.0

    drop_catchup_window_secs: float = 7200.0

    drop_min_staleness_secs: float = 10.0

    drop_staleness_band_secs: float = 30.0

    drop_min_keep_rate: float = 0.2

    drop_update_window_secs: float = 30.0

    def _warn_ignored_inherited_fields(self) -> None:
        problems: list[tuple[str, object, str]] = []

        if self.consumer_mode is not ConsumerMode.PER_PARTITION:
            problems.append(
                (
                    "consumer_mode",
                    self.consumer_mode,
                    "the Rust path is PER_PARTITION-only by design "
                    "(static membership + assign + split_partition_queue); "
                    "LEGACY mode requires group-protocol rebalancing which "
                    "the Rust path explicitly avoids",
                )
            )

        if self.per_partition_poll is not True:
            problems.append(
                (
                    "per_partition_poll",
                    self.per_partition_poll,
                    "applies only to the Python LEGACY consumer mode; "
                    "the Rust path uses split_partition_queue + N tokio "
                    "tasks instead",
                )
            )

        _DROP_KNOB_REASON = (
            "the Rust drop mode is staleness-based and computes its own "
            "drop rate; use drop_catchup_window_secs / "
            "drop_min_staleness_secs / drop_min_keep_rate instead"
        )
        for field_name in (
            "partition_high_watermark_offset_lag",
            "partition_low_watermark_offset_lag",
            "drop_max_drop_rate",
            "drop_increasing_duration_secs",
        ):
            value = getattr(self, field_name)
            if value != getattr(PhoenixKafkaDataset, field_name):
                problems.append((field_name, value, _DROP_KNOB_REASON))

        if not problems:
            return
        lines = [f"  - {name}={value!r}  ({reason})" for name, value, reason in problems]
        rank_logger.warning(
            "RustKafkaDataset: the following inherited PhoenixKafkaDataset config "
            "fields were set to non-default values but the Rust path does not "
            "honor them — they will have no effect on this run:\n%s",
            "\n".join(lines),
        )

    def _async_kafka_consumer_arrow(
        self,
        example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]],
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        stop_event: threading.Event | None = None,
        reset_event: threading.Event | None = None,
    ) -> None:
        del run_server

        rank_logger.info("[rust-kafka] Starting consumer thread")

        self._warn_ignored_inherited_fields()

        if self.num_kafka_partitions is None:
            raise ValueError(
                "RustKafkaDataset requires num_kafka_partitions to be set. "
                "In normal trainer use this is auto-discovered by "
                "PhoenixKafkaDataset.ensure_partition_count() during trainer "
                "init; if you are calling make() directly, either call "
                "self.ensure_partition_count() first or set "
                "num_kafka_partitions explicitly."
            )

        if self.num_global_negatives_per_example > 0:
            (
                self.global_post_ids,
                self.global_author_ids,
                _,
                self.global_post_sids,
            ) = load_global_ids_from_parquet_file(
                self.global_ids_file_path,
                read_post_sid=self.use_post_sid,
                sid_num_levels=self.sid_num_levels,
            )

        reload_thread = self._start_global_ids_reload_thread(stop_event)

        RecordBatchProvider = rust_ext.load("xai_recsys_kafka_reader").RecordBatchProvider

        provider: RecordBatchProvider | None = None
        reset_watcher: threading.Thread | None = None

        try:
            sasl_password = _resolve_sasl_password(self.bootstrap_servers)

            provider = RecordBatchProvider(
                topic=self.topic_name,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                batch_size=batch_size,
                num_shards=num_shards,
                shard_index=shard_index,
                num_partitions=self.num_kafka_partitions,
                sasl_mechanism=self.sasl_mechanism,
                sasl_username=self.sasl_plain_username,
                sasl_password=sasl_password,
                auto_offset_reset=self.auto_offset_reset,
                queue_size=self.queue_size,
                reset_to_latest=self.reset_to_latest,
                seek_to_offsets=(None if self.reset_to_latest else self.seek_to_offset),
                seek_to_timestamp_ms=(None if self.reset_to_latest else self.seek_to_timestamp_ms),
                sampling_factor=self.sampling_factor,
                keep_per_factor=self.keep_per_factor,
                required_columns=None,
                optional_columns=None,
                negative_downsample_rate=self.negative_downsample_rate,
                negative_downsample_positive_action_indices=(
                    self.negative_downsample_positive_action_indices
                ),
                enable_drop_on_high_qps=self.enable_drop_on_high_qps,
                drop_catchup_window_secs=self.drop_catchup_window_secs,
                drop_min_staleness_secs=self.drop_min_staleness_secs,
                drop_staleness_band_secs=self.drop_staleness_band_secs,
                drop_min_keep_rate=self.drop_min_keep_rate,
                drop_update_window_secs=self.drop_update_window_secs,
                otel_endpoint=(self.otel_endpoint if self.export_metrics else None),
                otel_export_interval_ms=self.metrics_export_interval_ms,
                worker_rank=shard_index,
                training_name=self.name,
                ranker_index=str(shard_index),
            )

            in_rebaseline = threading.Event()
            reset_watcher = self._start_reset_watcher(
                provider, stop_event, reset_event, in_rebaseline
            )

            rust_queue_cap = provider.queue_capacity()
            python_queue_cap = example_queue.maxsize
            last_qsize_log = time.monotonic()
            batches_since_log = 0
            last_offsets: dict[int, int] | None = None

            for record_batch, offsets in provider:
                if stop_event is not None and stop_event.is_set():
                    break
                if record_batch.num_rows == 0:
                    continue

                example = self.kafka_to_training_batch([record_batch], batch_size, shard_index)

                payload = (example, offsets)
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    try:
                        example_queue.put(payload, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                batches_since_log += 1

                now = time.monotonic()
                if now - last_qsize_log >= _QSIZE_LOG_INTERVAL_SECS:
                    elapsed = now - last_qsize_log
                    batches_per_s = batches_since_log / elapsed if elapsed > 0 else 0.0
                    rows_per_s = batches_per_s * batch_size
                    if in_rebaseline.is_set():
                        in_rebaseline.clear()
                        last_offsets = None
                    if last_offsets is not None and elapsed > 0:
                        in_msgs_per_s = (
                            sum(
                                off - last_offsets[p]
                                for p, off in offsets.items()
                                if p in last_offsets
                            )
                            / elapsed
                        )
                    else:
                        in_msgs_per_s = 0.0
                    last_offsets = offsets
                    last_qsize_log = now
                    batches_since_log = 0
                    line = (
                        f"rust_channel.qsize()={provider.queue_depth()}/{rust_queue_cap} "
                        f"| example_queue.qsize()={example_queue.qsize()}/{python_queue_cap} "
                        f"| in={in_msgs_per_s:.0f} msg/s "
                        f"| out={batches_per_s:.1f} batch/s ({rows_per_s:.0f} rows/s)"
                    )
                    drop_status = provider.drop_mode_status()
                    if drop_status is not None:
                        line += f" | {drop_status}"
                    rank_logger.info("%s", line)
        finally:
            if reset_watcher is not None:
                reset_watcher.join(timeout=2.0)
            if reload_thread is not None:
                reload_thread.join(timeout=5.0)

    def _start_global_ids_reload_thread(
        self, stop_event: threading.Event | None
    ) -> threading.Thread | None:
        if (
            self.num_global_negatives_per_example <= 0
            or self.global_ids_reload_interval_seconds <= 0
        ):
            return None

        interval = self.global_ids_reload_interval_seconds

        def reload_loop() -> None:
            last_reload_time = time.time()
            while not (stop_event is not None and stop_event.is_set()):
                try:
                    if time.time() - last_reload_time >= interval:
                        rank_logger.info("Start to reload global ids...")
                        new_post_ids, new_author_ids, _, new_post_sids = (
                            load_global_ids_from_parquet_file(
                                self.global_ids_file_path,
                                read_post_sid=self.use_post_sid,
                                sid_num_levels=self.sid_num_levels,
                            )
                        )
                        if new_post_ids is not None and new_author_ids is not None:
                            self.global_post_ids = new_post_ids
                            self.global_author_ids = new_author_ids
                            self.global_post_sids = new_post_sids
                            last_reload_time = time.time()
                        else:
                            rank_logger.warning(
                                "Failed to reload global ids, keeping existing values"
                            )
                    if stop_event is not None:
                        stop_event.wait(timeout=max(1.0, interval / 2))
                    else:
                        time.sleep(max(1.0, interval / 2))
                except Exception as e:
                    rank_logger.error(f"Error in global ids reload task: {e}")
                    if stop_event is not None:
                        stop_event.wait(timeout=max(1.0, interval / 2))
                    else:
                        time.sleep(max(1.0, interval / 2))

        thread = threading.Thread(
            target=reload_loop, name="rust-kafka-global-ids-reload", daemon=True
        )
        thread.start()
        return thread

    def _start_reset_watcher(
        self,
        provider,
        stop_event: threading.Event | None,
        reset_event: threading.Event | None,
        in_rebaseline: threading.Event | None = None,
    ) -> threading.Thread | None:
        def watcher() -> None:
            last_reset_ts = time.time()
            while not (stop_event is not None and stop_event.is_set()):
                if reset_event is not None:
                    http_hit = reset_event.wait(timeout=1.0)
                    if http_hit:
                        reset_event.clear()
                else:
                    http_hit = False
                    time.sleep(1.0)
                sentinel_mtime = _check_reset_sentinel(last_reset_ts)
                if sentinel_mtime is not None:
                    last_reset_ts = sentinel_mtime
                if not http_hit and sentinel_mtime is None:
                    continue
                trigger = "HTTP endpoint" if http_hit else "sentinel file"
                rank_logger.info(
                    "RustKafkaDataset: live reset via %s — signaling Rust "
                    "reader to seek all partitions to latest",
                    trigger,
                )
                try:
                    provider.request_seek_to_latest()
                    if in_rebaseline is not None:
                        in_rebaseline.set()
                except Exception as e:
                    if trigger == "HTTP endpoint" and reset_event is not None:
                        reset_event.set()
                    else:
                        last_reset_ts -= 1.0
                    rank_logger.error(
                        "Failed to forward live-reset signal to Rust reader "
                        "(will retry next tick): %s",
                        e,
                    )

        thread = threading.Thread(target=watcher, name="rust-kafka-reset-watcher", daemon=True)
        thread.start()
        return thread

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
        resume_position=None,
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        return super().make(
            batch_size=batch_size,
            shard_index=shard_index,
            num_shards=num_shards,
            run_server=run_server,
            server_hosts=server_hosts,
            server_port=server_port,
            skip_rows=skip_rows,
            keep_and_pad_partial_batch=keep_and_pad_partial_batch,
            prefetch_factor=prefetch_factor,
            resume_position=resume_position,
        )
