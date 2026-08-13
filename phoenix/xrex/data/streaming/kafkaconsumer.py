# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import asyncio
import enum
import gc
import io
import logging
import os
import queue
import ssl
import tempfile
import threading
import time
import traceback
import typing
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
from aiokafka import AIOKafkaConsumer, ConsumerRecord
from aiokafka.coordinator.assignors.sticky.sticky_assignor import StickyPartitionAssignor
from aiokafka.structs import TopicPartition

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch

DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES = ""


def parse_positive_action_indices(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part)


def _has_positive_candidate(
    rb: pa.RecordBatch,
    positive_action_indices: tuple[int, ...],
) -> bool:
    action_col = rb.column("actionNameMultiHotSeqSeq")
    vocab = action_col.type.value_type.list_size
    indices = tuple(idx for idx in positive_action_indices if 0 <= idx < vocab)
    if not indices:
        return False

    mask_col = "newEventMask" if "newEventMask" in rb.schema.names else "newEventMaskSeq"
    new_event = rb.column(mask_col).values.to_numpy(zero_copy_only=False).reshape(rb.num_rows, -1)
    if not new_event.any():
        return False

    actions = action_col.values.flatten().to_numpy(zero_copy_only=False)
    actions = actions.reshape(rb.num_rows, -1, vocab)
    return bool(actions[new_event][:, indices].any())


_OTEL_AVAILABLE = False
try:
    from opentelemetry import metrics
    from opentelemetry.metrics import Observation

    _OTEL_AVAILABLE = True
except ImportError:
    metrics = None

_consumer_messages_per_poll = None
_consumer_poll_latency = None
_consumer_uptime_gauge = None
_consumer_dropped_records = None
_consumer_deserialize_errors = None
_consumer_end_offsets_polled_count = None
_consumer_drop_mode_active = None
_consumer_partition_lag = None
_consumer_partition_consumed_offset = None
_consumer_partition_end_offset = None
_consumer_metrics_initialized = False

_consumer_start_times: dict[tuple[str, str, str], int] = {}

DEFAULT_DROP_MODE_QPS_THRESHOLD = 55000
DROP_MODE_QPS_DURATION_SECS = 60


def _init_consumer_metrics(
    lag_tracker: "PartitionLagTracker | None" = None, metric_attrs: dict[str, str] | None = None
) -> bool:
    global _consumer_messages_per_poll, _consumer_poll_latency, _consumer_uptime_gauge
    global _consumer_dropped_records, _consumer_deserialize_errors
    global _consumer_drop_mode_active, _consumer_partition_lag
    global _consumer_partition_consumed_offset, _consumer_partition_end_offset
    global _consumer_metrics_initialized, _consumer_end_offsets_polled_count

    if _consumer_metrics_initialized:
        return True

    if not _OTEL_AVAILABLE:
        return False

    assert metrics is not None

    try:
        meter = metrics.get_meter("recsys_kafka_consumer", "1.0.0")

        _consumer_messages_per_poll = meter.create_histogram(
            name="recsys_kafka_consumer_messages_per_poll",
            description="Number of messages received per Kafka poll",
            unit="1",
        )

        _consumer_poll_latency = meter.create_histogram(
            name="recsys_kafka_consumer_poll_latency_seconds",
            description="Latency of Kafka consumer poll operations",
            unit="1",
        )

        _consumer_dropped_records = meter.create_counter(
            name="recsys_kafka_consumer_dropped_records_total",
            description="Total number of Kafka records dropped due to high training consumer QPS and full queue",
            unit="1",
        )

        _consumer_deserialize_errors = meter.create_counter(
            name="recsys_kafka_consumer_deserialize_errors_total",
            description="Total number of Kafka messages that failed Arrow IPC deserialization",
            unit="1",
        )

        _consumer_drop_mode_active = meter.create_counter(
            name="recsys_kafka_consumer_drop_mode_activations_total",
            description="Number of times drop mode was activated due to sustained high training consumer QPS",
            unit="1",
        )
        _consumer_end_offsets_polled_count = meter.create_counter(
            name="recsys_kafka_consumer_end_offsets_polled_count_offsets_total",
            description="Number of times end offsets were polled",
            unit="1",
        )

        def _get_partition_lag_callback(_options) -> list[Observation]:
            if lag_tracker is None:
                return []

            base_attrs = dict(metric_attrs or {})

            return [
                Observation(
                    lag_tracker.get_partition_lag_offsets(partition),
                    {**base_attrs, "partition": str(partition)},
                )
                for partition in lag_tracker._partition_lag.keys()
            ]

        _consumer_partition_lag = meter.create_observable_gauge(
            name="recsys_kafka_consumer_partition_lag_offsets",
            description="Kafka consumer lag per partition (end_offset - latest_offset)",
            unit="1",
            callbacks=[_get_partition_lag_callback],
        )

        def _get_partition_consumed_offset_callback(_options) -> list[Observation]:
            if lag_tracker is None:
                return []
            base_attrs = dict(metric_attrs or {})
            return [
                Observation(
                    offset,
                    {**base_attrs, "partition": str(partition)},
                )
                for partition, offset in lag_tracker._latest_consumed_offsets.items()
            ]

        _consumer_partition_consumed_offset = meter.create_observable_gauge(
            name="recsys_kafka_consumer_partition_consumed_offset",
            description=(
                "Per-partition latest Kafka offset returned by getmany() "
                "(fetch-side, before round-robin merger). Compare against "
                "recsys_kafka_partition_delivered_offset to see merger queueing lag."
            ),
            unit="1",
            callbacks=[_get_partition_consumed_offset_callback],
        )

        def _get_partition_end_offset_callback(_options) -> list[Observation]:
            if lag_tracker is None:
                return []
            base_attrs = dict(metric_attrs or {})
            return [
                Observation(
                    entry[1],
                    {**base_attrs, "partition": str(partition)},
                )
                for partition, entry in lag_tracker._partition_lag.items()
            ]

        _consumer_partition_end_offset = meter.create_observable_gauge(
            name="recsys_kafka_consumer_partition_end_offset",
            description=(
                "Per-partition Kafka end offset (latest available in the topic) as last "
                "polled via consumer.end_offsets(). Refreshed every "
                "FETCH_END_OFFSETS_INTERVAL_IN_CONSUMER_ROUNDS polls."
            ),
            unit="1",
            callbacks=[_get_partition_end_offset_callback],
        )

        def _get_uptime_callback(_options):
            assert metrics is not None
            current_time_ms = int(time.time() * 1000)
            for (training_name, topic, ranker_idx), start_time_ms in _consumer_start_times.items():
                uptime_ms = current_time_ms - start_time_ms
                yield metrics.Observation(
                    uptime_ms,
                    {"training_name": training_name, "topic": topic, "ranker_index": ranker_idx},
                )

        _consumer_uptime_gauge = meter.create_observable_gauge(
            name="recsys_kafka_consumer_uptime_ms_milliseconds",
            description="Kafka consumer uptime in milliseconds",
            unit="1",
            callbacks=[_get_uptime_callback],
        )

        _consumer_metrics_initialized = True
        rank_logger.info("Initialized Kafka consumer metrics")
        return True
    except Exception as e:
        rank_logger.warning(f"Failed to initialize consumer metrics: {e}")
        return False


log_level_str = os.environ.get("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
rank_logger = logging.getLogger("rank")


KAFKA_MAX_FETCH_BYTES = 50 * 1024 * 1024
KAFKA_MIN_FETCH_BYTES = 64 * 1024
KAFKA_MAX_POLL_RECORDS = 256
KAFKA_POLL_TIMEOUT = 1000
KAFKA_PER_PARTITION_POLL_TIMEOUT = 300
KAFKA_SESSION_TIMEOUT = 90000
KAFKA_HEARTBEAT_INTERVAL = 30000
KAFKA_MAX_POLL_INTERVAL = 600000
KAFKA_REBALANCE_TIMEOUT = 90000
KAFKA_AUTO_COMMIT_INTERVAL_MS = 5000

KAFKA_RESET_SENTINEL_PATH = os.environ.get(
    "KAFKA_RESET_SENTINEL_PATH",
    os.path.join(tempfile.gettempdir(), "kafka_reset_to_latest"),
)

MESSAGE_QUEUE_SIZE = 1000
FETCH_END_OFFSETS_INTERVAL_IN_CONSUMER_ROUNDS = 8

MAX_COMMIT_FAILURES = 3
COMMIT_FAILURE_SLEEP_TIME = 15.0
GC_INTERVAL = 8192


def deserialize_arrow_ipc(data: bytes) -> pa.RecordBatch:
    try:
        t = time.time()
        stream = io.BytesIO(data)
        reader = ipc.RecordBatchStreamReader(stream)
        record_batch = reader.read_next_batch()
        reader.close()
        elapsed = time.time() - t
        rank_logger.debug(f"Timer: Deserialized Arrow IPC data in {elapsed:.5f} seconds")
        return record_batch
    except Exception as err:
        traceback.print_exc()
        print(f"Failed to deserialize Arrow IPC data: {err}")
        raise err


async def _deserialize_messages(
    batch: list[tuple[pa.RecordBatch, int, int]],
    batch_size: int,
    msg_queue: asyncio.Queue[ConsumerRecord],
    example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]],
    post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
    count: int = 0,
    sampling_factor: int = 1,
    negative_downsample_rate: int = 1,
    positive_action_indices: tuple[int, ...] = (),
) -> tuple[list[tuple[pa.RecordBatch, int, int]], int]:
    msg = await msg_queue.get()

    value = msg.value
    value = typing.cast(bytes, value)
    partition, offset = msg.partition, msg.offset
    count += 1
    if count % sampling_factor != 0:
        msg_queue.task_done()
        return batch, count

    try:
        user_action_sequence = await asyncio.to_thread(deserialize_arrow_ipc, value)
    except RuntimeError as e:
        if "cannot schedule new futures after shutdown" in str(e):
            msg_queue.task_done()
            raise _ExecutorShutdown() from e
        raise
    except Exception as e:
        headers_str = ", ".join(f"{k}={v!r}" for k, v in (msg.headers or []))
        payload_preview = repr(value[:100]) if value else "<empty>"
        rank_logger.warning(
            f"Skipping message from partition={partition} offset={offset}: "
            f"failed to deserialize Arrow IPC data: {e}\n"
            f"  headers: [{headers_str}]\n"
            f"  payload (first 100 bytes): {payload_preview}"
        )
        if _consumer_deserialize_errors is not None:
            _consumer_deserialize_errors.add(1, {"partition": str(partition)})
        msg_queue.task_done()
        return batch, count

    if negative_downsample_rate > 1:
        if not positive_action_indices:
            raise ValueError(
                "negative_downsample_positive_action_indices must be non-empty "
                "when negative_downsample_rate > 1"
            )
        sample_weight = 1.0
        if not _has_positive_candidate(user_action_sequence, positive_action_indices):
            if hash((partition, offset)) % negative_downsample_rate != 0:
                msg_queue.task_done()
                return batch, count
            sample_weight = float(negative_downsample_rate)
        n = user_action_sequence.num_rows
        weight_array = pa.array([sample_weight] * n, type=pa.float32())
        existing_names = user_action_sequence.schema.names
        if "sampleWeight" in existing_names:
            keep_indices = [i for i, name in enumerate(existing_names) if name != "sampleWeight"]
            user_action_sequence = user_action_sequence.select(keep_indices)
        user_action_sequence = user_action_sequence.append_column(
            "sampleWeight",
            weight_array,
        )

    batch.append((user_action_sequence, partition, offset))
    msg_queue.task_done()

    if len(batch) < batch_size:
        return batch, count

    while example_queue.full():
        await asyncio.sleep(0.1)

    records: list[pa.RecordBatch] = []
    offsets: dict[int, int] = {}
    for record, tp, ts in batch:
        offsets[tp] = max(offsets.get(tp, ts), ts)
        records.append(record)

    batch_processed = (post_process_fn(records), offsets)
    example_queue.put_nowait(batch_processed)
    return [], count


class _ExecutorShutdown(Exception):
    pass


async def deserialize_messages(
    msg_queue: asyncio.Queue[ConsumerRecord],
    example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]],
    batch_size: int,
    post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
    sampling_factor: int = 1,
    stop_event: threading.Event | None = None,
    negative_downsample_rate: int = 1,
    negative_downsample_positive_action_indices: str = DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES,
):
    batch: list[tuple[pa.RecordBatch, int, int]] = []
    count = 0
    positive_action_indices = parse_positive_action_indices(
        negative_downsample_positive_action_indices
    )
    if negative_downsample_rate > 1 and not positive_action_indices:
        raise ValueError(
            "negative_downsample_positive_action_indices must be non-empty "
            "when negative_downsample_rate > 1"
        )
    while not (stop_event and stop_event.is_set()):
        try:
            batch, count = await _deserialize_messages(
                batch,
                batch_size,
                msg_queue,
                example_queue,
                post_process_fn,
                count,
                sampling_factor,
                negative_downsample_rate,
                positive_action_indices,
            )
        except _ExecutorShutdown:
            rank_logger.info(
                "Kafka deserializer stopping: ThreadPoolExecutor shut down (process exiting)"
            )
            return
        except Exception as e:
            rank_logger.info(f"Error in deserialization loop at offset {e}")
            traceback.print_exc()
            raise


async def force_kafka_metadata_update(
    consumer: AIOKafkaConsumer, topic: str, max_retries=5, retry_interval_seconds=1
) -> set[int] | None:
    for attempt in range(max_retries):
        await consumer._client.force_metadata_update()
        available_topics = await consumer.topics()
        partitions = consumer.partitions_for_topic(topic)
        if partitions:
            logger.info(
                f"Found {len(partitions)} partitions for topic {topic} on attempt {attempt + 1}"
            )
            return partitions
        logger.warning(
            f"Attempt {attempt + 1}/{max_retries}: No partitions found for topic {topic}, "
            f"topic exists: {topic in available_topics}. Retrying..."
        )
        await asyncio.sleep(retry_interval_seconds)
    return None


async def discover_partition_count(
    topic: str,
    bootstrap_servers: str,
    sasl_mechanism: str,
    sasl_plain_username: str,
    sasl_plain_password: str,
) -> int:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    consumer = AIOKafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        security_protocol="SASL_SSL",
        sasl_kerberos_domain_name="kafka",
        sasl_kerberos_service_name="kafka",
        sasl_mechanism=sasl_mechanism,
        sasl_plain_username=sasl_plain_username,
        sasl_plain_password=sasl_plain_password,
        ssl_context=ssl_ctx,
        request_timeout_ms=30000,
    )
    await consumer.start()
    try:
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            partitions = await force_kafka_metadata_update(consumer, topic)
        if not partitions:
            available = await consumer.topics()
            raise ValueError(
                f"No partitions found for topic '{topic}'. Available topics: {available}"
            )
        count = len(partitions)
        rank_logger.info(f"Discovered {count} partitions for topic '{topic}' from Kafka metadata.")
        return count
    finally:
        try:
            await consumer.stop()
        except asyncio.CancelledError:
            pass


def _range_partitions(total_partitions: int, shard_index: int, num_shards: int) -> list[int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if total_partitions < 0:
        raise ValueError(f"total_partitions must be >= 0, got {total_partitions}")
    start = shard_index * total_partitions // num_shards
    end = (shard_index + 1) * total_partitions // num_shards
    return list(range(start, end))


async def handle_topic_offset(
    consumer: AIOKafkaConsumer,
    topic: str,
    shard_index: int,
    num_shards: int,
    reset_to_latest: bool,
    seek_to_timestamp_ms: int | None = None,
    seek_to_offset: dict[int, int] | None = None,
) -> list[TopicPartition]:
    assert seek_to_timestamp_ms is None or seek_to_offset is None, (
        "seek_to_timestamp_ms and seek_to_offset cannot both be set"
    )
    assert not (seek_to_timestamp_ms is not None and reset_to_latest), (
        "seek_to_timestamp_ms and reset_to_latest cannot both be set"
    )
    rank_logger.info(
        f"Try to get partition info for {topic}, shard_index: {shard_index}, num_shards: {num_shards}."
    )

    partitions: set[int] | None = consumer.partitions_for_topic(topic)
    if not partitions:
        partitions = await force_kafka_metadata_update(consumer, topic)
    if not partitions:
        partitions = await force_kafka_metadata_update(consumer, topic)
    if not partitions:
        available_topics = await consumer.topics()
        rank_logger.error(
            f"No partitions found for topic {topic}, available topics: {available_topics}"
        )
        raise ValueError(
            f"No partitions found for topic {topic}, available topics: {available_topics}"
        )

    total_partitions = len(partitions)

    partition_ids = _range_partitions(total_partitions, shard_index, num_shards)
    if not partition_ids:
        rank_logger.warning(
            f"Worker {shard_index} received 0 partitions (total_partitions={total_partitions}, "
            f"num_shards={num_shards}).  This worker will be idle."
        )
    assigned_partitions = [TopicPartition(topic, i) for i in partition_ids]
    for tp in assigned_partitions:
        rank_logger.info(
            f"Assigned partition {tp} to worker {shard_index}, num_shards: {num_shards}, total_partitions: {total_partitions}."
        )
    consumer.assign(assigned_partitions)
    rank_logger.info(
        f"Range-based assignment: worker {shard_index} got "
        f"{len(assigned_partitions)} of {total_partitions} partitions"
        + (f" ({partition_ids[0]}..{partition_ids[-1]})" if partition_ids else "")
    )
    if reset_to_latest:
        for tp in assigned_partitions:
            await consumer.seek_to_end(tp)
            rank_logger.info(f"Reset partition {tp} to latest offset")
    elif seek_to_timestamp_ms is not None:
        await seek_to_timestamp(consumer, assigned_partitions, seek_to_timestamp_ms)
    elif seek_to_offset is not None:
        await seek_consumer_to_offset(consumer, assigned_partitions, seek_to_offset)
    return assigned_partitions


async def seek_to_timestamp(
    consumer: AIOKafkaConsumer, assigned_partitions: list[TopicPartition], seek_to_timestamp_ms: int
):
    timestamps = {tp: seek_to_timestamp_ms for tp in assigned_partitions}
    offsets = await consumer.offsets_for_times(timestamps)

    for tp in assigned_partitions:
        offset_and_ts = offsets.get(tp)
        if offset_and_ts is not None and offset_and_ts.offset != -1:
            rank_logger.info(f"Seeking to offset {offset_and_ts.offset} for {tp}")
            consumer.seek(tp, offset_and_ts.offset)
        else:
            raise ValueError(f"No offset found for {tp} at timestamp {seek_to_timestamp_ms}")


async def seek_consumer_to_offset(
    consumer: AIOKafkaConsumer, assigned_partitions: list[TopicPartition], offsets: dict[int, int]
):
    for tp in assigned_partitions:
        if tp.partition in offsets:
            offset = offsets[tp.partition]
            rank_logger.info(f"Seeking to offset {offset} for {tp}")
            consumer.seek(tp, offset)
        else:
            raise ValueError(f"No offset found for {tp} at timestamp {offsets}")


class PartitionLagTracker:
    def __init__(self) -> None:
        self._partition_lag: dict[int, tuple[int, int]] = {}
        self._latest_consumed_offsets: dict[int, int] = {}

    def record_consumed_offsets(self, messages: dict[TopicPartition, list[ConsumerRecord]]) -> None:
        for tp, msg_list in messages.items():
            if msg_list:
                max_offset = max(msg.offset for msg in msg_list)
                prev = self._latest_consumed_offsets.get(tp.partition)
                if prev is None or max_offset > prev:
                    self._latest_consumed_offsets[tp.partition] = max_offset

    def update_partition_lag(self, partition: int, latest_offset: int, end_offset: int) -> None:
        self._partition_lag[partition] = (latest_offset, end_offset)

    def get_partition_lag_offsets(self, partition: int) -> int:
        entry = self._partition_lag.get(partition)
        if entry is None:
            return 0
        latest_offset, end_offset = entry
        return max(0, end_offset - latest_offset)

    def get_latest_consumed_offset(self, partition: int) -> int | None:
        latest = self._latest_consumed_offsets.get(partition)
        if latest is not None:
            return latest
        cached = self._partition_lag.get(partition)
        return cached[0] if cached is not None else None


class DropModeController:
    def __init__(
        self,
        lag_tracker: PartitionLagTracker,
        high_watermark_offsets: int = 100_000,
        low_watermark_offsets: int = 50_000,
        drop_rate_percents: int = 50,
        increasing_duration_secs: float = 120.0,
        lag_delta_epsilon_offsets: int = 10,
    ):
        self.lag_tracker: PartitionLagTracker = lag_tracker
        self.high_watermark_offsets: int = high_watermark_offsets
        self.low_watermark_offsets: int = low_watermark_offsets
        self._drop_rate_percents: int = drop_rate_percents
        self.increasing_duration_secs: float = increasing_duration_secs
        self._lag_delta_epsilon_offsets: int = lag_delta_epsilon_offsets

        self._drop_mode_active: dict[int, bool] = {}
        self._increase_start_time: dict[int, float | None] = {}
        self._last_lag_offsets: dict[int, int | None] = {}

        self.total_dropped_records: int = 0
        self._dropped_by_partition: dict[int, int] = {}

    def update_drop_mode_for_partition(
        self, partition: int, metric_attrs: dict[str, str] | None = None
    ) -> bool:
        lag_offsets = self.lag_tracker.get_partition_lag_offsets(partition)
        self._update_increase_state(partition, lag_offsets)
        now = time.time()

        start_time = self._increase_start_time.get(partition)
        lag_increasing_long_enough = (
            start_time is not None and (now - start_time) >= self.increasing_duration_secs
        )

        if (
            not self._drop_mode_active.get(partition, False)
            and lag_offsets >= self.high_watermark_offsets
            and lag_increasing_long_enough
        ):
            self._drop_mode_active[partition] = True
            rank_logger.info(
                f"DROP MODE ACTIVATED (partition {partition}): lag {lag_offsets} offsets > "
                f"high watermark {self.high_watermark_offsets}, drop_rate_percents: {self._drop_rate_percents}"
            )
            if metric_attrs is not None and _consumer_drop_mode_active is not None:
                part_attrs = {**metric_attrs, "partition": str(partition)}
                _consumer_drop_mode_active.add(1, part_attrs)
            return True

        if (
            self._drop_mode_active.get(partition, False)
            and lag_offsets <= self.low_watermark_offsets
        ):
            self._drop_mode_active[partition] = False
            dropped = self._dropped_by_partition.get(partition, 0)
            rank_logger.info(
                f"DROP MODE DEACTIVATED (partition {partition}): lag recovered. "
                f"Total dropped this session (partition): {dropped}"
            )

        return self._drop_mode_active.get(partition, False)

    def record_dropped(
        self, partition: int, count: int, metric_attrs: dict[str, str] | None = None
    ) -> None:
        self.total_dropped_records += count
        self._dropped_by_partition[partition] = self._dropped_by_partition.get(partition, 0) + count
        if metric_attrs is not None and _consumer_dropped_records is not None:
            part_attrs = {**metric_attrs, "partition": str(partition)}
            _consumer_dropped_records.add(count, part_attrs)

    def should_drop_message(self, partition: int, offset: int) -> bool:
        if not self._drop_mode_active.get(partition, False):
            return False
        return (offset % 100) < self._drop_rate_percents

    def _update_increase_state(self, partition: int, lag_offsets: int) -> None:
        now = time.time()
        last = self._last_lag_offsets.get(partition)
        if last is None:
            self._last_lag_offsets[partition] = lag_offsets
            self._increase_start_time[partition] = None
            return

        if lag_offsets > last + self._lag_delta_epsilon_offsets:
            if self._increase_start_time.get(partition) is None:
                self._increase_start_time[partition] = now
        else:
            self._increase_start_time[partition] = None

        self._last_lag_offsets[partition] = lag_offsets


async def _fetch_and_update_partition_lag(
    consumer: AIOKafkaConsumer,
    assigned_partitions: list[TopicPartition],
    lag_tracker: PartitionLagTracker,
    drop_controller: "DropModeController | None",
    metric_attrs: dict[str, str] | None,
) -> None:
    try:
        result = await consumer.end_offsets(assigned_partitions)
        if _consumer_end_offsets_polled_count is not None:
            _consumer_end_offsets_polled_count.add(len(assigned_partitions), metric_attrs)
        end_offsets: dict[TopicPartition, int] = dict(result) if result else {}
    except Exception as e:
        rank_logger.warning(f"Failed to fetch end offsets: {e}")
        return

    if not end_offsets:
        rank_logger.warning("No end offsets found")
        return

    for partition in assigned_partitions:
        end_offset = end_offsets.get(partition)
        if end_offset is None:
            continue

        latest_offset = lag_tracker.get_latest_consumed_offset(partition.partition)
        if latest_offset is None:
            continue

        lag_tracker.update_partition_lag(partition.partition, latest_offset, end_offset)
        if drop_controller is not None:
            drop_controller.update_drop_mode_for_partition(partition.partition, metric_attrs)


class ConsumerMode(enum.Enum):
    LEGACY = "legacy"
    PER_PARTITION = "per_partition"


async def _consumer_poll_loop(
    consumer_index: int,
    assigned_partitions: list[TopicPartition],
    per_consumer_queue: asyncio.Queue[ConsumerRecord],
    bootstrap_servers: str,
    group_id: str,
    group_instance_id: str,
    auto_offset_reset: str,
    sasl_mechanism: str,
    sasl_plain_username: str,
    sasl_plain_password: str,
    reset_to_latest: bool,
    seek_to_timestamp_ms: int | None,
    seek_to_offset: dict[int, int] | None,
    stop_event: threading.Event | None,
    lag_tracker: PartitionLagTracker,
    drop_controller: DropModeController | None,
    metric_attrs: dict[str, str],
    reset_to_latest_event: asyncio.Event | None = None,
) -> None:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    consumer_metric_attrs = {**metric_attrs, "consumer_index": str(consumer_index)}

    consumer = AIOKafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        group_instance_id=group_instance_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        auto_commit_interval_ms=KAFKA_AUTO_COMMIT_INTERVAL_MS,
        security_protocol="SASL_SSL",
        sasl_kerberos_domain_name="kafka",
        sasl_kerberos_service_name="kafka",
        sasl_mechanism=sasl_mechanism,
        sasl_plain_username=sasl_plain_username,
        sasl_plain_password=sasl_plain_password,
        ssl_context=ssl_ctx,
        partition_assignment_strategy=[StickyPartitionAssignor],
        fetch_max_bytes=KAFKA_MAX_FETCH_BYTES,
        fetch_min_bytes=KAFKA_MIN_FETCH_BYTES,
        max_poll_records=KAFKA_MAX_POLL_RECORDS,
        session_timeout_ms=KAFKA_SESSION_TIMEOUT,
        heartbeat_interval_ms=KAFKA_HEARTBEAT_INTERVAL,
        max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL,
        rebalance_timeout_ms=KAFKA_REBALANCE_TIMEOUT,
    )

    async with consumer:
        consumer.assign(assigned_partitions)
        rank_logger.info(
            f"Consumer c{consumer_index} assigned "
            f"{len(assigned_partitions)} partitions: "
            f"{[tp.partition for tp in assigned_partitions]}, "
            f"group_instance_id={group_instance_id}"
        )

        if reset_to_latest:
            for tp in assigned_partitions:
                await consumer.seek_to_end(tp)
                rank_logger.info(f"Consumer c{consumer_index}: reset {tp} to latest")
        elif seek_to_timestamp_ms is not None:
            await seek_to_timestamp(consumer, assigned_partitions, seek_to_timestamp_ms)
        elif seek_to_offset is not None:
            for tp in assigned_partitions:
                if tp.partition in seek_to_offset:
                    consumer.seek(tp, seek_to_offset[tp.partition])
                    rank_logger.info(
                        f"Consumer c{consumer_index}: seeked {tp} to offset "
                        f"{seek_to_offset[tp.partition]}"
                    )

        if drop_controller is not None:
            rank_logger.info(
                f"Consumer c{consumer_index}: drop mode enabled, "
                f"high_watermark={drop_controller.high_watermark_offsets}, "
                f"low_watermark={drop_controller.low_watermark_offsets}"
            )
            for tp in assigned_partitions:
                drop_controller.record_dropped(tp.partition, 0, metric_attrs=metric_attrs)
                part_attrs: dict[str, str] = {}
                if metric_attrs is not None:
                    part_attrs.update(metric_attrs)
                part_attrs["partition"] = str(tp.partition)
                if _consumer_drop_mode_active is not None:
                    _consumer_drop_mode_active.add(0, part_attrs)
                if _consumer_end_offsets_polled_count is not None:
                    _consumer_end_offsets_polled_count.add(0, part_attrs)

        poll_count = 0
        total_fetched = 0
        _last_reset_ts = time.time()

        while not (stop_event and stop_event.is_set()):
            should_reset = False
            if reset_to_latest_event is not None and reset_to_latest_event.is_set():
                rank_logger.info(f"Consumer c{consumer_index}: live reset via HTTP endpoint")
                reset_to_latest_event.clear()
                should_reset = True
            sentinel_mtime = _check_reset_sentinel(_last_reset_ts)
            if sentinel_mtime is not None:
                rank_logger.info(
                    f"Consumer c{consumer_index}: live reset via sentinel file "
                    f"(mtime={sentinel_mtime:.3f}, last_reset={_last_reset_ts:.3f})"
                )
                _last_reset_ts = sentinel_mtime
                should_reset = True
            if should_reset:
                await _seek_all_to_end(consumer, assigned_partitions)

            try:
                poll_start = time.time()
                messages: Any = await consumer.getmany(timeout_ms=KAFKA_POLL_TIMEOUT)
                poll_latency = time.time() - poll_start

                if _consumer_poll_latency is not None:
                    _consumer_poll_latency.record(poll_latency, consumer_metric_attrs)

                if not messages:
                    if _consumer_messages_per_poll is not None:
                        _consumer_messages_per_poll.record(0, consumer_metric_attrs)
                    continue

                total_messages = sum(len(msgs) for msgs in messages.values())
                if _consumer_messages_per_poll is not None:
                    _consumer_messages_per_poll.record(total_messages, consumer_metric_attrs)
                total_fetched += 1

                lag_tracker.record_consumed_offsets(messages)
                poll_count += 1
                if poll_count % FETCH_END_OFFSETS_INTERVAL_IN_CONSUMER_ROUNDS == 0:
                    await _fetch_and_update_partition_lag(
                        consumer, assigned_partitions, lag_tracker, drop_controller, metric_attrs
                    )

                dropped_in_poll = 0
                dropped_by_partition: dict[int, int] = {}
                for _, msg_list in messages.items():
                    for msg in msg_list:
                        if drop_controller is not None and drop_controller.should_drop_message(
                            msg.partition, msg.offset
                        ):
                            dropped_in_poll += 1
                            dropped_by_partition[msg.partition] = (
                                dropped_by_partition.get(msg.partition, 0) + 1
                            )
                            continue
                        await per_consumer_queue.put(msg)

                if drop_controller is not None and dropped_in_poll > 0:
                    for partition, count in dropped_by_partition.items():
                        drop_controller.record_dropped(partition, count, metric_attrs=metric_attrs)

                if total_fetched % GC_INTERVAL == 0:
                    gc.collect()

            except TimeoutError:
                continue
            except Exception as e:
                rank_logger.error(f"Consumer c{consumer_index} poll error: {e}")
                raise

    rank_logger.info(f"Consumer c{consumer_index} stopped.")


async def _round_robin_merger(
    per_consumer_queues: list[asyncio.Queue[ConsumerRecord]],
    message_queue: asyncio.Queue[ConsumerRecord],
    example_queue: queue.Queue,
    partition_ids: list[int],
    stop_event: threading.Event | None,
) -> None:
    num_queues = len(per_consumer_queues)
    idx = 0
    last_log_time = time.time()
    consecutive_empty = 0

    while not (stop_event and stop_event.is_set()):
        now = time.time()
        if now - last_log_time >= 1.0:
            cq_sizes = {p: q.qsize() for p, q in zip(partition_ids, per_consumer_queues)}
            rank_logger.info(
                f"{message_queue.qsize()=} | {example_queue.qsize()=} | "
                f"consumer_queues(partition:qsize)={cq_sizes}"
            )
            last_log_time = now

        q = per_consumer_queues[idx]
        try:
            msg = await asyncio.wait_for(q.get(), timeout=0.05)
            try:
                await message_queue.put(msg)
                idx = (idx + 1) % num_queues
            finally:
                q.task_done()
            consecutive_empty = 0
        except TimeoutError:
            consecutive_empty += 1
            if consecutive_empty >= num_queues:
                await asyncio.sleep(0.01)
                consecutive_empty = 0


async def _consume_multi_consumer(
    topic: str,
    group_id: str,
    example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]],
    batch_size: int,
    shard_index: int,
    num_shards: int,
    bootstrap_servers: str,
    post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
    sasl_mechanism: str,
    sasl_plain_username: str,
    sasl_plain_password: str,
    reset_to_latest: bool,
    seek_to_timestamp_ms: int | None,
    seek_to_offset: dict[int, int] | None,
    stop_event: threading.Event | None,
    sampling_factor: int,
    auto_offset_reset: str,
    lag_tracker: PartitionLagTracker,
    drop_controller: DropModeController | None,
    consumer_mode: ConsumerMode,
    metric_attrs: dict[str, str],
    reset_to_latest_event: threading.Event | None = None,
    negative_downsample_rate: int = 1,
    negative_downsample_positive_action_indices: str = DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES,
) -> None:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    discovery_consumer = AIOKafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        security_protocol="SASL_SSL",
        sasl_kerberos_domain_name="kafka",
        sasl_kerberos_service_name="kafka",
        sasl_mechanism=sasl_mechanism,
        sasl_plain_username=sasl_plain_username,
        sasl_plain_password=sasl_plain_password,
        ssl_context=ssl_ctx,
        request_timeout_ms=30000,
    )
    await discovery_consumer.start()
    try:
        all_partitions = discovery_consumer.partitions_for_topic(topic)
        if not all_partitions:
            all_partitions = await force_kafka_metadata_update(discovery_consumer, topic)
        if not all_partitions:
            all_partitions = await force_kafka_metadata_update(discovery_consumer, topic)
        if not all_partitions:
            available = await discovery_consumer.topics()
            raise ValueError(
                f"No partitions found for topic {topic}, available topics: {available}"
            )
    finally:
        try:
            await discovery_consumer.stop()
        except asyncio.CancelledError:
            pass

    total_partitions = len(all_partitions)
    partition_ids = _range_partitions(total_partitions, shard_index, num_shards)
    shard_partitions = [TopicPartition(topic, i) for i in partition_ids]
    if not partition_ids:
        rank_logger.warning(
            f"Shard {shard_index} received 0 partitions (total={total_partitions}, "
            f"num_shards={num_shards}).  This shard will be idle."
        )
    rank_logger.info(
        f"Discovered {total_partitions} partitions for {topic}, "
        f"range-based assignment: shard {shard_index}/{num_shards} got "
        f"{len(shard_partitions)} partitions"
        + (f" ({partition_ids[0]}..{partition_ids[-1]})" if partition_ids else "")
    )

    consumer_partition_groups = [[tp] for tp in shard_partitions]
    num_consumers = len(consumer_partition_groups)
    for ci, parts in enumerate(consumer_partition_groups):
        rank_logger.info(f"  consumer c{ci} (per_partition): partition {parts[0].partition}")

    message_queue: asyncio.Queue[ConsumerRecord] = asyncio.Queue(maxsize=MESSAGE_QUEUE_SIZE)

    deserialization_task = asyncio.create_task(
        deserialize_messages(
            message_queue,
            example_queue,
            batch_size,
            post_process_fn,
            sampling_factor,
            stop_event=stop_event,
            negative_downsample_rate=negative_downsample_rate,
            negative_downsample_positive_action_indices=negative_downsample_positive_action_indices,
        )
    )

    per_consumer_queue_size = max(1, MESSAGE_QUEUE_SIZE // num_consumers)
    rank_logger.info(
        f"Per-consumer queue size: {per_consumer_queue_size} "
        f"(MESSAGE_QUEUE_SIZE={MESSAGE_QUEUE_SIZE} / {num_consumers} consumers)"
    )
    per_consumer_queues: list[asyncio.Queue[ConsumerRecord]] = [
        asyncio.Queue(maxsize=per_consumer_queue_size) for _ in range(num_consumers)
    ]

    per_consumer_reset_events: list[asyncio.Event] = [asyncio.Event() for _ in range(num_consumers)]

    consumer_tasks: list[asyncio.Task[None]] = []
    for ci, parts in enumerate(consumer_partition_groups):
        group_instance_id = f"shard_{shard_index}_of_{num_shards}_c{ci}"
        task = asyncio.create_task(
            _consumer_poll_loop(
                consumer_index=ci,
                assigned_partitions=parts,
                per_consumer_queue=per_consumer_queues[ci],
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                group_instance_id=group_instance_id,
                auto_offset_reset=auto_offset_reset,
                sasl_mechanism=sasl_mechanism,
                sasl_plain_username=sasl_plain_username,
                sasl_plain_password=sasl_plain_password,
                reset_to_latest=reset_to_latest,
                seek_to_timestamp_ms=seek_to_timestamp_ms,
                seek_to_offset=seek_to_offset,
                stop_event=stop_event,
                lag_tracker=lag_tracker,
                drop_controller=drop_controller,
                metric_attrs=metric_attrs,
                reset_to_latest_event=per_consumer_reset_events[ci],
            )
        )
        consumer_tasks.append(task)

    partition_ids = [tp.partition for tp in shard_partitions[:num_consumers]]
    merger_task = asyncio.create_task(
        _round_robin_merger(
            per_consumer_queues,
            message_queue,
            example_queue,
            partition_ids,
            stop_event,
        )
    )

    rank_logger.info(
        f"Per-partition consumer mode: {num_consumers} consumers (1 per partition), "
        f"shard_index={shard_index}, num_shards={num_shards}, "
        f"total_partitions={total_partitions}, shard_partitions={len(shard_partitions)}"
    )

    async def _stop_event_waiter():
        while not (stop_event and stop_event.is_set()):
            await asyncio.sleep(0.5)

    async def _reset_event_fanout():
        while not (stop_event and stop_event.is_set()):
            if reset_to_latest_event is not None and reset_to_latest_event.is_set():
                reset_to_latest_event.clear()
                for ev in per_consumer_reset_events:
                    ev.set()
                rank_logger.info(f"Reset signal fanned out to {num_consumers} consumers")
            await asyncio.sleep(0.5)

    stop_waiter_task = asyncio.create_task(_stop_event_waiter())
    reset_fanout_task = asyncio.create_task(_reset_event_fanout())

    all_tasks = consumer_tasks + [merger_task, deserialization_task]
    auxiliary_tasks = [stop_waiter_task, reset_fanout_task]
    waitable: set[asyncio.Task[None]] = set(all_tasks) | {stop_waiter_task}
    try:
        while waitable:
            done, waitable = await asyncio.wait(waitable, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is stop_waiter_task:
                    rank_logger.info("Stop event received, shutting down consumers.")
                    waitable.clear()
                    break
                if not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        raise exc
    finally:
        for task in all_tasks + auxiliary_tasks:
            task.cancel()
        await asyncio.gather(*all_tasks, *auxiliary_tasks, return_exceptions=True)

    rank_logger.info("All consumers stopped.")


def _check_reset_sentinel(last_reset_ts: float) -> float | None:
    try:
        mtime = os.path.getmtime(KAFKA_RESET_SENTINEL_PATH)
        if mtime > last_reset_ts:
            return mtime
    except (FileNotFoundError, OSError):
        pass
    return None


async def _seek_all_to_end(
    consumer: AIOKafkaConsumer,
    assigned_partitions: list[TopicPartition],
) -> None:
    for tp in assigned_partitions:
        await consumer.seek_to_end(tp)
    rank_logger.info(
        "Live reset: seeked %d partitions to latest offset",
        len(assigned_partitions),
    )


async def consume_messages(
    topic: str,
    group_id: str,
    example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]],
    batch_size: int,
    shard_index: int,
    num_shards: int,
    bootstrap_servers: str,
    post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
    sasl_mechanism: str,
    sasl_plain_username: str,
    sasl_plain_password: str,
    reset_to_latest: bool = False,
    seek_to_timestamp_ms: int | None = None,
    seek_to_offset: dict[int, int] | None = None,
    stop_event: threading.Event | None = None,
    sampling_factor: int = 1,
    auto_offset_reset: str = "latest",
    training_name: str = "",
    lag_tracker: PartitionLagTracker | None = None,
    drop_controller: DropModeController | None = None,
    per_partition_poll: bool = False,
    consumer_mode: ConsumerMode = ConsumerMode.LEGACY,
    reset_to_latest_event: threading.Event | None = None,
    negative_downsample_rate: int = 1,
    negative_downsample_positive_action_indices: str = DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES,
):
    metric_attrs = {
        "training_name": training_name,
        "topic": topic,
        "ranker_index": str(shard_index),
    }
    _init_consumer_metrics(lag_tracker=lag_tracker, metric_attrs=metric_attrs)
    _consumer_start_times[(training_name, topic, str(shard_index))] = int(time.time() * 1000)

    if consumer_mode is not ConsumerMode.LEGACY:
        await _consume_multi_consumer(
            topic=topic,
            group_id=group_id,
            example_queue=example_queue,
            batch_size=batch_size,
            shard_index=shard_index,
            num_shards=num_shards,
            bootstrap_servers=bootstrap_servers,
            post_process_fn=post_process_fn,
            sasl_mechanism=sasl_mechanism,
            sasl_plain_username=sasl_plain_username,
            sasl_plain_password=sasl_plain_password,
            reset_to_latest=reset_to_latest,
            seek_to_timestamp_ms=seek_to_timestamp_ms,
            seek_to_offset=seek_to_offset,
            stop_event=stop_event,
            sampling_factor=sampling_factor,
            auto_offset_reset=auto_offset_reset,
            lag_tracker=lag_tracker or PartitionLagTracker(),
            drop_controller=drop_controller,
            consumer_mode=consumer_mode,
            metric_attrs=metric_attrs,
            reset_to_latest_event=reset_to_latest_event,
            negative_downsample_rate=negative_downsample_rate,
            negative_downsample_positive_action_indices=negative_downsample_positive_action_indices,
        )
        return

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    group_instance_id = f"shard_{shard_index}_of_{num_shards}"
    total_fetched_batch_count = 0

    consumer = AIOKafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        group_instance_id=group_instance_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        auto_commit_interval_ms=KAFKA_AUTO_COMMIT_INTERVAL_MS,
        security_protocol="SASL_SSL",
        sasl_kerberos_domain_name="kafka",
        sasl_kerberos_service_name="kafka",
        sasl_mechanism=sasl_mechanism,
        sasl_plain_username=sasl_plain_username,
        sasl_plain_password=sasl_plain_password,
        ssl_context=ssl_context,
        partition_assignment_strategy=[StickyPartitionAssignor],
        fetch_max_bytes=KAFKA_MAX_FETCH_BYTES,
        fetch_min_bytes=KAFKA_MIN_FETCH_BYTES,
        max_poll_records=KAFKA_MAX_POLL_RECORDS,
        session_timeout_ms=KAFKA_SESSION_TIMEOUT,
        heartbeat_interval_ms=KAFKA_HEARTBEAT_INTERVAL,
        max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL,
        rebalance_timeout_ms=KAFKA_REBALANCE_TIMEOUT,
    )

    message_queue: asyncio.Queue[ConsumerRecord] = asyncio.Queue(maxsize=MESSAGE_QUEUE_SIZE)

    deserialization_task = asyncio.create_task(
        deserialize_messages(
            message_queue,
            example_queue,
            batch_size,
            post_process_fn,
            sampling_factor,
            stop_event=stop_event,
            negative_downsample_rate=negative_downsample_rate,
            negative_downsample_positive_action_indices=negative_downsample_positive_action_indices,
        )
    )

    async with consumer:
        rank_logger.info(
            f"Consumer started, topic: {topic}, group_id: {group_id}, group_instance_id: {group_instance_id}, listening for messages..."
        )

        poll_count = 0

        assigned_partitions: list[TopicPartition] = await handle_topic_offset(
            consumer=consumer,
            topic=topic,
            shard_index=shard_index,
            num_shards=num_shards,
            reset_to_latest=reset_to_latest,
            seek_to_timestamp_ms=seek_to_timestamp_ms,
            seek_to_offset=seek_to_offset,
        )
        assigned_partitions_array_index = 0
        assigned_partitions_array_len = len(assigned_partitions)

        if drop_controller is not None:
            rank_logger.info(
                f"Drop mode enabled, high watermark: {drop_controller.high_watermark_offsets} offsets, "
                f"low watermark: {drop_controller.low_watermark_offsets} offsets"
            )
            for partition in assigned_partitions:
                drop_controller.record_dropped(partition.partition, 0, metric_attrs=metric_attrs)
                part_attrs: dict[str, str] = {}
                if metric_attrs is not None:
                    part_attrs.update(metric_attrs)
                part_attrs["partition"] = str(partition.partition)
                if _consumer_drop_mode_active is not None:
                    _consumer_drop_mode_active.add(0, part_attrs)
                if _consumer_end_offsets_polled_count is not None:
                    _consumer_end_offsets_polled_count.add(0, part_attrs)
        else:
            rank_logger.info("Drop mode disabled (lag tracking still active)")

        effective_lag_tracker = lag_tracker or PartitionLagTracker()

        _last_reset_ts = time.time()

        while not (stop_event and stop_event.is_set()):
            should_reset = False

            if reset_to_latest_event is not None and reset_to_latest_event.is_set():
                rank_logger.info("Live reset triggered via HTTP endpoint")
                reset_to_latest_event.clear()
                should_reset = True

            sentinel_mtime = _check_reset_sentinel(_last_reset_ts)
            if sentinel_mtime is not None:
                rank_logger.info(
                    "Live reset triggered via sentinel file (mtime=%.3f, last_reset=%.3f)",
                    sentinel_mtime,
                    _last_reset_ts,
                )
                _last_reset_ts = sentinel_mtime
                should_reset = True

            if should_reset:
                await _seek_all_to_end(consumer, assigned_partitions)

            rank_logger.info(f"{message_queue.qsize()=} | {example_queue.qsize()=}")

            try:
                poll_start = time.time()
                if per_partition_poll:
                    messages: Any = await consumer.getmany(
                        assigned_partitions[assigned_partitions_array_index],
                        timeout_ms=KAFKA_PER_PARTITION_POLL_TIMEOUT,
                    )
                    assigned_partitions_array_index = (
                        assigned_partitions_array_index + 1
                    ) % assigned_partitions_array_len
                else:
                    messages = await consumer.getmany(
                        timeout_ms=KAFKA_POLL_TIMEOUT,
                    )
                poll_latency = time.time() - poll_start

                if _consumer_poll_latency is not None:
                    _consumer_poll_latency.record(poll_latency, metric_attrs)
            except TimeoutError:
                rank_logger.info("Polling timeout")
                continue
            except Exception as e:
                rank_logger.error(f"Error while polling: {e!s}")
                raise
            else:
                if not messages:
                    if _consumer_messages_per_poll is not None:
                        _consumer_messages_per_poll.record(0, metric_attrs)
                    rank_logger.info("Nothing received")
                    continue

                total_messages = sum(len(msg_list) for msg_list in messages.values())
                if _consumer_messages_per_poll is not None:
                    _consumer_messages_per_poll.record(total_messages, metric_attrs)

                total_fetched_batch_count += 1

            effective_lag_tracker.record_consumed_offsets(messages)
            poll_count += 1
            if poll_count % FETCH_END_OFFSETS_INTERVAL_IN_CONSUMER_ROUNDS == 0:
                await _fetch_and_update_partition_lag(
                    consumer,
                    assigned_partitions,
                    effective_lag_tracker,
                    drop_controller,
                    metric_attrs,
                )

            dropped_in_poll = 0

            dropped_by_partition: dict[int, int] = {}
            for _, msg_list in messages.items():
                for msg in msg_list:
                    if drop_controller is not None and drop_controller.should_drop_message(
                        msg.partition, msg.offset
                    ):
                        dropped_in_poll += 1
                        dropped_by_partition[msg.partition] = (
                            dropped_by_partition.get(msg.partition, 0) + 1
                        )
                        continue
                    await message_queue.put(msg)

            if drop_controller is not None and dropped_in_poll > 0:
                for partition, count in dropped_by_partition.items():
                    drop_controller.record_dropped(partition, count, metric_attrs=metric_attrs)
                rank_logger.debug(
                    f"Dropped {dropped_in_poll}/{total_messages} messages due to lag; "
                    f"partitions with drop: {sorted(dropped_by_partition.keys())}, "
                    f"high_watermark={drop_controller.high_watermark_offsets} offsets, "
                    f"low_watermark={drop_controller.low_watermark_offsets} offsets"
                )

            if total_fetched_batch_count % GC_INTERVAL == 0:
                gc.collect()

    deserialization_task.cancel()
    try:
        await deserialization_task
    except asyncio.CancelledError:
        pass

    rank_logger.info("Consumer stopped.")
