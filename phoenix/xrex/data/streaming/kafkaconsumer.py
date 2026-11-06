# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import asyncio
import concurrent.futures
import contextlib
import dataclasses
import enum
import gc
import io
import logging
import os
import queue
import random
import ssl
import sys
import tempfile
import threading
import time
import traceback
import typing
from functools import partial
from numbers import Integral
from typing import Any, Callable, Mapping

import pyarrow as pa
import pyarrow.ipc as ipc
from aiokafka import AIOKafkaConsumer, ConsumerRecord
from aiokafka.coordinator.assignors.sticky.sticky_assignor import (
    StickyPartitionAssignor,
)
from aiokafka.errors import KafkaConnectionError
from aiokafka.structs import TopicPartition

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch

DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES = ""


def parse_positive_action_indices(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part)


@dataclasses.dataclass(frozen=True)
class KafkaBootstrapOffsets:
    topic_name: str
    offsets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.topic_name) is not str or not self.topic_name:
            raise ValueError("Kafka bootstrap requires a topic identity")
        if type(self.offsets) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 for pair in self.offsets
        ):
            raise ValueError("Kafka bootstrap offsets must be immutable partition/offset pairs")
        if len({pair[0] for pair in self.offsets}) != len(self.offsets):
            raise ValueError("Duplicate Kafka bootstrap partition")
        normalized = validate_offset_map(dict(self.offsets))
        object.__setattr__(self, "offsets", tuple(sorted(normalized.items())))


KafkaQueueItem = tuple[RecsysFeaturesBatch, dict[int, int]] | Exception | KafkaBootstrapOffsets


@dataclasses.dataclass(frozen=True)
class KafkaAuth:
    mode: str
    ca_path: str | None = None
    cert_path: str | None = None
    key_path: str | None = None
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    verify_broker: bool = False

    @classmethod
    def mtls(
        cls,
        ca_path: str | None,
        cert_path: str,
        key_path: str,
        verify_broker: bool = False,
    ) -> "KafkaAuth":
        return cls(
            mode="mtls",
            ca_path=ca_path,
            cert_path=cert_path,
            key_path=key_path,
            verify_broker=verify_broker,
        )

    @classmethod
    def sasl(cls, mechanism: str, username: str, password: str) -> "KafkaAuth":
        return cls(
            mode="sasl",
            sasl_mechanism=mechanism,
            sasl_username=username,
            sasl_password=password,
        )

    def ssl_context(self) -> ssl.SSLContext:
        if self.mode == "mtls":
            if not (self.cert_path and self.key_path):
                raise ValueError("mTLS KafkaAuth is missing cert/key paths")
            ctx = (
                ssl.create_default_context(cafile=self.ca_path)
                if (self.verify_broker and self.ca_path)
                else ssl.create_default_context()
            )
            ctx.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path)
            ctx.check_hostname = False
            if not self.verify_broker:
                ctx.verify_mode = ssl.CERT_NONE
            return ctx
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def consumer_kwargs(self) -> dict[str, Any]:
        ctx = self.ssl_context()
        if self.mode == "mtls":
            return {"security_protocol": "SSL", "ssl_context": ctx}
        return {
            "security_protocol": "SASL_SSL",
            "sasl_kerberos_domain_name": "kafka",
            "sasl_kerberos_service_name": "kafka",
            "sasl_mechanism": self.sasl_mechanism,
            "sasl_plain_username": self.sasl_username,
            "sasl_plain_password": self.sasl_password,
            "ssl_context": ctx,
        }


def _coerce_auth(
    auth: KafkaAuth | None,
    sasl_mechanism: str,
    sasl_plain_username: str,
    sasl_plain_password: str,
) -> KafkaAuth:
    if auth is not None:
        return auth
    return KafkaAuth.sasl(sasl_mechanism, sasl_plain_username, sasl_plain_password)


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
_consumer_width_mismatch_drops = None
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
    global _consumer_dropped_records, _consumer_deserialize_errors, _consumer_width_mismatch_drops
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

        _consumer_width_mismatch_drops = meter.create_counter(
            name="recsys_kafka_consumer_width_mismatch_dropped_total",
            description="Kafka messages dropped because a nested list width differs from the model's",
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
KAFKA_BATCH_BUILD_WORKERS = 2
KAFKA_DECODE_CHUNK = 64
FETCH_END_OFFSETS_INTERVAL_IN_CONSUMER_ROUNDS = 8

MAX_COMMIT_FAILURES = 3
COMMIT_FAILURE_SLEEP_TIME = 15.0
GC_INTERVAL = 8192


KAFKA_SHUTDOWN_TIMEOUT_SECONDS = 5.0
KAFKA_START_ATTEMPTS = 8
KAFKA_START_RETRY_BASE_SECONDS = 1.0
KAFKA_START_RETRY_MAX_SECONDS = 20.0


def _maybe_periodic_gc() -> None:
    if gc.isenabled():
        gc.collect()


async def _wait_for_stop(stop_event: threading.Event | None) -> None:
    if stop_event is None:
        await asyncio.Future()
    while not stop_event.is_set():
        await asyncio.sleep(0.05)


async def _stop_consumer(consumer: AIOKafkaConsumer, *, discovery: bool = False) -> None:
    primary = sys.exception()
    try:
        await asyncio.wait_for(consumer.stop(), KAFKA_SHUTDOWN_TIMEOUT_SECONDS)
    except TimeoutError:
        rank_logger.error(
            "Kafka consumer cleanup exceeded %.1f seconds", KAFKA_SHUTDOWN_TIMEOUT_SECONDS
        )
    except asyncio.CancelledError as cleanup:
        if primary is not None:
            primary.add_note(f"Kafka consumer cleanup cancelled: {cleanup!r}")
        elif not discovery:
            raise
    except Exception as cleanup:
        if primary is None:
            raise
        primary.add_note(f"Kafka consumer cleanup failed: {cleanup!r}")


async def _start_consumer_with_retry(
    factory: Callable[[], AIOKafkaConsumer],
    stop_event: threading.Event | None = None,
) -> AIOKafkaConsumer:
    for attempt in range(1, KAFKA_START_ATTEMPTS + 1):
        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError
        consumer = factory()
        try:
            await consumer.start()
            return consumer
        except KafkaConnectionError as error:
            await _stop_consumer(consumer)
            if attempt == KAFKA_START_ATTEMPTS:
                raise
            delay = min(
                KAFKA_START_RETRY_MAX_SECONDS,
                KAFKA_START_RETRY_BASE_SECONDS * 2 ** (attempt - 1),
            ) * random.uniform(0.5, 1.0)
            rank_logger.warning(
                "Kafka consumer start failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                KAFKA_START_ATTEMPTS,
                error,
                delay,
            )
            stopping = asyncio.create_task(_wait_for_stop(stop_event))
            try:
                done, _ = await asyncio.wait({stopping}, timeout=delay)
                if done:
                    raise asyncio.CancelledError
            finally:
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
        except BaseException:
            await _stop_consumer(consumer)
            raise
    raise RuntimeError("Kafka startup attempts exhausted")


@contextlib.asynccontextmanager
async def _started(
    factory: Callable[[], AIOKafkaConsumer], stop_event: threading.Event | None = None
):
    consumer = await _start_consumer_with_retry(factory, stop_event)
    try:
        yield consumer
    finally:
        await _stop_consumer(consumer)


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


_expected_inner_list_widths: dict[str, int] = {}
_MAX_WIDTH_MISMATCH_LOGS = 20
_width_mismatch_logs = 0


def set_expected_inner_list_widths(widths: typing.Mapping[str, int]) -> None:
    global _expected_inner_list_widths
    _expected_inner_list_widths = dict(widths)


def inner_list_width_mismatch(
    rb: pa.RecordBatch, expected: typing.Mapping[str, int]
) -> tuple[str, int, int] | None:
    for name, want in expected.items():
        index = rb.schema.get_field_index(name)
        if index < 0:
            continue
        field_type = rb.schema.field(index).type
        if not (
            pa.types.is_list(field_type)
            or pa.types.is_large_list(field_type)
            or pa.types.is_fixed_size_list(field_type)
        ):
            continue
        element = field_type.value_type
        if pa.types.is_fixed_size_list(element) and element.list_size != want:
            return name, want, element.list_size
    return None


def _drop_width_mismatched(rb: pa.RecordBatch, partition: int, offset: int) -> bool:
    global _width_mismatch_logs
    if not _expected_inner_list_widths:
        return False
    mismatch = inner_list_width_mismatch(rb, _expected_inner_list_widths)
    if mismatch is None:
        return False
    if _consumer_width_mismatch_drops is not None:
        _consumer_width_mismatch_drops.add(1, {"partition": str(partition)})
    if _width_mismatch_logs < _MAX_WIDTH_MISMATCH_LOGS:
        _width_mismatch_logs += 1
        column, want, got = mismatch
        rank_logger.warning(
            f"[width-guard {_width_mismatch_logs}/{_MAX_WIDTH_MISMATCH_LOGS}] dropping "
            f"partition={partition} offset={offset}: {column} element width {got}, model expects {want}"
        )
    return True


def _decode_chunk(
    msgs: list[ConsumerRecord],
    negative_downsample_rate: int,
    positive_action_indices: tuple[int, ...],
) -> tuple[list[pa.RecordBatch | None], Exception | None]:
    decoded: list[pa.RecordBatch | None] = []
    for msg in msgs:
        try:
            decoded.append(
                _decode_training_message(msg, negative_downsample_rate, positive_action_indices)
            )
        except Exception as err:
            return decoded, err
    return decoded, None


def _decode_training_message(
    msg: ConsumerRecord, negative_downsample_rate: int, positive_action_indices: tuple[int, ...]
) -> pa.RecordBatch | None:
    value = typing.cast(bytes, msg.value)
    partition, offset = msg.partition, msg.offset
    try:
        user_action_sequence = deserialize_arrow_ipc(value)
    except (pa.ArrowException, OSError, ValueError, TypeError) as e:
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
        return None

    if _drop_width_mismatched(user_action_sequence, partition, offset):
        return None

    if negative_downsample_rate > 1:
        if not positive_action_indices:
            raise ValueError(
                "negative_downsample_positive_action_indices must be non-empty "
                "when negative_downsample_rate > 1"
            )
        sample_weight = 1.0
        if not _has_positive_candidate(user_action_sequence, positive_action_indices):
            if hash((partition, offset)) % negative_downsample_rate != 0:
                return None
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

    return user_action_sequence


async def _deserialize_messages(
    batch: list[tuple[pa.RecordBatch, int, int]],
    batch_size: int,
    msg_queue: asyncio.Queue[ConsumerRecord],
    example_queue: queue.Queue[KafkaQueueItem],
    post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
    count: int = 0,
    sampling_factor: int = 1,
    negative_downsample_rate: int = 1,
    positive_action_indices: tuple[int, ...] = (),
    builder: "_OrderedBatchBuilder | None" = None,
) -> tuple[list[tuple[pa.RecordBatch, int, int]], int]:
    messages = [await msg_queue.get()]
    while len(messages) < KAFKA_DECODE_CHUNK and not msg_queue.empty():
        messages.append(msg_queue.get_nowait())

    to_decode: list[ConsumerRecord] = []
    for msg in messages:
        count += 1
        if count % sampling_factor != 0:
            msg_queue.task_done()
        else:
            to_decode.append(msg)

    if to_decode:
        decoded, error = await asyncio.to_thread(
            _decode_chunk, to_decode, negative_downsample_rate, positive_action_indices
        )
    else:
        decoded, error = [], None

    for msg, user_action_sequence in zip(to_decode, decoded):
        if user_action_sequence is not None:
            batch.append((user_action_sequence, msg.partition, msg.offset))
        msg_queue.task_done()
        if len(batch) < batch_size:
            continue

        records: list[pa.RecordBatch] = []
        offsets: dict[int, int] = {}
        for record, tp, ts in batch:
            offsets[tp] = max(offsets.get(tp, ts), ts)
            records.append(record)
        batch = []

        if builder is not None:
            await builder.submit(records, offsets)
        else:
            while example_queue.full():
                await asyncio.sleep(0.1)
            example_queue.put_nowait((post_process_fn(records), offsets))
    if error is not None:
        raise error
    return batch, count


class _OrderedBatchBuilder:
    def __init__(
        self,
        post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
        example_queue: queue.Queue[KafkaQueueItem],
        workers: int = KAFKA_BATCH_BUILD_WORKERS,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._fn = post_process_fn
        self._example_queue = example_queue
        self._executor = concurrent.futures.ThreadPoolExecutor(
            workers, thread_name_prefix="kafka-batch-build"
        )
        self._slots = asyncio.Semaphore(workers + 1)
        self._pending: asyncio.Queue[tuple[asyncio.Future, dict[int, int]]] = asyncio.Queue(
            maxsize=workers + 1
        )
        self._builds: set[concurrent.futures.Future] = set()
        self._stopping = asyncio.create_task(_wait_for_stop(stop_event))
        self._emitter = asyncio.create_task(self._emit_in_order())
        self._closed = False

    def raise_if_failed(self) -> None:
        if self._emitter.done():
            self._emitter.result()
            raise RuntimeError("Kafka batch emitter stopped")

    async def submit(self, records: list[pa.RecordBatch], offsets: dict[int, int]) -> None:
        if self._closed:
            raise RuntimeError("Kafka batch builder is closed")
        slot = asyncio.create_task(self._slots.acquire())
        enqueued = False
        try:
            await asyncio.wait(
                {slot, self._emitter, self._stopping}, return_when=asyncio.FIRST_COMPLETED
            )
            self.raise_if_failed()
            if self._stopping.done():
                raise asyncio.CancelledError
            if self._closed:
                raise RuntimeError("Kafka batch builder is closed")
            await slot
            loop = asyncio.get_running_loop()
            build = self._executor.submit(self._fn, records)
            self._builds.add(build)

            def finished(future):
                try:
                    loop.call_soon_threadsafe(self._builds.discard, future)
                except RuntimeError:
                    pass

            build.add_done_callback(finished)
            self._pending.put_nowait((asyncio.wrap_future(build), dict(offsets)))
            enqueued = True
        finally:
            if not slot.done():
                slot.cancel()
            await asyncio.gather(slot, return_exceptions=True)
            if not enqueued and not slot.cancelled() and slot.exception() is None:
                self._slots.release()

    async def _emit_in_order(self) -> None:
        while True:
            future, offsets = await self._pending.get()
            try:
                result = await future
                while not self._stopping.done():
                    try:
                        self._example_queue.put_nowait((result, offsets))
                        break
                    except queue.Full:
                        await asyncio.sleep(0.05)
            finally:
                self._pending.task_done()
                self._slots.release()

    async def close(self, *, drain: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = asyncio.get_running_loop().time() + KAFKA_SHUTDOWN_TIMEOUT_SECONDS
        primary = sys.exception()
        try:
            if drain:
                joined = asyncio.create_task(self._pending.join())
                try:
                    done, _ = await asyncio.wait(
                        {joined, self._emitter, self._stopping},
                        timeout=KAFKA_SHUTDOWN_TIMEOUT_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    self.raise_if_failed()
                    if not done:
                        rank_logger.error(
                            "Kafka batch drain exceeded %.1f seconds",
                            KAFKA_SHUTDOWN_TIMEOUT_SECONDS,
                        )
                finally:
                    joined.cancel()
                    await asyncio.gather(joined, return_exceptions=True)
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"Kafka batch drain failed: {cleanup!r}")
        finally:
            self._emitter.cancel()
            self._stopping.cancel()
            await asyncio.gather(self._emitter, self._stopping, return_exceptions=True)
            while not self._pending.empty():
                future, _ = self._pending.get_nowait()
                future.cancel()
                if future.done() and not future.cancelled():
                    future.exception()
                self._pending.task_done()
                self._slots.release()
            self._executor.shutdown(wait=False, cancel_futures=True)
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            builds = [asyncio.wrap_future(build) for build in self._builds]
            if builds:
                done, pending = await asyncio.wait(builds, timeout=remaining)
                for future in done:
                    if not future.cancelled():
                        future.exception()
                if pending:
                    rank_logger.error(
                        "Kafka batch shutdown left %d running builds after %.1f seconds",
                        len(pending),
                        KAFKA_SHUTDOWN_TIMEOUT_SECONDS,
                    )
                for future in pending:
                    future.cancel()


async def deserialize_messages(
    msg_queue: asyncio.Queue[ConsumerRecord],
    example_queue: queue.Queue[KafkaQueueItem],
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
    builder = _OrderedBatchBuilder(post_process_fn, example_queue, stop_event=stop_event)

    async def read_messages() -> None:
        nonlocal batch, count
        while not (stop_event and stop_event.is_set()):
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
                builder=builder,
            )

    async def wait_for_stop() -> None:
        while not (stop_event and stop_event.is_set()):
            await asyncio.sleep(0.05)

    read_task = asyncio.create_task(read_messages())
    stop_task = asyncio.create_task(wait_for_stop())
    try:
        done, _ = await asyncio.wait(
            {read_task, stop_task, builder._emitter}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            task.result()
    finally:
        primary = sys.exception()
        read_task.cancel()
        stop_task.cancel()
        await asyncio.gather(read_task, stop_task, return_exceptions=True)
        closing = asyncio.create_task(
            builder.close(drain=primary is None and not (stop_event and stop_event.is_set()))
        )
        try:
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    if primary is None:
                        primary = sys.exception()
            closing.result()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"Kafka batch cleanup failed: {cleanup!r}")
        if primary is not None and sys.exception() is None:
            raise primary


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
    auth: KafkaAuth | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    auth = _coerce_auth(auth, sasl_mechanism, sasl_plain_username, sasl_plain_password)

    consumer_factory = partial(
        AIOKafkaConsumer,
        bootstrap_servers=bootstrap_servers,
        **auth.consumer_kwargs(),
        request_timeout_ms=30000,
    )
    consumer = await _start_consumer_with_retry(consumer_factory, stop_event)
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
        await _stop_discovery_consumer(consumer)


async def _stop_discovery_consumer(consumer: AIOKafkaConsumer) -> None:
    await _stop_consumer(consumer, discovery=True)


def _range_partitions(total_partitions: int, shard_index: int, num_shards: int) -> list[int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if total_partitions < 0:
        raise ValueError(f"total_partitions must be >= 0, got {total_partitions}")
    start = shard_index * total_partitions // num_shards
    end = (shard_index + 1) * total_partitions // num_shards
    return list(range(start, end))


@dataclasses.dataclass(frozen=True)
class KafkaSourceIdentity:
    bootstrap_servers: str
    topic_name: str
    num_partitions: int

    def __post_init__(self) -> None:
        if not self.bootstrap_servers or not self.topic_name:
            raise ValueError("Kafka source requires broker and topic identity")
        if type(self.num_partitions) is not int or self.num_partitions <= 0:
            raise ValueError("Kafka source num_partitions must be a positive integer")


def validate_offset_map[K, V](
    offsets: Mapping[K, V],
    *,
    num_partitions: int | None = None,
    required_partitions: set[int] | None = None,
    offset_ranges: Mapping[int, tuple[int, int]] | None = None,
    max_offset: int = 2**31 - 1,
) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for partition, offset in offsets.items():
        if isinstance(partition, bool) or not isinstance(partition, Integral):
            raise ValueError(f"Invalid Kafka partition {partition!r}")
        partition_id = int(partition)
        if partition_id < 0:
            raise ValueError(f"Invalid Kafka partition {partition!r}")
        if num_partitions is not None and partition_id >= num_partitions:
            raise ValueError(f"Kafka partition {partition} outside source partition range")
        if isinstance(offset, bool) or not isinstance(offset, Integral):
            raise ValueError(
                f"Kafka offset {offset!r} for partition {partition} must be an integer"
            )
        offset_value = int(offset)
        if not 0 <= offset_value <= max_offset:
            raise ValueError(
                f"Kafka offset {offset!r} for partition {partition} outside range [0, {max_offset}]"
            )
        normalized[partition_id] = offset_value
    if required_partitions is not None and set(normalized) != required_partitions:
        raise ValueError("Kafka offset keyset differs from required partitions")
    if offset_ranges is not None:
        if set(offset_ranges) != set(normalized):
            raise ValueError("Kafka retained-range keyset differs from offsets")
        for partition, (begin, end) in offset_ranges.items():
            if type(begin) is not int or type(end) is not int or not 0 <= begin < end:
                raise ValueError(f"Invalid retained range for Kafka partition {partition}")
            if not begin <= normalized[partition] < end:
                raise ValueError(f"Kafka offset for partition {partition} outside retained range")
    return normalized


def validate_seek_options(
    seek_to_timestamp_ms: int | None, seek_to_offset: Mapping[int, int] | None
) -> None:
    if seek_to_timestamp_ms is not None and seek_to_offset is not None:
        raise ValueError("seek_to_timestamp_ms and seek_to_offset cannot both be set")
    if seek_to_offset is not None:
        validate_offset_map(seek_to_offset, max_offset=2**63 - 1)


async def _native_bootstrap_positions(
    consumer: AIOKafkaConsumer,
    assigned_partitions: list[TopicPartition],
    seek_to_timestamp_ms: int | None,
    seek_to_offset: dict[int, int] | None,
) -> dict[int, int]:
    if len({tp.partition for tp in assigned_partitions}) != len(assigned_partitions):
        raise ValueError("Duplicate native Kafka partition assignment")
    if len({tp.topic for tp in assigned_partitions}) > 1:
        raise ValueError("Mixed native Kafka topic assignment")
    required = {tp.partition for tp in assigned_partitions}
    if not required:
        return {}
    if seek_to_timestamp_ms is not None:
        resolved = await consumer.offsets_for_times(
            {tp: seek_to_timestamp_ms for tp in assigned_partitions}
        )
        if set(resolved) != set(assigned_partitions) or any(
            resolved[tp] is None for tp in assigned_partitions
        ):
            raise ValueError("Missing Kafka timestamp bootstrap positions")
        offsets = {tp.partition: resolved[tp].offset for tp in assigned_partitions}
    elif seek_to_offset is not None:
        validate_offset_map(seek_to_offset)
        if not required.issubset(seek_to_offset):
            raise ValueError("Missing Kafka native restart positions")
        offsets = {partition: seek_to_offset[partition] for partition in required}
    else:
        raise ValueError("Native Kafka bootstrap requires explicit timestamp or offset positions")
    offsets = validate_offset_map(offsets, required_partitions=required)
    for tp in assigned_partitions:
        consumer.seek(tp, offsets[tp.partition])
    return offsets


async def _publish_bootstrap_offsets(
    output: queue.Queue[KafkaQueueItem],
    event: KafkaBootstrapOffsets,
    stop_event: threading.Event | None,
) -> None:
    while not (stop_event and stop_event.is_set()):
        try:
            output.put_nowait(event)
            return
        except queue.Full:
            await asyncio.sleep(0.05)
    raise asyncio.CancelledError


class _KafkaBootstrapFence:
    topic: str
    partitions: set[int]
    output: queue.Queue[KafkaQueueItem]
    stop_event: threading.Event | None
    ready: asyncio.Event

    def __init__(
        self,
        topic: str,
        partitions: set[int],
        output: queue.Queue[KafkaQueueItem],
        stop_event: threading.Event | None,
    ) -> None:
        self.topic = topic
        self.partitions = partitions
        self.output = output
        self.stop_event = stop_event
        self.positions: dict[int, int] = {}
        self.ready = asyncio.Event()

    async def report(self, assigned: list[TopicPartition], offsets: dict[int, int]) -> None:
        if any(tp.topic != self.topic for tp in assigned):
            raise ValueError("Kafka bootstrap topic mismatch")
        required = {tp.partition for tp in assigned}
        if len(required) != len(assigned):
            raise ValueError("Duplicate Kafka bootstrap assignment")
        offsets = validate_offset_map(offsets, required_partitions=required)
        if not required.issubset(self.partitions) or required & self.positions.keys():
            raise ValueError("Duplicate or unexpected Kafka bootstrap report")
        self.positions.update(offsets)
        if self.positions.keys() == self.partitions:
            await _publish_bootstrap_offsets(
                self.output,
                KafkaBootstrapOffsets(self.topic, tuple(self.positions.items())),
                self.stop_event,
            )
            self.ready.set()
        await self.ready.wait()


def _subsample_partitions(partition_ids: list[int], partition_sampling_factor: int) -> list[int]:
    if partition_sampling_factor <= 1:
        return partition_ids
    if partition_ids and len(partition_ids) < partition_sampling_factor:
        raise ValueError(
            f"partition_sampling_factor={partition_sampling_factor} exceeds this shard's "
            f"{len(partition_ids)} partitions; some ranks would receive none and hang."
        )
    return partition_ids[::partition_sampling_factor]


async def handle_topic_offset(
    consumer: AIOKafkaConsumer,
    topic: str,
    shard_index: int,
    num_shards: int,
    reset_to_latest: bool,
    seek_to_timestamp_ms: int | None = None,
    seek_to_offset: dict[int, int] | None = None,
    bootstrap_output: queue.Queue[KafkaQueueItem] | None = None,
    stop_event: threading.Event | None = None,
    partition_sampling_factor: int = 1,
) -> list[TopicPartition]:
    validate_seek_options(seek_to_timestamp_ms, seek_to_offset)
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
    if bootstrap_output is not None and partitions != set(range(total_partitions)):
        raise ValueError("Native Kafka partition metadata must be a complete range")

    if bootstrap_output is not None and partition_sampling_factor != 1:
        raise ValueError("Native Kafka requires unsampled partitions")
    if bootstrap_output is not None and num_shards > total_partitions:
        raise ValueError("Native Kafka requires at least one partition per shard")
    partition_ids = _subsample_partitions(
        _range_partitions(total_partitions, shard_index, num_shards), partition_sampling_factor
    )
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
    if bootstrap_output is not None:
        offsets = await _native_bootstrap_positions(
            consumer, assigned_partitions, seek_to_timestamp_ms, seek_to_offset
        )
        await _publish_bootstrap_offsets(
            bootstrap_output, KafkaBootstrapOffsets(topic, tuple(offsets.items())), stop_event
        )
    elif reset_to_latest:
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
    validate_offset_map(offsets, max_offset=2**63 - 1)
    missing = {tp.partition for tp in assigned_partitions} - set(offsets)
    if missing:
        raise ValueError(f"No Kafka offset found for assigned partitions {sorted(missing)}")
    for tp in assigned_partitions:
        offset = offsets[tp.partition]
        rank_logger.info(f"Seeking to offset {offset} for {tp}")
        consumer.seek(tp, offset)


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
    auth: KafkaAuth | None = None,
    bootstrap_fence: _KafkaBootstrapFence | None = None,
) -> None:
    validate_seek_options(seek_to_timestamp_ms, seek_to_offset)
    auth = _coerce_auth(auth, sasl_mechanism, sasl_plain_username, sasl_plain_password)

    consumer_metric_attrs = {**metric_attrs, "consumer_index": str(consumer_index)}

    consumer_factory = partial(
        AIOKafkaConsumer,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        group_instance_id=group_instance_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        auto_commit_interval_ms=KAFKA_AUTO_COMMIT_INTERVAL_MS,
        **auth.consumer_kwargs(),
        partition_assignment_strategy=[StickyPartitionAssignor],
        fetch_max_bytes=KAFKA_MAX_FETCH_BYTES,
        fetch_min_bytes=KAFKA_MIN_FETCH_BYTES,
        max_poll_records=KAFKA_MAX_POLL_RECORDS,
        session_timeout_ms=KAFKA_SESSION_TIMEOUT,
        heartbeat_interval_ms=KAFKA_HEARTBEAT_INTERVAL,
        max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL,
        rebalance_timeout_ms=KAFKA_REBALANCE_TIMEOUT,
    )

    async with _started(consumer_factory, stop_event) as consumer:
        consumer.assign(assigned_partitions)
        rank_logger.info(
            f"Consumer c{consumer_index} assigned "
            f"{len(assigned_partitions)} partitions: "
            f"{[tp.partition for tp in assigned_partitions]}, "
            f"group_instance_id={group_instance_id}"
        )

        if bootstrap_fence is not None:
            offsets = await _native_bootstrap_positions(
                consumer, assigned_partitions, seek_to_timestamp_ms, seek_to_offset
            )
            await bootstrap_fence.report(assigned_partitions, offsets)
        elif reset_to_latest:
            for tp in assigned_partitions:
                await consumer.seek_to_end(tp)
                rank_logger.info(f"Consumer c{consumer_index}: reset {tp} to latest")
        elif seek_to_timestamp_ms is not None:
            await seek_to_timestamp(consumer, assigned_partitions, seek_to_timestamp_ms)
        elif seek_to_offset is not None:
            await seek_consumer_to_offset(consumer, assigned_partitions, seek_to_offset)

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
                await _apply_external_reset(
                    consumer, assigned_partitions, native=bootstrap_fence is not None
                )

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
                        consumer,
                        assigned_partitions,
                        lag_tracker,
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
                        await per_consumer_queue.put(msg)

                if drop_controller is not None and dropped_in_poll > 0:
                    for partition, count in dropped_by_partition.items():
                        drop_controller.record_dropped(partition, count, metric_attrs=metric_attrs)

                if total_fetched % GC_INTERVAL == 0:
                    _maybe_periodic_gc()

            except asyncio.TimeoutError:
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
        except asyncio.TimeoutError:
            consecutive_empty += 1
            if consecutive_empty >= num_queues:
                await asyncio.sleep(0.01)
                consecutive_empty = 0


async def _consume_multi_consumer(
    topic: str,
    group_id: str,
    example_queue: queue.Queue[KafkaQueueItem],
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
    partition_sampling_factor: int = 1,
    auth: KafkaAuth | None = None,
    native_bootstrap: bool = False,
) -> None:
    auth = _coerce_auth(auth, sasl_mechanism, sasl_plain_username, sasl_plain_password)

    discovery_factory = partial(
        AIOKafkaConsumer,
        bootstrap_servers=bootstrap_servers,
        **auth.consumer_kwargs(),
        request_timeout_ms=30000,
    )
    discovery_consumer = await _start_consumer_with_retry(discovery_factory, stop_event)
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
        await _stop_discovery_consumer(discovery_consumer)

    total_partitions = len(all_partitions)
    if native_bootstrap and all_partitions != set(range(total_partitions)):
        raise ValueError("Native Kafka partition metadata must be a complete range")
    if native_bootstrap and partition_sampling_factor != 1:
        raise ValueError("Native Kafka requires unsampled partitions")
    if native_bootstrap and num_shards > total_partitions:
        raise ValueError("Native Kafka requires at least one partition per shard")
    partition_ids = _subsample_partitions(
        _range_partitions(total_partitions, shard_index, num_shards), partition_sampling_factor
    )
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

    if not shard_partitions:
        if native_bootstrap:
            await _publish_bootstrap_offsets(
                example_queue, KafkaBootstrapOffsets(topic, ()), stop_event
            )
        return
    bootstrap_fence = (
        _KafkaBootstrapFence(topic, set(partition_ids), example_queue, stop_event)
        if native_bootstrap
        else None
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
                auth=auth,
                reset_to_latest=reset_to_latest,
                seek_to_timestamp_ms=seek_to_timestamp_ms,
                seek_to_offset=seek_to_offset,
                stop_event=stop_event,
                lag_tracker=lag_tracker,
                drop_controller=drop_controller,
                metric_attrs=metric_attrs,
                reset_to_latest_event=per_consumer_reset_events[ci],
                bootstrap_fence=bootstrap_fence,
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


async def _apply_external_reset(
    consumer: AIOKafkaConsumer, assigned_partitions: list[TopicPartition], *, native: bool
) -> None:
    if native:
        rank_logger.error("Ignoring external Kafka reset to preserve the native bootstrap frontier")
        return
    await _seek_all_to_end(consumer, assigned_partitions)


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
    example_queue: queue.Queue[KafkaQueueItem],
    batch_size: int,
    shard_index: int,
    num_shards: int,
    bootstrap_servers: str,
    post_process_fn: Callable[[list[pa.RecordBatch]], RecsysFeaturesBatch],
    sasl_mechanism: str,
    sasl_plain_username: str,
    sasl_plain_password: str,
    auth: KafkaAuth | None = None,
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
    fresh_training_bootstrap: bool = False,
    native_checkpoint_restart: bool = False,
    partition_sampling_factor: int = 1,
):
    validate_seek_options(seek_to_timestamp_ms, seek_to_offset)
    if type(fresh_training_bootstrap) is not bool or type(native_checkpoint_restart) is not bool:
        raise ValueError("Native Kafka bootstrap flags must be exact Booleans")
    native_bootstrap = fresh_training_bootstrap or native_checkpoint_restart
    if native_bootstrap:
        if partition_sampling_factor != 1:
            raise ValueError("Native Kafka requires unsampled partitions")
        if (
            type(shard_index) is not int
            or type(num_shards) is not int
            or not 0 <= shard_index < num_shards
        ):
            raise ValueError("Invalid native Kafka shard assignment")
        if (fresh_training_bootstrap and native_checkpoint_restart) or reset_to_latest:
            raise ValueError("Conflicting native Kafka bootstrap modes")
        if auto_offset_reset != "none":
            raise ValueError("Native Kafka bootstrap requires disabled offset fallback")
        if fresh_training_bootstrap and (
            type(seek_to_timestamp_ms) is not int or seek_to_timestamp_ms <= 0
        ):
            raise ValueError("Fresh Kafka bootstrap requires a positive timestamp")
        if native_checkpoint_restart and (
            seek_to_offset is None or seek_to_timestamp_ms is not None
        ):
            raise ValueError("Native Kafka restart requires only restored offsets")
    auth = _coerce_auth(auth, sasl_mechanism, sasl_plain_username, sasl_plain_password)
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
            auth=auth,
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
            native_bootstrap=native_bootstrap,
            partition_sampling_factor=partition_sampling_factor,
        )
        return

    group_instance_id = f"shard_{shard_index}_of_{num_shards}"
    total_fetched_batch_count = 0

    consumer_factory = partial(
        AIOKafkaConsumer,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        group_instance_id=group_instance_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        auto_commit_interval_ms=KAFKA_AUTO_COMMIT_INTERVAL_MS,
        **auth.consumer_kwargs(),
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

    async def poll_messages() -> None:
        nonlocal total_fetched_batch_count
        async with _started(consumer_factory, stop_event) as consumer:
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
                bootstrap_output=example_queue if native_bootstrap else None,
                stop_event=stop_event,
                partition_sampling_factor=partition_sampling_factor,
            )
            if not assigned_partitions:
                return
            assigned_partitions_array_index = 0
            assigned_partitions_array_len = len(assigned_partitions)

            if drop_controller is not None:
                rank_logger.info(
                    f"Drop mode enabled, high watermark: {drop_controller.high_watermark_offsets} offsets, "
                    f"low watermark: {drop_controller.low_watermark_offsets} offsets"
                )
                for partition in assigned_partitions:
                    drop_controller.record_dropped(
                        partition.partition, 0, metric_attrs=metric_attrs
                    )
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
                    await _apply_external_reset(
                        consumer, assigned_partitions, native=native_bootstrap
                    )

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
                except asyncio.TimeoutError:
                    rank_logger.info("Polling timeout")
                    continue
                except Exception as e:
                    rank_logger.error(f"Error while polling: {str(e)}")
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
                    _maybe_periodic_gc()

    async def wait_for_stop() -> None:
        while not (stop_event and stop_event.is_set()):
            await asyncio.sleep(0.5)

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
    poll_task = asyncio.create_task(poll_messages())
    stop_task = asyncio.create_task(wait_for_stop())
    tasks = [deserialization_task, poll_task, stop_task]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if not task.cancelled():
                error = task.exception()
                if error is not None:
                    raise error
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    rank_logger.info("Consumer stopped.")
