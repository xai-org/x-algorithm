# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import asyncio
import dataclasses
import errno
import logging
import os
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from functools import partial
from threading import Event
from typing import Any, cast

import numpy as np
import pyarrow as pa
from xai_configlib import configclass

_OTEL_AVAILABLE = False
try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except ImportError:
    metrics = None

from xrex import settings
from xrex.data.parquet_recsys import DataPosition, PhoenixDataset, load_global_ids_from_parquet_file
from xrex.data.recsys.feature_config import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
)
from xrex.data.recsys.recsys_batch import (
    EMBEDDING_CONFIG,
    CandidateNegativeMode,
    EmbeddingType,
    RecsysFeaturesBatch,
    from_record_batch,
)
from xrex.data.streaming.kafkaconsumer import (
    DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES,
    ConsumerMode,
    DropModeController,
    PartitionLagTracker,
    consume_messages,
    discover_partition_count,
)
from xrex.models.recsys_embedding import HashKeys, HashTable
from xrex.utils import cluster

log_level_str = os.environ.get("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
rank_logger = logging.getLogger("rank")


consumer: threading.Thread | None = None
stop_event = threading.Event()
metrics_server_started = False

_kafka_reset_event: threading.Event | None = None

_meter = None
_kafka_batches_processed = None
_kafka_messages_processed = None
_kafka_batch_processing_time = None
_kafka_queue_size = None
_kafka_empty_polls = None
_kafka_partition_delivered_offset = None

_training_metrics_initialized = False
_training_histograms: dict[str, Any] = {}

_queue_size_values: dict[tuple[str, str, str], int] = {}

_partition_delivered_offsets: dict[tuple[str, str, str, int], int] = {}

_training_metric_values: dict[tuple[str, str], float] = {}

_training_cluster_name: str = "unknown"

_TRAINER_KEY_MAP: dict[str, str] = {}

_KNOWN_OTEL_KEYS: set[str] = set()


def _start_reset_http_server(base_port: int, num_ports: int) -> int | None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _ResetHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/reset-kafka":
                if _kafka_reset_event is not None:
                    _kafka_reset_event.set()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Kafka reset signal sent\n")
                    rank_logger.info(
                        "Kafka reset signal received via HTTP POST /reset-kafka on port %d",
                        self.server.server_address[1],
                    )
                else:
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Consumer not started yet\n")
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/reset-kafka":
                is_pending = _kafka_reset_event is not None and _kafka_reset_event.is_set()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"reset_pending={is_pending}\n".encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    for port in range(base_port, base_port + num_ports):
        try:
            server = HTTPServer(("0.0.0.0", port), _ResetHandler)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                continue
            rank_logger.warning("Failed to start Kafka reset HTTP server on port %d: %s", port, e)
            return None
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        rank_logger.info("Kafka reset HTTP server listening on port %d (POST /reset-kafka)", port)
        return port
    rank_logger.warning(
        "Kafka reset HTTP server: no free port in %d-%d; live /reset-kafka disabled for this worker",
        base_port,
        base_port + num_ports - 1,
    )
    return None


def _init_otel_metrics(
    otel_endpoint: str,
    export_interval_ms: int = 30_000,
    worker_rank: int = 0,
    training_name: str = "recsys-kafka-loader",
) -> bool:
    global _meter, _kafka_batches_processed, _kafka_messages_processed
    global _kafka_batch_processing_time, _kafka_queue_size, _kafka_empty_polls
    global _kafka_partition_delivered_offset
    global _training_metrics_initialized, _training_histograms, _training_metric_values
    global _training_cluster_name, _TRAINER_KEY_MAP, _KNOWN_OTEL_KEYS
    global metrics_server_started

    if metrics_server_started:
        return True

    if not _OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry packages not installed. Metrics will be disabled. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-grpc"
        )
        return False

    assert metrics is not None

    resource = Resource.create(
        {
            "service.name": training_name,
            "service.instance_id": f"worker-{worker_rank}",
            "worker_rank": str(worker_rank),
            "k8s.pod.name": cluster.get_pod_name() or "unknown",
            "k8s.cluster.name": cluster.get_cluster() or "unknown",
            "k8s.node.name": cluster.get_node_name() or "unknown",
        }
    )

    import socket as _socket

    _host, _, _port_str = otel_endpoint.rpartition(":")
    _port = int(_port_str) if _port_str.isdigit() else 4317
    try:
        _addr = _socket.getaddrinfo(_host, _port, proto=_socket.IPPROTO_TCP)
        logger.info("OTel collector DNS resolved: %s -> %s", otel_endpoint, _addr[0][4])
    except Exception as dns_err:
        logger.warning(
            "OTel collector DNS resolution FAILED for %s: %s  -- metrics will not be exported",
            otel_endpoint,
            dns_err,
        )

    _otel_logger = logging.getLogger("opentelemetry")
    if not _otel_logger.handlers:
        _otel_logger.setLevel(logging.WARNING)
        _otel_logger.addHandler(logging.StreamHandler())

    exporter = OTLPMetricExporter(endpoint=otel_endpoint, insecure=True)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=export_interval_ms)

    _training_histogram_views = [
        View(
            instrument_name="recsys_training_step_time_seconds",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[
                    0.1,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                    2.1,
                    2.2,
                    2.3,
                    2.4,
                    2.5,
                    2.6,
                    2.7,
                    2.8,
                    2.9,
                    3.0,
                    3.1,
                    3.2,
                    3.3,
                    3.4,
                    3.5,
                    3.6,
                    3.7,
                    3.8,
                    3.9,
                    4.0,
                    4.2,
                    4.4,
                    4.6,
                    4.8,
                    5.0,
                    6.0,
                    7.5,
                    10.0,
                    15.0,
                    30.0,
                    60.0,
                ],
            ),
        ),
        View(
            instrument_name="recsys_training_loss",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[
                    0.01,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    4.0,
                    5.0,
                    7.5,
                    10.0,
                    20.0,
                ],
            ),
        ),
        View(
            instrument_name="recsys_training_origin_loss",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[
                    0.01,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    4.0,
                    5.0,
                    7.5,
                    10.0,
                    20.0,
                ],
            ),
        ),
        View(
            instrument_name="recsys_training_grad_norm",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[
                    0.01,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    1.0,
                    2.0,
                    5.0,
                    10.0,
                    25.0,
                    50.0,
                    100.0,
                    500.0,
                    1000.0,
                ],
            ),
        ),
        View(
            instrument_name="recsys_training_mfu",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[
                    0.01,
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.25,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],
            ),
        ),
        View(
            instrument_name="recsys_training_tflops",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000],
            ),
        ),
        View(
            instrument_name="recsys_training_valid_step",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[0.5],
            ),
        ),
    ]

    provider = MeterProvider(
        resource=resource,
        metric_readers=[reader],
        views=_training_histogram_views,
    )
    metrics.set_meter_provider(provider)

    _meter = metrics.get_meter("recsys_kafka_loader", "1.0.0")

    _kafka_batches_processed = _meter.create_counter(
        name="recsys_kafka_batches_processed_total",
        description="Total number of Kafka batches processed",
        unit="1",
    )

    _kafka_messages_processed = _meter.create_counter(
        name="recsys_kafka_messages_processed_total",
        description="Total number of Kafka messages processed",
        unit="1",
    )

    _kafka_batch_processing_time = _meter.create_histogram(
        name="recsys_kafka_batch_processing_seconds",
        description="Time spent processing Kafka batches",
        unit="1",
    )

    _kafka_empty_polls = _meter.create_counter(
        name="recsys_kafka_empty_polls_total",
        description="Total number of empty queue polls",
        unit="1",
    )

    def _get_queue_size_callback(_options):
        assert metrics is not None
        for (name, topic, ranker_idx), size in _queue_size_values.items():
            yield metrics.Observation(
                size, {"training_name": name, "topic": topic, "ranker_index": ranker_idx}
            )

    _kafka_queue_size = _meter.create_observable_gauge(
        name="recsys_kafka_queue_size",
        description="Current size of the example queue",
        unit="1",
        callbacks=[_get_queue_size_callback],
    )

    def _get_partition_delivered_offset_callback(_options):
        assert metrics is not None
        for (name, topic, ranker_idx, partition), offset in _partition_delivered_offsets.items():
            yield metrics.Observation(
                offset,
                {
                    "training_name": name,
                    "topic": topic,
                    "ranker_index": ranker_idx,
                    "partition": str(partition),
                },
            )

    _kafka_partition_delivered_offset = _meter.create_observable_gauge(
        name="recsys_kafka_partition_delivered_offset",
        description=(
            "Per-partition Kafka offset of the most recent message delivered to the "
            "training loop (read from offset_ptr after example_queue.get). "
            "max-min across partitions = the actual training data fairness skew."
        ),
        unit="1",
        callbacks=[_get_partition_delivered_offset_callback],
    )

    global _TRAINER_KEY_MAP
    _TRAINER_KEY_MAP = {
        "origin-loss": "origin_loss",
        "global_grad_norm": "grad_norm",
        "eta (hr)": "eta_hours",
    }

    _TRAINING_HISTOGRAM_DEFS = [
        ("recsys_training_loss", "loss", "Training loss per step"),
        ("recsys_training_origin_loss", "origin_loss", "Origin loss (pre-subset) per step"),
        ("recsys_training_grad_norm", "grad_norm", "Global gradient norm per step"),
        (
            "recsys_training_step_time_seconds",
            "step_time",
            "Wall-clock time per training step in seconds",
        ),
        ("recsys_training_mfu", "mfu", "Model FLOPS utilization"),
        ("recsys_training_tflops", "tflops", "Achieved TFLOPS"),
        (
            "recsys_training_valid_step",
            "valid_step",
            "Valid step (1) or skipped (0). Histogram captures valid/invalid ratio per interval",
        ),
    ]

    _TRAINING_GAUGE_DEFS = [
        ("recsys_training_learning_rate", "learning_rate", "Current learning rate"),
        ("recsys_training_examples_per_batch", "examples_per_batch", "Examples per batch"),
        ("recsys_training_peak_mem_gb", "peak_mem", "Peak GPU memory usage in GB"),
        ("recsys_training_weight_decay", "weight_decay", "Weight decay coefficient"),
        ("recsys_training_finished_percent", "finished_percent", "Training progress percentage"),
        ("recsys_training_eta_hours", "eta_hours", "Estimated time remaining in hours"),
    ]

    _TRAINING_COUNTER_DEFS = [
        ("recsys_training_step", "step", "Current training step number"),
        ("recsys_training_soft_step", "soft_step", "Soft step (valid steps only)"),
        ("recsys_training_elapsed_samples", "elapsed_samples", "Total samples processed"),
    ]

    for hist_name, metric_key, description in _TRAINING_HISTOGRAM_DEFS:
        _training_histograms[metric_key] = _meter.create_histogram(
            name=hist_name,
            description=description,
            unit="1",
        )

    _training_metric_values = {}
    _training_cluster_name = cluster.get_cluster() or "unknown"

    def _make_training_callback(metric_key: str):
        def _callback(_options):
            assert metrics is not None
            for (name, key), val in _training_metric_values.items():
                if key == metric_key:
                    yield metrics.Observation(
                        val,
                        {
                            "training_name": name,
                            "k8s_cluster_name": _training_cluster_name,
                        },
                    )

        return _callback

    for gauge_name, metric_key, description in _TRAINING_GAUGE_DEFS:
        _meter.create_observable_gauge(
            name=gauge_name,
            description=description,
            unit="1",
            callbacks=[_make_training_callback(metric_key)],
        )

    for counter_name, metric_key, description in _TRAINING_COUNTER_DEFS:
        _meter.create_observable_counter(
            name=counter_name,
            description=description,
            unit="1",
            callbacks=[_make_training_callback(metric_key)],
        )

    _KNOWN_OTEL_KEYS = set()
    for _, metric_key, _ in (
        _TRAINING_HISTOGRAM_DEFS + _TRAINING_GAUGE_DEFS + _TRAINING_COUNTER_DEFS
    ):
        _KNOWN_OTEL_KEYS.add(metric_key)

    _training_metrics_initialized = True

    metrics_server_started = True
    logger.info(
        "Initialized OTel push metrics (endpoint=%s, interval=%dms, worker_rank=%d)",
        otel_endpoint,
        export_interval_ms,
        worker_rank,
    )
    return True


def report_training_metrics(
    training_name: str,
    metrics_dict: Mapping[str, float | int | None],
) -> None:
    if not metrics_server_started:
        return

    attrs = {"training_name": training_name, "k8s_cluster_name": _training_cluster_name}
    for raw_key, value in metrics_dict.items():
        if value is None:
            continue
        otel_key = _TRAINER_KEY_MAP.get(raw_key, raw_key)
        if otel_key not in _KNOWN_OTEL_KEYS:
            continue
        fval = float(value)
        hist = _training_histograms.get(otel_key)
        if hist is not None:
            hist.record(fval, attrs)
        else:
            _training_metric_values[(training_name, otel_key)] = fval


def _cast_column(arr: pa.Array, target_type: pa.DataType) -> pa.Array:
    if arr.type == target_type:
        return arr

    if pa.types.is_fixed_size_list(arr.type) and pa.types.is_fixed_size_list(target_type):
        if arr.type.list_size != target_type.list_size:
            raise pa.ArrowInvalid(
                f"FixedSizeList size mismatch: {arr.type.list_size} vs {target_type.list_size}"
            )
        fsl_arr = cast(pa.FixedSizeListArray, arr)
        cast_inner = _cast_column(fsl_arr.values, target_type.value_type)
        if cast_inner is fsl_arr.values:
            return arr
        return pa.FixedSizeListArray.from_arrays(cast_inner, arr.type.list_size)

    if pa.types.is_list(arr.type) and pa.types.is_list(target_type):
        list_arr = cast(pa.ListArray, arr)
        cast_inner = _cast_column(list_arr.values, target_type.value_type)
        if cast_inner is list_arr.values:
            return arr
        return pa.ListArray.from_arrays(list_arr.offsets, cast_inner)

    if pa.types.is_fixed_size_list(arr.type) and pa.types.is_list(target_type):
        fsl_arr2 = cast(pa.FixedSizeListArray, arr)
        cast_inner = _cast_column(fsl_arr2.values, target_type.value_type)
        if cast_inner is fsl_arr2.values:
            return arr
        return pa.FixedSizeListArray.from_arrays(cast_inner, arr.type.list_size)

    if pa.types.is_list(arr.type) and pa.types.is_fixed_size_list(target_type):
        list_arr2 = cast(pa.ListArray, arr)
        cast_inner = _cast_column(list_arr2.values, target_type.value_type)
        if cast_inner is list_arr2.values:
            return arr
        return pa.ListArray.from_arrays(list_arr2.offsets, cast_inner)

    return arr.cast(target_type)


def _unify_record_batch_schemas(
    batches: list[pa.RecordBatch],
    required_columns: list[str],
    optional_columns: list[str],
) -> list[pa.RecordBatch]:
    if not batches:
        return batches

    all_batch_names: set[str] = set()
    for rb in batches:
        all_batch_names.update(rb.schema.names)

    col_names: list[str] = list(required_columns)
    for opt_name in optional_columns:
        if opt_name in all_batch_names:
            col_names.append(opt_name)

    canonical_types: dict[str, pa.DataType] = {}
    for name in col_names:
        for rb in batches:
            if name in rb.schema.names:
                canonical_types[name] = rb.schema.field(name).type
                break

    missing_everywhere = [name for name in col_names if name not in canonical_types]
    if missing_everywhere:
        logger.warning(f"Columns not found in any batch (skipped): {missing_everywhere}")

    col_names = [name for name in col_names if name in canonical_types]

    target_schema = pa.schema([pa.field(name, canonical_types[name]) for name in col_names])

    result: list[pa.RecordBatch] = []
    for rb in batches:
        if rb.schema.equals(target_schema):
            result.append(rb)
            continue
        num_rows = rb.num_rows
        new_columns: list[pa.Array] = []
        for name in col_names:
            target_type = canonical_types[name]
            if name in rb.schema.names:
                col = rb.column(name)
                if col.type != target_type:
                    try:
                        col = _cast_column(col, target_type)
                    except Exception:
                        logger.error(
                            f"Failed to cast column '{name}' from {col.type} to {target_type}",
                            exc_info=True,
                        )
                        raise
                new_columns.append(col)
            else:
                logger.warning(f"Column {name} not found in batch")
                new_columns.append(pa.nulls(num_rows, type=target_type))
        result.append(pa.RecordBatch.from_arrays(new_columns, names=col_names))

    return result


def _has_port(address: str) -> bool:
    for part in address.split(","):
        host_port = part.strip().rsplit(":", 1)
        if len(host_port) == 2 and host_port[1].isdigit():
            return True
    return False


def ensure_sasl_password_env(cluster_name: str) -> None:
    pass


def resolve_internal_bootstrap(bootstrap_servers: str) -> str:
    if _has_port(bootstrap_servers.strip()):
        return bootstrap_servers
    return settings.KAFKA_BOOTSTRAP_SERVERS or bootstrap_servers


def cluster_sasl_username(cluster_name: str) -> str:
    del cluster_name
    return settings.KAFKA_SASL_USERNAME


def _resolve_bootstrap_servers(bootstrap_servers: str) -> str:
    bs = bootstrap_servers.strip()
    if _has_port(bs):
        rank_logger.info(f"Using raw bootstrap servers: {bs}")
        return bs
    raise ValueError(
        f"bootstrap_servers={bs!r} has no port; provide a raw broker address "
        "(host:port[,host:port...]) — cluster-name resolution is internal-only."
    )


def _resolve_sasl_password(bootstrap_servers: str) -> str:
    password = os.environ.get("SASL_PLAIN_PASSWORD")
    if password:
        return password
    raise RuntimeError(
        "Kafka SASL password could not be resolved: set the SASL_PLAIN_PASSWORD "
        "environment variable before launching training."
    )


@configclass
class PhoenixKafkaDataset(PhoenixDataset):
    name: str = ""
    _logged_schema: bool = False

    bootstrap_servers: str = ""
    topic_name: str = ""
    group_id: str = ""
    pad_token: int = 0
    max_queue_size: int = 100
    is_eval: bool = False
    is_alive: bool = True
    sasl_plain_username: str = ""
    sasl_mechanism: str = "SCRAM-SHA-512"

    reset_to_latest: bool = False
    seek_to_timestamp_ms: int | None = None
    auto_offset_reset: str = "latest"
    seek_to_offset: dict[int, int] | None = None
    offset_ptr: dict[int, int] | None = None
    num_kafka_partitions: int | None = None
    sampling_factor: int = 1
    negative_downsample_rate: int = 1
    negative_downsample_positive_action_indices: str = (
        DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES
    )
    per_partition_poll: bool = True
    consumer_mode: ConsumerMode = ConsumerMode.PER_PARTITION

    i_know_what_i_am_doing: bool = False

    enable_drop_on_high_qps: bool = False
    partition_high_watermark_offset_lag: int = 55_000
    partition_low_watermark_offset_lag: int = 10_000
    drop_max_drop_rate: int = 30
    drop_increasing_duration_secs: float = 2.0

    global_post_ids: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_author_ids: np.ndarray | None = dataclasses.field(init=False, default=None)
    global_post_sids: np.ndarray | None = dataclasses.field(init=False, default=None)

    global_ids_reload_interval_seconds: float = 0

    export_metrics: bool = True
    otel_endpoint: str = ""
    metrics_export_interval_ms: int = 30_000

    num_local_workers: int = 8

    reset_base_port: int = 10098

    multimodal_embedding_type: EmbeddingType | None = None
    max_candidates_for_embeddings: int = 64

    compute_post_unexplored_label: bool = False

    def kafka_to_training_batch(
        self, batch: list[pa.RecordBatch], batch_size: int, shard_index: int = 0
    ) -> RecsysFeaturesBatch:
        del batch_size
        start_time = time.time()

        if not self._logged_schema and batch:
            first_rb = batch[0]
            available_cols = first_rb.schema.names
            expected_cols = REQUIRED_COLUMNS
            missing_cols = [col for col in expected_cols if col not in available_cols]
            if missing_cols:
                rank_logger.error(
                    f"Schema mismatch! Missing required columns: {missing_cols}. "
                    f"Available columns: {available_cols}"
                )
            else:
                rank_logger.info(f"Schema validated. Available columns: {available_cols}")
            self._logged_schema = True

        required_columns: list[str] = list(REQUIRED_COLUMNS)
        if self.multimodal_embedding_type is not None:
            embedding_col_name, _ = EMBEDDING_CONFIG[self.multimodal_embedding_type]
            if batch and embedding_col_name in batch[0].schema.names:
                required_columns.append(embedding_col_name)

        unified_batches = _unify_record_batch_schemas(batch, required_columns, OPTIONAL_COLUMNS)

        t = time.time()
        table = pa.Table.from_batches(unified_batches)
        elapsed = time.time() - t
        rank_logger.debug(
            f"Timer: Converted {len(batch)} messages to Arrow table in {elapsed:.3f} seconds"
        )

        t = time.time()
        table = table.combine_chunks()
        elapsed = time.time() - t
        rank_logger.debug(f"Timer: Combined Arrow table chunks in {elapsed:.3f} seconds")

        if table.num_rows == 0:
            raise ValueError("Combined table has no rows")

        t = time.time()
        example = from_record_batch(
            table.to_batches()[0],
            self.history_seq_len,
            self.candidate_seq_len,
            self.num_negatives_per_example,
            self.output_vocab_size,
            self.num_continuous_actions,
            self.hash_table,
            self.include_candidate_post_ids,
            self.num_global_negatives_per_example,
            self.global_post_ids,
            self.global_author_ids,
            self.max_candidates_for_embeddings,
            self.multimodal_embedding_type,
            search_query_embedding_dim=self.search_query_embedding_dim,
            candidate_negative_filter=self.candidate_negative_filter,
            candidate_negative_mode=(
                self.candidate_negative_mode
                if self.candidate_negative_mode is not None
                else CandidateNegativeMode.FILTER
            ),
            global_post_sids=self.global_post_sids,
            sid_num_levels=self.sid_num_levels if self.use_post_sid else 0,
            compute_post_unexplored_label=self.compute_post_unexplored_label,
        )

        elapsed = time.time() - t
        logger.debug(
            f"Timer: Prepared batch in {elapsed:.3f} seconds, batch size: {len(example['user_hashes'])}"
        )

        total_elapsed = time.time() - start_time
        if self.export_metrics:
            shard_attrs = {
                "training_name": self.name,
                "topic": self.topic_name,
                "ranker_index": str(shard_index),
            }
            if _kafka_batch_processing_time:
                _kafka_batch_processing_time.record(total_elapsed, shard_attrs)
            if _kafka_messages_processed:
                _kafka_messages_processed.add(len(batch), shard_attrs)

        return example

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
            server_hosts,
            keep_and_pad_partial_batch,
            skip_rows,
            server_port,
            prefetch_factor,
            resume_position,
        )
        global consumer, _kafka_reset_event
        if consumer is not None:
            raise ValueError("Called make() twice, process is already running.")

        self._reset_event: Event = threading.Event()
        _kafka_reset_event = self._reset_event

        if self.export_metrics:
            try:
                _init_otel_metrics(
                    otel_endpoint=self.otel_endpoint,
                    export_interval_ms=self.metrics_export_interval_ms,
                    worker_rank=shard_index,
                    training_name=self.name,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OTel push metrics: {e}")

        if not self.is_eval:
            _start_reset_http_server(self.reset_base_port, self.num_local_workers)

            if not self.i_know_what_i_am_doing:
                assert self.seek_to_timestamp_ms is None, (
                    "Seeking to offset via timestamp cannot be used in training."
                )
                assert not self.reset_to_latest, "Reset to latest cannot be used in training."
            else:
                logger.warning(
                    "If you are trying to recover from an outage in production, and are tempted to set reset_to_latest=True, then remember: you must harvest one checkpoint and restart training with reset_to_latest=False and i_know_what_i_am_doing=False"
                )

        example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]] = queue.Queue(
            maxsize=self.max_queue_size
        )
        consumer = threading.Thread(
            target=self._async_kafka_consumer_arrow,
            args=(
                example_queue,
                batch_size,
                shard_index,
                num_shards,
                run_server,
                stop_event,
                self._reset_event,
            ),
            daemon=True,
        )
        consumer.start()

        total_loop_count = 0
        total_empty_count = 0
        consecutive_empty_count = 0
        self.offset_ptr = {}
        shard_attrs = {
            "training_name": self.name,
            "topic": self.topic_name,
            "ranker_index": str(shard_index),
        }

        while self.is_alive:
            total_loop_count += 1
            try:
                if self.export_metrics:
                    _queue_size_values[(self.name, self.topic_name, str(shard_index))] = (
                        example_queue.qsize()
                    )

                batch, offsets = example_queue.get(timeout=1.0)
                self.offset_ptr.update(offsets)
                consecutive_empty_count = 0

                if self.export_metrics and _kafka_batches_processed:
                    _kafka_batches_processed.add(1, shard_attrs)
                    for _part, _off in offsets.items():
                        _partition_delivered_offsets[
                            (self.name, self.topic_name, str(shard_index), _part)
                        ] = _off

                yield batch, self.offset_ptr.copy()
            except queue.Empty:
                total_empty_count += 1
                consecutive_empty_count += 1

                if self.export_metrics and _kafka_empty_polls:
                    _kafka_empty_polls.add(1, shard_attrs)

                if consecutive_empty_count % 100 == 0:
                    rank_logger.warning(
                        f"No data in the queue. Consecutive empty count: {consecutive_empty_count}, total loop count: {total_loop_count}, total empty count: {total_empty_count}"
                    )

    def _resolve_bootstrap_servers(self) -> str:
        return _resolve_bootstrap_servers(self.bootstrap_servers)

    def ensure_partition_count(self) -> None:
        if self.num_kafka_partitions is not None:
            return
        if not self.topic_name:
            return
        try:
            bootstrap_servers = self._resolve_bootstrap_servers()
            sasl_plain_password = _resolve_sasl_password(self.bootstrap_servers)
            count = asyncio.run(
                discover_partition_count(
                    topic=self.topic_name,
                    bootstrap_servers=bootstrap_servers,
                    sasl_mechanism=self.sasl_mechanism,
                    sasl_plain_username=self.sasl_plain_username,
                    sasl_plain_password=sasl_plain_password,
                )
            )
            object.__setattr__(self, "num_kafka_partitions", count)
            rank_logger.info(
                "Early partition discovery: num_kafka_partitions=%d for topic '%s'.",
                count,
                self.topic_name,
            )
        except Exception as e:
            rank_logger.warning(
                "Early partition discovery failed for topic '%s': %s. "
                "Falling back to num_kafka_partitions=None.",
                self.topic_name,
                e,
            )

    async def _periodic_reload_task(self, stop_event: threading.Event | None):
        last_reload_time = time.time()

        while not (stop_event and stop_event.is_set()):
            try:
                current_time = time.time()
                time_since_reload = current_time - last_reload_time

                if time_since_reload >= (self.global_ids_reload_interval_seconds):
                    rank_logger.info("Start to reload global ids...")
                    (
                        new_post_ids,
                        new_author_ids,
                        _,
                        new_post_sids,
                    ) = load_global_ids_from_parquet_file(
                        self.global_ids_file_path,
                        read_post_sid=self.use_post_sid,
                        sid_num_levels=self.sid_num_levels,
                    )
                    if new_post_ids is not None and new_author_ids is not None:
                        self.global_post_ids = new_post_ids
                        self.global_author_ids = new_author_ids
                        self.global_post_sids = new_post_sids
                        last_reload_time = time.time()
                    else:
                        rank_logger.warning("Failed to reload global ids, keeping existing values")

                await asyncio.sleep(self.global_ids_reload_interval_seconds / 2)

            except Exception as e:
                rank_logger.error(f"Error in periodic reload task: {e}")
                await asyncio.sleep(self.global_ids_reload_interval_seconds / 2)

    def _async_kafka_consumer_arrow(
        self,
        example_queue: queue.Queue[tuple[RecsysFeaturesBatch, dict[int, int]]],
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        stop_event: threading.Event | None = None,
        reset_event: threading.Event | None = None,
    ):
        del run_server
        assert self.seek_to_timestamp_ms is None or self.seek_to_offset is None

        rank_logger.info(f"[{os.getpid()}] Starting kafka consumer")

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

        async def consumption_loop(stop_event):
            reload_task = None
            if (
                self.num_global_negatives_per_example > 0
                and self.global_ids_reload_interval_seconds > 0
            ):
                reload_task = asyncio.create_task(self._periodic_reload_task(stop_event))

            try:
                post_process_fn = partial(
                    self.kafka_to_training_batch,
                    batch_size=batch_size,
                    shard_index=shard_index,
                )

                bootstrap_servers = self._resolve_bootstrap_servers()
                sasl_plain_password = _resolve_sasl_password(self.bootstrap_servers)

                try:
                    discovered_count = await discover_partition_count(
                        topic=self.topic_name,
                        bootstrap_servers=bootstrap_servers,
                        sasl_mechanism=self.sasl_mechanism,
                        sasl_plain_username=self.sasl_plain_username,
                        sasl_plain_password=sasl_plain_password,
                    )
                except Exception as e:
                    if self.num_kafka_partitions is not None:
                        rank_logger.warning(
                            "Failed to discover partition count for topic '%s': %s. "
                            "Using configured num_kafka_partitions=%d.",
                            self.topic_name,
                            e,
                            self.num_kafka_partitions,
                        )
                        discovered_count = self.num_kafka_partitions
                    else:
                        raise ValueError(
                            f"Failed to discover partition count for topic '{self.topic_name}' "
                            f"and num_kafka_partitions is not configured. "
                            f"Either set num_kafka_partitions explicitly or fix the connection "
                            f"to Kafka: {e}"
                        ) from e
                if self.num_kafka_partitions is None:
                    object.__setattr__(self, "num_kafka_partitions", discovered_count)
                    rank_logger.info(
                        "Auto-discovered num_kafka_partitions=%d for topic '%s'.",
                        discovered_count,
                        self.topic_name,
                    )
                elif self.num_kafka_partitions != discovered_count:
                    rank_logger.warning(
                        "Config num_kafka_partitions=%d differs from discovered "
                        "partition count %d for topic '%s'. Using discovered count.",
                        self.num_kafka_partitions,
                        discovered_count,
                        self.topic_name,
                    )
                    object.__setattr__(self, "num_kafka_partitions", discovered_count)
                lag_tracker = PartitionLagTracker()
                drop_controller: DropModeController | None = None
                if self.enable_drop_on_high_qps:
                    drop_controller = DropModeController(
                        lag_tracker=lag_tracker,
                        high_watermark_offsets=self.partition_high_watermark_offset_lag,
                        low_watermark_offsets=self.partition_low_watermark_offset_lag,
                        drop_rate_percents=self.drop_max_drop_rate,
                        increasing_duration_secs=self.drop_increasing_duration_secs,
                    )

                await consume_messages(
                    topic=self.topic_name,
                    group_id=self.group_id,
                    example_queue=example_queue,
                    batch_size=batch_size,
                    shard_index=shard_index,
                    num_shards=num_shards,
                    sasl_mechanism=self.sasl_mechanism,
                    sasl_plain_username=self.sasl_plain_username,
                    sasl_plain_password=sasl_plain_password,
                    bootstrap_servers=bootstrap_servers,
                    reset_to_latest=self.reset_to_latest,
                    seek_to_timestamp_ms=self.seek_to_timestamp_ms,
                    seek_to_offset=self.seek_to_offset,
                    stop_event=stop_event,
                    post_process_fn=post_process_fn,
                    sampling_factor=self.sampling_factor,
                    auto_offset_reset=self.auto_offset_reset,
                    training_name=self.name,
                    lag_tracker=lag_tracker,
                    drop_controller=drop_controller,
                    per_partition_poll=self.per_partition_poll,
                    consumer_mode=self.consumer_mode,
                    reset_to_latest_event=reset_event,
                    negative_downsample_rate=self.negative_downsample_rate,
                    negative_downsample_positive_action_indices=(
                        self.negative_downsample_positive_action_indices
                    ),
                )
            finally:
                if reload_task:
                    reload_task.cancel()

        try:
            asyncio.run(consumption_loop(stop_event))
        except asyncio.CancelledError:
            rank_logger.info(f"Consumer [{os.getpid()}] cancelled (shutdown).")
        except Exception as e:
            rank_logger.error(f"Consumer [{os.getpid()}] failed: {e}", exc_info=True)

    def shutdown(self):
        global consumer
        global stop_event
        self.is_alive = False

        if consumer is not None:
            stop_event.set()
            rank_logger.info("Waiting for kafka consumer to finish...")
            consumer.join(timeout=10)
            consumer = None

        _training_metric_values.clear()
        _partition_delivered_offsets.clear()

        if self.export_metrics and _OTEL_AVAILABLE:
            try:
                assert metrics is not None
                provider = metrics.get_meter_provider()
                flush = getattr(provider, "force_flush", None)
                if flush is not None:
                    flush(timeout_millis=5000)
                    logger.info("Flushed OTel metrics on shutdown")
            except Exception as e:
                logger.warning(f"Failed to flush OTel metrics on shutdown: {e}")

        rank_logger.info("Killing kafka consumer process...")


async def main():
    kafka_recsys_dataset = PhoenixKafkaDataset(
        topic_name="my_training_topic",
        group_id="my_training_topic_demo_group",
        max_queue_size=100,
        path=None,
        hash_table=HashTable(
            hash_keys=HashKeys(
                user_id_table_size=128,
                item_id_table_size=128,
                author_id_table_size=128,
            ),
            output_vocab_size=128,
        ),
        history_seq_len=1024,
        candidate_seq_len=128,
        input_vocab_size=128,
        output_vocab_size=128,
        pad_token=0,
    )
    for example in kafka_recsys_dataset.make(
        batch_size=128,
        shard_index=0,
        num_shards=1,
        run_server=False,
        server_hosts=[],
        skip_rows=0,
        keep_and_pad_partial_batch=None,
    ):
        rank_logger.info(f"example: {example}")
        print("example", example)
        print("input shape", example["user_ids"].shape)
        return


if __name__ == "__main__":
    asyncio.run(main())
