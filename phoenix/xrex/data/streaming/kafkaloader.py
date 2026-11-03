# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import asyncio
import dataclasses
import errno
import logging
import os
import queue
import re
import ssl
import sys
import threading
import time
import urllib.parse
from functools import cached_property, partial
from threading import Event
from typing import Any, Iterator, Mapping, Optional, cast

import numpy as np
import pyarrow as pa

from xai_configlib import configclass

_OTEL_AVAILABLE = False
try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except ImportError:
    metrics = None

from xrex import settings
from xrex.data.parquet_recsys import (
    DataPosition,
    PhoenixDataset,
    load_global_ids_from_parquet_file,
)
from xrex.data.recsys import id_list_features, reply_features
from xrex.data.recsys.feature_config import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
)
from xrex.data.recsys.recsys_batch import (
    ADS_COUNT_COLUMNS,
    EMBEDDING_CONFIG,
    HISTORY_URL_ID_COLUMN,
    USER_ADS_COUNT_COLUMNS,
    CandidateNegativeMode,
    EmbeddingType,
    RecsysFeaturesBatch,
    from_record_batch,
)
from xrex.data.streaming.history_url_embedding import (
    URL_EMBEDDING_DIM,
    HistoryUrlEmbeddingLookup,
    UrlEmbeddingLookupStats,
)
from xrex.data.streaming.kafkaconsumer import (
    DEFAULT_NEGATIVE_DOWNSAMPLE_POSITIVE_ACTION_INDICES,
    ConsumerMode,
    DropModeController,
    KafkaAuth,
    KafkaBootstrapOffsets,
    KafkaQueueItem,
    KafkaSourceIdentity,
    PartitionLagTracker,
    consume_messages,
    discover_partition_count,
    validate_offset_map,
    validate_seek_options,
)
from xrex.models.recsys_embedding import HashKeys, HashTable
from xrex.utils import cluster

log_level_str = os.environ.get("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
rank_logger = logging.getLogger("rank")


consumer: Optional[threading.Thread] = None
stop_event = threading.Event()
metrics_server_started = False

_kafka_reset_event: Optional[threading.Event] = None

_kafka_catchup_event: threading.Event = threading.Event()
_kafka_catchup_window_secs: float = 7200.0
_kafka_catchup_lock: threading.Lock = threading.Lock()
_kafka_catchup_generation: int = 0

_CATCHUP_MIN_SECS = 60.0
_CATCHUP_MAX_SECS = 24.0 * 3600.0
_CATCHUP_DEFAULT_SECS = 7200.0

_CATCHUP_DURATION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|mins|h|hr|hrs|hour|hours)?\s*$",
    re.IGNORECASE,
)


def _parse_catchup_duration(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return _CATCHUP_DEFAULT_SECS
    m = _CATCHUP_DURATION_RE.match(raw)
    if m is None:
        raise ValueError(
            f"unparseable duration {raw!r} (examples: '30min', '2hr', '90m', "
            "'7200s'; a bare number means minutes)"
        )
    value = float(m.group(1))
    unit = (m.group(2) or "m").lower()
    if unit.startswith("s"):
        secs = value
    elif unit.startswith("m"):
        secs = value * 60.0
    else:
        secs = value * 3600.0
    if not _CATCHUP_MIN_SECS <= secs <= _CATCHUP_MAX_SECS:
        raise ValueError(
            f"duration {raw!r} = {secs:.0f}s is outside "
            f"[{_CATCHUP_MIN_SECS:.0f}s, {_CATCHUP_MAX_SECS:.0f}s] "
            "(below 1 min use /reset-kafka instead; above 24 h dropping "
            "is not worth it)"
        )
    return secs


def _post_catchup_request(window_secs: float) -> None:
    global _kafka_catchup_window_secs, _kafka_catchup_generation
    with _kafka_catchup_lock:
        _kafka_catchup_window_secs = window_secs
        _kafka_catchup_generation += 1
        _kafka_catchup_event.set()


def _take_catchup_request() -> tuple[int, float] | None:
    with _kafka_catchup_lock:
        if _kafka_catchup_event.is_set():
            _kafka_catchup_event.clear()
            return _kafka_catchup_generation, _kafka_catchup_window_secs
    return None


def _rearm_catchup_request(generation: int) -> None:
    with _kafka_catchup_lock:
        if _kafka_catchup_generation == generation:
            _kafka_catchup_event.set()


_meter = None
_kafka_batches_processed = None
_kafka_messages_processed = None
_kafka_embedding_zero_filled = None
_kafka_batch_processing_time = None
_kafka_queue_size = None
_kafka_empty_polls = None
_kafka_partition_delivered_offset = None
_kafka_history_url_embedding = None
_kafka_history_url_embedding_mget_seconds = None

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
        timeout = 10

        def _respond(self, status: int, body: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body.encode())

        def do_POST(self):
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/reset-kafka":
                if _kafka_reset_event is not None:
                    _kafka_reset_event.set()
                    self._respond(200, "Kafka reset signal sent\n")
                    rank_logger.info(
                        "Kafka reset signal received via HTTP POST /reset-kafka on port %d",
                        self.server.server_address[1],
                    )
                else:
                    self._respond(503, "Consumer not started yet\n")
            elif parsed.path == "/catchup-kafka":
                if _kafka_reset_event is None:
                    self._respond(503, "Consumer not started yet\n")
                    return
                raw = (urllib.parse.parse_qs(parsed.query).get("duration") or [None])[0]
                if raw is None:
                    try:
                        length = int(self.headers.get("Content-Length") or 0)
                    except ValueError:
                        self._respond(400, "Invalid Content-Length\n")
                        return
                    if length > 0:
                        raw = self.rfile.read(min(length, 64)).decode("utf-8", "replace")
                try:
                    window_secs = _parse_catchup_duration(raw)
                except ValueError as e:
                    self._respond(400, f"{e}\n")
                    return
                _post_catchup_request(window_secs)
                self._respond(
                    200,
                    f"Kafka catch-up signal sent: window={window_secs:.0f}s "
                    "(acted on by the Rust consumer path only)\n",
                )
                rank_logger.info(
                    "Kafka catch-up signal received via HTTP POST /catchup-kafka "
                    "on port %d (window=%.0fs)",
                    self.server.server_address[1],
                    window_secs,
                )
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/reset-kafka":
                is_pending = _kafka_reset_event is not None and _kafka_reset_event.is_set()
                self._respond(200, f"reset_pending={is_pending}\n")
            elif parsed.path == "/catchup-kafka":
                self._respond(
                    200,
                    f"catchup_pending={_kafka_catchup_event.is_set()} "
                    f"window_secs={_kafka_catchup_window_secs:.0f} "
                    "(consumed by the Rust consumer path only)\n",
                )
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
        rank_logger.info(
            "Kafka control HTTP server listening on port %d "
            "(POST /reset-kafka, POST /catchup-kafka)",
            port,
        )
        return port
    rank_logger.warning(
        "Kafka control HTTP server: no free port in %d-%d; live /reset-kafka "
        "and /catchup-kafka disabled for this worker",
        base_port,
        base_port + num_ports - 1,
    )
    return None


def _init_otel_metrics(
    otel_endpoint: str,
    export_interval_ms: int = 120_000,
    worker_rank: int = 0,
    training_name: str = "recsys-kafka-loader",
) -> bool:
    global _meter, _kafka_batches_processed, _kafka_messages_processed
    global _kafka_embedding_zero_filled
    global _kafka_batch_processing_time, _kafka_queue_size, _kafka_empty_polls
    global _kafka_partition_delivered_offset
    global _kafka_history_url_embedding, _kafka_history_url_embedding_mget_seconds
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

    _kafka_embedding_zero_filled = _meter.create_counter(
        name="recsys_kafka_embedding_zero_filled_total",
        description="Records read as zero MM embeddings (no column, or null after unification)",
        unit="1",
    )

    _kafka_history_url_embedding = _meter.create_counter(
        name="recsys_kafka_history_url_embedding_total",
        description="History URL embedding lookup counts by outcome (UrlEmbeddingLookupStats)",
        unit="1",
    )

    _kafka_history_url_embedding_mget_seconds = _meter.create_counter(
        name="recsys_kafka_history_url_embedding_mget_seconds_total",
        description="Time spent in history URL embedding Redis MGETs",
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


def _record_history_url_embedding_stats(
    stats: UrlEmbeddingLookupStats, attrs: Mapping[str, str]
) -> None:
    if _kafka_history_url_embedding is not None:
        for outcome, count in dataclasses.asdict(stats).items():
            if outcome != "mget_seconds" and count:
                _kafka_history_url_embedding.add(count, {**attrs, "outcome": outcome})
    if _kafka_history_url_embedding_mget_seconds is not None and stats.mget_seconds:
        _kafka_history_url_embedding_mget_seconds.add(stats.mget_seconds, dict(attrs))


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

    index_by_names: dict[tuple[str, ...], dict[str, int]] = {}
    batch_indices: list[dict[str, int]] = []
    for rb in batches:
        names = tuple(rb.schema.names)
        index = index_by_names.get(names)
        if index is None:
            index = {}
            for i, name in enumerate(names):
                index[name] = -1 if name in index else i
            index_by_names[names] = index
        batch_indices.append(index)

    all_batch_names: set[str] = set()
    for index in index_by_names.values():
        all_batch_names.update(index)

    col_names: list[str] = list(required_columns)
    for opt_name in optional_columns:
        if opt_name in all_batch_names:
            col_names.append(opt_name)

    canonical_types: dict[str, pa.DataType] = {}
    for name in col_names:
        for rb, index in zip(batches, batch_indices):
            if name in index:
                if index[name] < 0:
                    rb.schema.field(name)
                canonical_types[name] = rb.schema.field(index[name]).type
                break

    missing_everywhere = [name for name in col_names if name not in canonical_types]
    if missing_everywhere:
        logger.warning(f"Columns not found in any batch (skipped): {missing_everywhere}")

    col_names = [name for name in col_names if name in canonical_types]

    target_schema = pa.schema([pa.field(name, canonical_types[name]) for name in col_names])

    result: list[pa.RecordBatch] = []
    for rb, index in zip(batches, batch_indices):
        if rb.schema.equals(target_schema):
            result.append(rb)
            continue
        if all(name in index for name in col_names):
            for name in col_names:
                if index[name] < 0:
                    rb.column(name)
            selected = rb.select([index[name] for name in col_names])
            if selected.schema.equals(target_schema, check_metadata=True):
                result.append(selected)
                continue
        num_rows = rb.num_rows
        new_columns: list[pa.Array] = []
        for name in col_names:
            target_type = canonical_types[name]
            if name in index:
                if index[name] < 0:
                    rb.schema.field(name)
                col = rb.column(index[name])
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
                if name in required_columns:
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


def resolve_internal_bootstrap(bootstrap_servers: str, auth_mode: str = "sasl") -> str:
    del auth_mode
    if _has_port(bootstrap_servers.strip()):
        return bootstrap_servers
    return settings.KAFKA_BOOTSTRAP_SERVERS or bootstrap_servers


def cluster_sasl_username(cluster_name: str) -> str:
    del cluster_name
    return settings.KAFKA_SASL_USERNAME


_DEFAULT_MTLS_CERTS_DIR = "/root/client-certs"
_MTLS_CA_BUNDLE = "ca-bundle.crt"
_MTLS_CLIENT_CERT = "client.fullchain"
_MTLS_CLIENT_KEY = "client.key"

_PLATFORM_CA_BUNDLE_PATH = "/etc/ssl/internal-ca/ca-bundle.crt"


def _mtls_certs_dir() -> str:
    return os.environ.get("KAFKA_MTLS_CERTS_DIR", _DEFAULT_MTLS_CERTS_DIR)


def _mtls_cert_paths(certs_dir: str) -> tuple[str, str, str] | None:
    ca = os.path.join(certs_dir, _MTLS_CA_BUNDLE)
    cert = os.path.join(certs_dir, _MTLS_CLIENT_CERT)
    key = os.path.join(certs_dir, _MTLS_CLIENT_KEY)
    paths = (ca, cert, key)
    missing = [p for p in paths if not os.path.isfile(p) or os.path.getsize(p) == 0]
    if not missing:
        return paths
    present = [p for p in paths if os.path.exists(p)]
    if present:
        rank_logger.warning(
            "Incomplete Kafka mTLS certs in %s (missing/empty: %s); "
            "falling back to the downloaded certificate or SASL password.",
            certs_dir,
            ", ".join(os.path.basename(p) for p in missing),
        )
    return None


def _mtls_client_paths(certs_dir: str) -> tuple[str, str] | None:
    cert = os.path.join(certs_dir, _MTLS_CLIENT_CERT)
    key = os.path.join(certs_dir, _MTLS_CLIENT_KEY)
    pair = (cert, key)
    if all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in pair):
        return pair
    return None


def platform_ca_bundle_path() -> str | None:
    path = os.environ.get("KAFKA_MTLS_PLATFORM_CA_BUNDLE", "").strip() or _PLATFORM_CA_BUNDLE_PATH
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    return None


def system_ca_bundle_path() -> str | None:
    cafile = ssl.get_default_verify_paths().cafile
    if cafile and os.path.isfile(cafile) and os.path.getsize(cafile) > 0:
        return cafile
    return None


def _try_resolve_sasl_password(bootstrap_servers: str) -> str | None:
    del bootstrap_servers
    return os.environ.get("SASL_PLAIN_PASSWORD") or None


def _resolve_mtls_bootstrap(bootstrap_servers: str) -> str:
    bs = bootstrap_servers.strip()
    if _has_port(bs):
        return bs
    raise ValueError(
        f"Cannot resolve mTLS bootstrap for {bs!r}: provide a raw broker address "
        "(host:port[,host:port...]) — cluster-name resolution is internal-only."
    )


def auto_detect_auth(
    bootstrap_servers: str,
    sasl_mechanism: str,
    sasl_username: str,
    certs_dir: str | None = None,
    download_dir: str | None = None,
) -> KafkaAuth:
    del download_dir
    certs_dir = certs_dir if certs_dir is not None else _mtls_certs_dir()
    force = os.environ.get("KAFKA_AUTH", "").strip().lower()
    verify_broker = os.environ.get("KAFKA_MTLS_VERIFY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    paths = _mtls_cert_paths(certs_dir)

    if force == "sasl":
        rank_logger.info("Kafka auth: KAFKA_AUTH=sasl, skipping mTLS certs.")
    elif paths is not None:
        rank_logger.info("Kafka auth: using mTLS certificates from %s", certs_dir)
        ca, cert, key = paths
        return KafkaAuth.mtls(ca, cert, key, verify_broker=verify_broker)
    else:
        client = _mtls_client_paths(certs_dir)
        if client is not None:
            truststore = platform_ca_bundle_path() if verify_broker else None
            if verify_broker and truststore is None:
                rank_logger.warning(
                    "Kafka auth: client certificate mounted at %s without %s and no "
                    "platform CA bundle; verifying the brokers against the system "
                    "trust store. Point KAFKA_MTLS_PLATFORM_CA_BUNDLE at a PEM if the "
                    "handshake fails with CERTIFICATE_VERIFY_FAILED.",
                    certs_dir,
                    _MTLS_CA_BUNDLE,
                )
            else:
                rank_logger.info(
                    "Kafka auth: client certificate from %s, broker truststore %s",
                    certs_dir,
                    truststore or "off (set KAFKA_MTLS_VERIFY=1 to enable it)",
                )
            return KafkaAuth.mtls(truststore, *client, verify_broker=verify_broker)
        if force == "mtls":
            raise RuntimeError(
                f"KAFKA_AUTH=mtls but no client certs were found under {certs_dir} "
                f"(expected {_MTLS_CA_BUNDLE}, {_MTLS_CLIENT_CERT}, {_MTLS_CLIENT_KEY})."
            )

    password = _try_resolve_sasl_password(bootstrap_servers)
    if password:
        sasl_username = sasl_username or settings.KAFKA_SASL_USERNAME
        rank_logger.info("Kafka auth: using SASL/SCRAM (%s) as %s", sasl_mechanism, sasl_username)
        return KafkaAuth.sasl(sasl_mechanism, sasl_username, password)

    raise RuntimeError(
        "No Kafka credentials found.\n"
        f"  Tried mTLS certs at {certs_dir} "
        f"(need {_MTLS_CA_BUNDLE}, {_MTLS_CLIENT_CERT}, {_MTLS_CLIENT_KEY})\n"
        "  Tried SASL via SASL_PLAIN_PASSWORD.\n"
        f"Set SASL_PLAIN_PASSWORD, or mount a client certificate at {certs_dir}."
    )


def _resolve_bootstrap_servers(bootstrap_servers: str, auth_mode: str = "sasl") -> str:
    bs = bootstrap_servers.strip()
    if auth_mode == "mtls":
        return _resolve_mtls_bootstrap(bs)
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


class KafkaConsumerStoppedError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class PreparedBatchCardinality:
    topic_name: str
    shard_index: int
    decoded_rows: int
    prepared_rows: int
    expected_rows: int | None
    leading_dimensions: tuple[tuple[str, int | None], ...]


class PreparedRowCountError(ValueError):
    cardinality: PreparedBatchCardinality

    def __init__(self, cardinality: PreparedBatchCardinality) -> None:
        self.cardinality = cardinality
        super().__init__(f"Kafka prepared-row shape mismatch: {cardinality}")


def inspect_prepared_batch(
    example: RecsysFeaturesBatch,
    *,
    topic_name: str,
    shard_index: int,
    decoded_rows: int,
    expected_rows: int | None,
) -> PreparedBatchCardinality:
    dimensions: list[tuple[str, int | None]] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, np.ndarray):
            dimensions.append((path, value.shape[0] if value.ndim else None))
        elif isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))

    visit(example, "")
    rows = len(example["user_hashes"])
    return PreparedBatchCardinality(
        topic_name, shard_index, decoded_rows, rows, expected_rows, tuple(dimensions)
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
    fresh_training_bootstrap: bool = False
    native_checkpoint_restart: bool = False
    seek_to_timestamp_ms: int | None = None
    auto_offset_reset: str = "latest"
    seek_to_offset: dict[int, int] | None = None
    offset_ptr: dict[int, int] | None = None
    expected_prepared_rows: int | None = None
    native_initial_positions: dict[int, int] | None = dataclasses.field(init=False, default=None)
    _native_expected_partitions: set[int] | None = dataclasses.field(init=False, default=None)
    num_kafka_partitions: int | None = None
    sampling_factor: int = 1
    partition_sampling_factor: int = 1
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
    _global_ids_lock = threading.Lock()

    global_ids_reload_interval_seconds: float = 0

    export_metrics: bool = True
    otel_endpoint: str = ""
    metrics_export_interval_ms: int = 120_000

    num_local_workers: int = 8

    reset_base_port: int = 10098

    multimodal_embedding_type: EmbeddingType | None = None
    max_candidates_for_embeddings: int = 64

    compute_post_unexplored_label: bool = False
    search_negative_clear_word_match: bool = False

    history_url_embedding_cache: str = ""
    history_url_embedding_dc: str = "atla"
    history_url_embedding_cache_entries: int = 200_000
    history_url_embedding_timeout_s: float = 0.1
    history_url_embedding_sample_rate: float = 1.0
    history_url_embedding_negative_ttl_s: float = 3600.0

    exclude_required_columns: str = ""

    def prepare_continuation(
        self,
        source: KafkaSourceIdentity,
        offsets: Mapping[int, int],
        *,
        continuation: bool,
        offset_ranges: Mapping[int, tuple[int, int]] | None = None,
    ) -> "PhoenixKafkaDataset":
        if not continuation:
            raise ValueError("Kafka continuation mode must be explicitly requested")
        if (self.bootstrap_servers, self.topic_name, self.num_kafka_partitions) != (
            source.bootstrap_servers,
            source.topic_name,
            source.num_partitions,
        ):
            raise ValueError("Kafka continuation source identity mismatch")
        validated = validate_offset_map(
            offsets,
            num_partitions=source.num_partitions,
            required_partitions=set(range(source.num_partitions)),
            offset_ranges=offset_ranges,
        )
        return dataclasses.replace(self, seek_to_timestamp_ms=None, seek_to_offset=validated)

    def kafka_to_training_batch(
        self, batch: list[pa.RecordBatch], batch_size: int, shard_index: int = 0
    ) -> RecsysFeaturesBatch:
        del batch_size
        start_time = time.time()

        required_columns: list[str] = list(REQUIRED_COLUMNS)
        if self.exclude_required_columns:
            excluded = {c.strip() for c in self.exclude_required_columns.split(",") if c.strip()}
            required_columns = [c for c in required_columns if c not in excluded]

        if not self._logged_schema and batch:
            first_rb = batch[0]
            available_cols = first_rb.schema.names
            missing_cols = [col for col in required_columns if col not in available_cols]
            if missing_cols:
                rank_logger.error(
                    f"Schema mismatch! Missing required columns: {missing_cols}. "
                    f"Available columns: {available_cols}"
                )
            else:
                rank_logger.info(f"Schema validated. Available columns: {available_cols}")
            self._logged_schema = True

        optional_columns = OPTIONAL_COLUMNS
        if self.reply_parent_mode:
            optional_columns = list(
                dict.fromkeys(OPTIONAL_COLUMNS + reply_features.REPLY_OPTIONAL_COLUMNS)
            )
        if self.id_list_specs:
            optional_columns = [*optional_columns, id_list_features.COLUMN]
        if self.read_ads_engagement_counts:
            optional_columns = [*optional_columns, *ADS_COUNT_COLUMNS, *USER_ADS_COUNT_COLUMNS]
        embedding_col_name: str | None = None
        if self.multimodal_embedding_type is not None:
            embedding_col_name, _ = EMBEDDING_CONFIG[self.multimodal_embedding_type]
            optional_columns = [*optional_columns, embedding_col_name]
        history_url_lookup = self._history_url_lookup
        if history_url_lookup is not None:
            optional_columns = [*optional_columns, HISTORY_URL_ID_COLUMN]
        unified_batches = _unify_record_batch_schemas(batch, required_columns, optional_columns)

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
        embedding_zero_filled = 0
        if embedding_col_name is not None:
            embedding_zero_filled = (
                table.column(embedding_col_name).null_count
                if embedding_col_name in table.column_names
                else table.num_rows
            )

        with self._global_ids_lock:
            global_post_ids = self.global_post_ids
            global_author_ids = self.global_author_ids
            global_post_sids = self.global_post_sids

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
            global_post_ids,
            global_author_ids,
            self.max_candidates_for_embeddings,
            self.multimodal_embedding_type,
            search_query_embedding_dim=self.search_query_embedding_dim,
            aux_value_events=self.aux_value_events,
            candidate_negative_filter=self.candidate_negative_filter,
            candidate_negative_mode=(
                self.candidate_negative_mode
                if self.candidate_negative_mode is not None
                else CandidateNegativeMode.FILTER
            ),
            global_post_sids=global_post_sids,
            sid_num_levels=self.sid_num_levels if self.use_post_sid else 0,
            compute_post_unexplored_label=self.compute_post_unexplored_label,
            zero_stale_post_14d_candidate_counts=self.enable_stale_post,
            stale_post_30d=self.enable_stale_post_30d,
            search_negative_clear_word_match=self.search_negative_clear_word_match,
            reply_parent_mode=self.reply_parent_mode,
            ads_head_masking=self.ads_head_masking,
            ads_pixel_head_masking=self.ads_pixel_head_masking,
            ads_late_window_heads=self.ads_late_window_heads,
            page_records=self.page_records,
            id_list_specs=self.id_list_specs or None,
            history_url_embedding_dim=self.history_url_embedding_dim,
            history_url_embedding_fn=history_url_lookup,
            read_ads_engagement_counts=self.read_ads_engagement_counts,
            alloc=self._alloc,
        )

        cardinality = inspect_prepared_batch(
            example,
            topic_name=self.topic_name,
            shard_index=shard_index,
            decoded_rows=table.num_rows,
            expected_rows=self.expected_prepared_rows,
        )
        if self.expected_prepared_rows is not None and (
            cardinality.prepared_rows != self.expected_prepared_rows
            or any(
                rows != self.expected_prepared_rows for _, rows in cardinality.leading_dimensions
            )
        ):
            rank_logger.error("Kafka prepared-row shape mismatch: %s", cardinality)
            raise PreparedRowCountError(cardinality)

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
            if _kafka_embedding_zero_filled and embedding_col_name is not None:
                _kafka_embedding_zero_filled.add(embedding_zero_filled, shard_attrs)
            if history_url_lookup is not None:
                _record_history_url_embedding_stats(history_url_lookup.take_stats(), shard_attrs)

        return example

    @cached_property
    def _history_url_lookup(self) -> HistoryUrlEmbeddingLookup | None:
        if self.history_url_embedding_dim <= 0:
            return None
        if not self.history_url_embedding_cache:
            rank_logger.warning(
                "history_url_embedding_dim > 0 without history_url_embedding_cache: "
                "history URL embeddings stay zero"
            )
            return None
        if self.history_url_embedding_dim != URL_EMBEDDING_DIM:
            raise ValueError(
                f"history_url_embedding_dim={self.history_url_embedding_dim} must be "
                f"{URL_EMBEDDING_DIM} to read the URL embedding cache"
            )
        return HistoryUrlEmbeddingLookup.from_cache(
            self.history_url_embedding_cache,
            self.history_url_embedding_dc,
            self.history_url_embedding_timeout_s,
            self.history_url_embedding_cache_entries,
            self.history_url_embedding_sample_rate,
            self.history_url_embedding_negative_ttl_s,
        )

    def exports_python_otel(self, shard_index: int, num_shards: int) -> bool:
        return True

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
        validate_seek_options(self.seek_to_timestamp_ms, self.seek_to_offset)
        if type(self.native_checkpoint_restart) is not bool:
            raise ValueError("Native restart policy must be an exact Boolean")
        if self.native_checkpoint_restart:
            if type(self.num_kafka_partitions) is not int or self.num_kafka_partitions <= 0:
                raise ValueError("Native restart requires a known partition count")
            if (
                self.fresh_training_bootstrap
                or self.is_eval
                or not self.topic_name
                or self.seek_to_timestamp_ms is not None
                or self.reset_to_latest
                or self.i_know_what_i_am_doing
                or self.auto_offset_reset != "none"
                or self.sampling_factor != 1
                or self.negative_downsample_rate != 1
                or self.enable_drop_on_high_qps
                or self.seek_to_offset is None
            ):
                raise ValueError(
                    "Native restart requires restored offsets and unchanged input policy"
                )
            validate_offset_map(
                self.seek_to_offset,
                num_partitions=self.num_kafka_partitions,
                required_partitions=set(range(self.num_kafka_partitions)),
            )
        if type(self.fresh_training_bootstrap) is not bool:
            raise ValueError("Fresh bootstrap policy must be an exact Boolean")
        if self.fresh_training_bootstrap:
            if (
                self.is_eval
                or not self.topic_name
                or type(self.seek_to_timestamp_ms) is not int
                or self.seek_to_timestamp_ms <= 0
                or self.seek_to_offset is not None
                or self.reset_to_latest
                or self.i_know_what_i_am_doing
                or self.auto_offset_reset != "none"
                or self.sampling_factor != 1
                or self.negative_downsample_rate != 1
                or self.enable_drop_on_high_qps
                or type(self.num_kafka_partitions) is not int
                or self.num_kafka_partitions <= 0
            ):
                raise ValueError(
                    "Fresh training requires an explicit unsampled timestamp bootstrap"
                )
        elif not self.is_eval and not self.i_know_what_i_am_doing:
            if self.seek_to_timestamp_ms is not None or self.reset_to_latest:
                raise ValueError(
                    "Training timestamp or reset requires an explicit bootstrap policy"
                )
        if self.expected_prepared_rows is not None and (
            type(self.expected_prepared_rows) is not int or self.expected_prepared_rows <= 0
        ):
            raise ValueError("expected_prepared_rows must be a positive integer")
        global consumer, stop_event, _kafka_reset_event
        if consumer is not None:
            if consumer.is_alive():
                raise ValueError("Called make() twice, process is already running.")
            consumer = None

        native_bootstrap = self.fresh_training_bootstrap or self.native_checkpoint_restart
        if native_bootstrap:
            if self.partition_sampling_factor != 1:
                raise ValueError("Native Kafka requires unsampled partitions")
            if (
                type(shard_index) is not int
                or type(num_shards) is not int
                or not 0 <= shard_index < num_shards
            ):
                raise ValueError("Native Kafka shard index must be within the positive shard count")
            assert self.num_kafka_partitions is not None
            if num_shards > self.num_kafka_partitions:
                raise ValueError("Native Kafka requires at least one partition per shard")
            self._native_expected_partitions = set(
                range(
                    shard_index * self.num_kafka_partitions // num_shards,
                    (shard_index + 1) * self.num_kafka_partitions // num_shards,
                )
            )
            self.native_initial_positions = None
        stop_event = threading.Event()
        self._stop_event = stop_event
        self.is_alive = True
        self._reset_event: Event = threading.Event()
        _kafka_reset_event = self._reset_event

        if self.export_metrics and self.exports_python_otel(shard_index, num_shards):
            try:
                _init_otel_metrics(
                    otel_endpoint=self.otel_endpoint,
                    export_interval_ms=self.metrics_export_interval_ms,
                    worker_rank=shard_index,
                    training_name=self.name,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OTel push metrics: {e}")

        if (
            not self.is_eval
            and not self.fresh_training_bootstrap
            and not self.native_checkpoint_restart
        ):
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

        example_queue: queue.Queue[KafkaQueueItem] = queue.Queue(maxsize=self.max_queue_size)
        consumer = threading.Thread(
            target=self._async_kafka_consumer_arrow,
            args=(
                example_queue,
                batch_size,
                shard_index,
                num_shards,
                run_server,
                self._stop_event,
                self._reset_event,
            ),
            daemon=True,
        )
        self._consumer_thread = consumer
        consumer.start()

        if self.fresh_training_bootstrap or self.native_checkpoint_restart:
            return self._fresh_training_batches(example_queue, shard_index)
        return self._iterate_training_batches(example_queue, shard_index)

    def _fresh_training_batches(self, example_queue, shard_index):
        try:
            yield from self._iterate_training_batches(example_queue, shard_index)
        finally:
            primary = sys.exception()
            try:
                self.shutdown()
            except Exception as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"Fresh input cleanup failed: {cleanup!r}")

    def _iterate_training_batches(
        self,
        example_queue: queue.Queue[KafkaQueueItem],
        shard_index: int,
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        total_loop_count = 0
        total_empty_count = 0
        consecutive_empty_count = 0
        self.offset_ptr = {}
        native_bootstrap = self.fresh_training_bootstrap or self.native_checkpoint_restart
        shard_attrs = {
            "training_name": self.name,
            "topic": self.topic_name,
            "ranker_index": str(shard_index),
        }

        while self.is_alive:
            total_loop_count += 1
            if self.export_metrics:
                _queue_size_values[(self.name, self.topic_name, str(shard_index))] = (
                    example_queue.qsize()
                )

            try:
                item = example_queue.get(timeout=1.0)
            except queue.Empty:
                if consumer is not None and not consumer.is_alive() and example_queue.empty():
                    raise KafkaConsumerStoppedError("Kafka consumer stopped with an empty queue")
                total_empty_count += 1
                consecutive_empty_count += 1

                if self.export_metrics and _kafka_empty_polls:
                    _kafka_empty_polls.add(1, shard_attrs)

                if consecutive_empty_count % 100 == 0:
                    rank_logger.warning(
                        f"No data in the queue. Consecutive empty count: {consecutive_empty_count}, total loop count: {total_loop_count}, total empty count: {total_empty_count}"
                    )
                continue

            if isinstance(item, Exception):
                raise item
            if isinstance(item, KafkaBootstrapOffsets):
                if not native_bootstrap or item.topic_name != self.topic_name:
                    raise ValueError("Unexpected Kafka bootstrap topic or mode")
                if self.native_initial_positions is not None:
                    raise ValueError("Duplicate Kafka bootstrap frontier")
                if self._native_expected_partitions is None:
                    raise ValueError("Native Kafka bootstrap shard assignment is unavailable")
                offsets = validate_offset_map(
                    dict(item.offsets),
                    num_partitions=self.num_kafka_partitions,
                    required_partitions=self._native_expected_partitions,
                )
                if not offsets:
                    raise ValueError("Native Kafka shard has no partitions")
                self.native_initial_positions = offsets.copy()
                self.offset_ptr.update(offsets)
                continue
            batch, offsets = item
            if native_bootstrap:
                if self.native_initial_positions is None:
                    raise ValueError("Kafka data arrived before its native bootstrap frontier")
                offsets = validate_offset_map(offsets, num_partitions=self.num_kafka_partitions)
                if not offsets or not offsets.keys() <= self.native_initial_positions.keys():
                    raise ValueError("Unexpected Kafka partition in native batch")
                if any(
                    offset < self.offset_ptr[partition] for partition, offset in offsets.items()
                ):
                    raise ValueError("Kafka native batch offsets decreased")
            report_metrics = self.export_metrics and _kafka_batches_processed
            if report_metrics:
                advanced = {
                    _part: _off
                    for _part, _off in offsets.items()
                    if _part in self.offset_ptr and self.offset_ptr[_part] != _off
                }
            self.offset_ptr.update(offsets)
            consecutive_empty_count = 0

            if report_metrics:
                _kafka_batches_processed.add(1, shard_attrs)
                for _part, _off in advanced.items():
                    _partition_delivered_offsets[
                        (self.name, self.topic_name, str(shard_index), _part)
                    ] = _off

            yield batch, self.offset_ptr.copy()

    def _detect_auth(self) -> KafkaAuth:
        return auto_detect_auth(
            self.bootstrap_servers,
            self.sasl_mechanism,
            self.sasl_plain_username,
        )

    def _resolve_bootstrap_servers(self, auth: KafkaAuth | None = None) -> str:
        mode = auth.mode if auth is not None else "sasl"
        return _resolve_bootstrap_servers(self.bootstrap_servers, auth_mode=mode)

    def ensure_partition_count(self) -> None:
        validate_seek_options(self.seek_to_timestamp_ms, self.seek_to_offset)
        if self.num_kafka_partitions is not None:
            return
        if not self.topic_name:
            return
        try:
            auth = self._detect_auth()
            bootstrap_servers = self._resolve_bootstrap_servers(auth)
            count = asyncio.run(
                discover_partition_count(
                    topic=self.topic_name,
                    bootstrap_servers=bootstrap_servers,
                    sasl_mechanism=self.sasl_mechanism,
                    sasl_plain_username=self.sasl_plain_username,
                    sasl_plain_password=auth.sasl_password or "",
                    auth=auth,
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
                        with self._global_ids_lock:
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
        example_queue: queue.Queue[KafkaQueueItem],
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        stop_event: threading.Event | None = None,
        reset_event: threading.Event | None = None,
    ) -> None:
        try:
            self._run_kafka_consumer_arrow(
                example_queue,
                batch_size,
                shard_index,
                num_shards,
                run_server,
                stop_event,
                reset_event,
            )
        except asyncio.CancelledError:
            rank_logger.info(f"Consumer [{os.getpid()}] cancelled (shutdown).")
        except Exception as error:
            rank_logger.error(f"Consumer [{os.getpid()}] failed: {error}", exc_info=True)
            while not (stop_event and stop_event.is_set()):
                try:
                    example_queue.put(error, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _run_kafka_consumer_arrow(
        self,
        example_queue: queue.Queue[KafkaQueueItem],
        batch_size: int,
        shard_index: int,
        num_shards: int,
        run_server: bool,
        stop_event: threading.Event | None = None,
        reset_event: threading.Event | None = None,
    ):
        del run_server
        validate_seek_options(self.seek_to_timestamp_ms, self.seek_to_offset)

        rank_logger.info(f"[{os.getpid()}] Starting kafka consumer")

        if self.num_global_negatives_per_example > 0:
            (
                global_post_ids,
                global_author_ids,
                _,
                global_post_sids,
            ) = load_global_ids_from_parquet_file(
                self.global_ids_file_path,
                read_post_sid=self.use_post_sid,
                sid_num_levels=self.sid_num_levels,
            )
            with self._global_ids_lock:
                self.global_post_ids = global_post_ids
                self.global_author_ids = global_author_ids
                self.global_post_sids = global_post_sids

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

                auth = self._detect_auth()
                bootstrap_servers = self._resolve_bootstrap_servers(auth)
                sasl_plain_password = auth.sasl_password or ""

                try:
                    discovered_count = await discover_partition_count(
                        topic=self.topic_name,
                        bootstrap_servers=bootstrap_servers,
                        sasl_mechanism=self.sasl_mechanism,
                        sasl_plain_username=self.sasl_plain_username,
                        sasl_plain_password=sasl_plain_password,
                        auth=auth,
                        stop_event=stop_event,
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
                    auth=auth,
                    bootstrap_servers=bootstrap_servers,
                    reset_to_latest=self.reset_to_latest,
                    seek_to_timestamp_ms=self.seek_to_timestamp_ms,
                    seek_to_offset=self.seek_to_offset,
                    fresh_training_bootstrap=self.fresh_training_bootstrap,
                    native_checkpoint_restart=self.native_checkpoint_restart,
                    stop_event=stop_event,
                    post_process_fn=post_process_fn,
                    sampling_factor=self.sampling_factor,
                    partition_sampling_factor=self.partition_sampling_factor,
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
                    await asyncio.gather(reload_task, return_exceptions=True)

        asyncio.run(consumption_loop(stop_event))

    def shutdown(self):
        global consumer
        global stop_event
        self.is_alive = False

        owned_consumer = getattr(self, "_consumer_thread", consumer)
        owned_stop = getattr(self, "_stop_event", stop_event)
        if owned_consumer is not None:
            owned_stop.set()
            rank_logger.info("Waiting for kafka consumer to finish...")
            owned_consumer.join(timeout=10)
            if owned_consumer.is_alive():
                rank_logger.error(
                    "Kafka consumer did not stop within ten seconds; retaining its handle until it exits"
                )
            else:
                if consumer is owned_consumer:
                    consumer = None
                self._consumer_thread = None

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
