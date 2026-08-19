# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import concurrent.futures
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import threading
import time
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, final

import haiku as hk
import jax
import jax.numpy as jnp
import jax.profiler as jax_profiler
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import xai_recsys_engine
from jax.sharding import PartitionSpec as P
from xai_checkpointing.common import _unsafe_jax2np
from xai_checkpointing.tree_util import (
    keystr,
    tree_to_dict,
)
from xai_configlib import configclass
from xai_recsys_engine import (
    PyMmEmbeddingsClient,
    RankingBatchPrep,
    RetrievalBatchPrep,
)

from xrex.configs.xrecsys_two_tower import RecsysTwoTowerModelConfig
from xrex.cuda import adler32
from xrex.data.parquet_recsys import (
    PhoenixDataset,
)
from xrex.data.recsys import recsys_batch
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.data.recsys.sequence_packing import pack_batch
from xrex.data.retrieval_dataset import RetrievalDataset
from xrex.inference import debug_logger, service_registry
from xrex.inference.h2d import (
    EmbeddingSlices,
    H2DState,
    MultimodalEmbeddingBuffer,
    _create_mm_embedding_buffer_views,
    create_h2d_state,
    create_memmap_emb_table,
    lookup_h2d_embeddings,
    open_existing_memmap_emb_table,
    parallel_copyto,
    prefault_memmap,
)
from xrex.inference.metrics import MetricsPublisher, get_metrics_publisher
from xrex.inference.status_server import StatusServer, StatusServerConfig
from xrex.models.model_utils import Parameter, unwrap_tree
from xrex.models.recsys_embedding import RecsysEmbeddings
from xrex.models.recsys_model import RecsysAggregatedModelConfig
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.train.embedding_loader import load_embedding_table
from xrex.train.misc import PostEmbeddings, RecsysInferenceState
from xrex.train.trainer import Trainer
from xrex.train.trainer_recsys import (
    RecsysTrainer,
    Store,
    TrainerContext,
)
from xrex.utils import ocdbt
from xrex.utils.aot import JittedOrCompiled
from xrex.utils.log_timer import Timer
from xrex.utils.profiler import start_trace, stop_trace
from xrex.utils.utils import get_peak_bytes_in_use

logger = logging.getLogger(__name__)


def get_model_checkpoint_timestamp(ctx):
    if ctx.checkpoint is None:
        return 0.0
    checkpoint_dir = ctx.checkpoint.path
    completed_file = os.path.join(checkpoint_dir, "completed")
    if os.path.exists(completed_file):
        return os.path.getmtime(completed_file)
    else:
        return 0.0


def _classify_copy_port_error(error: Exception) -> str:
    error_str = str(error).lower()

    if "getaddrinfo" in error_str or "name or service not known" in error_str:
        return "dns_resolution"
    if "connect" in error_str and ("refused" in error_str or "failed" in error_str):
        return "grpc_connection"
    if "timed out" in error_str or "timeout" in error_str or "deadline_exceeded" in error_str:
        if "rdma" in error_str:
            return "rdma_timeout"
        return "grpc_timeout"
    if "size mismatch" in error_str:
        return "size_mismatch"
    if "checksum" in error_str:
        return "checksum_mismatch"
    if "transport" in error_str:
        return "transport"
    return "unknown"


COPY_PORT_PROBE_TIMEOUT = 12.0

MAX_INFLIGHT_COPY_PORT_PROBES = 4


def _resolve_copy_url_to_http(copy_url: str) -> str:
    name, port = copy_url.split(":")
    addr_info = socket.getaddrinfo(name, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
    return ",".join(f"http://{x[4][0]}:{port}" for x in addr_info)


_SHM_WEIGHTS_PATH = "/dev/shm/model_weights.bin"
_SHM_WEIGHTS_SIGNAL = "/dev/shm/model_weights.loaded"
_SHM_WEIGHTS_META = "/dev/shm/model_weights.meta"
_SHM_WEIGHTS_VERSION = "/dev/shm/model_weights.version"


def _read_version_file(path: str) -> int:
    try:
        return int(open(path).read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return 0


def _write_version_file(path: str, version: int) -> None:
    open(path, "w").write(str(version))


def _process_start_time_epoch() -> float | None:
    try:
        clk_tck = os.sysconf("SC_CLK_TCK")
        if clk_tck <= 0:
            return None
        with open("/proc/self/stat") as f:
            stat = f.read()
        starttime_ticks = int(stat[stat.rfind(")") + 2 :].split()[19])
        btime: float | None = None
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime "):
                    btime = float(line.split()[1])
                    break
        if btime is None:
            return None
        return btime + starttime_ticks / clk_tck
    except (OSError, ValueError, IndexError):
        return None


def _save_tensors_to_shm(
    tensors: list[tuple[str, np.ndarray]],
    prefix: str,
    checksums: str,
    created_timestamp: float,
) -> None:
    meta = {
        "prefix": prefix,
        "checksums": checksums,
        "created_timestamp": created_timestamp,
        "tensors": [(name, arr.nbytes) for name, arr in tensors],
    }
    with open(_SHM_WEIGHTS_META, "w") as f:
        json.dump(meta, f)

    total_bytes = sum(arr.nbytes for _, arr in tensors)
    fd = os.open(_SHM_WEIGHTS_PATH, os.O_CREAT | os.O_RDWR)
    os.ftruncate(fd, total_bytes)
    buf = mmap.mmap(fd, total_bytes)
    offset = 0
    for _, arr in tensors:
        buf[offset : offset + arr.nbytes] = bytes(arr)
        offset += arr.nbytes
    buf.close()
    os.close(fd)


def _load_tensors_from_shm(
    tensors: list[tuple[str, np.ndarray]],
) -> tuple[str, str, float]:
    with open(_SHM_WEIGHTS_META) as f:
        meta = json.load(f)

    fd = os.open(_SHM_WEIGHTS_PATH, os.O_RDONLY)
    total_bytes = sum(arr.nbytes for _, arr in tensors)
    buf = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_READ)
    offset = 0
    for _, arr in tensors:
        arr[:] = np.frombuffer(buf[offset : offset + arr.nbytes], dtype=np.uint8)
        offset += arr.nbytes
    buf.close()
    os.close(fd)

    return meta["prefix"], meta["checksums"], meta["created_timestamp"]


ModelConfig = TypeVar("ModelConfig")


class RequestBatchProtocol(Protocol):
    def len(self) -> int: ...


RequestBatch = TypeVar("RequestBatch", bound=RequestBatchProtocol)


@dataclass(frozen=True, slots=True)
class _PipelinedBatchResult(typing.Generic[RequestBatch]):
    request: RequestBatch
    out: jax.Array | np.ndarray | tuple[jax.Array, jax.Array, jax.Array]
    orig_batch_size: int
    bucket_size: int
    inf_timer: Timer
    batch_id: int


class ServerProtocol(Protocol, Generic[RequestBatch]):
    def set_ready(self, is_ready: bool): ...

    def cancel(self): ...

    def cancel_forcefully(self): ...

    def dequeue(self) -> RequestBatch | None: ...

    def check_reload_request(self) -> bool: ...

    def reset_reload_request(self) -> None: ...

    def reset_dequeue_timer(self) -> None: ...

    def record_service_time_ms(self, ms: float) -> None: ...

    def set_deadline_admission_enabled(self, enabled: bool) -> None: ...

    def set_checkpoint_loading(self) -> None: ...

    def clear_checkpoint_loading(self) -> None: ...

    def set_checkpoint_timestamp(self, timestamp: float) -> None: ...

    def compute_batch_inputs_rust_quick(
        self,
        request: RequestBatch,
        request_in_flight_id: int,
        input: Any,
        batch_size: int | None = None,
    ) -> bool: ...

    def is_finished(self) -> bool: ...


DEFAULT_COPY_PORT_TIMEOUT = 600


def _loader_subprocess_main(
    shm_dir: str,
    copy_url: str,
    emb_table_shape: tuple[int, int],
    emb_table_dtype: str,
    dense_staging_info: list[tuple[str, int, int]],
    dense_staging_total: int,
    pe_staging_size: int,
    pe_key: str,
    request_pipe,
    result_pipe,
    rate_limit_bytes_per_sec: int | None = None,
    max_concurrent_downloads: int | None = None,
):
    import json as _json
    import logging as _logging

    import numpy as np

    _log = _logging.getLogger("hotswap.subprocess")
    _logging.basicConfig(level=_logging.INFO)

    try:
        import xai_recsys_engine as _engine
    except Exception as exc:
        result_pipe.send(("error", f"failed to import xai_recsys_engine: {exc}"))
        return

    emb_dtype = np.dtype(emb_table_dtype)
    emb_slots = [
        np.memmap("/dev/shm/emb_table.dat", dtype=emb_dtype, mode="r+", shape=emb_table_shape),
        np.memmap(
            "/dev/shm/emb_table_standby.dat",
            dtype=emb_dtype,
            mode="r+",
            shape=emb_table_shape,
        ),
    ]
    dense_staging = (
        np.memmap(
            "/dev/shm/model_weights.bin",
            dtype=np.uint8,
            mode="r+",
            shape=(dense_staging_total,),
        )
        if dense_staging_total > 0
        else None
    )

    _log.info(
        "Subprocess started: shm_dir=%s, emb_shape=%s, dense_total=%d, pe_size=%d",
        shm_dir,
        emb_table_shape,
        dense_staging_total,
        pe_staging_size,
    )

    while True:
        try:
            msg = request_pipe.recv()
        except (EOFError, OSError):
            _log.info("Request pipe closed, exiting subprocess")
            break

        if msg == "stop":
            _log.info("Received stop signal, exiting subprocess")
            break

        elapsed_samples, active_slot, verify_checksums = msg
        standby_slot = 1 - active_slot

        try:
            resolved_url = _resolve_copy_url_to_http(copy_url)

            tensors = []
            if dense_staging is not None:
                for key, offset, nbytes in dense_staging_info:
                    arr = np.ndarray(nbytes, dtype=np.uint8, buffer=dense_staging, offset=offset)
                    tensors.append((key, arr))

            prefix, checksums, created_ts = _engine.maybe_load_fully_replicated_tensors(
                elapsed_samples,
                resolved_url,
                tensors,
                rate_limit_bytes_per_sec,
                max_concurrent_downloads,
            )

            standby_emb = emb_slots[standby_slot]
            emb_buf = np.ndarray(standby_emb.nbytes, dtype=np.uint8, buffer=standby_emb)
            try:
                _engine.madvise_sequential(emb_buf)
            except Exception:
                pass
            (emb_checksum,) = _engine.load_tensor_no_resharding(
                f"{prefix}/",
                resolved_url,
                "emb_table",
                emb_buf,
                rate_limit_bytes_per_sec,
                max_concurrent_downloads,
            )
            try:
                _engine.madvise_normal(emb_buf)
            except Exception:
                pass

            pe_checksum = None
            if pe_staging_size > 0:
                pe_staging = np.memmap(
                    "/dev/shm/post_embeddings.dat",
                    dtype=np.uint8,
                    mode="r+",
                    shape=(pe_staging_size,),
                )
                (pe_checksum,) = _engine.load_tensor_no_resharding(
                    f"{prefix}/",
                    resolved_url,
                    pe_key,
                    pe_staging,
                    rate_limit_bytes_per_sec,
                    max_concurrent_downloads,
                )

            if dense_staging is not None:
                dense_staging.flush()
            standby_emb.flush()
            if pe_staging_size > 0 and pe_staging is not None:
                pe_staging.flush()

            if verify_checksums:
                try:
                    global_ck = _json.loads(checksums)["global_checksums"]
                except Exception as exc:
                    result_pipe.send(("error", f"live verify: bad checksums json: {exc}"))
                    continue
                mismatch = None
                if dense_staging is not None:
                    for key, offset, nbytes in dense_staging_info:
                        if key not in global_ck:
                            continue
                        view = np.ndarray(
                            nbytes, dtype=np.uint8, buffer=dense_staging, offset=offset
                        )
                        computed = _engine.adler32_parallel(view)
                        if computed != global_ck[key]:
                            mismatch = (key, computed, global_ck[key])
                            break
                if mismatch is None and emb_checksum != global_ck.get("emb_table"):
                    mismatch = ("emb_table", emb_checksum, global_ck.get("emb_table"))
                if (
                    mismatch is None
                    and pe_checksum is not None
                    and pe_checksum != global_ck.get(pe_key)
                ):
                    mismatch = (pe_key, pe_checksum, global_ck.get(pe_key))
                if mismatch is not None:
                    k, computed, saved = mismatch
                    result_pipe.send(
                        (
                            "error",
                            f"live checksum mismatch on {k} (computed={computed}, saved={saved}); "
                            "if this recurs on every reload, adler32_parallel may not match the "
                            "training checksum algorithm -- disable verify_checksums or live swap",
                        )
                    )
                    continue
                _log.info("Checksums verified in subprocess for prefix=%s", prefix)

            result_pipe.send(
                (
                    "ok",
                    prefix,
                    checksums,
                    created_ts,
                    emb_checksum,
                    pe_checksum,
                )
            )
            _log.info("Download complete: prefix=%s", prefix)

        except Exception as e:
            _log.warning("Download failed: %s", e)
            result_pipe.send(("error", str(e)))


@configclass
class BaseModelRunner(RecsysTrainer, Generic[RequestBatch, ModelConfig], ABC):
    host_state: Any = None
    elapsed_samples: int = 0
    inference_batch_size: int = 256

    inference_batch_size_buckets: tuple[int, ...] = ()

    history_seq_len: int = 1000
    candidate_seq_len: int = 1400

    copy_url: str = ""
    copy_port_timeout: int = DEFAULT_COPY_PORT_TIMEOUT

    last_checkpoint_created_timestamp: float = 0.0

    grpc_port: int = 9988
    metrics_port: int = 9090
    readiness_port: int | None = None
    max_inflight_requests: int = 4096

    sid_endpoint: str | None = None

    channel_size: int = 2048
    enqueue_timeout_ms: int = 1000
    queue_max_staleness_ms: int = 1200

    enable_deadline_admission: bool = False
    deadline_margin_ms: float = 30.0
    deadline_post_ms: float = 2.0
    deadline_fallback_budget_ms: float = 1200.0
    service_time_ewma_alpha: float = 0.2
    _service_timer: Timer | None = field(default=None, init=False)
    use_pipelining: bool = True
    embedding_gather_threads: int = 16
    use_pinned_d2h: bool = False
    pinned_d2h_num_buffers: int = 3

    log_rotate: bool = False
    log_rotate_max_bytes: int = 3 * 1024 * 1024 * 1024
    log_rotate_backup_count: int = 7

    restart_period: float | None = None
    graceful_restart_timeout: float = 30

    max_inflight_backend_requests: int = 2
    requests_in_flight: int = 0
    current_request_id: int = 0
    dequeued_batch_id: int = field(default=0, init=False)
    persistent_buffers: dict[tuple[int, int], RecsysFeaturesBatch] = field(
        default_factory=dict, init=False
    )

    _last_observed_reload_trigger: int = field(default=0, init=False)

    _reload_probe_executor: concurrent.futures.ThreadPoolExecutor | None = field(
        default=None, init=False
    )
    _reload_probe_future: concurrent.futures.Future | None = field(default=None, init=False)
    _reload_probe_started: float = field(default=0.0, init=False)
    _reload_probe_inflight: list[concurrent.futures.Future] = field(
        default_factory=list, init=False
    )

    _reload_found_no_new_checkpoint: bool = field(default=False, init=False)

    _last_loaded_shm_weights_version: int = field(default=0, init=False)
    _last_loaded_emb_table_version: int = field(default=0, init=False)

    debug_log_dir: str | None = None
    debug_log_rotation_seconds: float = 900
    debug_log_path: str | None = None
    debug_writer: pq.ParquetWriter | None = None

    h2d_state: H2DState | None = None
    h2d_state_by_bs: dict[int, H2DState] = field(default_factory=dict, init=False)
    h2d_states: dict[tuple[int, int], H2DState] = field(default_factory=dict, init=False)

    _pinned_buffers: dict[tuple, Any] = field(default_factory=dict, init=False)

    metrics_publisher: MetricsPublisher | None = None

    jax_profile: bool = False
    jax_profile_path: str | None = None
    fake_mm_embeddings: bool = False
    enable_async_response_compression: bool = False
    retrieval_dataset_types: tuple[RetrievalDataset, ...] = (RetrievalDataset.HOME,)
    request_count: int = field(default=0, init=False)
    profiling_active: bool = field(default=False, init=False)

    training_ep: int = 0

    worker_id: int = 0
    num_workers: int = 1

    @property
    def sorted_buckets(self) -> tuple[int, ...]:
        if not self.inference_batch_size_buckets:
            return (self.inference_batch_size,)
        return tuple(sorted(set(self.inference_batch_size_buckets)))

    def select_bucket(self, orig_batch_size: int) -> int:
        buckets = self.sorted_buckets
        for b in buckets:
            if b >= orig_batch_size:
                return b
        return buckets[-1]

    @property
    def is_multi_worker(self) -> bool:
        return self.num_workers > 1

    @property
    def is_leader(self) -> bool:
        return self.worker_id == 0

    @property
    def shared_emb_table_path(self) -> str | None:
        if not self.is_multi_worker:
            return None
        return "/dev/shm/emb_table.dat"

    @property
    def hotswap_emb_table_path(self) -> str | None:
        if not self.enable_hotswap or self.is_multi_worker:
            return None
        return "/dev/shm/emb_table.dat"

    @property
    def shared_mm_emb_path(self) -> str | None:
        if not self.is_multi_worker:
            return None
        return "/dev/shm/mm_embeddings"

    def _read_reload_trigger(self) -> int:
        return _read_version_file("/dev/shm/reload_trigger")

    def _bump_reload_trigger(self) -> int:
        cur = self._read_reload_trigger()
        new = cur + 1
        _write_version_file("/dev/shm/reload_trigger", new)
        return new

    def _broadcast_reload_trigger(self) -> None:
        if not self.is_multi_worker:
            return
        self._last_observed_reload_trigger = self._bump_reload_trigger()
        logger.info(
            f"Worker {self.worker_id}: ReloadModel RPC received, "
            f"bumped reload trigger to {self._last_observed_reload_trigger}"
        )

    def _observe_peer_reload_trigger(self) -> bool:
        current = self._read_reload_trigger()
        if current <= self._last_observed_reload_trigger:
            return False
        self._last_observed_reload_trigger = current
        logger.info(
            f"Worker {self.worker_id}: observed peer reload trigger {current}, joining reload"
        )
        return True

    def _write_reload_done(self) -> None:
        _write_version_file(f"/dev/shm/reload_done_w{self.worker_id}", self._read_reload_trigger())

    _hotswap_setup_signal: str = "/dev/shm/hotswap_setup.done"
    _hotswap_standby_version_file: str = "/dev/shm/hotswap_standby.version"
    _hotswap_standby_meta_file: str = "/dev/shm/hotswap_standby.meta.json"

    numa_membind_all_nodes: bool = True

    enable_hotswap: bool = False
    hotswap_download_rate_limit_gbps: float | None = None
    hotswap_download_max_concurrent: int | None = None

    hotswap_stage_rate_limit_gbps: float | None = None
    hotswap_stage_chunk_mib: int = 32

    _emb_table_slots: list[np.ndarray] = field(default_factory=list, init=False)
    _active_emb_slot: int = field(default=0, init=False)
    _standby_emb_prefaulted: bool = field(default=False, init=False)
    _host_state_slots: list[Any] = field(default_factory=list, init=False)
    _active_host_slot: int = field(default=0, init=False)
    _reload_requested: threading.Event = field(default_factory=threading.Event, init=False)
    _swap_ready: threading.Event = field(default_factory=threading.Event, init=False)
    _swap_complete: threading.Event = field(default_factory=threading.Event, init=False)
    _hotswap_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _bg_loader_thread: threading.Thread | None = field(default=None, init=False)
    _pending_device_params: Any = field(default=None, init=False)
    _pending_post_embeddings: Any = field(default=None, init=False)
    _pending_elapsed_samples: int | None = field(default=None, init=False)
    _pending_checkpoint_timestamp: float | None = field(default=None, init=False)
    _hotswap_aborted: threading.Event = field(default_factory=threading.Event, init=False)
    _reload_noop: threading.Event = field(default_factory=threading.Event, init=False)
    _pending_prefix: str | None = field(default=None, init=False)
    _pending_checksums: str | None = field(default=None, init=False)
    _pending_emb_checksum: int = field(default=0, init=False)
    _pending_pe_checksum: int | None = field(default=None, init=False)

    _is_hotswap_leader: bool = field(default=True, init=False)
    _loader_process: Any = field(default=None, init=False)
    _request_pipe: Any = field(default=None, init=False)
    _result_pipe: Any = field(default=None, init=False)
    _dense_staging_info: list[tuple[str, int, int]] = field(default_factory=list, init=False)
    _dense_staging_total: int = field(default=0, init=False)
    _dense_staging_path: str | None = field(default=None, init=False)
    _pe_staging_size: int = field(default=0, init=False)
    _pe_staging_path: str | None = field(default=None, init=False)

    _live_swap_enabled: bool = field(default=False, init=False)
    _live_param_plan: list[tuple[str, int, tuple[int, ...], Any, Any]] = field(
        default_factory=list, init=False
    )
    _live_params_treedef: Any = field(default=None, init=False)
    _gpu_param_slots: list[Any] = field(default_factory=lambda: [None, None], init=False)
    _active_gpu_param_slot: int = field(default=0, init=False)
    _live_swap_ready: threading.Event = field(default_factory=threading.Event, init=False)

    _live_pe_enabled: bool = field(default=False, init=False)
    _gpu_pe_slots: list[Any] = field(default_factory=lambda: [None, None], init=False)
    _live_pe_meta_plan: list[tuple[str, int, tuple[int, ...], Any]] = field(
        default_factory=list, init=False
    )

    _mm_client: Any = field(default=None, init=False)
    _mm_preload_start_time: float = field(default=0.0, init=False)

    _startup_process_start_epoch: float | None = field(default=None, init=False)
    _startup_init_seconds: float = field(default=0.0, init=False)
    _startup_checkpoint_load_seconds: float = field(default=0.0, init=False)
    _startup_mm_preload_wait_seconds: float = field(default=0.0, init=False)
    _startup_warmup_seconds: float = field(default=0.0, init=False)

    large_k: int = 10000

    @classmethod
    def from_trainer(
        cls,
        t: RecsysTrainer,
        inference_batch_size: int,
        history_seq_len: int,
        candidate_seq_len: int,
        copy_url: str,
        grpc_port: int,
        metrics_port: int,
        max_inflight_requests: int,
        restart_period: float | None,
        graceful_restart_timeout: float,
        readiness_port: int | None = None,
        load: str | None = None,
        max_inflight_backend_requests: int = 2,
        debug_log_dir: str | None = None,
        debug_log_rotation_seconds: float = 900,
        jax_profile: bool = False,
        jax_profile_path: str | None = None,
        retrieval_dataset_types: tuple[RetrievalDataset, ...] = (RetrievalDataset.HOME,),
        large_k: int = 10000,
    ):
        init_params = t.to_dict()
        init_params.pop("__class")

        init_params["inference_batch_size"] = inference_batch_size
        init_params["history_seq_len"] = history_seq_len
        init_params["candidate_seq_len"] = candidate_seq_len
        init_params["copy_url"] = copy_url
        init_params["grpc_port"] = grpc_port
        init_params["metrics_port"] = metrics_port
        init_params["readiness_port"] = readiness_port
        init_params["max_inflight_requests"] = max_inflight_requests
        init_params["max_inflight_backend_requests"] = max_inflight_backend_requests
        init_params["restart_period"] = restart_period
        init_params["graceful_restart_timeout"] = graceful_restart_timeout
        init_params["debug_log_dir"] = debug_log_dir
        init_params["debug_log_rotation_seconds"] = debug_log_rotation_seconds
        if load is not None:
            init_params["load"] = load
        init_params["jax_profile"] = jax_profile
        init_params["jax_profile_path"] = jax_profile_path if jax_profile_path else "tensorboard"
        init_params["large_k"] = large_k
        init_params["retrieval_dataset_types"] = tuple(retrieval_dataset_types)
        return cls.from_dict(init_params, ensure_class=cls)

    @property
    def padded_input_vocab_size(self) -> int:
        assert isinstance(self.dataset, PhoenixDataset)
        raw = self.dataset.input_vocab_size
        ep = self.training_ep
        if ep <= 0:
            return raw
        return -(-raw // ep) * ep

    @property
    def padded_max_posts(self) -> int:
        if not isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return 0
        raw = self.model_config.candidate_tower_config.max_posts
        ep = self.training_ep
        if ep <= 0:
            return raw
        return -(-raw // ep) * ep

    @property
    def active_emb_table(self) -> np.ndarray:
        if self._emb_table_slots:
            return self._emb_table_slots[self._active_emb_slot]
        assert self.h2d_state is not None
        return self.h2d_state.emb_table

    @property
    def standby_emb_table(self) -> np.ndarray:
        assert self._emb_table_slots, "Hot-swap buffers not initialised"
        return self._emb_table_slots[1 - self._active_emb_slot]

    @property
    def _download_rate_limit_bytes_per_sec(self) -> int | None:
        if self.hotswap_download_rate_limit_gbps is None:
            return None
        return int(self.hotswap_download_rate_limit_gbps * (1 << 30))

    @property
    def _stage_rate_limit_bytes_per_sec(self) -> int | None:
        if self.hotswap_stage_rate_limit_gbps is None:
            return None
        rate = int(self.hotswap_stage_rate_limit_gbps * (1 << 30))
        return rate if rate > 0 else None

    def _log_gpu_memory_stats(self, label: str) -> None:
        try:
            stats = jax.local_devices()[0].memory_stats()
            if stats is None:
                return
            gib = 1 << 30
            logger.info(
                "[hotswap] GPU memory %s: "
                "bytes_in_use=%.2f GiB, peak_bytes_in_use=%.2f GiB, "
                "bytes_limit=%.2f GiB, free_in_pool=%.2f GiB",
                label,
                stats.get("bytes_in_use", 0) / gib,
                stats.get("peak_bytes_in_use", 0) / gib,
                stats.get("bytes_limit", 0) / gib,
                (stats.get("bytes_limit", 0) - stats.get("bytes_in_use", 0)) / gib,
            )
        except Exception as e:
            logger.warning("[hotswap] Could not read GPU memory stats: %s", e)

    def _log_param_sizes(self) -> None:
        assert isinstance(self.state, RecsysInferenceState)
        leaves = jax.tree.leaves(self.state.params)
        total = sum(getattr(x, "nbytes", 0) for x in leaves)
        logger.info(
            "[hotswap] Dense model params: %d tensors, %.2f GiB total on GPU",
            len(leaves),
            total / (1 << 30),
        )

    def _init_hotswap_buffers(self) -> None:
        assert self.h2d_state is not None
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        self._log_gpu_memory_stats("before hotswap init")
        self._log_param_sizes()

        shape = self.h2d_state.emb_table.shape
        assert len(shape) == 2, f"emb_table must be 2-D, got shape={shape}"
        emb_table_shape: tuple[int, int] = (shape[0], shape[1])
        logger.info("[hotswap] Initialising double-buffers: emb_table shape=%s", emb_table_shape)

        is_mw = self.is_multi_worker
        is_leader = not is_mw or self.worker_id == 0

        if is_leader:
            logger.info("[hotswap] Leader ready, files in /dev/shm/")

            if isinstance(self.h2d_state.emb_table, np.memmap):
                slot0 = self.h2d_state.emb_table
                logger.info(
                    "[hotswap] Reusing existing emb_table memmap as slot 0 "
                    "(%.2f GiB, no copy needed)",
                    slot0.nbytes / (1 << 30),
                )
            else:
                slot0 = create_memmap_emb_table(
                    "/dev/shm/emb_table.dat",
                    emb_table_shape,
                    self.model_config.embedding_dtype,
                )
                logger.warning(
                    "[hotswap] emb_table is a heap array (expected a memmap via "
                    "hotswap_emb_table_path); copying %.2f GiB heap -> memmap slot 0, "
                    "this can take minutes...",
                    slot0.nbytes / (1 << 30),
                )
                t_copy = time.time()
                parallel_copyto(slot0, self.h2d_state.emb_table)
                logger.info("[hotswap] emb_table copy complete in %.1fs", time.time() - t_copy)
                old_heap_emb = self.h2d_state.emb_table
                self.h2d_state = self.h2d_state._replace(emb_table=slot0)
                for _bs in list(self.h2d_state_by_bs):
                    self.h2d_state_by_bs[_bs] = self.h2d_state_by_bs[_bs]._replace(emb_table=slot0)
                for rid in list(self.h2d_states):
                    self.h2d_states[rid] = self.h2d_states[rid]._replace(emb_table=slot0)
                del old_heap_emb

            slot1 = create_memmap_emb_table(
                "/dev/shm/emb_table_standby.dat",
                emb_table_shape,
                self.model_config.embedding_dtype,
                prefault=False,
            )
            self._standby_emb_prefaulted = False
            logger.info(
                "[hotswap] Standby emb_table created sparsely (%.2f GiB); page "
                "prefault deferred to the coordinator thread",
                slot1.nbytes / (1 << 30),
            )
        else:
            logger.info(
                "[hotswap] Worker %d: waiting for leader to finish hotswap setup...",
                self.worker_id,
            )
            while not os.path.exists(self._hotswap_setup_signal):
                time.sleep(1)
            logger.info(
                "[hotswap] Worker %d: leader signaled done, opening shared emb_table memmaps",
                self.worker_id,
            )
            if isinstance(self.h2d_state.emb_table, np.memmap):
                slot0 = self.h2d_state.emb_table
            else:
                slot0 = open_existing_memmap_emb_table(
                    self.shared_emb_table_path or "/dev/shm/emb_table.dat",
                    emb_table_shape,
                    self.model_config.embedding_dtype,
                )
                old_heap_emb = self.h2d_state.emb_table
                self.h2d_state = self.h2d_state._replace(emb_table=slot0)
                for _bs in list(self.h2d_state_by_bs):
                    self.h2d_state_by_bs[_bs] = self.h2d_state_by_bs[_bs]._replace(emb_table=slot0)
                for rid in list(self.h2d_states):
                    self.h2d_states[rid] = self.h2d_states[rid]._replace(emb_table=slot0)
                del old_heap_emb
            slot1 = open_existing_memmap_emb_table(
                "/dev/shm/emb_table_standby.dat",
                emb_table_shape,
                self.model_config.embedding_dtype,
            )

        self._emb_table_slots = [slot0, slot1]
        self._active_emb_slot = 0
        self._is_hotswap_leader = is_leader
        logger.info(
            "[hotswap] emb_table memmap slots ready (role=%s): %.2f GiB each",
            "leader" if is_leader else "follower",
            self._emb_table_slots[0].nbytes / (1 << 30),
        )

        if self.host_state is None:
            self.host_state = jax.device_put(self.state, self.host_sharding)
        self._host_state_slots = [
            self.host_state,
            jax.device_put(self.state, self.host_sharding),
        ]
        self._active_host_slot = 0

        standby_host_state = self._host_state_slots[1]
        host_state_dict = tree_to_dict(unwrap_tree(standby_host_state).purge_opt_state())
        self._dense_staging_info = []
        offset = 0
        for key, value in host_state_dict.items():
            if value is None or not self._is_fully_replicated(value):
                continue
            nbytes = _unsafe_jax2np(value.addressable_shards[0].data).nbytes
            self._dense_staging_info.append((key, offset, nbytes))
            offset += nbytes
        self._dense_staging_total = offset
        self._dense_staging_path = _SHM_WEIGHTS_PATH
        if offset > 0 and is_leader:
            np.memmap(self._dense_staging_path, dtype=np.uint8, mode="w+", shape=(offset,))
        logger.info(
            "[hotswap] Dense weights buffer (%s): %d tensors, %.2f GiB total (role=%s)",
            self._dense_staging_path,
            len(self._dense_staging_info),
            offset / (1 << 30),
            "leader" if is_leader else "follower",
        )

        pe_key = "post_embeddings.embeddings"
        self._pe_staging_size = 0
        self._pe_staging_path = "/dev/shm/post_embeddings.dat"
        if pe_key in host_state_dict and host_state_dict[pe_key] is not None:
            pe_value = host_state_dict[pe_key]
            if not self._is_fully_replicated(pe_value):
                self._pe_staging_size = int(np.prod(pe_value.shape)) * pe_value.dtype.itemsize
                if is_leader:
                    np.memmap(
                        self._pe_staging_path,
                        dtype=np.uint8,
                        mode="w+",
                        shape=(self._pe_staging_size,),
                    )
                logger.info(
                    "[hotswap] Post-embeddings buffer (%s): %.2f GiB (role=%s)",
                    self._pe_staging_path,
                    self._pe_staging_size / (1 << 30),
                    "leader" if is_leader else "follower",
                )

        if is_leader:
            self._spawn_loader_subprocess(emb_table_shape, pe_key)
            if is_mw:
                with open(self._hotswap_setup_signal, "w") as f:
                    f.write(str(emb_table_shape))
                logger.info(
                    "[hotswap] Leader signaled setup-done at %s",
                    self._hotswap_setup_signal,
                )

        self._gpu_dual_copy: bool = False
        try:
            self._log_gpu_memory_stats("before dual-copy probe")
            probe_params = jax.device_put(self.state.params, self.state_sharding.params)
            param_bytes = sum(getattr(x, "nbytes", 0) for x in jax.tree.leaves(probe_params))

            probe_pe = None
            pe_bytes = 0
            assert isinstance(self.state, RecsysInferenceState)
            if (
                isinstance(self.model_config, RecsysTwoTowerModelConfig)
                and self.state.post_embeddings is not None
            ):
                pe_sharding = self.state_sharding.post_embeddings
                probe_pe = jax.device_put(self.state.post_embeddings, pe_sharding)
                pe_bytes = sum(getattr(x, "nbytes", 0) for x in jax.tree.leaves(probe_pe))

            logger.info(
                "[hotswap] GPU dual-copy probe SUCCEEDED: params=%.2f GiB, "
                "post_embeddings=%.2f GiB",
                param_bytes / (1 << 30),
                pe_bytes / (1 << 30),
            )
            self._gpu_dual_copy = True
            self._pending_device_params = probe_params
            self._pending_post_embeddings = probe_pe
            self._log_gpu_memory_stats("after dual-copy probe")
        except Exception as e:
            logger.info("[hotswap] GPU dual-copy probe failed (%s), using H2D fallback", e)
            self._gpu_dual_copy = False
            self._pending_device_params = None
            self._pending_post_embeddings = None

        logger.info("[hotswap] Double-buffers ready (gpu_dual_copy=%s)", self._gpu_dual_copy)

        self._maybe_init_live_dense_swap(self._host_state_slots[1])

    def _prefault_standby_emb_slot(self) -> None:
        if self._standby_emb_prefaulted:
            return
        try:
            slot1 = self._emb_table_slots[1]
            secs = prefault_memmap(slot1)
            logger.info("[hotswap] Pre-faulted standby emb_table pages in %.1fs (background)", secs)
        except Exception as e:
            logger.warning(
                "[hotswap] Standby emb_table prefault failed (%s); the loader "
                "subprocess will fault pages during the first download",
                e,
            )
        self._standby_emb_prefaulted = True

    def _maybe_init_live_dense_swap(self, standby_host_state: Any) -> None:
        if self.is_multi_worker:
            logger.info("[hotswap-live] multi-worker -> staying on drained finalize path")
            return
        if not self._gpu_dual_copy:
            logger.info("[hotswap-live] no GPU dual-copy headroom -> drained finalize path")
            return
        is_two_tower = isinstance(self.model_config, RecsysTwoTowerModelConfig)
        if not isinstance(self.model_config, RecsysAggregatedModelConfig) and not is_two_tower:
            logger.info(
                "[hotswap-live] unsupported model config (%s) -> drained finalize path",
                type(self.model_config).__name__,
            )
            return
        if is_two_tower and not self._live_swap_two_tower_supported():
            return
        try:
            purged = unwrap_tree(standby_host_state).purge_opt_state()
            full_leaves = jax.tree_util.tree_flatten_with_path(purged, is_leaf=lambda x: x is None)[
                0
            ]
            params_leaf_ids = {
                id(leaf) for leaf in jax.tree.leaves(unwrap_tree(standby_host_state.params))
            }
            staging_by_key = {k: (o, n) for k, o, n in self._dense_staging_info}
            params_entries: list[tuple[str, int, int]] = []
            for path, value in full_leaves:
                if value is None or not self._is_fully_replicated(value):
                    continue
                if id(value) in params_leaf_ids:
                    key = keystr(path)
                    if key not in staging_by_key:
                        logger.warning(
                            "[hotswap-live] param key %s missing from staging layout; "
                            "staying on drained finalize path",
                            key,
                        )
                        return
                    off, nbytes = staging_by_key[key]
                    params_entries.append((key, off, nbytes))

            wrapped_leaves, treedef = jax.tree_util.tree_flatten(standby_host_state.params)
            sharding_leaves = jax.tree_util.tree_leaves(self.state_sharding.params)
            if not (len(wrapped_leaves) == len(params_entries) == len(sharding_leaves)):
                logger.warning(
                    "[hotswap-live] leaf-count mismatch (wrapped=%d, staged=%d, shardings=%d); "
                    "staying on drained finalize path",
                    len(wrapped_leaves),
                    len(params_entries),
                    len(sharding_leaves),
                )
                return

            plan: list[tuple[str, int, tuple[int, ...], Any, Any]] = []
            total_bytes = 0
            for x_leaf, (key, off, nbytes), sharding in zip(
                wrapped_leaves, params_entries, sharding_leaves
            ):
                dtype = np.dtype(x_leaf.dtype)
                expected = int(np.prod(x_leaf.shape)) * dtype.itemsize
                if expected != nbytes:
                    logger.warning(
                        "[hotswap-live] byte mismatch for %s (leaf=%d, staged=%d); "
                        "staying on drained finalize path",
                        key,
                        expected,
                        nbytes,
                    )
                    return
                plan.append((key, off, tuple(x_leaf.shape), dtype, sharding))
                total_bytes += nbytes

            pe_meta_plan: list[tuple[str, int, tuple[int, ...], Any]] = []
            if is_two_tower:
                pe_meta_plan_or_none = self._build_live_pe_meta_plan(
                    tree_to_dict(purged), staging_by_key
                )
                if pe_meta_plan_or_none is None:
                    return
                pe_meta_plan = pe_meta_plan_or_none

            self._live_param_plan = plan
            self._live_params_treedef = treedef
            self._live_swap_enabled = True
            self._gpu_param_slots = [self.state.params, self._pending_device_params]
            self._active_gpu_param_slot = 0
            self._pending_device_params = None
            if is_two_tower:
                self._gpu_pe_slots = [self.state.post_embeddings, self._pending_post_embeddings]
                self._live_pe_meta_plan = pe_meta_plan
                self._live_pe_enabled = True
            self._pending_post_embeddings = None
            logger.info(
                "[hotswap-live] Live dense swap ENABLED (%s): %d param leaves, %.2f GiB, "
                "serving-thread swap is a pointer flip (no drain).",
                "two-tower params+post_embeddings" if is_two_tower else "ranking params",
                len(plan),
                total_bytes / (1 << 30),
            )
        except Exception as e:
            logger.warning(
                "[hotswap-live] plan init failed (%s); staying on drained finalize path", e
            )
            self._live_swap_enabled = False
            self._live_param_plan = []
            self._live_params_treedef = None
            self._gpu_param_slots = [None, None]
            self._live_pe_enabled = False
            self._live_pe_meta_plan = []
            self._gpu_pe_slots = [None, None]

    @property
    def _live_swap_blocked_by_candidate_filters(self) -> bool:
        return False

    def _live_swap_two_tower_supported(self) -> bool:
        assert isinstance(self.state, RecsysInferenceState)
        if self.state.post_embeddings is None or self._pending_post_embeddings is None:
            logger.info(
                "[hotswap-live] two-tower without post_embeddings payload -> drained finalize path"
            )
            return False
        if self._pe_staging_size <= 0:
            logger.info("[hotswap-live] no post_embeddings staging buffer -> drained finalize path")
            return False
        if self._live_swap_blocked_by_candidate_filters:
            logger.info(
                "[hotswap-live] serving-time candidate filters enabled -> drained finalize path"
            )
            return False
        return True

    def _build_live_pe_meta_plan(
        self,
        host_state_dict: dict[str, Any],
        staging_by_key: dict[str, tuple[int, int]],
    ) -> list[tuple[str, int, tuple[int, ...], Any]] | None:
        pe_meta_plan: list[tuple[str, int, tuple[int, ...], Any]] = []
        for meta_key in (
            "post_embeddings.post_ids",
            "post_embeddings.author_ids",
            "post_embeddings.dataset_types",
        ):
            value = host_state_dict.get(meta_key)
            if value is None or meta_key not in staging_by_key:
                logger.warning(
                    "[hotswap-live] %s missing from staging layout; "
                    "staying on drained finalize path",
                    meta_key,
                )
                return None
            dtype = np.dtype(value.dtype)
            expected = int(np.prod(value.shape)) * dtype.itemsize
            off, nbytes = staging_by_key[meta_key]
            if expected != nbytes:
                logger.warning(
                    "[hotswap-live] byte mismatch for %s (leaf=%d, staged=%d); "
                    "staying on drained finalize path",
                    meta_key,
                    expected,
                    nbytes,
                )
                return None
            pe_meta_plan.append((meta_key, off, tuple(value.shape), dtype))

        pe_value = host_state_dict.get("post_embeddings.embeddings")
        if pe_value is None:
            logger.warning(
                "[hotswap-live] post_embeddings.embeddings missing from host state; "
                "staying on drained finalize path"
            )
            return None
        pe_nbytes = int(np.prod(pe_value.shape)) * np.dtype(pe_value.dtype).itemsize
        if pe_nbytes != self._pe_staging_size:
            logger.warning(
                "[hotswap-live] post_embeddings byte mismatch (leaf=%d, staged=%d); "
                "staying on drained finalize path",
                pe_nbytes,
                self._pe_staging_size,
            )
            return None
        return pe_meta_plan

    def _stage_live_post_embeddings(self) -> bool:
        try:
            assert self._dense_staging_path is not None and self._pe_staging_path is not None
            standby_host_state = self._host_state_slots[1 - self._active_host_slot]
            host_state_dict = tree_to_dict(unwrap_tree(standby_host_state).purge_opt_state())
            t0 = time.time()

            pe_host = host_state_dict["post_embeddings.embeddings"]
            pe_staging = np.memmap(
                self._pe_staging_path,
                dtype=np.uint8,
                mode="r",
                shape=(self._pe_staging_size,),
            )
            self._scatter_contiguous_to_shards(
                pe_host,
                np.ndarray(self._pe_staging_size, dtype=np.uint8, buffer=pe_staging),
            )
            del pe_staging

            dense = np.memmap(
                self._dense_staging_path,
                dtype=np.uint8,
                mode="r",
                shape=(self._dense_staging_total,),
            )
            staged_meta: dict[str, np.ndarray] = {}
            for key, off, shape, dtype in self._live_pe_meta_plan:
                value = host_state_dict[key]
                nbytes = int(np.prod(shape)) * dtype.itemsize
                source = np.ndarray(shape=nbytes, dtype=np.uint8, buffer=dense, offset=off)
                shard_data = _unsafe_jax2np(value.addressable_shards[0].data)
                np.copyto(np.ndarray(shape=nbytes, dtype=np.uint8, buffer=shard_data), source)
                self._broadcast_tensor_to_all_shards(
                    value, np.ndarray(shape=nbytes, dtype=np.uint8, buffer=shard_data)
                )
                owned = np.empty(shape, dtype=dtype)
                np.copyto(np.ndarray(shape=nbytes, dtype=np.uint8, buffer=owned), source)
                staged_meta[key] = owned
            del dense
            host_elapsed = time.time() - t0

            t1 = time.time()
            standby = 1 - self._active_gpu_param_slot
            self._gpu_pe_slots[standby] = None
            with self.mesh:
                new_pe = jax.device_put(
                    standby_host_state.post_embeddings,
                    self.state_sharding.post_embeddings,
                )
                jax.block_until_ready(new_pe)
            self._gpu_pe_slots[standby] = new_pe

            self._prepare_live_swap_derived_state(staged_meta)

            logger.info(
                "[hotswap-live] Staged post_embeddings into slot %d "
                "(host_copy=%.2fs, device_put=%.2fs)",
                standby,
                host_elapsed,
                time.time() - t1,
            )
            self._log_gpu_memory_stats("after live PE staging")
            return True
        except Exception as e:
            self._abort_hotswap_cycle("live_stage", f"post_embeddings staging failed: {e}")
            return False

    def _prepare_live_swap_derived_state(self, staged_meta: dict[str, np.ndarray]) -> None:
        pass

    def _commit_live_swap_derived_state(self) -> None:
        pass

    @property
    def _live_swap_requires_pipeline_flush(self) -> bool:
        return False

    @property
    def _post_hotswap_warmup_needed(self) -> bool:
        return False

    def _stage_live_params(self) -> bool:
        try:
            assert self._dense_staging_path is not None
            standby = 1 - self._active_gpu_param_slot
            self._gpu_param_slots[standby] = None
            dense = np.memmap(
                self._dense_staging_path,
                dtype=np.uint8,
                mode="r",
                shape=(self._dense_staging_total,),
            )
            rate = self._stage_rate_limit_bytes_per_sec
            chunk_bytes = max(1, self.hotswap_stage_chunk_mib) * (1 << 20)
            t0 = time.time()
            with self.mesh:
                gpu_leaves = [
                    self._device_put_leaf_paced(
                        dense, off, shape, dtype, sharding, chunk_bytes, rate
                    )
                    for (_key, off, shape, dtype, sharding) in self._live_param_plan
                ]
                new_params = jax.tree_util.tree_unflatten(self._live_params_treedef, gpu_leaves)
                jax.block_until_ready(new_params)
            del dense

            self._gpu_param_slots[standby] = new_params
            logger.info(
                "[hotswap-live] Staged %d param leaves into slot %d in %.2fs "
                "(stage_rate_cap=%s GiB/s)",
                len(gpu_leaves),
                standby,
                time.time() - t0,
                self.hotswap_stage_rate_limit_gbps,
            )
            self._log_gpu_memory_stats("after live staging")
            return True
        except Exception as e:
            self._abort_hotswap_cycle("live_stage", f"staging failed: {e}")
            return False

    def _device_put_leaf_paced(
        self,
        dense: np.ndarray,
        off: int,
        shape: tuple[int, ...],
        dtype: Any,
        sharding: Any,
        chunk_bytes: int,
        rate: int | None,
    ) -> jax.Array:
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if rate is None or nbytes <= chunk_bytes or len(shape) == 0 or shape[0] <= 1:
            dev = jax.device_put(np.ndarray(shape, dtype=dtype, buffer=dense, offset=off), sharding)
            if rate is not None:
                dev.block_until_ready()
                time.sleep(nbytes / rate)
            return dev
        full = np.ndarray(shape, dtype=dtype, buffer=dense, offset=off)
        rows = shape[0]
        row_bytes = max(1, nbytes // rows)
        rows_per_chunk = max(1, chunk_bytes // row_bytes)
        blocks = []
        for r0 in range(0, rows, rows_per_chunk):
            r1 = min(rows, r0 + rows_per_chunk)
            block = jax.device_put(full[r0:r1], sharding)
            block.block_until_ready()
            time.sleep((r1 - r0) * row_bytes / rate)
            blocks.append(block)
        return jnp.concatenate(blocks, axis=0)

    def _apply_live_swap(self, server: ServerProtocol[RequestBatch]) -> None:
        try:
            self._apply_live_swap_inner(server)
        except Exception:
            self._live_swap_ready.clear()
            self._swap_complete.set()
            if self.metrics_publisher is not None:
                self.metrics_publisher.hotswap_failure_total.labels(stage="live_swap").inc()
            raise

    def _apply_live_swap_inner(self, server: ServerProtocol[RequestBatch]) -> None:
        assert self.metrics_publisher is not None
        assert isinstance(self.state, RecsysInferenceState)
        standby = 1 - self._active_gpu_param_slot
        assert self._gpu_param_slots[standby] is not None, "standby GPU param slot not staged"
        if self._live_pe_enabled:
            assert self._gpu_pe_slots[standby] is not None, "standby GPU PE slot not staged"
        swap_t = time.time()
        self._active_emb_slot = 1 - self._active_emb_slot
        self.metrics_publisher.hotswap_emb_table_swap_timestamp.set(swap_t)
        self._active_gpu_param_slot = standby
        replacements: dict[str, Any] = {"params": self._gpu_param_slots[standby]}
        if self._live_pe_enabled:
            replacements["post_embeddings"] = self._gpu_pe_slots[standby]
        self.state = self.state._replace(**replacements)
        self.metrics_publisher.hotswap_dense_weights_swap_timestamp.set(time.time())

        if self._pending_elapsed_samples is not None:
            self.elapsed_samples = self._pending_elapsed_samples
            self.metrics_publisher.model_checkpoint_elapsed_samples.set(self.elapsed_samples)
        if self._pending_checkpoint_timestamp is not None:
            ts = self._pending_checkpoint_timestamp
            self.last_checkpoint_created_timestamp = ts
            self.metrics_publisher.model_checkpoint_timestamp.set(ts)
            self._publish_checkpoint_timestamp(server, ts)
        self.metrics_publisher.model_checkpoint_load_success_timestamp.set(int(time.time()))
        self.metrics_publisher.hotswap_active_slot.set(self._active_emb_slot)
        self.metrics_publisher.hotswap_success_total.inc()
        self.metrics_publisher.hotswap_total_swap_seconds.observe(time.time() - swap_t)
        self.metrics_publisher.checkpoint_reload_method.labels(method="hotswap_live").inc()
        self.metrics_publisher.checkpoint_reload_success_total.labels(method="hotswap_live").inc()

        self._commit_live_swap_derived_state()

        self._live_swap_ready.clear()
        self._pending_prefix = None
        self._pending_checksums = None
        server.reset_reload_request()
        server.reset_dequeue_timer()
        self._swap_complete.set()
        logger.info(
            "[hotswap-live] Live swap applied in %.1fms (elapsed_samples=%d, active_slot=%d)",
            (time.time() - swap_t) * 1000.0,
            self.elapsed_samples,
            self._active_emb_slot,
        )

    def _spawn_loader_subprocess(
        self,
        emb_table_shape: tuple[int, int],
        pe_key: str,
    ) -> None:
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        match self.model_config.embedding_dtype.dtype.name:
            case "bfloat16":
                emb_dtype_name = "uint16"
            case "float32":
                emb_dtype_name = "uint32"
            case _:
                raise ValueError(
                    f"Unsupported embedding dtype: {self.model_config.embedding_dtype}"
                )

        sub_recv, main_send = multiprocessing.Pipe(duplex=False)
        main_recv, sub_send = multiprocessing.Pipe(duplex=False)
        self._request_pipe = main_send
        self._result_pipe = main_recv

        assert "/dev/shm" != None
        ctx = multiprocessing.get_context("spawn")
        self._loader_process = ctx.Process(
            target=_loader_subprocess_main,
            args=(
                "/dev/shm",
                self.copy_url,
                emb_table_shape,
                emb_dtype_name,
                self._dense_staging_info,
                self._dense_staging_total,
                self._pe_staging_size,
                pe_key,
                sub_recv,
                sub_send,
                self._download_rate_limit_bytes_per_sec,
                self.hotswap_download_max_concurrent,
            ),
            daemon=True,
            name="hotswap-loader-subprocess",
        )
        self._loader_process.start()
        rate_log = (
            f", rate_limit={self.hotswap_download_rate_limit_gbps} GiB/s"
            if self.hotswap_download_rate_limit_gbps is not None
            else ", rate_limit=unlimited"
        )
        logger.info(
            "[hotswap] Loader subprocess spawned (pid=%d%s)",
            self._loader_process.pid,
            rate_log,
        )

    def _respawn_loader_subprocess(self) -> None:
        if self._loader_process is not None:
            try:
                self._loader_process.kill()
                self._loader_process.join(timeout=5)
            except Exception:
                pass
        for p in (self._request_pipe, self._result_pipe):
            if p is not None:
                try:
                    p.close()
                except Exception:
                    pass

        assert self.h2d_state is not None
        shape = self.h2d_state.emb_table.shape
        emb_table_shape: tuple[int, int] = (shape[0], shape[1])
        pe_key = "post_embeddings.embeddings"
        self._spawn_loader_subprocess(emb_table_shape, pe_key)

    def _abort_hotswap_cycle(self, stage: str, reason: str) -> None:
        logger.warning(
            "[hotswap] Reload cycle aborted at stage=%s (%s); keeping current "
            "snapshot (elapsed_samples=%d), no legacy fallback",
            stage,
            reason,
            self.elapsed_samples,
        )
        if self.metrics_publisher is not None:
            self.metrics_publisher.hotswap_failure_total.labels(stage=stage).inc()
        self._pending_prefix = None
        self._pending_checksums = None
        self._pending_pe_checksum = None
        self._pending_elapsed_samples = None
        self._pending_checkpoint_timestamp = None
        if self.is_multi_worker:
            self._write_reload_done()
        self._hotswap_aborted.set()

    def _coordinator_loop(self) -> None:
        logger.info("[hotswap] Coordinator thread started (subprocess loader).")

        self._prefault_standby_emb_slot()

        while not self._hotswap_stop.is_set():
            if self._loader_process is not None and not self._loader_process.is_alive():
                logger.warning(
                    "[hotswap] Loader subprocess died (exit code %s), respawning",
                    self._loader_process.exitcode,
                )
                if self.metrics_publisher is not None:
                    self.metrics_publisher.hotswap_failure_total.labels(
                        stage="subprocess_crash"
                    ).inc()
                self._respawn_loader_subprocess()

            triggered = self._reload_requested.wait(timeout=5.0)
            if self._hotswap_stop.is_set():
                break
            if not triggered:
                continue

            self._reload_requested.clear()
            logger.info("[hotswap] Coordinator: reload triggered, sending to subprocess.")
            bg_start = time.time()

            try:
                assert self._request_pipe is not None
                assert self._result_pipe is not None
                verify_in_subprocess = (
                    self.checkpoint_config.verify_checksums and self._live_swap_enabled
                )
                self._request_pipe.send(
                    (self.elapsed_samples, self._active_emb_slot, verify_in_subprocess)
                )
                result = self._result_pipe.recv()
            except (BrokenPipeError, EOFError, OSError) as e:
                self._abort_hotswap_cycle("subprocess_crash", f"subprocess pipe broken: {e}")
                self._respawn_loader_subprocess()
                continue

            bg_elapsed = time.time() - bg_start

            if self.metrics_publisher is not None:
                self.metrics_publisher.hotswap_background_load_seconds.observe(bg_elapsed)
                self.metrics_publisher.checkpoint_reload_step_seconds.labels(
                    step="hotswap_background_load"
                ).observe(bg_elapsed)

            if result[0] == "error":
                error_msg = result[1]
                if "no greater than old prefix" in error_msg:
                    logger.info("[hotswap] Subprocess: no new checkpoint (%.2fs)", bg_elapsed)
                    self._reload_noop.set()
                    if self.is_multi_worker:
                        self._write_reload_done()
                    if self._hotswap_standby_meta_file is not None:
                        self._publish_standby_metadata({"status": "noop"})
                else:
                    self._abort_hotswap_cycle(
                        "background_load",
                        f"subprocess download failed after {bg_elapsed:.2f}s: {error_msg}",
                    )
                    if self._hotswap_standby_meta_file is not None:
                        self._publish_standby_metadata({"status": "error", "error": error_msg})
                continue

            _, prefix, checksums, created_ts, emb_checksum, pe_checksum = result
            self._pending_prefix = prefix
            self._pending_checksums = checksums
            self._pending_emb_checksum = emb_checksum
            self._pending_pe_checksum = pe_checksum
            self._pending_elapsed_samples = int(prefix.split("_")[2].split("/")[0])
            self._pending_checkpoint_timestamp = created_ts if created_ts > 0 else time.time()

            logger.info(
                "[hotswap] Subprocess download complete (%.2fs), prefix=%s",
                bg_elapsed,
                prefix,
            )

            if self._live_swap_enabled:
                staged = self._stage_live_params()
                if staged and self._live_pe_enabled:
                    staged = self._stage_live_post_embeddings()
                if staged:
                    t_stage = time.time() - bg_start
                    if self.metrics_publisher is not None:
                        self.metrics_publisher.checkpoint_reload_step_seconds.labels(
                            step="hotswap_live_stage"
                        ).observe(t_stage)
                    logger.info("[hotswap-live] Standby GPU state staged (%.2fs)", t_stage)
                    self._live_swap_ready.set()
                    self._swap_complete.wait()
                    self._swap_complete.clear()
                continue

            if self._hotswap_standby_meta_file is not None:
                self._publish_standby_metadata(
                    {
                        "status": "success",
                        "prefix": prefix,
                        "checksums": checksums,
                        "emb_checksum": emb_checksum,
                        "pe_checksum": pe_checksum,
                        "created_ts": created_ts,
                        "elapsed_samples": self._pending_elapsed_samples,
                    }
                )

            self._swap_ready.set()
            self._swap_complete.wait()
            self._swap_complete.clear()

        logger.info("[hotswap] Coordinator thread exiting.")

    def _publish_standby_metadata(self, payload: dict) -> None:
        import json as _json

        assert self._hotswap_standby_meta_file is not None
        assert self._hotswap_standby_version_file is not None
        tmp_path = self._hotswap_standby_meta_file + ".tmp"
        with open(tmp_path, "w") as f:
            _json.dump(payload, f)
        os.rename(tmp_path, self._hotswap_standby_meta_file)
        try:
            with open(self._hotswap_standby_version_file) as f:
                cur = int(f.read().strip() or "0")
        except (FileNotFoundError, ValueError):
            cur = 0
        with open(self._hotswap_standby_version_file + ".tmp", "w") as f:
            f.write(str(cur + 1))
        os.rename(
            self._hotswap_standby_version_file + ".tmp",
            self._hotswap_standby_version_file,
        )
        logger.info(
            "[hotswap] Leader published standby metadata version=%d (status=%s)",
            cur + 1,
            payload.get("status", "?"),
        )

    def _follower_coordinator_loop(self) -> None:
        import json as _json

        assert self._hotswap_standby_meta_file is not None
        assert self._hotswap_standby_version_file is not None
        wid = self.worker_id
        logger.info("[hotswap] Follower (worker %d) coordinator started.", wid)

        last_version = 0
        try:
            with open(self._hotswap_standby_version_file) as f:
                last_version = int(f.read().strip() or "0")
        except (FileNotFoundError, ValueError):
            last_version = 0

        while not self._hotswap_stop.is_set():
            triggered = self._reload_requested.wait(timeout=5.0)
            if self._hotswap_stop.is_set():
                break
            if not triggered:
                continue
            self._reload_requested.clear()

            t_wait = time.time()
            while not self._hotswap_stop.is_set():
                try:
                    with open(self._hotswap_standby_version_file) as f:
                        cur = int(f.read().strip() or "0")
                except (FileNotFoundError, ValueError):
                    cur = last_version
                if cur > last_version:
                    last_version = cur
                    break
                if time.time() - t_wait > 1800:
                    logger.warning(
                        "[hotswap] Worker %d: timed out waiting for leader's "
                        "standby payload after 30 min, giving up this cycle",
                        wid,
                    )
                    break
                time.sleep(0.5)
            else:
                continue

            try:
                with open(self._hotswap_standby_meta_file) as f:
                    meta = _json.load(f)
            except (FileNotFoundError, _json.JSONDecodeError) as e:
                logger.warning(
                    "[hotswap] Worker %d: failed to read leader metadata: %s",
                    wid,
                    e,
                )
                continue

            status = meta.get("status")
            if status == "success":
                self._pending_prefix = meta["prefix"]
                self._pending_checksums = meta["checksums"]
                self._pending_emb_checksum = meta.get("emb_checksum")
                self._pending_pe_checksum = meta.get("pe_checksum")
                self._pending_elapsed_samples = meta["elapsed_samples"]
                created_ts = meta.get("created_ts", 0)
                self._pending_checkpoint_timestamp = created_ts if created_ts > 0 else time.time()
                logger.info(
                    "[hotswap] Worker %d: leader standby ready (version=%d, prefix=%s)",
                    wid,
                    last_version,
                    self._pending_prefix,
                )
                self._swap_ready.set()
                self._swap_complete.wait()
                self._swap_complete.clear()
            elif status == "noop":
                logger.info("[hotswap] Worker %d: leader reported noop (no new ckpt)", wid)
                self._reload_noop.set()
                if self.is_multi_worker:
                    self._write_reload_done()
            else:
                self._abort_hotswap_cycle(
                    "leader_error",
                    f"worker {wid}: leader reported error/unknown status {status!r}",
                )

        logger.info("[hotswap] Follower coordinator (worker %d) exiting.", wid)

    def _finalize_hot_swap(
        self,
        server: ServerProtocol[RequestBatch],
    ) -> None:
        assert self.metrics_publisher is not None
        finalize_start = time.time()
        logger.info("[hotswap] Finalizing hot-swap (gpu_dual_copy=%s)...", self._gpu_dual_copy)

        try:
            standby_host_slot = 1 - self._active_host_slot
            standby_host_state = self._host_state_slots[standby_host_slot]

            t_staging = time.time()
            host_state_dict = tree_to_dict(unwrap_tree(standby_host_state).purge_opt_state())
            if self._dense_staging_path and self._dense_staging_total > 0:
                dense_staging = np.memmap(
                    self._dense_staging_path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(self._dense_staging_total,),
                )
                for key, offset, nbytes in self._dense_staging_info:
                    value = host_state_dict[key]
                    shard_data = _unsafe_jax2np(value.addressable_shards[0].data)
                    target = np.ndarray(shape=nbytes, dtype=np.uint8, buffer=shard_data)
                    source = np.ndarray(
                        shape=nbytes,
                        dtype=np.uint8,
                        buffer=dense_staging,
                        offset=offset,
                    )
                    np.copyto(target, source)
                del dense_staging

            pe_key = "post_embeddings.embeddings"
            if (
                self._pe_staging_size > 0
                and self._pe_staging_path
                and pe_key in host_state_dict
                and host_state_dict[pe_key] is not None
            ):
                pe_staging = np.memmap(
                    self._pe_staging_path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(self._pe_staging_size,),
                )
                pe_value = host_state_dict[pe_key]
                if not self._is_fully_replicated(pe_value):
                    pe_buf = np.ndarray(
                        self._pe_staging_size,
                        dtype=np.uint8,
                        buffer=pe_staging,
                    )
                    self._scatter_contiguous_to_shards(pe_value, pe_buf)
                del pe_staging

            staging_elapsed = time.time() - t_staging
            logger.info("[hotswap] Staging copy (shm -> JAX host) took %.3fs", staging_elapsed)

            t_broadcast = time.time()
            tensor_bytes: dict[str, np.ndarray] = {}
            for key, value in host_state_dict.items():
                if value is None or not self._is_fully_replicated(value):
                    continue
                arr = _unsafe_jax2np(value.addressable_shards[0].data)
                tensor_bytes[key] = np.ndarray(shape=arr.nbytes, dtype=np.uint8, buffer=arr)
            tensors = list(tensor_bytes.items())

            for key, tensor in tensors:
                self._broadcast_tensor_to_all_shards(host_state_dict[key], tensor)
            broadcast_elapsed = time.time() - t_broadcast

            t_checksums = time.time()
            if self.checkpoint_config.verify_checksums and self._pending_checksums is not None:
                saved = json.loads(self._pending_checksums)["global_checksums"]
                for key in tensor_bytes:
                    computed = adler32.adler32(host_state_dict[key]).item()
                    if key not in saved:
                        logger.warning("[hotswap] Checksum key %s not in checkpoint", key)
                    elif computed != saved[key]:
                        raise ValueError(f"[hotswap] Checksum mismatch on {key}")
                if self._pending_emb_checksum != saved.get("emb_table"):
                    raise ValueError("[hotswap] Checksum mismatch on emb_table")
                pe_key = "post_embeddings.embeddings"
                if self._pending_pe_checksum is not None and self._pending_pe_checksum != saved.get(
                    pe_key
                ):
                    raise ValueError(f"[hotswap] Checksum mismatch on {pe_key}")
                logger.info("[hotswap] All checksums verified")
            checksums_elapsed = time.time() - t_checksums

            t_device_put = time.time()
            if self._gpu_dual_copy:
                self._pending_device_params = None
                self._pending_post_embeddings = None
                try:
                    self._pending_device_params = jax.device_put(
                        standby_host_state.params
                        if hasattr(standby_host_state, "params")
                        else tree_to_dict(unwrap_tree(standby_host_state).purge_opt_state()),
                        self.state_sharding.params,
                    )
                    if (
                        isinstance(self.model_config, RecsysTwoTowerModelConfig)
                        and hasattr(standby_host_state, "post_embeddings")
                        and standby_host_state.post_embeddings is not None
                    ):
                        self._pending_post_embeddings = jax.device_put(
                            standby_host_state.post_embeddings,
                            self.state_sharding.post_embeddings,
                        )
                except Exception as e:
                    logger.warning("[hotswap] device_put failed (%s), using H2D fallback", e)
                    self._pending_device_params = None
                    self._pending_post_embeddings = None
            device_put_elapsed = time.time() - t_device_put

            t_swap = time.time()
            if self._gpu_dual_copy and self._pending_device_params is not None:
                old_params = self.state.params
                old_pe = self.state.post_embeddings
                assert isinstance(self.state, RecsysInferenceState)
                replacements: dict[str, Any] = {"params": self._pending_device_params}
                if self._pending_post_embeddings is not None:
                    replacements["post_embeddings"] = self._pending_post_embeddings
                self.state = self.state._replace(**replacements)
                self._pending_device_params = None
                self._pending_post_embeddings = None
                del old_params, old_pe
            else:
                self._host_state_slots[standby_host_slot], self.state = self.reload_state(
                    self._host_state_slots[standby_host_slot], self.state
                )
                self._active_host_slot = standby_host_slot
                self.host_state = self._host_state_slots[self._active_host_slot]

            self.metrics_publisher.hotswap_dense_weights_swap_timestamp.set(time.time())

            self._active_emb_slot = 1 - self._active_emb_slot
            self.metrics_publisher.hotswap_emb_table_swap_timestamp.set(time.time())
            swap_elapsed = time.time() - t_swap

            if self._pending_elapsed_samples is not None:
                self.elapsed_samples = self._pending_elapsed_samples
                self.metrics_publisher.model_checkpoint_elapsed_samples.set(self.elapsed_samples)
            if self._pending_checkpoint_timestamp is not None:
                ts = self._pending_checkpoint_timestamp
                self.last_checkpoint_created_timestamp = ts
                self.metrics_publisher.model_checkpoint_timestamp.set(ts)
                self._publish_checkpoint_timestamp(server, ts)
            self.metrics_publisher.model_checkpoint_load_success_timestamp.set(int(time.time()))

            self._on_post_hotswap()

            finalize_elapsed = time.time() - finalize_start
            self.metrics_publisher.hotswap_total_swap_seconds.observe(finalize_elapsed)
            self.metrics_publisher.hotswap_active_slot.set(self._active_emb_slot)
            self.metrics_publisher.hotswap_success_total.inc()
            self.metrics_publisher.model_checkpoint_load_time.observe(finalize_elapsed)
            self.metrics_publisher.checkpoint_reload_step_seconds.labels(step="total").observe(
                finalize_elapsed
            )
            self.metrics_publisher.checkpoint_reload_method.labels(method="hotswap").inc()
            self.metrics_publisher.checkpoint_reload_success_total.labels(method="hotswap").inc()

            self._log_gpu_memory_stats("after hot-swap")
            logger.info(
                "[hotswap] Finalization (not-ready window) took %.3fs "
                "(staging=%.3fs, broadcast=%.3fs, checksums=%.3fs, "
                "device_put=%.3fs, swap=%.3fs). "
                "Active slot=%d, elapsed_samples=%d",
                finalize_elapsed,
                staging_elapsed,
                broadcast_elapsed,
                checksums_elapsed,
                device_put_elapsed,
                swap_elapsed,
                self._active_emb_slot,
                self.elapsed_samples,
            )
        except Exception as e:
            logger.error("[hotswap] Finalization failed: %s", e)
            if self.metrics_publisher is not None:
                self.metrics_publisher.hotswap_failure_total.labels(stage="swap").inc()
        finally:
            self._swap_ready.clear()
            self._swap_complete.set()
            server.reset_reload_request()
            server.reset_dequeue_timer()
            if self.is_multi_worker:
                self._write_reload_done()

    def _on_post_hotswap(self) -> None:
        pass

    def request_completed(self):
        assert self.requests_in_flight > 0, "No requests in flight to complete"
        self.requests_in_flight -= 1

    def track_request_in_flight(self):
        if self.requests_in_flight >= self.max_inflight_backend_requests:
            raise RuntimeError(
                f"Too many requests in flight: {self.requests_in_flight} (max allowed: {self.max_inflight_backend_requests})"
            )
        self.requests_in_flight += 1

    def init(self, batch: RecsysFeaturesBatch, rng: jax.Array) -> RecsysInferenceState:
        rng, init_rng = jax.random.split(rng)
        if self.rng_seed == 0:
            init_rng = jax.random.PRNGKey(0), [jax.random.PRNGKey(0)] * 1000
        packing_layout = batch.get("packing_layout")
        batch = jax.tree.map(jnp.ones_like, batch)
        batch["packing_layout"] = packing_layout

        emb_table_init_data = self.create_embedding_init_data()
        if emb_table_init_data.user_embeddings is not None:
            emb_table_init_data.user_embeddings = Parameter(
                x=jnp.ones_like(emb_table_init_data.user_embeddings.x), pspec=P(None)
            )
        if emb_table_init_data.history_post_embeddings is not None:
            emb_table_init_data.history_post_embeddings = Parameter(
                x=jnp.ones_like(emb_table_init_data.history_post_embeddings.x),
                pspec=P(None),
            )
        emb_table_init_data.history_author_embeddings = Parameter(
            x=jnp.ones_like(emb_table_init_data.history_author_embeddings.x),
            pspec=P(None),
        )
        if emb_table_init_data.candidate_post_embeddings is not None:
            emb_table_init_data.candidate_post_embeddings = Parameter(
                x=jnp.ones_like(emb_table_init_data.candidate_post_embeddings.x),
                pspec=P(None),
            )
        emb_table_init_data.candidate_author_embeddings = Parameter(
            x=jnp.ones_like(emb_table_init_data.candidate_author_embeddings.x),
            pspec=P(None),
        )
        if emb_table_init_data.user_ip_embeddings is not None:
            emb_table_init_data.user_ip_embeddings = Parameter(
                x=jnp.ones_like(emb_table_init_data.user_ip_embeddings.x), pspec=P(None)
            )
        post_embeddings = PostEmbeddings(
            post_ids=jnp.arange(1).astype(jnp.int32),
            author_ids=jnp.arange(1).astype(jnp.int32),
            embeddings=Parameter(x=jnp.empty((1, 1), dtype=jnp.bfloat16), pspec=P(None, None)),
            dataset_types=jnp.array([RetrievalDataset.PAD.value], dtype=jnp.int32),
        )
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            post_embeddings = self.model_config.candidate_tower_config.make_post_embeddings()
            padded = self.padded_max_posts
            orig = self.model_config.candidate_tower_config.max_posts
            if padded > orig:
                emb = post_embeddings.embeddings
                padded_emb = jnp.empty((padded, emb.x.shape[1]), dtype=emb.x.dtype)
                post_embeddings = post_embeddings._replace(
                    embeddings=Parameter(
                        x=padded_emb,
                        pspec=emb.pspec,
                    )
                )

        initial_params = self.loss_fn.init(init_rng, batch, emb_table_init_data)
        return RecsysInferenceState(
            params=initial_params,
            rng=rng,
            step=jnp.array(0),
            opt_state=None,
            post_embeddings=post_embeddings,
        )

    @staticmethod
    def _is_fully_replicated(value: jax.Array) -> bool:
        return bool(
            value.addressable_shards
            and all(
                (x.start, x.stop, x.step) == (None,) * 3 for x in value.addressable_shards[0].index
            )
        )

    def _get_host_state_tensor_bytes(
        self,
    ) -> tuple[dict[str, jax.Array], dict[str, np.ndarray]]:
        if not hasattr(self, "host_state") or self.host_state is None:
            self.host_state = jax.device_put(self.state, self.host_sharding)
        host_state = tree_to_dict(unwrap_tree(self.host_state).purge_opt_state())
        tensor_bytes = {}
        for key, value in host_state.items():
            if value is None:
                continue
            if not self._is_fully_replicated(value):
                continue
            arr = _unsafe_jax2np(value.addressable_shards[0].data)
            tensor_bytes[key] = np.ndarray(shape=arr.nbytes, dtype=np.uint8, buffer=arr)
        return host_state, tensor_bytes

    def _broadcast_tensor_to_all_shards(self, value: jax.Array, source_bytes: np.ndarray) -> None:
        for shard in value.addressable_shards[1:]:
            shard_data = _unsafe_jax2np(shard.data)
            np.ndarray(shape=source_bytes.nbytes, dtype=np.uint8, buffer=shard_data)[:] = (
                source_bytes
            )

    @staticmethod
    def _scatter_contiguous_to_shards(
        host_array: jax.Array,
        contiguous_buf: np.ndarray,
    ) -> None:
        row_bytes = int(np.prod(host_array.shape[1:])) * host_array.dtype.itemsize
        for shard in host_array.addressable_shards:
            row_start = shard.index[0].start or 0
            row_stop = shard.index[0].stop or host_array.shape[0]
            shard_np = _unsafe_jax2np(shard.data)
            dst = np.ndarray(shape=shard_np.nbytes, dtype=np.uint8, buffer=shard_np)
            dst[:] = contiguous_buf[row_start * row_bytes : row_stop * row_bytes]

    def _get_copy_port_timeout(self) -> int:
        env_timeout = os.environ.get("COPY_PORT_TIMEOUT")
        if env_timeout is not None:
            try:
                return int(env_timeout)
            except ValueError:
                logger.warning(
                    "Invalid COPY_PORT_TIMEOUT env var %r, using default %d",
                    env_timeout,
                    self.copy_port_timeout,
                )
        return self.copy_port_timeout

    def _ensure_shm_signal_at_version(self) -> None:
        if self.worker_id != 0 or not self.shared_emb_table_path:
            return
        try:
            current = _read_version_file(_SHM_WEIGHTS_VERSION)
            if _read_version_file(_SHM_WEIGHTS_SIGNAL) < current:
                _write_version_file(_SHM_WEIGHTS_SIGNAL, current)
            emb_signal = f"{self.shared_emb_table_path}.loaded"
            if _read_version_file(emb_signal) < current:
                _write_version_file(emb_signal, current)
        except OSError as ex:
            logger.warning(f"Worker {self.worker_id}: failed to publish SHM signal catch-up: {ex}")

    def _load_via_copy_port(
        self, ctx: TrainerContext, tag: str | None = None
    ) -> tuple[bool, int, int]:
        assert self.h2d_state is not None
        worker_id = self.worker_id
        shared_path = self.shared_emb_table_path

        if not self.copy_url:
            raise OSError("--copy_url is empty")

        copy_url = _resolve_copy_url_to_http(self.copy_url)

        host_state, tensor_bytes = self._get_host_state_tensor_bytes()
        tensors = list(tensor_bytes.items())

        t_copy_start = time.time()
        if shared_path and worker_id != 0:
            while _read_version_file(_SHM_WEIGHTS_VERSION) <= self._last_loaded_shm_weights_version:
                time.sleep(0.05)
            expected = _read_version_file(_SHM_WEIGHTS_VERSION)
            logger.info(f"Worker {worker_id}: waiting for model weights version {expected}...")
            while _read_version_file(_SHM_WEIGHTS_SIGNAL) < expected:
                time.sleep(1)
            prefix, checksums, created_timestamp = _load_tensors_from_shm(tensors)
            self._last_loaded_shm_weights_version = expected
            logger.info(
                f"Worker {worker_id}: loaded model weights from shared memory (v{expected})"
            )
        else:
            prefix, checksums, created_timestamp = (
                xai_recsys_engine.maybe_load_fully_replicated_tensors(
                    self.elapsed_samples, copy_url, tensors
                )
            )
            if shared_path:
                _save_tensors_to_shm(tensors, prefix, checksums, created_timestamp)
                version = _read_version_file(_SHM_WEIGHTS_VERSION)
                _write_version_file(_SHM_WEIGHTS_SIGNAL, version)
                logger.info(f"Worker 0: model weights saved to shared memory (version={version})")

        t_copy_tensors = time.time()
        copy_tensors_bytes = sum(t.nbytes for _, t in tensors)
        copy_tensors_duration = max(t_copy_tensors - t_copy_start, 1e-9)
        logger.info(
            "Loading %d tensors (%d bytes) took %.2f sec; %.2f GB/s",
            len(tensors),
            copy_tensors_bytes,
            copy_tensors_duration,
            copy_tensors_bytes * 1e-9 / copy_tensors_duration,
        )

        for key, tensor in tensors:
            self._broadcast_tensor_to_all_shards(host_state[key], tensor)

        pe_key = "post_embeddings.embeddings"
        pe_checksum: int | None = None
        if pe_key in host_state and host_state[pe_key] is not None and pe_key not in tensor_bytes:
            pe_value = host_state[pe_key]
            pe_total_bytes = int(np.prod(pe_value.shape)) * pe_value.dtype.itemsize
            pe_buf = np.empty(pe_total_bytes, dtype=np.uint8)
            t_pe_start = time.time()
            (pe_checksum,) = xai_recsys_engine.load_tensor_no_resharding(
                f"{prefix}/",
                copy_url,
                pe_key,
                pe_buf,
            )
            self._scatter_contiguous_to_shards(pe_value, pe_buf)
            t_pe_end = time.time()
            pe_duration = max(t_pe_end - t_pe_start, 1e-9)
            logger.info(
                "Loading %s took %.2f sec; %.2f GB/s",
                pe_key,
                pe_duration,
                pe_total_bytes * 1e-9 / pe_duration,
            )
            if self.metrics_publisher is not None:
                self.metrics_publisher.checkpoint_reload_bytes.labels(step="post_embeddings").set(
                    pe_total_bytes
                )
                self.metrics_publisher.checkpoint_reload_throughput_bytes_per_second.labels(
                    step="post_embeddings"
                ).set(pe_total_bytes / pe_duration)

        self.host_state, self.state = self.reload_state(self.host_state, self.state)

        saved = {}
        if self.checkpoint_config.verify_checksums:
            saved = json.loads(checksums)["global_checksums"]
            state = tree_to_dict(unwrap_tree(self.state).purge_opt_state())
            row_sharded_keys = {pe_key} if pe_checksum is not None else set()
            state = {k: v for k, v in state.items() if k not in row_sharded_keys}
            sums = jax.tree.map(adler32.adler32, state)
            errors = [k for k, v in sums.items() if k != "opt_state" and v.item() != saved[k]]
            if errors:
                if self.metrics_publisher is not None:
                    self.metrics_publisher.checkpoint_reload_failure_total.labels(
                        method="copy_port"
                    ).inc()
                raise ValueError(f"checksums for {prefix} do not match: {errors}")
            if pe_checksum is not None and pe_checksum != saved.get(pe_key):
                if self.metrics_publisher is not None:
                    self.metrics_publisher.checkpoint_reload_failure_total.labels(
                        method="copy_port"
                    ).inc()
                raise ValueError(f"checksums for {prefix} do not match: {pe_key}")

        signal_file = f"{shared_path}.loaded" if shared_path is not None else ""

        emb_table = self.h2d_state.emb_table
        if shared_path and worker_id != 0:
            logger.info(
                f"Worker {worker_id}: waiting for worker 0 to load emb_table via copy_port..."
            )
            while _read_version_file(signal_file) <= self._last_loaded_emb_table_version:
                time.sleep(0.5)
            self._last_loaded_emb_table_version = _read_version_file(signal_file)
            logger.info(
                f"Worker {worker_id}: emb_table loaded by worker 0 "
                f"(v{self._last_loaded_emb_table_version}), continuing"
            )
            checksum = saved.get("emb_table", 0) if self.checkpoint_config.verify_checksums else 0
            emb_table_duration = 0.0
        else:
            t0 = time.time()
            (checksum,) = xai_recsys_engine.load_tensor_no_resharding(
                f"{prefix}/",
                copy_url,
                "emb_table",
                np.ndarray(emb_table.nbytes, dtype=np.uint8, buffer=self.h2d_state.emb_table),
            )
            t1 = time.time()
            emb_table_duration = max(t1 - t0, 1e-9)
            if signal_file:
                version = _read_version_file(_SHM_WEIGHTS_VERSION)
                _write_version_file(signal_file, version)
                self._last_loaded_emb_table_version = version
                logger.info(
                    f"Worker {worker_id}: emb_table loaded via copy_port "
                    f"(v{version}), signaled other workers"
                )
        logger.info(
            f"Loading embedding table took {emb_table_duration:.2f} sec; {emb_table.nbytes * 1e-9 / max(emb_table_duration, 1e-9):.2f} GB/s"
        )
        if self.metrics_publisher is not None:
            self.metrics_publisher.checkpoint_reload_step_seconds.labels(step="copy_port").observe(
                t_copy_tensors - t_copy_start
            )
            self.metrics_publisher.checkpoint_reload_step_seconds.labels(
                step="emb_table_copy_port"
            ).observe(emb_table_duration)
            self.metrics_publisher.checkpoint_reload_method.labels(method="copy_port").inc()
            self.metrics_publisher.checkpoint_reload_bytes.labels(step="copy_port").set(
                copy_tensors_bytes
            )
            self.metrics_publisher.checkpoint_reload_throughput_bytes_per_second.labels(
                step="copy_port"
            ).set(copy_tensors_bytes / max(copy_tensors_duration, 1e-9))
            self.metrics_publisher.checkpoint_reload_bytes.labels(step="emb_table_copy_port").set(
                emb_table.nbytes
            )
            self.metrics_publisher.checkpoint_reload_throughput_bytes_per_second.labels(
                step="emb_table_copy_port"
            ).set(emb_table.nbytes / max(emb_table_duration, 1e-9))
        expected_emb_table_checksum = saved.get("emb_table")
        if (
            self.checkpoint_config.verify_checksums
            and expected_emb_table_checksum is not None
            and checksum != expected_emb_table_checksum
        ):
            if self.metrics_publisher is not None:
                self.metrics_publisher.checkpoint_reload_failure_total.labels(
                    method="copy_port"
                ).inc()
            raise ValueError(f"checksums for {prefix} do not match: emb_table")
        if self.metrics_publisher is not None:
            self.metrics_publisher.checkpoint_reload_success_total.labels(method="copy_port").inc()
            ts = created_timestamp if created_timestamp > 0 else time.time()
            self.last_checkpoint_created_timestamp = ts
            self.metrics_publisher.model_checkpoint_timestamp.set(ts)
        return True, int(prefix.split("_")[2].split("/")[0]), 0

    def _load_via_nfs(self, ctx: TrainerContext, tag: str | None = None) -> tuple[bool, int, int]:
        assert self.h2d_state is not None
        worker_id = self.worker_id
        shared_path = self.shared_emb_table_path

        try:
            ckpt = self._newer_nfs_checkpoint(ctx)
        except Exception as disc_e:
            logger.warning("NFS fallback: checkpoint discovery failed: %s", disc_e)
            self._ensure_shm_signal_at_version()
            return False, self.elapsed_samples, 0

        if ckpt is None:
            self._ensure_shm_signal_at_version()
            return False, self.elapsed_samples, 0

        ctx.checkpoint = ckpt
        logger.info(
            "NFS fallback: discovered newer checkpoint on disk "
            "(elapsed_samples=%d > current %d, path=%s)",
            ckpt.elapsed_samples,
            self.elapsed_samples,
            ckpt.path,
        )

        try:
            t_nfs = time.time()
            res = Trainer.maybe_load_checkpoint(self, ctx, tag)

            if shared_path and worker_id == 0:
                self.host_state = jax.device_put(self.state, self.host_sharding)
                _, refreshed_tensor_bytes = self._get_host_state_tensor_bytes()
                refreshed_tensors = list(refreshed_tensor_bytes.items())
                nfs_checksums: dict[str, int] = {}
                if self.checkpoint_config.verify_checksums:
                    state_dict = tree_to_dict(unwrap_tree(self.state).purge_opt_state())
                    nfs_checksums = {
                        k: adler32.adler32(v).item()
                        for k, v in state_dict.items()
                        if k != "opt_state"
                    }
                nfs_checksums_json = json.dumps({"global_checksums": nfs_checksums})
                _save_tensors_to_shm(
                    refreshed_tensors,
                    f"elapsed_samples_{res[1]:018d}/",
                    nfs_checksums_json,
                    0.0,
                )
                version = _read_version_file(_SHM_WEIGHTS_VERSION)
                _write_version_file(_SHM_WEIGHTS_SIGNAL, version)
                logger.info(
                    f"Worker 0: model weights published to shm after NFS fallback (v{version})"
                )

            emb_load = self.h2d_state.emb_table
            if self.training_ep > 0 and isinstance(self.dataset, PhoenixDataset):
                raw_vocab = self.dataset.input_vocab_size
                padded_vocab = self.padded_input_vocab_size
                if raw_vocab != padded_vocab:
                    assert ctx.checkpoint is not None
                    assert isinstance(
                        self.model_config,
                        (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig),
                    )
                    orbax_path = ctx.checkpoint.path + "/orbax-ckpt"
                    srcs = typing.cast(
                        list[tuple[str, str, int, int]],
                        ocdbt.load_shard_sources(orbax_path, prefix="emb_table/c/"),
                    )
                    ckpt_bytes = sum(s[3] for s in srcs)
                    row_bytes = (
                        self.model_config.emb_table_width
                        * np.dtype(self.model_config.embedding_dtype).itemsize
                    )
                    if ckpt_bytes == raw_vocab * row_bytes:
                        logger.info(
                            "NFS checkpoint has un-padded emb_table (%d rows); "
                            "loading into padded buffer (%d rows)",
                            raw_vocab,
                            padded_vocab,
                        )
                        emb_load = self.h2d_state.emb_table[:raw_vocab]

            signal_file = f"{shared_path}.loaded" if shared_path is not None else ""

            if shared_path and worker_id != 0:
                logger.info(f"Worker {worker_id}: waiting for worker 0 to load emb_table...")
                while _read_version_file(signal_file) <= self._last_loaded_emb_table_version:
                    time.sleep(0.5)
                self._last_loaded_emb_table_version = _read_version_file(signal_file)
                logger.info(
                    f"Worker {worker_id}: emb_table loaded by worker 0 "
                    f"(v{self._last_loaded_emb_table_version}), continuing"
                )
            else:
                load_embedding_table(
                    ctx,
                    emb_load,
                    self.checkpoint_config.verify_checksums,
                )
                if signal_file:
                    version = _read_version_file(_SHM_WEIGHTS_VERSION)
                    _write_version_file(signal_file, version)
                    self._last_loaded_emb_table_version = version
                    logger.info(
                        f"Worker {worker_id}: emb_table loaded (v{version}), signaled other workers"
                    )

            if self.metrics_publisher is not None:
                self.metrics_publisher.checkpoint_reload_step_seconds.labels(
                    step="nfs_load"
                ).observe(time.time() - t_nfs)
                self.metrics_publisher.checkpoint_reload_method.labels(method="nfs").inc()
                self.metrics_publisher.checkpoint_reload_success_total.labels(method="nfs").inc()
                loaded = res[0]
                if loaded:
                    nfs_ts = get_model_checkpoint_timestamp(ctx)
                    if nfs_ts > 0:
                        self.last_checkpoint_created_timestamp = nfs_ts
                        self.metrics_publisher.model_checkpoint_timestamp.set(nfs_ts)

            self._ensure_shm_signal_at_version()
            return res

        except Exception as nfs_e:
            logger.warning(
                "NFS fallback: failed to load checkpoint %s: %s. Keeping current in-memory model.",
                getattr(ctx.checkpoint, "path", None),
                nfs_e,
            )
            if self.metrics_publisher is not None:
                self.metrics_publisher.checkpoint_reload_failure_total.labels(method="nfs").inc()
            self._ensure_shm_signal_at_version()
            return False, self.elapsed_samples, 0

    def maybe_load_checkpoint(
        self, ctx: TrainerContext, tag: str | None = None
    ) -> tuple[bool, int, int]:
        assert self.h2d_state is not None

        self._reload_found_no_new_checkpoint = False

        if os.environ.get("DEBUG_ALLOW_RANDOM_INIT") == "1":
            logger.info("DEBUG_ALLOW_RANDOM_INIT set; skipping checkpoint load")
            self._ensure_shm_signal_at_version()
            return True, 0, 0

        store = None
        for detect_store in service_registry.CHECKPOINT_STORE_DETECTORS:
            store = detect_store(self)
            break

        copy_port_start_time = time.time()
        copy_port_timeout = self._get_copy_port_timeout()

        if self.worker_id == 0:
            new_version = _read_version_file(_SHM_WEIGHTS_VERSION) + 1
            _write_version_file(_SHM_WEIGHTS_VERSION, new_version)
            try:
                os.remove(_SHM_WEIGHTS_SIGNAL)
            except FileNotFoundError:
                pass

        try:
            return self._load_via_copy_port(ctx, tag)
        except OSError as e:
            if self.elapsed_samples != 0 and "no greater than old prefix" in repr(e):
                logger.info("No new checkpoint available via copy_port: %s", e)
                self._reload_found_no_new_checkpoint = True
                self._ensure_shm_signal_at_version()
                return False, self.elapsed_samples, 0

            copy_port_elapsed = time.time() - copy_port_start_time
            logger.warning(
                "copy_port checkpoint load failed: copy_url=%s, elapsed_samples=%d, "
                "elapsed_time=%.2fs, timeout=%ds, error=%s",
                self.copy_url,
                self.elapsed_samples,
                copy_port_elapsed,
                copy_port_timeout,
                e,
            )
            if copy_port_timeout > 0 and copy_port_elapsed >= copy_port_timeout:
                logger.error(
                    "copy_port exceeded timeout (%ds): this may indicate training is down "
                    "or network issues. Falling back to storage/NFS.",
                    copy_port_timeout,
                )

            if self.metrics_publisher is not None:
                failure_reason = _classify_copy_port_error(e)
                self.metrics_publisher.copy_port_failure_total.labels(reason=failure_reason).inc()
                self.metrics_publisher.checkpoint_reload_failure_total.labels(
                    method="copy_port"
                ).inc()

            if not (self.load or self.store_load):
                if self.elapsed_samples == 0:
                    raise e
                self._ensure_shm_signal_at_version()
                return False, self.elapsed_samples, 0

        for maybe_load_from_storage in service_registry.CHECKPOINT_STORAGE_LOADERS:
            _storage_res = maybe_load_from_storage(self, ctx, store)
            if _storage_res is not None:
                return _storage_res
        logger.info("copy_port unavailable; falling back to NFS for checkpoint load")

        return self._load_via_nfs(ctx, tag)

    def maybe_rotate_debug_log_path(self):
        if self.debug_log_dir is None:
            return None
        os.makedirs(self.debug_log_dir, exist_ok=True)
        ts_ref = int(
            time.time() // self.debug_log_rotation_seconds * self.debug_log_rotation_seconds
        )
        path = os.path.join(self.debug_log_dir, f"{ts_ref}.parquet")
        if path != self.debug_log_path:
            if self.debug_writer is not None:
                self.debug_writer.close()
            self.debug_writer = pq.ParquetWriter(path, schema=debug_logger.build_debug_schema())
            self.debug_log_path = path
        return path

    def example_data(self, bs: int) -> RecsysFeaturesBatch:
        batch = self.dataset.example_data(bs)
        if self.using_seqpack:
            batch = pack_batch(
                batch=batch,
                num_devices_per_process=self.parallel_config.num_devices_per_process,
                num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                dist=None,
                rng=None,
                block_size=self._seqpack_block_size,
            )
            if self.using_fa4:
                batch = self.add_block_sparse_layout(batch)
        return batch

    def _get_persistent_buffer(
        self, request_in_flight_id: int, bucket_size: int | None = None
    ) -> RecsysFeaturesBatch:
        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        key = (bs, request_in_flight_id)
        if key not in self.persistent_buffers:
            logger.info(
                f"Creating persistent buffer for bucket={bs}, "
                f"request_in_flight_id={request_in_flight_id}"
            )

            assert isinstance(self.dataset, PhoenixDataset)
            self.persistent_buffers[key] = self.dataset.example_data(bs)

        return self.persistent_buffers[key]

    def _get_h2d_state(self, request_in_flight_id: int, bucket_size: int | None = None) -> H2DState:
        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        template = self.h2d_state_by_bs.get(bs) if self.h2d_state_by_bs else None
        if template is None:
            template = self.h2d_state
        assert template is not None, "h2d_state must be initialized before creating request states"

        key = (bs, request_in_flight_id)
        if key not in self.h2d_states:
            logger.info(
                f"Creating H2D state for bucket={bs}, request_in_flight_id={request_in_flight_id}"
            )

            if request_in_flight_id == 0:
                self.h2d_states[key] = template
            else:
                multimodal_embeddings_copy = None
                if template.multimodal_embeddings is not None:
                    mm_buf = template.multimodal_embeddings
                    mm_np = np.zeros(mm_buf.jax_buffer.shape, dtype=np.float16)
                    new_jax_buf = jax.device_put(
                        mm_np,
                        mm_buf.jax_buffer.sharding.with_memory_kind("pinned_host"),
                    )
                    del mm_np
                    multimodal_embeddings_copy = MultimodalEmbeddingBuffer(
                        jax_buffer=new_jax_buf,
                        embedding_dim=mm_buf.embedding_dim,
                        cached_numpy_views=_create_mm_embedding_buffer_views(new_jax_buf),
                    )
                orig_buf = template.embeddings_buffer
                new_np = np.zeros(orig_buf.shape, dtype=orig_buf.dtype)
                new_buf = jax.device_put(new_np, orig_buf.sharding.with_memory_kind("pinned_host"))
                del new_np

                self.h2d_states[key] = H2DState(
                    emb_table=template.emb_table,
                    embeddings_buffer=new_buf,
                    embedding_slices=template.embedding_slices,
                    multimodal_embeddings=multimodal_embeddings_copy,
                )

        return self.h2d_states[key]

    def _start_mm_embeddings_preload(self) -> None:
        if self._mm_client is not None:
            return
        if self.model_config.multimodal_embedding_type is None or self.fake_mm_embeddings:
            return
        if self.is_multi_worker and not self.is_leader:
            return
        assert isinstance(self.model_config, RecsysAggregatedModelConfig)
        emb_dim = self.model_config.multimodal_embedding_dim
        logger.info(
            "Worker %d: starting MM embeddings O2 preload early "
            "(shared=%s, emb_dim=%d) to overlap with checkpoint download",
            self.worker_id,
            self.is_multi_worker,
            emb_dim,
        )
        self._mm_preload_start_time = time.time()
        self._mm_client = xai_recsys_engine.PyMmEmbeddingsClient(
            shared=self.is_multi_worker,
            worker_id=self.worker_id,
            emb_dim=emb_dim,
        )

    def _await_mm_embeddings_client(self) -> PyMmEmbeddingsClient | None:
        if self.model_config.multimodal_embedding_type is None:
            return None
        if self.fake_mm_embeddings:
            logger.info("fake_mm_embeddings=True; skipping O2 fetch, buffer will be zeros")
            return None

        assert isinstance(self.model_config, RecsysAggregatedModelConfig)
        emb_dim = self.model_config.multimodal_embedding_dim
        worker_id = self.worker_id
        is_shared = self.is_multi_worker
        signal_file = "/dev/shm/mm_embeddings.loaded"

        if is_shared and worker_id != 0:
            logger.info(f"Worker {worker_id}: waiting for worker 0 to create shared MM cache...")
            while not os.path.exists(signal_file):
                time.sleep(1)
            logger.info(f"Worker {worker_id}: worker 0 done, opening shared MM cache")

        mm_client = self._mm_client
        if mm_client is None:
            logger.info(
                f"Worker {worker_id}: creating MM embeddings client "
                f"(shared={is_shared}, emb_dim={emb_dim})"
            )
            mm_client = xai_recsys_engine.PyMmEmbeddingsClient(
                shared=is_shared,
                worker_id=worker_id,
                emb_dim=emb_dim,
            )

        wait_start = time.time()
        mm_client.wait_until_ready()
        wait_elapsed = time.time() - wait_start
        if self._mm_preload_start_time > 0:
            total_preload = time.time() - self._mm_preload_start_time
            logger.info(
                "MultiModal embeddings are ready: blocked %.1fs in run() "
                "(total O2 preload %.1fs, ~%.1fs overlapped with checkpoint download)",
                wait_elapsed,
                total_preload,
                max(total_preload - wait_elapsed, 0.0),
            )
        else:
            logger.info("MultiModal embeddings are ready (waited %.1fs)", wait_elapsed)
        self._startup_mm_preload_wait_seconds = wait_elapsed

        if is_shared and worker_id == 0:
            open(signal_file, "w").close()
            logger.info("Worker 0: MM embeddings signal file created")

        return mm_client

    def _h2d_shape_kwargs(self, bs: int) -> dict[str, Any]:
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)
        mm_dim = (
            self.model_config.multimodal_embedding_dim
            if isinstance(self.model_config, RecsysAggregatedModelConfig)
            and self.model_config.multimodal_embedding_type is not None
            else 0
        )
        if self.using_seqpack:
            batch = self.example_data(bs)

            def _packed_shape(x) -> tuple[int, int]:
                return x.shape[0], int(np.prod(x.shape[1:]))

            return dict(
                user_embeddings_shape=_packed_shape(batch["user_hashes"]),
                history_post_embeddings_shape=_packed_shape(batch["history_seq"]["post_hashes"]),
                history_author_embeddings_shape=_packed_shape(batch["history_seq"]["auth_hashes"]),
                candidate_post_embeddings_shape=_packed_shape(
                    batch["candidate_seq"]["post_hashes"]
                ),
                candidate_author_embeddings_shape=_packed_shape(
                    batch["candidate_seq"]["auth_hashes"]
                ),
                candidate_multimodal_embeddings_shape=(
                    (
                        batch["candidate_seq"]["post_hashes"].shape[0],
                        batch["candidate_seq"]["post_hashes"].shape[1],
                        mm_dim,
                    )
                    if mm_dim
                    else None
                ),
                user_ip_embeddings_shape=(
                    _packed_shape(batch["user_ip_hashes"])
                    if self.model_config.use_ip_address
                    else None
                ),
            )
        ht = self.dataset.hash_table
        return dict(
            user_embeddings_shape=(bs, ht.num_user_hashes),
            history_post_embeddings_shape=(
                bs,
                ht.num_item_hashes * self.history_seq_len,
            ),
            history_author_embeddings_shape=(
                bs,
                ht.num_author_hashes * self.history_seq_len,
            ),
            candidate_post_embeddings_shape=(
                bs,
                ht.num_item_hashes * self.candidate_seq_len,
            ),
            candidate_author_embeddings_shape=(
                bs,
                ht.num_author_hashes * self.candidate_seq_len,
            ),
            candidate_multimodal_embeddings_shape=(
                (bs, self.candidate_seq_len, mm_dim) if mm_dim else None
            ),
            user_ip_embeddings_shape=(
                (bs, ht.num_ip_hashes) if self.model_config.use_ip_address else None
            ),
        )

    def _rebind_numa_all_nodes(self) -> None:
        if not self.numa_membind_all_nodes:
            return
        try:
            import numa

            nodes = numa.memory.get_allocation_allowed_nodes()
            if len(nodes) <= 1:
                return
            numa.schedule.run_on_nodes(*nodes)
            numa.memory.set_membind_nodes(*nodes)
            logger.info("NUMA: re-bound CPU+memory to nodes %s", nodes)
        except Exception as e:
            logger.warning("NUMA: failed to re-bind nodes: %s", e)

    def create_state(self, ctx: TrainerContext) -> None:
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)
        assert self.model_config.get_embed_memory_kind() == "unpinned_host"

        self._rebind_numa_all_nodes()

        self._startup_process_start_epoch = _process_start_time_epoch()
        create_state_start = time.time()
        if self._startup_process_start_epoch is not None:
            self._startup_init_seconds = create_state_start - self._startup_process_start_epoch

        self._start_mm_embeddings_preload()

        emb_table_shape = (
            self.padded_input_vocab_size,
            self.model_config.emb_table_width,
        )
        shared_table: npt.NDArray[np.uint16] | None = None
        self.h2d_state_by_bs = {}
        for bs in sorted(self.sorted_buckets, reverse=True):
            state = create_h2d_state(
                self.model_config.embedding_dtype,
                emb_table_shape=emb_table_shape,
                data_sharding=self.data_sharding,
                shared_emb_table_path=(
                    self.shared_emb_table_path if shared_table is None else None
                ),
                hotswap_emb_table_path=(
                    self.hotswap_emb_table_path if shared_table is None else None
                ),
                emb_table=shared_table,
                **self._h2d_shape_kwargs(bs),
            )
            if shared_table is None:
                shared_table = state.emb_table
            self.h2d_state_by_bs[bs] = state
        self.h2d_state = self.h2d_state_by_bs[max(self.sorted_buckets)]

        super().create_state(ctx)
        self._startup_checkpoint_load_seconds = time.time() - create_state_start

    @abstractmethod
    def get_batch_prep(
        self,
        persistent_buffer: RecsysFeaturesBatch,
        request_in_flight_id: int,
        bucket_size: int | None = None,
    ) -> Any:
        raise NotImplementedError

    def _build_eligible_mask_pipelined(
        self, request: RequestBatch | None, batch_id: int, bucket_size: int | None = None
    ) -> jax.Array | None:
        return None

    @abstractmethod
    def gather_embeddings_and_forward(
        self,
        state: RecsysInferenceState,
        batch: RecsysFeaturesBatch,
        rng: jax.Array,
        request_in_flight_id: int,
        batch_id: int,
        request: RequestBatch | None = None,
        eligible_mask: jax.Array | None = None,
        bucket_size: int | None = None,
    ) -> jax.Array | np.ndarray | tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def reply_request(
        self,
        request: RequestBatch,
        output_array: np.ndarray | tuple[np.ndarray, ...] | dict,
        orig_batch_size: int,
        bucket_size: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_server(
        self, mm_client: xai_recsys_engine.PyMmEmbeddingsClient | None = None
    ) -> ServerProtocol[RequestBatch]:
        raise NotImplementedError

    def wait_for_server_shutdown(
        self, server: ServerProtocol[RequestBatch], timeout: float = 5.0
    ) -> bool:
        start_time = time.time()
        while not server.is_finished():
            if time.time() - start_time > timeout:
                logger.warning(f"Server shutdown timed out after {timeout} seconds")
                return False
            time.sleep(0.1)
        return True

    def prepare_and_compute(
        self,
        state: RecsysInferenceState,
        server: ServerProtocol[RequestBatch],
        request: RequestBatch | None,
        rng: jax.Array,
        batch_id: int,
    ) -> _PipelinedBatchResult[RequestBatch] | None:
        if request is None:
            return None

        orig_batch_size = request.len()
        if orig_batch_size == 0:
            return None

        bucket_size = self.select_bucket(orig_batch_size)
        if len(self.sorted_buckets) > 1:
            logger.info(
                f"[batch={batch_id}] orig_batch_size={orig_batch_size} -> bucket={bucket_size}"
            )

        request_in_flight_id = self.current_request_id % self.max_inflight_backend_requests
        self.current_request_id += 1

        if self.jax_profile and jax.process_index() == 0:
            if self.request_count == 3 and not self.profiling_active:
                profiling_path = self.jax_profile_path or "tensorboard"
                logger.info(f"Starting JAX profiling to {profiling_path}")
                start_trace(log_dir=profiling_path)
                self.profiling_active = True
            elif self.request_count == 10 and self.profiling_active:
                logger.info("Stopping JAX profiling")
                stop_trace()
                self.profiling_active = False

        persistent_buffer: RecsysFeaturesBatch = self._get_persistent_buffer(
            request_in_flight_id, bucket_size
        )
        assert persistent_buffer["history_seq"]["actions"] is not None

        batch_prep = self.get_batch_prep(persistent_buffer, request_in_flight_id, bucket_size)
        batch_preparation_timer = Timer()
        with jax_profiler.TraceAnnotation(
            "batch_preparation",
            batch_id=batch_id,
        ):
            success = server.compute_batch_inputs_rust_quick(
                request,
                request_in_flight_id,
                batch_prep,
                batch_size=bucket_size,
            )
        batch_preparation_time = batch_preparation_timer.elapsed()

        if not success or orig_batch_size == 0:
            if self.metrics_publisher is not None:
                self.metrics_publisher.model_requests_failed.inc()
            return None

        self.track_request_in_flight()
        logger.info(
            f"[batch={batch_id}] Step 1 (batch_prep): {batch_preparation_time * 1000:.2f}ms, {orig_batch_size=}"
        )
        if self.metrics_publisher is not None:
            self.metrics_publisher.model_latency.labels("batch_prep").observe(
                batch_preparation_time
            )

        mask_t = Timer()
        eligible_mask = self._build_eligible_mask_pipelined(request, batch_id, bucket_size)
        mask_elapsed = mask_t.elapsed()
        if eligible_mask is not None:
            logger.info(
                f"[batch={batch_id}] Step 1.5 (eligible_mask, pipelined): {mask_elapsed * 1000:.1f}ms"
            )

        batch = persistent_buffer
        if self.using_seqpack:
            with jax_profiler.TraceAnnotation(
                "pack_batch",
                batch_id=batch_id,
            ):
                pack_t = Timer()
                batch = pack_batch(
                    batch=persistent_buffer,
                    num_devices_per_process=self.parallel_config.num_devices_per_process,
                    num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                    dist=None,
                    rng=None,
                    block_size=self._seqpack_block_size,
                )
                if self.using_fa4:
                    batch = self.add_block_sparse_layout(batch)
                pack_time = pack_t.elapsed()
            logger.info(f"[batch={batch_id}] Step 1.75 (pack_batch): {pack_time * 1000:.2f}ms")
            if self.metrics_publisher is not None:
                self.metrics_publisher.model_latency.labels("pack_batch").observe(pack_time)

        inference_timer = Timer()
        outputs = self.gather_embeddings_and_forward(
            state,
            batch,
            rng,
            request_in_flight_id,
            batch_id,
            request,
            eligible_mask=eligible_mask,
            bucket_size=bucket_size,
        )

        self.request_count += 1

        return _PipelinedBatchResult(
            request=request,
            out=outputs,
            orig_batch_size=orig_batch_size,
            bucket_size=bucket_size,
            inf_timer=inference_timer,
            batch_id=batch_id,
        )

    def _has_newer_checkpoint(self) -> bool:
        if self.elapsed_samples == 0 or not self.copy_url:
            return True
        try:
            return self._copy_port_has_newer_checkpoint()
        except Exception:
            pass
        checkpoint_path = self.checkpoint_config.checkpoint_dir
        store = Store.from_path(checkpoint_path) if checkpoint_path else Store.FS
        for path in (self.load, self.store_load):
            if path and Store.from_path(path) != Store.FS:
                return True
        if store != Store.FS:
            return True
        ctx = getattr(self, "ctx", None)
        if ctx is None or getattr(ctx, "meta", None) is None:
            return True
        try:
            if self._newer_nfs_checkpoint(ctx) is not None:
                return True
        except Exception as e:
            logger.warning("copy_port unreachable and NFS newness probe failed: %s", e)
            return True
        logger.info(
            "copy_port unreachable and NFS has nothing newer than elapsed_samples=%d; "
            "skipping reload without draining.",
            self.elapsed_samples,
        )
        return False

    def _copy_port_has_newer_checkpoint(self) -> bool:
        copy_url = _resolve_copy_url_to_http(self.copy_url)
        try:
            xai_recsys_engine.maybe_load_fully_replicated_tensors(
                self.elapsed_samples, copy_url, []
            )
            return True
        except OSError as e:
            if "no greater than old prefix" in repr(e):
                return False
            raise

    def _newer_nfs_checkpoint(self, ctx: TrainerContext):
        latest = ctx.meta.discover_checkpoint(self.name, self.load)
        if latest is None:
            logger.info("NFS: no checkpoint on disk (name=%s, load=%s)", self.name, self.load)
            return None
        ckpt, _ = latest
        if ckpt.elapsed_samples <= self.elapsed_samples:
            logger.info(
                "NFS: latest on disk (elapsed_samples=%d) not newer than in-memory (%d)",
                ckpt.elapsed_samples,
                self.elapsed_samples,
            )
            return None
        return ckpt

    def _poll_copy_port_probe(self) -> str:
        if self._reload_probe_executor is None:
            self._reload_probe_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_INFLIGHT_COPY_PORT_PROBES,
                thread_name_prefix="copy-port-probe",
            )
        self._reload_probe_inflight = [f for f in self._reload_probe_inflight if not f.done()]

        fut = self._reload_probe_future
        if fut is not None:
            if fut.done():
                self._reload_probe_future = None
                try:
                    has_newer = fut.result()
                except Exception:
                    has_newer = True
                return "newer" if has_newer else "nothing"
            if time.time() - self._reload_probe_started >= COPY_PORT_PROBE_TIMEOUT:
                self._reload_probe_future = None
                return "newer"
            return "pending"
        if len(self._reload_probe_inflight) >= MAX_INFLIGHT_COPY_PORT_PROBES:
            return "newer"
        new_fut = self._reload_probe_executor.submit(self._has_newer_checkpoint)
        self._reload_probe_inflight.append(new_fut)
        self._reload_probe_future = new_fut
        self._reload_probe_started = time.time()
        return "pending"

    @staticmethod
    def _reload_trigger_check_enabled(
        suppress_reload_trigger: bool, restart_timeout: float | None
    ) -> bool:
        if suppress_reload_trigger:
            return False
        return restart_timeout is None or restart_timeout > 45

    def handle_requests(
        self,
        state: RecsysInferenceState,
        server: ServerProtocol[RequestBatch],
        rng: jax.Array,
        start: float,
        restart_timeout: float | None,
        stop_event: threading.Event | None = None,
        suppress_reload_trigger: bool = False,
    ):
        assert self.metrics_publisher is not None
        prev_out: _PipelinedBatchResult[RequestBatch] | None = None
        self._service_timer = None
        prev_dispatched_real = False

        def as_np_array(x: jax.Array) -> np.ndarray:
            if self.use_pinned_d2h:
                from xrex.inference.pinned_d2h import PinnedD2HBuffer

                if x.dtype == jnp.bfloat16:
                    x = x.astype(jnp.float32)
                    x.block_until_ready()

                buffer_key = (x.shape, np.dtype(x.dtype))
                if buffer_key not in self._pinned_buffers:
                    n_shards = self.parallel_config.num_devices_per_process
                    self._pinned_buffers[buffer_key] = PinnedD2HBuffer(
                        shape=x.shape,
                        dtype=np.dtype(x.dtype),
                        num_shards=n_shards,
                        num_buffers=self.pinned_d2h_num_buffers,
                    )
                return self._pinned_buffers[buffer_key].transfer(x)
            if x.dtype == jnp.bfloat16:
                return np.asarray(x, dtype=np.float32)
            return np.asarray(x)

        def jax_array_bytes(x: jax.Array) -> int:
            return int(x.size * x.dtype.itemsize)

        def output_bytes_val(out: Any) -> int:
            try:
                return int(out.nbytes)
            except Exception:
                try:
                    return int(sum(arr.nbytes for arr in out))
                except Exception:
                    return 0

        check_reload = self._reload_trigger_check_enabled(suppress_reload_trigger, restart_timeout)

        def maybe_process_and_reply(
            p: _PipelinedBatchResult[RequestBatch] | None,
        ) -> None:
            if p is None:
                return
            assert self.metrics_publisher is not None
            last_req = p.request
            last_out = p.out
            last_orig_batch_size = p.orig_batch_size
            last_bucket_size = p.bucket_size
            last_inf_t = p.inf_timer
            last_batch_id = p.batch_id

            approx_output_bytes = None
            if isinstance(last_out, jax.Array):
                approx_output_bytes = jax_array_bytes(last_out)
            elif isinstance(last_out, tuple):
                approx_output_bytes = sum(jax_array_bytes(x) for x in last_out)

            has_nan: bool = False

            with jax_profiler.TraceAnnotation(
                "output_conversion",
                batch_id=last_batch_id,
                approx_output_bytes=approx_output_bytes,
            ):
                if isinstance(last_out, jax.Array):
                    with jax_profiler.TraceAnnotation(
                        "block_until_ready",
                        batch_id=last_batch_id,
                    ):
                        block_until_ready_t = Timer()
                        last_out = last_out.block_until_ready()
                        logger.info(
                            f"[batch={last_batch_id}] block_until_ready took {block_until_ready_t.elapsed() * 1000:.2f}ms"
                        )
                    with jax_profiler.TraceAnnotation(
                        "as_np_array",
                        batch_id=last_batch_id,
                    ):
                        as_np_array_t = Timer()
                        output_array = as_np_array(last_out)
                        logger.info(
                            f"[batch={last_batch_id}] as_np_array took {as_np_array_t.elapsed() * 1000:.2f}ms"
                        )
                elif isinstance(last_out, dict):
                    output_array = {}
                    block_until_ready_t = Timer()
                    for ds_type, arrays in last_out.items():
                        waited = jax.tree.map(lambda x: x.block_until_ready(), arrays)
                        output_array[ds_type] = tuple(as_np_array(x) for x in waited)
                    logger.info(
                        f"[batch={last_batch_id}] block_until_ready+as_np_array took {block_until_ready_t.elapsed() * 1000:.2f}ms"
                    )
                elif isinstance(last_out, tuple):
                    with jax_profiler.TraceAnnotation(
                        "block_until_ready",
                        batch_id=last_batch_id,
                    ):
                        block_until_ready_t = Timer()
                        last_out = jax.tree.map(lambda x: x.block_until_ready(), last_out)
                        logger.info(
                            f"[batch={last_batch_id}] block_until_ready took {block_until_ready_t.elapsed() * 1000:.2f}ms"
                        )
                    with jax_profiler.TraceAnnotation(
                        "as_np_array",
                        batch_id=last_batch_id,
                    ):
                        as_np_array_t = Timer()
                        assert len(last_out) in (3, 4), (
                            f"model should return (log_probs, cont_preds, has_nan) or "
                            f"(top_k_indices, top_k_scores, generated, has_nan), got {len(last_out)}"
                        )
                        *model_out_jax, has_nan_jax = last_out
                        output_array = tuple(as_np_array(x) for x in model_out_jax)
                        has_nan = bool(np.asarray(has_nan_jax))
                        logger.info(
                            f"[batch={last_batch_id}] as_np_array took {as_np_array_t.elapsed() * 1000:.2f}ms"
                        )
                else:
                    raise ValueError(f"last_out was of type {type(last_out)}")

            output_bytes = output_bytes_val(output_array)

            with jax_profiler.TraceAnnotation(
                "postprocess_before_reply",
                batch_id=last_batch_id,
                output_bytes=output_bytes,
            ):
                if has_nan:
                    logger.warning("NaN detected in model output!")
                    self.metrics_publisher.model_requests_nan_detected.inc()

                with jax_profiler.TraceAnnotation(
                    "request_completed",
                    batch_id=last_batch_id,
                ):
                    duration = last_inf_t.elapsed()
                    self.metrics_publisher.model_latency.labels("inference").observe(duration)
                    self.request_completed()

                    logger.debug(f"Request completed, inflight requests: {self.requests_in_flight}")

                resp_t = Timer()

            with jax_profiler.TraceAnnotation(
                "reply_request",
                batch_id=last_batch_id,
            ):
                self.reply_request(last_req, output_array, last_orig_batch_size, last_bucket_size)

            reply_duration = resp_t.elapsed()
            logger.info(
                f"[batch={last_batch_id}] request.reply() took {reply_duration * 1000:.2f}ms"
            )

            self.metrics_publisher.model_requests_success.inc()
            self.metrics_publisher.model_latency.labels("response_write").observe(reply_duration)
            if self.metrics_publisher.inference_original_batch_size is not None:
                self.metrics_publisher.inference_original_batch_size.observe(last_orig_batch_size)
            padded_batch_size = last_bucket_size
            self.metrics_publisher.observe_bucket_selection(last_orig_batch_size, padded_batch_size)
            seqs_per_second = padded_batch_size / max(duration, 1e-9)
            if self.metrics_publisher.inference_throughput_seqs_per_second is not None:
                self.metrics_publisher.inference_throughput_seqs_per_second.observe(seqs_per_second)
            logger.info(
                f"[batch={last_batch_id}] Time to process: {duration:.3f}s for {padded_batch_size} = {seqs_per_second:.3f} seq/s"
            )

        hotswap_last_reload_trigger = time.time()
        self._reload_probe_future = None
        self._reload_probe_started = 0.0

        while stop_event is None or not stop_event.is_set():
            if not self.enable_hotswap and restart_timeout is not None:
                if time.time() - start >= restart_timeout:
                    break

            reload_triggered = False
            hotswap_busy = self.enable_hotswap and (
                self._reload_requested.is_set()
                or self._swap_ready.is_set()
                or self._live_swap_ready.is_set()
            )
            if check_reload and not hotswap_busy:
                if self.is_multi_worker and self._observe_peer_reload_trigger():
                    reload_triggered = True
                elif server.check_reload_request():
                    if self.enable_hotswap or self.is_multi_worker:
                        self._broadcast_reload_trigger()
                        reload_triggered = True
                    else:
                        probe = self._poll_copy_port_probe()
                        if probe == "nothing":
                            logger.info(
                                "ReloadModel received but no source has anything newer than "
                                f"elapsed_samples={self.elapsed_samples}; staying ready (reload skipped)."
                            )
                            server.reset_reload_request()
                            if self.metrics_publisher is not None:
                                self.metrics_publisher.checkpoint_reload_skipped_total.labels(
                                    reason="no_new_checkpoint", drained="false"
                                ).inc()
                        elif probe == "newer":
                            reload_triggered = True

            if reload_triggered:
                if self.enable_hotswap:
                    if not self._reload_requested.is_set() and not self._swap_ready.is_set():
                        logger.info(
                            "[hotswap] Reload request received, signalling background loader"
                        )
                        self._reload_requested.set()
                else:
                    logger.info("Reload request received, exiting main loop")
                    break

            if self.enable_hotswap and restart_timeout is not None:
                if time.time() - hotswap_last_reload_trigger >= restart_timeout:
                    if (
                        not self._reload_requested.is_set()
                        and not self._swap_ready.is_set()
                        and not self._live_swap_ready.is_set()
                    ):
                        logger.info(
                            "[hotswap] Periodic reload trigger (every %.0fs)",
                            restart_timeout,
                        )
                        self._reload_requested.set()
                    hotswap_last_reload_trigger = time.time()

            if self.enable_hotswap and self._reload_noop.is_set():
                self._reload_noop.clear()
                logger.info("[hotswap] No new checkpoint, resetting reload request")
                server.reset_reload_request()
                server.reset_dequeue_timer()

            if self.enable_hotswap and self._hotswap_aborted.is_set():
                self._hotswap_aborted.clear()
                logger.warning(
                    "[hotswap] Reload cycle failed; continuing to serve current "
                    "snapshot (elapsed_samples=%d)",
                    self.elapsed_samples,
                )
                server.reset_reload_request()
                server.reset_dequeue_timer()

            if self.enable_hotswap and self._live_swap_ready.is_set():
                if self._live_swap_requires_pipeline_flush:
                    maybe_process_and_reply(prev_out)
                    prev_out = None
                self._apply_live_swap(server)
                state = self.state

            if self.enable_hotswap and self._swap_ready.is_set():
                maybe_process_and_reply(prev_out)
                prev_out = None
                logger.info("[hotswap] Download complete, exiting to drain before finalization")
                break

            self.maybe_rotate_debug_log_path()
            batch_id = self.dequeued_batch_id
            self.dequeued_batch_id += 1
            dequeue_t = Timer()

            if self.is_multi_worker:
                _dq_result: list = [None]
                _dq_event = threading.Event()

                def _bg_dequeue(_r=_dq_result, _e=_dq_event) -> None:
                    try:
                        _r[0] = server.dequeue()
                    except Exception:
                        pass
                    _e.set()

                threading.Thread(target=_bg_dequeue, daemon=True, name="dequeue-poll").start()
                _dq_event.wait(timeout=5.0)
                request = _dq_result[0]
            else:
                request = server.dequeue()

            if request is not None and request.len() > 0:
                logger.info(f"[batch={batch_id}] dequeue() took {dequeue_t.elapsed() * 1000:.2f}ms")
            this_out = self.prepare_and_compute(state, server, request, rng, batch_id)
            if this_out is not None:
                if self.metrics_publisher is not None:
                    self.metrics_publisher.model_requests_total.inc()
                if prev_dispatched_real and self._service_timer is not None:
                    server.record_service_time_ms(self._service_timer.elapsed() * 1000.0)
                self._service_timer = Timer()
            prev_dispatched_real = this_out is not None

            if self.use_pipelining:
                maybe_process_and_reply(prev_out)
                prev_out = this_out
            else:
                maybe_process_and_reply(this_out)

        maybe_process_and_reply(prev_out)

    def _publish_checkpoint_timestamp(self, server: Any, timestamp: float) -> None:
        setter = getattr(server, "set_checkpoint_timestamp", None)
        if setter is None:
            return
        try:
            setter(float(timestamp))
        except Exception as e:
            logger.warning(
                f"Failed to publish checkpoint timestamp {timestamp} to gRPC server: {e}"
            )

    def _warmup_buckets(self, rng: jax.Array) -> None:
        assert isinstance(self.state, RecsysInferenceState)
        for bs in self.sorted_buckets:
            batch = self._get_persistent_buffer(0, bs)
            if self.using_seqpack:
                batch = pack_batch(
                    batch=batch,
                    num_devices_per_process=self.parallel_config.num_devices_per_process,
                    num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                    dist=None,
                    rng=None,
                    block_size=self._seqpack_block_size,
                )
                if self.using_fa4:
                    batch = self.add_block_sparse_layout(batch)
            out = self.gather_embeddings_and_forward(self.state, batch, rng, 0, -1, bucket_size=bs)
            if isinstance(out, jax.Array):
                out = out.block_until_ready()
                logger.info(f"Warmup bucket={bs} output shape: {out.shape}")
            elif isinstance(out, tuple):
                out = jax.tree.map(lambda x: x.block_until_ready(), out)
                logger.info(f"Warmup bucket={bs} output shapes: {[x.shape for x in out]}")

    SUPPORTED_MODEL_CONFIGS = (
        RecsysAggregatedModelConfig,
        RecsysTwoTowerModelConfig,
    )

    def run(self):
        assert isinstance(self.model_config, self.SUPPORTED_MODEL_CONFIGS)
        assert isinstance(self.dataset, PhoenixDataset)

        logger.info(f"Jax version: {jax.__version__}")

        self.metrics_publisher = get_metrics_publisher(port=8000 + (self.grpc_port - 9988))
        self.metrics_publisher.register_inference_batch_metrics(
            self.inference_batch_size, buckets=self.sorted_buckets
        )
        if self._startup_process_start_epoch is not None:
            self.metrics_publisher.pod_start_timestamp.set(self._startup_process_start_epoch)

        if self.is_multi_worker:
            self._last_observed_reload_trigger = self._read_reload_trigger()
            logger.info(
                f"Worker {self.worker_id}: starting with reload trigger "
                f"watermark = {self._last_observed_reload_trigger}"
            )

        ready_event = threading.Event()
        stop_event = threading.Event()
        server = None

        def _handle_sigterm(signum, frame):
            logger.info("SIGTERM received, initiating graceful shutdown")
            ready_event.clear()
            stop_event.set()
            try:
                if server is not None:
                    server.set_ready(False)
            except Exception as err:
                logger.warning("Failed to set server not ready on SIGTERM: %s", err)

        signal.signal(signal.SIGTERM, _handle_sigterm)

        def _handle_shutdown_request() -> None:
            logger.info("Shutdown requested via readiness endpoint.")
            stop_event.set()
            try:
                if server is not None:
                    logger.info("Cancelling gRPC server due to shutdown request.")
                    server.cancel()
            except Exception as err:
                logger.warning("Shutdown cancel failed: %s", err)

        if self.readiness_port is not None:
            status_server = StatusServer(
                config=StatusServerConfig(
                    port=self.readiness_port,
                    ready_event=ready_event,
                    on_shutdown=_handle_shutdown_request,
                )
            )
            status_server.start()

        ctx = self.ctx
        ctx.on_train_start()

        assert isinstance(self.state, RecsysInferenceState), f"got {type(self.state)}"
        np.random.seed(42)
        random.seed(0)

        rng = jax.random.PRNGKey(self.rng_seed)

        initial_checkpoint_timestamp = get_model_checkpoint_timestamp(ctx)
        if initial_checkpoint_timestamp > 0:
            self.metrics_publisher.model_checkpoint_timestamp.set(initial_checkpoint_timestamp)
        else:
            self.metrics_publisher.model_checkpoint_timestamp.set(int(time.time()))
        self.metrics_publisher.model_checkpoint_load_success_timestamp.set(int(time.time()))

        mm_client: PyMmEmbeddingsClient | None = self._await_mm_embeddings_client()

        logger.info("Setting up gRPC server.")
        server = self.create_server(mm_client)
        assert server is not None
        logger.info("gRPC server ready.")

        self._publish_checkpoint_timestamp(
            server,
            self.last_checkpoint_created_timestamp
            if self.last_checkpoint_created_timestamp > 0
            else initial_checkpoint_timestamp,
        )

        logger.info("Model warm up starting (%d bucket(s)).", len(self.sorted_buckets))
        warmup_start = time.time()
        with self.mesh:
            self._warmup_buckets(rng)
        self._startup_warmup_seconds = time.time() - warmup_start
        logger.info("Model warm up finished.")
        self.metrics_publisher.model_warmup_seconds.labels(phase="initial").observe(
            time.time() - warmup_start
        )
        logger.info(f"peak_mem: {get_peak_bytes_in_use() / (1 << 30)}")

        if self.enable_hotswap:
            self._init_hotswap_buffers()
            if getattr(self, "_is_hotswap_leader", True):
                target = self._coordinator_loop
                role = "leader"
            else:
                target = self._follower_coordinator_loop
                role = "follower"
            self._bg_loader_thread = threading.Thread(
                target=target, daemon=True, name=f"hotswap-coordinator-{role}"
            )
            self._bg_loader_thread.start()
            logger.info("Hot-swap coordinator thread started (role=%s).", role)

        ready_event.set()
        server.set_ready(True)
        self.metrics_publisher.grpc_server_ready_timestamp.set(time.time())
        self.metrics_publisher.set_pod_status("serving")

        startup = self.metrics_publisher.inference_startup_seconds
        startup.labels(phase="init").set(self._startup_init_seconds)
        startup.labels(phase="checkpoint_load").set(self._startup_checkpoint_load_seconds)
        startup.labels(phase="mm_preload_wait").set(self._startup_mm_preload_wait_seconds)
        startup.labels(phase="warmup").set(self._startup_warmup_seconds)
        proc_start = self._startup_process_start_epoch
        if proc_start is not None:
            total = time.time() - proc_start
            startup.labels(phase="total").set(total)
            logger.info(
                "Server ready to serve: time-to-ready %.1fs "
                "(init=%.1fs, checkpoint_load=%.1fs, mm_preload_wait=%.1fs, warmup=%.1fs).",
                total,
                self._startup_init_seconds,
                self._startup_checkpoint_load_seconds,
                self._startup_mm_preload_wait_seconds,
                self._startup_warmup_seconds,
            )
        else:
            logger.info(
                "Server ready to serve "
                "(checkpoint_load=%.1fs, mm_preload_wait=%.1fs, warmup=%.1fs).",
                self._startup_checkpoint_load_seconds,
                self._startup_mm_preload_wait_seconds,
                self._startup_warmup_seconds,
            )

        while True:
            try:
                with self.mesh:
                    start = time.time()
                    self.metrics_publisher.set_pod_status("serving")
                    self.handle_requests(
                        self.state, server, rng, start, self.restart_period, stop_event
                    )

                    if self.enable_hotswap and self._swap_ready.is_set():
                        logger.info(
                            "[hotswap] Draining for %.0fs before finalization",
                            self.graceful_restart_timeout,
                        )
                        self.metrics_publisher.set_pod_status("draining")
                        ready_event.clear()
                        server.set_ready(False)

                        drain_start = time.time()
                        self.enable_hotswap = False
                        self.handle_requests(
                            self.state,
                            server,
                            rng,
                            drain_start,
                            self.graceful_restart_timeout,
                            stop_event,
                            suppress_reload_trigger=True,
                        )
                        drain_elapsed = time.time() - drain_start
                        logger.info(
                            "[hotswap] Drain complete (%.1fs), starting finalization",
                            drain_elapsed,
                        )

                        self._finalize_hot_swap(server)
                        if self._post_hotswap_warmup_needed:
                            logger.info("[hotswap] Post-swap warmup starting")
                            self.metrics_publisher.set_pod_status("warming_up")
                            warmup_t = Timer()
                            self._warmup_buckets(rng)
                            self.metrics_publisher.model_warmup_seconds.labels(
                                phase="reload"
                            ).observe(warmup_t.elapsed())
                            logger.info(
                                "[hotswap] Post-swap warmup finished (%.1fs)",
                                warmup_t.elapsed(),
                            )
                        self.enable_hotswap = True
                        ready_event.set()
                        server.set_ready(True)
                        self.metrics_publisher.set_pod_status("serving")
                        continue

                    if not self.enable_hotswap:
                        logger.info("begin graceful termination.")
                        self.metrics_publisher.set_pod_status("draining")
                        ready_event.clear()
                        server.set_ready(False)

                        graceful_term_start = time.time()
                        _restore_hotswap = self.enable_hotswap
                        self.enable_hotswap = False
                        self.handle_requests(
                            self.state,
                            server,
                            rng,
                            graceful_term_start,
                            self.graceful_restart_timeout,
                            stop_event,
                            suppress_reload_trigger=True,
                        )
                        logger.info(
                            "end graceful termination (drained %.1fs of %.0fs budget).",
                            time.time() - graceful_term_start,
                            self.graceful_restart_timeout,
                        )

                        server.set_checkpoint_loading()

                if stop_event.is_set():
                    logger.info("Stop event set; exiting inference loop.")
                    break

                if self.enable_hotswap:
                    logger.warning("[hotswap] handle_requests exited unexpectedly, re-entering")
                    continue

                reload_cycle_start = time.time()
                self.metrics_publisher.set_pod_status("reloading_checkpoint")

                t = Timer()
                loaded, new_elapsed_samples, _ = self.maybe_load_checkpoint(ctx)
                elapsed = t.elapsed()

                if loaded:
                    self.elapsed_samples = new_elapsed_samples
                    logger.info(f"New checkpoint loaded in {elapsed:.3f} seconds")
                    self.metrics_publisher.model_checkpoint_load_time.observe(elapsed)
                    self.metrics_publisher.model_checkpoint_load_success_timestamp.set(
                        int(time.time())
                    )
                    self.metrics_publisher.model_checkpoint_elapsed_samples.set(
                        self.elapsed_samples
                    )
                    if self.last_checkpoint_created_timestamp > 0:
                        self.metrics_publisher.model_checkpoint_timestamp.set(
                            self.last_checkpoint_created_timestamp
                        )
                elif self._reload_found_no_new_checkpoint:
                    logger.info(f"No new checkpoint available ({elapsed:.3f}s)")
                    self.metrics_publisher.checkpoint_reload_skipped_total.labels(
                        reason="no_new_checkpoint", drained="true"
                    ).inc()
                else:
                    logger.warning(
                        f"Checkpoint reload found nothing to load after draining "
                        f"({elapsed:.3f}s) and copy_port did not report 'nothing newer' -- "
                        f"treating as a failed reload, not a no-op skip."
                    )

                server.reset_dequeue_timer()

                logger.info("Model warm up starting after checkpoint reload.")
                self.metrics_publisher.set_pod_status("warming_up")
                warmup_start = time.time()
                with self.mesh:
                    self._warmup_buckets(rng)
                warmup_elapsed = time.time() - warmup_start
                logger.info("Model warm up finished after checkpoint reload.")
                self.metrics_publisher.checkpoint_reload_step_seconds.labels(step="warmup").observe(
                    warmup_elapsed
                )
                self.metrics_publisher.model_warmup_seconds.labels(phase="reload").observe(
                    warmup_elapsed
                )
                self.metrics_publisher.checkpoint_reload_step_seconds.labels(step="total").observe(
                    time.time() - reload_cycle_start
                )

                if self.is_multi_worker:
                    self._write_reload_done()
                    logger.info(
                        f"Worker {self.worker_id}: reload done file updated to "
                        f"trigger={self._last_observed_reload_trigger}"
                    )
                    num_workers = self.num_workers
                    target = self._last_observed_reload_trigger
                    wait_start = time.time()
                    wait_deadline = wait_start + 600.0
                    done_dir = "/dev/shm"
                    while time.time() < wait_deadline:
                        peer_done = [
                            _read_version_file(f"{done_dir}/reload_done_w{w}")
                            for w in range(num_workers)
                        ]
                        if all(d >= target for d in peer_done):
                            break
                        time.sleep(0.5)
                    peer_wait_elapsed = time.time() - wait_start
                    if peer_wait_elapsed >= 600.0:
                        logger.warning(
                            f"Worker {self.worker_id}: timed out waiting for "
                            f"peers to reach trigger={target} after "
                            f"{peer_wait_elapsed:.1f}s; releasing reload flag "
                            f"anyway (peers may lag); done={peer_done}"
                        )
                    elif peer_wait_elapsed > 0.5:
                        logger.info(
                            f"Worker {self.worker_id}: waited {peer_wait_elapsed:.1f}s "
                            f"for peers to reach trigger={target}"
                        )

                server.reset_reload_request()

                self._publish_checkpoint_timestamp(server, self.last_checkpoint_created_timestamp)

                server.clear_checkpoint_loading()
                ready_event.set()
                server.set_ready(True)
                self.enable_hotswap = _restore_hotswap
                self.metrics_publisher.set_pod_status("serving")
                logger.info("Checkpoint reload complete, server is ready again.")

            except Exception as e:
                self.metrics_publisher.model_checkpoint_load_failure_timestamp.set(int(time.time()))
                self.metrics_publisher.set_pod_status("error")
                try:
                    server.reset_reload_request()
                except Exception:
                    pass
                if server is not None and not server.is_finished():
                    server.cancel()
                    server.cancel_forcefully()
                    self.wait_for_server_shutdown(server)
                logger.error(f"Exception thrown during inference loop: {e}")
                if self.debug_writer is not None:
                    self.debug_writer.close()
                    self.debug_writer = None
                ctx.on_completed("Stopping recsys inference.")
                jax.distributed.shutdown()
                raise SystemExit(1)

        if self.enable_hotswap:
            self._hotswap_stop.set()
            self._reload_requested.set()

            if self._loader_process is not None and self._loader_process.is_alive():
                try:
                    if self._request_pipe is not None:
                        self._request_pipe.send("stop")
                except Exception:
                    pass
                self._loader_process.join(timeout=10)
                if self._loader_process.is_alive():
                    self._loader_process.kill()
                    self._loader_process.join(timeout=5)

            if self._bg_loader_thread is not None:
                self._bg_loader_thread.join(timeout=10)

            for p in (self._request_pipe, self._result_pipe):
                if p is not None:
                    try:
                        p.close()
                    except Exception:
                        pass

            if (
                "/dev/shm"
                and os.path.isdir("/dev/shm")
                and getattr(self, "_is_hotswap_leader", True)
            ):
                try:
                    shutil.rmtree("/dev/shm")
                    logger.info("[hotswap] Cleaned up shared memory dir: %s", "/dev/shm")
                except Exception as e:
                    logger.warning("[hotswap] Failed to clean up %s: %s", "/dev/shm", e)
                for path in (
                    self._hotswap_setup_signal,
                    self._hotswap_standby_meta_file,
                    self._hotswap_standby_version_file,
                ):
                    if path:
                        try:
                            os.remove(path)
                        except FileNotFoundError:
                            pass

        server.cancel()
        self.wait_for_server_shutdown(server)
        if not server.is_finished():
            server.cancel_forcefully()
            self.wait_for_server_shutdown(server)


@final
@configclass
class RankingModelRunner(
    BaseModelRunner[
        xai_recsys_engine.PredictRequestBatch,
        RecsysAggregatedModelConfig,
    ]
):
    forward_jit_by_bs: dict[int, JittedOrCompiled] = field(default_factory=dict, init=False)

    def _rng_and_init_data(self, seq_len: int | None = None, bs: int | None = None):
        bs = bs or self.inference_batch_size
        if self.using_seqpack:
            init_data = self.example_data(bs)
            rng = jax.ShapeDtypeStruct((2,), jnp.uint32)
        else:
            rng, init_data = super()._rng_and_init_data(seq_len=seq_len)
            init_data = typing.cast(RecsysFeaturesBatch, init_data)
            if bs != self.batch_size:
                init_data = jax.tree.map(
                    lambda x: jax.ShapeDtypeStruct((bs, *x.shape[1:]), x.dtype),
                    init_data,
                )

        assert isinstance(self.model_config, RecsysAggregatedModelConfig)
        if self.model_config.multimodal_embedding_type is not None:
            if self.using_seqpack:
                init_data["candidate_seq"]["embedding"] = jax.ShapeDtypeStruct(
                    (
                        init_data["candidate_seq"]["post_hashes"].shape[0],
                        init_data["candidate_seq"]["post_hashes"].shape[1],
                        self.model_config.multimodal_embedding_dim,
                    ),
                    np.float32,
                )
            else:
                init_data["candidate_seq"]["embedding"] = jax.ShapeDtypeStruct(
                    (
                        bs,
                        self.candidate_seq_len,
                        self.model_config.multimodal_embedding_dim,
                    ),
                    np.float32,
                )

        if self.model_config.use_post_sid and self.model_config.sid_num_levels > 0:
            for seq_name in ("history_seq", "candidate_seq"):
                post_hashes = init_data[seq_name]["post_hashes"]
                init_data[seq_name]["post_sids"] = jax.ShapeDtypeStruct(
                    (post_hashes.shape[0], post_hashes.shape[1], self.model_config.sid_num_levels),
                    np.uint16,
                )

        return rng, init_data

    def _get_persistent_buffer(
        self, request_in_flight_id: int, bucket_size: int | None = None
    ) -> RecsysFeaturesBatch:
        buffer = super()._get_persistent_buffer(request_in_flight_id, bucket_size)
        assert isinstance(self.model_config, RecsysAggregatedModelConfig)
        sid_num_levels = self.model_config.sid_num_levels
        if self.model_config.use_post_sid and sid_num_levels > 0:
            bs = bucket_size if bucket_size is not None else self.inference_batch_size
            if buffer["history_seq"].get("post_sids") is None:
                buffer["history_seq"]["post_sids"] = np.zeros(
                    (bs, self.history_seq_len, sid_num_levels), dtype=np.uint16
                )
            if buffer["candidate_seq"].get("post_sids") is None:
                buffer["candidate_seq"]["post_sids"] = np.zeros(
                    (bs, self.candidate_seq_len, sid_num_levels), dtype=np.uint16
                )
        return buffer

    def get_batch_prep(
        self,
        persistent_buffer: RecsysFeaturesBatch,
        request_in_flight_id: int,
        bucket_size: int | None = None,
    ) -> RankingBatchPrep:
        assert persistent_buffer["history_seq"]["actions"] is not None
        assert isinstance(self.model_config, RecsysAggregatedModelConfig)

        h2d_state = self._get_h2d_state(request_in_flight_id, bucket_size)
        mm_buf = h2d_state.multimodal_embeddings
        candidate_embeddings = (
            tuple(v.reshape(-1) for v in mm_buf.cached_numpy_views) if mm_buf is not None else None
        )

        return RankingBatchPrep(
            user_hashes=persistent_buffer["user_hashes"],
            user_ip_hashes=persistent_buffer["user_ip_hashes"],
            history_post_hashes=persistent_buffer["history_seq"]["post_hashes"],
            history_auth_hashes=persistent_buffer["history_seq"]["auth_hashes"],
            history_product_surfaces=persistent_buffer["history_seq"]["product_surface"],
            history_actions=persistent_buffer["history_seq"]["actions"],
            history_continuous_actions=persistent_buffer["history_seq"]["continuous_actions"],
            candidate_post_hashes=persistent_buffer["candidate_seq"]["post_hashes"],
            candidate_auth_hashes=persistent_buffer["candidate_seq"]["auth_hashes"],
            candidate_product_surfaces=persistent_buffer["candidate_seq"]["product_surface"],
            candidate_embeddings=candidate_embeddings,
            candidate_search_query_embeddings=persistent_buffer["candidate_seq"][
                "search_query_embeddings"
            ],
            user_categorical_features=persistent_buffer.get("user_categorical_features"),
            user_bool_features=persistent_buffer.get("user_bool_features"),
            user_float_features=persistent_buffer.get("user_float_features"),
            user_int64_features=persistent_buffer.get("user_int64_features"),
            user_installed_apps=persistent_buffer.get("user_installed_apps_multihot"),
            history_categorical_features=persistent_buffer["history_seq"]["categorical_features"],
            history_bool_features=persistent_buffer["history_seq"]["bool_features"],
            history_float_features=persistent_buffer["history_seq"]["float_features"],
            history_int64_features=persistent_buffer["history_seq"]["int64_features"],
            candidate_categorical_features=persistent_buffer["candidate_seq"][
                "categorical_features"
            ],
            candidate_bool_features=persistent_buffer["candidate_seq"]["bool_features"],
            candidate_float_features=persistent_buffer["candidate_seq"]["float_features"],
            candidate_int64_features=persistent_buffer["candidate_seq"]["int64_features"],
            history_impr_ts=persistent_buffer["history_seq"].get("impr_ts"),
            history_post_creation_ts_sec=persistent_buffer["history_seq"].get(
                "post_creation_ts_sec"
            ),
            candidate_impr_ts=persistent_buffer["candidate_seq"].get("impr_ts"),
            candidate_post_creation_ts_sec=persistent_buffer["candidate_seq"].get(
                "post_creation_ts_sec"
            ),
            history_post_sids=persistent_buffer["history_seq"].get("post_sids"),
            candidate_post_sids=persistent_buffer["candidate_seq"].get("post_sids"),
        )

    def gather_embeddings_and_forward(
        self,
        state: RecsysInferenceState,
        batch: RecsysFeaturesBatch,
        rng: jax.Array,
        request_in_flight_id: int,
        batch_id: int,
        request: xai_recsys_engine.PredictRequestBatch | None = None,
        eligible_mask: jax.Array | None = None,
        bucket_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert isinstance(self.model_config, RecsysAggregatedModelConfig)

        h2d_state = self._get_h2d_state(request_in_flight_id, bucket_size)

        with jax_profiler.TraceAnnotation(
            "lookup_h2d_embeddings",
            batch_id=batch_id,
        ):
            t = Timer()
            merged_embeddings: jax.Array
            candidate_multimodal_embeddings: jax.Array | None
            merged_embeddings, candidate_multimodal_embeddings = lookup_h2d_embeddings(
                h2d_state,
                batch,
                data_sharding=self.data_sharding,
                batch_id=batch_id,
                emb_table_override=self.active_emb_table if self._emb_table_slots else None,
                num_threads=self.embedding_gather_threads,
            )
            emb_lookup_time = t.elapsed()
            logger.info(
                f"[batch={batch_id}] Step 2 (embedding_gather+device_put): {emb_lookup_time * 1000:.2f}ms"
            )
            if self.metrics_publisher is not None:
                self.metrics_publisher.model_latency.labels("embedding_lookup").observe(
                    emb_lookup_time
                )

        if candidate_multimodal_embeddings is not None:
            batch["candidate_seq"]["embedding"] = candidate_multimodal_embeddings

        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        forward_jit = self.forward_jit_by_bs[bs]
        params = self.state.params if self._live_swap_enabled else state.params
        with jax_profiler.TraceAnnotation(
            "forward_jit/ranking",
            batch_id=batch_id,
        ):
            return forward_jit(params, rng, batch, merged_embeddings)

    def reply_request(
        self,
        request: xai_recsys_engine.PredictRequestBatch,
        output_array: tuple[np.ndarray, np.ndarray],
        orig_batch_size: int,
        bucket_size: int,
    ) -> None:
        log_probs, candidate_continuous_predictions = output_array
        if self.using_seqpack:
            log_probs = log_probs.reshape(bucket_size, self.candidate_seq_len, log_probs.shape[-1])
            candidate_continuous_predictions = candidate_continuous_predictions.reshape(
                bucket_size,
                self.candidate_seq_len,
                candidate_continuous_predictions.shape[-1],
            )

        request.reply(log_probs.copy(), candidate_continuous_predictions.copy(), orig_batch_size)

    def create_server(
        self,
        mm_client: xai_recsys_engine.PyMmEmbeddingsClient | None = None,
    ) -> xai_recsys_engine.RecsysPredictorServer:
        assert isinstance(self.dataset, PhoenixDataset)
        assert isinstance(self.model_config, RecsysAggregatedModelConfig)
        hash_keys = self.dataset.hash_table.hash_keys
        engine_sid_num_levels = (
            self.model_config.sid_num_levels if self.model_config.use_post_sid else 0
        )
        return xai_recsys_engine.RecsysPredictorServer(
            self.grpc_port,
            self.metrics_port,
            self.inference_batch_size,
            self.max_inflight_requests,
            channel_size=self.channel_size,
            enqueue_timeout_ms=self.enqueue_timeout_ms,
            queue_max_staleness_ms=self.queue_max_staleness_ms,
            enable_deadline_admission=self.enable_deadline_admission,
            deadline_margin_ms=self.deadline_margin_ms,
            deadline_post_ms=self.deadline_post_ms,
            deadline_fallback_budget_ms=self.deadline_fallback_budget_ms,
            service_time_ewma_alpha=self.service_time_ewma_alpha,
            pipeline_depth=1 if self.use_pipelining else 0,
            mm_client=mm_client,
            user_id_table_size=hash_keys.user_id_table_size,
            user_hash_scales=hash_keys.user_hash_scales,
            user_biases=hash_keys.user_biases,
            user_modulus=hash_keys.user_modulus,
            item_id_table_size=hash_keys.item_id_table_size,
            item_hash_vocab_size=hash_keys.item_hash_vocab_size,
            item_hash_scales=hash_keys.item_hash_scales,
            item_biases=hash_keys.item_biases,
            item_modulus=hash_keys.item_modulus,
            author_id_table_size=hash_keys.author_id_table_size,
            author_hash_scales=hash_keys.author_hash_scales,
            author_biases=hash_keys.author_biases,
            author_modulus=hash_keys.author_modulus,
            output_vocab_size=self.dataset.hash_table.output_vocab_size,
            num_continuous_actions=self.model_config.num_continuous_actions,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            multimodal_embedding_dim=self.model_config.multimodal_embedding_dim,
            search_query_embedding_dim=self.model_config.search_query_embedding_dim,
            ip_id_table_size=hash_keys.ip_id_table_size,
            ip_hash_scales=hash_keys.ip_hash_scales,
            ip_biases=hash_keys.ip_biases,
            ip_modulus=hash_keys.ip_modulus,
            num_user_categorical_features=recsys_batch.USER_CATEGORICAL_FEATURE_SIZE,
            num_user_bool_features=recsys_batch.USER_BOOL_FEATURE_SIZE,
            num_user_float_features=recsys_batch.USER_FLOAT_FEATURE_SIZE,
            num_user_int64_features=recsys_batch.USER_INT64_FEATURE_SIZE,
            num_user_installed_apps=recsys_batch.NUM_USER_INSTALLED_APPS,
            num_post_categorical_features=recsys_batch.POST_CATEGORICAL_FEATURE_SIZE,
            num_post_bool_features=recsys_batch.POST_BOOL_FEATURE_SIZE,
            num_post_float_features=recsys_batch.POST_FLOAT_FEATURE_SIZE,
            num_post_int64_features=recsys_batch.POST_INT64_FEATURE_SIZE,
            enable_async_response_compression=self.enable_async_response_compression,
            sid_num_levels=engine_sid_num_levels,
        )

    def create_executable(self) -> None:
        super().create_executable()
        if hasattr(self, "_registered_jitted") and self._registered_jitted:
            self._registered_jitted.pop("update")

        assert isinstance(self.model_config, RecsysAggregatedModelConfig)
        compiler_options = None if not self.xla_flag_overrides else self.xla_flag_overrides

        self.create_embedding_init_data()

        self.forward_jit_by_bs = {}
        for bs in self.sorted_buckets:
            self.forward_jit_by_bs[bs] = self._build_and_register_forward_jit(bs, compiler_options)
        self.forward_jit = self.forward_jit_by_bs[max(self.sorted_buckets)]

    def _build_and_register_forward_jit(
        self, bs: int, compiler_options: dict[str, Any] | None
    ) -> JittedOrCompiled:
        assert isinstance(self.model_config, RecsysAggregatedModelConfig)

        if self.using_seqpack:
            trace_batch = self.example_data(bs)
            _hist_post_seq = int(np.prod(trace_batch["history_seq"]["post_hashes"].shape[1:]))
            _hist_auth_seq = int(np.prod(trace_batch["history_seq"]["auth_hashes"].shape[1:]))
            _cand_post_seq = int(np.prod(trace_batch["candidate_seq"]["post_hashes"].shape[1:]))
            _cand_auth_seq = int(np.prod(trace_batch["candidate_seq"]["auth_hashes"].shape[1:]))
            _user_seq = int(np.prod(trace_batch["user_hashes"].shape[1:]))
            _ip_seq = (
                int(np.prod(trace_batch["user_ip_hashes"].shape[1:]))
                if self.model_config.use_ip_address
                else 0
            )
            merged_batch = trace_batch["user_hashes"].shape[0]
            bs_per_device = trace_batch["user_hashes"].shape[1]
            packed_history_len = trace_batch["history_seq"]["post_hashes"].shape[1]
            packed_candidate_len = trace_batch["candidate_seq"]["post_hashes"].shape[1]

            def _packed_recsys_embeddings(
                merged_embeddings: jax.Array, sl: EmbeddingSlices
            ) -> RecsysEmbeddings:
                return RecsysEmbeddings(
                    history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :].reshape(
                        merged_embeddings.shape[0],
                        packed_history_len,
                        -1,
                        merged_embeddings.shape[-1],
                    ),
                    history_author_embeddings=merged_embeddings[
                        :, sl.hist_post_end : sl.hist_auth_end, :
                    ].reshape(
                        merged_embeddings.shape[0],
                        packed_history_len,
                        -1,
                        merged_embeddings.shape[-1],
                    ),
                    candidate_post_embeddings=merged_embeddings[
                        :, sl.hist_auth_end : sl.cand_post_end, :
                    ].reshape(
                        merged_embeddings.shape[0],
                        packed_candidate_len,
                        -1,
                        merged_embeddings.shape[-1],
                    ),
                    candidate_author_embeddings=merged_embeddings[
                        :, sl.cand_post_end : sl.cand_auth_end, :
                    ].reshape(
                        merged_embeddings.shape[0],
                        packed_candidate_len,
                        -1,
                        merged_embeddings.shape[-1],
                    ),
                    user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :].reshape(
                        merged_embeddings.shape[0],
                        bs_per_device,
                        -1,
                        merged_embeddings.shape[-1],
                    ),
                    user_ip_embeddings=(
                        merged_embeddings[:, sl.user_end : sl.user_ip_end, :].reshape(
                            merged_embeddings.shape[0],
                            bs_per_device,
                            -1,
                            merged_embeddings.shape[-1],
                        )
                        if sl.user_ip_end > sl.user_end
                        else None
                    ),
                )
        else:
            assert isinstance(self.dataset, PhoenixDataset)
            _ht = self.dataset.hash_table
            _hist_post_seq = _ht.num_item_hashes * self.history_seq_len
            _hist_auth_seq = _ht.num_author_hashes * self.history_seq_len
            _cand_post_seq = _ht.num_item_hashes * self.candidate_seq_len
            _cand_auth_seq = _ht.num_author_hashes * self.candidate_seq_len
            _user_seq = _ht.num_user_hashes
            _use_ip = self.model_config.use_ip_address
            _ip_seq = _ht.num_ip_hashes if _use_ip else 0
            merged_batch = bs

        _user_end = _hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq + _user_seq
        embedding_slices = EmbeddingSlices(
            hist_post_end=_hist_post_seq,
            hist_auth_end=_hist_post_seq + _hist_auth_seq,
            cand_post_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq,
            cand_auth_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq,
            user_end=_user_end,
            user_ip_end=_user_end + _ip_seq,
        )

        @hk.transform
        def forward_fn(
            batch: RecsysFeaturesBatch,
            merged_embeddings: jax.Array,
        ):
            assert isinstance(self.model_config, RecsysAggregatedModelConfig)
            sl = embedding_slices
            if self.using_seqpack:
                recsys_embeddings = _packed_recsys_embeddings(merged_embeddings, sl)
            else:
                recsys_embeddings = RecsysEmbeddings(
                    history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :],
                    history_author_embeddings=merged_embeddings[
                        :, sl.hist_post_end : sl.hist_auth_end, :
                    ],
                    candidate_post_embeddings=merged_embeddings[
                        :, sl.hist_auth_end : sl.cand_post_end, :
                    ],
                    candidate_author_embeddings=merged_embeddings[
                        :, sl.cand_post_end : sl.cand_auth_end, :
                    ],
                    user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :],
                    user_ip_embeddings=(
                        merged_embeddings[:, sl.user_end : sl.user_ip_end, :]
                        if sl.user_ip_end > sl.user_end
                        else None
                    ),
                )
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            logits, candidate_continuous_predictions = model.forward(batch, recsys_embeddings)
            log_probs = jax.nn.log_sigmoid(logits).astype(jnp.float32)
            cont_preds = candidate_continuous_predictions.astype(jnp.float32)
            has_nan = jnp.any(jnp.isnan(log_probs))
            return log_probs, cont_preds, has_nan

        self.forward_fn = forward_fn

        forward_jit = JittedOrCompiled(
            jax.jit(
                forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=(self.data_sharding, self.data_sharding, None),
            ),
            name=f"forward_fn_bs{bs}",
        )

        rng, init_data = self._rng_and_init_data(bs=bs)

        merged_embeddings_init = jnp.zeros(
            (
                merged_batch,
                embedding_slices.user_ip_end,
                self.model_config.emb_table_width,
            ),
            dtype=self.model_config.embedding_dtype,
            device=self.data_sharding,
        )

        rng = jax.ShapeDtypeStruct((2,), np.uint32)
        packing_layout = init_data.get("packing_layout")
        init_data = jax.tree.map(jnp.ones_like, init_data)
        init_data["packing_layout"] = packing_layout

        self.register_jit_function(
            forward_jit,
            self.state_shape.params,
            rng,
            init_data,
            merged_embeddings_init,
            compiler_options=compiler_options,
        )
        return forward_jit


@final
@configclass
class RetrievalModelRunner(
    BaseModelRunner[
        xai_recsys_engine.RetrieveRequestBatch,
        RecsysTwoTowerModelConfig,
    ]
):
    two_tower_forward_jit: JittedOrCompiled | None = None
    two_tower_forward_jit_by_bs: dict[int, JittedOrCompiled] = field(
        default_factory=dict, init=False
    )
    all_post_ids: npt.NDArray[np.int64] | None = None
    all_author_ids: npt.NDArray[np.int64] | None = None
    all_dataset_types: npt.NDArray[np.int32] | None = None
    enable_dataset_slice_topk: bool = False
    enable_async_topk: bool = False
    enable_radix_select_topk: bool = False

    _dataset_ranges_by_type: dict[int, tuple[int, int]] | None = field(default=None, init=False)

    _staged_live_retrieval_meta: (
        tuple[
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int32],
            dict[int, tuple[int, int]] | None,
        ]
        | None
    ) = field(default=None, init=False)
    _live_forward_sds: dict[int, tuple[Any, ...]] = field(default_factory=dict, init=False)
    _staged_live_forward_compiled: (
        dict[tuple[int, tuple[tuple[int, int], ...] | None], JittedOrCompiled] | None
    ) = field(default=None, init=False)
    _live_forward_compiled: dict[
        tuple[int, tuple[tuple[int, int], ...] | None], JittedOrCompiled
    ] = field(default_factory=dict, init=False)

    @staticmethod
    def _dataset_ranges_from_types(
        types: np.ndarray,
    ) -> dict[int, tuple[int, int]] | None:
        types = np.asarray(types).reshape(-1)
        if types.size == 0:
            return None
        n = types.size
        change = np.flatnonzero(types[1:] != types[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [n]))
        ranges: dict[int, tuple[int, int]] = {}
        for s, e in zip(starts, ends):
            val = int(types[s])
            if val in ranges:
                logger.warning(
                    "Post table is NOT contiguous by dataset_type (value %d appears "
                    "in multiple runs); disabling fast per-dataset top_k path.",
                    val,
                )
                return None
            ranges[val] = (int(s), int(e))
        return ranges

    def _compute_dataset_ranges(self) -> None:
        if self.all_dataset_types is None:
            self._dataset_ranges_by_type = None
            return
        ranges = self._dataset_ranges_from_types(np.asarray(self.all_dataset_types))
        self._dataset_ranges_by_type = ranges
        if ranges is None:
            return
        summary = ", ".join(
            f"{next((d.name for d in RetrievalDataset if d.value == v), str(v))}"
            f"=[{s}:{e}]({e - s:,})"
            for v, (s, e) in sorted(ranges.items(), key=lambda kv: kv[1][0])
        )
        logger.info("Per-dataset post-table ranges (fast top_k path): %s", summary)

    def _get_dataset_ranges(
        self,
        target_dataset_types: tuple[int, ...],
        ranges_override: dict[int, tuple[int, int]] | None = None,
    ) -> tuple[tuple[int, int], ...] | None:
        if not self.enable_dataset_slice_topk:
            return None
        ranges = ranges_override if ranges_override is not None else self._dataset_ranges_by_type
        if not ranges:
            return None
        out: list[tuple[int, int]] = []
        for ds_value in target_dataset_types:
            rng = ranges.get(ds_value)
            if rng is None:
                return None
            out.append(rng)
        return tuple(out)

    def maybe_load_checkpoint(
        self, ctx: TrainerContext, tag: str | None = None
    ) -> tuple[bool, int, int]:
        new_ckpt_loaded, elapsed_samples, _ = super().maybe_load_checkpoint(ctx, tag)
        if new_ckpt_loaded:
            assert isinstance(self.state, RecsysInferenceState)
            assert (
                self.state.post_embeddings is not None
                and self.state.post_embeddings.post_ids is not None
                and self.state.post_embeddings.author_ids is not None
                and self.state.post_embeddings.dataset_types is not None
            )
            self.all_post_ids = self.two_int32_to_int64(self.state.post_embeddings.post_ids)
            self.all_author_ids = self.two_int32_to_int64(self.state.post_embeddings.author_ids)
            self.all_dataset_types = np.asarray(
                self.state.post_embeddings.dataset_types, dtype=np.int32
            )
            self._compute_dataset_ranges()
        return new_ckpt_loaded, elapsed_samples, _

    def _on_post_hotswap(self) -> None:
        assert isinstance(self.state, RecsysInferenceState)
        if (
            self.state.post_embeddings is not None
            and self.state.post_embeddings.post_ids is not None
        ):
            self.all_post_ids = self.two_int32_to_int64(self.state.post_embeddings.post_ids)
            self.all_author_ids = self.two_int32_to_int64(self.state.post_embeddings.author_ids)
            self.all_dataset_types = np.asarray(
                self.state.post_embeddings.dataset_types, dtype=np.int32
            )
            self._compute_dataset_ranges()
            logger.info("[hotswap] Rebuilt retrieval metadata (post_ids, author_ids)")

    def _prepare_live_swap_derived_state(self, staged_meta: dict[str, np.ndarray]) -> None:
        self._staged_live_retrieval_meta = None
        post_ids64 = self.two_int32_to_int64(staged_meta["post_embeddings.post_ids"])
        author_ids64 = self.two_int32_to_int64(staged_meta["post_embeddings.author_ids"])
        dataset_types = np.asarray(staged_meta["post_embeddings.dataset_types"], dtype=np.int32)
        ranges = self._dataset_ranges_from_types(dataset_types)
        self._precompile_live_two_tower_variants(ranges)
        self._staged_live_retrieval_meta = (post_ids64, author_ids64, dataset_types, ranges)

    def _commit_live_swap_derived_state(self) -> None:
        staged = self._staged_live_retrieval_meta
        if staged is None:
            logger.warning(
                "[hotswap-live] no staged retrieval metadata at commit; "
                "keeping previous reply-side arrays"
            )
            return
        post_ids64, author_ids64, dataset_types, ranges = staged
        self.all_post_ids = post_ids64
        self.all_author_ids = author_ids64
        self.all_dataset_types = dataset_types
        self._dataset_ranges_by_type = ranges
        self._staged_live_retrieval_meta = None
        staged_exec = self._staged_live_forward_compiled
        if staged_exec is not None:
            self._live_forward_compiled = staged_exec
            self._staged_live_forward_compiled = None
        logger.info(
            "[hotswap-live] Committed retrieval metadata (post_ids, author_ids, ranges)"
            + (f" + {len(staged_exec)} AOT executables" if staged_exec is not None else "")
        )

    @property
    def _live_swap_requires_pipeline_flush(self) -> bool:
        return True

    @property
    def _post_hotswap_warmup_needed(self) -> bool:
        return self.enable_dataset_slice_topk

    def _precompile_live_two_tower_variants(
        self, staged_ranges: dict[int, tuple[int, int]] | None
    ) -> None:
        self._staged_live_forward_compiled = None
        target_dataset_types = tuple(ds.value for ds in self.retrieval_dataset_types)
        ranges_tuple = self._get_dataset_ranges(
            target_dataset_types, ranges_override=staged_ranges or {}
        )
        if ranges_tuple == self._get_dataset_ranges(target_dataset_types):
            logger.info("[hotswap-live] dataset_ranges unchanged; skipping precompile")
            return
        staged: dict[tuple[int, tuple[tuple[int, int], ...] | None], JittedOrCompiled] = {}
        for bs in self.sorted_buckets:
            sds = self._live_forward_sds.get(bs)
            if sds is None:
                logger.warning(
                    "[hotswap-live] no forward shape capture for bucket %d; "
                    "first post-flip request may compile on the serving thread",
                    bs,
                )
                continue
            try:
                t0 = time.time()
                (
                    params_sds,
                    rng_sds,
                    batch_sds,
                    emb_sds,
                    pe_sds,
                    dtypes_sds,
                ) = sds
                with self.mesh:
                    lowered = self._forward_jit_for_bucket(bs).lower(
                        params_sds,
                        rng_sds,
                        batch_sds,
                        emb_sds,
                        pe_sds,
                        dtypes_sds,
                        self.large_k,
                        target_dataset_types,
                        ranges_tuple,
                    )
                    compiled = JittedOrCompiled(
                        lowered.compile(), name=f"two_tower_forward_fn_bs{bs}_live"
                    )
                staged[(bs, ranges_tuple)] = compiled
                logger.info(
                    "[hotswap-live] AOT-compiled bucket %d for staged dataset_ranges (%.2fs)",
                    bs,
                    time.time() - t0,
                )
            except Exception as e:
                logger.warning(
                    "[hotswap-live] AOT compile failed for bucket %d (%s); "
                    "first post-flip request may compile on the serving thread",
                    bs,
                    e,
                )
        self._staged_live_forward_compiled = staged or None

    def _get_persistent_buffer(
        self, request_in_flight_id: int, bucket_size: int | None = None
    ) -> RecsysFeaturesBatch:
        buffer = super()._get_persistent_buffer(request_in_flight_id, bucket_size)
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        if buffer["history_seq"].get("post_ids") is None:
            buffer["history_seq"]["post_ids"] = np.zeros((bs, self.history_seq_len), dtype=np.int64)
        if buffer["history_seq"].get("post_sids") is None:
            buffer["history_seq"]["post_sids"] = np.zeros(
                (
                    bs,
                    self.history_seq_len,
                    self.model_config.user_tower_config.sid_num_levels,
                ),
                dtype=np.uint16,
            )
        return buffer

    def get_batch_prep(
        self,
        persistent_buffer: RecsysFeaturesBatch,
        request_in_flight_id: int,
        bucket_size: int | None = None,
    ) -> RetrievalBatchPrep:
        assert persistent_buffer["history_seq"]["actions"] is not None
        assert persistent_buffer["history_seq"]["post_ids"] is not None
        return RetrievalBatchPrep(
            user_hashes=persistent_buffer["user_hashes"],
            user_ip_hashes=persistent_buffer["user_ip_hashes"],
            history_post_hashes=persistent_buffer["history_seq"]["post_hashes"],
            history_auth_hashes=persistent_buffer["history_seq"]["auth_hashes"],
            history_actions=persistent_buffer["history_seq"]["actions"],
            history_continuous_actions=persistent_buffer["history_seq"]["continuous_actions"],
            history_product_surfaces=persistent_buffer["history_seq"]["product_surface"],
            user_categorical_features=persistent_buffer.get("user_categorical_features"),
            user_bool_features=persistent_buffer.get("user_bool_features"),
            user_float_features=persistent_buffer.get("user_float_features"),
            user_int64_features=persistent_buffer.get("user_int64_features"),
            user_installed_apps=persistent_buffer.get("user_installed_apps_multihot"),
            history_categorical_features=persistent_buffer["history_seq"]["categorical_features"],
            history_bool_features=persistent_buffer["history_seq"]["bool_features"],
            history_float_features=persistent_buffer["history_seq"]["float_features"],
            history_int64_features=persistent_buffer["history_seq"]["int64_features"],
            history_post_ids=persistent_buffer["history_seq"]["post_ids"],
            history_post_sids=persistent_buffer["history_seq"].get("post_sids"),
        )

    def _forward_jit_for_bucket(self, bucket_size: int | None) -> JittedOrCompiled:
        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        return self.two_tower_forward_jit_by_bs[bs]

    def gather_embeddings_and_forward(
        self,
        state: RecsysInferenceState,
        batch: RecsysFeaturesBatch,
        rng: jax.Array,
        request_in_flight_id: int,
        batch_id: int,
        request: xai_recsys_engine.RetrieveRequestBatch | None = None,
        eligible_mask: jax.Array | None = None,
        bucket_size: int | None = None,
    ) -> dict[int, tuple[jax.Array, jax.Array]]:
        if self._live_swap_enabled:
            state = self.state
        forward_jit = self._forward_jit_for_bucket(bucket_size)
        assert state.post_embeddings is not None
        h2d_state = self._get_h2d_state(request_in_flight_id, bucket_size)
        with jax_profiler.TraceAnnotation(
            "lookup_h2d_embeddings",
            batch_id=batch_id,
        ):
            t = Timer()
            recsys_embeddings, _ = lookup_h2d_embeddings(
                h2d_state,
                batch,
                data_sharding=self.data_sharding,
                batch_id=batch_id,
                emb_table_override=self.active_emb_table if self._emb_table_slots else None,
                num_threads=self.embedding_gather_threads,
            )
            emb_lookup_time = t.elapsed()
            logger.info(f"[batch={batch_id}] Embedding gather took {emb_lookup_time * 1000:.1f}ms")
            if self.metrics_publisher is not None:
                self.metrics_publisher.model_latency.labels("embedding_lookup").observe(
                    emb_lookup_time
                )
        target_dataset_types = tuple(ds.value for ds in self.retrieval_dataset_types)

        dataset_ranges = self._get_dataset_ranges(target_dataset_types)

        bs = bucket_size if bucket_size is not None else self.inference_batch_size

        if self.enable_hotswap and bs not in self._live_forward_sds:

            def _sds(x: Any) -> Any:
                if x is None:
                    return None
                return jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), x)

            self._live_forward_sds[bs] = (
                _sds(state.params),
                _sds(rng),
                _sds(batch),
                _sds(recsys_embeddings),
                _sds(state.post_embeddings.embeddings.x),
                _sds(state.post_embeddings.dataset_types),
            )

        live_compiled = (
            self._live_forward_compiled.get((bs, dataset_ranges))
            if self._live_forward_compiled
            else None
        )
        with jax_profiler.TraceAnnotation(
            "forward_jit/retrieval",
            batch_id=batch_id,
        ):
            if live_compiled is not None:
                try:
                    results_tuple = live_compiled(
                        state.params,
                        rng,
                        batch,
                        recsys_embeddings,
                        state.post_embeddings.embeddings.x,
                        state.post_embeddings.dataset_types,
                    )
                except Exception:
                    logger.exception(
                        "[hotswap-live] AOT executable failed for bucket %d; "
                        "evicting it and falling back to the jit path",
                        bs,
                    )
                    self._live_forward_compiled.pop((bs, dataset_ranges), None)
                    live_compiled = None
            if live_compiled is None:
                results_tuple = forward_jit(
                    state.params,
                    rng,
                    batch,
                    recsys_embeddings,
                    state.post_embeddings.embeddings.x,
                    state.post_embeddings.dataset_types,
                    self.large_k,
                    target_dataset_types,
                    dataset_ranges,
                )
        results_dict = {
            ds_type: result for ds_type, result in zip(target_dataset_types, results_tuple)
        }
        return results_dict

    def reply_request(
        self,
        request: xai_recsys_engine.RetrieveRequestBatch,
        output_dict: dict[int, tuple[np.ndarray, np.ndarray]],
        orig_batch_size: int,
        bucket_size: int = 0,
    ) -> None:
        assert isinstance(self.state, RecsysInferenceState)
        assert self.all_post_ids is not None and self.all_author_ids is not None

        ds_types = list(output_dict.keys())
        all_indices = [
            np.array(output_dict[ds][0][:, : self.large_k], dtype=np.int32, copy=True)
            for ds in ds_types
        ]
        all_scores = [
            np.array(output_dict[ds][1][:, : self.large_k], dtype=np.float32, copy=True)
            for ds in ds_types
        ]

        request.reply(
            ds_types,
            all_indices,
            all_scores,
            self.all_post_ids,
            self.all_author_ids,
            self.large_k,
            orig_batch_size,
        )

    def create_server(
        self,
        mm_client: xai_recsys_engine.PyMmEmbeddingsClient | None = None,
    ) -> xai_recsys_engine.RecsysRetrievalPredictorServer:
        assert isinstance(self.dataset, PhoenixDataset)
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        hash_keys = self.dataset.hash_table.hash_keys

        sid_client = None
        _use_post_sid = self.model_config.user_tower_config.use_post_sid
        sid_num_levels = self.model_config.user_tower_config.sid_num_levels
        if _use_post_sid and self.sid_endpoint and sid_num_levels > 0:
            sid_client = xai_recsys_engine.PySemanticIdClient(
                self.sid_endpoint,
                sid_num_levels,
            )
            logger.info(
                "SID client connected: endpoint=%s, sid_num_levels=%d",
                self.sid_endpoint,
                sid_num_levels,
            )
        elif _use_post_sid and not self.sid_endpoint:
            raise ValueError("use_post_sid=True but no sid_endpoint configured")

        return xai_recsys_engine.RecsysRetrievalPredictorServer(
            self.grpc_port,
            self.metrics_port,
            self.inference_batch_size,
            self.max_inflight_requests,
            channel_size=self.channel_size,
            enqueue_timeout_ms=self.enqueue_timeout_ms,
            queue_max_staleness_ms=self.queue_max_staleness_ms,
            enable_deadline_admission=self.enable_deadline_admission,
            deadline_margin_ms=self.deadline_margin_ms,
            deadline_post_ms=self.deadline_post_ms,
            deadline_fallback_budget_ms=self.deadline_fallback_budget_ms,
            service_time_ewma_alpha=self.service_time_ewma_alpha,
            pipeline_depth=1 if self.use_pipelining else 0,
            mm_client=mm_client,
            sid_client=sid_client,
            user_id_table_size=hash_keys.user_id_table_size,
            user_hash_scales=hash_keys.user_hash_scales,
            user_biases=hash_keys.user_biases,
            user_modulus=hash_keys.user_modulus,
            item_id_table_size=hash_keys.item_id_table_size,
            item_hash_vocab_size=hash_keys.item_hash_vocab_size,
            item_hash_scales=hash_keys.item_hash_scales,
            item_biases=hash_keys.item_biases,
            item_modulus=hash_keys.item_modulus,
            author_id_table_size=hash_keys.author_id_table_size,
            author_hash_scales=hash_keys.author_hash_scales,
            author_biases=hash_keys.author_biases,
            author_modulus=hash_keys.author_modulus,
            output_vocab_size=self.dataset.hash_table.output_vocab_size,
            num_continuous_actions=self.model_config.num_continuous_actions,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            multimodal_embedding_dim=getattr(self.model_config, "multimodal_embedding_dim", 0),
            ip_id_table_size=hash_keys.ip_id_table_size,
            ip_hash_scales=hash_keys.ip_hash_scales,
            ip_biases=hash_keys.ip_biases,
            ip_modulus=hash_keys.ip_modulus,
            num_user_categorical_features=recsys_batch.USER_CATEGORICAL_FEATURE_SIZE,
            num_user_bool_features=recsys_batch.USER_BOOL_FEATURE_SIZE,
            num_user_float_features=recsys_batch.USER_FLOAT_FEATURE_SIZE,
            num_user_int64_features=recsys_batch.USER_INT64_FEATURE_SIZE,
            num_user_installed_apps=recsys_batch.NUM_USER_INSTALLED_APPS,
            num_post_categorical_features=recsys_batch.POST_CATEGORICAL_FEATURE_SIZE,
            num_post_bool_features=recsys_batch.POST_BOOL_FEATURE_SIZE,
            num_post_float_features=recsys_batch.POST_FLOAT_FEATURE_SIZE,
            num_post_int64_features=recsys_batch.POST_INT64_FEATURE_SIZE,
            enable_async_response_compression=self.enable_async_response_compression,
            sid_num_levels=sid_num_levels if sid_client is not None else 0,
        )

    def create_executable(self) -> None:
        super().create_executable()
        if hasattr(self, "_registered_jitted") and self._registered_jitted:
            self._registered_jitted.pop("update")

        assert self.inference_batch_size == max(self.sorted_buckets), (
            f"inference_batch_size ({self.inference_batch_size}) must equal the max "
            f"bucket ({max(self.sorted_buckets)})"
        )

        self.two_tower_forward_jit_by_bs = {
            bs: self._build_two_tower_jit(bs) for bs in self.sorted_buckets
        }
        self.two_tower_forward_jit = self.two_tower_forward_jit_by_bs[max(self.sorted_buckets)]

    def _build_two_tower_jit(self, bs: int) -> JittedOrCompiled:
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        assert isinstance(self.dataset, PhoenixDataset)

        seqpack_packed_history_len: int | None = None
        seqpack_packed_candidate_len: int | None = None
        seqpack_bs_per_device: int | None = None
        if self.using_seqpack:
            trace_batch = self.example_data(bs)
            _hist_post_seq = int(np.prod(trace_batch["history_seq"]["post_hashes"].shape[1:]))
            _hist_auth_seq = int(np.prod(trace_batch["history_seq"]["auth_hashes"].shape[1:]))
            _cand_post_seq = int(np.prod(trace_batch["candidate_seq"]["post_hashes"].shape[1:]))
            _cand_auth_seq = int(np.prod(trace_batch["candidate_seq"]["auth_hashes"].shape[1:]))
            _user_seq = int(np.prod(trace_batch["user_hashes"].shape[1:]))
            _ip_seq = (
                int(np.prod(trace_batch["user_ip_hashes"].shape[1:]))
                if self.model_config.use_ip_address
                else 0
            )
            seqpack_packed_history_len = trace_batch["history_seq"]["post_hashes"].shape[1]
            seqpack_packed_candidate_len = trace_batch["candidate_seq"]["post_hashes"].shape[1]
            seqpack_bs_per_device = trace_batch["user_hashes"].shape[1]
        else:
            _ht = self.dataset.hash_table
            _hist_post_seq = _ht.num_item_hashes * self.history_seq_len
            _hist_auth_seq = _ht.num_author_hashes * self.history_seq_len
            _cand_post_seq = _ht.num_item_hashes * self.candidate_seq_len
            _cand_auth_seq = _ht.num_author_hashes * self.candidate_seq_len
            _user_seq = _ht.num_user_hashes
            _ip_seq = _ht.num_ip_hashes if self.model_config.use_ip_address else 0
        _user_end = _hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq + _user_seq
        embedding_slices = EmbeddingSlices(
            hist_post_end=_hist_post_seq,
            hist_auth_end=_hist_post_seq + _hist_auth_seq,
            cand_post_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq,
            cand_auth_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq,
            user_end=_user_end,
            user_ip_end=_user_end + _ip_seq,
        )

        def _packed_recsys_embeddings(
            merged_embeddings: jax.Array, sl: EmbeddingSlices
        ) -> RecsysEmbeddings:
            assert seqpack_packed_history_len is not None
            assert seqpack_packed_candidate_len is not None
            assert seqpack_bs_per_device is not None
            D = merged_embeddings.shape[0]
            M = merged_embeddings.shape[-1]
            return RecsysEmbeddings(
                history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :].reshape(
                    D, seqpack_packed_history_len, -1, M
                ),
                history_author_embeddings=merged_embeddings[
                    :, sl.hist_post_end : sl.hist_auth_end, :
                ].reshape(D, seqpack_packed_history_len, -1, M),
                candidate_post_embeddings=merged_embeddings[
                    :, sl.hist_auth_end : sl.cand_post_end, :
                ].reshape(D, seqpack_packed_candidate_len, -1, M),
                candidate_author_embeddings=merged_embeddings[
                    :, sl.cand_post_end : sl.cand_auth_end, :
                ].reshape(D, seqpack_packed_candidate_len, -1, M),
                user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :].reshape(
                    D, seqpack_bs_per_device, -1, M
                ),
                user_ip_embeddings=(
                    merged_embeddings[:, sl.user_end : sl.user_ip_end, :].reshape(
                        D, seqpack_bs_per_device, -1, M
                    )
                    if sl.user_ip_end > sl.user_end
                    else None
                ),
            )

        @hk.transform
        def two_tower_forward_fn(
            batch: RecsysFeaturesBatch,
            merged_embeddings: jax.Array,
            post_embeddings: jax.Array,
            dataset_types: jax.Array,
            large_k: int,
            target_dataset_types: tuple[int, ...],
            dataset_ranges: tuple[tuple[int, int], ...] | None = None,
        ):
            assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
            sl = embedding_slices

            if self.using_seqpack:
                recsys_embeddings = _packed_recsys_embeddings(merged_embeddings, sl)
            else:
                recsys_embeddings = RecsysEmbeddings(
                    history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :],
                    history_author_embeddings=merged_embeddings[
                        :, sl.hist_post_end : sl.hist_auth_end, :
                    ],
                    candidate_post_embeddings=merged_embeddings[
                        :, sl.hist_auth_end : sl.cand_post_end, :
                    ],
                    candidate_author_embeddings=merged_embeddings[
                        :, sl.cand_post_end : sl.cand_auth_end, :
                    ],
                    user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :],
                    user_ip_embeddings=(
                        merged_embeddings[:, sl.user_end : sl.user_ip_end, :]
                        if sl.user_ip_end > sl.user_end
                        else None
                    ),
                )
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            return model.forward(
                batch,
                recsys_embeddings,
                post_embeddings,
                dataset_types,
                large_k,
                target_dataset_types,
                dataset_ranges=dataset_ranges,
                use_async_topk=self.enable_async_topk,
                use_radix_select_topk=self.enable_radix_select_topk,
            )

        return JittedOrCompiled(
            jax.jit(
                two_tower_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                    self.data_sharding,
                    None,
                ),
                out_shardings=None,
                static_argnums=[6, 7, 8],
            ),
            name=f"two_tower_forward_fn_bs{bs}",
        )
