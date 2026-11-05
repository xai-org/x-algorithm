# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import collections
import concurrent.futures
import enum
import functools
import gc
import itertools
import json
import logging
import math
import operator
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import typing
from dataclasses import dataclass, field, replace
from typing import Iterator, NamedTuple, Sequence

import grpc
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import psutil
from jax import shard_map
from jax.experimental import multihost_utils
from jax.sharding import (
    NamedSharding,
    SingleDeviceSharding,
)
from jax.sharding import PartitionSpec as P
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

import xai_recsys_engine
from xai_checkpointing.common import _unsafe_jax2np
from xai_checkpointing.tree_util import tree_to_dict
from xai_configlib import configclass
from xai_proto import copy_pb2, copy_pb2_grpc
from xrex.data.parquet_recsys import DataPosition, PhoenixDataset, ShuffleMode
from xrex.data.parquet_recsys_metadata import parse_date_bound

if typing.TYPE_CHECKING:
    from xrex.cuda.async_emb import (
        async_emb,
    )

    AsyncEmbContextHandle: typing.TypeAlias = async_emb.AsyncEmbContextHandle
else:
    try:
        from xrex.cuda.async_emb import async_emb
    except ImportError:
        async_emb = None

    AsyncEmbContextHandle = typing.Any

from xrex.data.recsys.array_allocator import ArrayAllocator
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.data.recsys.sequence_packing import (
    PACK_BLOCK_SIZE,
    FixedLengthDistribution,
    LengthDistribution,
    candidate_region_history_schedule,
    candidate_tokens_per_user_needed,
    pack_batch,
)
from xrex.data.retrieval_dataset import RetrievalDataset
from xrex.data.rust_kafka_recsys import RustKafkaDataset
from xrex.data.streaming.kafkaloader import PhoenixKafkaDataset, report_training_metrics
from xrex.eval.eval_utils import report_forward_eval_results
from xrex.eval.metrics_recsys import merge_report_extras
from xrex.eval.recsys_eval import RecsysTwoTowerEval, run_recsys_evals
from xrex.models.compress_token_ids import compress_token_ids, compress_token_ids_by_search
from xrex.models.fp8 import Fp8Gemms, check_compiled_fp8_gemms
from xrex.models.model_utils import Parameter, unwrap_tree
from xrex.models.recsys_embedding import (
    EmbTable,
    RecsysEmbeddings,
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_model import (
    RecsysAggregatedModel,
    RecsysAggregatedModelConfig,
)
from xrex.models.recsys_two_tower_model import RecsysTwoTowerModel, scale_post_table_slots
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.optimizers.optim import InjectHyperparamsState, apply_updates, partition_grad_norms
from xrex.optimizers.recsys import RecsysEmbeddingOptimConfig
from xrex.optimizers.recsys.protocol import AsyncEmbOptimizer
from xrex.train import retrieval_corpus_stager as corpus_stager
from xrex.train.follow_data_end import DataEnd, DataEndFollower, KeyValueCoordination, agree
from xrex.train.misc import (
    CheckpointConfig,
    PostEmbeddings,
    RecsysTrainingState,
    replicated_global_array,
)
from xrex.train.trainer import (
    RecsysTwoTowerModelConfig,
    Trainer,
    TrainerContext,
)
from xrex.utils import recsys_async_emb
from xrex.utils.aot import JittedOrCompiled
from xrex.utils.checkpointing import wait_until_finished
from xrex.utils.coalesced_device_put import device_put_coalesced, precompile_device_put_coalesced
from xrex.utils.gpu import peak_tflops
from xrex.utils.metrics import norm_metrics
from xrex.utils.utils import (
    cast_bfloat16,
)

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


OUT_PATH = "/dev/shm"

_DATA_POSITION_FILENAME = "data_position.json"
_CONFIG_FILENAME = "config.json"
_FORWARD_ONLY_PROGRESS_FILENAME = "forward_only_progress.json"
_FORWARD_ONLY_PROGRESS_EVERY_S = 60.0


@dataclass(frozen=True)
class FetchedBatch:
    batch: typing.Any
    offsets: dict[int, int]
    data_position: typing.Any


@dataclass
class BatchPipelineState:
    current: FetchedBatch | None = None
    reserve: FetchedBatch | None = None
    exhausted: bool = False


class IncrementalState(NamedTuple):
    unique_tokens: npt.NDArray[np.int32]
    token_count: int
    soft_step: int
    checkpoint_path: str


class Store(enum.Enum):
    FS = 0
    S3 = 1
    GCS = 2

    @staticmethod
    def from_path(path):
        if path.startswith("gs://"):
            return Store.GCS
        if path.startswith("s3://"):
            return Store.S3
        return Store.FS


def path_to_dotted(path):
    parts = []
    for p in path:
        if isinstance(p, DictKey):
            parts.append(str(p.key))
        elif isinstance(p, GetAttrKey):
            parts.append(p.name)
        elif isinstance(p, SequenceKey):
            parts.append(str(p.idx))
    return ".".join(parts)


def pbroadcast(x, axis_name, source):
    axis_name = tuple(axis_name) if not isinstance(axis_name, tuple) else axis_name
    masked = jnp.where(jax.lax.axis_index(axis_name) == source, x, jnp.zeros_like(x))
    return jax.lax.psum(masked, axis_name)


_CANDIDATE_SIZE_AGREEMENT_TIMEOUT_MS = 2 * 60 * 60 * 1000
_LOOKUP_CHUNK_BYTES = 4 << 30


def _pad_axis(x: jax.Array, axis: int, length: int | None) -> jax.Array:
    if length is None or x.shape[axis] == length:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, length - x.shape[axis])
    return jnp.pad(x, pad)


def restart_from_restored_offsets(dataset: PhoenixKafkaDataset) -> bool:
    changed = False
    if dataset.fresh_training_bootstrap:
        dataset.fresh_training_bootstrap = False
        dataset.native_checkpoint_restart = True
        changed = True
    if dataset.seek_to_timestamp_ms is not None:
        dataset.seek_to_timestamp_ms = None
        changed = True
    return changed


def max_over_processes(client, key_prefix: str, round_: int, value: int, timeout_ms: int) -> int:
    num_processes = jax.process_count()
    if num_processes == 1:
        return value
    round_key = f"{key_prefix}/{round_}"
    client.key_value_set(f"{round_key}/values/{jax.process_index()}", str(value))
    if client.key_value_increment(f"{round_key}/arrived", 1) == num_processes:
        values = client.key_value_dir_get(f"{round_key}/values/")
        client.key_value_set(f"{round_key}/max", str(max(int(v) for _, v in values)))
        if round_ > 0:
            client.key_value_delete(f"{key_prefix}/{round_ - 1}")
    return int(client.blocking_key_value_get(f"{round_key}/max", timeout_ms))


def _num_users(data: RecsysFeaturesBatch) -> int:
    return math.prod(data["user_hashes"].shape[:-1])


def _step_token_ids(token_ids: Sequence[jax.Array], data_axis) -> jax.Array:
    return jax.lax.with_sharding_constraint(jnp.stack(token_ids), P(None, data_axis, None))


def _pad_vocab_for_ep(emb: jax.Array, ep: int, name: str = "emb_table") -> jax.Array:
    V = emb.shape[0]
    target_V = -(-V // ep) * ep
    if target_V == V:
        return emb
    rank_logger.info("Padded %s rows %d -> %d for ep=%d save", name, V, target_V, ep)
    return jnp.pad(emb, ((0, target_V - V),) + ((0, 0),) * (emb.ndim - 1))


@functools.partial(jax.jit, donate_argnums=(0,))
def _write_rows_in_place(buffer: jax.Array, rows: jax.Array, offset: jax.Array) -> jax.Array:
    return jax.lax.dynamic_update_slice_in_dim(buffer, rows, offset, axis=0)


def _int64_as_int32_pairs(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.int64)
    if sys.byteorder == "little":
        return values.view(np.int32).reshape(-1, 2)
    return np.stack(
        [(values & 0xFFFFFFFF).astype(np.int32), (values >> 32).astype(np.int32)], axis=1
    )


def _chunked_row_forward_on_device(
    sharding: jax.sharding.Sharding,
    host_inputs: typing.Sequence[np.ndarray],
    run_chunk: typing.Callable[..., jax.Array],
    max_chunk_rows: int,
    max_chunks_in_flight: int = 2,
) -> jax.Array | None:
    if not host_inputs:
        raise ValueError("host_inputs must be nonempty")
    if max_chunk_rows <= 0 or max_chunks_in_flight <= 0:
        raise ValueError("chunk size and in-flight limit must be positive")
    num_rows = host_inputs[0].shape[0]
    if any(x.shape[0] != num_rows for x in host_inputs):
        raise ValueError("host_inputs must share the leading dimension")
    num_local = len(sharding.addressable_devices)
    if num_rows == 0 or num_rows % num_local != 0:
        return None
    rows_per_device = num_rows // num_local
    if num_local > 1:
        probe_shape = (rows_per_device * len(sharding.device_set), *host_inputs[0].shape[1:])
        try:
            index_map = sharding.addressable_devices_indices_map(probe_shape)
        except (IndexError, ValueError):
            return None
        if len({idx[0].start or 0 for idx in index_map.values()}) != num_local:
            return None
    chunk_rows = max(1, min(rows_per_device, max_chunk_rows // num_local))

    buffers: list[jax.Array] = []
    in_flight: collections.deque[jax.Array] = collections.deque()
    global_rows = 0
    out_trailing: tuple[int, ...] = ()
    out_sharding = None
    for start in range(0, rows_per_device, chunk_rows):
        end = min(start + chunk_rows, rows_per_device)
        chunk_inputs = []
        for x in host_inputs:
            if num_local == 1:
                local = x[start:end]
            else:
                local = np.concatenate(
                    [
                        x[d * rows_per_device + start : d * rows_per_device + end]
                        for d in range(num_local)
                    ],
                    axis=0,
                )
            chunk_inputs.append(jax.make_array_from_process_local_data(sharding, local))
        out = run_chunk(*chunk_inputs)
        shards = sorted(out.addressable_shards, key=lambda s: s.index[0].start or 0)
        if not buffers:
            buffers = [
                jnp.zeros((rows_per_device, *s.data.shape[1:]), s.data.dtype, device=s.device)
                for s in shards
            ]
        offset = np.int32(start)
        for d, shard in enumerate(shards):
            buffers[d] = _write_rows_in_place(buffers[d], shard.data, offset)
        global_rows += out.shape[0]
        out_trailing = out.shape[1:]
        out_sharding = out.sharding
        in_flight.append(out)
        if len(in_flight) > max_chunks_in_flight:
            jax.block_until_ready(in_flight.popleft())
        del out, shards

    assert out_sharding is not None
    return jax.make_array_from_single_device_arrays(
        (global_rows, *out_trailing), out_sharding, buffers
    )


def _write_all_bytes(path: str, mv: memoryview, chunk_size: int = 1 << 30) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total_size = len(mv)
    written = 0
    with open(path, "wb", buffering=0) as f:
        while written < total_size:
            chunk_end = min(written + chunk_size, total_size)
            chunk = mv[written:chunk_end]
            while len(chunk) > 0:
                n = f.write(chunk)
                if n is None or n <= 0:
                    raise OSError(
                        f"failed writing checkpoint shard {path}: "
                        f"wrote {written} of {total_size} bytes"
                    )
                written += n
                chunk = chunk[n:]

    if written != total_size:
        raise OSError(
            f"incomplete write for checkpoint shard {path}: wrote {written} of {total_size} bytes"
        )


_CHECKSUM_PARTS_KV_DIR = "recsys-shm-checksum-parts"
_ROW_SHARDED_CHECKSUM_KEYS = ("emb_table", "post_embeddings.embeddings")


def _shard_position(value: jax.Array) -> tuple[int, int, int] | None:
    index = value.addressable_shards[0].index
    sharded_axes = [axis for axis, index_slice in enumerate(index) if index_slice != slice(None)]
    if not sharded_axes:
        return None
    axis = sharded_axes[0]
    width = index[axis].stop - index[axis].start
    return axis, index[axis].start // width, value.shape[axis] // width


def _local_checksums(
    write_items: Sequence[tuple[str, jax.Array, npt.NDArray]], checksum_replicated: bool
) -> tuple[dict[str, int], dict[str, list[int]]]:
    checksums, parts = {}, {}
    for key, value, arr in write_items:
        arr_bytes = np.ndarray(shape=arr.nbytes, dtype=np.uint8, buffer=arr)
        shard_position = _shard_position(value)
        if shard_position is None:
            if checksum_replicated:
                checksums[key] = xai_recsys_engine.adler32_parallel(arr_bytes)
        elif key in _ROW_SHARDED_CHECKSUM_KEYS:
            parts[key] = [xai_recsys_engine.adler32_parallel(arr_bytes), arr.nbytes]
        else:
            _, shard_index, _ = shard_position
            shard_row_bytes, row_bytes = arr.shape[1] * arr.itemsize, value.shape[1] * arr.itemsize
            parts[key] = xai_recsys_engine.adler32_shard_partial(
                arr_bytes,
                shard_row_bytes,
                shard_index,
                row_bytes,
                128,
            ).tolist()
    return checksums, parts


def _combine_checksums(
    write_items: Sequence[tuple[str, jax.Array, npt.NDArray]],
    checksums: dict[str, int],
    parts_by_process: Sequence[dict[str, list[int]]],
) -> dict[str, int]:
    combined = {}
    for key, value, _ in write_items:
        shard_position = _shard_position(value)
        if shard_position is None:
            combined[key] = checksums[key]
            continue
        _, _, num_shards = shard_position
        first_process_of_each_shard = parts_by_process[:: len(parts_by_process) // num_shards]
        shard_parts = [process_parts[key] for process_parts in first_process_of_each_shard]
        if key in _ROW_SHARDED_CHECKSUM_KEYS:
            checksum = shard_parts[0][0]
            for shard_checksum, length in shard_parts[1:]:
                checksum = xai_recsys_engine.adler32_combine_py(checksum, shard_checksum, length)
        else:
            checksum = xai_recsys_engine.adler32_combine_partials(
                np.array(shard_parts, dtype=np.uint32), value.nbytes
            )
        combined[key] = checksum
    return combined


_ALLOCATOR_ROUNDING_BYTES = 256
_ALLOCATOR_SPLIT_BYTES = 128 << 20
_ANCHOR_BYTES = 64 << 20
_SMALL_BUFFER_REGION_BYTES = 256 << 20


def _device_zeros(num_bytes: int, device: jax.Device) -> jax.Array:
    zeros = jax.jit(
        lambda: jnp.zeros((num_bytes,), jnp.uint8), out_shardings=SingleDeviceSharding(device)
    )
    return zeros().block_until_ready()


def _read_checkpoint_kafka_config(ctx) -> tuple[str | None, int | None]:
    if ctx.checkpoint is None:
        return None, None
    config_path = os.path.join(ctx.checkpoint.path, _CONFIG_FILENAME)
    if not os.path.isfile(config_path):
        return None, None
    try:
        with open(config_path) as f:
            config = json.load(f)
        dataset_config = config.get("dataset", config)
        topic = dataset_config.get("topic_name")
        partitions = dataset_config.get("num_kafka_partitions")
        return topic, partitions
    except (json.JSONDecodeError, OSError) as e:
        rank_logger.warning("Failed to read kafka config from %s: %s", config_path, e)
        return None, None


@configclass
class RecsysCheckpointConfig(CheckpointConfig):
    keep_emb_opt_state: bool = False


@configclass
class RecsysTrainer(Trainer):
    offsets_to_commit: dict[int, int] = field(default_factory=dict)
    store_load: str | None = None
    exclude_nodes_file: str | None = None
    enable_metrics_file: bool = False
    emb_optim_config: RecsysEmbeddingOptimConfig = field(default_factory=RecsysEmbeddingOptimConfig)

    stop_at_data_end: bool = False
    follow_data_end: bool = False
    follow_data_end_check_every_n: int = 200

    empty_history_augmentation_rate: float = 0.0

    empty_history_user_dropout_rate: float = 0.0

    split_home_checkpoint: bool = False

    export_stablehlo_bundle: bool = False
    export_bundle_bs_per_device: str = "1,2,4"
    export_bundle_history_seq_len: int = 0
    export_bundle_candidate_seq_len: int = 0

    smoothing_windows: list[int] = field(default_factory=lambda: [1_048_576, 4_194_304])

    reset_data_position: bool = False

    seqpack_distribution: LengthDistribution | None = None
    min_packed_history_len: int | None = None

    seqpack_empty_history_min_slot: bool = False
    seqpack_fixed_length: bool = False
    seqpack_candidate_packing: bool = False
    seqpack_candidate_count: list[int] = field(default_factory=list)

    use_async_emb: bool = False

    use_row_emb: bool = False
    row_emb_recv_factor: float = 2.0
    num_microbatch: int = 1

    forward_only: bool = False
    _forward_only_first_metrics_s: float | None = field(default=None, init=False, repr=False)
    _forward_only_progress_written_s: float | None = field(default=None, init=False, repr=False)

    data_prefetch_factor: int = 2

    _async_emb_context: AsyncEmbContextHandle | None = field(default=None, init=False, repr=False)
    _emb_hash_vocab: int = field(default=0, init=False, repr=False)
    _first_step_embedding_lookup_start_jit: typing.Any = field(default=None, init=False, repr=False)
    _async_emb_step_jits: dict[int, typing.Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _candidate_size_round: int = field(default=0, init=False, repr=False)
    _data_end_follower: DataEndFollower | None = field(default=None, init=False, repr=False)
    _async_emb_lookup_pin: jax.Array | None = field(default=None, init=False, repr=False)
    _allocator_anchors: list[jax.Array] = field(default_factory=list, init=False, repr=False)
    _batch_pipeline: BatchPipelineState = field(
        default_factory=BatchPipelineState, init=False, repr=False
    )

    _alloc: ArrayAllocator = field(default_factory=ArrayAllocator, init=False, repr=False)
    _data_position: DataPosition | None = field(default=None, init=False, repr=False)
    _seqpack_rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    _history_user_dropout_rng: np.random.Generator | None = field(
        default=None, init=False, repr=False
    )
    _last_history_user_dropout_count: int = field(default=0, init=False, repr=False)
    _last_history_user_dropout_bsz: int = field(default=0, init=False, repr=False)
    _retrieval_post_emb_built: bool = field(default=False, init=False, repr=False)
    retrieval_staged_corpus: bool = False
    retrieval_staged_corpus_dir: str = "/dev/shm/retrieval_corpus"
    retrieval_staged_corpus_poll_seconds: float = 30.0
    retrieval_post_table_chunk_rows: int = 65536
    _retrieval_corpus_stager_proc: subprocess.Popen | None = field(
        default=None, init=False, repr=False
    )
    _post_table_ids_generation: int | None = field(default=None, init=False, repr=False)
    _post_table_lookup_jit: typing.Any = field(default=None, init=False, repr=False)
    _eval_lookup_jit: typing.Any = field(default=None, init=False, repr=False)
    _eval_post_table: jax.Array | None = field(default=None, init=False, repr=False)
    dense_params_ema_decay: float = 0.0
    dense_params_ema_in_checkpoint: bool = False
    _dense_params_ema: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_init_jit: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_update_jit: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_gap_jit: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_as_params_jit: typing.Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.retrieval_post_table_chunk_rows <= 0:
            raise ValueError("retrieval_post_table_chunk_rows must be positive")
        if self.retrieval_staged_corpus_poll_seconds <= 0:
            raise ValueError("retrieval_staged_corpus_poll_seconds must be positive")
        if self.using_seqpack and self.seqpack_fixed_length:
            assert isinstance(self.model_config, RecsysAggregatedModelConfig)
            hl = self.model_config.history_seq_len
            self.seqpack_distribution = FixedLengthDistribution(min_len=hl, max_len=hl, mean_len=hl)
        if isinstance(self.model_config, RecsysAggregatedModelConfig) and isinstance(
            self.dataset, PhoenixDataset
        ):
            events = self.model_config.aux_value_events
            missing = [e for e in events if e not in self.dataset.aux_value_events]
            if missing:
                self.dataset.aux_value_events = (*self.dataset.aux_value_events, *missing)
        if self.export_stablehlo_bundle and not self.checkpoint_config.copy_port:
            raise ValueError("export_stablehlo_bundle requires checkpoint_config.copy_port")
        if self.forward_only and (
            self.use_async_emb or self.use_row_emb or self.num_microbatch > 1
        ):
            raise ValueError(
                "forward_only runs the synchronous step: set use_async_emb=False, "
                "use_row_emb=False and num_microbatch=1"
            )
        if self.follow_data_end:
            self._validate_follow_data_end()

    state: RecsysTrainingState = field(init=False, repr=False, compare=False)
    _pending_shmem_ckpt_write_s: float | None = field(default=None, init=False, repr=False)
    _pending_checksum_s: float | None = field(default=None, init=False, repr=False)
    _pending_gc_collect_s: float | None = field(default=None, init=False, repr=False)
    _stablehlo_bundle_files: list | None = field(default=None, init=False, repr=False)

    _engine = None
    _shmem_write_pool = None
    _shmem_write_future = None
    _incremental_state = None
    _last_disk_checkpoint_ts = 0.0

    train_dataset: typing.Any = field(init=False, repr=False, compare=False, default=None)
    _eval_dataset: typing.Any = field(init=False, repr=False, compare=False, default=None)
    _emb_optim: typing.Any = field(init=False, repr=False, compare=False, default=None)
    loss_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    loss_fn_eval: typing.Any = field(init=False, repr=False, compare=False, default=None)
    microbatch_loss_fns: typing.Any = field(init=False, repr=False, compare=False, default=None)
    forward_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    two_tower_forward_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    two_tower_forward_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    mol_side_table_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    mol_side_table_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    candidate_tower_forward_fn: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    state_shape: typing.Any = field(init=False, repr=False, compare=False, default=None)
    stats_shape: typing.Any = field(init=False, repr=False, compare=False, default=None)
    state_sharding: typing.Any = field(init=False, repr=False, compare=False, default=None)
    data_sharding: typing.Any = field(init=False, repr=False, compare=False, default=None)
    init_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    update_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    forward_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    candidate_tower_forward_jit: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    host_state: typing.Any = field(init=False, repr=False, compare=False, default=None)

    def create_runtime(self, ctx):
        if os.environ.get("JAX_COORDINATOR_PORT"):
            ctx.coordinator_port = int(os.environ["JAX_COORDINATOR_PORT"])
        super().create_runtime(ctx)

    def example_data(
        self, bs: int, candidate_tokens_per_user: int | None = None
    ) -> RecsysFeaturesBatch:
        assert isinstance(self.dataset, PhoenixDataset)
        batch = self.dataset.example_data(bs)
        if self.using_seqpack:
            batch = self._pack_batch(batch, np.random.default_rng(0), candidate_tokens_per_user)
        return batch

    def prepare_data(self, batch, candidate_tokens_per_user: int | None = None):
        return super().prepare_data(self.transform_batch(batch, candidate_tokens_per_user))

    def transform_batch(
        self, batch: RecsysFeaturesBatch, candidate_tokens_per_user: int | None = None
    ) -> RecsysFeaturesBatch:
        if self.using_seqpack:
            batch = self._pack_batch(batch, self._seqpack_rng, candidate_tokens_per_user)

        def narrow_to_32bit(x):
            if not isinstance(x, np.ndarray):
                return x
            dtype = jax.dtypes.canonicalize_dtype(x.dtype)
            return x if dtype == x.dtype else self._alloc.copy(x, dtype)

        return jax.tree.map(narrow_to_32bit, batch)

    def _total_candidate_seq_len(self) -> int:
        assert isinstance(self.dataset, PhoenixDataset)
        num_neg_blocks = 2 if self.dataset.search_query_embedding_dim > 0 else 1
        return (
            self.dataset.candidate_seq_len
            * (1 + num_neg_blocks * self.dataset.num_negatives_per_example)
            + self.dataset.num_global_negatives_per_example
        )

    def _history_mean_len(self, candidate_tokens_per_user: int) -> int:
        assert self.seqpack_distribution is not None
        return self.seqpack_distribution.mean_len - (
            candidate_tokens_per_user - self.seqpack_candidate_count[0]
        )

    def _history_lookup_len(self, hashes_per_token: int) -> int | None:
        if not self.seqpack_candidate_packing:
            return None
        assert self.seqpack_distribution is not None
        return self.seqpack_distribution.mean_len * hashes_per_token

    def _step_candidate_tokens_per_user(
        self, microbatches: list[RecsysFeaturesBatch]
    ) -> int | None:
        from jax._src.distributed import global_state as jax_distributed

        if not self.seqpack_candidate_packing:
            return None
        if len(self.seqpack_candidate_count) == 1:
            return self.seqpack_candidate_count[0]
        num_devices = self.parallel_config.num_devices_per_process
        needed = max(candidate_tokens_per_user_needed(b, num_devices) for b in microbatches)
        needed = max_over_processes(
            jax_distributed.client,
            "seqpack_candidate_tokens",
            self._candidate_size_round,
            needed,
            _CANDIDATE_SIZE_AGREEMENT_TIMEOUT_MS,
        )
        self._candidate_size_round += 1
        return next(
            (size for size in self.seqpack_candidate_count if size >= needed),
            self.seqpack_candidate_count[-1],
        )

    def _pack_batch(
        self,
        batch: RecsysFeaturesBatch,
        rng: np.random.Generator | None,
        candidate_tokens_per_user: int | None,
    ) -> RecsysFeaturesBatch:
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        dist = self.seqpack_distribution
        if self.seqpack_candidate_packing:
            candidate_tokens_per_user = candidate_tokens_per_user or self.seqpack_candidate_count[0]
            assert dist is not None
            dist = replace(dist, mean_len=self._history_mean_len(candidate_tokens_per_user))
        batch = pack_batch(
            batch=batch,
            num_devices_per_process=self.parallel_config.num_devices_per_process,
            num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
            dist=dist,
            rng=rng,
            min_packed_history_len=self.min_packed_history_len if dist is not None else None,
            candidate_tokens_per_user=candidate_tokens_per_user,
            empty_history_min_slot=self.seqpack_empty_history_min_slot,
            alloc=self._alloc,
        )
        if self.using_fa4:
            batch = self.add_block_sparse_layout(batch)
        return batch

    def _value_head_ema_keys(self) -> dict[str, jax.Array]:
        if not isinstance(self.model_config, RecsysAggregatedModelConfig):
            return {}
        return self.model_config.value_head_ema_keys()

    def _metric_ema_cells(self) -> list[tuple[str, str]]:
        assert isinstance(self.model_config, RecsysAggregatedModelConfig)
        from xrex.data.recsys.ads_head_masking import metric_head_mask_keys
        from xrex.data.recsys.constants import engagement_to_ids
        from xrex.models.recsys_model import build_metric_masks

        dummy = jnp.zeros((1, 1))
        dummy_3d = jnp.zeros((1, 1, self.model_config.model_config.output_vocab_size))
        mask_keys = list(
            build_metric_masks(
                dummy,
                dummy_3d,
                dummy,
                dummy,
                trained_candidate_mask=dummy_3d,
                ads_head_masking=self.model_config.ads_head_masking,
                enable_platform_metrics=self.model_config.enable_platform_metrics,
                metric_mask_keys=self.model_config.metric_mask_keys,
            ).keys()
        )
        head_mask_keys = metric_head_mask_keys(
            engagement_to_ids(self.model_config.metric_group),
            mask_keys,
            own_slice=self.model_config.metric_own_slice,
        )
        return [(e, m) for e, ms in head_mask_keys.items() for m in ms]

    def _conversion_delay_slice_ema_keys(self, state_size: int) -> dict[str, jax.Array]:
        if not isinstance(self.model_config, RecsysAggregatedModelConfig):
            return {}
        return {
            f"{s.head}/{s.name}/{ws}": jnp.zeros((state_size,), dtype=jnp.float32)
            for s in self.model_config.conversion_delay_slices()
            for ws in self.smoothing_windows
        }

    def init(self, batch: RecsysFeaturesBatch, rng: jax.Array) -> RecsysTrainingState:
        assert isinstance(self.dataset, PhoenixDataset)
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        if self.using_seqpack:
            assert self.seqpack_distribution is not None
            assert self.seqpack_distribution.max_len == self.dataset.history_seq_len
            self._seqpack_rng = np.random.default_rng(self.rng_seed)

        rng, init_rng = jax.random.split(rng)
        if self.rng_seed == 0:
            init_rng = jax.random.PRNGKey(0), [jax.random.PRNGKey(0)] * 1000

        is_device = self.model_config.get_embed_memory_kind() == "device"
        emb_table, emb_table_state = self.model_config.make_embedding_table(
            init_rng,
            self.optim,
            self.dataset.input_vocab_size if is_device else 1,
            self.init_opt_state,
        )
        if self.use_row_emb:
            rows = (
                math.ceil(emb_table.x.shape[0] / self.parallel_config.ep) * self.parallel_config.ep
            )
            emb_table = replace(
                emb_table,
                x=jnp.pad(emb_table.x, ((0, rows - emb_table.x.shape[0]), (0, 0))),
                pspec=P("expert", None),
            )
        if self.emb_optim_config._active() == "rowwise_adagrad":
            sparse_emb_optim = self.emb_optim_config.make_optimizer(self.optim)
            emb_table_state = sparse_emb_optim.init({"table": emb_table})
            if self.use_row_emb:
                logical_rows = self.dataset.input_vocab_size
                emb_table_state = emb_table_state._replace(
                    row_sum_sq={
                        "table": emb_table_state.row_sum_sq["table"].at[logical_rows:].set(0)
                    }
                )
        emb_table = replace(emb_table, x=emb_table.x.at[0, :].set(0))
        post_embeddings = PostEmbeddings(
            post_ids=jnp.arange(1).astype(jnp.int32),
            author_ids=jnp.arange(1).astype(jnp.int32),
            embeddings=Parameter(x=jnp.empty((1, 1), dtype=jnp.bfloat16), pspec=P(None, None)),
            dataset_types=jnp.array([RetrievalDataset.PAD.value], dtype=jnp.int32),
        )
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            post_embeddings = self.model_config.candidate_tower_config.make_post_embeddings(
                self.model_config.num_table_slots
            )
            side_width = self.model_config.mol_side_table_width
            if side_width:
                emb = post_embeddings.embeddings
                post_embeddings = post_embeddings._replace(
                    mol_side_table=Parameter(
                        x=jnp.zeros((emb.x.shape[0], side_width), dtype=jnp.float32),
                        pspec=P(None, None),
                    )
                )

        packing_layout = batch.get("packing_layout")
        batch = jax.tree.map(jnp.ones_like, batch)
        batch["packing_layout"] = packing_layout

        recsys_embeddings = self.get_recsys_embeddings(batch, emb_table)

        initial_params = self.loss_fn.init(init_rng, batch, recsys_embeddings)

        if self.init_opt_state:
            initial_opt_state = self.optim.init(initial_params)
        else:
            initial_opt_state = None

        if self.precision_level == 0:
            initial_params = jax.tree.map(cast_bfloat16, initial_params)
        if self.precision_level < 2:
            initial_opt_state = jax.tree.map(cast_bfloat16, initial_opt_state)

        if isinstance(self.dataset, PhoenixKafkaDataset):
            self.dataset.ensure_partition_count()
        rep = self.dataset.num_kafka_partitions or 1

        rce_ema = None
        cells: list[tuple[str, str]] = []
        if self.smoothing_windows and isinstance(self.model_config, RecsysAggregatedModelConfig):
            cells = self._metric_ema_cells()
            rce_ema = {
                f"{e}/{m}/{ws}": jnp.zeros((3,), dtype=jnp.float32)
                for e, m in cells
                for ws in self.smoothing_windows
            }
            rce_ema.update(self._value_head_ema_keys())
            rce_ema.update(self._conversion_delay_slice_ema_keys(3))

        calib_ema = None
        if rce_ema is not None:
            calib_ema = {
                f"{e}/{m}/{ws}": jnp.zeros((2,), dtype=jnp.float32)
                for e, m in cells
                for ws in self.smoothing_windows
            }
            calib_ema.update(self._conversion_delay_slice_ema_keys(2))

        state = RecsysTrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            rng=rng,
            step=jnp.array(0),
            emb_table=emb_table,
            emb_table_state=emb_table_state,
            offset_keys=jnp.array([-1] * rep),
            offset_values=jnp.array([-1] * rep),
            post_embeddings=post_embeddings,
            rce_ema=rce_ema,
            calib_ema=calib_ema,
        )
        return state

    def _lookup(self, embedding_table_param: Parameter, token_ids: jax.Array) -> Parameter:
        data_axis = tuple(self.model_config.model_config.data_axis)
        token_ndim = token_ids.ndim
        if self.use_row_emb:
            token_ids = self._row_embedding_token_ids(token_ids)
            ep = self.mesh.shape["expert"]
            bytes_per_user = (
                ep
                * math.prod(token_ids.shape[1:])
                * embedding_table_param.x.shape[1]
                * embedding_table_param.x.dtype.itemsize
            )
            users_per_chunk = max(1, _LOOKUP_CHUNK_BYTES // bytes_per_user)

            @shard_map(
                mesh=self.mesh,
                in_specs=(P("expert", None), P(data_axis)),
                out_specs=P(data_axis, *((None,) * token_ndim)),
                check_vma=False,
            )
            def lookup_rows(table, ids):
                ids = jax.lax.all_gather(ids, "expert", axis=0)
                local_ids = ids - jax.lax.axis_index("expert") * table.shape[0]
                users = local_ids.shape[1]
                chunk = min(users_per_chunk, users)
                num_chunks = -(-users // chunk)
                pad = [(0, 0), (0, num_chunks * chunk - users)] + [(0, 0)] * (local_ids.ndim - 2)
                local_ids = jnp.pad(local_ids, pad, constant_values=-1)
                chunks = jnp.moveaxis(
                    local_ids.reshape(ep, num_chunks, chunk, *local_ids.shape[2:]), 1, 0
                )

                def one(chunk_ids):
                    values = jnp.take(table, chunk_ids, axis=0, mode="clip")
                    values = jnp.where(
                        ((chunk_ids >= 0) & (chunk_ids < table.shape[0]))[..., None], values, 0
                    )
                    return jax.lax.psum_scatter(values, "expert", scatter_dimension=0)

                out = jax.lax.map(one, chunks)
                return out.reshape(num_chunks * chunk, *out.shape[2:])[:users]

            return replace(embedding_table_param, x=lookup_rows(embedding_table_param.x, token_ids))

        in_token_spec = P(data_axis, *((None,) * (token_ndim - 1)))
        out_spec = P(data_axis, *((None,) * token_ndim))

        @shard_map(
            mesh=self.mesh,
            in_specs=(P(None, "expert"), in_token_spec),
            out_specs=out_spec,
        )
        def _gather(table: jax.Array, ids: jax.Array) -> jax.Array:
            bs_per_device = ids.shape[0]

            all_ids = jax.lax.all_gather(ids, axis_name="expert", axis=0, tiled=True)
            ep = all_ids.shape[0] // bs_per_device

            result = table[all_ids]

            result = result.reshape(ep, bs_per_device, *result.shape[1:])

            result = jax.lax.all_to_all(
                result,
                axis_name="expert",
                split_axis=0,
                concat_axis=result.ndim - 1,
                tiled=True,
            )

            return result.reshape(result.shape[1:])

        new_x = _gather(embedding_table_param.x, token_ids)
        return replace(embedding_table_param, x=new_x)

    def _embedding_hash_leaves(self, data: RecsysFeaturesBatch) -> list[tuple[jax.Array, bool]]:
        use_ip = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_ip_address
        )
        use_user_embedding = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_user_embedding
        )
        use_post_embedding = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_post_embedding
        )
        leaves: list[tuple[jax.Array, bool]] = []
        if use_user_embedding:
            leaves.append((data["user_hashes"], False))
        if use_post_embedding:
            leaves.append((data["history_seq"]["post_hashes"], True))
        leaves.append((data["history_seq"]["auth_hashes"], True))
        if use_post_embedding:
            leaves.append((data["candidate_seq"]["post_hashes"], False))
        leaves.append((data["candidate_seq"]["auth_hashes"], False))
        if use_ip:
            leaves.append((data["user_ip_hashes"], False))
        return leaves

    def _lookup_hash_ids(self, data: RecsysFeaturesBatch) -> jax.Array:
        users = _num_users(data)
        return jnp.concatenate(
            [
                _pad_axis(x.reshape(users, -1), 1, self._history_lookup_len(x.shape[-1]))
                if is_history
                else x.reshape(users, -1)
                for x, is_history in self._embedding_hash_leaves(data)
            ],
            axis=1,
        )

    def get_recsys_embeddings(
        self, data: RecsysFeaturesBatch, emb_table: Parameter
    ) -> RecsysEmbeddingsParameter:
        data_axis = tuple(self.model_config.model_config.data_axis)

        all_hashes = jax.lax.with_sharding_constraint(self._lookup_hash_ids(data), P(data_axis))
        all_embeddings = self._lookup(emb_table, all_hashes)
        all_embeddings = replace(
            all_embeddings,
            x=jax.lax.with_sharding_constraint(all_embeddings.x, P(data_axis, None, None)),
        )
        return self._unflatten_emb_lookup(data, all_embeddings)

    def _unflatten_emb_lookup(
        self, data: RecsysFeaturesBatch, table: Parameter
    ) -> RecsysEmbeddingsParameter:
        cfg = self.model_config
        has_emb_flags = isinstance(cfg, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
        leaves = self._embedding_hash_leaves(data)
        hash_leaves = [x for x, _ in leaves]
        lengths = [x.reshape(_num_users(data), -1).shape[1] for x in hash_leaves]
        padded_lengths = [
            (self._history_lookup_len(x.shape[-1]) if is_history else None) or length
            for (x, is_history), length in zip(leaves, lengths, strict=True)
        ]
        splits = jnp.split(table.x, np.cumsum(padded_lengths[:-1]), axis=-2)
        splits = [split[..., :length, :] for split, length in zip(splits, lengths, strict=True)]

        if self.using_seqpack:
            splits = [
                s.reshape(*s.shape[:-3], *leaf.shape, s.shape[-1])
                for s, leaf in zip(splits, hash_leaves)
            ]

        parts = iter(splits)
        segment = lambda on: replace(table, x=next(parts)) if on else None

        return RecsysEmbeddingsParameter(
            user_embeddings=segment(has_emb_flags and cfg.use_user_embedding),
            history_post_embeddings=segment(has_emb_flags and cfg.use_post_embedding),
            history_author_embeddings=segment(True),
            candidate_post_embeddings=segment(has_emb_flags and cfg.use_post_embedding),
            candidate_author_embeddings=segment(True),
            user_ip_embeddings=segment(has_emb_flags and cfg.use_ip_address),
        )

    def _flatten_emb_grads(self, grads: RecsysEmbeddingsParameter, users: int) -> jax.Array:
        cfg = self.model_config
        has_emb_flags = isinstance(cfg, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
        enabled = [
            (grads.user_embeddings, has_emb_flags and cfg.use_user_embedding, False),
            (grads.history_post_embeddings, has_emb_flags and cfg.use_post_embedding, True),
            (grads.history_author_embeddings, True, True),
            (grads.candidate_post_embeddings, has_emb_flags and cfg.use_post_embedding, False),
            (grads.candidate_author_embeddings, True, False),
            (grads.user_ip_embeddings, has_emb_flags and cfg.use_ip_address, False),
        ]
        segments = [(p.x, is_history) for p, on, is_history in enabled if on and p is not None]
        width = segments[0][0].shape[-1]
        return jnp.concatenate(
            [
                _pad_axis(
                    x.reshape(x.shape[0], users, -1, width),
                    2,
                    self._history_lookup_len(x.shape[-2]) if is_history else None,
                )
                for x, is_history in segments
            ],
            axis=2,
        )

    def local_global_hack(self, batch):
        batch = multihost_utils.global_array_to_host_local_array(
            batch,
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )

        batch = multihost_utils.host_local_array_to_global_array(
            batch,
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )
        return batch

    def _empty_history_augmentation_period(self) -> int:
        rate = self.empty_history_augmentation_rate
        return max(1, int(1.0 / rate)) if rate > 0 else 0

    def _apply_per_history_user_dropout(self, batch: RecsysFeaturesBatch, rate: float) -> int:
        if self._history_user_dropout_rng is None:
            self._history_user_dropout_rng = np.random.default_rng(
                self.rng_seed + self.data_rank + 1
            )

        bsz = batch["user_hashes"].shape[0]
        drop_mask = self._history_user_dropout_rng.random(bsz) < rate
        n_dropped = int(drop_mask.sum())
        if n_dropped == 0:
            return 0

        for arr in typing.cast("dict[str, np.ndarray | None]", batch["history_seq"]).values():
            if arr is not None:
                arr[drop_mask] = 0
        if batch.get("user_hashes") is not None:
            batch["user_hashes"][drop_mask] = 0
        if batch.get("user_ip_hashes") is not None:
            batch["user_ip_hashes"][drop_mask] = 0
        return n_dropped

    def _precompile_batch_transfer(self) -> None:
        if len(self.mesh.local_devices) != 1:
            return
        microbatch_size = self.read_bsz_per_process // self.num_microbatch
        candidate_sizes = self.seqpack_candidate_count if self.seqpack_candidate_packing else [None]
        for candidate_tokens_per_user in candidate_sizes:
            microbatch = self.example_data(microbatch_size, candidate_tokens_per_user)
            precompile_device_put_coalesced(
                [microbatch] * self.num_microbatch, self.mesh.local_devices[0]
            )

    def dataset_with_prepare(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        assert isinstance(self.dataset, PhoenixDataset)
        aug_period = self._empty_history_augmentation_period()
        history_user_drop_rate = self.empty_history_user_dropout_rate
        microbatches: list[tuple[RecsysFeaturesBatch, dict[int, int] | None]] = []
        for i, (batch, offsets) in enumerate(dataset):
            if history_user_drop_rate > 0.0:
                if i % self.num_microbatch == 0:
                    self._last_history_user_dropout_bsz = 0
                    self._last_history_user_dropout_count = 0
                self._last_history_user_dropout_bsz += batch["user_hashes"].shape[0]
                self._last_history_user_dropout_count += self._apply_per_history_user_dropout(
                    batch, history_user_drop_rate
                )

            microbatches.append((batch, offsets))
            if len(microbatches) == self.num_microbatch:
                candidate_tokens_per_user = self._step_candidate_tokens_per_user(
                    [microbatch for microbatch, _ in microbatches]
                )
                packed_batches = [
                    self.transform_batch(microbatch, candidate_tokens_per_user)
                    for microbatch, _ in microbatches
                ]
                if len(self.mesh.local_devices) == 1 and all(
                    isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(packed_batches)
                ):
                    packed_batches = device_put_coalesced(
                        packed_batches, self.mesh.local_devices[0], self._alloc.empty
                    )
                for packed, (_, packed_offsets) in zip(packed_batches, microbatches, strict=True):
                    yield self.local_global_hack(super().prepare_data(packed)), packed_offsets
                microbatches = []

            if aug_period > 0 and i % aug_period == 0:
                for key in list(batch["history_seq"]):
                    if batch["history_seq"][key] is not None:
                        batch["history_seq"][key] = np.zeros_like(batch["history_seq"][key])
                if batch.get("user_hashes") is not None:
                    batch["user_hashes"] = np.zeros_like(batch["user_hashes"])
                if batch.get("user_ip_hashes") is not None:
                    batch["user_ip_hashes"] = np.zeros_like(batch["user_ip_hashes"])
                yield (
                    self.local_global_hack(self.prepare_data(batch, candidate_tokens_per_user)),
                    offsets,
                )

    def dataset_without_prepare(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        assert isinstance(self.dataset, PhoenixDataset)
        for batch, offsets in dataset:
            if self.using_seqpack:
                batch = self._pack_batch(batch, np.random.default_rng(0), None)
            yield batch, offsets

    def dataset_thread_iterator(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
        stop=object(),
        prepare_data=True,
    ):
        assert isinstance(self.dataset, PhoenixDataset)
        if prepare_data:
            it = self.dataset_with_prepare(dataset)
            if self.use_async_emb:
                it = (
                    (tuple(batch for batch, _ in group), group[-1][1])
                    for group in itertools.batched(it, self.num_microbatch, strict=False)
                    if len(group) == self.num_microbatch
                )
        else:
            it = self.dataset_without_prepare(dataset)
        future = self.dataloading_thread.submit(next, it, stop)
        while True:
            result = future.result()
            if result is stop:
                return
            assert isinstance(result, tuple)
            future = self.dataloading_thread.submit(next, it, stop)

            batch, offsets = result

            if offsets:
                self.offsets_to_commit = offsets

            yield batch

    def _report_save_timing(self, metrics) -> None:
        if (
            self._pending_checksum_s is not None
            or self._pending_shmem_ckpt_write_s is not None
            or self._pending_gc_collect_s is not None
        ):
            ckpt_s = self._pending_checksum_s
            shmem_s = self._pending_shmem_ckpt_write_s
            gc_s = self._pending_gc_collect_s
            rank_logger.info(
                "save_checkpoint timings: background_checksum=%.2fs, shmem_write=%.2fs, "
                "gc_collect=%.2fs",
                ckpt_s or 0,
                shmem_s or 0,
                gc_s or 0,
            )
            if ckpt_s is not None:
                metrics["save_checkpoint/background_checksum_s"] = ckpt_s
                self._pending_checksum_s = None
            if shmem_s is not None:
                metrics["save_checkpoint/background_shmem_write_s"] = shmem_s
                self._pending_shmem_ckpt_write_s = None
            if gc_s is not None:
                metrics["save_checkpoint/gc_collect_s"] = gc_s
                self._pending_gc_collect_s = None

    def handle_metrics(
        self, soft_step: int, metrics: dict[str, float | int]
    ) -> dict[str, float | int]:
        metrics = super().handle_metrics(soft_step, metrics)

        if self.forward_only:
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            num_seq_per_sec_per_device = (
                self.batch_size / float(metrics["step_time"]) / self.num_devices
            )
            metrics["tflops"] = self.model_config.compute_forward_tflops(num_seq_per_sec_per_device)
            metrics["mfu"] = metrics["tflops"] / peak_tflops()

        if self.empty_history_user_dropout_rate > 0.0 and self._last_history_user_dropout_bsz > 0:
            metrics["train/empty_history_user_dropout_frac"] = (
                self._last_history_user_dropout_count / self._last_history_user_dropout_bsz
            )

        self._report_save_timing(metrics)

        if isinstance(self.dataset, PhoenixKafkaDataset) and self.dataset.export_metrics:
            report_training_metrics(
                training_name=self.dataset.name,
                metrics_dict=metrics,
            )

        if self.forward_only:
            self._maybe_write_forward_only_progress(int(metrics["step"]))

        return metrics

    def forward_only_progress_path(self) -> str:
        from xrex.utils.metadata import checkpoint_dir_or_default

        ckpt_dir = checkpoint_dir_or_default(self.checkpoint_config.checkpoint_dir)
        return os.path.join(ckpt_dir, self.name, _FORWARD_ONLY_PROGRESS_FILENAME)

    def _maybe_write_forward_only_progress(self, step: int) -> None:
        now = time.monotonic()
        if self._forward_only_first_metrics_s is None:
            self._forward_only_first_metrics_s = now
            return
        last = self._forward_only_progress_written_s or self._forward_only_first_metrics_s
        if now - last < _FORWARD_ONLY_PROGRESS_EVERY_S:
            return
        path = self.forward_only_progress_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(
                {
                    "step": step,
                    "elapsed_samples": self.elapsed_samples,
                    "updated_at_unix": time.time(),
                },
                f,
            )
        os.replace(tmp_path, path)
        self._forward_only_progress_written_s = now

    def _resume_forward_only(self, res: tuple[bool, int, int]) -> tuple[bool, int, int]:
        path = self.forward_only_progress_path()
        buf = np.zeros(1024, dtype=np.uint8)
        if jax.process_index() == 0 and os.path.isfile(path):
            with open(path, "rb") as f:
                raw = f.read()
            buf[: len(raw)] = list(raw)
        buf = multihost_utils.broadcast_one_to_all(buf)
        raw_progress = bytes(np.asarray(buf, dtype=np.uint8)).rstrip(b"\x00").decode("utf-8")
        if not raw_progress:
            return res
        progress = json.loads(raw_progress)

        if isinstance(self.dataset, PhoenixKafkaDataset):
            self.dataset.seek_to_offset = None
        assert isinstance(self.state, RecsysTrainingState)
        rce_ema = self.state.rce_ema
        calib_ema = self.state.calib_ema
        self.state = self.state._replace(
            step=jnp.array(int(progress["step"]) + 1),
            rce_ema=None if rce_ema is None else jax.tree.map(jnp.zeros_like, rce_ema),
            calib_ema=None if calib_ema is None else jax.tree.map(jnp.zeros_like, calib_ema),
        )
        rank_logger.info(
            "forward_only restart: resuming from the consumer group's committed offsets "
            "at step=%s elapsed_samples=%s (%s)",
            progress["step"],
            progress["elapsed_samples"],
            path,
        )
        return res[0], int(progress["elapsed_samples"]), res[2]

    def _validate_follow_data_end(self) -> None:
        dataset = self.dataset
        needs: list[str] = []
        if not self.stop_at_data_end:
            needs.append("stop_at_data_end=True")
        if self.follow_data_end_check_every_n <= 0:
            needs.append("follow_data_end_check_every_n > 0")
        if not isinstance(dataset, PhoenixDataset) or not dataset.supports_follow_data_end():
            needs.append("the Python offline reader (PhoenixDataset)")
        else:
            if dataset.path is None:
                needs.append("dataset.path")
            if dataset.is_eval or dataset.continuous:
                needs.append("a training dataset with continuous=False")
            if dataset.shuffle_mode != ShuffleMode.BLOCK_SHUFFLE:
                needs.append("shuffle_mode=block_shuffle")
            if (
                dataset.date_range is not None
                and parse_date_bound(dataset.date_range[1]) is not None
            ):
                needs.append("an open-ended date_range (date_range=<start>,none)")
        if needs:
            raise ValueError("follow_data_end needs " + ", ".join(needs))

    def _data_end_step(self, step: int, data_end: int) -> int:
        remaining_batches = data_end - step + 1
        return step + remaining_batches // self.num_microbatch - 1

    def _start_follow_data_end(
        self,
        num_shards: int,
        loader_batch_size: int,
        resume_position: DataPosition | None,
    ) -> None:
        from jax._src.distributed import global_state as jax_distributed

        dataset = self.dataset
        assert isinstance(dataset, PhoenixDataset)
        assert isinstance(self.state, RecsysTrainingState)
        step = self.state.step.item()
        configured_max_steps = self.max_steps
        start_batch_id: int | None = None

        def probe(after_end_batch_id: int) -> DataEnd | None:
            nonlocal start_batch_id
            end = dataset.available_end_batch_id(after_end_batch_id)
            if end is None:
                return None
            if start_batch_id is None:
                start_batch_id = dataset.start_batch_id()
            data_end = dataset.compute_max_steps(
                num_shards=num_shards,
                batch_size=loader_batch_size,
                current_step=step,
                resume_position=resume_position,
                end_batch_id=end,
                start_batch_id=start_batch_id,
            )
            if data_end is None:
                raise RuntimeError(f"could not read the dataset index under {dataset.path}")
            max_steps = self._data_end_step(step, data_end)
            if configured_max_steps is not None:
                max_steps = min(max_steps, configured_max_steps)
            return DataEnd(end_batch_id=end, max_steps=max_steps)

        is_process_0 = jax.process_index() == 0
        coordination = KeyValueCoordination(
            jax_distributed.client, jax.process_index(), jax.process_count()
        )

        def first_count() -> dict:
            start = probe(-1)
            if start is None:
                raise ValueError(f"follow_data_end: no readable batch ids under {dataset.path}")
            return {"end_batch_id": start.end_batch_id, "max_steps": start.max_steps}

        start = DataEnd(
            **agree(coordination, "follow_data_end/start", 0, first_count if is_process_0 else None)
        )

        dataset.set_end_batch_id(start.end_batch_id)
        self.max_steps = start.max_steps
        rank_logger.info(
            "follow_data_end: starting with the data end at batch_id %d, max_steps=%d; "
            "looking for new data every %d steps",
            start.end_batch_id,
            start.max_steps,
            self.follow_data_end_check_every_n,
        )
        self._data_end_follower = DataEndFollower(
            data_end=start,
            window=dataset.shuffle_window_time_slices,
            check_every_n=self.follow_data_end_check_every_n,
            is_process_0=is_process_0,
            probe=probe,
            coordination=coordination,
        )

    def before_stop_check(self, soft_step: int) -> None:
        follower = self._data_end_follower
        if follower is None:
            return
        assert isinstance(self.dataset, PhoenixDataset)
        data_end = follower.check(soft_step, self.dataset.reader_window_start())
        if data_end is None:
            return
        self.dataset.set_end_batch_id(data_end.end_batch_id)
        self.max_steps = data_end.max_steps

    def create_dataset(self, ctx: TrainerContext):
        assert isinstance(self.dataset, PhoenixDataset)
        if self.use_mock_gpu_client:
            logger.info("Using mock GPU client. Skipping dataset creation.")
            return

        data_rank, data_world_size = self.data_rank, self.data_world_size
        elapsed_samples = self.elapsed_samples
        if ctx.dataset_metadata_fn is not None:
            dataset_metadata = ctx.dataset_metadata_fn(self.data_rank, self.data_world_size)
            data_rank = dataset_metadata.data_rank
            data_world_size = dataset_metadata.data_world_size
            elapsed_samples = dataset_metadata.elapsed_samples

        if isinstance(self.dataset, RustKafkaDataset) and self.dataset.redistribution_reader_count(
            data_world_size
        ):
            if self.empty_history_augmentation_rate > 0:
                raise ValueError("Redistribution requires one consumed batch per training step")
        if (
            isinstance(self.dataset, PhoenixKafkaDataset)
            and ctx.rank == ctx.world_size - 1
            and not self.dataset.exports_python_otel(data_rank, data_world_size)
        ):
            raise ValueError(
                f"python_otel_on_last_shard_only: rank {ctx.rank} reports training metrics from "
                f"shard {data_rank} of {data_world_size}, which starts no Python OTel exporter"
            )

        assert elapsed_samples >= self.dataloader_offset, (
            f"elapsed_samples {elapsed_samples} < dataloader_offset {self.dataloader_offset}"
        )

        resume_position = self._data_position
        self._data_position = None
        should_reset_data = (
            self.reset_data_position
            and ctx.checkpoint is not None
            and ctx.checkpoint.is_manual_load()
        )
        if should_reset_data:
            resume_position = None
            rank_logger.info(
                "create_dataset: reset_data_position=True with manual load, ignoring saved data position"
            )
        rank_logger.info("create_dataset: resume_position=%s", resume_position)

        skip_rows = 0 if should_reset_data else max(0, elapsed_samples - self.dataloader_offset)
        loader_batch_size = self.read_bsz_per_process // self.num_microbatch
        if self.follow_data_end:
            self._start_follow_data_end(data_world_size, loader_batch_size, resume_position)
        train_dataset = self.dataset.make(
            batch_size=loader_batch_size,
            shard_index=data_rank,
            num_shards=data_world_size,
            server_hosts=ctx.ip_addrs,
            run_server=ctx.rank
            % (
                self.parallel_config.num_devices_per_node
                // self.parallel_config.num_devices_per_process
            )
            == 0,
            skip_rows=skip_rows,
            prefetch_factor=self.data_prefetch_factor,
            resume_position=resume_position,
        )
        self._eval_dataset = train_dataset

        if self.stop_at_data_end and not self.follow_data_end:
            assert isinstance(self.state, RecsysTrainingState)
            step = self.state.step.item()
            data_end = self.dataset.compute_max_steps(
                num_shards=data_world_size,
                batch_size=loader_batch_size,
                current_step=step,
                resume_position=resume_position,
            )
            if data_end is not None:
                data_end = self._data_end_step(step, data_end)
                if self.max_steps is None or data_end < self.max_steps:
                    self.max_steps: int | None = data_end

        self._precompile_batch_transfer()
        self.train_dataset = self.dataset_thread_iterator(train_dataset)

    def create_embedding_init_data(self) -> RecsysEmbeddingsParameter:
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)
        use_ip = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_ip_address
        )

        if self.using_seqpack:
            batch = self.example_data(self.read_bsz_per_process // self.num_microbatch)

            assert self.batch_size % self.bs_per_device == 0, (self.batch_size, self.bs_per_device)
            assert batch["user_hashes"].shape[1] == self.bs_per_device // self.num_microbatch, (
                batch["user_hashes"].shape[1],
                self.bs_per_device,
                self.num_microbatch,
            )
            assert self.num_devices == self.batch_size // self.bs_per_device, (
                self.num_devices,
                self.batch_size,
                self.bs_per_device,
            )

            emb_size = self.model_config.emb_table_width

            return RecsysEmbeddingsParameter(
                user_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (self.num_devices, *batch["user_hashes"].shape[1:], emb_size), jnp.bfloat16
                    )
                ),
                history_post_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["history_seq"]["post_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                history_author_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["history_seq"]["auth_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                candidate_post_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["candidate_seq"]["post_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                candidate_author_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["candidate_seq"]["auth_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                user_ip_embeddings=(
                    EmbTable(
                        x=jax.ShapeDtypeStruct(
                            (self.num_devices, *batch["user_ip_hashes"].shape[1:], emb_size),
                            jnp.bfloat16,
                        )
                    )
                    if use_ip
                    else None
                ),
            )

        batch_size = self.batch_size // self.num_microbatch
        emb_size = self.model_config.emb_table_width
        user_emb_len = self.model_config.hash_table.num_user_hashes
        history_post_len = (
            self.model_config.hash_table.num_item_hashes * self.dataset.history_seq_len
        )
        history_author_len = (
            self.model_config.hash_table.num_author_hashes * self.dataset.history_seq_len
        )
        total_candidate_seq_len = self._total_candidate_seq_len()
        candidate_post_len = self.model_config.hash_table.num_item_hashes * total_candidate_seq_len
        candidate_author_len = (
            self.model_config.hash_table.num_author_hashes * total_candidate_seq_len
        )
        if use_ip:
            user_ip_len = self.model_config.hash_table.num_ip_hashes
        else:
            user_ip_len = 0

        return RecsysEmbeddingsParameter(
            user_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, user_emb_len, emb_size), jnp.bfloat16)
            ),
            history_post_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, history_post_len, emb_size), jnp.bfloat16)
            ),
            history_author_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, history_author_len, emb_size), jnp.bfloat16),
            ),
            candidate_post_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, candidate_post_len, emb_size), jnp.bfloat16),
            ),
            candidate_author_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, candidate_author_len, emb_size), jnp.bfloat16),
            ),
            user_ip_embeddings=(
                EmbTable(
                    x=jax.ShapeDtypeStruct((batch_size, user_ip_len, emb_size), jnp.bfloat16),
                )
                if use_ip
                else None
            ),
        )

    def _row_embedding_token_ids(self, ids: jax.Array) -> jax.Array:
        rows = self.dataset.input_vocab_size
        return jnp.clip(jnp.where(ids < 0, ids + rows, ids), 0, rows - 1)

    def get_flattened_token_ids(self, data: RecsysFeaturesBatch) -> jax.Array:
        ids = self._lookup_hash_ids(data)
        return self._row_embedding_token_ids(ids) if self.use_row_emb else ids

    def _segment_sum(
        self,
        emb_gradients_p: RecsysEmbeddingsParameter,
        inverse_indices: jax.Array,
        num_unique: int,
    ) -> jax.Array:
        if self.using_seqpack:
            segments: list[jax.Array] = []
            if emb_gradients_p.user_embeddings is not None:
                segments.append(emb_gradients_p.user_embeddings.x)
            if emb_gradients_p.history_post_embeddings is not None:
                segments.append(emb_gradients_p.history_post_embeddings.x)
            segments.append(emb_gradients_p.history_author_embeddings.x)
            if emb_gradients_p.candidate_post_embeddings is not None:
                segments.append(emb_gradients_p.candidate_post_embeddings.x)
            segments.append(emb_gradients_p.candidate_author_embeddings.x)
            if emb_gradients_p.user_ip_embeddings is not None:
                segments.append(emb_gradients_p.user_ip_embeddings.x)
            emb_width = segments[0].shape[-1]
            emb_gradients: jax.Array = jnp.concatenate(
                [x.reshape(self.batch_size, -1, emb_width) for x in segments], axis=1
            ).reshape(-1, emb_width)
        else:
            grad_segments: list[jax.Array] = []
            if emb_gradients_p.user_embeddings is not None:
                grad_segments.append(emb_gradients_p.user_embeddings.x)
            if emb_gradients_p.history_post_embeddings is not None:
                grad_segments.append(emb_gradients_p.history_post_embeddings.x)
            grad_segments.append(emb_gradients_p.history_author_embeddings.x)
            if emb_gradients_p.candidate_post_embeddings is not None:
                grad_segments.append(emb_gradients_p.candidate_post_embeddings.x)
            grad_segments.append(emb_gradients_p.candidate_author_embeddings.x)
            if emb_gradients_p.user_ip_embeddings is not None:
                grad_segments.append(emb_gradients_p.user_ip_embeddings.x)
            emb_width = grad_segments[0].shape[-1]
            emb_gradients = jnp.concatenate(
                grad_segments,
                axis=1,
            ).reshape((-1, emb_width))

        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        @shard_map(
            mesh=self.mesh,
            in_specs=(
                P("replica", "expert"),
                P("replica"),
            ),
            out_specs=P(None, "expert"),
            check_vma=False,
        )
        def _seg_sum(grads: jax.Array, tokens: jax.Array):
            acc = jax.ops.segment_sum(
                grads.astype(jnp.float32),
                tokens,
                num_segments=num_unique,
            ).astype(jnp.bfloat16)
            return jax.lax.psum(acc, axis_name="replica")

        return _seg_sum(emb_gradients, inverse_indices)

    def update(
        self,
        state: RecsysTrainingState,
        data: RecsysFeaturesBatch,
        lr: float,
    ):
        assert state.emb_table is not None
        assert state.emb_table_state is not None
        assert state.opt_state is not None
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        token_ids = self.get_flattened_token_ids(data)
        assert isinstance(self.dataset, PhoenixDataset)
        hash_vocab = getattr(self.dataset, "hash_vocab_size", None) or state.emb_table.x.shape[0]
        num_unique = min(token_ids.shape[0] * token_ids.shape[1], hash_vocab) + 1
        unique_tokens, inverse_indices = compress_token_ids(
            token_ids,
            fill_size=num_unique,
            fill_value=hash_vocab,
            output_sharding=P(("stage", *self.model_config.model_config.data_axis)),
        )

        rng, new_rng = jax.random.split(state.rng)

        fprop_params = state.params
        if self.precision_level < 3:
            fprop_params = jax.tree.map(cast_bfloat16, state.params)

        emb_table = state.emb_table
        recsys_embeddings = self.get_recsys_embeddings(data, emb_table)
        recsys_embeddings, emb_carry = self._emb_optim.transform_embeddings(
            recsys_embeddings, token_ids, state.emb_table_state
        )

        rce_ema = state.rce_ema
        calib_ema = state.calib_ema
        rce_alpha = None
        if rce_ema is not None and self.smoothing_windows:
            rce_alpha = jnp.array(
                [min(1.0, self.batch_size / w) for w in self.smoothing_windows],
                dtype=jnp.float32,
            )

        loss_and_grad_fn = jax.value_and_grad(self.loss_fn.apply, argnums=(0, 3), has_aux=True)
        (loss, stats), (gradients, emb_gradients) = loss_and_grad_fn(
            fprop_params, rng, data, recsys_embeddings, rce_ema, rce_alpha, calib_ema
        )

        if self.precision_level >= 2:
            gradients = jax.tree.map(lambda x: x.astype(jnp.float32), gradients)
        emb_gradients = jax.tree.map(lambda x: x.astype(jnp.bfloat16), emb_gradients)

        updates, new_opt_state = self.optim.update(gradients, state.opt_state, params=fprop_params)
        new_opt_state = typing.cast(InjectHyperparamsState, new_opt_state)
        new_params: Parameter = apply_updates(state.params, updates, lr)

        valid_step, grad_norm = self.is_valid_step(gradients)

        segment_sum_result = self._segment_sum(emb_gradients, inverse_indices, num_unique)

        emb_gradients = replace(
            emb_gradients.history_author_embeddings,
            x=segment_sum_result,
        )

        emb_valid_step, emb_grad_norm = self.is_valid_step(emb_gradients)
        keep_step = valid_step & emb_valid_step

        def _update(updated, original):
            return jnp.where(keep_step, updated, original)

        new_params = jax.tree.map(_update, new_params, state.params)
        new_opt_state = jax.tree.map(_update, new_opt_state, state.opt_state)

        examples = np.prod(data["user_hashes"].shape[:-1])

        metrics = {
            "step": state.step,
            "loss": loss,
            "examples_per_batch": examples,
            "valid_step": keep_step,
            "global_grad_norm": grad_norm,
            "learning_rate": lr * new_opt_state.hyperparams["learning_rate"],
            "weight_decay": new_opt_state.hyperparams["weight_decay"],
            "b1": new_opt_state.hyperparams["b1"],
            "b2": new_opt_state.hyperparams["b2"],
        }
        if self.optim_config.clip_partitions:
            metrics.update(partition_grad_norms(gradients, self.optim_config.clip_partitions))
        if self.track_norm_metrics:
            metrics.update(norm_metrics(new_params, new_opt_state, gradients, updates))

        new_rce_ema = stats.pop("_rce_ema", state.rce_ema)
        new_calib_ema = stats.pop("_calib_ema", state.calib_ema)
        metrics.update(**stats)

        new_emb_table, emb_new_opt_state, emb_optim_metrics = self._emb_optim.sparse_update(
            grads=emb_gradients,
            full_state=state.emb_table_state,
            full_emb_table=emb_table,
            unique_tokens=unique_tokens,
            num_unique=num_unique,
            lr=lr,
            valid_step=keep_step,
            carry=emb_carry,
        )

        metrics["emb_grad_norm"] = emb_grad_norm
        metrics["emb_valid_step"] = emb_valid_step
        metrics.update(emb_optim_metrics)

        new_state = RecsysTrainingState(
            params=new_params,
            opt_state=new_opt_state,
            rng=new_rng,
            step=state.step + 1,
            emb_table=new_emb_table,
            emb_table_state=emb_new_opt_state,
            offset_keys=state.offset_keys,
            offset_values=state.offset_values,
            post_embeddings=state.post_embeddings,
            rce_ema=new_rce_ema,
            calib_ema=new_calib_ema,
        )

        assert isinstance(self.dataset, PhoenixDataset)
        metrics["offset_zero_values"] = jnp.sum(state.offset_values == 0)
        metrics = jax.tree.map(lambda x: x.astype(jnp.float32), metrics)

        metric_keys, metric_values = zip(*metrics.items())
        metrics = {
            metric_keys: jnp.stack([jnp.float32(v) for v in metric_values], axis=0),
            ("valid_step",): keep_step,
        }

        return new_state, metrics, {}

    def forward_only_step(
        self,
        state: RecsysTrainingState,
        data: RecsysFeaturesBatch,
        lr: float,
    ):
        del lr
        assert state.emb_table is not None
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        token_ids = self.get_flattened_token_ids(data)
        rng, new_rng = jax.random.split(state.rng)

        fprop_params = state.params
        if self.precision_level < 3:
            fprop_params = jax.tree.map(cast_bfloat16, state.params)

        recsys_embeddings = self.get_recsys_embeddings(data, state.emb_table)
        recsys_embeddings, _ = self._emb_optim.transform_embeddings(
            recsys_embeddings, token_ids, state.emb_table_state
        )

        rce_alpha = None
        if state.rce_ema is not None and self.smoothing_windows:
            rce_alpha = jnp.array(
                [min(1.0, self.batch_size / w) for w in self.smoothing_windows],
                dtype=jnp.float32,
            )

        loss, stats = self.loss_fn.apply(
            fprop_params, rng, data, recsys_embeddings, state.rce_ema, rce_alpha, state.calib_ema
        )
        new_rce_ema = stats.pop("_rce_ema", state.rce_ema)
        new_calib_ema = stats.pop("_calib_ema", state.calib_ema)

        valid_step = jnp.array(True)
        examples = np.prod(data["user_hashes"].shape[:-1])
        metrics = {
            "step": state.step,
            "loss": loss,
            "examples_per_batch": examples,
            "valid_step": valid_step,
        }
        metrics.update(**stats)
        metrics["offset_zero_values"] = jnp.sum(state.offset_values == 0)
        metrics = jax.tree.map(lambda x: x.astype(jnp.float32), metrics)

        metric_keys, metric_values = zip(*metrics.items())
        packed_metrics = {
            metric_keys: jnp.stack([jnp.float32(v) for v in metric_values], axis=0),
            ("valid_step",): valid_step,
        }

        new_state = state._replace(
            rng=new_rng,
            step=state.step + 1,
            rce_ema=new_rce_ema,
            calib_ema=new_calib_ema,
        )
        return new_state, packed_metrics, {}

    def _run_microbatches(
        self,
        microbatch_loss_fn: typing.Callable[..., typing.Any],
        params: Parameter,
        batches: RecsysFeaturesBatch,
        embeddings: RecsysEmbeddingsParameter,
        rngs: jax.Array,
        candidate_inputs: typing.Any,
        loss_normalizers: dict[str, jax.Array],
        step_fn: typing.Callable[..., typing.Any] | None = None,
    ):
        ctx = self._async_emb_context
        assert ctx is not None
        rows = batches["user_hashes"].shape[1]
        leaves = jax.tree.leaves((batches, candidate_inputs))
        assert all(x.shape[1] == rows for x in leaves), "a leaf without rows"
        assert all(ctx.mesh.shape[a] == 1 for a in ctx.mesh.axis_names if a not in ctx.data_axis)
        rows_spec = P(None, ctx.data_axis)

        @shard_map(
            mesh=ctx.mesh,
            in_specs=(P(), rows_spec, rows_spec, P(), rows_spec, P()),
            out_specs=(P(), P(), P(), rows_spec, P(ctx.data_axis)),
            check_vma=False,
        )
        def run(params, batches, embeddings, rngs, candidate_inputs, loss_normalizers):
            step_kwargs: dict[str, typing.Any] = {}
            pool, pool_vjp = None, None
            if step_fn is not None:
                pool, pool_vjp = jax.vjp(
                    lambda params, embeddings: step_fn(params, None, batches, embeddings),
                    params,
                    embeddings,
                )
                step_kwargs["step_inputs"] = candidate_inputs

            def loss(params, rng, batch, embeddings, candidate_inputs, pool):
                return microbatch_loss_fn(
                    params,
                    rng,
                    batch,
                    embeddings,
                    candidate_inputs=candidate_inputs,
                    loss_normalizers=loss_normalizers,
                    **step_kwargs,
                    **({} if pool is None else {"candidate_pool": pool}),
                )

            def accumulate(sums, terms):
                return jax.tree.map(lambda a, g: a + g.astype(jnp.float32), sums, terms)

            def body(gradients, pool_gradients, rng, batch, embeddings, candidate_inputs):
                rng = jax.random.fold_in(rng, jax.lax.axis_index(ctx.data_axis))
                argnums = (0, 3) if pool is None else (0, 3, 5)
                (
                    (loss_k, (stats, metric_inputs)),
                    (gradients_k, emb_gradients, *pool_gradients_k),
                ) = jax.value_and_grad(loss, argnums=argnums, has_aux=True)(
                    params, rng, batch, embeddings, candidate_inputs, pool
                )
                gradients = accumulate(gradients, gradients_k)
                if pool is not None:
                    pool_gradients = accumulate(pool_gradients, pool_gradients_k[0])
                return gradients, pool_gradients, (loss_k, stats, emb_gradients, metric_inputs)

            inputs = (rngs, batches, embeddings, candidate_inputs)
            inputs, loss_normalizers = jax.lax.optimization_barrier((inputs, loss_normalizers))
            gradients = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
            pool_gradients = (
                None
                if pool is None
                else jax.tree.map(lambda x: jnp.zeros(x.shape, jnp.float32), pool)
            )
            outputs = []
            for k in range(rngs.shape[0]):
                if k > 0:
                    inputs, gradients, pool_gradients, outputs[-1] = jax.lax.optimization_barrier(
                        (inputs, gradients, pool_gradients, outputs[-1])
                    )
                microbatch = jax.tree.map(operator.itemgetter(k), inputs)
                gradients, pool_gradients, output = body(gradients, pool_gradients, *microbatch)
                outputs.append(output)
            losses, stats, emb_gradients, metric_inputs = jax.tree.map(
                lambda *xs: jnp.stack(xs), *outputs
            )
            if pool_vjp is not None:
                pool_param_gradients, pool_emb_gradients = pool_vjp(
                    jax.tree.map(lambda g, x: g.astype(x.dtype), pool_gradients, pool)
                )
                gradients = accumulate(gradients, pool_param_gradients)
                emb_gradients = jax.tree.map(jnp.add, emb_gradients, pool_emb_gradients)
            gradients = jax.tree.map(lambda g, p: g.astype(p.dtype), gradients, params)
            loss_sum, stats = jax.tree.map(lambda x: x.sum(0), (losses, stats))
            joined = jax.tree.map(
                lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]),
                (metric_inputs, batches),
            )
            return (
                *jax.lax.psum((gradients, loss_sum, stats), ctx.data_axis),
                emb_gradients,
                joined,
            )

        return run(params, batches, embeddings, rngs, candidate_inputs, loss_normalizers)

    def async_emb_step(
        self,
        state: RecsysTrainingState,
        data: tuple[RecsysFeaturesBatch, ...],
        lr: float,
        next_step_data: tuple[RecsysFeaturesBatch, ...],
        prev_step_lookup_pin: jax.Array,
    ):
        assert state.emb_table is not None
        assert state.emb_table_state is not None
        assert state.opt_state is not None
        ctx = self._async_emb_context
        assert ctx is not None

        rng, new_rng = jax.random.split(state.rng)
        fprop_params = state.params
        if self.precision_level < 3:
            fprop_params = jax.tree.map(cast_bfloat16, state.params)

        rce_alpha = None
        if state.rce_ema is not None and self.smoothing_windows:
            rce_alpha = jnp.array(
                [min(1.0, self.batch_size / w) for w in self.smoothing_windows],
                dtype=jnp.float32,
            )

        emb_table = state.emb_table
        token_ids = _step_token_ids([self.get_flattened_token_ids(b) for b in data], ctx.data_axis)
        prefetched_embeddings = recsys_async_emb.lookup_done(
            ctx, prev_step_lookup_pin, token_ids.shape
        )

        def dedup_tokens(token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
            return compress_token_ids_by_search(
                token_ids,
                fill_size=ctx.num_unique,
                mesh=ctx.mesh,
                token_ids_spec=P(None, ctx.data_axis, None),
            )

        unique_tokens = segment_ids = None
        if self.num_microbatch == 1 and not self.use_row_emb:
            fenced_token_ids, _ = jax.lax.optimization_barrier((token_ids, prefetched_embeddings))
            unique_tokens, segment_ids = dedup_tokens(fenced_token_ids)

        gate = prefetched_embeddings[0, :, 0, :1]

        if self.num_microbatch > 1:
            two_tower = isinstance(self.model_config, RecsysTwoTowerModelConfig)
            step_fn = None
            if two_tower:
                loss_inputs_fn, step_fn, microbatch_loss_fn, metrics_fn = (
                    self.microbatch_loss_fns.apply
                )
            else:
                loss_inputs_fn, microbatch_loss_fn, metrics_fn = self.microbatch_loss_fns.apply
            data_shards = math.prod(self.mesh.shape[a] for a in ctx.data_axis)
            candidate_inputs, loss_normalizers = loss_inputs_fn(
                {}, None, data, self.num_microbatch * data_shards
            )
            batches = jax.tree.map(lambda *x: jnp.stack(x), *data)
            gate = recsys_async_emb.depend(
                ctx, gate, jnp.stack(jax.tree.leaves(loss_normalizers)).sum(), on_sharded=False
            )
            if two_tower and candidate_inputs.pool.tweet_counts is not None:
                gate = recsys_async_emb.depend(ctx, gate, candidate_inputs.pool.tweet_counts[0])

        update_start_pin, updating_table, updating_emb_state, emb_optim_metrics = (
            self._emb_optim.gradient_update_start(
                ctx, emb_table.x, state.emb_table_state, gate=gate
            )
        )

        embeddings = self._unflatten_emb_lookup(
            data[0],
            replace(
                emb_table,
                x=jax.lax.with_sharding_constraint(
                    prefetched_embeddings, P(None, ctx.data_axis, None, None)
                ),
            ),
        )
        embeddings, _ = self._emb_optim.transform_embeddings(
            embeddings, token_ids, state.emb_table_state
        )
        next_step_token_ids = _step_token_ids(
            [self.get_flattened_token_ids(b) for b in next_step_data], ctx.data_axis
        ).astype(jnp.int32)

        if self.num_microbatch == 1:
            embeddings = jax.tree.map(lambda x: x[0], embeddings)
            candidate_authors = embeddings.candidate_author_embeddings
            embeddings = replace(
                embeddings,
                candidate_author_embeddings=replace(
                    candidate_authors,
                    x=recsys_async_emb.depend(
                        ctx, candidate_authors.x, update_start_pin, on_sharded=False
                    ),
                ),
            )

            def loss_fn(params, embeddings):
                return self.loss_fn.apply(
                    params, rng, data[0], embeddings, state.rce_ema, rce_alpha, state.calib_ema
                )

            loss, loss_vjp, stats = jax.vjp(loss_fn, fprop_params, embeddings, has_aux=True)

            (
                emb_grad_norm,
                emb_valid_step,
                emb_update_pending,
                updated_emb_state,
                grad_update_done_pin,
            ) = self._emb_optim.gradient_update_done(ctx, updating_emb_state, loss)
            updated_emb_state, grad_update_done_pin = jax.lax.optimization_barrier(
                (updated_emb_state, grad_update_done_pin)
            )

            next_step_token_ids, _ = jax.lax.optimization_barrier((next_step_token_ids, loss))
            new_emb_table, next_step_lookup_pin = recsys_async_emb.lookup_start(
                ctx, next_step_token_ids, updating_table, grad_update_done_pin
            )
            loss_cotangent, next_step_lookup_pin = jax.lax.optimization_barrier(
                (jnp.ones_like(loss), next_step_lookup_pin)
            )
            gradients, emb_gradients = loss_vjp(loss_cotangent)
            emb_gradients = jax.tree.map(lambda x: x[None], emb_gradients)
        else:
            new_emb_table, next_step_lookup_pin = recsys_async_emb.lookup_start(
                ctx, next_step_token_ids, updating_table, update_start_pin
            )
            loss_normalizers = recsys_async_emb.depend(
                ctx, loss_normalizers, next_step_lookup_pin, x_sharded=False
            )
            gradients, loss, stats, emb_gradients, (metric_inputs, batch) = self._run_microbatches(
                microbatch_loss_fn,
                fprop_params,
                batches,
                embeddings,
                jax.random.split(rng, self.num_microbatch),
                candidate_inputs,
                loss_normalizers,
                step_fn=step_fn,
            )
            if not self.use_row_emb:
                token_ids, gradients = jax.lax.optimization_barrier((token_ids, gradients))
                unique_tokens, segment_ids = dedup_tokens(token_ids)
            (
                emb_grad_norm,
                emb_valid_step,
                emb_update_pending,
                updated_emb_state,
                grad_update_done_pin,
            ) = self._emb_optim.gradient_update_done(ctx, updating_emb_state, loss)
            updated_emb_state, grad_update_done_pin = jax.lax.optimization_barrier(
                (updated_emb_state, grad_update_done_pin)
            )
            if two_tower:
                stats = metrics_fn({}, None, metric_inputs, batch, stats)
            else:
                stats = metrics_fn(
                    {},
                    None,
                    metric_inputs,
                    batch,
                    stats,
                    state.rce_ema,
                    rce_alpha,
                    tuple(self.smoothing_windows) if state.rce_ema is not None else None,
                    state.calib_ema,
                )
        emb_valid_step = emb_valid_step | ~emb_update_pending
        if self.precision_level >= 2:
            gradients = jax.tree.map(lambda x: x.astype(jnp.float32), gradients)

        updates, new_opt_state = self.optim.update(gradients, state.opt_state, params=fprop_params)
        new_opt_state = typing.cast(InjectHyperparamsState, new_opt_state)
        new_params: Parameter = apply_updates(state.params, updates, lr)

        valid_step, grad_norm = self.is_valid_step(gradients)

        emb_gradients = jax.tree.map(lambda x: x.astype(jnp.bfloat16), emb_gradients)
        deferred_emb_valid_step, deferred_emb_grad_norm = self.is_valid_step(emb_gradients)
        keep_step = valid_step & deferred_emb_valid_step

        def _update(updated, original):
            return jnp.where(keep_step, updated, original)

        new_params = jax.tree.map(_update, new_params, state.params)
        new_opt_state = jax.tree.map(_update, new_opt_state, state.opt_state)

        metrics = {
            "step": state.step,
            "loss": loss,
            "examples_per_batch": sum(np.prod(batch["user_hashes"].shape[:-1]) for batch in data),
            "valid_step": keep_step,
            "global_grad_norm": grad_norm,
            "learning_rate": lr * new_opt_state.hyperparams["learning_rate"],
            "weight_decay": new_opt_state.hyperparams["weight_decay"],
            "b1": new_opt_state.hyperparams["b1"],
            "b2": new_opt_state.hyperparams["b2"],
            "emb_grad_norm": emb_grad_norm,
            "emb_valid_step": emb_valid_step,
            "deferred_emb_grad_norm": deferred_emb_grad_norm,
            "deferred_emb_valid_step": deferred_emb_valid_step,
        }
        if self.optim_config.clip_partitions:
            metrics.update(partition_grad_norms(gradients, self.optim_config.clip_partitions))
        metrics.update(emb_optim_metrics)
        if self.track_norm_metrics:
            metrics.update(norm_metrics(new_params, new_opt_state, gradients, updates))

        new_rce_ema = stats.pop("_rce_ema", state.rce_ema)
        new_calib_ema = stats.pop("_calib_ema", state.calib_ema)
        metrics.update(**stats)
        if self.seqpack_candidate_packing:
            metrics.update(self._candidate_region_metrics(data, grad_update_done_pin))
        metrics["offset_zero_values"] = jnp.sum(state.offset_values == 0)
        metrics = jax.tree.map(lambda x: x.astype(jnp.float32), metrics)

        metric_keys, metric_values = zip(*metrics.items())
        metrics = {
            metric_keys: jnp.stack([jnp.float32(v) for v in metric_values], axis=0),
            ("valid_step",): keep_step,
        }

        stage_pin = recsys_async_emb.stage_update(
            self._async_emb_context,
            self._flatten_emb_grads(emb_gradients, _num_users(data[0])),
            segment_ids,
            unique_tokens,
            keep_step,
            grad_update_done_pin,
        )
        next_step_lookup_pin = jnp.minimum(next_step_lookup_pin, stage_pin)

        new_state = RecsysTrainingState(
            params=new_params,
            opt_state=new_opt_state,
            rng=new_rng,
            step=state.step + 1,
            emb_table=replace(emb_table, x=new_emb_table),
            emb_table_state=updated_emb_state,
            offset_keys=state.offset_keys,
            offset_values=state.offset_values,
            post_embeddings=state.post_embeddings,
            rce_ema=new_rce_ema,
            calib_ema=new_calib_ema,
        )
        return new_state, metrics, {}, next_step_lookup_pin

    def async_emb_update(
        self,
        state: RecsysTrainingState,
        data: tuple[RecsysFeaturesBatch, ...],
        lr: float,
    ):
        assert self._batch_pipeline.reserve is not None
        try:
            if self._async_emb_lookup_pin is None:
                state, self._async_emb_lookup_pin = self._first_step_embedding_lookup_start_jit(
                    state, self._lookup_hash_fields(data)
                )

            step_jit = self._async_emb_step_jits[self._candidate_region_len(data)]
            state, metrics, extras, self._async_emb_lookup_pin = step_jit(
                state,
                data,
                lr,
                self._lookup_hash_fields(self._batch_pipeline.reserve.batch),
                self._async_emb_lookup_pin,
            )
        except Exception:
            assert self._async_emb_context is not None
            recsys_async_emb.kernel_api(self._async_emb_context).abort(
                self._async_emb_context.context_id
            )
            raise

        return state, metrics, extras

    def _pin_step_scratch(self) -> bool:
        if jax.default_backend() != "gpu" or not self.enable_aot:
            return False
        scratch_sizes = [
            step_jit.jitted.memory_analysis().temp_size_in_bytes
            for step_jit in self._async_emb_step_jits.values()
        ]
        block_bytes, smallest_scratch_bytes = max(scratch_sizes), min(scratch_sizes)
        if block_bytes - smallest_scratch_bytes >= min(
            smallest_scratch_bytes, _ALLOCATOR_SPLIT_BYTES
        ):
            rank_logger.warning(
                "Step scratch not pinned: its sizes %s differ enough for a smaller scratch to "
                "split the block",
                scratch_sizes,
            )
            return False
        anchors = []
        for device in self.mesh.local_devices:
            low, block, high = (
                _device_zeros(num_bytes, device)
                for num_bytes in (_ANCHOR_BYTES, block_bytes, _ANCHOR_BYTES)
            )
            low_ptr, block_ptr, high_ptr = (
                buffer.unsafe_buffer_pointer() for buffer in (low, block, high)
            )
            rounded_block_bytes = block_bytes + -block_bytes % _ALLOCATOR_ROUNDING_BYTES
            if (block_ptr, block_ptr + rounded_block_bytes) != (low_ptr + _ANCHOR_BYTES, high_ptr):
                rank_logger.warning(
                    "Step scratch not pinned: its block is not between its anchors on %s "
                    "(low %#x, block %#x, high %#x)",
                    device,
                    low_ptr,
                    block_ptr,
                    high_ptr,
                )
                return False
            anchors.extend((low, high))
            del block
        self._allocator_anchors = anchors
        rank_logger.info("Pinned a free %d-byte block for the step scratch", block_bytes)
        return True

    def _reserve_small_buffer_region(self) -> None:
        for device in self.mesh.local_devices:
            region = _device_zeros(_SMALL_BUFFER_REGION_BYTES, device)
            self._allocator_anchors.append(_device_zeros(_ANCHOR_BYTES, device))
            del region
        rank_logger.info(
            "Reserved a free %d-byte region for small buffers", _SMALL_BUFFER_REGION_BYTES
        )

    def _lookup_hash_fields(
        self, data: tuple[RecsysFeaturesBatch, ...]
    ) -> tuple[RecsysFeaturesBatch, ...]:
        def pad_history(x: jax.Array, users_per_device: int) -> jax.Array:
            length = self._history_lookup_len(x.shape[-1])
            if length is None or x.shape[1] * x.shape[-1] == users_per_device * length:
                return x
            per_user = _pad_axis(x.reshape(x.shape[0], users_per_device, -1), 2, length)
            return per_user.reshape(x.shape[0], -1, x.shape[-1])

        return tuple(
            typing.cast(
                RecsysFeaturesBatch,
                {
                    "user_hashes": batch["user_hashes"],
                    "user_ip_hashes": batch.get("user_ip_hashes"),
                    "history_seq": {
                        k: pad_history(batch["history_seq"][k], batch["user_hashes"].shape[1])
                        for k in ("post_hashes", "auth_hashes")
                    },
                    "candidate_seq": {
                        k: batch["candidate_seq"][k] for k in ("post_hashes", "auth_hashes")
                    },
                },
            )
            for batch in data
        )

    def _candidate_region_metrics(
        self, data: tuple[RecsysFeaturesBatch, ...], update_done_pin: jax.Array
    ) -> dict[str, jax.Array]:
        used_tokens, dropped = [], []
        for batch in data:
            layout = batch["packing_layout"]
            assert layout is not None and layout.candidate_cu_seqlens is not None
            assert layout.dropped_candidate_slots is not None
            used_tokens.append(layout.candidate_cu_seqlens[:, -1])
            dropped.append(layout.dropped_candidate_slots)
        assert self._async_emb_context is not None
        used_tokens, dropped = recsys_async_emb.depend(
            self._async_emb_context,
            (jnp.stack(used_tokens, axis=1), jnp.concatenate(dropped, axis=1)),
            update_done_pin,
            on_sharded=False,
        )
        return {
            "train/seqpack_candidate_tokens_used_max": jnp.max(used_tokens),
            "train/seqpack_candidate_token_budget": jnp.int32(self._candidate_region_len(data) - 1),
            "train/seqpack_dropped_candidate_slots": jnp.sum(dropped),
        }

    def _candidate_region_len(self, data: tuple[RecsysFeaturesBatch, ...]) -> int:
        if not self.seqpack_candidate_packing:
            return 0
        layout = data[0]["packing_layout"]
        assert layout is not None and layout.candidate_token_to_slot is not None
        return layout.candidate_token_to_slot.shape[-1]

    @property
    def using_seqpack(self) -> bool:
        return (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_seqpack
        )

    @property
    def using_fa4(self) -> bool:
        if not self.using_seqpack:
            return False
        if isinstance(self.model_config, RecsysAggregatedModelConfig):
            attn_config = self.model_config.model_config.attn_config
        elif isinstance(self.model_config, RecsysTwoTowerModelConfig):
            attn_config = self.model_config.user_tower_config.model_config.attn_config
        else:
            return False
        return getattr(attn_config, "attn_impl", None) == "cutedsl_ranker_varlen_attn"

    def add_block_sparse_layout(self, batch: RecsysFeaturesBatch) -> RecsysFeaturesBatch:
        from dataclasses import replace

        from xrex.cutedsl.ranker_attention_varlen_fa4 import build_block_sparse_layout

        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        layout = batch["packing_layout"]
        assert layout is not None, "FA4 path requires a packed batch"
        bs_per_device = layout.cu_seqlens.shape[1] - 1
        transformer_candidate_seq_len = layout.candidate_positions.shape[1] // bs_per_device
        _mc = (
            self.model_config.user_tower_config
            if isinstance(self.model_config, RecsysTwoTowerModelConfig)
            else self.model_config
        )
        cand_slot_lens = getattr(layout, "cand_slot_lens", None)
        cu_seqlens, padding_mask = layout.cu_seqlens, layout.padding_mask
        num_user_prefix_tokens = _mc.num_user_prefix_tokens
        if layout.candidate_cu_seqlens is not None:
            cu_seqlens, padding_mask = candidate_region_history_schedule(
                layout, transformer_candidate_seq_len, num_user_prefix_tokens
            )
            transformer_candidate_seq_len = 0
            num_user_prefix_tokens = 0
        block_sparse = build_block_sparse_layout(
            cu_seqlens=cu_seqlens,
            transformer_candidate_seq_len=transformer_candidate_seq_len,
            max_history_seq_len=(_mc.num_user_prefix_tokens + _mc.history_seq_len),
            packed_seq_len=int(layout.segment_ids.shape[1]),
            padding_mask=padding_mask,
            num_user_prefix_tokens=num_user_prefix_tokens,
            candidate_slot_lens=(
                np.asarray(cand_slot_lens) if cand_slot_lens is not None else None
            ),
        )
        return {**batch, "packing_layout": replace(layout, block_sparse=block_sparse)}

    @property
    def _seqpack_block_size(self) -> int:
        if isinstance(self.model_config, RecsysAggregatedModelConfig):
            attn_config = self.model_config.model_config.attn_config
        elif isinstance(self.model_config, RecsysTwoTowerModelConfig):
            attn_config = self.model_config.user_tower_config.model_config.attn_config
        else:
            raise ValueError(f"Unexpected model_config type {type(self.model_config)}")
        attn_impl = getattr(attn_config, "attn_impl", None)
        if attn_impl == "cutedsl_ranker_varlen_attn":
            from xrex.cutedsl.ranker_attention_varlen_fa4 import get_device_tuning_config

            return get_device_tuning_config().block_q
        if attn_impl == "pallas_ranker_varlen_attn":
            from xrex.pallas.ranker_attention_varlen import get_device_tuning_config

            return get_device_tuning_config().block_q
        raise ValueError(f"Unexpected attn_impl {attn_impl!r} for seqpack block size")

    def _row_emb_state_sharding(self, state_sharding):
        emb_state = state_sharding.emb_table_state
        if emb_state is None:
            return state_sharding
        row_sharding = NamedSharding(self.mesh, P("expert"))
        emb_state = emb_state._replace(row_sum_sq={**emb_state.row_sum_sq, "table": row_sharding})
        if emb_state.last_step is not None:
            emb_state = emb_state._replace(last_step={**emb_state.last_step, "table": row_sharding})
        return state_sharding._replace(emb_table_state=emb_state)

    def _create_async_emb_executables(self, init_data, lr_shape, compiler_options) -> None:
        if async_emb is None and not self.use_row_emb:
            raise RuntimeError(
                "use_async_emb requires the async_emb kernels (xrex.cuda.async_emb), "
                "which failed to import: no compiled binding was found, or the "
                "installed extension does not match this environment's NCCL."
            )

        data_axis = ("stage", *self.model_config.model_config.data_axis)
        assert int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8")) > 1
        assert os.environ.get("NCCL_RUNTIME_CONNECT") == "0"
        assert os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") == "1"
        assert self.parallel_config.num_devices_per_process == 1
        assert "expert" in data_axis
        assert all(self.mesh.shape[a] == 1 for a in data_axis if a != "expert")
        assert isinstance(self._emb_optim, AsyncEmbOptimizer)

        tokens_per_example = jax.eval_shape(self.get_flattened_token_ids, init_data).shape[1]
        tokens_per_batch = self.batch_size * tokens_per_example
        data_shards = math.prod(self.mesh.shape[a] for a in data_axis)
        emb_width = self.state_shape.emb_table.x.shape[1]

        self._emb_hash_vocab = (
            getattr(self.dataset, "hash_vocab_size", None) or self.dataset.input_vocab_size
        )
        assert self.use_row_emb or self._emb_hash_vocab >= self.state_shape.emb_table.x.shape[0]
        assert self._emb_hash_vocab < 2**31
        num_unique_tokens = min(tokens_per_batch, self._emb_hash_vocab) + 1

        if self.use_row_emb:
            try:
                from xrex.cuda.row_emb import row_emb
            except ImportError as e:
                raise RuntimeError(f"use_row_emb requires the row_emb kernels: {e}") from e

            self._async_emb_context = row_emb.make_context_handle(
                self.mesh,
                ("expert",),
                data_axis=data_axis,
                tokens_per_batch=tokens_per_batch,
                emb_width=emb_width,
                vocab_rows=self.state_shape.emb_table.x.shape[0],
                recv_factor=self.row_emb_recv_factor,
                num_devices_per_node=self.parallel_config.num_devices_per_node,
            )
        else:
            self._async_emb_context = async_emb.make_context_handle(
                self.mesh,
                ("expert",),
                data_axis=data_axis,
                tokens_per_batch=tokens_per_batch,
                emb_width=emb_width,
                num_unique=num_unique_tokens,
                num_devices_per_node=self.parallel_config.num_devices_per_node,
            )

        row_sharding = NamedSharding(self.mesh, P(data_axis, None))

        def first_step_embedding_lookup_start(state, data):
            token_ids = _step_token_ids(
                [self.get_flattened_token_ids(b) for b in data], data_axis
            ).astype(jnp.int32)
            emb_table, lookup_pin = recsys_async_emb.lookup_start(
                self._async_emb_context,
                token_ids,
                state.emb_table.x,
                gate=token_ids[0, :1, :1].astype(jnp.float32),
            )
            return (state._replace(emb_table=replace(state.emb_table, x=emb_table)), lookup_pin)

        lookup_pin_shape = jax.ShapeDtypeStruct((data_shards, 1), jnp.float32)
        lookup_data = self._lookup_hash_fields((init_data,) * self.num_microbatch)

        self._first_step_embedding_lookup_start_jit = JittedOrCompiled(
            jax.jit(
                first_step_embedding_lookup_start,
                in_shardings=(self.state_sharding, self.data_sharding),
                out_shardings=(self.state_sharding, row_sharding),
                donate_argnums=(0,),
            )
        )
        self.register_jit_function(
            self._first_step_embedding_lookup_start_jit,
            self.state_shape,
            lookup_data,
            compiler_options=compiler_options,
        )

        sizes = self.seqpack_candidate_count if self.seqpack_candidate_packing else []
        if sizes and sizes[-1] <= self._total_candidate_seq_len():
            rank_logger.warning(
                "seqpack_candidate_count ends at %d, not above the %d candidate slots, so "
                "batches whose users fill nearly every slot drop candidates "
                "(train/seqpack_dropped_candidate_slots)",
                sizes[-1],
                self._total_candidate_seq_len(),
            )
        metrics_sharding = None
        if len(sizes) > 1:
            memory_kind = "pinned_host" if jax.default_backend() == "gpu" else "unpinned_host"
            metrics_sharding = NamedSharding(self.mesh, P(), memory_kind=memory_kind)
        for candidate_tokens_per_user in sizes or [None]:
            data = init_data
            if candidate_tokens_per_user is not None and candidate_tokens_per_user != sizes[0]:
                data = jax.tree.map(
                    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
                    super().prepare_data(
                        self.example_data(
                            self.read_bsz_per_process // self.num_microbatch,
                            candidate_tokens_per_user,
                        )
                    ),
                )
            step_data = (data,) * self.num_microbatch
            step_jit = JittedOrCompiled(
                jax.jit(
                    self.async_emb_step,
                    in_shardings=(
                        self.state_sharding,
                        self.data_sharding,
                        None,
                        self.data_sharding,
                        row_sharding,
                    ),
                    out_shardings=(self.state_sharding, metrics_sharding, None, row_sharding),
                    donate_argnums=(0, 4),
                ),
                name=(
                    None
                    if candidate_tokens_per_user is None
                    else f"async_emb_step_{candidate_tokens_per_user}"
                ),
            )
            self.register_jit_function(
                step_jit,
                self.state_shape,
                step_data,
                lr_shape,
                lookup_data,
                lookup_pin_shape,
                compiler_options=compiler_options,
            )
            self._async_emb_step_jits[self._candidate_region_len(step_data)] = step_jit

        self.update_jit = self.async_emb_update

    def compile_or_load(self, phases: set[str] | None = None) -> None:
        super().compile_or_load(phases)
        if phases is not None or not self._async_emb_step_jits or jax.default_backend() != "gpu":
            return
        gemms = self.model_config.model_config.fp8_gemms
        if gemms != Fp8Gemms.OFF:
            for step in self._async_emb_step_jits.values():
                check_compiled_fp8_gemms(step.name(), step.jitted.as_text(), gemms)

    def create_executable(self) -> None:
        if self.num_microbatch < 1 or self.bs_per_device % self.num_microbatch:
            raise ValueError(
                f"num_microbatch={self.num_microbatch} must divide "
                f"bs_per_device={self.bs_per_device}"
            )
        if self.num_microbatch > 1:
            if not self.use_async_emb:
                raise ValueError("num_microbatch > 1 requires use_async_emb=True")
            if isinstance(self.model_config, RecsysTwoTowerModelConfig):
                if self.model_config.apply_u2u_and_i2i_loss:
                    raise ValueError("num_microbatch > 1 does not support apply_u2u_and_i2i_loss")
            elif type(self.model_config) is not RecsysAggregatedModelConfig:
                raise ValueError("num_microbatch > 1 supports only the ranker and the two-tower")
            if self.empty_history_augmentation_rate > 0:
                raise ValueError(
                    "num_microbatch > 1 does not support empty_history_augmentation_rate"
                )
        if self.seqpack_candidate_packing:
            if not isinstance(self.model_config, RecsysAggregatedModelConfig):
                raise ValueError("seqpack_candidate_packing supports only the ranker")
            if not self.using_seqpack:
                raise ValueError("seqpack_candidate_packing requires use_seqpack=True")
            if not self.using_fa4:
                raise ValueError(
                    "seqpack_candidate_packing requires attn_impl=cutedsl_ranker_varlen_attn"
                )
            attn_config = self.model_config.model_config.attn_config
            assert attn_config is not None
            if attn_config.attn_logit_cap > 0 and attn_config.attn_logit_cap_method != "none":
                raise ValueError("seqpack_candidate_packing does not support attn_logit_cap")
            if self.seqpack_distribution is None:
                raise ValueError("seqpack_candidate_packing requires seqpack_distribution")
            if not self.seqpack_candidate_count:
                raise ValueError("seqpack_candidate_packing requires seqpack_candidate_count")
            if list(self.seqpack_candidate_count) != sorted(set(self.seqpack_candidate_count)):
                raise ValueError(
                    "seqpack_candidate_count must be ascending and unique: "
                    f"{self.seqpack_candidate_count}"
                )
            if not self.use_async_emb:
                raise ValueError("seqpack_candidate_packing requires use_async_emb=True")
            users_per_microbatch = int(self.bs_per_device) // self.num_microbatch
            if any(
                users_per_microbatch * size % PACK_BLOCK_SIZE
                for size in self.seqpack_candidate_count
            ):
                raise ValueError(
                    f"every seqpack_candidate_count times the {users_per_microbatch} "
                    f"users per device and microbatch must be a multiple of {PACK_BLOCK_SIZE}: "
                    f"{self.seqpack_candidate_count}"
                )
            shortest_mean_history = self._history_mean_len(self.seqpack_candidate_count[-1])
            if shortest_mean_history < self.seqpack_distribution.min_len:
                raise ValueError(
                    f"the largest candidate size leaves a mean history of {shortest_mean_history} "
                    f"tokens, below seqpack_distribution.min_len="
                    f"{self.seqpack_distribution.min_len}"
                )
        if (
            isinstance(self.model_config, RecsysAggregatedModelConfig)
            and self.model_config.page_rl.enabled
        ):
            if not (isinstance(self.dataset, PhoenixDataset) and self.dataset.page_records):
                raise ValueError("page_rl needs a dataset with page_records=True")
        if self.model_config.model_config.fp8_gemms != Fp8Gemms.OFF and not self.forward_only:
            if not self.use_async_emb:
                raise ValueError("fp8_gemms requires use_async_emb=True")
            if not self.enable_aot:
                raise ValueError("fp8_gemms requires enable_aot=True")

        self._init_shmem_write_pool()
        self._free_ports()
        self._init_incremental_state()

        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)

        if self.use_row_emb:
            if not self.use_async_emb:
                raise ValueError("use_row_emb requires use_async_emb=True")
            if self.emb_optim_config._active() != "rowwise_adagrad":
                raise ValueError("use_row_emb requires the rowwise_adagrad embedding optimizer")

        if isinstance(self.model_config, RecsysAggregatedModelConfig):
            ctx = self.model_config.context_features
        elif isinstance(self.model_config, RecsysTwoTowerModelConfig):
            ctx = self.model_config.user_tower_config.context_features
        else:
            ctx = None
        if ctx is not None and ctx.enabled:
            num_cat = max(
                (f.index + 1 for f in ctx.categorical_features if f.embedding_dim > 0),
                default=0,
            )
            num_cat = max(num_cat, 16) if num_cat > 0 else 0
            object.__setattr__(self.dataset, "num_context_categorical_features", num_cat)

        self.create_optim()
        self._emb_optim = self.emb_optim_config.make_optimizer(self.optim)

        @hk.transform
        def loss_fn(
            batch: RecsysFeaturesBatch,
            recsys_embeddings: RecsysEmbeddingsParameter,
            rce_ema: dict[str, jax.Array] | None = None,
            rce_alpha: jax.Array | None = None,
            calib_ema: dict[str, jax.Array] | None = None,
        ):
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            smoothing_windows = tuple(self.smoothing_windows) if rce_ema is not None else None
            if type(model) is RecsysAggregatedModel:
                loss, (stats, metric_inputs) = model.loss(batch, recsys_embeddings, True)
                return loss, model.metrics(
                    metric_inputs, batch, stats, rce_ema, rce_alpha, smoothing_windows, calib_ema
                )
            metrics_state_kwargs = {}
            if rce_ema is not None:
                metrics_state_kwargs = {
                    "rce_ema": rce_ema,
                    "rce_alpha": rce_alpha,
                    "smoothing_windows": smoothing_windows,
                    "calib_ema": calib_ema,
                }
            return model.loss(None, batch, recsys_embeddings, True, **metrics_state_kwargs)

        @hk.transform
        def loss_fn_eval(batch: RecsysFeaturesBatch, recsys_embeddings: RecsysEmbeddingsParameter):
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            if type(model) is RecsysAggregatedModel:
                loss, (stats, metric_inputs) = model.loss(batch, recsys_embeddings, False)
                return loss, model.metrics(metric_inputs, batch, stats)
            return model.loss(None, batch, recsys_embeddings, is_training=False)

        @hk.transform
        def user_forward_fn(
            batch: RecsysFeaturesBatch,
            recsys_embeddings: RecsysEmbeddingsParameter,
        ):
            assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
            return self.model_config.make(
                sharding_context=make_legacy_sharding_context(self.mesh),
            ).user_embeddings_only(batch, recsys_embeddings, is_training=False)

        self.loss_fn = loss_fn
        self.loss_fn_eval = loss_fn_eval
        self.user_forward_fn: typing.Any = user_forward_fn

        if self.num_microbatch > 1:

            def microbatch_loss_fns():
                assert isinstance(
                    self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
                )
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                if isinstance(model, RecsysTwoTowerModel):

                    def loss(batch, recsys_embeddings, **kwargs):
                        return model.loss(None, batch, recsys_embeddings, True, **kwargs)

                    return model.loss, (
                        model.loss_inputs_and_normalizers,
                        model.candidate_pool,
                        loss,
                        model.metrics,
                    )
                return model.loss, (model.loss_inputs_and_normalizers, model.loss, model.metrics)

            self.microbatch_loss_fns = hk.multi_transform(microbatch_loss_fns)

        if isinstance(self.model_config, RecsysTwoTowerModelConfig):

            @hk.transform
            def two_tower_forward_fn(
                batch: RecsysFeaturesBatch,
                recsys_embeddings: RecsysEmbeddings,
                post_embeddings: jax.Array,
                dataset_types: jax.Array,
                top_k: int,
                target_dataset_types,
                eval_bs_per_device: int = 0,
                query_head_weights: tuple[float, ...] | None = None,
            ):
                assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                results = model.forward(
                    batch,
                    recsys_embeddings,
                    post_embeddings,
                    dataset_types,
                    top_k,
                    target_dataset_types,
                    eval_bs_per_device=eval_bs_per_device,
                    query_head_weights=query_head_weights,
                )
                top_k_indices, top_k_scores = results[0]
                return top_k_indices, top_k_scores

            @hk.transform
            def candidate_tower_forward_fn(
                post_author_embedding: jax.Array,
                post_sids: jax.Array,
                post_hashes: jax.Array,
                head_index: int = 0,
            ):
                assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                assert model.candidate_tower is not None
                return model.build_candidate_embeddings_from_lookups(
                    post_author_embedding,
                    post_sids,
                    post_hashes,
                    head_index=head_index,
                )

            @hk.transform
            def mol_side_table_fn(post_table: jax.Array):
                assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                return model.mol_side_table_rows(post_table)

            self.two_tower_forward_fn = two_tower_forward_fn
            self.candidate_tower_forward_fn = candidate_tower_forward_fn
            self.mol_side_table_fn = mol_side_table_fn

        @hk.transform
        def forward_fn(
            batch: RecsysFeaturesBatch,
            recsys_embeddings: RecsysEmbeddings,
        ):
            assert isinstance(self.model_config, (RecsysAggregatedModelConfig))
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            logits = model.forward(batch, recsys_embeddings)
            return logits

        self.forward_fn = forward_fn

        if self.using_seqpack:
            rng = jax.ShapeDtypeStruct((2,), jnp.uint32)
            init_data = typing.cast(
                RecsysFeaturesBatch,
                super().prepare_data(
                    self.example_data(self.read_bsz_per_process // self.num_microbatch)
                ),
            )
        else:
            rng, init_data = self._rng_and_init_data()
            if self.num_microbatch > 1:
                assert isinstance(self.dataset, PhoenixDataset)
                init_data = self.dataset.example_data_shape(self.batch_size // self.num_microbatch)
            init_data = typing.cast(RecsysFeaturesBatch, init_data)
        emb_table_init_data = self.create_embedding_init_data()

        lr_shape = jax.ShapeDtypeStruct((), dtype=jnp.float32)
        with self.mesh:
            self.state_shape = jax.eval_shape(self.init, init_data, rng)
            _, self.stats_shape = jax.eval_shape(
                self.loss_fn.apply,
                self.state_shape.params,
                rng,
                init_data,
                emb_table_init_data,
            )

        self.state_sharding = self.get_sharding(self.state_shape)
        if self.use_row_emb:
            self.state_sharding = self._row_emb_state_sharding(self.state_sharding)
        self.data_sharding = NamedSharding(
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )

        compiler_options = None if not self.xla_flag_overrides else self.xla_flag_overrides

        def init(r):
            return self.init(init_data, r)

        self.init_jit = JittedOrCompiled(
            jax.jit(init, in_shardings=None, out_shardings=self.state_sharding)
        )
        self.register_jit_function(self.init_jit, rng)

        if self.use_async_emb:
            self._create_async_emb_executables(init_data, lr_shape, compiler_options)
        else:
            step_fn = self.forward_only_step if self.forward_only else self.update
            self.update_jit = JittedOrCompiled(
                jax.jit(
                    step_fn,
                    in_shardings=(
                        self.state_sharding,
                        self.data_sharding,
                        None,
                    ),
                    out_shardings=(self.state_sharding, None, None),
                    donate_argnums=(0,),
                )
            )
            self.register_jit_function(
                self.update_jit,
                self.state_shape,
                init_data,
                lr_shape,
                compiler_options=compiler_options,
            )

        if self.dense_params_ema_decay > 0.0:
            self.update_jit = self._with_dense_params_ema(self.update_jit)

        self.forward_jit = JittedOrCompiled(
            jax.jit(
                self.forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=self.data_sharding,
            )
        )
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            self.candidate_tower_forward_jit = JittedOrCompiled(
                jax.jit(
                    self.candidate_tower_forward_fn.apply,
                    in_shardings=(
                        self.state_sharding.params,
                        None,
                        self.data_sharding,
                        self.data_sharding,
                        self.data_sharding,
                    ),
                    out_shardings=self.data_sharding,
                    static_argnums=(5,),
                )
            )

        self.create_extra_executables()
        self.maybe_create_eval_executables()

    def maybe_create_eval_executables(self):
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            self.two_tower_forward_jit = jax.jit(
                self.two_tower_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                    self.data_sharding,
                    None,
                ),
                out_shardings=(
                    None,
                    None,
                ),
                static_argnums=[6, 7, 8, 9],
            )

            self.loss_fn_eval_jit = jax.jit(
                self.loss_fn_eval.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=(None, None),
            )

            self.user_forward_jit = jax.jit(
                self.user_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=self.data_sharding,
            )

    def _is_copy_port_binder(self) -> bool:
        hostnames = self.ctx.hostnames
        if not hostnames:
            return self.ctx.rank == 0
        return hostnames.index(hostnames[self.ctx.rank]) == self.ctx.rank

    def _copy_port_channel(self, port: int) -> grpc.Channel:
        cc = self.checkpoint_config
        if not cc.copy_port_tls_cert:
            return grpc.insecure_channel(f"127.0.0.1:{port}")
        if not cc.copy_port_tls_ca or not cc.copy_port_tls_server_name:
            raise ValueError(
                "copy_port_tls_ca and copy_port_tls_server_name are required for the "
                "loopback client when copy_port TLS is enabled"
            )
        if cc.copy_port_tls_client_ca and not (
            cc.copy_port_tls_client_cert and cc.copy_port_tls_client_key
        ):
            raise ValueError(
                "copy_port_tls_client_ca (mTLS) requires copy_port_tls_client_cert/key "
                "for the loopback client"
            )

        def read(p: str) -> bytes | None:
            return pathlib.Path(p).read_bytes() if p else None

        creds = grpc.ssl_channel_credentials(
            root_certificates=read(cc.copy_port_tls_ca),
            private_key=read(cc.copy_port_tls_client_key),
            certificate_chain=read(cc.copy_port_tls_client_cert),
        )
        options = (("grpc.ssl_target_name_override", cc.copy_port_tls_server_name),)
        return grpc.secure_channel(f"127.0.0.1:{port}", creds, options=options)

    def _free_ports(self):
        if port := self.checkpoint_config.copy_port:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr and (conn.laddr.port == port or conn.laddr.port == port + 1):
                    try:
                        process = psutil.Process(conn.pid)
                        os.kill(process.pid, signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass

    def _init_shmem_write_pool(self) -> None:
        self._shmem_write_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-shmem-write"
        )

    def _init_incremental_state(self) -> None:
        self._incremental_state = IncrementalState(
            unique_tokens=np.empty(0, dtype=np.int32),
            token_count=0,
            soft_step=0,
            checkpoint_path="",
        )

    def _copy_port_emb_host_spec(self) -> P | None:
        if not self.checkpoint_config.copy_port:
            return None
        if getattr(self.state_shape, "emb_table", None) is None:
            return None
        ep = self.mesh.shape["expert"]
        vocab = self.state_shape.emb_table.x.shape[0]
        if vocab % ep != 0:
            rank_logger.warning(
                "copy_port emb host row-shard disabled: vocab %d %% ep %d != 0 "
                "(falling back to the save-time device reshard + second host copy)",
                vocab,
                ep,
            )
            return None
        return P("expert", None)

    def adjust_host_sharding(self, host_sharding):
        spec = self._copy_port_emb_host_spec()
        if spec is None:
            return host_sharding
        emb_host = host_sharding.emb_table
        rank_logger.info(
            "copy_port: host emb_table layout %s -> %s (offload writes the flat-file "
            "row shards directly; no save-time duplicate host copy)",
            emb_host.spec,
            spec,
        )
        return host_sharding._replace(
            emb_table=NamedSharding(self.mesh, spec, memory_kind=emb_host.memory_kind)
        )

    def _finalize_checkpoint_load(
        self,
        ctx: TrainerContext,
        res: tuple[bool, int, int],
    ) -> tuple[bool, int, int]:
        self._data_position = None
        if res[0] and ctx.checkpoint is not None:
            ckpt_path = ctx.checkpoint.path
            pos_path = os.path.join(ckpt_path, _DATA_POSITION_FILENAME)
            rank_logger.info("Looking for data position at %s", pos_path)
            if os.path.isfile(pos_path):
                with open(pos_path) as f:
                    self._data_position = json.loads(f.read())
                rank_logger.info(
                    "Restored data position from filesystem checkpoint: %s, "
                    "current read_bsz_per_process=%d, data_world_size=%d",
                    self._data_position,
                    self.read_bsz_per_process,
                    self.data_world_size,
                )
            else:
                rank_logger.info("No data_position.json found at %s", pos_path)

        assert isinstance(self.state, RecsysTrainingState)
        if self.state.offset_keys is not None:
            assert self.state.offset_values is not None
            k = np.array(self.state.offset_keys).tolist()
            v = np.array(self.state.offset_values).tolist()
            if k[0] != -1 and isinstance(self.dataset, PhoenixKafkaDataset):
                ckpt_topic, ckpt_partitions = _read_checkpoint_kafka_config(ctx)
                current_topic = self.dataset.topic_name
                current_partitions = self.dataset.num_kafka_partitions

                skip_reason: str | None = None
                if ckpt_topic is not None and ckpt_topic != current_topic:
                    skip_reason = (
                        f"topic mismatch: checkpoint={ckpt_topic!r} != current={current_topic!r}"
                    )
                elif (
                    ckpt_partitions is not None
                    and current_partitions is not None
                    and ckpt_partitions != current_partitions
                ):
                    skip_reason = (
                        f"partition count changed: checkpoint={ckpt_partitions} "
                        f"!= current={current_partitions}"
                    )

                if skip_reason is not None:
                    rank_logger.warning(
                        "Skipping Kafka offset restore: %s. Consumer will start from %s.",
                        skip_reason,
                        self.dataset.auto_offset_reset,
                    )
                else:
                    if ckpt_topic is None:
                        rank_logger.info(
                            "No kafka config in checkpoint config.json (legacy). "
                            "Applying Kafka offsets as-is."
                        )
                    else:
                        rank_logger.info(
                            "Checkpoint kafka config matches: topic=%r, partitions=%s",
                            ckpt_topic,
                            ckpt_partitions,
                        )
                    seek_to_offset = dict(zip(k, v))
                    self.dataset.seek_to_offset = seek_to_offset
                    rank_logger.info(
                        "Restored Kafka offsets for %d partitions", len(seek_to_offset)
                    )
                    own_restart = ctx.checkpoint is not None and not ctx.checkpoint.is_manual_load()
                    if own_restart and restart_from_restored_offsets(self.dataset):
                        rank_logger.info(
                            "Resuming this run's own Kafka offsets; dropped the first-start "
                            "position (seek_to_timestamp_ms / fresh_training_bootstrap)"
                        )

        if self.smoothing_windows and isinstance(self.model_config, RecsysAggregatedModelConfig):
            cells = self._metric_ema_cells()
            loaded_rce = self.state.rce_ema
            reconciled_rce: dict[str, jax.Array] = {}
            for e, m in cells:
                for ws in self.smoothing_windows:
                    key = f"{e}/{m}/{ws}"
                    if loaded_rce and key in loaded_rce and loaded_rce[key].shape == (3,):
                        reconciled_rce[key] = loaded_rce[key]
                    else:
                        reconciled_rce[key] = jnp.zeros((3,), dtype=jnp.float32)
            for key, zeros in {
                **self._value_head_ema_keys(),
                **self._conversion_delay_slice_ema_keys(3),
            }.items():
                loaded = loaded_rce.get(key) if loaded_rce else None
                reconciled_rce[key] = (
                    loaded if loaded is not None and loaded.shape == zeros.shape else zeros
                )

            loaded_calib = self.state.calib_ema
            reconciled_calib: dict[str, jax.Array] = {}
            for e, m in cells:
                for ws in self.smoothing_windows:
                    key = f"{e}/{m}/{ws}"
                    if loaded_calib and key in loaded_calib and loaded_calib[key].shape == (2,):
                        reconciled_calib[key] = loaded_calib[key]
                    else:
                        reconciled_calib[key] = jnp.zeros((2,), dtype=jnp.float32)
            for key, zeros in self._conversion_delay_slice_ema_keys(2).items():
                loaded = loaded_calib.get(key) if loaded_calib else None
                reconciled_calib[key] = (
                    loaded if loaded is not None and loaded.shape == zeros.shape else zeros
                )

            self.state = self.state._replace(
                rce_ema=reconciled_rce,
                calib_ema=reconciled_calib,
            )

        if (
            res[0]
            and self.step_count_start is not None
            and ctx.checkpoint is not None
            and ctx.checkpoint.is_manual_load()
        ):
            rank_logger.info(
                "step_count_start=%d with manual load: resetting step and elapsed counters",
                self.step_count_start,
            )
            self.state = self.state._replace(step=jnp.array(self.step_count_start))
            res = (True, 0, 0)

        if self.forward_only:
            res = self._resume_forward_only(res)

        return res

    def purge_opt_state_on_load(self, host_state):
        if getattr(self.checkpoint_config, "keep_emb_opt_state", False):
            rank_logger.info("Not loading dense optimizer state (keeping emb_table_state)")
            return host_state._replace(opt_state=None)
        return super().purge_opt_state_on_load(host_state)

    def warm_start_staging_spec(self):
        if getattr(self.checkpoint_config, "keep_emb_opt_state", False):
            return (lambda tree: tree._replace(opt_state=None)), {"opt_state"}
        return super().warm_start_staging_spec()

    def restore_checkpoint_arrays(self, path, arrays, load_mask, rename, tag, on_replaced):
        if not self.use_row_emb and not isinstance(self.model_config, RecsysAggregatedModelConfig):
            return {}
        from xrex.train.row_embedding_restore import restore_row_embeddings

        return restore_row_embeddings(
            path,
            arrays,
            load_mask,
            rename,
            tag,
            on_replaced,
            logical_rows=self.dataset.input_vocab_size,
            row_sharded=self.use_row_emb,
            verify_checksums=self.checkpoint_config.verify_checksums,
        )

    def maybe_load_checkpoint(self, ctx: TrainerContext, tag=None):
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)

        if self.store_load:
            raise ValueError(
                f"store_load={self.store_load!r} is not supported by the trainer; "
                "restore from a filesystem checkpoint with load=."
            )

        res = super().maybe_load_checkpoint(ctx, tag)
        return self._finalize_checkpoint_load(ctx, res)

    def next_data_batch(self):
        if self._shmem_write_future is not None and self._shmem_write_future.done():
            self._pending_shmem_ckpt_write_s, self._pending_checksum_s = (
                self._shmem_write_future.result()
            )
            self._shmem_write_future = None

        if not self.use_async_emb:
            return super().next_data_batch()

        if self._batch_pipeline.exhausted:
            raise StopIteration

        assert isinstance(self.dataset, PhoenixDataset)
        dataset, load = self.dataset, super().next_data_batch

        def fetch() -> FetchedBatch:
            batch = load()
            return FetchedBatch(batch, dict(self.offsets_to_commit), dataset.get_data_position())

        if self._batch_pipeline.reserve is None:
            if self.seqpack_candidate_packing and len(self.seqpack_candidate_count) > 1:
                if not self._pin_step_scratch():
                    self._reserve_small_buffer_region()
            self._batch_pipeline.reserve = fetch()

        self._batch_pipeline.current = self._batch_pipeline.reserve

        try:
            self._batch_pipeline.reserve = fetch()
        except StopIteration:
            self._batch_pipeline.exhausted = True

        self.offsets_to_commit = self._batch_pipeline.current.offsets

        assert self._async_emb_context is not None
        ready_step = recsys_async_emb.kernel_api(self._async_emb_context).wait_step_ready(
            self._async_emb_context.context_id
        )
        assert ready_step != 0 or self._async_emb_lookup_pin is None
        return self._batch_pipeline.current.batch

    def _checkpoint_data_position(self) -> DataPosition | None:
        assert isinstance(self.dataset, PhoenixDataset)
        if not self.use_async_emb or self._batch_pipeline.current is None:
            return self.dataset.get_data_position()
        return self._batch_pipeline.current.data_position

    def _maybe_build_stablehlo_bundle(self) -> list | None:
        if not self.export_stablehlo_bundle:
            return None
        if self._stablehlo_bundle_files is None:
            from xrex.train.recsys_bundle_export import build_bundle

            try:
                start = time.perf_counter()
                self._stablehlo_bundle_files = build_bundle(self)
                rank_logger.info(
                    "Built StableHLO bundle (%d files, %.1fs); it will be included in "
                    "every copy_port checkpoint publish",
                    len(self._stablehlo_bundle_files),
                    time.perf_counter() - start,
                )
            except Exception:
                rank_logger.exception(
                    "StableHLO bundle export failed; disabling for the rest of this run "
                    "(copy_port checkpoints continue without export/)"
                )
                self._stablehlo_bundle_files = []
        if self._engine is None:
            return None
        return self._stablehlo_bundle_files or None

    def _write_stablehlo_bundle_files(self, prefix: str, bundle_files: list | None) -> None:
        from xrex.train.recsys_bundle_export import MANIFEST_NAME, restamp_manifest

        try:
            for bundle_file in bundle_files or ():
                data = bundle_file.data
                if bundle_file.name == MANIFEST_NAME:
                    data = restamp_manifest(data)
                bundle_path = f"{OUT_PATH}/.{prefix}/{bundle_file.name}"
                os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
                _write_all_bytes(bundle_path, memoryview(data))
        except OSError:
            rank_logger.exception(
                "StableHLO bundle write failed; disabling for the rest of this run "
                "(this publish continues without export/)"
            )
            self._stablehlo_bundle_files = []
            for subdir in {f.name.split("/", 1)[0] for f in bundle_files or ()}:
                shutil.rmtree(f"{OUT_PATH}/.{prefix}/{subdir}", ignore_errors=True)

    def _write_shmem_checkpoint(
        self,
        write_items: list[tuple[str, typing.Any, npt.NDArray]],
        data_pos: typing.Any,
        prefix: str,
        stub: typing.Any,
        bundle_files: list | None = None,
    ) -> tuple[float, float]:
        start = time.perf_counter()
        proc_idx = jax.process_index()
        proc_count = jax.process_count()
        has_engine = self._engine is not None
        devices_per_node = self.parallel_config.num_devices_per_node
        use_rdma = os.environ.get("XAI_RECSYS_RDMA") == "1"

        for key, value, arr in write_items:
            shard = value.addressable_shards[0]
            sharded = [(x.start, x.stop, x.step) != (None,) * 3 for x in shard.index]
            path = f"{OUT_PATH}/.{prefix}/{key}/c"
            if any(sharded):
                err = NotImplementedError(f"unsupported sharding in tensor {key}: {shard.index}")
                if sharded not in ([False, True], [True, False]):
                    raise err
                i = sharded.index(True)
                if shard.index[i].step is not None:
                    raise err
                delta = shard.index[i].stop - shard.index[i].start
                if shard.index[i].start % delta != 0 or value.shape[i] % delta != 0:
                    raise err
                shard_idx = shard.index[i].start // delta
                num_shards = value.shape[i] // delta
                if proc_count % num_shards != 0:
                    raise err
                path += (f"/{shard_idx}/0", f"/0/{shard_idx}")[i]
                tensor_written = proc_idx % (proc_count // num_shards) == 0
            else:
                suffix = "/0" * value.ndim
                path += suffix
                tensor_written = has_engine
            if tensor_written:
                os.makedirs(path[: path.rindex("/")], exist_ok=True)
                mv = memoryview(np.ndarray(shape=arr.nbytes, dtype=np.uint8, buffer=arr))
                _write_all_bytes(path, mv)
                actual_size = os.path.getsize(path)
                if actual_size != arr.nbytes:
                    raise RuntimeError(
                        f"bad checkpoint shard size for {key} at {path}: "
                        f"wanted {arr.nbytes}, got {actual_size}"
                    )
                if use_rdma and any(sharded) and not key.startswith("emb_table_state"):
                    d = proc_idx * devices_per_node // proc_count
                    request = copy_pb2.RegisterRequest(
                        path=f"{OUT_PATH}/.",
                        name=path[len(OUT_PATH) + 2 :],
                        rdma_device_indexes=bytes([d]),
                    )
                    try:
                        stub.Register(request)
                    except Exception as e:
                        raise RuntimeError(repr(e))

        from jax._src.distributed import global_state as _jax_dist

        client = _jax_dist.client
        assert client is not None, "JAX distributed not initialized"
        checksum_start = time.perf_counter()
        checksums, local_parts = _local_checksums(write_items, checksum_replicated=has_engine)
        checksum_s = time.perf_counter() - checksum_start
        client.key_value_set(
            f"{_CHECKSUM_PARTS_KV_DIR}/{proc_idx}",
            json.dumps([prefix, local_parts]),
            allow_overwrite=True,
        )
        client.wait_at_barrier(f"recsys-save-checkpoint2-{prefix}", timeout_in_ms=300_000)

        if has_engine:
            published = dict(client.key_value_dir_get(f"{_CHECKSUM_PARTS_KV_DIR}/"))
            parts_by_process = []
            for process in range(proc_count):
                published_prefix, process_parts = json.loads(
                    published[f"{_CHECKSUM_PARTS_KV_DIR}/{process}"]
                )
                assert published_prefix == prefix, f"stale checksum parts of process {process}"
                parts_by_process.append(process_parts)
            checksums = _combine_checksums(write_items, checksums, parts_by_process)
            with open(f"{OUT_PATH}/.{prefix}/checksums.0.json", "w") as f:
                json.dump(
                    {
                        "global_checksums": checksums,
                        "created_timestamp": time.time(),
                    },
                    f,
                )
            self._write_stablehlo_bundle_files(prefix, bundle_files)
            if data_pos is not None:
                with open(f"{OUT_PATH}/.{prefix}/{_DATA_POSITION_FILENAME}", "w") as f:
                    json.dump(data_pos, f)
                rank_logger.info(
                    "Saved data position (engine) to %s/.%s: %s", OUT_PATH, prefix, data_pos
                )
            name = prefix[: prefix.index("/")]
            all_names = os.listdir(OUT_PATH)
            old_entries = sorted(
                x for x in all_names if x.startswith("elapsed_samples") and x < name
            )
            keep_old = max(self.checkpoint_config.shm_max_entries - 1, 0)
            expired = old_entries if keep_old == 0 else old_entries[:-keep_old]
            for x in expired + [
                x
                for x in all_names
                if (x.startswith("elapsed_samples") and x >= name)
                or (x.startswith(".elapsed_samples") and x != f".{name}")
            ]:
                shutil.rmtree(f"{OUT_PATH}/{x}", ignore_errors=True)
            suffix = prefix[prefix.index("/") + 1 :]
            for x in os.listdir(f"{OUT_PATH}/.{name}"):
                if x != suffix:
                    shutil.rmtree(f"{OUT_PATH}/.{name}/{x}", ignore_errors=True)
            os.rename(f"{OUT_PATH}/.{name}", f"{OUT_PATH}/{name}")

        return time.perf_counter() - start, checksum_s

    def save_checkpoint(self, *args, **kwargs):
        if self.forward_only:
            return None
        if not self.dense_params_ema_in_checkpoint or self._dense_params_ema is None:
            return self._save_checkpoint_current_params(*args, **kwargs)
        live_params = self.state.params
        self.state = self.state._replace(
            params=self._dense_params_ema_as_params_jit(self._dense_params_ema, live_params)
        )
        try:
            return self._save_checkpoint_current_params(*args, **kwargs)
        finally:
            self.state = self.state._replace(params=live_params)

    def _save_checkpoint_current_params(self, *args, **kwargs):
        if not gc.get_freeze_count():
            gc.collect()
            gc.freeze()
            rank_logger.info("Froze %d objects for cyclic GC", gc.get_freeze_count())

        dataset = self.dataset
        assert isinstance(dataset, PhoenixDataset), f"Got {type(dataset)}"
        assert isinstance(self.state, RecsysTrainingState), f"Got {type(self.state)}"

        if self.offsets_to_commit or (
            isinstance(dataset, RustKafkaDataset)
            and dataset.redistribution_reader_count(jax.process_count())
        ):
            items_list = list(self.offsets_to_commit.items())

            n_partitions = dataset.num_kafka_partitions
            assert n_partitions is not None, "num_kafka_partitions must be set for Kafka offsets"
            max_size = (n_partitions + jax.process_count() - 1) // jax.process_count()
            if isinstance(dataset, RustKafkaDataset) and dataset.redistribution_reader_count(
                jax.process_count()
            ):
                max_size = n_partitions

            items_padded = items_list + [(-1, 0)] * (max_size - len(items_list))

            items_array = np.array(items_padded, dtype=np.int64)

            all_offsets = multihost_utils.process_allgather(items_array, tiled=True)

            if np.any(all_offsets[:, 0] >= 0):
                u = collections.defaultdict(int)
                for k, v in all_offsets:
                    u[k] = max(v, u[k])

                n = dataset.num_kafka_partitions
                assert n is not None
                _k = np.arange(n, dtype=all_offsets.dtype)
                _v = np.array([u.get(k, 0) for k in range(n)], dtype=all_offsets.dtype)

                offset_keys: jax.Array = multihost_utils.host_local_array_to_global_array(
                    _k, self.mesh, P()
                )
                offsets_values: jax.Array = multihost_utils.host_local_array_to_global_array(
                    _v, self.mesh, P()
                )

                self.state = self.state._replace(offset_keys=offset_keys)
                self.state = self.state._replace(offset_values=offsets_values)

        self.maybe_build_retrieval_post_embeddings()

        force_disk_save = bool(kwargs.pop("force_disk_save", False))
        state_step = int(np.asarray(self.state.step).item())
        if self.max_steps is not None and state_step > self.max_steps:
            force_disk_save = True
        if self.max_samples is not None and self.elapsed_samples > self.max_samples:
            force_disk_save = True

        should_persist_disk = True
        disk_every_s = self.checkpoint_config.checkpoint_disk_every_s
        now = time.time()
        port = self.checkpoint_config.copy_port
        elapsed_since_disk_ckpt = -1.0
        if port and disk_every_s > 0:
            if force_disk_save:
                should_persist_disk = True
            elif jax.process_index() == 0 and self._last_disk_checkpoint_ts > 0:
                elapsed_since_disk_ckpt = now - self._last_disk_checkpoint_ts
                should_persist_disk = elapsed_since_disk_ckpt >= disk_every_s

            should_persist_disk = bool(
                multihost_utils.broadcast_one_to_all(
                    np.array([1 if should_persist_disk else 0], dtype=np.int32)
                )[0]
            )
            if not should_persist_disk and jax.process_index() == 0:
                rank_logger.info(
                    "Skipping disk checkpoint save at step=%s, elapsed_samples=%s; "
                    "copy_port publish continues (checkpoint_disk_every_s=%ss, %.1fs since last disk save)",
                    state_step,
                    self.elapsed_samples,
                    disk_every_s,
                    elapsed_since_disk_ckpt,
                )

        if port:
            if self._engine is None and self._is_copy_port_binder():
                cc = self.checkpoint_config
                self._engine = xai_recsys_engine.RecsysPredictorServer(
                    port,
                    port + 1,
                    1,
                    0,
                    copy_max_entries=cc.shm_max_entries,
                    tls_cert_path=cc.copy_port_tls_cert or None,
                    tls_key_path=cc.copy_port_tls_key or None,
                    tls_client_ca_path=cc.copy_port_tls_client_ca or None,
                )
            multihost_utils.sync_global_devices("recsys-copy-port-bind")

            checkpoint_path = self.get_checkpoint_path(self.ctx)
            prefix = "/".join(checkpoint_path.split("/")[-2:])
            wait_until_finished()
            if self._shmem_write_future is not None:
                self._pending_shmem_ckpt_write_s, self._pending_checksum_s = (
                    self._shmem_write_future.result()
                )
                self._shmem_write_future = None
            multihost_utils.sync_global_devices("recsys-save-checkpoint1")
            stub = copy_pb2_grpc.CopyStub(self._copy_port_channel(port))

            serving_state = self.state.purge_opt_state()
            if not hasattr(self, "host_state") or self.host_state is None:
                self.host_state = jax.device_put(
                    serving_state, self.host_sharding.purge_opt_state()
                )
            else:
                self.host_state = self.offload_state(serving_state, self.host_state)
            jax.block_until_ready(self.host_state)

            assert isinstance(self.state, RecsysTrainingState)
            host_state = tree_to_dict(unwrap_tree(self.host_state))

            ep = self.mesh.shape["expert"]

            if self._copy_port_emb_host_spec() is not None:
                pass
            else:
                emb = _pad_vocab_for_ep(self.state.emb_table.x, ep, "emb_table")
                sharding = NamedSharding(self.mesh, P("expert", None))
                emb_table = jax.lax.with_sharding_constraint(emb, sharding.spec)
                host_state["emb_table"] = jax.device_put(
                    emb_table,
                    sharding.with_memory_kind("pinned_host"),
                )

            pe_key = "post_embeddings.embeddings"
            if (
                isinstance(self.model_config, RecsysTwoTowerModelConfig)
                and pe_key in host_state
                and host_state[pe_key] is not None
            ):
                pe_padded = _pad_vocab_for_ep(host_state[pe_key], ep, pe_key)
                if pe_padded is not host_state[pe_key]:
                    host_state[pe_key] = jax.device_put(
                        pe_padded,
                        pe_padded.sharding.with_memory_kind("pinned_host"),
                    )

            write_items: list[tuple[str, typing.Any, npt.NDArray]] = []
            for key in host_state:
                value = host_state[key]
                if value is None:
                    continue
                shards = value.addressable_shards
                if len(shards) != 1:
                    continue
                write_items.append((key, value, _unsafe_jax2np(shards[0].data)))

            data_pos = self._checkpoint_data_position() if self._engine is not None else None

            bundle_files = self._maybe_build_stablehlo_bundle()

            self._shmem_write_future = self._shmem_write_pool.submit(
                self._write_shmem_checkpoint,
                write_items,
                data_pos,
                prefix,
                stub,
                bundle_files,
            )

        if port and not should_persist_disk:
            return None

        result = super().save_checkpoint(*args, **kwargs)
        self._last_disk_checkpoint_ts = now
        data_pos = self._checkpoint_data_position()
        if data_pos is not None:
            ckpt_path = self.get_checkpoint_path(self.ctx)
            pos_path = os.path.join(ckpt_path, _DATA_POSITION_FILENAME)
            os.makedirs(ckpt_path, exist_ok=True)
            with open(pos_path, "w") as f:
                json.dump(data_pos, f)
            rank_logger.info("Saved data position to %s: %s", pos_path, data_pos)

        gc_start = time.perf_counter()
        gc.collect()
        self._pending_gc_collect_s = time.perf_counter() - gc_start

        return result

    def run(self) -> None:
        gc.disable()
        rank_logger.info("Disabled Python cyclic GC for the recsys training loop")
        run_error = None
        try:
            self._maybe_start_retrieval_corpus_stager()
            super().run()
        except BaseException as error:
            run_error = error
        finally:
            cleanup_errors = []
            try:
                self._stop_retrieval_corpus_stager()
            except BaseException as error:
                cleanup_errors.append(error)

            failed = (
                run_error is not None and not isinstance(run_error, StopIteration)
            ) or cleanup_errors
            if failed and self._async_emb_context is not None:
                try:
                    recsys_async_emb.kernel_api(self._async_emb_context).abort(
                        self._async_emb_context.context_id
                    )
                except BaseException as error:
                    cleanup_errors.append(error)

            shutdown = getattr(self.dataset, "shutdown", None)
            if callable(shutdown):
                rank_logger.info("Stopping dataset background threads via shutdown()")
                try:
                    shutdown()
                except BaseException as error:
                    cleanup_errors.append(error)

            if cleanup_errors:
                errors = ([run_error] if run_error is not None else []) + cleanup_errors
                raise BaseExceptionGroup("Recsys training or cleanup failed", errors)
            if run_error is not None:
                raise run_error.with_traceback(run_error.__traceback__)

    def two_int32_to_int64(self, two_int32: jax.Array | np.ndarray) -> npt.NDArray[np.int64]:
        low_64 = np.asarray(two_int32[:, 0], dtype=np.int64) & 0xFFFFFFFF
        high_64 = np.asarray(two_int32[:, 1], dtype=np.int64) << 32
        return low_64 | high_64

    def int64_to_two_int32(self, post_ids: npt.NDArray[np.int64]) -> jax.Array:
        low_32 = jnp.asarray(post_ids & 0xFFFFFFFF, dtype=jnp.int32)
        high_32 = jnp.asarray(post_ids >> 32, dtype=jnp.int32)
        return jnp.stack([low_32, high_32], axis=1)

    def _with_dense_params_ema(self, update_fn):
        decay = self.dense_params_ema_decay
        params_sharding = self.state_sharding.params

        def ema_init(params):
            return jax.tree.map(lambda p: jnp.copy(p.astype(jnp.float32)), params)

        def ema_update(ema, params):
            return jax.tree.map(
                lambda e, p: decay * e + (1.0 - decay) * p.astype(jnp.float32), ema, params
            )

        def ema_gap(ema, params):
            pairs = zip(jax.tree.leaves(ema), jax.tree.leaves(params))
            gap_sq = sum(jnp.sum(jnp.square(e - p.astype(jnp.float32))) for e, p in pairs)
            norm_sq = sum(
                jnp.sum(jnp.square(p.astype(jnp.float32))) for p in jax.tree.leaves(params)
            )
            return jnp.sqrt(gap_sq), jnp.sqrt(norm_sq)

        self._dense_params_ema_init_jit = jax.jit(
            ema_init, in_shardings=(params_sharding,), out_shardings=params_sharding
        )
        self._dense_params_ema_update_jit = jax.jit(
            ema_update,
            in_shardings=(params_sharding, params_sharding),
            out_shardings=params_sharding,
            donate_argnums=(0,),
        )
        self._dense_params_ema_gap_jit = jax.jit(
            ema_gap, in_shardings=(params_sharding, params_sharding)
        )
        self._dense_params_ema_as_params_jit = jax.jit(
            lambda ema, params: jax.tree.map(lambda e, p: jnp.copy(e).astype(p.dtype), ema, params),
            in_shardings=(params_sharding, params_sharding),
            out_shardings=params_sharding,
        )

        def update_with_dense_params_ema(state, *args):
            state, metrics, extras = update_fn(state, *args)
            if self._dense_params_ema is None:
                self._dense_params_ema = self._dense_params_ema_init_jit(state.params)
                num_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(state.params))
                rank_logger.info(
                    f"dense params EMA started: decay {decay}, {num_params:,} params "
                    f"({num_params * 4 / 2**30:.2f} GiB in float32)"
                )
            else:
                self._dense_params_ema = self._dense_params_ema_update_jit(
                    self._dense_params_ema, state.params
                )
            return state, metrics, extras

        return update_with_dense_params_ema

    def eval(self, soft_step: int):
        if self._dense_params_ema is None:
            return self._eval_current_params(soft_step)
        live_params = self.state.params
        gap, norm = self._dense_params_ema_gap_jit(self._dense_params_ema, live_params)
        self.state = self.state._replace(
            params=jax.tree.map(lambda e, p: e.astype(p.dtype), self._dense_params_ema, live_params)
        )
        try:
            metrics = dict(self._eval_current_params(soft_step))
        finally:
            self.state = self.state._replace(params=live_params)
        metrics["dense_params_ema/relative_gap"] = float(gap) / max(float(norm), 1e-12)
        return metrics

    def _eval_current_params(self, soft_step: int):
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return self.eval_two_tower(soft_step)

        raise ValueError("Ranking model eval_every_n is not supported yet.")

    def _split_home_checkpoint(self) -> bool:
        return bool(self.split_home_checkpoint) or bool(
            getattr(self.model_config, "split_home_checkpoint", False)
        )

    def _retrieval_checkpoint_datasets(self) -> list[RetrievalDataset]:
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        if self.model_config.checkpoint_dataset_names is not None:
            valid_names = set(RetrievalDataset.__members__.keys())
            for name in self.model_config.checkpoint_dataset_names:
                if name not in valid_names:
                    raise ValueError(
                        f"Unknown dataset name '{name}' in checkpoint_dataset_names. "
                        f"Valid names: {sorted(valid_names)}"
                    )
            target_datasets = [
                RetrievalDataset[name] for name in self.model_config.checkpoint_dataset_names
            ]
            if self._split_home_checkpoint():
                target_datasets = RetrievalDataset.expand_home_to_cold_hot(target_datasets)
            rank_logger.info(
                f"Loading configured retrieval datasets: {[ds.name for ds in target_datasets]}"
            )
        else:
            eval_target_types: set[RetrievalDataset] = set()
            for eval_module in self.evals:
                if isinstance(eval_module.eval_conf, RecsysTwoTowerEval):
                    eval_target_types.add(eval_module.eval_conf.target_dataset_type)
            target_datasets = (
                list(eval_target_types) if eval_target_types else [RetrievalDataset.HOME]
            )
            if self._split_home_checkpoint():
                target_datasets = RetrievalDataset.expand_home_to_cold_hot(target_datasets)
            rank_logger.info(
                f"Loading configured retrieval datasets: {[ds.name for ds in target_datasets]}"
            )

        return target_datasets

    def _retrieval_stager_config(
        self, target_datasets: list[RetrievalDataset] | None = None
    ) -> corpus_stager.StagerConfig | None:
        if not self.retrieval_staged_corpus:
            return None
        if not isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return None
        if target_datasets is None:
            target_datasets = self._retrieval_checkpoint_datasets()
        names = tuple(ds.name for ds in target_datasets)
        if corpus_stager.dataset_logical_paths(names) is None:
            return None
        user_cfg = self.model_config.user_tower_config
        return corpus_stager.StagerConfig(
            root=self.retrieval_staged_corpus_dir,
            dataset_names=names,
            max_posts=self.model_config.candidate_tower_config.max_posts,
            read_post_sid=user_cfg.use_post_sid,
            use_post_sid=user_cfg.use_post_sid,
            sid_num_levels=user_cfg.sid_num_levels,
            hash_table=corpus_stager.hash_table_config(
                self.model_config.candidate_tower_config.hash_table
            ),
            poll_seconds=self.retrieval_staged_corpus_poll_seconds,
        )

    def _maybe_start_retrieval_corpus_stager(self) -> None:
        config = self._retrieval_stager_config()
        if config is None or not self._is_copy_port_binder():
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
        self._retrieval_corpus_stager_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "xrex.train.retrieval_corpus_stager",
                "--config-json",
                config.to_json(),
                "--parent-pid",
                str(os.getpid()),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
        )
        rank_logger.info(
            "Started retrieval corpus stager pid=%d staging into %s",
            self._retrieval_corpus_stager_proc.pid,
            config.staging_dir(),
        )

    def _stop_retrieval_corpus_stager(self) -> None:
        proc = self._retrieval_corpus_stager_proc
        if proc is None:
            return
        deadline = time.monotonic() + 10.0
        proc.terminate()
        try:
            proc.wait(timeout=min(5.0, max(0.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"Corpus stager {proc.pid} did not exit within its cleanup deadline"
                ) from error
        self._retrieval_corpus_stager_proc = None

    def _acquire_staged_retrieval_corpus(
        self, config: corpus_stager.StagerConfig
    ) -> tuple[int, dict[str, np.ndarray]] | None:
        staging_dir = config.staging_dir()
        generation = corpus_stager.agree_on_generation(
            corpus_stager.list_staged_generations(staging_dir),
            multihost_utils.process_allgather,
        )
        arrays: dict[str, np.ndarray] | None = None
        if generation is not None:
            try:
                arrays = corpus_stager.open_staged_generation(staging_dir, generation)
                corpus_stager.validate_staged_arrays(
                    config,
                    generation,
                    arrays,
                    corpus_stager.read_staged_meta(staging_dir, generation),
                )
            except Exception:
                rank_logger.exception("Could not open staged corpus generation %s", generation)
                arrays = None
        opened = multihost_utils.process_allgather(np.array([arrays is not None], dtype=np.int32))
        if generation is None or arrays is None or not bool(np.all(opened)):
            rank_logger.info(
                "No staged retrieval corpus usable on every process (generation=%s); "
                "loading synchronously",
                generation,
            )
            return None
        return generation, arrays

    def _staged_post_table_ids(
        self, generation: int, staged_arrays: dict[str, np.ndarray]
    ) -> tuple[jax.Array, jax.Array, typing.Any, bool]:
        current = self.state.post_embeddings
        num_rows = staged_arrays["post_ids"].shape[0]
        current_ids = getattr(current, "post_ids", None)
        if (
            generation == self._post_table_ids_generation
            and current_ids is not None
            and tuple(current_ids.shape) == (num_rows, 2)
        ):
            return current.post_ids, current.author_ids, current.dataset_types, True
        global_post_ids = multihost_utils.host_local_array_to_global_array(
            _int64_as_int32_pairs(staged_arrays["post_ids"]), self.mesh, P(None)
        )
        global_author_ids = multihost_utils.host_local_array_to_global_array(
            _int64_as_int32_pairs(staged_arrays["author_ids"]), self.mesh, P(None)
        )
        dataset_types = np.array(staged_arrays["dataset_types"], dtype=np.int32).reshape(-1, 1)
        return global_post_ids, global_author_ids, dataset_types, False

    def _load_retrieval_corpus_synchronously(
        self, target_datasets: list[RetrievalDataset], max_posts: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, float]:
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        _user_cfg = self.model_config.user_tower_config
        _use_post_sid = _user_cfg.use_post_sid
        _sid_num_levels = _user_cfg.sid_num_levels
        use_objective_weights = bool(self.model_config.objective_head_weights)
        objective_bits = None
        if jax.process_index() == 0:
            corpus = RetrievalDataset.load_corpus(
                target_datasets,
                max_posts=max_posts,
                read_post_sid=_use_post_sid,
                sid_num_levels=_sid_num_levels,
                cold_start_max_age_seconds=float(
                    getattr(self.model_config, "cold_start_max_age_seconds", 0.0) or 0.0
                ),
                read_objective_bits=use_objective_weights,
            )
            post_ids, author_ids, dataset_types, post_sids_raw = corpus[:4]
            objective_bits = corpus.objective_bits
        else:
            post_ids = np.zeros(max_posts, dtype=np.int64)
            author_ids = np.zeros(max_posts, dtype=np.int64)
            dataset_types = np.zeros(max_posts, dtype=np.int32)
            post_sids_raw = None
            if use_objective_weights:
                objective_bits = np.zeros(max_posts, dtype=np.int32)
        load_done = time.perf_counter()
        if post_sids_raw is None:
            post_sids_raw = np.full((max_posts, _sid_num_levels), -1, dtype=np.int32)
        post_ids_i32 = post_ids.view(np.int32)
        author_ids_i32 = author_ids.view(np.int32)
        post_ids_i32, author_ids_i32, dataset_types, post_sids_raw, objective_bits = (
            multihost_utils.broadcast_one_to_all(
                (post_ids_i32, author_ids_i32, dataset_types, post_sids_raw, objective_bits)
            )
        )
        post_ids = np.asarray(post_ids_i32, dtype=np.int32).view(np.int64)
        author_ids = np.asarray(author_ids_i32, dtype=np.int32).view(np.int64)
        dataset_types = np.asarray(dataset_types, dtype=np.int32)
        post_sids_raw = np.asarray(post_sids_raw, dtype=np.int32)
        if objective_bits is not None:
            objective_bits = np.asarray(objective_bits, dtype=np.int32)
        return post_ids, author_ids, dataset_types, post_sids_raw, objective_bits, load_done

    def maybe_build_retrieval_post_embeddings(self):
        if not isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return

        assert isinstance(self.state, RecsysTrainingState)
        assert self.state.emb_table is not None

        build_start = time.perf_counter()
        target_datasets = self._retrieval_checkpoint_datasets()

        _user_cfg = self.model_config.user_tower_config
        _use_post_sid = _user_cfg.use_post_sid
        _sid_num_levels = _user_cfg.sid_num_levels
        max_posts = self.model_config.candidate_tower_config.max_posts
        objective_bits: np.ndarray | None = None
        stager_config = self._retrieval_stager_config(target_datasets)
        staged = (
            self._acquire_staged_retrieval_corpus(stager_config)
            if stager_config is not None
            else None
        )
        staged_arrays: dict[str, np.ndarray] | None = None
        if staged is not None:
            corpus_generation, staged_arrays = staged
            corpus_label = f"staged:{corpus_generation}"
            post_ids = staged_arrays["post_ids"]
            author_ids = staged_arrays["author_ids"]
            dataset_types = staged_arrays["dataset_types"]
            load_done = broadcast_done = time.perf_counter()
        else:
            corpus_label = "sync"
            post_ids, author_ids, dataset_types, post_sids_raw, objective_bits, load_done = (
                self._load_retrieval_corpus_synchronously(target_datasets, max_posts)
            )
            broadcast_done = time.perf_counter()
        total_samples = len(post_ids)
        shard_size = total_samples // self.data_world_size
        start_idx = self.data_rank * shard_size
        end_idx = (
            (self.data_rank + 1) * shard_size
            if self.data_rank < self.data_world_size - 1
            else total_samples
        )
        dataset_types_shard = dataset_types[start_idx:end_idx]

        if staged_arrays is not None:
            post_hashes_shard = np.asarray(staged_arrays["item_hashes"][start_idx:end_idx])
            author_hashes_shard = np.asarray(staged_arrays["author_hashes"][start_idx:end_idx])
            post_sids_u16_shard = np.asarray(staged_arrays["post_sids_u16"][start_idx:end_idx])
        else:
            _hash_table = self.model_config.candidate_tower_config.hash_table
            post_hashes_shard = _hash_table.get_item_hash(post_ids[start_idx:end_idx])
            author_hashes_shard = _hash_table.get_author_hash(author_ids[start_idx:end_idx])
            post_sids_raw_shard = post_sids_raw[start_idx:end_idx]
            if _use_post_sid:
                post_sids_u16_shard = (post_sids_raw_shard + 1).astype(np.uint16)
            else:
                post_sids_u16_shard = np.zeros_like(post_sids_raw_shard, dtype=np.uint16)
        if _user_cfg.use_post_embedding:
            combined_hashes_np = np.concatenate([post_hashes_shard, author_hashes_shard], axis=1)
        else:
            combined_hashes_np = author_hashes_shard
        post_hashes_np = np.asarray(post_hashes_shard)
        hash_done = time.perf_counter()

        rng, _new_rng = jax.random.split(self.state.rng)

        _shard_rows = combined_hashes_np.shape[0]
        if total_samples % self.data_world_size == 0:
            _chunk_rows = min(_shard_rows, self.retrieval_post_table_chunk_rows)
        else:
            _chunk_rows = _shard_rows
        if self._post_table_lookup_jit is None:
            embedding_param = replace(self.state.emb_table, x=None)

            def lookup(table, hashes):
                return self._lookup(replace(embedding_param, x=table), hashes).x

            self._post_table_lookup_jit = jax.jit(
                lookup,
                in_shardings=(self.state.emb_table.x.sharding, self.data_sharding),
                out_shardings=self.data_sharding,
            )
        emb_table_x = self.state.emb_table.x

        def _run_chunk(
            head_index: int, hashes: jax.Array, sids: jax.Array, post_hashes: jax.Array
        ) -> jax.Array:
            post_author_embedding = self._post_table_lookup_jit(emb_table_x, hashes)
            return self.candidate_tower_forward_jit(
                self.state.params,
                rng,
                post_author_embedding,
                sids,
                post_hashes,
                head_index,
            )

        def _forward_chunked_host(head_index: int) -> jax.Array:
            outs: list[np.ndarray] = []
            for _start in range(0, _shard_rows, _chunk_rows):
                _sl = slice(_start, min(_start + _chunk_rows, _shard_rows))
                _out = _run_chunk(
                    head_index,
                    *(
                        jax.make_array_from_process_local_data(self.data_sharding, x[_sl])
                        for x in (combined_hashes_np, post_sids_u16_shard, post_hashes_np)
                    ),
                )
                _local_shards = sorted(_out.addressable_shards, key=lambda s: s.index[0].start or 0)
                outs.append(np.concatenate([np.asarray(s.data) for s in _local_shards], axis=0))
                del _out
            return jax.make_array_from_process_local_data(
                self.data_sharding, np.concatenate(outs, axis=0)
            )

        def _forward_chunked(head_index: int) -> jax.Array:
            if total_samples % self.data_world_size == 0:
                result = _chunked_row_forward_on_device(
                    self.data_sharding,
                    (combined_hashes_np, post_sids_u16_shard, post_hashes_np),
                    functools.partial(_run_chunk, head_index),
                    _chunk_rows,
                )
                if result is not None:
                    return result
            return _forward_chunked_host(head_index)

        head_dataset_mapping = getattr(self.model_config, "head_dataset_mapping", None)
        num_heads = self.model_config.candidate_tower_config.num_candidate_heads
        candidate_embeddings = _forward_chunked(0)
        if head_dataset_mapping is not None and num_heads > 1:
            dataset_to_head: dict[int, int] = {}
            for ds_name, head_idx in head_dataset_mapping.items():
                dataset_to_head[RetrievalDataset[ds_name].value] = head_idx
            head_indices_shard = np.array(
                [dataset_to_head.get(int(dt), 0) for dt in dataset_types_shard],
                dtype=np.int32,
            ).reshape(-1, 1)
            post_head_jax = jax.make_array_from_process_local_data(
                self.data_sharding,
                head_indices_shard,
            )
            for h in range(1, num_heads):
                if h not in dataset_to_head.values():
                    continue
                emb_h = _forward_chunked(h)
                mask_h = post_head_jax == h
                candidate_embeddings = jnp.where(mask_h, emb_h, candidate_embeddings)
                del emb_h

        for blend_head_name in self.model_config.blend_heads:
            emb_blend = _forward_chunked(self.model_config.head_names.index(blend_head_name))
            candidate_embeddings = jax.device_put(
                jnp.concatenate([candidate_embeddings, emb_blend], axis=-1), self.data_sharding
            )
            del emb_blend

        self._eval_post_table = None
        if objective_bits is not None:
            slot_scales = self.model_config.objective_slot_scales(objective_bits[start_idx:end_idx])
            self._eval_post_table = candidate_embeddings
            candidate_embeddings = jax.jit(
                scale_post_table_slots, out_shardings=self.data_sharding
            )(
                candidate_embeddings,
                jax.make_array_from_process_local_data(self.data_sharding, slot_scales),
            )
            rank_logger.info(
                "Scaled post-table slots by objective_head_weights=%s",
                self.model_config.objective_head_weights,
            )

        jax.block_until_ready(candidate_embeddings)
        forward_done = time.perf_counter()

        ids_reused = False
        if staged_arrays is not None:
            global_post_ids, global_author_ids, combined_dataset_types, ids_reused = (
                self._staged_post_table_ids(corpus_generation, staged_arrays)
            )
        else:
            global_post_ids = multihost_utils.host_local_array_to_global_array(
                self.int64_to_two_int32(post_ids), self.mesh, P(None)
            )
            global_author_ids = multihost_utils.host_local_array_to_global_array(
                self.int64_to_two_int32(author_ids), self.mesh, P(None)
            )
            combined_dataset_types = multihost_utils.process_allgather(
                dataset_types_shard, tiled=True
            ).reshape(-1, 1)
        if isinstance(combined_dataset_types, np.ndarray):
            combined_dataset_types = replicated_global_array(combined_dataset_types, self.mesh)

        original_embeddings = self.state.post_embeddings.embeddings
        mol_side_table = self.state.post_embeddings.mol_side_table
        if mol_side_table is not None:
            if self.mol_side_table_jit is None:
                self.mol_side_table_jit = jax.jit(
                    self.mol_side_table_fn.apply,
                    in_shardings=(self.state_sharding.params, None, self.data_sharding),
                    out_shardings=NamedSharding(self.mesh, P(None, None)),
                )
            t_side = time.time()
            side_rows = self.mol_side_table_jit(self.state.params, None, candidate_embeddings)
            mol_side_table = replace(mol_side_table, x=side_rows, pspec=P(None, None))
            rank_logger.info(
                "MoL serving side table %s built in %.1fs",
                tuple(side_rows.shape),
                time.time() - t_side,
            )
        post_embeddings = PostEmbeddings(
            post_ids=global_post_ids,
            author_ids=global_author_ids,
            embeddings=replace(
                original_embeddings, x=candidate_embeddings, pspec=self.data_sharding.spec
            ),
            dataset_types=combined_dataset_types,
            mol_side_table=mol_side_table,
        )
        self.state = self.state._replace(post_embeddings=post_embeddings)
        self._post_table_ids_generation = corpus_generation if staged_arrays is not None else None
        build_done = time.perf_counter()
        rank_logger.info(
            "Built retrieval post table: corpus=%s rows=%d shard_rows=%d chunk_rows=%d "
            "load=%.2fs broadcast=%.2fs hash=%.2fs forward=%.2fs ids=%.2fs total=%.2fs "
            "ids_reused=%d",
            corpus_label,
            total_samples,
            _shard_rows,
            _chunk_rows,
            load_done - build_start,
            broadcast_done - load_done,
            hash_done - broadcast_done,
            forward_done - hash_done,
            build_done - forward_done,
            build_done - build_start,
            int(ids_reused),
        )

    def _eval_embeddings_fn(
        self,
    ) -> typing.Callable[[RecsysFeaturesBatch], RecsysEmbeddingsParameter]:
        if self._eval_lookup_jit is None:
            embedding_param = replace(self.state.emb_table, x=None)

            def lookup(table, batch):
                return self.get_recsys_embeddings(batch, replace(embedding_param, x=table))

            self._eval_lookup_jit = jax.jit(lookup, out_shardings=self.data_sharding)
        last: list[typing.Any] = []

        def embeddings(batch: RecsysFeaturesBatch) -> RecsysEmbeddingsParameter:
            if not last or last[0] is not batch:
                last[:] = [batch, self._eval_lookup_jit(self.state.emb_table.x, batch)]
            return last[1]

        return embeddings

    def eval_two_tower(self, soft_step: int):
        assert isinstance(self.state, RecsysTrainingState)
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        rank_logger.info("Running evals at %s", soft_step)
        rng, _new_rng = jax.random.split(self.state.rng)
        eval_embeddings = self._eval_embeddings_fn()

        def _two_tower_eval_forward_fn(
            query_head_weights,
            batch,
            post_embeddings,
            dataset_types,
            top_k,
            target_dataset_types,
            eval_bs_per_device=0,
        ):
            recsys_embeddings = get_recsys_embed_param_to_jax_array(eval_embeddings(batch))
            return self.two_tower_forward_jit(
                self.state.params,
                rng,
                batch,
                recsys_embeddings,
                post_embeddings,
                dataset_types,
                top_k,
                target_dataset_types,
                eval_bs_per_device,
                query_head_weights,
            )

        _eval_user_cfg = self.model_config.user_tower_config

        head_dataset_mapping = getattr(self.model_config, "head_dataset_mapping", None)

        def _make_candidate_tower_eval_fn(head_idx: int = 0):
            def _fn(candidate_embeddings, post_sids=None, post_hashes=None):
                if post_sids is None:
                    post_sids_local = jnp.zeros(
                        (candidate_embeddings.shape[0], _eval_user_cfg.sid_num_levels),
                        dtype=jnp.uint16,
                    )
                else:
                    post_sids_local = post_sids
                if post_hashes is None:
                    post_hashes_local = jnp.zeros(
                        (
                            candidate_embeddings.shape[0],
                            _eval_user_cfg.hash_table.hash_keys.num_item_hashes,
                        ),
                        dtype=jnp.int64,
                    )
                else:
                    post_hashes_local = post_hashes
                return self.candidate_tower_forward_jit(
                    self.state.params,
                    rng,
                    candidate_embeddings,
                    post_sids_local,
                    post_hashes_local,
                    head_idx,
                )

            return _fn

        _candidate_tower_eval_forward_fn = _make_candidate_tower_eval_fn(0)

        def _loss_fn(batch: RecsysFeaturesBatch):
            return self.loss_fn_eval_jit(self.state.params, rng, batch, eval_embeddings(batch))

        def _user_emb_fn(batch: RecsysFeaturesBatch):
            return self.user_forward_jit(self.state.params, rng, batch, eval_embeddings(batch))

        all_eval_metrics = {}
        if self.state.post_embeddings is None or self.state.post_embeddings.post_ids is None:
            return all_eval_metrics

        if not self._retrieval_post_emb_built:
            rank_logger.info(
                "Building retrieval post-embedding table for the first time "
                "(eval would otherwise score against an uninitialized table)."
            )
            self.maybe_build_retrieval_post_embeddings()
            self._retrieval_post_emb_built = True

        for eval_module in self.evals:
            eval_name, eval_conf, _, eval_bs_per_device, _ = (
                eval_module.eval_name,
                eval_module.eval_conf,
                eval_module.eval_dataset,
                int(eval_module.eval_bs_per_device),
                eval_module.eval_seq_len,
            )
            if not isinstance(eval_conf, RecsysTwoTowerEval):
                continue

            eval_head_idx = 0
            if head_dataset_mapping is not None:
                eval_head_idx = head_dataset_mapping.get(eval_conf.target_dataset_type.name, 0)
            eval_cand_fn = _make_candidate_tower_eval_fn(eval_head_idx)
            query_head_weights = eval_conf.query_head_weights

            eval_results = run_recsys_evals(
                evals=[(eval_name, eval_conf)],
                train_dataset=self.dataset_thread_iterator(self._eval_dataset, prepare_data=False),
                forward_fn=functools.partial(_two_tower_eval_forward_fn, query_head_weights),
                loss_fn=_loss_fn,
                mesh=self.mesh,
                rng=rng,
                candidate_tower_forward_fn=eval_cand_fn,
                all_post_embeddings=(
                    self._eval_post_table
                    if self._eval_post_table is not None
                    else self.state.post_embeddings.embeddings.x
                ),
                all_post_ids=self.two_int32_to_int64(self.state.post_embeddings.post_ids),
                dataset_types=self.state.post_embeddings.dataset_types,
                eval_bs_per_device=eval_bs_per_device,
                user_emb_fn=_user_emb_fn,
            )

            eval_metrics = report_forward_eval_results(results=eval_results)
            eval_metrics = merge_report_extras(eval_results, eval_metrics)
            for k, v in (getattr(eval_conf, "train_metrics", None) or {}).items():
                eval_metrics[f"{eval_name}/train/{k}"] = v
            rank_logger.info(f"eval_metrics: {eval_metrics}")
            all_eval_metrics.update(eval_metrics)
        return all_eval_metrics
