# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import collections
import concurrent.futures
import enum
import functools
import gc
import json
import logging
import math
import os
import shutil
import signal
import sys
import tempfile
import time
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import grpc
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import psutil
import xai_recsys_engine
from jax import shard_map
from jax.experimental import multihost_utils
from jax.sharding import (
    Mesh,
    NamedSharding,
)
from jax.sharding import PartitionSpec as P
from jax.tree_util import DictKey, GetAttrKey, SequenceKey
from xai_checkpointing import checksum
from xai_checkpointing.common import _unsafe_jax2np
from xai_checkpointing.tree_util import tree_to_dict
from xai_configlib import configclass
from xai_proto import copy_pb2, copy_pb2_grpc

from xrex.data.parquet_recsys import DataPosition, PhoenixDataset

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

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.data.recsys.sequence_packing import (
    FixedLengthDistribution,
    LengthDistribution,
    pack_batch,
)
from xrex.data.retrieval_dataset import RetrievalDataset
from xrex.data.streaming.kafkaloader import PhoenixKafkaDataset, report_training_metrics
from xrex.eval.eval_utils import report_forward_eval_results
from xrex.eval.metrics_recsys import merge_report_extras
from xrex.eval.recsys_eval import RecsysTwoTowerEval, run_recsys_evals
from xrex.models.compress_token_ids import compress_token_ids
from xrex.models.model_utils import Parameter, unwrap_tree
from xrex.models.recsys_embedding import (
    EmbTable,
    RecsysEmbeddings,
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_model import RecsysAggregatedModelConfig
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.optimizers.optim import InjectHyperparamsState, apply_updates
from xrex.optimizers.recsys import RecsysEmbeddingOptimConfig
from xrex.optimizers.recsys.async_emb_gradient_update import (
    AsyncEmbGradientUpdate,
    AsyncEmbOptimizer,
)
from xrex.train.misc import (
    PostEmbeddings,
    RecsysTrainingState,
)
from xrex.train.trainer import (
    RecsysTwoTowerModelConfig,
    Trainer,
    TrainerContext,
)
from xrex.utils import cluster, recsys_async_emb_lookup
from xrex.utils.aot import JittedOrCompiled
from xrex.utils.checkpoint_cloud import (
    collect_cloud_upload_files,
    ensure_orbax_dir_finalized,
)
from xrex.utils.checkpointing import wait_until_finished
from xrex.utils.metrics import norm_metrics
from xrex.utils.utils import (
    cast_bfloat16,
)

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


_filtered_nt_cache: dict[tuple, type] = {}


def _deep_filter_nones(tree):
    _asdict = getattr(tree, "_asdict", None)
    if _asdict is not None:
        items = {k: _deep_filter_nones(v) for k, v in _asdict().items() if v is not None}
        fields = tree._fields
        if len(items) == len(fields):
            return type(tree)(**items)
        cache_key = (type(tree), tuple(items.keys()))
        NT = _filtered_nt_cache.get(cache_key)
        if NT is None:
            NT = NamedTuple("X", [(k, type(v)) for k, v in items.items()])
            _filtered_nt_cache[cache_key] = NT
        return NT(**items)
    return tree


OUT_PATH = "/dev/shm"

_DATA_POSITION_FILENAME = "data_position.json"
_CONFIG_FILENAME = "config.json"


def _download_ocdbt_coordinator(cloud_url: str, local_dir: str) -> None:
    logger.info("[cloud-restore] downloading root metadata from %s", cloud_url)
    xai_recsys_engine.download_prefix_to_local(
        cloud_url,
        local_dir,
        32,
        shallow=True,
    )

    root_d_url = f"{cloud_url.rstrip('/')}/d"
    root_d_local = os.path.join(local_dir, "d")
    os.makedirs(root_d_local, exist_ok=True)
    logger.info("[cloud-restore] downloading root OCDBT data from %s", root_d_url)
    xai_recsys_engine.download_prefix_to_local(root_d_url, root_d_local, 32)


def _find_cross_process_files(local_dir: str, proc_idx: int) -> list[str]:
    from xrex.utils.ocdbt import load_shard_sources

    def _tensor_name(key: str) -> str:
        idx = key.find("/c/")
        if idx != -1:
            return key[:idx]
        return key

    proc_dir = os.path.join(local_dir, f"ocdbt.process_{proc_idx}")
    own_sources = load_shard_sources(proc_dir)
    own_tensors = {_tensor_name(entry[0]) for entry in own_sources}

    coordinator_sources = load_shard_sources(local_dir)

    my_prefix = f"ocdbt.process_{proc_idx}/"
    cross_process_files: set[str] = set()
    for entry in coordinator_sources:
        if len(entry) != 4:
            continue
        key, data_file, _, _ = entry
        tensor = _tensor_name(key)
        if tensor in own_tensors:
            continue
        if data_file.startswith(my_prefix):
            continue
        cross_process_files.add(data_file)

    cross_process_manifests: set[str] = set()
    for f in cross_process_files:
        proc_dir_name = f.split("/")[0]
        cross_process_manifests.add(f"{proc_dir_name}/manifest.ocdbt")

    result = sorted(cross_process_files | cross_process_manifests)

    missing_tensors = set()
    cross_bytes = 0
    for entry in coordinator_sources:
        if len(entry) != 4:
            continue
        key, data_file, _, size = entry
        tensor = _tensor_name(key)
        if tensor not in own_tensors and data_file in cross_process_files:
            missing_tensors.add(tensor)
            cross_bytes += size

    logger.info(
        "[cloud-restore] proc %d: own db has %d tensors, "
        "%d replicated tensors need cross-process download: %s "
        "(%d files, %.1f MB)",
        proc_idx,
        len(own_tensors),
        len(missing_tensors),
        sorted(missing_tensors),
        len(result),
        cross_bytes / (1024 * 1024),
    )
    return result


def _download_own_process_dir(cloud_url: str, local_dir: str, proc_idx: int) -> None:
    proc_subdir = f"ocdbt.process_{proc_idx}"
    proc_url = f"{cloud_url.rstrip('/')}/{proc_subdir}"
    proc_local = os.path.join(local_dir, proc_subdir)
    os.makedirs(proc_local, exist_ok=True)
    logger.info(
        "[cloud-restore] proc %d downloading own shard data from %s",
        proc_idx,
        proc_url,
    )
    n_files = xai_recsys_engine.download_prefix_to_local(proc_url, proc_local, 32)

    manifest = os.path.join(proc_local, "manifest.ocdbt")
    if n_files == 0 or not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"[cloud-restore] proc {proc_idx}: own process directory "
            f"{proc_subdir}/ is missing or incomplete on cloud "
            f"(downloaded {n_files} files from {proc_url}). "
            f"The checkpoint may have been saved with fewer processes, "
            f"or some workers failed to upload their data."
        )


def _download_cross_process_files(
    cloud_url: str,
    local_dir: str,
    proc_idx: int,
    files: list[str],
) -> None:
    for rel_path in files:
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    logger.info(
        "[cloud-restore] proc %d downloading %d cross-process files",
        proc_idx,
        len(files),
    )
    xai_recsys_engine.download_files_from_storage(
        cloud_url,
        files,
        local_dir,
        32,
    )


def _count_local_files(directory: str) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            count += 1
            total_bytes += os.path.getsize(fpath)
    return count, total_bytes


def _download_ocdbt_from_cloud(cloud_url: str, local_dir: str, proc_idx: int) -> None:
    start = time.time()

    _download_ocdbt_coordinator(cloud_url, local_dir)
    _download_own_process_dir(cloud_url, local_dir, proc_idx)
    cross_files = _find_cross_process_files(local_dir, proc_idx)
    if cross_files:
        _download_cross_process_files(cloud_url, local_dir, proc_idx, cross_files)

    num_files, total_bytes = _count_local_files(local_dir)
    elapsed = time.time() - start
    logger.info(
        "[cloud-restore] proc %d download complete: %d files, %.1f MB, %.1fs",
        proc_idx,
        num_files,
        total_bytes / (1024 * 1024),
        elapsed,
    )


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


def _pad_vocab_for_ep(emb: jax.Array, ep: int, name: str = "emb_table") -> jax.Array:
    V = emb.shape[0]
    target_V = -(-V // ep) * ep
    if target_V == V:
        return emb
    rank_logger.info("Padded %s rows %d -> %d for ep=%d save", name, V, target_V, ep)
    return jnp.pad(emb, ((0, target_V - V),) + ((0, 0),) * (emb.ndim - 1))


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
class RecsysTrainer(Trainer):
    offsets_to_commit: dict[int, int] = field(default_factory=dict)
    store_load: str | None = None
    exclude_nodes_file: str | None = None
    enable_metrics_file: bool = False
    emb_optim_config: RecsysEmbeddingOptimConfig = field(default_factory=RecsysEmbeddingOptimConfig)

    stop_at_data_end: bool = False

    empty_history_augmentation_rate: float = 0.0

    empty_history_user_dropout_rate: float = 0.0

    checkpoint_storage_urls: str = ""

    smoothing_windows: list[int] = field(default_factory=lambda: [1_048_576, 4_194_304])

    reset_data_position: bool = False

    seqpack_distribution: LengthDistribution | None = None

    seqpack_fixed_length: bool = False

    use_async_emb: bool = False

    _async_emb_context: AsyncEmbContextHandle | None = field(default=None, init=False, repr=False)
    _emb_hash_vocab: int = field(default=0, init=False, repr=False)
    _first_step_embedding_lookup_start_jit: typing.Any = field(default=None, init=False, repr=False)
    _empty_embedding_grad_update_jit: typing.Any = field(default=None, init=False, repr=False)
    _apply_deferred_embedding_update_jit: typing.Any = field(default=None, init=False, repr=False)
    _async_emb_step_jit: typing.Any = field(default=None, init=False, repr=False)
    _async_emb_lookup_pin: jax.Array | None = field(default=None, init=False, repr=False)
    _deferred_emb_grad_update: AsyncEmbGradientUpdate | None = field(
        default=None, init=False, repr=False
    )
    _batch_pipeline: BatchPipelineState = field(
        default_factory=BatchPipelineState, init=False, repr=False
    )

    _data_position: DataPosition | None = field(default=None, init=False, repr=False)
    _seqpack_rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    _history_user_dropout_rng: np.random.Generator | None = field(
        default=None, init=False, repr=False
    )
    _last_history_user_dropout_count: int = field(default=0, init=False, repr=False)
    _last_history_user_dropout_bsz: int = field(default=0, init=False, repr=False)
    _retrieval_post_emb_built: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.using_seqpack and self.seqpack_fixed_length:
            assert isinstance(self.model_config, RecsysAggregatedModelConfig)
            hl = self.model_config.history_seq_len
            self.seqpack_distribution = FixedLengthDistribution(min_len=hl, max_len=hl, mean_len=hl)

    state: RecsysTrainingState = field(init=False, repr=False, compare=False)
    _pending_shmem_ckpt_write_s: float | None = field(default=None, init=False, repr=False)
    _pending_checksum_s: float | None = field(default=None, init=False, repr=False)
    _pending_gc_collect_s: float | None = field(default=None, init=False, repr=False)

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
    forward_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    two_tower_forward_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    two_tower_forward_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
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
    _filtered_sharding: typing.Any = field(init=False, repr=False, compare=False, default=None)
    _compute_checksums_filtered_jit: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    host_state: typing.Any = field(init=False, repr=False, compare=False, default=None)

    def create_runtime(self, ctx):
        if os.environ.get("JAX_COORDINATOR_PORT"):
            ctx.coordinator_port = int(os.environ["JAX_COORDINATOR_PORT"])
        super().create_runtime(ctx)

    def example_data(self, bs: int) -> RecsysFeaturesBatch:
        assert isinstance(self.dataset, PhoenixDataset)
        batch = self.dataset.example_data(bs)
        if self.using_seqpack:
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            batch = pack_batch(
                batch=batch,
                num_devices_per_process=self.parallel_config.num_devices_per_process,
                num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                dist=self.seqpack_distribution,
                rng=np.random.default_rng(0),
            )
            if self.using_fa4:
                batch = self.add_block_sparse_layout(batch)
        return batch

    def prepare_data(self, batch):
        if self.using_seqpack:
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            batch = pack_batch(
                batch=typing.cast(RecsysFeaturesBatch, batch),
                num_devices_per_process=self.parallel_config.num_devices_per_process,
                num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                dist=self.seqpack_distribution,
                rng=self._seqpack_rng,
            )
            if self.using_fa4:
                batch = self.add_block_sparse_layout(batch)
        return super().prepare_data(batch)

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
        if self.emb_optim_config._active() == "rowwise_adagrad":
            sparse_emb_optim = self.emb_optim_config.make_optimizer(self.optim)
            emb_table_state = sparse_emb_optim.init({"table": emb_table})
        emb_table = replace(emb_table, x=emb_table.x.at[0, :].set(0))
        post_embeddings = PostEmbeddings(
            post_ids=jnp.arange(1).astype(jnp.int32),
            author_ids=jnp.arange(1).astype(jnp.int32),
            embeddings=Parameter(x=jnp.empty((1, 1), dtype=jnp.bfloat16), pspec=P(None, None)),
            dataset_types=jnp.array([RetrievalDataset.PAD.value], dtype=jnp.int32),
        )
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            post_embeddings = self.model_config.candidate_tower_config.make_post_embeddings()

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
        if self.smoothing_windows and isinstance(self.model_config, RecsysAggregatedModelConfig):
            from xrex.data.recsys.constants import engagement_to_ids
            from xrex.models.recsys_model import build_metric_masks

            eng_names = list(engagement_to_ids(self.model_config.metric_group).keys())
            dummy = jnp.zeros((1, 1))
            dummy_3d = jnp.zeros((1, 1, self.model_config.model_config.output_vocab_size))
            mask_keys = list(
                build_metric_masks(
                    dummy,
                    dummy_3d,
                    dummy,
                    dummy,
                    enable_platform_metrics=self.model_config.enable_platform_metrics,
                ).keys()
            )
            rce_ema = {
                f"{e}/{m}/{ws}": jnp.zeros((3,), dtype=jnp.float32)
                for e in eng_names
                for m in mask_keys
                for ws in self.smoothing_windows
            }

        calib_ema = None
        if rce_ema is not None:
            calib_ema = {
                f"{e}/{m}/{ws}": jnp.zeros((2,), dtype=jnp.float32)
                for e in eng_names
                for m in mask_keys
                for ws in self.smoothing_windows
            }

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

    def _get_embedding_hash_leaves(self, data: RecsysFeaturesBatch) -> list[jax.Array]:
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
        leaves: list[jax.Array] = []
        if use_user_embedding:
            leaves.append(data["user_hashes"])
        if use_post_embedding:
            leaves.append(data["history_seq"]["post_hashes"])
        leaves.append(data["history_seq"]["auth_hashes"])
        if use_post_embedding:
            leaves.append(data["candidate_seq"]["post_hashes"])
        leaves.append(data["candidate_seq"]["auth_hashes"])
        if use_ip:
            leaves.append(data["user_ip_hashes"])
        return leaves

    def get_recsys_embeddings(
        self, data: RecsysFeaturesBatch, emb_table: Parameter
    ) -> RecsysEmbeddingsParameter:
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
        data_axis = tuple(self.model_config.model_config.data_axis)

        hash_leaves = self._get_embedding_hash_leaves(data)

        flat_hashes = [x.reshape(self.batch_size, -1) for x in hash_leaves]
        segment_lengths = [x.shape[1] for x in flat_hashes]
        all_hashes = jax.lax.with_sharding_constraint(
            jnp.concatenate(flat_hashes, axis=1), P(data_axis)
        )
        all_embeddings = self._lookup(emb_table, all_hashes)
        all_embeddings = replace(
            all_embeddings,
            x=jax.lax.with_sharding_constraint(all_embeddings.x, P(data_axis, None, None)),
        )
        splits = jnp.split(all_embeddings.x, np.cumsum(segment_lengths[:-1]), axis=1)

        if self.using_seqpack:
            emb_splits = [
                split.reshape(*leaf.shape, all_embeddings.x.shape[-1])
                for split, leaf in zip(splits, hash_leaves)
            ]
        else:
            emb_splits = list(splits)

        idx = 0
        if use_user_embedding:
            user_x = emb_splits[idx]
            idx += 1
        else:
            user_x = None
        if use_post_embedding:
            history_post_x = emb_splits[idx]
            idx += 1
        else:
            history_post_x = None
        history_author_x = emb_splits[idx]
        idx += 1
        if use_post_embedding:
            candidate_post_x = emb_splits[idx]
            idx += 1
        else:
            candidate_post_x = None
        candidate_author_x = emb_splits[idx]
        idx += 1
        user_ip_x = emb_splits[idx] if use_ip else None

        return RecsysEmbeddingsParameter(
            user_embeddings=(replace(all_embeddings, x=user_x) if user_x is not None else None),
            history_post_embeddings=(
                replace(all_embeddings, x=history_post_x) if history_post_x is not None else None
            ),
            history_author_embeddings=replace(all_embeddings, x=history_author_x),
            candidate_post_embeddings=(
                replace(all_embeddings, x=candidate_post_x)
                if candidate_post_x is not None
                else None
            ),
            candidate_author_embeddings=replace(all_embeddings, x=candidate_author_x),
            user_ip_embeddings=(
                replace(all_embeddings, x=user_ip_x) if user_ip_x is not None else None
            ),
        )

    def _unflatten_emb_lookup(
        self, data: RecsysFeaturesBatch, table: Parameter
    ) -> RecsysEmbeddingsParameter:
        cfg = self.model_config
        has_emb_flags = isinstance(cfg, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
        hash_leaves = self._get_embedding_hash_leaves(data)
        lengths = [x.reshape(self.batch_size, -1).shape[1] for x in hash_leaves]
        splits = jnp.split(table.x, np.cumsum(lengths[:-1]), axis=1)

        if self.using_seqpack:
            splits = [
                s.reshape(*leaf.shape, table.x.shape[-1]) for s, leaf in zip(splits, hash_leaves)
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

    def _flatten_emb_grads(self, grads: RecsysEmbeddingsParameter) -> jax.Array:
        cfg = self.model_config
        has_emb_flags = isinstance(cfg, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
        enabled = [
            (grads.user_embeddings, has_emb_flags and cfg.use_user_embedding),
            (grads.history_post_embeddings, has_emb_flags and cfg.use_post_embedding),
            (grads.history_author_embeddings, True),
            (grads.candidate_post_embeddings, has_emb_flags and cfg.use_post_embedding),
            (grads.candidate_author_embeddings, True),
            (grads.user_ip_embeddings, has_emb_flags and cfg.use_ip_address),
        ]

        segments = [p.x for p, on in enabled if on and p is not None]
        width = segments[0].shape[-1]

        if self.using_seqpack:
            segments = [s.reshape(self.batch_size, -1, width) for s in segments]
        return jnp.concatenate(segments, axis=1).reshape(-1, width)

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

    def _maybe_inject_global_neg_embeddings(
        self, batch: RecsysFeaturesBatch
    ) -> RecsysFeaturesBatch:
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

    def dataset_with_prepare(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        assert isinstance(self.dataset, PhoenixDataset)
        aug_period = self._empty_history_augmentation_period()
        history_user_drop_rate = self.empty_history_user_dropout_rate
        for i, (batch, offsets) in enumerate(dataset):
            batch = self._maybe_inject_global_neg_embeddings(batch)

            if history_user_drop_rate > 0.0:
                self._last_history_user_dropout_bsz = batch["user_hashes"].shape[0]
                self._last_history_user_dropout_count = self._apply_per_history_user_dropout(
                    batch, history_user_drop_rate
                )

            yield self.local_global_hack(self.prepare_data(batch)), offsets

            if aug_period > 0 and i % aug_period == 0:
                for key in list(batch["history_seq"]):
                    if batch["history_seq"][key] is not None:
                        batch["history_seq"][key] = np.zeros_like(batch["history_seq"][key])
                if batch.get("user_hashes") is not None:
                    batch["user_hashes"] = np.zeros_like(batch["user_hashes"])
                if batch.get("user_ip_hashes") is not None:
                    batch["user_ip_hashes"] = np.zeros_like(batch["user_ip_hashes"])
                yield self.local_global_hack(self.prepare_data(batch)), offsets

    def dataset_without_prepare(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        assert isinstance(self.dataset, PhoenixDataset)
        for batch, offsets in dataset:
            if self.using_seqpack:
                assert isinstance(
                    self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
                )
                batch = pack_batch(
                    batch=batch,
                    num_devices_per_process=self.parallel_config.num_devices_per_process,
                    num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                    dist=self.seqpack_distribution,
                    rng=np.random.default_rng(0),
                )
                if self.using_fa4:
                    batch = self.add_block_sparse_layout(batch)
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
                "save_checkpoint timings: checksum=%.2fs, shmem_write=%.2fs, gc_collect=%.2fs",
                ckpt_s or 0,
                shmem_s or 0,
                gc_s or 0,
            )
            if ckpt_s is not None:
                metrics["save_checkpoint/checksum_s"] = ckpt_s
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

        return metrics

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
        train_dataset = self.dataset.make(
            batch_size=self.read_bsz_per_process,
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
            resume_position=resume_position,
        )
        self._eval_dataset = train_dataset

        if self.stop_at_data_end:
            assert isinstance(self.state, RecsysTrainingState)
            step = self.state.step
            data_end = self.dataset.compute_max_steps(
                num_shards=data_world_size,
                batch_size=self.read_bsz_per_process,
                current_step=step.item(),
                resume_position=resume_position,
            )
            if data_end is not None and (self.max_steps is None or data_end < self.max_steps):
                self.max_steps: int | None = data_end

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
            batch = self.example_data(self.read_bsz_per_process)

            assert self.batch_size % self.bs_per_device == 0, (self.batch_size, self.bs_per_device)
            assert batch["user_hashes"].shape[1] == self.bs_per_device, (
                batch["user_hashes"].shape[1],
                self.bs_per_device,
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

        batch_size = self.batch_size
        emb_size = self.model_config.emb_table_width
        user_emb_len = self.model_config.hash_table.num_user_hashes
        history_post_len = (
            self.model_config.hash_table.num_item_hashes * self.dataset.history_seq_len
        )
        history_author_len = (
            self.model_config.hash_table.num_author_hashes * self.dataset.history_seq_len
        )
        num_neg_blocks = 2 if self.dataset.search_query_embedding_dim > 0 else 1
        total_candidate_seq_len = (
            self.dataset.candidate_seq_len
            * (1 + num_neg_blocks * self.dataset.num_negatives_per_example)
            + self.dataset.num_global_negatives_per_example
        )
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

    def get_flattened_token_ids(self, data: RecsysFeaturesBatch) -> jax.Array:
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
        segments: list[jax.Array] = []
        if use_user_embedding:
            segments.append(data["user_hashes"])
        if use_post_embedding:
            segments.append(data["history_seq"]["post_hashes"])
        segments.append(data["history_seq"]["auth_hashes"])
        if use_post_embedding:
            segments.append(data["candidate_seq"]["post_hashes"])
        segments.append(data["candidate_seq"]["auth_hashes"])
        if use_ip:
            segments.append(data["user_ip_hashes"])
        return jnp.concatenate([x.reshape(self.batch_size, -1) for x in segments], axis=1)

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

        def _update(updated, original):
            return jnp.where(valid_step, updated, original)

        new_params = jax.tree.map(_update, new_params, state.params)
        new_opt_state = jax.tree.map(_update, new_opt_state, state.opt_state)

        examples = np.prod(data["user_hashes"].shape[:-1])

        metrics = {
            "step": state.step,
            "loss": loss,
            "examples_per_batch": examples,
            "valid_step": valid_step,
            "global_grad_norm": grad_norm,
            "learning_rate": lr * new_opt_state.hyperparams["learning_rate"],
            "weight_decay": new_opt_state.hyperparams["weight_decay"],
            "b1": new_opt_state.hyperparams["b1"],
            "b2": new_opt_state.hyperparams["b2"],
        }
        if self.track_norm_metrics:
            metrics.update(norm_metrics(new_params, new_opt_state, gradients, updates))

        new_rce_ema = stats.pop("_rce_ema", state.rce_ema)
        new_calib_ema = stats.pop("_calib_ema", state.calib_ema)
        metrics.update(**stats)

        segment_sum_result = self._segment_sum(emb_gradients, inverse_indices, num_unique)

        emb_gradients = replace(
            emb_gradients.history_author_embeddings,
            x=segment_sum_result,
        )

        emb_valid_step, emb_grad_norm = self.is_valid_step(emb_gradients)

        new_emb_table, emb_new_opt_state, emb_optim_metrics = self._emb_optim.sparse_update(
            grads=emb_gradients,
            full_state=state.emb_table_state,
            full_emb_table=emb_table,
            unique_tokens=unique_tokens,
            num_unique=num_unique,
            lr=lr,
            valid_step=emb_valid_step,
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
            ("valid_step",): valid_step & emb_valid_step,
        }

        return new_state, metrics, {}

    def async_emb_step(
        self,
        state: RecsysTrainingState,
        data: RecsysFeaturesBatch,
        lr: float,
        next_step_data: RecsysFeaturesBatch,
        prev_step_lookup_pin: jax.Array,
        prev_step_grad_update: AsyncEmbGradientUpdate,
    ):
        assert state.emb_table is not None
        assert state.emb_table_state is not None
        assert state.opt_state is not None
        assert self._async_emb_context is not None

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
        flat_prefetched = recsys_async_emb_lookup.lookup_done(
            self._async_emb_context, prev_step_lookup_pin
        )
        prefetched_embeddings = flat_prefetched.reshape(
            self.batch_size, -1, flat_prefetched.shape[-1]
        )

        prev_update_pins, updating_table, updating_emb_state = (
            self._emb_optim.gradient_update_start(
                self._async_emb_context,
                prev_step_grad_update,
                emb_table.x,
                state.emb_table_state,
                gate=prefetched_embeddings[:, 0, :1],
            )
        )

        prefetched_embeddings, *prev_update_pins = jax.lax.optimization_barrier(
            (prefetched_embeddings, *prev_update_pins)
        )
        embeddings = self._unflatten_emb_lookup(
            data,
            replace(
                emb_table,
                x=jax.lax.with_sharding_constraint(
                    prefetched_embeddings, P(self._async_emb_context.data_axis, None, None)
                ),
            ),
        )
        embeddings, _ = self._emb_optim.transform_embeddings(
            embeddings, self.get_flattened_token_ids(data), state.emb_table_state
        )

        def loss_fn(params, embeddings):
            return self.loss_fn.apply(
                params, rng, data, embeddings, state.rce_ema, rce_alpha, state.calib_ema
            )

        loss, loss_vjp, stats = jax.vjp(loss_fn, fprop_params, embeddings, has_aux=True)

        emb_grad_norm, emb_valid_step, grad_update_done_pin = self._emb_optim.gradient_update_done(
            self._async_emb_context, loss
        )
        emb_valid_step = emb_valid_step | jnp.logical_not(prev_step_grad_update.pending)
        updated_table, updated_emb_state, grad_update_done_pin = jax.lax.optimization_barrier(
            (updating_table, updating_emb_state, grad_update_done_pin)
        )

        next_step_token_ids, _ = jax.lax.optimization_barrier(
            (self.get_flattened_token_ids(next_step_data).astype(jnp.int32), loss)
        )
        new_emb_table, next_step_lookup_pin = recsys_async_emb_lookup.lookup_start(
            self._async_emb_context, next_step_token_ids, updated_table, grad_update_done_pin
        )
        loss_cotangent, next_step_lookup_pin = jax.lax.optimization_barrier(
            (jnp.ones_like(loss), next_step_lookup_pin)
        )

        token_ids, _ = jax.lax.optimization_barrier(
            (self.get_flattened_token_ids(data), prefetched_embeddings)
        )
        unique_tokens, segment_ids = compress_token_ids(
            token_ids,
            fill_size=self._async_emb_context.num_unique,
            fill_value=self._emb_hash_vocab,
            output_sharding=P(self._async_emb_context.data_axis),
        )

        gradients, emb_gradients = loss_vjp(loss_cotangent)
        if self.precision_level >= 2:
            gradients = jax.tree.map(lambda x: x.astype(jnp.float32), gradients)

        updates, new_opt_state = self.optim.update(gradients, state.opt_state, params=fprop_params)
        new_opt_state = typing.cast(InjectHyperparamsState, new_opt_state)
        new_params: Parameter = apply_updates(state.params, updates, lr)

        valid_step, grad_norm = self.is_valid_step(gradients)

        def _update(updated, original):
            return jnp.where(valid_step, updated, original)

        new_params = jax.tree.map(_update, new_params, state.params)
        new_opt_state = jax.tree.map(_update, new_opt_state, state.opt_state)

        metrics = {
            "step": state.step,
            "loss": loss,
            "examples_per_batch": np.prod(data["user_hashes"].shape[:-1]),
            "valid_step": valid_step,
            "global_grad_norm": grad_norm,
            "learning_rate": lr * new_opt_state.hyperparams["learning_rate"],
            "weight_decay": new_opt_state.hyperparams["weight_decay"],
            "b1": new_opt_state.hyperparams["b1"],
            "b2": new_opt_state.hyperparams["b2"],
            "emb_grad_norm": emb_grad_norm,
            "emb_valid_step": emb_valid_step,
        }
        if self.track_norm_metrics:
            metrics.update(norm_metrics(new_params, new_opt_state, gradients, updates))

        new_rce_ema = stats.pop("_rce_ema", state.rce_ema)
        new_calib_ema = stats.pop("_calib_ema", state.calib_ema)
        metrics.update(**stats)
        metrics["offset_zero_values"] = jnp.sum(state.offset_values == 0)
        metrics = jax.tree.map(lambda x: x.astype(jnp.float32), metrics)

        metric_keys, metric_values = zip(*metrics.items())
        metrics = {
            metric_keys: jnp.stack([jnp.float32(v) for v in metric_values], axis=0),
            ("valid_step",): valid_step & emb_valid_step,
        }

        emb_gradients = jax.tree.map(lambda x: x.astype(jnp.bfloat16), emb_gradients)
        next_step_grad_update = AsyncEmbGradientUpdate(
            unique_tokens=unique_tokens,
            grads=jax.lax.with_sharding_constraint(
                self._flatten_emb_grads(emb_gradients),
                P(self._async_emb_context.data_axis, None),
            ),
            segment_ids=segment_ids,
            pending=jnp.asarray(True),
        )

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

        extras = {"grad_update_keepalive": tuple(prev_update_pins)}
        return new_state, metrics, extras, next_step_lookup_pin, next_step_grad_update

    def async_emb_update(
        self,
        state: RecsysTrainingState,
        data: RecsysFeaturesBatch,
        lr: float,
    ):
        assert self._batch_pipeline.reserve is not None
        try:
            if self._async_emb_lookup_pin is None:
                state, self._async_emb_lookup_pin = self._first_step_embedding_lookup_start_jit(
                    state, data
                )
                self._deferred_emb_grad_update = self._empty_embedding_grad_update_jit()

            state, metrics, extras, self._async_emb_lookup_pin, self._deferred_emb_grad_update = (
                self._async_emb_step_jit(
                    state,
                    data,
                    lr,
                    self._batch_pipeline.reserve.batch,
                    self._async_emb_lookup_pin,
                    self._deferred_emb_grad_update,
                )
            )
        except Exception:
            assert self._async_emb_context is not None
            async_emb.async_emb_api.abort(self._async_emb_context.context_id)
            raise

        return state, metrics, extras

    def _flush_deferred_embedding_update(self, state: RecsysTrainingState) -> RecsysTrainingState:
        if self._deferred_emb_grad_update is None or not bool(
            self._deferred_emb_grad_update.pending
        ):
            return state

        assert self._async_emb_context is not None
        assert state.emb_table is not None
        try:
            async_emb.async_emb_api.drain(self._async_emb_context.context_id)
            state, _keepalive, norm, valid = self._apply_deferred_embedding_update_jit(
                state, self._deferred_emb_grad_update
            )
            jax.block_until_ready(state.emb_table.x)
            if not bool(valid):
                rank_logger.warning(
                    "async_emb: skipped a non-finite deferred update (norm=%s)", float(norm)
                )
            async_emb.async_emb_api.reset_table_binding(self._async_emb_context.context_id)
        except Exception:
            async_emb.async_emb_api.abort(self._async_emb_context.context_id)
            raise

        self._deferred_emb_grad_update = self._empty_embedding_grad_update_jit()
        return state

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
        block_sparse = build_block_sparse_layout(
            cu_seqlens=layout.cu_seqlens,
            transformer_candidate_seq_len=transformer_candidate_seq_len,
            max_history_seq_len=(_mc.num_user_prefix_tokens + _mc.history_seq_len),
            packed_seq_len=int(layout.segment_ids.shape[1]),
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

    def _create_async_emb_executables(self, init_data, lr_shape, compiler_options) -> None:
        if async_emb is None:
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
        assert self.grad_norm_keep_threshold is None
        assert isinstance(self._emb_optim, AsyncEmbOptimizer)

        tokens_per_example = jax.eval_shape(self.get_flattened_token_ids, init_data).shape[1]
        tokens_per_batch = self.batch_size * tokens_per_example
        data_shards = math.prod(self.mesh.shape[a] for a in data_axis)
        emb_width = self.state_shape.emb_table.x.shape[1]

        self._emb_hash_vocab = (
            getattr(self.dataset, "hash_vocab_size", None) or self.state_shape.emb_table.x.shape[0]
        )
        assert self._emb_hash_vocab >= self.state_shape.emb_table.x.shape[0]
        assert self._emb_hash_vocab < 2**31
        num_unique_tokens = min(tokens_per_batch, self._emb_hash_vocab) + 1

        self._async_emb_context = async_emb.make_context_handle(
            self.mesh,
            ("expert",),
            data_axis=data_axis,
            tokens_per_batch=tokens_per_batch,
            emb_width=emb_width,
            num_unique=num_unique_tokens,
            num_devices_per_node=self.parallel_config.num_devices_per_node,
        )

        replicated = NamedSharding(self.mesh, P())
        row_sharding = NamedSharding(self.mesh, P(data_axis, None))
        segment_sharding = NamedSharding(self.mesh, P(data_axis))
        grad_update_sharding = AsyncEmbGradientUpdate(
            unique_tokens=replicated,
            grads=row_sharding,
            segment_ids=segment_sharding,
            pending=replicated,
        )

        def first_step_embedding_lookup_start(state, batch):
            token_ids = self.get_flattened_token_ids(batch).astype(jnp.int32)
            emb_table, lookup_pin = recsys_async_emb_lookup.lookup_start(
                self._async_emb_context,
                token_ids,
                state.emb_table.x,
                gate=token_ids[:1, :1].astype(jnp.float32),
            )
            return (state._replace(emb_table=replace(state.emb_table, x=emb_table)), lookup_pin)

        def empty_embedding_grad_update():
            return AsyncEmbGradientUpdate(
                unique_tokens=jnp.full((num_unique_tokens,), self._emb_hash_vocab, jnp.int32),
                grads=jnp.zeros((tokens_per_batch, emb_width), jnp.bfloat16),
                segment_ids=jnp.zeros((tokens_per_batch,), jnp.int32),
                pending=jnp.asarray(False),
            )

        def apply_deferred_embedding_update(state, grad_update):
            pins, updating_table, updating_emb_state = self._emb_optim.gradient_update_start(
                self._async_emb_context,
                grad_update,
                state.emb_table.x,
                state.emb_table_state,
                gate=grad_update.grads[:, :1],
            )
            norm, valid, done_pin = self._emb_optim.gradient_update_done(
                self._async_emb_context, pins[-1]
            )
            updated_table, updated_emb_state, _ = jax.lax.optimization_barrier(
                (updating_table, updating_emb_state, done_pin)
            )
            state = state._replace(
                emb_table=replace(state.emb_table, x=updated_table),
                emb_table_state=updated_emb_state,
            )
            return state, pins, norm, valid

        grad_update_shape = jax.eval_shape(empty_embedding_grad_update)
        lookup_pin_shape = jax.ShapeDtypeStruct((data_shards, 1), jnp.float32)

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
            init_data,
            compiler_options=compiler_options,
        )

        self._empty_embedding_grad_update_jit = jax.jit(
            empty_embedding_grad_update, out_shardings=grad_update_sharding
        )

        self._apply_deferred_embedding_update_jit = JittedOrCompiled(
            jax.jit(
                apply_deferred_embedding_update,
                in_shardings=(self.state_sharding, grad_update_sharding),
                out_shardings=(self.state_sharding, None, replicated, replicated),
                donate_argnums=(0, 1),
            )
        )
        self.register_jit_function(
            self._apply_deferred_embedding_update_jit,
            self.state_shape,
            grad_update_shape,
            compiler_options=compiler_options,
        )

        self._async_emb_step_jit = JittedOrCompiled(
            jax.jit(
                self.async_emb_step,
                in_shardings=(
                    self.state_sharding,
                    self.data_sharding,
                    None,
                    self.data_sharding,
                    row_sharding,
                    grad_update_sharding,
                ),
                out_shardings=(self.state_sharding, None, None, row_sharding, grad_update_sharding),
                donate_argnums=(0, 4, 5),
            )
        )
        self.register_jit_function(
            self._async_emb_step_jit,
            self.state_shape,
            init_data,
            lr_shape,
            init_data,
            lookup_pin_shape,
            grad_update_shape,
            compiler_options=compiler_options,
        )

        self.update_jit = self.async_emb_update

    def create_executable(self) -> None:
        self._init_shmem_write_pool()
        self._free_ports()
        self._init_incremental_state()

        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)

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
            metrics_state_kwargs = {}
            if rce_ema is not None:
                metrics_state_kwargs = {
                    "rce_ema": rce_ema,
                    "rce_alpha": rce_alpha,
                    "smoothing_windows": tuple(self.smoothing_windows),
                    "calib_ema": calib_ema,
                }
            return self.model_config.make(
                sharding_context=make_legacy_sharding_context(self.mesh),
            ).loss(None, batch, recsys_embeddings, is_training=True, **metrics_state_kwargs)

        @hk.transform
        def loss_fn_eval(batch: RecsysFeaturesBatch, recsys_embeddings: RecsysEmbeddingsParameter):
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            return self.model_config.make(
                sharding_context=make_legacy_sharding_context(self.mesh),
            ).loss(None, batch, recsys_embeddings, is_training=False)

        @hk.transform
        def user_forward_fn(
            batch: RecsysFeaturesBatch, recsys_embeddings: RecsysEmbeddingsParameter
        ):
            assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
            return self.model_config.make(
                sharding_context=make_legacy_sharding_context(self.mesh),
            ).user_embeddings_only(batch, recsys_embeddings, is_training=False)

        self.loss_fn = loss_fn
        self.loss_fn_eval = loss_fn_eval
        self.user_forward_fn: typing.Any = user_forward_fn

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

            self.two_tower_forward_fn = two_tower_forward_fn
            self.candidate_tower_forward_fn = candidate_tower_forward_fn

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
                super().prepare_data(self.example_data(self.read_bsz_per_process)),
            )
        else:
            rng, init_data = self._rng_and_init_data()
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
            self.update_jit = JittedOrCompiled(
                jax.jit(
                    self.update,
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
        self._register_recsys_checksum_jit()
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
                static_argnums=[6, 7, 8],
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

    def _register_recsys_checksum_jit(self):
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        self._filtered_sharding = _deep_filter_nones(self.state_sharding)
        filtered_state_shape = _deep_filter_nones(self.state_shape)

        unwrapped_filtered_shape = unwrap_tree(filtered_state_shape)

        def compute_checksums_filtered(state):
            return checksum.compute_checksums(state, self._filtered_sharding, self.mesh)

        self._compute_checksums_filtered_jit = JittedOrCompiled(
            jax.jit(compute_checksums_filtered, in_shardings=(self._filtered_sharding,))
        )
        self.register_jit_function(self._compute_checksums_filtered_jit, unwrapped_filtered_shape)

    def checksum_dict(self):
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        filtered_state = _deep_filter_nones(self.state)

        unwrapped_state = unwrap_tree(filtered_state)
        checksums = self._compute_checksums_filtered_jit(unwrapped_state)
        return checksum.get_checksum_dict(
            unwrapped_state, self._filtered_sharding, self.mesh, checksums
        )

    def _is_copy_port_binder(self) -> bool:
        hostnames = self.ctx.hostnames
        if not hostnames:
            return self.ctx.rank == 0
        return hostnames.index(hostnames[self.ctx.rank]) == self.ctx.rank

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
        full_path: str = "",
        data_pos_raw: str = "",
    ) -> tuple[bool, int, int]:
        if full_path:
            if data_pos_raw:
                self._data_position = json.loads(data_pos_raw)
                rank_logger.info(
                    "Restored data position from storage checkpoint: %s", self._data_position
                )
            else:
                self._data_position = None
        else:
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

        if self.smoothing_windows and isinstance(self.model_config, RecsysAggregatedModelConfig):
            from xrex.data.recsys.constants import engagement_to_ids
            from xrex.models.recsys_model import build_metric_masks

            eng_names = list(engagement_to_ids(self.model_config.metric_group).keys())
            dummy = jnp.zeros((1, 1))
            dummy_3d = jnp.zeros((1, 1, self.model_config.model_config.output_vocab_size))
            mask_keys = list(
                build_metric_masks(
                    dummy,
                    dummy_3d,
                    dummy,
                    dummy,
                    enable_platform_metrics=self.model_config.enable_platform_metrics,
                ).keys()
            )
            loaded_rce = self.state.rce_ema
            reconciled_rce: dict[str, jax.Array] = {}
            for e in eng_names:
                for m in mask_keys:
                    for ws in self.smoothing_windows:
                        key = f"{e}/{m}/{ws}"
                        if loaded_rce and key in loaded_rce and loaded_rce[key].shape == (3,):
                            reconciled_rce[key] = loaded_rce[key]
                        else:
                            reconciled_rce[key] = jnp.zeros((3,), dtype=jnp.float32)

            loaded_calib = self.state.calib_ema
            reconciled_calib: dict[str, jax.Array] = {}
            for e in eng_names:
                for m in mask_keys:
                    for ws in self.smoothing_windows:
                        key = f"{e}/{m}/{ws}"
                        if loaded_calib and key in loaded_calib and loaded_calib[key].shape == (2,):
                            reconciled_calib[key] = loaded_calib[key]
                        else:
                            reconciled_calib[key] = jnp.zeros((2,), dtype=jnp.float32)

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

        return res

    def _find_own_cloud_checkpoint(self) -> str | None:
        if not self.checkpoint_storage_urls:
            return None

        xai_user = cluster.get_user() or "unknown_user"
        xai_job_name = cluster.get_job_name() or "local"

        for base_url in self.checkpoint_storage_urls.split(","):
            base_url = base_url.strip()
            if not base_url:
                continue
            cloud_base = f"{base_url.rstrip('/')}/{xai_user}/{xai_job_name}/{self.name}"
            rank_logger.info(
                "[own-checkpoint] searching cloud for own checkpoints at %s",
                cloud_base,
            )
            try:
                result = xai_recsys_engine.find_latest_storage_checkpoint(
                    cloud_base,
                    [],
                    None,
                )
            except Exception as e:
                rank_logger.warning(
                    "[own-checkpoint] error searching %s: %s",
                    cloud_base,
                    e,
                )
                continue
            if result is not None:
                checkpoint_subpath = result[0]
                full_url = f"{cloud_base}/{checkpoint_subpath}"
                full_url = full_url.removesuffix("/orbax-ckpt")
                rank_logger.info(
                    "[own-checkpoint] found own checkpoint at %s",
                    full_url,
                )
                return full_url
            rank_logger.info(
                "[own-checkpoint] no checkpoint found at %s",
                cloud_base,
            )
        return None

    def _has_own_local_checkpoint(self) -> bool:
        from xrex.utils.metadata import checkpoint_dir_or_default

        ckpt_dir = checkpoint_dir_or_default(self.checkpoint_config.checkpoint_dir)
        own_dir = os.path.join(ckpt_dir, self.name)
        if not os.path.isdir(own_dir):
            return False
        return any(d.startswith("elapsed_samples_") for d in os.listdir(own_dir))

    def maybe_load_checkpoint(self, ctx: TrainerContext, tag=None):
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)

        has_own_local = self._has_own_local_checkpoint()

        if not has_own_local and self.checkpoint_storage_urls:
            if jax.process_index() == 0:
                own_cloud_url = self._find_own_cloud_checkpoint()
            else:
                own_cloud_url = None
            buf = np.zeros(10240, dtype=np.uint8)
            if own_cloud_url:
                encoded = own_cloud_url.encode("utf-8")
                buf[: len(encoded)] = list(encoded)
            buf = multihost_utils.broadcast_one_to_all(buf)
            own_cloud_url = bytes(buf).rstrip(b"\x00").decode("utf-8") or None

            if own_cloud_url:
                if self.store_load:
                    rank_logger.info(
                        "[own-checkpoint] found own checkpoint on cloud; "
                        "overriding store_load=%s with %s",
                        self.store_load,
                        own_cloud_url,
                    )
                self.store_load = own_cloud_url
        elif has_own_local and self.store_load:
            rank_logger.info(
                "[own-checkpoint] own local checkpoint found; ignoring store_load=%s",
                self.store_load,
            )
            self.store_load = None

        if self.store_load and not has_own_local:
            from xrex.utils.metadata import CheckpointMeta, LoadType

            store_url = self.store_load.rstrip("/")

            elapsed_samples = 0
            for part in store_url.split("/"):
                if part.startswith("elapsed_samples_"):
                    elapsed_samples = int(part.split("_")[-1])
                    break

            tmp_dir = tempfile.mkdtemp(prefix="orbax_load_")
            try:
                ocdbt_url = f"{store_url}/orbax-ckpt"
                logger.info(
                    "[cloud-restore] proc %d downloading OCDBT from %s to %s",
                    jax.process_index(),
                    ocdbt_url,
                    tmp_dir,
                )
                _download_ocdbt_from_cloud(ocdbt_url, tmp_dir, jax.process_index())
                load_parent = tempfile.mkdtemp(prefix="orbax_load_parent_")
                orbax_ckpt_dir = os.path.join(load_parent, "orbax-ckpt")
                os.rename(tmp_dir, orbax_ckpt_dir)
                tmp_dir = load_parent
                load_path = load_parent

                logger.info(
                    "[cloud-restore] proc %d loading from %s (elapsed_samples=%d)",
                    jax.process_index(),
                    load_path,
                    elapsed_samples,
                )

                try:
                    n = xai_recsys_engine.download_prefix_to_local(
                        store_url, load_parent, 32, shallow=True
                    )
                    logger.info(
                        "[cloud-restore] downloaded %d sidecar files to %s",
                        n,
                        load_parent,
                    )
                except Exception as e:
                    logger.info("[cloud-restore] could not download sidecar files: %s", e)

                ctx.checkpoint = CheckpointMeta(
                    run_id="cloud-restore",
                    elapsed_samples=elapsed_samples,
                    checkpoint_index=None,
                    path=load_path,
                    load_type=LoadType.MANUAL_LOAD,
                    source="cloud",
                    format="orbax",
                    elapsed_tokens=0,
                )
                self.checkpoint_config.from_checkpoint = True
                res = super().maybe_load_checkpoint(ctx, tag)
            finally:
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            return self._finalize_checkpoint_load(ctx, res)

        full_path = ""
        data_pos_raw = ""

        checkpoint_path = self.get_checkpoint_path(self.ctx)
        store = Store.from_path(checkpoint_path)

        use_cloud_load = store != Store.FS or bool(self.store_load)
        if self.checkpoint_config.copy_port and use_cloud_load:
            start_all = time.time()

            if not hasattr(self, "host_state") or self.host_state is None:
                self.host_state = jax.device_put(self.state, self.host_sharding)
            else:
                self.state, self.host_state = self.offload_state(self.state, self.host_state)

            host_state = tree_to_dict(unwrap_tree(self.host_state))
            full_path, checksums = (
                np.zeros(10240, dtype=np.uint8),
                np.zeros(102400, dtype=np.uint8),
            )
            max_shards = np.zeros(1, dtype=np.int64)
            data_pos_buf = np.zeros(1024, dtype=np.uint8)
            if jax.process_index() == 0:
                result = None
                if store != Store.FS:
                    base_path = f"{self.checkpoint_config.checkpoint_dir}/{self.name}"
                    result = xai_recsys_engine.find_latest_storage_checkpoint(
                        base_path,
                        [
                            (k, host_state[k].nbytes)
                            for k in host_state
                            if host_state[k] is not None
                        ],
                        None,
                    )
                if result is None and self.store_load:
                    path_stripped = self.store_load.rstrip("/")
                    parts = path_stripped.split("/")
                    fixed_prefix = f"{parts[-2]}/{parts[-1]}"
                    base_path = "/".join(parts[:-2])
                    result = xai_recsys_engine.find_latest_storage_checkpoint(
                        base_path,
                        [
                            (k, host_state[k].nbytes)
                            for k in host_state
                            if host_state[k] is not None
                        ],
                        fixed_prefix,
                    )
                if result is not None:
                    p, max_shards_val = result
                    max_shards[0] = max_shards_val
                    xai_recsys_engine.load_arrays_from_storage(
                        f"{base_path}/{p}",
                        [("checksums.0.json", checksums)],
                        32,
                    )
                    try:
                        xai_recsys_engine.load_arrays_from_storage(
                            f"{base_path}/{p}",
                            [(_DATA_POSITION_FILENAME, data_pos_buf)],
                            1,
                        )
                        rank_logger.info(
                            "Downloaded %s from %s/%s (%d non-zero bytes)",
                            _DATA_POSITION_FILENAME,
                            base_path,
                            p,
                            np.count_nonzero(data_pos_buf),
                        )
                    except Exception as e:
                        rank_logger.warning(
                            "Failed to download %s from %s/%s: %s",
                            _DATA_POSITION_FILENAME,
                            base_path,
                            p,
                            e,
                        )
                    data = f"{base_path}/{p}".encode()
                    full_path[: len(data)] = list(data)
            res = multihost_utils.broadcast_one_to_all(
                (full_path, checksums, max_shards, data_pos_buf)
            )
            full_path = bytes(res[0]).rstrip(b"\x00").decode("utf-8")
            checksums = bytes(res[1]).rstrip(b"\x00").decode("utf-8")
            num_sources = int(res[2][0])
            data_pos_raw = bytes(res[3]).rstrip(b"\x00").decode("utf-8")

            if self.store_load and not full_path:
                raise ValueError(
                    f"store_load='{self.store_load}' was specified but no valid checkpoint was found. "
                    "Verify that the path exists and contains a valid checkpoint."
                )

        if not full_path:
            res = super().maybe_load_checkpoint(ctx, tag)
        else:
            start_load = time.time()
            checksums = json.loads(checksums)["global_checksums"]

            def get_sharded(key, value):
                if self.parallel_config.ep == 1:
                    return key in (
                        "emb_table",
                        "emb_table_state.inner_state.1.0.mu.table",
                        "emb_table_state.inner_state.1.0.nu.table",
                    )
                shards = value.addressable_shards
                if len(shards) != 1:
                    raise NotImplementedError(
                        f"unsupported sharding in tensor {key}: {len(shards)} != 1 addressable shards"
                    )
                shard = shards[0]
                sharded = [(x.start, x.stop, x.step) != (None,) * 3 for x in shard.index]
                return any(sharded)

            get_axis = lambda key: 0 if key == "emb_table" else 1
            name_tensor_pairs = []
            tensors = {}
            for key, value in host_state.items():
                if value is None:
                    continue
                any_sharded = get_sharded(key, value)
                n = num_sources if any_sharded else 1
                shape = list(value.shape)
                m = max(1, n // len(jax.devices()))
                if n > 1:
                    shape[get_axis(key)] //= n
                arrs = [np.zeros(shape, value.dtype) for _ in range(m)]
                tensors[key] = arrs
                k = max(1, len(jax.devices()) // n)
                if jax.process_index() % k == 0:
                    i = jax.process_index() // k
                    for j, arr in enumerate(arrs):
                        arr = np.ndarray(shape=arr.nbytes, dtype=np.uint8, buffer=arr)
                        suffix = [0] * value.ndim
                        if any_sharded:
                            suffix[get_axis(key)] = i * m + j
                        suffix = "".join(f"/{x}" for x in suffix)
                        name_tensor_pairs.append((f"{key}/c{suffix}", arr))

            xai_recsys_engine.load_arrays_from_storage(full_path, name_tensor_pairs, 256)
            multihost_utils.sync_global_devices("recsys-load")

            start_reshard = time.time()
            mesh_params = []
            for n in 1, num_sources, len(jax.devices()):
                mesh_shards = min(n, len(jax.devices()))
                mesh_replicas = max(1, len(jax.devices()) // n)
                devices = np.array(jax.devices()).reshape(mesh_shards, mesh_replicas)
                mesh_params.append((Mesh(devices, ("shard", "init_replica")), mesh_replicas))

            state_sharding = tree_to_dict(self.state_sharding)

            def reshard(path, value):
                key = path_to_dotted(path)
                any_sharded = get_sharded(key, value)
                pspec = [None] * value.ndim
                if any_sharded:
                    pspec[get_axis(key)] = "shard"
                mesh, mesh_replicas = mesh_params[1 if any_sharded else 0]
                source_sharding = NamedSharding(mesh, P(*pspec))
                arrs = jax.device_put(tensors[key], jax.local_devices()[0])
                arr = jnp.concatenate(arrs, get_axis(key)) if len(arrs) > 1 else arrs[0]
                arr = jax.make_array_from_single_device_arrays(value.shape, source_sharding, [arr])
                axis = get_axis(key) if any_sharded else 0
                if mesh_replicas > 1:
                    pspec = [None] * value.ndim
                    if value.ndim > 0:
                        pspec[axis] = "shard"
                    pspec_for_shard_map = P(*pspec)

                    @functools.partial(jax.jit, donate_argnums=(0,))
                    @functools.partial(
                        shard_map,
                        mesh=mesh,
                        in_specs=(pspec_for_shard_map,),
                        out_specs=pspec_for_shard_map,
                        check_vma=False,
                    )
                    def broadcast_from_leader(x):
                        return pbroadcast(x, ("init_replica",), 0)

                    arr = broadcast_from_leader(arr)
                if any_sharded and self.parallel_config.ep > 1 and axis != 1:
                    pspec = [None] * value.ndim
                    pspec[1] = "shard"
                    arr = jax.device_put(arr, NamedSharding(mesh_params[2][0], P(*pspec)))
                return jax.device_put(arr, state_sharding[key])

            self.state = jax.tree_util.tree_map_with_path(reshard, self.state)

            start_checksum = time.time()
            checksum_message = ""
            if self.checkpoint_config.verify_checksums:
                checksum_dict = self.checksum_dict()["global_checksums"]
                unknown_checksums = []
                count = 0
                for key in host_state:
                    if key not in checksums:
                        unknown_checksums.append(key)
                    elif checksums[key] != checksum_dict[key]:
                        raise ValueError(f"Checksum mismatch on {key}")
                    else:
                        count += 1
                if unknown_checksums:
                    rank_logger.warning(
                        "The following keys were not found in the checkpoint checksums file and could not be verified: %s",
                        unknown_checksums,
                    )
                checksum_message = f"({count}/{len(host_state)} checksums match) "

            t = time.time()
            durations = [
                f"{x:.1f} s"
                for x in (
                    start_load - start_all,
                    start_reshard - start_load,
                    start_checksum - start_reshard,
                    t - start_checksum,
                    t - start_all,
                )
            ]
            rank_logger.info(
                "Loaded checkpoint %s%s; discovery %s, load %s, reshard %s, checksum %s, total %s",
                checksum_message,
                full_path,
                *durations,
            )

            elapsed_samples = int(full_path.split("/")[-2].split("_")[-1])
            res = (True, elapsed_samples, 0)

        return self._finalize_checkpoint_load(ctx, res, full_path, data_pos_raw)

    def next_data_batch(self):
        if self._shmem_write_future is not None and self._shmem_write_future.done():
            self._pending_shmem_ckpt_write_s = self._shmem_write_future.result()
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
            self._batch_pipeline.reserve = fetch()

        self._batch_pipeline.current = self._batch_pipeline.reserve

        try:
            self._batch_pipeline.reserve = fetch()
        except StopIteration:
            self._batch_pipeline.exhausted = True

        self.offsets_to_commit = self._batch_pipeline.current.offsets

        assert self._async_emb_context is not None
        ready_step = async_emb.async_emb_api.wait_step_ready(self._async_emb_context.context_id)
        assert ready_step != 0 or self._async_emb_lookup_pin is None
        return self._batch_pipeline.current.batch

    def _checkpoint_data_position(self) -> DataPosition | None:
        assert isinstance(self.dataset, PhoenixDataset)
        if not self.use_async_emb or self._batch_pipeline.current is None:
            return self.dataset.get_data_position()
        return self._batch_pipeline.current.data_position

    def _write_shmem_checkpoint(
        self,
        write_items: list[tuple[str, typing.Any, npt.NDArray]],
        checksums: dict,
        data_pos: typing.Any,
        prefix: str,
        stub: typing.Any,
    ) -> float:
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

        assert _jax_dist.client is not None, "JAX distributed not initialized"
        _jax_dist.client.wait_at_barrier(f"recsys-save-checkpoint2-{prefix}", timeout_in_ms=300_000)

        if has_engine:
            with open(f"{OUT_PATH}/.{prefix}/checksums.0.json", "w") as f:
                json.dump(
                    {
                        "global_checksums": checksums,
                        "created_timestamp": time.time(),
                    },
                    f,
                )
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

        return time.perf_counter() - start

    def save_checkpoint(self, *args, **kwargs):
        dataset = self.dataset
        assert isinstance(dataset, PhoenixDataset), f"Got {type(dataset)}"
        assert isinstance(self.state, RecsysTrainingState), f"Got {type(self.state)}"

        self.state = self._flush_deferred_embedding_update(self.state)

        if self.offsets_to_commit:
            items_list = list(self.offsets_to_commit.items())

            n_partitions = dataset.num_kafka_partitions
            assert n_partitions is not None, "num_kafka_partitions must be set for Kafka offsets"
            max_size = (n_partitions + jax.process_count() - 1) // jax.process_count()

            items_padded = items_list + [(-1, 0)] * (max_size - len(items_list))

            items_array = np.array(items_padded, dtype=np.int64)

            all_offsets = multihost_utils.process_allgather(items_array, tiled=True)

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
        self.maybe_build_gen_recs_post_embeddings()

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
                self._engine = xai_recsys_engine.RecsysPredictorServer(
                    port,
                    port + 1,
                    1,
                    0,
                    copy_max_entries=self.checkpoint_config.shm_max_entries,
                )
            multihost_utils.sync_global_devices("recsys-copy-port-bind")

            checkpoint_path = self.get_checkpoint_path(self.ctx)
            prefix = "/".join(checkpoint_path.split("/")[-2:])
            store = Store.from_path(checkpoint_path)
            wait_until_finished()
            if self._shmem_write_future is not None:
                self._pending_shmem_ckpt_write_s = self._shmem_write_future.result()
                self._shmem_write_future = None
            multihost_utils.sync_global_devices("recsys-save-checkpoint1")
            stub = copy_pb2_grpc.CopyStub(grpc.insecure_channel(f"127.0.0.1:{port}"))

            if not hasattr(self, "host_state") or self.host_state is None:
                self.host_state = jax.device_put(self.state, self.host_sharding)
            else:
                self.state, self.host_state = self.offload_state(self.state, self.host_state)

            assert isinstance(self.state, RecsysTrainingState)
            host_state = unwrap_tree(self.host_state)
            if store == Store.FS:
                host_state = host_state.purge_opt_state()
            host_state = tree_to_dict(host_state)

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

            start_checksum = time.perf_counter()
            checksums = {}

            for key in host_state:
                value = host_state[key]
                if value is None:
                    continue
                shards = value.addressable_shards
                if len(shards) != 1:
                    continue
                shard = shards[0]
                sharded = [(x.start, x.stop, x.step) != (None,) * 3 for x in shard.index]
                shard_data = _unsafe_jax2np(shard.data)
                shard_bytes = np.ndarray(shape=shard_data.nbytes, dtype=np.uint8, buffer=shard_data)

                if not any(sharded):
                    checksums[key] = xai_recsys_engine.adler32_parallel(
                        np.ascontiguousarray(shard_bytes)
                    )
                elif key in ("emb_table", "post_embeddings.embeddings"):
                    local_checksum = xai_recsys_engine.adler32_parallel(
                        np.ascontiguousarray(shard_bytes)
                    )
                    local_len = shard_bytes.nbytes

                    all_checksums = multihost_utils.process_allgather(
                        np.array([local_checksum], dtype=np.uint32), tiled=True
                    )
                    local_len_parts = np.array([local_len], dtype=np.uint64).view(np.uint32)
                    all_lens_split = multihost_utils.process_allgather(
                        local_len_parts.reshape(1, 2), tiled=True
                    )

                    i = sharded.index(True)
                    delta = shard.index[i].stop - shard.index[i].start
                    num_shards = value.shape[i] // delta
                    n = jax.process_count()
                    processes_per_shard = n // num_shards

                    shard_info = []
                    for proc_idx in range(n):
                        proc_shard_idx = proc_idx // processes_per_shard
                        if proc_idx % processes_per_shard == 0:
                            ck = int(all_checksums[proc_idx])
                            len_lo = int(all_lens_split[proc_idx, 0])
                            len_hi = int(all_lens_split[proc_idx, 1])
                            length = len_lo | (len_hi << 32)
                            shard_info.append((proc_shard_idx, ck, length))
                    shard_info.sort(key=lambda x: x[0])

                    combined = shard_info[0][1]
                    for _, ck, length in shard_info[1:]:
                        combined = xai_recsys_engine.adler32_combine_py(combined, ck, length)
                    checksums[key] = combined
                else:
                    i = sharded.index(True)
                    delta = shard.index[i].stop - shard.index[i].start
                    shard_idx = shard.index[i].start // delta
                    num_shards = value.shape[i] // delta
                    total_cols = value.shape[1]
                    shard_width = delta
                    num_rows = value.shape[0]

                    local_partial = xai_recsys_engine.adler32_shard_partial(
                        np.ascontiguousarray(shard_bytes),
                        shard_width * shard_data.dtype.itemsize,
                        shard_idx,
                        total_cols * shard_data.dtype.itemsize,
                        128,
                    )

                    all_partials_flat = multihost_utils.process_allgather(
                        local_partial.astype(np.uint32), tiled=True
                    )
                    all_partials_2d = all_partials_flat.reshape(-1, 2)

                    n = jax.process_count()
                    processes_per_shard = n // num_shards
                    unique_partials = []
                    for proc_idx in range(n):
                        if proc_idx % processes_per_shard == 0:
                            proc_shard_idx = proc_idx // processes_per_shard
                            unique_partials.append((proc_shard_idx, all_partials_2d[proc_idx]))

                    unique_partials.sort(key=lambda x: x[0])
                    partials_array = np.stack([p[1] for p in unique_partials], axis=0).astype(
                        np.uint32
                    )

                    total_len = num_rows * total_cols * shard_data.dtype.itemsize
                    checksums[key] = xai_recsys_engine.adler32_combine_partials(
                        partials_array, total_len
                    )

            self._pending_checksum_s = time.perf_counter() - start_checksum

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

            self._shmem_write_future = self._shmem_write_pool.submit(
                self._write_shmem_checkpoint,
                write_items,
                checksums,
                data_pos,
                prefix,
                stub,
            )

            if store != Store.FS:
                if should_persist_disk:
                    self._last_disk_checkpoint_ts = time.time()
                return None

            if should_persist_disk:
                if self._shmem_write_future is not None:
                    self._pending_shmem_ckpt_write_s = self._shmem_write_future.result()
                    self._shmem_write_future = None
                host_state.clear()
                del write_items
                self.host_state = None

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

        if self.checkpoint_storage_urls:
            xai_user = cluster.get_user() or "unknown_user"
            xai_job_name = cluster.get_job_name() or "local"
            tag = "orbax-ckpt"
            proc_idx = jax.process_index()

            from xrex.utils.checkpointing import get_checkpointer

            _orbax_ckpt = get_checkpointer()
            local_ckpt_dir = self.get_checkpoint_path(self.ctx)

            upload_targets = []
            for base_url in self.checkpoint_storage_urls.split(","):
                base_url = base_url.strip()
                if not base_url:
                    continue
                cloud_path = (
                    f"{base_url.rstrip('/')}/{xai_user}/{xai_job_name}"
                    f"/{self.name}/elapsed_samples_{self.elapsed_samples:018d}"
                )
                upload_targets.append(cloud_path)

            def _bg_upload():
                try:
                    logger.info(
                        "[cloud-ckpt] proc %d waiting for /data/ save",
                        proc_idx,
                    )
                    _orbax_ckpt.wait_until_finished()
                    logger.info(
                        "[cloud-ckpt] proc %d /data/ save done, starting upload",
                        proc_idx,
                    )

                    ensure_orbax_dir_finalized(local_ckpt_dir, tag)
                    file_list = collect_cloud_upload_files(local_ckpt_dir, tag, proc_idx)
                    my_files = [rel for rel, _ in file_list]
                    my_bytes = sum(size for _, size in file_list)

                    for cloud_dest in upload_targets:
                        upload_start = time.time()
                        logger.info(
                            "[cloud-ckpt] proc %d uploading %d files (%.1f MB) from %s to %s",
                            proc_idx,
                            len(my_files),
                            my_bytes / (1024 * 1024),
                            local_ckpt_dir,
                            cloud_dest,
                        )
                        if my_files:
                            xai_recsys_engine.upload_files_to_storage(
                                local_ckpt_dir,
                                my_files,
                                cloud_dest,
                            )
                        logger.info(
                            "[cloud-ckpt] proc %d upload done in %.1fs to %s",
                            proc_idx,
                            time.time() - upload_start,
                            cloud_dest,
                        )
                except Exception:
                    logger.exception(
                        "[cloud-ckpt] proc %d upload failed",
                        proc_idx,
                    )

            import threading

            threading.Thread(
                target=_bg_upload,
                daemon=True,
                name="cloud-ckpt-upload",
            ).start()

        gc_start = time.perf_counter()
        gc.collect()
        self._pending_gc_collect_s = time.perf_counter() - gc_start

        return result

    def run(self) -> None:
        gc.disable()
        rank_logger.info("Disabled Python cyclic GC for the recsys training loop")
        try:
            super().run()
        finally:
            exception_type = sys.exc_info()[0]
            if exception_type is None or issubclass(exception_type, StopIteration):
                self.state = self._flush_deferred_embedding_update(self.state)
            elif self._async_emb_context is not None:
                async_emb.async_emb_api.abort(self._async_emb_context.context_id)

            shutdown = getattr(self.dataset, "shutdown", None)
            if callable(shutdown):
                rank_logger.info("Stopping dataset background threads via shutdown()")
                try:
                    shutdown()
                except Exception:
                    rank_logger.exception("dataset.shutdown() failed during teardown")

    def two_int32_to_int64(self, two_int32: jax.Array | np.ndarray) -> npt.NDArray[np.int64]:
        low_64 = np.asarray(two_int32[:, 0], dtype=np.int64) & 0xFFFFFFFF
        high_64 = np.asarray(two_int32[:, 1], dtype=np.int64) << 32
        return low_64 | high_64

    def int64_to_two_int32(self, post_ids: npt.NDArray[np.int64]) -> jax.Array:
        low_32 = jnp.asarray(post_ids & 0xFFFFFFFF, dtype=jnp.int32)
        high_32 = jnp.asarray(post_ids >> 32, dtype=jnp.int32)
        return jnp.stack([low_32, high_32], axis=1)

    def eval(self, soft_step: int):
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return self.eval_two_tower(soft_step)

        raise ValueError("Ranking model eval_every_n is not supported yet.")

    def maybe_build_retrieval_post_embeddings(self):
        if not isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return

        assert isinstance(self.state, RecsysTrainingState)
        assert self.state.emb_table is not None

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
            rank_logger.info(
                f"Loading configured retrieval datasets: {[ds.name for ds in target_datasets]}"
            )

        _user_cfg = self.model_config.user_tower_config
        _use_post_sid = _user_cfg.use_post_sid
        _sid_num_levels = _user_cfg.sid_num_levels
        max_posts = self.model_config.candidate_tower_config.max_posts
        if jax.process_index() == 0:
            post_ids, author_ids, dataset_types, post_sids_raw = RetrievalDataset.load_datasets(
                target_datasets,
                max_posts=max_posts,
                read_post_sid=_use_post_sid,
                sid_num_levels=_sid_num_levels,
            )
        else:
            post_ids = np.zeros(max_posts, dtype=np.int64)
            author_ids = np.zeros(max_posts, dtype=np.int64)
            dataset_types = np.zeros(max_posts, dtype=np.int32)
            post_sids_raw = None
        if post_sids_raw is None:
            post_sids_raw = np.full((max_posts, _sid_num_levels), -1, dtype=np.int32)
        post_ids_i32 = post_ids.view(np.int32)
        author_ids_i32 = author_ids.view(np.int32)
        post_ids_i32, author_ids_i32, dataset_types, post_sids_raw = (
            multihost_utils.broadcast_one_to_all(
                (post_ids_i32, author_ids_i32, dataset_types, post_sids_raw)
            )
        )
        post_ids = np.asarray(post_ids_i32, dtype=np.int32).view(np.int64)
        author_ids = np.asarray(author_ids_i32, dtype=np.int32).view(np.int64)
        dataset_types = np.asarray(dataset_types, dtype=np.int32)
        post_sids_raw = np.asarray(post_sids_raw, dtype=np.int32)
        total_samples = len(post_ids)
        shard_size = total_samples // self.data_world_size
        start_idx = self.data_rank * shard_size
        end_idx = (
            (self.data_rank + 1) * shard_size
            if self.data_rank < self.data_world_size - 1
            else total_samples
        )
        post_ids_shard = post_ids[start_idx:end_idx]
        author_ids_shard = author_ids[start_idx:end_idx]
        dataset_types_shard = dataset_types[start_idx:end_idx]

        _hash_table = self.model_config.candidate_tower_config.hash_table
        post_hashes_shard = _hash_table.get_item_hash(post_ids_shard)
        author_hashes_shard = _hash_table.get_author_hash(author_ids_shard)
        if _user_cfg.use_post_embedding:
            combined_hashes_shard = jnp.concatenate(
                [post_hashes_shard, author_hashes_shard], axis=1
            )
        else:
            combined_hashes_shard = author_hashes_shard
        combined_hashes_jax = jax.make_array_from_process_local_data(
            self.data_sharding, np.asarray(combined_hashes_shard)
        )
        post_author_embeddings = self._lookup(self.state.emb_table, combined_hashes_jax)

        post_sids_raw_shard = post_sids_raw[start_idx:end_idx]
        if _use_post_sid:
            post_sids_u16_shard = (post_sids_raw_shard + 1).astype(np.uint16)
        else:
            post_sids_u16_shard = np.zeros_like(post_sids_raw_shard, dtype=np.uint16)
        post_sids_jax = jax.make_array_from_process_local_data(
            self.data_sharding, post_sids_u16_shard
        )
        post_hashes_jax = jax.make_array_from_process_local_data(
            self.data_sharding, np.asarray(post_hashes_shard)
        )

        rng, _new_rng = jax.random.split(self.state.rng)

        head_dataset_mapping = getattr(self.model_config, "head_dataset_mapping", None)
        num_heads = self.model_config.candidate_tower_config.num_candidate_heads
        candidate_embeddings = self.candidate_tower_forward_jit(
            self.state.params,
            rng,
            post_author_embeddings.x,
            post_sids_jax,
            post_hashes_jax,
            0,
        )
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
                emb_h = self.candidate_tower_forward_jit(
                    self.state.params,
                    rng,
                    post_author_embeddings.x,
                    post_sids_jax,
                    post_hashes_jax,
                    h,
                )
                mask_h = post_head_jax == h
                candidate_embeddings = jnp.where(mask_h, emb_h, candidate_embeddings)
                del emb_h
        else:
            candidate_embeddings = self.candidate_tower_forward_jit(
                self.state.params,
                rng,
                post_author_embeddings.x,
                post_sids_jax,
                post_hashes_jax,
                0,
            )

        global_post_ids = multihost_utils.host_local_array_to_global_array(
            self.int64_to_two_int32(post_ids), self.mesh, P(None)
        )
        global_author_ids = multihost_utils.host_local_array_to_global_array(
            self.int64_to_two_int32(author_ids), self.mesh, P(None)
        )
        combined_dataset_types = multihost_utils.process_allgather(
            dataset_types_shard, tiled=True
        ).reshape(-1, 1)

        original_embeddings = self.state.post_embeddings.embeddings
        post_embeddings = PostEmbeddings(
            post_ids=global_post_ids,
            author_ids=global_author_ids,
            embeddings=replace(
                original_embeddings, x=candidate_embeddings, pspec=self.data_sharding.spec
            ),
            dataset_types=combined_dataset_types,
        )
        self.state = self.state._replace(post_embeddings=post_embeddings)

    def maybe_build_gen_recs_post_embeddings(self) -> None:
        pass

    def eval_two_tower(self, soft_step: int):
        assert isinstance(self.state, RecsysTrainingState)
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        rank_logger.info("Running evals at %s", soft_step)
        rng, _new_rng = jax.random.split(self.state.rng)

        def _two_tower_eval_forward_fn(
            batch, post_embeddings, dataset_types, top_k, target_dataset_types, eval_bs_per_device=0
        ):
            recsys_embeddings_parameter = self.get_recsys_embeddings(
                batch,
                self.state.emb_table,
            )
            recsys_embeddings = get_recsys_embed_param_to_jax_array(recsys_embeddings_parameter)
            recsys_embeddings = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), recsys_embeddings
            )
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
            recsys_embeddings_param: RecsysEmbeddingsParameter = self.get_recsys_embeddings(
                batch,
                self.state.emb_table,
            )
            recsys_embeddings_param = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), recsys_embeddings_param
            )
            return self.loss_fn_eval_jit(self.state.params, rng, batch, recsys_embeddings_param)

        def _user_emb_fn(batch: RecsysFeaturesBatch):
            recsys_embeddings_param = self.get_recsys_embeddings(batch, self.state.emb_table)
            recsys_embeddings_param = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), recsys_embeddings_param
            )
            return self.user_forward_jit(self.state.params, rng, batch, recsys_embeddings_param)

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
                eval_module.eval_bs_per_device,
                eval_module.eval_seq_len,
            )
            if not isinstance(eval_conf, RecsysTwoTowerEval):
                continue

            eval_head_idx = 0
            if head_dataset_mapping is not None:
                eval_head_idx = head_dataset_mapping.get(eval_conf.target_dataset_type.name, 0)
            eval_cand_fn = _make_candidate_tower_eval_fn(eval_head_idx)

            eval_results = run_recsys_evals(
                evals=[(eval_name, eval_conf)],
                train_dataset=self.dataset_thread_iterator(self._eval_dataset, prepare_data=False),
                forward_fn=_two_tower_eval_forward_fn,
                loss_fn=_loss_fn,
                mesh=self.mesh,
                rng=rng,
                candidate_tower_forward_fn=eval_cand_fn,
                all_post_embeddings=self.state.post_embeddings.embeddings.x,
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
