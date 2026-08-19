# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import concurrent.futures
import datetime
import functools
import itertools
import logging
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from opentelemetry import trace
from xai_configlib import Config, configclass

from xrex.configs.config import Dataset
from xrex.eval.eval_utils import (
    EvalModule,
    report_forward_eval_results,
    run_forward_evals_new,
)
from xrex.utils.gpu import NUM_PHYSICAL_DEVICES_PER_NODE, pin_process_to_local_numa
from xrex.utils.jax_extensions import initialize_extensions
from xrex.utils.log_timer import log_elapsed_time

initialize_extensions()


from xai_checkpointing import checksum
from xai_checkpointing import common as checkpointing_common
from xai_checkpointing import load as checkpointing_load
from xai_checkpointing.tree_util import tree_to_dict

from xrex.models.model_utils import Parameter, unwrap_tree
from xrex.models.recsys_model import RecsysAggregatedModelConfig
from xrex.models.recsys_two_tower_model import RecsysTwoTowerModelConfig
from xrex.models.sharding_context import (
    make_legacy_sharding_context,
    make_sharding_context_from_config,
)
from xrex.optimizers.optim import OptimConfig
from xrex.optimizers.schedule import BaseSampleSchedule, ConstantSampleSchedule
from xrex.train.misc import (
    CheckpointConfig,
    TrainingState,
    TrainingStateProtocol,
)
from xrex.train.parallel_config import ParallelConfig
from xrex.utils import checkpointing, metadata
from xrex.utils import metrics as metrics_utils
from xrex.utils.aot import JittedOrCompiled, TracedWithOptions, compile_or_load_all_traced
from xrex.utils.metadata import CheckpointMeta
from xrex.utils.profiler import start_profiler_server, start_trace, stop_trace
from xrex.utils.tracer import TracerConfig, setup_tracer
from xrex.utils.utils import (
    cast_bfloat16,
    compute_norm,
    compute_num_elements,
    compute_num_stored_elements,
    get_bytes_in_use,
    get_peak_bytes_in_use,
    selective_tree_cast,
)

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")
tracer = trace.get_tracer(__name__)

DEFAULT_COORDINATOR_PORT = 1234


def default_name():
    return f"xrex-{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H%M%S')}"


class TrainerEvent(Enum):
    BOOT = "boot"
    TRAIN_START = "train_start"
    STEP = "step"
    COMPLETED = "completed"


class DriverMessage(Enum):
    CONTINUE = "continue"
    SAVE_CHECKPOINT_ASAP = "save_checkpoint_asap"
    TERMINATE = "terminate"


@dataclass
class DatasetMetadata:
    data_rank: int = 0
    data_world_size: int = 0
    elapsed_samples: int = 0


@dataclass
class TrainerContext:
    rank: int
    world_size: int

    hostnames: list[str]
    ip_addrs: list[str]

    config: Any
    run: metadata.Run
    run_dir: Path
    checkpoint: metadata.CheckpointMeta | None

    meta: metadata.MetadataManager

    qp: Any = None

    dataset_metadata_fn: Callable[[int, int], DatasetMetadata] | None = None

    coordinator_port: int = DEFAULT_COORDINATOR_PORT

    driver_ack_timeout: float = 60.0

    hooks: Any = None

    def _handle(self, event: TrainerEvent, data: Any) -> None:
        if self.qp:
            self.qp.send((event, data))

    def on_boot(self) -> None:
        self._handle(TrainerEvent.BOOT, None)

    def on_train_start(self) -> None:
        self._handle(TrainerEvent.TRAIN_START, None)

    def on_step(
        self,
        metrics: dict[str, Any],
        checkpoint_metrics: dict[str, list[int]],
        is_healthy: bool = True,
    ) -> DriverMessage:
        if self.hooks is not None and metrics:
            self.hooks.on_step(metrics)
        self._handle(TrainerEvent.STEP, (metrics, checkpoint_metrics, is_healthy))
        return self._wait_for_driver_ack()

    def on_completed(self, cause: str) -> None:
        self._handle(TrainerEvent.COMPLETED, cause)

    def _wait_for_driver_ack(self) -> DriverMessage:
        if self.qp is None:
            return DriverMessage.CONTINUE
        return self.qp.recv(timeout=self.driver_ack_timeout)


@configclass
class Trainer(Config):
    model_config: RecsysAggregatedModelConfig | RecsysTwoTowerModelConfig
    optim_config: OptimConfig
    parallel_config: ParallelConfig
    dataset: Dataset
    lr_schedule_in_samples_config: BaseSampleSchedule | None = None
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)

    bs_per_device: float = 1
    max_steps: int | None = 6
    max_samples: int | None = None
    dataloader_offset: int = 0

    name: str = field(default_factory=default_name)
    load: str | None = None
    reuse_run_id: bool = False
    reinit_on_load: bool = False
    step_count_start: int | None = None

    precision_level: int = 2
    rng_seed: int = 42
    startup_profile: bool = False

    jax_profile: bool = False
    jax_profile_rank: int = 0
    jax_profile_start_step: int = 2
    jax_profile_stop_step: int = 6

    nsys_start_rank: int | None = None
    nsys_start_step: int = 2
    nsys_stop_step: int = 6
    nsys_start_options: str | None = None
    nsys_session: str | None = None

    enable_determinism: bool = False

    setup_imex_domain_size: int | None = None

    enable_aot: bool = False
    aot_dump: bool = False
    aot_dump_loc: bool = False
    dump_all_hlos: bool = False
    aot_cache_dir: str = "/tmp/jax_aot_cache/"
    aot_metrics: dict[str, dict[str, float]] | None = None

    xla_flag_overrides: dict[str, str] = field(default_factory=dict)
    extra_jax_config: dict[str, str] = field(default_factory=dict)

    fdo_profile_dir: str | None = None

    grad_norm_keep_threshold: float | None = None
    track_norm_metrics: bool = True

    eval_every_n: int = -1
    evals: list[EvalModule] = field(default_factory=list)
    init_opt_state: bool = True

    use_mock_gpu_client: bool = False

    debug_tensor_dump_output_folder: str | None = None

    jax_log_level: str = "WARN"
    tracer_config: TracerConfig = field(default_factory=TracerConfig)

    state: TrainingState = field(init=False, repr=False, compare=False)
    ctx: TrainerContext = field(init=False, repr=False, compare=False)
    mesh: jax.sharding.Mesh = field(init=False, repr=False, compare=False)
    prev_step: int = field(init=False, repr=False, compare=False, default=-1)
    elapsed_samples: int = field(init=False, repr=False, compare=False, default=0)
    elapsed_tokens: int = field(init=False, repr=False, compare=False, default=0)
    prev_ts: float = field(init=False, repr=False, compare=False, default=0.0)
    num_devices: int = field(init=False, repr=False, compare=False, default=-1)
    batch_size: int = field(init=False, repr=False, compare=False, default=-1)
    read_bsz_per_process: int = field(init=False, repr=False, compare=False, default=-1)
    data_world_size: int = field(init=False, repr=False, compare=False, default=-1)
    num_processes_per_data_rank: int = field(init=False, repr=False, compare=False, default=-1)
    data_rank: int = field(init=False, repr=False, compare=False, default=-1)
    profile_ip: str | None = field(init=False, repr=False, compare=False, default=None)
    profile_port: int | None = field(init=False, repr=False, compare=False, default=None)
    lr_schedule_in_samples: Callable[..., float] = field(
        init=False, repr=False, compare=False, default=lambda *_a, **_kw: 1.0
    )
    current_ckpt_index: int = field(init=False, repr=False, compare=False, default=0)

    data_first_read_timeout = 60
    data_read_timeout = 30

    def initialize(self, ctx: TrainerContext) -> None:
        ctx.on_boot()
        if self.setup_imex_domain_size is not None:
            raise ValueError(
                "setup_imex_domain_size is no longer honored here: bringing up an "
                "IMEX NVLink domain spans nodes, and this trainer owns a single "
                "node. Have the launcher that acquires the nodes set the domain up "
                "before starting training, and leave this unset."
            )
        self.dataloading_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if self.jax_log_level != "WARN":
            jax.config.update("jax_logging_level", self.jax_log_level)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
        jax.config.update("jax_threefry_partitionable", True)
        jax.config.update("jax_check_proxy_envs", False)
        jax.config.update("jax_traceback_filtering", "off")
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 300.0)

        jax.config.update("jax_use_direct_linearize", False)

        if "jax_disable_lowering_cache" in jax.config.values:
            jax.config.update("jax_disable_lowering_cache", True)

        if self.use_mock_gpu_client:
            jax.config.update("jax_cuda_visible_devices", "0")
            jax.config.update("mock_num_gpu_processes", ctx.world_size)

        for key, value in self.extra_jax_config.items():
            jax.config.update(key, value)

        self.model_config.model_config.debug_tensor_dump_output_folder = (
            self.debug_tensor_dump_output_folder
        )

        setup_tracer("xrex.Trainer", self.tracer_config)

        self.ctx = ctx
        self.create_runtime(ctx)

        pin_process_to_local_numa(jax.process_index() % NUM_PHYSICAL_DEVICES_PER_NODE)

        if not self.enable_aot:
            jax.config.update(
                "jax_compilation_cache_dir", "/tmp/jax-cache/" + str(jax.process_index())
            )

        trace_startup = self.startup_profile and jax.process_index() == 0
        if trace_startup:
            self.start_startup_profile()
        self.override_model_config()

        self.create_executable()
        self.compile_or_load()

        if self.use_mock_gpu_client:
            return

        if trace_startup:
            self.stop_startup_profile()
        self.prev_step = -1

        self.create_state(ctx)
        self.create_dataset(ctx)

        self.prev_ts = time.time()

    def start_startup_profile(self) -> None:
        from xrex.utils.pyspy_profiler import start_pyspy_trace

        profiling_path = self.ctx.run_dir / "speedscope" / str(jax.process_index())
        rank_logger.info(f"Writing startup profiling to {profiling_path}...")
        start_pyspy_trace(profiling_path, blocking=True, rate=5)

    def stop_startup_profile(self) -> None:
        from xrex.utils.pyspy_profiler import stop_pyspy_trace

        stop_pyspy_trace()

    def maybe_nsys_step(self, soft_step: int, block_tensor) -> None:
        from xrex.utils.nsys import nsys_start, nsys_stop_and_shutdown

        if soft_step == self.nsys_start_step:
            nsys_start(self.ctx.run_dir, self.nsys_session, self.nsys_start_options)
        if soft_step == self.nsys_stop_step:
            jax.block_until_ready(block_tensor)
            rank_logger.info("Stopping nsys profile")
            nsys_stop_and_shutdown(self.nsys_session)

    def create_runtime(self, ctx: TrainerContext):
        if not self.use_mock_gpu_client:
            if jax.__version_info__ >= (0, 8):
                jax.config.update("jax_cpu_collectives_implementation", None)
            else:
                jax.config.update("jax_cpu_enable_gloo_collectives", False)
                jax.config.update("jax_cpu_collectives_implementation", None)
            if ctx.world_size > 1:
                coordinator_address = ctx.ip_addrs[0] + f":{ctx.coordinator_port}"
                jax.distributed.initialize(
                    coordinator_address=coordinator_address,
                    num_processes=ctx.world_size,
                    process_id=ctx.rank,
                    local_device_ids=list(range(self.parallel_config.num_devices_per_process)),
                    cluster_detection_method="deactivate",
                )
                self.create_profile_server(ctx)

        self.mesh = jax.sharding.Mesh(
            mesh_utils.create_device_mesh(self.parallel_config.mesh_shape()),
            self.parallel_config.mesh_axis_names(),
        )

        self.num_devices = self.parallel_config.num_devices()
        self.batch_size = int(self.bs_per_device * self.num_devices)
        self.read_bsz_per_process, self.data_world_size, self.num_processes_per_data_rank = (
            self.parallel_config.get_local_batch_size_and_rank(self.batch_size)
        )

        rank_logger.info(
            "JAX runtime mesh is %s, num_devices = %i, batch_size = %i",
            dict(self.mesh.shape),
            self.num_devices,
            self.batch_size,
        )

        self.data_rank = ctx.rank // self.num_processes_per_data_rank
        self.chunk_index = ctx.rank % self.num_processes_per_data_rank

    def create_profile_server(self, ctx: TrainerContext):
        if not self.use_mock_gpu_client:
            self.profile_ip = ctx.ip_addrs[ctx.rank]
            self.profile_port = 5000 + ctx.rank % NUM_PHYSICAL_DEVICES_PER_NODE
            start_profiler_server(self.profile_port, jax.process_index())
            rank_logger.info(f"Starting JAX profile server: {self.profile_ip}:{self.profile_port}")

    def prepare_data(self, batch):
        batch.pop("datasets", None)
        if "mask" in batch:
            batch["mask"] = jnp.bool(batch["mask"])
        batch = multihost_utils.host_local_array_to_global_array(
            batch,
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis)),
        )
        return batch

    def create_optim(self) -> None:
        self.optim = self.optim_config.make()
        if self.lr_schedule_in_samples_config is not None:
            assert isinstance(self.lr_schedule_in_samples_config, BaseSampleSchedule)
            assert self.optim_config.learning_rate == 1.0
            self.lr_schedule_in_samples = self.lr_schedule_in_samples_config.make()
        else:
            self.lr_schedule_in_samples = ConstantSampleSchedule(learning_rate=1.0).make()

    @tracer.start_as_current_span("create_dataset")
    def create_dataset(self, ctx: TrainerContext):
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

        def dataset_with_prepare(dataset):
            for batch in dataset:
                yield self.prepare_data(batch)

        def dataset_thread_iterator(dataset, stop=object()):
            it = dataset_with_prepare(dataset)
            future = self.dataloading_thread.submit(next, it, stop)
            future.result(timeout=self.data_first_read_timeout)
            while True:
                result = future.result(timeout=self.data_read_timeout)
                if result is stop:
                    return
                future = self.dataloading_thread.submit(next, it, stop)
                yield result

        if self.chunk_index == 0:
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
                skip_rows=max(0, elapsed_samples - self.dataloader_offset),
            )
            _dataset = dataset_thread_iterator(train_dataset)
        else:
            _dataset = itertools.repeat(self.prepare_data(self.zero_batch))

        def _train_dataset(dataset):
            with self.mesh:
                for batch in dataset:
                    batch = self.allreduce_data_jit(batch)
                    yield batch

        self.train_dataset = _train_dataset(_dataset)

    def warmup_dataloader(self) -> None:
        self.next_data_batch()

    def next_data_batch(self):
        return next(self.train_dataset)

    def _rng_and_init_data(self, seq_len: int | None = None):
        seq_len = seq_len or self.model_config.sequence_len
        rng = jax.ShapeDtypeStruct((2,), jnp.uint32)
        if hasattr(self.dataset, "example_data_shape"):
            init_data = self.dataset.example_data_shape(self.batch_size)
        else:
            init_data = {
                "inputs": jax.ShapeDtypeStruct((self.batch_size, seq_len), jnp.int32),
                "targets": jax.ShapeDtypeStruct((self.batch_size, seq_len), jnp.int32),
                "mask": jax.ShapeDtypeStruct((self.batch_size, seq_len), jnp.bool),
            }

        return rng, init_data

    def override_model_config(self):
        if self.enable_determinism:
            self.model_config.model_config.attn_config.enable_determinism = True
            self.model_config.one_hot_embed = True

    @log_elapsed_time(rank_logger)
    @tracer.start_as_current_span("create_executable")
    def create_executable(self) -> None:
        if self.model_config.sharding_config is None:
            sharding_context = make_legacy_sharding_context(self.mesh)
        else:
            sharding_context = make_sharding_context_from_config(
                "model_specific", self.mesh, self.model_config.sharding_config
            )

        @hk.transform
        def loss_fn(batch):
            return self.model_config.make(
                sharding_context=sharding_context,
            ).loss(None, batch, is_training=True)

        self.loss_fn = loss_fn

        @hk.transform
        def loss_fn_more_metrics(batch):
            return self.model_config.make(
                sharding_context=sharding_context,
            ).loss(None, batch, is_training=True, flag_compute_more_metrics=True)

        self.loss_fn_more_metrics = loss_fn_more_metrics

        @hk.transform
        def logits_eval_fn(inputs):
            return (
                self.model_config.make(
                    sharding_context=sharding_context,
                )
                .forward(inputs, is_training=False)[0]
                .output
            )

        self.logits_eval_fn = logits_eval_fn

        self.create_optim()
        rng, init_data = self._rng_and_init_data()

        lr_shape = jax.ShapeDtypeStruct((), dtype=jnp.float32)
        with self.mesh:
            self.state_shape = jax.eval_shape(self.init, init_data, rng)

        self.state_sharding = self.get_sharding(self.state_shape)

        def init(r):
            return self.init(init_data, r)

        self.init_jit = JittedOrCompiled(
            jax.jit(init, in_shardings=None, out_shardings=self.state_sharding)
        )

        self.register_jit_function(self.init_jit, rng)

        self.allreduce_data_jit_by_shape = {}
        self.create_extra_executables(init_data)
        if self.evals:
            self.loss_fn_more_metrics_jit_by_shape = {}
            self.logits_eval_fn_jit_by_shape = {}
            self.zero_batch_eval_by_shape = {}
            self.maybe_create_eval_executables()
            if self.eval_every_n <= 0:
                rank_logger.warn(
                    "Evals is not empty, but eval_every_n is <= 0. In this case update_jit won't be compiled and training job won't start. Consider setting eval_every_n to a positive large value to skip eval but keep training loop."
                )
                return

        if not self.init_opt_state:
            rank_logger.info("Not compiling update_jit because init_opt_state=False")
            self.update_jit = None
            return

        self.update_jit = JittedOrCompiled(
            jax.jit(
                self.update,
                in_shardings=(
                    self.state_sharding,
                    NamedSharding(
                        self.mesh,
                        P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
                    ),
                    None,
                ),
                out_shardings=(self.state_sharding, None, None),
                donate_argnums=(0,),
            )
        )

        compiler_options = None if not self.xla_flag_overrides else self.xla_flag_overrides
        self.register_jit_function(
            self.update_jit,
            self.state_shape,
            init_data,
            lr_shape,
            compiler_options=compiler_options,
        )

    def maybe_create_eval_executables(self) -> None:
        if not self.evals:
            return
        for eval_module in self.evals:
            eval_name, _, eval_dataset, eval_bs_per_device, eval_seq_len = (
                eval_module.eval_name,
                eval_module.eval_conf,
                eval_module.eval_dataset,
                eval_module.eval_bs_per_device,
                eval_module.eval_seq_len,
            )
            rank_logger.info(f"Creating eval executables for {eval_name}")
            _ = self.get_eval_executable_by_shape(
                eval_bs_per_device, eval_seq_len, dataset=eval_dataset
            )

    def get_eval_executable_by_shape(
        self, batch_size: int | float, seq_len: int, dataset: Any | None = None
    ):
        shape_dataset = dataset if dataset is not None else self.dataset
        dataset_key = f"{type(shape_dataset).__module__}.{type(shape_dataset).__qualname__}"
        key = f"{batch_size}_{seq_len}_{dataset_key}"
        if key in self.loss_fn_more_metrics_jit_by_shape:
            return (
                self.loss_fn_more_metrics_jit_by_shape[key],
                self.logits_eval_fn_jit_by_shape[key],
                self.allreduce_data_jit_by_shape[key],
                self.zero_batch_eval_by_shape[key],
            )

        rng = jax.ShapeDtypeStruct((2,), jnp.uint32)
        init_data = self.prepare_data(
            shape_dataset.example_data(
                int(self.read_bsz_per_process * batch_size / self.bs_per_device)
            )
        )

        data_sharding = NamedSharding(
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis)),
        )
        new_data_sharding = NamedSharding(
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )

        allreduce_data_func = lambda batch: self.allreduce_data(batch)
        allreduce_data_func.__name__ = f"allreduce_data_{batch_size}_{seq_len}"
        allreduce_data_jit = JittedOrCompiled(
            jax.jit(
                allreduce_data_func, in_shardings=(data_sharding,), out_shardings=new_data_sharding
            )
        )
        self.register_jit_function(allreduce_data_jit, init_data)
        self.allreduce_data_jit_by_shape[key] = allreduce_data_jit
        init_data = allreduce_data_jit(init_data)

        logits_eval_fn_inputs = init_data

        dummy_local_data = {
            k: jax.ShapeDtypeStruct(
                (int(self.read_bsz_per_process * batch_size / self.bs_per_device), *v.shape[1:]),
                v.dtype,
            )
            for k, v in init_data.items()
        }
        zero_batch_eval = jax.tree.map(jnp.zeros_like, dummy_local_data)

        self.zero_batch_eval_by_shape[key] = zero_batch_eval

        loss_fn_shape = lambda params, rng, x: self.loss_fn_more_metrics.apply(params, rng, x)
        loss_fn_shape.__name__ = f"loss_fn_{batch_size}_{seq_len}"
        loss_fn_more_metrics_jit = JittedOrCompiled(
            jax.jit(
                loss_fn_shape,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    NamedSharding(
                        self.mesh,
                        P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
                    ),
                ),
                out_shardings=(None, None),
            )
        )

        self.register_jit_function(
            loss_fn_more_metrics_jit,
            self.state_shape.params,
            rng,
            init_data,
        )
        self.loss_fn_more_metrics_jit_by_shape[key] = loss_fn_more_metrics_jit

        logits_eval_fn_shape = lambda params, rng, x: self.logits_eval_fn.apply(params, rng, x)
        logits_eval_fn_shape.__name__ = f"logits_eval_fn_{batch_size}_{seq_len}"
        logits_eval_fn_jit = JittedOrCompiled(
            jax.jit(
                logits_eval_fn_shape,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    NamedSharding(
                        self.mesh,
                        P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
                    ),
                ),
                out_shardings=NamedSharding(
                    self.mesh,
                    P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
                ),
            )
        )

        self.register_jit_function(
            logits_eval_fn_jit,
            self.state_shape.params,
            rng,
            logits_eval_fn_inputs,
        )
        self.logits_eval_fn_jit_by_shape[key] = logits_eval_fn_jit
        return (
            self.loss_fn_more_metrics_jit_by_shape[key],
            self.logits_eval_fn_jit_by_shape[key],
            self.allreduce_data_jit_by_shape[key],
            self.zero_batch_eval_by_shape[key],
        )

    def adjust_host_sharding(self, host_sharding):
        return host_sharding

    def create_extra_executables(self, init_data=None) -> None:
        host_memory_kind = "pinned_host"
        if jax.default_backend() != "gpu":
            host_memory_kind = "unpinned_host"

        self.host_sharding = self.adjust_host_sharding(
            jax.tree.map(lambda s: s.with_memory_kind(host_memory_kind), self.state_sharding)
        )

        def h2d_copy(src, dst):
            dst = jax.device_put(src, self.state_sharding)
            return src, dst

        from xrex.train import checkpoint_write

        checkpoint_write.install_offload_state(self)
        self.reload_state = JittedOrCompiled(
            jax.jit(
                h2d_copy,
                in_shardings=(self.host_sharding, self.state_sharding),
                out_shardings=(self.host_sharding, self.state_sharding),
                donate_argnums=(0, 1),
                keep_unused=True,
            )
        )
        checkpoint_write.register_offload_state(self)
        self.register_jit_function(
            self.reload_state, self.state_shape, self.state_shape, phase="early"
        )

        def identity(x):
            return x

        self.iden_jit = JittedOrCompiled(
            jax.jit(
                identity,
                in_shardings=(self.state_sharding,),
                out_shardings=self.state_sharding,
                donate_argnums=0,
            )
        )
        self.register_jit_function(self.iden_jit, self.state_shape, phase="early")

        unwrapped_state_shape = unwrap_tree(self.state_shape)

        def norm(tree):
            return jax.tree.map(
                lambda x: jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)))), tree
            )

        self.norm_jit = JittedOrCompiled(
            jax.jit(
                norm,
                in_shardings=(self.state_sharding,),
                out_shardings=jax.sharding.NamedSharding(self.mesh, P()),
            )
        )
        self.register_jit_function(self.norm_jit, unwrapped_state_shape, phase="early")

        def compute_checksums(state):
            return checksum.compute_checksums(state, self.state_sharding, self.mesh)

        self.compute_checksums_jit = JittedOrCompiled(
            jax.jit(compute_checksums, in_shardings=(self.state_sharding,))
        )
        self.register_jit_function(self.compute_checksums_jit, unwrapped_state_shape, phase="early")

        if init_data is None:
            return

        dummy_local_data = {
            k: jax.ShapeDtypeStruct((self.read_bsz_per_process, *v.shape[1:]), v.dtype)
            for k, v in init_data.items()
        }
        self.zero_batch = jax.tree.map(jnp.zeros_like, dummy_local_data)

        data_sharding = NamedSharding(
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis)),
        )
        new_data_sharding = NamedSharding(
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )

        @functools.partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(data_sharding.spec,),
            out_specs=new_data_sharding.spec,
            check_rep=False,
        )
        def allreduce_data(batch):
            batch = jax.lax.psum_scatter(
                jax.tree.map(lambda x: jnp.uint8(x) if x.dtype == jnp.bool_ else x, batch),
                ("seq", "model"),
                scatter_dimension=1,
                tiled=True,
            )
            for key in ("mask", "invalid_example", "invalid_image_example"):
                if key in batch:
                    batch[key] = jnp.bool(batch[key])
            return batch

        self.allreduce_data = allreduce_data
        self.allreduce_data_jit = JittedOrCompiled(
            jax.jit(allreduce_data, in_shardings=(data_sharding,), out_shardings=new_data_sharding)
        )
        self.register_jit_function(self.allreduce_data_jit, init_data)

    def checksum_dict(self):
        unwrapped_state = unwrap_tree(self.state)
        checksums = self.compute_checksums_jit(unwrapped_state)
        return checksum.get_checksum_dict(
            unwrapped_state,
            self.state_sharding,
            self.mesh,
            checksums,
        )

    def register_jit_function(
        self,
        jitted: JittedOrCompiled,
        *args: Any,
        compiler_options: dict[str, Any] | None = None,
        phase: str = "late",
    ) -> None:
        if not self.enable_aot:
            return
        if not hasattr(self, "_registered_jitted"):
            self._registered_jitted = {}

        fun_name = jitted.name()
        if fun_name in self._registered_jitted:
            raise ValueError(f"Unable to reregister Jitted function: {fun_name}")
        self._registered_jitted[fun_name] = (jitted, *args, compiler_options, phase)

    @log_elapsed_time(rank_logger)
    @tracer.start_as_current_span("compile_or_load")
    def compile_or_load(self, phases: set[str] | None = None) -> None:
        if not self.enable_aot:
            return
        if not self._registered_jitted:
            raise ValueError("No jitted functions registered.")

        selected = {
            name: entry
            for name, entry in self._registered_jitted.items()
            if phases is None or entry[-1] in phases
        }
        if not selected:
            return

        traced = {}
        with self.mesh:
            for name, jitted_and_args in selected.items():
                jitted, *args, compiler_options, _phase = jitted_and_args
                compiler_options = compiler_options or {}
                if self.dump_all_hlos:
                    compiler_options["dump_all_hlos"] = True
                traced[name] = TracedWithOptions(
                    traced=jitted.trace(*args),
                    options=compiler_options,
                    add_location_info=self.aot_dump_loc,
                )

        all_compiled, aot_metrics = compile_or_load_all_traced(
            traced,
            self.aot_cache_dir,
            self.fdo_profile_dir,
            str(self.ctx.run_dir),
            aot_dump=self.aot_dump,
            devices=self.mesh.devices,
        )
        self.aot_metrics = {**(self.aot_metrics or {}), **aot_metrics}
        for name, c in all_compiled.items():
            if name not in traced:
                raise ValueError(f"Unable to found the traced function for {name}.")
            jitted = self._registered_jitted[name][0]
            jitted.jitted = c

        for name in selected:
            del self._registered_jitted[name]
        if not self._registered_jitted:
            del self._registered_jitted

    def initialize_checkpoint_index(self, checkpoint: CheckpointMeta | None) -> None:
        self.current_ckpt_index = 0
        if checkpoint is not None:
            ckpt_index = checkpoint.checkpoint_index
            self.current_ckpt_index = ckpt_index if ckpt_index is not None else 0

    @tracer.start_as_current_span("create_state")
    def create_state(self, ctx: TrainerContext) -> None:
        with self.mesh:
            rng = jax.random.PRNGKey(self.rng_seed)
            self.state = None
            self.state = self.init_jit(rng)

        loaded, self.elapsed_samples, self.elapsed_tokens = self.maybe_load_checkpoint(ctx)

        if loaded:
            if self.reinit_on_load and ctx.checkpoint.is_manual_load():
                start_step = self.step_count_start if self.step_count_start is not None else 0
                rank_logger.info(
                    f"Resetting step and elapsed_samples to {start_step} because reinit_on_load={self.reinit_on_load}"
                )
                self.state = self.state._replace(step=jnp.array(start_step))
                self.elapsed_samples = 0
                self.elapsed_tokens = 0
        else:
            rank_logger.info("Starting from scratch")

        self.initialize_checkpoint_index(ctx.checkpoint)

        if self.state.opt_state:
            hyperparam_dtype = jnp.bfloat16 if self.precision_level < 2 else jnp.float32
            self.state.opt_state.hyperparams["weight_decay"] = jnp.array(
                self.optim_config.weight_decay, dtype=hyperparam_dtype
            )
            self.state.opt_state.hyperparams["b1"] = jnp.array(
                self.optim_config.b1, dtype=hyperparam_dtype
            )
            self.state.opt_state.hyperparams["b2"] = jnp.array(
                self.optim_config.b2, dtype=hyperparam_dtype
            )

        self.state = self.iden_jit(self.state)

        rank_logger.info(f"#parameters: {compute_num_elements(self.state.params) / 1e9:.2f} B")
        rank_logger.info(
            f"#opt-parameters: {compute_num_elements(self.state.opt_state) / 1e9:.2f} B"
        )
        rank_logger.info(
            f"#stored-parameters: {compute_num_stored_elements(self.state.params) / 1e9:.2f} B"
        )
        rank_logger.info(
            f"Memory usage after create_state: {get_bytes_in_use() / (1 << 30):.2f} GB"
        )

    def get_sharding(self, shapes):
        shapes = jax.tree.map(
            lambda p: NamedSharding(self.mesh, p.pspec if hasattr(p, "pspec") else P()),
            shapes,
            is_leaf=lambda x: isinstance(x, Parameter),
        )
        return shapes

    def init(self, batch, rng) -> TrainingStateProtocol:
        batch = jax.tree.map(jnp.ones_like, batch)
        rng, init_rng = jax.random.split(rng)

        if self.rng_seed == 0:
            init_rng = jax.random.PRNGKey(0), [jax.random.PRNGKey(0)] * 1000

        initial_params = self.loss_fn.init(init_rng, batch)

        if self.init_opt_state:
            initial_opt_state = self.optim.init(initial_params)
        else:
            initial_opt_state = None
        if self.precision_level == 0:
            initial_params = jax.tree.map(cast_bfloat16, initial_params)
        if self.precision_level < 2:
            initial_opt_state = selective_tree_cast(default_dtype="bfloat16")(initial_opt_state)

        return TrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            rng=rng,
            step=jnp.array(0),
        )

    def next_eval_batch(self, eval_dataset, data_world_size: int, num_processes_per_data_rank: int):
        try:
            eval_batch = next(eval_dataset)
            success = jnp.ones((1), dtype=jnp.int32)
        except StopIteration:
            rank_logger.info("Stop iter, current rank reached end of dataset")
            success = jnp.zeros((1), dtype=jnp.int32)
        except Exception as e:
            print(f"Error loading dataset at batch: {e}")
            success = jnp.zeros((1), dtype=jnp.int32)

        global_success = multihost_utils.process_allgather(success, tiled=True)
        num_success = np.array(global_success).sum().item()
        assert num_success <= data_world_size * num_processes_per_data_rank, (
            f"There are {data_world_size * num_processes_per_data_rank} workers but {num_success} workers actually load data"
        )

        if num_success < data_world_size * num_processes_per_data_rank:
            if num_success == 0:
                rank_logger.info("Stop iter: all ranks reached end of dataset")
            else:
                rank_logger.info("Stop iter: at least one rank failed to fetch more rows")
            return None

        if "datasets" in eval_batch:
            eval_batch.pop("datasets")
        return eval_batch

    def eval(
        self,
        soft_step: int,
        active_evals: list[EvalModule] | None = None,
        metric_folder: str | None = None,
        elapsed_samples: int | None = None,
    ):
        rank_logger.info("Running evals at %s", soft_step)
        rng = jax.random.PRNGKey(self.rng_seed)
        _, new_rng = jax.random.split(rng)

        all_eval_metrics = {}

        def _eval_iter(
            dataset, eval_bs_per_device, name=None, _allreduce_data_jit=lambda x: x, zero_batch=None
        ):
            if zero_batch is None:
                zero_batch = self.zero_batch
            batch_size = int(eval_bs_per_device * self.num_devices)
            read_bsz_per_process, data_world_size, num_processes_per_data_rank = (
                self.parallel_config.get_local_batch_size_and_rank(batch_size)
            )
            eval_dataset = dataset.make(
                shard_index=self.ctx.rank // num_processes_per_data_rank,
                num_shards=data_world_size,
                batch_size=read_bsz_per_process,
                skip_rows=0,
                server_hosts=self.ctx.ip_addrs,
                run_server=self.ctx.rank
                % (
                    self.parallel_config.num_devices_per_node
                    // self.parallel_config.num_devices_per_process
                )
                == 0,
                keep_and_pad_partial_batch=True,
            )

            bidx = 0
            total_yield = 0
            while True:
                next_eval_batch = self.next_eval_batch(
                    eval_dataset, data_world_size, num_processes_per_data_rank
                )
                if next_eval_batch is None:
                    break
                if self.chunk_index == 0:
                    eval_batch = self.prepare_data(next_eval_batch)
                else:
                    eval_batch = self.prepare_data(zero_batch)

                eval_batch = _allreduce_data_jit(eval_batch)
                yield eval_batch
                if "mask" in eval_batch and self.debug_tensor_dump_output_folder is None:
                    n_valid_samples = (np.sum(eval_batch["mask"], axis=-1) > 0).sum()
                    n_valid_samples = jnp.array(n_valid_samples)
                    n_valid_samples_all_ranks = multihost_utils.process_allgather(
                        n_valid_samples, tiled=True
                    )
                    total_valid_samples = np.array(n_valid_samples_all_ranks).sum().item()
                    total_yield += total_valid_samples
                    rank_logger.info(
                        f"Eval at {soft_step} on {name}, batch {bidx} with {n_valid_samples} valid examples on rank {self.ctx.rank}, global valid batch size = {total_valid_samples}, total yield = {total_yield}"
                    )
                bidx += 1

        evals_to_use = active_evals if active_evals is not None else self.evals
        for eval_module in evals_to_use:
            eval_name, eval_conf, eval_dataset, eval_bs_per_device, eval_seq_len = (
                eval_module.eval_name,
                eval_module.eval_conf,
                eval_module.eval_dataset,
                eval_module.eval_bs_per_device,
                eval_module.eval_seq_len,
            )
            _loss_fn, _logits_eval_fn, _allreduce_data_jit, _zero_batch_eval = (
                self.get_eval_executable_by_shape(
                    eval_bs_per_device, eval_seq_len, dataset=eval_dataset
                )
            )
            _eval_func = lambda x, params: _logits_eval_fn(params, new_rng, x)

            eval_results = run_forward_evals_new(
                evals=[(eval_name, eval_conf)],
                data_iter=_eval_iter(
                    eval_dataset,
                    eval_bs_per_device,
                    eval_name,
                    _allreduce_data_jit=_allreduce_data_jit,
                    zero_batch=_zero_batch_eval,
                ),
                forward_fn=_eval_func,
                state=self.state,
                mesh=self.mesh,
            )

            eval_metrics = report_forward_eval_results(
                results=eval_results,
                metric_folder=metric_folder,
                elapsed_samples=elapsed_samples,
                is_data_shard_output_rank=self.chunk_index == 0,
            )
            rank_logger.info(f"eval_metrics: {eval_metrics}")
            all_eval_metrics.update(eval_metrics)
        return all_eval_metrics

    def is_valid_step(self, grads: jax.Array) -> tuple[jax.Array, jax.Array]:
        def keep_step(grad_norm: jax.Array) -> jax.Array:
            if self.grad_norm_keep_threshold:
                return jnp.logical_and(
                    jnp.all(jnp.isfinite(grad_norm)),
                    jnp.all(jnp.less(grad_norm, self.grad_norm_keep_threshold)),
                )
            return jnp.all(jnp.isfinite(grad_norm))

        grad_norm = compute_norm(grads)
        valid_step = keep_step(grad_norm)
        return valid_step, grad_norm

    def _get_checkpoint_infos(self):
        if self.checkpoint_config.replication_mode == "none":
            return None, None, None
        elif self.checkpoint_config.replication_mode == "full":
            axes = checkpointing_common.get_replicated_axes(self.state_sharding)
        else:
            axes = jax.tree.map(
                lambda _, p=P(self.checkpoint_config.replica_axis): p, self.state_sharding
            )
        srcs = checkpointing_common.get_sources(self.state_sharding, axes)
        mask = checkpointing_common.get_load_mask(axes, srcs, self.mesh)
        return axes, srcs, mask

    def maybe_load_checkpoint(
        self, ctx: TrainerContext, tag: str | None = None
    ) -> tuple[bool, int, int]:
        if not self.checkpoint_config.from_checkpoint or not ctx.checkpoint:
            rank_logger.info("Not loading checkpoint; starting from scratch")
            return False, 0, 0

        restore_kind = next(
            k for k in jax.tree.leaves(jax.tree.map(lambda s: s.memory_kind, self.host_sharding))
        )
        restore_staging_sharding = jax.tree.map(
            lambda s: s.with_memory_kind(restore_kind), self.state_sharding
        )
        self.host_state = jax.device_put(self.state, restore_staging_sharding)

        rename = None

        loads: dict[str, dict[str, jax.Array]] = {}

        if ctx.checkpoint.format == "orbax":
            host_state = unwrap_tree(self.host_state)

            do_not_load_opt_state = (
                self.checkpoint_config.no_opt_state or self.reinit_on_load
            ) and ctx.checkpoint.is_manual_load()

            if do_not_load_opt_state:
                rank_logger.info("Not loading optimizer state from checkpoint")
                host_state = host_state.purge_opt_state()

            host_state = tree_to_dict(host_state)
            domains = None

            axes, srcs, mask = self._get_checkpoint_infos()
            mask = tree_to_dict(unwrap_tree(mask))

            loads[ctx.checkpoint.path] = host_state

            if ctx.checkpoint.is_manual_load():
                if self.checkpoint_config.no_loading:
                    regex = re.compile(self.checkpoint_config.no_loading, flags=re.X)
                    for name in host_state:
                        if regex.search(name):
                            rank_logger.info("Not loading tensor %s from checkpoint", name)
                            host_state[name] = None

                if self.checkpoint_config.extra_checkpoint_dirs:
                    patterns = {
                        re.compile(pattern, flags=re.X): checkpoint_path
                        for pattern, checkpoint_path in self.checkpoint_config.extra_checkpoint_dirs.items()
                    }
                    for key in tuple(host_state):
                        for pattern, checkpoint_path in patterns.items():
                            if pattern.search(key):
                                loads.setdefault(checkpoint_path, {})[key] = host_state.pop(key)
                                break

                rename_patterns = self.checkpoint_config.rename_patterns
                rename_state_patterns = self.checkpoint_config.rename_state_patterns

                if rename_patterns:
                    rename = lambda n: checkpointing_load.rename_tensor(n, rename_patterns)

                if rename_state_patterns:
                    for checkpoint_path, partial_host_state in loads.items():
                        loads[checkpoint_path] = {}
                        for name, tensor in partial_host_state.items():
                            name = checkpointing_load.rename_tensor(name, rename_state_patterns)
                            loads[checkpoint_path][name] = tensor

            for checkpoint_path, partial_host_state in loads.items():
                checkpointing_load.load_checkpoint(
                    checkpoint_path,
                    partial_host_state,
                    load_mask=mask,
                    rename=rename,
                    domains=domains,
                    tag=tag,
                    timeout=self.checkpoint_config.timeout_secs,
                )

            self.state = None
            self.state = jax.device_put(self.host_state, self.state_sharding)
            self.host_state = jax.device_put(self.state, self.host_sharding)

            if mask:
                axes_sizes = {}
                for key, axes_ in tree_to_dict(axes, keep_none=False).items():
                    axes_sizes[key] = np.prod([self.mesh.shape[axis] for axis in axes_])
                rank_logger.info(
                    "Broadcasting %i checkpoint tensors across %.1f devices on average",
                    len(axes_sizes),
                    np.mean(tuple(axes_sizes.values()), dtype=np.float32).item(),
                )

                _, treedef = jax.tree.flatten(self.state_shape)
                axes = jax.tree.unflatten(treedef, jax.tree.flatten(axes)[0])
                srcs = jax.tree.unflatten(treedef, jax.tree.flatten(srcs)[0])
                self.state = checkpointing_load.broadcast_replicated(
                    self.state, axes, srcs, self.mesh
                )

        else:
            logger.warning("Ignoring unknown checkpoint format %s", ctx.checkpoint.format)
            return None, 0, 0

        if self.checkpoint_config.verify_checksums:
            checksum_dict = self.checksum_dict()

        for i, (checkpoint_path, partial_host_state) in enumerate(loads.items()):
            restored_fields = set()
            names = []
            local_size = 0
            for name, entry in partial_host_state.items():
                if entry is None:
                    continue

                names.append(name)
                field, *_ = name.split(".", 1)
                restored_fields.add(field)

                shard_shape = entry.sharding.shard_shape(entry.shape)
                nbytes = np.dtype(entry.dtype).itemsize * np.prod(shard_shape)
                local_size += nbytes * len(entry.sharding.addressable_devices)

            extra0 = ""
            if len(loads) > 1:
                extra0 = f" ({i}/{len(loads)})"

            extra1 = ""
            if "step" in partial_host_state:
                extra1 = (
                    " checkpoint step is %i"
                    % (self.state[0] if isinstance(self.state, list) else self.state).step.item()
                )

            rank_logger.info(
                "Restored from checkpoint%s: %i tensors (%.1f GiB per GPU) on %i GPUs, fields: %s.%s",
                extra0,
                len(names),
                local_size / (1 << 30),
                self.mesh.size,
                restored_fields,
                extra1,
            )

            if self.checkpoint_config.verify_checksums:
                checksums_file = f"{checkpoint_path}/checksums.0.json"
                try:
                    if checksum.compare_checksum_dicts(
                        checksums_file, checksum_dict, rename, names=names
                    ):
                        rank_logger.info("Checkpoint checksums match%s", extra0)
                except FileNotFoundError:
                    rank_logger.warn("verify_checksums=True but %r not found", checksums_file)

        rank_logger.info("Deleting pinned CPU host state")
        self.host_state = None

        return True, ctx.checkpoint.elapsed_samples, ctx.checkpoint.elapsed_tokens

    def run(self) -> None:
        if self.use_mock_gpu_client:
            logger.info("use_mock_gpu_client is set to true. Skip the train loop.")
            return

        ctx = self.ctx
        with self.mesh:
            ctx.on_train_start()

            preemption_notifier = getattr(ctx, "preemption_notifier", None)
            if preemption_notifier is not None:
                preemption_notifier.enable()

            step = self.state[0].step if isinstance(self.state, list) else self.state.step
            step = step.item()
            saved_periodic = False
            for soft_step in itertools.count():
                if self.max_steps is not None and step > self.max_steps:
                    rank_logger.info(f"Step limit reached ({self.max_steps=})")
                    break

                if self.max_samples is not None and self.elapsed_samples > self.max_samples:
                    rank_logger.info(f"Step limit reached ({self.max_samples=})")
                    break

                next_data_batch_start = time.perf_counter()
                batch = self.next_data_batch()
                next_data_batch_time = time.perf_counter() - next_data_batch_start
                lr = self.lr_schedule_in_samples(self.elapsed_samples, self.batch_size)

                if ctx.rank == ctx.world_size - 1:
                    compute_step_start = time.perf_counter()
                self.state, metrics, extras = self.update_jit(self.state, batch, lr)

                if "valid_step" in metrics:
                    valid_step = metrics["valid_step"].item()
                elif ctx.rank == ctx.world_size - 1:
                    del metrics[("valid_step",)]
                    metrics = metrics_utils.extract_stacked_metrics(metrics)
                    valid_step = metrics["valid_step"].item()
                else:
                    valid_step = metrics.pop(("valid_step",))
                    valid_step = valid_step.item()

                self.elapsed_samples += self.batch_size
                del extras

                is_eval_step = (
                    self.eval_every_n > 0 and step % self.eval_every_n == 0 and step > 0
                ) or (self.max_steps is not None and step == self.max_steps and self.evals)
                if is_eval_step:
                    eval_metrics = self.eval(soft_step)
                    metrics.update(eval_metrics)

                self.handle_profiler(soft_step, metrics.get("loss"))

                casted_vis_metrics = {}

                if ctx.rank == ctx.world_size - 1:
                    metrics["next_data_batch_time"] = next_data_batch_time
                    metrics["gpu_step_time"] = time.perf_counter() - compute_step_start
                    metrics = self.handle_metrics(soft_step, metrics)
                else:
                    metrics.clear()

                step_metrics = cast_metrics(metrics) | casted_vis_metrics

                is_healthy = True
                preemption_notifier = getattr(ctx, "preemption_notifier", None)
                if preemption_notifier is not None and preemption_notifier.preempted.is_set():
                    is_healthy = False

                driver_cmd = ctx.on_step(step_metrics, metrics_utils.pop(), is_healthy)

                if driver_cmd == DriverMessage.SAVE_CHECKPOINT_ASAP:
                    rank_logger.info("Driver requested emergency checkpoint")
                    self.save_checkpoint(blocking=True)
                    ctx.on_completed("Emergency checkpoint saved")
                    raise RuntimeError("Exiting after emergency checkpoint saved")

                if driver_cmd == DriverMessage.TERMINATE:
                    checkpointing.wait_until_finished()
                    sys.exit(0)

                saved_periodic = self.checkpoint_config.should_save(soft_step)
                if saved_periodic:
                    self.save_checkpoint()

                step += 1

            if self.checkpoint_config.save_final_checkpoint and not saved_periodic:
                self.save_checkpoint()

        checkpointing.wait_until_finished()
        ctx.on_step({}, metrics_utils.pop())

    def handle_profiler(self, soft_step: int, block_tensor: jax.Array) -> None:
        if self.jax_profile and jax.process_index() == self.jax_profile_rank:
            if soft_step == self.jax_profile_start_step:
                jax.block_until_ready(block_tensor)
                profiling_path = str(self.ctx.run_dir) + "/tensorboard/" + str(jax.process_index())
                rank_logger.info(f"Write jax profiling to {profiling_path}")
                start_trace(profiling_path)
            if soft_step == self.jax_profile_stop_step:
                jax.block_until_ready(block_tensor)
                stop_trace()

        if self.nsys_start_rank == jax.process_index():
            self.maybe_nsys_step(soft_step, block_tensor)

    def handle_metrics(
        self, soft_step: int, metrics: dict[str, float | int]
    ) -> dict[str, float | int]:
        now = time.time()
        step_time = now - self.prev_ts
        self.prev_ts = time.time()
        num_seq_per_sec_per_device = self.batch_size / step_time / self.num_devices
        mfu = self.model_config.compute_mfu(num_seq_per_sec_per_device)
        tflops = self.model_config.compute_tflops(num_seq_per_sec_per_device)

        if self.max_steps is not None:
            eta = (self.max_steps - self.prev_step) * step_time / 3600
            metrics["finished_percent"] = int(metrics["step"]) / self.max_steps
        elif self.max_samples is not None:
            eta = (
                (self.max_samples - self.elapsed_samples)
                / float(metrics["examples_per_batch"])
                * step_time
                / 3600
            )
            metrics["finished_percent"] = self.elapsed_samples / self.max_samples
        else:
            eta = -1
            metrics["finished_percent"] = -1
        self.prev_step = int(metrics["step"])

        metrics["soft_step"] = soft_step
        metrics["eta (hr)"] = eta
        metrics["step_time"] = step_time
        metrics["elapsed_samples"] = self.elapsed_samples
        metrics["mfu"] = mfu
        metrics["tflops"] = tflops
        metrics["peak_mem"] = get_peak_bytes_in_use() / (1 << 30)
        metrics["num_live_arrays"] = len(jax.live_arrays())
        return metrics


def cast_metrics(
    metrics: dict[str, Any], metric_names: set[str] | list[str] | None = None
) -> dict[str, int | float | None]:
    out = {}
    for k, v in metrics.items():
        if metric_names and k not in metric_names:
            continue
        try:
            out[k] = cast_val(k, v)
        except Exception as e:
            rank_logger.error(e)
            rank_logger.error(f"Error when processing {k}. {type(v)=}")
            raise e
    return out


def cast_val(key: str, val: Any) -> int | float | None:
    if type(val) in (int, float, bool, type(None)):
        return val
    elif (
        isinstance(val, (np.ndarray, jax.Array))
        or np.issubdtype(val.dtype, np.floating)
        or np.issubdtype(val.dtype, np.integer)
    ):
        try:
            if val.size > 1:
                return np.array(val).tolist()
            return val.item()
        except BaseException as e:
            rank_logger.info(f"Error casting value of {key}, {val=}. Error: {e}")
    else:
        raise TypeError(f"Unrecognized metric type for {key!r}: {type(val)}")


from xrex.train.checkpoint_write import install as _install_checkpoint_write

_install_checkpoint_write(Trainer)
