# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import gc
import importlib.util
import logging
import os
import sys
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from serde.json import to_json
from tabulate import tabulate
from xai_configlib import Config

from xrex.train.trainer import TrainerContext
from xrex.utils import metadata
from xrex.utils.gpu import (
    set_gpu_clocks_to_max,
)
from xrex.utils.log_util import configure_logging
from xrex.utils.tracer import TracerConfig, setup_tracer
from xrex.utils.utils import (
    PreemptionNotifier,
    disable_core_dumps,
    increase_fd_limit,
)

Tee = None
if importlib.util.find_spec("xai_logging_utils") is not None:
    from xai_logging_utils.tee import Tee

if TYPE_CHECKING:
    from xrex.driver.hooks import HookManager

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


@dataclass
class DriverContext:
    env: dict[str, str]
    run_config: Any
    train_fn: Callable[[TrainerContext], None]
    meta: metadata.MetadataManager
    hooks: HookManager

    preemption_notifier: PreemptionNotifier | None = None
    tee: Tee | None = None

    on_restart: Callable[[BaseException, int], None] | None = None
    """Optional callback invoked when the driver restarts after a failure.

    Called with ``(exception, restart_count)`` after the restart count has
    been incremented but before the next iteration begins.
    """


class Completed(Exception):
    pass


class ExceptionHandlerRegistry:
    def __init__(self):
        self.handlers = {}

    def register(self, type):
        def decorator(func):
            self.handlers[type] = func
            return func

        return decorator

    def get_handler(self, type):
        return self.handlers.get(type, None)


@dataclass
class MultiRunDriverState:
    ctx: DriverContext
    last_stop_cause: str = "<unknown>"
    num_restarts: int = 0
    run: metadata.Run | None = None
    ckpt: metadata.CheckpointMeta | None = None
    run_dir: Path = ""
    xla_flags_from_env: str = ""


@dataclass
class DriverConfig(Config):
    auto_restart: bool = True

    run_dir: str = "/tmp/runs"

    enable_tee: bool = True

    tee_logs_to_pod: bool = False

    handlers = ExceptionHandlerRegistry()

    tracer_config: TracerConfig = field(default_factory=TracerConfig)

    machine: str = "h100"

    config_name: str | None = None

    def _init_multi_run(self, ctx: DriverContext) -> MultiRunDriverState:
        return MultiRunDriverState(ctx, xla_flags_from_env=ctx.env.get("XLA_FLAGS", ""))

    def _prepare_single_run(self, mr_state: MultiRunDriverState):
        ctx: DriverContext = mr_state.ctx
        run, ckpt = ctx.meta.init_or_restart(
            ctx.run_config.name, ctx.run_config, ctx.run_config.load
        )
        run.config_name = self.config_name
        os.environ["XAI_RUN_ID"] = run.run_id
        run_dir = init_run_dir(self.run_dir, run, ckpt)

        if ctx.tee:
            combined = str(run_dir / "combined.log")
            stdout = [str(run_dir / "stdout.log"), combined]
            stderr = [str(run_dir / "stderr.log"), combined]
            if self.tee_logs_to_pod:
                stdout.append("/proc/1/fd/1")
                stderr.append("/proc/1/fd/2")

            ctx.tee.to(stdout=stdout, stderr=stderr)
            weakref.finalize(ctx.tee, print, "Logs at", combined)
        mr_state.run = run
        mr_state.ckpt = ckpt
        mr_state.run_dir = run_dir

        additional_xla_flags = {}
        ctx.env["XLA_FLAGS"] = (
            mr_state.xla_flags_from_env
            + " "
            + " ".join(f"{key}={value}" for key, value in additional_xla_flags.items())
        )
        logger.info(f"XLA flags: {ctx.env['XLA_FLAGS']}")
        logger.info(f"Initialized run directory {run_dir}")

    def _single_run(self, mr_state: MultiRunDriverState):
        raise NotImplementedError()

    def _handle_exception(self, e: BaseException, mr_state: MultiRunDriverState):
        e_name = type(e).__name__
        mr_state.last_stop_cause = f"driver caught exception {e_name}"
        if self.auto_restart:
            mr_state.last_stop_cause += ", restarting"
            logger.exception("Driver caught %s, restarting...", e_name)
            mr_state.num_restarts += 1
            if mr_state.ctx.on_restart is not None:
                mr_state.ctx.on_restart(e, mr_state.num_restarts)
        else:
            raise

    @handlers.register(KeyboardInterrupt)
    def _handle_keyboard_interrupt(self, e: KeyboardInterrupt, mr_state: MultiRunDriverState):
        mr_state.last_stop_cause = "keyboard interrupt"
        logger.exception("driver loop exiting due to %r", e)
        sys.exit(2)

    @handlers.register(SystemExit)
    def _handle_system_exit(self, e: SystemExit, mr_state: MultiRunDriverState):
        mr_state.last_stop_cause = f"sys.exit({e.code}) called"
        logger.exception("driver loop exiting due to %r", e)
        raise e

    @handlers.register(Completed)
    def _handle_completed(self, e: Completed, mr_state: MultiRunDriverState):
        del e
        mr_state.last_stop_cause = "run completed"
        logger.info("Run completed, driver loop exiting")
        sys.exit(0)

    def run(self, ctx: DriverContext):
        mr_state = self._init_multi_run(ctx)

        ctx.preemption_notifier = PreemptionNotifier()

        if self.enable_tee:
            if Tee is None:
                logger.info(
                    "enable_tee is set but xai_logging_utils is not installed; "
                    "stdout/stderr will not be duplicated into the run directory."
                )
            else:
                ctx.tee = Tee()

        setup_tracer("xrex.Driver", config=self.tracer_config)

        while True:
            try:
                self._prepare_single_run(mr_state)
                gc.disable()
                self._single_run(mr_state)
            except BaseException as e:
                handler = self.handlers.get_handler(type(e))
                if handler:
                    handler(self, e, mr_state)
                else:
                    self._handle_exception(e, mr_state)
            finally:
                gc.enable()

                ctx.hooks.on_train_stop()

                if ctx.tee:
                    ctx.tee.close()

            time.sleep(2)
            gc.collect()

    def signal_complete(self):
        raise Completed()


last_log_metrics = time.time()


def logged_metric_keys():
    return {
        "elapsed_samples",
        "eta (hr)",
        "examples_per_batch",
        "global_grad_norm",
        "gpu_step_time",
        "learning_rate",
        "loss",
        "mfu",
        "next_data_batch_time",
        "origin-loss",
        "peak_mem",
        "soft_step",
        "step",
        "finished_percent",
        "step_time",
        "tflops",
        "valid_grad",
        "valid_step",
        "weight_decay",
    }


def log_metrics(metrics: dict[str, float | int]):
    global last_log_metrics
    table = sorted([(k, v) for k, v in metrics.items() if k in logged_metric_keys()])
    logger.info("\n" + tabulate(table, tablefmt="outline", numalign="right"))

    now = time.time()
    logger.info("Metrics time: %.2f sec", now - last_log_metrics)
    last_log_metrics = now


def init_run_dir(
    base_dir: Path | str,
    run: metadata.Run,
    ckpt: metadata.CheckpointMeta | None = None,
    link_local_latest: bool = True,
) -> Path:
    base_dir = Path(base_dir)

    base_id, restart_count = run.run_id_parts()
    path = base_dir / run.user / run.name / base_id / str(restart_count)
    path.mkdir(parents=True, exist_ok=True)

    run_json = to_json(run) + "\n"
    (path / "run.json").write_text(run_json)

    if ckpt is not None:
        ckpt_json = to_json(ckpt) + "\n"
        (path / "source_checkpoint.json").write_text(ckpt_json)

    rpath = base_dir / run.user / run.name
    run_latest = rpath / "latest"
    if os.path.islink(run_latest):
        try:
            os.remove(run_latest)
        except FileNotFoundError:
            pass
    try:
        os.symlink(str(path.relative_to(rpath)), run_latest)
    except FileExistsError:
        pass

    if run.config_name:
        config_latest = base_dir / run.user / f"{run.config_name}.latest"
        if os.path.islink(config_latest):
            try:
                os.remove(config_latest)
            except FileNotFoundError:
                pass
        try:
            os.symlink(str(path.relative_to(config_latest.parent)), config_latest)
        except FileExistsError:
            pass

    if link_local_latest:
        local_latest = Path("/tmp") / "runs" / "latest"
        local_latest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if os.path.islink(run_latest):
                os.remove(local_latest)
        except FileNotFoundError:
            pass
        try:
            os.symlink(path, local_latest)
        except FileExistsError:
            pass

    return path


def configure_worker(
    ctx: TrainerContext,
    configure_logger: bool = True,
    configure_gpu: bool = True,
):
    if configure_logger:
        configure_logging(
            rank=ctx.rank,
            log_dir=ctx.run_dir,
            log_rotate=getattr(ctx.config, "log_rotate", False),
            log_rotate_max_bytes=getattr(ctx.config, "log_rotate_max_bytes", 10 * 1024 * 1024),
            log_rotate_backup_count=getattr(ctx.config, "log_rotate_backup_count", 10_000_000),
        )
    disable_core_dumps()
    increase_fd_limit()

    sd_id = 0
    nccl_log_dir = ctx.run_dir / "logs" / "trainer" / "nccl"
    nccl_log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"%h.sd{sd_id}.r{ctx.rank}.%p.log"
    path_nccl = str(nccl_log_dir / f"nccl_debug.{suffix}")
    os.environ["NCCL_DEBUG_FILE"] = path_nccl
    path_mscclpp = str(nccl_log_dir / f"mscclpp_debug.{suffix}")
    os.environ["MSCCLPP_DEBUG_FILE"] = path_mscclpp
    path_xccl = str(nccl_log_dir / f"xccl_debug.{suffix}")
    os.environ["XCCL_DEBUG_FILE"] = path_xccl
    rank_logger.debug(f"Write NCCL logs to {path_nccl}")

    if configure_gpu:
        try:
            set_gpu_clocks_to_max()
        except BaseException as e:
            logger.exception(e)
