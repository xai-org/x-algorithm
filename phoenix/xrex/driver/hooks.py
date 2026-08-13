# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import logging
import os
import queue
import re
import signal
import subprocess
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import isfinite
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np

from xrex import settings
from xrex.utils import cluster as cluster_identity
from xrex.utils import launch_env
from xrex.utils.log_timer import Timer, log_elapsed_time
from xrex.utils.metadata import (
    Jsonable,
    Run,
    checkpoint_dir_or_default,
    metadata_file_exists,
    read_metadata_file,
)

logger = logging.getLogger(__name__)

_RM_BATCH_SIZE = 50
_FINE_GRAINED_TOOL_ERROR_RE = re.compile(r"tool_response\.error_count\..*\.")
_TOKEN_OBJECTIVE_WANDB_RE = re.compile(
    r"(?:reward_op\.token_objective_|(?:^|/)token_objective\.)(?!.*mean$)"
)
_TOOL_REQUEST_COUNT_AGG_RE = re.compile(r"tool_request\..*\.count$")


def _should_omit_wandb_metric(key: str) -> bool:
    return bool(
        _FINE_GRAINED_TOOL_ERROR_RE.search(key)
        or _TOKEN_OBJECTIVE_WANDB_RE.search(key)
        or _TOOL_REQUEST_COUNT_AGG_RE.search(key)
    )


def _filter_wandb_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if not _should_omit_wandb_metric(k)}


_CLICKHOUSE_HOST = settings.CLICKHOUSE_HOST
_CLICKHOUSE_RUN_URL = settings.CLICKHOUSE_RUN_URL


def _import_wandb():
    import wandb

    return wandb


class HookManager:
    def __init__(self):
        self.hooks: list[DriverHook] = []

    def add_hook(self, hook: DriverHook):
        self.hooks.append(hook)

    def on_boot(self, run: Run, run_dir: Path, config: Jsonable):
        for hook in self.hooks:
            hook.on_boot(run, run_dir, config)

    def on_train_start(self, model_config: Any):
        for hook in self.hooks:
            hook.on_train_start(model_config)

    def on_step(self, metrics: Mapping[str, Any]):
        for hook in self.hooks:
            hook.on_step(metrics)

    def on_checkpoint_save(
        self,
        *,
        path: Path,
        step: int,
        checkpoint_index: int | None = None,
    ) -> None:
        for hook in self.hooks:
            try:
                hook.on_checkpoint_save(path=path, step=step, checkpoint_index=checkpoint_index)
            except BaseException:
                logger.exception(
                    "%s.on_checkpoint_save raised; continuing with remaining hooks",
                    type(hook).__name__,
                )

    def on_train_rollback(self):
        for hook in self.hooks:
            hook.on_train_rollback()

    def on_train_stop(self):
        for hook in self.hooks:
            try:
                hook.on_train_stop()
            except BaseException:
                logger.exception(
                    "%s.on_train_stop raised; continuing with remaining hooks",
                    type(hook).__name__,
                )


class DriverHook:
    def on_boot(self, run: Run, run_dir: Path, config: Jsonable):
        pass

    def on_train_start(self, model_config: Any):
        pass

    def on_step(self, metrics: Mapping[str, Any]):
        pass

    def on_checkpoint_save(
        self,
        *,
        path: Path,
        step: int,
        checkpoint_index: int | None = None,
    ) -> None:
        pass

    def on_train_rollback(self):
        pass

    def on_train_stop(self):
        pass


@lru_cache
def _rich_media_types() -> tuple[type, ...]:
    if importlib.util.find_spec("wandb") is None:
        return ()
    wandb = _import_wandb()
    return (wandb.Image, wandb.Video, wandb.Table)


def filter_serializable(metrics: Mapping[str, Any]):
    import jax

    unserializable = _rich_media_types()

    cast_val = (
        lambda x: None
        if isinstance(x, unserializable) or (isinstance(x, float) and not isfinite(x))
        else x
    )
    return jax.tree.map(cast_val, metrics, is_leaf=lambda x: not isinstance(x, (dict, list)))


def write_metrics_file(path: Path, metrics: Mapping[str, Any]):
    with path.open("a") as f:
        try:
            json.dump(filter_serializable(metrics), f)
            f.write("\n")
        except Exception as e:
            logger.info(f"{type(e).__name__} saving metrics to {path}, {e}")


def write_wandb_log(metrics: Mapping[str, Any]):
    wandb = _import_wandb()

    if (step := metrics.get("step")) is not None:
        wandb.log(metrics, step=int(step), commit=True)
    else:
        wandb.log(metrics, commit=True)


@dataclass
class MetricsFileHook(DriverHook):
    path: Path | None = None

    def __post_init__(self):
        self.threadpool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="metrics_hook"
        )
        self.future = self.threadpool.submit(lambda: None)

    def on_boot(self, run: Run, run_dir: Path, config: Jsonable):
        self.path = run_dir / "metrics.jsonl"
        logger.info(f"Writing metrics to {self.path}")

    @log_elapsed_time
    def on_step(self, metrics: Mapping[str, Any]):
        if self.path is None:
            return
        self.future.result()
        self.future = self.threadpool.submit(write_metrics_file, path=self.path, metrics=metrics)

    def on_train_stop(self):
        self.future.result()


PROMETHEUS_DEFAULT_METRICS = (
    "step",
    "soft_step",
    "loss",
    "mfu",
    "step_time",
    "gpu_step_time",
    "next_data_batch_time",
    "elapsed_samples",
)


class PrometheusHook(DriverHook):
    def __init__(
        self,
        port: int = 9107,
        metric_names: tuple[str, ...] | list[str] = PROMETHEUS_DEFAULT_METRICS,
        labels: dict[str, str] | None = None,
        namespace: str | None = None,
    ):
        self.gauges: dict[str, Any] = {}
        self.port = port
        self.metrics_server = None
        self.metrics_thread = None
        self._metric_names = metric_names
        self._labels = labels or {}
        self._namespace = namespace

    def _create_gauges(self, run: Run):
        from prometheus_client import REGISTRY, Gauge

        if self.gauges:
            return
        labels = {**self._labels, "run_name": run.name}
        for metric_name in self._metric_names:
            prom_metric_name = (
                f"{self._namespace}:{metric_name}" if self._namespace is not None else metric_name
            )
            gauge = Gauge(
                name=prom_metric_name,
                documentation=prom_metric_name,
                labelnames=list(labels.keys()),
                registry=REGISTRY,
            ).labels(*labels.values())
            self.gauges[metric_name] = gauge

    def setup(self):
        import fastapi
        import uvicorn
        from prometheus_client import REGISTRY, make_asgi_app

        logger.info("Setting up Prometheus Client.")
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.cleanup()
        metrics_app = make_asgi_app(REGISTRY)

        async def route_metrics(request: fastapi.Request) -> fastapi.responses.Response:
            response = {}

            async def _send(asgi_dict):
                response.update(asgi_dict)

            await metrics_app(request.scope, request.receive, _send)

            raw_headers = []
            for header in response["headers"]:
                raw_headers.append(tuple(x.decode("utf8") for x in header))

            return fastapi.responses.Response(
                response["body"], response["status"], dict(raw_headers)
            )

        app = fastapi.FastAPI()
        app.get("/metrics")(route_metrics)
        self.metrics_server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.port,
                log_level="error",
                loop="asyncio",
            )
        )
        self.metrics_thread = Thread(target=self.metrics_server.run)

    def cleanup(self):
        if self.metrics_server is None:
            return
        logging.info("Cleaning up Prometheus Client.")
        self.metrics_server.handle_exit(sig=signal.SIGINT, frame=None)
        self.metrics_thread.join()

    def on_boot(self, run: Run, run_dir: Path, config: Jsonable):
        self._create_gauges(run)
        self.setup()
        self.metrics_thread.start()

    def on_step(self, metrics: Mapping[str, Any]):
        for name, gauge in self.gauges.items():
            if (value := metrics.get(name)) is not None:
                gauge.set(value)

    def on_train_stop(self):
        self.cleanup()


@dataclass
class WandbHook(DriverHook):
    project: str
    entity: str | None = None

    def __post_init__(self):
        self.threadpool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wandb_hook"
        )
        self.future = self.threadpool.submit(lambda: None)

    def on_boot(self, run: Run, run_dir: Path, config: Jsonable):
        wandb = _import_wandb()

        config = config.to_dict()
        config["launch_env"] = {
            "CHECKPOINT_DIR": launch_env.CHECKPOINT_DIR,
            "CLUSTER": launch_env.CLUSTER,
            "CLOUD": launch_env.CLOUD,
            "CONFIG": launch_env.CONFIG,
            "DATA_DIR": launch_env.DATA_DIR,
            "IS_LOCAL": launch_env.IS_LOCAL,
            "JOB_NAME": launch_env.JOB_NAME,
            "NUM_REPLICAS": launch_env.NUM_REPLICAS,
            "REPLICA_INDEX": launch_env.REPLICA_INDEX,
            "XAI_USER": launch_env.XAI_USER,
            "CPU_DOCKER_IMAGE": launch_env.CPU_DOCKER_IMAGE,
            "TRAIN_DOCKER_IMAGE": launch_env.TRAIN_DOCKER_IMAGE,
        }
        config["launch_env"]["commit_hash"] = run.commit_hash
        notes = f"Commit: {run.commit_hash}" if run.commit_hash else ""
        if files_note := launch_env.run_files_note(run_dir):
            notes = files_note + (f"\n{notes}" if notes else "")

        def _do_init(wandb_id: str, resume: str) -> None:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=run.name,
                settings=wandb.Settings(
                    start_method="fork",
                    _disable_stats=True,
                    quiet=True,
                    x_file_stream_max_bytes=100 * 1024 * 1024,
                    x_file_stream_max_line_bytes=100 * 1024 * 1024,
                ),
                config=config,
                id=wandb_id,
                resume=resume,
                notes=notes,
            )

        fallback_file = run_dir / ".wandb-fallback-id"
        if fallback_file.exists():
            wandb_id = fallback_file.read_text().strip() or run.run_id
        else:
            wandb_id = run.run_id

        try:
            _do_init(wandb_id, "allow")
        except Exception as e:
            msg = str(e).lower()
            is_collision = (
                type(e).__name__ == "ServerResponseError"
                or "is in use" in msg
                or "is currently active" in msg
                or "run id collision" in msg
            )
            if not is_collision:
                raise
            fallback_id = f"{run.run_id}-r{int(time.time())}"
            try:
                fallback_file.write_text(fallback_id)
            except OSError:
                logger.exception(
                    "failed to persist wandb fallback id to %s; "
                    "fallback id %r will not be reused across restarts",
                    fallback_file,
                    fallback_id,
                )
            logger.warning(
                "wandb id %r is in use on server; falling back to %r: %s",
                wandb_id,
                fallback_id,
                e,
            )
            _do_init(fallback_id, "never")

    @log_elapsed_time
    def on_step(self, metrics: Mapping[str, Any]):
        self.future.result()

        metrics = _filter_wandb_metrics(metrics)

        self.future = self.threadpool.submit(write_wandb_log, metrics=metrics)

    def on_train_rollback(self):
        wandb = _import_wandb()

        if wandb.run is None:
            return

        name = wandb.run.name
        config = wandb.run.config
        run_id = wandb.run.id
        settings = wandb.run.settings

        rewind_number = 0
        match = re.match(r"^(.+?)\((\d+)\)$", run_id)
        if match:
            run_id = match.group(1)
            rewind_number = int(match.group(2))

        rewind_number += 1
        run_id = f"{run_id}({rewind_number})"

        wandb.finish()
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            settings=settings,
            config=config,
            id=run_id,
            resume="allow",
        )

    def on_train_stop(self):
        self.future.result()

        wandb = _import_wandb()

        wandb.finish(quiet=True)


class ClickhouseHook(DriverHook):
    def __init__(self, driver_config: Any = None):
        self.run: Run | None = None
        self.threadpool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="clickhouse_hook"
        )
        self.client = None
        self.future = None
        self._driver_config = driver_config

        self.row = {
            "user": launch_env.XAI_USER,
            "job_name": launch_env.JOB_NAME,
            "run_name": None,
            "cluster": launch_env.CLUSTER,
            "run_id": None,
            "step": None,
            "substep": 0,
            "elapsed_samples": None,
            "metrics": {},
        }
        self.column_names = tuple(self.row.keys())

        self.ignored_keys = set(self.row)

        self._Error = RuntimeError

    def _insert(self, metrics):
        m = self.row["metrics"]
        m.clear()
        for key in metrics:
            if key in self.ignored_keys:
                continue
            value = np.array(metrics[key])
            if value.shape != ():
                logger.warning(
                    "Ignoring field %r with non-scalar shape %s (value: %s)",
                    key,
                    value.shape,
                    metrics[key],
                )
                self.ignored_keys.add(key)
                continue
            m[key] = value.item()

        self.row["step"] = metrics.get("step", 0)
        self.row["substep"] = metrics.get("substep", 0)
        self.row["elapsed_samples"] = metrics["elapsed_samples"]

        self.client.insert(
            "train_metrics_insert",
            (tuple(self.row.values()),),
            column_names=self.column_names,
        )

    def on_train_start(self, model_config: Any):
        del model_config
        import clickhouse_connect

        cfg = self._driver_config
        host = getattr(cfg, "clickhouse_host", _CLICKHOUSE_HOST)
        port = getattr(cfg, "clickhouse_port", None)
        secure = getattr(cfg, "clickhouse_secure", True)
        user = getattr(cfg, "clickhouse_user", "") or settings.CLICKHOUSE_USER
        password = getattr(cfg, "clickhouse_password", "") or os.environ.get(
            "RL2_CLICKHOUSE_PASSWORD", ""
        )
        database = getattr(cfg, "clickhouse_database", "default")

        logger.info(
            "ClickhouseHook: host=%s port=%s secure=%s user=%s db=%s",
            host,
            port,
            secure,
            user,
            database,
        )

        kwargs: dict[str, Any] = {
            "host": host,
            "secure": secure,
            "user": user,
            "password": password,
            "database": database,
            "port": port,
            "connect_timeout": 15,
            "send_receive_timeout": 60,
        }

        self.client = clickhouse_connect.get_client(**kwargs)

        self.future = self.threadpool.submit(lambda: None)
        self._Error = clickhouse_connect.driver.exceptions.Error

    def on_boot(self, run: Run, run_dir: Path, config: Jsonable):
        self.row["run_name"] = run.name
        self.row["run_id"] = run.run_id

        cluster = cluster_identity.get_cluster()
        if cluster and _CLICKHOUSE_RUN_URL:
            url = _CLICKHOUSE_RUN_URL.format(
                cluster=cluster, user=launch_env.XAI_USER, run_name=run.name
            )
            logger.info("Logging metrics. View at \033[1;35m%s\033[0m", url)

    @log_elapsed_time
    def on_step(self, metrics: Mapping[str, Any]):
        if metrics.get("elapsed_samples") is None:
            return

        try:
            self.future.result()
        except self._Error:
            logger.exception("ClickhouseHook.on_step failed")
        except Exception:
            self.future = None
            raise

        self.future = self.threadpool.submit(self._insert, metrics)

    @log_elapsed_time
    def on_train_stop(self):
        if self.future is not None:
            self.future.result()
            self.future = None


class CheckpointMonitoringState(Enum):
    RUN_CLEANUP = 0
    SHUTDOWN = 1


_should_keep_cache = {}


def _should_keep(checkpoint_dir: str, run_id: str, keep_every_n: int) -> bool:
    cache_key = (checkpoint_dir, run_id, keep_every_n)
    if cache_key in _should_keep_cache:
        return _should_keep_cache[cache_key]

    if keep_every_n <= 0:
        _should_keep_cache[cache_key] = False
        return False

    checkpoint_path = Path(os.path.join(checkpoint_dir, run_id))
    completion_data = read_metadata_file(checkpoint_path)
    if completion_data is None or completion_data[1] is None:
        logger.info(f"No checkpoint index found in {checkpoint_dir}. Keep it.")
        return True

    global_ckpt_index = completion_data[1]
    if keep_every_n > 0 and global_ckpt_index % keep_every_n == 0:
        logger.info(f"Keep candidate (ckpt_index: {global_ckpt_index}): {checkpoint_path}")
        _should_keep_cache[cache_key] = True
        return True

    _should_keep_cache[cache_key] = False
    return False


def _rename_for_deletion(path: str) -> str | None:
    logger.info(f"Renaming {path} for deletion")
    trash_path = os.path.join(os.path.dirname(path), f".trash_{os.path.basename(path)}")
    try:
        os.rename(path, trash_path)
        logger.info(f"Renamed {path} to {trash_path}")
        return trash_path
    except OSError as e:
        logger.warning(f"Could not rename {path} to trash: {e}")
        return None


def _rm_rf_batch(paths: list[str]):
    if not paths:
        return None
    logger.info(f"Running rm -rf on {paths}")
    return subprocess.Popen(
        ["rm", "-rf"] + paths,
        start_new_session=True,
    )


def _checkpoint_cleaner(checkpoint_dir: str, keep_last_n: int, keep_every_n: int, run_id: str):
    if not os.path.exists(checkpoint_dir):
        logger.debug(f"Checkpoint {checkpoint_dir} does not exist.")
        return

    entries = []
    leftover_trash = []
    for ckpt in os.scandir(checkpoint_dir):
        if ckpt.name.startswith(".trash_"):
            leftover_trash.append(ckpt.path)
            continue
        if not ckpt.name.startswith("elapsed"):
            continue
        checkpoint_path = Path(os.path.join(ckpt.path, run_id))
        if not metadata_file_exists(checkpoint_path):
            continue
        entries.append(ckpt)

    if leftover_trash:
        logger.info(
            f"Found {len(leftover_trash)} leftover trash dirs from previous run, cleaning up."
        )

    if len(entries) <= keep_last_n:
        trash_to_delete = leftover_trash
    else:
        entries.sort(key=lambda e: e.name)

        to_remove = []
        for i, candidate in enumerate(entries):
            if i >= len(entries) - keep_last_n:
                break
            if _should_keep(candidate.path, run_id, keep_every_n):
                continue
            to_remove.append(candidate)

        if not to_remove and not leftover_trash:
            return

        logger.info(f"About to remove ({len(to_remove)}) checkpoint(s) ({keep_last_n=})")

        trash_paths = []
        for entry in to_remove:
            run_path = os.path.join(entry.path, run_id)
            trash = _rename_for_deletion(run_path)
            if trash:
                trash_paths.append(trash)

        trash_to_delete = trash_paths + leftover_trash

    if not trash_to_delete:
        return

    t = Timer()
    batches = [
        trash_to_delete[i : i + _RM_BATCH_SIZE]
        for i in range(0, len(trash_to_delete), _RM_BATCH_SIZE)
    ]
    logger.info(f"Deleting {len(trash_to_delete)} trash dirs in {len(batches)} batch(es).")

    for batch in batches:
        proc = _rm_rf_batch(batch)
        while proc.poll() is None:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            except KeyboardInterrupt:
                logger.warning(
                    "Checkpoint cleanup interrupted — waiting for current batch to finish."
                )
                proc.wait()
                return

    logger.info(f"Removing {len(trash_to_delete)} old checkpoints took {t.elapsed()}.")


def _shutdown_checkpoint_worker(worker, tasks):
    if worker.is_alive():
        tasks.put_nowait(CheckpointMonitoringState.SHUTDOWN)
        worker.join()


class CheckpointMonitoringHook(DriverHook):
    def __init__(self):
        self.worker = None
        self._worker_shutdown = None
        self._is_shutdown = False

    def _checkpoint_monitoring(self, trainer_config: Jsonable, run_id: str):
        logger.info("Checkpoint monitoring started.")
        ckpt_config = trainer_config.checkpoint_config
        ckpt_dir = os.path.join(
            checkpoint_dir_or_default(ckpt_config.checkpoint_dir),
            trainer_config.name,
        )
        keep_last_n = ckpt_config.checkpoint_keep_last_n
        keep_every_n = ckpt_config.checkpoint_keep_every_nth

        while True:
            try:
                signal = self.tasks.get(timeout=10)
            except queue.Empty:
                if self._is_shutdown:
                    break
                continue

            if signal == CheckpointMonitoringState.RUN_CLEANUP:
                _checkpoint_cleaner(ckpt_dir, keep_last_n, keep_every_n, run_id)
            elif signal == CheckpointMonitoringState.SHUTDOWN:
                logger.info("Checkpoint monitoring thread got shutdown signal. Shutting down.")
                return

    def _is_worker_alive(self):
        return self.worker and self.worker.is_alive()

    def _shutdown_worker(self):
        if self._worker_shutdown is not None:
            self._worker_shutdown()
        self._is_shutdown = True

    def on_boot(self, run: Run, run_dir: Path, config: Jsonable):
        del run_dir
        try:
            self.trainer_config = config.trainer
        except AttributeError:
            self.trainer_config = config
        self.run_id = run.run_id
        self.worker = None
        self.tasks = queue.SimpleQueue()

    @log_elapsed_time
    def on_step(self, _: Mapping[str, Any]):
        if not self._is_worker_alive():
            self.worker = Thread(
                target=self._checkpoint_monitoring,
                name="CheckpointMonitoringThread",
                args=(self.trainer_config, self.run_id),
            )
            self.worker.start()
            self._worker_shutdown = weakref.finalize(
                self, _shutdown_checkpoint_worker, self.worker, self.tasks
            )

        if self.tasks.empty():
            self.tasks.put_nowait(CheckpointMonitoringState.RUN_CLEANUP)

    def on_train_stop(self):
        self._shutdown_worker()
        logger.info("Checkpoint monitoring hook closed.")
