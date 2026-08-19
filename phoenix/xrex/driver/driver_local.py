# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrex.driver.driver import (
    NUM_PHYSICAL_DEVICES_PER_NODE,
    DriverConfig,
    MultiRunDriverState,
    log_metrics,
)
from xrex.train.trainer import TrainerContext, TrainerEvent
from xrex.utils import cluster

logger = logging.getLogger(__name__)


class LocalTrainerContext(TrainerContext):
    def _handle(self, event: TrainerEvent, data: Any):
        if event == TrainerEvent.STEP:
            metrics, _, _ = data
            log_metrics(metrics)
        else:
            logger.info("Event: %s with data=%s", event, data)


@dataclass
class LocalDriverConfig(DriverConfig):
    hostname: str = "localhost"
    ip_addr: str = "127.0.0.1"
    rank: int = 0
    world_size: int = 1

    debug: str | None = None

    run_dir: str = "/tmp/runs"

    dump_all_hlos: bool = False

    auto_restart: bool = True

    allow_unused: bool = False

    def _single_run(self, mr_state: MultiRunDriverState):
        ctx = mr_state.ctx
        if (
            ctx.run_config.parallel_config.num_devices_per_process != NUM_PHYSICAL_DEVICES_PER_NODE
            and not ctx.run_config.use_mock_gpu_client
        ):
            logger.warning(
                f"The local runtime only supports "
                f"num_devices_per_process={NUM_PHYSICAL_DEVICES_PER_NODE}. "
                f"This value is automatically changed to {NUM_PHYSICAL_DEVICES_PER_NODE}."
            )
            ctx.run_config.parallel_config.num_devices_per_process = NUM_PHYSICAL_DEVICES_PER_NODE

        if ctx.run_config.use_mock_gpu_client:
            self.world_size = ctx.run_config.parallel_config.num_devices()

        os.environ.update(ctx.env)
        rank = self.rank
        world_size = self.world_size

        run, ckpt = mr_state.run, mr_state.ckpt
        base_id, restart_count = run.run_id_parts()
        run_dir = (
            Path("/tmp")
            / "runs"
            / (cluster.get_user() or "local")
            / run.name
            / base_id
            / str(restart_count)
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        worker_ctx = LocalTrainerContext(
            rank=rank,
            world_size=world_size,
            hostnames=[self.hostname],
            ip_addrs=[self.ip_addr],
            config=ctx.run_config,
            run=run,
            run_dir=run_dir,
            checkpoint=ckpt,
            meta=ctx.meta,
        )
        ctx.train_fn(worker_ctx)
        worker_ctx.on_completed("Complete")
        self.signal_complete()
