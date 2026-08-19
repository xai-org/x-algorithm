# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import json
import logging
import os
from collections import namedtuple
from collections.abc import Callable

import jax
from jax.experimental import multihost_utils
from opentelemetry.trace import Tracer

from xrex.models.model_utils import unwrap_tree
from xrex.utils import checkpointing
from xrex.utils import metrics as metrics_utils
from xrex.utils.aot import JittedOrCompiled
from xrex.utils.tracer import MockTracer

rank_logger = logging.getLogger("rank")


def install_offload_state(trainer) -> None:
    def d2h_copy(src, dst):
        dst = jax.device_put(src, trainer.host_sharding)
        return src, dst

    trainer.offload_state = JittedOrCompiled(
        jax.jit(
            d2h_copy,
            in_shardings=(trainer.state_sharding, trainer.host_sharding),
            out_shardings=(trainer.state_sharding, trainer.host_sharding),
            donate_argnums=(0, 1),
            keep_unused=True,
        )
    )


def register_offload_state(trainer) -> None:
    trainer.register_jit_function(
        trainer.offload_state, trainer.state_shape, trainer.state_shape, phase="early"
    )


def barrier(self, name: str) -> None:
    multihost_utils.sync_global_devices(name)


def get_checkpoint_path(self, ctx) -> str:
    return ctx.run.checkpoint_path(self.checkpoint_config.checkpoint_dir, self.elapsed_samples)


def save_checkpoint(
    self,
    tag: str | None = None,
    blocking: bool = False,
    callback: Callable[[int, str], None] | None = None,
    tracer: Tracer | None = None,
    checkpoint_index: int | None = None,
    checksum_dict: dict | None = None,
):
    if not tracer:
        tracer = MockTracer()
    if not hasattr(jax._src.config, "enable_memories"):
        _FakeState = namedtuple("_FakeState", ["value"])
        jax._src.config.enable_memories = _FakeState(value=True)

    path = self.get_checkpoint_path(self.ctx)

    with tracer.start_as_current_span("write_checksum"):
        if self.checkpoint_config.write_checksums:
            if checksum_dict is None:
                rank_logger.info("Computing checksums for state")
                checksum_dict = self.checksum_dict()
            if jax.process_index() == 0:
                os.makedirs(path, exist_ok=True)
                with open(f"{path}/checksums.{jax.process_index()}.json", "w") as f:
                    json.dump(checksum_dict, f)
                    rank_logger.info(
                        "Wrote %i checksums to %s",
                        len(checksum_dict["shard_checksums"]),
                        f.name,
                    )

    def _callback(
        elapsed_samples: int = self.elapsed_samples,
        elapsed_tokens: int = self.elapsed_tokens,
    ) -> None:
        if checkpoint_index is not None:
            self.current_ckpt_index = checkpoint_index
        else:
            self.current_ckpt_index += 1
        metrics_utils.addto("checkpoints_done", ((elapsed_samples, self.current_ckpt_index),), ())

        self.ctx.meta.record_completed_checkpoint(
            self.checkpoint_config.checkpoint_dir,
            elapsed_samples,
            self.current_ckpt_index,
            elapsed_tokens,
            self.checkpoint_config.checkpoint_ttl,
        )

        if callback is not None:
            callback(self.elapsed_samples, path)

    if self.checkpoint_config.save_method != "orbax":
        raise ValueError(
            f"save_method={self.checkpoint_config.save_method!r} is retired: the "
            '"tensorstore" save arm (xai_checkpointing background_save_ts plus its '
            "d2h offload staging) was removed because no fork configuration ever "
            "selected it — orbax is the only writer whose output the serving "
            "loader and repack tooling accept. Leave save_method unset (it "
            'defaults to "orbax"); the field survives only because stored '
            "run-configs serialize it and strict from_dict would reject a "
            "deleted key."
        )
    checkpointing.save_checkpoint(
        unwrap_tree(self.state),
        path=path,
        callback=_callback,
        tag=tag,
        blocking=blocking,
        timeout_secs=self.checkpoint_config.timeout_secs,
        chunked=self.checkpoint_config.checkpoint_chunked,
        compressed=self.checkpoint_config.checkpoint_compressed,
        chunk_byte_size=self.checkpoint_config.checkpoint_chunk_size_bytes,
        tracer=tracer,
    )


def install(trainer_cls) -> None:
    trainer_cls.barrier = barrier
    trainer_cls.get_checkpoint_path = get_checkpoint_path
    trainer_cls.save_checkpoint = save_checkpoint
