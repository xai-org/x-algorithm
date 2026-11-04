# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import contextlib
import ctypes
import fcntl
import gc
import logging
import os
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils
from opentelemetry.trace import Tracer

from xrex.utils.tracer import MockTracer

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

_CHECKPOINTER = None
_CHECKPOINTER_SAVE_CONCURRENT_GB: int | None = None

_SAVE_NODE_SERIALIZE_ENV = "XAI_SAVE_NODE_SERIALIZE"
_SAVE_NODE_LOCK_FILE_ENV = "XAI_SAVE_NODE_LOCK_FILE"


def _save_node_serialize_enabled() -> bool:
    return os.getenv(_SAVE_NODE_SERIALIZE_ENV, "0").lower() not in ("0", "", "false")


def _save_node_lock_path() -> str:
    path = os.getenv(_SAVE_NODE_LOCK_FILE_ENV)
    if path:
        return path
    if os.path.isdir("/dev/shm"):
        return "/dev/shm/xai_save_node_lock"
    return "/tmp/xai_save_node_lock"


class _NodeBatchLock:
    def __init__(self, path: str):
        self._path = path
        self._fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)

    def __enter__(self):
        t0 = time.time()
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 1.0:
            rank_logger.info(
                "save node-serialize: waited %.1fs for node lock %s", waited, self._path
            )
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self):
        try:
            os.close(self._fd)
        except OSError:
            pass


def _release_batch_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class _AsyncCheckpointer(ocp.AsyncCheckpointer):
    def wait_until_finished(self):
        super().wait_until_finished()
        _orbax_encrypted = sys.modules.get("xai_checkpointing.orbax_encrypted")
        if _orbax_encrypted is not None:
            _orbax_encrypted.wait_until_finished()


def get_checkpointer(timeout_secs=900, save_concurrent_gb: int | None = None):
    global _CHECKPOINTER, _CHECKPOINTER_SAVE_CONCURRENT_GB
    if _CHECKPOINTER is None:
        handler_kwargs: dict[str, Any] = {"use_zarr3": True, "restore_concurrent_gb": 1}
        if save_concurrent_gb is not None:
            handler_kwargs["save_concurrent_gb"] = save_concurrent_gb
            handler_kwargs["restore_concurrent_gb"] = save_concurrent_gb
        _CHECKPOINTER_SAVE_CONCURRENT_GB = save_concurrent_gb
        rank_logger.info(
            "Creating AsyncCheckpointer: PyTreeCheckpointHandler(use_zarr3=True, "
            "save_concurrent_gb=%s, restore_concurrent_gb=%s) "
            "[None => Orbax write-limiter default 96GB; D2H still all-at-once "
            "unless save_checkpoint registers ThrottledD2HArrayHandler]",
            save_concurrent_gb,
            save_concurrent_gb,
        )
        _CHECKPOINTER = _AsyncCheckpointer(
            ocp.PyTreeCheckpointHandler(**handler_kwargs), timeout_secs
        )
        if not hasattr(_CHECKPOINTER, "_post_finalization_callback"):
            raise RuntimeError("Orbax version is too old")
    elif save_concurrent_gb is not None and _CHECKPOINTER_SAVE_CONCURRENT_GB != save_concurrent_gb:
        rank_logger.warning(
            "get_checkpointer(save_concurrent_gb=%s) ignored; checkpointer already "
            "created with save_concurrent_gb=%s. D2H throttle (if enabled) still "
            "uses the value passed to save_checkpoint.",
            save_concurrent_gb,
            _CHECKPOINTER_SAVE_CONCURRENT_GB,
        )
    return _CHECKPOINTER


class NoCompressionArrayHandler(ocp.type_handlers.ArrayHandler):
    def _get_json_tspec_write(self, *args, **kwargs):
        spec = super()._get_json_tspec_write(*args, **kwargs)
        for codec in spec["metadata"]["codecs"]:
            cfg = codec["configuration"]
            cfg["codecs"] = [c for c in cfg["codecs"] if c["name"] != "zstd"]
        return spec


class ThrottledD2HArrayHandler(ocp.type_handlers.ArrayHandler):
    def __init__(self, concurrent_bytes: int, **kwargs):
        if concurrent_bytes <= 0:
            raise ValueError(f"concurrent_bytes must be > 0, got {concurrent_bytes}")
        super().__init__(**kwargs)
        self._concurrent_bytes = concurrent_bytes

    def _addressable_nbytes(self, arr: jax.Array) -> int:
        total = 0
        for shard in arr.addressable_shards:
            if self._replica_id is None or shard.replica_id == self._replica_id:
                total += int(shard.data.nbytes)
        return total

    async def serialize(self, values, infos, args=None):
        args = args or [ocp.SaveArgs()] * len(values)
        if not values:
            return []

        batches: list[tuple[list, list, list]] = []
        cur_v: list = []
        cur_i: list = []
        cur_a: list = []
        cur_b = 0
        max_leaf = 0
        total_b = 0
        for v, info, arg in zip(values, infos, args):
            nb = self._addressable_nbytes(v)
            max_leaf = max(max_leaf, nb)
            total_b += nb
            if cur_v and cur_b + nb > self._concurrent_bytes:
                batches.append((cur_v, cur_i, cur_a))
                cur_v, cur_i, cur_a, cur_b = [], [], [], 0
            cur_v.append(v)
            cur_i.append(info)
            cur_a.append(arg)
            cur_b += nb
        if cur_v:
            batches.append((cur_v, cur_i, cur_a))

        rank_logger.info(
            "ThrottledD2HArrayHandler: save_concurrent_bytes=%.2fGiB, "
            "addressable_total=%.2fGiB, max_leaf=%.2fGiB, num_leaves=%d, "
            "num_batches=%d (D2H+write one batch at a time)",
            self._concurrent_bytes / (1 << 30),
            total_b / (1 << 30),
            max_leaf / (1 << 30),
            len(values),
            len(batches),
        )
        if max_leaf > self._concurrent_bytes:
            rank_logger.warning(
                "ThrottledD2HArrayHandler: largest leaf %.2fGiB exceeds "
                "save_concurrent_bytes %.2fGiB; that leaf still D2Hs in one shot. "
                "Raise checkpoint_config.save_concurrent_gb if write limiter errors.",
                max_leaf / (1 << 30),
                self._concurrent_bytes / (1 << 30),
            )

        node_lock: _NodeBatchLock | None = None
        if _save_node_serialize_enabled():
            node_lock = _NodeBatchLock(_save_node_lock_path())
            rank_logger.info(
                "save node-serialize ACTIVE (flock per batch) lock=%s pid=%d num_batches=%d",
                node_lock._path,
                os.getpid(),
                len(batches),
            )

        try:
            for batch_idx, (bv, bi, ba) in enumerate(batches):
                with node_lock if node_lock is not None else contextlib.nullcontext():
                    batch_bytes = sum(self._addressable_nbytes(v) for v in bv)
                    rank_logger.info(
                        "ThrottledD2HArrayHandler: batch %d/%d leaves=%d bytes=%.2fGiB",
                        batch_idx + 1,
                        len(batches),
                        len(bv),
                        batch_bytes / (1 << 30),
                    )
                    futs = await super().serialize(bv, bi, ba)
                    for fut in futs:
                        fut.result()
                if node_lock is not None:
                    _release_batch_memory()
        finally:
            if node_lock is not None:
                node_lock.close()
        return []


class ThrottledNoCompressionArrayHandler(ThrottledD2HArrayHandler, NoCompressionArrayHandler):
    pass


def save_checkpoint(
    state,
    path,
    callback=None,
    tag=None,
    blocking=False,
    timeout_secs=900,
    chunked=True,
    compressed=True,
    chunk_byte_size: int = 1024 * 1024 * 4,
    ocdbt_target_data_file_size: int | None = None,
    tracer: Tracer | None = None,
    save_concurrent_gb: int | None = None,
    kms_client: object | None = None,
    encryption_chunk_size: int = 8 * 1024 * 1024,
):
    if not tracer:
        tracer = MockTracer()

    if tag is None:
        tag = "orbax-ckpt"
    os.makedirs(path, exist_ok=True)

    if kms_client is not None:
        from xai_checkpointing import orbax_encrypted

        rank_logger.info("Saving ENCRYPTED orbax checkpoint (xai_encrypted driver) to %s", path)
        orbax_encrypted._require_array_leaves(state)
        if save_concurrent_gb is not None:
            array_handler_cls = (
                ThrottledD2HArrayHandler if compressed else ThrottledNoCompressionArrayHandler
            )
        elif not compressed:
            array_handler_cls = NoCompressionArrayHandler
        else:
            array_handler_cls = ocp.type_handlers.ArrayHandler
        checkpointer = orbax_encrypted.get_encrypted_checkpointer(
            kms_client, encryption_chunk_size, timeout_secs, save_concurrent_gb, array_handler_cls
        )
    else:
        checkpointer = get_checkpointer(timeout_secs, save_concurrent_gb=save_concurrent_gb)

    with tracer.start_as_current_span("wait_for_previous_checkpoint"):
        checkpointer.wait_until_finished()

    dest = os.path.join(path, tag)
    if os.path.exists(dest):
        rank_logger.warning(
            "Checkpoint destination %s already exists; treating as a "
            "previously-committed save and skipping orbax.save (idempotent)",
            dest,
        )
        if callback and jax.process_index() == 0:
            callback()
        if blocking:
            multihost_utils.sync_global_devices("blocking-checkpoint")
        return

    if jax.devices()[0].platform == "cpu":
        mesh = state.step.sharding.mesh
        pspecs = jax.tree.map(lambda a: a.sharding.spec, state)
        state = multihost_utils.global_array_to_host_local_array(state, mesh, pspecs)
        state = jax.tree.map(lambda a: jnp.array(np.array(jax.device_get(a)).copy()), state)
        state = multihost_utils.host_local_array_to_global_array(state, mesh, pspecs)

    original_handler = ocp.type_handlers.get_type_handler(jax.Array)
    if kms_client is None:
        if save_concurrent_gb is not None:
            concurrent_bytes = int(save_concurrent_gb) * 10**9
            if compressed:
                handler = ThrottledD2HArrayHandler(concurrent_bytes)
            else:
                handler = ThrottledNoCompressionArrayHandler(concurrent_bytes)
            ocp.type_handlers.register_type_handler(jax.Array, handler, override=True)
            rank_logger.info(
                "save_checkpoint: ThrottledD2HArrayHandler ACTIVE "
                "save_concurrent_gb=%s compressed=%s path=%s",
                save_concurrent_gb,
                compressed,
                path,
            )
        elif not compressed:
            ocp.type_handlers.register_type_handler(
                jax.Array, NoCompressionArrayHandler(), override=True
            )
            rank_logger.info(
                "save_checkpoint: NoCompressionArrayHandler active (no D2H throttle) path=%s",
                path,
            )
        else:
            rank_logger.info(
                "save_checkpoint: stock Orbax ArrayHandler (no D2H throttle; "
                "full addressable state staged to host at once) path=%s",
                path,
            )

    def _callback():
        ocp.type_handlers.register_type_handler(jax.Array, original_handler, override=True)
        rank_logger.info("Finished writing checkpoint to %s", path)
        if not blocking and callback:
            callback()

    checkpointer._post_finalization_callback = _callback

    chunk_byte_size = chunk_byte_size if chunked else None
    with tracer.start_as_current_span(f"save_checkpoint_blocking_{blocking}"):
        save_args = jax.tree.map(
            lambda _: ocp.SaveArgs(chunk_byte_size=chunk_byte_size),
            state,
        )
        if kms_client is not None:
            checkpointer.save(
                dest,
                args=ocp.args.PyTreeSave(
                    state,
                    save_args=save_args,
                    ocdbt_target_data_file_size=ocdbt_target_data_file_size,
                ),
            )
        else:
            checkpointer.save(
                dest,
                args=ocp.args.PyTreeSave(
                    state,
                    save_args=save_args,
                    ocdbt_target_data_file_size=ocdbt_target_data_file_size,
                ),
            )

        rank_logger.info(
            "Started writing checkpoint to %s (save_concurrent_gb=%s, blocking=%s)",
            path,
            save_concurrent_gb,
            blocking,
        )

        if blocking:
            rank_logger.info("blocking wait until finished: waiting for Orbax write at %s", path)
            t0 = time.time()
            checkpointer.wait_until_finished()
            rank_logger.info(
                "blocking wait until finished: Orbax write done in %.2fs at %s",
                time.time() - t0,
                path,
            )
            if callback and jax.process_index() == 0:
                callback()
            multihost_utils.sync_global_devices("blocking-checkpoint")


def wait_until_finished():
    if _CHECKPOINTER is not None:
        _CHECKPOINTER.wait_until_finished()
    _orbax_encrypted = sys.modules.get("xai_checkpointing.orbax_encrypted")
    if _orbax_encrypted is not None:
        _orbax_encrypted.wait_until_finished()
