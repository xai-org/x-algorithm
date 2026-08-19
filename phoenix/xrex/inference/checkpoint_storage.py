# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import json
import logging
import os
import time

import numpy as np
import xai_recsys_engine
from xai_checkpointing.tree_util import tree_to_dict

from xrex.cuda import adler32
from xrex.data.parquet_recsys import PhoenixDataset
from xrex.models.model_utils import unwrap_tree
from xrex.models.recsys_model import RecsysAggregatedModelConfig
from xrex.train.trainer import RecsysTwoTowerModelConfig, TrainerContext
from xrex.train.trainer_recsys import Store

logger = logging.getLogger(__name__)

_STORAGE_ENGINE_FNS = ("find_latest_storage_checkpoint", "load_arrays_from_storage")


def _require_storage_engine() -> None:
    missing = [fn for fn in _STORAGE_ENGINE_FNS if not hasattr(xai_recsys_engine, fn)]
    if missing:
        raise RuntimeError(
            f"This xai_recsys_engine build has no object-storage loaders "
            f"(missing {', '.join(missing)}), so an s3:// or gs:// checkpoint "
            f"cannot be read. Serve from a filesystem checkpoint path, or build "
            f"the engine with its storage support enabled."
        )


def detect_store(runner) -> Store:
    checkpoint_path = runner.checkpoint_config.checkpoint_dir
    store = Store.from_path(checkpoint_path) if checkpoint_path else Store.FS
    for path in (runner.load, runner.store_load):
        if path and Store.from_path(path) != Store.FS:
            store = Store.from_path(path)
            break
    return store


def maybe_load_from_storage(runner, ctx, store: Store, tag=None):
    logger.info(
        "Falling back to %s for checkpoint load",
        "storage" if store != Store.FS else "NFS",
    )

    if store != Store.FS:
        t_storage = time.time()
        res = load_checkpoint_from_storage(runner, ctx)
        if runner.metrics_publisher is not None:
            runner.metrics_publisher.checkpoint_reload_step_seconds.labels(
                step="storage_load"
            ).observe(time.time() - t_storage)
            runner.metrics_publisher.checkpoint_reload_method.labels(method="storage").inc()
            runner.metrics_publisher.checkpoint_reload_success_total.labels(method="storage").inc()
            storage_ts = time.time()
            runner.last_checkpoint_created_timestamp = storage_ts
            runner.metrics_publisher.model_checkpoint_timestamp.set(storage_ts)
        return res
    return None


def resolve_checkpoint_arg(runner, checkpoint_path) -> bool:
    is_storage_path = bool(checkpoint_path) and Store.from_path(checkpoint_path) != Store.FS
    if is_storage_path:
        runner.store_load = checkpoint_path
    else:
        runner.load = checkpoint_path
    return is_storage_path


def append_storage_overrides(checkpoint_path: str, overrides: list) -> None:
    parts = checkpoint_path.rstrip("/").split("/")
    checkpoint_dir = "/".join(parts[:-3])
    overrides.append(f"checkpoint_dir={checkpoint_dir}")


def load_checkpoint_from_storage(self, ctx: TrainerContext) -> tuple[bool, int, int]:
    _require_storage_engine()
    assert self.h2d_state is not None
    assert isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
    assert isinstance(self.dataset, PhoenixDataset)

    start_all = time.time()

    host_state, tensor_bytes = self._get_host_state_tensor_bytes()

    emb_table_total_size = (
        self.padded_input_vocab_size
        * self.model_config.emb_table_width
        * np.dtype(self.model_config.embedding_dtype).itemsize
    )

    storage_load_path: str | None = None
    for path in (self.load, self.store_load):
        if path and Store.from_path(path) != Store.FS:
            storage_load_path = path
            break

    if storage_load_path is None:
        raise ValueError(
            "No valid storage load path found. "
            "Ensure either 'load' or 'store_load' is set to a valid S3/O2 path."
        )

    path_stripped = storage_load_path.rstrip("/")
    parts = path_stripped.split("/")
    fixed_prefix = f"{parts[-2]}/{parts[-1]}"
    base_path = "/".join(parts[:-2])

    emb_table_name_size = [("emb_table", emb_table_total_size)]
    result = xai_recsys_engine.find_latest_storage_checkpoint(
        base_path,
        emb_table_name_size,
        fixed_prefix,
    )

    if result is None:
        raise ValueError(
            f"load='{storage_load_path}' was specified but no valid checkpoint was found. "
            "Verify that the path exists and contains a valid checkpoint."
        )

    p, num_emb_shards = result
    full_path = f"{base_path}/{p}"

    pe_key = "post_embeddings.embeddings"
    pe_buf: np.ndarray | None = None
    pe_total_size = 0
    num_pe_shards = 0
    if pe_key in host_state and host_state[pe_key] is not None and pe_key not in tensor_bytes:
        pe_value = host_state[pe_key]
        pe_total_size = int(np.prod(pe_value.shape)) * pe_value.dtype.itemsize
        pe_result = xai_recsys_engine.find_latest_storage_checkpoint(
            base_path,
            [(pe_key, pe_total_size)],
            fixed_prefix,
        )
        if pe_result is not None:
            _, num_pe_shards = pe_result
            pe_buf = np.empty(pe_total_size, dtype=np.uint8)

    all_name_tensor_pairs = []

    checksums_arr = np.zeros(102400, dtype=np.uint8)
    all_name_tensor_pairs.append(("checksums.0.json", checksums_arr))

    for key, arr_bytes in tensor_bytes.items():
        value = host_state[key]
        suffix = "/0" * value.ndim
        all_name_tensor_pairs.append((f"{key}/c{suffix}", arr_bytes))

    emb_table = self.h2d_state.emb_table
    emb_table_shape = (
        self.padded_input_vocab_size,
        self.model_config.emb_table_width,
    )
    row_size = emb_table_shape[1] * np.dtype(self.model_config.embedding_dtype).itemsize
    total_rows = emb_table_shape[0]
    rows_per_shard = total_rows // num_emb_shards

    for shard_idx in range(num_emb_shards):
        start_row = shard_idx * rows_per_shard
        start_byte = start_row * row_size
        shard_bytes = rows_per_shard * row_size
        shard_arr = np.ndarray(
            shape=shard_bytes,
            dtype=np.uint8,
            buffer=emb_table,
            offset=start_byte,
        )
        all_name_tensor_pairs.append((f"emb_table/c/{shard_idx}/0", shard_arr))

    if pe_buf is not None and num_pe_shards > 0:
        pe_value = host_state[pe_key]
        pe_row_bytes = int(np.prod(pe_value.shape[1:])) * pe_value.dtype.itemsize
        pe_rows = pe_value.shape[0]
        pe_rows_per_shard = pe_rows // num_pe_shards
        for shard_idx in range(num_pe_shards):
            start_byte = shard_idx * pe_rows_per_shard * pe_row_bytes
            shard_bytes = pe_rows_per_shard * pe_row_bytes
            shard_arr = np.ndarray(
                shape=shard_bytes,
                dtype=np.uint8,
                buffer=pe_buf,
                offset=start_byte,
            )
            all_name_tensor_pairs.append((f"{pe_key}/c/{shard_idx}/0", shard_arr))

    max_concurrent = int(os.environ.get("S3_MAX_CONCURRENT_DOWNLOADS", "32"))
    start_load = time.time()
    if self.metrics_publisher is not None:
        self.metrics_publisher.checkpoint_reload_step_seconds.labels(
            step="storage_discovery"
        ).observe(start_load - start_all)
    xai_recsys_engine.load_arrays_from_storage(full_path, all_name_tensor_pairs, max_concurrent)

    checksums_str = bytes(checksums_arr).rstrip(b"\x00").decode("utf-8")
    checksums_dict = json.loads(checksums_str)["global_checksums"]

    start_reshard = time.time()

    for key, arr_bytes in tensor_bytes.items():
        self._broadcast_tensor_to_all_shards(host_state[key], arr_bytes)

    if pe_buf is not None:
        self._scatter_contiguous_to_shards(host_state[pe_key], pe_buf)

    self.host_state, self.state = self.reload_state(self.host_state, self.state)

    start_checksum = time.time()
    checksum_message = ""
    if self.checkpoint_config.verify_checksums:
        checksum_dict = {}
        row_sharded_keys: set[str] = set()
        state = tree_to_dict(unwrap_tree(self.state).purge_opt_state())
        for key in state:
            if state[key] is not None:
                if key == pe_key and pe_buf is not None:
                    row_sharded_keys.add(key)
                    checksum_dict[key] = xai_recsys_engine.adler32_parallel(pe_buf)
                else:
                    checksum_dict[key] = adler32.adler32(state[key]).item()
        emb_table_bytes = np.ndarray(emb_table.nbytes, dtype=np.uint8, buffer=emb_table)
        checksum_dict["emb_table"] = xai_recsys_engine.adler32_parallel(emb_table_bytes)

        unknown_checksums = []
        count = 0
        keys_to_check = [k for k in host_state.keys() if host_state[k] is not None] + ["emb_table"]
        for key in keys_to_check:
            if key not in checksums_dict:
                unknown_checksums.append(key)
            elif checksums_dict[key] != checksum_dict.get(key):
                raise ValueError(f"Checksum mismatch on {key}")
            else:
                count += 1
        if unknown_checksums:
            logger.warning(
                "The following keys were not found in the checkpoint checksums file "
                "and could not be verified: %s",
                unknown_checksums,
            )
        checksum_message = f"({count}/{len(keys_to_check)} checksums match) "

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
    logger.info(
        "Loaded checkpoint %s%s; discovery %s, load %s, reshard %s, checksum %s, total %s",
        checksum_message,
        full_path,
        *durations,
    )
    if self.metrics_publisher is not None:
        storage_load_duration = max(start_reshard - start_load, 1e-9)
        self.metrics_publisher.checkpoint_reload_step_seconds.labels(step="storage_load").observe(
            storage_load_duration
        )
        self.metrics_publisher.checkpoint_reload_step_seconds.labels(step="reshard").observe(
            start_checksum - start_reshard
        )
        self.metrics_publisher.checkpoint_reload_step_seconds.labels(step="checksum").observe(
            t - start_checksum
        )
        self.metrics_publisher.checkpoint_reload_method.labels(method="storage").inc()
        total_storage_bytes = (
            sum(arr.nbytes for arr in tensor_bytes.values())
            + emb_table.nbytes
            + (pe_buf.nbytes if pe_buf is not None else 0)
        )
        self.metrics_publisher.checkpoint_reload_bytes.labels(step="storage_load").set(
            total_storage_bytes
        )
        self.metrics_publisher.checkpoint_reload_throughput_bytes_per_second.labels(
            step="storage_load"
        ).set(total_storage_bytes / storage_load_duration)

    elapsed_samples = int(full_path.split("/")[-2].split("_")[-1])
    return True, elapsed_samples, 0
