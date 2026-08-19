# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import json
import logging
import os
import pathlib
import re
import time
from collections.abc import Callable
from typing import Any

import jax
import orbax.checkpoint as ocp
import tensorstore as ts
from jax.experimental.shard_map import shard_map

from xai_checkpointing import (
    common,
)
from xai_checkpointing.tree_util import has_subtree, tree_to_dict

logger = logging.getLogger("checkpointing")
rank_logger = logging.getLogger("rank")

PyTree = common.PyTree


def _read_into_shards(
    t: ts.TensorStore,
    array: jax.Array,
    mask: list[bool],
    restricted_domain: ts.IndexDomain | None = None,
):
    memory_kind = array.sharding.memory_kind
    is_cpu = all(d.platform == "cpu" for d in array.sharding.addressable_devices)
    assert memory_kind == "pinned_host" or is_cpu, (
        f"expected pinned_host memory, got {memory_kind!r}"
    )
    devices_indices_map = array.sharding.addressable_devices_indices_map(array.shape)

    for shard, should_load in zip(array.addressable_shards, mask, strict=True):
        if not should_load:
            continue

        index = devices_indices_map[shard.device]
        shard_domain = ts.IndexTransform(input_shape=array.shape)[index].domain

        domain = shard_domain
        if restricted_domain:
            domain = restricted_domain.intersect(shard_domain)

        src = t[domain]

        a = common._unsafe_jax2np(shard.data)
        assert (
            shard.data.addressable_data(0).unsafe_buffer_pointer()
            == a.__array_interface__["data"][0]
        )

        dest = ts.array(a)[ts.d[:].translate_to[shard_domain.origin]][domain]
        yield dest.write(src).commit


def load_checkpoint(
    path: str,
    host_state: PyTree[jax.Array],
    load_mask: PyTree[jax.Array] | None,
    rename: Callable[[str], str] | None = None,
    domains: dict[str, Any] | None = None,
    tag: str | None = None,
    timeout: float = 900.0,
):
    if tag is None:
        tag = "orbax-ckpt"

    host_state = tree_to_dict(host_state)
    load_mask = tree_to_dict(load_mask)
    domains = tree_to_dict(domains)

    for name, domain in domains.items():
        assert host_state.get(name) is not None, f"Cannot restrict domain of skipped tensor {name}"
        if isinstance(domain, dict):
            domains[name] = ts.IndexDomain(json=domain)
        elif isinstance(domain, ts.DimExpression):
            domains[name] = ts.IndexDomain(shape=host_state[name].shape)[domain]
        elif not isinstance(domain, ts.IndexDomain):
            raise ValueError(f"Unknown domain: {domain} for {name!r}")

    start = time.time()
    rank_logger.info("Restoring checkpoint from %s", path)

    path = pathlib.Path(path) / tag
    metadata = ocp.StandardCheckpointer().metadata(path)
    with (path / "_METADATA").open() as f:
        use_zarr3 = json.load(f)["use_zarr3"]

    ts_context = ts.Context(
        {
            "file_io_concurrency": {"limit": 128},
            "cache_pool#ocdbt": {"total_bytes_limit": 100000000},
        }
    )

    unloaded_state = host_state.copy()

    futures = {}
    for _i, checkpoint_name in enumerate(tree_to_dict(metadata, keep_none=False).keys()):
        name = checkpoint_name
        if rename is not None:
            name = rename(checkpoint_name)

        if host_state.get(name) is None:
            if not has_subtree(name, host_state):
                rank_logger.warning(
                    "Not loading %r from checkpoint because it's not in the initialized state", name
                )
            continue

        if not load_mask:
            mask = [True for _ in host_state[name].addressable_shards]
        else:
            mask = [shard.data.item() for shard in load_mask[name].addressable_shards]
        if any(mask):
            info = ocp.type_handlers.ParamInfo(
                name=checkpoint_name,
                path=path / checkpoint_name,
                parent_dir=path,
                is_ocdbt_checkpoint=True,
                use_zarr3=use_zarr3,
            )
            tspec = ocp.type_handlers.get_json_tspec_read(info, use_ocdbt=True)
            t = ts.open(ts.Spec(tspec), open=True, context=ts_context).result()
            if domains.get(name) is None and tuple(t.shape) != tuple(host_state[name].shape):
                raise ValueError(
                    f"Tensor {name!r}: checkpoint has shape {tuple(t.shape)}, "
                    f"but initialized state has shape {tuple(host_state[name].shape)}. "
                    f"Use 'no_loading' to skip this tensor or 'domains' to load a partial slice."
                )
            for s, future in enumerate(
                _read_into_shards(t, host_state[name], mask, domains.get(name))
            ):
                futures[future] = (checkpoint_name, name, s)

        del unloaded_state[name]

    for future in common._ready(list(futures), timeout=timeout):
        try:
            future.result()
        except Exception as e:
            logger.exception(e)
            checkpoint_name, name, _ = futures.pop(future, None)
            tensor = host_state[name]
            err = f"Checkpoint error from loading {path}. Error loading from {name} into {tensor.shape=}, {tensor.dtype=}."
            if checkpoint_name != name:
                err = f"{err} (in checkpoint: {checkpoint_name})"
            logger.error(err)
            raise

        checkpoint_name, name, _ = futures.pop(future, None)
        if checkpoint_name != name:
            rank_logger.debug("Loaded %s (name in checkpoint: %s)", name, checkpoint_name)
        else:
            rank_logger.debug("Loaded %s", name)

    rank_logger.info("Loading checkpoint took %.2f sec", time.time() - start)


def broadcast_replicated(
    state: PyTree[jax.Array],
    axes: PyTree[jax.sharding.PartitionSpec],
    srcs: PyTree[jax.Array],
    mesh: jax.sharding.Mesh,
):
    shardings = jax.tree.map(lambda t: t.sharding, state)
    pspecs = jax.tree.map(lambda s: s.spec, shardings)
    donate_argnums = () if os.getenv("XAI_CHECKPOINT_BROADCAST_DONATE", "1") == "0" else (0,)

    @functools.partial(
        jax.jit, donate_argnums=donate_argnums, in_shardings=(shardings,), out_shardings=shardings
    )
    @functools.partial(shard_map, mesh=mesh, in_specs=(pspecs,), out_specs=pspecs, check_rep=False)
    def broadcast(tree):
        return jax.tree.map(common.pbroadcast, tree, axes, srcs)

    return jax.block_until_ready(broadcast(state))


def rename_tensor(name: str, rename_patterns: dict[str, str]):
    for pattern, repl in rename_patterns.items():
        result, count = re.subn(pattern, repl, name, flags=re.X)
        if count:
            return result
    return name
