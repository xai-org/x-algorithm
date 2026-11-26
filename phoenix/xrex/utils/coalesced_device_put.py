# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import logging
import math
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import numpy.typing as npt

rank_logger = logging.getLogger("rank")

_ARRAY_ALIGNMENT_BYTES = 128

_Layout = tuple[tuple[np.dtype, tuple[int, ...], int, int], ...]


def _packed_layout(leaves: list[Any]) -> tuple[_Layout, int]:
    layout, end = [], 0
    for leaf in leaves:
        dtype = jax.dtypes.canonicalize_dtype(leaf.dtype)
        start = end + -end % _ARRAY_ALIGNMENT_BYTES
        end = start + math.prod(leaf.shape) * dtype.itemsize
        layout.append((dtype, tuple(leaf.shape), start, end))
    return tuple(layout), end


def _split_packed_buffer(buffer: jax.Array, layout: _Layout) -> list[jax.Array]:
    arrays = []
    for dtype, shape, start, stop in layout:
        array_bytes = buffer[start:stop]
        if dtype == np.bool_:
            array = array_bytes.astype(bool)
        else:
            array = jax.lax.bitcast_convert_type(array_bytes.reshape(-1, dtype.itemsize), dtype)
        arrays.append(array.reshape(shape))
    return arrays


@functools.cache
def _compiled_split(layout: _Layout, nbytes: int, device: jax.Device) -> jax.stages.Compiled:
    buffer = jax.ShapeDtypeStruct(
        (nbytes,), np.uint8, sharding=jax.sharding.SingleDeviceSharding(device)
    )
    compiled = jax.jit(_split_packed_buffer, static_argnums=1).lower(buffer, layout).compile()
    rank_logger.info(
        "Compiled the coalesced transfer's split for %d arrays (%.1f MiB)",
        len(layout),
        nbytes / 2**20,
    )
    return compiled


def precompile_device_put_coalesced(tree: Any, device: jax.Device) -> None:
    _compiled_split(*_packed_layout(jax.tree.leaves(tree)), device)


def device_put_coalesced(
    tree: Any,
    device: jax.Device,
    host_empty: Callable[[tuple[int, ...], npt.DTypeLike], np.ndarray] = np.empty,
) -> Any:
    leaves, treedef = jax.tree.flatten(tree)
    assert all(isinstance(leaf, np.ndarray) for leaf in leaves), [type(leaf) for leaf in leaves]
    layout, nbytes = _packed_layout(leaves)
    host_buffer = host_empty((nbytes,), np.uint8)
    for leaf, (dtype, shape, start, stop) in zip(leaves, layout, strict=True):
        np.copyto(host_buffer[start:stop].view(dtype).reshape(shape), leaf, casting="unsafe")

    device_buffer = jax.device_put(host_buffer, device)
    device_buffer.block_until_ready()
    return jax.tree.unflatten(treedef, _compiled_split(layout, nbytes, device)(device_buffer))
