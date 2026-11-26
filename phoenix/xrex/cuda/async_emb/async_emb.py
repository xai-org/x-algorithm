# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from xrex.cuda.async_emb.comm_utils import (
    get_context_id,
    get_flatten_replica_groups,
)

try:
    import xrex_cuda_kernels.async_emb_api as async_emb_api
except ModuleNotFoundError as e:
    raise ImportError(
        "no compiled async_emb binding: the xrex-cuda-kernels package is not "
        "installed in this environment."
        " use_async_emb requires the kernels."
    ) from e

try:
    jax.ffi.register_ffi_target(
        "xrex_async_emb_lookup_start",
        fn={
            "initialize": async_emb_api.lookup_start_init(),
            "execute": async_emb_api.lookup_start(),
        },
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "xrex_async_emb_lookup_done", fn=async_emb_api.lookup_done(), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "xrex_async_emb_stage_update", fn=async_emb_api.stage_update(), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "xrex_async_emb_rowwise_adagrad_update_start",
        fn={
            "initialize": async_emb_api.rowwise_adagrad_update_start_init(),
            "execute": async_emb_api.rowwise_adagrad_update_start(),
        },
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "xrex_async_emb_rowwise_adagrad_lazy_update_start",
        fn={
            "initialize": async_emb_api.rowwise_adagrad_lazy_update_start_init(),
            "execute": async_emb_api.rowwise_adagrad_lazy_update_start(),
        },
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "xrex_async_emb_rowwise_adagrad_update_done",
        fn=async_emb_api.rowwise_adagrad_update_done(),
        platform="CUDA",
    )
except AttributeError as e:
    raise ImportError(
        f"the compiled async_emb extension at {async_emb_api.__file__} does not "
        f"match these bindings (stale build?): {e}"
    ) from e


class AsyncEmbContextHandle(NamedTuple):
    context_id: int
    group_size: int
    flatten_replicas: tuple[int, ...]
    tokens_per_rank: int
    shard_width: int
    emb_width: int
    num_unique: int
    num_devices_per_node: int
    mesh: jax.sharding.Mesh
    table_axis: tuple[str, ...]
    data_axis: tuple[str, ...]

    def attrs(self) -> dict:
        return dict(
            context_id=self.context_id,
            group_size=self.group_size,
            flatten_replicas=np.asarray(self.flatten_replicas, dtype=np.int64),
            tokens_per_rank=self.tokens_per_rank,
            shard_width=self.shard_width,
            emb_width=self.emb_width,
            num_unique=self.num_unique,
            num_devices_per_node=self.num_devices_per_node,
        )


def make_context_handle(
    mesh: jax.sharding.Mesh,
    table_axis: tuple[str, ...],
    *,
    data_axis: tuple[str, ...],
    tokens_per_batch: int,
    emb_width: int,
    num_unique: int,
    num_devices_per_node: int,
) -> AsyncEmbContextHandle:
    for axis in dict.fromkeys((*table_axis, *data_axis)):
        if axis not in mesh.shape:
            raise ValueError(f"async_emb axis {axis!r} is not a mesh axis of {mesh}")
    group_size, flatten_replicas = get_flatten_replica_groups(mesh, table_axis)
    missing_axes = [axis for axis in table_axis if axis not in data_axis]
    if missing_axes:
        raise ValueError(
            f"async_emb requires token shards to vary across the communicator: "
            f"table_axis {missing_axes} missing from data_axis {data_axis}"
        )
    off_communicator_shards = math.prod(
        mesh.shape[axis] for axis in data_axis if axis not in table_axis
    )
    if off_communicator_shards != 1:
        raise ValueError(
            f"async_emb requires exactly one token shard per communicator rank: "
            f"data_axis {data_axis} shards tokens over {off_communicator_shards} "
            f"positions outside table_axis {table_axis}"
        )
    if tokens_per_batch % group_size != 0:
        raise ValueError(
            f"async_emb tokens_per_batch={tokens_per_batch} does not shard evenly "
            f"over the {group_size}-rank communicator"
        )
    if emb_width % group_size != 0:
        raise ValueError(
            f"async_emb emb_width={emb_width} does not shard evenly over the "
            f"{group_size}-rank communicator"
        )
    device_ids = [d.id for d in mesh.devices.flatten()]
    flatten_replicas = tuple(device_ids[pos] for pos in flatten_replicas)
    group_key = get_context_id(group_size, flatten_replicas)
    context_id = get_context_id(
        group_key,
        (
            tokens_per_batch // group_size,
            emb_width // group_size,
            emb_width,
            num_unique,
            num_devices_per_node,
        ),
    )
    return AsyncEmbContextHandle(
        context_id=context_id,
        group_size=group_size,
        flatten_replicas=flatten_replicas,
        tokens_per_rank=tokens_per_batch // group_size,
        shard_width=emb_width // group_size,
        emb_width=emb_width,
        num_unique=num_unique,
        num_devices_per_node=num_devices_per_node,
        mesh=mesh,
        table_axis=tuple(table_axis),
        data_axis=tuple(data_axis),
    )


def lookup_start(
    token_ids: jax.Array, table: jax.Array, gate: jax.Array, ctx: AsyncEmbContextHandle
):
    assert math.prod(token_ids.shape) == ctx.tokens_per_rank
    assert table.shape[1] == ctx.shard_width and table.dtype == jnp.bfloat16
    with jax.named_scope("async_emb.lookup_start"):
        outs = jax.ffi.ffi_call(
            "xrex_async_emb_lookup_start",
            [
                jax.ShapeDtypeStruct(table.shape, table.dtype),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
            input_output_aliases={1: 0},
        )(token_ids.reshape(-1).astype(jnp.int32), table, gate, **ctx.attrs())
    return outs[0], outs[1]


def lookup_done(pin: jax.Array, ctx: AsyncEmbContextHandle) -> jax.Array:
    with jax.named_scope("async_emb.lookup_done"):
        (embeddings,) = jax.ffi.ffi_call(
            "xrex_async_emb_lookup_done",
            [jax.ShapeDtypeStruct((ctx.tokens_per_rank, ctx.emb_width), jnp.bfloat16)],
        )(pin, context_id=ctx.context_id)
    return embeddings


def stage_update(
    grads: jax.Array,
    segment_ids: jax.Array,
    unique_tokens: jax.Array,
    pending: jax.Array,
    gate: jax.Array,
    ctx: AsyncEmbContextHandle,
) -> jax.Array:
    assert grads.shape == (ctx.tokens_per_rank, ctx.emb_width) and grads.dtype == jnp.bfloat16
    assert math.prod(segment_ids.shape) == ctx.tokens_per_rank
    assert math.prod(unique_tokens.shape) == ctx.num_unique
    with jax.named_scope("async_emb.stage_update"):
        (pin,) = jax.ffi.ffi_call(
            "xrex_async_emb_stage_update", [jax.ShapeDtypeStruct((1,), jnp.float32)]
        )(
            grads,
            segment_ids.reshape(-1).astype(jnp.int32),
            unique_tokens.reshape(-1).astype(jnp.int32),
            pending.astype(jnp.int32).reshape(1),
            gate,
            **ctx.attrs(),
        )
    return pin


def rowwise_adagrad_update_start(
    table: jax.Array,
    row_state: jax.Array,
    gate: jax.Array,
    ctx: AsyncEmbContextHandle,
    *,
    learning_rate: float,
    eps: float,
    decay_factor: float,
    weight_decay_factor: float = 1.0,
):
    with jax.named_scope("async_emb.rowwise_adagrad_update_start"):
        outs = jax.ffi.ffi_call(
            "xrex_async_emb_rowwise_adagrad_update_start",
            [
                jax.ShapeDtypeStruct(table.shape, table.dtype),
                jax.ShapeDtypeStruct(row_state.shape, row_state.dtype),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
            input_output_aliases={0: 0, 1: 1},
        )(
            table,
            row_state,
            gate,
            **ctx.attrs(),
            learning_rate=float(learning_rate),
            eps=float(eps),
            decay_factor=float(decay_factor),
            weight_decay_factor=float(weight_decay_factor),
        )
    return tuple(outs)


def rowwise_adagrad_lazy_update_start(
    table: jax.Array,
    row_state: jax.Array,
    last_step: jax.Array,
    step: jax.Array,
    gate: jax.Array,
    ctx: AsyncEmbContextHandle,
    *,
    learning_rate: float,
    eps: float,
    accum_decay_rate: float,
    weight_decay_rate: float,
    weight_decay_exempt_window: float = 0.0,
    stochastic_rounding: bool = False,
):
    assert last_step.shape == row_state.shape and last_step.dtype == jnp.int32
    with jax.named_scope("async_emb.rowwise_adagrad_lazy_update_start"):
        outs = jax.ffi.ffi_call(
            "xrex_async_emb_rowwise_adagrad_lazy_update_start",
            [
                jax.ShapeDtypeStruct(table.shape, table.dtype),
                jax.ShapeDtypeStruct(row_state.shape, row_state.dtype),
                jax.ShapeDtypeStruct(last_step.shape, jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
            input_output_aliases={0: 0, 1: 1, 2: 2},
        )(
            table,
            row_state,
            last_step,
            step.astype(jnp.int32).reshape(1),
            gate,
            **ctx.attrs(),
            learning_rate=float(learning_rate),
            eps=float(eps),
            accum_decay_rate=float(accum_decay_rate),
            weight_decay_rate=float(weight_decay_rate),
            weight_decay_exempt_window=float(weight_decay_exempt_window),
            stochastic_rounding=int(stochastic_rounding),
        )
    return tuple(outs)


def rowwise_adagrad_update_done(gate: jax.Array, ctx: AsyncEmbContextHandle):
    with jax.named_scope("async_emb.rowwise_adagrad_update_done"):
        outs = jax.ffi.ffi_call(
            "xrex_async_emb_rowwise_adagrad_update_done",
            [
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
        )(gate, context_id=ctx.context_id)
    return tuple(outs)
