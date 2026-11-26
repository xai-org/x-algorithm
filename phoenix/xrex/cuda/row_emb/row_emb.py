# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from xrex.cuda.row_emb.comm_utils import (
    get_context_id,
    get_flatten_replica_groups,
)

try:
    import xrex_cuda_kernels.row_emb_api as row_emb_api
except ModuleNotFoundError as e:
    raise ImportError(
        "no compiled row_emb binding: the xrex-cuda-kernels package in this "
        "environment does not carry row_emb_api."
        " use_row_emb requires the kernels."
    ) from e

try:
    jax.ffi.register_ffi_target(
        "xrex_row_emb_lookup_start",
        fn={
            "initialize": row_emb_api.lookup_start_init(),
            "execute": row_emb_api.lookup_start(),
        },
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "xrex_row_emb_lookup_done", fn=row_emb_api.lookup_done(), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "xrex_row_emb_stage_update", fn=row_emb_api.stage_update(), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "xrex_row_emb_rowwise_adagrad_update_start",
        fn={
            "initialize": row_emb_api.rowwise_adagrad_update_start_init(),
            "execute": row_emb_api.rowwise_adagrad_update_start(),
        },
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "xrex_row_emb_rowwise_adagrad_lazy_update_start",
        fn={
            "initialize": row_emb_api.rowwise_adagrad_lazy_update_start_init(),
            "execute": row_emb_api.rowwise_adagrad_lazy_update_start(),
        },
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "xrex_row_emb_rowwise_adagrad_update_done",
        fn=row_emb_api.rowwise_adagrad_update_done(),
        platform="CUDA",
    )
except AttributeError as e:
    raise ImportError(
        f"the compiled row_emb extension at {row_emb_api.__file__} does not "
        f"match these bindings (stale build?): {e}"
    ) from e


class RowEmbContextHandle(NamedTuple):
    context_id: int
    group_size: int
    flatten_replicas: tuple[int, ...]
    tokens_per_rank: int
    emb_width: int
    rows_per_rank: int
    recv_capacity: int
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
            emb_width=self.emb_width,
            rows_per_rank=self.rows_per_rank,
            recv_capacity=self.recv_capacity,
            num_devices_per_node=self.num_devices_per_node,
        )


def make_context_handle(
    mesh: jax.sharding.Mesh,
    table_axis: tuple[str, ...],
    *,
    data_axis: tuple[str, ...],
    tokens_per_batch: int,
    emb_width: int,
    vocab_rows: int,
    recv_factor: float,
    num_devices_per_node: int,
) -> RowEmbContextHandle:
    for axis in dict.fromkeys((*table_axis, *data_axis)):
        if axis not in mesh.shape:
            raise ValueError(f"row_emb axis {axis!r} is not a mesh axis of {mesh}")
    group_size, flatten_replicas = get_flatten_replica_groups(mesh, table_axis)
    missing_axes = [axis for axis in table_axis if axis not in data_axis]
    if missing_axes:
        raise ValueError(
            f"row_emb requires token shards to vary across the communicator: "
            f"table_axis {missing_axes} missing from data_axis {data_axis}"
        )
    off_communicator_shards = math.prod(
        mesh.shape[axis] for axis in data_axis if axis not in table_axis
    )
    if off_communicator_shards != 1:
        raise ValueError(
            f"row_emb requires exactly one token shard per communicator rank: "
            f"data_axis {data_axis} shards tokens over {off_communicator_shards} "
            f"positions outside table_axis {table_axis}"
        )
    if tokens_per_batch % group_size != 0:
        raise ValueError(
            f"row_emb tokens_per_batch={tokens_per_batch} does not shard evenly "
            f"over the {group_size}-rank communicator"
        )
    if vocab_rows % group_size != 0:
        raise ValueError(
            f"row_emb vocab_rows={vocab_rows} does not shard evenly over the "
            f"{group_size}-rank communicator; pad input_vocab_size to a multiple of it"
        )
    if emb_width % 8 != 0:
        raise ValueError(f"row_emb emb_width={emb_width} must be a multiple of 8")
    if recv_factor <= 0:
        raise ValueError(f"row_emb recv_factor={recv_factor} must be positive")
    tokens_per_rank = tokens_per_batch // group_size
    recv_capacity = min(int(math.ceil(tokens_per_rank * recv_factor)), tokens_per_rank * group_size)
    device_ids = [d.id for d in mesh.devices.flatten()]
    flatten_replicas = tuple(device_ids[pos] for pos in flatten_replicas)
    group_key = get_context_id(group_size, flatten_replicas)
    context_id = get_context_id(
        group_key,
        (
            tokens_per_rank,
            emb_width,
            vocab_rows // group_size,
            recv_capacity,
            num_devices_per_node,
            7,
        ),
    )
    return RowEmbContextHandle(
        context_id=context_id,
        group_size=group_size,
        flatten_replicas=flatten_replicas,
        tokens_per_rank=tokens_per_rank,
        emb_width=emb_width,
        rows_per_rank=vocab_rows // group_size,
        recv_capacity=recv_capacity,
        num_devices_per_node=num_devices_per_node,
        mesh=mesh,
        table_axis=tuple(table_axis),
        data_axis=tuple(data_axis),
    )


def lookup_start(token_ids: jax.Array, table: jax.Array, gate: jax.Array, ctx: RowEmbContextHandle):
    assert math.prod(token_ids.shape) == ctx.tokens_per_rank
    assert table.shape == (ctx.rows_per_rank, ctx.emb_width) and table.dtype == jnp.bfloat16
    with jax.named_scope("row_emb.lookup_start"):
        outs = jax.ffi.ffi_call(
            "xrex_row_emb_lookup_start",
            [
                jax.ShapeDtypeStruct(table.shape, table.dtype),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
            input_output_aliases={1: 0},
        )(token_ids.reshape(-1).astype(jnp.int32), table, gate, **ctx.attrs())
    return outs[0], outs[1]


def lookup_done(pin: jax.Array, ctx: RowEmbContextHandle) -> jax.Array:
    with jax.named_scope("row_emb.lookup_done"):
        (embeddings,) = jax.ffi.ffi_call(
            "xrex_row_emb_lookup_done",
            [jax.ShapeDtypeStruct((ctx.tokens_per_rank, ctx.emb_width), jnp.bfloat16)],
        )(pin, context_id=ctx.context_id)
    return embeddings


def stage_update(
    grads: jax.Array,
    pending: jax.Array,
    gate: jax.Array,
    ctx: RowEmbContextHandle,
) -> jax.Array:
    assert grads.shape == (ctx.tokens_per_rank, ctx.emb_width) and grads.dtype == jnp.bfloat16
    with jax.named_scope("row_emb.stage_update"):
        (pin,) = jax.ffi.ffi_call(
            "xrex_row_emb_stage_update", [jax.ShapeDtypeStruct((1,), jnp.float32)]
        )(grads, pending.astype(jnp.int32).reshape(1), gate, **ctx.attrs())
    return pin


def rowwise_adagrad_update_start(
    table: jax.Array,
    row_state: jax.Array,
    gate: jax.Array,
    ctx: RowEmbContextHandle,
    *,
    learning_rate: float,
    eps: float,
    decay_factor: float,
    weight_decay_factor: float = 1.0,
):
    assert row_state.shape == (ctx.rows_per_rank,) and row_state.dtype == jnp.float32
    with jax.named_scope("row_emb.rowwise_adagrad_update_start"):
        outs = jax.ffi.ffi_call(
            "xrex_row_emb_rowwise_adagrad_update_start",
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
    ctx: RowEmbContextHandle,
    *,
    learning_rate: float,
    eps: float,
    accum_decay_rate: float,
    weight_decay_rate: float,
    weight_decay_exempt_window: float = 0.0,
    stochastic_rounding: bool = False,
):
    assert last_step.shape == row_state.shape and last_step.dtype == jnp.int32
    if (weight_decay_exempt_window > 0 or stochastic_rounding) and not getattr(
        row_emb_api, "LAZY_UPDATE_EXEMPT_WINDOW_AND_STOCHASTIC_ROUNDING", False
    ):
        raise RuntimeError(
            "stale xrex-cuda-kernels build: row_emb has no weight-decay exemption window or "
            "stochastic rounding"
        )
    with jax.named_scope("row_emb.rowwise_adagrad_lazy_update_start"):
        outs = jax.ffi.ffi_call(
            "xrex_row_emb_rowwise_adagrad_lazy_update_start",
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


def rowwise_adagrad_update_done(gate: jax.Array, ctx: RowEmbContextHandle):
    with jax.named_scope("row_emb.rowwise_adagrad_update_done"):
        outs = jax.ffi.ffi_call(
            "xrex_row_emb_rowwise_adagrad_update_done",
            [
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ],
        )(gate, context_id=ctx.context_id)
    return tuple(outs)
