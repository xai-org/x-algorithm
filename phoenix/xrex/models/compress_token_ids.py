# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from typing import cast

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from xrex.cuda.unique import unique


def _replicated_token_ids_with_zero(token_ids: jax.Array) -> jax.Array:
    all_token_ids = jnp.concatenate([token_ids.reshape((-1,)), jnp.array([0])])
    return jax.lax.with_sharding_constraint(all_token_ids, P())


def _sorted_unique(values: jax.Array, size: int, fill_value: int) -> jax.Array:
    sorted_values = jnp.sort(values)
    is_first = jnp.concatenate([jnp.ones((1,), bool), sorted_values[1:] != sorted_values[:-1]])
    slot = jnp.where(is_first, jnp.cumsum(is_first, dtype=jnp.int32) - 1, size)
    return jnp.full((size,), fill_value, values.dtype).at[slot].set(sorted_values, mode="drop")


def compress_token_ids(
    token_ids: jax.Array,
    fill_size: int,
    fill_value: int,
    output_sharding: P = P(),
) -> tuple[jax.Array, jax.Array]:
    all_token_ids = _replicated_token_ids_with_zero(token_ids)
    unique_tokens, inverse_indices = unique(
        all_token_ids,
        return_inverse=True,
        size=fill_size,
        fill_value=fill_value,
    )
    unique_tokens = cast(jax.Array, unique_tokens)
    inverse_indices = cast(jax.Array, inverse_indices)

    inverse_indices = inverse_indices[:-1]

    inverse_indices = jax.lax.with_sharding_constraint(inverse_indices, output_sharding)
    inverse_indices = cast(jax.Array, jnp.where(inverse_indices == 0, -1, inverse_indices))

    return unique_tokens, inverse_indices


def compress_token_ids_by_search(
    token_ids: jax.Array, fill_size: int, mesh: Mesh, token_ids_spec: P
) -> tuple[jax.Array, jax.Array]:
    unique_tokens = _sorted_unique(
        _replicated_token_ids_with_zero(token_ids), fill_size, jnp.iinfo(token_ids.dtype).max
    )

    @shard_map(mesh=mesh, in_specs=(P(), token_ids_spec), out_specs=token_ids_spec, check_vma=False)
    def find_inverse_indices(unique_tokens: jax.Array, local_token_ids: jax.Array) -> jax.Array:
        inverse_indices = jnp.searchsorted(
            unique_tokens, local_token_ids, side="left", method="scan_unrolled"
        ).astype(unique_tokens.dtype)
        return jnp.where(inverse_indices == 0, -1, inverse_indices)

    return unique_tokens, find_inverse_indices(unique_tokens, token_ids)
