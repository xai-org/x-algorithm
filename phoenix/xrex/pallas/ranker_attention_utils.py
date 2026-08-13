# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

HISTORY_SEGMENT_ID = 1
CANDIDATE_SEGMENT_ID = -1
PADDING_SEGMENT_ID = 0


def history_candidate_mask(
    q_seq_base: jax.Array | int,
    block_q: int,
    kv_seq_base: jax.Array | int,
    block_kv: int,
    history_lower_bound: jax.Array | int,
    history_upper_bound: jax.Array | int,
    candidate_lower_bound: jax.Array | int,
    candidate_upper_bound: jax.Array | int,
    broadcasted_iota_fn: Callable[..., jax.Array] | None = None,
    layout: object | None = None,
) -> jax.Array:
    q_seq_base = jnp.asarray(q_seq_base, dtype=jnp.int32)
    kv_seq_base = jnp.asarray(kv_seq_base, dtype=jnp.int32)
    history_lower_bound = jnp.asarray(history_lower_bound, dtype=jnp.int32)
    history_upper_bound = jnp.asarray(history_upper_bound, dtype=jnp.int32)
    candidate_lower_bound = jnp.asarray(candidate_lower_bound, dtype=jnp.int32)
    candidate_upper_bound = jnp.asarray(candidate_upper_bound, dtype=jnp.int32)

    if broadcasted_iota_fn is None:
        q_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_kv), 0)
        kv_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_kv), 1)
    else:
        if layout is None:
            q_ids = broadcasted_iota_fn(jnp.int32, (block_q, block_kv), 0)
            kv_ids = broadcasted_iota_fn(jnp.int32, (block_q, block_kv), 1)
        else:
            q_ids = broadcasted_iota_fn(jnp.int32, (block_q, block_kv), 0, layout=layout)
            kv_ids = broadcasted_iota_fn(jnp.int32, (block_q, block_kv), 1, layout=layout)

    q_pos = q_ids + q_seq_base
    kv_pos = kv_ids + kv_seq_base

    q_is_history = (q_pos >= history_lower_bound) & (q_pos < history_upper_bound)
    q_is_candidate = (q_pos >= candidate_lower_bound) & (q_pos < candidate_upper_bound)
    kv_is_history = (kv_pos >= history_lower_bound) & (kv_pos < history_upper_bound)
    kv_is_candidate = (kv_pos >= candidate_lower_bound) & (kv_pos < candidate_upper_bound)

    history_mask = kv_is_history & (q_is_history | q_is_candidate)
    candidate_self_mask = q_is_candidate & kv_is_candidate & (q_pos == kv_pos)
    return history_mask | candidate_self_mask


def history_candidate_mask_kv_q(
    q_seq_base: jax.Array | int,
    block_q: int,
    kv_seq_base: jax.Array | int,
    block_kv: int,
    history_lower_bound: jax.Array | int,
    history_upper_bound: jax.Array | int,
    candidate_lower_bound: jax.Array | int,
    candidate_upper_bound: jax.Array | int,
    broadcasted_iota_fn: Callable[..., jax.Array] | None = None,
    layout: object | None = None,
) -> jax.Array:
    q_seq_base = jnp.asarray(q_seq_base, dtype=jnp.int32)
    kv_seq_base = jnp.asarray(kv_seq_base, dtype=jnp.int32)
    history_lower_bound = jnp.asarray(history_lower_bound, dtype=jnp.int32)
    history_upper_bound = jnp.asarray(history_upper_bound, dtype=jnp.int32)
    candidate_lower_bound = jnp.asarray(candidate_lower_bound, dtype=jnp.int32)
    candidate_upper_bound = jnp.asarray(candidate_upper_bound, dtype=jnp.int32)

    if broadcasted_iota_fn is None:
        kv_ids = lax.broadcasted_iota(jnp.int32, (block_kv, block_q), 0)
        q_ids = lax.broadcasted_iota(jnp.int32, (block_kv, block_q), 1)
    else:
        if layout is None:
            kv_ids = broadcasted_iota_fn(jnp.int32, (block_kv, block_q), 0)
            q_ids = broadcasted_iota_fn(jnp.int32, (block_kv, block_q), 1)
        else:
            kv_ids = broadcasted_iota_fn(jnp.int32, (block_kv, block_q), 0, layout=layout)
            q_ids = broadcasted_iota_fn(jnp.int32, (block_kv, block_q), 1, layout=layout)

    q_pos = q_ids + q_seq_base
    kv_pos = kv_ids + kv_seq_base

    q_is_history = (q_pos >= history_lower_bound) & (q_pos < history_upper_bound)
    q_is_candidate = (q_pos >= candidate_lower_bound) & (q_pos < candidate_upper_bound)
    kv_is_history = (kv_pos >= history_lower_bound) & (kv_pos < history_upper_bound)
    kv_is_candidate = (kv_pos >= candidate_lower_bound) & (kv_pos < candidate_upper_bound)

    history_mask = kv_is_history & (q_is_history | q_is_candidate)
    candidate_self_mask = q_is_candidate & kv_is_candidate & (q_pos == kv_pos)
    return history_mask | candidate_self_mask


def kv_block_for_step(
    kv_step: jax.Array | int,
    hist_k_start: jax.Array | int,
    hist_steps: jax.Array | int,
    cand_k_start: jax.Array | int,
) -> jax.Array:
    kv_step = jnp.asarray(kv_step, dtype=jnp.int32)
    hist_k_start = jnp.asarray(hist_k_start, dtype=jnp.int32)
    hist_steps = jnp.asarray(hist_steps, dtype=jnp.int32)
    cand_k_start = jnp.asarray(cand_k_start, dtype=jnp.int32)
    return jnp.where(
        kv_step < hist_steps,
        hist_k_start + kv_step,
        cand_k_start + (kv_step - hist_steps),
    )


def bounds_from_segment_ids(segment_ids: jax.Array) -> jax.Array:
    segment_ids = jnp.asarray(segment_ids)
    if segment_ids.ndim == 1:
        segment_ids = segment_ids[None, :]
    if segment_ids.ndim != 2:
        raise ValueError(f"segment_ids should have shape (B, S) or (S,), got {segment_ids.shape}")

    history_mask = segment_ids == HISTORY_SEGMENT_ID
    candidate_mask = segment_ids == CANDIDATE_SEGMENT_ID
    seq_len = segment_ids.shape[1]

    def _bounds(mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        any_mask = jnp.any(mask, axis=1)
        first = jnp.argmax(mask, axis=1)
        last = (seq_len - 1) - jnp.argmax(mask[:, ::-1], axis=1)
        lower = jnp.where(any_mask, first, 0)
        upper = jnp.where(any_mask, last + 1, 0)
        return lower.astype(jnp.int32), upper.astype(jnp.int32)

    history_lower, history_upper = _bounds(history_mask)
    candidate_lower, candidate_upper = _bounds(candidate_mask)

    return jnp.stack(
        [history_lower, history_upper, candidate_lower, candidate_upper], axis=1
    ).astype(jnp.int32)
