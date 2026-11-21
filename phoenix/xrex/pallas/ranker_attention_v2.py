# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas.triton import CompilerParams

from xrex.pallas.ranker_attention import _preprocess_backward

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
BLOCK_Q_FWD = 128
BLOCK_K_FWD = 128

BLOCK_Q_BWD_DQ = 128
BLOCK_K_BWD_DQ = 16

BLOCK_Q_BWD_DKV = 16
BLOCK_K_BWD_DKV = 128

HISTORY_SEGMENT_ID = 1
CANDIDATE_SEGMENT_ID = -1
PADDING_SEGMENT_ID = 0


def tanh(x):
    return 2 * jax.lax.logistic(2 * x) - 1


def compute_block_state_metadata(
    segment_ids: jnp.ndarray,
    block_size: int,
) -> jnp.ndarray:
    batch_size, seq_len = segment_ids.shape
    num_blocks = pl.cdiv(seq_len, block_size)

    block_segments = segment_ids.reshape(batch_size, num_blocks, block_size)
    has_history = jnp.any(block_segments == HISTORY_SEGMENT_ID, axis=-1)
    has_candidate = jnp.any(block_segments == CANDIDATE_SEGMENT_ID, axis=-1)
    has_padding = jnp.any(block_segments == PADDING_SEGMENT_ID, axis=-1)

    has_history_int = has_history.astype(jnp.int8)
    has_candidate_int = has_candidate.astype(jnp.int8)
    has_padding_int = has_padding.astype(jnp.int8)
    state_codes = (has_padding_int << 2) | (has_candidate_int << 1) | has_history_int
    return state_codes


def compute_last_history_block_idx(
    segment_ids: jnp.ndarray,
    block_size: int,
) -> jnp.ndarray:
    batch_size, seq_len = segment_ids.shape
    num_blocks = pl.cdiv(seq_len, block_size)
    block_segments = segment_ids.reshape(batch_size, num_blocks, block_size)
    has_history_tokens = jnp.any(block_segments == HISTORY_SEGMENT_ID, axis=-1)
    last_history_block_idx = jnp.max(
        jnp.where(has_history_tokens, jnp.arange(num_blocks), -1), axis=-1
    )
    return last_history_block_idx[:, None]


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    temp_ref,
    segment_ref,
    block_state_metadata_ref,
    last_history_k_block_idx_ref,
    o_ref,
    *residual_refs,
    sm_scale: float,
    cap: float,
    cap_method: str,
    causal: bool,
    window_len: int,
    block_q: int,
    block_d: int,
    block_k: int,
):
    del window_len
    start_q = pl.program_id(0)

    m_i = jnp.full(block_q, -float("inf"), dtype=jnp.float32)
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    acc = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    offset_q = start_q * block_q
    q = pl.load(q_ref, (pl.dslice(offset_q, block_q), pl.dslice(None)))

    seg_q = pl.load(
        segment_ref,
        (pl.dslice(offset_q, block_q),),
    ).astype(jnp.int8)[:, None]
    seq_q_is_not_padding = seg_q != PADDING_SEGMENT_ID

    span_q = offset_q + jnp.arange(block_q)

    temp = pl.load(temp_ref, (pl.dslice(offset_q, block_q),))
    temp = jnp.expand_dims(temp, axis=-1)

    def body(start_k, carry):
        acc, m_prev, l_prev = carry

        offset_k = start_k * block_k
        k = pl.load(k_ref, (slice(None), pl.dslice(offset_k, block_k)))
        seg_k = pl.load(
            segment_ref,
            (pl.dslice(offset_k, block_k),),
        ).astype(jnp.int8)[None, :]
        span_k = offset_k + jnp.arange(block_k)

        qk = pl.dot(q, k)
        if sm_scale != 1.0:
            qk *= sm_scale

        if cap > 0.0:
            if cap_method == "tanh":
                qk = cap * tanh(qk / cap)
            elif cap_method == "soft_sign":
                qk = qk / (1.0 + jnp.abs(qk) / cap)
            else:
                raise ValueError(f"cap_method must be in [tanh, soft_sign], got {cap_method}")

        qk *= temp

        mask = jnp.logical_or(seg_k == HISTORY_SEGMENT_ID, span_q[:, None] == span_k[None, :])
        mask = jnp.logical_and(seq_q_is_not_padding, mask)
        if causal:
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask = jnp.logical_and(mask, causal_mask)

        qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
        max_logit = jnp.max(qk, axis=1)
        max_logit = jnp.maximum(max_logit, 1.0)
        m_curr = jnp.maximum(m_prev, max_logit)
        p = jnp.exp(qk - m_curr[:, None])
        alpha = jnp.exp(m_prev - m_curr)
        l_prev = l_prev * alpha + jnp.sum(p, axis=1)

        p = p.astype(q.dtype)
        v = pl.load(v_ref, (pl.dslice(offset_k, block_k), pl.dslice(block_d)))
        acc = acc * alpha[:, None] + pl.dot(p, v)
        return acc, m_curr, l_prev

    q_block_state_metadata = jnp.squeeze(pl.load(block_state_metadata_ref, (start_q,)))

    last_history_k_block_idx = jnp.squeeze(pl.load(last_history_k_block_idx_ref, ()))
    history_upper_bound = jnp.where(
        q_block_state_metadata == 4,
        0,
        last_history_k_block_idx + 1,
    )

    acc, m_i, l_i = lax.fori_loop(
        0,
        history_upper_bound,
        body,
        (acc, m_i, l_i),
    )

    candidate_lower_bound = offset_q // block_k
    candidate_lower_bound = jnp.maximum(history_upper_bound, candidate_lower_bound)

    candidate_upper_bound = (offset_q + block_q - 1) // block_k + 1
    candidate_upper_bound = jnp.where(
        (q_block_state_metadata & 2) == 2,
        candidate_upper_bound,
        0,
    )

    acc, m_i, l_i = lax.fori_loop(
        candidate_lower_bound,
        candidate_upper_bound,
        body,
        (acc, m_i, l_i),
    )

    acc /= jnp.where(l_i == 0, 1.0, l_i)[:, None]

    if residual_refs:
        l_ref, m_ref = residual_refs
        pl.store(l_ref, (pl.ds(offset_q, block_q),), l_i)
        pl.store(m_ref, (pl.ds(offset_q, block_q),), m_i)
    acc = acc.astype(o_ref.dtype)
    pl.store(o_ref, (pl.dslice(offset_q, block_q), pl.dslice(None)), acc)


def mha_forward_kernel_inference(
    q_ref,
    k_ref,
    v_ref,
    temp_ref,
    segment_ref,
    block_state_metadata_ref,
    last_history_k_block_idx_ref,
    o_ref,
    *,
    sm_scale: float,
    cap: float,
    cap_method: str,
    block_q: int,
    block_d: int,
    block_k: int,
):
    start_q = pl.program_id(0)

    m_i = jnp.full(block_q, -float("inf"), dtype=jnp.float32)
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    acc = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    offset_q = start_q * block_q
    q = pl.load(q_ref, (pl.dslice(offset_q, block_q), pl.dslice(None)))

    seg_q = pl.load(segment_ref, (pl.dslice(offset_q, block_q),)).astype(jnp.int8)[:, None]
    seq_q_is_not_padding = seg_q != PADDING_SEGMENT_ID

    span_q = offset_q + jnp.arange(block_q)

    temp = pl.load(temp_ref, (pl.dslice(offset_q, block_q),))
    temp = jnp.expand_dims(temp, axis=-1)

    def history_body(start_k, carry):
        acc, m_prev, l_prev = carry

        offset_k = start_k * block_k
        k = pl.load(k_ref, (slice(None), pl.dslice(offset_k, block_k)))
        seg_k = pl.load(segment_ref, (pl.dslice(offset_k, block_k),)).astype(jnp.int8)[None, :]
        span_k = offset_k + jnp.arange(block_k)

        qk = pl.dot(q, k)
        if sm_scale != 1.0:
            qk *= sm_scale

        if cap > 0.0:
            if cap_method == "tanh":
                qk = cap * tanh(qk / cap)
            elif cap_method == "soft_sign":
                qk = qk / (1.0 + jnp.abs(qk) / cap)

        qk *= temp

        mask = jnp.logical_or(seg_k == HISTORY_SEGMENT_ID, span_q[:, None] == span_k[None, :])
        mask = jnp.logical_and(seq_q_is_not_padding, mask)
        qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        max_logit = jnp.max(qk, axis=1)
        max_logit = jnp.maximum(max_logit, 1.0)
        m_curr = jnp.maximum(m_prev, max_logit)
        p = jnp.exp(qk - m_curr[:, None])
        alpha = jnp.exp(m_prev - m_curr)
        l_prev = l_prev * alpha + jnp.sum(p, axis=1)

        p = p.astype(q.dtype)
        v = pl.load(v_ref, (pl.dslice(offset_k, block_k), pl.dslice(block_d)))
        acc = acc * alpha[:, None] + pl.dot(p, v)
        return acc, m_curr, l_prev

    def candidate_body(start_k, carry):
        acc, m_prev, l_prev = carry

        offset_k = start_k * block_k

        k = pl.load(k_ref, (slice(None), pl.dslice(offset_k, block_k)))

        qk_diag = jnp.sum(q.astype(jnp.float32) * k.T.astype(jnp.float32), axis=-1)

        if sm_scale != 1.0:
            qk_diag = qk_diag * sm_scale

        if cap > 0.0:
            if cap_method == "tanh":
                qk_diag = cap * tanh(qk_diag / cap)
            elif cap_method == "soft_sign":
                qk_diag = qk_diag / (1.0 + jnp.abs(qk_diag) / cap)

        qk_diag = qk_diag * temp.squeeze(-1)

        qk_diag = jnp.where(seq_q_is_not_padding.squeeze(-1), qk_diag, DEFAULT_MASK_VALUE)

        max_logit = jnp.maximum(qk_diag, 1.0)
        m_curr = jnp.maximum(m_prev, max_logit)
        p_diag = jnp.exp(qk_diag - m_curr)
        alpha = jnp.exp(m_prev - m_curr)
        l_prev = l_prev * alpha + p_diag

        v = pl.load(v_ref, (pl.dslice(offset_k, block_k), pl.dslice(block_d)))
        acc = acc * alpha[:, None] + p_diag[:, None] * v.astype(jnp.float32)
        return acc, m_curr, l_prev

    q_block_state_metadata = jnp.squeeze(pl.load(block_state_metadata_ref, (start_q,)))
    last_history_k_block_idx = jnp.squeeze(pl.load(last_history_k_block_idx_ref, ()))

    history_upper_bound = jnp.where(
        q_block_state_metadata == 4,
        0,
        last_history_k_block_idx + 1,
    )

    acc, m_i, l_i = lax.fori_loop(
        0,
        history_upper_bound,
        history_body,
        (acc, m_i, l_i),
    )

    candidate_lower_bound = offset_q // block_k
    candidate_lower_bound = jnp.maximum(history_upper_bound, candidate_lower_bound)

    candidate_upper_bound = (offset_q + block_q - 1) // block_k + 1
    candidate_upper_bound = jnp.where(
        (q_block_state_metadata & 2) == 2,
        candidate_upper_bound,
        0,
    )

    acc, m_i, l_i = lax.fori_loop(
        candidate_lower_bound,
        candidate_upper_bound,
        candidate_body,
        (acc, m_i, l_i),
    )

    acc /= jnp.where(l_i == 0, 1.0, l_i)[:, None]
    acc = acc.astype(o_ref.dtype)
    pl.store(o_ref, (pl.dslice(offset_q, block_q), pl.dslice(None)), acc)


def mha_inference_gqa(
    q,
    k,
    v,
    temp,
    segment_ids,
    sm_scale: float = 1.0,
    cap: float = -1.0,
    cap_method: str = "tanh",
    block_q: int = BLOCK_Q_FWD,
    block_k: int = BLOCK_K_FWD,
    num_warps: int | None = None,
    num_stages: int = 2,
    interpret: bool = False,
    debug: bool = False,
):
    batch_size, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    gqa_ratio = num_q_heads // num_kv_heads

    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)

    assert seq_len % block_q == 0, f"seq_len {seq_len} must be divisible by block_q {block_q}"
    assert seq_len % block_k == 0, f"seq_len {seq_len} must be divisible by block_k {block_k}"

    num_q_blocks = pl.cdiv(seq_len, block_q)
    grid = (num_q_blocks, batch_size, num_q_heads)

    if num_warps is None:
        num_warps = 4 if head_dim <= 64 else 8
    if num_stages == 2 and head_dim > 64:
        num_stages = 3

    q_block_state_metadata = compute_block_state_metadata(segment_ids, block_q)
    last_history_k_block_idx = compute_last_history_block_idx(segment_ids, block_k)

    kernel = functools.partial(
        mha_forward_kernel_inference,
        sm_scale=sm_scale,
        cap=cap,
        cap_method=cap_method,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
    )

    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)

    _gr = gqa_ratio

    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0),
                block_shape=(None, seq_len, None, head_dim),
            ),
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k // _gr, 0),
                block_shape=(None, head_dim, None, seq_len),
            ),
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k // _gr, 0),
                block_shape=(None, seq_len, None, head_dim),
            ),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, num_q_blocks)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, 1)),
        ],
        out_specs=pl.BlockSpec(
            index_map=lambda _, j, k: (j, 0, k, 0),
            block_shape=(None, seq_len, None, head_dim),
        ),
        compiler_params=CompilerParams(num_warps=num_warps, num_stages=num_stages),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward_inference_gqa",
    )(q, k.swapaxes(1, 3), v, temp, segment_ids, q_block_state_metadata, last_history_k_block_idx)


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "cap",
        "cap_method",
        "causal",
        "window_len",
        "z_loss_weight",
        "block_q_fwd",
        "block_k_fwd",
        "block_q_bwd_dq",
        "block_k_bwd_dq",
        "block_q_bwd_dkv",
        "block_k_bwd_dkv",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
    ],
)
def mha(
    q,
    k,
    v,
    temp,
    segment_ids,
    sm_scale: float = 1.0,
    cap: float = -1.0,
    cap_method: str = "tanh",
    causal: bool = True,
    window_len: int = -1,
    z_loss_weight: float = 0.0,
    block_q_fwd: int = BLOCK_Q_FWD,
    block_k_fwd: int = BLOCK_K_FWD,
    block_q_bwd_dq: int = BLOCK_Q_BWD_DQ,
    block_k_bwd_dq: int = BLOCK_K_BWD_DQ,
    block_q_bwd_dkv: int = BLOCK_Q_BWD_DKV,
    block_k_bwd_dkv: int = BLOCK_K_BWD_DKV,
    backward_pass_impl: str = "triton_split",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid=None,
    interpret: bool = False,
    debug: bool = False,
):
    del backward_pass_impl, block_q_bwd_dq, block_k_bwd_dq, block_q_bwd_dkv, block_k_bwd_dkv
    batch_size, seq_len, num_heads, head_dim = q.shape
    block_q_fwd = min(block_q_fwd, seq_len)
    block_k_fwd = min(block_k_fwd, seq_len)
    assert seq_len % block_q_fwd == 0, (
        f"seq_len {seq_len} must be divisible by block_q_fwd {block_q_fwd}"
    )
    assert seq_len % block_k_fwd == 0, (
        f"seq_len {seq_len} must be divisible by block_k_fwd {block_k_fwd}"
    )

    num_q_blocks = pl.cdiv(seq_len, block_q_fwd)
    grid_ = grid
    if grid_ is None:
        grid_ = (num_q_blocks, batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8

    q_block_state_metadata = compute_block_state_metadata(segment_ids, block_q_fwd)
    last_history_k_block_idx = compute_last_history_block_idx(segment_ids, block_k_fwd)

    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        cap=cap,
        cap_method=cap_method,
        block_q=block_q_fwd,
        block_k=block_k_fwd,
        block_d=head_dim,
        causal=causal,
        window_len=window_len,
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    return pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=[
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, head_dim, None, seq_len)
            ),
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, num_q_blocks)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, 1)),
        ],
        out_specs=pl.BlockSpec(
            index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
        ),
        compiler_params=CompilerParams(num_warps=num_warps_, num_stages=num_stages),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward_v3",
    )(q, k.swapaxes(1, 3), v, temp, segment_ids, q_block_state_metadata, last_history_k_block_idx)


def _mha_forward(
    q,
    k,
    v,
    temp,
    segment_ids,
    sm_scale: float,
    cap: float,
    cap_method: str,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q_fwd: int,
    block_k_fwd: int,
    block_q_bwd_dq: int,
    block_k_bwd_dq: int,
    block_q_bwd_dkv: int,
    block_k_bwd_dkv: int,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
    del backward_pass_impl
    batch_size, seq_len, num_heads, head_dim = q.shape
    block_q_fwd = min(block_q_fwd, seq_len)
    block_k_fwd = min(block_k_fwd, seq_len)
    assert seq_len % block_q_fwd == 0, (
        f"seq_len {seq_len} must be divisible by block_q_fwd {block_q_fwd}"
    )
    assert seq_len % block_k_fwd == 0, (
        f"seq_len {seq_len} must be divisible by block_k_fwd {block_k_fwd}"
    )
    num_q_blocks = pl.cdiv(seq_len, block_q_fwd)
    grid_ = grid
    if grid_ is None:
        grid_ = (num_q_blocks, batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8

    q_block_state_metadata = compute_block_state_metadata(segment_ids, block_q_fwd)
    last_history_k_block_idx = compute_last_history_block_idx(segment_ids, block_k_fwd)

    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        cap=cap,
        cap_method=cap_method,
        causal=causal,
        window_len=window_len,
        block_q=block_q_fwd,
        block_k=block_k_fwd,
        block_d=head_dim,
    )
    out, l, m = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=[
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, head_dim, None, seq_len)
            ),
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, num_q_blocks)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, 0), block_shape=(None, 1)),
        ],
        out_specs=[
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(index_map=lambda _, j, k: (j, k, 0), block_shape=(None, None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, k, 0), block_shape=(None, None, seq_len)),
        ],
        compiler_params=CompilerParams(num_warps=num_warps_, num_stages=num_stages),
        out_shape=[
            jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), dtype=jnp.float32),
        ],
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k.swapaxes(1, 3), v, temp, segment_ids, q_block_state_metadata, last_history_k_block_idx)
    return out, (
        q,
        k,
        v,
        temp,
        segment_ids,
        out,
        l,
        m,
    )


def mha_backward_kernel_dq(
    q_ref,
    k_ref,
    v_ref,
    temp_ref,
    segment_ref,
    q_block_state_metadata_ref,
    last_history_k_block_idx_ref,
    out_ref,
    do_scaled_ref,
    l_ref,
    m_ref,
    delta_ref,
    dq_ref,
    *,
    sm_scale: float,
    cap: float,
    cap_method: str,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q: int,
    block_d: int,
    block_k: int,
):
    del out_ref, window_len
    start_q = pl.program_id(2)

    offset_q = start_q * block_q
    q = pl.load(q_ref, (pl.ds(offset_q, block_q), slice(None)))
    m = pl.load(m_ref, (pl.ds(offset_q, block_q),))
    l = pl.load(l_ref, (pl.ds(offset_q, block_q),))
    do = pl.load(do_scaled_ref, (pl.ds(offset_q, block_q), slice(None)))
    di = pl.load(delta_ref, (pl.ds(offset_q, block_q),))
    dq = jnp.zeros([block_q, block_d], dtype=jnp.float32)

    span_q = offset_q + jnp.arange(block_q)
    seg_q = pl.load(
        segment_ref,
        (pl.ds(offset_q, block_q),),
    ).astype(jnp.int8)[:, None]
    seq_q_is_not_padding = seg_q != PADDING_SEGMENT_ID

    temp = pl.load(temp_ref, (pl.dslice(offset_q, block_q),))[:, None]

    def inner_loop(start_k, dq):
        offset_k = start_k * block_k
        k = pl.load(k_ref, (pl.ds(offset_k, block_k), slice(None)))
        v = pl.load(v_ref, (pl.ds(offset_k, block_k), slice(None)))
        seg_k = pl.load(
            segment_ref,
            (pl.dslice(offset_k, block_k),),
        ).astype(jnp.int8)[None, :]
        span_k = offset_k + jnp.arange(block_k)
        qk = pl.dot(q, k, trans_b=True)

        if sm_scale != 1.0:
            qk *= sm_scale

        if cap > 0.0:
            if cap_method == "tanh":
                qk_tanh = tanh(qk / cap)
                qk = cap * qk_tanh
            elif cap_method == "soft_sign":
                soft_sign = 1.0 / (1.0 + jnp.abs(qk) / cap)
                qk = qk * soft_sign
            else:
                raise ValueError("cap_method must be in [tanh, soft_sign]")

        qk *= temp

        mask = jnp.logical_or(seg_k == HISTORY_SEGMENT_ID, span_q[:, None] == span_k[None, :])
        mask = jnp.logical_and(seq_q_is_not_padding, mask)
        if causal:
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask = jnp.logical_and(mask, causal_mask)

        p = jnp.exp(qk - m[:, None])
        p = jnp.where(mask, p, 0.0)
        dp = pl.dot(do, v, trans_b=True).astype(jnp.float32) - di[:, None]
        ds = p * dp

        if z_loss_weight > 0:
            ds += z_loss_weight * p * ((jnp.log(l + 1e-12) + m) / l)[:, None]

        ds *= temp

        if cap > 0.0:
            if cap_method == "tanh":
                ds = ds * (1 - qk_tanh**2)
            elif cap_method == "soft_sign":
                ds = ds * soft_sign**2
            else:
                raise ValueError("cap_method must be in [tanh, soft_sign]")

        if sm_scale != 1.0:
            ds = ds * sm_scale

        dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
        return dq

    q_block_state_metadata = pl.load(q_block_state_metadata_ref, (start_q,))
    last_history_k_block_idx = pl.load(last_history_k_block_idx_ref, ())

    history_upper_bound = jnp.squeeze(last_history_k_block_idx) + 1
    history_upper_bound = jnp.where(
        q_block_state_metadata == 4,
        0,
        history_upper_bound,
    )
    dq = lax.fori_loop(0, history_upper_bound, inner_loop, dq)

    candidate_lower_bound = lax.div(offset_q, block_k)
    candidate_lower_bound = jnp.maximum(history_upper_bound, candidate_lower_bound)
    candidate_upper_bound = lax.div(offset_q + block_q - 1, block_k) + 1
    candidate_upper_bound = jnp.where(
        (q_block_state_metadata & 2) == 2,
        candidate_upper_bound,
        0,
    )
    dq = lax.fori_loop(candidate_lower_bound, candidate_upper_bound, inner_loop, dq)

    pl.store(dq_ref, (pl.ds(offset_q, block_q), slice(None)), dq.astype(q_ref.dtype))


def mha_backward_kernel_dkv(
    q_ref,
    k_ref,
    v_ref,
    temp_ref,
    segment_ref,
    block_state_metadata_ref,
    out_ref,
    do_scaled_ref,
    l_ref,
    m_ref,
    delta_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    cap: float,
    cap_method: str,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q: int,
    block_d: int,
    block_k: int,
):
    del out_ref, window_len
    seq_len = q_ref.shape[0]
    start_k = pl.program_id(2)
    offset_k = start_k * block_k

    dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    k = pl.load(k_ref, (pl.ds(offset_k, block_k), slice(None)))
    v = pl.load(v_ref, (pl.ds(offset_k, block_k), slice(None)))
    span_k = offset_k + jnp.arange(block_k)

    seg_k = pl.load(
        segment_ref,
        (pl.ds(offset_k, block_k),),
    ).astype(jnp.int8)[None, :]
    seg_k_is_history = seg_k == HISTORY_SEGMENT_ID

    def inner_loop(start_q, carry):
        dv, dk = carry
        offset_q = start_q * block_q
        q = pl.load(q_ref, (pl.ds(offset_q, block_q), slice(None)))
        qk = pl.dot(q, k, trans_b=True)
        seg_q = pl.load(
            segment_ref,
            (pl.ds(offset_q, block_q),),
        ).astype(jnp.int8)[:, None]
        seq_q_is_not_padding = seg_q != PADDING_SEGMENT_ID
        span_q = offset_q + jnp.arange(block_q)

        m = pl.load(m_ref, (pl.ds(offset_q, block_q),))
        do = pl.load(do_scaled_ref, (pl.ds(offset_q, block_q), slice(None)))
        di = pl.load(delta_ref, (pl.ds(offset_q, block_q),))

        if sm_scale != 1.0:
            qk *= sm_scale

        if cap > 0.0:
            if cap_method == "tanh":
                qk_tanh = tanh(qk / cap)
                qk = cap * qk_tanh
            elif cap_method == "soft_sign":
                soft_sign = 1.0 / (1.0 + jnp.abs(qk) / cap)
                qk = qk * soft_sign
            else:
                raise ValueError("cap_method must be in [tanh, soft_sign]")

        temp = pl.load(temp_ref, (pl.dslice(offset_q, block_q),))[:, None]
        qk *= temp

        mask = jnp.logical_or(seg_k_is_history, span_q[:, None] == span_k[None, :])
        mask = jnp.logical_and(seq_q_is_not_padding, mask)
        if causal:
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask = jnp.logical_and(mask, causal_mask)

        p = jnp.exp(qk - m[:, None])
        p = jnp.where(mask, p, 0.0)
        dot_result = pl.dot(p.astype(do.dtype), do, trans_a=True)
        dv = dv + dot_result.astype(dv.dtype)

        dp = pl.dot(do, v, trans_b=True) - di[:, None]

        ds = p * dp
        if z_loss_weight > 0:
            l = pl.load(l_ref, (pl.ds(offset_q, block_q),))
            ds += z_loss_weight * p * ((jnp.log(l + 1e-12) + m) / l)[:, None]
        ds *= temp
        if cap > 0.0:
            if cap_method == "tanh":
                ds = ds * (1 - qk_tanh**2)
            elif cap_method == "soft_sign":
                ds = ds * soft_sign**2
            else:
                raise ValueError("cap_method must be in [tanh, soft_sign]")
        if sm_scale != 1.0:
            ds = ds * sm_scale
        dk = dk + pl.dot(ds.astype(q_ref.dtype), q, trans_a=True).astype(dk.dtype)
        return dv, dk

    k_block_state_metadata = pl.load(block_state_metadata_ref, (start_k,))

    lower_bound = jnp.where(
        (k_block_state_metadata & 1) == 1,
        0,
        offset_k // block_q,
    )

    upper_bound = jnp.where(
        (k_block_state_metadata & 1) == 1,
        seq_len // block_q,
        (offset_k + block_k - 1) // block_q + 1,
    )
    dv, dk = lax.fori_loop(lower_bound, upper_bound, inner_loop, (dv, dk))

    pl.store(dv_ref, (pl.ds(offset_k, block_k), slice(None)), dv.astype(dv_ref.dtype))
    pl.store(dk_ref, (pl.ds(offset_k, block_k), slice(None)), dk.astype(dk_ref.dtype))


def _mha_backward(
    sm_scale: float,
    cap: float,
    cap_method: str,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q_fwd: int,
    block_k_fwd: int,
    block_q_bwd_dq: int,
    block_k_bwd_dq: int,
    block_q_bwd_dkv: int,
    block_k_bwd_dkv: int,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    res,
    do,
):
    del num_warps, num_stages, grid
    q, k, v, temp, segment_ids, out, l, m = res

    batch_size, seq_len, num_heads, head_dim = q.shape
    assert seq_len % block_q_bwd_dq == 0, (
        f"seq_len {seq_len} must be divisible by block_q_bwd_dq {block_q_bwd_dq}"
    )
    assert seq_len % block_k_bwd_dq == 0, (
        f"seq_len {seq_len} must be divisible by block_k_bwd_dq {block_k_bwd_dq}"
    )
    assert seq_len % block_q_bwd_dkv == 0, (
        f"seq_len {seq_len} must be divisible by block_q_bwd_dkv {block_q_bwd_dkv}"
    )
    assert seq_len % block_k_bwd_dkv == 0, (
        f"seq_len {seq_len} must be divisible by block_k_bwd_dkv {block_k_bwd_dkv}"
    )

    block_q_bwd_dq = min(block_q_bwd_dq, seq_len)
    block_k_bwd_dq = min(block_k_bwd_dq, seq_len)
    block_q_bwd_dkv = min(block_q_bwd_dkv, seq_len)
    block_k_bwd_dkv = min(block_k_bwd_dkv, seq_len)

    num_q_blocks_dq = pl.cdiv(seq_len, block_q_bwd_dq)
    num_k_blocks_dkv = pl.cdiv(seq_len, block_k_bwd_dkv)

    do_scaled, delta = _preprocess_backward(out, do, l, block_q_bwd_dq, debug, interpret)

    if backward_pass_impl == "triton_split":
        dtemp = jnp.zeros(temp.shape, temp.dtype)
        dsegment = jnp.zeros(segment_ids.shape, q.dtype)

        num_warps = 8
        num_stages = 4

        grid_q = (batch_size, num_heads, num_q_blocks_dq)
        q_block_bwd_dq_state_metadata = compute_block_state_metadata(segment_ids, block_q_bwd_dq)
        last_history_k_block_bwd_dq_idx = compute_last_history_block_idx(
            segment_ids, block_k_bwd_dq
        )

        dq = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_dq,
                block_q=block_q_bwd_dq,
                block_d=head_dim,
                block_k=block_k_bwd_dq,
                sm_scale=sm_scale,
                cap=cap,
                cap_method=cap_method,
                causal=causal,
                window_len=window_len,
                z_loss_weight=z_loss_weight,
            ),
            grid=grid_q,
            out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            in_specs=[
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(index_map=lambda j, k, _: (j, 0), block_shape=(None, seq_len)),
                pl.BlockSpec(index_map=lambda j, k, _: (j, 0), block_shape=(None, seq_len)),
                pl.BlockSpec(index_map=lambda j, k, _: (j, 0), block_shape=(None, num_q_blocks_dq)),
                pl.BlockSpec(index_map=lambda j, k, _: (j, 0), block_shape=(None, 1)),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, k, 0), block_shape=(None, None, seq_len)
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, k, 0), block_shape=(None, None, seq_len)
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, k, 0), block_shape=(None, None, seq_len)
                ),
            ],
            out_specs=pl.BlockSpec(
                index_map=lambda j, k, _: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            name="mha_backward_q",
            debug=debug,
            interpret=interpret,
            compiler_params=CompilerParams(num_warps=num_warps, num_stages=num_stages),
        )(
            q,
            k,
            v,
            temp,
            segment_ids,
            q_block_bwd_dq_state_metadata,
            last_history_k_block_bwd_dq_idx,
            out,
            do_scaled,
            l,
            m,
            delta,
        )

        grid_kv = (batch_size, num_heads, num_k_blocks_dkv)
        k_block_bwd_dkv_state_metadata = compute_block_state_metadata(segment_ids, block_k_bwd_dkv)
        out_shapes_kv = [
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ]
        dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_dkv,
                block_q=block_q_bwd_dkv,
                block_d=head_dim,
                block_k=block_k_bwd_dkv,
                sm_scale=sm_scale,
                cap=cap,
                cap_method=cap_method,
                causal=causal,
                window_len=window_len,
                z_loss_weight=z_loss_weight,
            ),
            grid=grid_kv,
            out_shape=out_shapes_kv,
            in_specs=[
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(index_map=lambda j, k, _: (j, 0), block_shape=(None, seq_len)),
                pl.BlockSpec(index_map=lambda j, k, _: (j, 0), block_shape=(None, seq_len)),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0), block_shape=(None, num_k_blocks_dkv)
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, k, 0), block_shape=(None, None, seq_len)
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, k, 0), block_shape=(None, None, seq_len)
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, k, 0), block_shape=(None, None, seq_len)
                ),
            ],
            out_specs=[
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
                pl.BlockSpec(
                    index_map=lambda j, k, _: (j, 0, k, 0),
                    block_shape=(None, seq_len, None, head_dim),
                ),
            ],
            name="mha_backward_kv",
            debug=debug,
            interpret=interpret,
            compiler_params=CompilerParams(num_warps=num_warps, num_stages=num_stages),
        )(q, k, v, temp, segment_ids, k_block_bwd_dkv_state_metadata, out, do_scaled, l, m, delta)
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv, dtemp, dsegment


mha.defvjp(_mha_forward, _mha_backward)
