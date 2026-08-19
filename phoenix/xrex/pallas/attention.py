# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas.triton import CompilerParams


def tanh(x):
    return 2 * jax.lax.logistic(2 * x) - 1


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    temp_ref,
    segment_ref,
    o_ref,
    *residual_refs,
    sm_scale: float,
    cap: float,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q: int,
    block_d: int,
    block_k: int,
):
    seq_len = q_ref.shape[0]
    start_q = pl.program_id(0)

    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    acc = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    q = q_ref[pl.dslice(start_q * block_q, block_q), :]

    seg_q = segment_ref[pl.dslice(start_q * block_q, block_q)]
    seg_q = jnp.expand_dims(seg_q, axis=-1)

    def body(start_k, carry):
        acc, m_prev, l_prev = carry

        k = k_ref[slice(None), pl.dslice(start_k * block_k, block_k)]
        seg_k = segment_ref[pl.dslice(start_k * block_k, block_k)]
        temp = temp_ref[pl.dslice(start_q * block_q, block_q)]
        temp = jnp.expand_dims(temp, axis=-1)
        mask = jnp.equal(seg_q, jnp.expand_dims(seg_k, axis=-2))
        qk = jnp.zeros([block_q, block_k], dtype=jnp.float32)
        qk += pl.dot(q, k)
        if sm_scale != 1.0:
            qk *= sm_scale

        if cap > 0.0:
            qk = cap * tanh(qk / cap)
        qk = qk * temp
        if causal:
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_k + jnp.arange(block_k)
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask = jnp.logical_and(causal_mask, mask)

        if window_len > 0:
            window_mask = span_k[None, :] > span_q[:, None] - window_len
            mask = jnp.logical_and(mask, window_mask)

        qk = jnp.where(mask, qk, float("-inf"))
        m_curr = jnp.maximum(jnp.max(qk, axis=1), m_prev)
        l_new = jnp.exp(m_prev - m_curr)
        l_new = jax.lax.select(jnp.isnan(l_new), jnp.ones_like(l_new), l_new)
        l_prev *= l_new

        p = jnp.exp(qk - m_curr[:, None])
        p = jax.lax.select(jnp.isnan(p), jnp.ones_like(p), p)
        l_curr = jnp.sum(p, axis=1) + l_prev

        l_rcp = 1.0 / l_curr
        p = p * l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        p = p.astype(q.dtype)

        v = v_ref[pl.dslice(start_k * block_k, block_k), pl.dslice(block_d)]
        acc += pl.dot(p, v)
        return acc, m_curr, l_curr

    if causal:
        upper_bound = lax.div(block_q * start_q, block_k) + 1
    else:
        upper_bound = pl.cdiv(seq_len, block_k)

    if window_len > 0:
        max_num_blocks = lax.div(window_len + block_q, block_k)
        lower_bound = jnp.maximum(upper_bound - max_num_blocks, 0)
    else:
        lower_bound = 0

    acc, m_i, l_i = lax.fori_loop(lower_bound, upper_bound, body, (acc, m_i, l_i))

    if residual_refs:
        l_ref, m_ref = residual_refs
        l_ref[pl.ds(start_q * block_q, block_q)] = l_i
        m_ref[pl.ds(start_q * block_q, block_q)] = m_i
    acc = acc.astype(o_ref.dtype)
    o_ref[pl.dslice(start_q * block_q, block_q), :] = acc


@functools.partial(jax.custom_vjp, nondiff_argnums=list(range(5, 18)))
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "cap",
        "causal",
        "window_len",
        "z_loss_weight",
        "block_q",
        "block_k",
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
    causal: bool = True,
    window_len: int = -1,
    z_loss_weight: float = 0.0,
    block_q: int = 128,
    block_k: int = 128,
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid=None,
    interpret: bool = False,
    debug: bool = False,
):
    del backward_pass_impl
    batch_size, seq_len, num_heads, head_dim = q.shape
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        cap=cap,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
        causal=causal,
        window_len=window_len,
        z_loss_weight=z_loss_weight,
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
            pl.BlockSpec(
                index_map=lambda _, j, k: (
                    j,
                    0,
                ),
                block_shape=(
                    None,
                    seq_len,
                ),
            ),
        ],
        out_specs=pl.BlockSpec(
            index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
        ),
        compiler_params=CompilerParams(num_warps=num_warps_, num_stages=num_stages),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k.swapaxes(1, 3), v, temp, segment_ids)


def _mha_forward(
    q,
    k,
    v,
    temp,
    segment_ids,
    sm_scale: float,
    cap: float,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q: int,
    block_k: int,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
    del backward_pass_impl
    batch_size, seq_len, num_heads, head_dim = q.shape
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        cap=cap,
        causal=causal,
        window_len=window_len,
        z_loss_weight=z_loss_weight,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
    )
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), dtype=jnp.float32),
    ]
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
            pl.BlockSpec(
                index_map=lambda _, j, k: (
                    j,
                    0,
                ),
                block_shape=(
                    None,
                    seq_len,
                ),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(index_map=lambda _, j, k: (j, k, 0), block_shape=(None, None, seq_len)),
            pl.BlockSpec(index_map=lambda _, j, k: (j, k, 0), block_shape=(None, None, seq_len)),
        ],
        compiler_params=CompilerParams(num_warps=num_warps_, num_stages=num_stages),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k.swapaxes(1, 3), v, temp, segment_ids)
    return out, (q, k, v, temp, segment_ids, out, l, m)


def _preprocess_backward_kernel(out_ref, dout_ref, l_ref, new_dout_ref, delta_ref, *, block_q: int):
    pid_m = pl.program_id(0)

    off_m = pl.ds(pid_m * block_q, block_q)
    o = pl.load(out_ref, (off_m, slice(None))).astype(jnp.float32)
    do = pl.load(dout_ref, (off_m, slice(None))).astype(jnp.float32)
    denom = pl.load(l_ref, (off_m,)).astype(jnp.float32)
    do = do / denom[:, None]
    delta = jnp.sum(o * do, axis=1)
    pl.store(new_dout_ref, (off_m, slice(None)), do.astype(new_dout_ref.dtype))
    pl.store(delta_ref, (off_m,), delta.astype(delta_ref.dtype))


def _preprocess_backward(out, do, l, block_q: int, debug: bool, interpret: bool):
    batch_size, seq_len, num_heads, head_dim = out.shape
    out_shape = [
        jax.ShapeDtypeStruct(do.shape, do.dtype),
        jax.ShapeDtypeStruct(l.shape, l.dtype),
    ]
    do_scaled, delta = pl.pallas_call(
        functools.partial(_preprocess_backward_kernel, block_q=block_q),
        grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
        in_specs=[
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(index_map=lambda _, j, k: (j, k, 0), block_shape=(None, None, seq_len)),
        ],
        out_specs=[
            pl.BlockSpec(
                index_map=lambda _, j, k: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(index_map=lambda _, j, k: (j, k, 0), block_shape=(None, None, seq_len)),
        ],
        compiler_params=CompilerParams(num_warps=8, num_stages=2),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_preprocess_backward",
    )(out, do, l)
    return do_scaled, delta


def mha_backward_kernel_dq(
    q_ref,
    k_ref,
    v_ref,
    temp_ref,
    segment_ref,
    out_ref,
    do_scaled_ref,
    l_ref,
    m_ref,
    delta_ref,
    dq_ref,
    *,
    sm_scale: float,
    cap: float,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q: int,
    block_d: int,
    block_k: int,
):
    del out_ref
    seq_len = q_ref.shape[0]
    start_q = pl.program_id(2)
    q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
    span_q = start_q * block_q + jnp.arange(block_q)
    m = pl.load(m_ref, (pl.ds(start_q * block_q, block_q),))
    l = pl.load(l_ref, (pl.ds(start_q * block_q, block_q),))
    do = pl.load(do_scaled_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
    di = pl.load(delta_ref, (pl.ds(start_q * block_q, block_q),))
    dq = jnp.zeros([block_q, block_d], dtype=jnp.float32)
    seg_q = pl.load(
        segment_ref,
        (pl.ds(start_q * block_q, block_q),),
    )
    seg_q = jnp.expand_dims(seg_q, axis=-1)

    def inner_loop(start_k, dq):
        k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
        v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
        seg_k = pl.load(
            segment_ref,
            (pl.dslice(start_k * block_k, block_k),),
        )
        mask = jnp.equal(seg_q, jnp.expand_dims(seg_k, axis=-2))
        temp = pl.load(temp_ref, (pl.dslice(start_q * block_q, block_q),))
        temp = jnp.expand_dims(temp, axis=-1)
        qk = jnp.zeros((block_q, block_k), dtype=jnp.float32)
        qk += pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        if cap > 0.0:
            qk_tanh = tanh(qk / cap)
            qk = cap * qk_tanh
        qk *= temp
        if causal:
            span_k = start_k * block_k + jnp.arange(block_k)
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask = jnp.logical_and(causal_mask, mask)
        if window_len > 0:
            window_mask = span_k[None, :] > span_q[:, None] - window_len
            mask = jnp.logical_and(mask, window_mask)
        qk = jnp.where(mask, qk, float("-inf"))
        p = jnp.exp(qk - m[:, None])
        dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
        ds = p * dp
        if z_loss_weight > 0:
            ds += z_loss_weight * p * ((jnp.log(l + 1e-12) + m) / l)[:, None]
        ds *= temp
        if cap > 0.0:
            ds = ds * (1 - qk_tanh**2)
        if sm_scale != 1.0:
            ds = ds * sm_scale
        dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
        return dq

    if causal:
        upper_bound = lax.div(start_q * block_q, block_k) + 1
    else:
        upper_bound = pl.cdiv(seq_len, block_k)

    if window_len > 0:
        max_num_blocks = lax.div(window_len + block_q, block_k)
        lower_bound = jnp.maximum(upper_bound - max_num_blocks, 0)
    else:
        lower_bound = 0
    dq = lax.fori_loop(lower_bound, upper_bound, inner_loop, dq)
    pl.store(dq_ref, (pl.ds(start_q * block_q, block_q), slice(None)), dq)


def mha_backward_kernel_dkv(
    q_ref,
    k_ref,
    v_ref,
    temp_ref,
    segment_ref,
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
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q: int,
    block_d: int,
    block_k: int,
):
    del out_ref
    seq_len = q_ref.shape[0]
    start_k = pl.program_id(2)

    dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
    v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
    span_k = start_k * block_k + jnp.arange(block_k)
    seg_k = pl.load(
        segment_ref,
        (pl.ds(start_k * block_k, block_k),),
    )
    seg_k = jnp.expand_dims(seg_k, axis=-2)

    def inner_loop(start_q, carry):
        dv, dk = carry
        q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
        qk = jnp.zeros((block_q, block_k), dtype=jnp.float32)
        qk += pl.dot(q, k.T)
        seg_q = pl.load(
            segment_ref,
            (pl.ds(start_q * block_q, block_q),),
        )
        mask = jnp.equal(jnp.expand_dims(seg_q, axis=-1), seg_k)
        temp = pl.load(temp_ref, (pl.dslice(start_q * block_q, block_q),))
        temp = jnp.expand_dims(temp, axis=-1)

        if sm_scale != 1.0:
            qk *= sm_scale
        if cap > 0.0:
            qk_tanh = tanh(qk / cap)
            qk = cap * qk_tanh
        qk *= temp
        if causal:
            span_q = start_q * block_q + jnp.arange(block_q)
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask = jnp.logical_and(causal_mask, mask)
        if window_len > 0:
            window_mask = span_k[None, :] > span_q[:, None] - window_len
            mask = jnp.logical_and(mask, window_mask)
        qk = jnp.where(mask, qk, float("-inf"))
        m = pl.load(m_ref, (pl.ds(start_q * block_q, block_q),))
        p = jnp.exp(qk - m[:, None])
        do = pl.load(do_scaled_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
        dv = dv + pl.dot(p.astype(do.dtype).T, do)
        di = pl.load(delta_ref, (pl.ds(start_q * block_q, block_q),))
        dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
        ds = p * dp
        if z_loss_weight > 0:
            l = pl.load(l_ref, (pl.ds(start_q * block_q, block_q),))
            ds += z_loss_weight * p * ((jnp.log(l + 1e-12) + m) / l)[:, None]
        ds *= temp
        if cap > 0.0:
            ds = ds * (1 - qk_tanh**2)
        if sm_scale != 1.0:
            ds = ds * sm_scale
        dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)
        return dv, dk

    if causal:
        lower_bound = lax.div(start_k * block_k, block_q)
    else:
        lower_bound = 0
    if window_len > 0:
        max_num_blocks = lax.div(window_len + block_k, block_q)
        upper_bound = jnp.minimum(lower_bound + max_num_blocks, pl.cdiv(seq_len, block_q))
    else:
        upper_bound = pl.cdiv(seq_len, block_q)
    dv, dk = lax.fori_loop(lower_bound, upper_bound, inner_loop, (dv, dk))
    pl.store(dv_ref, (pl.ds(start_k * block_k, block_k), slice(None)), dv.astype(dv_ref.dtype))
    pl.store(dk_ref, (pl.ds(start_k * block_k, block_k), slice(None)), dk.astype(dk_ref.dtype))


def _mha_backward(
    sm_scale: float,
    cap: float,
    causal: bool,
    window_len: int,
    z_loss_weight: float,
    block_q: int,
    block_k: int,
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
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    do_scaled, delta = _preprocess_backward(out, do, l, block_q, debug, interpret)

    if backward_pass_impl == "triton_split":
        out_shapes_q = jax.ShapeDtypeStruct(q.shape, jnp.float32)
        dtemp = jnp.zeros(temp.shape, temp.dtype)
        dsegment = jnp.zeros(segment_ids.shape, q.dtype)

        grid_q = (batch_size, num_heads, pl.cdiv(seq_len, block_q))
        num_warps = 8
        dq = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_dq,
                block_q=block_q,
                block_d=head_dim,
                block_k=block_k,
                sm_scale=sm_scale,
                cap=cap,
                causal=causal,
                window_len=window_len,
                z_loss_weight=z_loss_weight,
            ),
            grid=grid_q,
            out_shape=out_shapes_q,
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
                pl.BlockSpec(
                    index_map=lambda j, k, _: (
                        j,
                        0,
                    ),
                    block_shape=(None, seq_len),
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
            out_specs=pl.BlockSpec(
                index_map=lambda j, k, _: (j, 0, k, 0), block_shape=(None, seq_len, None, head_dim)
            ),
            name="mha_backward_q",
            debug=debug,
            interpret=interpret,
            compiler_params=CompilerParams(num_warps=num_warps, num_stages=2),
        )(q, k, v, temp, segment_ids, out, do_scaled, l, m, delta)

        grid_kv = (batch_size, num_heads, pl.cdiv(seq_len, block_k))
        out_shapes_kv = [
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ]
        dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_dkv,
                block_q=block_q,
                block_d=head_dim,
                block_k=block_k,
                sm_scale=sm_scale,
                cap=cap,
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
                pl.BlockSpec(
                    index_map=lambda j, k, _: (
                        j,
                        0,
                    ),
                    block_shape=(None, seq_len),
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
            compiler_params=CompilerParams(num_warps=num_warps, num_stages=2),
        )(q, k, v, temp, segment_ids, out, do_scaled, l, m, delta)
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv, dtemp, dsegment


mha.defvjp(_mha_forward, _mha_backward)
