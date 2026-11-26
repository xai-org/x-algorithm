# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import operator

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from xrex.utils import gpu

_VEC_WIDTH = 8
_CTAS_PER_SM = 3
_MAX_THREADS_PER_CTA = 1024
_WARP_SIZE = 32
_NUM_ROW_SUMS = 4


def qk_norm_rope_backward_supported(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: DTypeLike,
    device: jax.Device,
) -> bool:
    return (
        gpu.cuda_compute_capability(device) == (9, 0)
        and dtype == jnp.bfloat16
        and _shape_fits_kernel(num_tokens, num_q_heads, num_kv_heads, head_dim)
    )


def _shape_fits_kernel(num_tokens: int, num_q_heads: int, num_kv_heads: int, head_dim: int) -> bool:
    return (
        num_q_heads % num_kv_heads == 0
        and head_dim % (2 * _VEC_WIDTH) == 0
        and _num_threads(num_q_heads, num_kv_heads, head_dim) <= _MAX_THREADS_PER_CTA
        and num_tokens * (num_q_heads + 2 * num_kv_heads) * head_dim < 2**31
    )


def _num_threads(num_q_heads: int, num_kv_heads: int, head_dim: int) -> int:
    num_lane_groups = (num_q_heads + num_kv_heads) * (head_dim // (2 * _VEC_WIDTH))
    return -(-num_lane_groups // _WARP_SIZE) * _WARP_SIZE


def qk_norm_rope_backward(
    dq: jax.Array,
    q: jax.Array,
    dk: jax.Array,
    k: jax.Array,
    dv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    q_gain: jax.Array,
    k_gain: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert jax.default_backend() == "gpu", "qk_norm_rope_backward needs a GPU (CUDA kernel)"
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    assert _shape_fits_kernel(num_tokens, num_q_heads, num_kv_heads, head_dim), (q.shape, k.shape)
    kv_shape = (num_tokens, num_kv_heads, head_dim)
    assert dq.shape == q.shape and dk.shape == k.shape == dv.shape == kv_shape
    assert {x.dtype for x in (dq, q, dk, k, dv)} == {jnp.dtype(jnp.bfloat16)}
    assert cos.shape == sin.shape == (num_tokens, head_dim // 2), f"cos and sin {cos.shape}"
    assert q_gain.shape == (num_q_heads * head_dim,), f"q_gain {q_gain.shape}"
    assert k_gain.shape == (num_kv_heads * head_dim,), f"k_gain {k_gain.shape}"
    assert {x.dtype for x in (cos, sin, q_gain, k_gain)} == {jnp.dtype(jnp.float32)}
    num_q_per_kv = num_q_heads // num_kv_heads
    call = _compiled_kernel(num_tokens, num_kv_heads, num_q_per_kv, head_dim, float(eps))
    dqkv, dq_gain, dk_gain = call(
        *(x.reshape(num_tokens, -1) for x in (dq, q, dk, k, dv)), cos, sin, q_gain, k_gain
    )
    dqkv = dqkv.reshape(num_tokens, num_kv_heads, num_q_per_kv + 2, head_dim)
    return dqkv, dq_gain.sum(axis=0), dk_gain.sum(axis=0)


@functools.cache
def _compiled_kernel(
    num_tokens: int, num_kv_heads: int, num_q_per_kv: int, head_dim: int, eps: float
):
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, const_expr
    from cutlass.jax import cutlass_call

    from xrex.cutedsl.ranker_fa4.utils import warp_reduce

    num_q_heads = num_kv_heads * num_q_per_kv
    half_dim = head_dim // 2
    vecs_per_half_head = half_dim // _VEC_WIDTH
    q_threads = num_q_heads * vecs_per_half_head
    k_threads = num_kv_heads * vecs_per_half_head
    v_threads = num_kv_heads * head_dim // _VEC_WIDTH
    num_threads = _num_threads(num_q_heads, num_kv_heads, head_dim)
    num_warps = num_threads // _WARP_SIZE
    num_ctas = min(
        num_tokens, cutlass.utils.HardwareInfo().get_device_multiprocessor_count() * _CTAS_PER_SM
    )
    slots_per_kv_head = num_q_per_kv + 2
    q_width, k_width = float(num_q_heads * head_dim), float(num_kv_heads * head_dim)

    def tile(tensor, row, col):
        return cute.local_tile(tensor, (1, _VEC_WIDTH), (row, col // _VEC_WIDTH))

    def make_fragment(dtype=Float32):
        return cute.make_rmem_tensor((1, _VEC_WIDTH), dtype)

    def load_tile(tensor, row, col, dst):
        if const_expr(dst.element_type == tensor.element_type):
            cute.autovec_copy(tile(tensor, row, col), dst)
        else:
            src = make_fragment(tensor.element_type)
            cute.autovec_copy(tile(tensor, row, col), src)
            dst.store(src.load().to(dst.element_type))

    def store_tile(value, tensor, row, col):
        src = make_fragment(tensor.element_type)
        src.store(value.to(tensor.element_type))
        cute.autovec_copy(src, tile(tensor, row, col))

    def load_f32(fragment):
        return fragment.load().to(Float32)

    def reduce_sum(values):
        return values.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

    def load_rope_pair(dy, x, row, col, d_lo, d_hi, x_lo, x_hi):
        load_tile(dy, row, col, d_lo)
        load_tile(dy, row, col + half_dim, d_hi)
        load_tile(x, row, col, x_lo)
        load_tile(x, row, col + half_dim, x_hi)

    def rope_transpose(cos, sin, d_lo, d_hi):
        dy_lo, dy_hi = load_f32(d_lo), load_f32(d_hi)
        d_lo.store((dy_hi * sin.load() + dy_lo * cos.load()).to(d_lo.element_type))
        d_hi.store((dy_hi * cos.load() - dy_lo * sin.load()).to(d_hi.element_type))

    def partial_row_sums(d_lo, d_hi, x_lo, x_hi, g_lo, g_hi):
        x_lo_f32, x_hi_f32 = load_f32(x_lo), load_f32(x_hi)
        dot = reduce_sum(x_lo_f32 * (g_lo.load() * load_f32(d_lo)))
        dot += reduce_sum(x_hi_f32 * (g_hi.load() * load_f32(d_hi)))
        return dot, reduce_sum(x_lo_f32 * x_lo_f32) + reduce_sum(x_hi_f32 * x_hi_f32)

    def norm_transpose(rope_dy, x, gain, dot, sq_sum, width, out, row, col, dgain):
        mean_sq = sq_sum / width + eps
        inv_rms = cute.math.rsqrt(mean_sq)
        d_mean_sq = dot * (-0.5 * (inv_rms / mean_sq))
        d32, x32 = load_f32(rope_dy), load_f32(x)
        store_tile((gain.load() * d32) * inv_rms + x32 * ((d_mean_sq / width) * 2.0), out, row, col)
        dgain.store(dgain.load() + d32 * (x32 * inv_rms))

    @cute.kernel
    def kernel(mdQ, mQ, mdK, mK, mdV, mCos, mSin, mQG, mKG, mdQKV, mdQG, mdKG):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp, lane = tidx // _WARP_SIZE, tidx % _WARP_SIZE
        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_tensor(
            Float32, cute.make_layout((_NUM_ROW_SUMS, num_warps, 2)), byte_alignment=16
        )

        is_q = tidx < q_threads
        is_k = (tidx >= q_threads) & (tidx < q_threads + k_threads)
        is_v = tidx < v_threads
        lane_group = tidx if tidx < q_threads else tidx - q_threads
        head = lane_group // vecs_per_half_head
        col = (lane_group % vecs_per_half_head) * _VEC_WIDTH
        in_col = head * head_dim + col
        q_out = ((head // num_q_per_kv) * slots_per_kv_head + head % num_q_per_kv) * head_dim + col
        k_out = (head * slots_per_kv_head + num_q_per_kv) * head_dim + col
        v_head = tidx // (head_dim // _VEC_WIDTH)
        v_col = (tidx % (head_dim // _VEC_WIDTH)) * _VEC_WIDTH
        v_out = (v_head * slots_per_kv_head + num_q_per_kv + 1) * head_dim + v_col

        g_lo, g_hi, dg_lo, dg_hi, cos, sin = (make_fragment() for _ in range(6))
        for fragment in (g_lo, g_hi, dg_lo, dg_hi):
            fragment.fill(0.0)
        if is_q:
            load_tile(mQG, 0, in_col, g_lo)
            load_tile(mQG, 0, in_col + half_dim, g_hi)
        if is_k:
            load_tile(mKG, 0, in_col, g_lo)
            load_tile(mKG, 0, in_col + half_dim, g_hi)
        d_lo, d_hi, x_lo, x_hi, v = (make_fragment(cutlass.BFloat16) for _ in range(5))

        parity = cutlass.Int32(0)
        for row in cutlass.range(bidx, mdQ.shape[0], num_ctas, unroll=1):
            load_tile(mCos, row, col, cos)
            load_tile(mSin, row, col, sin)
            dot_q, sq_q, dot_k, sq_k = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            if is_q:
                load_rope_pair(mdQ, mQ, row, in_col, d_lo, d_hi, x_lo, x_hi)
                rope_transpose(cos, sin, d_lo, d_hi)
                dot_q, sq_q = partial_row_sums(d_lo, d_hi, x_lo, x_hi, g_lo, g_hi)
            if is_k:
                load_rope_pair(mdK, mK, row, in_col, d_lo, d_hi, x_lo, x_hi)
                rope_transpose(cos, sin, d_lo, d_hi)
                dot_k, sq_k = partial_row_sums(d_lo, d_hi, x_lo, x_hi, g_lo, g_hi)
            if is_v:
                load_tile(mdV, row, v_head * head_dim + v_col, v)
                cute.autovec_copy(v, tile(mdQKV, row, v_out))
            sums = [warp_reduce(x, operator.add) for x in (dot_q, sq_q, dot_k, sq_k)]
            if lane == 0:
                for i in cutlass.range_constexpr(_NUM_ROW_SUMS):
                    warp_sums[i, warp, parity] = sums[i]
            cute.arch.barrier()
            sums = [warp_sums[i, 0, parity] for i in range(_NUM_ROW_SUMS)]
            for other_warp in cutlass.range_constexpr(1, num_warps):
                for i in cutlass.range_constexpr(_NUM_ROW_SUMS):
                    sums[i] += warp_sums[i, other_warp, parity]
            dot_q, sq_q, dot_k, sq_k = sums
            if is_q:
                norm_transpose(d_lo, x_lo, g_lo, dot_q, sq_q, q_width, mdQKV, row, q_out, dg_lo)
                norm_transpose(
                    d_hi, x_hi, g_hi, dot_q, sq_q, q_width, mdQKV, row, q_out + half_dim, dg_hi
                )
            if is_k:
                norm_transpose(d_lo, x_lo, g_lo, dot_k, sq_k, k_width, mdQKV, row, k_out, dg_lo)
                norm_transpose(
                    d_hi, x_hi, g_hi, dot_k, sq_k, k_width, mdQKV, row, k_out + half_dim, dg_hi
                )
            parity ^= 1

        if is_q:
            store_tile(dg_lo.load(), mdQG, bidx, in_col)
            store_tile(dg_hi.load(), mdQG, bidx, in_col + half_dim)
        if is_k:
            store_tile(dg_lo.load(), mdKG, bidx, in_col)
            store_tile(dg_hi.load(), mdKG, bidx, in_col + half_dim)

    @cute.jit
    def launch(
        stream: cuda_driver.CUstream, mdQ, mQ, mdK, mK, mdV, mCos, mSin, mQG, mKG, mdQKV, mdQG, mdKG
    ):
        mQG, mKG = (
            cute.make_tensor(gain.iterator, cute.make_layout((1, gain.shape[0])))
            for gain in (mQG, mKG)
        )
        kernel(mdQ, mQ, mdK, mK, mdV, mCos, mSin, mQG, mKG, mdQKV, mdQG, mdKG).launch(
            grid=[num_ctas, 1, 1],
            block=[num_threads, 1, 1],
            min_blocks_per_mp=_CTAS_PER_SM,
            stream=stream,
        )

    return cutlass_call(
        launch,
        output_shape_dtype=[
            jax.ShapeDtypeStruct(
                (num_tokens, num_kv_heads * slots_per_kv_head * head_dim), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct((num_ctas, num_q_heads * head_dim), jnp.float32),
            jax.ShapeDtypeStruct((num_ctas, num_kv_heads * head_dim), jnp.float32),
        ],
        use_static_tensors=True,
    )
