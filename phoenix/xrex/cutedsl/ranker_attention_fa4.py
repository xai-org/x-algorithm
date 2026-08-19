# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

_FA4_KERNEL_CACHE = {}

_DENSE_BS_LAYOUT_CACHE = {}


def build_dense_block_sparse_layout(batch_size, seq_len, hist_len, num_q_heads):
    key = (batch_size, seq_len, hist_len, num_q_heads)
    if key not in _DENSE_BS_LAYOUT_CACHE:
        import numpy as np

        cand_len = seq_len - hist_len
        assert hist_len % 128 == 0 and cand_len % 128 == 0, (
            f"dense block-sparse needs tile-aligned lengths (hist={hist_len}, cand={cand_len})"
        )
        h_blocks = hist_len // 128
        num_blocks = seq_len // 128
        blocks = np.arange(num_blocks, dtype=np.int32)
        is_cand = blocks >= h_blocks

        fwd_mask_cnt = np.zeros(num_blocks, np.int32)
        fwd_mask_idx = np.zeros((num_blocks, 1), np.int32)
        fwd_full_cnt = np.full(num_blocks, h_blocks, np.int32)
        fwd_full_idx = np.tile(blocks[:h_blocks], (num_blocks, 1))
        fwd_diag_cnt = is_cand.astype(np.int32)
        fwd_diag_idx = (blocks * is_cand).astype(np.int32)[:, None]

        bwd_mask_cnt = np.zeros(num_blocks, np.int32)
        bwd_mask_idx = np.zeros((num_blocks, 1), np.int32)
        bwd_full_cnt = np.where(is_cand, 0, num_blocks).astype(np.int32)
        bwd_full_idx = np.tile(blocks, (num_blocks, 1))
        bwd_full_idx[is_cand] = 0
        bwd_diag_cnt = is_cand.astype(np.int32)
        bwd_diag_idx = (blocks * is_cand).astype(np.int32)[:, None]

        def _bcast(arr):
            return np.ascontiguousarray(np.broadcast_to(arr, (batch_size, num_q_heads) + arr.shape))

        _DENSE_BS_LAYOUT_CACHE[key] = (
            tuple(
                _bcast(a)
                for a in (
                    fwd_mask_cnt,
                    fwd_mask_idx,
                    fwd_full_cnt,
                    fwd_full_idx,
                    fwd_diag_cnt,
                    fwd_diag_idx,
                )
            ),
            tuple(
                _bcast(a)
                for a in (
                    bwd_mask_cnt,
                    bwd_mask_idx,
                    bwd_full_cnt,
                    bwd_full_idx,
                    bwd_diag_cnt,
                    bwd_diag_idx,
                )
            ),
        )
    return _DENSE_BS_LAYOUT_CACHE[key]


def ranker_attention_fa4(q, k, v, sm_scale, block_sparse_layout):
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.jax import cutlass_call

    from xrex.cutedsl.ranker_fa4.block_sparsity import BlockSparseTensors
    from xrex.cutedsl.ranker_fa4.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
    from xrex.cutedsl.ranker_fa4.flash_bwd_sm100 import FlashAttentionBackwardSm100
    from xrex.cutedsl.ranker_fa4.flash_fwd_sm100 import FlashAttentionForwardSm100

    batch_size, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    qhead_per_kvhead = num_q_heads // num_kv_heads
    m_block = 128
    n_block = 128
    hdr = ((head_dim + 31) // 32) * 32
    sr_q = ((seq_len + m_block - 1) // m_block) * m_block
    sr_k = ((seq_len + n_block - 1) // n_block) * n_block
    dKV_postprocess = True

    fwd_bs, bwd_bs = block_sparse_layout
    bs_num_blocks = int(fwd_bs[3].shape[-2])
    bs_max_hist_blocks = int(fwd_bs[3].shape[-1])
    if bs_num_blocks != (seq_len + m_block - 1) // m_block:
        raise ValueError(
            f"block-sparse arrays cover {bs_num_blocks} m-tiles but the kernel "
            f"iterates {(seq_len + m_block - 1) // m_block} (seq_len={seq_len})"
        )
    use_pack_gqa = qhead_per_kvhead > 1 and (m_block % qhead_per_kvhead == 0)

    cache_key = (
        head_dim,
        num_q_heads,
        num_kv_heads,
        batch_size,
        seq_len,
        bs_num_blocks,
        bs_max_hist_blocks,
        use_pack_gqa,
    )

    if cache_key not in _FA4_KERNEL_CACHE:
        fa_fwd = FlashAttentionForwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            pack_gqa=use_pack_gqa,
            is_persistent=True,
            mask_mod=None,
            q_stage=1,
        )

        q_shape = (batch_size, seq_len, num_q_heads, head_dim)
        k_shape = (batch_size, seq_len, num_kv_heads, head_dim)
        lse_shape = (batch_size, num_q_heads, seq_len)

        @cute.jit
        def launch_fwd(
            stream: cuda_driver.CUstream,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mMaskCnt: cute.Tensor,
            mMaskIdx: cute.Tensor,
            mFullCnt: cute.Tensor,
            mFullIdx: cute.Tensor,
            mDiagCnt: cute.Tensor,
            mDiagIdx: cute.Tensor,
            mO: cute.Tensor,
            mLSE: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            bs = BlockSparseTensors(
                mask_block_cnt=mMaskCnt,
                mask_block_idx=mMaskIdx,
                full_block_cnt=mFullCnt,
                full_block_idx=mFullIdx,
                cu_total_m_blocks=None,
                cu_block_idx_offsets=None,
                dq_write_order=None,
                dq_write_order_full=None,
                diag_block_cnt=mDiagCnt,
                diag_block_idx=mDiagIdx,
                dq_write_order_diag=None,
            )
            fa_fwd(
                mQ,
                mK,
                mV,
                mO,
                mLSE,
                softmax_scale,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                bs,
                None,
                stream,
            )

        fwd_call = cutlass_call(
            launch_fwd,
            output_shape_dtype=[
                jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
                jax.ShapeDtypeStruct(lse_shape, jnp.float32),
            ],
            use_static_tensors=False,
            softmax_scale=cutlass.Float32(sm_scale),
        )

        fa_bwd = FlashAttentionBackwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            mask_mod=None,
        )
        dq_accum_shape = (batch_size, num_q_heads, sr_q * hdr)

        if dKV_postprocess:
            dk_accum_shape = (batch_size, num_kv_heads, sr_k * hdr)

            @cute.jit
            def launch_bwd(
                stream: cuda_driver.CUstream,
                mQ: cute.Tensor,
                mK: cute.Tensor,
                mV: cute.Tensor,
                mdO: cute.Tensor,
                mLSE: cute.Tensor,
                mdPsum: cute.Tensor,
                mMaskCnt: cute.Tensor,
                mMaskIdx: cute.Tensor,
                mFullCnt: cute.Tensor,
                mFullIdx: cute.Tensor,
                mDiagCnt: cute.Tensor,
                mDiagIdx: cute.Tensor,
                mdQa: cute.Tensor,
                mdKa: cute.Tensor,
                mdVa: cute.Tensor,
                softmax_scale: cutlass.Float32,
            ):
                bs = BlockSparseTensors(
                    mask_block_cnt=mMaskCnt,
                    mask_block_idx=mMaskIdx,
                    full_block_cnt=mFullCnt,
                    full_block_idx=mFullIdx,
                    cu_total_m_blocks=None,
                    cu_block_idx_offsets=None,
                    dq_write_order=None,
                    dq_write_order_full=None,
                    diag_block_cnt=mDiagCnt,
                    diag_block_idx=mDiagIdx,
                    dq_write_order_diag=None,
                )
                fa_bwd(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    mdQa,
                    mdKa,
                    mdVa,
                    softmax_scale,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    bs,
                    stream,
                )

            bwd_call = cutlass_call(
                launch_bwd,
                output_shape_dtype=[
                    jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32),
                    jax.ShapeDtypeStruct(dk_accum_shape, jnp.float32),
                    jax.ShapeDtypeStruct(dk_accum_shape, jnp.float32),
                ],
                input_output_aliases={12: 0, 13: 1, 14: 2},
                use_static_tensors=False,
                softmax_scale=cutlass.Float32(sm_scale),
            )
        else:
            raise AssertionError("block-sparse backward requires dKV_postprocess")

        fa_post_dq = FlashAttentionBackwardPostprocess(
            cutlass.BFloat16, head_dim, 100, tile_m=m_block, num_threads=128
        )

        @cute.jit
        def launch_post_dq(
            stream: cuda_driver.CUstream,
            mAccum: cute.Tensor,
            mOut: cute.Tensor,
            scale: cutlass.Float32,
        ):
            fa_post_dq(mAccum, mOut, scale, None, None, stream)

        post_dq_call = cutlass_call(
            launch_post_dq,
            output_shape_dtype=[jax.ShapeDtypeStruct(q_shape, jnp.bfloat16)],
            use_static_tensors=False,
            scale=cutlass.Float32(sm_scale),
        )

        post_dk_call = None
        post_dv_call = None
        if dKV_postprocess:
            fa_post_dk = FlashAttentionBackwardPostprocess(
                cutlass.BFloat16, head_dim, 100, tile_m=n_block, num_threads=128
            )

            @cute.jit
            def launch_post_dk(
                stream: cuda_driver.CUstream,
                mAccum: cute.Tensor,
                mOut: cute.Tensor,
                scale: cutlass.Float32,
            ):
                fa_post_dk(mAccum, mOut, scale, None, None, stream)

            post_dk_call = cutlass_call(
                launch_post_dk,
                output_shape_dtype=[jax.ShapeDtypeStruct(k_shape, jnp.bfloat16)],
                use_static_tensors=False,
                scale=cutlass.Float32(sm_scale),
            )

            fa_post_dv = FlashAttentionBackwardPostprocess(
                cutlass.BFloat16, head_dim, 100, tile_m=n_block, num_threads=128
            )

            @cute.jit
            def launch_post_dv(
                stream: cuda_driver.CUstream,
                mAccum: cute.Tensor,
                mOut: cute.Tensor,
                scale: cutlass.Float32,
            ):
                fa_post_dv(mAccum, mOut, scale, None, None, stream)

            post_dv_call = cutlass_call(
                launch_post_dv,
                output_shape_dtype=[jax.ShapeDtypeStruct(k_shape, jnp.bfloat16)],
                use_static_tensors=False,
                scale=cutlass.Float32(1.0),
            )

        _FA4_KERNEL_CACHE[cache_key] = dict(
            fwd_call=fwd_call,
            bwd_call=bwd_call,
            post_dq_call=post_dq_call,
            post_dk_call=post_dk_call,
            post_dv_call=post_dv_call,
            dKV_postprocess=dKV_postprocess,
            dq_accum_shape=dq_accum_shape,
        )

    c = _FA4_KERNEL_CACHE[cache_key]

    _fwd_extra = tuple(fwd_bs)
    _bwd_extra = tuple(bwd_bs)

    @jax.custom_vjp
    def _attention(q, k, v):
        out, _lse = c["fwd_call"](q, k, v, *_fwd_extra)
        return out

    def _attention_fwd(q, k, v):
        out, lse = c["fwd_call"](q, k, v, *_fwd_extra)
        out = checkpoint_name(out, "attn_outputs")
        lse = checkpoint_name(lse, "attn_outputs")
        return out, (q, k, v, out, lse)

    def _attention_bwd(res, g):
        q, k, v, out, lse = res

        dpsum = jnp.sum(out.astype(jnp.float32) * g.astype(jnp.float32), axis=-1).transpose(0, 2, 1)
        if dpsum.shape[-1] < sr_q:
            dpsum = jnp.pad(dpsum, ((0, 0), (0, 0), (0, sr_q - dpsum.shape[-1])))
        lse_log2 = lse * jnp.float32(math.log2(math.e))
        if lse_log2.shape[-1] < sr_q:
            lse_log2 = jnp.pad(lse_log2, ((0, 0), (0, 0), (0, sr_q - lse_log2.shape[-1])))

        dq_accum_init = jnp.zeros(c["dq_accum_shape"], dtype=jnp.float32)

        dk_accum_init = jnp.zeros((batch_size, num_kv_heads, sr_k * hdr), dtype=jnp.float32)
        dv_accum_init = jnp.zeros_like(dk_accum_init)
        dq_accum, dk_accum, dv_accum = c["bwd_call"](
            q, k, v, g, lse_log2, dpsum, *_bwd_extra, dq_accum_init, dk_accum_init, dv_accum_init
        )
        (dq,) = c["post_dq_call"](dq_accum)
        (dk,) = c["post_dk_call"](dk_accum)
        (dv,) = c["post_dv_call"](dv_accum)

        return dq, dk, dv

    _attention.defvjp(_attention_fwd, _attention_bwd)
    return _attention(q, k, v)
