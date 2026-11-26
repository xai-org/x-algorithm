# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

_FA4_KERNEL_CACHE = {}


def cutedsl_arch():
    major = int(str(jax.devices()[0].compute_capability).split(".")[0])
    if major == 8:
        return 80
    if major == 9:
        return 90
    if major in (10, 11):
        return 100
    raise NotImplementedError(f"cutedsl ranker attention: unsupported compute capability {major}.x")


def attention_delta(out, grad_out, padded_len):
    batch, seq_len, heads, dim = out.shape
    grad_out = grad_out.astype(out.dtype)
    pad = ((0, 0), (0, padded_len - seq_len), (0, 0), (0, 0))
    out_tiles = jnp.pad(out, pad).reshape(-1, 32, dim)
    grad_tiles = jnp.pad(grad_out, pad).reshape(-1, 32, dim)
    dots = jnp.einsum(
        "bid,bjd->bij",
        out_tiles,
        grad_tiles,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    return (
        jnp.diagonal(dots, axis1=-2, axis2=-1).reshape(batch, padded_len, heads).transpose(0, 2, 1)
    )


def pad_softmax_stats(softmax_stats, padded_len):
    if not softmax_stats or softmax_stats[0].shape[-1] == padded_len:
        return softmax_stats
    pad = ((0, 0), (0, 0), (0, padded_len - softmax_stats[0].shape[-1]))
    row_max, inv_sum = softmax_stats
    return jnp.pad(row_max, pad, constant_values=jnp.inf), jnp.pad(inv_sum, pad)


def build_dense_block_sparse_layout(seq_len, hist_len, num_q_heads, hist_valid_len):
    block = 128
    cand_len = seq_len - hist_len
    assert hist_len % block == 0 and cand_len >= 0, (
        f"dense block-sparse needs tile-aligned history and seq_len >= hist_len "
        f"(hist={hist_len}, cand={cand_len})"
    )
    h_blocks = hist_len // block
    num_blocks = (seq_len + block - 1) // block
    cand_blocks = num_blocks - h_blocks
    idx_w = max(h_blocks, 1)

    valid_len = jnp.asarray(hist_valid_len, jnp.int32).reshape(-1)[:, None]
    batch_size = valid_len.shape[0]
    n_full = valid_len // block
    has_partial = (valid_len % block != 0).astype(jnp.int32)
    t = jnp.arange(num_blocks, dtype=jnp.int32)[None, :]
    is_hist = t < h_blocks
    is_cand_i32 = (~is_hist).astype(jnp.int32)
    is_full = ~is_hist | (t < n_full)
    is_partial = is_hist & (t == n_full) & (has_partial == 1)

    j_h = jnp.arange(idx_w, dtype=jnp.int32)[None, None, :]
    j_n = jnp.arange(num_blocks, dtype=jnp.int32)[None, None, :]
    n_full3 = n_full[:, :, None]

    fwd_mask_cnt = jnp.where(is_partial, n_full + 1, jnp.where(is_full, has_partial, 0))
    fwd_mask_idx = jnp.where(is_partial[:, :, None], j_h, n_full3)
    fwd_full_cnt = jnp.where(is_full, n_full, 0)
    fwd_full_idx = jnp.broadcast_to(j_h, (batch_size, num_blocks, idx_w))
    fwd_diag_cnt = jnp.broadcast_to(is_cand_i32, (batch_size, num_blocks))
    fwd_diag_idx = jnp.broadcast_to((t * is_cand_i32)[:, :, None], (batch_size, num_blocks, 1))

    full_q_seq = jnp.clip(
        jnp.where(j_n < n_full3, j_n, h_blocks + j_n - n_full3), 0, num_blocks - 1
    )
    mask_q_seq = jnp.clip(
        jnp.where(j_n <= n_full3, j_n, h_blocks + j_n - n_full3 - 1), 0, num_blocks - 1
    )
    bwd_mask_cnt = jnp.where(
        is_partial, n_full + cand_blocks + 1, jnp.where(t < n_full, has_partial, 0)
    )
    bwd_mask_idx = jnp.where(
        is_partial[:, :, None],
        jnp.broadcast_to(mask_q_seq, (batch_size, num_blocks, num_blocks)),
        n_full3,
    )
    bwd_full_cnt = jnp.where(t < n_full, n_full + cand_blocks, 0)
    bwd_full_idx = jnp.broadcast_to(full_q_seq, (batch_size, num_blocks, num_blocks))
    bwd_diag_cnt = fwd_diag_cnt
    bwd_diag_idx = fwd_diag_idx

    valid_block_upper = jnp.where(is_hist, jnp.clip(valid_len - block * t, 0, block), 0)
    valid_block_lower = jnp.where(is_hist, block, 0)

    def _bcast(arr):
        arr = arr.astype(jnp.int32)
        return jnp.broadcast_to(arr[:, None], (batch_size, num_q_heads) + arr.shape[1:])

    fwd_arrays = (
        fwd_mask_cnt,
        fwd_mask_idx,
        fwd_full_cnt,
        fwd_full_idx,
        fwd_diag_cnt,
        fwd_diag_idx,
    )
    bwd_arrays = (
        bwd_mask_cnt,
        bwd_mask_idx,
        bwd_full_cnt,
        bwd_full_idx,
        bwd_diag_cnt,
        bwd_diag_idx,
    )
    fwd_bs = tuple(_bcast(a) for a in fwd_arrays)
    bwd_bs = tuple(_bcast(a) for a in bwd_arrays)
    return fwd_bs, bwd_bs, _bcast(valid_block_upper), _bcast(valid_block_lower)


def fwd_launcher(arch, head_dim, qhead_per_kvhead, use_pack_gqa, score_mod):
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute

    from xrex.cutedsl.ranker_fa4.block_sparsity import BlockSparseTensors

    m_block = 128
    n_block = 128
    if arch == 80:
        from xrex.cutedsl.ranker_fa4.flash_fwd import FlashAttentionForwardSm80

        fa_fwd = FlashAttentionForwardSm80(
            cutlass.BFloat16,
            head_dim,
            head_dim,
            qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            pack_gqa=False,
            tile_m=64,
            tile_n=n_block,
            num_stages=1,
            num_threads=128,
            Q_in_regs=False,
            q_subtile_factor=m_block // 64,
        )
    elif arch == 90:
        from xrex.cutedsl.ranker_fa4.flash_fwd_sm90 import FlashAttentionForwardSm90

        fa_fwd = FlashAttentionForwardSm90(
            cutlass.BFloat16,
            head_dim,
            head_dim,
            qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            pack_gqa=use_pack_gqa,
            tile_m=m_block,
            tile_n=n_block,
            num_stages=2,
            num_threads=384,
            Q_in_regs=False,
            intra_wg_overlap=True,
            mma_pv_is_rs=True,
            mask_mod=None,
            score_mod=score_mod,
        )
    else:
        from xrex.cutedsl.ranker_fa4.flash_fwd_sm100 import FlashAttentionForwardSm100

        fa_fwd = FlashAttentionForwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            pack_gqa=use_pack_gqa,
            is_persistent=True,
            mask_mod=None,
            score_mod=score_mod,
            q_stage=1,
        )

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
        mValidUpper: cute.Tensor,
        mValidLower: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        *softmax_stats,
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
            valid_block_upper=mValidUpper,
            valid_block_lower=mValidLower,
        )
        if cutlass.const_expr(len(softmax_stats) == 2):
            fa_fwd(
                mQ,
                mK,
                mV,
                mO,
                mLSE,
                softmax_scale,
                blocksparse_tensors=bs,
                mSoftmaxStats=softmax_stats,
                stream=stream,
            )
        else:
            fa_fwd(
                mQ,
                mK,
                mV,
                mO,
                mLSE,
                softmax_scale,
                blocksparse_tensors=bs,
                stream=stream,
            )

    return launch_fwd


def ranker_attention_fa4(
    q,
    k,
    v,
    sm_scale,
    block_sparse_layout,
    valid_block_upper=None,
    valid_block_lower=None,
    cap=-1.0,
    cap_method="soft_sign",
):
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.jax import cutlass_call

    from xrex.cutedsl.ranker_attention_varlen_fa4 import zeros_after
    from xrex.cutedsl.ranker_fa4.block_sparsity import BlockSparseTensors
    from xrex.cutedsl.ranker_fa4.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
    from xrex.cutedsl.ranker_fa4.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
    from xrex.cutedsl.ranker_fa4.utils import create_cap_scoremods

    arch = cutedsl_arch()
    centered_stats = arch in (90, 100)
    score_mod, score_mod_bwd = create_cap_scoremods(cap, cap_method)
    cap_key = (float(cap), cap_method) if score_mod is not None else None
    if score_mod is not None and arch == 80:
        raise NotImplementedError(
            "cutedsl ranker attention: attn_logit_cap > 0 needs score_mod, which the SM80 "
            "kernels reject together with block sparsity; use qk_norm (cap <= 0) on A100"
        )
    batch_size, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    qhead_per_kvhead = num_q_heads // num_kv_heads
    m_block = 128
    n_block = 128
    hdr = ((head_dim + 31) // 32) * 32
    sr_q = ((seq_len + m_block - 1) // m_block) * m_block
    sr_k = ((seq_len + n_block - 1) // n_block) * n_block
    dKV_postprocess = qhead_per_kvhead > 1
    if arch == 80:
        assert seq_len % m_block == 0, "SM80 cutedsl ranker attention requires seq_len % 128 == 0"
        bwd_tile_m = 64
    elif arch == 90:
        bwd_tile_m = 64
    else:
        bwd_tile_m = m_block

    fwd_bs, bwd_bs = block_sparse_layout
    if valid_block_upper is None or valid_block_lower is None:
        if valid_block_upper is not None or valid_block_lower is not None:
            raise ValueError("valid_block_upper and valid_block_lower must be provided together")
        valid_block_upper = jnp.zeros(fwd_bs[2].shape, dtype=jnp.int32)
        valid_block_lower = jnp.zeros(fwd_bs[2].shape, dtype=jnp.int32)
    valid_block_upper = jnp.broadcast_to(valid_block_upper, fwd_bs[2].shape)
    valid_block_lower = jnp.broadcast_to(valid_block_lower, fwd_bs[2].shape)
    bs_num_blocks = int(fwd_bs[3].shape[-2])
    bs_max_hist_blocks = int(fwd_bs[3].shape[-1])
    if bs_num_blocks != (seq_len + m_block - 1) // m_block:
        raise ValueError(
            f"block-sparse arrays cover {bs_num_blocks} m-tiles but the kernel "
            f"iterates {(seq_len + m_block - 1) // m_block} (seq_len={seq_len})"
        )
    use_pack_gqa = qhead_per_kvhead > 1 and (m_block % qhead_per_kvhead == 0)

    cache_key = (
        arch,
        head_dim,
        num_q_heads,
        num_kv_heads,
        batch_size,
        seq_len,
        bs_num_blocks,
        bs_max_hist_blocks,
        int(fwd_bs[1].shape[-1]),
        int(bwd_bs[1].shape[-1]),
        use_pack_gqa,
        cap_key,
    )

    if cache_key not in _FA4_KERNEL_CACHE:
        launch_fwd = fwd_launcher(arch, head_dim, qhead_per_kvhead, use_pack_gqa, score_mod)

        q_shape = (batch_size, seq_len, num_q_heads, head_dim)
        k_shape = (batch_size, seq_len, num_kv_heads, head_dim)
        lse_shape = (batch_size, num_q_heads, seq_len)

        fwd_call = cutlass_call(
            launch_fwd,
            output_shape_dtype=[
                jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
                jax.ShapeDtypeStruct(lse_shape, jnp.float32),
            ]
            + ([jax.ShapeDtypeStruct(lse_shape, jnp.float32)] * 2 if centered_stats else []),
            use_static_tensors=False,
            softmax_scale=cutlass.Float32(sm_scale),
        )

        dq_accum_shape = (batch_size, num_q_heads, sr_q * hdr)
        rows_shape = (batch_size, num_q_heads, sr_q)
        fa_pre = FlashAttentionBackwardPreprocess(
            cutlass.BFloat16, head_dim, head_dim, tile_m=m_block
        )

        @cute.jit
        def launch_pre(
            stream: cuda_driver.CUstream,
            mO: cute.Tensor,
            mdO: cute.Tensor,
            mLSE: cute.Tensor,
            mPdPsum: cute.Tensor,
            mLSElog2: cute.Tensor,
            mdQaccum: cute.Tensor,
        ):
            fa_pre(mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, None, None, None, stream)

        pre_call = cutlass_call(
            launch_pre,
            output_shape_dtype=[
                jax.ShapeDtypeStruct(rows_shape, jnp.float32),
                jax.ShapeDtypeStruct(rows_shape, jnp.float32),
                jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32),
            ],
            use_static_tensors=False,
        )

        if arch == 80:
            from xrex.cutedsl.ranker_fa4.flash_bwd import FlashAttentionBackwardSm80

            bwd_atom_layout_dkv = 2
            fa_bwd = FlashAttentionBackwardSm80(
                cutlass.BFloat16,
                head_dim,
                head_dim,
                qhead_per_kvhead,
                m_block_size=bwd_tile_m,
                n_block_size=n_block,
                num_stages_Q=2,
                num_stages_dO=2,
                num_threads=256,
                pack_gqa=False,
                is_causal=False,
                SdP_swapAB=False,
                dKV_swapAB=False,
                dQ_swapAB=False,
                AtomLayoutMSdP=2,
                AtomLayoutNdKV=bwd_atom_layout_dkv,
                AtomLayoutMdQ=2,
                V_in_regs=False,
                q_subtile_factor=m_block // bwd_tile_m,
            )
            post_threads = 256
            post_dq_atom_layout = 2
        elif arch == 90:
            from xrex.cutedsl.ranker_fa4.flash_bwd_sm90 import FlashAttentionBackwardSm90

            bwd_atom_layout_dkv = 2
            fa_bwd = FlashAttentionBackwardSm90(
                cutlass.BFloat16,
                head_dim,
                head_dim,
                qhead_per_kvhead,
                False,
                is_local=False,
                deterministic=False,
                tile_m=bwd_tile_m,
                tile_n=n_block,
                Q_stage=2,
                dO_stage=2,
                PdS_stage=2,
                SdP_swapAB=True,
                dKV_swapAB=False,
                dQ_swapAB=False,
                AtomLayoutMSdP=1,
                AtomLayoutNdKV=bwd_atom_layout_dkv,
                AtomLayoutMdQ=1,
                num_threads=384,
                mask_mod=None,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                subtile_factor=m_block // bwd_tile_m,
            )
            post_threads = 256
            post_dq_atom_layout = 1
        else:
            from xrex.cutedsl.ranker_fa4.flash_bwd_sm100 import FlashAttentionBackwardSm100

            bwd_atom_layout_dkv = 1
            fa_bwd = FlashAttentionBackwardSm100(
                head_dim=head_dim,
                head_dim_v=head_dim,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=False,
                is_local=False,
                mask_mod=None,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
            post_threads = 128
            post_dq_atom_layout = 1
        dk_accum_shape = (batch_size, num_kv_heads, sr_k * hdr)
        dkv_out = jax.ShapeDtypeStruct(
            dk_accum_shape if dKV_postprocess else k_shape,
            jnp.float32 if dKV_postprocess else jnp.bfloat16,
        )

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
            mValidUpper: cute.Tensor,
            mValidLower: cute.Tensor,
            *rest,
            softmax_scale: cutlass.Float32,
        ):
            mdQa, mdKa, mdVa = rest[-3], rest[-2], rest[-1]
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
                valid_block_upper=mValidUpper,
                valid_block_lower=mValidLower,
            )
            if cutlass.const_expr(centered_stats):
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
                    blocksparse_tensors=bs,
                    mSoftmaxStats=(rest[0], rest[1]),
                    stream=stream,
                )
            else:
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
                    blocksparse_tensors=bs,
                    stream=stream,
                )

        bwd_call = cutlass_call(
            launch_bwd,
            output_shape_dtype=[
                jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32),
                dkv_out,
                dkv_out,
            ],
            input_output_aliases=(
                {16: 0, 17: 1, 18: 2} if centered_stats else {14: 0, 15: 1, 16: 2}
            ),
            use_static_tensors=False,
            softmax_scale=cutlass.Float32(sm_scale),
        )

        fa_post_dq = FlashAttentionBackwardPostprocess(
            cutlass.BFloat16,
            head_dim,
            arch,
            tile_m=bwd_tile_m,
            num_threads=post_threads,
            AtomLayoutMdQ=post_dq_atom_layout,
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
                cutlass.BFloat16,
                head_dim,
                arch,
                tile_m=n_block,
                num_threads=post_threads,
                AtomLayoutMdQ=bwd_atom_layout_dkv,
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
                cutlass.BFloat16,
                head_dim,
                arch,
                tile_m=n_block,
                num_threads=post_threads,
                AtomLayoutMdQ=bwd_atom_layout_dkv,
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
            pre_call=pre_call,
            bwd_call=bwd_call,
            post_dq_call=post_dq_call,
            post_dk_call=post_dk_call,
            post_dv_call=post_dv_call,
            dKV_postprocess=dKV_postprocess,
            dkv_out=dkv_out,
        )

    c = _FA4_KERNEL_CACHE[cache_key]

    @jax.custom_vjp
    def _attention(q, k, v, *bs_args):
        fbs = bs_args[:6]
        valid_bounds = bs_args[12:]
        out, *_stats = c["fwd_call"](q, k, v, *fbs, *valid_bounds)
        return out

    def _attention_fwd(q, k, v, *bs_args):
        fbs = bs_args[:6]
        valid_bounds = bs_args[12:]
        out, lse, *softmax_stats = c["fwd_call"](q, k, v, *fbs, *valid_bounds)
        out = checkpoint_name(out, "cutedsl_attn_outputs")
        lse = checkpoint_name(lse, "cutedsl_attn_outputs")
        softmax_stats = tuple(checkpoint_name(t, "cutedsl_attn_outputs") for t in softmax_stats)
        return out, (q, k, v, out, lse, softmax_stats, bs_args)

    def _attention_bwd(res, g):
        q, k, v, out, lse, softmax_stats, bs_args = res
        bbs = bs_args[6:12]
        valid_bounds = bs_args[12:]

        _, lse_log2, dq_accum_init = c["pre_call"](out, g.astype(out.dtype), lse)
        dpsum = attention_delta(out, g, sr_q)
        softmax_stats = pad_softmax_stats(softmax_stats, sr_q)

        dk_init, dv_init = zeros_after(g, c["dkv_out"], c["dkv_out"])
        dq_accum, dk, dv = c["bwd_call"](
            q,
            k,
            v,
            g,
            lse_log2,
            dpsum,
            *bbs,
            *valid_bounds,
            *softmax_stats,
            dq_accum_init,
            dk_init,
            dv_init,
        )
        (dq,) = c["post_dq_call"](dq_accum)
        if c["dKV_postprocess"]:
            (dk,) = c["post_dk_call"](dk)
            (dv,) = c["post_dv_call"](dv)

        return (dq, dk, dv) + (None,) * len(bs_args)

    _attention.defvjp(_attention_fwd, _attention_bwd)
    return _attention(
        q,
        k,
        v,
        *fwd_bs,
        *bwd_bs,
        valid_block_upper,
        valid_block_lower,
    )
