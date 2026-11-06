# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

_CANDIDATE_ATTENTION_CACHE: dict = {}


def candidate_attention_fwd_bwd(q_shape, k_shape, num_seqs, sm_scale):
    from xrex.cutedsl.ranker_attention_fa4 import attention_delta, cutedsl_arch
    from xrex.cutedsl.ranker_attention_varlen_fa4 import zeros_after

    total_q, num_q_heads, head_dim = q_shape
    total_k, num_kv_heads, _ = k_shape
    arch = cutedsl_arch()
    sm_scale = float(sm_scale)
    scale_log2 = sm_scale * math.log2(math.e)
    cache_key = (arch, total_q, total_k, num_seqs, num_q_heads, num_kv_heads, head_dim, sm_scale)
    if cache_key in _CANDIDATE_ATTENTION_CACHE:
        return _CANDIDATE_ATTENTION_CACHE[cache_key]

    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.jax import cutlass_call

    from xrex.cutedsl.ranker_fa4.flash_bwd_postprocess import FlashAttentionBackwardPostprocess

    if arch not in (90, 100):
        raise NotImplementedError(
            "Ranker candidate-only attention supports only SM90 (H100) and SM100 (GB200/GB300). "
            "On this GPU, set seqpack_candidate_packing=False and candidate_only_attn_impl=None "
            "to compute every token and keep only the candidates' outputs."
        )

    qpk = num_q_heads // num_kv_heads
    q_heads_per_bwd_head = qpk if arch == 90 else 1
    bwd_qpk = qpk // q_heads_per_bwd_head
    bwd_rows, bwd_num_heads = total_q * q_heads_per_bwd_head, num_q_heads // q_heads_per_bwd_head
    tile_n = 128
    bwd_tile_m = 64 if arch == 90 else 128
    hdr = ((head_dim + 31) // 32) * 32
    bwd_rows_padded = (bwd_rows + num_seqs * bwd_tile_m + bwd_tile_m - 1) // bwd_tile_m * bwd_tile_m
    total_k_padded = (total_k + num_seqs * tile_n + tile_n - 1) // tile_n * tile_n

    if arch == 90:
        from xrex.cutedsl.ranker_fa4.flash_fwd_sm90 import FlashAttentionForwardSm90

        fa_fwd = FlashAttentionForwardSm90(
            cutlass.BFloat16,
            head_dim,
            head_dim,
            qpk,
            is_causal=False,
            is_local=False,
            pack_gqa=qpk > 1,
            tile_m=128,
            tile_n=tile_n,
            num_stages=2,
            num_threads=384,
            Q_in_regs=False,
            intra_wg_overlap=True,
            mma_pv_is_rs=True,
        )
    else:
        from xrex.cutedsl.ranker_fa4.flash_fwd_sm100 import FlashAttentionForwardSm100

        fa_fwd = FlashAttentionForwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=qpk,
            is_causal=False,
            is_local=False,
            pack_gqa=qpk > 1 and 128 % qpk == 0,
            q_stage=1,
            is_persistent=False,
            is_varlen_q=True,
        )

    @cute.jit
    def launch_fwd(
        stream: cuda_driver.CUstream,
        mQ,
        mK,
        mV,
        mCuQ,
        mCuK,
        mSeqUsedK,
        mO,
        mLSE,
        mRowMax,
        mInvSum,
        softmax_scale: cutlass.Float32,
    ):
        fa_fwd(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale,
            mCuQ,
            mCuK,
            None,
            mSeqUsedK,
            mSoftmaxStats=(mRowMax, mInvSum),
            stream=stream,
        )

    fwd_call = cutlass_call(
        launch_fwd,
        output_shape_dtype=[
            jax.ShapeDtypeStruct((total_q, num_q_heads, head_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((num_q_heads, total_q), jnp.float32),
            jax.ShapeDtypeStruct((num_q_heads, total_q), jnp.float32),
            jax.ShapeDtypeStruct((num_q_heads, total_q), jnp.float32),
        ],
        use_static_tensors=False,
        softmax_scale=cutlass.Float32(sm_scale),
    )

    if arch == 90:
        from xrex.cutedsl.ranker_fa4.flash_bwd_sm90 import FlashAttentionBackwardSm90

        bwd_atom_layout_dkv = 2
        fa_bwd = FlashAttentionBackwardSm90(
            cutlass.BFloat16,
            head_dim,
            head_dim,
            bwd_qpk,
            False,
            is_local=False,
            deterministic=False,
            tile_m=bwd_tile_m,
            tile_n=tile_n,
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
        )
        post_threads = 256
    else:
        from xrex.cutedsl.ranker_fa4.flash_bwd_sm100 import FlashAttentionBackwardSm100

        bwd_atom_layout_dkv = 1
        fa_bwd = FlashAttentionBackwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=qpk,
            is_causal=False,
            is_local=False,
        )
        post_threads = 128
    dkv_postprocess = bwd_qpk > 1
    dkv_out = jax.ShapeDtypeStruct(
        (num_kv_heads, total_k_padded * hdr)
        if dkv_postprocess
        else (total_k, num_kv_heads, head_dim),
        jnp.float32 if dkv_postprocess else jnp.bfloat16,
    )
    dq_accum_shape = (bwd_num_heads, bwd_rows_padded * hdr)
    bwd_accumulators = [jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32), dkv_out, dkv_out]

    @cute.jit
    def launch_bwd(
        stream: cuda_driver.CUstream,
        mQ,
        mK,
        mV,
        mdO,
        mRowMax,
        mdPsum,
        mCuQ,
        mCuK,
        mSeqUsedK,
        mInvSum,
        mdQaccum,
        mdK,
        mdV,
        softmax_scale: cutlass.Float32,
    ):
        fa_bwd(
            mQ,
            mK,
            mV,
            mdO,
            mRowMax,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            softmax_scale,
            mCuQ,
            mCuK,
            None,
            mSeqUsedK,
            mSoftmaxStats=(mRowMax, mInvSum),
            stream=stream,
        )

    bwd_call = cutlass_call(
        launch_bwd,
        output_shape_dtype=bwd_accumulators,
        input_output_aliases={10: 0, 11: 1, 12: 2},
        use_static_tensors=False,
        softmax_scale=cutlass.Float32(sm_scale),
    )

    def postprocess(tile_m, atom_layout, out_shape, scale, with_seqused):
        fa_post = FlashAttentionBackwardPostprocess(
            cutlass.BFloat16,
            head_dim,
            arch,
            tile_m=tile_m,
            num_threads=post_threads,
            AtomLayoutMdQ=atom_layout,
        )

        if with_seqused:

            @cute.jit
            def launch(
                stream: cuda_driver.CUstream, mAccum, mCu, mSeqUsed, mOut, scale: cutlass.Float32
            ):
                fa_post(mAccum, mOut, scale, mCu, mSeqUsed, stream)

        else:

            @cute.jit
            def launch(stream: cuda_driver.CUstream, mAccum, mCu, mOut, scale: cutlass.Float32):
                fa_post(mAccum, mOut, scale, mCu, None, stream)

        return cutlass_call(
            launch,
            output_shape_dtype=[jax.ShapeDtypeStruct(out_shape, jnp.bfloat16)],
            use_static_tensors=False,
            scale=cutlass.Float32(scale),
        )

    kv_shape = (total_k, num_kv_heads, head_dim)
    post_dq = postprocess(bwd_tile_m, 1, (bwd_rows, bwd_num_heads, head_dim), sm_scale, False)
    post_dk = post_dv = None
    if dkv_postprocess:
        post_dk = postprocess(tile_n, bwd_atom_layout_dkv, kv_shape, sm_scale, True)
        post_dv = postprocess(tile_n, bwd_atom_layout_dkv, kv_shape, 1.0, True)

    def fold_q_heads(x):
        if q_heads_per_bwd_head == 1:
            return x
        x = x.reshape(total_q, bwd_num_heads, q_heads_per_bwd_head, *x.shape[2:]).swapaxes(1, 2)
        return x.reshape(bwd_rows, bwd_num_heads, *x.shape[3:])

    def unfold_q_heads(x):
        if q_heads_per_bwd_head == 1:
            return x
        x = x.reshape(total_q, q_heads_per_bwd_head, bwd_num_heads, *x.shape[2:]).swapaxes(1, 2)
        return x.reshape(total_q, num_q_heads, *x.shape[3:])

    def self_scores(q, k_self, query_valid):
        k_rep = jnp.repeat(k_self, qpk, axis=1)
        s = jnp.einsum("thd,thd->th", q.astype(jnp.float32), k_rep.astype(jnp.float32))
        return jnp.where(query_valid[:, None], s, -jnp.inf)

    def forward(q, k_keys, v_keys, k_self, v_self, cu_q, cu_k, used_k):
        query_valid = jnp.arange(total_q) < cu_q[-1]
        o_hist, _, hist_max, hist_inv_sum = fwd_call(q, k_keys, v_keys, cu_q, cu_k, used_k)
        has_history = query_valid[:, None] & (hist_inv_sum.T > 0)
        hist_max = jnp.where(has_history, hist_max.T, -jnp.inf)
        s = self_scores(q, k_self, query_valid)
        row_max = jnp.where(query_valid[:, None], jnp.maximum(hist_max, s), 0.0)
        hist_sum = jnp.exp2((hist_max - row_max) * scale_log2) / jnp.where(
            has_history, hist_inv_sum.T, 1.0
        )
        self_sum = jnp.exp2((s - row_max) * scale_log2)
        inv_sum = jnp.where(
            query_valid[:, None],
            1.0 / jnp.where(query_valid[:, None], hist_sum + self_sum, 1.0),
            0.0,
        )
        w_hist, w_self = hist_sum * inv_sum, self_sum * inv_sum
        v_rep = jnp.repeat(v_self, qpk, axis=1).astype(jnp.float32)
        o_hist = jnp.where(query_valid[:, None, None], o_hist.astype(jnp.float32), 0.0)
        o = (w_hist[..., None] * o_hist + w_self[..., None] * v_rep).astype(jnp.bfloat16)
        return o, jnp.stack((jnp.where(query_valid[:, None], row_max, jnp.inf), inv_sum))

    def backward(q, k_keys, v_keys, k_self, v_self, cu_q, cu_k, used_k, o, stats, do):
        query_valid = jnp.arange(total_q) < cu_q[-1]
        do = jnp.where(query_valid[:, None, None], do, 0).astype(jnp.bfloat16)
        bwd_cu_rows = cu_q if q_heads_per_bwd_head == 1 else cu_q * q_heads_per_bwd_head
        bwd_do = fold_q_heads(do)
        padded_starts = (
            (bwd_cu_rows[:-1] + jnp.arange(num_seqs) * bwd_tile_m) // bwd_tile_m * bwd_tile_m
        )
        rows = jnp.arange(bwd_rows_padded)
        seq = jnp.searchsorted(padded_starts, rows, side="right") - 1
        src = rows - padded_starts[seq] + bwd_cu_rows[seq]
        src = jnp.where(src < bwd_cu_rows[seq + 1], src, bwd_rows)

        def pad_stat(x, fill):
            return jnp.take(fold_q_heads(x), src, axis=0, mode="fill", fill_value=fill).T

        row_max, inv_sum = stats
        padded_max, padded_inv_sum = pad_stat(row_max, jnp.inf), pad_stat(inv_sum, 0.0)
        delta = attention_delta(o[None], do[None], (total_q + 31) // 32 * 32)[0, :, :total_q].T
        dpsum = pad_stat(delta, 0.0)
        dq_accum_init, dk_init, dv_init = zeros_after(do, *bwd_accumulators)
        dq_accum, dk_keys, dv_keys = bwd_call(
            fold_q_heads(q),
            k_keys,
            v_keys,
            bwd_do,
            padded_max,
            dpsum,
            bwd_cu_rows,
            cu_k,
            used_k,
            padded_inv_sum,
            dq_accum_init,
            dk_init,
            dv_init,
        )
        (dq,) = post_dq(dq_accum, bwd_cu_rows)
        if post_dk is not None and post_dv is not None:
            (dk_keys,) = post_dk(dk_keys, cu_k, used_k)
            (dv_keys,) = post_dv(dv_keys, cu_k, used_k)
        marks = jnp.zeros(total_k + 1, jnp.int32).at[cu_k[:-1]].add(1)
        marks = marks.at[cu_k[:-1] + used_k].add(-1)
        key_valid = jnp.cumsum(marks[:-1]) > 0
        dq = jnp.where(query_valid[:, None, None], unfold_q_heads(dq), 0)
        dk_keys = jnp.where(key_valid[:, None, None], dk_keys, 0)
        dv_keys = jnp.where(key_valid[:, None, None], dv_keys, 0)

        s = self_scores(q, k_self, query_valid)
        p_self = jnp.where(
            query_valid[:, None], jnp.exp2((s - row_max) * scale_log2) * inv_sum, 0.0
        )
        do32 = do.astype(jnp.float32)
        delta = jnp.sum(do32 * o.astype(jnp.float32), axis=-1)
        v_rep = jnp.repeat(v_self, qpk, axis=1).astype(jnp.float32)
        ds = p_self * (jnp.sum(do32 * v_rep, axis=-1) - delta)
        ds = jnp.where(query_valid[:, None], ds, 0.0)
        k_rep = jnp.repeat(k_self, qpk, axis=1).astype(jnp.float32)
        dq = dq.astype(jnp.float32) + ds[..., None] * k_rep * sm_scale

        def group_sum(x):
            return x.reshape(total_q, num_kv_heads, qpk, head_dim).sum(axis=2)

        dk_self = group_sum(ds[..., None] * q.astype(jnp.float32) * sm_scale)
        dv_self = group_sum(p_self[..., None] * do32)
        return (
            dq.astype(q.dtype),
            dk_keys.astype(k_keys.dtype),
            dv_keys.astype(v_keys.dtype),
            dk_self.astype(k_self.dtype),
            dv_self.astype(v_self.dtype),
        )

    _CANDIDATE_ATTENTION_CACHE[cache_key] = (forward, backward)
    return forward, backward


def ranker_attention_varlen_candidate_only_fa4(
    candidate_q,
    history_k,
    history_v,
    own_k,
    own_v,
    candidate_cu_seqlens,
    history_key_starts,
    history_key_counts,
    sm_scale,
):
    forward, backward = candidate_attention_fwd_bwd(
        candidate_q.shape, history_k.shape, candidate_cu_seqlens.shape[0] - 1, sm_scale
    )

    @jax.custom_vjp
    def attention(*args):
        return forward(*args)[0]

    def attention_fwd(*args):
        o, stats = forward(*args)
        o = checkpoint_name(o, "cutedsl_attn_outputs")
        stats = checkpoint_name(stats, "cutedsl_attn_outputs")
        return o, (*args, o, stats)

    def attention_bwd(res, do):
        return backward(*res, do) + (None, None, None)

    attention.defvjp(attention_fwd, attention_bwd)
    return attention(
        candidate_q,
        history_k,
        history_v,
        own_k,
        own_v,
        candidate_cu_seqlens,
        history_key_starts,
        history_key_counts,
    )


def ranker_attention_candidate_only_fa4(
    candidate_q, history_k, history_v, own_k, own_v, hist_valid_len, sm_scale
):
    batch_size, num_candidates = candidate_q.shape[:2]
    row_starts = jnp.arange(batch_size + 1, dtype=jnp.int32)
    output = ranker_attention_varlen_candidate_only_fa4(
        *(jax.lax.collapse(x, 0, 2) for x in (candidate_q, history_k, history_v, own_k, own_v)),
        row_starts * num_candidates,
        row_starts * history_k.shape[1],
        hist_valid_len,
        sm_scale,
    )
    return output.reshape(candidate_q.shape)
