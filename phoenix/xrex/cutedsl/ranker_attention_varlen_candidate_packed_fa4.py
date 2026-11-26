# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from xrex.cutedsl.ranker_attention_candidate_only_fa4 import candidate_attention_fwd_bwd


def ranker_attention_varlen_candidate_packed_fa4(
    q,
    k,
    v,
    sm_scale,
    block_sparse_layout,
    valid_block_upper,
    valid_block_lower,
    candidate_cu_seqlens,
    candidate_key_starts,
    candidate_key_counts,
    history_region_len,
):
    from xrex.cutedsl.ranker_attention_varlen_fa4 import (
        varlen_backward,
        varlen_forward,
        varlen_kernels,
    )

    assert q.shape[0] == 1, "one packed sequence per device"
    region_start = int(history_region_len)
    kernels, bs_args = varlen_kernels(
        q, k, v, sm_scale, block_sparse_layout, valid_block_upper, valid_block_lower
    )
    cand_forward, cand_backward = candidate_attention_fwd_bwd(
        q[0, region_start:].shape, k[0].shape, candidate_cu_seqlens.shape[0] - 1, sm_scale
    )

    def candidate_inputs(q, k, v, cand_cu, key_starts, key_counts):
        keys, values = k[0], v[0]
        return (
            q[0, region_start:],
            keys,
            values,
            keys[region_start:],
            values[region_start:],
            cand_cu,
            key_starts,
            key_counts,
        )

    def forward(q, k, v, cand_cu, key_starts, key_counts, bs_args):
        out, lse, softmax_stats = varlen_forward(kernels, q, k, v, bs_args)
        cand_o, cand_stats = cand_forward(
            *candidate_inputs(q, k, v, cand_cu, key_starts, key_counts)
        )
        return out.at[0, region_start:].set(cand_o), lse, softmax_stats, cand_stats

    @jax.custom_vjp
    def attention(q, k, v, cand_cu, key_starts, key_counts, *bs_args):
        return forward(q, k, v, cand_cu, key_starts, key_counts, bs_args)[0]

    def attention_fwd(q, k, v, cand_cu, key_starts, key_counts, *bs_args):
        out, lse, softmax_stats, cand_stats = forward(
            q, k, v, cand_cu, key_starts, key_counts, bs_args
        )
        out_bits = jax.lax.bitcast_convert_type(jax.lax.optimization_barrier(out), jnp.uint16)
        out_bits = checkpoint_name(out_bits, "cutedsl_attn_outputs")
        out = jax.lax.bitcast_convert_type(out_bits, out.dtype)
        lse = checkpoint_name(lse, "cutedsl_attn_outputs")
        softmax_stats = tuple(checkpoint_name(t, "cutedsl_attn_outputs") for t in softmax_stats)
        cand_stats = checkpoint_name(cand_stats, "cutedsl_attn_outputs")
        return out, (
            q,
            k,
            v,
            cand_cu,
            key_starts,
            key_counts,
            out,
            lse,
            softmax_stats,
            cand_stats,
            bs_args,
        )

    def attention_bwd(res, g):
        q, k, v, cand_cu, key_starts, key_counts, out, lse, softmax_stats, cand_stats, bs_args = res
        dq, dk, dv = varlen_backward(kernels, q, k, v, out, lse, softmax_stats, bs_args, g)
        dq_cand, dk_keys, dv_keys, dk_self, dv_self = cand_backward(
            *candidate_inputs(q, k, v, cand_cu, key_starts, key_counts),
            out[0, region_start:],
            cand_stats,
            g[0, region_start:],
        )
        dq = dq.at[0, region_start:].set(dq_cand)
        dk = (dk.at[0].add(dk_keys)).at[0, region_start:].add(dk_self)
        dv = (dv.at[0].add(dv_keys)).at[0, region_start:].add(dv_self)
        return (dq, dk, dv, None, None, None) + (None,) * len(bs_args)

    attention.defvjp(attention_fwd, attention_bwd)
    return attention(
        q, k, v, candidate_cu_seqlens, candidate_key_starts, candidate_key_counts, *bs_args
    )
