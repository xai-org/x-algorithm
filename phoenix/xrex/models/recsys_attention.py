# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

import jax.numpy as jnp
from xai_configlib import configclass

from xrex.models.attention import AttentionConfig, CustomAttention
from xrex.models.sharding_context import NamedShape


@configclass
class RecsysAttentionConfig(AttentionConfig):
    history_seq_len: int = 1024
    sequence_len: int = 1152
    num_user_prefix_tokens: int = 1

    attn_grad_clip: float = 0.0
    q_seg_ids_fn: str | None = None
    k_seg_ids_fn: str | None = None


class PallasRankerAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self):
        def sharded_mha(q, k, v, segment_ids, segment_ids_k, temp):
            assert isinstance(self.config, RecsysAttentionConfig), (
                "PallasRankerAttention only supports RecsysAttentionConfig"
            )
            assert not self.config.causal, "Causal attention is not supported"
            assert self.config.key_size % 64 == 0, "key_size must be a multiple of 64"

            from xrex.pallas import ranker_attention_fa3
            from xrex.utils.gpu import GpuArch, gpu_arch

            use_fa3 = gpu_arch() == GpuArch.H100 and str(self.config.fa_version) != "2"

            if use_fa3:
                block_q = 64
                block_kv = 64
                assert q.shape[1] % (block_q * 2) == 0 and k.shape[1] % block_kv == 0, (
                    "q.shape[1] and k.shape[1] must be a multiple of block_q * 2 and block_kv"
                )
                history_upper = self.config.num_user_prefix_tokens + self.config.history_seq_len
                candidate_lower = history_upper
                candidate_upper = q.shape[1]
                bound = (0, history_upper, candidate_lower, candidate_upper)
                compute_wgs_bwd = 2 if self.config.key_size <= 128 else 1
                tuning_config = ranker_attention_fa3.TuningConfig(
                    block_q=block_q,
                    block_kv=block_kv,
                    max_concurrent_steps=2,
                    compute_wgs_bwd=compute_wgs_bwd,
                    block_q_dkv=block_q,
                    block_kv_dkv=block_kv,
                    block_q_dq=block_q,
                    block_kv_dq=block_kv,
                )
                attn = ranker_attention_fa3.attention(
                    q,
                    k,
                    v,
                    tuning_config,
                    save_residuals=False,
                    bound=bound,
                    sm_scale=self.scale_config.attn_output_scale(self.config.key_size),
                    cap=self.config.attn_logit_cap,
                    cap_method=self.config.attn_logit_cap_method,
                )
                return attn, None
            else:
                from xrex.pallas import ranker_attention_v2

                num_repeat = self.config.num_q_heads // self.config.num_kv_heads
                k = jnp.repeat(k, num_repeat, axis=2)
                v = jnp.repeat(v, num_repeat, axis=2)
                return ranker_attention_v2.mha(
                    q,
                    k,
                    v,
                    temp,
                    segment_ids,
                    causal=self.config.causal,
                    sm_scale=self.scale_config.attn_output_scale(self.config.key_size),
                    cap=self.config.attn_logit_cap,
                    cap_method=self.config.attn_logit_cap_method,
                    backward_pass_impl="triton_split",
                    interpret=False,
                ), None

        return sharded_mha, ()


class PallasRankerAttentionInference(CustomAttention):
    def sharded_custom_op_with_extra_args(self):
        def sharded_mha(q, k, v, segment_ids, segment_ids_k, temp):
            assert isinstance(self.config, RecsysAttentionConfig), (
                "PallasRankerAttentionInference only supports RecsysAttentionConfig"
            )
            assert not self.config.causal, "Causal attention is not supported"
            assert self.config.key_size % 64 == 0, "key_size must be a multiple of 64"

            from xrex.pallas import ranker_attention_fa3
            from xrex.utils.gpu import GpuArch, gpu_arch

            use_h100_kernel = gpu_arch() == GpuArch.H100

            if use_h100_kernel:
                block_q = 64
                block_kv = 128
                assert q.shape[1] % (block_q * 2) == 0 and k.shape[1] % block_kv == 0, (
                    "q.shape[1] and k.shape[1] must be a multiple of block_q * 2 and block_kv"
                )
                history_upper = self.config.num_user_prefix_tokens + self.config.history_seq_len
                candidate_lower = history_upper
                candidate_upper = q.shape[1]
                bound = (0, history_upper, candidate_lower, candidate_upper)
                tuning_config = ranker_attention_fa3.TuningConfig(
                    block_q=block_q,
                    block_kv=block_kv,
                    max_concurrent_steps=2,
                    block_q_dkv=block_q,
                    block_kv_dkv=block_kv,
                    block_q_dq=block_q,
                    block_kv_dq=block_kv,
                )
                batch_size = q.shape[0]
                bound_arr = ranker_attention_fa3._normalize_bound(bound, batch_size, q.shape[1])
                attn = ranker_attention_fa3._attention_forward_inference(
                    q,
                    k,
                    v,
                    config=tuning_config,
                    bound=bound_arr,
                    sm_scale=self.scale_config.attn_output_scale(self.config.key_size),
                    cap=self.config.attn_logit_cap,
                    cap_method=self.config.attn_logit_cap_method,
                )
                return attn, None
            else:
                from xrex.pallas import ranker_attention_v2

                return ranker_attention_v2.mha_inference_gqa(
                    q,
                    k,
                    v,
                    temp,
                    segment_ids,
                    sm_scale=self.scale_config.attn_output_scale(self.config.key_size),
                    cap=self.config.attn_logit_cap,
                    cap_method=self.config.attn_logit_cap_method,
                ), None

        return sharded_mha, ()


class PallasRankerVarlenAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self, **kwargs):
        from xrex.pallas import ranker_attention_varlen

        assert isinstance(self.config, RecsysAttentionConfig)
        max_seq_len = self.config.sequence_len

        def sharded_mha(
            q,
            k,
            v,
            segment_ids,
            segment_ids_k,
            temp,
            cu_seqlens,
        ):
            del segment_ids_k
            assert q.shape[0] == 1, "Seqpack varlen attention expects one packed row per shard."

            return (
                ranker_attention_varlen.mha(
                    q[0],
                    k[0],
                    v[0],
                    temp[0],
                    segment_ids[0],
                    cu_seqlens[0],
                    cu_seqlens[0],
                    max_seqlen_q=max_seq_len,
                    max_seqlen_k=max_seq_len,
                    sm_scale=self.scale_config.attn_output_scale(self.config.key_size),
                    cap=self.config.attn_logit_cap,
                    cap_method=self.config.attn_logit_cap_method,
                    causal=self.config.causal,
                )[None, ...],
                None,
            )

        return sharded_mha, (("cu_seqlens", lambda s: NamedShape(s, ("batch", "replicated"))),)


class CutedslRankerAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self):
        assert isinstance(self.config, RecsysAttentionConfig)
        assert not self.config.causal, "CuTeDSL FA4 does not support causal attention"
        assert self.config.attn_logit_cap <= 0, (
            f"CuTeDSL FA4 does not support softcap (got attn_logit_cap={self.config.attn_logit_cap}). "
            f"Use qk_norm=True instead."
        )
        assert self.config.qk_norm, "CuTeDSL FA4 requires qk_norm=True for stable training"
        from xrex.utils.gpu import GpuArch, gpu_arch

        arch = gpu_arch()
        assert arch in (GpuArch.GB200, GpuArch.GB300), (
            f"CuTeDSL FA4 requires GB200 or GB300 (got {arch})"
        )
        config = self.config
        sm_scale = self.scale_config.attn_output_scale(self.config.key_size)

        from xrex.cutedsl.ranker_attention_fa4 import (
            build_dense_block_sparse_layout,
            ranker_attention_fa4,
        )

        def sharded_mha(q, k, v, segment_ids, segment_ids_k, temp):
            del segment_ids, segment_ids_k, temp
            batch_size, seq_len, num_q_heads, _ = q.shape
            hist_len = config.num_user_prefix_tokens + config.history_seq_len
            bs_layout = build_dense_block_sparse_layout(batch_size, seq_len, hist_len, num_q_heads)
            return ranker_attention_fa4(q, k, v, sm_scale, bs_layout), None

        return sharded_mha, ()


class CutedslRankerVarlenAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self, **kwargs):
        assert isinstance(self.config, RecsysAttentionConfig)
        assert not self.config.causal, "CuTeDSL FA4 does not support causal attention"
        assert self.config.attn_logit_cap <= 0, (
            f"CuTeDSL FA4 does not support softcap (got attn_logit_cap={self.config.attn_logit_cap}). "
            f"Use qk_norm=True instead."
        )
        assert self.config.qk_norm, "CuTeDSL FA4 requires qk_norm=True for stable training"
        from xrex.utils.gpu import GpuArch, gpu_arch

        arch = gpu_arch()
        assert arch in (GpuArch.GB200, GpuArch.GB300), (
            f"CuTeDSL FA4 requires GB200 or GB300 (got {arch})"
        )
        sm_scale = self.scale_config.attn_output_scale(self.config.key_size)

        from xrex.cutedsl.ranker_attention_varlen_fa4 import ranker_attention_varlen_fa4

        def sharded_mha(
            q,
            k,
            v,
            segment_ids,
            segment_ids_k,
            temp,
            bsp_fwd_mask_cnt,
            bsp_fwd_mask_idx,
            bsp_fwd_full_cnt,
            bsp_fwd_full_idx,
            bsp_fwd_diag_cnt,
            bsp_fwd_diag_idx,
            bsp_bwd_mask_cnt,
            bsp_bwd_mask_idx,
            bsp_bwd_full_cnt,
            bsp_bwd_full_idx,
            bsp_bwd_diag_cnt,
            bsp_bwd_diag_idx,
        ):
            del segment_ids, segment_ids_k, temp
            fwd_bs = (
                bsp_fwd_mask_cnt,
                bsp_fwd_mask_idx,
                bsp_fwd_full_cnt,
                bsp_fwd_full_idx,
                bsp_fwd_diag_cnt,
                bsp_fwd_diag_idx,
            )
            bwd_bs = (
                bsp_bwd_mask_cnt,
                bsp_bwd_mask_idx,
                bsp_bwd_full_cnt,
                bsp_bwd_full_idx,
                bsp_bwd_diag_cnt,
                bsp_bwd_diag_idx,
            )
            return (
                ranker_attention_varlen_fa4(
                    q,
                    k,
                    v,
                    sm_scale,
                    block_sparse_layout=(fwd_bs, bwd_bs),
                ),
                None,
            )

        def _rep(s):
            return NamedShape(s, ("batch_attn",) + tuple("replicated" for _ in s[1:]))

        bsp_names = (
            "bsp_fwd_mask_cnt",
            "bsp_fwd_mask_idx",
            "bsp_fwd_full_cnt",
            "bsp_fwd_full_idx",
            "bsp_fwd_diag_cnt",
            "bsp_fwd_diag_idx",
            "bsp_bwd_mask_cnt",
            "bsp_bwd_mask_idx",
            "bsp_bwd_full_cnt",
            "bsp_bwd_full_idx",
            "bsp_bwd_diag_cnt",
            "bsp_bwd_diag_idx",
        )
        extras = tuple((name, _rep) for name in bsp_names)
        return sharded_mha, extras
