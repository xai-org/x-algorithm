# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import jax.numpy as jnp

from xai_configlib import configclass
from xrex.models.attention import AttentionConfig, CustomAttention
from xrex.models.sharding_context import NamedShape


@configclass
class RecsysAttentionConfig(AttentionConfig):
    history_seq_len: int = 1024
    candidate_seq_len: int = 128
    sequence_len: int = 1152
    num_user_prefix_tokens: int = 1


class PallasRankerAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self):
        def sharded_mha(q, k, v, segment_ids, segment_ids_k, temp):
            assert isinstance(self.config, RecsysAttentionConfig), (
                "PallasRankerAttention only supports RecsysAttentionConfig"
            )
            assert not self.config.causal, "Causal attention is not supported"
            assert self.config.key_size % 64 == 0, "key_size must be a multiple of 64"

            from xrex.utils.gpu import GpuArch, gpu_arch

            use_fa3 = gpu_arch() == GpuArch.H100 and str(self.config.fa_version) != "2"

            if use_fa3:
                from xrex.pallas import ranker_attention_fa3

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

            from xrex.utils.gpu import GpuArch, gpu_arch

            use_h100_kernel = gpu_arch() == GpuArch.H100

            if use_h100_kernel:
                from xrex.pallas import ranker_attention_fa3

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
        has_cap = self.config.attn_logit_cap > 0 and self.config.attn_logit_cap_method != "none"
        from xrex.utils.gpu import GpuArch, gpu_arch

        arch = gpu_arch()
        assert arch in (GpuArch.A100, GpuArch.H100, GpuArch.GB200, GpuArch.GB300), (
            f"CuTeDSL ranker attention requires A100, H100, GB200 or GB300 (got {arch})"
        )
        assert not (has_cap and arch == GpuArch.A100), (
            "CuTeDSL FA4 attn_logit_cap is SM90/SM100 only (SM80 rejects score_mod with block "
            "sparsity); use qk_norm=True with attn_logit_cap=-1 on A100"
        )
        config = self.config
        sm_scale = self.scale_config.attn_output_scale(self.config.key_size)
        cap = config.attn_logit_cap if has_cap else -1.0
        cap_method = config.attn_logit_cap_method

        from xrex.cutedsl.ranker_attention_fa4 import (
            build_dense_block_sparse_layout,
            ranker_attention_fa4,
        )

        def sharded_mha(q, k, v, segment_ids, segment_ids_k, temp):
            del segment_ids_k, temp
            _, seq_len, num_q_heads, _ = q.shape
            hist_len = config.num_user_prefix_tokens + config.history_seq_len
            hist_valid_len = jnp.sum(segment_ids[:, :hist_len] == 1, axis=-1, dtype=jnp.int32)
            fwd_bs, bwd_bs, valid_upper, valid_lower = build_dense_block_sparse_layout(
                seq_len, hist_len, num_q_heads, hist_valid_len
            )
            return (
                ranker_attention_fa4(
                    q,
                    k,
                    v,
                    sm_scale,
                    (fwd_bs, bwd_bs),
                    valid_block_upper=valid_upper,
                    valid_block_lower=valid_lower,
                    cap=cap,
                    cap_method=cap_method,
                ),
                None,
            )

        return sharded_mha, ()


class CutedslRankerVarlenAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self, **kwargs):
        assert isinstance(self.config, RecsysAttentionConfig)
        assert not self.config.causal, "CuTeDSL FA4 does not support causal attention"
        has_cap = self.config.attn_logit_cap > 0 and self.config.attn_logit_cap_method != "none"
        assert self.config.qk_norm or has_cap, (
            "CuTeDSL FA4 needs logit stabilisation: qk_norm=True or attn_logit_cap > 0"
        )
        from xrex.utils.gpu import GpuArch, gpu_arch

        arch = gpu_arch()
        assert arch in (GpuArch.A100, GpuArch.H100, GpuArch.GB200, GpuArch.GB300), (
            f"CuTeDSL ranker varlen attention requires A100, H100, GB200 or GB300 (got {arch})"
        )
        assert not (has_cap and arch == GpuArch.A100), (
            "CuTeDSL FA4 attn_logit_cap is SM90/SM100 only (SM80 rejects score_mod with block "
            "sparsity); use qk_norm=True with attn_logit_cap=-1 on A100"
        )
        sm_scale = self.scale_config.attn_output_scale(self.config.key_size)
        cap = self.config.attn_logit_cap if has_cap else -1.0
        cap_method = self.config.attn_logit_cap_method

        from xrex.cutedsl.ranker_attention_varlen_candidate_packed_fa4 import (
            ranker_attention_varlen_candidate_packed_fa4,
        )
        from xrex.cutedsl.ranker_attention_varlen_fa4 import (
            ranker_attention_varlen_fa4,
        )

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
            bsp_valid_block_upper,
            bsp_valid_block_lower,
            *candidate_region_args,
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
            if candidate_region_args:
                assert cap <= 0, (
                    "attn_logit_cap is not supported with seqpack_candidate_packing=True"
                )
                candidate_cu_seqlens, candidate_token_to_slot, key_starts, key_counts = (
                    candidate_region_args
                )
                attn_out = ranker_attention_varlen_candidate_packed_fa4(
                    q,
                    k,
                    v,
                    sm_scale,
                    (fwd_bs, bwd_bs),
                    bsp_valid_block_upper,
                    bsp_valid_block_lower,
                    candidate_cu_seqlens[0],
                    key_starts[0],
                    key_counts[0],
                    history_region_len=q.shape[1] - candidate_token_to_slot.shape[-1],
                )
            else:
                attn_out = ranker_attention_varlen_fa4(
                    q,
                    k,
                    v,
                    sm_scale,
                    block_sparse_layout=(fwd_bs, bwd_bs),
                    valid_block_upper=bsp_valid_block_upper,
                    valid_block_lower=bsp_valid_block_lower,
                    cap=cap,
                    cap_method=cap_method,
                )
            return attn_out, None

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
            "bsp_valid_block_upper",
            "bsp_valid_block_lower",
        )
        extras = tuple((name, _rep) for name in bsp_names)
        if kwargs.get("candidate_cu_seqlens") is not None:
            extras += tuple(
                (name, _rep)
                for name in (
                    "candidate_cu_seqlens",
                    "candidate_token_to_slot",
                    "candidate_key_starts",
                    "candidate_key_counts",
                )
            )
        return sharded_mha, extras


class CutedslRankerCandidateOnlyAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self, **kwargs):
        assert isinstance(self.config, RecsysAttentionConfig)
        assert not (
            self.config.attn_logit_cap > 0 and self.config.attn_logit_cap_method != "none"
        ), (
            "The candidate-only kernel has no logit cap. Set candidate_only_attn_impl=None to run "
            "the candidate-only layers with attn_logit_cap."
        )
        sm_scale = self.scale_config.attn_output_scale(self.config.key_size)

        from xrex.cutedsl.ranker_attention_candidate_only_fa4 import (
            ranker_attention_candidate_only_fa4,
            ranker_attention_varlen_candidate_only_fa4,
        )

        has_user_rows = kwargs.get("user_key_starts") is not None

        def sharded_mha(
            q,
            k,
            v,
            segment_ids,
            segment_ids_k,
            temp,
            candidate_keys,
            candidate_values,
            *candidate_region_args,
        ):
            del segment_ids_k, temp
            if candidate_region_args:
                assert q.shape[0] == 1, "Seqpack varlen attention expects one packed row per shard."
                if has_user_rows:
                    (
                        q,
                        k,
                        v,
                        own_k,
                        own_v,
                        cu_seqlens,
                        key_starts,
                        key_counts,
                        user_key_starts,
                        user_key_counts,
                    ) = (
                        x[0]
                        for x in (q, k, v, candidate_keys, candidate_values, *candidate_region_args)
                    )
                    num_users = user_key_counts.shape[0]
                    region = q.shape[0] - num_users
                    region_out = ranker_attention_varlen_candidate_only_fa4(
                        q[:region],
                        k,
                        v,
                        own_k[:region],
                        own_v[:region],
                        cu_seqlens,
                        key_starts,
                        key_counts,
                        sm_scale,
                    )
                    user_out = ranker_attention_varlen_candidate_only_fa4(
                        q[region:],
                        k,
                        v,
                        own_k[region:],
                        own_v[region:],
                        jnp.arange(num_users + 1, dtype=jnp.int32),
                        user_key_starts,
                        user_key_counts,
                        sm_scale,
                    )
                    return jnp.concatenate([region_out, user_out])[None], None
                row_args = (
                    x[0]
                    for x in (q, k, v, candidate_keys, candidate_values, *candidate_region_args)
                )
                return ranker_attention_varlen_candidate_only_fa4(*row_args, sm_scale)[None], None
            history_lengths = jnp.sum(segment_ids[:, : k.shape[1]] == 1, axis=-1, dtype=jnp.int32)
            output = ranker_attention_candidate_only_fa4(
                q, k, v, candidate_keys, candidate_values, history_lengths, sm_scale
            )
            return output, None

        def _kv_heads(s):
            return NamedShape(s, ("batch_attn", "replicated", "kv_head", "hidden"))

        def _rep(s):
            return NamedShape(s, ("batch_attn",) + tuple("replicated" for _ in s[1:]))

        extras = (("candidate_keys", _kv_heads), ("candidate_values", _kv_heads))
        if kwargs.get("candidate_cu_seqlens") is not None:
            extras += tuple(
                (name, _rep)
                for name in ("candidate_cu_seqlens", "candidate_key_starts", "candidate_key_counts")
            )
        if has_user_rows:
            extras += (("user_key_starts", _rep), ("user_key_counts", _rep))
        return sharded_mha, extras
