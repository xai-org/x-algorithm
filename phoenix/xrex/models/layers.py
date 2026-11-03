# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math
from dataclasses import fields as dc_fields
from functools import partial
from typing import Callable, Optional, Type, TypeAlias

import haiku as hk
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from xrex.cutedsl.qk_norm_rope_backward import (
    qk_norm_rope_backward,
    qk_norm_rope_backward_supported,
)
from xrex.data.recsys.sequence_packing import SequencePackedLayout
from xrex.models.attention import (
    Attention,
    AttentionConfig,
    JaxAttention,
    PallasAttention,
    import_attention_fa3,
)
from xrex.models.fp8 import Fp8Dot, Fp8Gemms
from xrex.models.linear_layer import LinearConfig, LinearImpl
from xrex.models.model_utils import get_parameter
from xrex.models.normalization import all_heads_qk_norm, qk_norm, rms_normalize
from xrex.models.recsys_attention import (
    CutedslRankerAttention,
    CutedslRankerCandidateOnlyAttention,
    CutedslRankerVarlenAttention,
    PallasRankerAttention,
    PallasRankerAttentionInference,
    PallasRankerVarlenAttention,
)
from xrex.models.scaling import ScaleConfig
from xrex.models.sharding_context import NamedShape, ShardingContext
from xrex.utils.model import MemoryModelOutput
from xrex.utils.sharding import shard_map_unless_manual, with_sharding_constraint_unless_manual


def rope_angles(
    dim: int, base_exponent: int, seq_len: int, positions: jax.Array | None = None
) -> jax.Array:
    assert dim % 2 == 0, dim
    exponents = jnp.arange(0, dim, 2, dtype=jnp.float32)
    inv_freq = jnp.asarray(1.0 / (base_exponent ** (exponents / dim)), dtype=jnp.float32)
    if positions is None:
        positions = jnp.arange(seq_len, dtype=jnp.float32)[None]
    return jnp.einsum("bi,j->bij", positions, inv_freq)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    cos, sin = (jnp.tile(factor, (1, 2))[:, :, None] for factor in (cos, sin))
    fprop_dtype = x.dtype
    x = x.astype(jnp.float32)
    return (x * cos + rotate_half(x) * sin).astype(fprop_dtype)


class Linear(LinearImpl):
    def __init__(
        self,
        output_size: int,
        rms_clip_axes: Optional[tuple] = (-2, -1),
        with_bias: bool = False,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        pspec: Optional[P] = None,
        lr_multiplier: float = 1.0,
        init_scale: float = 1.0,
        sharding_context: ShardingContext | None = None,
        name: Optional[str] = None,
        dot: Callable[[jax.Array, jax.Array], jax.Array] = jnp.dot,
    ):
        assert sharding_context is not None

        config = LinearConfig(
            init_scale=init_scale,
            lr_multiplier=lr_multiplier,
        )

        super().__init__(
            config=config,
            output_size=output_size,
            sharding_context=sharding_context,
            rms_clip_axes=rms_clip_axes,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            pspec=pspec,
            name=name,
            dot=dot,
        )


MultiHeadAttentionOutput: TypeAlias = MemoryModelOutput


class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        config: AttentionConfig,
        scale_config: ScaleConfig,
        w_init: Optional[hk.initializers.Initializer] = None,
        model_size: Optional[int] = None,
        sharding_context: ShardingContext | None = None,
        name: Optional[str] = None,
        name_suffix: str = "",
        fp8_gemms: Fp8Gemms = Fp8Gemms.OFF,
        out_proj_fp8_input_name: Optional[str] = None,
    ):
        super().__init__(name=name)
        assert sharding_context is not None, "sharding_context is required."
        self.model_size = model_size
        self.config = config
        self.scale_config = scale_config
        self.sharding_context = sharding_context
        self.fp8_gemms = fp8_gemms
        self.out_proj_fp8_input_name = out_proj_fp8_input_name

        if w_init is None:
            w_init = hk.initializers.VarianceScaling(self.scale_config.attn_init_scale**2)
        self.w_init = w_init

        self.name_suffix = name_suffix

    @property
    def data_axis(self):
        return self.sharding_context.logical_axis_to_physical(
            "batch_attn", namespace=self.config.attn_sharding_namespace
        )

    def __call__(
        self,
        input_to_query: jax.Array,
        input_to_key: Optional[jax.Array],
        input_to_value: Optional[jax.Array],
        mask: Optional[jax.Array] = None,
        segment_ids: Optional[jax.Array] = None,
        segment_ids_k: Optional[jax.Array] = None,
        positions: jax.Array | None = None,
        seqpack_layout: SequencePackedLayout | None = None,
    ) -> MultiHeadAttentionOutput:
        projection = self._linear_projection
        assert segment_ids is not None, "segment_ids is required"

        assert input_to_key is not None
        assert input_to_value is not None
        assert input_to_key.shape[:2] == input_to_value.shape[:2], (
            f"key/value shape: {input_to_key.shape}/{input_to_value.shape}"
        )

        num_q_heads = self.config.num_q_heads
        num_kv_heads = self.config.num_kv_heads
        assert num_q_heads % num_kv_heads == 0

        input_size = input_to_query.shape[-1]
        lr_multiplier = (
            self.scale_config.hidden_lr_multiplier(input_size)
            * self.scale_config.transformer_matrix_lr_scale
        )

        qkv_pspec = P(
            self.data_axis,
            self.sharding_context.logical_axis_to_physical(
                "sequence", namespace=self.config.attn_sharding_namespace
            ),
            self.sharding_context.logical_axis_to_physical(
                "kv_head", namespace=self.config.attn_sharding_namespace
            ),
        )
        qkv_weights_pspec = P(None, None)
        fused_qk_norm_rope = False
        if self.config.qkv_merge:
            qkv = self._qkv_proj(input_size)(input_to_query)
            fused_qk_norm_rope = self._fused_qk_norm_rope_supported(qkv, qkv_pspec)
            if fused_qk_norm_rope:
                query_heads, key_heads, value_heads = self._fused_qk_norm_rope(
                    qkv, positions, qkv_pspec
                )
            else:
                query_heads, key_heads, value_heads = self._qkv_split(qkv)

        else:
            query_heads = projection(
                input_to_query,
                head_size=self.config.key_size,
                num_heads=num_q_heads,
                with_bias=self.config.with_bias,
                name=f"query{self.name_suffix}",
                pspec=qkv_weights_pspec,
                lr_multiplier=lr_multiplier,
            )

            assert input_to_key is not None
            key_heads = projection(
                input_to_key,
                head_size=self.config.key_size,
                num_heads=num_kv_heads,
                with_bias=self.config.with_bias,
                name=f"key{self.name_suffix}",
                pspec=qkv_weights_pspec,
                lr_multiplier=lr_multiplier,
            )

            assert input_to_value is not None
            value_heads = projection(
                input_to_value,
                head_size=self.config.key_size,
                num_heads=num_kv_heads,
                with_bias=self.config.with_bias,
                name=f"value{self.name_suffix}",
                pspec=qkv_weights_pspec,
                lr_multiplier=lr_multiplier,
            )

            query_heads = with_sharding_constraint_unless_manual(query_heads, qkv_pspec)
            key_heads = with_sharding_constraint_unless_manual(key_heads, qkv_pspec)
            value_heads = with_sharding_constraint_unless_manual(value_heads, qkv_pspec)

        if not fused_qk_norm_rope:
            query_heads = checkpoint_name(query_heads, "query_heads")
            key_heads = checkpoint_name(key_heads, "key_heads")
            value_heads = checkpoint_name(value_heads, "value_heads")

        activation_with_qheads_pspec = self.sharding_context.sharding_specs(
            NamedShape(key_heads.shape, names=("batch_attn", "replicated", "q_head", "hidden")),
            namespace=self.config.attn_sharding_namespace,
        )
        activation_with_kvheads_pspec = self.sharding_context.sharding_specs(
            NamedShape(key_heads.shape, names=("batch_attn", "replicated", "kv_head", "hidden")),
            namespace=self.config.attn_sharding_namespace,
        )

        query_heads = with_sharding_constraint_unless_manual(
            query_heads, activation_with_qheads_pspec
        )
        key_heads, value_heads = (
            with_sharding_constraint_unless_manual(x, activation_with_kvheads_pspec)
            for x in (key_heads, value_heads)
        )

        if not fused_qk_norm_rope:
            if self.config.qk_norm:
                query_heads, key_heads = (
                    all_heads_qk_norm(name, self.scale_config)(heads)
                    for name, heads in (("q_norm", query_heads), ("k_norm", key_heads))
                )
            if self.config.rotary:
                key_heads, query_heads = self._apply_rope(
                    key_heads, query_heads, positions=positions
                )
                query_heads = checkpoint_name(query_heads, "query_heads_rope")
                key_heads = checkpoint_name(key_heads, "key_heads_rope")

        if self.config.attn_impl == "recsys_flash_attn":
            raise NotImplementedError(
                f"FA2 has been removed but {self.config.attn_impl=} requires it"
            )

        extra_attn_kwargs: dict[str, jax.Array | None] = {}

        temp = jnp.ones(segment_ids.shape, dtype=jnp.int32)

        attn_class, extra_attn_kwargs = self._get_attn_impl(
            extra_attn_kwargs,
            mask=mask,
            seqpack_layout=seqpack_layout,
        )
        tmp = attn_class(
            self.config,
            self.scale_config,
            sharding_context=self.sharding_context,
        )(
            query_heads,
            key_heads,
            value_heads,
            segment_ids,
            segment_ids_k,
            temp,
            **extra_attn_kwargs,
        )

        attn = tmp[0] if isinstance(tmp, tuple) else tmp

        attn = checkpoint_name(attn, "attn")
        leading_dims = attn.shape[:2]
        attn = jnp.reshape(attn, (*leading_dims, -1))
        attn_output = attn

        if self.config.attn_output_gate:
            if self.fp8_gemms != Fp8Gemms.OFF:
                raise NotImplementedError("FP8 GEMMs with attn_output_gate are not implemented yet")
            z = projection(
                input_to_query,
                head_size=self.config.key_size,
                num_heads=num_q_heads,
                with_bias=self.config.with_bias,
                name=f"gate{self.name_suffix}",
                pspec=qkv_weights_pspec,
                lr_multiplier=lr_multiplier,
            )
            z = jnp.reshape(z, (z.shape[0], z.shape[1], -1))
            z = checkpoint_name(z, "attn_gate")
            if self.config.attn_output_gate_activation == "silu":
                attn = attn * jax.nn.silu(z)
            elif self.config.attn_output_gate_activation == "sigmoid":
                attn = attn * jax.nn.sigmoid(z)
            else:
                raise NotImplementedError(
                    f"Activation {self.config.attn_output_gate_activation} is not supported"
                )

        attn = with_sharding_constraint_unless_manual(
            attn,
            self.sharding_context.sharding_specs(
                NamedShape(attn.shape, names=("batch_attn", "sequence", "kv_head")),
                namespace=self.config.attn_sharding_namespace,
            ),
        )

        lr_multiplier = (
            self.scale_config.attn_output_lr_multiplier(attn.shape[-1])
            * self.scale_config.transformer_matrix_lr_scale
        )

        embeddings = self._out_projection(
            attn,
            self.model_size,
            w_init=self.w_init,
            with_bias=self.config.with_bias,
            pspec=qkv_weights_pspec,
            lr_multiplier=lr_multiplier,
            rms_clip_axes=(-2, -1),
            name=f"out{self.name_suffix}",
        )

        return MultiHeadAttentionOutput(
            output=embeddings,
            query_heads=query_heads.reshape(*query_heads.shape[:2], -1),
            key_heads=key_heads.reshape(*key_heads.shape[:2], -1),
            value_heads=value_heads.reshape(*value_heads.shape[:2], -1),
            attn_output=attn_output,
        )

    @hk.transparent
    def _qkv_proj(self, input_size: int) -> Linear:
        return Linear(
            (self.config.num_q_heads + 2 * self.config.num_kv_heads) * self.config.key_size,
            w_init=self.w_init,
            with_bias=self.config.with_bias,
            name=f"qkv{self.name_suffix}",
            pspec=P(None, None),
            lr_multiplier=self.scale_config.hidden_lr_multiplier(input_size)
            * self.scale_config.transformer_matrix_lr_scale,
            sharding_context=self.sharding_context,
            dot=Fp8Dot(self.fp8_gemms),
        )

    def _qkv_split(self, qkv: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        num_kv_heads = self.config.num_kv_heads
        qkv = jnp.reshape(qkv, (*qkv.shape[:2], num_kv_heads, -1, self.config.key_size))
        qkv_pspec = P(
            self.data_axis,
            self.sharding_context.logical_axis_to_physical(
                "sequence", namespace=self.config.attn_sharding_namespace
            ),
            self.sharding_context.logical_axis_to_physical(
                "kv_head", namespace=self.config.attn_sharding_namespace
            ),
        )
        return shard_map_unless_manual(
            _split_merged_qkv,
            out_specs=(qkv_pspec,) * 3,
            in_specs=qkv_pspec,
            mesh=self.sharding_context.mesh,
        )(qkv)

    def _apply_rope(self, *heads: jax.Array, positions: jax.Array | None) -> tuple[jax.Array, ...]:
        angles = self._rope_angles(heads[0].shape[1], positions)
        cos, sin = jnp.cos(angles), jnp.sin(angles)
        return tuple(apply_rope(x, cos, sin) for x in heads)

    def _rope_angles(self, seq_len: int, positions: jax.Array | None) -> jax.Array:
        if self.config.rope_type != "1d":
            raise NotImplementedError(f"Rope type {self.config.rope_type} is not supported")
        first_positions = None if positions is None else positions[..., 0].astype(jnp.float32)
        config = self.config
        return rope_angles(config.key_size, config.rope_base_exponent, seq_len, first_positions)

    def _fused_qk_norm_rope_supported(self, qkv: jax.Array, qkv_pspec: P) -> bool:
        config, ctx = self.config, self.sharding_context
        heads_split = ctx.logical_group_size("kv_head", config.attn_sharding_namespace) > 1
        rows = qkv.shape[:2]
        if not jax.sharding.get_abstract_mesh().manual_axes:
            rows = NamedSharding(ctx.mesh, P(*qkv_pspec[:2])).shard_shape(rows)
        return (
            config.qk_norm
            and config.rotary
            and not heads_split
            and qk_norm_rope_backward_supported(
                math.prod(rows),
                config.num_q_heads,
                config.num_kv_heads,
                config.key_size,
                qkv.dtype,
                jax.devices()[0],
            )
        )

    @hk.transparent
    def _fused_qk_norm_rope(
        self, qkv: jax.Array, positions: jax.Array | None, qkv_pspec: P
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        config = self.config
        qkv = jnp.reshape(qkv, (*qkv.shape[:2], config.num_kv_heads, -1, config.key_size))
        q_norm, k_norm = (qk_norm(name, self.scale_config) for name in ("q_norm", "k_norm"))
        angles = jnp.broadcast_to(
            self._rope_angles(qkv.shape[1], positions), (*qkv.shape[:2], config.key_size // 2)
        )
        rows_pspec = P(*qkv_pspec[:2])
        return shard_map_unless_manual(
            partial(_qk_norm_rope, eps=q_norm.eps),
            out_specs=(qkv_pspec,) * 3,
            in_specs=(qkv_pspec, P(None), P(None), rows_pspec, rows_pspec),
            mesh=self.sharding_context.mesh,
            check_vma=False,
        )(
            qkv,
            q_norm.gain(config.num_q_heads * config.key_size),
            k_norm.gain(config.num_kv_heads * config.key_size),
            jnp.cos(angles),
            jnp.sin(angles),
        )

    @hk.transparent
    def _get_attn_impl(
        self,
        extra_attn_kwargs: dict[str, jax.Array | None],
        *,
        mask: jax.Array | None,
        seqpack_layout: SequencePackedLayout | None,
    ) -> tuple[Type[Attention], dict[str, jax.Array | None]]:
        match self.config.attn_impl:
            case "jax_attn" | "jax_attn_interleaved":
                extra_attn_kwargs["masks"] = mask
                attn_class = JaxAttention
            case "pallas_attn":
                attn_class = PallasAttention
            case "flash_attn":
                if int(self.config.fa_version) == 2:
                    raise RuntimeError(
                        "FlashAttention2 has been removed. Please use fa_version='3' instead."
                    )
                attn_class = import_attention_fa3()
            case "pallas_ranker_attn":
                attn_class = PallasRankerAttention
            case "pallas_ranker_attn_infer":
                attn_class = PallasRankerAttentionInference
            case "pallas_ranker_varlen_attn":
                assert seqpack_layout is not None
                extra_attn_kwargs["cu_seqlens"] = jnp.asarray(
                    seqpack_layout.cu_seqlens, dtype=jnp.int32
                )
                attn_class = PallasRankerVarlenAttention
            case "cutedsl_ranker_varlen_attn":
                assert seqpack_layout is not None, (
                    "cutedsl_ranker_varlen_attn requires seqpack_layout"
                )
                assert seqpack_layout.block_sparse is not None, (
                    "FA4 block-sparse layout missing on seqpack_layout. "
                    "pack_batch must be called with a block-aligned "
                    "LengthDistribution so block_sparse is constructed."
                )
                attn_class = CutedslRankerVarlenAttention

                def _add_h_axis(arr):
                    arr = jnp.asarray(arr, dtype=jnp.int32)
                    return jnp.broadcast_to(
                        jnp.expand_dims(arr, axis=1),
                        (arr.shape[0], self.config.num_q_heads) + arr.shape[1:],
                    )

                bs = seqpack_layout.block_sparse
                extra_attn_kwargs.update(
                    {f"bsp_{f.name}": _add_h_axis(getattr(bs, f.name)) for f in dc_fields(bs)}
                )
                if seqpack_layout.candidate_cu_seqlens is not None:
                    for name in (
                        "candidate_cu_seqlens",
                        "candidate_token_to_slot",
                        "candidate_key_starts",
                        "candidate_key_counts",
                    ):
                        extra_attn_kwargs[name] = jnp.asarray(
                            getattr(seqpack_layout, name), dtype=jnp.int32
                        )
            case "cutedsl_ranker_attn":
                attn_class = CutedslRankerAttention
            case _:
                raise NotImplementedError(
                    f"Invalid attention implementation: {self.config.attn_impl}"
                )

        return attn_class, extra_attn_kwargs

    @hk.transparent
    def _out_projection(
        self,
        x: jax.Array,
        output_size: int,
        pspec: Optional[P] = None,
        lr_multiplier: float = 1.0,
        init_scale: float = 1.0,
        name: Optional[str] = None,
        rms_clip_axes=(-2, -1),
        with_bias: bool = False,
        w_init=None,
    ) -> jax.Array:
        if w_init is None:
            w_init = self.w_init

        final_projection = Linear(
            output_size,
            w_init=w_init,
            with_bias=with_bias,
            pspec=pspec,
            sharding_context=self.sharding_context,
            lr_multiplier=lr_multiplier,
            init_scale=init_scale,
            rms_clip_axes=rms_clip_axes,
            name=name,
            dot=Fp8Dot(
                self.fp8_gemms,
                input_is_materialized=True,
                transposed_input_checkpoint_name=self.out_proj_fp8_input_name,
            ),
        )
        embeddings = final_projection(x)
        return embeddings

    @hk.transparent
    def _linear_projection(
        self,
        x: jax.Array,
        head_size: int,
        num_heads: int,
        pspec: Optional[P] = None,
        lr_multiplier: float = 1.0,
        init_scale: float = 1.0,
        name: Optional[str] = None,
        rms_clip_axes=(-2, -1),
        with_bias=False,
        w_init=None,
    ) -> jax.Array:
        if w_init is None:
            w_init = self.w_init

        y = Linear(
            num_heads * head_size,
            w_init=w_init,
            with_bias=with_bias,
            name=name,
            pspec=pspec,
            lr_multiplier=lr_multiplier,
            init_scale=init_scale,
            sharding_context=self.sharding_context,
            rms_clip_axes=rms_clip_axes,
            dot=Fp8Dot(self.fp8_gemms),
        )(x)

        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, num_heads, head_size))


class CandidateOnlyAttention(MultiHeadAttention):
    def __call__(
        self,
        candidate_states: jax.Array,
        history_states: jax.Array,
        segment_ids: jax.Array,
        positions: jax.Array | None,
        seqpack_layout: SequencePackedLayout | None,
    ) -> MultiHeadAttentionOutput:
        config = self.config
        assert config.qkv_merge, (
            "Candidate-only attention is not implemented yet for qkv_merge=False. Set "
            "candidate_only_attn_impl=None to run the candidate-only layers with qkv_merge=False."
        )
        num_kv_heads = config.num_kv_heads
        num_q_per_kv = config.num_q_heads // num_kv_heads

        qkv = self._qkv_proj(candidate_states.shape[-1])
        query_heads, candidate_keys, candidate_values = self._qkv_split(qkv(candidate_states))

        def key_value_columns(weight_or_bias: jax.Array) -> jax.Array:
            leading = weight_or_bias.shape[:-1]
            heads = weight_or_bias.reshape(*leading, num_kv_heads, num_q_per_kv + 2, -1)
            return heads[..., num_q_per_kv:, :].reshape(*leading, -1)

        history_keys, history_values = jnp.unstack(
            qkv(history_states, output_columns=key_value_columns).reshape(
                *history_states.shape[:2], num_kv_heads, 2, -1
            ),
            axis=-2,
        )

        def shard_heads(x: jax.Array, head_axis: str) -> jax.Array:
            return with_sharding_constraint_unless_manual(
                x,
                self.sharding_context.sharding_specs(
                    NamedShape(x.shape, ("batch_attn", "replicated", head_axis, "hidden")),
                    namespace=config.attn_sharding_namespace,
                ),
            )

        query_heads = shard_heads(checkpoint_name(query_heads, "query_heads"), "q_head")
        candidate_keys, history_keys = (
            shard_heads(checkpoint_name(k, "key_heads"), "kv_head")
            for k in (candidate_keys, history_keys)
        )
        candidate_values, history_values = (
            shard_heads(checkpoint_name(v, "value_heads"), "kv_head")
            for v in (candidate_values, history_values)
        )

        if config.qk_norm:
            query_heads = all_heads_qk_norm("q_norm", self.scale_config)(query_heads)
            k_norm = all_heads_qk_norm("k_norm", self.scale_config)
            candidate_keys, history_keys = k_norm(candidate_keys), k_norm(history_keys)

        if config.rotary:
            assert positions is not None, "RoPE needs the history and candidate positions"
            history_positions, candidate_positions = jnp.split(
                positions, [history_states.shape[1]], axis=1
            )
            (history_keys,) = self._apply_rope(history_keys, positions=history_positions)
            candidate_keys, query_heads = self._apply_rope(
                candidate_keys, query_heads, positions=candidate_positions
            )
            query_heads = checkpoint_name(query_heads, "query_heads_rope")
            candidate_keys, history_keys = (
                checkpoint_name(k, "key_heads_rope") for k in (candidate_keys, history_keys)
            )

        candidate_region_kwargs = (
            {
                name: jnp.asarray(getattr(seqpack_layout, name), dtype=jnp.int32)
                for name in ("candidate_cu_seqlens", "candidate_key_starts", "candidate_key_counts")
            }
            if seqpack_layout is not None
            else {}
        )
        num_candidates = segment_ids.shape[1] - history_states.shape[1]
        num_user_rows = candidate_states.shape[1] - num_candidates
        if num_user_rows:
            assert candidate_region_kwargs, "User-token rows need seqpack_candidate_packing"
            key_starts = candidate_region_kwargs["candidate_key_starts"]
            assert key_starts.shape[1] == num_user_rows + 1, (key_starts.shape, num_user_rows)
            candidate_region_kwargs["user_key_starts"] = key_starts.at[:, :-1].add(1)
            candidate_region_kwargs["user_key_counts"] = (
                candidate_region_kwargs["candidate_key_counts"] - 1
            )
        attention = self._get_candidate_only_attn_impl()(
            config, self.scale_config, sharding_context=self.sharding_context
        )
        attn, _ = attention(
            query_heads,
            history_keys,
            history_values,
            segment_ids,
            None,
            jnp.ones(segment_ids.shape, dtype=jnp.int32),
            candidate_keys=candidate_keys,
            candidate_values=candidate_values,
            **candidate_region_kwargs,
        )

        assert not config.attn_output_gate, "Candidate-only attention has no output gate"
        attn = checkpoint_name(attn, "attn")
        attn_output = jnp.reshape(attn, (*attn.shape[:2], -1))
        attn = with_sharding_constraint_unless_manual(
            attn_output,
            self.sharding_context.sharding_specs(
                NamedShape(attn_output.shape, ("batch_attn", "sequence", "kv_head")),
                namespace=config.attn_sharding_namespace,
            ),
        )
        embeddings = self._out_projection(
            attn,
            self.model_size,
            w_init=self.w_init,
            with_bias=config.with_bias,
            pspec=P(None, None),
            lr_multiplier=self.scale_config.attn_output_lr_multiplier(attn.shape[-1])
            * self.scale_config.transformer_matrix_lr_scale,
            rms_clip_axes=(-2, -1),
            name=f"out{self.name_suffix}",
        )
        return MultiHeadAttentionOutput(output=embeddings, attn_output=attn_output)

    @hk.transparent
    def _get_candidate_only_attn_impl(self) -> Type[Attention]:
        match self.config.candidate_only_attn_impl:
            case "cutedsl_ranker_candidate_only_attn":
                return CutedslRankerCandidateOnlyAttention
            case _:
                raise ValueError(
                    f"Unknown candidate_only_attn_impl {self.config.candidate_only_attn_impl!r}"
                )


def _split_merged_qkv(qkv: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_q_per_kv = qkv.shape[3] - 2
    query_heads = qkv[:, :, :, :num_q_per_kv]
    key_heads = qkv[:, :, :, num_q_per_kv : num_q_per_kv + 1]
    value_heads = qkv[:, :, :, num_q_per_kv + 1 :]
    return tuple(
        heads.reshape(*qkv.shape[:2], -1, qkv.shape[-1])
        for heads in (query_heads, key_heads, value_heads)
    )


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _qk_norm_rope(
    qkv: jax.Array,
    q_gain: jax.Array,
    k_gain: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _qk_norm_rope_fwd(qkv, q_gain, k_gain, cos, sin, eps)[0]


def _qk_norm_rope_fwd(qkv, q_gain, k_gain, cos, sin, eps):
    query_heads, key_heads, value_heads = _split_merged_qkv(qkv)
    query_heads = checkpoint_name(query_heads, "query_heads")
    key_heads = checkpoint_name(key_heads, "key_heads")
    value_heads = checkpoint_name(value_heads, "value_heads")

    def norm_rope(heads, gain):
        normed = rms_normalize(heads.reshape(*heads.shape[:2], -1), gain, eps)
        return apply_rope(normed.reshape(heads.shape), cos, sin)

    outputs = (
        checkpoint_name(norm_rope(query_heads, q_gain), "query_heads_rope"),
        checkpoint_name(norm_rope(key_heads, k_gain), "key_heads_rope"),
        value_heads,
    )
    return outputs, (query_heads, key_heads, q_gain, k_gain, cos, sin)


def _qk_norm_rope_bwd(eps, residuals, cotangents):
    query_heads, key_heads, q_gain, k_gain, cos, sin = residuals
    dq, dk, dv = cotangents
    dqkv, dq_gain, dk_gain = qk_norm_rope_backward(
        *(x.reshape(-1, *x.shape[2:]) for x in (dq, query_heads, dk, key_heads, dv, cos, sin)),
        q_gain,
        k_gain,
        eps=eps,
    )
    return dqkv.reshape(*key_heads.shape[:2], *dqkv.shape[1:]), dq_gain, dk_gain, None, None


_qk_norm_rope.defvjp(_qk_norm_rope_fwd, _qk_norm_rope_bwd)


def rotate_half(
    x: jax.Array,
) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)
