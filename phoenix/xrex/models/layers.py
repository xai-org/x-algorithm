# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from dataclasses import fields as dc_fields
from typing import TypeAlias

import haiku as hk
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec as P

from xrex.data.recsys.sequence_packing import SequencePackedLayout
from xrex.models.attention import (
    Attention,
    AttentionConfig,
    JaxAttention,
    PallasAttention,
    import_attention_fa3,
)
from xrex.models.linear_layer import LinearConfig, LinearImpl
from xrex.models.normalization import all_heads_qk_norm_fn
from xrex.models.recsys_attention import (
    CutedslRankerAttention,
    CutedslRankerVarlenAttention,
    PallasRankerAttention,
    PallasRankerAttentionInference,
    PallasRankerVarlenAttention,
)
from xrex.models.scaling import ScaleConfig
from xrex.models.sharding_context import NamedShape, ShardingContext
from xrex.utils.model import MemoryModelOutput


class RotaryEmbedding(hk.Module):
    def __init__(
        self,
        dim: int,
        name: str | None = None,
        base_exponent: int = 10000,
    ):
        super().__init__(name)
        self.dim = dim
        self.base_exponent = base_exponent
        assert self.dim % 2 == 0

    def __call__(
        self,
        x: jax.Array,
        seq_dim: int,
        offset: int | jax.Array,
        t: jax.Array | None = None,
    ) -> jax.Array:
        fprop_dtype = x.dtype
        x = x.astype(jnp.float32)
        exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = jnp.asarray(
            1.0 / (self.base_exponent ** (exponents / self.dim)), dtype=jnp.float32
        )

        if jnp.shape(offset) == ():
            offset = jnp.expand_dims(offset, 0)

        if t is None:
            t = jnp.arange(x.shape[seq_dim], dtype=jnp.float32) + jnp.expand_dims(offset, -1)
        else:
            t = t + jnp.expand_dims(offset, -1)

        phase = jnp.einsum("bi,j->bij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]

        x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)

        x = x.astype(fprop_dtype)
        return x


class Linear(LinearImpl):
    def __init__(
        self,
        output_size: int,
        rms_clip_axes: tuple | None = (-2, -1),
        with_bias: bool = False,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        pspec: P | None = None,
        lr_multiplier: float = 1.0,
        init_scale: float = 1.0,
        sharding_context: ShardingContext | None = None,
        name: str | None = None,
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
        )


MultiHeadAttentionOutput: TypeAlias = MemoryModelOutput


class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        config: AttentionConfig,
        scale_config: ScaleConfig,
        w_init: hk.initializers.Initializer | None = None,
        model_size: int | None = None,
        sharding_context: ShardingContext | None = None,
        name: str | None = None,
        name_suffix: str = "",
    ):
        super().__init__(name=name)
        assert sharding_context is not None, "sharding_context is required."
        self.model_size = model_size
        self.config = config
        self.scale_config = scale_config
        self.sharding_context = sharding_context

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
        input_to_key: jax.Array | None,
        input_to_value: jax.Array | None,
        mask: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        segment_ids_k: jax.Array | None = None,
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
        num_q_per_kv = num_q_heads // num_kv_heads

        mesh = self.sharding_context.mesh
        input_size = input_to_query.shape[-1]
        lr_multiplier = self.scale_config.hidden_lr_multiplier(input_size)

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
        if self.config.qkv_merge:

            def slicing(input):
                b, s, _, _, d = input.shape
                query_heads = input[:, :, :, :num_q_per_kv]
                key_heads = input[:, :, :, num_q_per_kv : num_q_per_kv + 1]
                value_heads = input[:, :, :, num_q_per_kv + 1 :]
                return (
                    query_heads.reshape(b, s, -1, d),
                    key_heads.reshape(b, s, -1, d),
                    value_heads.reshape(b, s, -1, d),
                )

            qkv = projection(
                input_to_query,
                head_size=self.config.key_size,
                num_heads=num_q_heads + 2 * num_kv_heads,
                with_bias=self.config.with_bias,
                name=f"qkv{self.name_suffix}",
                pspec=qkv_weights_pspec,
                lr_multiplier=lr_multiplier,
            )
            qkv = jnp.reshape(qkv, (qkv.shape[0], qkv.shape[1], num_kv_heads, -1, qkv.shape[-1]))

            out_specs = (qkv_pspec,) * 3
            query_heads, key_heads, value_heads = shard_map(slicing, mesh, qkv_pspec, out_specs)(
                qkv
            )

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

            query_heads = with_sharding_constraint(query_heads, qkv_pspec)
            key_heads = with_sharding_constraint(key_heads, qkv_pspec)
            value_heads = with_sharding_constraint(value_heads, qkv_pspec)

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

        query_heads = with_sharding_constraint(query_heads, activation_with_qheads_pspec)
        key_heads, value_heads = (
            with_sharding_constraint(x, activation_with_kvheads_pspec)
            for x in (key_heads, value_heads)
        )

        add_rope = self.config.rotary

        if self.config.qk_norm:
            norm_fn = all_heads_qk_norm_fn
            qk_lr = self.scale_config.qk_norm_lr_multiplier()
            wd_mask = self.scale_config.ln_weight_decay_mask
            query_heads = norm_fn(
                query_heads, "q_norm", qk_lr, create_scale=True, weight_decay_mask=wd_mask
            )
            key_heads = norm_fn(
                key_heads, "k_norm", qk_lr, create_scale=True, weight_decay_mask=wd_mask
            )

        if add_rope:
            query_heads, key_heads = self._apply_rope(
                query_heads,
                key_heads,
                positions=positions,
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

        attn = with_sharding_constraint(
            attn,
            self.sharding_context.sharding_specs(
                NamedShape(attn.shape, names=("batch_attn", "sequence", "kv_head")),
                namespace=self.config.attn_sharding_namespace,
            ),
        )

        lr_multiplier = self.scale_config.attn_output_lr_multiplier(attn.shape[-1])

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
    def _apply_rope(
        self,
        query_heads: jax.Array,
        key_heads: jax.Array,
        *,
        positions: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        t: jax.Array | None = None
        if positions is not None:
            t = positions[..., 0].astype(jnp.float32)

        match self.config.rope_type:
            case "1d":
                rotate = RotaryEmbedding(
                    dim=self.config.key_size,
                    base_exponent=self.config.rope_base_exponent,
                )
                key_heads, query_heads = map(
                    rotate, (key_heads, query_heads), (1, 1), (0, 0), (t, t)
                )
            case _:
                raise NotImplementedError(f"Rope type {self.config.rope_type} is not supported")

        return query_heads, key_heads

    @hk.transparent
    def _get_attn_impl(
        self,
        extra_attn_kwargs: dict[str, jax.Array | None],
        *,
        mask: jax.Array | None,
        seqpack_layout: SequencePackedLayout | None,
    ) -> tuple[type[Attention], dict[str, jax.Array | None]]:
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
        pspec: P | None = None,
        lr_multiplier: float = 1.0,
        init_scale: float = 1.0,
        name: str | None = None,
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
        )
        embeddings = final_projection(x)
        return embeddings

    @hk.transparent
    def _linear_projection(
        self,
        x: jax.Array,
        head_size: int,
        num_heads: int,
        pspec: P | None = None,
        lr_multiplier: float = 1.0,
        init_scale: float = 1.0,
        name: str | None = None,
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
        )(x)

        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, num_heads, head_size))


def rotate_half(
    x: jax.Array,
) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)
