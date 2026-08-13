# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
from collections.abc import Callable
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint
from xai_configlib import Config, configclass

from xrex.models.scaling import ScaleConfig
from xrex.models.sharding_context import NamedShape, ShardingContext


@configclass
class AttentionConfig(Config):
    key_size: int
    num_q_heads: int
    num_kv_heads: int
    causal: bool = True
    with_bias: bool = False
    attn_logit_cap: float = -1.0
    attn_logit_cap_method: str = "tanh"
    attn_impl: str = "pallas_attn"
    attn_sharding_namespace: str = "attention"
    fa_version: str = "3"
    qk_norm: bool = False
    rotary: bool = False
    rope_type: str | None = "1d"
    rope_base_exponent: int = 10000
    temperature_scaling_const: int = -1
    qkv_merge: bool = False
    max_num_segments_per_batch: int = 128
    enable_determinism: bool = False


class Attention(hk.Module):
    window_size: int = -1

    def __init__(
        self,
        config: AttentionConfig,
        scale_config: ScaleConfig,
        name: str | None = None,
        sharding_context: ShardingContext = None,
    ):
        super().__init__(name=name)
        self.config = config
        self.scale_config = scale_config
        self.sharding_context = sharding_context

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array | None,
        value: jax.Array | None,
        segment_ids: jax.Array | None,
        segment_ids_k: jax.Array | None,
        temp: jax.Array | None,
        **kwargs,
    ):
        return self.call_attn(query, key, value, segment_ids, segment_ids_k, temp, **kwargs)

    def call_attn(
        self,
        query: jax.Array,
        key: jax.Array | None,
        value: jax.Array | None,
        segment_ids: jax.Array | None,
        segment_ids_k: jax.Array | None,
        temp: jax.Array | None,
        **kwargs,
    ):
        raise NotImplementedError("Please override this method for specific attention impl.")


class CustomAttention(Attention):
    def sharded_custom_op_with_extra_args(
        self, **kwargs
    ) -> tuple[Callable, dict[str, Callable[[str], NamedShape]]]:
        raise NotImplementedError

    def call_attn(self, query, key, value, segment_ids, segment_ids_k, temp, **kwargs):
        assert self.sharding_context is not None

        body_fn, extra_arg_shape_fns = self.sharded_custom_op_with_extra_args(**kwargs)
        sharding_rule = functools.partial(
            self.sharding_context.sharding_specs, namespace=self.config.attn_sharding_namespace
        )

        extra_args = []
        extra_arg_shapes = []
        for n, shape_fn in extra_arg_shape_fns:
            arg = kwargs.get(n)
            assert arg is not None, "Attempted getting arg with name {n} but got None."
            extra_args.append(arg)
            extra_arg_shapes.append(shape_fn(arg.shape))

        in_specs = tuple(
            map(
                sharding_rule,
                [
                    NamedShape(key.shape, ("batch_attn", "replicated", "q_head", "hidden")),
                    NamedShape(key.shape, ("batch_attn", "replicated", "kv_head", "hidden")),
                    NamedShape(key.shape, ("batch_attn", "replicated", "kv_head", "hidden")),
                    NamedShape(segment_ids.shape, ("batch_attn", "replicated")),
                    NamedShape(
                        segment_ids_k.shape if segment_ids_k is not None else segment_ids.shape,
                        ("batch_attn", "replicated"),
                    ),
                    NamedShape(temp.shape, ("batch_attn", "replicated")),
                    *extra_arg_shapes,
                ],
            )
        )

        out_shape = query.shape
        out_spec = tuple(
            map(
                sharding_rule,
                [
                    NamedShape(out_shape, ("batch_attn", "replicated", "q_head", "hidden")),
                    NamedShape((1, 1), ("batch_attn", "q_head")),
                ],
            )
        )

        return shard_map(
            body_fn,
            mesh=self.sharding_context.mesh,
            in_specs=in_specs,
            out_specs=out_spec,
            check_rep=False,
        )(
            query,
            key,
            value,
            segment_ids,
            segment_ids_k if segment_ids_k is not None else segment_ids,
            temp,
            *extra_args,
        )


class PallasAttention(CustomAttention):
    def sharded_custom_op_with_extra_args(self):
        def sharded_mha(q, k, v, segment_ids, segment_ids_k, temp):
            from xrex.pallas import attention as tt_attention

            num_repeat = self.config.num_q_heads // self.config.num_kv_heads
            k = jnp.repeat(k, num_repeat, axis=2)
            v = jnp.repeat(v, num_repeat, axis=2)
            return tt_attention.mha(
                q,
                k,
                v,
                temp,
                segment_ids,
                causal=self.config.causal,
                sm_scale=self.scale_config.attn_output_scale(self.config.key_size),
                cap=self.config.attn_logit_cap,
                backward_pass_impl="triton_split",
                interpret=False,
            ), None

        return sharded_mha, ()


def import_attention_fa3():
    import importlib.util

    if importlib.util.find_spec("xrex.models.attention_fa3") is not None:
        from xrex.models import attention_fa3

        return attention_fa3.FLASH_ATTN_IMPL
    raise NotImplementedError(
        "attn_impl='flash_attn' selects the fa3 training-kernel arm "
        "(xrex.models.attention_fa3, backed by the compiled xrex/cuda/fa3 "
        "kernels), which is not available in this tree. Choose another "
        "attn_impl (e.g. 'jax_attn' or 'pallas_ranker_attn')."
    )


def make_segment_ids_monotonic(segment_ids):
    segment_ids -= jnp.expand_dims(segment_ids[:, 0], axis=-1)
    segment_ids += jnp.expand_dims(jnp.pad(jnp.cumsum(segment_ids[:-1, -1] + 1), (1, 0)), axis=-1)
    return segment_ids


def identity_segment_ids(segment_ids):
    return segment_ids


def qk_tie_segment_ids(segment_ids, segment_ids_k):
    q_row_min = jnp.expand_dims(segment_ids[:, 0], axis=-1)
    k_row_min = jnp.expand_dims(segment_ids_k[:, 0], axis=-1)
    row_min = jnp.minimum(q_row_min, k_row_min)
    segment_ids -= row_min
    segment_ids_k -= row_min
    q_row_max = jnp.expand_dims(jnp.pad(jnp.cumsum(segment_ids[:-1, -1] + 1), (1, 0)), axis=-1)
    k_row_max = jnp.expand_dims(jnp.pad(jnp.cumsum(segment_ids_k[:-1, -1] + 1), (1, 0)), axis=-1)
    row_max = jnp.maximum(q_row_max, k_row_max)
    segment_ids += row_max
    segment_ids_k += row_max
    return segment_ids, segment_ids_k


SEG_IDS_FNS_REGISTRY = {
    "default": make_segment_ids_monotonic,
    "identity": identity_segment_ids,
}


def make_attention_mask(
    query_input: jax.Array,
    key_input: jax.Array,
    pairwise_fn: Callable[..., Any] = jnp.multiply,
    dtype: Any = jnp.bfloat16,
):
    mask = pairwise_fn(jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2))
    mask = jnp.expand_dims(mask, axis=-3)
    return mask.astype(dtype)


def _cap_attention_logits(logits: jax.Array, cap: float, method: str) -> jax.Array:
    if not cap or cap <= 0.0 or method == "none":
        return logits
    if method == "tanh":
        cap_arr = jnp.array(cap, dtype=logits.dtype)
        return cap_arr * jnp.tanh(logits / cap_arr)
    if method == "soft_sign":
        return logits / (1.0 + jnp.abs(logits) / cap)
    raise ValueError(f"attn_logit_cap_method {method!r} is not supported by JaxAttention.")


class JaxAttention(Attention):
    def call_attn(
        self,
        query: jax.Array,
        key: jax.Array | None,
        value: jax.Array | None,
        segment_ids: jax.Array | None,
        segment_ids_k: jax.Array | None,
        temp: jax.Array | None,
        **kwargs,
    ):
        mask = kwargs.get("masks")
        b, t, h, d = query.shape
        _, _, kv_h, _ = key.shape
        assert h % kv_h == 0, f"query_heads {h} must be a multiple of kv_heads {kv_h}"
        assert self.sharding_context is not None

        def sharding_rule(named_shape: NamedShape) -> jax.sharding.NamedSharding:
            return jax.sharding.NamedSharding(
                self.sharding_context.mesh,
                self.sharding_context.sharding_specs(
                    named_shape, namespace=self.config.attn_sharding_namespace
                ),
            )

        query = jnp.reshape(query, (b, t, kv_h, h // kv_h, d))
        query = with_sharding_constraint(
            query,
            sharding_rule(
                NamedShape(
                    query.shape, ("batch_attn", "replicated", "head", "replicated", "hidden")
                ),
            ),
        )

        attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query, key).astype(jnp.float32)
        attn_logits *= self.scale_config.attn_output_scale(self.config.key_size)
        attn_logits = _cap_attention_logits(
            attn_logits, self.config.attn_logit_cap, self.config.attn_logit_cap_method
        )
        if temp is not None:
            attn_logits *= temp[:, None, None, :, None]

        if self.config.causal:
            causal_mask = jnp.tril(jnp.ones((1, 1, t, t))).astype(query.dtype)
            mask = mask * causal_mask
        if segment_ids is not None:
            segment_ids = with_sharding_constraint(
                segment_ids,
                sharding_rule(
                    NamedShape(segment_ids.shape, ("batch_attn", "replicated")),
                ),
            )
            segment_ids_for_keys = segment_ids_k if segment_ids_k is not None else segment_ids
            if segment_ids_k is not None:
                segment_ids_for_keys = with_sharding_constraint(
                    segment_ids_for_keys,
                    sharding_rule(
                        NamedShape(segment_ids_for_keys.shape, ("batch_attn", "replicated")),
                    ),
                )
            segment_mask = make_attention_mask(
                segment_ids, segment_ids_for_keys, jnp.equal, dtype=query.dtype
            )
            mask *= segment_mask

        mask = mask[:, :, None, :, :]

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim} for {mask.shape}/{attn_logits.shape}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits).astype(query.dtype)

        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value)
        attn = with_sharding_constraint(
            attn,
            sharding_rule(
                NamedShape(
                    attn.shape, ("batch_attn", "replicated", "head", "replicated", "hidden")
                ),
            ),
        )

        return attn
