# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import enum
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name


class RematType(enum.IntEnum):
    WHOLE = 0
    SAVE_GB300_RECSYS = 29
    SAVE_H100_RECSYS = 32
    SAVE_H100_RECSYS_PALLAS_ATTN = 33


POST_ATTN_RESIDUAL = "post_attn_residual"
NORM_STATS = "norm_stats"
ATTN_FP8_T = "attn_fp8_t"
DENSE_GELU_FP8_T = "dense_gelu_fp8_t"


def custom_remat_policy(policy: RematType):
    if policy == RematType.WHOLE:
        return jax.checkpoint_policies.save_only_these_names(
            "scalar_stats",
        )
    elif policy == RematType.SAVE_GB300_RECSYS:
        return jax.checkpoint_policies.save_only_these_names(
            "attn_outputs",
            "dense_outputs",
            "dense_outputs_individual",
            "attn",
            "gate_up_proj",
            "dense_up_proj",
            "query_heads",
            "key_heads",
            "value_heads",
            "scalar_stats",
        )
    elif policy == RematType.SAVE_H100_RECSYS:
        return jax.checkpoint_policies.save_only_these_names(
            "query_heads",
            "key_heads",
            "value_heads",
            "query_heads_rope",
            "key_heads_rope",
            "cutedsl_attn_outputs",
            "attn_outputs",
            "gate_up_proj",
            "dense_up_proj",
            "scalar_stats",
            POST_ATTN_RESIDUAL,
            NORM_STATS,
            ATTN_FP8_T,
            DENSE_GELU_FP8_T,
        )
    elif policy == RematType.SAVE_H100_RECSYS_PALLAS_ATTN:
        return jax.checkpoint_policies.save_from_both_policies(
            custom_remat_policy(RematType.SAVE_H100_RECSYS),
            jax.checkpoint_policies.save_only_these_names("pallas_attn_outputs"),
        )
    else:
        raise NotImplementedError(f"Unknown remat policy: {policy}")


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def checkpoint_as_bits(x: jax.Array, name: str, barrier: bool = True) -> jax.Array:
    return checkpoint_name(x, name)


@checkpoint_as_bits.defjvp
def _checkpoint_as_bits_jvp(name, barrier, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    bits_dtype = jnp.dtype(f"uint{8 * x.dtype.itemsize}")
    bits = checkpoint_name(lax.bitcast_convert_type(x, bits_dtype), name)
    saved = lax.bitcast_convert_type(bits, x.dtype)
    return (lax.optimization_barrier(saved) if barrier else saved), x_dot
