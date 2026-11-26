# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections import abc
from typing import Callable, Optional, Sequence, Union

import haiku as hk
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P

from xrex.models.model_utils import get_parameter
from xrex.models.scaling import ScaleConfig
from xrex.utils.sharding import with_sharding_constraint_unless_manual


def rms_norm_fn(
    x: jax.Array,
    name: str,
    lr_multiplier: float = 1.0,
    eps: float = 1e-5,
    create_scale: bool = True,
    weight_decay_mask: float = 0.0,
) -> jax.Array:
    x = RMSNorm(
        axis=-1,
        create_scale=create_scale,
        pspec=P(None),
        lr_multiplier=lr_multiplier,
        name=name,
        eps=eps,
        weight_decay_mask=weight_decay_mask,
    )(x)
    return x


def qk_norm(name: str, scale_config: ScaleConfig) -> "RMSNorm":
    return RMSNorm(
        axis=-1,
        lr_multiplier=scale_config.qk_norm_lr_multiplier(),
        name=name,
        weight_decay_mask=scale_config.ln_weight_decay_mask,
    )


def all_heads_qk_norm(name: str, scale_config: ScaleConfig) -> Callable[[jax.Array], jax.Array]:
    norm = qk_norm(name, scale_config)
    return lambda x: jnp.reshape(norm(jnp.reshape(x, (*x.shape[:2], -1))), x.shape)


class RMSNorm(hk.Module):
    def __init__(
        self,
        axis: Union[int, Sequence[int], slice],
        eps: float = 1e-5,
        scale_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        create_scale: bool = True,
        pspec: Optional[P] = P(None),
        lr_multiplier: float = 1.0,
        weight_decay_mask: float = 0.0,
    ):
        super().__init__(name=name)
        if isinstance(axis, slice):
            self.axis = axis
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, abc.Iterable) and all(isinstance(ax, int) for ax in axis):
            self.axis = tuple(axis)
        else:
            raise ValueError("`axis` should be an int, slice or iterable of ints.")

        self.eps = eps
        self.create_scale = create_scale
        self.reparameterize = weight_decay_mask > 0
        if scale_init is None:
            scale_init = jnp.zeros if self.reparameterize else jnp.ones
        if self.reparameterize and scale_init is not jnp.zeros:
            raise ValueError(
                "RMSNorm: when weight_decay_mask > 0 the layer is "
                "re-parameterized as (1 + scale) * x and `scale_init` must be "
                f"jnp.zeros (got {scale_init!r}). Pass scale_init=jnp.zeros or "
                "leave it as None to use the default."
            )
        self.scale_init = scale_init
        self.pspec = pspec
        self.lr_multiplier = lr_multiplier
        self.weight_decay_mask = weight_decay_mask

    def __call__(self, inputs: jax.Array, stats_name: Optional[str] = None):
        gain = self.gain(inputs.shape[-1]) if self.create_scale else 1.0
        return rms_normalize(inputs, gain, self.eps, stats_name)

    def gain(self, size: int) -> jax.Array:
        scale = get_parameter(
            "scale",
            (size,),
            dtype=jnp.float32,
            init=self.scale_init,
            pspec=self.pspec,
            lr_multiplier=self.lr_multiplier,
            weight_decay_mask=self.weight_decay_mask,
        )
        if self.pspec:
            scale = with_sharding_constraint_unless_manual(scale, self.pspec)
        scale = scale.astype(jnp.float32)
        return 1.0 + scale if self.reparameterize else scale


def rms_normalize(
    inputs: jax.Array, gain: jax.Array | float, eps: float, stats_name: Optional[str] = None
) -> jax.Array:
    fprop_dtype = inputs.dtype
    inputs = inputs.astype(jnp.float32)
    mean_squared = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
    if stats_name is not None:
        mean_squared = checkpoint_name(mean_squared, stats_name)

    normed_inputs = inputs * jax.lax.rsqrt(mean_squared + eps)
    outputs = gain * normed_inputs

    return outputs.astype(fprop_dtype)
