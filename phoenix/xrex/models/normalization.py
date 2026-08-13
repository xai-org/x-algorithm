# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections import abc
from collections.abc import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec as P

from xrex.models.model_utils import get_parameter


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


def all_heads_qk_norm_fn(
    x: jax.Array,
    name: str,
    lr_multiplier: float = 1.0,
    eps: float = 1e-5,
    create_scale: bool = True,
    weight_decay_mask: float = 0.0,
) -> jax.Array:
    x_shape = x.shape
    x = jnp.reshape(x, (x_shape[0], x_shape[1], -1))

    x = RMSNorm(
        axis=-1,
        create_scale=create_scale,
        pspec=P(None),
        lr_multiplier=lr_multiplier,
        name=name,
        eps=eps,
        weight_decay_mask=weight_decay_mask,
    )(x)
    x = jnp.reshape(x, x_shape)
    return x


class RMSNorm(hk.Module):
    def __init__(
        self,
        axis: int | Sequence[int] | slice,
        eps: float = 1e-5,
        scale_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
        create_scale: bool = True,
        pspec: P | None = P(None),
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

    def __call__(self, inputs: jax.Array):
        fprop_dtype = inputs.dtype
        param_shape = (inputs.shape[-1],)
        if self.create_scale:
            scale = get_parameter(
                "scale",
                param_shape,
                dtype=jnp.float32,
                init=self.scale_init,
                pspec=self.pspec,
                lr_multiplier=self.lr_multiplier,
                weight_decay_mask=self.weight_decay_mask,
            )
            if self.pspec:
                scale = with_sharding_constraint(scale, self.pspec)
            scale = jnp.broadcast_to(scale.astype(jnp.float32), inputs.shape)
            gain = 1.0 + scale if self.reparameterize else scale
        else:
            gain = 1.0

        inputs = inputs.astype(jnp.float32)
        mean_squared = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
        mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)

        normed_inputs = inputs * jax.lax.rsqrt(mean_squared + self.eps)
        outputs = gain * normed_inputs

        return outputs.astype(fprop_dtype)
