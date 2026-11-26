# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xrex.models.model_utils import get_parameter
from xrex.models.sharding_context import ShardingContext


@configclass
class LinearConfig(Config):
    init_scale: float = 1.0
    lr_multiplier: float = 1.0


class LinearImpl(hk.Module):
    def __init__(
        self,
        config: LinearConfig,
        output_size: int,
        sharding_context: ShardingContext,
        rms_clip_axes: Optional[Tuple] = (-2, -1),
        with_bias: bool = False,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        pspec: Optional[P] = None,
        name: Optional[str] = None,
        dot: Callable[[jax.Array, jax.Array], jax.Array] = jnp.dot,
    ):
        super().__init__(name=name)
        self.dot = dot
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

        self.config = config
        self.rms_clip_axes = rms_clip_axes

        self.sharding_context = sharding_context
        self.pspec = pspec

    def __call__(
        self,
        inputs: jax.Array,
        *,
        output_columns: Optional[Callable[[jax.Array], jax.Array]] = None,
    ) -> jax.Array:
        fprop_dtype = inputs.dtype
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size

        w_init = self.w_init
        if w_init is None:
            stddev = self.config.init_scale / math.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)

        w = get_parameter(
            "w",
            [input_size, output_size],
            jnp.float32,
            init=w_init,
            pspec=self.pspec,
            lr_multiplier=self.config.lr_multiplier,
            rms_clip_axes=self.rms_clip_axes,
        )
        if output_columns is not None:
            w = output_columns(w)

        out = self.dot(inputs, w.astype(fprop_dtype))

        if self.with_bias:
            b = get_parameter("b", [self.output_size], jnp.float32, init=self.b_init, pspec=P())
            if output_columns is not None:
                b = output_columns(b)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b.astype(fprop_dtype)

        return out
