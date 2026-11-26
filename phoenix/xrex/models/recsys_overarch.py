# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math
from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from xrex.models.model_utils import get_parameter
from xrex.models.scaling import ScaleConfig


@dataclass
class OverArch(hk.Module):
    widths: tuple[int, ...]
    scale_config: ScaleConfig
    name: str | None = "over_arch"

    def __call__(self, inputs: jax.Array) -> jax.Array:
        dims = (inputs.shape[-1], *self.widths)
        h = inputs
        with hk.name_scope("mlp"):
            for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
                bound = 1.0 / math.sqrt(fan_in)
                init = hk.initializers.RandomUniform(-bound, bound)
                lr_multiplier = self.scale_config.hidden_lr_multiplier(fan_in)
                with hk.name_scope(f"layer_{i}"):
                    w = get_parameter(
                        "w",
                        [1, fan_in, fan_out],
                        dtype=jnp.float32,
                        init=init,
                        pspec=P(None, None, None),
                        lr_multiplier=lr_multiplier,
                        rms_clip_axes=(-2, -1),
                    )
                    b = get_parameter(
                        "b",
                        [1, fan_out],
                        dtype=jnp.float32,
                        init=init,
                        pspec=P(None, None),
                        lr_multiplier=lr_multiplier,
                    )
                h = jnp.einsum("bcd,dh->bch", h, w[0].astype(h.dtype)) + b[0].astype(h.dtype)
                if i < len(dims) - 2:
                    h = jax.nn.gelu(h)
        return h
