# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from xrex.models.loss_recsys import multihot_loss_weights

DISTILL_LOSS_TYPES = ("bce", "logit_mse")

_DISTILL_PROB_EPS = 1e-6


def distillation_loss_compute(
    logits: jax.Array,
    teacher_probs: jax.Array,
    padding_mask: jax.Array,
    loss_mask: jax.Array,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
    loss_type: str = "bce",
) -> tuple[jax.Array, jax.Array]:
    if loss_type not in DISTILL_LOSS_TYPES:
        raise ValueError(
            f"distillation loss_type must be one of {DISTILL_LOSS_TYPES}, got {loss_type!r}"
        )
    logits = logits.astype(jnp.float32)
    has_target = ~jnp.isnan(teacher_probs)
    targets = jnp.clip(jnp.where(has_target, teacher_probs, 0.0).astype(jnp.float32), 0.0, 1.0)
    mask_3d = jnp.expand_dims(padding_mask, axis=-1) * loss_mask * has_target
    if loss_type == "logit_mse":
        clipped = jnp.clip(targets, _DISTILL_PROB_EPS, 1.0 - _DISTILL_PROB_EPS)
        teacher_logits = jnp.log(clipped) - jnp.log1p(-clipped)
        per_target = 0.5 * jnp.square(logits - teacher_logits)
    else:
        per_target = optax.sigmoid_binary_cross_entropy(logits, targets)
    masked_loss = per_target * mask_3d
    weights = multihot_loss_weights(padding_mask, raw_weights)
    if raw_weights is not None:
        masked_loss = masked_loss * jnp.expand_dims(raw_weights, axis=-1)
    if normalizer is None:
        normalizer = jnp.sum(weights)
    covered = jnp.sum(weights * has_target.any(-1))
    return jnp.sum(masked_loss) / (normalizer + 1e-10), covered / (normalizer + 1e-10)
