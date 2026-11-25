# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P

from xrex.utils.sharding import with_sharding_constraint_unless_manual

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


def multihot_loss_weights(padding_mask: jax.Array, raw_weights: jax.Array | None) -> jax.Array:
    mask = padding_mask.astype(jnp.int32)
    return mask if raw_weights is None else mask * raw_weights


def multihot_loss_compute(
    logits: jax.Array,
    raw_targets: jax.Array,
    padding_mask: jax.Array,
    loss_mask: jax.Array,
    raw_weights: jax.Array | None = None,
    one_hot_targets_sharding=P(None),
    normalizer: jax.Array | None = None,
):
    logits = logits.astype(jnp.float32)

    one_hot_targets = with_sharding_constraint_unless_manual(raw_targets, one_hot_targets_sharding)

    assert logits.shape == one_hot_targets.shape

    mask_3d = jnp.expand_dims(padding_mask, axis=-1) * loss_mask
    mask = padding_mask.astype(jnp.int32)

    bce_per_element = optax.sigmoid_binary_cross_entropy(
        logits, one_hot_targets.astype(logits.dtype)
    )

    masked_bce = bce_per_element * mask_3d

    if raw_weights is not None:
        masked_bce = masked_bce * jnp.expand_dims(raw_weights, axis=-1)
    if normalizer is None:
        normalizer = jnp.sum(multihot_loss_weights(padding_mask, raw_weights))

    cross_entropy_loss = jnp.sum(masked_bce) / (normalizer + 1e-10)

    return (
        cross_entropy_loss,
        mask,
    )


def continuous_loss_weights(
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    mask_negatives: bool,
    raw_weights: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    loss_mask = valid_mask & (~negative_sample_mask) if mask_negatives else valid_mask
    return loss_mask, (loss_mask if raw_weights is None else loss_mask * raw_weights)


def continuous_loss_compute(
    gt_raw: jax.Array,
    pred_raw: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    norm_scale: float,
    loss_type: str = "mse",
    mask_negatives: bool = True,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    gt_raw = gt_raw.astype(jnp.float32)
    pred_raw = pred_raw.astype(jnp.float32)

    gt_clamped = jnp.clip(gt_raw, 0.0, norm_scale)
    gt_norm = gt_clamped / norm_scale
    pred_norm = pred_raw

    pred_in_original_units = pred_raw * norm_scale

    loss_mask, weights = continuous_loss_weights(
        valid_mask, negative_sample_mask, mask_negatives, raw_weights
    )
    if normalizer is None:
        normalizer = jnp.sum(weights)

    if loss_type == "mse":
        errors = (pred_norm - gt_norm) ** 2
    elif loss_type == "mae":
        errors = jnp.abs(pred_norm - gt_norm)
    elif loss_type == "huber":
        delta = 1.0
        abs_diff = jnp.abs(pred_norm - gt_norm)
        errors = jnp.where(abs_diff <= delta, 0.5 * abs_diff**2, delta * (abs_diff - 0.5 * delta))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    loss = jnp.sum(errors * weights) / jnp.maximum(normalizer, 1.0)

    return loss, gt_clamped, pred_in_original_units, loss_mask, errors


def cread_log_thresholds(
    norm_scale: float,
    log_min_threshold: float,
    num_thresholds: int,
) -> tuple[float, ...]:
    if num_thresholds < 2:
        raise ValueError(f"num_thresholds must be at least 2, got {num_thresholds}")
    if not 0.0 < log_min_threshold < norm_scale:
        raise ValueError(
            f"log_min_threshold ({log_min_threshold}) must be in (0, norm_scale={norm_scale})"
        )
    ratio = norm_scale / log_min_threshold
    thresholds = [
        log_min_threshold * ratio ** (k / (num_thresholds - 1)) for k in range(num_thresholds - 1)
    ]
    return tuple(thresholds) + (norm_scale,)


def cread_loss_compute(
    gt_raw: jax.Array,
    cread_logits: jax.Array,
    restored_pred: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    thresholds: tuple[float, ...],
    norm_scale: float,
    restoration_weight: float,
    mask_negatives: bool = True,
    sentinel_values: tuple[float, ...] = (),
    raw_weights: jax.Array | None = None,
    restoration_loss_type: str = "mae",
    huber_delta: float = 0.1,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    gt_raw = gt_raw.astype(jnp.float32)
    logits = cread_logits.astype(jnp.float32)
    restored = restored_pred.astype(jnp.float32)

    num_thresholds = logits.shape[-1]
    assert num_thresholds == len(thresholds), (
        f"cread_logits has {num_thresholds} thresholds, expected {len(thresholds)}: {thresholds}"
    )

    gt_clamped = jnp.clip(gt_raw, 0.0, norm_scale)

    threshold_arr = jnp.asarray(thresholds, dtype=jnp.float32)
    targets = (gt_raw[..., None] > threshold_arr).astype(jnp.float32)

    per_element_cls = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets), axis=-1)
    restore_residual = jnp.abs(restored - gt_clamped) / norm_scale
    if restoration_loss_type == "mae":
        per_element_restore = restore_residual
    elif restoration_loss_type == "huber":
        per_element_restore = jnp.where(
            restore_residual <= huber_delta,
            0.5 * restore_residual**2 / huber_delta,
            restore_residual - 0.5 * huber_delta,
        )
    else:
        raise ValueError(f"Unknown restoration_loss_type: {restoration_loss_type}")
    per_element_loss = per_element_cls + restoration_weight * per_element_restore

    if mask_negatives:
        loss_mask = valid_mask.astype(jnp.bool_) & (~negative_sample_mask)
    else:
        loss_mask = valid_mask.astype(jnp.bool_)
    for sentinel in sentinel_values:
        loss_mask = loss_mask & (gt_raw != sentinel)

    weights = loss_mask if raw_weights is None else loss_mask * raw_weights
    if normalizer is None:
        normalizer = jnp.sum(weights)
    num_loss_samples = jnp.maximum(normalizer, 1.0)
    cls_loss = jnp.sum(per_element_cls * weights) / num_loss_samples
    restoration_loss = jnp.sum(per_element_restore * weights) / num_loss_samples
    loss = cls_loss + restoration_weight * restoration_loss

    return loss, gt_clamped, restored, loss_mask, per_element_loss, cls_loss, restoration_loss


class EgmnMixture(NamedTuple):
    log_weights: jax.Array
    rate: jax.Array
    means: jax.Array
    sigmas: jax.Array


def egmn_initial_bias(
    num_gaussians: int, norm_scale: float, unit_secs: float, min_sigma_secs: float
) -> list[float]:
    def inverse_softplus(x: float) -> float:
        return math.log(math.expm1(max(x, 1e-4)))

    exp_mean = 4.0 / unit_secs
    centers = [
        3.0 * (norm_scale / 3.0) ** (k / max(num_gaussians - 1, 1)) / unit_secs
        for k in range(num_gaussians)
    ]
    gauss_weights = [1.0 / c for c in centers]
    total = sum(gauss_weights)
    weights = [0.5] + [0.5 * w / total for w in gauss_weights]
    return (
        [math.log(w) for w in weights]
        + [inverse_softplus(1.0 / exp_mean)]
        + [inverse_softplus(c - exp_mean) for c in centers]
        + [inverse_softplus(0.35 * c - min_sigma_secs / unit_secs) for c in centers]
    )


def egmn_mixture_from_raw(
    raw: jax.Array, num_gaussians: int, min_sigma_secs: float, unit_secs: float
) -> EgmnMixture:
    raw = raw.astype(jnp.float32)
    k = num_gaussians
    log_weights = jax.nn.log_softmax(raw[..., : k + 1], axis=-1)
    rate = jax.nn.softplus(raw[..., k + 1]) + 1e-3
    means = 1.0 / rate[..., None] + jax.nn.softplus(raw[..., k + 2 : 2 * k + 2])
    sigmas = min_sigma_secs / unit_secs + jax.nn.softplus(raw[..., 2 * k + 2 :])
    return EgmnMixture(log_weights, rate, means, sigmas)


def egmn_expected_value(mixture: EgmnMixture, unit_secs: float) -> jax.Array:
    weights = jnp.exp(mixture.log_weights)
    return unit_secs * (
        weights[..., 0] / mixture.rate + jnp.sum(weights[..., 1:] * mixture.means, axis=-1)
    )


def egmn_log_density(mixture: EgmnMixture, value: jax.Array) -> jax.Array:
    log_exp = mixture.log_weights[..., 0] + jnp.log(mixture.rate) - mixture.rate * value
    z = (value[..., None] - mixture.means) / mixture.sigmas
    log_gauss = (
        mixture.log_weights[..., 1:]
        - jnp.log(mixture.sigmas)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * z * z
    )
    return jax.nn.logsumexp(jnp.concatenate([log_exp[..., None], log_gauss], axis=-1), axis=-1)


def egmn_loss_compute(
    gt_raw: jax.Array,
    mixture: EgmnMixture,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    norm_scale: float,
    unit_secs: float,
    entropy_weight: float,
    regression_weight: float,
    regression_loss_type: str,
    mask_negatives: bool = True,
    sentinel_values: tuple[float, ...] = (),
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    gt_raw = gt_raw.astype(jnp.float32)
    gt_clamped = jnp.clip(gt_raw, 0.0, norm_scale)
    value = gt_clamped / unit_secs
    expected = egmn_expected_value(mixture, unit_secs)
    per_element_nll = -egmn_log_density(mixture, value)
    weights_k = jnp.exp(mixture.log_weights)
    per_element_entropy = jnp.sum(weights_k * mixture.log_weights, axis=-1)
    residual = (expected - gt_clamped) / unit_secs
    if regression_loss_type == "mae":
        per_element_regression = jnp.abs(residual)
    elif regression_loss_type == "mse":
        per_element_regression = residual * residual
    elif regression_loss_type == "huber":
        per_element_regression = optax.huber_loss(residual, delta=0.5)
    elif regression_loss_type == "none":
        per_element_regression = jnp.zeros_like(residual)
    else:
        raise ValueError(f"Unknown egmn regression loss: {regression_loss_type}")
    per_element_loss = (
        per_element_nll
        + entropy_weight * per_element_entropy
        + regression_weight * per_element_regression
    )

    if mask_negatives:
        loss_mask = valid_mask.astype(jnp.bool_) & (~negative_sample_mask)
    else:
        loss_mask = valid_mask.astype(jnp.bool_)
    for sentinel in sentinel_values:
        loss_mask = loss_mask & (gt_raw != sentinel)

    weights = loss_mask if raw_weights is None else loss_mask * raw_weights
    if normalizer is None:
        normalizer = jnp.sum(weights)
    num_loss_samples = jnp.maximum(normalizer, 1.0)

    def mean(x: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(loss_mask, x, 0.0) * weights) / num_loss_samples

    stats = {
        "nll": mean(per_element_nll),
        "entropy": mean(per_element_entropy),
        "regression": mean(per_element_regression),
        "exp_weight": mean(weights_k[..., 0]),
        "exp_mean_secs": mean(unit_secs / mixture.rate),
        "max_weight": mean(jnp.max(weights_k, axis=-1)),
    }
    loss = mean(per_element_loss)
    return loss, gt_clamped, expected, loss_mask, per_element_loss, stats


def binary_threshold_loss_compute(
    gt_raw: jax.Array,
    logit: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    threshold: float,
    mask_negatives: bool = True,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    logit = logit.astype(jnp.float32)
    gt_binary = (gt_raw.astype(jnp.float32) > threshold).astype(jnp.float32)
    per_element_loss = optax.sigmoid_binary_cross_entropy(logit, gt_binary)

    loss_mask, weights = continuous_loss_weights(
        valid_mask, negative_sample_mask, mask_negatives, raw_weights
    )
    if normalizer is None:
        normalizer = jnp.sum(weights)
    loss = jnp.sum(per_element_loss * weights) / jnp.maximum(normalizer, 1.0)

    pred_prob = jax.nn.sigmoid(logit)
    return loss, gt_binary, pred_prob, loss_mask, per_element_loss


def tweedie_loss_compute(
    gt_raw: jax.Array,
    pred_raw: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    p: float = 1.5,
    norm_scale: float = 300.0,
    mask_negatives: bool = True,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    gt = jnp.clip(gt_raw.astype(jnp.float32), 0.0, norm_scale)
    pred = jnp.maximum(pred_raw.astype(jnp.float32), 1e-6)

    if abs(p - 1.0) < 1e-8:
        deviance = -gt * jnp.log(pred) + pred
    elif abs(p - 2.0) < 1e-8:
        deviance = gt / pred + jnp.log(pred)
    else:
        log_pred = jnp.log(pred)
        deviance = -gt * jnp.exp((1.0 - p) * log_pred) / (1.0 - p) + jnp.exp(
            (2.0 - p) * log_pred
        ) / (2.0 - p)

    loss_mask, weights = continuous_loss_weights(
        valid_mask, negative_sample_mask, mask_negatives, raw_weights
    )
    if normalizer is None:
        normalizer = jnp.sum(weights)
    loss = jnp.sum(deviance * weights) / jnp.maximum(normalizer, 1.0)

    return loss, gt, pred, loss_mask, deviance
