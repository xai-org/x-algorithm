# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from xrex.data.recsys.ads_head_masking import EARLY_RELABEL_STREAM_ID

PURCHASE_VALUE_METRIC_PREFIX = "WebPurchaseValue_delayed_website_clicked"
PURCHASE_VALUE_VT_METRIC_PREFIX = "WebPurchaseValue_delayed_website"
PURCHASE_VALUE_EMA_KEY = "purchase_value"

VALUE_EMA_STATE_SIZE = 10


@dataclasses.dataclass(frozen=True)
class ValueBinTable:
    edges: tuple[float, ...]
    values: tuple[float, ...]
    prior: tuple[float, ...]
    smoothing: float

    @property
    def num_bins(self) -> int:
        return len(self.edges) + 1

    @property
    def log_prior(self) -> list[float]:
        return [math.log(q) for q in self.prior]

    def validate(self, name: str) -> None:
        edges = self.edges
        if (
            len(edges) < 2
            or any(not math.isfinite(e) or e <= 0 for e in edges)
            or any(b <= a for a, b in zip(edges, edges[1:]))
        ):
            raise ValueError(f"{name}_bin_edges must be >= 2 positive increasing ratios")
        if len(self.values) != self.num_bins or any(
            not math.isfinite(v) or v <= 0 for v in self.values
        ):
            raise ValueError(f"{name}_bin_values needs one positive value per bin")
        if (
            len(self.prior) != self.num_bins
            or any(not math.isfinite(q) or q <= 0 for q in self.prior)
            or abs(sum(self.prior) - 1.0) > 1e-3
        ):
            raise ValueError(f"{name}_bin_prior must be positive per bin and sum to 1")
        if not math.isfinite(self.smoothing) or self.smoothing < 0:
            raise ValueError(f"{name}_bin_smoothing must be finite and nonnegative")

    def targets(self, ratio: jax.Array) -> jax.Array:
        return value_bin_targets(ratio, self.edges, self.smoothing)

    def readout(self, logits: jax.Array) -> jax.Array:
        return value_bin_readout(logits, self.values)


def value_bin_targets(ratio: jax.Array, edges: tuple[float, ...], smoothing: float) -> jax.Array:
    edges_arr = jnp.asarray(edges, dtype=jnp.float32)
    num_bins = edges_arr.shape[0] + 1
    r = ratio.astype(jnp.float32)
    if smoothing == 0:
        return jax.nn.one_hot(jnp.searchsorted(edges_arr, r, side="right"), num_bins)
    log_edges = jnp.log(edges_arr)
    sigma = smoothing * (log_edges[-1] - log_edges[0]) / (num_bins - 2)
    cdf = ndtr((log_edges - jnp.log(r)[..., None]) / sigma)
    zeros = jnp.zeros_like(cdf[..., :1])
    return jnp.diff(jnp.concatenate([zeros, cdf, zeros + 1.0], axis=-1), axis=-1)


def value_bin_readout(logits: jax.Array, values: tuple[float, ...]) -> jax.Array:
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return jnp.sum(probs * jnp.asarray(values, dtype=jnp.float32), axis=-1)


def fit_value_bin_table(
    labels: jax.Array, edges: tuple[float, ...], smoothing: float
) -> tuple[jax.Array, jax.Array]:
    y = jnp.asarray(labels, dtype=jnp.float32)
    t = value_bin_targets(y, edges, smoothing)
    mass = jnp.sum(t, axis=0)
    values = jnp.sum(t * y[:, None], axis=0) / jnp.maximum(mass, 1e-12)
    return values, mass / y.shape[0]


def value_valid_mask(
    label_valid: jax.Array,
    padding_mask: jax.Array,
    negative_sample_mask: jax.Array,
    sample_source: jax.Array,
    has_click: jax.Array,
    keeper_mask: jax.Array,
    lanes: tuple[int, ...] = (EARLY_RELABEL_STREAM_ID,),
    include_view_through: bool = False,
) -> jax.Array:
    in_lane = functools.reduce(
        jnp.logical_or, [sample_source == lane for lane in lanes], jnp.zeros_like(padding_mask)
    )
    clicked = has_click.astype(jnp.bool_) | include_view_through
    kept = keeper_mask.astype(jnp.bool_) | include_view_through
    return (
        label_valid.astype(jnp.bool_)
        & padding_mask.astype(jnp.bool_)
        & ~negative_sample_mask.astype(jnp.bool_)
        & in_lane
        & clicked
        & kept
    )


def value_valid_weights(
    ratio: jax.Array,
    baseline: jax.Array,
    valid_mask: jax.Array,
    raw_weights: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    ratio = ratio.astype(jnp.float32)
    baseline = baseline.astype(jnp.float32)
    valid = (
        valid_mask.astype(jnp.bool_)
        & jnp.isfinite(ratio)
        & (ratio > 0)
        & jnp.isfinite(baseline)
        & (baseline > 0)
    )
    weights = jnp.ones_like(ratio) if raw_weights is None else raw_weights.astype(jnp.float32)
    valid = valid & jnp.isfinite(weights) & (weights > 0)
    return valid, jnp.where(valid, weights, 0.0)


def value_sums_key(metric_prefix: str) -> str:
    return f"_{metric_prefix}-sums"


def value_binned_sums_key(metric_prefix: str) -> str:
    return f"_{metric_prefix}-binned-sums"


def _trained_loss(
    loss_sum: jax.Array, row_denominator: jax.Array, loss_denominator: jax.Array | None
) -> jax.Array:
    if loss_denominator is None:
        return loss_sum / row_denominator
    return loss_sum / (loss_denominator.astype(jnp.float32) + 1e-10)


def value_ratio_stats(sums: jax.Array, metric_prefix: str) -> dict[str, jax.Array]:
    denominator = jnp.where(sums[4] > 0, sums[4], 1.0)
    return {
        f"{metric_prefix}_loss": sums[0] / denominator,
        f"{metric_prefix}_num_tokens": sums[4],
        f"{metric_prefix}_ratio_mae": sums[1] / denominator,
        f"{metric_prefix}_usd_mae": sums[7] / denominator,
        f"{metric_prefix}_calib": sums[8] / (sums[2] + 1e-12),
    }


def value_ratio_loss_compute(
    raw_ratio: jax.Array,
    pred_ratio: jax.Array,
    baseline_mean_usd: jax.Array,
    valid_mask: jax.Array,
    metric_prefix: str,
    delta: float = 1.0,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
    loss_denominator: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if not 0 < delta < float("inf"):
        raise ValueError("value head Huber delta must be finite and positive")
    ratio = raw_ratio.astype(jnp.float32)
    baseline = baseline_mean_usd.astype(jnp.float32)
    pred = pred_ratio.astype(jnp.float32)
    valid, weights = value_valid_weights(ratio, baseline, valid_mask, raw_weights)
    target = jnp.where(valid, ratio, 0.0)
    error = jnp.where(valid, jnp.where(valid, pred, 0.0) - target, 0.0)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    errors = 0.5 * quadratic**2 + delta * (abs_error - quadratic)
    weight_sum = jnp.sum(weights)
    if normalizer is None:
        normalizer = weight_sum
    row_denominator = jnp.where(normalizer > 0, normalizer, 1.0)
    loss = _trained_loss(jnp.sum(errors * weights), row_denominator, loss_denominator)
    usd_abs_error = abs_error * jnp.where(valid, baseline, 0.0)
    sums = jnp.stack(
        [
            jnp.sum(errors * weights),
            jnp.sum(abs_error * weights),
            jnp.sum(target * weights),
            jnp.sum(jnp.where(valid, baseline, 0.0) * weights),
            weight_sum,
            jnp.sum(valid).astype(jnp.float32),
            jnp.sum(jnp.abs(1.0 - target) * weights),
            jnp.sum(usd_abs_error * weights),
            jnp.sum(jnp.where(valid, pred, 0.0) * weights),
        ]
    )
    stats = value_ratio_stats(sums, metric_prefix)
    stats[value_sums_key(metric_prefix)] = sums
    return loss, stats


def value_binned_stats(sums: jax.Array, metric_prefix: str) -> dict[str, jax.Array]:
    denominator = jnp.where(sums[4] > 0, sums[4], 1.0)
    return {
        f"{metric_prefix}_binned_ce": sums[0] / denominator,
        f"{metric_prefix}_binned_entropy": sums[1] / denominator,
        f"{metric_prefix}_binned_top_prob": sums[2] / denominator,
        f"{metric_prefix}_binned_overflow_prob": sums[3] / denominator,
    }


def value_binned_loss_compute(
    raw_ratio: jax.Array,
    bin_logits: jax.Array,
    baseline_mean_usd: jax.Array,
    valid_mask: jax.Array,
    bins: ValueBinTable,
    metric_prefix: str,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
    loss_denominator: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    ratio = raw_ratio.astype(jnp.float32)
    baseline = baseline_mean_usd.astype(jnp.float32)
    valid, weights = value_valid_weights(ratio, baseline, valid_mask, raw_weights)
    targets = bins.targets(jnp.where(valid, ratio, 1.0))
    log_probs = jax.nn.log_softmax(bin_logits.astype(jnp.float32), axis=-1)
    ce = jnp.where(valid, -jnp.sum(targets * log_probs, axis=-1), 0.0)
    weight_sum = jnp.sum(weights)
    if normalizer is None:
        normalizer = weight_sum
    row_denominator = jnp.where(normalizer > 0, normalizer, 1.0)
    loss = _trained_loss(jnp.sum(ce * weights), row_denominator, loss_denominator)
    probs = jnp.exp(log_probs)
    entropy = jnp.where(valid, -jnp.sum(probs * log_probs, axis=-1), 0.0)
    top_prob = jnp.where(valid, jnp.max(probs, axis=-1), 0.0)
    overflow_prob = jnp.where(valid, probs[..., -1], 0.0)
    sums = jnp.stack(
        [
            jnp.sum(ce * weights),
            jnp.sum(entropy * weights),
            jnp.sum(top_prob * weights),
            jnp.sum(overflow_prob * weights),
            weight_sum,
        ]
    )
    stats = value_binned_stats(sums, metric_prefix)
    stats[value_binned_sums_key(metric_prefix)] = sums
    return loss, stats


def value_head_loss_compute(
    raw_ratio: jax.Array,
    baseline_mean_usd: jax.Array,
    valid_mask: jax.Array,
    metric_prefix: str,
    *,
    pred_ratio: jax.Array,
    bin_logits: jax.Array | None = None,
    bins: ValueBinTable | None = None,
    huber_delta: float = 1.0,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
    loss_denominator: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    loss, stats = value_ratio_loss_compute(
        raw_ratio,
        pred_ratio,
        baseline_mean_usd,
        valid_mask,
        metric_prefix,
        delta=huber_delta,
        raw_weights=raw_weights,
        normalizer=normalizer,
        loss_denominator=loss_denominator,
    )
    if bins is None:
        return loss, stats
    if bin_logits is None:
        raise ValueError(f"{metric_prefix}: a binned value head needs its bin logits")
    loss, binned_stats = value_binned_loss_compute(
        raw_ratio,
        bin_logits,
        baseline_mean_usd,
        valid_mask,
        bins,
        metric_prefix,
        raw_weights,
        normalizer=normalizer,
        loss_denominator=loss_denominator,
    )
    stats.update(binned_stats)
    return loss, stats


def value_head_stats(sums: dict[str, jax.Array], metric_prefix: str) -> dict[str, jax.Array]:
    stats = value_ratio_stats(sums[value_sums_key(metric_prefix)], metric_prefix)
    binned = sums.get(value_binned_sums_key(metric_prefix))
    if binned is not None:
        stats.update(value_binned_stats(binned, metric_prefix))
    return stats


def value_smoothed_stats(
    sums: jax.Array,
    slice_count: jax.Array,
    rce_ema: dict[str, jax.Array],
    batch_size: jax.Array,
    smoothing_windows: tuple[int, ...],
    ema_key: str,
    metric_prefix: str,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    batch_stat = jnp.concatenate([sums, slice_count.astype(jnp.float32)[None]])
    a = jnp.minimum(1.0, batch_size / jnp.array(smoothing_windows, dtype=jnp.float32))[:, None]
    old = jnp.stack(
        [
            rce_ema.get(f"{ema_key}/{ws}", jnp.zeros((VALUE_EMA_STATE_SIZE,), dtype=jnp.float32))
            for ws in smoothing_windows
        ]
    )
    raw_updated = (1.0 - a) * old + a * batch_stat[None, :]
    updated = jnp.where(
        jnp.isnan(raw_updated),
        jnp.where(jnp.isnan(old), batch_stat[None, :], old),
        raw_updated,
    )
    weight = jnp.maximum(updated[:, 4], 1e-12)
    has_labels = updated[:, 5] > 0
    stats: dict[str, jax.Array] = {}
    new_ema: dict[str, jax.Array] = {}
    for i, ws in enumerate(smoothing_windows):
        new_ema[f"{ema_key}/{ws}"] = updated[i]
        prefix = f"{metric_prefix}_smoothed"
        for name, col in (("loss", 0), ("ratio_mae", 1), ("usd_mae", 7)):
            stats[f"{prefix}_{name}_{ws}"] = jnp.where(
                has_labels[i], updated[i, col] / weight[i], 0.0
            )
        stats[f"{prefix}_mae_vs_advertiser_avg_{ws}"] = jnp.where(
            updated[i, 6] > 0, updated[i, 1] / jnp.maximum(updated[i, 6], 1e-12), 0.0
        )
        stats[f"{prefix}_calib_{ws}"] = jnp.where(
            updated[i, 2] > 0, updated[i, 8] / jnp.maximum(updated[i, 2], 1e-12), 0.0
        )
        stats[f"{prefix}_ratio_pos_{ws}"] = updated[i, 4] / jnp.maximum(updated[i, 9], 1.0)
        stats[f"{prefix}_num_tokens_{ws}"] = updated[i, 4]
    return stats, new_ema


WITHIN_LI_SLICES = ("all", "clean", "ct", "vt")
WITHIN_LI_STATE_COLS = 6
WITHIN_LI_CLEAN_MIN_USD = 1.0


def within_li_ema_key(ema_key: str, slice_name: str, window: int) -> str:
    return f"{ema_key}/within_li/{slice_name}/{window}"


def line_item_buckets(line_item_id: jax.Array, num_buckets: int) -> jax.Array:
    bits = num_buckets.bit_length() - 1
    hashed = line_item_id.astype(jnp.uint32) * jnp.uint32(2654435761)
    return (hashed >> jnp.uint32(32 - bits)).astype(jnp.int32)


def within_li_bucket_sums(
    target_ratio: jax.Array,
    pred_ratio: jax.Array,
    weight: jax.Array,
    bucket: jax.Array,
    num_buckets: int,
) -> jax.Array:
    w = weight.astype(jnp.float32)
    keep = w > 0
    y = jnp.where(keep, jnp.log(jnp.maximum(target_ratio.astype(jnp.float32), 1e-6)), 0.0)
    p = jnp.where(keep, jnp.log(jnp.maximum(pred_ratio.astype(jnp.float32), 1e-6)), 0.0)
    rows = jnp.stack([w, w * y, w * p, w * y * y, w * p * p, w * y * p], axis=-1)
    return jax.ops.segment_sum(
        rows.reshape(-1, WITHIN_LI_STATE_COLS),
        bucket.reshape(-1),
        num_segments=num_buckets,
    )


def within_li_correlation(sums: jax.Array) -> jax.Array:
    n = sums[:, 0]
    inv = jnp.where(n > 0, 1.0 / jnp.maximum(n, 1e-12), 0.0)
    ss_yy = jnp.sum(sums[:, 3] - sums[:, 1] ** 2 * inv)
    ss_pp = jnp.sum(sums[:, 4] - sums[:, 2] ** 2 * inv)
    ss_yp = jnp.sum(sums[:, 5] - sums[:, 1] * sums[:, 2] * inv)
    ok = (ss_yy > 1e-9) & (ss_pp > 1e-9)
    return jnp.where(ok, ss_yp / jnp.sqrt(jnp.maximum(ss_yy * ss_pp, 1e-18)), 0.0)


def value_within_li_smoothed_stats(
    raw_ratio: jax.Array,
    baseline_mean_usd: jax.Array,
    valid_mask: jax.Array,
    pred_ratio: jax.Array,
    line_item_id: jax.Array,
    clicked: jax.Array,
    rce_ema: dict[str, jax.Array],
    batch_size: jax.Array,
    windows: tuple[int, ...],
    num_buckets: int,
    ema_key: str,
    metric_prefix: str,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    valid, _ = value_valid_weights(raw_ratio, baseline_mean_usd, valid_mask, None)
    valid = valid & (line_item_id > 0)
    usd = raw_ratio.astype(jnp.float32) * baseline_mean_usd.astype(jnp.float32)
    clicked = clicked.astype(jnp.bool_)
    masks = {
        "all": valid,
        "clean": valid & (usd >= WITHIN_LI_CLEAN_MIN_USD),
        "ct": valid & clicked,
        "vt": valid & ~clicked,
    }
    bucket = line_item_buckets(line_item_id, num_buckets)
    batch_sums = {
        name: within_li_bucket_sums(raw_ratio, pred_ratio, mask, bucket, num_buckets)
        for name, mask in masks.items()
    }
    stats: dict[str, jax.Array] = {}
    new_ema: dict[str, jax.Array] = {}
    for ws in windows:
        decay = 1.0 - jnp.minimum(1.0, batch_size / ws)
        for name, sums in batch_sums.items():
            key = within_li_ema_key(ema_key, name, ws)
            old = rce_ema.get(key, jnp.zeros_like(sums))
            updated = decay * old + sums
            updated = jnp.where(jnp.isnan(updated), old, updated)
            new_ema[key] = updated
            slice_name = "" if name == "all" else f"_{name}"
            stats[f"{metric_prefix}_smoothed_correlation{slice_name}_{ws}"] = within_li_correlation(
                updated
            )
    return stats, new_ema
