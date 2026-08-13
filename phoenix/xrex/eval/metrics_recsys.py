# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.nn import sigmoid
from xai_configlib import configclass

from xrex.data.recsys.constants import engagement_to_ids
from xrex.eval.metrics import ForwardMetrics, SimpleSumAccumulator

_CLAMP_EPS = 1e-7


@dataclass
class RecsysRCEAccumulator:
    sum_ce_example: float = 0.0
    total_valid: float = 0.0
    num_pos: float = 0.0
    num_neg: float = 0.0

    def clear(self):
        self.sum_ce_example = 0.0
        self.total_valid = 0.0
        self.num_pos = 0.0
        self.num_neg = 0.0

    def add(self, ce_example, total_valid, num_pos, num_neg):
        self.sum_ce_example += ce_example
        self.total_valid += total_valid
        self.num_pos += num_pos
        self.num_neg += num_neg

    def get(self):
        if self.total_valid == 0.0 or self.num_pos == 0.0 or self.num_neg == 0.0:
            return jnp.array(0.0)

        bce_ref = self.get_bce()
        ce_model = self.sum_ce_example / self.total_valid
        return 100.0 * (bce_ref - ce_model) / (bce_ref + 1e-12)

    def get_ce(self):
        if self.total_valid == 0.0:
            return 0.0
        return self.sum_ce_example / self.total_valid

    def get_bce(self):
        if self.total_valid == 0.0 or self.num_pos == 0.0 or self.num_neg == 0.0:
            return 0.0
        p_bar = self.num_pos / self.total_valid
        p_bar_c = jnp.clip(p_bar, _CLAMP_EPS, 1.0 - _CLAMP_EPS)
        bce_ref = -(p_bar_c * jnp.log(p_bar_c) + (1 - p_bar_c) * jnp.log(1 - p_bar_c))
        return bce_ref


@dataclass
class RecallAccumulator(ForwardMetrics):
    true_positives: float = 0.0
    total_positives: float = 0.0

    def clear(self):
        self.true_positives = 0.0
        self.total_positives = 0.0

    def add(self, true_positives: float, total_positives: float):
        self.true_positives += true_positives
        self.total_positives += total_positives

    def get(self) -> float:
        if self.total_positives == 0.0:
            return 0.0
        return self.true_positives / self.total_positives


@configclass
class RecsysAccumulatedLossMetrics(ForwardMetrics):
    loss: float = 0.0
    num_batches: float = 0.0

    def clear(self):
        self.loss = 0.0
        self.num_batches = 0.0

    def add(self, batch_loss, num_batches=1):
        self.loss += batch_loss.item()
        self.num_batches += num_batches

    def get(self):
        if self.num_batches == 0.0:
            return 0.0
        return self.loss / self.num_batches


@configclass
class RecsysFloatTracker(ForwardMetrics):
    value: float = 0.0

    def clear(self):
        self.value = 0.0

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


@configclass
class RecsysSumMetrics(ForwardMetrics):
    total: float = 0.0

    def clear(self):
        self.total = 0.0

    def add(self, value: float):
        self.total += float(value)

    def get(self) -> float:
        local = np.array([self.total], dtype=np.float64)
        if jax.process_count() > 1:
            all_counts = np.asarray(multihost_utils.process_allgather(local, tiled=True))
            return float(all_counts.reshape(-1).sum())
        return float(self.total)


@jax.jit
def get_rce(logits, targets, mask, id_array):
    logits = logits.astype(jnp.float32)
    probs = sigmoid(logits)

    y = jnp.any(targets[..., id_array] == 1, axis=-1).astype(jnp.float32)
    p = probs[..., id_array].max(axis=-1)
    p = jnp.clip(p, _CLAMP_EPS, 1.0 - _CLAMP_EPS)

    valid_mask = mask
    total_valid = valid_mask.sum()
    num_pos = (valid_mask * y).sum()
    num_neg = total_valid - num_pos

    p_c = jnp.clip(p, _CLAMP_EPS, 1.0 - _CLAMP_EPS)
    ce_example = -(y * jnp.log(p_c) + (1 - y) * jnp.log(1 - p_c))
    ce_example_sum = (valid_mask * ce_example).sum()
    return ce_example_sum, total_valid, num_pos, num_neg


@configclass
class RecsysRCEMetrics(ForwardMetrics):
    engagement_type: str

    def clear(self):
        self.accumulator.clear()

    def initialize(self):
        self.accumulator = RecsysRCEAccumulator()

    def run_batch(self, logits, targets, mask):
        assert self.accumulator is not None, "Accumulator not initialized"
        assert logits.ndim == 3 and targets.ndim == 3 and mask.ndim == 2, (
            f"logits.ndim: {logits.ndim}, targets.ndim: {targets.ndim}, mask.ndim: {mask.ndim}"
        )
        ids = engagement_to_ids("default")[self.engagement_type]
        id_array = jnp.array(ids)
        ce_example, total_valid, num_pos, num_neg = get_rce(logits, targets, mask, id_array)

        self.accumulator.add(
            ce_example=ce_example,
            total_valid=total_valid,
            num_pos=num_pos,
            num_neg=num_neg,
        )

    def get(self):
        assert self.accumulator is not None, "Accumulator not initialized"
        return self.accumulator.get()

    def get_ratio(self) -> float:
        assert self.accumulator is not None, "Accumulator not initialized"
        return (
            self.accumulator.num_pos / self.accumulator.total_valid
            if self.accumulator.total_valid > 0
            else 0.0
        )


@configclass
class RecsysNumTokensMetrics(ForwardMetrics):
    def clear(self):
        self.accumulator.add(-1 * self.accumulator.get())

    def initialize(self):
        self.accumulator = SimpleSumAccumulator()

    def run_batch(self, logits, targets, mask):
        self.accumulator.add(mask.sum())

    def get(self):
        assert self.accumulator is not None, "Accumulator not initialized"
        return self.accumulator.get()


@configclass
class RecsysCEMetrics(RecsysRCEMetrics):
    def clear(self):
        self.accumulator.clear()

    def get(self):
        assert self.accumulator is not None, "Accumulator not initialized"
        return self.accumulator.get_ce()


@dataclass
class AUROCAccumulator:
    n_thresholds: int = 256

    def clear(self):
        self.tp_counts = jnp.zeros(self.n_thresholds)
        self.tn_counts = jnp.zeros(self.n_thresholds)
        self.fp_counts = jnp.zeros(self.n_thresholds)
        self.fn_counts = jnp.zeros(self.n_thresholds)
        self.total_positives = 0.0
        self.total_negatives = 0.0

    def __init__(self, n_thresholds: int = 256):
        self.n_thresholds = n_thresholds
        self.thresholds: jnp.ndarray = jnp.linspace(0, 1, n_thresholds)[::-1]
        self.tp_counts: jnp.ndarray = jnp.zeros(n_thresholds)
        self.tn_counts: jnp.ndarray = jnp.zeros(n_thresholds)
        self.fp_counts: jnp.ndarray = jnp.zeros(n_thresholds)
        self.fn_counts: jnp.ndarray = jnp.zeros(n_thresholds)
        self.total_positives: float = 0.0
        self.total_negatives: float = 0.0

    def update(self, predictions, labels):
        predictions = jnp.asarray(predictions)
        labels = jnp.asarray(labels)

        def compute_counts(threshold):
            pred = (predictions >= threshold).astype(jnp.float32)
            tp = jnp.sum(pred * labels)
            tn = jnp.sum((1 - pred) * (1 - labels))
            fp = jnp.sum(pred * (1 - labels))
            fn = jnp.sum((1 - pred) * labels)
            return tp, tn, fp, fn

        vcompute = jax.vmap(compute_counts, in_axes=(0,))
        tp, tn, fp, fn = vcompute(self.thresholds)

        self.tp_counts += tp
        self.tn_counts += tn
        self.fp_counts += fp
        self.fn_counts += fn

        self.total_positives += float(jnp.sum(labels))
        self.total_negatives += float(jnp.sum(1 - labels))

    def compute(self):
        tpr = jnp.where(self.total_positives > 0, self.tp_counts / self.total_positives, 0.0)
        fpr = jnp.where(self.total_negatives > 0, self.fp_counts / self.total_negatives, 0.0)
        return fpr, tpr, self.thresholds


@configclass
class RecsysAUROC(ForwardMetrics):
    engagement_type: str | None = None

    def clear(self):
        self.accumulator.clear()

    def initialize(self, n_thresholds=2048):
        self.accumulator = AUROCAccumulator(n_thresholds)

    def update(self, predictions, labels):
        assert self.accumulator is not None, "Accumulator not initialized"

        self.accumulator.update(predictions, labels)

    def get(self):
        assert self.accumulator is not None, "Accumulator not initialized"
        fpr, tpr, _ = self.accumulator.compute()
        dx = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        return jnp.sum(dx * height)

    def get_valid_entries(self, y, p, valid_mask):
        valid_indices = jnp.where(valid_mask > 0)
        y_valid = y[valid_indices].flatten()
        p_valid = p[valid_indices].flatten()
        return y_valid, p_valid

    def run_batch(self, logits, targets, valid_mask):
        assert self.accumulator is not None, "Accumulator not initialized"
        assert self.engagement_type is not None, "Engagement type not set"

        logits = logits.astype(jnp.float32)
        ids = engagement_to_ids("default")[self.engagement_type]
        id_array = jnp.array(ids)
        probs = sigmoid(logits)

        y = jnp.any(targets[..., id_array] == 1, axis=-1).astype(jnp.float32)
        p = probs[..., id_array].max(axis=-1)
        p = jnp.clip(p, _CLAMP_EPS, 1.0 - _CLAMP_EPS)

        y_valid, p_valid = self.get_valid_entries(y, p, valid_mask)
        self.accumulator.update(p_valid, y_valid)


@dataclass
class PRAccumulator:
    n_thresholds: int = 8192

    def clear(self):
        self.tp_counts = jnp.zeros(self.n_thresholds)
        self.fp_counts = jnp.zeros(self.n_thresholds)
        self.fn_counts = jnp.zeros(self.n_thresholds)
        self.total_positives = 0.0
        self.total_negatives = 0.0

    def __init__(self, n_thresholds: int = 8192):
        self.n_thresholds = n_thresholds
        self.thresholds: jnp.ndarray = jnp.linspace(1, 0, n_thresholds)
        self.tp_counts: jnp.ndarray = jnp.zeros(n_thresholds)
        self.fp_counts: jnp.ndarray = jnp.zeros(n_thresholds)
        self.fn_counts: jnp.ndarray = jnp.zeros(n_thresholds)
        self.total_positives: float = 0.0
        self.total_negatives: float = 0.0

    def update(self, predictions, labels):
        predictions = jnp.asarray(predictions)
        labels = jnp.asarray(labels)

        def compute_counts(threshold):
            pred = (predictions >= threshold).astype(jnp.float32)
            tp = jnp.sum(pred * labels)
            fp = jnp.sum(pred * (1 - labels))
            fn = jnp.sum((1 - pred) * labels)
            return tp, fp, fn

        vcompute = jax.vmap(compute_counts, in_axes=(0,))
        tp, fp, fn = vcompute(self.thresholds)

        self.tp_counts += tp
        self.fp_counts += fp
        self.fn_counts += fn
        self.total_positives += float(jnp.sum(labels))
        self.total_negatives += float(jnp.sum(1 - labels))

    def compute(self):
        precision = jnp.where(
            (self.tp_counts + self.fp_counts) > 0,
            self.tp_counts / (self.tp_counts + self.fp_counts),
            0.0,
        )
        recall = jnp.where(self.total_positives > 0, self.tp_counts / self.total_positives, 0.0)
        return precision, recall, self.thresholds


@dataclass
class RecsysPRAUC(ForwardMetrics):
    engagement_type: str | None = None

    def clear(self):
        self.accumulator.clear()

    def initialize(self, n_thresholds=8192):
        self.accumulator = PRAccumulator(n_thresholds)

    def update(self, predictions, labels):
        assert self.accumulator is not None, "Accumulator not initialized"
        self.accumulator.update(predictions, labels)

    def get(self):
        assert self.accumulator is not None, "Accumulator not initialized"
        precision, recall, _ = self.accumulator.compute()
        dx = recall[1:] - recall[:-1]
        height = (precision[1:] + precision[:-1]) / 2
        return jnp.sum(dx * height)

    def get_valid_entries(self, y, p, valid_mask):
        valid_indices = jnp.where(valid_mask > 0)
        y_valid = y[valid_indices].flatten()
        p_valid = p[valid_indices].flatten()
        return y_valid, p_valid

    def run_batch(self, logits, targets, valid_mask):
        assert self.accumulator is not None, "Accumulator not initialized"
        assert self.engagement_type is not None, "Engagement type not set"

        logits = logits.astype(jnp.float32)
        ids = engagement_to_ids("default")[self.engagement_type]
        id_array = jnp.array(ids)
        probs = sigmoid(logits)

        y = jnp.any(targets[..., id_array] == 1, axis=-1).astype(jnp.float32)
        p = probs[..., id_array].max(axis=-1)
        p = jnp.clip(p, _CLAMP_EPS, 1.0 - _CLAMP_EPS)

        y_valid, p_valid = self.get_valid_entries(y, p, valid_mask)
        self.accumulator.update(p_valid, y_valid)


def build_post_id_sorter(all_post_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sort_idx = np.argsort(all_post_ids, kind="stable")
    return all_post_ids[sort_idx], sort_idx


def lookup_post_indices(
    target_post_ids: np.ndarray,
    sorted_ids: np.ndarray,
    sort_idx: np.ndarray,
) -> np.ndarray:
    if sorted_ids.size == 0:
        return np.full(target_post_ids.shape, -1, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, target_post_ids, side="right") - 1
    pos_clipped = np.maximum(pos, 0)
    found = sorted_ids[pos_clipped] == target_post_ids
    return np.where(found, sort_idx[pos_clipped], -1)


@dataclass
class RecallReportState:
    last_tp: float = 0.0
    last_total: float = 0.0
    last_inst: float = 0.0
    ema_tp: float = 0.0
    ema_total: float = 0.0
    needs_baseline: bool = False


@configclass
class RecsysRecallMetrics(ForwardMetrics):
    k: int
    accumulator: RecallAccumulator | None = None
    ema_half_life_reports: float = 50.0

    def clear(self):
        assert self.accumulator is not None, "Accumulator not initialized"
        self.accumulator.clear()
        self._report_state = RecallReportState()

    def _get_report_state(self) -> RecallReportState:
        if not hasattr(self, "_report_state"):
            acc = self.accumulator
            self._report_state = RecallReportState(
                needs_baseline=acc is not None and acc.total_positives > 0.0
            )
        return self._report_state

    def _check_half_life(self) -> None:
        hl = self.ema_half_life_reports
        if not (math.isfinite(hl) and hl > 0.0):
            raise ValueError(f"ema_half_life_reports must be finite and > 0, got {hl}")

    def initialize(self):
        self._check_half_life()
        self.accumulator = RecallAccumulator()
        self.clear()

    def run_batch(self, logits):
        assert self.accumulator is not None, "Accumulator not initialized"
        assert logits.ndim == 2, f"logits.ndim: {logits.ndim}"

        true_positives, total_positives = self.get_recall(logits)
        self.accumulator.add(true_positives, total_positives)

    def add_ratio_counts(self, hits: float, total: float) -> None:
        assert self.accumulator is not None, "Accumulator not initialized"
        if total > 0.0:
            self.accumulator.add(float(hits), float(total))

    def run_batch_global_recall(
        self,
        topk_post_ids: np.ndarray,
        target_post_ids: np.ndarray,
        actions: np.ndarray,
        positive_actions: list[int],
        negative_actions: list[int],
        global_post_ids: np.ndarray,
        is_in_global_posts: np.ndarray | None = None,
    ):
        has_positive_actions = np.sum(actions[:, :, positive_actions], axis=-1) > 0
        has_negative_actions = np.sum(actions[:, :, negative_actions], axis=-1) > 0
        if is_in_global_posts is None:
            is_in_global_posts = np.isin(target_post_ids, global_post_ids)
        valid_positive_targets = has_positive_actions & ~has_negative_actions & is_in_global_posts
        positive_targets = np.where(valid_positive_targets, target_post_ids, -1)
        num_positive_targets = np.sum(valid_positive_targets.astype(np.float32), axis=-1)

        topk_expanded = np.expand_dims(topk_post_ids, axis=2)
        positive_expanded = np.expand_dims(positive_targets, axis=1)

        matches = topk_expanded == positive_expanded
        item_matches = np.any(matches, axis=1)
        valid_matches = np.where(valid_positive_targets, item_matches, False)
        num_valid_matches = valid_matches.sum(axis=-1)

        assert self.accumulator is not None, "Accumulator not initialized"
        true_positives = float(num_valid_matches.sum())
        total_positives = float(num_positive_targets.sum())
        if total_positives > 0.0:
            self.accumulator.add(true_positives, total_positives)

    def run_in_batch_recall_by_forward_fn(
        self,
        scores,
        is_in_global_posts,
        actions: np.ndarray,
        positive_actions: list[int],
    ):
        B = scores.shape[0]
        assert scores.shape[1] % B == 0, f"scores.shape: {scores.shape}, B: {B}"
        C = scores.shape[1] // B

        has_positive_actions = np.sum(actions[:, :, positive_actions], axis=-1) > 0
        valid_positive_mask_np = is_in_global_posts & has_positive_actions
        total_positives = float(valid_positive_mask_np.sum())
        if total_positives == 0.0:
            return

        assert self.accumulator is not None, "Accumulator not initialized"
        num_negative_candidates = (B - 1) * C
        if self.k > num_negative_candidates:
            self.accumulator.add(total_positives, total_positives)
            return

        scores_by_user = scores.reshape(B, B, C)
        target_scores = scores_by_user[jnp.arange(B), jnp.arange(B), :]

        same_user_mask = jnp.eye(B, dtype=jnp.bool_)[:, :, None]
        masked_scores = jnp.where(same_user_mask, jnp.finfo(scores.dtype).min, scores_by_user)
        kth_negative_scores = jax.lax.top_k(masked_scores.reshape(B, B * C), self.k)[0][:, -1]

        valid_positive_mask = jnp.asarray(valid_positive_mask_np)
        target_scores = jnp.where(valid_positive_mask, target_scores, -jnp.inf)
        is_in_top_k = target_scores >= kth_negative_scores[:, None]
        recall = jnp.sum(is_in_top_k.astype(jnp.float32))
        self.accumulator.add(float(recall), total_positives)

    def get(self):
        assert self.accumulator is not None, "Accumulator not initialized"
        local_counts = np.array(
            [self.accumulator.true_positives, self.accumulator.total_positives],
            dtype=np.float64,
        )
        if jax.process_count() > 1:
            all_counts = np.asarray(multihost_utils.process_allgather(local_counts, tiled=True))
            local_counts = all_counts.reshape(-1, 2).sum(axis=0)
        true_positives, total_positives = local_counts
        self._advance_report_state(float(true_positives), float(total_positives))
        if total_positives == 0.0:
            return 0.0
        return float(true_positives / total_positives)

    def _advance_report_state(self, reduced_tp: float, reduced_total: float) -> None:
        s = self._get_report_state()
        if s.needs_baseline:
            s.last_tp = reduced_tp
            s.last_total = reduced_total
            s.needs_baseline = False
            return
        delta_tp = reduced_tp - s.last_tp
        delta_total = reduced_total - s.last_total
        if delta_total <= 0.0:
            return
        self._check_half_life()
        s.last_inst = delta_tp / delta_total
        if s.ema_total == 0.0:
            s.ema_tp = delta_tp
            s.ema_total = delta_total
        else:
            alpha = 1.0 - 0.5 ** (1.0 / self.ema_half_life_reports)
            s.ema_tp += alpha * (delta_tp - s.ema_tp)
            s.ema_total += alpha * (delta_total - s.ema_total)
        s.last_tp = reduced_tp
        s.last_total = reduced_total

    def get_report_extras(self) -> dict[str, float]:
        s = self._get_report_state()
        if s.ema_total <= 0.0:
            return {}
        return {"inst": s.last_inst, "ema": s.ema_tp / s.ema_total}

    def get_recall(self, logits):
        B = logits.shape[0]
        if self.k > B:
            B_float = float(B)
            return B_float, B_float

        indices = jnp.argsort(-logits, axis=-1)
        top_k_indices = indices[:, : self.k]

        true_pos_indices = jnp.arange(B, dtype=indices.dtype)[:, None]
        true_pos_exp_indices = jnp.tile(true_pos_indices, (1, self.k))
        is_in_top_k = jnp.any(top_k_indices == true_pos_exp_indices, axis=-1)

        true_positives = float(is_in_top_k.astype(jnp.float32).sum().item())
        total_positives = float(B)
        return true_positives, total_positives


def merge_report_extras(
    results: Mapping[str, Any], flat_metrics: dict[str, float]
) -> dict[str, float]:
    for eval_name, task in results.items():
        for metric_name, metric in task.metrics or []:
            get_extras = getattr(metric, "get_report_extras", None)
            if get_extras is not None:
                for extra_key, extra_value in get_extras().items():
                    flat_metrics[f"{eval_name}/{metric_name}.{extra_key}"] = extra_value
    return flat_metrics
