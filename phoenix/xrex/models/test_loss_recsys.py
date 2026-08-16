# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
"""Unit tests for xrex/models/loss_recsys.py.

These exercise the pure loss-computation functions with small, hand-checkable
arrays on CPU. They do not require a GPU, a mesh, or any training
infrastructure, so they can run in any environment (including plain `uv sync`
without the `engine` extra) and in CI.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from xrex.models.loss_recsys import (
    continuous_loss_compute,
    multihot_loss_compute,
    tweedie_loss_compute,
)

# multihot_loss_compute calls jax.lax.with_sharding_constraint with a
# PartitionSpec, which needs (a) an active mesh in context and (b) inputs
# that already carry a matching NamedSharding -- a plain jnp.array gets an
# implicit SingleDeviceSharding, which with_sharding_constraint rejects even
# inside a mesh context. Tests for it run inside this trivial one-device
# mesh, with inputs explicitly (replicated-)sharded via `_shard`, so they
# work in any environment, including CI with no accelerators.
_SINGLE_DEVICE_MESH = Mesh(np.array(jax.devices()).reshape(1), ("x",))
_REPLICATED = NamedSharding(_SINGLE_DEVICE_MESH, P())


def _shard(array_like):
    return jax.device_put(jnp.asarray(array_like), _REPLICATED)


class TestMultihotLossCompute:
    def test_perfect_predictions_give_near_zero_loss(self):
        # Large-magnitude logits of the correct sign essentially saturate the
        # sigmoid, so binary cross-entropy against the matching target should
        # be close to zero.
        logits = _shard([[[10.0, -10.0]]])
        targets = _shard([[[1.0, 0.0]]])
        padding_mask = _shard([[1]])
        loss_mask = _shard(np.ones((1, 1, 2)))

        with jax.set_mesh(_SINGLE_DEVICE_MESH):
            loss, mask = multihot_loss_compute(logits, targets, padding_mask, loss_mask)

        assert loss < 1e-3
        np.testing.assert_array_equal(mask, np.array([[1]]))  # mask keeps padding_mask's shape

    def test_padding_mask_excludes_padded_rows(self):
        # Two rows: one real (correct, low loss), one padded with adversarial
        # logits. If padding is respected, the padded row must not move the
        # aggregate loss.
        logits = _shard([[[10.0, -10.0]], [[-10.0, 10.0]]])
        targets = _shard([[[1.0, 0.0]], [[1.0, 0.0]]])
        padding_mask = _shard([[1], [0]])
        loss_mask = _shard(np.ones((2, 1, 2)))

        with jax.set_mesh(_SINGLE_DEVICE_MESH):
            loss, mask = multihot_loss_compute(logits, targets, padding_mask, loss_mask)

        assert loss < 1e-3
        np.testing.assert_array_equal(mask, np.array([[1], [0]]))

    def test_raw_weights_scale_contribution(self):
        # Two rows with deliberately different per-element loss (row 0: a
        # correct, near-zero-loss prediction; row 1: a wrong, high-loss one)
        # so a weighted vs. unweighted average is actually distinguishable --
        # with identical rows, any weighting reduces to the same constant.
        logits = _shard([[[10.0]], [[-10.0]]])
        targets = _shard([[[1.0]], [[1.0]]])
        padding_mask = _shard([[1], [1]])
        loss_mask = _shard(np.ones((2, 1, 1)))

        with jax.set_mesh(_SINGLE_DEVICE_MESH):
            loss_unweighted, _ = multihot_loss_compute(logits, targets, padding_mask, loss_mask)
            # raw_weights is combined as `mask * raw_weights`, so it must be
            # shaped like padding_mask (batch, seq), not a flat per-row
            # array -- a mismatched shape broadcasts silently instead of
            # raising, so this is worth getting right explicitly.
            loss_row0_only, _ = multihot_loss_compute(
                logits, targets, padding_mask, loss_mask, raw_weights=_shard([[1.0], [0.0]])
            )

        # Zeroing out the high-loss second row's weight should pull the
        # aggregate loss down toward the near-zero loss of the first row
        # alone, well below the unweighted two-row average.
        assert loss_row0_only < loss_unweighted
        assert loss_row0_only < 1e-3

    def test_shape_mismatch_raises(self):
        logits = _shard(np.zeros((1, 1, 2)))
        targets = _shard(np.zeros((1, 1, 3)))
        padding_mask = _shard([[1]])
        loss_mask = _shard(np.ones((1, 1, 3)))

        with jax.set_mesh(_SINGLE_DEVICE_MESH), pytest.raises(AssertionError):
            multihot_loss_compute(logits, targets, padding_mask, loss_mask)


class TestContinuousLossCompute:
    @pytest.mark.parametrize("loss_type", ["mse", "mae", "huber"])
    def test_perfect_prediction_gives_zero_loss(self, loss_type):
        gt = jnp.array([5.0, 10.0, 20.0])
        pred = gt / 100.0  # pred_raw is compared in normalized units
        valid_mask = jnp.array([True, True, True])
        negative_mask = jnp.array([False, False, False])

        loss, gt_clamped, pred_units, loss_mask, errors = continuous_loss_compute(
            gt, pred, valid_mask, negative_mask, norm_scale=100.0, loss_type=loss_type
        )

        assert loss < 1e-6
        np.testing.assert_allclose(gt_clamped, gt, atol=1e-5)

    def test_ground_truth_is_clamped_to_norm_scale(self):
        gt = jnp.array([500.0])  # above norm_scale
        pred = jnp.array([1.0])  # normalized prediction of 1.0 == norm_scale
        valid_mask = jnp.array([True])
        negative_mask = jnp.array([False])

        loss, gt_clamped, _, _, _ = continuous_loss_compute(
            gt, pred, valid_mask, negative_mask, norm_scale=100.0, loss_type="mse"
        )

        assert float(gt_clamped[0]) == pytest.approx(100.0)
        assert loss < 1e-6  # clamped target matches normalized prediction

    def test_mask_negatives_excludes_negative_samples_by_default(self):
        gt = jnp.array([1.0, 1.0])
        pred = jnp.array([1.0, 100.0])  # second entry is wildly wrong
        valid_mask = jnp.array([True, True])
        negative_mask = jnp.array([False, True])  # second entry is a negative sample

        loss, _, _, loss_mask, _ = continuous_loss_compute(
            gt, pred, valid_mask, negative_mask, norm_scale=1.0, loss_type="mse"
        )

        np.testing.assert_array_equal(loss_mask, np.array([True, False]))
        assert loss < 1e-6  # only the correct, non-negative entry counts

    def test_mask_negatives_false_includes_negative_samples(self):
        gt = jnp.array([1.0, 1.0])
        pred = jnp.array([1.0, 100.0])
        valid_mask = jnp.array([True, True])
        negative_mask = jnp.array([False, True])

        loss, _, _, loss_mask, _ = continuous_loss_compute(
            gt, pred, valid_mask, negative_mask,
            norm_scale=1.0, loss_type="mse", mask_negatives=False,
        )

        np.testing.assert_array_equal(loss_mask, np.array([True, True]))
        assert loss > 1.0  # the wildly-wrong entry now contributes

    def test_unknown_loss_type_raises(self):
        gt = jnp.array([1.0])
        pred = jnp.array([1.0])
        valid_mask = jnp.array([True])
        negative_mask = jnp.array([False])

        with pytest.raises(ValueError, match="Unknown loss_type"):
            continuous_loss_compute(
                gt, pred, valid_mask, negative_mask, norm_scale=1.0, loss_type="not_a_loss"
            )

    def test_all_masked_out_does_not_divide_by_zero(self):
        gt = jnp.array([1.0, 1.0])
        pred = jnp.array([1.0, 1.0])
        valid_mask = jnp.array([False, False])
        negative_mask = jnp.array([False, False])

        loss, _, _, _, _ = continuous_loss_compute(
            gt, pred, valid_mask, negative_mask, norm_scale=1.0, loss_type="mse"
        )

        assert jnp.isfinite(loss)


class TestTweedieLossCompute:
    def test_p_near_one_matches_poisson_branch(self):
        # At p == 1 the deviance reduces to the Poisson-like term
        # -gt * log(pred) + pred. Check the function's p=1.0 branch directly
        # against that closed form on a hand-picked point.
        gt = jnp.array([2.0])
        pred = jnp.array([3.0])
        valid_mask = jnp.array([True])
        negative_mask = jnp.array([False])

        loss, gt_out, pred_out, _, deviance = tweedie_loss_compute(
            gt, pred, valid_mask, negative_mask, p=1.0, norm_scale=100.0
        )

        expected_deviance = -2.0 * jnp.log(3.0) + 3.0
        np.testing.assert_allclose(deviance[0], expected_deviance, rtol=1e-5)

    def test_p_near_two_matches_gamma_branch(self):
        gt = jnp.array([2.0])
        pred = jnp.array([3.0])
        valid_mask = jnp.array([True])
        negative_mask = jnp.array([False])

        _, _, _, _, deviance = tweedie_loss_compute(
            gt, pred, valid_mask, negative_mask, p=2.0, norm_scale=100.0
        )

        expected_deviance = 2.0 / 3.0 + jnp.log(3.0)
        np.testing.assert_allclose(deviance[0], expected_deviance, rtol=1e-5)

    def test_general_p_branch_is_finite_and_matches_manual_formula(self):
        gt = jnp.array([2.0])
        pred = jnp.array([3.0])
        p = 1.5

        _, _, _, _, deviance = tweedie_loss_compute(
            gt, pred, jnp.array([True]), jnp.array([False]), p=p, norm_scale=100.0
        )

        log_pred = jnp.log(3.0)
        expected = -2.0 * jnp.exp((1.0 - p) * log_pred) / (1.0 - p) + jnp.exp(
            (2.0 - p) * log_pred
        ) / (2.0 - p)
        assert jnp.isfinite(deviance[0])
        np.testing.assert_allclose(deviance[0], expected, rtol=1e-5)

    def test_ground_truth_clamped_and_prediction_floored(self):
        gt = jnp.array([-5.0, 1000.0])  # below zero and above norm_scale
        pred = jnp.array([-1.0, 1.0])  # non-positive prediction

        _, gt_out, pred_out, _, deviance = tweedie_loss_compute(
            gt, pred, jnp.array([True, True]), jnp.array([False, False]),
            p=1.5, norm_scale=100.0,
        )

        assert float(gt_out[0]) == pytest.approx(0.0)
        assert float(gt_out[1]) == pytest.approx(100.0)
        # jnp.maximum floors this at float32's nearest representable value to
        # 1e-6, which can round a hair below the float64 literal -- compare
        # with a tolerance appropriate to float32 rather than an exact bound.
        assert float(pred_out[0]) == pytest.approx(1e-6, abs=1e-9)
        assert jnp.all(jnp.isfinite(deviance))

    def test_negative_samples_masked_out_by_default(self):
        gt = jnp.array([1.0, 1.0])
        pred = jnp.array([1.0, 1.0])
        valid_mask = jnp.array([True, True])
        negative_mask = jnp.array([False, True])

        _, _, _, loss_mask, _ = tweedie_loss_compute(
            gt, pred, valid_mask, negative_mask, p=1.5, norm_scale=100.0
        )

        np.testing.assert_array_equal(loss_mask, np.array([True, False]))
