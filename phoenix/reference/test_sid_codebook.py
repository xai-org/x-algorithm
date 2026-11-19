#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sid_io import save_codebook_npz
from sid_codebook import (
    _emeans_alpha0,
    _emeans_batch_size,
    _emeans_step_size,
    _split_dead_codes,
    train_rq_emeans,
)


class EmeansTest(unittest.TestCase):
    def test_schedule_starts_at_alpha0_and_anneals_to_tenth(self):
        self.assertAlmostEqual(_emeans_step_size(0.8, 0, 3), 0.8)
        self.assertAlmostEqual(_emeans_step_size(0.8, 1, 3), 0.44)
        self.assertAlmostEqual(_emeans_step_size(0.8, 2, 3), 0.08)
        self.assertAlmostEqual(_emeans_step_size(0.8, 0, 1), 0.8)

    def test_scaling_laws_produce_bounded_batch_and_step(self):
        quality_batch = _emeans_batch_size(1000, 32, 5, 8)
        throughput_batch = _emeans_batch_size(1000, 32, 5, 64)

        self.assertGreaterEqual(quality_batch, 1)
        self.assertLessEqual(quality_batch, 1000)
        self.assertLess(quality_batch, throughput_batch)
        self.assertGreater(_emeans_alpha0(1000, 32, quality_batch, 5), 0)
        self.assertLessEqual(_emeans_alpha0(1000, 32, quality_batch, 5), 1)

    def test_dead_code_revival_splits_an_active_donor(self):
        codebooks = np.array([[0.0, 0.0], [9.0, 9.0]], dtype=np.float32)
        counts = np.array([20.0, 0.0], dtype=np.float32)
        sums = codebooks * counts[:, None]
        batch = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        revived = _split_dead_codes(
            codebooks,
            counts,
            sums,
            batch,
            np.array([0, 0]),
            np.random.RandomState(7),
            threshold=1.0,
        )

        self.assertEqual(revived, 1)
        self.assertAlmostEqual(float(counts.sum()), 20.0)
        self.assertGreaterEqual(float(counts[1]), 1.0)
        self.assertFalse(np.array_equal(codebooks[0], codebooks[1]))

    def test_rq_emeans_trains_residual_stages(self):
        embeddings = np.array(
            [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32
        )

        codebooks = train_rq_emeans(
            embeddings,
            num_levels=2,
            num_centroids=2,
            num_epochs=1,
            seed=7,
        )

        self.assertEqual(codebooks.shape, (2, 2, 2))
        self.assertTrue(np.isfinite(codebooks).all())

    def test_saved_codebook_records_e_means_epochs(self):
        codebooks = np.zeros((1, 2, 3), dtype=np.float32)
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "codebook.npz"
            save_codebook_npz(path, codebooks, n_iters=5, seed=7, n_train=10)

            with np.load(path, allow_pickle=True) as saved:
                self.assertEqual(str(saved["model_type"]), "rq_emeans")
                self.assertEqual(int(saved["epochs"]), 5)
                self.assertEqual(int(saved["iters"]), 5)


if __name__ == "__main__":
    unittest.main()
