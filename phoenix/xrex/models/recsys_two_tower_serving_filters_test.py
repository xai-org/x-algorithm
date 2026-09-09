# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import unittest

import jax.numpy as jnp
import numpy as np

from xrex.cuda.top_k_by_key import gather_selected_validity, top_k_by_key


class SelectedValidityTest(unittest.TestCase):
    def test_underfilled_top_k_marks_masked_candidates(self):
        eligibility = jnp.array([[False, True, False, False]])
        scores = jnp.array([[3.0, 8.0, 7.0, 6.0]], dtype=jnp.bfloat16)
        masked_scores = jnp.where(eligibility, scores, jnp.finfo(jnp.bfloat16).min)

        _, selected_indices = top_k_by_key(masked_scores, 3, heuristic_pivot_ratio=0.1)
        selected_validity = np.asarray(gather_selected_validity(eligibility, selected_indices))

        self.assertEqual(selected_validity.shape, (1, 3))
        self.assertEqual(int(selected_validity.sum()), 1)
        self.assertTrue(selected_validity[0, 0])


if __name__ == "__main__":
    unittest.main()
