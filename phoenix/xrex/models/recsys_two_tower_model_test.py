# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import unittest

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from xrex.cuda.top_k_by_key import gather_selected_validity, top_k_by_key
from xrex.models.recsys_two_tower_model import RecsysTwoTowerModel, RecsysTwoTowerModelConfig
from xrex.models.sharding_context import ShardingContext
from xrex.models.topic_categories import NUM_TOPIC_INT32S


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


class FixedUserModel(RecsysTwoTowerModel):
    def __call__(self, batch, recsys_embeddings, is_training=False):
        return batch, [], {}


class RetrievalValidityTest(unittest.TestCase):
    def setUp(self):
        devices = jax.local_devices()[:2]
        self.context = ShardingContext("test", jax.sharding.Mesh(devices, ("expert",)))
        self.users = jnp.array([[1.0, 0.0]] * 4, dtype=jnp.bfloat16)
        self.posts = jnp.array([[9.0, 0.0], [8.0, 0.0], [7.0, 0.0], [6.0, 0.0]], dtype=jnp.bfloat16)
        self.dataset_types = jnp.array([1, 1, 2, 2], dtype=jnp.int32)

    def forward(self, *, return_validity=True, dataset_ranges=None, **kwargs):
        dynamic_ranges = dataset_ranges if isinstance(dataset_ranges, jax.Array) else None

        @hk.without_apply_rng
        @hk.transform
        def retrieve(users, posts, ranges):
            config = RecsysTwoTowerModelConfig(user_tower_config=None, candidate_tower_config=None)
            model = FixedUserModel(config, None, None, self.context)
            return model.forward(
                users,
                None,
                posts,
                self.dataset_types,
                top_k=3,
                return_validity=return_validity,
                dataset_ranges=dataset_ranges if ranges is None else ranges,
                **kwargs,
            )

        return jax.jit(retrieve.apply)({}, self.users, self.posts, dynamic_ranges)

    def assert_selected(self, result, expected):
        indices, scores, validity = map(np.asarray, result)
        self.assertEqual(indices.shape, scores.shape)
        self.assertEqual(indices.shape, validity.shape)
        self.assertEqual(validity.dtype, np.bool_)
        self.assertEqual([row[valid].tolist() for row, valid in zip(indices, validity)], expected)

    def test_dataset_mask_excludes_top_k_padding(self):
        (result,) = self.forward()
        self.assert_selected(result, [[0, 1]] * 4)

    def test_per_user_eligibility_is_preserved(self):
        eligibility = jnp.array(
            [
                [False, True, True, True],
                [True, False, True, True],
                [False, False, True, True],
                [True, True, True, True],
            ]
        )
        (result,) = self.forward(eligible_mask=eligibility)
        self.assert_selected(result, [[1], [0], [], [0, 1]])

    def test_topic_filter_is_included_in_validity(self):
        topics = jnp.zeros((4, NUM_TOPIC_INT32S), dtype=jnp.int32)
        topics = topics.at[:, 0].set(jnp.array([1, 2, 1, 4]))
        user_topics = topics.at[:, 0].set(jnp.array([2, 1, 0, 4]))
        (result,) = self.forward(topic_bitmaps=topics, topic_user_bitmasks=user_topics)
        self.assert_selected(result, [[1], [0], [0, 1], []])

    def test_dynamic_windows_exclude_padding_and_empty_datasets(self):
        results = self.forward(
            target_dataset_types=(1, 2),
            dataset_ranges=jnp.array([[3, 4], [2, 2]], dtype=jnp.int32),
            dataset_capacities=(3, 3),
        )
        self.assert_selected(results[0], [[3]] * 4)
        self.assert_selected(results[1], [[]] * 4)

    def test_static_slices_allow_underfilled_and_empty_datasets(self):
        results = self.forward(target_dataset_types=(1, 2), dataset_ranges=((1, 2), (2, 2)))
        self.assert_selected(results[0], [[1]] * 4)
        self.assert_selected(results[1], [[]] * 4)

    def test_legacy_outputs_are_unchanged(self):
        for kwargs in (
            {},
            {"dataset_ranges": ((1, 2),)},
            {
                "dataset_ranges": jnp.array([[3, 4]], dtype=jnp.int32),
                "dataset_capacities": (3,),
            },
        ):
            with self.subTest(kwargs=kwargs):
                (legacy,) = self.forward(return_validity=False, **kwargs)
                (with_validity,) = self.forward(**kwargs)
                self.assertEqual(len(legacy), 2)
                for old, new in zip(legacy, with_validity[:2]):
                    np.testing.assert_array_equal(old, new)


if __name__ == "__main__":
    unittest.main()
