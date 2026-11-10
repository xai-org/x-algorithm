# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from world import TOPIC_NAMES, generate_world


@pytest.mark.parametrize("num_topics", [2, 3, 4, len(TOPIC_NAMES)])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_world_supports_every_small_valid_topic_count(num_topics, seed):
    kwargs = dict(seed=seed, num_users=8, num_authors=16, num_posts=20, num_topics=num_topics)

    world = generate_world(**kwargs)
    repeat = generate_world(**kwargs)

    assert world.num_topics == num_topics
    assert world.num_users == 8
    for user_row in range(world.num_users):
        topics = world.user_topic_ids[user_row]
        weights = world.user_topic_w[user_row]
        valid = topics >= 0
        assert 2 <= valid.sum() <= min(4, num_topics)
        assert np.all(topics[valid] < num_topics)
        assert len(set(topics[valid].tolist())) == valid.sum()
        np.testing.assert_allclose(weights[valid].sum(), 1.0)
        assert np.all(weights[~valid] == 0)
    np.testing.assert_array_equal(world.user_topic_ids, repeat.user_topic_ids)
    np.testing.assert_array_equal(world.user_topic_w, repeat.user_topic_w)
