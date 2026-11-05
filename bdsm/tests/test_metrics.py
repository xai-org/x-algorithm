import numpy as np
import pytest

from metrics import average_precision


@pytest.mark.parametrize("labels", [[1, 0], [0, 1]])
def test_average_precision_groups_equal_scores(labels):
    # The only decision threshold retrieves both users: precision is 1 / 2.
    assert average_precision(np.asarray(labels), np.asarray([0.5, 0.5])) == 0.5


@pytest.mark.parametrize("labels", [[1, 0, 1, 0], [1, 1, 0, 0]])
def test_average_precision_uses_precision_at_each_distinct_threshold(labels):
    # At 0.9, recall increases by 1/2 with precision 1. At 0.5, recall
    # increases by 1/2 with precision 2/3. The order within the tie is irrelevant.
    scores = np.asarray([0.9, 0.5, 0.5, 0.1])
    assert np.isclose(average_precision(np.asarray(labels), scores), 5 / 6)


def test_average_precision_all_equal_scores_matches_positive_fraction():
    labels = np.asarray([1, 0, 1, 0, 0])
    assert average_precision(labels, np.full(5, 0.25)) == 0.4
