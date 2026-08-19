import heads
import labels
import numpy as np


def test_impression_farmer_sets_no_head():
    for excluded in ("IMPRESSION_FARMER", "RENDER_ONLY_NO_ENGAGEMENT", "VIDEO_VIEW_FARMER"):
        vec, mask = labels.labels_to_vectors([excluded])
        assert not vec.any(), f"{excluded} must set NO head (got {np.flatnonzero(vec)})"
        assert mask.all()


def test_dropped_head_strings_map_to_nothing():
    for dropped in (
        "BOOKMARK_SCRAPER",
        "ALGORITHMIC_TIMING_PATTERN",
        "NOTIFICATION_TRIGGERED_ENGAGE",
        "PHANTOM_CLIENT_ENGAGEMENT",
    ):
        vec, _ = labels.labels_to_vectors([dropped])
        assert not vec.any(), f"{dropped} belongs to an unused class, must map to nothing"


def test_empty_labels_mean_legitimate_user():
    for empty in (None, []):
        vec, mask = labels.labels_to_vectors(empty)
        assert vec[heads.LEGITIMATE_USER_INDEX]
        assert vec.sum() == 1
        assert mask.all()


def test_head_names_route_to_their_index():
    for h in heads.HEADS:
        if h.name == "LegitimateUser":
            continue
        vec, _ = labels.labels_to_vectors([h.name])
        assert vec[h.index] and vec.sum() == 1


def test_flywheel_strings_route_and_multi_label_unions():
    vec, _ = labels.labels_to_vectors(["PURE_LIKE_API_BURST", "TWEET_CREATE_BURST"])
    assert vec[heads.HEAD_INDEX["LikeBot"]]
    assert vec[heads.HEAD_INDEX["TweetSpamBot"]]
    assert vec.sum() == 2


def test_unknown_string_sets_no_head_but_not_legitimate():
    vec, _ = labels.labels_to_vectors(["SOME_FUTURE_LABEL"])
    assert not vec.any()
