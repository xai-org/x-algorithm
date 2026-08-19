from __future__ import annotations

import pickle

import numpy as np
import pytest
from score_layout import (
    HEAD_ORDER,
    action_histogram,
    behavioral_summary,
    to_score_row,
)


def test_to_score_row_placement():
    head_names = list(HEAD_ORDER)
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    row = to_score_row(probs, head_names)
    assert row.shape == (8,)
    assert np.array_equal(row, probs)


def test_to_score_row_rejects_unknown_head():
    with pytest.raises(ValueError):
        to_score_row(np.zeros(1, np.float32), ["NotAHead"])


def _mini_batch():
    s, n_at = 8, 16
    action_types = np.zeros((s, n_at), dtype=bool)
    for i, idx in enumerate((3, 3, 3, 4, 4, 1)):
        action_types[i, idx] = True
    pm = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    app_ids = np.array([3033300, 3033300, 0, 42, 3033300, 7, 0, 0], dtype=np.int64)
    idx_to_name = {1: "SERVER_TWEET_FAV", 3: "SERVER_TWEET_CREATE", 4: "SERVER_TWEET_REPLY"}
    return action_types, pm, app_ids, idx_to_name


def test_action_histogram():
    action_types, pm, _, idx_to_name = _mini_batch()
    hist = action_histogram(action_types, pm, idx_to_name)
    assert dict(hist) == {"SERVER_TWEET_CREATE": 3, "SERVER_TWEET_REPLY": 2, "SERVER_TWEET_FAV": 1}


def test_behavioral_summary_dom_client_fail_safe_denominator():
    action_types, pm, app_ids, idx_to_name = _mini_batch()
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    s = behavioral_summary(action_types, pm, app_ids, idx_to_name, name_to_idx)
    assert s["n_create"] == 3
    assert s["dom_create_app_id"] == 3033300
    assert s["dom_create_app_frac"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["n_reply"] == 2
    assert s["dom_reply_app_frac"] == pytest.approx(0.5, abs=1e-4)
    assert "SERVER_TWEET_CREATEx3" in s["decoded_timeline"]


def test_message_payload_survives_restricted_unpickler():
    from pipeline_security import safe_pickle_loads

    payload = {
        "uids": [123, 456],
        "scores": np.random.default_rng(0).random((2, 8)).astype(np.float32),
        "score_ids": ["a", "b"],
        "action_histograms": [[("SERVER_TWEET_FAV", 3)], []],
        "behavioral_summaries": [
            {"n_create": 1, "dom_create_app_id": 3033300, "dom_create_app_frac": 1.0},
            {},
        ],
    }
    out = safe_pickle_loads(pickle.dumps(payload, protocol=5))
    assert out["uids"] == [123, 456]
    assert np.array_equal(out["scores"], payload["scores"])
    assert out["action_histograms"][0][0] == ("SERVER_TWEET_FAV", 3)
