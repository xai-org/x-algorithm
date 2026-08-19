import feature_norm
import numpy as np


def test_normalize_rewrites_action_group_from_one_hot():
    s, n = 4, 256
    action_types = np.zeros((s, n), dtype=bool)
    action_types[0, 1] = True
    action_types[1, 21] = True
    action_types[2, 0] = True
    aseq = {
        "action_types": action_types,
        "padding_mask": np.array([1, 1, 1, 0], dtype=bool),
        "client_app_id": np.array([258901, 129032, 3033300, 0], dtype=np.int64),
        "action_group": np.zeros(s, dtype=np.int32),
        "client_platform": np.zeros(s, dtype=np.int32),
    }
    out = feature_norm.normalize_aseq(aseq)
    assert out["action_group"][0] == 1
    assert out["action_group"][1] == 8
    assert out["action_group"][2] == 0
    assert out["client_platform"][0] == 1
    assert out["client_platform"][1] == 2
    assert out["client_platform"][2] == 3
    assert aseq["action_group"][0] == 0


def test_action_types_are_single_set_bit():
    action_types = np.zeros((3, 8), dtype=bool)
    action_types[0, 2] = True
    action_types[1, 5] = True
    action_types[2, 0] = True
    assert (action_types.sum(axis=-1) == 1).all()
    assert np.argmax(action_types, axis=-1).tolist() == [2, 5, 0]
