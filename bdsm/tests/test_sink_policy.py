import dataclasses
import importlib.util
import os

import pytest

pytest.importorskip("requests")
pytest.importorskip("yaml")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNTIME = os.path.join(_ROOT, "runtime")


def _load_sink_module():
    spec = importlib.util.spec_from_file_location(
        "score_results_sink_focal",
        os.path.join(_RUNTIME, "score_results_sink_focal.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shipped_policy_matches_baked_in_defaults():
    m = _load_sink_module()
    pol = m._load_policy(os.path.join(_RUNTIME, "sink_policy.yaml"))
    got = dataclasses.asdict(pol)
    want = dataclasses.asdict(m.DEFAULT_POLICY)
    for k in ("version", "source"):
        got.pop(k)
        want.pop(k)
    assert got == want, "sink_policy.yaml drifted from the baked-in defaults"
    assert pol.min_actions_for_enforcement > 10_000


def test_missing_policy_file_falls_back_to_defaults():
    m = _load_sink_module()
    assert m._load_policy("/nonexistent/sink_policy.yaml") is m.DEFAULT_POLICY


def test_invalid_policy_key_fails_loudly(tmp_path):
    m = _load_sink_module()
    bad = tmp_path / "bad_policy.yaml"
    bad.write_text("version: x\nnot_a_policy_key: 1\n")
    with pytest.raises(ValueError):
        m._load_policy(str(bad))


def _hist(*pairs):
    return [{"action_type": a, "cnt": n} for a, n in pairs]


def _heads(**scores):
    return [
        {"head_index": i, "head_name": n, "score": s} for i, (n, s) in enumerate(scores.items())
    ]


def test_enforcement_note_gates_and_redacted_prose():
    m = _load_sink_module()
    gate = 25
    heads = _heads(FollowBot=0.91, LegitimateUser=0.1)
    short = _hist(("SERVER_PROFILE_FOLLOW", 10))
    long = _hist(("SERVER_PROFILE_FOLLOW", 40), ("SERVER_PROFILE_UNFOLLOW", 5))

    assert m.build_enforcement_note(heads, short, gate) is None
    assert m.build_enforcement_note(heads, None, gate) is None
    assert m.build_enforcement_note(heads, [], gate) is None
    assert m.build_enforcement_note(_heads(FollowBot=0.4, LegitimateUser=0.8), long, gate) is None

    note = m.build_enforcement_note(heads, long, gate)
    assert note == f"{m._REDACTED_TEMPLATE} [model: FollowBot=0.91]"


def test_enforcement_templates_are_the_redacted_sentinel():
    m = _load_sink_module()
    assert set(m._ENFORCEMENT_TEMPLATES) == {
        "FollowBot",
        "LikeBot",
        "EngagementAmplifier",
        "ReplySpamBot",
        "TweetSpamBot",
        "RTBot",
        "MultiActionBot",
    }
    for tmpl in m._ENFORCEMENT_TEMPLATES.values():
        assert tmpl == m._REDACTED_TEMPLATE
    assert not hasattr(m, "_ACTION_FRIENDLY")
