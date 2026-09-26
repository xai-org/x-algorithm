import dataclasses
import importlib.util
import logging
import os
from types import SimpleNamespace

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


def test_shipped_policy_warns_that_redacted_lanes_are_disabled(caplog):
    m = _load_sink_module()
    with caplog.at_level(logging.WARNING, logger="results_sink"):
        m._load_policy(os.path.join(_RUNTIME, "sink_policy.yaml"))

    assert "REDACTED 9.99 sentinel" in caplog.text
    assert "affected enforcement lanes are disabled" in caplog.text


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


def test_valid_custom_policy_documents_pair_semantics(tmp_path, caplog):
    m = _load_sink_module()
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        """\
version: test
min_actions_for_enforcement: 10
thresholds:
  FollowBot: [0.8, 0.2]
cusp_delta: 0.05
cusp_heads: {}
paused_liveness_thresholds: {}
spam_bounce_thresholds: {}
spam_bounce_action_key: {}
reply_spam_hard_suspend_tau: 0.95
"""
    )

    with caplog.at_level(logging.WARNING, logger="results_sink"):
        loaded = m._load_policy(str(policy))

    assert loaded.thresholds["FollowBot"] == (0.8, 0.2)
    assert "REDACTED 9.99 sentinel" not in caplog.text


@pytest.mark.parametrize(
    ("policy_fragment", "message"),
    [
        ("thresholds: []\n", "thresholds must be a mapping"),
        ("thresholds:\n  FollowBot: [0.8]\n", "exactly two values"),
        ("thresholds:\n  FollowBot: [nope, 0.2]\n", "must be a number"),
        ("thresholds:\n  FollowBot: [.nan, 0.2]\n", "must be finite"),
        ("thresholds:\n  FollowBot: [1.1, 0.2]\n", "must be in"),
        (
            "thresholds:\n  FollowBot: [9.99, 0.2]\n",
            "redaction sentinel for both values",
        ),
        ("thresholds:\n  TypoBot: [0.8, 0.2]\n", "not a model score head"),
    ],
)
def test_invalid_threshold_tables_fail_loudly(tmp_path, policy_fragment, message):
    m = _load_sink_module()
    bad = tmp_path / "bad_policy.yaml"
    bad.write_text("version: test\n" + policy_fragment)

    with pytest.raises(ValueError, match=message):
        m._load_policy(str(bad))


def test_redacted_cusp_delta_cannot_enable_a_configured_cusp_head(tmp_path):
    m = _load_sink_module()
    bad = tmp_path / "bad_policy.yaml"
    bad.write_text(
        """\
version: test
cusp_delta: 9.99
cusp_heads:
  EngagementAmplifier: [0.8, 0.2]
"""
    )

    with pytest.raises(ValueError, match="cusp_delta uses the redaction sentinel"):
        m._load_policy(str(bad))


def test_shipped_redacted_cusp_policy_cannot_match_scores():
    m = _load_sink_module()
    decision = m._Decision()

    m._cusp_lane(
        uid_int=1,
        dec=decision,
        user_labels=[],
        scores_by_name={"EngagementAmplifier": 0.5},
        legit=0.5,
        total_actions_gate=m.DEFAULT_POLICY.min_actions_for_enforcement,
        pol=m.DEFAULT_POLICY,
        args=None,
        r=None,
        metrics={},
    )

    assert not decision.cusp_met
    assert decision.cusp_head is None


def test_configured_cusp_policy_still_matches_scores():
    m = _load_sink_module()
    decision = m._Decision()
    labels = []
    policy = dataclasses.replace(
        m.DEFAULT_POLICY,
        cusp_delta=0.05,
        cusp_heads={"EngagementAmplifier": (0.5, 0.3)},
    )

    m._cusp_lane(
        uid_int=1,
        dec=decision,
        user_labels=labels,
        scores_by_name={"EngagementAmplifier": 0.5},
        legit=0.3,
        total_actions_gate=policy.min_actions_for_enforcement,
        pol=policy,
        args=SimpleNamespace(cusp_liveness_sample_per_10k=0),
        r=None,
        metrics={},
    )

    assert decision.cusp_met
    assert decision.cusp_head == "EngagementAmplifier"
    assert "EngagementAmplifier" in labels
