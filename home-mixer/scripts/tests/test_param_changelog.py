from __future__ import annotations

from param_changelog import NOTE_RE, Param, diff_params, parse_params, read_notes

SRC = """\
// mirrored from config feature-switch defaults; last sync 2026-10-16T16:00:32Z
use xai_feature_switches::param;

param!(FavoriteWeight, f64, "rust_home_mixer_favorite_weight", 0.5);
// A comment with a param!(Fake, f64, "fake", 1.0) inside it.
param!(
    ClusterUrl,
    String,
    "rust_home_mixer_cluster_url",
    "xds://root.prod:grpc"
);
param!(
    Rates,
    Vec<String>,
    "rust_home_mixer_rates",
    vec!["a:1.5".to_string(), "b".to_string()]
);
"""


def test_parses_single_and_multi_line():
    params = parse_params(SRC)
    assert set(params) == {"FavoriteWeight", "ClusterUrl", "Rates"}
    assert params["FavoriteWeight"] == Param(
        "FavoriteWeight", "f64", "rust_home_mixer_favorite_weight", "0.5"
    )
    assert params["ClusterUrl"].default == '"xds://root.prod:grpc"'
    assert params["Rates"].type == "Vec<String>"
    assert params["Rates"].default == 'vec!["a:1.5".to_string(), "b".to_string()]'


def test_reformatting_is_not_a_change():
    reformatted = SRC.replace(
        'param!(FavoriteWeight, f64, "rust_home_mixer_favorite_weight", 0.5);',
        'param!(\n    FavoriteWeight,\n    f64,\n    "rust_home_mixer_favorite_weight",\n'
        "    0.5\n);",
    )
    assert diff_params(parse_params(SRC), parse_params(reformatted)) == []


def test_diff_reports_changed_added_removed():
    old = parse_params(SRC)
    new = parse_params(SRC.replace("0.5);", "0.6);").replace("Rates", "NewRates"))
    changes = {c.name: (c.old, c.new) for c in diff_params(old, new)}
    assert changes["FavoriteWeight"][1].default == "0.6"
    assert changes["NewRates"][0] is None
    assert changes["Rates"][1] is None
    assert "ClusterUrl" not in changes


def test_notes_round_trip(tmp_path):
    doc = "## x\n<!-- note:abc1234 -->\nLikes count a bit more.\n<!-- /note -->\n"
    path = tmp_path / "PARAM_CHANGES.md"
    path.write_text(doc)
    assert read_notes(path) == {"abc1234": "Likes count a bit more.\n"}
    assert len(NOTE_RE.findall(doc)) == 1
