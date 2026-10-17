# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import gen_recs_artifacts_gen as artifacts
from world import DEFAULT_NOW_MS, POST_MAX_AGE_MS, POST_MIN_AGE_MS, TWITTER_EPOCH_MS


@pytest.fixture
def small_world(monkeypatch):
    generate_world = artifacts.generate_world

    def bounded_world(**kwargs):
        return generate_world(num_users=4, num_authors=6, num_posts=200, num_topics=4, **kwargs)

    monkeypatch.setattr(artifacts, "generate_world", bounded_world)


@pytest.mark.parametrize("now_ms", [None, DEFAULT_NOW_MS + 86_400_000])
def test_cli_self_check_uses_generation_clock(tmp_path, small_world, now_ms):
    argv = ["--out", str(tmp_path), "--self-check"]
    if now_ms is not None:
        argv.extend(["--now-ms", str(now_ms)])

    assert artifacts.main(argv) == 0

    post_ids = np.load(tmp_path / "gen_recs_artifacts" / "post_ids.npy")
    created_ms = (post_ids >> 22) + TWITTER_EPOCH_MS
    expected_now = DEFAULT_NOW_MS if now_ms is None else now_ms
    assert np.all(created_ms >= expected_now - POST_MAX_AGE_MS)
    assert np.all(created_ms <= expected_now - POST_MIN_AGE_MS)
    pool = pq.read_table(tmp_path / "gen_recs_artifacts" / "global_ids.parquet")
    np.testing.assert_array_equal(
        pool.column("created_at").to_numpy().astype("datetime64[ms]").astype(np.int64),
        (pool.column("post_id").to_numpy() >> 22) + TWITTER_EPOCH_MS,
    )
