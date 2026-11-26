import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

import train_head


def _write_shard(path, user_ids, cls, strength):
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array(user_ids, pa.int64()),
                "cls": pa.array(cls, pa.list_(pa.float32(), 4)),
                "labels": pa.array([["LikeBot"]] * len(user_ids), pa.list_(pa.string())),
                "label_source": pa.array(["enforcement"] * len(user_ids), pa.string()),
                "label_strength": pa.array(strength, pa.string()),
            }
        ),
        path,
    )


@pytest.mark.parametrize("filtered", ["weak", "nonfinite", "empty"])
def test_load_cached_continues_after_a_shard_without_eligible_rows(tmp_path, filtered):
    if filtered == "weak":
        _write_shard(tmp_path / "a.parquet", [1], [[0.0] * 4], ["weak"])
    elif filtered == "nonfinite":
        _write_shard(tmp_path / "a.parquet", [1], [[np.nan, 0.0, 0.0, 0.0]], ["strong"])
    else:
        _write_shard(tmp_path / "a.parquet", [], [], [])
    _write_shard(tmp_path / "b.parquet", [2], [[1.0, 2.0, 3.0, 4.0]], ["strong"])

    out = train_head.load_cached(str(tmp_path))
    assert out["user_id"].tolist() == [2]
    assert out["source"].tolist() == ["enforcement"]
    assert out["x"].tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert out["y"].shape == (1, 8)
    assert out["y"][0, 1]


def test_load_cached_reports_a_split_without_eligible_rows(tmp_path):
    _write_shard(tmp_path / "a.parquet", [1], [[0.0] * 4], ["weak"])
    with pytest.raises(ValueError, match="no finite strong-label rows"):
        train_head.load_cached(str(tmp_path))
