import heads
import numpy as np
import pytest
from metrics import average_precision


def test_average_precision_known_values():
    y = np.asarray([1, 0, 1, 0])
    s = np.asarray([0.9, 0.8, 0.7, 0.1])
    assert np.isclose(average_precision(y, s), (1 + 2 / 3) / 2)


def test_average_precision_no_positives_is_nan():
    assert np.isnan(average_precision(np.zeros(3), np.asarray([0.5, 0.4, 0.3])))


def test_average_precision_perfect_ranking():
    y = np.asarray([1, 1, 0, 0])
    s = np.asarray([0.9, 0.8, 0.2, 0.1])
    assert average_precision(y, s) == 1.0


def test_head_param_count_under_spec_cap():
    emb_dim = 1024
    n = emb_dim * heads.N_HEADS + heads.N_HEADS
    assert n == 8200
    assert n <= 5_000_000


def test_head_logits_shapes_and_cap():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from task_heads import head_logits, init_head_params

    x = jnp.zeros((5, 1024), jnp.float32)
    for hidden in ((), (512,), (512, 256)):
        params = init_head_params(1024, hidden)
        out = head_logits(params, x)
        assert out.shape == (5, heads.N_HEADS)
        n = sum(int(p.size) for p in params.values())
        assert n <= 5_000_000
    linear = init_head_params(1024, ())
    assert sum(int(p.size) for p in linear.values()) == 8200


def test_holdout_score_labels_roundtrip():
    y = np.zeros(heads.N_HEADS, dtype=bool)
    y[heads.HEAD_INDEX["LikeBot"]] = True
    y[heads.HEAD_INDEX["RTBot"]] = True
    names = [h.name for h in heads.HEADS if y[h.index]]
    assert names == ["LikeBot", "RTBot"]


def test_load_cached_drops_nonfinite_rows(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import train_head

    cls = np.zeros((3, 4), dtype=np.float32)
    cls[1, 2] = np.nan
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array([1, 2, 3], pa.int64()),
                "cls": pa.array(cls.tolist(), pa.list_(pa.float32(), 4)),
                "labels": pa.array([["LikeBot"], ["RTBot"], []], pa.list_(pa.string())),
                "label_source": pa.array(["enforcement"] * 3, pa.string()),
                "label_strength": pa.array(["strong"] * 3, pa.string()),
            }
        ),
        tmp_path / "cls_test.parquet",
    )

    out = train_head.load_cached(str(tmp_path))
    assert len(out["x"]) == 2
    assert out["user_id"].tolist() == [1, 3]
    assert np.isfinite(out["x"]).all()
