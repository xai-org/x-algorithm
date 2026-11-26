import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
zstandard = pytest.importorskip("zstandard")

from manhattan_arrow_adapter import _extract_columns_from_arrow, arrow_bytes_to_batch


def _table(rows):
    action_type = pa.struct(
        [
            pa.field("action_name", pa.int32()),
            pa.field("tweet_id", pa.int64()),
            pa.field("author_id", pa.int64()),
            pa.field("engagement_time_ms", pa.int64()),
            pa.field("dwell_time", pa.int32()),
        ]
    )
    return pa.table({"user_actions": pa.array([rows], type=pa.list_(action_type))})


def _compressed(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return zstandard.ZstdCompressor().compress(sink.getvalue().to_pybytes())


def test_nullable_integer_columns_fill_nulls_without_rounding_present_ids():
    # Integer -> float -> integer conversion loses the low bit of this ID.
    large_id = 2**60 + 1
    cols = _extract_columns_from_arrow(
        _table(
            [
                {"action_name": 1, "tweet_id": large_id, "author_id": large_id + 2},
                {"action_name": None, "tweet_id": None, "author_id": None},
            ]
        )
    )
    assert cols["tweet_id"].tolist() == [large_id, 0]
    assert cols["author_id"].tolist() == [large_id + 2, 0]
    assert cols["action_name"].tolist() == [1, 0]
    assert cols["engagement_time_ms"].tolist() == [0, 0]


def test_arrow_batch_preserves_nullable_ids_and_uses_zero_for_missing_values():
    large_id = 2**60 + 1
    raw = _compressed(
        _table(
            [
                {
                    "action_name": 1,
                    "tweet_id": large_id,
                    "author_id": large_id + 2,
                    "engagement_time_ms": 60_000,
                    "dwell_time": 1000,
                },
                {"action_name": 4},
            ]
        )
    )
    batch, uids = arrow_bytes_to_batch([(42, raw)], seq_len=3)
    assert uids.tolist() == [42]
    assert batch["tweet_ids_raw"].tolist() == [[0, large_id, 0]]
    assert batch["author_ids_raw"].tolist() == [[0, large_id + 2, 0]]
    assert batch["action_seq"]["padding_mask"].tolist() == [[False, True, True]]
    assert batch["action_seq"]["action_types"][0, 2, 4]
    assert batch["summary_counters"][0, 6:8].tolist() == [1, 1]
    assert np.isfinite(batch["action_seq"]["continuous_actions"]).all()
