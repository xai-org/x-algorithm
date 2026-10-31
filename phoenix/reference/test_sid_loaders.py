# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reference.dump_gen import _load_sid_codes
from reference.sid_index_server import _load_index


def _write_snapshot(tmp_path, rows, post_ids=(101, 102, 103), list_size=None):
    path = tmp_path / "sid.parquet"
    pq.write_table(
        pa.table(
            {
                "post_id": pa.array(post_ids, type=pa.int64()),
                "post_sid": pa.array(
                    rows, type=pa.list_(pa.int64(), list_size) if list_size else pa.list_(pa.int64())
                ),
            }
        ),
        path,
    )
    return str(path)


@pytest.mark.parametrize("loader", ["server", "dump"])
@pytest.mark.parametrize(
    "rows, error",
    [
        (
            [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24, 25, 26]],
            "fixed-width",
        ),
        ([None, [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], "null"),
        ([[0, 1, 2, 3, 4, 5], None, [20, 21, 22, 23, 24, 25]], "null"),
        (
            [[0, 1, 2, 3, 4, 5], [10, 11, None, 13, 14, 15], [20, 21, 22, 23, 24, 25]],
            "null",
        ),
    ],
    ids=["compensating-ragged-rows", "null-first-row", "null-later-row", "null-element"],
)
def test_loaders_reject_malformed_sid_rows(tmp_path, loader, rows, error):
    path = _write_snapshot(tmp_path, rows)
    if loader == "server":
        with pytest.raises(ValueError, match=error):
            _load_index(path)
    else:
        world = SimpleNamespace(num_posts=3, post_id=np.array([101, 102, 103]))
        with pytest.raises(SystemExit, match=error):
            _load_sid_codes(path, world)


@pytest.mark.parametrize("list_size", [None, 6], ids=["list", "fixed-size-list"])
def test_server_preserves_codes_when_sorting_valid_rows(tmp_path, list_size):
    path = _write_snapshot(
        tmp_path,
        [[20, 21, 22, 23, 24, 25], [-1, -1, -1, -1, -1, -1], [0, 1, 2, 3, 4, 5]],
        post_ids=[103, 102, 101],
        list_size=list_size,
    )
    post_ids, codes = _load_index(path)
    np.testing.assert_array_equal(post_ids, [101, 102, 103])
    np.testing.assert_array_equal(
        codes, [[0, 1, 2, 3, 4, 5], [-1, -1, -1, -1, -1, -1], [20, 21, 22, 23, 24, 25]]
    )
    assert post_ids.dtype == np.int64
    assert codes.dtype == np.int32


@pytest.mark.parametrize("list_size", [None, 6], ids=["list", "fixed-size-list"])
def test_dump_preserves_valid_codes_in_world_order(tmp_path, list_size):
    rows = [[20, 21, 22, 23, 24, 25], [-1, -1, -1, -1, -1, -1], [0, 1, 2, 3, 4, 5]]
    path = _write_snapshot(tmp_path, rows, post_ids=[103, 102, 101], list_size=list_size)
    world = SimpleNamespace(num_posts=3, post_id=np.array([103, 102, 101]))
    codes = _load_sid_codes(path, world)
    np.testing.assert_array_equal(
        codes, [[20, 21, 22, 23, 24, 25], [-1, -1, -1, -1, -1, -1], [0, 1, 2, 3, 4, 5]]
    )
    assert codes.dtype == np.int32


def test_dump_rejects_wrong_fixed_size_list_width(tmp_path):
    path = _write_snapshot(
        tmp_path,
        [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
        list_size=5,
    )
    world = SimpleNamespace(num_posts=3, post_id=np.array([101, 102, 103]))
    with pytest.raises(SystemExit, match="fixed-width 6"):
        _load_sid_codes(path, world)
