# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from sid_snapshot_gen import write_sid_snapshot


@pytest.mark.parametrize("readonly", [False, True])
def test_missing_mask_does_not_mutate_supplied_codes(tmp_path, readonly):
    codes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    before = codes.copy()
    if readonly:
        codes.flags.writeable = False
    path = tmp_path / "snapshot.parquet"

    assert write_sid_snapshot(
        str(path),
        [101, 102],
        [201, 202],
        num_levels=2,
        codes=codes,
        missing_mask=np.array([True, False]),
    ) == 2

    np.testing.assert_array_equal(codes, before)
    assert pq.read_table(path).column("post_sid").to_pylist() == [[-1, -1], [3, 4]]


def test_supplied_codes_can_be_reused_for_an_unmasked_snapshot(tmp_path):
    codes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    write_sid_snapshot(
        str(tmp_path / "masked.parquet"),
        [101, 102],
        [201, 202],
        num_levels=2,
        codes=codes,
        missing_mask=np.array([True, False]),
    )
    path = tmp_path / "unmasked.parquet"

    write_sid_snapshot(str(path), [101, 102], [201, 202], num_levels=2, codes=codes)

    assert pq.read_table(path).column("post_sid").to_pylist() == [[1, 2], [3, 4]]
