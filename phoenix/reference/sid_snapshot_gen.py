#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oss_recsys_synth import sid_codes_for_posts

_MISSING_SENTINEL = -1


def write_sid_snapshot(
    path: str,
    post_ids: np.ndarray | Sequence[int],
    author_ids: np.ndarray | Sequence[int],
    *,
    num_levels: int = 6,
    codebook_size: int = 256,
    seed: int = 0,
    missing_mask: np.ndarray | None = None,
    codes: np.ndarray | None = None,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    post_ids = np.asarray(post_ids, dtype=np.int64)
    author_ids = np.asarray(author_ids, dtype=np.int64)
    if post_ids.ndim != 1:
        raise ValueError(f"post_ids must be 1-D, got shape {post_ids.shape}")
    if author_ids.shape != post_ids.shape:
        raise ValueError(f"author_ids shape {author_ids.shape} != post_ids shape {post_ids.shape}")
    n = post_ids.size

    if codes is None:
        codes = sid_codes_for_posts(post_ids, num_levels, codebook_size, seed=seed).astype(np.int64)
    else:
        codes = np.asarray(codes, dtype=np.int64)
        if codes.shape != (n, num_levels):
            raise ValueError(
                f"codes shape {codes.shape} != expected {(n, num_levels)} "
                "(must align to post_ids and num_levels)"
            )
        if (codes < 0).any() or (codes >= codebook_size).any():
            raise ValueError(
                f"codes out of [0, {codebook_size}): min={int(codes.min())} max={int(codes.max())}"
            )
    if codes.shape != (n, num_levels):
        raise AssertionError(f"unexpected codes shape {codes.shape} != {(n, num_levels)}")

    if missing_mask is not None:
        missing_mask = np.asarray(missing_mask, dtype=bool)
        if missing_mask.shape != post_ids.shape:
            raise ValueError(
                f"missing_mask shape {missing_mask.shape} != post_ids shape {post_ids.shape}"
            )
        codes[missing_mask] = _MISSING_SENTINEL

    flat = pa.array(codes.reshape(-1), type=pa.int64())
    offsets = pa.array(np.arange(0, (n + 1) * num_levels, num_levels, dtype=np.int32))
    sid_col = pa.ListArray.from_arrays(offsets, flat)

    table = pa.table(
        {
            "post_id": pa.array(post_ids, type=pa.int64()),
            "post_sid": sid_col,
            "author_id": pa.array(author_ids, type=pa.int64()),
        }
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out))
    return n


def _self_check() -> None:
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    import sid_index_server

    num_levels, codebook_size, seed = 6, 256, 0
    rng = np.random.default_rng(0)
    post_ids = rng.permutation(np.arange(1000, 1000 + 64, dtype=np.int64))
    author_ids = (post_ids % 7 + 5000).astype(np.int64)
    missing_mask = np.zeros(post_ids.size, dtype=bool)
    missing_mask[[3, 17]] = True
    missing_ids = set(int(p) for p in post_ids[missing_mask])

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ref_sid_snapshot.parquet")
        n = write_sid_snapshot(
            path,
            post_ids,
            author_ids,
            num_levels=num_levels,
            codebook_size=codebook_size,
            seed=seed,
            missing_mask=missing_mask,
        )
        assert n == post_ids.size, (n, post_ids.size)

        schema = pq.read_schema(path)
        assert schema.field("post_id").type == pa.int64(), schema
        assert schema.field("author_id").type == pa.int64(), schema
        sid_type = schema.field("post_sid").type
        assert pa.types.is_list(sid_type), sid_type
        assert sid_type.value_type == pa.int64(), sid_type

        pid_sorted, codes = sid_index_server._load_index(path)
        assert pid_sorted.size == post_ids.size, (pid_sorted.size, post_ids.size)
        assert codes.shape == (post_ids.size, num_levels), codes.shape
        assert (np.diff(pid_sorted) > 0).all(), "loader did not return sorted unique post_ids"

        valid = codes[:, 0] >= 0
        assert valid.sum() == post_ids.size - len(missing_ids), valid.sum()
        v = codes[valid]
        assert (v >= 0).all() and (v < codebook_size).all(), "codes out of [0, codebook)"

        miss_rows = {int(p): codes[i] for i, p in enumerate(pid_sorted) if not valid[i]}
        assert set(miss_rows) == missing_ids, (set(miss_rows), missing_ids)
        for row in miss_rows.values():
            assert int(row[0]) == _MISSING_SENTINEL, row

        by_id = {int(p): codes[i] for i, p in enumerate(pid_sorted) if valid[i]}
        check_ids = np.asarray(sorted(by_id), dtype=np.int64)
        ref = sid_codes_for_posts(check_ids, num_levels, codebook_size, seed=seed)
        for j, pid in enumerate(check_ids):
            got = by_id[int(pid)]
            assert np.array_equal(got, ref[j].astype(np.int64)), (
                f"post {int(pid)}: snapshot {got.tolist()} != reference {ref[j].tolist()}"
            )

        path2 = str(Path(d) / "ref_sid_snapshot_2.parquet")
        write_sid_snapshot(
            path2,
            post_ids,
            author_ids,
            num_levels=num_levels,
            codebook_size=codebook_size,
            seed=seed,
            missing_mask=missing_mask,
        )
        pid2, codes2 = sid_index_server._load_index(path2)
        assert np.array_equal(pid2, pid_sorted) and np.array_equal(codes2, codes), (
            "snapshot generation not deterministic"
        )

        distinct = {tuple(int(c) for c in by_id[k]) for k in by_id}
        assert len(distinct) > 1, "SIDs collapsed (all posts identical)"

        sample = int(check_ids[0])
        print(
            "sid_snapshot_gen self-check PASS  "
            f"rows={post_ids.size} valid={int(valid.sum())} missing={len(missing_ids)} "
            f"levels={num_levels} codebook={codebook_size} "
            f"schema=(post_id:int64, post_sid:{sid_type}, author_id:int64)\n"
            f"  sample post {sample} -> sid {by_id[sample].tolist()}  "
            "(⚠ REFERENCE synth codes — NOT production SIDs)"
        )


if __name__ == "__main__":
    _self_check()
