#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oss_recsys_synth import synth_mm_embeddings

_MM_V5_DIM = 1024


def _embed_posts(embedder, posts, post_ids, dim):
    if embedder is None or posts is None:
        raise ValueError(
            "the real-v5 path needs BOTH embedder= and posts= "
            "(omit both to use the default synth path)"
        )
    if not callable(embedder):
        raise TypeError(f"embedder must be callable, got {type(embedder)!r}")

    posts = list(posts)
    n = len(posts)
    if n == 0:
        raise ValueError("posts is empty")

    if post_ids is None:
        try:
            post_ids = np.asarray([int(p.id) for p in posts], dtype=np.uint64)
        except (AttributeError, TypeError) as e:
            raise ValueError("posts have no usable .id; pass post_ids= aligned with posts") from e
    else:
        post_ids = np.asarray(post_ids, dtype=np.uint64)
        if post_ids.shape != (n,):
            raise ValueError(f"post_ids shape {post_ids.shape} != ({n},) (need one id per post)")

    from mm_encoder import truncate_l2

    rows = []
    for i, p in enumerate(posts):
        vec = np.asarray(embedder(p), dtype=np.float32).reshape(-1)
        unit = truncate_l2(vec, dim)
        if unit.shape != (dim,):
            raise ValueError(
                f"post #{i}: embedder returned {vec.shape[0]} dims; need >= {dim} to "
                f"truncate to [{dim}] (reduced to {unit.shape})"
            )
        rows.append(unit)
    emb = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
    return post_ids, emb


def write_mm_snapshot(
    path: str,
    post_ids: np.ndarray | Sequence[int] | None = None,
    *,
    dim: int = _MM_V5_DIM,
    version: str = "v5",
    seed: int = 0,
    embedder: Callable[[Any], np.ndarray] | None = None,
    posts: Sequence[Any] | None = None,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    if embedder is not None or posts is not None:
        post_ids, emb = _embed_posts(embedder, posts, post_ids, dim)
    else:
        post_ids = np.asarray(post_ids, dtype=np.uint64)
        if post_ids.ndim != 1:
            raise ValueError(f"post_ids must be 1-D, got shape {post_ids.shape}")
        n = post_ids.size
        emb = synth_mm_embeddings(post_ids, dim, seed=seed)
        if emb.shape != (n, dim):
            raise AssertionError(f"unexpected embedding shape {emb.shape} != {(n, dim)}")
        emb = np.ascontiguousarray(emb, dtype=np.float32)

    n = post_ids.size

    flat = pa.array(emb.reshape(-1), type=pa.float32())
    offsets = pa.array(np.arange(0, (n + 1) * dim, dim, dtype=np.int64))
    emb_col = pa.LargeListArray.from_arrays(offsets, flat)

    table = pa.table(
        {
            "postID": pa.array(post_ids, type=pa.uint64()),
            "version": pa.array([version] * n, type=pa.utf8()),
            "embedding": emb_col,
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

    dim, version, seed = 64, "v5", 0
    post_ids = np.arange(1000, 1000 + 32, dtype=np.uint64)

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ref_mm_snapshot.parquet")
        n = write_mm_snapshot(path, post_ids, dim=dim, version=version, seed=seed)
        assert n == post_ids.size, (n, post_ids.size)

        schema = pq.read_schema(path)
        assert schema.names == ["postID", "version", "embedding"], schema.names
        assert schema.field(0).type == pa.uint64(), schema.field(0).type
        assert pa.types.is_string(schema.field(1).type), schema.field(1).type
        emb_type = schema.field(2).type
        assert pa.types.is_large_list(emb_type), emb_type
        assert emb_type.value_type == pa.float32(), emb_type.value_type

        tbl = pq.read_table(path)
        got_ids = np.asarray(tbl.column("postID").to_pylist(), dtype=np.uint64)
        assert np.array_equal(got_ids, post_ids), "postID column did not round-trip"
        rows = tbl.column("embedding").to_pylist()
        assert all(len(r) == dim for r in rows), "embedding width != dim for some row"
        assert (np.asarray(tbl.column("version").to_pylist()) == version).all()
        emb = np.asarray(rows, dtype=np.float32)
        assert emb.shape == (post_ids.size, dim), emb.shape

        norms = np.linalg.norm(emb, axis=1)
        assert (norms > 0.5).all(), "an embedding collapsed toward zero"
        assert np.allclose(norms, 1.0, atol=1e-3), f"not unit-norm: {norms.min()}..{norms.max()}"

        emb_f16 = emb.astype(np.float16).astype(np.float32)
        assert (np.linalg.norm(emb_f16, axis=1) > 0.5).all(), "f16 round-trip zeroed a vector"

        path2 = str(Path(d) / "ref_mm_snapshot_2.parquet")
        write_mm_snapshot(path2, post_ids, dim=dim, version=version, seed=seed)
        emb2 = np.asarray(pq.read_table(path2).column("embedding").to_pylist(), dtype=np.float32)
        assert np.array_equal(emb, emb2), "snapshot generation not deterministic"

        uniq = {tuple(np.round(v, 5)) for v in emb}
        assert len(uniq) == post_ids.size, "MM vectors collapsed (posts not distinct)"

        sample = int(post_ids[0])
        print(
            "mm_snapshot_gen self-check PASS  "
            f"rows={post_ids.size} dim={dim} version={version!r} "
            f"unit_norm=[{norms.min():.4f},{norms.max():.4f}] "
            f"schema=(postID:uint64, version:utf8, embedding:{emb_type})\n"
            f"  sample post {sample} -> emb[:4]={emb[0, :4].tolist()}  "
            "(⚠ REFERENCE synth Fourier-feature vectors — NOT production v5)"
        )


def _self_check_embedder() -> None:
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    dim, version = _MM_V5_DIM, "v5"
    post_ids = np.arange(2000, 2000 + 16, dtype=np.uint64)

    def stub_embedder(post):
        rng = np.random.default_rng(int(post.id))
        return rng.standard_normal(4096).astype(np.float32) * 9.0

    class _StubPost:
        def __init__(self, pid):
            self.id = int(pid)

    posts = [_StubPost(p) for p in post_ids]

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ref_mm_real_v5.parquet")
        n = write_mm_snapshot(path, embedder=stub_embedder, posts=posts, dim=dim, version=version)
        assert n == post_ids.size, (n, post_ids.size)

        schema = pq.read_schema(path)
        assert schema.names == ["postID", "version", "embedding"], schema.names
        assert schema.field(0).type == pa.uint64(), schema.field(0).type
        assert pa.types.is_string(schema.field(1).type), schema.field(1).type
        emb_type = schema.field(2).type
        assert pa.types.is_large_list(emb_type) and emb_type.value_type == pa.float32(), emb_type

        tbl = pq.read_table(path)
        got_ids = np.asarray(tbl.column("postID").to_pylist(), dtype=np.uint64)
        assert np.array_equal(got_ids, post_ids), "derived postID column did not round-trip"
        rows = tbl.column("embedding").to_pylist()
        assert all(len(r) == dim for r in rows), "real-v5 embedding width != 1024"
        emb = np.asarray(rows, dtype=np.float32)

        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"real-v5 not unit-norm: {norms.min()}..{norms.max()}"
        )
        uniq = {tuple(np.round(v, 5)) for v in emb}
        assert len(uniq) == post_ids.size, "real-v5 vectors collapsed (posts not distinct)"

        path2 = str(Path(d) / "ref_mm_real_v5_2.parquet")
        write_mm_snapshot(
            path2, post_ids, embedder=stub_embedder, posts=posts, dim=dim, version=version
        )
        emb2 = np.asarray(pq.read_table(path2).column("embedding").to_pylist(), dtype=np.float32)
        assert np.array_equal(emb, emb2), "real-v5 path not deterministic / post_ids= disagrees"

        try:
            write_mm_snapshot(str(Path(d) / "x.parquet"), embedder=stub_embedder, posts=None)
            raise AssertionError("real path accepted embedder= without posts=")
        except ValueError:
            pass

    print(
        "mm_snapshot_gen real-v5 (stub embedder) self-check PASS  "
        f"rows={post_ids.size} dim={dim} version={version!r} "
        f"unit_norm=[{norms.min():.4f},{norms.max():.4f}] "
        "(4096 stub -> MRL truncate->L2 -> [1024]; SAME schema as synth)\n"
        "  (⚠ STUB embedder — real path uses mm_encoder.embed_post on STOCK Qwen weights)"
    )


if __name__ == "__main__":
    _self_check()
    _self_check_embedder()
