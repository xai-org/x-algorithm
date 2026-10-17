#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from world import (
    MM_DIM,
    SURFACE_GALLERY,
    TWITTER_EPOCH_MS,
    World,
    generate_world,
)

_DIR_REL = "gen_recs_artifacts"
_POST_IDS_REL = f"{_DIR_REL}/post_ids.npy"
_EMBEDDINGS_REL = f"{_DIR_REL}/embeddings.npy"
_INFERENCE_POSTS_REL = f"{_DIR_REL}/inference_posts.parquet"
_GLOBAL_IDS_REL = f"{_DIR_REL}/global_ids.parquet"


def _mm_matrix(world: World) -> np.ndarray:
    rows = np.arange(world.num_posts, dtype=np.int64)
    mm = np.ascontiguousarray(world.mm_vector(rows), dtype=np.float32)
    assert mm.shape == (world.num_posts, MM_DIM), mm.shape
    return mm


def _write_npy_table(world: World, mm: np.ndarray, out: Path) -> int:
    post_ids = world.post_id.astype(np.int64)
    assert (np.diff(post_ids) > 0).all(), "world post ids must be strictly increasing"
    (out / _DIR_REL).mkdir(parents=True, exist_ok=True)
    np.save(out / _POST_IDS_REL, post_ids)
    np.save(out / _EMBEDDINGS_REL, mm)
    return post_ids.size


def _gallery_rows(world: World) -> np.ndarray:
    rows = np.flatnonzero(np.asarray(world.post_surface) == SURFACE_GALLERY)
    assert rows.size > 0, "the world has no gallery posts"
    return rows


def _write_inference_posts(world: World, mm: np.ndarray, rows: np.ndarray, out: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    flat = pa.array(mm[rows].reshape(-1), type=pa.float32())
    table = pa.table(
        {
            "post_id": pa.array(world.post_id[rows], type=pa.int64()),
            "author_id": pa.array(world.author_id[world.author_row_of_post[rows]], type=pa.int64()),
            "mm_emb_v5": pa.FixedSizeListArray.from_arrays(flat, MM_DIM),
        }
    )
    pq.write_table(table, str(out / _INFERENCE_POSTS_REL))
    return rows.size


def _write_global_ids(world: World, rows: np.ndarray, out: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    created_at = world.post_created_ms[rows].astype("datetime64[ms]")
    table = pa.table(
        {
            "post_id": pa.array(world.post_id[rows], type=pa.int64()),
            "author_id": pa.array(world.author_id[world.author_row_of_post[rows]], type=pa.int64()),
            "created_at": pa.array(created_at, type=pa.timestamp("ms")),
        }
    )
    pq.write_table(table, str(out / _GLOBAL_IDS_REL))
    return rows.size


def write_artifacts(out_dir: str, seed: int = 20260721, now_ms: int | None = None) -> dict:
    out = Path(out_dir)
    world = generate_world(seed=seed, now_ms=now_ms)
    mm = _mm_matrix(world)
    rows = _gallery_rows(world)
    n_all = _write_npy_table(world, mm, out)
    n_eval = _write_inference_posts(world, mm, rows, out)
    n_pool = _write_global_ids(world, rows, out)
    assert n_eval == n_pool
    return {"posts": n_all, "gallery": n_eval, "dim": MM_DIM, "seed": seed}


def _check_with_pyarrow(out: Path, seed: int, now_ms: int | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    world = generate_world(seed=seed, now_ms=now_ms)
    post_ids = np.load(out / _POST_IDS_REL)
    emb = np.load(out / _EMBEDDINGS_REL)

    assert post_ids.dtype == np.int64 and emb.dtype == np.float32
    assert emb.shape == (post_ids.size, MM_DIM), emb.shape
    assert (np.diff(post_ids) > 0).all(), "post_ids.npy must be sorted (searchsorted contract)"
    assert np.array_equal(post_ids, world.post_id), "npy table not row-aligned to the world"
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "MM vectors must be unit norm"
    probe = np.linspace(0, post_ids.size - 1, 16, dtype=np.int64)
    assert np.array_equal(emb[probe], world.mm_vector(probe)), "MM vectors not deterministic"

    inf = pq.read_table(out / _INFERENCE_POSTS_REL)
    assert inf.schema.field("post_id").type == pa.int64()
    assert inf.schema.field("author_id").type == pa.int64()
    emb_t = inf.schema.field("mm_emb_v5").type
    assert pa.types.is_fixed_size_list(emb_t) and emb_t.list_size == MM_DIM, emb_t
    inf_pids = inf.column("post_id").to_numpy(zero_copy_only=False)
    gallery = world.post_id[np.asarray(world.post_surface) == SURFACE_GALLERY]
    assert np.array_equal(inf_pids, gallery), "eval table must cover exactly the gallery posts"
    inf_emb = (
        inf.column("mm_emb_v5")
        .combine_chunks()
        .values.to_numpy(zero_copy_only=False)
        .reshape(-1, MM_DIM)
    )
    idx = np.searchsorted(post_ids, inf_pids)
    assert np.array_equal(inf_emb, emb[idx]), "eval vectors must equal the npy table's rows"

    pool = pq.read_table(out / _GLOBAL_IDS_REL)
    assert np.array_equal(pool.column("post_id").to_numpy(zero_copy_only=False), inf_pids)
    assert np.array_equal(
        pool.column("author_id").to_numpy(zero_copy_only=False),
        inf.column("author_id").to_numpy(zero_copy_only=False),
    )
    created = pool.column("created_at").to_numpy(zero_copy_only=False)
    created_ms = created.astype("datetime64[ms]").astype(np.int64)
    assert np.array_equal(created_ms, (inf_pids >> 22) + TWITTER_EPOCH_MS), (
        "created_at must equal the snowflake creation time"
    )

    print(
        "gen_recs_artifacts self-check PASS [1/2]  pyarrow invariants  "
        f"posts={post_ids.size} gallery={inf_pids.size} dim={MM_DIM} "
        f"gallery_frac={inf_pids.size / post_ids.size:.3f}"
    )


def _check_with_trainer(out: Path) -> None:
    cwd = Path.cwd()
    if (cwd / "xrex" / "__init__.py").exists() and str(cwd) not in sys.path:
        sys.path.append(str(cwd))
    try:
        from xrex.data.parquet_recsys import load_global_ids_from_parquet_file
        from xrex.data.recsys.gen_recs_semantics import load_mm_embedding_table
        from xrex.data.recsys.recsys_batch import PostEmbeddingTable
    except ImportError as e:
        print(f"gen_recs_artifacts self-check SKIP [2/2]  training package not importable ({e})")
        return

    table = PostEmbeddingTable(str(out / _DIR_REL))
    known = table.post_ids[[0, len(table.post_ids) // 2, -1]]
    vecs, found = table.lookup(np.concatenate([known, np.array([1], dtype=np.int64)]))
    assert found[:3].all() and not found[3], "PostEmbeddingTable hit/miss contract broken"
    assert np.allclose(np.linalg.norm(vecs[:3], axis=1), 1.0, atol=1e-3)
    assert vecs[3].sum() == 0.0, "a miss must return the zero vector"

    pids, aids, embs = load_mm_embedding_table(str(out / _INFERENCE_POSTS_REL), embedding_type="v5")
    assert pids.size == aids.size == embs.shape[0] and embs.shape[1] == MM_DIM
    tvecs, tfound = table.lookup(pids)
    assert tfound.all() and np.array_equal(tvecs, embs), (
        "eval table rows must resolve in the npy table"
    )

    g_pids, g_aids, g_created, _ = load_global_ids_from_parquet_file(
        out / _GLOBAL_IDS_REL, read_creation_datetime=True
    )
    assert g_pids is not None and g_aids is not None and g_created is not None
    assert g_pids.size == pids.size, "pool and eval table must cover the same posts"

    print(
        "gen_recs_artifacts self-check PASS [2/2]  loader round-trip  "
        f"table={len(table.post_ids)} eval={pids.size} pool={g_pids.size}"
    )


def run_self_check(out_dir: str, seed: int, now_ms: int | None = None) -> None:
    out = Path(out_dir)
    _check_with_pyarrow(out, seed, now_ms=now_ms)
    _check_with_trainer(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output root (use the quickstart's synth_index; adds gen_recs_artifacts/)",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--now-ms", type=int, default=None, help="world clock override (default: fixed constant)"
    )
    parser.add_argument("--self-check", action="store_true", help="validate the written artifacts")
    args = parser.parse_args(argv)

    summary = write_artifacts(args.out, seed=args.seed, now_ms=args.now_ms)
    print(
        f"wrote {summary['posts']} posts ({summary['gallery']} gallery) -> "
        f"{args.out}/{_DIR_REL} (dim={summary['dim']}, seed={summary['seed']})"
    )
    if args.self_check:
        run_self_check(args.out, args.seed, now_ms=args.now_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
