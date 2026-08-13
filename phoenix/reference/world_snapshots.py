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
    DEFAULT_NOW_MS,
    MM_DIM,
    TOPIC_NAMES,
    generate_world,
)

_SID_NUM_LEVELS = 6
_SID_CODEBOOK_SIZE = 256

_SID_TRAIN_CAP = 20_000
_SID_KMEANS_ITERS = 15

_MM_REL = "mm_snapshot/post_mm_v5.parquet"
_SID_REL = "sid_snapshot/post_sid_v5_256x6.parquet"
_CODEBOOK_REL = "sid_snapshot/codebook_v5_256x6.npz"
_POST_CREATION_REL = "post_creation_snapshots/post_creation_1day.parquet"
_RETRIEVAL_CORPUS_REL = "post_sid_v8_256x6_snapshots/1fav_1day.parquet"


class _WorldPost:
    __slots__ = ("id", "row")

    def __init__(self, post_id: int, row: int):
        self.id = int(post_id)
        self.row = int(row)


def _build_mm_matrix(world) -> np.ndarray:
    rows = np.arange(world.num_posts, dtype=np.int64)
    mm = np.ascontiguousarray(world.mm_vector(rows), dtype=np.float32)
    if mm.shape != (world.num_posts, MM_DIM):
        raise AssertionError(f"unexpected MM matrix shape {mm.shape}")
    return mm


def _write_mm(world, mm: np.ndarray, out: Path) -> int:
    from mm_snapshot_gen import write_mm_snapshot

    post_ids = world.post_id.astype(np.uint64)
    posts = [_WorldPost(pid, r) for r, pid in enumerate(post_ids)]
    return write_mm_snapshot(
        str(out / _MM_REL),
        post_ids,
        dim=MM_DIM,
        version="v5",
        embedder=lambda p: mm[p.row],
        posts=posts,
    )


def _train_and_assign_sids(
    mm: np.ndarray, seed: int, skip_train: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    n = mm.shape[0]
    if skip_train:
        print(
            "  [SID] --skip-sid-train: using oss_recsys_synth RANDOM codebook "
            "(⚠ structureless reference codes, NOT a quantization of the MM matrix)",
            flush=True,
        )
        from oss_recsys_synth import synth_sid_codebooks
        from sid_assign import encode_batch

        stages = synth_sid_codebooks(_SID_NUM_LEVELS, _SID_CODEBOOK_SIZE, MM_DIM, seed=seed).astype(
            np.float32
        )
        codes = encode_batch(mm, stages, None)
        return codes.astype(np.int64), None

    from sid_assign import encode_batch
    from sid_codebook import train_rq_kmeans

    if n > _SID_TRAIN_CAP:
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(7,)))
        train_idx = np.sort(rng.choice(n, size=_SID_TRAIN_CAP, replace=False))
        print(
            f"  [SID] training RQ-KMeans on a {_SID_TRAIN_CAP}/{n} deterministic "
            "subsample (codebook generalises to all posts via the assigner)",
            flush=True,
        )
        train_mm = np.ascontiguousarray(mm[train_idx])
    else:
        train_mm = mm

    codebooks = train_rq_kmeans(
        train_mm,
        num_levels=_SID_NUM_LEVELS,
        num_centroids=_SID_CODEBOOK_SIZE,
        n_iters=_SID_KMEANS_ITERS,
        seed=seed,
    )
    codes = encode_batch(mm, codebooks.astype(np.float32), None)
    return codes.astype(np.int64), codebooks.astype(np.float32)


def _write_sid(world, codes: np.ndarray, codebooks: np.ndarray | None, out: Path) -> int:
    from sid_snapshot_gen import write_sid_snapshot

    author_ids = world.author_id[world.author_row_of_post].astype(np.int64)
    n = write_sid_snapshot(
        str(out / _SID_REL),
        world.post_id.astype(np.int64),
        author_ids,
        num_levels=_SID_NUM_LEVELS,
        codebook_size=_SID_CODEBOOK_SIZE,
        codes=codes,
    )
    if codebooks is not None:
        from sid_io import save_codebook_npz

        save_codebook_npz(
            str(out / _CODEBOOK_REL),
            codebooks,
            n_iters=_SID_KMEANS_ITERS,
            seed=world.seed,
            n_train=min(world.num_posts, _SID_TRAIN_CAP),
        )
    return n


def _write_post_creation(world, codes: np.ndarray, out: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    post_ids = world.post_id.astype(np.int64)
    author_ids = world.author_id[world.author_row_of_post].astype(np.int64)
    created_at = world.post_created_ms.astype("datetime64[ms]")
    n = post_ids.size

    flat = pa.array(codes.reshape(-1), type=pa.int64())
    offsets = pa.array(np.arange(0, (n + 1) * _SID_NUM_LEVELS, _SID_NUM_LEVELS, dtype=np.int32))
    sid_col = pa.ListArray.from_arrays(offsets, flat)

    table = pa.table(
        {
            "post_id": pa.array(post_ids, type=pa.int64()),
            "author_id": pa.array(author_ids, type=pa.int64()),
            "created_at": pa.array(created_at, type=pa.timestamp("ms")),
            "post_sid": sid_col,
        }
    )
    dst = out / _POST_CREATION_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(dst))
    return n


def _write_retrieval_corpus(world, codes: np.ndarray, out: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    post_ids = world.post_id.astype(np.int64)
    author_ids = world.author_id[world.author_row_of_post].astype(np.int64)
    n = post_ids.size

    flat = pa.array(codes.reshape(-1), type=pa.int64())
    offsets = pa.array(np.arange(0, (n + 1) * _SID_NUM_LEVELS, _SID_NUM_LEVELS, dtype=np.int32))
    sid_col = pa.ListArray.from_arrays(offsets, flat)

    table = pa.table(
        {
            "post_id": pa.array(post_ids, type=pa.int64()),
            "author_id": pa.array(author_ids, type=pa.int64()),
            "post_sid": sid_col,
        }
    )
    dst = out / _RETRIEVAL_CORPUS_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(dst))
    return n


def _topic_coherence_demo(world, mm: np.ndarray, codes: np.ndarray, n_probe: int = 3) -> str:
    rng = np.random.default_rng(np.random.SeedSequence(entropy=world.seed, spawn_key=(1234,)))
    probes = rng.choice(world.num_posts, size=min(n_probe, world.num_posts), replace=False)
    lines = ["topic-coherence demo (MM nearest neighbours + SID codes):"]
    for q in probes:
        sims = mm @ mm[q]
        order = np.argsort(-sims)
        nbrs = [r for r in order if r != q][:3]
        qtopic = TOPIC_NAMES[int(world.post_topic[q])]
        lines.append(f"  probe post {int(world.post_id[q])} [{qtopic}] sid={codes[q].tolist()}")
        for r in nbrs:
            rtopic = TOPIC_NAMES[int(world.post_topic[r])]
            share = int((codes[q] == codes[r]).cumprod().sum())
            lines.append(
                f"      nn cos={sims[r]:.3f} [{rtopic}] sid={codes[r].tolist()} "
                f"shared_early_levels={share}"
            )
    return "\n".join(lines)


def _self_check(out: Path, world, codes: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    mm_schema = pq.read_schema(out / _MM_REL)
    assert mm_schema.names == ["postID", "version", "embedding"], mm_schema.names
    assert mm_schema.field(0).type == pa.uint64()
    assert pa.types.is_string(mm_schema.field(1).type)
    et = mm_schema.field(2).type
    assert pa.types.is_large_list(et) and et.value_type == pa.float32(), et
    mm_row0 = pq.read_table(out / _MM_REL).column("embedding")[0].as_py()
    assert len(mm_row0) == MM_DIM, len(mm_row0)

    sid_t = pq.read_table(out / _SID_REL)
    sid_col = sid_t.column("post_sid").combine_chunks()
    assert len(sid_col[0].as_py()) == _SID_NUM_LEVELS
    flat = sid_col.values.to_numpy(zero_copy_only=False)
    assert flat.min() >= 0 and flat.max() < _SID_CODEBOOK_SIZE, (flat.min(), flat.max())

    pc = pq.read_table(out / _POST_CREATION_REL)
    assert set(pc.column_names) >= {"post_id", "author_id", "created_at", "post_sid"}
    assert len(pc.column("post_sid").combine_chunks()[0].as_py()) == _SID_NUM_LEVELS
    print(
        f"  [self-check] PASS  mm=(postID:uint64,version:utf8,embedding:large_list<f32>[{MM_DIM}]) "
        f"sid=fixed-{_SID_NUM_LEVELS} in [0,{_SID_CODEBOOK_SIZE}) "
        f"post_creation cols={pc.column_names}",
        flush=True,
    )


def build_snapshots(
    out: Path,
    *,
    seed: int,
    num_posts: int,
    num_users: int,
    num_authors: int,
    num_topics: int,
    now_ms: int | None,
    skip_sid_train: bool,
    self_check: bool,
) -> None:
    print(
        f"generate_world(seed={seed}, num_users={num_users}, num_authors={num_authors}, "
        f"num_posts={num_posts}, num_topics={num_topics}, now_ms={now_ms})",
        flush=True,
    )
    world = generate_world(
        seed=seed,
        num_users=num_users,
        num_authors=num_authors,
        num_posts=num_posts,
        num_topics=num_topics,
        now_ms=now_ms,
    )
    out.mkdir(parents=True, exist_ok=True)

    mm = _build_mm_matrix(world)
    n_mm = _write_mm(world, mm, out)
    print(f"  [MM]  wrote {n_mm} rows -> {out / _MM_REL}", flush=True)

    codes, codebooks = _train_and_assign_sids(mm, seed, skip_sid_train)
    n_sid = _write_sid(world, codes, codebooks, out)
    print(f"  [SID] wrote {n_sid} rows -> {out / _SID_REL}", flush=True)

    n_pc = _write_post_creation(world, codes, out)
    print(f"  [PC]  wrote {n_pc} rows -> {out / _POST_CREATION_REL}", flush=True)

    n_rc = _write_retrieval_corpus(world, codes, out)
    print(f"  [RC]  wrote {n_rc} rows -> {out / _RETRIEVAL_CORPUS_REL}", flush=True)

    if self_check:
        _self_check(out, world, codes)
    print(_topic_coherence_demo(world, mm, codes), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output snapshot root (PHOENIX_INDEX_BASE)")
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--num-posts", type=int, default=50_000)
    ap.add_argument("--num-users", type=int, default=10_000)
    ap.add_argument("--num-authors", type=int, default=2_000)
    ap.add_argument("--num-topics", type=int, default=16)
    ap.add_argument("--now-ms", type=int, default=None, help=f"default {DEFAULT_NOW_MS}")
    ap.add_argument(
        "--skip-sid-train",
        action="store_true",
        help="use the oss_recsys_synth RANDOM codebook instead of training RQ-KMeans",
    )
    ap.add_argument("--self-check", action="store_true", help="assert reader schemas after write")
    args = ap.parse_args(argv)

    build_snapshots(
        Path(args.out),
        seed=args.seed,
        num_posts=args.num_posts,
        num_users=args.num_users,
        num_authors=args.num_authors,
        num_topics=args.num_topics,
        now_ms=args.now_ms,
        skip_sid_train=args.skip_sid_train,
        self_check=args.self_check,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
