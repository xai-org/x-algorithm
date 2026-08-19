#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from world import (
    ACTION_NAMES,
    ACTION_ORDINALS,
    LABEL_WIDTH,
    MM_DIM,
    SURFACE_GALLERY,
    TOPIC_NAMES,
    TWITTER_EPOCH_MS,
    World,
    generate_world,
)

HIST_MAX = 1022
CAND_MIN, CAND_MAX = 8, 64
SEQ_LEN = HIST_MAX + CAND_MAX

NUM_CONTINUOUS = 8
DWELL_INDEX = 1

SAFETY_AUTHOR_NSFW_BIT = 2

SID_NUM_LEVELS = 6
SID_CODEBOOK_SIZE = 256
SID_MISSING = -1

LENGTH_BUCKETS = ((0.30, 26, 125), (0.50, 126, 509), (0.20, 510, HIST_MAX))

BATCHES_PER_SUBDIR = 2000

SERVE_JITTER_MS = (60_000, 1_800_000)
HIST_GAP_MS = 3_600_000


def _batch_relpath(partition: int, batch_id: int) -> str:
    return f"partition={partition}/{batch_id // BATCHES_PER_SUBDIR}/batch_{batch_id}.parquet"


def _load_sid_codes(path: str, world: World) -> np.ndarray:
    import pyarrow.parquet as pq

    if not Path(path).is_file():
        raise SystemExit(
            f"error: --sid {path} not found. Write the snapshot first, e.g.:\n"
            f"    python reference/world_snapshots.py --out synth_index --seed <SEED>"
        )
    t = pq.read_table(path, columns=["post_id", "post_sid"])
    pid = t.column("post_id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    if pid.size != world.num_posts:
        raise SystemExit(
            f"error: {path} covers {pid.size} posts but this world has "
            f"{world.num_posts}. Generate both from the same --seed."
        )
    if not np.array_equal(pid, world.post_id):
        raise SystemExit(
            f"error: {path} covers the right number of posts but different ids, so "
            "it was assigned against another world. Generate both from the same --seed."
        )
    flat = t.column("post_sid").combine_chunks().values.to_numpy(zero_copy_only=False)
    if flat.size != pid.size * SID_NUM_LEVELS:
        raise SystemExit(
            f"error: post_sid in {path} is not fixed-width {SID_NUM_LEVELS} "
            f"({flat.size} codes for {pid.size} posts)."
        )
    return flat.astype(np.int32).reshape(pid.size, SID_NUM_LEVELS)


def _pick_users(rng: np.random.Generator, world: World, n: int) -> np.ndarray:
    p = world.user_activity / world.user_activity.sum()
    users = rng.choice(world.num_users, size=n, p=p)
    for i in range(1, n):
        if users[i] == users[i - 1]:
            users[i] = (users[i - 1] + 1 + rng.integers(0, world.num_users - 1)) % world.num_users
    return users


def _history_lengths(rng: np.random.Generator, n: int) -> np.ndarray:
    probs = np.array([b[0] for b in LENGTH_BUCKETS])
    bucket = rng.choice(len(LENGTH_BUCKETS), size=n, p=probs / probs.sum())
    lo = np.array([b[1] for b in LENGTH_BUCKETS])[bucket]
    hi = np.array([b[2] for b in LENGTH_BUCKETS])[bucket]
    return (lo + rng.random(n) * (hi + 1 - lo)).astype(np.int64)


def _strictly_increasing(times: np.ndarray, cap_ms: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    t = times[order] + np.arange(len(times), dtype=np.int64)
    bound = cap_ms - (len(times) - 1 - np.arange(len(times), dtype=np.int64))
    return np.minimum(t, bound), order


def _build_rows(
    world: World,
    rng: np.random.Generator,
    users: np.ndarray,
    offset0: int,
    sid_codes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    n = len(users)
    cols = {
        "tweet": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "author": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "impressed": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "surface": np.zeros((n, SEQ_LEN), dtype=np.int32),
        "timezone": np.zeros((n, SEQ_LEN), dtype=np.int32),
        "client_app": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "ip": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "padding": np.zeros((n, SEQ_LEN), dtype=bool),
        "new_event": np.zeros((n, SEQ_LEN), dtype=bool),
        "safety": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "fav_count": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "reply_count": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "repost_count": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "quote_count": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "view_count": np.zeros((n, SEQ_LEN), dtype=np.int64),
        "actions": np.zeros((n, SEQ_LEN, LABEL_WIDTH), dtype=bool),
        "continuous": np.zeros((n, SEQ_LEN, NUM_CONTINUOUS), dtype=np.float32),
        "apps": np.zeros((n, 64), dtype=bool),
        "length": np.zeros(n, dtype=np.int64),
        "user_id": np.zeros(n, dtype=np.int64),
        "serve_ms": np.zeros(n, dtype=np.int64),
        "offset": offset0 + np.arange(n, dtype=np.int64),
        "cand_rows": np.zeros((n, CAND_MAX), dtype=np.int64),
        "n_cand": np.zeros(n, dtype=np.int64),
    }
    if sid_codes is not None:
        cols["sid"] = np.full((n, SEQ_LEN, SID_NUM_LEVELS), SID_MISSING, dtype=np.int32)
    hist_lens = _history_lengths(rng, n)
    cand_lens = rng.integers(CAND_MIN, CAND_MAX + 1, size=n)
    serve = world.now_ms - rng.integers(*SERVE_JITTER_MS, size=n)

    for i in range(n):
        u = int(users[i])
        hist_cap = int(serve[i]) - HIST_GAP_MS

        hist = world.sample_engaged_posts(
            rng, u, int(hist_lens[i]), max_created_ms=hist_cap - 600_000
        )
        times = world.engagement_times(rng, u, hist, cap_ms=hist_cap)
        times, order = _strictly_increasing(times, hist_cap)
        hist = hist[order]
        labels, dwell = world.sample_engagement(rng, u, hist, ensure_engaged=True)

        cand = world.sample_candidate_posts(rng, u, int(cand_lens[i]))
        clabels, cdwell = world.sample_engagement(rng, u, cand, ensure_engaged=False)

        length, n_cand = len(hist), len(cand)
        row_posts = np.concatenate([hist, cand])
        valid = slice(0, length + n_cand)
        cols["tweet"][i, valid] = world.post_id[row_posts]
        cols["author"][i, valid] = world.author_id[world.author_row_of_post[row_posts]]
        cols["impressed"][i, :length] = times
        cols["impressed"][i, length : length + n_cand] = serve[i]
        cols["surface"][i, valid] = world.post_surface[row_posts]
        cols["timezone"][i, valid] = world.user_timezone[u]
        cols["client_app"][i, valid] = world.user_client_app[u]
        cols["ip"][i, valid] = world.user_ip[u]
        cols["fav_count"][i, valid] = world.post_fav_count[row_posts]
        cols["reply_count"][i, valid] = world.post_reply_count[row_posts]
        cols["repost_count"][i, valid] = world.post_repost_count[row_posts]
        cols["quote_count"][i, valid] = world.post_quote_count[row_posts]
        cols["view_count"][i, valid] = world.post_view_count[row_posts]
        cols["safety"][i, valid] = (
            world.author_is_nsfw[world.author_row_of_post[row_posts]].astype(np.int64)
            << SAFETY_AUTHOR_NSFW_BIT
        )
        cols["padding"][i, valid] = True
        cols["new_event"][i, length : length + n_cand] = True
        cols["actions"][i, :length] = labels
        cols["actions"][i, length : length + n_cand] = clabels
        cols["continuous"][i, :length, DWELL_INDEX] = dwell
        cols["continuous"][i, length : length + n_cand, DWELL_INDEX] = cdwell
        cols["apps"][i] = world.user_apps[u]
        cols["length"][i] = length + n_cand
        cols["user_id"][i] = world.user_id[u]
        cols["serve_ms"][i] = serve[i]
        cols["cand_rows"][i, :n_cand] = cand
        cols["n_cand"][i] = n_cand
        if sid_codes is not None:
            cols["sid"][i, valid] = sid_codes[row_posts]
    cols["users"] = users
    return cols


def _write_batch_file(path: Path, world: World, cols: dict[str, np.ndarray], mm: str | None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def fsl(arr2d: np.ndarray, typ: pa.DataType) -> pa.Array:
        flat = pa.array(arr2d.reshape(-1), type=typ)
        return pa.FixedSizeListArray.from_arrays(flat, arr2d.shape[1])

    def fsl2(arr3d: np.ndarray, typ: pa.DataType) -> pa.Array:
        flat = pa.array(arr3d.reshape(-1), type=typ)
        inner = pa.FixedSizeListArray.from_arrays(flat, arr3d.shape[2])
        return pa.FixedSizeListArray.from_arrays(inner, arr3d.shape[1])

    n = len(cols["user_id"])
    users = cols["users"]
    zeros_i64 = np.zeros((n, SEQ_LEN), dtype=np.int64)
    named: dict[str, pa.Array] = {
        "length": pa.array(cols["length"], type=pa.int64()),
        "userId": pa.array(cols["user_id"], type=pa.int64()),
        "userGender": pa.array(world.user_gender[users], type=pa.uint16()),
        "userAgeBracket": pa.array(world.user_age_bracket[users], type=pa.uint16()),
        "userState": pa.array(world.user_state[users], type=pa.uint16()),
        "userLanguageCode": pa.array(world.user_language[users], type=pa.uint16()),
        "userCountryCode": pa.array(world.user_country[users], type=pa.uint16()),
        "userLongitude": pa.array(world.user_lon[users], type=pa.float32()),
        "userLatitude": pa.array(world.user_lat[users], type=pa.float32()),
        "userDmaCode": pa.array(world.user_dma[users], type=pa.uint16()),
        "userAgeInYears": pa.array(world.user_age_years[users], type=pa.uint8()),
        "userInferredGender": pa.array(world.user_gender[users].astype(np.uint8), type=pa.uint8()),
        "userInferredGenderScore": pa.array(np.full(n, 0.5, dtype=np.float32), type=pa.float32()),
        "installedAppsMultiHot": fsl(cols["apps"], pa.bool_()),
        "tweetIdSeq": fsl(cols["tweet"], pa.int64()),
        "authorIdSeq": fsl(cols["author"], pa.int64()),
        "promotedIdSeq": fsl(zeros_i64, pa.int64()),
        "impressedTimeMsSeq": fsl(cols["impressed"], pa.int64()),
        "clientAppIdSeq": fsl(cols["client_app"], pa.int64()),
        "ipAddressSeq": fsl(cols["ip"], pa.int64()),
        "timezoneSeq": fsl(cols["timezone"], pa.int32()),
        "productSurfaceSeq": fsl(cols["surface"], pa.int32()),
        "lineItemObjectiveSeq": fsl(zeros_i64.astype(np.int16), pa.int16()),
        "safetyLabelMaskSeq": fsl(cols["safety"], pa.int64()),
        "favCountSeq": fsl(cols["fav_count"], pa.int64()),
        "replyCountSeq": fsl(cols["reply_count"], pa.int64()),
        "repostCountSeq": fsl(cols["repost_count"], pa.int64()),
        "quoteCountSeq": fsl(cols["quote_count"], pa.int64()),
        "viewCountSeq": fsl(cols["view_count"], pa.int64()),
        "actionNameMultiHotSeqSeq": fsl2(cols["actions"], pa.bool_()),
        "continuousActionValuesSeqSeq": fsl2(cols["continuous"], pa.float32()),
        "paddingMask": fsl(cols["padding"], pa.bool_()),
        "newEventMask": fsl(cols["new_event"], pa.bool_()),
        "sampleWeight": pa.array(np.ones(n, dtype=np.float32), type=pa.float32()),
        "kafka_timestamp_ms": pa.array(cols["serve_ms"], type=pa.int64()),
        "kafka_offset": pa.array(cols["offset"], type=pa.int64()),
    }
    if mm is not None:
        emb = np.zeros((n, CAND_MAX, MM_DIM), dtype=np.float32)
        for i in range(n):
            nc = int(cols["n_cand"][i])
            emb[i, :nc] = world.mm_vector(cols["cand_rows"][i, :nc])
        named[f"embedding{mm.upper()}Seq"] = fsl2(emb, pa.float32())
    if "sid" in cols:
        named["semanticIdSeq"] = fsl2(cols["sid"], pa.int32())

    table = pa.table(named).replace_schema_metadata(
        {
            "min_kafka_timestamp_ms": str(int(cols["serve_ms"].min())),
            "max_kafka_timestamp_ms": str(int(cols["serve_ms"].max())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), compression="zstd", row_group_size=1024)


def _write_world_tables(world: World, out_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    world_dir = out_dir / "world"
    world_dir.mkdir(parents=True, exist_ok=True)
    author_rows = world.author_row_of_post
    posts = pa.table(
        {
            "post_id": pa.array(world.post_id, type=pa.int64()),
            "author_id": pa.array(world.author_id[author_rows], type=pa.int64()),
            "author_handle": pa.array([f"author_{r:05d}" for r in author_rows]),
            "topic": pa.array([TOPIC_NAMES[t] for t in world.post_topic]),
            "created_ms": pa.array(world.post_created_ms, type=pa.int64()),
            "quality": pa.array(world.post_quality.astype(np.float32), type=pa.float32()),
            "popularity": pa.array(world.post_popularity.astype(np.float32), type=pa.float32()),
            "product_surface": pa.array(world.post_surface, type=pa.int32()),
            "fav_count": pa.array(world.post_fav_count, type=pa.int64()),
            "reply_count": pa.array(world.post_reply_count, type=pa.int64()),
            "repost_count": pa.array(world.post_repost_count, type=pa.int64()),
            "quote_count": pa.array(world.post_quote_count, type=pa.int64()),
            "view_count": pa.array(world.post_view_count, type=pa.int64()),
            "text": pa.array(world.post_text),
        }
    )
    pq.write_table(posts, str(world_dir / "world_posts.parquet"), compression="zstd")

    users = pa.table(
        {
            "user_id": pa.array(world.user_id, type=pa.int64()),
            "handle": pa.array([f"user_{int(u):05d}" for u in world.user_id]),
            "topic_prefs": pa.array([world.user(i).prefs_str() for i in range(world.num_users)]),
            "activity": pa.array(world.user_activity.astype(np.float32), type=pa.float32()),
            "timezone": pa.array(world.user_timezone.astype(np.int32), type=pa.int32()),
            "client_app_id": pa.array(world.user_client_app, type=pa.int64()),
            "country": pa.array(world.user_country.astype(np.int32), type=pa.int32()),
            "language": pa.array(world.user_language.astype(np.int32), type=pa.int32()),
            "gender": pa.array(world.user_gender.astype(np.int32), type=pa.int32()),
            "age_bracket": pa.array(world.user_age_bracket.astype(np.int32), type=pa.int32()),
            "age_years": pa.array(world.user_age_years.astype(np.int32), type=pa.int32()),
            "dma": pa.array(world.user_dma.astype(np.int32), type=pa.int32()),
            "state": pa.array(world.user_state.astype(np.int32), type=pa.int32()),
            "lon": pa.array(world.user_lon, type=pa.float32()),
            "lat": pa.array(world.user_lat, type=pa.float32()),
            "ip_int": pa.array(world.user_ip, type=pa.int64()),
            "installed_apps": pa.array(
                [
                    ",".join(map(str, np.flatnonzero(world.user_apps[i])))
                    for i in range(world.num_users)
                ]
            ),
        }
    )
    pq.write_table(users, str(world_dir / "world_users.parquet"), compression="zstd")

    authors = pa.table(
        {
            "author_id": pa.array(world.author_id, type=pa.int64()),
            "handle": pa.array([f"author_{i:05d}" for i in range(world.num_authors)]),
            "primary_topic": pa.array([TOPIC_NAMES[t] for t in world.author_primary_topic]),
            "secondary_topic": pa.array([TOPIC_NAMES[t] for t in world.author_secondary_topic]),
            "quality": pa.array(world.author_quality.astype(np.float32), type=pa.float32()),
            "is_nsfw": pa.array(world.author_is_nsfw, type=pa.bool_()),
        }
    )
    pq.write_table(authors, str(world_dir / "world_authors.parquet"), compression="zstd")


def write_dump(
    out_dir: str,
    seed: int = 20260721,
    num_rows: int = 12_288,
    partitions: int = 4,
    rows_per_file: int = 1_024,
    mm: str | None = None,
    sid: str | None = None,
    now_ms: int | None = None,
) -> dict:
    if num_rows % (partitions * rows_per_file) != 0:
        raise ValueError(
            f"num_rows ({num_rows}) must be a multiple of partitions*rows_per_file "
            f"({partitions}*{rows_per_file}={partitions * rows_per_file})"
        )
    out = Path(out_dir)
    world = generate_world(seed=seed, now_ms=now_ms)
    _write_world_tables(world, out)
    sid_codes = _load_sid_codes(sid, world) if sid is not None else None

    num_batches = num_rows // (partitions * rows_per_file)
    for p in range(partitions):
        for b in range(num_batches):
            rng = np.random.default_rng(
                np.random.SeedSequence(entropy=seed, spawn_key=(1000, p, b))
            )
            users = _pick_users(rng, world, rows_per_file)
            cols = _build_rows(world, rng, users, offset0=b * rows_per_file, sid_codes=sid_codes)
            _write_batch_file(out / _batch_relpath(p, b), world, cols, mm)

    meta = {
        "min_valid_batch": 0,
        "max_valid_batch": num_batches - 1,
        "num_partitions": partitions,
    }
    with open(out / ".valid_batches.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return {
        "rows": num_rows,
        "partitions": partitions,
        "batches": num_batches,
        "seq_len": SEQ_LEN,
        "sid": sid_codes is not None,
    }


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%m-%d %H:%M")


def _fmt_actions(row: np.ndarray) -> str:
    names = [ACTION_NAMES.get(int(o), f"a{o}") for o in np.flatnonzero(row)]
    return ",".join(names) if names else "-"


def preview_sessions(out_dir: str, n: int) -> str:
    import pyarrow.parquet as pq

    out = Path(out_dir)
    posts = pq.read_table(out / "world" / "world_posts.parquet")
    post_text = dict(zip(posts.column("post_id").to_pylist(), posts.column("text").to_pylist()))
    users = pq.read_table(out / "world" / "world_users.parquet")
    user_info = {
        uid: (handle, prefs, tz)
        for uid, handle, prefs, tz in zip(
            users.column("user_id").to_pylist(),
            users.column("handle").to_pylist(),
            users.column("topic_prefs").to_pylist(),
            users.column("timezone").to_pylist(),
        )
    }

    table = pq.read_table(out / _batch_relpath(0, 0)).slice(0, n)
    lines = []
    for i in range(min(n, table.num_rows)):
        uid = table.column("userId")[i].as_py()
        tweet = np.array(table.column("tweetIdSeq")[i].as_py())
        impressed = np.array(table.column("impressedTimeMsSeq")[i].as_py())
        padding = np.array(table.column("paddingMask")[i].as_py())
        new_event = np.array(table.column("newEventMask")[i].as_py())
        actions = np.array(table.column("actionNameMultiHotSeqSeq")[i].as_py())
        dwell = np.array(table.column("continuousActionValuesSeqSeq")[i].as_py())[:, DWELL_INDEX]
        surface = np.array(table.column("productSurfaceSeq")[i].as_py())

        hist = np.flatnonzero(padding & ~new_event)
        cand = np.flatnonzero(padding & new_event)
        handle, prefs, tz = user_info[uid]
        lines.append(f"-- session row {i}: {handle}  prefs=[{prefs}]  tz={tz}")
        lines.append(f"   history ({len(hist)} engagements, last 4):")
        for j in hist[-4:]:
            lines.append(
                f"     {_fmt_ts(int(impressed[j]))}  {post_text[int(tweet[j])]}"
                f"  [{_fmt_actions(actions[j])} dwell={dwell[j]:.1f}s]"
            )
        lines.append(
            f"   candidates ({len(cand)} served at {_fmt_ts(int(impressed[cand[0]]))}, first 8):"
        )
        for j in cand[:8]:
            g = "G" if surface[j] == SURFACE_GALLERY else "H"
            label = _fmt_actions(actions[j])
            dw = f" dwell={dwell[j]:.1f}s" if dwell[j] > 0 else ""
            lines.append(f"     [{g}] {post_text[int(tweet[j])]}  -> {label}{dw}")
    return "\n".join(lines)


def _check_with_pyarrow(out: Path, mm: str | None, sid: bool = False) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with open(out / ".valid_batches.json") as f:
        meta = json.load(f)
    for p in range(meta["num_partitions"]):
        for b in range(meta["min_valid_batch"], meta["max_valid_batch"] + 1):
            assert (out / _batch_relpath(p, b)).exists(), f"missing batch file p={p} b={b}"

    pf = pq.ParquetFile(out / _batch_relpath(0, 0))
    kv = pf.metadata.metadata
    assert b"min_kafka_timestamp_ms" in kv and b"max_kafka_timestamp_ms" in kv, (
        "footer must carry kafka timestamp bounds"
    )
    schema = pf.schema_arrow
    for name, typ in {
        "userId": pa.int64(),
        "length": pa.int64(),
        "sampleWeight": pa.float32(),
        "kafka_timestamp_ms": pa.int64(),
    }.items():
        assert schema.field(name).type == typ, (name, schema.field(name).type)
    for name in ("tweetIdSeq", "authorIdSeq", "impressedTimeMsSeq", "promotedIdSeq"):
        t = schema.field(name).type
        assert pa.types.is_fixed_size_list(t) and t.list_size == SEQ_LEN, (name, t)
        assert t.value_type == pa.int64(), (name, t)
    for name in ("paddingMask", "newEventMask"):
        t = schema.field(name).type
        assert t == pa.list_(pa.bool_(), SEQ_LEN), (name, t)
    act_t = schema.field("actionNameMultiHotSeqSeq").type
    assert act_t.list_size == SEQ_LEN and act_t.value_type.list_size == LABEL_WIDTH, act_t
    cont_t = schema.field("continuousActionValuesSeqSeq").type
    assert cont_t.list_size == SEQ_LEN and cont_t.value_type.list_size == NUM_CONTINUOUS, cont_t
    apps_t = schema.field("installedAppsMultiHot").type
    assert apps_t == pa.list_(pa.bool_(), 64), apps_t
    if mm is not None:
        emb_t = schema.field(f"embedding{mm.upper()}Seq").type
        assert emb_t.list_size == CAND_MAX and emb_t.value_type.list_size == MM_DIM, emb_t
    if sid:
        sid_t = schema.field("semanticIdSeq").type
        assert sid_t.list_size == SEQ_LEN, sid_t
        assert sid_t.value_type.list_size == SID_NUM_LEVELS, sid_t
        assert sid_t.value_type.value_type == pa.int32(), sid_t

    table = pf.read()
    n = table.num_rows

    def col2d(name: str, dtype) -> np.ndarray:
        return np.array(
            table.column(name).combine_chunks().values.to_numpy(zero_copy_only=False), dtype=dtype
        ).reshape(n, -1)

    user_ids = np.array(table.column("userId").to_pylist(), dtype=np.int64)
    tweet = col2d("tweetIdSeq", np.int64)
    author = col2d("authorIdSeq", np.int64)
    impressed = col2d("impressedTimeMsSeq", np.int64)
    padding = col2d("paddingMask", bool)
    new_event = col2d("newEventMask", bool)
    actions = np.array(
        table.column("actionNameMultiHotSeqSeq")
        .combine_chunks()
        .flatten()
        .flatten()
        .to_numpy(zero_copy_only=False),
        dtype=bool,
    ).reshape(n, SEQ_LEN, LABEL_WIDTH)
    dwell = np.array(
        table.column("continuousActionValuesSeqSeq")
        .combine_chunks()
        .flatten()
        .flatten()
        .to_numpy(zero_copy_only=False),
        dtype=np.float32,
    ).reshape(n, SEQ_LEN, NUM_CONTINUOUS)[:, :, DWELL_INDEX]

    assert (user_ids[1:] != user_ids[:-1]).all(), "adjacent rows share a userId"
    for start in range(0, n - 63, 64):
        assert len(np.unique(user_ids[start : start + 64])) >= 2

    assert (user_ids >= 1).all()
    assert (tweet[padding] >= 1).all() and (author[padding] >= 1).all()
    assert (tweet[~padding] == 0).all() and (impressed[~padding] == 0).all()

    hist_mask = padding & ~new_event
    cand_mask = padding & new_event
    hist_counts = hist_mask.sum(axis=1)
    cand_counts = cand_mask.sum(axis=1)
    assert (hist_counts >= 26).all() and (hist_counts <= HIST_MAX).all()
    assert (cand_counts >= CAND_MIN).all() and (cand_counts <= CAND_MAX).all()

    if sid:
        sids = np.array(
            table.column("semanticIdSeq")
            .combine_chunks()
            .flatten()
            .flatten()
            .to_numpy(zero_copy_only=False),
            dtype=np.int32,
        ).reshape(n, SEQ_LEN, SID_NUM_LEVELS)
        assert (sids[padding] >= 0).all(), "valid position carries the -1 no-SID sentinel"
        assert (sids[padding] < SID_CODEBOOK_SIZE).all(), "SID code outside the codebook"
        assert (sids[~padding] == SID_MISSING).all(), "padding position carries a SID"

    fav_rate = actions[:, :, int(ACTION_ORDINALS[0])][cand_mask].mean()
    engaged_cands = actions[:, :, ACTION_ORDINALS][cand_mask].any(axis=-1)
    for i in range(n):
        h = np.flatnonzero(hist_mask[i])
        c = np.flatnonzero(cand_mask[i])
        assert h.size and c.size and h[-1] < c[0], "history must precede candidates"
        assert (np.diff(h) == 1).all() and (np.diff(c) == 1).all(), "valid block not contiguous"
        assert (np.diff(impressed[i, h]) > 0).all(), "history times must strictly increase"
        assert len(set(impressed[i, c].tolist())) == 1, "candidates must share one serve time"
        assert impressed[i, h[-1]] < impressed[i, c[0]], "serve must postdate history"
        created_ms = (tweet[i, padding[i]] >> 22) + TWITTER_EPOCH_MS
        assert (created_ms <= impressed[i, padding[i]]).all(), "impression before creation"
        assert (impressed[i, c[0]] - created_ms <= 40 * 3600 * 1000).all(), "stale snowflake"
        assert actions[i, h][:, ACTION_ORDINALS].any(axis=1).all(), "history rows must engage"
    assert 0.005 <= fav_rate <= 0.20, f"candidate fav rate {fav_rate:.4f} outside sane band"
    assert (dwell[~actions[:, :, ACTION_ORDINALS].any(axis=-1)] == 0).all(), (
        "dwell on non-engaged position"
    )
    assert dwell.max() <= 300.0

    fav_count = col2d("favCountSeq", np.int64)
    view_count = col2d("viewCountSeq", np.int64)
    safety = col2d("safetyLabelMaskSeq", np.int64)
    for name in ("favCountSeq", "replyCountSeq", "repostCountSeq", "quoteCountSeq"):
        counts = col2d(name, np.int64)
        assert (counts[~padding] == 0).all(), f"{name} nonzero on padding"
        assert (counts[padding] >= 0).all() and (counts[padding] <= view_count[padding]).all(), (
            f"{name} outside [0, viewCount]"
        )
    assert (view_count[~padding] == 0).all() and (fav_count[padding] >= 1).all()
    order_t = np.argsort(tweet[padding], kind="stable")
    same_post = tweet[padding][order_t][1:] == tweet[padding][order_t][:-1]
    for name, counts in (("favCountSeq", fav_count), ("viewCountSeq", view_count)):
        flat = counts[padding][order_t]
        assert (flat[1:][same_post] == flat[:-1][same_post]).all(), (
            f"{name} differs across appearances of the same post"
        )

    assert (safety[~padding] == 0).all(), "safety mask set on padding"
    assert (safety & ~(1 << SAFETY_AUTHOR_NSFW_BIT) == 0).all(), "unexpected safety bits"
    nsfw_frac = (safety[padding] != 0).mean()
    assert 0.0 < nsfw_frac < 0.35, f"author-NSFW rate {nsfw_frac:.4f} outside sane band"

    print(
        "dump_gen self-check PASS [1/2]  pyarrow invariants  "
        f"rows={n} seq={SEQ_LEN} hist=[{hist_counts.min()},{hist_counts.max()}] "
        f"cand=[{cand_counts.min()},{cand_counts.max()}] "
        f"fav_rate={fav_rate:.3f} engaged_cand_rate={engaged_cands.mean():.3f} "
        f"view_count_max={view_count.max()} nsfw_frac={nsfw_frac:.3f}"
        + (f" sid={SID_NUM_LEVELS}x{SID_CODEBOOK_SIZE}" if sid else "")
    )


def _check_with_trainer(out: Path, sid: bool = False) -> None:
    cwd = Path.cwd()
    if (cwd / "xrex" / "__init__.py").exists() and str(cwd) not in sys.path:
        sys.path.append(str(cwd))
    try:
        from xrex.data.recsys.feature_config import CategoricalFeature, Int64Feature
        from xrex.data.recsys.recsys_batch import from_record_batch
        from xrex.models.recsys_embedding import HashKeys, HashTable
    except ImportError as e:
        print(f"dump_gen self-check SKIP [2/2]  training package not importable ({e})")
        return
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(out / _batch_relpath(0, 0))
    rb = next(pf.iter_batches(batch_size=64))
    batch = from_record_batch(
        rb,
        max_history_post_action_pairs=HIST_MAX,
        max_candidate_post_action_pairs=CAND_MAX,
        num_negatives_per_example=1,
        output_vocab_size=LABEL_WIDTH,
        num_continuous_actions=NUM_CONTINUOUS,
        hash_table=HashTable(output_vocab_size=LABEL_WIDTH, hash_keys=HashKeys()),
        sid_num_levels=SID_NUM_LEVELS if sid else 0,
    )
    bsz = rb.num_rows
    hist, cand = batch["history_seq"], batch["candidate_seq"]
    assert batch["user_hashes"].shape == (bsz, 2), batch["user_hashes"].shape
    assert hist["post_hashes"].shape == (bsz, HIST_MAX, 2), hist["post_hashes"].shape
    assert hist["actions"].shape == (bsz, HIST_MAX, LABEL_WIDTH)
    assert hist["continuous_actions"].shape == (bsz, HIST_MAX, NUM_CONTINUOUS)
    assert cand["post_hashes"].shape == (bsz, 2 * CAND_MAX, 2), cand["post_hashes"].shape
    assert cand["actions"].shape == (bsz, 2 * CAND_MAX, LABEL_WIDTH)
    n_users = len(np.unique(rb.column("userId").to_numpy(zero_copy_only=False)))
    assert n_users >= 2, "negative sampling needs >=2 users per window"
    valid_hist = (hist["impr_ts"] > 0).sum(axis=1)
    valid_cand = (cand["impr_ts"][:, :CAND_MAX] > 0).sum(axis=1)
    assert (valid_hist >= 26).all() and (valid_hist <= HIST_MAX).all()
    assert (valid_cand >= CAND_MIN).all() and (valid_cand <= CAND_MAX).all()
    assert (hist["post_creation_ts_sec"] >= 0).all()

    hist_valid_mask = hist["impr_ts"] > 0
    hist_i64 = hist["int64_features"]
    hist_favs = hist_i64[..., Int64Feature.favCountSeq.value]
    hist_views = hist_i64[..., Int64Feature.viewCountSeq.value]
    assert (hist_favs[hist_valid_mask] >= 1).all(), "history position lost its fav count"
    assert (hist_views[hist_valid_mask] >= hist_favs[hist_valid_mask]).all()
    nsfw_cat = hist["categorical_features"][..., CategoricalFeature.authorIsNsfwSeq.value]
    assert set(np.unique(nsfw_cat).tolist()) <= {0, 1}, "authorIsNsfwSeq must be binary"
    nsfw_hist_frac = nsfw_cat[hist_valid_mask].mean()
    assert 0.0 < nsfw_hist_frac < 0.35, f"decoded NSFW rate {nsfw_hist_frac:.4f} outside band"

    sid_note = ""
    if sid:
        hsid = hist["post_sids"]
        assert hsid is not None, "sid_num_levels>0 but the batch carries no post_sids"
        assert hsid.shape == (bsz, HIST_MAX, SID_NUM_LEVELS), hsid.shape
        assert hsid.max() > 0, "history SIDs decoded as all-missing (semanticIdSeq lost?)"
        assert hsid[hist["impr_ts"] > 0].min() >= 1, "valid history position has no SID"
        assert hsid.max() <= SID_CODEBOOK_SIZE, hsid.max()
        sid_note = f" sid_levels={hsid.shape[2]} sid_max={hsid.max()}"
    print(
        "dump_gen self-check PASS [2/2]  from_record_batch round-trip  "
        f"batch={bsz} users={n_users} hist_valid=[{valid_hist.min()},{valid_hist.max()}] "
        f"cand_valid=[{valid_cand.min()},{valid_cand.max()}] "
        f"cand_axis={cand['post_hashes'].shape[1]} (=2x{CAND_MAX} with in-batch negatives) "
        f"fav_count_max={hist_favs.max()} nsfw_frac={nsfw_hist_frac:.3f}" + sid_note
    )


def run_self_check(out_dir: str, mm: str | None = None, sid: bool = False) -> None:
    out = Path(out_dir)
    _check_with_pyarrow(out, mm, sid)
    _check_with_trainer(out, sid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-rows", type=int, default=12_288, help="total rows across partitions")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--rows-per-file", type=int, default=1_024)
    parser.add_argument("--mm", choices=("v5",), default=None, help="also emit embeddingV5Seq")
    parser.add_argument(
        "--sid",
        default=None,
        metavar="PARQUET",
        help=(
            "also emit semanticIdSeq, reusing the codes in the post-SID snapshot "
            "written by world_snapshots.py (same --seed). Configs with "
            "use_post_sid=True train with all-missing SIDs without this."
        ),
    )
    parser.add_argument(
        "--now-ms", type=int, default=None, help="world clock override (default: fixed constant)"
    )
    parser.add_argument(
        "--preview", type=int, default=0, metavar="N", help="print N decoded sessions"
    )
    parser.add_argument("--self-check", action="store_true", help="validate the written dump")
    args = parser.parse_args(argv)

    summary = write_dump(
        args.out,
        seed=args.seed,
        num_rows=args.num_rows,
        partitions=args.partitions,
        rows_per_file=args.rows_per_file,
        mm=args.mm,
        sid=args.sid,
        now_ms=args.now_ms,
    )
    print(
        f"wrote {summary['rows']} rows -> {args.out} "
        f"(partitions={summary['partitions']}, batches/partition={summary['batches']}, "
        f"S={summary['seq_len']}, seed={args.seed}"
        + (f", semanticIdSeq={SID_NUM_LEVELS}x{SID_CODEBOOK_SIZE}" if summary["sid"] else "")
        + ")"
    )
    if args.preview > 0:
        print(preview_sessions(args.out, args.preview))
    if args.self_check:
        run_self_check(args.out, mm=args.mm, sid=args.sid is not None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
