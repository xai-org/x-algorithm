# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xrex.data.recsys import history_id_lists as hil
from xrex.data.recsys import id_list_features as ilf
from xrex.data.recsys import id_list_presets as presets
from xrex.data.recsys.constants import action_type_map_raw

RECORDS = "/data/phoenix/kafka/ads/phoenix_training_records"
SIDECARS = "/data/phoenix/kafka/ads/list_features_backfill/list_features"


def _col(rb, name, n, dtype):
    values = pc.list_flatten(rb.column(name).combine_chunks())
    if values.null_count:
        values = values.fill_null(False if dtype is np.bool_ else 0)
    return values.to_numpy(zero_copy_only=False).reshape(n, -1).astype(dtype)


def history_counts(files: list[str], specs) -> np.ndarray:
    big = tuple(ilf.IdListFeatureSpec(s.name, 2000) for s in specs if s.is_history)
    bits = hil.ActionBits.for_vocab(action_type_map_raw, 256)
    out = []
    for f in files:
        t = pq.read_table(f)
        n = t.num_rows
        actions = (
            pc.list_flatten(pc.list_flatten(t.column("actionNameMultiHotSeqSeq").combine_chunks()))
            .to_numpy(zero_copy_only=False)
            .reshape(n, -1, 256)
        )
        pad = _col(t, "paddingMask", n, np.bool_)
        new = _col(t, "newEventMask", n, np.bool_)
        imp = _col(t, "impressedTimeMsSeq", n, np.int64)
        served = (
            _col(t, "servedTimeMsSeq", n, np.int64) if "servedTimeMsSeq" in t.schema.names else None
        )
        cont = (
            pc.list_flatten(
                pc.list_flatten(t.column("continuousActionValuesSeqSeq").combine_chunks())
            )
            .to_numpy(zero_copy_only=False)
            .reshape(n, -1, 8)
        )
        arrays = hil.HistoryArrays(
            actions=actions,
            author_ids=_col(t, "authorIdSeq", n, np.int64),
            tweet_ids=_col(t, "tweetIdSeq", n, np.int64),
            promoted_ids=_col(t, "promotedIdSeq", n, np.int64),
            impressed_ms=imp,
            request_ms=hil.request_time_ms(imp, served, pad & new, pad & ~new),
            history_mask=pad & ~new,
            dwell_secs=cont[:, :, hil.DWELL_TIME_SLOT].astype(np.float32),
        )
        ids, mask = ilf.empty_arrays(n, big)
        hil.fill(ids, mask, big, arrays, bits)
        offsets = ilf.slot_offsets(big)
        out.append(
            np.stack([mask[:, offsets[i] : offsets[i + 1]].sum(axis=1) for i in range(len(big))], 1)
        )
    return np.concatenate(out) if out else np.zeros((0, len(big)), np.int64)


def hydrator_counts(files: list[str], specs) -> np.ndarray:
    hyd = [s for s in specs if not s.is_history]
    out = []
    for f in files:
        col = pq.read_table(f, columns=[ilf.COLUMN]).column(0).combine_chunks()
        n = len(col)
        structs = pc.list_flatten(col)
        names = structs.field("name").to_numpy(zero_copy_only=False)
        lens = (
            pc.list_value_length(structs.field("ids")).fill_null(0).to_numpy(zero_copy_only=False)
        )
        row_of = np.repeat(
            np.arange(n), pc.list_value_length(col).fill_null(0).to_numpy(zero_copy_only=False)
        )
        counts = np.zeros((n, len(hyd)), np.int64)
        for j, s in enumerate(hyd):
            sel = names == s.name
            counts[row_of[sel], j] = lens[sel]
        out.append(counts)
    return np.concatenate(out) if out else np.zeros((0, len(hyd)), np.int64)


def propose_k(counts: np.ndarray, cap: int | None, max_k: int) -> int:
    nonempty = float((counts > 0).mean()) if counts.size else 0.0
    q = 95 if nonempty < 0.5 else 90
    k = int(np.percentile(counts, q)) if counts.size else 0
    k = max(5, int(5 * round(k / 5)))
    k = min(k, max_k)
    if cap is not None:
        k = min(k, cap)
    return k


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument(
        "--sidecars", default=SIDECARS, help="list-features sidecar root; '' to skip hydrator lists"
    )
    ap.add_argument("--min-batch", type=int, required=True)
    ap.add_argument("--max-batch", type=int, required=True)
    ap.add_argument("--batch-step", type=int, default=200)
    ap.add_argument("--partitions", default="0,64,128,192")
    ap.add_argument(
        "--specs", default=None, help="id_list_features string; default: id_list_presets.ALL_LISTS"
    )
    ap.add_argument(
        "--max-k",
        type=int,
        default=50,
        help="cap for proposed k (dense lists are capped by design)",
    )
    args = ap.parse_args()

    specs = (
        ilf.parse_specs(args.specs)
        if args.specs
        else ilf.parse_specs(presets.ADS_ID_LIST_FEATURES_ALL)
    )
    parts = [int(p) for p in args.partitions.split(",")]
    batches = range(args.min_batch, args.max_batch + 1, args.batch_step)
    rec_files = [
        f"{args.records}/partition={p}/{b // 2000}/batch_{b}.parquet"
        for b in batches
        for p in parts
    ]
    rec_files = [f for f in rec_files if os.path.exists(f)]
    random.Random(0).shuffle(rec_files)
    hist_specs = [s for s in specs if s.is_history]
    hyd_specs = [s for s in specs if not s.is_history]

    rows: dict[str, tuple[np.ndarray, int | None]] = {}
    if hist_specs:
        counts = history_counts(rec_files, specs)
        for j, s in enumerate(hist_specs):
            rows[s.feature_name] = (counts[:, j], None)
    if hyd_specs and args.sidecars:
        side_files = [
            f"{args.sidecars}/partition={p}/{b // 2000}/batch_{b}.list_features.parquet"
            for b in batches
            for p in parts
        ]
        side_files = [f for f in side_files if os.path.exists(f)]
        if not side_files:
            print(
                f"no sidecars in [{args.min_batch}, {args.max_batch}] under {args.sidecars}; hydrator lists skipped"
            )
        else:
            counts = hydrator_counts(side_files, specs)
            for j, s in enumerate(hyd_specs):
                rows[s.feature_name] = (counts[:, j], ilf.source_cap(s.name))

    print(
        f"records files {len(rec_files)} | users (records) {sum(len(v[0]) for n, v in rows.items() if n.startswith('hist_')) // max(len(hist_specs), 1)}"
    )
    print(
        f"{'feature':54s} {'k':>3s} {'>=1':>6s} {'full':>6s} {'fill':>5s} {'p50':>5s} {'p90':>5s} {'p99':>5s} {'propose':>7s}"
    )
    by_feature = {s.feature_name: s for s in specs}
    changes = {}
    for name, (c, cap) in rows.items():
        s = by_feature[name]
        if c.size == 0:
            continue
        k = s.k
        prop = propose_k(c, cap, args.max_k)
        print(
            f"{name:54s} {k:>3d} {(c > 0).mean() * 100:>5.1f}% {(c >= k).mean() * 100:>5.1f}% "
            f"{np.minimum(c, k).mean() / k * 100:>4.0f}% {np.percentile(c, 50):>5.0f} {np.percentile(c, 90):>5.0f} "
            f"{np.percentile(c, 99):>5.0f} {prop:>7d}{'  *' if prop != k else ''}"
        )
        if prop != k:
            changes[s.identity] = prop
    if changes:
        print("\nproposed with_k overrides (identity -> k):")
        for ident, k in changes.items():
            print(f'    "{ident}": {k},')


if __name__ == "__main__":
    main()
