#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import argparse
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "_sid_proto"))

import sid_lookup_pb2
import sid_lookup_pb2_grpc


def _load_index(parquet: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=["post_id", "post_sid"])
    pid = t.column("post_id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    sid_col = t.column("post_sid").combine_chunks()
    if sid_col.null_count:
        raise ValueError("post_sid contains null rows")
    if pa.types.is_fixed_size_list(sid_col.type):
        row_lengths = np.full(len(sid_col), sid_col.type.list_size)
    else:
        row_lengths = np.diff(sid_col.offsets.to_numpy(zero_copy_only=False))
    width = int(row_lengths[0])
    if (row_lengths != width).any():
        raise ValueError(
            f"post_sid is not fixed-width {width} "
            "(ragged list column not supported)"
        )
    if sid_col.values.null_count:
        raise ValueError("post_sid contains null elements")
    flat = sid_col.values.to_numpy(zero_copy_only=False).astype(np.int32)
    codes = flat.reshape(pid.size, width)
    order = np.argsort(pid, kind="stable")
    return pid[order], codes[order]


class SidIndexServicer(sid_lookup_pb2_grpc.SidLookupServiceServicer):
    def __init__(self, post_ids: np.ndarray, codes: np.ndarray):
        self.post_ids = post_ids
        self.codes = codes
        self.num_levels = codes.shape[1]
        self.valid = codes[:, 0] >= 0

    def LookupSids(self, request, context):
        post_ids = list(request.post_ids)
        if not post_ids:
            return sid_lookup_pb2.LookupSidsResponse(results=[])
        q = np.asarray(post_ids, dtype=np.int64)
        idx = np.searchsorted(self.post_ids, q)
        in_range = idx < self.post_ids.size
        idx_clip = np.where(in_range, idx, 0)
        hit = in_range & (self.post_ids[idx_clip] == q) & self.valid[idx_clip] & (q > 0)
        results = []
        for i in range(len(post_ids)):
            if hit[i]:
                results.append(
                    sid_lookup_pb2.PostSids(codes=[int(c) for c in self.codes[idx_clip[i]]])
                )
            else:
                results.append(sid_lookup_pb2.PostSids(codes=[]))
        return sid_lookup_pb2.LookupSidsResponse(results=results)


def serve(parquet: str, port: int, max_workers: int = 8) -> None:
    t0 = time.time()
    post_ids, codes = _load_index(parquet)
    load_s = time.time() - t0
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=(("grpc.so_reuseport", 0),),
    )
    sid_lookup_pb2_grpc.add_SidLookupServiceServicer_to_server(
        SidIndexServicer(post_ids, codes), server
    )
    try:
        bound = server.add_insecure_port(f"[::]:{port}")
    except RuntimeError:
        bound = 0
    if bound == 0:
        raise SystemExit(f"error: could not bind gRPC port {port} (already in use?)")
    server.start()
    n_valid = int((codes[:, 0] >= 0).sum())
    print(
        f"SID-INDEX-SERVER READY :{bound} posts={post_ids.size} valid_sid={n_valid} "
        f"levels={codes.shape[1]} load={load_s:.1f}s src={parquet}",
        flush=True,
    )
    server.wait_for_termination()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--parquet", required=True, help="post-SID snapshot parquet (post_id, post_sid)"
    )
    ap.add_argument("--port", type=int, default=50061)
    args = ap.parse_args()
    serve(args.parquet, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
