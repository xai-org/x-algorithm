#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sid_io import mlp_forward_np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


log = logging.getLogger(__name__)


def _encode_chunk(latent: np.ndarray, stages: np.ndarray) -> np.ndarray:
    M = stages.shape[0]
    N = len(latent)
    sids = np.zeros((N, M), dtype=np.int32)
    residual = latent.copy()
    for s in range(M):
        cb = stages[s]
        c_sq = (cb * cb).sum(axis=1)
        r_sq = (residual * residual).sum(axis=1, keepdims=True)
        dists = r_sq + c_sq[None, :] - 2.0 * residual @ cb.T
        sids[:, s] = dists.argmin(axis=1)
        residual -= cb[sids[:, s]]
    return sids


def encode_batch(
    embeddings: np.ndarray, stages: np.ndarray, rqvae_params, batch_size: int = 100_000
) -> np.ndarray:
    if rqvae_params is not None:
        enc_params, _ = rqvae_params
    else:
        enc_params = None

    N = len(embeddings)
    M = stages.shape[0]
    sids = np.zeros((N, M), dtype=np.int32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = embeddings[start:end].astype(np.float32)
        if enc_params is not None:
            chunk = mlp_forward_np(enc_params, chunk)
        sids[start:end] = _encode_chunk(chunk, stages)

    return sids


def process_shard(
    files: list[str],
    stages: np.ndarray,
    rqvae_params,
    output_path: str,
    num_levels: int = 6,
    batch_size: int = 100_000,
    limit: int | None = None,
) -> int:
    schema = pa.schema(
        [("post_id", pa.int64())] + [(f"sid_{i}", pa.uint8()) for i in range(num_levels)]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    total_rows = 0
    t0 = time.time()

    for fi, f in enumerate(files):
        try:
            table = pq.read_table(f)
            ids = table.column("post_id").to_numpy()
            col = table.column("embedding").combine_chunks()
            flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32)
            embs = flat.reshape(-1, flat.shape[0] // len(ids))
            del table, col, flat

            if limit is not None:
                remaining = limit - total_rows
                if remaining <= 0:
                    break
                if len(ids) > remaining:
                    ids = ids[:remaining]
                    embs = embs[:remaining]

            sids = encode_batch(embs, stages, rqvae_params, batch_size=batch_size)
            del embs

            columns = {"post_id": pa.array(ids, type=pa.int64())}
            for lv in range(num_levels):
                columns[f"sid_{lv}"] = pa.array(sids[:, lv].astype(np.uint8), type=pa.uint8())

            batch = pa.table(columns)
            writer.write_table(batch)
            total_rows += len(ids)

            elapsed = time.time() - t0
            rate = total_rows / elapsed if elapsed > 0 else 0
            log.info(
                "file %d/%d: %s rows, total %s, %.0f rows/s",
                fi + 1,
                len(files),
                f"{len(ids):,}",
                f"{total_rows:,}",
                rate,
            )
        except Exception as e:
            log.error("error processing %s: %s", f, e)

    writer.close()
    return total_rows
