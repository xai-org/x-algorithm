# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import glob
import json
import logging
import time
import typing

import numpy as np
import numpy.typing as npt
import tensorstore as ts
import xai_recsys_engine

from xrex.train.trainer import TrainerContext
from xrex.utils import ocdbt

AVX_512_ALIGNMENT = 64
NUM_INIT_THREADS = 2**7

logger = logging.getLogger(__name__)


def _create_aligned_array(shape: tuple[int, ...], itemsize: int, dtype):
    _buffer_size_bytes = functools.reduce(lambda x, y: x * y, shape) * itemsize
    buffer_size_bytes = _buffer_size_bytes + AVX_512_ALIGNMENT
    buffer = np.empty(buffer_size_bytes, dtype=np.uint8)
    if buffer_size_bytes >= 1 << 21:
        xai_recsys_engine.madvise_hugepage(buffer)
    else:
        logger.info(
            f"Buffer too small, skipping hugepages for embedding table allocation of size {buffer_size_bytes}"
        )
    xai_recsys_engine.init_parallel(buffer, num_threads=NUM_INIT_THREADS)
    return np.ndarray(
        shape,
        dtype=dtype,
        buffer=buffer[-buffer.__array_interface__["data"][0] & (AVX_512_ALIGNMENT - 1) :],
    )


def load_embedding_table(
    ctx: TrainerContext,
    emb_table: npt.NDArray[np.uint16],
    verify_checksums: bool = False,
):
    assert ctx.checkpoint is not None

    t0 = time.time()
    n_rows, emb_dim = emb_table.shape
    row_size_bytes = emb_dim * emb_table.itemsize
    tbl_size_bytes = n_rows * row_size_bytes
    logger.info(f"Loading embedding table: {tbl_size_bytes * 1e-9:.2f} GB")

    orbax_path = ctx.checkpoint.path + "/orbax-ckpt"
    shard_sources = sorted(
        ocdbt.load_shard_sources(orbax_path, prefix="emb_table/c/"),
        key=lambda x: tuple(int(y) for y in x[0].split("/")[2:]),
    )
    num_row_segments = int(shard_sources[-1][0].split("/")[2]) + 1
    row_segment_len = len(shard_sources) // num_row_segments

    for shard_idx, shard_source in enumerate(shard_sources):
        row_segment_idx = shard_idx // row_segment_len
        idx = shard_idx % row_segment_len
        if shard_source[0] != f"emb_table/c/{row_segment_idx}/{idx}":
            raise NotImplementedError(f"bad {shard_source=}")
    shard_sources = typing.cast(list[tuple[str, str, int, int]], shard_sources)

    tensor_buffer = np.ndarray(tbl_size_bytes, dtype=np.uint8, buffer=emb_table)
    xai_recsys_engine.load_tensor(
        orbax_path,
        "",
        shard_sources,
        tensor_buffer,
        row_size_bytes,
        num_row_segments,
    )
    if verify_checksums:
        checksum = xai_recsys_engine.adler32_parallel(tensor_buffer)
        with open(f"{ctx.checkpoint.path}/checksums.0.json") as f:
            saved = json.load(f)["global_checksums"]["emb_table"]
        if checksum != saved:
            raise ValueError("emb_table: checksum does not match")

    t1 = time.time()
    logger.info(
        f"Loading embedding table took {t1 - t0:.2f} sec; {tbl_size_bytes * 1e-9 / (t1 - t0):.2f} GB/s"
    )

    start_row = load_embedding_table_partial(orbax_path, 0, emb_table)
    _ = start_row


def load_embedding_table_partial(orbax_path: str, start_row: int, emb_table: npt.NDArray) -> int:
    spec = {
        "kvstore": {
            "driver": "ocdbt",
            "base": {"driver": "file", "path": f"{orbax_path}/ocdbt.process_0"},
        },
        "path": "inc_reverse_map",
        "driver": "zarr3",
        "open": True,
        "create": False,
    }
    try:
        store = ts.open(spec, read=True, write=False).result()
    except ValueError:
        return start_row

    inc_reverse_map = store[start_row:].read().result()
    if inc_reverse_map.size == 0:
        return start_row

    shard_sources = ocdbt.load_shard_sources(
        f"{orbax_path}/ocdbt.process_0", prefix="inc_emb_table/zarr.json"
    )
    if len(shard_sources) != 1 or len(shard_sources[0]) != 2:
        raise NotImplementedError(f"cannot find inc_emb_table in {orbax_path}")

    shape = inc_reverse_map.shape + emb_table.shape[1:]
    zarr_json = json.loads(shard_sources[0][1])
    itemsize = ts.dtype(zarr_json["data_type"]).numpy_dtype.itemsize
    if itemsize != emb_table.itemsize:
        raise NotImplementedError(
            f"bad data type size: wanted {emb_table.itemsize}, got {itemsize}"
        )

    chunk_shape = zarr_json["chunk_grid"]["configuration"]["chunk_shape"]
    a, b = zarr_json["shape"]
    if a < shape[0] or b != shape[1]:
        raise NotImplementedError(f"bad shape: wanted (>={shape[0]}, {shape[1]}), got ({a}, {b})")

    start_chunk = start_row // chunk_shape[0]
    stop_chunk = (start_row + shape[0] + chunk_shape[0] - 1) // chunk_shape[0]
    num_chunks = stop_chunk - start_chunk

    key = lambda x: tuple(int(y) for y in x[0].split("/")[2:])
    shard_sources = []
    for path in glob.glob(f"{orbax_path}/ocdbt.process_*"):
        shard_sources.extend(
            (n, f"{path[path.rindex('/') + 1 :]}/{p}", o, s)
            for n, p, o, s in sorted(
                typing.cast(
                    list[tuple[str, str, int, int]],
                    ocdbt.load_shard_sources(path, prefix="inc_emb_table/c/"),
                ),
                key=key,
            )[start_chunk:stop_chunk]
        )
    shard_sources.sort(key=key)
    num_processes = shape[1] // chunk_shape[1]
    if num_processes * num_chunks != len(shard_sources):
        raise NotImplementedError(
            f"bad number of shard sources: wanted {num_processes * num_chunks}, got {len(shard_sources)}"
        )

    t0 = time.time()
    size = chunk_shape[0] * shape[1] * emb_table.itemsize
    logger.info(f"Loading embedding table delta: {num_chunks * size * 1e-9:.2f} GB")
    for chunk_idx in range(num_chunks):
        inc_emb_table = _create_aligned_array(
            (chunk_shape[0], shape[1]), emb_table.itemsize, emb_table.dtype
        )
        buffer = np.ndarray(size, dtype=np.uint8, buffer=inc_emb_table)
        i0 = num_processes * chunk_idx
        i1 = i0 + num_processes
        xai_recsys_engine.load_tensor(
            orbax_path,
            "",
            shard_sources[i0:i1],
            buffer,
            shape[1] * emb_table.itemsize,
            1,
        )
        i0 = start_row - start_chunk * chunk_shape[0]
        i1 = chunk_shape[0] * chunk_idx
        j0 = i0 if chunk_idx == 0 else i1
        j1 = i0 + shape[0] if chunk_idx == num_chunks - 1 else i1 + chunk_shape[0]
        emb_table[inc_reverse_map[j0 - i0 : j1 - i0]] = inc_emb_table[j0 - i1 : j1 - i1]
    t1 = time.time()
    logger.info(
        f"Loading embedding table delta took {t1 - t0:.2f} sec; "
        f"{num_chunks * size * 1e-9 / (t1 - t0):.2f} GB/s"
    )

    return start_row + shape[0]
