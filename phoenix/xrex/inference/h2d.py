# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import concurrent.futures
import functools
import logging
import math
import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import xai_recsys_engine
from xai_checkpointing.common import _unsafe_jax2np

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

_debug_initialised_tables: set[int] = set()

AVX_512_ALIGNMENT = 64
NUM_INIT_THREADS = 2**7


class MultimodalEmbeddingBuffer(NamedTuple):
    jax_buffer: jax.Array
    embedding_dim: int
    cached_numpy_views: tuple[npt.NDArray[np.float16], ...]


class EmbeddingSlices(NamedTuple):
    hist_post_end: int
    hist_auth_end: int
    cand_post_end: int
    cand_auth_end: int
    user_end: int
    user_ip_end: int


class H2DState(NamedTuple):
    emb_table: npt.NDArray[np.uint16]

    embeddings_buffer: jax.Array

    embedding_slices: EmbeddingSlices

    multimodal_embeddings: MultimodalEmbeddingBuffer | None = None


def _create_aligned_array(shape: tuple[int, ...], itemsize: int, dtype):
    _buffer_size_bytes = functools.reduce(lambda x, y: x * y, shape) * itemsize
    buffer_size_bytes = _buffer_size_bytes + AVX_512_ALIGNMENT
    buffer = np.zeros(buffer_size_bytes, dtype=np.uint8)
    if buffer_size_bytes >= 1 << 21:
        try:
            xai_recsys_engine.madvise_hugepage(buffer)
        except OSError as e:
            rank_logger.warning(
                f"madvise_hugepage failed ({e}); continuing WITHOUT hugepages "
                f"for the {buffer_size_bytes}-byte embedding table buffer"
            )
    else:
        rank_logger.info(
            f"Buffer too small, skipping hugepages for embedding table allocation of size {buffer_size_bytes}"
        )
    xai_recsys_engine.init_parallel(buffer, num_threads=NUM_INIT_THREADS)
    return np.ndarray(
        shape,
        dtype=dtype,
        buffer=buffer[-buffer.__array_interface__["data"][0] & (AVX_512_ALIGNMENT - 1) :],
    )


def _create_mm_embedding_buffer_views(
    jax_array: jax.Array,
) -> tuple[npt.NDArray[np.float16], ...]:
    return tuple(
        np.ndarray(
            shape=(shard.data.size,),
            dtype=np.float16,
            buffer=_unsafe_jax2np(shard.data),
        )
        for shard in jax_array.addressable_shards
    )


def open_shared_emb_table(
    shape: tuple[int, int],
    dtype: type = np.uint16,
    path: str | None = None,
) -> np.memmap:
    if path is None:
        raise ValueError("No shared emb_table path provided")
    logger.info(f"Opening shared emb_table: path={path}, shape={shape}, dtype={dtype}")
    mm = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
    try:
        xai_recsys_engine.madvise_hugepage(np.ndarray(mm.nbytes, dtype=np.uint8, buffer=mm))
    except Exception as e:
        logger.warning(f"madvise_hugepage failed on shared emb_table: {e}")
    return mm


def create_memmap_emb_table(
    path: str,
    emb_table_shape: tuple[int, int],
    embedding_dtype,
    prefault: bool = True,
) -> np.memmap:
    match embedding_dtype.dtype.name:
        case "bfloat16":
            dtype = np.uint16
        case "float32":
            dtype = np.uint32
        case _:
            raise ValueError(f"{embedding_dtype} is not supported")
    mm = np.memmap(path, dtype=dtype, mode="w+", shape=emb_table_shape)
    try:
        buf = np.ndarray(mm.nbytes, dtype=np.uint8, buffer=mm)
        xai_recsys_engine.madvise_hugepage(buf)
    except Exception as e:
        rank_logger.warning(f"madvise_hugepage failed on memmap emb_table: {e}")
    if prefault:
        prefault_memmap(mm)
    return mm


def prefault_memmap(mm: np.ndarray) -> float:
    t0 = time.time()
    xai_recsys_engine.init_parallel(
        np.ndarray(mm.nbytes, dtype=np.uint8, buffer=mm),
        num_threads=NUM_INIT_THREADS,
    )
    return time.time() - t0


_PARALLEL_COPY_CHUNK_BYTES = 1 << 30
_PARALLEL_COPY_THREADS = 16


def parallel_copyto(dst: np.ndarray, src: np.ndarray) -> None:
    if dst.shape != src.shape:
        raise ValueError(f"shape mismatch: dst={dst.shape} src={src.shape}")
    if dst.ndim < 1 or dst.shape[0] <= 1 or dst.nbytes <= _PARALLEL_COPY_CHUNK_BYTES:
        np.copyto(dst, src)
        return
    row_bytes = dst.nbytes // dst.shape[0]
    rows_per_chunk = max(1, _PARALLEL_COPY_CHUNK_BYTES // max(row_bytes, 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL_COPY_THREADS) as pool:
        futures = [
            pool.submit(
                np.copyto, dst[start : start + rows_per_chunk], src[start : start + rows_per_chunk]
            )
            for start in range(0, dst.shape[0], rows_per_chunk)
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()


def open_existing_memmap_emb_table(
    path: str,
    emb_table_shape: tuple[int, int],
    embedding_dtype,
) -> np.memmap:
    match embedding_dtype.dtype.name:
        case "bfloat16":
            dtype = np.uint16
        case "float32":
            dtype = np.uint32
        case _:
            raise ValueError(f"{embedding_dtype} is not supported")
    mm = np.memmap(path, dtype=dtype, mode="r+", shape=emb_table_shape)
    try:
        buf = np.ndarray(mm.nbytes, dtype=np.uint8, buffer=mm)
        xai_recsys_engine.madvise_hugepage(buf)
    except Exception as e:
        rank_logger.warning(f"madvise_hugepage failed on shared emb_table memmap: {e}")
    return mm


def create_h2d_state(
    embedding_dtype,
    emb_table_shape: tuple[int, int],
    user_embeddings_shape: tuple[int, int],
    history_post_embeddings_shape: tuple[int, int],
    history_author_embeddings_shape: tuple[int, int],
    candidate_post_embeddings_shape: tuple[int, int],
    candidate_author_embeddings_shape: tuple[int, int],
    data_sharding,
    candidate_multimodal_embeddings_shape: tuple[int, int, int] | None = None,
    user_ip_embeddings_shape: tuple[int, int] | None = None,
    shared_emb_table_path: str | None = None,
    hotswap_emb_table_path: str | None = None,
    emb_table: npt.NDArray[np.uint16] | None = None,
) -> H2DState:
    match embedding_dtype.dtype.name:
        case "bfloat16":
            itemsize = 2
            dtype = np.uint16
        case "float32":
            itemsize = 4
            dtype = np.uint32
        case _:
            raise ValueError(f"{embedding_dtype} is not supported")

    hist_post_seq = history_post_embeddings_shape[1]
    hist_auth_seq = history_author_embeddings_shape[1]
    cand_post_seq = candidate_post_embeddings_shape[1]
    cand_auth_seq = candidate_author_embeddings_shape[1]
    user_seq = user_embeddings_shape[1]
    user_ip_seq = user_ip_embeddings_shape[1] if user_ip_embeddings_shape is not None else 0
    total_seq = (
        hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq + user_seq + user_ip_seq
    )

    _user_end = hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq + user_seq
    embedding_slices = EmbeddingSlices(
        hist_post_end=hist_post_seq,
        hist_auth_end=hist_post_seq + hist_auth_seq,
        cand_post_end=hist_post_seq + hist_auth_seq + cand_post_seq,
        cand_auth_end=hist_post_seq + hist_auth_seq + cand_post_seq + cand_auth_seq,
        user_end=_user_end,
        user_ip_end=total_seq,
    )

    batch_size = user_embeddings_shape[0]
    emb_dim = emb_table_shape[1]
    combined_shape = (batch_size, total_seq, emb_dim)

    logger.info(
        f"Initializing offloaded state: {AVX_512_ALIGNMENT=} {itemsize=} {emb_table_shape=} "
        f"combined_embeddings_shape={combined_shape} {embedding_slices=} "
        f"{candidate_multimodal_embeddings_shape=} {user_ip_embeddings_shape=}"
    )

    pinned_sharding = data_sharding.with_memory_kind("pinned_host")
    np_buf = np.zeros(combined_shape, dtype=dtype)
    embeddings_buffer = jax.device_put(np_buf, pinned_sharding)
    del np_buf
    logger.info(
        f"Created combined pinned embeddings buffer: shape={combined_shape}, "
        f"size={embeddings_buffer.nbytes / 1e6:.1f} MB, "
        f"num_shards={len(embeddings_buffer.addressable_shards)}"
    )

    multimodal_embeddings = None
    if candidate_multimodal_embeddings_shape is not None:
        mm_batch, candidate_seq_len, mm_embedding_dim = candidate_multimodal_embeddings_shape
        flat_shape = (mm_batch, candidate_seq_len * mm_embedding_dim)
        mm_np = np.zeros(flat_shape, dtype=np.float16)
        jax_buffer = jax.device_put(mm_np, pinned_sharding)
        del mm_np
        cached_numpy_views = _create_mm_embedding_buffer_views(jax_buffer)
        multimodal_embeddings = MultimodalEmbeddingBuffer(
            jax_buffer=jax_buffer,
            embedding_dim=mm_embedding_dim,
            cached_numpy_views=cached_numpy_views,
        )
        logger.info(
            f"Created pinned multimodal embeddings buffer: shape={flat_shape} "
            f"size={jax_buffer.nbytes / 1e6:.1f} MB, "
            f"num_shards={len(jax_buffer.addressable_shards)}"
        )

    if emb_table is None:
        if shared_emb_table_path is not None:
            emb_table = open_shared_emb_table(emb_table_shape, dtype, shared_emb_table_path)
        elif hotswap_emb_table_path is not None:
            t0 = time.time()
            emb_table = create_memmap_emb_table(
                hotswap_emb_table_path, emb_table_shape, embedding_dtype
            )
            logger.info(
                f"Allocated emb_table directly as hotswap memmap {hotswap_emb_table_path} "
                f"({emb_table.nbytes / (1 << 30):.2f} GiB, prefaulted in {time.time() - t0:.1f}s)"
            )
        else:
            emb_table = _create_aligned_array(emb_table_shape, itemsize, dtype)

    return H2DState(
        emb_table=emb_table,
        embeddings_buffer=embeddings_buffer,
        embedding_slices=embedding_slices,
        multimodal_embeddings=multimodal_embeddings,
    )


def lookup_h2d_embeddings(
    h2d_state: H2DState,
    batch: RecsysFeaturesBatch,
    data_sharding,
    batch_id: int | None = None,
    emb_table_override: npt.NDArray[np.uint16] | None = None,
    num_threads: int = 16,
) -> tuple[jax.Array, jax.Array | None]:
    emb_table = emb_table_override if emb_table_override is not None else h2d_state.emb_table
    num_shards = len(h2d_state.embeddings_buffer.addressable_shards)
    total_batch = h2d_state.embeddings_buffer.shape[0]

    hist_post_src = batch["history_seq"]["post_hashes"]
    hist_auth_src = batch["history_seq"]["auth_hashes"]
    cand_post_src = batch["candidate_seq"]["post_hashes"]
    cand_auth_src = batch["candidate_seq"]["auth_hashes"]
    user_src = batch["user_hashes"]

    segments = [
        hist_post_src.reshape(total_batch, -1),
        hist_auth_src.reshape(total_batch, -1),
        cand_post_src.reshape(total_batch, -1),
        cand_auth_src.reshape(total_batch, -1),
        user_src.reshape(total_batch, -1),
    ]
    if h2d_state.embedding_slices.user_ip_end > h2d_state.embedding_slices.user_end:
        user_ip_src = batch["user_ip_hashes"]
        segments.append(user_ip_src.reshape(total_batch, -1))
    combined_indices = np.concatenate(segments, axis=1).reshape(-1)

    if os.environ.get("DEBUG_ALLOW_RANDOM_INIT") == "1":
        table_id = id(emb_table)
        if table_id not in _debug_initialised_tables:
            _maybe_init_rows(
                emb_table,
                np.ndarray(combined_indices.size, dtype=np.uint32, buffer=combined_indices),
            )
            _debug_initialised_tables.add(table_id)

    row_size = emb_table.shape[1] * emb_table.itemsize

    shard_rows = tuple(
        np.ndarray(
            shape=shard.data.size * shard.data.itemsize,
            dtype=np.uint8,
            buffer=_unsafe_jax2np(shard.data),
        )
        for shard in h2d_state.embeddings_buffer.addressable_shards
    )

    logger.info(
        f"[batch={batch_id}] Step 2 (embedding_gather): "
        f"{len(combined_indices)} total indices, {num_shards} shards, row_size={row_size}B"
    )

    with jax.profiler.TraceAnnotation(
        "embedding_gather/combined",
        **({} if batch_id is None else {"batch_id": batch_id}),
    ):
        xai_recsys_engine.embedding_gather(
            table=np.ndarray(
                emb_table.size * emb_table.itemsize,
                dtype=np.uint8,
                buffer=emb_table,
            ),
            row_indexes=np.ndarray(combined_indices.size, dtype=np.uint32, buffer=combined_indices),
            rows=shard_rows,
            row_size=row_size,
            num_threads=num_threads,
        )

    merged_device = jax.device_put(h2d_state.embeddings_buffer, data_sharding)
    if h2d_state.embeddings_buffer.dtype == np.uint16:
        merged_device = merged_device.view(jax.numpy.bfloat16)

    candidate_multimodal_embeddings_gpu = None
    if h2d_state.multimodal_embeddings is not None:
        mm_buf = h2d_state.multimodal_embeddings
        mm_2d = jax.device_put(
            mm_buf.jax_buffer.reshape(total_batch, -1),
            data_sharding,
        )
        if mm_buf.jax_buffer.dtype == jnp.float16:
            pass
        elif mm_buf.jax_buffer.dtype == np.uint16:
            mm_2d = mm_2d.view(jax.numpy.float16)
        mm_cand_seq = mm_2d.shape[1] // mm_buf.embedding_dim
        candidate_multimodal_embeddings_gpu = mm_2d.reshape(
            total_batch, mm_cand_seq, mm_buf.embedding_dim
        )

    return merged_device, candidate_multimodal_embeddings_gpu


def _maybe_init_rows(
    embeddings: npt.NDArray,
    row_indexes: npt.NDArray[np.uint32],
) -> None:
    logger.info(
        "Lazy init emb_table rows: unique_rows=%d shape=%s dtype=%s",
        len(np.unique(row_indexes)),
        embeddings.shape,
        embeddings.dtype,
    )
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D emb_table, got shape={embeddings.shape}")
    row_size = embeddings.shape[1]
    scale = 1.0 / math.sqrt(row_size)
    unique_rows = np.unique(row_indexes)
    for row in unique_rows:
        idx = int(row)
        first = int(embeddings[idx, 0])
        if first != 0:
            continue
        rng = np.random.default_rng(idx)
        values = rng.uniform(-scale, scale, size=row_size).astype(np.float32)
        if embeddings.dtype == np.uint16:
            embeddings[idx] = (values.view(np.uint32) >> 16).astype(np.uint16)
        elif embeddings.dtype == np.uint32:
            embeddings[idx] = values.view(np.uint32)
        else:
            raise ValueError(f"Unsupported emb_table dtype: {embeddings.dtype}")
