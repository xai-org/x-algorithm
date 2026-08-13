# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.recsys_embedding import RecsysEmbeddings
from xrex.models.recsys_model import block_candidate_reduce, cast_jax

EPS = 1e-8


class TargetSpaceMode(str, Enum):
    MM_ONLY = "mm_only"
    POST_COMBINED = "post_combined"


@dataclass
class TargetEmbeddings:
    embeddings: jax.Array
    valid_mask: jax.Array
    post_ids: jax.Array | None
    author_ids: jax.Array | None
    mode: TargetSpaceMode


@dataclass
class CandidateSourceFeatures:
    post_ids: np.ndarray
    author_ids: np.ndarray
    mm_embeddings: np.ndarray


@dataclass
class CandidateTable:
    post_ids: np.ndarray
    author_ids: np.ndarray
    embeddings: np.ndarray
    mode: TargetSpaceMode


def get_target_space_mode(cfg) -> TargetSpaceMode:
    if cfg.candidate_embedding_mode == "post_combined":
        return TargetSpaceMode.POST_COMBINED
    return TargetSpaceMode.MM_ONLY


def l2_normalize(x: jax.Array) -> jax.Array:
    sq_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    return x / jnp.sqrt(sq_norm + EPS)


def project_mm_only_candidate_space(
    mm_embeddings: jax.Array,
    mm_to_candidate_space_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    projected = mm_to_candidate_space_fn(mm_embeddings)
    return l2_normalize(projected)


def _slice_flat_candidate_embedding_window(
    flat_candidate_embeddings: jax.Array,
    *,
    num_hashes: int,
    start: int,
    length: int,
) -> jax.Array:
    B, flat_seq_len, D = flat_candidate_embeddings.shape
    if flat_seq_len % num_hashes != 0:
        raise ValueError(
            f"Expected flat candidate embedding length divisible by num_hashes; got {flat_seq_len=} {num_hashes=}"
        )
    total_seq_len = flat_seq_len // num_hashes
    end = start + length
    if start < 0 or end > total_seq_len:
        raise ValueError(
            f"Requested candidate window [{start}, {end}) outside total_seq_len={total_seq_len}"
        )
    return flat_candidate_embeddings.reshape(B, total_seq_len, num_hashes, D)[
        :, start:end, :, :
    ].reshape(B, length * num_hashes, D)


def build_post_combined_candidate_space(
    *,
    post_hashes: jax.Array,
    author_hashes: jax.Array,
    candidate_post_embeddings: jax.Array,
    candidate_author_embeddings: jax.Array,
    candidate_product_surface_embeddings: jax.Array,
    hash_keys: Any,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    use_product_surface: bool,
    multimodal_embeddings: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    embs, padding_mask = block_candidate_reduce(
        post_hashes,
        author_hashes,
        candidate_post_embeddings,
        candidate_author_embeddings,
        candidate_product_surface_embeddings,
        hash_keys,
        lr_multiplier_func,
        embed_init_scale,
        use_product_surface,
        multimodal_embeddings=multimodal_embeddings,
    )
    return l2_normalize(embs), padding_mask


def _build_post_combined_targets(
    *,
    cfg,
    batch: RecsysFeaturesBatch,
    recsys_embeddings: RecsysEmbeddings,
    mm_embeddings: jax.Array,
    product_surface_embedding_fn: Callable[[jax.Array], jax.Array],
    seq_start: int,
    seq_length: int,
) -> tuple[jax.Array, jax.Array]:
    candidate_ps = batch["candidate_seq"].get("product_surface")
    assert recsys_embeddings.candidate_post_embeddings is not None
    B = recsys_embeddings.candidate_post_embeddings.shape[0]
    num_item_hashes = cfg.hash_table.hash_keys.num_item_hashes
    num_author_hashes = cfg.hash_table.hash_keys.num_author_hashes
    end = seq_start + seq_length

    candidate_post_embeddings = _slice_flat_candidate_embedding_window(
        recsys_embeddings.candidate_post_embeddings,
        num_hashes=num_item_hashes,
        start=seq_start,
        length=seq_length,
    )
    candidate_author_embeddings = _slice_flat_candidate_embedding_window(
        recsys_embeddings.candidate_author_embeddings,
        num_hashes=num_author_hashes,
        start=seq_start,
        length=seq_length,
    )
    candidate_ps_embeddings = (
        jnp.zeros(
            (B, seq_length, cfg.emb_table_width),
            dtype=recsys_embeddings.candidate_post_embeddings.dtype,
        )
        if candidate_ps is None
        else product_surface_embedding_fn(cast_jax(candidate_ps[:, seq_start:end]))
    )
    return build_post_combined_candidate_space(
        post_hashes=cast_jax(batch["candidate_seq"]["post_hashes"][:, seq_start:end, :]),
        author_hashes=cast_jax(batch["candidate_seq"]["auth_hashes"][:, seq_start:end, :]),
        candidate_post_embeddings=candidate_post_embeddings,
        candidate_author_embeddings=candidate_author_embeddings,
        candidate_product_surface_embeddings=candidate_ps_embeddings,
        hash_keys=cfg.hash_table.hash_keys,
        lr_multiplier_func=cfg.model_config.scale_config.emb_lr_multiplier,
        embed_init_scale=cfg.embed_init_scale,
        use_product_surface=cfg.use_product_surface,
        multimodal_embeddings=mm_embeddings,
    )


def build_target_embeddings_from_batch(
    *,
    cfg,
    batch: RecsysFeaturesBatch,
    recsys_embeddings: RecsysEmbeddings,
    mm_to_candidate_space_fn: Callable[[jax.Array], jax.Array],
    product_surface_embedding_fn: Callable[[jax.Array], jax.Array],
) -> TargetEmbeddings:
    mode = get_target_space_mode(cfg)
    mm_raw = batch["candidate_seq"].get("embedding")
    assert mm_raw is not None
    mm_all = cast_jax(mm_raw) if not isinstance(mm_raw, jax.Array) else mm_raw
    C = cfg.candidate_seq_len
    mm_targets = mm_all[:, :C, :]
    mm_valid = (jnp.sum(jnp.abs(mm_targets), axis=-1) > 0).astype(jnp.float32)
    post_ids = batch["candidate_seq"].get("post_ids")
    author_ids = batch["candidate_seq"].get("promoted_ids")

    if mode == TargetSpaceMode.MM_ONLY:
        candidate_space = project_mm_only_candidate_space(mm_targets, mm_to_candidate_space_fn)
        return TargetEmbeddings(
            embeddings=candidate_space,
            valid_mask=mm_valid,
            post_ids=None if post_ids is None else cast_jax(post_ids[:, :C]),
            author_ids=None if author_ids is None else cast_jax(author_ids[:, :C]),
            mode=mode,
        )

    embs, padding_mask = _build_post_combined_targets(
        cfg=cfg,
        batch=batch,
        recsys_embeddings=recsys_embeddings,
        mm_embeddings=mm_targets,
        product_surface_embedding_fn=product_surface_embedding_fn,
        seq_start=0,
        seq_length=C,
    )
    valid_mask = padding_mask.astype(jnp.float32) * mm_valid
    return TargetEmbeddings(
        embeddings=embs,
        valid_mask=valid_mask,
        post_ids=None if post_ids is None else cast_jax(post_ids[:, :C]),
        author_ids=None if author_ids is None else cast_jax(author_ids[:, :C]),
        mode=mode,
    )


def build_global_negative_target_embeddings(
    *,
    cfg,
    batch: RecsysFeaturesBatch,
    recsys_embeddings: RecsysEmbeddings,
    mm_global: jax.Array,
    gn_valid: jax.Array,
    mm_to_candidate_space_fn: Callable[[jax.Array], jax.Array],
    product_surface_embedding_fn: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    mode = get_target_space_mode(cfg)
    if mode == TargetSpaceMode.MM_ONLY:
        gn_embs = project_mm_only_candidate_space(mm_global, mm_to_candidate_space_fn)
        return gn_embs, gn_valid

    G = mm_global.shape[1]
    C = cfg.candidate_seq_len
    gn_embs, gn_padding_mask = _build_post_combined_targets(
        cfg=cfg,
        batch=batch,
        recsys_embeddings=recsys_embeddings,
        mm_embeddings=mm_global,
        product_surface_embedding_fn=product_surface_embedding_fn,
        seq_start=C,
        seq_length=G,
    )
    return gn_embs, gn_padding_mask.astype(jnp.float32) * gn_valid


def build_candidate_space_table_from_source_features(
    *,
    cfg,
    source_features: CandidateSourceFeatures,
    mm_only_project_fn: Callable[[jax.Array], jax.Array],
    post_combined_project_fn: Callable[[jax.Array, np.ndarray, np.ndarray], jax.Array]
    | None = None,
) -> CandidateTable:
    mode = get_target_space_mode(cfg)
    mm = jnp.asarray(source_features.mm_embeddings, dtype=jnp.float32)
    if mode == TargetSpaceMode.MM_ONLY:
        embs = np.asarray(project_mm_only_candidate_space(mm, mm_only_project_fn), dtype=np.float32)
    else:
        assert post_combined_project_fn is not None
        embs = np.asarray(
            post_combined_project_fn(mm, source_features.post_ids, source_features.author_ids),
            dtype=np.float32,
        )
    return CandidateTable(
        post_ids=source_features.post_ids,
        author_ids=source_features.author_ids,
        embeddings=embs,
        mode=mode,
    )


def score_against_candidate_table(
    predictions: jax.Array,
    candidate_table_embeddings: jax.Array,
) -> jax.Array:
    preds = l2_normalize(predictions.astype(jnp.float32))
    table = l2_normalize(candidate_table_embeddings.astype(jnp.float32))
    return jnp.dot(preds, table.T)


def prepare_eval_table(
    candidate_table: CandidateTable,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    table_embs = np.array(candidate_table.embeddings, dtype=np.float32)
    table_post_ids = np.array(candidate_table.post_ids, dtype=np.int64)
    table_norms = np.linalg.norm(table_embs, axis=-1, keepdims=True)
    valid_rows = table_norms.squeeze(-1) > 1e-8
    table_embs = table_embs[valid_rows]
    table_post_ids = table_post_ids[valid_rows]
    table_norm = table_embs / np.sqrt(np.sum(np.square(table_embs), axis=-1, keepdims=True) + 1e-8)
    pid_to_idx = {int(pid): idx for idx, pid in enumerate(table_post_ids)}
    return table_norm, table_post_ids, pid_to_idx


def compute_recall_for_batch(
    preds: np.ndarray,
    targets: np.ndarray,
    mm_valid: np.ndarray,
    cand_pids: np.ndarray,
    table_norm: np.ndarray,
    table_pid_to_idx: dict[int, int],
    table_jax: jax.Array,
    ks: list[int] = [1, 10, 100, 1000],
    k_top: int = 1000,
    num_track_pos: int = 10,
) -> dict[str, float | int]:
    _B, _C, _D = preds.shape

    pn = np.linalg.norm(preds, axis=-1, keepdims=True)
    preds_n = np.where(pn > 1e-8, preds / pn, 0.0)

    valid_mask = mm_valid > 0
    in_table = np.vectorize(lambda pid: int(pid) in table_pid_to_idx)(cand_pids)
    valid_mask = valid_mask & in_table

    result: dict[str, float | int] = {
        "gt_total": int(np.sum(mm_valid > 0)),
        "gt_found": int(np.sum(valid_mask)),
    }

    v_preds = preds_n[valid_mask]
    v_positions = np.where(valid_mask)
    N = len(v_preds)
    result["num_eval"] = N
    if N == 0:
        return result

    v_post_ids = cand_pids[valid_mask]
    e_gt = np.array([table_pid_to_idx[int(pid)] for pid in v_post_ids])

    targets_n = targets / np.sqrt(np.sum(np.square(targets), axis=-1, keepdims=True) + 1e-8)
    targets_flat = targets_n[valid_mask]
    table_lookup = np.array(table_norm[e_gt])
    result["target_table_cos_sum"] = float(np.sum(targets_flat * table_lookup))
    result["pred_target_cos_sum"] = float(np.sum(v_preds * targets_flat))
    result["pred_table_cos_sum"] = float(np.sum(v_preds * table_lookup))

    v_preds_jax = jnp.array(v_preds, dtype=jnp.bfloat16)
    scores_jax = jnp.dot(v_preds_jax, table_jax.T).astype(jnp.float32)
    _, top_k_idx = jax.lax.top_k(scores_jax, k_top)
    top_k_idx = np.array(jax.device_get(top_k_idx))

    for k in ks:
        in_top_k = np.any(top_k_idx[:, :k] == e_gt[:, None], axis=1)
        result[f"hits@{k}"] = int(np.sum(in_top_k))

    e_pos = v_positions[1]
    for c in range(num_track_pos):
        pos_mask = np.array([int(e_pos[i]) == c for i in range(N)])
        result[f"pos_{c}_total"] = int(np.sum(pos_mask))
        for k in ks:
            in_top_k = np.any(top_k_idx[:, :k] == e_gt[:, None], axis=1)
            result[f"pos_{c}_hits@{k}"] = int(np.sum(in_top_k & pos_mask))

    return result


def build_recall_metrics(
    agg: dict[str, float],
    ks: list[int] = [1, 10, 100, 1000],
    num_track_pos: int = 10,
) -> dict[str, float]:
    total_eval = agg.get("num_eval", 0)
    gt_total = agg.get("gt_total", 0)
    gt_found = agg.get("gt_found", 0)

    metrics: dict[str, float] = {}
    if total_eval > 0:
        for k in ks:
            metrics[f"gen-recs-global-recall@{k}"] = agg.get(f"hits@{k}", 0) / total_eval
    for c in range(num_track_pos):
        pt = agg.get(f"pos_{c}_total", 0)
        if pt > 0:
            for k in ks:
                metrics[f"gen-recs-global-recall@{k}/pos_{c}"] = (
                    agg.get(f"pos_{c}_hits@{k}", 0) / pt
                )
    if gt_total > 0:
        metrics["gen-recs-eval-in-table-rate"] = gt_found / gt_total
    if total_eval > 0:
        metrics["gen-recs-target-table-cosine"] = agg.get("target_table_cos_sum", 0) / total_eval
        metrics["gen-recs-pred-target-cosine"] = agg.get("pred_target_cos_sum", 0) / total_eval
        metrics["gen-recs-pred-table-cosine"] = agg.get("pred_table_cos_sum", 0) / total_eval
    metrics["gen-recs-eval-num-predictions"] = float(total_eval)
    return metrics


_PARQUET_EMB_CONFIG: dict[str, tuple[str, int]] = {
    "v1": ("mm_emb_v1", 1536),
    "v3": ("mm_emb_v3", 1024),
    "v5": ("mm_emb_v5", 1024),
    "v6": ("mm_emb_v6", 1024),
}


def load_mm_embedding_table(
    parquet_path: str,
    embedding_type: str = "v3",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import logging
    import os
    import time

    import pyarrow.parquet as pq

    logger = logging.getLogger(__name__)

    if embedding_type not in _PARQUET_EMB_CONFIG:
        raise ValueError(
            f"Unknown embedding_type={embedding_type!r}, expected one of {list(_PARQUET_EMB_CONFIG)}"
        )
    emb_col_name, expected_dim = _PARQUET_EMB_CONFIG[embedding_type]

    parquet_path = os.path.realpath(parquet_path)
    logger.info(f"Loading MM embedding table from {parquet_path} (embedding_type={embedding_type})")
    t0 = time.time()

    columns = ["post_id", "author_id", emb_col_name]
    table = pq.read_table(parquet_path, columns=columns)
    N = table.num_rows
    post_ids = table.column("post_id").to_numpy().astype(np.int64)
    author_ids = table.column("author_id").to_numpy().astype(np.int64)

    emb_col = table.column(emb_col_name)
    emb_parts = []
    idx_parts = []
    row_offset = 0
    for chunk in emb_col.chunks:
        n_rows = len(chunk)
        non_null = chunk.is_valid().to_numpy(zero_copy_only=False)
        if non_null.any():
            filtered = chunk.filter(chunk.is_valid())
            vals = filtered.values.to_numpy(zero_copy_only=False)
            n_valid = len(filtered)
            emb_parts.append(vals.reshape(n_valid, expected_dim).astype(np.float32))
            idx_parts.append(np.where(non_null)[0] + row_offset)
        row_offset += n_rows

    embeddings = np.concatenate(emb_parts)
    valid_indices = np.concatenate(idx_parts)
    post_ids = post_ids[valid_indices]
    author_ids = author_ids[valid_indices]

    norms = np.linalg.norm(embeddings, axis=-1)
    valid = norms > EPS
    post_ids = post_ids[valid]
    author_ids = author_ids[valid]
    embeddings = embeddings[valid]

    logger.info(
        f"Loaded {len(post_ids)} posts ({embeddings.shape[1]}D) from {N} rows in {time.time() - t0:.1f}s"
    )
    return post_ids, author_ids, embeddings
