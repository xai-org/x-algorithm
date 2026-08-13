# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import jax
import numpy as np
from xai_configlib import Config, configclass

from xrex.data.recsys.recsys_batch import PostSeq, RecsysFeaturesBatch


@configclass
class LengthDistribution(Config, ABC):
    min_len: int
    max_len: int
    mean_len: int

    @abstractmethod
    def sample(
        self,
        bs_per_device: int,
        num_devices_per_process: int,
        rng: np.random.Generator,
        num_user_prefix_tokens: int,
        transformer_candidate_seq_len: int,
    ) -> np.ndarray: ...


@configclass
class BetaLengthDistribution(LengthDistribution):
    min_len: int
    max_len: int
    mean_len: int
    block_size: int
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self):
        assert self.min_len <= self.mean_len <= self.max_len
        assert self.alpha > 0
        assert self.beta > 0
        assert self.block_size > 0

    def sample(
        self,
        bs_per_device: int,
        num_devices_per_process: int,
        rng: np.random.Generator,
        num_user_prefix_tokens: int,
        transformer_candidate_seq_len: int,
    ) -> np.ndarray:
        samples = rng.beta(self.alpha, self.beta, size=bs_per_device * num_devices_per_process)
        lengths = self.min_len + np.floor((self.max_len - self.min_len + 1) * samples)
        lengths = np.clip(lengths, self.min_len, self.max_len).astype(np.int32)
        lengths = lengths.reshape(num_devices_per_process, bs_per_device)

        bs = self.block_size
        off = num_user_prefix_tokens + transformer_candidate_seq_len
        block_counts = (lengths + off + bs - 1) // bs
        min_blocks = (self.min_len + off + bs - 1) // bs
        max_blocks = (self.max_len + off + bs - 1) // bs
        total_tokens = bs_per_device * (self.mean_len + off)
        assert total_tokens % bs == 0, (
            f"bs_per_device * (mean_len + prefix + candidates) = "
            f"{bs_per_device} * ({self.mean_len} + {off}) = {total_tokens} "
            f"must be a multiple of block_size={bs}"
        )
        block_counts = fit_sample_budget(block_counts, total_tokens // bs, min_blocks, max_blocks)
        return (block_counts * bs - off).astype(np.int32, copy=False)


@configclass
class FixedLengthDistribution(LengthDistribution):
    min_len: int
    max_len: int
    mean_len: int

    def __post_init__(self):
        assert self.min_len == self.max_len == self.mean_len

    def sample(
        self,
        bs_per_device: int,
        num_devices_per_process: int,
        rng: np.random.Generator,
        num_user_prefix_tokens: int,
        transformer_candidate_seq_len: int,
    ) -> np.ndarray:
        del rng, num_user_prefix_tokens, transformer_candidate_seq_len
        return np.full((num_devices_per_process, bs_per_device), self.mean_len, dtype=np.int32)


def fit_sample_budget(lengths: np.ndarray, target_sum: int, lo: int, hi: int) -> np.ndarray:
    assert lengths.shape[1] * lo <= target_sum <= lengths.shape[1] * hi
    _, bs_per_device = lengths.shape

    lengths = np.ascontiguousarray(lengths)
    current_sum = lengths.sum(axis=1, dtype=np.int32)
    if np.all(current_sum == target_sum):
        return lengths

    delta = target_sum - current_sum

    pos_rows = np.flatnonzero(delta > 0)
    if pos_rows.size:
        order = np.argsort(lengths[pos_rows], axis=1, kind="stable")
        sorted_vals = np.take_along_axis(lengths[pos_rows], order, axis=1)
        capacities = hi - sorted_vals
        for row_idx, original_row in enumerate(pos_rows):
            remaining = int(delta[original_row])
            for col_idx in range(bs_per_device):
                capacity = int(capacities[row_idx, col_idx])
                if capacity <= 0:
                    continue
                increment = min(capacity, remaining)
                sorted_vals[row_idx, col_idx] += increment
                remaining -= increment
                if remaining == 0:
                    break
        updated = np.empty_like(sorted_vals)
        np.put_along_axis(updated, order, sorted_vals, axis=1)
        lengths[pos_rows] = updated

    neg_rows = np.flatnonzero(delta < 0)
    if neg_rows.size:
        order = np.argsort(lengths[neg_rows], axis=1, kind="stable")[:, ::-1]
        sorted_vals = np.take_along_axis(lengths[neg_rows], order, axis=1)
        capacities = sorted_vals - lo
        for row_idx, original_row in enumerate(neg_rows):
            remaining = int(-delta[original_row])
            for col_idx in range(bs_per_device):
                capacity = int(capacities[row_idx, col_idx])
                if capacity <= 0:
                    continue
                decrement = min(capacity, remaining)
                sorted_vals[row_idx, col_idx] -= decrement
                remaining -= decrement
                if remaining == 0:
                    break
        updated = np.empty_like(sorted_vals)
        np.put_along_axis(updated, order, sorted_vals, axis=1)
        lengths[neg_rows] = updated

    return lengths


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class SequencePackedLayout:
    cu_seqlens: np.ndarray
    segment_ids: np.ndarray
    history_positions: np.ndarray
    candidate_positions: np.ndarray
    padding_mask: np.ndarray
    positions: np.ndarray
    block_sparse: object | None = None


def pack_batch(
    batch: RecsysFeaturesBatch,
    num_devices_per_process: int,
    num_user_prefix_tokens: int,
    dist: LengthDistribution | None,
    rng: np.random.Generator | None,
    block_size: int = 128,
) -> RecsysFeaturesBatch:
    read_bsz_per_process = batch["user_hashes"].shape[0]
    history_seq_len = batch["history_seq"]["post_hashes"].shape[1]
    candidate_seq_len = batch["candidate_seq"]["post_hashes"].shape[1]
    transformer_candidate_seq_len = (
        0 if batch["candidate_seq"].get("post_ids") is not None else candidate_seq_len
    )
    bs_per_device = read_bsz_per_process // num_devices_per_process

    assert read_bsz_per_process % num_devices_per_process == 0
    assert batch["history_seq"]["post_hashes"].shape[0] == read_bsz_per_process
    assert batch["candidate_seq"]["post_hashes"].shape[0] == read_bsz_per_process

    def _reshape_sequence(seq, seq_len: int, packed_len: int) -> PostSeq:
        out = {}
        for k, v in seq.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == seq_len:
                out[k] = v.reshape(num_devices_per_process, packed_len, *v.shape[2:])
            else:
                out[k] = v
        return cast(PostSeq, out)

    def _gather_scatter_sequence(
        seq,
        seq_len: int,
        src_idx: np.ndarray,
        dst_idx: np.ndarray,
        packed_len: int,
    ) -> PostSeq:
        out = {}
        total_packed = num_devices_per_process * packed_len
        use_dense_gather = src_idx.shape == dst_idx.shape and np.array_equal(
            dst_idx,
            np.arange(total_packed, dtype=np.intp),
        )
        full_src: np.ndarray | None = None
        for k, v in seq.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == seq_len:
                flat = v.reshape(read_bsz_per_process * seq_len, *v.shape[2:])
                if use_dense_gather:
                    out[k] = flat[src_idx].reshape(
                        num_devices_per_process, packed_len, *v.shape[2:]
                    )
                else:
                    if full_src is None:
                        full_src = np.full(total_packed, flat.shape[0], dtype=np.intp)
                        full_src[dst_idx] = src_idx
                    flatz = np.concatenate([flat, np.zeros((1, *v.shape[2:]), dtype=v.dtype)])
                    out[k] = flatz[full_src].reshape(
                        num_devices_per_process, packed_len, *v.shape[2:]
                    )
            else:
                out[k] = v
        return cast(PostSeq, out)

    def _make_batch(
        packed_history: PostSeq, packed_candidates: PostSeq, layout: SequencePackedLayout
    ) -> RecsysFeaturesBatch:
        D, B = num_devices_per_process, bs_per_device
        return RecsysFeaturesBatch(
            user_hashes=batch["user_hashes"].reshape(D, B, -1),
            user_ip_hashes=batch["user_ip_hashes"].reshape(D, B, -1),
            history_seq=packed_history,
            candidate_seq=packed_candidates,
            user_categorical_features=batch["user_categorical_features"].reshape(
                D, B, *batch["user_categorical_features"].shape[1:]
            ),
            user_bool_features=batch["user_bool_features"].reshape(
                D, B, *batch["user_bool_features"].shape[1:]
            ),
            user_float_features=batch["user_float_features"].reshape(
                D, B, *batch["user_float_features"].shape[1:]
            ),
            user_int64_features=batch["user_int64_features"].reshape(
                D, B, *batch["user_int64_features"].shape[1:]
            ),
            user_installed_apps_multihot=batch["user_installed_apps_multihot"].reshape(
                D, B, *batch["user_installed_apps_multihot"].shape[1:]
            ),
            num_positive_candidates=(
                batch["num_positive_candidates"].reshape(
                    D, B, *batch["num_positive_candidates"].shape[1:]
                )
                if batch["num_positive_candidates"] is not None
                else None
            ),
            sample_weights=(
                batch["sample_weights"].reshape(D, B, *batch["sample_weights"].shape[1:])
                if batch.get("sample_weights") is not None
                else None
            ),
            packing_layout=layout,
        )

    history_post_hashes = batch["history_seq"]["post_hashes"]
    real_history_len = np.count_nonzero(history_post_hashes[:, :, 0], axis=1).reshape(
        num_devices_per_process, bs_per_device
    )
    real_history_len = real_history_len.astype(np.int32, copy=False)
    packed_user_hashes = batch["user_hashes"].reshape(num_devices_per_process, bs_per_device, -1)
    user_valid = packed_user_hashes[:, :, 0] != 0
    real_history_len = np.where(user_valid, real_history_len, 0).astype(np.int32, copy=False)

    user_idx = np.arange(bs_per_device, dtype=np.intp)
    device_idx = np.arange(num_devices_per_process, dtype=np.intp)[:, None]
    user_stride = num_user_prefix_tokens + transformer_candidate_seq_len
    packed_candidate_len = bs_per_device * candidate_seq_len

    if dist is None:
        history_real_to_use = real_history_len
        off = num_user_prefix_tokens + transformer_candidate_seq_len
        assert (off + history_seq_len) % block_size == 0, (
            f"off + history_seq_len ({off} + {history_seq_len}) must be a multiple "
            f"of block_size={block_size} for block-aligned inference packing"
        )
        history_len = ((real_history_len + off + block_size - 1) // block_size) * block_size - off
        packed_history_len = bs_per_device * history_seq_len
    else:
        assert rng is not None
        assert dist.max_len <= history_seq_len
        sampled_history_len = dist.sample(
            bs_per_device,
            num_devices_per_process,
            rng,
            num_user_prefix_tokens=num_user_prefix_tokens,
            transformer_candidate_seq_len=transformer_candidate_seq_len,
        )
        history_real_to_use = np.minimum(sampled_history_len, real_history_len)
        history_len = sampled_history_len
        packed_history_len = bs_per_device * dist.mean_len

    history_src_start = real_history_len - history_real_to_use
    real_len_flat = history_real_to_use.reshape(-1).astype(np.intp, copy=False)
    pad_count_flat = (history_len - history_real_to_use).reshape(-1).astype(np.intp, copy=False)
    history_cu_seqlens = np.concatenate(
        [
            np.zeros((num_devices_per_process, 1), dtype=np.int32),
            np.cumsum(history_len, axis=1, dtype=np.int32),
        ],
        axis=1,
    )
    history_start_flat = history_cu_seqlens[:, :-1].reshape(-1).astype(np.intp, copy=False)
    total_real_tokens = int(real_len_flat.sum())
    real_len_cum_starts_flat = np.concatenate(
        [np.zeros(1, dtype=np.intp), np.cumsum(real_len_flat[:-1])]
    ).astype(np.intp, copy=False)
    real_offsets_flat = np.arange(total_real_tokens, dtype=np.intp) - np.repeat(
        real_len_cum_starts_flat, real_len_flat
    )
    history_user_idx_flat = np.repeat(np.tile(user_idx, num_devices_per_process), real_len_flat)
    real_per_device = history_real_to_use.sum(axis=1).astype(np.intp, copy=False)
    device_id_by_token = np.repeat(
        np.arange(num_devices_per_process, dtype=np.intp), real_per_device
    )
    history_offsets_flat = np.repeat(pad_count_flat, real_len_flat) + real_offsets_flat
    history_start_by_token = np.repeat(history_start_flat, real_len_flat)
    packed_pos_flat = history_start_by_token + history_offsets_flat
    history_src_start_flat = history_src_start.reshape(-1).astype(np.intp, copy=False)
    history_src_start_by_token = np.repeat(history_src_start_flat, real_len_flat)
    src_history_idx = (
        device_id_by_token * (bs_per_device * history_seq_len)
        + history_user_idx_flat * history_seq_len
        + history_src_start_by_token
        + real_offsets_flat
    )
    dst_history_idx = device_id_by_token * packed_history_len + packed_pos_flat
    packed_history_seq = _gather_scatter_sequence(
        batch["history_seq"], history_seq_len, src_history_idx, dst_history_idx, packed_history_len
    )
    packed_candidate_seq = _reshape_sequence(
        batch["candidate_seq"], candidate_seq_len, packed_candidate_len
    )

    packed_seq_len = (
        bs_per_device * (num_user_prefix_tokens + transformer_candidate_seq_len)
        + packed_history_len
    )
    seq_starts = history_cu_seqlens[:, :-1] + user_idx[None, :] * user_stride
    seq_starts_flat = seq_starts.reshape(-1).astype(np.intp, copy=False)
    segment_ids = np.zeros((num_devices_per_process, packed_seq_len), dtype=np.int32)
    segment_ids_flat = segment_ids.reshape(-1)

    for prefix_offset in range(num_user_prefix_tokens):
        segment_ids[device_idx, seq_starts + prefix_offset] = user_valid.astype(np.int32)

    history_positions_flat = (
        np.repeat(seq_starts_flat, real_len_flat) + num_user_prefix_tokens + history_offsets_flat
    )
    segment_ids_flat[device_id_by_token * packed_seq_len + history_positions_flat] = 1
    history_positions = np.zeros((num_devices_per_process, packed_history_len), dtype=np.int32)
    history_positions[device_id_by_token, packed_pos_flat] = history_positions_flat.astype(
        np.int32, copy=False
    )

    if transformer_candidate_seq_len > 0:
        packed_candidate_post_hashes = packed_candidate_seq["post_hashes"]
        assert packed_candidate_post_hashes is not None
        candidate_user_idx = np.repeat(user_idx, transformer_candidate_seq_len)
        candidate_offsets = np.tile(
            np.arange(transformer_candidate_seq_len, dtype=np.int32), bs_per_device
        )
        candidate_positions = (
            seq_starts[:, candidate_user_idx]
            + num_user_prefix_tokens
            + history_len[:, candidate_user_idx]
            + candidate_offsets[None, :]
        ).astype(np.int32, copy=False)
        segment_ids[device_idx, candidate_positions] = -(
            packed_candidate_post_hashes[:, : candidate_positions.shape[1], 0] != 0
        ).astype(np.int32)
    else:
        candidate_positions = np.zeros((num_devices_per_process, 0), dtype=np.int32)
    cu_seqlens = np.concatenate(
        [
            np.zeros((num_devices_per_process, 1), dtype=np.int32),
            np.cumsum(
                num_user_prefix_tokens + history_len + transformer_candidate_seq_len,
                axis=1,
                dtype=np.int32,
            ),
        ],
        axis=1,
    )
    padding_mask = segment_ids != 0

    positions = np.zeros((num_devices_per_process, packed_seq_len, 3), dtype=np.float32)
    for prefix_offset in range(1, num_user_prefix_tokens):
        positions[device_idx, seq_starts + prefix_offset, 0] = np.where(
            user_valid, float(prefix_offset), 0.0
        )
    if transformer_candidate_seq_len > 0:
        positions[device_idx, candidate_positions, 0] = float(
            num_user_prefix_tokens + history_seq_len
        )
    history_len_by_token = np.repeat(history_len.reshape(-1), real_len_flat)
    positions.reshape(-1, 3)[device_id_by_token * packed_seq_len + history_positions_flat, 0] = (
        num_user_prefix_tokens + history_seq_len - history_len_by_token + history_offsets_flat
    ).astype(np.float32, copy=False)

    return _make_batch(
        packed_history_seq,
        packed_candidate_seq,
        SequencePackedLayout(
            cu_seqlens=cu_seqlens,
            segment_ids=segment_ids,
            history_positions=history_positions,
            candidate_positions=candidate_positions,
            padding_mask=padding_mask,
            positions=positions,
        ),
    )
