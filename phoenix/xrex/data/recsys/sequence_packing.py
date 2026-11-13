# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import jax
import numpy as np

from xai_configlib import Config, configclass
from xrex.data.recsys.array_allocator import ArrayAllocator
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


def shrink_empty_history_slots(
    history_len: np.ndarray,
    real_history_len: np.ndarray,
    off: int,
    block_size: int,
    max_len: int,
) -> np.ndarray:
    blocks = (history_len + off) // block_size
    empty = real_history_len == 0
    if not empty.any():
        return history_len
    row_total = blocks.sum(axis=1)
    min_blocks = max(1, (off + block_size - 1) // block_size)
    max_blocks = (max_len + off + block_size - 1) // block_size
    blocks = np.where(empty, np.minimum(blocks, min_blocks), blocks)
    for d in range(blocks.shape[0]):
        others = np.flatnonzero(~empty[d])
        if others.size == 0:
            continue
        need = int(row_total[d] - blocks[d].sum())
        if need > 0:
            sub = fit_sample_budget(
                blocks[d, others][None, :],
                int(blocks[d, others].sum()) + need,
                min_blocks,
                max_blocks,
            )
            blocks[d, others] = sub[0]
    return (blocks * block_size - off).astype(np.int32, copy=False)


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
    cand_slot_lens: np.ndarray | None = None
    candidate_cu_seqlens: np.ndarray | None = None
    candidate_key_starts: np.ndarray | None = None
    candidate_key_counts: np.ndarray | None = None
    prefix_positions: np.ndarray | None = None
    candidate_token_to_slot: np.ndarray | None = None
    candidate_slot_to_token: np.ndarray | None = None
    token_to_source: np.ndarray | None = None
    source_to_token: np.ndarray | None = None
    dropped_candidate_slots: np.ndarray | None = None
    value_scale: np.ndarray | None = None


PACK_BLOCK_SIZE = 128

CANDIDATE_TOKEN_ALIGN = 16


def packed_prefix_positions(layout: SequencePackedLayout) -> np.ndarray:
    if layout.prefix_positions is not None:
        return layout.prefix_positions
    return layout.cu_seqlens[:, :-1]


def candidate_region_history_schedule(
    layout: SequencePackedLayout, candidate_seq_len: int, num_user_prefix_tokens: int
) -> tuple[np.ndarray, np.ndarray]:
    assert layout.prefix_positions is not None, "not a history and candidate packed layout"
    bs_per_device = layout.cu_seqlens.shape[1] - 1
    cu_seqlens = layout.cu_seqlens - np.arange(bs_per_device + 1) * candidate_seq_len
    padding_mask = layout.padding_mask.copy()
    device_idx = np.arange(padding_mask.shape[0])[:, None]
    for j in range(num_user_prefix_tokens):
        padding_mask[device_idx, layout.prefix_positions + j] = True
    return cu_seqlens.astype(np.int32), padding_mask


def candidate_tokens_per_user_needed(
    batch: RecsysFeaturesBatch, num_devices_per_process: int
) -> int:
    post_hashes = batch["candidate_seq"]["post_hashes"]
    assert post_hashes is not None
    counts = (post_hashes[:, :, 0] != 0).sum(axis=1).reshape(num_devices_per_process, -1)
    region_len = _aligned_candidate_tokens(counts).sum(axis=1) + 1
    return int(-(-region_len.max() // counts.shape[1]))


def compact_candidate_layout(
    batch: RecsysFeaturesBatch,
    *,
    num_user_prefix_tokens: int,
    block_size: int,
    packed_seq_len: int,
) -> SequencePackedLayout:
    layout = batch.get("packing_layout")
    assert layout is not None
    cu_full = np.asarray(layout.cu_seqlens)
    D, bs_plus_1 = cu_full.shape
    bs = bs_plus_1 - 1
    hist_hashes = batch["history_seq"]["post_hashes"]
    cand_hashes = batch["candidate_seq"]["post_hashes"]
    assert hist_hashes is not None and cand_hashes is not None
    packed_history_len = hist_hashes.shape[1]
    cand_total = cand_hashes.shape[1]
    candidate_seq_len = cand_total // bs
    prefix = num_user_prefix_tokens
    history_len = np.diff(cu_full, axis=1) - prefix - candidate_seq_len
    cand_valid = np.asarray(cand_hashes)[:, :, 0].reshape(D, bs, candidate_seq_len) != 0
    last_valid = np.where(
        cand_valid.any(axis=2),
        candidate_seq_len - np.argmax(cand_valid[:, :, ::-1], axis=2),
        0,
    )
    cand_slot = np.minimum(
        ((last_valid + block_size - 1) // block_size) * block_size, candidate_seq_len
    ).astype(np.int32)
    per = prefix + history_len + cand_slot
    cu = np.concatenate(
        [np.zeros((D, 1), np.int32), np.cumsum(per, axis=1, dtype=np.int32)], axis=1
    )
    row_end = cu[:, -1]
    if int(row_end.max()) > packed_seq_len:
        raise ValueError(
            f"compact layout needs {int(row_end.max())} tokens > packed_seq_len={packed_seq_len}"
        )

    seg_full = np.asarray(layout.segment_ids)
    pos_full = np.asarray(layout.positions)
    hist_pos_full = np.asarray(layout.history_positions)
    seg = np.zeros((D, packed_seq_len), np.int32)
    positions = np.zeros((D, packed_seq_len, 3), np.float32)
    history_positions = np.zeros((D, packed_history_len), np.int32)
    candidate_positions = np.zeros((D, bs * candidate_seq_len), np.int32)
    t = np.arange(candidate_seq_len, dtype=np.int32)
    for d in range(D):
        hist_cursor = 0
        for u in range(bs):
            hl = int(history_len[d, u])
            cl = int(cand_slot[d, u])
            src = int(cu_full[d, u])
            dst = int(cu[d, u])
            span = prefix + hl + cl
            seg[d, dst : dst + span] = seg_full[d, src : src + span]
            positions[d, dst : dst + span] = pos_full[d, src : src + span]
            assert prefix > 0, "compact layout relies on 0 marking a padded history slot"
            hp = hist_pos_full[d, hist_cursor : hist_cursor + hl]
            history_positions[d, hist_cursor : hist_cursor + hl] = np.where(
                hp != 0, hp + (dst - src), 0
            )
            hist_cursor += hl
            cand_base = dst + prefix + hl
            candidate_positions[d, u * candidate_seq_len : (u + 1) * candidate_seq_len] = np.where(
                t < cl, cand_base + t, max(int(row_end[d]) - 1, 0)
            )
    return SequencePackedLayout(
        cu_seqlens=cu,
        segment_ids=seg,
        history_positions=history_positions,
        candidate_positions=candidate_positions,
        padding_mask=seg != 0,
        positions=positions,
        cand_slot_lens=cand_slot,
    )


def pack_batch(
    batch: RecsysFeaturesBatch,
    num_devices_per_process: int,
    num_user_prefix_tokens: int,
    dist: LengthDistribution | None,
    rng: np.random.Generator | None,
    block_size: int = PACK_BLOCK_SIZE,
    candidate_tokens_per_user: int | None = None,
    min_packed_history_len: int | None = None,
    empty_history_min_slot: bool = False,
    *,
    alloc: ArrayAllocator,
) -> RecsysFeaturesBatch:
    if empty_history_min_slot and dist is not None:
        if not isinstance(dist, BetaLengthDistribution) or dist.block_size != block_size:
            raise ValueError("empty_history_min_slot requires a block-matched Beta distribution")
        if candidate_tokens_per_user is not None:
            raise ValueError("empty_history_min_slot does not support candidate-region packing")
    if min_packed_history_len is not None:
        if (
            isinstance(min_packed_history_len, bool)
            or type(min_packed_history_len) is not int
            or min_packed_history_len < 1
        ):
            raise ValueError("min_packed_history_len must be an integer >= 1")
        if dist is None:
            raise ValueError("min_packed_history_len requires a training distribution")
    _hem = batch["candidate_seq"].get("head_eligible_mask")
    assert _hem is None or bool(np.asarray(_hem).all()), (
        "head_eligible_mask with masked heads is not supported with sequence packing"
    )

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
        if use_dense_gather:
            full_src, pad_rows = src_idx, None
        else:
            full_src = np.zeros(total_packed, dtype=np.intp)
            full_src[dst_idx] = src_idx
            padding = np.ones(total_packed, dtype=np.bool_)
            padding[dst_idx] = False
            pad_rows = np.flatnonzero(padding)
        for k, v in seq.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == seq_len:
                flat = v.reshape(read_bsz_per_process * seq_len, *v.shape[2:])
                packed = alloc.empty((total_packed, *v.shape[2:]), v.dtype)
                np.take(flat, full_src, axis=0, mode="clip", out=packed)
                if pad_rows is not None:
                    packed[pad_rows] = 0
                out[k] = packed.reshape(num_devices_per_process, packed_len, *v.shape[2:])
            else:
                out[k] = v
        return cast(PostSeq, out)

    def _make_batch(
        packed_history: PostSeq, packed_candidates: PostSeq, layout: SequencePackedLayout
    ) -> RecsysFeaturesBatch:
        D, B = num_devices_per_process, bs_per_device
        out = RecsysFeaturesBatch(
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
                sw.reshape(D, B, *sw.shape[1:])
                if (sw := batch.get("sample_weights")) is not None
                else None
            ),
            sample_source=(
                ss.reshape(D, B, *ss.shape[1:])
                if (ss := batch.get("sample_source")) is not None
                else None
            ),
            packing_layout=layout,
        )
        if (il := batch.get("user_id_list_hashes")) is not None and (
            ilm := batch.get("user_id_list_mask")
        ) is not None:
            out["user_id_list_hashes"] = il.reshape(D, B, *il.shape[1:])
            out["user_id_list_mask"] = ilm.reshape(D, B, *ilm.shape[1:])
        return out

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
        if min_packed_history_len is not None:
            off = num_user_prefix_tokens + transformer_candidate_seq_len
            slot_granularity = dist.block_size if isinstance(dist, BetaLengthDistribution) else 1
            slot_offset = off if isinstance(dist, BetaLengthDistribution) else 0
            lo = (
                max(min_packed_history_len, dist.min_len) + slot_offset + slot_granularity - 1
            ) // slot_granularity
            hi = (dist.max_len + slot_offset + slot_granularity - 1) // slot_granularity
            total_tokens = bs_per_device * (dist.mean_len + slot_offset)
            if (
                total_tokens % slot_granularity
                or not bs_per_device * lo <= total_tokens // slot_granularity <= bs_per_device * hi
            ):
                raise ValueError("min_packed_history_len is infeasible for the sampled budget")
            sampled_slots = (sampled_history_len + slot_offset) // slot_granularity
            sampled_slots = np.clip(sampled_slots, lo, hi)
            sampled_slots = fit_sample_budget(
                sampled_slots, total_tokens // slot_granularity, lo, hi
            )
            sampled_history_len = (sampled_slots * slot_granularity - slot_offset).astype(
                np.int32, copy=False
            )
        if empty_history_min_slot:
            sampled_history_len = shrink_empty_history_slots(
                sampled_history_len,
                real_history_len,
                off=num_user_prefix_tokens + transformer_candidate_seq_len,
                block_size=block_size,
                max_len=dist.max_len,
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
    dropped_candidate_slots = None
    candidate_region_len = None
    if candidate_tokens_per_user is not None:
        assert dist is not None and transformer_candidate_seq_len > 0, (
            "candidate_tokens_per_user is training-only and needs transformer candidates"
        )
        candidate_region_len = bs_per_device * candidate_tokens_per_user
        assert candidate_region_len % block_size == 0, (
            f"{bs_per_device=} (per microbatch) * {candidate_tokens_per_user=} must be a multiple "
            f"of {block_size=}"
        )
        packed_candidate_seq, dropped_candidate_slots = _drop_candidates_over_budget(
            packed_candidate_seq, candidate_seq_len, candidate_region_len - 1, alloc=alloc
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

    layout = SequencePackedLayout(
        cu_seqlens=cu_seqlens,
        segment_ids=segment_ids,
        history_positions=history_positions,
        candidate_positions=candidate_positions,
        padding_mask=padding_mask,
        positions=positions,
    )
    if candidate_region_len is not None:
        assert dropped_candidate_slots is not None
        layout = _candidate_region_layout(
            layout,
            seq_starts,
            history_len + num_user_prefix_tokens,
            num_user_prefix_tokens,
            candidate_region_len,
            dropped_candidate_slots,
        )
    return _make_batch(packed_history_seq, packed_candidate_seq, layout)


def _aligned_candidate_tokens(counts: np.ndarray) -> np.ndarray:
    return -(-counts // CANDIDATE_TOKEN_ALIGN) * CANDIDATE_TOKEN_ALIGN


def _drop_candidates_over_budget(
    seq: PostSeq, slots_per_user: int, budget: int, *, alloc: ArrayAllocator
) -> tuple[PostSeq, np.ndarray]:
    post_hashes = seq["post_hashes"]
    assert post_hashes is not None
    num_devices, num_slots = post_hashes.shape[:2]
    valid = post_hashes[:, :, 0] != 0
    counts = valid.reshape(num_devices, -1, slots_per_user).sum(axis=2)
    dropped = np.zeros(num_devices, dtype=np.int32)
    over_budget = np.flatnonzero(_aligned_candidate_tokens(counts).sum(axis=1) > budget)
    if over_budget.size == 0:
        return seq, dropped
    drop = np.zeros_like(valid)
    slot_in_user = np.arange(num_slots) % slots_per_user
    for d in over_budget:
        used = int(_aligned_candidate_tokens(counts[d]).sum())
        valid_slots = np.flatnonzero(valid[d])
        for i in valid_slots[np.argsort(-slot_in_user[valid_slots], kind="stable")]:
            if used <= budget:
                break
            u = i // slots_per_user
            used -= int(
                _aligned_candidate_tokens(counts[d, u])
                - _aligned_candidate_tokens(counts[d, u] - 1)
            )
            counts[d, u] -= 1
            drop[d, i] = True
            dropped[d] += 1
        dropped_slots = np.flatnonzero(drop[d])
        for i in dropped_slots[np.argsort(slot_in_user[dropped_slots], kind="stable")]:
            u = i // slots_per_user
            extra = int(
                _aligned_candidate_tokens(counts[d, u] + 1)
                - _aligned_candidate_tokens(counts[d, u])
            )
            if used + extra <= budget:
                used += extra
                counts[d, u] += 1
                drop[d, i] = False
                dropped[d] -= 1
    kept_seq = {}
    for key, value in seq.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[1] == num_slots:
            value = alloc.copy(value, value.dtype)
            value[drop] = 0
        kept_seq[key] = value
    return cast(PostSeq, kept_seq), dropped


def _candidate_region_layout(
    layout: SequencePackedLayout,
    seq_starts: np.ndarray,
    span_lens: np.ndarray,
    num_user_prefix_tokens: int,
    candidate_region_len: int,
    dropped_candidate_slots: np.ndarray,
) -> SequencePackedLayout:
    assert num_user_prefix_tokens > 0, (
        "history position 0 marks padding, so a prefix token is needed"
    )
    num_devices, old_len = layout.segment_ids.shape
    bs_per_device = seq_starts.shape[1]
    device_idx = np.arange(num_devices)[:, None]
    old_candidate_tokens = layout.candidate_positions
    slots_per_user = old_candidate_tokens.shape[1] // bs_per_device
    valid = layout.segment_ids[device_idx, old_candidate_tokens] != 0
    history_region_len = int(span_lens[0].sum())
    assert (span_lens.sum(axis=1) == history_region_len).all()
    packed_seq_len = history_region_len + candidate_region_len

    old_token = np.full((num_devices, packed_seq_len), old_len, dtype=np.int64)
    in_history = np.ones((num_devices, old_len), dtype=bool)
    in_history[device_idx, old_candidate_tokens] = False
    old_history_tokens = np.nonzero(in_history)[1].reshape(num_devices, history_region_len)
    span_starts = np.cumsum(span_lens, axis=1) - span_lens
    key_counts = np.zeros((num_devices, bs_per_device), dtype=np.int64)
    for d in range(num_devices):
        token_user = np.repeat(np.arange(bs_per_device), span_lens[d])
        offset = np.arange(history_region_len) - span_starts[d][token_user]
        is_prefix = offset < num_user_prefix_tokens
        is_key = layout.padding_mask[d, old_history_tokens[d]] | is_prefix
        key_counts[d] = np.bincount(token_user[is_key], minlength=bs_per_device)
        user_pad = (span_lens[d] - key_counts[d])[token_user]
        new_offset = np.where(
            is_prefix,
            offset + user_pad,
            np.where(
                offset < num_user_prefix_tokens + user_pad, offset - num_user_prefix_tokens, offset
            ),
        )
        old_token[d, span_starts[d][token_user] + new_offset] = old_history_tokens[d]

    counts = valid.reshape(num_devices, bs_per_device, slots_per_user).sum(axis=2)
    candidate_cu_seqlens = np.zeros((num_devices, bs_per_device + 1), dtype=np.int32)
    candidate_cu_seqlens[:, 1:] = np.cumsum(_aligned_candidate_tokens(counts), axis=1)
    assert int(candidate_cu_seqlens[:, -1].max()) <= candidate_region_len - 1
    rank_in_user = (np.cumsum(valid.reshape(num_devices, bs_per_device, -1), axis=2) - 1).reshape(
        num_devices, -1
    )
    slot_user = np.arange(old_candidate_tokens.shape[1]) // slots_per_user
    slot_token = history_region_len + candidate_cu_seqlens[:, slot_user] + rank_in_user
    d_valid, s_valid = np.nonzero(valid)
    old_token[d_valid, slot_token[d_valid, s_valid]] = old_candidate_tokens[d_valid, s_valid]

    is_copy = old_token < old_len
    new_token = np.full((num_devices, old_len), packed_seq_len - 1, dtype=np.int32)
    d_copy, t_copy = np.nonzero(is_copy)
    new_token[d_copy, old_token[d_copy, t_copy]] = t_copy

    clamped_old_token = np.minimum(old_token, old_len - 1)
    segment_ids = np.where(
        is_copy, np.take_along_axis(layout.segment_ids, clamped_old_token, axis=1), 0
    )
    positions = np.take_along_axis(layout.positions, clamped_old_token[:, :, None], axis=1)
    positions[~is_copy] = 0.0
    padding_mask = segment_ids != 0

    history_positions = np.take_along_axis(new_token, layout.history_positions, axis=1)
    candidate_positions = np.where(
        valid, np.take_along_axis(new_token, old_candidate_tokens, axis=1), packed_seq_len - 1
    ).astype(np.int32)
    prefix_positions = np.take_along_axis(new_token, seq_starts.astype(np.intp), axis=1)

    region_token = slot_token[d_valid, s_valid] - history_region_len
    num_slots = old_candidate_tokens.shape[1]
    candidate_token_to_slot = np.full(
        (num_devices, candidate_region_len), num_slots, dtype=np.int32
    )
    candidate_token_to_slot[d_valid, region_token] = s_valid
    candidate_slot_to_token = np.full(
        old_candidate_tokens.shape, candidate_region_len, dtype=np.int32
    )
    candidate_slot_to_token[d_valid, s_valid] = region_token

    num_prefix_sources = bs_per_device * num_user_prefix_tokens
    num_history_entries = layout.history_positions.shape[1]
    zero_source = num_prefix_sources + num_history_entries + candidate_region_len
    token_to_source = np.full((num_devices, packed_seq_len), zero_source, dtype=np.int32)
    for j in range(num_user_prefix_tokens):
        token_to_source[device_idx, prefix_positions + j] = (
            np.arange(bs_per_device) * num_user_prefix_tokens + j
        )
    d_h, s_h = np.nonzero(layout.history_positions != 0)
    token_to_source[d_h, history_positions[d_h, s_h]] = num_prefix_sources + s_h
    token_to_source[d_valid, history_region_len + region_token] = (
        num_prefix_sources + num_history_entries + region_token
    )
    source_to_token = np.full((num_devices, zero_source + 1), packed_seq_len, dtype=np.int32)
    d_used, t_used = np.nonzero(token_to_source != zero_source)
    source_to_token[d_used, token_to_source[d_used, t_used]] = t_used

    span_ends = span_starts + span_lens
    key_starts = np.empty((num_devices, bs_per_device + 1), dtype=np.int32)
    key_starts[:, :-1] = span_ends - key_counts
    key_starts[:, -1] = history_region_len
    return SequencePackedLayout(
        cu_seqlens=layout.cu_seqlens,
        segment_ids=segment_ids.astype(np.int32, copy=False),
        history_positions=history_positions,
        candidate_positions=candidate_positions,
        padding_mask=padding_mask,
        positions=positions,
        candidate_cu_seqlens=candidate_cu_seqlens,
        candidate_key_starts=key_starts,
        candidate_key_counts=key_counts.astype(np.int32),
        prefix_positions=prefix_positions,
        candidate_token_to_slot=candidate_token_to_slot,
        candidate_slot_to_token=candidate_slot_to_token,
        token_to_source=token_to_source,
        source_to_token=source_to_token,
        dropped_candidate_slots=dropped_candidate_slots[:, None],
    )
