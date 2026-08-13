# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from xrex.data.retrieval_dataset import RetrievalDataset

logger = logging.getLogger(__name__)


class SidPostIndex:
    def __init__(
        self,
        post_ids: npt.NDArray[np.int64],
        author_ids: npt.NDArray[np.int64],
        post_sids: npt.NDArray[np.int32],
        *,
        sid_num_levels: int = 6,
        codebook_size: int = 256,
        dataset_types: npt.NDArray[np.int32] | None = None,
    ):
        if sid_num_levels <= 0:
            raise ValueError(f"sid_num_levels must be > 0, got {sid_num_levels}")
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be > 0, got {codebook_size}")
        if post_ids.shape != author_ids.shape:
            raise ValueError(
                f"post_ids {post_ids.shape} and author_ids {author_ids.shape} disagree"
            )
        if post_sids.shape != (len(post_ids), sid_num_levels):
            raise ValueError(
                f"post_sids shape {post_sids.shape} != ({len(post_ids)}, {sid_num_levels})"
            )

        self.sid_num_levels: int = sid_num_levels
        self.codebook_size: int = codebook_size

        if dataset_types is None:
            dataset_types = np.zeros(len(post_ids), dtype=np.int32)

        t0 = time.time()
        valid_mask = post_sids[:, 0] != -1
        n_total = len(post_ids)
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            raise ValueError("No rows with valid SIDs in the input corpus; corpus is unusable")
        self.post_ids: npt.NDArray[np.int64] = np.ascontiguousarray(
            post_ids[valid_mask], dtype=np.int64
        )
        self.author_ids: npt.NDArray[np.int64] = np.ascontiguousarray(
            author_ids[valid_mask], dtype=np.int64
        )
        self.sids: npt.NDArray[np.int32] = np.ascontiguousarray(
            post_sids[valid_mask], dtype=np.int32
        )
        self.dataset_types: npt.NDArray[np.int32] = np.ascontiguousarray(
            dataset_types[valid_mask], dtype=np.int32
        )

        logger.info(
            "SidPostIndex loaded: %d posts (filtered %d → %d valid), "
            "sid_num_levels=%d codebook_size=%d, build=%.2fs",
            n_valid,
            n_total,
            n_valid,
            sid_num_levels,
            codebook_size,
            time.time() - t0,
        )

    @classmethod
    def from_datasets(
        cls,
        datasets: list[RetrievalDataset],
        *,
        sid_num_levels: int = 6,
        codebook_size: int = 256,
    ) -> SidPostIndex:
        t0 = time.time()
        post_ids, author_ids, dataset_types, post_sids = RetrievalDataset.load_datasets(
            datasets,
            read_post_sid=True,
            sid_num_levels=sid_num_levels,
        )
        if post_sids is None:
            raise ValueError(
                f"None of the {len(datasets)} datasets had a post_sid column; "
                "corpus is unusable for SID retrieval"
            )
        logger.info(
            "SidPostIndex: loaded %d datasets in %.2fs (%d total rows)",
            len(datasets),
            time.time() - t0,
            len(post_ids),
        )
        return cls(
            post_ids=post_ids,
            author_ids=author_ids,
            post_sids=post_sids,
            sid_num_levels=sid_num_levels,
            codebook_size=codebook_size,
            dataset_types=dataset_types,
        )

    def __len__(self) -> int:
        return len(self.post_ids)


@dataclass(frozen=True)
class ResolveStats:
    n_beams: int = 0
    n_empty: int = 0
    n_collision: int = 0
    n_filled: int = 0
    n_deduped: int = 0


class SidBeamResolver:
    def __init__(
        self,
        post_ids: npt.NDArray[np.int64],
        author_ids: npt.NDArray[np.int64],
        sid_to_post_idx: dict[tuple[int, ...], npt.NDArray[np.int32]],
        *,
        prefix_len: int,
    ):
        self.post_ids: npt.NDArray[np.int64] = np.ascontiguousarray(post_ids, dtype=np.int64)
        self.author_ids: npt.NDArray[np.int64] = np.ascontiguousarray(author_ids, dtype=np.int64)
        self.sid_to_post_idx: dict[tuple[int, ...], npt.NDArray[np.int32]] = sid_to_post_idx
        self.prefix_len: int = prefix_len
        self.num_posts: int = int(self.post_ids.shape[0])
        self.sentinel_idx: int = self.num_posts
        self.post_ids_with_sentinel: npt.NDArray[np.int64] = np.ascontiguousarray(
            np.append(self.post_ids, np.int64(0))
        )
        self.author_ids_with_sentinel: npt.NDArray[np.int64] = np.ascontiguousarray(
            np.append(self.author_ids, np.int64(0))
        )
        self.post_id_set: frozenset[int] = frozenset(int(x) for x in self.post_ids.tolist())

    @classmethod
    def from_sid_post_index(
        cls,
        index: SidPostIndex,
        *,
        prefix_len: int | None = None,
    ) -> SidBeamResolver:
        sids = index.sids
        plen = int(prefix_len) if prefix_len else int(sids.shape[1])
        if plen <= 0 or plen > sids.shape[1]:
            raise ValueError(f"prefix_len={plen} invalid for sids shape {sids.shape}")
        lookup: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx in range(sids.shape[0]):
            lookup[tuple(int(x) for x in sids[idx, :plen])].append(idx)
        sid_to_post_idx = {k: np.asarray(v, dtype=np.int32) for k, v in lookup.items()}
        n_colliding = sum(1 for v in sid_to_post_idx.values() if len(v) > 1)
        logger.info(
            "SidBeamResolver: %d posts, %d unique SID prefixes (prefix_len=%d), "
            "%d multi-post cells (%.2f%%)",
            len(index),
            len(sid_to_post_idx),
            plen,
            n_colliding,
            100.0 * n_colliding / max(len(sid_to_post_idx), 1),
        )
        return cls(
            post_ids=index.post_ids,
            author_ids=index.author_ids,
            sid_to_post_idx=sid_to_post_idx,
            prefix_len=plen,
        )

    @classmethod
    def from_datasets(
        cls,
        datasets: list[RetrievalDataset],
        *,
        sid_num_levels: int = 6,
        codebook_size: int = 256,
        prefix_len: int | None = None,
    ) -> SidBeamResolver:
        index = SidPostIndex.from_datasets(
            datasets,
            sid_num_levels=sid_num_levels,
            codebook_size=codebook_size,
        )
        return cls.from_sid_post_index(index, prefix_len=prefix_len)

    def resolve_to_indices(
        self,
        beam_codes: npt.NDArray[np.int32],
        *,
        max_posts: int,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], ResolveStats]:
        if beam_codes.ndim != 3:
            raise ValueError(f"beam_codes must be [B,K,L], got {beam_codes.shape}")
        if max_posts <= 0:
            raise ValueError(f"max_posts must be > 0, got {max_posts}")

        B, K_beam, _L = beam_codes.shape
        indices = np.full((B, max_posts), self.sentinel_idx, dtype=np.int32)
        n_beams = 0
        n_empty = 0
        n_collision = 0
        n_filled = 0
        n_deduped = 0

        beam_keys = np.ascontiguousarray(beam_codes.astype(np.int32))
        plen = self.prefix_len
        for b in range(B):
            chosen: set[int] = set()
            out_i = 0
            for k in range(K_beam):
                if out_i >= max_posts:
                    break
                n_beams += 1
                key = tuple(int(x) for x in beam_keys[b, k, :plen])
                hit = self.sid_to_post_idx.get(key)
                if hit is None or len(hit) == 0:
                    n_empty += 1
                    continue
                if len(hit) > 1:
                    n_collision += 1
                pick = int(hit.min())
                if pick in chosen:
                    n_deduped += 1
                    continue
                chosen.add(pick)
                indices[b, out_i] = pick
                out_i += 1
                n_filled += 1

        scores = np.tile(np.arange(max_posts, 0, -1, dtype=np.float32), (B, 1))
        stats = ResolveStats(
            n_beams=n_beams,
            n_empty=n_empty,
            n_collision=n_collision,
            n_filled=n_filled,
            n_deduped=n_deduped,
        )
        return indices, scores, stats

    def resolve_to_post_ids(
        self,
        beam_codes: npt.NDArray[np.int32],
        *,
        max_posts: int,
    ) -> tuple[npt.NDArray[np.int64], ResolveStats]:
        indices, _scores, stats = self.resolve_to_indices(beam_codes, max_posts=max_posts)
        post_ids = self.post_ids_with_sentinel[indices]
        return post_ids, stats
