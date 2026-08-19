# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import os
import time
from dataclasses import field

import haiku as hk
import jax
import jax.numpy as jnp
import jax.profiler as jax_profiler
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import xai_recsys_engine
from xai_configlib import configclass

from xrex.data.parquet_recsys import PhoenixDataset
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.data.retrieval_dataset import PHOENIX_INDEX_BASE
from xrex.inference.h2d import (
    EmbeddingSlices,
    lookup_h2d_embeddings,
)
from xrex.inference.model_runner import (
    RecsysInferenceState,
    RetrievalModelRunner,
)
from xrex.models.recsys_embedding import RecsysEmbeddings
from xrex.models.recsys_two_tower_serving_filters import forward_with_filters
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.models.topic_categories import (
    NUM_TOPIC_INT32S,
    TOPIC_ID_TO_BITS,
    bitmaps_to_int32_array,
    topic_ids_to_bitmap,
)
from xrex.train.trainer import RecsysTwoTowerModelConfig, TrainerContext
from xrex.utils.aot import JittedOrCompiled
from xrex.utils.log_timer import Timer

logger = logging.getLogger(__name__)

_TOPIC_PARQUET_DIR = str(PHOENIX_INDEX_BASE / "post_creation_snapshots_topic")
_TOPIC_PARQUET_PATHS: dict[int, str] = {
    0: f"{_TOPIC_PARQUET_DIR}/1fav_topic_1day.parquet",
    1: f"{_TOPIC_PARQUET_DIR}/1fav_topic_option_1_1day.parquet",
    2: f"{_TOPIC_PARQUET_DIR}/1fav_topic_option_2_1day.parquet",
    3: f"{_TOPIC_PARQUET_DIR}/1fav_topic_option_3_1day.parquet",
    4: f"{_TOPIC_PARQUET_DIR}/1fav_topic_option_4_1day.parquet",
    5: f"{_TOPIC_PARQUET_DIR}/1fav_topic_option_5_1day.parquet",
}


@configclass
class FilteredRetrievalModelRunner(RetrievalModelRunner):
    enable_bloom_filter: bool = False
    enable_topic_filter: bool = False

    _all_topic_bitmaps: dict[int, jax.Array] = field(default_factory=dict)

    _mask_pinned_by_bs: dict[int, jax.Array] = field(default_factory=dict)
    _mask_shard_views_by_bs: dict[int, tuple[npt.NDArray[np.uint8], ...]] = field(
        default_factory=dict
    )
    _all_post_ids_np: npt.NDArray[np.int64] | None = None
    _all_dataset_types_np: npt.NDArray[np.int32] | None = None
    _mask_missing_warned: set[int] = field(default_factory=set, init=False)

    @property
    def _live_swap_blocked_by_candidate_filters(self) -> bool:
        return self.enable_bloom_filter or self.enable_topic_filter

    def _get_dataset_ranges(self, target_dataset_types, ranges_override=None):
        if self.enable_bloom_filter or self.enable_topic_filter:
            return None
        return super()._get_dataset_ranges(target_dataset_types, ranges_override)

    def maybe_load_checkpoint(
        self, ctx: TrainerContext, tag: str | None = None
    ) -> tuple[bool, int, int]:
        new_ckpt_loaded, elapsed_samples, _ = super().maybe_load_checkpoint(ctx, tag)
        if new_ckpt_loaded:
            assert isinstance(self.state, RecsysInferenceState)
            assert (
                self.state.post_embeddings is not None
                and self.state.post_embeddings.post_ids is not None
                and self.state.post_embeddings.author_ids is not None
                and self.state.post_embeddings.dataset_types is not None
            )
            self.all_post_ids = self.two_int32_to_int64(self.state.post_embeddings.post_ids)
            self.all_author_ids = self.two_int32_to_int64(self.state.post_embeddings.author_ids)
            self.all_dataset_types = np.asarray(
                self.state.post_embeddings.dataset_types, dtype=np.int32
            )
            self._compute_dataset_ranges()
            if self.enable_topic_filter:
                self._reload_topic_bitmaps()
            if self.enable_bloom_filter and self.all_post_ids is not None:
                self._init_bloom_filter_pinned_buffer()
        return new_ckpt_loaded, elapsed_samples, _

    def _on_post_hotswap(self) -> None:
        assert isinstance(self.state, RecsysInferenceState)
        if (
            self.state.post_embeddings is not None
            and self.state.post_embeddings.post_ids is not None
        ):
            self.all_post_ids = self.two_int32_to_int64(self.state.post_embeddings.post_ids)
            self.all_author_ids = self.two_int32_to_int64(self.state.post_embeddings.author_ids)
            self.all_dataset_types = np.asarray(
                self.state.post_embeddings.dataset_types, dtype=np.int32
            )
            self._compute_dataset_ranges()
            if self.enable_bloom_filter:
                self._init_bloom_filter_pinned_buffer()
            if self.enable_topic_filter:
                self._reload_topic_bitmaps()
            logger.info("[hotswap] Rebuilt retrieval metadata (post_ids, author_ids, bloom filter)")

    def _reload_topic_bitmaps(self) -> None:
        new_bitmaps: dict[int, jax.Array] = {}
        for option, path in _TOPIC_PARQUET_PATHS.items():
            bitmap = self._load_topic_bitmaps_from_parquet(path)
            if bitmap is not None:
                new_bitmaps[option] = bitmap
                logger.info(f"Loaded topic bitmaps for option {option}: shape={bitmap.shape}")
        if not new_bitmaps:
            logger.warning(
                "No topic bitmaps loaded from any parquet file; "
                "topic filtering will silently pass through all candidates."
            )
        self._all_topic_bitmaps = new_bitmaps

    def _load_topic_bitmaps_from_parquet(self, parquet_path: str) -> jax.Array | None:
        if self.all_post_ids is None:
            return None

        if not os.path.exists(parquet_path):
            logger.warning(f"Topic parquet file not found: {parquet_path}")
            return None

        logger.info(f"Loading topic bitmaps from parquet: {parquet_path}")

        try:
            table = pq.read_table(parquet_path, columns=["post_id", "topic_entity_ids"])
            parquet_post_ids = table.column("post_id").to_numpy()
            topic_ids_list = table.column("topic_entity_ids").to_pylist()

            raw_bitmaps = [topic_ids_to_bitmap(tids) for tids in topic_ids_list]
            parquet_bitmaps = bitmaps_to_int32_array(raw_bitmaps)

            sort_idx = np.argsort(parquet_post_ids)
            sorted_ids = parquet_post_ids[sort_idx]
            sorted_bitmaps = parquet_bitmaps[sort_idx]

            indices = np.searchsorted(sorted_ids, self.all_post_ids)
            clamped_indices = np.minimum(indices, len(sorted_ids) - 1)
            valid = (indices < len(sorted_ids)) & (sorted_ids[clamped_indices] == self.all_post_ids)

            bitmaps = np.zeros((len(self.all_post_ids), NUM_TOPIC_INT32S), dtype=np.int32)
            bitmaps[valid] = sorted_bitmaps[clamped_indices[valid]]

            matched_count = np.sum(valid)
            logger.info(
                f"Loaded topic bitmaps: {matched_count}/{len(self.all_post_ids)} posts matched"
            )
            return jnp.array(bitmaps, dtype=jnp.int32)

        except Exception as e:
            logger.error(f"Failed to load topic bitmaps from parquet: {e}")
            return None

    def _init_bloom_filter_pinned_buffer(self) -> None:
        from xai_checkpointing.common import (
            _unsafe_jax2np,
        )

        M = len(self.all_post_ids)

        self._mask_pinned_by_bs = {}
        self._mask_shard_views_by_bs = {}
        for bs in self.sorted_buckets:
            with self.mesh:
                pinned = jnp.empty(
                    (bs, M),
                    dtype=jnp.bool_,
                    device=self.data_sharding.with_memory_kind("pinned_host"),
                )

            views = tuple(
                np.ndarray(
                    shape=(shard.data.size,),
                    dtype=np.uint8,
                    buffer=_unsafe_jax2np(shard.data),
                )
                for shard in pinned.addressable_shards
            )
            assert bs % len(views) == 0, (
                f"bloom bucket {bs} must divide evenly across {len(views)} mask shards"
            )
            for view in views:
                view[:] = 1
            self._mask_pinned_by_bs[bs] = pinned
            self._mask_shard_views_by_bs[bs] = views

        self._all_post_ids_np = np.ascontiguousarray(self.all_post_ids, dtype=np.int64)
        self._all_dataset_types_np = np.ascontiguousarray(
            np.asarray(self.all_dataset_types).reshape(-1), dtype=np.int32
        )

        total_mb = sum(bs * M for bs in self.sorted_buckets) / 1e6
        logger.info(
            f"Allocated pinned bloom filter masks: buckets={self.sorted_buckets}, "
            f"M={M}, total={total_mb:.1f} MB, "
            f"num_shards={len(next(iter(self._mask_shard_views_by_bs.values())))}"
        )

    def _build_eligible_mask(
        self,
        request: xai_recsys_engine.RetrieveRequestBatch | None,
        batch_id: int,
        bucket_size: int | None = None,
    ) -> jax.Array | None:
        if not self.enable_bloom_filter:
            return None

        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        pinned = self._mask_pinned_by_bs.get(bs)
        shard_views = self._mask_shard_views_by_bs.get(bs)
        if pinned is None or shard_views is None:
            if request is not None and bs not in self._mask_missing_warned:
                self._mask_missing_warned.add(bs)
                logger.warning(
                    "[BloomFilter] no pinned mask buffer for bucket %d "
                    "(initialized buckets: %s); serving UNFILTERED",
                    bs,
                    sorted(self._mask_pinned_by_bs),
                )
            return None

        t0 = time.monotonic()

        if request is None or self.all_post_ids is None:
            for view in shard_views:
                view[:] = 1
        else:
            post_ids = (
                self._all_post_ids_np
                if self._all_post_ids_np is not None
                else (np.ascontiguousarray(self.all_post_ids, dtype=np.int64))
            )
            ds_types = getattr(self, "_all_dataset_types_np", None)
            if ds_types is None:
                ds_types = np.ascontiguousarray(
                    np.asarray(self.all_dataset_types).reshape(-1), dtype=np.int32
                )
            request.build_eligible_mask_into_shards(
                post_ids,
                ds_types,
                bs,
                list(shard_views),
            )

        t1 = time.monotonic()

        result = jax.device_put(pinned, self.data_sharding)

        t2 = time.monotonic()
        rust_ms = (t1 - t0) * 1000
        h2d_ms = (t2 - t1) * 1000
        logger.info(
            f"[BloomFilter] batch_id={batch_id} bucket={bs} | "
            f"rust_build={rust_ms:.1f}ms, h2d={h2d_ms:.1f}ms, total={rust_ms + h2d_ms:.1f}ms"
        )
        return result

    def _build_eligible_mask_pipelined(
        self,
        request: xai_recsys_engine.RetrieveRequestBatch | None,
        batch_id: int,
        bucket_size: int | None = None,
    ) -> jax.Array | None:
        return self._build_eligible_mask(request, batch_id, bucket_size)

    def gather_embeddings_and_forward(
        self,
        state: RecsysInferenceState,
        batch: RecsysFeaturesBatch,
        rng: jax.Array,
        request_in_flight_id: int,
        batch_id: int,
        request: xai_recsys_engine.RetrieveRequestBatch | None = None,
        eligible_mask: jax.Array | None = None,
        bucket_size: int | None = None,
    ) -> dict[int, tuple[jax.Array, jax.Array]]:
        forward_jit = self._forward_jit_for_bucket(bucket_size)
        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        assert state.post_embeddings is not None
        h2d_state = self._get_h2d_state(request_in_flight_id, bucket_size)
        with jax_profiler.TraceAnnotation(
            "lookup_h2d_embeddings",
            batch_id=batch_id,
        ):
            t = Timer()
            recsys_embeddings, _ = lookup_h2d_embeddings(
                h2d_state,
                batch,
                data_sharding=self.data_sharding,
                batch_id=batch_id,
                emb_table_override=self.active_emb_table if self._emb_table_slots else None,
                num_threads=self.embedding_gather_threads,
            )
            emb_lookup_time = t.elapsed()
            logger.info(f"[batch={batch_id}] Embedding gather took {emb_lookup_time * 1000:.1f}ms")
            if self.metrics_publisher is not None:
                self.metrics_publisher.model_latency.labels("embedding_lookup").observe(
                    emb_lookup_time
                )
        target_dataset_types = tuple(ds.value for ds in self.retrieval_dataset_types)

        dataset_ranges = self._get_dataset_ranges(target_dataset_types)

        if eligible_mask is None:
            eligible_mask = self._build_eligible_mask(request, batch_id, bucket_size)
        if eligible_mask is not None:
            assert eligible_mask.shape[0] == bs, (
                f"eligible_mask rows {eligible_mask.shape[0]} != bucket {bs}; "
                "per-bucket mask build must match the bucket forward"
            )

        topic_bitmaps = None
        topic_user_bitmasks = None
        if self.enable_topic_filter:
            topic_filter_mode = 0
            if request is not None:
                topic_filter_mode = request.get_topic_filter_mode()

            topic_bitmaps = self._all_topic_bitmaps.get(topic_filter_mode)
            if topic_bitmaps is None:
                if topic_filter_mode != 0:
                    logger.warning(
                        "topic_filter_mode=%d not loaded in _all_topic_bitmaps; "
                        "falling back to mode 0",
                        topic_filter_mode,
                    )
                topic_bitmaps = self._all_topic_bitmaps.get(0)

            if topic_bitmaps is None and state.post_embeddings is not None:
                N = state.post_embeddings.embeddings.x.shape[0]
                topic_bitmaps = jnp.zeros((N, NUM_TOPIC_INT32S), dtype=jnp.int32)

            batch_size = batch["user_hashes"].shape[0]
            if request is not None and self._all_topic_bitmaps:
                topic_entity_id_list = request.get_topic_entity_ids()
                raw_bitmasks = [0] * batch_size
                for i, tids in enumerate(topic_entity_id_list):
                    if i >= batch_size:
                        break
                    mask = 0
                    for tid in tids:
                        if tid != 0 and tid in TOPIC_ID_TO_BITS:
                            for bit in TOPIC_ID_TO_BITS[tid]:
                                mask |= 1 << bit
                    raw_bitmasks[i] = mask
                topic_user_bitmasks = jnp.array(
                    bitmaps_to_int32_array(raw_bitmasks), dtype=jnp.int32
                )
            else:
                topic_user_bitmasks = jnp.zeros((batch_size, NUM_TOPIC_INT32S), dtype=jnp.int32)

        with jax_profiler.TraceAnnotation(
            "forward_jit/retrieval",
            batch_id=batch_id,
        ):
            results_tuple = forward_jit(
                state.params,
                rng,
                batch,
                recsys_embeddings,
                state.post_embeddings.embeddings.x,
                state.post_embeddings.dataset_types,
                self.large_k,
                target_dataset_types,
                eligible_mask,
                topic_bitmaps,
                topic_user_bitmasks,
                dataset_ranges,
            )
        results_dict = {
            ds_type: result for ds_type, result in zip(target_dataset_types, results_tuple)
        }
        return results_dict

    def _build_two_tower_jit(self, bs: int) -> JittedOrCompiled:
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        assert isinstance(self.dataset, PhoenixDataset)

        seqpack_packed_history_len: int | None = None
        seqpack_packed_candidate_len: int | None = None
        seqpack_bs_per_device: int | None = None
        if self.using_seqpack:
            trace_batch = self.example_data(bs)
            _hist_post_seq = int(np.prod(trace_batch["history_seq"]["post_hashes"].shape[1:]))
            _hist_auth_seq = int(np.prod(trace_batch["history_seq"]["auth_hashes"].shape[1:]))
            _cand_post_seq = int(np.prod(trace_batch["candidate_seq"]["post_hashes"].shape[1:]))
            _cand_auth_seq = int(np.prod(trace_batch["candidate_seq"]["auth_hashes"].shape[1:]))
            _user_seq = int(np.prod(trace_batch["user_hashes"].shape[1:]))
            _ip_seq = (
                int(np.prod(trace_batch["user_ip_hashes"].shape[1:]))
                if self.model_config.use_ip_address
                else 0
            )
            seqpack_packed_history_len = trace_batch["history_seq"]["post_hashes"].shape[1]
            seqpack_packed_candidate_len = trace_batch["candidate_seq"]["post_hashes"].shape[1]
            seqpack_bs_per_device = trace_batch["user_hashes"].shape[1]
        else:
            _ht = self.dataset.hash_table
            _hist_post_seq = _ht.num_item_hashes * self.history_seq_len
            _hist_auth_seq = _ht.num_author_hashes * self.history_seq_len
            _cand_post_seq = _ht.num_item_hashes * self.candidate_seq_len
            _cand_auth_seq = _ht.num_author_hashes * self.candidate_seq_len
            _user_seq = _ht.num_user_hashes
            _ip_seq = _ht.num_ip_hashes if self.model_config.use_ip_address else 0
        _user_end = _hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq + _user_seq
        embedding_slices = EmbeddingSlices(
            hist_post_end=_hist_post_seq,
            hist_auth_end=_hist_post_seq + _hist_auth_seq,
            cand_post_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq,
            cand_auth_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq,
            user_end=_user_end,
            user_ip_end=_user_end + _ip_seq,
        )

        def _packed_recsys_embeddings(
            merged_embeddings: jax.Array, sl: EmbeddingSlices
        ) -> RecsysEmbeddings:
            assert seqpack_packed_history_len is not None
            assert seqpack_packed_candidate_len is not None
            assert seqpack_bs_per_device is not None
            D = merged_embeddings.shape[0]
            M = merged_embeddings.shape[-1]
            return RecsysEmbeddings(
                history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :].reshape(
                    D, seqpack_packed_history_len, -1, M
                ),
                history_author_embeddings=merged_embeddings[
                    :, sl.hist_post_end : sl.hist_auth_end, :
                ].reshape(D, seqpack_packed_history_len, -1, M),
                candidate_post_embeddings=merged_embeddings[
                    :, sl.hist_auth_end : sl.cand_post_end, :
                ].reshape(D, seqpack_packed_candidate_len, -1, M),
                candidate_author_embeddings=merged_embeddings[
                    :, sl.cand_post_end : sl.cand_auth_end, :
                ].reshape(D, seqpack_packed_candidate_len, -1, M),
                user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :].reshape(
                    D, seqpack_bs_per_device, -1, M
                ),
                user_ip_embeddings=(
                    merged_embeddings[:, sl.user_end : sl.user_ip_end, :].reshape(
                        D, seqpack_bs_per_device, -1, M
                    )
                    if sl.user_ip_end > sl.user_end
                    else None
                ),
            )

        @hk.transform
        def two_tower_forward_fn(
            batch: RecsysFeaturesBatch,
            merged_embeddings: jax.Array,
            post_embeddings: jax.Array,
            dataset_types: jax.Array,
            large_k: int,
            target_dataset_types: tuple[int, ...],
            eligible_mask: jax.Array | None = None,
            topic_bitmaps: jax.Array | None = None,
            topic_user_bitmasks: jax.Array | None = None,
            dataset_ranges: tuple[tuple[int, int], ...] | None = None,
        ):
            assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
            sl = embedding_slices

            if self.using_seqpack:
                recsys_embeddings = _packed_recsys_embeddings(merged_embeddings, sl)
            else:
                recsys_embeddings = RecsysEmbeddings(
                    history_post_embeddings=merged_embeddings[:, : sl.hist_post_end, :],
                    history_author_embeddings=merged_embeddings[
                        :, sl.hist_post_end : sl.hist_auth_end, :
                    ],
                    candidate_post_embeddings=merged_embeddings[
                        :, sl.hist_auth_end : sl.cand_post_end, :
                    ],
                    candidate_author_embeddings=merged_embeddings[
                        :, sl.cand_post_end : sl.cand_auth_end, :
                    ],
                    user_embeddings=merged_embeddings[:, sl.cand_auth_end : sl.user_end, :],
                    user_ip_embeddings=(
                        merged_embeddings[:, sl.user_end : sl.user_ip_end, :]
                        if sl.user_ip_end > sl.user_end
                        else None
                    ),
                )
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            return forward_with_filters(
                model,
                batch,
                recsys_embeddings,
                post_embeddings,
                dataset_types,
                large_k,
                target_dataset_types,
                eligible_mask,
                topic_bitmaps=topic_bitmaps,
                topic_user_bitmasks=topic_user_bitmasks,
                dataset_ranges=dataset_ranges,
                use_async_topk=self.enable_async_topk,
            )

        return JittedOrCompiled(
            jax.jit(
                two_tower_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                    self.data_sharding,
                    None,
                    self.data_sharding if self.enable_bloom_filter else None,
                    None,
                    None,
                ),
                out_shardings=None,
                static_argnums=[
                    6,
                    7,
                    11,
                ],
            ),
            name=f"two_tower_forward_fn_bs{bs}",
        )
