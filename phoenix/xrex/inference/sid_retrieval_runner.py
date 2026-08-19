# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import copy
import dataclasses
import logging
import time

import haiku as hk
import jax
import jax.numpy as jnp
import jax.profiler as jax_profiler
import numpy as np
import numpy.typing as npt
import xai_recsys_engine
from xai_configlib import configclass

from xrex.data.parquet_recsys import PhoenixDataset
from xrex.data.recsys import recsys_batch
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.inference.h2d import EmbeddingSlices, lookup_h2d_embeddings
from xrex.inference.model_runner import BaseModelRunner
from xrex.inference.sid_post_index import SidBeamResolver, SidPostIndex
from xrex.models.recsys_embedding import (
    RecsysEmbeddings,
    get_recsys_embed_jax_array_to_param,
)
from xrex.models.recsys_sid_retrieval_model import RecsysSIDRetrievalConfig
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.train.misc import RecsysInferenceState
from xrex.utils.aot import JittedOrCompiled
from xrex.utils.log_timer import Timer

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _CorpusView:
    resolver: SidBeamResolver

    @property
    def post_ids(self) -> npt.NDArray[np.int64]:
        return self.resolver.post_ids_with_sentinel

    @property
    def author_ids(self) -> npt.NDArray[np.int64]:
        return self.resolver.author_ids_with_sentinel

    @property
    def sentinel_idx(self) -> int:
        return self.resolver.sentinel_idx

    @property
    def num_posts(self) -> int:
        return self.resolver.num_posts


@configclass
class SidRetrievalModelRunner(
    BaseModelRunner[
        xai_recsys_engine.RetrieveRequestBatch,
        RecsysSIDRetrievalConfig,
    ]
):
    sid_retrieval_forward_jit: JittedOrCompiled | None = None
    sid_retrieval_forward_fn: hk.Transformed | None = None
    beam_width: int = 1
    decode_levels: int = 0

    _corpus: _CorpusView | None = None

    def _setup_sid_retrieval_jit(self) -> None:
        assert isinstance(self.model_config, RecsysSIDRetrievalConfig)
        if self.beam_width < 1:
            raise ValueError(
                f"beam_width must be >= 1, got {self.beam_width}. "
                "Use --beam_width=N on the launch command."
            )
        logger.info(
            "Setting up SID retrieval JIT: beam_width=%d "
            "sid_num_levels=%d sid_codebook_size=%d corpus_size=%d large_k=%d",
            self.beam_width,
            self.model_config.sid_num_levels,
            self.model_config.sid_codebook_size,
            self._corpus.num_posts if self._corpus is not None else 0,
            self.large_k,
        )

        _ht = self.model_config.hash_table.hash_keys
        _hist_post_seq = _ht.num_item_hashes * self.history_seq_len
        _hist_auth_seq = _ht.num_author_hashes * self.history_seq_len
        _cand_post_seq = _ht.num_item_hashes * self.candidate_seq_len
        _cand_auth_seq = _ht.num_author_hashes * self.candidate_seq_len
        _user_seq = _ht.num_user_hashes
        _user_ip_seq = (
            _ht.num_ip_hashes if getattr(self.model_config, "use_ip_address", False) else 0
        )
        _user_end = _hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq + _user_seq
        embedding_slices = EmbeddingSlices(
            hist_post_end=_hist_post_seq,
            hist_auth_end=_hist_post_seq + _hist_auth_seq,
            cand_post_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq,
            cand_auth_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq,
            user_end=_user_end,
            user_ip_end=_user_end + _user_ip_seq,
        )

        mesh = self.mesh
        data_sharding = self.data_sharding
        beam_width = self.beam_width
        large_k = self.large_k
        decode_levels = self.decode_levels or None
        model_config = copy.deepcopy(self.model_config)
        object.__setattr__(model_config.model_config.attn_config, "attn_impl", "jax_attn")

        @hk.transform
        def sid_retrieval_forward_fn(
            batch: RecsysFeaturesBatch,
            merged_embeddings: jax.Array,
        ):
            sl = embedding_slices
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
            )
            emb_param = get_recsys_embed_jax_array_to_param(recsys_embeddings, data_sharding)
            model = model_config.make(sharding_context=make_legacy_sharding_context(mesh))
            beam_codes, beam_scores = model.generate(
                batch, emb_param, beam_width=beam_width, decode_levels=decode_levels
            )
            has_nan = jnp.any(jnp.isnan(beam_scores))

            B = beam_codes.shape[0]
            top_indices = jnp.zeros((B, large_k), dtype=jnp.int32)
            top_scores = jnp.zeros((B, large_k), dtype=jnp.float32)
            return top_indices, top_scores, beam_codes, has_nan

        self.sid_retrieval_forward_fn = sid_retrieval_forward_fn
        self.sid_retrieval_forward_jit = JittedOrCompiled(
            jax.jit(
                sid_retrieval_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=(
                    self.data_sharding,
                    self.data_sharding,
                    self.data_sharding,
                    None,
                ),
            ),
            name="sid_retrieval_forward_fn",
        )
        logger.info("sid_retrieval_forward_jit compiled")

    def _get_persistent_buffer(
        self, request_in_flight_id: int, bucket_size: int | None = None
    ) -> RecsysFeaturesBatch:
        buffer = super()._get_persistent_buffer(request_in_flight_id, bucket_size)
        assert isinstance(self.model_config, RecsysSIDRetrievalConfig)
        bs = bucket_size if bucket_size is not None else self.inference_batch_size
        if buffer["history_seq"].get("post_ids") is None:
            buffer["history_seq"]["post_ids"] = np.zeros((bs, self.history_seq_len), dtype=np.int64)
        if buffer["history_seq"].get("post_sids") is None:
            buffer["history_seq"]["post_sids"] = np.zeros(
                (bs, self.history_seq_len, self.model_config.sid_num_levels),
                dtype=np.uint16,
            )
        return buffer

    def get_batch_prep(
        self,
        persistent_buffer: RecsysFeaturesBatch,
        request_in_flight_id: int,
        bucket_size: int | None = None,
    ) -> xai_recsys_engine.RetrievalBatchPrep:
        assert persistent_buffer["history_seq"]["actions"] is not None
        assert persistent_buffer["history_seq"]["post_ids"] is not None
        return xai_recsys_engine.RetrievalBatchPrep(
            user_hashes=persistent_buffer["user_hashes"],
            user_ip_hashes=persistent_buffer["user_ip_hashes"],
            history_post_hashes=persistent_buffer["history_seq"]["post_hashes"],
            history_auth_hashes=persistent_buffer["history_seq"]["auth_hashes"],
            history_actions=persistent_buffer["history_seq"]["actions"],
            history_continuous_actions=persistent_buffer["history_seq"]["continuous_actions"],
            history_product_surfaces=persistent_buffer["history_seq"]["product_surface"],
            user_categorical_features=persistent_buffer.get("user_categorical_features"),
            user_bool_features=persistent_buffer.get("user_bool_features"),
            user_float_features=persistent_buffer.get("user_float_features"),
            user_int64_features=persistent_buffer.get("user_int64_features"),
            user_installed_apps=persistent_buffer.get("user_installed_apps_multihot"),
            history_categorical_features=persistent_buffer["history_seq"]["categorical_features"],
            history_bool_features=persistent_buffer["history_seq"]["bool_features"],
            history_float_features=persistent_buffer["history_seq"]["float_features"],
            history_int64_features=persistent_buffer["history_seq"]["int64_features"],
            history_post_ids=persistent_buffer["history_seq"]["post_ids"],
            history_post_sids=persistent_buffer["history_seq"].get("post_sids"),
        )

    def gather_embeddings_and_forward(
        self,
        state: RecsysInferenceState,
        batch: RecsysFeaturesBatch,
        rng: jax.Array,
        request_in_flight_id: int,
        batch_id: int,
        request=None,
        eligible_mask=None,
        bucket_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        assert self.sid_retrieval_forward_jit is not None

        h2d_state = self._get_h2d_state(request_in_flight_id, bucket_size)

        with jax_profiler.TraceAnnotation("lookup_h2d_embeddings", batch_id=batch_id):
            t_emb = Timer()
            merged_embeddings, _ = lookup_h2d_embeddings(
                h2d_state,
                batch,
                data_sharding=self.data_sharding,
                batch_id=batch_id,
            )
            logger.info(
                "[batch=%d] Step 2 (embedding_gather+device_put): %.2fms",
                batch_id,
                t_emb.elapsed() * 1000,
            )

        with jax_profiler.TraceAnnotation("forward_jit/sid_retrieval", batch_id=batch_id):
            t_fwd = Timer()
            result = self.sid_retrieval_forward_jit(
                state.params,
                rng,
                batch,
                merged_embeddings,
            )
            logger.info(
                "[batch=%d] Step 4 (forward_jit dispatch, beam_width=%d): %.2fms",
                batch_id,
                self.beam_width,
                t_fwd.elapsed() * 1000,
            )
        return result

    def reply_request(
        self,
        request: xai_recsys_engine.RetrieveRequestBatch,
        output_array: tuple[np.ndarray, np.ndarray, np.ndarray],
        orig_batch_size: int,
        bucket_size: int | None = None,
    ) -> None:
        view = self._corpus
        assert view is not None

        top_indices, top_scores, beam_codes = output_array
        B, large_k = top_indices.shape
        assert large_k == self.large_k, (
            f"top_indices second dim {large_k} != configured large_k {self.large_k}"
        )

        indices_i32 = np.ascontiguousarray(np.asarray(top_indices))
        scores_f32 = np.ascontiguousarray(np.asarray(top_scores))
        beam_codes_i32 = np.ascontiguousarray(np.asarray(beam_codes, dtype=np.int32))

        indices_i32, scores_f32 = self._beam_only_lookup(beam_codes_i32, view)

        if B > 0:
            t_diag = Timer()
            head = min(3, B)
            beam_codes_np = np.asarray(beam_codes)
            preview_top1_codes = beam_codes_np[:head, 0, :].tolist()
            preview_top3_indices = indices_i32[:head, :3].tolist()
            preview_top3_scores = scores_f32[:head, :3].tolist()
            preview_bot1_scores = scores_f32[:head, -1].tolist()

            n_distinct_top1_posts = int(np.unique(indices_i32[:, 0]).size)
            n_distinct_top1_sids = int(np.unique(beam_codes_np[:, 0, :], axis=0).shape[0])

            score_gap_mean = float((scores_f32[:, 0] - scores_f32[:, -1]).mean())

            logger.info(
                "sid_retrieval reply: B=%d K_beam=%d large_k=%d "
                "top1_codes_preview=%s "
                "top3_indices_preview=%s top3_scores_preview=%s "
                "bot1_scores_preview=%s "
                "distinct_top1_posts=%d/%d distinct_top1_sids=%d/%d "
                "score_gap_top1_to_topK=%.4f "
                "(diag=%.2fms)",
                B,
                beam_codes.shape[1],
                large_k,
                preview_top1_codes,
                preview_top3_indices,
                preview_top3_scores,
                preview_bot1_scores,
                n_distinct_top1_posts,
                B,
                n_distinct_top1_sids,
                B,
                score_gap_mean,
                t_diag.elapsed() * 1000,
            )

        assert self.retrieval_dataset_types, "retrieval_dataset_types required"
        dataset_types = [self.retrieval_dataset_types[0].value]
        request.reply(
            dataset_types,
            [indices_i32],
            [scores_f32],
            view.post_ids,
            view.author_ids,
            large_k,
            orig_batch_size,
        )

    def maybe_load_checkpoint(self, ctx, tag=None):
        result = super().maybe_load_checkpoint(ctx, tag)
        loaded = bool(result[0])

        if self._corpus is None or loaded:
            self._corpus = self._build_corpus_view()

        if self.sid_retrieval_forward_jit is None:
            self._setup_sid_retrieval_jit()

        return result

    def _build_corpus_view(self) -> _CorpusView:
        t0 = time.time()
        datasets = list(self.retrieval_dataset_types)
        sid_post_index = SidPostIndex.from_datasets(
            datasets,
            sid_num_levels=self.model_config.sid_num_levels,
            codebook_size=self.model_config.sid_codebook_size,
        )
        prefix_len = self.decode_levels or sid_post_index.sid_num_levels
        resolver = SidBeamResolver.from_sid_post_index(sid_post_index, prefix_len=prefix_len)
        logger.info(
            "SID corpus view built: %d posts, prefix_len=%d, large_k=%d (%.1fs)",
            resolver.num_posts,
            prefix_len,
            self.large_k,
            time.time() - t0,
        )
        return _CorpusView(resolver=resolver)

    def _beam_only_lookup(
        self, beam_codes: np.ndarray, view: _CorpusView
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, scores, _stats = view.resolver.resolve_to_indices(
            beam_codes, max_posts=self.large_k
        )
        return indices, scores

    def run(self) -> None:
        object.__setattr__(self.model_config, "multimodal_embedding_type", None)
        super().run()

    def create_server(
        self,
        mm_client: xai_recsys_engine.PyMmEmbeddingsClient | None = None,
    ) -> xai_recsys_engine.RecsysRetrievalPredictorServer:
        assert isinstance(self.dataset, PhoenixDataset)
        assert isinstance(self.model_config, RecsysSIDRetrievalConfig)
        hash_keys = self.dataset.hash_table.hash_keys

        sid_client = None
        if self.model_config.use_post_sid and self.model_config.sid_num_levels > 0:
            if not self.sid_endpoint:
                raise ValueError(
                    "v4 SID retrieval has use_post_sid=True; "
                    "must pass --sid_endpoint <addr> to launch_inference.py"
                )
            sid_client = xai_recsys_engine.PySemanticIdClient(
                self.sid_endpoint,
                self.model_config.sid_num_levels,
            )
            logger.info(
                "SID client connected: endpoint=%s, sid_num_levels=%d",
                self.sid_endpoint,
                self.model_config.sid_num_levels,
            )

        return xai_recsys_engine.RecsysRetrievalPredictorServer(
            self.grpc_port,
            self.metrics_port,
            self.inference_batch_size,
            self.max_inflight_requests,
            channel_size=self.channel_size,
            enqueue_timeout_ms=self.enqueue_timeout_ms,
            queue_max_staleness_ms=self.queue_max_staleness_ms,
            enable_deadline_admission=self.enable_deadline_admission,
            deadline_margin_ms=self.deadline_margin_ms,
            deadline_post_ms=self.deadline_post_ms,
            deadline_fallback_budget_ms=self.deadline_fallback_budget_ms,
            service_time_ewma_alpha=self.service_time_ewma_alpha,
            pipeline_depth=1 if self.use_pipelining else 0,
            mm_client=mm_client,
            sid_client=sid_client,
            user_id_table_size=hash_keys.user_id_table_size,
            user_hash_scales=hash_keys.user_hash_scales,
            user_biases=hash_keys.user_biases,
            user_modulus=hash_keys.user_modulus,
            item_id_table_size=hash_keys.item_id_table_size,
            item_hash_vocab_size=hash_keys.item_hash_vocab_size,
            item_hash_scales=hash_keys.item_hash_scales,
            item_biases=hash_keys.item_biases,
            item_modulus=hash_keys.item_modulus,
            author_id_table_size=hash_keys.author_id_table_size,
            author_hash_scales=hash_keys.author_hash_scales,
            author_biases=hash_keys.author_biases,
            author_modulus=hash_keys.author_modulus,
            output_vocab_size=self.dataset.hash_table.output_vocab_size,
            num_continuous_actions=self.model_config.num_continuous_actions,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            multimodal_embedding_dim=0,
            ip_id_table_size=hash_keys.ip_id_table_size,
            ip_hash_scales=hash_keys.ip_hash_scales,
            ip_biases=hash_keys.ip_biases,
            ip_modulus=hash_keys.ip_modulus,
            num_user_categorical_features=recsys_batch.USER_CATEGORICAL_FEATURE_SIZE,
            num_user_bool_features=recsys_batch.USER_BOOL_FEATURE_SIZE,
            num_user_float_features=recsys_batch.USER_FLOAT_FEATURE_SIZE,
            num_user_int64_features=recsys_batch.USER_INT64_FEATURE_SIZE,
            num_user_installed_apps=recsys_batch.NUM_USER_INSTALLED_APPS,
            num_post_categorical_features=recsys_batch.POST_CATEGORICAL_FEATURE_SIZE,
            num_post_bool_features=recsys_batch.POST_BOOL_FEATURE_SIZE,
            num_post_float_features=recsys_batch.POST_FLOAT_FEATURE_SIZE,
            num_post_int64_features=recsys_batch.POST_INT64_FEATURE_SIZE,
            enable_async_response_compression=self.enable_async_response_compression,
            sid_num_levels=(self.model_config.sid_num_levels if sid_client is not None else 0),
        )
