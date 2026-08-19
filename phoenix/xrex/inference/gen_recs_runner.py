# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import xai_recsys_engine
from jax.sharding import PartitionSpec as P

from xrex.data.parquet_recsys import PhoenixDataset
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.inference.h2d import EmbeddingSlices, lookup_h2d_embeddings
from xrex.inference.model_runner import BaseModelRunner
from xrex.models.recsys_embedding import RecsysEmbeddings
from xrex.models.recsys_gen_recs_model import RecsysGenRecsModelConfig
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.train.misc import RecsysInferenceState
from xrex.utils.aot import JittedOrCompiled

logger = logging.getLogger(__name__)

from xai_configlib import configclass


@configclass
class GenRecsModelRunner(
    BaseModelRunner[
        xai_recsys_engine.RetrieveRequestBatch,
        RecsysGenRecsModelConfig,
    ]
):
    def init(self, batch, rng: jax.Array) -> RecsysInferenceState:
        state = super().init(batch, rng)
        return state._replace(
            post_embeddings=self._family_init_post_embeddings(state.post_embeddings)
        )

    def _family_init_post_embeddings(self, post_embeddings):
        if isinstance(self.model_config, RecsysGenRecsModelConfig):
            post_embeddings = self.model_config.make_post_embeddings()
        return post_embeddings

    gen_recs_forward_jit: JittedOrCompiled | None = None
    all_post_ids: npt.NDArray[np.int64] | None = None
    all_author_ids: npt.NDArray[np.int64] | None = None
    table_embeddings: np.ndarray | None = None
    table_embeddings_jax: jax.Array | None = None
    feedback_embeddings_jax: jax.Array | None = None

    num_generate: int = 1
    top_k_per_position: int = 10

    def _setup_gen_recs_jit(self) -> None:
        assert isinstance(self.model_config, RecsysGenRecsModelConfig)

        logger.info(
            f"Setting up gen_recs JIT with num_generate={self.num_generate}, "
            f"snap_to_real={self.table_embeddings_jax is not None}"
        )

        _ht = self.model_config.hash_table.hash_keys
        _hist_post_seq = _ht.num_item_hashes * self.history_seq_len
        _hist_auth_seq = _ht.num_author_hashes * self.history_seq_len
        _cand_post_seq = _ht.num_item_hashes * self.candidate_seq_len
        _cand_auth_seq = _ht.num_author_hashes * self.candidate_seq_len
        _user_seq = _ht.num_user_hashes
        _user_ip_seq = getattr(_ht, "num_ip_hashes", 0)
        _total = (
            _hist_post_seq
            + _hist_auth_seq
            + _cand_post_seq
            + _cand_auth_seq
            + _user_seq
            + _user_ip_seq
        )
        embedding_slices = EmbeddingSlices(
            hist_post_end=_hist_post_seq,
            hist_auth_end=_hist_post_seq + _hist_auth_seq,
            cand_post_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq,
            cand_auth_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq,
            user_end=_hist_post_seq + _hist_auth_seq + _cand_post_seq + _cand_auth_seq + _user_seq,
            user_ip_end=_total,
        )

        model_config = self.model_config
        mesh = self.mesh

        top_k_per_position = self.top_k_per_position

        @hk.transform
        def gen_recs_forward_fn(
            batch: RecsysFeaturesBatch,
            merged_embeddings: jax.Array,
            num_generate: int = 1,
            embedding_table: jax.Array | None = None,
            feedback_table: jax.Array | None = None,
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
                user_embeddings=merged_embeddings[:, sl.cand_auth_end :, :],
            )
            model = model_config.make(sharding_context=make_legacy_sharding_context(mesh))
            generated = model.generate(
                batch,
                recsys_embeddings,
                num_generate,
                embedding_table,
                feedback_table,
            )

            B, N, D = generated.shape
            flat_gen = generated.reshape(B * N, D)
            preds_norm = flat_gen / jnp.sqrt(
                jnp.sum(jnp.square(flat_gen), axis=-1, keepdims=True) + 1e-8
            )
            scores = jnp.dot(preds_norm, embedding_table.T)
            scores = scores.reshape(B, N, -1)
            top_k_scores, top_k_indices = jax.lax.top_k(scores, top_k_per_position)
            has_nan = jnp.any(jnp.isnan(scores))
            return top_k_indices, top_k_scores, generated, has_nan

        self.gen_recs_forward_fn = gen_recs_forward_fn

        self.gen_recs_forward_jit = JittedOrCompiled(
            jax.jit(
                gen_recs_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                    None,
                    None,
                ),
                out_shardings=(
                    self.data_sharding,
                    self.data_sharding,
                    self.data_sharding,
                    None,
                ),
                static_argnums=(4,),
            ),
            name="gen_recs_forward_fn",
        )
        logger.info("gen_recs_forward_jit compiled")

    def get_batch_prep(
        self,
        persistent_buffer: RecsysFeaturesBatch,
        request_in_flight_id: int,
        bucket_size: int | None = None,
    ) -> xai_recsys_engine.RetrievalBatchPrep:
        assert persistent_buffer["history_seq"]["actions"] is not None
        history_post_ids = persistent_buffer["history_seq"].get("post_ids")
        if history_post_ids is None:
            history_post_ids = np.zeros(
                (self.inference_batch_size, self.history_seq_len), dtype=np.int64
            )
        return xai_recsys_engine.RetrievalBatchPrep(
            user_hashes=persistent_buffer["user_hashes"],
            user_ip_hashes=persistent_buffer["user_ip_hashes"],
            history_post_hashes=persistent_buffer["history_seq"]["post_hashes"],
            history_auth_hashes=persistent_buffer["history_seq"]["auth_hashes"],
            history_actions=persistent_buffer["history_seq"]["actions"],
            history_continuous_actions=persistent_buffer["history_seq"]["continuous_actions"],
            history_product_surfaces=persistent_buffer["history_seq"]["product_surface"],
            history_post_ids=history_post_ids,
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
        assert self.gen_recs_forward_jit is not None
        assert self.table_embeddings_jax is not None

        h2d_state = self._get_h2d_state(request_in_flight_id, bucket_size)
        merged_embeddings, _ = lookup_h2d_embeddings(
            h2d_state, batch, data_sharding=self.data_sharding, batch_id=batch_id
        )

        return self.gen_recs_forward_jit(
            state.params,
            rng,
            batch,
            merged_embeddings,
            self.num_generate,
            self.table_embeddings_jax,
            self.feedback_embeddings_jax,
        )

    def reply_request(
        self,
        request: xai_recsys_engine.RetrieveRequestBatch,
        output_array: tuple[np.ndarray, np.ndarray, np.ndarray],
        orig_batch_size: int,
        bucket_size: int = 0,
    ) -> None:
        assert self.all_post_ids is not None and self.all_author_ids is not None
        top_k_indices, top_k_scores, generated = output_array

        B, N, K = top_k_indices.shape
        total_k = N * K

        flat_indices = np.ascontiguousarray(top_k_indices.reshape(B, total_k).astype(np.int32))
        flat_scores = np.ascontiguousarray(top_k_scores.reshape(B, total_k).astype(np.float32))

        request.reply(
            [0],
            [flat_indices],
            [flat_scores],
            self.all_post_ids,
            self.all_author_ids,
            total_k,
            orig_batch_size,
        )

    def maybe_load_checkpoint(self, ctx, tag=None):
        result = super().maybe_load_checkpoint(ctx, tag)

        has_checkpoint_embeddings = (
            hasattr(self, "state")
            and self.state is not None
            and self.state.post_embeddings is not None
            and self.state.post_embeddings.embeddings is not None
            and self.state.post_embeddings.embeddings.x is not None
            and self.state.post_embeddings.embeddings.x.shape[0] > 1
            and self.state.post_embeddings.post_ids is not None
        )

        if has_checkpoint_embeddings:
            logger.info("Using post embeddings from checkpoint (two-tower pattern)")

            self.all_post_ids = self.two_int32_to_int64(self.state.post_embeddings.post_ids)
            self.all_author_ids = self.two_int32_to_int64(self.state.post_embeddings.author_ids)

            num_valid = int(np.sum(self.all_post_ids != 0))
            logger.info(f"Loaded {num_valid} posts from checkpoint")

            ckpt_embs_np = np.array(self.state.post_embeddings.embeddings.x, dtype=np.float32)

            replicated = jax.sharding.NamedSharding(self.mesh, P())
            self.feedback_embeddings_jax = jax.device_put(
                jnp.array(ckpt_embs_np, dtype=jnp.float32),
                replicated,
            )

            norms = np.linalg.norm(ckpt_embs_np, axis=-1, keepdims=True)
            table_embs = ckpt_embs_np / np.sqrt(
                np.sum(np.square(ckpt_embs_np), axis=-1, keepdims=True) + 1e-8
            )
            logger.info(
                f"Checkpoint embedding norms: mean={norms.mean():.3f} std={norms.std():.3f}"
            )

            self.table_embeddings_jax = jax.device_put(
                jnp.array(table_embs, dtype=jnp.bfloat16),
                replicated,
            )
            logger.info(
                f"Scoring table from checkpoint: {self.table_embeddings_jax.shape} "
                f"({num_valid} valid posts)"
            )
            self.table_embeddings = table_embs
        else:
            raise RuntimeError(
                "Checkpoint missing post_embeddings; "
                "gen-recs inference requires checkpoint post embeddings"
            )

        if self.gen_recs_forward_jit is None:
            self._setup_gen_recs_jit()

        return result

    def _start_mm_embeddings_preload(self) -> None:
        return

    def run(self) -> None:
        object.__setattr__(self.model_config, "multimodal_embedding_type", None)
        super().run()

    def create_server(
        self,
        mm_client: xai_recsys_engine.PyMmEmbeddingsClient | None = None,
    ) -> xai_recsys_engine.RecsysRetrievalPredictorServer:
        assert isinstance(self.dataset, PhoenixDataset)
        assert isinstance(self.model_config, RecsysGenRecsModelConfig)
        hash_keys = self.dataset.hash_table.hash_keys
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
            multimodal_embedding_dim=self.model_config.mm_target_dim,
        )
