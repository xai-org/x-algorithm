# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec as P
from xai_configlib import configclass

from xrex.data.recsys.gen_recs_semantics import (
    build_global_negative_target_embeddings,
    build_target_embeddings_from_batch,
    l2_normalize,
)
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.layers import get_parameter
from xrex.models.recsys_embedding import (
    RecsysEmbeddings,
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_model import (
    DTYPE_BY_NAME,
    RecsysAggregatedModel,
    RecsysAggregatedModelConfig,
    block_candidate_reduce,
    block_history_reduce,
    block_user_reduce,
    cast_jax,
)
from xrex.models.sharding_context import ShardingContext
from xrex.utils.utils import Summary

logger = logging.getLogger(__name__)

EPS = 1e-8
INF = 1e12


def autoregressive_generate(
    model: RecsysGenRecsModel,
    input_embeddings: jax.Array,
    padding_mask: jax.Array,
    num_generate: int,
    mm_target_dim: int,
    embedding_table: jax.Array | None = None,
    feedback_table: jax.Array | None = None,
) -> jax.Array:
    return model._autoregressive_loop(
        input_embeddings,
        padding_mask,
        num_generate,
        mm_target_dim,
        embedding_table=embedding_table,
        feedback_table=feedback_table,
    )


@configclass
class RecsysGenRecsModelConfig(RecsysAggregatedModelConfig):
    product_surface_vocab_size: int = 16

    mm_target_dim: int = 1024

    cosine_loss_temperature: float = 0.05

    candidate_embedding_mode: Literal["mm_only", "post_combined"] = "mm_only"

    max_posts: int = 2_097_152

    num_global_negatives: int = 0

    log_q_correction: bool = False

    eval_post_emb_path: str | None = None

    def make_post_embeddings(self):
        from xrex.data.retrieval_dataset import RetrievalDataset
        from xrex.models.model_utils import Parameter
        from xrex.train.misc import PostEmbeddings

        if self.candidate_embedding_mode == "post_combined":
            emb_dim = self.emb_table_width
        else:
            emb_dim = self.mm_target_dim

        M = self.max_posts
        return PostEmbeddings(
            post_ids=jnp.arange(M * 2, dtype=jnp.int32).reshape(-1, 2),
            author_ids=jnp.arange(M * 2, dtype=jnp.int32).reshape(-1, 2),
            embeddings=Parameter(
                x=jnp.empty((M, emb_dim), dtype=jnp.bfloat16),
                pspec=P(("expert", "replica"), ("seq", "model")),
            ),
            dataset_types=jnp.full((M, 1), RetrievalDataset.PAD.value, dtype=jnp.int32),
        )

    def make(self, sharding_context: ShardingContext) -> RecsysGenRecsModel:
        return RecsysGenRecsModel(
            model=self.model_config.make(sharding_context=sharding_context),
            config=self,
            sharding_context=sharding_context,
        )


@dataclass
class RecsysGenRecsModel(RecsysAggregatedModel):
    config: RecsysGenRecsModelConfig

    @hk.transparent
    def _project_to_target_space(self, hidden: jax.Array) -> jax.Array:
        cfg = self.config
        if cfg.candidate_embedding_mode == "post_combined":
            return self._project_to_post_space(hidden)
        return self._project_to_mm_space(hidden)

    @hk.transparent
    def _project_to_mm_space(self, hidden: jax.Array) -> jax.Array:
        cfg = self.config
        emb_size = cfg.model_config.emb_size
        mm_dim = cfg.mm_target_dim
        embed_init = hk.initializers.VarianceScaling(cfg.embed_init_scale, mode="fan_out")

        proj = get_parameter(
            "gen_recs_out_proj",
            [emb_size, mm_dim],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=cfg.model_config.scale_config.emb_lr_multiplier(emb_size),
        )
        out = jnp.dot(hidden.astype(proj.dtype), proj)

        out = out / jnp.sqrt(jnp.sum(jnp.square(out), axis=-1, keepdims=True) + EPS)
        return out.astype(DTYPE_BY_NAME[self.config.fprop_dtype])

    @hk.transparent
    def _project_to_post_space(self, hidden: jax.Array) -> jax.Array:
        cfg = self.config
        emb_size = cfg.model_config.emb_size
        target_dim = cfg.emb_table_width
        embed_init = hk.initializers.VarianceScaling(cfg.embed_init_scale, mode="fan_out")

        proj = get_parameter(
            "gen_recs_post_out_proj",
            [emb_size, target_dim],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=cfg.model_config.scale_config.emb_lr_multiplier(emb_size),
        )
        out = jnp.dot(hidden.astype(proj.dtype), proj)

        out = out / jnp.sqrt(jnp.sum(jnp.square(out), axis=-1, keepdims=True) + EPS)
        return out.astype(DTYPE_BY_NAME[cfg.fprop_dtype])

    @hk.transparent
    def _project_mm_to_candidate_space(self, mm_embeddings: jax.Array) -> jax.Array:
        cfg = self.config
        mm_dim = cfg.mm_target_dim
        D = cfg.emb_table_width
        embed_init = hk.initializers.VarianceScaling(cfg.embed_init_scale, mode="fan_out")

        proj = get_parameter(
            "gen_recs_mm_input_proj",
            [mm_dim, D],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=cfg.model_config.scale_config.emb_lr_multiplier(mm_dim),
        )
        out = jnp.dot(mm_embeddings.astype(proj.dtype), proj)
        return out.astype(DTYPE_BY_NAME[cfg.fprop_dtype])

    @hk.transparent
    def build_history_only(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        *,
        apply_tfmr_projection: bool = True,
        apply_input_scale: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config
        assert cfg.model_config.output_vocab_size is not None

        history_actions = batch["history_seq"]["actions"]
        assert history_actions is not None
        history_actions_embeddings, _ = self.multi_hot_to_embeddings(
            cast_jax(history_actions),
            cfg.model_config.output_vocab_size,
            cfg.emb_table_width,
            cfg.embed_init_scale,
            "action_embedding_table",
        )

        history_product_surface = batch["history_seq"]["product_surface"]
        history_ps_embeddings, _ = self.single_hot_to_embeddings(
            cast_jax(history_product_surface),
            cfg.product_surface_vocab_size,
            cfg.emb_table_width,
            cfg.embed_init_scale,
            "product_surface_embedding_table",
        )

        user_embeddings, user_padding_mask = block_user_reduce(
            cast_jax(batch["user_hashes"]),
            recsys_embeddings.user_embeddings,
            cfg.hash_table.hash_keys,
            cfg.model_config.scale_config.emb_lr_multiplier,
            cfg.embed_init_scale,
        )

        history_embeddings, history_padding_mask = block_history_reduce(
            cast_jax(batch["history_seq"]["post_hashes"]),
            cast_jax(batch["history_seq"]["auth_hashes"]),
            recsys_embeddings.history_post_embeddings,
            recsys_embeddings.history_author_embeddings,
            history_ps_embeddings,
            history_actions_embeddings,
            cfg.hash_table.hash_keys,
            cfg.model_config.scale_config.emb_lr_multiplier,
            cfg.embed_init_scale,
            cfg.use_product_surface,
            cast_jax(batch["history_seq"]["continuous_actions"]),
            DTYPE_BY_NAME[cfg.fprop_dtype],
            cfg.continuous_action_losses,
            cfg.continuous_action_hidden_dim,
        )

        embeddings = jnp.concatenate([user_embeddings, history_embeddings], axis=1)
        if apply_input_scale:
            embeddings *= cfg.model_config.scale_config.input_scale(cfg.emb_table_width)
        embeddings = with_sharding_constraint(embeddings, P(self.data_axis, ("seq", "model")))

        padding_mask = jnp.concatenate([user_padding_mask, history_padding_mask], axis=1)
        if apply_tfmr_projection:
            embeddings = self.maybe_tfmr_project_embeddings(embeddings, cfg)

        return embeddings, padding_mask

    @hk.transparent
    def _autoregressive_loop(
        self,
        input_embeddings: jax.Array,
        padding_mask: jax.Array,
        num_generate: int,
        mm_target_dim: int,
        embedding_table: jax.Array | None = None,
        feedback_table: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config

        B = input_embeddings.shape[0]
        H = input_embeddings.shape[1]
        D = input_embeddings.shape[2]
        max_len = H + num_generate

        full_embeddings = jnp.zeros((B, max_len, D), dtype=input_embeddings.dtype)
        full_embeddings = full_embeddings.at[:, :H, :].set(input_embeddings)
        full_mask = jnp.zeros((B, max_len), dtype=padding_mask.dtype)
        full_mask = full_mask.at[:, :H].set(padding_mask)
        generated = jnp.zeros((B, num_generate, mm_target_dim), dtype=input_embeddings.dtype)

        input_scale = cfg.model_config.scale_config.input_scale(cfg.emb_table_width)
        assert embedding_table is not None, (
            "embedding_table is required for autoregressive generation"
        )

        def _feedback_token_to_transformer_input(next_emb_3d: jax.Array) -> jax.Array:
            candidate_input = next_emb_3d * input_scale

            if cfg.model_config.emb_size == cfg.emb_table_width:
                return candidate_input

            w_init = hk.initializers.VarianceScaling(
                cfg.model_config.scale_config.attn_init_scale**2
            )
            lr_multiplier = cfg.model_config.scale_config.hidden_lr_multiplier(cfg.emb_table_width)
            with hk.name_scope("FORCE:tfmr_proj"):
                tfmr_proj_w = get_parameter(
                    "w",
                    [cfg.emb_table_width, cfg.model_config.emb_size],
                    dtype=jnp.float32,
                    init=w_init,
                    pspec=P(None, None),
                    lr_multiplier=lr_multiplier,
                )

            out = jnp.dot(candidate_input.astype(tfmr_proj_w.dtype), tfmr_proj_w)
            return out.astype(DTYPE_BY_NAME[cfg.fprop_dtype])

        M = embedding_table.shape[0]
        used_mask = jnp.zeros((B, M), dtype=jnp.bool_)

        def body_fn(i, carry):
            full_emb, full_msk, gen_buf, used = carry

            predicted = self.forward_gen_recs(full_emb, full_msk, is_training=False)
            pos = H - 1 + i
            last_pred = predicted[:, pos, :]

            gen_buf = gen_buf.at[:, i, :].set(last_pred)

            preds_norm = l2_normalize(last_pred.astype(jnp.float32))
            scores = jnp.dot(preds_norm, embedding_table.T)

            scores = jnp.where(used, -INF, scores)

            top1_idx = jnp.argmax(scores, axis=-1)

            used = used.at[jnp.arange(B), top1_idx].set(True)

            fb_table = feedback_table if feedback_table is not None else embedding_table
            next_emb = fb_table[top1_idx]
            next_emb_3d = next_emb[:, None, :].astype(last_pred.dtype)

            candidate_input = _feedback_token_to_transformer_input(next_emb_3d)

            write_pos = H + i
            full_emb = jax.lax.dynamic_update_slice(full_emb, candidate_input, (0, write_pos, 0))
            full_msk = full_msk.at[:, write_pos].set(1.0)

            return (full_emb, full_msk, gen_buf, used)

        _, _, generated, _ = jax.lax.fori_loop(
            0,
            num_generate,
            body_fn,
            (full_embeddings, full_mask, generated, used_mask),
        )

        return generated

    @hk.transparent
    def generate(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        num_generate: int,
        embedding_table: jax.Array | None = None,
        feedback_table: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config
        target_dim = cfg.emb_table_width
        history_embeddings, history_mask = self.build_history_only(batch, recsys_embeddings)
        return autoregressive_generate(
            self,
            history_embeddings,
            history_mask,
            num_generate,
            target_dim,
            embedding_table=embedding_table,
            feedback_table=feedback_table,
        )

    @hk.transparent
    def project_candidates(
        self,
        post_hashes: jax.Array,
        author_hashes: jax.Array,
        post_hash_embs: jax.Array,
        author_hash_embs: jax.Array,
        ps_ids: jax.Array,
        mm_embs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config
        ps_embs, _ = self.single_hot_to_embeddings(
            ps_ids,
            cfg.product_surface_vocab_size,
            cfg.emb_table_width,
            cfg.embed_init_scale,
            "product_surface_embedding_table",
        )
        return block_candidate_reduce(
            post_hashes,
            author_hashes,
            post_hash_embs,
            author_hash_embs,
            ps_embs,
            cfg.hash_table.hash_keys,
            cfg.model_config.scale_config.emb_lr_multiplier,
            cfg.embed_init_scale,
            cfg.use_product_surface,
            multimodal_embeddings=mm_embs,
        )

    @hk.transparent
    def build_gen_recs_inputs(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> tuple[jax.Array, jax.Array, int, jax.Array, jax.Array]:
        cfg = self.config
        assert cfg.model_config.output_vocab_size is not None

        history_actions = batch["history_seq"]["actions"]
        assert history_actions is not None
        history_actions_embeddings, _ = self.multi_hot_to_embeddings(
            cast_jax(history_actions),
            cfg.model_config.output_vocab_size,
            cfg.emb_table_width,
            cfg.embed_init_scale,
            "action_embedding_table",
        )

        history_product_surface = batch["history_seq"]["product_surface"]
        history_ps_embeddings, _ = self.single_hot_to_embeddings(
            cast_jax(history_product_surface),
            cfg.product_surface_vocab_size,
            cfg.emb_table_width,
            cfg.embed_init_scale,
            "product_surface_embedding_table",
        )

        user_embeddings, user_padding_mask = block_user_reduce(
            cast_jax(batch["user_hashes"]),
            recsys_embeddings.user_embeddings,
            cfg.hash_table.hash_keys,
            cfg.model_config.scale_config.emb_lr_multiplier,
            cfg.embed_init_scale,
        )

        history_embeddings, history_padding_mask = block_history_reduce(
            cast_jax(batch["history_seq"]["post_hashes"]),
            cast_jax(batch["history_seq"]["auth_hashes"]),
            recsys_embeddings.history_post_embeddings,
            recsys_embeddings.history_author_embeddings,
            history_ps_embeddings,
            history_actions_embeddings,
            cfg.hash_table.hash_keys,
            cfg.model_config.scale_config.emb_lr_multiplier,
            cfg.embed_init_scale,
            cfg.use_product_surface,
            cast_jax(batch["history_seq"]["continuous_actions"]),
            DTYPE_BY_NAME[cfg.fprop_dtype],
            cfg.continuous_action_losses,
            cfg.continuous_action_hidden_dim,
        )

        mm_raw = batch["candidate_seq"].get("embedding")
        assert mm_raw is not None, "Gen recs model requires multimodal embeddings in candidate_seq"

        def _product_surface_embedding_fn(ps_ids: jax.Array) -> jax.Array:
            return self.single_hot_to_embeddings(
                ps_ids,
                cfg.product_surface_vocab_size,
                cfg.emb_table_width,
                cfg.embed_init_scale,
                "product_surface_embedding_table",
            )[0]

        target_info = build_target_embeddings_from_batch(
            cfg=cfg,
            batch=batch,
            recsys_embeddings=recsys_embeddings,
            mm_to_candidate_space_fn=self._project_mm_to_candidate_space,
            product_surface_embedding_fn=_product_surface_embedding_fn,
        )
        target_embeddings = target_info.embeddings
        candidate_padding_mask = target_info.valid_mask.astype(jnp.bool_)
        candidate_embeddings = target_embeddings

        embeddings = jnp.concatenate(
            [user_embeddings, history_embeddings, candidate_embeddings], axis=1
        )
        embeddings *= cfg.model_config.scale_config.input_scale(cfg.emb_table_width)
        embeddings = with_sharding_constraint(embeddings, P(self.data_axis, ("seq", "model")))

        padding_mask = jnp.concatenate(
            [user_padding_mask, history_padding_mask, candidate_padding_mask], axis=1
        )
        candidate_start_offset = user_padding_mask.shape[1] + history_padding_mask.shape[1]

        embeddings = self.maybe_tfmr_project_embeddings(embeddings, cfg)

        mm_valid_mask = target_info.valid_mask.astype(jnp.float32)

        return embeddings, padding_mask, candidate_start_offset, target_embeddings, mm_valid_mask

    def forward_gen_recs(
        self,
        input_embeddings: jax.Array,
        padding_mask: jax.Array,
        *,
        is_training: bool = True,
    ) -> jax.Array:
        from xrex.models.normalization import rms_norm_fn

        cfg = self.config
        B, T, _ = input_embeddings.shape

        segment_ids = jnp.zeros((B, T), dtype=jnp.int32)

        positions = jnp.full((B, T, 3), 0, dtype=jnp.float32)
        positions = positions.at[:, :, 0].set(jnp.arange(T)[None, :])

        attn_mask = jnp.ones((B, T), dtype=jnp.int32)

        model_out = self.model(
            input_embeddings,
            attn_mask,
            segment_ids=segment_ids,
            is_training=is_training,
            decoding=False,
            padding_mask=padding_mask,
            positions=positions,
        )

        out = model_out.output
        out = with_sharding_constraint(out, P(self.data_axis, ("seq", "model")))

        out = rms_norm_fn(
            out,
            name="final_layer_norm",
            weight_decay_mask=cfg.model_config.scale_config.ln_weight_decay_mask,
        )

        predicted_embeddings = self._project_to_target_space(out)
        return predicted_embeddings

    @hk.transparent
    def loss(
        self,
        inter,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        is_training: bool = True,
        **kwargs,
    ):
        del inter

        input_embeddings, padding_mask, cand_offset, target_embeddings, mm_valid_mask = (
            self.build_gen_recs_inputs(
                batch,
                get_recsys_embed_param_to_jax_array(recsys_embeddings),
            )
        )

        B, T, _ = input_embeddings.shape
        C = target_embeddings.shape[1]

        with Summary() as summarizer:
            predicted_embeddings = self.forward_gen_recs(
                input_embeddings, padding_mask, is_training=is_training
            )

        pred_start = cand_offset - 1
        pred_end = cand_offset + C - 1
        preds = predicted_embeddings[:, pred_start:pred_end, :]

        targets_norm = l2_normalize(target_embeddings).astype(preds.dtype)

        pred_padding = padding_mask[:, pred_start:pred_end]
        loss_mask = pred_padding * mm_valid_mask

        global_neg_targets = None
        global_neg_mask = None
        cfg_G = self.config.num_global_negatives
        if cfg_G > 0:
            mm_raw_all = batch["candidate_seq"].get("embedding")
            mm_all_jax = (
                cast_jax(mm_raw_all) if not isinstance(mm_raw_all, jax.Array) else mm_raw_all
            )
            mm_global = mm_all_jax[:, C:, :]
            gn_valid = (jnp.sum(jnp.abs(mm_global), axis=-1) > 0).astype(jnp.float32)

            def _product_surface_embedding_fn(ps_ids: jax.Array) -> jax.Array:
                return self.single_hot_to_embeddings(
                    ps_ids,
                    self.config.product_surface_vocab_size,
                    self.config.emb_table_width,
                    self.config.embed_init_scale,
                    "product_surface_embedding_table",
                )[0]

            global_neg_targets, global_neg_mask = build_global_negative_target_embeddings(
                cfg=self.config,
                batch=batch,
                recsys_embeddings=get_recsys_embed_param_to_jax_array(recsys_embeddings),
                mm_global=mm_global,
                gn_valid=gn_valid,
                mm_to_candidate_space_fn=self._project_mm_to_candidate_space,
                product_surface_embedding_fn=_product_surface_embedding_fn,
            )
            global_neg_targets = global_neg_targets.astype(preds.dtype)

        log_q_weights = None
        if self.config.log_q_correction:
            log_q_weights = _compute_log_q_weights(batch, C, cfg_G, self.config.log_q_num_bins)

        pred_loss_mask = None
        npc_raw = batch.get("num_positive_candidates")
        if npc_raw is not None:
            npc = cast_jax(npc_raw) if not isinstance(npc_raw, jax.Array) else npc_raw
            col_indices = jnp.arange(C)[None, :]
            positive_mask = (col_indices < npc).astype(jnp.float32)
            pred_loss_mask = loss_mask * positive_mask

        temperature = self.config.cosine_loss_temperature

        infonce_loss, infonce_stats = _compute_infonce_loss(
            preds,
            targets_norm,
            loss_mask,
            temperature,
            data_axis=self.data_axis,
            mesh=self.sharding_context.mesh,
            global_neg_targets=global_neg_targets,
            global_neg_mask=global_neg_mask,
            log_q_weights=log_q_weights,
            pred_loss_mask=pred_loss_mask,
        )

        loss = infonce_loss

        cosine_sim = jnp.sum(preds * targets_norm, axis=-1)
        num_valid = jnp.maximum(jnp.sum(loss_mask), 1.0)

        stats = summarizer.get()
        stats["gen-recs-infonce-loss"] = infonce_loss
        stats["gen-recs-mean-cosine-sim"] = jnp.sum(cosine_sim * loss_mask) / num_valid
        stats["gen-recs-num-valid-tokens"] = jnp.sum(loss_mask)
        stats["gen-recs-num-candidates"] = jnp.float32(C)
        stats["gen-recs-temperature"] = jnp.float32(temperature)
        if pred_loss_mask is not None:
            num_positives = jnp.sum(pred_loss_mask)
            stats["gen-recs-num-positive-candidates"] = num_positives
            stats["gen-recs-num-impression-negatives"] = jnp.sum(loss_mask) - num_positives
        stats.update(infonce_stats)

        cosine_loss_per_token = 1.0 - cosine_sim
        stats["gen-recs-cosine-loss"] = jnp.sum(cosine_loss_per_token * loss_mask) / num_valid

        l2_dist = jnp.sqrt(jnp.sum((preds - targets_norm) ** 2, axis=-1) + EPS)
        stats["gen-recs-mean-l2-dist"] = jnp.sum(l2_dist * loss_mask) / num_valid

        eval_mask = pred_loss_mask if pred_loss_mask is not None else loss_mask
        stats.update(
            _compute_batch_topk_accuracy(
                preds,
                targets_norm,
                eval_mask,
                data_axis=self.data_axis,
                mesh=self.sharding_context.mesh,
            )
        )

        history_impr_ts = batch["history_seq"]["impr_ts"]
        candidate_impr_ts = batch["candidate_seq"]["impr_ts"]
        if history_impr_ts is not None:
            stats["max_ts_history"] = jnp.max(cast_jax(history_impr_ts))
        if candidate_impr_ts is not None:
            stats["max_ts_candidates"] = jnp.max(cast_jax(candidate_impr_ts))

        stats["gen-recs-pred-norm-mean"] = jnp.mean(jnp.linalg.norm(preds, axis=-1) * loss_mask)
        stats["gen-recs-target-norm-mean"] = jnp.mean(
            jnp.linalg.norm(targets_norm, axis=-1) * mm_valid_mask
        )

        stats.update(
            _compute_per_position_accuracy(
                preds,
                targets_norm,
                eval_mask,
                data_axis=self.data_axis,
                mesh=self.sharding_context.mesh,
                num_positions=min(10, C),
            )
        )

        return loss, stats


def _compute_per_position_accuracy(
    preds: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
    data_axis: tuple,
    mesh: jax.sharding.Mesh,
    num_positions: int = 10,
) -> dict[str, jax.Array]:
    @shard_map(
        mesh=mesh,
        in_specs=(P(data_axis), P(data_axis), P(data_axis)),
        out_specs=P(),
        check_vma=False,
    )
    def _sharded(local_preds, local_targets, local_mask):
        b, C, D = local_preds.shape
        N = b * C

        flat_targets = local_targets.reshape(N, D).astype(jnp.float32)
        flat_mask = local_mask.reshape(N)

        results = []
        for pos_idx in range(num_positions):
            pos_preds = local_preds[:, pos_idx, :].astype(jnp.float32)
            pos_mask = local_mask[:, pos_idx]

            sim = jnp.dot(pos_preds, flat_targets.T)
            sim = jnp.where(flat_mask[None, :] > 0, sim, -INF)

            correct_idx = jnp.arange(b) * C + pos_idx
            predicted_idx = jnp.argmax(sim, axis=-1)
            top1 = (predicted_idx == correct_idx).astype(jnp.float32)

            pos_valid = jnp.sum(pos_mask)
            top1_hits = jnp.sum(top1 * pos_mask)
            results.extend([top1_hits, pos_valid])

        return jax.lax.psum(jnp.stack(results), axis_name=data_axis)

    raw = _sharded(preds, targets, loss_mask)

    stats = {}
    for pos_idx in range(num_positions):
        hits = raw[pos_idx * 2]
        valid = jnp.maximum(raw[pos_idx * 2 + 1], 1.0)
        stats[f"gen-recs-pos{pos_idx}-top1-acc"] = hits / valid
        stats[f"gen-recs-pos{pos_idx}-valid-tokens"] = raw[pos_idx * 2 + 1]

    return stats


def _compute_log_q_weights(
    batch: RecsysFeaturesBatch,
    C: int,
    num_global_negatives: int,
    log_q_num_bins: int,
) -> jax.Array:
    C_total = C + num_global_negatives
    hashed_ids = cast_jax(batch["candidate_seq"]["post_hashes"])[:, :C_total, 0]
    hashed_ids = hashed_ids % log_q_num_bins
    flat_ids = hashed_ids.reshape(-1)

    counts = jnp.bincount(flat_ids, length=log_q_num_bins, minlength=log_q_num_bins)
    per_target_counts = counts[flat_ids].reshape(hashed_ids.shape)

    return jnp.log(jnp.maximum(per_target_counts.astype(jnp.float32), 1.0))


def _compute_infonce_loss(
    preds: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
    temperature: float,
    data_axis: tuple,
    mesh: jax.sharding.Mesh,
    global_neg_targets: jax.Array | None = None,
    global_neg_mask: jax.Array | None = None,
    log_q_weights: jax.Array | None = None,
    pred_loss_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    has_global_neg = global_neg_targets is not None and global_neg_mask is not None
    has_pred_loss_mask = pred_loss_mask is not None

    base_in_specs = [P(data_axis), P(data_axis), P(data_axis)]
    if has_pred_loss_mask:
        base_in_specs.append(P(data_axis))
    if has_global_neg:
        base_in_specs.extend([P(data_axis), P(data_axis)])
    if log_q_weights is not None:
        base_in_specs.append(P(data_axis))

    @shard_map(
        mesh=mesh,
        in_specs=tuple(base_in_specs),
        out_specs=P(),
        check_vma=False,
    )
    def _sharded_infonce(*args) -> jax.Array:
        idx = 0
        local_preds = args[idx]
        idx += 1
        local_targets = args[idx]
        idx += 1
        local_mask = args[idx]
        idx += 1
        if has_pred_loss_mask:
            local_pred_loss_mask = args[idx]
            idx += 1
        if has_global_neg:
            local_gn_targets = args[idx]
            idx += 1
            local_gn_mask = args[idx]
            idx += 1
        if log_q_weights is not None:
            local_log_q = args[idx]
            idx += 1

        b, C_pred, D = local_preds.shape
        N_pred = b * C_pred

        flat_preds = local_preds.reshape(N_pred, D)
        flat_targets = local_targets.reshape(N_pred, D)
        flat_mask = local_mask.reshape(N_pred)
        flat_loss_mask = local_pred_loss_mask.reshape(N_pred) if has_pred_loss_mask else flat_mask

        if has_global_neg:
            N_gn = local_gn_targets.shape[0] * local_gn_targets.shape[1]
            flat_gn = local_gn_targets.reshape(N_gn, D)
            flat_gn_mask = local_gn_mask.reshape(N_gn)
            all_targets = jnp.concatenate([flat_targets, flat_gn], axis=0)
            all_mask = jnp.concatenate([flat_mask, flat_gn_mask], axis=0)
        else:
            all_targets = flat_targets
            all_mask = flat_mask

        N_total = all_targets.shape[0]

        logits = (
            jnp.dot(
                flat_preds.astype(jnp.float32),
                all_targets.astype(jnp.float32).T,
            )
            / temperature
        )

        if log_q_weights is not None:
            flat_log_q = local_log_q.reshape(-1)[:N_total]
            logits = logits - flat_log_q[None, :]

        target_col_mask = all_mask[None, :]
        logits = jnp.where(target_col_mask > 0, logits, -INF)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        positive_log_probs = log_probs[jnp.arange(N_pred), jnp.arange(N_pred)]

        nce_loss = -positive_log_probs
        total_loss = jnp.sum(nce_loss * flat_loss_mask)
        total_valid = jnp.sum(flat_loss_mask)

        top1_correct = (jnp.argmax(logits, axis=-1) == jnp.arange(N_pred)).astype(jnp.float32)
        top1_hits = jnp.sum(top1_correct * flat_loss_mask)

        pos_sim_sum = jnp.sum(
            logits[jnp.arange(N_pred), jnp.arange(N_pred)] * temperature * flat_loss_mask
        )

        num_gn_valid = jnp.sum(all_mask) - jnp.sum(flat_mask)

        return jax.lax.psum(
            jnp.stack([total_loss, total_valid, top1_hits, pos_sim_sum, num_gn_valid]),
            axis_name=data_axis,
        )

    call_args = [preds, targets, loss_mask]
    if has_pred_loss_mask:
        call_args.append(pred_loss_mask)
    if has_global_neg:
        call_args.extend([global_neg_targets, global_neg_mask])
    if log_q_weights is not None:
        call_args.append(log_q_weights)

    raw = _sharded_infonce(*call_args)

    total_loss = raw[0]
    total_valid = jnp.maximum(raw[1], 1.0)
    top1_hits = raw[2]
    pos_sim_sum = raw[3]
    num_gn_valid = raw[4]

    loss = total_loss / total_valid

    stats = {
        "gen-recs-infonce-top1-acc": top1_hits / total_valid,
        "gen-recs-infonce-pos-sim": pos_sim_sum / total_valid,
        "gen-recs-infonce-num-negatives": total_valid - 1.0,
        "gen-recs-infonce-num-global-neg": num_gn_valid,
        "gen-recs-inbatch-recall@1": top1_hits / total_valid,
    }

    return loss, stats


def _compute_batch_topk_accuracy(
    preds: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
    data_axis: tuple,
    mesh: jax.sharding.Mesh,
) -> dict[str, jax.Array]:
    preds = jax.lax.stop_gradient(preds)
    targets = jax.lax.stop_gradient(targets)
    loss_mask = jax.lax.stop_gradient(loss_mask)

    @shard_map(
        mesh=mesh,
        in_specs=(P(data_axis), P(data_axis), P(data_axis)),
        out_specs=P(),
        check_vma=False,
    )
    def _sharded_topk(
        local_preds: jax.Array,
        local_targets: jax.Array,
        local_mask: jax.Array,
    ) -> jax.Array:
        b, C, D = local_preds.shape
        N = b * C

        flat_preds = local_preds.reshape(N, D)
        flat_targets = local_targets.reshape(N, D)
        flat_mask = local_mask.reshape(N)

        sim_matrix = jnp.dot(flat_preds, flat_targets.T)
        correct_sim = jnp.diag(sim_matrix)

        user_ids = jnp.arange(N) // C
        same_user_mask = user_ids[:, None] == user_ids[None, :]
        sim_matrix = jnp.where(same_user_mask, -jnp.inf, sim_matrix)

        results = []
        for k in [1, 5, 10]:
            num_higher = jnp.sum(sim_matrix > correct_sim[:, None], axis=-1)
            in_top_k = (num_higher < k).astype(jnp.float32)
            results.append(jnp.sum(in_top_k * flat_mask))
            results.append(jnp.sum(flat_mask))

        return jax.lax.psum(jnp.stack(results), axis_name=data_axis)

    raw = _sharded_topk(preds, targets, loss_mask)

    metrics = {}
    for i, k in enumerate([1, 5, 10]):
        hits = raw[i * 2]
        valid = raw[i * 2 + 1]
        acc = hits / jnp.maximum(valid, 1.0)
        metrics[f"gen-recs-batch-top{k}-acc"] = acc
        metrics[f"gen-recs-inbatch-recall@{k}"] = acc

    return metrics
