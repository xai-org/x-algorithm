# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec as P
from xai_configlib import configclass
from xai_proto import recsys_pb2

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.layers import get_parameter
from xrex.models.normalization import rms_norm_fn
from xrex.models.recsys_embedding import (
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_model import (
    DTYPE_BY_NAME,
    RecsysAggregatedModel,
    RecsysAggregatedModelConfig,
    block_history_reduce,
    cast_jax,
    embed_entity_sid,
)
from xrex.models.recsys_user_features import (
    build_user_feature_parts,
    build_user_features_token,
)
from xrex.models.sharding_context import ShardingContext
from xrex.utils.utils import Summary

POSITIVE_ACTIONS = [
    recsys_pb2.ActionName.SERVER_TWEET_FAV,
]

HARD_NEGATIVE_ACTIONS = [
    recsys_pb2.ActionName.CLIENT_TWEET_REPORT,
    recsys_pb2.ActionName.CLIENT_TWEET_NOT_INTERESTED_IN,
    recsys_pb2.ActionName.CLIENT_TWEET_SEE_FEWER,
    recsys_pb2.ActionName.CLIENT_TWEET_UNFOLLOW_AUTHOR,
    recsys_pb2.ActionName.CLIENT_TWEET_BLOCK_AUTHOR,
    recsys_pb2.ActionName.CLIENT_TWEET_MUTE_AUTHOR,
    recsys_pb2.ActionName.CLIENT_TWEET_NOT_RELEVANT,
]

NUM_SID_LEVELS = 6


@configclass
class RecsysSIDRetrievalConfig(RecsysAggregatedModelConfig):
    sid_codebook_size: int = 256
    sid_num_levels: int = NUM_SID_LEVELS

    sid_prediction_hidden_dim: int = 512

    sid_emb_dim: int = 128

    sid_level_loss_decay: float = 1.0

    eval_num_batches: int = 20
    eval_beam_width: int = 128
    eval_bs_per_device: int = 8

    def make(self, sharding_context: ShardingContext) -> RecsysSIDRetrievalModel:
        return RecsysSIDRetrievalModel(
            model=self.model_config.make(sharding_context=sharding_context),
            config=self,
            sharding_context=sharding_context,
        )


@dataclass
class RecsysSIDRetrievalModel(RecsysAggregatedModel):
    config: RecsysSIDRetrievalConfig

    @hk.transparent
    def _embed_sid_token(self, code: jax.Array, level: int) -> jax.Array:
        cfg = self.config
        K = cfg.sid_codebook_size
        sid_dim = cfg.sid_emb_dim
        D = cfg.emb_table_width
        embed_init = hk.initializers.VarianceScaling(cfg.embed_init_scale, mode="fan_out")

        table = get_parameter(
            f"sid_emb_level_{level}",
            [K, sid_dim],
            dtype=jnp.float32,
            init=embed_init,
            pspec=P(None, None),
        )
        one_hot = jax.nn.one_hot(code, K, dtype=table.dtype)
        emb = jnp.dot(one_hot, table)

        proj = get_parameter(
            f"sid_emb_proj_{level}",
            [sid_dim, D],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
        )
        return jnp.dot(emb, proj).astype(DTYPE_BY_NAME[cfg.fprop_dtype])

    @hk.transparent
    def _predict_sid_level(self, hidden: jax.Array, level: int) -> jax.Array:
        cfg = self.config
        emb_size = cfg.model_config.emb_size
        hidden_dim = cfg.sid_prediction_hidden_dim
        K = cfg.sid_codebook_size
        embed_init = hk.initializers.VarianceScaling(cfg.embed_init_scale, mode="fan_out")

        w1 = get_parameter(
            "sid_pred_w1",
            [emb_size, hidden_dim],
            dtype=jnp.float32,
            init=embed_init,
            pspec=P(None, None),
        )
        h = jnp.dot(hidden.astype(w1.dtype), w1)
        h = jax.nn.relu(h)

        w_head = get_parameter(
            f"sid_pred_head_{level}",
            [hidden_dim, K],
            dtype=jnp.float32,
            init=embed_init,
            pspec=P(None, None),
        )
        return jnp.dot(h, w_head)

    @hk.transparent
    def _build_post_sid_token(
        self,
        post_sids: jax.Array,
        post_hashes: jax.Array,
    ) -> jax.Array:
        cfg = self.config
        needs_hashes = cfg.sid_hash_level
        return embed_entity_sid(
            cast_jax(post_sids),
            cfg.emb_table_width,
            cfg.sid_embed_dim,
            cfg.sid_num_levels,
            cfg.sid_codebook_size,
            cfg.model_config.scale_config.emb_lr_multiplier,
            cfg.embed_init_scale,
            DTYPE_BY_NAME[cfg.fprop_dtype],
            "post",
            sid_hash_level=cfg.sid_hash_level,
            entity_hashes=cast_jax(post_hashes) if needs_hashes else None,
            sid_cross_attn=cfg.sid_cross_attn,
        )

    def _sid_token_or_zeros_init(self, seq) -> jax.Array | None:
        cfg = self.config
        if not cfg.use_post_sid:
            return None
        sids = seq.get("post_sids")
        if sids is None:
            hashes = cast_jax(seq["post_hashes"])
            sids = jnp.zeros(
                (hashes.shape[0], hashes.shape[1], cfg.sid_num_levels), dtype=jnp.uint16
            )
        return self._build_post_sid_token(sids, seq["post_hashes"])

    @hk.transparent
    def _build_history_embeddings(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config

        history_actions = batch["history_seq"]["actions"]
        assert history_actions is not None
        history_actions_embeddings, _ = self.multi_hot_to_embeddings(
            cast_jax(history_actions),
            cfg.model_config.output_vocab_size,
            cfg.emb_table_width,
            cfg.embed_init_scale,
            "action_embedding_table",
        )

        history_cat_features: jax.Array | None = None
        if cfg.context_features.enabled:
            raw_hist_cat = batch["history_seq"].get("categorical_features")
            if raw_hist_cat is not None:
                history_cat_features = cast_jax(raw_hist_cat)
        history_unified_context = self.build_unified_context_embedding(
            product_surface=cast_jax(batch["history_seq"]["product_surface"]),
            cat_features=history_cat_features,
        )

        user_emb_result = self._build_user_embedding(
            cast_jax(batch["user_hashes"]),
            recsys_embeddings,
        )

        sid_post_emb_h = self._sid_token_or_zeros_init(batch["history_seq"])

        history_embeddings, history_padding_mask = block_history_reduce(
            cast_jax(batch["history_seq"]["post_hashes"]),
            cast_jax(batch["history_seq"]["auth_hashes"]),
            recsys_embeddings.history_post_embeddings if cfg.use_post_embedding else None,
            recsys_embeddings.history_author_embeddings,
            history_unified_context,
            history_actions_embeddings,
            cfg.hash_table.hash_keys,
            cfg.model_config.scale_config.emb_lr_multiplier,
            cfg.embed_init_scale,
            history_unified_context is not None,
            cast_jax(batch["history_seq"]["continuous_actions"]),
            DTYPE_BY_NAME[cfg.fprop_dtype],
            cfg.continuous_action_losses,
            cfg.continuous_action_hidden_dim,
            sid_post_embeddings=sid_post_emb_h,
        )

        sequence_parts: list[jax.Array] = []
        mask_parts: list[jax.Array] = []
        if user_emb_result is not None:
            user_embeddings, user_padding_mask = user_emb_result
            assert user_padding_mask is not None
            sequence_parts.append(user_embeddings)
            mask_parts.append(user_padding_mask)

        uf = cfg.user_features
        if uf.has_user_features:
            feature_parts = build_user_feature_parts(batch, uf, self.sharding_context)
            user_features_token = build_user_features_token(
                feature_parts,
                uf.user_features_concat_dim,
                cfg.emb_table_width,
                DTYPE_BY_NAME[cfg.fprop_dtype],
                self.sharding_context,
                use_mlp=uf.user_features_mlp,
                pad=uf.user_features_concat_pad,
            )
            sequence_parts.append(user_features_token)
            mask_parts.append(jnp.ones((history_embeddings.shape[0], 1), dtype=jnp.bool_))

        sequence_parts.append(history_embeddings)
        mask_parts.append(history_padding_mask)

        embeddings = jnp.concatenate(sequence_parts, axis=1)
        padding_mask = jnp.concatenate(mask_parts, axis=1)
        return embeddings, padding_mask

    @hk.transparent
    def _pick_target(self, batch: RecsysFeaturesBatch) -> tuple[jax.Array, jax.Array, jax.Array]:
        cand_sids = cast_jax(batch["candidate_seq"]["post_sids"]).astype(jnp.int32) - 1
        cand_actions = batch["candidate_seq"]["actions"]
        assert cand_actions is not None
        cand_actions_jax = cast_jax(cand_actions)

        has_positive = jnp.sum(cand_actions_jax[:, :, POSITIVE_ACTIONS], axis=-1) > 0
        has_hard_neg = jnp.sum(cand_actions_jax[:, :, HARD_NEGATIVE_ACTIONS], axis=-1) > 0
        has_sid = jnp.all(cand_sids >= 0, axis=-1)
        valid_positive = has_positive & ~has_hard_neg & has_sid
        valid_pos_float = valid_positive.astype(jnp.float32)

        B = cand_sids.shape[0]
        first_pos_idx = jnp.argmax(valid_pos_float, axis=-1)
        has_target = (jnp.sum(valid_pos_float, axis=-1) > 0).astype(jnp.float32)
        num_favs_per_user = jnp.sum(valid_pos_float, axis=-1)
        target_sids = cand_sids[jnp.arange(B), first_pos_idx, :]

        return target_sids, has_target, num_favs_per_user

    @hk.transparent
    def _run_transformer(
        self,
        embeddings: jax.Array,
        padding_mask: jax.Array,
        *,
        is_training: bool = False,
        positions: jax.Array | None = None,
        attn_mask: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config
        B, T, _ = embeddings.shape

        embeddings = embeddings * cfg.model_config.scale_config.input_scale(cfg.emb_table_width)
        embeddings = with_sharding_constraint(embeddings, P(self.data_axis, ("seq", "model")))
        embeddings = self._apply_tfmr_proj(embeddings, cfg)

        segment_ids = jnp.zeros((B, T), dtype=jnp.int32)
        if positions is None:
            positions = jnp.full((B, T, 3), 0, dtype=jnp.float32)
            positions = positions.at[:, :, 0].set(jnp.arange(T)[None, :])
        if attn_mask is None:
            attn_mask = jnp.ones((B, T), dtype=jnp.int32)

        model_out = self.model(
            embeddings,
            attn_mask,
            segment_ids=segment_ids,
            is_training=is_training,
            padding_mask=padding_mask,
            positions=positions,
        )
        return model_out.output

    @hk.transparent
    def _apply_tfmr_proj(self, embeddings: jax.Array, cfg) -> jax.Array:
        if cfg.model_config.emb_size == cfg.emb_table_width:
            return embeddings

        w_init = hk.initializers.VarianceScaling(cfg.model_config.scale_config.attn_init_scale**2)
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
        out = jnp.dot(embeddings.astype(tfmr_proj_w.dtype), tfmr_proj_w)
        return out.astype(DTYPE_BY_NAME[cfg.fprop_dtype])

    @hk.transparent
    def _make_final_ln(self):
        from xrex.models.normalization import RMSNorm

        ln = RMSNorm(
            axis=-1, create_scale=True, scale_init=jnp.ones, pspec=P(None), name="final_layer_norm"
        )
        return ln

    @hk.transparent
    def _decode_hidden(self, emb, mask, level, apply_ln):
        hidden = self._run_transformer(emb, mask, is_training=False)
        h_last = apply_ln(hidden[:, -1:, :])[:, 0, :]
        return self._predict_sid_level(h_last, level)

    @hk.transparent
    def generate(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        beam_width: int = 1,
        decode_levels: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        return self.generate_parallel(
            batch,
            recsys_embeddings,
            beam_width=beam_width,
            decode_levels=decode_levels,
        )

    @hk.transparent
    def _build_block_causal_mask(self, B: int, H: int, K: int, level: int) -> jax.Array:
        T = H + K * level
        q = jnp.arange(T)[:, None]
        kv = jnp.arange(T)[None, :]
        q_hist, kv_hist = q < H, kv < H
        hist_causal = q_hist & kv_hist & (kv <= q)
        beam_to_hist = (~q_hist) & kv_hist
        same_beam = ((q - H) // level) == ((kv - H) // level)
        beam_causal = (~q_hist) & (~kv_hist) & same_beam & (kv <= q)
        base = hist_causal | beam_to_hist | beam_causal
        return jnp.broadcast_to(base[None, None, :, :], (B, 1, T, T))

    @hk.transparent
    def _build_beam_positions(self, B: int, H: int, K: int, level: int) -> jax.Array:
        T = H + K * level
        p = jnp.arange(T)
        pos1d = jnp.where(p < H, p, H + ((p - H) % level)).astype(jnp.float32)
        positions = jnp.zeros((B, T, 3), dtype=jnp.float32)
        return positions.at[:, :, 0].set(jnp.broadcast_to(pos1d[None, :], (B, T)))

    @hk.transparent
    def _embed_beam_tokens(self, beam_codes: jax.Array, level: int) -> jax.Array:
        per_level = [self._embed_sid_token(beam_codes[:, :, j], j) for j in range(level)]
        stacked = jnp.stack(per_level, axis=2)
        B, K, _, D = stacked.shape
        return stacked.reshape(B, K * level, D)

    @hk.transparent
    def _gather_last_token_hidden(self, hidden: jax.Array, H: int, K: int, level: int) -> jax.Array:
        idx = H + jnp.arange(K) * level + (level - 1)
        return hidden[:, idx, :]

    @hk.transparent
    def generate_parallel(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        beam_width: int,
        decode_levels: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config
        attn_impl = cfg.model_config.attn_config.attn_impl
        assert attn_impl in ("jax_attn", "jax_attn_interleaved"), (
            f"generate_parallel requires attn_impl='jax_attn' so the dense "
            f"block-causal mask reaches JaxAttention; got {attn_impl!r}. "
            f"Override model_config.attn_config.attn_impl=jax_attn at launch."
        )
        recsys_emb = get_recsys_embed_param_to_jax_array(recsys_embeddings)
        V = cfg.sid_codebook_size
        apply_ln = self._make_final_ln()
        max_L = cfg.sid_num_levels
        L = decode_levels if decode_levels is not None else max_L
        assert 1 <= L <= max_L, f"decode_levels={L} out of range [1, {max_L}]"
        K = beam_width

        hist_emb, hist_mask = self._build_history_embeddings(batch, recsys_emb)
        B, H, _ = hist_emb.shape

        logits_0 = self._decode_hidden(hist_emb, hist_mask, 0, apply_ln)
        log_probs_0 = jax.nn.log_softmax(logits_0)
        K_l0 = min(K, V)
        top_l0_scores, top_l0_codes = jax.lax.top_k(log_probs_0, K_l0)
        beam_codes = jnp.zeros((B, K, L), dtype=jnp.int32)
        beam_codes = beam_codes.at[:, :K_l0, 0].set(top_l0_codes)
        beam_scores = jnp.full((B, K), -jnp.inf, dtype=top_l0_scores.dtype)
        beam_scores = beam_scores.at[:, :K_l0].set(top_l0_scores)

        for level in range(1, L):
            beam_tokens = self._embed_beam_tokens(beam_codes, level)
            seq_emb = jnp.concatenate([hist_emb, beam_tokens], axis=1)
            seq_pad_mask = jnp.concatenate(
                [hist_mask, jnp.ones((B, K * level), dtype=hist_mask.dtype)], axis=1
            )
            block_mask = self._build_block_causal_mask(B, H, K, level)
            positions = self._build_beam_positions(B, H, K, level)

            hidden = self._run_transformer(
                seq_emb,
                seq_pad_mask,
                is_training=False,
                attn_mask=block_mask,
                positions=positions,
            )

            last_hidden = self._gather_last_token_hidden(hidden, H, K, level)
            last_hidden = apply_ln(last_hidden)
            logits = self._predict_sid_level(last_hidden, level)
            log_probs = jax.nn.log_softmax(logits)

            candidate_scores = beam_scores[:, :, None] + log_probs
            candidate_scores_flat = candidate_scores.reshape(B, K * V)
            top_k_flat_scores, top_k_flat_indices = jax.lax.top_k(candidate_scores_flat, K)
            source_beam = top_k_flat_indices // V
            new_code = (top_k_flat_indices % V).astype(jnp.int32)

            new_beam_codes = jnp.zeros((B, K, L), dtype=jnp.int32)
            for prev_level in range(level):
                gathered = jnp.take_along_axis(beam_codes[:, :, prev_level], source_beam, axis=1)
                new_beam_codes = new_beam_codes.at[:, :, prev_level].set(gathered)
            new_beam_codes = new_beam_codes.at[:, :, level].set(new_code)

            beam_codes = new_beam_codes
            beam_scores = top_k_flat_scores

        return beam_codes, beam_scores

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
        cfg = self.config
        recsys_emb = get_recsys_embed_param_to_jax_array(recsys_embeddings)

        target_sids, has_target, num_favs_per_user = self._pick_target(batch)
        B = target_sids.shape[0]
        num_valid = jnp.maximum(jnp.sum(has_target), 1.0)

        with Summary() as summarizer:
            hist_emb, hist_mask = self._build_history_embeddings(batch, recsys_emb)
        H_total = hist_emb.shape[1]

        L = cfg.sid_num_levels
        K = cfg.sid_codebook_size
        sid_tf_list = []
        for level in range(L):
            code = target_sids[:, level]
            emb = self._embed_sid_token(code, level)
            sid_tf_list.append(emb[:, None, :])

        sid_tokens = jnp.concatenate(sid_tf_list, axis=1)
        sid_mask = jnp.ones((B, L), dtype=hist_mask.dtype)

        embeddings = jnp.concatenate([hist_emb, sid_tokens], axis=1)
        padding_mask = jnp.concatenate([hist_mask, sid_mask], axis=1)

        hidden = self._run_transformer(embeddings, padding_mask, is_training=is_training)
        hidden = rms_norm_fn(hidden, name="final_layer_norm")

        stats = summarizer.get()
        total_weighted_loss = jnp.float32(0.0)
        total_weight = jnp.float32(0.0)
        decay = cfg.sid_level_loss_decay

        for level in range(L):
            pred_pos = H_total - 1 + level
            h = hidden[:, pred_pos, :]
            logits = self._predict_sid_level(h, level)

            target_code = target_sids[:, level]
            safe_code = jnp.clip(target_code, 0, K - 1)
            level_loss = -jax.nn.log_softmax(logits)[jnp.arange(B), safe_code]
            level_loss = jnp.sum(level_loss * has_target) / num_valid

            level_weight = decay**level
            total_weighted_loss = total_weighted_loss + level_weight * level_loss
            total_weight = total_weight + level_weight

            preds = jnp.argmax(logits, axis=-1)
            correct = (preds == target_code).astype(jnp.float32) * has_target

            stats[f"sid/L{level}_loss"] = level_loss
            stats[f"sid/L{level}_weight"] = jnp.float32(level_weight)
            stats[f"sid/L{level}_accuracy"] = jnp.sum(correct) / num_valid

            top5 = jax.lax.top_k(logits, min(5, K))[1]
            in_top5 = (
                jnp.any(top5 == target_code[:, None], axis=-1).astype(jnp.float32) * has_target
            )
            stats[f"sid/L{level}_top5"] = jnp.sum(in_top5) / num_valid

        loss = total_weighted_loss / total_weight

        stats["sid/loss"] = loss
        stats["sid/num_valid_users"] = jnp.sum(has_target)
        stats["sid/num_invalid_users"] = B - jnp.sum(has_target)
        stats["sid/valid_pct"] = jnp.sum(has_target) / B
        stats["sid/total_favs"] = jnp.sum(num_favs_per_user)
        stats["sid/avg_favs_per_seq"] = jnp.sum(num_favs_per_user) / B
        stats["sid/avg_favs_per_valid_seq"] = jnp.sum(num_favs_per_user) / num_valid

        return loss, stats
