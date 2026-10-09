# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
import math
import os
import typing
from dataclasses import dataclass, field
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import shard_map
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xai_proto import recsys_pb2
from xrex.cuda.top_k_by_key import gather_selected_validity, top_k_by_key
from xrex.data.cold_pool_filter import DEFAULT_COLD_START_MAX_AGE_SECONDS
from xrex.data.recsys.constants import action_type_map
from xrex.data.recsys.recsys_batch import EmbeddingType
from xrex.data.recsys.safety_filter import (
    SafetyFilterMode,
    apply_safety_filter,
    safety_filter_stats,
)
from xrex.data.retrieval_dataset import RetrievalDataset
from xrex.models.layers import Linear
from xrex.models.model_utils import Parameter, get_parameter
from xrex.models.recsys_attention import RecsysAttentionConfig
from xrex.models.recsys_embedding import (
    HashTable,
    RecsysEmbeddings,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_feature_prep import build_feature_prep_inputs
from xrex.models.recsys_model import (
    DTYPE_BY_NAME,
    MemoryKind,
    RecsysAggregatedModel,
    RecsysAggregatedModelConfig,
    RecsysEmbeddingsParameter,
    RecsysFeaturesBatch,
    UserFeaturesConfig,
    block_history_reduce,
    build_user_feature_parts,
    build_user_features_token,
    cast_jax,
    embed_entity_sid,
    get_candidate_tweet_counts,
    pad_to_next_128_multiple,
    right_anchored_rope_positions,
)
from xrex.models.scaling import ScaleConfig
from xrex.models.sharding_context import ShardingContext
from xrex.models.topic_categories import NUM_TOPIC_INT32S
from xrex.pallas.ranker_attention_utils import HISTORY_SEGMENT_ID, PADDING_SEGMENT_ID
from xrex.train.misc import PostEmbeddings
from xrex.utils.utils import Summary

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")
QUERY_HEAD_RECENCY_SCALE_TOKENS = 16.0
EPS = 1e-12
INF = 1e12


def user_tower_segment_ids(
    batch: int,
    seq_len: int,
    *,
    use_history_segment_ids: bool,
    padding_mask: jax.Array | None = None,
) -> jax.Array:
    if use_history_segment_ids:
        return jnp.full((batch, seq_len), HISTORY_SEGMENT_ID, dtype=jnp.int32)
    if padding_mask is None:
        return jnp.zeros((batch, seq_len), dtype=jnp.int32)
    return jnp.where(padding_mask, HISTORY_SEGMENT_ID, PADDING_SEGMENT_ID).astype(jnp.int32)


def _l2_normalize_candidates(embeddings: jax.Array) -> jax.Array:
    norm_sq = jnp.sum(embeddings**2, axis=-1, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(norm_sq, EPS))
    return embeddings / norm


class RecsysCandidateTower(hk.Module):
    config: RecsysCandidateModelConfig
    sharding_context: ShardingContext
    embeddings: jax.Array
    post_ids: npt.NDArray[np.int64]

    def __init__(
        self,
        config: RecsysCandidateModelConfig,
        sharding_context: ShardingContext,
        *,
        use_post_embedding: bool = True,
        use_post_sid: bool = False,
        use_project_then_sum: bool = False,
    ):
        super().__init__(name="candidate_tower")
        self.config = config
        self.sharding_context = sharding_context
        self.use_post_embedding = use_post_embedding
        self.use_post_sid = use_post_sid
        self.use_project_then_sum = use_project_then_sum

    @hk.transparent
    def _concat_then_mlp(self, post_author_embedding: jax.Array, head_index: int = 0) -> jax.Array:
        if len(post_author_embedding.shape) == 4:
            B, C, _, _ = post_author_embedding.shape
            post_author_embedding = jnp.reshape(post_author_embedding, (B, C, -1))
        else:
            B, _, _ = post_author_embedding.shape
            post_author_embedding = jnp.reshape(post_author_embedding, (B, -1))

        w_init = hk.initializers.VarianceScaling(self.config.scale_config.attn_init_scale**2)
        lr_multiplier = self.config.scale_config.hidden_lr_multiplier(self.config.emb_table_width)
        init_scale = 1.0

        if head_index == 0:
            proj_1_name = "candidate_tower_projection_1"
            proj_2_name = "candidate_tower_projection_2"
        else:
            proj_1_name = f"candidate_tower_head_{head_index}_projection_1"
            proj_2_name = f"candidate_tower_head_{head_index}_projection_2"

        candidate_tower_projection_1 = Linear(
            self.config.emb_table_width * 2,
            w_init=w_init,
            with_bias=False,
            pspec=P(None, None),
            sharding_context=self.sharding_context,
            lr_multiplier=lr_multiplier,
            init_scale=init_scale,
            name=proj_1_name,
        )
        candidate_tower_projection_2 = Linear(
            self.config.emb_table_width,
            w_init=w_init,
            with_bias=False,
            pspec=P(None, None),
            sharding_context=self.sharding_context,
            lr_multiplier=lr_multiplier,
            init_scale=init_scale,
            name=proj_2_name,
        )
        candidate_embeddings = candidate_tower_projection_2(
            jax.nn.silu(candidate_tower_projection_1(inputs=post_author_embedding))
        )
        return _l2_normalize_candidates(candidate_embeddings)

    @hk.transparent
    def _project_then_sum(
        self,
        post_author_embedding: jax.Array,
        head_index: int = 0,
    ) -> jax.Array:
        num_item_hashes = self.config.hash_table.hash_keys.num_item_hashes
        num_author_hashes = self.config.hash_table.hash_keys.num_author_hashes
        use_post = self.use_post_embedding
        use_sid = self.use_post_sid
        expected_tokens = (
            (num_item_hashes if use_post else 0) + num_author_hashes + (1 if use_sid else 0)
        )
        total_tokens = post_author_embedding.shape[-2]
        assert total_tokens == expected_tokens, (
            f"project-then-sum expected {expected_tokens} tokens on axis -2 "
            f"(use_post_embedding={use_post}, num_item_hashes={num_item_hashes}, "
            f"num_author_hashes={num_author_hashes}, use_post_sid={use_sid}), "
            f"got {total_tokens}"
        )

        offset = 0
        post_embeddings: jax.Array | None = None
        if use_post:
            post_embeddings = post_author_embedding[..., offset : offset + num_item_hashes, :]
            offset += num_item_hashes
        author_embeddings = post_author_embedding[..., offset : offset + num_author_hashes, :]
        offset += num_author_hashes
        sid_embedding: jax.Array | None = None
        if use_sid:
            sid_embedding = post_author_embedding[..., offset, :]
            offset += 1

        W = self.config.emb_table_width
        D = self.config.emb_table_width
        B = author_embeddings.shape[0]
        if author_embeddings.ndim == 4:
            out_shape_prefix = (B, author_embeddings.shape[1])
        else:
            out_shape_prefix = (B,)

        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        lr_multiplier = self.config.scale_config.hidden_lr_multiplier(W)

        def _proj_one(emb: jax.Array, name: str) -> jax.Array:
            proj = typing.cast(
                jax.Array,
                get_parameter(
                    name,
                    [W, D],
                    dtype=jnp.float32,
                    init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
                    pspec=P(None, None),
                    lr_multiplier=lr_multiplier,
                ),
            )
            return jnp.dot(emb.astype(proj.dtype), proj).astype(emb.dtype)

        head_suffix = "" if head_index == 0 else f"_head_{head_index}"
        result = jnp.zeros((*out_shape_prefix, D), dtype=author_embeddings.dtype)

        if post_embeddings is not None:
            for i in range(num_item_hashes):
                h = post_embeddings[..., i, :]
                result = result + _proj_one(h, f"cand_post_hash_{i}_proj{head_suffix}")

        for i in range(num_author_hashes):
            h = author_embeddings[..., i, :]
            result = result + _proj_one(h, f"cand_author_hash_{i}_proj{head_suffix}")

        if sid_embedding is not None:
            result = result + _proj_one(sid_embedding, f"cand_sid_proj{head_suffix}")

        return _l2_normalize_candidates(result)

    def __call__(self, post_author_embedding: jax.Array, head_index: int = 0) -> jax.Array:
        if self.use_project_then_sum:
            return self._project_then_sum(post_author_embedding, head_index=head_index)
        if self.config.enable_linear_proj:
            return self._concat_then_mlp(post_author_embedding, head_index=head_index)
        return self._mean_pool(post_author_embedding)

    def _mean_pool(self, post_author_embedding: jax.Array) -> jax.Array:
        return _l2_normalize_candidates(jnp.mean(post_author_embedding, axis=-2))


@configclass
class RecsysCandidateModelConfig(Config):
    hash_table: HashTable
    enable_linear_proj: bool = False
    emb_table_width: int = 128
    scale_config: ScaleConfig = ScaleConfig()
    max_posts: int = 10_240_000
    num_candidate_heads: int = 1

    def make(
        self,
        sharding_context: ShardingContext,
        *,
        use_post_embedding: bool = True,
        use_post_sid: bool = False,
        use_project_then_sum: bool = False,
    ):
        if use_project_then_sum and self.enable_linear_proj:
            raise ValueError(
                "feature_prep_enabled (candidate project-then-sum) and "
                "enable_linear_proj are mutually exclusive; enable at most one "
                "candidate-tower combine mode (or neither for mean-pool)."
            )

        return RecsysCandidateTower(
            self,
            sharding_context,
            use_post_embedding=use_post_embedding,
            use_post_sid=use_post_sid,
            use_project_then_sum=use_project_then_sum,
        )

    def make_post_embeddings(self):
        all_post_ids = jnp.arange(self.max_posts * 2).reshape(-1, 2).astype(jnp.int32)
        all_author_ids = jnp.arange(self.max_posts * 2).reshape(-1, 2).astype(jnp.int32)
        all_dataset_types = jnp.full(
            (self.max_posts, 1), RetrievalDataset.PAD.value, dtype=jnp.int32
        )
        scale = 1.0 / math.sqrt(self.emb_table_width)
        if os.environ.get("DEBUG_ALLOW_RANDOM_INIT") == "1":
            emb_init = jax.random.uniform(
                jax.random.PRNGKey(0),
                (self.max_posts, self.emb_table_width),
                dtype=jnp.bfloat16,
                minval=-scale,
                maxval=scale,
            )
        else:
            emb_init = jnp.empty((self.max_posts, self.emb_table_width), dtype=jnp.bfloat16)
        post_embeddings = Parameter(
            x=emb_init,
            pspec=P(("stage", "expert", "replica", "data"), ("seq", "model")),
        )
        return PostEmbeddings(
            post_ids=all_post_ids,
            author_ids=all_author_ids,
            embeddings=post_embeddings,
            dataset_types=all_dataset_types,
        )


def _compute_recall_at_k(
    N: int,
    pos_scores: jax.Array,
    neg_scores: jax.Array,
    valid_mask: jax.Array,
    implicit_negative_mask: jax.Array,
    explicit_negative_mask: jax.Array,
    use_in_batch_negatives: bool,
    data_axis: tuple,
    mesh: jax.sharding.Mesh,
    debug_mode: bool = False,
) -> dict[str, jax.Array]:
    if use_in_batch_negatives:

        @shard_map(
            mesh=mesh,
            in_specs=(P(data_axis)),
            out_specs=(P(data_axis), P(data_axis)),
            check_vma=False,
        )
        def _pos_scores(pos_scores_shard: jax.Array) -> tuple[jax.Array, jax.Array]:
            b = pos_scores_shard.shape[0]
            pos_scores_shard = pos_scores_shard.reshape((b, b, -1))

            off_diag_cols = (jnp.arange(b)[:, None] + jnp.arange(1, b)[None, :]) % b
            off_diag_cols = jnp.sort(off_diag_cols, axis=1)
            in_batch_negatives = pos_scores_shard[jnp.arange(b)[:, None], off_diag_cols].reshape(
                b, -1
            )

            self_cands = pos_scores_shard[jnp.arange(b), jnp.arange(b)]

            return self_cands, in_batch_negatives

        pos_scores, inbatch_negatives = _pos_scores(pos_scores)
        if N > 0:
            neg_scores = jnp.concatenate((inbatch_negatives, neg_scores), axis=-1)
        else:
            neg_scores = inbatch_negatives

    recall_metrics = {}
    if not debug_mode:
        return recall_metrics

    sorted_neg_scores = jnp.sort(neg_scores, axis=-1)
    for k in [1, 10, 100, 1000]:
        kth_negative_scores = sorted_neg_scores[:, -k]
        all_is_in_top_k = pos_scores >= kth_negative_scores[:, None]
        is_in_top_k = all_is_in_top_k * valid_mask.astype(jnp.float32)
        recall = jnp.where(
            jnp.any(valid_mask),
            jnp.sum(is_in_top_k.astype(jnp.float32)) / jnp.sum(valid_mask.astype(jnp.float32)),
            1.0,
        )
        recall_metrics[f"InBatchRecall@{k}"] = recall

        implicit_negative_in_top_k = all_is_in_top_k * implicit_negative_mask.astype(jnp.float32)
        implicit_negative_recall = jnp.where(
            jnp.any(implicit_negative_mask),
            jnp.sum(implicit_negative_in_top_k.astype(jnp.float32))
            / jnp.sum(implicit_negative_mask.astype(jnp.float32)),
            1.0,
        )
        recall_metrics[f"ImplicitNegativeInBatchRecall@{k}"] = implicit_negative_recall

        explicit_negative_in_top_k = all_is_in_top_k * explicit_negative_mask.astype(jnp.float32)
        explicit_negative_recall = jnp.where(
            jnp.any(explicit_negative_mask),
            jnp.sum(explicit_negative_in_top_k.astype(jnp.float32))
            / jnp.sum(explicit_negative_mask.astype(jnp.float32)),
            1.0,
        )
        recall_metrics[f"ExplicitNegativeInBatchRecall@{k}"] = explicit_negative_recall

    return recall_metrics


def _compute_score_stats(
    self_scores: jax.Array,
    global_neg_scores: jax.Array,
    valid_positive_mask: jax.Array,
    use_in_batch_negatives: bool,
    data_axis: tuple,
    mesh: jax.sharding.Mesh,
    debug_mode: bool = False,
) -> dict[str, jax.Array]:
    if use_in_batch_negatives:

        @shard_map(
            mesh=mesh,
            in_specs=(P(data_axis)),
            out_specs=(P(data_axis)),
            check_vma=False,
        )
        def _self_scores(self_scores_shard: jax.Array) -> jax.Array:
            b = self_scores_shard.shape[0]
            self_scores_shard = self_scores_shard.reshape((b, b, -1))
            return self_scores_shard[jnp.arange(b), jnp.arange(b)]

        self_scores = _self_scores(self_scores)

    score_stats = {}

    valid_count = jnp.sum(valid_positive_mask.astype(jnp.float32))
    valid_offset = 1.0 - (valid_count / valid_positive_mask.size)

    score_stats["two_tower_positive_scores_mean"] = jnp.mean(self_scores, where=valid_positive_mask)
    score_stats["two_tower_positive_scores_std"] = jnp.std(self_scores, where=valid_positive_mask)

    score_stats["two_tower_positive_scores_min"] = jnp.min(
        self_scores, where=valid_positive_mask, initial=1.0
    )

    if debug_mode:
        masked_for_sort = jnp.where(valid_positive_mask, self_scores, -INF)
        offsets = jnp.asarray([0.05, 0.95, 1.0]) + valid_offset
        p5, p95, mx = jnp.quantile(masked_for_sort, offsets, method="higher")

        score_stats["two_tower_positive_scores_max"] = mx
        score_stats["two_tower_positive_scores_p5"] = p5
        score_stats["two_tower_positive_scores_p95"] = p95

        score_stats["two_tower_negative_scores_mean"] = global_neg_scores.mean()
        score_stats["two_tower_negative_scores_std"] = global_neg_scores.std()

        mn, p5, p95, mx = jnp.quantile(global_neg_scores, jnp.asarray([0.0, 0.05, 0.95, 1.0]))
        score_stats["two_tower_negative_scores_min"] = mn
        score_stats["two_tower_negative_scores_max"] = mx
        score_stats["two_tower_negative_scores_p5"] = p5
        score_stats["two_tower_negative_scores_p95"] = p95

        score_stats["two_tower_postive_minus_negative_scores_mean"] = (
            score_stats["two_tower_positive_scores_mean"]
            - score_stats["two_tower_negative_scores_mean"]
        )

    return score_stats


def _compute_retrieval_metrics(
    C: int,
    N: int,
    raw_batch_scores: jax.Array,
    raw_global_neg_scores: jax.Array,
    self_scores: jax.Array,
    global_neg_scores: jax.Array,
    valid_positive_mask: jax.Array,
    implicit_negative_mask: jax.Array,
    explicit_negative_mask: jax.Array,
    candidate_padding_mask: jax.Array,
    actions: npt.NDArray[np.bool_],
    contrastive_loss: jax.Array,
    has_hard_negative_actions: jax.Array,
    positive_actions: list[int],
    hard_negative_actions: list[int],
    soft_negative_actions: list[int],
    use_in_batch_negatives: bool,
    data_axis: tuple,
    mesh: jax.sharding.Mesh,
    debug_mode: bool = False,
) -> dict[str, jax.Array]:
    recall_metrics = _compute_recall_at_k(
        N,
        raw_batch_scores,
        raw_global_neg_scores,
        valid_positive_mask,
        implicit_negative_mask,
        explicit_negative_mask,
        use_in_batch_negatives,
        data_axis,
        mesh,
        debug_mode,
    )

    score_stats = _compute_score_stats(
        raw_batch_scores,
        raw_global_neg_scores,
        valid_positive_mask,
        use_in_batch_negatives,
        data_axis,
        mesh,
        debug_mode,
    )

    config_actions = positive_actions + hard_negative_actions + soft_negative_actions
    action_value_to_name = {v: k for k, v in action_type_map.items()}
    num_examples_with_action = {
        f"num_examples_with_action_{action_value_to_name[action_name]}_per_batch": actions[
            :, :C, action_name
        ]
        .astype(jnp.float32)
        .sum()
        for action_name in config_actions
    }

    return {
        "contrastive_loss": contrastive_loss,
        "num_valid_positive_examples_per_batch": jnp.sum(valid_positive_mask.astype(jnp.float32)),
        "fraction_valid_positive_examples": jnp.mean(valid_positive_mask.astype(jnp.float32)),
        "num_negatives_per_example": jnp.float32(global_neg_scores.shape[-1]),
        "num_padding_candidates_per_example": (~candidate_padding_mask)
        .astype(jnp.float32)
        .sum(axis=-1)
        .mean(),
        "num_hard_negatives_per_example": (
            has_hard_negative_actions & candidate_padding_mask[:, :C]
        )
        .astype(jnp.float32)
        .sum(axis=-1)
        .mean(),
        **num_examples_with_action,
        **recall_metrics,
        **score_stats,
    }


class MolScorer(typing.NamedTuple):
    adapter_v: jax.Array
    adapter_u: jax.Array
    gate_w1: jax.Array
    gate_b1: jax.Array
    gate_w2: jax.Array
    gate_b2: jax.Array
    gate_user_w: jax.Array | None = None
    gate_item_w: jax.Array | None = None

    def item_components(self, items: jax.Array) -> jax.Array:
        x = items.astype(jnp.float32)
        delta = jnp.einsum("...d,ldr->...lr", x, self.adapter_v)
        delta = jnp.einsum("...lr,lrd->...ld", delta, self.adapter_u).astype(items.dtype)
        comps = items[..., None, :] + delta
        sq = jnp.sum(jnp.square(comps.astype(jnp.float32)), axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(jnp.maximum(sq, EPS)).astype(items.dtype)
        return comps * inv_norm

    def mix(self, cosines: jax.Array, bias: jax.Array | None = None) -> jax.Array:
        flat = cosines.reshape(*cosines.shape[:-2], -1)
        pre = flat.astype(jnp.bfloat16) @ self.gate_w1.astype(jnp.bfloat16)
        hidden = jax.nn.silu(pre + self.gate_b1.astype(jnp.bfloat16))
        logits = (hidden @ self.gate_w2.astype(jnp.bfloat16)).astype(jnp.float32) + self.gate_b2
        if bias is not None:
            logits = logits + bias
        gate = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(gate * flat.astype(jnp.float32), axis=-1)

    def user_gate_bias(self, users: jax.Array) -> jax.Array | None:
        if self.gate_user_w is None:
            return None
        return users[:, 0].astype(jnp.float32) @ self.gate_user_w

    def item_gate_bias(self, items: jax.Array) -> jax.Array | None:
        if self.gate_item_w is None:
            return None
        return items.astype(jnp.float32) @ self.gate_item_w

    @staticmethod
    def _combined_bias(user_bias: jax.Array | None, item_bias: jax.Array | None):
        if user_bias is None and item_bias is None:
            return None
        bias = 0.0
        if user_bias is not None:
            bias = bias + user_bias
        if item_bias is not None:
            bias = bias + item_bias
        return bias

    def pair_scores_from_components(
        self, users: jax.Array, comps: jax.Array, item_bias: jax.Array | None
    ) -> jax.Array:
        cos = jnp.einsum("bkd,nld->bnkl", users.astype(comps.dtype), comps)
        user_bias = self.user_gate_bias(users)
        bias = self._combined_bias(
            None if user_bias is None else user_bias[:, None, :],
            None if item_bias is None else item_bias[None, :, :],
        )
        return self.mix(cos, bias)

    def pair_scores(self, users: jax.Array, items: jax.Array, user_chunk: int = 0) -> jax.Array:
        comps = self.item_components(items)
        item_bias = self.item_gate_bias(items)
        batch_size = users.shape[0]
        if user_chunk <= 0 or user_chunk >= batch_size or batch_size % user_chunk != 0:
            return self.pair_scores_from_components(users, comps, item_bias)

        def one(u):
            return self.pair_scores_from_components(u, comps, item_bias)

        out = jax.lax.map(
            jax.checkpoint(one),
            users.reshape(batch_size // user_chunk, user_chunk, *users.shape[1:]),
        )
        return out.reshape(batch_size, -1)

    def per_candidate_scores(self, users: jax.Array, items: jax.Array) -> jax.Array:
        comps = self.item_components(items)
        cos = jnp.einsum("bkd,bcld->bckl", users.astype(comps.dtype), comps)
        user_bias = self.user_gate_bias(users)
        bias = self._combined_bias(
            None if user_bias is None else user_bias[:, None, :], self.item_gate_bias(items)
        )
        return self.mix(cos, bias)

    def serving_side_tables(
        self, posts: jax.Array, post_scales: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        L, D, R = self.adapter_v.shape
        kl = self.gate_w1.shape[0]
        item_w = (
            self.gate_item_w if self.gate_item_w is not None else jnp.zeros((D, kl), jnp.float32)
        )
        proj = jnp.concatenate(
            [self.adapter_v[l] for l in range(L)]
            + [jnp.transpose(self.adapter_u[l]) for l in range(L)]
            + [item_w],
            axis=1,
        ).astype(jnp.bfloat16)
        x = posts.astype(jnp.bfloat16)
        y = jnp.dot(x, proj, preferred_element_type=jnp.float32)
        sq = jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1)
        if post_scales is not None:
            y = y * post_scales[:, None]
            sq = sq * jnp.square(post_scales)
        rows = posts.shape[0]
        t = y[:, : L * R].reshape(rows, L, R)
        u_t = y[:, L * R : 2 * L * R].reshape(rows, L, R)
        gram = jnp.einsum("lrd,lsd->lrs", self.adapter_u, self.adapter_u)
        sq_norms = (
            sq[:, None]
            + 2.0 * jnp.sum(t * u_t, axis=-1)
            + jnp.einsum("mlr,lrs,mls->ml", t, gram, t)
        )
        norms = jnp.sqrt(jnp.maximum(sq_norms, EPS))
        return jnp.transpose(norms), jnp.transpose(y[:, 2 * L * R :])

    def side_table_rows(self, posts: jax.Array) -> jax.Array:
        norms, ibias = self.serving_side_tables(posts)
        return jnp.concatenate([jnp.transpose(norms), jnp.transpose(ibias)], axis=1)

    def sharded_serving_side_tables(self, posts: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _mol_side_tables_chunked(self, posts.astype(jnp.bfloat16), None)

    def transformed_queries(self, users: jax.Array) -> jax.Array:
        u = users.astype(jnp.float32)
        t = jnp.einsum("bkd,lrd->bklr", u, self.adapter_u)
        t = jnp.einsum("bklr,ldr->bkld", t, self.adapter_v)
        return u[:, :, None, :] + t

    def chunked_pair_scores(self, users: jax.Array, items: jax.Array, chunk: int) -> jax.Array:
        comps = self.item_components(items)
        item_bias = self.item_gate_bias(items)
        batch_size = users.shape[0]
        chunk = min(chunk, batch_size)
        assert batch_size % chunk == 0, (
            f"eval users {batch_size} must be divisible by mol_eval_user_chunk {chunk}"
        )

        def one(u):
            return self.pair_scores_from_components(u, comps, item_bias).astype(jnp.bfloat16)

        out = jax.lax.map(one, users.reshape(batch_size // chunk, chunk, *users.shape[1:]))
        return out.reshape(batch_size, -1)


def _mol_side_tables_chunked(
    scorer: MolScorer,
    posts: jax.Array,
    post_scales: jax.Array | None,
    max_chunk_rows: int = 1 << 22,
) -> tuple[jax.Array, jax.Array]:
    rows, width = posts.shape
    chunk = next(c for c in range(min(rows, max_chunk_rows), 0, -1) if rows % c == 0)
    blocks = posts.reshape(rows // chunk, chunk, width)
    if post_scales is None:
        norms, ibias = jax.lax.map(scorer.serving_side_tables, blocks)
    else:
        norms, ibias = jax.lax.map(
            lambda args: scorer.serving_side_tables(*args),
            (blocks, post_scales.reshape(rows // chunk, chunk)),
        )
    return (
        jnp.transpose(norms, (1, 0, 2)).reshape(norms.shape[1], rows),
        jnp.transpose(ibias, (1, 0, 2)).reshape(ibias.shape[1], rows),
    )


def _mol_transformed_query_dots(
    q_cb: jax.Array, posts: jax.Array, post_scales: jax.Array | None
) -> jax.Array:
    if post_scales is None:
        return jnp.matmul(q_cb.astype(posts.dtype), posts.T)
    qf = q_cb.astype(jnp.float32)
    qs = jnp.maximum(jnp.max(jnp.abs(qf), axis=1) / 127.0, 1e-12)
    qq = jnp.clip(jnp.round(qf / qs[:, None]), -127, 127).astype(jnp.int8)
    acc = jax.lax.dot_general(qq, posts, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32)
    return (acc.astype(jnp.float32) * qs[:, None] * post_scales[None, :]).astype(jnp.bfloat16)


def mol_kernel_scores(
    scorer: MolScorer,
    users: jax.Array,
    posts: jax.Array,
    post_scales: jax.Array | None = None,
    side_tables: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    from xrex.cuda import mol_epilogue

    if post_scales is None:
        posts = posts.astype(jnp.bfloat16)
    if side_tables is None:
        norms, ibias = _mol_side_tables_chunked(scorer, posts, post_scales)
    else:
        norms, ibias = side_tables
    B, K, D = users.shape
    q = scorer.transformed_queries(users)
    L = q.shape[2]
    kl = K * L
    q_cb = jnp.transpose(q, (1, 2, 0, 3)).reshape(kl * B, D)
    dots = _mol_transformed_query_dots(q_cb, posts, post_scales)
    ubias = scorer.user_gate_bias(users)
    ubias = jnp.zeros((B, kl), jnp.float32) if ubias is None else ubias.astype(jnp.float32)
    return mol_epilogue.mol_epilogue(
        dots,
        norms,
        jnp.transpose(ubias),
        ibias,
        scorer.gate_w1,
        scorer.gate_b1,
        scorer.gate_w2,
        scorer.gate_b2,
        num_users=B,
        num_components=L,
    )


def positive_count_user_weight(per_user_count: jax.Array, power: float) -> jax.Array:
    valid_user = per_user_count > 0
    num_valid_users = jnp.sum(valid_user.astype(jnp.float32))
    weight = jnp.where(valid_user, jnp.maximum(per_user_count, 1.0) ** power, 0.0)
    return weight * num_valid_users / jnp.maximum(jnp.sum(weight), 1e-6)


def compute_retrieval_loss(
    batch: RecsysFeaturesBatch,
    user_representation: jax.Array,
    candidate_representation: jax.Array,
    temperature: jax.Array,
    data_axis: tuple,
    mesh: jax.sharding.Mesh,
    num_global_negatives_per_example: int,
    candidate_seq_len: int,
    use_in_batch_negatives: bool,
    positive_actions: list[int],
    hard_negative_actions: list[int],
    soft_negative_actions: list[int],
    logq_correction_scale: float,
    enable_fake_positives: bool = False,
    debug_mode: bool = False,
    apply_u2u_and_i2i_loss: bool = True,
    ads_only_candidates: bool = False,
    safety_filter_mode: SafetyFilterMode = "off",
    safety_filter_bits: int = 0b11,
    safety_filter_soft_weight: float = 0.0,
    safety_filter_apply_to_candidates: bool = False,
    own_negative_logit_offset: float = 0.0,
    positive_count_user_weight_power: float = 0.0,
    mol_scorer: MolScorer | None = None,
    mol_train_user_chunk: int = 0,
    mask_in_batch_false_negatives: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    B, L, _ = candidate_representation.shape
    N = num_global_negatives_per_example
    C = candidate_seq_len
    assert L == C + N, (
        f"candidate_representation.shape[1] ({candidate_representation.shape[1]}) must be candidate_seq_len ({C}) + num_global_negatives_per_example ({N})"
    )

    use_mol = mol_scorer is not None

    @jax.checkpoint
    def _mol_pair_scores(users, items, scorer):
        return scorer.pair_scores(users, items, user_chunk=mol_train_user_chunk)

    @jax.checkpoint
    def _mol_candidate_scores(users, items, scorer):
        return scorer.per_candidate_scores(users, items)

    @shard_map(
        mesh=mesh,
        in_specs=(P(data_axis), P(data_axis), P(data_axis), P()),
        out_specs=(P(data_axis), P(data_axis)),
        check_vma=False,
    )
    def _sharded_full_matmul(
        local_user: jax.Array, local_candidate: jax.Array, padding_mask: jax.Array, scorer
    ) -> tuple[jax.Array, jax.Array]:
        local_B = local_user.shape[0]

        if use_in_batch_negatives:
            inbatch_cand = local_candidate[:, :C, :].reshape((local_B * C, -1))
            if use_mol:
                inbatch_scores = _mol_pair_scores(local_user, inbatch_cand, scorer)
            else:
                inbatch_scores = local_user @ inbatch_cand.T

            inbatch_pad = padding_mask[:, :C].reshape((1, local_B * C))
            inbatch_scores = jnp.where(inbatch_pad, inbatch_scores, -INF)
        else:
            inbatch_cand = local_candidate[:, :C, :]
            if use_mol:
                inbatch_scores = _mol_candidate_scores(local_user, inbatch_cand, scorer)
            else:
                inbatch_scores = jnp.einsum("bd,bcd->bc", local_user, inbatch_cand)

            inbatch_pad = padding_mask[:, :C]
            inbatch_scores = jnp.where(inbatch_pad, inbatch_scores, -INF)

        if N == 0:
            global_neg_scores = jnp.array([[0]])
        else:
            global_cand = local_candidate[:, C:, :].reshape((local_B * N, -1))
            if use_mol:
                global_neg_scores = _mol_pair_scores(local_user, global_cand, scorer)
            else:
                global_neg_scores = local_user @ global_cand.T

            global_pad = padding_mask[:, C:].reshape((1, local_B * N))
            global_neg_scores = jnp.where(global_pad, global_neg_scores, -INF)

        return inbatch_scores, global_neg_scores

    @shard_map(
        mesh=mesh,
        in_specs=(P(data_axis), P(data_axis), P(data_axis), P(data_axis)),
        out_specs=(P(data_axis), P(data_axis)),
        check_vma=False,
    )
    def _apply_logq_correction(
        local_batch: jax.Array,
        local_neg: jax.Array,
        local_batch_correction: jax.Array,
        global_batch_correction: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if use_in_batch_negatives:
            local_batch_correction = local_batch_correction.reshape((1, -1))
        local_batch += local_batch_correction

        if N > 0:
            local_neg += global_batch_correction.reshape((1, -1))

        return local_batch, local_neg

    candidate_padding_mask = batch["candidate_seq"]["post_hashes"][:, :, 0] != 0

    candidate_safety_mask = batch["candidate_seq"].get("safety_label_mask")
    retrieval_safety_stats = safety_filter_stats(
        candidate_safety_mask,
        candidate_padding_mask,
        bits=safety_filter_bits,
        prefix="safety_filter_candidates",
    )
    if safety_filter_apply_to_candidates and safety_filter_mode == "hard":
        candidate_padding_mask, _ = apply_safety_filter(
            candidate_safety_mask,
            candidate_padding_mask,
            None,
            mode="hard",
            bits=safety_filter_bits,
            soft_weight=safety_filter_soft_weight,
        )

    ad_mask_candidates = jnp.ones((B, C), dtype=jnp.bool_)
    if ads_only_candidates:
        promoted_ids = batch["candidate_seq"]["promoted_ids"]
        if promoted_ids is not None:
            ad_mask = promoted_ids != 0
            candidate_padding_mask = candidate_padding_mask & ad_mask
            ad_mask_candidates = ad_mask[:, :C]

    actions = batch["candidate_seq"]["actions"]
    assert actions is not None
    has_positive_actions = jnp.sum(actions[:, :C, positive_actions], axis=-1) > 0
    has_hard_negative_actions = jnp.sum(actions[:, :C, hard_negative_actions], axis=-1) > 0
    has_soft_negative_actions = jnp.sum(actions[:, :C, soft_negative_actions], axis=-1) > 0
    if enable_fake_positives:
        fake_positive_mask = jnp.sum(has_positive_actions, axis=-1, keepdims=True) == 0
        fake_positive_mask = jnp.concatenate(
            [fake_positive_mask, jnp.zeros((B, C - 1), dtype=jnp.bool_)], axis=-1
        )
        has_positive_actions = has_positive_actions | fake_positive_mask

    valid_positive_mask = (
        has_positive_actions
        & ~has_hard_negative_actions
        & ~has_soft_negative_actions
        & ad_mask_candidates
        & candidate_padding_mask[:, :C]
    ).astype(jnp.bool_)

    raw_batch_scores, raw_global_neg_scores = _sharded_full_matmul(
        user_representation,
        candidate_representation,
        candidate_padding_mask,
        mol_scorer if use_mol else (),
    )

    metrics = {}
    metrics.update(retrieval_safety_stats)
    metrics["safety_filter_active"] = jnp.float32(1.0 if safety_filter_mode != "off" else 0.0)

    warm_batch_scores = raw_batch_scores / temperature
    warm_global_neg_scores = raw_global_neg_scores / temperature
    metrics["temperature"] = temperature

    if logq_correction_scale > 0.0:
        tweet_counts = get_candidate_tweet_counts(
            cast_jax(batch["candidate_seq"]["post_hashes"]),
            log_q_num_bins=100_000_000,
            negative_sample_mask=jnp.ones((B, L), dtype=jnp.bool_),
        )
        sampling_weight = jnp.where(tweet_counts == 0.0, 1.0, 1.0 / tweet_counts)
        logq_correction = jnp.log(sampling_weight)

        batch_correction = logq_correction[:, :C] * logq_correction_scale
        global_correction = logq_correction[:, C:] * logq_correction_scale

        batch_scores, global_neg_scores = _apply_logq_correction(
            warm_batch_scores, warm_global_neg_scores, batch_correction, global_correction
        )

        metrics["logq_max"] = jnp.max(batch_correction)
        metrics["logq_min"] = jnp.min(batch_correction)
        metrics["logq_mean"] = jnp.mean(batch_correction)

        metrics["tweet_counts_max"] = jnp.max(tweet_counts[:, :C])
        metrics["tweet_counts_min"] = jnp.min(tweet_counts[:, :C])
        metrics["tweet_counts_mean"] = jnp.mean(tweet_counts[:, :C])

        metrics["logq_max_mask"] = jnp.max(
            batch_correction, where=candidate_padding_mask[:, :C], initial=-jnp.inf
        )
        metrics["logq_min_mask"] = jnp.min(
            batch_correction, where=candidate_padding_mask[:, :C], initial=jnp.inf
        )
        metrics["logq_mean_mask"] = jnp.mean(batch_correction, where=candidate_padding_mask[:, :C])

        metrics["tweet_counts_max_mask"] = jnp.max(
            tweet_counts[:, :C], where=candidate_padding_mask[:, :C], initial=0
        )
        metrics["tweet_counts_min_mask"] = jnp.min(
            tweet_counts[:, :C], where=candidate_padding_mask[:, :C], initial=1
        )
        metrics["tweet_counts_mean_mask"] = jnp.mean(
            tweet_counts[:, :C], where=candidate_padding_mask[:, :C]
        )
    else:
        batch_scores = warm_batch_scores
        global_neg_scores = warm_global_neg_scores

    @shard_map(
        mesh=mesh,
        in_specs=(P(data_axis), P(data_axis), P(data_axis), P(data_axis)),
        out_specs=(P(data_axis), P(data_axis), P(data_axis)),
        check_vma=False,
    )
    def _rearrange_negatives(
        batch_scores_shard: jax.Array,
        global_neg_scores_shard: jax.Array,
        cand_hash_shard: jax.Array,
        pos_mask_shard: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        b = batch_scores_shard.shape[0]
        n_false_neg = jnp.zeros((b,), dtype=jnp.float32)
        if not use_in_batch_negatives:
            return batch_scores_shard, global_neg_scores_shard, n_false_neg

        batch_scores_shard = batch_scores_shard.reshape((b, b, C))

        self_cands = batch_scores_shard[jnp.arange(b), jnp.arange(b)]

        off_diag_cols = (jnp.arange(b)[:, None] + jnp.arange(1, b)[None, :]) % b
        off_diag_cols = jnp.sort(off_diag_cols, axis=1)
        in_batch_negatives = batch_scores_shard[jnp.arange(b)[:, None], off_diag_cols].reshape(
            b, (b - 1) * C
        )
        if mask_in_batch_false_negatives:
            col_hash = jnp.broadcast_to(cand_hash_shard[None], (b, b, C))[
                jnp.arange(b)[:, None], off_diag_cols
            ].reshape(b, (b - 1) * C)
            same_post = jnp.any(
                (col_hash[:, None, :] == cand_hash_shard[:, :, None]) & pos_mask_shard[:, :, None],
                axis=1,
            )
            in_batch_negatives = jnp.where(same_post, -INF, in_batch_negatives)
            n_false_neg = jnp.sum(same_post.astype(jnp.float32), axis=-1)

        if N > 0:
            return (
                self_cands,
                jnp.concatenate([in_batch_negatives, global_neg_scores_shard], axis=1),
                n_false_neg,
            )
        else:
            return self_cands, in_batch_negatives, n_false_neg

    self_scores, common_neg_scores, n_false_neg = _rearrange_negatives(
        batch_scores,
        global_neg_scores,
        batch["candidate_seq"]["post_hashes"][:, :C, 0].astype(jnp.int32),
        valid_positive_mask,
    )
    assert isinstance(self_scores, jax.Array)
    assert isinstance(common_neg_scores, jax.Array)
    metrics["false_negative_cols_per_user"] = jnp.mean(n_false_neg)

    numerator_logits = self_scores

    true_neg_scores = jnp.where(
        has_hard_negative_actions | has_soft_negative_actions,
        self_scores + own_negative_logit_offset,
        -INF,
    )
    denominator_logits = jnp.concatenate([common_neg_scores, true_neg_scores], axis=-1)

    tiled_denominator_logits = jnp.tile(denominator_logits[:, None, :], (1, C, 1))
    tiled_denominator_logits = jnp.concatenate(
        [numerator_logits[:, :, None], tiled_denominator_logits], axis=-1
    )

    if apply_u2u_and_i2i_loss:

        @shard_map(
            mesh=mesh,
            in_specs=(P(data_axis), P(data_axis), P(data_axis), P(data_axis)),
            out_specs=(P(data_axis), P(data_axis)),
            check_vma=False,
        )
        def _sharded_u2u_and_i2i_matmul(
            local_user: jax.Array,
            local_candidate: jax.Array,
            padding_mask: jax.Array,
            numerator_logits: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            local_B = local_user.shape[0]
            user_user_logits = (
                jnp.where(
                    ~jnp.eye(local_B, dtype=jnp.bool_),
                    jnp.matmul(local_user, local_user.T),
                    -1e12,
                )
                .reshape(local_B, 1, local_B)
                .repeat(C, axis=1)
            )
            user_user_mask = user_user_logits < (0.1 + numerator_logits[:, :, None])
            user_user_logits = jnp.where(user_user_mask, user_user_logits / temperature, -1e12)

            item_item_logits = jnp.matmul(
                (local_candidate[:, :C, :].reshape(local_B * C, -1)),
                (local_candidate).reshape(local_B * L, -1).T,
            )
            item_item_mask = jnp.eye(local_B * C, local_B * L, dtype=jnp.bool_)
            row_padding_mask = padding_mask[:, :C].reshape(-1, 1)
            col_padding_mask = padding_mask.reshape(1, -1)
            item_item_logits = jnp.where(
                ~item_item_mask & row_padding_mask & ~col_padding_mask,
                item_item_logits / temperature,
                -1e12,
            )
            item_item_logits = item_item_logits.reshape(local_B, C, -1)

            return user_user_logits, item_item_logits

        user_user_logits, item_item_logits = _sharded_u2u_and_i2i_matmul(
            user_representation[:, 0] if user_representation.ndim == 3 else user_representation,
            candidate_representation,
            candidate_padding_mask,
            numerator_logits,
        )

        log_denominator = jax.scipy.special.logsumexp(
            jnp.concatenate(
                [tiled_denominator_logits, user_user_logits, item_item_logits], axis=-1
            ),
            axis=-1,
        )
    else:
        log_denominator = jax.scipy.special.logsumexp(tiled_denominator_logits, axis=-1)

    neg_log_probs = log_denominator - numerator_logits
    neg_log_likelihood = jnp.where(valid_positive_mask, neg_log_probs, 0.0)

    per_user_sum = jnp.sum(neg_log_likelihood, axis=-1)
    per_user_count = jnp.sum(valid_positive_mask.astype(jnp.float32), axis=-1)
    batch_mean = jnp.where(per_user_count > 0, per_user_sum / per_user_count, 0.0)

    num_valid_users = jnp.sum((per_user_count > 0).astype(jnp.float32))
    if positive_count_user_weight_power > 0.0:
        batch_mean = batch_mean * positive_count_user_weight(
            per_user_count, positive_count_user_weight_power
        )
    contrastive_loss = jnp.where(
        num_valid_users > 0,
        jnp.sum(batch_mean) / num_valid_users,
        0.0,
    )

    loss = contrastive_loss

    metrics.update(
        _compute_retrieval_metrics(
            C,
            N,
            raw_batch_scores,
            raw_global_neg_scores,
            self_scores,
            global_neg_scores,
            valid_positive_mask,
            has_soft_negative_actions,
            has_hard_negative_actions,
            candidate_padding_mask,
            actions,
            contrastive_loss,
            has_hard_negative_actions,
            positive_actions,
            hard_negative_actions,
            soft_negative_actions,
            use_in_batch_negatives,
            data_axis,
            mesh,
            debug_mode,
        )
    )

    return loss, metrics


@dataclass
class RecsysTwoTowerModel(hk.Module):
    config: RecsysTwoTowerModelConfig
    user_tower: RecsysAggregatedModel
    candidate_tower: RecsysCandidateTower
    sharding_context: ShardingContext

    @property
    def data_axis(self):
        return ("stage", *self.config.model_config.data_axis)

    @hk.transparent
    def build_user_inputs(
        self,
        recsys_features_batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        is_training: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        _config = self.config.user_tower_config
        assert _config.model_config.output_vocab_size is not None, "output_vocab_size is required"

        if _config.feature_prep_enabled:
            fp = _config.feature_prep
            scale_multiplier = fp.scale_config.input_scale(fp.emb_size)
            tokens, padding_mask, _ = build_feature_prep_inputs(
                batch=recsys_features_batch,
                recsys_embeddings=recsys_embeddings,
                config=fp,
                hash_keys=_config.hash_table.hash_keys,
                input_scale=scale_multiplier,
                output_vocab_size=_config.model_config.output_vocab_size,
                is_training=is_training,
                include_candidates=False,
            )
            tokens = with_sharding_constraint(
                tokens, P(self.user_tower.data_axis, ("seq", "model"))
            )
            return tokens, padding_mask

        history_actions = recsys_features_batch["history_seq"]["actions"]
        assert history_actions is not None
        self.action_embedding_table: jax.Array
        history_actions_embeddings, self.action_embedding_table = (
            self.user_tower.multi_hot_to_embeddings(
                cast_jax(history_actions),
                _config.model_config.output_vocab_size,
                _config.emb_table_width,
                _config.embed_init_scale,
                "action_embedding_table",
            )
        )

        ctx_config = _config.context_features
        history_cat_features: jax.Array | None = None
        if ctx_config.enabled:
            raw_hist_cat = recsys_features_batch["history_seq"].get("categorical_features")
            if raw_hist_cat is not None:
                history_cat_features = cast_jax(raw_hist_cat)

        history_unified_context = self.user_tower.build_unified_context_embedding(
            product_surface=cast_jax(recsys_features_batch["history_seq"]["product_surface"]),
            cat_features=history_cat_features,
        )

        _sid_post_emb_h = self._sid_token_or_zeros_init(
            recsys_features_batch["history_seq"], _config
        )

        history_embeddings, history_padding_mask = block_history_reduce(
            cast_jax(recsys_features_batch["history_seq"]["post_hashes"]),
            cast_jax(recsys_features_batch["history_seq"]["auth_hashes"]),
            recsys_embeddings.history_post_embeddings if _config.use_post_embedding else None,
            recsys_embeddings.history_author_embeddings,
            history_unified_context,
            history_actions_embeddings,
            _config.hash_table.hash_keys,
            _config.model_config.scale_config.emb_lr_multiplier,
            _config.embed_init_scale,
            history_unified_context is not None,
            sid_post_embeddings=_sid_post_emb_h,
        )

        if _config.use_seqpack:
            layout = recsys_features_batch.get("packing_layout")
            assert layout is not None, (
                "use_seqpack=True but batch has no packing_layout; "
                "run pack_batch on eval/training batches before the model."
            )
            num_devices, bs_per_device, _ = recsys_features_batch["user_hashes"].shape
            total_batch = num_devices * bs_per_device

            def _flatten_lead(x):
                return None if x is None else cast_jax(x).reshape(total_batch, *x.shape[2:])

            user_emb_result = self.user_tower._build_user_embedding(
                cast_jax(recsys_features_batch["user_hashes"]),
                recsys_embeddings,
                reshape_for_seqpack=(num_devices, bs_per_device),
            )
            user_embeddings = user_emb_result[0] if user_emb_result is not None else None

            uf = self.config.user_features
            user_features_token = None
            if uf.has_user_features:
                flat_user_feature_keys = (
                    "user_categorical_features",
                    "user_bool_features",
                    "user_float_features",
                    "user_int64_features",
                    "user_installed_apps_multihot",
                )
                flat_batch = typing.cast(
                    RecsysFeaturesBatch,
                    {
                        **recsys_features_batch,
                        **{
                            k: _flatten_lead(recsys_features_batch[k])
                            for k in flat_user_feature_keys
                        },
                    },
                )
                user_features_token = build_user_features_token(
                    build_user_feature_parts(flat_batch, uf, self.sharding_context),
                    uf.user_features_concat_dim,
                    _config.emb_table_width,
                    DTYPE_BY_NAME[_config.fprop_dtype],
                    self.sharding_context,
                    use_mlp=uf.user_features_mlp,
                    pad=uf.user_features_concat_pad,
                ).reshape(num_devices, bs_per_device, -1)

            padding_mask = cast_jax(layout.padding_mask)
            seq_starts = cast_jax(layout.cu_seqlens[:, :-1])
            device_idx = jnp.arange(num_devices, dtype=jnp.int32)[:, None]

            embeddings = jnp.zeros(
                (*padding_mask.shape, _config.emb_table_width),
                dtype=DTYPE_BY_NAME[_config.fprop_dtype],
            )

            prefix_offset = 0
            if user_embeddings is not None:
                embeddings = embeddings.at[device_idx, seq_starts].set(user_embeddings)
                prefix_offset = 1
            if user_features_token is not None:
                embeddings = embeddings.at[device_idx, seq_starts + prefix_offset].set(
                    user_features_token
                )
            embeddings = embeddings.at[device_idx, cast_jax(layout.history_positions)].add(
                jnp.where(history_padding_mask[:, :, None], history_embeddings, 0)
            )
            embeddings *= self.config.model_config.scale_config.input_scale(
                self.config.emb_table_width
            )
            embeddings = with_sharding_constraint(
                embeddings, P(self.user_tower.data_axis, ("seq", "model"))
            )
            return self.user_tower.maybe_tfmr_project_embeddings(embeddings, _config), padding_mask

        sequence_parts: list[jax.Array] = []
        mask_parts: list[jax.Array] = []

        user_emb_result = self.user_tower._build_user_embedding(
            cast_jax(recsys_features_batch["user_hashes"]),
            recsys_embeddings,
        )
        if user_emb_result is not None:
            user_emb, user_mask = user_emb_result
            assert user_mask is not None
            sequence_parts.append(user_emb)
            mask_parts.append(user_mask)

        uf = self.config.user_features
        user_features_token = None
        if uf.has_user_features:
            feature_parts = build_user_feature_parts(
                recsys_features_batch, uf, self.sharding_context
            )
            user_features_token = build_user_features_token(
                feature_parts,
                uf.user_features_concat_dim,
                _config.emb_table_width,
                DTYPE_BY_NAME[_config.fprop_dtype],
                self.sharding_context,
                use_mlp=uf.user_features_mlp,
                pad=uf.user_features_concat_pad,
            )
        if user_features_token is not None:
            sequence_parts.append(user_features_token)
            mask_parts.append(jnp.ones((history_embeddings.shape[0], 1), dtype=jnp.bool_))

        sequence_parts.append(history_embeddings)
        mask_parts.append(history_padding_mask)

        embeddings = jnp.concatenate(sequence_parts, axis=1)

        embeddings *= self.config.model_config.scale_config.input_scale(self.config.emb_table_width)
        embeddings = with_sharding_constraint(
            embeddings, P(self.user_tower.data_axis, ("seq", "model"))
        )
        padding_mask = jnp.concatenate(mask_parts, axis=1)

        embeddings = self.user_tower.maybe_tfmr_project_embeddings(
            embeddings, self.config.user_tower_config
        )
        return embeddings, padding_mask

    @hk.transparent
    def _build_post_sid_token(
        self,
        post_sids: jax.Array,
        post_hashes: jax.Array,
    ) -> jax.Array:
        _config = self.config.user_tower_config
        needs_hashes = _config.sid_hash_level
        return embed_entity_sid(
            cast_jax(post_sids),
            _config.emb_table_width,
            _config.sid_embed_dim,
            _config.sid_num_levels,
            _config.sid_codebook_size,
            _config.model_config.scale_config.emb_lr_multiplier,
            _config.embed_init_scale,
            DTYPE_BY_NAME[_config.fprop_dtype],
            "post",
            sid_hash_level=_config.sid_hash_level,
            entity_hashes=cast_jax(post_hashes) if needs_hashes else None,
            sid_cross_attn=_config.sid_cross_attn,
            sid_prefix2_embed_dim=_config.sid_prefix2_embed_dim,
            sid_prefix3_embed_dim=_config.sid_prefix3_embed_dim,
            sid_prefix3_rows=_config.sid_prefix3_rows,
        )

    def _sid_token_or_zeros_init(self, seq, _config) -> jax.Array | None:
        if not _config.use_post_sid:
            return None
        sids = seq.get("post_sids")
        if sids is None:
            hashes = cast_jax(seq["post_hashes"])
            sids = jnp.zeros(
                (hashes.shape[0], hashes.shape[1], _config.sid_num_levels), dtype=jnp.uint16
            )
        return self._build_post_sid_token(sids, seq["post_hashes"])

    def _candidate_feature_tensors(
        self,
        recsys_features_batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> tuple[jax.Array | None, jax.Array, jax.Array | None]:
        _config = self.config.user_tower_config
        candidate_post_hashes = recsys_features_batch["candidate_seq"]["post_hashes"]
        B, L, _ = candidate_post_hashes.shape
        num_item_hashes = _config.hash_table.hash_keys.num_item_hashes
        num_author_hashes = _config.hash_table.hash_keys.num_author_hashes

        post_embeddings: jax.Array | None = None
        if _config.use_post_embedding:
            assert recsys_embeddings.candidate_post_embeddings is not None
            post_embeddings = recsys_embeddings.candidate_post_embeddings.reshape(
                B, L, num_item_hashes, -1
            )

        author_embeddings = recsys_embeddings.candidate_author_embeddings.reshape(
            B, L, num_author_hashes, -1
        )

        sid_embedding = self._sid_token_or_zeros_init(
            recsys_features_batch["candidate_seq"], _config
        )
        return post_embeddings, author_embeddings, sid_embedding

    def _stack_candidate_tokens(
        self,
        post_embeddings: jax.Array | None,
        author_embeddings: jax.Array,
        sid_embedding: jax.Array | None,
    ) -> jax.Array:
        token_parts: list[jax.Array] = []
        if post_embeddings is not None:
            token_parts.append(post_embeddings)
        token_parts.append(author_embeddings)
        if sid_embedding is not None:
            token_parts.append(sid_embedding[..., None, :])
        return jnp.concatenate(token_parts, axis=-2)

    def build_candidate_embeddings(
        self,
        recsys_features_batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        head_index: int = 0,
    ) -> jax.Array:
        _config = self.config.user_tower_config
        assert _config.model_config.output_vocab_size is not None, "output_vocab_size is required"
        assert _config.use_post_embedding or _config.use_post_sid, (
            "At least one of use_post_embedding / use_post_sid must be True for the candidate tower"
        )

        post_embeddings, author_embeddings, sid_embedding = self._candidate_feature_tensors(
            recsys_features_batch, recsys_embeddings
        )
        stacked = self._stack_candidate_tokens(post_embeddings, author_embeddings, sid_embedding)
        return self.candidate_tower(stacked, head_index=head_index)

    def build_candidate_embeddings_from_lookups(
        self,
        post_author_embedding: jax.Array,
        post_sids: jax.Array | None,
        post_hashes: jax.Array | None,
        head_index: int = 0,
    ) -> jax.Array:
        _config = self.config.user_tower_config
        num_item_hashes = _config.hash_table.hash_keys.num_item_hashes
        num_author_hashes = _config.hash_table.hash_keys.num_author_hashes

        expected_hash_tokens = (
            num_item_hashes + num_author_hashes if _config.use_post_embedding else num_author_hashes
        )
        total_tokens = post_author_embedding.shape[-2]
        assert total_tokens == expected_hash_tokens, (
            f"expected {expected_hash_tokens} hash tokens on axis -2, got {total_tokens}"
        )

        if _config.use_post_sid:
            assert post_sids is not None, (
                "use_post_sid=True but post_sids was not passed to "
                "build_candidate_embeddings_from_lookups"
            )
            sid_embedding = self._build_post_sid_token(
                post_sids, post_hashes if post_hashes is not None else jnp.zeros_like(post_sids)
            )
            post_author_embedding = jnp.concatenate(
                [post_author_embedding, sid_embedding[..., None, :]], axis=-2
            )

        return self.candidate_tower(post_author_embedding, head_index=head_index)

    def __call__(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        attn_mask: jax.Array | None = None,
        is_training: bool = True,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        stats = {}

        user_embeddings, user_padding_mask = self.build_user_inputs(
            batch, recsys_embeddings, is_training=is_training
        )

        if self.config.use_seqpack:
            layout = batch.get("packing_layout")
            assert layout is not None, (
                "use_seqpack=True but batch has no packing_layout; "
                "run pack_batch on eval/training batches before the model."
            )
            bs_per_device = batch["user_hashes"].shape[1]

            assert self.config.user_tower_config.model_config.attn_config.attn_impl in (
                "pallas_ranker_varlen_attn",
                "cutedsl_ranker_varlen_attn",
            )
            user_outputs, _, _ = self.user_tower(
                user_embeddings,
                user_padding_mask,
                is_training=is_training,
                positions=cast_jax(layout.positions),
                segment_ids=cast_jax(layout.segment_ids),
                seqpack_layout=layout,
            )

            data_axis = self.data_axis

            @shard_map(
                mesh=self.sharding_context.mesh,
                in_specs=(P(data_axis), P(data_axis), P(data_axis)),
                out_specs=P(data_axis),
                check_vma=False,
            )
            def _pool(outputs, mask, cu_seqlens):
                rows, packed_seq_len, emb_dim = outputs.shape
                positions = jnp.arange(packed_seq_len, dtype=cu_seqlens.dtype)
                local_user_ids = jnp.clip(
                    jax.vmap(lambda starts: jnp.searchsorted(starts, positions, side="right") - 1)(
                        cu_seqlens
                    ),
                    0,
                    bs_per_device - 1,
                )
                seg_ids = (
                    jnp.arange(rows, dtype=cu_seqlens.dtype)[:, None] * bs_per_device
                    + local_user_ids
                ).reshape(-1)
                valid = mask.astype(jnp.float32)
                num_segments = rows * bs_per_device
                summed = jax.ops.segment_sum(
                    (outputs.astype(jnp.float32) * valid[:, :, None]).reshape(-1, emb_dim),
                    seg_ids,
                    num_segments=num_segments,
                )
                counts = jax.ops.segment_sum(valid.reshape(-1), seg_ids, num_segments=num_segments)
                return summed / jnp.maximum(counts[:, None], 1.0)

            if self.config.user_query_heads > 0:
                user_representation = self._packed_query_heads(
                    user_outputs,
                    cast_jax(layout.padding_mask),
                    cast_jax(layout.cu_seqlens),
                    bs_per_device,
                )
            else:
                user_representation = _pool(
                    user_outputs, cast_jax(layout.padding_mask), cast_jax(layout.cu_seqlens)
                )

            user_representation = with_sharding_constraint(
                user_representation,
                P(self.data_axis, None)
                if user_representation.ndim == 2
                else P(self.data_axis, None, None),
            )
        else:
            user_embeddings, user_padding_mask, *_ = pad_to_next_128_multiple(
                user_embeddings,
                user_padding_mask,
                jnp.zeros_like(user_padding_mask),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

            B, T = user_padding_mask.shape
            user_segment_ids = user_tower_segment_ids(
                B,
                T,
                use_history_segment_ids=self.config.use_history_segment_ids,
                padding_mask=user_padding_mask,
            )

            if self.config.user_tower_config.right_anchored_rope:
                user_positions = right_anchored_rope_positions(
                    user_padding_mask,
                    self.config.user_tower_config.history_seq_len,
                    self.config.num_user_prefix_tokens,
                )
            else:
                user_positions = jnp.full((B, T, 3), 0, dtype=jnp.float32)
                user_positions = user_positions.at[:, :, 0].set(jnp.arange(start=0, stop=T))
            user_outputs, _, _ = self.user_tower(
                user_embeddings,
                user_padding_mask,
                is_training=is_training,
                positions=user_positions,
                segment_ids=user_segment_ids.astype(jnp.int32),
            )

            user_mask_float = user_padding_mask.astype(jnp.float32)
            user_embeddings_masked = user_outputs * user_mask_float[:, :, None]
            user_embedding_sum = jnp.sum(user_embeddings_masked, axis=1)
            user_mask_sum = jnp.sum(user_mask_float, axis=1, keepdims=True)
            user_representation = user_embedding_sum / jnp.maximum(user_mask_sum, 1.0)
            if self.config.user_query_heads > 0:
                from_end = jnp.cumsum(user_mask_float[:, ::-1], axis=1)[:, ::-1]
                user_representation = self._query_heads_pool(
                    user_outputs, user_mask_float, from_end - 1.0
                )

        num_heads = self.config.candidate_tower_config.num_candidate_heads
        all_candidate_reprs: list[jax.Array] = []
        for h in range(num_heads):
            cand_repr = self.build_candidate_embeddings(batch, recsys_embeddings, head_index=h)
            if self.config.use_seqpack:
                per_user_C = cand_repr.shape[1] // batch["user_hashes"].shape[1]
                cand_repr = cand_repr.reshape(-1, per_user_C, cand_repr.shape[-1])
                cand_repr = with_sharding_constraint(cand_repr, P(self.data_axis, None, None))
            all_candidate_reprs.append(cand_repr)

        user_norm_sq = jnp.sum(user_representation**2, axis=-1, keepdims=True)
        user_norm = jnp.sqrt(jnp.maximum(user_norm_sq, EPS))
        user_representation = user_representation / user_norm
        stats["two_tower_user_pre_norm"] = user_norm.mean()

        return user_representation, all_candidate_reprs, stats

    def _query_head_params(self, emb_dim: int) -> tuple[jax.Array, jax.Array]:
        num_heads = self.config.user_query_heads
        std = self.config.user_query_head_init_std
        q = get_parameter(
            "user_query_head_q",
            shape=[num_heads, emb_dim],
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(std) if std > 0 else hk.initializers.Constant(0.0),
            pspec=P(),
        )
        gamma = get_parameter(
            "user_query_head_recency_gamma",
            shape=[num_heads],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(),
        )
        return q, gamma

    def _query_head_logits(
        self, outputs: jax.Array, from_newest: jax.Array, q: jax.Array, gamma: jax.Array
    ) -> jax.Array:
        scale = 1.0 / math.sqrt(outputs.shape[-1])
        content = jnp.einsum("...td,kd->...tk", outputs.astype(jnp.float32), q) * scale
        recency = from_newest[..., None] / QUERY_HEAD_RECENCY_SCALE_TOKENS
        return content - gamma * recency

    def _query_heads_pool(
        self, outputs: jax.Array, valid: jax.Array, from_newest: jax.Array
    ) -> jax.Array:
        q, gamma = self._query_head_params(outputs.shape[-1])
        logits = self._query_head_logits(outputs, from_newest, q, gamma)
        logits = jnp.where(valid[:, :, None] > 0, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=1) * valid[:, :, None]
        return jnp.einsum("btk,btd->bkd", weights, outputs.astype(jnp.float32))

    def _packed_query_heads(
        self,
        outputs: jax.Array,
        padding_mask: jax.Array,
        cu_seqlens: jax.Array,
        bs_per_device: int,
    ) -> jax.Array:
        q, gamma = self._query_head_params(outputs.shape[-1])
        K = self.config.user_query_heads
        data_axis = self.data_axis

        @shard_map(
            mesh=self.sharding_context.mesh,
            in_specs=(P(data_axis), P(data_axis), P(data_axis), P(), P()),
            out_specs=P(data_axis),
            check_vma=False,
        )
        def _pool_heads(outputs, mask, cu_seqlens, q, gamma):
            rows, packed_len, emb_dim = outputs.shape
            positions = jnp.arange(packed_len, dtype=cu_seqlens.dtype)
            local_user_ids = jnp.clip(
                jax.vmap(lambda starts: jnp.searchsorted(starts, positions, side="right") - 1)(
                    cu_seqlens
                ),
                0,
                bs_per_device - 1,
            )
            seg_end = jnp.take_along_axis(cu_seqlens, local_user_ids + 1, axis=1)
            from_newest = (seg_end - positions[None, :] - 1).astype(jnp.float32)
            seg_ids = (
                jnp.arange(rows, dtype=cu_seqlens.dtype)[:, None] * bs_per_device + local_user_ids
            ).reshape(-1)
            num_segments = rows * bs_per_device
            valid = mask.reshape(-1).astype(jnp.float32)
            logits = self._query_head_logits(outputs, from_newest, q, gamma).reshape(-1, K)
            logits = jnp.where(valid[:, None] > 0, logits, -1e9)
            seg_max = jax.ops.segment_max(logits, seg_ids, num_segments=num_segments)
            weights = jnp.exp(logits - seg_max[seg_ids]) * valid[:, None]
            seg_sum = jax.ops.segment_sum(weights, seg_ids, num_segments=num_segments)
            weights = weights / jnp.maximum(seg_sum[seg_ids], 1e-30)
            flat = outputs.astype(jnp.float32).reshape(-1, emb_dim)
            pooled = [
                jax.ops.segment_sum(
                    flat * weights[:, k : k + 1], seg_ids, num_segments=num_segments
                )
                for k in range(K)
            ]
            return jnp.stack(pooled, axis=1)

        return _pool_heads(outputs, padding_mask, cu_seqlens, q, gamma)

    def _mol_scorer(self) -> MolScorer | None:
        num_components = self.config.mol_item_components
        if num_components == 0:
            return None
        num_mix = max(1, self.config.user_query_heads) * num_components
        emb_dim = self.config.candidate_tower_config.emb_table_width
        rank = self.config.mol_item_adapter_rank
        hidden = self.config.mol_gate_hidden
        conditioned = self.config.mol_conditioned_gate
        zeros = hk.initializers.Constant(0.0)

        def param(name, shape, init):
            return get_parameter(name, shape=shape, dtype=jnp.float32, init=init, pspec=P())

        return MolScorer(
            adapter_v=param(
                "mol_item_adapter_v",
                [num_components, emb_dim, rank],
                hk.initializers.RandomNormal(1.0 / math.sqrt(emb_dim)),
            ),
            adapter_u=param("mol_item_adapter_u", [num_components, rank, emb_dim], zeros),
            gate_w1=param(
                "mol_gate_w1",
                [num_mix, hidden],
                hk.initializers.RandomNormal(1.0 / math.sqrt(num_mix)),
            ),
            gate_b1=param("mol_gate_b1", [hidden], zeros),
            gate_w2=param("mol_gate_w2", [hidden, num_mix], zeros),
            gate_b2=param("mol_gate_b2", [num_mix], zeros),
            gate_user_w=param("mol_gate_user_w", [emb_dim, num_mix], zeros)
            if conditioned
            else None,
            gate_item_w=param("mol_gate_item_w", [emb_dim, num_mix], zeros)
            if conditioned
            else None,
        )

    def mol_side_table_rows(self, post_table: jax.Array) -> jax.Array:
        scorer = self._mol_scorer()
        assert scorer is not None
        return scorer.side_table_rows(post_table)

    def mol_side_tables(self, post_table: jax.Array) -> tuple[jax.Array, jax.Array]:
        scorer = self._mol_scorer()
        assert scorer is not None
        mesh = self.sharding_context.mesh
        saxis = "expert"
        return shard_map(
            scorer.sharded_serving_side_tables,
            mesh=mesh,
            in_specs=(P(saxis),),
            out_specs=(P(None, saxis), P(None, saxis)),
            check_vma=False,
        )(post_table)

    @hk.transparent
    def forward(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        post_embeddings: jax.Array,
        dataset_types: jax.Array | None,
        top_k: int = 0,
        target_dataset_types: tuple[int, ...] = (1,),
        eligible_mask: jax.Array | None = None,
        topic_bitmaps: jax.Array | None = None,
        topic_user_bitmasks: jax.Array | None = None,
        eval_bs_per_device: int = 0,
        dataset_ranges: tuple[tuple[int, int], ...] | jax.Array | None = None,
        use_async_topk: bool = False,
        use_radix_select_topk: bool = False,
        post_scales: jax.Array | None = None,
        dataset_capacities: tuple[int, ...] | None = None,
        mol_side_tables: tuple[jax.Array, jax.Array] | None = None,
        return_validity: bool = False,
    ) -> tuple[tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array], ...]:
        saxis = "expert"

        user_representation, _, _ = self(
            batch=batch,
            recsys_embeddings=recsys_embeddings,
            is_training=False,
        )

        if post_embeddings is not None:
            mesh = self.sharding_context.mesh

        skip_dataset_mask = os.environ.get("DEBUG_ALLOW_RANDOM_INIT") == "1"

        def local_top_k(scores: jax.Array, k: int):
            N = scores.shape[1]
            k_actual = min(k, N)

            return top_k_by_key(
                scores,
                k_actual,
                heuristic_pivot_ratio=0.1,
                use_async=use_async_topk,
                use_radix_select=use_radix_select_topk,
            )

        use_topic_filter = topic_bitmaps is not None and topic_user_bitmasks is not None

        def _trim_eval_users(user_embedding: jax.Array) -> jax.Array:
            if eval_bs_per_device > 0:
                users_per_expert = user_embedding.shape[0] // mesh.shape[saxis]
                max_eval_users = eval_bs_per_device * mesh.shape[saxis]
                if eval_bs_per_device < users_per_expert:
                    rest = user_embedding.shape[1:]
                    user_embedding = user_embedding.reshape(
                        mesh.shape[saxis], users_per_expert, *rest
                    )[:, :eval_bs_per_device].reshape(max_eval_users, *rest)
            return user_embedding

        mol_scorer = self._mol_scorer()
        use_mol = mol_scorer is not None
        use_mol_kernel = use_mol and self.config.mol_serving_kernel

        side_spec = P(None, saxis) if mol_side_tables is not None else P()
        side_args = mol_side_tables if mol_side_tables is not None else ((), ())

        def _side(norms, ibias):
            return (norms, ibias) if mol_side_tables is not None else None

        def _score_posts(
            user_embedding: jax.Array, posts: jax.Array, scorer, side=None
        ) -> jax.Array:
            if use_mol:
                users = user_embedding if user_embedding.ndim == 3 else user_embedding[:, None, :]
                if use_mol_kernel:
                    return mol_kernel_scores(scorer, users, posts, side_tables=side).astype(
                        jnp.bfloat16
                    )
                return scorer.chunked_pair_scores(
                    users, posts.astype(jnp.bfloat16), self.config.mol_eval_user_chunk
                )
            return jnp.matmul(user_embedding.astype(jnp.bfloat16), posts.T)

        @shard_map(
            mesh=mesh,
            in_specs=(P(saxis), P(), P(), side_spec, side_spec),
            out_specs=P(),
            check_vma=False,
        )
        def compute_top_k(
            post_embeddings: jax.Array, user_embedding: jax.Array, scorer, norms, ibias
        ) -> jax.Array:
            user_embedding = _trim_eval_users(user_embedding)
            scores = _score_posts(user_embedding, post_embeddings, scorer, _side(norms, ibias))
            scores = jax.lax.all_to_all(
                scores, axis_name=saxis, split_axis=0, concat_axis=1, tiled=True
            )
            return scores

        mask_in_spec = P(saxis) if eligible_mask is not None else P()

        topic_bitmaps_in_spec = P(saxis, None) if use_topic_filter else P()
        topic_user_bitmasks_in_spec = P(None, None) if use_topic_filter else P()

        @shard_map(
            mesh=mesh,
            in_specs=(P(), P(), mask_in_spec, topic_bitmaps_in_spec, topic_user_bitmasks_in_spec),
            out_specs=(P(), P(), P()) if return_validity else (P(), P()),
            check_vma=False,
        )
        def mask_and_top_k(
            all_scores: jax.Array,
            type_mask: jax.Array,
            user_eligible_mask: jax.Array,
            topic_bitmaps_shard: jax.Array,
            topic_user_bitmasks_full: jax.Array,
        ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
            combined = type_mask & user_eligible_mask

            if use_topic_filter:
                topic_bitmaps_full = jax.lax.all_gather(
                    topic_bitmaps_shard, axis_name=saxis, axis=0, tiled=True
                )
                B_local = combined.shape[0]
                shard_idx = jax.lax.axis_index(saxis)
                start_idx = shard_idx * B_local
                user_bitmasks_local = jax.lax.dynamic_slice(
                    topic_user_bitmasks_full, (start_idx, 0), (B_local, NUM_TOPIC_INT32S)
                )
                topic_mask = jnp.any(
                    (topic_bitmaps_full[None, :, :] & user_bitmasks_local[:, None, :]) != 0,
                    axis=-1,
                )
                no_filter_mask = jnp.all(user_bitmasks_local == 0, axis=-1)[:, None]
                topic_mask = jnp.where(no_filter_mask, True, topic_mask)
                combined = combined & topic_mask

            masked_scores = jnp.where(combined, all_scores, jnp.finfo(jnp.bfloat16).min)
            sorted_scores, sorted_indices = local_top_k(masked_scores, top_k)
            top_k_scores = jax.lax.all_gather(sorted_scores, axis_name=saxis, axis=0, tiled=True)
            top_k_indices = jax.lax.all_gather(sorted_indices, axis_name=saxis, axis=0, tiled=True)
            if return_validity:
                sorted_validity = gather_selected_validity(combined, sorted_indices)
                top_k_validity = jax.lax.all_gather(
                    sorted_validity, axis_name=saxis, axis=0, tiled=True
                )
                return top_k_scores, top_k_indices, top_k_validity
            return top_k_scores, top_k_indices

        if post_scales is not None:

            @shard_map(
                mesh=mesh,
                in_specs=(P(saxis), P(), P(saxis), P(), side_spec, side_spec),
                out_specs=P(),
                check_vma=False,
            )
            def compute_top_k_int8(
                post_q8: jax.Array,
                user_embedding: jax.Array,
                scales: jax.Array,
                scorer,
                norms,
                ibias,
            ) -> jax.Array:
                user_embedding = _trim_eval_users(user_embedding)
                if use_mol_kernel:
                    users = (
                        user_embedding if user_embedding.ndim == 3 else user_embedding[:, None, :]
                    )
                    scores = mol_kernel_scores(
                        scorer,
                        users,
                        post_q8,
                        post_scales=scales,
                        side_tables=_side(norms, ibias),
                    ).astype(jnp.bfloat16)
                    return jax.lax.all_to_all(
                        scores, axis_name=saxis, split_axis=0, concat_axis=1, tiled=True
                    )
                if use_mol:
                    posts = post_q8.astype(jnp.float32) * scales[:, None]
                    scores = _score_posts(user_embedding, posts, scorer)
                    return jax.lax.all_to_all(
                        scores, axis_name=saxis, split_axis=0, concat_axis=1, tiled=True
                    )
                uf = user_embedding.astype(jnp.float32)
                us = jnp.maximum(jnp.max(jnp.abs(uf), axis=1) / 127.0, 1e-12)
                uq = jnp.clip(jnp.round(uf / us[:, None]), -127, 127).astype(jnp.int8)
                acc = jax.lax.dot_general(
                    uq, post_q8, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32
                )
                scores = (acc.astype(jnp.float32) * us[:, None] * scales[None, :]).astype(
                    jnp.bfloat16
                )
                return jax.lax.all_to_all(
                    scores, axis_name=saxis, split_axis=0, concat_axis=1, tiled=True
                )

            all_scores = compute_top_k_int8(
                post_embeddings,
                user_representation,
                post_scales,
                mol_scorer if use_mol else (),
                *side_args,
            )
        else:
            all_scores = compute_top_k(
                post_embeddings, user_representation, mol_scorer if use_mol else (), *side_args
            )

        no_per_post_filter = (
            eligible_mask is None and not use_topic_filter and not skip_dataset_mask
        )
        if dataset_capacities is not None:
            assert isinstance(dataset_ranges, jax.Array) and no_per_post_filter, (
                "dataset_capacities need dataset_ranges as an array input and no per-post filter"
            )
            assert len(dataset_capacities) == len(target_dataset_types), (
                f"dataset_capacities ({len(dataset_capacities)}) must align with "
                f"target_dataset_types ({len(target_dataset_types)})"
            )
            results = []
            for d, capacity in enumerate(dataset_capacities):

                @shard_map(mesh=mesh, in_specs=(P(), P()), out_specs=(P(), P()), check_vma=False)
                def window_and_top_k(all_scores, ranges, _d=d, _w=capacity):
                    start, end = ranges[_d, 0], ranges[_d, 1]
                    start_c = jnp.minimum(start, all_scores.shape[1] - _w)
                    window = jax.lax.dynamic_slice_in_dim(all_scores, start_c, _w, axis=1)
                    col = start_c + jnp.arange(_w, dtype=jnp.int32)
                    masked = jnp.where(
                        ((col >= start) & (col < end))[None, :],
                        window,
                        jnp.finfo(all_scores.dtype).min,
                    )
                    sorted_scores, sorted_indices = local_top_k(masked, top_k)
                    sorted_indices = sorted_indices + start_c
                    return (
                        jax.lax.all_gather(sorted_scores, axis_name=saxis, axis=0, tiled=True),
                        jax.lax.all_gather(sorted_indices, axis_name=saxis, axis=0, tiled=True),
                    )

                top_k_scores, top_k_indices = window_and_top_k(all_scores, dataset_ranges)
                if return_validity:
                    start, end = dataset_ranges[d, 0], dataset_ranges[d, 1]
                    top_k_validity = (top_k_indices >= start) & (top_k_indices < end)
                    results.append(
                        (top_k_indices, top_k_scores.astype(jnp.float32), top_k_validity)
                    )
                else:
                    results.append((top_k_indices, top_k_scores.astype(jnp.float32)))
            return tuple(results)

        use_slice_path = dataset_ranges is not None and no_per_post_filter
        if use_slice_path:
            assert len(dataset_ranges) == len(target_dataset_types), (
                f"dataset_ranges ({len(dataset_ranges)}) must align with "
                f"target_dataset_types ({len(target_dataset_types)})"
            )
            results = []
            for start, end in dataset_ranges:

                @shard_map(mesh=mesh, in_specs=(P(),), out_specs=(P(), P()), check_vma=False)
                def slice_and_top_k(all_scores, _start=start, _end=end):
                    sliced = jax.lax.slice_in_dim(all_scores, _start, _end, axis=1)
                    sorted_scores, sorted_indices = local_top_k(sliced, top_k)
                    sorted_indices = sorted_indices + _start
                    top_k_scores = jax.lax.all_gather(
                        sorted_scores, axis_name=saxis, axis=0, tiled=True
                    )
                    top_k_indices = jax.lax.all_gather(
                        sorted_indices, axis_name=saxis, axis=0, tiled=True
                    )
                    return top_k_scores, top_k_indices

                top_k_scores, top_k_indices = slice_and_top_k(all_scores)
                if return_validity:
                    top_k_validity = jnp.ones_like(top_k_indices, dtype=jnp.bool_)
                    results.append(
                        (top_k_indices, top_k_scores.astype(jnp.float32), top_k_validity)
                    )
                else:
                    results.append((top_k_indices, top_k_scores.astype(jnp.float32)))
            return tuple(results)

        B = all_scores.shape[0]
        M = post_embeddings.shape[0]
        if eligible_mask is None:
            user_eligible_mask = jnp.ones((B, M), dtype=jnp.bool_)
        else:
            user_eligible_mask = eligible_mask

        results = []
        for ds_type in target_dataset_types:
            if dataset_types is not None and not skip_dataset_mask:
                stored = dataset_types.squeeze()
                match = jnp.asarray(RetrievalDataset.mask_values(ds_type), dtype=stored.dtype)
                type_mask = jnp.isin(stored, match)
            else:
                type_mask = jnp.ones(post_embeddings.shape[0], dtype=jnp.bool_)

            _topic_bitmaps = topic_bitmaps if use_topic_filter else jnp.zeros((), dtype=jnp.int32)
            _topic_user_bitmasks = (
                topic_user_bitmasks if use_topic_filter else jnp.zeros((), dtype=jnp.int32)
            )
            selected = mask_and_top_k(
                all_scores, type_mask, user_eligible_mask, _topic_bitmaps, _topic_user_bitmasks
            )
            top_k_scores, top_k_indices = selected[:2]
            if return_validity:
                results.append((top_k_indices, top_k_scores.astype(jnp.float32), selected[2]))
            else:
                results.append((top_k_indices, top_k_scores.astype(jnp.float32)))

        return tuple(results)

    @hk.transparent
    def user_embeddings_only(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        is_training: bool = False,
    ) -> jax.Array:
        user_representation, _, _ = self(
            batch=batch,
            recsys_embeddings=get_recsys_embed_param_to_jax_array(recsys_embeddings),
            is_training=is_training,
        )
        return user_representation

    @hk.transparent
    def loss(
        self,
        inter,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        is_training: bool = True,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del inter

        with Summary() as summarizer:
            user_representation, all_candidate_reprs, training_stats = self(
                batch=batch,
                recsys_embeddings=get_recsys_embed_param_to_jax_array(recsys_embeddings),
                attn_mask=None,
                is_training=is_training,
            )
        stats = summarizer.get()
        stats.update(training_stats)

        temperature = get_parameter(
            "temperature",
            shape=[],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.1),
            pspec=P(),
        )
        if self.config.fixed_temperature > 0.0:
            temperature = jnp.asarray(self.config.fixed_temperature, dtype=jnp.float32)

        loss_batch = batch
        if self.config.use_seqpack:
            num_devices = batch["user_hashes"].shape[0]
            bs_per_device = batch["user_hashes"].shape[1]
            packed_C = batch["candidate_seq"]["post_hashes"].shape[1]
            per_user_C = packed_C // bs_per_device
            new_leading = num_devices * bs_per_device
            loss_batch = typing.cast(
                RecsysFeaturesBatch,
                {
                    **batch,
                    "candidate_seq": {
                        k: v
                        if (v is None or v.ndim < 2 or v.shape[1] != packed_C)
                        else v.reshape(new_leading, per_user_C, *v.shape[2:])
                        for k, v in batch["candidate_seq"].items()
                    },
                },
            )

        per_head_actions = self.config.get_per_head_actions()
        num_heads = len(all_candidate_reprs)
        head_names = self.config.head_names

        total_loss = jnp.array(0.0)
        mol_scorer = self._mol_scorer()
        for h in range(num_heads):
            pos_actions, hard_neg_actions, soft_neg_actions = per_head_actions[h]
            head_loss, head_metrics = compute_retrieval_loss(
                loss_batch,
                user_representation,
                all_candidate_reprs[h],
                temperature,
                self.data_axis,
                self.sharding_context.mesh,
                self.config.num_global_negatives_per_example if is_training else 0,
                self.config.user_tower_config.candidate_seq_len,
                self.config.use_in_batch_negatives,
                pos_actions,
                hard_neg_actions,
                soft_neg_actions,
                self.config.logq_correction_scale,
                self.config.enable_fake_positives,
                self.config.debug_mode,
                self.config.apply_u2u_and_i2i_loss,
                self.config.ads_only_candidates,
                safety_filter_mode=self.config.safety_filter_mode,
                safety_filter_bits=self.config.safety_filter_bits,
                safety_filter_soft_weight=self.config.safety_filter_soft_weight,
                safety_filter_apply_to_candidates=self.config.safety_filter_apply_to_candidates,
                own_negative_logit_offset=self.config.own_negative_logit_offset,
                positive_count_user_weight_power=self.config.positive_count_user_weight_power,
                mol_scorer=mol_scorer,
                mol_train_user_chunk=self.config.mol_train_user_chunk,
                mask_in_batch_false_negatives=self.config.mask_in_batch_false_negatives,
            )
            total_loss = total_loss + head_loss
            if num_heads > 1:
                for k, v in head_metrics.items():
                    stats[f"{head_names[h]}/{k}"] = v
            else:
                stats.update(head_metrics)

        loss = total_loss
        stats["origin-loss"] = loss

        stats = self.user_tower.compute_sid_metrics(batch=batch, stats=stats)

        regularization_loss = (
            loss + self.config.user_tower_config.act_l2_weight * stats["act-l2-loss"]
        )
        return (
            regularization_loss,
            stats,
        )


@configclass
class RecsysTwoTowerModelConfig(Config):
    user_tower_config: RecsysAggregatedModelConfig
    candidate_tower_config: RecsysCandidateModelConfig

    use_in_batch_negatives: bool = True

    logq_correction_scale: float = 2.0

    fixed_temperature: float = 0.0
    own_negative_logit_offset: float = 0.0
    positive_count_user_weight_power: float = 0.0

    enable_fake_positives: bool = False

    num_global_negatives_per_example: int = 0

    debug_mode: bool = False

    apply_u2u_and_i2i_loss: bool = False

    ads_only_candidates: bool = False

    use_history_segment_ids: bool = False

    multimodal_embedding_type: EmbeddingType | None = None

    user_features: UserFeaturesConfig = UserFeaturesConfig()

    user_query_heads: int = 0
    user_query_head_init_std: float = 0.0
    mol_item_components: int = 0
    mol_item_adapter_rank: int = 64
    mol_gate_hidden: int = 32
    mol_conditioned_gate: bool = False
    mol_train_user_chunk: int = 0
    mol_eval_user_chunk: int = 32
    mol_serving_kernel: bool = False
    mol_side_table_in_checkpoint: bool = False
    mask_in_batch_false_negatives: bool = False

    positive_actions: list[int] = field(
        default_factory=lambda: [
            recsys_pb2.ActionName.SERVER_TWEET_FAV,
        ]
    )

    hard_negative_actions: list[int] = field(
        default_factory=lambda: [
            recsys_pb2.ActionName.CLIENT_TWEET_REPORT,
            recsys_pb2.ActionName.CLIENT_TWEET_NOT_INTERESTED_IN,
            recsys_pb2.ActionName.CLIENT_TWEET_SEE_FEWER,
            recsys_pb2.ActionName.CLIENT_TWEET_UNFOLLOW_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_BLOCK_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_MUTE_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_NOT_RELEVANT,
        ]
    )

    soft_negative_actions: list[int] = field(
        default_factory=lambda: [
            recsys_pb2.ActionName.CLIENT_TWEET_RECAP_NOT_DWELLED,
        ]
    )

    checkpoint_dataset_names: list[str] | None = None

    split_home_checkpoint: bool = False
    cold_start_max_age_seconds: float = DEFAULT_COLD_START_MAX_AGE_SECONDS

    immersive_positive_actions: list[int] = field(
        default_factory=lambda: [
            recsys_pb2.ActionName.SERVER_TWEET_FAV,
            recsys_pb2.ActionName.SERVER_TWEET_REPLY,
            recsys_pb2.ActionName.SERVER_TWEET_QUOTE,
            recsys_pb2.ActionName.SERVER_TWEET_RETWEET,
            recsys_pb2.ActionName.CLIENT_TWEET_VIDEO_QUALITY_VIEW,
            recsys_pb2.ActionName.CLIENT_TWEET_FOLLOW_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_BOOKMARK,
            recsys_pb2.ActionName.CLIENT_TWEET_SHARE,
        ]
    )
    immersive_hard_negative_actions: list[int] = field(
        default_factory=lambda: [
            recsys_pb2.ActionName.CLIENT_TWEET_REPORT,
            recsys_pb2.ActionName.CLIENT_TWEET_NOT_INTERESTED_IN,
            recsys_pb2.ActionName.CLIENT_TWEET_SEE_FEWER,
            recsys_pb2.ActionName.CLIENT_TWEET_UNFOLLOW_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_BLOCK_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_MUTE_AUTHOR,
            recsys_pb2.ActionName.CLIENT_TWEET_NOT_RELEVANT,
        ]
    )
    immersive_soft_negative_actions: list[int] = field(
        default_factory=lambda: [
            recsys_pb2.ActionName.CLIENT_TWEET_RECAP_NOT_DWELLED,
        ]
    )

    head_dataset_mapping: dict[str, int] | None = None

    head_names: list[str] = field(default_factory=lambda: ["home"])

    @property
    def mol_side_table_width(self) -> int:
        wanted = self.mol_serving_kernel or self.mol_side_table_in_checkpoint
        if not (wanted and self.mol_item_components > 0):
            return 0
        return self.mol_item_components * (1 + max(self.user_query_heads, 1))

    @classmethod
    def get_positive_actions(cls) -> list[int]:
        return cls.__dataclass_fields__["positive_actions"].default_factory()

    @classmethod
    def get_hard_negative_actions(cls) -> list[int]:
        return cls.__dataclass_fields__["hard_negative_actions"].default_factory()

    @classmethod
    def get_soft_negative_actions(cls) -> list[int]:
        return cls.__dataclass_fields__["soft_negative_actions"].default_factory()

    def get_per_head_actions(self) -> list[tuple[list[int], list[int], list[int]]]:
        configs = [(self.positive_actions, self.hard_negative_actions, self.soft_negative_actions)]
        if self.candidate_tower_config.num_candidate_heads > 1:
            configs.append(
                (
                    self.immersive_positive_actions,
                    self.immersive_hard_negative_actions,
                    self.immersive_soft_negative_actions,
                )
            )
        return configs

    embed_memory_kind: MemoryKind = "device"

    @property
    def use_ip_address(self) -> bool:
        return self.user_tower_config.use_ip_address

    @property
    def use_user_embedding(self) -> bool:
        return self.user_tower_config.use_user_embedding

    @property
    def use_post_embedding(self) -> bool:
        return self.user_tower_config.use_post_embedding

    @property
    def emb_table_width(self) -> int:
        return self.user_tower_config.emb_table_width

    safety_filter_mode: Literal["off", "hard", "soft"] = "off"
    safety_filter_bits: int = 0b11
    safety_filter_soft_weight: float = 0.0
    safety_filter_apply_to_candidates: bool = True

    @property
    def hash_table(self):
        return self.user_tower_config.hash_table

    @property
    def model_config(self):
        return self.user_tower_config.model_config

    @property
    def use_seqpack(self) -> bool:
        return self.user_tower_config.use_seqpack

    @property
    def num_user_prefix_tokens(self) -> int:
        return self.user_tower_config.num_user_prefix_tokens

    @property
    def sequence_len(self):
        return self.user_tower_config.model_config.sequence_len

    def compute_mfu(self, num_seq_per_sec_per_device):
        return self.user_tower_config.compute_mfu(num_seq_per_sec_per_device)

    def compute_tflops(self, num_seq_per_sec_per_device):
        return self.user_tower_config.compute_tflops(num_seq_per_sec_per_device)

    def get_embed_memory_kind(self):
        return self.embed_memory_kind

    @property
    def embedding_dtype(self):
        return self.user_tower_config.embedding_dtype

    def make_embedding_table(self, rng, optim, input_vocab_size: int, init_opt_state: bool):
        return self.user_tower_config.make_embedding_table(
            rng, optim, input_vocab_size, init_opt_state
        )

    def make(self, sharding_context: ShardingContext):
        if self.use_seqpack:
            attn_config = self.user_tower_config.model_config.attn_config
            assert self.user_tower_config.right_anchored_rope
            assert (
                isinstance(attn_config, RecsysAttentionConfig)
                and attn_config.num_user_prefix_tokens == self.num_user_prefix_tokens
            )

        user_tower = self.user_tower_config.make(sharding_context)

        candidate_tower = self.candidate_tower_config.make(
            sharding_context,
            use_post_embedding=self.user_tower_config.use_post_embedding,
            use_post_sid=self.user_tower_config.use_post_sid,
            use_project_then_sum=self.user_tower_config.feature_prep_enabled,
        )

        return RecsysTwoTowerModel(
            config=self,
            user_tower=user_tower,
            candidate_tower=candidate_tower,
            sharding_context=sharding_context,
        )
