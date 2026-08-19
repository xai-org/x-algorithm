# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
import math
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec as P
from numpy import typing as npt
from xai_configlib import Config as Config
from xai_configlib import configclass as configclass
from xai_proto import recsys_pb2

from xrex.data.recsys.constants import (
    CLICK_ACTION_INDEX,
    CLICK_CONDITIONED_ACTION_INDICES,
    NEGATIVE_FEEDBACK_HEAD_INDICES,
    action_type_map,
    engagement_to_ids,
)
from xrex.data.recsys.feature_config import ENGAGEMENT_COUNT_BUCKET_MAP, CategoricalFeature
from xrex.data.recsys.recsys_batch import EMBEDDING_CONFIG, EmbeddingType, RecsysFeaturesBatch
from xrex.data.recsys.safety_filter import apply_safety_filter, safety_filter_stats
from xrex.data.recsys.sequence_packing import SequencePackedLayout
from xrex.models.layers import Linear, get_parameter
from xrex.models.loss_recsys import (
    continuous_loss_compute,
    multihot_loss_compute,
    tweedie_loss_compute,
)
from xrex.models.model_utils import Parameter
from xrex.models.normalization import rms_norm_fn
from xrex.models.recsys_attention import RecsysAttentionConfig
from xrex.models.recsys_embedding import (
    HashKeys,
    HashTable,
    RecsysEmbeddings,
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_feature_prep import (
    FeaturePrepConfig,
    build_feature_prep_inputs,
)
from xrex.models.recsys_user_features import (
    UserFeaturesConfig,
    build_user_feature_parts,
    build_user_features_token,
)
from xrex.models.sharding_context import ShardingContext
from xrex.models.transformer import (
    Transformer,
    TransformerConfig,
)
from xrex.utils.utils import Summary, dump_to_file

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


DTYPE_BY_NAME = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}

_CLAMP_eps = 1e-7
_INITIAL_AUC_THRESHOLD = 1e-5
_CALIB_POS_BASE = 1e-4
_NUM_AUC_THRESHOLDS = 256

IOS_CLIENT_APP_IDS = (
    129032,
    191841,
    1082764,
    557701,
)

ANDROID_CLIENT_APP_IDS = (
    258901,
    640835,
    4959597,
    5778172,
)


POST_AGE_MAX_MINUTES = 4800


def compute_post_age_bucket(
    impr_ts_sec: jax.Array,
    post_creation_ts_sec: jax.Array,
    granularity_mins: int = 60,
    max_mins: int = POST_AGE_MAX_MINUTES,
    strategy: Literal["linear", "log1p"] = "linear",
    num_buckets: int = 80,
) -> jax.Array:
    post_age_minutes = (impr_ts_sec - post_creation_ts_sec) // 60

    if strategy == "linear":
        n_buckets = max_mins // granularity_mins
        bucket = (post_age_minutes // granularity_mins) + 1
    elif strategy == "log1p":
        n_buckets = num_buckets
        log_max = math.log1p(max_mins)
        clamped_age = jnp.maximum(post_age_minutes, 0).astype(jnp.float32)
        bucket = jnp.floor(jnp.log1p(clamped_age) / log_max * n_buckets).astype(jnp.int32) + 1
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    overflow_bucket = n_buckets + 1
    bucket = jnp.clip(bucket, 0, overflow_bucket)

    bucket = jnp.where(
        (post_age_minutes < 0) | (impr_ts_sec == 0) | (post_creation_ts_sec == 0),
        0,
        bucket,
    )
    return bucket.astype(jnp.int32)


ENGAGEMENT_COUNT_NUM_BUCKETS = 32
ENGAGEMENT_COUNT_MAX_BUCKET: dict[CategoricalFeature, int] = {
    CategoricalFeature.favCountBucketSeq: 18,
    CategoricalFeature.replyCountBucketSeq: 13,
    CategoricalFeature.repostCountBucketSeq: 15,
    CategoricalFeature.quoteCountBucketSeq: 13,
    CategoricalFeature.viewCountBucketSeq: 25,
}


def compute_engagement_count_bucket(
    raw_counts: jax.Array, max_bucket: int = ENGAGEMENT_COUNT_NUM_BUCKETS - 1
) -> jax.Array:
    counts_f = jnp.maximum(raw_counts.astype(jnp.float32), 1.0)
    bucket = jnp.floor(jnp.log2(counts_f)).astype(jnp.int32) + 1
    bucket = jnp.clip(bucket, 0, max_bucket)
    bucket = jnp.where(raw_counts <= 0, 0, bucket)
    return bucket


def right_anchored_rope_positions(
    padding_mask: jax.Array, history_seq_len: int, num_user_prefix_tokens: int
):
    history_start = num_user_prefix_tokens
    history_end = num_user_prefix_tokens + history_seq_len

    idx = jnp.arange(padding_mask.shape[1], dtype=jnp.int32)[None, :]
    history_len = padding_mask[:, history_start:history_end].sum(axis=1, dtype=jnp.int32)

    positions = jnp.where(
        (history_start <= idx) & (idx < history_end),
        history_end - history_len[:, None] + idx - history_start,
        idx,
    )

    positions = jnp.where(idx >= history_end, history_end, positions)
    positions = jnp.where(padding_mask, positions, 0).astype(jnp.float32)

    return jnp.concatenate(
        [positions[..., None], jnp.zeros((*positions.shape, 2), positions.dtype)],
        axis=-1,
    )


def packed_per_user_history_event_count(
    history_actions: jax.Array,
    history_post_hashes: jax.Array,
    history_slot_len: jax.Array,
) -> jax.Array:
    has_event = jnp.any(history_actions, axis=-1) | (history_post_hashes[:, :, 0] != 0)
    has_event = has_event.astype(jnp.int32)
    slot_starts = jnp.cumsum(history_slot_len, axis=1) - history_slot_len
    pos = jnp.arange(history_post_hashes.shape[1], dtype=jnp.int32)[None, :, None]
    starts = slot_starts[:, None, :]
    ends = starts + history_slot_len[:, None, :]
    in_slot = (pos >= starts) & (pos < ends)
    return jnp.sum(has_event[:, :, None] * in_slot, axis=1)


@configclass
class NormConfig(Config):
    norm_scale: float = 30.0

    use_log: bool = False


@configclass
class ContinuousActionLossConfig(Config):
    action_index: int = 0

    loss_weight: float = 0.0

    loss_type: Literal["mse", "mae", "huber", "tweedie"] = "mae"

    tweedie_power: float = 1.5

    activation: Literal["sigmoid", "softplus"] | None = None

    output_cap: float = -1.0

    norm_config: NormConfig = NormConfig()

    metric_name: str | None = None

    mask_negatives: bool = True

    product_surfaces: tuple[int, ...] = ()

    exclude_product_surfaces: tuple[int, ...] = ()

    def __post_init__(self):
        if self.metric_name is None:
            self.metric_name = (
                recsys_pb2.ContinuousActionName.Name(self.action_index).lower().replace("_", "-")
            )
        if self.activation is None:
            if self.loss_type in ("mse", "mae", "huber"):
                self.activation = "sigmoid"
            elif self.loss_type == "tweedie":
                self.activation = "softplus"


def _get_surface_mask(
    loss_config: ContinuousActionLossConfig,
    product_surface: jnp.ndarray,
) -> jnp.ndarray:
    if loss_config.product_surfaces:
        return jnp.isin(product_surface, jnp.array(loss_config.product_surfaces))
    elif loss_config.exclude_product_surfaces:
        return ~jnp.isin(product_surface, jnp.array(loss_config.exclude_product_surfaces))
    else:
        return jnp.ones(product_surface.shape, dtype=jnp.bool_)


MemoryKind = Literal["unpinned_host", "device"]


@configclass
class CategoricalFeatureConfig(Config):
    index: int = 0

    feature_name: str = ""

    cardinality: int = 1

    embedding_dim: int = 0

    embedding_name: str = ""

    @property
    def resolved_embedding_name(self) -> str:
        return self.embedding_name if self.embedding_name else f"ctx_cat_{self.feature_name}_emb"


@configclass
class ContextFeaturesConfig(Config):
    enabled: bool = False

    categorical_features: list[CategoricalFeatureConfig] = field(default_factory=list)

    enable_engagement_counts: bool = False

    enable_author_nsfw: bool = False


def metric_num_tokens(y: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    return (valid_mask * y).sum()


def metric_loss(p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    def _compute_loss(y, p, valid_mask, total_valid):
        p_c = jnp.clip(p, _CLAMP_eps, 1.0 - _CLAMP_eps)
        ce_example = -(y * jnp.log(p_c) + (1 - y) * jnp.log(1 - p_c))
        return (valid_mask * ce_example).sum() / jnp.maximum(total_valid, 1)

    total_valid = valid_mask.sum()
    return jnp.where(
        total_valid > 0,
        _compute_loss(y, p, valid_mask, total_valid),
        0.0,
    )


def metric_rce(p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    def _compute_rce(y, p, valid_mask, num_pos, total_valid):
        p_bar = num_pos / jnp.maximum(total_valid, 1)
        p_bar_c = jnp.clip(p_bar, _CLAMP_eps, 1.0 - _CLAMP_eps)
        BCE_ref = -(p_bar_c * jnp.log(p_bar_c) + (1 - p_bar_c) * jnp.log(1 - p_bar_c))
        p_c = jnp.clip(p, _CLAMP_eps, 1.0 - _CLAMP_eps)
        ce_example = -(y * jnp.log(p_c) + (1 - y) * jnp.log(1 - p_c))
        CE_model = (valid_mask * ce_example).sum() / jnp.maximum(total_valid, 1)
        return 100.0 * (BCE_ref - CE_model) / (BCE_ref + 1e-12)

    num_pos = (valid_mask * y).sum()
    num_neg = valid_mask.sum() - num_pos
    total_valid = valid_mask.sum()
    return jnp.where(
        (num_pos > 0) & (num_neg > 0),
        _compute_rce(y, p, valid_mask, num_pos, total_valid),
        0.0,
    )


def metric_prauc(
    p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray, auc_thresholds: jnp.ndarray
) -> jnp.ndarray:
    valid_y = y * valid_mask
    num_pos = jnp.sum(valid_y)

    def compute_pr_at_threshold(threshold):
        pred = p >= threshold
        sum_pred = jnp.sum(pred * valid_mask)
        sum_pred_y = jnp.sum(pred * valid_y)
        precision = sum_pred_y / jnp.maximum(sum_pred, 1e-6)
        recall = sum_pred_y / jnp.maximum(num_pos, 1e-6)
        return precision, recall

    precisions, recalls = jax.vmap(compute_pr_at_threshold)(auc_thresholds)
    sorted_indices = jnp.argsort(recalls)
    sorted_precisions = precisions[sorted_indices]
    sorted_recalls = recalls[sorted_indices]
    recall_diff = jnp.diff(sorted_recalls, prepend=0.0)
    num_neg = valid_mask.sum() - num_pos
    return ((num_pos > 0) & (num_neg > 0)) * jnp.sum(sorted_precisions * recall_diff)


def metric_ratio_pos(p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    del p
    total_valid = valid_mask.sum()
    num_pos = (valid_mask * y).sum()
    return jax.lax.cond(
        total_valid > 0,
        lambda _, n=num_pos, t=total_valid: n / jnp.maximum(t, 1),
        lambda _: 0.0,
        None,
    )


def metric_ndcg(p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    def compute_single_ndcg(p_row: jnp.ndarray, y_row: jnp.ndarray, mask_row: jnp.ndarray):
        valid_p = p_row * mask_row
        valid_y = y_row * mask_row

        masked_p = jnp.where(mask_row > 0, valid_p, -jnp.inf)

        pred_order = jnp.argsort(-masked_p)
        sorted_y = valid_y[pred_order]

        seq_len = mask_row.shape[0]
        positions = jnp.arange(1, seq_len + 1, dtype=jnp.float32)
        discounts = jnp.log2(positions + 1)

        dcg = jnp.sum(sorted_y / discounts)

        idcg = jnp.sum(jnp.sort(valid_y, descending=True) / discounts)

        return jnp.where(idcg > 0, dcg / idcg, 1.0)

    ndcg_scores = jax.vmap(compute_single_ndcg)(p, y, valid_mask)

    valid_queries = jnp.sum(valid_mask, axis=1) >= 1
    return jax.lax.cond(
        jnp.sum(valid_queries) > 0,
        lambda _, scores=ndcg_scores, valid=valid_queries: (
            jnp.sum(jnp.where(valid, scores, 0.0)) / jnp.sum(valid)
        ),
        lambda _: 0.0,
        None,
    )


def metric_calib(p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    num_pos = (valid_mask * y).sum() + _CALIB_POS_BASE
    num_prob_pos = (valid_mask * p).sum()
    return jax.lax.cond(
        num_pos > 0,
        lambda _, num_prob_pos=num_prob_pos, num_pos=num_pos: (
            num_prob_pos / jnp.maximum(num_pos, 1)
        ),
        lambda _: 0.0,
        None,
    )


def engagement_metrics(p, y, masks, auc_thresholds):
    return jnp.array(
        [
            jax.vmap(f, in_axes=(None, None, 0))(p, y, masks)
            for f in (metric_loss, metric_rce, metric_ndcg, metric_ratio_pos, metric_calib)
        ]
        + [jax.vmap(metric_prauc, in_axes=(None, None, 0, None))(p, y, masks, auc_thresholds)]
        + [jax.vmap(metric_num_tokens, in_axes=(None, 0))(y, masks)]
    )


def get_probs_and_labels(
    logits: jax.Array, raw_targets: jax.Array, metric_group: str = "default"
) -> dict[str, tuple[jax.Array, jax.Array]]:
    logits = logits.astype(jnp.float32)
    probs = jax.nn.sigmoid(logits)
    prob_labels = {}
    eng_to_ids = engagement_to_ids(metric_group)
    for eng_name, client_event_list in eng_to_ids.items():
        id_array = jnp.array(client_event_list)
        y = jnp.any(raw_targets[..., id_array] == 1, axis=-1).astype(jnp.float32)
        p = probs[..., id_array].max(axis=-1)
        p = jnp.clip(p, _CLAMP_eps, 1.0 - _CLAMP_eps)
        prob_labels[eng_name] = (p, y)
    return prob_labels


def metrics_calc(
    p_list: list[jax.Array],
    y_list: list[jax.Array],
    eng_names: list[str],
    mask_keys: list[str],
    mask_values: list[jax.Array],
    stats: dict,
):
    auc_thresholds = jnp.geomspace(_INITIAL_AUC_THRESHOLD, 1.0, num=_NUM_AUC_THRESHOLDS)
    res = jax.vmap(engagement_metrics, in_axes=(0, 0, None, None))(
        jnp.stack(p_list), jnp.stack(y_list), jnp.stack(mask_values), auc_thresholds
    )
    stats.update(
        {
            f"{eng_name}_{mask_key}_{name}": res[i][j][k]
            for i, eng_name in enumerate(eng_names)
            for j, name in enumerate(
                ("loss", "RCE", "NDCG", "ratio_pos", "calib", "PRAUC", "num_tokens")
            )
            for k, mask_key in enumerate(mask_keys)
        }
    )


def metrics_calc_global_num_tokens(masks: dict[str, jax.Array], stats: dict):
    mask_keys = []
    mask_values = []
    for mask_key, mask_value in masks.items():
        mask_keys.append(mask_key)
        mask_values.append(mask_value)
    res = jax.vmap(lambda x: x.sum(), in_axes=(0,))(jnp.stack(mask_values))
    stats.update(
        {
            f"{mask_key}_{name}": res[i]
            for i, mask_key in enumerate(mask_keys)
            for name in ("num_tokens",)
        }
    )


@configclass
class RecsysAggregatedModelConfig(Config):
    model_config: TransformerConfig
    hash_table: HashTable
    pad_token: int
    emb_table_width: int

    continuous_metrics_mae_mean: bool = False
    act_l2_weight: float = 0.0

    num_continuous_actions: int = 8

    continuous_action_losses: list[ContinuousActionLossConfig] = field(
        default_factory=lambda: [
            ContinuousActionLossConfig(
                action_index=recsys_pb2.ContinuousActionName.DWELL_TIME,
                loss_weight=0.1,
                loss_type="mae",
                norm_config=NormConfig(norm_scale=30.0, use_log=False),
            )
        ]
    )

    continuous_action_hidden_dim: int = 64

    final_logit_cap: float = -1.0
    embed_init_scale: float = 1.0
    fprop_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    embed_memory_kind: MemoryKind = "device"
    log_q_num_bins: int = 100_000_000

    log_q_correction: bool = False
    history_seq_len: int = 1024
    candidate_seq_len: int = 128

    right_anchored_rope: bool = False

    use_seqpack: bool = False

    effective_sequence_len: int | None = None

    transformer_output_only: bool = False
    use_product_surface: bool = False

    post_age_granularity_mins: int = 60

    post_age_max_mins: int = POST_AGE_MAX_MINUTES

    post_age_bucket_strategy: Literal["linear", "log1p"] = "linear"

    post_age_num_buckets: int = 80

    mask_candidate_positive_when_negative_action_present: bool = False

    metric_group: str = "default"

    multimodal_embedding_type: EmbeddingType | None = None

    search_query_embedding_dim: int = 0

    use_post_sid: bool = False
    use_post_embedding: bool = True
    sid_embed_dim: int = 1024
    sid_num_levels: int = 6
    sid_codebook_size: int = 1024
    sid_hash_level: bool = False
    sid_cross_attn: bool = False

    use_user_embedding: bool = True

    use_ip_address: bool = False

    feature_prep_enabled: bool = False
    feature_prep: FeaturePrepConfig = FeaturePrepConfig()

    concat_history_bridge_prob: bool = False

    mask_neg_feedback_on_negatives: bool = True

    condition_conversion_on_click: bool = False

    enable_platform_metrics: bool = False

    user_features: UserFeaturesConfig = UserFeaturesConfig()

    context_features: ContextFeaturesConfig = ContextFeaturesConfig()

    unified_context_dwell_time_dim: int = 64

    safety_filter_mode: Literal["off", "hard", "soft"] = "off"
    safety_filter_bits: int = 0b11
    safety_filter_soft_weight: float = 0.0
    safety_filter_apply_to_candidates: bool = True
    safety_filter_apply_to_history: bool = False

    @property
    def multimodal_embedding_dim(self) -> int:
        if self.multimodal_embedding_type is None:
            return 0
        return EMBEDDING_CONFIG[self.multimodal_embedding_type][1]

    def make(self, sharding_context: ShardingContext) -> RecsysAggregatedModel:
        if self.feature_prep_enabled:
            fp = self.feature_prep
            for name, fp_val, parent_val in (
                ("emb_size", fp.emb_size, self.model_config.emb_size),
                ("emb_table_width", fp.emb_table_width, self.emb_table_width),
                ("scale_config", fp.scale_config, self.model_config.scale_config),
                (
                    "multimodal_embedding_dim",
                    fp.multimodal_embedding_dim,
                    self.multimodal_embedding_dim,
                ),
                (
                    "search_query_embedding_dim",
                    fp.search_query_embedding_dim,
                    self.search_query_embedding_dim,
                ),
                ("enable_post_embedding", fp.enable_post_embedding, self.use_post_embedding),
                ("enable_user_embedding", fp.enable_user_embedding, self.use_user_embedding),
            ):
                assert fp_val == parent_val, (
                    f"feature_prep.{name}={fp_val!r} disagrees with parent {parent_val!r}; "
                    "build feature_prep via _make_feature_prep_config or set them consistently."
                )

        if self.use_seqpack:
            attn_config = self.model_config.attn_config
            assert self.right_anchored_rope
            assert (
                isinstance(attn_config, RecsysAttentionConfig)
                and attn_config.num_user_prefix_tokens == self.num_user_prefix_tokens
            )

        return RecsysAggregatedModel(
            model=self.model_config.make(sharding_context=sharding_context),
            config=self,
            sharding_context=sharding_context,
        )

    def get_embed_memory_kind(self) -> MemoryKind:
        return self.embed_memory_kind

    @property
    def embedding_dtype(self):
        return DTYPE_BY_NAME[self.fprop_dtype]

    def make_embedding_table(self, rng, optim, input_vocab_size: int, init_opt_state: bool):
        init_scale = 1.0
        init = jax.nn.initializers.variance_scaling(
            scale=init_scale,
            mode="fan_out",
            distribution="normal",
            dtype=self.embedding_dtype,
        )
        emb_table = Parameter(
            x=init(rng, (input_vocab_size, self.emb_table_width)),
            pspec=P(None, ("expert")),
            rms_clip_axes=(-2, -1),
        )
        if init_opt_state:
            emb_table_state = optim.init({"table": emb_table})
        else:
            emb_table_state = None

        return emb_table, emb_table_state

    def compute_tflops(self, num_seq_per_sec: float) -> float:
        S = (
            self.effective_sequence_len
            if self.effective_sequence_len is not None
            else self.sequence_len
        )
        return self.model_config.compute_tflops(num_seq_per_sec, S)

    def compute_mfu(self, num_seq_per_sec: float) -> float:
        S = (
            self.effective_sequence_len
            if self.effective_sequence_len is not None
            else self.sequence_len
        )
        return self.model_config.compute_mfu(num_seq_per_sec, S)

    @property
    def num_user_prefix_tokens(self) -> int:
        if self.feature_prep_enabled:
            return int(self.feature_prep.enable_user_embedding) + int(
                self.feature_prep.has_user_features
            )
        return int(self.use_user_embedding) + int(self.user_features.has_user_features)

    @property
    def sequence_len(self):
        return self.model_config.sequence_len

    @property
    def num_layers(self) -> int:
        return self.model_config.num_layers

    @property
    def num_kv_heads(self) -> int:
        assert self.model_config.attn_config is not None
        return self.model_config.attn_config.num_kv_heads

    @property
    def key_size(self) -> int:
        assert self.model_config.attn_config is not None
        return self.model_config.attn_config.key_size


def get_candidate_tweet_counts(
    data: RecsysFeaturesBatch, log_q_num_bins: int, negative_sample_mask: jnp.ndarray
) -> jax.Array:
    hashed_item_ids_flat = data["candidate_seq"]["post_hashes"][:, :, 0] % log_q_num_bins

    hashed_item_ids_masked_flat = hashed_item_ids_flat * negative_sample_mask

    counts = jnp.bincount(
        hashed_item_ids_masked_flat.ravel(),
        length=log_q_num_bins,
        minlength=log_q_num_bins,
    )
    tweet_counts_flat = counts[hashed_item_ids_masked_flat]
    tweet_counts_flat = jnp.where(negative_sample_mask, tweet_counts_flat, 0)
    return tweet_counts_flat


def normalize_continuous_value(
    values: jnp.ndarray,
    config: NormConfig,
) -> jnp.ndarray:
    values_clamped = jnp.clip(values, 0.0, config.norm_scale)

    if config.use_log:
        return jnp.log1p(values_clamped) / jnp.log1p(config.norm_scale)
    else:
        return values_clamped / config.norm_scale


def compute_continuous_metrics(
    gt_raw: jnp.ndarray,
    gt_clamped: jnp.ndarray,
    pred_in_original_units: jnp.ndarray,
    valid_mask: jnp.ndarray,
    loss_mask: jnp.ndarray,
    loss: jnp.ndarray,
    loss_weight: float,
    metric_name: str,
    mask_suffix: str,
    stats: dict[str, jnp.ndarray],
    mean_baseline: bool = False,
) -> None:
    name = f"{metric_name}_{mask_suffix}" if mask_suffix else metric_name

    num_loss_samples = jnp.sum(loss_mask)

    mae_raw = jnp.sum(jnp.abs(pred_in_original_units - gt_clamped) * loss_mask) / jnp.maximum(
        num_loss_samples, 1.0
    )

    loss_mask_values = jnp.where(loss_mask, gt_clamped, jnp.nan)
    batch_baseline = (
        jnp.nanmean(loss_mask_values) if mean_baseline else jnp.nanmedian(loss_mask_values)
    )
    batch_baseline = jnp.where(jnp.isnan(batch_baseline), 0.0, batch_baseline)

    mae_baseline = jnp.sum(jnp.abs(batch_baseline - gt_clamped) * loss_mask) / jnp.maximum(
        num_loss_samples, 1.0
    )

    rmae = 100.0 * (mae_baseline - mae_raw) / (mae_baseline + 1e-12)

    gt_f32 = gt_clamped.astype(jnp.float32)
    pred_f32 = pred_in_original_units.astype(jnp.float32)
    mask_f32 = loss_mask.astype(jnp.float32)
    n = jnp.sum(mask_f32)
    gt_mean = jnp.sum(gt_f32 * mask_f32) / jnp.maximum(n, 1.0)
    ss_res = jnp.sum(((gt_f32 - pred_f32) ** 2) * mask_f32)
    ss_tot = jnp.sum(((gt_f32 - gt_mean) ** 2) * mask_f32)
    r_squared = jnp.where(
        (n > 1) & (ss_tot > 0.0),
        jnp.clip(100.0 * (1.0 - ss_res / ss_tot), -100.0, 100.0),
        0.0,
    )

    sum_actual = jnp.sum(gt_clamped * valid_mask)
    sum_pred = jnp.sum(pred_in_original_units * valid_mask)
    calib = sum_pred / (sum_actual + 1e-12)

    stats[f"{name}-loss"] = loss
    stats[f"{name}-loss-weighted"] = loss_weight * loss
    stats[f"{name}-mae"] = mae_raw
    stats[f"{name}-baseline-mae"] = mae_baseline
    stats[f"{name}-rmae"] = rmae
    stats[f"{name}-r-squared"] = r_squared
    stats[f"{name}-calib"] = calib
    stats[f"{name}-num-loss-samples"] = num_loss_samples

    stats[f"{name}-gt-min"] = jnp.where(
        n > 0, jnp.min(jnp.where(loss_mask, gt_clamped, jnp.inf)), 0.0
    )
    stats[f"{name}-gt-mean"] = jnp.where(n > 0, gt_mean, 0.0)
    stats[f"{name}-gt-max"] = jnp.where(
        n > 0, jnp.max(jnp.where(loss_mask, gt_clamped, -jnp.inf)), 0.0
    )
    pred_mean = jnp.sum(pred_f32 * mask_f32) / jnp.maximum(n, 1.0)
    stats[f"{name}-pred-min"] = jnp.where(
        n > 0, jnp.min(jnp.where(loss_mask, pred_in_original_units, jnp.inf)), 0.0
    )
    stats[f"{name}-pred-mean"] = jnp.where(n > 0, pred_mean, 0.0)
    stats[f"{name}-pred-max"] = jnp.where(
        n > 0, jnp.max(jnp.where(loss_mask, pred_in_original_units, -jnp.inf)), 0.0
    )


def project_continuous_value_to_embedding(
    values: jnp.ndarray,
    D: int,
    param_name: str,
    embed_init: hk.initializers.Initializer,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    fprop_dtype: jnp.dtype,
    norm_config: NormConfig,
    hidden_dim: int = 64,
) -> jax.Array:
    values_normalized = normalize_continuous_value(values, norm_config)
    values_expanded = values_normalized[..., None]

    proj1 = get_parameter(
        f"{param_name}_proj1",
        [1, hidden_dim],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(hidden_dim),
    )
    hidden = jnp.dot(values_expanded.astype(proj1.dtype), proj1)

    hidden = jax.nn.gelu(hidden)

    proj2 = get_parameter(
        f"{param_name}_proj2",
        [hidden_dim, D],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(D),
    )
    embedding = jnp.dot(hidden, proj2)

    return embedding.astype(fprop_dtype)


def block_user_reduce(
    user_hashes: jnp.ndarray,
    user_embeddings: jnp.ndarray,
    hash_keys: HashKeys,
    lr_multiplier_func: Callable[[int], float] | None = None,
    embed_init_scale: float | None = None,
    user_ip_embeddings: jnp.ndarray | None = None,
    use_ip_address: bool = False,
) -> tuple[jax.Array, jax.Array]:
    num_user_token_hashes = hash_keys.num_user_hashes
    num_ip_hashes = hash_keys.num_ip_hashes if use_ip_address else 0
    B, _, D = user_embeddings.shape
    assert lr_multiplier_func is not None
    assert embed_init_scale is not None
    user_embedding = user_embeddings.reshape((B, 1, num_user_token_hashes * D))

    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")

    proj_mat_1 = get_parameter(
        "proj_mat_1",
        [num_user_token_hashes * D, D],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(num_user_token_hashes * D),
    )

    user_embedding = jnp.dot(user_embedding.astype(proj_mat_1.dtype), proj_mat_1).astype(
        user_embedding.dtype
    )

    if user_ip_embeddings is not None:
        ip_emb = user_ip_embeddings.reshape((B, num_ip_hashes, D))
        ip_emb = jnp.sum(ip_emb, axis=1, keepdims=True)
        user_embedding = user_embedding + ip_emb

    user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1).astype(jnp.bool_)

    return user_embedding, user_padding_mask


def embed_entity_sid(
    sids: jnp.ndarray,
    target_dim: int,
    sid_embed_dim: int,
    sid_num_levels: int,
    sid_codebook_size: int,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    fprop_dtype: jnp.dtype,
    name_prefix: str,
    sid_hash_level: bool = False,
    entity_hashes: jnp.ndarray | None = None,
    sid_cross_attn: bool = False,
) -> jnp.ndarray:
    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
    sids = sids.astype(jnp.int32)
    table_size = sid_codebook_size + 1

    num_unigram_levels = sid_num_levels + (1 if sid_hash_level else 0)
    combined_table = get_parameter(
        f"{name_prefix}_sid_combined",
        [num_unigram_levels * table_size, sid_embed_dim],
        dtype=jnp.float32,
        init=embed_init,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(sid_embed_dim),
    )
    pad_indices = jnp.arange(num_unigram_levels) * table_size
    combined_table = combined_table.at[pad_indices].set(0.0)
    table_3d = combined_table.reshape(num_unigram_levels, table_size, sid_embed_dim)

    all_raw_codes = sids

    if sid_hash_level:
        assert entity_hashes is not None, "entity_hashes required when sid_hash_level=True"
        h = entity_hashes[..., 0].astype(jnp.int32)
        mixed = h * jnp.int32(1540483477)
        mixed = mixed ^ (mixed >> 16)
        hash_code = jnp.abs(mixed) % sid_codebook_size + 1
        is_missing = sids[..., 0] == 0
        hash_code = jnp.where(is_missing, 0, hash_code)
        all_raw_codes = jnp.concatenate([all_raw_codes, hash_code[..., None]], axis=-1)

    all_onehot = jax.nn.one_hot(all_raw_codes, table_size, dtype=combined_table.dtype)
    unigram_embs = jnp.einsum("...lc,lcd->...ld", all_onehot, table_3d)

    if sid_cross_attn:
        D = sid_embed_dim

        Wqkv = get_parameter(
            f"{name_prefix}_sid_xattn_qkv",
            [D, 3 * D],
            dtype=jnp.float32,
            init=embed_init,
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(D),
        )
        qkv = jnp.dot(unigram_embs, Wqkv)
        Q, K, V = jnp.split(qkv, 3, axis=-1)

        attn_weights = jnp.einsum("...id,...jd->...ij", Q, K) / math.sqrt(D)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attended = jnp.einsum("...ij,...jd->...id", attn_weights, V)

        summed = jnp.mean(attended, axis=-2) + jnp.mean(unigram_embs, axis=-2)
    else:
        summed = unigram_embs.sum(axis=-2)

    if sid_embed_dim == target_dim:
        return summed.astype(fprop_dtype)

    proj = get_parameter(
        f"{name_prefix}_sid_proj",
        [sid_embed_dim, target_dim],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(sid_embed_dim),
    )
    return jnp.dot(summed.astype(proj.dtype), proj).astype(fprop_dtype)


def block_history_reduce(
    history_post_hashes: jnp.ndarray,
    history_author_hashes: jnp.ndarray,
    history_post_embeddings: jnp.ndarray | None,
    history_author_embeddings: jnp.ndarray,
    history_product_surface_embeddings: jnp.ndarray | None,
    history_actions_embeddings: jnp.ndarray,
    hash_keys: HashKeys,
    lr_multiplier_func: Callable[[int], float] | None = None,
    embed_init_scale: float | None = None,
    use_product_surface: bool = False,
    history_continuous_actions: jnp.ndarray | None = None,
    fprop_dtype: jnp.dtype = jnp.bfloat16,
    continuous_action_losses: list[ContinuousActionLossConfig] | None = None,
    continuous_action_hidden_dim: int = 64,
    sid_post_embeddings: jnp.ndarray | None = None,
    history_bridge_prob: jnp.ndarray | None = None,
) -> tuple[jax.Array, jax.Array]:
    num_item_hashes = hash_keys.num_item_hashes
    num_author_hashes = hash_keys.num_author_hashes
    assert lr_multiplier_func is not None
    assert embed_init_scale is not None
    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")

    B, _, D = history_actions_embeddings.shape
    embeddings = []

    if history_post_embeddings is not None:
        embeddings.append(history_post_embeddings.reshape((B, -1, num_item_hashes * D)))
    if sid_post_embeddings is not None:
        embeddings.append(sid_post_embeddings)

    embeddings.append(history_author_embeddings.reshape((B, -1, num_author_hashes * D)))

    embeddings.append(history_actions_embeddings)

    if use_product_surface and history_product_surface_embeddings is not None:
        embeddings.append(history_product_surface_embeddings)

    post_author_embedding = jnp.concatenate(embeddings, axis=-1)
    if history_bridge_prob is not None:
        p = jnp.clip(history_bridge_prob.astype(post_author_embedding.dtype), 0.0, 1.0)
        post_author_embedding = jnp.concatenate([post_author_embedding, p[..., None]], axis=-1)

    proj_mat_3 = get_parameter(
        "proj_mat_3",
        [post_author_embedding.shape[-1], D],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(post_author_embedding.shape[-1]),
    )
    history_post_author_embedding = jnp.dot(
        post_author_embedding.astype(proj_mat_3.dtype), proj_mat_3
    ).astype(post_author_embedding.dtype)

    post_action_embedding_seq = history_post_author_embedding.reshape(B, -1, D)

    if history_continuous_actions is not None and continuous_action_losses is not None:
        seen_action_indices: set[int] = set()
        for loss_config in continuous_action_losses:
            if loss_config.loss_weight > 0 and loss_config.action_index not in seen_action_indices:
                seen_action_indices.add(loss_config.action_index)
                action_values = history_continuous_actions[:, :, loss_config.action_index]
                action_embedding = project_continuous_value_to_embedding(
                    action_values,
                    D,
                    f"{loss_config.metric_name}_proj",
                    embed_init,
                    lr_multiplier_func,
                    embed_init_scale,
                    fprop_dtype,
                    loss_config.norm_config,
                    hidden_dim=continuous_action_hidden_dim,
                )
                post_action_embedding_seq += action_embedding

    history_padding_mask = (history_post_hashes[:, :, 0] != 0).reshape(B, -1)

    return post_action_embedding_seq, history_padding_mask


def block_candidate_reduce(
    candidate_post_hashes: jnp.ndarray,
    candidate_author_hashes: jnp.ndarray,
    candidate_post_embeddings: jnp.ndarray | None,
    candidate_author_embeddings: jnp.ndarray,
    candidate_product_surface_embeddings: jnp.ndarray | None,
    hash_keys: HashKeys,
    lr_multiplier_func: Callable[[int], float] | None = None,
    embed_init_scale: float | None = None,
    use_product_surface: bool = False,
    multimodal_embeddings: jnp.ndarray | None = None,
    candidate_search_query_embeddings: jnp.ndarray | None = None,
    search_query_embedding_dim: int = 0,
    fprop_dtype: jnp.dtype = jnp.bfloat16,
    sid_post_embeddings: jnp.ndarray | None = None,
) -> tuple[jax.Array, jax.Array]:
    num_item_hashes = hash_keys.num_item_hashes
    num_author_hashes = hash_keys.num_author_hashes
    assert lr_multiplier_func is not None
    assert embed_init_scale is not None
    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")

    B, _, D = candidate_author_embeddings.shape
    embeddings = []

    if candidate_post_embeddings is not None:
        embeddings.append(candidate_post_embeddings.reshape((B, -1, num_item_hashes * D)))
    if sid_post_embeddings is not None:
        embeddings.append(sid_post_embeddings)

    embeddings.append(candidate_author_embeddings.reshape((B, -1, num_author_hashes * D)))

    if use_product_surface and candidate_product_surface_embeddings is not None:
        embeddings.append(candidate_product_surface_embeddings)

    post_author_embedding = jnp.concatenate(embeddings, axis=-1)

    proj_input_dim = post_author_embedding.shape[-1]

    proj_mat_2 = get_parameter(
        "proj_mat_2",
        [proj_input_dim, D],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(proj_input_dim),
    )

    candidate_post_author_embedding = jnp.dot(
        post_author_embedding.astype(proj_mat_2.dtype), proj_mat_2
    ).astype(post_author_embedding.dtype)

    if multimodal_embeddings is not None:
        multimodal_dim = multimodal_embeddings.shape[-1]
        multimodal_proj_mat = get_parameter(
            "multimodal_proj_mat",
            [multimodal_dim, D],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(multimodal_dim),
        )
        multimodal_proj = jnp.dot(
            multimodal_embeddings.astype(multimodal_proj_mat.dtype), multimodal_proj_mat
        ).astype(candidate_post_author_embedding.dtype)
        candidate_post_author_embedding += multimodal_proj

    if search_query_embedding_dim > 0:
        proj_mat_search_query = get_parameter(
            "proj_mat_search_query",
            [search_query_embedding_dim, D],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(search_query_embedding_dim),
        )
        if candidate_search_query_embeddings is not None:
            search_query_projected = jnp.dot(
                candidate_search_query_embeddings.astype(proj_mat_search_query.dtype),
                proj_mat_search_query,
            ).astype(fprop_dtype)
            candidate_post_author_embedding = (
                candidate_post_author_embedding + search_query_projected
            )

    candidate_padding_mask = (candidate_post_hashes[:, :, 0] != 0).reshape(B, -1).astype(jnp.bool_)

    return candidate_post_author_embedding, candidate_padding_mask


def pad_to_next_128_multiple(
    embeddings: jax.Array,
    mask: jax.Array,
    product_surface: jax.Array,
    targets: jax.Array | None,
    raw_weights: jax.Array | None,
    negative_sample_mask: jax.Array | None,
    promoted_ids: jax.Array | None = None,
    continuous_actions: jax.Array | None = None,
    client_app_id: jax.Array | None = None,
    line_item_objective: jax.Array | None = None,
    safety_label_mask: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    batch_size, seq_len, emb_dim = embeddings.shape

    target_len = ((seq_len + 127) // 128) * 128
    pad_length = target_len - seq_len

    pad_embeddings = jnp.zeros((batch_size, pad_length, emb_dim), dtype=embeddings.dtype)
    pad_mask = jnp.zeros((batch_size, pad_length), dtype=mask.dtype)

    padded_embeddings = jnp.concatenate([embeddings, pad_embeddings], axis=1)
    padded_mask = jnp.concatenate([mask, pad_mask], axis=1)
    padded_product_surface = jnp.concatenate([product_surface, pad_mask], axis=1)

    if targets is not None:
        _, _, target_vocab_size = targets.shape
        pad_targets = jnp.zeros((batch_size, pad_length, target_vocab_size), dtype=targets.dtype)
        padded_targets = jnp.concatenate([targets, pad_targets], axis=1)
    else:
        padded_targets = None

    if raw_weights is not None:
        padded_raw_weights = jnp.concatenate([raw_weights, pad_mask], axis=1)
    else:
        padded_raw_weights = None

    if negative_sample_mask is not None:
        padded_negative_sample_mask = jnp.concatenate([negative_sample_mask, pad_mask], axis=1)
    else:
        padded_negative_sample_mask = None

    if promoted_ids is not None:
        pad_promoted_ids = jnp.zeros((batch_size, pad_length), dtype=promoted_ids.dtype)
        padded_promoted_ids = jnp.concatenate([promoted_ids, pad_promoted_ids], axis=1)
    else:
        padded_promoted_ids = None

    if continuous_actions is not None:
        _, _, num_continuous_actions = continuous_actions.shape
        pad_continuous_actions = jnp.zeros(
            (batch_size, pad_length, num_continuous_actions), dtype=continuous_actions.dtype
        )
        padded_continuous_actions = jnp.concatenate(
            [continuous_actions, pad_continuous_actions], axis=1
        )
    else:
        padded_continuous_actions = None

    if client_app_id is not None:
        pad_client_app_id = jnp.zeros((batch_size, pad_length), dtype=client_app_id.dtype)
        padded_client_app_id = jnp.concatenate([client_app_id, pad_client_app_id], axis=1)
    else:
        padded_client_app_id = None

    if line_item_objective is not None:
        pad_line_item_objective = jnp.zeros(
            (batch_size, pad_length), dtype=line_item_objective.dtype
        )
        padded_line_item_objective = jnp.concatenate(
            [line_item_objective, pad_line_item_objective], axis=1
        )
    else:
        padded_line_item_objective = None

    if safety_label_mask is not None:
        pad_safety_label_mask = jnp.zeros((batch_size, pad_length), dtype=safety_label_mask.dtype)
        padded_safety_label_mask = jnp.concatenate(
            [safety_label_mask, pad_safety_label_mask], axis=1
        )
    else:
        padded_safety_label_mask = None

    return (
        padded_embeddings,
        padded_mask,
        padded_product_surface,
        padded_targets,
        padded_raw_weights,
        padded_negative_sample_mask,
        padded_promoted_ids,
        padded_continuous_actions,
        padded_client_app_id,
        padded_line_item_objective,
        padded_safety_label_mask,
    )


def cast_jax(arr: npt.NDArray) -> jax.Array:
    return typing.cast(jax.Array, arr)


def build_metric_masks(
    mask: jax.Array,
    raw_targets: jax.Array,
    negative_sample_mask: jax.Array,
    product_surface: jax.Array,
    promoted_ids: jax.Array | None = None,
    client_app_id: jax.Array | None = None,
    new_user_mask: jax.Array | None = None,
    line_item_objective: jax.Array | None = None,
    no_history_mask: jax.Array | None = None,
    *,
    condition_conversion_on_click: bool = False,
    enable_platform_metrics: bool = False,
) -> dict[str, jax.Array]:
    promoted_mask = mask * (promoted_ids != 0) if promoted_ids is not None else jnp.zeros_like(mask)

    new_user_metric_mask = jnp.zeros_like(mask[:, :1]) if new_user_mask is None else new_user_mask
    no_history_metric_mask = (
        jnp.zeros_like(mask[:, :1]) if no_history_mask is None else no_history_mask
    )

    ios_mask = None
    android_mask = None
    if enable_platform_metrics:
        if client_app_id is not None:
            ios_ids = jnp.array(IOS_CLIENT_APP_IDS, dtype=jnp.int32)
            android_ids = jnp.array(ANDROID_CLIENT_APP_IDS, dtype=jnp.int32)
            ios_mask = mask * jnp.any(
                jnp.expand_dims(client_app_id, -1) == ios_ids, axis=-1
            ).astype(mask.dtype)
            android_mask = mask * jnp.any(
                jnp.expand_dims(client_app_id, -1) == android_ids, axis=-1
            ).astype(mask.dtype)
        else:
            ios_mask = jnp.zeros_like(mask)
            android_mask = jnp.zeros_like(mask)

    non_negative_mask = mask * (1 - negative_sample_mask)

    masks = {
        "all": mask,
        "non_negative": non_negative_mask,
        "negative": mask * negative_sample_mask,
        "non_negative_home": non_negative_mask
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_HOME_TIMELINE_RANKING),
        "non_negative_gallery": non_negative_mask
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_GALLERY_PAGE),
        "gallery": mask
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_GALLERY_PAGE),
        "non_negative_ranked_following": non_negative_mask
        * (
            product_surface
            == recsys_pb2.ProductSurface.PRODUCT_SURFACE_HOME_TIMELINE_RANKED_FOLLOWING
        ),
        "non_negative_search_results_page": non_negative_mask
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_SEARCH_RESULTS_PAGE),
        "promoted_home": mask
        * promoted_mask
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_HOME_TIMELINE_RANKING),
        "non_promoted_home": mask
        * (1 - promoted_mask)
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_HOME_TIMELINE_RANKING),
        "non_negative_promoted_home": non_negative_mask
        * promoted_mask
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_HOME_TIMELINE_RANKING),
        "non_negative_promoted_tweet_details": non_negative_mask
        * promoted_mask
        * (product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_TWEET_DETAILS_PAGE),
        "non_negative_new_user": non_negative_mask * new_user_metric_mask,
        "non_negative_non_new_user": non_negative_mask * (1 - new_user_metric_mask),
        "non_negative_no_history": non_negative_mask * no_history_metric_mask,
    }

    home_timeline_mask = (
        product_surface == recsys_pb2.ProductSurface.PRODUCT_SURFACE_HOME_TIMELINE_RANKING
    )
    website_clicks_objective = (
        (line_item_objective == 5).astype(mask.dtype)
        if line_item_objective is not None
        else jnp.zeros_like(mask)
    )
    masks["promoted_home_website_clicks"] = (
        mask * promoted_mask * home_timeline_mask * website_clicks_objective
    )
    masks["non_negative_promoted_home_website_clicks"] = (
        non_negative_mask * promoted_mask * home_timeline_mask * website_clicks_objective
    )

    if condition_conversion_on_click:
        click_mask = raw_targets[:, :, CLICK_ACTION_INDEX].astype(mask.dtype)
        masks["clicked"] = mask * click_mask
        masks["non_negative_clicked"] = mask * (1 - negative_sample_mask) * click_mask

    if enable_platform_metrics and ios_mask is not None and android_mask is not None:
        masks["ios"] = ios_mask
        masks["android"] = android_mask
        masks["ios_non_negative"] = ios_mask * (1 - negative_sample_mask)
        masks["android_non_negative"] = android_mask * (1 - negative_sample_mask)

    return masks


@dataclass
class RecsysAggregatedModel(hk.Module):
    config: RecsysAggregatedModelConfig
    model: Transformer
    sharding_context: ShardingContext

    @property
    def data_axis(self):
        return ("stage", *self.config.model_config.data_axis)

    def _compute_metrics_after_masks(
        self,
        masks: dict,
        raw_targets: jax.Array,
        logits: jax.Array,
        stats: dict,
        rce_ema: dict[str, jax.Array] | None = None,
        rce_alpha: jax.Array | None = None,
        smoothing_windows: tuple[int, ...] | None = None,
        calib_ema: dict[str, jax.Array] | None = None,
        raw_weights: jax.Array | None = None,
    ) -> dict:
        prob_labels = get_probs_and_labels(logits, raw_targets, self.config.metric_group)

        eng_names = []
        p_list = []
        y_list = []
        for eng_name, (p, y) in prob_labels.items():
            eng_names.append(eng_name)
            p_list.append(p)
            y_list.append(y)
        mask_keys = list(masks.keys())
        mask_values = list(masks.values())
        if raw_weights is not None:
            mask_values = [m * raw_weights for m in mask_values]

        metrics_calc(p_list, y_list, eng_names, mask_keys, mask_values, stats)

        metrics_calc_global_num_tokens(masks, stats)

        for mask_key, mval in zip(mask_keys, mask_values):
            stats[f"{mask_key}_effective_num_tokens"] = mval.sum()

        if rce_ema is not None and rce_alpha is not None and smoothing_windows is not None:
            _eps = 1e-7
            a = rce_alpha[:, None]
            new_rce_ema: dict[str, jax.Array] = {}

            for eng_name in eng_names:
                for mask_key in mask_keys:
                    base_key = f"{eng_name}/{mask_key}"

                    total_count = stats.get(
                        f"{mask_key}_effective_num_tokens",
                        stats[f"{mask_key}_num_tokens"],
                    )
                    ce_sum = stats[f"{eng_name}_{mask_key}_loss"] * total_count
                    pos_sum = stats[f"{eng_name}_{mask_key}_num_tokens"]
                    batch_stat = jnp.stack([ce_sum, pos_sum, total_count])

                    old = jnp.stack([rce_ema[f"{base_key}/{ws}"] for ws in smoothing_windows])
                    raw_updated = (1.0 - a) * old + a * batch_stat[None, :]
                    updated = jnp.where(
                        jnp.isnan(raw_updated),
                        jnp.where(jnp.isnan(old), batch_stat[None, :], old),
                        raw_updated,
                    )
                    for i, ws in enumerate(smoothing_windows):
                        new_rce_ema[f"{base_key}/{ws}"] = updated[i]

                    ema_ce = updated[:, 0] / jnp.maximum(updated[:, 2], 1.0)
                    ema_pb = updated[:, 1] / jnp.maximum(updated[:, 2], 1.0)
                    pb_c = jnp.clip(ema_pb, _eps, 1.0 - _eps)
                    bce_ref = -(pb_c * jnp.log(pb_c) + (1.0 - pb_c) * jnp.log(1.0 - pb_c))
                    s_rce = 100.0 * (bce_ref - ema_ce) / (bce_ref + 1e-12)
                    s_rce = jnp.where(updated[:, 2] > 0, s_rce, 0.0)

                    for i, ws in enumerate(smoothing_windows):
                        stats[f"{eng_name}_{mask_key}_smoothed_RCE_{ws}"] = s_rce[i]

            stats["_rce_ema"] = new_rce_ema

        if calib_ema is not None and rce_alpha is not None and smoothing_windows is not None:
            _eps = 1e-7
            a = rce_alpha[:, None]
            new_calib_ema: dict[str, jax.Array] = {}

            for eng_name, (p, y) in prob_labels.items():
                for mask_key, mask_val in zip(mask_keys, mask_values):
                    base_key = f"{eng_name}/{mask_key}"

                    pred_sum = jnp.sum(p * mask_val)
                    pos_sum_batch = jnp.sum(y * mask_val)
                    batch_stat = jnp.stack([pred_sum, pos_sum_batch])

                    old = jnp.stack([calib_ema[f"{base_key}/{ws}"] for ws in smoothing_windows])
                    raw_updated = (1.0 - a) * old + a * batch_stat[None, :]
                    updated = jnp.where(
                        jnp.isnan(raw_updated),
                        jnp.where(jnp.isnan(old), batch_stat[None, :], old),
                        raw_updated,
                    )
                    for i, ws in enumerate(smoothing_windows):
                        new_calib_ema[f"{base_key}/{ws}"] = updated[i]

                    s_calib = updated[:, 0] / jnp.maximum(updated[:, 1], _eps)
                    s_calib = jnp.where(updated[:, 1] > 0, s_calib, 0.0)
                    for i, ws in enumerate(smoothing_windows):
                        stats[f"{eng_name}_{mask_key}_smoothed_calib_{ws}"] = s_calib[i]

            stats["_calib_ema"] = new_calib_ema

        return stats

    def compute_length_bucketed_metrics(
        self,
        raw_targets: jax.Array,
        logits: jax.Array,
        mask: jax.Array,
        history_len: jax.Array,
        packed_candidate_seq_len: int,
        stats: dict,
        raw_weights: jax.Array | None = None,
    ) -> dict:
        prob_labels = get_probs_and_labels(logits, raw_targets, self.config.metric_group)

        len_per_candidate = jnp.repeat(history_len, packed_candidate_seq_len, axis=1)

        buckets = [
            ("len_0_255", 0, 255),
            ("len_256_511", 256, 511),
            ("len_512_767", 512, 767),
            ("len_768_1022", 768, 1022),
        ]

        for eng_name, (p, y) in prob_labels.items():
            for bucket_name, lo, hi in buckets:
                bucket_mask = mask & (len_per_candidate >= lo) & (len_per_candidate <= hi)
                bucket_mask_f = bucket_mask.astype(jnp.float32)

                if raw_weights is not None:
                    bucket_mask_f = bucket_mask_f * raw_weights

                total = bucket_mask_f.sum()
                num_pos = (bucket_mask_f * y).sum()

                stats[f"{eng_name}_{bucket_name}_RCE"] = jnp.where(
                    (num_pos > 0) & (total - num_pos > 0),
                    metric_rce(p, y, bucket_mask_f),
                    0.0,
                )

                pred_sum = jnp.sum(p * bucket_mask_f)
                stats[f"{eng_name}_{bucket_name}_calib"] = jnp.where(
                    num_pos > 0,
                    pred_sum / num_pos,
                    0.0,
                )

                stats[f"{bucket_name}_num_tokens"] = total

        return stats

    def _build_metric_masks(
        self,
        mask: jax.Array,
        raw_targets: jax.Array,
        negative_sample_mask: jax.Array,
        product_surface: jax.Array,
        promoted_ids: jax.Array | None = None,
        client_app_id: jax.Array | None = None,
        new_user_mask: jax.Array | None = None,
        line_item_objective: jax.Array | None = None,
        no_history_mask: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        return build_metric_masks(
            mask,
            raw_targets,
            negative_sample_mask,
            product_surface,
            promoted_ids,
            client_app_id,
            new_user_mask,
            line_item_objective,
            no_history_mask,
            condition_conversion_on_click=self.config.condition_conversion_on_click,
            enable_platform_metrics=self.config.enable_platform_metrics,
        )

    def compute_recsys_metrics(
        self,
        raw_targets: jax.Array,
        mask: jax.Array,
        product_surface: jax.Array,
        negative_sample_mask: jax.Array,
        logits: jax.Array,
        client_app_id: jax.Array | None = None,
        promoted_ids: jax.Array | None = None,
        new_user_mask: jax.Array | None = None,
        line_item_objective: jax.Array | None = None,
        no_history_mask: jax.Array | None = None,
        stats: dict | None = None,
        rce_ema: dict[str, jax.Array] | None = None,
        rce_alpha: jax.Array | None = None,
        smoothing_windows: tuple[int, ...] | None = None,
        calib_ema: dict[str, jax.Array] | None = None,
        raw_weights: jax.Array | None = None,
    ) -> dict:
        if stats is None:
            stats = {}

        masks = self._build_metric_masks(
            mask,
            raw_targets,
            negative_sample_mask,
            product_surface,
            promoted_ids,
            client_app_id,
            new_user_mask,
            line_item_objective,
            no_history_mask,
        )

        return self._compute_metrics_after_masks(
            masks,
            raw_targets,
            logits,
            stats,
            rce_ema=rce_ema,
            rce_alpha=rce_alpha,
            smoothing_windows=smoothing_windows,
            calib_ema=calib_ema,
            raw_weights=raw_weights,
        )

    def compute_timestamp_metrics(
        self,
        batch: RecsysFeaturesBatch,
        stats: dict | None = None,
    ) -> dict:
        if stats is None:
            stats = {}

        history_impr_ts = batch["history_seq"]["impr_ts"]
        assert history_impr_ts is not None
        candidate_impr_ts = batch["candidate_seq"]["impr_ts"]
        assert candidate_impr_ts is not None

        stats["max_ts_history"] = jnp.max(cast_jax(history_impr_ts))
        stats["max_ts_candidates"] = jnp.max(cast_jax(candidate_impr_ts))

        return stats

    def compute_sid_metrics(
        self,
        batch: RecsysFeaturesBatch,
        stats: dict | None = None,
    ) -> dict:
        if stats is None:
            stats = {}

        _config = self.config

        def _sid_coverage(sids, padding_mask, prefix: str) -> None:
            sids = cast_jax(sids)
            padding_mask = cast_jax(padding_mask)
            has_sid = (sids[..., 0] != 0).astype(jnp.float32)
            non_pad = (padding_mask > 0).astype(jnp.float32)
            total = jnp.sum(non_pad)
            present = jnp.sum(has_sid * non_pad)
            stats[f"sid/coverage_{prefix}"] = jnp.where(total > 0, present / total, 0.0)

        if _config.use_post_sid:
            _hp = batch["history_seq"].get("post_sids")
            _cp = batch["candidate_seq"].get("post_sids")
            hist_mask = batch["history_seq"]["post_hashes"][..., 0]
            cand_mask = batch["candidate_seq"]["post_hashes"][..., 0]
            if _hp is not None:
                _sid_coverage(_hp, hist_mask, "history_post")
            if _cp is not None:
                _sid_coverage(_cp, cand_mask, "candidate_post")

        return stats

    def compute_engagement_count_metrics(
        self,
        batch: RecsysFeaturesBatch | None = None,
        stats: dict | None = None,
    ) -> dict:
        if stats is None:
            stats = {}
        ctx_config = self.config.context_features
        if not ctx_config.enabled or not ctx_config.enable_engagement_counts or batch is None:
            return stats

        for cat_feat_enum, int64_feat_enum in ENGAGEMENT_COUNT_BUCKET_MAP:
            name = cat_feat_enum.name.removesuffix("CountBucketSeq").lower()
            max_bucket = ENGAGEMENT_COUNT_MAX_BUCKET.get(
                cat_feat_enum, ENGAGEMENT_COUNT_NUM_BUCKETS - 1
            )
            for seq_name, side in (("history_seq", "history"), ("candidate_seq", "candidate")):
                seq = batch[seq_name]
                raw_i64 = seq.get("int64_features")
                post_hashes = seq.get("post_hashes")
                if raw_i64 is None or post_hashes is None:
                    continue
                raw_counts = cast_jax(raw_i64)[:, :, int64_feat_enum.value]
                buckets = compute_engagement_count_bucket(raw_counts, max_bucket)
                valid = (cast_jax(post_hashes)[..., 0] > 0).astype(jnp.float32)
                n_valid = jnp.maximum(jnp.sum(valid), 1.0)
                coverage = jnp.sum((buckets > 0).astype(jnp.float32) * valid) / n_valid
                mean_bucket = jnp.sum(buckets.astype(jnp.float32) * valid) / n_valid
                stats[f"ec/{name}/{side}/coverage"] = coverage
                stats[f"ec/{name}/{side}/mean_bucket"] = mean_bucket
        return stats

    def compute_author_nsfw_metrics(
        self,
        batch: RecsysFeaturesBatch,
        stats: dict | None = None,
    ) -> dict:
        if stats is None:
            stats = {}

        ctx_config = self.config.context_features
        if not ctx_config.enable_author_nsfw:
            return stats
        nsfw_cfg = next(
            (cf for cf in ctx_config.categorical_features if cf.feature_name == "author_is_nsfw"),
            None,
        )
        if nsfw_cfg is None:
            return stats

        for seq_name, side in (("candidate_seq", "candidate"), ("history_seq", "history")):
            cat = batch[seq_name].get("categorical_features")
            hashes = batch[seq_name].get("post_hashes")
            if cat is None or hashes is None or nsfw_cfg.index >= cat.shape[-1]:
                continue
            cat = cast_jax(cat)
            is_nsfw = (cat[:, :, nsfw_cfg.index] == 1).astype(jnp.float32)
            non_pad = (cast_jax(hashes)[..., 0] != 0).astype(jnp.float32)
            total = jnp.sum(non_pad)
            present = jnp.sum(is_nsfw * non_pad)
            stats[f"author_nsfw/frac_{side}"] = jnp.where(total > 0, present / total, 0.0)
            stats[f"author_nsfw/count_{side}"] = present

        return stats

    @hk.transparent
    def multi_hot_to_embeddings(
        self,
        input: jax.Array,
        output_vocab_size: int,
        emb_size: int,
        embed_init_scale: float,
        name: str,
    ) -> tuple[jax.Array, jax.Array]:
        embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
        embedding_table = get_parameter(
            name,
            shape=[
                1,
                emb_size,
            ],
            init=embed_init,
            dtype=jnp.float32,
            pspec=P(),
            rms_clip_axes=(-2, -1),
        )
        table_reshaped = embedding_table.reshape(output_vocab_size, emb_size // output_vocab_size)

        B, S, V = input.shape
        D = emb_size
        D_per_V = table_reshaped.shape[1]

        if V * D_per_V != D:
            raise ValueError(
                f"{emb_size=} was divided into {output_vocab_size=} equal parts of length {D_per_V} each. But received an input with {V=}."
            )

        input_reshaped = (2 * input - 1)[:, :, :, None]
        table_reshaped = table_reshaped[None, None, :, :]

        selected_embeddings = input_reshaped * table_reshaped

        output = selected_embeddings.reshape(B, S, D)

        mask = jnp.any(input, axis=-1)
        output = output * mask[..., None]
        output = output.astype(DTYPE_BY_NAME[self.config.fprop_dtype])

        return output, embedding_table

    @hk.transparent
    def single_hot_to_embeddings(
        self,
        input: jax.Array,
        output_vocab_size: int,
        emb_size: int,
        embed_init_scale: float,
        name: str,
    ) -> tuple[jax.Array, jax.Array]:
        embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
        embedding_table = get_parameter(
            name,
            shape=[
                output_vocab_size,
                emb_size,
            ],
            init=embed_init,
            dtype=jnp.float32,
            pspec=P(),
            rms_clip_axes=(-2, -1),
        )
        input_one_hot = jax.nn.one_hot(input, output_vocab_size)
        output = jnp.dot(input_one_hot, embedding_table)
        output = output.astype(DTYPE_BY_NAME[self.config.fprop_dtype])

        return output, embedding_table

    @hk.transparent
    def embed_categorical_context_features(
        self,
        cat_features: jax.Array,
        is_candidate: bool = False,
    ) -> jax.Array | None:
        _config = self.config
        ctx_config = _config.context_features
        if not ctx_config.enabled or not ctx_config.categorical_features:
            return None

        is_2d = cat_features.ndim == 2
        if is_2d:
            cat_features = cat_features[:, None, :]

        _DEDICATED_FEATURE_NAMES = {"product_surface"}

        cat_embeddings_list: list[jax.Array] = []
        for cat_feat in ctx_config.categorical_features:
            if cat_feat.feature_name in _DEDICATED_FEATURE_NAMES:
                continue
            if (
                cat_feat.feature_name.endswith("_count_bucket")
                and not ctx_config.enable_engagement_counts
            ):
                continue
            if cat_feat.feature_name == "author_is_nsfw":
                if not ctx_config.enable_author_nsfw:
                    continue
            if cat_feat.embedding_dim <= 0 or cat_feat.index >= cat_features.shape[-1]:
                continue
            feat_values = cat_features[:, :, cat_feat.index]
            feat_values = jnp.clip(feat_values, 0, cat_feat.cardinality - 1)
            feat_emb, _ = self.single_hot_to_embeddings(
                feat_values,
                cat_feat.cardinality,
                cat_feat.embedding_dim,
                _config.embed_init_scale,
                cat_feat.resolved_embedding_name,
            )
            cat_embeddings_list.append(feat_emb)

        if not cat_embeddings_list:
            return None

        result = jnp.concatenate(cat_embeddings_list, axis=-1)

        if is_2d:
            result = result[:, 0, :]

        return result

    @hk.transparent
    def build_unified_context_embedding(
        self,
        product_surface: jax.Array,
        dwell_time_values: jax.Array | None = None,
        cat_features: jax.Array | None = None,
        name: str = "unified_ctx",
        is_candidate: bool = False,
    ) -> jax.Array | None:
        _config = self.config
        ctx_config = _config.context_features
        if not ctx_config.enabled:
            return None
        fprop_dtype = DTYPE_BY_NAME[_config.fprop_dtype]
        D = _config.emb_table_width
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        lr_multiplier_func = _config.model_config.scale_config.emb_lr_multiplier

        cat_cfg_by_name: dict[str, CategoricalFeatureConfig] = {}
        for cat_feat in ctx_config.categorical_features:
            if cat_feat.embedding_dim > 0:
                cat_cfg_by_name[cat_feat.feature_name] = cat_feat

        parts: list[jax.Array] = []

        ps_cfg = cat_cfg_by_name.get("product_surface")
        if ps_cfg is not None:
            ps_emb, self.product_surface_embedding_table = self.single_hot_to_embeddings(
                product_surface,
                ps_cfg.cardinality,
                ps_cfg.embedding_dim,
                _config.embed_init_scale,
                "product_surface_embedding_table",
            )
            parts.append(ps_emb)

        if dwell_time_values is not None:
            dwell_dim = _config.unified_context_dwell_time_dim
            hidden_dim = _config.continuous_action_hidden_dim
            norm_config = NormConfig(norm_scale=30.0, use_log=False)
            for lc in _config.continuous_action_losses:
                if lc.action_index == 1:
                    norm_config = lc.norm_config
                    break

            values_normalized = normalize_continuous_value(dwell_time_values, norm_config)
            values_expanded = values_normalized[..., None]

            proj1 = get_parameter(
                f"{name}_dwell_proj1",
                [1, hidden_dim],
                dtype=jnp.float32,
                init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
                pspec=P(None, None),
                lr_multiplier=lr_multiplier_func(hidden_dim),
            )
            hidden = jnp.dot(values_expanded.astype(proj1.dtype), proj1)
            hidden = jax.nn.gelu(hidden)

            proj2 = get_parameter(
                f"{name}_dwell_proj2",
                [hidden_dim, dwell_dim],
                dtype=jnp.float32,
                init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
                pspec=P(None, None),
                lr_multiplier=lr_multiplier_func(dwell_dim),
            )
            dwell_emb = jnp.dot(hidden, proj2).astype(fprop_dtype)
            parts.append(dwell_emb)

        if cat_features is not None:
            cat_emb = self.embed_categorical_context_features(
                cat_features, is_candidate=is_candidate
            )
            if cat_emb is not None:
                parts.append(cat_emb)

        concat = jnp.concatenate(parts, axis=-1)
        total_dim = concat.shape[-1]

        unified_ctx_proj = get_parameter(
            f"{name}_proj",
            [total_dim, D],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(total_dim),
        )
        result = jnp.dot(concat.astype(unified_ctx_proj.dtype), unified_ctx_proj).astype(
            fprop_dtype
        )
        return result

    @hk.transparent
    def _get_unembedding(self) -> jax.Array:
        _config = self.config

        out_pspec = P(None, None)
        emb_size = _config.model_config.emb_size
        embed_init = hk.initializers.VarianceScaling(_config.embed_init_scale, mode="fan_out")

        unembed_mat = get_parameter(
            "unembeddings",
            [emb_size, _config.model_config.output_vocab_size],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=out_pspec,
            lr_multiplier=_config.model_config.scale_config.emb_lr_multiplier(emb_size),
            rms_clip_axes=(-1, -2),
        )
        self.unembed_mat: jax.Array = with_sharding_constraint(unembed_mat, out_pspec)
        return unembed_mat

    @hk.transparent
    def decode(self, inputs: jax.Array) -> jax.Array:
        unembeddings = self._get_unembedding()
        return jnp.dot(inputs.astype(unembeddings.dtype), unembeddings).astype(inputs.dtype)

    @hk.transparent
    def _get_continuous_unembedding(self) -> jax.Array:
        _config = self.config

        out_pspec = P(None, None)
        emb_size = _config.model_config.emb_size
        embed_init = hk.initializers.VarianceScaling(_config.embed_init_scale, mode="fan_out")

        unembed_mat = get_parameter(
            "continuous_unembeddings",
            [emb_size, _config.num_continuous_actions],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=out_pspec,
            lr_multiplier=_config.model_config.scale_config.emb_lr_multiplier(emb_size),
            rms_clip_axes=(-1, -2),
        )
        return with_sharding_constraint(unembed_mat, out_pspec)

    @hk.transparent
    def decode_continuous(
        self, inputs: jax.Array, product_surface: jax.Array | None = None
    ) -> jax.Array:
        unembeddings = self._get_continuous_unembedding()
        logits = jnp.dot(inputs.astype(unembeddings.dtype), unembeddings).astype(inputs.dtype)

        configs_by_index: dict[int, list[ContinuousActionLossConfig]] = {}
        for lc in self.config.continuous_action_losses:
            configs_by_index.setdefault(lc.action_index, []).append(lc)

        num_heads = logits.shape[-1]
        activated_slices = []

        for i in range(num_heads):
            head_logits = logits[..., i : i + 1]
            configs = configs_by_index.get(i, [])

            if not configs:
                activated_slices.append(head_logits)
                continue

            needs_surface = len(configs) > 1 or any(
                c.product_surfaces or c.exclude_product_surfaces for c in configs
            )
            if not needs_surface:
                c = configs[0]
                head_out = self._apply_activation(head_logits, c.activation)
                if c.output_cap > 0:
                    head_out = c.output_cap * jnp.tanh(head_out / c.output_cap)
                activated_slices.append(head_out)
            else:
                assert product_surface is not None, (
                    f"product_surface required: action index {i} has surface-filtered configs"
                )
                head_out = head_logits
                for c in configs:
                    smask = _get_surface_mask(c, product_surface)[..., None]
                    activated = self._apply_activation(head_logits, c.activation)
                    if c.output_cap > 0:
                        activated = c.output_cap * jnp.tanh(activated / c.output_cap)
                    head_out = jnp.where(smask, activated, head_out)
                activated_slices.append(head_out)

        return jnp.concatenate(activated_slices, axis=-1)

    def _apply_activation(self, logits: jax.Array, activation: str | None) -> jax.Array:
        if activation == "softplus":
            return jax.nn.softplus(logits)
        elif activation == "sigmoid":
            return jax.nn.sigmoid(logits)
        else:
            return logits

    @hk.transparent
    def maybe_tfmr_project_embeddings(self, embeddings, config):
        if config.model_config.emb_size != config.emb_table_width:
            w_init = hk.initializers.VarianceScaling(
                config.model_config.scale_config.attn_init_scale**2
            )
            lr_multiplier = config.model_config.scale_config.hidden_lr_multiplier(
                config.emb_table_width
            )
            init_scale = 1.0

            tfmr_projection = Linear(
                config.model_config.emb_size,
                w_init=w_init,
                with_bias=False,
                pspec=P(None, None),
                sharding_context=self.sharding_context,
                lr_multiplier=lr_multiplier,
                init_scale=init_scale,
                name="tfmr_proj",
            )

            embeddings = tfmr_projection(embeddings)
        return embeddings

    @hk.transparent
    def _build_user_embedding(
        self,
        user_hashes: jax.Array,
        recsys_embeddings: RecsysEmbeddings,
        reshape_for_seqpack: tuple[int, int] | None = None,
    ) -> tuple[jax.Array, jax.Array | None] | None:
        if not self.config.use_user_embedding:
            return None

        _config = self.config
        assert recsys_embeddings.user_embeddings is not None
        if reshape_for_seqpack is not None:
            num_devices, bs_per_device = reshape_for_seqpack
            total_batch = num_devices * bs_per_device
            flat_hashes = user_hashes.reshape(total_batch, -1)
            flat_embs = recsys_embeddings.user_embeddings.reshape(
                total_batch,
                recsys_embeddings.user_embeddings.shape[-2],
                recsys_embeddings.user_embeddings.shape[-1],
            )
            flat_ip_embs = (
                recsys_embeddings.user_ip_embeddings.reshape(
                    total_batch,
                    recsys_embeddings.user_ip_embeddings.shape[-2],
                    recsys_embeddings.user_ip_embeddings.shape[-1],
                )
                if recsys_embeddings.user_ip_embeddings is not None
                else None
            )
            user_embeddings, _ = block_user_reduce(
                flat_hashes,
                flat_embs,
                self.config.hash_table.hash_keys,
                self.config.model_config.scale_config.emb_lr_multiplier,
                self.config.embed_init_scale,
                user_ip_embeddings=flat_ip_embs,
                use_ip_address=_config.use_ip_address,
            )
            return user_embeddings.reshape(num_devices, bs_per_device, -1), None
        else:
            user_ip_embeddings = (
                recsys_embeddings.user_ip_embeddings
                if recsys_embeddings.user_ip_embeddings is not None
                else None
            )
            user_embeddings, user_padding_mask = block_user_reduce(
                user_hashes,
                recsys_embeddings.user_embeddings,
                self.config.hash_table.hash_keys,
                self.config.model_config.scale_config.emb_lr_multiplier,
                self.config.embed_init_scale,
                user_ip_embeddings=user_ip_embeddings,
                use_ip_address=_config.use_ip_address,
            )
            return user_embeddings, user_padding_mask

    @hk.transparent
    def build_inputs(
        self,
        recsys_features_batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        is_training: bool = True,
    ) -> tuple[jax.Array, jax.Array, int | None, jax.Array | None]:
        _config = self.config
        assert _config.model_config.output_vocab_size is not None, "output_vocab_size is required"

        if _config.feature_prep_enabled:
            fp = _config.feature_prep
            scale_multiplier = fp.scale_config.input_scale(fp.emb_size)
            tokens, padding_mask, candidate_start_offset = build_feature_prep_inputs(
                batch=recsys_features_batch,
                recsys_embeddings=recsys_embeddings,
                config=fp,
                hash_keys=_config.hash_table.hash_keys,
                input_scale=scale_multiplier,
                output_vocab_size=_config.model_config.output_vocab_size,
                is_training=is_training,
            )
            tokens = with_sharding_constraint(tokens, P(self.data_axis, ("seq", "model")))
            return tokens, padding_mask, candidate_start_offset, None

        ctx_config = _config.context_features
        history_cat_features: jax.Array | None = None
        candidate_cat_features: jax.Array | None = None
        if ctx_config.enabled:
            raw_hist_cat = recsys_features_batch["history_seq"].get("categorical_features")
            if raw_hist_cat is not None:
                history_cat_features = cast_jax(raw_hist_cat)
            raw_cand_cat = recsys_features_batch["candidate_seq"].get("categorical_features")
            if raw_cand_cat is not None:
                candidate_cat_features = cast_jax(raw_cand_cat)

        pa_cfg = next(
            (cf for cf in ctx_config.categorical_features if cf.feature_name == "post_age"),
            None,
        )
        if pa_cfg is not None:
            history_impr_ts_pa = recsys_features_batch["history_seq"]["impr_ts"]
            candidate_impr_ts_pa = recsys_features_batch["candidate_seq"]["impr_ts"]
            assert history_impr_ts_pa is not None
            assert candidate_impr_ts_pa is not None

            history_post_age_buckets = compute_post_age_bucket(
                cast_jax(history_impr_ts_pa),
                cast_jax(recsys_features_batch["history_seq"]["post_creation_ts_sec"]),
                _config.post_age_granularity_mins,
                _config.post_age_max_mins,
                _config.post_age_bucket_strategy,
                _config.post_age_num_buckets,
            )
            candidate_post_age_buckets = compute_post_age_bucket(
                cast_jax(candidate_impr_ts_pa),
                cast_jax(recsys_features_batch["candidate_seq"]["post_creation_ts_sec"]),
                _config.post_age_granularity_mins,
                _config.post_age_max_mins,
                _config.post_age_bucket_strategy,
                _config.post_age_num_buckets,
            )

            pa_idx = pa_cfg.index
            if history_cat_features is not None:
                history_cat_features = history_cat_features.at[:, :, pa_idx].set(
                    history_post_age_buckets.astype(history_cat_features.dtype)
                )
            if candidate_cat_features is not None:
                candidate_cat_features = candidate_cat_features.at[:, :, pa_idx].set(
                    candidate_post_age_buckets.astype(candidate_cat_features.dtype)
                )

        history_dwell_time: jax.Array | None = None
        history_bridge_prob: jax.Array | None = None
        history_continuous_actions = recsys_features_batch["history_seq"]["continuous_actions"]
        if history_continuous_actions is not None:
            ca = cast_jax(history_continuous_actions)
            if ca.shape[-1] > 1:
                history_dwell_time = ca[:, :, 1]
            if _config.concat_history_bridge_prob and ca.shape[-1] > 0:
                history_bridge_prob = ca[:, :, 0]

        if ctx_config.enable_engagement_counts:
            for cat_feat_enum, int64_feat_enum in ENGAGEMENT_COUNT_BUCKET_MAP:
                for seq_name, cat_ref in [
                    ("history_seq", "hist"),
                    ("candidate_seq", "cand"),
                ]:
                    cat_feats = (
                        history_cat_features if cat_ref == "hist" else candidate_cat_features
                    )
                    if cat_feats is None:
                        continue
                    raw_i64 = recsys_features_batch[seq_name].get("int64_features")
                    if raw_i64 is None:
                        continue
                    raw_counts = cast_jax(raw_i64)[:, :, int64_feat_enum.value]
                    buckets = compute_engagement_count_bucket(
                        raw_counts,
                        ENGAGEMENT_COUNT_MAX_BUCKET.get(
                            cat_feat_enum, ENGAGEMENT_COUNT_NUM_BUCKETS - 1
                        ),
                    )
                    cat_feats = cat_feats.at[:, :, cat_feat_enum.value].set(
                        buckets.astype(cat_feats.dtype)
                    )
                    if cat_ref == "hist":
                        history_cat_features = cat_feats
                    else:
                        candidate_cat_features = cat_feats

        self.product_surface_embedding_table: jax.Array
        history_unified_context = self.build_unified_context_embedding(
            product_surface=cast_jax(recsys_features_batch["history_seq"]["product_surface"]),
            dwell_time_values=history_dwell_time,
            cat_features=history_cat_features,
            name="unified_ctx_history",
        )

        candidate_unified_context = self.build_unified_context_embedding(
            product_surface=cast_jax(recsys_features_batch["candidate_seq"]["product_surface"]),
            dwell_time_values=None,
            cat_features=candidate_cat_features,
            name="unified_ctx_cand",
            is_candidate=True,
        )

        history_actions = recsys_features_batch["history_seq"]["actions"]
        assert history_actions is not None
        self.action_embedding_table: jax.Array
        history_actions_embeddings, self.action_embedding_table = self.multi_hot_to_embeddings(
            cast_jax(history_actions),
            _config.model_config.output_vocab_size,
            _config.emb_table_width,
            _config.embed_init_scale,
            "action_embedding_table",
        )
        multimodal_embeddings: jax.Array | None = None

        if _config.multimodal_embedding_type is not None:
            embedding = recsys_features_batch["candidate_seq"].get("embedding")
            assert embedding is not None, (
                "embedding must be present when multimodal_embedding_type is set"
            )
            if isinstance(embedding, jax.Array):
                multimodal_embeddings = embedding
            else:
                multimodal_embeddings = cast_jax(embedding)

        candidate_search_query_embeddings = None
        if self.config.search_query_embedding_dim > 0:
            search_query_emb = recsys_features_batch["candidate_seq"]["search_query_embeddings"]
            if search_query_emb is not None:
                candidate_search_query_embeddings = cast_jax(search_query_emb)

        user_hashes = cast_jax(recsys_features_batch["user_hashes"])
        history_post_hashes = cast_jax(recsys_features_batch["history_seq"]["post_hashes"])
        history_auth_hashes = cast_jax(recsys_features_batch["history_seq"]["auth_hashes"])
        candidate_post_hashes = cast_jax(recsys_features_batch["candidate_seq"]["post_hashes"])
        candidate_auth_hashes = cast_jax(recsys_features_batch["candidate_seq"]["auth_hashes"])

        if _config.use_seqpack:
            assert not _config.use_post_sid, (
                "use_post_sid is not yet supported with use_seqpack; "
                "SID embeddings would need reshape into the packed layout."
            )
            layout = recsys_features_batch.get("packing_layout")
            assert layout is not None

            num_devices, bs_per_device, _ = user_hashes.shape
            total_batch = num_devices * bs_per_device

            user_emb_result = self._build_user_embedding(
                user_hashes,
                recsys_embeddings,
                reshape_for_seqpack=(num_devices, bs_per_device),
            )
            user_embeddings = user_emb_result[0] if user_emb_result is not None else None

            assert recsys_embeddings.history_post_embeddings is not None
            history_embeddings, history_padding_mask = block_history_reduce(
                history_post_hashes,
                history_auth_hashes,
                recsys_embeddings.history_post_embeddings.reshape(
                    num_devices, -1, recsys_embeddings.history_post_embeddings.shape[-1]
                ),
                recsys_embeddings.history_author_embeddings.reshape(
                    num_devices, -1, recsys_embeddings.history_author_embeddings.shape[-1]
                ),
                history_unified_context,
                history_actions_embeddings,
                self.config.hash_table.hash_keys,
                self.config.model_config.scale_config.emb_lr_multiplier,
                self.config.embed_init_scale,
                history_unified_context is not None,
                history_bridge_prob=(
                    None
                    if history_bridge_prob is None
                    else history_bridge_prob.reshape(num_devices, -1)
                ),
            )

            assert recsys_embeddings.candidate_post_embeddings is not None
            candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
                candidate_post_hashes,
                candidate_auth_hashes,
                recsys_embeddings.candidate_post_embeddings.reshape(
                    num_devices, -1, recsys_embeddings.candidate_post_embeddings.shape[-1]
                ),
                recsys_embeddings.candidate_author_embeddings.reshape(
                    num_devices, -1, recsys_embeddings.candidate_author_embeddings.shape[-1]
                ),
                candidate_unified_context,
                self.config.hash_table.hash_keys,
                self.config.model_config.scale_config.emb_lr_multiplier,
                self.config.embed_init_scale,
                use_product_surface=candidate_unified_context is not None,
                multimodal_embeddings=multimodal_embeddings,
                candidate_search_query_embeddings=candidate_search_query_embeddings,
                search_query_embedding_dim=self.config.search_query_embedding_dim,
                fprop_dtype=DTYPE_BY_NAME[self.config.fprop_dtype],
            )

            user_features_token = None
            if _config.user_features.has_user_features:
                flat_batch = typing.cast(
                    RecsysFeaturesBatch,
                    {
                        **recsys_features_batch,
                        "user_categorical_features": cast_jax(
                            recsys_features_batch["user_categorical_features"]
                        ).reshape(total_batch, -1),
                        "user_bool_features": cast_jax(
                            recsys_features_batch["user_bool_features"]
                        ).reshape(total_batch, -1),
                        "user_float_features": cast_jax(
                            recsys_features_batch["user_float_features"]
                        ).reshape(total_batch, -1),
                        "user_int64_features": cast_jax(
                            recsys_features_batch["user_int64_features"]
                        ).reshape(total_batch, -1),
                        "user_installed_apps_multihot": cast_jax(
                            recsys_features_batch["user_installed_apps_multihot"]
                        ).reshape(total_batch, -1),
                    },
                )

                feature_parts = build_user_feature_parts(
                    flat_batch, _config.user_features, self.sharding_context
                )
                user_features_token = build_user_features_token(
                    feature_parts,
                    _config.user_features.user_features_concat_dim,
                    self.config.emb_table_width,
                    DTYPE_BY_NAME[self.config.fprop_dtype],
                    self.sharding_context,
                    use_mlp=_config.user_features.user_features_mlp,
                    pad=_config.user_features.user_features_concat_pad,
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
            embeddings = embeddings.at[device_idx, cast_jax(layout.candidate_positions)].add(
                jnp.where(candidate_padding_mask[:, :, None], candidate_embeddings, 0)
            )

            candidate_start_offset = None
        else:
            sequence_parts: list[jax.Array] = []
            mask_parts: list[jax.Array] = []

            user_emb_result = self._build_user_embedding(user_hashes, recsys_embeddings)
            if user_emb_result is not None:
                user_emb, user_mask = user_emb_result
                assert user_mask is not None
                sequence_parts.append(user_emb)
                mask_parts.append(user_mask)

            def _embed_post_sid(seq_name: str) -> jnp.ndarray | None:
                seq = recsys_features_batch[seq_name]
                sids_in = seq.get("post_sids")
                if sids_in is None:
                    assert hk.running_init(), (
                        f"use_post_sid=True but {seq_name}.post_sids is missing from batch; "
                        "verify the upstream kafka topic emits `semanticIdSeq`."
                    )
                    hashes = cast_jax(seq["post_hashes"])
                    sids_jax: jax.Array = jnp.zeros(
                        (hashes.shape[0], hashes.shape[1], _config.sid_num_levels),
                        dtype=jnp.uint16,
                    )
                else:
                    sids_jax = cast_jax(sids_in)
                needs_hashes = _config.sid_hash_level
                return embed_entity_sid(
                    sids_jax,
                    _config.emb_table_width,
                    _config.sid_embed_dim,
                    _config.sid_num_levels,
                    _config.sid_codebook_size,
                    self.config.model_config.scale_config.emb_lr_multiplier,
                    self.config.embed_init_scale,
                    DTYPE_BY_NAME[_config.fprop_dtype],
                    "post",
                    sid_hash_level=_config.sid_hash_level,
                    entity_hashes=cast_jax(seq["post_hashes"]) if needs_hashes else None,
                    sid_cross_attn=_config.sid_cross_attn,
                )

            _sid_post_emb_h = _embed_post_sid("history_seq") if _config.use_post_sid else None
            _sid_post_emb_c = _embed_post_sid("candidate_seq") if _config.use_post_sid else None

            history_embeddings, history_padding_mask = block_history_reduce(
                history_post_hashes,
                history_auth_hashes,
                recsys_embeddings.history_post_embeddings if _config.use_post_embedding else None,
                recsys_embeddings.history_author_embeddings,
                history_unified_context,
                history_actions_embeddings,
                self.config.hash_table.hash_keys,
                self.config.model_config.scale_config.emb_lr_multiplier,
                self.config.embed_init_scale,
                history_unified_context is not None,
                sid_post_embeddings=_sid_post_emb_h,
                history_bridge_prob=history_bridge_prob,
            )

            if (
                self.config.safety_filter_mode == "hard"
                and self.config.safety_filter_apply_to_history
            ):
                history_safety_mask = recsys_features_batch["history_seq"].get("safety_label_mask")
                history_padding_mask, _ = apply_safety_filter(
                    history_safety_mask,
                    history_padding_mask,
                    None,
                    mode="hard",
                    bits=self.config.safety_filter_bits,
                    soft_weight=self.config.safety_filter_soft_weight,
                )

            candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
                candidate_post_hashes,
                candidate_auth_hashes,
                recsys_embeddings.candidate_post_embeddings if _config.use_post_embedding else None,
                recsys_embeddings.candidate_author_embeddings,
                candidate_unified_context,
                self.config.hash_table.hash_keys,
                self.config.model_config.scale_config.emb_lr_multiplier,
                self.config.embed_init_scale,
                use_product_surface=candidate_unified_context is not None,
                multimodal_embeddings=multimodal_embeddings,
                candidate_search_query_embeddings=candidate_search_query_embeddings,
                search_query_embedding_dim=self.config.search_query_embedding_dim,
                fprop_dtype=DTYPE_BY_NAME[self.config.fprop_dtype],
                sid_post_embeddings=_sid_post_emb_c,
            )

            if _config.user_features.has_user_features:
                feature_parts = build_user_feature_parts(
                    recsys_features_batch, _config.user_features, self.sharding_context
                )
                user_features_token = build_user_features_token(
                    feature_parts,
                    _config.user_features.user_features_concat_dim,
                    self.config.emb_table_width,
                    DTYPE_BY_NAME[self.config.fprop_dtype],
                    self.sharding_context,
                    use_mlp=_config.user_features.user_features_mlp,
                    pad=_config.user_features.user_features_concat_pad,
                )
                sequence_parts.append(user_features_token)
                mask_parts.append(jnp.ones((user_hashes.shape[0], 1), dtype=jnp.bool_))

            sequence_parts.extend([history_embeddings, candidate_embeddings])
            mask_parts.extend([history_padding_mask, candidate_padding_mask])
            embeddings = jnp.concatenate(sequence_parts, axis=1)
            padding_mask = jnp.concatenate(mask_parts, axis=1)
            candidate_start_offset = sum(p.shape[1] for p in mask_parts[:-1])

        embeddings *= self.config.model_config.scale_config.input_scale(self.config.emb_table_width)
        embeddings = with_sharding_constraint(embeddings, P(self.data_axis, ("seq", "model")))

        embeddings = self.maybe_tfmr_project_embeddings(embeddings, self.config)
        return embeddings, padding_mask, candidate_start_offset, multimodal_embeddings

    def __call__(
        self,
        input_embeddings: jax.Array,
        padding_mask: jax.Array | None,
        *,
        is_training: bool = True,
        positions: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        candidate_start_offset: int | None = None,
        product_surface: jax.Array | None = None,
        seqpack_layout: SequencePackedLayout | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        _debug_dir = self.config.model_config.debug_tensor_dump_output_folder
        dump_to_file(_debug_dir, "input_embeddings", input_embeddings)

        config = self.config
        scale_config = config.model_config.scale_config
        fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]

        B, T, _ = input_embeddings.shape
        if segment_ids is None:
            segment_ids = jnp.zeros((B, T), dtype=jnp.int32)

        attn_mask = jnp.ones((B, T)).astype(jnp.int32)

        model_out = self.model(
            input_embeddings,
            attn_mask,
            segment_ids=segment_ids,
            is_training=is_training,
            decoding=False,
            padding_mask=padding_mask,
            positions=positions,
            seqpack_layout=seqpack_layout,
        )

        out_embeddings = model_out.output

        out_embeddings = with_sharding_constraint(
            out_embeddings, P(self.data_axis, ("seq", "model"))
        )

        dump_to_file(_debug_dir, "last_hidden_before_norm", out_embeddings)

        out_embeddings = rms_norm_fn(
            out_embeddings,
            name="final_layer_norm",
            weight_decay_mask=scale_config.ln_weight_decay_mask,
        )
        assert out_embeddings.dtype == fprop_dtype, (
            f"out_embeddings.dtype: {out_embeddings.dtype} fprop_dtype: {fprop_dtype}"
        )

        dump_to_file(_debug_dir, "last_hidden_after_norm", out_embeddings)

        if self.config.transformer_output_only:
            return out_embeddings, jnp.zeros((0, 0, 0))

        assert candidate_start_offset is None or seqpack_layout is None

        if seqpack_layout is not None:
            out_embeddings = jnp.take_along_axis(
                out_embeddings, cast_jax(seqpack_layout.candidate_positions)[:, :, None], axis=1
            )
        elif candidate_start_offset is not None:
            out_embeddings = out_embeddings[:, candidate_start_offset:, :]

        logits = self.decode(out_embeddings)
        logits *= scale_config.output_scale(config.model_config.emb_size)
        logits = with_sharding_constraint(logits, P(self.data_axis, ("seq", "model")))

        dump_to_file(_debug_dir, "after_unembeddings", logits)

        if config.final_logit_cap > 0.0:
            logits = config.final_logit_cap * jnp.tanh(logits / config.final_logit_cap)

        continuous_predictions = self.decode_continuous(out_embeddings, product_surface)

        return logits, continuous_predictions

    @hk.transparent
    def loss(
        self,
        inter,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        is_training: bool = True,
        rce_ema: dict[str, jax.Array] | None = None,
        rce_alpha: jax.Array | None = None,
        smoothing_windows: tuple[int, ...] | None = None,
        calib_ema: dict[str, jax.Array] | None = None,
    ):
        del inter
        assert self.config.model_config.output_vocab_size is not None, (
            "RecsysAggregatedModel output_vocab_size is None"
        )
        input_embeddings, padding_mask, candidate_start_offset, mm_emb = self.build_inputs(
            batch,
            get_recsys_embed_param_to_jax_array(recsys_embeddings),
            is_training=is_training,
        )
        targets = batch["candidate_seq"]["actions"]
        product_surface = batch["candidate_seq"]["product_surface"]
        promoted_ids = batch["candidate_seq"]["promoted_ids"]
        candidate_continuous_actions = batch["candidate_seq"]["continuous_actions"]
        raw_client_app_id = batch["candidate_seq"].get("client_app_id")
        raw_line_item_objective = batch["candidate_seq"].get("line_item_objective")
        candidate_safety_mask = batch["candidate_seq"].get("safety_label_mask")
        assert targets is not None
        targets = cast_jax(targets)
        product_surface = cast_jax(product_surface)
        promoted_ids = cast_jax(promoted_ids) if promoted_ids is not None else None
        candidate_continuous_actions = (
            cast_jax(candidate_continuous_actions)
            if candidate_continuous_actions is not None
            else None
        )
        client_app_id = (
            cast_jax(raw_client_app_id)
            if self.config.enable_platform_metrics and raw_client_app_id is not None
            else None
        )
        line_item_objective = (
            cast_jax(raw_line_item_objective) if raw_line_item_objective is not None else None
        )
        candidate_safety_mask = (
            cast_jax(candidate_safety_mask) if candidate_safety_mask is not None else None
        )
        raw_weights: jax.Array | None = None

        if self.config.use_seqpack:
            bs_per_device = batch["user_hashes"].shape[1]
            packed_candidate_seq_len = targets.shape[1] // bs_per_device
            negative_sample_mask = (
                jnp.arange(targets.shape[1], dtype=jnp.int32)[None, :] % packed_candidate_seq_len
            ) >= self.config.candidate_seq_len
        else:
            negative_sample_mask = jnp.zeros((targets.shape[0], targets.shape[1]), dtype=jnp.bool_)
            negative_sample_mask = negative_sample_mask.at[:, self.config.candidate_seq_len :].set(
                True
            )

        sample_weights = batch.get("sample_weights")
        if sample_weights is not None:
            sample_weights = cast_jax(sample_weights).astype(jnp.float32)
            if self.config.use_seqpack:
                raw_weights = jnp.repeat(
                    sample_weights.squeeze(-1), packed_candidate_seq_len, axis=1
                )
            else:
                raw_weights = jnp.broadcast_to(sample_weights, targets.shape[:2])

        if self.config.log_q_correction:
            tweet_counts = get_candidate_tweet_counts(
                batch, self.config.log_q_num_bins, negative_sample_mask
            )

            logq_weights = jnp.where(
                tweet_counts == 0.0,
                1.0,
                1.0 / tweet_counts,
            )
            raw_weights = logq_weights if raw_weights is None else raw_weights * logq_weights

        if self.config.use_seqpack:
            layout = batch.get("packing_layout")
            assert layout is not None

            attn_config = self.config.model_config.attn_config
            assert attn_config is not None
            assert attn_config.attn_impl in (
                "pallas_ranker_varlen_attn",
                "cutedsl_ranker_varlen_attn",
            ), f"unsupported attn_impl for seqpack: {attn_config.attn_impl}"

            segment_ids = cast_jax(layout.segment_ids)
            positions = cast_jax(layout.positions)
            candidate_positions = cast_jax(layout.candidate_positions)
            cu_seqlens = cast_jax(layout.cu_seqlens)
            packed_padding_mask = cast_jax(layout.padding_mask)
            packed_candidate_seq_len = candidate_positions.shape[1] // bs_per_device

            target_padding_mask = jnp.take_along_axis(
                packed_padding_mask, candidate_positions, axis=1
            )

            history_len = (
                jnp.diff(cu_seqlens, axis=1)
                - self.config.num_user_prefix_tokens
                - packed_candidate_seq_len
            )
            history_actions = batch["history_seq"]["actions"]
            history_post_hashes = batch["history_seq"]["post_hashes"]
            assert history_actions is not None
            history_actions_packed = cast_jax(history_actions)
            history_post_hashes_packed = cast_jax(history_post_hashes)
            history_event_count = packed_per_user_history_event_count(
                history_actions_packed,
                history_post_hashes_packed,
                history_len,
            )
            new_user_mask = jnp.repeat(
                (history_event_count < 128).astype(target_padding_mask.dtype),
                packed_candidate_seq_len,
                axis=1,
            )
            no_history_mask = jnp.repeat(
                (history_event_count == 0).astype(target_padding_mask.dtype),
                packed_candidate_seq_len,
                axis=1,
            )

            with Summary() as summarizer:
                candidate_logits, candidate_continuous_preds = self(
                    input_embeddings,
                    padding_mask,
                    is_training=is_training,
                    positions=positions,
                    segment_ids=segment_ids,
                    seqpack_layout=layout,
                    product_surface=product_surface,
                )
        else:
            assert candidate_start_offset is not None

            (
                input_embeddings,
                padding_mask,
                product_surface,
                targets,
                raw_weights,
                negative_sample_mask,
                promoted_ids,
                candidate_continuous_actions,
                client_app_id,
                line_item_objective,
                candidate_safety_mask,
            ) = pad_to_next_128_multiple(
                input_embeddings,
                padding_mask,
                product_surface,
                targets,
                raw_weights,
                negative_sample_mask,
                promoted_ids,
                candidate_continuous_actions,
                client_app_id,
                line_item_objective,
                candidate_safety_mask,
            )

            idx = jnp.arange(padding_mask.shape[1], dtype=jnp.int32)[None, :]
            segment_ids = jnp.broadcast_to(
                jnp.where(padding_mask, jnp.where(idx >= candidate_start_offset, -1, 1), 0),
                padding_mask.shape,
            ).astype(jnp.int32)

            if self.config.right_anchored_rope:
                positions = right_anchored_rope_positions(
                    padding_mask,
                    self.config.history_seq_len,
                    self.config.num_user_prefix_tokens,
                )
            else:
                positions = jnp.full(
                    (padding_mask.shape[0], padding_mask.shape[1], 3), 0, dtype=jnp.float32
                )
                seq_indices = jnp.arange(padding_mask.shape[1])[None, :]
                is_candidate = seq_indices >= candidate_start_offset

                positions = positions.at[:, :, 0].set(
                    jnp.where(is_candidate, candidate_start_offset, seq_indices)
                )

            assert self.config.model_config.attn_config is not None
            using_ranker_attn = self.config.model_config.attn_config.attn_impl in (
                "pallas_ranker_attn",
                "pallas_ranker_attn_infer",
                "cutedsl_ranker_attn",
            )
            assert using_ranker_attn

            with Summary() as summarizer:
                candidate_logits, candidate_continuous_preds = self(
                    input_embeddings,
                    padding_mask,
                    is_training=is_training,
                    positions=positions,
                    segment_ids=segment_ids.astype(jnp.int32),
                    candidate_start_offset=candidate_start_offset,
                    product_surface=product_surface,
                )

            target_padding_mask = padding_mask[:, candidate_start_offset:]

            history_padding = padding_mask[
                :,
                self.config.num_user_prefix_tokens : candidate_start_offset,
            ]
            num_history_actions = history_padding.sum(axis=1)
            new_user_mask = (num_history_actions < 128).astype(target_padding_mask.dtype)[:, None]
            no_history_mask = (num_history_actions == 0).astype(target_padding_mask.dtype)[:, None]

        assert targets is not None
        assert negative_sample_mask is not None

        targets_for_loss = targets
        if self.config.mask_candidate_positive_when_negative_action_present:
            negative_feedback_action_indices = jnp.array(
                [
                    action_type_map["ServerTweetReport"],
                    action_type_map["ClientTweetNotInterestedIn"],
                    action_type_map["ClientTweetSeeFewer"],
                    action_type_map["ClientTweetUnfollowAuthor"],
                    action_type_map["ClientTweetBlockAuthor"],
                    action_type_map["ClientTweetMuteAuthor"],
                    action_type_map["ClientTweetMuteConversation"],
                    action_type_map["ClientTweetNotRelevant"],
                    action_type_map["ClientNotificationSeeLessOften"],
                ],
                dtype=jnp.int32,
            )
            negative_feedback_action_mask = (
                jnp.zeros(targets_for_loss.shape[-1], dtype=jnp.bool_)
                .at[negative_feedback_action_indices]
                .set(True)
            )
            non_negative_feedback_action_mask = ~negative_feedback_action_mask
            has_negative_feedback_action = jnp.any(
                targets_for_loss[:, :, negative_feedback_action_indices], axis=-1
            )
            has_negative_feedback_action_expanded = jnp.expand_dims(
                has_negative_feedback_action, axis=-1
            )
            zero_mask = has_negative_feedback_action_expanded & non_negative_feedback_action_mask
            targets_for_loss = jnp.where(
                zero_mask,
                jnp.asarray(False, dtype=targets_for_loss.dtype),
                targets_for_loss,
            )

        num_actions = targets.shape[-1]
        loss_mask = jnp.ones((*target_padding_mask.shape, num_actions))

        if self.config.mask_neg_feedback_on_negatives:
            neg_head_mask = (
                jnp.zeros(num_actions).at[jnp.array(NEGATIVE_FEEDBACK_HEAD_INDICES)].set(1.0)
            )
            zero_mask = negative_sample_mask[:, :, None] * neg_head_mask
            loss_mask = loss_mask * (1 - zero_mask)

        if self.config.condition_conversion_on_click:
            has_click = targets[:, :, CLICK_ACTION_INDEX]
            no_click = 1 - has_click
            conv_head_mask = (
                jnp.zeros(num_actions).at[jnp.array(CLICK_CONDITIONED_ACTION_INDICES)].set(1.0)
            )
            conv_zero_mask = no_click[:, :, None] * conv_head_mask
            loss_mask = loss_mask * (1 - conv_zero_mask)

        safety_stats = safety_filter_stats(
            candidate_safety_mask,
            target_padding_mask,
            bits=self.config.safety_filter_bits,
            prefix="safety_filter_candidates",
        )
        if self.config.safety_filter_apply_to_candidates:
            target_padding_mask, raw_weights = apply_safety_filter(
                candidate_safety_mask,
                target_padding_mask,
                raw_weights,
                mode=self.config.safety_filter_mode,
                bits=self.config.safety_filter_bits,
                soft_weight=self.config.safety_filter_soft_weight,
            )

        (
            loss,
            mask,
        ) = multihot_loss_compute(
            logits=candidate_logits,
            raw_targets=targets_for_loss,
            loss_mask=loss_mask,
            padding_mask=target_padding_mask,
            raw_weights=raw_weights,
            one_hot_targets_sharding=P(self.data_axis, ("seq", "model")),
        )

        stats = summarizer.get()
        stats.update(safety_stats)
        stats["safety_filter_active"] = jnp.float32(
            1.0 if self.config.safety_filter_mode != "off" else 0.0
        )
        stats["origin-loss"] = loss

        stats = self.compute_recsys_metrics(
            raw_targets=targets,
            mask=mask.astype(jnp.bool_),
            product_surface=product_surface,
            negative_sample_mask=negative_sample_mask,
            logits=candidate_logits,
            client_app_id=client_app_id,
            promoted_ids=promoted_ids,
            new_user_mask=new_user_mask,
            line_item_objective=line_item_objective,
            no_history_mask=no_history_mask,
            stats=stats,
            rce_ema=rce_ema,
            rce_alpha=rce_alpha,
            smoothing_windows=smoothing_windows,
            calib_ema=calib_ema,
            raw_weights=raw_weights,
        )
        stats = self.compute_timestamp_metrics(batch=batch, stats=stats)
        stats = self.compute_sid_metrics(batch=batch, stats=stats)
        stats = self.compute_engagement_count_metrics(batch=batch, stats=stats)
        stats = self.compute_author_nsfw_metrics(batch=batch, stats=stats)

        if self.config.use_seqpack and batch.get("packing_layout") is not None:
            stats = self.compute_length_bucketed_metrics(
                raw_targets=targets,
                logits=candidate_logits,
                mask=mask.astype(jnp.bool_),
                history_len=history_len,
                packed_candidate_seq_len=packed_candidate_seq_len,
                stats=stats,
                raw_weights=raw_weights,
            )

        continuous_action_loss_total = jnp.array(0.0)
        if candidate_continuous_actions is not None:
            data_num_continuous = candidate_continuous_actions.shape[-1]

            continuous_masks = self._build_metric_masks(
                mask=target_padding_mask,
                raw_targets=targets,
                negative_sample_mask=negative_sample_mask,
                product_surface=product_surface,
                new_user_mask=new_user_mask,
                no_history_mask=no_history_mask,
            )

            for loss_config in self.config.continuous_action_losses:
                if loss_config.loss_weight > 0 and loss_config.action_index < data_num_continuous:
                    gt_raw = candidate_continuous_actions[:, :, loss_config.action_index]
                    pred_raw = candidate_continuous_preds[:, :, loss_config.action_index]

                    head_valid_mask = target_padding_mask
                    if loss_config.product_surfaces or loss_config.exclude_product_surfaces:
                        head_surface_mask = _get_surface_mask(loss_config, product_surface)
                        head_valid_mask = head_valid_mask * head_surface_mask.astype(
                            head_valid_mask.dtype
                        )

                    if loss_config.loss_type == "tweedie":
                        (
                            action_loss,
                            gt_clamped,
                            pred_in_original_units,
                            _cont_loss_mask,
                            per_element_loss,
                        ) = tweedie_loss_compute(
                            gt_raw=gt_raw,
                            pred_raw=pred_raw,
                            valid_mask=head_valid_mask,
                            negative_sample_mask=negative_sample_mask,
                            p=loss_config.tweedie_power,
                            norm_scale=loss_config.norm_config.norm_scale,
                            mask_negatives=loss_config.mask_negatives,
                            raw_weights=raw_weights,
                        )
                    else:
                        (
                            action_loss,
                            gt_clamped,
                            pred_in_original_units,
                            _cont_loss_mask,
                            per_element_loss,
                        ) = continuous_loss_compute(
                            gt_raw=gt_raw,
                            pred_raw=pred_raw,
                            valid_mask=head_valid_mask,
                            negative_sample_mask=negative_sample_mask,
                            norm_scale=loss_config.norm_config.norm_scale,
                            loss_type=loss_config.loss_type,
                            mask_negatives=loss_config.mask_negatives,
                            raw_weights=raw_weights,
                        )
                    continuous_action_loss_total += loss_config.loss_weight * action_loss

                    assert loss_config.metric_name is not None
                    for mask_suffix, variant_mask in continuous_masks.items():
                        head_variant_mask = variant_mask
                        if loss_config.product_surfaces or loss_config.exclude_product_surfaces:
                            head_variant_mask = head_variant_mask * head_surface_mask.astype(
                                head_variant_mask.dtype
                            )
                        if loss_config.mask_negatives:
                            variant_loss_mask = head_variant_mask & (~negative_sample_mask)
                        else:
                            variant_loss_mask = head_variant_mask
                        n_variant = jnp.sum(variant_loss_mask)
                        variant_loss = jnp.sum(per_element_loss * variant_loss_mask) / jnp.maximum(
                            n_variant, 1.0
                        )

                        compute_continuous_metrics(
                            gt_raw=gt_raw,
                            gt_clamped=gt_clamped,
                            pred_in_original_units=pred_in_original_units,
                            valid_mask=head_variant_mask,
                            loss_mask=variant_loss_mask,
                            loss=variant_loss,
                            loss_weight=loss_config.loss_weight,
                            metric_name=loss_config.metric_name,
                            mask_suffix=mask_suffix,
                            stats=stats,
                            mean_baseline=self.config.continuous_metrics_mae_mean,
                        )

        if self.config.multimodal_embedding_type is not None and mm_emb is not None:
            if self.config.use_seqpack:
                packed_candidate_seq_len = mm_emb.shape[1] // batch["user_hashes"].shape[1]
                candidate_seq_len = self.config.candidate_seq_len
                num_devices, _, emb_dim = mm_emb.shape

                mm_emb_by_user = mm_emb.reshape(num_devices, -1, packed_candidate_seq_len, emb_dim)
                padding_by_user = target_padding_mask.reshape(
                    target_padding_mask.shape[0], -1, packed_candidate_seq_len
                )

                mm_emb_positives = mm_emb_by_user[:, :, :candidate_seq_len, :].reshape(
                    num_devices, -1, emb_dim
                )
                candidate_padding = padding_by_user[:, :, :candidate_seq_len].reshape(
                    target_padding_mask.shape[0], -1
                )
            else:
                assert candidate_start_offset is not None

                mm_emb_positives = mm_emb[:, : self.config.candidate_seq_len, :]
                candidate_padding = padding_mask[
                    :,
                    candidate_start_offset : candidate_start_offset + self.config.candidate_seq_len,
                ]

            mm_l2_norms = jnp.sqrt(jnp.sum(mm_emb_positives.astype(jnp.float32) ** 2, axis=-1))

            num_valid = jnp.maximum(candidate_padding.sum(), 1.0)
            stats["mm_embedding_l2_norm_mean"] = (
                jnp.sum(mm_l2_norms * candidate_padding) / num_valid
            )

            is_zero = (mm_l2_norms < 1e-8).astype(jnp.float32)
            stats["mm_embedding_zero_frac"] = jnp.sum(is_zero * candidate_padding) / num_valid
            stats["mm_embedding_l2_norm_max"] = jnp.max(
                jnp.where(candidate_padding, mm_l2_norms, 0.0)
            )

        if self.config.search_query_embedding_dim > 0:
            sq_emb_raw = batch["candidate_seq"]["search_query_embeddings"]
            if sq_emb_raw is not None:
                assert candidate_start_offset is not None
                sq_emb = cast_jax(sq_emb_raw)
                sq_emb_positives = sq_emb[:, : self.config.candidate_seq_len, :]
                sq_candidate_padding = padding_mask[
                    :,
                    candidate_start_offset : candidate_start_offset + self.config.candidate_seq_len,
                ]

                sq_l2_norms = jnp.sqrt(jnp.sum(sq_emb_positives.astype(jnp.float32) ** 2, axis=-1))
                sq_num_valid = jnp.maximum(sq_candidate_padding.sum(), 1.0)
                stats["search_query_embedding_l2_norm_mean"] = (
                    jnp.sum(sq_l2_norms * sq_candidate_padding) / sq_num_valid
                )
                sq_is_zero = (sq_l2_norms < 1e-8).astype(jnp.float32)
                stats["search_query_embedding_zero_frac"] = (
                    jnp.sum(sq_is_zero * sq_candidate_padding) / sq_num_valid
                )
                stats["search_query_embedding_l2_norm_max"] = jnp.max(
                    jnp.where(sq_candidate_padding, sq_l2_norms, 0.0)
                )

        regularization_loss = (
            loss + self.config.act_l2_weight * stats["act-l2-loss"] + continuous_action_loss_total
        )
        return (
            regularization_loss,
            stats,
        )

    @hk.transparent
    def forward(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> tuple[jax.Array, jax.Array]:
        targets = batch["candidate_seq"]["actions"]
        product_surface = batch["candidate_seq"]["product_surface"]
        if targets is not None:
            targets = cast_jax(targets)

        product_surface = cast_jax(product_surface)

        input_embeddings, padding_mask, candidate_start_offset, _mm_emb = self.build_inputs(
            batch,
            recsys_embeddings,
            is_training=False,
        )

        if self.config.use_seqpack:
            layout = batch.get("packing_layout")
            assert layout is not None

            attn_config = self.config.model_config.attn_config
            assert attn_config is not None
            assert attn_config.attn_impl in (
                "pallas_ranker_varlen_attn",
                "cutedsl_ranker_varlen_attn",
            ), f"unsupported attn_impl for seqpack: {attn_config.attn_impl}"

            segment_ids = cast_jax(layout.segment_ids)
            positions = cast_jax(layout.positions)

            candidate_logits, candidate_continuous_predictions = self(
                input_embeddings,
                padding_mask,
                is_training=False,
                positions=positions,
                segment_ids=segment_ids,
                seqpack_layout=layout,
                product_surface=product_surface,
            )
        else:
            assert candidate_start_offset is not None

            input_embeddings, padding_mask, product_surface, targets, *_ = pad_to_next_128_multiple(
                input_embeddings,
                padding_mask,
                product_surface,
                targets,
                None,
                None,
                None,
                None,
                None,
            )
            idx = jnp.arange(padding_mask.shape[1], dtype=jnp.int32)[None, :]
            segment_ids = jnp.broadcast_to(
                jnp.where(padding_mask, jnp.where(idx >= candidate_start_offset, -1, 1), 0),
                padding_mask.shape,
            ).astype(jnp.int32)

            if self.config.right_anchored_rope:
                positions = right_anchored_rope_positions(
                    padding_mask,
                    self.config.history_seq_len,
                    self.config.num_user_prefix_tokens,
                )
            else:
                positions = jnp.full(
                    (padding_mask.shape[0], padding_mask.shape[1], 3), 0, dtype=jnp.float32
                )
                seq_indices = jnp.arange(padding_mask.shape[1])[None, :]
                is_candidate = seq_indices >= candidate_start_offset

                positions = positions.at[:, :, 0].set(
                    jnp.where(is_candidate, candidate_start_offset, seq_indices)
                )

            assert self.config.model_config.attn_config is not None
            using_ranker_attn = self.config.model_config.attn_config.attn_impl in (
                "pallas_ranker_attn",
                "pallas_ranker_attn_infer",
                "cutedsl_ranker_attn",
            )
            assert using_ranker_attn

            candidate_logits, candidate_continuous_predictions = self(
                input_embeddings,
                padding_mask,
                is_training=False,
                positions=positions,
                segment_ids=segment_ids.astype(jnp.int32),
                candidate_start_offset=candidate_start_offset,
                product_surface=product_surface,
            )

        for loss_config in self.config.continuous_action_losses:
            act = loss_config.activation
            if act != "sigmoid":
                continue
            idx = loss_config.action_index
            scale = loss_config.norm_config.norm_scale
            preds = candidate_continuous_predictions[:, :, idx]
            if loss_config.product_surfaces or loss_config.exclude_product_surfaces:
                smask = _get_surface_mask(loss_config, product_surface)
                preds = jnp.where(smask, preds * scale, preds)
            else:
                preds = preds * scale
            candidate_continuous_predictions = candidate_continuous_predictions.at[:, :, idx].set(
                preds
            )

        return candidate_logits, candidate_continuous_predictions
