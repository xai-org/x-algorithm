# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import functools
import logging
import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from numpy import typing as npt

from xai_configlib import Config as Config
from xai_configlib import configclass as configclass
from xai_proto import recsys_pb2
from xrex.data.recsys.ads_head_masking import (
    ADS_FAMILY_SLICE_HEADS,
    EARLY_RELABEL_STREAM_ID,
    FRESH_STREAM_ID,
    LATE_SPLIT_STREAM_ID,
    metric_head_mask_keys,
)
from xrex.data.recsys.ads_late_window import (
    ADS_LATE_WINDOW_TWIN_HEAD_INDICES,
    ADS_LATE_WINDOW_VT_TWIN_HEAD_INDICES,
)
from xrex.data.recsys.constants import (
    CLICK_ACTION_INDEX,
    CLICK_CONDITIONED_ACTION_INDICES,
    CONVERSION_DELAY_HEAD_COLUMNS,
    CONVERSION_DELAY_NONE,
    MACT_IN_APP_LOSS_ACTION_INDICES,
    MMP_CLICK_ACTION_INDEX,
    NEGATIVE_FEEDBACK_HEAD_INDICES,
    PURCHASE_VALUE_ACTION_INDEX,
    SEARCH_RELEVANCE_ACTION_INDICES,
    VIEW_LOSS_ACTION_INDICES,
    LineItemObjective,
    action_type_map,
    engagement_to_ids,
)
from xrex.data.recsys.feature_config import (
    ADS_COUNT_BUCKETS,
    ADS_PRODUCT_KEY_TABLE_SIZE,
    ENGAGEMENT_COUNT_BUCKET_MAP,
    BoolFeature,
    CategoricalFeature,
    FloatFeature,
    Int64Feature,
)
from xrex.data.recsys.recsys_batch import (
    AUX_VALUE_EVENTS,
    EMBEDDING_CONFIG,
    VALUE_LINE_ITEM_KEY,
    VALUE_LINE_ITEM_OPERAND,
    EmbeddingType,
    RecsysFeaturesBatch,
    aux_value_label_keys,
)
from xrex.data.recsys.safety_filter import apply_safety_filter, safety_filter_stats
from xrex.data.recsys.sequence_packing import SequencePackedLayout, packed_prefix_positions
from xrex.models.layers import Linear, get_parameter
from xrex.models.loss_recsys import (
    binary_threshold_loss_compute,
    continuous_loss_compute,
    continuous_loss_weights,
    cread_log_thresholds,
    cread_loss_compute,
    multihot_loss_compute,
    multihot_loss_weights,
    tweedie_loss_compute,
)
from xrex.models.model_utils import Parameter
from xrex.models.normalization import rms_norm_fn
from xrex.models.page_rl import (
    REALIZED_NEGATIVE_ACTIONS,
    REALIZED_POSITIVE_ACTIONS,
    PageDecodeInputs,
    PageRlConfig,
    page_decode_noop_outputs,
    page_rl_head_features,
    page_rl_loss_and_stats,
    run_page_decode,
)
from xrex.models.recsys_attention import RecsysAttentionConfig
from xrex.models.recsys_embedding import (
    HashKeys,
    HashTable,
    RecsysEmbeddings,
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_feature_prep import (
    DWELL_TIME_DEFAULT_SENTINEL_SEC,
    MAX_VALID_DWELL_SEC,
    MIN_VALID_DWELL_SEC,
    FeaturePrepConfig,
    build_feature_prep_inputs,
)
from xrex.models.recsys_sid import (
    reconstruct_entity_sid,
    sid_prefix2_lookup,
    sid_prefix3_lookup,
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
from xrex.models.value_bin_tables import (
    AUX_VALUE_BIN_TABLES,
    PURCHASE_VALUE_BIN_PRIOR,
    PURCHASE_VALUE_BIN_VALUES,
    VALUE_BIN_EDGES,
    VALUE_BIN_SMOOTHING,
)
from xrex.models.value_heads import (
    PURCHASE_VALUE_EMA_KEY,
    PURCHASE_VALUE_METRIC_PREFIX,
    PURCHASE_VALUE_VT_METRIC_PREFIX,
    VALUE_EMA_STATE_SIZE,
    WITHIN_LI_SLICES,
    WITHIN_LI_STATE_COLS,
    ValueBinTable,
    value_binned_sums_key,
    value_head_loss_compute,
    value_head_stats,
    value_smoothed_stats,
    value_sums_key,
    value_valid_mask,
    value_valid_weights,
    value_within_li_smoothed_stats,
    within_li_ema_key,
)
from xrex.utils.sharding import with_sharding_constraint_unless_manual
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

RECENCY_ACTIONS = (
    recsys_pb2.ActionName.CLIENT_TWEET_OPEN_LINK,
    recsys_pb2.ActionName.CLIENT_TWEET_EXTERNAL_LINK_LONG_DWELLED,
)
RECENCY_LOG_DT_SCALE = 15.0
RECENCY_FOURIER = 8


def recency_time_features(x: jax.Array, k: int = RECENCY_FOURIER) -> jax.Array:
    freqs = jnp.arange(1, k + 1, dtype=jnp.float32) * jnp.pi
    return jnp.concatenate([x, jnp.sin(x * freqs), jnp.cos(x * freqs)], axis=-1)


_DPA_PRODUCT_KEY_SLOTS: list[int] = [
    Int64Feature.firstDpaProductKey.value,
    Int64Feature.firstDpaProductKeyHash2.value,
]

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


MATCHED_WORD_FRACTION_NUM_BUCKETS = 6


def compute_matched_word_fraction_bucket(fraction: jax.Array, is_set: jax.Array) -> jax.Array:
    f = fraction.astype(jnp.float32)
    bucket = jnp.where(f <= 0.0, 1, jnp.where(f < 0.5, 2, jnp.where(f == 0.5, 3, 4)))
    bucket = jnp.where(f >= 1.0, MATCHED_WORD_FRACTION_NUM_BUCKETS - 1, bucket)
    return jnp.where(is_set, bucket, 0)


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


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def take_candidate_slots(
    num_rows: int,
    embeddings: jax.Array,
    candidate_positions: jax.Array,
    candidate_token_to_slot: jax.Array,
) -> jax.Array:
    del num_rows, candidate_token_to_slot
    return jnp.take_along_axis(embeddings, candidate_positions[:, :, None], axis=1)


def _take_candidate_slots_fwd(num_rows, embeddings, candidate_positions, candidate_token_to_slot):
    slot_maps = (candidate_positions, candidate_token_to_slot)
    return take_candidate_slots(num_rows, embeddings, *slot_maps), slot_maps


def _take_candidate_slots_bwd(num_rows, slot_maps, d_candidates):
    candidate_positions, candidate_token_to_slot = slot_maps
    d_region = jnp.take_along_axis(
        d_candidates, candidate_token_to_slot[:, :, None], axis=1, mode="fill", fill_value=0
    )
    is_invalid_slot = (candidate_positions == num_rows - 1)[:, :, None]
    d_last_row = jnp.sum(d_candidates, axis=1, where=is_invalid_slot)
    d_region = d_region.at[:, -1].set(d_last_row)
    num_region_tokens = candidate_token_to_slot.shape[1]
    d_embeddings = jnp.pad(d_region, ((0, 0), (num_rows - num_region_tokens, 0), (0, 0)))
    return d_embeddings, None, None


take_candidate_slots.defvjp(_take_candidate_slots_fwd, _take_candidate_slots_bwd)


@configclass
class NormConfig(Config):
    norm_scale: float = 30.0

    use_log: bool = False


@configclass
class ContinuousActionLossConfig(Config):
    action_index: int = 0

    loss_weight: float = 0.0

    loss_type: Literal["mse", "mae", "huber", "tweedie", "binary", "cread"] = "mae"

    tweedie_power: float = 1.5

    binary_threshold: float = 10.0

    cread_log_min_threshold: float = 2.0

    cread_log_num_thresholds: int = 30

    cread_restoration_weight: float = 0.5

    cread_restoration_loss: Literal["mae", "huber"] = "huber"

    cread_huber_delta_sec: float = 5.0

    cread_mask_sentinel_labels: bool = True

    activation: Literal["sigmoid", "softplus"] | None = None

    output_cap: float = -1.0

    norm_config: NormConfig = NormConfig()

    metric_name: str | None = None

    mask_negatives: bool = True

    product_surfaces: tuple[int, ...] = ()

    exclude_product_surfaces: tuple[int, ...] = ()
    condition_action_index: int | None = None

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
            elif self.loss_type == "binary":
                self.activation = "sigmoid"
        if self.loss_type == "binary":
            assert self.activation == "sigmoid", (
                f"loss_type='binary' outputs a probability and requires activation='sigmoid', "
                f"got {self.activation}"
            )
            assert self.binary_threshold > 0, (
                f"binary_threshold must be positive, got {self.binary_threshold}"
            )
            assert self.output_cap <= 0, (
                "loss_type='binary' outputs a probability in [0, 1]; output_cap does not apply"
            )

        if self.loss_type == "cread":
            _ = self.cread_threshold_values
            if not 0.0 < self.cread_huber_delta_sec < self.norm_config.norm_scale:
                raise ValueError(
                    f"cread_huber_delta_sec ({self.cread_huber_delta_sec}) must be in "
                    f"(0, norm_scale={self.norm_config.norm_scale})"
                )

    @property
    def cread_threshold_values(self) -> tuple[float, ...]:
        return cread_log_thresholds(
            self.norm_config.norm_scale,
            self.cread_log_min_threshold,
            self.cread_log_num_thresholds,
        )


def validate_continuous_action_losses(losses: list[ContinuousActionLossConfig]) -> None:
    seen: dict[tuple[int, tuple[int, ...], tuple[int, ...]], str | None] = {}
    for lc in losses:
        if lc.loss_weight <= 0:
            continue
        key = (lc.action_index, tuple(lc.product_surfaces), tuple(lc.exclude_product_surfaces))
        if key in seen:
            raise ValueError(
                f"continuous_action_losses entries {seen[key]!r} and {lc.metric_name!r} are "
                "both active (loss_weight > 0) with the same action index and surface "
                "filters; zero out the one being replaced"
            )
        seen[key] = lc.metric_name


@configclass
class AuxValueHeadConfig(Config):
    event: str = "content_view"

    loss_weight: float = 0.0

    bin_edges: tuple[float, ...] = VALUE_BIN_EDGES
    bin_values: tuple[float, ...] = ()
    bin_prior: tuple[float, ...] = ()
    bin_smoothing: float = VALUE_BIN_SMOOTHING

    smoothing_windows: tuple[int, ...] = (1 << 26, 1 << 28)

    lanes: tuple[int, ...] = (EARLY_RELABEL_STREAM_ID,)

    include_view_through: bool = False

    def __post_init__(self):
        table = AUX_VALUE_BIN_TABLES.get(self.event)
        if table is not None and self.bin_edges == table.edges:
            self.bin_values = self.bin_values or table.values
            self.bin_prior = self.bin_prior or table.prior

    @property
    def param_prefix(self) -> str:
        return f"aux_value_{self.event}"

    @property
    def metric_prefix(self) -> str:
        population = "delayed_website" if self.include_view_through else "delayed_website_clicked"
        return f"aux-value-{self.event.replace('_', '-')}_{population}"

    @property
    def bins(self) -> ValueBinTable:
        return ValueBinTable(self.bin_edges, self.bin_values, self.bin_prior, self.bin_smoothing)

    def validate(self) -> None:
        if self.event not in AUX_VALUE_EVENTS:
            raise ValueError(
                f"aux value head event {self.event!r} must be one of {[*AUX_VALUE_EVENTS]}"
            )
        if not math.isfinite(self.loss_weight) or self.loss_weight < 0:
            raise ValueError(f"aux value head {self.event}: loss_weight must be finite and >= 0")
        if self.loss_weight == 0:
            return
        self.bins.validate(self.param_prefix)
        if not self.smoothing_windows or any(w <= 0 for w in self.smoothing_windows):
            raise ValueError(f"aux value head {self.event}: smoothing_windows must be positive")
        valid_lanes = (EARLY_RELABEL_STREAM_ID, LATE_SPLIT_STREAM_ID)
        if not self.lanes or any(lane not in valid_lanes for lane in self.lanes):
            raise ValueError(
                f"aux value head {self.event}: lanes must be a subset of {valid_lanes}"
            )


def parse_aux_value_heads(
    weights: str, head: str = "binned", include_view_through: bool = False
) -> tuple[AuxValueHeadConfig, ...]:
    if head != "binned":
        raise ValueError(f"aux_value_head_type must be 'binned', got {head!r}")
    heads: list[AuxValueHeadConfig] = []
    for item in weights.split(","):
        if not item.strip():
            continue
        event, sep, raw_weight = (part.strip() for part in item.partition(":"))
        if event not in AUX_VALUE_EVENTS:
            raise ValueError(
                f"aux_value_head_weights: unknown event {event!r} in {item!r}, "
                f"expected event:weight with event in {[*AUX_VALUE_EVENTS]}"
            )
        try:
            weight = float(raw_weight) if sep else math.nan
        except ValueError:
            weight = math.nan
        if not (math.isfinite(weight) and weight > 0):
            raise ValueError(
                f"aux_value_head_weights: {item!r} needs a finite weight > 0 (event:weight)"
            )
        if any(h.event == event for h in heads):
            raise ValueError(f"aux_value_head_weights: event {event!r} given twice")
        heads.append(
            AuxValueHeadConfig(
                event=event, loss_weight=weight, include_view_through=include_view_through
            )
        )
    return tuple(heads)


@dataclass(frozen=True)
class ValueHead:
    metric_prefix: str
    ema_key: str
    loss_weight: float
    smoothing_windows: tuple[int, ...]
    valid_key: str
    baseline_key: str
    lanes: tuple[int, ...]
    include_view_through: bool
    bins: ValueBinTable | None
    huber_delta: float = 1.0
    slot: int | None = None
    event: str | None = None
    ratio_key: str | None = None


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


def _rescale_sigmoid_heads_for_inference(
    candidate_continuous_predictions: jax.Array,
    continuous_action_losses: Sequence[ContinuousActionLossConfig],
    product_surface: jax.Array,
) -> jax.Array:
    for loss_config in continuous_action_losses:
        if (
            loss_config.activation != "sigmoid"
            or loss_config.loss_type == "binary"
            or loss_config.loss_weight <= 0
        ):
            continue
        idx = loss_config.action_index
        scale = loss_config.norm_config.norm_scale
        preds = candidate_continuous_predictions[:, :, idx]
        if loss_config.product_surfaces or loss_config.exclude_product_surfaces:
            smask = _get_surface_mask(loss_config, product_surface)
            preds = jnp.where(smask, preds * scale, preds)
        else:
            preds = preds * scale
        candidate_continuous_predictions = candidate_continuous_predictions.at[:, :, idx].set(preds)
    return candidate_continuous_predictions


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

    enable_day_of_week: bool = True

    enable_search_lexical_match: bool = False

    enable_web_conv_tracking_integration: bool = False


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


_XAUC_MAX_SAMPLES = 2048


def metric_xauc(
    pred: jnp.ndarray,
    gt: jnp.ndarray,
    valid_mask: jnp.ndarray,
    max_samples: int = _XAUC_MAX_SAMPLES,
) -> jnp.ndarray:
    pred = pred.reshape(-1).astype(jnp.float32)
    gt = gt.reshape(-1).astype(jnp.float32)
    mask = valid_mask.reshape(-1).astype(jnp.bool_)

    stride = max(pred.shape[0] // max_samples, 1)
    pred = pred[::stride][:max_samples]
    gt = gt[::stride][:max_samples]
    mask = mask[::stride][:max_samples]

    dy = gt[:, None] - gt[None, :]
    dp = pred[:, None] - pred[None, :]
    pair_valid = mask[:, None] & mask[None, :] & (dy != 0)
    concordant = (jnp.sign(dy) == jnp.sign(dp)) & pair_valid
    tied = (dp == 0) & pair_valid

    num_pairs = jnp.sum(pair_valid)
    xauc = (jnp.sum(concordant) + 0.5 * jnp.sum(tied)) / jnp.maximum(num_pairs, 1)
    return jnp.where(num_pairs > 0, xauc, 0.0)


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


def ne_from_rce(rce: jax.Array) -> jax.Array:
    return 1.0 - rce / 100.0


def metric_prauc(
    p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray, auc_thresholds: jnp.ndarray
) -> jnp.ndarray:
    valid_y = y * valid_mask
    num_pos = jnp.sum(valid_y)

    p = p.reshape(-1)
    bucket = jnp.searchsorted(auc_thresholds, p, side="right", method="scan_unrolled")
    bucket = jnp.where(jnp.isnan(p), 0, bucket)
    weights = jnp.stack([valid_mask.reshape(-1), valid_y.reshape(-1)], axis=1).astype(jnp.float32)
    num_buckets = auc_thresholds.shape[0] + 1
    bucket = jnp.where(weights[:, 0] != 0, bucket, num_buckets)
    hist = jnp.zeros((num_buckets, 2), jnp.float32).at[bucket].add(weights, mode="drop")
    cleared = jnp.cumsum(hist[::-1], axis=0)[::-1]
    sum_pred, sum_pred_y = cleared[1:].T
    precisions = sum_pred_y / jnp.maximum(sum_pred, 1e-6)
    recalls = sum_pred_y / jnp.maximum(num_pos, 1e-6)
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
    return jnp.where(total_valid > 0, num_pos / jnp.maximum(total_valid, 1), 0.0)


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
    return _mean_ndcg(ndcg_scores, valid_mask)


def _mean_ndcg(ndcg_scores: jax.Array, valid_mask: jax.Array) -> jax.Array:
    valid_queries = jnp.sum(valid_mask, axis=1) >= 1
    num_valid_queries = jnp.sum(valid_queries)
    return jnp.where(
        num_valid_queries > 0,
        jnp.sum(jnp.where(valid_queries, ndcg_scores, 0.0)) / num_valid_queries,
        0.0,
    )


def metric_ndcg_masks(p: jax.Array, y: jax.Array, masks: jax.Array) -> jax.Array:
    item_weight = jnp.max(masks, axis=0)
    pred_order = jnp.argsort(-jnp.where(item_weight > 0, p * item_weight, -jnp.inf), axis=-1)
    ideal_order = jnp.argsort(-(y * item_weight), axis=-1)
    in_mask = jnp.moveaxis(masks > 0, 0, -1)
    valid_y = jnp.moveaxis(y * masks, 0, -1)

    def dcg(order: jax.Array) -> jax.Array:
        sorted_gains = jnp.take_along_axis(valid_y, order[..., None], axis=1)
        sorted_in_mask = jnp.take_along_axis(in_mask, order[..., None], axis=1)
        rank = jnp.cumsum(sorted_in_mask, axis=1, dtype=jnp.float32)
        return jnp.sum(jnp.where(sorted_in_mask, sorted_gains / jnp.log2(rank + 1), 0.0), axis=1)

    idcg = dcg(ideal_order)
    ndcg_scores = jnp.where(idcg > 0, dcg(pred_order) / idcg, 1.0)
    return jax.vmap(_mean_ndcg, in_axes=(1, 0))(ndcg_scores, masks)


def metric_calib(p: jnp.ndarray, y: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    num_pos = (valid_mask * y).sum() + _CALIB_POS_BASE
    num_prob_pos = (valid_mask * p).sum()
    return jnp.where(num_pos > 0, num_prob_pos / jnp.maximum(num_pos, 1), 0.0)


def engagement_metrics(p, y, masks, auc_thresholds, *, ndcg_shared_sorts: bool):
    def per_mask(metric):
        return jax.vmap(metric, in_axes=(None, None, 0))(p, y, masks)

    loss, rce = per_mask(metric_loss), per_mask(metric_rce)
    ndcg = metric_ndcg_masks(p, y, masks) if ndcg_shared_sorts else per_mask(metric_ndcg)
    ratio_pos, calib = per_mask(metric_ratio_pos), per_mask(metric_calib)
    prauc = jax.vmap(metric_prauc, in_axes=(None, None, 0, None))(p, y, masks, auc_thresholds)
    num_tokens = jax.vmap(metric_num_tokens, in_axes=(None, 0))(y, masks)
    return jnp.array([loss, rce, ndcg, ratio_pos, calib, prauc, num_tokens])


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
    per_head_mask_values: jax.Array | None = None,
    ndcg_shared_sorts: bool = False,
):
    auc_thresholds = jnp.geomspace(_INITIAL_AUC_THRESHOLD, 1.0, num=_NUM_AUC_THRESHOLDS)
    if per_head_mask_values is None:
        masks, masks_axis = jnp.stack(mask_values), None
    else:
        masks, masks_axis = per_head_mask_values, 0
    res = jax.vmap(
        functools.partial(engagement_metrics, ndcg_shared_sorts=ndcg_shared_sorts),
        in_axes=(0, 0, masks_axis, None),
    )(jnp.stack(p_list), jnp.stack(y_list), masks, auc_thresholds)
    if ndcg_shared_sorts:
        stats["ndcg_negative_weights"] = jnp.sum(masks < 0, dtype=jnp.float32)
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
    stats.update(
        {
            f"{eng_name}_{mask_key}_NE": ne_from_rce(stats[f"{eng_name}_{mask_key}_RCE"])
            for eng_name in eng_names
            for mask_key in mask_keys
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


CONVERSION_DELAY_BASE_MASK = "delayed_website_clicked"


@dataclass(frozen=True)
class ConversionDelaySlice:
    head: str
    name: str
    column_index: int
    threshold_ms: int
    slow: bool

    def mask(self, base: jax.Array, y: jax.Array, delay_ms: jax.Array) -> jax.Array:
        known = delay_ms != CONVERSION_DELAY_NONE
        slow = known & (delay_ms >= self.threshold_ms)
        keep_positive = slow if self.slow else (known & ~slow)
        return base * ((1 - y) + y * keep_positive.astype(base.dtype))


def conversion_delay_base_mask(mask: jax.Array, trained_candidate_mask: jax.Array) -> jax.Array:
    trained = trained_candidate_mask.astype(mask.dtype)
    return mask * trained[:, :, CLICK_CONDITIONED_ACTION_INDICES[0]]


def conversion_delay_slice_metrics(
    slices: Sequence[ConversionDelaySlice],
    prob_labels: dict[str, tuple[jax.Array, jax.Array]],
    base: jax.Array,
    delay_ms: jax.Array,
    raw_weights: jax.Array | None,
    stats: dict,
    per_head_eligible: dict[str, jax.Array] | None = None,
) -> dict[tuple[str, str], jax.Array]:
    auc_thresholds = jnp.geomspace(_INITIAL_AUC_THRESHOLD, 1.0, num=_NUM_AUC_THRESHOLDS)
    masks: dict[tuple[str, str], jax.Array] = {}
    unknown_done: set[str] = set()
    for s in slices:
        p, y = prob_labels[s.head]
        head_delay = delay_ms[..., s.column_index]
        head_base = base
        if per_head_eligible is not None:
            head_base = head_base * per_head_eligible[s.head]
        m = s.mask(head_base, y, head_delay)
        if raw_weights is not None:
            m = m * raw_weights
        masks[(s.head, s.name)] = m
        prefix = f"{s.head}_{s.name}"
        stats[f"{prefix}_loss"] = metric_loss(p, y, m)
        stats[f"{prefix}_RCE"] = metric_rce(p, y, m)
        stats[f"{prefix}_NE"] = ne_from_rce(stats[f"{prefix}_RCE"])
        stats[f"{prefix}_NDCG"] = metric_ndcg(p, y, m)
        stats[f"{prefix}_ratio_pos"] = metric_ratio_pos(p, y, m)
        stats[f"{prefix}_calib"] = metric_calib(p, y, m)
        stats[f"{prefix}_PRAUC"] = metric_prauc(p, y, m, auc_thresholds)
        stats[f"{prefix}_num_tokens"] = metric_num_tokens(y, m)
        stats[f"{prefix}_effective_num_tokens"] = m.sum()
        if s.head not in unknown_done:
            unknown_done.add(s.head)
            unknown = base * y * (head_delay == CONVERSION_DELAY_NONE).astype(base.dtype)
            if raw_weights is not None:
                unknown = unknown * raw_weights
            stats[f"{s.head}_conv_delay_unknown_pos_tokens"] = unknown.sum()
    return masks


_EMA_EPS = 1e-7


def _ema_step(
    ema: dict[str, jax.Array],
    base_key: str,
    batch_stat: jax.Array,
    alpha: jax.Array,
    smoothing_windows: tuple[int, ...],
) -> jax.Array:
    old = jnp.stack(
        [
            ema.get(f"{base_key}/{ws}", jnp.zeros(batch_stat.shape, dtype=jnp.float32))
            for ws in smoothing_windows
        ]
    )
    raw_updated = (1.0 - alpha) * old + alpha * batch_stat[None, :]
    return jnp.where(
        jnp.isnan(raw_updated),
        jnp.where(jnp.isnan(old), batch_stat[None, :], old),
        raw_updated,
    )


def _smoothed_rce(updated: jax.Array) -> jax.Array:
    ema_ce = updated[:, 0] / jnp.maximum(updated[:, 2], 1.0)
    ema_pb = updated[:, 1] / jnp.maximum(updated[:, 2], 1.0)
    pb_c = jnp.clip(ema_pb, _EMA_EPS, 1.0 - _EMA_EPS)
    bce_ref = -(pb_c * jnp.log(pb_c) + (1.0 - pb_c) * jnp.log(1.0 - pb_c))
    s_rce = 100.0 * (bce_ref - ema_ce) / (bce_ref + 1e-12)
    has_both = (updated[:, 1] > 0) & (updated[:, 1] < updated[:, 2])
    return jnp.where(has_both, s_rce, 0.0)


def _smoothed_calib(updated: jax.Array) -> jax.Array:
    s_calib = updated[:, 0] / jnp.maximum(updated[:, 1], _EMA_EPS)
    return jnp.where(updated[:, 1] > 0, s_calib, 0.0)


@configclass
class RecsysAggregatedModelConfig(Config):
    model_config: TransformerConfig
    hash_table: HashTable
    pad_token: int
    emb_table_width: int

    continuous_metrics_mae_mean: bool = False
    per_layer_l2_weight: float = 0.0
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

    purchase_value_loss_weight: float = 0.0
    purchase_value_huber_delta: float = 1.0
    purchase_value_include_view_through: bool = False
    purchase_value_smoothing_windows: tuple[int, ...] = (1 << 28, 1 << 30)
    purchase_value_head: str = "regression"
    purchase_value_bin_edges: tuple[float, ...] = VALUE_BIN_EDGES
    purchase_value_bin_values: tuple[float, ...] = PURCHASE_VALUE_BIN_VALUES
    purchase_value_bin_prior: tuple[float, ...] = PURCHASE_VALUE_BIN_PRIOR
    purchase_value_bin_smoothing: float = VALUE_BIN_SMOOTHING
    purchase_value_history_tower: bool = False
    purchase_value_history_tower_hidden: int = 64
    purchase_value_history_tower_proj: int = 32
    purchase_value_history_ratio_features: bool = False
    purchase_value_within_li_metric: bool = False
    purchase_value_within_li_buckets: int = 1 << 16
    purchase_value_within_li_windows: tuple[int, ...] = ()
    aux_value_heads: tuple[AuxValueHeadConfig, ...] = ()
    aux_value_head_weights: str = ""
    aux_value_head_type: str = "binned"
    aux_value_include_view_through: bool = False
    value_loss_normalization: Literal["valid_rows", "positions"] = "valid_rows"

    @property
    def purchase_value_enabled(self) -> bool:
        return self.purchase_value_loss_weight > 0

    @property
    def purchase_value_binned(self) -> bool:
        return self.purchase_value_enabled and self.purchase_value_head == "binned"

    @property
    def purchase_value_bins(self) -> ValueBinTable:
        return ValueBinTable(
            self.purchase_value_bin_edges,
            self.purchase_value_bin_values,
            self.purchase_value_bin_prior,
            self.purchase_value_bin_smoothing,
        )

    @property
    def all_aux_value_heads(self) -> tuple[AuxValueHeadConfig, ...]:
        return (
            *self.aux_value_heads,
            *parse_aux_value_heads(
                self.aux_value_head_weights,
                self.aux_value_head_type,
                self.aux_value_include_view_through,
            ),
        )

    @property
    def active_aux_value_heads(self) -> tuple[AuxValueHeadConfig, ...]:
        if not self.purchase_value_enabled:
            return ()
        return tuple(head for head in self.all_aux_value_heads if head.loss_weight > 0)

    @property
    def aux_value_events(self) -> tuple[str, ...]:
        events = tuple(head.event for head in self.active_aux_value_heads)
        if self.purchase_value_enabled and self.purchase_value_within_li_metric:
            events = (*events, VALUE_LINE_ITEM_OPERAND)
        return events

    @property
    def within_li_windows(self) -> tuple[int, ...]:
        return self.purchase_value_within_li_windows or self.purchase_value_smoothing_windows

    def value_head_ema_keys(self) -> dict[str, jax.Array]:
        keys = {
            f"{head.ema_key}/{ws}": jnp.zeros((VALUE_EMA_STATE_SIZE,), dtype=jnp.float32)
            for head in self.value_heads()
            for ws in head.smoothing_windows
        }
        if self.purchase_value_enabled and self.purchase_value_within_li_metric:
            shape = (self.purchase_value_within_li_buckets, WITHIN_LI_STATE_COLS)
            keys.update(
                {
                    within_li_ema_key(PURCHASE_VALUE_EMA_KEY, name, ws): jnp.zeros(
                        shape, dtype=jnp.float32
                    )
                    for name in WITHIN_LI_SLICES
                    for ws in self.within_li_windows
                }
            )
        return keys

    def value_heads(self) -> tuple[ValueHead, ...]:
        heads: list[ValueHead] = []
        if self.purchase_value_enabled:
            heads.append(
                ValueHead(
                    metric_prefix=(
                        PURCHASE_VALUE_VT_METRIC_PREFIX
                        if self.purchase_value_include_view_through
                        else PURCHASE_VALUE_METRIC_PREFIX
                    ),
                    ema_key=PURCHASE_VALUE_EMA_KEY,
                    loss_weight=self.purchase_value_loss_weight,
                    smoothing_windows=self.purchase_value_smoothing_windows,
                    valid_key="value_label_valid",
                    baseline_key="value_baseline_mean_usd",
                    lanes=(EARLY_RELABEL_STREAM_ID,),
                    include_view_through=self.purchase_value_include_view_through,
                    bins=self.purchase_value_bins if self.purchase_value_binned else None,
                    huber_delta=self.purchase_value_huber_delta,
                    slot=PURCHASE_VALUE_ACTION_INDEX,
                )
            )
        for head in self.active_aux_value_heads:
            ratio_key, valid_key, baseline_key = aux_value_label_keys(head.event)
            heads.append(
                ValueHead(
                    metric_prefix=head.metric_prefix,
                    ema_key=head.param_prefix,
                    loss_weight=head.loss_weight,
                    smoothing_windows=head.smoothing_windows,
                    valid_key=valid_key,
                    baseline_key=baseline_key,
                    lanes=head.lanes,
                    include_view_through=head.include_view_through,
                    bins=head.bins,
                    event=head.event,
                    ratio_key=ratio_key,
                )
            )
        return tuple(heads)

    def validate_aux_value_heads(self) -> None:
        heads = self.all_aux_value_heads
        events = [head.event for head in heads]
        if len(set(events)) != len(events):
            raise ValueError(f"aux_value_heads events must be unique, got {events}")
        for head in heads:
            head.validate()
        weighted = [head.event for head in heads if head.loss_weight > 0]
        if weighted and not self.purchase_value_enabled:
            raise ValueError(
                f"aux_value_heads {weighted} need purchase_value_loss_weight > 0: the value "
                "heads share one switch so runs without them are untouched"
            )

    def validate_purchase_value(self) -> None:
        if (
            not math.isfinite(self.purchase_value_loss_weight)
            or self.purchase_value_loss_weight < 0
        ):
            raise ValueError("purchase_value_loss_weight must be finite and nonnegative")
        if self.value_loss_normalization not in ("valid_rows", "positions"):
            raise ValueError(
                f"value_loss_normalization must be valid_rows or positions, "
                f"got {self.value_loss_normalization!r}"
            )
        if self.purchase_value_history_tower:
            if not (self.purchase_value_enabled and self.purchase_value_binned):
                raise ValueError(
                    "purchase_value_history_tower needs the binned purchase-value head"
                )
            if self.use_seqpack:
                raise ValueError("purchase_value_history_tower is not supported with use_seqpack")
            if self.num_continuous_actions <= WEB_CONV_VALUE_USD_IDX:
                raise ValueError("purchase_value_history_tower needs history value slot 6")
        if self.purchase_value_history_ratio_features:
            if not self.purchase_value_history_tower:
                raise ValueError(
                    "purchase_value_history_ratio_features needs purchase_value_history_tower"
                )
            if self.num_continuous_actions <= WEB_CONV_VALUE_RATIO_IDX:
                raise ValueError("purchase_value_history_ratio_features needs history slot 7")
        if self.purchase_value_within_li_metric:
            if not self.purchase_value_enabled:
                raise ValueError("purchase_value_within_li_metric needs the purchase-value head")
            if self.use_seqpack:
                raise ValueError(
                    "purchase_value_within_li_metric is not supported with use_seqpack"
                )
            buckets = self.purchase_value_within_li_buckets
            if buckets < 2 or buckets & (buckets - 1):
                raise ValueError("purchase_value_within_li_buckets must be a power of two >= 2")
            if any(w <= 0 for w in self.purchase_value_within_li_windows):
                raise ValueError("purchase_value_within_li_windows must be positive sample counts")
        if not self.purchase_value_enabled:
            return
        if self.num_continuous_actions <= PURCHASE_VALUE_ACTION_INDEX:
            raise ValueError("purchase value requires continuous slot 5")
        if (
            self.model_config.output_vocab_size or 0
        ) <= recsys_pb2.ActionName.ADS_PURCHASE_CONVERSION:
            raise ValueError("purchase value requires purchase head 200")
        if (
            not math.isfinite(self.purchase_value_huber_delta)
            or self.purchase_value_huber_delta <= 0
        ):
            raise ValueError("purchase value Huber delta must be finite and positive")
        if any(
            lc.action_index == PURCHASE_VALUE_ACTION_INDEX for lc in self.continuous_action_losses
        ):
            raise ValueError("purchase value slot cannot also use a generic continuous loss")
        if not self.purchase_value_smoothing_windows or any(
            w <= 0 for w in self.purchase_value_smoothing_windows
        ):
            raise ValueError("purchase_value_smoothing_windows must be positive sample counts")
        if self.purchase_value_head not in ("regression", "binned"):
            raise ValueError("purchase_value_head must be 'regression' or 'binned'")
        if self.purchase_value_head == "binned":
            self.purchase_value_bins.validate("purchase_value")

    def conversion_delay_slices(self) -> list[ConversionDelaySlice]:
        if not self.conversion_delay_slice_thresholds_s or not self.ads_head_masking:
            return []
        heads = engagement_to_ids(self.metric_group)
        return [
            ConversionDelaySlice(
                head=head,
                name=f"{CONVERSION_DELAY_BASE_MASK}_{side}{threshold_s}s",
                column_index=k,
                threshold_ms=threshold_s * 1000,
                slow=side == "ge",
            )
            for k, head in enumerate(CONVERSION_DELAY_HEAD_COLUMNS)
            if head in heads
            for threshold_s in self.conversion_delay_slice_thresholds_s
            for side in ("ge", "lt")
        ]

    def validate_ads_head_masking(self) -> None:
        if self.ads_head_masking and self.use_seqpack:
            raise ValueError(
                "ads_head_masking is not supported with use_seqpack: the per-candidate, "
                "per-head trained_candidate_mask has no packed-layout equivalent"
            )
        if self.ads_pixel_head_masking and not self.ads_head_masking:
            raise ValueError("ads_pixel_head_masking requires ads_head_masking")
        if self.ads_late_window_heads and not self.ads_head_masking:
            raise ValueError("ads_late_window_heads requires ads_head_masking")
        if self.serve_late_window_composed and not self.ads_late_window_heads:
            raise ValueError("serve_late_window_composed requires ads_late_window_heads")
        if self.ads_pixel_head_masking and self.use_seqpack:
            raise ValueError("ads_pixel_head_masking is not supported with use_seqpack")
        if self.metric_own_slice and not self.ads_head_masking:
            raise ValueError("metric_own_slice requires ads_head_masking")

    @property
    def has_side_inputs(self) -> bool:
        return bool(self.ads_counts_emb_dim or self.wallclock_recency)

    def validate_side_inputs(self) -> None:
        if self.has_side_inputs and self.feature_prep_enabled:
            raise ValueError("ads_counts_emb_dim / wallclock_recency need feature_prep off")

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

    attend_padded_history: bool = False

    pad_candidates_to_128: bool = True

    use_seqpack: bool = False

    ads_counts_emb_dim: int = 0
    wallclock_recency: bool = False

    effective_sequence_len: int | None = None

    transformer_output_only: bool = False
    use_dense_action_table: bool = False
    use_product_surface: bool = False

    post_age_granularity_mins: int = 60

    post_age_max_mins: int = POST_AGE_MAX_MINUTES

    post_age_bucket_strategy: Literal["linear", "log1p"] = "linear"

    post_age_num_buckets: int = 80

    mask_candidate_positive_when_negative_action_present: bool = False

    metric_group: str = "default"

    metric_mask_keys: list[str] | None = None
    metric_own_slice: bool = False
    metrics_ndcg_shared_sorts: bool = False

    conversion_delay_slice_thresholds_s: tuple[int, ...] = ()

    multimodal_embedding_type: EmbeddingType | None = None

    history_url_embedding_dim: int = 0

    search_query_embedding_dim: int = 0

    use_post_sid: bool = False
    use_post_embedding: bool = True
    sid_embed_dim: int = 1024
    sid_num_levels: int = 6
    sid_codebook_size: int = 1024
    sid_hash_level: bool = False
    sid_cross_attn: bool = False
    sid_prefix2_embed_dim: int = 0
    sid_prefix3_embed_dim: int = 0
    sid_prefix3_rows: int = 262144

    sid_embedding_mode: Literal["learned", "recon"] = "learned"
    sid_decoder_path: str = ""

    use_user_embedding: bool = True

    use_ip_address: bool = False

    feature_prep_enabled: bool = False
    feature_prep: FeaturePrepConfig = FeaturePrepConfig()

    concat_history_bridge_prob: bool = False
    concat_history_web_conv_time_on_site: bool = False
    web_conv_time_on_site_norm_scale: float = 1800.0
    add_history_web_conv_value: bool = False
    web_conv_value_usd_cap: float = 10_000.0
    web_conv_value_ratio_cap: float = 50.0

    mask_neg_feedback_on_negatives: bool = True

    ads_head_masking: bool = False

    ads_pixel_head_masking: bool = False

    ads_late_window_heads: bool = False
    serve_late_window_composed: bool = False

    mact_in_app_loss_weight: float = 1.0

    view_loss_weight: float = 1.0

    condition_search_relevance_on_prompt: bool = False

    enable_platform_metrics: bool = False

    dpa_product_embed_dim: int = 32

    dpa_product_table_size: int = ADS_PRODUCT_KEY_TABLE_SIZE

    enable_dpa_input_embedding: bool = False

    user_features: UserFeaturesConfig = UserFeaturesConfig()

    context_features: ContextFeaturesConfig = ContextFeaturesConfig()

    unified_context_dwell_time_dim: int = 64

    safety_filter_mode: Literal["off", "hard", "soft"] = "off"
    safety_filter_bits: int = 0b11
    safety_filter_soft_weight: float = 0.0
    safety_filter_apply_to_candidates: bool = True
    safety_filter_apply_to_history: bool = False

    page_rl: PageRlConfig = PageRlConfig()

    @property
    def multimodal_embedding_dim(self) -> int:
        if self.multimodal_embedding_type is None:
            return 0
        return EMBEDDING_CONFIG[self.multimodal_embedding_type][1]

    def make(self, sharding_context: ShardingContext) -> RecsysAggregatedModel:
        self.validate_purchase_value()
        self.validate_aux_value_heads()
        self.validate_ads_head_masking()
        self.validate_side_inputs()
        if (
            not (self.pad_candidates_to_128 or self.use_seqpack)
            and self.model_config.attn_config.attn_impl != "cutedsl_ranker_attn"
        ):
            raise ValueError("pad_candidates_to_128=False requires attn_impl=cutedsl_ranker_attn")
        if self.history_url_embedding_dim < 0 or (
            self.history_url_embedding_dim > 0 and self.feature_prep_enabled
        ):
            raise ValueError(
                f"history_url_embedding_dim={self.history_url_embedding_dim}: must be >= 0 "
                "and is only wired on the legacy (feature_prep_enabled=False) path"
            )

        if self.model_config.num_candidate_only_layers:
            if self.transformer_output_only:
                raise ValueError(
                    "transformer_output_only reads every final hidden state, but candidate-only "
                    "layers update only the candidates."
                )
            if (
                self.page_rl.enabled
                and not self.model_config.candidate_only_layers_update_user_token
            ):
                raise ValueError(
                    "page_rl reads each user's first prefix token; set "
                    "candidate_only_layers_update_user_token=True to update it in candidate-only "
                    "layers."
                )
            if (
                self.model_config.candidate_only_layers_update_user_token
                and self.model_config.num_candidate_only_layers > 1
            ):
                raise ValueError(
                    "candidate_only_layers_update_user_token supports one candidate-only layer."
                )

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

        if self.sid_embedding_mode == "recon":
            assert not self.sid_hash_level and not self.sid_cross_attn, (
                "sid_embedding_mode='recon' replaces the learned SID tables outright; "
                "sid_hash_level / sid_cross_attn only apply to mode='learned'."
            )
            assert self.sid_decoder_path, (
                "sid_embedding_mode='recon' requires sid_decoder_path "
                "(codebook.safetensors with `stages` + `dec_*` tensors)."
            )
        if self.use_seqpack:
            attn_config = self.model_config.attn_config
            assert (
                isinstance(attn_config, RecsysAttentionConfig)
                and attn_config.num_user_prefix_tokens == self.num_user_prefix_tokens
            )

        validate_continuous_action_losses(self.continuous_action_losses)

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

    def compute_forward_tflops(self, num_seq_per_sec: float) -> float:
        S = (
            self.effective_sequence_len
            if self.effective_sequence_len is not None
            else self.sequence_len
        )
        attn, ffn = self.model_config.compute_forward_flops_per_sec(num_seq_per_sec, S)
        return round(attn + ffn, 2)

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
    post_hashes: jax.Array, log_q_num_bins: int, negative_sample_mask: jnp.ndarray
) -> jax.Array:
    hashed_item_ids_flat = post_hashes[:, :, 0] % log_q_num_bins

    hashed_item_ids_masked_flat = hashed_item_ids_flat * negative_sample_mask

    sorted_ids = jnp.sort(hashed_item_ids_masked_flat.ravel())
    tweet_counts_flat = jnp.searchsorted(
        sorted_ids, hashed_item_ids_masked_flat, side="right", method="scan_unrolled"
    ) - jnp.searchsorted(
        sorted_ids, hashed_item_ids_masked_flat, side="left", method="scan_unrolled"
    )
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
    pred_mask: jnp.ndarray,
    num_out_of_range: jnp.ndarray | None,
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
    if num_out_of_range is not None:
        stats[f"{name}-num-out-of-range"] = num_out_of_range

    stats[f"{name}-gt-min"] = jnp.where(
        n > 0, jnp.min(jnp.where(loss_mask, gt_clamped, jnp.inf)), 0.0
    )
    stats[f"{name}-gt-mean"] = jnp.where(n > 0, gt_mean, 0.0)
    stats[f"{name}-gt-max"] = jnp.where(
        n > 0, jnp.max(jnp.where(loss_mask, gt_clamped, -jnp.inf)), 0.0
    )
    pred_mask_f32 = pred_mask.astype(jnp.float32)
    n_pred = jnp.sum(pred_mask_f32)
    pred_mean = jnp.sum(pred_f32 * pred_mask_f32) / jnp.maximum(n_pred, 1.0)
    stats[f"{name}-pred-min"] = jnp.where(
        n_pred > 0, jnp.min(jnp.where(pred_mask, pred_in_original_units, jnp.inf)), 0.0
    )
    stats[f"{name}-pred-mean"] = jnp.where(n_pred > 0, pred_mean, 0.0)
    stats[f"{name}-pred-max"] = jnp.where(
        n_pred > 0, jnp.max(jnp.where(pred_mask, pred_in_original_units, -jnp.inf)), 0.0
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
    sid_prefix2_embed_dim: int = 0,
    sid_prefix3_embed_dim: int = 0,
    sid_prefix3_rows: int = 262144,
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

    if sid_prefix2_embed_dim > 0:
        prefix2_embs = sid_prefix2_lookup(
            sids,
            sid_codebook_size,
            sid_prefix2_embed_dim,
            lr_multiplier_func,
            embed_init_scale,
            name_prefix,
        )
        prefix2_proj = get_parameter(
            f"{name_prefix}_sid_prefix2_proj",
            [sid_prefix2_embed_dim, sid_embed_dim],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(sid_prefix2_embed_dim),
        )
        summed = summed + jnp.dot(prefix2_embs, prefix2_proj).astype(summed.dtype)

    if sid_prefix3_embed_dim > 0:
        prefix3_embs = sid_prefix3_lookup(
            sids,
            sid_codebook_size,
            sid_prefix3_embed_dim,
            sid_prefix3_rows,
            lr_multiplier_func,
            embed_init_scale,
            name_prefix,
        )
        prefix3_proj = get_parameter(
            f"{name_prefix}_sid_prefix3_proj",
            [sid_prefix3_embed_dim, sid_embed_dim],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(sid_prefix3_embed_dim),
        )
        summed = summed + jnp.dot(prefix3_embs, prefix3_proj).astype(summed.dtype)

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


def normalize_web_conv_time_on_site(millis: jax.Array, scale_secs: float) -> jax.Array:
    secs = jnp.clip(millis.astype(jnp.float32) / 1000.0, 0.0, scale_secs)
    return jnp.log1p(secs) / jnp.log1p(jnp.float32(scale_secs))


WEB_CONV_VALUE_USD_IDX = 6
WEB_CONV_VALUE_RATIO_IDX = 7


def normalize_web_conv_value(values: jax.Array, usd_cap: float, ratio_cap: float) -> jax.Array:
    caps = jnp.array([usd_cap, ratio_cap], dtype=jnp.float32)
    x = jnp.clip(values.astype(jnp.float32), 0.0, caps)
    return jnp.log1p(x) / jnp.log1p(caps)


PURCHASE_VALUE_HISTORY_FEATURES = 9
PURCHASE_VALUE_HISTORY_RATIO_FEATURES = 7


def purchase_value_history_features(
    history_continuous_actions: jax.Array,
    history_auth_hashes: jax.Array,
    history_impr_ts: jax.Array,
    candidate_auth_hashes: jax.Array,
    usd_cap: float,
    ratio_cap: float | None = None,
) -> jax.Array:
    usd = history_continuous_actions[..., WEB_CONV_VALUE_USD_IDX].astype(jnp.float32)
    valid = jnp.isfinite(usd) & (usd > 0)
    v = jnp.where(valid, jnp.log1p(jnp.clip(usd, 0.0, usd_cap)) / math.log1p(usd_cap), 0.0)
    num_rows = history_continuous_actions.shape[1]
    log_count = lambda n: jnp.log1p(n.astype(jnp.float32)) / math.log1p(num_rows)
    same = (
        jnp.all(
            history_auth_hashes[:, None, :, :] == candidate_auth_hashes[:, :, None, :],
            axis=-1,
        )
        & valid[:, None, :]
    )
    v_same = jnp.where(same, v[:, None, :], 0.0)
    n_same = same.sum(-1)
    ts = history_impr_ts.astype(jnp.int32)
    ts_rel = (ts - ts.max(axis=-1, keepdims=True)).astype(jnp.float32)
    last_idx = jnp.argmax(jnp.where(same, ts_rel[:, None, :], -jnp.inf), axis=-1)
    last_v = jnp.take_along_axis(v_same, last_idx[..., None], axis=-1)[..., 0]
    n_user = valid.sum(-1)
    user = jnp.stack(
        [
            (n_user > 0).astype(jnp.float32),
            log_count(n_user),
            v.max(-1),
            v.sum(-1) / jnp.maximum(n_user, 1),
        ],
        axis=-1,
    )
    per_candidate = jnp.stack(
        [
            (n_same > 0).astype(jnp.float32),
            log_count(n_same),
            v_same.max(-1),
            v_same.sum(-1) / jnp.maximum(n_same, 1),
            jnp.where(n_same > 0, last_v, 0.0),
        ],
        axis=-1,
    )
    user = jnp.broadcast_to(user[:, None, :], per_candidate.shape[:2] + (user.shape[-1],))
    if ratio_cap is None:
        return jnp.concatenate([per_candidate, user], axis=-1)

    ratio = history_continuous_actions[..., WEB_CONV_VALUE_RATIO_IDX].astype(jnp.float32)
    valid_r = jnp.isfinite(ratio) & (ratio > 0)
    lr = jnp.where(
        valid_r,
        jnp.log(jnp.clip(ratio, 1.0 / ratio_cap, ratio_cap)) / math.log(ratio_cap),
        0.0,
    )
    same_r = same & valid_r[:, None, :]
    n_same_r = same_r.sum(-1)
    mean_same_r = jnp.einsum("bch,bh->bc", same_r.astype(jnp.float32), lr) / jnp.maximum(
        n_same_r, 1
    )
    last_same_idx = jnp.argmax(jnp.where(same_r, ts_rel[:, None, :], -jnp.inf), axis=-1)
    last_same_r = jnp.take_along_axis(lr, last_same_idx, axis=-1)
    n_user_r = valid_r.sum(-1)
    last_user_idx = jnp.argmax(jnp.where(valid_r, ts_rel, -jnp.inf), axis=-1)
    has_user_r = n_user_r > 0
    ratio_user = jnp.stack(
        [
            has_user_r.astype(jnp.float32),
            lr.sum(-1) / jnp.maximum(n_user_r, 1),
            jnp.where(has_user_r, jnp.take_along_axis(lr, last_user_idx[:, None], -1)[:, 0], 0.0),
            jnp.where(has_user_r, jnp.where(valid_r, lr, -1.0).max(-1), 0.0),
        ],
        axis=-1,
    )
    ratio_candidate = jnp.stack(
        [
            (n_same_r > 0).astype(jnp.float32),
            mean_same_r,
            jnp.where(n_same_r > 0, last_same_r, 0.0),
        ],
        axis=-1,
    )
    ratio_user = jnp.broadcast_to(
        ratio_user[:, None, :], per_candidate.shape[:2] + (ratio_user.shape[-1],)
    )
    return jnp.concatenate([per_candidate, user, ratio_candidate, ratio_user], axis=-1)


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
    history_web_conv_time_on_site: jnp.ndarray | None = None,
    history_url_embeddings: jnp.ndarray | None = None,
    history_web_conv_value: jnp.ndarray | None = None,
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
    if history_web_conv_time_on_site is not None:
        t = jnp.clip(history_web_conv_time_on_site.astype(post_author_embedding.dtype), 0.0, 1.0)
        post_author_embedding = jnp.concatenate([post_author_embedding, t], axis=-1)

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
    if history_url_embeddings is not None:
        url_dim = history_url_embeddings.shape[-1]
        hist_url_proj_mat = get_parameter(
            "hist_url_proj_mat",
            [url_dim, D],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(url_dim),
        )
        history_post_author_embedding = history_post_author_embedding + jnp.dot(
            history_url_embeddings.astype(hist_url_proj_mat.dtype), hist_url_proj_mat
        ).astype(history_post_author_embedding.dtype)
    if history_web_conv_value is not None:
        value_dirs = get_parameter(
            "hist_web_conv_value_dirs",
            [2, D],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(None, None),
            lr_multiplier=lr_multiplier_func(2),
        )
        history_post_author_embedding = history_post_author_embedding + jnp.dot(
            history_web_conv_value.astype(value_dirs.dtype), value_dirs
        ).astype(history_post_author_embedding.dtype)

    post_action_embedding_seq = history_post_author_embedding.reshape(B, -1, D)

    if history_continuous_actions is not None and continuous_action_losses is not None:
        seen_action_indices: set[int] = set()
        for loss_config in continuous_action_losses:
            if loss_config.action_index == PURCHASE_VALUE_ACTION_INDEX:
                continue
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
    dpa_product_key: jax.Array | None = None,
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

    if dpa_product_key is not None:
        pad_dpa_product_key = jnp.zeros(
            (batch_size, pad_length) + dpa_product_key.shape[2:], dtype=dpa_product_key.dtype
        )
        padded_dpa_product_key = jnp.concatenate([dpa_product_key, pad_dpa_product_key], axis=1)
    else:
        padded_dpa_product_key = None

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
        padded_dpa_product_key,
    )


def cast_jax(arr: npt.NDArray) -> jax.Array:
    return typing.cast(jax.Array, arr)


def attention_segment_ids(
    padding_mask: jax.Array, candidate_start_offset: int, attend_padded_history: bool
) -> jax.Array:
    idx = jnp.arange(padding_mask.shape[1], dtype=jnp.int32)[None, :]
    is_candidate = idx >= candidate_start_offset
    valid = padding_mask
    if attend_padded_history:
        valid = jnp.logical_or(padding_mask, jnp.logical_not(is_candidate))
    return jnp.broadcast_to(
        jnp.where(valid, jnp.where(is_candidate, -1, 1), 0),
        padding_mask.shape,
    ).astype(jnp.int32)


def untrained_candidates_excluded(
    padding_mask: jax.Array,
    trained_heads: jax.Array | None,
    *,
    candidate_positions: jax.Array | None = None,
    candidate_start_offset: int = 0,
) -> jax.Array | None:
    if trained_heads is None:
        return None
    untrained = (~trained_heads.any(-1)).astype(jnp.int8)
    if candidate_positions is None:
        drop = jnp.pad(untrained, ((0, 0), (candidate_start_offset, 0)))
    else:
        rows = jnp.arange(padding_mask.shape[0])[:, None]
        drop = jnp.zeros(padding_mask.shape, jnp.int8).at[rows, candidate_positions].max(untrained)
    return jnp.where(drop > 0, jnp.zeros_like(padding_mask), padding_mask)


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
    dpa_product_key: jax.Array | None = None,
    sample_source: jax.Array | None = None,
    trained_candidate_mask: jax.Array | None = None,
    *,
    ads_head_masking: bool = False,
    condition_search_relevance_on_prompt: bool = False,
    enable_platform_metrics: bool = False,
    metric_mask_keys: list[str] | None = None,
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
        (line_item_objective == LineItemObjective.WEBSITE_CLICKS).astype(mask.dtype)
        if line_item_objective is not None
        else jnp.zeros_like(mask)
    )
    masks["promoted_home_website_clicks"] = (
        mask * promoted_mask * home_timeline_mask * website_clicks_objective
    )
    masks["non_negative_promoted_home_website_clicks"] = (
        non_negative_mask * promoted_mask * home_timeline_mask * website_clicks_objective
    )

    dpa_mask = (
        (dpa_product_key != 0).astype(mask.dtype)
        if dpa_product_key is not None
        else jnp.zeros_like(mask)
    )
    masks["dpa"] = mask * dpa_mask
    masks["non_negative_dpa"] = non_negative_mask * dpa_mask

    if ads_head_masking:
        web_click_mask = raw_targets[:, :, CLICK_ACTION_INDEX].astype(mask.dtype)
        app_click_mask = raw_targets[:, :, MMP_CLICK_ACTION_INDEX].astype(mask.dtype)
        masks["website_clicked"] = mask * web_click_mask
        masks["non_negative_website_clicked"] = non_negative_mask * web_click_mask
        masks["app_clicked"] = mask * app_click_mask
        masks["non_negative_app_clicked"] = non_negative_mask * app_click_mask
        assert trained_candidate_mask is not None, "ads_head_masking needs trained_candidate_mask"
        if sample_source is None:
            fresh = jnp.ones_like(mask)
        else:
            fresh = (sample_source == FRESH_STREAM_ID).astype(mask.dtype)
        masks["fresh"] = mask * fresh
        trained = trained_candidate_mask.astype(mask.dtype)
        for slice_name, head in ADS_FAMILY_SLICE_HEADS.items():
            masks[slice_name] = mask * trained[:, :, head]
        masks["late_split_no_early"] = mask * jnp.prod(
            trained[:, :, jnp.array(ADS_LATE_WINDOW_TWIN_HEAD_INDICES)], axis=-1
        )
        masks["late_split_non_clicked_no_early"] = mask * jnp.prod(
            trained[:, :, jnp.array(ADS_LATE_WINDOW_VT_TWIN_HEAD_INDICES)], axis=-1
        )
        masks["fresh_home_website_clicks"] = (
            masks["fresh"] * home_timeline_mask * website_clicks_objective
        )

    if condition_search_relevance_on_prompt:
        prompt_mask = jnp.any(
            raw_targets[:, :, jnp.array(SEARCH_RELEVANCE_ACTION_INDICES)] == 1, axis=-1
        ).astype(mask.dtype)
        masks["prompted"] = mask * prompt_mask
        masks["non_negative_prompted"] = non_negative_mask * prompt_mask

    if enable_platform_metrics and ios_mask is not None and android_mask is not None:
        masks["ios"] = ios_mask
        masks["android"] = android_mask
        masks["ios_non_negative"] = ios_mask * (1 - negative_sample_mask)
        masks["android_non_negative"] = android_mask * (1 - negative_sample_mask)

    if metric_mask_keys:
        keep = set(metric_mask_keys)
        unknown = keep - masks.keys()
        if unknown:
            raise ValueError(
                f"metric_mask_keys entries {sorted(unknown)} not in available masks {sorted(masks)}"
            )
        masks = {k: v for k, v in masks.items() if k in keep}

    return masks


class CandidateInputs(NamedTuple):
    targets: jax.Array
    target_padding_mask: jax.Array
    negative_sample_mask: jax.Array
    product_surface: jax.Array
    promoted_ids: jax.Array | None
    continuous_actions: jax.Array | None
    client_app_id: jax.Array | None
    line_item_objective: jax.Array | None
    safety_label_mask: jax.Array | None
    dpa_product_key: jax.Array | None
    source_id: jax.Array | None
    trained_heads: jax.Array | None
    eligible_heads: jax.Array | None
    loss_padding_mask: jax.Array
    loss_weights: jax.Array | None
    continuous_heads: dict[int, tuple[jax.Array, jax.Array | None]]
    value_heads: dict[str, tuple[jax.Array, jax.Array, jax.Array]]


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
        conversion_delay_base: jax.Array | None = None,
        conversion_delay_ms: jax.Array | None = None,
        head_eligible_mask: jax.Array | None = None,
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

        per_head_eligible: dict[str, jax.Array] | None = None
        per_head_mask_values: jax.Array | None = None
        if head_eligible_mask is not None:
            eng_to_ids = engagement_to_ids(self.config.metric_group)
            eligible = head_eligible_mask.astype(mask_values[0].dtype)
            per_head_eligible = {
                eng_name: jnp.max(eligible[..., jnp.array(eng_to_ids[eng_name])], axis=-1)
                for eng_name in eng_names
            }
            per_head_mask_values = (
                jnp.stack(mask_values)[None]
                * jnp.stack([per_head_eligible[eng_name] for eng_name in eng_names])[:, None]
            )

        head_mask_keys = metric_head_mask_keys(
            engagement_to_ids(self.config.metric_group),
            mask_keys,
            own_slice=self.config.metric_own_slice,
        )
        groups: dict[tuple[str, ...], list[int]] = {}
        for eng_idx, eng_name in enumerate(eng_names):
            groups.setdefault(tuple(head_mask_keys[eng_name]), []).append(eng_idx)
        for group_keys, eng_idxs in groups.items():
            if not group_keys:
                continue
            full_grid = list(group_keys) == mask_keys and len(eng_idxs) == len(eng_names)
            if full_grid:
                group_mask_values, group_per_head = mask_values, per_head_mask_values
            else:
                mask_idxs = [mask_keys.index(k) for k in group_keys]
                group_mask_values = [mask_values[j] for j in mask_idxs]
                group_per_head = (
                    None
                    if per_head_mask_values is None
                    else per_head_mask_values[jnp.array(eng_idxs)][:, jnp.array(mask_idxs)]
                )
            metrics_calc(
                [p_list[i] for i in eng_idxs],
                [y_list[i] for i in eng_idxs],
                [eng_names[i] for i in eng_idxs],
                list(group_keys),
                group_mask_values,
                stats,
                per_head_mask_values=group_per_head,
                ndcg_shared_sorts=self.config.metrics_ndcg_shared_sorts,
            )

        metrics_calc_global_num_tokens(masks, stats)

        for mask_key, mval in zip(mask_keys, mask_values):
            stats[f"{mask_key}_effective_num_tokens"] = mval.sum()

        cells: list[tuple[str, str, jax.Array, jax.Array]] = []
        for eng_idx, eng_name in enumerate(eng_names):
            for mask_key in head_mask_keys[eng_name]:
                j = mask_keys.index(mask_key)
                if per_head_mask_values is None:
                    cells.append(
                        (
                            eng_name,
                            mask_key,
                            mask_values[j],
                            stats[f"{mask_key}_effective_num_tokens"],
                        )
                    )
                else:
                    mask_val = per_head_mask_values[eng_idx][j]
                    cells.append((eng_name, mask_key, mask_val, mask_val.sum()))
        slices = self.config.conversion_delay_slices()
        if slices and conversion_delay_base is not None and conversion_delay_ms is not None:
            slice_masks = conversion_delay_slice_metrics(
                slices,
                prob_labels,
                conversion_delay_base,
                conversion_delay_ms,
                raw_weights,
                stats,
                per_head_eligible=per_head_eligible,
            )
            cells.extend(
                (head, name, m, stats[f"{head}_{name}_effective_num_tokens"])
                for (head, name), m in slice_masks.items()
            )

        if rce_ema is not None and rce_alpha is not None and smoothing_windows is not None:
            a = rce_alpha[:, None]
            new_rce_ema: dict[str, jax.Array] = {}
            for eng_name, mask_key, _, total_count in cells:
                base_key = f"{eng_name}/{mask_key}"
                ce_sum = stats[f"{eng_name}_{mask_key}_loss"] * total_count
                pos_sum = stats[f"{eng_name}_{mask_key}_num_tokens"]
                updated = _ema_step(
                    rce_ema,
                    base_key,
                    jnp.stack([ce_sum, pos_sum, total_count]),
                    a,
                    smoothing_windows,
                )
                s_rce = _smoothed_rce(updated)
                s_ne = ne_from_rce(s_rce)
                for i, ws in enumerate(smoothing_windows):
                    new_rce_ema[f"{base_key}/{ws}"] = updated[i]
                    stats[f"{eng_name}_{mask_key}_smoothed_RCE_{ws}"] = s_rce[i]
                    stats[f"{eng_name}_{mask_key}_smoothed_NE_{ws}"] = s_ne[i]
            stats["_rce_ema"] = new_rce_ema

        if calib_ema is not None and rce_alpha is not None and smoothing_windows is not None:
            a = rce_alpha[:, None]
            new_calib_ema: dict[str, jax.Array] = {}
            for eng_name, mask_key, mask_val, _ in cells:
                base_key = f"{eng_name}/{mask_key}"
                p, y = prob_labels[eng_name]
                batch_stat = jnp.stack([jnp.sum(p * mask_val), jnp.sum(y * mask_val)])
                updated = _ema_step(calib_ema, base_key, batch_stat, a, smoothing_windows)
                s_calib = _smoothed_calib(updated)
                for i, ws in enumerate(smoothing_windows):
                    new_calib_ema[f"{base_key}/{ws}"] = updated[i]
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
                stats[f"{eng_name}_{bucket_name}_NE"] = ne_from_rce(
                    stats[f"{eng_name}_{bucket_name}_RCE"]
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
        dpa_product_key: jax.Array | None = None,
        sample_source: jax.Array | None = None,
        trained_candidate_mask: jax.Array | None = None,
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
            dpa_product_key,
            sample_source,
            trained_candidate_mask,
            ads_head_masking=self.config.ads_head_masking,
            condition_search_relevance_on_prompt=self.config.condition_search_relevance_on_prompt,
            enable_platform_metrics=self.config.enable_platform_metrics,
            metric_mask_keys=self.config.metric_mask_keys,
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
        dpa_product_key: jax.Array | None = None,
        sample_source: jax.Array | None = None,
        trained_candidate_mask: jax.Array | None = None,
        stats: dict | None = None,
        rce_ema: dict[str, jax.Array] | None = None,
        rce_alpha: jax.Array | None = None,
        smoothing_windows: tuple[int, ...] | None = None,
        calib_ema: dict[str, jax.Array] | None = None,
        raw_weights: jax.Array | None = None,
        conversion_delay_ms: jax.Array | None = None,
        head_eligible_mask: jax.Array | None = None,
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
            dpa_product_key,
            sample_source,
            trained_candidate_mask,
        )
        conversion_delay_base = (
            conversion_delay_base_mask(mask, trained_candidate_mask)
            if self.config.conversion_delay_slices() and trained_candidate_mask is not None
            else None
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
            conversion_delay_base=conversion_delay_base,
            conversion_delay_ms=conversion_delay_ms,
            head_eligible_mask=head_eligible_mask,
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
        fp_on = (
            self.config.feature_prep_enabled and self.config.feature_prep.enable_engagement_counts
        )
        ctx_on = (
            self.config.context_features.enabled
            and self.config.context_features.enable_engagement_counts
        )
        stale_on = self.config.feature_prep_enabled and self.config.feature_prep.enable_stale_post
        if batch is None or not (fp_on or ctx_on or stale_on):
            return stats

        def _add_scalar_stats(
            name: str,
            side: str,
            raw_counts: jax.Array,
            valid_float: jax.Array,
        ) -> None:
            raw_non_negative = jnp.maximum(raw_counts.astype(jnp.float32), 0.0)
            log2_counts = jnp.log2(raw_non_negative + 1.0)
            n_valid = jnp.maximum(jnp.sum(valid_float), 1.0)
            positive = (raw_non_negative > 0.0).astype(jnp.float32)

            stats[f"ec/{name}/{side}/coverage"] = jnp.sum(positive * valid_float) / n_valid
            stats[f"ec/{name}/{side}/mean_raw"] = jnp.sum(raw_non_negative * valid_float) / n_valid
            stats[f"ec/{name}/{side}/mean_log2"] = jnp.sum(log2_counts * valid_float) / n_valid

        def _add_bucket_percentiles(
            name: str,
            side: str,
            bucket_values: jax.Array,
            valid_float: jax.Array,
            nbins: int,
        ) -> None:
            n_valid = jnp.maximum(jnp.sum(valid_float), 1.0)
            binned = jnp.clip(bucket_values.astype(jnp.int32), 0, nbins - 1)
            hist = (
                jnp.zeros((nbins,), jnp.float32)
                .at[jnp.reshape(binned, (-1,))]
                .add(jnp.reshape(valid_float, (-1,)))
            ) / n_valid
            cdf = jnp.cumsum(hist)
            for q_name, q in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
                stats[f"ec/{name}/{side}/{q_name}_bucket"] = jnp.argmax(cdf >= q).astype(
                    jnp.float32
                )

        for seq_name, side in (("history_seq", "history"), ("candidate_seq", "candidate")):
            seq = batch[seq_name]
            raw_i64 = seq.get("int64_features")
            post_hashes = seq.get("post_hashes")
            if raw_i64 is None or post_hashes is None:
                continue

            i64 = cast_jax(raw_i64)
            valid = (cast_jax(post_hashes)[..., 0] > 0).astype(jnp.float32)
            n_valid = jnp.maximum(jnp.sum(valid), 1.0)
            raw_by_name: dict[str, jax.Array] = {}

            for cat_feat_enum, int64_feat_enum in ENGAGEMENT_COUNT_BUCKET_MAP:
                name = cat_feat_enum.name.removesuffix("CountBucketSeq").lower()
                max_bucket = ENGAGEMENT_COUNT_MAX_BUCKET.get(
                    cat_feat_enum, ENGAGEMENT_COUNT_NUM_BUCKETS - 1
                )
                raw_counts = i64[:, :, int64_feat_enum.value]
                raw_by_name[name] = jnp.maximum(raw_counts.astype(jnp.float32), 0.0)
                buckets = compute_engagement_count_bucket(raw_counts, max_bucket)

                _add_scalar_stats(name, side, raw_counts, valid)
                stats[f"ec/{name}/{side}/mean_bucket"] = (
                    jnp.sum(buckets.astype(jnp.float32) * valid) / n_valid
                )
                _add_bucket_percentiles(name, side, buckets, valid, max_bucket + 1)

            non_view_engagement = (
                raw_by_name["fav"]
                + raw_by_name["reply"]
                + raw_by_name["repost"]
                + raw_by_name["quote"]
            )
            view = raw_by_name["view"]
            _add_scalar_stats("engagement_no_view", side, non_view_engagement, valid)
            _add_bucket_percentiles(
                "engagement_no_view",
                side,
                compute_engagement_count_bucket(
                    non_view_engagement, ENGAGEMENT_COUNT_NUM_BUCKETS - 1
                ),
                valid,
                ENGAGEMENT_COUNT_NUM_BUCKETS,
            )

            view_missing_like = ((view <= 0.0) & (non_view_engagement > 0.0)).astype(jnp.float32)
            stats[f"ec/view/{side}/missing_like_rate"] = (
                jnp.sum(view_missing_like * valid) / n_valid
            )

            bools = seq.get("bool_features")
            if (
                side == "candidate"
                and stale_on
                and bools is not None
                and bools.shape[-1] > BoolFeature.isStalePost14d.value
            ):
                is_stale = cast_jax(bools)[:, :, BoolFeature.isStalePost14d.value].astype(
                    jnp.float32
                )
                stats[f"ec/stale_post_14d/{side}/zeroed_frac"] = jnp.sum(is_stale * valid) / n_valid

        return stats

    def compute_search_lexical_match_metrics(
        self,
        batch: RecsysFeaturesBatch,
        stats: dict | None = None,
    ) -> dict:
        if stats is None:
            stats = {}
        if not self.config.context_features.enable_search_lexical_match:
            return stats

        seq = batch["candidate_seq"]
        cat = seq.get("categorical_features")
        floats = seq.get("float_features")
        hashes = seq.get("post_hashes")
        exact_idx = CategoricalFeature.exactPhraseSeq.value
        frac_idx = FloatFeature.matchedWordFractionSeq.value
        if cat is None or floats is None or hashes is None:
            return stats
        if exact_idx >= cat.shape[-1] or frac_idx >= floats.shape[-1]:
            return stats

        exact_phrase = cast_jax(cat)[:, :, exact_idx]
        fraction = cast_jax(floats)[:, :, frac_idx].astype(jnp.float32)
        non_pad = (cast_jax(hashes)[..., 0] != 0).astype(jnp.float32)
        is_set = (exact_phrase != 0).astype(jnp.float32) * non_pad
        total = jnp.sum(non_pad)
        n_set = jnp.sum(is_set)
        stats["search_lexical_match/frac_set_candidate"] = jnp.where(total > 0, n_set / total, 0.0)
        stats["search_lexical_match/frac_exact_phrase_candidate"] = jnp.where(
            n_set > 0, jnp.sum((exact_phrase == 2).astype(jnp.float32) * is_set) / n_set, 0.0
        )
        stats["search_lexical_match/mean_matched_word_fraction_candidate"] = jnp.where(
            n_set > 0, jnp.sum(fraction * is_set) / n_set, 0.0
        )
        buckets = compute_matched_word_fraction_bucket(fraction, exact_phrase != 0)
        stats["search_lexical_match/mean_bucket_candidate"] = jnp.where(
            n_set > 0, jnp.sum(buckets.astype(jnp.float32) * is_set) / n_set, 0.0
        )
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
        dense: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
        if dense:
            embedding_table = get_parameter(
                name,
                shape=[output_vocab_size, emb_size],
                init=embed_init,
                dtype=jnp.float32,
                pspec=P(),
                rms_clip_axes=(-2, -1),
            )
            output = jnp.dot(input.astype(embedding_table.dtype), embedding_table)
            output = output.astype(DTYPE_BY_NAME[self.config.fprop_dtype])
            return output, embedding_table
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
            if cat_feat.feature_name == "local_day_of_week":
                if not ctx_config.enable_day_of_week:
                    continue
            if cat_feat.feature_name in ("exact_phrase", "matched_word_fraction_bucket"):
                if not (ctx_config.enable_search_lexical_match and is_candidate):
                    continue
            if cat_feat.feature_name == "web_conv_tracking_integration":
                if not ctx_config.enable_web_conv_tracking_integration or is_candidate:
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

    @staticmethod
    def set_matched_word_fraction_bucket(
        cat_features: jax.Array, float_features: jax.Array | None
    ) -> jax.Array:
        bucket_idx = CategoricalFeature.matchedWordFractionBucketSeq.value
        exact_idx = CategoricalFeature.exactPhraseSeq.value
        frac_idx = FloatFeature.matchedWordFractionSeq.value
        if (
            float_features is None
            or bucket_idx >= cat_features.shape[-1]
            or exact_idx >= cat_features.shape[-1]
            or frac_idx >= float_features.shape[-1]
        ):
            return cat_features
        buckets = compute_matched_word_fraction_bucket(
            float_features[:, :, frac_idx], cat_features[:, :, exact_idx] != 0
        )
        return cat_features.at[:, :, bucket_idx].set(buckets.astype(cat_features.dtype))

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
            lr_multiplier=_config.model_config.scale_config.head_lr_multiplier(emb_size),
            rms_clip_axes=(-1, -2),
        )
        self.unembed_mat: jax.Array = with_sharding_constraint_unless_manual(unembed_mat, out_pspec)
        return unembed_mat

    @hk.transparent
    def _get_dpa_product_embedding_table(self) -> jax.Array:
        _config = self.config
        dim = _config.dpa_product_embed_dim
        return get_parameter(
            "dpa_product_embedding_table",
            [_config.dpa_product_table_size, dim],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0.0),
            pspec=P(),
            rms_clip_axes=(-2, -1),
        )

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
            lr_multiplier=_config.model_config.scale_config.head_lr_multiplier(emb_size),
            rms_clip_axes=(-1, -2),
        )
        return with_sharding_constraint_unless_manual(unembed_mat, out_pspec)

    @hk.transparent
    def _value_bin_logits(
        self, inputs: jax.Array, param_prefix: str, bins: ValueBinTable
    ) -> jax.Array:
        _config = self.config
        emb_size = _config.model_config.emb_size
        num_bins = bins.num_bins
        embed_init = hk.initializers.VarianceScaling(_config.embed_init_scale, mode="fan_out")
        unembed = get_parameter(
            f"{param_prefix}_bin_unembeddings",
            [emb_size, num_bins],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=_config.model_config.scale_config.emb_lr_multiplier(emb_size),
            rms_clip_axes=(-1, -2),
            dtype_unwrap=jnp.float32,
        )
        log_prior = bins.log_prior
        bias = get_parameter(
            f"{param_prefix}_bin_bias",
            [num_bins],
            dtype=jnp.float32,
            init=lambda shape, dtype: jnp.asarray(log_prior, dtype=dtype).reshape(shape),
            pspec=P(None),
            lr_multiplier=1.0,
            rms_clip_axes=(),
            dtype_unwrap=jnp.float32,
        )
        return jnp.dot(inputs.astype(jnp.float32), unembed) + bias

    @hk.transparent
    def _purchase_value_bin_logits(self, inputs: jax.Array) -> jax.Array:
        return self._value_bin_logits(inputs, "purchase_value", self.config.purchase_value_bins)

    def _purchase_value_features(self, batch: RecsysFeaturesBatch) -> jax.Array | None:
        if not self.config.purchase_value_history_tower:
            return None
        history_impr_ts = batch["history_seq"]["impr_ts"]
        assert history_impr_ts is not None
        return purchase_value_history_features(
            cast_jax(batch["history_seq"]["continuous_actions"]),
            cast_jax(batch["history_seq"]["auth_hashes"]),
            cast_jax(history_impr_ts),
            cast_jax(batch["candidate_seq"]["auth_hashes"]),
            self.config.web_conv_value_usd_cap,
            ratio_cap=(
                self.config.web_conv_value_ratio_cap
                if self.config.purchase_value_history_ratio_features
                else None
            ),
        )

    @hk.transparent
    def _purchase_value_history_tower(self, inputs: jax.Array, features: jax.Array) -> jax.Array:
        _config = self.config
        emb_size = _config.model_config.emb_size
        proj, hidden = (
            _config.purchase_value_history_tower_proj,
            _config.purchase_value_history_tower_hidden,
        )
        num_bins = _config.purchase_value_bins.num_bins
        lr = _config.model_config.scale_config.emb_lr_multiplier

        def weight(name, fan_in, fan_out, init):
            return get_parameter(
                f"purchase_value_history_tower_{name}",
                [fan_in, fan_out],
                dtype=jnp.float32,
                init=init,
                pspec=P(None, None),
                lr_multiplier=lr(fan_in),
                rms_clip_axes=(-1, -2),
                dtype_unwrap=jnp.float32,
            )

        def bias(name, size):
            return get_parameter(
                f"purchase_value_history_tower_{name}",
                [size],
                dtype=jnp.float32,
                init=hk.initializers.Constant(0.0),
                pspec=P(None),
                lr_multiplier=1.0,
                rms_clip_axes=(),
                dtype_unwrap=jnp.float32,
            )

        fan_in_init = hk.initializers.VarianceScaling(1.0, mode="fan_in")
        w_state = weight("state_proj", emb_size, proj, fan_in_init)
        w_in = weight("w_in", features.shape[-1] + proj, hidden, fan_in_init)
        b_in = bias("b_in", hidden)
        w_out = weight("w_out", hidden, num_bins, hk.initializers.Constant(0.0))
        b_out = bias("b_out", num_bins)
        pad = inputs.shape[1] - features.shape[1]
        feats = jnp.pad(features.astype(jnp.float32), ((0, 0), (0, pad), (0, 0)))
        state = jnp.dot(jax.lax.stop_gradient(inputs).astype(jnp.float32), w_state)
        x = jnp.concatenate([feats, state], axis=-1)
        return jnp.dot(jax.nn.gelu(jnp.dot(x, w_in) + b_in), w_out) + b_out

    @hk.transparent
    def decode_aux_value_heads(self, inputs: jax.Array) -> dict[str, jax.Array]:
        return {
            head.event: self._value_bin_logits(inputs, head.param_prefix, head.bins)
            for head in self.config.active_aux_value_heads
        }

    @hk.transparent
    def _decode_cread(
        self, inputs: jax.Array, loss_config: ContinuousActionLossConfig
    ) -> jax.Array:
        _config = self.config
        emb_size = _config.model_config.emb_size
        thresholds = loss_config.cread_threshold_values
        num_thresholds = len(thresholds)
        embed_init = hk.initializers.VarianceScaling(_config.embed_init_scale, mode="fan_out")
        unembed_mat = get_parameter(
            f"{loss_config.metric_name}_cread_unembeddings",
            [emb_size, num_thresholds],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=_config.model_config.scale_config.head_lr_multiplier(emb_size),
            rms_clip_axes=(-1, -2),
        )
        cread_logits = jnp.dot(inputs.astype(unembed_mat.dtype), unembed_mat)
        assert loss_config.metric_name is not None
        self.cread_logits[loss_config.metric_name] = cread_logits
        phi = jax.nn.sigmoid(cread_logits.astype(jnp.float32))
        deltas = jnp.asarray(
            [t - t_prev for t_prev, t in zip((0.0,) + thresholds, thresholds)], dtype=jnp.float32
        )
        restored = jnp.dot(phi, deltas)[..., None]
        return restored.astype(inputs.dtype)

    @hk.transparent
    def decode_continuous(
        self,
        inputs: jax.Array,
        product_surface: jax.Array | None = None,
        purchase_value_features: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[int, jax.Array]]:
        self.cread_logits: dict[str, jax.Array] = {}
        unembeddings = self._get_continuous_unembedding()
        logits = jnp.dot(inputs.astype(unembeddings.dtype), unembeddings).astype(inputs.dtype)

        configs_by_index: dict[int, list[ContinuousActionLossConfig]] = {}
        owner_by_index: dict[int, ContinuousActionLossConfig] = {}
        for lc in self.config.continuous_action_losses:
            if lc.loss_type == "binary":
                if lc.loss_weight > 0:
                    assert lc.action_index not in owner_by_index, (
                        f"multiple active binary configs for action index {lc.action_index}"
                    )
                    owner_by_index[lc.action_index] = lc
                continue
            if lc.loss_weight > 0:
                configs_by_index.setdefault(lc.action_index, []).append(lc)

        num_heads = logits.shape[-1]
        activated_slices = []
        head_logits_by_index: dict[int, jax.Array] = {}

        for i in range(num_heads):
            head_logits = logits[..., i : i + 1]
            configs = configs_by_index.get(i, [])

            if i == PURCHASE_VALUE_ACTION_INDEX and self.config.purchase_value_enabled:
                if self.config.purchase_value_binned:
                    bin_logits = self._purchase_value_bin_logits(inputs)
                    if self.config.purchase_value_history_tower:
                        assert purchase_value_features is not None
                        bin_logits = bin_logits + self._purchase_value_history_tower(
                            inputs, purchase_value_features
                        )
                    head_logits_by_index[i] = bin_logits
                    head_out = self.config.purchase_value_bins.readout(bin_logits)[..., None]
                else:
                    head_out = jax.nn.softplus(head_logits.astype(jnp.float32))
                activated_slices.append(head_out.astype(head_logits.dtype))
                continue

            owner_config = owner_by_index.get(i)
            if owner_config is not None:
                assert all(c.loss_weight == 0 for c in configs), (
                    f"action index {i} has an active binary config; scalar configs on the "
                    f"same slot must have loss_weight == 0"
                )
                head_logits_by_index[i] = head_logits.astype(jnp.float32)
                activated_slices.append(
                    jax.nn.sigmoid(head_logits.astype(jnp.float32)).astype(logits.dtype)
                )
                continue

            if not configs:
                activated_slices.append(head_logits)
                continue

            def _activated_for_config(
                c: ContinuousActionLossConfig, head_logits: jax.Array
            ) -> jax.Array:
                if c.loss_type == "cread":
                    return self._decode_cread(inputs, c)
                out = self._apply_activation(head_logits, c.activation)
                if c.output_cap > 0:
                    out = c.output_cap * jnp.tanh(out / c.output_cap)
                return out

            needs_surface = len(configs) > 1 or any(
                c.product_surfaces or c.exclude_product_surfaces for c in configs
            )
            if not needs_surface:
                activated_slices.append(_activated_for_config(configs[0], head_logits))
            else:
                assert product_surface is not None, (
                    f"product_surface required: action index {i} has surface-filtered configs"
                )
                head_out = head_logits
                for c in configs:
                    smask = _get_surface_mask(c, product_surface)[..., None]
                    head_out = jnp.where(smask, _activated_for_config(c, head_logits), head_out)
                activated_slices.append(head_out)

        return jnp.concatenate(activated_slices, axis=-1), head_logits_by_index

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
    def _add_side_inputs(
        self,
        batch: RecsysFeaturesBatch,
        history_embeddings: jax.Array,
        candidate_embeddings: jax.Array,
        layout: SequencePackedLayout | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.config
        D = cfg.emb_table_width
        scale = cfg.embed_init_scale
        lr_multiplier_func = cfg.model_config.scale_config.emb_lr_multiplier
        f32 = jnp.float32

        def add(emb: jax.Array, feat: jax.Array, name: str) -> jax.Array:
            din = feat.shape[-1]
            w = get_parameter(
                name,
                [din, D],
                dtype=f32,
                init=hk.initializers.Constant(0.0),
                pspec=P(None, None),
                lr_multiplier=lr_multiplier_func(din),
            )
            return emb + jnp.dot(feat.astype(w.dtype), w).astype(emb.dtype)

        if cfg.ads_counts_emb_dim > 0:
            counts = batch["candidate_seq"].get("ads_count_buckets")
            assert counts is not None, (
                "ads_counts_emb_dim needs the dataset flag read_ads_engagement_counts=True"
            )
            counts = cast_jax(counts).astype(jnp.int32)
            num_feats = counts.shape[-1]
            dim = cfg.ads_counts_emb_dim
            table = get_parameter(
                "ads_counts_table",
                shape=[num_feats * ADS_COUNT_BUCKETS, dim],
                init=hk.initializers.VarianceScaling(scale, mode="fan_out"),
                dtype=f32,
                pspec=P(),
            )
            offsets = jnp.arange(num_feats, dtype=jnp.int32) * ADS_COUNT_BUCKETS
            emb = jnp.take(table, counts + offsets, axis=0).astype(DTYPE_BY_NAME[cfg.fprop_dtype])
            emb = emb.reshape(*emb.shape[:-2], num_feats * dim)
            pad = candidate_embeddings.shape[1] - emb.shape[1]
            if pad > 0:
                emb = jnp.pad(emb, ((0, 0), (0, pad), (0, 0)))
            candidate_embeddings = add(candidate_embeddings, emb, "ads_counts_proj")

        if cfg.wallclock_recency:
            hist_impr = batch["history_seq"]["impr_ts"]
            cand_impr = batch["candidate_seq"]["impr_ts"]
            hist_act = batch["history_seq"]["actions"]
            assert hist_impr is not None and cand_impr is not None and hist_act is not None, (
                "wallclock_recency needs impression timestamps and history actions"
            )
            hist_ts = cast_jax(hist_impr).astype(jnp.int32)
            cand_ts = cast_jax(cand_impr).astype(jnp.int32)
            rows = hist_ts.shape[0]
            if layout is None:
                num_users = 1
                hist_user = jnp.zeros(hist_ts.shape, jnp.int32)
            else:
                cu = cast_jax(layout.cu_seqlens).astype(jnp.int32)
                num_users = cu.shape[1] - 1
                pos = cast_jax(layout.history_positions).astype(jnp.int32)
                hist_user = jax.vmap(lambda c, p: jnp.searchsorted(c[1:], p, side="right"))(cu, pos)
                hist_user = jnp.clip(hist_user, 0, num_users - 1).astype(jnp.int32)
            t_req = jnp.max(cand_ts.reshape(rows, num_users, -1), axis=2)
            t_tok = jnp.take_along_axis(t_req, hist_user, axis=1)
            hist_valid = (hist_ts > 0) & (t_tok > 0) & (hist_ts <= t_tok + 60)
            dt = jnp.where(hist_valid, jnp.maximum(t_tok - hist_ts, 0), 0).astype(f32)
            x = (jnp.log1p(dt) / RECENCY_LOG_DT_SCALE)[..., None]
            hist_feat = recency_time_features(x) * hist_valid[..., None].astype(f32)
            history_embeddings = add(history_embeddings, hist_feat, "recency_hist_proj")

            owner = hist_user[..., None] == jnp.arange(num_users, dtype=jnp.int32)

            def per_user_max(v: jax.Array) -> jax.Array:
                return jnp.max(jnp.where(owner, v[..., None], jnp.asarray(-1, v.dtype)), axis=1)

            def per_user_sum(v: jax.Array) -> jax.Array:
                return jnp.sum(jnp.where(owner, v[..., None], jnp.asarray(0, v.dtype)), axis=1)

            def since(mask: jax.Array) -> tuple[jax.Array, jax.Array]:
                last = per_user_max(jnp.where(mask, hist_ts, -1))[..., None]
                has = last > 0
                age = jnp.maximum(t_req[..., None] - last, 0).astype(f32)
                gap = jnp.where(has, jnp.log1p(age) / RECENCY_LOG_DT_SCALE, 1.0)
                return gap, has.astype(f32)

            hist_actions = cast_jax(hist_act)
            gap_any, has_any = since(hist_valid)
            parts = [gap_any, recency_time_features(gap_any, 4), has_any]
            for idx in RECENCY_ACTIONS:
                g, h = since(hist_valid & hist_actions[:, :, idx].astype(jnp.bool_))
                parts += [g, h]
            for window_s in (300.0, 3600.0, 86400.0):
                n = per_user_sum((hist_valid & (dt <= window_s)).astype(jnp.int32))
                parts.append(jnp.log1p(n.astype(f32))[..., None] / math.log(1024.0))
            sess = jnp.concatenate(parts, axis=-1)
            per_user = candidate_embeddings.shape[1] // num_users
            sess = jnp.broadcast_to(
                sess[:, :, None, :], (rows, num_users, per_user, sess.shape[-1])
            ).reshape(rows, num_users * per_user, sess.shape[-1])
            candidate_embeddings = add(candidate_embeddings, sess, "recency_cand_proj")

        return history_embeddings, candidate_embeddings

    @hk.transparent
    def _maybe_add_dpa_input_embedding(
        self,
        candidate_embeddings: jax.Array,
        recsys_features_batch: RecsysFeaturesBatch,
    ) -> jax.Array:
        _config = self.config
        if not _config.enable_dpa_input_embedding:
            return candidate_embeddings
        table = self._get_dpa_product_embedding_table()
        dpa_dim = _config.dpa_product_embed_dim
        embed_init = hk.initializers.VarianceScaling(_config.embed_init_scale, mode="fan_out")
        proj = get_parameter(
            "dpa_input_proj",
            [dpa_dim, _config.emb_table_width],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=_config.model_config.scale_config.emb_lr_multiplier(dpa_dim),
        )
        raw_i64 = recsys_features_batch["candidate_seq"].get("int64_features")
        if raw_i64 is None:
            return candidate_embeddings
        keys = cast_jax(raw_i64)[:, :, _DPA_PRODUCT_KEY_SLOTS].astype(jnp.int32)
        assert keys.shape[:2] == candidate_embeddings.shape[:2], (
            f"dpa keys shape {keys.shape} must match candidate embeddings "
            f"{candidate_embeddings.shape[:2]}"
        )
        valid = keys != 0
        ids = jnp.where(valid, jnp.clip(keys, 1, _config.dpa_product_table_size - 1), 0)
        rows = jnp.take(table, ids, axis=0)
        product_emb = jnp.where(valid[..., None], rows, 0.0).sum(axis=2)
        emb = jnp.dot(product_emb.astype(proj.dtype), proj)
        return candidate_embeddings + emb.astype(candidate_embeddings.dtype)

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
            assert not _config.enable_dpa_input_embedding, (
                "enable_dpa_input_embedding is not implemented for the feature_prep path"
            )
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
                mesh=self.sharding_context.mesh if is_training else None,
                data_axis=self.data_axis,
            )
            tokens = with_sharding_constraint_unless_manual(
                tokens, P(self.data_axis, ("seq", "model"))
            )
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
        history_web_conv_time_on_site: jax.Array | None = None
        history_web_conv_value: jax.Array | None = None
        history_continuous_actions = recsys_features_batch["history_seq"]["continuous_actions"]
        if history_continuous_actions is not None:
            ca = cast_jax(history_continuous_actions)
            if ca.shape[-1] > 1:
                history_dwell_time = ca[:, :, 1]
            _bridge_idx = recsys_pb2.ContinuousActionName.BRIDGE_PROBABILITY
            if _config.concat_history_bridge_prob and ca.shape[-1] > _bridge_idx:
                history_bridge_prob = ca[:, :, _bridge_idx]
            if _config.add_history_web_conv_value and ca.shape[-1] > WEB_CONV_VALUE_RATIO_IDX:
                history_web_conv_value = normalize_web_conv_value(
                    ca[:, :, [WEB_CONV_VALUE_USD_IDX, WEB_CONV_VALUE_RATIO_IDX]],
                    _config.web_conv_value_usd_cap,
                    _config.web_conv_value_ratio_cap,
                )
        if _config.concat_history_web_conv_time_on_site:
            history_int64 = recsys_features_batch["history_seq"].get("int64_features")
            _tos_idx = [
                Int64Feature.webConvTimeOnSiteInferredMsSeq,
                Int64Feature.webConvTimeOnSiteMeasuredMsSeq,
            ]
            if history_int64 is not None and history_int64.shape[-1] > max(_tos_idx):
                history_web_conv_time_on_site = normalize_web_conv_time_on_site(
                    cast_jax(history_int64)[:, :, _tos_idx],
                    _config.web_conv_time_on_site_norm_scale,
                )

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

        if ctx_config.enable_search_lexical_match and candidate_cat_features is not None:
            raw_cand_float = recsys_features_batch["candidate_seq"].get("float_features")
            candidate_cat_features = self.set_matched_word_fraction_bucket(
                candidate_cat_features,
                None if raw_cand_float is None else cast_jax(raw_cand_float),
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
            dense=_config.use_dense_action_table,
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

        history_url_embeddings: jax.Array | None = None
        if _config.history_url_embedding_dim > 0:
            history_embedding = recsys_features_batch["history_seq"].get("embedding")
            assert history_embedding is not None, (
                "history_seq embedding must be present when history_url_embedding_dim > 0"
            )
            assert history_embedding.shape[-1] == _config.history_url_embedding_dim, (
                f"history_seq embedding width {history_embedding.shape[-1]} != "
                f"history_url_embedding_dim {_config.history_url_embedding_dim}"
            )
            history_url_embeddings = (
                history_embedding
                if isinstance(history_embedding, jax.Array)
                else cast_jax(history_embedding)
            )

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
                history_web_conv_time_on_site=(
                    None
                    if history_web_conv_time_on_site is None
                    else history_web_conv_time_on_site.reshape(num_devices, -1, 2)
                ),
                history_url_embeddings=(
                    None
                    if history_url_embeddings is None
                    else history_url_embeddings.reshape(
                        num_devices, -1, _config.history_url_embedding_dim
                    )
                ),
                history_web_conv_value=(
                    None
                    if history_web_conv_value is None
                    else history_web_conv_value.reshape(num_devices, -1, 2)
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
            candidate_embeddings = self._maybe_add_dpa_input_embedding(
                candidate_embeddings, recsys_features_batch
            )
            if _config.has_side_inputs:
                history_embeddings, candidate_embeddings = self._add_side_inputs(
                    recsys_features_batch, history_embeddings, candidate_embeddings, layout=layout
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
            seq_starts = cast_jax(packed_prefix_positions(layout))
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
                if _config.sid_embedding_mode == "recon":
                    return reconstruct_entity_sid(
                        sids_jax,
                        _config.emb_table_width,
                        _config.sid_decoder_path,
                        self.config.model_config.scale_config.emb_lr_multiplier,
                        self.config.embed_init_scale,
                        DTYPE_BY_NAME[_config.fprop_dtype],
                        "post",
                    )
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
                    sid_prefix2_embed_dim=_config.sid_prefix2_embed_dim,
                    sid_prefix3_embed_dim=_config.sid_prefix3_embed_dim,
                    sid_prefix3_rows=_config.sid_prefix3_rows,
                )

            _sid_post_emb_h = _embed_post_sid("history_seq") if _config.use_post_sid else None
            _sid_post_emb_c = _embed_post_sid("candidate_seq") if _config.use_post_sid else None
            _sid_recon = _config.use_post_sid and _config.sid_embedding_mode == "recon"

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
                sid_post_embeddings=None if _sid_recon else _sid_post_emb_h,
                history_bridge_prob=history_bridge_prob,
                history_web_conv_time_on_site=history_web_conv_time_on_site,
                history_url_embeddings=history_url_embeddings,
                history_web_conv_value=history_web_conv_value,
            )
            if _sid_recon and _sid_post_emb_h is not None:
                history_embeddings += _sid_post_emb_h.astype(history_embeddings.dtype)

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
                sid_post_embeddings=None if _sid_recon else _sid_post_emb_c,
            )
            if _sid_recon and _sid_post_emb_c is not None:
                candidate_embeddings += _sid_post_emb_c.astype(candidate_embeddings.dtype)
            candidate_embeddings = self._maybe_add_dpa_input_embedding(
                candidate_embeddings, recsys_features_batch
            )
            if _config.has_side_inputs:
                history_embeddings, candidate_embeddings = self._add_side_inputs(
                    recsys_features_batch, history_embeddings, candidate_embeddings
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
        embeddings = with_sharding_constraint_unless_manual(
            embeddings, P(self.data_axis, ("seq", "model"))
        )

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
        return_candidate_embeddings: bool = False,
        act_l2_mask: jax.Array | None = None,
        with_aux_value_heads: bool = False,
        purchase_value_features: jax.Array | None = None,
    ) -> tuple[jax.Array, ...]:
        _debug_dir = self.config.model_config.debug_tensor_dump_output_folder
        dump_to_file(_debug_dir, "input_embeddings", input_embeddings)

        config = self.config
        scale_config = config.model_config.scale_config
        fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]

        B, T, _ = input_embeddings.shape
        if segment_ids is None:
            segment_ids = jnp.zeros((B, T), dtype=jnp.int32)

        attn_mask = jnp.ones((B, T)).astype(jnp.int32)

        if seqpack_layout is None:
            candidate_start = candidate_start_offset
        elif seqpack_layout.candidate_token_to_slot is not None:
            candidate_start = T - seqpack_layout.candidate_token_to_slot.shape[-1]
        else:
            candidate_start = None

        model_out = self.model(
            input_embeddings,
            attn_mask,
            segment_ids=segment_ids,
            is_training=is_training,
            decoding=False,
            padding_mask=padding_mask,
            positions=positions,
            seqpack_layout=seqpack_layout,
            act_l2_mask=act_l2_mask,
            candidate_start=candidate_start,
        )

        out_embeddings = model_out.output

        out_embeddings = with_sharding_constraint_unless_manual(
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
            return out_embeddings, jnp.zeros((0, 0, 0)), {}, {}

        assert candidate_start_offset is None or seqpack_layout is None

        if return_candidate_embeddings:
            if seqpack_layout is not None:
                user_positions = cast_jax(packed_prefix_positions(seqpack_layout)).astype(jnp.int32)
                user_embeddings = jnp.take_along_axis(
                    out_embeddings, user_positions[:, :, None], axis=1
                )
            else:
                user_embeddings = out_embeddings[:, 0, :]

        if seqpack_layout is not None and seqpack_layout.candidate_token_to_slot is not None:
            out_embeddings = take_candidate_slots(
                out_embeddings.shape[1],
                out_embeddings,
                cast_jax(seqpack_layout.candidate_positions),
                cast_jax(seqpack_layout.candidate_token_to_slot),
            )
        elif seqpack_layout is not None:
            out_embeddings = jnp.take_along_axis(
                out_embeddings, cast_jax(seqpack_layout.candidate_positions)[:, :, None], axis=1
            )
        elif candidate_start_offset is not None:
            out_embeddings = out_embeddings[:, candidate_start_offset:, :]

        logits = self.decode(out_embeddings)
        logits *= scale_config.output_scale(config.model_config.emb_size)
        logits = with_sharding_constraint_unless_manual(logits, P(self.data_axis, ("seq", "model")))

        dump_to_file(_debug_dir, "after_unembeddings", logits)

        if config.final_logit_cap > 0.0:
            logits = config.final_logit_cap * jnp.tanh(logits / config.final_logit_cap)

        continuous_predictions, head_logits_by_index = self.decode_continuous(
            out_embeddings, product_surface, purchase_value_features
        )
        aux_value_logits = (
            self.decode_aux_value_heads(out_embeddings) if with_aux_value_heads else {}
        )

        if not return_candidate_embeddings:
            return logits, continuous_predictions, head_logits_by_index, aux_value_logits
        if seqpack_layout is not None:
            input_candidates = jnp.take_along_axis(
                input_embeddings, cast_jax(seqpack_layout.candidate_positions)[:, :, None], axis=1
            )
        elif candidate_start_offset is not None:
            input_candidates = input_embeddings[:, candidate_start_offset:, :]
        else:
            input_candidates = input_embeddings
        dwell_norm_scale = max(
            (
                lc.norm_config.norm_scale
                for lc in config.continuous_action_losses
                if lc.action_index == recsys_pb2.ContinuousActionName.DWELL_TIME
            ),
            default=300.0,
        )
        candidate_embeddings = jnp.concatenate(
            [
                out_embeddings,
                rms_norm_fn(input_candidates, name="page_rl_input_norm", create_scale=False).astype(
                    out_embeddings.dtype
                ),
                page_rl_head_features(logits, continuous_predictions, dwell_norm_scale).astype(
                    out_embeddings.dtype
                ),
            ],
            axis=-1,
        )
        return (
            logits,
            continuous_predictions,
            head_logits_by_index,
            aux_value_logits,
            candidate_embeddings,
            user_embeddings,
        )

    @hk.transparent
    def loss_inputs_and_normalizers(
        self, batches: Sequence[RecsysFeaturesBatch], num_partials: int | None = None
    ) -> tuple[CandidateInputs, dict[str, jax.Array]]:
        negatives = []
        for batch in batches:
            actions = batch["candidate_seq"]["actions"]
            assert actions is not None
            if self.config.use_seqpack:
                packed_candidate_seq_len = actions.shape[1] // batch["user_hashes"].shape[1]
                negatives.append(
                    (
                        jnp.arange(actions.shape[1], dtype=jnp.int32)[None, :]
                        % packed_candidate_seq_len
                    )
                    >= self.config.candidate_seq_len
                )
            else:
                mask = jnp.zeros((actions.shape[0], actions.shape[1]), dtype=jnp.bool_)
                negatives.append(mask.at[:, self.config.candidate_seq_len :].set(True))

        logq_weights: Sequence[jax.Array | None] = [None] * len(batches)
        if self.config.log_q_correction:
            counts = get_candidate_tweet_counts(
                jnp.concatenate(
                    [cast_jax(b["candidate_seq"]["post_hashes"]) for b in batches], axis=1
                ),
                self.config.log_q_num_bins,
                jnp.concatenate(negatives, axis=1),
            )
            logq_weights = jnp.split(
                jnp.where(counts == 0.0, 1.0, 1.0 / counts), len(batches), axis=1
            )

        candidate_inputs, totals = [], []
        for batch, negative_sample_mask, logq_weight in zip(batches, negatives, logq_weights):
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
            raw_candidate_int64 = batch["candidate_seq"].get("int64_features")
            dpa_product_key = (
                cast_jax(raw_candidate_int64)[:, :, _DPA_PRODUCT_KEY_SLOTS]
                if raw_candidate_int64 is not None
                else None
            )
            candidate_safety_mask = (
                cast_jax(candidate_safety_mask) if candidate_safety_mask is not None else None
            )
            raw_weights: jax.Array | None = None
            if self.config.use_seqpack:
                packed_candidate_seq_len = targets.shape[1] // batch["user_hashes"].shape[1]

            sample_weights = batch.get("sample_weights")
            if sample_weights is not None:
                sample_weights = cast_jax(sample_weights).astype(jnp.float32)
                if self.config.use_seqpack:
                    raw_weights = jnp.repeat(
                        sample_weights.squeeze(-1), packed_candidate_seq_len, axis=1
                    )
                else:
                    raw_weights = jnp.broadcast_to(sample_weights, targets.shape[:2])

            source_id: jax.Array | None = None
            sample_source = batch.get("sample_source")
            if sample_source is not None:
                source_id = cast_jax(sample_source).astype(jnp.float32)
                if self.config.use_seqpack:
                    source_id = jnp.repeat(source_id.squeeze(-1), packed_candidate_seq_len, axis=1)
            if self.config.ads_head_masking:
                assert source_id is not None, (
                    "ads_head_masking requires the sample_source batch field (from the "
                    "sample_source column); training unsplit silently would defeat it"
                )

            if logq_weight is not None:
                raw_weights = logq_weight if raw_weights is None else raw_weights * logq_weight

            trained_heads: jax.Array | None = None
            eligible_heads: jax.Array | None = None
            if self.config.use_seqpack:
                layout = batch.get("packing_layout")
                assert layout is not None
                target_padding_mask = jnp.take_along_axis(
                    cast_jax(layout.padding_mask), cast_jax(layout.candidate_positions), axis=1
                )
            else:
                target_padding_mask = cast_jax(batch["candidate_seq"]["post_hashes"])[:, :, 0] != 0
                head_eligible_mask = batch["candidate_seq"].get("head_eligible_mask")
                if self.config.ads_pixel_head_masking and head_eligible_mask is not None:
                    eligible_heads = cast_jax(head_eligible_mask)
            trained_candidate_mask = batch["candidate_seq"].get("trained_candidate_mask")
            value_padding_mask = target_padding_mask
            if trained_candidate_mask is not None:
                trained_heads = cast_jax(trained_candidate_mask)
                target_padding_mask = target_padding_mask & trained_heads.any(-1)

            negative_sample_mask = jnp.broadcast_to(negative_sample_mask, target_padding_mask.shape)

            loss_padding_mask, loss_weights = target_padding_mask, raw_weights
            if self.config.safety_filter_apply_to_candidates and self.config.value_heads():
                value_padding_mask, _ = apply_safety_filter(
                    candidate_safety_mask,
                    value_padding_mask,
                    None,
                    mode=self.config.safety_filter_mode,
                    bits=self.config.safety_filter_bits,
                    soft_weight=self.config.safety_filter_soft_weight,
                )
            if self.config.safety_filter_apply_to_candidates:
                loss_padding_mask, loss_weights = apply_safety_filter(
                    candidate_safety_mask,
                    target_padding_mask,
                    raw_weights,
                    mode=self.config.safety_filter_mode,
                    bits=self.config.safety_filter_bits,
                    soft_weight=self.config.safety_filter_soft_weight,
                )
            total = {"multihot": jnp.sum(multihot_loss_weights(loss_padding_mask, loss_weights))}

            continuous_heads = {}
            if candidate_continuous_actions is not None:
                continuous_base_mask = loss_padding_mask
                if self.config.ads_head_masking and source_id is not None:
                    continuous_base_mask = continuous_base_mask * (
                        1 - (source_id > 0).astype(continuous_base_mask.dtype)
                    )
                for index, loss_config in enumerate(self.config.continuous_action_losses):
                    if (
                        loss_config.loss_weight > 0
                        and loss_config.action_index < candidate_continuous_actions.shape[-1]
                    ):
                        head_valid_mask = continuous_base_mask
                        head_mask = None
                        if loss_config.product_surfaces or loss_config.exclude_product_surfaces:
                            head_mask = _get_surface_mask(loss_config, product_surface)
                        if loss_config.condition_action_index is not None:
                            condition_mask = targets[:, :, loss_config.condition_action_index] > 0
                            head_mask = (
                                condition_mask if head_mask is None else head_mask & condition_mask
                            )
                        if head_mask is not None:
                            head_valid_mask = head_valid_mask * head_mask.astype(
                                head_valid_mask.dtype
                            )
                        if loss_config.action_index == recsys_pb2.ContinuousActionName.DWELL_TIME:
                            dwell = candidate_continuous_actions[:, :, loss_config.action_index]
                            in_range = (dwell >= MIN_VALID_DWELL_SEC) & (
                                dwell <= MAX_VALID_DWELL_SEC
                            )
                            head_valid_mask = head_valid_mask * in_range.astype(
                                head_valid_mask.dtype
                            )
                        continuous_heads[index] = (head_valid_mask, head_mask)
                        valid_mask = head_valid_mask
                        if loss_config.loss_type == "cread":
                            valid_mask = valid_mask.astype(jnp.bool_)
                        _, weights = continuous_loss_weights(
                            valid_mask,
                            negative_sample_mask,
                            loss_config.mask_negatives,
                            loss_weights,
                        )
                        if (
                            loss_config.loss_type == "cread"
                            and loss_config.cread_mask_sentinel_labels
                        ):
                            gt_raw = candidate_continuous_actions[
                                :, :, loss_config.action_index
                            ].astype(jnp.float32)
                            weights = weights * (
                                (gt_raw != 0.0) & (gt_raw != DWELL_TIME_DEFAULT_SENTINEL_SEC)
                            )
                        total[f"continuous/{index}"] = jnp.sum(weights)

            value_heads = {}
            for head in self.config.value_heads():

                def value_operand(
                    name: str, dtype: jnp.dtype, batch=batch, mask=loss_padding_mask
                ) -> jax.Array:
                    value = batch["candidate_seq"].get(name)
                    if value is None:
                        return jnp.zeros(mask.shape, dtype=dtype)
                    value = cast_jax(value).astype(dtype)
                    pad_len = mask.shape[1] - value.shape[1]
                    return jnp.pad(value, ((0, 0), (0, pad_len)))

                keeper_mask = (
                    trained_heads.any(-1)
                    if trained_heads is not None
                    else jnp.ones_like(loss_padding_mask, dtype=jnp.bool_)
                )
                value_mask = value_valid_mask(
                    label_valid=value_operand(head.valid_key, jnp.bool_),
                    padding_mask=value_padding_mask,
                    negative_sample_mask=negative_sample_mask,
                    sample_source=source_id
                    if source_id is not None
                    else jnp.zeros_like(loss_padding_mask),
                    has_click=targets[:, :, CLICK_ACTION_INDEX],
                    keeper_mask=keeper_mask,
                    lanes=head.lanes,
                    include_view_through=head.include_view_through,
                )
                if head.slot is not None:
                    value_ratio = (
                        candidate_continuous_actions[:, :, head.slot]
                        if candidate_continuous_actions is not None
                        and candidate_continuous_actions.shape[-1] > head.slot
                        else jnp.zeros_like(loss_padding_mask, dtype=jnp.float32)
                    )
                else:
                    assert head.ratio_key is not None
                    value_ratio = value_operand(head.ratio_key, jnp.float32)
                baseline = value_operand(head.baseline_key, jnp.float32)
                value_heads[head.ema_key] = (value_ratio, baseline, value_mask)
                _, weights = value_valid_weights(value_ratio, baseline, value_mask, loss_weights)
                total[f"value/{head.ema_key}"] = jnp.sum(weights)

            candidate_inputs.append(
                CandidateInputs(
                    targets=targets,
                    target_padding_mask=target_padding_mask,
                    negative_sample_mask=negative_sample_mask,
                    product_surface=product_surface,
                    promoted_ids=promoted_ids,
                    continuous_actions=candidate_continuous_actions,
                    client_app_id=client_app_id,
                    line_item_objective=line_item_objective,
                    safety_label_mask=candidate_safety_mask,
                    dpa_product_key=dpa_product_key,
                    source_id=source_id,
                    trained_heads=trained_heads,
                    eligible_heads=eligible_heads,
                    loss_padding_mask=loss_padding_mask,
                    loss_weights=loss_weights,
                    continuous_heads=continuous_heads,
                    value_heads=value_heads,
                )
            )
            totals.append(total)
        normalizers = jax.tree.map(lambda *x: sum(x, jnp.float32(0)), *totals)
        normalizers["num_partials"] = jnp.float32(num_partials or len(batches))
        return jax.tree.map(lambda *x: jnp.stack(x), *candidate_inputs), normalizers

    @hk.transparent
    def loss(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddingsParameter,
        is_training: bool = True,
        candidate_inputs: CandidateInputs | None = None,
        loss_normalizers: dict[str, jax.Array] | None = None,
    ) -> tuple[jax.Array, tuple[dict[str, Any], dict[str, Any]]]:
        assert self.config.model_config.output_vocab_size is not None, (
            "RecsysAggregatedModel output_vocab_size is None"
        )
        input_embeddings, padding_mask, candidate_start_offset, mm_emb = self.build_inputs(
            batch,
            get_recsys_embed_param_to_jax_array(recsys_embeddings),
            is_training=is_training,
        )
        c = candidate_inputs
        if c is None:
            c = jax.tree.map(lambda x: x[0], self.loss_inputs_and_normalizers([batch])[0])
        purchase_value_features = None

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
            cu_seqlens = cast_jax(layout.cu_seqlens)
            bs_per_device = batch["user_hashes"].shape[1]
            packed_candidate_seq_len = layout.candidate_positions.shape[1] // bs_per_device

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
                (history_event_count < 128).astype(c.target_padding_mask.dtype),
                packed_candidate_seq_len,
                axis=1,
            )
            no_history_mask = jnp.repeat(
                (history_event_count == 0).astype(c.target_padding_mask.dtype),
                packed_candidate_seq_len,
                axis=1,
            )

            act_l2_mask = (
                untrained_candidates_excluded(
                    padding_mask,
                    c.trained_heads,
                    candidate_positions=cast_jax(layout.candidate_positions),
                )
                if self.config.page_rl.enabled
                else None
            )
            with Summary() as summarizer:
                outputs = self(
                    input_embeddings,
                    padding_mask,
                    is_training=is_training,
                    positions=positions,
                    segment_ids=segment_ids,
                    seqpack_layout=layout,
                    product_surface=c.product_surface,
                    return_candidate_embeddings=self.config.page_rl.enabled,
                    act_l2_mask=act_l2_mask,
                    with_aux_value_heads=True,
                )
        else:
            assert candidate_start_offset is not None
            history_len = None

            if self.config.pad_candidates_to_128:
                (
                    input_embeddings,
                    padding_mask,
                    product_surface,
                    targets,
                    loss_weights,
                    negatives,
                    promoted_ids,
                    continuous_actions,
                    client_app_id,
                    line_item_objective,
                    safety_label_mask,
                    dpa_product_key,
                ) = pad_to_next_128_multiple(
                    input_embeddings,
                    padding_mask,
                    c.product_surface,
                    c.targets,
                    c.loss_weights,
                    c.negative_sample_mask,
                    c.promoted_ids,
                    c.continuous_actions,
                    c.client_app_id,
                    c.line_item_objective,
                    c.safety_label_mask,
                    c.dpa_product_key,
                )
                assert targets is not None and negatives is not None
                pad_len = targets.shape[1] - c.targets.shape[1]

                def pad(x: jax.Array) -> jax.Array:
                    return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2))

                c = c._replace(
                    targets=targets,
                    negative_sample_mask=negatives,
                    loss_weights=loss_weights,
                    product_surface=product_surface,
                    promoted_ids=promoted_ids,
                    continuous_actions=continuous_actions,
                    client_app_id=client_app_id,
                    line_item_objective=line_item_objective,
                    safety_label_mask=safety_label_mask,
                    dpa_product_key=dpa_product_key,
                    trained_heads=None
                    if c.trained_heads is None
                    else jnp.pad(
                        c.trained_heads, ((0, 0), (0, pad_len), (0, 0)), constant_values=True
                    ),
                    eligible_heads=None
                    if c.eligible_heads is None
                    else jnp.pad(
                        c.eligible_heads, ((0, 0), (0, pad_len), (0, 0)), constant_values=True
                    ),
                    target_padding_mask=pad(c.target_padding_mask),
                    loss_padding_mask=pad(c.loss_padding_mask),
                    continuous_heads=jax.tree.map(pad, c.continuous_heads),
                    value_heads=jax.tree.map(pad, c.value_heads),
                )

            segment_ids = attention_segment_ids(
                padding_mask, candidate_start_offset, self.config.attend_padded_history
            )

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

            act_l2_mask = (
                untrained_candidates_excluded(
                    padding_mask, c.trained_heads, candidate_start_offset=candidate_start_offset
                )
                if self.config.page_rl.enabled
                else None
            )
            purchase_value_features = self._purchase_value_features(batch)
            with Summary() as summarizer:
                outputs = self(
                    input_embeddings,
                    padding_mask,
                    is_training=is_training,
                    positions=positions,
                    segment_ids=segment_ids.astype(jnp.int32),
                    candidate_start_offset=candidate_start_offset,
                    product_surface=c.product_surface,
                    return_candidate_embeddings=self.config.page_rl.enabled,
                    act_l2_mask=act_l2_mask,
                    with_aux_value_heads=True,
                    purchase_value_features=purchase_value_features,
                )

            history_padding = padding_mask[
                :,
                self.config.num_user_prefix_tokens : candidate_start_offset,
            ]
            num_history_actions = history_padding.sum(axis=1)
            new_user_mask = (num_history_actions < 128).astype(c.target_padding_mask.dtype)[:, None]
            no_history_mask = (num_history_actions == 0).astype(c.target_padding_mask.dtype)[
                :, None
            ]

        candidate_logits, candidate_continuous_preds, head_logits_by_index, aux_logits = outputs[:4]
        targets, negative_sample_mask, trained_heads, eligible_heads = (
            c.targets,
            c.negative_sample_mask,
            c.trained_heads,
            c.eligible_heads,
        )

        targets_for_loss = targets
        if self.config.mask_candidate_positive_when_negative_action_present:
            negative_feedback_action_indices = jnp.array(
                [
                    action_type_map["ServerTweetReport"],
                    action_type_map["ClientTweetReport"],
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
        loss_mask = jnp.ones(targets.shape)

        if self.config.mask_neg_feedback_on_negatives:
            neg_head_mask = (
                jnp.zeros(num_actions).at[jnp.array(NEGATIVE_FEEDBACK_HEAD_INDICES)].set(1.0)
            )
            zero_mask = negative_sample_mask[:, :, None] * neg_head_mask
            loss_mask = loss_mask * (1 - zero_mask)

        if trained_heads is not None:
            loss_mask = loss_mask * trained_heads.astype(loss_mask.dtype)

        if eligible_heads is not None:
            loss_mask = loss_mask * eligible_heads.astype(loss_mask.dtype)

        if self.config.mact_in_app_loss_weight != 1.0:
            mact_w = (
                jnp.ones(num_actions)
                .at[jnp.array(MACT_IN_APP_LOSS_ACTION_INDICES)]
                .set(self.config.mact_in_app_loss_weight)
            )
            loss_mask = loss_mask * mact_w

        if self.config.view_loss_weight != 1.0:
            view_w = (
                jnp.ones(num_actions)
                .at[jnp.array(VIEW_LOSS_ACTION_INDICES)]
                .set(self.config.view_loss_weight)
            )
            loss_mask = loss_mask * view_w

        if self.config.condition_search_relevance_on_prompt:
            prompt_shown = jnp.any(targets[:, :, SEARCH_RELEVANCE_ACTION_INDICES], axis=-1)
            no_prompt = 1 - prompt_shown
            search_head_mask = (
                jnp.zeros(num_actions).at[jnp.array(SEARCH_RELEVANCE_ACTION_INDICES)].set(1.0)
            )
            search_zero_mask = no_prompt[:, :, None] * search_head_mask
            loss_mask = loss_mask * (1 - search_zero_mask)

        loss_padding_mask, loss_weights = c.loss_padding_mask, c.loss_weights

        def normalizer(term: str) -> jax.Array | None:
            return None if loss_normalizers is None else loss_normalizers[term]

        loss, _ = multihot_loss_compute(
            logits=candidate_logits,
            raw_targets=targets_for_loss,
            loss_mask=loss_mask,
            padding_mask=loss_padding_mask,
            raw_weights=loss_weights,
            one_hot_targets_sharding=P(self.data_axis, ("seq", "model")),
            normalizer=normalizer("multihot"),
        )

        continuous_action_loss_total = jnp.array(0.0)
        continuous_metric_inputs = {}
        cread_stats = {}
        for index, (head_valid_mask, _) in c.continuous_heads.items():
            loss_config = self.config.continuous_action_losses[index]
            assert c.continuous_actions is not None
            gt_raw = c.continuous_actions[:, :, loss_config.action_index]
            pred_raw = candidate_continuous_preds[:, :, loss_config.action_index]
            head_normalizer = normalizer(f"continuous/{index}")

            if loss_config.loss_type == "cread":
                assert loss_config.metric_name is not None
                sentinel_values = (
                    (0.0, DWELL_TIME_DEFAULT_SENTINEL_SEC)
                    if loss_config.cread_mask_sentinel_labels
                    else ()
                )
                (
                    action_loss,
                    gt_clamped,
                    pred_in_original_units,
                    cont_loss_mask,
                    per_element_loss,
                    cread_cls_loss,
                    cread_restoration_loss,
                ) = cread_loss_compute(
                    gt_raw=gt_raw,
                    cread_logits=self.cread_logits[loss_config.metric_name],
                    restored_pred=pred_raw,
                    valid_mask=head_valid_mask,
                    negative_sample_mask=c.negative_sample_mask,
                    thresholds=loss_config.cread_threshold_values,
                    norm_scale=loss_config.norm_config.norm_scale,
                    restoration_weight=loss_config.cread_restoration_weight,
                    mask_negatives=loss_config.mask_negatives,
                    sentinel_values=sentinel_values,
                    raw_weights=loss_weights,
                    restoration_loss_type=loss_config.cread_restoration_loss,
                    huber_delta=(
                        loss_config.cread_huber_delta_sec / loss_config.norm_config.norm_scale
                    ),
                    normalizer=head_normalizer,
                )
                cread_stats[f"{loss_config.metric_name}/cls_loss"] = cread_cls_loss
                cread_stats[f"{loss_config.metric_name}/restore_loss"] = cread_restoration_loss
            elif loss_config.loss_type == "tweedie":
                (
                    action_loss,
                    gt_clamped,
                    pred_in_original_units,
                    cont_loss_mask,
                    per_element_loss,
                ) = tweedie_loss_compute(
                    gt_raw=gt_raw,
                    pred_raw=pred_raw,
                    valid_mask=head_valid_mask,
                    negative_sample_mask=c.negative_sample_mask,
                    p=loss_config.tweedie_power,
                    norm_scale=loss_config.norm_config.norm_scale,
                    mask_negatives=loss_config.mask_negatives,
                    raw_weights=loss_weights,
                    normalizer=head_normalizer,
                )
            elif loss_config.loss_type == "binary":
                (
                    action_loss,
                    gt_clamped,
                    pred_in_original_units,
                    cont_loss_mask,
                    per_element_loss,
                ) = binary_threshold_loss_compute(
                    gt_raw=gt_raw,
                    logit=head_logits_by_index[loss_config.action_index][..., 0],
                    valid_mask=head_valid_mask,
                    negative_sample_mask=c.negative_sample_mask,
                    threshold=loss_config.binary_threshold,
                    mask_negatives=loss_config.mask_negatives,
                    raw_weights=loss_weights,
                    normalizer=head_normalizer,
                )
            else:
                (
                    action_loss,
                    gt_clamped,
                    pred_in_original_units,
                    cont_loss_mask,
                    per_element_loss,
                ) = continuous_loss_compute(
                    gt_raw=gt_raw,
                    pred_raw=pred_raw,
                    valid_mask=head_valid_mask,
                    negative_sample_mask=c.negative_sample_mask,
                    norm_scale=loss_config.norm_config.norm_scale,
                    loss_type=loss_config.loss_type,
                    mask_negatives=loss_config.mask_negatives,
                    raw_weights=loss_weights,
                    normalizer=head_normalizer,
                )
            continuous_action_loss_total += loss_config.loss_weight * action_loss
            continuous_metric_inputs[index] = {
                "gt_raw": gt_raw,
                "gt_clamped": gt_clamped,
                "pred": pred_in_original_units,
                "per_element_loss": per_element_loss,
                "loss_mask": cont_loss_mask,
            }

        stats = summarizer.get()
        if loss_normalizers is not None:
            assert stats.keys() <= {"act-l2-loss", "attn-loss", "ffn-loss"}, stats.keys()
            stats = {key: value / loss_normalizers["num_partials"] for key, value in stats.items()}
        stats["origin-loss"] = loss
        stats.update(cread_stats)
        value_heads = self.config.value_heads()
        value_loss_denominator = None
        if value_heads and self.config.value_loss_normalization == "positions":
            value_loss_denominator = normalizer("multihot")
            if value_loss_denominator is None:
                value_loss_denominator = jnp.sum(
                    multihot_loss_weights(loss_padding_mask, loss_weights)
                ).astype(jnp.float32)
            stats["value-loss-denominator"] = (
                value_loss_denominator
                if loss_normalizers is None
                else value_loss_denominator / loss_normalizers["num_partials"]
            )
        if purchase_value_features is not None:
            pvf = jnp.pad(
                purchase_value_features.astype(jnp.float32),
                (
                    (0, 0),
                    (0, loss_padding_mask.shape[1] - purchase_value_features.shape[1]),
                    (0, 0),
                ),
            )

            def history_value_sums(weights: jax.Array) -> jax.Array:
                w = weights.astype(jnp.float32)
                return jnp.stack([(pvf[..., 0] * w).sum(), (pvf[..., 5] * w).sum(), w.sum()])

            stats["_purchase-value-history-sums"] = history_value_sums(loss_padding_mask)
        value_preds: dict[str, jax.Array] = {}
        for head in value_heads:
            value_ratio, baseline, value_mask = c.value_heads[head.ema_key]
            if purchase_value_features is not None and head.slot == PURCHASE_VALUE_ACTION_INDEX:
                stats[f"_{head.metric_prefix}-history-sums"] = history_value_sums(value_mask)
            if head.slot is not None:
                pred_ratio = candidate_continuous_preds[:, :, head.slot]
                value_preds[head.ema_key] = pred_ratio
                bin_logits = head_logits_by_index.get(head.slot) if head.bins else None
            else:
                assert head.event is not None and head.bins is not None
                bin_logits = aux_logits[head.event]
                pred_ratio = head.bins.readout(bin_logits)
            value_loss, value_stats = value_head_loss_compute(
                value_ratio,
                baseline,
                value_mask,
                head.metric_prefix,
                pred_ratio=pred_ratio,
                bin_logits=bin_logits,
                bins=head.bins,
                huber_delta=head.huber_delta,
                raw_weights=loss_weights,
                normalizer=normalizer(f"value/{head.ema_key}"),
                loss_denominator=value_loss_denominator,
            )
            continuous_action_loss_total += head.loss_weight * value_loss
            for key in (
                value_sums_key(head.metric_prefix),
                value_binned_sums_key(head.metric_prefix),
            ):
                if key in value_stats:
                    stats[key] = value_stats[key]

        regularization_loss = loss + self.config.act_l2_weight * stats["act-l2-loss"]
        if self.config.per_layer_l2_weight != 0:
            regularization_loss += self.config.per_layer_l2_weight * (
                stats["ffn-loss"] + stats["attn-loss"]
            )
        regularization_loss += continuous_action_loss_total

        if self.config.page_rl.enabled:
            page_loss, page_stats = self.page_rl_loss(batch, c, *outputs[4:])
            if loss_normalizers is not None:
                page_loss = page_loss / loss_normalizers["num_partials"]
                page_stats = {
                    k: v / loss_normalizers["num_partials"] for k, v in page_stats.items()
                }
            regularization_loss += page_loss
            stats.update(page_stats)
            stats["page_rl/total_loss"] = page_loss

        embedding_norms: dict[str, tuple[jax.Array, jax.Array]] = {}
        if self.config.multimodal_embedding_type is not None and mm_emb is not None:
            if self.config.use_seqpack:
                packed_candidate_seq_len = mm_emb.shape[1] // batch["user_hashes"].shape[1]
                candidate_seq_len = self.config.candidate_seq_len
                num_devices, _, emb_dim = mm_emb.shape

                mm_emb_by_user = mm_emb.reshape(num_devices, -1, packed_candidate_seq_len, emb_dim)
                padding_by_user = loss_padding_mask.reshape(
                    loss_padding_mask.shape[0], -1, packed_candidate_seq_len
                )

                mm_emb_positives = mm_emb_by_user[:, :, :candidate_seq_len, :].reshape(
                    num_devices, -1, emb_dim
                )
                candidate_padding = padding_by_user[:, :, :candidate_seq_len].reshape(
                    loss_padding_mask.shape[0], -1
                )
            else:
                assert candidate_start_offset is not None

                mm_emb_positives = mm_emb[:, : self.config.candidate_seq_len, :]
                candidate_padding = padding_mask[
                    :,
                    candidate_start_offset : candidate_start_offset + self.config.candidate_seq_len,
                ]
            embedding_norms["mm_embedding"] = (
                jnp.sqrt(jnp.sum(mm_emb_positives.astype(jnp.float32) ** 2, axis=-1)),
                candidate_padding,
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
                embedding_norms["search_query_embedding"] = (
                    jnp.sqrt(jnp.sum(sq_emb_positives.astype(jnp.float32) ** 2, axis=-1)),
                    sq_candidate_padding,
                )

        line_item_id = batch["candidate_seq"].get(VALUE_LINE_ITEM_KEY)
        if line_item_id is not None and self.config.purchase_value_within_li_metric:
            line_item_id = cast_jax(line_item_id).astype(jnp.int32)
            line_item_id = jnp.pad(
                line_item_id, ((0, 0), (0, c.loss_padding_mask.shape[1] - line_item_id.shape[1]))
            )
        else:
            line_item_id = None
        metric_inputs = {
            "candidates": c,
            "logits": candidate_logits,
            "new_user_mask": new_user_mask,
            "no_history_mask": no_history_mask,
            "history_len": history_len,
            "continuous": continuous_metric_inputs,
            "embedding_norms": embedding_norms,
            "value_preds": value_preds,
            "value_line_item_id": line_item_id,
        }
        return regularization_loss, (stats, metric_inputs)

    @hk.transparent
    def page_rl_loss(
        self,
        batch: RecsysFeaturesBatch,
        c: CandidateInputs,
        candidate_embeddings: jax.Array,
        user_embeddings: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cand = batch["candidate_seq"]
        num_candidates = c.targets.shape[1]
        if self.config.use_seqpack:
            per_user = num_candidates // batch["user_hashes"].shape[1]
            layout = batch.get("packing_layout")
            assert layout is not None
            valid = jnp.take_along_axis(
                cast_jax(layout.padding_mask), cast_jax(layout.candidate_positions), axis=1
            )
        else:
            valid = cast_jax(cand["post_hashes"])[:, :, 0] != 0

        def rows(x: Any, fill: float | int = 0) -> jax.Array:
            x = cast_jax(x)
            if self.config.use_seqpack:
                return x.reshape(-1, per_user, *x.shape[2:])
            pad = ((0, 0), (0, num_candidates - x.shape[1])) + ((0, 0),) * (x.ndim - 2)
            return jnp.pad(x, pad, constant_values=fill)

        def label(name: str) -> jax.Array:
            return c.targets[..., action_type_map[name]]

        def labels(names: tuple[str, ...]) -> jax.Array:
            return c.targets[..., [action_type_map[n] for n in names]]

        assert c.continuous_actions is not None
        return page_rl_loss_and_stats(
            config=self.config.page_rl,
            candidate_embeddings=rows(candidate_embeddings),
            user_embeddings=user_embeddings.reshape(-1, user_embeddings.shape[-1]),
            rl_served_slot=rows(cand["rl_served_slot"], -1),
            rl_weighted_score=rows(cand["rl_weighted_score"]),
            rl_final_score=rows(cand["rl_final_score"]),
            rl_rerank_prob=rows(cand["rl_rerank_prob"]),
            rl_decode_prob=rows(cand["rl_decode_prob"]),
            rl_impressed=rows(cand["rl_impressed"]).astype(jnp.bool_),
            dwell_s=rows(
                c.continuous_actions[..., recsys_pb2.ContinuousActionName.DWELL_TIME].astype(
                    jnp.float32
                )
            ),
            not_dwelled=rows(label("ClientTweetRecapNotDwelled").astype(jnp.bool_)),
            vqv_observed=rows(label("ClientTweetVideoQualityView").astype(jnp.bool_)),
            engagement_count=rows(labels(REALIZED_POSITIVE_ACTIONS).astype(jnp.float32).sum(-1)),
            negative_any=rows(labels(REALIZED_NEGATIVE_ACTIONS).any(-1)),
            valid_mask=rows(valid, False),
            post_sids=rows(cand["post_sids"]),
            auth_hashes=rows(cast_jax(cand["auth_hashes"])[..., 0].astype(jnp.int32)),
        )

    @hk.transparent
    def metrics(
        self,
        metric_inputs: dict[str, Any],
        batch: RecsysFeaturesBatch,
        stats: dict[str, Any],
        rce_ema: dict[str, jax.Array] | None = None,
        rce_alpha: jax.Array | None = None,
        smoothing_windows: tuple[int, ...] | None = None,
        calib_ema: dict[str, jax.Array] | None = None,
    ) -> dict[str, Any]:
        c: CandidateInputs = metric_inputs["candidates"]
        mask = c.loss_padding_mask
        dpa_product_key = c.dpa_product_key[..., 0] if c.dpa_product_key is not None else None
        stats = dict(stats)
        value_heads = self.config.value_heads()
        value_sums = {
            key: stats.pop(key)
            for head in value_heads
            for key in (
                value_sums_key(head.metric_prefix),
                value_binned_sums_key(head.metric_prefix),
            )
            if key in stats
        }

        stats.update(
            safety_filter_stats(
                c.safety_label_mask,
                c.target_padding_mask,
                bits=self.config.safety_filter_bits,
                prefix="safety_filter_candidates",
            )
        )
        stats["safety_filter_active"] = jnp.float32(
            1.0 if self.config.safety_filter_mode != "off" else 0.0
        )
        stats["mact-in-app-loss-weight"] = jnp.float32(self.config.mact_in_app_loss_weight)
        stats["view-loss-weight"] = jnp.float32(self.config.view_loss_weight)

        conversion_delay_ms = None
        if self.config.conversion_delay_slices():
            raw_delays = batch["candidate_seq"].get("conversion_delay_ms")
            if raw_delays is None:
                conversion_delay_ms = jnp.full(
                    (*c.target_padding_mask.shape, len(CONVERSION_DELAY_HEAD_COLUMNS)),
                    CONVERSION_DELAY_NONE,
                    dtype=jnp.int32,
                )
            else:
                delays = cast_jax(raw_delays)
                pad_len = c.target_padding_mask.shape[1] - delays.shape[1]
                conversion_delay_ms = jnp.pad(
                    delays, ((0, 0), (0, pad_len), (0, 0)), constant_values=CONVERSION_DELAY_NONE
                )

        stats = self.compute_recsys_metrics(
            raw_targets=c.targets,
            mask=mask,
            product_surface=c.product_surface,
            negative_sample_mask=c.negative_sample_mask,
            logits=metric_inputs["logits"],
            client_app_id=c.client_app_id,
            promoted_ids=c.promoted_ids,
            new_user_mask=metric_inputs["new_user_mask"],
            line_item_objective=c.line_item_objective,
            no_history_mask=metric_inputs["no_history_mask"],
            dpa_product_key=dpa_product_key,
            sample_source=c.source_id,
            trained_candidate_mask=c.trained_heads,
            stats=stats,
            rce_ema=rce_ema,
            rce_alpha=rce_alpha,
            smoothing_windows=smoothing_windows,
            calib_ema=calib_ema,
            raw_weights=c.loss_weights,
            conversion_delay_ms=conversion_delay_ms,
            head_eligible_mask=c.eligible_heads,
        )
        stats = self.compute_timestamp_metrics(batch=batch, stats=stats)
        stats = self.compute_sid_metrics(batch=batch, stats=stats)
        stats = self.compute_engagement_count_metrics(batch=batch, stats=stats)
        stats = self.compute_author_nsfw_metrics(batch=batch, stats=stats)
        stats = self.compute_search_lexical_match_metrics(batch=batch, stats=stats)

        history_len = metric_inputs["history_len"]
        if history_len is not None:
            stats = self.compute_length_bucketed_metrics(
                raw_targets=c.targets,
                logits=metric_inputs["logits"],
                mask=mask,
                history_len=history_len,
                packed_candidate_seq_len=c.targets.shape[1] // history_len.shape[1],
                stats=stats,
                raw_weights=c.loss_weights,
            )

        continuous_masks = self._build_metric_masks(
            mask=mask,
            raw_targets=c.targets,
            negative_sample_mask=c.negative_sample_mask,
            product_surface=c.product_surface,
            new_user_mask=metric_inputs["new_user_mask"],
            no_history_mask=metric_inputs["no_history_mask"],
            dpa_product_key=dpa_product_key,
            sample_source=c.source_id,
            trained_candidate_mask=c.trained_heads,
        )
        for index, head in metric_inputs["continuous"].items():
            loss_config = self.config.continuous_action_losses[index]
            assert loss_config.metric_name is not None
            _, head_surface_mask = c.continuous_heads[index]
            for mask_suffix, variant_mask in continuous_masks.items():
                head_variant_mask = variant_mask
                if head_surface_mask is not None:
                    head_variant_mask = head_variant_mask * head_surface_mask.astype(
                        head_variant_mask.dtype
                    )
                if loss_config.mask_negatives:
                    variant_loss_mask = head_variant_mask & (~c.negative_sample_mask)
                else:
                    variant_loss_mask = head_variant_mask
                if loss_config.loss_type == "cread" and loss_config.cread_mask_sentinel_labels:
                    variant_loss_mask = (
                        variant_loss_mask
                        & (head["gt_raw"] != 0.0)
                        & (head["gt_raw"] != DWELL_TIME_DEFAULT_SENTINEL_SEC)
                    )
                pred_mask = variant_loss_mask
                num_out_of_range = None
                if loss_config.action_index == recsys_pb2.ContinuousActionName.DWELL_TIME:
                    in_range = (head["gt_raw"] >= MIN_VALID_DWELL_SEC) & (
                        head["gt_raw"] <= MAX_VALID_DWELL_SEC
                    )
                    num_out_of_range = jnp.sum(pred_mask & ~in_range)
                    head_variant_mask = head_variant_mask * in_range.astype(head_variant_mask.dtype)
                    variant_loss_mask = variant_loss_mask & in_range
                n_variant = jnp.sum(variant_loss_mask)
                variant_loss = jnp.sum(head["per_element_loss"] * variant_loss_mask) / jnp.maximum(
                    n_variant, 1.0
                )

                compute_continuous_metrics(
                    gt_raw=head["gt_raw"],
                    gt_clamped=head["gt_clamped"],
                    pred_in_original_units=head["pred"],
                    valid_mask=head_variant_mask,
                    loss_mask=variant_loss_mask,
                    pred_mask=pred_mask,
                    num_out_of_range=num_out_of_range,
                    loss=variant_loss,
                    loss_weight=loss_config.loss_weight,
                    metric_name=loss_config.metric_name,
                    mask_suffix=mask_suffix,
                    stats=stats,
                    mean_baseline=self.config.continuous_metrics_mae_mean,
                )

                if loss_config.loss_type == "binary":
                    variant_name = (
                        f"{loss_config.metric_name}_{mask_suffix}"
                        if mask_suffix
                        else loss_config.metric_name
                    )
                    stats[f"{variant_name}-rce"] = metric_rce(
                        head["pred"], head["gt_clamped"], variant_loss_mask
                    )
                    stats[f"{variant_name}-ratio-pos"] = metric_ratio_pos(
                        head["pred"], head["gt_clamped"], variant_loss_mask
                    )

            if loss_config.loss_type == "binary":
                xauc_gt = jnp.clip(
                    head["gt_raw"].astype(jnp.float32),
                    0.0,
                    loss_config.norm_config.norm_scale,
                )
            else:
                xauc_gt = head["gt_clamped"]
            xauc_mask = head["loss_mask"] & (xauc_gt > 0)
            stats[f"{loss_config.metric_name}-xauc-clicked"] = metric_xauc(
                pred=head["pred"],
                gt=xauc_gt,
                valid_mask=xauc_mask,
            )

        history_sums = {
            prefix: stats.pop(f"_{prefix}-history-sums")
            for prefix in ("purchase-value", *(head.metric_prefix for head in value_heads))
            if f"_{prefix}-history-sums" in stats
        }
        for prefix, sums in history_sums.items():
            count = jnp.maximum(sums[2], 1.0)
            stats[f"{prefix}-history-same-adv-share"] = sums[0] / count
            stats[f"{prefix}-history-user-valued-share"] = sums[1] / count
        if value_heads:
            delayed_clicked = stats.get(
                "delayed_website_clicked_num_tokens", jnp.zeros((), jnp.float32)
            )
            delayed_all = delayed_clicked + stats.get(
                "delayed_website_non_clicked_num_tokens", jnp.zeros((), jnp.float32)
            )
            smoothing_batch_size = (
                rce_alpha[0] * smoothing_windows[0]
                if rce_ema is not None and rce_alpha is not None and smoothing_windows is not None
                else None
            )
            for head in value_heads:
                value_stats = value_head_stats(value_sums, head.metric_prefix)
                stats.update(value_stats)
                slice_count = delayed_all if head.include_view_through else delayed_clicked
                stats[f"{head.metric_prefix}-ratio-valid"] = value_stats[
                    f"{head.metric_prefix}-valid-count"
                ] / jnp.maximum(slice_count, 1)
                if smoothing_batch_size is not None:
                    assert rce_ema is not None
                    smoothed_stats, value_ema = value_smoothed_stats(
                        sums=value_sums[value_sums_key(head.metric_prefix)],
                        slice_count=slice_count,
                        rce_ema=rce_ema,
                        batch_size=smoothing_batch_size,
                        smoothing_windows=head.smoothing_windows,
                        ema_key=head.ema_key,
                        metric_prefix=head.metric_prefix,
                    )
                    stats.update(smoothed_stats)
                    stats["_rce_ema"] = {**stats.get("_rce_ema", {}), **value_ema}
            line_item_id = metric_inputs.get("value_line_item_id")
            purchase_pred = metric_inputs.get("value_preds", {}).get(PURCHASE_VALUE_EMA_KEY)
            if (
                self.config.purchase_value_within_li_metric
                and smoothing_batch_size is not None
                and line_item_id is not None
                and purchase_pred is not None
            ):
                assert rce_ema is not None
                ratio, baseline, value_mask = c.value_heads[PURCHASE_VALUE_EMA_KEY]
                li_stats, li_ema = value_within_li_smoothed_stats(
                    raw_ratio=ratio,
                    baseline_mean_usd=baseline,
                    valid_mask=value_mask,
                    pred_ratio=purchase_pred,
                    line_item_id=line_item_id,
                    clicked=c.targets[..., CLICK_ACTION_INDEX] > 0,
                    rce_ema=rce_ema,
                    batch_size=smoothing_batch_size,
                    windows=self.config.within_li_windows,
                    num_buckets=self.config.purchase_value_within_li_buckets,
                    ema_key=PURCHASE_VALUE_EMA_KEY,
                    metric_prefix=value_heads[0].metric_prefix,
                )
                stats.update(li_stats)
                stats["_rce_ema"] = {**stats.get("_rce_ema", {}), **li_ema}

        embedding_norms: dict[str, tuple[jax.Array, jax.Array]] = metric_inputs["embedding_norms"]
        for name, (l2_norms, padding) in embedding_norms.items():
            num_valid = jnp.maximum(padding.sum(), 1.0)
            stats[f"{name}_l2_norm_mean"] = jnp.sum(l2_norms * padding) / num_valid
            is_zero = (l2_norms < 1e-8).astype(jnp.float32)
            stats[f"{name}_zero_frac"] = jnp.sum(is_zero * padding) / num_valid
            stats[f"{name}_l2_norm_max"] = jnp.max(jnp.where(padding, l2_norms, 0.0))

        return stats

    @hk.transparent
    def forward(
        self,
        batch: RecsysFeaturesBatch,
        recsys_embeddings: RecsysEmbeddings,
        page_decode: PageDecodeInputs | None = None,
        run_decoder: bool = True,
    ) -> tuple[jax.Array, ...]:
        decode = page_decode is not None and run_decoder
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

            outputs = self(
                input_embeddings,
                padding_mask,
                is_training=False,
                positions=positions,
                segment_ids=segment_ids,
                seqpack_layout=layout,
                product_surface=product_surface,
                return_candidate_embeddings=decode,
            )
        else:
            assert candidate_start_offset is not None

            if self.config.pad_candidates_to_128:
                input_embeddings, padding_mask, product_surface, targets, *_ = (
                    pad_to_next_128_multiple(
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
                )
            segment_ids = attention_segment_ids(
                padding_mask, candidate_start_offset, self.config.attend_padded_history
            )

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

            outputs = self(
                input_embeddings,
                padding_mask,
                is_training=False,
                positions=positions,
                segment_ids=segment_ids.astype(jnp.int32),
                candidate_start_offset=candidate_start_offset,
                product_surface=product_surface,
                return_candidate_embeddings=decode,
                purchase_value_features=self._purchase_value_features(batch),
            )

        candidate_logits = outputs[0]
        candidate_continuous_predictions = _rescale_sigmoid_heads_for_inference(
            outputs[1],
            self.config.continuous_action_losses,
            product_surface,
        )

        if page_decode is None:
            return candidate_logits, candidate_continuous_predictions
        if not decode:
            noop = page_decode_noop_outputs(self.config.page_rl, page_decode)
            return candidate_logits, candidate_continuous_predictions, *noop
        candidate_embeddings, user_embeddings = outputs[4:]
        per_user = [
            candidate_logits,
            candidate_continuous_predictions,
            candidate_embeddings,
            cast_jax(batch["candidate_seq"]["auth_hashes"])[..., 0],
            cast_jax(batch["candidate_seq"]["post_sids"]),
        ]
        if self.config.use_seqpack:
            per_user_c = candidate_embeddings.shape[1] // user_embeddings.shape[1]
            per_user = [x.reshape(-1, per_user_c, *x.shape[2:]) for x in per_user]
            user_embeddings = user_embeddings.reshape(-1, user_embeddings.shape[-1])
        logits_u, cont_u, emb_u, auth_u, sids_u = per_user
        decoded = run_page_decode(
            self.config.page_rl,
            logits_u,
            cont_u,
            emb_u,
            user_embeddings,
            page_decode,
            auth_u,
            sids_u,
        )
        return candidate_logits, candidate_continuous_predictions, *decoded
