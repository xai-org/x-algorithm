# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math
import typing
from typing import Any, Literal, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from xai_configlib import Config, configclass

from xrex.data.recsys.feature_config import (
    CategoricalFeature,
    UserCategoricalFeature,
    UserFloatFeature,
)
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.layers import get_parameter
from xrex.models.recsys_embedding import HashKeys, RecsysEmbeddings
from xrex.models.scaling import ScaleConfig


class FeaturePrepStreams(NamedTuple):
    user_tokens: jax.Array
    user_mask: jax.Array
    history_tokens: jax.Array
    history_mask: jax.Array
    candidate_tokens: jax.Array | None = None
    candidate_mask: jax.Array | None = None


POST_AGE_MAX_MINUTES = 4800


def _compute_post_age_bucket_linear(
    impr_ts_sec: jax.Array,
    post_creation_ts_sec: jax.Array,
    granularity_mins: int = 60,
) -> jax.Array:
    num_normal_buckets = POST_AGE_MAX_MINUTES // granularity_mins
    overflow_bucket = num_normal_buckets + 1
    post_age_minutes = (impr_ts_sec - post_creation_ts_sec) // 60
    bucket = (post_age_minutes // granularity_mins) + 1
    bucket = jnp.clip(bucket, 0, overflow_bucket)
    bucket = jnp.where(
        (post_age_minutes < 0) | (impr_ts_sec == 0) | (post_creation_ts_sec == 0),
        0,
        bucket,
    )
    return bucket.astype(jnp.int32)


DTYPE_BY_NAME = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}


def _cast_jax(x: typing.Any) -> jax.Array:
    return jnp.asarray(x)


def _cosine_compact_kernel(u: jax.Array) -> jax.Array:
    abs_u = jnp.abs(u)
    return jnp.where(abs_u <= 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * u)), 0.0)


def _box_kernel(u: jax.Array) -> jax.Array:
    return ((u >= -0.5) & (u < 0.5)).astype(jnp.float32)


def _triangle_kernel(u: jax.Array) -> jax.Array:
    return jnp.maximum(1.0 - jnp.abs(u), 0.0)


def _cyclic_kernel_weights(u: jax.Array, kernel: str, width: float) -> jax.Array:
    if kernel == "box":
        return _box_kernel(u)
    half_width = width / 2.0
    if kernel == "triangle":
        return _triangle_kernel(u / half_width)
    if kernel == "cosine":
        return _cosine_compact_kernel(u / half_width)
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _compute_cyclic_time_phase(
    impr_ts_sec: jax.Array,
    period_seconds: int,
) -> tuple[jax.Array, jax.Array]:
    sec_in_period = jnp.mod(impr_ts_sec.astype(jnp.int32), period_seconds).astype(jnp.float32)
    phase = sec_in_period / float(period_seconds)
    valid_mask = impr_ts_sec > 0
    phase = jnp.where(valid_mask, phase, 0.0)
    return phase, valid_mask


def _compute_cyclic_kernel_embedding(
    phase: jax.Array,
    valid_mask: jax.Array,
    embedding_table: jax.Array,
    fprop_dtype: jnp.dtype,
    kernel: str,
    num_centers: int,
    kernel_width: float,
    mean_alpha: float = 0.0,
) -> jax.Array:
    N = num_centers
    spacing = 1.0 / N
    centers = (jnp.arange(N, dtype=jnp.float32) * spacing)[None, None, :]
    diff = jnp.abs(phase[..., None] - centers)
    circ_dist = jnp.minimum(diff, 1.0 - diff)
    u = circ_dist / spacing
    weights = _cyclic_kernel_weights(u, kernel, kernel_width)
    weights_sum = jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.where(weights_sum > 0, weights / weights_sum, weights)
    valid_emb_table = embedding_table[1:, :]
    smooth_emb = jnp.einsum("bsh,hd->bsd", weights.astype(valid_emb_table.dtype), valid_emb_table)
    if mean_alpha > 0.0:
        mean_emb = jnp.mean(valid_emb_table, axis=0)
        smooth_emb = (1.0 - mean_alpha) * smooth_emb + mean_alpha * mean_emb
    unknown_emb = jnp.broadcast_to(embedding_table[0:1, :], smooth_emb.shape)
    out = jnp.where(valid_mask[..., None], smooth_emb, unknown_emb)
    return out.astype(fprop_dtype)


@configclass
class FeaturePrepConfig(Config):
    emb_size: int = 2560
    emb_table_width: int = 1024
    fprop_dtype: str = "bfloat16"
    embed_init_scale: float = 1.0
    post_age_granularity_mins: int = 60

    scale_config: ScaleConfig = ScaleConfig()

    enable_user_embedding: bool = True
    enable_ip_address: bool = True

    enable_user_country: bool = True
    enable_user_language: bool = True
    enable_user_state: bool = False
    enable_user_dma_code: bool = False
    enable_user_location: bool = True
    enable_user_gender: bool = True
    enable_user_age: bool = True
    enable_user_installed_apps: bool = True
    location_num_frequencies: int = 8
    num_countries: int = 250
    num_languages: int = 200
    num_states: int = 100
    num_dma_codes: int = 300
    num_age_brackets: int = 10
    num_installed_apps: int = 64

    enable_post_embedding: bool = True

    enable_product_surface: bool = True
    enable_post_age: bool = True
    enable_timezone: bool = True
    enable_dwell_time: bool = True
    dwell_time_norm_scale: float = 30.0
    enable_bridge_prob: bool = False
    product_surface_cardinality: int = 16
    timezone_cardinality: int = 32

    enable_time_of_day: bool = False
    time_of_day_kernel: Literal["box", "triangle", "cosine"] = "cosine"
    time_of_day_kernel_width: float = 4.0
    time_of_day_resolution: int = 48
    time_of_day_mean_alpha: float = 0.0
    time_of_day_dither: float = 0.0
    time_of_day_dither_fraction: float = 1.0

    enable_time_of_week: bool = False
    time_of_week_kernel: Literal["box", "triangle", "cosine"] = "cosine"
    time_of_week_kernel_width: float = 4.0
    time_of_week_resolution: int = 14
    time_of_week_mean_alpha: float = 0.0
    time_of_week_dither: float = 0.0
    time_of_week_dither_fraction: float = 1.0

    enable_hour_of_day: bool = True
    hour_of_day_cardinality: int = 25
    hour_of_day_dither_fraction: float = 0.0
    enable_day_of_week: bool = False
    day_of_week_cardinality: int = 8

    enable_post_sid: bool = False
    sid_embed_dim: int = 1024
    sid_num_levels: int = 6
    sid_codebook_size: int = 256
    sid_hash_level: bool = True
    sid_cross_attn: bool = True

    multimodal_embedding_dim: int = 0
    search_query_embedding_dim: int = 0

    @property
    def has_user_features(self) -> bool:
        return (
            self.enable_user_country
            or self.enable_user_language
            or self.enable_user_state
            or self.enable_user_dma_code
            or self.enable_user_location
            or self.enable_user_gender
            or self.enable_user_age
            or self.enable_user_installed_apps
        )

    @property
    def post_age_cardinality(self) -> int:
        return POST_AGE_MAX_MINUTES // self.post_age_granularity_mins + 2


def _feature_scale_spec(
    scale_config: ScaleConfig | None,
    embed_init_scale: float,
    model_width: int,
    role: Literal["embedding", "hidden_proj", "input_proj"],
    dim: int,
) -> tuple[float, float]:
    eis = embed_init_scale
    D = model_width
    if role == "embedding":
        lr = scale_config.emb_lr_multiplier(dim) if scale_config is not None else 1.0
        return eis / math.sqrt(dim), lr
    if role == "hidden_proj":
        lr = scale_config.hidden_lr_multiplier(dim) if scale_config is not None else 1.0
        return eis / math.sqrt(dim), lr
    if role == "input_proj":
        lr = scale_config.emb_lr_multiplier(D) if scale_config is not None else 1.0
        return eis / math.sqrt(dim * D), lr
    raise ValueError(f"unknown scaling role: {role!r}")


def _get_proj(
    name: str,
    in_dim: int,
    out_dim: int,
    config: FeaturePrepConfig,
    *,
    role: Literal["hidden_proj", "input_proj"],
) -> jax.Array:
    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
    _, lr_multiplier = _feature_scale_spec(
        config.scale_config, config.embed_init_scale, config.emb_size, role, in_dim
    )
    return typing.cast(
        jax.Array,
        get_parameter(
            name,
            [in_dim, out_dim],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
            pspec=P(None, None),
            lr_multiplier=lr_multiplier,
        ),
    )


def _get_emb_table(name: str, cardinality: int, dim: int, config: FeaturePrepConfig) -> jax.Array:
    _, lr_multiplier = _feature_scale_spec(
        config.scale_config, config.embed_init_scale, config.emb_size, "embedding", dim
    )
    return typing.cast(
        jax.Array,
        get_parameter(
            name,
            [cardinality, dim],
            dtype=jnp.float32,
            init=hk.initializers.VarianceScaling(config.embed_init_scale, mode="fan_out"),
            pspec=P(),
            lr_multiplier=lr_multiplier,
            rms_clip_axes=(-2, -1),
        ),
    )


def _get_learned_vector(name: str, dim: int, config: FeaturePrepConfig) -> jax.Array:
    _, lr_multiplier = _feature_scale_spec(
        config.scale_config, config.embed_init_scale, config.emb_size, "embedding", dim
    )
    return typing.cast(
        jax.Array,
        get_parameter(
            name,
            [dim],
            dtype=jnp.float32,
            init=hk.initializers.VarianceScaling(config.embed_init_scale, mode="fan_out"),
            pspec=P(),
            lr_multiplier=lr_multiplier,
        ),
    )


def _project_hash(embedding: jax.Array, name: str, config: FeaturePrepConfig) -> jax.Array:
    proj = _get_proj(name, config.emb_table_width, config.emb_size, config, role="hidden_proj")
    return jnp.dot(embedding.astype(proj.dtype), proj)


def _embed_categorical(
    values: jax.Array, cardinality: int, name: str, config: FeaturePrepConfig
) -> jax.Array:
    table = _get_emb_table(name, cardinality, config.emb_size, config)
    one_hot = jax.nn.one_hot(jnp.clip(values, 0, cardinality - 1), cardinality)
    return jnp.dot(one_hot, table)


def _dither_hour_forward(hour: jax.Array, fraction: float, num_hours: int) -> jax.Array:
    shifted = jnp.mod(hour, num_hours) + 1
    do_shift = (jax.random.uniform(hk.next_rng_key(), hour.shape) < fraction) & (hour > 0)
    return jnp.where(do_shift, shifted, hour)


def _embed_scalar_times_vector(
    scalar: jax.Array, name: str, config: FeaturePrepConfig
) -> jax.Array:
    vec = _get_learned_vector(name, config.emb_size, config)
    return scalar[..., None] * vec


def _embed_cyclic(
    impr_ts_sec: jax.Array,
    period_seconds: int,
    kernel: str,
    kernel_width: float,
    resolution: int,
    name: str,
    config: FeaturePrepConfig,
    mean_alpha: float = 0.0,
    dither: float = 0.0,
    dither_fraction: float = 1.0,
    is_training: bool = True,
) -> jax.Array:
    if is_training and dither > 0:
        spacing_sec = float(period_seconds) / resolution
        max_dither_sec = dither * spacing_sec
        noise = jax.random.uniform(
            hk.next_rng_key(), impr_ts_sec.shape, minval=0.0, maxval=max_dither_sec
        )
        if dither_fraction < 1.0:
            mask = jax.random.uniform(hk.next_rng_key(), impr_ts_sec.shape) < dither_fraction
            noise = jnp.where(mask, noise, 0.0)
        impr_ts_sec = impr_ts_sec + noise.astype(impr_ts_sec.dtype)
    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]
    D = config.emb_size
    table = _get_emb_table(name, resolution + 1, D, config)
    phase, valid_mask = _compute_cyclic_time_phase(impr_ts_sec, period_seconds)
    return _compute_cyclic_kernel_embedding(
        phase, valid_mask, table, fprop_dtype, kernel, resolution, kernel_width, mean_alpha
    )


def _embed_multi_hot(
    values: jax.Array, name: str, vocab_size: int, config: FeaturePrepConfig
) -> jax.Array:
    table = _get_emb_table(name, vocab_size, config.emb_size, config)
    return jnp.dot(values.astype(table.dtype), table)


def _embed_location(
    lat: jax.Array, lon: jax.Array, num_frequencies: int, name: str, config: FeaturePrepConfig
) -> jax.Array:
    lat_rad = jnp.pi * lat.astype(jnp.float32) / 90.0
    lon_rad = jnp.pi * lon.astype(jnp.float32) / 180.0
    freq_bands = jnp.float32(2.0) ** jnp.arange(num_frequencies, dtype=jnp.float32)
    lat_scaled = lat_rad[:, None] * freq_bands[None, :]
    lon_scaled = lon_rad[:, None] * freq_bands[None, :]
    geo_features = jnp.concatenate(
        [jnp.sin(lat_scaled), jnp.cos(lat_scaled), jnp.sin(lon_scaled), jnp.cos(lon_scaled)],
        axis=-1,
    )
    proj = _get_proj(name, 4 * num_frequencies, config.emb_size, config, role="input_proj")
    return jnp.dot(geo_features.astype(proj.dtype), proj)


@hk.transparent
def _build_user_token(
    batch: RecsysFeaturesBatch,
    embeddings: RecsysEmbeddings,
    config: FeaturePrepConfig,
    hash_keys: HashKeys,
) -> tuple[jax.Array, jax.Array]:
    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]
    D = config.emb_size
    user_embs = embeddings.user_embeddings
    B = user_embs.shape[0]
    num_user_hashes = hash_keys.num_user_hashes

    result = jnp.zeros((B, 1, D), dtype=fprop_dtype)

    for i in range(num_user_hashes):
        h = user_embs[:, i : i + 1, :]
        result = result + _project_hash(h, f"user_hash_{i}_proj", config).astype(fprop_dtype)

    if config.enable_ip_address and embeddings.user_ip_embeddings is not None:
        ip_embs = embeddings.user_ip_embeddings
        num_ip_hashes = hash_keys.num_ip_hashes
        for i in range(num_ip_hashes):
            h = ip_embs[:, i : i + 1, :]
            result = result + _project_hash(h, f"user_ip_hash_{i}_proj", config).astype(fprop_dtype)

    user_hashes = _cast_jax(batch["user_hashes"])
    user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1).astype(jnp.bool_)

    return result, user_padding_mask


@hk.transparent
def _build_user_features_token(
    batch: RecsysFeaturesBatch,
    config: FeaturePrepConfig,
) -> tuple[jax.Array, jax.Array] | None:
    if not config.has_user_features:
        return None

    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]
    D = config.emb_size
    user_cat = batch.get("user_categorical_features")
    user_float = batch.get("user_float_features")
    user_tensors = (
        user_cat,
        user_float,
        batch.get("user_installed_apps_multihot"),
    )
    B = next(_cast_jax(t).shape[0] for t in user_tensors if t is not None)

    result = jnp.zeros((B, D), dtype=fprop_dtype)

    if config.enable_user_country and user_cat is not None:
        codes = _cast_jax(user_cat)[:, UserCategoricalFeature.userCountryCode]
        result = result + _embed_categorical(
            codes, config.num_countries, "user_feat_country_emb", config
        ).astype(fprop_dtype)

    if config.enable_user_language and user_cat is not None:
        codes = _cast_jax(user_cat)[:, UserCategoricalFeature.userLanguageCode]
        result = result + _embed_categorical(
            codes, config.num_languages, "user_feat_language_emb", config
        ).astype(fprop_dtype)

    if config.enable_user_state and user_cat is not None:
        codes = _cast_jax(user_cat)[:, UserCategoricalFeature.userState]
        result = result + _embed_categorical(
            codes, config.num_states, "user_feat_state_emb", config
        ).astype(fprop_dtype)

    if config.enable_user_dma_code and user_cat is not None:
        codes = _cast_jax(user_cat)[:, UserCategoricalFeature.userDmaCode]
        result = result + _embed_categorical(
            codes, config.num_dma_codes, "user_feat_dma_code_emb", config
        ).astype(fprop_dtype)

    if config.enable_user_location and user_float is not None:
        lat = _cast_jax(user_float)[:, UserFloatFeature.userLatitude]
        lon = _cast_jax(user_float)[:, UserFloatFeature.userLongitude]
        result = result + _embed_location(
            lat, lon, config.location_num_frequencies, "user_feat_location_proj", config
        ).astype(fprop_dtype)

    if config.enable_user_installed_apps and batch.get("user_installed_apps_multihot") is not None:
        apps = _cast_jax(batch.get("user_installed_apps_multihot")).astype(jnp.float32)
        proj = _get_proj(
            "user_feat_installed_apps_proj",
            config.num_installed_apps,
            config.emb_size,
            config,
            role="input_proj",
        )
        result = result + jnp.dot(apps.astype(proj.dtype), proj).astype(fprop_dtype)

    if config.enable_user_gender and user_cat is not None and user_float is not None:
        profile_gender = _cast_jax(user_cat)[:, UserCategoricalFeature.userGender]
        result = result + _embed_categorical(
            profile_gender, 3, "user_feat_profile_gender_emb", config
        ).astype(fprop_dtype)
        inferred_gender = _cast_jax(user_cat)[:, UserCategoricalFeature.userInferredGender]
        inferred_emb = _embed_categorical(
            inferred_gender, 2, "user_feat_inferred_gender_emb", config
        )
        score = _cast_jax(user_float)[:, UserFloatFeature.userInferredGenderScore].astype(
            jnp.float32
        )
        result = result + (score[:, None] * inferred_emb).astype(fprop_dtype)

    if config.enable_user_age and user_cat is not None and user_float is not None:
        bracket = _cast_jax(user_cat)[:, UserCategoricalFeature.userAgeBracket]
        result = result + _embed_categorical(
            bracket, config.num_age_brackets, "user_feat_age_bracket_emb", config
        ).astype(fprop_dtype)
        age_years = _cast_jax(user_float)[:, UserFloatFeature.userAgeInYears].astype(jnp.float32)
        log_age = jnp.log1p(age_years)
        result = result + _embed_scalar_times_vector(
            log_age, "user_feat_age_years_vec", config
        ).astype(fprop_dtype)

    mask = jnp.ones((B, 1), dtype=jnp.bool_)
    return result[:, None, :], mask


@hk.transparent
def _add_context_features(
    result: jax.Array,
    batch_seq: dict,
    config: FeaturePrepConfig,
    prefix: str,
    is_training: bool = True,
) -> jax.Array:
    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]

    if config.enable_product_surface:
        ps = _cast_jax(batch_seq["product_surface"])
        result = result + _embed_categorical(
            ps, config.product_surface_cardinality, f"{prefix}_product_surface_emb", config
        ).astype(fprop_dtype)

    if config.enable_post_age:
        impr_ts = _cast_jax(batch_seq["impr_ts"])
        creation_ts = _cast_jax(batch_seq["post_creation_ts_sec"])
        buckets = _compute_post_age_bucket_linear(
            impr_ts, creation_ts, config.post_age_granularity_mins
        )
        result = result + _embed_categorical(
            buckets, config.post_age_cardinality, f"{prefix}_post_age_emb", config
        ).astype(fprop_dtype)

    if config.enable_timezone:
        cat_features = batch_seq.get("categorical_features")
        if cat_features is not None:
            tz = _cast_jax(cat_features)[:, :, CategoricalFeature.timezoneSeq]
            result = result + _embed_categorical(
                tz, config.timezone_cardinality, f"{prefix}_timezone_emb", config
            ).astype(fprop_dtype)

    if config.enable_time_of_day:
        impr_ts = _cast_jax(batch_seq["impr_ts"])
        result = result + _embed_cyclic(
            impr_ts,
            86_400,
            config.time_of_day_kernel,
            config.time_of_day_kernel_width,
            config.time_of_day_resolution,
            f"{prefix}_time_of_day_emb",
            config,
            mean_alpha=config.time_of_day_mean_alpha,
            dither=config.time_of_day_dither,
            dither_fraction=config.time_of_day_dither_fraction,
            is_training=is_training,
        ).astype(fprop_dtype)

    if config.enable_time_of_week:
        impr_ts = _cast_jax(batch_seq["impr_ts"])
        result = result + _embed_cyclic(
            impr_ts,
            604_800,
            config.time_of_week_kernel,
            config.time_of_week_kernel_width,
            config.time_of_week_resolution,
            f"{prefix}_time_of_week_emb",
            config,
            mean_alpha=config.time_of_week_mean_alpha,
            dither=config.time_of_week_dither,
            dither_fraction=config.time_of_week_dither_fraction,
            is_training=is_training,
        ).astype(fprop_dtype)

    if config.enable_hour_of_day:
        cat_features = batch_seq.get("categorical_features")
        if cat_features is not None:
            hour = _cast_jax(cat_features)[:, :, CategoricalFeature.localHourOfDaySeq]
            if is_training and config.hour_of_day_dither_fraction > 0.0:
                hour = _dither_hour_forward(
                    hour, config.hour_of_day_dither_fraction, config.hour_of_day_cardinality - 1
                )
            result = result + _embed_categorical(
                hour, config.hour_of_day_cardinality, f"{prefix}_hour_of_day_emb", config
            ).astype(fprop_dtype)

    if config.enable_day_of_week:
        cat_features = batch_seq.get("categorical_features")
        if cat_features is not None:
            dow = _cast_jax(cat_features)[:, :, CategoricalFeature.localDayOfWeekSeq]
            result = result + _embed_categorical(
                dow, config.day_of_week_cardinality, f"{prefix}_day_of_week_emb", config
            ).astype(fprop_dtype)

    return result


def _embed_entity_sid_scaled(
    sids: jnp.ndarray,
    target_dim: int,
    sid_embed_dim: int,
    sid_num_levels: int,
    sid_codebook_size: int,
    embed_init_scale: float,
    fprop_dtype: jnp.dtype,
    name_prefix: str,
    sid_hash_level: bool = False,
    entity_hashes: jnp.ndarray | None = None,
    sid_cross_attn: bool = False,
    scale_config: ScaleConfig | None = None,
) -> jnp.ndarray:
    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
    sids = sids.astype(jnp.int32)
    table_size = sid_codebook_size + 1

    _, table_lr = _feature_scale_spec(
        scale_config, embed_init_scale, target_dim, "embedding", sid_embed_dim
    )
    _, proj_lr = _feature_scale_spec(
        scale_config, embed_init_scale, target_dim, "hidden_proj", sid_embed_dim
    )

    num_unigram_levels = sid_num_levels + (1 if sid_hash_level else 0)
    combined_table = get_parameter(
        f"{name_prefix}_sid_combined",
        [num_unigram_levels * table_size, sid_embed_dim],
        dtype=jnp.float32,
        init=embed_init,
        pspec=P(None, None),
        lr_multiplier=table_lr,
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
        level_embs: list[jnp.ndarray] = [unigram_embs[..., i, :] for i in range(num_unigram_levels)]
        stacked = jnp.stack(level_embs, axis=-2)
        D = sid_embed_dim

        Wqkv = get_parameter(
            f"{name_prefix}_sid_xattn_qkv",
            [D, 3 * D],
            dtype=jnp.float32,
            init=embed_init,
            pspec=P(None, None),
            lr_multiplier=proj_lr,
        )
        qkv = jnp.dot(stacked, Wqkv)
        Q, K, V = jnp.split(qkv, 3, axis=-1)

        attn_weights = jnp.einsum("...id,...jd->...ij", Q, K) / math.sqrt(D)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attended = jnp.einsum("...ij,...jd->...id", attn_weights, V)

        summed = jnp.mean(attended, axis=-2) + jnp.mean(stacked, axis=-2)
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
        lr_multiplier=proj_lr,
    )
    return jnp.dot(summed.astype(proj.dtype), proj).astype(fprop_dtype)


@hk.transparent
def _add_sid_features(
    result: jax.Array,
    batch_seq: dict,
    config: FeaturePrepConfig,
) -> jax.Array:
    if not config.enable_post_sid:
        return result

    sids_in = batch_seq.get("post_sids")
    if sids_in is None:
        assert hk.running_init(), (
            "enable_post_sid=True but post_sids is missing from batch; "
            "verify the upstream kafka topic emits `semanticIdSeq`."
        )
        hashes = _cast_jax(batch_seq["post_hashes"])
        sids_jax: jax.Array = jnp.zeros(
            (hashes.shape[0], hashes.shape[1], config.sid_num_levels),
            dtype=jnp.uint16,
        )
    else:
        sids_jax = _cast_jax(sids_in)
    entity_hashes = _cast_jax(batch_seq["post_hashes"]) if config.sid_hash_level else None
    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]
    sid_emb = _embed_entity_sid_scaled(
        sids_jax,
        config.emb_size,
        config.sid_embed_dim,
        config.sid_num_levels,
        config.sid_codebook_size,
        config.embed_init_scale,
        fprop_dtype,
        "feat_prep_post",
        sid_hash_level=config.sid_hash_level,
        entity_hashes=entity_hashes,
        sid_cross_attn=config.sid_cross_attn,
        scale_config=config.scale_config,
    )
    return result + sid_emb.astype(fprop_dtype)


@hk.transparent
def _add_history_features(
    result: jax.Array,
    batch: RecsysFeaturesBatch,
    config: FeaturePrepConfig,
    output_vocab_size: int,
) -> jax.Array:
    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]

    actions = batch["history_seq"]["actions"]
    if actions is not None:
        actions_jax = _cast_jax(actions).astype(jnp.float32)
        action_mask = jnp.any(actions_jax != 0, axis=-1)
        action_emb = _embed_multi_hot(actions_jax, "hist_action_emb", output_vocab_size, config)
        action_emb = action_emb * action_mask[..., None]
        result = result + action_emb.astype(fprop_dtype)

    if config.enable_dwell_time:
        cont_actions = batch["history_seq"].get("continuous_actions")
        if cont_actions is not None:
            dwell_raw = _cast_jax(cont_actions)[:, :, 1].astype(jnp.float32)
            dwell_normalized = (
                jnp.clip(dwell_raw, 0.0, config.dwell_time_norm_scale)
                / config.dwell_time_norm_scale
            )
            result = result + _embed_scalar_times_vector(
                dwell_normalized, "hist_dwell_time_vec", config
            ).astype(fprop_dtype)

    if config.enable_bridge_prob:
        cont_actions = batch["history_seq"].get("continuous_actions")
        if cont_actions is not None:
            bridge_p = jnp.clip(_cast_jax(cont_actions)[:, :, 0].astype(jnp.float32), 0.0, 1.0)
            result = result + _embed_scalar_times_vector(
                bridge_p, "hist_bridge_prob_vec", config
            ).astype(fprop_dtype)

    return result


@hk.transparent
def _add_candidate_features(
    result: jax.Array,
    batch: RecsysFeaturesBatch,
    config: FeaturePrepConfig,
) -> jax.Array:
    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]

    if config.multimodal_embedding_dim > 0:
        mm_emb = batch["candidate_seq"].get("embedding")
        assert mm_emb is not None, (
            "candidate_seq 'embedding' must be present when multimodal_embedding_dim > 0"
        )
        mm = _cast_jax(mm_emb) if not isinstance(mm_emb, jax.Array) else mm_emb
        proj = _get_proj(
            "cand_multimodal_proj",
            config.multimodal_embedding_dim,
            config.emb_size,
            config,
            role="input_proj",
        )
        result = result + jnp.dot(mm.astype(proj.dtype), proj).astype(fprop_dtype)

    if config.search_query_embedding_dim > 0:
        sq_emb = batch["candidate_seq"].get("search_query_embeddings")
        assert sq_emb is not None, (
            "candidate_seq 'search_query_embeddings' must be present when "
            "search_query_embedding_dim > 0"
        )
        sq = _cast_jax(sq_emb)
        proj = _get_proj(
            "cand_search_query_proj",
            config.search_query_embedding_dim,
            config.emb_size,
            config,
            role="input_proj",
        )
        result = result + jnp.dot(sq.astype(proj.dtype), proj).astype(fprop_dtype)

    return result


@hk.transparent
def _build_seq_tokens(
    post_embeddings: jax.Array,
    author_embeddings: jax.Array,
    post_hashes: jax.Array,
    config: FeaturePrepConfig,
    hash_keys: HashKeys,
    prefix: str,
) -> tuple[jax.Array, jax.Array]:
    fprop_dtype = DTYPE_BY_NAME[config.fprop_dtype]
    D = config.emb_size
    num_item_hashes = hash_keys.num_item_hashes
    num_author_hashes = hash_keys.num_author_hashes
    B = author_embeddings.shape[0]
    S = author_embeddings.shape[1] // num_author_hashes

    result = jnp.zeros((B, S, D), dtype=fprop_dtype)

    if config.enable_post_embedding:
        post_reshaped = post_embeddings.reshape(B, S, num_item_hashes, -1)
        for i in range(num_item_hashes):
            h = post_reshaped[:, :, i, :]
            result = result + _project_hash(h, f"{prefix}_post_hash_{i}_proj", config).astype(
                fprop_dtype
            )

    author_reshaped = author_embeddings.reshape(B, S, num_author_hashes, -1)
    for i in range(num_author_hashes):
        h = author_reshaped[:, :, i, :]
        result = result + _project_hash(h, f"{prefix}_author_hash_{i}_proj", config).astype(
            fprop_dtype
        )

    padding_mask = (post_hashes[:, :, 0] != 0).reshape(B, S).astype(jnp.bool_)

    return result, padding_mask


def _flatten_user_batch_for_seqpack(
    batch: RecsysFeaturesBatch,
    recsys_embeddings: RecsysEmbeddings,
) -> tuple[RecsysFeaturesBatch, RecsysEmbeddings, int, int]:
    user_hashes = _cast_jax(batch["user_hashes"])
    num_devices, bs_per_device = user_hashes.shape[0], user_hashes.shape[1]
    total_batch = num_devices * bs_per_device

    def _flat_user(field: str) -> jax.Array | None:
        raw = batch.get(field)
        if raw is None:
            return None
        arr = _cast_jax(raw)
        return arr.reshape(total_batch, *arr.shape[2:])

    flat_batch = typing.cast(
        RecsysFeaturesBatch,
        {
            **batch,
            "user_hashes": user_hashes.reshape(total_batch, -1),
            **{
                k: v
                for k, v in {
                    "user_categorical_features": _flat_user("user_categorical_features"),
                    "user_float_features": _flat_user("user_float_features"),
                    "user_bool_features": _flat_user("user_bool_features"),
                    "user_int64_features": _flat_user("user_int64_features"),
                    "user_installed_apps_multihot": _flat_user("user_installed_apps_multihot"),
                }.items()
                if v is not None
            },
        },
    )

    flat_user_emb = (
        recsys_embeddings.user_embeddings.reshape(
            total_batch,
            recsys_embeddings.user_embeddings.shape[-2],
            recsys_embeddings.user_embeddings.shape[-1],
        )
        if recsys_embeddings.user_embeddings is not None
        else None
    )
    flat_user_ip_emb = (
        recsys_embeddings.user_ip_embeddings.reshape(
            total_batch,
            recsys_embeddings.user_ip_embeddings.shape[-2],
            recsys_embeddings.user_ip_embeddings.shape[-1],
        )
        if recsys_embeddings.user_ip_embeddings is not None
        else None
    )

    def _reshape_seq(emb: jax.Array | None) -> jax.Array | None:
        return emb.reshape(num_devices, -1, emb.shape[-1]) if emb is not None else None

    flat_embeddings = RecsysEmbeddings(
        user_embeddings=flat_user_emb,
        user_ip_embeddings=flat_user_ip_emb,
        history_post_embeddings=_reshape_seq(recsys_embeddings.history_post_embeddings),
        history_author_embeddings=_reshape_seq(recsys_embeddings.history_author_embeddings),
        candidate_post_embeddings=_reshape_seq(recsys_embeddings.candidate_post_embeddings),
        candidate_author_embeddings=_reshape_seq(recsys_embeddings.candidate_author_embeddings),
    )
    return flat_batch, flat_embeddings, num_devices, bs_per_device


@hk.transparent
def build_feature_prep_streams(
    batch: RecsysFeaturesBatch,
    recsys_embeddings: RecsysEmbeddings,
    config: FeaturePrepConfig,
    hash_keys: HashKeys,
    input_scale: float,
    output_vocab_size: int = 64,
    is_training: bool = True,
    include_candidates: bool = True,
) -> FeaturePrepStreams:
    user_tokens: jax.Array | None = None
    user_mask: jax.Array | None = None
    if config.enable_user_embedding:
        user_tokens, user_mask = _build_user_token(batch, recsys_embeddings, config, hash_keys)

    user_features_result = _build_user_features_token(batch, config)
    if user_features_result is not None:
        uf_token, uf_mask = user_features_result
        if user_tokens is not None:
            user_tokens = jnp.concatenate([user_tokens, uf_token], axis=1)
            user_mask = jnp.concatenate([user_mask, uf_mask], axis=1)
        else:
            user_tokens, user_mask = uf_token, uf_mask

    assert user_tokens is not None and user_mask is not None, (
        "feature_prep requires at least one user prefix token "
        "(enable_user_embedding and/or user features)"
    )

    history_tokens, history_mask = _build_seq_tokens(
        recsys_embeddings.history_post_embeddings,
        recsys_embeddings.history_author_embeddings,
        _cast_jax(batch["history_seq"]["post_hashes"]),
        config,
        hash_keys,
        prefix="hist",
    )
    history_tokens = _add_context_features(
        history_tokens, batch["history_seq"], config, prefix="hist", is_training=is_training
    )
    history_tokens = _add_sid_features(history_tokens, batch["history_seq"], config)
    history_tokens = _add_history_features(history_tokens, batch, config, output_vocab_size)

    candidate_tokens: jax.Array | None = None
    candidate_mask: jax.Array | None = None
    if include_candidates:
        candidate_tokens, candidate_mask = _build_seq_tokens(
            recsys_embeddings.candidate_post_embeddings,
            recsys_embeddings.candidate_author_embeddings,
            _cast_jax(batch["candidate_seq"]["post_hashes"]),
            config,
            hash_keys,
            prefix="cand",
        )
        candidate_tokens = _add_context_features(
            candidate_tokens,
            batch["candidate_seq"],
            config,
            prefix="cand",
            is_training=is_training,
        )
        candidate_tokens = _add_sid_features(candidate_tokens, batch["candidate_seq"], config)
        candidate_tokens = _add_candidate_features(candidate_tokens, batch, config)
        candidate_tokens = candidate_tokens * input_scale

    user_tokens = user_tokens * input_scale
    history_tokens = history_tokens * input_scale

    return FeaturePrepStreams(
        user_tokens=user_tokens,
        user_mask=user_mask,
        history_tokens=history_tokens,
        history_mask=history_mask,
        candidate_tokens=candidate_tokens,
        candidate_mask=candidate_mask,
    )


def _assemble_unpacked(streams: FeaturePrepStreams) -> tuple[jax.Array, jax.Array, int]:
    sequence_tokens = [streams.user_tokens, streams.history_tokens]
    sequence_masks = [streams.user_mask, streams.history_mask]
    if streams.candidate_tokens is not None and streams.candidate_mask is not None:
        sequence_tokens.append(streams.candidate_tokens)
        sequence_masks.append(streams.candidate_mask)
    tokens = jnp.concatenate(sequence_tokens, axis=1)
    padding_mask = jnp.concatenate(sequence_masks, axis=1)
    candidate_start_offset = streams.user_mask.shape[1] + streams.history_mask.shape[1]
    return tokens, padding_mask, candidate_start_offset


def _assemble_packed(
    streams: FeaturePrepStreams,
    layout: Any,
    num_devices: int,
    bs_per_device: int,
    config: FeaturePrepConfig,
) -> tuple[jax.Array, jax.Array, None]:
    D = config.emb_size
    padding_mask = _cast_jax(layout.padding_mask)
    seq_starts = _cast_jax(layout.cu_seqlens[:, :-1])
    device_idx = jnp.arange(num_devices, dtype=jnp.int32)[:, None]

    tokens = jnp.zeros((*padding_mask.shape, D), dtype=streams.user_tokens.dtype)

    U = streams.user_tokens.shape[1]
    user_per_dev = streams.user_tokens.reshape(num_devices, bs_per_device, U, D)
    for u in range(U):
        tokens = tokens.at[device_idx, seq_starts + u].set(user_per_dev[:, :, u, :])

    tokens = tokens.at[device_idx, _cast_jax(layout.history_positions)].add(
        jnp.where(streams.history_mask[:, :, None], streams.history_tokens, 0)
    )
    if streams.candidate_tokens is not None and streams.candidate_mask is not None:
        tokens = tokens.at[device_idx, _cast_jax(layout.candidate_positions)].add(
            jnp.where(streams.candidate_mask[:, :, None], streams.candidate_tokens, 0)
        )
    return tokens, padding_mask, None


@hk.transparent
def build_feature_prep_inputs(
    batch: RecsysFeaturesBatch,
    recsys_embeddings: RecsysEmbeddings,
    config: FeaturePrepConfig,
    hash_keys: HashKeys,
    input_scale: float,
    output_vocab_size: int = 64,
    is_training: bool = True,
    include_candidates: bool = True,
) -> tuple[jax.Array, jax.Array, int | None]:
    layout = batch.get("packing_layout")

    if layout is None:
        streams = build_feature_prep_streams(
            batch,
            recsys_embeddings,
            config,
            hash_keys,
            input_scale,
            output_vocab_size,
            is_training=is_training,
            include_candidates=include_candidates,
        )
        return _assemble_unpacked(streams)

    flat_batch, flat_embeddings, num_devices, bs_per_device = _flatten_user_batch_for_seqpack(
        batch, recsys_embeddings
    )
    streams = build_feature_prep_streams(
        flat_batch,
        flat_embeddings,
        config,
        hash_keys,
        input_scale,
        output_vocab_size,
        is_training=is_training,
        include_candidates=include_candidates,
    )
    return _assemble_packed(streams, layout, num_devices, bs_per_device, config)
