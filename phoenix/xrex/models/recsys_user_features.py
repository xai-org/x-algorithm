# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from xai_configlib import Config, configclass

from xrex.data.recsys.feature_config import UserCategoricalFeature, UserFloatFeature
from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.models.layers import Linear, get_parameter
from xrex.models.sharding_context import ShardingContext


def cast_jax(x):
    return jnp.asarray(x)


@configclass
class UserFeaturesConfig(Config):
    enable_user_country_feature: bool = False
    enable_user_language_feature: bool = False
    enable_user_state_feature: bool = False
    enable_user_dma_code_feature: bool = False
    enable_user_location_feature: bool = False
    enable_user_installed_apps: bool = False
    enable_user_gender_feature: bool = False
    enable_user_age_feature: bool = False

    num_countries: int = 250
    country_emb_dim: int = 128
    num_languages: int = 200
    language_emb_dim: int = 64
    num_states: int = 100
    state_emb_dim: int = 16
    num_dma_codes: int = 300
    dma_code_emb_dim: int = 16
    location_emb_dim: int = 64
    location_num_frequencies: int = 8
    num_installed_apps: int = 64
    installed_apps_emb_dim: int = 128
    gender_score_proj_dim: int = 16
    num_age_brackets: int = 10
    age_bracket_emb_dim: int = 8
    age_mlp_hidden_dim: int = 8
    age_mlp_output_dim: int = 16

    user_features_concat_dim: int = 2048
    user_features_concat_pad: bool = True
    user_features_mlp: bool = False

    @property
    def has_user_features(self) -> bool:
        return (
            self.enable_user_country_feature
            or self.enable_user_language_feature
            or self.enable_user_state_feature
            or self.enable_user_dma_code_feature
            or self.enable_user_location_feature
            or self.enable_user_installed_apps
            or self.enable_user_gender_feature
            or self.enable_user_age_feature
        )


def build_user_feature_parts(
    recsys_features_batch: RecsysFeaturesBatch,
    config: UserFeaturesConfig,
    sharding_context: ShardingContext,
) -> list[jax.Array]:
    parts: list[jax.Array] = []

    if config.enable_user_country_feature:
        country_codes = recsys_features_batch["user_categorical_features"][
            :, UserCategoricalFeature.userCountryCode
        ]
        country_emb_table = get_parameter(
            "country_embedding_table",
            shape=[config.num_countries, config.country_emb_dim],
            dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=0.02),
            pspec=P(None, None),
        )
        parts.append(country_emb_table[cast_jax(country_codes)])

    if config.enable_user_language_feature:
        language_codes = recsys_features_batch["user_categorical_features"][
            :, UserCategoricalFeature.userLanguageCode
        ]
        language_emb_table = get_parameter(
            "language_embedding_table",
            shape=[config.num_languages, config.language_emb_dim],
            dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=0.02),
            pspec=P(None, None),
        )
        parts.append(language_emb_table[cast_jax(language_codes)])

    if config.enable_user_state_feature:
        state_codes = recsys_features_batch["user_categorical_features"][
            :, UserCategoricalFeature.userState
        ]
        state_emb_table = get_parameter(
            "state_embedding_table",
            shape=[config.num_states, config.state_emb_dim],
            dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=0.02),
            pspec=P(None, None),
        )
        parts.append(state_emb_table[cast_jax(state_codes)])

    if config.enable_user_dma_code_feature:
        dma_codes = recsys_features_batch["user_categorical_features"][
            :, UserCategoricalFeature.userDmaCode
        ]
        dma_code_emb_table = get_parameter(
            "dma_code_embedding_table",
            shape=[config.num_dma_codes, config.dma_code_emb_dim],
            dtype=jnp.float32,
            init=hk.initializers.TruncatedNormal(stddev=0.02),
            pspec=P(None, None),
        )
        parts.append(dma_code_emb_table[cast_jax(dma_codes)])

    if config.enable_user_location_feature:
        parts.extend(
            build_location_features(
                recsys_features_batch,
                config.location_emb_dim,
                sharding_context,
                num_frequencies=config.location_num_frequencies,
            )
        )

    if config.enable_user_installed_apps:
        apps_float = cast_jax(recsys_features_batch["user_installed_apps_multihot"]).astype(
            jnp.float32
        )
        apps_proj = Linear(
            config.installed_apps_emb_dim,
            w_init=hk.initializers.VarianceScaling(1.0),
            with_bias=False,
            pspec=P(None, None),
            sharding_context=sharding_context,
            name="installed_apps_feature_proj",
        )
        parts.append(apps_proj(apps_float))

    if config.enable_user_gender_feature:
        parts.extend(build_gender_features(recsys_features_batch, config, sharding_context))

    if config.enable_user_age_feature:
        parts.extend(build_age_features(recsys_features_batch, config, sharding_context))

    return parts


def build_location_features(
    recsys_features_batch: RecsysFeaturesBatch,
    location_emb_dim: int,
    sharding_context: ShardingContext,
    num_frequencies: int = 8,
) -> list[jax.Array]:
    user_lat = cast_jax(
        recsys_features_batch["user_float_features"][:, UserFloatFeature.userLatitude]
    ).astype(jnp.float32)
    user_lon = cast_jax(
        recsys_features_batch["user_float_features"][:, UserFloatFeature.userLongitude]
    ).astype(jnp.float32)

    lat_rad = jnp.pi * user_lat / 90.0
    lon_rad = jnp.pi * user_lon / 180.0

    freq_bands = jnp.float32(2.0) ** jnp.arange(num_frequencies, dtype=jnp.float32)
    lat_scaled = lat_rad[:, None] * freq_bands[None, :]
    lon_scaled = lon_rad[:, None] * freq_bands[None, :]

    geo_features = jnp.concatenate(
        [jnp.sin(lat_scaled), jnp.cos(lat_scaled), jnp.sin(lon_scaled), jnp.cos(lon_scaled)],
        axis=-1,
    )

    geo_hidden = Linear(
        location_emb_dim,
        w_init=hk.initializers.VarianceScaling(1.0),
        with_bias=True,
        pspec=P(None, None),
        sharding_context=sharding_context,
        name="user_geo_hidden",
    )(geo_features)
    geo_out = Linear(
        location_emb_dim,
        w_init=hk.initializers.VarianceScaling(1.0),
        with_bias=False,
        pspec=P(None, None),
        sharding_context=sharding_context,
        name="user_geo_proj",
    )(jax.nn.gelu(geo_hidden))

    return [geo_out]


def build_gender_features(
    recsys_features_batch: RecsysFeaturesBatch,
    config: UserFeaturesConfig,
    sharding_context: ShardingContext,
) -> list[jax.Array]:
    user_gender = cast_jax(
        recsys_features_batch["user_categorical_features"][:, UserCategoricalFeature.userGender]
    ).astype(jnp.float32)[:, None]

    inferred_gender = cast_jax(
        recsys_features_batch["user_categorical_features"][
            :, UserCategoricalFeature.userInferredGender
        ]
    ).astype(jnp.float32)[:, None]

    gender_score = cast_jax(
        recsys_features_batch["user_float_features"][:, UserFloatFeature.userInferredGenderScore]
    ).astype(jnp.float32)[:, None]

    gender_score_proj = Linear(
        config.gender_score_proj_dim,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        with_bias=False,
        pspec=P(None, None),
        sharding_context=sharding_context,
        name="user_gender_score_proj",
    )

    return [user_gender, inferred_gender, gender_score_proj(gender_score)]


def build_age_features(
    recsys_features_batch: RecsysFeaturesBatch,
    config: UserFeaturesConfig,
    sharding_context: ShardingContext,
) -> list[jax.Array]:
    age_bracket = recsys_features_batch["user_categorical_features"][
        :, UserCategoricalFeature.userAgeBracket
    ]
    age_bracket_emb_table = get_parameter(
        "age_bracket_embedding_table",
        shape=[config.num_age_brackets, config.age_bracket_emb_dim],
        dtype=jnp.float32,
        init=hk.initializers.TruncatedNormal(stddev=0.02),
        pspec=P(None, None),
    )
    age_bracket_emb = age_bracket_emb_table[cast_jax(age_bracket)]

    age_years = cast_jax(
        recsys_features_batch["user_float_features"][:, UserFloatFeature.userAgeInYears]
    ).astype(jnp.float32)
    log_age = jnp.log1p(age_years)[:, None]

    age_hidden = Linear(
        config.age_mlp_hidden_dim,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        with_bias=False,
        pspec=P(None, None),
        sharding_context=sharding_context,
        name="user_age_hidden",
    )
    age_proj = Linear(
        config.age_mlp_output_dim,
        w_init=hk.initializers.TruncatedNormal(stddev=0.02),
        with_bias=False,
        pspec=P(None, None),
        sharding_context=sharding_context,
        name="user_age_proj",
    )
    age_repr = age_proj(jax.nn.gelu(age_hidden(log_age)))

    return [age_bracket_emb, age_repr]


def build_user_features_token(
    feature_parts: list[jax.Array],
    concat_dim: int,
    emb_table_width: int,
    fprop_dtype: jnp.dtype,
    sharding_context: ShardingContext,
    use_mlp: bool = True,
    pad: bool = True,
) -> jax.Array:
    concat_features = jnp.concatenate(feature_parts, axis=-1)

    if pad:
        pad_width = concat_dim - concat_features.shape[-1]
        if pad_width > 0:
            concat_features = jnp.pad(concat_features, ((0, 0), (0, pad_width)))

    if use_mlp:
        user_features_hidden = Linear(
            concat_dim,
            w_init=hk.initializers.VarianceScaling(1.0),
            with_bias=False,
            pspec=P(None, None),
            sharding_context=sharding_context,
            name="user_features_hidden",
        )
        user_features_proj = Linear(
            emb_table_width,
            w_init=hk.initializers.VarianceScaling(1.0),
            with_bias=False,
            pspec=P(None, None),
            sharding_context=sharding_context,
            name="user_features_proj",
        )
        token = user_features_proj(jax.nn.gelu(user_features_hidden(concat_features)))
    else:
        user_features_proj = Linear(
            emb_table_width,
            w_init=hk.initializers.VarianceScaling(1.0),
            with_bias=False,
            pspec=P(None, None),
            sharding_context=sharding_context,
            name="user_features_proj",
        )
        token = user_features_proj(concat_features)

    return token[:, None, :].astype(fprop_dtype)
