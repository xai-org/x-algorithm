// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::model_config::ModelConfig;
use arrow::array::{Array, AsArray, BooleanArray, Float32Array, Int32Array, Int64Array};
use arrow::ipc::reader::StreamReader;
use half::f16;
use lazy_static::lazy_static;
use prometheus::{IntCounterVec, register_int_counter_vec};
use std::io::Cursor;
use std::time::{SystemTime, UNIX_EPOCH};
use xai_recsys_proto as pb;

lazy_static! {
    static ref USER_FEATURE_FILL_TOTAL: IntCounterVec = register_int_counter_vec!(
        "recsys_user_feature_fill_total",
        "User feature fill rate by feature name and status.",
        &["feature", "status"]
    )
    .unwrap();

    static ref SID_COVERAGE_TOTAL: IntCounterVec = register_int_counter_vec!(
        "recsys_sid_coverage_total",
        "Per-post SID coverage at inference by sequence (history/candidate) and status (present/missing).",
        &["sequence", "status"]
    )
    .unwrap();
}

fn record_sid_coverage(sequence: &str, present: u64, count: u64) {
    SID_COVERAGE_TOTAL
        .with_label_values(&[sequence, "present"])
        .inc_by(present);
    SID_COVERAGE_TOTAL
        .with_label_values(&[sequence, "missing"])
        .inc_by(count.saturating_sub(present));
}

use crate::feature_config::bool_feature::{
    IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ, IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ_COLUMN,
    IS_AUTHOR_FOLLOWING_VIEWER_SEQ, IS_AUTHOR_FOLLOWING_VIEWER_SEQ_COLUMN, IS_STALE_POST14D,
};
use crate::feature_config::categorical_feature::{
    AUTHOR_IS_NSFW_SEQ, LOCAL_DAY_OF_WEEK_SEQ, LOCAL_HOUR_OF_DAY_SEQ, PRODUCT_SURFACE_SEQ,
    PRODUCT_SURFACE_SEQ_COLUMN, TIMEZONE_SEQ,
};
#[cfg(recsys_ads_dpa)]
use crate::feature_config::constants::{
    ADS_PRODUCT_KEY_HASH_BIAS, ADS_PRODUCT_KEY_HASH_BIAS_2, ADS_PRODUCT_KEY_HASH_MODULUS,
    ADS_PRODUCT_KEY_HASH_SCALE, ADS_PRODUCT_KEY_HASH_SCALE_2, ADS_PRODUCT_KEY_TABLE_SIZE,
    STALE_POST_14D_TTL_SEC,
};
use crate::feature_config::int64_feature::{
    FAV_COUNT_SEQ, FAV_COUNT_SEQ_COLUMN, QUOTE_COUNT_SEQ, QUOTE_COUNT_SEQ_COLUMN, REPLY_COUNT_SEQ,
    REPLY_COUNT_SEQ_COLUMN, REPOST_COUNT_SEQ, REPOST_COUNT_SEQ_COLUMN, VIEW_COUNT_SEQ,
    VIEW_COUNT_SEQ_COLUMN,
};
#[cfg(recsys_ads_dpa)]
use crate::feature_config::int64_feature::{FIRST_DPA_PRODUCT_KEY, FIRST_DPA_PRODUCT_KEY_HASH2};

pub const PRODUCT_SURFACE_CATEGORICAL_IDX: usize = PRODUCT_SURFACE_SEQ;
const TIMEZONE_IDX: usize = TIMEZONE_SEQ;
const LOCAL_HOUR_IDX: usize = LOCAL_HOUR_OF_DAY_SEQ;
const LOCAL_DOW_IDX: usize = LOCAL_DAY_OF_WEEK_SEQ;
pub const AUTHOR_IS_NSFW_CATEGORICAL_IDX: usize = AUTHOR_IS_NSFW_SEQ;

const AUTHOR_NSFW_BIT: u64 = 2;

fn stamp_engagement_counts(
    dest: &mut [i64],
    num_features: usize,
    entry_idx: usize,
    fav: u64,
    reply: u64,
    retweet: u64,
    quote: u64,
    view: u64,
) {
    const INT32_MAX: u64 = i32::MAX as u64;
    let base = entry_idx * num_features;
    if num_features > FAV_COUNT_SEQ {
        dest[base + FAV_COUNT_SEQ] = fav.min(INT32_MAX) as i64;
    }
    if num_features > REPLY_COUNT_SEQ {
        dest[base + REPLY_COUNT_SEQ] = reply.min(INT32_MAX) as i64;
    }
    if num_features > REPOST_COUNT_SEQ {
        dest[base + REPOST_COUNT_SEQ] = retweet.min(INT32_MAX) as i64;
    }
    if num_features > QUOTE_COUNT_SEQ {
        dest[base + QUOTE_COUNT_SEQ] = quote.min(INT32_MAX) as i64;
    }
    if num_features > VIEW_COUNT_SEQ {
        dest[base + VIEW_COUNT_SEQ] = view.min(INT32_MAX) as i64;
    }
}

#[cfg(recsys_ads_dpa)]
fn hash_dpa_product_key_with(raw_key: i64, scale: i64, bias: i64) -> i64 {
    if raw_key == 0 {
        return 0;
    }
    raw_key
        .wrapping_mul(scale)
        .wrapping_add(bias)
        .rem_euclid(ADS_PRODUCT_KEY_HASH_MODULUS)
        .rem_euclid(ADS_PRODUCT_KEY_TABLE_SIZE - 1)
        + 1
}

#[cfg(recsys_ads_dpa)]
pub fn hash_dpa_product_key(raw_key: i64) -> i64 {
    hash_dpa_product_key_with(
        raw_key,
        ADS_PRODUCT_KEY_HASH_SCALE,
        ADS_PRODUCT_KEY_HASH_BIAS,
    )
}

#[cfg(recsys_ads_dpa)]
pub fn hash_dpa_product_key_2(raw_key: i64) -> i64 {
    hash_dpa_product_key_with(
        raw_key,
        ADS_PRODUCT_KEY_HASH_SCALE_2,
        ADS_PRODUCT_KEY_HASH_BIAS_2,
    )
}

pub fn stamp_i32_as_categorical(
    source: &[i32],
    dest: &mut [i16],
    num_features: usize,
    feature_idx: usize,
) {
    if num_features > feature_idx {
        for (i, &val) in source.iter().enumerate() {
            dest[i * num_features + feature_idx] = val as i16;
        }
    }
}

pub fn stamp_bool_seq(source: &[bool], dest: &mut [bool], num_features: usize, feature_idx: usize) {
    if num_features > feature_idx {
        for (i, &val) in source.iter().enumerate() {
            dest[i * num_features + feature_idx] = val;
        }
    }
}

pub fn stamp_local_time_features(
    impr_ts_sec: &[i32],
    tz_enums: &[i16],
    dest: &mut [i16],
    num_features: usize,
) {
    for (i, (&ts, &tz)) in impr_ts_sec.iter().zip(tz_enums.iter()).enumerate() {
        if num_features > TIMEZONE_IDX {
            dest[i * num_features + TIMEZONE_IDX] = tz;
        }
        if ts <= 0 {
            continue;
        }
        let (hour, dow) = compute_local_time(ts as i64, tz);
        if num_features > LOCAL_HOUR_IDX {
            dest[i * num_features + LOCAL_HOUR_IDX] = hour;
        }
        if num_features > LOCAL_DOW_IDX {
            dest[i * num_features + LOCAL_DOW_IDX] = dow;
        }
    }
}

const TWITTER_EPOCH_MS: i64 = 1288834974657;

fn snowflake_to_creation_ts_sec(tweet_id: i64) -> i32 {
    if tweet_id == 0 {
        return 0;
    }
    let created_ms = (tweet_id >> 22) + TWITTER_EPOCH_MS;
    (created_ms / 1000) as i32
}

fn now_epoch_sec() -> i32 {
    static FIXED: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    let fixed = FIXED.get_or_init(|| {
        std::env::var("RECSYS_FIXED_NOW_SEC")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
    });
    if let Some(ts) = fixed {
        return *ts;
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i32)
        .unwrap_or(0)
}

const TZ_ENUM_TO_UTC_OFFSET_SECS: [i64; 32] = [
    0, -18000, -21600, -28800, -25200, -10800, 0, 3600, 3600, 3600, 3600, 3600, 10800, 10800,
    32400, 19800, 28800, 32400, 25200, 25200, 14400, 10800, 28800, 28800, 39600, -36000, -25200,
    -18000, -21600, -10800, 3600, 0,
];

pub fn compute_local_time(utc_epoch_secs: i64, tz_enum: i16) -> (i16, i16) {
    let offset = if tz_enum > 0 && (tz_enum as usize) < TZ_ENUM_TO_UTC_OFFSET_SECS.len() {
        TZ_ENUM_TO_UTC_OFFSET_SECS[tz_enum as usize]
    } else {
        0
    };
    let local_secs = utc_epoch_secs + offset;
    let sec_of_day = local_secs.rem_euclid(86400);
    let hour = (sec_of_day / 3600).clamp(0, 23) as i16 + 1;
    let local_days = local_secs.div_euclid(86400);
    let dow = ((local_days + 3).rem_euclid(7)) as i16 + 1;
    (hour, dow)
}

pub fn parse_ipv4_to_u32(ip: &str) -> u32 {
    let mut result: u32 = 0;
    let mut count: usize = 0;
    for part in ip.split('.') {
        if count >= 4 {
            return 0;
        }
        match part.parse::<u8>() {
            Ok(octet) => result |= (octet as u32) << (24 - 8 * count),
            Err(_) => return 0,
        }
        count += 1;
    }
    if count != 4 { 0 } else { result }
}

pub fn compute_user_ip_hashes(model_config: &ModelConfig, ip_id: i64) -> Vec<i32> {
    let num_ip_hashes = model_config.hash_table.num_ip_hashes();
    let mut user_ip_hashes = vec![0i32; num_ip_hashes];
    for (k, slot) in user_ip_hashes.iter_mut().enumerate().take(num_ip_hashes) {
        *slot = model_config.hash_table.hash_ip_id(ip_id, k);
    }
    user_ip_hashes
}

fn encode_client_id(app_id: i64) -> i32 {
    match app_id {
        129032 => 1,
        191841 => 2,
        258901 => 3,
        557701 => 4,
        1082764 => 5,
        3033300 => 6,
        14191373 => 7,
        17698388 => 8,
        _ => 0,
    }
}

fn encode_country_code_crc32(cc: &str) -> i32 {
    if cc.is_empty() {
        return 0;
    }
    (crc32fast::hash(cc.as_bytes()) % 256) as i32
}

pub struct UserFeatures {
    pub categorical_features: Vec<i16>,
    pub bool_features: Vec<bool>,
    pub float_features: Vec<f32>,
    pub int64_features: Vec<i64>,
    pub installed_apps: Vec<bool>,
}

pub struct InputBuffer {
    pub user_hashes: Vec<i32>,
    pub user_ip_hashes: Vec<i32>,
    pub history_post_hashes: Vec<i32>,
    pub history_auth_hashes: Vec<i32>,
    pub history_product_surfaces: Vec<i32>,
    pub history_actions: Vec<bool>,
    pub history_continuous_actions: Vec<f32>,
    pub candidate_post_hashes: Vec<i32>,
    pub candidate_auth_hashes: Vec<i32>,
    pub candidate_product_surfaces: Vec<i32>,
    pub candidate_embeddings: Vec<f16>,
    pub candidate_search_query_embeddings: Vec<f32>,
    pub user_categorical_features: Vec<i16>,
    pub user_bool_features: Vec<bool>,
    pub user_float_features: Vec<f32>,
    pub user_int64_features: Vec<i64>,
    pub user_installed_apps: Vec<bool>,
    pub history_categorical_features: Vec<i16>,
    pub history_bool_features: Vec<bool>,
    pub history_float_features: Vec<f32>,
    pub history_int64_features: Vec<i64>,
    pub candidate_categorical_features: Vec<i16>,
    pub candidate_bool_features: Vec<bool>,
    pub candidate_float_features: Vec<f32>,
    pub candidate_int64_features: Vec<i64>,
    pub history_impr_ts: Vec<i32>,
    pub history_post_creation_ts_sec: Vec<i32>,
    pub candidate_impr_ts: Vec<i32>,
    pub candidate_post_creation_ts_sec: Vec<i32>,

    pub candidate_line_item_ids: Vec<i64>,
    pub candidate_campaign_ids: Vec<i64>,
    pub candidate_funding_instrument_ids: Vec<i64>,
    pub user_client_id_idx: Vec<i32>,
    pub user_country_code_idx: Vec<i32>,
    pub user_gender_idx: Vec<i32>,
    pub user_age_bucket_idx: Vec<i32>,
    pub user_age_in_years_idx: Vec<i32>,
    pub user_inferred_gender_idx: Vec<i32>,
    pub user_inferred_gender_score: Vec<f32>,
    pub candidate_conversion_dense_features: Vec<f32>,
    pub user_conversion_history_hashes: Vec<i32>,
    pub candidate_account_hashes: Vec<i32>,
    pub history_post_ids: Vec<i64>,
    pub history_semantic_ids: Vec<u16>,
    pub candidate_semantic_ids: Vec<u16>,
    pub num_history: usize,
}

struct CandidateData {
    post_hashes: Vec<i32>,
    auth_hashes: Vec<i32>,
    product_surfaces: Vec<i32>,
    embeddings: Vec<f16>,
    search_query_embeddings: Vec<f32>,
    categorical_features: Vec<i16>,
    bool_features: Vec<bool>,
    float_features: Vec<f32>,
    int64_features: Vec<i64>,
    impr_ts: Vec<i32>,
    post_creation_ts_sec: Vec<i32>,
    line_item_ids: Vec<i64>,
    campaign_ids: Vec<i64>,
    funding_instrument_ids: Vec<i64>,
    semantic_ids: Vec<u16>,
}

impl InputBuffer {
    fn new_with_candidates(
        model_config: &ModelConfig,
        candidate_set: &pb::CandidateSet,
        mm_embeddings_opt: Option<Vec<f16>>,
    ) -> CandidateData {
        let num_author_hashes = model_config.hash_table.num_author_hashes();
        let num_item_hashes = model_config.hash_table.num_item_hashes();
        let candidate_seq_len = model_config.candidate_seq_len;
        let search_query_embedding_dim = model_config.hash_table.search_query_embedding_dim;

        let mut candidate_post_hashes = vec![0i32; candidate_seq_len * num_item_hashes];
        let mut candidate_auth_hashes = vec![0i32; candidate_seq_len * num_author_hashes];
        let mut candidate_product_surfaces = vec![0i32; candidate_seq_len];
        let mut candidate_impr_ts = vec![0i32; candidate_seq_len];
        let mut candidate_post_creation_ts_sec = vec![0i32; candidate_seq_len];
        let mut candidate_line_item_ids = vec![0i64; candidate_seq_len];
        let mut candidate_campaign_ids = vec![0i64; candidate_seq_len];
        let mut candidate_funding_instrument_ids = vec![0i64; candidate_seq_len];

        let sid_num_levels = model_config.sid_num_levels;
        let mut candidate_semantic_ids = vec![0u16; candidate_seq_len * sid_num_levels];

        let candidates_to_process = candidate_set.candidates.len().min(candidate_seq_len);

        let now_sec = now_epoch_sec();

        for (j, candidate) in candidate_set
            .candidates
            .iter()
            .take(candidates_to_process)
            .enumerate()
        {
            let base_author_idx = j * num_author_hashes;
            let base_tweet_idx = j * num_item_hashes;

            for k in 0..num_author_hashes {
                candidate_auth_hashes[base_author_idx + k] = model_config
                    .hash_table
                    .hash_author_id(candidate.author_id as i64, k);
            }

            for k in 0..num_item_hashes {
                candidate_post_hashes[base_tweet_idx + k] = model_config
                    .hash_table
                    .hash_item_id(candidate.tweet_id as i64, k);
            }

            candidate_product_surfaces[j] = candidate_set.product_surface;

            candidate_impr_ts[j] = now_sec;
            candidate_post_creation_ts_sec[j] =
                snowflake_to_creation_ts_sec(candidate.tweet_id as i64);

            if let Some(ad) = &candidate.ad_info {
                candidate_line_item_ids[j] = ad.line_item_id;
                candidate_campaign_ids[j] = ad.campaign_id;
                candidate_funding_instrument_ids[j] = ad.funding_instrument_id;
            }

            if sid_num_levels > 0 && candidate.semantic_ids.len() == sid_num_levels {
                let base = j * sid_num_levels;
                for (d, &c) in candidate_semantic_ids[base..base + sid_num_levels]
                    .iter_mut()
                    .zip(candidate.semantic_ids.iter())
                {
                    *d = (c + 1) as u16;
                }
            }
        }

        if sid_num_levels > 0 {
            let present = (0..candidates_to_process)
                .filter(|&j| candidate_semantic_ids[j * sid_num_levels] != 0)
                .count() as u64;
            record_sid_coverage("candidate", present, candidates_to_process as u64);
        }

        let mut candidate_search_query_embeddings =
            vec![0.0f32; candidate_seq_len * search_query_embedding_dim];
        if search_query_embedding_dim > 0 && !candidate_set.search_query_embedding.is_empty() {
            let provided_dim = candidate_set.search_query_embedding.len();
            if provided_dim == search_query_embedding_dim {
                for j in 0..candidates_to_process {
                    let base_idx = j * search_query_embedding_dim;
                    candidate_search_query_embeddings
                        [base_idx..base_idx + search_query_embedding_dim]
                        .copy_from_slice(&candidate_set.search_query_embedding);
                }
            }
        }

        let n_post_cat = model_config.hash_table.num_post_categorical_features;
        let n_post_bool = model_config.hash_table.num_post_bool_features;
        let n_post_float = model_config.hash_table.num_post_float_features;
        let n_post_int64 = model_config.hash_table.num_post_int64_features;

        let mut categorical_features = vec![0i16; candidate_seq_len * n_post_cat];
        stamp_i32_as_categorical(
            &candidate_product_surfaces,
            &mut categorical_features,
            n_post_cat,
            PRODUCT_SURFACE_CATEGORICAL_IDX,
        );

        let cand_tz_enum = candidate_set
            .device_feature
            .as_ref()
            .map(|df| df.timezone as i16)
            .unwrap_or(0);
        let candidate_tz_enums = vec![cand_tz_enum; candidate_seq_len];
        stamp_local_time_features(
            &candidate_impr_ts,
            &candidate_tz_enums,
            &mut categorical_features,
            n_post_cat,
        );

        let mut candidate_int64_features = vec![0i64; candidate_seq_len * n_post_int64];
        let mut candidate_is_author_followed = vec![false; candidate_seq_len];
        let mut candidate_is_author_following = vec![false; candidate_seq_len];
        let mut candidate_is_stale_post = vec![false; candidate_seq_len];
        let mut candidate_bool_features = vec![false; candidate_seq_len * n_post_bool];

        let stale_post_enabled = model_config.hash_table.enable_stale_post;
        for (j, candidate) in candidate_set
            .candidates
            .iter()
            .take(candidates_to_process)
            .enumerate()
        {
            let creation_valid = candidate_post_creation_ts_sec[j] > 0;
            let original_age_sec = now_sec as i64 - candidate_post_creation_ts_sec[j] as i64;
            let is_stale =
                stale_post_enabled && creation_valid && original_age_sec > STALE_POST_14D_TTL_SEC;
            candidate_is_stale_post[j] = is_stale;
            if is_stale {
                stamp_engagement_counts(
                    &mut candidate_int64_features,
                    n_post_int64,
                    j,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
            } else {
                stamp_engagement_counts(
                    &mut candidate_int64_features,
                    n_post_int64,
                    j,
                    candidate.fav_count,
                    candidate.reply_count,
                    candidate.retweet_count,
                    candidate.quote_count,
                    candidate.view_count,
                );
            }

            #[cfg(recsys_ads_dpa)]
            {
                let raw_key = candidate
                    .ad_info
                    .as_ref()
                    .and_then(|a| a.product_keys.first().copied())
                    .unwrap_or(0);
                if n_post_int64 > FIRST_DPA_PRODUCT_KEY {
                    candidate_int64_features[j * n_post_int64 + FIRST_DPA_PRODUCT_KEY] =
                        hash_dpa_product_key(raw_key);
                }
                if n_post_int64 > FIRST_DPA_PRODUCT_KEY_HASH2 {
                    candidate_int64_features[j * n_post_int64 + FIRST_DPA_PRODUCT_KEY_HASH2] =
                        hash_dpa_product_key_2(raw_key);
                }
            }
            candidate_is_author_followed[j] = candidate.is_author_followed_by_user;
            candidate_is_author_following[j] = candidate
                .author_info
                .as_ref()
                .and_then(|ai| ai.is_following_user)
                .unwrap_or(false);
        }

        let candidate_author_is_nsfw: Vec<i32> = candidate_set
            .candidates
            .iter()
            .take(candidates_to_process)
            .map(|c| ((c.safety_label_mask >> AUTHOR_NSFW_BIT) & 1) as i32)
            .collect();
        stamp_i32_as_categorical(
            &candidate_author_is_nsfw,
            &mut categorical_features,
            n_post_cat,
            AUTHOR_IS_NSFW_CATEGORICAL_IDX,
        );

        stamp_bool_seq(
            &candidate_is_stale_post,
            &mut candidate_bool_features,
            n_post_bool,
            IS_STALE_POST14D,
        );
        stamp_bool_seq(
            &candidate_is_author_followed,
            &mut candidate_bool_features,
            n_post_bool,
            IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ,
        );
        stamp_bool_seq(
            &candidate_is_author_following,
            &mut candidate_bool_features,
            n_post_bool,
            IS_AUTHOR_FOLLOWING_VIEWER_SEQ,
        );

        CandidateData {
            post_hashes: candidate_post_hashes,
            auth_hashes: candidate_auth_hashes,
            product_surfaces: candidate_product_surfaces,
            embeddings: mm_embeddings_opt.unwrap_or_default(),
            search_query_embeddings: candidate_search_query_embeddings,
            categorical_features,
            bool_features: candidate_bool_features,
            float_features: vec![0.0f32; candidate_seq_len * n_post_float],
            int64_features: candidate_int64_features,
            impr_ts: candidate_impr_ts,
            post_creation_ts_sec: candidate_post_creation_ts_sec,
            line_item_ids: candidate_line_item_ids,
            campaign_ids: candidate_campaign_ids,
            funding_instrument_ids: candidate_funding_instrument_ids,
            semantic_ids: candidate_semantic_ids,
        }
    }

    pub fn extract_user_features(
        config: &crate::model_config::HashTableConfig,
        client_ctx: Option<&pb::ClientContext>,
        user_ctx: Option<&pb::UserContext>,
    ) -> UserFeatures {
        use crate::feature_config::user_categorical_feature::{
            USER_AGE_BRACKET, USER_COUNTRY_CODE, USER_DMA_CODE, USER_GENDER, USER_INFERRED_GENDER,
            USER_LANGUAGE_CODE, USER_STATE,
        };
        use crate::feature_config::user_float_feature::{
            USER_AGE_IN_YEARS, USER_INFERRED_GENDER_SCORE, USER_LATITUDE, USER_LONGITUDE,
        };

        assert!(config.num_user_categorical_features > USER_COUNTRY_CODE);
        assert!(config.num_user_categorical_features > USER_LANGUAGE_CODE);
        assert!(config.num_user_categorical_features > USER_STATE);
        assert!(config.num_user_categorical_features > USER_GENDER);
        assert!(config.num_user_categorical_features > USER_AGE_BRACKET);
        assert!(config.num_user_categorical_features > USER_DMA_CODE);
        assert!(config.num_user_categorical_features > USER_INFERRED_GENDER);
        assert!(config.num_user_float_features > USER_LONGITUDE);
        assert!(config.num_user_float_features > USER_LATITUDE);
        assert!(config.num_user_float_features > USER_INFERRED_GENDER_SCORE);
        assert!(config.num_user_float_features > USER_AGE_IN_YEARS);

        let mut categorical_features = vec![0i16; config.num_user_categorical_features];
        let bool_features = vec![false; config.num_user_bool_features];
        let mut float_features = vec![0.0f32; config.num_user_float_features];
        let int64_features = vec![0i64; config.num_user_int64_features];

        fn record_fill(feature: &str, filled: bool) {
            USER_FEATURE_FILL_TOTAL
                .with_label_values(&[feature, if filled { "filled" } else { "missing" }])
                .inc();
        }

        if let Some(ctx) = client_ctx {
            categorical_features[USER_LANGUAGE_CODE] =
                pb::language_code_string_to_enum(&ctx.language_code) as i16;
            record_fill(
                "language_code",
                categorical_features[USER_LANGUAGE_CODE] != 0,
            );
            categorical_features[USER_COUNTRY_CODE] =
                pb::country_code_string_to_enum(&ctx.country_code) as i16;
            record_fill("country_code", categorical_features[USER_COUNTRY_CODE] != 0);
        }
        if let Some(ctx) = user_ctx {
            categorical_features[USER_STATE] = ctx.user_state as i16;
            record_fill("user_state", ctx.user_state != 0);
            categorical_features[USER_GENDER] = match pb::Gender::try_from(ctx.user_gender) {
                Ok(pb::Gender::Male) => 2,
                Ok(pb::Gender::Female) => 1,
                _ => 0,
            };
            record_fill("gender", categorical_features[USER_GENDER] != 0);
            categorical_features[USER_AGE_BRACKET] = ctx.user_age_bracket as i16;
            record_fill("age_bracket", ctx.user_age_bracket != 0);
            categorical_features[USER_DMA_CODE] = ctx.user_dma_code as i16;
            record_fill("dma_code", ctx.user_dma_code != 0);
            float_features[USER_LONGITUDE] = ctx.user_longitude;
            record_fill("longitude", ctx.user_longitude != 0.0);
            float_features[USER_LATITUDE] = ctx.user_latitude;
            record_fill("latitude", ctx.user_latitude != 0.0);
            categorical_features[USER_INFERRED_GENDER] = ctx.user_inferred_gender as i16;
            record_fill("inferred_gender", ctx.user_inferred_gender != 0);
            float_features[USER_INFERRED_GENDER_SCORE] = ctx.user_inferred_gender_score;
            record_fill(
                "inferred_gender_score",
                ctx.user_inferred_gender_score != 0.0,
            );
            float_features[USER_AGE_IN_YEARS] = ctx.user_age_in_years as f32;
            record_fill("age_in_years", ctx.user_age_in_years != 0);
        } else {
            record_fill("user_state", false);
            record_fill("gender", false);
            record_fill("age_bracket", false);
            record_fill("dma_code", false);
            record_fill("longitude", false);
            record_fill("latitude", false);
            record_fill("inferred_gender", false);
            record_fill("inferred_gender_score", false);
            record_fill("age_in_years", false);
        }

        let num_apps = config.num_user_installed_apps;
        let installed_apps = user_ctx
            .filter(|ctx| ctx.user_installed_apps.len() == num_apps)
            .map(|ctx| ctx.user_installed_apps.clone())
            .unwrap_or_else(|| vec![false; num_apps]);

        UserFeatures {
            categorical_features,
            bool_features,
            float_features,
            int64_features,
            installed_apps,
        }
    }

    pub fn compute_for_item(
        model_config: &ModelConfig,
        sequence: &Option<pb::UserActionSequence>,
        candidate_set: &pb::CandidateSet,
        mm_embeddings_opt: Option<Vec<f16>>,
        client_context: Option<&pb::ClientContext>,
        user_context: Option<&pb::UserContext>,
        conversion_history: Option<&pb::ConversionHistoryContext>,
    ) -> Self {
        use xai_recsys_proto::user_action_sequence_data_container::Data;

        let CandidateData {
            post_hashes: candidate_post_hashes,
            auth_hashes: candidate_auth_hashes,
            product_surfaces: candidate_product_surfaces,
            embeddings: candidate_embeddings,
            search_query_embeddings: candidate_search_query_embeddings,
            categorical_features: candidate_categorical_features,
            bool_features: candidate_bool_features,
            float_features: candidate_float_features,
            int64_features: candidate_int64_features,
            impr_ts: candidate_impr_ts,
            post_creation_ts_sec: candidate_post_creation_ts_sec,
            line_item_ids: candidate_line_item_ids,
            campaign_ids: candidate_campaign_ids,
            funding_instrument_ids: candidate_funding_instrument_ids,
            semantic_ids: candidate_semantic_ids,
        } = Self::new_with_candidates(model_config, candidate_set, mm_embeddings_opt);

        let num_user_hashes = model_config.hash_table.num_user_hashes();
        let num_author_hashes = model_config.hash_table.num_author_hashes();
        let num_item_hashes = model_config.hash_table.num_item_hashes();
        let history_seq_len = model_config.history_seq_len;
        let output_vocab_size = model_config.hash_table.output_vocab_size;
        let num_continuous_actions = model_config.hash_table.num_continuous_actions;

        let mut user_hashes = vec![0i32; num_user_hashes];
        let user_id = if candidate_set.user_id != 0 {
            candidate_set.user_id as i64
        } else {
            sequence.as_ref().map(|s| s.user_id as i64).unwrap_or(0)
        };
        for (k, slot) in user_hashes.iter_mut().enumerate().take(num_user_hashes) {
            *slot = model_config.hash_table.hash_user_id(user_id, k);
        }

        let n_post_int64 = model_config.hash_table.num_post_int64_features;
        let mut history_post_hashes = vec![0i32; history_seq_len * num_item_hashes];
        let mut history_auth_hashes = vec![0i32; history_seq_len * num_author_hashes];
        let mut history_product_surfaces = vec![0i32; history_seq_len];
        let mut history_actions = vec![false; history_seq_len * output_vocab_size];
        let mut history_continuous_actions = vec![0.0f32; history_seq_len * num_continuous_actions];
        let mut history_impr_ts = vec![0i32; history_seq_len];
        let mut history_post_creation_ts_sec = vec![0i32; history_seq_len];
        let mut history_tz_enums = vec![0i16; history_seq_len];
        let mut history_author_is_nsfw = vec![0i32; history_seq_len];
        let mut history_is_author_followed = vec![false; history_seq_len];
        let mut history_is_author_following = vec![false; history_seq_len];
        let mut history_post_ids = vec![0i64; history_seq_len];
        let mut history_int64_features = vec![0i64; history_seq_len * n_post_int64];

        let _empty = Vec::new();
        let agg_user_actions = match sequence {
            Some(sequence) => match &sequence.user_actions_data {
                Some(container) => match &container.data {
                    Some(Data::OrderedAggregatedUserActionsList(list)) => {
                        &list.aggregated_user_actions
                    }
                    _ => &_empty,
                },
                None => &_empty,
            },
            None => &_empty,
        };

        let start = agg_user_actions.len().saturating_sub(history_seq_len);
        let user_actions = &agg_user_actions[start..];

        let mut valid_entry_count = 0;
        for aggregated_user_action in user_actions.iter() {
            if let Some(tweet_info) = &aggregated_user_action.tweet_info {
                if tweet_info.author_id == 0 || tweet_info.tweet_id == 0 {
                    continue;
                }
                let author_id = tweet_info.author_id as i64;
                let tweet_id = tweet_info.tweet_id as i64;
                let base_author_idx = valid_entry_count * num_author_hashes;
                let base_tweet_idx = valid_entry_count * num_item_hashes;

                for k in 0..num_author_hashes {
                    history_auth_hashes[base_author_idx + k] =
                        model_config.hash_table.hash_author_id(author_id, k);
                }

                for k in 0..num_item_hashes {
                    history_post_hashes[base_tweet_idx + k] =
                        model_config.hash_table.hash_item_id(tweet_id, k);
                }

                history_post_ids[valid_entry_count] = tweet_id;

                let base_idx = valid_entry_count * output_vocab_size;
                let continuous_base_idx = valid_entry_count * num_continuous_actions;
                let mut total_dwell_time: f64 = 0.0;
                for action_info in &aggregated_user_action.actions {
                    let action_idx = action_info.action_name as usize;
                    if action_idx < output_vocab_size {
                        history_actions[base_idx + action_idx] = true;
                    }
                    if let Some(user_action_meta) = &action_info.user_action_meta {
                        total_dwell_time += user_action_meta.dwell_time;
                    }
                }
                let dwell_time_seconds = (total_dwell_time / 1000.0) as f32;
                let dwell_time_idx = pb::ContinuousActionName::DwellTime as usize;
                if dwell_time_idx < num_continuous_actions {
                    history_continuous_actions[continuous_base_idx + dwell_time_idx] =
                        dwell_time_seconds;
                }

                let product_surface_val = {
                    let product_surfaces: Vec<i32> = aggregated_user_action
                        .actions
                        .iter()
                        .filter_map(|action| {
                            action
                                .user_action_meta
                                .as_ref()
                                .map(|meta| meta.product_surface)
                        })
                        .collect();

                    if product_surfaces.is_empty() {
                        pb::ProductSurface::Unknown as i32
                    } else {
                        product_surfaces[0]
                    }
                };
                history_product_surfaces[valid_entry_count] = product_surface_val;

                history_impr_ts[valid_entry_count] =
                    (aggregated_user_action.impressed_time_ms / 1000) as i32;

                history_post_creation_ts_sec[valid_entry_count] =
                    snowflake_to_creation_ts_sec(tweet_id);

                let tz_enum_val = aggregated_user_action
                    .actions
                    .iter()
                    .filter_map(|action| {
                        action
                            .user_action_meta
                            .as_ref()
                            .filter(|meta| !meta.time_zone.is_empty())
                            .map(|meta| pb::timezone_string_to_enum(&meta.time_zone) as i16)
                    })
                    .next()
                    .unwrap_or(0);
                history_tz_enums[valid_entry_count] = tz_enum_val;

                let hist_safety_mask = aggregated_user_action
                    .actions
                    .iter()
                    .filter_map(|action| {
                        action
                            .user_action_meta
                            .as_ref()
                            .map(|m| m.safety_label_mask)
                    })
                    .fold(0u64, |acc, m| acc | m);
                history_author_is_nsfw[valid_entry_count] =
                    ((hist_safety_mask >> AUTHOR_NSFW_BIT) & 1) as i32;

                stamp_engagement_counts(
                    &mut history_int64_features,
                    n_post_int64,
                    valid_entry_count,
                    tweet_info.fav_count,
                    tweet_info.reply_count,
                    tweet_info.retweet_count,
                    tweet_info.quote_count,
                    tweet_info.view_count,
                );

                history_is_author_followed[valid_entry_count] =
                    tweet_info.is_author_followed_by_user;
                history_is_author_following[valid_entry_count] = tweet_info
                    .author_info
                    .as_ref()
                    .and_then(|ai| ai.is_following_user)
                    .unwrap_or(false);

                valid_entry_count += 1;
            }
        }

        let user_features =
            Self::extract_user_features(&model_config.hash_table, client_context, user_context);

        let n_post_cat = model_config.hash_table.num_post_categorical_features;
        let n_post_bool = model_config.hash_table.num_post_bool_features;
        let n_post_float = model_config.hash_table.num_post_float_features;

        let mut history_categorical_features = vec![0i16; history_seq_len * n_post_cat];
        stamp_i32_as_categorical(
            &history_product_surfaces,
            &mut history_categorical_features,
            n_post_cat,
            PRODUCT_SURFACE_CATEGORICAL_IDX,
        );
        stamp_local_time_features(
            &history_impr_ts,
            &history_tz_enums,
            &mut history_categorical_features,
            n_post_cat,
        );
        stamp_i32_as_categorical(
            &history_author_is_nsfw,
            &mut history_categorical_features,
            n_post_cat,
            AUTHOR_IS_NSFW_CATEGORICAL_IDX,
        );

        let mut history_bool_features = vec![false; history_seq_len * n_post_bool];
        stamp_bool_seq(
            &history_is_author_followed,
            &mut history_bool_features,
            n_post_bool,
            IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ,
        );
        stamp_bool_seq(
            &history_is_author_following,
            &mut history_bool_features,
            n_post_bool,
            IS_AUTHOR_FOLLOWING_VIEWER_SEQ,
        );

        let request_ip = candidate_set
            .device_feature
            .as_ref()
            .map(|df| &df.ip_address)
            .filter(|ip| !ip.is_empty())
            .map(|ip| parse_ipv4_to_u32(ip) as i64)
            .unwrap_or(0);
        let user_ip_hashes = compute_user_ip_hashes(model_config, request_ip);

        let user_client_id_idx = vec![
            client_context
                .map(|c| encode_client_id(c.app_id))
                .unwrap_or(0),
        ];
        let user_country_code_idx = vec![
            client_context
                .map(|c| encode_country_code_crc32(&c.country_code))
                .unwrap_or(0),
        ];
        let user_gender_idx = vec![user_context.map(|u| u.user_gender).unwrap_or(0)];
        let user_age_bucket_idx = vec![user_context.map(|u| u.user_age_bracket).unwrap_or(0)];
        let user_age_in_years_idx = vec![user_context.map(|u| u.user_age_in_years).unwrap_or(0)];
        let user_inferred_gender_idx =
            vec![user_context.map(|u| u.user_inferred_gender).unwrap_or(0)];
        let user_inferred_gender_score = vec![
            user_context
                .map(|u| u.user_inferred_gender_score)
                .unwrap_or(0.0),
        ];

        let candidate_conversion_dense_features = Self::extract_candidate_conversion_dense_features(
            conversion_history,
            model_config.candidate_seq_len,
        );
        let user_conversion_history_hashes =
            Self::extract_user_conversion_history_hashes(conversion_history);
        let candidate_account_hashes = Self::extract_candidate_account_hashes(
            conversion_history,
            model_config.candidate_seq_len,
        );

        Self {
            user_hashes,
            user_ip_hashes,
            history_post_hashes,
            history_auth_hashes,
            history_product_surfaces,
            history_actions,
            history_continuous_actions,
            candidate_post_hashes,
            candidate_auth_hashes,
            candidate_product_surfaces,
            candidate_embeddings,
            candidate_search_query_embeddings,
            user_categorical_features: user_features.categorical_features,
            user_bool_features: user_features.bool_features,
            user_float_features: user_features.float_features,
            user_int64_features: user_features.int64_features,
            user_installed_apps: user_features.installed_apps,
            history_categorical_features,
            history_bool_features,
            history_float_features: vec![0.0f32; history_seq_len * n_post_float],
            history_int64_features,
            candidate_categorical_features,
            candidate_bool_features,
            candidate_float_features,
            candidate_int64_features,
            history_impr_ts,
            history_post_creation_ts_sec,
            candidate_impr_ts,
            candidate_post_creation_ts_sec,
            candidate_line_item_ids,
            candidate_campaign_ids,
            candidate_funding_instrument_ids,
            user_client_id_idx,
            user_country_code_idx,
            user_gender_idx,
            user_age_bucket_idx,
            user_age_in_years_idx,
            user_inferred_gender_idx,
            user_inferred_gender_score,
            candidate_conversion_dense_features,
            user_conversion_history_hashes,
            candidate_account_hashes,
            history_post_ids,
            history_semantic_ids: vec![0u16; history_seq_len * model_config.sid_num_levels],
            candidate_semantic_ids,
            num_history: valid_entry_count,
        }
    }

    pub fn compute_from_columnar_bytes(
        model_config: &ModelConfig,
        columnar_bytes: &[u8],
        candidate_set: &pb::CandidateSet,
        mm_embeddings_opt: Option<Vec<f16>>,
        client_context: Option<&pb::ClientContext>,
        user_context: Option<&pb::UserContext>,
        conversion_history: Option<&pb::ConversionHistoryContext>,
    ) -> Result<Self, arrow::error::ArrowError> {
        let CandidateData {
            post_hashes: candidate_post_hashes,
            auth_hashes: candidate_auth_hashes,
            product_surfaces: candidate_product_surfaces,
            embeddings: candidate_embeddings,
            search_query_embeddings: candidate_search_query_embeddings,
            categorical_features: candidate_categorical_features,
            bool_features: candidate_bool_features,
            float_features: candidate_float_features,
            int64_features: candidate_int64_features,
            impr_ts: candidate_impr_ts,
            post_creation_ts_sec: candidate_post_creation_ts_sec,
            line_item_ids: candidate_line_item_ids,
            campaign_ids: candidate_campaign_ids,
            funding_instrument_ids: candidate_funding_instrument_ids,
            semantic_ids: candidate_semantic_ids,
        } = Self::new_with_candidates(model_config, candidate_set, mm_embeddings_opt);

        let num_user_hashes = model_config.hash_table.num_user_hashes();
        let num_author_hashes = model_config.hash_table.num_author_hashes();
        let num_item_hashes = model_config.hash_table.num_item_hashes();
        let history_seq_len = model_config.history_seq_len;
        let output_vocab_size = model_config.hash_table.output_vocab_size;
        let num_continuous_actions = model_config.hash_table.num_continuous_actions;

        let mut user_hashes = vec![0i32; num_user_hashes];
        let user_id = candidate_set.user_id as i64;
        for (k, slot) in user_hashes.iter_mut().enumerate().take(num_user_hashes) {
            *slot = model_config.hash_table.hash_user_id(user_id, k);
        }

        let n_post_int64 = model_config.hash_table.num_post_int64_features;
        let mut history_post_hashes = vec![0i32; history_seq_len * num_item_hashes];
        let mut history_auth_hashes = vec![0i32; history_seq_len * num_author_hashes];
        let mut history_product_surfaces = vec![0i32; history_seq_len];
        let mut history_actions = vec![false; history_seq_len * output_vocab_size];
        let mut history_continuous_actions = vec![0.0f32; history_seq_len * num_continuous_actions];
        let mut history_impr_ts = vec![0i32; history_seq_len];
        let mut history_post_creation_ts_sec = vec![0i32; history_seq_len];
        let mut history_tz_enums = vec![0i16; history_seq_len];
        let mut history_post_ids = vec![0i64; history_seq_len];
        let mut history_int64_features = vec![0i64; history_seq_len * n_post_int64];
        let mut history_is_author_followed = vec![false; history_seq_len];
        let mut history_is_author_following = vec![false; history_seq_len];
        let sid_num_levels = model_config.sid_num_levels;
        let mut history_semantic_ids = vec![0u16; history_seq_len * sid_num_levels];

        let cursor = Cursor::new(columnar_bytes);
        let mut reader = StreamReader::try_new(cursor, None)?;

        let batch = match reader.next() {
            Some(Ok(batch)) if batch.num_rows() > 0 => batch,
            Some(Err(e)) => return Err(e),
            _ => {
                let user_features = Self::extract_user_features(
                    &model_config.hash_table,
                    client_context,
                    user_context,
                );
                let n_post_cat = model_config.hash_table.num_post_categorical_features;
                let n_post_bool = model_config.hash_table.num_post_bool_features;
                let n_post_float = model_config.hash_table.num_post_float_features;
                let n_post_int64 = model_config.hash_table.num_post_int64_features;
                let history_bool_features = vec![false; history_seq_len * n_post_bool];
                let request_ip = candidate_set
                    .device_feature
                    .as_ref()
                    .map(|df| &df.ip_address)
                    .filter(|ip| !ip.is_empty())
                    .map(|ip| parse_ipv4_to_u32(ip) as i64)
                    .unwrap_or(0);
                let user_ip_hashes = compute_user_ip_hashes(model_config, request_ip);
                let user_client_id_idx = vec![
                    client_context
                        .map(|c| encode_client_id(c.app_id))
                        .unwrap_or(0),
                ];
                let user_country_code_idx = vec![
                    client_context
                        .map(|c| encode_country_code_crc32(&c.country_code))
                        .unwrap_or(0),
                ];
                let user_gender_idx = vec![user_context.map(|u| u.user_gender).unwrap_or(0)];
                let user_age_bucket_idx =
                    vec![user_context.map(|u| u.user_age_bracket).unwrap_or(0)];
                let user_age_in_years_idx =
                    vec![user_context.map(|u| u.user_age_in_years).unwrap_or(0)];
                let user_inferred_gender_idx =
                    vec![user_context.map(|u| u.user_inferred_gender).unwrap_or(0)];
                let user_inferred_gender_score = vec![
                    user_context
                        .map(|u| u.user_inferred_gender_score)
                        .unwrap_or(0.0),
                ];
                let candidate_conversion_dense_features =
                    Self::extract_candidate_conversion_dense_features(
                        conversion_history,
                        model_config.candidate_seq_len,
                    );
                let user_conversion_history_hashes =
                    Self::extract_user_conversion_history_hashes(conversion_history);
                let candidate_account_hashes = Self::extract_candidate_account_hashes(
                    conversion_history,
                    model_config.candidate_seq_len,
                );
                return Ok(Self {
                    user_hashes,
                    user_ip_hashes,
                    history_post_hashes,
                    history_auth_hashes,
                    history_product_surfaces,
                    history_actions,
                    history_continuous_actions,
                    candidate_post_hashes,
                    candidate_auth_hashes,
                    candidate_product_surfaces,
                    candidate_embeddings,
                    candidate_search_query_embeddings,
                    user_categorical_features: user_features.categorical_features,
                    user_bool_features: user_features.bool_features,
                    user_float_features: user_features.float_features,
                    user_int64_features: user_features.int64_features,
                    user_installed_apps: user_features.installed_apps,
                    history_categorical_features: vec![0i16; history_seq_len * n_post_cat],
                    history_bool_features,
                    history_float_features: vec![0.0f32; history_seq_len * n_post_float],
                    history_int64_features: vec![0i64; history_seq_len * n_post_int64],
                    candidate_categorical_features,
                    candidate_bool_features,
                    candidate_float_features,
                    candidate_int64_features,
                    history_impr_ts,
                    history_post_creation_ts_sec,
                    candidate_impr_ts,
                    candidate_post_creation_ts_sec,
                    candidate_line_item_ids,
                    candidate_campaign_ids,
                    candidate_funding_instrument_ids,
                    user_client_id_idx,
                    user_country_code_idx,
                    user_gender_idx,
                    user_age_bucket_idx,
                    user_age_in_years_idx,
                    user_inferred_gender_idx,
                    user_inferred_gender_score,
                    candidate_conversion_dense_features,
                    user_conversion_history_hashes,
                    candidate_account_hashes,
                    history_post_ids: vec![0i64; history_seq_len],
                    history_semantic_ids: vec![0u16; history_seq_len * sid_num_levels],
                    candidate_semantic_ids,
                    num_history: 0,
                });
            }
        };

        let num_rows = batch.num_rows();

        let tweet_ids = batch
            .column_by_name("tweetId")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let author_ids = batch
            .column_by_name("authorId")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let product_surfaces = batch
            .column_by_name(PRODUCT_SURFACE_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let action_multi_hot = batch.column_by_name("actionNameMultiHot");
        let continuous_actions = batch.column_by_name("continuousActionValuesSeq");

        let col_impressed_time_ms = batch
            .column_by_name("impressedTimeMs")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let col_timezone_id = batch
            .column_by_name("timezoneId")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());

        let col_semantic_id = batch.column_by_name("semanticId");

        let col_fav_count = batch
            .column_by_name(FAV_COUNT_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let col_reply_count = batch
            .column_by_name(REPLY_COUNT_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let col_repost_count = batch
            .column_by_name(REPOST_COUNT_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let col_quote_count = batch
            .column_by_name(QUOTE_COUNT_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let col_view_count = batch
            .column_by_name(VIEW_COUNT_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let col_is_author_followed_by_viewer = batch
            .column_by_name(IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());
        let col_is_author_following_viewer = batch
            .column_by_name(IS_AUTHOR_FOLLOWING_VIEWER_SEQ_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

        let start_row = num_rows.saturating_sub(history_seq_len);
        let mut valid_entry_count = 0;

        for row_idx in start_row..num_rows {
            let tweet_id = tweet_ids.map_or(0i64, |arr| arr.value(row_idx));
            let author_id = author_ids.map_or(0i64, |arr| arr.value(row_idx));

            if tweet_id == 0 || author_id == 0 {
                continue;
            }

            let base_tweet_idx = valid_entry_count * num_item_hashes;
            let base_author_idx = valid_entry_count * num_author_hashes;

            for k in 0..num_item_hashes {
                history_post_hashes[base_tweet_idx + k] =
                    model_config.hash_table.hash_item_id(tweet_id, k);
            }

            for k in 0..num_author_hashes {
                history_auth_hashes[base_author_idx + k] =
                    model_config.hash_table.hash_author_id(author_id, k);
            }

            history_post_ids[valid_entry_count] = tweet_id;

            if sid_num_levels > 0
                && let Some(sid_col) = col_semantic_id
            {
                let fsl = sid_col.as_fixed_size_list();
                if !fsl.is_null(row_idx) {
                    let inner = fsl.value(row_idx);
                    if let Some(codes) = inner.as_any().downcast_ref::<Int32Array>()
                        && codes.len() == sid_num_levels
                    {
                        let base = valid_entry_count * sid_num_levels;
                        for d in 0..sid_num_levels {
                            history_semantic_ids[base + d] = (codes.value(d) + 1) as u16;
                        }
                    }
                }
            }

            if let Some(ps) = product_surfaces {
                history_product_surfaces[valid_entry_count] = ps.value(row_idx);
            }

            if let Some(action_col) = action_multi_hot {
                let fsl = action_col.as_fixed_size_list();
                let inner_bools = fsl.value(row_idx);
                if let Some(bools) = inner_bools.as_any().downcast_ref::<BooleanArray>() {
                    let base_idx = valid_entry_count * output_vocab_size;
                    let copy_len = output_vocab_size.min(bools.len());
                    for j in 0..copy_len {
                        history_actions[base_idx + j] = bools.value(j);
                    }
                }
            }

            if let Some(cont_col) = continuous_actions {
                let fsl = cont_col.as_fixed_size_list();
                let inner_floats = fsl.value(row_idx);
                if let Some(floats) = inner_floats.as_any().downcast_ref::<Float32Array>() {
                    let continuous_base_idx = valid_entry_count * num_continuous_actions;
                    let copy_len = num_continuous_actions.min(floats.len());
                    for j in 0..copy_len {
                        history_continuous_actions[continuous_base_idx + j] = floats.value(j);
                    }
                }
            }

            history_post_creation_ts_sec[valid_entry_count] =
                snowflake_to_creation_ts_sec(tweet_id);

            if let Some(ts) = col_impressed_time_ms {
                history_impr_ts[valid_entry_count] = (ts.value(row_idx) / 1000) as i32;
            }

            if let Some(tz) = col_timezone_id {
                history_tz_enums[valid_entry_count] = tz.value(row_idx) as i16;
            }

            stamp_engagement_counts(
                &mut history_int64_features,
                n_post_int64,
                valid_entry_count,
                col_fav_count.map_or(0, |arr| arr.value(row_idx)) as u64,
                col_reply_count.map_or(0, |arr| arr.value(row_idx)) as u64,
                col_repost_count.map_or(0, |arr| arr.value(row_idx)) as u64,
                col_quote_count.map_or(0, |arr| arr.value(row_idx)) as u64,
                col_view_count.map_or(0, |arr| arr.value(row_idx)) as u64,
            );

            history_is_author_followed[valid_entry_count] = col_is_author_followed_by_viewer
                .is_some_and(|arr| !arr.is_null(row_idx) && arr.value(row_idx));
            history_is_author_following[valid_entry_count] = col_is_author_following_viewer
                .is_some_and(|arr| !arr.is_null(row_idx) && arr.value(row_idx));

            valid_entry_count += 1;
        }

        if sid_num_levels > 0 {
            let present = (0..valid_entry_count)
                .filter(|&i| history_semantic_ids[i * sid_num_levels] != 0)
                .count() as u64;
            record_sid_coverage("history", present, valid_entry_count as u64);
        }

        let user_features =
            Self::extract_user_features(&model_config.hash_table, client_context, user_context);

        let n_post_cat = model_config.hash_table.num_post_categorical_features;
        let n_post_bool = model_config.hash_table.num_post_bool_features;
        let n_post_float = model_config.hash_table.num_post_float_features;

        let mut history_categorical_features = vec![0i16; history_seq_len * n_post_cat];
        stamp_i32_as_categorical(
            &history_product_surfaces,
            &mut history_categorical_features,
            n_post_cat,
            PRODUCT_SURFACE_CATEGORICAL_IDX,
        );
        stamp_local_time_features(
            &history_impr_ts,
            &history_tz_enums,
            &mut history_categorical_features,
            n_post_cat,
        );

        let mut history_bool_features = vec![false; history_seq_len * n_post_bool];
        stamp_bool_seq(
            &history_is_author_followed,
            &mut history_bool_features,
            n_post_bool,
            IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ,
        );
        stamp_bool_seq(
            &history_is_author_following,
            &mut history_bool_features,
            n_post_bool,
            IS_AUTHOR_FOLLOWING_VIEWER_SEQ,
        );

        let request_ip = candidate_set
            .device_feature
            .as_ref()
            .map(|df| &df.ip_address)
            .filter(|ip| !ip.is_empty())
            .map(|ip| parse_ipv4_to_u32(ip) as i64)
            .unwrap_or(0);
        let user_ip_hashes = compute_user_ip_hashes(model_config, request_ip);

        let user_client_id_idx = vec![
            client_context
                .map(|c| encode_client_id(c.app_id))
                .unwrap_or(0),
        ];
        let user_country_code_idx = vec![
            client_context
                .map(|c| encode_country_code_crc32(&c.country_code))
                .unwrap_or(0),
        ];
        let user_gender_idx = vec![user_context.map(|u| u.user_gender).unwrap_or(0)];
        let user_age_bucket_idx = vec![user_context.map(|u| u.user_age_bracket).unwrap_or(0)];
        let user_age_in_years_idx = vec![user_context.map(|u| u.user_age_in_years).unwrap_or(0)];
        let user_inferred_gender_idx =
            vec![user_context.map(|u| u.user_inferred_gender).unwrap_or(0)];
        let user_inferred_gender_score = vec![
            user_context
                .map(|u| u.user_inferred_gender_score)
                .unwrap_or(0.0),
        ];

        let candidate_conversion_dense_features = Self::extract_candidate_conversion_dense_features(
            conversion_history,
            model_config.candidate_seq_len,
        );
        let user_conversion_history_hashes =
            Self::extract_user_conversion_history_hashes(conversion_history);
        let candidate_account_hashes = Self::extract_candidate_account_hashes(
            conversion_history,
            model_config.candidate_seq_len,
        );

        Ok(Self {
            user_hashes,
            user_ip_hashes,
            history_post_hashes,
            history_auth_hashes,
            history_product_surfaces,
            history_actions,
            history_continuous_actions,
            candidate_post_hashes,
            candidate_auth_hashes,
            candidate_product_surfaces,
            candidate_embeddings,
            candidate_search_query_embeddings,
            user_categorical_features: user_features.categorical_features,
            user_bool_features: user_features.bool_features,
            user_float_features: user_features.float_features,
            user_int64_features: user_features.int64_features,
            user_installed_apps: user_features.installed_apps,
            history_categorical_features,
            history_bool_features,
            history_float_features: vec![0.0f32; history_seq_len * n_post_float],
            history_int64_features,
            candidate_categorical_features,
            candidate_bool_features,
            candidate_float_features,
            candidate_int64_features,
            history_impr_ts,
            history_post_creation_ts_sec,
            candidate_impr_ts,
            candidate_post_creation_ts_sec,
            candidate_line_item_ids,
            candidate_campaign_ids,
            candidate_funding_instrument_ids,
            user_client_id_idx,
            user_country_code_idx,
            user_gender_idx,
            user_age_bucket_idx,
            user_age_in_years_idx,
            user_inferred_gender_idx,
            user_inferred_gender_score,
            candidate_conversion_dense_features,
            user_conversion_history_hashes,
            candidate_account_hashes,
            history_post_ids,
            history_semantic_ids,
            candidate_semantic_ids,
            num_history: valid_entry_count,
        })
    }

    const NUM_CONVERSION_DENSE_FEATURES: usize = 29;
    const NUM_CONVERSION_HISTORY_HASHES: usize = 8;
    const PER_CANDIDATE_STRIDE: usize = Self::NUM_CONVERSION_DENSE_FEATURES;

    fn extract_candidate_conversion_dense_features(
        ch: Option<&pb::ConversionHistoryContext>,
        candidate_seq_len: usize,
    ) -> Vec<f32> {
        let total = candidate_seq_len * Self::NUM_CONVERSION_DENSE_FEATURES;
        let Some(ch) = ch else {
            return vec![0.0; total];
        };

        let mut result = Vec::with_capacity(total);
        for i in 0..candidate_seq_len {
            let base = i * Self::PER_CANDIDATE_STRIDE;
            if base + Self::PER_CANDIDATE_STRIDE <= ch.per_candidate_features.len() {
                result.extend_from_slice(
                    &ch.per_candidate_features[base..base + Self::PER_CANDIDATE_STRIDE],
                );
            } else {
                result.extend_from_slice(&[0.0; Self::PER_CANDIDATE_STRIDE]);
            }
        }
        result
    }

    fn extract_user_conversion_history_hashes(
        ch: Option<&pb::ConversionHistoryContext>,
    ) -> Vec<i32> {
        match ch {
            Some(ch) if !ch.hist_adv_hashes.is_empty() => {
                let mut hashes = ch.hist_adv_hashes.clone();
                hashes.resize(Self::NUM_CONVERSION_HISTORY_HASHES, 0);
                hashes
            }
            _ => vec![0; Self::NUM_CONVERSION_HISTORY_HASHES],
        }
    }

    fn extract_candidate_account_hashes(
        ch: Option<&pb::ConversionHistoryContext>,
        candidate_seq_len: usize,
    ) -> Vec<i32> {
        let mut result = vec![0i32; candidate_seq_len];
        if let Some(ch) = ch {
            let n = ch.per_candidate_account_hashes.len().min(candidate_seq_len);
            result[..n].copy_from_slice(&ch.per_candidate_account_hashes[..n]);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_config::categorical_feature::{
        PRODUCT_SURFACE_SEQ, PRODUCT_SURFACE_SEQ_COLUMN, PRODUCT_SURFACE_SEQ_NAME,
    };
    use crate::feature_config::int64_feature::{CLIENT_APP_ID_SEQ, CLIENT_APP_ID_SEQ_NAME};

    fn test_model_config(num_post_categorical_features: usize) -> ModelConfig {
        ModelConfig {
            hash_table: crate::model_config::HashTableConfig {
                user_id_table_size: 128,
                user_hash_scales: vec![1, 2],
                user_biases: vec![0, 0],
                user_modulus: 1_000_003,
                item_id_table_size: 128,
                item_hash_vocab_size: 0,
                item_hash_scales: vec![3, 4],
                item_biases: vec![0, 0],
                item_modulus: 1_000_003,
                author_id_table_size: 128,
                author_hash_scales: vec![5, 6],
                author_biases: vec![0, 0],
                author_modulus: 1_000_003,
                ip_id_table_size: 0,
                ip_hash_scales: vec![],
                ip_biases: vec![],
                ip_modulus: 1_073_741_789,
                output_vocab_size: 64,
                num_continuous_actions: 2,
                search_query_embedding_dim: 0,
                num_user_categorical_features: 0,
                num_user_bool_features: 0,
                num_user_float_features: 0,
                num_user_int64_features: 0,
                num_user_installed_apps: 0,
                num_post_categorical_features,
                num_post_bool_features: 0,
                num_post_float_features: 0,
                num_post_int64_features: 0,
                enable_stale_post: false,
            },
            history_seq_len: 4,
            candidate_seq_len: 3,
            multimodal_embedding_dim: 0,
            search_query_embedding_dim: 0,
            num_categorical_features: 0,
            sid_num_levels: 0,
        }
    }

    #[test]
    fn feature_config_name_and_value() {
        assert_eq!(0, PRODUCT_SURFACE_SEQ);
        assert_eq!("productSurfaceSeq", PRODUCT_SURFACE_SEQ_NAME);
        assert_eq!("productSurface", PRODUCT_SURFACE_SEQ_COLUMN);
        assert_eq!(0, CLIENT_APP_ID_SEQ);
        assert_eq!("clientAppIdSeq", CLIENT_APP_ID_SEQ_NAME);
    }

    #[test]
    fn stamp_i32_as_categorical_basic() {
        let source = vec![10i32, 20, 30];
        let num_features = 3;
        let feature_idx = 1;
        let mut dest = vec![0i16; source.len() * num_features];

        stamp_i32_as_categorical(&source, &mut dest, num_features, feature_idx);

        assert_eq!(vec![0, 10, 0, 0, 20, 0, 0, 30, 0], dest);
    }

    #[test]
    fn stamp_i32_as_categorical_noop_when_feature_idx_out_of_range() {
        let source = vec![1i32, 2, 3];
        let num_features = 2;
        let mut dest = vec![0i16; source.len() * num_features];

        stamp_i32_as_categorical(&source, &mut dest, num_features, num_features);
        assert!(dest.iter().all(|&v| v == 0));
    }

    #[test]
    fn product_surface_stamped_into_candidate_categorical_features() {
        let n_post_cat = 3;
        let model_config = test_model_config(n_post_cat);
        let candidate_seq_len = model_config.candidate_seq_len;

        let mut candidate_set = pb::CandidateSet {
            product_surface: pb::ProductSurface::HomeTimelineRanking as i32,
            ..Default::default()
        };

        for i in 0..candidate_seq_len {
            candidate_set.candidates.push(pb::TweetInfo {
                tweet_id: 1000 + i as u64,
                author_id: 2000 + i as u64,
                ..Default::default()
            });
        }

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        for j in 0..candidate_seq_len {
            assert_eq!(
                cand.product_surfaces[j] as i16,
                cand.categorical_features[j * n_post_cat + PRODUCT_SURFACE_CATEGORICAL_IDX],
                "candidate {j}: product_surfaces[{j}] != categorical_features[{idx}]",
                idx = j * n_post_cat + PRODUCT_SURFACE_CATEGORICAL_IDX,
            );
        }

        for j in 0..candidate_seq_len {
            for f in 0..n_post_cat {
                if f != PRODUCT_SURFACE_CATEGORICAL_IDX {
                    assert_eq!(
                        0i16,
                        cand.categorical_features[j * n_post_cat + f],
                        "candidate {j}: non-product-surface slot {f} should be zero",
                    );
                }
            }
        }
    }

    #[test]
    fn author_is_nsfw_stamped_into_candidate_categorical_features() {
        let n_post_cat = AUTHOR_IS_NSFW_CATEGORICAL_IDX + 1;
        let model_config = test_model_config(n_post_cat);
        let candidate_seq_len = model_config.candidate_seq_len;

        let mut candidate_set = pb::CandidateSet::default();
        for i in 0..candidate_seq_len {
            candidate_set.candidates.push(pb::TweetInfo {
                tweet_id: 1000 + i as u64,
                author_id: 2000 + i as u64,
                safety_label_mask: if i % 2 == 0 { 1 << AUTHOR_NSFW_BIT } else { 0 },
                ..Default::default()
            });
        }

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        for j in 0..candidate_seq_len {
            let expected = if j % 2 == 0 { 1i16 } else { 0i16 };
            assert_eq!(
                expected,
                cand.categorical_features[j * n_post_cat + AUTHOR_IS_NSFW_CATEGORICAL_IDX],
                "candidate {j}: authorIsNsfwSeq mismatch at idx {idx}",
                idx = j * n_post_cat + AUTHOR_IS_NSFW_CATEGORICAL_IDX,
            );
        }
    }

    #[test]
    fn author_is_nsfw_noop_when_feature_idx_out_of_range() {
        let n_post_cat = 1;
        let model_config = test_model_config(n_post_cat);
        let candidate_seq_len = model_config.candidate_seq_len;

        let mut candidate_set = pb::CandidateSet::default();
        for i in 0..candidate_seq_len {
            candidate_set.candidates.push(pb::TweetInfo {
                tweet_id: 1000 + i as u64,
                safety_label_mask: 1 << AUTHOR_NSFW_BIT,
                ..Default::default()
            });
        }

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);
        assert_eq!(
            cand.categorical_features.len(),
            candidate_seq_len * n_post_cat
        );
    }

    #[test]
    fn product_surface_padding_slots_are_zero() {
        let n_post_cat = 1;
        let model_config = test_model_config(n_post_cat);
        let candidate_seq_len = model_config.candidate_seq_len;

        let mut candidate_set = pb::CandidateSet {
            product_surface: pb::ProductSurface::SearchResultsPage as i32,
            ..Default::default()
        };
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: 1000,
            author_id: 2000,
            ..Default::default()
        });

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        assert_eq!(6i16, cand.categorical_features[0]);
        assert_eq!(6, cand.product_surfaces[0]);

        for j in 1..candidate_seq_len {
            assert_eq!(
                0i16, cand.categorical_features[j],
                "padding slot {j} should be zero"
            );
            assert_eq!(
                0i32, cand.product_surfaces[j],
                "padding product_surfaces[{j}] should be zero"
            );
        }
    }

    fn test_user_action_sequence(actions: &[(u64, u64)]) -> pb::UserActionSequence {
        use pb::user_action_sequence_data_container::Data;
        let aggregated_user_actions = actions
            .iter()
            .map(|&(tweet_id, author_id)| pb::AggregatedUserAction {
                tweet_info: Some(pb::TweetInfo {
                    tweet_id,
                    author_id,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect();
        pb::UserActionSequence {
            user_actions_data: Some(pb::UserActionSequenceDataContainer {
                data: Some(Data::OrderedAggregatedUserActionsList(
                    pb::AggregatedUserActionList {
                        aggregated_user_actions,
                        ..Default::default()
                    },
                )),
            }),
            ..Default::default()
        }
    }

    fn test_history_model_config() -> ModelConfig {
        let mut model_config = test_model_config(1);
        model_config.hash_table.num_user_categorical_features = 7;
        model_config.hash_table.num_user_float_features = 4;
        model_config
    }

    fn test_num_history_for(actions: &[(u64, u64)]) -> usize {
        let sequence = Some(test_user_action_sequence(actions));
        InputBuffer::compute_for_item(
            &test_history_model_config(),
            &sequence,
            &pb::CandidateSet::default(),
            None,
            None,
            None,
            None,
        )
        .num_history
    }

    #[test]
    fn num_history_counts_only_real_pre_padding_positions() {
        let num_history =
            test_num_history_for(&[(10, 20), (11, 21), (12, 22), (0, 23), (13, 24), (14, 0)]);
        assert_eq!(2, num_history);
    }

    #[test]
    fn num_history_caps_at_history_seq_len() {
        let num_history = test_num_history_for(&[(10, 20), (11, 21), (12, 22), (13, 23), (14, 24)]);
        assert_eq!(4, num_history);
    }

    #[test]
    fn num_history_zero_when_no_sequence() {
        let buf = InputBuffer::compute_for_item(
            &test_history_model_config(),
            &None,
            &pb::CandidateSet::default(),
            None,
            None,
            None,
            None,
        );
        assert_eq!(0, buf.num_history);
    }

    #[test]
    fn stamp_engagement_counts_writes_correct_indices() {
        let num_features = VIEW_COUNT_SEQ + 1;
        let mut dest = vec![0i64; 2 * num_features];

        stamp_engagement_counts(&mut dest, num_features, 1, 11, 22, 33, 44, 55);

        let base = num_features;
        assert_eq!(11, dest[base + FAV_COUNT_SEQ]);
        assert_eq!(22, dest[base + REPLY_COUNT_SEQ]);
        assert_eq!(33, dest[base + REPOST_COUNT_SEQ]);
        assert_eq!(44, dest[base + QUOTE_COUNT_SEQ]);
        assert_eq!(55, dest[base + VIEW_COUNT_SEQ]);

        assert!(dest[0..num_features].iter().all(|&v| v == 0));
    }

    #[test]
    fn stamp_engagement_counts_noop_when_num_features_too_small() {
        let mut dest: Vec<i64> = vec![];
        stamp_engagement_counts(&mut dest, 0, 0, 1, 2, 3, 4, 5);
        assert!(dest.is_empty());
    }

    #[test]
    fn stamp_engagement_counts_saturates_to_i32_max() {
        let num_features = VIEW_COUNT_SEQ + 1;
        let mut dest = vec![0i64; num_features];
        let huge = 3_000_000_000u64;

        stamp_engagement_counts(&mut dest, num_features, 0, 10, 20, 30, 40, huge);

        let i32_max = i32::MAX as i64;
        assert_eq!(10, dest[FAV_COUNT_SEQ]);
        assert_eq!(40, dest[QUOTE_COUNT_SEQ]);
        assert_eq!(i32_max, dest[VIEW_COUNT_SEQ]);
        assert!(
            dest[VIEW_COUNT_SEQ] as i32 > 0,
            "must not wrap negative as i32"
        );
    }

    #[test]
    fn engagement_counts_stamped_into_candidate_int64_features() {
        let n_post_cat = 1;
        let mut model_config = test_model_config(n_post_cat);
        let n_post_int64 = VIEW_COUNT_SEQ + 1;
        model_config.hash_table.num_post_int64_features = n_post_int64;
        let candidate_seq_len = model_config.candidate_seq_len;

        let mut candidate_set = pb::CandidateSet {
            product_surface: pb::ProductSurface::SearchResultsPage as i32,
            ..Default::default()
        };
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: 1000,
            author_id: 2000,
            fav_count: 7,
            reply_count: 8,
            retweet_count: 9,
            quote_count: 10,
            view_count: 11,
            ..Default::default()
        });

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        assert_eq!(7, cand.int64_features[FAV_COUNT_SEQ]);
        assert_eq!(8, cand.int64_features[REPLY_COUNT_SEQ]);
        assert_eq!(9, cand.int64_features[REPOST_COUNT_SEQ]);
        assert_eq!(10, cand.int64_features[QUOTE_COUNT_SEQ]);
        assert_eq!(11, cand.int64_features[VIEW_COUNT_SEQ]);

        for j in 1..candidate_seq_len {
            let base = j * n_post_int64;
            for f in 0..n_post_int64 {
                assert_eq!(
                    0,
                    cand.int64_features[base + f],
                    "padding candidate {j} slot {f} should be zero"
                );
            }
        }
    }

    #[cfg(recsys_ads_dpa)]
    #[test]
    fn hash_dpa_product_key_matches_training_numpy() {
        let cases: [(i64, i64); 7] = [
            (0, 0),
            (1, 2_647_763),
            (42, 5_063_774),
            (999_999_937, 57_483),
            (1_780_416_091_123_456_789, 6_745_275),
            (i64::MAX, 5_447_293),
            (-5, 3_562_486),
        ];
        for (raw, expected) in cases {
            assert_eq!(expected, hash_dpa_product_key(raw), "key {raw}");
        }
        let cases_2: [(i64, i64); 7] = [
            (0, 0),
            (1, 7_661_376),
            (42, 2_595_051),
            (999_999_937, 9_064_104),
            (1_780_416_091_123_456_789, 20_492),
            (i64::MAX, 6_508_045),
            (-5, 4_201_377),
        ];
        for (raw, expected) in cases_2 {
            assert_eq!(expected, hash_dpa_product_key_2(raw), "key {raw} (hash 2)");
        }
        for raw in [1i64, -1, i64::MAX, i64::MIN] {
            for (h, which) in [
                (hash_dpa_product_key(raw), "hash1"),
                (hash_dpa_product_key_2(raw), "hash2"),
            ] {
                assert!(
                    (1..ADS_PRODUCT_KEY_TABLE_SIZE).contains(&h),
                    "key {raw} {which} -> {h}, outside [1, table_size)"
                );
            }
        }
    }

    #[cfg(recsys_ads_dpa)]
    #[test]
    fn first_dpa_product_key_stamped_into_candidate_int64_features() {
        let n_post_cat = 1;
        let mut model_config = test_model_config(n_post_cat);
        let n_post_int64 = FIRST_DPA_PRODUCT_KEY_HASH2 + 1;
        model_config.hash_table.num_post_int64_features = n_post_int64;

        let mut candidate_set = pb::CandidateSet {
            product_surface: pb::ProductSurface::SearchResultsPage as i32,
            ..Default::default()
        };
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: 1000,
            author_id: 2000,
            ad_info: Some(pb::AdInfo {
                product_keys: vec![1_780_416_091_123_456_789, 222],
                ..Default::default()
            }),
            ..Default::default()
        });
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: 1001,
            author_id: 2001,
            ad_info: Some(pb::AdInfo::default()),
            ..Default::default()
        });
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: 1002,
            author_id: 2002,
            ..Default::default()
        });

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        assert_eq!(
            hash_dpa_product_key(1_780_416_091_123_456_789),
            cand.int64_features[FIRST_DPA_PRODUCT_KEY],
            "DPA candidate should carry the hashed first product key"
        );
        assert_eq!(
            hash_dpa_product_key_2(1_780_416_091_123_456_789),
            cand.int64_features[FIRST_DPA_PRODUCT_KEY_HASH2],
            "DPA candidate should carry the second hash of the product key"
        );
        assert_ne!(0, cand.int64_features[FIRST_DPA_PRODUCT_KEY]);
        assert_ne!(0, cand.int64_features[FIRST_DPA_PRODUCT_KEY_HASH2]);
        assert_ne!(
            cand.int64_features[FIRST_DPA_PRODUCT_KEY],
            cand.int64_features[FIRST_DPA_PRODUCT_KEY_HASH2]
        );
        for slot in [FIRST_DPA_PRODUCT_KEY, FIRST_DPA_PRODUCT_KEY_HASH2] {
            assert_eq!(0, cand.int64_features[n_post_int64 + slot]);
            assert_eq!(0, cand.int64_features[2 * n_post_int64 + slot]);
        }
    }

    #[test]
    fn stale_post_14d_zeroes_and_flags_old_candidate() {
        let n_post_cat = 1;
        let mut model_config = test_model_config(n_post_cat);
        let n_post_int64 = VIEW_COUNT_SEQ + 1;
        model_config.hash_table.num_post_int64_features = n_post_int64;
        model_config.hash_table.num_post_bool_features = IS_STALE_POST14D + 1;
        model_config.hash_table.enable_stale_post = true;
        let n_post_bool = model_config.hash_table.num_post_bool_features;

        let now_ms = now_epoch_sec() as i64 * 1000;
        let tweet_id_for_age_h = |age_h: i64| -> u64 {
            let creation_ms = now_ms - age_h * 3600 * 1000;
            (((creation_ms - TWITTER_EPOCH_MS) << 22) as u64) & !((1u64 << 22) - 1)
        };

        let mut candidate_set = pb::CandidateSet::default();
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: tweet_id_for_age_h(400),
            author_id: 2000,
            fav_count: 7,
            reply_count: 8,
            retweet_count: 9,
            quote_count: 10,
            view_count: 11,
            ..Default::default()
        });
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: tweet_id_for_age_h(1),
            author_id: 2001,
            fav_count: 1,
            reply_count: 2,
            retweet_count: 3,
            quote_count: 4,
            view_count: 5,
            ..Default::default()
        });

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        assert_eq!(0, cand.int64_features[FAV_COUNT_SEQ]);
        assert_eq!(0, cand.int64_features[REPLY_COUNT_SEQ]);
        assert_eq!(0, cand.int64_features[REPOST_COUNT_SEQ]);
        assert_eq!(0, cand.int64_features[QUOTE_COUNT_SEQ]);
        assert_eq!(0, cand.int64_features[VIEW_COUNT_SEQ]);
        assert!(cand.bool_features[IS_STALE_POST14D]);

        let base1 = n_post_int64;
        assert_eq!(1, cand.int64_features[base1 + FAV_COUNT_SEQ]);
        assert_eq!(5, cand.int64_features[base1 + VIEW_COUNT_SEQ]);
        assert!(!cand.bool_features[n_post_bool + IS_STALE_POST14D]);
    }
}
