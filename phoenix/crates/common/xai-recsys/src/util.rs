// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::model_config::ModelConfig;
use arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int16Array, Int32Array, Int64Array,
};
use arrow::ipc::reader::StreamReader;
use half::f16;
use lazy_static::lazy_static;
use prometheus::{IntCounterVec, register_int_counter_vec};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use xai_recsys_proto as pb;

lazy_static! {
    pub(crate) static ref USER_FEATURE_FILL_TOTAL: IntCounterVec = register_int_counter_vec!(
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

    static ref COLUMNAR_COLUMN_MISTYPED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "recsys_columnar_column_mistyped_total",
        "Columnar request columns skipped because they had an unexpected Arrow type.",
        &["column"]
    )
    .unwrap();
}

pub const WEB_CONV_FAKE_TWEET_ID: i64 = 4;

pub fn is_web_conv_row(tweet_id: i64, has_conv_bit: bool) -> bool {
    tweet_id == WEB_CONV_FAKE_TWEET_ID || has_conv_bit
}

pub use xai_user_debias::HISTORY_STATS_DIM as USER_HISTORY_STATS_DIM;

pub const WEB_CONV_URL_ID_COLUMN: &str = "webConvUrlId";
pub const WEB_CONV_RANKED_URL_ID_COLUMNS: [&str; 3] =
    ["webConvUrlIdRanked", "webConvUrlId2", "webConvUrlId3"];

const FIRST_VIDEO_DURATION_MS_COLUMN: &str = "firstVideoDurationMs";

fn stamp_video_duration(
    model_config: &ModelConfig,
    history_int64_features: &mut [i64],
    row: usize,
    duration_ms: i64,
) {
    let n = model_config.hash_table.num_post_int64_features;
    if model_config.hash_table.video_completion_history
        && n > int64_feature::FIRST_VIDEO_DURATION_MS_SEQ
    {
        history_int64_features[row * n + int64_feature::FIRST_VIDEO_DURATION_MS_SEQ] =
            duration_ms.clamp(0, i32::MAX as i64);
    }
}

pub fn conv_asset_map(ids: Option<&pb::ConvAssetIds>) -> HashMap<(i64, i64), i64> {
    ids.filter(|ids| {
        ids.asset_id.len() == ids.author_id.len()
            && ids.asset_id.len() == ids.impressed_time_ms.len()
    })
    .map(|ids| {
        (0..ids.asset_id.len())
            .filter(|&i| ids.asset_id[i] > 0)
            .map(|i| {
                (
                    (ids.author_id[i], ids.impressed_time_ms[i]),
                    ids.asset_id[i],
                )
            })
            .collect()
    })
    .unwrap_or_default()
}

pub fn conv_asset_ids_for_batch(
    batch: &arrow::record_batch::RecordBatch,
    map: &HashMap<(i64, i64), i64>,
) -> Vec<i64> {
    let n = batch.num_rows();
    let mut out = vec![0i64; n];
    if map.is_empty() {
        return out;
    }
    let col_i64 = |name: &str| {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
    };
    let (Some(tweet_ids), Some(author_ids), Some(impressed_ms)) = (
        col_i64("tweetId"),
        col_i64("authorId"),
        col_i64("impressedTimeMs"),
    ) else {
        return out;
    };
    let conv_bit = pb::ActionName::AdsWebConversion as usize;
    let multi_hot = batch
        .column_by_name("actionNameMultiHot")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
        .filter(|fsl| conv_bit < fsl.value_length() as usize);
    let (bools, vocab) = match multi_hot {
        Some(fsl) => (
            fsl.values().as_any().downcast_ref::<BooleanArray>(),
            fsl.value_length() as usize,
        ),
        None => (None, 0),
    };
    for (row, slot) in out.iter_mut().enumerate() {
        let has_conv_bit = bools.is_some_and(|b| b.value(row * vocab + conv_bit));
        if is_web_conv_row(tweet_ids.value(row), has_conv_bit)
            && let Some(&asset_id) = map.get(&(author_ids.value(row), impressed_ms.value(row)))
        {
            *slot = asset_id;
        }
    }
    out
}

pub fn record_sid_coverage(sequence: &str, present: u64, count: u64) {
    SID_COVERAGE_TOTAL
        .with_label_values(&[sequence, "present"])
        .inc_by(present);
    SID_COVERAGE_TOTAL
        .with_label_values(&[sequence, "missing"])
        .inc_by(count.saturating_sub(present));
}

fn typed_column<'a, T: 'static>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Option<&'a T> {
    let typed = batch.column_by_name(name)?.as_any().downcast_ref::<T>();
    if typed.is_none() {
        COLUMNAR_COLUMN_MISTYPED_TOTAL
            .with_label_values(&[name])
            .inc();
    }
    typed
}

pub fn stamp_semantic_ids(dst: &mut [u16], entry_idx: usize, sid_num_levels: usize, codes: &[i32]) {
    if sid_num_levels > 0 && codes.len() == sid_num_levels {
        let base = entry_idx * sid_num_levels;
        for (d, &c) in dst[base..base + sid_num_levels]
            .iter_mut()
            .zip(codes.iter())
        {
            *d = (c + 1) as u16;
        }
    }
}

use crate::feature_config::bool_feature::{
    IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ, IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ_COLUMN,
    IS_AUTHOR_FOLLOWING_VIEWER_SEQ, IS_AUTHOR_FOLLOWING_VIEWER_SEQ_COLUMN, IS_STALE_POST14D,
};
use crate::feature_config::categorical_feature::{
    AUTHOR_IS_NSFW_SEQ, EXACT_PHRASE_SEQ, LOCAL_DAY_OF_WEEK_SEQ, LOCAL_HOUR_OF_DAY_SEQ,
    PRODUCT_SURFACE_SEQ, PRODUCT_SURFACE_SEQ_COLUMN, TIMEZONE_SEQ,
    WEB_CONV_TRACKING_INTEGRATION_SEQ, WEB_CONV_TRACKING_INTEGRATION_SEQ_COLUMN,
};
#[cfg(recsys_ads_dpa)]
use crate::feature_config::constants::{
    ADS_PRODUCT_KEY_HASH_BIAS, ADS_PRODUCT_KEY_HASH_BIAS_2, ADS_PRODUCT_KEY_HASH_MODULUS,
    ADS_PRODUCT_KEY_HASH_SCALE, ADS_PRODUCT_KEY_HASH_SCALE_2, ADS_PRODUCT_KEY_TABLE_SIZE,
};
#[cfg(recsys_dpa_sid)]
use crate::feature_config::constants::{DPA_SID_CODEBOOK_SIZE, DPA_SID_NUM_LEVELS};
use crate::feature_config::constants::{STALE_POST_14D_TTL_SEC, STALE_POST_30D_TTL_SEC};
use crate::feature_config::float_feature::MATCHED_WORD_FRACTION_SEQ;
use crate::feature_config::int64_feature;
#[cfg(recsys_dpa_sid)]
use crate::feature_config::int64_feature::{
    DPA_SID_LEVEL0, DPA_SID_LEVEL1, DPA_SID_LEVEL2, DPA_SID_LEVEL3, DPA_SID_LEVEL4, DPA_SID_LEVEL5,
};
use crate::feature_config::int64_feature::{
    FAV_COUNT_SEQ, FAV_COUNT_SEQ_COLUMN, QUOTE_COUNT_SEQ, QUOTE_COUNT_SEQ_COLUMN, REPLY_COUNT_SEQ,
    REPLY_COUNT_SEQ_COLUMN, REPOST_COUNT_SEQ, REPOST_COUNT_SEQ_COLUMN, VIEW_COUNT_SEQ,
    VIEW_COUNT_SEQ_COLUMN, WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ,
    WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ_COLUMN, WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ,
    WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ_COLUMN,
};
#[cfg(recsys_ads_dpa)]
use crate::feature_config::int64_feature::{FIRST_DPA_PRODUCT_KEY, FIRST_DPA_PRODUCT_KEY_HASH2};
#[cfg(recsys_time_gap)]
use crate::feature_config::int64_feature::{
    LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ, REQUEST_TIME_SEC_SEQ,
};
#[cfg(recsys_notification_counts)]
use crate::feature_config::int64_feature::{
    NOT_INTERESTED_COUNT_SEQ, NOTIFICATION_CLICK_COUNT_SEQ, NOTIFICATION_DISMISS_COUNT_SEQ,
    NOTIFICATION_OPEN_COUNT_SEQ, NOTIFICATION_SEE_LESS_OFTEN_COUNT_SEQ,
    NOTIFICATION_SENT_COUNT_SEQ,
};
#[cfg(any(recsys_time_gap, recsys_dpa_sid, recsys_notification_counts))]
use crate::reply::is_reply_width;

pub const PRODUCT_SURFACE_CATEGORICAL_IDX: usize = PRODUCT_SURFACE_SEQ;
const TIMEZONE_IDX: usize = TIMEZONE_SEQ;
const LOCAL_HOUR_IDX: usize = LOCAL_HOUR_OF_DAY_SEQ;
const LOCAL_DOW_IDX: usize = LOCAL_DAY_OF_WEEK_SEQ;
pub const AUTHOR_IS_NSFW_CATEGORICAL_IDX: usize = AUTHOR_IS_NSFW_SEQ;
pub const EXACT_PHRASE_CATEGORICAL_IDX: usize = EXACT_PHRASE_SEQ;

const AUTHOR_NSFW_BIT: u64 = 2;

pub mod ad_side {
    const ID_BUCKETS: u64 = 1 << 19;
    const CARD_TYPES: i32 = 48;
    const MEDIA_TYPES: i32 = 8;
    const MEDIA_CONTENT_TYPE_UNKNOWN: i32 = 4;
    const CREATIVE_UNKNOWN: i32 = CARD_TYPES + MEDIA_CONTENT_TYPE_UNKNOWN;
    const OBJECTIVES: i64 = 64;
    const CREATIVE_UNLOGGED_START_MS: i64 = 1_787_788_800_000;
    const CREATIVE_UNLOGGED_END_MS: i64 = 1_789_833_600_000;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Field {
        Campaign,
        LineItem,
        Creative,
        Objective,
        UrlDomain,
    }

    impl std::str::FromStr for Field {
        type Err = String;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "campaign" => Ok(Self::Campaign),
                "line_item" => Ok(Self::LineItem),
                "creative" => Ok(Self::Creative),
                "objective" => Ok(Self::Objective),
                "url_domain" => Ok(Self::UrlDomain),
                other => Err(format!(
                    "ad-side field {other:?} cannot be served \
                     (expected campaign|line_item|creative|objective|url_domain)"
                )),
            }
        }
    }

    pub fn parse_fields<S: AsRef<str>>(names: &[S]) -> Result<Vec<Field>, String> {
        names.iter().map(|name| name.as_ref().parse()).collect()
    }

    pub fn hash_id(id: i64) -> i32 {
        if id == 0 {
            return 0;
        }
        let mut x = id as u64;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        (x % (ID_BUCKETS - 1)) as i32 + 1
    }

    pub fn creative_format(card_type: i32, media_type: i32, impressed_time_ms: i64) -> i32 {
        let card = card_type.clamp(0, CARD_TYPES - 1);
        let media = media_type.clamp(0, MEDIA_TYPES - 1);
        let unlogged =
            (CREATIVE_UNLOGGED_START_MS..CREATIVE_UNLOGGED_END_MS).contains(&impressed_time_ms);
        if card > 0 {
            card
        } else if media == 0 && unlogged {
            CREATIVE_UNKNOWN
        } else {
            CARD_TYPES + media
        }
    }

    pub fn objective(objective: i64) -> i32 {
        objective.clamp(0, OBJECTIVES - 1) as i32
    }
}

pub mod ads_counts {
    use xai_recsys_proto as pb;

    pub const COUNTERS: [i32; 7] = [1, 2, 14, 3, 4, 5, 8];
    pub const WINDOWS_SECS: [u32; 8] = [
        3_600, 10_800, 43_200, 86_400, 259_200, 604_800, 1_209_600, 2_419_200,
    ];
    pub const PER_GROUP: usize = COUNTERS.len() * WINDOWS_SECS.len();
    pub const NUM_FEATURES: usize = 3 * PER_GROUP;
    pub const CTR_NUM_FEATURES: usize = 3 * WINDOWS_SECS.len();
    pub const NUM_BUCKETS: usize = 33;
    pub const NUM_CTR_BUCKETS: usize = 16;
    pub const VIEW_COUNTER: i32 = 1;
    pub const CLICK_COUNTER: i32 = 2;

    pub fn ctr_bucket(clicks: u32, views: u32) -> u8 {
        if views == 0 {
            0
        } else if clicks == 0 {
            1
        } else {
            let bps = (clicks as u64 * 10_000 / views as u64).max(1);
            (2 + bps.ilog2() as u8).min((NUM_CTR_BUCKETS - 1) as u8)
        }
    }

    fn count_in(counts: &pb::EngagementCounts, counter: i32, window_secs: u32) -> Option<u32> {
        if !counts.present {
            return None;
        }
        if !counts.collected_actions.is_empty() && !counts.collected_actions.contains(&counter) {
            return None;
        }
        let col = counts.window_secs.iter().position(|s| *s == window_secs)?;
        Some(
            counts
                .rows
                .iter()
                .find(|r| r.action == counter)
                .and_then(|r| r.counts.get(col).copied())
                .unwrap_or(0),
        )
    }

    pub fn fill_ctr(counts: Option<&pb::EngagementCounts>, out: &mut [u8]) {
        let Some(c) = counts.filter(|c| c.present) else {
            return;
        };
        for (w, secs) in WINDOWS_SECS.iter().enumerate() {
            let Some(views) = count_in(c, VIEW_COUNTER, *secs) else {
                continue;
            };
            let clicks = count_in(c, CLICK_COUNTER, *secs).unwrap_or(0);
            out[w] = ctr_bucket(clicks, views);
        }
    }

    pub fn bucket(count: u32) -> u8 {
        (u32::BITS - count.leading_zeros()) as u8
    }

    pub fn fill_group(counts: Option<&pb::EngagementCounts>, out: &mut [u8]) {
        let Some(c) = counts.filter(|c| c.present) else {
            return;
        };
        for (w, secs) in WINDOWS_SECS.iter().enumerate() {
            let Some(col) = c.window_secs.iter().position(|s| s == secs) else {
                continue;
            };
            for (f, counter) in COUNTERS.iter().enumerate() {
                if !c.collected_actions.is_empty() && !c.collected_actions.contains(counter) {
                    continue;
                }
                let count = c
                    .rows
                    .iter()
                    .find(|r| r.action == *counter)
                    .and_then(|r| r.counts.get(col).copied())
                    .unwrap_or(0);
                out[f * WINDOWS_SECS.len() + w] = bucket(count);
            }
        }
    }

    #[cfg(test)]
    mod ctr_tests {
        use super::*;

        #[test]
        fn ctr_bucket_marks_zero_click_and_missing() {
            assert_eq!(ctr_bucket(0, 0), 0);
            assert_eq!(ctr_bucket(0, 84), 1);
            assert_eq!(ctr_bucket(1, 10_000), 2);
            assert_eq!(ctr_bucket(1, 1), 15);
        }
    }
}

pub mod history_display_location {
    use xai_recsys_proto::ProductSurface;

    pub const COLUMN: &str = "displayLocation";

    const SEARCH_TWEETS: i16 = 1;
    const TIMELINE_HOME: i16 = 2;
    const PROFILE_TWEETS: i16 = 9;
    const TWEET_CONVERSATION: i16 = 28;
    const TIMELINE_HOME_REVERSE_CHRON: i16 = 29;
    const MEDIA_VIEWER: i16 = 30;

    pub fn product_surface(display_location: i16) -> i32 {
        let surface = match display_location {
            SEARCH_TWEETS => ProductSurface::SearchResultsPage,
            TIMELINE_HOME => ProductSurface::HomeTimelineRanking,
            PROFILE_TWEETS => ProductSurface::ProfilePage,
            TWEET_CONVERSATION => ProductSurface::TweetDetailsPage,
            TIMELINE_HOME_REVERSE_CHRON => ProductSurface::HomeTimelineLatest,
            MEDIA_VIEWER => ProductSurface::GalleryPage,
            _ => ProductSurface::Unknown,
        };
        surface as i32
    }

    pub fn overlay(
        surfaces: &mut [i32],
        display_locations: &[i16],
        num_history: usize,
    ) -> Vec<bool> {
        let mut ad_rows = vec![false; surfaces.len()];
        for (i, (surface, ad_row)) in surfaces
            .iter_mut()
            .zip(ad_rows.iter_mut())
            .take(num_history)
            .enumerate()
        {
            if *surface == 0 {
                *ad_row = true;
                *surface = product_surface(display_locations.get(i).copied().unwrap_or(0));
            }
        }
        ad_rows
    }
}

fn exact_phrase_categorical(lexical_match: Option<&pb::SearchLexicalMatch>) -> i32 {
    match lexical_match {
        Some(m) => 1 + m.exact_phrase as i32,
        None => 0,
    }
}

fn stamp_matched_word_fraction(
    dest: &mut [f32],
    num_features: usize,
    entry_idx: usize,
    lexical_match: Option<&pb::SearchLexicalMatch>,
) {
    if let Some(m) = lexical_match
        && num_features > MATCHED_WORD_FRACTION_SEQ
    {
        dest[entry_idx * num_features + MATCHED_WORD_FRACTION_SEQ] = m.matched_word_fraction;
    }
}

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

#[cfg(recsys_notification_counts)]
fn stamp_notification_counts(
    dest: &mut [i64],
    num_features: usize,
    entry_idx: usize,
    not_interested: u64,
    sent: u64,
    open: u64,
    click: u64,
    dismiss: u64,
    see_less_often: u64,
) {
    if is_reply_width(num_features) {
        return;
    }
    const INT32_MAX: u64 = i32::MAX as u64;
    let base = entry_idx * num_features;
    if num_features > NOT_INTERESTED_COUNT_SEQ {
        dest[base + NOT_INTERESTED_COUNT_SEQ] = not_interested.min(INT32_MAX) as i64;
    }
    if num_features > NOTIFICATION_SENT_COUNT_SEQ {
        dest[base + NOTIFICATION_SENT_COUNT_SEQ] = sent.min(INT32_MAX) as i64;
    }
    if num_features > NOTIFICATION_OPEN_COUNT_SEQ {
        dest[base + NOTIFICATION_OPEN_COUNT_SEQ] = open.min(INT32_MAX) as i64;
    }
    if num_features > NOTIFICATION_CLICK_COUNT_SEQ {
        dest[base + NOTIFICATION_CLICK_COUNT_SEQ] = click.min(INT32_MAX) as i64;
    }
    if num_features > NOTIFICATION_DISMISS_COUNT_SEQ {
        dest[base + NOTIFICATION_DISMISS_COUNT_SEQ] = dismiss.min(INT32_MAX) as i64;
    }
    if num_features > NOTIFICATION_SEE_LESS_OFTEN_COUNT_SEQ {
        dest[base + NOTIFICATION_SEE_LESS_OFTEN_COUNT_SEQ] = see_less_often.min(INT32_MAX) as i64;
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

pub(crate) fn snowflake_to_creation_ts_sec(tweet_id: i64) -> i32 {
    if tweet_id == 0 {
        return 0;
    }
    let created_ms = (tweet_id >> 22) + TWITTER_EPOCH_MS;
    (created_ms / 1000) as i32
}

thread_local! {
    static FIXED_NOW_SEC: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

pub fn with_fixed_now_sec<R>(now: i32, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<i32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FIXED_NOW_SEC.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FIXED_NOW_SEC.with(|c| c.replace(Some(now))));
    f()
}

fn fixed_now_sec() -> Option<i32> {
    if let Some(ts) = FIXED_NOW_SEC.with(std::cell::Cell::get) {
        return Some(ts);
    }
    static FIXED: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    *FIXED.get_or_init(|| {
        std::env::var("RECSYS_FIXED_NOW_SEC")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
    })
}

fn candidate_impression_sec(candidate_set: &pb::CandidateSet) -> i32 {
    if let Some(ts) = fixed_now_sec() {
        return ts;
    }
    match candidate_set.request_timestamp_ms {
        Some(ms) if ms > 0 => (ms / 1000) as i32,
        _ => now_epoch_sec(),
    }
}

fn now_epoch_sec() -> i32 {
    if let Some(ts) = fixed_now_sec() {
        return ts;
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

#[cfg(recsys_time_gap)]
pub fn stamp_time_gap_timestamps(
    history_impr_ts: &[i32],
    candidate_impr_ts: &[i32],
    history_int64_features: &mut [i64],
    candidate_int64_features: &mut [i64],
    n_post_int64: usize,
) {
    if n_post_int64 <= LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ || is_reply_width(n_post_int64) {
        return;
    }
    let request_sec = candidate_impr_ts
        .iter()
        .filter(|&&ts| ts > 0)
        .min()
        .map_or(0, |&ts| ts as i64);
    let latest_history_sec = history_impr_ts
        .iter()
        .map(|&ts| ts as i64)
        .filter(|&ts| ts > 0 && ts <= request_sec)
        .max()
        .unwrap_or(0);
    for row in history_int64_features
        .chunks_exact_mut(n_post_int64)
        .take(history_impr_ts.len())
    {
        row[REQUEST_TIME_SEC_SEQ] = request_sec;
    }
    for (row, &ts) in candidate_int64_features
        .chunks_exact_mut(n_post_int64)
        .zip(candidate_impr_ts)
    {
        if ts > 0 {
            row[REQUEST_TIME_SEC_SEQ] = request_sec;
            row[LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ] = latest_history_sec;
        }
    }
}

#[cfg(recsys_dpa_sid)]
const DPA_SID_SLOTS: [usize; 6] = [
    DPA_SID_LEVEL0,
    DPA_SID_LEVEL1,
    DPA_SID_LEVEL2,
    DPA_SID_LEVEL3,
    DPA_SID_LEVEL4,
    DPA_SID_LEVEL5,
];

#[cfg(recsys_dpa_sid)]
const _: () = assert!(DPA_SID_SLOTS.len() as i64 == DPA_SID_NUM_LEVELS);

#[cfg(recsys_dpa_sid)]
fn stamp_dpa_sid_codes(dest: &mut [i64], n_post_int64: usize, entry_idx: usize, codes: &[i32]) {
    if codes.len() != DPA_SID_SLOTS.len()
        || n_post_int64 <= DPA_SID_SLOTS.into_iter().max().unwrap_or(0)
        || is_reply_width(n_post_int64)
    {
        return;
    }
    let base = entry_idx * n_post_int64;
    for (&slot, &code) in DPA_SID_SLOTS.iter().zip(codes) {
        dest[base + slot] = if code < 0 {
            0
        } else {
            (i64::from(code) + 1).min(DPA_SID_CODEBOOK_SIZE)
        };
    }
}

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

pub const CREAD_DECODE_MAX_POWER: f32 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreadDecodeKind {
    Power,
    Scale,
}

const CREAD_DECODE_KEYS: [(&str, CreadDecodeKind); 8] = [
    ("dwell_cread_power", CreadDecodeKind::Power),
    ("dwell_cread_scale", CreadDecodeKind::Scale),
    ("immersive_dwell_cread_power", CreadDecodeKind::Power),
    ("immersive_dwell_cread_scale", CreadDecodeKind::Scale),
    ("video_continuation_cread_power", CreadDecodeKind::Power),
    ("video_continuation_cread_scale", CreadDecodeKind::Scale),
    ("profile_visit_cread_power", CreadDecodeKind::Power),
    ("profile_visit_cread_scale", CreadDecodeKind::Scale),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreadDecodeInput(usize);

impl CreadDecodeInput {
    pub fn all() -> impl Iterator<Item = Self> {
        (0..CREAD_DECODE_KEYS.len()).map(Self)
    }

    pub fn parse(key: &str) -> Option<Self> {
        CREAD_DECODE_KEYS
            .iter()
            .position(|(k, _)| *k == key)
            .map(Self)
    }

    pub fn key(self) -> &'static str {
        CREAD_DECODE_KEYS[self.0].0
    }

    pub fn kind(self) -> CreadDecodeKind {
        CREAD_DECODE_KEYS[self.0].1
    }

    pub fn accepts(self, value: f32) -> bool {
        value.is_finite()
            && value > 0.0
            && (self.kind() == CreadDecodeKind::Scale || value <= CREAD_DECODE_MAX_POWER)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CreadDecodeOverride {
    values: [Option<f32>; CREAD_DECODE_KEYS.len()],
}

impl CreadDecodeOverride {
    pub fn get(&self, input: CreadDecodeInput) -> Option<f32> {
        self.values[input.0]
    }

    pub fn set(&mut self, input: CreadDecodeInput, value: f32) {
        self.values[input.0] = Some(value);
    }
}

#[derive(Default)]
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
    pub candidate_ads_count_buckets: Vec<u8>,
    pub candidate_ads_ctr_buckets: Vec<u8>,
    pub candidate_ad_side_ids: Vec<i32>,
    pub history_ad_row_mask: Vec<bool>,
    pub history_post_ids: Vec<i64>,
    pub history_web_conv_url_ids: Vec<(u32, i64)>,
    pub history_web_conv_ranked_url_ids: Vec<(u32, i64)>,
    pub history_url_embeddings: Vec<(u32, Arc<[f16]>)>,
    pub history_semantic_ids: Vec<u16>,
    pub candidate_semantic_ids: Vec<u16>,
    pub num_history: usize,
    pub cread_decode: CreadDecodeOverride,
    pub user_history_stats: Vec<f32>,
}

fn user_history_stats(
    model_config: &ModelConfig,
    history: Option<&arrow::record_batch::RecordBatch>,
    candidate_set: &pb::CandidateSet,
) -> Vec<f32> {
    if model_config.user_history_stats_dim == 0 {
        return Vec::new();
    }
    let mut rows = xai_user_debias::SequenceRows::default();
    let filled =
        history.is_some_and(|batch| xai_user_debias::arrow::append_batch(&mut rows, batch).is_ok());
    USER_FEATURE_FILL_TOTAL
        .with_label_values(&[
            "user_history_stats",
            if filled { "filled" } else { "missing" },
        ])
        .inc();
    if !filled {
        rows = xai_user_debias::SequenceRows::default();
    }
    let req_ms = i64::from(candidate_impression_sec(candidate_set)) * 1000;
    xai_user_debias::history_stats(&rows, req_ms)
}

pub fn repeat_query_into(dest: &mut [f32], query: &[f32], n_slots: usize) {
    let dim = query.len();
    if dim == 0 || dest.is_empty() || n_slots == 0 {
        return;
    }
    let max_slots = dest.len() / dim;
    let n = n_slots.min(max_slots);
    for i in 0..n {
        let off = i * dim;
        dest[off..off + dim].copy_from_slice(query);
    }
}

impl InputBuffer {
    pub fn num_real_candidates(&self, num_item_hashes: usize, cap: usize) -> usize {
        let w = num_item_hashes.max(1);
        self.candidate_post_hashes
            .chunks(w)
            .take(cap)
            .take_while(|ch| ch.first().is_some_and(|&h| h != 0))
            .count()
    }

    pub fn stamp_ads_count_buckets(
        &mut self,
        model_config: &ModelConfig,
        candidate_set: &pb::CandidateSet,
        user_counts: Option<&pb::EngagementCounts>,
    ) {
        use ads_counts::{NUM_FEATURES, PER_GROUP, fill_group};
        if !model_config.hash_table.ads_count_features {
            return;
        }
        let mut user = [0u8; PER_GROUP];
        fill_group(user_counts, &mut user);
        let c = model_config.candidate_seq_len;
        let mut out = vec![0u8; c * NUM_FEATURES];
        for (row, candidate) in out
            .chunks_exact_mut(NUM_FEATURES)
            .zip(candidate_set.candidates.iter())
        {
            let ad = candidate.ad_info.as_ref();
            let (ad_row, rest) = row.split_at_mut(PER_GROUP);
            let (user_advertiser_row, user_row) = rest.split_at_mut(PER_GROUP);
            fill_group(ad.and_then(|a| a.engagement_counts.as_ref()), ad_row);
            fill_group(
                ad.and_then(|a| a.viewer_advertiser_engagement_counts.as_ref()),
                user_advertiser_row,
            );
            user_row.copy_from_slice(&user);
        }
        self.candidate_ads_count_buckets = out;
    }

    pub fn stamp_ads_ctr_buckets(
        &mut self,
        model_config: &ModelConfig,
        candidate_set: &pb::CandidateSet,
    ) {
        use ads_counts::{CTR_NUM_FEATURES, WINDOWS_SECS, fill_ctr};
        if !model_config.hash_table.ads_ctr_features {
            return;
        }
        let c = model_config.candidate_seq_len;
        let mut out = vec![0u8; c * CTR_NUM_FEATURES];
        for (row, candidate) in out
            .chunks_exact_mut(CTR_NUM_FEATURES)
            .zip(candidate_set.candidates.iter())
        {
            let ad = candidate.ad_info.as_ref();
            let (ad_row, rest) = row.split_at_mut(WINDOWS_SECS.len());
            let (advertiser_row, user_ad_row) = rest.split_at_mut(WINDOWS_SECS.len());
            fill_ctr(ad.and_then(|a| a.engagement_counts.as_ref()), ad_row);
            fill_ctr(
                ad.and_then(|a| a.viewer_advertiser_engagement_counts.as_ref()),
                advertiser_row,
            );
            fill_ctr(
                ad.and_then(|a| a.viewer_ad_engagement_counts.as_ref()),
                user_ad_row,
            );
        }
        self.candidate_ads_ctr_buckets = out;
    }

    pub fn stamp_candidate_ad_side_ids(
        &mut self,
        model_config: &ModelConfig,
        candidate_set: &pb::CandidateSet,
        url_domain_id: impl Fn(&str) -> i64,
    ) {
        use ad_side::Field;
        let fields = &model_config.hash_table.ad_side_fields;
        if fields.is_empty() {
            return;
        }
        let num_item_hashes = model_config.hash_table.num_item_hashes().max(1);
        let mut out = vec![0i32; model_config.candidate_seq_len * fields.len()];
        for (j, (row, candidate)) in out
            .chunks_exact_mut(fields.len())
            .zip(&candidate_set.candidates)
            .enumerate()
        {
            if self
                .candidate_post_hashes
                .get(j * num_item_hashes)
                .is_none_or(|&h| h == 0)
            {
                continue;
            }
            let ad = candidate.ad_info.as_ref();
            let impressed_time_ms = i64::from(self.candidate_impr_ts[j]) * 1000;
            for (id, field) in row.iter_mut().zip(fields) {
                *id = match field {
                    Field::Campaign => ad_side::hash_id(self.candidate_campaign_ids[j]),
                    Field::LineItem => ad_side::hash_id(self.candidate_line_item_ids[j]),
                    Field::Creative => ad_side::creative_format(
                        ad.map_or(0, |a| a.unified_card_type),
                        ad.map_or(0, |a| a.media_content_type),
                        impressed_time_ms,
                    ),
                    Field::Objective => ad_side::objective(ad.map_or(0, |a| a.line_item_objective)),
                    Field::UrlDomain => {
                        ad_side::hash_id(url_domain_id(ad.map_or("", |a| &a.url_domain)))
                    }
                };
            }
        }
        self.candidate_ad_side_ids = out;
    }

    pub fn drop_recent_history(
        &mut self,
        model_config: &ModelConfig,
        rate: f64,
        window_secs: i64,
        mut uniform: impl FnMut() -> f64,
    ) -> usize {
        let h = model_config.history_seq_len;
        let n_item = model_config.hash_table.num_item_hashes().max(1);
        let real = |hashes: &[i32], row: usize| hashes.get(row * n_item).is_some_and(|&x| x != 0);
        let Some(request_sec) = (0..self.candidate_impr_ts.len())
            .filter(|&c| real(&self.candidate_post_hashes, c) && self.candidate_impr_ts[c] > 0)
            .map(|c| i64::from(self.candidate_impr_ts[c]))
            .min()
        else {
            return 0;
        };
        if self.history_impr_ts.len() < h {
            return 0;
        }
        let n_actions = self.history_actions.len() / h;
        let negative = |row: usize| {
            NEGATIVE_FEEDBACK_ACTIONS.iter().any(|&a| {
                (a as usize) < n_actions && self.history_actions[row * n_actions + a as usize]
            })
        };
        let keep: Vec<bool> = (0..h)
            .map(|t| {
                let recent = real(&self.history_post_hashes, t)
                    && i64::from(self.history_impr_ts[t]) > request_sec - window_secs;
                !(recent && !negative(t) && uniform() < rate)
            })
            .collect();
        let removed = keep.iter().filter(|&&k| !k).count();
        if removed == 0 {
            return 0;
        }
        compact_rows(&mut self.history_post_hashes, &keep);
        compact_rows(&mut self.history_auth_hashes, &keep);
        compact_rows(&mut self.history_product_surfaces, &keep);
        compact_rows(&mut self.history_ad_row_mask, &keep);
        compact_rows(&mut self.history_actions, &keep);
        compact_rows(&mut self.history_continuous_actions, &keep);
        compact_rows(&mut self.history_categorical_features, &keep);
        compact_rows(&mut self.history_bool_features, &keep);
        compact_rows(&mut self.history_float_features, &keep);
        compact_rows(&mut self.history_int64_features, &keep);
        compact_rows(&mut self.history_impr_ts, &keep);
        compact_rows(&mut self.history_post_creation_ts_sec, &keep);
        compact_rows(&mut self.history_post_ids, &keep);
        compact_rows(&mut self.history_semantic_ids, &keep);
        let new_row: Vec<u32> = keep
            .iter()
            .scan(0u32, |next, &k| {
                let row = *next;
                *next += u32::from(k);
                Some(row)
            })
            .collect();
        let remap = |(row, _): &mut (u32, i64)| {
            let r = *row as usize;
            let kept = keep.get(r).copied().unwrap_or(false);
            if kept {
                *row = new_row[r];
            }
            kept
        };
        self.history_web_conv_url_ids.retain_mut(remap);
        self.history_web_conv_ranked_url_ids.retain_mut(remap);
        self.num_history = self.num_history.saturating_sub(removed);
        removed
    }
}

pub const NEGATIVE_FEEDBACK_ACTIONS: [pb::ActionName; 10] = [
    pb::ActionName::ServerTweetReport,
    pb::ActionName::ClientTweetReport,
    pb::ActionName::ClientTweetNotInterestedIn,
    pb::ActionName::ClientTweetSeeFewer,
    pb::ActionName::ClientTweetUnfollowAuthor,
    pb::ActionName::ClientTweetBlockAuthor,
    pb::ActionName::ClientTweetMuteAuthor,
    pb::ActionName::ClientTweetMuteConversation,
    pb::ActionName::ClientTweetNotRelevant,
    pb::ActionName::ClientNotificationSeeLessOften,
];

fn compact_rows<T: Copy + Default>(buf: &mut [T], keep: &[bool]) {
    if buf.is_empty() {
        return;
    }
    debug_assert_eq!(buf.len() % keep.len(), 0);
    let width = buf.len() / keep.len();
    let mut dst = 0;
    for (row, _) in keep.iter().enumerate().filter(|&(_, &k)| k) {
        if dst != row {
            buf.copy_within(row * width..(row + 1) * width, dst * width);
        }
        dst += 1;
    }
    buf[dst * width..].fill(T::default());
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
        let search_query_embedding_dim = model_config.search_query_embedding_dim;

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

        let now_sec = candidate_impression_sec(candidate_set);

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

            stamp_semantic_ids(
                &mut candidate_semantic_ids,
                j,
                sid_num_levels,
                &candidate.semantic_ids,
            );
        }

        if sid_num_levels > 0 {
            let present = (0..candidates_to_process)
                .filter(|&j| candidate_semantic_ids[j * sid_num_levels] != 0)
                .count() as u64;
            record_sid_coverage("candidate", present, candidates_to_process as u64);
        }

        let candidate_search_query_embeddings = if search_query_embedding_dim > 0
            && candidate_set.search_query_embedding.len() == search_query_embedding_dim
        {
            candidate_set.search_query_embedding.clone()
        } else {
            if search_query_embedding_dim > 0 && !candidate_set.search_query_embedding.is_empty() {
                log::error!(
                    "search_query_embedding dim {} != model {search_query_embedding_dim}; leaving empty",
                    candidate_set.search_query_embedding.len()
                );
            }
            Vec::new()
        };

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
        let mut candidate_float_features = vec![0.0f32; candidate_seq_len * n_post_float];
        let mut candidate_is_author_followed = vec![false; candidate_seq_len];
        let mut candidate_is_author_following = vec![false; candidate_seq_len];
        let mut candidate_is_stale_post = vec![false; candidate_seq_len];
        let mut candidate_exact_phrase = vec![0i32; candidate_seq_len];
        let mut candidate_bool_features = vec![false; candidate_seq_len * n_post_bool];

        let stale_post_enabled = model_config.hash_table.enable_stale_post;
        let stale_post_ttl_sec = if model_config.hash_table.enable_stale_post_30d {
            STALE_POST_30D_TTL_SEC
        } else {
            STALE_POST_14D_TTL_SEC
        };
        for (j, candidate) in candidate_set
            .candidates
            .iter()
            .take(candidates_to_process)
            .enumerate()
        {
            let creation_valid = candidate_post_creation_ts_sec[j] > 0;
            let original_age_sec = now_sec as i64 - candidate_post_creation_ts_sec[j] as i64;
            let is_stale =
                stale_post_enabled && creation_valid && original_age_sec > stale_post_ttl_sec;
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
            #[cfg(recsys_notification_counts)]
            stamp_notification_counts(
                &mut candidate_int64_features,
                n_post_int64,
                j,
                candidate.not_interested_count,
                candidate.notification_sent_count,
                candidate.notification_open_count,
                candidate.notification_click_count,
                candidate.notification_dismiss_count,
                candidate.notification_see_less_often_count,
            );

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
            #[cfg(recsys_dpa_sid)]
            stamp_dpa_sid_codes(
                &mut candidate_int64_features,
                n_post_int64,
                j,
                candidate
                    .ad_info
                    .as_ref()
                    .and_then(|a| a.dpa_product_info.as_ref())
                    .map_or(&[][..], |info| info.sid_codes.as_slice()),
            );
            candidate_is_author_followed[j] = candidate.is_author_followed_by_user;
            candidate_is_author_following[j] = candidate
                .author_info
                .as_ref()
                .and_then(|ai| ai.is_following_user)
                .unwrap_or(false);
            let lexical_match = candidate.search_lexical_match.as_ref();
            candidate_exact_phrase[j] = exact_phrase_categorical(lexical_match);
            stamp_matched_word_fraction(
                &mut candidate_float_features,
                n_post_float,
                j,
                lexical_match,
            );
        }

        stamp_i32_as_categorical(
            &candidate_exact_phrase,
            &mut categorical_features,
            n_post_cat,
            EXACT_PHRASE_CATEGORICAL_IDX,
        );

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
            float_features: candidate_float_features,
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
            categorical_features[USER_GENDER] = ctx.user_gender as i16;
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
        let sid_num_levels = model_config.sid_num_levels;
        let mut history_semantic_ids = vec![0u16; history_seq_len * sid_num_levels];

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

                stamp_semantic_ids(
                    &mut history_semantic_ids,
                    valid_entry_count,
                    sid_num_levels,
                    &tweet_info.semantic_ids,
                );

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
                stamp_video_duration(
                    model_config,
                    &mut history_int64_features,
                    valid_entry_count,
                    tweet_info.first_video_duration_ms.min(i32::MAX as u64) as i64,
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

        let history_ad_row_mask = if model_config.hash_table.history_display_location {
            history_display_location::overlay(&mut history_product_surfaces, &[], valid_entry_count)
        } else {
            Vec::new()
        };

        #[cfg(recsys_time_gap)]
        let mut candidate_int64_features = candidate_int64_features;
        #[cfg(recsys_time_gap)]
        stamp_time_gap_timestamps(
            &history_impr_ts[..valid_entry_count],
            &candidate_impr_ts,
            &mut history_int64_features,
            &mut candidate_int64_features,
            n_post_int64,
        );

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
            candidate_ads_count_buckets: Vec::new(),
            candidate_ads_ctr_buckets: Vec::new(),
            candidate_ad_side_ids: Vec::new(),
            history_ad_row_mask,
            history_post_ids,
            history_web_conv_url_ids: Vec::new(),
            history_web_conv_ranked_url_ids: Vec::new(),
            history_url_embeddings: Vec::new(),
            history_semantic_ids,
            candidate_semantic_ids,
            num_history: valid_entry_count,
            cread_decode: CreadDecodeOverride::default(),
            user_history_stats: user_history_stats(model_config, None, candidate_set),
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
        let mut history_web_conv_tracking_integration = vec![0i32; history_seq_len];
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
                    candidate_ads_count_buckets: Vec::new(),
                    candidate_ads_ctr_buckets: Vec::new(),
                    candidate_ad_side_ids: Vec::new(),
                    history_ad_row_mask: if model_config.hash_table.history_display_location {
                        vec![false; history_seq_len]
                    } else {
                        Vec::new()
                    },
                    history_post_ids: vec![0i64; history_seq_len],
                    history_web_conv_url_ids: Vec::new(),
                    history_web_conv_ranked_url_ids: Vec::new(),
                    history_url_embeddings: Vec::new(),
                    history_semantic_ids: vec![0u16; history_seq_len * sid_num_levels],
                    candidate_semantic_ids,
                    num_history: 0,
                    cread_decode: CreadDecodeOverride::default(),
                    user_history_stats: user_history_stats(model_config, None, candidate_set),
                });
            }
        };

        let num_rows = batch.num_rows();

        let tweet_ids = typed_column::<Int64Array>(&batch, "tweetId");
        let author_ids = typed_column::<Int64Array>(&batch, "authorId");
        let product_surfaces = typed_column::<Int32Array>(&batch, PRODUCT_SURFACE_SEQ_COLUMN);
        let action_multi_hot = typed_column::<FixedSizeListArray>(&batch, "actionNameMultiHot");
        let continuous_actions =
            typed_column::<FixedSizeListArray>(&batch, "continuousActionValuesSeq");

        let col_impressed_time_ms = typed_column::<Int64Array>(&batch, "impressedTimeMs");
        let col_timezone_id = typed_column::<Int32Array>(&batch, "timezoneId");

        let col_semantic_id = typed_column::<FixedSizeListArray>(&batch, "semanticId");

        let col_fav_count = typed_column::<Int64Array>(&batch, FAV_COUNT_SEQ_COLUMN);
        let col_reply_count = typed_column::<Int64Array>(&batch, REPLY_COUNT_SEQ_COLUMN);
        let col_repost_count = typed_column::<Int64Array>(&batch, REPOST_COUNT_SEQ_COLUMN);
        let col_quote_count = typed_column::<Int64Array>(&batch, QUOTE_COUNT_SEQ_COLUMN);
        let col_view_count = typed_column::<Int64Array>(&batch, VIEW_COUNT_SEQ_COLUMN);
        let col_web_conv_url_id = typed_column::<Int64Array>(&batch, WEB_CONV_URL_ID_COLUMN);
        let col_web_conv_ranked_url_ids =
            WEB_CONV_RANKED_URL_ID_COLUMNS.map(|name| typed_column::<Int64Array>(&batch, name));
        let mut history_web_conv_url_ids = Vec::new();
        let mut history_web_conv_ranked_url_ids = Vec::new();
        let col_web_conv_tracking_integration =
            typed_column::<Int32Array>(&batch, WEB_CONV_TRACKING_INTEGRATION_SEQ_COLUMN);
        let col_web_conv_time_on_site = [
            (
                WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ,
                typed_column::<Int64Array>(&batch, WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ_COLUMN),
            ),
            (
                WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ,
                typed_column::<Int64Array>(&batch, WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ_COLUMN),
            ),
        ];
        let col_is_author_followed_by_viewer =
            typed_column::<BooleanArray>(&batch, IS_AUTHOR_FOLLOWED_BY_VIEWER_SEQ_COLUMN);
        let col_is_author_following_viewer =
            typed_column::<BooleanArray>(&batch, IS_AUTHOR_FOLLOWING_VIEWER_SEQ_COLUMN);
        let col_first_video_duration_ms = model_config
            .hash_table
            .video_completion_history
            .then(|| typed_column::<Int64Array>(&batch, FIRST_VIDEO_DURATION_MS_COLUMN))
            .flatten();
        let display_location_on = model_config.hash_table.history_display_location;
        let col_display_location = display_location_on
            .then(|| typed_column::<Int16Array>(&batch, history_display_location::COLUMN))
            .flatten();
        let mut history_display_locations =
            vec![0i16; col_display_location.map_or(0, |_| history_seq_len)];

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

            if let Some(fsl) = col_semantic_id
                && !fsl.is_null(row_idx)
            {
                let inner = fsl.value(row_idx);
                if let Some(codes) = inner.as_any().downcast_ref::<Int32Array>() {
                    stamp_semantic_ids(
                        &mut history_semantic_ids,
                        valid_entry_count,
                        sid_num_levels,
                        codes.values(),
                    );
                }
            }

            if let Some(ps) = product_surfaces {
                history_product_surfaces[valid_entry_count] =
                    if display_location_on && ps.is_null(row_idx) {
                        0
                    } else {
                        ps.value(row_idx)
                    };
            }
            if let Some(locations) = col_display_location
                && locations.is_valid(row_idx)
            {
                history_display_locations[valid_entry_count] = locations.value(row_idx);
            }

            if let Some(fsl) = action_multi_hot {
                let inner_bools = fsl.value(row_idx);
                if let Some(bools) = inner_bools.as_any().downcast_ref::<BooleanArray>() {
                    let base_idx = valid_entry_count * output_vocab_size;
                    let copy_len = output_vocab_size.min(bools.len());
                    for j in 0..copy_len {
                        history_actions[base_idx + j] = bools.value(j);
                    }
                }
            }

            if let Some(fsl) = continuous_actions {
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
            if let Some(code) = col_web_conv_tracking_integration
                && !code.is_null(row_idx)
            {
                history_web_conv_tracking_integration[valid_entry_count] = code.value(row_idx);
            }
            if let Some(url_id) = col_web_conv_url_id
                && !url_id.is_null(row_idx)
                && url_id.value(row_idx) != 0
            {
                history_web_conv_url_ids.push((valid_entry_count as u32, url_id.value(row_idx)));
            }
            let first_url = history_web_conv_ranked_url_ids.len();
            for col in col_web_conv_ranked_url_ids.iter().flatten() {
                if col.is_null(row_idx) || col.value(row_idx) == 0 {
                    continue;
                }
                let url_id = col.value(row_idx);
                if !history_web_conv_ranked_url_ids[first_url..]
                    .iter()
                    .any(|&(_, id)| id == url_id)
                {
                    history_web_conv_ranked_url_ids.push((valid_entry_count as u32, url_id));
                }
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
            stamp_video_duration(
                model_config,
                &mut history_int64_features,
                valid_entry_count,
                col_first_video_duration_ms
                    .filter(|d| !d.is_null(row_idx))
                    .map_or(0, |d| d.value(row_idx)),
            );
            for (feature_idx, col) in col_web_conv_time_on_site {
                if let Some(arr) = col
                    && n_post_int64 > feature_idx
                    && !arr.is_null(row_idx)
                {
                    history_int64_features[valid_entry_count * n_post_int64 + feature_idx] =
                        arr.value(row_idx).max(0);
                }
            }

            history_is_author_followed[valid_entry_count] = col_is_author_followed_by_viewer
                .is_some_and(|arr| !arr.is_null(row_idx) && arr.value(row_idx));
            history_is_author_following[valid_entry_count] = col_is_author_following_viewer
                .is_some_and(|arr| !arr.is_null(row_idx) && arr.value(row_idx));

            valid_entry_count += 1;
        }

        let history_ad_row_mask = if display_location_on {
            history_display_location::overlay(
                &mut history_product_surfaces,
                &history_display_locations,
                valid_entry_count,
            )
        } else {
            Vec::new()
        };

        #[cfg(recsys_time_gap)]
        let mut candidate_int64_features = candidate_int64_features;
        #[cfg(recsys_time_gap)]
        stamp_time_gap_timestamps(
            &history_impr_ts[..valid_entry_count],
            &candidate_impr_ts,
            &mut history_int64_features,
            &mut candidate_int64_features,
            n_post_int64,
        );

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
        stamp_i32_as_categorical(
            &history_web_conv_tracking_integration,
            &mut history_categorical_features,
            n_post_cat,
            WEB_CONV_TRACKING_INTEGRATION_SEQ,
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
            candidate_ads_count_buckets: Vec::new(),
            candidate_ads_ctr_buckets: Vec::new(),
            candidate_ad_side_ids: Vec::new(),
            history_ad_row_mask,
            history_post_ids,
            history_web_conv_url_ids,
            history_web_conv_ranked_url_ids,
            history_url_embeddings: Vec::new(),
            history_semantic_ids,
            candidate_semantic_ids,
            num_history: valid_entry_count,
            cread_decode: CreadDecodeOverride::default(),
            user_history_stats: user_history_stats(model_config, Some(&batch), candidate_set),
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

    #[test]
    fn conv_asset_ids_for_batch_predicate_and_keys() {
        use arrow::array::{ArrayRef, BooleanArray, FixedSizeListArray, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let rows = [
            (11, 21, 111_000, false),
            (WEB_CONV_FAKE_TWEET_ID, 7, 1_000, false),
            (555, 8, 2_000, true),
            (WEB_CONV_FAKE_TWEET_ID, 7, 3_000, false),
            (WEB_CONV_FAKE_TWEET_ID, 7, 4_000, false),
        ];
        let vocab = 256usize;
        let conv_bit = pb::ActionName::AdsWebConversion as usize;
        assert!(conv_bit < vocab);
        let mut bits = vec![false; rows.len() * vocab];
        for (i, r) in rows.iter().enumerate() {
            bits[i * vocab + conv_bit] = r.3;
        }
        let multi_hot = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Boolean, true)),
            vocab as i32,
            Arc::new(BooleanArray::from(bits)),
            None,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("tweetId", DataType::Int64, false),
            Field::new("authorId", DataType::Int64, false),
            Field::new("impressedTimeMs", DataType::Int64, false),
            Field::new("actionNameMultiHot", multi_hot.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )) as ArrayRef,
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                Arc::new(multi_hot),
            ],
        )
        .unwrap();

        let ids = pb::ConvAssetIds {
            impressed_time_ms: vec![111_000, 1_000, 2_000, 4_000],
            author_id: vec![21, 7, 8, 7],
            asset_id: vec![99, 42, 9, 0],
        };
        let map = conv_asset_map(Some(&ids));
        assert_eq!(map.len(), 3, "asset_id 0 windows are dropped from the map");
        assert_eq!(conv_asset_ids_for_batch(&batch, &map), vec![0, 42, 9, 0, 0]);

        let ragged = pb::ConvAssetIds {
            impressed_time_ms: vec![1_000],
            author_id: vec![7, 8],
            asset_id: vec![42],
        };
        assert!(conv_asset_map(Some(&ragged)).is_empty());
        assert!(conv_asset_map(None).is_empty());
        assert_eq!(
            conv_asset_ids_for_batch(&batch, &HashMap::new()),
            vec![0; 5]
        );
    }
    use super::*;
    use crate::feature_config::categorical_feature::{
        PRODUCT_SURFACE_SEQ, PRODUCT_SURFACE_SEQ_COLUMN, PRODUCT_SURFACE_SEQ_NAME,
    };
    use crate::feature_config::int64_feature::{CLIENT_APP_ID_SEQ, CLIENT_APP_ID_SEQ_NAME};
    use arrow::datatypes::DataType;

    #[test]
    fn candidate_impression_time_uses_request_timestamp() {
        let model_config = test_model_config(1);
        let mut candidate_set = pb::CandidateSet::default();
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: 1000,
            author_id: 2000,
            ..Default::default()
        });
        candidate_set.request_timestamp_ms = Some(1_760_000_123_999);
        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);
        assert_eq!(cand.impr_ts[0], 1_760_000_123);

        candidate_set.request_timestamp_ms = None;
        let before = now_epoch_sec();
        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);
        assert!((before..=now_epoch_sec()).contains(&cand.impr_ts[0]));
    }

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
                enable_stale_post_30d: false,
                video_completion_history: false,
                ads_count_features: false,
                ads_ctr_features: false,
                ad_side_fields: Vec::new(),
                history_display_location: false,
            },
            history_seq_len: 4,
            candidate_seq_len: 3,
            multimodal_embedding_dim: 0,
            search_query_embedding_dim: 0,
            num_categorical_features: 0,
            sid_num_levels: 0,
            user_history_stats_dim: 0,
        }
    }

    #[test]
    fn drop_recent_history_removes_window_rows_and_moves_kept_rows_up() {
        let mc = test_model_config(0);
        let mut buf = InputBuffer {
            candidate_post_hashes: vec![7, 7, 7, 7, 0, 0],
            candidate_impr_ts: vec![10_050, 10_000, 9_000],
            history_post_hashes: vec![1, 1, 2, 2, 3, 3, 0, 0],
            history_impr_ts: vec![9_500, 8_000, 9_900, 0],
            history_actions: vec![true, false, false, true, true, true, false, false],
            history_post_ids: vec![11, 12, 13, 0],
            history_ad_row_mask: vec![true, false, true, false],
            history_web_conv_url_ids: vec![(1, 42), (2, 43)],
            num_history: 3,
            ..Default::default()
        };
        let mut draws = [0.1, 0.9].into_iter();
        assert_eq!(
            buf.drop_recent_history(&mc, 0.5, 900, || draws.next().unwrap()),
            1
        );
        assert_eq!(buf.history_post_hashes, vec![2, 2, 3, 3, 0, 0, 0, 0]);
        assert_eq!(buf.history_ad_row_mask, vec![false, true, false, false]);
        assert_eq!(buf.history_impr_ts, vec![8_000, 9_900, 0, 0]);
        assert_eq!(
            buf.history_actions,
            vec![false, true, true, true, false, false, false, false]
        );
        assert_eq!(buf.history_post_ids, vec![12, 13, 0, 0]);
        assert_eq!(buf.history_web_conv_url_ids, vec![(0, 42), (1, 43)]);
        assert_eq!(buf.num_history, 2);

        let not_interested = pb::ActionName::ClientTweetNotInterestedIn as usize;
        let mut history_actions = vec![false; 4 * 64];
        history_actions[not_interested] = true;
        let mut buf = InputBuffer {
            candidate_post_hashes: vec![7, 7],
            candidate_impr_ts: vec![10_000],
            history_post_hashes: vec![1, 1, 2, 2, 0, 0, 0, 0],
            history_impr_ts: vec![9_500, 9_900, 0, 0],
            history_actions,
            num_history: 2,
            ..Default::default()
        };
        assert_eq!(buf.drop_recent_history(&mc, 1.0, 900, || 0.0), 1);
        assert_eq!(buf.history_post_hashes, vec![1, 1, 0, 0, 0, 0, 0, 0]);
        assert!(buf.history_actions[not_interested]);
        assert_eq!(buf.num_history, 1);
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
    fn search_lexical_match_stamped_with_unset_as_zero() {
        let n_post_cat = EXACT_PHRASE_CATEGORICAL_IDX + 1;
        let n_post_float = MATCHED_WORD_FRACTION_SEQ + 1;
        let mut model_config = test_model_config(n_post_cat);
        model_config.hash_table.num_post_float_features = n_post_float;

        let lexical = |fraction: f32, exact: bool| pb::SearchLexicalMatch {
            matched_word_fraction: fraction,
            exact_phrase: exact,
        };
        let candidate_set = pb::CandidateSet {
            candidates: vec![
                pb::TweetInfo {
                    tweet_id: 1,
                    search_lexical_match: Some(lexical(0.5, false)),
                    ..Default::default()
                },
                pb::TweetInfo {
                    tweet_id: 2,
                    search_lexical_match: Some(lexical(1.0, true)),
                    ..Default::default()
                },
                pb::TweetInfo {
                    tweet_id: 3,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        let exact_phrase =
            |j: usize| cand.categorical_features[j * n_post_cat + EXACT_PHRASE_CATEGORICAL_IDX];
        let fraction = |j: usize| cand.float_features[j * n_post_float + MATCHED_WORD_FRACTION_SEQ];
        assert_eq!((1, 0.5), (exact_phrase(0), fraction(0)));
        assert_eq!((2, 1.0), (exact_phrase(1), fraction(1)));
        assert_eq!((0, 0.0), (exact_phrase(2), fraction(2)));
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

    #[test]
    fn ad_side_ids_match_python() {
        use ad_side::{creative_format, hash_id, objective};
        let hashes = [
            (0, 0),
            (1, 61_547),
            (7, 460_609),
            (1 << 40, 463_028),
            (-5, 498_382),
            (i64::MAX, 478_915),
            (i64::MIN, 156_306),
        ];
        for (id, row) in hashes {
            assert_eq!(hash_id(id), row, "id {id}");
        }
        let (start, end) = (1_787_788_800_000, 1_789_833_600_000);
        let creatives = [
            ((1, 0, end), 1),
            ((0, 0, end), 48),
            ((0, 0, start - 1), 48),
            ((0, 0, start), 52),
            ((0, 0, end - 1), 52),
            ((0, 2, start), 50),
            ((0, 4, end), 52),
            ((-3, 9, end), 55),
            ((60, 2, end), 47),
        ];
        for ((card, media, t), creative) in creatives {
            assert_eq!(
                creative_format(card, media, t),
                creative,
                "{card} {media} {t}"
            );
        }
        assert_eq!(
            [-3, 0, 11, 63, 64, 1 << 40].map(objective),
            [0, 0, 11, 63, 63, 63]
        );
    }

    #[test]
    fn ad_side_fields_parse_servable_names_only() {
        use ad_side::Field;
        assert_eq!(
            ad_side::parse_fields(&[
                "objective",
                "campaign",
                "creative",
                "line_item",
                "url_domain"
            ]),
            Ok(vec![
                Field::Objective,
                Field::Campaign,
                Field::Creative,
                Field::LineItem,
                Field::UrlDomain
            ])
        );
        for bad in ["advertiser", "domain", ""] {
            assert!(ad_side::parse_fields(&[bad]).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn candidate_ad_side_ids_match_the_trainer() {
        use ad_side::Field;
        let ad = |tweet_id: u64, campaign_id, line_item_id, card, media, objective| pb::TweetInfo {
            tweet_id,
            author_id: 2000,
            ad_info: Some(pb::AdInfo {
                campaign_id,
                line_item_id,
                unified_card_type: card,
                media_content_type: media,
                line_item_objective: objective,
                ..Default::default()
            }),
            ..Default::default()
        };
        let organic = pb::TweetInfo {
            tweet_id: 1003,
            author_id: 2001,
            ..Default::default()
        };
        let stamp = |model_config: &ModelConfig, candidates: Vec<pb::TweetInfo>, request_ms| {
            let candidate_set = pb::CandidateSet {
                candidates,
                request_timestamp_ms: Some(request_ms),
                ..Default::default()
            };
            let mut input = InputBuffer::compute_for_item(
                model_config,
                &None,
                &candidate_set,
                None,
                None,
                None,
                None,
            );
            input.stamp_candidate_ad_side_ids(model_config, &candidate_set, |_| 0);
            input.candidate_ad_side_ids
        };
        let mut model_config = test_history_model_config();
        model_config.hash_table.ad_side_fields = vec![
            Field::Campaign,
            Field::LineItem,
            Field::Creative,
            Field::Objective,
        ];
        let served_ms = 1_793_750_400_000;

        let first = vec![ad(1000, 11, 31, 2, 0, 5), ad(1001, 12, 0, 0, 4, 99)];
        assert_eq!(
            stamp(&model_config, first, served_ms),
            [353_680, 323_194, 2, 5, 348_192, 0, 52, 63, 0, 0, 0, 0]
        );
        let second = vec![ad(1002, 23, 43, 29, 0, 11)];
        assert_eq!(
            stamp(&model_config, second, served_ms),
            [228_842, 15_213, 29, 11, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        let organic_ids = stamp(&model_config, vec![organic.clone()], served_ms);
        assert_eq!(organic_ids[..4], [0, 0, 48, 0]);
        let unlogged = stamp(&model_config, vec![organic], 1_788_000_000_000);
        assert_eq!(unlogged[..4], [0, 0, 52, 0]);

        model_config.hash_table.ad_side_fields = vec![Field::Objective, Field::Creative];
        let one = || vec![ad(1000, 11, 31, 2, 0, 5)];
        assert_eq!(stamp(&model_config, one(), served_ms), [5, 2, 0, 0, 0, 0]);
        model_config.hash_table.ad_side_fields.clear();
        assert!(stamp(&model_config, one(), served_ms).is_empty());
    }

    #[test]
    fn candidate_url_domain_ids_hash_the_callers_id() {
        let ad = |tweet_id: u64, url_domain: &str| pb::TweetInfo {
            tweet_id,
            author_id: 2000,
            ad_info: Some(pb::AdInfo {
                url_domain: url_domain.to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let organic = pb::TweetInfo {
            tweet_id: 1002,
            author_id: 2001,
            ..Default::default()
        };
        let candidate_set = pb::CandidateSet {
            candidates: vec![ad(1000, "nike.com"), ad(1001, ""), organic],
            request_timestamp_ms: Some(1_793_750_400_000),
            ..Default::default()
        };
        let mut model_config = test_history_model_config();
        model_config.hash_table.ad_side_fields =
            vec![ad_side::Field::Objective, ad_side::Field::UrlDomain];
        let mut input = InputBuffer::compute_for_item(
            &model_config,
            &None,
            &candidate_set,
            None,
            None,
            None,
            None,
        );
        input.stamp_candidate_ad_side_ids(&model_config, &candidate_set, |domain| {
            domain.len() as i64
        });
        assert_eq!(
            input.candidate_ad_side_ids,
            [0, ad_side::hash_id(8), 0, 0, 0, 0]
        );
    }

    #[test]
    fn search_query_stays_one_vector() {
        let mut model_config = test_model_config(1);
        model_config.search_query_embedding_dim = 4;
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let candidate_set = pb::CandidateSet {
            search_query_embedding: query.clone(),
            candidates: vec![
                pb::TweetInfo {
                    tweet_id: 1000,
                    author_id: 2000,
                    ..Default::default()
                },
                pb::TweetInfo {
                    tweet_id: 1001,
                    author_id: 2001,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);
        assert_eq!(cand.search_query_embeddings, query);
        assert_eq!(cand.search_query_embeddings.len(), 4);
    }

    #[test]
    fn repeat_query_into_writes_prefix_and_keeps_tail() {
        let query = [1.0f32, 2.0];
        let mut dest = vec![9.0f32; 8];
        repeat_query_into(&mut dest, &query, 2);
        assert_eq!(dest, vec![1.0, 2.0, 1.0, 2.0, 9.0, 9.0, 9.0, 9.0]);
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
    fn proto_history_semantic_ids_use_ranking_plus_one_shift() {
        let mut model_config = test_history_model_config();
        model_config.sid_num_levels = 3;
        let sequence = Some(pb::UserActionSequence {
            user_actions_data: Some(pb::UserActionSequenceDataContainer {
                data: Some(
                    pb::user_action_sequence_data_container::Data::OrderedAggregatedUserActionsList(
                        pb::AggregatedUserActionList {
                            aggregated_user_actions: vec![
                                pb::AggregatedUserAction {
                                    tweet_info: Some(pb::TweetInfo {
                                        tweet_id: 11,
                                        author_id: 21,
                                        semantic_ids: vec![0, 7, -1],
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                },
                                pb::AggregatedUserAction {
                                    tweet_info: Some(pb::TweetInfo {
                                        tweet_id: 12,
                                        author_id: 22,
                                        semantic_ids: vec![3, 4],
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        },
                    ),
                ),
            }),
            ..Default::default()
        });

        let buf = InputBuffer::compute_for_item(
            &model_config,
            &sequence,
            &pb::CandidateSet::default(),
            None,
            None,
            None,
            None,
        );

        assert_eq!(
            buf.history_semantic_ids,
            vec![1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(buf.history_post_ids, vec![11, 12, 0, 0]);
    }

    #[test]
    fn stamp_semantic_ids_shifts_and_skips_wrong_arity() {
        let mut dst = vec![0u16; 6];
        stamp_semantic_ids(&mut dst, 0, 3, &[0, 7, -1]);
        stamp_semantic_ids(&mut dst, 1, 3, &[3, 4]);
        assert_eq!(dst, vec![1, 8, 0, 0, 0, 0]);
    }

    #[test]
    fn proto_and_columnar_history_fill_the_video_duration_slot_like_training() {
        use arrow::array::{FixedSizeListArray, Float32Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let cases = [
            (None, 100.0f32),
            (Some(40_000i64), 39.7),
            (Some(i64::from(i32::MAX) + 1), 12.0),
        ];
        let n = cases.len();
        let sequence = Some(pb::UserActionSequence {
            user_actions_data: Some(pb::UserActionSequenceDataContainer {
                data: Some(
                    pb::user_action_sequence_data_container::Data::OrderedAggregatedUserActionsList(
                        pb::AggregatedUserActionList {
                            aggregated_user_actions: cases
                                .iter()
                                .enumerate()
                                .map(|(i, &(duration_ms, dwell_s))| pb::AggregatedUserAction {
                                    tweet_info: Some(pb::TweetInfo {
                                        tweet_id: 11 + i as u64,
                                        author_id: 21,
                                        first_video_duration_ms: duration_ms.unwrap_or(0) as u64,
                                        ..Default::default()
                                    }),
                                    actions: vec![pb::ActionInfo {
                                        user_action_meta: Some(pb::UserActionMeta {
                                            dwell_time: f64::from(dwell_s) * 1000.0,
                                            ..Default::default()
                                        }),
                                        ..Default::default()
                                    }],
                                    ..Default::default()
                                })
                                .collect(),
                            ..Default::default()
                        },
                    ),
                ),
            }),
            ..Default::default()
        });
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Schema::new(vec![
            Field::new("tweetId", DataType::Int64, false),
            Field::new("authorId", DataType::Int64, false),
            Field::new(
                "continuousActionValuesSeq",
                DataType::FixedSizeList(item.clone(), 2),
                true,
            ),
            Field::new("firstVideoDurationMs", DataType::Int64, true),
        ]);
        let dwell: Vec<f32> = cases
            .iter()
            .flat_map(|&(_, dwell_s)| [0.0, dwell_s])
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from_iter_values((0..n as i64).map(|i| 11 + i))),
                Arc::new(Int64Array::from(vec![21i64; n])),
                Arc::new(FixedSizeListArray::new(
                    item,
                    2,
                    Arc::new(Float32Array::from(dwell)),
                    None,
                )),
                Arc::new(cases.iter().map(|&(d, _)| d).collect::<Int64Array>()),
            ],
        )
        .unwrap();
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let candidate_set = pb::CandidateSet::default();
        let inputs = |model_config: &ModelConfig| {
            let proto = InputBuffer::compute_for_item(
                model_config,
                &sequence,
                &candidate_set,
                None,
                None,
                None,
                None,
            );
            let columnar = InputBuffer::compute_from_columnar_bytes(
                model_config,
                &bytes,
                &candidate_set,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let width = model_config.hash_table.output_vocab_size;
            let n_int64 = model_config.hash_table.num_post_int64_features;
            let video_completed = pb::ActionName::ClientNotificationSent as usize;
            [proto, columnar].map(|buf| {
                assert!((0..n).all(|row| !buf.history_actions[row * width + video_completed]));
                (0..n)
                    .map(|row| {
                        (
                            buf.history_int64_features
                                [row * n_int64 + int64_feature::FIRST_VIDEO_DURATION_MS_SEQ],
                            buf.history_continuous_actions[row * 2 + 1],
                        )
                    })
                    .collect::<Vec<_>>()
            })
        };
        let mut model_config = test_history_model_config();
        model_config.history_seq_len = n;
        model_config.hash_table.num_post_int64_features =
            int64_feature::FIRST_VIDEO_DURATION_MS_SEQ + 1;
        let off: Vec<(i64, f32)> = cases.iter().map(|&(_, dwell_s)| (0, dwell_s)).collect();
        assert_eq!(inputs(&model_config), [off.clone(), off]);

        model_config.hash_table.video_completion_history = true;
        let on: Vec<(i64, f32)> = cases
            .iter()
            .map(|&(duration_ms, dwell_s)| {
                (
                    duration_ms.unwrap_or(0).clamp(0, i64::from(i32::MAX)),
                    dwell_s,
                )
            })
            .collect();
        assert_eq!(inputs(&model_config), [on.clone(), on]);
    }

    #[test]
    fn columnar_history_semantic_ids_read_semantic_id_column() {
        use arrow::array::{FixedSizeListArray, Int32Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let item = Arc::new(Field::new("item", DataType::Int32, true));
        let schema = Schema::new(vec![
            Field::new("tweetId", DataType::Int64, false),
            Field::new("authorId", DataType::Int64, false),
            Field::new("semanticId", DataType::FixedSizeList(item.clone(), 3), true),
        ]);
        let sid = FixedSizeListArray::new(
            item,
            3,
            Arc::new(Int32Array::from(vec![0, 7, -1, 3, 4, 5])),
            None,
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![11i64, 12])),
                Arc::new(Int64Array::from(vec![21i64, 22])),
                Arc::new(sid),
            ],
        )
        .unwrap();
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let mut model_config = test_history_model_config();
        model_config.sid_num_levels = 3;
        let buf = InputBuffer::compute_from_columnar_bytes(
            &model_config,
            &bytes,
            &pb::CandidateSet::default(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            buf.history_semantic_ids,
            vec![1, 8, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn columnar_mistyped_columns_are_skipped_and_counted() {
        use arrow::array::{Int32Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        for name in [
            "actionNameMultiHot",
            "continuousActionValuesSeq",
            "semanticId",
            "impressedTimeMs",
        ] {
            let schema = Schema::new(vec![
                Field::new("tweetId", DataType::Int64, false),
                Field::new("authorId", DataType::Int64, false),
                Field::new(name, DataType::Int32, false),
            ]);
            let batch = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(Int64Array::from(vec![11i64])),
                    Arc::new(Int64Array::from(vec![21i64])),
                    Arc::new(Int32Array::from(vec![7])),
                ],
            )
            .unwrap();
            let mut bytes = Vec::new();
            {
                let mut writer =
                    StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let mut model_config = test_history_model_config();
            model_config.sid_num_levels = 3;
            let mistyped = COLUMNAR_COLUMN_MISTYPED_TOTAL.with_label_values(&[name]);
            let before = mistyped.get();
            let buf = InputBuffer::compute_from_columnar_bytes(
                &model_config,
                &bytes,
                &pb::CandidateSet::default(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

            assert_eq!(mistyped.get(), before + 1, "{name}");
            assert_eq!(buf.history_post_ids, vec![11, 0, 0, 0], "{name}");
            assert!(buf.history_actions.iter().all(|&a| !a), "{name}");
            assert!(buf.history_semantic_ids.iter().all(|&s| s == 0), "{name}");
        }
    }

    #[test]
    fn columnar_history_web_conv_columns_fill_their_feature_slots() {
        use arrow::array::{Int32Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let to_bytes = |batch: &RecordBatch| {
            let mut bytes = Vec::new();
            let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(batch).unwrap();
            writer.finish().unwrap();
            bytes
        };
        let with_columns = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("tweetId", DataType::Int64, false),
                Field::new("authorId", DataType::Int64, false),
                Field::new(
                    WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ_COLUMN,
                    DataType::Int64,
                    true,
                ),
                Field::new(
                    WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ_COLUMN,
                    DataType::Int64,
                    true,
                ),
                Field::new(
                    WEB_CONV_TRACKING_INTEGRATION_SEQ_COLUMN,
                    DataType::Int32,
                    true,
                ),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![11i64, 12])),
                Arc::new(Int64Array::from(vec![21i64, 22])),
                Arc::new(Int64Array::from(vec![Some(125_400i64), Some(0)])),
                Arc::new(Int64Array::from(vec![Some(61_999i64), None])),
                Arc::new(Int32Array::from(vec![Some(5i32), None])),
            ],
        )
        .unwrap();
        let without_columns = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("tweetId", DataType::Int64, false),
                Field::new("authorId", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![11i64, 12])),
                Arc::new(Int64Array::from(vec![21i64, 22])),
            ],
        )
        .unwrap();

        let mut model_config = test_history_model_config();
        model_config.hash_table.num_post_int64_features = WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ + 1;
        model_config.hash_table.num_post_categorical_features =
            WEB_CONV_TRACKING_INTEGRATION_SEQ + 1;
        let n = model_config.hash_table.num_post_int64_features;
        let n_cat = model_config.hash_table.num_post_categorical_features;
        let history_slot =
            |buf: &InputBuffer, row: usize, idx: usize| buf.history_int64_features[row * n + idx];
        let history_cat = |buf: &InputBuffer, row: usize| {
            buf.history_categorical_features[row * n_cat + WEB_CONV_TRACKING_INTEGRATION_SEQ]
        };

        let buf = InputBuffer::compute_from_columnar_bytes(
            &model_config,
            &to_bytes(&with_columns),
            &pb::CandidateSet::default(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            125_400,
            history_slot(&buf, 0, WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ)
        );
        assert_eq!(
            61_999,
            history_slot(&buf, 0, WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ)
        );
        assert_eq!(
            0,
            history_slot(&buf, 1, WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ)
        );
        assert_eq!(
            0,
            history_slot(&buf, 1, WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ)
        );
        assert_eq!(5, history_cat(&buf, 0));
        assert_eq!(0, history_cat(&buf, 1));

        let buf = InputBuffer::compute_from_columnar_bytes(
            &model_config,
            &to_bytes(&without_columns),
            &pb::CandidateSet::default(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        for row in 0..2 {
            assert_eq!(
                0,
                history_slot(&buf, row, WEB_CONV_TIME_ON_SITE_INFERRED_MS_SEQ)
            );
            assert_eq!(
                0,
                history_slot(&buf, row, WEB_CONV_TIME_ON_SITE_MEASURED_MS_SEQ)
            );
            assert_eq!(0, history_cat(&buf, row));
        }
    }

    #[test]
    fn columnar_user_history_stats_featurize_the_request_history() {
        use arrow::array::{ArrayRef, BooleanArray, Int32Array, Int64Array};
        use arrow::datatypes::{DataType, Field};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let i64s = |v: Vec<Option<i64>>| Arc::new(Int64Array::from(v)) as ArrayRef;
        let i32s = |v: Vec<i32>| Arc::new(Int32Array::from(v)) as ArrayRef;
        let bools = |v: Vec<Option<bool>>| Arc::new(BooleanArray::from(v)) as ArrayRef;
        let hot = BooleanArray::from(vec![
            true, false, false, false, false, true, true, false, false, false, false, true,
        ]);
        let list_field = Arc::new(Field::new("item", DataType::Boolean, true));
        let batch = RecordBatch::try_from_iter([
            ("tweetId", i64s(vec![Some(11), Some(12), Some(13)])),
            ("authorId", i64s(vec![Some(21), Some(22), Some(23)])),
            (
                "impressedTimeMs",
                i64s(vec![Some(1_000_000), Some(2_000_000), Some(3_000_000)]),
            ),
            ("dwellTimeMs", i64s(vec![Some(1_500), Some(9_000), None])),
            (
                "firstVideoDurationMs",
                i64s(vec![Some(0), Some(30_000), Some(12_000)]),
            ),
            ("timezoneId", i32s(vec![21, 21, 21])),
            ("productSurface", i32s(vec![1, 10, 10])),
            ("hasMedia", bools(vec![Some(false), Some(true), Some(true)])),
            (
                "isAuthorFollowedByViewer",
                bools(vec![Some(false), Some(true), None]),
            ),
            ("tweetPosition", i32s(vec![4, 0, 7])),
            ("mediaContentType", i32s(vec![0, 0, 0])),
            (
                "actionNameMultiHot",
                Arc::new(FixedSizeListArray::new(list_field, 4, Arc::new(hot), None)) as ArrayRef,
            ),
        ])
        .unwrap();
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        let candidate_set = pb::CandidateSet {
            request_timestamp_ms: Some(9_000_000),
            ..Default::default()
        };
        let compute = |model_config: &ModelConfig| {
            InputBuffer::compute_from_columnar_bytes(
                model_config,
                &bytes,
                &candidate_set,
                None,
                None,
                None,
                None,
            )
            .unwrap()
            .user_history_stats
        };

        let mut model_config = test_history_model_config();
        assert!(compute(&model_config).is_empty());
        model_config.user_history_stats_dim = xai_user_debias::HISTORY_STATS_DIM;
        let rows = xai_user_debias::parse_sequence(&bytes).unwrap();
        assert_eq!(
            compute(&model_config),
            xai_user_debias::history_stats(&rows, 9_000_000)
        );
    }

    #[test]
    fn columnar_history_web_conv_url_ids_follow_kept_positions() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let to_bytes = |batch: &RecordBatch| {
            let mut bytes = Vec::new();
            let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(batch).unwrap();
            writer.finish().unwrap();
            bytes
        };
        let ids = || -> Vec<Arc<dyn arrow::array::Array>> {
            vec![
                Arc::new(Int64Array::from(vec![11i64, 0, 13, 14])),
                Arc::new(Int64Array::from(vec![21i64, 22, 23, 24])),
            ]
        };
        let id_fields = || {
            vec![
                Field::new("tweetId", DataType::Int64, false),
                Field::new("authorId", DataType::Int64, false),
            ]
        };
        let mut fields = id_fields();
        fields.push(Field::new(WEB_CONV_URL_ID_COLUMN, DataType::Int64, true));
        let mut columns = ids();
        columns.push(Arc::new(Int64Array::from(vec![
            Some(-1_392_645_021_888_827_343i64),
            Some(99),
            None,
            Some(7),
        ])));
        let with_column = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        let without_column =
            RecordBatch::try_new(Arc::new(Schema::new(id_fields())), ids()).unwrap();

        let model_config = test_history_model_config();
        let compute = |batch: &RecordBatch| {
            InputBuffer::compute_from_columnar_bytes(
                &model_config,
                &to_bytes(batch),
                &pb::CandidateSet::default(),
                None,
                None,
                None,
                None,
            )
            .unwrap()
        };
        let buf = compute(&with_column);
        assert_eq!(buf.num_history, 3);
        assert_eq!(
            buf.history_web_conv_url_ids,
            vec![(0, -1_392_645_021_888_827_343), (2, 7)]
        );
        assert!(compute(&without_column).history_web_conv_url_ids.is_empty());

        let mut columns = with_column.columns().to_vec();
        let mut fields: Vec<Field> = with_column
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        for (name, values) in [
            (
                WEB_CONV_RANKED_URL_ID_COLUMNS[0],
                vec![Some(5i64), Some(6), Some(8), Some(7)],
            ),
            (
                WEB_CONV_RANKED_URL_ID_COLUMNS[1],
                vec![Some(5i64), None, None, Some(9)],
            ),
            (
                WEB_CONV_RANKED_URL_ID_COLUMNS[2],
                vec![Some(0i64), None, None, Some(10)],
            ),
        ] {
            fields.push(Field::new(name, DataType::Int64, true));
            columns.push(Arc::new(Int64Array::from(values)));
        }
        let with_ranked = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        let buf = compute(&with_ranked);
        assert_eq!(
            buf.history_web_conv_url_ids,
            vec![(0, -1_392_645_021_888_827_343), (2, 7)]
        );
        assert_eq!(
            buf.history_web_conv_ranked_url_ids,
            vec![(0, 5), (1, 8), (2, 7), (2, 9), (2, 10)]
        );
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

    #[cfg(recsys_notification_counts)]
    #[test]
    fn notification_counts_stamped_into_candidate_int64_features() {
        let mut model_config = test_model_config(1);
        let n_post_int64 = NOTIFICATION_SEE_LESS_OFTEN_COUNT_SEQ + 1;
        model_config.hash_table.num_post_int64_features = n_post_int64;
        let mut candidate_set = pb::CandidateSet::default();
        candidate_set.candidates.push(pb::TweetInfo {
            tweet_id: 1000,
            author_id: 2000,
            not_interested_count: 12,
            notification_sent_count: 13,
            notification_open_count: 14,
            notification_click_count: 15,
            notification_dismiss_count: 16,
            notification_see_less_often_count: 17,
            ..Default::default()
        });

        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);

        assert_eq!(12, cand.int64_features[NOT_INTERESTED_COUNT_SEQ]);
        assert_eq!(13, cand.int64_features[NOTIFICATION_SENT_COUNT_SEQ]);
        assert_eq!(14, cand.int64_features[NOTIFICATION_OPEN_COUNT_SEQ]);
        assert_eq!(15, cand.int64_features[NOTIFICATION_CLICK_COUNT_SEQ]);
        assert_eq!(16, cand.int64_features[NOTIFICATION_DISMISS_COUNT_SEQ]);
        assert_eq!(
            17,
            cand.int64_features[NOTIFICATION_SEE_LESS_OFTEN_COUNT_SEQ]
        );

        let reply_width = (0..).find(|&w| is_reply_width(w)).unwrap();
        let mut reply_row = vec![0i64; reply_width];
        stamp_notification_counts(&mut reply_row, reply_width, 0, 1, 2, 3, 4, 5, 6);
        assert!(reply_row.iter().all(|&v| v == 0));
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

    #[cfg(recsys_dpa_sid)]
    #[test]
    fn dpa_sid_codes_stamped_as_code_plus_one() {
        let mut model_config = test_model_config(1);
        let n_post_int64 = DPA_SID_SLOTS.into_iter().max().unwrap() + 1;
        model_config.hash_table.num_post_int64_features = n_post_int64;
        model_config.candidate_seq_len = 4;

        let candidate = |tweet_id: u64, sid_codes: Option<Vec<i32>>| pb::TweetInfo {
            tweet_id,
            author_id: tweet_id + 1000,
            ad_info: Some(pb::AdInfo {
                dpa_product_info: sid_codes.map(|sid_codes| pb::DpaProductInfo {
                    sid_codes,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let candidate_set = pb::CandidateSet {
            candidates: vec![
                candidate(1, Some(vec![3, 17, 42, 99, 200, 255])),
                candidate(2, Some(vec![-1, 0, 1, 255, 256, i32::MAX])),
                candidate(3, Some(vec![3, 17, 42])),
                candidate(4, None),
            ],
            ..Default::default()
        };
        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);
        let sids =
            |j: usize| DPA_SID_SLOTS.map(|slot| cand.int64_features[j * n_post_int64 + slot]);
        assert_eq!(sids(0), [4, 18, 43, 100, 201, 256]);
        assert_eq!(sids(1), [0, 1, 2, 256, 256, 256]);
        assert_eq!(sids(2), [0; 6]);
        assert_eq!(sids(3), [0; 6]);
    }

    #[cfg(recsys_dpa_sid)]
    #[test]
    fn dpa_sid_codes_skip_narrow_and_reply_width_tables() {
        let codes = [1, 2, 3, 4, 5, 6];
        let n = DPA_SID_SLOTS.into_iter().max().unwrap() + 1;
        assert!(!is_reply_width(n));
        let reply_width = (n..).find(|&w| is_reply_width(w)).unwrap();
        for width in [n - 1, reply_width] {
            let mut values = vec![0i64; width];
            stamp_dpa_sid_codes(&mut values, width, 0, &codes);
            assert!(values.iter().all(|&v| v == 0), "width {width}");
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

        model_config.hash_table.enable_stale_post_30d = true;
        let cand = InputBuffer::new_with_candidates(&model_config, &candidate_set, None);
        assert_eq!(7, cand.int64_features[FAV_COUNT_SEQ]);
        assert!(!cand.bool_features[IS_STALE_POST14D]);
    }

    #[test]
    fn fixed_now_sec_applies_to_this_thread_only_and_is_restored() {
        let fixed = 1_700_000_000;
        let candidates = pb::CandidateSet {
            candidates: vec![pb::TweetInfo {
                tweet_id: 11,
                author_id: 21,
                ..Default::default()
            }],
            ..Default::default()
        };
        let (cand, nested, other_thread) = with_fixed_now_sec(fixed, || {
            let nested = with_fixed_now_sec(42, now_epoch_sec);
            let other_thread = std::thread::spawn(now_epoch_sec).join().unwrap();
            let cand = InputBuffer::new_with_candidates(&test_model_config(0), &candidates, None);
            (cand, nested, other_thread)
        });
        assert_eq!(cand.impr_ts[0], fixed);
        assert_eq!(nested, 42);
        assert_ne!(other_thread, fixed);
        assert_ne!(now_epoch_sec(), fixed);

        let unwound = std::panic::catch_unwind(|| with_fixed_now_sec(7, || panic!("unwind")));
        assert!(unwound.is_err());
        assert_ne!(now_epoch_sec(), 7);
    }

    #[test]
    fn user_gender_is_raw_proto_enum() {
        let mut cfg = test_model_config(0).hash_table;
        cfg.num_user_categorical_features = 7;
        cfg.num_user_float_features = 4;
        let gender_idx = crate::feature_config::user_categorical_feature::USER_GENDER;
        for g in [pb::Gender::Male, pb::Gender::Female, pb::Gender::Unknown] {
            let ctx = pb::UserContext {
                user_gender: g as i32,
                ..Default::default()
            };
            let f = InputBuffer::extract_user_features(&cfg, None, Some(&ctx));
            assert_eq!(f.categorical_features[gender_idx], g as i16, "{g:?}");
        }
    }

    #[cfg(recsys_time_gap)]
    #[test]
    fn time_gap_timestamps() {
        let stamp = |n: usize| {
            let (mut hist, mut cand) = (vec![0i64; 3 * n], vec![0i64; 3 * n]);
            let hist_ts = [1_600_000_000, 0, 1_600_000_002];
            let cand_ts = [1_600_000_001, 1_600_000_001, 0];
            stamp_time_gap_timestamps(&hist_ts, &cand_ts, &mut hist, &mut cand, n);
            (hist, cand)
        };
        let n = LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ + 1;
        let col = |v: &[i64], slot: usize| v.chunks(n).map(|row| row[slot]).collect::<Vec<_>>();
        let (hist, cand) = stamp(n);
        assert_eq!(col(&hist, REQUEST_TIME_SEC_SEQ), [1_600_000_001; 3]);
        assert_eq!(col(&hist, LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ), [0; 3]);
        assert_eq!(
            col(&cand, REQUEST_TIME_SEC_SEQ),
            [1_600_000_001, 1_600_000_001, 0]
        );
        let latest = col(&cand, LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ);
        assert_eq!(latest, [1_600_000_000, 1_600_000_000, 0]);

        assert!(!is_reply_width(n));
        let reply_width = (n..).find(|&w| is_reply_width(w)).unwrap();
        for width in [n - 1, reply_width] {
            let (hist, cand) = stamp(width);
            assert!(hist.iter().chain(&cand).all(|&v| v == 0), "width {width}");
        }
    }

    #[cfg(recsys_time_gap)]
    #[test]
    fn proto_and_columnar_history_stamp_time_gap_timestamps() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let now_sec = 1_600_000_010;
        let impressed_ms = [1_600_000_000_000i64, 1_600_000_005_000, 1_600_000_015_000];
        let mut model_config = test_history_model_config();
        model_config.hash_table.num_post_int64_features =
            LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ + 1;
        let n = model_config.hash_table.num_post_int64_features;
        let mut candidate_set = pb::CandidateSet::default();
        for tweet_id in [101, 102] {
            candidate_set.candidates.push(pb::TweetInfo {
                tweet_id,
                author_id: 2000,
                ..Default::default()
            });
        }

        let sequence = Some(pb::UserActionSequence {
            user_actions_data: Some(pb::UserActionSequenceDataContainer {
                data: Some(
                    pb::user_action_sequence_data_container::Data::OrderedAggregatedUserActionsList(
                        pb::AggregatedUserActionList {
                            aggregated_user_actions: impressed_ms
                                .iter()
                                .enumerate()
                                .map(|(i, &ms)| pb::AggregatedUserAction {
                                    tweet_info: Some(pb::TweetInfo {
                                        tweet_id: 11 + i as u64,
                                        author_id: 21,
                                        ..Default::default()
                                    }),
                                    impressed_time_ms: ms,
                                    ..Default::default()
                                })
                                .collect(),
                            ..Default::default()
                        },
                    ),
                ),
            }),
            ..Default::default()
        });
        let history = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("tweetId", DataType::Int64, false),
                Field::new("authorId", DataType::Int64, false),
                Field::new("impressedTimeMs", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![11i64, 12, 13])),
                Arc::new(Int64Array::from(vec![21i64, 21, 21])),
                Arc::new(Int64Array::from(impressed_ms.to_vec())),
            ],
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, history.schema().as_ref()).unwrap();
        writer.write(&history).unwrap();
        writer.finish().unwrap();

        let proto = with_fixed_now_sec(now_sec, || {
            InputBuffer::compute_for_item(
                &model_config,
                &sequence,
                &candidate_set,
                None,
                None,
                None,
                None,
            )
        });
        let columnar = with_fixed_now_sec(now_sec, || {
            InputBuffer::compute_from_columnar_bytes(
                &model_config,
                &bytes,
                &candidate_set,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        });
        for buf in [&proto, &columnar] {
            assert_eq!(buf.num_history, 3);
            for i in 0..3 {
                assert_eq!(
                    buf.history_int64_features[i * n + REQUEST_TIME_SEC_SEQ],
                    now_sec as i64
                );
            }
            for j in 0..2 {
                assert_eq!(
                    buf.candidate_int64_features[j * n + REQUEST_TIME_SEC_SEQ],
                    now_sec as i64
                );
                assert_eq!(
                    buf.candidate_int64_features[j * n + LATEST_HISTORY_IMPRESSION_TIME_SEC_SEQ],
                    1_600_000_005
                );
            }
            assert!(
                buf.candidate_int64_features[2 * n..]
                    .iter()
                    .all(|&v| v == 0)
            );
        }
    }

    type HistoryRow = (i64, i64, Option<i32>, Option<i16>);

    fn display_location_history(rows: &[HistoryRow], location_type: Option<DataType>) -> Vec<u8> {
        use arrow::array::ArrayRef;
        use arrow::datatypes::{Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;

        let mut fields = vec![
            Field::new("tweetId", DataType::Int64, false),
            Field::new("authorId", DataType::Int64, false),
            Field::new(PRODUCT_SURFACE_SEQ_COLUMN, DataType::Int32, true),
        ];
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(rows.iter().map(|r| r.0).collect::<Int64Array>()),
            Arc::new(rows.iter().map(|r| r.1).collect::<Int64Array>()),
            Arc::new(rows.iter().map(|r| r.2).collect::<Int32Array>()),
        ];
        if let Some(data_type) = location_type {
            let locations = rows.iter().map(|r| r.3);
            columns.push(match data_type {
                DataType::Int16 => Arc::new(locations.collect::<Int16Array>()),
                DataType::Int32 => {
                    Arc::new(locations.map(|l| l.map(i32::from)).collect::<Int32Array>())
                }
                other => panic!("unsupported displayLocation type {other}"),
            });
            fields.push(Field::new(
                history_display_location::COLUMN,
                data_type,
                true,
            ));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        bytes
    }

    fn columnar_input(model_config: &ModelConfig, bytes: &[u8]) -> InputBuffer {
        InputBuffer::compute_from_columnar_bytes(
            model_config,
            bytes,
            &pb::CandidateSet::default(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    fn history_surface_categories(model_config: &ModelConfig, buf: &InputBuffer) -> Vec<i32> {
        let n_cat = model_config.hash_table.num_post_categorical_features;
        buf.history_categorical_features
            .chunks(n_cat)
            .map(|row| i32::from(row[PRODUCT_SURFACE_SEQ]))
            .collect()
    }

    fn mask(bits: &[u8]) -> Vec<bool> {
        bits.iter().map(|&b| b == 1).collect()
    }

    #[test]
    fn display_location_product_surfaces_match_the_trainer() {
        use history_display_location::product_surface;
        let pairs = [
            (0, 0),
            (1, 6),
            (2, 1),
            (8, 0),
            (9, 9),
            (26, 0),
            (28, 8),
            (29, 2),
            (30, 10),
            (3, 0),
            (31, 0),
            (-1, 0),
            (255, 0),
            (i16::MAX, 0),
            (i16::MIN, 0),
        ];
        for (location, surface) in pairs {
            assert_eq!(product_surface(location), surface, "{location}");
        }
        let mapped: Vec<i16> = (i16::MIN..=i16::MAX)
            .filter(|&location| product_surface(location) != 0)
            .collect();
        assert_eq!(mapped, [1, 2, 9, 28, 29, 30]);
    }

    #[test]
    fn history_display_location_matches_the_trainer_reader() {
        let long: [HistoryRow; 15] = [
            (101, 201, Some(0), Some(2)),
            (102, 202, Some(4), None),
            (103, 203, Some(0), Some(30)),
            (104, 204, Some(1), None),
            (105, 205, Some(0), Some(2)),
            (106, 206, Some(0), Some(29)),
            (107, 207, Some(0), Some(1)),
            (108, 208, Some(0), Some(28)),
            (109, 209, Some(0), Some(9)),
            (110, 210, Some(0), Some(30)),
            (111, 211, Some(0), Some(8)),
            (112, 212, Some(0), Some(26)),
            (113, 213, Some(7), Some(28)),
            (114, 214, Some(0), Some(0)),
            (115, 215, Some(0), None),
        ];
        let short: [HistoryRow; 5] = [
            (121, 221, None, Some(2)),
            (122, 222, Some(0), Some(31)),
            (123, 223, Some(0), Some(-1)),
            (124, 224, Some(0), Some(i16::MAX)),
            (125, 225, Some(3), Some(0)),
        ];
        let cases = [
            (
                &long[..],
                [1, 1, 2, 6, 8, 9, 10, 0, 0, 7, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
            ),
            (
                &short[..],
                [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];
        let mut model_config = test_history_model_config();
        model_config.history_seq_len = 12;
        for (rows, on_surfaces, on_mask, off_surfaces) in cases {
            let bytes = display_location_history(rows, Some(DataType::Int16));
            for (on, surfaces, ad_rows) in [
                (true, on_surfaces, mask(&on_mask)),
                (false, off_surfaces, Vec::new()),
            ] {
                model_config.hash_table.history_display_location = on;
                let buf = columnar_input(&model_config, &bytes);
                assert_eq!(buf.history_product_surfaces, surfaces, "on={on}");
                assert_eq!(
                    history_surface_categories(&model_config, &buf),
                    surfaces,
                    "on={on}"
                );
                assert_eq!(buf.history_ad_row_mask, ad_rows, "on={on}");
            }
        }
    }

    #[test]
    fn history_display_location_stays_on_its_row() {
        let rows: [HistoryRow; 5] = [
            (11, 21, Some(0), Some(2)),
            (12, 22, Some(0), Some(28)),
            (0, 23, Some(0), Some(30)),
            (14, 24, Some(5), Some(9)),
            (15, 25, Some(0), Some(29)),
        ];
        let mut model_config = test_history_model_config();
        model_config.hash_table.history_display_location = true;
        let buf = columnar_input(
            &model_config,
            &display_location_history(&rows, Some(DataType::Int16)),
        );
        assert_eq!(buf.history_post_ids, [12, 14, 15, 0]);
        assert_eq!(buf.history_product_surfaces, [8, 5, 2, 0]);
        assert_eq!(buf.history_ad_row_mask, [true, false, true, false]);
    }

    #[test]
    fn history_without_display_locations_gets_only_the_ad_row_mask() {
        let rows: [HistoryRow; 3] = [
            (11, 21, Some(0), Some(2)),
            (12, 22, Some(6), Some(0)),
            (13, 23, Some(0), Some(28)),
        ];
        let mut model_config = test_history_model_config();
        model_config.hash_table.history_display_location = true;
        let mistyped =
            COLUMNAR_COLUMN_MISTYPED_TOTAL.with_label_values(&[history_display_location::COLUMN]);
        for location_type in [None, Some(DataType::Int32)] {
            let before = mistyped.get();
            let buf = columnar_input(
                &model_config,
                &display_location_history(&rows, location_type.clone()),
            );
            assert_eq!(
                buf.history_product_surfaces,
                [0, 6, 0, 0],
                "{location_type:?}"
            );
            assert_eq!(
                buf.history_ad_row_mask,
                [true, false, true, false],
                "{location_type:?}"
            );
            let counted = u64::from(location_type.is_some());
            assert_eq!(mistyped.get(), before + counted, "{location_type:?}");
        }
    }

    #[test]
    fn history_display_location_reads_a_null_surface_as_zero() {
        use arrow::array::ArrayRef;
        use arrow::buffer::NullBuffer;
        use arrow::datatypes::{Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;

        let surfaces =
            Int32Array::new(vec![5, 3].into(), Some(NullBuffer::from(vec![false, true])));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("tweetId", DataType::Int64, false),
                Field::new("authorId", DataType::Int64, false),
                Field::new(PRODUCT_SURFACE_SEQ_COLUMN, DataType::Int32, true),
                Field::new(history_display_location::COLUMN, DataType::Int16, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![11i64, 12])) as ArrayRef,
                Arc::new(Int64Array::from(vec![21i64, 22])),
                Arc::new(surfaces),
                Arc::new(Int16Array::from(vec![30i16, 30])),
            ],
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);

        let mut model_config = test_history_model_config();
        model_config.hash_table.history_display_location = true;
        let on = columnar_input(&model_config, &bytes);
        assert_eq!(on.history_product_surfaces, [10, 3, 0, 0]);
        assert_eq!(on.history_ad_row_mask, [true, false, false, false]);
        model_config.hash_table.history_display_location = false;
        let off = columnar_input(&model_config, &bytes);
        assert_eq!(off.history_product_surfaces, [5, 3, 0, 0]);
    }

    #[test]
    fn proto_history_gets_the_ad_row_mask_without_an_overlay() {
        use pb::user_action_sequence_data_container::Data;
        let entry = |tweet_id: u64, surface: Option<i32>| pb::AggregatedUserAction {
            tweet_info: Some(pb::TweetInfo {
                tweet_id,
                author_id: 21,
                ..Default::default()
            }),
            actions: surface
                .map(|product_surface| pb::ActionInfo {
                    user_action_meta: Some(pb::UserActionMeta {
                        product_surface,
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let sequence = Some(pb::UserActionSequence {
            user_actions_data: Some(pb::UserActionSequenceDataContainer {
                data: Some(Data::OrderedAggregatedUserActionsList(
                    pb::AggregatedUserActionList {
                        aggregated_user_actions: vec![
                            entry(11, Some(0)),
                            entry(0, Some(0)),
                            entry(12, Some(2)),
                            entry(13, None),
                        ],
                        ..Default::default()
                    },
                )),
            }),
            ..Default::default()
        });
        let mut model_config = test_history_model_config();
        for on in [true, false] {
            model_config.hash_table.history_display_location = on;
            let buf = InputBuffer::compute_for_item(
                &model_config,
                &sequence,
                &pb::CandidateSet::default(),
                None,
                None,
                None,
                None,
            );
            assert_eq!(buf.history_product_surfaces, [0, 2, 0, 0], "on={on}");
            let ad_rows = if on {
                vec![true, false, true, false]
            } else {
                Vec::new()
            };
            assert_eq!(buf.history_ad_row_mask, ad_rows, "on={on}");
        }
    }

    macro_rules! assert_same_fields {
        ($a:expr, $b:expr, $what:expr, [$($field:ident),* $(,)?]) => {
            $( assert_eq!($a.$field, $b.$field, "{} {}", $what, stringify!($field)); )*
        };
    }

    #[test]
    fn history_display_location_changes_nothing_else() {
        use arrow::array::ArrayRef;
        use arrow::datatypes::{Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;

        let rows = 7usize;
        let vocab = 64usize;
        let to_bytes = |location_type: Option<DataType>| {
            let item = |data_type| Arc::new(Field::new("item", data_type, true));
            let actions = FixedSizeListArray::new(
                item(DataType::Boolean),
                vocab as i32,
                Arc::new(BooleanArray::from(
                    (0..rows * vocab).map(|j| j % 9 == 2).collect::<Vec<_>>(),
                )),
                None,
            );
            let continuous = FixedSizeListArray::new(
                item(DataType::Float32),
                2,
                Arc::new(Float32Array::from(
                    (0..rows * 2).map(|j| j as f32).collect::<Vec<_>>(),
                )),
                None,
            );
            let mut fields = vec![
                Field::new("tweetId", DataType::Int64, false),
                Field::new("authorId", DataType::Int64, false),
                Field::new(PRODUCT_SURFACE_SEQ_COLUMN, DataType::Int32, true),
                Field::new("actionNameMultiHot", actions.data_type().clone(), false),
                Field::new(
                    "continuousActionValuesSeq",
                    continuous.data_type().clone(),
                    false,
                ),
                Field::new("impressedTimeMs", DataType::Int64, false),
                Field::new("timezoneId", DataType::Int32, false),
                Field::new(FAV_COUNT_SEQ_COLUMN, DataType::Int64, false),
            ];
            let mut columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..rows as i64).map(|i| 100 + i),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..rows as i64).map(|i| 200 + i % 3),
                )),
                Arc::new(Int32Array::from_iter_values(
                    (0..rows as i32).map(|i| i % 3),
                )),
                Arc::new(actions),
                Arc::new(continuous),
                Arc::new(Int64Array::from_iter_values(
                    (0..rows as i64).map(|i| 1_700_000_000_000 + 60_000 * i),
                )),
                Arc::new(Int32Array::from_iter_values(
                    (0..rows as i32).map(|i| i % 4),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..rows as i64).map(|i| 7 * i),
                )),
            ];
            let locations = (0..rows).map(|i| (i % 3 == 0).then_some(2i16));
            match location_type {
                Some(DataType::Int16) => columns.push(Arc::new(locations.collect::<Int16Array>())),
                Some(DataType::Int32) => columns.push(Arc::new(
                    locations.map(|l| l.map(i32::from)).collect::<Int32Array>(),
                )),
                _ => {}
            }
            if let Some(data_type) = location_type {
                fields.push(Field::new(
                    history_display_location::COLUMN,
                    data_type,
                    true,
                ));
            }
            let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
            let mut bytes = Vec::new();
            let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
            drop(writer);
            bytes
        };
        let mut model_config = test_history_model_config();
        model_config.history_seq_len = 6;
        model_config.hash_table.num_post_categorical_features = 18;
        model_config.hash_table.num_post_int64_features = 15;
        let candidates = pb::CandidateSet {
            user_id: 7,
            candidates: vec![pb::TweetInfo {
                tweet_id: 900,
                author_id: 901,
                ..Default::default()
            }],
            ..Default::default()
        };
        let input = |model_config: &ModelConfig, location_type: Option<DataType>| {
            with_fixed_now_sec(1_700_000_600, || {
                InputBuffer::compute_from_columnar_bytes(
                    model_config,
                    &to_bytes(location_type),
                    &candidates,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
        };
        let unstamped = input(&model_config, None);
        assert!(unstamped.history_ad_row_mask.is_empty());
        let stamped = input(&model_config, Some(DataType::Int16));
        let mistyped = input(&model_config, Some(DataType::Int32));
        model_config.hash_table.history_display_location = true;
        let on = input(&model_config, Some(DataType::Int16));
        for (what, other) in [("stamped", &stamped), ("mistyped", &mistyped), ("on", &on)] {
            assert_same_fields!(
                unstamped,
                other,
                what,
                [
                    user_hashes,
                    user_ip_hashes,
                    history_post_hashes,
                    history_auth_hashes,
                    history_actions,
                    history_continuous_actions,
                    candidate_post_hashes,
                    candidate_auth_hashes,
                    candidate_product_surfaces,
                    candidate_embeddings,
                    candidate_search_query_embeddings,
                    user_categorical_features,
                    user_bool_features,
                    user_float_features,
                    user_int64_features,
                    user_installed_apps,
                    history_bool_features,
                    history_float_features,
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
                    candidate_ads_count_buckets,
                    candidate_ads_ctr_buckets,
                    candidate_ad_side_ids,
                    history_post_ids,
                    history_web_conv_url_ids,
                    history_web_conv_ranked_url_ids,
                    history_url_embeddings,
                    history_semantic_ids,
                    candidate_semantic_ids,
                    num_history,
                    cread_decode,
                    user_history_stats,
                ]
            );
        }
        for (what, other) in [("stamped", &stamped), ("mistyped", &mistyped)] {
            assert_same_fields!(
                unstamped,
                other,
                what,
                [
                    history_product_surfaces,
                    history_categorical_features,
                    history_ad_row_mask,
                ]
            );
        }
        assert_eq!(unstamped.history_product_surfaces, [1, 2, 0, 1, 2, 0]);
        assert_eq!(on.history_product_surfaces, [1, 2, 1, 1, 2, 1]);
        assert_eq!(
            on.history_ad_row_mask,
            [false, false, true, false, false, true]
        );
        assert_eq!(
            history_surface_categories(&model_config, &on),
            on.history_product_surfaces
        );
        let n_cat = model_config.hash_table.num_post_categorical_features;
        for (i, (a, b)) in unstamped
            .history_categorical_features
            .iter()
            .zip(&on.history_categorical_features)
            .enumerate()
        {
            if i % n_cat != PRODUCT_SURFACE_SEQ {
                assert_eq!(a, b, "categorical feature {i}");
            }
        }
    }
}
