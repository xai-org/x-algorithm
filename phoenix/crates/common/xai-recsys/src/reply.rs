// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::collections::HashMap;

use lazy_static::lazy_static;
use prometheus::{
    Histogram, IntCounterVec, exponential_buckets, register_histogram, register_int_counter_vec,
};
use xai_recsys_proto as pb;

use crate::model_config::ModelConfig;
use crate::util::{
    InputBuffer, USER_FEATURE_FILL_TOTAL, snowflake_to_creation_ts_sec, stamp_i32_as_categorical,
};

lazy_static! {
    static ref CANDIDATE_FEATURE_FILL_TOTAL: IntCounterVec = register_int_counter_vec!(
        "recsys_candidate_feature_fill_total",
        "Per-candidate feature fill rate by feature name and status.",
        &["feature", "status"]
    )
    .unwrap();
    static ref REQUEST_NUM_CANDIDATES: Histogram = register_histogram!(
        "recsys_engine_request_num_candidates",
        "Candidates per reply request before padding to candidate_seq_len",
        exponential_buckets(1.0, 2.0, 13).unwrap()
    )
    .unwrap();
}

const POST_LANGUAGE_CODE_SEQ: usize = 4;
const FIRST_VIDEO_DURATION_MS_SEQ: usize = 13;
const REPLY_HAS_MEDIA_INT64_INDEX: usize = 4;
const REPLY_AUTHOR_FOLLOWER_COUNT_INT64_INDEX: usize = 14;
const REPLY_NEGFB_TOTAL_COUNT_INT64_INDEX: usize = 15;
const REPLY_FOCAL_AUTHOR_FOLLOWS_INT64_INDEX: usize = 16;
const REPLY_SECTION_INT64_INDEX: usize = 17;
const REPLY_GROX_SCORE_INT64_INDEX: usize = 18;
const REPLY_AUTHOR_VERIFIED_INT64_INDEX: usize = 19;
const REPLY_FAV_BY_ROOT_AUTHOR_INT64_INDEX: usize = 20;
const REPLY_FOCAL_AUTHOR_REPLY_INT64_INDEX: usize = 21;
const REPLY_SELF_REPLY_INT64_INDEX: usize = 22;
const REPLY_PLACEMENT_INT64_INDICES: [usize; 6] = [
    REPLY_SECTION_INT64_INDEX,
    REPLY_GROX_SCORE_INT64_INDEX,
    REPLY_AUTHOR_VERIFIED_INT64_INDEX,
    REPLY_FAV_BY_ROOT_AUTHOR_INT64_INDEX,
    REPLY_FOCAL_AUTHOR_REPLY_INT64_INDEX,
    REPLY_SELF_REPLY_INT64_INDEX,
];
const FOCAL_POST_CREATION_TS_SEC: usize = 0;
const FOCAL_FAV_COUNT: usize = 1;
const FOCAL_REPLY_COUNT: usize = 2;
const FOCAL_VIEW_COUNT: usize = 3;
const FOCAL_AUTHOR_FOLLOWER_COUNT: usize = 4;
const FOCAL_POST_LANGUAGE_CODE: usize = 5;
const VIEWER_IS_FOCAL_AUTHOR: usize = 6;
const FOCAL_IN_VIEWER_HISTORY: usize = 7;

const NEGFB_TOTAL_COMPONENTS: [&str; 3] = ["negfb_report", "negfb_block", "negfb_mute"];
const FOCAL_AUTHOR_FOLLOWS_KEY: &str = "focal_author_follows";
const PLACEMENT_FEATURES: [&str; 6] = [
    "section",
    "grox_score",
    "author_verified",
    "fav_by_root_author",
    "focal_author_reply",
    "self_reply",
];
const GROX_SCORE_OFFSET: i64 = 1001;

#[derive(Default, Clone, Copy, Debug)]
pub struct FocalFeatureValues {
    pub fav_count: i64,
    pub reply_count: i64,
    pub view_count: i64,
    pub author_follower_count: i64,
    pub language_code: i64,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ReplyCandidateExtras {
    pub negfb_total_count: HashMap<u64, i64>,
    pub focal_author_follows: HashMap<u64, i64>,
    pub section: HashMap<u64, i64>,
    pub grox_score: HashMap<u64, i64>,
    pub author_verified: HashMap<u64, i64>,
    pub fav_by_root_author: HashMap<u64, i64>,
    pub focal_author_reply: HashMap<u64, i64>,
    pub self_reply: HashMap<u64, i64>,
}

impl ReplyCandidateExtras {
    pub fn placement(&self, tweet_id: u64) -> [i64; 6] {
        self.placement_with(&self.focal_author_reply, &self.self_reply, tweet_id)
    }

    fn placement_with(
        &self,
        focal_author_reply: &HashMap<u64, i64>,
        self_reply: &HashMap<u64, i64>,
        tweet_id: u64,
    ) -> [i64; 6] {
        let get = |m: &HashMap<u64, i64>| m.get(&tweet_id).copied().unwrap_or(0);
        [
            get(&self.section),
            get(&self.grox_score),
            get(&self.author_verified),
            get(&self.fav_by_root_author),
            get(focal_author_reply),
            get(self_reply),
        ]
    }

    pub fn copy_candidate(&mut self, other: &Self, tweet_id: u64) {
        for (dst, src) in [
            (&mut self.negfb_total_count, &other.negfb_total_count),
            (&mut self.focal_author_follows, &other.focal_author_follows),
            (&mut self.section, &other.section),
            (&mut self.grox_score, &other.grox_score),
            (&mut self.author_verified, &other.author_verified),
            (&mut self.fav_by_root_author, &other.fav_by_root_author),
            (&mut self.focal_author_reply, &other.focal_author_reply),
            (&mut self.self_reply, &other.self_reply),
        ] {
            if let Some(v) = src.get(&tweet_id) {
                dst.insert(tweet_id, *v);
            }
        }
    }
}

fn derive_author_relations(
    candidate_set: &pb::CandidateSet,
    focal_author_id: u64,
) -> (HashMap<u64, i64>, HashMap<u64, i64>) {
    if focal_author_id == 0 {
        return Default::default();
    }
    let replied_to: std::collections::HashSet<u64> = candidate_set
        .candidates
        .iter()
        .filter(|d| d.author_id == focal_author_id && d.in_reply_to_tweet_id != 0)
        .map(|d| d.in_reply_to_tweet_id)
        .collect();
    let mut focal_author_reply = HashMap::with_capacity(candidate_set.candidates.len());
    let mut self_reply = HashMap::with_capacity(candidate_set.candidates.len());
    for c in candidate_set.candidates.iter().filter(|c| c.tweet_id != 0) {
        focal_author_reply.insert(
            c.tweet_id,
            if replied_to.contains(&c.tweet_id) {
                2
            } else {
                1
            },
        );
        self_reply.insert(
            c.tweet_id,
            if c.author_id == focal_author_id { 2 } else { 1 },
        );
    }
    (focal_author_reply, self_reply)
}

#[derive(Debug)]
pub struct ReplyMetadata {
    pub focal_post_id: u64,
    pub focal_author_id: u64,
    pub focal_semantic_ids: Vec<i32>,
    pub author_follower_counts: HashMap<u64, u64>,
    pub focal_features: FocalFeatureValues,
    pub extras: ReplyCandidateExtras,
}

fn id_keyed_i64(v: Option<&serde_json::Value>) -> HashMap<u64, i64> {
    v.and_then(|m| m.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.parse::<u64>().ok()?, v.as_i64()?)))
                .collect()
        })
        .unwrap_or_default()
}

fn id_keyed_slot(
    v: Option<&serde_json::Value>,
    range: impl std::ops::RangeBounds<i64>,
    encode: impl Fn(i64) -> i64,
) -> HashMap<u64, i64> {
    id_keyed_i64(v)
        .into_iter()
        .filter(|(_, v)| range.contains(v))
        .map(|(k, v)| (k, encode(v)))
        .collect()
}

fn parse_extras(metadata: &serde_json::Value) -> ReplyCandidateExtras {
    let mut negfb_total_count: HashMap<u64, i64> = HashMap::new();
    let cf = metadata.get("candidate_features");
    if let Some(cf) = cf {
        for key in NEGFB_TOTAL_COMPONENTS {
            for (tweet_id, n) in id_keyed_i64(cf.get(key)) {
                *negfb_total_count.entry(tweet_id).or_insert(1) += n.max(0);
            }
        }
    }
    let cf_key = |key: &str| cf.and_then(|cf| cf.get(key));
    let as_sent = |v| v;
    let focal_author_follows = metadata
        .get(FOCAL_AUTHOR_FOLLOWS_KEY)
        .and_then(|m| m.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    Some((k.parse::<u64>().ok()?, if v.as_bool()? { 2 } else { 1 }))
                })
                .collect()
        })
        .unwrap_or_default();
    ReplyCandidateExtras {
        negfb_total_count,
        focal_author_follows,
        section: id_keyed_slot(cf_key("section"), 1..=9, as_sent),
        grox_score: id_keyed_slot(cf_key("grox_score_x100"), .., |v| {
            v.saturating_add(GROX_SCORE_OFFSET)
                .clamp(1, i32::MAX as i64)
        }),
        author_verified: id_keyed_slot(cf_key("author_verified"), 1..=4, as_sent),
        fav_by_root_author: id_keyed_slot(cf_key("fav_by_root_author"), 1..=2, as_sent),
        focal_author_reply: HashMap::new(),
        self_reply: HashMap::new(),
    }
}

impl ReplyMetadata {
    pub fn parse(metadata: &str) -> Option<Self> {
        if !metadata.contains("focal_post_id") {
            return None;
        }
        let v = serde_json::from_str::<serde_json::Value>(metadata).ok()?;
        let nonzero_u64 = |key: &str| v.get(key)?.as_u64().filter(|&id| id != 0);
        let focal_post_id = nonzero_u64("focal_post_id")?;
        let focal_semantic_ids = v
            .get("focal_semantic_ids")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|c| c.as_i64().and_then(|c| i32::try_from(c).ok()))
                    .collect::<Option<Vec<i32>>>()
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        let author_follower_counts = v
            .get("author_follower_counts")
            .and_then(|m| m.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| Some((k.parse::<u64>().ok()?, v.as_u64()?)))
                    .collect()
            })
            .unwrap_or_default();
        let focal_features = v
            .get("focal_features")
            .and_then(|f| f.as_object())
            .map(|f| {
                let get = |k: &str| f.get(k).and_then(|x| x.as_i64()).unwrap_or(0);
                FocalFeatureValues {
                    fav_count: get("fav_count"),
                    reply_count: get("reply_count"),
                    view_count: get("view_count"),
                    author_follower_count: get("author_follower_count"),
                    language_code: get("language_code"),
                }
            })
            .unwrap_or_default();
        Some(Self {
            focal_post_id,
            focal_author_id: nonzero_u64("focal_author_id").unwrap_or(0),
            focal_semantic_ids,
            author_follower_counts,
            focal_features,
            extras: parse_extras(&v),
        })
    }

    pub fn author_follower_count(&self, author_id: u64) -> u64 {
        self.author_follower_counts
            .get(&author_id)
            .copied()
            .unwrap_or(0)
    }

    pub fn author_relations(
        &self,
        candidate_set: &pb::CandidateSet,
    ) -> (HashMap<u64, i64>, HashMap<u64, i64>) {
        derive_author_relations(candidate_set, self.focal_author_id)
    }
}

fn record_user_fill(feature: &str, filled: bool) {
    USER_FEATURE_FILL_TOTAL
        .with_label_values(&[feature, if filled { "filled" } else { "missing" }])
        .inc();
}

fn record_candidate_fill(feature: &str, filled: u64, total: u64) {
    CANDIDATE_FEATURE_FILL_TOTAL
        .with_label_values(&[feature, "filled"])
        .inc_by(filled);
    CANDIDATE_FEATURE_FILL_TOTAL
        .with_label_values(&[feature, "missing"])
        .inc_by(total.saturating_sub(filled));
}

pub(crate) fn is_reply_width(n_post_int64: usize) -> bool {
    n_post_int64 > REPLY_SELF_REPLY_INT64_INDEX
}

impl InputBuffer {
    pub fn apply_reply_features(
        &mut self,
        model_config: &ModelConfig,
        candidate_set: &pb::CandidateSet,
        sequence: Option<&pb::UserActionSequence>,
        reply: &ReplyMetadata,
    ) {
        REQUEST_NUM_CANDIDATES.observe(candidate_set.candidates.len() as f64);
        self.fill_focal_user_int64(candidate_set, sequence, reply);
        self.apply_focal_post(model_config, reply.focal_post_id, reply.focal_author_id);
        self.apply_focal_sids(model_config, &reply.focal_semantic_ids);
        self.stamp_reply_candidate_features(model_config, candidate_set, reply);
    }

    fn apply_focal_post(
        &mut self,
        model_config: &ModelConfig,
        focal_post_id: u64,
        focal_author_id: u64,
    ) {
        let hash_table = &model_config.hash_table;
        for (k, slot) in self
            .history_post_hashes
            .iter_mut()
            .take(hash_table.num_item_hashes())
            .enumerate()
        {
            *slot = hash_table.hash_item_id(focal_post_id as i64, k);
        }
        for (k, slot) in self
            .history_auth_hashes
            .iter_mut()
            .take(hash_table.num_author_hashes())
            .enumerate()
        {
            *slot = if focal_author_id != 0 {
                hash_table.hash_author_id(focal_author_id as i64, k)
            } else {
                0
            };
        }
    }

    fn apply_focal_sids(&mut self, model_config: &ModelConfig, codes: &[i32]) {
        let levels = model_config.sid_num_levels;
        for (d, slot) in self
            .history_semantic_ids
            .iter_mut()
            .take(levels)
            .enumerate()
        {
            *slot = if codes.len() == levels {
                (codes[d] + 1).max(0) as u16
            } else {
                0
            };
        }
    }

    fn fill_focal_user_int64(
        &mut self,
        candidate_set: &pb::CandidateSet,
        sequence: Option<&pb::UserActionSequence>,
        reply: &ReplyMetadata,
    ) {
        let int64_features = &mut self.user_int64_features;
        if int64_features.len() <= FOCAL_IN_VIEWER_HISTORY {
            return;
        }
        let focal_post_id = reply.focal_post_id as i64;
        int64_features[FOCAL_POST_CREATION_TS_SEC] =
            snowflake_to_creation_ts_sec(focal_post_id) as i64;
        let in_history = self.history_post_ids.contains(&focal_post_id);
        int64_features[FOCAL_IN_VIEWER_HISTORY] = if in_history { 2 } else { 1 };
        let focal_author_id = reply.focal_author_id as i64;
        if focal_author_id > 0 {
            let user_id = if candidate_set.user_id != 0 {
                candidate_set.user_id as i64
            } else {
                sequence.map(|s| s.user_id as i64).unwrap_or(0)
            };
            int64_features[VIEWER_IS_FOCAL_AUTHOR] = if user_id != 0 && user_id == focal_author_id {
                2
            } else {
                1
            };
        }
        let focal_features = &reply.focal_features;
        let i32_max = i32::MAX as i64;
        int64_features[FOCAL_FAV_COUNT] = focal_features.fav_count.clamp(0, i32_max);
        int64_features[FOCAL_REPLY_COUNT] = focal_features.reply_count.clamp(0, i32_max);
        int64_features[FOCAL_VIEW_COUNT] = focal_features.view_count.clamp(0, i32_max);
        int64_features[FOCAL_AUTHOR_FOLLOWER_COUNT] =
            focal_features.author_follower_count.clamp(0, i32_max);
        int64_features[FOCAL_POST_LANGUAGE_CODE] = focal_features.language_code.clamp(0, i32_max);
        record_user_fill(
            "focal_creation_ts",
            int64_features[FOCAL_POST_CREATION_TS_SEC] != 0,
        );
        record_user_fill("focal_fav_count", int64_features[FOCAL_FAV_COUNT] != 0);
        record_user_fill("focal_reply_count", int64_features[FOCAL_REPLY_COUNT] != 0);
        record_user_fill("focal_view_count", int64_features[FOCAL_VIEW_COUNT] != 0);
        record_user_fill(
            "focal_author_follower_count",
            int64_features[FOCAL_AUTHOR_FOLLOWER_COUNT] != 0,
        );
        record_user_fill(
            "focal_language_code",
            int64_features[FOCAL_POST_LANGUAGE_CODE] != 0,
        );
    }

    fn stamp_reply_candidate_features(
        &mut self,
        model_config: &ModelConfig,
        candidate_set: &pb::CandidateSet,
        reply: &ReplyMetadata,
    ) {
        let n_post_cat = model_config.hash_table.num_post_categorical_features;
        let n_post_int64 = model_config.hash_table.num_post_int64_features;
        let candidates_to_process = candidate_set
            .candidates
            .len()
            .min(model_config.candidate_seq_len);
        let candidates = &candidate_set.candidates[..candidates_to_process];

        if n_post_cat > POST_LANGUAGE_CODE_SEQ {
            let language_codes: Vec<i32> = candidates.iter().map(|c| c.language_code).collect();
            stamp_i32_as_categorical(
                &language_codes,
                &mut self.candidate_categorical_features,
                n_post_cat,
                POST_LANGUAGE_CODE_SEQ,
            );
            let lang_filled = language_codes.iter().filter(|&&c| c != 0).count() as u64;
            record_candidate_fill("post_language", lang_filled, candidates_to_process as u64);
        }

        if !is_reply_width(n_post_int64) {
            return;
        }
        let extras = &reply.extras;
        let (focal_author_reply, self_reply) = reply.author_relations(candidate_set);
        let mut video_filled = 0u64;
        let mut followed_filled = 0u64;
        let mut has_media_filled = 0u64;
        let mut follower_filled = 0u64;
        let mut negfb_filled = 0u64;
        let mut focal_follows_filled = 0u64;
        let mut placement_filled = [0u64; REPLY_PLACEMENT_INT64_INDICES.len()];
        for (j, candidate) in candidates.iter().enumerate() {
            let base = j * n_post_int64;
            self.candidate_int64_features[base + FIRST_VIDEO_DURATION_MS_SEQ] =
                candidate.first_video_duration_ms.min(i32::MAX as u64) as i64;
            video_filled += (candidate.first_video_duration_ms != 0) as u64;
            let has_media = candidate
                .tweet_bool_features
                .as_ref()
                .map(|b| b.has_media)
                .unwrap_or(false);
            self.candidate_int64_features[base + REPLY_HAS_MEDIA_INT64_INDEX] = has_media as i64;
            followed_filled += candidate.is_author_followed_by_user as u64;
            has_media_filled += has_media as u64;
            let author_follower_count = reply.author_follower_count(candidate.author_id);
            follower_filled += (author_follower_count != 0) as u64;
            self.candidate_int64_features[base + REPLY_AUTHOR_FOLLOWER_COUNT_INT64_INDEX] =
                author_follower_count.min(i32::MAX as u64) as i64;
            let negfb = extras
                .negfb_total_count
                .get(&candidate.tweet_id)
                .copied()
                .unwrap_or(0);
            negfb_filled += (negfb != 0) as u64;
            self.candidate_int64_features[base + REPLY_NEGFB_TOTAL_COUNT_INT64_INDEX] =
                negfb.clamp(0, i32::MAX as i64);
            let focal_follows = extras
                .focal_author_follows
                .get(&candidate.tweet_id)
                .copied()
                .unwrap_or(0);
            focal_follows_filled += (focal_follows != 0) as u64;
            self.candidate_int64_features[base + REPLY_FOCAL_AUTHOR_FOLLOWS_INT64_INDEX] =
                focal_follows;
            let values =
                extras.placement_with(&focal_author_reply, &self_reply, candidate.tweet_id);
            for (k, &index) in REPLY_PLACEMENT_INT64_INDICES.iter().enumerate() {
                placement_filled[k] += (values[k] != 0) as u64;
                self.candidate_int64_features[base + index] = values[k];
            }
        }
        let followed_label = if candidate_set.user_id != 0 {
            "is_author_followed_logged_in"
        } else {
            "is_author_followed_guest"
        };
        for (feature, filled) in [
            ("first_video_duration_ms", video_filled),
            (followed_label, followed_filled),
            ("has_media", has_media_filled),
            ("author_follower_count", follower_filled),
            ("negfb_total_count", negfb_filled),
            ("focal_author_follows", focal_follows_filled),
        ] {
            record_candidate_fill(feature, filled, candidates_to_process as u64);
        }
        for (feature, filled) in PLACEMENT_FEATURES.into_iter().zip(placement_filled) {
            record_candidate_fill(feature, filled, candidates_to_process as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_slot_encoding() {
        let m = ReplyMetadata::parse(
            r#"{"focal_post_id":7,
                "candidate_features":{
                  "negfb_report":{"11":2,"12":0},
                  "negfb_block":{"11":1},
                  "negfb_mute":{"13":4},
                  "section":{"11":1,"12":9,"13":0,"14":10},
                  "grox_score_x100":{"11":-250,"12":0,"13":-5000,"14":99999999999},
                  "author_verified":{"11":1,"12":4,"13":5},
                  "fav_by_root_author":{"11":2,"12":1,"13":3}},
                "focal_author_follows":{"11":true,"12":false,"13":1}}"#,
        )
        .unwrap();
        let e = &m.extras;
        assert_eq!(e.negfb_total_count, [(11, 4), (12, 1), (13, 5)].into());
        assert_eq!(e.focal_author_follows, [(11, 2), (12, 1)].into());
        assert_eq!(e.section, [(11, 1), (12, 9)].into());
        assert_eq!(
            e.grox_score,
            [(11, 751), (12, 1001), (13, 1), (14, i32::MAX as i64)].into()
        );
        assert_eq!(e.author_verified, [(11, 1), (12, 4)].into());
        assert_eq!(e.fav_by_root_author, [(11, 2), (12, 1)].into());
    }

    #[test]
    fn candidate_extras_derives_author_relations_from_the_candidate_set() {
        let info = |tweet_id, author_id, in_reply_to_tweet_id| pb::TweetInfo {
            tweet_id,
            author_id,
            in_reply_to_tweet_id,
            ..Default::default()
        };
        let cs = pb::CandidateSet {
            candidates: vec![info(1, 5, 7), info(2, 9, 1), info(3, 6, 7), info(4, 9, 99)],
            ..Default::default()
        };
        let m = ReplyMetadata::parse(r#"{"focal_post_id":7,"focal_author_id":9}"#).unwrap();
        let (focal_author_reply, self_reply) = m.author_relations(&cs);
        assert_eq!(focal_author_reply, [(1, 2), (2, 1), (3, 1), (4, 1)].into());
        assert_eq!(self_reply, [(1, 1), (2, 2), (3, 1), (4, 2)].into());

        let m = ReplyMetadata::parse(r#"{"focal_post_id":7}"#).unwrap();
        let (focal_author_reply, self_reply) = m.author_relations(&cs);
        assert!(focal_author_reply.is_empty() && self_reply.is_empty());
    }

    #[test]
    fn stamps_the_reply_layout() {
        let mut model_config = ModelConfig::from_params(
            8,
            vec![1],
            vec![0],
            1009,
            8,
            0,
            vec![1],
            vec![0],
            1009,
            8,
            vec![1],
            vec![0],
            1009,
            4,
            2,
            4,
            1,
            0,
            0,
            0,
            vec![],
            vec![],
            1009,
            0,
            7,
            0,
            4,
            FOCAL_IN_VIEWER_HISTORY + 1,
            0,
            POST_LANGUAGE_CODE_SEQ + 1,
            0,
            0,
            REPLY_SELF_REPLY_INT64_INDEX + 1,
            false,
        );
        model_config.sid_num_levels = 3;
        let focal_post_id: i64 = 1_900_000_000_000_000_000;
        let candidate_set = pb::CandidateSet {
            user_id: 77,
            candidates: vec![pb::TweetInfo {
                tweet_id: 1000,
                author_id: 77,
                language_code: pb::LanguageCode::LangJa as i32,
                first_video_duration_ms: 15_000,
                tweet_bool_features: Some(pb::TweetBoolFeatures {
                    has_media: true,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let reply = ReplyMetadata::parse(
            r#"{"focal_post_id":1900000000000000000,"focal_author_id":77,
                "focal_semantic_ids":[0,7,255],
                "focal_features":{"fav_count":1000,"view_count":5000000000},
                "author_follower_counts":{"77":123456},
                "candidate_features":{"negfb_mute":{"1000":4},"section":{"1000":3},
                  "grox_score_x100":{"1000":-250},"author_verified":{"1000":4},
                  "fav_by_root_author":{"1000":2}},
                "focal_author_follows":{"1000":true}}"#,
        )
        .unwrap();

        let mut buf = InputBuffer::compute_for_item(
            &model_config,
            &None,
            &candidate_set,
            None,
            None,
            None,
            None,
        );
        buf.apply_reply_features(&model_config, &candidate_set, None, &reply);

        let creation_ts = snowflake_to_creation_ts_sec(focal_post_id) as i64;
        assert_eq!(
            buf.user_int64_features,
            [creation_ts, 1000, 0, i32::MAX as i64, 0, 0, 2, 1]
        );
        assert_eq!(
            buf.history_post_hashes[0],
            model_config.hash_table.hash_item_id(focal_post_id, 0)
        );
        assert_eq!(&buf.history_semantic_ids[..3], &[1, 8, 256]);
        assert_eq!(
            buf.candidate_categorical_features[POST_LANGUAGE_CODE_SEQ],
            pb::LanguageCode::LangJa as i16
        );
        let int64 = &buf.candidate_int64_features;
        assert_eq!(int64[REPLY_HAS_MEDIA_INT64_INDEX], 1);
        assert_eq!(
            &int64[FIRST_VIDEO_DURATION_MS_SEQ..],
            &[15_000, 123_456, 5, 2, 3, 751, 4, 2, 1, 2]
        );
    }
}
