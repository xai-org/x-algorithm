use crate::models::brand_safety::BrandSafetyVerdict;
use crate::models::content_features;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster;
pub use xai_candidate_pipeline::component_library::models::PhoenixScores;
use xai_home_mixer_proto as pb;
use xai_proto::engagement_counter::PostEngagers;
use xai_recsys_proto::SAFETY_BIT_AUTHOR_NSFW;
use xai_visibility_filtering::models as vf;

#[derive(Clone, Debug)]
pub struct RankerVmRank {
    pub response: xai_vm_ranker_proto::RankResponse,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PostCandidate {
    pub tweet_id: u64,
    pub author_id: u64,
    pub tweet_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_reply_to_tweet_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retweeted_tweet_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retweeted_user_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_tweet_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_user_id: Option<u64>,
    pub phoenix_scores: PhoenixScores,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction_request_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_scored_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weighted_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cold_start_lift_to_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slate_context: Option<Box<SlateContext>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_heads: Vec<(String, f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub served_slate_context: Option<Box<SlateContext>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reranker_head_tag: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub page_decode_slot: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub page_decode_prob: Option<f64>,
    #[serde(skip)]
    pub ranker_vm_rank: Option<Arc<RankerVmRank>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub page_rl_exploration_prob: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backbone_scores: Option<Box<PhoenixScores>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub immersive_scores: Option<PhoenixScores>,
    #[serde(skip)]
    pub video_embedding: Option<Vec<f32>>,
    #[serde(
        serialize_with = "serialize_served_type",
        deserialize_with = "deserialize_served_type"
    )]
    pub served_type: Option<pb::ServedType>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieval_sources: Vec<RetrievalSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_network: Option<bool>,
    pub ancestors: Vec<u64>,
    pub tombstone_ancestor_ids: Vec<u64>,
    pub ancestor_users: Vec<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub conversation_id: Option<u64>,
    pub ancestor_texts: HashMap<u64, String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ancestor_has_video: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_tweet_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_video_duration_ms: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_video_duration_ms: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_photo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_video: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub video_aspect_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub video_media_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub video_md5_hashes: Option<Vec<String>>,
    #[serde(default)]
    pub video_carousel_extra: bool,
    #[serde(default)]
    pub carousel_only: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_video_duration_ms: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_has_media: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_has_photo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_has_video: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_media_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_max_video_duration_ms: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_followers_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_screen_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retweeted_screen_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility_action: Option<vf::Action>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility_reason: Option<vf::FilteredReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop_ancillary_posts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscription_author_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tweet_type_metrics: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_blocks_viewer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_author_blocks_viewer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filtered_topic_ids: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unfiltered_topic_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub following_replied_user_ids: Vec<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_media: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub broadcast_is_live: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fav_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repost_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quote_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub view_count_on_home: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bookmark_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutual_follow_jaccard: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_mutual_follow_author: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_follows_viewer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brand_safety_verdict: Option<BrandSafetyVerdict>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nsfw_author: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nsfw_author_ads: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nsfw_author_phoenix: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft_nsfw: Option<bool>,
    #[serde(default)]
    pub safety_labels: Vec<SafetyLabelInfo>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub semantic_ids: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic_feedback_topic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic_feedback_topic_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grok_topics: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_trend_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_trend_id: Option<String>,
    #[serde(skip)]
    pub post_engagers: Option<Arc<PostEngagers>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalSource {
    #[serde(
        serialize_with = "serialize_served_type_value",
        deserialize_with = "deserialize_served_type_value"
    )]
    pub served_type: pb::ServedType,
    pub cluster: Option<PhoenixRetrievalCluster>,
    pub score: Option<f32>,
    pub position: Option<u32>,
}

impl RetrievalSource {
    pub fn from_served_type(served_type: pb::ServedType) -> Self {
        Self {
            served_type,
            cluster: None,
            score: None,
            position: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SlateContext {
    pub k: u32,
    pub pool_rank: u32,
    pub pool_rank_gap: Option<u32>,
    pub fatigue: f64,
    pub pre_diversity_score: f64,
    pub sid_known: bool,
    pub sid_k_l1: u32,
    pub sid_k_l2: u32,
    pub sid_k_l3: u32,
    pub sid_gap_l1: Option<u32>,
    pub sid_gap_l2: Option<u32>,
    pub sid_gap_l3: Option<u32>,
    #[serde(default)]
    pub recon_cos_milli: Option<u32>,
    #[serde(default)]
    pub recon_count_above: Option<u32>,
    #[serde(default)]
    pub recon_gap_above: Option<u32>,
    #[serde(default)]
    pub exact_k: Option<u32>,
    #[serde(default)]
    pub exact_gap: Option<u32>,
}

impl From<xai_recsys_proto::SlateContext> for SlateContext {
    fn from(c: xai_recsys_proto::SlateContext) -> Self {
        Self {
            k: c.k,
            pool_rank: c.pool_rank,
            pool_rank_gap: c.pool_rank_gap,
            fatigue: c.fatigue,
            pre_diversity_score: c.pre_diversity_score,
            sid_known: c.sid_known,
            sid_k_l1: c.sid_k1,
            sid_k_l2: c.sid_k2,
            sid_k_l3: c.sid_k3,
            sid_gap_l1: c.sid_gap1,
            sid_gap_l2: c.sid_gap2,
            sid_gap_l3: c.sid_gap3,
            recon_cos_milli: c.recon_cos_milli,
            recon_count_above: c.recon_count_above,
            recon_gap_above: c.recon_gap_above,
            exact_k: c.exact_k,
            exact_gap: c.exact_gap,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyLabelInfo {
    #[serde(with = "xai_safety_label_store::types::serde_label_type")]
    pub label_type: xai_x_thrift::tweet_safety_label::SafetyLabelType,
    pub description: Option<String>,
    pub source: Option<String>,
}

fn serialize_served_type<S>(
    served_type: &Option<pb::ServedType>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    served_type.map(|value| value as i32).serialize(serializer)
}

fn deserialize_served_type<'de, D>(deserializer: D) -> Result<Option<pb::ServedType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<i32>::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(raw_value) => pb::ServedType::try_from(raw_value)
            .map(Some)
            .map_err(|_| serde::de::Error::custom("invalid ServedType value")),
    }
}

fn serialize_served_type_value<S>(
    served_type: &pb::ServedType,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    (*served_type as i32).serialize(serializer)
}

fn deserialize_served_type_value<'de, D>(deserializer: D) -> Result<pb::ServedType, D::Error>
where
    D: serde::Deserializer<'de>,
{
    pb::ServedType::try_from(i32::deserialize(deserializer)?)
        .map_err(|_| serde::de::Error::custom("invalid ServedType value"))
}

pub trait CandidateHelpers {
    fn get_screen_names(&self) -> HashMap<u64, String>;
    fn get_original_tweet_id(&self) -> u64;
    fn get_original_author_id(&self) -> u64;
    fn semantic_id_prefix(&self, levels: usize) -> Option<&[i32]>;
    fn as_tweet_info(&self, is_followed_by_viewer: bool) -> xai_recsys_proto::TweetInfo;
    fn as_score_info_no_prediction_scores(&self) -> xai_recsys_proto::ScoreInfo;
}

impl CandidateHelpers for PostCandidate {
    fn get_screen_names(&self) -> HashMap<u64, String> {
        let mut screen_names = HashMap::<u64, String>::new();
        if let Some(author_screen_name) = self.author_screen_name.clone() {
            screen_names.insert(self.author_id, author_screen_name);
        }
        if let (Some(retweeted_screen_name), Some(retweeted_user_id)) =
            (self.retweeted_screen_name.clone(), self.retweeted_user_id)
        {
            screen_names.insert(retweeted_user_id, retweeted_screen_name);
        }
        screen_names
    }

    fn get_original_tweet_id(&self) -> u64 {
        self.retweeted_tweet_id.unwrap_or(self.tweet_id)
    }

    fn get_original_author_id(&self) -> u64 {
        self.retweeted_user_id.unwrap_or(self.author_id)
    }

    fn semantic_id_prefix(&self, levels: usize) -> Option<&[i32]> {
        self.semantic_ids
            .as_deref()
            .filter(|codes| codes.len() >= levels && levels > 0)
            .map(|codes| &codes[..levels])
    }

    fn as_score_info_no_prediction_scores(&self) -> xai_recsys_proto::ScoreInfo {
        xai_recsys_proto::ScoreInfo {
            prediction_scores: Default::default(),
            weighted_score: self.weighted_score,
            final_score: self.score,
            slate_context: self
                .slate_context
                .as_deref()
                .map(|c| xai_recsys_proto::SlateContext {
                    k: c.k,
                    pool_rank: c.pool_rank,
                    pool_rank_gap: c.pool_rank_gap,
                    fatigue: c.fatigue,
                    pre_diversity_score: c.pre_diversity_score,
                    sid_known: c.sid_known,
                    sid_k1: c.sid_k_l1,
                    sid_k2: c.sid_k_l2,
                    sid_k3: c.sid_k_l3,
                    sid_gap1: c.sid_gap_l1,
                    sid_gap2: c.sid_gap_l2,
                    sid_gap3: c.sid_gap_l3,
                    recon_cos_milli: c.recon_cos_milli,
                    recon_count_above: c.recon_count_above,
                    recon_gap_above: c.recon_gap_above,
                    exact_k: c.exact_k,
                    exact_gap: c.exact_gap,
                }),
            reward_rerank_slot_prob: None,
            page_decode_slot_prob: None,
            reranker_head_tag: self.reranker_head_tag,
            prediction_heads: Vec::new(),
            prediction_values: Vec::new(),
            page_decoded: None,
        }
    }

    fn as_tweet_info(&self, is_followed_by_viewer: bool) -> xai_recsys_proto::TweetInfo {
        xai_recsys_proto::TweetInfo {
            tweet_id: self.get_original_tweet_id(),
            author_id: self.get_original_author_id(),
            retweeting_tweet_id: if self.retweeted_tweet_id.is_some() {
                self.tweet_id
            } else {
                0
            },
            retweeting_author_id: if self.retweeted_user_id.is_some() {
                self.author_id
            } else {
                0
            },
            quoted_tweet_id: self.quoted_tweet_id.unwrap_or(0),
            quoted_author_id: self.quoted_user_id.unwrap_or(0),
            in_reply_to_tweet_id: self.in_reply_to_tweet_id.unwrap_or(0),
            is_author_followed_by_user: is_followed_by_viewer,
            safety_label_mask: if self.retweeted_user_id.is_none()
                && self.nsfw_author_phoenix.unwrap_or(false)
            {
                SAFETY_BIT_AUTHOR_NSFW
            } else {
                0
            },
            min_video_duration_ms: self.min_video_duration_ms.map(|ms| ms as u64).unwrap_or(0),
            fav_count: self.fav_count.unwrap_or(0) as u64,
            retweet_count: self.repost_count.unwrap_or(0) as u64,
            quote_count: self.quote_count.unwrap_or(0) as u64,
            reply_count: self.reply_count.unwrap_or(0) as u64,
            view_count: self.view_count.unwrap_or(0),
            bookmark_count: self.bookmark_count.unwrap_or(0) as u64,
            language_code: xai_recsys_proto::language_code_string_to_enum(
                self.language_code.as_deref().unwrap_or(""),
            ) as i32,
            tweet_bool_features: Some(xai_recsys_proto::TweetBoolFeatures {
                has_media: self.has_media.unwrap_or(false),
                is_retweet: self.retweeted_tweet_id.is_some(),
                is_quote: self.quoted_tweet_id.is_some(),
                is_reply: self.in_reply_to_tweet_id.is_some(),
                ..Default::default()
            }),
            author_info: Some(xai_recsys_proto::AuthorInfo {
                author_id: self.get_original_author_id(),
                is_followed_by_user: is_followed_by_viewer,
                is_following_user: if self.retweeted_user_id.is_none() {
                    self.author_follows_viewer
                } else {
                    None
                },
                followers: if self.retweeted_user_id.is_none() {
                    self.author_followers_count.map(|c| c.max(0) as u64)
                } else {
                    None
                },
            }),
            semantic_ids: self.semantic_ids.clone().unwrap_or_default(),
            content_features: Some(content_features::build(self)),
            quoted_content_features: content_features::build_quoted(self),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_label_info_serde_roundtrip() {
        use xai_x_thrift::tweet_safety_label::SafetyLabelType;

        let info = SafetyLabelInfo {
            label_type: SafetyLabelType::NSFW_HIGH_PRECISION,
            description: Some("test desc".to_string()),
            source: Some("Content".to_string()),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"label_type\":3"), "got: {json}");

        let deserialized: SafetyLabelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.label_type,
            SafetyLabelType::NSFW_HIGH_PRECISION
        );
        assert_eq!(deserialized.description, Some("test desc".to_string()));
        assert_eq!(deserialized.source, Some("Content".to_string()));
    }

    #[test]
    fn safety_label_info_deserializes_from_i32() {
        use xai_x_thrift::tweet_safety_label::SafetyLabelType;

        let json = r#"{"label_type":1,"description":null,"source":null}"#;
        let info: SafetyLabelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.label_type, SafetyLabelType::SPAM);
    }

    #[test]
    fn boxed_reranker_fields_keep_their_serialized_form() {
        let context = SlateContext {
            k: 2,
            pool_rank: 7,
            ..Default::default()
        };
        let scores = PhoenixScores {
            favorite_score: Some(0.5),
            ..Default::default()
        };
        let candidate = PostCandidate {
            slate_context: Some(Box::new(context)),
            served_slate_context: Some(Box::new(context)),
            backbone_scores: Some(Box::new(scores.clone())),
            ..Default::default()
        };

        let value = serde_json::to_value(&candidate).unwrap();
        assert_eq!(
            value["slate_context"],
            serde_json::to_value(context).unwrap()
        );
        assert_eq!(
            value["served_slate_context"],
            serde_json::to_value(context).unwrap()
        );
        assert_eq!(
            value["backbone_scores"],
            serde_json::to_value(&scores).unwrap()
        );
        let decoded: PostCandidate = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.slate_context.as_deref(), Some(&context));
        assert_eq!(decoded.served_slate_context.as_deref(), Some(&context));
        assert_eq!(
            decoded.backbone_scores.unwrap().favorite_score,
            scores.favorite_score
        );
    }

    #[test]
    fn post_candidate_deserializes_without_slate_context_field() {
        let mut value = serde_json::to_value(PostCandidate::default()).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.remove("slate_context");
        for field in [
            "has_photo",
            "has_video",
            "media_count",
            "max_video_duration_ms",
            "quoted_has_media",
            "quoted_has_photo",
            "quoted_has_video",
            "quoted_media_count",
            "quoted_max_video_duration_ms",
        ] {
            obj.remove(field);
        }
        obj.insert(
            "field_from_newer_build".to_string(),
            serde_json::json!(true),
        );

        let candidate: PostCandidate = serde_json::from_value(value).unwrap();
        assert_eq!(candidate.slate_context, None);
        assert_eq!(candidate.has_video, None);
        assert_eq!(candidate.media_count, None);
    }

    #[test]
    fn post_candidate_deserializes_cache_entries_without_page_rl_fields() {
        let mut value = serde_json::to_value(PostCandidate {
            tweet_id: 9,
            page_decode_slot: Some(1),
            page_decode_prob: Some(0.5),
            page_rl_exploration_prob: Some(0.25),
            ..Default::default()
        })
        .unwrap();
        let obj = value.as_object_mut().unwrap();
        for field in [
            "page_decode_slot",
            "page_decode_prob",
            "page_rl_exploration_prob",
        ] {
            assert!(obj.remove(field).is_some(), "{field} must be serialized");
        }

        let candidate: PostCandidate = serde_json::from_value(value).unwrap();

        assert_eq!(candidate.tweet_id, 9);
        assert_eq!(candidate.page_decode_slot, None);
        assert_eq!(candidate.page_decode_prob, None);
        assert_eq!(candidate.page_rl_exploration_prob, None);
    }

    #[test]
    fn post_candidate_with_safety_labels_roundtrip() {
        use xai_x_thrift::tweet_safety_label::SafetyLabelType;

        let candidate = PostCandidate {
            tweet_id: 123,
            author_id: 456,
            safety_labels: vec![SafetyLabelInfo {
                label_type: SafetyLabelType::BOUNCE,
                description: None,
                source: None,
            }],
            ..Default::default()
        };

        let json = serde_json::to_string(&candidate).unwrap();
        let deserialized: PostCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.safety_labels.len(), 1);
        assert_eq!(
            deserialized.safety_labels[0].label_type,
            SafetyLabelType::BOUNCE
        );
    }

    #[test]
    fn absent_option_fields_match_explicit_nulls() {
        let candidate = PostCandidate {
            tweet_id: 9,
            author_id: 3,
            tweet_text: "hello".into(),
            retweeted_tweet_id: Some(11),
            in_network: Some(false),
            score: Some(0.0),
            language_code: Some("en".into()),
            ..Default::default()
        };
        let value = serde_json::to_value(&candidate).unwrap();
        let obj = value.as_object().unwrap();
        assert!(obj.get("quoted_tweet_id").is_none());
        assert!(obj.get("slate_context").is_none());
        assert_eq!(obj.get("retweeted_tweet_id"), Some(&serde_json::json!(11)));
        assert_eq!(obj.get("in_network"), Some(&serde_json::json!(false)));
        assert_eq!(obj.get("score"), Some(&serde_json::json!(0.0)));

        let mut with_nulls = value.clone();
        let map = with_nulls.as_object_mut().unwrap();
        for key in [
            "quoted_tweet_id",
            "quoted_user_id",
            "slate_context",
            "served_slate_context",
            "in_reply_to_tweet_id",
            "cold_start_lift_to_rank",
        ] {
            map.insert(key.to_string(), serde_json::Value::Null);
        }
        let from_nulls: PostCandidate = serde_json::from_value(with_nulls).unwrap();
        let from_skipped: PostCandidate = serde_json::from_value(value).unwrap();
        assert_eq!(
            serde_json::to_value(&from_nulls).unwrap(),
            serde_json::to_value(&from_skipped).unwrap()
        );
        assert_eq!(from_skipped.retweeted_tweet_id, Some(11));
        assert_eq!(from_skipped.in_network, Some(false));
        assert_eq!(from_skipped.quoted_tweet_id, None);
    }

    #[test]
    fn phoenix_score_nulls_are_omitted_and_still_deserialize() {
        let scored = PostCandidate {
            tweet_id: 1,
            phoenix_scores: PhoenixScores {
                favorite_score: Some(0.5),
                reply_score: Some(0.0),
                ..Default::default()
            },
            served_type: Some(xai_home_mixer_proto::ServedType::ForYouPhoenixRetrieval),
            ..Default::default()
        };
        let value = serde_json::to_value(&scored).unwrap();
        let heads = value["phoenix_scores"].as_object().unwrap();
        assert_eq!(heads.get("favorite_score"), Some(&serde_json::json!(0.5)));
        assert_eq!(heads.get("reply_score"), Some(&serde_json::json!(0.0)));
        assert!(!heads.contains_key("vqv_score"));
        assert!(!heads.contains_key("dwell_time"));

        let roundtrip: PostCandidate = serde_json::from_value(value).unwrap();
        assert_eq!(roundtrip.phoenix_scores.favorite_score, Some(0.5));
        assert_eq!(roundtrip.phoenix_scores.reply_score, Some(0.0));
        assert_eq!(roundtrip.phoenix_scores.vqv_score, None);

        let mut legacy = serde_json::to_value(&scored).unwrap();
        let heads = legacy["phoenix_scores"].as_object_mut().unwrap();
        heads.insert("vqv_score".into(), serde_json::Value::Null);
        heads.insert("dwell_time".into(), serde_json::Value::Null);
        let from_legacy: PostCandidate = serde_json::from_value(legacy).unwrap();
        assert_eq!(from_legacy.phoenix_scores.favorite_score, Some(0.5));
        assert_eq!(from_legacy.phoenix_scores.vqv_score, None);
        assert_eq!(from_legacy.phoenix_scores.dwell_time, None);
    }
}
