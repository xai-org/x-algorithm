use crate::models::candidate::PostCandidate;
use crate::models::engagement_signals::EngagementSignalsByType;
use crate::models::in_network_reply::{serialize_in_network_replies, InNetworkReplies};
use crate::models::user_features::UserFeatures;
use serde::Serialize;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use xai_candidate_pipeline::candidate_pipeline::PipelineQuery;
use xai_core_entities::entities::SubscriptionLevel;
use xai_decider::Decider;
use xai_feature_switches::Params;
use xai_recsys_proto::gender_prediction::InferredGenderLabel;
use xai_recsys_proto::{DeviceNetworkType, Timezone};
use xai_twittercontext_proto::{GetTwitterContextViewer, TwitterContextViewer};
use xai_urt_thrift::cursor::UrtOrderedCursor;
use xai_x_thrift::non_polling_timestamps::NonPollingTimestamps;
use xai_x_thrift::served_history::ServedHistory;

#[derive(Clone, Debug)]
pub struct ImpressionBloomFilterEntry {
    pub bloom_filter: Vec<u64>,
    pub size_cap: i32,
    pub false_positive_rate: f64,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, strum::Display)]
pub enum RequestType {
    #[default]
    ForYou,
    RankedFollowing,
    Following,
    ScoredPosts,
    PhoenixScores,
}

#[derive(Clone, Default, Debug)]
pub struct FollowingPaginationMeta {
    pub response_truncated: bool,
    pub bottom_tweet_id: Option<u64>,
}

#[derive(Clone, Default, Debug, Serialize)]
pub struct ScoredPostsQuery {
    pub user_id: u64,
    pub client_app_id: i32,
    pub country_code: String,
    pub language_code: String,
    pub seen_ids: Vec<u64>,
    pub served_ids: Vec<u64>,
    pub in_network_only: bool,
    pub request_type: RequestType,
    pub is_bottom_request: bool,
    pub is_top_request: bool,
    #[serde(skip)]
    pub bloom_filter_entries: Vec<ImpressionBloomFilterEntry>,
    #[serde(skip)]
    pub scoring_sequence: Option<xai_recsys_proto::UserActionSequence>,
    #[serde(skip)]
    pub columnar_scoring_sequence: Option<bytes::Bytes>,
    #[serde(skip)]
    pub retrieval_sequence: Option<xai_recsys_proto::UserActionSequence>,
    #[serde(skip)]
    pub columnar_retrieval_sequence: Option<bytes::Bytes>,
    pub user_features: UserFeatures,
    pub user_roles: Vec<String>,
    pub params: Params,
    #[serde(skip)]
    pub decider: Option<Decider>,
    pub request_id: u64,
    pub prediction_id: u64,
    pub request_time_ms: i64,
    pub cached_posts: Vec<PostCandidate>,
    pub has_cached_posts: bool,
    pub return_backbone_scores: bool,
    pub topic_ids: Vec<i64>,
    pub excluded_topic_ids: Vec<i64>,
    pub exclude_videos: bool,
    #[serde(serialize_with = "serialize_in_network_replies")]
    pub in_network_replies: InNetworkReplies,
    pub viewer_minhash: Option<Vec<i64>>,
    pub ip_address: String,
    pub user_agent: String,
    #[serde(serialize_with = "serialize_debug")]
    pub time_zone: Timezone,
    #[serde(serialize_with = "serialize_debug")]
    pub device_network_type: DeviceNetworkType,
    pub client_version: String,
    pub device_id: String,
    pub mobile_device_id: String,
    pub mobile_device_ad_id: String,
    #[serde(skip)]
    pub dsp_client_context: Option<xai_recsys_proto::DspClientContext>,
    pub user_demographics: Option<UserDemographics>,
    pub ip_location: Option<xai_geo_ip::LocationInfo>,
    pub user_age_in_years: Option<i32>,
    pub resurrection_time_ms: Option<i64>,
    #[serde(serialize_with = "serialize_debug")]
    pub user_inferred_gender: Option<InferredGenderLabel>,
    pub user_inferred_gender_score: Option<f32>,
    pub followed_grok_topics: Option<[bool; 32]>,
    pub followed_starter_packs: Option<[bool; 20]>,
    pub explicit_engagement_signals: Option<EngagementSignalsByType>,
    pub implicit_engagement_signals: Option<EngagementSignalsByType>,
    #[serde(serialize_with = "serialize_debug")]
    pub user_installed_apps: Option<Vec<bool>>,
    pub subscription_level: Option<SubscriptionLevel>,
    pub is_shadow_traffic: bool,
    pub is_preview: bool,
    pub is_polling: bool,
    #[serde(serialize_with = "serialize_debug")]
    pub cursor: Option<UrtOrderedCursor>,
    pub request_context: String,
    #[serde(skip)]
    pub served_history: Vec<ServedHistory>,
    /// Distinguishes a successful empty history from a failed/skipped read.
    #[serde(skip)]
    pub served_history_loaded: bool,
    pub who_to_follow_eligible: bool,
    pub feed_survey_eligible: bool,
    #[serde(serialize_with = "serialize_debug")]
    pub non_polling_timestamps: Option<NonPollingTimestamps>,
    pub impressed_post_ids: Vec<u64>,
    pub push_to_home_post_id: Option<u64>,
    pub seed_candidate_post_ids: Vec<u64>,
    #[serde(serialize_with = "serialize_debug")]
    pub following_pagination_meta: Arc<OnceLock<FollowingPaginationMeta>>,
}

pub use xai_candidate_pipeline::component_library::clients::strato_client::UserDemographics;

impl ScoredPostsQuery {
    pub fn new(
        user_id: u64,
        client_app_id: i32,
        country_code: String,
        language_code: String,
        seen_ids: Vec<u64>,
        served_ids: Vec<u64>,
        in_network_only: bool,
        is_bottom_request: bool,
        params: Params,
        decider: Decider,
        user_roles: Vec<String>,
        muted_keywords: Vec<String>,
        follower_count: Option<i64>,
        topic_ids: Vec<i64>,
        excluded_topic_ids: Vec<i64>,
        exclude_videos: bool,
        request_id: u64,
        prediction_id: u64,
        ip_address: String,
        user_agent: String,
        time_zone: Timezone,
        device_network_type: DeviceNetworkType,
        client_version: String,
        device_id: String,
        mobile_device_id: String,
        mobile_device_ad_id: String,
        subscription_level: Option<SubscriptionLevel>,
        is_shadow_traffic: bool,
        is_preview: bool,
        age_in_years: Option<i32>,
        push_to_home_post_id: Option<u64>,
    ) -> Self {
        Self {
            user_id,
            client_app_id,
            country_code,
            language_code,
            seen_ids,
            served_ids,
            in_network_only,
            request_type: RequestType::ForYou,
            is_bottom_request,
            is_top_request: false,
            bloom_filter_entries: vec![],
            scoring_sequence: None,
            columnar_scoring_sequence: None,
            retrieval_sequence: None,
            columnar_retrieval_sequence: None,
            user_features: UserFeatures {
                muted_keywords,
                follower_count,
                ..Default::default()
            },
            user_roles,
            params,
            decider: Some(decider),
            request_id,
            prediction_id,
            request_time_ms: current_time_ms(),
            cached_posts: vec![],
            has_cached_posts: false,
            return_backbone_scores: false,
            topic_ids,
            excluded_topic_ids,
            exclude_videos,
            in_network_replies: Default::default(),
            viewer_minhash: None,
            ip_address,
            user_agent,
            time_zone,
            device_network_type,
            client_version,
            device_id,
            mobile_device_id,
            mobile_device_ad_id,
            dsp_client_context: None,
            user_demographics: None,
            ip_location: None,
            user_age_in_years: age_in_years,
            resurrection_time_ms: None,
            user_inferred_gender: None,
            user_inferred_gender_score: None,
            followed_grok_topics: None,
            followed_starter_packs: None,
            explicit_engagement_signals: None,
            implicit_engagement_signals: None,
            user_installed_apps: None,
            subscription_level,
            is_shadow_traffic,
            is_preview,
            is_polling: false,
            cursor: None,
            request_context: String::new(),
            served_history: vec![],
            served_history_loaded: false,
            who_to_follow_eligible: false,
            feed_survey_eligible: false,
            non_polling_timestamps: None,
            impressed_post_ids: Vec::new(),
            push_to_home_post_id,
            seed_candidate_post_ids: Vec::new(),
            following_pagination_meta: Arc::new(OnceLock::new()),
        }
    }

    pub fn is_topic_request(&self) -> bool {
        !self.topic_ids.is_empty()
    }

    pub fn is_bulk_topic_request(&self) -> bool {
        self.topic_ids.len() > 6
    }

    pub fn has_excluded_topics(&self) -> bool {
        !self.excluded_topic_ids.is_empty()
    }
}

impl GetTwitterContextViewer for ScoredPostsQuery {
    fn get_viewer(&self) -> Option<TwitterContextViewer> {
        Some(TwitterContextViewer {
            user_id: self.user_id as i64,
            client_application_id: self.client_app_id as i64,
            request_country_code: self.country_code.clone(),
            request_language_code: self.language_code.clone(),
            ..Default::default()
        })
    }
}

impl PipelineQuery for ScoredPostsQuery {
    fn params(&self) -> &Params {
        &self.params
    }

    fn decider(&self) -> Option<&Decider> {
        self.decider.as_ref()
    }
}

pub fn current_time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn serialize_debug<S, T: std::fmt::Debug>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&format!("{:?}", value))
}
