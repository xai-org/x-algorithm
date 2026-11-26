use crate::candidate_pipeline::phoenix_candidate_pipeline::PhoenixCandidatePipeline;
use crate::models::brand_safety::BrandSafetyVerdict;
use crate::models::candidate::CandidateHelpers;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::server::{PipelineOutput, QueryBuilder};
use crate::side_effects::phoenix_request_cache_side_effect::served_post_score_infos;
use crate::util::under_the_hood;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Instant;
use tonic::Status;
use tracing::info;
use xai_candidate_pipeline::candidate_pipeline::{CandidatePipeline, PipelineResult};
use xai_home_mixer_proto::{ScoredPost, VideoCarouselModule};
use xai_stats_receiver::global_stats_receiver;
use xai_x_thrift::tweet_safety_label::SafetyLabelType;

const PRODUCT_SURFACE_METRIC: &str = "ScoredPostsServer.product_surface";
const TOPIC_METRIC: &str = "ScoredPosts.topic";
const SURFACE_RANKED_FOLLOWING: &str = "ranked_following";
const SURFACE_TOPICS: &str = "topics";
const SURFACE_FOR_YOU: &str = "for_you";
const SURFACE_FOR_YOU_WITH_SNOOZED_TOPICS: &str = "for_you_with_snoozed_topics";

pub struct ScoredPostsServer {
    pub(crate) phoenix_candidate_pipeline: Arc<PhoenixCandidatePipeline>,
    pub(crate) query_builder: QueryBuilder,
}

impl ScoredPostsServer {
    pub fn new(
        query_builder: QueryBuilder,
        phoenix_candidate_pipeline: Arc<PhoenixCandidatePipeline>,
    ) -> Self {
        Self {
            phoenix_candidate_pipeline,
            query_builder,
        }
    }

    pub(crate) async fn run_pipeline(
        &self,
        query: ScoredPostsQuery,
    ) -> Result<PipelineOutput, Status> {
        if params::TEST_USER_IDS.contains(&query.user_id) {
            return Ok(PipelineOutput {
                scored_posts: vec![],
                video_carousel: None,
                pipeline_result: PipelineResult::empty(),
                pipeline_trace: None,
            });
        }

        log_request_info(&query);

        let start = Instant::now();

        let (pipeline_result, pipeline_trace) =
            self.phoenix_candidate_pipeline.execute_traced(query).await;

        info!(
            "Scored Posts response - request_id {} - {} posts ({} ms)",
            pipeline_result.query.request_id,
            pipeline_result.selected_candidates.len(),
            start.elapsed().as_millis()
        );

        log_response_stats(&pipeline_result);

        let scored_posts = candidates_to_scored_posts(
            &pipeline_result.query,
            &pipeline_result.selected_candidates,
        );
        if pipeline_result.query.params.get(params::EnablePageLogging) {
            let _ = pipeline_result
                .query
                .served_post_score_infos
                .set(served_post_score_infos(
                    &pipeline_result.query,
                    &pipeline_result.selected_candidates,
                ));
        }
        let video_carousel = pipeline_result
            .query
            .video_carousel
            .get()
            .filter(|videos| !videos.is_empty())
            .map(|videos| VideoCarouselModule {
                videos: candidates_to_scored_posts(&pipeline_result.query, videos),
            });
        Ok(PipelineOutput {
            scored_posts,
            video_carousel,
            pipeline_trace: pipeline_trace
                .filter(|_| pipeline_result.query.is_under_the_hood_request)
                .map(|trace| under_the_hood::pipeline_trace(&trace)),
            pipeline_result,
        })
    }
}


fn candidates_to_scored_posts(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
) -> Vec<ScoredPost> {
    candidates
        .iter()
        .map(|candidate| {
            let screen_names = candidate.get_screen_names();
            ScoredPost {
                tweet_id: candidate.tweet_id,
                author_id: candidate.author_id,
                retweeted_tweet_id: candidate.retweeted_tweet_id.unwrap_or(0),
                retweeted_user_id: candidate.retweeted_user_id.unwrap_or(0),
                in_reply_to_tweet_id: candidate.in_reply_to_tweet_id.unwrap_or(0),
                score: candidate.score.unwrap_or(0.0) as f32,
                in_network: candidate.in_network.unwrap_or(false),
                served_type: candidate.served_type.map(|t| t as i32).unwrap_or_default(),
                last_scored_timestamp_ms: candidate.last_scored_at_ms.unwrap_or(0),
                prediction_request_id: candidate.prediction_request_id.unwrap_or(0),
                ancestors: candidate.ancestors.clone(),
                tombstone_ancestor_ids: candidate.tombstone_ancestor_ids.clone(),
                ancestor_users: candidate.ancestor_users.clone(),
                quoted_user_id: candidate.quoted_user_id.unwrap_or(0),
                screen_names,
                visibility_reason: candidate.visibility_reason.clone().map(|r| r.into()),
                tweet_type_metrics: Bytes::from(
                    candidate.tweet_type_metrics.clone().unwrap_or_default(),
                ),
                following_replied_user_ids: candidate.following_replied_user_ids.clone(),
                brand_safety_verdict: candidate
                    .brand_safety_verdict
                    .unwrap_or(BrandSafetyVerdict::MediumRisk)
                    as i32,
                safety_label_types: candidate
                    .safety_labels
                    .iter()
                    .filter_map(|l| safety_label_to_proto(l.label_type))
                    .collect(),
                tweet_text: candidate.tweet_text.clone(),
                predicted_dwell_sec: candidate
                    .phoenix_scores
                    .dwell_time
                    .filter(|d| d.is_finite() && *d > 0.0)
                    .map(|d| d as f32),
                topic_feedback_topic: candidate.topic_feedback_topic.clone(),
                topic_feedback_topic_id: candidate.topic_feedback_topic_id.clone(),
                ai_trend_name: candidate.ai_trend_name.clone(),
                ai_trend_id: candidate.ai_trend_id.clone(),
                under_the_hood: query
                    .is_under_the_hood_request
                    .then(|| under_the_hood::post_scores(query, candidate)),
                nsfw_author_ads: candidate.nsfw_author_ads.unwrap_or(false),
            }
        })
        .collect()
}

pub(crate) fn build_debug_json(
    pipeline_result: &PipelineResult<ScoredPostsQuery, PostCandidate>,
) -> String {
    let debug = serde_json::json!({
        "query": pipeline_result.query.as_ref(),
        "retrieved_candidates": &pipeline_result.retrieved_candidates,
        "filtered_candidates": &pipeline_result.filtered_candidates,
        "selected_candidates": &pipeline_result.selected_candidates,
        "stats": {
            "retrieved_count": pipeline_result.retrieved_candidates.len(),
            "filtered_count": pipeline_result.filtered_candidates.len(),
            "selected_count": pipeline_result.selected_candidates.len(),
        },
    });

    serde_json::to_string(&debug)
        .unwrap_or_else(|e| format!("{{\"error\": \"Failed to serialize debug info: {}\"}}", e))
}


fn log_request_info(query: &ScoredPostsQuery) {
    let seen_ids_count = query.seen_ids.len();
    let surface = if query.in_network_only {
        SURFACE_RANKED_FOLLOWING
    } else if !query.topic_ids.is_empty() {
        SURFACE_TOPICS
    } else if query.has_excluded_topics() {
        SURFACE_FOR_YOU_WITH_SNOOZED_TOPICS
    } else {
        SURFACE_FOR_YOU
    };

    if surface == SURFACE_TOPICS {
        info!(
            "Scored Posts request - {} - request_id {} - seen_ids {} - topic_ids {:?}",
            surface, query.request_id, seen_ids_count, query.topic_ids
        );
    } else if query.has_excluded_topics() {
        info!(
            "Scored Posts request - {} - request_id {} - seen_ids {} - excluded_topic_ids {:?}",
            surface, query.request_id, seen_ids_count, query.excluded_topic_ids
        );
    } else {
        info!(
            "Scored Posts request - {} - request_id {} - seen_ids {}",
            surface, query.request_id, seen_ids_count
        );
    }

    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(PRODUCT_SURFACE_METRIC, &[("surface", surface)], 1);
    }
}

fn log_response_stats(pipeline_result: &PipelineResult<ScoredPostsQuery, PostCandidate>) {
    if pipeline_result.query.topic_ids.len() == 1
        && let Some(receiver) = global_stats_receiver()
    {
        let tid_str = pipeline_result.query.topic_ids[0].to_string();
        receiver.incr(
            TOPIC_METRIC,
            &[("type", "request"), ("topic_id", &tid_str)],
            1,
        );
        if pipeline_result.selected_candidates.is_empty() {
            receiver.incr(
                TOPIC_METRIC,
                &[("type", "empty"), ("topic_id", &tid_str)],
                1,
            );
        }
    }
}

fn safety_label_to_proto(label: SafetyLabelType) -> Option<i32> {
    use xai_home_mixer_proto::SafetyLabelType as HM;
    let v = match label {
        SafetyLabelType::NSFW_HIGH_PRECISION => HM::NsfwHighPrecision,
        SafetyLabelType::NSFW_HIGH_RECALL => HM::NsfwHighRecall,
        SafetyLabelType::NSFA_HIGH_PRECISION => HM::NsfaHighPrecision,
        SafetyLabelType::NSFA_KEYWORDS_HIGH_PRECISION => HM::NsfaKeywordsHighPrecision,
        SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION => HM::GoreAndViolenceHighPrecision,
        SafetyLabelType::NSFW_REPORTED_HEURISTICS => HM::NsfwReportedHeuristics,
        SafetyLabelType::GORE_AND_VIOLENCE_REPORTED_HEURISTICS => {
            HM::GoreAndViolenceReportedHeuristics
        }
        SafetyLabelType::NSFW_CARD_IMAGE => HM::NsfwCardImage,
        SafetyLabelType::DO_NOT_AMPLIFY => HM::DoNotAmplify,
        SafetyLabelType::NSFA_COMMUNITY_NOTE => HM::NsfaCommunityNote,
        SafetyLabelType::PDNA => HM::Pdna,
        SafetyLabelType::EGREGIOUS_NSFW => HM::EgregiousNsfw,
        SafetyLabelType::GROK_NSFA => HM::GrokNsfa,
        SafetyLabelType::NSFW_TEXT => HM::NsfwText,
        SafetyLabelType::NSFA_LIMITED_INVENTORY => HM::NsfaLimitedInventory,
        SafetyLabelType::GROK_NSFA_LIMITED => HM::GrokNsfaLimited,
        SafetyLabelType::NSFA_HIGH_RECALL => HM::NsfaHighRecall,
        SafetyLabelType::GROK_SFA => HM::GrokSfa,
        SafetyLabelType::MALICIOUS_URL => HM::MaliciousUrl,
        SafetyLabelType::GROK_SFA_V2 => HM::GrokSfaV2,
        SafetyLabelType::GROK_NSFA_LIMITED_V2 => HM::GrokNsfaLimitedV2,
        SafetyLabelType::GROK_NSFA_V2 => HM::GrokNsfaV2,
        SafetyLabelType::GROK_NSFA_EXPANDED_V2 => HM::GrokNsfaExpandedV2,
        SafetyLabelType::PTOS_REVIEWED => HM::PtosReviewed,
        _ => return None,
    };
    Some(v.into())
}
