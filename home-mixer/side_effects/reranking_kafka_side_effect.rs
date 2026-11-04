use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::scorers::value_model;
use prost::Message;
use rand::random;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::KafkaPublisherClient;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto as pb;

const TOP_K: usize = 50;

const WEIGHTED_VALUE_MODEL_MODE: &str = "weighted";

pub struct RerankingKafkaSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl RerankingKafkaSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for RerankingKafkaSideEffect {
    fn enable(&self, _query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && random::<f64>() < 0.05
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let mut candidates: Vec<&PostCandidate> = Vec::new();
        for c in &input.selected_candidates {
            candidates.push(c);
        }
        for c in &input.non_selected_candidates {
            candidates.push(c);
        }

        let total_count = candidates.len() as i32;

        candidates.sort_by(|a, b| {
            let sa = a.score.unwrap_or(f64::MIN);
            let sb = b.score.unwrap_or(f64::MIN);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(TOP_K);

        if candidates.is_empty() {
            return Ok(());
        }

        let scored_candidates: Vec<pb::ScoredCandidate> = candidates
            .iter()
            .enumerate()
            .map(|(position, candidate)| build_scored_candidate(candidate, position as i32))
            .collect();

        let request_time_ms = input.query.request_time_ms;

        let prediction_request_id = candidates
            .iter()
            .find_map(|c| c.prediction_request_id.map(|id| id as i64));

        let product_surface = if input.query.in_network_only {
            xai_recsys_proto::ProductSurface::HomeTimelineRankedFollowing
        } else {
            xai_recsys_proto::ProductSurface::HomeTimelineRanking
        };

        let batch = pb::ScoredCandidateBatch {
            candidates: scored_candidates,
            viewer_id: Some(input.query.user_id),
            request_time_ms: Some(request_time_ms),
            prediction_request_id,
            served_request_id: prediction_request_id,
            served_id: prediction_request_id,
            total_candidates_count: Some(total_count),
            request_join_id: Some(input.query.request_id),
            product_surface: product_surface.into(),
            applied_weights: value_model::applied_weights(&input.query),
            value_model_mode: Some(WEIGHTED_VALUE_MODEL_MODE.to_string()),
        };

        let bytes = batch.encode_to_vec();
        self.kafka_client
            .send(&bytes)
            .await
            .map_err(|e| format!("Kafka publish failed: {e}"))
    }
}

fn build_scored_candidate(candidate: &PostCandidate, position: i32) -> pb::ScoredCandidate {
    let s = &candidate.phoenix_scores;

    let mut prediction_scores: HashMap<String, f64> = HashMap::new();
    insert_score(&mut prediction_scores, "favorite", s.favorite_score);
    insert_score(&mut prediction_scores, "reply", s.reply_score);
    insert_score(&mut prediction_scores, "retweet", s.retweet_score);
    insert_score(&mut prediction_scores, "photo_expand", s.photo_expand_score);
    insert_score(&mut prediction_scores, "video_open", s.video_open_score);
    insert_score(&mut prediction_scores, "click", s.click_score);
    insert_score(&mut prediction_scores, "open_link", s.open_link_score);
    insert_score(
        &mut prediction_scores,
        "profile_click",
        s.profile_click_score,
    );
    insert_score(&mut prediction_scores, "vqv", s.vqv_score);
    insert_score(&mut prediction_scores, "share", s.share_score);
    insert_score(&mut prediction_scores, "share_via_dm", s.share_via_dm_score);
    insert_score(
        &mut prediction_scores,
        "share_via_copy_link",
        s.share_via_copy_link_score,
    );
    insert_score(&mut prediction_scores, "dwell", s.dwell_score);
    insert_score(&mut prediction_scores, "quote", s.quote_score);
    insert_score(&mut prediction_scores, "quoted_click", s.quoted_click_score);
    insert_score(&mut prediction_scores, "quoted_vqv", s.quoted_vqv_score);
    insert_score(
        &mut prediction_scores,
        "follow_author",
        s.follow_author_score,
    );
    insert_score(
        &mut prediction_scores,
        "not_interested",
        s.not_interested_score,
    );
    insert_score(&mut prediction_scores, "block_author", s.block_author_score);
    insert_score(&mut prediction_scores, "mute_author", s.mute_author_score);
    insert_score(&mut prediction_scores, "report", s.report_score);
    insert_score(&mut prediction_scores, "not_dwelled", s.not_dwelled_score);
    insert_score(
        &mut prediction_scores,
        "post_unexplored",
        s.post_unexplored_score,
    );
    insert_score(&mut prediction_scores, "pdwell", s.post_unexplored_score);
    insert_score(&mut prediction_scores, "post_video", s.post_video_score);
    insert_score(&mut prediction_scores, "grok_analyze", s.grok_analyze_score);
    insert_score(&mut prediction_scores, "dwell_time", s.dwell_time);
    insert_score(
        &mut prediction_scores,
        "click_dwell_time",
        s.click_dwell_time,
    );

    let source_tweet_id = candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id);

    let served_type = candidate.served_type.map(|st| format!("{:?}", st));

    pb::ScoredCandidate {
        tweet_id: candidate.tweet_id,
        score: candidate.score,
        prediction_scores,
        weighted_model_score: candidate.weighted_score,
        author_id: Some(candidate.author_id),
        source_tweet_id: Some(source_tweet_id),
        served_type,
        is_cached: candidate.last_scored_at_ms.is_some(),
        in_network: candidate.in_network.unwrap_or(false),
        position,
        is_mutual_follow_author: candidate.is_mutual_follow_author,
        broadcast_is_live: candidate.broadcast_is_live,
        video_duration_ms: candidate.min_video_duration_ms,
    }
}

fn insert_score(map: &mut HashMap<String, f64>, name: &str, value: Option<f64>) {
    map.insert(name.to_string(), value.unwrap_or(0.0));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::PhoenixScores;
    use crate::params::{
        BidirectionalFollowReplyWeightBoost, ContDwellTimeWeight, FavoriteWeight, NotDwelledWeight,
        ReplyWeight, ReportWeight,
    };

    fn default_weights_map() -> HashMap<String, f64> {
        value_model::applied_weights(&ScoredPostsQuery::default())
    }

    #[test]
    fn applied_weights_match_param_defaults() {
        let params = xai_feature_switches::Params::default();
        let weights = default_weights_map();
        assert_eq!(weights["favorite"], params.get(FavoriteWeight));
        assert_eq!(weights["reply"], params.get(ReplyWeight));
        assert_eq!(weights["dwell_time"], params.get(ContDwellTimeWeight));
        assert_eq!(weights["not_dwelled"], params.get(NotDwelledWeight));
        assert_eq!(weights["report"], params.get(ReportWeight));
        assert_eq!(
            weights["boost.bidirectional_follow_reply"],
            params.get(BidirectionalFollowReplyWeightBoost)
        );
    }

    #[test]
    fn every_logged_head_has_an_applied_weight() {
        let candidate = PostCandidate::default();
        let scored = build_scored_candidate(&candidate, 0);
        let weights = default_weights_map();

        for head in scored.prediction_scores.keys() {
            assert!(
                weights.contains_key(head),
                "logged head {head} has no applied weight entry"
            );
        }
        for key in weights.keys() {
            if key.starts_with("boost.") || key.starts_with("gate.") {
                continue;
            }
            assert!(
                scored.prediction_scores.contains_key(key),
                "weight {key} has no logged prediction head"
            );
        }
    }

    #[test]
    fn logs_previously_missing_heads() {
        let candidate = PostCandidate {
            phoenix_scores: PhoenixScores {
                open_link_score: Some(0.3),
                video_open_score: Some(0.2),
                not_dwelled_score: Some(0.4),
                post_unexplored_score: Some(0.5),
                click_dwell_time: Some(1.5),
                ..Default::default()
            },
            ..Default::default()
        };
        let scored = build_scored_candidate(&candidate, 0);
        assert_eq!(scored.prediction_scores["open_link"], 0.3);
        assert_eq!(scored.prediction_scores["video_open"], 0.2);
        assert_eq!(scored.prediction_scores["not_dwelled"], 0.4);
        assert_eq!(scored.prediction_scores["post_unexplored"], 0.5);
        assert_eq!(scored.prediction_scores["click_dwell_time"], 1.5);
    }

    #[test]
    fn passes_through_weight_context_fields() {
        let candidate = PostCandidate {
            is_mutual_follow_author: Some(true),
            broadcast_is_live: Some(false),
            min_video_duration_ms: Some(12_000),
            ..Default::default()
        };
        let scored = build_scored_candidate(&candidate, 3);
        assert_eq!(scored.is_mutual_follow_author, Some(true));
        assert_eq!(scored.broadcast_is_live, Some(false));
        assert_eq!(scored.video_duration_ms, Some(12_000));
        assert_eq!(scored.position, 3);
    }
}
