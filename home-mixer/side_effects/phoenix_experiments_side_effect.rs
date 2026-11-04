use crate::clients::user_action_aggregation_client::UserActionAggregationClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{MaxPostsToCache, PhoenixAggregationType};
use crate::query_hydrators::scoring_sequence_query_hydrator::fetch_scoring_sequence;
use crate::side_effects::redis_post_candidate_cache_side_effect::{
    RedisPostCandidateCacheSideEffect, REDIS_TTL_SECONDS,
};
use crate::util::phoenix_request::build_prediction_request;
use crate::util::shadow::{
    is_shadow_sampled_for_cluster, is_shadow_traffic, shadow_aggregation_type_for_cluster,
};
use crate::util::shadow_scores::{
    decode_shadow_scores, encode_shadow_scores, shadow_scores_key, ShadowScoresByPost,
};
use futures::future::join_all;
use prost::Message;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use strum::VariantArray;
use thrift::OrderedFloat;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::KafkaPublisherClient;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PhoenixPredictionClient,
};
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::component_library::models::PhoenixScores;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_proto::recsys_logging::LoggedScoredCandidate;
use xai_recsys_logging_thrift::{serialize_to_bytes_binary, PredictionScore, ScoredCandidate};
use xai_recsys_proto::{
    language_code_string_to_enum, PredictNextActionsRequest, ProductSurface,
    UserActionAggregationType,
};
use xai_stats_receiver::global_stats_receiver;

const STORED_SCORES_METRIC: &str = "PhoenixExperimentsSideEffect.stored_scores";
const STORED_SCORES_RETRY_WAITS_MS: [u64; 3] = [500, 1_000, 2_000];
const SEQUENCE_OVERRIDE_METRIC: &str = "PhoenixExperimentsSideEffect.sequence_override";

pub struct PhoenixExperimentsSideEffect {
    phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
    kafka_client: Arc<dyn KafkaPublisherClient>,
    logged_scored_candidates_kafka_client: Arc<dyn KafkaPublisherClient>,
    shadow_scores_redis_client: Arc<dyn RedisClient>,
    user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
}

impl PhoenixExperimentsSideEffect {
    pub fn new(
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        kafka_client: Arc<dyn KafkaPublisherClient>,
        logged_scored_candidates_kafka_client: Arc<dyn KafkaPublisherClient>,
        shadow_scores_redis_client: Arc<dyn RedisClient>,
        user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
    ) -> Self {
        Self {
            phoenix_client,
            kafka_client,
            logged_scored_candidates_kafka_client,
            shadow_scores_redis_client,
            user_action_aggregation_client,
        }
    }

    async fn override_requests(
        &self,
        query: &ScoredPostsQuery,
        clusters: &[PhoenixCluster],
        candidates: &[PostCandidate],
        product_surface: ProductSurface,
        serve_time_ms: i64,
    ) -> HashMap<UserActionAggregationType, Option<PredictNextActionsRequest>> {
        let serving_type =
            UserActionAggregationType::from_str_name(&query.params.get(PhoenixAggregationType));
        let override_types: HashSet<UserActionAggregationType> = clusters
            .iter()
            .filter_map(|&c| shadow_aggregation_type_for_cluster(&query.params, c))
            .filter(|t| Some(*t) != serving_type)
            .collect();
        let fetches = override_types
            .into_iter()
            .map(|aggregation_type| async move {
                let fetched = fetch_scoring_sequence(
                    self.user_action_aggregation_client.as_ref(),
                    query,
                    aggregation_type,
                    None,
                )
                .await;
                let request = match fetched {
                    Ok((scoring_sequence, columnar_scoring_sequence)) => {
                        record_sequence_override("fetched");
                        let mut override_query = query.clone();
                        override_query.scoring_sequence = scoring_sequence;
                        override_query.columnar_scoring_sequence = columnar_scoring_sequence;
                        Some(build_prediction_request(
                            &override_query,
                            candidates,
                            product_surface,
                            serve_time_ms,
                        ))
                    }
                    Err(err) => {
                        record_sequence_override("fetch_error");
                        tracing::error!(
                            "Shadow sequence fetch for {:?} failed: {}",
                            aggregation_type,
                            err
                        );
                        None
                    }
                };
                (aggregation_type, request)
            });
        join_all(fetches).await.into_iter().collect()
    }

    async fn score_on_experiment_clusters(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> ShadowScoresByPost {
        let product_surface = if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };
        let serve_time_ms = candidates
            .iter()
            .find_map(|c| c.last_scored_at_ms)
            .map_or(query.request_time_ms, |ms| ms as i64);
        let base_request =
            build_prediction_request(query, candidates, product_surface, serve_time_ms);
        let sampled_clusters: Vec<PhoenixCluster> = PhoenixCluster::VARIANTS
            .iter()
            .copied()
            .filter(|&c| is_shadow_sampled_for_cluster(&query.params, query.prediction_id, c))
            .collect();
        let override_requests = self
            .override_requests(
                query,
                &sampled_clusters,
                candidates,
                product_surface,
                serve_time_ms,
            )
            .await;

        let futures = sampled_clusters
            .iter()
            .filter_map(|&cluster_id| {
                let request = match shadow_aggregation_type_for_cluster(&query.params, cluster_id)
                    .and_then(|t| override_requests.get(&t))
                {
                    Some(Some(request)) => request.clone(),
                    Some(None) => return None,
                    None => base_request.clone(),
                };
                Some((cluster_id, request))
            })
            .map(|(cluster_id, request)| {
                let client = Arc::clone(&self.phoenix_client);
                async move {
                    let result: Result<_, tonic::Status> =
                        client.predict(cluster_id, request, 0).await;
                    if let Err(ref err) = result {
                        tracing::error!(
                            "Phoenix experiment {:?} request failed: {}",
                            cluster_id,
                            err
                        );
                    }
                    (cluster_id, result)
                }
            });
        let experiment_results: Vec<_> = join_all(futures).await;

        candidates
            .iter()
            .map(|candidate| {
                let mut prediction_scores = HashMap::new();
                for (cluster_id, result) in &experiment_results {
                    let Ok(predictions) = result else {
                        continue;
                    };
                    let cluster_name = format!("{:?}", cluster_id);
                    let scores = predictions.candidate_scores(&candidate.get_original_tweet_id());
                    insert_cluster_scores(&mut prediction_scores, &cluster_name, &scores);
                }
                (candidate.tweet_id, prediction_scores)
            })
            .collect()
    }

    async fn store_scores(&self, prediction_request_id: u64, scores: ShadowScoresByPost) {
        if scores.is_empty() {
            return;
        }
        let encoded = match tokio::task::spawn_blocking(move || encode_shadow_scores(&scores))
            .await
            .map_err(|err| err.to_string())
            .and_then(|encoded| encoded)
        {
            Ok(encoded) => encoded,
            Err(err) => {
                record_stored_scores("encode_error");
                tracing::error!("Failed to encode shadow scores: {err}");
                return;
            }
        };
        match self
            .shadow_scores_redis_client
            .set_ex(
                shadow_scores_key(prediction_request_id),
                encoded,
                REDIS_TTL_SECONDS,
            )
            .await
        {
            Ok(()) => record_stored_scores("written"),
            Err(err) => {
                record_stored_scores("write_error");
                tracing::error!("Failed to store shadow scores: {err}");
            }
        }
    }

    async fn load_scores(&self, prediction_request_id: u64) -> Option<ShadowScoresByPost> {
        let key = shadow_scores_key(prediction_request_id);
        let mut waits_ms = STORED_SCORES_RETRY_WAITS_MS.iter();
        let mut waited = false;
        let bytes = loop {
            match self.shadow_scores_redis_client.get_opt(key.clone()).await {
                Ok(Some(bytes)) => break bytes,
                Ok(None) => match waits_ms.next() {
                    Some(&wait_ms) => {
                        waited = true;
                        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    }
                    None => {
                        record_stored_scores("missing");
                        return None;
                    }
                },
                Err(err) => {
                    record_stored_scores("read_error");
                    tracing::error!("Failed to read shadow scores: {err}");
                    return None;
                }
            }
        };
        match tokio::task::spawn_blocking(move || decode_shadow_scores(&bytes))
            .await
            .map_err(|err| err.to_string())
            .and_then(|decoded| decoded)
        {
            Ok(scores) => {
                record_stored_scores(if waited { "hit_after_wait" } else { "hit" });
                Some(scores)
            }
            Err(err) => {
                record_stored_scores("decode_error");
                tracing::error!("Failed to decode shadow scores: {err}");
                None
            }
        }
    }

    async fn publish_scored_candidate(
        &self,
        query: &ScoredPostsQuery,
        candidate: &PostCandidate,
        prediction_scores: HashMap<String, f64>,
    ) {
        let user_id = query.user_id;
        let request_time_ms = query.request_time_ms;
        let thrift_scores: BTreeSet<Box<PredictionScore>> = prediction_scores
            .iter()
            .map(|(name, value)| {
                Box::new(PredictionScore::new(
                    Some(name.clone()),
                    Some(OrderedFloat::from(*value)),
                ))
            })
            .collect();

        let source_tweet_id = candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id);
        let scored = ScoredCandidate {
            tweet_id: candidate.tweet_id as i64,
            viewer_id: Some(user_id as i64),
            author_id: Some(candidate.author_id as i64),
            request_join_id: Some(query.request_id as i64),
            score: None,
            suggest_type: None,
            is_in_network: candidate.in_network,
            in_reply_to_tweet_id: candidate.in_reply_to_tweet_id.map(|id| id as i64),
            quoted_tweet_id: candidate.quoted_tweet_id.map(|id| id as i64),
            quoted_user_id: candidate.quoted_user_id.map(|id| id as i64),
            request_time_ms: Some(request_time_ms),
            source_tweet_id: Some(source_tweet_id as i64),
            prediction_scores: Some(thrift_scores),
            fav_count: candidate.fav_count,
            reply_count: candidate.reply_count,
            retweet_count: candidate.repost_count,
            quote_count: candidate.quote_count,
            has_media: candidate.has_media,
            language_code: candidate
                .language_code
                .as_deref()
                .map(|lc| language_code_string_to_enum(lc) as i32),
            video_duration_ms: candidate.min_video_duration_ms,
            raw_query: None,
        };

        let thrift_bytes = serialize_to_bytes_binary(&scored);
        let proto_bytes = build_logged_scored_candidate(
            candidate,
            user_id,
            query.request_id,
            request_time_ms,
            prediction_scores,
        )
        .encode_to_vec();
        let (thrift_result, proto_result) = tokio::join!(
            async {
                match thrift_bytes {
                    Ok(bytes) => self.kafka_client.send(&bytes).await,
                    Err(e) => {
                        tracing::error!("Failed to serialize scored candidate: {e}");
                        Ok(())
                    }
                }
            },
            self.logged_scored_candidates_kafka_client
                .send(&proto_bytes)
        );
        if let Err(e) = thrift_result {
            tracing::error!("Failed to publish scored candidate to Kafka: {e}");
        }
        if let Err(e) = proto_result {
            tracing::error!("Failed to publish logged scored candidate to Kafka: {e}");
        }
    }

    async fn shadow_score_fresh_request(
        &self,
        input: &SideEffectInput<ScoredPostsQuery, PostCandidate>,
    ) {
        let query = &input.query;
        let candidates = shadow_candidates(
            &input.selected_candidates,
            &input.non_selected_candidates,
            query.params.get(MaxPostsToCache),
        );
        let mut scores = self.score_on_experiment_clusters(query, &candidates).await;
        let selected_scores: Vec<HashMap<String, f64>> = input
            .selected_candidates
            .iter()
            .map(|c| scores.get(&c.tweet_id).cloned().unwrap_or_default())
            .collect();
        scores.retain(|_, post_scores| !post_scores.is_empty());
        let publish = async {
            for (candidate, prediction_scores) in
                input.selected_candidates.iter().zip(selected_scores)
            {
                self.publish_scored_candidate(query, candidate, prediction_scores)
                    .await;
            }
        };
        tokio::join!(self.store_scores(query.prediction_id, scores), publish);
    }

    async fn publish_stored_scores(
        &self,
        input: &SideEffectInput<ScoredPostsQuery, PostCandidate>,
    ) {
        let query = &input.query;
        let prediction_request_ids: BTreeSet<u64> = input
            .selected_candidates
            .iter()
            .filter_map(|c| c.prediction_request_id)
            .collect();
        let mut stored = HashMap::new();
        for id in prediction_request_ids {
            if !is_shadow_traffic(&query.params, id) {
                continue;
            }
            if let Some(scores) = self.load_scores(id).await {
                stored.insert(id, scores);
            }
        }
        for candidate in &input.selected_candidates {
            let prediction_scores = candidate
                .prediction_request_id
                .and_then(|id| stored.get(&id))
                .and_then(|scores| scores.get(&candidate.tweet_id));
            if let Some(prediction_scores) = prediction_scores {
                self.publish_scored_candidate(query, candidate, prediction_scores.clone())
                    .await;
            }
        }
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for PhoenixExperimentsSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        query.is_shadow_traffic || query.has_cached_posts
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        if input.query.has_cached_posts {
            self.publish_stored_scores(&input).await;
        } else if input.query.is_shadow_traffic && input.query.scoring_sequence.is_some() {
            self.shadow_score_fresh_request(&input).await;
        }
        Ok(())
    }
}

fn shadow_candidates(
    selected: &[PostCandidate],
    non_selected: &[PostCandidate],
    max_posts_to_cache: usize,
) -> Vec<PostCandidate> {
    let mut seen = HashSet::new();
    selected
        .iter()
        .chain(RedisPostCandidateCacheSideEffect::get_candidates_to_cache(
            selected,
            non_selected,
            max_posts_to_cache,
        ))
        .filter(|c| seen.insert(c.tweet_id))
        .cloned()
        .collect()
}

fn record_sequence_override(outcome: &str) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(SEQUENCE_OVERRIDE_METRIC, &[("outcome", outcome)], 1);
    }
}

fn record_stored_scores(outcome: &str) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(STORED_SCORES_METRIC, &[("outcome", outcome)], 1);
    }
}

pub(crate) fn insert_cluster_scores(
    prediction_scores: &mut HashMap<String, f64>,
    cluster_name: &str,
    s: &PhoenixScores,
) {
    let score_fields: &[(&str, Option<f64>)] = &[
        ("favorite", s.favorite_score),
        ("reply", s.reply_score),
        ("retweet", s.retweet_score),
        ("photo_expand", s.photo_expand_score),
        ("video_open", s.video_open_score),
        ("click", s.click_score),
        ("open_link", s.open_link_score),
        ("profile_click", s.profile_click_score),
        ("vqv", s.vqv_score),
        ("share", s.share_score),
        ("share_via_dm", s.share_via_dm_score),
        ("share_via_copy_link", s.share_via_copy_link_score),
        ("bookmark", s.bookmark_score),
        ("dwell", s.dwell_score),
        ("quote", s.quote_score),
        ("quoted_click", s.quoted_click_score),
        ("quoted_vqv", s.quoted_vqv_score),
        ("follow_author", s.follow_author_score),
        ("not_interested", s.not_interested_score),
        ("block_author", s.block_author_score),
        ("mute_author", s.mute_author_score),
        ("report", s.report_score),
        ("not_dwelled", s.not_dwelled_score),
        ("post_unexplored", s.post_unexplored_score),
        ("pdwell", s.post_unexplored_score),
        ("post_video", s.post_video_score),
        ("grok_analyze", s.grok_analyze_score),
        ("dwell_time", s.dwell_time),
        ("click_dwell_time", s.click_dwell_time),
    ];
    for (name, score) in score_fields {
        if let Some(value) = score {
            prediction_scores.insert(format!("phoenix.{cluster_name}.{name}"), *value);
        }
    }
}

pub(crate) fn build_logged_scored_candidate(
    candidate: &PostCandidate,
    user_id: u64,
    request_id: u64,
    request_time_ms: i64,
    prediction_scores: HashMap<String, f64>,
) -> LoggedScoredCandidate {
    let source_post_id = candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id);
    LoggedScoredCandidate {
        post_id: candidate.tweet_id as i64,
        viewer_id: Some(user_id as i64),
        author_id: Some(candidate.author_id as i64),
        request_join_id: Some(request_id as i64),
        request_time_ms: Some(request_time_ms),
        source_post_id: Some(source_post_id as i64),
        is_in_network: candidate.in_network,
        in_reply_to_post_id: candidate.in_reply_to_tweet_id.map(|id| id as i64),
        quoted_post_id: candidate.quoted_tweet_id.map(|id| id as i64),
        quoted_user_id: candidate.quoted_user_id.map(|id| id as i64),
        prediction_scores,
        fav_count: candidate.fav_count,
        reply_count: candidate.reply_count,
        retweet_count: candidate.repost_count,
        quote_count: candidate.quote_count,
        has_media: candidate.has_media,
        language_code: candidate
            .language_code
            .as_deref()
            .map(|lc| language_code_string_to_enum(lc) as i32),
        video_duration_ms: candidate.min_video_duration_ms,
        carousel_position: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_present_heads_with_cluster() {
        let mut prediction_scores = HashMap::new();
        insert_cluster_scores(
            &mut prediction_scores,
            "Experiment1",
            &PhoenixScores {
                favorite_score: Some(0.12),
                reply_score: None,
                post_unexplored_score: Some(0.4),
                ..Default::default()
            },
        );
        assert_eq!(
            prediction_scores.get("phoenix.Experiment1.favorite"),
            Some(&0.12)
        );
        assert_eq!(
            prediction_scores.get("phoenix.Experiment1.pdwell"),
            Some(&0.4)
        );
        assert_eq!(
            prediction_scores.get("phoenix.Experiment1.post_unexplored"),
            Some(&0.4)
        );
        assert!(!prediction_scores.contains_key("phoenix.Experiment1.reply"));
    }

    #[test]
    fn copies_candidate_identity_and_features() {
        let candidate = PostCandidate {
            tweet_id: 11,
            author_id: 22,
            retweeted_tweet_id: Some(33),
            in_reply_to_tweet_id: Some(44),
            quoted_tweet_id: Some(55),
            quoted_user_id: Some(66),
            in_network: Some(true),
            fav_count: Some(1),
            reply_count: Some(2),
            repost_count: Some(3),
            quote_count: Some(4),
            has_media: Some(true),
            language_code: Some("en".to_string()),
            min_video_duration_ms: Some(1500),
            ..Default::default()
        };
        let logged = build_logged_scored_candidate(
            &candidate,
            99,
            123,
            1_700_000_000_000,
            HashMap::from([("phoenix.Prod.favorite".to_string(), 0.5)]),
        );
        assert_eq!(logged.post_id, 11);
        assert_eq!(logged.viewer_id, Some(99));
        assert_eq!(logged.author_id, Some(22));
        assert_eq!(logged.source_post_id, Some(33));
        assert_eq!(logged.in_reply_to_post_id, Some(44));
        assert_eq!(logged.quoted_post_id, Some(55));
        assert_eq!(logged.quoted_user_id, Some(66));
        assert_eq!(logged.request_join_id, Some(123));
        assert_eq!(logged.request_time_ms, Some(1_700_000_000_000));
        assert_eq!(logged.is_in_network, Some(true));
        assert_eq!(logged.fav_count, Some(1));
        assert_eq!(logged.reply_count, Some(2));
        assert_eq!(logged.retweet_count, Some(3));
        assert_eq!(logged.quote_count, Some(4));
        assert_eq!(logged.has_media, Some(true));
        assert_eq!(logged.video_duration_ms, Some(1500));
        assert_eq!(
            logged.prediction_scores.get("phoenix.Prod.favorite"),
            Some(&0.5)
        );
        assert!(logged.language_code.is_some());
    }
}
