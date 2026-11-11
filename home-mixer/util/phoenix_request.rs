use crate::models::candidate::{CandidateHelpers, PostCandidate, RetrievalSource};
use crate::models::query::ScoredPostsQuery;
use crate::params::{PhoenixExperimentOverrides, RerankerHeadTag};
use crate::util::post_engagers::set_post_engagers;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use tracing::warn;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::TOP_LOG_PROBS_NUM;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster;
use xai_geo_ip::zip_to_dma_code;
use xai_home_mixer_proto as pb;
use xai_recsys_proto::{
    country_code_string_to_enum, CandidateSet, ClientContext, DeviceFeature,
    PredictNextActionsRequest, ProductSurface, RetrieveTopKCandidatesResponse, TweetInfo,
    UserContext,
};
use xai_stats_receiver::global_stats_receiver;

const PHOENIX_CLIENT_MAX_CANDIDATES: usize = 2800;
const RETRIEVAL_VALUE_MODEL_WEIGHTS_METRIC: &str = "PhoenixRetrieval.ValueModelWeights";
const MAX_CACHED_VALUE_MODEL_WEIGHTS: usize = 64;

type ParsedValueModelWeights = Option<Arc<HashMap<String, f32>>>;

static VALUE_MODEL_WEIGHTS: LazyLock<RwLock<HashMap<String, ParsedValueModelWeights>>> =
    LazyLock::new(Default::default);

pub fn candidates_from_retrieval_response(
    response: RetrieveTopKCandidatesResponse,
    served_type_for_dataset: impl Fn(u32) -> pb::ServedType,
    cluster: PhoenixRetrievalCluster,
) -> Vec<PostCandidate> {
    let scored: Vec<(TweetInfo, f32, pb::ServedType)> = response
        .top_k_candidates
        .into_iter()
        .flat_map(|scored_candidates| scored_candidates.candidates)
        .filter_map(|scored_candidate| {
            let served_type = served_type_for_dataset(scored_candidate.dataset_type);
            scored_candidate
                .candidate
                .map(|tweet_info| (tweet_info, scored_candidate.score, served_type))
        })
        .collect();
    let positions = positions_within_served_type(&scored);

    scored
        .into_iter()
        .zip(positions)
        .map(
            |((tweet_info, score, served_type), position)| PostCandidate {
                tweet_id: tweet_info.tweet_id,
                author_id: tweet_info.author_id,
                in_reply_to_tweet_id: (tweet_info.in_reply_to_tweet_id != 0)
                    .then_some(tweet_info.in_reply_to_tweet_id),
                retweeted_tweet_id: (tweet_info.retweeted_tweet_id != 0)
                    .then_some(tweet_info.retweeted_tweet_id),
                retrieval_sources: vec![RetrievalSource {
                    served_type,
                    cluster: Some(cluster),
                    score: Some(score),
                    position: Some(position),
                }],
                served_type: Some(served_type),
                ..Default::default()
            },
        )
        .collect()
}

fn positions_within_served_type(scored: &[(TweetInfo, f32, pb::ServedType)]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..scored.len()).collect();
    order.sort_by(|&a, &b| scored[b].1.total_cmp(&scored[a].1));
    let mut next = FxHashMap::default();
    let mut positions = vec![0; scored.len()];
    for idx in order {
        let position = next.entry(scored[idx].2).or_insert(0);
        *position += 1;
        positions[idx] = *position;
    }
    positions
}

pub fn build_client_context(query: &ScoredPostsQuery) -> Option<ClientContext> {
    if query.user_id == 0 {
        return None;
    }

    Some(ClientContext {
        user_id: query.user_id as i64,
        app_id: query.client_app_id as i64,
        country_code: query.country_code.clone(),
        language_code: query.language_code.clone(),
        user_roles: query.user_roles.clone(),
        ip_address: query.ip_address.clone(),
        client_version: query.client_version.clone(),
        ..Default::default()
    })
}

pub fn build_user_context(query: &ScoredPostsQuery) -> Option<UserContext> {
    Some(UserContext {
        user_age_bracket: query
            .user_demographics
            .as_ref()
            .map(|d| d.user_age_bracket)
            .unwrap_or(0),
        user_gender: query
            .user_demographics
            .as_ref()
            .map(|d| d.user_gender)
            .unwrap_or(0),
        user_state: query
            .user_demographics
            .as_ref()
            .map(|d| d.user_state)
            .unwrap_or(0),
        user_age_in_years: query.user_age_in_years.unwrap_or(0),
        user_inferred_gender: query.user_inferred_gender.map(|v| v as i32).unwrap_or(0),
        user_inferred_gender_score: query.user_inferred_gender_score.unwrap_or(0.0),
        followed_grok_topics: query
            .followed_grok_topics
            .map(|t| t.to_vec())
            .unwrap_or_default(),
        followed_starter_packs: query
            .followed_starter_packs
            .map(|p| p.to_vec())
            .unwrap_or_default(),
        user_longitude: query
            .ip_location
            .as_ref()
            .and_then(|l| l.longitude)
            .unwrap_or(0.0) as f32,
        user_latitude: query
            .ip_location
            .as_ref()
            .and_then(|l| l.latitude)
            .unwrap_or(0.0) as f32,
        user_dma_code: zip_to_dma_code(
            query
                .ip_location
                .as_ref()
                .and_then(|l| l.zip_code.as_deref()),
        ) as i32,
        user_installed_apps: query.user_installed_apps.clone().unwrap_or_default(),
        recent_search_queries: query.recent_search_queries.clone(),
        ..Default::default()
    })
}

pub fn build_device_feature(query: &ScoredPostsQuery) -> DeviceFeature {
    DeviceFeature {
        engaging_ip_country_code: country_code_string_to_enum(query.country_code.as_ref()) as i32,
        device_network_type: query.device_network_type as i32,
        timezone: query.time_zone as i32,
        ip_address: query.ip_address.clone(),
    }
}

fn followed_user_ids(query: &ScoredPostsQuery) -> FxHashSet<u64> {
    query
        .user_features
        .followed_user_ids
        .iter()
        .copied()
        .map(|id| id as u64)
        .collect()
}

fn candidate_to_tweet_info(
    c: &PostCandidate,
    query: &ScoredPostsQuery,
    followed_ids: &FxHashSet<u64>,
    serve_time_ms: i64,
) -> xai_recsys_proto::TweetInfo {
    let is_followed = followed_ids.contains(&c.get_original_author_id())
        || c.get_original_author_id() == query.user_id;
    let mut tweet_info = c.as_tweet_info(is_followed);
    if let Some(record) = &c.post_engagers {
        set_post_engagers(&mut tweet_info, record, query.user_id, serve_time_ms);
    }
    tweet_info
}

pub fn build_tweet_infos(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
    serve_time_ms: i64,
) -> Vec<xai_recsys_proto::TweetInfo> {
    let followed_ids = followed_user_ids(query);
    candidates
        .iter()
        .take(PHOENIX_CLIENT_MAX_CANDIDATES)
        .map(|c| candidate_to_tweet_info(c, query, &followed_ids, serve_time_ms))
        .collect()
}

pub fn build_request_without_sequence_and_candidates(
    query: &ScoredPostsQuery,
    product_surface: ProductSurface,
) -> PredictNextActionsRequest {
    let candidate_set = CandidateSet {
        user_id: query.user_id,
        product_surface: product_surface as i32,
        device_feature: Some(build_device_feature(query)),
        ..Default::default()
    };
    PredictNextActionsRequest {
        candidate_sets: vec![candidate_set],
        return_logprob: true,
        top_logprobs_num: TOP_LOG_PROBS_NUM,
        return_backbone_scores: query.return_backbone_scores,
        client_context: build_client_context(query),
        user_context: build_user_context(query),
        metadata: query.request_id.to_string(),
        experiment_overrides: parse_experiment_overrides(
            &query.params.get(PhoenixExperimentOverrides),
        ),
        ..Default::default()
    }
}

pub fn parse_experiment_overrides(spec: &str) -> std::collections::HashMap<String, String> {
    spec.split(';')
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            let (k, v) = (k.trim(), v.trim());
            (!k.is_empty()).then(|| (k.to_string(), v.to_string()))
        })
        .collect()
}

pub fn retrieval_value_model_weights(spec: &str) -> HashMap<String, f32> {
    if spec.is_empty() {
        return HashMap::new();
    }
    let parsed = cached_value_model_weights(spec);
    if let Some(receiver) = global_stats_receiver() {
        let result = if parsed.is_some() { "sent" } else { "invalid" };
        receiver.incr(
            RETRIEVAL_VALUE_MODEL_WEIGHTS_METRIC,
            &[("result", result)],
            1,
        );
    }
    parsed
        .map(|weights| weights.as_ref().clone())
        .unwrap_or_default()
}

fn cached_value_model_weights(spec: &str) -> ParsedValueModelWeights {
    if let Some(parsed) = VALUE_MODEL_WEIGHTS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(spec)
    {
        return parsed.clone();
    }
    let parsed = parse_value_model_weights(spec)
        .inspect_err(|e| {
            warn!("rust_home_mixer_retrieval_value_model_weights {spec:?} is not sent: {e}")
        })
        .ok()
        .map(Arc::new);
    let mut cache = VALUE_MODEL_WEIGHTS
        .write()
        .unwrap_or_else(|e| e.into_inner());
    if cache.len() >= MAX_CACHED_VALUE_MODEL_WEIGHTS {
        cache.clear();
    }
    cache.insert(spec.to_string(), parsed.clone());
    parsed
}

pub fn parse_value_model_weights(spec: &str) -> Result<HashMap<String, f32>, String> {
    let weights: HashMap<String, f64> = serde_json::from_str(spec).map_err(|e| e.to_string())?;
    weights
        .into_iter()
        .map(|(name, weight)| {
            let weight = weight as f32;
            if weight.is_finite() {
                Ok((name, weight))
            } else {
                Err(format!("weight of {name:?} is not a finite f32"))
            }
        })
        .collect()
}

pub fn build_prediction_request(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
    product_surface: ProductSurface,
    serve_time_ms: i64,
) -> PredictNextActionsRequest {
    let mut request = build_request_without_sequence_and_candidates(query, product_surface);
    request.return_candidate_tweet_id_only = true;
    request.candidate_sets[0].candidates = build_tweet_infos(query, candidates, serve_time_ms);
    let mut sequence = query.scoring_sequence.clone().unwrap_or_default();
    sequence.reranker_head_tag = Some(query.params.get(RerankerHeadTag) as u32);
    request.sequences = vec![sequence];
    request.columnar_sequences = query.columnar_scoring_sequence.iter().cloned().collect();
    request
}

pub fn build_immersive_prediction_request(
    query: &ScoredPostsQuery,
    candidates: &[&PostCandidate],
) -> PredictNextActionsRequest {
    let mut request =
        build_request_without_sequence_and_candidates(query, ProductSurface::GalleryPage);
    request.return_backbone_scores = false;
    request.return_candidate_tweet_id_only = true;
    let followed_ids = followed_user_ids(query);
    request.candidate_sets[0].candidates = candidates
        .iter()
        .map(|candidate| {
            let tweet_info =
                candidate_to_tweet_info(candidate, query, &followed_ids, query.request_time_ms);
            TweetInfo {
                first_video_duration_ms: tweet_info.min_video_duration_ms,
                ..tweet_info
            }
        })
        .collect();
    request.columnar_sequences = query
        .immersive_columnar_scoring_sequence
        .iter()
        .cloned()
        .collect();
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_id: 42,
            ..Default::default()
        }
    }

    fn candidate(tweet_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id: 7,
            ..Default::default()
        }
    }

    #[test]
    fn prediction_request_asks_for_tweet_id_only_candidates() {
        let request = build_prediction_request(
            &query(),
            &[candidate(1)],
            ProductSurface::HomeTimelineRanking,
            0,
        );
        assert!(request.return_candidate_tweet_id_only);
    }

    #[test]
    fn immersive_prediction_request_asks_for_tweet_id_only_candidates() {
        let candidate = candidate(1);
        let request = build_immersive_prediction_request(&query(), &[&candidate]);
        assert!(request.return_candidate_tweet_id_only);
    }
}
