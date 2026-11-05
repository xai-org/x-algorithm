use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixSource, PhoenixColdStartMaxResults, PhoenixMaxResults,
    PhoenixRetrievalExcludeSeenPosts, PhoenixRetrievalInferenceClusterId,
    PhoenixRetrievalNewUserHistoryThreshold, PhoenixRetrievalNewUserInferenceClusterId,
    PhoenixXdsRetrievalMaxRetries, RetrievalValueModelWeights,
};
use crate::util::phoenix_request::{
    build_client_context, build_user_context, candidates_from_retrieval_response,
    retrieval_value_model_weights,
};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalClient;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    PhoenixRetrievalCluster, RetrievalExclusions,
};
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_recsys_proto::RetrievalDatasetType;
use xai_stats_receiver::global_stats_receiver;

const ZERO_FAV_RETRIEVED_METRIC: &str = "ScoredStats.PhoenixRetrievalZeroFavTweets.Retrieved";

pub struct PhoenixSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

impl PhoenixSource {
    pub(crate) fn resolve_cluster(query: &ScoredPostsQuery) -> PhoenixRetrievalCluster {
        let configured_cluster =
            PhoenixRetrievalCluster::parse(&query.params.get(PhoenixRetrievalInferenceClusterId));

        let threshold: u64 = query.params.get(PhoenixRetrievalNewUserHistoryThreshold);
        if threshold > 0 {
            let action_count = query
                .retrieval_sequence
                .as_ref()
                .and_then(|s| s.metadata.as_ref())
                .map(|m| m.length)
                .unwrap_or(0);

            if action_count < threshold {
                return PhoenixRetrievalCluster::parse(
                    &query.params.get(PhoenixRetrievalNewUserInferenceClusterId),
                );
            }
        }

        if let Some(decider) = &query.decider {
            let is_prod = matches!(
                configured_cluster,
                PhoenixRetrievalCluster::Experiment1Fou | PhoenixRetrievalCluster::Experiment2Fou
            );
            if is_prod {
                if decider.enabled("override_retrieval_use_experiment2_fou") {
                    return PhoenixRetrievalCluster::Experiment2Fou;
                }
                if decider.enabled("override_retrieval_use_experiment1_fou") {
                    return PhoenixRetrievalCluster::Experiment1Fou;
                }
            }
        }

        configured_cluster
    }

    pub(crate) fn seen_post_exclusions(query: &ScoredPostsQuery) -> RetrievalExclusions {
        if !query.params.get(PhoenixRetrievalExcludeSeenPosts) {
            return RetrievalExclusions::default();
        }
        let mut post_ids: Vec<u64> = query
            .seen_ids
            .iter()
            .chain(query.served_ids.iter())
            .copied()
            .collect();
        if query.bloom_filter_entries.is_empty() {
            post_ids.extend(query.impressed_post_ids.iter().copied());
        }
        post_ids.sort_unstable();
        post_ids.dedup();
        let bloom_filters = query
            .bloom_filter_entries
            .iter()
            .map(|e| xai_recsys_proto::ExcludedPostsBloomFilter {
                bit_array: e.bloom_filter.clone(),
                size_cap: e.size_cap,
                false_positive_rate: e.false_positive_rate,
            })
            .collect();
        RetrievalExclusions {
            post_ids,
            bloom_filters,
        }
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePhoenixSource)
            && (!query.is_topic_request() || query.is_bulk_topic_request())
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixSource: missing retrieval_sequence".to_string())?;

        let cluster = Self::resolve_cluster(query);
        let client_context = build_client_context(query);
        let user_context = build_user_context(query);

        let response = self
            .phoenix_retrieval_client
            .retrieve_excluding(
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                quality_factor::apply(query.params.get(PhoenixMaxResults)),
                query.params.get(PhoenixColdStartMaxResults),
                vec![],
                None,
                client_context,
                user_context,
                query.params.get(PhoenixXdsRetrievalMaxRetries),
                vec![],
                Self::seen_post_exclusions(query),
                retrieval_value_model_weights(&query.params.get(RetrievalValueModelWeights)),
            )
            .await
            .map_err(|e| format!("PhoenixSource: {e}"))?;

        let candidates =
            candidates_from_retrieval_response(response, served_type_for_dataset, cluster);
        record_zero_fav_retrieved(&candidates);
        Ok(candidates)
    }
}

fn record_zero_fav_retrieved(candidates: &[PostCandidate]) {
    let count = candidates
        .iter()
        .filter(|c| c.served_type == Some(pb::ServedType::ForYouPhoenixRetrievalZeroFav))
        .count() as u64;
    if count == 0 {
        return;
    }
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    receiver.incr(ZERO_FAV_RETRIEVED_METRIC, &[("type", "sum")], count);
    receiver.incr(ZERO_FAV_RETRIEVED_METRIC, &[("type", "requests")], 1);
}

fn served_type_for_dataset(dataset_type: u32) -> pb::ServedType {
    match RetrievalDatasetType::try_from(dataset_type as i32) {
        Ok(RetrievalDatasetType::HomeCold) => pb::ServedType::ForYouPhoenixRetrievalCold,
        Ok(RetrievalDatasetType::ZeroFav) => pb::ServedType::ForYouPhoenixRetrievalZeroFav,
        _ => pb::ServedType::ForYouPhoenixRetrieval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::MockRetrievalClient;

    fn source() -> PhoenixSource {
        PhoenixSource {
            phoenix_retrieval_client: Arc::new(MockRetrievalClient),
        }
    }

    #[test]
    fn enable_returns_false_when_in_network_only() {
        let query = ScoredPostsQuery {
            in_network_only: true,
            ..Default::default()
        };
        assert!(!source().enable(&query));
    }

    #[test]
    fn enable_returns_false_when_has_cached_posts() {
        let query = ScoredPostsQuery {
            has_cached_posts: true,
            ..Default::default()
        };
        assert!(!source().enable(&query));
    }

    #[test]
    fn candidates_from_response_sets_served_type() {
        use xai_recsys_proto::{
            RetrieveTopKCandidatesResponse, ScoredCandidate, ScoredCandidates, TweetInfo,
        };

        let tweet = |id, author, dataset_type| ScoredCandidate {
            candidate: Some(TweetInfo {
                tweet_id: id,
                author_id: author,
                ..Default::default()
            }),
            dataset_type,
            ..Default::default()
        };
        let response = RetrieveTopKCandidatesResponse {
            top_k_candidates: vec![ScoredCandidates {
                user_id: 1,
                candidates: vec![
                    tweet(10, 1, RetrievalDatasetType::Home as u32),
                    tweet(11, 2, RetrievalDatasetType::HomeCold as u32),
                ],
            }],
        };
        let got: Vec<(u64, pb::ServedType)> = candidates_from_retrieval_response(
            response,
            served_type_for_dataset,
            PhoenixRetrievalCluster::Experiment1Fou,
        )
        .into_iter()
        .map(|c| (c.tweet_id, c.served_type.unwrap()))
        .collect();
        assert_eq!(
            got,
            vec![
                (10, pb::ServedType::ForYouPhoenixRetrieval),
                (11, pb::ServedType::ForYouPhoenixRetrievalCold),
            ]
        );
    }
}
