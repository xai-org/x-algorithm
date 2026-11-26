use crate::filters::topic_ids_filter::{TopicFilteringOverrideMap, TopicIdExpansion};
use crate::models::candidate::PostCandidate;
use crate::models::candidate_features::TopicFilteringExperiment;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    PhoenixMaxResults, PhoenixRetrievalTopicInferenceClusterId, PhoenixXdsRetrievalMaxRetries,
    TopicFilteringId, TopicFilteringOverrides,
};
use crate::util::egress::phoenix_retrieval;
use crate::util::phoenix_request::candidates_from_retrieval_response;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalClient;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_stats_receiver::global_stats_receiver;

const INCLUDED_TOPIC_METRIC: &str = "PhoenixTopicsSource.included_topic_id";

pub struct PhoenixTopicsSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixTopicsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.is_topic_request()
            && !query.is_bulk_topic_request()
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixTopicsSource: missing retrieval_sequence".to_string())?;

        let cluster = PhoenixRetrievalCluster::parse(
            &query.params.get(PhoenixRetrievalTopicInferenceClusterId),
        );

        let default_experiment =
            TopicFilteringExperiment::parse(&query.params.get(TopicFilteringId));
        let override_map =
            TopicFilteringOverrideMap::parse(&query.params.get(TopicFilteringOverrides));
        let topic_filter_mode = override_map
            .resolve(&query.topic_ids, default_experiment)
            .as_proto_mode();

        let max_results = quality_factor::apply(query.params.get(PhoenixMaxResults));

        let topic_entity_ids: Vec<u64> = query
            .topic_ids
            .iter()
            .map(|&tid| TopicIdExpansion::resolve_first(tid) as u64)
            .collect();

        if let Some(receiver) = global_stats_receiver() {
            for &tid in &query.topic_ids {
                let tid_str = tid.to_string();
                receiver.incr(INCLUDED_TOPIC_METRIC, &[("topic_id", &tid_str)], 1);
            }
        }

        let response = phoenix_retrieval(query, cluster, self.phoenix_retrieval_client.as_ref())
            .retrieve(
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                max_results,
                0,
                topic_entity_ids,
                Some(topic_filter_mode),
                None,
                None,
                query.params.get(PhoenixXdsRetrievalMaxRetries),
                vec![],
            )
            .await
            .map_err(|e| format!("PhoenixTopicsSource: {e}"))?;

        Ok(candidates_from_retrieval_response(
            response,
            |_| pb::ServedType::ForYouPhoenixRetrieval,
            cluster,
        ))
    }
}
