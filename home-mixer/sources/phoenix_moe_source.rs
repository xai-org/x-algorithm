use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixMOESource, PhoenixMOEMaxResults, PhoenixMoeColdStartMaxResults,
    PhoenixRetrievalMOEInferenceClusterId, PhoenixXdsRetrievalMaxRetries,
};
use crate::sources::sid_source::sid_source_enabled;
use crate::util::egress::phoenix_retrieval;
use crate::util::phoenix_request::candidates_from_retrieval_response;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalClient;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

pub const PHOENIX_MOE_SOURCE_KILL_SWITCH_DECIDER: &str = "disable_home_mixer_phoenix_moe_source";

pub struct PhoenixMOESource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixMOESource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePhoenixMOESource)
            && !sid_source_enabled(query)
            && (!query.is_topic_request() || query.is_bulk_topic_request())
            && !query.in_network_only
            && !query.has_cached_posts
            && !query
                .decider
                .as_ref()
                .is_some_and(|d| d.enabled(PHOENIX_MOE_SOURCE_KILL_SWITCH_DECIDER))
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixMOESource: missing retrieval_sequence".to_string())?;

        let cluster = PhoenixRetrievalCluster::parse(
            &query.params.get(PhoenixRetrievalMOEInferenceClusterId),
        );

        let response = phoenix_retrieval(query, cluster, self.phoenix_retrieval_client.as_ref())
            .retrieve(
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                quality_factor::apply(query.params.get(PhoenixMOEMaxResults)),
                query.params.get(PhoenixMoeColdStartMaxResults),
                vec![],
                None,
                None,
                None,
                query.params.get(PhoenixXdsRetrievalMaxRetries),
                vec![],
            )
            .await
            .map_err(|e| format!("PhoenixMOESource: {e}"))?;

        Ok(candidates_from_retrieval_response(
            response,
            |_| pb::ServedType::ForYouPhoenixRetrievalMoe,
            cluster,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::MockRetrievalClient;
    use xai_decider::{Decider, DeciderStore};
    use xai_feature_switches::{FeatureSwitches, RecipientBuilder};

    const ENABLE_FS: &str = "rust_home_mixer_enable_phoenix_moe_source";
    const SID_SOURCE_FS: &str = "rust_home_mixer_enable_sid_source";

    fn source() -> PhoenixMOESource {
        PhoenixMOESource {
            phoenix_retrieval_client: Arc::new(MockRetrievalClient),
        }
    }

    fn query(kill_switch: Option<bool>) -> ScoredPostsQuery {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs(ENABLE_FS.to_string(), "true");
        results.override_fs(SID_SOURCE_FS.to_string(), "false");
        let decider = kill_switch.map(|killed| {
            Decider::new(DeciderStore::new(HashMap::new()))
                .with_overrides(HashMap::from([(
                    PHOENIX_MOE_SOURCE_KILL_SWITCH_DECIDER.to_string(),
                    killed,
                )]))
                .with_recipient(1)
        });
        ScoredPostsQuery {
            user_id: 1,
            params: results.into(),
            decider,
            ..Default::default()
        }
    }

    #[test]
    fn enable_when_kill_switch_absent() {
        assert!(source().enable(&query(None)));
    }

    #[test]
    fn disable_when_kill_switch_on() {
        assert!(!source().enable(&query(Some(true))));
    }
}
