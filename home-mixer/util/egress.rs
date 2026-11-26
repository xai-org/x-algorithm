use crate::models::query::ScoredPostsQuery;
use crate::params::PhoenixXdsRetrievalUncompressedClustersList;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    PhoenixRetrievalClient, PhoenixRetrievalCluster, RetrievalCompression,
};
use xai_feature_switches::Param;

pub use xai_candidate_pipeline::component_library::egress::PredictionDispatch;

pub fn phoenix_retrieval<'a>(
    query: &ScoredPostsQuery,
    cluster: PhoenixRetrievalCluster,
    client: &'a (dyn PhoenixRetrievalClient + Send + Sync),
) -> RetrievalCompression<'a> {
    let name: &'static str = cluster.into();
    let uncompressed = query
        .params
        .0
        .as_ref()
        .and_then(|fs| fs.get_array(PhoenixXdsRetrievalUncompressedClustersList.key()))
        .is_some_and(|names| names.iter().any(|n| n.as_str() == Some(name)));
    RetrievalCompression {
        client,
        uncompressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::MockRetrievalClient;
    use xai_feature_switches::{FeatureSwitches, Param, RecipientBuilder, Value};

    #[test]
    fn retrieval_is_uncompressed_only_for_listed_clusters() {
        let fs = FeatureSwitches::new(vec![]).unwrap();
        let mut results = fs.match_recipient(&RecipientBuilder::new().build());
        results.override_value(
            PhoenixXdsRetrievalUncompressedClustersList
                .key()
                .to_string(),
            Value::Array(vec![Value::from("Experiment1Fou")]),
        );
        let query = ScoredPostsQuery {
            params: results.into(),
            ..Default::default()
        };
        let client = MockRetrievalClient;
        assert!(
            phoenix_retrieval(&query, PhoenixRetrievalCluster::Experiment1Fou, &client)
                .uncompressed
        );
        assert!(
            !phoenix_retrieval(&query, PhoenixRetrievalCluster::Experiment1Amp, &client)
                .uncompressed
        );
    }
}
