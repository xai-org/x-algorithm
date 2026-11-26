use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableVideoCarouselPopularVideoSource, PhoenixXdsRetrievalMaxRetries,
    VideoCarouselPopularVideoClusterId, VideoCarouselPopularVideoMaxResults,
};
use crate::util::egress::phoenix_retrieval;
use crate::util::phoenix_request::candidates_from_retrieval_response;
use crate::util::video_carousel;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    PhoenixRetrievalClient, PhoenixRetrievalCluster,
};
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

pub struct PopularVideoSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PopularVideoSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableVideoCarouselPopularVideoSource)
            && video_carousel::is_enabled(query)
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PopularVideoSource: missing retrieval_sequence".to_string())?;
        let cluster =
            PhoenixRetrievalCluster::parse(&query.params.get(VideoCarouselPopularVideoClusterId));

        let response = phoenix_retrieval(query, cluster, self.phoenix_retrieval_client.as_ref())
            .retrieve(
                cluster,
                query.user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                quality_factor::apply(query.params.get(VideoCarouselPopularVideoMaxResults)),
                0,
                vec![],
                None,
                None,
                None,
                query.params.get(PhoenixXdsRetrievalMaxRetries),
                vec![],
            )
            .await
            .map_err(|e| format!("PopularVideoSource: {e}"))?;

        Ok(candidates_from_retrieval_response(
            response,
            |_| pb::ServedType::ForYouPhoenixRetrievalPopularVideo,
            cluster,
        ))
    }
}
