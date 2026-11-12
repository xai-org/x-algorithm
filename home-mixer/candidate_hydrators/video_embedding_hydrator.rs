use crate::clients::post_v8_search_embedding_client::PostV8SearchEmbeddingClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableVideoCarouselCtrModel, EnableVideoCarouselEmbeddingDedupe, VideoCarouselEmbeddingVersion,
};
use crate::util::video_carousel;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

pub struct VideoEmbeddingHydrator {
    client: Arc<dyn PostV8SearchEmbeddingClient>,
}

impl VideoEmbeddingHydrator {
    pub fn new(client: Arc<dyn PostV8SearchEmbeddingClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for VideoEmbeddingHydrator {
    type Fetched = Vec<Option<Vec<f32>>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query)
            && (query.params.get(EnableVideoCarouselEmbeddingDedupe)
                || query.params.get(EnableVideoCarouselCtrModel))
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Vec<f32>>>, String> {
        let post_ids: Vec<u64> = candidates
            .iter()
            .filter(|candidate| video_carousel::is_carousel_video(candidate))
            .map(|candidate| candidate.get_original_tweet_id())
            .collect();
        let version = query.params.get(VideoCarouselEmbeddingVersion);
        let embeddings = self.client.fetch_embeddings(&post_ids, &version).await;

        Ok(candidates
            .iter()
            .map(|candidate| {
                video_carousel::is_carousel_video(candidate)
                    .then(|| embeddings.get(&candidate.get_original_tweet_id()))
                    .flatten()
                    .cloned()
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        embeddings: Vec<Option<Vec<f32>>>,
    ) {
        for (candidate, embedding) in candidates.iter_mut().zip(embeddings) {
            candidate.video_embedding = embedding;
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.video_embedding = hydrated.video_embedding;
    }
}
