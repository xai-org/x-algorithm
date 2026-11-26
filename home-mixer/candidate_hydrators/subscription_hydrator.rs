use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};

pub struct SubscriptionHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: QuickCache<u64, Option<u64>>,
}

impl SubscriptionHydrator {
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        let cache = default_quick_cache();
        Self { tes_client, cache }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for SubscriptionHydrator {
    type Fetched = Vec<Option<Option<u64>>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Option<u64>>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<Option<u64>>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<Option<u64>>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            if let Some(value) = value {
                candidate.subscription_author_id = value;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.subscription_author_id = hydrated.subscription_author_id;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for SubscriptionHydrator {
    type CacheKey = u64;
    type Value = Option<u64>;

    fn cache_store(&self) -> &dyn CacheStore<u64, Option<u64>> {
        &self.cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.tweet_id
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<Option<u64>>> {
        let tweet_ids: Vec<u64> = misses.iter().map(|c| c.tweet_id).collect();
        let author_ids = self
            .tes_client
            .get_subscription_author_ids(tweet_ids.clone())
            .await;
        tweet_ids
            .iter()
            .map(|tweet_id| match author_ids.get(tweet_id) {
                Some(Ok(author_id)) => Some(*author_id),
                Some(Err(_)) | None => None,
            })
            .collect()
    }
}
