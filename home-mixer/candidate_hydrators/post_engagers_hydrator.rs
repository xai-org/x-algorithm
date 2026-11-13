use crate::clients::post_engagers_client::PostEngagersClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnablePostEngagers;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::utils::{
    build_moka_cache, MokaCache, MokaCacheConfig,
};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};
use xai_proto::engagement_counter::PostEngagers;

const CACHE_SIZE: u64 = 500_000;
const CACHE_TTL: Duration = Duration::from_secs(60);

pub struct PostEngagersHydrator {
    client: Arc<dyn PostEngagersClient>,
    cache: MokaCache<u64, Option<Arc<PostEngagers>>>,
    no_engagers: Arc<PostEngagers>,
}

impl PostEngagersHydrator {
    pub fn new(client: Arc<dyn PostEngagersClient>) -> Self {
        Self {
            client,
            cache: build_moka_cache(MokaCacheConfig {
                size: CACHE_SIZE,
                ttl: CACHE_TTL,
            }),
            no_engagers: Arc::new(PostEngagers::default()),
        }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for PostEngagersHydrator {
    type Fetched = Vec<Option<Option<Arc<PostEngagers>>>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        (query.params.get(EnablePostEngagers) || query.is_shadow_traffic) && !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Option<Arc<PostEngagers>>>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<Option<Arc<PostEngagers>>>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<Option<Arc<PostEngagers>>>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            if let Some(value) = value {
                candidate.post_engagers = value;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.post_engagers = hydrated.post_engagers;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for PostEngagersHydrator {
    type CacheKey = u64;
    type Value = Option<Arc<PostEngagers>>;

    fn cache_store(&self) -> &dyn CacheStore<u64, Option<Arc<PostEngagers>>> {
        &self.cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.get_original_tweet_id()
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<Option<Arc<PostEngagers>>>> {
        let mut post_ids: Vec<u64> = misses.iter().map(|c| c.get_original_tweet_id()).collect();
        post_ids.sort_unstable();
        post_ids.dedup();

        let records: HashMap<u64, Arc<PostEngagers>> = self
            .client
            .get_post_engagers(&post_ids)
            .await
            .into_iter()
            .map(|(post_id, record)| {
                let record = record.map_or_else(|| Arc::clone(&self.no_engagers), Arc::new);
                (post_id, record)
            })
            .collect();

        misses
            .iter()
            .map(|c| {
                records
                    .get(&c.get_original_tweet_id())
                    .map(|record| Some(Arc::clone(record)))
            })
            .collect()
    }
}
