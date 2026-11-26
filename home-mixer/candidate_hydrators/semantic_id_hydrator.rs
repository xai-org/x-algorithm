use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableSemanticIdHydration, SidServerEndpoint};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::clients::SidClient;
use xai_candidate_pipeline::component_library::utils::{
    build_moka_cache, MokaCache, MokaCacheConfig,
};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};

const CACHE_SIZE: u64 = 1_000_000;
const CACHE_TTL: Duration = Duration::from_secs(7200);

fn endpoint_hash(endpoint: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    endpoint.hash(&mut hasher);
    hasher.finish()
}

pub struct SemanticIdHydrator {
    client: Arc<dyn SidClient>,
    cache: MokaCache<(u64, u64), Vec<i32>>,
}

impl SemanticIdHydrator {
    pub fn new(client: Arc<dyn SidClient>) -> Self {
        let cache = build_moka_cache(MokaCacheConfig {
            size: CACHE_SIZE,
            ttl: CACHE_TTL,
        });
        Self { client, cache }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for SemanticIdHydrator {
    type Fetched = Vec<Option<Vec<i32>>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
            && query.params.get(EnableSemanticIdHydration)
            && !query.params.get(SidServerEndpoint).is_empty()
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Vec<i32>>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<Vec<i32>>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<Vec<i32>>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            if let Some(value) = value {
                candidate.semantic_ids = Some(value);
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.semantic_ids = hydrated.semantic_ids;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for SemanticIdHydrator {
    type CacheKey = (u64, u64);
    type Value = Vec<i32>;

    fn cache_store(&self) -> &dyn CacheStore<(u64, u64), Vec<i32>> {
        &self.cache
    }

    fn cache_key(&self, query: &ScoredPostsQuery, candidate: &PostCandidate) -> (u64, u64) {
        (
            candidate.get_original_tweet_id(),
            endpoint_hash(&query.params.get(SidServerEndpoint)),
        )
    }

    async fn fetch_values(
        &self,
        query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<Vec<i32>>> {
        let endpoint = query.params.get(SidServerEndpoint);

        let mut unique_ids: Vec<i64> = misses
            .iter()
            .map(|c| c.get_original_tweet_id() as i64)
            .collect();
        unique_ids.sort_unstable();
        unique_ids.dedup();

        let Some(looked_up) = self.client.lookup(&endpoint, &unique_ids).await else {
            return vec![None; misses.len()];
        };

        let codes_by_id: HashMap<i64, Option<Vec<i32>>> =
            unique_ids.into_iter().zip(looked_up).collect();

        misses
            .iter()
            .map(|candidate| {
                codes_by_id
                    .get(&(candidate.get_original_tweet_id() as i64))
                    .cloned()
                    .flatten()
            })
            .collect()
    }
}
