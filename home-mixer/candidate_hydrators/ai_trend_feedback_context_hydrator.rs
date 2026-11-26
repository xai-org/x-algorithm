use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableAiTrendFeedbackContext;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::component_library::utils::{
    build_moka_cache, build_moka_cache_value_ttl, MokaCache, MokaCacheConfig,
};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};

const TOP_TREND_COUNT: usize = 2;
const TREND_CACHE_SIZE: u64 = 500_000;
const TREND_TTL: Duration = Duration::from_secs(5 * 60);
const NO_TREND_TTL: Duration = Duration::from_secs(30);
const NAME_CACHE_SIZE: u64 = 100_000;
const NAME_TTL: Duration = Duration::from_secs(30 * 60);

fn trend_ttl(trend_id: &Option<i64>) -> Duration {
    if trend_id.is_some() {
        TREND_TTL
    } else {
        NO_TREND_TTL
    }
}

pub struct AiTrendFeedbackContextHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
    trend_cache: MokaCache<u64, Option<i64>>,
    name_cache: MokaCache<i64, String>,
}

impl AiTrendFeedbackContextHydrator {
    pub fn new(strato_client: Arc<dyn StratoClient + Send + Sync>) -> Self {
        Self {
            strato_client,
            trend_cache: build_moka_cache_value_ttl(TREND_CACHE_SIZE, trend_ttl),
            name_cache: build_moka_cache(MokaCacheConfig {
                size: NAME_CACHE_SIZE,
                ttl: NAME_TTL,
            }),
        }
    }

    async fn fetch_trends(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<HashMap<usize, (String, String)>, String> {
        let candidate_trends: Vec<(usize, i64)> = self
            .fetch_cached(query, candidates, stage)
            .await?
            .into_iter()
            .enumerate()
            .filter_map(|(idx, trend_id)| Some((idx, trend_id.flatten()?)))
            .collect();
        let selected_ids = Self::select_feedback_targets(&candidate_trends);
        if selected_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut unique_ids: Vec<i64> = selected_ids.values().copied().collect();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        let name_by_id = self.trend_names(unique_ids).await;
        Ok(selected_ids
            .into_iter()
            .filter_map(|(idx, trend_id)| {
                name_by_id
                    .get(&trend_id)
                    .cloned()
                    .map(|name| (idx, (name, trend_id.to_string())))
            })
            .collect())
    }

    async fn trend_names(&self, trend_ids: Vec<i64>) -> HashMap<i64, String> {
        let mut name_by_id = HashMap::new();
        let mut misses = Vec::new();
        for trend_id in trend_ids {
            match self.name_cache.get(&trend_id).await {
                Some(name) => {
                    name_by_id.insert(trend_id, name);
                }
                None => misses.push(trend_id),
            }
        }
        if misses.is_empty() {
            return name_by_id;
        }
        let names = self.strato_client.batch_get_ai_trend_name(&misses).await;
        let fetched: Vec<(i64, String)> = misses
            .into_iter()
            .zip(names)
            .filter_map(|(trend_id, result)| match result {
                Ok(Some(name)) if !name.is_empty() => Some((trend_id, name)),
                Ok(_) => None,
                Err(e) => {
                    warn!(
                        "AiTrendFeedbackContextHydrator: trend name fetch error: {}",
                        e
                    );
                    None
                }
            })
            .collect();
        for (trend_id, name) in fetched {
            self.name_cache.insert(trend_id, name.clone()).await;
            name_by_id.insert(trend_id, name);
        }
        name_by_id
    }

    fn is_eligible_original_post(candidate: &PostCandidate) -> bool {
        candidate.in_reply_to_tweet_id.is_none()
            && candidate.retweeted_tweet_id.is_none()
            && candidate.ancestors.is_empty()
            && candidate.following_replied_user_ids.is_empty()
    }

    fn select_feedback_targets(candidate_trends: &[(usize, i64)]) -> HashMap<usize, i64> {
        let mut frequency: HashMap<i64, usize> = HashMap::new();
        for (_, trend_id) in candidate_trends {
            *frequency.entry(*trend_id).or_default() += 1;
        }

        let mut ranked: Vec<(i64, usize)> = frequency.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(TOP_TREND_COUNT);

        let mut rng = rand::rng();
        let mut selected: HashMap<usize, i64> = HashMap::new();
        let mut used_indices = std::collections::HashSet::new();

        for (trend_id, _) in ranked {
            let candidates_for_trend: Vec<usize> = candidate_trends
                .iter()
                .filter(|(idx, id)| !used_indices.contains(idx) && *id == trend_id)
                .map(|(idx, _)| *idx)
                .collect();
            let Some(&idx) = candidates_for_trend.choose(&mut rng) else {
                continue;
            };
            used_indices.insert(idx);
            selected.insert(idx, trend_id);
        }

        selected
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for AiTrendFeedbackContextHydrator {
    type Fetched = HashMap<usize, (String, String)>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableAiTrendFeedbackContext)
            && !query.is_topic_request()
            && !query.in_network_only
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<HashMap<usize, (String, String)>, String> {
        self.fetch_trends(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<HashMap<usize, (String, String)>, String> {
        self.fetch_trends(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        mut trends: HashMap<usize, (String, String)>,
    ) {
        for (i, candidate) in candidates.iter_mut().enumerate() {
            let (name, trend_id) = trends
                .remove(&i)
                .map_or((None, None), |(name, id)| (Some(name), Some(id)));
            candidate.ai_trend_name = name;
            candidate.ai_trend_id = trend_id;
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.ai_trend_name = hydrated.ai_trend_name;
        candidate.ai_trend_id = hydrated.ai_trend_id;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for AiTrendFeedbackContextHydrator {
    type CacheKey = u64;
    type Value = Option<i64>;

    fn cache_store(&self) -> &dyn CacheStore<u64, Option<i64>> {
        &self.trend_cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.tweet_id
    }

    fn existing_value(&self, candidate: &PostCandidate) -> Option<Option<i64>> {
        (!Self::is_eligible_original_post(candidate)).then_some(None)
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<Option<i64>>> {
        let tweet_ids: Vec<u64> = misses.iter().map(|c| c.tweet_id).collect();
        self.strato_client
            .batch_get_tweet_ai_trend(&tweet_ids)
            .await
            .into_iter()
            .map(|result| match result {
                Ok(Some(trend_id)) if trend_id != 0 => Some(Some(trend_id)),
                Ok(_) => Some(None),
                Err(e) => {
                    warn!("AiTrendFeedbackContextHydrator: strato fetch error: {}", e);
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use xai_candidate_pipeline::component_library::clients::MockStratoClient;

    #[tokio::test]
    async fn trends_and_names_are_served_from_cache_on_repeat_requests() {
        let strato = Arc::new(MockStratoClient {
            tweet_ai_trends: HashMap::from([(1, 7), (2, 7), (3, 0)]),
            ai_trend_names: HashMap::from([(7, "Trend".to_string())]),
            ..Default::default()
        });
        let hydrator = AiTrendFeedbackContextHydrator::new(strato.clone());
        let query = ScoredPostsQuery::default();
        let candidates: Vec<PostCandidate> = [1, 2, 3]
            .into_iter()
            .map(|tweet_id| PostCandidate {
                tweet_id,
                ..Default::default()
            })
            .collect();

        let mut calls_after = Vec::new();
        for _ in 0..2 {
            let mut hydrated = candidates.clone();
            let fetched = hydrator.fetch(&query, &hydrated).await.unwrap();
            hydrator.apply(&query, &mut hydrated, fetched);
            let labeled: Vec<_> = hydrated
                .iter()
                .filter(|c| c.ai_trend_name.is_some())
                .map(|c| (c.tweet_id, c.ai_trend_name.clone(), c.ai_trend_id.clone()))
                .collect();
            assert_eq!(labeled.len(), 1);
            assert!(matches!(labeled[0].0, 1 | 2));
            assert_eq!(labeled[0].1.as_deref(), Some("Trend"));
            assert_eq!(labeled[0].2.as_deref(), Some("7"));
            calls_after.push(strato.call_count.load(Ordering::SeqCst));
        }
        assert_eq!(calls_after, vec![2, 2]);
    }
}
