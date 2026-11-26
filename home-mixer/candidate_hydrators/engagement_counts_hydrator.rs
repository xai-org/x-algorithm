use crate::clients::engagement_counts_client::EngagementCountsClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableEngagementCountsHydration;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::utils::{
    build_moka_cache, MokaCache, MokaCacheConfig,
};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};
use xai_proto::engagement_counter::EngagementCounts;

#[derive(Clone, Debug, Default)]
pub struct CachedCounts {
    fav_count: Option<i64>,
    reply_count: Option<i64>,
    repost_count: Option<i64>,
    quote_count: Option<i64>,
    view_count: Option<u64>,
    view_count_on_home: Option<u64>,
    bookmark_count: Option<i64>,
}

impl CachedCounts {
    fn from_proto(c: &EngagementCounts) -> Self {
        Self {
            fav_count: Some(c.fav_count as i64),
            reply_count: Some(c.reply_count as i64),
            repost_count: Some(c.retweet_count as i64),
            quote_count: Some(c.quote_count as i64),
            view_count: Some(c.view_count),
            view_count_on_home: Some(c.view_count_on_home),
            bookmark_count: Some(c.bookmark_count as i64),
        }
    }
}

pub struct EngagementCountsHydrator {
    pub client: Arc<dyn EngagementCountsClient + Send + Sync>,
    cache: MokaCache<u64, CachedCounts>,
}

impl EngagementCountsHydrator {
    pub async fn new(client: Arc<dyn EngagementCountsClient + Send + Sync>) -> Self {
        let cache = build_moka_cache(MokaCacheConfig {
            size: 1_000_000,
            ttl: Duration::from_secs(60),
        });
        Self { client, cache }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for EngagementCountsHydrator {
    type Fetched = Vec<Option<CachedCounts>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableEngagementCountsHydration) || query.is_shadow_traffic
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<CachedCounts>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<CachedCounts>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<CachedCounts>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            if let Some(value) = value {
                candidate.fav_count = value.fav_count;
                candidate.reply_count = value.reply_count;
                candidate.repost_count = value.repost_count;
                candidate.quote_count = value.quote_count;
                candidate.view_count = value.view_count;
                candidate.view_count_on_home = value.view_count_on_home;
                candidate.bookmark_count = value.bookmark_count;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.fav_count = hydrated.fav_count;
        candidate.reply_count = hydrated.reply_count;
        candidate.repost_count = hydrated.repost_count;
        candidate.quote_count = hydrated.quote_count;
        candidate.view_count = hydrated.view_count;
        candidate.view_count_on_home = hydrated.view_count_on_home;
        candidate.bookmark_count = hydrated.bookmark_count;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for EngagementCountsHydrator {
    type CacheKey = u64;
    type Value = CachedCounts;

    fn cache_store(&self) -> &dyn CacheStore<u64, CachedCounts> {
        &self.cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.get_original_tweet_id()
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<CachedCounts>> {
        let mut unique_ids: Vec<u64> = misses.iter().map(|c| c.get_original_tweet_id()).collect();
        unique_ids.sort_unstable();
        unique_ids.dedup();

        let counts = if unique_ids.is_empty() {
            HashMap::new()
        } else {
            self.client
                .get_engagement_counts(&unique_ids)
                .await
                .unwrap_or_else(|e| {
                    warn!(error = %e, "engagement_counts_hydration dragonfly_error");
                    HashMap::new()
                })
        };

        misses
            .iter()
            .map(|c| {
                Some(
                    counts
                        .get(&c.get_original_tweet_id())
                        .map(CachedCounts::from_proto)
                        .unwrap_or_default(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::engagement_counts_client::MockEngagementCountsClient;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const COUNTS: &str = "rust_home_mixer_enable_engagement_counts_hydration";

    fn query(has_cached_posts: bool, flags: &[(&str, &str)]) -> ScoredPostsQuery {
        let mut query = ScoredPostsQuery {
            has_cached_posts,
            ..Default::default()
        };
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        for (key, value) in flags {
            results.override_fs(key.to_string(), value);
        }
        query.params = results.into();
        query
    }

    fn view_counts(entries: &[(u64, u64)]) -> HashMap<u64, EngagementCounts> {
        entries
            .iter()
            .map(|&(id, view_count)| {
                (
                    id,
                    EngagementCounts {
                        view_count,
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    async fn hydrator(counts: HashMap<u64, EngagementCounts>) -> EngagementCountsHydrator {
        EngagementCountsHydrator::new(Arc::new(MockEngagementCountsClient { counts })).await
    }

    #[tokio::test]
    async fn enable_matrix() {
        let h = hydrator(HashMap::new()).await;
        assert!(h.enable(&query(false, &[(COUNTS, "true")])));
        assert!(h.enable(&query(true, &[(COUNTS, "true")])));
        assert!(!h.enable(&query(false, &[(COUNTS, "false")])));
        assert!(!h.enable(&query(true, &[(COUNTS, "false")])));
        let mut shadow = query(true, &[(COUNTS, "false")]);
        shadow.is_shadow_traffic = true;
        assert!(h.enable(&shadow));
    }

    #[tokio::test]
    async fn no_cached_posts_hydrates_all() {
        let h = hydrator(HashMap::from([(
            20,
            EngagementCounts {
                view_count: 7,
                view_count_on_home: 3,
                ..Default::default()
            },
        )]))
        .await;
        let candidates = [PostCandidate {
            tweet_id: 20,
            author_id: 2,
            author_followers_count: Some(5000),
            view_count: Some(999),
            ..Default::default()
        }];
        let q = query(false, &[(COUNTS, "true")]);
        let misses: Vec<&PostCandidate> = candidates.iter().collect();
        let result = h.fetch_values(&q, &misses).await;
        assert_eq!(result[0].as_ref().unwrap().view_count, Some(7));
        assert_eq!(result[0].as_ref().unwrap().view_count_on_home, Some(3));
    }

    #[tokio::test]
    async fn cached_posts_refetch_all_candidates() {
        let h = hydrator(view_counts(&[(10, 5), (20, 6), (30, 7)])).await;
        let candidates = [
            PostCandidate {
                tweet_id: 10,
                author_followers_count: Some(100),
                view_count: Some(999),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 20,
                author_followers_count: Some(5000),
                view_count: Some(999),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 30,
                in_reply_to_tweet_id: Some(1),
                view_count: Some(888),
                ..Default::default()
            },
        ];
        let q = query(true, &[(COUNTS, "true")]);
        let misses: Vec<&PostCandidate> = candidates.iter().collect();
        let result = h.fetch_values(&q, &misses).await;
        let views: Vec<_> = result
            .iter()
            .map(|r| r.as_ref().unwrap().view_count)
            .collect();
        assert_eq!(views, vec![Some(5), Some(6), Some(7)]);
    }

    #[derive(Default)]
    struct CountingClient {
        counts: HashMap<u64, EngagementCounts>,
        calls: AtomicUsize,
    }

    #[tonic::async_trait]
    impl EngagementCountsClient for CountingClient {
        async fn get_engagement_counts(
            &self,
            tweet_ids: &[u64],
        ) -> Result<HashMap<u64, EngagementCounts>, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(tweet_ids
                .iter()
                .filter_map(|&id| self.counts.get(&id).map(|c| (id, *c)))
                .collect())
        }
    }

    #[tokio::test]
    async fn repeat_hydration_served_from_moka_cache() {
        let client = Arc::new(CountingClient {
            counts: view_counts(&[(10, 5)]),
            calls: AtomicUsize::new(0),
        });
        let h = EngagementCountsHydrator::new(client.clone()).await;
        let candidates = vec![PostCandidate {
            tweet_id: 10,
            author_id: 1,
            author_followers_count: Some(100),
            ..Default::default()
        }];
        let q = query(true, &[]);

        let first = xai_candidate_pipeline::hydrator::Hydrator::hydrate(&h, &q, &candidates).await;
        let second = xai_candidate_pipeline::hydrator::Hydrator::hydrate(&h, &q, &candidates).await;

        assert_eq!(first[0].as_ref().unwrap().view_count, Some(5));
        assert_eq!(second[0].as_ref().unwrap().view_count, Some(5));
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
    }
}
