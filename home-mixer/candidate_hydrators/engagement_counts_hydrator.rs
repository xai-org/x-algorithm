use crate::clients::engagement_counts_client::EngagementCountsClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    ColdStartFollowerCap, ColdStartFollowerDecayWidth, EnableEngagementCountsHydration,
    EnableViewerColdStart,
};
use crate::scorers::author_cold_start::cold_start_base_eligible;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::component_library::utils::{
    MokaCache, MokaCacheConfig, build_moka_cache,
};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};
use xai_proto::engagement_counter::EngagementCounts;

#[derive(Clone, Debug, Default)]
pub struct CachedCounts {
    fav_count: Option<i64>,
    reply_count: Option<i64>,
    repost_count: Option<i64>,
    quote_count: Option<i64>,
    view_count: Option<u64>,
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
            bookmark_count: Some(c.bookmark_count as i64),
        }
    }

    fn to_candidate(&self) -> PostCandidate {
        PostCandidate {
            fav_count: self.fav_count,
            reply_count: self.reply_count,
            repost_count: self.repost_count,
            quote_count: self.quote_count,
            view_count: self.view_count,
            bookmark_count: self.bookmark_count,
            ..Default::default()
        }
    }
}

fn preserve_counts(c: &PostCandidate) -> PostCandidate {
    PostCandidate {
        fav_count: c.fav_count,
        reply_count: c.reply_count,
        repost_count: c.repost_count,
        quote_count: c.quote_count,
        view_count: c.view_count,
        bookmark_count: c.bookmark_count,
        ..Default::default()
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
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for EngagementCountsHydrator {
    type CacheKey = u64;
    type CacheValue = CachedCounts;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableViewerColdStart)
            || (!query.has_cached_posts
                && (query.params.get(EnableEngagementCountsHydration) || query.is_shadow_traffic))
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }

    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.get_original_tweet_id()
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        CachedCounts {
            fav_count: hydrated.fav_count,
            reply_count: hydrated.reply_count,
            repost_count: hydrated.repost_count,
            quote_count: hydrated.quote_count,
            view_count: hydrated.view_count,
            bookmark_count: hydrated.bookmark_count,
        }
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        value.to_candidate()
    }

    async fn hydrate_from_client(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let follower_cap = query.params.get(ColdStartFollowerCap);
        let follower_decay_width = query.params.get(ColdStartFollowerDecayWidth);
        let fetch_counts = |c: &PostCandidate| {
            !query.has_cached_posts
                || cold_start_base_eligible(c, follower_cap, follower_decay_width)
        };

        let mut unique_ids: Vec<u64> = candidates
            .iter()
            .filter(|c| fetch_counts(c))
            .map(|c| c.get_original_tweet_id())
            .collect();
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

        candidates
            .iter()
            .map(|c| {
                if !fetch_counts(c) {
                    return Ok(preserve_counts(c));
                }
                match counts.get(&c.get_original_tweet_id()) {
                    Some(proto) => Ok(CachedCounts::from_proto(proto).to_candidate()),
                    None => Ok(CachedCounts::default().to_candidate()),
                }
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.fav_count = hydrated.fav_count;
        candidate.reply_count = hydrated.reply_count;
        candidate.repost_count = hydrated.repost_count;
        candidate.quote_count = hydrated.quote_count;
        candidate.view_count = hydrated.view_count;
        candidate.bookmark_count = hydrated.bookmark_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::engagement_counts_client::MockEngagementCountsClient;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const COUNTS: &str = "rust_home_mixer_enable_engagement_counts_hydration";
    const COLD_START: &str = "rust_home_mixer_enable_viewer_cold_start_boost";
    const CAP: &str = "rust_home_mixer_cold_start_follower_cap";
    const FOLLOWER_DECAY_WIDTH: &str = "rust_home_mixer_cold_start_follower_decay_width";

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
        assert!(h.enable(&query(false, &[(COLD_START, "true"), (COUNTS, "false")])));
        assert!(h.enable(&query(true, &[(COLD_START, "true")])));
        assert!(!h.enable(&query(true, &[(COUNTS, "true"), (COLD_START, "false")])));
        assert!(!h.enable(&query(false, &[(COUNTS, "false"), (COLD_START, "false")])));
        assert!(!h.enable(&query(true, &[(COUNTS, "false"), (COLD_START, "false")])));
    }

    #[tokio::test]
    async fn no_cached_posts_hydrates_all() {
        let h = hydrator(view_counts(&[(20, 7)])).await;
        let candidates = vec![PostCandidate {
            tweet_id: 20,
            author_id: 2,
            author_followers_count: Some(5000),
            view_count: Some(999),
            ..Default::default()
        }];
        let q = query(false, &[(COUNTS, "true"), (CAP, "1000")]);
        let result = h.hydrate_from_client(&q, &candidates).await;
        assert_eq!(result[0].as_ref().unwrap().view_count, Some(7));
    }

    #[tokio::test]
    async fn author_cold_start_on_cached_posts_refreshes_eligible_and_preserves_ineligible() {
        let h = hydrator(view_counts(&[(10, 5)])).await;
        let candidates = vec![
            PostCandidate {
                tweet_id: 10,
                author_id: 1,
                author_followers_count: Some(100),
                view_count: Some(999),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 20,
                author_id: 2,
                author_followers_count: Some(5000),
                view_count: Some(999),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 30,
                author_id: 3,
                author_followers_count: Some(100),
                in_reply_to_tweet_id: Some(1),
                view_count: Some(888),
                ..Default::default()
            },
        ];
        let q = query(
            true,
            &[(COLD_START, "true"), (COUNTS, "true"), (CAP, "1000")],
        );
        let result = h.hydrate_from_client(&q, &candidates).await;
        assert_eq!(result[0].as_ref().unwrap().view_count, Some(5));
        assert_eq!(result[1].as_ref().unwrap().view_count, Some(999));
        assert_eq!(result[2].as_ref().unwrap().view_count, Some(888));
    }

    #[tokio::test]
    async fn cached_posts_refresh_counts_inside_follower_decay_band() {
        let h = hydrator(view_counts(&[(10, 5)])).await;
        let candidates = vec![PostCandidate {
            tweet_id: 10,
            author_id: 1,
            author_followers_count: Some(1100),
            view_count: Some(999),
            ..Default::default()
        }];
        let q = query(
            true,
            &[
                (COLD_START, "true"),
                (CAP, "1000"),
                (FOLLOWER_DECAY_WIDTH, "250"),
            ],
        );

        let result = h.hydrate_from_client(&q, &candidates).await;

        assert_eq!(result[0].as_ref().unwrap().view_count, Some(5));
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
    async fn cached_cold_start_serves_repeat_from_moka_cache() {
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
        let q = query(true, &[(COLD_START, "true"), (CAP, "1000")]);

        let first = xai_candidate_pipeline::hydrator::Hydrator::hydrate(&h, &q, &candidates).await;
        let second = xai_candidate_pipeline::hydrator::Hydrator::hydrate(&h, &q, &candidates).await;

        assert_eq!(first[0].as_ref().unwrap().view_count, Some(5));
        assert_eq!(second[0].as_ref().unwrap().view_count, Some(5));
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
    }
}
