use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};

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
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for SubscriptionHydrator {
    type CacheKey = u64;
    type CacheValue = Option<u64>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }

    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.get_original_tweet_id()
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        hydrated.subscription_author_id
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            subscription_author_id: value,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let client = &self.tes_client;

        let tweet_ids: Vec<u64> = candidates
            .iter()
            .map(|c| c.get_original_tweet_id())
            .collect();

        let post_features = client.get_subscription_author_ids(tweet_ids.clone()).await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for tweet_id in tweet_ids {
            let post_features = post_features.get(&tweet_id);
            let hydrated = match post_features {
                Some(Ok(value)) => Ok(PostCandidate {
                    subscription_author_id: *value,
                    ..Default::default()
                }),
                None => Err(format!(
                    "Missing subscription author id for tweet_id={}",
                    tweet_id
                )),
                Some(Err(err)) => Err(err.to_string()),
            };
            hydrated_candidates.push(hydrated);
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.subscription_author_id = hydrated.subscription_author_id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::tweet_entity_service_client::MockTESClient;
    use crate::filters::ineligible_subscription_filter::IneligibleSubscriptionFilter;
    use std::collections::HashMap;
    use xai_candidate_pipeline::filter::Filter;
    use xai_candidate_pipeline::hydrator::Hydrator;

    fn tes(exclusive_original: u64, author: u64) -> Arc<dyn TESClient + Send + Sync> {
        let mut subscription_author_ids = HashMap::new();
        subscription_author_ids.insert(exclusive_original, Some(author));
        Arc::new(MockTESClient {
            subscription_author_ids,
            ..Default::default()
        })
    }

    async fn hydrate(
        client: Arc<dyn TESClient + Send + Sync>,
        candidates: &[PostCandidate],
    ) -> Vec<PostCandidate> {
        let hydrator = SubscriptionHydrator::new(client).await;
        let mut out = candidates.to_vec();
        let hydrated = hydrator
            .hydrate(&ScoredPostsQuery::default(), candidates)
            .await;
        hydrator.update_all(&mut out, hydrated);
        out
    }

    #[tokio::test]
    async fn native_exclusive_post_still_reads_wrapper_id() {
        let out = hydrate(
            tes(100, 7),
            &[PostCandidate {
                tweet_id: 100,
                ..Default::default()
            }],
        )
        .await;
        assert_eq!(out[0].subscription_author_id, Some(7));
    }

    #[tokio::test]
    async fn in_network_retweet_reads_original_exclusive_author() {
        let out = hydrate(
            tes(100, 7),
            &[PostCandidate {
                tweet_id: 200,
                retweeted_tweet_id: Some(100),
                in_network: Some(true),
                ..Default::default()
            }],
        )
        .await;
        assert_eq!(out[0].subscription_author_id, Some(7));
    }

    #[tokio::test]
    async fn public_post_stays_none() {
        let mut subscription_author_ids = HashMap::new();
        subscription_author_ids.insert(300u64, None);
        let client = Arc::new(MockTESClient {
            subscription_author_ids,
            ..Default::default()
        });
        let out = hydrate(
            client,
            &[PostCandidate {
                tweet_id: 300,
                ..Default::default()
            }],
        )
        .await;
        assert_eq!(out[0].subscription_author_id, None);
    }

    #[tokio::test]
    async fn non_subscriber_loses_retweet_of_exclusive_original() {
        let out = hydrate(
            tes(100, 7),
            &[PostCandidate {
                tweet_id: 200,
                retweeted_tweet_id: Some(100),
                in_network: Some(true),
                ..Default::default()
            }],
        )
        .await;
        let mut query = ScoredPostsQuery::default();
        query.user_features.subscribed_user_ids = vec![99];
        let result = IneligibleSubscriptionFilter.filter(&query, out);
        assert_eq!(result.kept.len(), 0);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 200);
    }

    #[tokio::test]
    async fn subscriber_keeps_retweet_of_exclusive_original() {
        let out = hydrate(
            tes(100, 7),
            &[PostCandidate {
                tweet_id: 200,
                retweeted_tweet_id: Some(100),
                in_network: Some(true),
                ..Default::default()
            }],
        )
        .await;
        let mut query = ScoredPostsQuery::default();
        query.user_features.subscribed_user_ids = vec![7];
        let result = IneligibleSubscriptionFilter.filter(&query, out);
        assert_eq!(result.removed.len(), 0);
        assert_eq!(result.kept.len(), 1);
    }
}
