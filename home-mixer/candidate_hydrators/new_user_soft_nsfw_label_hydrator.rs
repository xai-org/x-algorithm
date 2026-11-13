use crate::filters::new_user_soft_nsfw_filter::new_user_soft_nsfw_filter_enabled;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;
use xai_visibility_filtering::tweet_safety_label::TweetSafetyLabelClient;
use xai_x_thrift::tweet_safety_label::SafetyLabelType;

pub struct NewUserSoftNsfwLabelHydrator {
    pub client: Arc<dyn TweetSafetyLabelClient>,
}

fn followed_or_self(query: &ScoredPostsQuery) -> HashSet<u64> {
    query
        .user_features
        .followed_user_ids
        .iter()
        .map(|&id| id as u64)
        .chain(std::iter::once(query.user_id))
        .collect()
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for NewUserSoftNsfwLabelHydrator {
    type Fetched = Vec<Option<bool>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        new_user_soft_nsfw_filter_enabled(query)
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<bool>>, String> {
        let in_network_authors = followed_or_self(query);
        let needs_label =
            |c: &PostCandidate| c.soft_nsfw.is_none() && !in_network_authors.contains(&c.author_id);
        let tweet_ids: Vec<u64> = candidates
            .iter()
            .filter(|c| needs_label(c))
            .map(|c| c.tweet_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if tweet_ids.is_empty() {
            return Ok(vec![None; candidates.len()]);
        }
        let batch = self
            .client
            .get_safety_labels(tweet_ids)
            .await
            .map_err(|e| format!("new user SoftNsfw label lookup failed: {e}"))?;

        Ok(candidates
            .iter()
            .map(|c| {
                if !needs_label(c) || batch.failures.contains_key(&c.tweet_id) {
                    return None;
                }
                Some(
                    batch
                        .labels
                        .get(&c.tweet_id)
                        .is_some_and(|labels| labels.contains_key(&SafetyLabelType::SOFT_NSFW)),
                )
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        soft_nsfw: Vec<Option<bool>>,
    ) {
        for (candidate, soft_nsfw) in candidates.iter_mut().zip(soft_nsfw) {
            if soft_nsfw.is_some() {
                candidate.soft_nsfw = soft_nsfw;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.soft_nsfw = hydrated.soft_nsfw;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::new_user_soft_nsfw_filter::tests::{
        candidate, enabled_query, new_user_id, old_user_id, query,
    };
    use crate::filters::new_user_soft_nsfw_filter::NewUserSoftNsfwFilter;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tonic::Status;
    use xai_candidate_pipeline::filter::Filter;
    use xai_safety_label_store::types::SafetyLabelMap;
    use xai_visibility_filtering::tweet_safety_label::{SafetyLabelFailure, SafetyLabelsBatch};
    use xai_x_thrift::tweet_safety_label::SafetyLabel;

    struct FakeVfClient {
        result: Result<SafetyLabelsBatch, Status>,
        requested: Mutex<Vec<u64>>,
    }

    impl FakeVfClient {
        fn new(result: Result<SafetyLabelsBatch, Status>) -> Arc<Self> {
            Arc::new(Self {
                result,
                requested: Mutex::new(Vec::new()),
            })
        }

        fn requested(&self) -> Vec<u64> {
            let mut ids = self.requested.lock().unwrap().clone();
            ids.sort();
            ids
        }
    }

    #[async_trait]
    impl TweetSafetyLabelClient for FakeVfClient {
        async fn get_safety_labels(
            &self,
            tweet_ids: Vec<u64>,
        ) -> Result<SafetyLabelsBatch, Status> {
            self.requested.lock().unwrap().extend(tweet_ids);
            self.result.clone()
        }
    }

    fn labels(types: &[SafetyLabelType]) -> SafetyLabelMap {
        types.iter().map(|t| (*t, SafetyLabel::default())).collect()
    }

    fn soft_nsfw_batch(soft_nsfw_ids: &[u64]) -> SafetyLabelsBatch {
        SafetyLabelsBatch {
            labels: soft_nsfw_ids
                .iter()
                .map(|&id| (id, labels(&[SafetyLabelType::SOFT_NSFW])))
                .collect(),
            failures: HashMap::new(),
        }
    }

    fn hydrator(client: Arc<FakeVfClient>) -> NewUserSoftNsfwLabelHydrator {
        NewUserSoftNsfwLabelHydrator { client }
    }

    async fn hydrate_then_filter(
        client: Arc<FakeVfClient>,
        query: &ScoredPostsQuery,
        mut candidates: Vec<PostCandidate>,
    ) -> Vec<u64> {
        let h = hydrator(client);
        if let Ok(fetched) = h.fetch(query, &candidates).await {
            h.apply(query, &mut candidates, fetched);
        }
        NewUserSoftNsfwFilter
            .filter(query, candidates)
            .kept
            .iter()
            .map(|c| c.tweet_id)
            .collect()
    }

    #[tokio::test]
    async fn out_of_network_soft_nsfw_post_is_removed() {
        let kept = hydrate_then_filter(
            FakeVfClient::new(Ok(soft_nsfw_batch(&[1]))),
            &enabled_query(),
            vec![
                candidate(1, Some(false), None),
                candidate(2, Some(false), None),
            ],
        )
        .await;
        assert_eq!(kept, vec![2]);
    }

    #[tokio::test]
    async fn in_network_post_is_kept_and_not_looked_up() {
        let followed = 42;
        let mut query = enabled_query();
        query.user_features.followed_user_ids = vec![followed as i64];
        let in_network = PostCandidate {
            author_id: followed,
            ..candidate(1, Some(true), None)
        };
        let client = FakeVfClient::new(Ok(soft_nsfw_batch(&[1])));
        let kept = hydrate_then_filter(client.clone(), &query, vec![in_network]).await;
        assert_eq!(kept, vec![1]);
        assert!(client.requested().is_empty());
    }

    #[tokio::test]
    async fn lookup_failure_keeps_the_post() {
        let failed_batch = SafetyLabelsBatch {
            labels: HashMap::new(),
            failures: HashMap::from([(1, SafetyLabelFailure::BatchFailed)]),
        };
        let kept = hydrate_then_filter(
            FakeVfClient::new(Ok(failed_batch)),
            &enabled_query(),
            vec![candidate(1, Some(false), None)],
        )
        .await;
        assert_eq!(kept, vec![1]);
    }

    #[tokio::test]
    async fn failed_call_keeps_every_post() {
        let kept = hydrate_then_filter(
            FakeVfClient::new(Err(Status::unavailable("down"))),
            &enabled_query(),
            vec![
                candidate(1, Some(false), None),
                candidate(2, Some(false), None),
            ],
        )
        .await;
        assert_eq!(kept, vec![1, 2]);
    }

    #[tokio::test]
    async fn cached_candidates_are_labeled_only_when_unlabeled() {
        let mut cached = enabled_query();
        cached.has_cached_posts = true;
        let h = hydrator(FakeVfClient::new(Ok(soft_nsfw_batch(&[]))));
        assert!(h.enable(&cached));

        let client = FakeVfClient::new(Ok(soft_nsfw_batch(&[2])));
        let kept = hydrate_then_filter(
            client.clone(),
            &cached,
            vec![
                candidate(1, Some(false), Some(true)),
                candidate(2, Some(false), None),
                candidate(3, Some(false), Some(false)),
            ],
        )
        .await;
        assert_eq!(kept, vec![3]);
        assert_eq!(client.requested(), vec![2]);
    }

    #[tokio::test]
    async fn skips_the_call_when_nothing_needs_a_label() {
        let client = FakeVfClient::new(Err(Status::unavailable("unused")));
        let h = hydrator(client.clone());
        let query = enabled_query();
        let mut candidates = vec![candidate(1, Some(false), Some(false))];
        let fetched = h.fetch(&query, &candidates).await.unwrap();
        h.apply(&query, &mut candidates, fetched);
        assert_eq!(candidates[0].soft_nsfw, Some(false));
        assert!(client.requested().is_empty());
    }

    #[test]
    fn gated_like_the_filter() {
        let h = hydrator(FakeVfClient::new(Ok(soft_nsfw_batch(&[]))));
        assert!(h.enable(&query(new_user_id(), true, 86_400)));
        assert!(!h.enable(&query(new_user_id(), false, 86_400)));
        assert!(!h.enable(&query(old_user_id(), true, 86_400)));
    }
}
