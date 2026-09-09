use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableXaiVfClient;
use anyhow::Result;
use futures::future::join;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_twittercontext_proto::GetTwitterContextViewer;
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering::models::{Action, FilteredReason};
use xai_visibility_filtering::vf_client::SafetyLevel;
use xai_visibility_filtering::vf_client::SafetyLevel::{TimelineHome, TimelineHomeRecommendations};
use xai_visibility_filtering::vf_client::{TweetVisibility, VfClient};

pub struct VFCandidateHydrator {
    pub strato_vf_client: Arc<dyn VfClient + Send + Sync>,
    pub xai_vf_client: Arc<dyn VfClient + Send + Sync>,
}

impl VFCandidateHydrator {
    pub async fn new(
        strato_vf_client: Arc<dyn VfClient + Send + Sync>,
        xai_vf_client: Arc<dyn VfClient + Send + Sync>,
    ) -> Self {
        Self {
            strato_vf_client,
            xai_vf_client,
        }
    }

    async fn fetch_vf_results(
        client: &Arc<dyn VfClient + Send + Sync>,
        tweet_ids: Vec<u64>,
        safety_level: SafetyLevel,
        for_user_id: u64,
        context: Option<TwitterContextViewer>,
    ) -> HashMap<u64, Result<TweetVisibility>> {
        if tweet_ids.is_empty() {
            return HashMap::new();
        }

        client
            .get_result(tweet_ids, safety_level, for_user_id, context)
            .await
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for VFCandidateHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let context = query.get_viewer();
        let user_id = query.user_id;
        // Fully migrated to Rust VF. Old VF available in the event of production issues.
        let client = if query.params.get(EnableXaiVfClient) {
            &self.xai_vf_client
        } else {
            &self.strato_vf_client
        };

        let mut in_network_ids: Vec<u64> = Vec::new();
        let mut oon_ids: Vec<u64> = Vec::new();

        for candidate in candidates.iter() {
            if candidate.in_network.unwrap_or(false) {
                in_network_ids.push(candidate.tweet_id);
            } else {
                oon_ids.push(candidate.tweet_id);
            }
            for &ancestor_id in &candidate.ancestors {
                oon_ids.push(ancestor_id);
            }
            if let Some(quoted_id) = candidate.quoted_tweet_id {
                oon_ids.push(quoted_id);
            }
            if let Some(retweeted_id) = candidate.retweeted_tweet_id {
                in_network_ids.push(retweeted_id);
            }
        }

        in_network_ids.sort_unstable();
        in_network_ids.dedup();
        oon_ids.sort_unstable();
        oon_ids.dedup();

        let in_network_future = Self::fetch_vf_results(
            client,
            in_network_ids,
            TimelineHome,
            user_id,
            context.clone(),
        );

        let oon_future = Self::fetch_vf_results(
            client,
            oon_ids,
            TimelineHomeRecommendations,
            user_id,
            context,
        );

        let (in_network_result, oon_result) = join(in_network_future, oon_future).await;
        let mut all_results: HashMap<u64, Result<Option<FilteredReason>>> = HashMap::new();
        all_results.extend(
            oon_result
                .into_iter()
                .chain(in_network_result)
                .map(|(id, r)| (id, r.map(|t| t.reason))),
        );

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let primary_result = all_results.get(&primary_vf_id(candidate));
            let visibility_reason = match primary_result {
                Some(Ok(Some(reason))) => Some(reason.clone()),
                _ => None,
            };

            let drop_ancillary = should_drop_ancillary(candidate, &all_results);

            let hydrated = match primary_result {
                Some(Err(err)) => Err(err.to_string()),
                _ => Ok(PostCandidate {
                    visibility_reason,
                    drop_ancillary_posts: Some(drop_ancillary),
                    ..Default::default()
                }),
            };
            hydrated_candidates.push(hydrated);
        }
        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.visibility_reason = hydrated.visibility_reason;
        candidate.drop_ancillary_posts = hydrated.drop_ancillary_posts;
    }
}

pub(crate) fn primary_vf_id(candidate: &PostCandidate) -> u64 {
    candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id)
}

pub(crate) fn should_drop_ancillary(
    candidate: &PostCandidate,
    vf_results: &HashMap<u64, Result<Option<FilteredReason>>>,
) -> bool {
    for &ancestor_id in &candidate.ancestors {
        if candidate.tombstone_ancestor_ids.contains(&ancestor_id) {
            continue;
        }
        if let Some(Ok(Some(reason))) = vf_results.get(&ancestor_id)
            && should_drop_reason(reason)
        {
            return true;
        }
    }

    if let Some(quoted_id) = candidate.quoted_tweet_id
        && let Some(Ok(Some(reason))) = vf_results.get(&quoted_id)
        && should_drop_reason(reason)
    {
        return true;
    }

    if let Some(retweeted_id) = candidate.retweeted_tweet_id
        && let Some(Ok(Some(reason))) = vf_results.get(&retweeted_id)
        && should_drop_reason(reason)
    {
        return true;
    }

    false
}

fn should_drop_reason(reason: &FilteredReason) -> bool {
    match reason {
        FilteredReason::SafetyResult(safety_result) => {
            matches!(safety_result.action, Action::Drop(_))
        }
        _ => true, 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_safety_label_store::types::SafetyLabelMap;
    use xai_visibility_filtering::models::{
        Action, DropReason, SafetyResult, SafetyResultReason,
    };

    struct MapClient {
        results: HashMap<u64, FilteredReason>,
    }

    fn vis(reason: FilteredReason) -> TweetVisibility {
        TweetVisibility {
            reason: Some(reason),
            safety_labels: Ok(SafetyLabelMap::default()),
        }
    }

    fn interstitial() -> FilteredReason {
        FilteredReason::SafetyResult(SafetyResult {
            reason: Some(SafetyResultReason::NsfwHighPrecision),
            action: Action::Interstitial,
        })
    }

    fn allow() -> FilteredReason {
        FilteredReason::SafetyResult(SafetyResult {
            reason: None,
            action: Action::Allow,
        })
    }

    fn drop_reason() -> FilteredReason {
        FilteredReason::SafetyResult(SafetyResult {
            reason: Some(SafetyResultReason::NsfwHighPrecision),
            action: Action::Drop(DropReason {}),
        })
    }

    #[async_trait]
    impl VfClient for MapClient {
        async fn get_result(
            &self,
            post_ids: Vec<u64>,
            _safety_level: SafetyLevel,
            _for_user_id: u64,
            _context: Option<TwitterContextViewer>,
        ) -> HashMap<u64, Result<TweetVisibility>> {
            post_ids
                .into_iter()
                .filter_map(|id| {
                    self.results
                        .get(&id)
                        .cloned()
                        .map(|reason| (id, Ok(vis(reason))))
                })
                .collect()
        }
    }

    async fn hydrate(
        results: HashMap<u64, FilteredReason>,
        candidates: &[PostCandidate],
    ) -> Vec<std::result::Result<PostCandidate, String>> {
        let client = Arc::new(MapClient { results });
        VFCandidateHydrator::new(client.clone(), client)
            .await
            .hydrate(&ScoredPostsQuery::default(), candidates)
            .await
    }

    #[tokio::test]
    async fn native_post_still_uses_its_own_id() {
        let results = hydrate(
            HashMap::from([(20, interstitial())]),
            &[PostCandidate {
                tweet_id: 20,
                in_network: Some(true),
                ..Default::default()
            }],
        )
        .await;
        let hydrated = results[0].as_ref().unwrap();
        assert!(matches!(
            hydrated.visibility_reason,
            Some(FilteredReason::SafetyResult(ref s)) if s.action == Action::Interstitial
        ));
        assert_eq!(hydrated.drop_ancillary_posts, Some(false));
    }

    #[tokio::test]
    async fn retweet_uses_original_interstitial() {
        let results = hydrate(
            HashMap::from([(10, allow()), (20, interstitial())]),
            &[PostCandidate {
                tweet_id: 10,
                retweeted_tweet_id: Some(20),
                in_network: Some(true),
                ..Default::default()
            }],
        )
        .await;
        let hydrated = results[0].as_ref().unwrap();
        assert!(matches!(
            hydrated.visibility_reason,
            Some(FilteredReason::SafetyResult(ref s)) if s.action == Action::Interstitial
        ));
        assert_eq!(hydrated.drop_ancillary_posts, Some(false));
    }

    #[tokio::test]
    async fn wrapper_interstitial_is_not_the_primary() {
        let results = hydrate(
            HashMap::from([(10, interstitial())]),
            &[PostCandidate {
                tweet_id: 10,
                retweeted_tweet_id: Some(20),
                in_network: Some(true),
                ..Default::default()
            }],
        )
        .await;
        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(hydrated.visibility_reason, None);
    }

    #[tokio::test]
    async fn original_drop_still_ancillary_drops() {
        let results = hydrate(
            HashMap::from([(10, allow()), (20, drop_reason())]),
            &[PostCandidate {
                tweet_id: 10,
                retweeted_tweet_id: Some(20),
                in_network: Some(true),
                ..Default::default()
            }],
        )
        .await;
        let hydrated = results[0].as_ref().unwrap();
        assert!(matches!(
            hydrated.visibility_reason,
            Some(FilteredReason::SafetyResult(ref s)) if matches!(s.action, Action::Drop(_))
        ));
        assert_eq!(hydrated.drop_ancillary_posts, Some(true));
    }
}
