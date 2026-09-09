use crate::candidate_hydrators::vf_candidate_hydrator::{primary_vf_id, should_drop_ancillary};
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableXaiVfClient;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_twittercontext_proto::GetTwitterContextViewer;
use xai_visibility_filtering::models::FilteredReason;
use xai_visibility_filtering::vf_client::SafetyLevel::TimelineHome;
use xai_visibility_filtering::vf_client::VfClient;

pub struct VFFollowingCandidateHydrator {
    pub strato_vf_client: Arc<dyn VfClient + Send + Sync>,
    pub xai_vf_client: Arc<dyn VfClient + Send + Sync>,
}

impl VFFollowingCandidateHydrator {
    pub fn new(
        strato_vf_client: Arc<dyn VfClient + Send + Sync>,
        xai_vf_client: Arc<dyn VfClient + Send + Sync>,
    ) -> Self {
        Self {
            strato_vf_client,
            xai_vf_client,
        }
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for VFFollowingCandidateHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let context = query.get_viewer();
        let client = if query.params.get(EnableXaiVfClient) {
            &self.xai_vf_client
        } else {
            &self.strato_vf_client
        };

        let mut post_ids: Vec<u64> = Vec::new();
        for candidate in candidates {
            post_ids.push(candidate.tweet_id);
            post_ids.extend(candidate.ancestors.iter().copied());
            if let Some(quoted_post_id) = candidate.quoted_tweet_id {
                post_ids.push(quoted_post_id);
            }
            if let Some(reposted_post_id) = candidate.retweeted_tweet_id {
                post_ids.push(reposted_post_id);
            }
        }
        post_ids.sort_unstable();
        post_ids.dedup();

        let all_results: HashMap<u64, Result<Option<FilteredReason>>> = if post_ids.is_empty() {
            HashMap::new()
        } else {
            client
                .get_result(post_ids, TimelineHome, query.user_id, context)
                .await
                .into_iter()
                .map(|(id, r)| (id, r.map(|t| t.reason)))
                .collect()
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use xai_safety_label_store::types::SafetyLabelMap;
    use xai_twittercontext_proto::TwitterContextViewer;
    use xai_visibility_filtering::models::{Action, SafetyResult, SafetyResultReason};
    use xai_visibility_filtering::vf_client::{SafetyLevel, TweetVisibility};

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

    fn hydrator(results: HashMap<u64, FilteredReason>) -> VFFollowingCandidateHydrator {
        let client = Arc::new(MapClient { results });
        VFFollowingCandidateHydrator::new(client.clone(), client)
    }

    #[tokio::test]
    async fn native_post_still_uses_its_own_id() {
        let results = hydrator(HashMap::from([(20, interstitial())]))
            .hydrate(
                &ScoredPostsQuery::default(),
                &[PostCandidate {
                    tweet_id: 20,
                    ..Default::default()
                }],
            )
            .await;
        let hydrated = results[0].as_ref().unwrap();
        assert!(matches!(
            hydrated.visibility_reason,
            Some(FilteredReason::SafetyResult(ref s)) if s.action == Action::Interstitial
        ));
    }

    #[tokio::test]
    async fn retweet_uses_original_interstitial() {
        let results = hydrator(HashMap::from([(10, allow()), (20, interstitial())]))
            .hydrate(
                &ScoredPostsQuery::default(),
                &[PostCandidate {
                    tweet_id: 10,
                    retweeted_tweet_id: Some(20),
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
}
