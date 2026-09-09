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
        // A post can be both an in-network primary and another candidate's ancestor or quote.
        // Keep the safety-level results separate until the candidate role is known.
        let timeline_home_results: HashMap<u64, Result<Option<FilteredReason>>> = in_network_result
            .into_iter()
            .map(|(id, result)| (id, result.map(|visibility| visibility.reason)))
            .collect();
        let recommendations_results: HashMap<u64, Result<Option<FilteredReason>>> = oon_result
            .into_iter()
            .map(|(id, result)| (id, result.map(|visibility| visibility.reason)))
            .collect();

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let primary_result = if candidate.in_network.unwrap_or(false) {
                timeline_home_results.get(&candidate.tweet_id)
            } else {
                recommendations_results.get(&candidate.tweet_id)
            };
            let visibility_reason = match primary_result {
                Some(Ok(Some(reason))) => Some(reason.clone()),
                _ => None,
            };

            let drop_ancillary = should_drop_ancillary_by_safety_level(
                candidate,
                &timeline_home_results,
                &recommendations_results,
            );

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

pub(crate) fn should_drop_ancillary(
    candidate: &PostCandidate,
    vf_results: &HashMap<u64, Result<Option<FilteredReason>>>,
) -> bool {
    should_drop_ancillary_by_safety_level(candidate, vf_results, vf_results)
}

fn should_drop_ancillary_by_safety_level(
    candidate: &PostCandidate,
    timeline_home_results: &HashMap<u64, Result<Option<FilteredReason>>>,
    recommendations_results: &HashMap<u64, Result<Option<FilteredReason>>>,
) -> bool {
    for &ancestor_id in &candidate.ancestors {
        if candidate.tombstone_ancestor_ids.contains(&ancestor_id) {
            continue;
        }
        if let Some(Ok(Some(reason))) = recommendations_results.get(&ancestor_id)
            && should_drop_reason(reason)
        {
            return true;
        }
    }

    if let Some(quoted_id) = candidate.quoted_tweet_id
        && let Some(Ok(Some(reason))) = recommendations_results.get(&quoted_id)
        && should_drop_reason(reason)
    {
        return true;
    }

    if let Some(retweeted_id) = candidate.retweeted_tweet_id
        && let Some(Ok(Some(reason))) = timeline_home_results.get(&retweeted_id)
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
    use xai_visibility_filtering::models::SafetyResult;
    use xai_visibility_filtering::tweet_safety_label::SafetyLabelFailure;

    struct LevelAwareVfClient {
        timeline_home: HashMap<u64, FilteredReason>,
        recommendations: HashMap<u64, FilteredReason>,
    }

    #[async_trait]
    impl VfClient for LevelAwareVfClient {
        async fn get_result(
            &self,
            tweet_ids: Vec<u64>,
            safety_level: SafetyLevel,
            _for_user_id: u64,
            _context: Option<TwitterContextViewer>,
        ) -> HashMap<u64, Result<TweetVisibility>> {
            let reasons = match safety_level {
                TimelineHome => &self.timeline_home,
                TimelineHomeRecommendations => &self.recommendations,
                _ => return HashMap::new(),
            };

            tweet_ids
                .into_iter()
                .filter_map(|tweet_id| {
                    reasons.get(&tweet_id).cloned().map(|reason| {
                        (
                            tweet_id,
                            Ok(TweetVisibility {
                                reason: Some(reason),
                                safety_labels: Err(SafetyLabelFailure::LookupFailed),
                            }),
                        )
                    })
                })
                .collect()
        }
    }

    fn safety_reason(action: Action) -> FilteredReason {
        FilteredReason::SafetyResult(SafetyResult {
            action,
            ..Default::default()
        })
    }

    fn hydrator(client: LevelAwareVfClient) -> VFCandidateHydrator {
        let client: Arc<dyn VfClient + Send + Sync> = Arc::new(client);
        VFCandidateHydrator {
            strato_vf_client: Arc::clone(&client),
            xai_vf_client: client,
        }
    }

    #[test]
    fn ancillary_roles_use_their_assigned_safety_level() {
        let timeline_home_results: HashMap<u64, Result<Option<FilteredReason>>> = HashMap::from([
            (1, Ok(Some(safety_reason(Action::Allow)))),
            (2, Ok(Some(safety_reason(Action::Allow)))),
            (3, Ok(Some(safety_reason(Action::Drop(Default::default()))))),
        ]);
        let recommendations_results: HashMap<u64, Result<Option<FilteredReason>>> =
            HashMap::from([
                (1, Ok(Some(safety_reason(Action::Drop(Default::default()))))),
                (2, Ok(Some(safety_reason(Action::Drop(Default::default()))))),
                (3, Ok(Some(safety_reason(Action::Allow)))),
            ]);
        let candidates = [
            (
                "ancestor",
                PostCandidate {
                    ancestors: vec![1],
                    ..Default::default()
                },
            ),
            (
                "quoted post",
                PostCandidate {
                    quoted_tweet_id: Some(2),
                    ..Default::default()
                },
            ),
            (
                "retweeted post",
                PostCandidate {
                    retweeted_tweet_id: Some(3),
                    ..Default::default()
                },
            ),
        ];

        for (role, candidate) in candidates {
            assert!(
                should_drop_ancillary_by_safety_level(
                    &candidate,
                    &timeline_home_results,
                    &recommendations_results,
                ),
                "{role} must use its assigned safety level"
            );
        }
    }

    #[tokio::test]
    async fn preserves_distinct_primary_and_quoted_post_safety_levels() {
        let primary_id = 10;
        let quote_id = 20;
        let timeline_home_reason = safety_reason(Action::Interstitial);
        let recommendations_reason = safety_reason(Action::Drop(Default::default()));
        let quote_reason = safety_reason(Action::Allow);
        let hydrator = hydrator(LevelAwareVfClient {
            timeline_home: HashMap::from([(primary_id, timeline_home_reason.clone())]),
            recommendations: HashMap::from([
                (primary_id, recommendations_reason),
                (quote_id, quote_reason.clone()),
            ]),
        });
        let candidates = vec![
            PostCandidate {
                tweet_id: primary_id,
                in_network: Some(true),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: quote_id,
                in_network: Some(false),
                quoted_tweet_id: Some(primary_id),
                ..Default::default()
            },
        ];

        let hydrated: Vec<PostCandidate> = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await
            .into_iter()
            .map(|result| result.expect("VF hydration should succeed"))
            .collect();

        assert_eq!(
            hydrated[0].visibility_reason,
            Some(timeline_home_reason),
            "the in-network primary must keep its TimelineHome verdict"
        );
        assert_eq!(hydrated[0].drop_ancillary_posts, Some(false));
        assert_eq!(hydrated[1].visibility_reason, Some(quote_reason));
        assert_eq!(
            hydrated[1].drop_ancillary_posts,
            Some(true),
            "the quoted-post check must keep its TimelineHomeRecommendations verdict"
        );
    }
}
