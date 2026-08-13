use crate::hydration::{HydrationOutput, HydrationPipeline, HydrationRequest};
use crate::models::{RawCandidate, TweetId, VfAction};
use crate::rules::metrics as ft_metrics;
use crate::rules::{Policies, SafetyLevel, Verdict};
use std::collections::HashMap;
use tracing::{debug, info};
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterRequest {
    pub viewer_id: Option<u64>,
    pub country_code: Option<String>,
    pub safety_level: SafetyLevel,
    pub candidates: Vec<RawCandidate>,
}

pub struct FilterOutcome {
    pub tweet_id: TweetId,
    pub verdict: Verdict,
    pub safety_labels: Option<vf_pb::SafetyLabelMap>,
}

pub struct FilterSummary {
    pub tweet_count: usize,
    pub drop_count: usize,
    pub unresolved_author_count: usize,
}

pub struct FilterResponse {
    pub outcomes: Vec<FilterOutcome>,
    pub summary: FilterSummary,
}

pub struct FilterTweets {
    hydration_pipeline: HydrationPipeline,
    policies: Policies,
}

impl FilterTweets {
    pub(crate) fn new(hydration_pipeline: HydrationPipeline, policies: Policies) -> Self {
        Self {
            hydration_pipeline,
            policies,
        }
    }

    pub async fn run(&self, request: FilterRequest) -> FilterResponse {
        let hydration = self
            .hydration_pipeline
            .hydrate(HydrationRequest::new(
                request.viewer_id,
                request.country_code,
                &request.candidates,
                request.safety_level,
            ))
            .await;
        let HydrationOutput {
            viewer_features,
            candidates: hydrated_candidates,
            safety_labels,
        } = hydration;
        let unresolved_author_count = request.candidates.len() - hydrated_candidates.len();
        if unresolved_author_count > 0 {
            info!(
                count = unresolved_author_count,
                "Tweets with unresolved author_id"
            );
        }

        let evaluated: HashMap<TweetId, Verdict> = hydrated_candidates
            .iter()
            .map(|candidate| {
                (
                    TweetId(candidate.tweet_id),
                    self.policies
                        .evaluate(request.safety_level, &viewer_features, candidate),
                )
            })
            .collect();

        let outcomes: Vec<FilterOutcome> = request
            .candidates
            .iter()
            .map(|candidate| {
                let verdict = evaluated
                    .get(&candidate.tweet_id)
                    .cloned()
                    .unwrap_or_else(Verdict::unresolved_author);
                debug!(
                    viewer_id = request.viewer_id,
                    tweet_id = candidate.tweet_id.0,
                    action = ?verdict.action,
                    rule = ?verdict.decided_by,
                    safety_level = ?request.safety_level,
                    "VF verdict"
                );
                FilterOutcome {
                    tweet_id: candidate.tweet_id,
                    verdict,
                    safety_labels: safety_labels.get(&candidate.tweet_id).cloned(),
                }
            })
            .collect();

        ft_metrics::record_verdicts(
            request.safety_level,
            outcomes.iter().map(|outcome| &outcome.verdict),
        );

        FilterResponse {
            summary: FilterSummary {
                tweet_count: outcomes.len(),
                drop_count: outcomes
                    .iter()
                    .filter(|outcome| matches!(outcome.verdict.action, VfAction::Drop(_)))
                    .count(),
                unresolved_author_count,
            },
            outcomes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::MockSocialgraphClient;
    use crate::models::SafetyLabelType;
    use crate::safety_label_source::SafetyLabelSource;
    use crate::safety_label_source::lookup::{ManhattanLookup, RemoteSource, TwemcacheLookup};
    use crate::safety_label_source::types::{ManhattanOutcome, TwemcacheOutcome};
    use std::sync::Arc;
    use tonic::async_trait;
    use xai_core_entities::gizmoduck_client::{GizmoduckClient, MockGizmoduckClient};
    use xai_core_entities::tweet_entity_service_client::{MockTESClient, TESClient};

    fn full_label_map() -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(999_999, vf_pb::SafetyLabel::default())]),
        }
    }

    fn spam_high_recall_label_map(expires_at_msec: i64) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                i32::from(SafetyLabelType::SPAM_HIGH_RECALL),
                vf_pb::SafetyLabel {
                    expires_at_msec: Some(expires_at_msec),
                    ..Default::default()
                },
            )]),
        }
    }

    struct FakeTwemcache;

    #[async_trait]
    impl TwemcacheLookup for FakeTwemcache {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, TwemcacheOutcome> {
            ids.iter()
                .copied()
                .map(|id| {
                    let outcome = match id {
                        2 => TwemcacheOutcome::Hit(full_label_map()),
                        3 => TwemcacheOutcome::Hit(spam_high_recall_label_map(0)),
                        4 => TwemcacheOutcome::Hit(spam_high_recall_label_map(i64::MAX)),
                        _ => TwemcacheOutcome::Miss,
                    };
                    (id, outcome)
                })
                .collect()
        }
    }

    struct FakeManhattan;

    #[async_trait]
    impl ManhattanLookup for FakeManhattan {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, ManhattanOutcome> {
            ids.iter()
                .copied()
                .map(|id| (id, ManhattanOutcome::Resolved(full_label_map())))
                .collect()
        }
    }

    fn filter_tweets() -> FilterTweets {
        let tes: Arc<dyn TESClient + Send + Sync> = Arc::new(MockTESClient::default());
        let gizmoduck: Arc<dyn GizmoduckClient + Send + Sync> =
            Arc::new(MockGizmoduckClient::default());
        let socialgraph = Arc::new(MockSocialgraphClient::default());
        let twemcache = Arc::new(FakeTwemcache);
        let manhattan = Arc::new(FakeManhattan);
        let labels = Arc::new(SafetyLabelSource::new(Arc::new(RemoteSource::new(
            twemcache, manhattan,
        ))));

        FilterTweets::new(
            HydrationPipeline::new(
                tes,
                gizmoduck,
                socialgraph,
                labels,
                crate::hydration::FallbackCacheMode::Disabled,
                crate::hydration::FallbackCacheMode::Disabled,
            ),
            Policies::new(),
        )
    }

    fn candidate(tweet_id: u64, author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id: author_id,
        }
    }

    #[tokio::test]
    async fn run_preserves_order_duplicates_unresolved_authors_and_labels() {
        let response = filter_tweets()
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![
                    candidate(2, Some(20)),
                    candidate(1, None),
                    candidate(2, Some(20)),
                ],
            })
            .await;

        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.tweet_id)
                .collect::<Vec<_>>(),
            vec![TweetId(2), TweetId(1), TweetId(2)]
        );
        assert!(matches!(
            response.outcomes[0].verdict.action,
            VfAction::Allow
        ));
        assert!(matches!(
            response.outcomes[1].verdict.action,
            VfAction::Drop(_)
        ));
        assert_eq!(
            response.outcomes[1].verdict.decided_by,
            Some("unresolved_author_id")
        );
        assert!(matches!(
            response.outcomes[2].verdict.action,
            VfAction::Allow
        ));
        assert_eq!(response.summary.tweet_count, 3);
        assert_eq!(response.summary.drop_count, 1);
        assert_eq!(response.summary.unresolved_author_count, 1);
        assert!(response
            .outcomes
            .iter()
            .all(|outcome| outcome.safety_labels.is_some()));
        assert!(!response.outcomes[0]
            .safety_labels
            .as_ref()
            .unwrap()
            .labels
            .is_empty());
        assert_eq!(
            response.outcomes[0].safety_labels,
            response.outcomes[2].safety_labels
        );
    }

    #[tokio::test]
    async fn run_selects_policy_from_safety_level() {
        let service = filter_tweets();
        let request = |safety_level| FilterRequest {
            viewer_id: None,
            country_code: None,
            safety_level,
            candidates: vec![candidate(1, Some(10))],
        };

        let home = service.run(request(SafetyLevel::TimelineHome)).await;
        let filter_all = service.run(request(SafetyLevel::FilterAll)).await;

        assert!(matches!(home.outcomes[0].verdict.action, VfAction::Allow));
        assert!(matches!(
            filter_all.outcomes[0].verdict.action,
            VfAction::Drop(_)
        ));
        assert_eq!(
            filter_all.outcomes[0].verdict.decided_by,
            Some("FilterAllRule")
        );
    }

    #[tokio::test]
    async fn expired_spam_high_recall_allows_oon_and_is_removed_from_response() {
        let response = filter_tweets()
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                safety_level: SafetyLevel::TimelineHomeRecommendations,
                candidates: vec![candidate(3, Some(30))],
            })
            .await;

        assert!(matches!(
            response.outcomes[0].verdict.action,
            VfAction::Allow
        ));
        assert_eq!(response.outcomes[0].verdict.decided_by, None);
        assert!(
            response.outcomes[0]
                .safety_labels
                .as_ref()
                .is_some_and(|labels| labels.labels.is_empty())
        );
    }

    #[tokio::test]
    async fn unexpired_spam_high_recall_still_drops_oon() {
        let response = filter_tweets()
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                safety_level: SafetyLevel::TimelineHomeRecommendations,
                candidates: vec![candidate(4, Some(40))],
            })
            .await;

        assert!(matches!(
            response.outcomes[0].verdict.action,
            VfAction::Drop(_)
        ));
        assert_eq!(
            response.outcomes[0].verdict.decided_by,
            Some("SpamHighRecallDropRule")
        );
        assert!(
            response.outcomes[0]
                .safety_labels
                .as_ref()
                .is_some_and(|labels| labels
                    .labels
                    .contains_key(&i32::from(SafetyLabelType::SPAM_HIGH_RECALL)))
        );
    }
}
