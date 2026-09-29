use crate::filters::vf_filter::should_drop_action;
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
        let mut all_results = oon_result;
        all_results.extend(in_network_result);

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let primary_result = all_results.get(&candidate.tweet_id);
            let (visibility_action, visibility_reason) = visibility_fields(primary_result);

            let drop_ancillary = should_drop_ancillary(candidate, &all_results);

            let hydrated = match primary_result {
                Some(Err(err)) => Err(err.to_string()),
                _ => Ok(PostCandidate {
                    visibility_action,
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
        candidate.visibility_action = hydrated.visibility_action;
        candidate.visibility_reason = hydrated.visibility_reason;
        candidate.drop_ancillary_posts = hydrated.drop_ancillary_posts;
    }
}

pub(crate) fn should_drop_ancillary(
    candidate: &PostCandidate,
    vf_results: &HashMap<u64, Result<TweetVisibility>>,
) -> bool {
    for &ancestor_id in &candidate.ancestors {
        if candidate.tombstone_ancestor_ids.contains(&ancestor_id) {
            continue;
        }
        if let Some(Ok(visibility)) = vf_results.get(&ancestor_id)
            && should_drop_action(&visibility.action)
        {
            return true;
        }
    }

    if let Some(quoted_id) = candidate.quoted_tweet_id
        && let Some(Ok(visibility)) = vf_results.get(&quoted_id)
        && should_drop_action(&visibility.action)
    {
        return true;
    }

    if let Some(retweeted_id) = candidate.retweeted_tweet_id
        && let Some(Ok(visibility)) = vf_results.get(&retweeted_id)
        && should_drop_action(&visibility.action)
    {
        return true;
    }

    false
}

pub(crate) fn visibility_fields(
    result: Option<&Result<TweetVisibility>>,
) -> (Option<Action>, Option<FilteredReason>) {
    match result {
        Some(Ok(visibility)) => (
            Some(visibility.action.clone()),
            visibility.to_visibility_reason(),
        ),
        // Err or missing id: fail closed. VFFilter drops NotEvaluated.
        // Leaving (None, None) kept the post (fail-open) because the filter
        // only partitions on Some(action).
        _ => (
            Some(Action::NotEvaluated),
            Some(FilteredReason::UnspecifiedReason),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_visibility_filtering::models::{DropReason, SafetyResult};
    use xai_visibility_filtering::tweet_safety_label::SafetyLabelFailure;

    fn visibility(action: Action) -> Result<TweetVisibility> {
        Ok(TweetVisibility {
            action,
            reason: None,
            safety_labels: Err(SafetyLabelFailure::LookupFailed),
        })
    }

    struct StratoReply(Vec<u8>);

    #[tonic::async_trait]
    impl xai_strato::strato_proto::rpc_server::Rpc for StratoReply {
        async fn issue3(
            &self,
            request: tonic::Request<xai_strato::strato_proto::Issue3>,
        ) -> Result<tonic::Response<xai_strato::strato_proto::ResultList>, tonic::Status> {
            use xai_strato::strato_proto::{Result as Reply, ResultList, result::ResultType};
            Ok(tonic::Response::new(ResultList {
                results: request
                    .into_inner()
                    .calls
                    .iter()
                    .map(|_| Reply {
                        result_type: Some(ResultType::Ok(self.0.clone().into())),
                    })
                    .collect(),
            }))
        }
    }

    #[tokio::test]
    async fn strato_suppression_bytes_keep_primary_and_ancillaries() {
        use crate::filters::{ancillary_vf_filter::AncillaryVFFilter, vf_filter::VFFilter};
        use xai_candidate_pipeline::filter::Filter;
        use xai_strato::{StratoGrpc, strato_proto::rpc_server::RpcServer};
        use xai_visibility_filtering::vf_client::StratoVfClient;

        for (arm, payload, dropped, raw_action) in [
            (4, vec![0], false, Action::NotEvaluated),
            (10, vec![0], false, Action::NotEvaluated),
            (13, vec![0], false, Action::NotEvaluated),
            (13, vec![12, 0, 5, 0, 0], false, Action::Avoid),
            (15, vec![0], false, Action::Avoid),
            (1, vec![0], false, Action::NotEvaluated),
            (3, vec![0], true, Action::Drop(DropReason {})),
        ] {
            let mut bytes = vec![
                12, 0, 4, 12, 9, 252, 12, 0, 118, 12, 105, 20, 12, 0, 11, 12, 0, 2, 12, 0, arm,
            ];
            bytes.extend(payload);
            bytes.extend([0; 7]);
            let service = tower::ServiceBuilder::new()
                .map_err(|never| match never {})
                .service(RpcServer::new(StratoReply(bytes)));
            let client = Arc::new(StratoVfClient {
                grpc_client: Arc::new(StratoGrpc::from_boxed_service(
                    tower::util::BoxCloneSyncService::new(service),
                )),
            });
            let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
            let mut params =
                fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
            params.override_fs("rust_home_mixer_enable_xai_vf_client".to_string(), "false");
            let query = ScoredPostsQuery {
                params: params.into(),
                ..Default::default()
            };
            let raw = client
                .get_result(vec![1], SafetyLevel::TimelineHomeRecommendations, 0, None)
                .await;
            let reason = Some(FilteredReason::SafetyResult(SafetyResult {
                reason: None,
                action: raw_action,
            }));
            assert_eq!(raw[&1].as_ref().unwrap().reason, reason);
            let hydrator = VFCandidateHydrator::new(client.clone(), client).await;
            let candidates = [PostCandidate {
                tweet_id: 1,
                retweeted_tweet_id: Some(2),
                quoted_tweet_id: Some(3),
                ancestors: vec![4],
                ..Default::default()
            }];
            let mut hydrated = hydrator.hydrate(&query, &candidates).await;
            let candidate = hydrated.pop().unwrap().unwrap();
            assert_eq!(candidate.drop_ancillary_posts, Some(dropped), "arm {arm}");
            assert_eq!(candidate.visibility_reason, reason, "arm {arm}");
            let result = AncillaryVFFilter.filter(&query, vec![candidate.clone()]);
            assert_eq!(result.removed.len(), usize::from(dropped), "arm {arm}");
            let result = VFFilter.filter(&query, vec![candidate]);
            assert_eq!(result.removed.len(), usize::from(dropped), "arm {arm}");
        }
    }

    #[test]
    fn ancillary_drops_on_dropped_quoted_tweet_but_not_avoid() {
        let candidate = PostCandidate {
            tweet_id: 1,
            quoted_tweet_id: Some(2),
            ..Default::default()
        };

        let dropped: HashMap<u64, Result<TweetVisibility>> =
            HashMap::from([(2, visibility(Action::Drop(DropReason {})))]);
        assert!(should_drop_ancillary(&candidate, &dropped));

        let avoided: HashMap<u64, Result<TweetVisibility>> =
            HashMap::from([(2, visibility(Action::Avoid))]);
        assert!(!should_drop_ancillary(&candidate, &avoided));

        let tombstoned: HashMap<u64, Result<TweetVisibility>> =
            HashMap::from([(2, visibility(Action::Tombstone))]);
        assert!(should_drop_ancillary(&candidate, &tombstoned));
    }

    #[test]
    fn ancillary_skips_tombstoned_ancestors_and_errors() {
        let candidate = PostCandidate {
            tweet_id: 1,
            ancestors: vec![2, 3],
            tombstone_ancestor_ids: vec![2],
            ..Default::default()
        };

        let results: HashMap<u64, Result<TweetVisibility>> = HashMap::from([
            (2, visibility(Action::Drop(DropReason {}))),
            (3, Err(anyhow::anyhow!("vf unavailable"))),
        ]);
        assert!(!should_drop_ancillary(&candidate, &results));
    }

    #[test]
    fn primary_lookup_miss_and_err_fail_closed() {
        let miss = visibility_fields(None);
        assert_eq!(miss.0, Some(Action::NotEvaluated));
        assert_eq!(miss.1, Some(FilteredReason::UnspecifiedReason));

        let err: Result<TweetVisibility> = Err(anyhow::anyhow!("vf unavailable"));
        let failed = visibility_fields(Some(&err));
        assert_eq!(failed.0, Some(Action::NotEvaluated));
        assert_eq!(failed.1, Some(FilteredReason::UnspecifiedReason));
    }
}
