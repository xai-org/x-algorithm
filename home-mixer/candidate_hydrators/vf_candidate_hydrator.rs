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

        candidates
            .iter()
            .map(|candidate| resolve_visibility(candidate, &all_results))
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.visibility_reason = hydrated.visibility_reason;
        candidate.drop_ancillary_posts = hydrated.drop_ancillary_posts;
    }
}

/// Same sentinel `XaiVfClient::results_to_map` already writes for a missing
/// response id. `VFFilter` hard-drops every non-`SafetyResult` reason, so this
/// reaches the filter. Returning `Err` does not: `Hydrator::update_all` skips
/// the write and leaves `visibility_reason = None`, which `VFFilter` keeps.
fn vf_lookup_unavailable() -> FilteredReason {
    FilteredReason::UnspecifiedReason
}

pub(crate) fn resolve_visibility(
    candidate: &PostCandidate,
    vf_results: &HashMap<u64, Result<Option<FilteredReason>>>,
) -> Result<PostCandidate, String> {
    let primary_result = vf_results.get(&candidate.tweet_id);
    // Ok(None) is a successful Allow. Err and a missing map key are not.
    let visibility_reason = match primary_result {
        Some(Ok(Some(reason))) => Some(reason.clone()),
        Some(Ok(None)) => None,
        Some(Err(_)) | None => Some(vf_lookup_unavailable()),
    };

    Ok(PostCandidate {
        visibility_reason,
        drop_ancillary_posts: Some(should_drop_ancillary(candidate, vf_results)),
        ..Default::default()
    })
}

fn ancillary_verdict_blocks(verdict: Option<&Result<Option<FilteredReason>>>) -> bool {
    match verdict {
        Some(Ok(Some(reason))) => should_drop_reason(reason),
        Some(Ok(None)) => false,
        Some(Err(_)) | None => true,
    }
}

pub(crate) fn should_drop_ancillary(
    candidate: &PostCandidate,
    vf_results: &HashMap<u64, Result<Option<FilteredReason>>>,
) -> bool {
    for &ancestor_id in &candidate.ancestors {
        if candidate.tombstone_ancestor_ids.contains(&ancestor_id) {
            continue;
        }
        if ancillary_verdict_blocks(vf_results.get(&ancestor_id)) {
            return true;
        }
    }

    if let Some(quoted_id) = candidate.quoted_tweet_id
        && ancillary_verdict_blocks(vf_results.get(&quoted_id))
    {
        return true;
    }

    if let Some(retweeted_id) = candidate.retweeted_tweet_id
        && ancillary_verdict_blocks(vf_results.get(&retweeted_id))
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

    fn results(
        entries: Vec<(u64, Result<Option<FilteredReason>>)>,
    ) -> HashMap<u64, Result<Option<FilteredReason>>> {
        entries.into_iter().collect()
    }

    #[test]
    fn primary_lookup_error_stamps_unspecified_reason() {
        let vf_results = results(vec![(1, Err(anyhow::anyhow!("vf unavailable")))]);
        let post = PostCandidate {
            tweet_id: 1,
            in_network: Some(true),
            ..Default::default()
        };

        let hydrated = resolve_visibility(&post, &vf_results).unwrap();

        assert_eq!(
            hydrated.visibility_reason,
            Some(FilteredReason::UnspecifiedReason)
        );
        assert_eq!(hydrated.drop_ancillary_posts, Some(false));
    }

    #[test]
    fn primary_missing_key_stamps_unspecified_reason() {
        let vf_results = results(vec![]);
        let post = PostCandidate {
            tweet_id: 1,
            in_network: Some(true),
            ..Default::default()
        };

        let hydrated = resolve_visibility(&post, &vf_results).unwrap();

        assert_eq!(
            hydrated.visibility_reason,
            Some(FilteredReason::UnspecifiedReason)
        );
    }

    #[test]
    fn primary_allow_none_stays_none() {
        let vf_results = results(vec![(1, Ok(None))]);
        let post = PostCandidate {
            tweet_id: 1,
            in_network: Some(true),
            ..Default::default()
        };

        let hydrated = resolve_visibility(&post, &vf_results).unwrap();

        assert_eq!(hydrated.visibility_reason, None);
        assert_eq!(hydrated.drop_ancillary_posts, Some(false));
    }

    #[test]
    fn ancillary_error_drops() {
        let vf_results = results(vec![
            (1, Ok(None)),
            (10, Err(anyhow::anyhow!("vf unavailable"))),
        ]);
        let quote = PostCandidate {
            tweet_id: 1,
            in_network: Some(true),
            quoted_tweet_id: Some(10),
            ..Default::default()
        };

        assert!(should_drop_ancillary(&quote, &vf_results));
        let hydrated = resolve_visibility(&quote, &vf_results).unwrap();
        assert_eq!(hydrated.visibility_reason, None);
        assert_eq!(hydrated.drop_ancillary_posts, Some(true));
    }

    #[test]
    fn ancillary_missing_key_drops() {
        let vf_results = results(vec![(1, Ok(None))]);
        let reply = PostCandidate {
            tweet_id: 1,
            in_network: Some(true),
            ancestors: vec![10],
            ..Default::default()
        };

        assert!(should_drop_ancillary(&reply, &vf_results));
    }

    #[test]
    fn ancillary_allow_none_does_not_drop() {
        let vf_results = results(vec![(1, Ok(None)), (10, Ok(None))]);
        let quote = PostCandidate {
            tweet_id: 1,
            in_network: Some(true),
            quoted_tweet_id: Some(10),
            ..Default::default()
        };

        assert!(!should_drop_ancillary(&quote, &vf_results));
    }
}
