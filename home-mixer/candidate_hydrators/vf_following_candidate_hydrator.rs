use crate::candidate_hydrators::vf_candidate_hydrator::{should_drop_ancillary, visibility_fields};
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableXaiVfClient;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;
use xai_twittercontext_proto::GetTwitterContextViewer;
use xai_visibility_filtering::models::{Action, FilteredReason};
use xai_visibility_filtering::vf_client::SafetyLevel::TimelineHome;
use xai_visibility_filtering::vf_client::{TweetVisibility, VfClient};

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

pub struct Visibility {
    action: Option<Action>,
    reason: Option<FilteredReason>,
    drop_ancillary_posts: bool,
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for VFFollowingCandidateHydrator {
    type Fetched = Vec<Option<Visibility>>;

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Visibility>>, String> {
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

        let all_results: HashMap<u64, Result<TweetVisibility>> = if post_ids.is_empty() {
            HashMap::new()
        } else {
            client
                .get_result(post_ids, TimelineHome, query.user_id, context)
                .await
        };

        Ok(candidates
            .iter()
            .map(|candidate| {
                let primary_result = all_results.get(&candidate.tweet_id);
                if let Some(Err(_)) = primary_result {
                    return None;
                }
                let (action, reason) = visibility_fields(primary_result);
                Some(Visibility {
                    action,
                    reason,
                    drop_ancillary_posts: should_drop_ancillary(candidate, &all_results),
                })
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        visibility: Vec<Option<Visibility>>,
    ) {
        for (candidate, visibility) in candidates.iter_mut().zip(visibility) {
            if let Some(visibility) = visibility {
                candidate.visibility_action = visibility.action;
                candidate.visibility_reason = visibility.reason;
                candidate.drop_ancillary_posts = Some(visibility.drop_ancillary_posts);
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.visibility_action = hydrated.visibility_action;
        candidate.visibility_reason = hydrated.visibility_reason;
        candidate.drop_ancillary_posts = hydrated.drop_ancillary_posts;
    }
}
