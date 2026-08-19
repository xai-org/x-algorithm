use crate::candidate_hydrators::vf_candidate_hydrator::should_drop_ancillary;
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
        };

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let primary_result = all_results.get(&candidate.tweet_id);
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
