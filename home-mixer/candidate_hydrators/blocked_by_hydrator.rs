use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::SocialGraphClientOps;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

pub struct BlockedByHydrator {
    pub socialgraph_client: Arc<dyn SocialGraphClientOps>,
}

impl BlockedByHydrator {
    pub async fn new(socialgraph_client: Arc<dyn SocialGraphClientOps>) -> Self {
        Self { socialgraph_client }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for BlockedByHydrator {
    type Fetched = Vec<bool>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<bool>, String> {
        let author_ids: Vec<u64> = candidates.iter().map(|x| x.author_id).collect();
        let blocked_by_user_ids = self
            .socialgraph_client
            .check_blocked_by(query.user_id, &author_ids)
            .await
            .map_err(|e| e.to_string())?;
        Ok(candidates
            .iter()
            .map(|candidate| blocked_by_user_ids.contains(&candidate.author_id))
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        author_blocks_viewer: Vec<bool>,
    ) {
        for (candidate, blocks) in candidates.iter_mut().zip(author_blocks_viewer) {
            candidate.author_blocks_viewer = Some(blocks);
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.author_blocks_viewer = hydrated.author_blocks_viewer;
    }
}
