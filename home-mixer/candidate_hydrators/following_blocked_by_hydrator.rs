use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::SocialGraphClientOps;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

pub struct FollowingBlockedByHydrator {
    socialgraph_client: Arc<dyn SocialGraphClientOps>,
}

impl FollowingBlockedByHydrator {
    pub async fn new(socialgraph_client: Arc<dyn SocialGraphClientOps>) -> Self {
        Self { socialgraph_client }
    }
}

pub struct BlocksViewer {
    author: bool,
    quoted_author: Option<bool>,
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for FollowingBlockedByHydrator {
    type Fetched = Vec<BlocksViewer>;

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<BlocksViewer>, String> {
        let user_ids: Vec<u64> = candidates
            .iter()
            .flat_map(|c| c.quoted_user_id.into_iter().chain(c.retweeted_user_id))
            .collect();

        let blocked_by_user_ids = self
            .socialgraph_client
            .check_blocked_by(query.user_id, &user_ids)
            .await
            .map_err(|e| e.to_string())?;
        Ok(candidates
            .iter()
            .map(|candidate| BlocksViewer {
                author: candidate
                    .retweeted_user_id
                    .is_some_and(|uid| blocked_by_user_ids.contains(&uid)),
                quoted_author: candidate
                    .quoted_user_id
                    .map(|uid| blocked_by_user_ids.contains(&uid)),
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        blocks_viewer: Vec<BlocksViewer>,
    ) {
        for (candidate, blocks) in candidates.iter_mut().zip(blocks_viewer) {
            candidate.author_blocks_viewer = Some(blocks.author);
            if blocks.quoted_author.is_some() {
                candidate.quoted_author_blocks_viewer = blocks.quoted_author;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.author_blocks_viewer = hydrated.author_blocks_viewer;
        if hydrated.quoted_author_blocks_viewer.is_some() {
            candidate.quoted_author_blocks_viewer = hydrated.quoted_author_blocks_viewer;
        }
    }
}
