use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnablePhoenixOonReplies;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;
use xai_home_mixer_proto as pb;

pub fn is_phoenix_home_retrieval(candidate: &PostCandidate) -> bool {
    matches!(
        candidate.served_type,
        Some(pb::ServedType::ForYouPhoenixRetrieval | pb::ServedType::ForYouPhoenixRetrievalCold)
    )
}

fn needs_ancestors(candidate: &PostCandidate) -> bool {
    is_phoenix_home_retrieval(candidate) && candidate.ancestors.is_empty()
}

pub struct PhoenixReplyAncestorsHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for PhoenixReplyAncestorsHydrator {
    type Fetched = Vec<Option<ReplyAncestors>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts && query.params.get(EnablePhoenixOonReplies)
    }

    async fn fetch(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<ReplyAncestors>>, String> {
        let ids: Vec<u64> = candidates
            .iter()
            .filter(|c| needs_ancestors(c))
            .map(|c| c.tweet_id)
            .collect::<HashSet<u64>>()
            .into_iter()
            .collect();
        if ids.is_empty() {
            return Ok(candidates.iter().map(|_| None).collect());
        }

        let core_datas = self.tes_client.get_tweet_core_datas(ids).await;
        let root_ids: Vec<u64> = core_datas
            .values()
            .filter_map(|r| match r {
                Ok(Some(core)) => {
                    let parent = core.in_reply_to_tweet_id?;
                    core.conversation_id.filter(|&root| root != parent)
                }
                _ => None,
            })
            .collect::<HashSet<u64>>()
            .into_iter()
            .collect();
        let root_datas = if root_ids.is_empty() {
            Default::default()
        } else {
            self.tes_client.get_tweet_core_datas(root_ids).await
        };

        Ok(candidates
            .iter()
            .map(|candidate| {
                if !needs_ancestors(candidate) {
                    return None;
                }
                let Some(Ok(Some(core))) = core_datas.get(&candidate.tweet_id) else {
                    return None;
                };
                let parent = core.in_reply_to_tweet_id?;
                let mut ancestors = vec![parent];
                let mut ancestor_users = Vec::with_capacity(2);
                if let Some(parent_author) = core.in_reply_to_user_id {
                    ancestor_users.push(parent_author);
                }
                if let Some(root) = core.conversation_id.filter(|&root| root != parent) {
                    ancestors.push(root);
                    if let Some(Ok(Some(root_core))) = root_datas.get(&root) {
                        ancestor_users.push(root_core.author_id);
                    }
                }
                Some(ReplyAncestors {
                    ancestors,
                    ancestor_users,
                    conversation_id: core.conversation_id,
                })
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        reply_ancestors: Vec<Option<ReplyAncestors>>,
    ) {
        for (candidate, reply_ancestors) in candidates.iter_mut().zip(reply_ancestors) {
            if let Some(reply_ancestors) = reply_ancestors {
                candidate.ancestors = reply_ancestors.ancestors;
                candidate.ancestor_users = reply_ancestors.ancestor_users;
                candidate.conversation_id = reply_ancestors.conversation_id;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        if !hydrated.ancestors.is_empty() {
            candidate.ancestors = hydrated.ancestors;
            candidate.ancestor_users = hydrated.ancestor_users;
            candidate.conversation_id = hydrated.conversation_id;
        }
    }
}

pub struct ReplyAncestors {
    ancestors: Vec<u64>,
    ancestor_users: Vec<u64>,
    conversation_id: Option<u64>,
}
