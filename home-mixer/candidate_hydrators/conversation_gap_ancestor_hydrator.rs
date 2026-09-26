use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_core_entities::entities::PureCoreData;

pub struct ConversationGapAncestorHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
}

impl ConversationGapAncestorHydrator {
    pub fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        Self { tes_client }
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for ConversationGapAncestorHydrator {
    async fn hydrate(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let ancestor_ids: Vec<u64> = candidates
            .iter()
            .flat_map(|c| c.ancestors.iter().copied())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let mut ancestor_core = if ancestor_ids.is_empty() {
            HashMap::new()
        } else {
            self.tes_client.get_tweet_core_datas(ancestor_ids).await
        };

        let expanded: Vec<Vec<u64>> = candidates
            .iter()
            .map(|candidate| {
                expand_ancestors_for_gap(&candidate.ancestors, &ancestor_core)
                    .unwrap_or_else(|_| candidate.ancestors.to_vec())
            })
            .collect();

        let extra_ids: Vec<u64> = expanded
            .iter()
            .flatten()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|id| !ancestor_core.contains_key(id))
            .collect();
        if !extra_ids.is_empty() {
            ancestor_core.extend(self.tes_client.get_tweet_core_datas(extra_ids).await);
        }

        candidates
            .iter()
            .zip(expanded)
            .map(|(candidate, ancestors)| {
                Ok(PostCandidate {
                    tombstone_ancestor_ids: tombstone_ancestor_ids(&ancestors, &ancestor_core),
                    ancestor_texts: ancestor_texts(&ancestors, &ancestor_core),
                    ancestor_users: merge_ancestor_users(
                        &candidate.ancestor_users,
                        &ancestors,
                        &ancestor_core,
                    ),
                    ancestors,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.ancestors = hydrated.ancestors;
        candidate.tombstone_ancestor_ids = hydrated.tombstone_ancestor_ids;
        candidate.ancestor_texts = hydrated.ancestor_texts;
        candidate.ancestor_users = hydrated.ancestor_users;
    }
}

fn expand_ancestors_for_gap(
    ancestors: &[u64],
    parent_core: &HashMap<u64, anyhow::Result<Option<PureCoreData>>>,
) -> Result<Vec<u64>, String> {
    let [parent, root] = match ancestors {
        [parent, root] => [*parent, *root],
        _ => return Ok(ancestors.to_vec()),
    };

    match parent_core.get(&parent) {
        Some(Err(err)) => Err(err.to_string()),
        Some(Ok(Some(core))) => Ok(match core.in_reply_to_tweet_id {
            Some(grandparent) if grandparent != root && grandparent != parent => {
                vec![parent, grandparent, root]
            }
            _ => ancestors.to_vec(),
        }),
        Some(Ok(None)) | None => Ok(ancestors.to_vec()),
    }
}

fn tombstone_ancestor_ids(
    ancestors: &[u64],
    ancestor_core: &HashMap<u64, anyhow::Result<Option<PureCoreData>>>,
) -> Vec<u64> {
    ancestors
        .iter()
        .copied()
        .filter(|id| matches!(ancestor_core.get(id), Some(Ok(None))))
        .collect()
}

fn ancestor_texts(
    ancestors: &[u64],
    ancestor_core: &HashMap<u64, anyhow::Result<Option<PureCoreData>>>,
) -> HashMap<u64, String> {
    ancestors
        .iter()
        .copied()
        .filter_map(|id| match ancestor_core.get(&id) {
            Some(Ok(Some(data))) if !data.text.is_empty() => Some((id, data.text.clone())),
            _ => None,
        })
        .collect()
}

fn merge_ancestor_users(
    existing: &[u64],
    ancestors: &[u64],
    ancestor_core: &HashMap<u64, anyhow::Result<Option<PureCoreData>>>,
) -> Vec<u64> {
    let mut users = existing.to_vec();
    for id in ancestors {
        if let Some(Ok(Some(data))) = ancestor_core.get(id)
            && data.author_id != 0
            && !users.contains(&data.author_id)
        {
            users.push(data.author_id);
        }
    }
    users
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::tweet_entity_service_client::MockTESClient;
    use anyhow::anyhow;

    fn candidate(tweet_id: u64, ancestors: Vec<u64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            ancestors,
            ..Default::default()
        }
    }

    fn parent_core(parent_id: u64, in_reply_to: Option<u64>) -> HashMap<u64, Option<PureCoreData>> {
        let mut core_data = HashMap::new();
        core_data.insert(
            parent_id,
            Some(PureCoreData {
                in_reply_to_tweet_id: in_reply_to,
                ..Default::default()
            }),
        );
        core_data
    }

    async fn run(
        core_data: HashMap<u64, Option<PureCoreData>>,
        candidates: Vec<PostCandidate>,
    ) -> Vec<PostCandidate> {
        let client = Arc::new(MockTESClient {
            core_data,
            ..Default::default()
        });
        let hydrator =
            ConversationGapAncestorHydrator::new(client as Arc<dyn TESClient + Send + Sync>);
        let mut candidates = candidates;
        let hydrated = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        hydrator.update_all(&mut candidates, hydrated);
        candidates
    }

    #[tokio::test]
    async fn deep_chain_inserts_grandparent() {
        let out = run(parent_core(20, Some(15)), vec![candidate(30, vec![20, 10])]).await;
        assert_eq!(out[0].ancestors, vec![20, 15, 10]);
    }

    #[tokio::test]
    async fn spliced_grandparent_gets_text_users_and_tombstone() {
        let mut core_data = parent_core(20, Some(15));
        core_data.insert(
            15,
            Some(PureCoreData {
                author_id: 1500,
                text: "muted grandparent".to_string(),
                ..Default::default()
            }),
        );
        let mut reply = candidate(30, vec![20, 10]);
        reply.ancestor_users = vec![2000, 1000];
        let out = run(core_data, vec![reply]).await;
        assert_eq!(out[0].ancestors, vec![20, 15, 10]);
        assert_eq!(
            out[0].ancestor_texts.get(&15).map(String::as_str),
            Some("muted grandparent")
        );
        assert!(out[0].ancestor_users.contains(&1500));
        assert!(out[0].ancestor_users.contains(&2000));
        assert!(out[0].tombstone_ancestor_ids.is_empty());
    }

    #[tokio::test]
    async fn deleted_spliced_grandparent_is_tombstone() {
        let mut core_data = parent_core(20, Some(15));
        core_data.insert(15, None);
        let out = run(core_data, vec![candidate(30, vec![20, 10])]).await;
        assert_eq!(out[0].ancestors, vec![20, 15, 10]);
        assert_eq!(out[0].tombstone_ancestor_ids, vec![15]);
    }

    #[tokio::test]
    async fn short_chain_unchanged_when_parent_replies_to_root() {
        let out = run(parent_core(20, Some(10)), vec![candidate(30, vec![20, 10])]).await;
        assert_eq!(out[0].ancestors, vec![20, 10]);
    }

    #[tokio::test]
    async fn single_ancestor_unchanged() {
        let out = run(HashMap::new(), vec![candidate(30, vec![20])]).await;
        assert_eq!(out[0].ancestors, vec![20]);
    }

    #[tokio::test]
    async fn tes_miss_fail_open() {
        let out = run(HashMap::new(), vec![candidate(30, vec![20, 10])]).await;
        assert_eq!(out[0].ancestors, vec![20, 10]);
    }

    #[tokio::test]
    async fn deleted_root_is_tombstone() {
        let mut core_data = parent_core(20, Some(10));
        core_data.insert(10, None);
        let out = run(core_data, vec![candidate(30, vec![20, 10])]).await;
        assert_eq!(out[0].tombstone_ancestor_ids, vec![10]);
    }

    #[tokio::test]
    async fn parent_without_reply_to_unchanged() {
        let out = run(parent_core(20, None), vec![candidate(30, vec![20, 10])]).await;
        assert_eq!(out[0].ancestors, vec![20, 10]);
    }

    #[test]
    fn tes_err_surfaces_as_err() {
        let mut parent_core: HashMap<u64, anyhow::Result<Option<PureCoreData>>> = HashMap::new();
        parent_core.insert(20, Err(anyhow!("tes unavailable")));
        let err = expand_ancestors_for_gap(&[20, 10], &parent_core).unwrap_err();
        assert!(err.contains("tes unavailable"));
    }

    #[tokio::test]
    async fn batches_unique_parents_across_candidates() {
        let mut core_data = parent_core(20, Some(15));
        core_data.insert(
            40,
            Some(PureCoreData {
                in_reply_to_tweet_id: Some(35),
                ..Default::default()
            }),
        );
        let client = Arc::new(MockTESClient {
            core_data,
            ..Default::default()
        });
        let hydrator = ConversationGapAncestorHydrator::new(
            Arc::clone(&client) as Arc<dyn TESClient + Send + Sync>
        );
        let mut candidates = vec![
            candidate(30, vec![20, 10]),
            candidate(31, vec![20, 10]),
            candidate(50, vec![40, 10]),
            candidate(60, vec![]),
        ];
        let hydrated = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        hydrator.update_all(&mut candidates, hydrated);

        assert_eq!(candidates[0].ancestors, vec![20, 15, 10]);
        assert_eq!(candidates[1].ancestors, vec![20, 15, 10]);
        assert_eq!(candidates[2].ancestors, vec![40, 35, 10]);
        assert!(candidates[3].ancestors.is_empty());
        // Original ancestor ids, then a second TES batch for spliced grandparents.
        assert_eq!(client.call_count(), 2);
    }
}
