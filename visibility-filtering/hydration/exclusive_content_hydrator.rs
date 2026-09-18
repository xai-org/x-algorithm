use crate::clients::socialgraph_client::SocialgraphClient;
use crate::hydration::metrics::{
    batch_outcome, record_batch_size, timed_rpc, HydratorOutcome,
};
use crate::models::{ExclusiveContentFeatures, ExclusiveHydration, TweetId, Viewer};
use crate::rules::SafetyLevel;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use xai_core_entities::tweet_entity_service_client::TESClient;

const CLIENT_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const CLIENT: &str = "exclusive_content";

pub struct ExclusiveContentHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub sg_client: Arc<dyn SocialgraphClient + Send + Sync>,
}

impl ExclusiveContentHydrator {
    pub async fn hydrate(
        &self,
        tweet_ids: &[TweetId],
        viewer: Viewer,
        safety_level: SafetyLevel,
    ) -> HashMap<TweetId, ExclusiveHydration> {
        let raw_ids: Vec<u64> = tweet_ids.iter().map(|t| t.0).collect();
        let candidate_count = raw_ids.len();
        record_batch_size(CLIENT, candidate_count);

        let (outcome, exclusive_controls) = timed_rpc(
            CLIENT,
            "get_exclusive_controls",
            safety_level,
            candidate_count,
            CLIENT_TIMEOUT,
            batch_outcome,
            self.tes_client.get_exclusive_controls(raw_ids),
        )
        .await;

        let authors: HashMap<u64, Result<Option<u64>, String>> = exclusive_controls
            .into_iter()
            .map(|(id, result)| {
                (
                    id,
                    result
                        .map(|opt| opt.map(|ctrl| ctrl.conversation_author_id))
                        .map_err(|_| "tes exclusive_controls failed".to_string()),
                )
            })
            .collect();

        let root_author_ids: Vec<u64> = authors
            .values()
            .filter_map(|r| r.as_ref().ok().copied().flatten())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let super_follows = match viewer.user_id() {
            Some(vid) if !root_author_ids.is_empty() => {
                let (_, follows) = timed_rpc(
                    CLIENT,
                    "batch_check_super_follows",
                    safety_level,
                    candidate_count,
                    CLIENT_TIMEOUT,
                    |_| HydratorOutcome::Success,
                    self.sg_client
                        .batch_check_super_follows(vid, &root_author_ids),
                )
                .await;
                follows
            }
            _ => HashMap::new(),
        };

        resolve_exclusive(
            tweet_ids,
            outcome == HydratorOutcome::Timeout,
            &authors,
            &super_follows,
        )
    }
}

pub(crate) fn resolve_exclusive(
    tweet_ids: &[TweetId],
    timed_out: bool,
    exclusive_authors: &HashMap<u64, Result<Option<u64>, String>>,
    super_follows: &HashMap<u64, bool>,
) -> HashMap<TweetId, ExclusiveHydration> {
    tweet_ids
        .iter()
        .map(|tweet_id| {
            if timed_out {
                return (*tweet_id, ExclusiveHydration::Failed);
            }
            let hydration = match exclusive_authors.get(&tweet_id.0) {
                Some(Err(_)) => ExclusiveHydration::Failed,
                Some(Ok(None)) | None => ExclusiveHydration::Public,
                Some(Ok(Some(author_id))) => {
                    ExclusiveHydration::Exclusive(ExclusiveContentFeatures {
                        conversation_author_id: *author_id,
                        viewer_super_follows_author: super_follows
                            .get(author_id)
                            .copied()
                            .unwrap_or(false),
                    })
                }
            };
            (*tweet_id, hydration)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> TweetId {
        TweetId(n)
    }

    #[test]
    fn tes_ok_none_is_public() {
        let authors = HashMap::from([(1u64, Ok(None))]);
        let out = resolve_exclusive(&[id(1)], false, &authors, &HashMap::new());
        assert_eq!(out[&id(1)], ExclusiveHydration::Public);
    }

    #[test]
    fn tes_ok_some_is_exclusive() {
        let authors = HashMap::from([(1u64, Ok(Some(42u64)))]);
        let follows = HashMap::from([(42u64, false)]);
        let out = resolve_exclusive(&[id(1)], false, &authors, &follows);
        assert_eq!(
            out[&id(1)],
            ExclusiveHydration::Exclusive(ExclusiveContentFeatures {
                conversation_author_id: 42,
                viewer_super_follows_author: false,
            })
        );
    }

    #[test]
    fn tes_err_is_failed() {
        let authors = HashMap::from([(1u64, Err("tes unavailable".to_string()))]);
        let out = resolve_exclusive(&[id(1)], false, &authors, &HashMap::new());
        assert_eq!(out[&id(1)], ExclusiveHydration::Failed);
    }

    #[test]
    fn tes_timeout_fails_the_whole_batch() {
        let authors = HashMap::new();
        let out = resolve_exclusive(&[id(1), id(2)], true, &authors, &HashMap::new());
        assert_eq!(out[&id(1)], ExclusiveHydration::Failed);
        assert_eq!(out[&id(2)], ExclusiveHydration::Failed);
    }

    #[test]
    fn successful_empty_map_stays_public() {
        let out = resolve_exclusive(&[id(1)], false, &HashMap::new(), &HashMap::new());
        assert_eq!(out[&id(1)], ExclusiveHydration::Public);
    }

    #[test]
    fn subscriber_flag_comes_from_super_follows_map() {
        let authors = HashMap::from([(1u64, Ok(Some(42u64)))]);
        let follows = HashMap::from([(42u64, true)]);
        let out = resolve_exclusive(&[id(1)], false, &authors, &follows);
        match &out[&id(1)] {
            ExclusiveHydration::Exclusive(features) => {
                assert!(features.viewer_super_follows_author);
            }
            other => panic!("expected exclusive, got {other:?}"),
        }
    }
}
