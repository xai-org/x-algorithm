use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{QuickCache, default_quick_cache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};
use xai_core_entities::entities::PureCoreData;
use xai_stats_receiver::global_stats_receiver;

const FOUND_SCOPE: [(&str, &str); 1] = [("hydration", "found")];
const MISSING_SCOPE: [(&str, &str); 1] = [("hydration", "missing")];

#[derive(Clone)]
pub struct CoreDataCandidateHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: Arc<QuickCache<u64, CoreDataCacheValue>>,
}

impl CoreDataCandidateHydrator {
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        Self {
            tes_client,
            cache: Arc::new(default_quick_cache()),
        }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for CoreDataCandidateHydrator {
    type CacheKey = u64;

    type CacheValue = CoreDataCacheValue;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    fn already_hydrated(&self, candidate: &PostCandidate) -> bool {
        candidate.author_id != 0 && !candidate.tweet_text.is_empty()
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        self.cache.as_ref()
    }
    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.tweet_id
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        CoreDataCacheValue {
            author_id: hydrated.author_id,
            retweeted_user_id: hydrated.retweeted_user_id,
            retweeted_tweet_id: hydrated.retweeted_tweet_id,
            in_reply_to_tweet_id: hydrated.in_reply_to_tweet_id,
            ancestor_users: hydrated.ancestor_users.clone(),
            tweet_text: hydrated.tweet_text.clone(),
        }
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            author_id: value.author_id,
            retweeted_user_id: value.retweeted_user_id,
            retweeted_tweet_id: value.retweeted_tweet_id,
            in_reply_to_tweet_id: value.in_reply_to_tweet_id,
            ancestor_users: value.ancestor_users,
            tweet_text: value.tweet_text,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let client = &self.tes_client;

        let post_features = client
            .get_tweet_core_datas(core_data_fetch_ids(candidates))
            .await;
        let source_ids: Vec<u64> = candidates
            .iter()
            .filter_map(|c| match post_features.get(&c.tweet_id) {
                Some(Ok(Some(core_data))) => core_data.source_tweet_id,
                _ => None,
            })
            .collect::<std::collections::HashSet<u64>>()
            .into_iter()
            .collect();
        let source_features = if source_ids.is_empty() {
            HashMap::new()
        } else {
            client.get_tweet_core_datas(source_ids).await
        };

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        let mut hydrated_count = 0usize;
        let mut missing_count = 0usize;
        for candidate in candidates {
            match post_features.get(&candidate.tweet_id) {
                Some(Ok(Some(core_data))) => {
                    hydrated_count += 1;
                    let source_text =
                        core_data
                            .source_tweet_id
                            .and_then(|id| match source_features.get(&id) {
                                Some(Ok(Some(source))) => Some(source.text.clone()),
                                _ => None,
                            });
                    let text = source_text.unwrap_or_else(|| core_data.text.clone());
                    let ancestor_users = build_ancestor_users(candidate, core_data, &post_features);
                    let hydrated = PostCandidate {
                        author_id: core_data.author_id,
                        retweeted_user_id: core_data.source_user_id,
                        retweeted_tweet_id: core_data.source_tweet_id,
                        in_reply_to_tweet_id: core_data.in_reply_to_tweet_id,
                        ancestor_users,
                        tweet_text: text,
                        ..Default::default()
                    };
                    hydrated_candidates.push(Ok(hydrated));
                }
                Some(Ok(None)) | None => {
                    missing_count += 1;
                    hydrated_candidates.push(Ok(PostCandidate::default()));
                }
                Some(Err(err)) => {
                    hydrated_candidates.push(Err(err.to_string()));
                }
            }
        }

        self.record_hydration_stats(hydrated_count, missing_count);

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        apply_tes_core_data(candidate, &hydrated);
    }
}

/// TES author wins when TES resolved one. Source author_id is kept only on TES miss.
///
/// #128 runs InNetwork after this write. A leftover fill-if-zero kept Phoenix /
/// Thunder's nonzero author_id, so the stamp used a stale id and VF fetched
/// Recs. NSFW HP / gore / card then hard-dropped (Home would interstitial).
fn apply_tes_core_data(candidate: &mut PostCandidate, hydrated: &PostCandidate) {
    if hydrated.author_id != 0 {
        candidate.author_id = hydrated.author_id;
    }
    candidate.retweeted_user_id = hydrated.retweeted_user_id;
    candidate.retweeted_tweet_id = hydrated.retweeted_tweet_id;
    candidate.in_reply_to_tweet_id = hydrated.in_reply_to_tweet_id;
    candidate.ancestor_users = hydrated.ancestor_users.clone();
    candidate.tweet_text = hydrated.tweet_text.clone();
}

fn core_data_fetch_ids(candidates: &[PostCandidate]) -> Vec<u64> {
    let mut fetch_ids: Vec<u64> = candidates.iter().map(|c| c.tweet_id).collect();
    fetch_ids.extend(
        candidates
            .iter()
            .filter(|c| c.ancestors.len() == 2)
            .map(|c| c.ancestors[1]),
    );
    fetch_ids
}

fn build_ancestor_users(
    candidate: &PostCandidate,
    core_data: &PureCoreData,
    core_datas: &HashMap<u64, anyhow::Result<Option<PureCoreData>>>,
) -> Vec<u64> {
    let mut ancestor_users = Vec::with_capacity(candidate.ancestors.len());
    if !candidate.ancestors.is_empty()
        && let Some(parent_author) = core_data.in_reply_to_user_id
    {
        ancestor_users.push(parent_author);
    }
    if candidate.ancestors.len() == 2
        && let Some(Ok(Some(root))) = core_datas.get(&candidate.ancestors[1])
    {
        ancestor_users.push(root.author_id);
    }
    ancestor_users
}

#[derive(Clone, Debug)]
pub struct CoreDataCacheValue {
    pub author_id: u64,
    pub retweeted_user_id: Option<u64>,
    pub retweeted_tweet_id: Option<u64>,
    pub in_reply_to_tweet_id: Option<u64>,
    pub ancestor_users: Vec<u64>,
    pub tweet_text: String,
}

impl CoreDataCandidateHydrator {
    fn record_hydration_stats(&self, hydrated_count: usize, missing_count: usize) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.hydrate", self.name());
            receiver.incr(metric_name.as_str(), &FOUND_SCOPE, hydrated_count as u64);
            receiver.incr(metric_name.as_str(), &MISSING_SCOPE, missing_count as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_hydrators::in_network_candidate_hydrator::InNetworkCandidateHydrator;
    use crate::models::user_features::UserFeatures;
    use xai_candidate_pipeline::hydrator::Hydrator;

    fn apply_tes_author(source_author: u64, tes_author: u64) -> u64 {
        let mut candidate = PostCandidate {
            tweet_id: 1,
            author_id: source_author,
            ..Default::default()
        };
        apply_tes_core_data(
            &mut candidate,
            &PostCandidate {
                author_id: tes_author,
                tweet_text: "tes".into(),
                ..Default::default()
            },
        );
        candidate.author_id
    }

    #[test]
    fn tes_author_fills_zero_source() {
        assert_eq!(apply_tes_author(0, 42), 42);
    }

    #[test]
    fn tes_author_replaces_stale_nonzero_source() {
        assert_eq!(
            apply_tes_author(99, 42),
            42,
            "Phoenix/Thunder nonzero author must not block TES"
        );
    }

    #[test]
    fn tes_miss_keeps_source_author() {
        assert_eq!(apply_tes_author(99, 0), 99);
        assert_eq!(apply_tes_author(0, 0), 0);
    }

    #[tokio::test]
    async fn tes_author_then_in_network_stamps_followed_home() {
        // Residual after #128: CoreData runs first, but a stale source author
        // used to survive TES. InNetwork then stamped OON and VF Recs-dropped
        // NSFW HP / gore / card (Home would interstitial).
        let tes_author = 42u64;
        let stale_source = 99u64;
        let author_id = apply_tes_author(stale_source, tes_author);
        assert_eq!(author_id, tes_author);

        let query = ScoredPostsQuery {
            user_id: 1,
            user_features: UserFeatures {
                followed_user_ids: vec![tes_author as i64],
                ..Default::default()
            },
            ..Default::default()
        };
        let hydrator = InNetworkCandidateHydrator;
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            author_id,
            ..Default::default()
        }];
        let hydrated = hydrator.hydrate(&query, &candidates).await;
        let mut candidate = candidates.into_iter().next().unwrap();
        hydrator.update(
            &mut candidate,
            hydrated.into_iter().next().unwrap().unwrap(),
        );
        assert_eq!(
            candidate.in_network,
            Some(true),
            "followed TES author must stamp Home, not Recs"
        );
    }

    #[tokio::test]
    async fn stale_source_author_used_to_stamp_oon() {
        let query = ScoredPostsQuery {
            user_id: 1,
            user_features: UserFeatures {
                followed_user_ids: vec![42],
                ..Default::default()
            },
            ..Default::default()
        };
        let hydrator = InNetworkCandidateHydrator;
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            author_id: 99,
            ..Default::default()
        }];
        let hydrated = hydrator.hydrate(&query, &candidates).await;
        let mut candidate = candidates.into_iter().next().unwrap();
        hydrator.update(
            &mut candidate,
            hydrated.into_iter().next().unwrap().unwrap(),
        );
        assert_eq!(
            candidate.in_network,
            Some(false),
            "stale source author 99 is not followed"
        );
    }
}
