use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::utils::{QuickCache, default_quick_cache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};
use xai_core_entities::entities::PureCoreData;
use xai_stats_receiver::global_stats_receiver;

const FOUND_SCOPE: [(&str, &str); 1] = [("hydration", "found")];
const MISSING_SCOPE: [(&str, &str); 1] = [("hydration", "missing")];

#[derive(Clone)]
pub struct CoreDataCandidateHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: Arc<QuickCache<u64, Option<CoreDataCacheValue>>>,
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
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for CoreDataCandidateHydrator {
    type Fetched = Vec<Option<Option<CoreDataCacheValue>>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Option<CoreDataCacheValue>>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<Option<CoreDataCacheValue>>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<Option<CoreDataCacheValue>>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            let Some(Some(value)) = value else {
                continue;
            };
            if candidate.author_id == 0 && value.author_id != 0 {
                candidate.author_id = value.author_id;
            }
            candidate.retweeted_user_id = value.retweeted_user_id;
            candidate.retweeted_tweet_id = value.retweeted_tweet_id;
            candidate.in_reply_to_tweet_id = value.in_reply_to_tweet_id;
            candidate.ancestor_users = value.ancestor_users;
            candidate.tweet_text = value.tweet_text;
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        if candidate.author_id == 0 && hydrated.author_id != 0 {
            candidate.author_id = hydrated.author_id;
        }
        candidate.retweeted_user_id = hydrated.retweeted_user_id;
        candidate.retweeted_tweet_id = hydrated.retweeted_tweet_id;
        candidate.in_reply_to_tweet_id = hydrated.in_reply_to_tweet_id;
        candidate.ancestor_users = hydrated.ancestor_users;
        candidate.tweet_text = hydrated.tweet_text;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for CoreDataCandidateHydrator {
    type CacheKey = u64;
    type Value = Option<CoreDataCacheValue>;

    fn cache_store(&self) -> &dyn CacheStore<u64, Option<CoreDataCacheValue>> {
        self.cache.as_ref()
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.tweet_id
    }

    fn existing_value(&self, candidate: &PostCandidate) -> Option<Option<CoreDataCacheValue>> {
        (candidate.author_id != 0 && !candidate.tweet_text.is_empty()).then(|| {
            Some(CoreDataCacheValue {
                author_id: candidate.author_id,
                retweeted_user_id: candidate.retweeted_user_id,
                retweeted_tweet_id: candidate.retweeted_tweet_id,
                in_reply_to_tweet_id: candidate.in_reply_to_tweet_id,
                ancestor_users: candidate.ancestor_users.clone(),
                tweet_text: candidate.tweet_text.clone(),
            })
        })
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<Option<CoreDataCacheValue>>> {
        let client = &self.tes_client;

        let post_features = client
            .get_tweet_core_datas(core_data_fetch_ids(misses))
            .await;
        let source_ids: Vec<u64> = misses
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

        let mut values = Vec::with_capacity(misses.len());
        let mut hydrated_count = 0usize;
        let mut missing_count = 0usize;
        for candidate in misses {
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
                    values.push(Some(Some(CoreDataCacheValue {
                        author_id: core_data.author_id,
                        retweeted_user_id: core_data.source_user_id,
                        retweeted_tweet_id: core_data.source_tweet_id,
                        in_reply_to_tweet_id: core_data.in_reply_to_tweet_id,
                        ancestor_users: build_ancestor_users(candidate, core_data, &post_features),
                        tweet_text: source_text.unwrap_or_else(|| core_data.text.clone()),
                    })));
                }
                Some(Ok(None)) | None => {
                    missing_count += 1;
                    // Cache the miss as an explicit inner None. The outer None
                    // remains reserved for a transport/error result.
                    values.push(Some(None));
                }
                Some(Err(_)) => values.push(None),
            }
        }

        self.record_hydration_stats(hydrated_count, missing_count);

        values
    }
}

fn core_data_fetch_ids(candidates: &[&PostCandidate]) -> Vec<u64> {
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
            let metric_name = format!("{}.hydrate", TwoPhaseHydrator::name(self));
            receiver.incr(metric_name.as_str(), &FOUND_SCOPE, hydrated_count as u64);
            receiver.incr(metric_name.as_str(), &MISSING_SCOPE, missing_count as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::tweet_entity_service_client::{MockTESClient, TESClient};
    use std::sync::Arc;
    use xai_candidate_pipeline::hydrator::Hydrator;

    fn thunder_retweet() -> PostCandidate {
        PostCandidate {
            tweet_id: 10,
            author_id: 100,
            retweeted_tweet_id: Some(99),
            retweeted_user_id: Some(200),
            in_reply_to_tweet_id: Some(50),
            tweet_text: String::new(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn tes_success_fills_and_can_clear_graph_fields() {
        let hydrator = CoreDataCandidateHydrator::new(
            Arc::new(MockTESClient::default()) as Arc<dyn TESClient + Send + Sync>
        )
        .await;
        let mut candidate = thunder_retweet();
        Hydrator::update(
            &hydrator,
            &mut candidate,
            PostCandidate {
                author_id: 7,
                retweeted_tweet_id: None,
                retweeted_user_id: None,
                in_reply_to_tweet_id: None,
                tweet_text: "original".to_string(),
                ancestor_users: vec![],
                ..Default::default()
            },
        );
        assert_eq!(candidate.author_id, 100);
        assert_eq!(candidate.retweeted_tweet_id, None);
        assert_eq!(candidate.retweeted_user_id, None);
        assert_eq!(candidate.in_reply_to_tweet_id, None);
        assert_eq!(candidate.tweet_text, "original");
    }

    #[tokio::test]
    async fn tes_miss_hydrate_leaves_thunder_retweet_ids() {
        let hydrator = CoreDataCandidateHydrator::new(
            Arc::new(MockTESClient::default()) as Arc<dyn TESClient + Send + Sync>
        )
        .await;
        let mut candidates = vec![thunder_retweet()];
        let hydrated = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        assert!(
            hydrated[0].is_ok(),
            "miss stays Ok so it can be negative-cached"
        );
        hydrator.update_all(&mut candidates, hydrated);
        assert_eq!(candidates[0].retweeted_tweet_id, Some(99));
        assert_eq!(candidates[0].retweeted_user_id, Some(200));
        assert_eq!(candidates[0].in_reply_to_tweet_id, Some(50));
        assert_eq!(candidates[0].author_id, 100);
    }

    #[test]
    fn two_phase_apply_ignores_negative_cache_and_error_slots() {
        let hydrator = CoreDataCandidateHydrator {
            tes_client: Arc::new(MockTESClient::default()) as Arc<dyn TESClient + Send + Sync>,
            cache: Arc::new(default_quick_cache()),
        };
        let mut candidates = vec![thunder_retweet()];
        TwoPhaseHydrator::apply(
            &hydrator,
            &ScoredPostsQuery::default(),
            &mut candidates,
            vec![Some(None)],
        );
        assert_eq!(candidates[0].retweeted_tweet_id, Some(99));
        assert_eq!(candidates[0].retweeted_user_id, Some(200));
        assert_eq!(candidates[0].in_reply_to_tweet_id, Some(50));

        TwoPhaseHydrator::apply(
            &hydrator,
            &ScoredPostsQuery::default(),
            &mut candidates,
            vec![None],
        );
        assert_eq!(candidates[0].retweeted_tweet_id, Some(99));
        assert_eq!(candidates[0].retweeted_user_id, Some(200));
        assert_eq!(candidates[0].in_reply_to_tweet_id, Some(50));
    }

    #[test]
    fn two_phase_apply_keeps_real_hit_authoritative() {
        let hydrator = CoreDataCandidateHydrator {
            tes_client: Arc::new(MockTESClient::default()) as Arc<dyn TESClient + Send + Sync>,
            cache: Arc::new(default_quick_cache()),
        };
        let mut candidates = vec![thunder_retweet()];
        TwoPhaseHydrator::apply(
            &hydrator,
            &ScoredPostsQuery::default(),
            &mut candidates,
            vec![Some(Some(CoreDataCacheValue {
                author_id: 7,
                retweeted_user_id: None,
                retweeted_tweet_id: None,
                in_reply_to_tweet_id: None,
                ancestor_users: vec![],
                tweet_text: "original".to_string(),
            }))],
        );
        assert_eq!(candidates[0].author_id, 100);
        assert_eq!(candidates[0].retweeted_tweet_id, None);
        assert_eq!(candidates[0].retweeted_user_id, None);
        assert_eq!(candidates[0].in_reply_to_tweet_id, None);
        assert_eq!(candidates[0].tweet_text, "original");
    }
}
