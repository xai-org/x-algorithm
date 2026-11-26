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
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for CoreDataCandidateHydrator {
    type Fetched = Vec<Option<CoreDataCacheValue>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<CoreDataCacheValue>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<CoreDataCacheValue>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<CoreDataCacheValue>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            let Some(value) = value else {
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
    type Value = CoreDataCacheValue;

    fn cache_store(&self) -> &dyn CacheStore<u64, CoreDataCacheValue> {
        self.cache.as_ref()
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.tweet_id
    }

    fn existing_value(&self, candidate: &PostCandidate) -> Option<CoreDataCacheValue> {
        (candidate.author_id != 0 && !candidate.tweet_text.is_empty()).then(|| CoreDataCacheValue {
            author_id: candidate.author_id,
            retweeted_user_id: candidate.retweeted_user_id,
            retweeted_tweet_id: candidate.retweeted_tweet_id,
            in_reply_to_tweet_id: candidate.in_reply_to_tweet_id,
            ancestor_users: candidate.ancestor_users.clone(),
            tweet_text: candidate.tweet_text.clone(),
        })
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<CoreDataCacheValue>> {
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
                    values.push(Some(CoreDataCacheValue {
                        author_id: core_data.author_id,
                        retweeted_user_id: core_data.source_user_id,
                        retweeted_tweet_id: core_data.source_tweet_id,
                        in_reply_to_tweet_id: core_data.in_reply_to_tweet_id,
                        ancestor_users: build_ancestor_users(candidate, core_data, &post_features),
                        tweet_text: source_text.unwrap_or_else(|| core_data.text.clone()),
                    }));
                }
                Some(Ok(None)) | None => {
                    missing_count += 1;
                    values.push(Some(CoreDataCacheValue {
                        author_id: 0,
                        retweeted_user_id: None,
                        retweeted_tweet_id: None,
                        in_reply_to_tweet_id: None,
                        ancestor_users: Vec::new(),
                        tweet_text: String::new(),
                    }));
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
