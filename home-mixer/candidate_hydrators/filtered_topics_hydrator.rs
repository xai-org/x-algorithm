use crate::filters::topic_ids_filter::TopicFilteringOverrideMap;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::candidate_features::{FilteredTopicsByExperiment, TopicFilteringExperiment};
use crate::models::query::ScoredPostsQuery;
use crate::params::{TopicFilteringId, TopicFilteringOverrides};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::component_library::utils::{build_moka_cache_value_ttl, MokaCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};
use xai_strato::{decode, Bytes, StratoResult, StratoValue};

const CACHE_SIZE: u64 = 1_000_000;
const ANNOTATED_TTL: Duration = Duration::from_secs(30 * 60);
const UNANNOTATED_TTL: Duration = Duration::from_secs(30);

type Topics = Option<Arc<FilteredTopicsByExperiment>>;

fn decode_topics(result: &Result<Bytes, Box<dyn std::error::Error>>) -> Option<Topics> {
    match result {
        Ok(bytes) if !bytes.is_empty() => {
            let decoded: StratoResult<StratoValue<FilteredTopicsByExperiment>> = decode(bytes);
            match decoded {
                StratoResult::Ok(v) => Some(v.v.map(Arc::new)),
                StratoResult::Err(_) => None,
            }
        }
        Ok(_) => Some(None),
        Err(e) => {
            warn!("FilteredTopicsHydrator: strato fetch error: {}", e);
            None
        }
    }
}

fn topics_ttl(topics: &Topics) -> Duration {
    if topics.is_some() {
        ANNOTATED_TTL
    } else {
        UNANNOTATED_TTL
    }
}

fn experiment(query: &ScoredPostsQuery) -> TopicFilteringExperiment {
    if query.is_bulk_topic_request() || query.has_excluded_topics() {
        return TopicFilteringExperiment::Unfiltered;
    }
    let default_experiment = TopicFilteringExperiment::parse(&query.params.get(TopicFilteringId));
    let override_map = TopicFilteringOverrideMap::parse(&query.params.get(TopicFilteringOverrides));
    override_map.resolve(&query.topic_ids, default_experiment)
}

pub struct FilteredTopicsHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
    cache: MokaCache<u64, Topics>,
}

impl FilteredTopicsHydrator {
    pub fn new(strato_client: Arc<dyn StratoClient + Send + Sync>) -> Self {
        Self {
            strato_client,
            cache: build_moka_cache_value_ttl(CACHE_SIZE, topics_ttl),
        }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for FilteredTopicsHydrator {
    type Fetched = Vec<Option<Topics>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.is_topic_request() || query.has_excluded_topics()
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Topics>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<Topics>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<Topics>>,
    ) {
        let experiment = experiment(query);
        let need_unfiltered = experiment != TopicFilteringExperiment::Unfiltered;
        for (candidate, topics) in candidates.iter_mut().zip(values) {
            let Some(topics) = topics else {
                continue;
            };
            let topic_ids = |experiment| {
                topics
                    .as_ref()
                    .and_then(|topics| topics.topic_ids_for_experiment(experiment).cloned())
            };
            candidate.filtered_topic_ids = topic_ids(experiment);
            candidate.unfiltered_topic_ids = if need_unfiltered {
                topic_ids(TopicFilteringExperiment::Unfiltered)
            } else {
                None
            };
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.filtered_topic_ids = hydrated.filtered_topic_ids;
        candidate.unfiltered_topic_ids = hydrated.unfiltered_topic_ids;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for FilteredTopicsHydrator {
    type CacheKey = u64;
    type Value = Topics;

    fn cache_store(&self) -> &dyn CacheStore<u64, Topics> {
        &self.cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.get_original_tweet_id()
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<Topics>> {
        let mut post_ids: Vec<u64> = misses.iter().map(|c| c.get_original_tweet_id()).collect();
        post_ids.sort_unstable();
        post_ids.dedup();

        let results = self
            .strato_client
            .batch_get_filtered_topics_by_experiment(&post_ids)
            .await;
        let topics_by_id: HashMap<u64, Option<Topics>> = post_ids
            .into_iter()
            .zip(results.iter().map(decode_topics))
            .collect();

        misses
            .iter()
            .map(|c| {
                topics_by_id
                    .get(&c.get_original_tweet_id())
                    .cloned()
                    .flatten()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use xai_candidate_pipeline::component_library::clients::MockStratoClient;

    #[tokio::test]
    async fn unannotated_posts_are_looked_up_once_per_original_and_then_cached() {
        let strato = Arc::new(MockStratoClient::default());
        let hydrator = FilteredTopicsHydrator::new(strato.clone());
        let query = ScoredPostsQuery::default();
        let candidates = vec![
            PostCandidate {
                tweet_id: 1,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 1,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                retweeted_tweet_id: Some(1),
                ..Default::default()
            },
        ];

        for _ in 0..2 {
            let mut hydrated = candidates.clone();
            let fetched = hydrator.fetch(&query, &hydrated).await.unwrap();
            hydrator.apply(&query, &mut hydrated, fetched);
            assert!(hydrated.iter().all(|c| c.filtered_topic_ids.is_none()));
        }
        assert_eq!(strato.call_count.load(Ordering::SeqCst), 1);
    }
}
