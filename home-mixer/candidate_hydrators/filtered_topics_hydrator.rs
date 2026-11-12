use crate::filters::topic_ids_filter::TopicFilteringOverrideMap;
use crate::models::candidate::PostCandidate;
use crate::models::candidate_features::{FilteredTopicsByExperiment, TopicFilteringExperiment};
use crate::models::query::ScoredPostsQuery;
use crate::params::{TopicFilteringId, TopicFilteringOverrides};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;
use xai_strato::{decode, Bytes, StratoResult, StratoValue};

fn decode_topics_pair(
    result: &Result<Bytes, Box<dyn std::error::Error>>,
    experiment: TopicFilteringExperiment,
    need_unfiltered: bool,
) -> (Option<Vec<i64>>, Option<Vec<i64>>) {
    match result {
        Ok(bytes) if !bytes.is_empty() => {
            let decoded: StratoResult<StratoValue<FilteredTopicsByExperiment>> = decode(bytes);
            match decoded {
                StratoResult::Ok(v) => {
                    let ft = v.v;
                    let exp_topics = ft
                        .as_ref()
                        .and_then(|ft| ft.topic_ids_for_experiment(experiment).cloned());
                    let unf_topics = if need_unfiltered {
                        ft.as_ref().and_then(|ft| {
                            ft.topic_ids_for_experiment(TopicFilteringExperiment::Unfiltered)
                                .cloned()
                        })
                    } else {
                        None
                    };
                    (exp_topics, unf_topics)
                }
                StratoResult::Err(_) => (None, None),
            }
        }
        Ok(_) => (None, None),
        Err(e) => {
            warn!("FilteredTopicsHydrator: strato fetch error: {}", e);
            (None, None)
        }
    }
}

pub struct FilteredTopicsHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for FilteredTopicsHydrator {
    type Fetched = Vec<TopicIds>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.is_topic_request() || query.has_excluded_topics()
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<TopicIds>, String> {
        let experiment = if query.is_bulk_topic_request() || query.has_excluded_topics() {
            TopicFilteringExperiment::Unfiltered
        } else {
            let default_experiment =
                TopicFilteringExperiment::parse(&query.params.get(TopicFilteringId));
            let override_map =
                TopicFilteringOverrideMap::parse(&query.params.get(TopicFilteringOverrides));
            override_map.resolve(&query.topic_ids, default_experiment)
        };

        let client = &self.strato_client;
        let need_unfiltered = experiment != TopicFilteringExperiment::Unfiltered;

        let mut all_ids: Vec<u64> = candidates.iter().map(|c| c.tweet_id).collect();
        let retweet_offset = all_ids.len();
        for c in candidates {
            if let Some(rt_id) = c.retweeted_tweet_id {
                all_ids.push(rt_id);
            }
        }

        let all_results = client
            .batch_get_filtered_topics_by_experiment(&all_ids)
            .await;

        let mut retweet_topics: HashMap<u64, Vec<i64>> = HashMap::new();
        let mut retweet_unfiltered: HashMap<u64, Vec<i64>> = HashMap::new();
        let mut rt_idx = retweet_offset;
        for c in candidates {
            if let Some(rt_id) = c.retweeted_tweet_id {
                let (exp, unf) =
                    decode_topics_pair(&all_results[rt_idx], experiment, need_unfiltered);
                if let Some(topics) = exp {
                    retweet_topics.insert(rt_id, topics);
                }
                if let Some(topics) = unf {
                    retweet_unfiltered.insert(rt_id, topics);
                }
                rt_idx += 1;
            }
        }

        Ok(candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let (filtered, unfiltered) = if let Some(rt_id) = c.retweeted_tweet_id {
                    (
                        retweet_topics.get(&rt_id).cloned(),
                        retweet_unfiltered.get(&rt_id).cloned(),
                    )
                } else {
                    decode_topics_pair(&all_results[i], experiment, need_unfiltered)
                };
                TopicIds {
                    filtered,
                    unfiltered,
                }
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        topic_ids: Vec<TopicIds>,
    ) {
        for (candidate, topic_ids) in candidates.iter_mut().zip(topic_ids) {
            candidate.filtered_topic_ids = topic_ids.filtered;
            candidate.unfiltered_topic_ids = topic_ids.unfiltered;
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.filtered_topic_ids = hydrated.filtered_topic_ids;
        candidate.unfiltered_topic_ids = hydrated.unfiltered_topic_ids;
    }
}

pub struct TopicIds {
    filtered: Option<Vec<i64>>,
    unfiltered: Option<Vec<i64>>,
}
