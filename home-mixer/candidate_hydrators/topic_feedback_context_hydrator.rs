use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableTopicFeedbackContext;
use crate::util::post_annotations::decode_post_annotations;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;
use xai_recsys_proto::grok_topics::{topic_index, PARENT_TOPIC_IDS};

const TOP_TOPIC_COUNT: usize = 2;

pub struct TopicFeedbackContextHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

impl TopicFeedbackContextHydrator {
    fn is_eligible_original_post(candidate: &PostCandidate) -> bool {
        candidate.in_reply_to_tweet_id.is_none()
            && candidate.retweeted_tweet_id.is_none()
            && candidate.ancestors.is_empty()
            && candidate.following_replied_user_ids.is_empty()
    }

    fn is_subtopic(topic_id: i64) -> bool {
        topic_index(topic_id).is_some() && !PARENT_TOPIC_IDS.contains(&topic_id)
    }

    fn decode_topics(bytes: &[u8]) -> (Vec<TopicAnnotation>, Vec<String>) {
        match decode_post_annotations(bytes) {
            Ok(Some(annotations)) => {
                let topics = annotations
                    .entities
                    .iter()
                    .enumerate()
                    .filter_map(|(i, entity)| {
                        let topic_id = entity.entity_id;
                        if topic_id == 0 || !Self::is_subtopic(topic_id) {
                            return None;
                        }
                        let topic_name = annotations.topics.get(i)?.clone();
                        Some(TopicAnnotation {
                            topic_id,
                            topic_name,
                        })
                    })
                    .collect();
                (topics, annotations.topics)
            }
            Ok(None) => (Vec::new(), Vec::new()),
            Err(e) => {
                warn!("TopicFeedbackContextHydrator: decode error: {}", e);
                (Vec::new(), Vec::new())
            }
        }
    }

    fn select_feedback_targets(
        candidate_topics: &[(usize, Vec<TopicAnnotation>)],
    ) -> HashMap<usize, (String, String)> {
        let mut frequency: HashMap<i64, usize> = HashMap::new();
        let mut name_by_id: HashMap<i64, String> = HashMap::new();
        for (_, topics) in candidate_topics {
            for topic in topics {
                *frequency.entry(topic.topic_id).or_default() += 1;
                name_by_id
                    .entry(topic.topic_id)
                    .or_insert_with(|| topic.topic_name.clone());
            }
        }

        let mut ranked: Vec<(i64, usize)> = frequency.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(TOP_TOPIC_COUNT);

        let mut rng = rand::rng();
        let mut selected: HashMap<usize, (String, String)> = HashMap::new();
        let mut used_indices = std::collections::HashSet::new();

        for (topic_id, _) in ranked {
            let name = match name_by_id.get(&topic_id) {
                Some(n) => n.clone(),
                None => continue,
            };
            let candidates_for_topic: Vec<usize> = candidate_topics
                .iter()
                .filter(|(idx, topics)| {
                    !used_indices.contains(idx) && topics.iter().any(|t| t.topic_id == topic_id)
                })
                .map(|(idx, _)| *idx)
                .collect();
            let Some(&idx) = candidates_for_topic.choose(&mut rng) else {
                continue;
            };
            used_indices.insert(idx);
            selected.insert(idx, (name, topic_id.to_string()));
        }

        selected
    }
}

#[derive(Default)]
pub struct TopicFeedbackContext {
    feedback_topics: HashMap<usize, (String, String)>,
    grok_topics: HashMap<usize, Vec<String>>,
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for TopicFeedbackContextHydrator {
    type Fetched = TopicFeedbackContext;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableTopicFeedbackContext)
            && !query.is_topic_request()
            && !query.in_network_only
    }

    async fn fetch(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<TopicFeedbackContext, String> {
        let eligible: Vec<(usize, u64)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| Self::is_eligible_original_post(c))
            .map(|(i, c)| (i, c.tweet_id))
            .collect();
        if eligible.is_empty() {
            return Ok(TopicFeedbackContext::default());
        }

        let tweet_ids: Vec<u64> = eligible.iter().map(|(_, id)| *id).collect();
        let results = self
            .strato_client
            .batch_get_unified_post_annotations(&tweet_ids)
            .await;

        let mut grok_topics: HashMap<usize, Vec<String>> = HashMap::new();
        let mut candidate_topics: Vec<(usize, Vec<TopicAnnotation>)> = Vec::new();
        for ((idx, _), result) in eligible.into_iter().zip(results) {
            match result {
                Ok(bytes) if !bytes.is_empty() => {
                    let (annotations, topic_names) = Self::decode_topics(&bytes);
                    if !topic_names.is_empty() {
                        grok_topics.insert(idx, topic_names);
                    }
                    if !annotations.is_empty() {
                        candidate_topics.push((idx, annotations));
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    warn!("TopicFeedbackContextHydrator: strato fetch error: {}", e);
                }
            }
        }
        Ok(TopicFeedbackContext {
            feedback_topics: Self::select_feedback_targets(&candidate_topics),
            grok_topics,
        })
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        mut context: TopicFeedbackContext,
    ) {
        for (i, candidate) in candidates.iter_mut().enumerate() {
            let (topic, topic_id) = context
                .feedback_topics
                .remove(&i)
                .map_or((None, None), |(topic, id)| (Some(topic), Some(id)));
            candidate.topic_feedback_topic = topic;
            candidate.topic_feedback_topic_id = topic_id;
            candidate.grok_topics = context.grok_topics.remove(&i);
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.topic_feedback_topic = hydrated.topic_feedback_topic;
        candidate.topic_feedback_topic_id = hydrated.topic_feedback_topic_id;
        candidate.grok_topics = hydrated.grok_topics;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TopicAnnotation {
    topic_id: i64,
    topic_name: String,
}
