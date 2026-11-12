use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableAiTrendFeedbackContext;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

const TOP_TREND_COUNT: usize = 2;

pub struct AiTrendFeedbackContextHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

impl AiTrendFeedbackContextHydrator {
    fn is_eligible_original_post(candidate: &PostCandidate) -> bool {
        candidate.in_reply_to_tweet_id.is_none()
            && candidate.retweeted_tweet_id.is_none()
            && candidate.ancestors.is_empty()
            && candidate.following_replied_user_ids.is_empty()
    }

    fn select_feedback_targets(candidate_trends: &[(usize, i64)]) -> HashMap<usize, i64> {
        let mut frequency: HashMap<i64, usize> = HashMap::new();
        for (_, trend_id) in candidate_trends {
            *frequency.entry(*trend_id).or_default() += 1;
        }

        let mut ranked: Vec<(i64, usize)> = frequency.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(TOP_TREND_COUNT);

        let mut rng = rand::rng();
        let mut selected: HashMap<usize, i64> = HashMap::new();
        let mut used_indices = std::collections::HashSet::new();

        for (trend_id, _) in ranked {
            let candidates_for_trend: Vec<usize> = candidate_trends
                .iter()
                .filter(|(idx, id)| !used_indices.contains(idx) && *id == trend_id)
                .map(|(idx, _)| *idx)
                .collect();
            let Some(&idx) = candidates_for_trend.choose(&mut rng) else {
                continue;
            };
            used_indices.insert(idx);
            selected.insert(idx, trend_id);
        }

        selected
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for AiTrendFeedbackContextHydrator {
    type Fetched = HashMap<usize, (String, String)>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableAiTrendFeedbackContext)
            && !query.is_topic_request()
            && !query.in_network_only
    }

    async fn fetch(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<HashMap<usize, (String, String)>, String> {
        let eligible: Vec<(usize, u64)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| Self::is_eligible_original_post(c))
            .map(|(i, c)| (i, c.tweet_id))
            .collect();
        if eligible.is_empty() {
            return Ok(HashMap::new());
        }

        let tweet_ids: Vec<u64> = eligible.iter().map(|(_, id)| *id).collect();
        let results = self
            .strato_client
            .batch_get_tweet_ai_trend(&tweet_ids)
            .await;

        let mut candidate_trends: Vec<(usize, i64)> = Vec::new();
        for ((idx, _), result) in eligible.into_iter().zip(results) {
            match result {
                Ok(Some(trend_id)) if trend_id != 0 => {
                    candidate_trends.push((idx, trend_id));
                }
                Ok(_) => {}
                Err(e) => {
                    warn!("AiTrendFeedbackContextHydrator: strato fetch error: {}", e);
                }
            }
        }
        let selected_ids = Self::select_feedback_targets(&candidate_trends);
        if selected_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut unique_ids: Vec<i64> = selected_ids.values().copied().collect();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        let names = self
            .strato_client
            .batch_get_ai_trend_name(&unique_ids)
            .await;
        let mut name_by_id: HashMap<i64, String> = HashMap::new();
        for (trend_id, result) in unique_ids.into_iter().zip(names) {
            match result {
                Ok(Some(name)) if !name.is_empty() => {
                    name_by_id.insert(trend_id, name);
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(
                        "AiTrendFeedbackContextHydrator: trend name fetch error: {}",
                        e
                    );
                }
            }
        }
        Ok(selected_ids
            .into_iter()
            .filter_map(|(idx, trend_id)| {
                name_by_id
                    .get(&trend_id)
                    .cloned()
                    .map(|name| (idx, (name, trend_id.to_string())))
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        mut trends: HashMap<usize, (String, String)>,
    ) {
        for (i, candidate) in candidates.iter_mut().enumerate() {
            let (name, trend_id) = trends
                .remove(&i)
                .map_or((None, None), |(name, id)| (Some(name), Some(id)));
            candidate.ai_trend_name = name;
            candidate.ai_trend_id = trend_id;
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.ai_trend_name = hydrated.ai_trend_name;
        candidate.ai_trend_id = hydrated.ai_trend_id;
    }
}
