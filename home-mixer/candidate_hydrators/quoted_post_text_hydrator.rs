use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

pub struct QuotedPostTextHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
}

impl QuotedPostTextHydrator {
    pub fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        Self { tes_client }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for QuotedPostTextHydrator {
    type Fetched = Vec<Option<String>>;

    async fn fetch(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<String>>, String> {
        let quoted_ids: Vec<u64> = candidates
            .iter()
            .filter_map(|c| c.quoted_tweet_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let quoted_core = if quoted_ids.is_empty() {
            HashMap::new()
        } else {
            self.tes_client.get_tweet_core_datas(quoted_ids).await
        };

        Ok(candidates
            .iter()
            .map(|candidate| {
                candidate
                    .quoted_tweet_id
                    .and_then(|id| match quoted_core.get(&id) {
                        Some(Ok(Some(data))) if !data.text.is_empty() => Some(data.text.clone()),
                        _ => None,
                    })
            })
            .collect())
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        quoted_texts: Vec<Option<String>>,
    ) {
        for (candidate, quoted_text) in candidates.iter_mut().zip(quoted_texts) {
            candidate.quoted_tweet_text = quoted_text;
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.quoted_tweet_text = hydrated.quoted_tweet_text;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::tweet_entity_service_client::MockTESClient;
    use xai_core_entities::entities::PureCoreData;

    #[tokio::test]
    async fn fills_quoted_text_and_leaves_non_quotes_empty() {
        let mut core_data = HashMap::new();
        core_data.insert(
            99,
            Some(PureCoreData {
                text: "quoted text".to_string(),
                ..Default::default()
            }),
        );
        let client = Arc::new(MockTESClient {
            core_data,
            ..Default::default()
        });
        let hydrator = QuotedPostTextHydrator::new(client as Arc<dyn TESClient + Send + Sync>);

        let query = ScoredPostsQuery::default();
        let mut candidates = vec![
            PostCandidate {
                tweet_id: 1,
                quoted_tweet_id: Some(99),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                ..Default::default()
            },
        ];

        let fetched = hydrator.fetch(&query, &candidates).await.unwrap();
        hydrator.apply(&query, &mut candidates, fetched);

        assert_eq!(
            candidates[0].quoted_tweet_text.as_deref(),
            Some("quoted text")
        );
        assert_eq!(candidates[1].quoted_tweet_text, None);
    }
}
