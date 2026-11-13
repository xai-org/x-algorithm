use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};

pub struct LanguageCodeHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: QuickCache<u64, Option<String>>,
}

impl LanguageCodeHydrator {
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        let cache = default_quick_cache();
        Self { tes_client, cache }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for LanguageCodeHydrator {
    type Fetched = Vec<Option<Option<String>>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<Option<String>>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<Option<String>>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<Option<String>>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            if let Some(value) = value {
                candidate.language_code = value;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.language_code = hydrated.language_code;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for LanguageCodeHydrator {
    type CacheKey = u64;
    type Value = Option<String>;

    fn cache_store(&self) -> &dyn CacheStore<u64, Option<String>> {
        &self.cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.get_original_tweet_id()
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<Option<String>>> {
        let tweet_ids: Vec<u64> = misses.iter().map(|c| c.get_original_tweet_id()).collect();
        let language_codes = self.tes_client.get_language_code(tweet_ids.clone()).await;
        tweet_ids
            .iter()
            .map(|tweet_id| match language_codes.get(tweet_id) {
                Some(Ok(language_code)) => Some(language_code.clone()),
                Some(Err(_)) | None => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::tweet_entity_service_client::MockTESClient;
    use std::collections::HashMap;
    use xai_candidate_pipeline::hydrator::Hydrator;

    #[tokio::test]
    async fn test_language_code_found() {
        let mut language_code = HashMap::new();
        language_code.insert(1001u64, Some("en".to_string()));

        let client = Arc::new(MockTESClient {
            language_code,
            ..Default::default()
        });
        let hydrator = LanguageCodeHydrator::new(client as Arc<dyn TESClient + Send + Sync>).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1001,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        assert_eq!(
            result[0].as_ref().unwrap().language_code,
            Some("en".to_string())
        );
    }

    #[tokio::test]
    async fn test_language_code_none() {
        let mut language_code = HashMap::new();
        language_code.insert(1002u64, None);

        let client = Arc::new(MockTESClient {
            language_code,
            ..Default::default()
        });
        let hydrator = LanguageCodeHydrator::new(client as Arc<dyn TESClient + Send + Sync>).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1002,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        assert_eq!(result[0].as_ref().unwrap().language_code, None);
    }

    #[tokio::test]
    async fn test_language_code_caching() {
        let mut language_code = HashMap::new();
        language_code.insert(1003u64, Some("ja".to_string()));

        let client = Arc::new(MockTESClient {
            language_code,
            ..Default::default()
        });
        let hydrator = LanguageCodeHydrator::new(client as Arc<dyn TESClient + Send + Sync>).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1003,
            ..Default::default()
        }];

        let first = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        assert_eq!(
            first[0].as_ref().unwrap().language_code,
            Some("ja".to_string())
        );

        let second = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        assert_eq!(
            second[0].as_ref().unwrap().language_code,
            Some("ja".to_string())
        );
    }

    #[tokio::test]
    async fn test_language_code_multiple_candidates() {
        let mut language_code = HashMap::new();
        language_code.insert(2001u64, Some("en".to_string()));
        language_code.insert(2002u64, Some("fr".to_string()));
        language_code.insert(2003u64, None);

        let client = Arc::new(MockTESClient {
            language_code,
            ..Default::default()
        });
        let hydrator = LanguageCodeHydrator::new(client as Arc<dyn TESClient + Send + Sync>).await;

        let candidates = vec![
            PostCandidate {
                tweet_id: 2001,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2002,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2003,
                ..Default::default()
            },
        ];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].as_ref().unwrap().language_code,
            Some("en".to_string())
        );
        assert_eq!(
            result[1].as_ref().unwrap().language_code,
            Some("fr".to_string())
        );
        assert_eq!(result[2].as_ref().unwrap().language_code, None);
    }
}
