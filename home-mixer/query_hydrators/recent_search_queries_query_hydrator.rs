use crate::clients::recent_search_queries_store_client::RecentSearchQueriesStoreClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableRecentSearchQueriesHydration;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::{RecentSearchQueries, RecentSearchQuery};

const MH_GET_TIMEOUT: Duration = Duration::from_millis(300);
const MAX_QUERIES: usize = 16;
const EMBEDDING_BYTES: usize = 1024 * 2;

pub struct RecentSearchQueriesQueryHydrator {
    client: Arc<dyn RecentSearchQueriesStoreClient>,
}

impl RecentSearchQueriesQueryHydrator {
    pub fn new(client: Arc<dyn RecentSearchQueriesStoreClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for RecentSearchQueriesQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        (query.params.get(EnableRecentSearchQueriesHydration) || query.is_shadow_traffic)
            && query.user_id != 0
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let list = tokio::time::timeout(MH_GET_TIMEOUT, self.client.fetch(query.user_id))
            .await
            .map_err(|_| "RecentSearchQueries MH get timed out".to_string())??;
        Ok(ScoredPostsQuery {
            recent_search_queries: list.map(|l| select(l, query.request_time_ms)),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.recent_search_queries = hydrated.recent_search_queries;
    }
}

fn select(list: RecentSearchQueries, cutoff_ms: i64) -> RecentSearchQueries {
    let mut queries: Vec<RecentSearchQuery> = list
        .queries
        .into_iter()
        .filter(|q| q.timestamp_ms < cutoff_ms && q.embedding_bf16.len() == EMBEDDING_BYTES)
        .map(|q| RecentSearchQuery {
            query: String::new(),
            ..q
        })
        .collect();
    queries.sort_by_key(|q| std::cmp::Reverse(q.timestamp_ms));
    queries.truncate(MAX_QUERIES);
    RecentSearchQueries { queries }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::phoenix_request::build_user_context;

    fn entry(timestamp_ms: i64, embedding_bytes: usize) -> RecentSearchQuery {
        RecentSearchQuery {
            query: format!("q{timestamp_ms}"),
            timestamp_ms,
            embedding_bf16: vec![1u8; embedding_bytes].into(),
        }
    }

    #[test]
    fn select_matches_training_and_drops_text() {
        let request_time_ms = 1_000;
        let after_request = entry(5_000, EMBEDDING_BYTES);
        let wrong_embedding_size = entry(150, 8);
        let mut queries: Vec<_> = (0..20).map(|t| entry(100 + t, EMBEDDING_BYTES)).collect();
        queries.extend([after_request, wrong_embedding_size]);
        let out = select(RecentSearchQueries { queries }, request_time_ms);

        let ts: Vec<i64> = out.queries.iter().map(|q| q.timestamp_ms).collect();
        assert_eq!(ts, (104..120).rev().collect::<Vec<_>>());
        assert!(out.queries.iter().all(|q| q.query.is_empty()));
        assert!(out
            .queries
            .iter()
            .all(|q| q.embedding_bf16.len() == EMBEDDING_BYTES));
    }

    struct FakeStore(RecentSearchQueries);

    #[async_trait]
    impl RecentSearchQueriesStoreClient for FakeStore {
        async fn fetch(&self, _user_id: u64) -> Result<Option<RecentSearchQueries>, String> {
            Ok(Some(self.0.clone()))
        }
    }

    #[tokio::test]
    async fn hydrated_searches_reach_the_phoenix_user_context() {
        let stored = RecentSearchQueries {
            queries: vec![entry(900, EMBEDDING_BYTES), entry(2_000, EMBEDDING_BYTES)],
        };
        let hydrator = RecentSearchQueriesQueryHydrator::new(Arc::new(FakeStore(stored)));
        let mut query = ScoredPostsQuery {
            user_id: 7,
            request_time_ms: 1_000,
            ..Default::default()
        };
        let hydrated = hydrator.hydrate(&query).await.unwrap();
        hydrator.update(&mut query, hydrated);

        let sent = build_user_context(&query)
            .unwrap()
            .recent_search_queries
            .unwrap();
        assert_eq!(sent.queries.len(), 1);
        assert_eq!(sent.queries[0].timestamp_ms, 900);
        assert!(sent.queries[0].query.is_empty());
    }
}
