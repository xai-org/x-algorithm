use crate::ads::user_load;
use crate::models::query::ScoredPostsQuery;
use crate::query_hydrators::user_demographics_query_hydrator::UserDemographicsQueryHydrator;
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

pub struct AdsUserStateQueryHydrator {
    pub inner: UserDemographicsQueryHydrator,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for AdsUserStateQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        user_load::is_enabled(query)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        self.inner.hydrate(query).await
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        self.inner.update(query, hydrated);
    }
}
