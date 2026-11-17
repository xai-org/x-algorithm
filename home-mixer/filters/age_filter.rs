use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::time::{Duration, SystemTime};
use xai_candidate_pipeline::component_library::utils::time_from_id_opt;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct AgeFilter {
    pub max_age: Duration,
}

impl AgeFilter {
    pub fn new(max_age: Duration) -> Self {
        Self { max_age }
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for AgeFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let now = SystemTime::now();
        let max_age = self.max_age;
        FilterResult::partition(candidates, |c| {
            time_from_id_opt(c.tweet_id)
                .and_then(|created| now.duration_since(created).ok())
                .is_some_and(|age| age <= max_age)
        })
    }
}
