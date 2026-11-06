use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::time::Duration;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct AgeFilter {
    pub max_age: Duration,
}

impl AgeFilter {
    pub fn new(max_age: Duration) -> Self {
        Self { max_age }
    }

    fn is_within_age(&self, tweet_id: u64) -> bool {
        duration_since_creation_opt(tweet_id)
            .map(|age| age <= self.max_age)
            .unwrap_or(false)
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for AgeFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        FilterResult::partition(candidates, |c| self.is_within_age(c.tweet_id))
    }
}
