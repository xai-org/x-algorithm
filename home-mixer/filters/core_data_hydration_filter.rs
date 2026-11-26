use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct CoreDataHydrationFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for CoreDataHydrationFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        FilterResult::partition(candidates, |c| c.author_id != 0)
    }
}
