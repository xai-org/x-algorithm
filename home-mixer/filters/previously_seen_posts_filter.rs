use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::util::candidates_util::related_post_ids_iter;
use std::collections::HashSet;
use xai_candidate_pipeline::component_library::utils::BloomFilter;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct PreviouslySeenPostsFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslySeenPostsFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let bloom_filters = query
            .bloom_filter_entries
            .iter()
            .map(|e| BloomFilter::from_parts(e.size_cap, e.false_positive_rate, &e.bloom_filter))
            .collect::<Vec<_>>();

        let seen_ids: HashSet<u64> = query.seen_ids.iter().copied().collect();

        FilterResult::partition(candidates, |c| {
            !related_post_ids_iter(c).any(|post_id| {
                seen_ids.contains(&post_id)
                    || bloom_filters
                        .iter()
                        .any(|filter| filter.may_contain(post_id))
            })
        })
    }
}
