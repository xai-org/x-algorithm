use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    ControlledSourceReexposureMaxServesK, EnableControlledSourceReexposure,
    EnableServedFilterAllRequests,
};
use crate::util::candidates_util::related_post_ids_iter;
use std::collections::HashSet;
use xai_candidate_pipeline::component_library::utils::client_utils::RequestContext::{
    self, ForegroundTruncate,
};
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct PreviouslyServedPostsFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslyServedPostsFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        let req_context = RequestContext::parse(&query.request_context);
        let enable_all = query.params.get(EnableServedFilterAllRequests);

        enable_all || (query.is_bottom_request && req_context != ForegroundTruncate)
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        // PR-F5: when ControlledSourceReexposure is armed (Enable && K>1), defer
        // served burn to ControlledSourceReexposureFilter (wired immediately after).
        // Enable=false or K<=1 keeps today's drop-on-served behavior.
        if query.params.get(EnableControlledSourceReexposure)
            && query.params.get(ControlledSourceReexposureMaxServesK) > 1
        {
            return FilterResult {
                kept: candidates,
                removed: Vec::new(),
            };
        }

        let served_ids: HashSet<u64> = query.served_ids.iter().copied().collect();

        let (removed, kept): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| related_post_ids_iter(c).any(|id| served_ids.contains(&id)));

        FilterResult { kept, removed }
    }
}
