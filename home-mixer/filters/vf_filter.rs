use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_visibility_filtering::models::{Action, FilteredReason};

pub struct VFFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for VFFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (removed, kept): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| should_drop(&c.visibility_reason));

        FilterResult { kept, removed }
    }
}

fn should_drop(reason: &Option<FilteredReason>) -> bool {
    match reason {
        Some(FilteredReason::SafetyResult(safety_result)) => {
            matches!(safety_result.action, Action::Drop(_))
        }
        // Includes UnspecifiedReason, which VFCandidateHydrator stamps when the
        // primary VF lookup is Err or the id is missing from the result map.
        Some(_) => true,
        // Successful VF Allow is Ok(None) -> visibility_reason None. A lookup
        // miss is stamped above and must not land here.
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_with_reason(reason: Option<FilteredReason>) -> PostCandidate {
        PostCandidate {
            visibility_reason: reason,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn drops_when_action_is_drop() {
        let filter = VFFilter;
        let query = ScoredPostsQuery::default();

        let drop_reason =
            FilteredReason::SafetyResult(xai_visibility_filtering::models::SafetyResult {
                action: Action::Drop(Default::default()),
                ..Default::default()
            });
        let candidates = vec![
            candidate_with_reason(Some(drop_reason)),
            candidate_with_reason(None),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.kept.len(), 1);
    }

    #[tokio::test]
    async fn keeps_when_action_allows() {
        let filter = VFFilter;
        let query = ScoredPostsQuery::default();

        let allowed_reason =
            FilteredReason::SafetyResult(xai_visibility_filtering::models::SafetyResult {
                action: Action::Allow,
                ..Default::default()
            });
        let candidates = vec![candidate_with_reason(Some(allowed_reason))];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.removed.len(), 0);
        assert_eq!(result.kept.len(), 1);
    }

    #[tokio::test]
    async fn drops_when_user_filtered_state_present() {
        let filter = VFFilter;
        let query = ScoredPostsQuery::default();

        let reason = FilteredReason::AuthorBlockViewer;
        let candidates = vec![
            candidate_with_reason(Some(reason)),
            candidate_with_reason(None),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.kept.len(), 1);
    }

    #[tokio::test]
    async fn drops_unspecified_reason_from_vf_lookup_miss() {
        let filter = VFFilter;
        let query = ScoredPostsQuery::default();

        let result = filter.filter(
            &query,
            vec![
                candidate_with_reason(Some(FilteredReason::UnspecifiedReason)),
                candidate_with_reason(None),
            ],
        );

        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].visibility_reason, None);
    }
}
