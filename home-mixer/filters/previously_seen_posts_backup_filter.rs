use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::util::candidates_util::related_post_ids_iter;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct PreviouslySeenPostsBackupFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslySeenPostsBackupFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        if query.impressed_post_ids.is_empty() {
            return FilterResult {
                kept: candidates,
                removed: Vec::new(),
            };
        }

        let impressed_ids: HashSet<u64> = query.impressed_post_ids.iter().copied().collect();

        FilterResult::partition(candidates, |c| {
            !related_post_ids_iter(c).any(|id| impressed_ids.contains(&id))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(tweet_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            ..Default::default()
        }
    }

    #[test]
    fn test_filters_out_impressed_posts() {
        let query = ScoredPostsQuery {
            impressed_post_ids: vec![1, 2, 3],
            ..Default::default()
        };

        let candidates = vec![
            make_candidate(1),
            make_candidate(4),
            make_candidate(2),
            make_candidate(5),
        ];

        let result = PreviouslySeenPostsBackupFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 2);
        assert!(result
            .kept
            .iter()
            .all(|c| c.tweet_id == 4 || c.tweet_id == 5));
    }

    #[test]
    fn test_no_impressed_posts_keeps_all() {
        let query = ScoredPostsQuery {
            impressed_post_ids: Vec::new(),
            ..Default::default()
        };

        let candidates = vec![make_candidate(1), make_candidate(2)];

        let result = PreviouslySeenPostsBackupFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 2);
        assert!(result.removed.is_empty());
    }

    #[test]
    fn test_empty_candidates() {
        let query = ScoredPostsQuery {
            impressed_post_ids: vec![1, 2],
            ..Default::default()
        };

        let result = PreviouslySeenPostsBackupFilter.filter(&query, Vec::new());
        assert!(result.kept.is_empty());
        assert!(result.removed.is_empty());
    }
}
