use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct IneligibleSubscriptionFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for IneligibleSubscriptionFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let subscribed_user_ids: HashSet<u64> = query
            .user_features
            .subscribed_user_ids
            .iter()
            .map(|id| *id as u64)
            .collect();

        FilterResult::partition(candidates, |candidate| {
            match candidate.subscription_author_id {
                Some(author_id) => subscribed_user_ids.contains(&author_id),
                None => true,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(subscription_author_id: Option<u64>) -> PostCandidate {
        PostCandidate {
            subscription_author_id,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn keeps_only_subscribed_authors() {
        let filter = IneligibleSubscriptionFilter;
        let mut query = ScoredPostsQuery::default();
        query.user_features.subscribed_user_ids = vec![1, 2];

        let candidates = vec![candidate(Some(1)), candidate(Some(3)), candidate(None)];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 1);
        assert!(result
            .kept
            .iter()
            .any(|c| c.subscription_author_id == Some(1)));
        assert!(result
            .kept
            .iter()
            .any(|c| c.subscription_author_id.is_none()));
        assert!(result
            .removed
            .iter()
            .any(|c| c.subscription_author_id == Some(3)));
    }
}
