use rustc_hash::FxHashSet;

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct AuthorSocialgraphFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for AuthorSocialgraphFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let viewer_blocked_user_ids: FxHashSet<i64> = query
            .user_features
            .blocked_user_ids
            .iter()
            .copied()
            .collect();
        let viewer_muted_user_ids: FxHashSet<i64> =
            query.user_features.muted_user_ids.iter().copied().collect();

        let mut kept: Vec<PostCandidate> = Vec::with_capacity(candidates.len());
        let mut removed: Vec<PostCandidate> = Vec::new();

        for candidate in candidates {
            let author_id = candidate.author_id as i64;
            let muted = viewer_muted_user_ids.contains(&author_id);
            let blocked = viewer_blocked_user_ids.contains(&author_id);
            let author_blocks_viewer = candidate.author_blocks_viewer.unwrap_or(false);

            let quoted_author_blocks_viewer =
                candidate.quoted_author_blocks_viewer.unwrap_or(false);
            let viewer_blocks_quoted_author = candidate
                .quoted_user_id
                .map(|uid| viewer_blocked_user_ids.contains(&(uid as i64)))
                .unwrap_or(false);

            let viewer_blocks_retweeted_user = candidate
                .retweeted_user_id
                .map(|uid| viewer_blocked_user_ids.contains(&(uid as i64)))
                .unwrap_or(false);
            let viewer_mutes_quoted_author = candidate
                .quoted_user_id
                .map(|uid| viewer_muted_user_ids.contains(&(uid as i64)))
                .unwrap_or(false);

            if muted
                || blocked
                || author_blocks_viewer
                || quoted_author_blocks_viewer
                || viewer_blocks_quoted_author
                || viewer_blocks_retweeted_user
                || viewer_mutes_quoted_author
            {
                removed.push(candidate);
            } else {
                kept.push(candidate);
            }
        }

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::user_features::UserFeatures;

    fn make_candidate(tweet_id: u64, author_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id,
            ..Default::default()
        }
    }

    fn make_query_with_features(user_features: UserFeatures) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_features,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_no_blocked_or_muted_users_keeps_all() {
        let filter = AuthorSocialgraphFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 3);
        assert_eq!(result.removed.len(), 0);
    }

    #[tokio::test]
    async fn test_blocked_user_is_removed() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            blocked_user_ids: vec![200],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].author_id, 100);
        assert_eq!(result.kept[1].author_id, 300);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].author_id, 200);
    }

    #[tokio::test]
    async fn test_muted_user_is_removed() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            muted_user_ids: vec![200],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].author_id, 100);
        assert_eq!(result.kept[1].author_id, 300);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].author_id, 200);
    }

    #[tokio::test]
    async fn test_multiple_blocked_users_are_removed() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            blocked_user_ids: vec![100, 300],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
            make_candidate(4, 400),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].author_id, 200);
        assert_eq!(result.kept[1].author_id, 400);
        assert_eq!(result.removed.len(), 2);
        assert_eq!(result.removed[0].author_id, 100);
        assert_eq!(result.removed[1].author_id, 300);
    }

    #[tokio::test]
    async fn test_multiple_muted_users_are_removed() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            muted_user_ids: vec![100, 300],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
            make_candidate(4, 400),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].author_id, 200);
        assert_eq!(result.kept[1].author_id, 400);
        assert_eq!(result.removed.len(), 2);
        assert_eq!(result.removed[0].author_id, 100);
        assert_eq!(result.removed[1].author_id, 300);
    }

    #[tokio::test]
    async fn test_both_blocked_and_muted_users_are_removed() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            blocked_user_ids: vec![100],
            muted_user_ids: vec![300],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
            make_candidate(4, 400),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].author_id, 200);
        assert_eq!(result.kept[1].author_id, 400);
        assert_eq!(result.removed.len(), 2);
        assert_eq!(result.removed[0].author_id, 100);
        assert_eq!(result.removed[1].author_id, 300);
    }

    #[tokio::test]
    async fn test_user_in_both_blocked_and_muted_lists() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            blocked_user_ids: vec![200],
            muted_user_ids: vec![200],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].author_id, 200);
    }

    #[tokio::test]
    async fn test_empty_candidates_list() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            blocked_user_ids: vec![100],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 0);
        assert_eq!(result.removed.len(), 0);
    }

    #[tokio::test]
    async fn test_all_candidates_blocked() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            blocked_user_ids: vec![100, 200, 300],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let candidates = vec![
            make_candidate(1, 100),
            make_candidate(2, 200),
            make_candidate(3, 300),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 0);
        assert_eq!(result.removed.len(), 3);
    }


    #[tokio::test]
    async fn test_muted_quoted_author_is_removed() {
        let filter = AuthorSocialgraphFilter;
        let user_features = UserFeatures {
            muted_user_ids: vec![200],
            ..Default::default()
        };
        let query = make_query_with_features(user_features);

        let mut quote = make_candidate(1, 100);
        quote.quoted_user_id = Some(200);

        let candidates = vec![quote, make_candidate(3, 300)];
        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].author_id, 300);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].quoted_user_id, Some(200));
    }

    #[tokio::test]
    async fn test_unmuted_quoted_author_is_kept() {
        let filter = AuthorSocialgraphFilter;
        let query = make_query_with_features(UserFeatures::default());

        let mut quote = make_candidate(1, 100);
        quote.quoted_user_id = Some(200);

        let result = filter.filter(&query, vec![quote]);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].quoted_user_id, Some(200));
        assert!(result.removed.is_empty());
    }
    #[tokio::test]
    async fn test_author_blocks_viewer() {
        let filter = AuthorSocialgraphFilter;
        let query = ScoredPostsQuery::default();

        let mut candidate_blocked = make_candidate(1, 100);
        candidate_blocked.author_blocks_viewer = Some(true);

        let candidates = vec![
            candidate_blocked,
            make_candidate(2, 200),
            make_candidate(3, 300),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].author_id, 200);
        assert_eq!(result.kept[1].author_id, 300);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].author_id, 100);
    }

    #[tokio::test]
    async fn test_multiple_authors_block_viewer() {
        let filter = AuthorSocialgraphFilter;
        let query = ScoredPostsQuery::default();

        let mut candidate_blocked_1 = make_candidate(1, 100);
        candidate_blocked_1.author_blocks_viewer = Some(true);

        let mut candidate_blocked_2 = make_candidate(3, 300);
        candidate_blocked_2.author_blocks_viewer = Some(true);

        let candidates = vec![
            candidate_blocked_1,
            make_candidate(2, 200),
            candidate_blocked_2,
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].author_id, 200);
        assert_eq!(result.removed.len(), 2);
        assert_eq!(result.removed[0].author_id, 100);
        assert_eq!(result.removed[1].author_id, 300);
    }
}
