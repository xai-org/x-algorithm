use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct VideoFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for VideoFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.exclude_videos
    }

    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        FilterResult::partition(candidates, |c| !has_video(c))
    }
}

fn has_video(candidate: &PostCandidate) -> bool {
    candidate.min_video_duration_ms.is_some() || candidate.quoted_max_video_duration_ms.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_by_default() {
        let query = ScoredPostsQuery::default();
        assert!(!VideoFilter.enable(&query));
    }

    #[test]
    fn test_enabled_when_set() {
        let query = ScoredPostsQuery {
            exclude_videos: true,
            ..Default::default()
        };
        assert!(VideoFilter.enable(&query));
    }

    #[test]
    fn test_removes_video_posts() {
        let query = ScoredPostsQuery {
            exclude_videos: true,
            ..Default::default()
        };

        let candidates = vec![
            PostCandidate {
                tweet_id: 1,
                min_video_duration_ms: Some(5000),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                min_video_duration_ms: None,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 3,
                min_video_duration_ms: Some(0),
                ..Default::default()
            },
        ];

        let result = VideoFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 2);
        assert_eq!(result.removed.len(), 2);
    }

    #[test]
    fn test_removes_quotes_of_video_posts() {
        let query = ScoredPostsQuery {
            exclude_videos: true,
            ..Default::default()
        };

        let candidates = vec![
            PostCandidate {
                tweet_id: 1,
                quoted_tweet_id: Some(10),
                quoted_max_video_duration_ms: Some(5000),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                quoted_tweet_id: Some(20),
                quoted_has_photo: Some(true),
                ..Default::default()
            },
        ];

        let result = VideoFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 2);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 1);
    }

    #[test]
    fn test_keeps_all_when_no_videos() {
        let query = ScoredPostsQuery {
            exclude_videos: true,
            ..Default::default()
        };

        let candidates = vec![
            PostCandidate {
                tweet_id: 1,
                min_video_duration_ms: None,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                min_video_duration_ms: None,
                ..Default::default()
            },
        ];

        let result = VideoFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 2);
        assert!(result.removed.is_empty());
    }
}
