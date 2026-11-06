use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct SelfTweetFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for SelfTweetFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let viewer_id = query.user_id;
        FilterResult::partition(candidates, |c| c.author_id != viewer_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_candidate_pipeline::candidate_pipeline::PipelineStage;

    #[test]
    fn all_kept_filter_preserves_the_input_allocation_and_payloads() {
        let query = ScoredPostsQuery {
            user_id: 42,
            ..Default::default()
        };
        let input: Vec<_> = (0..3000)
            .map(|i| PostCandidate {
                tweet_id: i,
                author_id: 1000 + i % 600,
                tweet_text: format!("post-{i}"),
                ancestors: vec![i + 10_000],
                score: Some(i as f64),
                ..Default::default()
            })
            .collect();
        let allocation = input.as_ptr();
        let expected = serde_json::to_value(&input).unwrap();
        let result = SelfTweetFilter.run(&query, input, PipelineStage::Filter);
        assert_eq!(result.kept.as_ptr(), allocation);
        assert_eq!(serde_json::to_value(&result.kept).unwrap(), expected);
        assert!(result.removed.is_empty());
    }

    #[test]
    fn mixed_and_all_removed_filters_keep_both_partitions_stable() {
        let query = ScoredPostsQuery {
            user_id: 42,
            ..Default::default()
        };
        for authors in [vec![42, 5, 42, 7, 42, 9], vec![42, 42], vec![]] {
            let input: Vec<_> = authors
                .iter()
                .enumerate()
                .map(|(i, author)| PostCandidate {
                    tweet_id: i as u64,
                    author_id: *author,
                    tweet_text: format!("post-{i}"),
                    ..Default::default()
                })
                .collect();
            let expected_kept: Vec<_> = input
                .iter()
                .filter(|c| c.author_id != query.user_id)
                .map(|c| c.tweet_id)
                .collect();
            let expected_removed: Vec<_> = input
                .iter()
                .filter(|c| c.author_id == query.user_id)
                .map(|c| c.tweet_id)
                .collect();
            let result = SelfTweetFilter.run(&query, input, PipelineStage::Filter);
            assert_eq!(
                result.kept.iter().map(|c| c.tweet_id).collect::<Vec<_>>(),
                expected_kept
            );
            assert_eq!(
                result
                    .removed
                    .iter()
                    .map(|c| c.tweet_id)
                    .collect::<Vec<_>>(),
                expected_removed
            );
        }
    }
}
