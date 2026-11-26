use crate::filters::drop_duplicates_filter::deduplicate;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct RetweetDeduplicationFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for RetweetDeduplicationFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        deduplicate(candidates, |candidate| {
            candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id)
        })
    }
}

#[cfg(test)]
mod dedup_benchmark {
    use super::*;
    use crate::filters::drop_duplicates_filter::DropDuplicatesFilter;
    use crate::models::candidate::RetrievalSource;
    use xai_home_mixer_proto::ServedType;

    fn candidates() -> Vec<PostCandidate> {
        (0..3000u64)
            .map(|i| PostCandidate {
                tweet_id: 1 + if i.is_multiple_of(10) { i / 2 } else { i },
                retweeted_tweet_id: (i.is_multiple_of(9)).then_some(100_000 + i % 400),
                author_id: 1000 + i % 600,
                tweet_text: format!("dedup post {i} with timeline text café"),
                retrieval_sources: vec![
                    RetrievalSource {
                        served_type: ServedType::ForYouPhoenixRetrieval,
                        cluster: None,
                        score: Some((i % 100) as f32 / 100.0),
                        position: Some(i as u32),
                    };
                    1 + (i % 2) as usize
                ],
                ..Default::default()
            })
            .collect()
    }

    fn benchmark(name: &str, filter: &impl Filter<ScoredPostsQuery, PostCandidate>) {
        let candidates = candidates();
        let query = ScoredPostsQuery::default();
        let expected = filter.filter(&query, candidates.clone()).kept.len();
        assert!(expected > 0 && expected < candidates.len());
        let mut samples = Vec::with_capacity(50);
        for iteration in 0..55 {
            let input = candidates.clone();
            let start = std::time::Instant::now();
            let result = std::hint::black_box(filter.filter(&query, input));
            let ns = start.elapsed().as_nanos();
            assert_eq!(result.kept.len(), expected);
            assert_eq!(result.kept.len() + result.removed.len(), candidates.len());
            if iteration >= 5 {
                samples.push(ns);
                println!("{name} sample={} ns={ns}", iteration - 5);
            }
        }
        samples.sort_unstable();
        println!(
            "{name} warmup=5 samples=50 median_ns={}",
            samples[samples.len() / 2]
        );
    }

    #[test]
    fn benchmark_drop_duplicates_filter_3000() {
        benchmark("drop_duplicates_filter", &DropDuplicatesFilter);
    }

    #[test]
    fn benchmark_retweet_deduplication_filter_3000() {
        benchmark("retweet_deduplication_filter", &RetweetDeduplicationFilter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(tweet_id: u64, retweeted_tweet_id: Option<u64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            retweeted_tweet_id,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_multiple_retweets_keeps_first() {
        let filter = RetweetDeduplicationFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![
            make_candidate(1, Some(100)),
            make_candidate(2, Some(100)),
            make_candidate(3, Some(100)),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test]
    async fn test_original_before_retweets_removes_retweets() {
        let filter = RetweetDeduplicationFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![
            make_candidate(100, None),
            make_candidate(1, Some(100)),
            make_candidate(2, Some(100)),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 100);
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test]
    async fn test_retweet_before_original_removes_original() {
        let filter = RetweetDeduplicationFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![
            make_candidate(1, Some(100)),
            make_candidate(100, None),
            make_candidate(2, Some(100)),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test]
    async fn test_non_retweets_always_kept() {
        let filter = RetweetDeduplicationFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![
            make_candidate(1, None),
            make_candidate(2, None),
            make_candidate(3, None),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 3);
        assert_eq!(result.removed.len(), 0);
    }

    #[tokio::test]
    async fn test_mixed_retweets_of_different_tweets() {
        let filter = RetweetDeduplicationFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![
            make_candidate(1, Some(100)),
            make_candidate(2, Some(200)),
            make_candidate(3, Some(100)),
            make_candidate(4, Some(200)),
            make_candidate(5, Some(300)),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 3);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.kept[1].tweet_id, 2);
        assert_eq!(result.kept[2].tweet_id, 5);
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test]
    async fn test_retweet_before_original_deduplicates() {
        let filter = RetweetDeduplicationFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![make_candidate(1, Some(100)), make_candidate(100, None)];

        let result = filter.filter(&query, candidates);

        assert_eq!(
            result.kept.len(),
            1,
            "expected only one candidate to survive dedup"
        );
        assert_eq!(result.removed.len(), 1);
    }
}
