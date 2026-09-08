use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnablePreferOriginalRetweetDedup;
use rustc_hash::FxHashSet;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct RetweetDeduplicationFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for RetweetDeduplicationFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        if query.params.get(EnablePreferOriginalRetweetDedup) {
            prefer_original_dedup(candidates)
        } else {
            first_wins_dedup(candidates)
        }
    }
}

/// Legacy For You behavior: first candidate for a source_id wins.
fn first_wins_dedup(candidates: Vec<PostCandidate>) -> FilterResult<PostCandidate> {
    let mut seen_tweet_ids: FxHashSet<u64> =
        FxHashSet::with_capacity_and_hasher(candidates.len(), Default::default());
    let mut kept = Vec::with_capacity(candidates.len());
    let mut removed = Vec::new();

    for candidate in candidates {
        let dedup_id = candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id);
        if seen_tweet_ids.insert(dedup_id) {
            kept.push(candidate);
        } else {
            removed.push(candidate);
        }
    }

    FilterResult { kept, removed }
}

/// Following parity: prefer native/original over RT for the same source.
fn prefer_original_dedup(candidates: Vec<PostCandidate>) -> FilterResult<PostCandidate> {
    let (retweets, native_tweets): (Vec<_>, Vec<_>) = candidates
        .iter()
        .cloned()
        .partition(|c| c.retweeted_tweet_id.is_some());

    let mut seen_tweet_ids: HashSet<u64> = HashSet::with_capacity(candidates.len());
    let mut kept_ids: HashSet<u64> = HashSet::with_capacity(candidates.len());
    for native in &native_tweets {
        seen_tweet_ids.insert(native.tweet_id);
        kept_ids.insert(native.tweet_id);
    }

    for retweet in &retweets {
        let source_id = retweet.retweeted_tweet_id.expect("partitioned as retweet");
        let ids = [retweet.tweet_id, source_id];
        if ids.iter().any(|id| seen_tweet_ids.contains(id)) {
            continue;
        }
        seen_tweet_ids.extend(ids);
        kept_ids.insert(retweet.tweet_id);
    }

    let mut kept = Vec::with_capacity(kept_ids.len());
    let mut removed = Vec::new();
    for candidate in candidates {
        if kept_ids.contains(&candidate.tweet_id) {
            kept.push(candidate);
        } else {
            removed.push(candidate);
        }
    }

    FilterResult { kept, removed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::{FeatureSwitches, Params, RecipientBuilder};

    fn make_candidate(tweet_id: u64, retweeted_tweet_id: Option<u64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            retweeted_tweet_id,
            ..Default::default()
        }
    }

    fn params_prefer_original(enable: bool) -> Params {
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_prefer_original_retweet_dedup".to_string(),
            if enable { "true" } else { "false" },
        );
        results.into()
    }

    fn query_with(enable_prefer_original: bool) -> ScoredPostsQuery {
        ScoredPostsQuery {
            params: params_prefer_original(enable_prefer_original),
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

    #[tokio::test]
    async fn test_prefer_original_flag_off_keeps_first_wins() {
        let filter = RetweetDeduplicationFilter;
        let query = query_with(false);
        let candidates = vec![make_candidate(1, Some(100)), make_candidate(100, None)];
        let result = filter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
    }

    #[tokio::test]
    async fn test_prefer_original_flag_on_original_wins_even_if_rt_first() {
        let filter = RetweetDeduplicationFilter;
        let query = query_with(true);
        let candidates = vec![
            make_candidate(1, Some(100)),
            make_candidate(100, None),
            make_candidate(2, Some(100)),
        ];
        let result = filter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 100);
        assert!(result.kept[0].retweeted_tweet_id.is_none());
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test]
    async fn test_prefer_original_flag_on_keeps_one_rt_when_no_native() {
        let filter = RetweetDeduplicationFilter;
        let query = query_with(true);
        let candidates = vec![
            make_candidate(1, Some(100)),
            make_candidate(2, Some(100)),
        ];
        let result = filter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
    }
}
