use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_post_text::{MatchTweetGroup, TokenSequence, TweetTokenizer, UserMutes};

pub struct ViewerMutedKeywordFilter {
    pub tokenizer: Arc<TweetTokenizer>,
}

impl ViewerMutedKeywordFilter {
    pub fn new() -> Self {
        let tokenizer = TweetTokenizer::new();
        Self {
            tokenizer: Arc::new(tokenizer),
        }
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for ViewerMutedKeywordFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let muted_keywords = query.user_features.muted_keywords.clone();

        if muted_keywords.is_empty() {
            return FilterResult {
                kept: candidates,
                removed: vec![],
            };
        }

        let tokenizer = self.tokenizer.clone();
        tokio::task::block_in_place(|| {
            let tokenized = muted_keywords.iter().map(|k| tokenizer.tokenize(k));
            let token_sequences: Vec<TokenSequence> = tokenized.collect::<Vec<_>>();
            let user_mutes = UserMutes::new(token_sequences);
            let matcher = MatchTweetGroup::new(user_mutes);

            FilterResult::partition(candidates, |candidate| {
                let tweet_text_token_sequence = tokenizer.tokenize(&candidate.tweet_text);
                !matcher.matches(&tweet_text_token_sequence)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::PostCandidate;
    use crate::models::query::ScoredPostsQuery;
    use crate::models::user_features::UserFeatures;

    fn create_test_candidate(tweet_id: u64, tweet_text: &str) -> PostCandidate {
        PostCandidate {
            tweet_id,
            tweet_text: tweet_text.to_string(),
            author_id: 12345,
            ..Default::default()
        }
    }

    fn create_test_query(muted_keywords: Vec<String>) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_features: UserFeatures {
                muted_keywords,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_no_muted_keywords() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec![]);

        let candidates = vec![
            create_test_candidate(1, "This is spam content"),
            create_test_candidate(2, "This is good content"),
        ];

        let result = filter.filter(&query, candidates.clone());

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_keyword_match() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["spam".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "This is spam content"),
            create_test_candidate(2, "This is good content"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 2);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_hashtag_keyword_without_hash() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["widget".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "#widget launch event"),
            create_test_candidate(2, "widget speaks at summit"),
            create_test_candidate(3, "unrelated product news"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 3);
        assert_eq!(result.removed.len(), 2);
        assert!(result.removed.iter().any(|c| c.tweet_id == 1));
        assert!(result.removed.iter().any(|c| c.tweet_id == 2));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_hashtag_keyword_with_hash() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["#launch".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "#LAUNCH day"),
            create_test_candidate(2, "support launch movement"),
            create_test_candidate(3, "unrelated tweet"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_mention_keyword() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["exampleuser".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "Hey @exampleuser check this out"),
            create_test_candidate(2, "exampleuser posted something"),
            create_test_candidate(3, "different user posting"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 3);
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_word_phrase() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["crypto scam".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "This is a crypto scam warning"),
            create_test_candidate(2, "crypto is great, scam artists are bad"),
            create_test_candidate(3, "unrelated content"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multiple_muted_keywords() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec![
            "spam".to_string(),
            "scam".to_string(),
            "#blocked".to_string(),
        ]);

        let candidates = vec![
            create_test_candidate(1, "This is spam content"),
            create_test_candidate(2, "This is a scam alert"),
            create_test_candidate(3, "#blocked user posting"),
            create_test_candidate(4, "This is good content"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 4);
        assert_eq!(result.removed.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_case_insensitive_matching() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["SPAM".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "This is SPAM content"),
            create_test_candidate(2, "This is spam content"),
            create_test_candidate(3, "This is SpAm content"),
            create_test_candidate(4, "This is good content"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 4);
        assert_eq!(result.removed.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_unicode_and_accents() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["café".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "I love café culture"),
            create_test_candidate(2, "I love cafe culture"),
            create_test_candidate(3, "I love coffee culture"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 3);
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_empty_candidates() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["spam".to_string()]);

        let candidates = vec![];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 0);
        assert_eq!(result.removed.len(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_all_candidates_removed() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["spam".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "spam spam spam"),
            create_test_candidate(2, "more spam here"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 0);
        assert_eq!(result.removed.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cjk_keyword_whole_token_only() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["京都".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "東京都に行くのが楽しみ"),
            create_test_candidate(2, "I visited 京都 last week"),
            create_test_candidate(3, "unrelated content"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 2);
        assert_eq!(result.kept.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_punctuation_handling() {
        let filter = ViewerMutedKeywordFilter::new();
        let query = create_test_query(vec!["bitcoin".to_string()]);

        let candidates = vec![
            create_test_candidate(1, "Buy bitcoin! It's great!!!"),
            create_test_candidate(2, "bitcoin, ethereum, and more"),
            create_test_candidate(3, "(bitcoin is volatile)"),
            create_test_candidate(4, "stocks and bonds"),
        ];

        let result = filter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 4);
        assert_eq!(result.removed.len(), 3);
    }
}
