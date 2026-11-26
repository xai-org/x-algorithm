use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_post_text::{MatchTweetGroup, TokenSequence, TweetTokenizer, UserMutes};

pub struct FollowingViewerMutedKeywordFilter {
    pub tokenizer: Arc<TweetTokenizer>,
}

impl FollowingViewerMutedKeywordFilter {
    pub fn new() -> Self {
        Self {
            tokenizer: Arc::new(TweetTokenizer::new()),
        }
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for FollowingViewerMutedKeywordFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let muted_keywords = &query.user_features.muted_keywords;
        if muted_keywords.is_empty() {
            return FilterResult {
                kept: candidates,
                removed: vec![],
            };
        }

        let tokenizer = Arc::clone(&self.tokenizer);
        tokio::task::block_in_place(|| {
            let token_sequences: Vec<TokenSequence> = muted_keywords
                .iter()
                .map(|k| tokenizer.tokenize(k))
                .collect();
            let matcher = MatchTweetGroup::new(UserMutes::new(token_sequences));
            let use_ascii_prefilter = matcher.ascii_prefilter_can_reject();
            FilterResult::partition(candidates, |candidate| {
                !candidate_matches(candidate, &tokenizer, &matcher, use_ascii_prefilter)
            })
        })
    }
}

fn candidate_matches(
    candidate: &PostCandidate,
    tokenizer: &TweetTokenizer,
    matcher: &MatchTweetGroup,
    use_ascii_prefilter: bool,
) -> bool {
    std::iter::once(candidate.tweet_text.as_str())
        .chain(candidate.quoted_tweet_text.as_deref())
        .chain(candidate.ancestor_texts.values().map(String::as_str))
        .filter(|text| !text.is_empty())
        .any(|text| {
            if use_ascii_prefilter && !matcher.may_match_ascii_text(text) {
                return false;
            }
            matcher.matches(&tokenizer.tokenize(text))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::user_features::UserFeatures;

    fn query(muted: &[&str]) -> ScoredPostsQuery {
        ScoredPostsQuery {
            user_features: UserFeatures {
                muted_keywords: muted.iter().map(|s| (*s).to_string()).collect(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn post(tweet_id: u64, text: &str) -> PostCandidate {
        PostCandidate {
            tweet_id,
            tweet_text: text.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn drops_focal_quote_and_ancestor_hits_and_keeps_the_rest() {
        let filter = FollowingViewerMutedKeywordFilter::new();
        let mut quoted = post(2, "clean focal");
        quoted.quoted_tweet_text = Some("a crypto scam alert".to_string());
        let mut ancestor = post(3, "clean focal");
        ancestor
            .ancestor_texts
            .insert(9, "please #blocked this".to_string());
        let candidates = vec![
            post(1, "this is spam"),
            quoted,
            ancestor,
            post(4, "nothing to see here"),
            post(5, "weather and football"),
        ];
        let result = filter.filter(&query(&["spam", "crypto scam", "#blocked"]), candidates);
        let kept: Vec<_> = result.kept.iter().map(|c| c.tweet_id).collect();
        let removed: Vec<_> = result.removed.iter().map(|c| c.tweet_id).collect();
        assert_eq!(kept, vec![4, 5]);
        assert_eq!(removed, vec![1, 2, 3]);
    }

    #[test]
    fn empty_mute_list_keeps_every_candidate() {
        let filter = FollowingViewerMutedKeywordFilter::new();
        let candidates = vec![post(1, "spam"), post(2, "ok")];
        let result = filter.filter(&query(&[]), candidates);
        assert_eq!(result.kept.len(), 2);
        assert!(result.removed.is_empty());
    }
}
