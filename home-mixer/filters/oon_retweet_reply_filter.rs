use crate::candidate_hydrators::phoenix_reply_ancestors_hydrator::is_phoenix_home_retrieval;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnablePhoenixOonReplies, PhoenixOonRepliesRootOnly};
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct OONRetweetReplyFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for OONRetweetReplyFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let allow_phoenix_oon_replies = query.params.get(EnablePhoenixOonReplies);
        let root_only = query.params.get(PhoenixOonRepliesRootOnly);
        FilterResult::partition(candidates, |c| {
            let is_reply = c.in_reply_to_tweet_id.is_some();
            let is_retweet = c.retweeted_tweet_id.is_some();
            let replies_to_root =
                c.conversation_id.is_some() && c.in_reply_to_tweet_id == c.conversation_id;
            let oon_reply_dropped = is_reply
                && !(allow_phoenix_oon_replies
                    && is_phoenix_home_retrieval(c)
                    && (!root_only || replies_to_root));
            !((c.in_network == Some(false) && (is_retweet || oon_reply_dropped))
                || (is_reply && c.ancestors.is_empty()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        tweet_id: u64,
        retweeted_tweet_id: Option<u64>,
        in_reply_to_tweet_id: Option<u64>,
        ancestors: Vec<u64>,
        in_network: Option<bool>,
    ) -> PostCandidate {
        PostCandidate {
            tweet_id,
            retweeted_tweet_id,
            in_reply_to_tweet_id,
            ancestors,
            in_network,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn drops_oon_retweets_replies_and_replies_missing_ancestors() {
        let filter = OONRetweetReplyFilter;
        let query = ScoredPostsQuery::default();

        let candidates = vec![
            candidate(1, Some(100), None, vec![], Some(false)),
            candidate(2, None, Some(200), vec![200], Some(false)),
            candidate(3, None, None, vec![], Some(false)),
            candidate(4, Some(400), None, vec![], Some(true)),
            candidate(5, None, Some(500), vec![500], Some(true)),
            candidate(6, None, None, vec![], Some(true)),
            candidate(7, Some(700), None, vec![], None),
            candidate(8, None, Some(800), vec![], Some(true)),
            candidate(9, None, Some(900), vec![], None),
        ];

        let result = filter.filter(&query, candidates);

        let removed_ids: Vec<u64> = result.removed.iter().map(|c| c.tweet_id).collect();
        assert_eq!(removed_ids, vec![1, 2, 8, 9]);

        let kept_ids: Vec<u64> = result.kept.iter().map(|c| c.tweet_id).collect();
        assert_eq!(kept_ids, vec![3, 4, 5, 6, 7]);
    }

    #[tokio::test]
    async fn integrates_with_in_network_hydrator() {
        use crate::candidate_hydrators::in_network_candidate_hydrator::InNetworkCandidateHydrator;
        use crate::models::user_features::UserFeatures;
        use xai_candidate_pipeline::hydrator::TwoPhaseHydrator;

        let query = ScoredPostsQuery {
            user_id: 1,
            user_features: UserFeatures {
                followed_user_ids: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut candidates = vec![
            PostCandidate {
                tweet_id: 10,
                author_id: 3,
                retweeted_tweet_id: Some(100),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 11,
                author_id: 3,
                in_reply_to_tweet_id: Some(200),
                ancestors: vec![200],
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 12,
                author_id: 2,
                in_reply_to_tweet_id: Some(300),
                ancestors: vec![300],
                ..Default::default()
            },
        ];

        let hydrator = InNetworkCandidateHydrator;
        let fetched = hydrator.fetch(&query, &candidates).await.expect("fetch ok");
        hydrator.apply(&query, &mut candidates, fetched);
        assert_eq!(candidates[0].in_network, Some(false));
        assert_eq!(candidates[1].in_network, Some(false));
        assert_eq!(candidates[2].in_network, Some(true));

        let result = OONRetweetReplyFilter.filter(&query, candidates);
        let removed_ids: Vec<u64> = result.removed.iter().map(|c| c.tweet_id).collect();
        assert_eq!(removed_ids, vec![10, 11]);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 12);
    }
}
