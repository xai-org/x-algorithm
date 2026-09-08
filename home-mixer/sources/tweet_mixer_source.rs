use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableTweetMixerSource, TweetMixerMaxResults};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::tweet_mixer_client::TweetMixerClient;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_x_thrift::tweet_mixer::{
    ClientContext, HomeRecommendedTweetsProductContext, Product, ProductContext, TweetMixerRequest,
};

const MAX_POST_AGE: Duration = Duration::from_secs(48 * 60 * 60);

pub struct TweetMixerSource {
    pub tweet_mixer_client: Arc<dyn TweetMixerClient>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for TweetMixerSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableTweetMixerSource)
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let excluded_tweet_ids: Option<BTreeSet<i64>> = if query.seen_ids.is_empty() {
            None
        } else {
            Some(query.seen_ids.iter().map(|&id| id as i64).collect())
        };

        let opt = |s: &str| {
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        };

        let request = TweetMixerRequest {
            client_context: Box::new(ClientContext {
                user_id: Some(query.user_id as i64),
                app_id: Some(query.client_app_id as i64),
                user_agent: opt(&query.user_agent),
                country_code: opt(&query.country_code),
                language_code: opt(&query.language_code),
                ..Default::default()
            }),
            product: Box::new(Product::HOME_RECOMMENDED_TWEETS),
            product_context: Some(Box::new(
                ProductContext::HomeRecommendedTweetsProductContext(
                    HomeRecommendedTweetsProductContext {
                        excluded_tweet_ids,
                        get_random_tweets: None,
                        prediction_request_id: None,
                    },
                ),
            )),
            cursor: None,
            max_results: Some(quality_factor::apply(query.params.get(TweetMixerMaxResults)) as i32),
        };

        let candidates = self
            .tweet_mixer_client
            .get_recommendations(request)
            .await
            .map_err(|e| format!("TweetMixerSource: {}", e))?;

        let result = candidates
            .into_iter()
            .filter_map(|candidate| {
                let tweet_id = candidate.tweet_id as u64;

                let within_age = duration_since_creation_opt(tweet_id)
                    .map(|age| age <= MAX_POST_AGE)
                    .unwrap_or(false);
                if !within_age {
                    return None;
                }

                // 0 until CoreData/TES. InNetwork must run after that fill.
                let author_id = candidate
                    .author_id
                    .and_then(|id| u64::try_from(id).ok())
                    .unwrap_or_default();
                let in_reply_to_tweet_id = candidate
                    .in_reply_to_tweet_id
                    .and_then(|id| u64::try_from(id).ok());

                Some(PostCandidate {
                    tweet_id,
                    author_id,
                    in_reply_to_tweet_id,
                    retweeted_tweet_id: None,
                    served_type: Some(pb::ServedType::ForYouTweetMixer),
                    ..Default::default()
                })
            })
            .collect();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_candidate_pipeline::component_library::clients::tweet_mixer_client::MockTweetMixerClient;

    fn source() -> TweetMixerSource {
        TweetMixerSource {
            tweet_mixer_client: Arc::new(MockTweetMixerClient),
        }
    }

    #[test]
    fn enable_returns_false_when_in_network_only() {
        let query = ScoredPostsQuery {
            in_network_only: true,
            ..Default::default()
        };
        assert!(!source().enable(&query));
    }

    #[test]
    fn enable_returns_false_when_has_cached_posts() {
        let query = ScoredPostsQuery {
            has_cached_posts: true,
            ..Default::default()
        };
        assert!(!source().enable(&query));
    }
}
