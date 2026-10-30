use crate::candidate_pipeline::reverse_chron_posts_pipeline::ReverseChronPostsPipeline;
use crate::models::brand_safety::BrandSafetyVerdict;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::{FollowingPaginationMeta, ScoredPostsQuery};
use crate::params::FOLLOWING_POST_FETCH_SIZE;
use bytes::Bytes;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::{feed_item, FeedItem, ScoredPost};

pub struct ReverseChronPostsSource {
    pipeline: Arc<ReverseChronPostsPipeline>,
}

impl ReverseChronPostsSource {
    pub fn new(pipeline: Arc<ReverseChronPostsPipeline>) -> Self {
        Self { pipeline }
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for ReverseChronPostsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.user_features.followed_user_ids.is_empty()
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let result = self.pipeline.execute(query.clone()).await;
        let response_truncated = result.retrieved_candidates.len() >= FOLLOWING_POST_FETCH_SIZE;
        let bottom_tweet_id = result
            .retrieved_candidates
            .iter()
            .map(|c| c.tweet_id)
            .filter(|&id| id > 0)
            .min();

        let _ = query
            .following_pagination_meta
            .set(FollowingPaginationMeta {
                response_truncated,
                bottom_tweet_id,
            });

        Ok(result
            .selected_candidates
            .into_iter()
            .map(|c| FeedItem {
                position: 0,
                item: Some(feed_item::Item::Post(post_candidate_to_scored_post(&c))),
            })
            .collect())
    }
}

fn post_candidate_to_scored_post(candidate: &PostCandidate) -> ScoredPost {
    ScoredPost {
        tweet_id: candidate.tweet_id,
        author_id: candidate.author_id,
        retweeted_tweet_id: candidate.retweeted_tweet_id.unwrap_or(0),
        retweeted_user_id: candidate.retweeted_user_id.unwrap_or(0),
        in_reply_to_tweet_id: candidate.in_reply_to_tweet_id.unwrap_or(0),
        score: candidate.score.unwrap_or(0.0) as f32,
        in_network: candidate.in_network.unwrap_or(false),
        served_type: candidate.served_type.map(|t| t as i32).unwrap_or_default(),
        last_scored_timestamp_ms: candidate.last_scored_at_ms.unwrap_or(0),
        prediction_request_id: candidate.prediction_request_id.unwrap_or(0),
        ancestors: candidate.ancestors.clone(),
        tombstone_ancestor_ids: candidate.tombstone_ancestor_ids.clone(),
        ancestor_users: candidate.ancestor_users.clone(),
        quoted_user_id: candidate.quoted_user_id.unwrap_or(0),
        screen_names: candidate.get_screen_names(),
        visibility_reason: candidate.visibility_reason.clone().map(|r| r.into()),
        tweet_type_metrics: Bytes::from(candidate.tweet_type_metrics.clone().unwrap_or_default()),
        following_replied_user_ids: candidate.following_replied_user_ids.clone(),
        brand_safety_verdict: candidate
            .brand_safety_verdict
            .unwrap_or(BrandSafetyVerdict::MediumRisk) as i32,
        tweet_text: candidate.tweet_text.clone(),
        ..Default::default()
    }
}
