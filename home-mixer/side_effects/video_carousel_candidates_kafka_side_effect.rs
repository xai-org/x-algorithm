use crate::models::candidate::CandidateHelpers;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableVideoCarouselCandidatesLogging;
use crate::side_effects::phoenix_experiments_side_effect::{
    build_logged_scored_candidate, insert_cluster_scores,
};
use crate::util::video_carousel::CarouselScoring;
use futures::future::join_all;
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaCluster, KafkaPublisherClient, ProdKafkaPublisherClient,
    VIDEO_CAROUSEL_LOGGED_SCORED_CANDIDATES_TOPIC,
};
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::feed_item::Item as FeedItemKind;
use xai_home_mixer_proto::FeedItem;
use xai_proto::recsys_logging::LoggedScoredCandidate;

pub struct VideoCarouselCandidatesKafkaSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl VideoCarouselCandidatesKafkaSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }

    pub async fn prod() -> Self {
        Self::new(Arc::new(
            ProdKafkaPublisherClient::new(
                VIDEO_CAROUSEL_LOGGED_SCORED_CANDIDATES_TOPIC,
                KafkaCluster::Phoenix,
            )
            .await,
        ))
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for VideoCarouselCandidatesKafkaSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnableVideoCarouselCandidatesLogging)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let sends = served_carousel_candidates(&input.query, &input.selected_candidates)
            .into_iter()
            .map(|logged| {
                let bytes = logged.encode_to_vec();
                let kafka_client = Arc::clone(&self.kafka_client);
                async move { kafka_client.send(&bytes).await }
            });
        for result in join_all(sends).await {
            if let Err(e) = result {
                tracing::error!("Failed to publish video carousel candidate to Kafka: {e}");
            }
        }
        Ok(())
    }
}

fn served_carousel_candidates(
    query: &ScoredPostsQuery,
    served: &[FeedItem],
) -> Vec<LoggedScoredCandidate> {
    let carousel_served = served
        .iter()
        .any(|item| matches!(item.item, Some(FeedItemKind::VideoCarousel(_))));
    let Some(videos) = query.video_carousel.get().filter(|_| carousel_served) else {
        return Vec::new();
    };
    let scoring = CarouselScoring::from_query(query);
    videos
        .iter()
        .enumerate()
        .map(|(position, video)| {
            let mut prediction_scores = HashMap::new();
            insert_cluster_scores(&mut prediction_scores, "home", &video.phoenix_scores);
            if let Some(immersive_scores) = &video.immersive_scores {
                insert_cluster_scores(&mut prediction_scores, "immersive", immersive_scores);
            }
            prediction_scores.insert("video_carousel.score".to_string(), scoring.score(video));
            LoggedScoredCandidate {
                post_id: video.get_original_tweet_id() as i64,
                author_id: Some(video.get_original_author_id() as i64),
                carousel_position: Some(position as i32),
                ..build_logged_scored_candidate(
                    video,
                    query.user_id,
                    query.request_id,
                    query.request_time_ms,
                    prediction_scores,
                )
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::{PhoenixScores, PostCandidate};
    use xai_home_mixer_proto::{ScoredPost, VideoCarouselModule};

    fn video(tweet_id: u64, video_open: f64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            phoenix_scores: PhoenixScores {
                video_open_score: Some(video_open),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn logs_carousel_videos_in_order_only_when_the_carousel_is_served() {
        let mut query = ScoredPostsQuery {
            user_id: 7,
            request_id: 99,
            ..Default::default()
        };
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_video_carousel_video_open_weight".to_string(),
            "2.0",
        );
        results.override_fs(
            "rust_home_mixer_video_carousel_immersive_vqv_weight".to_string(),
            "0.0",
        );
        query.params = results.into();
        let retweet = PostCandidate {
            author_id: 20,
            retweeted_tweet_id: Some(3),
            retweeted_user_id: Some(30),
            ..video(2, 0.4)
        };
        let _ = query.video_carousel.set(vec![video(1, 0.9), retweet]);
        let carousel = FeedItem {
            position: 5,
            item: Some(FeedItemKind::VideoCarousel(VideoCarouselModule {
                videos: vec![ScoredPost::default(); 2],
            })),
        };

        let logged = served_carousel_candidates(&query, &[carousel]);

        let rows: Vec<_> = logged
            .iter()
            .map(|c| {
                (
                    c.post_id,
                    c.author_id,
                    c.carousel_position,
                    c.prediction_scores.get("phoenix.home.video_open").copied(),
                    c.prediction_scores.get("video_carousel.score").copied(),
                )
            })
            .collect();
        assert_eq!(
            rows,
            vec![
                (1, Some(0), Some(0), Some(0.9), Some(1.8)),
                (3, Some(30), Some(1), Some(0.4), Some(0.8))
            ]
        );
        assert!(logged
            .iter()
            .all(|c| c.viewer_id == Some(7) && c.request_join_id == Some(99)));
        assert!(served_carousel_candidates(&query, &[]).is_empty());
    }
}
