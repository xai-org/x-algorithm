use crate::models::query::ScoredPostsQuery;
use crate::scored_posts_server::ScoredPostsServer;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::{feed_item, FeedItem};

pub struct ScoredPostsSource {
    pub scored_posts_server: Arc<ScoredPostsServer>,
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for ScoredPostsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.is_ads_only_timeline()
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let output = self
            .scored_posts_server
            .run_pipeline(query.clone())
            .await
            .map_err(|e| format!("ScoredPostsSource: {e}"))?;

        let posts = output.scored_posts.into_iter().map(|post| FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(post)),
        });
        let video_carousel = output.video_carousel.map(|carousel| FeedItem {
            position: 0,
            item: Some(feed_item::Item::VideoCarousel(carousel)),
        });

        Ok(posts.chain(video_carousel).collect())
    }
}
