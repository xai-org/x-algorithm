use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableCachedPosts;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tonic::async_trait;
use tracing::warn;
use xai_candidate_pipeline::component_library::clients::redis_client::{self, RedisClient};
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

const MIN_CACHED_POSTS_THRESHOLD: usize = 500;
const REDIS_GET_TIMEOUT: Duration = Duration::from_millis(300);

pub struct CachedPostsQueryHydrator {
    pub redis_client: Arc<dyn RedisClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for CachedPostsQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableCachedPosts)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let cache_key = redis_client::cached_posts_key(
            query.user_id,
            &query.topic_ids,
            query.in_network_only,
            query.exclude_videos,
        );

        let start = Instant::now();
        let payload =
            match tokio::time::timeout(REDIS_GET_TIMEOUT, self.redis_client.get(cache_key.clone()))
                .await
            {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(err)) => {
                    warn!(
                        cache_key = %cache_key,
                        latency_ms = %start.elapsed().as_millis(),
                        error = %err,
                        "CachedPostsQueryHydrator redis GET error"
                    );
                    return Err(err.to_string());
                }
                Err(_) => {
                    warn!(
                        cache_key = %cache_key,
                        latency_ms = %start.elapsed().as_millis(),
                        timeout_ms = %REDIS_GET_TIMEOUT.as_millis(),
                        "CachedPostsQueryHydrator redis GET timed out"
                    );
                    return Err(format!(
                        "Redis GET timed out after {}ms",
                        REDIS_GET_TIMEOUT.as_millis()
                    ));
                }
            };

        if payload.is_empty() {
            return Ok(ScoredPostsQuery::default());
        }

        let decompressed = zstd::decode_all(payload.as_slice()).map_err(|err| err.to_string())?;
        let cached_posts: Vec<PostCandidate> =
            serde_json::from_slice(&decompressed).map_err(|err| err.to_string())?;

        let has_cached_posts = cached_posts
            .iter()
            .filter(|post| !post.carousel_only)
            .count()
            >= MIN_CACHED_POSTS_THRESHOLD;

        Ok(ScoredPostsQuery {
            cached_posts,
            has_cached_posts,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.cached_posts = hydrated.cached_posts;
        query.has_cached_posts = hydrated.has_cached_posts;
    }
}
