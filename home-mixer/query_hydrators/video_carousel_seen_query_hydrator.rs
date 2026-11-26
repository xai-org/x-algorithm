use crate::models::query::ScoredPostsQuery;
use crate::params::EnableVideoCarouselSeenFilter;
use crate::util::video_carousel;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

const KEY_PREFIX: &str = "vcs:1:";
const SEEN_WINDOW_MS: i64 = 3 * 24 * 60 * 60 * 1000;
const MAX_IDS: u32 = 2_000;

pub struct VideoCarouselSeenQueryHydrator {
    pub redis_client: Arc<dyn RedisClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for VideoCarouselSeenQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query) && query.params.get(EnableVideoCarouselSeenFilter)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let rows = self
            .redis_client
            .zrevrangebyscore(
                format!("{KEY_PREFIX}{}", query.user_id),
                query.request_time_ms - SEEN_WINDOW_MS,
                i64::MAX,
                MAX_IDS,
            )
            .await
            .map_err(|err| err.to_string())?;
        Ok(ScoredPostsQuery {
            seen_carousel_video_ids: rows
                .into_iter()
                .filter_map(|(_, member)| std::str::from_utf8(&member).ok()?.parse().ok())
                .collect(),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.seen_carousel_video_ids = hydrated.seen_carousel_video_ids;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use xai_candidate_pipeline::component_library::clients::redis_client::MockRedisClient;

    const DAY_MS: i64 = 24 * 60 * 60 * 1000;

    async fn record(redis: &MockRedisClient, user_id: u64, video_id: u64, at_ms: i64) {
        redis
            .zadd_prune_expire(
                format!("{KEY_PREFIX}{user_id}"),
                at_ms,
                video_id.to_string().into_bytes(),
                0,
                3 * 24 * 60 * 60,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn reads_videos_seen_in_the_last_three_days() {
        let redis = Arc::new(MockRedisClient::default());
        let now = 10 * DAY_MS;
        record(&redis, 7, 101, now - 4 * DAY_MS).await;
        record(&redis, 7, 102, now - DAY_MS).await;
        record(&redis, 8, 201, now).await;
        let hydrator = VideoCarouselSeenQueryHydrator {
            redis_client: redis,
        };
        let query = ScoredPostsQuery {
            user_id: 7,
            request_time_ms: now,
            ..Default::default()
        };

        let hydrated = hydrator.hydrate(&query).await.unwrap();

        assert_eq!(hydrated.seen_carousel_video_ids, HashSet::from([102]));
    }
}
