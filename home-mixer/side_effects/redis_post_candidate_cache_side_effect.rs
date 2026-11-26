use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{MaxPostsToCache, VideoCarouselExtraCandidates};
use crate::util::video_carousel::CarouselScoring;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::redis_client::{self, RedisClient};
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};

pub(crate) const REDIS_TTL_SECONDS: u64 = 180;
const ZSTD_COMPRESSION_LEVEL: i32 = 6;

pub struct RedisPostCandidateCacheSideEffect {
    redis_client: Arc<dyn RedisClient>,
}

impl RedisPostCandidateCacheSideEffect {
    pub fn new(redis_client: Arc<dyn RedisClient>) -> Self {
        Self { redis_client }
    }

    pub(crate) fn get_candidates_to_cache<'a>(
        query: &ScoredPostsQuery,
        selected: &'a [PostCandidate],
        non_selected: &'a [PostCandidate],
    ) -> Vec<&'a PostCandidate> {
        let (mut carousel_only, mut all_candidates): (Vec<&PostCandidate>, Vec<&PostCandidate>) =
            selected
                .iter()
                .chain(non_selected.iter())
                .filter(|c| c.weighted_score.is_some_and(|s| s > 0.0))
                .partition(|c| c.carousel_only);
        all_candidates.sort_by(|a, b| {
            b.weighted_score
                .unwrap()
                .partial_cmp(&a.weighted_score.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_candidates.truncate(query.params.get(MaxPostsToCache));
        let scoring = CarouselScoring::from_query(query);
        carousel_only.sort_by(|a, b| scoring.rank(a, b));
        carousel_only.truncate(query.params.get(VideoCarouselExtraCandidates));
        all_candidates.extend(carousel_only);
        all_candidates
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for RedisPostCandidateCacheSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && !query.has_cached_posts
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let user_id = input.query.user_id;

        let candidates_to_cache = Self::get_candidates_to_cache(
            &input.query,
            &input.selected_candidates,
            &input.non_selected_candidates,
        );

        let cache_key = redis_client::cached_posts_key(
            user_id,
            &input.query.topic_ids,
            input.query.in_network_only,
            input.query.exclude_videos,
        );
        let json_payload =
            serde_json::to_vec(&candidates_to_cache).map_err(|err| err.to_string())?;
        let uncompressed_size = json_payload.len();
        let compressed_payload = tokio::task::spawn_blocking(move || {
            zstd::encode_all(json_payload.as_slice(), ZSTD_COMPRESSION_LEVEL)
        })
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| err.to_string())?;

        tracing::debug!(
            user_id = user_id,
            count = candidates_to_cache.len(),
            cache_key = cache_key.clone(),
            uncompressed_size = uncompressed_size,
            compressed_size = compressed_payload.len(),
            "RedisPostCandidateCacheSideEffect caching candidates"
        );

        self.redis_client
            .set_ex(cache_key, compressed_payload, REDIS_TTL_SECONDS)
            .await
            .map_err(|err| err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::query::ScoredPostsQuery;
    use xai_candidate_pipeline::component_library::clients::redis_client::MockRedisClient;

    fn test_query() -> Arc<ScoredPostsQuery> {
        Arc::new(ScoredPostsQuery::default())
    }

    fn candidate(tweet_id: u64, author_id: u64, score: Option<f64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id,
            weighted_score: score,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn redis_side_effect_writes_payload_with_ttl() {
        let redis_client = Arc::new(MockRedisClient::default());
        let side_effect = RedisPostCandidateCacheSideEffect::new(redis_client.clone());
        let query = test_query();
        let input = Arc::new(SideEffectInput {
            query: query.clone(),
            selected_candidates: vec![candidate(123, 456, Some(0.9))],
            non_selected_candidates: vec![candidate(789, 101, Some(0.8))],
        });

        if let Err(err) = side_effect.side_effect(input).await {
            panic!("side effect should write to redis: {err}");
        }

        let cache_key = format!("scored-posts_{}", query.user_id);
        let payload: Vec<u8> = redis_client
            .get(cache_key.clone())
            .await
            .expect("Failed to read cached value");
        let decompressed = zstd::decode_all(payload.as_slice()).expect("Failed to decompress");
        let cached: Vec<PostCandidate> =
            serde_json::from_slice(&decompressed).expect("Failed to deserialize");

        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].tweet_id, 123);
        assert_eq!(cached[0].author_id, 456);
        assert_eq!(cached[1].tweet_id, 789);
        assert_eq!(cached[1].author_id, 101);

        let ttl: i64 = redis_client
            .ttl(cache_key.clone())
            .await
            .expect("Failed to read TTL");
        assert_eq!(ttl, REDIS_TTL_SECONDS as i64);

        let _: () = redis_client
            .del(cache_key)
            .await
            .expect("Failed to delete test key");
    }

    #[tokio::test]
    async fn caches_selected_and_non_selected_sorted_by_score_desc() {
        let redis_client = Arc::new(MockRedisClient::default());
        let side_effect = RedisPostCandidateCacheSideEffect::new(redis_client.clone());
        let query = test_query();

        let input = Arc::new(SideEffectInput {
            query: query.clone(),
            selected_candidates: vec![candidate(1, 100, Some(0.9)), candidate(2, 200, Some(0.7))],
            non_selected_candidates: vec![
                candidate(3, 300, Some(0.8)),
                candidate(4, 400, Some(0.6)),
                candidate(5, 500, None),
                candidate(6, 600, Some(0.0)),
            ],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed");

        let cache_key = format!("scored-posts_{}", query.user_id);
        let payload: Vec<u8> = redis_client
            .get(cache_key)
            .await
            .expect("Failed to read cached value");
        let decompressed = zstd::decode_all(payload.as_slice()).expect("Failed to decompress");
        let cached: Vec<PostCandidate> =
            serde_json::from_slice(&decompressed).expect("Failed to deserialize");

        assert_eq!(cached.len(), 4);
        assert_eq!(cached[0].tweet_id, 1);
        assert_eq!(cached[1].tweet_id, 3);
        assert_eq!(cached[2].tweet_id, 2);
        assert_eq!(cached[3].tweet_id, 4);
    }

    #[tokio::test]
    async fn page_rl_fields_survive_the_cached_posts_round_trip() {
        use crate::query_hydrators::cached_posts_query_hydrator::CachedPostsQueryHydrator;
        use xai_candidate_pipeline::query_hydrator::QueryHydrator;

        let redis_client = Arc::new(MockRedisClient::default());
        let side_effect = RedisPostCandidateCacheSideEffect::new(redis_client.clone());
        let query = test_query();
        let decoded = PostCandidate {
            page_decode_slot: Some(4),
            page_decode_prob: Some(0.35),
            page_rl_exploration_prob: Some(0.6),
            ..candidate(11, 110, Some(0.9))
        };
        let input = Arc::new(SideEffectInput {
            query: query.clone(),
            selected_candidates: vec![decoded],
            non_selected_candidates: vec![candidate(12, 120, Some(0.5))],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed");
        let hydrated = CachedPostsQueryHydrator {
            redis_client: redis_client.clone(),
        }
        .hydrate(&query)
        .await
        .expect("cached posts should hydrate");

        let cached = &hydrated.cached_posts;
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].tweet_id, 11);
        assert_eq!(cached[0].page_decode_slot, Some(4));
        assert_eq!(cached[0].page_decode_prob, Some(0.35));
        assert_eq!(cached[0].page_rl_exploration_prob, Some(0.6));
        assert_eq!(cached[1].tweet_id, 12);
        assert_eq!(cached[1].page_decode_slot, None);
        assert_eq!(cached[1].page_decode_prob, None);
        assert_eq!(cached[1].page_rl_exploration_prob, None);
    }

    #[test]
    fn get_candidates_to_cache_truncates_to_max() {
        let max_posts_to_cache = 750;
        let selected: Vec<PostCandidate> = (0..600)
            .map(|i| candidate(i, i, Some((i + 1) as f64)))
            .collect();
        let non_selected: Vec<PostCandidate> = (600..1200)
            .map(|i| candidate(i, i, Some((i + 1) as f64)))
            .collect();

        let result = RedisPostCandidateCacheSideEffect::get_candidates_to_cache(
            &ScoredPostsQuery::default(),
            &selected,
            &non_selected,
        );

        assert_eq!(result.len(), max_posts_to_cache);
        assert_eq!(result[0].tweet_id, 1199);
        assert_eq!(result[max_posts_to_cache - 1].tweet_id, 450);
    }

    #[test]
    fn get_candidates_to_cache_filters_none_and_zero_scores() {
        let selected = vec![
            candidate(1, 100, Some(5.0)),
            candidate(2, 200, None),
            candidate(3, 300, Some(0.0)),
        ];
        let non_selected = vec![candidate(4, 400, Some(3.0)), candidate(5, 500, Some(-1.0))];

        let result = RedisPostCandidateCacheSideEffect::get_candidates_to_cache(
            &ScoredPostsQuery::default(),
            &selected,
            &non_selected,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tweet_id, 1);
        assert_eq!(result[1].tweet_id, 4);
    }

    #[test]
    fn carousel_only_posts_skip_feed_cache_spots() {
        let selected: Vec<PostCandidate> = (0..1000).map(|i| candidate(i, i, Some(1.0))).collect();
        let non_selected: Vec<PostCandidate> = (1000..1040)
            .map(|i| PostCandidate {
                carousel_only: true,
                ..candidate(i, i, Some(2.0))
            })
            .collect();

        let cached = RedisPostCandidateCacheSideEffect::get_candidates_to_cache(
            &ScoredPostsQuery::default(),
            &selected,
            &non_selected,
        );

        assert_eq!(cached.iter().filter(|c| !c.carousel_only).count(), 750);
        assert_eq!(cached.iter().filter(|c| c.carousel_only).count(), 30);
        assert!(crate::util::video_carousel::is_carousel_only(
            cached[750],
            false
        ));
    }
}
