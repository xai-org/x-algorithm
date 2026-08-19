use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixRequestCacheSideEffect, LogSlateContext, PhoenixRequestCacheSideEffectTtlSeconds,
};
use crate::util::phoenix_request::{
    build_request_without_sequence_and_candidates, build_tweet_infos,
};
use futures::future::join_all;
use prost::Message;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_recsys_proto::ProductSurface;

const KEY_PREFIX: &str = "phoenix_request_cache";
const ZSTD_COMPRESSION_LEVEL: i32 = 3;

fn encode_cache_value<T: Message>(message: &T) -> Result<Vec<u8>, String> {
    zstd::encode_all(message.encode_to_vec().as_slice(), ZSTD_COMPRESSION_LEVEL)
        .map_err(|e| format!("zstd compression failed: {e}"))
}

pub struct PhoenixRequestCacheSideEffect {
    phoenix_request_cache_redis_atla_client: Arc<dyn RedisClient>,
    phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient>,
}

impl PhoenixRequestCacheSideEffect {
    pub fn new(
        phoenix_request_cache_redis_atla_client: Arc<dyn RedisClient>,
        phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient>,
    ) -> Self {
        Self {
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
        }
    }
}

pub fn request_level_cache_key(prediction_request_id: u64) -> String {
    format!("{KEY_PREFIX}_{prediction_request_id}")
}

pub fn candidate_cache_key(prediction_request_id: u64, post_id: u64) -> String {
    format!("{KEY_PREFIX}_{prediction_request_id}_{post_id}")
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for PhoenixRequestCacheSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnablePhoenixRequestCacheSideEffect)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let ttl = query.params.get(PhoenixRequestCacheSideEffectTtlSeconds);

        if input.selected_candidates.is_empty() && input.non_selected_candidates.is_empty() {
            return Ok(());
        }

        let prediction_request_id = input
            .selected_candidates
            .iter()
            .find_map(|c| c.prediction_request_id);
        let Some(prediction_request_id) = prediction_request_id else {
            return Ok(());
        };

        let product_surface = if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };

        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();

        if !query.has_cached_posts {
            let phoenix_request =
                build_request_without_sequence_and_candidates(query, product_surface);
            entries.push((
                request_level_cache_key(prediction_request_id),
                encode_cache_value(&phoenix_request)?,
            ));
        }

        let log_slate_context = query.params.get(LogSlateContext);
        let mut tweet_infos = build_tweet_infos(query, &input.selected_candidates);
        if log_slate_context {
            for (info, candidate) in tweet_infos.iter_mut().zip(input.selected_candidates.iter()) {
                info.score_info = Some(candidate.as_score_info());
            }
        }
        for candidate in tweet_infos {
            entries.push((
                candidate_cache_key(prediction_request_id, candidate.tweet_id),
                encode_cache_value(&candidate)?,
            ));
        }

        let mut futures: Vec<
            std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + '_>>,
        > = Vec::new();

        for (key, value) in entries {
            let pdxa_key = key.clone();
            let pdxa_value = value.clone();
            futures.push(Box::pin(async move {
                self.phoenix_request_cache_redis_atla_client
                    .set_ex(key, value, ttl)
                    .await
                    .map_err(|e| e.to_string())
            }));
            futures.push(Box::pin(async move {
                self.phoenix_request_cache_redis_pdxa_client
                    .set_ex(pdxa_key, pdxa_value, ttl)
                    .await
                    .map_err(|e| format!("xdc: {e}"))
            }));
        }

        let results = join_all(futures).await;
        let total = results.len();
        let mut first_error: Option<String> = None;
        let mut error_count = 0usize;
        for result in results {
            if let Err(e) = result {
                error_count += 1;
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }

        if total > 0 && error_count * 10 > total {
            return Err(first_error.unwrap_or_default());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::query::ScoredPostsQuery;
    use xai_candidate_pipeline::component_library::clients::redis_client::MockRedisClient;
    use xai_feature_switches::Param;
    use xai_recsys_proto::{PredictNextActionsRequest, TweetInfo};

    const TEST_PREDICTION_REQUEST_ID: u64 = 99;

    fn test_query() -> Arc<ScoredPostsQuery> {
        Arc::new(ScoredPostsQuery {
            user_id: 42,
            ..Default::default()
        })
    }

    fn candidate(tweet_id: u64, author_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id,
            prediction_request_id: Some(TEST_PREDICTION_REQUEST_ID),
            ..Default::default()
        }
    }

    async fn decode_tweet_info(redis_client: &MockRedisClient, key: &str) -> TweetInfo {
        let payload: Vec<u8> = redis_client
            .get(key.to_string())
            .await
            .expect("Failed to read cached value");
        let decompressed =
            zstd::decode_all(payload.as_slice()).expect("Failed to decompress proto");
        TweetInfo::decode(decompressed.as_slice()).expect("Failed to decode proto")
    }

    async fn decode_request(
        redis_client: &MockRedisClient,
        key: &str,
    ) -> PredictNextActionsRequest {
        let payload: Vec<u8> = redis_client
            .get(key.to_string())
            .await
            .expect("Failed to read cached value");
        let decompressed =
            zstd::decode_all(payload.as_slice()).expect("Failed to decompress proto");
        PredictNextActionsRequest::decode(decompressed.as_slice()).expect("Failed to decode proto")
    }

    fn test_query_with_flags(flags: &[(&str, &str)]) -> Arc<ScoredPostsQuery> {
        let mut query = ScoredPostsQuery {
            user_id: 42,
            ..Default::default()
        };
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        for (key, value) in flags {
            results.override_fs(key.to_string(), value);
        }
        query.params = results.into();
        Arc::new(query)
    }

    #[tokio::test]
    async fn attaches_score_info_when_slate_context_logging_enabled() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            PhoenixRequestCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client);
        let query = test_query_with_flags(&[("rust_home_mixer_log_slate_context", "true")]);

        let with_context = PostCandidate {
            weighted_score: Some(0.9),
            score: Some(0.6),
            phoenix_scores: crate::models::candidate::PhoenixScores {
                favorite_score: Some(0.25),
                ..Default::default()
            },
            slate_context: Some(crate::models::candidate::SlateContext {
                k: 2,
                pool_rank: 7,
                pool_rank_gap: Some(3),
                fatigue: 1.5,
                pre_diversity_score: 0.42,
                ..Default::default()
            }),
            ..candidate(1, 100)
        };
        let input = Arc::new(SideEffectInput {
            query,
            selected_candidates: vec![with_context, candidate(2, 200)],
            non_selected_candidates: vec![],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed");

        let t1 = decode_tweet_info(
            &atla_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 1),
        )
        .await;
        let score_info = t1.score_info.expect("score_info should be set");
        assert_eq!(score_info.prediction_scores["favorite"], 0.25);
        assert_eq!(score_info.prediction_scores["vqv"], 0.0);
        assert_eq!(score_info.weighted_score, Some(0.9));
        assert_eq!(score_info.final_score, Some(0.6));
        let slate_context = score_info
            .slate_context
            .expect("slate_context should be set");
        assert_eq!(slate_context.k, 2);
        assert_eq!(slate_context.pool_rank, 7);
        assert_eq!(slate_context.pool_rank_gap, Some(3));
        assert_eq!(slate_context.fatigue, 1.5);
        assert_eq!(slate_context.pre_diversity_score, 0.42);

        let t2 = decode_tweet_info(
            &atla_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 2),
        )
        .await;
        assert!(t2
            .score_info
            .expect("score_info should be set")
            .slate_context
            .is_none());
    }

    #[tokio::test]
    async fn omits_score_info_when_slate_context_logging_disabled() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            PhoenixRequestCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client);
        let input = Arc::new(SideEffectInput {
            query: test_query_with_flags(&[("rust_home_mixer_log_slate_context", "false")]),
            selected_candidates: vec![candidate(1, 100)],
            non_selected_candidates: vec![],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed");

        let t1 = decode_tweet_info(
            &atla_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 1),
        )
        .await;
        assert!(t1.score_info.is_none());
    }

    #[tokio::test]
    async fn writes_request_and_candidate_entries() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            PhoenixRequestCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client);
        let query = test_query();
        let input = Arc::new(SideEffectInput {
            query: query.clone(),
            selected_candidates: vec![candidate(1, 100), candidate(2, 200)],
            non_selected_candidates: vec![candidate(3, 300)],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed");

        let req_key = request_level_cache_key(TEST_PREDICTION_REQUEST_ID);
        let request = decode_request(&atla_redis_client, &req_key).await;
        assert_eq!(request.candidate_sets.len(), 1);
        assert_eq!(request.candidate_sets[0].user_id, 42);

        let t1 = decode_tweet_info(
            &atla_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 1),
        )
        .await;
        assert_eq!(t1.tweet_id, 1);

        let t2 = decode_tweet_info(
            &atla_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 2),
        )
        .await;
        assert_eq!(t2.tweet_id, 2);

        assert!(atla_redis_client
            .get(candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 3))
            .await
            .is_err());

        let ttl: i64 = atla_redis_client
            .ttl(req_key)
            .await
            .expect("Failed to read TTL");
        assert_eq!(
            ttl,
            PhoenixRequestCacheSideEffectTtlSeconds.default() as i64
        );
    }

    #[tokio::test]
    async fn writes_to_xdc_redis() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect = PhoenixRequestCacheSideEffect::new(
            atla_redis_client.clone(),
            pdxa_redis_client.clone(),
        );
        let query = test_query();
        let input = Arc::new(SideEffectInput {
            query: query.clone(),
            selected_candidates: vec![candidate(1, 100), candidate(2, 200)],
            non_selected_candidates: vec![],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed");

        let req_key = request_level_cache_key(TEST_PREDICTION_REQUEST_ID);

        let request = decode_request(&atla_redis_client, &req_key).await;
        assert_eq!(request.candidate_sets.len(), 1);

        let pdxa_request = decode_request(&pdxa_redis_client, &req_key).await;
        assert_eq!(pdxa_request.candidate_sets.len(), 1);

        let pdxa_t1 = decode_tweet_info(
            &pdxa_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 1),
        )
        .await;
        assert_eq!(pdxa_t1.tweet_id, 1);

        let pdxa_t2 = decode_tweet_info(
            &pdxa_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 2),
        )
        .await;
        assert_eq!(pdxa_t2.tweet_id, 2);
    }

    #[tokio::test]
    async fn skips_request_write_for_cached_posts() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            PhoenixRequestCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client);
        let query = Arc::new(ScoredPostsQuery {
            user_id: 42,
            has_cached_posts: true,
            ..Default::default()
        });
        let input = Arc::new(SideEffectInput {
            query,
            selected_candidates: vec![candidate(1, 100)],
            non_selected_candidates: vec![],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed");

        assert!(atla_redis_client
            .get(request_level_cache_key(TEST_PREDICTION_REQUEST_ID))
            .await
            .is_err());

        let t1 = decode_tweet_info(
            &atla_redis_client,
            &candidate_cache_key(TEST_PREDICTION_REQUEST_ID, 1),
        )
        .await;
        assert_eq!(t1.tweet_id, 1);
    }

    #[tokio::test]
    async fn skips_when_no_prediction_request_id() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            PhoenixRequestCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client);
        let query = test_query();
        let input = Arc::new(SideEffectInput {
            query,
            selected_candidates: vec![PostCandidate {
                tweet_id: 1,
                author_id: 100,
                prediction_request_id: None,
                ..Default::default()
            }],
            non_selected_candidates: vec![],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("should succeed without prediction_request_id");
    }

    #[tokio::test]
    async fn skips_when_no_candidates() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            PhoenixRequestCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client);
        let query = test_query();
        let input = Arc::new(SideEffectInput {
            query,
            selected_candidates: vec![],
            non_selected_candidates: vec![candidate(1, 100)],
        });

        side_effect
            .side_effect(input)
            .await
            .expect("side effect should succeed on empty selected candidates");
    }
}
