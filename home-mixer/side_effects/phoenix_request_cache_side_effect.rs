use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePageDecodeRerank, EnablePageLogging, EnablePageRlExploration,
    EnablePhoenixRequestCacheSideEffect, LogSlateContext, PageDecodeTemperature, PageLogTtlSeconds,
    PhoenixRequestCacheSideEffectTtlSeconds,
};
use crate::util::phoenix_request::{
    build_request_without_sequence_and_candidates, build_tweet_infos,
};
use crate::util::post_engagers::clear_post_engagers;
use futures::future::{join_all, BoxFuture};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::{feed_item, FeedItem, ScoredPost};
use xai_recsys_proto::{CandidateSet, ProductSurface, ScoreInfo, TweetInfo};
use xai_stats_receiver::global_stats_receiver;

const KEY_PREFIX: &str = "phoenix_request_cache";
const ZSTD_COMPRESSION_LEVEL: i32 = 3;
const MAX_SERVED_SLATE_POSTS: usize = 64;
const SLATE_WRITES_METRIC: &str = "ServedSlateCacheSideEffect.slate_writes";

fn encode_cache_value<T: Message>(message: &T) -> Result<Vec<u8>, String> {
    zstd::encode_all(message.encode_to_vec().as_slice(), ZSTD_COMPRESSION_LEVEL)
        .map_err(|e| format!("zstd compression failed: {e}"))
}

fn product_surface(query: &ScoredPostsQuery) -> ProductSurface {
    if query.in_network_only {
        ProductSurface::HomeTimelineRankedFollowing
    } else {
        ProductSurface::HomeTimelineRanking
    }
}

async fn write_to_both_dcs(
    atla_client: &dyn RedisClient,
    pdxa_client: &dyn RedisClient,
    entries: Vec<(String, Vec<u8>)>,
    ttl: u64,
) -> Result<(), String> {
    let mut futures: Vec<BoxFuture<'_, Result<(), String>>> = Vec::new();

    for (key, value) in entries {
        let pdxa_key = key.clone();
        let pdxa_value = value.clone();
        futures.push(Box::pin(async move {
            atla_client
                .set_ex(key, value, ttl)
                .await
                .map_err(|e| e.to_string())
        }));
        futures.push(Box::pin(async move {
            pdxa_client
                .set_ex(pdxa_key, pdxa_value, ttl)
                .await
                .map_err(|e| format!("xdc: {e}"))
        }));
    }

    fail_on_more_than_a_tenth_errors(join_all(futures).await)
}

async fn append_to_both_dcs(
    atla_client: &dyn RedisClient,
    pdxa_client: &dyn RedisClient,
    entries: Vec<(String, Vec<u8>)>,
    score: i64,
    ttl: u64,
) -> Result<(), String> {
    let mut futures: Vec<BoxFuture<'_, Result<(), String>>> = Vec::new();

    for (key, member) in entries {
        let pdxa_key = key.clone();
        let pdxa_member = member.clone();
        futures.push(Box::pin(async move {
            atla_client
                .zadd_prune_expire(key, score, member, i64::MIN, ttl)
                .await
                .map_err(|e| e.to_string())
        }));
        futures.push(Box::pin(async move {
            pdxa_client
                .zadd_prune_expire(pdxa_key, score, pdxa_member, i64::MIN, ttl)
                .await
                .map_err(|e| format!("xdc: {e}"))
        }));
    }

    fail_on_more_than_a_tenth_errors(join_all(futures).await)
}

fn fail_on_more_than_a_tenth_errors(results: Vec<Result<(), String>>) -> Result<(), String> {
    let total = results.len();
    let mut errors = results.into_iter().filter_map(Result::err);
    let first_error = errors.next();
    let error_count = first_error.iter().count() + errors.count();
    if total > 0 && error_count * 10 > total {
        return Err(first_error.unwrap_or_default());
    }
    Ok(())
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

        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();

        if !query.has_cached_posts {
            let phoenix_request =
                build_request_without_sequence_and_candidates(query, product_surface(query));
            entries.push((
                request_level_cache_key(prediction_request_id),
                encode_cache_value(&phoenix_request)?,
            ));
        }

        let log_slate_context = query.params.get(LogSlateContext);
        let mut tweet_infos =
            build_tweet_infos(query, &input.selected_candidates, query.request_time_ms);
        tweet_infos.iter_mut().for_each(clear_post_engagers);
        if log_slate_context {
            for (info, candidate) in tweet_infos.iter_mut().zip(input.selected_candidates.iter()) {
                info.score_info = Some(candidate.as_score_info_no_prediction_scores());
            }
        }
        for candidate in tweet_infos {
            entries.push((
                candidate_cache_key(prediction_request_id, candidate.tweet_id),
                encode_cache_value(&candidate)?,
            ));
        }

        write_to_both_dcs(
            self.phoenix_request_cache_redis_atla_client.as_ref(),
            self.phoenix_request_cache_redis_pdxa_client.as_ref(),
            entries,
            ttl,
        )
        .await
    }
}

pub fn served_page_log_key(prediction_request_id: u64) -> String {
    format!("{KEY_PREFIX}_{prediction_request_id}_page")
}

pub(crate) fn served_post_score_infos(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
) -> HashMap<u64, ScoreInfo> {
    let decode_applied = query.params.get(EnablePageDecodeRerank);
    let decode_sampled = query.params.get(PageDecodeTemperature) > 0.0;
    let exploration_applied = query.params.get(EnablePageRlExploration);
    candidates
        .iter()
        .map(|candidate| {
            let decoded = decode_applied && candidate.page_decode_slot.is_some();
            let score_info = ScoreInfo {
                weighted_score: candidate.weighted_score,
                final_score: candidate.score,
                reward_rerank_slot_prob: candidate
                    .page_rl_exploration_prob
                    .filter(|_| exploration_applied),
                page_decode_slot_prob: candidate
                    .page_decode_prob
                    .filter(|_| decoded && decode_sampled),
                page_decoded: decoded.then_some(true),
                ..Default::default()
            };
            (candidate.tweet_id, score_info)
        })
        .collect()
}

fn original_tweet_id(post: &ScoredPost) -> u64 {
    if post.retweeted_tweet_id != 0 {
        post.retweeted_tweet_id
    } else {
        post.tweet_id
    }
}

pub(crate) fn served_slates(
    query: &ScoredPostsQuery,
    served: &[FeedItem],
) -> Vec<(u64, CandidateSet)> {
    let score_infos = query.served_post_score_infos.get();
    let mut slates: Vec<(u64, CandidateSet)> = Vec::new();
    let posts = served
        .iter()
        .filter_map(|item| match &item.item {
            Some(feed_item::Item::Post(post)) if post.prediction_request_id != 0 => Some(post),
            _ => None,
        })
        .take(MAX_SERVED_SLATE_POSTS);
    for post in posts {
        let index = match slates.iter().position(|(prediction_request_id, _)| {
            *prediction_request_id == post.prediction_request_id
        }) {
            Some(index) => index,
            None => {
                slates.push((
                    post.prediction_request_id,
                    CandidateSet {
                        user_id: query.user_id,
                        product_surface: product_surface(query) as i32,
                        request_timestamp_ms: Some(query.request_time_ms),
                        ..Default::default()
                    },
                ));
                slates.len() - 1
            }
        };
        let slate = &mut slates[index].1;
        slate.candidates.push(TweetInfo {
            tweet_id: original_tweet_id(post),
            tweet_timeline_position: slate.candidates.len() as u32,
            score_info: score_infos.and_then(|infos| infos.get(&post.tweet_id).cloned()),
            ..Default::default()
        });
    }
    slates
}

fn record_slate_write(outcome: &str) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(SLATE_WRITES_METRIC, &[("outcome", outcome)], 1);
    }
}

pub struct ServedSlateCacheSideEffect {
    phoenix_request_cache_redis_atla_client: Arc<dyn RedisClient>,
    phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient>,
}

impl ServedSlateCacheSideEffect {
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

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for ServedSlateCacheSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnablePageLogging)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let slates = served_slates(query, &input.selected_candidates);
        if slates.is_empty() {
            record_slate_write("no_posts");
            return Ok(());
        }

        let entries = slates
            .iter()
            .map(|(prediction_request_id, slate)| {
                Ok((
                    served_page_log_key(*prediction_request_id),
                    encode_cache_value(slate)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let result = append_to_both_dcs(
            self.phoenix_request_cache_redis_atla_client.as_ref(),
            self.phoenix_request_cache_redis_pdxa_client.as_ref(),
            entries,
            query.request_time_ms,
            query.params.get(PageLogTtlSeconds),
        )
        .await;
        record_slate_write(if result.is_ok() { "success" } else { "error" });
        result
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
            slate_context: Some(Box::new(crate::models::candidate::SlateContext {
                k: 2,
                pool_rank: 7,
                pool_rank_gap: Some(3),
                fatigue: 1.5,
                pre_diversity_score: 0.42,
                ..Default::default()
            })),
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
        assert!(score_info.prediction_scores.is_empty());
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

    const SERVED_AT_MS: i64 = 1_760_000_000_000;
    const PAGE_PREDICTION_ID: u64 = 4242;

    fn page_rl_query(
        request_time_ms: i64,
        has_cached_posts: bool,
        page_rl_switches: &str,
        decode_temperature: &str,
    ) -> ScoredPostsQuery {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        for (key, value) in [
            ("rust_home_mixer_enable_page_logging", "true"),
            (
                "rust_home_mixer_enable_page_decode_rerank",
                page_rl_switches,
            ),
            (
                "rust_home_mixer_enable_page_rl_exploration",
                page_rl_switches,
            ),
            (
                "rust_home_mixer_page_decode_temperature",
                decode_temperature,
            ),
            ("rust_home_mixer_page_log_ttl_seconds", "600"),
        ] {
            results.override_fs(key.to_string(), value);
        }
        ScoredPostsQuery {
            user_id: 42,
            request_id: request_time_ms as u64,
            request_time_ms,
            has_cached_posts,
            params: results.into(),
            ..Default::default()
        }
    }

    fn scored(tweet_id: u64, weighted_score: f64, score: f64) -> PostCandidate {
        PostCandidate {
            weighted_score: Some(weighted_score),
            score: Some(score),
            ..candidate(tweet_id, tweet_id * 10)
        }
    }

    fn post_of(prediction_request_id: u64, tweet_id: u64, retweeted_tweet_id: u64) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost {
                tweet_id,
                retweeted_tweet_id,
                prediction_request_id,
                ..Default::default()
            })),
        }
    }

    fn post_item(tweet_id: u64, retweeted_tweet_id: u64) -> FeedItem {
        post_of(PAGE_PREDICTION_ID, tweet_id, retweeted_tweet_id)
    }

    fn module_item(item: feed_item::Item) -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(item),
        }
    }

    async fn decode_page_log(
        redis_client: &MockRedisClient,
        prediction_request_id: u64,
    ) -> Vec<(i64, CandidateSet)> {
        redis_client
            .zrevrangebyscore(
                served_page_log_key(prediction_request_id),
                i64::MIN,
                i64::MAX,
                u32::MAX,
            )
            .await
            .expect("Failed to read page log")
            .into_iter()
            .rev()
            .map(|(score, payload)| {
                let decompressed =
                    zstd::decode_all(payload.as_slice()).expect("Failed to decompress proto");
                (
                    score,
                    CandidateSet::decode(decompressed.as_slice()).expect("Failed to decode proto"),
                )
            })
            .collect()
    }

    fn slate_tweet_ids(slate: &CandidateSet) -> Vec<u64> {
        slate.candidates.iter().map(|c| c.tweet_id).collect()
    }

    fn only_slate(slates: Vec<(u64, CandidateSet)>) -> CandidateSet {
        let mut slates = slates.into_iter();
        let (prediction_request_id, slate) = slates.next().expect("one slate");
        assert_eq!(prediction_request_id, PAGE_PREDICTION_ID);
        assert!(slates.next().is_none());
        slate
    }

    #[test]
    fn served_slate_logs_organic_posts_in_served_order() {
        let decoded = PostCandidate {
            page_decode_slot: Some(0),
            page_decode_prob: Some(0.9),
            ..scored(1, 0.8, 5.0)
        };
        let explored = PostCandidate {
            page_rl_exploration_prob: Some(0.4),
            ..scored(2, 0.7, 4.0)
        };
        let retweet = PostCandidate {
            retweeted_tweet_id: Some(300),
            ..scored(3, 0.6, 3.0)
        };
        let candidates = vec![decoded, explored, retweet, scored(4, 0.5, 2.0)];
        let served = vec![
            module_item(feed_item::Item::PushToHome(
                xai_home_mixer_proto::PushToHomePost {
                    tweet_id: 99,
                    ..Default::default()
                },
            )),
            post_item(2, 0),
            module_item(feed_item::Item::Ad(xai_recsys_proto::AdIndexInfo::default())),
            post_item(1, 0),
            module_item(feed_item::Item::WhoToFollow(Default::default())),
            post_item(3, 300),
            module_item(feed_item::Item::Prompt(Default::default())),
            module_item(feed_item::Item::VideoCarousel(
                xai_home_mixer_proto::VideoCarouselModule {
                    videos: vec![ScoredPost {
                        tweet_id: 77,
                        prediction_request_id: PAGE_PREDICTION_ID,
                        ..Default::default()
                    }],
                },
            )),
            post_item(4, 0),
        ];

        for (decode_temperature, logged_decode_prob) in [("0.0", None), ("0.2", Some(0.9))] {
            let query = page_rl_query(SERVED_AT_MS, false, "true", decode_temperature);
            query
                .served_post_score_infos
                .set(served_post_score_infos(&query, &candidates))
                .unwrap();

            let slate = only_slate(served_slates(&query, &served));

            assert_eq!(slate.user_id, 42);
            assert_eq!(
                slate.product_surface,
                ProductSurface::HomeTimelineRanking as i32
            );
            assert_eq!(slate.request_timestamp_ms, Some(SERVED_AT_MS));
            assert_eq!(slate_tweet_ids(&slate), vec![2, 1, 300, 4]);
            let positions: Vec<u32> = slate
                .candidates
                .iter()
                .map(|c| c.tweet_timeline_position)
                .collect();
            assert_eq!(positions, vec![0, 1, 2, 3]);

            let info = |i: usize| slate.candidates[i].score_info.as_ref().unwrap();
            assert_eq!(info(0).weighted_score, Some(0.7));
            assert_eq!(info(0).final_score, Some(4.0));
            assert_eq!(info(0).reward_rerank_slot_prob, Some(0.4));
            assert_eq!(info(0).page_decode_slot_prob, None);
            assert_eq!(info(0).page_decoded, None);
            assert_eq!(info(1).weighted_score, Some(0.8));
            assert_eq!(info(1).final_score, Some(5.0));
            assert_eq!(info(1).reward_rerank_slot_prob, None);
            assert_eq!(info(1).page_decode_slot_prob, logged_decode_prob);
            assert_eq!(info(1).page_decoded, Some(true));
            for i in [2, 3] {
                assert_eq!(info(i).reward_rerank_slot_prob, None);
                assert_eq!(info(i).page_decode_slot_prob, None);
                assert_eq!(info(i).page_decoded, None);
            }
            assert_eq!(info(2).final_score, Some(3.0));
            assert!(slate.candidates.iter().all(|c| {
                let info = c.score_info.as_ref().unwrap();
                info.prediction_scores.is_empty() && info.slate_context.is_none()
            }));
        }
    }

    #[test]
    fn served_slate_keeps_at_most_64_organic_posts() {
        let query = page_rl_query(SERVED_AT_MS, false, "true", "0.0");
        let candidates: Vec<PostCandidate> = (1..=70).map(|id| scored(id, 1.0, 1.0)).collect();
        query
            .served_post_score_infos
            .set(served_post_score_infos(&query, &candidates))
            .unwrap();
        let served: Vec<FeedItem> = (1..=70)
            .flat_map(|id| {
                [
                    post_item(id, 0),
                    module_item(feed_item::Item::Ad(xai_recsys_proto::AdIndexInfo::default())),
                ]
            })
            .collect();

        let slate = only_slate(served_slates(&query, &served));

        assert_eq!(slate_tweet_ids(&slate), (1..=64).collect::<Vec<u64>>());
        assert!(slate
            .candidates
            .iter()
            .enumerate()
            .all(|(i, c)| c.tweet_timeline_position == i as u32));
    }

    #[test]
    fn served_slate_ignores_page_rl_fields_when_their_switches_are_off() {
        let query = page_rl_query(SERVED_AT_MS, true, "false", "0.2");
        let candidates = vec![PostCandidate {
            page_decode_slot: Some(0),
            page_decode_prob: Some(0.9),
            page_rl_exploration_prob: Some(0.4),
            ..scored(1, 0.8, 5.0)
        }];
        query
            .served_post_score_infos
            .set(served_post_score_infos(&query, &candidates))
            .unwrap();

        let slate = only_slate(served_slates(&query, &[post_item(1, 0)]));

        let info = slate.candidates[0].score_info.as_ref().unwrap();
        assert_eq!(info.final_score, Some(5.0));
        assert_eq!(info.reward_rerank_slot_prob, None);
        assert_eq!(info.page_decode_slot_prob, None);
        assert_eq!(info.page_decoded, None);
    }

    #[tokio::test]
    async fn fresh_and_cached_pages_append_to_their_prediction_id_page_log() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            ServedSlateCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client.clone());
        let other_prediction_id = PAGE_PREDICTION_ID + 1;
        let fresh = page_rl_query(SERVED_AT_MS, false, "true", "0.0");
        let fresh_candidates = vec![
            PostCandidate {
                page_decode_slot: Some(0),
                page_decode_prob: Some(1.0),
                ..scored(1, 0.9, 9.0)
            },
            scored(2, 0.8, 8.0),
        ];
        fresh
            .served_post_score_infos
            .set(served_post_score_infos(&fresh, &fresh_candidates))
            .unwrap();
        let cached = page_rl_query(SERVED_AT_MS + 30_000, true, "true", "0.0");
        let cached_candidates = vec![
            PostCandidate {
                page_decode_slot: Some(36),
                page_decode_prob: Some(1.0),
                ..scored(3, 0.7, 9.0)
            },
            scored(4, 0.6, 8.0),
            scored(5, 0.5, 7.0),
        ];
        cached
            .served_post_score_infos
            .set(served_post_score_infos(&cached, &cached_candidates))
            .unwrap();

        for (query, served) in [
            (
                cached,
                vec![
                    post_item(3, 0),
                    module_item(feed_item::Item::Ad(xai_recsys_proto::AdIndexInfo::default())),
                    post_of(other_prediction_id, 5, 0),
                    post_item(4, 0),
                ],
            ),
            (fresh, vec![post_item(1, 0), post_item(2, 0)]),
        ] {
            side_effect
                .side_effect(Arc::new(SideEffectInput {
                    query: Arc::new(query),
                    selected_candidates: served,
                    non_selected_candidates: vec![],
                }))
                .await
                .expect("page log write should succeed");
        }

        for redis_client in [&atla_redis_client, &pdxa_redis_client] {
            let pages = decode_page_log(redis_client, PAGE_PREDICTION_ID).await;
            let page_scores: Vec<i64> = pages.iter().map(|(score, _)| *score).collect();
            assert_eq!(page_scores, vec![SERVED_AT_MS, SERVED_AT_MS + 30_000]);
            assert_eq!(slate_tweet_ids(&pages[0].1), vec![1, 2]);
            assert_eq!(slate_tweet_ids(&pages[1].1), vec![3, 4]);
            assert_eq!(pages[1].1.candidates[1].tweet_timeline_position, 1);
            assert_eq!(
                pages[1].1.candidates[0]
                    .score_info
                    .as_ref()
                    .unwrap()
                    .page_decoded,
                Some(true)
            );
            let other = decode_page_log(redis_client, other_prediction_id).await;
            assert_eq!(other.len(), 1);
            assert_eq!(slate_tweet_ids(&other[0].1), vec![5]);
            assert_eq!(
                redis_client.zset_ttl(&served_page_log_key(PAGE_PREDICTION_ID)),
                Some(600)
            );
        }
    }

    #[tokio::test]
    async fn ranked_following_serving_event_appends_a_page_with_its_surface() {
        let atla_redis_client = Arc::new(MockRedisClient::default());
        let pdxa_redis_client = Arc::new(MockRedisClient::default());
        let side_effect =
            ServedSlateCacheSideEffect::new(atla_redis_client.clone(), pdxa_redis_client.clone());
        let query = ScoredPostsQuery {
            request_type: crate::models::query::RequestType::RankedFollowing,
            in_network_only: true,
            ..page_rl_query(SERVED_AT_MS, false, "true", "0.0")
        };
        let candidates = vec![
            scored(5, 0.9, 3.0),
            PostCandidate {
                retweeted_tweet_id: Some(600),
                ..scored(6, 0.8, 2.0)
            },
        ];
        query
            .served_post_score_infos
            .set(served_post_score_infos(&query, &candidates))
            .unwrap();
        let served = vec![
            post_item(6, 600),
            module_item(feed_item::Item::Ad(xai_recsys_proto::AdIndexInfo::default())),
            post_item(5, 0),
        ];

        side_effect
            .side_effect(Arc::new(SideEffectInput {
                query: Arc::new(query),
                selected_candidates: served,
                non_selected_candidates: vec![],
            }))
            .await
            .expect("page log write should succeed");

        for redis_client in [&atla_redis_client, &pdxa_redis_client] {
            let pages = decode_page_log(redis_client, PAGE_PREDICTION_ID).await;
            assert_eq!(pages.len(), 1);
            let slate = &pages[0].1;
            assert_eq!(
                slate.product_surface,
                ProductSurface::HomeTimelineRankedFollowing as i32
            );
            assert_eq!(slate_tweet_ids(slate), vec![600, 5]);
            let positions: Vec<u32> = slate
                .candidates
                .iter()
                .map(|c| c.tweet_timeline_position)
                .collect();
            assert_eq!(positions, vec![0, 1]);
            let final_scores: Vec<Option<f64>> = slate
                .candidates
                .iter()
                .map(|c| c.score_info.as_ref().unwrap().final_score)
                .collect();
            assert_eq!(final_scores, vec![Some(2.0), Some(3.0)]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn served_slate_side_effect_runs_in_for_you_and_ranked_following_feeds() {
        use crate::candidate_pipeline::for_you_candidate_pipeline::ForYouCandidatePipeline;
        use crate::candidate_pipeline::phoenix_candidate_pipeline::PhoenixCandidatePipeline;
        use crate::candidate_pipeline::ranked_following_candidate_pipeline::RankedFollowingCandidatePipeline;
        use crate::scored_posts_server::ScoredPostsServer;
        use crate::server::QueryBuilder;
        use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;

        xai_init_utils::init().rustls();
        let scored_posts_server = Arc::new(ScoredPostsServer::new(
            QueryBuilder::mock(),
            Arc::new(PhoenixCandidatePipeline::mock().await),
        ));
        let for_you = ForYouCandidatePipeline::mock(Arc::clone(&scored_posts_server)).await;
        let ranked_following = RankedFollowingCandidatePipeline::mock(scored_posts_server).await;

        for side_effects in [for_you.side_effects(), ranked_following.side_effects()] {
            assert!(side_effects
                .iter()
                .any(|side_effect| side_effect.name() == "ServedSlateCacheSideEffect"));
        }
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
