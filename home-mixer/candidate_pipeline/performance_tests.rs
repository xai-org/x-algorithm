use super::*;
use crate::models::query::RequestType;
use crate::models::user_features::UserFeatures;
use crate::scorers::author_cold_start::AuthorColdStart;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use xai_candidate_pipeline::candidate_pipeline::PipelineResult;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PredictNextActions,
};
use xai_candidate_pipeline::component_library::clients::redis_client::cached_posts_key;
use xai_candidate_pipeline::component_library::clients::MockThunderCapiClient;
use xai_core_entities::entities::PureCoreData;
use xai_decider::{Decider, DeciderStore};
use xai_feature_switches::RecipientBuilder;
use xai_recsys_proto::{
    ActionName, CandidateDistributionSet, NextActionDistribution, PredictNextActionsRequest,
    PredictNextActionsResponse, RetrievalDatasetType, RetrieveTopKCandidatesRequest,
    RetrieveTopKCandidatesResponse, ScoredCandidate, ScoredCandidates, TweetInfo,
};
use xai_vm_ranker_proto::{RankRequest, RankResponse, RankedCandidate};
use xai_x_thrift::tweet_media_info::TweetMediaInfo;

const COUNT: usize = 3000;
const VIEWER: u64 = 424242;

struct RetrievalFixture {
    tweets: Vec<TweetInfo>,
    calls: AtomicUsize,
}

#[async_trait]
impl PhoenixRetrievalClient for RetrievalFixture {
    async fn retrieve_request(
        &self,
        _cluster: xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster,
        request: RetrieveTopKCandidatesRequest,
        retries: u32,
    ) -> Result<RetrieveTopKCandidatesResponse, tonic::Status> {
        assert_eq!(request.user_ids, vec![VIEWER]);
        assert_eq!(retries, 0);
        assert_eq!(request.top_k as usize, COUNT);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(RetrieveTopKCandidatesResponse {
            top_k_candidates: vec![ScoredCandidates {
                user_id: VIEWER,
                candidates: self
                    .tweets
                    .iter()
                    .enumerate()
                    .map(|(i, tweet)| ScoredCandidate {
                        candidate: Some(tweet.clone()),
                        score: ((i * 691) % COUNT) as f32 / COUNT as f32,
                        dataset_type: RetrievalDatasetType::Home as u32,
                        source_idx: 0,
                    })
                    .collect(),
            }],
        })
    }
}

struct PredictionFixture {
    probabilities: HashMap<u64, f64>,
    calls: AtomicUsize,
}

#[async_trait]
impl PhoenixPredictionClient for PredictionFixture {
    async fn predict(
        &self,
        _cluster: PhoenixCluster,
        request: PredictNextActionsRequest,
        retries: u32,
    ) -> Result<PredictNextActions, tonic::Status> {
        assert_eq!(retries, 0);
        self.calls.fetch_add(1, Ordering::Relaxed);
        let candidates: Vec<_> = request
            .candidate_sets
            .into_iter()
            .flat_map(|set| set.candidates)
            .collect();
        assert!(candidates.len() <= 2800);
        Ok(PredictNextActions::new(PredictNextActionsResponse {
            distribution_sets: vec![CandidateDistributionSet {
                candidate_distributions: candidates
                    .into_iter()
                    .map(|tweet| NextActionDistribution {
                        index_to_logits: HashMap::from([
                            (
                                ActionName::ServerTweetFav as u32,
                                self.probabilities[&tweet.tweet_id].ln() as f32,
                            ),
                            (ActionName::ServerTweetReply as u32, -4.0),
                            (ActionName::ServerTweetRetweet as u32, -3.0),
                        ]),
                        candidate: Some(tweet),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }],
            ..Default::default()
        }))
    }
}

struct RankingFixture {
    scores: HashMap<u64, f64>,
    calls: AtomicUsize,
}

#[async_trait]
impl VMRankerClient for RankingFixture {
    async fn rank(
        &self,
        _cluster: crate::clients::vm_ranker_client::VMRankerCluster,
        request: RankRequest,
    ) -> Result<RankResponse, tonic::Status> {
        assert_eq!(request.viewer_id, VIEWER);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(RankResponse {
            candidates: request
                .candidates
                .into_iter()
                .map(|candidate| RankedCandidate {
                    tweet_id: candidate.tweet_id,
                    score: self.scores[&candidate.tweet_id],
                    weighted_score: candidate.weighted_score.or(candidate.score),
                    ..Default::default()
                })
                .collect(),
        })
    }
}

struct Fixture {
    base: u64,
    timestamp_ms: i64,
    tweets: Vec<TweetInfo>,
    core: HashMap<u64, Option<PureCoreData>>,
    media: HashMap<u64, Option<TweetMediaInfo>>,
    scores: HashMap<u64, f64>,
}

impl Fixture {
    fn new() -> Self {
        let timestamp_ms = std::env::var("HOME_MIXER_BENCH_TIMESTAMP_MS")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64
                    - 60_000
            });
        let base = ((timestamp_ms as u64 - 1_288_834_974_657) << 22) + 1000;
        let mut tweets = Vec::new();
        let mut core = HashMap::new();
        let mut media = HashMap::new();
        let mut scores = HashMap::new();
        for i in 0..COUNT {
            let id = base + i as u64;
            let author = 500_000 + (i % 600) as u64;
            let retweet = (i % 11 == 0).then_some(base + COUNT as u64 + i as u64);
            let parent = (i % 11 != 0 && i % 7 == 0).then_some(base - 10_000 - i as u64);
            let text = format!(
                "Post {i}: telescope observations, launch updates and Unicode café. {} {}",
                "Realistic content with contextual details. ".repeat(3),
                if i % 17 == 0 {
                    "muted_fixture_word"
                } else {
                    "visible_fixture_word"
                }
            );
            core.insert(
                id,
                Some(PureCoreData {
                    author_id: author,
                    text: text.clone(),
                    source_tweet_id: retweet,
                    source_user_id: retweet.map(|_| author + 10000),
                    in_reply_to_tweet_id: parent,
                    in_reply_to_user_id: parent.map(|_| author + 20000),
                    created_at_secs: Some(timestamp_ms / 1000),
                    conversation_id: parent,
                }),
            );
            if let Some(original) = retweet {
                core.insert(
                    original,
                    Some(PureCoreData {
                        author_id: author + 10000,
                        text,
                        ..Default::default()
                    }),
                );
            }
            if let Some(parent) = parent {
                core.insert(
                    parent,
                    Some(PureCoreData {
                        author_id: author + 20000,
                        text: format!("Parent of {i}"),
                        conversation_id: Some(parent),
                        ..Default::default()
                    }),
                );
            }
            let info = TweetMediaInfo {
                has_media: i % 3 == 0,
                has_photo: i % 3 == 0 && i % 6 != 0,
                has_video: i % 6 == 0,
                media_count: i64::from(i % 3 == 0),
                video_durations_ms: if i % 6 == 0 { vec![15_000] } else { vec![] },
            };
            media.insert(id, Some(info.clone()));
            if let Some(original) = retweet {
                media.insert(original, Some(info));
            }
            tweets.push(TweetInfo {
                tweet_id: id,
                author_id: author,
                retweeted_tweet_id: retweet.unwrap_or(0),
                in_reply_to_tweet_id: parent.unwrap_or(0),
                ..Default::default()
            });
            scores.insert(id, ((i * 691) % COUNT + 1) as f64 / (COUNT + 1) as f64);
        }
        Self {
            base,
            timestamp_ms,
            tweets,
            core,
            media,
            scores,
        }
    }

    fn query(&self, cached: bool) -> ScoredPostsQuery {
        let config: HashMap<String, serde_json::Value> =
            serde_json::from_str(include_str!("performance_params.json")).unwrap();
        let mut results = FeatureSwitches::new(vec![])
            .unwrap()
            .match_recipient(&RecipientBuilder::new().build());
        for (key, value) in config {
            let value = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
            results.override_fs(key, &value);
        }
        results.override_fs(
            "rust_home_mixer_enable_cached_posts".to_string(),
            if cached { "true" } else { "false" },
        );
        ScoredPostsQuery {
            user_id: VIEWER,
            request_type: RequestType::ScoredPosts,
            country_code: "US".to_string(),
            language_code: "en".to_string(),
            prediction_id: 77,
            request_id: 88,
            request_time_ms: self.timestamp_ms,
            scoring_sequence: Some(Default::default()),
            retrieval_sequence: Some(Default::default()),
            seen_ids: (0..30).map(|i| self.base + i).collect(),
            user_features: UserFeatures {
                muted_keywords: vec!["muted_fixture_word".to_string()],
                ..Default::default()
            },
            params: results.into(),
            decider: Some(
                Decider::new(DeciderStore::new(HashMap::new()))
                    .with_overrides(HashMap::from([
                        ("enable_vm_ranker_xds_traffic".to_string(), true),
                        ("enable_thunder_capi_home_mixer".to_string(), true),
                        ("kill_phoenix_ranker_home_mixer".to_string(), false),
                    ]))
                    .with_recipient(VIEWER),
            ),
            ..Default::default()
        }
    }

    async fn pipeline(
        &self,
    ) -> (
        PhoenixCandidatePipeline,
        Arc<MockRedisClient>,
        Arc<RetrievalFixture>,
        Arc<RankingFixture>,
    ) {
        xai_init_utils::init().rustls();
        let mut pipeline = PhoenixCandidatePipeline::mock().await;
        let tes = Arc::new(MockTESClient {
            core_data: self.core.clone(),
            language_code: self
                .core
                .keys()
                .map(|id| (*id, Some("en".to_string())))
                .collect(),
            ..Default::default()
        });
        let media = Arc::new(MockMediaInfoCacheClient {
            media_info: self.media.clone(),
        });
        let retrieval = Arc::new(RetrievalFixture {
            tweets: self.tweets.clone(),
            calls: AtomicUsize::new(0),
        });
        let ranking = Arc::new(RankingFixture {
            scores: self.scores.clone(),
            calls: AtomicUsize::new(0),
        });
        let mut probabilities = self.scores.clone();
        for tweet in &self.tweets {
            if tweet.retweeted_tweet_id != 0 {
                probabilities.insert(tweet.retweeted_tweet_id, self.scores[&tweet.tweet_id]);
            }
        }
        let prediction = Arc::new(PredictionFixture {
            probabilities,
            calls: AtomicUsize::new(0),
        });
        let redis = Arc::new(MockRedisClient::default());
        for source in &mut pipeline.sources {
            match source.name() {
                "PhoenixSource" => {
                    *source = Box::new(PhoenixSource {
                        phoenix_retrieval_client: retrieval.clone(),
                    })
                }
                "ThunderSource" => {
                    *source = Box::new(ThunderSource {
                        thunder_client: Arc::new(ThunderClient::mock()),
                        thunder_capi_client: Some(Arc::new(MockThunderCapiClient)),
                    })
                }
                _ => {}
            }
        }
        for hydrator in &mut pipeline.hydrators {
            match hydrator.name() {
                "CoreDataCandidateHydrator" => {
                    *hydrator = Box::new(CoreDataCandidateHydrator::new(tes.clone()).await)
                }
                "PhoenixReplyAncestorsHydrator" => {
                    *hydrator = Box::new(PhoenixReplyAncestorsHydrator {
                        tes_client: tes.clone(),
                    })
                }
                "MediaInfoHydrator" => {
                    *hydrator = Box::new(MediaInfoHydrator::new(media.clone()).await)
                }
                "LanguageCodeHydrator" => {
                    *hydrator = Box::new(LanguageCodeHydrator::new(tes.clone()).await)
                }
                "SubscriptionHydrator" => {
                    *hydrator = Box::new(SubscriptionHydrator::new(tes.clone()).await)
                }
                _ => {}
            }
        }
        for hydrator in &mut pipeline.query_hydrators {
            if hydrator.name() == "CachedPostsQueryHydrator" {
                *hydrator = Box::new(CachedPostsQueryHydrator {
                    redis_client: redis.clone(),
                });
            }
        }
        for scorer in &mut pipeline.scorers {
            match scorer.name() {
                "PhoenixScorer" => {
                    *scorer = Box::new(PhoenixScorer {
                        dispatch: crate::util::egress::PredictionDispatch {
                            client: prediction.clone(),
                            max_retries_key: "rust_home_mixer_phoenix_xds_max_retries",
                        },
                    })
                }
                "VMRanker" => {
                    *scorer = Box::new(VMRanker {
                        client: ranking.clone(),
                        xds_client: Some(ranking.clone()),
                        author_cold_start: AuthorColdStart {
                            author_rules: Arc::new(
                                xai_feature_switches::AuthorRulesEvaluator::new(Arc::new(
                                    FeatureSwitches::new(vec![]).unwrap(),
                                )),
                            ),
                        },
                        strato_client: Arc::new(MockStratoClient::default()),
                    })
                }
                _ => {}
            }
        }
        (pipeline, redis, retrieval, ranking)
    }

    fn assert_result(
        &self,
        result: &PipelineResult<ScoredPostsQuery, PostCandidate>,
        cached: bool,
    ) {
        assert_eq!(result.retrieved_candidates.len(), COUNT);
        assert_eq!(result.query.has_cached_posts, cached);
        let mut expected: Vec<_> = self
            .tweets
            .iter()
            .enumerate()
            .filter(|(i, _)| *i >= 30 && i % 11 != 0 && i % 17 != 0)
            .map(|(_, t)| t.tweet_id)
            .collect();
        expected.sort_by(|a, b| self.scores[b].total_cmp(&self.scores[a]));
        expected.truncate(params::RESULT_SIZE);
        assert_eq!(
            result
                .selected_candidates
                .iter()
                .map(|c| c.tweet_id)
                .collect::<Vec<_>>(),
            expected
        );
        for candidate in &result.selected_candidates {
            let core = self.core[&candidate.tweet_id].as_ref().unwrap();
            assert_eq!(candidate.author_id, core.author_id);
            assert_eq!(candidate.tweet_text, core.text);
            assert_eq!(candidate.score, Some(self.scores[&candidate.tweet_id]));
            assert_eq!(candidate.language_code.as_deref(), Some("en"));
            assert!(candidate.retrieval_sources.len() == 1);
        }
    }

    async fn seed_cache(&self, pipeline: &PhoenixCandidatePipeline, redis: &MockRedisClient) {
        let query = self.query(false);
        let retrieved = pipeline.fetch_candidates(&query).await;
        let hydrated = pipeline.hydrate(&query, retrieved).await;
        let scored = pipeline.score(&query, hydrated).await;
        assert_eq!(scored.len(), COUNT);
        let json = serde_json::to_vec(&scored).unwrap();
        let compressed = zstd::encode_all(json.as_slice(), 6).unwrap();
        redis
            .set_ex(cached_posts_key(VIEWER, &[], false, false), compressed, 180)
            .await
            .unwrap();
    }
}

fn fingerprint(result: &PipelineResult<ScoredPostsQuery, PostCandidate>) -> u64 {
    let normalize = |candidates: &[PostCandidate]| {
        candidates
            .iter()
            .cloned()
            .map(|mut c| {
                c.last_scored_at_ms = None;
                c
            })
            .collect::<Vec<_>>()
    };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_vec(&(
        normalize(&result.retrieved_candidates),
        normalize(&result.filtered_candidates),
        normalize(&result.selected_candidates),
    ))
    .unwrap()
    .hash(&mut hasher);
    hasher.finish()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn populated_pipeline_returns_expected_posts_repeatedly_for_fresh_and_cached_requests() {
    let fixture = Fixture::new();
    let (pipeline, redis, retrieval, ranking) = fixture.pipeline().await;
    let mut previous = None;
    for _ in 0..3 {
        let result = pipeline.execute(fixture.query(false)).await;
        fixture.assert_result(&result, false);
        let current = fingerprint(&result);
        assert!(previous.is_none_or(|previous| previous == current));
        previous = Some(current);
        tokio::task::yield_now().await;
    }
    fixture.seed_cache(&pipeline, &redis).await;
    let calls = retrieval.calls.load(Ordering::Relaxed);
    let mut previous = None;
    for _ in 0..3 {
        let before = ranking.calls.load(Ordering::Relaxed);
        let result = pipeline.execute(fixture.query(true)).await;
        fixture.assert_result(&result, true);
        assert_eq!(retrieval.calls.load(Ordering::Relaxed), calls);
        assert_eq!(ranking.calls.load(Ordering::Relaxed), before + 1);
        let current = fingerprint(&result);
        assert!(previous.is_none_or(|previous| previous == current));
        previous = Some(current);
        tokio::task::yield_now().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn returned_prediction_fields_and_cached_scores_match_their_actual_inputs() {
    let fixture = Fixture::new();
    let (pipeline, redis, _, _) = fixture.pipeline().await;
    let fresh = pipeline.execute(fixture.query(false)).await;
    fixture.assert_result(&fresh, false);
    for candidate in &fresh.selected_candidates {
        let probability = (fixture.scores[&candidate.tweet_id].ln() as f32 as f64).exp();
        assert_eq!(candidate.phoenix_scores.favorite_score, Some(probability));
        assert_eq!(
            candidate.weighted_score,
            Some(probability + xai_value_model::NEGATIVE_SCORES_OFFSET)
        );
        assert_eq!(
            candidate.prediction_request_id,
            Some(fresh.query.prediction_id)
        );
        assert!(candidate.last_scored_at_ms.is_some());
    }
    fixture.seed_cache(&pipeline, &redis).await;
    let payload = redis
        .get(cached_posts_key(VIEWER, &[], false, false))
        .await
        .unwrap();
    let decompressed = zstd::decode_all(payload.as_slice()).unwrap();
    let inputs: Vec<PostCandidate> = serde_json::from_slice(&decompressed).unwrap();
    let originals: HashMap<_, _> = inputs.into_iter().map(|c| (c.tweet_id, c)).collect();
    let cached = pipeline.execute(fixture.query(true)).await;
    fixture.assert_result(&cached, true);
    for candidate in &cached.selected_candidates {
        let original = &originals[&candidate.tweet_id];
        assert_eq!(
            candidate.phoenix_scores.favorite_score,
            original.phoenix_scores.favorite_score
        );
        assert_eq!(candidate.weighted_score, original.weighted_score);
        assert_eq!(
            candidate.prediction_request_id,
            original.prediction_request_id
        );
        assert_eq!(candidate.last_scored_at_ms, original.last_scored_at_ms);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn benchmark_populated_pipeline_3000() {
    let fixture = Fixture::new();
    println!(
        "pipeline_fixture timestamp_ms={} base={} candidates={COUNT} boundary=execute_traced external_clients=deterministic",
        fixture.timestamp_ms, fixture.base
    );
    for workload in ["fresh_cold", "fresh_warm", "viewer_cached"] {
        let (mut pipeline, redis, _, _) = fixture.pipeline().await;
        if workload == "viewer_cached" {
            fixture.seed_cache(&pipeline, &redis).await;
        }
        let mut timings = Vec::new();
        let mut previous = None;
        for sample in 0..35 {
            if workload == "fresh_cold" {
                pipeline = fixture.pipeline().await.0;
            }
            let query = fixture.query(workload == "viewer_cached");
            let start = Instant::now();
            let (result, trace) = std::hint::black_box(pipeline.execute_traced(query).await);
            let ns = start.elapsed().as_nanos();
            fixture.assert_result(&result, workload == "viewer_cached");
            let current = fingerprint(&result);
            assert!(previous.is_none_or(|previous| previous == current));
            previous = Some(current);
            if sample >= 5 {
                timings.push(ns);
                println!(
                    "pipeline_sample workload={workload} sample={} ns={ns} fingerprint={current}",
                    sample - 5
                );
                if sample == 5 {
                    for stage in trace.unwrap().stages {
                        println!(
                            "pipeline_stage workload={workload} name={} us={:?} input={:?}",
                            stage.name, stage.latency_us, stage.size
                        );
                        for component in stage.components {
                            println!(
                                "pipeline_component workload={workload} name={} us={} input={:?}",
                                component.name, component.latency_us, component.input_count
                            );
                        }
                    }
                }
            }
            tokio::task::yield_now().await;
        }
        timings.sort_unstable();
        let median = timings[timings.len() / 2];
        let mean = timings.iter().sum::<u128>() as f64 / timings.len() as f64;
        println!(
            "pipeline_result workload={workload} warmup=5 samples=30 median_ns={median} mean_ns={mean:.0} requests_per_second={:.2}",
            1_000_000_000.0 / mean
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fixture_preserves_production_wiring_and_required_components() {
    let fixture = Fixture::new();
    let (pipeline, _, _, _) = fixture.pipeline().await;
    let original = PhoenixCandidatePipeline::mock().await;
    let wiring = |pipeline: &PhoenixCandidatePipeline| {
        pipeline
            .components()
            .into_iter()
            .map(|stage| (format!("{:?}", stage.stage), stage.components))
            .collect::<Vec<_>>()
    };
    assert_eq!(wiring(&pipeline), wiring(&original));
    let names: std::collections::HashSet<_> = pipeline
        .components()
        .into_iter()
        .flat_map(|stage| stage.components)
        .collect();
    for required in [
        "PhoenixSource",
        "CachedPostsSource",
        "CachedPostsQueryHydrator",
        "CoreDataCandidateHydrator",
        "BlockedByHydrator",
        "CoreDataHydrationFilter",
        "AgeFilter",
        "AuthorSocialgraphFilter",
        "PreviouslySeenPostsFilter",
        "ViewerMutedKeywordFilter",
        "PhoenixScorer",
        "VMRanker",
        "TopKScoreSelector",
        "VFFilter",
        "AncillaryVFFilter",
    ] {
        assert!(
            names.contains(required),
            "missing required wired component: {required}"
        );
    }
}
