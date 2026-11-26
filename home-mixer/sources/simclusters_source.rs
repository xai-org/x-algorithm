use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::clients::simclusters_ann_cache_client::SimClustersAnnCacheClient;
use crate::clients::simclusters_ann_client::SimClustersAnnClient;
use crate::models::candidate::{PostCandidate, RetrievalSource};
use crate::models::engagement_signals::EngagementSignal;
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableSimclustersSource, SimclustersMaxCandidateAgeHours};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use thrift::OrderedFloat;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{
    build_moka_cache, duration_since_creation_opt, MokaCache, MokaCacheConfig,
};
use xai_candidate_pipeline::hydrator::{CacheStore, TwoPhaseHydrator};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_stats_receiver::global_stats_receiver;
use xai_x_thrift::simclusters_ann::{
    EmbeddingType, InternalId, ModelVersion, Query, ScoringAlgorithm, SimClustersANNConfig,
    SimClustersANNTweetCandidate, SimClustersEmbeddingId,
};

const MAX_SANN_CANDIDATES: usize = 10_000;
const MAX_RESULTS: usize = 800;
const SOURCE_EMBEDDING_TYPE: EmbeddingType = EmbeddingType::LOG_FAV_LONGEST_L2_EMBEDDING_TWEET;
const CANDIDATE_EMBEDDING_TYPE: EmbeddingType = EmbeddingType::LOG_FAV_BASED_TWEET;
const MODEL_VERSION: ModelVersion = ModelVersion::MODEL_20M_145K_2020;
const ANN_MAX_NUM_RESULTS: i32 = 200;
const ANN_MIN_SCORE: f64 = 0.0;
const ANN_MAX_TOP_POSTS_PER_CLUSTER: i32 = 800;
const ANN_MAX_SCAN_CLUSTERS: i32 = 50;
const ANN_MIN_POST_CANDIDATE_AGE_HOURS: i32 = 0;
const POST_ANN_MIN_SCORE: f64 = 0.5;
const MAX_SEED_AGE: Duration = Duration::from_secs(90 * 24 * 60 * 60);
const CACHE_METRIC: &str = "SimclustersSource.cache";
const SHARED_CACHE_WRITE_METRIC: &str = "SimclustersSource.shared_cache_write";

pub struct SimclustersSource {
    client: Arc<dyn SimClustersAnnClient + Send + Sync>,
    shared_cache: Arc<dyn SimClustersAnnCacheClient>,
    cache: MokaCache<(i64, i32), Vec<SimClustersANNTweetCandidate>>,
    core_data_hydrator: CoreDataCandidateHydrator,
}

impl SimclustersSource {
    pub fn new(
        client: Arc<dyn SimClustersAnnClient + Send + Sync>,
        shared_cache: Arc<dyn SimClustersAnnCacheClient>,
        core_data_hydrator: CoreDataCandidateHydrator,
    ) -> Self {
        Self {
            client,
            shared_cache,
            cache: build_moka_cache(MokaCacheConfig {
                size: 2_000_000,
                ttl: Duration::from_secs(600),
            }),
            core_data_hydrator,
        }
    }

    async fn candidates_by_seed(
        &self,
        signal_ids: &[i64],
        max_candidate_age_hours: i32,
    ) -> Result<HashMap<i64, Vec<SimClustersANNTweetCandidate>>, String> {
        let mut stats = CacheStats::default();
        let mut found = HashMap::with_capacity(signal_ids.len());

        let mut pod_misses = Vec::new();
        for &signal_id in signal_ids {
            match self.cache.get(&(signal_id, max_candidate_age_hours)).await {
                Some(candidates) => {
                    stats.cache_hit += 1;
                    found.insert(signal_id, candidates);
                }
                None => pod_misses.push(signal_id),
            }
        }
        stats.cache_miss = pod_misses.len() as u64;

        let mut sann_seeds = Vec::new();
        if !pod_misses.is_empty() {
            let mut shared = self
                .shared_cache
                .multi_get(&pod_misses, max_candidate_age_hours)
                .await;
            for signal_id in pod_misses {
                match shared.remove(&signal_id) {
                    Some(Ok(Some(candidates))) => {
                        stats.shared_hit += 1;
                        self.cache
                            .insert((signal_id, max_candidate_age_hours), candidates.clone())
                            .await;
                        found.insert(signal_id, candidates);
                    }
                    Some(Err(_)) => stats.shared_error += 1,
                    Some(Ok(None)) | None => {
                        stats.shared_miss += 1;
                        sann_seeds.push(signal_id);
                    }
                }
            }
        }
        stats.emit();

        let results =
            futures::future::join_all(sann_seeds.into_iter().map(|signal_id| async move {
                let result = self
                    .client
                    .get_tweet_candidates(build_query(signal_id, max_candidate_age_hours))
                    .await;
                (signal_id, result)
            }))
            .await;

        let mut first_error = None;
        let mut fetched = Vec::new();
        for (signal_id, result) in results {
            match result {
                Ok(candidates) => {
                    let candidates = above_min_score(candidates);
                    self.cache
                        .insert((signal_id, max_candidate_age_hours), candidates.clone())
                        .await;
                    fetched.push((signal_id, candidates.clone()));
                    found.insert(signal_id, candidates);
                }
                Err(e) => {
                    first_error.get_or_insert_with(|| format!("SimclustersSource: {e}"));
                }
            }
        }
        self.write_shared_cache(fetched, max_candidate_age_hours);

        match first_error {
            Some(e) => Err(e),
            None => Ok(found),
        }
    }

    fn write_shared_cache(
        &self,
        entries: Vec<(i64, Vec<SimClustersANNTweetCandidate>)>,
        max_candidate_age_hours: i32,
    ) {
        if entries.is_empty() {
            return;
        }
        let shared_cache = Arc::clone(&self.shared_cache);
        tokio::spawn(async move {
            let results =
                futures::future::join_all(entries.iter().map(|(signal_id, candidates)| {
                    shared_cache.set(*signal_id, max_candidate_age_hours, candidates)
                }))
                .await;
            let failures = results.iter().filter(|result| result.is_err()).count();
            if let Some(receiver) = global_stats_receiver() {
                receiver.incr(
                    SHARED_CACHE_WRITE_METRIC,
                    &[("result", "success")],
                    (results.len() - failures) as u64,
                );
                receiver.incr(
                    SHARED_CACHE_WRITE_METRIC,
                    &[("result", "failure")],
                    failures as u64,
                );
            }
        });
    }
}

#[derive(Default)]
struct CacheStats {
    cache_hit: u64,
    cache_miss: u64,
    shared_hit: u64,
    shared_miss: u64,
    shared_error: u64,
}

impl CacheStats {
    fn emit(&self) {
        let Some(receiver) = global_stats_receiver() else {
            return;
        };
        for (label, count) in [
            ("cache_hit", self.cache_hit),
            ("cache_miss", self.cache_miss),
            ("shared_hit", self.shared_hit),
            ("shared_miss", self.shared_miss),
            ("shared_error", self.shared_error),
        ] {
            if count > 0 {
                receiver.incr(CACHE_METRIC, &[("requests", label)], count);
            }
        }
    }
}

fn above_min_score(
    candidates: Vec<SimClustersANNTweetCandidate>,
) -> Vec<SimClustersANNTweetCandidate> {
    candidates
        .into_iter()
        .filter(|c| *c.score > POST_ANN_MIN_SCORE)
        .collect()
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for SimclustersSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableSimclustersSource)
            && !query.in_network_only
            && !query.has_cached_posts
            && has_post_signals(query)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let signal_ids: Vec<i64> = post_signal_ids(query)
            .into_iter()
            .filter(|&signal_id| {
                duration_since_creation_opt(signal_id as u64).is_some_and(|age| age <= MAX_SEED_AGE)
            })
            .collect();
        if signal_ids.is_empty() {
            return Ok(vec![]);
        }

        let max_per_query = (MAX_SANN_CANDIDATES / signal_ids.len()).max(1);
        let max_candidate_age_hours: i32 = query.params.get(SimclustersMaxCandidateAgeHours);

        let mut candidates_by_seed = self
            .candidates_by_seed(&signal_ids, max_candidate_age_hours)
            .await?;
        let per_query_results: Vec<Vec<SimClustersANNTweetCandidate>> = signal_ids
            .iter()
            .filter_map(|signal_id| candidates_by_seed.remove(signal_id))
            .map(|candidates| {
                candidates
                    .into_iter()
                    .filter(|c| *c.score > POST_ANN_MIN_SCORE)
                    .take(max_per_query)
                    .collect()
            })
            .collect();

        let mut interleaved = interleave_by_post_id(per_query_results);
        interleaved.truncate(MAX_RESULTS);
        let mut candidates: Vec<PostCandidate> = interleaved
            .into_iter()
            .enumerate()
            .map(|(index, c)| PostCandidate {
                tweet_id: c.tweet_id as u64,
                served_type: Some(pb::ServedType::ForYouSimclusters),
                retrieval_sources: vec![RetrievalSource {
                    served_type: pb::ServedType::ForYouSimclusters,
                    cluster: None,
                    score: Some(*c.score as f32),
                    position: Some(index as u32 + 1),
                }],
                ..Default::default()
            })
            .collect();

        if let Ok(fetched) = self.core_data_hydrator.fetch(query, &candidates).await {
            self.core_data_hydrator
                .apply(query, &mut candidates, fetched);
        }
        candidates.retain(|c| c.author_id != 0);
        Ok(candidates)
    }
}

pub(crate) fn has_post_signals(query: &ScoredPostsQuery) -> bool {
    [
        query.explicit_engagement_signals.as_ref(),
        query.implicit_engagement_signals.as_ref(),
    ]
    .into_iter()
    .flatten()
    .any(|by_type| by_type.values().any(|list| !list.is_empty()))
}

pub(crate) fn post_signal_ids(query: &ScoredPostsQuery) -> Vec<i64> {
    let mut signals: Vec<&EngagementSignal> = Vec::new();
    for by_type in [
        query.explicit_engagement_signals.as_ref(),
        query.implicit_engagement_signals.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        for list in by_type.values() {
            signals.extend(list.iter());
        }
    }

    signals.sort_by_key(|b| std::cmp::Reverse(b.engaged_at_ms));

    let mut seen = HashSet::new();
    let mut ids = Vec::new();
    for signal in signals {
        if seen.insert(signal.tweet_id) {
            ids.push(signal.tweet_id);
        }
    }
    ids
}

fn build_query(signal_id: i64, max_candidate_age_hours: i32) -> Query {
    Query {
        source_embedding_id: SimClustersEmbeddingId {
            embedding_type: SOURCE_EMBEDDING_TYPE,
            model_version: MODEL_VERSION,
            internal_id: InternalId::TweetId(signal_id),
        },
        config: SimClustersANNConfig {
            max_num_results: ANN_MAX_NUM_RESULTS,
            min_score: OrderedFloat(ANN_MIN_SCORE),
            candidate_embedding_type: CANDIDATE_EMBEDDING_TYPE,
            max_top_tweets_per_cluster: ANN_MAX_TOP_POSTS_PER_CLUSTER,
            max_scan_clusters: ANN_MAX_SCAN_CLUSTERS,
            max_tweet_candidate_age_hours: max_candidate_age_hours,
            min_tweet_candidate_age_hours: ANN_MIN_POST_CANDIDATE_AGE_HOURS,
            ann_algorithm: ScoringAlgorithm::COSINE_SIMILARITY,
            engagement_threshold: None,
            is_cluster_detail_based_filtering_enabled: None,
            cluster_detail_based_threshold: None,
        },
    }
}

fn interleave_by_post_id(
    candidates: Vec<Vec<SimClustersANNTweetCandidate>>,
) -> Vec<SimClustersANNTweetCandidate> {
    let mut queues: Vec<VecDeque<_>> = candidates.into_iter().map(VecDeque::from).collect();
    let mut active: VecDeque<usize> = (0..queues.len()).collect();
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    while let Some(idx) = active.pop_front() {
        let Some(candidate) = queues[idx].pop_front() else {
            continue;
        };
        if seen.insert(candidate.tweet_id) {
            result.push(candidate);
            if !queues[idx].is_empty() {
                active.push_back(idx);
            }
        } else if !queues[idx].is_empty() {
            active.push_front(idx);
        }
    }

    result
}
