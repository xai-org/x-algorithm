use arc_swap::ArcSwap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use tonic::async_trait;
use tracing::warn;
use xai_cache::{CacheClient, KVCacheError, Key, Value};
use xai_visibility_filtering_proto as vf_pb;

use super::decode::{self, CachedLabels, RawSafetyLabel};
use super::dragonfly::{
    DragonflySource, EntryRead, Fill, FillJob, FillLabels, LabelCacheSwitches, ReadContext,
};
use super::metrics::{
    self, BatchStage, CacheResult, CacheTier, LabelMetrics, ManhattanResult, OutageResult,
    SourceOutcome,
};
use super::mh_client::ManhattanLabelFetcher;
use super::source::system_unix_ms;
use super::warmer::CacheWarmer;

const KEY_PREFIX: &str = "slm_";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum LabelSource {
    Twemcache,
    Dragonfly,
    Manhattan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum FallbackReason {
    Timeout,
    Backpressure,
    Decode,
    MissingResponse,
    Other,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, enum_map::Enum, strum::IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub enum FailureKind {
    ManhattanFetch,
    ManhattanDecode,
        LabelCacheUnavailable,
}

#[derive(Debug, PartialEq)]
pub(crate) enum TwemcacheOutcome {
    Hit(vf_pb::SafetyLabelMap),
    NotFound,
    Miss,
    FallThrough(FallbackReason),
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{kind:?}: {message}")]
pub struct LookupError {
    pub kind: FailureKind,
    pub(crate) message: String,
}

impl LookupError {
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

pub(crate) type LookupResults = FxHashMap<u64, Result<vf_pb::SafetyLabelMap, LookupError>>;

fn label_cache_unavailable() -> LookupError {
    LookupError::new(
        FailureKind::LabelCacheUnavailable,
        "label cache read failed",
    )
}

#[async_trait]
pub(crate) trait RemoteLabels: Send + Sync {
                async fn get(&self, ids: &[u64], stale: &FxHashSet<u64>) -> LookupResults;

    fn set_switches(&self, switches: LabelCacheSwitches);
}

pub(crate) type CacheOpResult<T> = std::result::Result<T, KVCacheError>;

#[async_trait]
pub(crate) trait CacheRead: Send + Sync + 'static {
    async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>>;
}

#[async_trait]
impl CacheRead for CacheClient {
    async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>> {
        match CacheClient::multi_get(self, keys).await {
            Ok(map) => map,
            Err(e) => keys.iter().cloned().map(|k| (k, Err(e.clone()))).collect(),
        }
    }
}

#[async_trait]
pub(crate) trait CacheWrite: CacheRead {
    async fn set(&self, key: &Key, value: Value, ttl_secs: u32) -> CacheOpResult<()>;
}

#[async_trait]
impl CacheWrite for CacheClient {
    async fn set(&self, key: &Key, value: Value, ttl_secs: u32) -> CacheOpResult<()> {
        CacheClient::set(self, key, value, ttl_secs).await
    }
}

pub(crate) type NowMs = Arc<dyn Fn() -> u64 + Send + Sync>;

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum ReplayOutcome {
    Found,
    NotFound,
    Failed,
}

pub(crate) struct RemoteSource {
    twemcache: TwemcacheSource,
    manhattan: ManhattanSource,
    warmer: Option<Arc<CacheWarmer>>,
    dragonfly: Option<Arc<DragonflySource>>,
    switches: ArcSwap<LabelCacheSwitches>,
    now_ms: NowMs,
    metrics: LabelMetrics,
}

impl RemoteSource {
    pub(crate) fn new(
        cache: Arc<dyn CacheRead>,
        fetcher: Arc<dyn ManhattanLabelFetcher>,
        metrics: LabelMetrics,
    ) -> Self {
        Self {
            twemcache: TwemcacheSource {
                cache,
                metrics: metrics.clone(),
            },
            manhattan: ManhattanSource {
                fetcher,
                metrics: metrics.clone(),
            },
            warmer: None,
            dragonfly: None,
            switches: ArcSwap::from_pointee(LabelCacheSwitches::default()),
            now_ms: Arc::new(|| system_unix_ms().cast_unsigned()),
            metrics,
        }
    }

    pub(crate) fn with_warmer(mut self, warmer: Arc<CacheWarmer>) -> Self {
        self.warmer = Some(warmer);
        self
    }

    pub(crate) fn with_dragonfly(mut self, cache: Arc<dyn CacheWrite>) -> Self {
        self.dragonfly = Some(DragonflySource::spawn(
            cache,
            self.manhattan.clone(),
            self.metrics.clone(),
        ));
        self
    }

    #[cfg(test)]
    pub(crate) fn with_test_dragonfly(
        mut self,
        dragonfly: Arc<DragonflySource>,
        now_ms: NowMs,
    ) -> Self {
        self.dragonfly = Some(dragonfly);
        self.now_ms = now_ms;
        self
    }

    #[cfg(test)]
    pub(crate) fn manhattan(&self) -> ManhattanSource {
        self.manhattan.clone()
    }
}

#[async_trait]
impl RemoteLabels for RemoteSource {
    async fn get(&self, ids: &[u64], stale: &FxHashSet<u64>) -> LookupResults {
        let context = ReadContext {
            now_ms: (self.now_ms)(),
            switches: **self.switches.load(),
        };
        let dragonfly = self
            .dragonfly
            .as_ref()
            .filter(|_| context.switches.serve_buckets > 0);
        let (served, unserved): (Vec<u64>, Cow<'_, [u64]>) = match dragonfly {
            Some(_) => {
                let (served, unserved) = ids.iter().partition(|&&id| context.switches.serves(id));
                (served, Cow::Owned(unserved))
            }
            None => (Vec::new(), Cow::Borrowed(ids)),
        };
        let served_reads = async {
            match dragonfly {
                Some(dragonfly) if !served.is_empty() => {
                    dragonfly
                        .read(&served, &context, CacheTier::Dragonfly)
                        .await
                }
                _ => Vec::new(),
            }
        };
        let (mut twemcache_results, served_reads) =
            tokio::join!(self.twemcache.get(&unserved), served_reads);
        let mut results = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        let mut fallback_ids = Vec::new();
        let mut fallback_counts: BTreeMap<FallbackReason, usize> = BTreeMap::new();
        let mut warm_ids = Vec::new();

        for &tweet_id in unserved.iter() {
            if results.contains_key(&tweet_id) || fallback_ids.contains(&tweet_id) {
                continue;
            }
            match twemcache_results.remove(&tweet_id) {
                Some(TwemcacheOutcome::Hit(label_map)) => {
                    results.insert(tweet_id, Ok(label_map));
                }
                Some(TwemcacheOutcome::NotFound) => {
                    results.insert(
                        tweet_id,
                        Ok(vf_pb::SafetyLabelMap {
                            labels: HashMap::new(),
                        }),
                    );
                }
                Some(TwemcacheOutcome::Miss) => {
                    fallback_ids.push(tweet_id);
                    if self.warmer.is_some() {
                        warm_ids.push(tweet_id);
                    }
                }
                Some(TwemcacheOutcome::FallThrough(reason)) => {
                    fallback_ids.push(tweet_id);
                    *fallback_counts.entry(reason).or_insert(0) += 1;
                }
                None => {
                    fallback_ids.push(tweet_id);
                    *fallback_counts
                        .entry(FallbackReason::MissingResponse)
                        .or_insert(0) += 1;
                }
            }
        }

        for (reason, count) in fallback_counts {
            metrics::record_cache_fallback_keys(
                &self.metrics,
                LabelSource::Twemcache,
                reason,
                count,
            );
        }

        if let Some(warmer) = &self.warmer
            && !warm_ids.is_empty()
        {
            warmer.warm(warm_ids);
        }

        let mut fills = Vec::new();
        let mut fill_ids = FxHashSet::default();
        let mut outage_reads = 0;
        for (tweet_id, read) in served_reads {
            if results.contains_key(&tweet_id) || fill_ids.contains(&tweet_id) {
                continue;
            }
            match read {
                EntryRead::Hit { envelope, refresh } => {
                    if let Some(fill) = refresh {
                        fills.push(FillJob {
                            tweet_id,
                            fill,
                            labels: FillLabels::Reread,
                        });
                    }
                    results.insert(tweet_id, Ok(envelope.into_labels()));
                }
                EntryRead::Miss => {
                    fill_ids.insert(tweet_id);
                    fallback_ids.push(tweet_id);
                }
                EntryRead::Failed => {
                    let admitted = !stale.contains(&tweet_id)
                        && dragonfly.is_some_and(|dragonfly| dragonfly.admit_outage_read());
                    if admitted {
                        outage_reads += 1;
                        fallback_ids.push(tweet_id);
                    } else {
                        results.insert(tweet_id, Err(label_cache_unavailable()));
                    }
                }
            }
        }
        metrics::record_outage_keys(&self.metrics, OutageResult::Manhattan, outage_reads);

        metrics::record_batch_size(
            &self.metrics,
            BatchStage::ManhattanFallback,
            fallback_ids.len(),
        );
        for (tweet_id, result) in self.manhattan.scan(&fallback_ids).await {
            let result = match result {
                Ok(Scan { items, labels }) => {
                    if fill_ids.contains(&tweet_id) {
                        fills.push(FillJob {
                            tweet_id,
                            fill: Fill::new(context.now_ms),
                            labels: FillLabels::Scanned(items),
                        });
                    }
                    Ok(labels)
                }
                Err(e) => Err(e),
            };
            results.insert(tweet_id, result);
        }
        if let Some(dragonfly) = dragonfly {
            dragonfly.fill(fills);
        }

        if let Some(dragonfly) = &self.dragonfly
            && context.switches.shadow_buckets > context.switches.serve_buckets
        {
            let mut shadowed: Vec<u64> = unserved
                .iter()
                .copied()
                .filter(|&id| context.switches.shadows(id))
                .filter(|id| matches!(results.get(id), Some(Ok(_))))
                .collect();
            shadowed.sort_unstable();
            shadowed.dedup();
            if !shadowed.is_empty()
                && let Some(permit) = dragonfly.shadow_permit(shadowed.len())
            {
                let answers = shadowed
                    .into_iter()
                    .filter_map(|id| match results.get(&id) {
                        Some(Ok(labels)) => Some((id, labels.clone())),
                        _ => None,
                    })
                    .collect();
                dragonfly.shadow(permit, answers, context);
            }
        }

        if self.dragonfly.is_some() && context.switches.replay_bucket.is_some() {
            log_replay(ids, &context, &results);
        }
        results
    }

    fn set_switches(&self, switches: LabelCacheSwitches) {
        self.switches.store(Arc::new(switches));
    }
}

fn log_replay(ids: &[u64], context: &ReadContext, results: &LookupResults) {
    let reads: Vec<(u64, &'static str)> = ids
        .iter()
        .filter(|&&id| context.switches.replays(id))
        .map(|&id| {
            let outcome = match results.get(&id) {
                Some(Ok(labels)) if labels.labels.is_empty() => ReplayOutcome::NotFound,
                Some(Ok(_)) => ReplayOutcome::Found,
                Some(Err(_)) | None => ReplayOutcome::Failed,
            };
            (id, outcome.into())
        })
        .collect();
    if !reads.is_empty() {
        warn!(time_ms = context.now_ms, ?reads, "label cache replay reads");
    }
}

pub(crate) fn cache_key(id: u64) -> Result<Key, KVCacheError> {
    Key::new(format!("{KEY_PREFIX}{id}").into_bytes())
}

struct TwemcacheSource {
    cache: Arc<dyn CacheRead>,
    metrics: LabelMetrics,
}

#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "KVCacheError belongs to the shared xai-cache client; only its timeout and backpressure errors get their own FallbackReason, and every other error counts as Other, including any the client adds"
)]
pub(crate) fn fallback_reason(e: &KVCacheError) -> FallbackReason {
    match e {
        KVCacheError::Timeout(_) => FallbackReason::Timeout,
        KVCacheError::Backpressure => FallbackReason::Backpressure,
        _ => FallbackReason::Other,
    }
}

#[derive(Default)]
struct ItemCounts {
    hit: usize,
    not_found: usize,
    miss: usize,
}

impl ItemCounts {
    fn count(&mut self, outcome: &TwemcacheOutcome) {
        match outcome {
            TwemcacheOutcome::Hit(_) => self.hit += 1,
            TwemcacheOutcome::NotFound => self.not_found += 1,
            TwemcacheOutcome::Miss => self.miss += 1,
            TwemcacheOutcome::FallThrough(_) => {}
        }
    }

    fn record_metrics(&self, metrics: &LabelMetrics) {
        metrics::record_cache_keys(metrics, CacheTier::Twemcache, CacheResult::Hit, self.hit);
        metrics::record_cache_keys(
            metrics,
            CacheTier::Twemcache,
            CacheResult::NotFound,
            self.not_found,
        );
        metrics::record_cache_keys(metrics, CacheTier::Twemcache, CacheResult::Miss, self.miss);
    }
}

impl TwemcacheSource {
    async fn get(&self, ids: &[u64]) -> FxHashMap<u64, TwemcacheOutcome> {
        let mut results = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        if ids.is_empty() {
            return results;
        }

        let start = Instant::now();

        let keys: Vec<Key> = match ids
            .iter()
            .map(|&id| cache_key(id))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(keys) => keys,
            Err(_) => {
                for &tweet_id in ids {
                    results.insert(
                        tweet_id,
                        TwemcacheOutcome::FallThrough(FallbackReason::Other),
                    );
                }
                metrics::record_source_request(
                    &self.metrics,
                    LabelSource::Twemcache,
                    SourceOutcome::Failure,
                    start.elapsed().as_secs_f64() * 1000.0,
                );
                return results;
            }
        };

        let mut counts = ItemCounts::default();

        let map = self.cache.multi_get(&keys).await;
        for (&tweet_id, key) in ids.iter().zip(&keys) {
            let outcome = match map.get(key) {
                Some(Ok(Some(bytes))) => match decode::cached_value(bytes) {
                    Some(CachedLabels::Found(label_map)) => TwemcacheOutcome::Hit(label_map),
                    Some(CachedLabels::NotFound) => TwemcacheOutcome::NotFound,
                    Some(CachedLabels::Miss) => TwemcacheOutcome::Miss,
                    None => TwemcacheOutcome::FallThrough(FallbackReason::Decode),
                },
                Some(Ok(None)) => TwemcacheOutcome::Miss,
                Some(Err(e)) => TwemcacheOutcome::FallThrough(fallback_reason(e)),
                None => TwemcacheOutcome::FallThrough(FallbackReason::MissingResponse),
            };
            counts.count(&outcome);
            results.insert(tweet_id, outcome);
        }
        metrics::record_source_request(
            &self.metrics,
            LabelSource::Twemcache,
            SourceOutcome::Success,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        counts.record_metrics(&self.metrics);

        results
    }
}

#[derive(Clone)]
pub(crate) struct ManhattanSource {
    fetcher: Arc<dyn ManhattanLabelFetcher>,
    metrics: LabelMetrics,
}

pub(crate) struct Scan {
    pub(crate) items: Vec<RawSafetyLabel>,
    pub(crate) labels: vf_pb::SafetyLabelMap,
}

impl ManhattanSource {
    pub(crate) async fn scan(&self, ids: &[u64]) -> FxHashMap<u64, Result<Scan, LookupError>> {
        let mut results = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        if ids.is_empty() {
            return results;
        }

        let start = Instant::now();
        let i64_ids: Vec<i64> = ids.iter().map(|&id| id.cast_signed()).collect();
        let mut successes = 0usize;
        let mut failures = 0usize;

        let request_outcome = match self.fetcher.fetch_labels(&i64_ids).await {
            Ok(fetch_results) => {
                let mut fetch_results = fetch_results.into_iter();
                for &tweet_id in ids {
                    match fetch_results.next() {
                        Some(Ok(items)) => match decode::raw_labels(&items) {
                            Some(labels) => {
                                successes += 1;
                                results.insert(tweet_id, Ok(Scan { items, labels }));
                            }
                            None => {
                                failures += 1;
                                results.insert(
                                    tweet_id,
                                    Err(LookupError::new(
                                        FailureKind::ManhattanDecode,
                                        "decode panic",
                                    )),
                                );
                            }
                        },
                        Some(Err(e)) => {
                            failures += 1;
                            results.insert(
                                tweet_id,
                                Err(LookupError::new(FailureKind::ManhattanFetch, e.to_string())),
                            );
                        }
                        None => {
                            failures += 1;
                            results.insert(
                                tweet_id,
                                Err(LookupError::new(
                                    FailureKind::ManhattanFetch,
                                    "missing from MH response",
                                )),
                            );
                        }
                    }
                }
                SourceOutcome::Success
            }
            Err(e) => {
                for &tweet_id in ids {
                    failures += 1;
                    results.insert(
                        tweet_id,
                        Err(LookupError::new(FailureKind::ManhattanFetch, e.to_string())),
                    );
                }
                SourceOutcome::Failure
            }
        };

        metrics::record_source_request(
            &self.metrics,
            LabelSource::Manhattan,
            request_outcome,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        metrics::record_manhattan_keys(&self.metrics, ManhattanResult::Success, successes);
        metrics::record_manhattan_keys(&self.metrics, ManhattanResult::Failed, failures);

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use xai_manhattan::ManhattanError;
    use xai_x_thrift::tweet_safety_label::SafetyLabelType;

    use crate::decode::RawSafetyLabel;
    use crate::decode::tests::{
        cached_value_found_mval, cached_value_not_found, minimal_mval, mval_with_score, raw_label,
    };
    use crate::mh_client::FetchResult;

    struct FakeCache {
        results: HashMap<Key, CacheOpResult<Option<Value>>>,
    }

    impl FakeCache {
        fn new<const N: usize>(entries: [(u64, CacheOpResult<Option<Value>>); N]) -> Arc<Self> {
            Arc::new(Self {
                results: entries
                    .into_iter()
                    .map(|(id, result)| (cache_key(id).unwrap(), result))
                    .collect(),
            })
        }
    }

    #[async_trait]
    impl CacheRead for FakeCache {
        async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>> {
            keys.iter()
                .filter_map(|k| self.results.get(k).map(|r| (k.clone(), r.clone())))
                .collect()
        }
    }

    #[derive(Default)]
    struct FakeLabelFetcher {
        items: HashMap<i64, Vec<RawSafetyLabel>>,
        per_key_errors: HashMap<i64, String>,
        batch_error: Option<String>,
        truncate_to: Option<usize>,
        calls: Mutex<Vec<Vec<i64>>>,
    }

    impl FakeLabelFetcher {
        fn with_label(mut self, tweet_id: i64, lt: SafetyLabelType, score: Option<f64>) -> Self {
            let mval = score.map_or_else(minimal_mval, mval_with_score);
            self.items
                .entry(tweet_id)
                .or_default()
                .push(raw_label(lt, mval));
            self
        }

        fn calls(&self) -> Vec<Vec<i64>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ManhattanLabelFetcher for FakeLabelFetcher {
        async fn fetch_labels(
            &self,
            tweet_ids: &[i64],
        ) -> Result<Vec<FetchResult>, ManhattanError> {
            self.calls.lock().unwrap().push(tweet_ids.to_vec());
            if let Some(msg) = &self.batch_error {
                return Err(ManhattanError::NativeProtocol(msg.clone()));
            }
            let mut out: Vec<FetchResult> = tweet_ids
                .iter()
                .map(|id| match self.per_key_errors.get(id) {
                    Some(msg) => Err(ManhattanError::NativeProtocol(msg.clone())),
                    None => Ok(self.items.get(id).cloned().unwrap_or_default()),
                })
                .collect();
            if let Some(len) = self.truncate_to {
                out.truncate(len);
            }
            Ok(out)
        }
    }

    #[tokio::test]
    async fn duplicate_ids_resolve_once_without_a_phantom_miss() {
        let fetcher = Arc::new(FakeLabelFetcher::default());
        let source = RemoteSource::new(
            FakeCache::new([(42, Ok(None))]),
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        );

        let results = source.get(&[42, 42], &FxHashSet::default()).await;

        assert!(results.get(&42).unwrap().is_ok());
        assert_eq!(fetcher.calls(), vec![vec![42]]);
    }

    #[tokio::test]
    async fn plain_miss_publishes_to_warmer() {
        let cache = FakeCache::new([
            (1, Ok(Some(cached_value_found_mval()))),
            (2, Ok(Some(cached_value_not_found()))),
            (3, Ok(None)),
            (4, Err(KVCacheError::Timeout("get".into()))),
        ]);
        let fetcher = Arc::new(FakeLabelFetcher::default());
        let (warmer, mut published) = CacheWarmer::without_drain_task(2);
        let source = RemoteSource::new(
            cache,
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        )
        .with_warmer(Arc::new(warmer));

        let results = source.get(&[1, 2, 3, 4, 5], &FxHashSet::default()).await;

        assert_eq!(results.len(), 5);
        assert!(results.values().all(Result::is_ok));
        assert!(
            results[&1]
                .as_ref()
                .unwrap()
                .labels
                .contains_key(&i32::from(SafetyLabelType::NSFA_HIGH_PRECISION))
        );
        assert!(results[&2].as_ref().unwrap().labels.is_empty());
        assert_eq!(fetcher.calls(), vec![vec![3, 4, 5]]);
        assert_eq!(published.try_recv().unwrap(), vec![3]);
        assert!(published.try_recv().is_err());
    }

    #[tokio::test]
    async fn a_miss_manhattan_does_not_answer_is_a_fetch_error() {
        let fetcher = Arc::new(FakeLabelFetcher {
            truncate_to: Some(0),
            ..Default::default()
        });
        let source = RemoteSource::new(
            FakeCache::new([(42, Ok(None))]),
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        );

        let results = source.get(&[42], &FxHashSet::default()).await;

        assert!(matches!(
            results.get(&42),
            Some(Err(failure)) if failure.kind == FailureKind::ManhattanFetch
        ));
        assert_eq!(fetcher.calls(), vec![vec![42]]);
    }

    #[tokio::test]
    async fn full_warm_channel_does_not_affect_fallback_result() {
        let (warmer, _rx) = CacheWarmer::without_drain_task(1);
        let warmer = Arc::new(warmer);
        warmer.warm(vec![0]);

        let fetcher = Arc::new(FakeLabelFetcher::default());
        let source = RemoteSource::new(
            FakeCache::new([(42, Ok(None))]),
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        )
        .with_warmer(warmer);

        let results = source.get(&[42], &FxHashSet::default()).await;

        assert!(results.get(&42).unwrap().is_ok());
        assert_eq!(fetcher.calls(), vec![vec![42]]);
    }

    #[tokio::test]
    async fn cache_transport_results_select_lookup_outcome() {
        let cases = [
            (FakeCache::new([(42, Ok(None))]), TwemcacheOutcome::Miss),
            (
                FakeCache::new([(42, Err(KVCacheError::Io("conn refused".into())))]),
                TwemcacheOutcome::FallThrough(FallbackReason::Other),
            ),
            (
                FakeCache::new([(42, Err(KVCacheError::Timeout("get".into())))]),
                TwemcacheOutcome::FallThrough(FallbackReason::Timeout),
            ),
            (
                FakeCache::new([(42, Err(KVCacheError::Backpressure))]),
                TwemcacheOutcome::FallThrough(FallbackReason::Backpressure),
            ),
            (
                FakeCache::new([]),
                TwemcacheOutcome::FallThrough(FallbackReason::MissingResponse),
            ),
        ];

        for (cache, expected) in cases {
            let results = TwemcacheSource {
                cache,
                metrics: LabelMetrics::default(),
            }
            .get(&[42])
            .await;
            assert_eq!(results.get(&42), Some(&expected));
        }
    }

    async fn get_with_fetcher(fetcher: FakeLabelFetcher, ids: &[u64]) -> LookupResults {
        ManhattanSource {
            fetcher: Arc::new(fetcher),
            metrics: LabelMetrics::default(),
        }
        .scan(ids)
        .await
        .into_iter()
        .map(|(tweet_id, result)| (tweet_id, result.map(|scan| scan.labels)))
        .collect()
    }

    #[tokio::test]
    async fn manhattan_answers_resolve_or_fail_each_id() {
        let fetch_failed = |result: Option<&Result<_, LookupError>>, message: &str| {
            matches!(result, Some(Err(failure))
                if failure.kind == FailureKind::ManhattanFetch && failure.message.contains(message))
        };

        let empty = get_with_fetcher(FakeLabelFetcher::default(), &[100]).await;
        assert!(matches!(empty.get(&100), Some(Ok(label_map)) if label_map.labels.is_empty()));

        let per_key_error = FakeLabelFetcher {
            per_key_errors: HashMap::from([(2, "key failed".to_owned())]),
            ..Default::default()
        }
        .with_label(1, SafetyLabelType::SPAM, Some(0.5));
        let results = get_with_fetcher(per_key_error, &[1, 2]).await;
        assert!(matches!(
            results.get(&1),
            Some(Ok(label_map)) if label_map.labels.contains_key(&i32::from(SafetyLabelType::SPAM))
        ));
        assert!(fetch_failed(results.get(&2), "key failed"));

        let batch_error = FakeLabelFetcher {
            batch_error: Some("test error".to_owned()),
            ..Default::default()
        };
        let results = get_with_fetcher(batch_error, &[1, 2, 3]).await;
        for id in [1, 2, 3] {
            assert!(fetch_failed(results.get(&id), "test error"));
        }

        let short_response = FakeLabelFetcher {
            truncate_to: Some(2),
            ..Default::default()
        }
        .with_label(1, SafetyLabelType::SPAM, Some(0.5))
        .with_label(2, SafetyLabelType::SPAM, None);
        let results = get_with_fetcher(short_response, &[1, 2, 3]).await;
        assert_eq!(results.len(), 3);
        assert!(results.get(&1).unwrap().is_ok());
        assert!(results.get(&2).unwrap().is_ok());
        assert!(fetch_failed(results.get(&3), "missing from MH response"));
    }
}
