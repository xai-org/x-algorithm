use arc_swap::ArcSwap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
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
    self, BatchStage, CacheKeyCounts, CacheTier, LabelMetrics, ManhattanResult, OutageResult,
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

pub(crate) type StaleEntries = FxHashMap<u64, Arc<vf_pb::SafetyLabelMap>>;

fn label_cache_unavailable() -> LookupError {
    LookupError::new(
        FailureKind::LabelCacheUnavailable,
        "label cache read failed",
    )
}

#[async_trait]
pub(crate) trait RemoteLabels: Send + Sync {
                async fn get(&self, ids: &[u64], stale: &StaleEntries) -> LookupResults;

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
    async fn delete(&self, key: &Key) -> CacheOpResult<()>;
}

#[async_trait]
impl CacheWrite for CacheClient {
    async fn set(&self, key: &Key, value: Value, ttl_secs: u32) -> CacheOpResult<()> {
        CacheClient::set(self, key, value, ttl_secs).await
    }

    async fn delete(&self, key: &Key) -> CacheOpResult<()> {
        CacheClient::delete(self, key).await
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
        twemcache: Option<TwemcacheSource>,
    manhattan: ManhattanSource,
    dragonfly: Option<Arc<DragonflySource>>,
    switches: ArcSwap<LabelCacheSwitches>,
    now_ms: NowMs,
    metrics: LabelMetrics,
}

impl RemoteSource {
        pub(crate) fn new(
        twemcache: Option<Arc<dyn CacheRead>>,
        warmer: Option<Arc<CacheWarmer>>,
        fetcher: Arc<dyn ManhattanLabelFetcher>,
        metrics: LabelMetrics,
    ) -> Self {
        Self {
            twemcache: twemcache.map(|cache| TwemcacheSource {
                cache,
                warmer,
                metrics: metrics.clone(),
            }),
            manhattan: ManhattanSource {
                fetcher,
                metrics: metrics.clone(),
            },
            dragonfly: None,
            switches: ArcSwap::from_pointee(LabelCacheSwitches::default()),
            now_ms: Arc::new(|| system_unix_ms().cast_unsigned()),
            metrics,
        }
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
    async fn get(&self, ids: &[u64], stale: &StaleEntries) -> LookupResults {
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
        let twemcache_reads = async {
            match &self.twemcache {
                Some(twemcache) => twemcache.get(&unserved).await,
                None => unserved.iter().map(|&id| (id, None)).collect(),
            }
        };
        let (twemcache_reads, served_reads) = tokio::join!(twemcache_reads, served_reads);
        let mut results = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        let mut fallback_ids = Vec::new();
        for (tweet_id, labels) in twemcache_reads {
            match labels {
                Some(labels) => {
                    results.insert(tweet_id, Ok(labels));
                }
                None => fallback_ids.push(tweet_id),
            }
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
                    fills.extend(refresh);
                    results.insert(tweet_id, Ok(envelope.into_labels()));
                }
                EntryRead::Miss => {
                    fill_ids.insert(tweet_id);
                    fallback_ids.push(tweet_id);
                }
                EntryRead::Failed => {
                    let admitted = !stale.contains_key(&tweet_id)
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

        fallback_ids.sort_unstable();
        fallback_ids.dedup();
        metrics::record_batch_size(
            &self.metrics,
            BatchStage::ManhattanFallback,
            fallback_ids.len(),
        );
        let read_at = tokio::time::Instant::now();
        for (tweet_id, result) in self.manhattan.scan(&fallback_ids).await {
            let result = match result {
                Ok(Scan { items, labels }) => {
                    if fill_ids.contains(&tweet_id) {
                        fills.push(FillJob {
                            tweet_id,
                            fill: Fill::new(context.now_ms),
                            labels: FillLabels::Scanned { items, read_at },
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
            let mut shadowed: Vec<(u64, &vf_pb::SafetyLabelMap)> = unserved
                .iter()
                .filter(|&&id| context.switches.shadows(id))
                .filter_map(|&id| match results.get(&id) {
                    Some(Ok(labels)) => Some((id, labels)),
                    _ => None,
                })
                .collect();
            shadowed.sort_unstable_by_key(|&(id, _)| id);
            shadowed.dedup_by_key(|&mut (id, _)| id);
            if !shadowed.is_empty() {
                dragonfly.shadow(shadowed, context);
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

pub(crate) fn cache_key(id: u64) -> Key {
    prefixed_key(KEY_PREFIX, id)
}

pub(crate) fn prefixed_key(prefix: &str, id: u64) -> Key {
    let mut key = String::with_capacity(prefix.len() + 20);
    write!(key, "{prefix}{id}").ok();
    Key::from_bytes(key.into_bytes())
}

struct TwemcacheSource {
    cache: Arc<dyn CacheRead>,
    warmer: Option<Arc<CacheWarmer>>,
    metrics: LabelMetrics,
}

#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "KVCacheError belongs to the shared xai-cache client; only its timeout and backpressure errors get their own FallbackReason, and every other error counts as Other, including any the client adds"
)]
fn fallback_reason(e: &KVCacheError) -> FallbackReason {
    match e {
        KVCacheError::Timeout(_) => FallbackReason::Timeout,
        KVCacheError::Backpressure => FallbackReason::Backpressure,
        _ => FallbackReason::Other,
    }
}

pub(crate) fn cache_reply(
    reply: Option<&CacheOpResult<Option<Value>>>,
) -> Result<Option<&Value>, FallbackReason> {
    match reply {
        Some(Ok(value)) => Ok(value.as_ref()),
        Some(Err(e)) => Err(fallback_reason(e)),
        None => Err(FallbackReason::MissingResponse),
    }
}

fn twemcache_read(
    reply: Option<&CacheOpResult<Option<Value>>>,
) -> Result<CachedLabels, FallbackReason> {
    match cache_reply(reply)? {
        Some(bytes) => decode::cached_value(bytes).ok_or(FallbackReason::Decode),
        None => Ok(CachedLabels::Miss),
    }
}

impl TwemcacheSource {
            async fn get(&self, ids: &[u64]) -> Vec<(u64, Option<vf_pb::SafetyLabelMap>)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let start = Instant::now();
        let keys: Vec<Key> = ids.iter().map(|&id| cache_key(id)).collect();
        let replies = self.cache.multi_get(&keys).await;
        let mut counts = CacheKeyCounts::default();
        let mut fallbacks: BTreeMap<FallbackReason, usize> = BTreeMap::new();
        let mut warm_ids = Vec::new();
        let mut reads = Vec::with_capacity(ids.len());
        for (&tweet_id, key) in ids.iter().zip(&keys) {
            let labels = match twemcache_read(replies.get(key)) {
                Ok(CachedLabels::Found(labels)) => {
                    counts.hit += 1;
                    Some(labels)
                }
                Ok(CachedLabels::NotFound) => {
                    counts.not_found += 1;
                    Some(vf_pb::SafetyLabelMap::default())
                }
                Ok(CachedLabels::Miss) => {
                    counts.miss += 1;
                    if self.warmer.is_some() {
                        warm_ids.push(tweet_id);
                    }
                    None
                }
                Err(reason) => {
                    *fallbacks.entry(reason).or_default() += 1;
                    None
                }
            };
            reads.push((tweet_id, labels));
        }
        let m = &self.metrics;
        metrics::record_source_request(
            m,
            LabelSource::Twemcache,
            SourceOutcome::Success,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        metrics::record_cache_keys(m, CacheTier::Twemcache, &counts);
        metrics::record_cache_fallback_keys(m, LabelSource::Twemcache, fallbacks);
        if let Some(warmer) = &self.warmer
            && !warm_ids.is_empty()
        {
            warmer.warm(warm_ids);
        }
        reads
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
        let mut successes = 0usize;
        let request_outcome = match self.fetcher.fetch_labels(ids).await {
            Ok(fetch_results) => {
                let mut fetch_results = fetch_results.into_iter();
                for &tweet_id in ids {
                    let result = match fetch_results.next() {
                        Some(Ok(items)) => match decode::raw_labels(&items) {
                            Some(labels) => Ok(Scan { items, labels }),
                            None => Err(LookupError::new(
                                FailureKind::ManhattanDecode,
                                "decode panic",
                            )),
                        },
                        Some(Err(e)) => {
                            Err(LookupError::new(FailureKind::ManhattanFetch, e.to_string()))
                        }
                        None => Err(LookupError::new(
                            FailureKind::ManhattanFetch,
                            "missing from MH response",
                        )),
                    };
                    successes += usize::from(result.is_ok());
                    results.insert(tweet_id, result);
                }
                SourceOutcome::Success
            }
            Err(e) => {
                for &tweet_id in ids {
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
        metrics::record_manhattan_keys(
            &self.metrics,
            ManhattanResult::Failed,
            ids.len() - successes,
        );

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
                    .map(|(id, result)| (cache_key(id), result))
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
        items: HashMap<u64, Vec<RawSafetyLabel>>,
        per_key_errors: HashMap<u64, String>,
        batch_error: Option<String>,
        truncate_to: Option<usize>,
        calls: Mutex<Vec<Vec<u64>>>,
    }

    impl FakeLabelFetcher {
        fn with_label(mut self, tweet_id: u64, lt: SafetyLabelType, score: Option<f64>) -> Self {
            let mval = score.map_or_else(minimal_mval, mval_with_score);
            self.items
                .entry(tweet_id)
                .or_default()
                .push(raw_label(lt, mval));
            self
        }

        fn calls(&self) -> Vec<Vec<u64>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ManhattanLabelFetcher for FakeLabelFetcher {
        async fn fetch_labels(
            &self,
            tweet_ids: &[u64],
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
            Some(FakeCache::new([(42, Ok(None))]) as _),
            None,
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        );

        let results = source.get(&[42, 42], &StaleEntries::default()).await;

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
            Some(cache as _),
            Some(Arc::new(warmer)),
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        );

        let results = source.get(&[1, 2, 3, 4, 5], &StaleEntries::default()).await;

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
            Some(FakeCache::new([(42, Ok(None))]) as _),
            None,
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        );

        let results = source.get(&[42], &StaleEntries::default()).await;

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
            Some(FakeCache::new([(42, Ok(None))]) as _),
            Some(warmer),
            Arc::<FakeLabelFetcher>::clone(&fetcher),
            LabelMetrics::default(),
        );

        let results = source.get(&[42], &StaleEntries::default()).await;

        assert!(results.get(&42).unwrap().is_ok());
        assert_eq!(fetcher.calls(), vec![vec![42]]);
    }

    #[tokio::test]
    async fn cache_transport_results_select_lookup_outcome() {
        let cases = [
            (FakeCache::new([(42, Ok(None))]), Ok(CachedLabels::Miss)),
            (
                FakeCache::new([(42, Err(KVCacheError::Io("conn refused".into())))]),
                Err(FallbackReason::Other),
            ),
            (
                FakeCache::new([(42, Err(KVCacheError::Timeout("get".into())))]),
                Err(FallbackReason::Timeout),
            ),
            (
                FakeCache::new([(42, Err(KVCacheError::Backpressure))]),
                Err(FallbackReason::Backpressure),
            ),
            (FakeCache::new([]), Err(FallbackReason::MissingResponse)),
        ];

        for (cache, expected) in cases {
            let replies = cache.multi_get(&[cache_key(42)]).await;
            assert_eq!(twemcache_read(replies.get(&cache_key(42))), expected);
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
