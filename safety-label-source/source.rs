use quanta::Clock;
use quick_cache::sync::Cache;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::info;
use xai_visibility_filtering_proto as vf_pb;

use super::metrics::{self, BatchStage, CacheResult, CacheTier, LabelMetrics};
use super::remote::{LookupError, LookupResults, RemoteLabels};

const TTL: Duration = Duration::from_secs(10);

fn tweet_timestamp_ms(tweet_id: u64) -> u64 {
    const SNOWFLAKE_EPOCH_MS: u64 = 1288834974657;
    const SNOWFLAKE_TIMESTAMP_SHIFT: u32 = 22;
    (tweet_id >> SNOWFLAKE_TIMESTAMP_SHIFT) + SNOWFLAKE_EPOCH_MS
}

fn is_future_dated(tweet_id: u64, now: SystemTime) -> bool {
    UNIX_EPOCH
        .checked_add(Duration::from_millis(tweet_timestamp_ms(tweet_id)))
        .is_none_or(|created_at| created_at > now)
}

fn system_unix_ms() -> i64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    i64::try_from(millis).unwrap_or(i64::MAX)
}

fn label_is_live(expires_at_msec: Option<i64>, now_ms: i64) -> bool {
    match expires_at_msec {
        Some(expires_at_msec) => expires_at_msec.wrapping_sub(now_ms) > 0,
        None => true,
    }
}

fn drop_expired_labels(map: &vf_pb::SafetyLabelMap, now_ms: i64) -> vf_pb::SafetyLabelMap {
    vf_pb::SafetyLabelMap {
        labels: map
            .labels
            .iter()
            .filter(|(_, label)| label_is_live(label.expires_at_msec, now_ms))
            .map(|(id, label)| (*id, label.clone()))
            .collect(),
    }
}

#[derive(Clone)]
struct Entry {
    labels: Arc<vf_pb::SafetyLabelMap>,
    expires_at: quanta::Instant,
}

pub struct SafetyLabelSource {
        cache: Option<Cache<u64, Entry>>,
    clock: Clock,
    remote: Arc<dyn RemoteLabels>,
    now_ms: Arc<dyn Fn() -> i64 + Send + Sync>,
    metrics: LabelMetrics,
}

impl SafetyLabelSource {
    pub(crate) fn new(
        remote: Arc<dyn RemoteLabels>,
        cache_capacity: Option<usize>,
        metrics: LabelMetrics,
    ) -> Self {
        Self::with_clock(remote, cache_capacity, Clock::new(), metrics)
    }

    fn with_clock(
        remote: Arc<dyn RemoteLabels>,
        cache_capacity: Option<usize>,
        clock: Clock,
        metrics: LabelMetrics,
    ) -> Self {
        Self::with_clocks(
            remote,
            cache_capacity,
            clock,
            Arc::new(system_unix_ms),
            metrics,
        )
    }

    fn with_clocks(
        remote: Arc<dyn RemoteLabels>,
        cache_capacity: Option<usize>,
        clock: Clock,
        now_ms: Arc<dyn Fn() -> i64 + Send + Sync>,
        metrics: LabelMetrics,
    ) -> Self {
        info!(
            cache_capacity = cache_capacity.unwrap_or(0),
            ttl_secs = TTL.as_secs(),
            "Local safety-label cache initialized (single pool, per-entry TTL)"
        );
        Self {
            cache: cache_capacity.map(|capacity| {
                let cache = Cache::new(capacity);
                cache.reserve(capacity);
                cache
            }),
            clock,
            remote,
            now_ms,
            metrics,
        }
    }

    pub async fn get(
        &self,
        ids: &[u64],
    ) -> FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>> {
        let total = ids.len();
        let mut results: FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>> =
            FxHashMap::with_capacity_and_hasher(total, Default::default());

        let (local_misses, expired) = self.get_local(ids, &mut results);
        let batch_size = local_misses.len();

        let remote_results = self.remote.get(&local_misses).await;
        self.backfill_local(remote_results, &mut results);

        self.emit_stats(total - batch_size, batch_size, expired);

        let now_ms = (self.now_ms)();
        let mut dropped = 0;
        for labels in results.values_mut() {
            let Ok(map) = labels else { continue };
            if map
                .labels
                .values()
                .any(|label| !label_is_live(label.expires_at_msec, now_ms))
            {
                let live = drop_expired_labels(map, now_ms);
                dropped += map.labels.len() - live.labels.len();
                *map = Arc::new(live);
            }
        }
        metrics::record_expired_labels(&self.metrics, dropped);

        results
    }

    fn get_local(
        &self,
        ids: &[u64],
        results: &mut FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
    ) -> (Vec<u64>, usize) {
        let Some(cache) = &self.cache else {
            return (ids.to_vec(), 0);
        };
        let mut misses = Vec::with_capacity(ids.len());
        let mut expired = 0;
        for &tweet_id in ids {
            let now = self.clock.now();
            match cache.get(&tweet_id) {
                Some(entry) if now < entry.expires_at => {
                    results.insert(tweet_id, Ok(entry.labels));
                }
                Some(_) => {
                    expired += 1;
                    misses.push(tweet_id);
                }
                None => misses.push(tweet_id),
            }
        }
        (misses, expired)
    }

    fn backfill_local(
        &self,
        remote_results: LookupResults,
        results: &mut FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
    ) {
        let wall_now = SystemTime::now();
        for (id, result) in remote_results {
            let result = result.map(Arc::new);
            if let Some(cache) = &self.cache
                && let Ok(label_map) = &result
                && !is_future_dated(id, wall_now)
            {
                cache.insert(
                    id,
                    Entry {
                        labels: Arc::clone(label_map),
                        expires_at: self.clock.now() + TTL,
                    },
                );
            }
            results.insert(id, result);
        }
    }

    fn emit_stats(&self, hits: usize, batch_size: usize, expired: usize) {
        let metrics = &self.metrics;
        metrics::record_cache_keys(metrics, CacheTier::Local, CacheResult::Hit, hits);
        metrics::record_cache_keys(
            metrics,
            CacheTier::Local,
            CacheResult::Miss,
            batch_size - expired,
        );
        metrics::record_cache_keys(metrics, CacheTier::Local, CacheResult::Expired, expired);
        metrics::record_batch_size(metrics, BatchStage::LocalMiss, batch_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::remote::FailureKind;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
    use tonic::async_trait;

    struct FakeRemote {
        results: HashMap<u64, Result<vf_pb::SafetyLabelMap, LookupError>>,
        fetched_keys: AtomicUsize,
    }

    impl FakeRemote {
        fn new(results: HashMap<u64, Result<vf_pb::SafetyLabelMap, LookupError>>) -> Arc<Self> {
            Arc::new(Self {
                results,
                fetched_keys: AtomicUsize::new(0),
            })
        }

        fn fetched_keys(&self) -> usize {
            self.fetched_keys.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl RemoteLabels for FakeRemote {
        async fn get(&self, ids: &[u64]) -> LookupResults {
            self.fetched_keys.fetch_add(ids.len(), Ordering::SeqCst);
            ids.iter()
                .filter_map(|id| self.results.get(id).map(|result| (*id, result.clone())))
                .collect()
        }
    }

    fn proto_label(expires_at_msec: Option<i64>) -> vf_pb::SafetyLabel {
        vf_pb::SafetyLabel {
            expires_at_msec,
            ..vf_pb::SafetyLabel::default()
        }
    }

    fn found(tweet_id: u64) -> Arc<FakeRemote> {
        FakeRemote::new(HashMap::from([(
            tweet_id,
            Ok(vf_pb::SafetyLabelMap {
                labels: HashMap::from([(1, proto_label(None))]),
            }),
        )]))
    }

    fn l1_expiry(source: &SafetyLabelSource, tweet_id: u64) -> Option<quanta::Instant> {
        source
            .cache
            .as_ref()
            .and_then(|cache| cache.get(&tweet_id))
            .map(|entry| entry.expires_at)
    }

    fn tweet_id_created_at(created_at: SystemTime) -> u64 {
        let since_epoch = created_at.duration_since(UNIX_EPOCH).unwrap();
        let created_ms = since_epoch
            .as_secs()
            .checked_mul(1000)
            .and_then(|secs_ms| secs_ms.checked_add(u64::from(since_epoch.subsec_millis())))
            .unwrap();
        created_ms.checked_sub(tweet_timestamp_ms(0)).unwrap() << 22
    }

    #[tokio::test]
    async fn get_backfills_remote_successes_into_l1() {
        let remote = found(42);
        let source = SafetyLabelSource::new(
            Arc::<FakeRemote>::clone(&remote),
            Some(1024),
            LabelMetrics::default(),
        );

        let results1 = source.get(&[42]).await;
        let first = Arc::clone(results1.get(&42).unwrap().as_ref().unwrap());
        assert_eq!(remote.fetched_keys(), 1);
        assert!(l1_expiry(&source, 42).is_some());

        let results2 = source.get(&[42]).await;
        assert!(Arc::ptr_eq(
            results2.get(&42).unwrap().as_ref().unwrap(),
            &first
        ));
        assert_eq!(remote.fetched_keys(), 1);
    }

    #[tokio::test]
    async fn get_never_backfills_a_remote_error_into_l1() {
        let remote = FakeRemote::new(HashMap::from([(
            42,
            Err(LookupError::new(FailureKind::ManhattanFetch, "down")),
        )]));
        let source = SafetyLabelSource::new(
            Arc::<FakeRemote>::clone(&remote),
            Some(1024),
            LabelMetrics::default(),
        );

        let results1 = source.get(&[42]).await;
        assert!(results1.get(&42).unwrap().is_err());
        assert!(l1_expiry(&source, 42).is_none());

        let results2 = source.get(&[42]).await;
        assert!(results2.get(&42).unwrap().is_err());
        assert_eq!(remote.fetched_keys(), 2);
    }

    #[tokio::test]
    async fn without_an_l1_every_read_goes_remote() {
        let remote = found(42);
        let source = SafetyLabelSource::new(
            Arc::<FakeRemote>::clone(&remote),
            None,
            LabelMetrics::default(),
        );

        for _ in 0..2 {
            assert!(source.get(&[42]).await.get(&42).unwrap().is_ok());
        }
        assert_eq!(remote.fetched_keys(), 2);
    }

    #[tokio::test]
    async fn get_does_not_backfill_future_tweet_id_into_l1() {
        let tweet_id = tweet_id_created_at(SystemTime::now() + Duration::from_secs(60));
        let source = SafetyLabelSource::new(found(tweet_id), Some(1024), LabelMetrics::default());

        let results = source.get(&[tweet_id]).await;

        assert!(results.get(&tweet_id).unwrap().is_ok());
        assert!(l1_expiry(&source, tweet_id).is_none());
    }

    #[tokio::test]
    async fn entry_at_ttl_is_refetched_and_restamped() {
        let (clock, mock) = Clock::mock();
        let tweet_id = tweet_id_created_at(SystemTime::now() - Duration::from_secs(60));
        let remote = found(tweet_id);
        let source = SafetyLabelSource::with_clock(
            Arc::<FakeRemote>::clone(&remote),
            Some(1024),
            clock,
            LabelMetrics::default(),
        );

        source.get(&[tweet_id]).await;
        let first_expiry = l1_expiry(&source, tweet_id).expect("backfilled");

        mock.increment(TTL);
        source.get(&[tweet_id]).await;
        let restamped = l1_expiry(&source, tweet_id).expect("re-backfilled");

        assert_eq!(remote.fetched_keys(), 2);
        assert!(restamped > first_expiry);
    }

    fn label_ids(map: &vf_pb::SafetyLabelMap) -> Vec<i32> {
        let mut ids: Vec<i32> = map.labels.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    #[tokio::test]
    async fn a_cached_label_drops_out_after_it_expires_without_a_refetch() {
        let now = Arc::new(AtomicI64::new(1_000));
        let now_ms = Arc::clone(&now);
        let remote = FakeRemote::new(HashMap::from([(
            42,
            Ok(vf_pb::SafetyLabelMap {
                labels: HashMap::from([
                    (1, proto_label(None)),
                    (2, proto_label(Some(1_000))),
                    (3, proto_label(Some(5_000))),
                ]),
            }),
        )]));
        let source = SafetyLabelSource::with_clocks(
            Arc::<FakeRemote>::clone(&remote),
            Some(8),
            Clock::new(),
            Arc::new(move || now_ms.load(Ordering::Relaxed)),
            LabelMetrics::default(),
        );

        let first = source.get(&[42]).await;
        assert_eq!(
            label_ids(first.get(&42).unwrap().as_ref().unwrap()),
            vec![1, 3]
        );
        assert_eq!(remote.fetched_keys(), 1);

        now.store(5_000, Ordering::Relaxed);
        let later = source.get(&[42]).await;
        assert_eq!(
            label_ids(later.get(&42).unwrap().as_ref().unwrap()),
            vec![1]
        );
        assert_eq!(remote.fetched_keys(), 1);

        now.store(0, Ordering::Relaxed);
        let rewound = source.get(&[42]).await;
        assert_eq!(
            label_ids(rewound.get(&42).unwrap().as_ref().unwrap()),
            vec![1, 2, 3]
        );
        assert_eq!(remote.fetched_keys(), 1);
    }
}
