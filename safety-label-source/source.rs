use enum_map::EnumMap;
use quanta::Clock;
use quick_cache::sync::Cache;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::Instant;
use tracing::info;
use xai_visibility_filtering_proto as vf_pb;

use super::dragonfly::LabelCacheSwitches;
use super::metrics::{self, BatchStage, CacheKeyCounts, CacheTier, LabelMetrics, OutageResult};
use super::remote::{FailureKind, LookupError, LookupResults, RemoteLabels, StaleEntries};

const TTL: Duration = Duration::from_secs(10);
const REPLY_ALLOWANCE: Duration = Duration::from_millis(5);

pub(crate) fn tweet_timestamp_ms(tweet_id: u64) -> u64 {
    const SNOWFLAKE_EPOCH_MS: u64 = 1288834974657;
    const SNOWFLAKE_TIMESTAMP_SHIFT: u32 = 22;
    (tweet_id >> SNOWFLAKE_TIMESTAMP_SHIFT) + SNOWFLAKE_EPOCH_MS
}

fn is_future_dated(tweet_id: u64, now: SystemTime) -> bool {
    UNIX_EPOCH
        .checked_add(Duration::from_millis(tweet_timestamp_ms(tweet_id)))
        .is_none_or(|created_at| created_at > now)
}

pub(crate) fn system_unix_ms() -> i64 {
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

pub(crate) fn drop_expired_labels(
    map: &vf_pb::SafetyLabelMap,
    now_ms: i64,
) -> vf_pb::SafetyLabelMap {
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

pub struct LabelResults {
        pub labels: FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
        pub stale: Vec<(u64, Arc<vf_pb::SafetyLabelMap>)>,
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
        let now_ms = Arc::new(system_unix_ms);
        Self::with_clocks(remote, cache_capacity, Clock::new(), now_ms, metrics)
    }

        pub(crate) fn with_clocks(
        remote: Arc<dyn RemoteLabels>,
        cache_capacity: Option<usize>,
        clock: Clock,
        now_ms: Arc<dyn Fn() -> i64 + Send + Sync>,
        metrics: LabelMetrics,
    ) -> Self {
        info!(
            cache_capacity = cache_capacity.unwrap_or(0),
            ttl_secs = TTL.as_secs(),
            "Hot-key cache initialized (single pool, per-entry TTL)"
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
        deadline: Option<Instant>,
    ) -> FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>> {
        let LabelResults { mut labels, stale } = self.get_marking_stale(ids, deadline).await;
        labels.extend(
            stale
                .into_iter()
                .map(|(tweet_id, stale)| (tweet_id, Ok(stale))),
        );
        labels
    }

        pub async fn get_marking_stale(&self, ids: &[u64], deadline: Option<Instant>) -> LabelResults {
        let total = ids.len();
        let mut results: FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>> =
            FxHashMap::with_capacity_and_hasher(total, Default::default());

        let (local_misses, stale) = self.get_local(ids, &mut results);
        let (batch_size, expired) = (local_misses.len(), stale.len());

        let answer_by =
            deadline.map(|deadline| deadline.checked_sub(REPLY_ALLOWANCE).unwrap_or(deadline));
        let remote_results = self.remote.get(&local_misses, &stale, answer_by).await;
        let failed = self.backfill_local(remote_results, &mut results);
        let mut stale_answers = if failed.is_empty() {
            Vec::new()
        } else {
            self.serve_stale(failed, stale)
        };

        self.emit_stats(total - batch_size, batch_size, expired);

        let now_ms = (self.now_ms)();
        let mut dropped = 0;
        let answered = results
            .values_mut()
            .filter_map(|result| result.as_mut().ok());
        for map in answered.chain(stale_answers.iter_mut().map(|(_, labels)| labels)) {
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

        LabelResults {
            labels: results,
            stale: stale_answers,
        }
    }

    pub(crate) fn hold_expired(&self, tweet_id: u64, labels: Arc<vf_pb::SafetyLabelMap>) {
        if let Some(cache) = &self.cache {
            let expires_at = self.clock.now();
            cache.insert(tweet_id, Entry { labels, expires_at });
        }
    }

        pub fn set_label_cache_switches(&self, switches: LabelCacheSwitches) {
        self.remote.set_switches(switches);
    }

    fn serve_stale(
        &self,
        failed: Vec<(u64, FailureKind)>,
        mut stale: StaleEntries,
    ) -> Vec<(u64, Arc<vf_pb::SafetyLabelMap>)> {
        let mut answers = Vec::new();
        let mut counts = EnumMap::<FailureKind, (usize, usize)>::default();
        for (tweet_id, kind) in failed {
            let labels = self
                .cache
                .as_ref()
                .and_then(|cache| cache.peek(&tweet_id))
                .map(|entry| entry.labels)
                .or_else(|| stale.remove(&tweet_id));
            let (answered, unanswered) = &mut counts[kind];
            match labels {
                Some(labels) => {
                    answers.push((tweet_id, labels));
                    *answered += 1;
                }
                None => *unanswered += 1,
            }
        }
        for (kind, (answered, unanswered)) in counts {
            metrics::record_outage_keys(&self.metrics, OutageResult::Stale, kind, answered);
            metrics::record_outage_keys(&self.metrics, OutageResult::Failed, kind, unanswered);
        }
        answers
    }

    fn get_local(
        &self,
        ids: &[u64],
        results: &mut FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
    ) -> (Vec<u64>, StaleEntries) {
        let Some(cache) = &self.cache else {
            return (ids.to_vec(), StaleEntries::default());
        };
        let mut misses = Vec::with_capacity(ids.len());
        let mut stale = StaleEntries::default();
        let now = self.clock.now();
        for &tweet_id in ids {
            match cache.get(&tweet_id) {
                Some(entry) if now < entry.expires_at => {
                    results.insert(tweet_id, Ok(entry.labels));
                }
                Some(entry) => {
                    stale.insert(tweet_id, entry.labels);
                    misses.push(tweet_id);
                }
                None => misses.push(tweet_id),
            }
        }
        (misses, stale)
    }

    fn backfill_local(
        &self,
        remote_results: LookupResults,
        results: &mut FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
    ) -> Vec<(u64, FailureKind)> {
        let wall_now = SystemTime::now();
        let mut expires_at = None;
        let mut failed = Vec::new();
        for (id, result) in remote_results {
            if let Err(failure) = &result {
                failed.push((id, failure.kind));
            }
            let result = result.map(Arc::new);
            if let Some(cache) = &self.cache
                && let Ok(label_map) = &result
                && !is_future_dated(id, wall_now)
            {
                cache.insert(
                    id,
                    Entry {
                        labels: Arc::clone(label_map),
                        expires_at: *expires_at.get_or_insert_with(|| self.clock.now() + TTL),
                    },
                );
            }
            results.insert(id, result);
        }
        failed
    }

    fn emit_stats(&self, hits: usize, batch_size: usize, expired: usize) {
        let counts = CacheKeyCounts {
            hit: hits,
            miss: batch_size - expired,
            expired,
            ..CacheKeyCounts::default()
        };
        metrics::record_cache_keys(&self.metrics, CacheTier::Local, &counts);
        metrics::record_batch_size(&self.metrics, BatchStage::LocalMiss, batch_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::remote::FailureKind;
    use crate::testing::RecordedStats;
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
        async fn get(
            &self,
            ids: &[u64],
            _stale: &StaleEntries,
            _answer_by: Option<Instant>,
        ) -> LookupResults {
            self.fetched_keys.fetch_add(ids.len(), Ordering::SeqCst);
            ids.iter()
                .filter_map(|id| self.results.get(id).map(|result| (*id, result.clone())))
                .collect()
        }

        fn set_switches(&self, _switches: LabelCacheSwitches) {}
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

        let results1 = source.get(&[42], None).await;
        let first = Arc::clone(results1.get(&42).unwrap().as_ref().unwrap());
        assert_eq!(remote.fetched_keys(), 1);
        assert!(l1_expiry(&source, 42).is_some());

        let results2 = source.get(&[42], None).await;
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

        let results1 = source.get(&[42], None).await;
        assert!(results1.get(&42).unwrap().is_err());
        assert!(l1_expiry(&source, 42).is_none());

        let results2 = source.get(&[42], None).await;
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
            assert!(source.get(&[42], None).await.get(&42).unwrap().is_ok());
        }
        assert_eq!(remote.fetched_keys(), 2);
    }

    #[tokio::test]
    async fn get_does_not_backfill_future_tweet_id_into_l1() {
        let tweet_id = tweet_id_created_at(SystemTime::now() + Duration::from_secs(60));
        let source = SafetyLabelSource::new(found(tweet_id), Some(1024), LabelMetrics::default());

        let results = source.get(&[tweet_id], None).await;

        assert!(results.get(&tweet_id).unwrap().is_ok());
        assert!(l1_expiry(&source, tweet_id).is_none());
    }

    #[tokio::test]
    async fn entry_at_ttl_is_refetched_and_restamped() {
        let (clock, mock) = Clock::mock();
        let tweet_id = tweet_id_created_at(SystemTime::now() - Duration::from_secs(60));
        let remote = found(tweet_id);
        let source = SafetyLabelSource::with_clocks(
            Arc::<FakeRemote>::clone(&remote),
            Some(1024),
            clock,
            Arc::new(system_unix_ms),
            LabelMetrics::default(),
        );

        source.get(&[tweet_id], None).await;
        let first_expiry = l1_expiry(&source, tweet_id).expect("backfilled");

        mock.increment(TTL);
        source.get(&[tweet_id], None).await;
        let restamped = l1_expiry(&source, tweet_id).expect("re-backfilled");

        assert_eq!(remote.fetched_keys(), 2);
        assert!(restamped > first_expiry);
    }

    #[tokio::test]
    async fn an_expired_entry_evicted_by_the_batchs_backfill_still_answers_the_outage() {
        let (clock, _mock) = Clock::mock();
        let remote = FakeRemote::new(
            (2..66)
                .map(|id| (id, Ok(vf_pb::SafetyLabelMap::default())))
                .chain([(
                    1,
                    Err(LookupError::new(FailureKind::LabelCacheUnavailable, "down")),
                )])
                .collect(),
        );
        let source = SafetyLabelSource::with_clocks(
            remote,
            Some(1),
            clock,
            Arc::new(system_unix_ms),
            LabelMetrics::default(),
        );
        let stale = Arc::new(vf_pb::SafetyLabelMap {
            labels: HashMap::from([(1, proto_label(None))]),
        });
        source.cache.as_ref().unwrap().insert(
            1,
            Entry {
                labels: Arc::clone(&stale),
                expires_at: source.clock.now(),
            },
        );

        let ids: Vec<u64> = (1..66).collect();
        let results = source.get(&ids, None).await;

        assert_eq!(results.get(&1).unwrap().as_deref().ok(), Some(&*stale));
    }

    #[tokio::test]
    async fn a_failed_manhattan_read_answers_from_the_expired_entry_and_fails_without_one() {
        let (clock, _mock) = Clock::mock();
        let failed = || Err(LookupError::new(FailureKind::ManhattanFetch, "down"));
        let stats = Arc::new(RecordedStats::default());
        let source = SafetyLabelSource::with_clocks(
            FakeRemote::new(HashMap::from([(1, failed()), (2, failed())])),
            Some(8),
            clock,
            Arc::new(system_unix_ms),
            LabelMetrics::new(Some(Arc::<RecordedStats>::clone(&stats) as _)),
        );
        let stale = Arc::new(vf_pb::SafetyLabelMap {
            labels: HashMap::from([(1, proto_label(None))]),
        });
        source.hold_expired(1, Arc::clone(&stale));

        let results = source.get(&[1, 2], None).await;

        assert_eq!(results.get(&1).unwrap().as_deref().ok(), Some(&*stale));
        assert!(matches!(
            results.get(&2),
            Some(Err(failure)) if failure.kind == FailureKind::ManhattanFetch
        ));
        let outage = |result| {
            stats.count(
                "safety_labels_outage_keys",
                &[("result", result), ("kind", "manhattan_fetch")],
            )
        };
        assert_eq!((outage("stale"), outage("failed")), (1, 1));
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

        let first = source.get(&[42], None).await;
        assert_eq!(
            label_ids(first.get(&42).unwrap().as_ref().unwrap()),
            vec![1, 3]
        );
        assert_eq!(remote.fetched_keys(), 1);

        now.store(5_000, Ordering::Relaxed);
        let later = source.get(&[42], None).await;
        assert_eq!(
            label_ids(later.get(&42).unwrap().as_ref().unwrap()),
            vec![1]
        );
        assert_eq!(remote.fetched_keys(), 1);

        now.store(0, Ordering::Relaxed);
        let rewound = source.get(&[42], None).await;
        assert_eq!(
            label_ids(rewound.get(&42).unwrap().as_ref().unwrap()),
            vec![1, 2, 3]
        );
        assert_eq!(remote.fetched_keys(), 1);
    }
}
