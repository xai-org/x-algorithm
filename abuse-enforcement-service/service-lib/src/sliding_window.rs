use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::task::JoinHandle;
use tracing::{info, warn};
use xai_manhattan::{Key, LkeySelector, NativeManhattanClient, Tenant};

use crate::limiter::LimiterClient;
use crate::manhattan::ttl_nanos_from_secs;
use crate::metrics;

pub const USER_SNAP_PKEY: &str = "global_rate_limit";
pub const POST_SNAP_PKEY: &str = "global_rate_limit:post";
const WINDOW_SECS: u64 = 86_400;
const SNAPSHOT_INTERVAL_SECS: u64 = 60;
const SCAN_LIMIT: u32 = 200_000;

#[derive(Clone, Copy, Debug)]
pub struct WindowSpec {
    pub windows_secs: &'static [u64],
    pub snapshot_interval_secs: u64,
}

impl WindowSpec {
    fn longest_secs(&self) -> u64 {
        self.windows_secs.iter().copied().max().unwrap_or(0)
    }

    fn snapshot_ttl_secs(&self) -> u64 {
        self.longest_secs() + 3_600
    }
}

pub const DAILY_CAP: WindowSpec = WindowSpec {
    windows_secs: &[WINDOW_SECS],
    snapshot_interval_secs: SNAPSHOT_INTERVAL_SECS,
};

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn snapshot_lkey(ts_secs: u64, interval: u64) -> String {
    format!("{:020}", (ts_secs / interval) * interval)
}

fn window_usage(counts: &[i64]) -> Option<(i64, i64)> {
    let first = *counts.first()?;
    let baseline = counts.iter().copied().max().unwrap_or(first);
    Some((baseline - first, baseline))
}

fn remaining(cap: i64, used: i64) -> i64 {
    (cap - used).max(0)
}

fn window_cache(
    rows: &[(String, i64)],
    now: u64,
    window: u64,
    floor: u64,
    interval: u64,
    last_count: i64,
) -> Cached {
    let start = snapshot_lkey(floor.max(now.saturating_sub(window)), interval);
    let count = |(_, c): &(String, i64)| *c;
    let mut counts: Vec<i64> = Vec::new();
    counts.extend(rows.iter().rfind(|(l, _)| *l <= start).map(count));
    counts.extend(rows.iter().filter(|(l, _)| *l > start).map(count));
    match window_usage(&counts) {
        Some((rolling_tail, baseline)) => Cached {
            rolling_tail,
            baseline,
            present: true,
        },
        None => Cached {
            rolling_tail: 0,
            baseline: last_count,
            present: false,
        },
    }
}

#[async_trait]
trait CounterSource: Send + Sync {
    async fn increment(&self) -> Option<i64>;
}

#[async_trait]
impl CounterSource for LimiterClient {
    async fn increment(&self) -> Option<i64> {
        self.increment_count().await
    }
}

#[async_trait]
trait SnapshotStore: Send + Sync {
    async fn scan(
        &self,
        pkey: &str,
        from: Option<&str>,
        to: Option<&str>,
        limit: u32,
    ) -> anyhow::Result<Vec<(String, String)>>;
    async fn put_with_ttl(
        &self,
        pkey: &str,
        lkey: &str,
        value: String,
        ttl_nanos: i64,
    ) -> anyhow::Result<()>;
    async fn delete(&self, pkey: &str, lkey: &str) -> anyhow::Result<()>;
}

struct ManhattanSnapshotStore {
    mh: Arc<NativeManhattanClient>,
    tenant: Tenant,
}

#[async_trait]
impl SnapshotStore for ManhattanSnapshotStore {
    async fn scan(
        &self,
        pkey: &str,
        from: Option<&str>,
        to: Option<&str>,
        limit: u32,
    ) -> anyhow::Result<Vec<(String, String)>> {
        let items = self
            .mh
            .scan(
                self.tenant.clone(),
                [pkey],
                LkeySelector::Range {
                    from: from.map(|f| Key::from_parts([f])),
                    to: to.map(|t| Key::from_parts([t])),
                },
                limit,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(items
            .iter()
            .filter_map(|it| {
                let lkey =
                    String::from_utf8(it.lkey().into_byte_vecs().into_iter().next()?).ok()?;
                let value = String::from_utf8(it.value().into_bytes()).ok()?;
                Some((lkey, value))
            })
            .collect())
    }

    async fn put_with_ttl(
        &self,
        pkey: &str,
        lkey: &str,
        value: String,
        ttl_nanos: i64,
    ) -> anyhow::Result<()> {
        self.mh
            .put_with_ttl(self.tenant.clone(), [pkey], [lkey], value, ttl_nanos)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    async fn delete(&self, pkey: &str, lkey: &str) -> anyhow::Result<()> {
        self.mh
            .delete(self.tenant.clone(), [pkey], [lkey])
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

#[derive(Clone, Copy, Default)]
struct Cached {
    rolling_tail: i64,
    baseline: i64,
    present: bool,
}

impl Cached {
    fn rolling(&self, c_now: i64) -> i64 {
        self.rolling_tail + (c_now - self.baseline).max(0)
    }
}

#[derive(Clone)]
pub struct SlidingWindowLimiter {
    counter: Arc<dyn CounterSource>,
    store: Arc<dyn SnapshotStore>,
    snap_pkey: String,
    metric_label: &'static str,
    spec: WindowSpec,
    last_count: Arc<AtomicI64>,
    cached: Arc<RwLock<Vec<Cached>>>,
    seeded: Arc<AtomicBool>,
    dirty: Arc<AtomicBool>,
    floor_ms: Arc<AtomicU64>,
    fail_open: bool,
}

impl SlidingWindowLimiter {
    pub fn new(
        limiter: LimiterClient,
        mh: Arc<NativeManhattanClient>,
        tenant: Tenant,
        fail_open: bool,
        snap_pkey: impl Into<String>,
        metric_label: &'static str,
        spec: WindowSpec,
    ) -> Self {
        Self::from_parts(
            Arc::new(limiter),
            Arc::new(ManhattanSnapshotStore { mh, tenant }),
            fail_open,
            snap_pkey,
            metric_label,
            spec,
        )
    }

    fn from_parts(
        counter: Arc<dyn CounterSource>,
        store: Arc<dyn SnapshotStore>,
        fail_open: bool,
        snap_pkey: impl Into<String>,
        metric_label: &'static str,
        spec: WindowSpec,
    ) -> Self {
        Self {
            counter,
            store,
            snap_pkey: snap_pkey.into(),
            metric_label,
            spec,
            last_count: Arc::new(AtomicI64::new(0)),
            cached: Arc::new(RwLock::new(vec![
                Cached::default();
                spec.windows_secs.len()
            ])),
            seeded: Arc::new(AtomicBool::new(false)),
            dirty: Arc::new(AtomicBool::new(false)),
            floor_ms: Arc::new(AtomicU64::new(0)),
            fail_open,
        }
    }

    pub async fn try_record(&self, cap: i64) -> bool {
        match self.increment_and_rolling().await {
            Some(rolling) => {
                let rolling = rolling[0];
                metrics::RATE_LIMIT_REMAINING
                    .with_label_values(&[self.metric_label])
                    .set(remaining(cap, rolling));
                if !self.seeded.load(Ordering::Relaxed) {
                    return true;
                }
                rolling < cap
            }
            None => self.fail_open,
        }
    }

    #[cfg(test)]
    pub(crate) fn seed_for_test(&self) {
        self.seeded.store(true, Ordering::Relaxed);
    }

    pub fn stats(&self, cap: i64) -> (u32, u32, u32) {
        let used = self.rolling_cached()[0];
        (
            used.max(0) as u32,
            cap.max(0) as u32,
            remaining(cap, used) as u32,
        )
    }

    pub async fn record_report(&self, cap: i64) -> Option<serde_json::Value> {
        let rolling = self.increment_and_rolling().await?[0];
        metrics::RATE_LIMIT_REMAINING
            .with_label_values(&[self.metric_label])
            .set(remaining(cap, rolling));
        let allowed = !self.seeded.load(Ordering::Relaxed) || rolling < cap;
        Some(serde_json::json!({
            "allowed": allowed,
            "used": rolling,
            "cap": cap,
            "remaining": remaining(cap, rolling),
        }))
    }

    pub async fn reset(&self) -> Option<u64> {
        let items = match self
            .store
            .scan(self.snap_pkey.as_str(), None, None, SCAN_LIMIT)
            .await
        {
            Ok(items) => items,
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                warn!(pkey = %self.snap_pkey, "sliding-window reset scan failed: {e}");
                return None;
            }
        };

        let mut deleted = 0u64;
        for (lkey, _) in &items {
            if let Err(e) = self.store.delete(self.snap_pkey.as_str(), lkey).await {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                warn!(pkey = %self.snap_pkey, "sliding-window reset delete failed for lkey={lkey}: {e}");
            } else {
                deleted += 1;
            }
        }

        let baseline = self.last_count.load(Ordering::Relaxed);
        for c in self
            .cached
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .iter_mut()
        {
            c.rolling_tail = 0;
            c.baseline = baseline;
        }
        self.seeded.store(true, Ordering::Relaxed);
        warn!(pkey = %self.snap_pkey, deleted, baseline, "sliding-window reset");
        Some(deleted)
    }

    pub fn spawn_background(&self) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            this.refresh().await;
            let mut tick =
                tokio::time::interval(Duration::from_secs(this.spec.snapshot_interval_secs));
            loop {
                tick.tick().await;
                this.write_snapshot().await;
                this.refresh().await;
            }
        })
    }

    pub fn set_floor(&self, floor_ms: u64) -> bool {
        if self.floor_ms.fetch_max(floor_ms, Ordering::Relaxed) >= floor_ms {
            return false;
        }
        let last = self.last_count.load(Ordering::Relaxed);
        for c in self
            .cached
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .iter_mut()
        {
            *c = Cached {
                rolling_tail: 0,
                baseline: last,
                present: false,
            };
        }
        true
    }

    pub fn floor_ms(&self) -> u64 {
        self.floor_ms.load(Ordering::Relaxed)
    }

    pub async fn increment_and_rolling(&self) -> Option<Vec<i64>> {
        let c_now = self.counter.increment().await?;
        let prev = self.last_count.swap(c_now, Ordering::Relaxed);
        self.dirty.store(true, Ordering::Relaxed);
        if c_now < prev {
            metrics::LIMITER_COUNTER_RESET_TOTAL
                .with_label_values(&[self.metric_label])
                .inc();
        }
        {
            let cached = self.cached.read().unwrap_or_else(|e| e.into_inner());
            if cached.iter().all(|c| c.present) {
                return Some(cached.iter().map(|c| c.rolling(c_now)).collect());
            }
        }
        let mut cached = self.cached.write().unwrap_or_else(|e| e.into_inner());
        for c in cached.iter_mut().filter(|c| !c.present) {
            c.baseline = c_now;
        }
        Some(cached.iter().map(|c| c.rolling(c_now)).collect())
    }

    pub fn rolling_cached(&self) -> Vec<i64> {
        let last = self.last_count.load(Ordering::Relaxed);
        self.cached
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|c| c.rolling(last))
            .collect()
    }

    pub async fn write_snapshot(&self) {
        if !self.dirty.load(Ordering::Relaxed) {
            return;
        }
        let c = self.last_count.load(Ordering::Relaxed);
        let lkey = snapshot_lkey(now_secs(), self.spec.snapshot_interval_secs);
        let ttl = ttl_nanos_from_secs(self.spec.snapshot_ttl_secs());
        if let Err(e) = self
            .store
            .put_with_ttl(self.snap_pkey.as_str(), lkey.as_str(), c.to_string(), ttl)
            .await
        {
            metrics::MANHATTAN_ERRORS_TOTAL.inc();
            warn!(pkey = %self.snap_pkey, "sliding-window snapshot write failed: {e}");
            return;
        }
        self.dirty.store(false, Ordering::Relaxed);
        if self.last_count.load(Ordering::Relaxed) != c {
            self.dirty.store(true, Ordering::Relaxed);
        }
    }

    pub async fn refresh(&self) {
        let now = now_secs();
        let interval = self.spec.snapshot_interval_secs;
        let floor_ms = self.floor_ms.load(Ordering::Relaxed);
        let from = snapshot_lkey(now.saturating_sub(self.spec.longest_secs()), interval);
        let to = snapshot_lkey(now, interval);
        let items = match self
            .store
            .scan(
                self.snap_pkey.as_str(),
                Some(from.as_str()),
                Some(to.as_str()),
                SCAN_LIMIT,
            )
            .await
        {
            Ok(items) => items,
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                warn!(pkey = %self.snap_pkey, "sliding-window scan failed: {e}");
                return;
            }
        };
        let rows: Vec<(String, i64)> = items
            .into_iter()
            .filter_map(|(l, v)| Some((l, v.trim().parse().ok()?)))
            .collect();
        let last = self.last_count.load(Ordering::Relaxed);
        let persisted = !self.dirty.load(Ordering::Relaxed);
        if rows.is_empty() && !self.seeded.load(Ordering::Relaxed) {
            info!(
                pkey = %self.snap_pkey,
                "sliding-window: cold start (no snapshots), baselining at C={last}"
            );
        }
        let fresh: Vec<Cached> = self
            .spec
            .windows_secs
            .iter()
            .map(|&w| {
                let mut c = window_cache(&rows, now, w, floor_ms / 1_000, interval, last);
                if persisted {
                    c.baseline = c.baseline.max(last);
                }
                c
            })
            .collect();
        let first = fresh[0];
        {
            let mut cached = self.cached.write().unwrap_or_else(|e| e.into_inner());
            if self.floor_ms.load(Ordering::Relaxed) != floor_ms {
                return;
            }
            *cached = fresh;
        }
        self.seeded.store(true, Ordering::Relaxed);
        warn!(
            pkey = %self.snap_pkey,
            from = %from,
            to = %to,
            snapshots = rows.len(),
            rolling_tail = first.rolling_tail,
            baseline = first.baseline,
            "sliding-window refresh"
        );
    }
}

#[cfg(test)]
pub(crate) fn counting_limiter_for_test() -> (SlidingWindowLimiter, Arc<AtomicI64>) {
    struct Counting(Arc<AtomicI64>);
    #[async_trait]
    impl CounterSource for Counting {
        async fn increment(&self) -> Option<i64> {
            Some(self.0.fetch_add(1, Ordering::Relaxed) + 1)
        }
    }
    struct NoStore;
    #[async_trait]
    impl SnapshotStore for NoStore {
        async fn scan(
            &self,
            _pkey: &str,
            _from: Option<&str>,
            _to: Option<&str>,
            _limit: u32,
        ) -> anyhow::Result<Vec<(String, String)>> {
            Ok(Vec::new())
        }
        async fn put_with_ttl(
            &self,
            _pkey: &str,
            _lkey: &str,
            _value: String,
            _ttl_nanos: i64,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn delete(&self, _pkey: &str, _lkey: &str) -> anyhow::Result<()> {
            Ok(())
        }
    }
    let tickets = Arc::new(AtomicI64::new(0));
    let limiter = SlidingWindowLimiter::from_parts(
        Arc::new(Counting(tickets.clone())),
        Arc::new(NoStore),
        true,
        USER_SNAP_PKEY,
        "user",
        DAILY_CAP,
    );
    (limiter, tickets)
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    pub(crate) struct MemCounter {
        pub count: AtomicI64,
        pub fail: AtomicBool,
    }

    #[async_trait]
    impl CounterSource for MemCounter {
        async fn increment(&self) -> Option<i64> {
            if self.fail.load(Ordering::Relaxed) {
                return None;
            }
            Some(self.count.fetch_add(1, Ordering::SeqCst) + 1)
        }
    }

    #[derive(Default)]
    pub(crate) struct MemSnapshots {
        pub rows: Mutex<BTreeMap<(String, String), String>>,
        pub hang_scan: AtomicBool,
        pub hang_put: AtomicBool,
    }

    impl MemSnapshots {
        pub(crate) fn put_at(&self, pkey: &str, ts_secs: u64, value: i64) {
            self.rows.lock().unwrap().insert(
                (
                    pkey.to_owned(),
                    snapshot_lkey(ts_secs, SNAPSHOT_INTERVAL_SECS),
                ),
                value.to_string(),
            );
        }
    }

    #[async_trait]
    impl SnapshotStore for MemSnapshots {
        async fn scan(
            &self,
            pkey: &str,
            from: Option<&str>,
            to: Option<&str>,
            _limit: u32,
        ) -> anyhow::Result<Vec<(String, String)>> {
            if self.hang_scan.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            Ok(self
                .rows
                .lock()
                .unwrap()
                .iter()
                .filter(|((p, l), _)| {
                    p == pkey
                        && from.is_none_or(|f| l.as_str() >= f)
                        && to.is_none_or(|t| l.as_str() <= t)
                })
                .map(|((_, l), v)| (l.clone(), v.clone()))
                .collect())
        }

        async fn put_with_ttl(
            &self,
            pkey: &str,
            lkey: &str,
            value: String,
            _ttl_nanos: i64,
        ) -> anyhow::Result<()> {
            if self.hang_put.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            self.rows
                .lock()
                .unwrap()
                .insert((pkey.to_owned(), lkey.to_owned()), value);
            Ok(())
        }

        async fn delete(&self, pkey: &str, lkey: &str) -> anyhow::Result<()> {
            self.rows
                .lock()
                .unwrap()
                .remove(&(pkey.to_owned(), lkey.to_owned()));
            Ok(())
        }
    }

    pub(crate) fn limiter(
        counter: Arc<MemCounter>,
        store: Arc<MemSnapshots>,
        snap_pkey: &str,
        metric_label: &'static str,
        spec: WindowSpec,
    ) -> SlidingWindowLimiter {
        SlidingWindowLimiter::from_parts(counter, store, true, snap_pkey, metric_label, spec)
    }

    pub(crate) fn now() -> u64 {
        now_secs()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use super::test_support::MemSnapshots as MemStore;
    use super::*;

    #[test]
    fn window_usage_monotone_equals_last_minus_first() {
        assert_eq!(window_usage(&[10, 13, 20, 100]), Some((90, 100)));
    }

    #[test]
    fn window_usage_dip_and_recover_adds_no_phantom_usage() {
        let incident = [4_650_000, 4_800_000, 5_000_000, 12, 5_000_000, 40];
        assert_eq!(window_usage(&incident), Some((350_000, 5_000_000)));
    }

    #[test]
    fn window_usage_true_reset_undercounts_never_overcounts() {
        assert_eq!(window_usage(&[10, 100, 5, 30]), Some((90, 100)));
    }

    #[test]
    fn window_usage_oscillation_bounded_by_range() {
        assert_eq!(window_usage(&[50, 10, 60, 20, 55]), Some((10, 60)));
    }

    #[test]
    fn window_usage_matches_old_math_on_monotone_series() {
        let series: &[&[i64]] = &[
            &[0, 1, 2, 3],
            &[42],
            &[10, 13, 20, 100],
            &[4_650_000, 4_800_000, 5_000_000],
            &[7, 7, 7],
        ];
        for counts in series {
            let old_tail: i64 = counts.windows(2).map(|w| (w[1] - w[0]).max(0)).sum();
            let old_baseline = *counts.last().unwrap();
            assert_eq!(
                window_usage(counts),
                Some((old_tail, old_baseline)),
                "diverged from old math on healthy series {counts:?}"
            );
        }
    }

    #[test]
    fn window_usage_empty_and_single() {
        assert_eq!(window_usage(&[]), None);
        assert_eq!(window_usage(&[42]), Some((0, 42)));
    }

    #[test]
    fn remaining_clamps_at_zero() {
        assert_eq!(remaining(500_000, 100), 499_900);
        assert_eq!(remaining(500_000, 500_000), 0);
        assert_eq!(remaining(500_000, 600_000), 0);
    }

    #[test]
    fn snapshot_lkey_aligns_to_interval() {
        assert_eq!(snapshot_lkey(125, 60), snapshot_lkey(179, 60));
        assert_ne!(snapshot_lkey(125, 60), snapshot_lkey(185, 60));
    }

    #[test]
    fn snapshot_lkey_is_fixed_width_and_lexically_ordered() {
        let a = snapshot_lkey(120, 60);
        let b = snapshot_lkey(1_000_000, 60);
        assert_eq!(a.len(), 20);
        assert_eq!(b.len(), 20);
        assert!(a < b);
    }

    struct ScriptedCounter {
        script: Mutex<VecDeque<i64>>,
    }

    impl ScriptedCounter {
        fn new(values: impl IntoIterator<Item = i64>) -> Arc<Self> {
            Arc::new(Self {
                script: Mutex::new(values.into_iter().collect()),
            })
        }
    }

    #[async_trait]
    impl CounterSource for ScriptedCounter {
        async fn increment(&self) -> Option<i64> {
            self.script.lock().unwrap().pop_front()
        }
    }

    const CAP: i64 = 2_000_000;
    const PKEY: &str = "global_rate_limit:post";

    fn pod(counter: Arc<ScriptedCounter>, store: Arc<MemStore>) -> SlidingWindowLimiter {
        SlidingWindowLimiter::from_parts(counter, store, true, PKEY, "post", DAILY_CAP)
    }

    async fn seed(store: &MemStore, age_secs: u64, value: i64) {
        let lkey = snapshot_lkey(now_secs().saturating_sub(age_secs), SNAPSHOT_INTERVAL_SECS);
        store
            .put_with_ttl(PKEY, &lkey, value.to_string(), 0)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn incident_replay_poisoned_series_does_not_wedge_the_cap() {
        let store = Arc::new(MemStore::default());
        seed(&store, 300, 4_650_000).await;
        seed(&store, 240, 4_800_000).await;
        seed(&store, 180, 5_000_000).await;
        seed(&store, 120, 12).await;
        seed(&store, 60, 5_000_000).await;

        let pod_a = pod(ScriptedCounter::new([]), store.clone());
        pod_a.refresh().await;

        let (used, cap, rem) = pod_a.stats(CAP);
        assert_eq!(used, 350_000, "phantom usage from the dip-recover");
        assert_eq!(cap, 2_000_000);
        assert_eq!(rem, 1_650_000, "remaining must not cliff to 0");
    }

    #[tokio::test]
    async fn post_reset_observation_keeps_allowing_and_counts_reset() {
        let store = Arc::new(MemStore::default());
        seed(&store, 240, 4_650_000).await;
        seed(&store, 120, 5_000_000).await;

        let pod_a = pod(ScriptedCounter::new([5_000_010, 12]), store.clone());
        pod_a.refresh().await;

        assert!(pod_a.try_record(CAP).await, "pre-reset: well under cap");
        let before = metrics::LIMITER_COUNTER_RESET_TOTAL
            .with_label_values(&["post"])
            .get();
        assert!(
            pod_a.try_record(CAP).await,
            "post-reset observation must not deny"
        );
        let after = metrics::LIMITER_COUNTER_RESET_TOTAL
            .with_label_values(&["post"])
            .get();
        assert_eq!(after - before, 1, "backwards step must tick the metric");

        let (used, _, rem) = pod_a.stats(CAP);
        assert_eq!(used, 350_000);
        assert_eq!(rem, 1_650_000);
    }

    #[tokio::test]
    async fn stale_pod_stops_writing_snapshots() {
        let store = Arc::new(MemStore::default());
        let stale = pod(ScriptedCounter::new([5_000_000]), store.clone());

        stale.write_snapshot().await;
        assert!(
            store.scan(PKEY, None, None, 100).await.unwrap().is_empty(),
            "fresh pod must not persist its placeholder"
        );

        stale.increment_and_rolling().await.unwrap();
        stale.write_snapshot().await;
        assert_eq!(store.scan(PKEY, None, None, 100).await.unwrap().len(), 1);

        let fresh = pod(ScriptedCounter::new([12]), store.clone());
        fresh.refresh().await;
        fresh.increment_and_rolling().await.unwrap();
        fresh.write_snapshot().await;

        let rows_before = store.scan(PKEY, None, None, 100).await.unwrap();
        stale.write_snapshot().await;
        let rows_after = store.scan(PKEY, None, None, 100).await.unwrap();
        assert_eq!(
            rows_before, rows_after,
            "stale pod re-injected a snapshot without a fresh observation"
        );
    }

    #[tokio::test]
    async fn refresh_does_not_resurrect_stale_high_into_last_count() {
        let store = Arc::new(MemStore::default());
        seed(&store, 120, 5_000_000).await;

        let pod_a = pod(ScriptedCounter::new([12]), store.clone());
        pod_a.refresh().await;
        pod_a.increment_and_rolling().await.unwrap();
        pod_a.refresh().await;

        assert_eq!(
            pod_a.last_count.load(Ordering::Relaxed),
            12,
            "scanned stale high must not overwrite an own observation"
        );
        pod_a.write_snapshot().await;
        let rows = store.scan(PKEY, None, None, 100).await.unwrap();
        assert!(
            rows.iter().any(|(_, v)| v == "12"),
            "pod must persist its own fresh observation"
        );
        assert_eq!(
            rows.iter().filter(|(_, v)| v == "5000000").count(),
            1,
            "no additional stale-high rows may appear"
        );
    }

    #[tokio::test]
    async fn true_reset_counts_fresh_after_window_slides() {
        let store = Arc::new(MemStore::default());
        seed(&store, WINDOW_SECS + 600, 5_000_000).await;
        seed(&store, 180, 12).await;
        seed(&store, 60, 40).await;

        let pod_a = pod(ScriptedCounter::new([]), store.clone());
        pod_a.refresh().await;
        let (used, _, rem) = pod_a.stats(CAP);
        assert_eq!(used, 28, "post-reset ramp counts once highs age out");
        assert_eq!(rem as i64, CAP - 28);
    }

    const HOUR_AND_DAY: WindowSpec = WindowSpec {
        windows_secs: &[3_600, WINDOW_SECS],
        snapshot_interval_secs: SNAPSHOT_INTERVAL_SECS,
    };

    fn multi_window_pod(
        counter: Arc<ScriptedCounter>,
        store: Arc<MemStore>,
    ) -> SlidingWindowLimiter {
        SlidingWindowLimiter::from_parts(counter, store, true, PKEY, "post", HOUR_AND_DAY)
    }

    #[tokio::test]
    async fn each_window_counts_its_own_span_of_one_series() {
        let store = Arc::new(MemStore::default());
        seed(&store, 7_200, 10).await;
        seed(&store, 4_000, 15).await;
        seed(&store, 1_800, 20).await;
        let pod_a = multi_window_pod(ScriptedCounter::new([21]), store);
        pod_a.refresh().await;
        assert_eq!(pod_a.increment_and_rolling().await, Some(vec![6, 11]));
        assert_eq!(pod_a.stats(CAP).0, 6, "stats read the first window");

        assert!(pod_a.set_floor((now_secs() - 2 * WINDOW_SECS) * 1_000));
        pod_a.refresh().await;
        assert_eq!(
            pod_a.rolling_cached(),
            vec![6, 11],
            "a floor before every window changes nothing"
        );
    }

    #[tokio::test]
    async fn an_hour_idle_at_its_start_counts_a_burst_at_once() {
        let store = Arc::new(MemStore::default());
        seed(&store, 7_200, 10).await;
        seed(&store, 5_400, 20).await;
        let pod_a = multi_window_pod(ScriptedCounter::new([21, 22, 23]), store);
        pod_a.refresh().await;
        for n in 1..=3 {
            assert_eq!(
                pod_a.increment_and_rolling().await,
                Some(vec![n, 10 + n]),
                "the day keeps its tail"
            );
        }
    }

    #[tokio::test]
    async fn an_hour_after_a_counter_reset_counts_from_the_latest_snapshot() {
        let store = Arc::new(MemStore::default());
        seed(&store, 10_800, 100).await;
        seed(&store, 7_200, 5).await;
        let pod_a = multi_window_pod(ScriptedCounter::new([6, 7, 8]), store);
        pod_a.refresh().await;
        for n in 1..=3 {
            assert_eq!(pod_a.increment_and_rolling().await.unwrap()[0], n);
        }
    }

    #[tokio::test]
    async fn a_persisted_pre_reset_count_adds_nothing_after_a_counter_reset() {
        let store = Arc::new(MemStore::default());
        let pod_a = multi_window_pod(ScriptedCounter::new([1_000, 21]), store.clone());
        pod_a.refresh().await;
        pod_a.increment_and_rolling().await.unwrap();
        pod_a.write_snapshot().await;

        store.rows.lock().unwrap().clear();
        seed(&store, 10_800, 1_000).await;
        seed(&store, 7_200, 5).await;
        seed(&store, 1_800, 20).await;
        pod_a.refresh().await;
        assert_eq!(pod_a.rolling_cached(), vec![15, 0]);
        assert_eq!(pod_a.increment_and_rolling().await, Some(vec![15, 0]));
    }

    #[tokio::test]
    async fn a_dropped_snapshot_write_is_retried() {
        let store = Arc::new(MemStore::default());
        seed(&store, 1_800, 20).await;
        let pod_a = pod(ScriptedCounter::new([25]), store.clone());
        pod_a.refresh().await;
        assert_eq!(pod_a.increment_and_rolling().await, Some(vec![5]));

        store.hang_put.store(true, Ordering::Relaxed);
        let cut = tokio::time::timeout(Duration::from_millis(20), pod_a.write_snapshot()).await;
        assert!(cut.is_err(), "the put hangs");
        pod_a.refresh().await;
        assert_eq!(pod_a.rolling_cached(), vec![5], "not treated as persisted");

        store.hang_put.store(false, Ordering::Relaxed);
        pod_a.write_snapshot().await;
        let rows = store.scan(PKEY, None, None, 100).await.unwrap();
        assert!(rows.iter().any(|(_, v)| v == "25"), "retried: {rows:?}");
    }

    #[tokio::test]
    async fn cold_start_counts_fresh_from_the_live_count() {
        let store = Arc::new(MemStore::default());
        let pod_a = pod(
            ScriptedCounter::new([5_000_000, 5_000_001, 5_000_002]),
            store,
        );
        pod_a.refresh().await;
        assert_eq!(pod_a.increment_and_rolling().await, Some(vec![0]));
        assert_eq!(pod_a.increment_and_rolling().await, Some(vec![0]));
        pod_a.write_snapshot().await;
        pod_a.refresh().await;
        assert_eq!(pod_a.increment_and_rolling().await, Some(vec![1]));
        assert_eq!(pod_a.stats(CAP).0, 1);
    }

    #[tokio::test]
    async fn a_floor_restarts_every_window_it_cuts() {
        let store = Arc::new(MemStore::default());
        seed(&store, 7_200, 10).await;
        seed(&store, 1_800, 20).await;
        seed(&store, 600, 25).await;
        let pod_a = multi_window_pod(ScriptedCounter::new([26]), store);
        pod_a.refresh().await;
        assert_eq!(pod_a.increment_and_rolling().await, Some(vec![16, 16]));

        let floor = (now_secs() - 1_200) * 1_000;
        assert!(pod_a.set_floor(floor));
        assert_eq!(pod_a.floor_ms(), floor);
        assert_eq!(
            pod_a.rolling_cached(),
            vec![0, 0],
            "nothing before the floor counts, even before the next refresh"
        );
        pod_a.refresh().await;
        assert_eq!(pod_a.rolling_cached(), vec![6, 6]);

        assert!(
            !pod_a.set_floor(floor - 600_000),
            "a lower floor is ignored"
        );
        assert!(!pod_a.set_floor(floor), "an equal floor is ignored");
        assert!(pod_a.set_floor(floor + 1), "a floor 1 ms later restarts");
        assert_eq!(pod_a.floor_ms(), floor + 1);
    }

    #[tokio::test]
    async fn reset_clears_poisoned_series_and_rebaselines() {
        let store = Arc::new(MemStore::default());
        seed(&store, 180, 5_000_000).await;
        seed(&store, 120, 12).await;
        seed(&store, 60, 5_000_000).await;

        let pod_a = pod(ScriptedCounter::new([5_000_001]), store.clone());
        pod_a.refresh().await;
        pod_a.increment_and_rolling().await.unwrap();

        let deleted = pod_a.reset().await.expect("reset must succeed");
        assert_eq!(deleted, 3);
        assert!(store.scan(PKEY, None, None, 100).await.unwrap().is_empty());
        let (used, _, rem) = pod_a.stats(CAP);
        assert_eq!(used, 0);
        assert_eq!(rem as i64, CAP);
    }
}
