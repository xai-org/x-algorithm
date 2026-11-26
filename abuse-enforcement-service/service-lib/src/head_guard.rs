use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;
use tracing::{error, info, warn};

pub use xai_abuse_enforcement_types::head_guard::{
    HEAD_GUARD_RECORD_VERSION, HeadGuardListResponse, HeadGuardPutRequest, HeadGuardPutResponse,
    HeadGuardRecord, HeadGuardView, MAX_OVERTURN_WINDOW_HOURS, OverturnGuard,
};

use crate::action_control::{ControlStore, UnreadableRecord, new_nonce, raw_version};
use crate::growthbook::DynamicConfig;
use crate::metrics;
use crate::off_switch::{
    FIRST_LOAD_BACKOFF, FIRST_LOAD_ERROR_LOG_EVERY, LOCAL_WRITE_GRACE, MAX_NAME_LEN,
    MAX_REASON_LEN, OffSwitch, OffSwitchPutRequest, OffSwitchRecord, RECORD_TTL_SECS,
    REFRESH_ERROR_LOG_EVERY, REFRESH_INTERVAL, lkey_for, valid_name,
};
use crate::sliding_window::{SlidingWindowLimiter, WindowSpec};

pub const MH_PKEY: &str = "head_guard";

pub const COUNT_PKEY_PREFIX: &str = "head_guard_count:";

pub const UPDATED_BY: &str = "head-guard";

pub const WINDOWS: WindowSpec = WindowSpec {
    windows_secs: &[3_600, 86_400],
    snapshot_interval_secs: 60,
};

const STORE_TIMEOUT: Duration = Duration::from_secs(5);

const TRIP_RETRY_AFTER: Duration = Duration::from_secs(5);

pub const MAX_LIMIT: u32 = 10_000_000;

pub const MAX_MIN_DECIDED_APPEALS: u32 = 100_000;

pub fn bucket_id(lkey: &str) -> i64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in lkey.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    1_000 + (h % (i64::MAX as u64 - 1_000)) as i64
}

pub fn counting_floor_ms(switch: Option<&OffSwitchRecord>, now_ms: i64) -> Option<i64> {
    match switch {
        None => Some(0),
        Some(s) if !s.off => Some(s.updated_at_ms),
        Some(s) => s.auto_on_at_ms.filter(|t| *t <= now_ms),
    }
}

fn ceilings(rec: &HeadGuardRecord) -> [(&'static str, Option<u32>); 2] {
    [
        ("1h", rec.max_actions_per_hour),
        ("24h", rec.max_actions_per_day),
    ]
}

fn count_metric(head: &str, scope: &str, window: &str, result: &str) {
    metrics::HEAD_GUARD_TOTAL
        .with_label_values(&[head, scope, window, result])
        .inc();
}

fn check_fields(
    head: &str,
    classifier: Option<&str>,
    per_hour: Option<u32>,
    per_day: Option<u32>,
    overturn: Option<&OverturnGuard>,
) -> Result<(), (&'static str, String)> {
    if !valid_name(head) {
        return Err((
            "invalid_head",
            "head must be 1-256 characters of [A-Za-z0-9_.:@-]".into(),
        ));
    }
    if classifier.is_some_and(|c| !valid_name(c)) {
        return Err((
            "invalid_classifier",
            "classifier must be null or 1-256 characters of [A-Za-z0-9_.:@-]".into(),
        ));
    }
    for (name, limit) in [
        ("max_actions_per_hour", per_hour),
        ("max_actions_per_day", per_day),
    ] {
        if limit.is_some_and(|n| !(1..=MAX_LIMIT).contains(&n)) {
            return Err(("invalid_limit", format!("{name} must be 1-{MAX_LIMIT}")));
        }
    }
    if let (Some(h), Some(d)) = (per_hour, per_day)
        && h > d
    {
        return Err((
            "invalid_limit",
            "max_actions_per_hour is more than max_actions_per_day".into(),
        ));
    }
    if let Some(o) = overturn {
        if !(o.max_overturn_rate > 0.0 && o.max_overturn_rate < 1.0) {
            return Err((
                "invalid_overturn",
                "overturn.max_overturn_rate must be strictly between 0 and 1".into(),
            ));
        }
        if !(1..=MAX_MIN_DECIDED_APPEALS).contains(&o.min_decided_appeals) {
            return Err((
                "invalid_overturn",
                format!("overturn.min_decided_appeals must be 1-{MAX_MIN_DECIDED_APPEALS}"),
            ));
        }
        if !(1..=MAX_OVERTURN_WINDOW_HOURS).contains(&o.window_hours) {
            return Err((
                "invalid_overturn",
                format!("overturn.window_hours must be 1-{MAX_OVERTURN_WINDOW_HOURS}"),
            ));
        }
    }
    Ok(())
}

fn parse_record(lkey: &str, bytes: &[u8]) -> Result<HeadGuardRecord, String> {
    let rec: HeadGuardRecord = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if rec.v > HEAD_GUARD_RECORD_VERSION {
        return Err(format!("record version {} is newer than this pod", rec.v));
    }
    check_fields(
        &rec.head,
        rec.classifier.as_deref(),
        rec.max_actions_per_hour,
        rec.max_actions_per_day,
        rec.overturn.as_ref(),
    )
    .map_err(|(_, detail)| detail)?;
    let own = lkey_for(&rec.head, rec.classifier.as_deref());
    if own != lkey {
        return Err(format!("the record names {own}, not its lkey"));
    }
    Ok(rec)
}

fn labels_of_lkey(lkey: &str) -> (&str, &'static str) {
    match lkey.split('/').collect::<Vec<_>>().as_slice() {
        ["h", head] if valid_name(head) => (*head, "head"),
        ["p", head, c] if valid_name(head) && valid_name(c) => (*head, "pair"),
        _ => ("", ""),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reached {
    pub head: String,
    pub classifier: Option<String>,
        pub window: &'static str,
    pub limit: u32,
        pub count: i64,
    pub version: u64,
}

impl Reached {
            fn at_ceiling(rec: &HeadGuardRecord, counts: &[i64]) -> Option<Reached> {
        let mut out: Option<Reached> = None;
        for ((window, limit), &count) in ceilings(rec).into_iter().zip(counts) {
            let Some(limit) = limit else { continue };
            let over = count - i64::from(limit);
            if over >= 0
                && out
                    .as_ref()
                    .is_none_or(|o| over > o.count - i64::from(o.limit))
            {
                out = Some(Reached {
                    head: rec.head.clone(),
                    classifier: rec.classifier.clone(),
                    window,
                    limit,
                    count,
                    version: rec.version,
                });
            }
        }
        out
    }

        pub fn scope(&self) -> &'static str {
        if self.classifier.is_some() {
            "pair"
        } else {
            "head"
        }
    }

    fn lkey(&self) -> String {
        lkey_for(&self.head, self.classifier.as_deref())
    }

        pub fn trip_reason(&self) -> String {
        format!(
            "action ceiling: {} in {} (guard v{})",
            self.limit, self.window, self.version
        )
    }

        pub fn info_keys(&self) -> Vec<(String, String)> {
        vec![
            ("head_guard_scope".into(), self.scope().into()),
            ("head_guard_window".into(), self.window.into()),
            ("head_guard_limit".into(), self.limit.to_string()),
            ("head_guard_count".into(), self.count.to_string()),
            ("head_guard_version".into(), self.version.to_string()),
        ]
    }
}

pub type CounterFactory = Box<dyn Fn(&str) -> SlidingWindowLimiter + Send + Sync>;

struct ScopeCounter {
    limiter: SlidingWindowLimiter,
    task: JoinHandle<()>,
}

impl Drop for ScopeCounter {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Default)]
struct Snapshot {
        guards: HashMap<String, HeadGuardRecord>,
        unreadable: HashMap<String, (String, u64)>,
    loaded_at: Option<Instant>,
}

#[derive(Debug, PartialEq)]
pub enum SetError {
        Invalid(&'static str, String),
            Conflict(Option<HeadGuardRecord>, Option<u64>),
        LostRace(Option<HeadGuardRecord>),
        Store(String),
}

#[derive(Debug)]
pub struct SetOutcome {
    pub view: HeadGuardView,
    pub before: Option<HeadGuardRecord>,
}

type TripState = (u64, Option<Instant>);

pub struct HeadGuard {
    store: Arc<dyn ControlStore>,
    snapshot: RwLock<Arc<Snapshot>>,
    local_writes: Mutex<HashMap<String, (HeadGuardRecord, Instant)>>,
    created_at: Instant,
    store_timeout: Duration,
        make_counter: Option<CounterFactory>,
        counters: RwLock<HashMap<String, Arc<ScopeCounter>>>,
        trips: Mutex<HashMap<String, TripState>>,
}

impl HeadGuard {
            pub fn new(store: Arc<dyn ControlStore>, make_counter: Option<CounterFactory>) -> Self {
        Self {
            store,
            snapshot: RwLock::new(Arc::new(Snapshot::default())),
            local_writes: Mutex::new(HashMap::new()),
            created_at: Instant::now(),
            store_timeout: STORE_TIMEOUT,
            make_counter,
            counters: RwLock::new(HashMap::new()),
            trips: Mutex::new(HashMap::new()),
        }
    }

    #[cfg(test)]
    fn with_store_timeout(mut self, timeout: Duration) -> Self {
        self.store_timeout = timeout;
        self
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }


                    async fn scopes<'a>(
        &self,
        snap: &'a Snapshot,
        off_switch: &OffSwitch,
        head: &str,
        classifier: &str,
        now_ms: i64,
    ) -> Vec<(&'a HeadGuardRecord, Arc<ScopeCounter>)> {
        if !valid_name(head) {
            return Vec::new();
        }
        let mut lkeys = vec![lkey_for(head, None)];
        if valid_name(classifier) {
            lkeys.push(lkey_for(head, Some(classifier)));
        }
        let found: Vec<(&HeadGuardRecord, Arc<ScopeCounter>, Option<i64>)> = {
            let counters = self.counters.read().unwrap_or_else(|p| p.into_inner());
            lkeys
                .iter()
                .filter_map(|lkey| {
                    let rec = snap.guards.get(lkey)?;
                    let counter = counters.get(lkey)?.clone();
                    let floor_ms = off_switch.with_record(lkey, |s| counting_floor_ms(s, now_ms));
                    Some((rec, counter, floor_ms))
                })
                .collect()
        };
        let mut out = Vec::with_capacity(found.len());
        for (rec, counter, floor_ms) in found {
            if let Some(ms) = floor_ms
                && counter.limiter.set_floor(ms.max(0) as u64)
            {
                let _ = tokio::time::timeout(self.store_timeout, counter.limiter.refresh()).await;
            }
            out.push((rec, counter));
        }
        out
    }

                pub async fn check(
        &self,
        off_switch: &OffSwitch,
        head: &str,
        classifier: &str,
        now_ms: i64,
    ) -> Option<Reached> {
        let snap = self.snapshot();
        for (rec, counter) in self
            .scopes(&snap, off_switch, head, classifier, now_ms)
            .await
        {
            let Some(r) = Reached::at_ceiling(rec, &counter.limiter.rolling_cached()) else {
                continue;
            };
            self.trip(off_switch, &counter, &r, now_ms).await;
            count_metric(&r.head, r.scope(), r.window, "skipped");
            return Some(r);
        }
        None
    }

                        pub async fn count(
        &self,
        off_switch: &OffSwitch,
        head: &str,
        classifier: &str,
        now_ms: i64,
    ) -> Option<Reached> {
        let snap = self.snapshot();
        for (rec, counter) in self
            .scopes(&snap, off_switch, head, classifier, now_ms)
            .await
        {
            let Some(counts) = counter.limiter.increment_and_rolling().await else {
                count_metric(&rec.head, rec.scope(), "", "unavailable");
                continue;
            };
            for (window, limit) in ceilings(rec) {
                if limit.is_some() {
                    count_metric(&rec.head, rec.scope(), window, "counted");
                }
            }
            let Some(r) = Reached::at_ceiling(rec, &counts) else {
                continue;
            };
            self.trip(off_switch, &counter, &r, now_ms).await;
            if r.count > i64::from(r.limit) {
                count_metric(&r.head, r.scope(), r.window, "skipped");
                return Some(r);
            }
        }
        None
    }

    async fn trip(&self, off_switch: &OffSwitch, counter: &ScopeCounter, r: &Reached, now_ms: i64) {
        let result = self.trip_once(off_switch, counter, r, now_ms).await;
        count_metric(&r.head, r.scope(), r.window, result);
    }

                                            async fn trip_once(
        &self,
        off_switch: &OffSwitch,
        counter: &ScopeCounter,
        r: &Reached,
        now_ms: i64,
    ) -> &'static str {
        let lkey = r.lkey();
        let floor = counter.limiter.floor_ms();
        {
            let mut trips = self.trips.lock().unwrap_or_else(|p| p.into_inner());
            if let Some((f, retry_at)) = trips.get(&lkey)
                && *f == floor
                && retry_at.is_none_or(|t| Instant::now() < t)
            {
                return "trip_skipped";
            }
            trips.insert(lkey.clone(), (floor, None));
        }
        if off_switch.refresh().await.is_ok() {
            let resumed_at = off_switch.with_record(&lkey, |s| counting_floor_ms(s, now_ms));
            if resumed_at.is_none_or(|ms| ms.max(0) as u64 != floor) {
                return "trip_skipped";
            }
        }
        let _ = tokio::time::timeout(self.store_timeout, counter.limiter.write_snapshot()).await;
        let req = OffSwitchPutRequest {
            head: r.head.clone(),
            classifier: r.classifier.clone(),
            off: true,
            reason: r.trip_reason(),
            auto_on_at_ms: None,
            updated_by: Some(UPDATED_BY.into()),
        };
        let written =
            match tokio::time::timeout(self.store_timeout, off_switch.set(req, UPDATED_BY, now_ms))
                .await
            {
                Ok(Ok(_)) => Ok(()),
                Ok(Err(e)) => Err(format!("{e:?}")),
                Err(_) => Err("timed out".to_owned()),
            };
        match written {
            Ok(()) => {
                warn!(
                    lkey,
                    window = r.window,
                    limit = r.limit,
                    count = r.count,
                    version = r.version,
                    "head guard: action ceiling reached; off switch turned on"
                );
                "tripped"
            }
            Err(e) => {
                let mut trips = self.trips.lock().unwrap_or_else(|p| p.into_inner());
                if trips.get(&lkey) == Some(&(floor, None)) {
                    trips.insert(
                        lkey.clone(),
                        (floor, Some(Instant::now() + TRIP_RETRY_AFTER)),
                    );
                }
                error!(lkey, error = %e, "head guard: action ceiling reached but the off-switch write failed; a later decision retries");
                "trip_error"
            }
        }
    }


            pub async fn refresh(&self) -> anyhow::Result<usize> {
        let scanned = match tokio::time::timeout(self.store_timeout, self.store.scan(MH_PKEY)).await
        {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!(
                "head-guard scan timed out after {} ms",
                self.store_timeout.as_millis()
            )),
        };
        let scanned = scanned.and_then(|rows| {
            if rows.len() >= crate::action_control::SCAN_LIMIT as usize {
                Err(anyhow::anyhow!(
                    "head-guard scan returned {} rows, its limit; the list may be incomplete",
                    rows.len()
                ))
            } else {
                Ok(rows)
            }
        });
        let rows = match scanned {
            Ok(rows) => rows,
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                return Err(e);
            }
        };
        let n = rows.len();
        let prev = self.snapshot();
        let mut guards = HashMap::with_capacity(n);
        let mut unreadable = HashMap::new();
        for (lkey, value) in rows {
            match parse_record(&lkey, &value) {
                Ok(rec) => {
                    guards.insert(lkey, rec);
                }
                Err(err) => {
                    let (head, scope) = labels_of_lkey(&lkey);
                    metrics::HEAD_GUARD_TOTAL
                        .with_label_values(&[head, scope, "", "unreadable"])
                        .inc();
                    warn!(lkey = %lkey, error = %err, "head-guard record is unreadable; keeping the previous copy if any");
                    match prev.guards.get(&lkey) {
                        Some(old) => {
                            guards.insert(lkey, old.clone());
                        }
                        None => {
                            unreadable.insert(lkey, (err, raw_version(&value)));
                        }
                    }
                }
            }
        }
        {
            let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
            local.retain(|_, (_, at)| at.elapsed() < LOCAL_WRITE_GRACE);
            for (key, (rec, _)) in local.iter() {
                if guards.get(key).is_none_or(|s| s.version < rec.version) {
                    unreadable.remove(key);
                    guards.insert(key.clone(), rec.clone());
                }
            }
            self.publish(guards, unreadable, true);
        }
        Ok(n)
    }

    fn publish(
        &self,
        guards: HashMap<String, HeadGuardRecord>,
        unreadable: HashMap<String, (String, u64)>,
        loaded_now: bool,
    ) {
        let with_ceiling: Vec<String> = guards
            .iter()
            .filter(|(_, g)| ceilings(g).iter().any(|(_, limit)| limit.is_some()))
            .map(|(k, _)| k.clone())
            .collect();
        {
            let mut slot = self.snapshot.write().unwrap_or_else(|p| p.into_inner());
            let loaded_at = if loaded_now {
                Some(Instant::now())
            } else {
                slot.loaded_at
            };
            *slot = Arc::new(Snapshot {
                guards,
                unreadable,
                loaded_at,
            });
        }
        self.sync_counters(&with_ceiling);
    }

            fn sync_counters(&self, with_ceiling: &[String]) {
        let Some(make) = &self.make_counter else {
            return;
        };
        let mut counters = self.counters.write().unwrap_or_else(|p| p.into_inner());
        counters.retain(|k, _| with_ceiling.contains(k));
        for lkey in with_ceiling {
            counters.entry(lkey.clone()).or_insert_with(|| {
                let limiter = make(lkey);
                let task = limiter.spawn_background();
                Arc::new(ScopeCounter { limiter, task })
            });
        }
    }

        pub fn first_load_done(&self) -> bool {
        self.snapshot().loaded_at.is_some()
    }

                pub fn ready(&self, enabled: bool) -> bool {
        !enabled || self.first_load_done()
    }

        pub fn snapshot_age_ms(&self) -> Option<i64> {
        self.snapshot()
            .loaded_at
            .map(|t| t.elapsed().as_millis() as i64)
    }

                    pub fn spawn_refresh(self: &Arc<Self>, dynamic_config: DynamicConfig) -> JoinHandle<()> {
        self.spawn_refresh_with(
            move || dynamic_config.off_switch_enabled(),
            REFRESH_INTERVAL,
            FIRST_LOAD_BACKOFF,
        )
    }

    fn spawn_refresh_with(
        self: &Arc<Self>,
        enabled: impl Fn() -> bool + Send + 'static,
        interval: Duration,
        first_backoff: Duration,
    ) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            let mut failures: u64 = 0;
            let mut backoff = first_backoff;
            loop {
                if !enabled() {
                    tokio::time::sleep(interval).await;
                    continue;
                }
                let first = !this.first_load_done();
                match this.refresh().await {
                    Ok(n) => {
                        if first {
                            info!(
                                records = n,
                                attempts = failures + 1,
                                since_start_ms = this.created_at.elapsed().as_millis() as u64,
                                "head guard: first read ok; the pod can decide"
                            );
                        } else if failures > 0 {
                            info!(failures, "head guard: read ok again");
                        }
                        failures = 0;
                    }
                    Err(e) => {
                        failures += 1;
                        if first {
                            if failures == 1 || failures.is_multiple_of(FIRST_LOAD_ERROR_LOG_EVERY)
                            {
                                error!(
                                    error = %e,
                                    failures,
                                    "head guard: first read failed; the pod stays not ready and \
                                     does not consume Kafka until a read succeeds"
                                );
                            }
                        } else if failures == 1 || failures.is_multiple_of(REFRESH_ERROR_LOG_EVERY)
                        {
                            warn!(
                                error = %e,
                                failures,
                                age_ms = this.snapshot_age_ms().unwrap_or(-1),
                                "head guard: read failed; keeping the last good list"
                            );
                        }
                    }
                }
                let wait = if this.first_load_done() {
                    interval
                } else {
                    let wait = backoff;
                    backoff = (backoff * 2).min(interval);
                    wait
                };
                tokio::time::sleep(wait).await;
            }
        })
    }


    fn view(rec: HeadGuardRecord) -> HeadGuardView {
        HeadGuardView {
            scope: rec.scope().to_owned(),
            active: rec.is_active(),
            record: rec,
        }
    }

                pub async fn list(&self, enabled: bool) -> HeadGuardListResponse {
        let (source, error) = match self.refresh().await {
            Ok(_) => ("store", None),
            Err(e) => ("pod_snapshot", Some(e.to_string())),
        };
        let snap = self.snapshot();
        let mut guards: Vec<HeadGuardView> =
            snap.guards.values().cloned().map(Self::view).collect();
        guards.sort_by(|a, b| {
            (&a.record.head, &a.record.classifier).cmp(&(&b.record.head, &b.record.classifier))
        });
        let mut unreadable: Vec<UnreadableRecord> = snap
            .unreadable
            .iter()
            .map(|(lkey, (error, version))| UnreadableRecord {
                lkey: lkey.clone(),
                error: error.clone(),
                version: *version,
            })
            .collect();
        unreadable.sort_by(|a, b| a.lkey.cmp(&b.lkey));
        HeadGuardListResponse {
            guards,
            unreadable,
            source: source.to_owned(),
            snapshot_age_ms: self.snapshot_age_ms().unwrap_or(-1),
            enabled: enabled && self.make_counter.is_some(),
            error,
        }
    }

        pub fn validate(req: &HeadGuardPutRequest, who: &str) -> Result<(), SetError> {
        check_fields(
            &req.head,
            req.classifier.as_deref(),
            req.max_actions_per_hour,
            req.max_actions_per_day,
            req.overturn.as_ref(),
        )
        .map_err(|(code, detail)| SetError::Invalid(code, detail))?;
        if req.reason.trim().is_empty() {
            return Err(SetError::Invalid(
                "invalid_reason",
                "reason is required".into(),
            ));
        }
        if req.reason.len() > MAX_REASON_LEN {
            return Err(SetError::Invalid(
                "invalid_reason",
                format!("reason is longer than {MAX_REASON_LEN} bytes"),
            ));
        }
        if who.len() > MAX_NAME_LEN {
            return Err(SetError::Invalid(
                "invalid_updated_by",
                format!("who is longer than {MAX_NAME_LEN} bytes"),
            ));
        }
        Ok(())
    }

    async fn get_raw(&self, lkey: &str) -> Result<Option<Vec<u8>>, SetError> {
        match tokio::time::timeout(self.store_timeout, self.store.get(MH_PKEY, lkey)).await {
            Ok(Ok(raw)) => Ok(raw),
            Ok(Err(e)) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                Err(SetError::Store(e.to_string()))
            }
            Err(_) => Err(SetError::Store("store read timed out".into())),
        }
    }

                    pub async fn set(
        &self,
        req: HeadGuardPutRequest,
        who: &str,
        key_id: &str,
        now_ms: i64,
    ) -> Result<SetOutcome, SetError> {
        Self::validate(&req, who)?;
        let lkey = lkey_for(&req.head, req.classifier.as_deref());
        let stored = self.get_raw(&lkey).await?;
        let stored_version = stored.as_deref().map(raw_version);
        let before = stored.as_deref().and_then(|b| parse_record(&lkey, b).ok());
        if stored_version != req.expected_version {
            return Err(SetError::Conflict(before, stored_version));
        }
        let rec = HeadGuardRecord {
            v: HEAD_GUARD_RECORD_VERSION,
            head: req.head,
            classifier: req.classifier,
            max_actions_per_hour: req.max_actions_per_hour,
            max_actions_per_day: req.max_actions_per_day,
            overturn: req.overturn,
            reason: req.reason.trim().to_owned(),
            updated_by: who.to_owned(),
            key_id: key_id.to_owned(),
            updated_at_ms: now_ms,
            version: stored_version.unwrap_or(0) + 1,
            write_nonce: new_nonce(),
        };
        let value = serde_json::to_vec(&rec).map_err(|e| SetError::Store(e.to_string()))?;
        let write = self.store.put(MH_PKEY, &lkey, value, RECORD_TTL_SECS);
        match tokio::time::timeout(self.store_timeout, write).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                return Err(SetError::Store(e.to_string()));
            }
            Err(_) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                return Err(SetError::Store("store write timed out".into()));
            }
        }
        let readback = match self.get_raw(&lkey).await {
            Ok(Some(b)) => parse_record(&lkey, &b).ok(),
            Ok(None) | Err(_) => None,
        };
        match readback {
            Some(r) if r.write_nonce == rec.write_nonce => {}
            Some(r) => return Err(SetError::LostRace(Some(r))),
            None => {
                return Err(SetError::Store(
                    "the write landed but reading it back failed; pods pick up whatever the store holds within 5 s, so reload to see it"
                        .into(),
                ));
            }
        }
        self.apply_local(lkey.clone(), rec.clone());
        info!(
            lkey,
            version = rec.version,
            active = rec.is_active(),
            updated_by = %rec.updated_by,
            key_id,
            "head guard written"
        );
        Ok(SetOutcome {
            view: Self::view(rec),
            before,
        })
    }

            fn apply_local(&self, lkey: String, rec: HeadGuardRecord) {
        let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
        local.insert(lkey.clone(), (rec.clone(), Instant::now()));
        let snap = self.snapshot();
        let mut guards = snap.guards.clone();
        let mut unreadable = snap.unreadable.clone();
        unreadable.remove(&lkey);
        guards.insert(lkey, rec);
        self.publish(guards, unreadable, false);
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::action_control::test_support::MemControlStore;
    use crate::sliding_window::test_support::{self as sw, MemCounter, MemSnapshots};

        pub(crate) type Buckets = Arc<Mutex<HashMap<String, Arc<MemCounter>>>>;

    pub(crate) fn guard(
        head: &str,
        classifier: Option<&str>,
        per_hour: Option<u32>,
        per_day: Option<u32>,
    ) -> HeadGuardRecord {
        HeadGuardRecord {
            v: HEAD_GUARD_RECORD_VERSION,
            head: head.into(),
            classifier: classifier.map(Into::into),
            max_actions_per_hour: per_hour,
            max_actions_per_day: per_day,
            overturn: None,
            reason: format!("guard for {head}"),
            updated_by: "operator@example.com".into(),
            key_id: "pv2".into(),
            updated_at_ms: 1_000,
            version: 1,
            write_nonce: 7,
        }
    }

    pub(crate) fn put_raw(store: &MemControlStore, lkey: &str, value: Vec<u8>) {
        store
            .rows
            .lock()
            .unwrap()
            .insert((MH_PKEY.into(), lkey.into()), value);
    }

            pub(crate) async fn with_counters(
        guards: &[HeadGuardRecord],
    ) -> (Arc<HeadGuard>, Arc<MemControlStore>, Buckets) {
        let buckets = Buckets::default();
        let (hg, store) = with_counters_on(guards, &buckets, &Arc::default()).await;
        (hg, store, buckets)
    }

        pub(crate) async fn with_counters_on(
        guards: &[HeadGuardRecord],
        buckets: &Buckets,
        snapshots: &Arc<MemSnapshots>,
    ) -> (Arc<HeadGuard>, Arc<MemControlStore>) {
        let store = Arc::new(MemControlStore::default());
        for g in guards {
            let lkey = lkey_for(&g.head, g.classifier.as_deref());
            put_raw(&store, &lkey, serde_json::to_vec(g).unwrap());
            snapshots.put_at(&format!("{COUNT_PKEY_PREFIX}{lkey}"), sw::now() - 120, 0);
        }
        let make: CounterFactory = {
            let (buckets, snapshots) = (buckets.clone(), snapshots.clone());
            Box::new(move |lkey: &str| {
                let counter = buckets
                    .lock()
                    .unwrap()
                    .entry(lkey.to_owned())
                    .or_default()
                    .clone();
                sw::limiter(
                    counter,
                    snapshots.clone(),
                    &format!("{COUNT_PKEY_PREFIX}{lkey}"),
                    "head_guard",
                    WINDOWS,
                )
            })
        };
        let hg = Arc::new(
            HeadGuard::new(store.clone(), Some(make))
                .with_store_timeout(Duration::from_millis(100)),
        );
        hg.refresh().await.unwrap();
        let counters: Vec<Arc<ScopeCounter>> =
            hg.counters.read().unwrap().values().cloned().collect();
        for c in counters {
            c.limiter.refresh().await;
        }
        (hg, store)
    }

    pub(crate) fn bucket_count(buckets: &Buckets, lkey: &str) -> i64 {
        buckets
            .lock()
            .unwrap()
            .get(lkey)
            .map_or(0, |c| c.count.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use serde_json::{Value, json};

    use super::test_support::{bucket_count, guard, put_raw, with_counters, with_counters_on};
    use super::*;
    use crate::action_control::test_support::MemControlStore;
    use crate::off_switch::test_support::{MemStore, loaded, record};

    fn turn_on(head: &str, classifier: Option<&str>) -> OffSwitchPutRequest {
        OffSwitchPutRequest {
            head: head.into(),
            classifier: classifier.map(Into::into),
            off: false,
            reason: "resume".into(),
            auto_on_at_ms: None,
            updated_by: None,
        }
    }

    fn req(head: &str, classifier: Option<&str>) -> HeadGuardPutRequest {
        HeadGuardPutRequest {
            head: head.into(),
            classifier: classifier.map(Into::into),
            max_actions_per_hour: Some(10),
            max_actions_per_day: Some(100),
            overturn: None,
            reason: "runaway guard".into(),
            expected_version: None,
            updated_by: None,
        }
    }

    fn metric(head: &str, scope: &str, window: &str, result: &str) -> u64 {
        metrics::HEAD_GUARD_TOTAL
            .with_label_values(&[head, scope, window, result])
            .get()
    }

    #[test]
    fn bucket_ids_are_stable_and_clear_of_the_daily_caps() {
        assert_eq!(bucket_id("h/H"), 3_762_801_577_324_039_240);
        assert_eq!(bucket_id("p/H/c@1"), 6_395_408_749_496_773_277);
        for lkey in ["", "h/H", "p/H/c@1", "h/x"] {
            assert!(
                bucket_id(lkey) >= 1_000,
                "{lkey} may take a daily cap's bucket"
            );
        }
    }

    #[test]
    fn counting_restarts_at_the_last_turn_on() {
        assert_eq!(counting_floor_ms(None, 9_000), Some(0));
        let mut on = record("H", None, false, None);
        on.updated_at_ms = 5_000;
        assert_eq!(counting_floor_ms(Some(&on), 9_000), Some(5_000));
        let off = record("H", None, true, None);
        assert_eq!(counting_floor_ms(Some(&off), 9_000), None);
        let lapsing = record("H", None, true, Some(8_000));
        assert_eq!(counting_floor_ms(Some(&lapsing), 7_999), None);
        assert_eq!(counting_floor_ms(Some(&lapsing), 9_000), Some(8_000));
    }

    #[tokio::test]
    async fn reaching_a_ceiling_writes_the_off_switch_exactly_once() {
        let h = "TripOnceHead";
        let (hg, _, buckets) = with_counters(&[guard(h, None, Some(3), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        let (counted, tripped, skipped) = (
            metric(h, "head", "1h", "counted"),
            metric(h, "head", "1h", "tripped"),
            metric(h, "head", "1h", "trip_skipped"),
        );
        for _ in 0..2 {
            assert_eq!(hg.count(&sw, h, "c@1", now).await, None);
        }
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 0);

        assert_eq!(hg.count(&sw, h, "c@1", now).await, None);
        let s = sw.get(&format!("h/{h}")).expect("switched off");
        assert!(s.is_active(now));
        assert_eq!(s.classifier, None);
        assert_eq!(s.reason, "action ceiling: 3 in 1h (guard v1)");
        assert_eq!(s.updated_by, UPDATED_BY);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1);

        let over = hg.count(&sw, h, "c@1", now).await.expect("past it");
        assert_eq!(
            (
                over.scope(),
                over.window,
                over.limit,
                over.count,
                over.version
            ),
            ("head", "1h", 3, 4, 1)
        );
        let at = hg.check(&sw, h, "c@1", now).await.expect("at it");
        assert_eq!(at.count, 4);
        assert_eq!(bucket_count(&buckets, &format!("h/{h}")), 4);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1);
        assert_eq!(metric(h, "head", "1h", "counted"), counted + 4);
        assert_eq!(metric(h, "head", "1h", "tripped"), tripped + 1);
        assert_eq!(metric(h, "head", "1h", "trip_skipped"), skipped + 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_decisions_trip_once() {
        let (hg, _, _) = with_counters(&[guard("H", None, None, Some(5))]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        let tasks: Vec<_> = (0..40)
            .map(|_| {
                let (hg, sw) = (hg.clone(), sw.clone());
                tokio::spawn(async move { hg.count(&sw, "H", "c@1", now).await })
            })
            .collect();
        let mut acted = 0;
        for t in tasks {
            match t.await.unwrap() {
                None => acted += 1,
                Some(r) => assert_eq!((r.window, r.limit), ("24h", 5)),
            }
        }
        assert_eq!(acted, 5);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1);
        assert_eq!(
            sw.get("h/H").unwrap().reason,
            "action ceiling: 5 in 24h (guard v1)"
        );
    }

                #[tokio::test]
    async fn a_second_pod_does_not_trip_a_switch_already_on() {
        let guards = [guard("H", None, Some(2), None)];
        let buckets = test_support::Buckets::default();
        let snapshots = Arc::default();
        let (pod_a, _) = with_counters_on(&guards, &buckets, &snapshots).await;
        let (pod_b, _) = with_counters_on(&guards, &buckets, &snapshots).await;
        let sw_store = Arc::new(MemStore::default());
        let switches = || {
            Arc::new(OffSwitch::new(
                sw_store.clone(),
                Arc::new(MemStore::default()),
            ))
        };
        let (sw_a, sw_b) = (switches(), switches());
        sw_a.refresh().await.unwrap();
        sw_b.refresh().await.unwrap();
        let now = crate::now_millis();

        assert_eq!(pod_a.count(&sw_a, "H", "c", now).await, None);
        assert_eq!(pod_a.count(&sw_a, "H", "c", now).await, None);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1, "pod A tripped");
        assert!(sw_b.get("h/H").is_none(), "pod B's list is stale");

        let r = pod_b.count(&sw_b, "H", "c", now).await.expect("past it");
        assert_eq!(r.count, 3);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1, "no second write");

        let resumed = now + 1_000;
        sw_a.set(turn_on("H", None), "pv2", resumed).await.unwrap();
        sw_b.refresh().await.unwrap();
        assert_eq!(pod_b.count(&sw_b, "H", "c", resumed).await, None);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 3);
        assert!(sw_b.get("h/H").unwrap().is_active(resumed));
    }

    #[tokio::test]
    async fn a_failed_trip_is_retried_after_a_pause() {
        let h = "RetryHead";
        let lkey = format!("h/{h}");
        let (hg, _, _) = with_counters(&[guard(h, None, Some(1), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        let errors = metric(h, "head", "1h", "trip_error");
        sw_store.fail_put.store(true, Ordering::SeqCst);
        assert_eq!(hg.count(&sw, h, "c@1", now).await, None);
        assert!(sw.get(&lkey).is_none());
        assert_eq!(metric(h, "head", "1h", "trip_error"), errors + 1);

        sw_store.fail_put.store(false, Ordering::SeqCst);
        assert!(hg.check(&sw, h, "c@1", now).await.is_some());
        assert!(sw.get(&lkey).is_none());

        hg.trips.lock().unwrap().get_mut(&lkey).unwrap().1 = Some(Instant::now());
        assert!(hg.check(&sw, h, "c@1", now).await.is_some());
        assert!(sw.get(&lkey).unwrap().is_active(now));
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn counting_restarts_when_the_switch_is_turned_back_on() {
        let (hg, _, buckets) = with_counters(&[guard("H", None, Some(2), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        for _ in 0..2 {
            assert_eq!(hg.count(&sw, "H", "c@1", now).await, None);
        }
        assert!(sw.get("h/H").unwrap().is_active(now));

        let resumed = now + 1_000;
        sw.set(turn_on("H", None), "pv2", resumed).await.unwrap();
        assert_eq!(
            hg.check(&sw, "H", "c@1", resumed).await,
            None,
            "nothing before the resume counts"
        );
        assert_eq!(hg.count(&sw, "H", "c@1", resumed).await, None);
        let counter = hg.counters.read().unwrap()["h/H"].clone();
        assert_eq!(counter.limiter.floor_ms(), resumed as u64);
        counter.limiter.write_snapshot().await;

        let resaved = resumed + 1_000;
        sw.set(turn_on("H", None), "pv2", resaved).await.unwrap();
        assert_eq!(sw.get("h/H").unwrap().updated_at_ms, resaved);
        assert_eq!(hg.count(&sw, "H", "c@1", resaved).await, None);
        assert_eq!(counter.limiter.floor_ms(), resaved as u64);
        assert!(
            !sw.get("h/H").unwrap().is_active(resaved),
            "the decision before the re-save no longer counts"
        );
        assert_eq!(hg.count(&sw, "H", "c@1", resaved).await, None);
        assert!(sw.get("h/H").unwrap().is_active(resaved), "tripped again");
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 4);
        assert_eq!(bucket_count(&buckets, "h/H"), 5);
    }

    #[tokio::test]
    async fn a_resume_in_the_same_second_as_the_last_one_restarts_counting() {
        let (hg, _, _) = with_counters(&[guard("H", None, Some(1), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let second = crate::now_millis() / 1_000 * 1_000;
        sw.set(turn_on("H", None), "pv2", second + 100)
            .await
            .unwrap();
        assert_eq!(hg.count(&sw, "H", "c", second + 200).await, None);
        assert!(sw.get("h/H").unwrap().is_active(second + 200), "tripped");

        let resumed = second + 300;
        sw.set(turn_on("H", None), "pv2", resumed).await.unwrap();
        assert_eq!(
            hg.check(&sw, "H", "c", resumed).await,
            None,
            "nothing before the resume counts"
        );
        assert_eq!(hg.count(&sw, "H", "c", resumed).await, None);
        assert!(sw.get("h/H").unwrap().is_active(resumed), "tripped again");
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn a_passed_auto_on_restarts_counting() {
        let (hg, _, _) = with_counters(&[guard("H", None, Some(1), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        assert_eq!(hg.count(&sw, "H", "c", now).await, None);
        assert!(sw.get("h/H").unwrap().is_active(now));

        let mut lapsing = record("H", None, true, Some(now + 500));
        lapsing.updated_at_ms = now;
        sw_store.put_record(&lapsing);
        sw.refresh().await.unwrap();
        assert!(
            hg.check(&sw, "H", "c", now + 100).await.is_some(),
            "still on"
        );
        let later = now + 2_000;
        assert_eq!(hg.check(&sw, "H", "c", later).await, None, "lapsed");
        assert_eq!(hg.count(&sw, "H", "c", later).await, None);
        let s = sw.get("h/H").unwrap();
        assert!(
            s.is_active(later) && s.updated_by == UPDATED_BY,
            "tripped again"
        );
    }

    #[tokio::test]
    async fn with_both_ceilings_the_hour_wins_a_tie() {
        let (hg, _, _) = with_counters(&[guard("H", None, Some(2), Some(2))]).await;
        let (sw, _) = loaded(&[]).await;
        let now = crate::now_millis();
        assert_eq!(hg.count(&sw, "H", "c", now).await, None);
        assert_eq!(hg.count(&sw, "H", "c", now).await, None);
        assert_eq!(
            sw.get("h/H").unwrap().reason,
            "action ceiling: 2 in 1h (guard v1)"
        );
    }

    #[tokio::test]
    async fn head_guards_count_every_classifier_and_pair_guards_one() {
        let (hg, _, buckets) = with_counters(&[
            guard("H", None, Some(4), None),
            guard("H", Some("c@1"), Some(1), None),
        ])
        .await;
        let (sw, _) = loaded(&[]).await;
        let now = crate::now_millis();

        assert_eq!(hg.count(&sw, "H", "c@2", now).await, None);
        assert_eq!(hg.count(&sw, "G", "c@1", now).await, None);
        assert_eq!(hg.count(&sw, "H", "", now).await, None);
        assert_eq!(bucket_count(&buckets, "h/H"), 2);
        assert_eq!(bucket_count(&buckets, "p/H/c@1"), 0);

        assert_eq!(hg.count(&sw, "H", "c@1", now).await, None);
        assert_eq!(bucket_count(&buckets, "h/H"), 3);
        assert_eq!(bucket_count(&buckets, "p/H/c@1"), 1);
        let pair = sw.get("p/H/c@1").expect("pair switched off");
        assert!(pair.is_active(now));
        assert_eq!(pair.classifier.as_deref(), Some("c@1"));
        assert!(sw.get("h/H").is_none());

        assert_eq!(hg.count(&sw, "H", "c@2", now).await, None);
        assert_eq!(bucket_count(&buckets, "h/H"), 4);
        assert!(sw.get("h/H").unwrap().is_active(now));

        assert!(hg.count(&sw, "H", "c@1", now).await.is_some());
        assert_eq!(bucket_count(&buckets, "p/H/c@1"), 1);

        sw.set(turn_on("H", None), "pv2", now + 1_000)
            .await
            .unwrap();
        let r = hg.check(&sw, "H", "c@1", now + 1_000).await.unwrap();
        assert_eq!((r.scope(), r.count), ("pair", 1));
        assert_eq!(hg.check(&sw, "H", "c@2", now + 1_000).await, None);
    }

    #[tokio::test]
    async fn without_a_limiter_nothing_counts() {
        let store = Arc::new(MemControlStore::default());
        put_raw(
            &store,
            "h/H",
            serde_json::to_vec(&guard("H", None, Some(1), None)).unwrap(),
        );
        let hg = HeadGuard::new(store, None);
        let (sw, sw_store) = loaded(&[]).await;
        let list = hg.list(true).await;
        assert_eq!(list.guards.len(), 1);
        assert!(!list.enabled);
        for _ in 0..3 {
            assert_eq!(hg.count(&sw, "H", "c", crate::now_millis()).await, None);
        }
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 0);
    }

            #[tokio::test]
    async fn the_counts_fail_open() {
        let h = "FailOpenHead";
        let (hg, _, buckets) = with_counters(&[guard(h, None, Some(1), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        let unavailable = metric(h, "head", "", "unavailable");
        let bucket = buckets.lock().unwrap()[&format!("h/{h}")].clone();
        bucket.fail.store(true, Ordering::SeqCst);
        for _ in 0..3 {
            assert_eq!(hg.count(&sw, h, "c", now).await, None);
        }
        assert_eq!(bucket_count(&buckets, &format!("h/{h}")), 0);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 0);
        assert_eq!(metric(h, "head", "", "unavailable"), unavailable + 3);
        bucket.fail.store(false, Ordering::SeqCst);
        assert_eq!(hg.count(&sw, h, "c", now).await, None);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1, "works again");

        let store = Arc::new(MemControlStore::default());
        put_raw(
            &store,
            "h/U",
            serde_json::to_vec(&guard("U", None, Some(1), None)).unwrap(),
        );
        let cold = test_support::Buckets::default();
        let make: CounterFactory = {
            let cold = cold.clone();
            Box::new(move |lkey: &str| {
                let counter = cold
                    .lock()
                    .unwrap()
                    .entry(lkey.to_owned())
                    .or_default()
                    .clone();
                crate::sliding_window::test_support::limiter(
                    counter,
                    Arc::default(),
                    lkey,
                    "head_guard",
                    WINDOWS,
                )
            })
        };
        let hg = HeadGuard::new(store, Some(make));
        hg.refresh().await.unwrap();
        cold.lock().unwrap()["h/U"]
            .count
            .store(1_000_000, Ordering::SeqCst);
        assert_eq!(hg.check(&sw, "U", "c", now).await, None);
        assert_eq!(hg.count(&sw, "U", "c", now).await, None);
        assert_eq!(hg.check(&sw, "U", "c", now).await, None);
        assert!(sw.get("h/U").is_none());
    }

            #[tokio::test]
    async fn a_counter_reset_does_not_wedge_a_scope_at_its_ceiling() {
        let guards = [guard("H", None, Some(10), None)];
        let buckets = test_support::Buckets::default();
        let snapshots = Arc::new(crate::sliding_window::test_support::MemSnapshots::default());
        let (hg, _) = with_counters_on(&guards, &buckets, &snapshots).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        let pkey = format!("{COUNT_PKEY_PREFIX}h/H");
        let counter = hg.counters.read().unwrap()["h/H"].clone();
        let bucket = buckets.lock().unwrap()["h/H"].clone();

        snapshots.rows.lock().unwrap().clear();
        counter.limiter.refresh().await;
        bucket.count.store(1_000, Ordering::SeqCst);
        assert_eq!(hg.count(&sw, "H", "c", now).await, None);
        counter.limiter.write_snapshot().await;

        snapshots.rows.lock().unwrap().clear();
        let secs = (now / 1_000) as u64;
        snapshots.put_at(&pkey, secs - 10_800, 1_001);
        snapshots.put_at(&pkey, secs - 7_200, 3);
        snapshots.put_at(&pkey, secs - 1_800, 8);
        bucket.count.store(8, Ordering::SeqCst);
        counter.limiter.refresh().await;

        assert_eq!(hg.check(&sw, "H", "c", now).await, None);
        assert_eq!(hg.count(&sw, "H", "c", now).await, None);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn malformed_records_are_unreadable() {
        let (hg, store, _) = with_counters(&[guard("Ok", None, Some(5), None)]).await;
        let metric_head = "UnreadableMetricHead";
        let unreadable_before = metric(metric_head, "head", "", "unreadable");
        let bad = [
            ("h/A", json!("{not json")),
            (
                "h/B",
                json!({"head": "B", "version": 4, "max_actions_per_hour": 0}),
            ),
            (
                "h/C",
                json!({"head": "C", "version": 1, "max_actions_per_hour": 10, "max_actions_per_day": 5}),
            ),
            (
                "h/D",
                json!({"head": "D", "version": 1, "overturn": {"max_overturn_rate": 0.2, "window_hours": 169}}),
            ),
            (
                "h/E",
                json!({"head": "E", "version": 1, "overturn": {"max_overturn_rate": 1.0}}),
            ),
            (
                "h/F",
                json!({"head": "F", "version": 1, "overturn": {"max_overturn_rate": 0.2, "min_decided_appeals": 0}}),
            ),
            ("h/G", json!({"head": "G", "v": 2, "version": 1})),
            ("h/I", json!({"head": "I", "max_actions_per_day": 5})),
            ("h/J", json!({"head": "J/x", "version": 1})),
            (
                "p/K/c",
                json!({"head": "K", "classifier": "", "version": 1, "max_actions_per_day": 5}),
            ),
            (
                "h/M",
                json!({"head": "N", "version": 2, "max_actions_per_day": 5}),
            ),
            (
                "p/P/c",
                json!({"head": "P", "version": 3, "max_actions_per_day": 5}),
            ),
            ("h/UnreadableMetricHead", json!("[]")),
        ];
        for (lkey, value) in bad {
            let bytes = match value {
                Value::String(s) => s.into_bytes(),
                v => serde_json::to_vec(&v).unwrap(),
            };
            put_raw(&store, lkey, bytes);
        }
        let list = hg.list(true).await;
        assert_eq!(list.source, "store");
        assert_eq!(
            list.guards
                .iter()
                .map(|g| g.record.head.as_str())
                .collect::<Vec<_>>(),
            ["Ok"]
        );
        let unreadable: Vec<(&str, u64)> = list
            .unreadable
            .iter()
            .map(|u| (u.lkey.as_str(), u.version))
            .collect();
        assert_eq!(
            unreadable,
            [
                ("h/A", 0),
                ("h/B", 4),
                ("h/C", 1),
                ("h/D", 1),
                ("h/E", 1),
                ("h/F", 1),
                ("h/G", 1),
                ("h/I", 0),
                ("h/J", 1),
                ("h/M", 2),
                ("h/UnreadableMetricHead", 0),
                ("p/K/c", 1),
                ("p/P/c", 3),
            ]
        );
        assert_eq!(
            metric(metric_head, "head", "", "unreadable"),
            unreadable_before + 1
        );

        put_raw(&store, "h/Ok", b"{}".to_vec());
        let list = hg.list(true).await;
        assert_eq!(list.guards[0].record.max_actions_per_hour, Some(5));
        assert!(list.unreadable.iter().all(|u| u.lkey != "h/Ok"));

        let mut fix = req("B", None);
        fix.expected_version = Some(4);
        let out = hg.set(fix, "op@example.com", "pv2", 5_000).await.unwrap();
        assert_eq!(out.view.record.version, 5);
        assert_eq!(out.before, None);
        let list = hg.list(true).await;
        assert!(list.guards.iter().any(|g| g.record.head == "B"));
        assert!(list.unreadable.iter().all(|u| u.lkey != "h/B"));
    }

    #[tokio::test]
    async fn list_falls_back_to_the_pods_copy_and_local_writes_win_for_a_while() {
        let (hg, store, _) = with_counters(&[guard("H", None, Some(5), None)]).await;
        assert!(hg.list(true).await.enabled);
        assert!(!hg.list(false).await.enabled);

        let mut raise = req("H", None);
        raise.expected_version = Some(1);
        hg.set(raise, "op@example.com", "pv2", 5_000).await.unwrap();
        put_raw(
            &store,
            "h/H",
            serde_json::to_vec(&guard("H", None, Some(5), None)).unwrap(),
        );
        let list = hg.list(true).await;
        assert_eq!(list.guards[0].record.version, 2);

        store.fail_scan.store(true, Ordering::SeqCst);
        let list = hg.list(true).await;
        assert_eq!(list.source, "pod_snapshot");
        assert!(list.error.is_some());
        assert_eq!(list.guards[0].record.version, 2);
    }

    type Edit = fn(&mut HeadGuardPutRequest);

    #[test]
    fn put_validation_matches_the_read_rules() {
        let cases: [(&str, Edit); 10] = [
            ("invalid_head", |r| r.head = "a/b".into()),
            ("invalid_head", |r| r.head = String::new()),
            ("invalid_classifier", |r| r.classifier = Some(String::new())),
            ("invalid_limit", |r| r.max_actions_per_hour = Some(0)),
            ("invalid_limit", |r| {
                r.max_actions_per_day = Some(MAX_LIMIT + 1)
            }),
            ("invalid_limit", |r| r.max_actions_per_day = Some(9)),
            ("invalid_overturn", |r| {
                r.overturn = Some(OverturnGuard {
                    max_overturn_rate: 0.0,
                    min_decided_appeals: 30,
                    window_hours: 72,
                })
            }),
            ("invalid_overturn", |r| {
                r.overturn = Some(OverturnGuard {
                    max_overturn_rate: 0.2,
                    min_decided_appeals: 30,
                    window_hours: MAX_OVERTURN_WINDOW_HOURS + 1,
                })
            }),
            ("invalid_reason", |r| r.reason = "  ".into()),
            ("invalid_reason", |r| r.reason = "x".repeat(1_001)),
        ];
        for (code, edit) in cases {
            let mut r = req("H", None);
            edit(&mut r);
            match HeadGuard::validate(&r, "op") {
                Err(SetError::Invalid(got, _)) => assert_eq!(got, code, "{r:?}"),
                other => panic!("{r:?}: {other:?}"),
            }
        }
        assert!(matches!(
            HeadGuard::validate(&req("H", None), &"x".repeat(257)),
            Err(SetError::Invalid("invalid_updated_by", _))
        ));
        let mut overturn_only = req("H", Some("c@1"));
        overturn_only.max_actions_per_hour = None;
        overturn_only.max_actions_per_day = None;
        overturn_only.overturn = Some(OverturnGuard {
            max_overturn_rate: 0.2,
            min_decided_appeals: 30,
            window_hours: MAX_OVERTURN_WINDOW_HOURS,
        });
        assert_eq!(HeadGuard::validate(&overturn_only, "op"), Ok(()));
    }

    #[tokio::test]
    async fn put_versions_and_clears_a_guard() {
        let (hg, _, _) = with_counters(&[]).await;
        let out = hg
            .set(req("H", None), "op@example.com", "pv2", 5_000)
            .await
            .unwrap();
        let rec = &out.view.record;
        assert_eq!((rec.v, rec.version), (HEAD_GUARD_RECORD_VERSION, 1));
        assert_eq!(
            (
                rec.updated_by.as_str(),
                rec.key_id.as_str(),
                rec.updated_at_ms
            ),
            ("op@example.com", "pv2", 5_000)
        );
        assert_eq!((out.view.scope.as_str(), out.view.active), ("head", true));
        assert_eq!(out.before, None);
        assert!(hg.counters.read().unwrap().contains_key("h/H"));

        match hg.set(req("H", None), "op@example.com", "pv2", 6_000).await {
            Err(SetError::Conflict(Some(current), Some(1))) => assert_eq!(current.version, 1),
            other => panic!("{other:?}"),
        }

        let mut clear = req("H", None);
        clear.max_actions_per_hour = None;
        clear.max_actions_per_day = None;
        clear.expected_version = Some(1);
        let out = hg.set(clear, "op@example.com", "pv2", 7_000).await.unwrap();
        assert_eq!(out.view.record.version, 2);
        assert!(!out.view.active);
        assert_eq!(out.before.unwrap().max_actions_per_hour, Some(10));
        assert!(!hg.counters.read().unwrap().contains_key("h/H"));
        let list = hg.list(true).await;
        assert_eq!(list.guards.len(), 1, "a cleared guard stays listed");
        assert!(!list.guards[0].active);
    }

    #[tokio::test]
    async fn put_reports_lost_races_and_store_failures() {
        let (hg, store, _) = with_counters(&[]).await;
        let mut other = guard("H", None, Some(3), None);
        other.write_nonce = 99;
        *store.clobber_with.lock().unwrap() = Some(serde_json::to_vec(&other).unwrap());
        match hg.set(req("H", None), "op", "pv2", 5_000).await {
            Err(SetError::LostRace(Some(r))) => assert_eq!(r.write_nonce, 99),
            other => panic!("{other:?}"),
        }
        assert!(
            !hg.snapshot().guards.contains_key("h/H"),
            "the losing write is not installed"
        );

        *store.clobber_with.lock().unwrap() = Some(b"{}".to_vec());
        let mut retry = req("H", None);
        retry.expected_version = Some(1);
        assert!(matches!(
            hg.set(retry.clone(), "op", "pv2", 5_000).await,
            Err(SetError::Store(e)) if e.contains("reading it back failed")
        ));
        *store.clobber_with.lock().unwrap() = None;

        store.fail_get.store(true, Ordering::SeqCst);
        assert!(matches!(
            hg.set(retry.clone(), "op", "pv2", 5_000).await,
            Err(SetError::Store(_))
        ));
        store.fail_get.store(false, Ordering::SeqCst);
        store.fail_put.store(true, Ordering::SeqCst);
        retry.expected_version = Some(0);
        assert!(matches!(
            hg.set(retry, "op", "pv2", 5_000).await,
            Err(SetError::Store(_))
        ));
    }

            #[tokio::test]
    async fn a_hung_trip_write_times_out() {
        let h = "HungTripHead";
        let lkey = format!("h/{h}");
        let (hg, _, _) = with_counters(&[guard(h, None, Some(1), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        sw_store.hang_put.store(true, Ordering::SeqCst);
        let (errors, skipped) = (
            metric(h, "head", "1h", "trip_error"),
            metric(h, "head", "1h", "trip_skipped"),
        );
        let bounded = Duration::from_secs(2);
        let r = tokio::time::timeout(bounded, hg.count(&sw, h, "c", now)).await;
        assert_eq!(r.expect("bounded by the store timeout"), None);
        assert_eq!(metric(h, "head", "1h", "trip_error"), errors + 1);
        assert!(sw.get(&lkey).is_none());
        let r = tokio::time::timeout(bounded, hg.check(&sw, h, "c", now)).await;
        assert!(r.expect("no second write in the pause").is_some());
        assert_eq!(metric(h, "head", "1h", "trip_skipped"), skipped + 1);
    }

                #[tokio::test]
    async fn a_hung_snapshot_write_does_not_stall_the_trip() {
        let h = "HungSnapshotHead";
        let guards = [guard(h, None, Some(1), None)];
        let buckets = test_support::Buckets::default();
        let snapshots = Arc::new(crate::sliding_window::test_support::MemSnapshots::default());
        let (hg, _) = with_counters_on(&guards, &buckets, &snapshots).await;
        let (sw, sw_store) = loaded(&[]).await;
        let now = crate::now_millis();
        snapshots.hang_put.store(true, Ordering::SeqCst);
        let tripped = metric(h, "head", "1h", "tripped");
        let r = tokio::time::timeout(Duration::from_secs(2), hg.count(&sw, h, "c", now)).await;
        assert_eq!(r.expect("bounded by the store timeout"), None);
        assert!(sw.get(&format!("h/{h}")).unwrap().is_active(now));
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1);
        assert_eq!(metric(h, "head", "1h", "tripped"), tripped + 1);

        snapshots.hang_put.store(false, Ordering::SeqCst);
        let counter = hg.counters.read().unwrap()[&format!("h/{h}")].clone();
        counter.limiter.write_snapshot().await;
        let rows = snapshots.rows.lock().unwrap();
        assert!(rows.values().any(|v| v == "1"), "retried: {rows:?}");
    }

            #[tokio::test]
    async fn a_hung_read_after_a_resume_counts_from_the_floor() {
        let guards = [guard("H", None, Some(2), None)];
        let buckets = test_support::Buckets::default();
        let snapshots = Arc::new(crate::sliding_window::test_support::MemSnapshots::default());
        let (hg, _) = with_counters_on(&guards, &buckets, &snapshots).await;
        let (sw, _) = loaded(&[]).await;
        let now = crate::now_millis();
        for _ in 0..2 {
            assert_eq!(hg.count(&sw, "H", "c", now).await, None);
        }
        let resumed = now + 1_000;
        sw.set(turn_on("H", None), "pv2", resumed).await.unwrap();
        snapshots.hang_scan.store(true, Ordering::SeqCst);
        let r =
            tokio::time::timeout(Duration::from_secs(2), hg.count(&sw, "H", "c", resumed)).await;
        assert_eq!(r.expect("bounded by the store timeout"), None);
        let counter = hg.counters.read().unwrap()["h/H"].clone();
        assert_eq!(counter.limiter.floor_ms(), resumed as u64);
        assert_eq!(counter.limiter.rolling_cached(), vec![0, 0]);
        assert!(!sw.get("h/H").unwrap().is_active(resumed));
    }

    #[tokio::test]
    async fn the_refresh_loop_retries_the_first_read_and_sleeps_while_disabled() {
        let store = Arc::new(MemControlStore::default());
        store.fail_scan.store(true, Ordering::SeqCst);
        let hg = Arc::new(HeadGuard::new(store.clone(), None));
        let task =
            hg.spawn_refresh_with(|| true, Duration::from_millis(40), Duration::from_millis(2));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!hg.first_load_done());
        assert!(store.scans.load(Ordering::SeqCst) > 1, "retried");
        store.fail_scan.store(false, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(2), async {
            while !hg.first_load_done() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("loaded once a read succeeds");
        task.abort();

        let store = Arc::new(MemControlStore::default());
        let hg = Arc::new(HeadGuard::new(store.clone(), None));
        let task = hg.spawn_refresh_with(
            || false,
            Duration::from_millis(10),
            Duration::from_millis(2),
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(store.scans.load(Ordering::SeqCst), 0);
        assert!(hg.ready(false) && !hg.ready(true));
        task.abort();
    }
}
