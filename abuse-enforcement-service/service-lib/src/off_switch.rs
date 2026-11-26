use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};
use xai_manhattan::{LkeySelector, NativeManhattanClient, Tenant};

pub use xai_abuse_enforcement_types::off_switch::{
    OFF_SWITCH_RECORD_VERSION, OffSwitchListResponse, OffSwitchPutRequest, OffSwitchRecord,
    OffSwitchSeen, OffSwitchView, PreviousOff,
};

use crate::growthbook::DynamicConfig;
use crate::metrics;

pub const MH_PKEY: &str = "off_switch";

pub const STATUS_SWITCHED_OFF: &str = "enforcement_switched_off";

pub const CONFIG_KEY: &str = "enforcement_off_switch";

pub const REFRESH_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) const FIRST_LOAD_BACKOFF: Duration = Duration::from_secs(1);

pub(crate) const FIRST_LOAD_ERROR_LOG_EVERY: u64 = 12;

const SCAN_TIMEOUT: Duration = Duration::from_secs(5);

const SCAN_PAGE: u32 = 1_000;

const SCAN_MAX_ROWS: usize = 100_000;

pub const RECORD_TTL_SECS: u64 = 10 * 365 * 86_400;

pub const SEEN_PKEY: &str = "off_switch_seen";

pub const SEEN_WRITE_EVERY: Duration = Duration::from_secs(600);

pub const SEEN_TTL_SECS: u64 = 30 * 86_400;

const SEEN_QUEUE: usize = 256;

const SEEN_GUARD_MAX: usize = 10_000;

pub(crate) const LOCAL_WRITE_GRACE: Duration = Duration::from_secs(30);

pub(crate) const MAX_NAME_LEN: usize = 256;
pub(crate) const MAX_REASON_LEN: usize = 1_000;

pub(crate) const REFRESH_ERROR_LOG_EVERY: u64 = 60;

pub fn lkey_for(head: &str, classifier: Option<&str>) -> String {
    match classifier {
        None => format!("h/{head}"),
        Some(c) => format!("p/{head}/{c}"),
    }
}

pub fn seen_lkey(head: &str, classifier: &str) -> String {
    format!("{head}/{classifier}")
}

pub fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_NAME_LEN
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'@' | b'-'))
}


#[async_trait]
pub trait OffSwitchStore: Send + Sync {
        async fn scan(&self) -> anyhow::Result<Vec<(String, Vec<u8>)>>;
    async fn put(&self, lkey: &str, value: Vec<u8>) -> anyhow::Result<()>;
}

pub struct ManhattanOffSwitchStore {
    client: Arc<NativeManhattanClient>,
    tenant: Tenant,
    pkey: &'static str,
    ttl_secs: u64,
}

impl ManhattanOffSwitchStore {
    pub fn new(
        client: Arc<NativeManhattanClient>,
        tenant: Tenant,
        pkey: &'static str,
        ttl_secs: u64,
    ) -> Self {
        Self {
            client,
            tenant,
            pkey,
            ttl_secs,
        }
    }
}

async fn scan_pages<F, Fut>(
    page: u32,
    max_rows: usize,
    mut fetch: F,
) -> anyhow::Result<Vec<(Vec<u8>, Vec<u8>)>>
where
    F: FnMut(Option<Vec<u8>>, u32) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<Vec<(Vec<u8>, Vec<u8>)>>>,
{
    let mut rows: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut from: Option<Vec<u8>> = None;
    loop {
        let batch = fetch(from.clone(), page).await?;
        let full = batch.len() >= page as usize;
        let read_before = rows.len();
        rows.extend(
            batch
                .into_iter()
                .filter(|(lkey, _)| from.as_ref() != Some(lkey)),
        );
        if rows.len() > max_rows {
            anyhow::bail!("partition has more than {max_rows} rows; not using a partial read");
        }
        if !full {
            return Ok(rows);
        }
        if rows.len() == read_before {
            anyhow::bail!("scan made no progress past a full page");
        }
        from = rows.last().map(|(lkey, _)| lkey.clone());
    }
}

#[async_trait]
impl OffSwitchStore for ManhattanOffSwitchStore {
    async fn scan(&self) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
        let rows = scan_pages(SCAN_PAGE, SCAN_MAX_ROWS, |from, limit| async move {
            let items = self
                .client
                .scan(
                    self.tenant.clone(),
                    [self.pkey],
                    LkeySelector::Range {
                        from: from.map(|lkey| [lkey].into()),
                        to: None,
                    },
                    limit,
                )
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let rows: Vec<(Vec<u8>, Vec<u8>)> = items
                .iter()
                .filter_map(|it| {
                    let lkey = it.lkey().into_byte_vecs().into_iter().next()?;
                    Some((lkey, it.value().as_bytes().to_vec()))
                })
                .collect();
            anyhow::Ok(rows)
        })
        .await
        .map_err(|e| anyhow::anyhow!("scan of pkey {}: {e:#}", self.pkey))?;
        Ok(rows
            .into_iter()
            .filter_map(|(lkey, value)| Some((String::from_utf8(lkey).ok()?, value)))
            .collect())
    }

    async fn put(&self, lkey: &str, value: Vec<u8>) -> anyhow::Result<()> {
        self.client
            .put_with_ttl(
                self.tenant.clone(),
                [self.pkey],
                [lkey],
                value,
                crate::manhattan::ttl_nanos_from_secs(self.ttl_secs),
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}


pub fn enabled_from_config(config: Option<&Value>) -> bool {
    static LAST_BAD: Mutex<Option<String>> = Mutex::new(None);
    let Some(v) = config.and_then(|c| c.get(CONFIG_KEY)) else {
        return true;
    };
    if v.is_null() {
        return true;
    }
    if let Some(enabled) = v.get("enabled").and_then(Value::as_bool) {
        return enabled;
    }
    let bad = v.to_string();
    let mut last = LAST_BAD.lock().unwrap_or_else(|p| p.into_inner());
    if last.as_deref() != Some(bad.as_str()) {
        error!(
            value = %bad,
            "GrowthBook {CONFIG_KEY} is malformed (want {{\"enabled\": true|false}}); keeping the off switch enabled"
        );
        *last = Some(bad);
    }
    true
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Head,
    Pair,
}

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Head => "head",
            Scope::Pair => "pair",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OffSwitchHit {
    pub scope: Scope,
    pub record: OffSwitchRecord,
}

impl OffSwitchHit {
        pub fn info_keys(&self) -> Vec<(String, String)> {
        let mut keys = vec![
            (
                "off_switch_scope".to_owned(),
                self.scope.as_str().to_owned(),
            ),
            ("off_switch_reason".to_owned(), self.record.reason.clone()),
            (
                "off_switch_updated_by".to_owned(),
                self.record.updated_by.clone(),
            ),
            (
                "off_switch_updated_at_ms".to_owned(),
                self.record.updated_at_ms.to_string(),
            ),
        ];
        if let Some(t) = self.record.auto_on_at_ms {
            keys.push(("off_switch_auto_on_at_ms".to_owned(), t.to_string()));
        }
        keys
    }
}

#[derive(Default)]
struct Snapshot {
        entries: HashMap<String, OffSwitchRecord>,
    loaded_at: Option<Instant>,
}

#[derive(Debug, PartialEq)]
pub enum SetError {
        Invalid(String),
        Store(String),
}

#[derive(Debug)]
pub struct SetOutcome {
    pub view: OffSwitchView,
    pub before: Option<OffSwitchRecord>,
}

pub struct OffSwitch {
    store: Arc<dyn OffSwitchStore>,
    snapshot: RwLock<Arc<Snapshot>>,
    local_writes: Mutex<HashMap<String, (OffSwitchRecord, Instant)>>,
    created_at: Instant,
    scan_timeout: Duration,
    seen_store: Arc<dyn OffSwitchStore>,
        seen_guard: Mutex<HashMap<String, Instant>>,
    seen_tx: mpsc::Sender<OffSwitchSeen>,
        seen_rx: Mutex<Option<mpsc::Receiver<OffSwitchSeen>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeenOutcome {
    Queued,
        Recent,
        Dropped,
    Invalid,
}

impl OffSwitch {
            pub fn new(store: Arc<dyn OffSwitchStore>, seen_store: Arc<dyn OffSwitchStore>) -> Self {
        Self::with_seen_queue(store, seen_store, SEEN_QUEUE)
    }

    fn with_seen_queue(
        store: Arc<dyn OffSwitchStore>,
        seen_store: Arc<dyn OffSwitchStore>,
        queue: usize,
    ) -> Self {
        let (seen_tx, seen_rx) = mpsc::channel(queue);
        Self {
            store,
            snapshot: RwLock::new(Arc::new(Snapshot::default())),
            local_writes: Mutex::new(HashMap::new()),
            created_at: Instant::now(),
            scan_timeout: SCAN_TIMEOUT,
            seen_store,
            seen_guard: Mutex::new(HashMap::new()),
            seen_tx,
            seen_rx: Mutex::new(Some(seen_rx)),
        }
    }

    #[cfg(test)]
    fn with_scan_timeout(mut self, timeout: Duration) -> Self {
        self.scan_timeout = timeout;
        self
    }


                pub fn record_seen(&self, head: &str, classifier: &str, now_ms: i64) -> SeenOutcome {
        self.record_seen_at(head, classifier, now_ms, Instant::now())
    }

    fn record_seen_at(
        &self,
        head: &str,
        classifier: &str,
        now_ms: i64,
        now: Instant,
    ) -> SeenOutcome {
        if !valid_name(head) || !valid_name(classifier) {
            metrics::OFF_SWITCH_SEEN_TOTAL
                .with_label_values(&["invalid"])
                .inc();
            return SeenOutcome::Invalid;
        }
        let key = seen_lkey(head, classifier);
        {
            let mut guard = self.seen_guard.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(at) = guard.get(&key)
                && now.saturating_duration_since(*at) < SEEN_WRITE_EVERY
            {
                return SeenOutcome::Recent;
            }
            if guard.len() >= SEEN_GUARD_MAX && !guard.contains_key(&key) {
                guard.retain(|_, at| now.saturating_duration_since(*at) < SEEN_WRITE_EVERY);
                if guard.len() >= SEEN_GUARD_MAX {
                    metrics::OFF_SWITCH_SEEN_TOTAL
                        .with_label_values(&["dropped"])
                        .inc();
                    return SeenOutcome::Dropped;
                }
            }
            guard.insert(key.clone(), now);
        }
        let seen = OffSwitchSeen {
            head: head.to_owned(),
            classifier: classifier.to_owned(),
            last_seen_ms: now_ms,
        };
        match self.seen_tx.try_send(seen) {
            Ok(()) => {
                metrics::OFF_SWITCH_SEEN_TOTAL
                    .with_label_values(&["queued"])
                    .inc();
                SeenOutcome::Queued
            }
            Err(_) => {
                metrics::OFF_SWITCH_SEEN_TOTAL
                    .with_label_values(&["dropped"])
                    .inc();
                self.forget_seen(&key, Some(now));
                SeenOutcome::Dropped
            }
        }
    }

            fn forget_seen(&self, key: &str, at: Option<Instant>) {
        let mut guard = self.seen_guard.lock().unwrap_or_else(|p| p.into_inner());
        if at.is_none_or(|at| guard.get(key) == Some(&at)) {
            guard.remove(key);
        }
    }

            pub fn spawn_seen_writer(self: &Arc<Self>) -> Option<JoinHandle<()>> {
        let mut rx = self
            .seen_rx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()?;
        let this = self.clone();
        Some(tokio::spawn(async move {
            let mut failures: u64 = 0;
            while let Some(seen) = rx.recv().await {
                match this.write_seen(&seen).await {
                    Ok(()) => failures = 0,
                    Err(e) => {
                        failures += 1;
                        if failures == 1 || failures.is_multiple_of(REFRESH_ERROR_LOG_EVERY) {
                            warn!(
                                error = %e,
                                failures,
                                head = %seen.head,
                                classifier = %seen.classifier,
                                "off switch: seen-pair write failed; the pair's next decision retries"
                            );
                        }
                    }
                }
            }
        }))
    }

    async fn write_seen(&self, seen: &OffSwitchSeen) -> anyhow::Result<()> {
        let key = seen_lkey(&seen.head, &seen.classifier);
        let value = serde_json::to_vec(seen)?;
        let result =
            match tokio::time::timeout(self.scan_timeout, self.seen_store.put(&key, value)).await {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!(
                    "seen-pair write timed out after {} ms",
                    self.scan_timeout.as_millis()
                )),
            };
        match &result {
            Ok(()) => metrics::OFF_SWITCH_SEEN_TOTAL
                .with_label_values(&["written"])
                .inc(),
            Err(_) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                metrics::OFF_SWITCH_SEEN_TOTAL
                    .with_label_values(&["error"])
                    .inc();
                self.forget_seen(&key, None);
            }
        }
        result
    }

        async fn scan_seen(&self) -> anyhow::Result<Vec<OffSwitchSeen>> {
        let rows = match tokio::time::timeout(self.scan_timeout, self.seen_store.scan()).await {
            Ok(rows) => rows?,
            Err(_) => anyhow::bail!(
                "seen-pair scan timed out after {} ms",
                self.scan_timeout.as_millis()
            ),
        };
        let mut seen: Vec<OffSwitchSeen> = rows
            .iter()
            .filter_map(|(_, value)| serde_json::from_slice::<OffSwitchSeen>(value).ok())
            .filter(|s| valid_name(&s.head) && valid_name(&s.classifier))
            .collect();
        seen.sort_by(|a, b| (&a.head, &a.classifier).cmp(&(&b.head, &b.classifier)));
        Ok(seen)
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

            pub fn lookup(&self, head: &str, classifier: &str, now_ms: i64) -> Option<OffSwitchHit> {
        if head.is_empty() {
            return None;
        }
        let snap = self.snapshot();
        if snap.entries.is_empty() {
            return None;
        }
        if let Some(r) = snap.entries.get(&lkey_for(head, None))
            && r.is_active(now_ms)
        {
            return Some(OffSwitchHit {
                scope: Scope::Head,
                record: r.clone(),
            });
        }
        if classifier.is_empty() {
            return None;
        }
        snap.entries
            .get(&lkey_for(head, Some(classifier)))
            .filter(|r| r.is_active(now_ms))
            .map(|r| OffSwitchHit {
                scope: Scope::Pair,
                record: r.clone(),
            })
    }

            pub fn with_record<R>(&self, lkey: &str, f: impl FnOnce(Option<&OffSwitchRecord>) -> R) -> R {
        f(self.snapshot().entries.get(lkey))
    }

    #[cfg(test)]
    pub(crate) fn get(&self, lkey: &str) -> Option<OffSwitchRecord> {
        self.with_record(lkey, |r| r.cloned())
    }

            pub fn active_lkeys(&self, now_ms: i64) -> Vec<String> {
        let mut keys: Vec<String> = self
            .snapshot()
            .entries
            .iter()
            .filter(|(_, r)| r.is_active(now_ms))
            .map(|(k, _)| k.clone())
            .collect();
        keys.sort();
        keys
    }

        pub fn snapshot_age_ms(&self) -> Option<i64> {
        self.snapshot()
            .loaded_at
            .map(|t| t.elapsed().as_millis() as i64)
    }

            pub async fn refresh(&self) -> anyhow::Result<usize> {
        let scanned = match tokio::time::timeout(self.scan_timeout, self.store.scan()).await {
            Ok(scanned) => scanned,
            Err(_) => Err(anyhow::anyhow!(
                "off-switch scan timed out after {} ms",
                self.scan_timeout.as_millis()
            )),
        };
        let rows = match scanned {
            Ok(rows) => rows,
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                metrics::OFF_SWITCH_REFRESH_TOTAL
                    .with_label_values(&["error"])
                    .inc();
                self.publish_metrics(crate::now_millis());
                return Err(e);
            }
        };
        let n = rows.len();
        let prev = self.snapshot();
        let mut entries = HashMap::with_capacity(n);
        for (lkey, value) in rows {
            match serde_json::from_slice::<OffSwitchRecord>(&value) {
                Ok(rec)
                    if valid_name(&rec.head)
                        && rec.classifier.as_deref().is_none_or(valid_name) =>
                {
                    entries.insert(lkey_for(&rec.head, rec.classifier.as_deref()), rec);
                }
                _ => {
                    metrics::OFF_SWITCH_BAD_RECORDS_TOTAL.inc();
                    warn!(lkey = %lkey, "off-switch record does not parse; keeping the previous copy if any");
                    if let Some(old) = prev.entries.get(&lkey) {
                        entries.insert(lkey, old.clone());
                    }
                }
            }
        }
        {
            let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
            local.retain(|_, (_, at)| at.elapsed() < LOCAL_WRITE_GRACE);
            for (key, (rec, _)) in local.iter() {
                if entries
                    .get(key)
                    .is_none_or(|stored| stored.updated_at_ms < rec.updated_at_ms)
                {
                    entries.insert(key.clone(), rec.clone());
                }
            }
            self.publish(entries, true);
        }
        metrics::OFF_SWITCH_FIRST_LOAD_PENDING.set(0);
        metrics::OFF_SWITCH_REFRESH_TOTAL
            .with_label_values(&["ok"])
            .inc();
        Ok(n)
    }

    fn publish(&self, entries: HashMap<String, OffSwitchRecord>, loaded_now: bool) {
        {
            let mut slot = self.snapshot.write().unwrap_or_else(|p| p.into_inner());
            let loaded_at = if loaded_now {
                Some(Instant::now())
            } else {
                slot.loaded_at
            };
            *slot = Arc::new(Snapshot { entries, loaded_at });
        }
        self.publish_metrics(crate::now_millis());
    }

                fn publish_metrics(&self, now_ms: i64) {
        let snap = self.snapshot();
        let active = snap
            .entries
            .values()
            .filter(|r| r.is_active(now_ms))
            .count();
        metrics::OFF_SWITCH_ENTRIES
            .with_label_values(&["active"])
            .set(active as i64);
        metrics::OFF_SWITCH_ENTRIES
            .with_label_values(&["inactive"])
            .set((snap.entries.len() - active) as i64);
        let age = snap.loaded_at.unwrap_or(self.created_at).elapsed();
        metrics::OFF_SWITCH_SNAPSHOT_AGE_SECONDS.set(age.as_secs() as i64);
    }

        pub fn first_load_done(&self) -> bool {
        self.snapshot().loaded_at.is_some()
    }

                pub fn ready(&self, enabled: bool) -> bool {
        !enabled || self.first_load_done()
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
                let enabled = enabled();
                metrics::OFF_SWITCH_ENABLED.set(i64::from(enabled));
                if !enabled {
                    metrics::OFF_SWITCH_FIRST_LOAD_PENDING.set(0);
                    metrics::OFF_SWITCH_REFRESH_TOTAL
                        .with_label_values(&["disabled"])
                        .inc();
                    this.publish_metrics(crate::now_millis());
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
                                "off switch: first read ok; the pod can decide"
                            );
                        } else if failures > 0 {
                            info!(failures, "off switch: read ok again");
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
                                    "off switch: first read failed; the pod stays not ready and \
                                     does not consume Kafka until a read succeeds"
                                );
                            }
                        } else if failures == 1 || failures.is_multiple_of(REFRESH_ERROR_LOG_EVERY)
                        {
                            warn!(
                                error = %e,
                                failures,
                                age_ms = this.snapshot_age_ms().unwrap_or(-1),
                                "off switch: read failed; keeping the last good list"
                            );
                        }
                    }
                }
                let pending = !this.first_load_done();
                metrics::OFF_SWITCH_FIRST_LOAD_PENDING.set(i64::from(pending));
                let wait = if pending {
                    let wait = backoff;
                    backoff = (backoff * 2).min(interval);
                    wait
                } else {
                    interval
                };
                tokio::time::sleep(wait).await;
            }
        })
    }


    fn view(rec: OffSwitchRecord, now_ms: i64) -> OffSwitchView {
        OffSwitchView {
            scope: rec.scope().to_owned(),
            active: rec.is_active(now_ms),
            record: rec,
        }
    }

                pub async fn list(&self, enabled: bool, now_ms: i64) -> OffSwitchListResponse {
        let (refreshed, seen) = tokio::join!(self.refresh(), self.scan_seen());
        let (source, error) = match refreshed {
            Ok(_) => ("store", None),
            Err(e) => ("pod_snapshot", Some(e.to_string())),
        };
        let (seen, seen_error) = match seen {
            Ok(seen) => (seen, None),
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                (Vec::new(), Some(e.to_string()))
            }
        };
        let snap = self.snapshot();
        let mut switches: Vec<OffSwitchView> = snap
            .entries
            .values()
            .cloned()
            .map(|r| Self::view(r, now_ms))
            .collect();
        switches.sort_by(|a, b| {
            (&a.record.head, &a.record.classifier).cmp(&(&b.record.head, &b.record.classifier))
        });
        OffSwitchListResponse {
            switches,
            seen,
            source: source.to_owned(),
            snapshot_age_ms: self.snapshot_age_ms().unwrap_or(-1),
            enabled,
            error,
            seen_error,
        }
    }

        pub fn validate(req: &OffSwitchPutRequest, now_ms: i64) -> Result<(), String> {
        if !valid_name(&req.head) {
            return Err(format!(
                "head must be 1-{MAX_NAME_LEN} characters of [A-Za-z0-9_.:@-]"
            ));
        }
        if let Some(c) = &req.classifier
            && !valid_name(c)
        {
            return Err(format!(
                "classifier must be null or 1-{MAX_NAME_LEN} characters of [A-Za-z0-9_.:@-]"
            ));
        }
        if req.off && req.reason.trim().is_empty() {
            return Err("reason is required when turning enforcement off".into());
        }
        if req.reason.len() > MAX_REASON_LEN {
            return Err(format!("reason is longer than {MAX_REASON_LEN} bytes"));
        }
        if req
            .updated_by
            .as_deref()
            .is_some_and(|u| u.len() > MAX_NAME_LEN)
        {
            return Err(format!("updated_by is longer than {MAX_NAME_LEN} bytes"));
        }
        match req.auto_on_at_ms {
            Some(_) if !req.off => Err("auto_on_at_ms only applies when off is true".into()),
            Some(t) if t <= now_ms => Err("auto_on_at_ms must be in the future".into()),
            _ => Ok(()),
        }
    }

        pub fn updated_by(req: &OffSwitchPutRequest, key_id: &str) -> String {
        req.updated_by
            .as_deref()
            .map(str::trim)
            .filter(|u| !u.is_empty())
            .unwrap_or(key_id)
            .to_owned()
    }

                                pub async fn set(
        &self,
        req: OffSwitchPutRequest,
        key_id: &str,
        now_ms: i64,
    ) -> Result<SetOutcome, SetError> {
        Self::validate(&req, now_ms).map_err(SetError::Invalid)?;
        let updated_by = Self::updated_by(&req, key_id);
        let lkey = lkey_for(&req.head, req.classifier.as_deref());
        let previous_off = if req.off {
            None
        } else {
            self.with_record(&lkey, |r| {
                r.filter(|r| r.off).map(|r| PreviousOff {
                    reason: r.reason.clone(),
                    updated_by: r.updated_by.clone(),
                    updated_at_ms: r.updated_at_ms,
                })
            })
        };
        let rec = OffSwitchRecord {
            v: OFF_SWITCH_RECORD_VERSION,
            head: req.head,
            classifier: req.classifier,
            off: req.off,
            reason: req.reason.trim().to_owned(),
            updated_by,
            key_id: key_id.to_owned(),
            updated_at_ms: now_ms,
            auto_on_at_ms: req.auto_on_at_ms,
            previous_off,
        };
        let value = serde_json::to_vec(&rec).map_err(|e| SetError::Store(e.to_string()))?;
        if let Err(e) = self.store.put(&lkey, value).await {
            metrics::MANHATTAN_ERRORS_TOTAL.inc();
            return Err(SetError::Store(e.to_string()));
        }
        let before = self.apply_local(lkey, rec.clone());
        info!(
            head = %rec.head,
            classifier = rec.classifier.as_deref().unwrap_or(""),
            off = rec.off,
            updated_by = %rec.updated_by,
            key_id,
            "off switch written"
        );
        Ok(SetOutcome {
            view: Self::view(rec, now_ms),
            before,
        })
    }

            fn apply_local(&self, lkey: String, rec: OffSwitchRecord) -> Option<OffSwitchRecord> {
        let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
        local.insert(lkey.clone(), (rec.clone(), Instant::now()));
        let mut entries = self.snapshot().entries.clone();
        let before = entries.insert(lkey, rec);
        self.publish(entries, false);
        before
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

        #[derive(Default)]
    pub(crate) struct MemStore {
        pub rows: Mutex<Vec<(String, Vec<u8>)>>,
        pub fail_scan: std::sync::atomic::AtomicBool,
        pub fail_put: std::sync::atomic::AtomicBool,
        pub hang_scan: std::sync::atomic::AtomicBool,
        pub hang_put: std::sync::atomic::AtomicBool,
        pub scans: std::sync::atomic::AtomicUsize,
                pub puts: std::sync::atomic::AtomicUsize,
    }

    impl MemStore {
        pub(crate) fn put_record(&self, rec: &OffSwitchRecord) {
            let lkey = lkey_for(&rec.head, rec.classifier.as_deref());
            let value = serde_json::to_vec(rec).unwrap();
            let mut rows = self.rows.lock().unwrap();
            rows.retain(|(k, _)| *k != lkey);
            rows.push((lkey, value));
        }
    }

    #[async_trait]
    impl OffSwitchStore for MemStore {
        async fn scan(&self) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
            use std::sync::atomic::Ordering;
            self.scans.fetch_add(1, Ordering::Relaxed);
            if self.hang_scan.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            if self.fail_scan.load(Ordering::Relaxed) {
                anyhow::bail!("scan failed (test)");
            }
            Ok(self.rows.lock().unwrap().clone())
        }

        async fn put(&self, lkey: &str, value: Vec<u8>) -> anyhow::Result<()> {
            if self.hang_put.load(std::sync::atomic::Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            if self.fail_put.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("put failed (test)");
            }
            self.puts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut rows = self.rows.lock().unwrap();
            rows.retain(|(k, _)| k != lkey);
            rows.push((lkey.to_owned(), value));
            Ok(())
        }
    }

    pub(crate) fn record(
        head: &str,
        classifier: Option<&str>,
        off: bool,
        auto_on_at_ms: Option<i64>,
    ) -> OffSwitchRecord {
        OffSwitchRecord {
            v: OFF_SWITCH_RECORD_VERSION,
            head: head.into(),
            classifier: classifier.map(Into::into),
            off,
            reason: format!("reason for {head}"),
            updated_by: "operator@example.com".into(),
            key_id: "pv2".into(),
            updated_at_ms: 1_000,
            auto_on_at_ms,
            previous_off: None,
        }
    }

            pub(crate) async fn loaded(records: &[OffSwitchRecord]) -> (Arc<OffSwitch>, Arc<MemStore>) {
        let store = Arc::new(MemStore::default());
        for r in records {
            store.put_record(r);
        }
        let sw = Arc::new(OffSwitch::new(store.clone(), Arc::new(MemStore::default())));
        sw.refresh().await.unwrap();
        (sw, store)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use std::sync::atomic::Ordering;

    const NOW: i64 = 1_800_000_000_000;

    #[tokio::test]
    async fn whole_head_wins_over_pair() {
        let (sw, _) = loaded(&[
            record("H", None, true, None),
            record("H", Some("c1"), true, None),
        ])
        .await;
        let hit = sw.lookup("H", "c1", NOW).unwrap();
        assert_eq!(hit.scope, Scope::Head);
        assert_eq!(sw.lookup("H", "other", NOW).unwrap().scope, Scope::Head);
        assert_eq!(sw.lookup("H", "", NOW).unwrap().scope, Scope::Head);
    }

    #[tokio::test]
    async fn pair_applies_only_to_its_classifier() {
        let (sw, _) = loaded(&[record("H", Some("c1"), true, None)]).await;
        assert_eq!(sw.lookup("H", "c1", NOW).unwrap().scope, Scope::Pair);
        assert!(sw.lookup("H", "c2", NOW).is_none());
        assert!(sw.lookup("H", "", NOW).is_none());
        assert!(sw.lookup("Other", "c1", NOW).is_none());
    }

    #[tokio::test]
    async fn head_turned_back_on_does_not_clear_a_pair() {
        let (sw, _) = loaded(&[
            record("H", None, false, None),
            record("H", Some("c1"), true, None),
        ])
        .await;
        assert_eq!(sw.lookup("H", "c1", NOW).unwrap().scope, Scope::Pair);
        assert!(sw.lookup("H", "c2", NOW).is_none());
    }

    #[tokio::test]
    async fn empty_head_is_never_switched_off() {
        let (sw, _) = loaded(&[record("H", None, true, None)]).await;
        assert!(sw.lookup("", "c1", NOW).is_none());
    }

    #[tokio::test]
    async fn auto_on_time_ends_the_switch() {
        let (sw, _) = loaded(&[
            record("H", None, true, Some(NOW + 1)),
            record("P", Some("c"), true, Some(NOW)),
        ])
        .await;
        assert!(sw.lookup("H", "x", NOW).is_some());
        assert!(sw.lookup("H", "x", NOW + 1).is_none());
        assert!(sw.lookup("P", "c", NOW - 1).is_some());
        assert!(sw.lookup("P", "c", NOW).is_none());
    }

    #[tokio::test]
    async fn auto_on_head_falls_through_to_pair() {
        let (sw, _) = loaded(&[
            record("H", None, true, Some(NOW)),
            record("H", Some("c1"), true, None),
        ])
        .await;
        assert_eq!(sw.lookup("H", "c1", NOW).unwrap().scope, Scope::Pair);
    }

    #[tokio::test]
    async fn failed_reads_keep_the_last_good_list() {
        let (sw, store) = loaded(&[record("H", None, true, None)]).await;
        let age_before = sw.snapshot_age_ms().unwrap();
        store.fail_scan.store(true, Ordering::Relaxed);
        for _ in 0..5 {
            assert!(sw.refresh().await.is_err());
        }
        assert!(sw.lookup("H", "c", NOW).is_some());
        assert!(sw.snapshot_age_ms().unwrap() >= age_before);
        store.fail_scan.store(false, Ordering::Relaxed);
        store.put_record(&record("H", None, false, None));
        sw.refresh().await.unwrap();
        assert!(sw.lookup("H", "c", NOW).is_none());
    }

    #[tokio::test]
    async fn unparseable_record_keeps_previous_copy() {
        let (sw, store) = loaded(&[record("H", None, true, None)]).await;
        store.rows.lock().unwrap()[0].1 = b"not json".to_vec();
        sw.refresh().await.unwrap();
        assert!(sw.lookup("H", "c", NOW).is_some());
        store
            .rows
            .lock()
            .unwrap()
            .push(("h/New".into(), b"{".to_vec()));
        sw.refresh().await.unwrap();
        assert!(sw.lookup("New", "c", NOW).is_none());
    }

    #[tokio::test]
    async fn reader_trusts_value_fields_not_the_lkey() {
        let store = Arc::new(MemStore::default());
        store.rows.lock().unwrap().push((
            "h/WrongName".into(),
            serde_json::to_vec(&record("Right", None, true, None)).unwrap(),
        ));
        let sw = OffSwitch::new(store, Arc::new(MemStore::default()));
        sw.refresh().await.unwrap();
        assert!(sw.lookup("Right", "c", NOW).is_some());
        assert!(sw.lookup("WrongName", "c", NOW).is_none());
    }

    async fn wait_for_scans(store: &MemStore, n: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while store.scans.load(Ordering::Relaxed) < n {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("background reads keep retrying");
    }

    async fn until_ready(sw: &OffSwitch, within: Duration) -> bool {
        tokio::time::timeout(within, async {
            while !sw.ready(true) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .is_ok()
    }

    #[tokio::test]
    async fn not_ready_until_the_first_read_succeeds() {
        let store = Arc::new(MemStore::default());
        store.put_record(&record("H", None, true, None));
        store.fail_scan.store(true, Ordering::Relaxed);
        let sw = Arc::new(OffSwitch::new(store.clone(), Arc::new(MemStore::default())));
        let task =
            sw.spawn_refresh_with(|| true, Duration::from_millis(40), Duration::from_millis(2));
        wait_for_scans(&store, 4).await;
        assert!(!sw.ready(true), "a failed first read is not an empty list");
        assert!(!sw.first_load_done());
        assert!(sw.snapshot_age_ms().is_none());
        assert!(
            !until_ready(&sw, Duration::from_millis(300)).await,
            "not ready while reads fail"
        );

        store.fail_scan.store(false, Ordering::Relaxed);
        assert!(
            until_ready(&sw, Duration::from_secs(5)).await,
            "ready once a read succeeds"
        );
        assert!(sw.lookup("H", "c", NOW).is_some());
        task.abort();
    }

    #[tokio::test]
    async fn a_hung_first_read_keeps_the_pod_waiting() {
        let store = Arc::new(MemStore::default());
        store.put_record(&record("H", None, true, None));
        store.hang_scan.store(true, Ordering::Relaxed);
        let sw = Arc::new(
            OffSwitch::new(store.clone(), Arc::new(MemStore::default()))
                .with_scan_timeout(Duration::from_millis(20)),
        );
        let task =
            sw.spawn_refresh_with(|| true, Duration::from_millis(40), Duration::from_millis(2));
        wait_for_scans(&store, 3).await;
        assert!(!sw.ready(true), "each hung read times out and is retried");

        store.hang_scan.store(false, Ordering::Relaxed);
        assert!(
            until_ready(&sw, Duration::from_secs(5)).await,
            "ready once a read answers"
        );
        assert!(sw.lookup("H", "c", NOW).is_some());
        task.abort();
    }

    #[tokio::test]
    async fn disabled_feature_does_not_wait_or_read() {
        let store = Arc::new(MemStore::default());
        store.fail_scan.store(true, Ordering::Relaxed);
        let sw = Arc::new(OffSwitch::new(store.clone(), Arc::new(MemStore::default())));
        let task = sw.spawn_refresh_with(
            || false,
            Duration::from_millis(10),
            Duration::from_millis(2),
        );
        assert!(sw.ready(false), "no wait with the feature off");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(store.scans.load(Ordering::Relaxed), 0);
        task.abort();
    }

            fn page_of(
        rows: &std::collections::BTreeMap<Vec<u8>, Vec<u8>>,
        from: Option<Vec<u8>>,
        limit: u32,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        rows.range(from.unwrap_or_default()..)
            .take(limit as usize)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn numbered_rows(n: usize) -> std::collections::BTreeMap<Vec<u8>, Vec<u8>> {
        (0..n)
            .map(|i| (format!("h/H{i:05}").into_bytes(), vec![b'v']))
            .collect()
    }

    #[tokio::test]
    async fn scan_reads_every_page_once() {
        for (n, want_calls) in [(2_500, 3), (2_000, 3), (999, 1), (0, 1)] {
            let rows = numbered_rows(n);
            let mut calls = 0;
            let got = scan_pages(1_000, SCAN_MAX_ROWS, |from, limit| {
                calls += 1;
                let page = page_of(&rows, from, limit);
                async move { anyhow::Ok(page) }
            })
            .await
            .unwrap();
            let want: Vec<Vec<u8>> = rows.keys().cloned().collect();
            let got: Vec<Vec<u8>> = got.into_iter().map(|(k, _)| k).collect();
            assert_eq!(got, want, "{n} rows: each once, in order");
            assert_eq!(calls, want_calls, "{n} rows");
        }
    }

    #[tokio::test]
    async fn scan_fails_past_the_row_maximum_or_a_failed_page() {
        let rows = numbered_rows(30);
        let err = scan_pages(10, 25, |from, limit| {
            let page = page_of(&rows, from, limit);
            async move { anyhow::Ok(page) }
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("more than 25 rows"), "{err}");

        let mut calls = 0;
        let err = scan_pages(10, 1_000, |from, limit| {
            calls += 1;
            let page = if calls == 2 {
                Err(anyhow::anyhow!("page 2 failed"))
            } else {
                Ok(page_of(&rows, from, limit))
            };
            async move { page }
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("page 2 failed"), "{err}");
    }

        #[derive(Default)]
    struct SmallPagedStore {
        rows: Mutex<std::collections::BTreeMap<Vec<u8>, Vec<u8>>>,
    }

    #[async_trait]
    impl OffSwitchStore for SmallPagedStore {
        async fn scan(&self) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
            let rows = self.rows.lock().unwrap().clone();
            let got = scan_pages(2, 3, |from, limit| {
                let page = page_of(&rows, from, limit);
                async move { anyhow::Ok(page) }
            })
            .await?;
            Ok(got
                .into_iter()
                .map(|(k, v)| (String::from_utf8(k).unwrap(), v))
                .collect())
        }

        async fn put(&self, _lkey: &str, _value: Vec<u8>) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_scan_over_the_row_maximum_keeps_the_last_complete_list() {
        let store = Arc::new(SmallPagedStore::default());
        let add = |head: &str| {
            let rec = record(head, None, true, None);
            store.rows.lock().unwrap().insert(
                lkey_for(&rec.head, None).into_bytes(),
                serde_json::to_vec(&rec).unwrap(),
            );
        };
        for head in ["A", "B", "C"] {
            add(head);
        }
        let sw = OffSwitch::new(store.clone(), Arc::new(MemStore::default()));
        assert_eq!(sw.refresh().await.unwrap(), 3, "two pages");

        add("D");
        let err = sw.refresh().await.unwrap_err();
        assert!(err.to_string().contains("more than 3 rows"), "{err}");
        for head in ["A", "B", "C"] {
            assert!(sw.lookup(head, "c", NOW).is_some(), "{head} still off");
        }
        assert!(
            sw.lookup("D", "c", NOW).is_none(),
            "no partial list swapped in"
        );
    }

    #[tokio::test]
    async fn a_hung_scan_times_out_and_keeps_the_last_good_list() {
        let store = Arc::new(MemStore::default());
        store.put_record(&record("H", None, true, None));
        let sw = OffSwitch::new(store.clone(), Arc::new(MemStore::default()))
            .with_scan_timeout(Duration::from_millis(50));
        sw.refresh().await.unwrap();
        store.hang_scan.store(true, Ordering::Relaxed);
        let err = tokio::time::timeout(Duration::from_secs(5), sw.refresh())
            .await
            .expect("refresh must end at the scan timeout")
            .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(sw.lookup("H", "c", NOW).is_some());
    }

    #[test]
    fn kill_switch_config() {
        use serde_json::json;
        assert!(enabled_from_config(None));
        assert!(enabled_from_config(Some(&json!({}))));
        assert!(enabled_from_config(Some(&json!({CONFIG_KEY: null}))));
        assert!(enabled_from_config(Some(
            &json!({CONFIG_KEY: {"enabled": true}})
        )));
        assert!(!enabled_from_config(Some(
            &json!({CONFIG_KEY: {"enabled": false}})
        )));
        assert!(enabled_from_config(Some(&json!({CONFIG_KEY: false}))));
        assert!(enabled_from_config(Some(
            &json!({CONFIG_KEY: {"enabled": "no"}})
        )));
    }

    #[test]
    fn names_and_lkeys() {
        assert!(valid_name("UserEmbeddingPostSpamBot"));
        assert!(valid_name("inauthentic_detection_v45@1"));
        assert!(valid_name("bb1_tweetcreate_abuse_any_v15_user_512_mlp"));
        assert!(!valid_name(""));
        assert!(!valid_name("a/b"));
        assert!(!valid_name("a b"));
        assert!(!valid_name(&"x".repeat(257)));
        assert_eq!(lkey_for("H", None), "h/H");
        assert_eq!(lkey_for("H", Some("c@1")), "p/H/c@1");
    }


    fn put(head: &str, classifier: Option<&str>, off: bool) -> OffSwitchPutRequest {
        OffSwitchPutRequest {
            head: head.into(),
            classifier: classifier.map(Into::into),
            off,
            reason: "test reason".into(),
            auto_on_at_ms: None,
            updated_by: Some("operator@example.com".into()),
        }
    }

    #[tokio::test]
    async fn put_writes_record_and_updates_snapshot_at_once() {
        let (sw, store) = loaded(&[]).await;
        let scans_before = store.scans.load(Ordering::Relaxed);
        let out = sw
            .set(put("H", Some("c1"), true), "pv2", NOW)
            .await
            .unwrap();
        assert_eq!(out.before, None);
        assert_eq!(out.view.scope, "pair");
        assert!(out.view.active);
        assert_eq!(out.view.record.updated_by, "operator@example.com");
        assert_eq!(out.view.record.key_id, "pv2");
        assert_eq!(out.view.record.updated_at_ms, NOW);
        assert_eq!(store.scans.load(Ordering::Relaxed), scans_before);
        assert_eq!(sw.lookup("H", "c1", NOW).unwrap().scope, Scope::Pair);
        let rows = store.rows.lock().unwrap().clone();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "p/H/c1");
        let stored: OffSwitchRecord = serde_json::from_slice(&rows[0].1).unwrap();
        assert_eq!(stored, out.view.record);
    }

    #[tokio::test]
    async fn turning_on_writes_off_false_and_keeps_the_record() {
        let (sw, store) = loaded(&[record("H", None, true, None)]).await;
        let mut req = put("H", None, false);
        req.reason = String::new();
        let out = sw.set(req, "pv2", NOW).await.unwrap();
        assert!(out.before.unwrap().off);
        assert!(!out.view.record.off);
        assert!(!out.view.active);
        assert!(sw.lookup("H", "c", NOW).is_none());
        let rows = store.rows.lock().unwrap().clone();
        assert_eq!(rows.len(), 1, "never deleted");
        let stored: OffSwitchRecord = serde_json::from_slice(&rows[0].1).unwrap();
        assert!(!stored.off);
    }

    #[tokio::test]
    async fn turning_on_after_off_keeps_who_turned_it_off() {
        let mut off = record("H", None, true, None);
        off.updated_by = crate::head_guard::UPDATED_BY.into();
        off.reason = "action ceiling: 5 in 1h (guard v1)".into();
        let (sw, store) = loaded(&[off]).await;
        let out = sw.set(put("H", None, false), "pv2", NOW).await.unwrap();
        let want = PreviousOff {
            reason: "action ceiling: 5 in 1h (guard v1)".into(),
            updated_by: crate::head_guard::UPDATED_BY.into(),
            updated_at_ms: 1_000,
        };
        assert_eq!(out.view.record.previous_off.as_ref(), Some(&want));
        assert_eq!(
            out.view.record.updated_at_ms, NOW,
            "the on write stamps now"
        );
        let rows = store.rows.lock().unwrap().clone();
        let stored: OffSwitchRecord = serde_json::from_slice(&rows[0].1).unwrap();
        assert_eq!(stored.previous_off, Some(want), "stored and parsed back");
        assert_eq!(sw.get("h/H").unwrap().previous_off, stored.previous_off);
    }

                #[tokio::test]
    async fn previous_off_is_absent_unless_on_replaces_off() {
        let (sw, store) = loaded(&[]).await;
        let first_on = sw.set(put("H", None, false), "pv2", NOW).await.unwrap();
        assert_eq!(first_on.view.record.previous_off, None, "no record before");

        sw.set(put("H", None, true), "pv2", NOW).await.unwrap();
        let on = sw.set(put("H", None, false), "pv2", NOW + 1).await.unwrap();
        assert!(on.view.record.previous_off.is_some());
        let again = sw.set(put("H", None, false), "pv2", NOW + 2).await.unwrap();
        assert_eq!(again.view.record.previous_off, None, "on over on");

        let off = sw.set(put("H", None, true), "pv2", NOW + 3).await.unwrap();
        assert_eq!(off.view.record.previous_off, None, "off writes");
        let raw = String::from_utf8(store.rows.lock().unwrap()[0].1.clone()).unwrap();
        assert!(!raw.contains("previous_off"), "absent on the wire: {raw}");
    }

    #[tokio::test]
    async fn a_head_guard_trip_is_kept_when_turned_back_on() {
        let (sw, _) = loaded(&[]).await;
        let trip = OffSwitchPutRequest {
            reason: "action ceiling: 5 in 1h (guard v1)".into(),
            updated_by: Some(crate::head_guard::UPDATED_BY.into()),
            ..put("H", Some("c1"), true)
        };
        sw.set(trip, crate::head_guard::UPDATED_BY, NOW)
            .await
            .unwrap();
        let on = sw
            .set(put("H", Some("c1"), false), "pv2", NOW + 1)
            .await
            .unwrap();
        let prev = on.view.record.previous_off.unwrap();
        assert_eq!(prev.updated_by, crate::head_guard::UPDATED_BY);
        assert_eq!(prev.updated_at_ms, NOW);
    }

    #[tokio::test]
    async fn updated_by_defaults_to_key_id() {
        let (sw, _) = loaded(&[]).await;
        let mut req = put("H", None, true);
        req.updated_by = Some("  ".into());
        let out = sw.set(req, "slackbot", NOW).await.unwrap();
        assert_eq!(out.view.record.updated_by, "slackbot");
        let mut req = put("H2", None, true);
        req.updated_by = None;
        let out = sw.set(req, "slackbot", NOW).await.unwrap();
        assert_eq!(out.view.record.updated_by, "slackbot");
    }

    #[tokio::test]
    async fn put_validation_rejects_before_the_store() {
        let (sw, store) = loaded(&[]).await;
        let cases: Vec<(OffSwitchPutRequest, &str)> = vec![
            (put("", None, true), "head"),
            (put("a/b", None, true), "head"),
            (put("H", Some(""), true), "classifier"),
            (put("H", Some("x y"), true), "classifier"),
            (
                OffSwitchPutRequest {
                    reason: " ".into(),
                    ..put("H", None, true)
                },
                "reason is required",
            ),
            (
                OffSwitchPutRequest {
                    reason: "r".repeat(MAX_REASON_LEN + 1),
                    ..put("H", None, true)
                },
                "reason is longer",
            ),
            (
                OffSwitchPutRequest {
                    auto_on_at_ms: Some(NOW + 1),
                    ..put("H", None, false)
                },
                "only applies",
            ),
            (
                OffSwitchPutRequest {
                    auto_on_at_ms: Some(NOW),
                    ..put("H", None, true)
                },
                "future",
            ),
        ];
        for (req, want) in cases {
            match sw.set(req.clone(), "pv2", NOW).await {
                Err(SetError::Invalid(msg)) => {
                    assert!(msg.contains(want), "{req:?}: {msg}")
                }
                other => panic!("{req:?}: expected Invalid, got {other:?}"),
            }
        }
        assert!(store.rows.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn auto_on_is_stored_and_honoured() {
        let (sw, _) = loaded(&[]).await;
        let req = OffSwitchPutRequest {
            auto_on_at_ms: Some(NOW + 60_000),
            ..put("H", None, true)
        };
        let out = sw.set(req, "pv2", NOW).await.unwrap();
        assert_eq!(out.view.record.auto_on_at_ms, Some(NOW + 60_000));
        assert!(sw.lookup("H", "c", NOW + 59_999).is_some());
        assert!(sw.lookup("H", "c", NOW + 60_000).is_none());
    }

    #[tokio::test]
    async fn failed_put_changes_nothing() {
        let (sw, store) = loaded(&[]).await;
        store.fail_put.store(true, Ordering::Relaxed);
        let err = sw.set(put("H", None, true), "pv2", NOW).await.unwrap_err();
        assert!(matches!(err, SetError::Store(_)));
        assert!(sw.lookup("H", "c", NOW).is_none());
    }

    #[tokio::test]
    async fn local_write_survives_a_stale_scan() {
        let (sw, store) = loaded(&[]).await;
        sw.set(put("H", None, true), "pv2", NOW).await.unwrap();
        store.rows.lock().unwrap().clear();
        sw.refresh().await.unwrap();
        assert!(sw.lookup("H", "c", NOW).is_some());
        let mut newer = record("H", None, false, None);
        newer.updated_at_ms = NOW + 1;
        store.put_record(&newer);
        sw.refresh().await.unwrap();
        assert!(sw.lookup("H", "c", NOW).is_none());
    }

    #[tokio::test]
    async fn list_reads_the_store_and_falls_back_to_the_snapshot() {
        let (sw, store) = loaded(&[record("H", Some("c1"), true, None)]).await;
        store.put_record(&record("A", None, true, None));
        store.put_record(&record("H", None, false, None));
        let resp = sw.list(true, NOW).await;
        assert_eq!(resp.source, "store");
        assert!(resp.enabled);
        assert!(resp.error.is_none());
        assert!(resp.snapshot_age_ms >= 0);
        let keys: Vec<(String, Option<String>, bool)> = resp
            .switches
            .iter()
            .map(|v| (v.record.head.clone(), v.record.classifier.clone(), v.active))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("A".into(), None, true),
                ("H".into(), None, false),
                ("H".into(), Some("c1".into()), true),
            ]
        );
        store.fail_scan.store(true, Ordering::Relaxed);
        let resp = sw.list(false, NOW).await;
        assert_eq!(resp.source, "pod_snapshot");
        assert!(!resp.enabled);
        assert!(resp.error.is_some());
        assert_eq!(resp.switches.len(), 3);
    }


    fn with_seen(queue: usize) -> (Arc<OffSwitch>, Arc<MemStore>, Arc<MemStore>) {
        let store = Arc::new(MemStore::default());
        let seen = Arc::new(MemStore::default());
        let sw = OffSwitch::with_seen_queue(store.clone(), seen.clone(), queue)
            .with_scan_timeout(Duration::from_millis(200));
        (Arc::new(sw), store, seen)
    }

    fn seen_row(head: &str, classifier: &str, last_seen_ms: i64) -> (String, Vec<u8>) {
        let s = OffSwitchSeen {
            head: head.into(),
            classifier: classifier.into(),
            last_seen_ms,
        };
        (seen_lkey(head, classifier), serde_json::to_vec(&s).unwrap())
    }

    #[tokio::test]
    async fn seen_guard_records_each_pair_once_per_window() {
        let (sw, _, _) = with_seen(16);
        let t0 = Instant::now();
        assert_eq!(sw.record_seen_at("H", "c@1", NOW, t0), SeenOutcome::Queued);
        assert_eq!(
            sw.record_seen_at(
                "H",
                "c@1",
                NOW,
                t0 + SEEN_WRITE_EVERY - Duration::from_millis(1)
            ),
            SeenOutcome::Recent
        );
        assert_eq!(sw.record_seen_at("H", "c@2", NOW, t0), SeenOutcome::Queued);
        assert_eq!(sw.record_seen_at("H2", "c@1", NOW, t0), SeenOutcome::Queued);
        assert_eq!(
            sw.record_seen_at("H", "c@1", NOW, t0 + SEEN_WRITE_EVERY),
            SeenOutcome::Queued
        );
        for (h, c) in [("", "c"), ("H", ""), ("a/b", "c"), ("H", "c d")] {
            assert_eq!(
                sw.record_seen_at(h, c, NOW, t0),
                SeenOutcome::Invalid,
                "{h:?} {c:?}"
            );
        }
    }

    #[tokio::test]
    async fn full_seen_queue_drops_without_blocking_and_retries_later() {
        let (sw, _, _) = with_seen(1);
        let t0 = Instant::now();
        assert_eq!(sw.record_seen_at("H", "c1", NOW, t0), SeenOutcome::Queued);
        assert_eq!(sw.record_seen_at("H", "c2", NOW, t0), SeenOutcome::Dropped);
        assert_eq!(sw.record_seen_at("H", "c2", NOW, t0), SeenOutcome::Dropped);
        let mut rx = sw.seen_rx.lock().unwrap().take().unwrap();
        assert_eq!(rx.recv().await.unwrap().classifier, "c1");
        assert_eq!(sw.record_seen_at("H", "c2", NOW, t0), SeenOutcome::Queued);
        assert_eq!(rx.recv().await.unwrap().classifier, "c2");
    }

    async fn wait_for_rows(store: &MemStore, n: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while store.rows.lock().unwrap().len() < n {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("seen rows written");
    }

    #[tokio::test]
    async fn seen_writer_stores_the_pair_and_a_failed_write_retries() {
        let (sw, _, seen) = with_seen(16);
        assert!(sw.spawn_seen_writer().is_some());
        assert!(sw.spawn_seen_writer().is_none(), "one writer per pod");

        seen.fail_put.store(true, Ordering::Relaxed);
        assert_eq!(sw.record_seen("H", "c@1", NOW), SeenOutcome::Queued);
        tokio::time::timeout(Duration::from_secs(5), async {
            while sw.seen_guard.lock().unwrap().contains_key("H/c@1") {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("failed write releases the guard");

        seen.fail_put.store(false, Ordering::Relaxed);
        assert_eq!(sw.record_seen("H", "c@1", NOW + 1), SeenOutcome::Queued);
        wait_for_rows(&seen, 1).await;
        let rows = seen.rows.lock().unwrap().clone();
        assert_eq!(rows[0].0, "H/c@1");
        let stored: OffSwitchSeen = serde_json::from_slice(&rows[0].1).unwrap();
        assert_eq!(
            stored,
            OffSwitchSeen {
                head: "H".into(),
                classifier: "c@1".into(),
                last_seen_ms: NOW + 1,
            }
        );
    }

    #[tokio::test]
    async fn get_returns_switches_and_seen_pairs() {
        let (sw, store, seen) = with_seen(16);
        store.put_record(&record("H", None, true, None));
        {
            let mut rows = seen.rows.lock().unwrap();
            rows.push(seen_row("Z", "c@1", NOW - 5));
            rows.push(seen_row("H", "c@2", NOW - 1));
            rows.push(seen_row("H", "c@1", NOW - 2));
            rows.push(("H/bad".into(), b"not json".to_vec()));
        }
        let resp = sw.list(true, NOW).await;
        assert!(resp.seen_error.is_none());
        let pairs: Vec<(&str, &str, i64)> = resp
            .seen
            .iter()
            .map(|s| (s.head.as_str(), s.classifier.as_str(), s.last_seen_ms))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("H", "c@1", NOW - 2),
                ("H", "c@2", NOW - 1),
                ("Z", "c@1", NOW - 5)
            ]
        );

        let j = serde_json::to_value(&resp).unwrap();
        assert_eq!(j["switches"][0]["head"], "H");
        assert_eq!(j["switches"][0]["active"], true);
        assert_eq!(
            j["seen"][0],
            serde_json::json!({"head": "H", "classifier": "c@1", "last_seen_ms": NOW - 2})
        );
        assert!(j.get("seen_error").is_none());

        seen.fail_scan.store(true, Ordering::Relaxed);
        let resp = sw.list(true, NOW).await;
        assert_eq!(resp.source, "store");
        assert_eq!(resp.switches.len(), 1);
        assert!(resp.seen.is_empty());
        assert!(resp.seen_error.is_some());
    }
}
