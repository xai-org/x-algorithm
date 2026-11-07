use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};
use xai_manhattan::{LkeySelector, NativeManhattanClient, Tenant};

pub use xai_abuse_enforcement_types::action_control::{
    ACTION_CONTROL_RECORD_VERSION, ActionControlHistoryEntry, ActionControlHistoryResponse,
    ActionControlListResponse, ActionControlPutRequest, ActionControlRecord, ActionControlSeen,
    ActionControlSeenRow, ActionControlView, AllowedActions, ConfidenceEvidence, ConfidenceLevel,
    ConfidenceView, ControlEntity, ControlMode, CseLists, EvidenceOverride, LevelRow, Lifetime,
    PageAction, PodActionControlView, ROLLOUT_ALL_BP, ROWS_RECORD_VERSION, RowSetting, RowView,
    RowsRequest, SafetyChecksByEntity, SuspensionPolicy, UnreadableRecord, in_rollout,
    rollout_bucket,
};

mod levels;
pub use levels::level_lkey;

use crate::confidence::{ConfidenceConfig, RowOutcome};
use crate::decision::ActionSpec;
use crate::facts::EntityType;
use crate::gizmoduck_labels::{CLIENT_EVENT_PHOENIX_SKIP, SPAM_HIGH_RECALL};
use crate::growthbook::DynamicConfig;
use crate::metrics;
use crate::off_switch::valid_name;
use crate::safety_checks::AuthorSkips;

pub const MH_PKEY: &str = "action_control";
pub const SEEN_PKEY: &str = "action_control_seen";
pub const SEEN_ROWS_PKEY: &str = "action_control_seen_rows";
pub const SAVE_PKEY_PREFIX: &str = "action_control_save:";
const SAVE_HOLD_MS: i64 = 420_000;
const SAVE_TTL_SECS: u64 = 3_600;
const SAVE_BUDGET: Duration = Duration::from_secs(110);
pub const HISTORY_PKEY_PREFIX: &str = "action_control_history:";
pub const CONFIG_KEY: &str = "action_control";
pub const DRY_RUN_LABEL_PREFIX: &str = "dry_run__";
pub const ROLLOUT_SALT_ENV: &str = "ACTION_CONTROL_ROLLOUT_SALT";

pub const REFRESH_INTERVAL: Duration = Duration::from_secs(5);
pub const STALE_AFTER: Duration = Duration::from_secs(60);
const SCAN_TIMEOUT: Duration = Duration::from_secs(5);
pub const ROLLOUT_WAIT_MS: i64 = 120_000;
const _: () = assert!(
    ROLLOUT_WAIT_MS as u128
        >= STALE_AFTER.as_millis() + SCAN_TIMEOUT.as_millis() + REFRESH_INTERVAL.as_millis()
);
pub(crate) const SCAN_LIMIT: u32 = 10_000;
const FIRST_LOAD_BACKOFF: Duration = Duration::from_secs(1);
const FIRST_LOAD_LOG_EVERY: u64 = 12;
const ENSURE_LOADED_RETRY: Duration = Duration::from_secs(1);
const REFRESH_ERROR_LOG_EVERY: u64 = 60;
pub const RECORD_TTL_SECS: u64 = 10 * 365 * 86_400;
pub const HISTORY_TTL_SECS: u64 = 10 * 365 * 86_400;
pub const SEEN_TTL_SECS: u64 = 30 * 86_400;
pub const SEEN_WRITE_EVERY: Duration = Duration::from_secs(600);
const SEEN_QUEUE: usize = 256;
const SEEN_GUARD_MAX: usize = 10_000;
const LOCAL_WRITE_GRACE: Duration = Duration::from_secs(30);
const MAX_REASON_LEN: usize = 1_000;
const MAX_NAME_LEN: usize = 256;
pub const HISTORY_LIMIT_DEFAULT: usize = 50;
pub const HISTORY_LIMIT_MAX: usize = 500;
pub const CONFIRM_WORD: &str = "ENFORCE";
pub const PAGE_RULE_ID: &str = "page";
pub const STATUS_NO_ACTIONS: &str = "no_actions";
pub const FORCED_RECORDING_KEY: &str = "action_control_forced_recording";
pub const FOLLOWER_THRESHOLD_KEY: &str = "action_control_follower_threshold";
pub const SKIP_HIGH_CREDIBILITY_KEY: &str = "action_control_skip_high_credibility";

pub fn lkey_for(
    entity: ControlEntity,
    head: &str,
    classifier: Option<&str>,
    band: Option<&str>,
) -> String {
    let e = entity.as_str();
    match (classifier, band) {
        (None, None) => format!("{e}/h/{head}"),
        (Some(c), None) => format!("{e}/c/{head}/{c}"),
        (None, Some(b)) => format!("{e}/b/{head}/{b}"),
        (Some(c), Some(b)) => format!("{e}/cb/{head}/{c}/{b}"),
    }
}

pub fn record_lkey(rec: &ActionControlRecord) -> String {
    match rec.confidence {
        Some(level) => level_lkey(rec.entity_type, &rec.head, rec.classifier.as_deref(), level),
        None => lkey_for(
            rec.entity_type,
            &rec.head,
            rec.classifier.as_deref(),
            rec.band.as_deref(),
        ),
    }
}

fn record_lkey_of(req: &ActionControlPutRequest) -> String {
    match req.confidence {
        Some(row) => level_lkey(req.entity_type, &req.head, req.classifier.as_deref(), row),
        None => lkey_for(
            req.entity_type,
            &req.head,
            req.classifier.as_deref(),
            req.band.as_deref(),
        ),
    }
}

pub fn seen_lkey(seen: &ActionControlSeen) -> String {
    format!(
        "{}/{}/{}/{}",
        seen.entity_type.as_str(),
        seen.head,
        seen.classifier,
        seen.band
    )
}

pub fn strip_dry_run_prefix(label: &str) -> &str {
    label.strip_prefix(DRY_RUN_LABEL_PREFIX).unwrap_or(label)
}

pub fn has_dry_run_prefix(labels: &[String]) -> bool {
    labels.iter().any(|l| l.starts_with(DRY_RUN_LABEL_PREFIX))
}

pub fn entity_of(e: EntityType) -> ControlEntity {
    match e {
        EntityType::User => ControlEntity::User,
        EntityType::Post => ControlEntity::Post,
    }
}


const RISKY_HIGH_VIZ_REPLY: &str = "RiskyHighVizReply";

pub fn page_specs(actions: &[PageAction]) -> Vec<ActionSpec> {
    let rank = |a: &PageAction| match a {
        PageAction::SpamHighRecall { .. }
        | PageAction::PhoenixSkip { .. }
        | PageAction::PostSpamHighRecall { .. }
        | PageAction::HideReply { .. } => 0,
        PageAction::LivenessCheck {} => 1,
        PageAction::Suspend { .. } | PageAction::SuspendAuthor {} => 2,
    };
    let mut sorted: Vec<PageAction> = actions.to_vec();
    sorted.sort_by_key(rank);
    sorted
        .into_iter()
        .map(|a| match a {
            PageAction::SpamHighRecall { lifetime } => ActionSpec::AddLabelsV2 {
                labels: vec![SPAM_HIGH_RECALL.into()],
                ttl_msec: lifetime.ttl_msec(),
            },
            PageAction::PhoenixSkip { lifetime } => ActionSpec::AddLabelsV2 {
                labels: vec![CLIENT_EVENT_PHOENIX_SKIP.into()],
                ttl_msec: lifetime.ttl_msec(),
            },
            PageAction::PostSpamHighRecall { lifetime } => ActionSpec::AddPostLabelsV2 {
                labels: vec![SPAM_HIGH_RECALL.into()],
                ttl_msec: lifetime.ttl_msec(),
            },
            PageAction::HideReply { lifetime } => ActionSpec::AddPostLabelsV2 {
                labels: vec![RISKY_HIGH_VIZ_REPLY.into()],
                ttl_msec: lifetime.ttl_msec(),
            },
            PageAction::LivenessCheck {} => ActionSpec::SpamLivenessCheck,
            PageAction::Suspend { permanent, policy } => ActionSpec::SuspendUser {
                perm: permanent,
                policy: policy.as_str().into(),
            },
            PageAction::SuspendAuthor {} => ActionSpec::SuspendUser {
                perm: false,
                policy: SuspensionPolicy::PlatformManipulation.as_str().into(),
            },
        })
        .collect()
}

pub fn label_only(actions: &[PageAction]) -> Option<&'static str> {
    match actions {
        [PageAction::SpamHighRecall { .. }] => Some(SPAM_HIGH_RECALL),
        [PageAction::PhoenixSkip { .. }] => Some(CLIENT_EVENT_PHOENIX_SKIP),
        _ => None,
    }
}

fn ttl_days(ttl_msec: i64) -> Value {
    if ttl_msec % 86_400_000 == 0 {
        Value::from(ttl_msec / 86_400_000)
    } else {
        Value::from(ttl_msec as f64 / 86_400_000.0)
    }
}

pub fn action_specs_value(specs: &[ActionSpec]) -> Value {
    let mut out = Vec::new();
    for spec in specs {
        match spec {
            ActionSpec::SuspendUser { perm, policy } => out.push(serde_json::json!({
                "kind": "suspend", "permanent": perm, "policy": policy,
            })),
            ActionSpec::AddLabelsV2 { labels, ttl_msec }
            | ActionSpec::AddPostLabelsV2 { labels, ttl_msec } => {
                let kind = if matches!(spec, ActionSpec::AddLabelsV2 { .. }) {
                    "label"
                } else {
                    "post_label"
                };
                for label in labels {
                    let mut o = serde_json::json!({"kind": kind, "label": label});
                    match ttl_msec {
                        Some(t) => {
                            o["ttl_days"] = ttl_days(*t);
                            o["permanent"] = Value::Bool(false);
                        }
                        None => o["permanent"] = Value::Bool(true),
                    }
                    out.push(o);
                }
            }
            ActionSpec::SpamLivenessCheck => {
                out.push(serde_json::json!({"kind": "liveness_check"}))
            }
            ActionSpec::Arkose => out.push(serde_json::json!({"kind": "arkose"})),
            ActionSpec::Captcha => out.push(serde_json::json!({"kind": "captcha"})),
        }
    }
    Value::Array(out)
}

pub fn action_specs_json(specs: &[ActionSpec]) -> String {
    action_specs_value(specs).to_string()
}

fn page_class(actions: &[PageAction]) -> u8 {
    page_specs(actions)
        .iter()
        .map(ActionSpec::dedup_action_class)
        .max()
        .unwrap_or(crate::dedup_cache::CLASS_NONE)
}


#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ControlConfig {
                pub enabled: bool,
        pub ignore_store: bool,
                pub aliases: HashMap<String, String>,
                pub updated_by_keys: Option<HashSet<String>>,
        pub confidence: ConfidenceConfig,
}

impl ControlConfig {
                    pub fn from_config(config: Option<&Value>) -> Self {
        static LAST_BAD: Mutex<Option<String>> = Mutex::new(None);
        let mut out = Self::default();
        let Some(v) = config
            .and_then(|c| c.get(CONFIG_KEY))
            .filter(|v| !v.is_null())
        else {
            return out;
        };
        out.confidence = ConfidenceConfig::from_value(v.get(crate::confidence::CONFIG_KEY));
        let mut bad = Vec::new();
        match v.get("enabled") {
            None | Some(Value::Null) => {}
            Some(Value::Bool(b)) => out.enabled = *b,
            Some(other) => {
                out.enabled = false;
                bad.push(format!("enabled={other}"));
            }
        }
        match v.get("ignore_store") {
            None | Some(Value::Null) => {}
            Some(Value::Bool(b)) => out.ignore_store = *b,
            Some(other) => bad.push(format!("ignore_store={other}")),
        }
        if let Some(map) = v.get("aliases").and_then(Value::as_object) {
            out.aliases.extend(map.iter().filter_map(|(new, old)| {
                let old = old.as_str()?;
                (valid_name(new) && valid_name(old)).then(|| (new.clone(), old.to_owned()))
            }));
        }
        match v.get("updated_by_keys") {
            None | Some(Value::Null) => {}
            Some(keys) => {
                let parsed: Option<HashSet<String>> = keys.as_array().and_then(|arr| {
                    arr.iter()
                        .map(|k| {
                            k.as_str()
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .map(str::to_owned)
                        })
                        .collect()
                });
                if parsed.is_none() {
                    bad.push(format!("updated_by_keys={keys}"));
                }
                out.updated_by_keys = Some(parsed.unwrap_or_default());
            }
        }
        if !v.is_object() {
            out.enabled = false;
            bad.push(v.to_string());
        }
        if !bad.is_empty() {
            let bad = bad.join(", ");
            let mut last = LAST_BAD.lock().unwrap_or_else(|p| p.into_inner());
            if last.as_deref() != Some(bad.as_str()) {
                error!(
                    value = %bad,
                    "GrowthBook {CONFIG_KEY} is malformed; page-live scopes run as recording only until it is fixed"
                );
                *last = Some(bad);
            }
        }
        out
    }

    pub fn may_name_updater(&self, key_id: &str) -> bool {
        self.updated_by_keys
            .as_ref()
            .is_none_or(|keys| keys.contains(key_id))
    }
}


#[async_trait]
pub trait ControlStore: Send + Sync {
    async fn scan(&self, pkey: &str) -> anyhow::Result<Vec<(String, Vec<u8>)>>;
    async fn get(&self, pkey: &str, lkey: &str) -> anyhow::Result<Option<Vec<u8>>>;
    async fn put(
        &self,
        pkey: &str,
        lkey: &str,
        value: Vec<u8>,
        ttl_secs: u64,
    ) -> anyhow::Result<()>;
    async fn delete(&self, pkey: &str, lkey: &str) -> anyhow::Result<()>;
}

pub struct ManhattanControlStore {
    client: Arc<NativeManhattanClient>,
    tenant: Tenant,
}

impl ManhattanControlStore {
    pub fn new(client: Arc<NativeManhattanClient>, tenant: Tenant) -> Self {
        Self { client, tenant }
    }
}

#[async_trait]
impl ControlStore for ManhattanControlStore {
    async fn scan(&self, pkey: &str) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
        let items = self
            .client
            .scan(
                self.tenant.clone(),
                [pkey],
                LkeySelector::Range {
                    from: None,
                    to: None,
                },
                SCAN_LIMIT,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if items.len() >= SCAN_LIMIT as usize {
            warn!(
                pkey,
                limit = SCAN_LIMIT,
                "action-control scan hit its row limit; rows past it are not read"
            );
        }
        Ok(items
            .iter()
            .filter_map(|it| {
                let lkey =
                    String::from_utf8(it.lkey().into_byte_vecs().into_iter().next()?).ok()?;
                Some((lkey, it.value().as_bytes().to_vec()))
            })
            .collect())
    }

    async fn get(&self, pkey: &str, lkey: &str) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self
            .client
            .get(self.tenant.clone(), [pkey], [lkey])
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .map(|it| it.value().into_bytes()))
    }

    async fn put(
        &self,
        pkey: &str,
        lkey: &str,
        value: Vec<u8>,
        ttl_secs: u64,
    ) -> anyhow::Result<()> {
        self.client
            .put_with_ttl(
                self.tenant.clone(),
                [pkey],
                [lkey],
                value,
                crate::manhattan::ttl_nanos_from_secs(ttl_secs),
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    async fn delete(&self, pkey: &str, lkey: &str) -> anyhow::Result<()> {
        self.client
            .delete(self.tenant.clone(), [pkey], [lkey])
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}


#[derive(Default)]
struct Snapshot {
        records: HashMap<String, ActionControlRecord>,
            unreadable: HashMap<String, (String, u64)>,
    loaded_at: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matched {
    pub lkey: String,
    pub record: ActionControlRecord,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub matched: Option<Matched>,
        pub mode: ControlMode,
        pub page_actions: Vec<PageAction>,
        pub forced: Option<&'static str>,
            pub dedup_class: Option<u8>,
        pub producer_dry: bool,
            pub row: Option<RowOutcome>,
            pub by_row: bool,
            pub factory: bool,
                pub unlisted: bool,
            pub unknown: bool,
            pub author_skips: AuthorSkips,
}

impl Default for Resolved {
    fn default() -> Self {
        Self {
            matched: None,
            mode: ControlMode::Rules,
            page_actions: Vec::new(),
            forced: None,
            dedup_class: None,
            producer_dry: false,
            row: None,
            by_row: false,
            factory: false,
            unlisted: false,
            unknown: false,
            author_skips: AuthorSkips::default(),
        }
    }
}

impl Resolved {
        pub fn page_driven(&self) -> bool {
        self.factory
            || self.unlisted
            || (self.mode != ControlMode::Rules && (self.by_row || !self.page_actions.is_empty()))
    }

            pub fn rules_unread(&self) -> bool {
        self.factory || self.unlisted || self.unknown
    }

                            pub fn admit_unlisted(mut self) -> Option<Self> {
        if self.factory || self.mode == ControlMode::Rules {
            return None;
        }
        self.unlisted = true;
        self.dedup_class = Some(page_class(&self.page_actions));
        Some(self)
    }

    pub fn recording(&self) -> bool {
        self.mode == ControlMode::Recording
    }

                pub fn record_only(&self) -> bool {
        self.recording() || self.producer_dry
    }

        pub fn dry_run_source(&self, global: bool) -> Option<&'static str> {
        if global {
            Some("global")
        } else if self.recording() {
            Some("page")
        } else if self.producer_dry {
            Some("producer")
        } else {
            None
        }
    }

                pub fn incoming_dedup_class(&self, rules: impl FnOnce() -> u8) -> u8 {
        self.dedup_class.unwrap_or_else(rules)
    }

            pub fn info_keys(&self) -> Vec<(String, String)> {
        let mut keys = vec![(
            "action_control_mode".to_owned(),
            self.mode.as_str().to_owned(),
        )];
        if let Some(m) = &self.matched {
            keys.extend([
                (
                    "action_control_scope".to_owned(),
                    m.record.scope().to_owned(),
                ),
                (
                    "action_control_updated_by".to_owned(),
                    m.record.updated_by.clone(),
                ),
                (
                    "action_control_updated_at_ms".to_owned(),
                    m.record.updated_at_ms.to_string(),
                ),
                (
                    "action_control_version".to_owned(),
                    m.record.version.to_string(),
                ),
                ("action_control_lkey".to_owned(), m.lkey.clone()),
            ]);
        }
        if let Some(f) = self.forced {
            keys.push((FORCED_RECORDING_KEY.to_owned(), f.to_owned()));
        }
        if self.page_driven() {
            if let Some(n) = self.author_skips.follower_threshold {
                keys.push((FOLLOWER_THRESHOLD_KEY.to_owned(), n.to_string()));
            }
            if let Some(on) = self.author_skips.skip_high_credibility {
                keys.push((SKIP_HIGH_CREDIBILITY_KEY.to_owned(), on.to_string()));
            }
        }
        if let Some(r) = &self.row {
            keys.extend(r.info_keys(self.mode == ControlMode::Recording, self.forced));
        }
        keys
    }
}

pub struct ResolveInput<'a> {
    pub head: &'a str,
    pub classifier: &'a str,
    pub labels: &'a [String],
    pub entity: EntityType,
    pub factory: bool,
    pub producer_dry: bool,
        pub confidence: Option<crate::confidence::Claim>,
        pub author_id: i64,
}

enum Hit<'a> {
    Record(&'a str, &'a ActionControlRecord),
    Unreadable,
    Ambiguous,
}

impl Snapshot {
    fn at<'a>(&'a self, lkey: &str) -> Option<Hit<'a>> {
        if let Some((k, r)) = self.records.get_key_value(lkey) {
            return Some(Hit::Record(k, r));
        }
        self.unreadable.get(lkey).map(|_| Hit::Unreadable)
    }

            fn names_head(&self, entity: ControlEntity, names: &[&str]) -> bool {
        let e = entity.as_str();
        self.records.keys().chain(self.unreadable.keys()).any(|k| {
            let mut parts = k.split('/');
            parts.next() == Some(e)
                && parts.next().is_some()
                && parts.next().is_some_and(|h| names.contains(&h))
        })
    }

                    fn levels(
        entity: ControlEntity,
        head: &str,
        alias: Option<&str>,
        classifier: &str,
        bands: &[&str],
    ) -> Vec<Vec<Vec<String>>> {
        let names: Vec<&str> = std::iter::once(head).chain(alias).collect();
        let slot = |c: Option<&str>, b: Option<&str>| -> Vec<String> {
            names.iter().map(|n| lkey_for(entity, n, c, b)).collect()
        };
        let c = (!classifier.is_empty() && valid_name(classifier)).then_some(classifier);
        let mut levels = Vec::with_capacity(4);
        if let Some(c) = c {
            levels.push(bands.iter().map(|b| slot(Some(c), Some(b))).collect());
            levels.push(vec![slot(Some(c), None)]);
        }
        levels.push(bands.iter().map(|b| slot(None, Some(b))).collect());
        levels.push(vec![slot(None, None)]);
        levels
    }

    fn find(&self, levels: &[Vec<Vec<String>>]) -> Option<Hit<'_>> {
        for level in levels {
            let hits: Vec<Hit<'_>> = level
                .iter()
                .filter_map(|slot| slot.iter().find_map(|k| self.at(k)))
                .collect();
            match hits.len() {
                0 => continue,
                1 => return hits.into_iter().next(),
                _ => return Some(Hit::Ambiguous),
            }
        }
        None
    }

                    fn plain_decider(
        &self,
        entity: ControlEntity,
        head: &str,
        alias: Option<&str>,
        classifier: &str,
        bands: &[&str],
    ) -> Option<Hit<'_>> {
        self.find(&Self::levels(entity, head, alias, classifier, bands))
    }

                fn author_skips(&self, entity: ControlEntity, names: &[&str], factory: bool) -> AuthorSkips {
        if factory {
            return AuthorSkips::default();
        }
        names
            .iter()
            .find_map(|n| self.at(&lkey_for(entity, n, None, None)))
            .map_or_else(AuthorSkips::default, |hit| match hit {
                Hit::Record(_, r) => AuthorSkips::of(r),
                Hit::Unreadable | Hit::Ambiguous => AuthorSkips::default(),
            })
    }

            fn of_rows(rows: &[(String, Vec<u8>)], except: &[&str]) -> Snapshot {
        let mut snap = Snapshot::default();
        for (lkey, value) in rows.iter().filter(|(k, _)| !except.contains(&k.as_str())) {
            match parse_record(value) {
                Ok(rec) => {
                    snap.records.insert(record_lkey(&rec), rec);
                }
                Err(err) => {
                    snap.unreadable
                        .insert(lkey.clone(), (err, raw_version(value)));
                }
            }
        }
        for k in except {
            snap.records.remove(*k);
        }
        snap
    }
}

fn effective(
    record: &ActionControlRecord,
    entity: EntityType,
    cfg: &ControlConfig,
    stale: bool,
    safety_ok: bool,
) -> (ControlMode, Vec<PageAction>, Option<&'static str>) {
    let entity_ok = record.entity_type == entity_of(entity)
        && record
            .actions
            .iter()
            .all(|a| a.entity() == record.entity_type);
    match record.mode {
        ControlMode::Rules => (ControlMode::Rules, Vec::new(), None),
        ControlMode::Rows => (ControlMode::Recording, Vec::new(), Some("unreadable")),
        ControlMode::Recording if record.actions.is_empty() => {
            (ControlMode::Recording, Vec::new(), None)
        }
        ControlMode::Recording if !entity_ok => {
            (ControlMode::Recording, Vec::new(), Some("wrong_entity"))
        }
        ControlMode::Recording => (ControlMode::Recording, record.actions.clone(), None),
        ControlMode::Live if record.actions.is_empty() || !entity_ok => {
            (ControlMode::Recording, Vec::new(), Some("wrong_entity"))
        }
        ControlMode::Live => {
            let forced = if !cfg.enabled {
                Some("kill_switch")
            } else if stale {
                Some("stale_snapshot")
            } else if !safety_ok {
                Some("safety_checks_changed")
            } else {
                None
            };
            let mode = if forced.is_some() {
                ControlMode::Recording
            } else {
                ControlMode::Live
            };
            (mode, record.actions.clone(), forced)
        }
    }
}


#[derive(Debug, PartialEq)]
pub enum SetError {
        Invalid(&'static str, String),
            Conflict(Option<Box<ActionControlRecord>>, Option<u64>),
        LostRace(Option<Box<ActionControlRecord>>),
        Ignored,
        Store(String),
}

#[derive(Debug)]
pub struct SetOutcome {
    pub view: ActionControlView,
    pub before: Option<ActionControlRecord>,
    pub harsher_because: Vec<String>,
    pub history_written: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeenOutcome {
    Queued,
    Recent,
    Dropped,
    Invalid,
}

pub struct ActionControl {
    store: Arc<dyn ControlStore>,
    snapshot: RwLock<Arc<Snapshot>>,
    local_writes: Mutex<HashMap<String, (ActionControlRecord, Instant)>>,
    created_at: Instant,
    scan_timeout: Duration,
    save_budget: Duration,
    pub env: String,
    pub pod: String,
        rollout_salt: String,
    seen_guard: Mutex<HashMap<String, Instant>>,
    seen_tx: mpsc::Sender<SeenWrite>,
    seen_rx: Mutex<Option<mpsc::Receiver<SeenWrite>>>,
        #[cfg(test)]
    age_offset: Mutex<Duration>,
}

type SeenWrite = (&'static str, String, Vec<u8>);

fn seen_guard_key(pkey: &str, lkey: &str) -> String {
    if pkey == SEEN_PKEY {
        lkey.to_owned()
    } else {
        format!("{pkey}:{lkey}")
    }
}

pub(crate) fn new_nonce() -> u64 {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    nanos
        ^ ((std::process::id() as u64) << 32)
        ^ SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn label_days(a: &PageAction) -> Option<i64> {
    match a {
        PageAction::SpamHighRecall { lifetime }
        | PageAction::PhoenixSkip { lifetime }
        | PageAction::PostSpamHighRecall { lifetime }
        | PageAction::HideReply { lifetime } => {
            Some(lifetime.ttl_msec().map_or(i64::MAX, |t| t / 86_400_000))
        }
        _ => None,
    }
}

pub fn rollout_percent(bp: u32) -> String {
    let pct = format!("{}.{:02}", bp / 100, bp % 100);
    format!("{}%", pct.trim_end_matches('0').trim_end_matches('.'))
}

pub fn harsher_because(
    before: Option<&ActionControlRecord>,
    after: &ActionControlRecord,
    rules_specs: Option<&[Value]>,
) -> Vec<String> {
    let before_mode = before.map_or(ControlMode::Rules, |b| b.mode);
    let before_actions: &[PageAction] = before.map_or(&[], |b| &b.actions);
    let mut why = Vec::new();
    match (before_mode, after.mode) {
        (ControlMode::Live, ControlMode::Live) => {}
        (_, ControlMode::Live) => why.push("turns the scope live".to_owned()),
        (ControlMode::Recording, ControlMode::Rules) => {
            why.push("hands the scope back to the GrowthBook rules, which act".to_owned())
        }
        (ControlMode::Live, ControlMode::Rules) => {
            let page = action_specs_value(&page_specs(before_actions));
            let page = page.as_array().map(Vec::as_slice).unwrap_or_default();
            match rules_specs {
                Some(rules) if !rules.is_empty() && rules.iter().all(|r| covered_by(r, page)) => {}
                _ => why.push(
                    "hands the scope back to the GrowthBook rules, whose actions may be harsher"
                        .to_owned(),
                ),
            }
        }
        _ => {}
    }
    if after.mode == ControlMode::Live && before_mode == ControlMode::Live {
        let share = |r: Option<&ActionControlRecord>| {
            r.and_then(|r| r.rollout_bp)
                .map_or(ROLLOUT_ALL_BP, |bp| bp.min(ROLLOUT_ALL_BP))
        };
        let (was, now) = (share(before), share(Some(after)));
        if now > was {
            why.push(format!(
                "acts on more accounts (rollout {} to {})",
                rollout_percent(was),
                rollout_percent(now)
            ));
        }
        for a in &after.actions {
            let same_kind = before_actions.iter().find(|b| b.kind() == a.kind());
            match (a, same_kind) {
                (PageAction::Suspend { .. } | PageAction::SuspendAuthor {}, None) => {
                    why.push("adds suspend".into())
                }
                (
                    PageAction::Suspend { permanent, policy },
                    Some(PageAction::Suspend {
                        permanent: was_permanent,
                        policy: was_policy,
                    }),
                ) => {
                    if *permanent && !was_permanent {
                        why.push("makes the suspension permanent".into());
                    }
                    if policy != was_policy {
                        why.push(format!(
                            "changes the suspension policy ({} to {})",
                            was_policy.as_str(),
                            policy.as_str()
                        ));
                    }
                }
                (PageAction::LivenessCheck {}, None) => why.push("adds liveness check".into()),
                (_, None) => why.push("adds a label".into()),
                (_, Some(b)) => {
                    if let (Some(new), Some(old)) = (label_days(a), label_days(b))
                        && new > old
                    {
                        why.push("makes the label last longer".into());
                    }
                }
            }
        }
    }
    why
}

pub(crate) fn admits_unlisted_after(
    req: &ActionControlPutRequest,
    before: Option<&ActionControlRecord>,
    unknown: bool,
) -> bool {
    req.classifier.is_none()
        && req.band.is_none()
        && req.confidence.is_none()
        && req
            .admit_unlisted_labels
            .unwrap_or_else(|| unknown || before.is_some_and(|b| b.admit_unlisted_labels))
}

pub(crate) fn first_save_of_unknown_head(
    req: &ActionControlPutRequest,
    unknown: bool,
) -> Result<(), SetError> {
    if unknown && (req.classifier.is_some() || req.band.is_some()) {
        return Err(SetError::Invalid(
            "unknown_head_needs_head_record",
            "save the whole head first, so its decisions keep getting past the topic's label list"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn author_skips_after(
    req: &ActionControlPutRequest,
    before: Option<&ActionControlRecord>,
) -> AuthorSkips {
    if req.classifier.is_some() || req.band.is_some() || req.confidence.is_some() {
        return AuthorSkips::default();
    }
    let kept = before.map_or_else(AuthorSkips::default, AuthorSkips::of);
    AuthorSkips {
        follower_threshold: req.follower_threshold.unwrap_or(kept.follower_threshold),
        skip_high_credibility: req
            .skip_high_credibility
            .unwrap_or(kept.skip_high_credibility),
    }
}

pub(crate) fn author_skips_harsher(
    before: Option<&ActionControlRecord>,
    record: &ActionControlRecord,
) -> Vec<String> {
    let was = before.map_or_else(AuthorSkips::default, AuthorSkips::of);
    AuthorSkips::of(record).harsher_than(was)
}

pub(crate) fn admits_unlisted_harsher(
    before: Option<&ActionControlRecord>,
    after: &ActionControlRecord,
    unknown: bool,
) -> Option<&'static str> {
    (after.admit_unlisted_labels && !unknown && !before.is_some_and(|b| b.admit_unlisted_labels))
        .then_some("decides this head's decisions that the topic's label list drops")
}

fn covered_by(rule: &Value, page: &[Value]) -> bool {
    let field = |o: &Value, k: &str| o.get(k).cloned().unwrap_or(Value::Null);
    let permanent = |o: &Value| o.get("permanent").and_then(Value::as_bool) == Some(true);
    let days = |o: &Value| {
        if permanent(o) {
            f64::INFINITY
        } else {
            o.get("ttl_days")
                .and_then(Value::as_f64)
                .unwrap_or(f64::INFINITY)
        }
    };
    page.iter().any(|p| {
        if field(rule, "kind") != field(p, "kind") {
            return false;
        }
        match rule.get("kind").and_then(Value::as_str) {
            Some("suspend") => {
                field(rule, "policy") == field(p, "policy") && (!permanent(rule) || permanent(p))
            }
            Some("label" | "post_label") => {
                field(rule, "label") == field(p, "label") && days(rule) <= days(p)
            }
            Some("liveness_check") => true,
            _ => false,
        }
    })
}

impl ActionControl {
    pub fn new(store: Arc<dyn ControlStore>, env: String, pod: String) -> Self {
        let (seen_tx, seen_rx) = mpsc::channel(SEEN_QUEUE);
        Self {
            store,
            snapshot: RwLock::new(Arc::new(Snapshot::default())),
            local_writes: Mutex::new(HashMap::new()),
            created_at: Instant::now(),
            scan_timeout: SCAN_TIMEOUT,
            save_budget: SAVE_BUDGET,
            env,
            pod,
            rollout_salt: String::new(),
            seen_guard: Mutex::new(HashMap::new()),
            seen_tx,
            seen_rx: Mutex::new(Some(seen_rx)),
            #[cfg(test)]
            age_offset: Mutex::new(Duration::ZERO),
        }
    }

            pub fn with_rollout_salt(mut self, salt: String) -> Self {
        self.rollout_salt = salt;
        self
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn age_for_test(&self, by: Duration) {
        *self.age_offset.lock().unwrap() = by;
    }

    fn snapshot_age(&self, snap: &Snapshot) -> Option<Duration> {
        let age = snap.loaded_at?.elapsed();
        #[cfg(test)]
        let age = age + *self.age_offset.lock().unwrap();
        Some(age)
    }

            pub fn admits_unlisted(&self, entity: EntityType, head: &str) -> bool {
        let lkey = lkey_for(entity_of(entity), head, None, None);
        self.snapshot()
            .records
            .get(&lkey)
            .is_some_and(|r| r.admit_unlisted_labels)
    }

    pub fn is_loaded(&self) -> bool {
        self.snapshot().loaded_at.is_some()
    }

            pub fn ready(&self, cfg: &ControlConfig) -> bool {
        cfg.ignore_store || self.is_loaded()
    }

        pub fn snapshot_age_ms(&self) -> i64 {
        self.snapshot_age(&self.snapshot())
            .map_or(-1, |a| a.as_millis() as i64)
    }

        pub fn is_stale(&self) -> bool {
        self.snapshot_age(&self.snapshot())
            .is_none_or(|a| a > STALE_AFTER)
    }

            pub fn resolve(
        &self,
        input: &ResolveInput<'_>,
        cfg: &ControlConfig,
        safety_ok: impl FnOnce(EntityType) -> bool,
    ) -> Resolved {
        let mut out = Resolved {
            producer_dry: input.producer_dry,
            factory: input.factory,
            ..Default::default()
        };
        let claim = || {
            input
                .confidence
                .clone()
                .unwrap_or_else(|| crate::confidence::Claim {
                    row: LevelRow::Missing,
                    unknown: false,
                    evidence: None,
                    now_ms: crate::now_millis(),
                })
        };
        if cfg.ignore_store || input.head.is_empty() {
            if input.factory {
                out.mode = ControlMode::Recording;
                out.forced = cfg.ignore_store.then_some("ignore_store");
                out.row = Some(RowOutcome::new(&claim()));
                out.by_row = true;
                out.dedup_class = Some(page_class(&[]));
            }
            return out;
        }
        let snap = self.snapshot();
        if !input.factory {
            let names: Vec<&str> = std::iter::once(input.head)
                .chain(cfg.aliases.get(input.head).map(String::as_str))
                .collect();
            if !snap.names_head(entity_of(input.entity), &names) {
                if valid_name(input.head) {
                    out.mode = ControlMode::Recording;
                    out.row = Some(RowOutcome::new(&claim()));
                    out.by_row = true;
                    out.unknown = true;
                    out.dedup_class = Some(page_class(&[]));
                }
                return out;
            }
        }
        let mut safety_ok = Some(safety_ok);
        let mut safety_seen: Option<bool> = None;
        let mut safety = |e: EntityType| {
            *safety_seen.get_or_insert_with(|| safety_ok.take().is_none_or(|f| f(e)))
        };
        let mut bands: Vec<&str> = input
            .labels
            .iter()
            .map(|l| strip_dry_run_prefix(l))
            .filter(|l| valid_name(l))
            .collect();
        bands.sort_unstable();
        bands.dedup();
        let entity = entity_of(input.entity);
        let stale = self.snapshot_age(&snap).is_none_or(|a| a > STALE_AFTER);
        let by_row = levels::resolve_rows(
            &snap,
            &levels::RowContext {
                entity: input.entity,
                head: input.head,
                alias: cfg.aliases.get(input.head).map(String::as_str),
                classifier: input.classifier,
                factory: input.factory,
                stale,
                author_id: input.author_id,
                rollout_salt: &self.rollout_salt,
            },
            &claim(),
            cfg,
            &mut safety,
        );
        let hit = match by_row {
            Some(r) => {
                out = Resolved {
                    producer_dry: input.producer_dry,
                    factory: input.factory,
                    ..r
                };
                None
            }
            None => snap.plain_decider(
                entity,
                input.head,
                cfg.aliases.get(input.head).map(String::as_str),
                input.classifier,
                &bands,
            ),
        };
        match hit {
            None => {}
            Some(Hit::Ambiguous) => {
                out.mode = ControlMode::Recording;
                out.forced = Some("band_ambiguous");
            }
            Some(Hit::Unreadable) => {
                out.mode = ControlMode::Recording;
                out.forced = Some("unreadable");
            }
            Some(Hit::Record(lkey, record)) => {
                let (mode, actions, forced) = effective(
                    record,
                    input.entity,
                    cfg,
                    stale,
                    record.mode != ControlMode::Live || safety(input.entity),
                );
                out.mode = mode;
                out.page_actions = actions;
                out.forced = forced;
                out.matched = Some(Matched {
                    lkey: lkey.to_owned(),
                    record: record.clone(),
                });
            }
        }
        out.dedup_class = out.page_driven().then(|| page_class(&out.page_actions));
        let names: Vec<&str> = std::iter::once(input.head)
            .chain(cfg.aliases.get(input.head).map(String::as_str))
            .collect();
        out.author_skips = snap.author_skips(entity, &names, input.factory);
        if let Some(m) = &out.matched {
            metrics::ACTION_CONTROL_DECISIONS_TOTAL
                .with_label_values(&[
                    m.record.head.as_str(),
                    out.mode.as_str(),
                    out.forced.unwrap_or(""),
                ])
                .inc();
        }
        out
    }

            pub fn resolve_run(
        &self,
        input: &ResolveInput<'_>,
        cfg: &ControlConfig,
        safety_ok: impl FnOnce(EntityType) -> bool,
        actions: &[PageAction],
    ) -> Resolved {
        if cfg.ignore_store {
            return Resolved {
                mode: ControlMode::Recording,
                page_actions: actions.to_vec(),
                forced: Some("ignore_store"),
                dedup_class: Some(page_class(actions)),
                factory: input.factory,
                ..Default::default()
            };
        }
        let snap = self.snapshot();
        let stale = self.snapshot_age(&snap).is_none_or(|a| a > STALE_AFTER);
        let claim = input
            .confidence
            .clone()
            .unwrap_or_else(|| crate::confidence::Claim {
                row: LevelRow::Missing,
                unknown: false,
                evidence: None,
                now_ms: crate::now_millis(),
            });
        let mut safety_ok = Some(safety_ok);
        let mut safety = |e: EntityType| safety_ok.take().is_none_or(|f| f(e));
        levels::resolve_run(
            &snap,
            &levels::RowContext {
                entity: input.entity,
                head: input.head,
                alias: cfg.aliases.get(input.head).map(String::as_str),
                classifier: input.classifier,
                factory: input.factory,
                stale,
                author_id: input.author_id,
                rollout_salt: &self.rollout_salt,
            },
            &claim,
            cfg,
            &mut safety,
            actions,
        )
    }

            pub async fn stored_version(&self, lkey: &str) -> anyhow::Result<Option<u64>> {
        Ok(self
            .store
            .get(MH_PKEY, lkey)
            .await?
            .map(|b| raw_version(&b)))
    }


            pub async fn refresh(&self) -> anyhow::Result<usize> {
        let scanned = match tokio::time::timeout(self.scan_timeout, self.store.scan(MH_PKEY)).await
        {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!(
                "action-control scan timed out after {} ms",
                self.scan_timeout.as_millis()
            )),
        };
        let scanned = scanned.and_then(|rows| {
            if rows.len() >= SCAN_LIMIT as usize {
                Err(anyhow::anyhow!(
                    "action-control scan returned {} rows, its limit; the list may be incomplete",
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
                metrics::ACTION_CONTROL_REFRESH_TOTAL
                    .with_label_values(&["error"])
                    .inc();
                self.publish_metrics();
                return Err(e);
            }
        };
        let n = rows.len();
        let prev = self.snapshot();
        let mut records = HashMap::with_capacity(n);
        let mut unreadable = HashMap::new();
        for (lkey, value) in rows {
            match parse_record(&value) {
                Ok(rec) => {
                    records.insert(record_lkey(&rec), rec);
                }
                Err(err) => {
                    metrics::ACTION_CONTROL_UNREADABLE_TOTAL.inc();
                    warn!(lkey = %lkey, error = %err, "action-control record does not parse; keeping the previous copy if any");
                    match prev.records.get(&lkey) {
                        Some(old) => {
                            records.insert(lkey, old.clone());
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
                if records.get(key).is_none_or(|s| s.version < rec.version) {
                    unreadable.remove(key);
                    records.insert(key.clone(), rec.clone());
                }
            }
            self.publish(records, unreadable, true);
        }
        metrics::ACTION_CONTROL_REFRESH_TOTAL
            .with_label_values(&["ok"])
            .inc();
        Ok(n)
    }

    fn publish(
        &self,
        records: HashMap<String, ActionControlRecord>,
        unreadable: HashMap<String, (String, u64)>,
        loaded_now: bool,
    ) {
        {
            let mut slot = self.snapshot.write().unwrap_or_else(|p| p.into_inner());
            let loaded_at = if loaded_now {
                Some(Instant::now())
            } else {
                slot.loaded_at
            };
            *slot = Arc::new(Snapshot {
                records,
                unreadable,
                loaded_at,
            });
        }
        self.publish_metrics();
    }

    fn publish_metrics(&self) {
        let snap = self.snapshot();
        metrics::ACTION_CONTROL_LOADED.set(i64::from(snap.loaded_at.is_some()));
        for mode in [
            ControlMode::Rules,
            ControlMode::Recording,
            ControlMode::Live,
            ControlMode::Rows,
        ] {
            metrics::ACTION_CONTROL_ENTRIES
                .with_label_values(&[mode.as_str()])
                .set(snap.records.values().filter(|r| r.mode == mode).count() as i64);
        }
        let age = snap.loaded_at.unwrap_or(self.created_at).elapsed();
        metrics::ACTION_CONTROL_SNAPSHOT_AGE_SECONDS.set(age.as_secs() as i64);
    }

                pub async fn ensure_loaded(&self, cfg: &ControlConfig) {
        if !self.ready(cfg) {
            self.read_until_loaded(ENSURE_LOADED_RETRY, u64::MAX).await;
        }
    }

    async fn read_until_loaded(&self, retry: Duration, max_attempts: u64) -> bool {
        let mut attempt = 0u64;
        loop {
            attempt += 1;
            match self.refresh().await {
                Ok(_) => return true,
                Err(e) => {
                    if attempt == 1 || attempt.is_multiple_of(FIRST_LOAD_LOG_EVERY) {
                        error!(
                            error = %e,
                            attempts = attempt,
                            "action control: no successful read yet; holding this message until one succeeds"
                        );
                    }
                    if attempt >= max_attempts {
                        return false;
                    }
                    tokio::time::sleep(retry).await;
                }
            }
        }
    }

                            pub fn spawn_refresh(self: &Arc<Self>, dynamic_config: DynamicConfig) -> JoinHandle<()> {
        self.spawn_refresh_with(
            move || dynamic_config.action_control_config().ignore_store,
            REFRESH_INTERVAL,
            FIRST_LOAD_BACKOFF,
        )
    }

    fn spawn_refresh_with(
        self: &Arc<Self>,
        ignore_store: impl Fn() -> bool + Send + 'static,
        interval: Duration,
        first_backoff: Duration,
    ) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            let mut failures: u64 = 0;
            let mut backoff = first_backoff;
            loop {
                if ignore_store() {
                    metrics::ACTION_CONTROL_REFRESH_TOTAL
                        .with_label_values(&["ignored"])
                        .inc();
                    this.publish_metrics();
                    tokio::time::sleep(interval).await;
                    continue;
                }
                let first = !this.is_loaded();
                match this.refresh().await {
                    Ok(n) => {
                        if first {
                            info!(
                                records = n,
                                attempts = failures + 1,
                                since_start_ms = this.created_at.elapsed().as_millis() as u64,
                                "action control: first read ok"
                            );
                        } else if failures > 0 {
                            info!(failures, "action control: read ok again");
                        }
                        failures = 0;
                    }
                    Err(e) if first => {
                        failures += 1;
                        if failures == 1 || failures.is_multiple_of(FIRST_LOAD_LOG_EVERY) {
                            error!(
                                error = %e,
                                failures,
                                "action control: no successful read yet; this pod takes no Kafka work until one succeeds"
                            );
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        if failures == 1 || failures.is_multiple_of(REFRESH_ERROR_LOG_EVERY) {
                            warn!(
                                error = %e,
                                failures,
                                age_ms = this.snapshot_age_ms(),
                                "action control: read failed; keeping the last good list (live scopes run as recording only after 60 s)"
                            );
                        }
                    }
                }
                let wait = if this.is_loaded() {
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


            pub fn record_seen(&self, seen: ActionControlSeen) -> SeenOutcome {
        self.record_seen_at(seen, Instant::now())
    }

    fn record_seen_at(&self, seen: ActionControlSeen, now: Instant) -> SeenOutcome {
        let band_ok = seen.band == "-" || valid_name(&seen.band);
        let classifier_ok = seen.classifier == "-" || valid_name(&seen.classifier);
        if !valid_name(&seen.head) || !band_ok || !classifier_ok {
            metrics::ACTION_CONTROL_SEEN_TOTAL
                .with_label_values(&["invalid"])
                .inc();
            return SeenOutcome::Invalid;
        }
        let value = serde_json::to_vec(&seen).unwrap_or_default();
        self.queue_seen(SEEN_PKEY, seen_lkey(&seen), value, now)
    }

            pub fn record_seen_row(&self, seen: ActionControlSeenRow) -> SeenOutcome {
        self.record_seen_row_at(seen, Instant::now())
    }

    fn record_seen_row_at(&self, seen: ActionControlSeenRow, now: Instant) -> SeenOutcome {
        let classifier_ok = seen.classifier == "-" || valid_name(&seen.classifier);
        if !valid_name(&seen.head) || !classifier_ok {
            metrics::ACTION_CONTROL_SEEN_TOTAL
                .with_label_values(&["invalid"])
                .inc();
            return SeenOutcome::Invalid;
        }
        let lkey = format!(
            "{}/{}/{}/{}",
            seen.entity_type.as_str(),
            seen.head,
            seen.classifier,
            seen.row.as_str()
        );
        let value = serde_json::to_vec(&seen).unwrap_or_default();
        self.queue_seen(SEEN_ROWS_PKEY, lkey, value, now)
    }

    fn queue_seen(
        &self,
        pkey: &'static str,
        lkey: String,
        value: Vec<u8>,
        now: Instant,
    ) -> SeenOutcome {
        let key = seen_guard_key(pkey, &lkey);
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
                    metrics::ACTION_CONTROL_SEEN_TOTAL
                        .with_label_values(&["dropped"])
                        .inc();
                    return SeenOutcome::Dropped;
                }
            }
            guard.insert(key.clone(), now);
        }
        match self.seen_tx.try_send((pkey, lkey, value)) {
            Ok(()) => {
                metrics::ACTION_CONTROL_SEEN_TOTAL
                    .with_label_values(&["queued"])
                    .inc();
                SeenOutcome::Queued
            }
            Err(_) => {
                metrics::ACTION_CONTROL_SEEN_TOTAL
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
            while let Some((pkey, lkey, value)) = rx.recv().await {
                let key = seen_guard_key(pkey, &lkey);
                let result = match tokio::time::timeout(
                    this.scan_timeout,
                    this.store.put(pkey, &lkey, value, SEEN_TTL_SECS),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_) => Err(anyhow::anyhow!("seen write timed out")),
                };
                match result {
                    Ok(()) => {
                        failures = 0;
                        metrics::ACTION_CONTROL_SEEN_TOTAL
                            .with_label_values(&["written"])
                            .inc();
                    }
                    Err(e) => {
                        failures += 1;
                        metrics::MANHATTAN_ERRORS_TOTAL.inc();
                        metrics::ACTION_CONTROL_SEEN_TOTAL
                            .with_label_values(&["error"])
                            .inc();
                        this.forget_seen(&key, None);
                        if failures == 1 || failures.is_multiple_of(REFRESH_ERROR_LOG_EVERY) {
                            warn!(error = %e, failures, key, "action control: seen write failed; the next decision retries");
                        }
                    }
                }
            }
        }))
    }

    pub(crate) async fn scan_seen(&self) -> anyhow::Result<Vec<ActionControlSeen>> {
        let rows = match tokio::time::timeout(self.scan_timeout, self.store.scan(SEEN_PKEY)).await {
            Ok(r) => r?,
            Err(_) => anyhow::bail!("seen scan timed out"),
        };
        let mut seen: Vec<ActionControlSeen> = rows
            .iter()
            .filter_map(|(_, v)| serde_json::from_slice(v).ok())
            .collect();
        seen.sort_by(|a, b| {
            (&a.head, &a.classifier, &a.band, a.entity_type.as_str()).cmp(&(
                &b.head,
                &b.classifier,
                &b.band,
                b.entity_type.as_str(),
            ))
        });
        Ok(seen)
    }

    async fn scan_seen_rows(&self) -> anyhow::Result<Vec<ActionControlSeenRow>> {
        let rows =
            match tokio::time::timeout(self.scan_timeout, self.store.scan(SEEN_ROWS_PKEY)).await {
                Ok(r) => r?,
                Err(_) => anyhow::bail!("seen rows scan timed out"),
            };
        let mut seen: Vec<ActionControlSeenRow> = rows
            .iter()
            .filter_map(|(_, v)| serde_json::from_slice(v).ok())
            .collect();
        seen.sort_by(|a, b| {
            (&a.head, &a.classifier, a.row, a.entity_type.as_str()).cmp(&(
                &b.head,
                &b.classifier,
                b.row,
                b.entity_type.as_str(),
            ))
        });
        Ok(seen)
    }


    fn view(
        &self,
        lkey: &str,
        rec: &ActionControlRecord,
        cfg: &ControlConfig,
        stale: bool,
        safety: &SafetyChecksByEntity,
    ) -> ActionControlView {
        let entity = rec.entity_type;
        let safety_ok = match entity {
            ControlEntity::User => safety.user.ok,
            ControlEntity::Post => safety.post.ok,
        };
        let (mode, _, forced) = if rec.mode == ControlMode::Rows {
            (ControlMode::Rows, Vec::new(), None)
        } else {
            effective(
                rec,
                match entity {
                    ControlEntity::User => EntityType::User,
                    ControlEntity::Post => EntityType::Post,
                },
                cfg,
                stale,
                safety_ok,
            )
        };
        ActionControlView {
            record: rec.clone(),
            scope: rec.scope().to_owned(),
            lkey: lkey.to_owned(),
            effective_mode: mode,
            forced_recording_reason: forced.map(str::to_owned),
        }
    }

        pub async fn list(
        &self,
        cfg: &ControlConfig,
        safety: SafetyChecksByEntity,
    ) -> ActionControlListResponse {
        let (refreshed, seen, seen_rows) =
            tokio::join!(self.refresh(), self.scan_seen(), self.scan_seen_rows());
        let (source, error) = match refreshed {
            Ok(_) => ("store", None),
            Err(e) => ("pod_snapshot", Some(e.to_string())),
        };
        let (seen, mut seen_error) = match seen {
            Ok(s) => (s, None),
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                (Vec::new(), Some(e.to_string()))
            }
        };
        let seen_rows = match seen_rows {
            Ok(s) => s,
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                seen_error.get_or_insert_with(|| e.to_string());
                Vec::new()
            }
        };
        let snap = self.snapshot();
        let stale = self.is_stale();
        let mut records: Vec<ActionControlView> = snap
            .records
            .iter()
            .map(|(k, r)| self.view(k, r, cfg, stale, &safety))
            .collect();
        records.sort_by(|a, b| a.lkey.cmp(&b.lkey));
        let mut unreadable: Vec<UnreadableRecord> = snap
            .unreadable
            .iter()
            .map(|(k, (e, v))| UnreadableRecord {
                lkey: k.clone(),
                error: e.clone(),
                version: *v,
            })
            .collect();
        unreadable.sort_by(|a, b| a.lkey.cmp(&b.lkey));
        ActionControlListResponse {
            feature_version: ACTION_CONTROL_RECORD_VERSION,
            env: self.env.clone(),
            pod: self.pod.clone(),
            enabled: cfg.enabled,
            ignore_store: cfg.ignore_store,
            source: source.to_owned(),
            error,
            snapshot_age_ms: self.snapshot_age_ms(),
            stale,
            stale_after_ms: STALE_AFTER.as_millis() as i64,
            records,
            unreadable,
            seen,
            seen_error,
            seen_rows,
            safety_checks: safety,
            allowed_actions: AllowedActions {
                user: PageAction::menu(ControlEntity::User),
                post: PageAction::menu(ControlEntity::Post),
            },
            cse: CseLists::default(),
            confidence: Some(levels::view(cfg)),
            follower_skip: Some(crate::safety_checks::follower_skip_view()),
        }
    }

        pub fn pod_view(&self) -> PodActionControlView {
        let snap = self.snapshot();
        let mut page_managed: Vec<String> = snap
            .records
            .iter()
            .filter(|(_, r)| r.mode != ControlMode::Rules)
            .map(|(k, _)| k.clone())
            .chain(snap.unreadable.keys().cloned())
            .collect();
        page_managed.sort();
        PodActionControlView {
            loaded: snap.loaded_at.is_some(),
            snapshot_age_ms: self.snapshot_age_ms(),
            stale: self.is_stale(),
            records: snap.records.len(),
            page_managed,
        }
    }

            pub fn validate(
        req: &ActionControlPutRequest,
        seen: &[ActionControlSeen],
    ) -> Result<(), SetError> {
        let bad = |code: &'static str, msg: String| Err(SetError::Invalid(code, msg));
        if !valid_name(&req.head) {
            return bad(
                "invalid_head",
                format!("head must be 1-{MAX_NAME_LEN} characters of [A-Za-z0-9_.:@-]"),
            );
        }
        if req.classifier.as_deref().is_some_and(|c| !valid_name(c)) {
            return bad(
                "invalid_classifier",
                "classifier must be null or 1-256 characters of [A-Za-z0-9_.:@-]".into(),
            );
        }
        if req
            .band
            .as_deref()
            .is_some_and(|b| !valid_name(b) || b.starts_with(DRY_RUN_LABEL_PREFIX))
        {
            return bad(
                "invalid_band",
                "band must be null or a routing label (1-256 characters of [A-Za-z0-9_.:@-], no dry_run__ prefix)"
                    .into(),
            );
        }
        let reason = req.reason.trim();
        if reason.is_empty() {
            return bad(
                "reason_required",
                "say why in reason; it is kept in the history".into(),
            );
        }
        if reason.len() > MAX_REASON_LEN {
            return bad(
                "reason_too_long",
                format!("reason is longer than {MAX_REASON_LEN} bytes"),
            );
        }
        if req
            .updated_by
            .as_deref()
            .is_some_and(|u| u.len() > MAX_NAME_LEN)
        {
            return bad(
                "updated_by_too_long",
                format!("updated_by is longer than {MAX_NAME_LEN} bytes"),
            );
        }
        match req.mode {
            ControlMode::Rules if !req.actions.is_empty() => {
                return bad(
                    "rules_takes_no_actions",
                    "mode rules follows the GrowthBook rules; send actions: []".into(),
                );
            }
            ControlMode::Live if req.actions.is_empty() => {
                return bad(
                    "live_needs_actions",
                    "mode live needs at least one action".into(),
                );
            }
            _ => {}
        }
        let mut kinds = HashSet::new();
        for a in &req.actions {
            if !kinds.insert(a.kind()) {
                return bad(
                    "duplicate_action",
                    format!("{} appears more than once", a.kind()),
                );
            }
        }
        if let Some(a) = req
            .actions
            .iter()
            .find(|a| !PageAction::menu(a.entity()).contains(a))
        {
            return bad(
                "invalid_lifetime",
                format!(
                    "{} is not offered with that lifetime; a post label lasts 30 days or permanently",
                    a.kind()
                ),
            );
        }
        let entity = req.entity_type;
        if req.actions.iter().any(|a| a.entity() != entity) {
            return bad(
                "wrong_entity_actions",
                format!(
                    "a {} record takes only {} actions",
                    entity.as_str(),
                    entity.as_str()
                ),
            );
        }
        let head_rows: Vec<&ActionControlSeen> =
            seen.iter().filter(|s| s.head == req.head).collect();
        if !head_rows.is_empty() && head_rows.iter().all(|s| s.entity_type != entity) {
            return bad(
                "wrong_entity",
                format!(
                    "this head has only sent {} decisions; a {} record would never match",
                    head_rows[0].entity_type.as_str(),
                    entity.as_str()
                ),
            );
        }
        let admits = req.admit_unlisted_labels == Some(true);
        if admits && (req.classifier.is_some() || req.band.is_some() || req.confidence.is_some()) {
            return bad(
                "admit_unlisted_labels_scope",
                "only a head record (no classifier, band or row) can admit decisions whose labels the topic doesn't list"
                    .into(),
            );
        }
        let sets_skips = req.follower_threshold.is_some() || req.skip_high_credibility.is_some();
        if sets_skips
            && (req.classifier.is_some() || req.band.is_some() || req.confidence.is_some())
        {
            return bad(
                "author_skips_scope",
                "the follower threshold and the credibility skip are set on the head record (no classifier, band or row)"
                    .into(),
            );
        }
        let factory = head_rows
            .iter()
            .any(|s| s.factory && s.entity_type == entity);
        if factory && admits {
            return bad(
                "factory_admit_unlisted_labels",
                "a Model Factory head's decisions keep the topic's label list".into(),
            );
        }
        if factory && sets_skips {
            return bad(
                "factory_author_skips",
                "a Model Factory head keeps the default follower and credibility skips".into(),
            );
        }
        let skips = crate::safety_checks::follower_skip_view();
        let threshold = req.follower_threshold.flatten().map(u64::from);
        if threshold.is_some_and(|n| n > 0 && n < skips.today_threshold) {
            return bad(
                "follower_threshold_below_today",
                format!(
                    "a threshold below today's rule ({} or more followers) isn't supported yet; send 0 to turn the follower skip off, or {} or more",
                    skips.today_threshold, skips.today_threshold
                ),
            );
        }
        let ceiling = skips.ceiling;
        if threshold.is_some_and(|n| n >= ceiling) {
            return bad(
                "follower_threshold_too_high",
                format!(
                    "follower_threshold must be below {ceiling}, where accounts are always skipped"
                ),
            );
        }
        if !factory {
            return Ok(());
        }
        let settings = || {
            req.rows
                .iter()
                .flat_map(|r| LevelRow::ALL.into_iter().filter_map(|row| r.get(row)))
        };
        if req.rollout_bp.is_some() || settings().any(|s| s.rollout_bp.is_some()) {
            return bad(
                "factory_rollout",
                "a Model Factory head's rows act on every account; they take no rollout".into(),
            );
        }
        if req.mode == ControlMode::Rules || settings().any(|s| s.mode == ControlMode::Rules) {
            return bad(
                "factory_rules",
                "a Model Factory head is decided by its rows only, never by the GrowthBook rules; set the row to recording or live"
                    .into(),
            );
        }
        if req.band.is_some() {
            return bad(
                "factory_band",
                "a Model Factory head is decided by its rows only; it takes no band records".into(),
            );
        }
        let scope_acts = req.mode == ControlMode::Live || !req.actions.is_empty();
        if req.confidence.is_none() && req.mode != ControlMode::Rows && scope_acts {
            return bad(
                "factory_scope",
                "a Model Factory head acts only through its rows; set its rows".into(),
            );
        }
        if req.evidence.is_some() || settings().any(|s| s.evidence.is_some()) {
            return bad(
                "factory_evidence",
                "a Model Factory decision carries its own evidence from its policy; send no evidence on its rows"
                    .into(),
            );
        }
        Ok(())
    }

    async fn get_record(
        &self,
        lkey: &str,
    ) -> Result<Option<(Option<ActionControlRecord>, u64)>, SetError> {
        Ok(self
            .get_raw(lkey)
            .await?
            .map(|bytes| (parse_record(&bytes).ok(), raw_version(&bytes))))
    }

    async fn get_raw(&self, lkey: &str) -> Result<Option<Vec<u8>>, SetError> {
        match tokio::time::timeout(self.scan_timeout, self.store.get(MH_PKEY, lkey)).await {
            Ok(Ok(raw)) => Ok(raw),
            Ok(Err(e)) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                Err(SetError::Store(e.to_string()))
            }
            Err(_) => Err(SetError::Store("store read timed out".into())),
        }
    }

                    #[allow(clippy::too_many_arguments)]
    pub async fn set(
        &self,
        req: ActionControlPutRequest,
        who: &str,
        key_id: &str,
        now_ms: i64,
        cfg: &ControlConfig,
        safety: &SafetyChecksByEntity,
    ) -> Result<SetOutcome, SetError> {
        if cfg.ignore_store {
            return Err(SetError::Ignored);
        }
        let pkey = format!(
            "{SAVE_PKEY_PREFIX}{}",
            lkey_for(req.entity_type, &req.head, None, None)
        );
        let mine = format!("{now_ms:013}-{:020}", new_nonce());
        let deadline = Instant::now() + self.save_budget;
        self.save_marker(&pkey, &mine, now_ms, false).await?;
        let out = match self.other_save(&pkey, &mine, now_ms).await {
            Ok(false) => {
                self.set_unlocked(req, who, key_id, now_ms, cfg, safety, deadline)
                    .await
            }
            Ok(true) => {
                let lkey = record_lkey_of(&req);
                match self.get_record(&lkey).await {
                    Ok(stored) => Err(SetError::Conflict(
                        stored.as_ref().and_then(|(r, _)| r.clone()).map(Box::new),
                        stored.map(|(_, v)| v),
                    )),
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        };
        if let Err(e) = self.save_marker(&pkey, &mine, now_ms, true).await {
            warn!(
                ?e,
                pkey, "action control: save marker not cleared; it lapses in 7 minutes"
            );
        }
        out
    }

                async fn put_record(
        &self,
        lkey: &str,
        value: Vec<u8>,
        deadline: Instant,
    ) -> Result<(), SetError> {
        self.write_raw(lkey, Some(value), deadline).await
    }

        async fn write_raw(
        &self,
        lkey: &str,
        value: Option<Vec<u8>>,
        deadline: Instant,
    ) -> Result<(), SetError> {
        if Instant::now() + self.scan_timeout > deadline {
            return Err(SetError::Store(
                "the save took longer than its hold on the head; nothing more was written, reload and save again"
                    .into(),
            ));
        }
        let write = async {
            match value {
                Some(v) => self.store.put(MH_PKEY, lkey, v, RECORD_TTL_SECS).await,
                None => self.store.delete(MH_PKEY, lkey).await,
            }
        };
        match tokio::time::timeout(self.scan_timeout, write).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                Err(SetError::Store(e.to_string()))
            }
            Err(_) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                Err(SetError::Store("store write timed out".into()))
            }
        }
    }

    async fn save_marker(
        &self,
        pkey: &str,
        lkey: &str,
        now_ms: i64,
        done: bool,
    ) -> Result<(), SetError> {
        let value = serde_json::json!({"at_ms": now_ms, "done": done}).to_string();
        self.store
            .put(pkey, lkey, value.into_bytes(), SAVE_TTL_SECS)
            .await
            .map_err(|e| {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                SetError::Store(e.to_string())
            })
    }

            async fn other_save(&self, pkey: &str, mine: &str, now_ms: i64) -> Result<bool, SetError> {
        let rows = match tokio::time::timeout(self.scan_timeout, self.store.scan(pkey)).await {
            Ok(Ok(rows)) => rows,
            Ok(Err(e)) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                return Err(SetError::Store(e.to_string()));
            }
            Err(_) => return Err(SetError::Store("save marker scan timed out".into())),
        };
        Ok(rows.iter().filter(|(k, _)| k != mine).any(|(_, v)| {
            let v: Value = serde_json::from_slice(v).unwrap_or(Value::Null);
            let at = v.get("at_ms").and_then(Value::as_i64).unwrap_or(i64::MIN);
            v.get("done").and_then(Value::as_bool) != Some(true)
                && now_ms.saturating_sub(at).abs() < SAVE_HOLD_MS
        }))
    }

    #[allow(clippy::too_many_arguments)]
    async fn set_unlocked(
        &self,
        req: ActionControlPutRequest,
        who: &str,
        key_id: &str,
        now_ms: i64,
        cfg: &ControlConfig,
        safety: &SafetyChecksByEntity,
        deadline: Instant,
    ) -> Result<SetOutcome, SetError> {
        if cfg.ignore_store {
            return Err(SetError::Ignored);
        }
        let seen = self
            .scan_seen()
            .await
            .map_err(|e| SetError::Store(format!("the seen list could not be read: {e}")))?;
        Self::validate(&req, &seen)?;
        levels::validate_confidence(&req, now_ms)?;
        if req.mode == ControlMode::Rows || req.confidence.is_some() {
            return self
                .set_rows(req, who, key_id, now_ms, cfg, safety, &seen, deadline)
                .await;
        }
        let is_rows = |stored: Option<(Option<ActionControlRecord>, u64)>| {
            stored.is_some_and(|(r, _)| r.is_some_and(|r| r.mode == ControlMode::Rows))
        };
        let head_lkey = lkey_for(req.entity_type, &req.head, None, None);
        let classifier_lkey = req
            .classifier
            .as_deref()
            .map(|c| lkey_for(req.entity_type, &req.head, Some(c), None));
        if req.band.is_none() {
            let scope = classifier_lkey.as_deref().unwrap_or(&head_lkey);
            if is_rows(self.get_record(scope).await?) {
                return self
                    .set_rows(req, who, key_id, now_ms, cfg, safety, &seen, deadline)
                    .await;
            }
        } else if req.mode != ControlMode::Rules {
            let mut on_rows = is_rows(self.get_record(&head_lkey).await?);
            if let Some(c) = &classifier_lkey {
                on_rows |= is_rows(self.get_record(c).await?);
            }
            if on_rows {
                return Err(SetError::Invalid(
                    "scope_on_rows",
                    "this head is on rows; its band settings stay rules (set each row instead)"
                        .into(),
                ));
            }
        }
        let lkey = match req.confidence {
            Some(level) => level_lkey(req.entity_type, &req.head, req.classifier.as_deref(), level),
            None => lkey_for(
                req.entity_type,
                &req.head,
                req.classifier.as_deref(),
                req.band.as_deref(),
            ),
        };
        let stored = self.get_record(&lkey).await?;
        let stored_version = stored.as_ref().map(|(_, v)| *v);
        let before = stored.and_then(|(r, _)| r);
        if stored_version != req.expected_version {
            return Err(SetError::Conflict(before.map(Box::new), stored_version));
        }
        let alias = cfg.aliases.get(&req.head).map(String::as_str);
        let names: Vec<&str> = std::iter::once(req.head.as_str()).chain(alias).collect();
        let factory = seen
            .iter()
            .any(|s| s.factory && s.head == req.head && s.entity_type == req.entity_type);
        let rows = match stored_version {
            None => Some(self.scan_rows().await?),
            Some(_) => None,
        };
        let unknown = !factory
            && rows.as_deref().is_some_and(|rows| {
                !Snapshot::of_rows(rows, &[&lkey]).names_head(req.entity_type, &names)
            });
        first_save_of_unknown_head(&req, unknown)?;
        let mut rec = ActionControlRecord {
            v: ACTION_CONTROL_RECORD_VERSION,
            entity_type: req.entity_type,
            head: req.head.clone(),
            classifier: req.classifier.clone(),
            band: req.band.clone(),
            mode: req.mode,
            actions: req.actions.clone(),
            reason: req.reason.trim().to_owned(),
            updated_by: who.to_owned(),
            key_id: key_id.to_owned(),
            updated_at_ms: now_ms,
            version: stored_version.unwrap_or(0) + 1,
            write_nonce: new_nonce(),
            confidence: req.confidence,
            evidence: req.evidence.clone().map(Box::new),
            evidence_override: None,
            admit_unlisted_labels: admits_unlisted_after(&req, before.as_ref(), unknown),
            rollout_bp: None,
            follower_threshold: None,
            skip_high_credibility: None,
        };
        let skips = author_skips_after(&req, before.as_ref());
        rec.follower_threshold = skips.follower_threshold;
        rec.skip_high_credibility = skips.skip_high_credibility;
        rec.v = rec.schema_version();
        let recording = || ActionControlRecord {
            mode: ControlMode::Recording,
            actions: Vec::new(),
            ..rec.clone()
        };
        let priors: Vec<Option<ActionControlRecord>> = match (&before, stored_version) {
            (Some(b), _) => vec![Some(b.clone())],
            (None, Some(_)) => vec![Some(recording())],
            (None, None) => {
                let rows = rows.as_deref().unwrap_or_default();
                let snap = Snapshot::of_rows(rows, &[&lkey]);
                let uncovered = || (!snap.names_head(req.entity_type, &names)).then(recording);
                let exact = match req.band.as_deref() {
                    Some(band) => Some(
                        match snap.plain_decider(
                            req.entity_type,
                            &req.head,
                            alias,
                            req.classifier.as_deref().unwrap_or(""),
                            &[band],
                        ) {
                            Some(Hit::Record(_, r)) => Some(r.clone()),
                            Some(_) => Some(recording()),
                            None => uncovered(),
                        },
                    ),
                    None => None,
                };
                match exact {
                    Some(p) => vec![p],
                    None => {
                        let shadowed = Self::shadowed(&lkey, &req, cfg, rows);
                        let head_covers = shadowed.iter().flatten().any(|r| {
                            r.classifier.is_none() && r.band.is_none() && r.confidence.is_none()
                        });
                        let mut all: Vec<Option<ActionControlRecord>> = shadowed
                            .into_iter()
                            .map(|r| Some(r.unwrap_or_else(recording)))
                            .collect();
                        if !head_covers {
                            all.push(uncovered());
                        }
                        all
                    }
                }
            }
        };
        let scope_rows: Vec<&ActionControlSeen> = seen
            .iter()
            .filter(|s| s.head == req.head && s.entity_type == req.entity_type)
            .filter(|s| req.classifier.as_ref().is_none_or(|c| &s.classifier == c))
            .filter(|s| req.band.as_ref().is_none_or(|b| &s.band == b))
            .collect();
        let rules_specs: Option<Vec<Value>> = (!scope_rows.is_empty()).then(|| {
            scope_rows
                .iter()
                .flat_map(|s| s.action_specs.as_array().cloned().unwrap_or_default())
                .collect()
        });
        let stale = self.is_stale();
        let as_running = |p: ActionControlRecord| {
            if req.mode != ControlMode::Rules || p.mode != ControlMode::Live {
                return p;
            }
            let (entity, safety_ok) = match p.entity_type {
                ControlEntity::User => (EntityType::User, safety.user.ok),
                ControlEntity::Post => (EntityType::Post, safety.post.ok),
            };
            let (mode, _, _) = effective(&p, entity, cfg, stale, safety_ok);
            ActionControlRecord { mode, ..p }
        };
        let mut harsher: Vec<String> = Vec::new();
        for prior in &priors {
            let why = harsher_because(
                prior
                    .as_ref()
                    .map(levels::for_compare)
                    .map(&as_running)
                    .as_ref(),
                &levels::for_compare(&rec),
                rules_specs.as_deref(),
            );
            for w in why {
                if !harsher.contains(&w) {
                    harsher.push(w);
                }
            }
        }
        if let Some(why) = admits_unlisted_harsher(before.as_ref(), &rec, unknown) {
            harsher.push(why.to_owned());
        }
        for why in author_skips_harsher(before.as_ref(), &rec) {
            if !harsher.contains(&why) {
                harsher.push(why);
            }
        }
        if !harsher.is_empty() && req.confirm.as_deref() != Some(CONFIRM_WORD) {
            return Err(SetError::Invalid(
                "confirm_required",
                format!(
                    "this change enforces more ({}); send confirm: \"{CONFIRM_WORD}\"",
                    harsher.join("; ")
                ),
            ));
        }
        let value = serde_json::to_vec(&rec).map_err(|e| SetError::Store(e.to_string()))?;
        self.put_record(&lkey, value, deadline).await?;
        let entry = ActionControlHistoryEntry {
            at_ms: now_ms,
            version: rec.version,
            lkey: lkey.clone(),
            scope: rec.scope().to_owned(),
            updated_by: rec.updated_by.clone(),
            key_id: key_id.to_owned(),
            reason: rec.reason.clone(),
            env: self.env.clone(),
            before: before.clone(),
            after: rec.clone(),
        };
        let history_written = match serde_json::to_vec(&entry) {
            Ok(v) => self
                .store
                .put(
                    &format!("{HISTORY_PKEY_PREFIX}{}", rec.head),
                    &format!("{now_ms:013}-{:010}-{lkey}", rec.version),
                    v,
                    HISTORY_TTL_SECS,
                )
                .await
                .map_err(|e| warn!(error = %e, lkey, "action control: history write failed"))
                .is_ok(),
            Err(_) => false,
        };
        let readback = match self.get_record(&lkey).await {
            Ok(Some((Some(r), _))) => r,
            Ok(_) | Err(_) => {
                return Err(SetError::Store(
                    "the write landed but reading it back failed; pods pick up whatever the store holds within 5 s, so reload to see it"
                        .into(),
                ));
            }
        };
        if readback.write_nonce != rec.write_nonce {
            return Err(SetError::LostRace(Some(Box::new(readback))));
        }
        self.apply_local(lkey.clone(), rec.clone());
        info!(
            lkey,
            mode = rec.mode.as_str(),
            version = rec.version,
            updated_by = %rec.updated_by,
            key_id,
            "action control written"
        );
        let view = self.view(&lkey, &rec, cfg, self.is_stale(), safety);
        Ok(SetOutcome {
            view,
            before,
            harsher_because: harsher,
            history_written,
        })
    }

                        fn shadowed(
        own_lkey: &str,
        req: &ActionControlPutRequest,
        cfg: &ControlConfig,
        rows: &[(String, Vec<u8>)],
    ) -> Vec<Option<ActionControlRecord>> {
        let e = req.entity_type;
        let names: Vec<&str> = std::iter::once(req.head.as_str())
            .chain(cfg.aliases.get(&req.head).map(String::as_str))
            .collect();
        let (c, b) = (req.classifier.as_deref(), req.band.as_deref());
        let mut exact: Vec<String> = Vec::new();
        let mut band_prefixes: Vec<String> = Vec::new();
        for n in &names {
            exact.push(lkey_for(e, n, c, b));
            exact.push(lkey_for(e, n, None, None));
            match (c, b) {
                (Some(c), Some(b)) => {
                    exact.push(lkey_for(e, n, Some(c), None));
                    exact.push(lkey_for(e, n, None, Some(b)));
                }
                (Some(_), None) => band_prefixes.push(format!("{}/b/{n}/", e.as_str())),
                _ => {}
            }
        }
        rows.iter()
            .filter(|(k, _)| k != own_lkey)
            .filter(|(k, _)| exact.contains(k) || band_prefixes.iter().any(|p| k.starts_with(p)))
            .map(|(_, v)| parse_record(v).ok())
            .collect()
    }

        async fn scan_rows(&self) -> Result<Vec<(String, Vec<u8>)>, SetError> {
        match tokio::time::timeout(self.scan_timeout, self.store.scan(MH_PKEY)).await {
            Ok(Ok(rows)) if rows.len() < SCAN_LIMIT as usize => Ok(rows),
            Ok(Ok(_)) => Err(SetError::Store(
                "the record list is at its row limit".into(),
            )),
            Ok(Err(e)) => Err(SetError::Store(e.to_string())),
            Err(_) => Err(SetError::Store("record scan timed out".into())),
        }
    }

    fn apply_local(&self, lkey: String, rec: ActionControlRecord) {
        let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
        local.insert(lkey.clone(), (rec.clone(), Instant::now()));
        let snap = self.snapshot();
        let mut records = snap.records.clone();
        let mut unreadable = snap.unreadable.clone();
        unreadable.remove(&lkey);
        records.insert(lkey, rec);
        self.publish(records, unreadable, false);
    }

    fn forget_local(&self, lkey: &str) {
        let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
        local.remove(lkey);
        let snap = self.snapshot();
        let mut records = snap.records.clone();
        let mut unreadable = snap.unreadable.clone();
        records.remove(lkey);
        unreadable.remove(lkey);
        self.publish(records, unreadable, false);
    }

        pub async fn history(&self, head: &str, limit: usize) -> ActionControlHistoryResponse {
        let limit = limit.clamp(1, HISTORY_LIMIT_MAX);
        let pkey = format!("{HISTORY_PKEY_PREFIX}{head}");
        match tokio::time::timeout(self.scan_timeout, self.store.scan(&pkey)).await {
            Ok(Ok(rows)) => {
                let mut entries: Vec<ActionControlHistoryEntry> = rows
                    .iter()
                    .filter_map(|(_, v)| serde_json::from_slice(v).ok())
                    .collect();
                entries.sort_by(|a, b| {
                    (b.at_ms, b.version, &b.lkey).cmp(&(a.at_ms, a.version, &a.lkey))
                });
                entries.truncate(limit);
                ActionControlHistoryResponse {
                    head: head.to_owned(),
                    entries,
                    error: None,
                }
            }
            Ok(Err(e)) => ActionControlHistoryResponse {
                head: head.to_owned(),
                entries: Vec::new(),
                error: Some(e.to_string()),
            },
            Err(_) => ActionControlHistoryResponse {
                head: head.to_owned(),
                entries: Vec::new(),
                error: Some("history read timed out".into()),
            },
        }
    }
}

pub(crate) fn raw_version(bytes: &[u8]) -> u64 {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|v| v.get("version").and_then(Value::as_u64))
        .unwrap_or(0)
}

fn parse_record(bytes: &[u8]) -> Result<ActionControlRecord, String> {
    let rec = ActionControlRecord::parse_strict(bytes)?;
    if rec.v > ACTION_CONTROL_RECORD_VERSION {
        return Err(format!("record version {} is newer than this pod", rec.v));
    }
    if !valid_name(&rec.head)
        || rec.classifier.as_deref().is_some_and(|c| !valid_name(c))
        || rec.band.as_deref().is_some_and(|b| !valid_name(b))
    {
        return Err("invalid head, classifier or band".into());
    }
    if let Some(problem) = levels::record_problem(&rec, crate::now_millis()) {
        return Err(problem.into());
    }
    Ok(rec)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        #[derive(Default)]
    pub(crate) struct MemControlStore {
        pub rows: Mutex<HashMap<(String, String), Vec<u8>>>,
        pub fail_scan: AtomicBool,
        pub fail_put: AtomicBool,
        pub fail_get: AtomicBool,
        pub scans: AtomicUsize,
                                pub clobber_with: Mutex<Option<Vec<u8>>>,
                pub record_writes: AtomicUsize,
                                pub fail_record_writes: Mutex<HashMap<usize, bool>>,
    }

    impl MemControlStore {
                fn record_write_fails(&self, pkey: &str) -> Option<bool> {
            if pkey != MH_PKEY {
                return None;
            }
            let n = self.record_writes.fetch_add(1, Ordering::SeqCst);
            self.fail_record_writes.lock().unwrap().get(&n).copied()
        }

        pub(crate) fn put_record(&self, rec: &ActionControlRecord) {
            self.rows.lock().unwrap().insert(
                (MH_PKEY.into(), record_lkey(rec)),
                serde_json::to_vec(rec).unwrap(),
            );
        }

        pub(crate) fn put_raw(&self, lkey: &str, value: &[u8]) {
            self.rows
                .lock()
                .unwrap()
                .insert((MH_PKEY.into(), lkey.into()), value.to_vec());
        }

        pub(crate) fn pkey_rows(&self, pkey: &str) -> Vec<(String, Vec<u8>)> {
            let mut rows: Vec<(String, Vec<u8>)> = self
                .rows
                .lock()
                .unwrap()
                .iter()
                .filter(|((p, _), _)| p == pkey)
                .map(|((_, l), v)| (l.clone(), v.clone()))
                .collect();
            rows.sort();
            rows
        }
    }

    #[async_trait]
    impl ControlStore for MemControlStore {
        async fn scan(&self, pkey: &str) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
            self.scans.fetch_add(1, Ordering::Relaxed);
            if self.fail_scan.load(Ordering::Relaxed) {
                anyhow::bail!("scan failed (test)");
            }
            Ok(self.pkey_rows(pkey))
        }

        async fn get(&self, pkey: &str, lkey: &str) -> anyhow::Result<Option<Vec<u8>>> {
            if self.fail_get.load(Ordering::Relaxed) {
                anyhow::bail!("get failed (test)");
            }
            Ok(self
                .rows
                .lock()
                .unwrap()
                .get(&(pkey.to_owned(), lkey.to_owned()))
                .cloned())
        }

        async fn put(
            &self,
            pkey: &str,
            lkey: &str,
            value: Vec<u8>,
            _ttl: u64,
        ) -> anyhow::Result<()> {
            let fails = self.record_write_fails(pkey);
            if self.fail_put.load(Ordering::Relaxed) || fails == Some(false) {
                anyhow::bail!("put failed (test)");
            }
            let records = pkey == MH_PKEY || pkey == crate::head_guard::MH_PKEY;
            let value = match (records, self.clobber_with.lock().unwrap().clone()) {
                (true, Some(other)) => other,
                _ => value,
            };
            self.rows
                .lock()
                .unwrap()
                .insert((pkey.to_owned(), lkey.to_owned()), value);
            if fails == Some(true) {
                anyhow::bail!("put timed out after it landed (test)");
            }
            Ok(())
        }

        async fn delete(&self, pkey: &str, lkey: &str) -> anyhow::Result<()> {
            let fails = self.record_write_fails(pkey);
            if fails == Some(false) {
                anyhow::bail!("delete failed (test)");
            }
            self.rows
                .lock()
                .unwrap()
                .remove(&(pkey.to_owned(), lkey.to_owned()));
            if fails == Some(true) {
                anyhow::bail!("delete timed out after it landed (test)");
            }
            Ok(())
        }
    }

        pub(crate) fn record(
        head: &str,
        classifier: Option<&str>,
        band: Option<&str>,
        mode: ControlMode,
        actions: Vec<PageAction>,
    ) -> ActionControlRecord {
        ActionControlRecord {
            v: ACTION_CONTROL_RECORD_VERSION,
            entity_type: actions.first().map_or(ControlEntity::User, |a| a.entity()),
            head: head.into(),
            classifier: classifier.map(Into::into),
            band: band.map(Into::into),
            mode,
            actions,
            reason: format!("reason for {head}"),
            updated_by: "operator@example.com".into(),
            key_id: "pv2".into(),
            updated_at_ms: 1_000,
            version: 1,
            write_nonce: 7,
            confidence: None,
            evidence: None,
            evidence_override: None,
            admit_unlisted_labels: false,
            rollout_bp: None,
            follower_threshold: None,
            skip_high_credibility: None,
        }
    }

            pub(crate) fn parse_as_older_image(bytes: &[u8]) -> Result<ActionControlRecord, String> {
        let value: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if let Some(k) = value.as_object().and_then(|o| {
            o.keys()
                .find(|k| *k == "follower_threshold" || *k == "skip_high_credibility")
        }) {
            return Err(format!("unknown field `{k}`"));
        }
        let rec = parse_record(bytes)?;
        if rec.v > xai_abuse_enforcement_types::action_control::ROLLOUT_RECORD_VERSION {
            return Err(format!("record version {} is newer than this pod", rec.v));
        }
        Ok(rec)
    }

                        pub(crate) async fn refresh_as_older_image(ac: &ActionControl) {
        let rows = ac.store.scan(MH_PKEY).await.unwrap();
        let prev = ac.snapshot();
        let mut records = HashMap::new();
        let mut unreadable = HashMap::new();
        for (lkey, value) in rows {
            match parse_as_older_image(&value) {
                Ok(rec) => {
                    records.insert(record_lkey(&rec), rec);
                }
                Err(err) => match prev.records.get(&lkey).filter(|r| {
                    !r.sets_author_skips()
                        && r.v
                            <= xai_abuse_enforcement_types::action_control::ROLLOUT_RECORD_VERSION
                }) {
                    Some(old) => {
                        records.insert(lkey, old.clone());
                    }
                    None => {
                        unreadable.insert(lkey, (err, raw_version(&value)));
                    }
                },
            }
        }
        ac.publish(records, unreadable, true);
    }

    pub(crate) async fn loaded(
        records: &[ActionControlRecord],
    ) -> (Arc<ActionControl>, Arc<MemControlStore>) {
        let store = Arc::new(MemControlStore::default());
        for r in records {
            store.put_record(r);
        }
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            "pod-0".into(),
        ));
        ac.refresh().await.unwrap();
        (ac, store)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use xai_abuse_enforcement_types::action_control::{
        AUTHOR_SKIPS_RECORD_VERSION, SUSPENSION_OPTIONS_RECORD_VERSION,
    };

    const SUSPEND: PageAction = PageAction::SUSPEND;
    const LIVENESS: PageAction = PageAction::LivenessCheck {};
    const SHR7: PageAction = PageAction::SpamHighRecall {
        lifetime: Lifetime::Days7,
    };
    const SHR30: PageAction = PageAction::SpamHighRecall {
        lifetime: Lifetime::Days30,
    };

        fn on() -> ControlConfig {
        ControlConfig {
            enabled: true,
            ..Default::default()
        }
    }

    fn input<'a>(head: &'a str, classifier: &'a str, labels: &'a [String]) -> ResolveInput<'a> {
        ResolveInput {
            head,
            classifier,
            labels,
            entity: EntityType::User,
            factory: false,
            producer_dry: false,
            confidence: None,
            author_id: 1,
        }
    }

    fn baked_safety() -> SafetyChecksByEntity {
        let rules = crate::rules::RulesCache::new();
        SafetyChecksByEntity {
            user: crate::safety_checks::status(
                EntityType::User,
                &rules.resolve(EntityType::User, None),
            ),
            post: crate::safety_checks::status(
                EntityType::Post,
                &rules.resolve(EntityType::Post, None),
            ),
        }
    }

    fn labels(ls: &[&str]) -> Vec<String> {
        ls.iter().map(|l| (*l).to_owned()).collect()
    }

    fn resolve(ac: &ActionControl, head: &str, classifier: &str, ls: &[&str]) -> Resolved {
        let ls = labels(ls);
        ac.resolve(&input(head, classifier, &ls), &on(), |_| true)
    }

    fn live(
        head: &str,
        c: Option<&str>,
        b: Option<&str>,
        a: Vec<PageAction>,
    ) -> ActionControlRecord {
        record(head, c, b, ControlMode::Live, a)
    }

    #[tokio::test]
    async fn most_specific_record_wins() {
        let (ac, _) = loaded(&[
            live("H", None, None, vec![SUSPEND]),
            live("H", None, Some("band_a"), vec![LIVENESS]),
            live("H", Some("c1"), None, vec![SHR7]),
            live("H", Some("c1"), Some("band_a"), vec![SHR30]),
        ])
        .await;
        assert_eq!(
            resolve(&ac, "H", "c1", &["band_a"]).page_actions,
            vec![SHR30]
        );
        assert_eq!(
            resolve(&ac, "H", "c1", &["band_b"]).page_actions,
            vec![SHR7]
        );
        assert_eq!(
            resolve(&ac, "H", "c2", &["band_a"]).page_actions,
            vec![LIVENESS]
        );
        assert_eq!(resolve(&ac, "H", "c2", &[]).page_actions, vec![SUSPEND]);
        let r = resolve(&ac, "H", "c1", &["band_a"]);
        assert_eq!(r.matched.unwrap().lkey, "user/cb/H/c1/band_a");
        assert_eq!(r.mode, ControlMode::Live);
    }

            #[tokio::test]
    async fn a_head_no_record_names_records_only() {
        for store in [vec![], vec![live("Other", None, None, vec![SUSPEND])]] {
            let (ac, _) = loaded(&store).await;
            let r = resolve(&ac, "H", "c", &["x"]);
            assert_eq!(
                (r.mode, r.forced, r.matched.as_ref()),
                (ControlMode::Recording, None, None)
            );
            assert!(r.unknown && r.by_row && r.page_driven() && r.rules_unread());
            assert!(r.record_only() && r.page_actions.is_empty());
            assert_eq!(r.incoming_dedup_class(|| 2), page_class(&[]));
            assert_eq!(logged_because(&r).as_deref(), Some("no_selection"));
        }
        let (ac, _) = loaded(&[]).await;
        for head in ["", "not a head"] {
            assert_eq!(
                resolve(&ac, head, "c", &[]),
                Resolved::default(),
                "{head:?}"
            );
        }
        let ls = labels(&[]);
        let ignore = ControlConfig {
            ignore_store: true,
            ..on()
        };
        assert_eq!(
            ac.resolve(&input("H", "c", &ls), &ignore, |_| true),
            Resolved::default()
        );
        for rules in [
            record("H", None, None, ControlMode::Rules, vec![]),
            record("H", Some("other"), None, ControlMode::Rules, vec![]),
            record("H", None, Some("other"), ControlMode::Rules, vec![]),
        ] {
            let (ac, _) = loaded(&[rules]).await;
            let r = resolve(&ac, "H", "c", &["x"]);
            assert_eq!(r.mode, ControlMode::Rules);
            assert!(!r.unknown && !r.record_only() && !r.rules_unread());
            assert!(r.admit_unlisted().is_none());
        }
        let post = ActionControlRecord {
            entity_type: ControlEntity::Post,
            ..record("H", None, None, ControlMode::Rules, vec![])
        };
        let (ac, _) = loaded(&[post]).await;
        assert!(
            resolve(&ac, "H", "c", &[]).unknown,
            "another entity's record"
        );
        let (ac, store) = loaded(&[]).await;
        store.put_raw("user/c/H/c", b"{not json");
        ac.refresh().await.unwrap();
        assert!(
            !resolve(&ac, "H", "other", &[]).unknown,
            "an unreadable record names it"
        );
    }

    #[tokio::test]
    async fn band_matches_through_the_dry_run_prefix() {
        let (ac, _) = loaded(&[live("H", None, Some("band_a"), vec![LIVENESS])]).await;
        let r = resolve(&ac, "H", "c", &["dry_run__band_a"]);
        assert_eq!(r.page_actions, vec![LIVENESS]);
        for ls in [&["band_a", "band_a"][..], &["band_a", "dry_run__band_a"]] {
            let r = resolve(&ac, "H", "c", ls);
            assert_eq!(
                (r.mode, r.page_actions),
                (ControlMode::Live, vec![LIVENESS]),
                "{ls:?}"
            );
        }
    }

    #[tokio::test]
    async fn two_band_records_at_one_level_record_only() {
        let (ac, _) = loaded(&[
            live("H", None, Some("a"), vec![SUSPEND]),
            record("H", None, Some("b"), ControlMode::Rules, vec![]),
            live("H", None, None, vec![SUSPEND]),
        ])
        .await;
        let r = resolve(&ac, "H", "c", &["a", "b"]);
        assert_eq!(r.mode, ControlMode::Recording);
        assert_eq!(r.forced, Some("band_ambiguous"));
        assert!(r.page_actions.is_empty(), "records what the rules would do");
    }

    #[tokio::test]
    async fn unreadable_record_keeps_previous_copy_else_records_only() {
        let (ac, store) = loaded(&[live("H", None, None, vec![SUSPEND])]).await;
        store.put_raw("user/h/H", b"{not json");
        store.put_raw(
            "user/h/New",
            br#"{"v":1,"entity_type":"user","head":"New","mode":"live","version":1,"future":1}"#,
        );
        store.put_raw(
            "user/h/Newer",
            br#"{"v":9,"entity_type":"user","head":"Newer","mode":"rules","version":1}"#,
        );
        ac.refresh().await.unwrap();
        assert_eq!(
            resolve(&ac, "H", "c", &[]).mode,
            ControlMode::Live,
            "previous copy kept"
        );
        for head in ["New", "Newer"] {
            let r = resolve(&ac, head, "c", &[]);
            assert_eq!(r.mode, ControlMode::Recording, "{head}");
            assert_eq!(r.forced, Some("unreadable"));
        }
        assert_eq!(
            ac.pod_view().page_managed,
            vec!["user/h/H", "user/h/New", "user/h/Newer"]
        );
    }

    #[tokio::test]
    async fn kill_switch_records_live_scopes_and_keeps_recording() {
        let (ac, _) = loaded(&[
            live("L", None, None, vec![SUSPEND]),
            record("R", None, None, ControlMode::Recording, vec![]),
            record("G", None, None, ControlMode::Rules, vec![]),
        ])
        .await;
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };
        let ls = labels(&[]);
        let l = ac.resolve(&input("L", "c", &ls), &killed, |_| true);
        assert_eq!(
            (l.mode, l.forced),
            (ControlMode::Recording, Some("kill_switch"))
        );
        assert_eq!(l.page_actions, vec![SUSPEND]);
        let r = ac.resolve(&input("R", "c", &ls), &killed, |_| true);
        assert_eq!(r.mode, ControlMode::Recording);
        let g = ac.resolve(&input("G", "c", &ls), &killed, |_| true);
        assert_eq!(g.mode, ControlMode::Rules);
    }

    #[tokio::test]
    async fn stale_copy_records_live_scopes() {
        let (ac, store) = loaded(&[
            live("L", None, None, vec![SUSPEND]),
            record("R", None, None, ControlMode::Recording, vec![]),
        ])
        .await;
        store.fail_scan.store(true, Ordering::Relaxed);
        assert!(ac.refresh().await.is_err());
        assert_eq!(
            resolve(&ac, "L", "c", &[]).mode,
            ControlMode::Live,
            "fresh enough"
        );
        ac.age_for_test(STALE_AFTER + Duration::from_secs(1));
        let l = resolve(&ac, "L", "c", &[]);
        assert_eq!(
            (l.mode, l.forced),
            (ControlMode::Recording, Some("stale_snapshot"))
        );
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
        assert!(ac.is_stale());
    }

    #[tokio::test]
    async fn changed_safety_checks_record_live_scopes() {
        let (ac, _) = loaded(&[live("L", None, None, vec![SUSPEND])]).await;
        let ls = labels(&[]);
        let r = ac.resolve(&input("L", "c", &ls), &on(), |_| false);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("safety_checks_changed"))
        );
    }

    #[tokio::test]
    async fn a_child_safety_head_or_label_acts_like_any_other() {
        let (ac, _) = loaded(&[
            live("NearDupEmbeddingCseMatch", None, None, vec![SUSPEND]),
            live(
                "AnchorHead",
                None,
                Some("anchor_campaign_suspend_cse"),
                vec![SUSPEND],
            ),
        ])
        .await;
        let head = resolve(
            &ac,
            "NearDupEmbeddingCseMatch",
            "cse_cluster_neardup_v1",
            &[],
        );
        assert_eq!((head.mode, head.forced), (ControlMode::Live, None));
        let label = resolve(&ac, "AnchorHead", "c", &["anchor_campaign_suspend_cse"]);
        assert_eq!((label.mode, label.forced), (ControlMode::Live, None));
        let cfg = ControlConfig::from_config(Some(&serde_json::json!({
            CONFIG_KEY: {
                "enabled": true,
                "cse_heads": ["NearDupEmbeddingCseMatch"],
                "cse_labels": ["anchor_campaign_suspend_cse"],
            }
        })));
        assert_eq!(cfg, on());
        let ls = labels(&[]);
        assert_eq!(
            ac.resolve(&input("NearDupEmbeddingCseMatch", "c", &ls), &cfg, |_| true)
                .mode,
            ControlMode::Live
        );
    }

    fn factory_input<'a>(
        head: &'a str,
        classifier: &'a str,
        confidence: Option<crate::confidence::Claim>,
    ) -> ResolveInput<'a> {
        ResolveInput {
            factory: true,
            confidence,
            ..input(head, classifier, &[])
        }
    }

    fn logged_because(r: &Resolved) -> Option<String> {
        r.info_keys()
            .into_iter()
            .find(|(k, _)| k == "logged_because")
            .map(|(_, v)| v)
    }

        fn seen_factory(store: &MemControlStore, head: &str) {
        let seen = ActionControlSeen {
            head: head.into(),
            classifier: "c".into(),
            band: "-".into(),
            entity_type: ControlEntity::User,
            factory: true,
            last_seen_ms: 1,
            rule_id: String::new(),
            action_specs: serde_json::json!([]),
        };
        store.rows.lock().unwrap().insert(
            (SEEN_PKEY.into(), seen_lkey(&seen)),
            serde_json::to_vec(&seen).unwrap(),
        );
    }

        fn claims_without_the_rules(r: &Resolved) {
        let want = page_class(&r.page_actions);
        assert_eq!(r.dedup_class, Some(want));
        assert_eq!(
            r.incoming_dedup_class(|| panic!("rules asked for a factory decision")),
            want
        );
    }

                    #[tokio::test]
    async fn a_factory_head_records_only_until_its_row_is_set() {
        let (empty, _) = loaded(&[]).await;
        let r = empty.resolve(&factory_input("F", "c", None), &on(), |_| true);
        assert_eq!(r.mode, ControlMode::Recording);
        assert_eq!((r.forced, r.matched.is_none()), (None, true));
        assert!(r.factory && r.by_row && r.page_driven());
        assert_eq!(r.row.as_ref().map(|o| o.row), Some(LevelRow::Missing));
        assert_eq!(logged_because(&r).as_deref(), Some("no_selection"));
        claims_without_the_rules(&r);

        let (ac, _) = loaded(&[
            live("F", None, None, vec![SUSPEND]),
            live("F", Some("c"), None, vec![SUSPEND]),
            live("F", None, Some("model_factory"), vec![SUSPEND]),
        ])
        .await;
        let ls = labels(&["model_factory"]);
        let r = ac.resolve(
            &ResolveInput {
                labels: &ls,
                ..factory_input("F", "c", None)
            },
            &on(),
            |_| true,
        );
        assert_eq!((r.mode, r.forced), (ControlMode::Recording, None));
        assert!(r.page_actions.is_empty() && r.matched.is_none());
        claims_without_the_rules(&r);
        assert_eq!(resolve(&ac, "F", "c", &[]).mode, ControlMode::Live);

        let ignore = ControlConfig {
            ignore_store: true,
            ..on()
        };
        let r = ac.resolve(&factory_input("F", "c", None), &ignore, |_| true);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("ignore_store"))
        );
        assert_eq!(logged_because(&r).as_deref(), Some("ignore_store"));
        claims_without_the_rules(&r);
        let r = ac.resolve(&factory_input("", "c", None), &on(), |_| true);
        assert_eq!((r.mode, r.forced), (ControlMode::Recording, None));
        assert_eq!(logged_because(&r).as_deref(), Some("no_selection"));
        assert!(r.page_driven());
        claims_without_the_rules(&r);
    }

            #[tokio::test]
    async fn a_factory_head_lands_on_its_implicit_rows() {
        let now = crate::now_millis();
        let ev = ConfidenceEvidence {
            read_at_ms: now - 86_400_000,
            ..read(0.96, 59, 0)
        };
        let (ac, _) = loaded(&[
            row_rec("F", LevelRow::High, ControlMode::Live, vec![SUSPEND]),
            row_rec("F", LevelRow::Missing, ControlMode::Live, vec![LIVENESS]),
            row_rec("F", LevelRow::Low, ControlMode::Rules, vec![]),
        ])
        .await;
        let at = |row: LevelRow| {
            let claim = crate::confidence::Claim {
                row,
                unknown: false,
                evidence: Some(ev.clone()),
                now_ms: now,
            };
            ac.resolve(&factory_input("F", "c", Some(claim)), &on(), |_| true)
        };
        let high = at(LevelRow::High);
        assert_eq!((high.mode, high.forced), (ControlMode::Live, None));
        assert_eq!(high.page_actions, vec![SUSPEND]);
        claims_without_the_rules(&high);
        assert_eq!(high.matched.unwrap().lkey, "user/l/F/high");
        let missing = at(LevelRow::Missing);
        assert_eq!(missing.mode, ControlMode::Live);
        assert_eq!(missing.page_actions, vec![LIVENESS]);
        let mid = at(LevelRow::Mid);
        assert_eq!((mid.mode, mid.forced), (ControlMode::Recording, None));
        assert!(mid.page_actions.is_empty());
        assert_eq!(logged_because(&mid).as_deref(), Some("no_selection"));
        let low = at(LevelRow::Low);
        assert_eq!(
            (low.mode, low.forced),
            (ControlMode::Recording, Some("factory_rules_row"))
        );
        assert!(low.page_driven() && low.page_actions.is_empty());
        assert_eq!(logged_because(&low).as_deref(), Some("factory_rules_row"));
        claims_without_the_rules(&low);
    }

                #[tokio::test]
    async fn a_factory_classifier_on_rows_wins_over_the_head_rows() {
        let now = crate::now_millis();
        let (ac, store) = loaded(&[
            row_rec("F", LevelRow::Missing, ControlMode::Live, vec![SUSPEND]),
            row_rec("F", LevelRow::High, ControlMode::Live, vec![SUSPEND]),
            record("F", Some("c1"), None, ControlMode::Rows, vec![]),
            ActionControlRecord {
                confidence: Some(LevelRow::Missing),
                ..record("F", Some("c1"), None, ControlMode::Live, vec![LIVENESS])
            },
        ])
        .await;
        store.put_raw("user/c/F/bad", b"{not a record");
        ac.refresh().await.unwrap();
        let at = |classifier: &str, row: LevelRow| {
            let claim = crate::confidence::Claim {
                row,
                unknown: false,
                evidence: Some(ConfidenceEvidence {
                    read_at_ms: now - 86_400_000,
                    ..read(0.96, 59, 0)
                }),
                now_ms: now,
            };
            ac.resolve(&factory_input("F", classifier, Some(claim)), &on(), |_| {
                true
            })
        };
        let c1 = at("c1", LevelRow::Missing);
        assert_eq!(c1.page_actions, vec![LIVENESS]);
        assert_eq!(c1.matched.unwrap().lkey, "user/cl/F/c1/missing");
        let c1_high = at("c1", LevelRow::High);
        assert_eq!(
            (c1_high.mode, c1_high.forced),
            (ControlMode::Recording, None)
        );
        assert!(c1_high.page_actions.is_empty());
        assert_eq!(logged_because(&c1_high).as_deref(), Some("no_selection"));
        assert_eq!(c1_high.matched.unwrap().lkey, "user/c/F/c1");
        let c2 = at("c2", LevelRow::Missing);
        assert_eq!(c2.page_actions, vec![SUSPEND]);
        assert_eq!(c2.matched.unwrap().lkey, "user/l/F/missing");
        for row in [LevelRow::Missing, LevelRow::High] {
            let bad = at("bad", row);
            assert_eq!(
                (bad.mode, bad.forced),
                (ControlMode::Recording, Some("unreadable"))
            );
            assert!(bad.page_actions.is_empty() && bad.matched.is_none());
            claims_without_the_rules(&bad);
        }
    }

            #[tokio::test]
    async fn a_factory_head_reads_and_writes_its_own_name_rows() {
        let (ac, store) = loaded(&[
            record("Old", None, None, ControlMode::Rows, vec![]),
            row_rec("Old", LevelRow::Missing, ControlMode::Live, vec![SUSPEND]),
        ])
        .await;
        seen_factory(&store, "New");
        let mut cfg = on();
        cfg.aliases.insert("New".into(), "Old".into());
        let decide = |factory: bool| {
            let mut i = factory_input("New", "c", None);
            i.factory = factory;
            ac.resolve(&i, &cfg, |_| true)
        };
        assert_eq!(decide(false).page_actions, vec![SUSPEND], "alias applies");
        let r = decide(true);
        assert_eq!((r.mode, r.forced), (ControlMode::Recording, None));
        assert!(r.page_actions.is_empty() && r.matched.is_none());
        put_with(
            &ac,
            &cfg,
            row_put("New", LevelRow::Missing, ControlMode::Live, vec![LIVENESS]),
        )
        .await
        .unwrap();
        let r = decide(true);
        assert_eq!(
            (r.mode, r.page_actions.clone()),
            (ControlMode::Live, vec![LIVENESS])
        );
        assert_eq!(r.matched.unwrap().lkey, "user/l/New/missing");
    }

                        #[tokio::test]
    async fn a_level_row_acts_whatever_its_evidence_and_records_what_it_supports() {
        use ConfidenceLevel::High;
        const DAY: i64 = 86_400_000;
        let now = crate::now_millis();
        let (ac, _) = loaded(&[
            record("F", None, None, ControlMode::Rows, vec![]),
            ActionControlRecord {
                evidence: Some(Box::new(ConfidenceEvidence {
                    read_at_ms: now - DAY,
                    ..read(0.96, 59, 0)
                })),
                ..row_rec("F", LevelRow::High, ControlMode::Live, vec![SUSPEND])
            },
            record("G", None, None, ControlMode::Rows, vec![]),
            row_rec("G", LevelRow::High, ControlMode::Live, vec![SUSPEND]),
        ])
        .await;
        let decide_with = |head: &str,
                           evidence: Option<ConfidenceEvidence>,
                           factory: bool,
                           cfg: &ControlConfig| {
            let claim = crate::confidence::Claim {
                row: LevelRow::High,
                unknown: false,
                evidence,
                now_ms: now,
            };
            let mut i = factory_input(head, "c", Some(claim));
            i.factory = factory;
            let r = ac.resolve(&i, cfg, |_| true);
            assert_eq!(
                (r.mode, r.forced, r.page_actions.clone()),
                (ControlMode::Live, None, vec![SUSPEND]),
                "{head} factory={factory}"
            );
            r.row.unwrap().evidence_supports.unwrap()
        };
        let decide = |head: &str, evidence: Option<ConfidenceEvidence>, factory: bool| {
            decide_with(head, evidence, factory, &on())
        };
        let aged = |days: i64| ConfidenceEvidence {
            read_at_ms: now - days * DAY,
            ..read(0.96, 59, 0)
        };
        assert_eq!(decide("F", Some(aged(364)), true), Some(High));
        assert_eq!(decide("F", Some(aged(366)), true), None, "too old");
        assert_eq!(
            decide("F", None, true),
            None,
            "the row's own evidence doesn't count for a factory head"
        );
        let weak = ConfidenceEvidence {
            read_at_ms: now - DAY,
            ..read(0.40, 10, 6)
        };
        assert_eq!(decide("F", Some(weak.clone()), true), None, "weak");
        assert_eq!(decide("F", None, false), Some(High));
        assert_eq!(decide("G", Some(aged(29)), false), Some(High));
        assert_eq!(decide("G", Some(aged(31)), false), None, "expired");
        assert_eq!(decide("G", Some(weak), false), None, "weak");
        assert_eq!(decide("G", None, false), None, "no evidence");
        let short = ControlConfig {
            confidence: crate::confidence::ConfidenceConfig::from_value(Some(
                &serde_json::json!({"factory_evidence_max_age_days": 30}),
            )),
            ..on()
        };
        assert_eq!(decide_with("F", Some(aged(31)), true, &short), None);
        let unreadable = ControlConfig {
            confidence: crate::confidence::ConfidenceConfig::from_value(Some(
                &serde_json::json!({"label_map": {}}),
            )),
            ..on()
        };
        assert_eq!(
            decide_with("G", Some(aged(1)), false, &unreadable),
            None,
            "an unreadable confidence config makes no evidence current, and the row still acts"
        );
    }

    #[tokio::test]
    async fn a_shared_head_name_keeps_user_and_post_records_apart() {
        let head = "SharedSpamBot";
        let post_input = |ls: &'static [String]| ResolveInput {
            entity: EntityType::Post,
            ..input(head, "c", ls)
        };
        let (ac, _) = loaded(&[live(head, None, None, vec![SUSPEND])]).await;
        let user = resolve(&ac, head, "c", &[]);
        assert_eq!(
            (user.mode, user.page_actions),
            (ControlMode::Live, vec![SUSPEND])
        );
        let post = ac.resolve(&post_input(&[]), &on(), |_| true);
        assert!(
            post.unknown && post.matched.is_none() && post.page_actions.is_empty(),
            "a user record never touches post decisions"
        );
        let author = PageAction::SuspendAuthor {};
        let (ac, _) = loaded(&[
            live(head, None, None, vec![SUSPEND]),
            live(head, None, None, vec![author]),
        ])
        .await;
        assert_eq!(resolve(&ac, head, "c", &[]).page_actions, vec![SUSPEND]);
        let post = ac.resolve(&post_input(&[]), &on(), |_| true);
        assert_eq!(post.page_actions, vec![author]);
        assert_eq!(post.matched.unwrap().lkey, "post/h/SharedSpamBot");
        let mut odd = live(head, None, None, vec![author]);
        odd.entity_type = ControlEntity::User;
        let (ac, _) = loaded(&[odd]).await;
        let r = resolve(&ac, head, "c", &[]);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("wrong_entity"))
        );
        assert!(r.page_actions.is_empty());
    }

    #[tokio::test]
    async fn ignore_store_ignores_every_record() {
        let (ac, _) = loaded(&[record("R", None, None, ControlMode::Recording, vec![])]).await;
        let cfg = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        let ls = labels(&[]);
        assert_eq!(
            ac.resolve(&input("R", "c", &ls), &cfg, |_| true).mode,
            ControlMode::Rules
        );
    }

    #[tokio::test]
    async fn renamed_head_uses_the_old_name_through_an_alias() {
        let (ac, _) = loaded(&[record(
            "OldName",
            None,
            None,
            ControlMode::Recording,
            vec![],
        )])
        .await;
        let ls = labels(&[]);
        let mut cfg = on();
        let unaliased = ac.resolve(&input("NewName", "c", &ls), &cfg, |_| true);
        assert!(unaliased.unknown && unaliased.matched.is_none());
        cfg.aliases.insert("NewName".into(), "OldName".into());
        let r = ac.resolve(&input("NewName", "c", &ls), &cfg, |_| true);
        assert_eq!(r.mode, ControlMode::Recording);
        assert_eq!(r.matched.unwrap().lkey, "user/h/OldName");
    }

    #[tokio::test]
    async fn alias_records_keep_their_specificity() {
        let (ac, _) = loaded(&[
            live("NewName", None, None, vec![SUSPEND]),
            record("OldName", None, Some("b1"), ControlMode::Recording, vec![]),
            live("NewName", None, Some("b2"), vec![LIVENESS]),
            record("OldName", None, Some("b2"), ControlMode::Recording, vec![]),
            live("OldName", None, Some("b3"), vec![LIVENESS]),
        ])
        .await;
        let mut cfg = on();
        cfg.aliases.insert("NewName".into(), "OldName".into());
        let run = |ls: &[&str]| {
            let ls = labels(ls);
            ac.resolve(&input("NewName", "c", &ls), &cfg, |_| true)
        };
        let r = run(&["b1"]);
        assert_eq!(
            (r.mode, r.matched.unwrap().lkey),
            (ControlMode::Recording, "user/b/OldName/b1".to_owned()),
            "the old name's band record beats the new name's head record"
        );
        let r = run(&["b2"]);
        assert_eq!(
            r.matched.unwrap().lkey,
            "user/b/NewName/b2",
            "same scope: the new name wins"
        );
        let r = run(&["b2", "b3"]);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("band_ambiguous")),
            "bands split across the two names"
        );
    }

    #[tokio::test]
    async fn put_refuses_when_the_seen_list_cant_be_read() {
        let (ac, store) = loaded(&[]).await;
        store.fail_scan.store(true, Ordering::Relaxed);
        assert!(matches!(
            set(&ac, put("H", ControlMode::Live, vec![SUSPEND])).await,
            Err(SetError::Store(_))
        ));
        assert!(store.pkey_rows(MH_PKEY).is_empty());
    }

    #[tokio::test]
    async fn page_driven_decisions_claim_their_own_strongest_action() {
        let (ac, _) = loaded(&[
            live("A", None, None, vec![LIVENESS, SHR7]),
            record("R", None, None, ControlMode::Recording, vec![SHR7]),
            record("G", None, None, ControlMode::Rules, vec![]),
            record("E", None, None, ControlMode::Recording, vec![]),
        ])
        .await;
        let a = resolve(&ac, "A", "c", &[]);
        assert_eq!((a.dedup_class, a.incoming_dedup_class(|| 3)), (Some(2), 2));
        assert_eq!(resolve(&ac, "R", "c", &[]).incoming_dedup_class(|| 3), 1);
        let unknown = resolve(&ac, "None", "c", &[]);
        assert_eq!(unknown.incoming_dedup_class(|| 3), page_class(&[]));
        for head in ["G", "E"] {
            let r = resolve(&ac, head, "c", &[]);
            assert_eq!(r.dedup_class, None, "{head}");
            assert_eq!(
                r.incoming_dedup_class(|| 3),
                3,
                "{head}: same class as the rules"
            );
        }
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };
        let ls = labels(&[]);
        let r = ac.resolve(&input("A", "c", &ls), &killed, |_| true);
        assert!(r.record_only(), "kill switch: peeks, never claims");
        assert_eq!(r.incoming_dedup_class(|| 3), 2);
    }

    #[test]
    fn recording_sources() {
        let page = Resolved {
            mode: ControlMode::Recording,
            ..Default::default()
        };
        assert!(page.record_only());
        assert_eq!(page.dry_run_source(false), Some("page"));
        assert_eq!(page.dry_run_source(true), Some("global"));
        let producer = Resolved {
            producer_dry: true,
            ..Default::default()
        };
        assert_eq!(producer.dry_run_source(false), Some("producer"));
        assert_eq!(Resolved::default().dry_run_source(false), None);
    }

    #[test]
    fn page_specs_are_ordered_label_liveness_suspend() {
        let specs = page_specs(&[SUSPEND, LIVENESS, SHR7]);
        assert_eq!(
            specs,
            vec![
                ActionSpec::AddLabelsV2 {
                    labels: vec!["SpamHighRecall".into()],
                    ttl_msec: Some(604_800_000)
                },
                ActionSpec::SpamLivenessCheck,
                ActionSpec::SuspendUser {
                    perm: false,
                    policy: "PlatformManipulation".into()
                },
            ]
        );
        assert_eq!(
            page_specs(&[PageAction::PostSpamHighRecall {
                lifetime: Lifetime::Permanent
            }]),
            vec![ActionSpec::AddPostLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: None
            }]
        );
        let phoenix30 = PageAction::PhoenixSkip {
            lifetime: Lifetime::Days30,
        };
        assert_eq!(
            page_specs(&[LIVENESS, phoenix30, SHR7]),
            vec![
                ActionSpec::AddLabelsV2 {
                    labels: vec!["ClientEventPhoenixSkip".into()],
                    ttl_msec: Some(2_592_000_000)
                },
                ActionSpec::AddLabelsV2 {
                    labels: vec!["SpamHighRecall".into()],
                    ttl_msec: Some(604_800_000)
                },
                ActionSpec::SpamLivenessCheck,
            ]
        );
        assert_eq!(label_only(&[SHR30]), Some("SpamHighRecall"));
        assert_eq!(label_only(&[phoenix30]), Some("ClientEventPhoenixSkip"));
        assert_eq!(label_only(&[SHR30, LIVENESS]), None);
        assert_eq!(label_only(&[SHR30, phoenix30]), None);
    }

    #[test]
    fn hide_reply_sends_the_risky_high_viz_reply_post_label() {
        let hide = |lifetime| PageAction::HideReply { lifetime };
        assert_eq!(
            page_specs(&[PageAction::SuspendAuthor {}, hide(Lifetime::Permanent)]),
            vec![
                ActionSpec::AddPostLabelsV2 {
                    labels: vec!["RiskyHighVizReply".into()],
                    ttl_msec: None
                },
                ActionSpec::SuspendUser {
                    perm: false,
                    policy: "PlatformManipulation".into()
                },
            ]
        );
        assert_eq!(
            page_specs(&[hide(Lifetime::Days7)]),
            vec![ActionSpec::AddPostLabelsV2 {
                labels: vec!["RiskyHighVizReply".into()],
                ttl_msec: Some(604_800_000)
            }]
        );
        assert_eq!(
            action_specs_value(&page_specs(&[hide(Lifetime::Days30)])),
            serde_json::json!([
                {"kind": "post_label", "label": "RiskyHighVizReply", "ttl_days": 30, "permanent": false}
            ])
        );
        assert_eq!(label_only(&[hide(Lifetime::Permanent)]), None);
    }

    #[test]
    fn action_specs_have_the_contract_shape() {
        let v = action_specs_value(&[
            ActionSpec::SuspendUser {
                perm: false,
                policy: "PlatformManipulation".into(),
            },
            ActionSpec::AddLabelsV2 {
                labels: vec!["SpamHighRecall".into(), "Other".into()],
                ttl_msec: Some(7 * 86_400_000),
            },
            ActionSpec::AddPostLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: None,
            },
            ActionSpec::AddLabelsV2 {
                labels: vec!["X".into()],
                ttl_msec: Some(43_200_000),
            },
            ActionSpec::SpamLivenessCheck,
            ActionSpec::Arkose,
            ActionSpec::Captcha,
        ]);
        assert_eq!(
            v,
            serde_json::json!([
                {"kind": "suspend", "permanent": false, "policy": "PlatformManipulation"},
                {"kind": "label", "label": "SpamHighRecall", "ttl_days": 7, "permanent": false},
                {"kind": "label", "label": "Other", "ttl_days": 7, "permanent": false},
                {"kind": "post_label", "label": "SpamHighRecall", "permanent": true},
                {"kind": "label", "label": "X", "ttl_days": 0.5, "permanent": false},
                {"kind": "liveness_check"},
                {"kind": "arkose"},
                {"kind": "captcha"},
            ])
        );
    }

    #[test]
    fn config_parsing_fails_toward_less_enforcement() {
        use serde_json::json;
        let c = |v: serde_json::Value| ControlConfig::from_config(Some(&v));
        assert!(
            !ControlConfig::from_config(None).enabled,
            "no config = not on"
        );
        assert!(!c(json!({})).enabled, "no action_control object = not on");
        assert!(!c(json!({CONFIG_KEY: null})).enabled);
        assert!(
            !c(json!({CONFIG_KEY: {}})).enabled,
            "enabled absent = not on"
        );
        assert!(!c(json!({CONFIG_KEY: {"enabled": null}})).enabled);
        assert!(c(json!({CONFIG_KEY: {"enabled": true}})).enabled);
        assert!(!c(json!({CONFIG_KEY: {"enabled": false}})).enabled);
        assert!(
            !c(json!({CONFIG_KEY: {"enabled": "no"}})).enabled,
            "malformed = killed"
        );
        assert!(!c(json!({CONFIG_KEY: false})).enabled, "malformed = killed");
        assert!(!c(json!({CONFIG_KEY: {"ignore_store": "yes"}})).ignore_store);
        assert!(c(json!({CONFIG_KEY: {"ignore_store": true}})).ignore_store);
        let extra = c(json!({CONFIG_KEY: {"aliases": {"New": "Old", "bad/name": "Old"}}}));
        assert_eq!(extra.aliases.len(), 1);
    }

    #[tokio::test]
    async fn a_scan_cut_at_its_limit_is_a_failed_read() {
        let (ac, store) = loaded(&[record("R", None, None, ControlMode::Recording, vec![])]).await;
        for i in 0..SCAN_LIMIT {
            store.put_record(&record(
                &format!("Filler{i}"),
                None,
                None,
                ControlMode::Rules,
                vec![],
            ));
        }
        assert!(ac.refresh().await.is_err());
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
        assert_eq!(ac.pod_view().records, 1, "last good list kept");
    }

    #[tokio::test]
    async fn failed_reads_keep_the_last_good_list() {
        let (ac, store) = loaded(&[record("R", None, None, ControlMode::Recording, vec![])]).await;
        store.fail_scan.store(true, Ordering::Relaxed);
        for _ in 0..3 {
            assert!(ac.refresh().await.is_err());
        }
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
    }

    #[tokio::test]
    async fn a_pod_booted_under_ignore_store_reads_before_deciding() {
        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("R", None, None, ControlMode::Recording, vec![]));
        let ac = ActionControl::new(store.clone(), "prod".into(), String::new());
        let ignoring = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        ac.ensure_loaded(&ignoring).await;
        assert!(!ac.is_loaded(), "the break-glass flag reads nothing");
        ac.ensure_loaded(&on()).await;
        assert!(ac.is_loaded());
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
    }

    #[tokio::test]
    async fn an_unreadable_record_shows_its_version_and_can_be_replaced() {
        let (ac, store) = loaded(&[]).await;
        store.put_raw(
            "user/h/H",
            br#"{"v":1,"entity_type":"user","head":"H","mode":"live","version":7,"x":1}"#,
        );
        ac.refresh().await.unwrap();
        let list = ac.list(&on(), baked_safety()).await;
        assert_eq!(list.unreadable[0].version, 7);
        match set(&ac, put("H", ControlMode::Recording, vec![])).await {
            Err(SetError::Conflict(None, Some(7))) => {}
            other => panic!("{other:?}"),
        }
        let mut replace = put("H", ControlMode::Recording, vec![]);
        replace.expected_version = Some(7);
        assert_eq!(set(&ac, replace).await.unwrap().view.record.version, 8);
    }

    async fn wait_for_scans(store: &MemControlStore, n: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while store.scans.load(Ordering::Relaxed) < n {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("reads keep retrying");
    }

    #[tokio::test]
    async fn the_background_read_retries_until_the_first_success() {
        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("R", None, None, ControlMode::Recording, vec![]));
        store.fail_scan.store(true, Ordering::Relaxed);
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            String::new(),
        ));
        let task = ac.spawn_refresh_with(
            || false,
            Duration::from_millis(40),
            Duration::from_millis(2),
        );
        wait_for_scans(&store, 4).await;
        assert!(!ac.is_loaded());
        assert!(!ac.ready(&on()), "a failed first read is not an empty list");
        assert!(ac.is_stale(), "never loaded counts as stale");

        store.fail_scan.store(false, Ordering::Relaxed);
        tokio::time::timeout(Duration::from_secs(5), async {
            while !ac.ready(&on()) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("ready once a read succeeds");
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
        task.abort();
    }

    #[tokio::test]
    async fn ignore_store_reads_nothing_and_does_not_wait() {
        let store = Arc::new(MemControlStore::default());
        store.fail_scan.store(true, Ordering::Relaxed);
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            String::new(),
        ));
        let ignoring = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        let task =
            ac.spawn_refresh_with(|| true, Duration::from_millis(10), Duration::from_millis(2));
        assert!(ac.ready(&ignoring));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(store.scans.load(Ordering::Relaxed), 0);
        task.abort();
    }

    #[tokio::test]
    async fn a_message_read_has_no_time_out() {
        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("R", None, None, ControlMode::Recording, vec![]));
        store.fail_scan.store(true, Ordering::Relaxed);
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            String::new(),
        ));
        assert!(!ac.read_until_loaded(Duration::from_millis(1), 3).await);
        let task = {
            let ac = ac.clone();
            tokio::spawn(async move {
                ac.read_until_loaded(Duration::from_millis(5), u64::MAX)
                    .await
            })
        };
        wait_for_scans(&store, 6).await;
        assert!(!task.is_finished(), "no time-out: still waiting");
        store.fail_scan.store(false, Ordering::Relaxed);
        assert!(task.await.unwrap());
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
    }


    fn put(head: &str, mode: ControlMode, actions: Vec<PageAction>) -> ActionControlPutRequest {
        ActionControlPutRequest {
            head: head.into(),
            classifier: None,
            band: None,
            mode,
            actions,
            reason: "because".into(),
            expected_version: None,
            confirm: Some(CONFIRM_WORD.into()),
            entity_type: ControlEntity::User,
            updated_by: None,
            confidence: None,
            evidence: None,
            evidence_override: None,
            rows: None,
            admit_unlisted_labels: None,
            rollout_bp: None,
            follower_threshold: None,
            skip_high_credibility: None,
        }
    }

    const NOW_MS: i64 = 1_791_000_000_000;

    async fn set(ac: &ActionControl, req: ActionControlPutRequest) -> Result<SetOutcome, SetError> {
        set_at(ac, req, NOW_MS).await
    }

    async fn set_at(
        ac: &ActionControl,
        req: ActionControlPutRequest,
        now_ms: i64,
    ) -> Result<SetOutcome, SetError> {
        ac.set(
            req,
            "operator@example.com",
            "pv2",
            now_ms,
            &on(),
            &baked_safety(),
        )
        .await
    }

    fn code(r: Result<SetOutcome, SetError>) -> &'static str {
        match r {
            Err(SetError::Invalid(c, _)) => c,
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_validation_codes() {
        let (ac, store) = loaded(&[]).await;
        let seen = |head: &str, factory: bool, entity: ControlEntity| ActionControlSeen {
            head: head.into(),
            classifier: "c".into(),
            band: "-".into(),
            entity_type: entity,
            factory,
            last_seen_ms: 1,
            rule_id: String::new(),
            action_specs: serde_json::json!([]),
        };
        for s in [
            seen("Factory", true, ControlEntity::User),
            seen("Factory", false, ControlEntity::Post),
            seen("PostHead", false, ControlEntity::Post),
        ] {
            store.rows.lock().unwrap().insert(
                (SEEN_PKEY.into(), seen_lkey(&s)),
                serde_json::to_vec(&s).unwrap(),
            );
        }
        let cases: Vec<(ActionControlPutRequest, &str)> = vec![
            (put("a/b", ControlMode::Rules, vec![]), "invalid_head"),
            (
                ActionControlPutRequest {
                    classifier: Some("x y".into()),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "invalid_classifier",
            ),
            (
                ActionControlPutRequest {
                    band: Some("dry_run__b".into()),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "invalid_band",
            ),
            (
                ActionControlPutRequest {
                    reason: " ".into(),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "reason_required",
            ),
            (
                ActionControlPutRequest {
                    reason: "r".repeat(1001),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "reason_too_long",
            ),
            (
                put("H", ControlMode::Rules, vec![SUSPEND]),
                "rules_takes_no_actions",
            ),
            (put("H", ControlMode::Live, vec![]), "live_needs_actions"),
            (
                put("H", ControlMode::Live, vec![SHR7, SHR30]),
                "duplicate_action",
            ),
            (
                put(
                    "H",
                    ControlMode::Live,
                    vec![SUSPEND, PageAction::SuspendAuthor {}],
                ),
                "wrong_entity_actions",
            ),
            (
                ActionControlPutRequest {
                    entity_type: ControlEntity::Post,
                    ..put("PostHead", ControlMode::Live, vec![SUSPEND])
                },
                "wrong_entity_actions",
            ),
            (
                put("PostHead", ControlMode::Recording, vec![]),
                "wrong_entity",
            ),
            (
                ActionControlPutRequest {
                    entity_type: ControlEntity::Post,
                    ..put(
                        "PostHead",
                        ControlMode::Live,
                        vec![PageAction::PostSpamHighRecall {
                            lifetime: Lifetime::Days7,
                        }],
                    )
                },
                "invalid_lifetime",
            ),
            (put("Factory", ControlMode::Rules, vec![]), "factory_rules"),
            (
                put("Factory", ControlMode::Live, vec![SUSPEND]),
                "factory_scope",
            ),
            (
                put("Factory", ControlMode::Recording, vec![SUSPEND]),
                "factory_scope",
            ),
            (
                ActionControlPutRequest {
                    classifier: Some("c".into()),
                    ..put("Factory", ControlMode::Live, vec![SUSPEND])
                },
                "factory_scope",
            ),
            (
                row_put("Factory", LevelRow::Low, ControlMode::Rules, vec![]),
                "factory_rules",
            ),
            (
                move_put(
                    "Factory",
                    RowsRequest {
                        high: Some(setting(ControlMode::Live, vec![SUSPEND])),
                        ..all_rules()
                    },
                ),
                "factory_rules",
            ),
            (
                ActionControlPutRequest {
                    band: Some("model_factory".into()),
                    ..put("Factory", ControlMode::Live, vec![SUSPEND])
                },
                "factory_band",
            ),
            (
                ActionControlPutRequest {
                    evidence: Some(read(0.96, 59, 0)),
                    ..row_put("Factory", LevelRow::High, ControlMode::Live, vec![SUSPEND])
                },
                "factory_evidence",
            ),
            (
                move_put(
                    "Factory",
                    RowsRequest {
                        high: Some(RowSetting {
                            evidence: Some(read(0.96, 59, 0)),
                            ..setting(ControlMode::Live, vec![SUSPEND])
                        }),
                        ..all_rows(ControlMode::Recording, vec![])
                    },
                ),
                "factory_evidence",
            ),
            (
                ActionControlPutRequest {
                    classifier: Some("c".into()),
                    ..row_put("Factory", LevelRow::High, ControlMode::Live, vec![SUSPEND])
                },
                "scope_not_on_rows",
            ),
            (
                ActionControlPutRequest {
                    confirm: None,
                    ..put("H", ControlMode::Live, vec![SUSPEND])
                },
                "confirm_required",
            ),
            (
                ActionControlPutRequest {
                    evidence_override: Some(person(1_791_000_000_000 + 86_400_000)),
                    ..row_put("Factory", LevelRow::Low, ControlMode::Live, vec![SHR7])
                },
                "invalid_override",
            ),
        ];
        for (req, want) in cases {
            assert_eq!(code(set(&ac, req.clone()).await), want, "{req:?}");
        }
        assert!(store.pkey_rows(MH_PKEY).is_empty(), "nothing written");
        for req in [
            row_put("Factory", LevelRow::High, ControlMode::Live, vec![SUSPEND]),
            row_put(
                "Factory",
                LevelRow::Mid,
                ControlMode::Recording,
                vec![LIVENESS],
            ),
            row_put("Factory", LevelRow::Low, ControlMode::Live, vec![SHR7]),
        ] {
            let got = set(&ac, req.clone()).await;
            assert!(got.is_ok(), "{req:?}: {got:?}");
        }
        let written = store.pkey_rows(MH_PKEY);
        assert_eq!(written.len(), 3, "three row records and no scope record");
        assert!(
            written
                .iter()
                .all(|(k, _)| k.starts_with("user/l/Factory/"))
        );
        let r = ac.resolve(
            &factory_input(
                "Factory",
                "c",
                Some(crate::confidence::Claim {
                    row: LevelRow::Low,
                    unknown: false,
                    evidence: None,
                    now_ms: 1_791_000_000_000,
                }),
            ),
            &on(),
            |_| true,
        );
        assert_eq!(
            (r.mode, r.forced, r.page_actions.clone()),
            (ControlMode::Live, None, vec![SHR7]),
            "a live row acts with no evidence"
        );
        assert_eq!(r.row.unwrap().evidence_supports, Some(None));
        for req in [
            put("Factory", ControlMode::Recording, vec![]),
            ActionControlPutRequest {
                classifier: Some("c".into()),
                ..move_put("Factory", all_rows(ControlMode::Recording, vec![]))
            },
            ActionControlPutRequest {
                entity_type: ControlEntity::Post,
                ..put(
                    "Factory",
                    ControlMode::Live,
                    vec![PageAction::PostSpamHighRecall {
                        lifetime: Lifetime::Days30,
                    }],
                )
            },
        ] {
            let got = set(&ac, req.clone()).await;
            assert!(got.is_ok(), "{req:?}: {got:?}");
        }
        set(&ac, put("H", ControlMode::Rules, vec![]))
            .await
            .unwrap();
        for req in [
            put(
                "NearDupEmbeddingCseNcmecMatch",
                ControlMode::Live,
                vec![SUSPEND],
            ),
            ActionControlPutRequest {
                band: Some("cse_reports_embedding_v2".into()),
                ..put("H", ControlMode::Live, vec![SUSPEND])
            },
        ] {
            let got = set(&ac, req.clone()).await;
            assert!(got.is_ok(), "{req:?}: {got:?}");
        }
        let p = set(
            &ac,
            ActionControlPutRequest {
                entity_type: ControlEntity::Post,
                ..put(
                    "PostHead",
                    ControlMode::Live,
                    vec![PageAction::SuspendAuthor {}],
                )
            },
        )
        .await;
        assert!(p.is_ok(), "{p:?}");
    }

    #[tokio::test]
    async fn recording_back_to_the_rules_is_harsher_for_every_head() {
        let (ac, _) = loaded(&[]).await;
        let back =
            |head: &str, band: Option<&str>, confirm: Option<&str>| ActionControlPutRequest {
                band: band.map(Into::into),
                expected_version: Some(1),
                confirm: confirm.map(Into::into),
                ..put(head, ControlMode::Rules, vec![])
            };
        for (head, band) in [
            ("H", None),
            ("NearDupEmbeddingCseMatch", None),
            ("AnchorHead", Some("anchor_campaign_suspend_cse")),
        ] {
            if band.is_some() {
                set(&ac, put(head, ControlMode::Rules, vec![]))
                    .await
                    .unwrap();
            }
            let recording = ActionControlPutRequest {
                band: band.map(Into::into),
                ..put(head, ControlMode::Recording, vec![])
            };
            set(&ac, recording).await.unwrap();
            assert_eq!(
                code(set(&ac, back(head, band, None)).await),
                "confirm_required",
                "{head}"
            );
            let out = set(&ac, back(head, band, Some(CONFIRM_WORD)))
                .await
                .unwrap();
            assert_eq!(
                out.harsher_because,
                vec!["hands the scope back to the GrowthBook rules, which act"],
                "{head}"
            );
        }
    }

                #[tokio::test]
    async fn an_unlisted_decision_is_admitted_only_by_a_record_that_decides() {
        let (ac, _) = loaded(&[
            record("Moved", None, None, ControlMode::Rows, vec![]),
            record("Recording", None, None, ControlMode::Recording, vec![]),
            live("Live", None, None, vec![SUSPEND]),
            record("OnRules", None, None, ControlMode::Rules, vec![]),
            record("RulesRow", None, None, ControlMode::Rows, vec![]),
            row_rec("RulesRow", LevelRow::Missing, ControlMode::Rules, vec![]),
        ])
        .await;
        for head in ["Moved", "Recording", "Live", "None"] {
            let r = resolve(&ac, head, "c", &[]);
            let page = r.page_actions.clone();
            let r = r
                .admit_unlisted()
                .unwrap_or_else(|| panic!("{head} admitted"));
            assert!(r.unlisted && r.page_driven() && r.rules_unread(), "{head}");
            assert_eq!(
                r.incoming_dedup_class(|| panic!("rules asked for {head}")),
                page_class(&page)
            );
        }
        let moved = resolve(&ac, "Moved", "c", &[]).admit_unlisted().unwrap();
        assert_eq!(moved.mode, ControlMode::Recording);
        assert!(moved.page_actions.is_empty());
        assert_eq!(logged_because(&moved).as_deref(), Some("no_selection"));
        for head in ["OnRules", "RulesRow"] {
            assert!(
                resolve(&ac, head, "c", &[]).admit_unlisted().is_none(),
                "{head}"
            );
        }
        let factory = ac.resolve(&factory_input("Moved", "c", None), &on(), |_| true);
        assert!(
            factory.admit_unlisted().is_none(),
            "factory heads unchanged"
        );
        let listed = resolve(&ac, "Moved", "c", &[]);
        assert!(!listed.unlisted && !listed.rules_unread());
    }

    #[tokio::test]
    async fn put_writes_versions_history_and_applies_at_once() {
        let (ac, store) = loaded(&[]).await;
        let out = set(&ac, put("H", ControlMode::Recording, vec![]))
            .await
            .unwrap();
        assert_eq!(out.view.record.version, 1);
        assert_eq!(out.view.record.updated_by, "operator@example.com");
        assert_eq!(out.view.record.key_id, "pv2");
        assert!(out.history_written);
        assert_eq!(
            resolve(&ac, "H", "c", &[]).mode,
            ControlMode::Recording,
            "applied locally"
        );
        match set(&ac, put("H", ControlMode::Rules, vec![])).await {
            Err(SetError::Conflict(Some(cur), Some(1))) => assert_eq!(cur.version, 1),
            other => panic!("{other:?}"),
        }
        let mut next = put("H", ControlMode::Live, vec![SUSPEND]);
        next.expected_version = Some(1);
        let out = set(&ac, next).await.unwrap();
        assert_eq!(out.view.record.version, 2);
        assert_eq!(out.before.unwrap().mode, ControlMode::Recording);
        assert_eq!(out.harsher_because, vec!["turns the scope live"]);
        let h = ac.history("H", 50).await;
        assert_eq!(h.entries.len(), 2);
        assert_eq!(h.entries[0].version, 2, "newest first");
        assert_eq!(h.entries[0].before.as_ref().unwrap().version, 1);
        assert_eq!(h.entries[1].before, None);
        assert_eq!(ac.history("H", 1).await.entries.len(), 1);
        assert_eq!(
            store.pkey_rows(MH_PKEY).len(),
            1,
            "one record per scope, never deleted"
        );
    }

    #[tokio::test]
    async fn put_reports_a_lost_race() {
        let (ac, store) = loaded(&[]).await;
        let mut other = record("H", None, None, ControlMode::Rules, vec![]);
        other.write_nonce = 99;
        *store.clobber_with.lock().unwrap() = Some(serde_json::to_vec(&other).unwrap());
        match set(&ac, put("H", ControlMode::Recording, vec![])).await {
            Err(SetError::LostRace(Some(cur))) => assert_eq!(cur.write_nonce, 99),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            ac.history("H", 10).await.entries.len(),
            1,
            "history keeps the write that was live"
        );
    }

    #[tokio::test]
    async fn a_rules_record_that_shadows_recording_is_harsher() {
        let (ac, store) = loaded(&[
            record("H", None, None, ControlMode::Recording, vec![]),
            record(
                "NearDupEmbeddingCseMatch",
                None,
                None,
                ControlMode::Recording,
                vec![],
            ),
            record("OldName", None, None, ControlMode::Recording, vec![]),
            record("H2", None, Some("bx"), ControlMode::Recording, vec![]),
        ])
        .await;
        let clf_rules = |head: &str| ActionControlPutRequest {
            classifier: Some("clf".into()),
            confirm: None,
            ..put(head, ControlMode::Rules, vec![])
        };
        assert_eq!(code(set(&ac, clf_rules("H2")).await), "confirm_required");
        store.put_record(&record("H3", None, None, ControlMode::Recording, vec![]));
        let band_rules = |head: &str, confirm: Option<&str>| ActionControlPutRequest {
            band: Some("some_band".into()),
            confirm: confirm.map(Into::into),
            ..put(head, ControlMode::Rules, vec![])
        };
        for head in ["H", "NearDupEmbeddingCseMatch", "H3"] {
            assert_eq!(
                code(set(&ac, band_rules(head, None)).await),
                "confirm_required",
                "{head}"
            );
            let saved = set(&ac, band_rules(head, Some(CONFIRM_WORD))).await;
            assert!(saved.is_ok(), "{head}: {saved:?}");
        }
        let mut cfg = on();
        cfg.aliases.insert("NewName".into(), "OldName".into());
        let renamed = |head: &str, confirm: Option<&str>| ActionControlPutRequest {
            confirm: confirm.map(Into::into),
            ..put(head, ControlMode::Rules, vec![])
        };
        let safety = baked_safety();
        let set_cfg = |req| ac.set(req, "w", "k", 1, &cfg, &safety);
        assert_eq!(
            code(set_cfg(renamed("NewName", None)).await),
            "confirm_required"
        );
        assert!(
            set_cfg(renamed("NewName", Some(CONFIRM_WORD)))
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn put_answer_shows_this_pods_effective_mode() {
        let (ac, _) = loaded(&[]).await;
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };
        let out = ac
            .set(
                put("H", ControlMode::Live, vec![SUSPEND]),
                "w",
                "k",
                1,
                &killed,
                &baked_safety(),
            )
            .await
            .unwrap();
        assert_eq!(out.view.effective_mode, ControlMode::Recording);
        assert_eq!(
            out.view.forced_recording_reason.as_deref(),
            Some("kill_switch")
        );
    }

    #[tokio::test]
    async fn a_live_record_running_as_recording_back_to_covered_rules_needs_the_confirmation() {
        let covered = || async {
            let (ac, store) = loaded(&[live("H", None, None, vec![SUSPEND])]).await;
            let s = ActionControlSeen {
                head: "H".into(),
                classifier: "c".into(),
                band: "-".into(),
                entity_type: ControlEntity::User,
                factory: false,
                last_seen_ms: 1,
                rule_id: "act_x".into(),
                action_specs: serde_json::json!([
                    {"kind": "suspend", "permanent": false, "policy": "PlatformManipulation"}
                ]),
            };
            store.rows.lock().unwrap().insert(
                (SEEN_PKEY.into(), seen_lkey(&s)),
                serde_json::to_vec(&s).unwrap(),
            );
            ac
        };
        let back = |confirm: Option<&str>| ActionControlPutRequest {
            expected_version: Some(1),
            confirm: confirm.map(Into::into),
            ..put("H", ControlMode::Rules, vec![])
        };
        let safety = baked_safety();
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };

        let ac = covered().await;
        assert_eq!(
            code(ac.set(back(None), "w", "k", 1, &killed, &safety).await),
            "confirm_required",
            "page control off"
        );
        let out = ac
            .set(back(Some(CONFIRM_WORD)), "w", "k", 1, &killed, &safety)
            .await
            .unwrap();
        assert_eq!(
            out.harsher_because,
            vec!["hands the scope back to the GrowthBook rules, which act"]
        );

        let ac = covered().await;
        ac.age_for_test(STALE_AFTER + Duration::from_secs(1));
        assert_eq!(
            code(set(&ac, back(None)).await),
            "confirm_required",
            "stale copy"
        );
        assert!(set(&ac, back(Some(CONFIRM_WORD))).await.is_ok());

        let ac = covered().await;
        let out = set(&ac, back(None)).await.unwrap();
        assert!(out.harsher_because.is_empty(), "{:?}", out.harsher_because);
    }

    #[tokio::test]
    async fn an_unreadable_read_back_is_not_applied_locally() {
        let (ac, store) = loaded(&[]).await;
        *store.clobber_with.lock().unwrap() = Some(b"{not json".to_vec());
        assert!(matches!(
            set(&ac, put("H", ControlMode::Recording, vec![])).await,
            Err(SetError::Store(_))
        ));
        assert!(resolve(&ac, "H", "c", &[]).matched.is_none());
    }

    #[tokio::test]
    async fn failed_store_write_changes_nothing() {
        let (ac, store) = loaded(&[]).await;
        store.fail_put.store(true, Ordering::Relaxed);
        assert!(matches!(
            set(&ac, put("H", ControlMode::Recording, vec![])).await,
            Err(SetError::Store(_))
        ));
        assert!(resolve(&ac, "H", "c", &[]).matched.is_none());
        store.fail_put.store(false, Ordering::Relaxed);
        store.fail_get.store(true, Ordering::Relaxed);
        assert!(matches!(
            set(&ac, put("H", ControlMode::Recording, vec![])).await,
            Err(SetError::Store(_))
        ));
        let ignore = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        assert_eq!(
            ac.set(
                put("H", ControlMode::Rules, vec![]),
                "w",
                "k",
                1,
                &ignore,
                &baked_safety()
            )
            .await
            .unwrap_err(),
            SetError::Ignored
        );
    }

    #[test]
    fn harsher_changes() {
        let r = |mode, actions: Vec<PageAction>| record("H", None, None, mode, actions);
        let h =
            |b: Option<&ActionControlRecord>, a: &ActionControlRecord| harsher_because(b, a, None);
        assert_eq!(
            h(None, &r(ControlMode::Recording, vec![])),
            Vec::<String>::new()
        );
        assert!(!h(None, &r(ControlMode::Live, vec![LIVENESS])).is_empty());
        let rec = r(ControlMode::Recording, vec![]);
        assert!(
            !h(Some(&rec), &r(ControlMode::Rules, vec![])).is_empty(),
            "recording → rules acts"
        );
        let l = r(ControlMode::Live, vec![SHR7]);
        assert!(
            h(Some(&l), &r(ControlMode::Live, vec![SHR7, SUSPEND]))
                .contains(&"adds suspend".to_owned())
        );
        assert!(
            h(Some(&l), &r(ControlMode::Live, vec![SHR7, LIVENESS]))
                .contains(&"adds liveness check".to_owned())
        );
        assert!(
            h(Some(&l), &r(ControlMode::Live, vec![SHR30]))
                .contains(&"makes the label last longer".to_owned())
        );
        let shorter = h(
            Some(&r(ControlMode::Live, vec![SHR30])),
            &r(ControlMode::Live, vec![SHR7]),
        );
        assert!(shorter.is_empty(), "{shorter:?}");
        assert!(
            h(Some(&l), &r(ControlMode::Recording, vec![SHR7])).is_empty(),
            "live → recording is softer"
        );
        let back = |page: Vec<PageAction>, rules: serde_json::Value| {
            harsher_because(
                Some(&r(ControlMode::Live, page)),
                &r(ControlMode::Rules, vec![]),
                Some(rules.as_array().unwrap()),
            )
        };
        use serde_json::json;
        let temp_suspend =
            json!({"kind": "suspend", "permanent": false, "policy": "PlatformManipulation"});
        let label7 =
            json!({"kind": "label", "label": "SpamHighRecall", "ttl_days": 7, "permanent": false});
        assert!(back(vec![SHR7, SUSPEND], json!([label7, temp_suspend])).is_empty());
        assert!(
            back(vec![SHR30], json!([label7])).is_empty(),
            "rules' label is shorter"
        );
        assert!(
            !back(vec![SHR30], json!([])).is_empty(),
            "seen rows with no actions: unknown"
        );
        for harsher in [
            json!([{"kind": "suspend", "permanent": true, "policy": "PlatformManipulation"}]),
            json!([{"kind": "suspend", "permanent": false, "policy": "Cse"}]),
            json!([{"kind": "label", "label": "SpamHighRecall", "permanent": true}]),
            json!([{"kind": "label", "label": "Other", "ttl_days": 7, "permanent": false}]),
            json!([{"kind": "arkose"}]),
        ] {
            assert!(
                !back(vec![SHR30, SUSPEND, LIVENESS], harsher.clone()).is_empty(),
                "{harsher}"
            );
        }
        assert!(
            !harsher_because(Some(&l), &r(ControlMode::Rules, vec![]), None).is_empty(),
            "rules' actions unknown"
        );
    }

    #[tokio::test]
    async fn seen_rows_are_written_at_most_once_per_window() {
        let (ac, store) = loaded(&[]).await;
        let row = |band: &str| ActionControlSeen {
            head: "H".into(),
            classifier: "c@1".into(),
            band: band.into(),
            entity_type: ControlEntity::User,
            factory: false,
            last_seen_ms: 5,
            rule_id: "act_x".into(),
            action_specs: serde_json::json!([{"kind": "suspend"}]),
        };
        let t0 = Instant::now();
        assert_eq!(ac.record_seen_at(row("a"), t0), SeenOutcome::Queued);
        assert_eq!(
            ac.record_seen_at(row("a"), t0 + Duration::from_secs(1)),
            SeenOutcome::Recent
        );
        assert_eq!(ac.record_seen_at(row("-"), t0), SeenOutcome::Queued);
        assert_eq!(ac.record_seen_at(row("a/b"), t0), SeenOutcome::Invalid);
        ac.spawn_seen_writer().unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while store.pkey_rows(SEEN_PKEY).len() < 2 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let list = ac.list(&on(), baked_safety()).await;
        assert_eq!(list.seen.len(), 2);
        assert_eq!(list.seen[0].band, "-");
        assert_eq!(list.source, "store");
    }


    const PUT_NOW: i64 = 1_791_000_000_000;

    fn row_put(
        head: &str,
        row: LevelRow,
        mode: ControlMode,
        actions: Vec<PageAction>,
    ) -> ActionControlPutRequest {
        ActionControlPutRequest {
            confidence: Some(row),
            ..put(head, mode, actions)
        }
    }

    fn setting(mode: ControlMode, actions: Vec<PageAction>) -> RowSetting {
        RowSetting {
            mode,
            actions,
            evidence: None,
            evidence_override: None,
            expected_version: None,
            rollout_bp: None,
        }
    }

        fn move_put(head: &str, rows: RowsRequest) -> ActionControlPutRequest {
        ActionControlPutRequest {
            rows: Some(rows),
            ..put(head, ControlMode::Rows, vec![])
        }
    }

    fn all_rows(mode: ControlMode, actions: Vec<PageAction>) -> RowsRequest {
        RowsRequest {
            low: Some(setting(mode, actions.clone())),
            mid: Some(setting(mode, actions.clone())),
            high: Some(setting(mode, actions.clone())),
            missing: Some(setting(mode, actions)),
        }
    }

    fn read(lb: f64, reads: u32, humans: u32) -> ConfidenceEvidence {
        ConfidenceEvidence {
            precision_lower_bound: lb,
            reads,
            humans,
            read_at_ms: PUT_NOW - 86_400_000,
            link: "documents/read.md".into(),
        }
    }

    fn person(expires_at_ms: i64) -> EvidenceOverride {
        EvidenceOverride {
            by: "operator@example.com".into(),
            reason: "seeded while the read runs".into(),
            expires_at_ms,
        }
    }

    fn row_rec(
        head: &str,
        row: LevelRow,
        mode: ControlMode,
        actions: Vec<PageAction>,
    ) -> ActionControlRecord {
        ActionControlRecord {
            confidence: Some(row),
            ..record(head, None, None, mode, actions)
        }
    }

    async fn put_with(
        ac: &ActionControl,
        cfg: &ControlConfig,
        req: ActionControlPutRequest,
    ) -> Result<SetOutcome, SetError> {
        ac.set(
            req,
            "operator@example.com",
            "pv2",
            PUT_NOW,
            cfg,
            &baked_safety(),
        )
        .await
    }

    fn high_claim() -> crate::confidence::Claim {
        crate::confidence::Claim {
            row: LevelRow::High,
            unknown: false,
            evidence: Some(read(0.96, 59, 0)),
            now_ms: crate::now_millis(),
        }
    }

                #[test]
    fn the_unlisted_opt_in_reads_false_and_is_written_only_when_true() {
        let r = record("H", None, None, ControlMode::Rows, vec![]);
        let bytes = serde_json::to_vec(&r).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("admit_unlisted_labels"));
        assert!(!parse_record(&bytes).unwrap().admit_unlisted_labels);
        let on = ActionControlRecord {
            admit_unlisted_labels: true,
            ..r
        };
        let bytes = serde_json::to_vec(&on).unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("\"admit_unlisted_labels\":true"));
        assert!(parse_record(&bytes).unwrap().admit_unlisted_labels);
    }

                #[tokio::test]
    async fn the_unlisted_opt_in_is_saved_on_the_head_record_only() {
        let (ac, store) = loaded(&[record("H", Some("c"), None, ControlMode::Rules, vec![])]).await;
        let opt_in = |confirm: Option<&str>| ActionControlPutRequest {
            admit_unlisted_labels: Some(true),
            confirm: confirm.map(Into::into),
            ..move_put("H", RowsRequest::default())
        };
        match set(&ac, opt_in(None)).await {
            Err(SetError::Invalid("confirm_required", msg)) => {
                assert!(msg.contains("label list"), "{msg}")
            }
            other => panic!("{other:?}"),
        }
        assert!(!ac.admits_unlisted(EntityType::User, "H"));
        let out = set(&ac, opt_in(Some(CONFIRM_WORD))).await.unwrap();
        assert!(out.view.record.admit_unlisted_labels);
        assert!(ac.admits_unlisted(EntityType::User, "H"));
        assert!(!ac.admits_unlisted(EntityType::Post, "H"));
        let rows = store.pkey_rows(MH_PKEY);
        for (k, v) in &rows {
            let r = parse_record(v).unwrap();
            assert_eq!(r.admit_unlisted_labels, k == "user/h/H", "{k}");
        }

        let leave = ActionControlPutRequest {
            expected_version: Some(1),
            ..put("H", ControlMode::Recording, vec![])
        };
        set(&ac, leave).await.unwrap();
        assert!(
            ac.admits_unlisted(EntityType::User, "H"),
            "kept when left out"
        );
        let off = ActionControlPutRequest {
            expected_version: Some(2),
            admit_unlisted_labels: Some(false),
            ..put("H", ControlMode::Recording, vec![])
        };
        set(&ac, off).await.unwrap();
        assert!(!ac.admits_unlisted(EntityType::User, "H"));

        let narrow = ActionControlPutRequest {
            classifier: Some("c".into()),
            admit_unlisted_labels: Some(true),
            ..put("H2", ControlMode::Recording, vec![])
        };
        assert_eq!(code(set(&ac, narrow).await), "admit_unlisted_labels_scope");
        seen_factory(&store, "Factory");
        let factory = ActionControlPutRequest {
            admit_unlisted_labels: Some(true),
            ..move_put("Factory", RowsRequest::default())
        };
        assert_eq!(
            code(set(&ac, factory).await),
            "factory_admit_unlisted_labels"
        );
    }

                    #[tokio::test]
    async fn a_first_save_of_an_unknown_head_keeps_its_decisions_admitted() {
        let unconfirmed = |req| ActionControlPutRequest {
            confirm: None,
            ..req
        };
        for first in [
            put("H", ControlMode::Recording, vec![]),
            move_put("H", RowsRequest::default()),
        ] {
            let (ac, store) = loaded(&[]).await;
            let out = set(&ac, unconfirmed(first.clone())).await.unwrap();
            assert!(out.harsher_because.is_empty(), "{:?}", out.harsher_because);
            assert!(out.view.record.admit_unlisted_labels);
            assert!(ac.admits_unlisted(EntityType::User, "H"));
            assert!(!ac.admits_unlisted(EntityType::Post, "H"));
            assert!(stored(&store, "user/h/H").admit_unlisted_labels);

            let (ac, _) = loaded(&[]).await;
            let off = ActionControlPutRequest {
                admit_unlisted_labels: Some(false),
                ..unconfirmed(first)
            };
            set(&ac, off).await.unwrap();
            assert!(!ac.admits_unlisted(EntityType::User, "H"));
        }

        let (ac, _) = loaded(&[record("H", Some("c"), None, ControlMode::Rules, vec![])]).await;
        set(&ac, put("H", ControlMode::Recording, vec![]))
            .await
            .unwrap();
        assert!(!ac.admits_unlisted(EntityType::User, "H"), "a known head");

        let (ac, _) = loaded(&[]).await;
        for narrow in [
            ActionControlPutRequest {
                classifier: Some("c".into()),
                ..put("H", ControlMode::Recording, vec![])
            },
            ActionControlPutRequest {
                band: Some("b".into()),
                ..put("H", ControlMode::Recording, vec![])
            },
            ActionControlPutRequest {
                classifier: Some("c".into()),
                ..move_put("H", RowsRequest::default())
            },
        ] {
            assert_eq!(
                code(set(&ac, narrow).await),
                "unknown_head_needs_head_record"
            );
        }
        assert!(resolve(&ac, "H", "c", &[]).unknown, "nothing written");

        let (ac, store) = loaded(&[]).await;
        seen_factory(&store, "F");
        let factory_classifier = ActionControlPutRequest {
            classifier: Some("c".into()),
            ..put("F", ControlMode::Recording, vec![])
        };
        let got = set(&ac, factory_classifier).await;
        assert!(got.is_ok(), "a factory head is never unknown: {got:?}");
    }

    fn stored(store: &MemControlStore, lkey: &str) -> ActionControlRecord {
        let rows = store.pkey_rows(MH_PKEY);
        let (_, v) = rows.iter().find(|(k, _)| k == lkey).expect(lkey);
        parse_record(v).unwrap()
    }

    fn skips_of(r: &ActionControlRecord) -> (Option<u32>, Option<bool>) {
        (r.follower_threshold, r.skip_high_credibility)
    }

            #[tokio::test]
    async fn the_author_skips_are_kept_when_left_out_and_cleared_by_null() {
        let more = crate::safety_checks::follower_skip_view().today_threshold as u32 * 3;
        let (ac, store) = loaded(&[]).await;
        let first = ActionControlPutRequest {
            follower_threshold: Some(Some(0)),
            skip_high_credibility: Some(Some(false)),
            ..put("H", ControlMode::Recording, vec![])
        };
        let out = set(&ac, first).await.unwrap();
        assert_eq!(skips_of(&out.view.record), (Some(0), Some(false)));
        assert_eq!(out.view.record.v, AUTHOR_SKIPS_RECORD_VERSION);
        assert_eq!(out.harsher_because.len(), 2, "{:?}", out.harsher_because);

        let leave = ActionControlPutRequest {
            expected_version: Some(1),
            ..put("H", ControlMode::Recording, vec![])
        };
        set(&ac, leave).await.unwrap();
        assert_eq!(
            skips_of(&stored(&store, "user/h/H")),
            (Some(0), Some(false))
        );

        let change = ActionControlPutRequest {
            expected_version: Some(2),
            follower_threshold: Some(Some(more)),
            ..put("H", ControlMode::Recording, vec![])
        };
        set(&ac, change).await.unwrap();
        assert_eq!(
            skips_of(&stored(&store, "user/h/H")),
            (Some(more), Some(false))
        );

        let clear = ActionControlPutRequest {
            expected_version: Some(3),
            follower_threshold: Some(None),
            ..put("H", ControlMode::Recording, vec![])
        };
        let out = set(&ac, clear).await.unwrap();
        assert_eq!(skips_of(&out.view.record), (None, Some(false)));
        assert_eq!(out.view.record.v, AUTHOR_SKIPS_RECORD_VERSION);
        let clear = ActionControlPutRequest {
            expected_version: Some(4),
            skip_high_credibility: Some(None),
            ..put("H", ControlMode::Recording, vec![])
        };
        let out = set(&ac, clear).await.unwrap();
        assert_eq!(skips_of(&out.view.record), (None, None));
        assert_eq!(
            out.view.record.v,
            xai_abuse_enforcement_types::action_control::BAND_RECORD_VERSION,
            "an older image reads it again"
        );
        assert_eq!(out.view.record.version, 5);

        let mut history: Vec<ActionControlHistoryEntry> = store
            .pkey_rows(&format!("{HISTORY_PKEY_PREFIX}H"))
            .iter()
            .map(|(_, v)| serde_json::from_slice(v).unwrap())
            .collect();
        history.sort_by_key(|e| e.version);
        let trail: Vec<_> = history
            .iter()
            .map(|e| (e.before.as_ref().map(skips_of), skips_of(&e.after)))
            .collect();
        assert_eq!(
            trail,
            [
                (None, (Some(0), Some(false))),
                (Some((Some(0), Some(false))), (Some(0), Some(false))),
                (Some((Some(0), Some(false))), (Some(more), Some(false))),
                (Some((Some(more), Some(false))), (None, Some(false))),
                (Some((None, Some(false))), (None, None)),
            ]
        );
    }

                #[tokio::test]
    async fn lifting_a_skip_needs_the_confirm_on_every_head_record() {
        let d = crate::safety_checks::follower_skip_view().today_threshold as u32;
        let (ac, store) = loaded(&[]).await;
        for (head, mode, actions) in [
            ("Live", ControlMode::Live, vec![SUSPEND]),
            ("Recording", ControlMode::Recording, vec![]),
            ("Rules", ControlMode::Rules, vec![]),
        ] {
            set(&ac, put(head, mode, actions.clone())).await.unwrap();
            let change =
                |version: u64, threshold: Option<Option<u32>>, cred: Option<Option<bool>>| {
                    ActionControlPutRequest {
                        expected_version: Some(version),
                        confirm: None,
                        follower_threshold: threshold,
                        skip_high_credibility: cred,
                        ..put(head, mode, actions.clone())
                    }
                };
            match set(&ac, change(1, Some(Some(0)), None)).await {
                Err(SetError::Invalid("confirm_required", msg)) => assert!(
                    msg.contains(&format!("skips fewer high-follower accounts ({d}+ to off)")),
                    "{head}: {msg}"
                ),
                other => panic!("{head}: {other:?}"),
            }
            match set(&ac, change(1, None, Some(Some(false)))).await {
                Err(SetError::Invalid("confirm_required", msg)) => assert!(
                    msg.contains("stops skipping high-credibility accounts"),
                    "{head}: {msg}"
                ),
                other => panic!("{head}: {other:?}"),
            }
            assert_eq!(
                code(set(&ac, change(1, Some(Some(d * 5)), None)).await),
                "confirm_required",
                "{head}: from today's count to a higher one"
            );

            let same = set(&ac, change(1, Some(Some(d)), None)).await.unwrap();
            assert!(same.harsher_because.is_empty(), "{head}: today's count");
            let kept = set(&ac, change(2, None, None)).await.unwrap();
            assert!(kept.harsher_because.is_empty(), "{head}: left out");
            let confirmed = ActionControlPutRequest {
                confirm: Some(CONFIRM_WORD.into()),
                ..change(3, Some(Some(d * 5)), Some(Some(false)))
            };
            let out = set(&ac, confirmed).await.unwrap();
            assert_eq!(
                out.harsher_because.len(),
                2,
                "{head}: {:?}",
                out.harsher_because
            );
            for (version, threshold, cred) in [
                (4, Some(Some(d * 2)), None),
                (5, Some(None), None),
                (6, None, Some(Some(true))),
                (7, Some(Some(0)), None),
            ] {
                let out = set(&ac, change(version, threshold, cred)).await;
                if version == 7 {
                    assert_eq!(code(out), "confirm_required", "{head}: back to off");
                    continue;
                }
                let out = out.unwrap();
                assert!(
                    out.harsher_because.is_empty(),
                    "{head} v{version}: {:?}",
                    out.harsher_because
                );
            }
            assert_eq!(
                skips_of(&stored(&store, &format!("user/h/{head}"))),
                (None, Some(true))
            );
        }
        let to_zero = ActionControlPutRequest {
            confirm: None,
            follower_threshold: Some(Some(0)),
            skip_high_credibility: Some(Some(false)),
            ..put("Fresh", ControlMode::Rules, vec![])
        };
        set(&ac, to_zero.clone()).await.unwrap_err();
        let stricter = ActionControlPutRequest {
            expected_version: Some(1),
            confirm: None,
            follower_threshold: Some(Some(d)),
            skip_high_credibility: Some(Some(true)),
            ..put("Fresh", ControlMode::Rules, vec![])
        };
        set(
            &ac,
            ActionControlPutRequest {
                confirm: Some(CONFIRM_WORD.into()),
                ..to_zero
            },
        )
        .await
        .unwrap();
        let out = set(&ac, stricter).await.unwrap();
        assert!(
            out.harsher_because.is_empty(),
            "from off back to today's skips"
        );
    }

            #[tokio::test]
    async fn lifting_a_skip_on_a_renamed_heads_old_record_needs_the_confirm() {
        let mut cfg = on();
        cfg.aliases.insert("NewName".into(), "OldName".into());
        let safety = baked_safety();
        for shadowed in [false, true] {
            let (ac, _) = loaded(&[]).await;
            let save = |req: ActionControlPutRequest| {
                ac.set(req, "operator@example.com", "pv2", NOW_MS, &cfg, &safety)
            };
            if shadowed {
                save(put("NewName", ControlMode::Rules, vec![]))
                    .await
                    .unwrap();
            }
            save(put("OldName", ControlMode::Rules, vec![]))
                .await
                .unwrap();
            let lift = ActionControlPutRequest {
                expected_version: Some(1),
                confirm: None,
                follower_threshold: Some(Some(0)),
                ..put("OldName", ControlMode::Rules, vec![])
            };
            assert_eq!(
                code(save(lift.clone()).await),
                "confirm_required",
                "{shadowed}"
            );
            let confirmed = ActionControlPutRequest {
                confirm: Some(CONFIRM_WORD.into()),
                ..lift
            };
            save(confirmed).await.unwrap();
        }
    }

    #[tokio::test]
    async fn a_move_carries_the_author_skips_on_the_head_record_only() {
        let (ac, store) = loaded(&[]).await;
        let off = ActionControlPutRequest {
            confirm: None,
            follower_threshold: Some(Some(0)),
            ..move_put("H", all_rows(ControlMode::Recording, vec![]))
        };
        assert_eq!(
            code(set(&ac, off.clone()).await),
            "confirm_required",
            "even with every row recording only"
        );
        let off = ActionControlPutRequest {
            confirm: Some(CONFIRM_WORD.into()),
            ..off
        };
        set(&ac, off).await.unwrap();
        assert_eq!(skips_of(&stored(&store, "user/h/H")), (Some(0), None));
        assert_eq!(stored(&store, "user/h/H").v, AUTHOR_SKIPS_RECORD_VERSION);
        for row in LevelRow::ALL {
            let r = stored(&store, &format!("user/l/H/{}", row.as_str()));
            assert_eq!(skips_of(&r), (None, None));
            assert_eq!(r.v, ROWS_RECORD_VERSION);
        }

        let high = ActionControlPutRequest {
            expected_version: Some(1),
            ..row_put("H", LevelRow::High, ControlMode::Live, vec![SUSPEND])
        };
        set(&ac, high).await.unwrap();
        assert_eq!(
            skips_of(&stored(&store, "user/h/H")),
            (Some(0), None),
            "a row's save leaves the head record alone"
        );

        let with_versions = |mode: ControlMode, actions: Vec<PageAction>, v: u64| RowSetting {
            expected_version: Some(v),
            ..setting(mode, actions)
        };
        let cred = ActionControlPutRequest {
            expected_version: Some(1),
            confirm: None,
            skip_high_credibility: Some(Some(false)),
            ..move_put(
                "H",
                RowsRequest {
                    low: Some(with_versions(ControlMode::Recording, vec![], 1)),
                    mid: Some(with_versions(ControlMode::Recording, vec![], 1)),
                    high: Some(with_versions(ControlMode::Live, vec![SUSPEND], 2)),
                    missing: Some(with_versions(ControlMode::Recording, vec![], 1)),
                },
            )
        };
        match set(&ac, cred.clone()).await {
            Err(SetError::Invalid("confirm_required", msg)) => {
                assert!(
                    msg.contains("stops skipping high-credibility accounts"),
                    "{msg}"
                )
            }
            other => panic!("{other:?}"),
        }
        let confirmed = ActionControlPutRequest {
            confirm: Some(CONFIRM_WORD.into()),
            ..cred
        };
        set(&ac, confirmed).await.unwrap();
        assert_eq!(
            skips_of(&stored(&store, "user/h/H")),
            (Some(0), Some(false)),
            "the threshold left out is kept"
        );
    }

    #[tokio::test]
    async fn the_author_skips_refusals() {
        let ceiling = crate::safety_checks::follower_ceiling() as u32;
        let today = crate::safety_checks::follower_skip_view().today_threshold as u32;
        let (ac, store) = loaded(&[]).await;
        let at = |threshold: u32| ActionControlPutRequest {
            follower_threshold: Some(Some(threshold)),
            ..put("H", ControlMode::Recording, vec![])
        };
        for below in [1, today / 2, today - 1] {
            assert_eq!(
                code(set(&ac, at(below)).await),
                "follower_threshold_below_today",
                "{below}"
            );
        }
        assert_eq!(
            code(set(&ac, at(ceiling)).await),
            "follower_threshold_too_high"
        );
        assert_eq!(
            code(set(&ac, at(u32::MAX)).await),
            "follower_threshold_too_high"
        );
        set(&ac, at(ceiling - 1)).await.unwrap();
        let v = |n: u64| ActionControlPutRequest {
            expected_version: Some(n),
            ..at(today)
        };
        set(&ac, v(1)).await.unwrap();
        let off = ActionControlPutRequest {
            follower_threshold: Some(Some(0)),
            ..v(2)
        };
        set(&ac, off).await.unwrap();

        for narrow in [
            ActionControlPutRequest {
                classifier: Some("c".into()),
                follower_threshold: Some(Some(0)),
                ..put("H2", ControlMode::Recording, vec![])
            },
            ActionControlPutRequest {
                band: Some("b".into()),
                skip_high_credibility: Some(Some(false)),
                ..put("H2", ControlMode::Recording, vec![])
            },
            ActionControlPutRequest {
                follower_threshold: Some(None),
                ..row_put("H2", LevelRow::High, ControlMode::Recording, vec![])
            },
        ] {
            assert_eq!(code(set(&ac, narrow).await), "author_skips_scope");
        }

        seen_factory(&store, "Factory");
        for factory in [
            ActionControlPutRequest {
                follower_threshold: Some(Some(0)),
                ..move_put("Factory", RowsRequest::default())
            },
            ActionControlPutRequest {
                skip_high_credibility: Some(None),
                ..move_put("Factory", RowsRequest::default())
            },
            ActionControlPutRequest {
                follower_threshold: Some(Some(1)),
                ..move_put("Factory", RowsRequest::default())
            },
            ActionControlPutRequest {
                follower_threshold: Some(Some(ceiling)),
                ..move_put("Factory", RowsRequest::default())
            },
        ] {
            assert_eq!(code(set(&ac, factory).await), "factory_author_skips");
        }
    }

                #[tokio::test]
    async fn a_decision_takes_the_head_records_skips() {
        let holder = |head: &str| ActionControlRecord {
            follower_threshold: Some(0),
            skip_high_credibility: Some(false),
            ..record(head, None, None, ControlMode::Rules, vec![])
        };
        let (ac, _) = loaded(&[
            holder("H"),
            live("H", Some("c"), None, vec![SUSPEND]),
            live("Plain", None, None, vec![SUSPEND]),
            holder("OldName"),
            live("NewName", Some("c"), None, vec![SUSPEND]),
        ])
        .await;
        let want = AuthorSkips {
            follower_threshold: Some(0),
            skip_high_credibility: Some(false),
        };
        let r = resolve(&ac, "H", "c", &[]);
        assert!(r.page_driven());
        assert_eq!(r.author_skips, want);
        assert!(
            r.info_keys()
                .contains(&(FOLLOWER_THRESHOLD_KEY.into(), "0".into()))
        );
        assert!(
            r.info_keys()
                .contains(&(SKIP_HIGH_CREDIBILITY_KEY.into(), "false".into()))
        );
        let rules = resolve(&ac, "H", "other", &[]);
        assert!(!rules.page_driven());
        assert!(
            !rules
                .info_keys()
                .iter()
                .any(|(k, _)| k == FOLLOWER_THRESHOLD_KEY)
        );
        let plain = resolve(&ac, "Plain", "c", &[]);
        assert_eq!(plain.author_skips, AuthorSkips::default());
        assert!(
            !plain
                .info_keys()
                .iter()
                .any(|(k, _)| k == FOLLOWER_THRESHOLD_KEY)
        );

        let mut cfg = on();
        cfg.aliases.insert("NewName".into(), "OldName".into());
        let ls = labels(&[]);
        let renamed = ac.resolve(&input("NewName", "c", &ls), &cfg, |_| true);
        assert_eq!(renamed.author_skips, want);

        let factory = ac.resolve(
            &ResolveInput {
                factory: true,
                ..input("H", "c", &ls)
            },
            &on(),
            |_| true,
        );
        assert_eq!(factory.author_skips, AuthorSkips::default());
    }

    #[tokio::test]
    async fn the_list_says_the_service_takes_the_skips_and_their_default() {
        let (ac, _) = loaded(&[]).await;
        let list = ac.list(&on(), baked_safety()).await;
        assert_eq!(list.feature_version, 6);
        let view = list.follower_skip.unwrap();
        assert_eq!(view, crate::safety_checks::follower_skip_view());
        assert!(view.today_threshold < view.ceiling);
    }

    #[tokio::test]
    async fn a_move_writes_four_rows_then_the_scope_and_a_left_out_row_records() {
        let (ac, store) = loaded(&[]).await;
        let out = set(
            &ac,
            move_put(
                "H",
                RowsRequest {
                    high: Some(setting(ControlMode::Live, vec![SUSPEND])),
                    low: Some(setting(ControlMode::Rules, vec![])),
                    ..Default::default()
                },
            ),
        )
        .await
        .unwrap();
        assert_eq!(out.view.lkey, "user/h/H");
        assert_eq!(out.view.effective_mode, ControlMode::Rows);
        assert_eq!(out.view.record.v, ROWS_RECORD_VERSION);
        let rows = store.pkey_rows(MH_PKEY);
        let mode_at = |lkey: &str| {
            rows.iter()
                .find(|(k, _)| k == lkey)
                .map(|(_, v)| parse_record(v).unwrap().mode)
        };
        assert_eq!(mode_at("user/h/H"), Some(ControlMode::Rows));
        assert_eq!(mode_at("user/l/H/high"), Some(ControlMode::Live));
        assert_eq!(mode_at("user/l/H/low"), Some(ControlMode::Rules));
        assert_eq!(mode_at("user/l/H/mid"), Some(ControlMode::Recording));
        assert_eq!(mode_at("user/l/H/missing"), Some(ControlMode::Recording));
        for (k, v) in rows.iter().filter(|(k, _)| k.starts_with("user/l/H/")) {
            assert_eq!(
                parse_record(v).unwrap().v,
                3,
                "{k}: images reading v: 2 level records refuse a row"
            );
        }
        assert_eq!(
            store.pkey_rows(&format!("{HISTORY_PKEY_PREFIX}H")).len(),
            5,
            "one history entry per record written"
        );
        let r = ac.resolve(
            &ResolveInput {
                confidence: Some(high_claim()),
                ..input("H", "clf", &[])
            },
            &on(),
            |_| true,
        );
        assert_eq!(r.mode, ControlMode::Live, "applied on this pod at once");
        assert_eq!(r.page_actions, vec![SUSPEND]);
    }

    #[tokio::test]
    async fn row_put_validation_codes() {
        let (ac, _) = loaded(&[record("H", None, None, ControlMode::Rows, vec![])]).await;
        let edit = |row, mode, actions| ActionControlPutRequest {
            expected_version: None,
            ..row_put("H", row, mode, actions)
        };
        let cases: Vec<(ActionControlPutRequest, &str)> = vec![
            (
                ActionControlPutRequest {
                    band: Some("b".into()),
                    ..edit(LevelRow::High, ControlMode::Recording, vec![])
                },
                "invalid_scope",
            ),
            (
                ActionControlPutRequest {
                    band: Some("b".into()),
                    ..put("H", ControlMode::Rows, vec![])
                },
                "invalid_scope",
            ),
            (
                put("H", ControlMode::Rows, vec![SUSPEND]),
                "rows_take_no_actions",
            ),
            (
                ActionControlPutRequest {
                    rows: Some(RowsRequest::default()),
                    ..put("H", ControlMode::Live, vec![SUSPEND])
                },
                "rows_only_with_mode_rows",
            ),
            (
                edit(LevelRow::High, ControlMode::Rows, vec![]),
                "invalid_scope",
            ),
            (
                move_put(
                    "H",
                    RowsRequest {
                        high: Some(setting(ControlMode::Rows, vec![])),
                        ..Default::default()
                    },
                ),
                "invalid_row",
            ),
            (
                move_put(
                    "H",
                    RowsRequest {
                        mid: Some(setting(ControlMode::Live, vec![])),
                        ..Default::default()
                    },
                ),
                "live_needs_actions",
            ),
            (
                move_put(
                    "H",
                    RowsRequest {
                        missing: Some(RowSetting {
                            evidence: Some(read(0.96, 59, 0)),
                            ..setting(ControlMode::Live, vec![SUSPEND])
                        }),
                        ..Default::default()
                    },
                ),
                "invalid_evidence",
            ),
            (
                ActionControlPutRequest {
                    evidence: Some(read(0.96, 59, 0)),
                    ..put("H", ControlMode::Live, vec![SUSPEND])
                },
                "invalid_evidence",
            ),
            (
                ActionControlPutRequest {
                    evidence: Some(read(0.96, 5, 6)),
                    ..edit(LevelRow::High, ControlMode::Live, vec![SUSPEND])
                },
                "invalid_evidence",
            ),
            (
                ActionControlPutRequest {
                    evidence_override: Some(person(PUT_NOW + 86_400_000)),
                    ..edit(LevelRow::High, ControlMode::Recording, vec![])
                },
                "invalid_override",
            ),
            (
                ActionControlPutRequest {
                    evidence_override: Some(person(PUT_NOW + 86_400_000)),
                    ..edit(LevelRow::High, ControlMode::Live, vec![SUSPEND])
                },
                "invalid_override",
            ),
            (
                move_put(
                    "H",
                    RowsRequest {
                        mid: Some(RowSetting {
                            evidence_override: Some(person(PUT_NOW + 86_400_000)),
                            ..setting(ControlMode::Live, vec![SUSPEND])
                        }),
                        ..Default::default()
                    },
                ),
                "invalid_override",
            ),
            (
                ActionControlPutRequest {
                    evidence_override: Some(person(PUT_NOW + 86_400_000)),
                    ..edit(LevelRow::Missing, ControlMode::Live, vec![SUSPEND])
                },
                "invalid_override",
            ),
            (
                row_put("Unmoved", LevelRow::High, ControlMode::Live, vec![SUSPEND]),
                "scope_not_on_rows",
            ),
        ];
        for (req, want) in cases {
            let label = format!("{req:?}");
            assert_eq!(code(set(&ac, req).await), want, "{label}");
        }
    }

                    #[tokio::test]
    async fn factory_rows_are_judged_for_enforce_as_they_decide() {
        let (ac, store) = loaded(&[row_rec("F", LevelRow::High, ControlMode::Rules, vec![])]).await;
        seen_factory(&store, "F");
        let unconfirmed = |row: LevelRow, actions: Vec<PageAction>, version: Option<u64>| {
            ActionControlPutRequest {
                confirm: None,
                expected_version: version,
                ..row_put("F", row, ControlMode::Live, actions)
            }
        };
        let got = set(&ac, unconfirmed(LevelRow::High, vec![SUSPEND], Some(1))).await;
        assert_eq!(code(got), "confirm_required", "a stored rules row");
        let got = set(
            &ac,
            unconfirmed(LevelRow::Mid, vec![SUSPEND, LIVENESS], None),
        )
        .await;
        assert_eq!(code(got), "confirm_required", "an unset row");
        set(
            &ac,
            row_put(
                "F",
                LevelRow::Mid,
                ControlMode::Live,
                vec![SUSPEND, LIVENESS],
            ),
        )
        .await
        .unwrap();
        let got = set(&ac, unconfirmed(LevelRow::Mid, vec![SUSPEND], Some(1))).await;
        assert!(got.is_ok(), "dropping an action: {got:?}");
        let got = set(
            &ac,
            unconfirmed(LevelRow::Mid, vec![SUSPEND, SHR7], Some(2)),
        )
        .await;
        assert_eq!(code(got), "confirm_required", "adding an action");
        let unreadable = ControlConfig {
            confidence: crate::confidence::ConfidenceConfig::from_value(Some(
                &serde_json::json!({"label_map": {}}),
            )),
            ..on()
        };
        let got = put_with(
            &ac,
            &unreadable,
            unconfirmed(LevelRow::Mid, vec![SUSPEND], Some(2)),
        )
        .await;
        assert!(got.is_ok(), "unreadable confidence config: {got:?}");
    }

                #[tokio::test]
    async fn a_factory_classifier_leaving_rows_is_judged_against_the_head_rows() {
        let (ac, store) = loaded(&[
            row_rec("F", LevelRow::High, ControlMode::Live, vec![SUSPEND]),
            record("F", Some("c1"), None, ControlMode::Rows, vec![]),
        ])
        .await;
        seen_factory(&store, "F");
        let leave = |confirm: Option<String>| ActionControlPutRequest {
            classifier: Some("c1".into()),
            expected_version: Some(1),
            confirm,
            ..put("F", ControlMode::Recording, vec![])
        };
        let got = set(&ac, leave(None)).await;
        assert_eq!(code(got), "confirm_required");
        let got = set(&ac, leave(Some(CONFIRM_WORD.into()))).await;
        assert!(got.is_ok(), "{got:?}");
    }

            #[tokio::test]
    async fn an_unreadable_factory_classifier_is_judged_as_recording() {
        let (ac, store) = loaded(&[row_rec(
            "F",
            LevelRow::High,
            ControlMode::Live,
            vec![SUSPEND],
        )])
        .await;
        store.put_raw("user/c/F/bad", b"{not a record");
        ac.refresh().await.unwrap();
        seen_factory(&store, "F");
        let mv = |confirm: Option<String>| ActionControlPutRequest {
            classifier: Some("bad".into()),
            expected_version: Some(0),
            confirm,
            ..move_put(
                "F",
                RowsRequest {
                    high: Some(setting(ControlMode::Live, vec![SUSPEND])),
                    ..all_rows(ControlMode::Recording, vec![])
                },
            )
        };
        let got = set(&ac, mv(None)).await;
        assert_eq!(code(got), "confirm_required");
        let got = set(&ac, mv(Some(CONFIRM_WORD.into()))).await;
        assert!(got.is_ok(), "{got:?}");
    }

            #[tokio::test]
    async fn a_band_record_never_holds_back_a_factory_classifier_move() {
        let (ac, store) = loaded(&[live("F", None, Some("old_band"), vec![SUSPEND])]).await;
        let mv = ActionControlPutRequest {
            classifier: Some("c".into()),
            ..move_put("F", all_rows(ControlMode::Recording, vec![]))
        };
        assert_eq!(
            code(set(&ac, mv.clone()).await),
            "band_settings_not_rules",
            "any other head"
        );
        seen_factory(&store, "F");
        let got = set(&ac, mv).await;
        assert!(got.is_ok(), "{got:?}");
    }

    #[tokio::test]
    async fn factory_heads_move_without_rules_rows_and_child_safety_heads_are_set_like_any_other() {
        let (ac, store) = loaded(&[]).await;
        let live_high = || RowsRequest {
            high: Some(setting(ControlMode::Live, vec![SUSPEND])),
            ..all_rules()
        };
        let out = set(&ac, move_put(CHILD_SAFETY_HEAD, live_high()))
            .await
            .unwrap();
        assert_eq!(out.view.effective_mode, ControlMode::Rows);
        assert_eq!(
            out.harsher_because,
            vec![
                "low row: hands the scope back to the GrowthBook rules, which act",
                "mid row: hands the scope back to the GrowthBook rules, which act",
                "high row: turns the scope live",
                "missing row: hands the scope back to the GrowthBook rules, which act",
            ],
            "judged like any other head no record names"
        );
        let mut cfg = on();
        cfg.aliases
            .insert("Renamed".into(), CHILD_SAFETY_HEAD.into());
        assert!(
            put_with(&ac, &cfg, move_put("Renamed", live_high()))
                .await
                .is_ok(),
            "a rename onto a child-safety head"
        );
        set(&ac, put("CseRtScoreHiV2DryRun", ControlMode::Rules, vec![]))
            .await
            .unwrap();
        let label_row = ActionControlPutRequest {
            classifier: Some("c".into()),
            ..move_put("CseRtScoreHiV2DryRun", all_rules())
        };
        assert!(set(&ac, label_row).await.is_ok());
        for (head, factory, refused) in [("SentCse", false, false), ("Factory", true, true)] {
            let s = ActionControlSeen {
                head: head.into(),
                classifier: "c".into(),
                band: "anchor_campaign_suspend_cse".into(),
                entity_type: ControlEntity::User,
                factory,
                last_seen_ms: 1,
                rule_id: String::new(),
                action_specs: serde_json::json!([]),
            };
            store.rows.lock().unwrap().insert(
                (SEEN_PKEY.into(), seen_lkey(&s)),
                serde_json::to_vec(&s).unwrap(),
            );
            let got = set(&ac, move_put(head, live_high())).await;
            if refused {
                assert_eq!(code(got), "factory_rules", "{head}");
                let recording_rows = RowsRequest {
                    high: Some(setting(ControlMode::Live, vec![SUSPEND])),
                    ..all_rows(ControlMode::Recording, vec![])
                };
                let got = set(&ac, move_put(head, recording_rows)).await;
                assert!(got.is_ok(), "{head}: {got:?}");
            } else {
                assert!(got.is_ok(), "{head}: {got:?}");
            }
        }
        let edit = ActionControlPutRequest {
            expected_version: Some(1),
            ..row_put(
                CHILD_SAFETY_HEAD,
                LevelRow::Mid,
                ControlMode::Live,
                vec![LIVENESS],
            )
        };
        assert!(set(&ac, edit).await.is_ok());
        let whole = put(
            "NearDupEmbeddingCseNcmecMatch",
            ControlMode::Live,
            vec![SUSPEND],
        );
        assert!(set(&ac, whole).await.is_ok());
        let r = resolve(&ac, "NearDupEmbeddingCseNcmecMatch", "clf", &[]);
        assert_eq!((r.mode, r.forced), (ControlMode::Live, None));
        assert_eq!(r.page_actions, vec![SUSPEND]);
        let r = ac.resolve(
            &ResolveInput {
                confidence: Some(high_claim()),
                ..input(CHILD_SAFETY_HEAD, "clf", &[])
            },
            &on(),
            |_| true,
        );
        assert_eq!((r.mode, r.forced), (ControlMode::Live, None));
        assert_eq!(r.page_actions, vec![SUSPEND]);
        let list = ac.list(&on(), baked_safety()).await;
        let high = list
            .records
            .iter()
            .find(|v| v.lkey == format!("user/l/{CHILD_SAFETY_HEAD}/high"))
            .unwrap();
        assert_eq!(high.effective_mode, ControlMode::Live);
        assert_eq!(high.forced_recording_reason, None);
    }

    #[tokio::test]
    async fn moving_a_live_record_to_live_rows_is_judged_by_what_it_does_for_every_head() {
        let move_live = |head: &str, confirm: Option<&str>| ActionControlPutRequest {
            expected_version: Some(1),
            confirm: confirm.map(Into::into),
            ..move_put(head, all_rows(ControlMode::Live, vec![SUSPEND]))
        };
        for head in ["H", CHILD_SAFETY_HEAD] {
            let stored = [record(head, None, None, ControlMode::Live, vec![SUSPEND])];
            let (ac, _) = loaded(&stored).await;
            let ls = labels(&["anchor_campaign_suspend_cse"]);
            let before = ac.resolve(&input(head, "c", &ls), &on(), |_| true);
            assert_eq!(
                (before.mode, before.forced),
                (ControlMode::Live, None),
                "{head}"
            );
            let out = set(&ac, move_live(head, None)).await;
            assert!(
                out.as_ref().is_ok_and(|o| o.harsher_because.is_empty()),
                "{head}: {out:?}"
            );
            let after = ac.resolve(
                &ResolveInput {
                    confidence: Some(high_claim()),
                    ..input(head, "c", &ls)
                },
                &on(),
                |_| true,
            );
            assert_eq!(
                (after.mode, after.forced),
                (ControlMode::Live, None),
                "{head}"
            );

            let (ac, _) = loaded(&stored).await;
            let killed = ControlConfig {
                enabled: false,
                ..Default::default()
            };
            assert_eq!(
                code(put_with(&ac, &killed, move_live(head, None)).await),
                "confirm_required",
                "{head}"
            );
        }
    }

    #[tokio::test]
    async fn a_child_safety_row_back_to_the_rules_is_judged_like_any_other() {
        for (row_mode, actions, harsher) in [
            (ControlMode::Live, vec![SUSPEND], false),
            (ControlMode::Recording, vec![], true),
        ] {
            let (ac, store) = loaded(&[
                record(CHILD_SAFETY_HEAD, None, None, ControlMode::Rows, vec![]),
                ActionControlRecord {
                    evidence: Some(Box::new(read(0.96, 59, 0))),
                    ..row_rec(CHILD_SAFETY_HEAD, LevelRow::High, row_mode, actions)
                },
            ])
            .await;
            let s = ActionControlSeen {
                head: CHILD_SAFETY_HEAD.into(),
                classifier: "c".into(),
                band: "-".into(),
                entity_type: ControlEntity::User,
                factory: false,
                last_seen_ms: 1,
                rule_id: "act_x".into(),
                action_specs: serde_json::json!([
                    {"kind": "suspend", "permanent": false, "policy": "PlatformManipulation"}
                ]),
            };
            store.rows.lock().unwrap().insert(
                (SEEN_PKEY.into(), seen_lkey(&s)),
                serde_json::to_vec(&s).unwrap(),
            );
            let back = ActionControlPutRequest {
                expected_version: Some(1),
                confirm: None,
                ..row_put(
                    CHILD_SAFETY_HEAD,
                    LevelRow::High,
                    ControlMode::Rules,
                    vec![],
                )
            };
            let got = set(&ac, back).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "{row_mode:?}");
            } else {
                assert!(got.is_ok(), "{row_mode:?}: {got:?}");
            }
        }
    }

        const CHILD_SAFETY_HEAD: &str = "NearDupEmbeddingCseMatch";

    fn all_rules() -> RowsRequest {
        all_rows(ControlMode::Rules, vec![])
    }

    #[tokio::test]
    async fn the_move_is_refused_while_a_band_setting_under_it_is_not_rules() {
        for (taken, refused) in [
            (
                record("H", None, Some("b1"), ControlMode::Live, vec![LIVENESS]),
                true,
            ),
            (
                record("H", Some("c1"), Some("b1"), ControlMode::Recording, vec![]),
                true,
            ),
            (
                record("H", None, Some("b1"), ControlMode::Rules, vec![]),
                false,
            ),
            (
                record("H", Some("c1"), None, ControlMode::Live, vec![SUSPEND]),
                false,
            ),
        ] {
            let label = format!("{} {:?}", record_lkey(&taken), taken.mode);
            let (ac, store) = loaded(&[taken]).await;
            let got = set(&ac, move_put("H", all_rules())).await;
            if refused {
                match got {
                    Err(SetError::Invalid("band_settings_not_rules", msg)) => {
                        assert!(msg.contains("/H/"), "{label}: names the setting: {msg}")
                    }
                    other => panic!("{label}: {other:?}"),
                }
                assert_eq!(
                    store.pkey_rows(MH_PKEY).len(),
                    1,
                    "{label}: nothing written"
                );
            } else {
                assert!(got.is_ok(), "{label}: {got:?}");
            }
        }
        let (ac, store) = loaded(&[]).await;
        store.put_raw("user/b/H/b1", b"{not json");
        ac.refresh().await.unwrap();
        assert_eq!(
            code(set(&ac, move_put("H", all_rules())).await),
            "band_settings_not_rules"
        );
        let (ac, _) = loaded(&[record("H", None, None, ControlMode::Rows, vec![])]).await;
        let band = |mode, actions| ActionControlPutRequest {
            band: Some("b1".into()),
            ..put("H", mode, actions)
        };
        assert_eq!(
            code(set(&ac, band(ControlMode::Live, vec![SUSPEND])).await),
            "scope_on_rows"
        );
        assert!(set(&ac, band(ControlMode::Rules, vec![])).await.is_ok());
    }

    #[tokio::test]
    async fn a_move_with_every_row_rules_needs_no_confirmation_and_live_rows_do() {
        let unconfirmed = |rows| ActionControlPutRequest {
            confirm: None,
            ..move_put("H", rows)
        };
        let (ac, _) = loaded(&[record("H", None, None, ControlMode::Rules, vec![])]).await;
        let req = ActionControlPutRequest {
            expected_version: Some(1),
            ..unconfirmed(all_rules())
        };
        let out = set(&ac, req).await.unwrap();
        assert!(out.harsher_because.is_empty(), "{:?}", out.harsher_because);

        let (ac, _) = loaded(&[]).await;
        assert_eq!(
            code(set(&ac, unconfirmed(all_rules())).await),
            "confirm_required"
        );

        let (ac, _) = loaded(&[]).await;
        assert!(
            set(&ac, unconfirmed(RowsRequest::default())).await.is_ok(),
            "every row recording only is never harsher"
        );

        let (ac, _) = loaded(&[]).await;
        let live_high = RowsRequest {
            high: Some(setting(ControlMode::Live, vec![SUSPEND])),
            ..all_rules()
        };
        match set(&ac, unconfirmed(live_high.clone())).await {
            Err(SetError::Invalid("confirm_required", msg)) => {
                assert!(msg.contains("high row: turns the scope live"), "{msg}")
            }
            other => panic!("{other:?}"),
        }
        assert!(set(&ac, move_put("H", live_high)).await.is_ok());

        let (ac, _) = loaded(&[record("H", None, None, ControlMode::Recording, vec![])]).await;
        let req = ActionControlPutRequest {
            expected_version: Some(1),
            ..unconfirmed(all_rules())
        };
        assert_eq!(code(set(&ac, req).await), "confirm_required");

        let (ac, _) = loaded(&[
            record("H", None, None, ControlMode::Live, vec![SUSPEND]),
            record("H", None, Some("b1"), ControlMode::Rules, vec![]),
        ])
        .await;
        let req = ActionControlPutRequest {
            expected_version: Some(1),
            ..unconfirmed(all_rows(ControlMode::Live, vec![SUSPEND]))
        };
        assert_eq!(code(set(&ac, req).await), "confirm_required");
    }

    #[tokio::test]
    async fn typed_enforce_uses_what_a_row_does_right_now() {
        let stored = || {
            vec![
                record("H", None, None, ControlMode::Rows, vec![]),
                ActionControlRecord {
                    evidence: Some(Box::new(read(0.96, 59, 0))),
                    ..row_rec("H", LevelRow::High, ControlMode::Live, vec![SUSPEND])
                },
            ]
        };
        let covered_seen = |store: &MemControlStore| {
            let s = ActionControlSeen {
                head: "H".into(),
                classifier: "c".into(),
                band: "-".into(),
                entity_type: ControlEntity::User,
                factory: false,
                last_seen_ms: 1,
                rule_id: "act_x".into(),
                action_specs: serde_json::json!([
                    {"kind": "suspend", "permanent": false, "policy": "PlatformManipulation"}
                ]),
            };
            store.rows.lock().unwrap().insert(
                (SEEN_PKEY.into(), seen_lkey(&s)),
                serde_json::to_vec(&s).unwrap(),
            );
        };
        let back = |confirm: Option<&str>| ActionControlPutRequest {
            expected_version: Some(1),
            confirm: confirm.map(Into::into),
            ..row_put("H", LevelRow::High, ControlMode::Rules, vec![])
        };

        let (ac, store) = loaded(&stored()).await;
        covered_seen(&store);
        let out = set(&ac, back(None)).await.unwrap();
        assert!(out.harsher_because.is_empty(), "{:?}", out.harsher_because);

        let (ac, store) = loaded(&stored()).await;
        covered_seen(&store);
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };
        assert_eq!(
            code(put_with(&ac, &killed, back(None)).await),
            "confirm_required"
        );
        assert!(
            put_with(&ac, &killed, back(Some(CONFIRM_WORD)))
                .await
                .is_ok()
        );

        let (ac, store) = loaded(&stored()).await;
        covered_seen(&store);
        ac.age_for_test(STALE_AFTER + Duration::from_secs(1));
        assert_eq!(code(set(&ac, back(None)).await), "confirm_required");

        let mut weak = stored();
        weak[1].evidence = Some(Box::new(read(0.88, 50, 2)));
        let (ac, store) = loaded(&weak).await;
        covered_seen(&store);
        let out = set(&ac, back(None)).await.unwrap();
        assert!(out.harsher_because.is_empty(), "{:?}", out.harsher_because);

        let (ac, _) = loaded(&stored()).await;
        let mid_live = ActionControlPutRequest {
            confirm: None,
            ..row_put("H", LevelRow::Mid, ControlMode::Live, vec![LIVENESS])
        };
        assert_eq!(code(set(&ac, mid_live).await), "confirm_required");
        let high_more = ActionControlPutRequest {
            expected_version: Some(1),
            confirm: None,
            evidence: Some(read(0.96, 59, 0)),
            ..row_put("H", LevelRow::High, ControlMode::Live, vec![SUSPEND, SHR7])
        };
        assert_eq!(code(set(&ac, high_more).await), "confirm_required");

        let (ac, _) = loaded(&[record("H", None, None, ControlMode::Rows, vec![])]).await;
        let leave = ActionControlPutRequest {
            expected_version: Some(1),
            confirm: None,
            ..put("H", ControlMode::Rules, vec![])
        };
        assert_eq!(code(set(&ac, leave.clone()).await), "confirm_required");
        let mut every_rules = vec![record("H", None, None, ControlMode::Rows, vec![])];
        every_rules.extend(LevelRow::ALL.map(|r| row_rec("H", r, ControlMode::Rules, vec![])));
        let (ac, _) = loaded(&every_rules).await;
        assert!(
            set(&ac, leave).await.is_ok(),
            "every row already follows the rules"
        );
    }

            struct HoldingStore {
        inner: MemControlStore,
        marker_scans: AtomicUsize,
        first_checked: tokio::sync::Notify,
        second_done: tokio::sync::Notify,
    }

    #[async_trait]
    impl ControlStore for HoldingStore {
        async fn scan(&self, pkey: &str) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
            let rows = self.inner.scan(pkey).await?;
            if pkey.starts_with(SAVE_PKEY_PREFIX)
                && self.marker_scans.fetch_add(1, Ordering::SeqCst) == 0
            {
                self.first_checked.notify_one();
                self.second_done.notified().await;
            }
            Ok(rows)
        }

        async fn get(&self, pkey: &str, lkey: &str) -> anyhow::Result<Option<Vec<u8>>> {
            self.inner.get(pkey, lkey).await
        }

        async fn put(
            &self,
            pkey: &str,
            lkey: &str,
            value: Vec<u8>,
            ttl_secs: u64,
        ) -> anyhow::Result<()> {
            self.inner.put(pkey, lkey, value, ttl_secs).await
        }

        async fn delete(&self, pkey: &str, lkey: &str) -> anyhow::Result<()> {
            self.inner.delete(pkey, lkey).await
        }
    }

    #[tokio::test]
    async fn two_saves_from_one_stored_version_the_second_conflicts() {
        let inner = MemControlStore::default();
        inner.put_record(&row_rec(
            "H",
            LevelRow::High,
            ControlMode::Live,
            vec![SUSPEND],
        ));
        inner.put_record(&record("H", None, None, ControlMode::Rows, vec![]));
        let store = Arc::new(HoldingStore {
            inner,
            marker_scans: AtomicUsize::new(0),
            first_checked: tokio::sync::Notify::new(),
            second_done: tokio::sync::Notify::new(),
        });
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            "pod-0".into(),
        ));
        ac.refresh().await.unwrap();
        let req = |mode, actions| ActionControlPutRequest {
            expected_version: Some(1),
            confirm: None,
            ..row_put("H", LevelRow::High, mode, actions)
        };
        let first = {
            let ac = ac.clone();
            let r = req(ControlMode::Recording, vec![]);
            tokio::spawn(async move { set(&ac, r).await })
        };
        store.first_checked.notified().await;
        let second = set(&ac, req(ControlMode::Live, vec![SUSPEND])).await;
        store.second_done.notify_one();
        let first = first.await.unwrap();
        assert!(first.is_ok(), "{first:?}");
        assert!(
            matches!(second, Err(SetError::Conflict(_, Some(1)))),
            "{second:?}"
        );
        let rows = store.inner.pkey_rows(MH_PKEY);
        let high = rows.iter().find(|(k, _)| k == "user/l/H/high").unwrap();
        assert_eq!(parse_record(&high.1).unwrap().mode, ControlMode::Recording);

        let next = ActionControlPutRequest {
            expected_version: Some(2),
            ..req(ControlMode::Live, vec![SUSPEND])
        };
        assert!(
            set(
                &ac,
                ActionControlPutRequest {
                    confirm: Some(CONFIRM_WORD.into()),
                    ..next
                }
            )
            .await
            .is_ok()
        );
        assert!(matches!(
            set(&ac, req(ControlMode::Recording, vec![])).await,
            Err(SetError::Conflict(_, Some(3)))
        ));
    }

    #[tokio::test]
    async fn a_save_past_its_hold_on_the_head_writes_nothing() {
        let store = Arc::new(MemControlStore::default());
        let mut ac = ActionControl::new(store.clone(), "prod".into(), "pod-0".into());
        ac.save_budget = Duration::ZERO;
        ac.refresh().await.unwrap();
        for req in [
            move_put("H", all_rows(ControlMode::Live, vec![SUSPEND])),
            put("H", ControlMode::Live, vec![SUSPEND]),
        ] {
            match set(&ac, req).await {
                Err(SetError::Store(msg)) => assert!(msg.contains("hold"), "{msg}"),
                other => panic!("{other:?}"),
            }
        }
        assert!(store.pkey_rows(MH_PKEY).is_empty());
        let markers = store.pkey_rows(&format!("{SAVE_PKEY_PREFIX}user/h/H"));
        assert_eq!(markers.len(), 2);
        for (_, v) in markers {
            let v: Value = serde_json::from_slice(&v).unwrap();
            assert_eq!(v["done"], true, "the marker is cleared");
        }

        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("H", None, None, ControlMode::Rows, vec![]));
        let mut ac = ActionControl::new(store.clone(), "prod".into(), "pod-0".into());
        ac.save_budget = SCAN_TIMEOUT * 4;
        ac.refresh().await.unwrap();
        let resave = ActionControlPutRequest {
            expected_version: Some(1),
            ..move_put("H", all_rows(ControlMode::Live, vec![SUSPEND]))
        };
        match set(&ac, resave).await {
            Err(SetError::Store(msg)) => assert!(msg.contains("nothing was written"), "{msg}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(store.pkey_rows(MH_PKEY).len(), 1, "no row written");
    }

        fn on_rows(mode: ControlMode, actions: Vec<PageAction>) -> Vec<ActionControlRecord> {
        let mut recs = vec![record("H", None, None, ControlMode::Rows, vec![])];
        recs.extend(LevelRow::ALL.map(|r| row_rec("H", r, mode, actions.clone())));
        recs
    }

    #[tokio::test]
    async fn a_write_failing_part_way_through_a_save_puts_every_record_back() {
        let seen_v1 = || RowSetting {
            expected_version: Some(1),
            ..setting(ControlMode::Live, vec![SUSPEND])
        };
        let resave = ActionControlPutRequest {
            expected_version: Some(1),
            ..move_put(
                "H",
                RowsRequest {
                    low: Some(seen_v1()),
                    mid: Some(seen_v1()),
                    high: Some(seen_v1()),
                    missing: Some(seen_v1()),
                },
            )
        };
        let leave = ActionControlPutRequest {
            expected_version: Some(1),
            ..put("H", ControlMode::Live, vec![SUSPEND])
        };
        let cases = [
            (on_rows(ControlMode::Recording, vec![]), resave, 2),
            (
                vec![record("H", None, None, ControlMode::Recording, vec![])],
                ActionControlPutRequest {
                    expected_version: Some(1),
                    ..move_put("H", all_rows(ControlMode::Live, vec![SUSPEND]))
                },
                4,
            ),
            (on_rows(ControlMode::Recording, vec![]), leave, 2),
        ];
        for (stored, req, fail_at) in cases {
            let (ac, store) = loaded(&stored).await;
            let before = store.pkey_rows(MH_PKEY);
            store
                .fail_record_writes
                .lock()
                .unwrap()
                .insert(fail_at, false);
            match set(&ac, req).await {
                Err(SetError::Store(msg)) => {
                    assert!(msg.contains("put back as it was"), "{fail_at}: {msg}")
                }
                other => panic!("{fail_at}: {other:?}"),
            }
            assert!(store.record_writes.load(Ordering::SeqCst) > fail_at + 1);
            assert_eq!(store.pkey_rows(MH_PKEY), before, "{fail_at}");
            let resolved = ac.resolve(
                &ResolveInput {
                    confidence: Some(high_claim()),
                    ..input("H", "c", &[])
                },
                &on(),
                |_| true,
            );
            assert_ne!(resolved.mode, ControlMode::Live, "{fail_at}");
        }
    }

    #[tokio::test]
    async fn a_failed_put_back_leaves_the_old_setup_or_the_confirmed_one() {
        let failing = |store: &MemControlStore, writes: &[(usize, bool)]| {
            store
                .fail_record_writes
                .lock()
                .unwrap()
                .extend(writes.iter().copied());
        };
        let may_have_applied = |out: Result<SetOutcome, SetError>| match out {
            Err(SetError::Store(msg)) => assert!(msg.contains("may have applied"), "{msg}"),
            other => panic!("{other:?}"),
        };
        let mode_of = |store: &MemControlStore, lkey: &str| {
            store
                .pkey_rows(MH_PKEY)
                .into_iter()
                .find(|(k, _)| k == lkey)
                .map(|(_, v)| parse_record(&v).unwrap().mode)
        };
        let decides = |ac: &ActionControl, row: Option<LevelRow>| {
            ac.resolve(
                &ResolveInput {
                    confidence: row.map(|row| crate::confidence::Claim {
                        row,
                        ..high_claim()
                    }),
                    ..input("H", "c", &[])
                },
                &on(),
                |_| true,
            )
            .mode
        };

        for put_back_lands in [false, true] {
            let stored = [
                record("H", None, None, ControlMode::Recording, vec![]),
                row_rec("H", LevelRow::High, ControlMode::Live, vec![SUSPEND]),
            ];
            let (ac, store) = loaded(&stored).await;
            failing(&store, &[(4, true), (5, put_back_lands)]);
            let rows = RowsRequest {
                high: Some(RowSetting {
                    expected_version: Some(1),
                    ..setting(ControlMode::Recording, vec![])
                }),
                ..all_rows(ControlMode::Recording, vec![])
            };
            let req = ActionControlPutRequest {
                expected_version: Some(1),
                ..move_put("H", rows)
            };
            may_have_applied(set(&ac, req).await);
            for row in LevelRow::ALL {
                let lkey = format!("user/l/H/{}", row.as_str());
                assert_eq!(
                    mode_of(&store, &lkey),
                    Some(ControlMode::Recording),
                    "{lkey} keeps its confirmed setting"
                );
            }
            ac.refresh().await.unwrap();
            assert_eq!(
                decides(&ac, Some(LevelRow::High)),
                ControlMode::Recording,
                "put-back landed: {put_back_lands}"
            );
        }

        let (ac, store) = loaded(&on_rows(ControlMode::Live, vec![SUSPEND])).await;
        failing(&store, &[(2, true), (4, false)]);
        let leave = ActionControlPutRequest {
            expected_version: Some(1),
            ..put("H", ControlMode::Recording, vec![])
        };
        may_have_applied(set(&ac, leave).await);
        assert_eq!(mode_of(&store, "user/h/H"), Some(ControlMode::Recording));
        ac.refresh().await.unwrap();
        for row in [Some(LevelRow::Low), Some(LevelRow::High), None] {
            assert_eq!(decides(&ac, row), ControlMode::Recording, "{row:?}");
        }

        let (ac, store) = loaded(&on_rows(ControlMode::Recording, vec![])).await;
        failing(&store, &[(0, true), (1, false)]);
        let edit = ActionControlPutRequest {
            expected_version: Some(1),
            ..row_put("H", LevelRow::Missing, ControlMode::Live, vec![SUSPEND])
        };
        may_have_applied(set(&ac, edit).await);
        assert_eq!(mode_of(&store, "user/l/H/missing"), Some(ControlMode::Live));
    }

    #[tokio::test]
    async fn leaving_rows_removes_the_scopes_row_records() {
        let mut stored = on_rows(ControlMode::Recording, vec![]);
        stored.push(record("H", Some("c1"), None, ControlMode::Rows, vec![]));
        stored.push(ActionControlRecord {
            confidence: Some(LevelRow::High),
            ..record("H", Some("c1"), None, ControlMode::Recording, vec![])
        });
        let (ac, store) = loaded(&stored).await;
        let leave = ActionControlPutRequest {
            expected_version: Some(1),
            ..put("H", ControlMode::Recording, vec![])
        };
        let out = set(&ac, leave).await.unwrap();
        assert_eq!(out.view.record.mode, ControlMode::Recording);
        let keys: Vec<String> = store
            .pkey_rows(MH_PKEY)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(
            keys,
            ["user/c/H/c1", "user/cl/H/c1/high", "user/h/H"],
            "the head's rows are gone; the classifier, still on rows, keeps its own"
        );
        let listed = ac.list(&on(), baked_safety()).await;
        assert!(
            listed
                .records
                .iter()
                .all(|r| !r.lkey.starts_with("user/l/")),
            "{:?}",
            listed.records
        );
        let again = ActionControlPutRequest {
            expected_version: Some(2),
            ..move_put("H", all_rows(ControlMode::Recording, vec![]))
        };
        assert!(set(&ac, again).await.is_ok());
    }

    #[tokio::test]
    async fn a_classifier_leaving_rows_for_the_rules_is_judged_against_the_heads_rows() {
        let classifier_row = |row, mode| ActionControlRecord {
            confidence: Some(row),
            ..record("H", Some("c1"), None, mode, vec![])
        };
        let stored = |classifier_mode: ControlMode, head_high: ActionControlRecord| {
            let mut recs = vec![
                record("H", None, None, ControlMode::Rows, vec![]),
                record("H", Some("c1"), None, ControlMode::Rows, vec![]),
                head_high,
            ];
            recs.extend(LevelRow::ALL.map(|r| classifier_row(r, classifier_mode)));
            recs
        };
        let leave = ActionControlPutRequest {
            classifier: Some("c1".into()),
            expected_version: Some(1),
            confirm: None,
            ..put("H", ControlMode::Rules, vec![])
        };
        let head_live = ActionControlRecord {
            evidence: Some(Box::new(read(0.96, 59, 0))),
            ..row_rec("H", LevelRow::High, ControlMode::Live, vec![SUSPEND])
        };
        let (ac, _) = loaded(&stored(ControlMode::Rules, head_live)).await;
        match set(&ac, leave.clone()).await {
            Err(SetError::Invalid("confirm_required", msg)) => {
                assert!(msg.contains("high row: turns the scope live"), "{msg}")
            }
            other => panic!("{other:?}"),
        }
        let head_recording = row_rec("H", LevelRow::High, ControlMode::Recording, vec![]);
        let (ac, _) = loaded(&stored(ControlMode::Recording, head_recording)).await;
        let out = set(&ac, leave).await.unwrap();
        assert!(out.harsher_because.is_empty(), "{:?}", out.harsher_because);
    }

    #[tokio::test]
    async fn a_classifier_move_is_refused_while_a_head_band_setting_is_not_rules() {
        let (ac, _) = loaded(&[record(
            "H",
            None,
            Some("b1"),
            ControlMode::Live,
            vec![LIVENESS],
        )])
        .await;
        let req = ActionControlPutRequest {
            classifier: Some("c1".into()),
            ..move_put("H", all_rules())
        };
        match set(&ac, req.clone()).await {
            Err(SetError::Invalid("band_settings_not_rules", msg)) => {
                assert!(msg.contains("user/b/H/b1"), "{msg}")
            }
            other => panic!("{other:?}"),
        }
        let (ac, _) = loaded(&[record("H", None, Some("b1"), ControlMode::Rules, vec![])]).await;
        let live = ActionControlPutRequest {
            confirm: None,
            rows: Some(all_rows(ControlMode::Live, vec![SUSPEND])),
            ..req
        };
        assert_eq!(code(set(&ac, live).await), "confirm_required");
    }

    #[tokio::test]
    async fn a_save_marker_from_a_pod_minutes_behind_still_holds_the_head() {
        let (ac, store) = loaded(&[]).await;
        for (behind_ms, held) in [
            (240_000, true),
            (-240_000, true),
            (400_000, true),
            (430_000, false),
        ] {
            store.rows.lock().unwrap().insert(
                (
                    format!("{SAVE_PKEY_PREFIX}user/h/H"),
                    format!("other-{behind_ms}"),
                ),
                serde_json::json!({"at_ms": PUT_NOW - behind_ms, "done": false})
                    .to_string()
                    .into_bytes(),
            );
            let got = set(&ac, put("H", ControlMode::Recording, vec![])).await;
            if held {
                assert!(
                    matches!(got, Err(SetError::Conflict(..))),
                    "{behind_ms}: {got:?}"
                );
            } else {
                assert!(got.is_ok(), "{behind_ms}: {got:?}");
            }
            store
                .rows
                .lock()
                .unwrap()
                .retain(|(p, _), _| !p.starts_with(SAVE_PKEY_PREFIX));
        }
    }

    #[tokio::test]
    async fn a_moves_rows_are_checked_against_the_versions_the_page_saw() {
        let (ac, _) = loaded(&[
            record("H", None, None, ControlMode::Rows, vec![]),
            row_rec("H", LevelRow::High, ControlMode::Recording, vec![]),
        ])
        .await;
        let resave = |high_seen: Option<u64>| ActionControlPutRequest {
            expected_version: Some(1),
            ..move_put(
                "H",
                RowsRequest {
                    high: Some(RowSetting {
                        expected_version: high_seen,
                        ..setting(ControlMode::Recording, vec![])
                    }),
                    ..Default::default()
                },
            )
        };
        assert!(matches!(
            set(&ac, resave(None)).await,
            Err(SetError::Conflict(_, Some(1)))
        ));
        assert!(set(&ac, resave(Some(1))).await.is_ok());
    }

    #[tokio::test]
    async fn a_row_left_out_of_a_move_never_overwrites_a_stored_row() {
        let edited = ActionControlRecord {
            version: 2,
            ..row_rec("H", LevelRow::Low, ControlMode::Live, vec![SUSPEND])
        };
        let (ac, store) =
            loaded(&[record("H", None, None, ControlMode::Rows, vec![]), edited]).await;
        let before = store.pkey_rows(MH_PKEY);
        let resave = |low: Option<RowSetting>| ActionControlPutRequest {
            expected_version: Some(1),
            ..move_put(
                "H",
                RowsRequest {
                    low,
                    high: Some(setting(ControlMode::Recording, vec![])),
                    ..Default::default()
                },
            )
        };
        assert!(matches!(
            set(&ac, resave(None)).await,
            Err(SetError::Conflict(_, Some(2)))
        ));
        assert_eq!(store.pkey_rows(MH_PKEY), before, "nothing written");
        let low_seen = RowSetting {
            expected_version: Some(2),
            ..setting(ControlMode::Recording, vec![])
        };
        assert!(set(&ac, resave(Some(low_seen))).await.is_ok());
    }

    #[test]
    fn stored_time_boundaries_are_exact() {
        let now = PUT_NOW;
        let skew = crate::confidence::CLOCK_SKEW_MS;
        let day = 86_400_000;
        let mut r = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        r.updated_at_ms = now + skew;
        assert_eq!(
            levels::record_problem(&r, now),
            None,
            "written exactly the skew ahead"
        );
        r.updated_at_ms += 1;
        assert!(levels::record_problem(&r, now).is_some());

        let mut lv = row_rec("H", LevelRow::High, ControlMode::Live, vec![SUSPEND]);
        lv.evidence = Some(Box::new(ConfidenceEvidence {
            read_at_ms: now + skew,
            ..read(0.96, 59, 0)
        }));
        assert_eq!(
            levels::record_problem(&lv, now),
            None,
            "read exactly the skew ahead"
        );
        lv.evidence.as_mut().unwrap().read_at_ms += 1;
        assert!(levels::record_problem(&lv, now).is_some());

        let with_override = |updated_at_ms: i64, expires_at_ms: i64| {
            let mut r = row_rec("H", LevelRow::High, ControlMode::Live, vec![SUSPEND]);
            r.updated_at_ms = updated_at_ms;
            r.evidence_override = Some(Box::new(EvidenceOverride {
                expires_at_ms,
                ..person(0)
            }));
            levels::record_problem(&r, now)
        };
        assert_eq!(with_override(now, now + 7 * day), None, "exactly 7 days");
        assert!(with_override(now, now + 7 * day + 1).is_some());
        assert!(with_override(now, now).is_some(), "expires at its write");
        assert_eq!(with_override(now, now + 1), None);
        assert!(
            with_override(now + skew, now + 60_000).is_some(),
            "expires before its write"
        );
    }

    #[test]
    fn the_get_view_serves_the_table_only_as_suggestions() {
        let mut cfg = on();
        let view = levels::view(&cfg);
        assert_eq!(view.config_error, None);
        assert_eq!(view.factory_evidence_max_age_ms, 365 * 86_400_000);
        let with_key = ControlConfig {
            confidence: crate::confidence::ConfidenceConfig::from_value(Some(
                &serde_json::json!({"factory_evidence_max_age_days": 30}),
            )),
            ..on()
        };
        assert_eq!(
            levels::view(&with_key).factory_evidence_max_age_ms,
            30 * 86_400_000
        );
        let rows: Vec<LevelRow> = view.rows.iter().map(|r| r.row).collect();
        assert_eq!(rows, LevelRow::ALL.to_vec());
        assert_eq!(view.rows[2].suggested_actions.user, vec![SUSPEND]);
        assert_eq!(view.rows[3].floor, None);
        assert!(view.rows[3].suggested_actions.user.is_empty());
        cfg.confidence = crate::confidence::ConfidenceConfig::from_value(Some(
            &serde_json::json!({"label_map": {"x": "high"}}),
        ));
        let view = levels::view(&cfg);
        assert!(view.config_error.is_some_and(|e| e.contains("label_map")));
        assert_eq!(view.evidence_max_age_ms, 0);
        assert_eq!(view.factory_evidence_max_age_ms, 0);
    }

    #[tokio::test]
    async fn a_stored_record_breaking_the_row_rules_is_unreadable() {
        let mut both = row_rec("H", LevelRow::High, ControlMode::Live, vec![SUSPEND]);
        both.band = Some("b".into());
        assert!(parse_record(&serde_json::to_vec(&both).unwrap()).is_err());
        let rows_with_actions = record("H", None, None, ControlMode::Rows, vec![SUSPEND]);
        assert!(parse_record(&serde_json::to_vec(&rows_with_actions).unwrap()).is_err());
        let rows_on_a_row = row_rec("H", LevelRow::High, ControlMode::Rows, vec![]);
        assert!(parse_record(&serde_json::to_vec(&rows_on_a_row).unwrap()).is_err());
        let mut stray_evidence = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        stray_evidence.evidence = Some(Box::new(read(0.96, 59, 0)));
        assert!(parse_record(&serde_json::to_vec(&stray_evidence).unwrap()).is_err());
        let mut missing_evidence =
            row_rec("H", LevelRow::Missing, ControlMode::Live, vec![SUSPEND]);
        missing_evidence.evidence = Some(Box::new(read(0.96, 59, 0)));
        assert!(parse_record(&serde_json::to_vec(&missing_evidence).unwrap()).is_err());
        let mut newer = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        newer.v = ACTION_CONTROL_RECORD_VERSION + 1;
        assert!(parse_record(&serde_json::to_vec(&newer).unwrap()).is_err());
        let retired =
            br#"{"v":2,"entity_type":"user","head":"H","mode":"confidence_defaults","version":1}"#;
        assert!(parse_record(retired).is_err());
    }

    #[tokio::test]
    async fn seen_rows_are_listed_per_row() {
        let (ac, _) = loaded(&[]).await;
        let task = ac.spawn_seen_writer().unwrap();
        let seen = |row| ActionControlSeenRow {
            entity_type: ControlEntity::User,
            head: "H".into(),
            classifier: "clf".into(),
            row,
            last_seen_ms: 1,
        };
        assert_eq!(
            ac.record_seen_row(seen(LevelRow::High)),
            SeenOutcome::Queued
        );
        assert_eq!(
            ac.record_seen_row(seen(LevelRow::High)),
            SeenOutcome::Recent
        );
        assert_eq!(
            ac.record_seen_row(seen(LevelRow::Missing)),
            SeenOutcome::Queued
        );
        let mut bad = seen(LevelRow::Low);
        bad.head = "bad head".into();
        assert_eq!(ac.record_seen_row(bad), SeenOutcome::Invalid);
        for _ in 0..50 {
            if ac.list(&on(), baked_safety()).await.seen_rows.len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let list = ac.list(&on(), baked_safety()).await;
        let rows: Vec<LevelRow> = list.seen_rows.iter().map(|s| s.row).collect();
        assert_eq!(rows, vec![LevelRow::High, LevelRow::Missing]);
        assert!(list.seen.is_empty(), "the band list is separate");
        task.abort();
    }

    #[tokio::test]
    async fn band_scopes_of_a_renamed_head_are_compared_like_the_decision_path() {
        for (new_mode, old_mode, harsher) in [
            (ControlMode::Recording, ControlMode::Live, true),
            (ControlMode::Live, ControlMode::Recording, false),
        ] {
            let band = |head: &str, mode| {
                let actions = if mode == ControlMode::Live {
                    vec![SUSPEND]
                } else {
                    vec![]
                };
                record(head, None, Some("b1"), mode, actions)
            };
            let (ac, _) = loaded(&[band("ZNew", new_mode), band("AOld", old_mode)]).await;
            let mut cfg = on();
            cfg.aliases.insert("ZNew".into(), "AOld".into());
            let ls = vec!["b1".to_owned()];
            let decided = ac
                .resolve(&input("ZNew", "clf", &ls), &cfg, |_| true)
                .matched
                .map(|m| m.lkey);
            assert_eq!(decided.as_deref(), Some("user/b/ZNew/b1"));
            let req = ActionControlPutRequest {
                classifier: Some("clf".into()),
                band: Some("b1".into()),
                confirm: None,
                ..put("ZNew", ControlMode::Live, vec![SUSPEND])
            };
            let got = put_with(&ac, &cfg, req).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "{new_mode:?}");
            } else {
                assert!(got.is_ok(), "{new_mode:?}: {got:?}");
            }
        }
    }

    #[tokio::test]
    async fn a_first_head_write_with_classifiers_needs_the_confirmation() {
        let (ac, _) = loaded(&[
            record("H", Some("clf-a"), None, ControlMode::Live, vec![SUSPEND]),
            record("H", Some("clf-b"), None, ControlMode::Recording, vec![]),
        ])
        .await;
        let head = ActionControlPutRequest {
            confirm: None,
            ..put("H", ControlMode::Live, vec![SUSPEND])
        };
        assert_eq!(code(set(&ac, head.clone()).await), "confirm_required");
        let (ac, _) = loaded(&[
            record("OldH", None, None, ControlMode::Live, vec![SUSPEND]),
            record("H", Some("clf-b"), None, ControlMode::Recording, vec![]),
        ])
        .await;
        let mut cfg = on();
        cfg.aliases.insert("H".into(), "OldH".into());
        let got = put_with(&ac, &cfg, head).await;
        assert!(got.is_ok(), "{got:?}");
    }

            async fn rollout_bed(rollout_bp: Option<u32>) -> (Arc<ActionControl>, Arc<MemControlStore>) {
        loaded(&[
            record("H", None, None, ControlMode::Rows, vec![]),
            ActionControlRecord {
                rollout_bp,
                ..row_rec("H", LevelRow::Missing, ControlMode::Live, vec![SUSPEND])
            },
        ])
        .await
    }

    fn by_author(ac: &ActionControl, author_id: i64) -> Resolved {
        ac.resolve(
            &ResolveInput {
                author_id,
                ..input("H", "c", &[])
            },
            &on(),
            |_| true,
        )
    }

    #[tokio::test]
    async fn a_row_rollout_acts_on_the_accounts_inside_it_and_records_the_rest() {
        let (ac, _) = rollout_bed(Some(500)).await;
        let mut inside = 0;
        for id in 0..4_000i64 {
            let r = by_author(&ac, id);
            if in_rollout(Some(500), "", "H", id) {
                inside += 1;
                assert_eq!((r.mode, r.forced), (ControlMode::Live, None), "{id}");
            } else {
                assert_eq!(
                    (r.mode, r.forced),
                    (ControlMode::Recording, Some("outside_rollout")),
                    "{id}"
                );
                assert_eq!(logged_because(&r).as_deref(), Some("outside_rollout"));
            }
            assert_eq!(r.page_actions, vec![SUSPEND], "{id}");
            let keys = r.info_keys();
            assert!(keys.contains(&("rollout_bp".into(), "500".into())));
            assert!(keys.contains(&(
                "rollout_bucket".into(),
                rollout_bucket("", "H", id).to_string()
            )));
        }
        assert!((120..280).contains(&inside), "{inside} of 4,000 inside 5%");
    }

            #[tokio::test]
    async fn a_salted_pod_places_accounts_by_the_salted_bucket() {
        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("H", None, None, ControlMode::Rows, vec![]));
        store.put_record(&ActionControlRecord {
            rollout_bp: Some(5_000),
            ..row_rec("H", LevelRow::Missing, ControlMode::Live, vec![SUSPEND])
        });
        let ac = ActionControl::new(store, "prod".into(), "pod-0".into())
            .with_rollout_salt("test-salt".into());
        ac.refresh().await.unwrap();
        let mut moved = 0;
        for id in 0..2_000i64 {
            let r = by_author(&ac, id);
            let inside = in_rollout(Some(5_000), "test-salt", "H", id);
            assert_eq!(r.forced.is_none(), inside, "{id}");
            let bucket = rollout_bucket("test-salt", "H", id).to_string();
            assert!(r.info_keys().contains(&("rollout_bucket".into(), bucket)));
            moved += usize::from(inside != in_rollout(Some(5_000), "", "H", id));
        }
        assert!(moved > 0, "the salt changes which accounts are inside");
    }

    #[tokio::test]
    async fn no_rollout_acts_on_every_account_and_zero_on_none() {
        let (all, _) = rollout_bed(None).await;
        let (none, _) = rollout_bed(Some(0)).await;
        for id in 0..2_000i64 {
            let r = by_author(&all, id);
            assert_eq!((r.mode, r.forced), (ControlMode::Live, None), "{id}");
            assert!(!r.info_keys().iter().any(|(k, _)| k.starts_with("rollout")));
            let r = by_author(&none, id);
            assert_eq!(r.forced, Some("outside_rollout"), "{id}");
        }
    }

    #[tokio::test]
    async fn a_rollout_never_makes_a_row_act_that_would_not() {
        let (ac, _) = rollout_bed(Some(10_000)).await;
        let off = ControlConfig {
            enabled: false,
            ..on()
        };
        let r = ac.resolve(&input("H", "c", &[]), &off, |_| true);
        assert_eq!(r.forced, Some("kill_switch"));
        let r = ac.resolve(&input("H", "c", &[]), &on(), |_| false);
        assert_eq!(r.forced, Some("safety_checks_changed"));
    }

    fn missing_row_put(rollout_bp: Option<u32>, confirm: Option<&str>) -> ActionControlPutRequest {
        ActionControlPutRequest {
            expected_version: Some(1),
            confirm: confirm.map(Into::into),
            rollout_bp,
            ..row_put("H", LevelRow::Missing, ControlMode::Live, vec![SUSPEND])
        }
    }

    #[tokio::test]
    async fn a_saved_rollout_is_stored_on_its_row_and_all_accounts_as_none() {
        let (ac, store) = loaded(&[]).await;
        let mut rows = all_rows(ControlMode::Recording, vec![]);
        rows.missing = Some(RowSetting {
            rollout_bp: Some(100),
            ..setting(ControlMode::Live, vec![SUSPEND])
        });
        rows.low = Some(RowSetting {
            rollout_bp: Some(ROLLOUT_ALL_BP),
            ..setting(ControlMode::Recording, vec![])
        });
        let out = set(&ac, move_put("H", rows)).await;
        assert!(out.is_ok(), "{out:?}");
        let stored = |row| {
            let bytes = store.rows.lock().unwrap()[&(
                MH_PKEY.to_owned(),
                level_lkey(ControlEntity::User, "H", None, row),
            )]
                .clone();
            parse_record(&bytes).unwrap()
        };
        let missing = stored(LevelRow::Missing);
        assert_eq!((missing.rollout_bp, missing.v), (Some(100), 4));
        let low = stored(LevelRow::Low);
        assert_eq!((low.rollout_bp, low.v), (None, ROWS_RECORD_VERSION));
        assert_eq!(stored(LevelRow::High).rollout_bp, None);
        let r = by_author(&ac, 7);
        assert_eq!(r.forced.is_none(), in_rollout(Some(100), "", "H", 7));
    }

    #[tokio::test]
    async fn raising_a_live_rows_rollout_needs_confirm_and_lowering_it_does_not() {
        let (ac, _) = rollout_bed(Some(100)).await;
        match set(&ac, missing_row_put(Some(500), None)).await {
            Err(SetError::Invalid("confirm_required", msg)) => {
                assert!(msg.contains("rollout 1% to 5%"), "{msg}")
            }
            other => panic!("{other:?}"),
        }
        let (ac, _) = rollout_bed(Some(100)).await;
        assert!(
            set(&ac, missing_row_put(Some(500), Some(CONFIRM_WORD)))
                .await
                .is_ok()
        );
        let (ac, _) = rollout_bed(Some(500)).await;
        let lower = set(&ac, missing_row_put(Some(100), None)).await;
        assert!(lower.is_ok(), "{lower:?}");
        let (ac, _) = rollout_bed(Some(500)).await;
        assert_eq!(
            code(set(&ac, missing_row_put(None, None)).await),
            "confirm_required",
            "back to every account"
        );
    }

    #[tokio::test]
    async fn a_rollout_belongs_on_a_row_and_within_all_accounts() {
        let (ac, _) = rollout_bed(None).await;
        assert_eq!(
            code(set(&ac, missing_row_put(Some(ROLLOUT_ALL_BP + 1), None)).await),
            "invalid_rollout"
        );
        let head = ActionControlPutRequest {
            rollout_bp: Some(100),
            ..put("H2", ControlMode::Live, vec![SUSPEND])
        };
        assert_eq!(code(set(&ac, head).await), "invalid_rollout");
        let mut moved = move_put("H3", all_rows(ControlMode::Recording, vec![]));
        moved.rollout_bp = Some(100);
        assert_eq!(code(set(&ac, moved).await), "invalid_rollout");
        let stored = ActionControlRecord {
            rollout_bp: Some(100),
            ..record("H", None, None, ControlMode::Live, vec![SUSPEND])
        };
        assert!(parse_record(&serde_json::to_vec(&stored).unwrap()).is_err());
        let stored = ActionControlRecord {
            rollout_bp: Some(ROLLOUT_ALL_BP + 1),
            ..row_rec("H", LevelRow::Missing, ControlMode::Live, vec![SUSPEND])
        };
        assert!(parse_record(&serde_json::to_vec(&stored).unwrap()).is_err());
    }

    #[test]
    fn rollout_percentages_read_plainly() {
        assert_eq!(rollout_percent(100), "1%");
        assert_eq!(rollout_percent(50), "0.5%");
        assert_eq!(rollout_percent(1_225), "12.25%");
        assert_eq!(rollout_percent(0), "0%");
        assert_eq!(rollout_percent(ROLLOUT_ALL_BP), "100%");
    }

                            #[tokio::test]
    async fn a_pod_that_cannot_read_a_newer_row_keeps_its_last_copy() {
        let (ac, store) = rollout_bed(None).await;
        assert_eq!(by_author(&ac, 7).mode, ControlMode::Live);
        let lkey = level_lkey(ControlEntity::User, "H", None, LevelRow::Missing);
        let mut newer = serde_json::to_value(ActionControlRecord {
            rollout_bp: Some(100),
            version: 2,
            ..row_rec("H", LevelRow::Missing, ControlMode::Live, vec![SUSPEND])
        })
        .unwrap();
        newer["v"] = (ACTION_CONTROL_RECORD_VERSION + 1).into();
        newer["field_this_image_lacks"] = true.into();
        store.put_raw(&lkey, &serde_json::to_vec(&newer).unwrap());
        ac.refresh().await.unwrap();
        let held = by_author(&ac, 7);
        assert_eq!(
            (held.mode, held.forced),
            (ControlMode::Live, None),
            "the copy it held still acts, on every account"
        );
        let fresh = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            "pod-1".into(),
        ));
        fresh.refresh().await.unwrap();
        let r = by_author(&fresh, 7);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("unreadable"))
        );

        let (ac, store) = rollout_bed(None).await;
        store.put_record(&ActionControlRecord {
            version: 2,
            ..row_rec("H", LevelRow::Missing, ControlMode::Recording, vec![])
        });
        ac.refresh().await.unwrap();
        newer["version"] = 3.into();
        store.put_raw(&lkey, &serde_json::to_vec(&newer).unwrap());
        ac.refresh().await.unwrap();
        assert_eq!(by_author(&ac, 7).mode, ControlMode::Recording);
    }

    #[tokio::test]
    async fn a_row_live_for_every_account_takes_a_rollout_only_after_a_log_only_save() {
        let (ac, _) = rollout_bed(None).await;
        match set(&ac, missing_row_put(Some(500), Some(CONFIRM_WORD))).await {
            Err(SetError::Invalid("rollout_from_every_account", msg)) => {
                assert!(msg.contains("set it to log only and save"), "{msg}")
            }
            other => panic!("{other:?}"),
        }
        let paused_with_rollout = ActionControlPutRequest {
            mode: ControlMode::Recording,
            actions: vec![],
            ..missing_row_put(Some(500), None)
        };
        assert_eq!(
            code(set(&ac, paused_with_rollout).await),
            "rollout_from_every_account",
            "a log-only row with a rollout is unreadable to that image too"
        );
        let mut moved = all_rows(ControlMode::Recording, vec![]);
        moved.missing = Some(RowSetting {
            rollout_bp: Some(500),
            expected_version: Some(1),
            ..setting(ControlMode::Live, vec![SUSPEND])
        });
        let move_req = ActionControlPutRequest {
            expected_version: Some(1),
            ..move_put("H", moved)
        };
        assert_eq!(
            code(set(&ac, move_req).await),
            "rollout_from_every_account",
            "through a save of all four rows too"
        );
        let all = set(&ac, missing_row_put(Some(ROLLOUT_ALL_BP), None)).await;
        assert!(all.is_ok(), "100% is no rollout: {all:?}");

        let off = ActionControlPutRequest {
            expected_version: Some(2),
            mode: ControlMode::Recording,
            actions: vec![],
            ..missing_row_put(None, None)
        };
        assert!(set(&ac, off).await.is_ok());
        let on = ActionControlPutRequest {
            expected_version: Some(3),
            ..missing_row_put(Some(500), Some(CONFIRM_WORD))
        };
        let on = set_at(&ac, on, NOW_MS + ROLLOUT_WAIT_MS).await;
        assert!(on.is_ok(), "{on:?}");
        assert!(in_rollout(Some(500), "", "H", 7) == by_author(&ac, 7).forced.is_none());
    }

            async fn log_only_at_now() -> (Arc<ActionControl>, Arc<MemControlStore>) {
        let (ac, store) = rollout_bed(None).await;
        let off = ActionControlPutRequest {
            mode: ControlMode::Recording,
            actions: vec![],
            ..missing_row_put(None, None)
        };
        let off = set(&ac, off).await;
        assert!(off.is_ok(), "{off:?}");
        (ac, store)
    }

    fn rollout_after_log_only(expected_version: u64) -> ActionControlPutRequest {
        ActionControlPutRequest {
            expected_version: Some(expected_version),
            ..missing_row_put(Some(500), Some(CONFIRM_WORD))
        }
    }

    fn refusal(r: Result<SetOutcome, SetError>) -> (&'static str, String) {
        match r {
            Err(SetError::Invalid(c, msg)) => (c, msg),
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_rollout_waits_two_minutes_after_the_log_only_save() {
        let (ac, _) = log_only_at_now().await;
        let (c, msg) = refusal(set_at(&ac, rollout_after_log_only(2), NOW_MS + 119_000).await);
        assert_eq!(c, "rollout_wait");
        assert!(
            msg.contains("stopped acting on every account 119 s ago")
                && msg.contains("set its rollout in 1 s"),
            "{msg}"
        );
        let at_121 = set_at(&ac, rollout_after_log_only(2), NOW_MS + 121_000).await;
        assert!(at_121.is_ok(), "{at_121:?}");
    }

            #[tokio::test]
    async fn the_wait_runs_from_the_save_that_ended_every_account() {
        let (ac, _) = log_only_at_now().await;
        let again = ActionControlPutRequest {
            expected_version: Some(2),
            mode: ControlMode::Recording,
            actions: vec![],
            reason: "paused again".into(),
            ..missing_row_put(None, None)
        };
        assert!(set_at(&ac, again, NOW_MS + 60_000).await.is_ok());
        let (c, msg) = refusal(set_at(&ac, rollout_after_log_only(3), NOW_MS + 100_000).await);
        assert_eq!(c, "rollout_wait");
        assert!(
            msg.contains("100 s ago") && msg.contains("in 20 s"),
            "{msg}"
        );
        let later = set_at(&ac, rollout_after_log_only(3), NOW_MS + 121_000).await;
        assert!(later.is_ok(), "{later:?}");
    }

                        #[tokio::test]
    async fn a_pod_missing_the_log_only_read_never_sees_a_rollout_saved_seconds_later() {
        let (saver, store) = rollout_bed(None).await;
        let old = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            "pod-old".into(),
        ));
        let missed = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            "pod-missed".into(),
        ));
        for pod in [&old, &missed] {
            pod.refresh().await.unwrap();
            assert_eq!(by_author(pod, 7).mode, ControlMode::Live);
        }
        let off = ActionControlPutRequest {
            mode: ControlMode::Recording,
            actions: vec![],
            ..missing_row_put(None, None)
        };
        assert!(set(&saver, off).await.is_ok());
        let (c, _) = refusal(set_at(&saver, rollout_after_log_only(2), NOW_MS + 5_000).await);
        assert_eq!(c, "rollout_wait");
        let lkey = level_lkey(ControlEntity::User, "H", None, LevelRow::Missing);
        let raw = store
            .pkey_rows(MH_PKEY)
            .into_iter()
            .find(|(l, _)| *l == lkey)
            .unwrap()
            .1;
        let held = parse_record(&raw).unwrap();
        assert_eq!(
            (held.version, held.mode, held.rollout_bp, held.v),
            (2, ControlMode::Recording, None, ROWS_RECORD_VERSION),
            "the log-only row, in a shape an image without rollouts reads"
        );
        assert_eq!(
            by_author(&old, 7).mode,
            ControlMode::Live,
            "no read since the log-only save"
        );
        old.refresh().await.unwrap();
        assert_eq!(by_author(&old, 7).mode, ControlMode::Recording);

        missed.age_for_test(Duration::from_millis(ROLLOUT_WAIT_MS as u64));
        let r = by_author(&missed, 7);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("stale_snapshot")),
            "a pod whose last read is a whole wait old acts on no one"
        );
    }

    #[tokio::test]
    async fn rows_never_live_for_every_account_take_a_rollout_at_once() {
        let (ac, _) = loaded(&[record("H", None, None, ControlMode::Rows, vec![])]).await;
        let first = ActionControlPutRequest {
            expected_version: None,
            mode: ControlMode::Recording,
            actions: vec![],
            ..missing_row_put(None, None)
        };
        assert!(
            set(&ac, first).await.is_ok(),
            "an unset row saved as log only"
        );
        let on = set_at(&ac, rollout_after_log_only(1), NOW_MS + 1_000).await;
        assert!(on.is_ok(), "{on:?}");
        let raise = ActionControlPutRequest {
            rollout_bp: Some(3_000),
            ..rollout_after_log_only(2)
        };
        let raised = set_at(&ac, raise, NOW_MS + 2_000).await;
        assert!(raised.is_ok(), "already on a rollout: {raised:?}");
    }

    #[tokio::test]
    async fn a_recent_row_with_no_history_waits_from_its_own_save() {
        let (ac, store) = rollout_bed(None).await;
        store.put_record(&ActionControlRecord {
            version: 2,
            updated_at_ms: NOW_MS - 30_000,
            ..row_rec("H", LevelRow::Missing, ControlMode::Recording, vec![])
        });
        let (c, msg) = refusal(set(&ac, rollout_after_log_only(2)).await);
        assert_eq!(c, "rollout_wait");
        assert!(
            msg.contains("history doesn't show") && msg.contains("in 90 s"),
            "{msg}"
        );
        let later = set_at(&ac, rollout_after_log_only(2), NOW_MS + 90_000).await;
        assert!(later.is_ok(), "{later:?}");
    }

    const SUSPEND_PERMANENT_CSE: PageAction = PageAction::Suspend {
        permanent: true,
        policy: SuspensionPolicy::Cse,
    };

    #[test]
    fn a_page_suspend_maps_to_its_permanence_and_policy() {
        assert_eq!(
            page_specs(&[SUSPEND]),
            vec![ActionSpec::SuspendUser {
                perm: false,
                policy: "PlatformManipulation".into()
            }]
        );
        assert_eq!(
            page_specs(&[PageAction::SuspendAuthor {}]),
            page_specs(&[SUSPEND]),
            "the author suspend is unchanged"
        );
        assert_eq!(
            page_specs(&[SUSPEND_PERMANENT_CSE]),
            vec![ActionSpec::SuspendUser {
                perm: true,
                policy: "Cse".into()
            }]
        );
        assert_eq!(
            action_specs_json(&page_specs(&[SUSPEND_PERMANENT_CSE])),
            r#"[{"kind":"suspend","permanent":true,"policy":"Cse"}]"#
        );
        assert!(page_class(&[SUSPEND_PERMANENT_CSE]) > page_class(&[SUSPEND]));
        assert_eq!(page_class(&[SUSPEND]), crate::dedup_cache::CLASS_SUSPEND);
    }

    #[test]
    fn a_suspension_made_permanent_or_moved_to_another_policy_is_harsher() {
        let r = |actions: Vec<PageAction>| record("H", None, None, ControlMode::Live, actions);
        let why = harsher_because(
            Some(&r(vec![SUSPEND])),
            &r(vec![SUSPEND_PERMANENT_CSE]),
            None,
        );
        assert_eq!(
            why,
            vec![
                "makes the suspension permanent".to_owned(),
                "changes the suspension policy (PlatformManipulation to Cse)".to_owned()
            ]
        );
        assert!(
            harsher_because(
                Some(&r(vec![SUSPEND_PERMANENT_CSE])),
                &r(vec![SUSPEND]),
                None
            )
            .iter()
            .all(|w| !w.contains("permanent")),
            "back to temporary is softer"
        );
        assert_eq!(
            harsher_because(Some(&r(vec![SUSPEND])), &r(vec![SUSPEND]), None),
            Vec::<String>::new()
        );
        let rules_perm =
            serde_json::json!([{"kind": "suspend", "permanent": true, "policy": "Cse"}]);
        assert!(
            harsher_because(
                Some(&r(vec![SUSPEND_PERMANENT_CSE])),
                &record("H", None, None, ControlMode::Rules, vec![]),
                Some(rules_perm.as_array().unwrap()),
            )
            .is_empty(),
            "rules that send the same permanent suspension are no harsher"
        );
    }

    #[tokio::test]
    async fn a_row_saved_with_a_suspension_option_is_refused_by_an_image_before_them() {
        let (ac, store) = loaded(&[record("H", None, None, ControlMode::Rows, vec![])]).await;
        let first = ActionControlPutRequest {
            expected_version: None,
            ..row_put("H", LevelRow::High, ControlMode::Recording, vec![])
        };
        assert!(set(&ac, first).await.is_ok());
        let old = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            "pod-old".into(),
        ));
        refresh_as_older_image(&old).await;
        let live_row = ActionControlPutRequest {
            expected_version: Some(1),
            confirm: Some(CONFIRM_WORD.into()),
            rollout_bp: Some(1_000),
            ..row_put(
                "H",
                LevelRow::High,
                ControlMode::Live,
                vec![SUSPEND_PERMANENT_CSE],
            )
        };
        let out = set(&ac, live_row).await;
        assert!(out.is_ok(), "{out:?}");
        let row = stored(&store, "user/l/H/high");
        assert_eq!(
            (row.actions.clone(), row.rollout_bp, row.v),
            (
                vec![SUSPEND_PERMANENT_CSE],
                Some(1_000),
                SUSPENSION_OPTIONS_RECORD_VERSION
            )
        );
        ac.refresh().await.unwrap();
        let r = ac.resolve(
            &ResolveInput {
                confidence: Some(crate::confidence::Claim {
                    row: LevelRow::High,
                    unknown: false,
                    evidence: None,
                    now_ms: NOW_MS,
                }),
                author_id: (0..10_000)
                    .find(|&id| in_rollout(Some(1_000), "", "H", id))
                    .unwrap(),
                ..input("H", "c", &[])
            },
            &on(),
            |_| true,
        );
        assert_eq!(r.mode, ControlMode::Live);
        assert_eq!(r.page_actions, vec![SUSPEND_PERMANENT_CSE]);
        refresh_as_older_image(&old).await;
        let held = old.snapshot();
        let row = &held.records["user/l/H/high"];
        assert_eq!(
            (row.mode, row.version),
            (ControlMode::Recording, 1),
            "an older image keeps the log-only copy it held"
        );
    }
}
