use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroU32;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use prost::Message;
use rustc_hash::FxHashSet;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tracing::warn;
use xai_cache::{KVCacheError, Key};
use xai_safety_label_store::types::safety_label_metas;
use xai_visibility_filtering_proto as vf_pb;

use super::decode::{self, RawSafetyLabel};
use super::metrics::{
    self, CacheResult, CacheTier, FillKeyResult, LabelMetrics, MhCheckClass, MhCheckSample,
    MhCheckView, ShadowResult, SourceOutcome,
};
use super::remote::{CacheWrite, FallbackReason, LabelSource, ManhattanSource, fallback_reason};
use super::source::{drop_expired_labels, tweet_timestamp_ms};

const KEY_PREFIX: &str = "sls:";
const VALUE_VERSION: u8 = 2;
const REFRESH_STEPS: [Duration; 2] = [Duration::from_secs(30), Duration::from_secs(90)];
const WRITE_TTL_SECS: u32 = 24 * 60 * 60;
const FILL_QUEUE_KEYS: usize = 10_000;
const FILL_BATCH_KEYS: usize = 100;
const OUTAGE_MANHATTAN_KEYS_PER_SEC: u32 = 2_500;
const SHADOW_TASKS: usize = 64;
const SHADOW_DEADLINE: Duration = Duration::from_millis(100);
const MISMATCH_LOGS_PER_SEC: u32 = 20;
const MATCH_CHECK_ONE_IN: u64 = 20;
const MATCH_CHECK_SEED: u64 = 1;
const RAMP_BUCKETS: u64 = 10_000;
const MIN_TTL: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelCacheSwitches {
            pub shadow_buckets: u64,
        pub serve_buckets: u64,
        pub found_ttl: Duration,
    pub not_found_ttl: Duration,
            pub refresh_cutoff_ms: u64,
            pub replay_bucket: Option<u64>,
        pub shadow_mh_check_keys_per_sec: u32,
        pub shadow_fill_keys_per_sec: u32,
}

impl Default for LabelCacheSwitches {
    fn default() -> Self {
        Self {
            shadow_buckets: 0,
            serve_buckets: 0,
            found_ttl: Duration::from_secs(24 * 60 * 60),
            not_found_ttl: Duration::from_secs(10 * 60),
            refresh_cutoff_ms: 0,
            replay_bucket: None,
            shadow_mh_check_keys_per_sec: 0,
            shadow_fill_keys_per_sec: 0,
        }
    }
}

impl LabelCacheSwitches {
    pub(crate) fn serves(&self, tweet_id: u64) -> bool {
        ramp_bucket(tweet_id) < self.serve_buckets
    }

    pub(crate) fn shadows(&self, tweet_id: u64) -> bool {
        let bucket = ramp_bucket(tweet_id);
        bucket < self.shadow_buckets && bucket >= self.serve_buckets
    }

    pub(crate) fn replays(&self, tweet_id: u64) -> bool {
        self.replay_bucket == Some(ramp_bucket(tweet_id))
    }
}

fn ramp_bucket(tweet_id: u64) -> u64 {
    xxhash_rust::xxh64::xxh64(&tweet_id.to_le_bytes(), 0) % RAMP_BUCKETS
}

#[derive(Clone, Copy)]
pub(crate) struct ReadContext {
    pub(crate) now_ms: u64,
    pub(crate) switches: LabelCacheSwitches,
}

#[derive(Debug, Clone, PartialEq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum Status {
    Found(vf_pb::SafetyLabelMap),
    NotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fill {
    filled_at_ms: u64,
    step: u8,
}

impl Fill {
    pub(crate) fn new(filled_at_ms: u64) -> Self {
        Self {
            filled_at_ms,
            step: 0,
        }
    }

    fn due_refresh(self, context: &ReadContext) -> Option<Self> {
        let cutoff_ms = context.switches.refresh_cutoff_ms;
        if self.filled_at_ms < cutoff_ms && cutoff_ms <= context.now_ms {
            return Some(Self::new(context.now_ms));
        }
        let age_ms = context.now_ms.saturating_sub(self.filled_at_ms);
        let due = REFRESH_STEPS
            .iter()
            .take_while(|&&after| age_ms >= millis(after))
            .count();
        let step = u8::try_from(due).ok()?;
        (step > self.step).then_some(Self {
            filled_at_ms: self.filled_at_ms,
            step,
        })
    }
}

#[derive(Clone, PartialEq, prost::Message)]
struct StoredEntry {
    #[prost(uint64, tag = "1")]
    filled_at_ms: u64,
    #[prost(uint32, tag = "2")]
    step: u32,
    #[prost(message, repeated, tag = "3")]
    items: Vec<StoredItem>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct StoredItem {
    #[prost(bytes = "vec", tag = "1")]
    lkey: Vec<u8>,
    #[prost(bytes = "vec", tag = "2")]
    mval: Vec<u8>,
}

fn encode_entry(fill: Fill, items: Vec<RawSafetyLabel>) -> Vec<u8> {
    let entry = StoredEntry {
        filled_at_ms: fill.filled_at_ms,
        step: u32::from(fill.step),
        items: items
            .into_iter()
            .map(|item| StoredItem {
                lkey: item.lkey.to_vec(),
                mval: item.mval,
            })
            .collect(),
    };
    let mut buf = vec![VALUE_VERSION];
    buf.extend(entry.encode_to_vec());
    buf
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Envelope {
    fill: Fill,
    status: Status,
}

impl Envelope {
    fn decode(bytes: &[u8]) -> Option<Self> {
        let (&version, body) = bytes.split_first()?;
        if version != VALUE_VERSION {
            return None;
        }
        let entry = StoredEntry::decode(body).ok()?;
        let fill = Fill {
            filled_at_ms: entry.filled_at_ms,
            step: u8::try_from(entry.step).ok()?,
        };
        if entry.items.is_empty() {
            return Some(Self {
                fill,
                status: Status::NotFound,
            });
        }
        let items = entry
            .items
            .into_iter()
            .map(|item| {
                Some(RawSafetyLabel {
                    lkey: item.lkey.try_into().ok()?,
                    mval: item.mval,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let labels = decode::raw_labels(&items)?;
        Some(Self {
            fill,
            status: Status::Found(twemcache_types_only(labels)),
        })
    }

    fn is_live(&self, context: &ReadContext) -> bool {
        let ttl = match self.status {
            Status::Found(_) => context.switches.found_ttl,
            Status::NotFound => context.switches.not_found_ttl,
        };
        context.now_ms.saturating_sub(self.fill.filled_at_ms) < millis(ttl.max(MIN_TTL))
    }

    fn filtered(&self, now_ms: i64) -> Self {
        let status = match &self.status {
            Status::Found(labels) => Status::Found(filtered_labels(labels, now_ms)),
            Status::NotFound => Status::NotFound,
        };
        Self {
            fill: self.fill,
            status,
        }
    }

    fn answers(&self, labels: &vf_pb::SafetyLabelMap) -> bool {
        match &self.status {
            Status::Found(cached) => cached == labels,
            Status::NotFound => labels.labels.is_empty(),
        }
    }

    pub(crate) fn into_labels(self) -> vf_pb::SafetyLabelMap {
        match self.status {
            Status::Found(labels) => labels,
            Status::NotFound => vf_pb::SafetyLabelMap {
                labels: HashMap::new(),
            },
        }
    }
}

fn filtered_labels(labels: &vf_pb::SafetyLabelMap, now_ms: i64) -> vf_pb::SafetyLabelMap {
    twemcache_types_only(drop_expired_labels(labels, now_ms))
}

fn twemcache_types_only(mut labels: vf_pb::SafetyLabelMap) -> vf_pb::SafetyLabelMap {
    static DECODED: LazyLock<FxHashSet<i32>> =
        LazyLock::new(|| safety_label_metas().iter().map(|meta| meta.value).collect());
    labels
        .labels
        .retain(|label_type, _| DECODED.contains(label_type));
    labels
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum MismatchClass {
        Stale,
    Other,
}

#[derive(Debug, PartialEq)]
struct DifferingLabel {
    label_type: i32,
    entry_created_at_ms: Option<i64>,
    answer_created_at_ms: Option<i64>,
    entry_expires_at_ms: Option<i64>,
    answer_expires_at_ms: Option<i64>,
    entry_source: Option<String>,
    answer_source: Option<String>,
        fields: Vec<&'static str>,
}

#[derive(Debug, PartialEq)]
struct LabelDiff {
    only_entry: Vec<DifferingLabel>,
    only_answer: Vec<DifferingLabel>,
    changed: Vec<DifferingLabel>,
    class: MismatchClass,
}

impl LabelDiff {
    fn new(entry: &Envelope, answer: &vf_pb::SafetyLabelMap) -> Self {
        let entry_labels = match &entry.status {
            Status::Found(labels) => &labels.labels,
            Status::NotFound => &HashMap::new(),
        };
        let label_types: BTreeSet<i32> = entry_labels
            .keys()
            .chain(answer.labels.keys())
            .copied()
            .collect();
        let filled_at_ms = i64::try_from(entry.fill.filled_at_ms).unwrap_or(i64::MAX);
        let mut diff = Self {
            only_entry: Vec::new(),
            only_answer: Vec::new(),
            changed: Vec::new(),
            class: MismatchClass::Other,
        };
        for label_type in label_types {
            let in_entry = entry_labels.get(&label_type);
            let in_answer = answer.labels.get(&label_type);
            let side = match (in_entry, in_answer) {
                (Some(cached), Some(served)) if cached == served => continue,
                (Some(_), Some(_)) => &mut diff.changed,
                (Some(_), None) => &mut diff.only_entry,
                (None, _) => &mut diff.only_answer,
            };
            let answer_created_at_ms = in_answer.and_then(|label| label.created_at_msec);
            if answer_created_at_ms.is_some_and(|at| at > filled_at_ms) {
                diff.class = MismatchClass::Stale;
            }
            side.push(DifferingLabel {
                label_type,
                entry_created_at_ms: in_entry.and_then(|label| label.created_at_msec),
                answer_created_at_ms,
                entry_expires_at_ms: in_entry.and_then(|label| label.expires_at_msec),
                answer_expires_at_ms: in_answer.and_then(|label| label.expires_at_msec),
                entry_source: in_entry.and_then(source_tag),
                answer_source: in_answer.and_then(source_tag),
                fields: in_entry
                    .zip(in_answer)
                    .map_or_else(Vec::new, |(cached, served)| {
                        differing_fields(cached, served)
                    }),
            });
        }
        diff
    }
}

fn source_tag(label: &vf_pb::SafetyLabel) -> Option<String> {
    use vf_pb::safety_label::SafetyLabelSource;
    label
        .safety_label_source
        .as_ref()
        .map(|source| match source {
            SafetyLabelSource::BotmakerAction(a) => format!("botmaker:{}", a.rule_id),
            SafetyLabelSource::ToolAction(a) => format!("tool:{}", a.agent_tool),
            SafetyLabelSource::GrokAnnotationAction(a) => format!("grok:{}", a.source),
        })
}

fn differing_fields(a: &vf_pb::SafetyLabel, b: &vf_pb::SafetyLabel) -> Vec<&'static str> {
    [
        ("score", a.score != b.score),
        ("applicable_users", a.applicable_users != b.applicable_users),
        (
            "holdback_experiment",
            a.holdback_experiment != b.holdback_experiment,
        ),
        ("source", a.source != b.source),
        ("created_at_msec", a.created_at_msec != b.created_at_msec),
        ("expires_at_msec", a.expires_at_msec != b.expires_at_msec),
        (
            "applicable_countries",
            a.applicable_countries != b.applicable_countries,
        ),
        (
            "safety_label_source",
            a.safety_label_source != b.safety_label_source,
        ),
    ]
    .into_iter()
    .filter_map(|(field, differs)| differs.then_some(field))
    .collect()
}

fn cache_key(tweet_id: u64) -> Result<Key, KVCacheError> {
    Key::new(format!("{KEY_PREFIX}{tweet_id}").into_bytes())
}

pub(crate) enum EntryRead {
    Hit {
        envelope: Envelope,
        refresh: Option<Fill>,
    },
    Miss,
    Failed,
}

pub(crate) struct FillJob {
    pub(crate) tweet_id: u64,
    pub(crate) fill: Fill,
    pub(crate) labels: FillLabels,
}

pub(crate) enum FillLabels {
        Scanned(Vec<RawSafetyLabel>),
        Reread,
}

#[derive(Default)]
struct ReadCounts {
    hit: usize,
    not_found: usize,
    miss: usize,
    expired: usize,
}

pub(crate) struct DragonflySource {
    cache: Arc<dyn CacheWrite>,
    manhattan: ManhattanSource,
    fill_tx: mpsc::Sender<FillJob>,
    pending_fills: Arc<Mutex<FxHashSet<u64>>>,
    outage_budget: DefaultDirectRateLimiter,
    mismatch_logs: DefaultDirectRateLimiter,
    mh_check_budget: ArcSwapOption<KeyBudget>,
    shadow_fill_budget: ArcSwapOption<KeyBudget>,
    mh_mismatch_logs: DefaultDirectRateLimiter,
    shadow_tasks: Arc<Semaphore>,
    metrics: LabelMetrics,
}

struct KeyBudget {
    keys_per_sec: NonZeroU32,
    limiter: DefaultDirectRateLimiter,
}

struct ShadowHit {
    tweet_id: u64,
    sample: MhCheckSample,
    entry: Envelope,
    answer: vf_pb::SafetyLabelMap,
}

impl DragonflySource {
    pub(crate) fn spawn(
        cache: Arc<dyn CacheWrite>,
        manhattan: ManhattanSource,
        metrics: LabelMetrics,
    ) -> Arc<Self> {
        let (source, rx) = Self::without_fill_task(cache, manhattan, metrics, FILL_QUEUE_KEYS);
        let writer = Writer {
            cache: Arc::clone(&source.cache),
            manhattan: source.manhattan.clone(),
            pending_fills: Arc::clone(&source.pending_fills),
            metrics: source.metrics.clone(),
        };
        tokio::spawn(drain(rx, writer));
        Arc::new(source)
    }

    fn without_fill_task(
        cache: Arc<dyn CacheWrite>,
        manhattan: ManhattanSource,
        metrics: LabelMetrics,
        queue_keys: usize,
    ) -> (Self, mpsc::Receiver<FillJob>) {
        let (fill_tx, rx) = mpsc::channel(queue_keys);
        let per_second = |count| {
            RateLimiter::direct(Quota::per_second(
                NonZeroU32::new(count).unwrap_or(NonZeroU32::MIN),
            ))
        };
        let source = Self {
            cache,
            manhattan,
            fill_tx,
            pending_fills: Arc::default(),
            outage_budget: per_second(OUTAGE_MANHATTAN_KEYS_PER_SEC),
            mismatch_logs: per_second(MISMATCH_LOGS_PER_SEC),
            mh_check_budget: ArcSwapOption::empty(),
            shadow_fill_budget: ArcSwapOption::empty(),
            mh_mismatch_logs: per_second(MISMATCH_LOGS_PER_SEC),
            shadow_tasks: Arc::new(Semaphore::new(SHADOW_TASKS)),
            metrics,
        };
        (source, rx)
    }

    pub(crate) async fn read(
        &self,
        ids: &[u64],
        context: &ReadContext,
        tier: CacheTier,
    ) -> Vec<(u64, EntryRead)> {
        let start = Instant::now();
        let keys: Vec<Option<Key>> = ids.iter().map(|&id| cache_key(id).ok()).collect();
        let present: Vec<Key> = keys.iter().flatten().cloned().collect();
        let replies = self.cache.multi_get(&present).await;

        let mut counts = ReadCounts::default();
        let mut failures: BTreeMap<FallbackReason, usize> = BTreeMap::new();
        let mut reads = Vec::with_capacity(ids.len());
        for (&id, key) in ids.iter().zip(&keys) {
            let reply = key.as_ref().map(|key| replies.get(key));
            let read = match reply {
                Some(Some(Ok(Some(bytes)))) => match Envelope::decode(bytes) {
                    Some(envelope) if envelope.is_live(context) => {
                        match envelope.status {
                            Status::Found(_) => counts.hit += 1,
                            Status::NotFound => counts.not_found += 1,
                        }
                        EntryRead::Hit {
                            refresh: envelope.fill.due_refresh(context),
                            envelope,
                        }
                    }
                    Some(_) => {
                        counts.expired += 1;
                        EntryRead::Miss
                    }
                    None => {
                        counts.miss += 1;
                        EntryRead::Miss
                    }
                },
                Some(Some(Ok(None))) => {
                    counts.miss += 1;
                    EntryRead::Miss
                }
                Some(Some(Err(e))) => {
                    *failures.entry(fallback_reason(e)).or_default() += 1;
                    EntryRead::Failed
                }
                Some(None) => {
                    *failures.entry(FallbackReason::MissingResponse).or_default() += 1;
                    EntryRead::Failed
                }
                None => {
                    *failures.entry(FallbackReason::Other).or_default() += 1;
                    EntryRead::Failed
                }
            };
            reads.push((id, read));
        }

        let m = &self.metrics;
        metrics::record_source_request(
            m,
            LabelSource::Dragonfly,
            SourceOutcome::Success,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        metrics::record_cache_keys(m, tier, CacheResult::Hit, counts.hit);
        metrics::record_cache_keys(m, tier, CacheResult::NotFound, counts.not_found);
        metrics::record_cache_keys(m, tier, CacheResult::Miss, counts.miss);
        metrics::record_cache_keys(m, tier, CacheResult::Expired, counts.expired);
        for (reason, count) in failures {
            metrics::record_cache_fallback_keys(m, LabelSource::Dragonfly, reason, count);
        }
        reads
    }

        pub(crate) fn fill(&self, jobs: Vec<FillJob>) {
        if jobs.is_empty() {
            return;
        }
        let mut fresh = Vec::with_capacity(jobs.len());
        let mut deduped = Vec::with_capacity(jobs.len());
        {
            let mut pending = self
                .pending_fills
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for job in jobs {
                if pending.insert(job.tweet_id) {
                    fresh.push(job);
                } else {
                    deduped.push(job);
                }
            }
        }
        let enqueued = fresh.len();
        let mut dropped = Vec::new();
        for job in fresh {
            let tweet_id = job.tweet_id;
            if self.fill_tx.try_send(job).is_err() {
                dropped.push(tweet_id);
            }
        }
        if !dropped.is_empty() {
            let mut pending = self
                .pending_fills
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for tweet_id in &dropped {
                pending.remove(tweet_id);
            }
        }
        let record = |result, count| metrics::record_cache_fill_keys(&self.metrics, result, count);
        record(FillKeyResult::Enqueued, enqueued - dropped.len());
        record(FillKeyResult::Deduped, deduped.len());
        record(FillKeyResult::DroppedQueueFull, dropped.len());
    }

    pub(crate) fn admit_outage_read(&self) -> bool {
        self.outage_budget.check().is_ok()
    }

            pub(crate) fn shadow_permit(&self, keys: usize) -> Option<OwnedSemaphorePermit> {
        let permit = Arc::clone(&self.shadow_tasks).try_acquire_owned().ok();
        if permit.is_none() {
            record_shed(&self.metrics, keys);
        }
        permit
    }

    pub(crate) fn shadow(
        self: &Arc<Self>,
        permit: OwnedSemaphorePermit,
        answers: Vec<(u64, vf_pb::SafetyLabelMap)>,
        context: ReadContext,
    ) {
        let source = Arc::clone(self);
        tokio::spawn(async move {
            let _permit = permit;
            let deadline = tokio::time::Instant::now() + SHADOW_DEADLINE;
            let count = answers.len();
            let compared = source.compare(answers, &context);
            match tokio::time::timeout_at(deadline, compared).await {
                Ok(hits) => source.check_manhattan(hits, context.now_ms, deadline).await,
                Err(_) => record_shed(&source.metrics, count),
            }
        });
    }

    async fn compare(
        &self,
        answers: Vec<(u64, vf_pb::SafetyLabelMap)>,
        context: &ReadContext,
    ) -> Vec<ShadowHit> {
        let ids: Vec<u64> = answers.iter().map(|(id, _)| *id).collect();
        let reads = self.read(&ids, context, CacheTier::DragonflyShadow).await;
        let budget = key_budget(
            &self.mh_check_budget,
            context.switches.shadow_mh_check_keys_per_sec,
        );
        let fill_budget = key_budget(
            &self.shadow_fill_budget,
            context.switches.shadow_fill_keys_per_sec,
        );
        let served_at_ms = i64::try_from(context.now_ms).unwrap_or(i64::MAX);
        let (mut matched, mut mismatched, mut failed) = (0, 0, 0);
        let mut would_read = 0;
        let mut fills = Vec::new();
        let mut hits = Vec::new();
        for ((tweet_id, read), (_, answer)) in reads.into_iter().zip(answers) {
            let job = match read {
                EntryRead::Hit { envelope, refresh } => {
                    let filtered_entry = envelope.filtered(served_at_ms);
                    let sample = if filtered_entry.answers(&filtered_labels(&answer, served_at_ms))
                    {
                        matched += 1;
                        MhCheckSample::Match
                    } else {
                        mismatched += 1;
                        if self.mismatch_logs.check().is_ok() {
                            log_mismatch(tweet_id, &envelope, &answer, context.now_ms);
                        }
                        MhCheckSample::Mismatch
                    };
                    if budget.is_some() {
                        hits.push(ShadowHit {
                            tweet_id,
                            sample,
                            entry: envelope,
                            answer: answer.clone(),
                        });
                    }
                    if refresh.is_some() {
                        would_read += 1;
                    }
                    refresh.map(|fill| (fill, FillLabels::Reread))
                }
                EntryRead::Miss => {
                    would_read += 1;
                    fill_budget
                        .as_ref()
                        .is_some_and(|budget| budget.limiter.check().is_ok())
                        .then(|| (Fill::new(context.now_ms), FillLabels::Reread))
                }
                EntryRead::Failed => {
                    failed += 1;
                    None
                }
            };
            if let Some((fill, labels)) = job {
                fills.push(FillJob {
                    tweet_id,
                    fill,
                    labels,
                });
            }
        }
        let m = &self.metrics;
        metrics::record_shadow_keys(m, ShadowResult::Match, matched);
        metrics::record_shadow_keys(m, ShadowResult::Mismatch, mismatched);
        metrics::record_shadow_keys(m, ShadowResult::Failed, failed);
        metrics::record_shadow_manhattan_keys(m, would_read + failed);
        self.fill(fills);
        match budget {
            Some(budget) => sample_mh_checks(hits, || budget.limiter.check().is_ok()),
            None => Vec::new(),
        }
    }

                async fn check_manhattan(
        &self,
        hits: Vec<ShadowHit>,
        now_ms: u64,
        deadline: tokio::time::Instant,
    ) {
        if hits.is_empty() {
            return;
        }
        let ids: Vec<u64> = hits.iter().map(|hit| hit.tweet_id).collect();
        let mut manhattan = tokio::time::timeout_at(deadline, self.manhattan.scan(&ids))
            .await
            .unwrap_or_default();
        let served_at_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let mut counts: BTreeMap<(MhCheckClass, MhCheckSample, MhCheckView), usize> =
            BTreeMap::new();
        let mut repairs = Vec::new();
        for hit in hits {
            let (items, labels) = match manhattan.remove(&hit.tweet_id).and_then(Result::ok) {
                Some(scan) => (Some(scan.items), Some(scan.labels)),
                None => (None, None),
            };
            let raw_class = mh_check_class(&hit.entry, &hit.answer, labels.as_ref());
            *counts
                .entry((raw_class, hit.sample, MhCheckView::Raw))
                .or_default() += 1;
            let filtered = labels
                .as_ref()
                .map(|labels| filtered_labels(labels, served_at_ms));
            let class = mh_check_class(
                &hit.entry.filtered(served_at_ms),
                &filtered_labels(&hit.answer, served_at_ms),
                filtered.as_ref(),
            );
            *counts
                .entry((class, hit.sample, MhCheckView::Filtered))
                .or_default() += 1;
            if let Some(labels) = &labels
                && matches!(
                    class,
                    MhCheckClass::DragonflyDiffers
                        | MhCheckClass::AllDiffer
                        | MhCheckClass::ManhattanDiffers
                )
                && self.mh_mismatch_logs.check().is_ok()
            {
                log_mh_mismatch(&hit, labels, class, now_ms);
            }
            if let Some(items) = items
                && matches!(
                    class,
                    MhCheckClass::DragonflyDiffers
                        | MhCheckClass::AllDiffer
                        | MhCheckClass::ManhattanDiffers
                )
            {
                repairs.push(FillJob {
                    tweet_id: hit.tweet_id,
                    fill: Fill::new(now_ms),
                    labels: FillLabels::Scanned(items),
                });
            }
        }
        for ((class, sample, view), count) in counts {
            metrics::record_shadow_mh_check_keys(&self.metrics, class, sample, view, count);
        }
        self.fill(repairs);
    }
}

fn key_budget(slot: &ArcSwapOption<KeyBudget>, keys_per_sec: u32) -> Option<Arc<KeyBudget>> {
    let keys_per_sec = NonZeroU32::new(keys_per_sec)?;
    if let Some(budget) = slot.load_full()
        && budget.keys_per_sec == keys_per_sec
    {
        return Some(budget);
    }
    let budget = Arc::new(KeyBudget {
        keys_per_sec,
        limiter: RateLimiter::direct(Quota::per_second(keys_per_sec)),
    });
    slot.store(Some(Arc::clone(&budget)));
    Some(budget)
}

fn sample_mh_checks(hits: Vec<ShadowHit>, mut admit: impl FnMut() -> bool) -> Vec<ShadowHit> {
    let (mismatches, matches): (Vec<_>, Vec<_>) = hits
        .into_iter()
        .partition(|hit| hit.sample == MhCheckSample::Mismatch);
    mismatches
        .into_iter()
        .chain(
            matches
                .into_iter()
                .filter(|hit| in_match_check_slice(hit.tweet_id)),
        )
        .take_while(|_| admit())
        .collect()
}

fn in_match_check_slice(tweet_id: u64) -> bool {
    xxhash_rust::xxh64::xxh64(&tweet_id.to_le_bytes(), MATCH_CHECK_SEED)
        .is_multiple_of(MATCH_CHECK_ONE_IN)
}

fn mh_check_class(
    entry: &Envelope,
    answer: &vf_pb::SafetyLabelMap,
    manhattan: Option<&vf_pb::SafetyLabelMap>,
) -> MhCheckClass {
    let Some(manhattan) = manhattan else {
        return MhCheckClass::ManhattanFailed;
    };
    match (
        entry.answers(answer),
        entry.answers(manhattan),
        answer == manhattan,
    ) {
        (true, true, _) => MhCheckClass::AllEqual,
        (false, true, _) => MhCheckClass::TwemcacheDiffers,
        (true, false, _) => MhCheckClass::ManhattanDiffers,
        (false, false, true) => MhCheckClass::DragonflyDiffers,
        (false, false, false) => MhCheckClass::AllDiffer,
    }
}

fn log_mismatch(tweet_id: u64, entry: &Envelope, answer: &vf_pb::SafetyLabelMap, now_ms: u64) {
    let diff = LabelDiff::new(entry, answer);
    warn!(
        tweet_id,
        post_age_ms = now_ms.saturating_sub(tweet_timestamp_ms(tweet_id)),
        entry_age_ms = now_ms.saturating_sub(entry.fill.filled_at_ms),
        entry_step = entry.fill.step,
        entry_status = <&'static str>::from(&entry.status),
        class = <&'static str>::from(diff.class),
        only_entry = ?diff.only_entry,
        only_answer = ?diff.only_answer,
        changed = ?diff.changed,
        "label cache shadow mismatch"
    );
}

fn log_mh_mismatch(
    hit: &ShadowHit,
    manhattan: &vf_pb::SafetyLabelMap,
    mh_class: MhCheckClass,
    now_ms: u64,
) {
    let entry = &hit.entry;
    let diff = LabelDiff::new(entry, &hit.answer);
    let mh_diff = LabelDiff::new(entry, manhattan);
    warn!(
        tweet_id = hit.tweet_id,
        post_age_ms = now_ms.saturating_sub(tweet_timestamp_ms(hit.tweet_id)),
        entry_age_ms = now_ms.saturating_sub(entry.fill.filled_at_ms),
        entry_step = entry.fill.step,
        entry_status = <&'static str>::from(&entry.status),
        class = <&'static str>::from(diff.class),
        only_entry = ?diff.only_entry,
        only_answer = ?diff.only_answer,
        changed = ?diff.changed,
        mh_class = <&'static str>::from(mh_class),
        mh_only_entry = ?mh_diff.only_entry,
        only_manhattan = ?mh_diff.only_answer,
        mh_changed = ?mh_diff.changed,
        "label cache shadow manhattan mismatch"
    );
}

fn record_shed(metrics: &LabelMetrics, count: usize) {
    metrics::record_shadow_keys(metrics, ShadowResult::Shed, count);
    metrics::record_shadow_manhattan_keys(metrics, count);
}

struct Writer {
    cache: Arc<dyn CacheWrite>,
    manhattan: ManhattanSource,
    pending_fills: Arc<Mutex<FxHashSet<u64>>>,
    metrics: LabelMetrics,
}

impl Writer {
        async fn write(&self, jobs: Vec<FillJob>) {
        let ids: Vec<u64> = jobs.iter().map(|job| job.tweet_id).collect();
        let reread: Vec<u64> = jobs
            .iter()
            .filter(|job| matches!(job.labels, FillLabels::Reread))
            .map(|job| job.tweet_id)
            .collect();
        let mut reread_results = self.manhattan.scan(&reread).await;
        let writes: Vec<_> = jobs
            .into_iter()
            .filter_map(|job| {
                let items = match job.labels {
                    FillLabels::Scanned(items) => items,
                    FillLabels::Reread => reread_results.remove(&job.tweet_id)?.ok()?.items,
                };
                let key = cache_key(job.tweet_id).ok()?;
                let value = encode_entry(job.fill, items);
                Some(async move { self.cache.set(&key, value, WRITE_TTL_SECS).await.is_ok() })
            })
            .collect();
        let written = futures::future::join_all(writes)
            .await
            .into_iter()
            .filter(|&written| written)
            .count();
        metrics::record_cache_fill_keys(&self.metrics, FillKeyResult::Written, written);
        metrics::record_cache_fill_keys(&self.metrics, FillKeyResult::Failed, ids.len() - written);
        let mut pending = self
            .pending_fills
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for tweet_id in &ids {
            pending.remove(tweet_id);
        }
    }
}

async fn drain(mut rx: mpsc::Receiver<FillJob>, writer: Writer) {
    let mut jobs = Vec::with_capacity(FILL_BATCH_KEYS);
    while rx.recv_many(&mut jobs, FILL_BATCH_KEYS).await > 0 {
        writer.write(std::mem::take(&mut jobs)).await;
    }
}

pub(crate) fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tonic::async_trait;
    use xai_cache::Value;
    use xai_manhattan::ManhattanError;
    use xai_safety_label_store::types::encode_lkey;
    use xai_stats_receiver::{HistogramBuckets, StatsReceiverExt};
    use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelType};

    use crate::decode::tests::{cached_value_found_mval, minimal_mval, mval_with_score, raw_label};
    use crate::mh_client::{FetchResult, ManhattanLabelFetcher};
    use crate::remote::{
        self, CacheOpResult, FailureKind, LookupError, NowMs, RemoteLabels, RemoteSource,
    };
    use crate::source::SafetyLabelSource;

    const BASE_MS: u64 = 1_700_000_000_000;
    const DAY_SECS: u32 = 86_400;

    #[derive(Default)]
    struct FakeDragonfly {
        entries: Mutex<HashMap<Key, (Value, u32)>>,
        failing: AtomicBool,
        failing_keys: Mutex<FxHashSet<Key>>,
        keys_read: AtomicUsize,
    }

    impl FakeDragonfly {
        fn fail(&self, id: u64) {
            self.failing_keys
                .lock()
                .unwrap()
                .insert(cache_key(id).unwrap());
        }

        fn entry(&self, id: u64) -> Option<(Envelope, u32)> {
            let entries = self.entries.lock().unwrap();
            let (bytes, ttl) = entries.get(&cache_key(id).unwrap())?;
            Some((Envelope::decode(bytes).unwrap(), *ttl))
        }

        fn put_bytes(&self, id: u64, bytes: Vec<u8>) {
            self.entries
                .lock()
                .unwrap()
                .insert(cache_key(id).unwrap(), (bytes, DAY_SECS));
        }

        fn put(&self, id: u64, status: Status, fill: Fill) {
            self.put_bytes(id, stored(&Envelope { fill, status }));
        }

        fn stored_items(&self, id: u64) -> Vec<StoredItem> {
            let entries = self.entries.lock().unwrap();
            let (bytes, _) = entries.get(&cache_key(id).unwrap()).unwrap();
            StoredEntry::decode(&bytes[1..]).unwrap().items
        }
    }

    #[async_trait]
    impl remote::CacheRead for FakeDragonfly {
        async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>> {
            self.keys_read.fetch_add(keys.len(), Ordering::SeqCst);
            let entries = self.entries.lock().unwrap();
            let failing = self.failing.load(Ordering::SeqCst);
            let failing_keys = self.failing_keys.lock().unwrap();
            keys.iter()
                .map(|key| {
                    let reply = if failing || failing_keys.contains(key) {
                        Err(KVCacheError::Timeout("get".into()))
                    } else {
                        Ok(entries.get(key).map(|(value, _)| value.clone()))
                    };
                    (key.clone(), reply)
                })
                .collect()
        }
    }

    #[async_trait]
    impl CacheWrite for FakeDragonfly {
        async fn set(&self, key: &Key, value: Value, ttl_secs: u32) -> CacheOpResult<()> {
            self.entries
                .lock()
                .unwrap()
                .insert(key.clone(), (value, ttl_secs));
            Ok(())
        }
    }

    #[derive(Default)]
    struct FakeTwemcache {
        found: Mutex<FxHashSet<u64>>,
        keys_read: AtomicUsize,
    }

    #[async_trait]
    impl remote::CacheRead for FakeTwemcache {
        async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>> {
            self.keys_read.fetch_add(keys.len(), Ordering::SeqCst);
            let found: Vec<Key> = self
                .found
                .lock()
                .unwrap()
                .iter()
                .map(|&id| remote::cache_key(id).unwrap())
                .collect();
            keys.iter()
                .map(|key| {
                    let value = found.contains(key).then(cached_value_found_mval);
                    (key.clone(), Ok(value))
                })
                .collect()
        }
    }

    #[derive(Default)]
    struct FakeManhattan {
        scores: Mutex<HashMap<i64, f64>>,
        others: Mutex<HashMap<i64, Vec<SafetyLabelType>>>,
        calls: Mutex<Vec<Vec<i64>>>,
    }

    impl FakeManhattan {
        fn set_spam(&self, id: i64, score: f64) {
            self.scores.lock().unwrap().insert(id, score);
        }

        fn add_label(&self, id: i64, label_type: SafetyLabelType) {
            self.others
                .lock()
                .unwrap()
                .entry(id)
                .or_default()
                .push(label_type);
        }

        fn calls(&self) -> Vec<Vec<i64>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ManhattanLabelFetcher for FakeManhattan {
        async fn fetch_labels(&self, ids: &[i64]) -> Result<Vec<FetchResult>, ManhattanError> {
            self.calls.lock().unwrap().push(ids.to_vec());
            let scores = self.scores.lock().unwrap();
            let others = self.others.lock().unwrap();
            Ok(ids
                .iter()
                .map(|id| {
                    let spam = scores
                        .get(id)
                        .map(|&s| raw_label(SafetyLabelType::SPAM, mval_with_score(s)));
                    let others = others
                        .get(id)
                        .into_iter()
                        .flatten()
                        .map(|&label_type| raw_label(label_type, minimal_mval()));
                    Ok(spam
                        .into_iter()
                        .chain(others)
                        .collect::<Vec<RawSafetyLabel>>())
                })
                .collect())
        }
    }

    #[derive(Default)]
    struct RecordedStats(Mutex<HashMap<String, u64>>);

    impl RecordedStats {
        fn count(&self, name: &str, scopes: &[(&str, &str)]) -> u64 {
            let key = format!("{name}{scopes:?}");
            self.0.lock().unwrap().get(&key).copied().unwrap_or(0)
        }
    }

    impl StatsReceiverExt for RecordedStats {
        fn incr(&self, name: &str, scopes: &[(&str, &str)], value: u64) {
            *self
                .0
                .lock()
                .unwrap()
                .entry(format!("{name}{scopes:?}"))
                .or_default() += value;
        }
        fn observe(&self, _: &str, _: &[(&str, &str)], _: f64, _: HistogramBuckets) {}
        fn observe_expo(&self, _: &str, _: &[(&str, &str)], _: f64) {}
        fn observe_vm(&self, _: &str, _: &[(&str, &str)], _: f64) {}
        fn gauge(&self, _: &str, _: &[(&str, &str)], _: f64) {}
    }

    fn paused_clock() -> NowMs {
        let start = tokio::time::Instant::now();
        Arc::new(move || BASE_MS + start.elapsed().as_millis() as u64)
    }

    struct World {
        dragonfly: Arc<FakeDragonfly>,
        twemcache: Arc<FakeTwemcache>,
        manhattan: Arc<FakeManhattan>,
        stats: Arc<RecordedStats>,
        source: SafetyLabelSource,
    }

    struct Built {
        world: World,
        label_cache: Arc<DragonflySource>,
        fill_rx: Option<mpsc::Receiver<FillJob>>,
    }

    fn build(
        switches: LabelCacheSwitches,
        hot_keys: Option<quanta::Clock>,
        undrained_fill_queue: Option<usize>,
    ) -> Built {
        let dragonfly = Arc::new(FakeDragonfly::default());
        let twemcache = Arc::new(FakeTwemcache::default());
        let manhattan = Arc::new(FakeManhattan::default());
        let stats = Arc::new(RecordedStats::default());
        let metrics = LabelMetrics::new(Some(Arc::<RecordedStats>::clone(&stats) as _));
        let remote = RemoteSource::new(
            Arc::<FakeTwemcache>::clone(&twemcache),
            Arc::<FakeManhattan>::clone(&manhattan),
            metrics.clone(),
        );
        let cache = Arc::<FakeDragonfly>::clone(&dragonfly);
        let (label_cache, fill_rx) = match undrained_fill_queue {
            None => (
                DragonflySource::spawn(cache, remote.manhattan(), metrics.clone()),
                None,
            ),
            Some(keys) => {
                let (source, rx) = DragonflySource::without_fill_task(
                    cache,
                    remote.manhattan(),
                    metrics.clone(),
                    keys,
                );
                (Arc::new(source), Some(rx))
            }
        };
        let remote = remote.with_test_dragonfly(Arc::clone(&label_cache), paused_clock());
        remote.set_switches(switches);
        let remote = Arc::new(remote);
        let source = match hot_keys {
            Some(clock) => SafetyLabelSource::with_clock(remote, Some(4096), clock, metrics),
            None => SafetyLabelSource::new(remote, None, metrics),
        };
        Built {
            world: World {
                dragonfly,
                twemcache,
                manhattan,
                stats,
                source,
            },
            label_cache,
            fill_rx,
        }
    }

    fn world(switches: LabelCacheSwitches) -> World {
        build(switches, None, None).world
    }

    fn serving_all() -> LabelCacheSwitches {
        LabelCacheSwitches {
            serve_buckets: RAMP_BUCKETS,
            ..LabelCacheSwitches::default()
        }
    }

    fn spam(score: f64) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                i32::from(SafetyLabelType::SPAM),
                vf_pb::SafetyLabel {
                    score: Some(score),
                    ..vf_pb::SafetyLabel::default()
                },
            )]),
        }
    }

    fn nsfa() -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                i32::from(SafetyLabelType::NSFA_HIGH_PRECISION),
                vf_pb::SafetyLabel::default(),
            )]),
        }
    }

    fn empty() -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap::default()
    }

    type Answers = rustc_hash::FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>;

    fn labels(answers: &Answers, id: u64) -> vf_pb::SafetyLabelMap {
        (**answers[&id].as_ref().unwrap()).clone()
    }

    async fn settle() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    fn fill(filled_at_ms: u64, step: u8) -> Fill {
        Fill { filled_at_ms, step }
    }

        fn stored(envelope: &Envelope) -> Vec<u8> {
        let items = match &envelope.status {
            Status::Found(labels) => labels
                .labels
                .iter()
                .map(|(&label_type, label)| RawSafetyLabel {
                    lkey: encode_lkey(SafetyLabelType(label_type)),
                    mval: crate::testing::mval(&manhattan_label(label)).unwrap(),
                })
                .collect(),
            Status::NotFound => Vec::new(),
        };
        encode_entry(envelope.fill, items)
    }

    fn manhattan_label(label: &vf_pb::SafetyLabel) -> SafetyLabel {
        use vf_pb::safety_label::SafetyLabelSource as Source;
        use xai_x_thrift::tweet_safety_label as thrift;
        let source = label
            .safety_label_source
            .as_ref()
            .map(|source| match source {
                Source::BotmakerAction(a) => {
                    thrift::SafetyLabelSource::BotMakerAction(thrift::BotMakerAction {
                        rule_id: a.rule_id,
                    })
                }
                Source::ToolAction(a) => {
                    thrift::SafetyLabelSource::ToolAction(thrift::ToolAction {
                        agent_tool: thrift::AgentTool(a.agent_tool),
                        actor_ldap: a.actor_ldap.clone(),
                    })
                }
                Source::GrokAnnotationAction(a) => {
                    thrift::SafetyLabelSource::GrokAnnotationAction(thrift::GrokAnnotationAction {
                        grok_annotation_source: thrift::GrokAnnotationSource(a.source),
                    })
                }
            });
        SafetyLabel {
            score: label.score.map(Into::into),
            applicable_users: (!label.applicable_users.is_empty()).then(|| {
                label
                    .applicable_users
                    .iter()
                    .map(|&user_id| thrift::PerspectivalUser { user_id })
                    .collect()
            }),
            holdback_experiment: label.holdback_experiment,
            source: label.source.clone(),
            safety_label_source: source,
            created_at_msec: label.created_at_msec,
            expires_at_msec: label.expires_at_msec,
            applicable_countries: (!label.applicable_countries.is_empty())
                .then(|| label.applicable_countries.clone()),
        }
    }

    #[test]
    fn an_entry_round_trips_and_an_unknown_version_or_malformed_body_reads_as_none() {
        let found = Envelope {
            fill: fill(BASE_MS, 1),
            status: Status::Found(spam(0.5)),
        };
        let not_found = Envelope {
            fill: fill(BASE_MS + 1, 2),
            status: Status::NotFound,
        };
        for envelope in [&found, &not_found] {
            assert_eq!(Envelope::decode(&stored(envelope)).as_ref(), Some(envelope));
        }

        let mut other_version = stored(&found);
        other_version[0] = VALUE_VERSION + 1;
        let body = |entry: StoredEntry| [vec![VALUE_VERSION], entry.encode_to_vec()].concat();
        let short_lkey = body(StoredEntry {
            filled_at_ms: BASE_MS,
            step: 0,
            items: vec![StoredItem {
                lkey: vec![0x80, 0x00, 0x00],
                mval: minimal_mval(),
            }],
        });
        let step_past_u8 = body(StoredEntry {
            filled_at_ms: BASE_MS,
            step: 256,
            items: Vec::new(),
        });
        for bytes in [
            other_version,
            short_lkey,
            step_past_u8,
            vec![VALUE_VERSION, 0xFF],
            Vec::new(),
        ] {
            assert_eq!(Envelope::decode(&bytes), None);
        }
    }

    #[test]
    fn an_entry_stores_the_manhattan_items_as_scanned() {
        let item = RawSafetyLabel {
            lkey: [0x80, 0x00, 0x00, 0x01],
            mval: vec![0xDE, 0xAD],
        };
        let filled_at = [0x80, 0xD0, 0x95, 0xFF, 0xBC, 0x31];
        assert_eq!(
            encode_entry(fill(BASE_MS, 1), vec![item]),
            [
                &[VALUE_VERSION, 0x08][..],
                &filled_at,
                &[
                    0x10, 0x01, 0x1A, 0x0A, 0x0A, 0x04, 0x80, 0x00, 0x00, 0x01, 0x12, 0x02
                ],
                &[0xDE, 0xAD],
            ]
            .concat()
        );
        assert_eq!(
            encode_entry(fill(BASE_MS, 0), Vec::new()),
            [&[VALUE_VERSION, 0x08][..], &filled_at].concat()
        );
    }

    fn label_created_at(
        label_type: SafetyLabelType,
        score: f64,
        created_at_ms: Option<u64>,
    ) -> (i32, vf_pb::SafetyLabel) {
        let label = vf_pb::SafetyLabel {
            score: Some(score),
            created_at_msec: created_at_ms.map(u64::cast_signed),
            ..vf_pb::SafetyLabel::default()
        };
        (i32::from(label_type), label)
    }

    fn found(labels: impl IntoIterator<Item = (i32, vf_pb::SafetyLabel)>) -> Envelope {
        Envelope {
            fill: fill(BASE_MS, 1),
            status: Status::Found(label_map(labels)),
        }
    }

    fn label_map(
        labels: impl IntoIterator<Item = (i32, vf_pb::SafetyLabel)>,
    ) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: labels.into_iter().collect(),
        }
    }

    #[test]
    fn a_label_the_answer_added_after_the_fill_is_stale() {
        let entry = found([label_created_at(
            SafetyLabelType::SPAM,
            0.5,
            Some(BASE_MS - 10),
        )]);
        let answer = label_map([
            label_created_at(SafetyLabelType::SPAM, 0.5, Some(BASE_MS - 10)),
            label_created_at(SafetyLabelType::ABUSIVE, 1.0, Some(BASE_MS + 5_000)),
        ]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(
            diff,
            LabelDiff {
                only_entry: vec![],
                only_answer: vec![DifferingLabel {
                    label_type: i32::from(SafetyLabelType::ABUSIVE),
                    entry_created_at_ms: None,
                    answer_created_at_ms: Some((BASE_MS + 5_000).cast_signed()),
                    entry_expires_at_ms: None,
                    answer_expires_at_ms: None,
                    entry_source: None,
                    answer_source: None,
                    fields: vec![],
                }],
                changed: vec![],
                class: MismatchClass::Stale,
            }
        );
    }

    #[test]
    fn a_label_the_answer_removed_is_other_even_when_the_entry_label_postdates_the_fill() {
        let entry = found([
            label_created_at(SafetyLabelType::SPAM, 0.5, Some(BASE_MS - 10)),
            label_created_at(SafetyLabelType::ABUSIVE, 1.0, Some(BASE_MS + 30_000)),
        ]);
        let answer = label_map([label_created_at(
            SafetyLabelType::SPAM,
            0.5,
            Some(BASE_MS - 10),
        )]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(
            diff,
            LabelDiff {
                only_entry: vec![DifferingLabel {
                    label_type: i32::from(SafetyLabelType::ABUSIVE),
                    entry_created_at_ms: Some((BASE_MS + 30_000).cast_signed()),
                    answer_created_at_ms: None,
                    entry_expires_at_ms: None,
                    answer_expires_at_ms: None,
                    entry_source: None,
                    answer_source: None,
                    fields: vec![],
                }],
                only_answer: vec![],
                changed: vec![],
                class: MismatchClass::Other,
            }
        );
    }

    #[test]
    fn a_label_on_both_sides_with_different_values_is_changed_with_both_timestamps_and_fields() {
        let entry = found([label_created_at(
            SafetyLabelType::SPAM,
            0.5,
            Some(BASE_MS - 10),
        )]);
        let answer = label_map([label_created_at(
            SafetyLabelType::SPAM,
            0.9,
            Some(BASE_MS + 20_000),
        )]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(
            diff,
            LabelDiff {
                only_entry: vec![],
                only_answer: vec![],
                changed: vec![DifferingLabel {
                    label_type: i32::from(SafetyLabelType::SPAM),
                    entry_created_at_ms: Some((BASE_MS - 10).cast_signed()),
                    answer_created_at_ms: Some((BASE_MS + 20_000).cast_signed()),
                    entry_expires_at_ms: None,
                    answer_expires_at_ms: None,
                    entry_source: None,
                    answer_source: None,
                    fields: vec!["score", "created_at_msec"],
                }],
                class: MismatchClass::Stale,
            }
        );
    }

    fn spam_expiring_at(score: f64, expires_at_ms: u64) -> (i32, vf_pb::SafetyLabel) {
        let label = vf_pb::SafetyLabel {
            score: Some(score),
            expires_at_msec: Some(expires_at_ms.cast_signed()),
            ..vf_pb::SafetyLabel::default()
        };
        (i32::from(SafetyLabelType::SPAM), label)
    }

    #[test]
    fn a_differing_label_carries_each_sides_expiry() {
        let entry = found([spam_expiring_at(0.5, BASE_MS + 1_000)]);
        let answer = label_map([spam_expiring_at(0.5, BASE_MS + 2_000)]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(
            diff.changed,
            vec![DifferingLabel {
                label_type: i32::from(SafetyLabelType::SPAM),
                entry_created_at_ms: None,
                answer_created_at_ms: None,
                entry_expires_at_ms: Some((BASE_MS + 1_000).cast_signed()),
                answer_expires_at_ms: Some((BASE_MS + 2_000).cast_signed()),
                entry_source: None,
                answer_source: None,
                fields: vec!["expires_at_msec"],
            }]
        );
    }

    fn spam_from(source: vf_pb::safety_label::SafetyLabelSource) -> (i32, vf_pb::SafetyLabel) {
        let label = vf_pb::SafetyLabel {
            safety_label_source: Some(source),
            ..vf_pb::SafetyLabel::default()
        };
        (i32::from(SafetyLabelType::SPAM), label)
    }

    #[test]
    fn a_changed_label_whose_source_arm_differs_shows_each_sides_source() {
        use vf_pb::safety_label::SafetyLabelSource;
        let entry = found([spam_from(SafetyLabelSource::BotmakerAction(
            vf_pb::BotmakerAction { rule_id: 7 },
        ))]);
        let answer = label_map([spam_from(SafetyLabelSource::GrokAnnotationAction(
            vf_pb::GrokAnnotationAction { source: 3 },
        ))]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(
            diff.changed,
            vec![DifferingLabel {
                label_type: i32::from(SafetyLabelType::SPAM),
                entry_created_at_ms: None,
                answer_created_at_ms: None,
                entry_expires_at_ms: None,
                answer_expires_at_ms: None,
                entry_source: Some("botmaker:7".to_owned()),
                answer_source: Some("grok:3".to_owned()),
                fields: vec!["safety_label_source"],
            }]
        );
    }

    #[test]
    fn a_tool_source_renders_without_its_actor_ldap() {
        let entry = found([spam_from(
            vf_pb::safety_label::SafetyLabelSource::ToolAction(vf_pb::ToolAction {
                agent_tool: 2,
                actor_ldap: "synthetic_actor".to_owned(),
            }),
        )]);

        let diff = LabelDiff::new(&entry, &label_map([]));

        assert_eq!(diff.only_entry[0].entry_source.as_deref(), Some("tool:2"));
        assert_eq!(diff.only_entry[0].answer_source, None);
        assert!(!format!("{diff:?}").contains("synthetic_actor"));
    }

    #[test]
    fn a_label_rewritten_with_its_created_at_kept_is_other_and_names_the_field() {
        let created = Some(BASE_MS - 10);
        let entry = found([label_created_at(SafetyLabelType::SPAM, 0.5, created)]);
        let answer = label_map([label_created_at(SafetyLabelType::SPAM, 0.9, created)]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(diff.class, MismatchClass::Other);
        assert_eq!(diff.changed[0].fields, vec!["score"]);
    }

    #[test]
    fn a_not_found_entry_lists_every_answer_label_and_one_without_a_timestamp_is_not_stale() {
        let entry = Envelope {
            fill: fill(BASE_MS, 0),
            status: Status::NotFound,
        };
        let answer = label_map([
            label_created_at(SafetyLabelType::SPAM, 0.5, None),
            label_created_at(SafetyLabelType::ABUSIVE, 1.0, Some(BASE_MS - 10)),
        ]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(
            diff,
            LabelDiff {
                only_entry: vec![],
                only_answer: vec![
                    DifferingLabel {
                        label_type: i32::from(SafetyLabelType::SPAM),
                        entry_created_at_ms: None,
                        answer_created_at_ms: None,
                        entry_expires_at_ms: None,
                        answer_expires_at_ms: None,
                        entry_source: None,
                        answer_source: None,
                        fields: vec![],
                    },
                    DifferingLabel {
                        label_type: i32::from(SafetyLabelType::ABUSIVE),
                        entry_created_at_ms: None,
                        answer_created_at_ms: Some((BASE_MS - 10).cast_signed()),
                        entry_expires_at_ms: None,
                        answer_expires_at_ms: None,
                        entry_source: None,
                        answer_source: None,
                        fields: vec![],
                    },
                ],
                changed: vec![],
                class: MismatchClass::Other,
            }
        );
    }

    #[test]
    fn labels_older_than_the_fill_on_the_answer_side_are_other() {
        let entry = found([label_created_at(
            SafetyLabelType::SPAM,
            0.5,
            Some(BASE_MS - 20),
        )]);
        let answer = label_map([
            label_created_at(SafetyLabelType::SPAM, 0.9, Some(BASE_MS - 10)),
            label_created_at(SafetyLabelType::ABUSIVE, 1.0, Some(BASE_MS)),
        ]);

        let diff = LabelDiff::new(&entry, &answer);

        assert_eq!(diff.class, MismatchClass::Other);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.only_answer.len(), 1);
    }

    #[test]
    fn manhattan_classes_each_three_way_comparison_and_a_failed_read() {
        let entry = found([label_created_at(SafetyLabelType::SPAM, 0.5, None)]);
        let same = label_map([label_created_at(SafetyLabelType::SPAM, 0.5, None)]);
        let other = label_map([label_created_at(SafetyLabelType::SPAM, 0.9, None)]);
        let third = label_map([label_created_at(SafetyLabelType::ABUSIVE, 1.0, None)]);
        let class = |answer, manhattan| mh_check_class(&entry, answer, manhattan);

        assert_eq!(class(&same, Some(&same)), MhCheckClass::AllEqual);
        assert_eq!(class(&other, Some(&same)), MhCheckClass::TwemcacheDiffers);
        assert_eq!(class(&other, Some(&other)), MhCheckClass::DragonflyDiffers);
        assert_eq!(class(&other, Some(&third)), MhCheckClass::AllDiffer);
        assert_eq!(class(&same, Some(&other)), MhCheckClass::ManhattanDiffers);
        assert_eq!(class(&same, None), MhCheckClass::ManhattanFailed);

        let not_found = Envelope {
            fill: fill(BASE_MS, 0),
            status: Status::NotFound,
        };
        assert_eq!(
            mh_check_class(&not_found, &empty(), Some(&empty())),
            MhCheckClass::AllEqual
        );
    }

    fn shadow_hit(tweet_id: u64, sample: MhCheckSample) -> ShadowHit {
        ShadowHit {
            tweet_id,
            sample,
            entry: found([]),
            answer: empty(),
        }
    }

    fn budget(keys: usize) -> impl FnMut() -> bool {
        let mut left = keys;
        move || {
            let admitted = left > 0;
            left = left.saturating_sub(1);
            admitted
        }
    }

    fn checked(hits: Vec<ShadowHit>, keys: usize) -> Vec<(u64, MhCheckSample)> {
        sample_mh_checks(hits, budget(keys))
            .into_iter()
            .map(|hit| (hit.tweet_id, hit.sample))
            .collect()
    }

    #[test]
    fn a_match_check_slice_is_xxhash64_seed_1_of_the_ids_little_endian_bytes_mod_20() {
        let in_slice: Vec<u64> = (0..100).filter(|&id| in_match_check_slice(id)).collect();
        assert_eq!(in_slice, [30, 45, 98]);
    }

    #[test]
    fn manhattan_checks_spend_the_budget_on_mismatches_then_on_matches_in_the_slice() {
        use MhCheckSample::{Match, Mismatch};
        let hits = || {
            vec![
                shadow_hit(30, Match),
                shadow_hit(10, Match),
                shadow_hit(1, Mismatch),
                shadow_hit(45, Match),
                shadow_hit(2, Mismatch),
            ]
        };

        assert_eq!(checked(hits(), 0), []);
        assert_eq!(checked(hits(), 2), [(1, Mismatch), (2, Mismatch)]);
        assert_eq!(
            checked(hits(), 3),
            [(1, Mismatch), (2, Mismatch), (30, Match)]
        );
        assert_eq!(
            checked(hits(), 10),
            [(1, Mismatch), (2, Mismatch), (30, Match), (45, Match)]
        );
    }

    #[test]
    fn a_ramp_bucket_is_xxhash64_of_the_ids_little_endian_bytes_mod_10000() {
        assert_eq!(
            [0, 1, 42, 99_999_999_999_999].map(ramp_bucket),
            [1579, 1269, 5587, 9345]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn with_both_switches_at_zero_the_label_cache_is_neither_read_nor_written() {
        let w = world(LabelCacheSwitches::default());
        w.twemcache.found.lock().unwrap().insert(1);
        w.manhattan.set_spam(2, 0.5);

        let answers = w.source.get(&[1, 2, 3]).await;
        settle().await;

        assert_eq!(labels(&answers, 1), nsfa());
        assert_eq!(labels(&answers, 2), spam(0.5));
        assert_eq!(labels(&answers, 3), empty());
        assert_eq!(w.manhattan.calls().concat().len(), 2);
        assert_eq!(w.dragonfly.keys_read.load(Ordering::SeqCst), 0);
        assert!(w.dragonfly.entries.lock().unwrap().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn a_served_miss_reads_manhattan_and_fills_the_entry_for_a_day() {
        let w = world(serving_all());
        w.manhattan.set_spam(1, 0.5);

        let first = w.source.get(&[1, 2]).await;
        settle().await;

        assert_eq!(labels(&first, 1), spam(0.5));
        assert_eq!(labels(&first, 2), empty());
        assert_eq!(w.manhattan.calls().concat().len(), 2);
        assert_eq!(
            w.dragonfly.entry(1),
            Some((
                Envelope {
                    fill: fill(BASE_MS, 0),
                    status: Status::Found(spam(0.5)),
                },
                DAY_SECS
            ))
        );
        assert_eq!(
            w.dragonfly.entry(2),
            Some((
                Envelope {
                    fill: fill(BASE_MS, 0),
                    status: Status::NotFound,
                },
                DAY_SECS
            ))
        );

        let second = w.source.get(&[1, 2]).await;
        assert_eq!(labels(&second, 1), spam(0.5));
        assert_eq!(labels(&second, 2), empty());
        assert_eq!(
            w.manhattan.calls().len(),
            1,
            "a filled entry skips Manhattan"
        );
        assert_eq!(w.twemcache.keys_read.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn an_entry_in_an_unknown_version_reads_as_a_miss_and_is_overwritten() {
        let w = world(serving_all());
        let mut newer = stored(&Envelope {
            fill: fill(BASE_MS, 2),
            status: Status::Found(spam(0.1)),
        });
        newer[0] = VALUE_VERSION + 1;
        w.dragonfly.put_bytes(1, newer);
        w.manhattan.set_spam(1, 0.5);

        let answers = w.source.get(&[1]).await;
        settle().await;

        assert_eq!(labels(&answers, 1), spam(0.5));
        assert_eq!(w.manhattan.calls(), vec![vec![1]]);
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0.status,
            Status::Found(spam(0.5))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn entries_expire_at_read_from_their_fill_time_under_the_ttl_switches() {
        let w = world(serving_all());
        let ten_minutes_ago = BASE_MS - 600_000;
        w.dragonfly
            .put(1, Status::Found(spam(0.5)), fill(ten_minutes_ago, 2));
        w.dragonfly
            .put(2, Status::NotFound, fill(ten_minutes_ago, 2));
        w.dragonfly
            .put(3, Status::NotFound, fill(ten_minutes_ago + 1, 2));
        w.manhattan.set_spam(1, 0.9);
        w.manhattan.set_spam(2, 0.7);

        let answers = w.source.get(&[1, 2, 3]).await;
        assert_eq!(labels(&answers, 1), spam(0.5));
        assert_eq!(labels(&answers, 2), spam(0.7));
        assert_eq!(labels(&answers, 3), empty());
        assert_eq!(w.manhattan.calls(), vec![vec![2]]);

        w.source.set_label_cache_switches(LabelCacheSwitches {
            found_ttl: Duration::from_secs(300),
            ..serving_all()
        });
        let answers = w.source.get(&[1]).await;
        assert_eq!(labels(&answers, 1), spam(0.9));
    }

    #[tokio::test(start_paused = true)]
    async fn a_ttl_switch_below_a_minute_reads_as_a_minute() {
        let w = world(LabelCacheSwitches {
            found_ttl: Duration::ZERO,
            not_found_ttl: Duration::ZERO,
            ..serving_all()
        });
        w.dragonfly
            .put(1, Status::Found(spam(0.5)), fill(BASE_MS - 59_999, 2));
        w.dragonfly
            .put(2, Status::NotFound, fill(BASE_MS - 59_999, 2));
        w.dragonfly
            .put(3, Status::NotFound, fill(BASE_MS - 60_000, 2));
        w.manhattan.set_spam(1, 0.9);
        w.manhattan.set_spam(3, 0.7);

        let answers = w.source.get(&[1, 2, 3]).await;

        assert_eq!(labels(&answers, 1), spam(0.5));
        assert_eq!(labels(&answers, 2), empty());
        assert_eq!(labels(&answers, 3), spam(0.7));
    }

    #[tokio::test(start_paused = true)]
    async fn a_filled_entry_is_reread_at_30s_and_90s_then_never() {
        let w = world(serving_all());
        let start = tokio::time::Instant::now();
        let at = |secs| tokio::time::sleep_until(start + Duration::from_secs(secs));
        w.manhattan.set_spam(1, 0.5);
        w.source.get(&[1]).await;
        settle().await;
        w.manhattan.set_spam(1, 0.9);

        at(30).await;
        let first = w.source.get(&[1]).await;
        let second = w.source.get(&[1]).await;
        assert_eq!(labels(&first, 1), spam(0.5));
        assert_eq!(labels(&second, 1), spam(0.5));
        settle().await;
        assert_eq!(
            w.manhattan.calls(),
            vec![vec![1], vec![1]],
            "one reread for two due reads"
        );
        assert_eq!(w.dragonfly.entry(1).unwrap().0.fill, fill(BASE_MS, 1));
        assert_eq!(labels(&w.source.get(&[1]).await, 1), spam(0.9));

        at(89).await;
        w.source.get(&[1]).await;
        settle().await;
        assert_eq!(
            w.manhattan.calls().len(),
            2,
            "step 1 waits for 90 s after the fill"
        );

        at(90).await;
        w.source.get(&[1]).await;
        settle().await;
        assert_eq!(w.manhattan.calls().len(), 3);
        assert_eq!(w.dragonfly.entry(1).unwrap().0.fill, fill(BASE_MS, 2));

        at(3600).await;
        w.source.get(&[1]).await;
        settle().await;
        assert_eq!(w.manhattan.calls().len(), 3, "step 2 is never reread");
    }

    #[tokio::test(start_paused = true)]
    async fn fills_and_rereads_drop_the_label_types_twemcache_does_not_decode() {
        let w = world(serving_all());
        let abusive = i32::from(SafetyLabelType::ABUSIVE);
        w.manhattan.set_spam(1, 0.5);
        w.manhattan.add_label(1, SafetyLabelType::ABUSIVE);

        let answers = w.source.get(&[1]).await;
        settle().await;

        assert!(
            labels(&answers, 1).labels.contains_key(&abusive),
            "a miss answers with Manhattan's labels, as a twemcache miss does"
        );
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0.status,
            Status::Found(spam(0.5))
        );

        w.dragonfly
            .put(1, Status::Found(spam(0.1)), fill(BASE_MS - 30_000, 0));
        w.source.get(&[1]).await;
        settle().await;

        assert_eq!(w.manhattan.calls().len(), 2);
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0,
            Envelope {
                fill: fill(BASE_MS - 30_000, 1),
                status: Status::Found(spam(0.5)),
            }
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_entry_first_read_after_90s_is_reread_once_to_the_last_step() {
        let w = world(LabelCacheSwitches {
            serve_buckets: RAMP_BUCKETS,
            ..LabelCacheSwitches::default()
        });
        w.dragonfly
            .put(1, Status::Found(spam(0.5)), fill(BASE_MS - 100_000, 0));
        w.manhattan.set_spam(1, 0.9);

        w.source.get(&[1]).await;
        settle().await;
        w.source.get(&[1]).await;
        settle().await;

        assert_eq!(w.manhattan.calls(), vec![vec![1]]);
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0,
            Envelope {
                fill: fill(BASE_MS - 100_000, 2),
                status: Status::Found(spam(0.9)),
            }
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_entry_filled_before_the_refresh_cutoff_is_served_then_refilled() {
        let w = world(LabelCacheSwitches {
            refresh_cutoff_ms: BASE_MS - 1_000,
            ..serving_all()
        });
        w.dragonfly
            .put(1, Status::Found(spam(0.5)), fill(BASE_MS - 1_001, 2));
        w.dragonfly
            .put(2, Status::Found(spam(0.5)), fill(BASE_MS - 1_000, 2));
        w.manhattan.set_spam(1, 0.9);
        w.manhattan.set_spam(2, 0.9);

        let answers = w.source.get(&[1, 2]).await;
        settle().await;

        assert_eq!(labels(&answers, 1), spam(0.5));
        assert_eq!(labels(&answers, 2), spam(0.5));
        assert_eq!(w.manhattan.calls(), vec![vec![1]]);
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0,
            Envelope {
                fill: fill(BASE_MS, 0),
                status: Status::Found(spam(0.9)),
            }
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_refresh_cutoff_later_than_the_read_is_ignored() {
        let w = world(LabelCacheSwitches {
            serve_buckets: RAMP_BUCKETS,
            refresh_cutoff_ms: BASE_MS + 60_000,
            ..LabelCacheSwitches::default()
        });
        w.dragonfly
            .put(1, Status::Found(spam(0.5)), fill(BASE_MS - 1_000, 2));
        w.manhattan.set_spam(1, 0.9);

        let answers = w.source.get(&[1]).await;
        settle().await;

        assert_eq!(labels(&answers, 1), spam(0.5));
        assert!(w.manhattan.calls().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn a_full_fill_queue_drops_the_fill_but_not_the_answer() {
        let Built {
            world: w,
            label_cache,
            fill_rx,
        } = build(serving_all(), None, Some(1));
        let mut fill_rx = fill_rx.unwrap();
        w.manhattan.set_spam(1, 0.5);
        w.manhattan.set_spam(2, 0.6);

        let first = w.source.get(&[1]).await;
        let second = w.source.get(&[2]).await;

        assert_eq!(labels(&first, 1), spam(0.5));
        assert_eq!(labels(&second, 2), spam(0.6));
        assert!(matches!(fill_rx.try_recv(), Ok(job) if job.tweet_id == 1));
        assert!(fill_rx.try_recv().is_err());
        let fills = |result| {
            w.stats
                .count("safety_labels_cache_fill_keys", &[("result", result)])
        };
        assert_eq!(fills("dropped_queue_full"), 1);
        assert!(
            !label_cache.pending_fills.lock().unwrap().contains(&2),
            "a dropped fill can be queued again"
        );

        w.source.get(&[2]).await;
        assert!(matches!(fill_rx.try_recv(), Ok(job) if job.tweet_id == 2));
        assert_eq!(fills("enqueued"), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn shadow_answers_as_before_and_fills_no_miss_without_a_fill_budget() {
        let w = world(LabelCacheSwitches {
            shadow_buckets: RAMP_BUCKETS,
            ..LabelCacheSwitches::default()
        });
        w.twemcache.found.lock().unwrap().insert(1);
        w.manhattan.set_spam(2, 0.5);

        let answers = w.source.get(&[1, 2]).await;
        settle().await;

        assert_eq!(labels(&answers, 1), nsfa());
        assert_eq!(labels(&answers, 2), spam(0.5));
        assert_eq!(w.manhattan.calls(), vec![vec![2]]);
        assert!(w.dragonfly.entry(1).is_none());
        assert!(w.dragonfly.entry(2).is_none());
        assert_eq!(w.stats.count("safety_labels_shadow_manhattan_keys", &[]), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn shadow_fills_misses_from_manhattan_within_its_budget_and_counts_mismatches() {
        let w = world(LabelCacheSwitches {
            shadow_buckets: RAMP_BUCKETS,
            shadow_fill_keys_per_sec: 1,
            ..LabelCacheSwitches::default()
        });
        w.twemcache.found.lock().unwrap().insert(1);
        w.manhattan.set_spam(1, 0.9);
        w.manhattan.set_spam(2, 0.5);
        let shadow = |result| {
            w.stats
                .count("safety_labels_shadow_keys", &[("result", result)])
        };

        let answers = w.source.get(&[1, 2]).await;
        settle().await;

        assert_eq!(labels(&answers, 1), nsfa());
        assert_eq!(labels(&answers, 2), spam(0.5));
        assert_eq!(w.manhattan.calls(), vec![vec![2], vec![1]]);
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0,
            Envelope {
                fill: fill(BASE_MS, 0),
                status: Status::Found(spam(0.9)),
            }
        );
        assert!(w.dragonfly.entry(2).is_none(), "past the budget");
        assert_eq!(w.stats.count("safety_labels_shadow_manhattan_keys", &[]), 2);

        w.dragonfly
            .put(2, Status::Found(spam(0.5)), fill(BASE_MS, 2));
        w.source.get(&[1, 2]).await;
        settle().await;
        assert_eq!(shadow("mismatch"), 1);
        assert_eq!(shadow("match"), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_fill_stores_manhattans_items_as_scanned_and_reads_back_twemcache_types_only() {
        let w = world(serving_all());
        w.manhattan.set_spam(1, 0.5);
        w.manhattan.add_label(1, SafetyLabelType::ABUSIVE);

        w.source.get(&[1]).await;
        settle().await;

        assert_eq!(
            w.dragonfly.stored_items(1),
            vec![
                StoredItem {
                    lkey: encode_lkey(SafetyLabelType::SPAM).to_vec(),
                    mval: mval_with_score(0.5),
                },
                StoredItem {
                    lkey: encode_lkey(SafetyLabelType::ABUSIVE).to_vec(),
                    mval: minimal_mval(),
                },
            ]
        );
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0.status,
            Status::Found(spam(0.5))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn shadow_refreshes_and_refresh_cutoff_refills_reread_manhattan() {
        let w = world(LabelCacheSwitches {
            shadow_buckets: RAMP_BUCKETS,
            refresh_cutoff_ms: BASE_MS - 40_000,
            ..LabelCacheSwitches::default()
        });
        w.twemcache.found.lock().unwrap().extend([1, 2]);
        w.dragonfly
            .put(1, Status::Found(nsfa()), fill(BASE_MS - 30_000, 0));
        w.dragonfly
            .put(2, Status::Found(nsfa()), fill(BASE_MS - 40_001, 2));
        w.manhattan.set_spam(1, 0.9);
        w.manhattan.set_spam(2, 0.8);

        let answers = w.source.get(&[1, 2]).await;
        settle().await;

        assert_eq!(labels(&answers, 1), nsfa());
        assert_eq!(w.manhattan.calls(), vec![vec![1, 2]]);
        assert_eq!(
            w.dragonfly.entry(1).unwrap().0,
            Envelope {
                fill: fill(BASE_MS - 30_000, 1),
                status: Status::Found(spam(0.9)),
            }
        );
        assert_eq!(
            w.dragonfly.entry(2).unwrap().0,
            Envelope {
                fill: fill(BASE_MS, 0),
                status: Status::Found(spam(0.8)),
            }
        );
        assert_eq!(w.stats.count("safety_labels_shadow_manhattan_keys", &[]), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn a_mismatch_the_manhattan_check_confirms_is_overwritten_with_manhattans_answer() {
        for (mh_check_keys_per_sec, repaired) in [(0, false), (10, true)] {
            let w = world(LabelCacheSwitches {
                shadow_buckets: RAMP_BUCKETS,
                shadow_mh_check_keys_per_sec: mh_check_keys_per_sec,
                ..LabelCacheSwitches::default()
            });
            w.twemcache.found.lock().unwrap().insert(1);
            let stale = Envelope {
                fill: fill(BASE_MS - 5_000, 2),
                status: Status::NotFound,
            };
            w.dragonfly.put(1, stale.status.clone(), stale.fill);
            w.manhattan.set_spam(1, 0.9);

            w.source.get(&[1]).await;
            settle().await;

            let repair = Envelope {
                fill: fill(BASE_MS, 0),
                status: Status::Found(spam(0.9)),
            };
            let expected = if repaired { repair } else { stale };
            assert_eq!(w.dragonfly.entry(1).unwrap().0, expected);
            assert_eq!(
                w.manhattan.calls().len(),
                usize::from(repaired),
                "only the check reads Manhattan"
            );
            assert_eq!(
                w.stats.count("safety_labels_shadow_manhattan_keys", &[]),
                0,
                "serving would not have read Manhattan for it"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn with_a_manhattan_check_budget_shadow_rereads_a_mismatch_from_manhattan_and_classes_it()
    {
        let w = world(LabelCacheSwitches {
            shadow_buckets: RAMP_BUCKETS,
            shadow_mh_check_keys_per_sec: 10,
            ..LabelCacheSwitches::default()
        });
        w.manhattan.set_spam(2, 0.5);
        w.dragonfly.put(2, Status::NotFound, fill(BASE_MS, 2));

        let answers = w.source.get(&[2]).await;
        settle().await;

        assert_eq!(labels(&answers, 2), spam(0.5));
        assert_eq!(
            w.manhattan.calls(),
            vec![vec![2]; 2],
            "the answer and the check"
        );
        for view in ["filtered", "raw"] {
            assert_eq!(
                w.stats.count(
                    "safety_labels_shadow_mh_check_keys",
                    &[
                        ("class", "dragonfly_differs"),
                        ("sample", "mismatch"),
                        ("view", view)
                    ]
                ),
                1
            );
        }
    }

    fn nsfa_and_spam_expiring_at(expires_at_ms: u64) -> Status {
        let mut labels = nsfa();
        labels.labels.extend([spam_expiring_at(0.5, expires_at_ms)]);
        Status::Found(labels)
    }

    #[tokio::test(start_paused = true)]
    async fn shadow_drops_labels_expired_at_compare_time_so_only_a_live_difference_mismatches() {
        let w = world(LabelCacheSwitches {
            shadow_buckets: RAMP_BUCKETS,
            ..LabelCacheSwitches::default()
        });
        w.twemcache.found.lock().unwrap().extend([1, 2]);
        w.dragonfly
            .put(1, nsfa_and_spam_expiring_at(BASE_MS - 1), fill(BASE_MS, 2));
        w.dragonfly.put(
            2,
            nsfa_and_spam_expiring_at(BASE_MS + 60_000),
            fill(BASE_MS, 2),
        );
        let shadow = |result| {
            w.stats
                .count("safety_labels_shadow_keys", &[("result", result)])
        };

        w.source.get(&[1, 2]).await;
        settle().await;

        assert_eq!(shadow("match"), 1);
        assert_eq!(shadow("mismatch"), 1);
    }

        fn with_decoder_gaps(mut labels: vf_pb::SafetyLabelMap) -> vf_pb::SafetyLabelMap {
        labels.labels.insert(
            i32::from(SafetyLabelType::ABUSIVE),
            vf_pb::SafetyLabel::default(),
        );
        labels
    }

    #[tokio::test(start_paused = true)]
    async fn shadow_compares_without_twemcaches_missing_types_but_with_every_field() {
        let (in_match_slice, differs, source_differs): (u64, u64, u64) = (30, 2, 3);
        let w = world(LabelCacheSwitches {
            shadow_buckets: RAMP_BUCKETS,
            shadow_mh_check_keys_per_sec: 10,
            ..LabelCacheSwitches::default()
        });
        for id in [in_match_slice, differs, source_differs] {
            w.manhattan.set_spam(id.cast_signed(), 0.5);
        }
        let mut with_source = spam(0.5);
        for label in with_source.labels.values_mut() {
            label.safety_label_source =
                Some(vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                    vf_pb::BotmakerAction { rule_id: 7 },
                ));
        }
        w.dragonfly
            .put(source_differs, Status::Found(with_source), fill(BASE_MS, 2));
        w.dragonfly.put(
            in_match_slice,
            Status::Found(with_decoder_gaps(spam(0.5))),
            fill(BASE_MS, 2),
        );
        w.dragonfly.put(
            differs,
            Status::Found(with_decoder_gaps(spam(0.9))),
            fill(BASE_MS, 2),
        );
        let shadow = |result| {
            w.stats
                .count("safety_labels_shadow_keys", &[("result", result)])
        };
        let mh_check = |class, sample, view| {
            w.stats.count(
                "safety_labels_shadow_mh_check_keys",
                &[("class", class), ("sample", sample), ("view", view)],
            )
        };

        w.source
            .get(&[in_match_slice, differs, source_differs])
            .await;
        settle().await;

        assert_eq!(shadow("match"), 1);
        assert_eq!(shadow("mismatch"), 2, "a differing label source mismatches");
        for view in ["filtered", "raw"] {
            assert_eq!(
                mh_check("all_equal", "match", view),
                1,
                "a read drops the types twemcache skips"
            );
        }
        assert_eq!(mh_check("dragonfly_differs", "mismatch", "filtered"), 2);
        assert_eq!(mh_check("dragonfly_differs", "mismatch", "raw"), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn a_manhattan_check_classes_each_hit_with_expired_labels_dropped_and_raw() {
        let in_match_slice: u64 = 30;
        let w = world(LabelCacheSwitches {
            shadow_buckets: RAMP_BUCKETS,
            shadow_mh_check_keys_per_sec: 10,
            ..LabelCacheSwitches::default()
        });
        let mut entry = spam(0.5);
        entry.labels.insert(
            i32::from(SafetyLabelType::NSFA_HIGH_PRECISION),
            vf_pb::SafetyLabel {
                expires_at_msec: Some((BASE_MS - 1).cast_signed()),
                ..vf_pb::SafetyLabel::default()
            },
        );
        w.dragonfly
            .put(in_match_slice, Status::Found(entry), fill(BASE_MS, 2));
        w.manhattan.set_spam(in_match_slice.cast_signed(), 0.5);
        let mh_check = |class, view| {
            w.stats.count(
                "safety_labels_shadow_mh_check_keys",
                &[("class", class), ("sample", "match"), ("view", view)],
            )
        };

        w.source.get(&[in_match_slice]).await;
        settle().await;

        assert_eq!(mh_check("all_equal", "filtered"), 1);
        assert_eq!(mh_check("dragonfly_differs", "raw"), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn shadow_counts_the_keys_it_could_not_read_as_would_be_manhattan_reads() {
        let Built {
            world: w,
            label_cache,
            ..
        } = build(
            LabelCacheSwitches {
                shadow_buckets: RAMP_BUCKETS,
                ..LabelCacheSwitches::default()
            },
            None,
            None,
        );
        let projected = || w.stats.count("safety_labels_shadow_manhattan_keys", &[]);
        w.dragonfly.failing.store(true, Ordering::SeqCst);

        w.source.get(&[1, 2]).await;
        settle().await;
        assert_eq!(projected(), 2, "failed reads");

        let _permits = Arc::clone(&label_cache.shadow_tasks)
            .try_acquire_many_owned(SHADOW_TASKS as u32)
            .unwrap();
        w.source.get(&[3]).await;
        assert_eq!(
            w.stats
                .count("safety_labels_shadow_keys", &[("result", "shed")]),
            1
        );
        assert_eq!(projected(), 3, "shed reads");
    }

    #[tokio::test(start_paused = true)]
    async fn when_the_label_cache_fails_stale_hot_keys_then_budgeted_manhattan_then_failed_ids() {
        let (clock, mock) = quanta::Clock::mock();
        let w = build(serving_all(), Some(clock), None).world;
        w.manhattan.set_spam(1, 0.5);
        w.source.get(&[1]).await;
        mock.increment(Duration::from_secs(10));
        w.manhattan.set_spam(1, 0.9);
        w.dragonfly.failing.store(true, Ordering::SeqCst);
        let uncached: Vec<u64> = (2..3_002).collect();

        let answers = w.source.get(&[vec![1], uncached.clone()].concat()).await;

        assert_eq!(labels(&answers, 1), spam(0.5), "the expired hot-key entry");
        let admitted = uncached.iter().filter(|id| answers[id].is_ok()).count();
        assert!((2_500..3_000).contains(&admitted), "{admitted}");
        assert!(uncached.iter().all(|id| match &answers[id] {
            Ok(_) => true,
            Err(failure) => failure.kind == FailureKind::LabelCacheUnavailable,
        }));
        let outage = |result| {
            w.stats
                .count("safety_labels_outage_keys", &[("result", result)])
        };
        assert_eq!(outage("stale"), 1);
        assert_eq!(outage("manhattan"), admitted as u64);
        assert_eq!(outage("failed"), (3_000 - admitted) as u64);
    }

    #[tokio::test(start_paused = true)]
    async fn an_outage_reads_manhattan_in_the_batchs_one_call_and_keeps_stale_hot_keys() {
        let (clock, mock) = quanta::Clock::mock();
        let w = build(serving_all(), Some(clock), None).world;
        for id in 1..=3 {
            w.manhattan.set_spam(id, 0.5);
        }
        w.source.get(&[1]).await;
        settle().await;
        mock.increment(Duration::from_secs(10));
        w.manhattan.set_spam(1, 0.9);
        w.dragonfly.fail(1);
        w.dragonfly.fail(2);

        let answers = w.source.get(&[1, 2, 3]).await;

        assert_eq!(labels(&answers, 1), spam(0.5), "the expired hot-key entry");
        assert_eq!(labels(&answers, 2), spam(0.5));
        assert_eq!(labels(&answers, 3), spam(0.5));
        let mut calls = w.manhattan.calls();
        assert_eq!(calls.len(), 2, "{calls:?}");
        calls[1].sort_unstable();
        assert_eq!(calls[1], vec![2, 3]);
        let outage = |result| {
            w.stats
                .count("safety_labels_outage_keys", &[("result", result)])
        };
        assert_eq!(
            (outage("stale"), outage("manhattan"), outage("failed")),
            (1, 1, 0)
        );
    }
}
