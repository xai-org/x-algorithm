use std::sync::Arc;

use xai_stats_receiver::{HistogramBuckets, StatsReceiverExt};

use super::remote::{FallbackReason, LabelSource};

const SOURCE_REQUESTS: &str = "safety_labels_source_requests";
const SOURCE_LATENCY_MS: &str = "safety_labels_source_latency_ms";
const CACHE_KEYS: &str = "safety_labels_cache_keys";
const MANHATTAN_KEYS: &str = "safety_labels_manhattan_keys";
const CACHE_FALLBACK_KEYS: &str = "safety_labels_cache_fallback_keys";
const BATCH_SIZE: &str = "safety_labels_lookup_batch_size";
const CACHE_WARM_KEYS: &str = "safety_labels_cache_warm_keys";
const EXPIRED_LABELS: &str = "safety_labels_expired_labels";

#[derive(Clone, Default)]
pub(crate) struct LabelMetrics {
    sink: Option<Arc<dyn StatsReceiverExt>>,
}

impl LabelMetrics {
    pub(crate) fn new(sink: Option<Arc<dyn StatsReceiverExt>>) -> Self {
        Self { sink }
    }

    fn incr(&self, metric: &str, labels: &[(&str, &str)], count: usize) {
        if let Some(sink) = &self.sink
            && count > 0
        {
            sink.incr(metric, labels, count as u64);
        }
    }

    fn observe(&self, metric: &str, labels: &[(&str, &str)], value: f64) {
        if let Some(sink) = &self.sink {
            sink.observe(metric, labels, value, HistogramBuckets::Bucket50To500);
        }
    }

    fn observe_vm(&self, metric: &str, labels: &[(&str, &str)], value: f64) {
        if let Some(sink) = &self.sink {
            sink.observe_vm(metric, labels, value);
        }
    }
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum SourceOutcome {
    Success,
    Failure,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum CacheTier {
    Local,
    Twemcache,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum CacheResult {
    Hit,
    NotFound,
    Miss,
    Expired,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum ManhattanResult {
    Success,
    Failed,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum BatchStage {
    LocalMiss,
    ManhattanFallback,
}

pub(crate) fn record_source_request(
    metrics: &LabelMetrics,
    source: LabelSource,
    outcome: SourceOutcome,
    elapsed_ms: f64,
) {
    let source = ("source", source.into());
    metrics.incr(SOURCE_REQUESTS, &[source, ("outcome", outcome.into())], 1);
    metrics.observe_vm(SOURCE_LATENCY_MS, &[source], elapsed_ms);
}

pub(crate) fn record_cache_keys(
    metrics: &LabelMetrics,
    tier: CacheTier,
    result: CacheResult,
    count: usize,
) {
    let labels = [("tier", tier.into()), ("result", result.into())];
    metrics.incr(CACHE_KEYS, &labels, count);
}

pub(crate) fn record_manhattan_keys(metrics: &LabelMetrics, result: ManhattanResult, count: usize) {
    metrics.incr(MANHATTAN_KEYS, &[("result", result.into())], count);
}

pub(crate) fn record_cache_fallback_keys(
    metrics: &LabelMetrics,
    source: LabelSource,
    reason: FallbackReason,
    count: usize,
) {
    let labels = [("source", source.into()), ("reason", reason.into())];
    metrics.incr(CACHE_FALLBACK_KEYS, &labels, count);
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum WarmKeyResult {
    EligibleMiss,
    Enqueued,
    DroppedChannelFull,
    FetchIssued,
    FetchFailed,
}

pub(crate) fn record_cache_warm_keys(metrics: &LabelMetrics, result: WarmKeyResult, count: usize) {
    metrics.incr(CACHE_WARM_KEYS, &[("result", result.into())], count);
}

pub(crate) fn record_expired_labels(metrics: &LabelMetrics, count: usize) {
    metrics.incr(EXPIRED_LABELS, &[], count);
}

pub(crate) fn record_batch_size(metrics: &LabelMetrics, stage: BatchStage, size: usize) {
    metrics.observe(BATCH_SIZE, &[("stage", stage.into())], size as f64);
}
