use std::collections::HashMap;
use std::fmt::Display;
use std::future::Future;
use std::hash::Hash;
use std::time::{Duration, Instant};

use tracing::debug;
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

use crate::hydration::batch::{Hydrated, HydrationBatch, HydrationError};
use crate::rules::SafetyLevel;

const HYDRATOR_REQUESTS: &str = "vf_hydrator_requests";
const HYDRATOR_LATENCY_MS: &str = "vf_hydrator_latency_ms";
const HYDRATOR_KEYS: &str = "vf_hydrator_keys";
const HYDRATOR_TWEET_IDS: &str = "vf_hydrator_tweet_ids";
const HYDRATOR_BATCH_SIZE: &str = "vf_hydrator_batch_size";
const FALLBACK_CACHE_KEYS: &str = "vf_fallback_cache_keys";
const FALLBACK_CACHE_ENTRIES: &str = "vf_fallback_cache_entries";
const AUTHOR_LABELS: &str = "vf_author_labels";

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum HydratorOutcome {
    Success,
    Partial,
    Timeout,
    Error,
}

pub(crate) fn batch_outcome<K, V, E>(map: &HashMap<K, Result<V, E>>) -> HydratorOutcome {
    if !map.is_empty() && map.values().all(|result| result.is_err()) {
        HydratorOutcome::Error
    } else {
        HydratorOutcome::Success
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KeyedResultCounts {
    success_keys: usize,
    timeout_keys: usize,
    error_keys: usize,
    success_candidates: usize,
    timeout_candidates: usize,
    error_candidates: usize,
}

impl KeyedResultCounts {
    fn from_batch<K, V>(
        candidate_count_by_key: &HashMap<K, usize>,
        batch: &HydrationBatch<K, V>,
    ) -> Self
    where
        K: Eq + Hash,
    {
        let mut counts = Self::default();
        for (key, candidate_count) in candidate_count_by_key {
            match batch.hydrated(key) {
                Some(Hydrated::Found(_) | Hydrated::NotFound) => {
                    counts.success_keys += 1;
                    counts.success_candidates += candidate_count;
                }
                Some(Hydrated::Failed(HydrationError::Timeout)) => {
                    counts.timeout_keys += 1;
                    counts.timeout_candidates += candidate_count;
                }
                _ => {
                    counts.error_keys += 1;
                    counts.error_candidates += candidate_count;
                }
            }
        }
        counts
    }

    fn outcome(self) -> HydratorOutcome {
        let failures = self.timeout_keys + self.error_keys;
        match (self.success_keys, failures) {
            (_, 0) => HydratorOutcome::Success,
            (0, _) if self.timeout_keys > 0 && self.error_keys == 0 => HydratorOutcome::Timeout,
            (0, _) => HydratorOutcome::Error,
            _ => HydratorOutcome::Partial,
        }
    }
}

pub(crate) fn record_hydrator_request(
    client: &'static str,
    method: &'static str,
    safety_level: SafetyLevel,
    outcome: HydratorOutcome,
    candidate_count: usize,
    latency_ms: f64,
) {
    if outcome != HydratorOutcome::Success {
        debug!(
            client,
            method,
            outcome = <&str>::from(outcome),
            candidate_count,
            "Hydrator fail-open"
        );
    }
    incr(
        HYDRATOR_REQUESTS,
        &[
            ("client", client),
            ("method", method),
            ("outcome", outcome.into()),
            ("safety_level", safety_level.into()),
        ],
        1,
    );
    incr_nonzero(
        HYDRATOR_TWEET_IDS,
        &[
            ("client", client),
            ("method", method),
            ("result", outcome.into()),
            ("safety_level", safety_level.into()),
        ],
        candidate_count as u64,
    );
    observe(
        HYDRATOR_LATENCY_MS,
        &[
            ("client", client),
            ("method", method),
            ("safety_level", safety_level.into()),
        ],
        latency_ms,
        HistogramBuckets::Bucket50To500,
    );
}

fn record_keyed_hydrator_request(
    client: &'static str,
    method: &'static str,
    safety_level: SafetyLevel,
    counts: KeyedResultCounts,
    latency_ms: f64,
) {
    let outcome = counts.outcome();
    if outcome != HydratorOutcome::Success {
        debug!(
            client,
            method,
            outcome = <&str>::from(outcome),
            success_keys = counts.success_keys,
            timeout_keys = counts.timeout_keys,
            error_keys = counts.error_keys,
            "Hydrator fail-open"
        );
    }
    incr(
        HYDRATOR_REQUESTS,
        &[
            ("client", client),
            ("method", method),
            ("outcome", outcome.into()),
            ("safety_level", safety_level.into()),
        ],
        1,
    );
    for (result, keys, candidates) in [
        ("success", counts.success_keys, counts.success_candidates),
        ("timeout", counts.timeout_keys, counts.timeout_candidates),
        ("error", counts.error_keys, counts.error_candidates),
    ] {
        let labels = [
            ("client", client),
            ("method", method),
            ("result", result),
            ("safety_level", safety_level.into()),
        ];
        incr_nonzero(HYDRATOR_KEYS, &labels, keys as u64);
        incr_nonzero(HYDRATOR_TWEET_IDS, &labels, candidates as u64);
    }
    observe(
        HYDRATOR_LATENCY_MS,
        &[
            ("client", client),
            ("method", method),
            ("safety_level", safety_level.into()),
        ],
        latency_ms,
        HistogramBuckets::Bucket50To500,
    );
}

pub(crate) fn record_author_labels(mapped: usize, unmapped: usize) {
    for (result, count) in [("mapped", mapped), ("unmapped", unmapped)] {
        incr_nonzero(AUTHOR_LABELS, &[("result", result)], count as u64);
    }
}

pub(crate) fn record_batch_size(client: &'static str, candidate_count: usize) {
    observe(
        HYDRATOR_BATCH_SIZE,
        &[("client", client)],
        candidate_count as f64,
        HistogramBuckets::Bucket50To500,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_fallback_cache_keys(
    facet: &'static str,
    fresh: usize,
    stale: usize,
    stale_not_found: usize,
    not_found: usize,
    unavailable: usize,
) {
    for (result, count) in [
        ("fresh", fresh),
        ("stale", stale),
        ("stale_not_found", stale_not_found),
        ("not_found", not_found),
        ("unavailable", unavailable),
    ] {
        incr_nonzero(
            FALLBACK_CACHE_KEYS,
            &[("facet", facet), ("result", result)],
            count as u64,
        );
    }
}

pub(crate) async fn timed_rpc<T: Default>(
    client: &'static str,
    method: &'static str,
    safety_level: SafetyLevel,
    candidate_count: usize,
    timeout: Duration,
    classify: impl FnOnce(&T) -> HydratorOutcome,
    fut: impl Future<Output = T>,
) -> (HydratorOutcome, T) {
    let start = Instant::now();
    let (outcome, body) = match tokio::time::timeout(timeout, fut).await {
        Ok(body) => {
            let outcome = classify(&body);
            (outcome, body)
        }
        Err(_) => (HydratorOutcome::Timeout, T::default()),
    };
    record_hydrator_request(
        client,
        method,
        safety_level,
        outcome,
        candidate_count,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    (outcome, body)
}

pub(crate) async fn timed_results<K, V, E>(
    client: &'static str,
    method: &'static str,
    safety_level: SafetyLevel,
    candidate_count_by_key: &HashMap<K, usize>,
    timeout: Duration,
    fut: impl Future<Output = HashMap<K, Result<Option<V>, E>>>,
) -> HydrationBatch<K, V>
where
    K: Copy + Eq + Hash,
    E: Display,
{
    timed_batch(
        client,
        method,
        safety_level,
        candidate_count_by_key,
        timeout,
        fut,
        |body| HydrationBatch::from_results(candidate_count_by_key.keys().copied(), body),
    )
    .await
}

pub(crate) async fn timed_values<K, V>(
    client: &'static str,
    method: &'static str,
    safety_level: SafetyLevel,
    candidate_count_by_key: &HashMap<K, usize>,
    timeout: Duration,
    fut: impl Future<Output = HashMap<K, V>>,
) -> HydrationBatch<K, V>
where
    K: Copy + Eq + Hash,
{
    timed_batch(
        client,
        method,
        safety_level,
        candidate_count_by_key,
        timeout,
        fut,
        |body| HydrationBatch::from_values(candidate_count_by_key.keys().copied(), body),
    )
    .await
}

async fn timed_batch<K, V, F: Future>(
    client: &'static str,
    method: &'static str,
    safety_level: SafetyLevel,
    candidate_count_by_key: &HashMap<K, usize>,
    timeout: Duration,
    fut: F,
    into_batch: impl FnOnce(F::Output) -> HydrationBatch<K, V>,
) -> HydrationBatch<K, V>
where
    K: Copy + Eq + Hash,
{
    let start = Instant::now();
    let batch = match tokio::time::timeout(timeout, fut).await {
        Ok(body) => into_batch(body),
        Err(_) => HydrationBatch::timed_out(candidate_count_by_key.keys().copied()),
    };
    record_keyed_hydrator_request(
        client,
        method,
        safety_level,
        KeyedResultCounts::from_batch(candidate_count_by_key, &batch),
        start.elapsed().as_secs_f64() * 1000.0,
    );
    batch
}

fn incr(metric: &str, labels: &[(&str, &str)], count: u64) {
    if let Some(sr) = global_stats_receiver() {
        sr.incr(metric, labels, count);
    }
}

fn incr_nonzero(metric: &str, labels: &[(&str, &str)], count: u64) -> bool {
    if count > 0 {
        incr(metric, labels, count);
        true
    } else {
        false
    }
}

pub(crate) fn record_fallback_cache_entries(facet: &'static str, entries: usize) {
    if let Some(sr) = global_stats_receiver() {
        sr.gauge(FALLBACK_CACHE_ENTRIES, &[("facet", facet)], entries as f64);
    }
}

fn observe(metric: &str, labels: &[(&str, &str)], value: f64, buckets: HistogramBuckets) {
    if let Some(sr) = global_stats_receiver() {
        sr.observe(metric, labels, value, buckets);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_generator_pins_the_author_labels_metric_name() {
        let cargo = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/dashboard.py");
        let ws = "crates/x-product/xai-visibility-filtering-service/scripts/dashboard.py";
        let path = if std::path::Path::new(cargo).exists() {
            cargo
        } else {
            ws
        };
        let dashboard =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert!(dashboard.contains(&format!("AUTHOR_LABELS_METRIC = \"{AUTHOR_LABELS}\"")));
    }

    #[test]
    fn empty_batch_is_success() {
        let map: HashMap<u64, Result<u8, ()>> = HashMap::new();
        assert_eq!(batch_outcome(&map), HydratorOutcome::Success);
    }

    #[test]
    fn all_keys_errored_is_error() {
        let map: HashMap<u64, Result<u8, ()>> = HashMap::from([(1, Err(())), (2, Err(()))]);
        assert_eq!(batch_outcome(&map), HydratorOutcome::Error);
    }

    #[test]
    fn any_success_preserves_legacy_batch_success() {
        let map: HashMap<u64, Result<u8, ()>> = HashMap::from([(1, Err(())), (2, Ok(7))]);
        assert_eq!(batch_outcome(&map), HydratorOutcome::Success);
    }

    #[test]
    fn from_batch_counts_not_found_as_success_and_missing_as_error() {
        let expected = HashMap::from([(1, 1), (2, 2), (3, 3), (4, 4)]);
        let batch: HydrationBatch<u64, u8> = HydrationBatch::from_results(
            [1, 2, 3, 4],
            HashMap::from([(1, Ok(Some(7))), (2, Ok(None)), (3, Err("boom"))]),
        );

        let counts = KeyedResultCounts::from_batch(&expected, &batch);

        assert_eq!(counts.success_keys, 2);
        assert_eq!(counts.success_candidates, 3);
        assert_eq!(counts.error_keys, 2);
        assert_eq!(counts.error_candidates, 7);
        assert_eq!(counts.outcome(), HydratorOutcome::Partial);
    }

    #[test]
    fn from_batch_counts_timeout_keys() {
        let expected = HashMap::from([(1, 2), (2, 3)]);
        let batch: HydrationBatch<u64, u8> = HydrationBatch::timed_out([1, 2]);

        let counts = KeyedResultCounts::from_batch(&expected, &batch);

        assert_eq!(counts.timeout_keys, 2);
        assert_eq!(counts.timeout_candidates, 5);
        assert_eq!(counts.outcome(), HydratorOutcome::Timeout);
    }

    #[tokio::test]
    async fn timed_results_outer_timeout_fails_every_expected_key() {
        let expected = HashMap::from([(1, 2), (2, 3)]);
        let never = std::future::pending::<HashMap<u64, Result<Option<u8>, &str>>>();

        let returned = timed_results(
            "test",
            "timeout",
            SafetyLevel::TimelineHome,
            &expected,
            Duration::ZERO,
            never,
        )
        .await;

        assert!(matches!(
            returned.hydrated(&1),
            Some(Hydrated::Failed(HydrationError::Timeout))
        ));
    }

    #[tokio::test]
    async fn timed_values_backfills_absent_expected_keys() {
        let expected = HashMap::from([(1, 1), (2, 1)]);

        let returned = timed_values(
            "test",
            "values",
            SafetyLevel::TimelineHome,
            &expected,
            Duration::from_secs(1),
            std::future::ready(HashMap::from([(1, 7)])),
        )
        .await;

        assert_eq!(returned.get(&1), Some(&7));
        assert!(matches!(
            returned.hydrated(&2),
            Some(Hydrated::Failed(HydrationError::MissingResponse))
        ));
    }
}
