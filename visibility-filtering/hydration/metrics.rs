use rustc_hash::FxHashMap;
use std::hash::Hash;

use tracing::debug;

use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::sources::Expired;
use crate::hydration::tweet_source::TweetState;
use crate::metrics::families::{
    FALLBACK_CACHE_KEYS, FALLBACK_CACHE_RESIDENT_KEYS, FLOCK_MISSING_KEYS, HYDRATOR_KEYS,
    HYDRATOR_LATENCY_MS, HYDRATOR_REQUESTS, HYDRATOR_TWEET_IDS, REQUEST_COUNTRY, TWEET_STATE,
    VIEWER_COUNTRY,
};
use crate::metrics::{Metrics, RequestMetrics};
use crate::params::CountryList;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum HydratorOutcome {
    Success,
    Partial,
    Timeout,
    Error,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum FallbackCacheResult {
    Fresh,
    Stale,
    NotFound,
    Partial,
    PartialStale,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KeyedResultCounts {
    success_keys: usize,
    partial_keys: usize,
    timeout_keys: usize,
    error_keys: usize,
    success_candidates: usize,
    partial_candidates: usize,
    timeout_candidates: usize,
    error_candidates: usize,
}

impl KeyedResultCounts {
    fn from_batch<K, V>(
        candidate_count_by_claimed_key: &FxHashMap<K, usize>,
        batches: &impl AsRef<[HydrationBatch<K, V>]>,
    ) -> Self
    where
        K: Eq + Hash,
    {
        let mut counts = Self::default();
        for (key, &candidate_count) in candidate_count_by_claimed_key {
            let hydrated = batches
                .as_ref()
                .iter()
                .find_map(|batch| batch.hydrated(key));
            match hydrated {
                Some(Hydrated::Found(_) | Hydrated::NotFound) => {
                    counts.success_keys += 1;
                    counts.success_candidates += candidate_count;
                }
                Some(Hydrated::Partial(_)) => {
                    counts.partial_keys += 1;
                    counts.partial_candidates += candidate_count;
                }
                Some(Hydrated::Failed | Hydrated::Unanswered) | None => {
                    counts.error_keys += 1;
                    counts.error_candidates += candidate_count;
                }
            }
        }
        counts
    }

    fn timed_out<K>(candidate_count_by_claimed_key: &FxHashMap<K, usize>) -> Self {
        Self {
            timeout_keys: candidate_count_by_claimed_key.len(),
            timeout_candidates: candidate_count_by_claimed_key.values().sum(),
            ..Self::default()
        }
    }

    fn outcome(self) -> HydratorOutcome {
        let answered = self.success_keys + self.partial_keys;
        let failures = self.timeout_keys + self.error_keys;
        match (answered, failures) {
            (_, 0) if self.partial_keys == 0 => HydratorOutcome::Success,
            (0, _) if self.error_keys == 0 => HydratorOutcome::Timeout,
            (0, _) => HydratorOutcome::Error,
            _ => HydratorOutcome::Partial,
        }
    }
}

pub(crate) fn record_keyed_hydrator_request<K, V>(
    metrics: &RequestMetrics,
    backend: &str,
    call: &str,
    candidate_count_by_claimed_key: &FxHashMap<K, usize>,
    answer: Result<&impl AsRef<[HydrationBatch<K, V>]>, &Expired>,
    latency_ms: f64,
) where
    K: Eq + Hash,
{
    let counts = match answer {
        Ok(answer) => KeyedResultCounts::from_batch(candidate_count_by_claimed_key, answer),
        Err(Expired) => KeyedResultCounts::timed_out(candidate_count_by_claimed_key),
    };
    let outcome = counts.outcome();
    if outcome != HydratorOutcome::Success {
        debug!(
            backend,
            call,
            outcome = <&str>::from(outcome),
            success_keys = counts.success_keys,
            partial_keys = counts.partial_keys,
            timeout_keys = counts.timeout_keys,
            error_keys = counts.error_keys,
            "Hydrator fail-open"
        );
    }
    metrics.count(&HYDRATOR_REQUESTS, [backend, call, outcome.into()], 1);
    for (result, keys, candidates) in [
        (
            HydratorOutcome::Success,
            counts.success_keys,
            counts.success_candidates,
        ),
        (
            HydratorOutcome::Partial,
            counts.partial_keys,
            counts.partial_candidates,
        ),
        (
            HydratorOutcome::Timeout,
            counts.timeout_keys,
            counts.timeout_candidates,
        ),
        (
            HydratorOutcome::Error,
            counts.error_keys,
            counts.error_candidates,
        ),
    ] {
        let values = [backend, call, result.into()];
        metrics.count(&HYDRATOR_KEYS, values, keys as u64);
        metrics.count(&HYDRATOR_TWEET_IDS, values, candidates as u64);
    }
    metrics
        .plain()
        .observe(&HYDRATOR_LATENCY_MS, [backend, call], latency_ms);
}

pub(crate) fn record_viewer_country(metrics: &RequestMetrics, result: &'static str) {
    metrics.count(&VIEWER_COUNTRY, [result], 1);
}

pub(crate) fn record_tweet_states(
    metrics: &RequestMetrics,
    states: RawHydrationBatch<TweetState>,
) -> RawHydrationBatch<TweetState> {
    let answers = states.into_hydrated();
    let mut keys_by_state: FxHashMap<&'static str, u64> = FxHashMap::default();
    for answer in answers.values() {
        let state = match answer {
            Hydrated::Found(state) | Hydrated::Partial(state) => (*state).into(),
            Hydrated::NotFound => TweetState::NotFound.into(),
            Hydrated::Failed | Hydrated::Unanswered => "failed",
        };
        *keys_by_state.entry(state).or_default() += 1;
    }
    for (state, keys) in keys_by_state {
        metrics.count(&TWEET_STATE, [state], keys);
    }
    HydrationBatch::from_hydrated(answers)
}

const REQUEST_COUNTRY_EXTRA: [&str; 28] = [
    "ae", "ch", "cl", "co", "ec", "eg", "gh", "hk", "il", "iq", "jp", "ke", "kw", "my", "ng", "om",
    "pe", "pk", "sa", "sg", "tr", "tw", "ua", "us", "ve", "vn", "ye", "za",
];

fn request_country_label(request_country: Option<&str>) -> &'static str {
    let Some(code) = request_country else {
        return "none";
    };
    CountryList::listed_by_default(code)
        .or_else(|| {
            REQUEST_COUNTRY_EXTRA
                .into_iter()
                .find(|&extra| extra == code)
        })
        .unwrap_or("other")
}

pub(crate) fn record_request_country(metrics: &Metrics, request_country: Option<&str>) {
    metrics.count(
        &REQUEST_COUNTRY,
        [request_country_label(request_country)],
        1,
    );
}

pub(crate) fn record_flock_missing_keys(
    metrics: &RequestMetrics,
    backend: &str,
    call: &str,
    keys: usize,
) {
    metrics.count(&FLOCK_MISSING_KEYS, [backend, call], keys as u64);
}

#[derive(Default)]
pub(crate) struct FallbackCacheCounts {
    pub(crate) fresh: usize,
    pub(crate) stale: usize,
    pub(crate) not_found: usize,
    pub(crate) partial: usize,
    pub(crate) partial_stale: usize,
    pub(crate) unavailable: usize,
    pub(crate) resident: usize,
}

impl FallbackCacheCounts {
    pub(crate) fn record(&self, metrics: &Metrics, cache: &'static str, column: &'static str) {
        for (result, count) in [
            (FallbackCacheResult::Fresh, self.fresh),
            (FallbackCacheResult::Stale, self.stale),
            (FallbackCacheResult::NotFound, self.not_found),
            (FallbackCacheResult::Partial, self.partial),
            (FallbackCacheResult::PartialStale, self.partial_stale),
            (FallbackCacheResult::Unavailable, self.unavailable),
        ] {
            metrics.count(
                &FALLBACK_CACHE_KEYS,
                [cache, column, result.into()],
                count as u64,
            );
        }
        let live = self.fresh + self.not_found + self.partial + self.partial_stale;
        for (result, count) in [
            ("resident", self.resident),
            ("absent", live - self.resident),
        ] {
            metrics.count(
                &FALLBACK_CACHE_RESIDENT_KEYS,
                [cache, column, result],
                count as u64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::sources::{Reply, Source};
    use crate::hydration::store::Store;
    use crate::models::{RawCandidate, RequestFacts, RetweetMerge, TweetFeatures, TweetId};
    use crate::rules::RuleEngine;
    use crate::rules::SafetyLevel;
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn a_request_country_reads_its_listed_code_other_or_none() {
        for (country, label) in [
            (Some("de"), "de"),
            (Some("us"), "us"),
            (Some("zz"), "other"),
            (None, "none"),
        ] {
            assert_eq!(request_country_label(country), label, "{country:?}");
        }
        for extra in REQUEST_COUNTRY_EXTRA {
            assert_eq!(
                CountryList::listed_by_default(extra),
                None,
                "{extra} is listed"
            );
        }
    }

    #[test]
    fn partial_keys_make_a_partial_call_and_a_failed_one_key_call_is_an_error() {
        use crate::hydration::decode::author::author_batch;
        use xai_core_entities::entities::{GizmoduckUserResult, UserResponseState};
        let user = |state| {
            Ok::<_, ()>(Some(GizmoduckUserResult {
                user: Some(Default::default()),
                response_state: Some(state),
            }))
        };
        let authors = |states: &[(u64, UserResponseState)]| {
            let results = states
                .iter()
                .map(|&(author, state)| (author, user(state)))
                .collect::<HashMap<_, _>>();
            let expected: Vec<u64> = results.keys().copied().collect();
            author_batch(HydrationBatch::from_results(expected, results))
        };
        let counts = FxHashMap::from_iter([(10, 1), (20, 2)]);

        let one_partial = KeyedResultCounts::from_batch(
            &counts,
            &authors(&[
                (10, UserResponseState::Found),
                (20, UserResponseState::Partial),
            ]),
        );
        assert_eq!(
            (
                one_partial.success_keys,
                one_partial.partial_keys,
                one_partial.partial_candidates
            ),
            (1, 1, 2)
        );
        assert_eq!(one_partial.outcome(), HydratorOutcome::Partial);
        let all_partial = KeyedResultCounts::from_batch(
            &counts,
            &authors(&[
                (10, UserResponseState::Failed),
                (20, UserResponseState::Partial),
            ]),
        );
        assert_eq!(all_partial.outcome(), HydratorOutcome::Partial);

        let viewer: HydrationBatch<u64, u8> = HydrationBatch::from_results(
            [50],
            HashMap::from([(50, Err::<Option<u8>, _>("gizmoduck unavailable"))]),
        );
        let viewer = KeyedResultCounts::from_batch(&FxHashMap::from_iter([(50, 1)]), &viewer);
        assert_eq!((viewer.error_keys, viewer.error_candidates), (1, 1));
        assert_eq!(viewer.outcome(), HydratorOutcome::Error);
    }

    #[test]
    fn from_batch_counts_not_found_as_success_and_missing_as_error() {
        let expected = FxHashMap::from_iter([(1, 1), (2, 2), (3, 3), (4, 4)]);
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
    fn a_tweet_keyed_call_counts_its_keys_before_pure_core_resolves_authors() {
        let engine = RuleEngine::for_tests();
        let plan = engine.plan(SafetyLevel::TimelineHomeHydration);
        let raw = [RawCandidate {
            tweet_id: TweetId(1),
            request_author_id: None,
        }];
        let mut store = Store::new(
            plan,
            RequestFacts::logged_in(50),
            &raw,
            RetweetMerge::Unmerged,
        );
        let group = |source| {
            plan.groups()
                .find(|group| group.fetcher.source == source)
                .unwrap()
        };
        let tweets = store.offer(group(Source::TesTweetForVisibility)).unwrap();
        let circle = TweetFeatures {
            trusted_friends_list_id: Some(7),
            ..Default::default()
        };
        let landed =
            HydrationBatch::from_results([1], HashMap::from([(1, Ok::<_, ()>(Some(circle)))]));
        store.land(
            &tweets,
            Reply::TesTweetForVisibility(landed),
            Duration::ZERO,
        );

        let call = store.offer(group(Source::TrustedFriends)).unwrap();
        let failed: HydrationBatch<u64, bool> =
            HydrationBatch::from_results(call.keys.clone(), HashMap::<_, Result<_, ()>>::new());
        let counts = KeyedResultCounts::from_batch(&call.candidate_count_by_claimed_key, &failed);
        assert_eq!((counts.error_keys, counts.error_candidates), (1, 1));
        assert_eq!(counts.outcome(), HydratorOutcome::Error);
    }

    #[test]
    fn a_claimed_key_no_candidate_counted_still_counts_its_timeout() {
        let counts = KeyedResultCounts::timed_out(&FxHashMap::from_iter([(40, 0), (41, 3)]));

        assert_eq!((counts.timeout_keys, counts.timeout_candidates), (2, 3));
        assert_eq!(counts.outcome(), HydratorOutcome::Timeout);
    }
}
