use std::fmt;
use std::marker::PhantomData;

use xai_stats_receiver::HistogramBuckets;

#[derive(Clone, Copy)]
pub(crate) enum Buckets {
    Fixed(HistogramBuckets),
    Vm,
}

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Counter,
    Histogram(Buckets),
    Gauge,
}

#[derive(Clone, Copy)]
pub(crate) enum RequestLabels {
    None,
    Rpc,
    RpcAndLevel,
}

impl RequestLabels {
    pub(crate) const fn labels(self) -> &'static [&'static str] {
        match self {
            Self::None => &[],
            Self::Rpc => &["rpc"],
            Self::RpcAndLevel => &["rpc", "safety_level"],
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Family {
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
    pub(crate) labels: &'static [&'static str],
    pub(crate) request: RequestLabels,
}

pub(crate) const MAX_LABELS: usize = 11;

impl Family {
    const fn new(
        name: &'static str,
        kind: Kind,
        labels: &'static [&'static str],
        request: RequestLabels,
    ) -> Self {
        assert!(labels.len() + request.labels().len() <= MAX_LABELS);
        Self {
            name,
            kind,
            labels,
            request,
        }
    }

    pub(crate) fn all_labels(&self) -> impl Iterator<Item = &'static str> {
        self.labels.iter().chain(self.request.labels()).copied()
    }
}

pub(crate) struct ByRequest;
pub(crate) struct Plain;

pub(crate) struct Counter<const N: usize, S> {
    pub(crate) family: Family,
    scope: PhantomData<S>,
}

pub(crate) struct Histogram<const N: usize, S> {
    pub(crate) family: Family,
    scope: PhantomData<S>,
}

pub(crate) struct Gauge<const N: usize> {
    pub(crate) family: Family,
}

impl<const N: usize> Counter<N, ByRequest> {
    const fn by_request(name: &'static str, labels: &'static [&'static str; N]) -> Self {
        Self {
            family: Family::new(name, Kind::Counter, labels, RequestLabels::RpcAndLevel),
            scope: PhantomData,
        }
    }
}

impl<const N: usize> Counter<N, Plain> {
    const fn plain(name: &'static str, labels: &'static [&'static str; N]) -> Self {
        Self {
            family: Family::new(name, Kind::Counter, labels, RequestLabels::None),
            scope: PhantomData,
        }
    }
}

impl<const N: usize> Histogram<N, ByRequest> {
    const fn by_request(
        name: &'static str,
        buckets: Buckets,
        labels: &'static [&'static str; N],
    ) -> Self {
        let kind = Kind::Histogram(buckets);
        Self {
            family: Family::new(name, kind, labels, RequestLabels::Rpc),
            scope: PhantomData,
        }
    }
}

impl<const N: usize> Histogram<N, Plain> {
    const fn plain(
        name: &'static str,
        buckets: Buckets,
        labels: &'static [&'static str; N],
    ) -> Self {
        let kind = Kind::Histogram(buckets);
        Self {
            family: Family::new(name, kind, labels, RequestLabels::None),
            scope: PhantomData,
        }
    }
}

impl<const N: usize> Gauge<N> {
    const fn new(name: &'static str, labels: &'static [&'static str; N]) -> Self {
        Self {
            family: Family::new(name, Kind::Gauge, labels, RequestLabels::None),
        }
    }
}

pub(crate) const REQUESTS: Counter<1, ByRequest> = Counter::by_request("vf_requests", &["outcome"]);

pub(crate) const REQUEST_LATENCY_MS: Histogram<0, ByRequest> =
    Histogram::by_request("vf_request_latency_ms", Buckets::Vm, &[]);

pub(crate) const REQUEST_DEADLINE: Counter<1, ByRequest> =
    Counter::by_request("vf_request_deadline", &["outcome"]);

pub(crate) const REQUEST_BATCH_SIZE: Histogram<0, ByRequest> = Histogram::by_request(
    "vf_request_batch_size",
    Buckets::Fixed(HistogramBuckets::Bucket50To500),
    &[],
);

pub(crate) const REQUESTS_BY_CALLER: Counter<3, Plain> = Counter::plain(
    "vf_requests_by_caller",
    &["caller_identity", "rpc", "outcome"],
);

pub(crate) const REQUEST_LATENCY_BY_CALLER_MS: Histogram<2, Plain> = Histogram::plain(
    "vf_request_latency_by_caller_ms",
    Buckets::Vm,
    &["caller_identity", "rpc"],
);

pub(crate) const VERDICTS: Counter<1, ByRequest> = Counter::by_request("vf_verdicts", &["action"]);

pub(crate) const VERDICTS_BY_RULE: Counter<2, ByRequest> =
    Counter::by_request("vf_verdicts_by_rule", &["rule", "action"]);

pub(crate) const EVALUATED_CANDIDATES: Counter<1, ByRequest> =
    Counter::by_request("vf_evaluated_candidates", &["fail_open_defaults"]);

pub(crate) const UNRESOLVED_TWEETS: Counter<2, ByRequest> =
    Counter::by_request("vf_unresolved_tweets", &["lookup", "cause"]);

pub(crate) const LOGGED_OUT_VIEWER: Counter<0, Plain> = Counter::plain("vf_logged_out_viewer", &[]);

pub(crate) const VIEWER_ID_NORMALIZED: Counter<0, Plain> =
    Counter::plain("vf_viewer_id_normalized", &[]);

pub(crate) const PHASE_MS: Histogram<2, ByRequest> =
    Histogram::by_request("vf_phase_ms", Buckets::Vm, &["stage", "sources"]);

pub(crate) const NOTIFICATION_RESULTS: Counter<3, ByRequest> = Counter::by_request(
    "filter_notifications_results",
    &["notification_type", "outcome", "reason"],
);

pub(crate) const RETWEET_SOURCES: Counter<1, ByRequest> =
    Counter::by_request("evaluate_tweets_retweet_sources", &["outcome"]);

pub(crate) const QUOTE_PASS: Counter<1, ByRequest> =
    Counter::by_request("vf_quote_pass", &["outcome"]);

pub(crate) const HYDRATOR_REQUESTS: Counter<3, ByRequest> =
    Counter::by_request("vf_hydrator_requests", &["backend", "call", "outcome"]);

pub(crate) const HYDRATOR_KEYS: Counter<3, ByRequest> =
    Counter::by_request("vf_hydrator_keys", &["backend", "call", "result"]);

pub(crate) const HYDRATOR_TWEET_IDS: Counter<3, ByRequest> =
    Counter::by_request("vf_hydrator_tweet_ids", &["backend", "call", "result"]);

pub(crate) const HYDRATOR_LATENCY_MS: Histogram<2, Plain> = Histogram::plain(
    "vf_hydrator_latency_ms",
    Buckets::Fixed(HistogramBuckets::Bucket50To500),
    &["backend", "call"],
);

pub(crate) const HYDRATOR_EXPANDED_BATCHES: Counter<2, Plain> =
    Counter::plain("vf_hydrator_expanded_batches", &["backend", "call"]);

pub(crate) const HYDRATOR_EXPANDED_KEYS: Counter<2, Plain> =
    Counter::plain("vf_hydrator_expanded_keys", &["backend", "call"]);

pub(crate) const HYDRATOR_UNASKED_KEYS: Counter<2, Plain> =
    Counter::plain("vf_hydrator_unasked_keys", &["backend", "call"]);

pub(crate) const FLOCK_MISSING_KEYS: Counter<2, ByRequest> =
    Counter::by_request("vf_flock_missing_keys", &["backend", "call"]);

pub(crate) const FALLBACK_CACHE_KEYS: Counter<3, Plain> =
    Counter::plain("vf_fallback_cache_keys", &["cache", "column", "result"]);

pub(crate) const FALLBACK_CACHE_RESIDENT_KEYS: Counter<3, Plain> = Counter::plain(
    "vf_fallback_cache_resident_keys",
    &["cache", "column", "result"],
);

pub(crate) const FALLBACK_CACHE_ENTRIES: Gauge<1> =
    Gauge::new("vf_fallback_cache_entries", &["cache"]);

pub(crate) const VIEWER_COUNTRY: Counter<1, ByRequest> =
    Counter::by_request("vf_viewer_country", &["result"]);

pub(crate) const VIEWER_STATE: Counter<1, ByRequest> =
    Counter::by_request("vf_viewer_state", &["state"]);

pub(crate) const TWEET_STATE: Counter<1, ByRequest> =
    Counter::by_request("vf_tweet_state", &["state"]);

pub(crate) const REQUEST_COUNTRY: Counter<1, Plain> =
    Counter::plain("vf_request_country", &["country"]);

pub(crate) const VIEWER_NOT_AGE_VERIFIED: Counter<2, Plain> =
    Counter::plain("vf_viewer_not_age_verified", &["country", "age"]);

pub(crate) const LOOKUP_TWEET_IDS: Counter<1, Plain> =
    Counter::plain("safety_labels_lookup_tweet_ids", &["result"]);

pub(crate) const LOOKUP_FAILURES: Counter<1, Plain> =
    Counter::plain("safety_labels_lookup_failures", &["kind"]);

pub(crate) const TWEETYPIE_REFERENCE_COMPARE: Counter<11, Plain> = Counter::plain(
    "vf_tweetypie_reference_compare",
    &[
        "safety_level",
        "placement",
        "tp",
        "tp_reason",
        "vf",
        "vf_reason",
        "vf_rule",
        "quoted_tweet_rule",
        "quote_pass",
        "renders_alike",
        "build_sha",
    ],
);

pub(crate) const TWEETYPIE_REFERENCE_BATCHES: Counter<3, Plain> = Counter::plain(
    "vf_tweetypie_reference_batches",
    &["safety_level", "outcome", "build_sha"],
);

pub(crate) const FEATURE_SWITCH_LOAD_FAILURES: Counter<1, Plain> =
    Counter::plain("feature_switch_load_failures", &["file"]);

pub(crate) const LIMITED_ACTIONS_POLICY_MISSING: Counter<2, Plain> =
    Counter::plain("limited_actions_policy_missing", &["reason", "missing"]);

pub(crate) const LIMITED_ACTIONS_COPY_KEYS: Gauge<0> = Gauge::new("limited_actions_copy_keys", &[]);

pub(crate) const LIMITED_ACTIONS_COPY_DRIFT_CHECKS: Counter<1, Plain> =
    Counter::plain("limited_actions_copy_drift_checks", &["result"]);

#[cfg(test)]
pub(crate) const REQUEST_LABEL_EXEMPTIONS: &[(&Family, &str)] = &[
    (
        &HYDRATOR_LATENCY_MS.family,
        "its one panel groups by backend and call; rpc waits for a panel that splits by it",
    ),
    (&HYDRATOR_EXPANDED_BATCHES.family, "no reader"),
    (&HYDRATOR_EXPANDED_KEYS.family, "no reader"),
    (
        &HYDRATOR_UNASKED_KEYS.family,
        "reads 0 on every request; a count is a bug whatever the request",
    ),
    (
        &FALLBACK_CACHE_KEYS.family,
        "measures the shared fallback cache",
    ),
    (
        &FALLBACK_CACHE_RESIDENT_KEYS.family,
        "measures the shared fallback cache",
    ),
    (
        &FALLBACK_CACHE_ENTRIES.family,
        "measures the shared fallback cache",
    ),
    (
        &REQUEST_COUNTRY.family,
        "country times safety_level multiplies the series",
    ),
    (
        &VIEWER_NOT_AGE_VERIFIED.family,
        "country times safety_level multiplies the series",
    ),
    (
        &REQUESTS_BY_CALLER.family,
        "keeps its own rpc; caller times safety_level multiplies the series",
    ),
    (
        &REQUEST_LATENCY_BY_CALLER_MS.family,
        "keeps its own rpc beside the caller",
    ),
    (&LOGGED_OUT_VIEWER.family, "no reader"),
    (&VIEWER_ID_NORMALIZED.family, "no reader"),
    (
        &LOOKUP_TWEET_IDS.family,
        "GetSafetyLabels only, whose rpc and safety_level never vary",
    ),
    (
        &LOOKUP_FAILURES.family,
        "GetSafetyLabels only, whose rpc and safety_level never vary",
    ),
    (
        &LIMITED_ACTIONS_POLICY_MISSING.family,
        "EvaluateTweets only; no reader",
    ),
    (
        &TWEETYPIE_REFERENCE_COMPARE.family,
        "keeps its own safety_level, the staging comparator's re-evaluation level",
    ),
    (
        &TWEETYPIE_REFERENCE_BATCHES.family,
        "keeps its own safety_level, the staging comparator's re-evaluation level",
    ),
    (
        &FEATURE_SWITCH_LOAD_FAILURES.family,
        "not on the request path",
    ),
    (&LIMITED_ACTIONS_COPY_KEYS.family, "not on the request path"),
    (
        &LIMITED_ACTIONS_COPY_DRIFT_CHECKS.family,
        "not on the request path",
    ),
];

#[cfg(test)]
pub(crate) const FAMILIES: &[Family] = &[
    RETWEET_SOURCES.family,
    FEATURE_SWITCH_LOAD_FAILURES.family,
    NOTIFICATION_RESULTS.family,
    LIMITED_ACTIONS_COPY_DRIFT_CHECKS.family,
    LIMITED_ACTIONS_COPY_KEYS.family,
    LIMITED_ACTIONS_POLICY_MISSING.family,
    LOOKUP_FAILURES.family,
    LOOKUP_TWEET_IDS.family,
    EVALUATED_CANDIDATES.family,
    FALLBACK_CACHE_ENTRIES.family,
    FALLBACK_CACHE_KEYS.family,
    FALLBACK_CACHE_RESIDENT_KEYS.family,
    FLOCK_MISSING_KEYS.family,
    HYDRATOR_EXPANDED_BATCHES.family,
    HYDRATOR_EXPANDED_KEYS.family,
    HYDRATOR_KEYS.family,
    HYDRATOR_LATENCY_MS.family,
    HYDRATOR_REQUESTS.family,
    HYDRATOR_TWEET_IDS.family,
    HYDRATOR_UNASKED_KEYS.family,
    LOGGED_OUT_VIEWER.family,
    PHASE_MS.family,
    QUOTE_PASS.family,
    REQUEST_BATCH_SIZE.family,
    REQUEST_COUNTRY.family,
    REQUEST_DEADLINE.family,
    REQUEST_LATENCY_BY_CALLER_MS.family,
    REQUEST_LATENCY_MS.family,
    REQUESTS.family,
    REQUESTS_BY_CALLER.family,
    TWEET_STATE.family,
    TWEETYPIE_REFERENCE_BATCHES.family,
    TWEETYPIE_REFERENCE_COMPARE.family,
    UNRESOLVED_TWEETS.family,
    VERDICTS.family,
    VERDICTS_BY_RULE.family,
    VIEWER_COUNTRY.family,
    VIEWER_ID_NORMALIZED.family,
    VIEWER_NOT_AGE_VERIFIED.family,
    VIEWER_STATE.family,
];

fn bucket_name(buckets: HistogramBuckets) -> &'static str {
    match buckets {
        HistogramBuckets::Bucket0To1 => "Bucket0To1",
        HistogramBuckets::Bucket0To50 => "Bucket0To50",
        HistogramBuckets::Bucket50To500 => "Bucket50To500",
        HistogramBuckets::Bucket500To1000 => "Bucket500To1000",
        HistogramBuckets::Bucket500To2500 => "Bucket500To2500",
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        match self.kind {
            Kind::Counter => write!(f, " counter")?,
            Kind::Histogram(Buckets::Vm) => write!(f, " histogram:vm")?,
            Kind::Histogram(Buckets::Fixed(buckets)) => {
                write!(f, " histogram:{}", bucket_name(buckets))?;
            }
            Kind::Gauge => write!(f, " gauge")?,
        }
        for key in self.labels {
            write!(f, " {key}")?;
        }
        for key in self.request.labels() {
            write!(f, " +{key}")?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const SAFETY_LABEL_SOURCE_PREFIX: &str = "safety_labels_";

    #[test]
    fn the_registry_is_pinned() {
        let rendered: String = FAMILIES.iter().map(ToString::to_string).collect();
        crate::pinned::assert_pinned("tests/fixtures/metric_families.txt", &rendered);
    }

    #[test]
    fn families_are_sorted_by_name_and_named_once() {
        let names: Vec<&str> = FAMILIES.iter().map(|family| family.name).collect();
        assert!(names.is_sorted_by(|a, b| a < b), "{names:?}");
    }

    #[test]
    fn each_family_takes_the_request_labels_or_names_its_exemption() {
        for family in FAMILIES {
            let exempt = REQUEST_LABEL_EXEMPTIONS
                .iter()
                .any(|(exempt, _)| exempt.name == family.name);
            let by_request = !matches!(family.request, RequestLabels::None);
            assert_ne!(
                by_request, exempt,
                "{}: record it by request or list it in REQUEST_LABEL_EXEMPTIONS",
                family.name
            );
            let own_request_key = family
                .labels
                .iter()
                .any(|&key| matches!(key, "rpc" | "safety_level"));
            assert!(!(by_request && own_request_key), "{}", family.name);
        }
    }

    fn documented_keys() -> Vec<(String, BTreeSet<String>)> {
        let doc = std::fs::read_to_string(crate::pinned::crate_path("docs/metrics.md")).unwrap();
        let server = doc.split("Client-side").next().unwrap();
        server
            .lines()
            .filter_map(|line| {
                let mut cells = line.strip_prefix("| `")?.splitn(3, '|');
                let name = cells.next()?.trim().strip_suffix('`')?.to_owned();
                let tags = cells.next()?.replace('`', "");
                let mut depth = 0;
                let mut keys = BTreeSet::new();
                let mut key = String::new();
                for c in tags.chars().chain([',']) {
                    match c {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        ',' if depth == 0 => {
                            let tag = std::mem::take(&mut key);
                            let tag = tag.split('=').next().unwrap_or_default().trim();
                            if !matches!(tag, "" | "-") {
                                keys.insert(tag.to_owned());
                            }
                        }
                        _ if depth == 0 => key.push(c),
                        _ => {}
                    }
                }
                Some((name, keys))
            })
            .collect()
    }

    #[test]
    fn docs_metrics_md_lists_each_family_with_its_label_keys() {
        let documented = documented_keys();
        for family in FAMILIES {
            let keys: BTreeSet<String> = family.all_labels().map(str::to_owned).collect();
            assert!(
                documented.contains(&(family.name.to_owned(), keys.clone())),
                "docs/metrics.md has no `{}` row with tags {keys:?}",
                family.name
            );
        }
        for (name, _) in &documented {
            assert!(
                name.starts_with(SAFETY_LABEL_SOURCE_PREFIX)
                    || FAMILIES.iter().any(|family| family.name == name),
                "docs/metrics.md documents {name}, which FAMILIES lacks"
            );
        }
    }
}
