use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tonic::metadata::MetadataMap;
use tonic::{Request, Status};
#[cfg(test)]
use xai_stats_receiver::HistogramBuckets;
use xai_stats_receiver::StatsReceiverExt;

use crate::caller_identity;
use crate::hydration::{Cause, Hydrators, Lookup, INBOUND_ALLOWANCE};
use crate::models::{Evaluation, Verdict, VerdictAction};
use crate::rules::SafetyLevel;
use crate::treatment;

pub(crate) mod families;

use families::{
    Buckets, ByRequest, Counter, Family, Gauge, Histogram, Kind, Plain, EVALUATED_CANDIDATES,
    LOGGED_OUT_VIEWER, MAX_LABELS, NOTIFICATION_RESULTS, PHASE_MS, REQUESTS, REQUESTS_BY_CALLER,
    REQUEST_BATCH_SIZE, REQUEST_DEADLINE, REQUEST_LATENCY_BY_CALLER_MS, REQUEST_LATENCY_MS,
    UNRESOLVED_TWEETS, VERDICTS, VERDICTS_BY_RULE, VIEWER_ID_NORMALIZED,
};

const FAIL_OPEN_DEFAULTS_NONE: &str = "none";
const UNKNOWN_IDENTITY: &str = "unknown";

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Endpoint {
    FilterTweets,
    EvaluateTweets,
    GetSafetyLabels,
    FilterNotifications,
}

#[derive(Clone, Copy)]
pub(crate) enum RequestLevel {
    Named(SafetyLevel),
    Unknown,
    Absent,
}

impl RequestLevel {
    pub(crate) fn of(level: Option<SafetyLevel>) -> Self {
        level.map_or(Self::Unknown, Self::Named)
    }
}

impl From<RequestLevel> for &'static str {
    fn from(level: RequestLevel) -> Self {
        match level {
            RequestLevel::Named(level) => level.into(),
            RequestLevel::Unknown => "unknown",
            RequestLevel::Absent => "none",
        }
    }
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum RequestOutcome {
    Started,
    Cancelled,
    Success,
    Failure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum DeadlineOutcome {
    Absent,
    Within,
    Overrun,
    CancelledEarly,
    CancelledAtDeadline,
    CancelledNoDeadline,
}

impl DeadlineOutcome {
    pub(crate) fn of(
        outcome: RequestOutcome,
        grpc_timeout: Option<Duration>,
        elapsed: Duration,
    ) -> Option<Self> {
        match (outcome, grpc_timeout) {
            (RequestOutcome::Started | RequestOutcome::Failure, _) => None,
            (RequestOutcome::Success, None) => Some(Self::Absent),
            (RequestOutcome::Cancelled, None) => Some(Self::CancelledNoDeadline),
            (RequestOutcome::Success, Some(deadline)) => Some(if elapsed <= deadline {
                Self::Within
            } else {
                Self::Overrun
            }),
            (RequestOutcome::Cancelled, Some(deadline)) => {
                Some(if elapsed + INBOUND_ALLOWANCE < deadline {
                    Self::CancelledEarly
                } else {
                    Self::CancelledAtDeadline
                })
            }
        }
    }
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Stage {
    Hydration,
    PostHydration,
}

#[derive(Clone)]
pub(crate) struct Metrics {
    sink: Option<Arc<dyn StatsReceiverExt>>,
}

impl Metrics {
    pub(crate) fn new(sink: Option<Arc<dyn StatsReceiverExt>>) -> Self {
        Self { sink }
    }

    pub(crate) fn none() -> Self {
        Self { sink: None }
    }

    pub(crate) fn request(&self, endpoint: Endpoint, level: RequestLevel) -> RequestMetrics {
        RequestMetrics {
            metrics: self.clone(),
            endpoint,
            level,
        }
    }

    pub(crate) fn count<const N: usize>(
        &self,
        family: &Counter<N, Plain>,
        values: [&str; N],
        n: u64,
    ) {
        count(self, &family.family, &values, &[], n);
    }

    pub(crate) fn observe<const N: usize>(
        &self,
        family: &Histogram<N, Plain>,
        values: [&str; N],
        value: f64,
    ) {
        observe(self, &family.family, &values, &[], value);
    }

    pub(crate) fn set<const N: usize>(&self, family: &Gauge<N>, values: [&str; N], value: f64) {
        self.emit(&family.family, &values, &[], |sink, name, pairs| {
            sink.gauge(name, pairs, value);
        });
    }

    fn emit(
        &self,
        family: &Family,
        own: &[&str],
        request: &[&str],
        record: impl FnOnce(&dyn StatsReceiverExt, &str, &[(&str, &str)]),
    ) {
        let Some(sink) = &self.sink else {
            return;
        };
        let mut pairs = [("", ""); MAX_LABELS];
        let mut len = 0;
        for (pair, (key, &value)) in pairs
            .iter_mut()
            .zip(family.all_labels().zip(own.iter().chain(request)))
        {
            *pair = (key, value);
            len += 1;
        }
        record(&**sink, family.name, pairs.get(..len).unwrap_or_default());
    }
}

fn count(metrics: &Metrics, family: &Family, own: &[&str], request: &[&str], n: u64) {
    if n > 0 {
        metrics.emit(family, own, request, |sink, name, pairs| {
            sink.incr(name, pairs, n);
        });
    }
}

fn observe(metrics: &Metrics, family: &Family, own: &[&str], request: &[&str], value: f64) {
    metrics.emit(family, own, request, |sink, name, pairs| {
        match family.kind {
            Kind::Histogram(Buckets::Vm) => sink.observe_vm(name, pairs, value),
            Kind::Histogram(Buckets::Fixed(buckets)) => sink.observe(name, pairs, value, buckets),
            Kind::Counter | Kind::Gauge => {}
        }
    });
}

#[derive(Clone)]
pub(crate) struct RequestMetrics {
    metrics: Metrics,
    endpoint: Endpoint,
    level: RequestLevel,
}

impl RequestMetrics {
    pub(crate) fn count<const N: usize>(
        &self,
        family: &Counter<N, ByRequest>,
        values: [&str; N],
        n: u64,
    ) {
        let request = [self.endpoint.into(), self.level.into()];
        count(&self.metrics, &family.family, &values, &request, n);
    }

    pub(crate) fn observe<const N: usize>(
        &self,
        family: &Histogram<N, ByRequest>,
        values: [&str; N],
        value: f64,
    ) {
        let request = [self.endpoint.into()];
        observe(&self.metrics, &family.family, &values, &request, value);
    }

    pub(crate) fn at_level(&self, level: SafetyLevel) -> Self {
        self.metrics
            .request(self.endpoint, RequestLevel::Named(level))
    }

    pub(crate) fn endpoint(&self) -> Endpoint {
        self.endpoint
    }

    pub(crate) fn level(&self) -> RequestLevel {
        self.level
    }

    pub(crate) fn plain(&self) -> &Metrics {
        &self.metrics
    }
}

pub(crate) fn record_viewer_state(metrics: &Metrics, raw: Option<u64>, normalized: Option<u64>) {
    if normalized.is_some() {
        return;
    }
    metrics.count(&LOGGED_OUT_VIEWER, [], 1);
    if raw.is_some() {
        metrics.count(&VIEWER_ID_NORMALIZED, [], 1);
    }
}

pub(crate) fn record_verdicts<'a>(
    metrics: &RequestMetrics,
    verdicts: impl IntoIterator<Item = &'a Verdict>,
) {
    let mut mix: HashMap<VerdictAction, u64> = HashMap::new();
    let mut by_rule: HashMap<(&str, VerdictAction), u64> = HashMap::new();
    for verdict in verdicts {
        *mix.entry(treatment::verdict_action(verdict)).or_default() += 1;
        for row in treatment::decided_rows(verdict) {
            *by_rule.entry(row).or_default() += 1;
        }
    }
    for (action, count) in mix {
        metrics.count(&VERDICTS, [action.into()], count);
    }
    for ((rule, action), count) in by_rule {
        metrics.count(&VERDICTS_BY_RULE, [rule, action.into()], count);
    }
}

pub(crate) fn record_sent<'a>(
    metrics: &RequestMetrics,
    evaluations: impl Iterator<Item = &'a Evaluation> + Clone,
    unevaluated: usize,
) {
    static LOOKUP_FAILED: Verdict = Verdict::lookup_failed();
    record_verdicts(
        metrics,
        evaluations
            .clone()
            .map(Evaluation::verdict)
            .chain(std::iter::repeat_n(&LOOKUP_FAILED, unevaluated)),
    );
    match metrics.endpoint() {
        Endpoint::FilterTweets | Endpoint::EvaluateTweets => {
            record_unresolved(metrics, evaluations.clone());
        }
        Endpoint::GetSafetyLabels | Endpoint::FilterNotifications => {}
    }
    record_fail_open_defaults(metrics, evaluations.map(Evaluation::fail_open_defaults));
}

fn record_unresolved<'a>(
    metrics: &RequestMetrics,
    evaluations: impl IntoIterator<Item = &'a Evaluation>,
) {
    let mut counts: HashMap<(Lookup, Cause), u64> = HashMap::new();
    for evaluation in evaluations {
        let unresolved = match *evaluation {
            Evaluation::NotFound(lookup) => (lookup, Cause::NotFound),
            Evaluation::Deleted(_) => (Lookup::Tweet, Cause::NotFound),
            Evaluation::Failed(lookup) => (lookup, Cause::Failed),
            Evaluation::Complete { .. } | Evaluation::Partial { .. } => continue,
        };
        *counts.entry(unresolved).or_default() += 1;
    }
    for ((lookup, cause), count) in counts {
        metrics.count(&UNRESOLVED_TWEETS, [lookup.into(), cause.into()], count);
    }
}

fn record_fail_open_defaults(
    metrics: &RequestMetrics,
    fail_open_defaults: impl IntoIterator<Item = Hydrators>,
) {
    let mut on_none = 0;
    let mut on_failed: HashMap<Hydrators, u64> = HashMap::new();
    for nodes in fail_open_defaults {
        if nodes.is_empty() {
            on_none += 1;
        } else {
            *on_failed.entry(nodes).or_default() += 1;
        }
    }
    metrics.count(&EVALUATED_CANDIDATES, [FAIL_OPEN_DEFAULTS_NONE], on_none);
    for (nodes, count) in on_failed {
        let nodes: Vec<&str> = nodes.iter().map(<&str>::from).collect();
        metrics.count(&EVALUATED_CANDIDATES, [&nodes.join("+")], count);
    }
}

pub(crate) async fn observe_request<R, T, F>(
    metrics: &Metrics,
    endpoint: Endpoint,
    level: RequestLevel,
    request: Request<R>,
    handle: impl FnOnce(Request<R>, xai_x_rpc::CallContext, RequestMetrics) -> F,
) -> Result<T, Status>
where
    F: Future<Output = Result<T, Status>>,
{
    let entered = tokio::time::Instant::now();
    let grpc_timeout = parse_grpc_timeout(request.metadata());
    let metrics = metrics.request(endpoint, level);
    metrics.count(&REQUESTS, [RequestOutcome::Started.into()], 1);
    let mut guard = RequestMetricsGuard {
        metrics: metrics.clone(),
        caller_identity: caller_identity::identity(&request),
        grpc_timeout,
        start: Instant::now(),
        outcome: RequestOutcome::Cancelled,
    };
    let result = handle(
        request,
        crate::hydration::request_context(entered, grpc_timeout),
        metrics,
    )
    .await;
    guard.outcome = if result.is_ok() {
        RequestOutcome::Success
    } else {
        RequestOutcome::Failure
    };
    result
}

pub(crate) fn parse_grpc_timeout(metadata: &MetadataMap) -> Option<Duration> {
    let raw = metadata.get("grpc-timeout")?.to_str().ok()?;
    let (digits, unit) = raw.split_at(raw.len().checked_sub(1)?);
    if digits.is_empty() || digits.len() > 8 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: u64 = digits.parse().ok()?;
    let nanos_per_unit = match unit {
        "H" => 3_600_000_000_000,
        "M" => 60_000_000_000,
        "S" => 1_000_000_000,
        "m" => 1_000_000,
        "u" => 1_000,
        "n" => 1,
        _ => return None,
    };
    Some(Duration::from_nanos(value.checked_mul(nanos_per_unit)?))
}

struct RequestMetricsGuard {
    metrics: RequestMetrics,
    caller_identity: Option<String>,
    grpc_timeout: Option<Duration>,
    start: Instant,
    outcome: RequestOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum RetweetSources {
    #[strum(to_string = "none")]
    NoSource,
    InBatch,
    Fetched,
}

pub(crate) fn record_phase(
    metrics: &RequestMetrics,
    stage: Stage,
    sources: RetweetSources,
    elapsed: Duration,
) {
    metrics.observe(&PHASE_MS, [stage.into(), sources.into()], millis(elapsed));
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

impl RequestMetricsGuard {
    fn emit(&self, elapsed: Duration) {
        let metrics = &self.metrics;
        let outcome = self.outcome.into();
        let rpc = metrics.endpoint().into();
        let caller = self.caller_identity.as_deref().unwrap_or(UNKNOWN_IDENTITY);
        metrics.count(&REQUESTS, [outcome], 1);
        metrics
            .plain()
            .count(&REQUESTS_BY_CALLER, [caller, rpc, outcome], 1);
        if let RequestOutcome::Success = self.outcome {
            metrics.observe(&REQUEST_LATENCY_MS, [], millis(elapsed));
            metrics.plain().observe(
                &REQUEST_LATENCY_BY_CALLER_MS,
                [caller, rpc],
                millis(elapsed),
            );
        }
        if let Some(deadline) = DeadlineOutcome::of(self.outcome, self.grpc_timeout, elapsed) {
            metrics.count(&REQUEST_DEADLINE, [deadline.into()], 1);
        }
    }
}

impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        self.emit(self.start.elapsed());
    }
}

pub(crate) fn record_notification_results(
    metrics: &RequestMetrics,
    notification_type: &str,
    results: impl IntoIterator<Item = (&'static str, &'static str)>,
) {
    let mut counts: HashMap<(&str, &str), u64> = HashMap::new();
    for result in results {
        *counts.entry(result).or_default() += 1;
    }
    for ((outcome, reason), count) in counts {
        metrics.count(
            &NOTIFICATION_RESULTS,
            [notification_type, outcome, reason],
            count,
        );
    }
}

pub(crate) fn record_batch_size(metrics: &RequestMetrics, size: usize) {
    metrics.observe(&REQUEST_BATCH_SIZE, [], size as f64);
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct Recording(std::sync::Mutex<Vec<Recorded>>);

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Recorded {
    pub(crate) name: String,
    pub(crate) labels: Vec<(String, String)>,
    pub(crate) value: f64,
}

#[cfg(test)]
impl Recorded {
    pub(crate) fn label(&self, key: &str) -> Option<&str> {
        self.labels
            .iter()
            .find(|(label, _)| label == key)
            .map(|(_, value)| value.as_str())
    }
}

#[cfg(test)]
impl std::fmt::Display for Recorded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let labels: Vec<String> = self
            .labels
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        write!(f, "{}{{{}}} {}", self.name, labels.join(","), self.value)
    }
}

#[cfg(test)]
impl Metrics {
    pub(crate) fn recording() -> (Self, Arc<Recording>) {
        let recording = Arc::new(Recording::default());
        let sink: Arc<dyn StatsReceiverExt> = Arc::<Recording>::clone(&recording);
        (Self::new(Some(sink)), recording)
    }
}

#[cfg(test)]
impl Recording {
    pub(crate) fn take(&self, family: &Family) -> Vec<Recorded> {
        let mut recorded = self.0.lock().unwrap();
        let (taken, kept) = recorded
            .drain(..)
            .partition(|sample| sample.name == family.name);
        *recorded = kept;
        taken
    }

    fn push(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
        let declared = families::FAMILIES.iter().any(|family| {
            family.name == name
                && family.all_labels().count() == scopes.len()
                && family
                    .all_labels()
                    .zip(scopes)
                    .all(|(declared, &(key, _))| declared == key)
        });
        assert!(declared, "{name}{scopes:?} matches no family in FAMILIES");
        self.0.lock().unwrap().push(Recorded {
            name: name.to_owned(),
            labels: scopes
                .iter()
                .map(|&(key, value)| (key.to_owned(), value.to_owned()))
                .collect(),
            value,
        });
    }
}

#[cfg(test)]
impl StatsReceiverExt for Recording {
    fn incr(&self, name: &str, scopes: &[(&str, &str)], count: u64) {
        self.push(name, scopes, count as f64);
    }
    fn observe(&self, name: &str, scopes: &[(&str, &str)], value: f64, _: HistogramBuckets) {
        self.push(name, scopes, value);
    }
    fn observe_expo(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
        self.push(name, scopes, value);
    }
    fn observe_vm(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
        self.push(name, scopes, value);
    }
    fn gauge(&self, name: &str, scopes: &[(&str, &str)], value: f64) {
        self.push(name, scopes, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Decided, DropReason, LimitedEngagementReason, Withholding};
    use crate::rules::fixtures::{allow, blurred, blurred_and_limited, limited};
    use crate::rules::RuleId;
    use xai_visibility_filtering::models::FilteredReason;
    use xai_visibility_filtering::tweet_safety_label::{
        BATCH_LATENCY_VM_METRIC, LATENCY_VM_METRIC,
    };
    use xai_visibility_filtering::vf_client::FILTER_TWEETS_LATENCY_VM_METRIC;
    use xai_x_thrift::action::InterstitialReason;

    #[test]
    fn dashboard_generator_pins_the_client_vm_latency_metrics() {
        let dashboard =
            std::fs::read_to_string(crate::pinned::crate_path("scripts/dashboard.py")).unwrap();
        for pin in [
            format!("VF_CLIENT_LATENCY_VM_METRIC = \"{LATENCY_VM_METRIC}\""),
            format!("VF_CLIENT_BATCH_LATENCY_VM_METRIC = \"{BATCH_LATENCY_VM_METRIC}\""),
            format!("VF_CLIENT_FT_LATENCY_VM_METRIC = \"{FILTER_TWEETS_LATENCY_VM_METRIC}\""),
        ] {
            assert!(dashboard.contains(&pin), "dashboard.py is missing `{pin}`");
        }
    }

    fn drop_by(rule: &'static str) -> Verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::UnspecifiedReason)),
            id: RuleId::FilterAll,
            by: rule,
        })
    }

    #[test]
    fn each_outcome_emits_its_request_deadline_and_caller_series() {
        let (metrics, recording) = Metrics::recording();
        let metrics = metrics.request(
            Endpoint::FilterNotifications,
            RequestLevel::Named(SafetyLevel::NotificationsFollow),
        );
        for outcome in [
            RequestOutcome::Success,
            RequestOutcome::Failure,
            RequestOutcome::Cancelled,
        ] {
            let guard = std::mem::ManuallyDrop::new(RequestMetricsGuard {
                metrics: metrics.clone(),
                caller_identity: None,
                grpc_timeout: Some(Duration::from_millis(400)),
                start: Instant::now(),
                outcome,
            });
            guard.emit(Duration::from_millis(100));
        }
        let rendered = |family: &Family| -> Vec<String> {
            recording
                .take(family)
                .iter()
                .map(ToString::to_string)
                .collect()
        };
        assert_eq!(
            rendered(&REQUESTS.family),
            [
                "vf_requests{outcome=success,rpc=filter_notifications,safety_level=notifications_follow} 1",
                "vf_requests{outcome=failure,rpc=filter_notifications,safety_level=notifications_follow} 1",
                "vf_requests{outcome=cancelled,rpc=filter_notifications,safety_level=notifications_follow} 1",
            ]
        );
        assert_eq!(
            rendered(&REQUESTS_BY_CALLER.family),
            [
                "vf_requests_by_caller{caller_identity=unknown,rpc=filter_notifications,outcome=success} 1",
                "vf_requests_by_caller{caller_identity=unknown,rpc=filter_notifications,outcome=failure} 1",
                "vf_requests_by_caller{caller_identity=unknown,rpc=filter_notifications,outcome=cancelled} 1",
            ]
        );
        assert_eq!(
            rendered(&REQUEST_LATENCY_MS.family),
            ["vf_request_latency_ms{rpc=filter_notifications} 100"]
        );
        assert_eq!(
            rendered(&REQUEST_LATENCY_BY_CALLER_MS.family),
            [
                "vf_request_latency_by_caller_ms{caller_identity=unknown,rpc=filter_notifications} 100"
            ]
        );
        assert_eq!(
            rendered(&REQUEST_DEADLINE.family),
            [
                "vf_request_deadline{outcome=within,rpc=filter_notifications,safety_level=notifications_follow} 1",
                "vf_request_deadline{outcome=cancelled_early,rpc=filter_notifications,safety_level=notifications_follow} 1",
            ]
        );
    }

    #[tokio::test]
    async fn each_endpoint_counts_started_and_its_outcome_at_one_level() {
        for (endpoint, level, label) in [
            (
                Endpoint::FilterTweets,
                RequestLevel::Named(SafetyLevel::TimelineHome),
                "timeline_home",
            ),
            (Endpoint::EvaluateTweets, RequestLevel::Unknown, "unknown"),
            (Endpoint::GetSafetyLabels, RequestLevel::Absent, "none"),
            (
                Endpoint::FilterNotifications,
                RequestLevel::Named(SafetyLevel::NotificationsWriterV2),
                "notifications_writer_v2",
            ),
        ] {
            let (metrics, recording) = Metrics::recording();
            let served = observe_request(
                &metrics,
                endpoint,
                level,
                Request::new(()),
                |_, _, _| async { Ok::<_, Status>(()) },
            )
            .await;
            assert!(served.is_ok());
            let recorded: Vec<Vec<(String, String)>> = recording
                .take(&REQUESTS.family)
                .into_iter()
                .map(|sample| sample.labels)
                .collect();
            let rpc: &str = endpoint.into();
            let labels = |outcome: &str| {
                [("outcome", outcome), ("rpc", rpc), ("safety_level", label)]
                    .map(|(name, value)| (name.to_owned(), value.to_owned()))
                    .to_vec()
            };
            assert_eq!(recorded, [labels("started"), labels("success")], "{rpc}");
        }
    }

    #[test]
    fn a_cancel_counts_whether_the_caller_hung_up_before_its_deadline() {
        let deadline = Some(Duration::from_millis(400));
        let of = |outcome, ms| DeadlineOutcome::of(outcome, deadline, Duration::from_millis(ms));
        assert_eq!(
            of(RequestOutcome::Cancelled, 389),
            Some(DeadlineOutcome::CancelledEarly)
        );
        assert_eq!(
            of(RequestOutcome::Cancelled, 390),
            Some(DeadlineOutcome::CancelledAtDeadline)
        );
        assert_eq!(
            of(RequestOutcome::Cancelled, 401),
            Some(DeadlineOutcome::CancelledAtDeadline)
        );
        assert_eq!(
            DeadlineOutcome::of(RequestOutcome::Cancelled, None, Duration::ZERO),
            Some(DeadlineOutcome::CancelledNoDeadline)
        );
        assert_eq!(
            DeadlineOutcome::of(RequestOutcome::Success, None, Duration::ZERO),
            Some(DeadlineOutcome::Absent)
        );
        assert_eq!(
            of(RequestOutcome::Success, 400),
            Some(DeadlineOutcome::Within)
        );
        assert_eq!(
            of(RequestOutcome::Success, 401),
            Some(DeadlineOutcome::Overrun)
        );
        assert_eq!(of(RequestOutcome::Failure, 5), None);
    }

    #[test]
    fn each_filled_slot_counts_its_own_row() {
        let a = allow();
        let b = allow();
        let d = drop_by("nsfw_media");
        let limit_only = limited(
            LimitedEngagementReason::ConversationControl,
            "conversation_control",
        );
        let both = blurred_and_limited(
            blurred(InterstitialReason::Sensitive(true), "nsfw_media"),
            limit_only.clone(),
        );
        let recorded = |family: &Family| {
            let (metrics, recording) = Metrics::recording();
            record_verdicts(
                &metrics.request(
                    Endpoint::FilterTweets,
                    RequestLevel::Named(SafetyLevel::FilterAll),
                ),
                [&a, &b, &d, &both, &limit_only],
            );
            let mut counts: Vec<(Option<String>, String, f64)> = recording
                .take(family)
                .into_iter()
                .map(|sample| {
                    (
                        sample.label("rule").map(str::to_owned),
                        sample.label("action").unwrap().to_owned(),
                        sample.value,
                    )
                })
                .collect();
            counts.sort_by(|x, y| (&x.0, &x.1).cmp(&(&y.0, &y.1)));
            counts
        };
        let row = |rule: Option<&str>, action: &str, count: f64| {
            (rule.map(str::to_owned), action.to_owned(), count)
        };

        assert_eq!(
            recorded(&VERDICTS.family),
            [
                row(None, "allow", 2.0),
                row(None, "drop", 1.0),
                row(None, "limited_engagement", 1.0),
                row(None, "tweet_interstitial", 1.0),
            ]
        );
        assert_eq!(
            recorded(&VERDICTS_BY_RULE.family),
            [
                row(Some("conversation_control"), "limited_engagement", 2.0),
                row(Some("nsfw_media"), "drop", 1.0),
                row(Some("nsfw_media"), "interstitial", 1.0),
            ]
        );
    }

    #[test]
    fn parse_grpc_timeout_units_and_garbage() {
        let parsed = |value: &str| {
            let mut metadata = MetadataMap::new();
            metadata.insert("grpc-timeout", value.parse().unwrap());
            parse_grpc_timeout(&metadata)
        };
        assert_eq!(parsed("400m"), Some(Duration::from_millis(400)));
        assert_eq!(parsed("1S"), Some(Duration::from_secs(1)));
        assert_eq!(parsed("2M"), Some(Duration::from_secs(120)));
        assert_eq!(parsed("500u"), Some(Duration::from_micros(500)));
        assert_eq!(parsed("3H"), Some(Duration::from_secs(10_800)));
        assert_eq!(parsed("9n"), Some(Duration::from_nanos(9)));
        assert_eq!(parsed("400"), None);
        assert_eq!(parsed("m"), None);
        assert_eq!(parsed("400x"), None);
        assert_eq!(parsed("+400m"), None);
        assert_eq!(parsed("12345678m"), Some(Duration::from_millis(12_345_678)));
        assert_eq!(parsed("123456789m"), None);
        assert_eq!(parse_grpc_timeout(&MetadataMap::new()), None);
    }
}
