use std::sync::Arc;
use std::time::{Duration, Instant};

use xai_stats_receiver::{StatsReceiverExt, global_stats_receiver};

use crate::tweet_safety_label::{BATCH_LATENCY_VM_METRIC, LATENCY_VM_METRIC};
use crate::vf_client::{FILTER_TWEETS_LATENCY_VM_METRIC, grpc_timeout_expired};

pub(crate) struct ClientRpc {
    requests: &'static str,
    latency_ms: &'static str,
    batch_error: &'static str,
    batch_latency_ms: &'static str,
    failed_ids: Option<&'static str>,
}

impl ClientRpc {
    pub(crate) const GET_SAFETY_LABELS: Self = Self {
        requests: "vf_client_get_safety_labels",
        latency_ms: LATENCY_VM_METRIC,
        batch_error: "vf_client_get_safety_labels_batch_error",
        batch_latency_ms: BATCH_LATENCY_VM_METRIC,
        failed_ids: Some("vf_client_get_safety_labels_failed_ids"),
    };
    pub(crate) const FILTER_TWEETS: Self = Self {
        requests: "vf_client_filter_tweets",
        latency_ms: FILTER_TWEETS_LATENCY_VM_METRIC,
        batch_error: "vf_client_filter_tweets_batch_error",
        batch_latency_ms: "vf_client_filter_tweets_batch_latency_ms_vm",
        failed_ids: Some("vf_client_filter_tweets_failed_ids"),
    };
    pub(crate) const EVALUATE_TWEETS: Self = Self {
        requests: "vf_client_evaluate_tweets",
        latency_ms: "vf_client_evaluate_tweets_latency_ms_vm",
        batch_error: "vf_client_evaluate_tweets_batch_error",
        batch_latency_ms: "vf_client_evaluate_tweets_batch_latency_ms_vm",
        failed_ids: None,
    };
    pub(crate) const FILTER_NOTIFICATIONS: Self = Self {
        requests: "vf_client_filter_notifications",
        latency_ms: "vf_client_filter_notifications_latency_ms_vm",
        batch_error: "vf_client_filter_notifications_batch_error",
        batch_latency_ms: "vf_client_filter_notifications_batch_latency_ms_vm",
        failed_ids: None,
    };
}

fn error_code(status: &tonic::Status) -> &'static str {
    if grpc_timeout_expired(status) {
        return "deadline_exceeded";
    }
    match status.code() {
        tonic::Code::DeadlineExceeded => "deadline_exceeded",
        tonic::Code::Cancelled => "cancelled",
        tonic::Code::Unavailable => "unavailable",
        _ => "other",
    }
}

fn millis(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

pub(crate) struct RequestMetrics {
    rpc: &'static ClientRpc,
    receiver: Option<Arc<dyn StatsReceiverExt>>,
    start: Instant,
    status: &'static str,
}

impl RequestMetrics {
    pub(crate) fn start(rpc: &'static ClientRpc) -> Self {
        Self::with_receiver(rpc, global_stats_receiver())
    }

    pub(crate) fn with_receiver(
        rpc: &'static ClientRpc,
        receiver: Option<Arc<dyn StatsReceiverExt>>,
    ) -> Self {
        Self {
            rpc,
            receiver,
            start: Instant::now(),
            status: "cancelled",
        }
    }

    pub(crate) fn batch(&self, elapsed: Duration, error: Option<&tonic::Status>) {
        let Some(sr) = &self.receiver else {
            return;
        };
        let result = match error {
            None => "success",
            Some(status) => {
                sr.incr(self.rpc.batch_error, &[("code", error_code(status))], 1);
                "error"
            }
        };
        sr.observe_vm(
            self.rpc.batch_latency_ms,
            &[("result", result)],
            millis(elapsed),
        );
    }

        pub(crate) fn answered(&mut self, failed_ids: usize) {
        self.status = if failed_ids > 0 {
            "degraded"
        } else {
            "completed"
        };
        let Some(sr) = &self.receiver else {
            return;
        };
        sr.observe_vm(self.rpc.latency_ms, &[], millis(self.start.elapsed()));
        if failed_ids > 0
            && let Some(metric) = self.rpc.failed_ids
        {
            sr.incr(metric, &[], failed_ids as u64);
        }
    }

        pub(crate) fn failed(&mut self) {
        self.status = "failed";
        if let Some(sr) = &self.receiver {
            sr.observe_vm(self.rpc.latency_ms, &[], millis(self.start.elapsed()));
        }
    }
}

impl Drop for RequestMetrics {
    fn drop(&mut self) {
        if let Some(sr) = &self.receiver {
            sr.incr(self.rpc.requests, &[("requests", self.status)], 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::RecordingReceiver;

    fn series(rpc: &'static ClientRpc, call: impl FnOnce(RequestMetrics)) -> Vec<String> {
        let sr = Arc::new(RecordingReceiver::default());
        call(RequestMetrics::with_receiver(rpc, Some(sr.clone())));
        sr.series()
    }

    fn completed(mut request: RequestMetrics) {
        request.batch(Duration::from_millis(5), None);
        request.answered(0);
    }

    fn degraded(mut request: RequestMetrics) {
        request.batch(Duration::from_millis(5), None);
        request.batch(
            Duration::from_millis(5),
            Some(&tonic::Status::unavailable("")),
        );
        request.answered(50);
    }

    fn cancelled(request: RequestMetrics) {
        drop(request);
    }

    #[test]
    fn each_rpc_emits_these_series_when_completed_degraded_or_cancelled() {
        assert_eq!(
            series(&ClientRpc::GET_SAFETY_LABELS, completed),
            [
                "incr vf_client_get_safety_labels|requests=completed",
                "observe_vm vf_client_get_safety_labels_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_get_safety_labels_latency_ms_vm",
            ]
        );
        assert_eq!(
            series(&ClientRpc::GET_SAFETY_LABELS, degraded),
            [
                "incr vf_client_get_safety_labels_batch_error|code=unavailable",
                "incr vf_client_get_safety_labels_failed_ids",
                "incr vf_client_get_safety_labels|requests=degraded",
                "observe_vm vf_client_get_safety_labels_batch_latency_ms_vm|result=error",
                "observe_vm vf_client_get_safety_labels_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_get_safety_labels_latency_ms_vm",
            ]
        );
        assert_eq!(
            series(&ClientRpc::GET_SAFETY_LABELS, cancelled),
            ["incr vf_client_get_safety_labels|requests=cancelled"]
        );
        assert_eq!(
            series(&ClientRpc::FILTER_TWEETS, completed),
            [
                "incr vf_client_filter_tweets|requests=completed",
                "observe_vm vf_client_filter_tweets_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_filter_tweets_latency_ms_vm",
            ]
        );
        assert_eq!(
            series(&ClientRpc::FILTER_TWEETS, degraded),
            [
                "incr vf_client_filter_tweets_batch_error|code=unavailable",
                "incr vf_client_filter_tweets_failed_ids",
                "incr vf_client_filter_tweets|requests=degraded",
                "observe_vm vf_client_filter_tweets_batch_latency_ms_vm|result=error",
                "observe_vm vf_client_filter_tweets_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_filter_tweets_latency_ms_vm",
            ]
        );
        assert_eq!(
            series(&ClientRpc::FILTER_TWEETS, cancelled),
            ["incr vf_client_filter_tweets|requests=cancelled"]
        );
    }

    #[test]
    fn a_batch_error_counts_its_grpc_code() {
        let sr = Arc::new(RecordingReceiver::default());
        let request = RequestMetrics::with_receiver(&ClientRpc::FILTER_TWEETS, Some(sr.clone()));
        for code in [
            tonic::Code::DeadlineExceeded,
            tonic::Code::Cancelled,
            tonic::Code::Unavailable,
            tonic::Code::Internal,
            tonic::Code::Unknown,
        ] {
            request.batch(Duration::ZERO, Some(&tonic::Status::new(code, "")));
        }
        for (code, count) in [
            ("deadline_exceeded", 1),
            ("cancelled", 1),
            ("unavailable", 1),
            ("other", 2),
        ] {
            let key = format!("vf_client_filter_tweets_batch_error|code={code}");
            assert_eq!(sr.counter(&key), count, "{key}");
        }
    }
}
