use lazy_static::lazy_static;
use prometheus::{
    exponential_buckets, register_histogram, register_histogram_vec, register_int_counter,
    register_int_counter_vec, register_int_gauge, register_int_gauge_vec, Histogram, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec,
};

lazy_static! {

        pub static ref ENFORCEMENT_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_total",
            "Total enforcement actions attempted.",
            &["source", "topic", "entity_type", "step", "status", "action", "label", "head", "model"])
            .unwrap();

                                                                                    pub static ref FIRED_HEADS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_fired_heads_total",
            "Per-fired-head occurrences (one row per head in summary.fired_heads).",
            &["topic", "entity_type", "model", "head", "status"])
            .unwrap();

                                                                                pub static ref ACTION_DISPATCH_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_action_dispatch_total",
            "Composite-action dispatch outcomes (one row per composite execution).",
            &["kind", "outcome"])
            .unwrap();

                                                    pub static ref DEDUP_CLASS_BYPASS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_dedup_class_bypass_total",
            "Dedup entries re-claimed by a strictly higher-class incoming row.",
            &["topic", "head", "held_class", "incoming_class"])
            .unwrap();

                                                                    pub static ref DEDUP_WRITE_FENCED_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_dedup_write_fenced_total",
            "Dedup terminal writes dropped because a strictly stronger terminal outcome is held.",
            &["op", "held_class", "writer_class"])
            .unwrap();

                                    pub static ref DEDUP_DISPLACED_RESTORED_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_dedup_displaced_restored_total",
            "Displaced dedup holds restored after a bypassing decision resolved without acting.",
            &["op"])
            .unwrap();

                    pub static ref DEDUP_BYPASS_CLAIM_LOST_TOTAL: IntCounter =
        register_int_counter!(
            "abuse_enforcement_dedup_bypass_claim_lost_total",
            "Bypass claims lost to a concurrent claimant at nonce verification.")
            .unwrap();

            pub static ref POST_LABEL_READ_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_post_label_read_total",
            "Reads of a post's current labels before a page decision labels it.",
            &["result"])
            .unwrap();

                            pub static ref STRATO_LATENCY: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_strato_latency_seconds",
            "Strato call latency by column.",
            &["column", "status"],
            exponential_buckets(0.01, 2.0, 12).unwrap())
            .unwrap();

                                                                    pub static ref AIS_LATENCY: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_ais_latency_seconds",
            "AIS thriftmux RPC latency by method, transport outcome, and call context.",
            &["method", "status", "source", "topic", "entity_type", "action"],
            exponential_buckets(0.01, 2.0, 12).unwrap())
            .unwrap();


            pub static ref KAFKA_MESSAGE_LATENCY: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_kafka_message_seconds",
            "Per-message processing latency by outcome.",
            &["topic", "entity_type", "status", "enforced"],
            exponential_buckets(0.001, 2.0, 14).unwrap())
            .unwrap();


                                                                                                                                                    pub static ref USER_ACTIVE_SECONDS: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_user_active_seconds",
            "UAS activeSeconds per client app, after successful act-path AIS (or dry_run).",
            &["period", "client_app_id", "entity_type", "status", "action", "label"],
            exponential_buckets(1.0, 2.0, 20).unwrap())
            .unwrap();


                            pub static ref GIZMODUCK_CHECK_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_gizmoduck_check_total",
            "Gizmoduck precheck observed user state.",
            &["status"])
            .unwrap();

                    pub static ref GIZMODUCK_LATENCY: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_gizmoduck_core_latency_seconds",
            "Gizmoduck get-V2 fed-grpc fetch latency.",
            &["status"],
            exponential_buckets(0.01, 2.0, 12).unwrap())
            .unwrap();


                pub static ref STRATO_RETRIES: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_strato_retries",
            "Number of retries per Strato call by column and final status.",
            &["column", "status"],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap();


                pub static ref RETRY_ATTEMPTS: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_retry_attempts",
            "Number of attempts before a retry sequence terminates.",
            &["status"],
            vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap();

        pub static ref RETRY_QUEUE_SIZE: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_retry_queue_size",
            "Number of messages in the retry queue.")
            .unwrap();


                                        pub static ref LIMITER_DECISIONS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_limiter_decisions_total",
            "Outcome of Limiter IncrementFeature calls for the global cap.",
            &["outcome"])
            .unwrap();

                        pub static ref RATE_LIMIT_REMAINING: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_rate_limit_remaining",
            "Remaining enforcement cap in the current window, per entity_type (-1 if unavailable).",
            &["entity_type"])
            .unwrap();

                        pub static ref CONFIG_RESTART_PENDING: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_config_restart_pending",
            "1 while a restart-requiring config change is pending, by whether it validates.",
            &["valid"])
            .unwrap();

                                                pub static ref LIMITER_COUNTER_RESET_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_limiter_counter_reset_total",
            "Detected Limiter cumulative-counter resets (backwards steps), per entity_type.",
            &["entity_type"])
            .unwrap();


                        pub static ref MANHATTAN_ERRORS_TOTAL: IntCounter =
        register_int_counter!(
            "abuse_enforcement_manhattan_errors_total",
            "Manhattan dedup/allowlist errors (service fails open).")
            .unwrap();


                                                        pub static ref KAFKA_SELF_DELETE_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_kafka_self_delete_total",
            "Self-inflicted pod deletions due to unreachable Kafka brokers.",
            &["reason"])
            .unwrap();


                                                pub static ref KAFKA_CONSUMER_START_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_kafka_consumer_start_total",
            "Per-topic consumer startup outcomes at boot.",
            &["topic", "cluster", "result"])
            .unwrap();


                            pub static ref RULES_YAML_COMPILED: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_rules_yaml_compiled_total",
            "GrowthBook rules-YAML compile attempts, by entity_type and result (success keeps/updates last-good; fail keeps last-good).",
            &["entity_type", "result"])
            .unwrap();


        pub static ref HTTP_LATENCY: HistogramVec =
        register_histogram_vec!(
            "abuse_enforcement_http_latency_seconds",
            "HTTP handler latency.",
            &["endpoint", "status"],
            exponential_buckets(0.01, 2.0, 12).unwrap())
            .unwrap();

        pub static ref HTTP_INFLIGHT: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_http_inflight",
            "Number of HTTP requests currently being processed.",
            &["endpoint"])
            .unwrap();


                                pub static ref KAFKA_PUBLISH_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_kafka_publish_total",
            "Records published to a Kafka publish sink, by sink and produce result.",
            &["sink", "result"])
            .unwrap();


                                                                                                                                            pub static ref HOLD_GATE_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_hold_gate_total",
            "Overturn-hold gate decisions on suspend- or gated-label-shaped rows, by mode, outcome and topic.",
            &["mode", "outcome", "topic"])
            .unwrap();

                pub static ref HOLD_GATE_BREAKER_SKIP_TOTAL: IntCounter =
        register_int_counter!(
            "abuse_enforcement_hold_gate_breaker_skip_total",
            "Overturn-hold gate probes skipped because the circuit breaker was open.")
            .unwrap();

                                            pub static ref HOLD_GATE_BACKOFF_SKIP_TOTAL: IntCounter =
        register_int_counter!(
            "abuse_enforcement_hold_gate_backoff_skip_total",
            "Overturn-hold gate probes skipped because the reachable-error backoff was armed (ledger answers but cannot serve).")
            .unwrap();

                                        pub static ref HOLD_GATE_PROBE_SECONDS: Histogram =
        register_histogram!(
            "abuse_enforcement_hold_gate_probe_seconds",
            "Overturn-hold gate ledger probe latency.",
            vec![0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0])
            .unwrap();

                                            pub static ref HOLD_GATE_PROBE_SCHED_DELAY_SECONDS: Histogram =
        register_histogram!(
            "abuse_enforcement_hold_gate_probe_sched_delay_seconds",
            "Overturn-hold gate: delay between a probe finishing and the decision task observing it.",
            vec![0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0])
            .unwrap();

                            pub static ref HOLD_GATE_MODE: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_hold_gate_mode",
            "Overturn-hold gate mode in effect on this pod (1 for the current mode, 0 otherwise).",
            &["mode"])
            .unwrap();

                        pub static ref HOLD_GATE_CONFIG_INFO: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_hold_gate_config_info",
            "Overturn-hold gate effective config (1 for the current mode/on_probe_error/kinds triple).",
            &["mode", "on_probe_error", "kinds"])
            .unwrap();

                                    pub static ref HOLD_GATE_DB_CONNECTED: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_hold_gate_db_connected",
            "Overturn-hold gate: ledger service reachable (1 = the last request got an HTTP response).")
            .unwrap();

                                        pub static ref HOLD_GATE_EXPIRED_HOLD_TOTAL: IntCounter =
        register_int_counter!(
            "abuse_enforcement_hold_gate_expired_hold_total",
            "Overturn-hold gate: holds returned by the ledger service already past expires_at (dropped).")
            .unwrap();

                            pub static ref HOLD_GATE_BREAKER_OPEN: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_hold_gate_breaker_open",
            "Overturn-hold gate circuit breaker state (1 = open, probes skipped).")
            .unwrap();


                    pub static ref OFF_SWITCH_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_off_switch_total",
            "Decisions skipped because an enforcement off switch matched.",
            &["topic", "head", "scope"])
            .unwrap();

                pub static ref OFF_SWITCH_SNAPSHOT_AGE_SECONDS: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_off_switch_snapshot_age_seconds",
            "Seconds since the last successful off-switch list read.")
            .unwrap();

                pub static ref OFF_SWITCH_ENTRIES: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_off_switch_entries",
            "Off-switch records in this pod's in-memory list.",
            &["state"])
            .unwrap();

                pub static ref OFF_SWITCH_REFRESH_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_off_switch_refresh_total",
            "Off-switch list reads by result.",
            &["result"])
            .unwrap();

            pub static ref OFF_SWITCH_BAD_RECORDS_TOTAL: IntCounter =
        register_int_counter!(
            "abuse_enforcement_off_switch_bad_records_total",
            "Off-switch records that failed to parse.")
            .unwrap();

            pub static ref OFF_SWITCH_ENABLED: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_off_switch_enabled",
            "Whether the decision path honours enforcement off switches (1 = yes).")
            .unwrap();

                    pub static ref OFF_SWITCH_FIRST_LOAD_PENDING: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_off_switch_first_load_pending",
            "1 while the pod waits for its first off-switch list read (not ready, not consuming).")
            .unwrap();

                    pub static ref OFF_SWITCH_HEADLESS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_off_switch_headless_total",
            "Decisions with no driving head, which no off switch can match.",
            &["topic", "model"])
            .unwrap();

                        pub static ref OFF_SWITCH_SEEN_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_off_switch_seen_total",
            "Seen head + classifier pair records by result.",
            &["result"])
            .unwrap();


                                                pub static ref HEAD_GUARD_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_head_guard_total",
            "Head-guard events by head, scope, window and result.",
            &["head", "scope", "window", "result"])
            .unwrap();

            pub static ref ACTION_CONTROL_LOADED: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_action_control_loaded",
            "Whether this pod has read the action-control records (1 = yes).")
            .unwrap();

                pub static ref ACTION_CONTROL_SNAPSHOT_AGE_SECONDS: IntGauge =
        register_int_gauge!(
            "abuse_enforcement_action_control_snapshot_age_seconds",
            "Seconds since the last successful action-control read.")
            .unwrap();

        pub static ref ACTION_CONTROL_ENTRIES: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_action_control_entries",
            "Action-control records in this pod's list by mode.",
            &["mode"])
            .unwrap();

            pub static ref ACTION_CONTROL_REFRESH_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_action_control_refresh_total",
            "Action-control record reads by result.",
            &["result"])
            .unwrap();

            pub static ref ACTION_CONTROL_UNREADABLE_TOTAL: IntCounter =
        register_int_counter!(
            "abuse_enforcement_action_control_unreadable_total",
            "Action-control records that failed to parse.")
            .unwrap();

            pub static ref ACTION_CONTROL_DECISIONS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_action_control_decisions_total",
            "Decisions an action-control record matched, by effective mode.",
            &["head", "mode", "forced"])
            .unwrap();

                pub static ref ACTION_CONTROL_ROLLOUT_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_action_control_rollout_total",
            "Decisions a live row's rollout placed, by head, row and side (inside, outside).",
            &["head", "row", "side"])
            .unwrap();

            pub static ref ACTION_CONTROL_SAFETY_CHECKS_OK: IntGaugeVec =
        register_int_gauge_vec!(
            "abuse_enforcement_action_control_safety_checks_ok",
            "Whether the live rules' safety skips match the service's list (1 = yes).",
            &["entity_type"])
            .unwrap();

            pub static ref ACTION_CONTROL_SEEN_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "abuse_enforcement_action_control_seen_total",
            "Action-control seen records by result.",
            &["result"])
            .unwrap();
}

pub fn observe_user_active_seconds(
    raw: &str,
    entity_type: crate::facts::EntityType,
    status: &str,
    action_dims: &[(&str, &str)],
) {
    if action_dims.is_empty() {
        return;
    }
    let Some(body) = crate::facts::UasBody::parse(raw) else {
        return;
    };
    let et = entity_type.as_str();
    let rows: Vec<_> = body
        .activities()
        .filter(|(_, a)| a.active_seconds.is_finite())
        .collect();
    if rows.is_empty() {
        return;
    }
    for &(action, label) in action_dims {
        for &(period, activity) in &rows {
            USER_ACTIVE_SECONDS
                .with_label_values(&[
                    period,
                    activity.client_app_id.as_str(),
                    et,
                    status,
                    action,
                    label,
                ])
                .observe(activity.active_seconds);
        }
    }
}

pub fn init() {
    let _ = &*ENFORCEMENT_TOTAL;
    let _ = &*FIRED_HEADS_TOTAL;
    let _ = &*ACTION_DISPATCH_TOTAL;
    let _ = &*DEDUP_CLASS_BYPASS_TOTAL;
    let _ = &*DEDUP_WRITE_FENCED_TOTAL;
    let _ = &*DEDUP_DISPLACED_RESTORED_TOTAL;
    let _ = &*DEDUP_BYPASS_CLAIM_LOST_TOTAL;
    let _ = &*POST_LABEL_READ_TOTAL;
    let _ = &*STRATO_LATENCY;
    let _ = &*AIS_LATENCY;
    let _ = &*KAFKA_MESSAGE_LATENCY;
    let _ = &*USER_ACTIVE_SECONDS;
    let _ = &*GIZMODUCK_CHECK_TOTAL;
    let _ = &*GIZMODUCK_LATENCY;
    let _ = &*STRATO_RETRIES;
    let _ = &*RETRY_ATTEMPTS;
    let _ = &*RETRY_QUEUE_SIZE;
    let _ = &*LIMITER_DECISIONS_TOTAL;
    let _ = &*RATE_LIMIT_REMAINING;
    let _ = &*CONFIG_RESTART_PENDING;
    let _ = &*LIMITER_COUNTER_RESET_TOTAL;
    let _ = &*MANHATTAN_ERRORS_TOTAL;
    let _ = &*KAFKA_SELF_DELETE_TOTAL;
    let _ = &*RULES_YAML_COMPILED;
    let _ = &*HTTP_LATENCY;
    let _ = &*HTTP_INFLIGHT;
    let _ = &*KAFKA_PUBLISH_TOTAL;
    let _ = &*HOLD_GATE_TOTAL;
    let _ = &*HOLD_GATE_BREAKER_SKIP_TOTAL;
    let _ = &*HOLD_GATE_BACKOFF_SKIP_TOTAL;
    let _ = &*HOLD_GATE_PROBE_SECONDS;
    let _ = &*HOLD_GATE_PROBE_SCHED_DELAY_SECONDS;
    let _ = &*HOLD_GATE_DB_CONNECTED;
    let _ = &*HOLD_GATE_EXPIRED_HOLD_TOTAL;
    let _ = &*HOLD_GATE_BREAKER_OPEN;
    let _ = &*HOLD_GATE_MODE;
    let _ = &*HOLD_GATE_CONFIG_INFO;
    let _ = &*ACTION_CONTROL_LOADED;
    let _ = &*ACTION_CONTROL_SNAPSHOT_AGE_SECONDS;
    let _ = &*ACTION_CONTROL_ENTRIES;
    let _ = &*ACTION_CONTROL_REFRESH_TOTAL;
    let _ = &*ACTION_CONTROL_UNREADABLE_TOTAL;
    let _ = &*ACTION_CONTROL_DECISIONS_TOTAL;
    let _ = &*ACTION_CONTROL_ROLLOUT_TOTAL;
    let _ = &*ACTION_CONTROL_SAFETY_CHECKS_OK;
    let _ = &*ACTION_CONTROL_SEEN_TOTAL;
    let _ = &*HEAD_GUARD_TOTAL;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::EntityType;

    #[test]
    fn observe_user_active_seconds_records_each_row() {
        let raw = r#"{
          "userId": "1",
          "lastDay": [
            {"clientAppId": "3033300", "activeSeconds": 12.5}
          ],
          "lastWeek": [
            {"clientAppId": "3033300", "activeSeconds": 80.0}
          ],
          "lastMonth": [
            {"clientAppId": "3033300", "activeSeconds": 392.06},
            {"clientAppId": "129032", "activeSeconds": 1768.37}
          ],
          "lastYear": [
            {"clientAppId": "3033300", "activeSeconds": 10.0}
          ]
        }"#;

        let h_day = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastDay",
            "3033300",
            "user",
            "success",
            "addLabelsV2",
            "SpamHighRecall",
        ]);
        let h_week = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastWeek",
            "3033300",
            "user",
            "success",
            "addLabelsV2",
            "SpamHighRecall",
        ]);
        let h_month_a = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastMonth",
            "3033300",
            "user",
            "success",
            "addLabelsV2",
            "SpamHighRecall",
        ]);
        let h_month_b = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastMonth",
            "129032",
            "user",
            "success",
            "addLabelsV2",
            "SpamHighRecall",
        ]);
        let h_year_a = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastYear",
            "3033300",
            "user",
            "success",
            "addLabelsV2",
            "SpamHighRecall",
        ]);
        let before_d = h_day.get_sample_count();
        let before_w = h_week.get_sample_count();
        let before_a = h_month_a.get_sample_count();
        let before_b = h_month_b.get_sample_count();
        let before_y = h_year_a.get_sample_count();
        let sum_before_a = h_month_a.get_sample_sum();

        observe_user_active_seconds(
            raw,
            EntityType::User,
            "success",
            &[("addLabelsV2", "SpamHighRecall")],
        );

        assert_eq!(h_day.get_sample_count() - before_d, 1);
        assert_eq!(h_week.get_sample_count() - before_w, 1);
        assert_eq!(h_month_a.get_sample_count() - before_a, 1);
        assert_eq!(h_month_b.get_sample_count() - before_b, 1);
        assert_eq!(h_year_a.get_sample_count() - before_y, 1);
        assert!((h_month_a.get_sample_sum() - sum_before_a - 392.06).abs() < 1e-6);

        let h_dry = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastMonth",
            "3033300",
            "user",
            "dry_run",
            "addLabelsV2",
            "SpamHighRecall",
        ]);
        let before_dry = h_dry.get_sample_count();
        observe_user_active_seconds(
            raw,
            EntityType::User,
            "dry_run",
            &[("addLabelsV2", "SpamHighRecall")],
        );
        assert_eq!(h_dry.get_sample_count() - before_dry, 1);
        assert_eq!(
            h_month_a.get_sample_count() - before_a,
            1,
            "success series must not receive the dry_run observation"
        );

        let h_bounce = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastMonth",
            "3033300",
            "user",
            "success",
            "bounceCaptcha",
            "",
        ]);
        let before_bounce = h_bounce.get_sample_count();
        let before_label_again = h_month_a.get_sample_count();
        observe_user_active_seconds(
            raw,
            EntityType::User,
            "success",
            &[("addLabelsV2", "SpamHighRecall"), ("bounceCaptcha", "")],
        );
        assert_eq!(h_bounce.get_sample_count() - before_bounce, 1);
        assert_eq!(
            h_month_a.get_sample_count() - before_label_again,
            1,
            "fan-out must hit label series once more alongside bounce"
        );

        let before_empty = h_month_a.get_sample_count();
        observe_user_active_seconds(raw, EntityType::User, "success", &[]);
        assert_eq!(h_month_a.get_sample_count(), before_empty);
    }

    #[test]
    fn observe_user_active_seconds_skips_bad_json_and_non_finite() {
        let h = USER_ACTIVE_SECONDS.with_label_values(&[
            "lastMonth",
            "x",
            "post",
            "success",
            "addPostLabelsV2",
            "SomeLabel",
        ]);
        let before = h.get_sample_count();
        let dims = [("addPostLabelsV2", "SomeLabel")];

        observe_user_active_seconds("not-json", EntityType::Post, "success", &dims);
        observe_user_active_seconds(
            r#"{"lastMonth":[{"clientAppId":"x","activeSeconds":"nan"}]}"#,
            EntityType::Post,
            "success",
            &dims,
        );
        observe_user_active_seconds(
            r#"{"lastMonth":[{"clientAppId":"x","activeSeconds":null}]}"#,
            EntityType::Post,
            "success",
            &dims,
        );
        assert_eq!(
            h.get_sample_count(),
            before,
            "garbage / unparsable must not observe"
        );

        observe_user_active_seconds(
            r#"{"lastMonth":[{"clientAppId":"x","activeSeconds":1e999}]}"#,
            EntityType::Post,
            "success",
            &dims,
        );
        assert_eq!(
            h.get_sample_count(),
            before,
            "non-finite activeSeconds must be skipped"
        );
    }
}
