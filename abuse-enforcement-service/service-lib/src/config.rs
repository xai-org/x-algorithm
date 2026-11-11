use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_PKG_NAME"),
    about = "Abuse enforcement HTTP service backed by AIS via ThriftMux"
)]
pub struct Config {
    #[arg(long, default_value_t = 9090, env = "PORT")]
    pub port: u16,

    #[arg(
        long,
        default_value = "/s/action-intake-service/action-intake-service:thrift",
        env = "AIS_WILY_PATH"
    )]
    pub ais_wily_path: String,

    #[arg(
        long,
        default_value = "action-intake-service.action-intake-service.prod.atla.s2s.twttr.net",
        env = "AIS_TLS_SERVER_NAME"
    )]
    pub ais_tls_server_name: String,

    #[arg(long, default_value = "atla", env = "AIS_ZONE")]
    pub ais_zone: String,

    #[arg(long, env = "STRATO_CA_CERT_PATH")]
    pub strato_ca_cert_path: Option<String>,

    #[arg(long, env = "STRATO_CLIENT_CERT_PATH")]
    pub strato_client_cert_path: Option<String>,

    #[arg(long, env = "STRATO_CLIENT_KEY_PATH")]
    pub strato_client_key_path: Option<String>,

    #[arg(long, env = "SOCKS_PROXY")]
    pub socks_proxy: Option<String>,

    #[arg(long, default_value_t = false, env = "KAFKA_CONSUMER_ENABLED")]
    pub kafka_consumer_enabled: bool,

    #[arg(long, default_value = "phoenix", env = "KAFKA_CONSUMER_MTLS_CLUSTER")]
    pub kafka_consumer_mtls_cluster: String,

    #[arg(long, default_value = "atla", env = "KAFKA_CONSUMER_MTLS_ZONE")]
    pub kafka_consumer_mtls_zone: String,

    #[arg(
        long,
        default_value = "/usr/local/etc/resources/topic_labels.json",
        env = "TOPIC_LABELS_CONFIG"
    )]
    pub topic_labels_config: String,

    #[arg(long, env = "BACKFILL_TOPIC")]
    pub backfill_topic: Option<String>,

    #[arg(long, env = "BACKFILL_CLAIMS_PG_DSN", hide_env_values = true)]
    pub backfill_claims_pg_dsn: Option<String>,

    #[arg(long, env = "BACKFILL_PAGE_KEYS", value_delimiter = ',')]
    pub backfill_page_keys: Vec<String>,

    #[arg(long, env = "BACKFILL_SENDER_KEYS", value_delimiter = ',')]
    pub backfill_sender_keys: Vec<String>,

    #[arg(long, default_value_t = 0, env = "BACKFILL_MAX_RATE_PER_SEC")]
    pub backfill_max_rate_per_sec: u32,

    #[arg(long, env = "BACKFILL_SERVICE_CONFIG_JSON")]
    pub backfill_service_config_json: Option<String>,

    #[arg(long, env = "KAFKA_CONSUMER_GROUP_ID")]
    pub kafka_consumer_group_id: Option<String>,

    #[arg(long, default_value_t = 256, env = "KAFKA_MAX_MESSAGES_PER_POLL")]
    pub kafka_max_messages_per_poll: usize,

    #[arg(long, default_value_t = 64, env = "KAFKA_MAX_IN_FLIGHT")]
    pub kafka_max_in_flight: usize,

    #[arg(
        long,
        default_value = "mltraining",
        env = "KAFKA_PRODUCER_MTLS_CLUSTER"
    )]
    pub kafka_producer_mtls_cluster: String,

    #[arg(long, default_value = "atla", env = "KAFKA_PRODUCER_MTLS_ZONE")]
    pub kafka_producer_mtls_zone: String,

    #[arg(long, default_value_t = true, env = "KAFKA_SELF_DELETE_ENABLED")]
    pub kafka_self_delete_enabled: bool,

    #[arg(long, default_value_t = 15, env = "KAFKA_WATCHDOG_INTERVAL_SECS")]
    pub kafka_watchdog_interval_secs: u64,

    #[arg(long, default_value_t = 120, env = "KAFKA_WATCHDOG_STALE_SECS")]
    pub kafka_watchdog_stale_secs: u64,

    #[arg(long, default_value_t = 240, env = "KAFKA_WATCHDOG_SELF_DELETE_SECS")]
    pub kafka_watchdog_self_delete_secs: u64,

    #[arg(long, default_value_t = 2.0, env = "KAFKA_WATCHDOG_ERROR_RATE_PER_SEC")]
    pub kafka_watchdog_error_rate_per_sec: f64,

    #[arg(
        long,
        default_value_t = 60,
        env = "KAFKA_WATCHDOG_ERROR_SELF_DELETE_SECS"
    )]
    pub kafka_watchdog_error_self_delete_secs: u64,

    #[arg(long, default_value = "atla", env = "GIZMODUCK_ZONE")]
    pub gizmoduck_zone: String,

    #[arg(
        long,
        default_value = "xai-abuse-enforcement-service.prod",
        env = "GIZMODUCK_CLIENT_ID"
    )]
    pub gizmoduck_client_id: String,

    #[arg(long, default_value = "atla", env = "POST_LABELS_ZONE")]
    pub post_labels_zone: String,

    #[arg(
        long,
        default_value = "platform-manipulation/isHighPageRankV2.User",
        env = "HIGH_PAGE_RANK_COLUMN"
    )]
    pub high_page_rank_column: String,

    #[arg(
        long,
        default_value = "platform-manipulation/isGreyBadge.User",
        env = "GREY_BADGE_COLUMN"
    )]
    pub grey_badge_column: String,

    #[arg(
        long,
        default_value = "user_activity/uas_by_client.User",
        env = "UAS_COLUMN"
    )]
    pub uas_column: String,

    #[arg(long, env = "GROWTHBOOK_URL")]
    pub growthbook_url: Option<String>,

    #[arg(long, env = "GROWTHBOOK_KEY")]
    pub growthbook_key: Option<String>,

    #[arg(
        long,
        default_value = "https://api.growthbook.io",
        env = "GROWTHBOOK_ADMIN_API_HOST"
    )]
    pub growthbook_admin_api_host: String,

    #[arg(long, env = "GROWTHBOOK_ADMIN_API_KEY")]
    pub growthbook_admin_api_key: Option<String>,

    #[arg(
        long,
        default_value = "xai_abuse_enforcement_service_config",
        env = "GROWTHBOOK_FEATURE_KEY"
    )]
    pub growthbook_feature_key: String,

    #[arg(long, default_value = "production", env = "GROWTHBOOK_ENVIRONMENT")]
    pub growthbook_environment: String,

    #[arg(long, default_value = "omega", env = "MH_CLUSTER")]
    pub mh_cluster: String,

    #[arg(
        long,
        default_value = "xai_abuse_enforcement_service",
        env = "MH_APP_ID"
    )]
    pub mh_app_id: String,

    #[arg(
        long,
        default_value = "enforcement_service_staging",
        env = "MH_DATASET"
    )]
    pub mh_dataset: String,

    #[arg(long, default_value = "atla", env = "MH_DATACENTER")]
    pub mh_datacenter: String,

    #[arg(long, env = "LIMITER_DATACENTER")]
    pub limiter_datacenter: Option<String>,

    #[arg(
        long,
        default_value = "xai-abuse-enforcement-service-enforcements-staging",
        env = "LIMITER_FEATURE"
    )]
    pub limiter_feature: String,

    #[arg(
        long,
        default_value = "xai_abuse_enforcement_service",
        env = "LIMITER_CLIENT_ID"
    )]
    pub limiter_client_id: String,

    #[arg(long, default_value_t = true, env = "LIMITER_FAIL_OPEN")]
    pub limiter_fail_open: bool,

    #[arg(long, env = "LIMITER_CA_CERT_PATH")]
    pub limiter_ca_cert_path: Option<String>,

    #[arg(long, env = "LIMITER_CLIENT_CERT_PATH")]
    pub limiter_client_cert_path: Option<String>,

    #[arg(long, env = "LIMITER_CLIENT_KEY_PATH")]
    pub limiter_client_key_path: Option<String>,

    #[arg(long)]
    pub dry_run: bool,

    #[arg(long, default_value_t = 500_000, env = "MAX_ENFORCEMENTS_PER_DAY")]
    pub max_enforcements_per_day: u32,

    #[arg(long, default_value_t = 500_000, env = "MAX_POST_ENFORCEMENTS_PER_DAY")]
    pub max_post_enforcements_per_day: u32,

    #[arg(long, env = "OVERTURN_HOLD_LEDGER_URL")]
    pub overturn_hold_ledger_url: Option<String>,

    #[arg(long, env = "OVERTURN_HOLD_STARTUP_PROBE")]
    pub overturn_hold_startup_probe: Option<String>,

    #[arg(long, env = "OVERTURN_HOLD_ENV")]
    pub overturn_hold_env: Option<String>,

    #[arg(long, default_value_t = 86400, env = "DEDUP_TTL_SECS")]
    pub dedup_ttl_secs: u64,

    #[arg(long, default_value_t = 900, env = "DEDUP_SKIP_TTL_SECS")]
    pub dedup_skip_ttl_secs: u64,

    #[arg(long, default_value_t = 5, env = "DRAIN_PERIOD_SECS")]
    pub drain_period_secs: u64,

    #[arg(long, env = "API_KEYS_JSON")]
    pub api_keys_json: Option<String>,
}

impl Config {
    pub fn kafka_group_id(&self) -> String {
        self.kafka_consumer_group_id
            .clone()
            .unwrap_or_else(|| env!("CARGO_PKG_NAME").to_string())
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Config;

    #[test]
    fn kafka_defaults_preserve_consumer_and_watchdog_behavior() {
        let config = Config::parse_from(["xai-abuse-enforcement-service"]);

        assert!(!config.kafka_consumer_enabled);
        assert_eq!(config.backfill_topic, None);
        assert_eq!(config.kafka_consumer_mtls_cluster, "phoenix");
        assert_eq!(config.kafka_consumer_mtls_zone, "atla");
        assert_eq!(config.kafka_group_id(), "xai-abuse-enforcement-service");
        assert_eq!(config.kafka_max_messages_per_poll, 256);
        assert_eq!(config.kafka_max_in_flight, 64);
        assert!(config.kafka_self_delete_enabled);
        assert_eq!(config.kafka_watchdog_interval_secs, 15);
        assert_eq!(config.kafka_watchdog_stale_secs, 120);
        assert_eq!(config.kafka_watchdog_self_delete_secs, 240);
        assert_eq!(config.kafka_watchdog_error_rate_per_sec, 2.0);
        assert_eq!(config.kafka_watchdog_error_self_delete_secs, 60);
    }

    #[test]
    fn overturn_hold_ledger_url_defaults_unset() {
        let config = Config::parse_from(["xai-abuse-enforcement-service"]);
        assert_eq!(config.overturn_hold_ledger_url, None);
        let config = Config::parse_from([
            "xai-abuse-enforcement-service",
            "--overturn-hold-ledger-url=http://ledger.example.invalid:8080",
        ]);
        assert_eq!(
            config.overturn_hold_ledger_url.as_deref(),
            Some("http://ledger.example.invalid:8080")
        );
        assert!(format!("{config:?}").contains("http://ledger.example.invalid:8080"));
    }

    #[test]
    fn overturn_hold_startup_probe_defaults_off() {
        let config = Config::parse_from(["xai-abuse-enforcement-service"]);
        assert_eq!(config.overturn_hold_startup_probe, None);
        assert!(!crate::overturn_hold::startup_probe_requested(
            config.overturn_hold_startup_probe.as_deref()
        ));
        let config = Config::parse_from([
            "xai-abuse-enforcement-service",
            "--overturn-hold-startup-probe=1",
        ]);
        assert!(crate::overturn_hold::startup_probe_requested(
            config.overturn_hold_startup_probe.as_deref()
        ));
        let config = Config::parse_from([
            "xai-abuse-enforcement-service",
            "--overturn-hold-startup-probe=0",
        ]);
        assert!(!crate::overturn_hold::startup_probe_requested(
            config.overturn_hold_startup_probe.as_deref()
        ));
    }

    #[test]
    fn mltraining_producer_defaults_use_mtls() {
        let config = Config::parse_from(["xai-abuse-enforcement-service"]);

        assert_eq!(config.kafka_producer_mtls_cluster, "mltraining");
        assert_eq!(config.kafka_producer_mtls_zone, "atla");
    }
}
