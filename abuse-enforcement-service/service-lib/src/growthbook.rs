use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use growthbook_rust_sdk::client::{GrowthBookClient, GrowthBookClientTrait};
use serde::Deserialize;
use serde_json::Value;
use tracing::{info, warn};

use crate::facts::EntityType;
use crate::sliding_window::SlidingWindowLimiter;

const GROWTHBOOK_CONFIG_KEY: &str = "xai_abuse_enforcement_service_config";

const MAX_USER_ENFORCEMENTS_KEY: &str = "max_user_enforcements_per_day";

const MAX_ENFORCEMENTS_KEY: &str = "max_enforcements_per_day";

const MAX_POST_ENFORCEMENTS_KEY: &str = "max_post_enforcements_per_day";

const TOPIC_LABELS_KEY: &str = "topic_labels";

const KAFKA_KEY: &str = "kafka";
const CONSUMER_KEY: &str = "consumer";
const PRODUCER_KEY: &str = "producer";

const GROWTHBOOK_RULES_USER_KEY: &str = "xai_abuse_enforcement_service_rule_user";
const GROWTHBOOK_RULES_POST_KEY: &str = "xai_abuse_enforcement_service_rule_post";

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProducerSpec {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub topic: Option<String>,
    #[serde(default)]
    pub cluster: Option<String>,
    #[serde(default)]
    pub zone: Option<String>,
}

#[derive(Clone)]
struct EntityCap {
    sliding: Option<SlidingWindowLimiter>,
    default_max: u32,
    gb_keys: &'static [&'static str],
}

#[derive(Clone)]
pub struct DynamicConfig {
    client: Option<Arc<GrowthBookClient>>,
    snapshot: Arc<RwLock<Arc<ConfigSnapshot>>>,
    dry_run_default: bool,
    user_cap: EntityCap,
    post_cap: EntityCap,
}

#[derive(Debug, Default)]
pub struct ConfigSnapshot {
    pub config: Option<Value>,
    pub rules_user: Option<Arc<str>>,
    pub rules_post: Option<Arc<str>>,
}

fn next_snapshot(
    prev: &ConfigSnapshot,
    config: Option<Value>,
    rules_user: Option<&str>,
    rules_post: Option<&str>,
) -> ConfigSnapshot {
    fn keep_or_new(prev: &Option<Arc<str>>, next: Option<&str>) -> Option<Arc<str>> {
        let next = next.filter(|s| !s.trim().is_empty())?;
        match prev {
            Some(p) if **p == *next => Some(p.clone()),
            _ => Some(Arc::from(next)),
        }
    }
    ConfigSnapshot {
        config,
        rules_user: keep_or_new(&prev.rules_user, rules_user),
        rules_post: keep_or_new(&prev.rules_post, rules_post),
    }
}

impl DynamicConfig {
    pub async fn new(
        url: Option<&str>,
        key: Option<&str>,
        dry_run_default: bool,
        max_user_enforcements_default: u32,
        max_post_enforcements_default: u32,
        sliding_user: Option<SlidingWindowLimiter>,
        sliding_post: Option<SlidingWindowLimiter>,
    ) -> Result<Self, String> {
        let client = match (url, key) {
            (Some(url), Some(key)) if !key.is_empty() => {
                info!("Initializing GrowthBook client at {url}");
                match GrowthBookClient::new(url, key, None, None).await {
                    Ok(c) => {
                        info!("GrowthBook client initialized successfully");
                        Some(Arc::new(c))
                    }
                    Err(e) => {
                        return Err(format!("Failed to initialize GrowthBook client: {e}"));
                    }
                }
            }
            _ => {
                info!("GrowthBook URL or key not provided — using CLI defaults");
                None
            }
        };

        let dc = Self {
            client,
            snapshot: Arc::default(),
            dry_run_default,
            user_cap: EntityCap {
                sliding: sliding_user,
                default_max: max_user_enforcements_default,
                gb_keys: &[MAX_USER_ENFORCEMENTS_KEY, MAX_ENFORCEMENTS_KEY],
            },
            post_cap: EntityCap {
                sliding: sliding_post,
                default_max: max_post_enforcements_default,
                gb_keys: &[MAX_POST_ENFORCEMENTS_KEY],
            },
        };
        dc.refresh();
        Ok(dc)
    }

    pub fn from_static(
        config: Value,
        dry_run_default: bool,
        max_user_enforcements_default: u32,
        max_post_enforcements_default: u32,
        sliding_user: Option<SlidingWindowLimiter>,
        sliding_post: Option<SlidingWindowLimiter>,
    ) -> Self {
        Self {
            client: None,
            snapshot: Arc::new(RwLock::new(Arc::new(ConfigSnapshot {
                config: Some(config),
                rules_user: None,
                rules_post: None,
            }))),
            dry_run_default,
            user_cap: EntityCap {
                sliding: sliding_user,
                default_max: max_user_enforcements_default,
                gb_keys: &[MAX_USER_ENFORCEMENTS_KEY, MAX_ENFORCEMENTS_KEY],
            },
            post_cap: EntityCap {
                sliding: sliding_post,
                default_max: max_post_enforcements_default,
                gb_keys: &[MAX_POST_ENFORCEMENTS_KEY],
            },
        }
    }

    pub fn reads_growthbook(&self) -> bool {
        self.client.is_some()
    }

    pub fn refresh(&self) {
        let Some(client) = self.client.as_ref() else {
            return;
        };
        let config = Some(client.feature_result(GROWTHBOOK_CONFIG_KEY, None).value);
        let rules_user = client.feature_result(GROWTHBOOK_RULES_USER_KEY, None).value;
        let rules_post = client.feature_result(GROWTHBOOK_RULES_POST_KEY, None).value;
        let next = next_snapshot(
            &self.snapshot(),
            config,
            rules_user.as_str(),
            rules_post.as_str(),
        );
        *self.snapshot.write().unwrap_or_else(|p| p.into_inner()) = Arc::new(next);
    }

    pub fn snapshot(&self) -> Arc<ConfigSnapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn set_snapshot_for_test(&self, snapshot: ConfigSnapshot) {
        *self.snapshot.write().unwrap() = Arc::new(snapshot);
    }

    fn cap_for(&self, entity_type: EntityType) -> &EntityCap {
        match entity_type {
            EntityType::User => &self.user_cap,
            EntityType::Post => &self.post_cap,
        }
    }

    pub fn bool(&self, key: &str, default: bool) -> bool {
        parse_bool(self.snapshot().config.as_ref(), key, default)
    }

    pub fn is_dry_run(&self) -> bool {
        self.bool("dry_run", self.dry_run_default)
    }

    pub async fn try_enforce(&self, entity_type: EntityType) -> bool {
        match self.cap_for(entity_type).sliding.as_ref() {
            Some(sw) => {
                sw.try_record(self.max_enforcements_per_day(entity_type) as i64)
                    .await
            }
            None => true,
        }
    }

    pub fn rate_limit_stats(&self, entity_type: EntityType) -> Option<(u32, u32, u32)> {
        Some(
            self.cap_for(entity_type)
                .sliding
                .as_ref()?
                .stats(self.max_enforcements_per_day(entity_type) as i64),
        )
    }

    pub async fn record_enforcement_report(
        &self,
        entity_type: EntityType,
    ) -> Option<serde_json::Value> {
        self.cap_for(entity_type)
            .sliding
            .as_ref()?
            .record_report(self.max_enforcements_per_day(entity_type) as i64)
            .await
    }

    pub async fn reset_rate_limit(&self, entity_type: EntityType) -> Option<u64> {
        self.cap_for(entity_type).sliding.as_ref()?.reset().await
    }

    pub fn max_enforcements_per_day(&self, entity_type: EntityType) -> u32 {
        let cap = self.cap_for(entity_type);
        pick_u32(
            self.snapshot().config.as_ref(),
            cap.gb_keys,
            cap.default_max,
        )
    }

    pub fn enforcement_rules_yaml(&self, entity_type: EntityType) -> Option<Arc<str>> {
        let snapshot = self.snapshot();
        match entity_type {
            EntityType::User => snapshot.rules_user.clone(),
            EntityType::Post => snapshot.rules_post.clone(),
        }
    }

    pub fn overturn_hold_gate(&self) -> crate::overturn_hold::HoldGateConfig {
        crate::overturn_hold::HoldGateConfig::from_config(self.snapshot().config.as_ref())
    }

    pub fn off_switch_enabled(&self) -> bool {
        crate::off_switch::enabled_from_config(self.snapshot().config.as_ref())
    }

    pub fn action_control_config(&self) -> crate::action_control::ControlConfig {
        crate::action_control::ControlConfig::from_config(self.snapshot().config.as_ref())
    }

    pub fn confidence_config(&self) -> crate::confidence::ConfidenceConfig {
        crate::confidence::ConfidenceConfig::from_config(self.snapshot().config.as_ref())
    }

    pub fn config(&self) -> Option<Value> {
        self.snapshot().config.clone()
    }
}

fn get_path<'a>(config: Option<&'a Value>, path: &[&str]) -> Option<&'a Value> {
    let mut cur = config?;
    for seg in path {
        cur = cur.get(seg)?;
    }
    Some(cur)
}

pub(crate) fn kafka_consumer_topic_labels_from_config(config: Option<&Value>) -> Option<Value> {
    get_path(config, &[KAFKA_KEY, CONSUMER_KEY, TOPIC_LABELS_KEY])
        .or_else(|| config.and_then(|c| c.get(TOPIC_LABELS_KEY)))
        .cloned()
}

pub(crate) fn kafka_producers_from_config(config: Option<&Value>) -> HashMap<String, ProducerSpec> {
    let Some(obj) = get_path(config, &[KAFKA_KEY, PRODUCER_KEY]).and_then(Value::as_object) else {
        return HashMap::new();
    };
    obj.iter()
        .filter_map(
            |(name, v)| match serde_json::from_value::<ProducerSpec>(v.clone()) {
                Ok(spec) => Some((name.clone(), spec)),
                Err(e) => {
                    warn!("kafka producer '{name}' config is malformed; sink disabled: {e}");
                    None
                }
            },
        )
        .collect()
}

fn parse_bool(config: Option<&Value>, key: &str, default: bool) -> bool {
    config
        .and_then(|cfg| cfg.get(key)?.as_bool())
        .unwrap_or(default)
}

fn pick_u32(config: Option<&Value>, keys: &[&str], default: u32) -> u32 {
    keys.iter()
        .find_map(|key| config?.get(*key)?.as_u64())
        .map(|v| v as u32)
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg_without_gb(dry_run_default: bool) -> DynamicConfig {
        DynamicConfig {
            client: None,
            snapshot: Arc::default(),
            dry_run_default,
            user_cap: EntityCap {
                sliding: None,
                default_max: 500_000,
                gb_keys: &[MAX_USER_ENFORCEMENTS_KEY, MAX_ENFORCEMENTS_KEY],
            },
            post_cap: EntityCap {
                sliding: None,
                default_max: 500_000,
                gb_keys: &[MAX_POST_ENFORCEMENTS_KEY],
            },
        }
    }

    #[test]
    fn next_snapshot_reuses_arc_for_unchanged_yaml() {
        let first = next_snapshot(&ConfigSnapshot::default(), None, Some("a: 1"), Some("b: 2"));
        let same = next_snapshot(&first, None, Some("a: 1"), Some("b: 3"));
        assert!(Arc::ptr_eq(
            first.rules_user.as_ref().unwrap(),
            same.rules_user.as_ref().unwrap()
        ));
        assert!(!Arc::ptr_eq(
            first.rules_post.as_ref().unwrap(),
            same.rules_post.as_ref().unwrap()
        ));
        assert_eq!(same.rules_post.as_deref(), Some("b: 3"));
    }

    #[test]
    fn next_snapshot_treats_blank_yaml_as_unset() {
        let s = next_snapshot(&ConfigSnapshot::default(), None, Some("  \n"), None);
        assert!(s.rules_user.is_none());
        assert!(s.rules_post.is_none());
    }

    #[test]
    fn accessors_read_the_snapshot() {
        let dc = cfg_without_gb(false);
        dc.set_snapshot_for_test(ConfigSnapshot {
            config: Some(json!({"dry_run": true, "max_post_enforcements_per_day": 9})),
            rules_user: Some(Arc::from("rules")),
            rules_post: None,
        });
        assert!(dc.is_dry_run());
        assert_eq!(dc.max_enforcements_per_day(EntityType::Post), 9);
        assert_eq!(
            dc.enforcement_rules_yaml(EntityType::User).as_deref(),
            Some("rules")
        );
        assert_eq!(dc.enforcement_rules_yaml(EntityType::Post), None);
        assert!(dc.clone().is_dry_run());
    }

    #[test]
    fn refresh_without_client_keeps_the_snapshot() {
        let dc = cfg_without_gb(false);
        dc.set_snapshot_for_test(ConfigSnapshot {
            config: Some(json!({"dry_run": true})),
            ..Default::default()
        });
        dc.refresh();
        assert!(dc.is_dry_run());
    }

    #[test]
    fn is_dry_run_returns_default_when_no_client() {
        assert!(cfg_without_gb(true).is_dry_run());
        assert!(!cfg_without_gb(false).is_dry_run());
    }

    #[test]
    fn bool_returns_default_when_no_client() {
        let dc = cfg_without_gb(false);
        assert!(dc.bool("anything", true));
        assert!(!dc.bool("anything", false));
    }

    #[test]
    fn overturn_hold_gate_defaults_off_when_no_client() {
        let gate = cfg_without_gb(false).overturn_hold_gate();
        assert_eq!(gate.mode, crate::overturn_hold::GateMode::Off);
        assert!(!gate.enabled());
    }

    #[test]
    fn parse_bool_returns_value_when_present() {
        let cfg = json!({"dry_run": true});
        assert!(parse_bool(Some(&cfg), "dry_run", false));

        let cfg = json!({"dry_run": false});
        assert!(!parse_bool(Some(&cfg), "dry_run", true));
    }

    #[test]
    fn parse_bool_returns_default_when_key_missing() {
        let cfg = json!({"other": 123});
        assert!(parse_bool(Some(&cfg), "dry_run", true));
        assert!(!parse_bool(Some(&cfg), "dry_run", false));
    }

    #[test]
    fn parse_bool_returns_default_when_wrong_type() {
        let cfg = json!({"dry_run": "yes"});
        assert!(parse_bool(Some(&cfg), "dry_run", true));
    }

    #[test]
    fn parse_bool_returns_default_when_config_is_none() {
        assert!(parse_bool(None, "dry_run", true));
        assert!(!parse_bool(None, "dry_run", false));
    }

    #[test]
    fn pick_u32_prefers_first_key_then_falls_back_then_defaults() {
        let user_keys = &[MAX_USER_ENFORCEMENTS_KEY, MAX_ENFORCEMENTS_KEY];

        let cfg = json!({
            "max_user_enforcements_per_day": 200,
            "max_enforcements_per_day": 100,
        });
        assert_eq!(pick_u32(Some(&cfg), user_keys, 7), 200);

        let cfg = json!({ "max_enforcements_per_day": 100 });
        assert_eq!(pick_u32(Some(&cfg), user_keys, 7), 100);

        let cfg = json!({ "other": 1 });
        assert_eq!(pick_u32(Some(&cfg), user_keys, 7), 7);

        assert_eq!(pick_u32(None, user_keys, 7), 7);

        let cfg = json!({ "max_post_enforcements_per_day": 42 });
        assert_eq!(pick_u32(Some(&cfg), &[MAX_POST_ENFORCEMENTS_KEY], 7), 42);
    }

    #[test]
    fn topic_labels_prefers_nested_then_falls_back_to_flat() {
        let cfg = json!({
            "kafka": { "consumer": { "topic_labels": { "t1": {} } } },
            "topic_labels": { "legacy": {} },
        });
        assert_eq!(
            kafka_consumer_topic_labels_from_config(Some(&cfg)),
            Some(json!({ "t1": {} }))
        );

        let cfg = json!({ "topic_labels": { "legacy": {} } });
        assert_eq!(
            kafka_consumer_topic_labels_from_config(Some(&cfg)),
            Some(json!({ "legacy": {} }))
        );

        assert_eq!(
            kafka_consumer_topic_labels_from_config(Some(&json!({ "other": 1 }))),
            None
        );
        assert_eq!(kafka_consumer_topic_labels_from_config(None), None);
    }

    #[test]
    fn kafka_producers_from_config_iterates_and_skips_malformed() {
        let cfg = json!({
            "kafka": { "producer": {
                "decisions": { "enabled": true, "topic": "t1" },
                "audit": { "enabled": false },
                "broken": { "enabled": "yes" },
            } },
        });
        let m = kafka_producers_from_config(Some(&cfg));
        assert_eq!(m.len(), 2);
        assert!(m["decisions"].enabled);
        assert_eq!(m["decisions"].topic.as_deref(), Some("t1"));
        assert!(!m["audit"].enabled);
        assert_eq!(m["audit"].topic, None);
        assert!(!m.contains_key("broken"));

        assert!(kafka_producers_from_config(Some(&json!({ "other": 1 }))).is_empty());
        assert!(kafka_producers_from_config(None).is_empty());
    }

    #[test]
    fn kafka_producers_from_config_carries_optional_cluster_override() {
        let cfg = json!({
            "kafka": { "producer": {
                "decisions": { "enabled": true, "topic": "t1" },
                "elsewhere": {
                    "enabled": true,
                    "topic": "t2",
                    "cluster": "coredata",
                    "zone": "pdxa",
                },
            } },
        });
        let m = kafka_producers_from_config(Some(&cfg));

        assert_eq!(m["decisions"].cluster, None);
        assert_eq!(m["decisions"].zone, None);
        assert_eq!(m["elsewhere"].cluster.as_deref(), Some("coredata"));
        assert_eq!(m["elsewhere"].zone.as_deref(), Some("pdxa"));
    }

    #[tokio::test]
    async fn try_enforce_allows_when_no_counter() {
        let dc = cfg_without_gb(false);
        assert!(dc.try_enforce(EntityType::User).await);
        assert!(dc.try_enforce(EntityType::User).await);
        assert!(dc.try_enforce(EntityType::Post).await);
        assert!(dc.try_enforce(EntityType::Post).await);
    }
}
