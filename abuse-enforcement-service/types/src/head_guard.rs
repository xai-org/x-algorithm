use serde::{Deserialize, Serialize};

use crate::action_control::UnreadableRecord;

pub const HEAD_GUARD_RECORD_VERSION: u32 = 1;

pub const DEFAULT_MIN_DECIDED_APPEALS: u32 = 30;

pub const DEFAULT_OVERTURN_WINDOW_HOURS: u32 = 72;

pub const MAX_OVERTURN_WINDOW_HOURS: u32 = 168;

fn default_min_decided_appeals() -> u32 {
    DEFAULT_MIN_DECIDED_APPEALS
}

fn default_window_hours() -> u32 {
    DEFAULT_OVERTURN_WINDOW_HOURS
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OverturnGuard {
    pub max_overturn_rate: f64,
    #[serde(default = "default_min_decided_appeals")]
    pub min_decided_appeals: u32,
    #[serde(default = "default_window_hours")]
    pub window_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct HeadGuardRecord {
    #[serde(default)]
    pub v: u32,
    pub head: String,
    #[serde(default)]
    pub classifier: Option<String>,
    #[serde(default)]
    pub max_actions_per_hour: Option<u32>,
    #[serde(default)]
    pub max_actions_per_day: Option<u32>,
    #[serde(default)]
    pub overturn: Option<OverturnGuard>,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub updated_by: String,
    #[serde(default)]
    pub key_id: String,
    #[serde(default)]
    pub updated_at_ms: i64,
    pub version: u64,
    #[serde(default)]
    pub write_nonce: u64,
}

impl HeadGuardRecord {
    pub fn scope(&self) -> &'static str {
        if self.classifier.is_some() {
            "pair"
        } else {
            "head"
        }
    }

    pub fn is_active(&self) -> bool {
        self.max_actions_per_hour.is_some()
            || self.max_actions_per_day.is_some()
            || self.overturn.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct HeadGuardPutRequest {
    pub head: String,
    #[serde(default)]
    pub classifier: Option<String>,
    #[serde(default)]
    pub max_actions_per_hour: Option<u32>,
    #[serde(default)]
    pub max_actions_per_day: Option<u32>,
    #[serde(default, deserialize_with = "strict_overturn")]
    pub overturn: Option<OverturnGuard>,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub expected_version: Option<u64>,
    #[serde(default)]
    pub updated_by: Option<String>,
}

fn strict_overturn<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<OverturnGuard>, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Strict {
        max_overturn_rate: f64,
        #[serde(default = "default_min_decided_appeals")]
        min_decided_appeals: u32,
        #[serde(default = "default_window_hours")]
        window_hours: u32,
    }
    Ok(Option::<Strict>::deserialize(d)?.map(|s| OverturnGuard {
        max_overturn_rate: s.max_overturn_rate,
        min_decided_appeals: s.min_decided_appeals,
        window_hours: s.window_hours,
    }))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct HeadGuardView {
    #[serde(flatten)]
    pub record: HeadGuardRecord,
    pub scope: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct HeadGuardPutResponse {
    pub guard: HeadGuardView,
    pub before: Option<HeadGuardRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct HeadGuardListResponse {
    pub guards: Vec<HeadGuardView>,
    #[serde(default)]
    pub unreadable: Vec<UnreadableRecord>,
    pub source: String,
    pub snapshot_age_ms: i64,
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_parses_minimal_and_applies_overturn_defaults() {
        let r: HeadGuardRecord = serde_json::from_str(
            r#"{"head":"H","version":1,"overturn":{"max_overturn_rate":0.2},"future_field":1}"#,
        )
        .unwrap();
        assert_eq!(r.classifier, None);
        assert_eq!(r.scope(), "head");
        assert!(r.is_active());
        assert_eq!(
            r.overturn,
            Some(OverturnGuard {
                max_overturn_rate: 0.2,
                min_decided_appeals: 30,
                window_hours: 72,
            })
        );
    }

    #[test]
    fn a_record_with_no_limits_is_inactive() {
        let r: HeadGuardRecord =
            serde_json::from_str(r#"{"head":"H","classifier":"c@1","version":3}"#).unwrap();
        assert_eq!(r.scope(), "pair");
        assert!(!r.is_active());
    }

    #[test]
    fn put_request_rejects_unknown_fields() {
        assert!(serde_json::from_str::<HeadGuardPutRequest>(
            r#"{"head":"H","reason":"r","max_actions_per_hr":5}"#
        )
        .is_err());
        assert!(serde_json::from_str::<HeadGuardPutRequest>(
            r#"{"head":"H","reason":"r","overturn":{"max_overturn_rate":0.1,"window":3}}"#
        )
        .is_err());
        let r: HeadGuardPutRequest = serde_json::from_str(
            r#"{"head":"H","reason":"r","overturn":{"max_overturn_rate":0.1}}"#,
        )
        .unwrap();
        assert_eq!(r.overturn.unwrap().window_hours, 72, "defaults apply");
        let r: HeadGuardPutRequest =
            serde_json::from_str(r#"{"head":"H","reason":"r","overturn":null}"#).unwrap();
        assert_eq!(r.overturn, None);
    }

    #[test]
    fn a_stored_overturn_ignores_unknown_keys() {
        let r: HeadGuardRecord = serde_json::from_str(
            r#"{"head":"H","version":1,"overturn":{"max_overturn_rate":0.2,"future_field":1}}"#,
        )
        .unwrap();
        assert_eq!(r.overturn.unwrap().max_overturn_rate, 0.2);
    }

    #[test]
    fn view_flattens_record() {
        let v = HeadGuardView {
            record: serde_json::from_str(r#"{"head":"H","version":2,"max_actions_per_day":100}"#)
                .unwrap(),
            scope: "head".into(),
            active: true,
        };
        let j = serde_json::to_value(&v).unwrap();
        assert_eq!(j["head"], "H");
        assert_eq!(j["classifier"], serde_json::Value::Null);
        assert_eq!(j["max_actions_per_day"], 100);
        assert_eq!(j["max_actions_per_hour"], serde_json::Value::Null);
        assert_eq!(j["overturn"], serde_json::Value::Null);
        assert_eq!(j["version"], 2);
        assert_eq!(j["active"], true);
    }
}
