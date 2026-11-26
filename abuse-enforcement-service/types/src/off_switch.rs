use serde::{Deserialize, Serialize};

pub const OFF_SWITCH_RECORD_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OffSwitchRecord {
    #[serde(default)]
    pub v: u32,
    pub head: String,
    #[serde(default)]
    pub classifier: Option<String>,
    pub off: bool,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub updated_by: String,
    #[serde(default)]
    pub key_id: String,
    #[serde(default)]
    pub updated_at_ms: i64,
    #[serde(default)]
    pub auto_on_at_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_off: Option<PreviousOff>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PreviousOff {
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub updated_by: String,
    #[serde(default)]
    pub updated_at_ms: i64,
}

impl OffSwitchRecord {
    pub fn scope(&self) -> &'static str {
        if self.classifier.is_some() {
            "pair"
        } else {
            "head"
        }
    }

    pub fn is_active(&self, now_ms: i64) -> bool {
        self.off && self.auto_on_at_ms.is_none_or(|t| now_ms < t)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OffSwitchPutRequest {
    pub head: String,
    #[serde(default)]
    pub classifier: Option<String>,
    pub off: bool,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub auto_on_at_ms: Option<i64>,
    #[serde(default)]
    pub updated_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OffSwitchView {
    #[serde(flatten)]
    pub record: OffSwitchRecord,
    pub scope: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OffSwitchSeen {
    pub head: String,
    pub classifier: String,
    pub last_seen_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OffSwitchListResponse {
    pub switches: Vec<OffSwitchView>,
    #[serde(default)]
    pub seen: Vec<OffSwitchSeen>,
    pub source: String,
    pub snapshot_age_ms: i64,
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seen_error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(off: bool, auto_on_at_ms: Option<i64>) -> OffSwitchRecord {
        OffSwitchRecord {
            v: OFF_SWITCH_RECORD_VERSION,
            head: "H".into(),
            classifier: None,
            off,
            reason: String::new(),
            updated_by: String::new(),
            key_id: String::new(),
            updated_at_ms: 0,
            auto_on_at_ms,
            previous_off: None,
        }
    }

    #[test]
    fn active_honours_off_and_auto_on() {
        assert!(rec(true, None).is_active(1_000));
        assert!(!rec(false, None).is_active(1_000));
        assert!(rec(true, Some(2_000)).is_active(1_999));
        assert!(!rec(true, Some(2_000)).is_active(2_000));
        assert!(!rec(false, Some(2_000)).is_active(1_000));
    }

    #[test]
    fn record_parses_minimal_and_unknown_fields() {
        let r: OffSwitchRecord =
            serde_json::from_str(r#"{"head":"H","off":true,"future_field":1}"#).unwrap();
        assert_eq!(r.classifier, None);
        assert_eq!(r.auto_on_at_ms, None);
        assert_eq!(r.previous_off, None);
        assert_eq!(r.scope(), "head");
    }

    #[test]
    fn previous_off_round_trips_and_is_omitted_when_absent() {
        let mut r = rec(false, None);
        assert!(
            !serde_json::to_string(&r).unwrap().contains("previous_off"),
            "absent stays absent on the wire"
        );
        r.previous_off = Some(PreviousOff {
            reason: "action ceiling: 5 in 1h (guard v1)".into(),
            updated_by: "head-guard".into(),
            updated_at_ms: 7,
        });
        let back: OffSwitchRecord =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
        let lenient: OffSwitchRecord = serde_json::from_str(
            r#"{"head":"H","off":false,"previous_off":{"updated_by":"x","future":1}}"#,
        )
        .unwrap();
        assert_eq!(
            lenient.previous_off,
            Some(PreviousOff {
                reason: String::new(),
                updated_by: "x".into(),
                updated_at_ms: 0,
            })
        );
    }

    #[test]
    fn view_flattens_record() {
        let v = OffSwitchView {
            record: rec(true, None),
            scope: "head".into(),
            active: true,
        };
        let j = serde_json::to_value(&v).unwrap();
        assert_eq!(j["head"], "H");
        assert_eq!(j["classifier"], serde_json::Value::Null);
        assert_eq!(j["active"], true);
        assert_eq!(j["scope"], "head");
    }
}
