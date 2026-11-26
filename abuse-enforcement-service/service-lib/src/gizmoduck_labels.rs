pub const SPAM_HIGH_RECALL: &str = "SpamHighRecall";

pub const CLIENT_EVENT_PHOENIX_SKIP: &str = "ClientEventPhoenixSkip";

pub fn label_ordinal_to_name(ordinal: i32) -> String {
    match ordinal {
        23 => SPAM_HIGH_RECALL.to_owned(),
        118 => CLIENT_EVENT_PHOENIX_SKIP.to_owned(),
        other => format!("UnknownLabel({other})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_abuse_thrift_codec::{Transcoder, SCHEMA_AIS};

    #[test]
    fn names_match_the_ais_label_enum() {
        let t = Transcoder::from_schema_bytes(SCHEMA_AIS).unwrap();
        let encode = |label: serde_json::Value| {
            t.encode_call(
                "intakeAction",
                "",
                1,
                "com.twitter.agenttools.common.labels.thriftscala.LabelIdType",
                &serde_json::json!({ "nativeUserLabels": label }),
                1,
            )
        };
        for (ordinal, name) in [(23, SPAM_HIGH_RECALL), (118, CLIENT_EVENT_PHOENIX_SKIP)] {
            assert_eq!(label_ordinal_to_name(ordinal), name);
            assert_eq!(
                encode(serde_json::json!(name)).unwrap(),
                encode(serde_json::json!(ordinal)).unwrap(),
                "{name} is not LabelValue {ordinal} in the AIS schema"
            );
        }
        assert!(encode(serde_json::json!("ClientEventPhoenixSkipp")).is_err());
    }
}
