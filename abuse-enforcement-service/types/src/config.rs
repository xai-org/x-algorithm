use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ConfigPatchRequest {
    pub value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ConfigPendingReview {
    pub status: String,
    pub feature: String,
    pub revision: i64,
    pub environment: String,
    pub url: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pending_review_decodes_the_services_202_body() {
        let body = json!({
            "status": "pending_review",
            "feature": "xai_abuse_enforcement_service_config",
            "revision": 83,
            "environment": "staging",
            "url": "https://app.growthbook.io/features/xai_abuse_enforcement_service_config?v=83",
        });
        let got: ConfigPendingReview = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(got.revision, 83);
        assert_eq!(got.environment, "staging");
        assert_eq!(serde_json::to_value(&got).unwrap(), body);
    }
}
