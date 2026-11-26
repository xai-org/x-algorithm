use anyhow::{anyhow, bail, Context, Result};
use reqwest::{Client, Method, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{info, warn};
use xai_abuse_enforcement_types::config::ConfigPendingReview;

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);

const APP_URL: &str = "https://app.growthbook.io";

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Rejected(String);

pub fn is_rejected(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.is::<Rejected>())
}

#[derive(Clone)]
pub struct GrowthBookWriter {
    api_base: String,
    api_key: String,
    client: Client,
}

impl GrowthBookWriter {
    pub fn new(api_host: &str, api_key: String) -> Self {
        let api_base = format!("{}/api/v2", api_host.trim_end_matches('/'));
        let client = Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .expect("reqwest client builder should succeed with defaults");
        Self {
            api_base,
            api_key,
            client,
        }
    }

    pub async fn propose_value(
        &self,
        feature_key: &str,
        environment: &str,
        note: &str,
        value: Value,
    ) -> Result<ConfigPendingReview> {
        self.propose(feature_key, environment, note, |_| Ok(value))
            .await
    }

    pub async fn propose_field(
        &self,
        feature_key: &str,
        environment: &str,
        note: &str,
        field: &str,
        value: Value,
    ) -> Result<ConfigPendingReview> {
        self.propose(feature_key, environment, note, |current| {
            let Value::Object(mut obj) = current else {
                return Err(Rejected(format!(
                    "environment '{environment}' value is not a JSON object — can't patch field '{field}'"
                ))
                .into());
            };
            obj.insert(field.to_owned(), value);
            Ok(Value::Object(obj))
        })
        .await
    }

    async fn propose(
        &self,
        feature_key: &str,
        environment: &str,
        note: &str,
        edit: impl FnOnce(Value) -> Result<Value>,
    ) -> Result<ConfigPendingReview> {
        let created = self
            .call(
                Method::POST,
                &format!("/features/{feature_key}/revisions"),
                json!({"title": note, "comment": note}),
                false,
            )
            .await?;
        let revision = created
            .get("revision")
            .ok_or_else(|| anyhow!("create draft response missing 'revision'"))?;
        let version = revision
            .get("version")
            .and_then(Value::as_i64)
            .ok_or_else(|| anyhow!("create draft response missing 'revision.version'"))?;
        let revision_path = format!("/features/{feature_key}/revisions/{version}");
        let edited = async {
            let (rule_id, current) = env_rule(revision, environment)?;
            let value = serde_json::to_string(&edit(current)?)?;
            self.call(
                Method::PUT,
                &format!("{revision_path}/rules/{rule_id}"),
                json!({"rule": {"value": value}}),
                true,
            )
            .await?;
            self.call(
                Method::POST,
                &format!("{revision_path}/request-review"),
                json!({"comment": note}),
                false,
            )
            .await
        }
        .await;
        if let Err(e) = edited {
            let discarded = self
                .call(
                    Method::POST,
                    &format!("{revision_path}/discard"),
                    json!({}),
                    false,
                )
                .await;
            return Err(match discarded {
                Ok(_) => e.context(format!("draft revision {version} was discarded")),
                Err(d) => e.context(format!(
                    "draft revision {version} is left behind (discard failed: {d:#})"
                )),
            });
        }
        info!("GrowthBook draft {version} awaits review (feature={feature_key} env={environment})");
        Ok(ConfigPendingReview {
            status: "pending_review".into(),
            feature: feature_key.to_owned(),
            revision: version,
            environment: environment.to_owned(),
            url: format!("{APP_URL}/features/{feature_key}?v={version}"),
        })
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        body: Value,
        sends_the_edit: bool,
    ) -> Result<Value> {
        let url = format!("{}{path}", self.api_base);
        let resp = self
            .client
            .request(method.clone(), &url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("{method} {url}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status != StatusCode::OK {
            warn!("{method} {url} -> {status} body={text}");
            let message = format!("GrowthBook {method} {url} failed: {status} {text}");
            if sends_the_edit
                && matches!(
                    status,
                    StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY
                )
            {
                return Err(Rejected(message).into());
            }
            bail!(message);
        }
        serde_json::from_str(&text).with_context(|| format!("decoding {method} {url}"))
    }
}

fn env_rule(revision: &Value, environment: &str) -> Result<(String, Value)> {
    let rules = revision
        .get("rules")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("draft has no rules array"))?;
    let only_env = |r: &&Value| {
        r.get("type").and_then(Value::as_str) == Some("force")
            && r.get("allEnvironments").and_then(Value::as_bool) != Some(true)
            && r.get("environments") == Some(&json!([environment]))
    };
    let matching: Vec<&Value> = rules.iter().filter(only_env).collect();
    let [rule] = matching.as_slice() else {
        bail!(
            "expected exactly one force rule scoped to environment '{environment}' alone, found {}",
            matching.len()
        );
    };
    let id = rule
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("environment '{environment}' rule has no id"))?;
    let raw = rule.get("value").cloned().unwrap_or(Value::Null);
    let value = match &raw {
        Value::String(s) => serde_json::from_str(s).map_err(|e| {
            anyhow!("rule value for environment '{environment}' is not valid JSON: {e}")
        })?,
        _ => raw,
    };
    Ok((id.to_owned(), value))
}

#[cfg(test)]
pub(crate) mod test_support {
    use serde_json::{json, Value};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    pub(crate) const FEATURE: &str = "xai_abuse_enforcement_service_config";
    pub(crate) const VERSION: i64 = 83;

    pub(crate) fn draft() -> Value {
        json!({"revision": {
            "version": VERSION,
            "status": "draft",
            "rules": [
                {"id": "fr_prod", "type": "force", "allEnvironments": false,
                 "environments": ["production"],
                 "value": "{\"dry_run\":false,\"max_user_enforcements_per_day\":600000}"},
                {"id": "fr_stg", "type": "force", "allEnvironments": false,
                 "environments": ["staging"], "value": "{\"dry_run\":true}"},
            ],
        }})
    }

    pub(crate) async fn growthbook(draft: Value) -> MockServer {
        let server = MockServer::start().await;
        mount_growthbook(&server, draft).await;
        server
    }

    pub(crate) async fn mount_growthbook(server: &MockServer, draft: Value) {
        let base = format!("/api/v2/features/{FEATURE}/revisions");
        for (verb, tail) in [
            ("POST", String::new()),
            ("PUT", format!("/{VERSION}/rules/fr_prod")),
            ("PUT", format!("/{VERSION}/rules/fr_stg")),
            ("POST", format!("/{VERSION}/request-review")),
            ("POST", format!("/{VERSION}/discard")),
        ] {
            Mock::given(method(verb))
                .and(path(format!("{base}{tail}")))
                .respond_with(ResponseTemplate::new(200).set_body_json(draft.clone()))
                .mount(server)
                .await;
        }
    }

    pub(crate) async fn requests(server: &MockServer) -> Vec<(String, String, Value)> {
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .map(|r| {
                (
                    r.method.to_string(),
                    r.url.path().to_owned(),
                    serde_json::from_slice(&r.body).unwrap_or(Value::Null),
                )
            })
            .collect()
    }

    pub(crate) async fn assert_never_published(server: &MockServer) {
        for (verb, p, _) in requests(server).await {
            assert!(!p.ends_with("/publish"), "{verb} {p} publishes");
            assert!(
                !(verb == "POST" && p == format!("/api/v2/features/{FEATURE}")),
                "{verb} {p} updates the feature directly"
            );
            assert!(
                p.starts_with(&format!("/api/v2/features/{FEATURE}/revisions")),
                "{verb} {p} is outside the draft endpoints"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn writer(server: &MockServer) -> GrowthBookWriter {
        GrowthBookWriter::new(&server.uri(), "secret_test".into())
    }

    fn base() -> String {
        format!("/api/v2/features/{FEATURE}/revisions")
    }

    #[tokio::test]
    async fn replace_drafts_the_env_rule_and_requests_review() {
        let server = growthbook(draft()).await;
        let out = writer(&server)
            .propose_value(FEATURE, "staging", "note", json!({"dry_run": false}))
            .await
            .unwrap();
        assert_eq!(
            out,
            ConfigPendingReview {
                status: "pending_review".into(),
                feature: FEATURE.into(),
                revision: VERSION,
                environment: "staging".into(),
                url: format!("https://app.growthbook.io/features/{FEATURE}?v={VERSION}"),
            }
        );
        let got = requests(&server).await;
        let calls: Vec<(&str, &str)> = got
            .iter()
            .map(|(m, p, _)| (m.as_str(), p.as_str()))
            .collect();
        assert_eq!(
            calls,
            vec![
                ("POST", base().as_str()),
                ("PUT", format!("{}/{VERSION}/rules/fr_stg", base()).as_str()),
                (
                    "POST",
                    format!("{}/{VERSION}/request-review", base()).as_str()
                ),
            ]
        );
        assert_eq!(got[0].2, json!({"title": "note", "comment": "note"}));
        assert_eq!(got[1].2, json!({"rule": {"value": "{\"dry_run\":false}"}}));
        assert_eq!(got[2].2, json!({"comment": "note"}));
        let all = server.received_requests().await.unwrap();
        assert!(all
            .iter()
            .all(|r| r.headers.get("authorization").unwrap() == "Bearer secret_test"));
        assert_never_published(&server).await;
    }

    #[tokio::test]
    async fn field_patch_merges_onto_the_drafts_env_value() {
        let server = growthbook(draft()).await;
        writer(&server)
            .propose_field(FEATURE, "production", "note", "dry_run", json!(true))
            .await
            .unwrap();
        let got = requests(&server).await;
        assert_eq!(got[1].1, format!("{}/{VERSION}/rules/fr_prod", base()));
        let sent: Value =
            serde_json::from_str(got[1].2["rule"]["value"].as_str().unwrap()).unwrap();
        assert_eq!(
            sent,
            json!({"dry_run": true, "max_user_enforcements_per_day": 600000})
        );
        assert_never_published(&server).await;
    }

    #[tokio::test]
    async fn no_rule_for_the_env_alone_discards_the_draft() {
        let shared = json!({"revision": {"version": VERSION, "rules": [
            {"id": "fr_all", "type": "force", "allEnvironments": true, "environments": [],
             "value": "{}"},
            {"id": "fr_both", "type": "force", "allEnvironments": false,
             "environments": ["staging", "production"], "value": "{}"},
        ]}});
        let two = json!({"revision": {"version": VERSION, "rules": [
            {"id": "a", "type": "force", "allEnvironments": false,
             "environments": ["staging"], "value": "{}"},
            {"id": "b", "type": "force", "allEnvironments": false,
             "environments": ["staging"], "value": "{}"},
        ]}});
        for (draft, want) in [(shared, "found 0"), (two, "found 2")] {
            let server = growthbook(draft).await;
            let err = writer(&server)
                .propose_value(FEATURE, "staging", "note", json!({}))
                .await
                .unwrap_err();
            let msg = format!("{err:#}");
            assert!(msg.contains(want), "{msg}");
            assert!(msg.contains("discarded"), "{msg}");
            let calls: Vec<String> = requests(&server)
                .await
                .into_iter()
                .map(|(m, p, _)| format!("{m} {p}"))
                .collect();
            assert_eq!(
                calls,
                vec![
                    format!("POST {}", base()),
                    format!("POST {}/{VERSION}/discard", base()),
                ]
            );
            assert_never_published(&server).await;
        }
    }

    #[tokio::test]
    async fn patching_a_non_object_value_discards_the_draft() {
        let scalar = json!({"revision": {"version": VERSION, "rules": [
            {"id": "fr_stg", "type": "force", "allEnvironments": false,
             "environments": ["staging"], "value": "7"},
        ]}});
        let server = growthbook(scalar).await;
        let err = writer(&server)
            .propose_field(FEATURE, "staging", "note", "dry_run", json!(true))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("not a JSON object"), "{err:#}");
        assert!(is_rejected(&err), "the patch can't apply: the edit's fault");
        let last = requests(&server).await.pop().unwrap();
        assert_eq!(last.1, format!("{}/{VERSION}/discard", base()));
    }

    #[tokio::test]
    async fn a_failed_step_discards_the_draft() {
        for (failing, rejected) in [
            (("PUT", format!("/{VERSION}/rules/fr_stg")), true),
            (("POST", format!("/{VERSION}/request-review")), false),
        ] {
            let server = MockServer::start().await;
            Mock::given(method(failing.0))
                .and(path(format!("{}{}", base(), failing.1)))
                .respond_with(ResponseTemplate::new(400).set_body_string("schema says no"))
                .with_priority(1)
                .mount(&server)
                .await;
            mount_growthbook(&server, draft()).await;
            let err = writer(&server)
                .propose_value(FEATURE, "staging", "note", json!({}))
                .await
                .unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.contains("400") && msg.contains("schema says no"),
                "{msg}"
            );
            assert!(msg.contains("discarded"), "{msg}");
            assert_eq!(is_rejected(&err), rejected, "{msg}");
            let last = requests(&server).await.pop().unwrap();
            assert_eq!(last.1, format!("{}/{VERSION}/discard", base()));
            assert_never_published(&server).await;
        }
    }

    #[tokio::test]
    async fn a_failed_discard_says_the_draft_is_left_behind() {
        let server = MockServer::start().await;
        let no_rules = json!({"revision": {"version": VERSION, "rules": []}});
        Mock::given(method("POST"))
            .and(path(base()))
            .respond_with(ResponseTemplate::new(200).set_body_json(no_rules))
            .mount(&server)
            .await;
        let err = writer(&server)
            .propose_value(FEATURE, "staging", "note", json!({}))
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("left behind"), "{msg}");
    }

    #[tokio::test]
    async fn a_failed_or_held_create_stops_there() {
        for status in [500, 400, 202] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(base()))
                .respond_with(ResponseTemplate::new(status).set_body_json(
                    json!({"confirmation": {"id": "c1", "status": "pending"}, "message": "held"}),
                ))
                .mount(&server)
                .await;
            let err = writer(&server)
                .propose_value(FEATURE, "staging", "note", json!({}))
                .await
                .unwrap_err();
            assert!(format!("{err:#}").contains(&status.to_string()), "{err:#}");
            assert!(!is_rejected(&err), "{status}");
            assert_eq!(requests(&server).await.len(), 1, "{status}");
        }
    }
}
