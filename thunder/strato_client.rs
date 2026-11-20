use anyhow::{anyhow, Context, Result};
use log::warn;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

use crate::metrics;

#[derive(Debug, Deserialize)]
struct StratoResponse<T> {
    v: T,
}

fn deserialize_string_to_i64<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Deserialize};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrInt {
        String(String),
        Int(i64),
    }

    match StringOrInt::deserialize(deserializer)? {
        StringOrInt::String(s) => s.parse::<i64>().map_err(de::Error::custom),
        StringOrInt::Int(i) => Ok(i),
    }
}

fn preview_for_log(text: &str) -> &str {
    const MAX_PREVIEW_BYTES: usize = 300;
    if text.len() <= MAX_PREVIEW_BYTES {
        return text;
    }
    // Truncating at a fixed byte offset can split a multi-byte UTF-8
    // character (user names in these responses are not ASCII), which
    // would panic; back up to the nearest char boundary instead.
    let mut end = MAX_PREVIEW_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[derive(Debug, Deserialize)]
struct UserProfile {
    name: String,
    #[serde(rename = "screenName")]
    screen_name: String,
}

#[derive(Debug, Deserialize)]
struct UserCounts {
    #[serde(deserialize_with = "deserialize_string_to_i64")]
    followers: i64,
    #[serde(deserialize_with = "deserialize_string_to_i64")]
    following: i64,
}

#[derive(Debug, Deserialize)]
struct UserData {
    profile: UserProfile,
    counts: UserCounts,
}

#[derive(Debug, Clone)]
pub struct UserMetadata {
    pub user_id: i64,
    pub name: String,
    pub screen_name: String,
    pub followers: i64,
    pub following: i64,
}

pub struct StratoClient {
    client: Client,
    base_url: String,
}

impl Default for StratoClient {
    fn default() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10000)
            .pool_idle_timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap();

        Self {
            client,
            base_url: "https://strato.twitter.biz".to_string(),
        }
    }
}

impl StratoClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn fetch_following_list(&self, user_id: i64, limit: i32) -> Result<Vec<i64>> {
        let ans = self.fetch_following_list_internal(user_id, limit).await?;
        ans.into_iter()
            .map(|x| x.parse::<i64>().context("Failed to parse user ID"))
            .collect()
    }

    async fn fetch_following_list_internal(&self, user_id: i64, limit: i32) -> Result<Vec<String>> {
        let start = std::time::Instant::now();

        let url = format!(
            "{}/op/fetch/socialgraph/serviceV2/followingInternal.User",
            self.base_url
        );

        let payload = serde_json::json!([user_id, {"limit": limit}]);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(&payload)?)
            .send()
            .await
            .context("Failed to fetch following list")?;

        let duration = start.elapsed();
        metrics::STRATO_REQUEST_DURATION
            .with_label_values(&["fetch_following_list"])
            .observe(duration.as_secs_f64());

        if !response.status().is_success() {
            metrics::STRATO_REQUESTS
                .with_label_values(&["fetch_following_list", "error"])
                .inc();
            warn!(
                "Following list fetch failed for {}: {}",
                user_id,
                response.status()
            );
            return Err(anyhow!(response.status().to_string()));
        }

        let text = response.text().await?;

        if text == "{\"ttl\":-1}" {
            metrics::STRATO_REQUESTS
                .with_label_values(&["fetch_following_list", "not_found"])
                .inc();
            return Err(anyhow!("ttl=-1"));
        }

        match serde_json::from_str::<StratoResponse<Vec<String>>>(&text) {
            Ok(result) => {
                metrics::STRATO_REQUESTS
                    .with_label_values(&["fetch_following_list", "success"])
                    .inc();
                Ok(result.v)
            }
            Err(e) => {
                metrics::STRATO_REQUESTS
                    .with_label_values(&["fetch_following_list", "parse_error"])
                    .inc();
                warn!(
                    "Failed to parse following list response for {}: {}. Response preview: {}",
                    user_id,
                    e,
                    preview_for_log(&text)
                );
                Err(anyhow!(e))
            }
        }
    }

    pub async fn fetch_user_metadata(&self, user_id: i64) -> Result<Option<UserMetadata>> {
        let start = std::time::Instant::now();

        let url = format!("{}/op/fetch/gizmoduck/composite.User", self.base_url);

        let payload = serde_json::json!([user_id, (serde_json::json!({}), ["profile", "counts"])]);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(&payload)?)
            .send()
            .await
            .context("Failed to fetch user metadata")?;

        let duration = start.elapsed();
        metrics::STRATO_REQUEST_DURATION
            .with_label_values(&["fetch_user_metadata"])
            .observe(duration.as_secs_f64());

        if !response.status().is_success() {
            metrics::STRATO_REQUESTS
                .with_label_values(&["fetch_user_metadata", "error"])
                .inc();
            warn!(
                "User metadata fetch failed for {}: {}",
                user_id,
                response.status()
            );
            return Ok(None);
        }

        let text = response.text().await?;

        if text == "{\"ttl\":-1}" {
            metrics::STRATO_REQUESTS
                .with_label_values(&["fetch_user_metadata", "not_found"])
                .inc();
            return Ok(None);
        }

        match serde_json::from_str::<StratoResponse<UserData>>(&text) {
            Ok(result) => {
                metrics::STRATO_REQUESTS
                    .with_label_values(&["fetch_user_metadata", "success"])
                    .inc();
                Ok(Some(UserMetadata {
                    user_id,
                    name: result.v.profile.name,
                    screen_name: result.v.profile.screen_name,
                    followers: result.v.counts.followers,
                    following: result.v.counts.following,
                }))
            }
            Err(e) => {
                metrics::STRATO_REQUESTS
                    .with_label_values(&["fetch_user_metadata", "parse_error"])
                    .inc();
                warn!(
                    "Failed to parse user metadata response for {}: {}. Response preview: {}",
                    user_id,
                    e,
                    preview_for_log(&text)
                );
                Ok(None)
            }
        }
    }
}
