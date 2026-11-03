use std::sync::LazyLock;
use std::time::Duration;

use tonic::async_trait;
use xai_manhattan::s2s::S2sConfig;
use xai_manhattan::{Key, LkeySelector, ManhattanError, NativeManhattanClient, Tenant};
use xai_safety_label_store::types::encode_pkey;

use super::decode::RawSafetyLabel;

const SCAN_LIMIT: u32 = 512;
const DEFAULT_TIMEOUT: Duration = Duration::from_millis(200);
const DEFAULT_RETRIES: usize = 1;

pub(crate) type FetchResult = Result<Vec<RawSafetyLabel>, ManhattanError>;

#[async_trait]
pub(crate) trait ManhattanLabelFetcher: Send + Sync {
    async fn fetch_labels(&self, tweet_ids: &[i64]) -> Result<Vec<FetchResult>, ManhattanError>;
}

static TENANT: LazyLock<Tenant> = LazyLock::new(|| Tenant {
    cluster: "jammer".to_string(),
    app_id: "safety_label_store".to_string(),
    dataset: "tweet_labels".to_string(),
});

pub(crate) struct MhLabelClient {
    mh: NativeManhattanClient,
}

impl MhLabelClient {
    pub(crate) async fn new(
        dc: &str,
        s2s: S2sConfig,
        deterministic_aperture: bool,
    ) -> Result<Self, ManhattanError> {
        let mut builder = NativeManhattanClient::builder_from_tenant_s2s(&TENANT, dc, s2s)
            .timeout(DEFAULT_TIMEOUT)
            .retries(DEFAULT_RETRIES);
        if deterministic_aperture {
            builder = builder.deterministic();
        }
        let client = builder.build().await?;
        Ok(Self { mh: client })
    }
}

#[async_trait]
impl ManhattanLabelFetcher for MhLabelClient {
    async fn fetch_labels(&self, tweet_ids: &[i64]) -> Result<Vec<FetchResult>, ManhattanError> {
        if tweet_ids.is_empty() {
            return Ok(Vec::new());
        }

        let pkeys: Vec<Key> = tweet_ids
            .iter()
            .map(|&id| Key::from([&encode_pkey(id)[..]]))
            .collect();
        let selector = LkeySelector::Range {
            from: None,
            to: None,
        };

        let all_results = self
            .mh
            .batch_scan(TENANT.clone(), pkeys, selector, SCAN_LIMIT)
            .await?;

        Ok(all_results
            .into_iter()
            .map(|scan_result| {
                scan_result.map(|items| {
                    items
                        .into_iter()
                        .filter_map(|item| {
                            let lkey_parts = item.lkey().into_byte_vecs();
                            let lkey_vec = lkey_parts.into_iter().next()?;
                            let lkey: [u8; 4] = match lkey_vec.try_into() {
                                Ok(arr) => arr,
                                Err(vec) => {
                                    tracing::debug!(
                                        lkey_len = vec.len(),
                                        "skipping MH item with non-4-byte lkey"
                                    );
                                    return None;
                                }
                            };
                            let mval = item.value().into_bytes();
                            Some(RawSafetyLabel { lkey, mval })
                        })
                        .collect()
                })
            })
            .collect())
    }
}
