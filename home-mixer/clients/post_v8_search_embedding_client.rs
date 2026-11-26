use std::collections::HashMap;
use std::time::Duration;

use tonic::async_trait;
use tracing::warn;
use xai_manhattan::{s2s::S2sConfig, Key, NativeManhattanClient, Tenant};
use xai_uas_thrift::deserialize_from_bytes_binary;
use xai_uas_thrift::tweet_embedding::TweetEmbedding;

use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};

const MH_CLUSTER: &str = "nash";
const MH_APP_ID: &str = "phoenix_online_training_mm";
const MH_DATASET: &str = "tweet_multimodal_embeddings";
const MH_TIMEOUT: Duration = Duration::from_millis(100);
const EMBEDDING_DIM: usize = 1024;

#[async_trait]
pub trait PostV8SearchEmbeddingClient: Send + Sync {
    async fn fetch_embeddings(&self, post_ids: &[u64], version: &str) -> HashMap<u64, Vec<f32>>;
}

pub struct NoopPostV8SearchEmbeddingClient;

#[async_trait]
impl PostV8SearchEmbeddingClient for NoopPostV8SearchEmbeddingClient {
    async fn fetch_embeddings(&self, _post_ids: &[u64], _version: &str) -> HashMap<u64, Vec<f32>> {
        HashMap::new()
    }
}

pub struct ManhattanPostV8SearchEmbeddingClient {
    mh: NativeManhattanClient,
}

impl ManhattanPostV8SearchEmbeddingClient {
    pub async fn new(datacenter: &str) -> anyhow::Result<Self> {
        let s2s = S2sConfig {
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            ca_cert_path: S2S_CHAIN_PATH.clone(),
        };
        let mh = NativeManhattanClient::builder_from_tenant_s2s(&tenant(), datacenter, s2s)
            .timeout(MH_TIMEOUT)
            .retries(0)
            .multiplexed()
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("post v8_search embedding Manhattan build failed: {e}"))?;
        Ok(Self { mh })
    }
}

#[async_trait]
impl PostV8SearchEmbeddingClient for ManhattanPostV8SearchEmbeddingClient {
    async fn fetch_embeddings(&self, post_ids: &[u64], version: &str) -> HashMap<u64, Vec<f32>> {
        if post_ids.is_empty() {
            return HashMap::new();
        }
        let version = version.as_bytes().to_vec();
        let keys: Vec<(Key<'_>, Key<'_>)> = post_ids
            .iter()
            .map(|&post_id| {
                (
                    vec![(post_id as i64).to_be_bytes().to_vec()].into(),
                    vec![version.clone()].into(),
                )
            })
            .collect();
        let batch = match self.mh.batch_get(tenant(), keys).await {
            Ok(batch) => batch,
            Err(err) => {
                warn!("post v8_search embedding fetch failed: {err}");
                return HashMap::new();
            }
        };
        post_ids
            .iter()
            .zip(batch.items)
            .filter_map(|(&post_id, item)| {
                let item = item.ok().flatten()?;
                let embedding =
                    deserialize_from_bytes_binary::<TweetEmbedding>(item.value().as_bytes())
                        .ok()?
                        .embedding1?;
                (embedding.len() == EMBEDDING_DIM).then(|| {
                    (
                        post_id,
                        embedding
                            .into_iter()
                            .map(|x| x.into_inner() as f32)
                            .collect(),
                    )
                })
            })
            .collect()
    }
}

fn tenant() -> Tenant {
    Tenant {
        cluster: MH_CLUSTER.to_string(),
        app_id: MH_APP_ID.to_string(),
        dataset: MH_DATASET.to_string(),
    }
}
