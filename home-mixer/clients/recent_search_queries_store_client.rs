use prost::Message;
use tonic::async_trait;
use xai_manhattan::{s2s::S2sConfig, NativeManhattanClient, Tenant};
use xai_recsys_proto::RecentSearchQueries;

use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};

const MH_CLUSTER: &str = "nash";
const MH_APP_ID: &str = "search_mixer_nash";
const MH_DATASET: &str = "search_recent_queries_nash";
const MH_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(100);

fn tenant() -> Tenant {
    Tenant {
        cluster: MH_CLUSTER.to_string(),
        app_id: MH_APP_ID.to_string(),
        dataset: MH_DATASET.to_string(),
    }
}

#[async_trait]
pub trait RecentSearchQueriesStoreClient: Send + Sync {
    async fn fetch(&self, user_id: u64) -> Result<Option<RecentSearchQueries>, String>;
}

pub struct ProdRecentSearchQueriesStoreClient {
    mh: NativeManhattanClient,
}

impl ProdRecentSearchQueriesStoreClient {
    pub async fn new(datacenter: &str) -> Result<Self, String> {
        let s2s = S2sConfig {
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            ca_cert_path: S2S_CHAIN_PATH.clone(),
        };
        let mh = NativeManhattanClient::builder_from_tenant_s2s(&tenant(), datacenter, s2s)
            .timeout(MH_TIMEOUT)
            .retries(1)
            .multiplexed()
            .no_batch()
            .build()
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self { mh })
    }
}

#[async_trait]
impl RecentSearchQueriesStoreClient for ProdRecentSearchQueriesStoreClient {
    async fn fetch(&self, user_id: u64) -> Result<Option<RecentSearchQueries>, String> {
        let pkey: Vec<Vec<u8>> = vec![user_id.to_be_bytes().to_vec()];
        let lkey: Vec<Vec<u8>> = vec![];
        let item = self
            .mh
            .get(tenant(), pkey, lkey)
            .await
            .map_err(|e| e.to_string())?;
        item.map(|item| RecentSearchQueries::decode(item.value().as_bytes()))
            .transpose()
            .map_err(|e| format!("decoding RecentSearchQueries: {e}"))
    }
}

pub struct MockRecentSearchQueriesStoreClient;

#[async_trait]
impl RecentSearchQueriesStoreClient for MockRecentSearchQueriesStoreClient {
    async fn fetch(&self, _user_id: u64) -> Result<Option<RecentSearchQueries>, String> {
        Ok(None)
    }
}
