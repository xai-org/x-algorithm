use std::collections::HashMap;
use tonic::async_trait;
use tracing::warn;
use wingman_client::{Exists, Fleet};
use wingman_wire::reason;
use wingman_wire::v1::IntersectMode;

const FOLLOWS_GRAPH: u32 = 1;
const FOLLOWS_FLEET: Fleet<'static> = Fleet {
    wily_path: "/s/wingman/wingman-api-follows",
    authority_template: "wingman.wingman-api-follows.prod.{datacenter}.s2s.twttr.net",
    max_decoding_message_size: None,
    wily_client_name: "visibility-filtering-service",
};

#[async_trait]
pub trait WingmanClient: Send + Sync {
    async fn batch_exists_intersect(
        &self,
        viewer_id: u64,
        root_author_ids: &[u64],
    ) -> HashMap<u64, Exists>;
}

pub struct ProdWingmanClient {
    client: wingman_client::Client,
}

impl ProdWingmanClient {
    pub async fn new(datacenter: &str) -> anyhow::Result<Self> {
        let client = wingman_client::connect(datacenter, &FOLLOWS_FLEET).await?;
        Ok(Self { client })
    }
}

#[async_trait]
impl WingmanClient for ProdWingmanClient {
    async fn batch_exists_intersect(
        &self,
        viewer_id: u64,
        root_author_ids: &[u64],
    ) -> HashMap<u64, Exists> {
        let pairs = root_author_ids.iter().map(|&root| (root, viewer_id));
        let mode = IntersectMode::FwdXRev;
        wingman_client::exists_intersect(&self.client, FOLLOWS_GRAPH, mode, pairs)
            .await
            .inspect_err(|status| {
                warn!(
                    reason = reason::of(status).label(),
                    "Wingman exists_intersect failed"
                );
            })
            .map(|exists| root_author_ids.iter().copied().zip(exists).collect())
            .unwrap_or_default()
    }
}
