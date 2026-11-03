use tonic::async_trait;
use xai_safety_label_source::WarmFetcher;
use xai_strato::{encode, StratoGrpc};

use crate::clients::strato;

const WARM_COLUMN_PATH: &str = "visibility/baseTweetSafetyLabelMap";

pub(crate) struct StratoWarmFetcher {
    grpc: StratoGrpc,
}

impl StratoWarmFetcher {
    pub(crate) fn new(grpc: StratoGrpc) -> Self {
        Self { grpc }
    }
}

#[async_trait]
impl WarmFetcher for StratoWarmFetcher {
    async fn fetch(&self, ids: &[u64]) -> usize {
        let keys = ids
            .iter()
            .map(|id| (WARM_COLUMN_PATH, encode(&(id.cast_signed(), ()))));
        strato::fetch_each(&self.grpc, keys, None)
            .await
            .iter()
            .filter(|result| result.is_err())
            .count()
    }
}
