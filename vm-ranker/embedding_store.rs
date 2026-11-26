use std::sync::Arc;

use anyhow::Result;
use xai_recsys_mm_server::mm_embedding_client::{
    get_mm_client, MmEmbeddingsClient, O2PreloadFuture,
};

pub struct EmbeddingStore {
    pub(crate) client: MmEmbeddingsClient,
    dim: usize,
}

impl EmbeddingStore {
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[cfg(test)]
    pub(crate) fn in_process(dim: usize) -> Self {
        Self {
            client: MmEmbeddingsClient::in_process(),
            dim,
        }
    }
}

pub fn init_store(dim: usize) -> Result<(Arc<EmbeddingStore>, O2PreloadFuture)> {
    let (client, preload_future) = get_mm_client(0, dim, false)?;
    Ok((Arc::new(EmbeddingStore { client, dim }), preload_future))
}
