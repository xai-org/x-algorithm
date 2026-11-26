use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tonic::async_trait;

use super::metrics::{self, LabelMetrics, WarmKeyResult};

const WARM_CHANNEL_CAPACITY: usize = 1024;
const WARM_LINGER: Duration = Duration::from_millis(500);
const WARM_FETCH_MAX_KEYS: usize = 50;

#[async_trait]
pub trait WarmFetcher: Send + Sync {
    async fn fetch(&self, ids: &[u64]) -> usize;
}

pub(crate) struct CacheWarmer {
    tx: mpsc::Sender<Vec<u64>>,
    metrics: LabelMetrics,
}

impl CacheWarmer {
    pub(crate) fn spawn(fetcher: Arc<dyn WarmFetcher>, metrics: LabelMetrics) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(WARM_CHANNEL_CAPACITY);
        tokio::spawn(drain(rx, fetcher, metrics.clone()));
        Arc::new(Self { tx, metrics })
    }

    #[cfg(test)]
    pub(crate) fn without_drain_task(capacity: usize) -> (Self, mpsc::Receiver<Vec<u64>>) {
        let (tx, rx) = mpsc::channel(capacity);
        let metrics = LabelMetrics::default();
        (Self { tx, metrics }, rx)
    }

    pub(crate) fn warm(&self, miss_ids: Vec<u64>) {
        let record = |result, count| metrics::record_cache_warm_keys(&self.metrics, result, count);
        record(WarmKeyResult::EligibleMiss, miss_ids.len());
        if miss_ids.is_empty() {
            return;
        }
        let count = miss_ids.len();
        match self.tx.try_send(miss_ids) {
            Ok(()) => record(WarmKeyResult::Enqueued, count),
            Err(_) => record(WarmKeyResult::DroppedChannelFull, count),
        }
    }
}

async fn drain(
    mut rx: mpsc::Receiver<Vec<u64>>,
    fetcher: Arc<dyn WarmFetcher>,
    metrics: LabelMetrics,
) {
    while let Some(mut ids) = rx.recv().await {
        let deadline = tokio::time::Instant::now() + WARM_LINGER;
        while ids.len() < WARM_FETCH_MAX_KEYS {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(mut more)) => ids.append(&mut more),
                _ => break,
            }
        }
        for chunk in ids.chunks(WARM_FETCH_MAX_KEYS) {
            let failed = fetcher.fetch(chunk).await;
            metrics::record_cache_warm_keys(&metrics, WarmKeyResult::FetchIssued, chunk.len());
            metrics::record_cache_warm_keys(&metrics, WarmKeyResult::FetchFailed, failed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct FakeFetcher {
        batches: Mutex<Vec<Vec<u64>>>,
    }

    impl FakeFetcher {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                batches: Mutex::new(Vec::new()),
            })
        }

        fn batches(&self) -> Vec<Vec<u64>> {
            self.batches.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl WarmFetcher for FakeFetcher {
        async fn fetch(&self, ids: &[u64]) -> usize {
            self.batches.lock().unwrap().push(ids.to_vec());
            0
        }
    }

    #[tokio::test(start_paused = true)]
    async fn drain_lingers_then_flushes_in_chunks() {
        let fetcher = FakeFetcher::new();
        let warmer =
            CacheWarmer::spawn(Arc::<FakeFetcher>::clone(&fetcher), LabelMetrics::default());

        warmer.warm((0..30).collect());
        warmer.warm((30..60).collect());
        tokio::time::sleep(WARM_LINGER * 2).await;
        warmer.warm(vec![100]);
        tokio::time::sleep(WARM_LINGER * 2).await;

        let batches = fetcher.batches();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], (0..50).collect::<Vec<u64>>());
        assert_eq!(batches[1], (50..60).collect::<Vec<u64>>());
        assert_eq!(batches[2], vec![100]);

        for start in (0..150).step_by(30) {
            warmer.warm((start..start + 30).collect());
        }
        tokio::time::sleep(WARM_LINGER * 2).await;
        let batches = fetcher.batches();
        assert!(batches
            .iter()
            .all(|batch| batch.len() <= WARM_FETCH_MAX_KEYS));
        assert_eq!(batches[3..].concat(), (0..150).collect::<Vec<u64>>());
    }
}
