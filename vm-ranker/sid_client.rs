use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use moka::sync::Cache;
use tonic::transport::{Channel, Endpoint};
use xai_recsys_sid_proto::sid_lookup_service_client::SidLookupServiceClient;
use xai_recsys_sid_proto::LookupSidsRequest;

use crate::metrics::{
    DPP_HISTORY_SIDS, SID_CACHE_CAPACITY, SID_CACHE_ENTRIES, SID_LOOKUP_DURATION,
};

const LOOKUP_TIMEOUT: Duration = Duration::from_millis(30);
const NUM_CHANNELS: usize = 32;
const CACHE_CAPACITY: u64 = 10_000_000;
const CACHE_TTL: Duration = Duration::from_secs(48 * 60 * 60);
const BREAKER_FAILURES: u32 = 5;
const BREAKER_PAUSE_MS: u64 = 1000;

#[derive(Clone)]
pub struct SidClient {
    clients: Arc<[SidLookupServiceClient<Channel>]>,
    next: Arc<AtomicUsize>,
    cache: Cache<u64, i32>,
    epoch: Instant,
    failures: Arc<AtomicU32>,
    paused_until_ms: Arc<AtomicU64>,
}

impl SidClient {
    pub fn connect_lazy(endpoint: &str) -> Result<Self> {
        let endpoint = Endpoint::from_shared(endpoint.to_string())?
            .connect_timeout(Duration::from_secs(1))
            .tcp_nodelay(true)
            .http2_keep_alive_interval(Duration::from_secs(10))
            .keep_alive_timeout(Duration::from_secs(5))
            .keep_alive_while_idle(true);
        SID_CACHE_CAPACITY.set(CACHE_CAPACITY as i64);
        let clients = (0..NUM_CHANNELS)
            .map(|_| SidLookupServiceClient::new(endpoint.connect_lazy()))
            .collect();
        Ok(Self {
            clients,
            next: Arc::new(AtomicUsize::new(0)),
            cache: Cache::builder()
                .max_capacity(CACHE_CAPACITY)
                .time_to_live(CACHE_TTL)
                .build(),
            epoch: Instant::now(),
            failures: Arc::new(AtomicU32::new(0)),
            paused_until_ms: Arc::new(AtomicU64::new(0)),
        })
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    pub async fn l1_codes(&self, post_ids: &[u64]) -> HashMap<u64, i32> {
        let mut codes: HashMap<u64, i32> = post_ids
            .iter()
            .filter_map(|&id| Some((id, self.cache.get(&id)?)))
            .collect();
        DPP_HISTORY_SIDS
            .with_label_values(&["cached"])
            .inc_by(codes.len() as u64);
        let misses: Vec<u64> = post_ids
            .iter()
            .copied()
            .filter(|id| !codes.contains_key(id))
            .collect();
        if misses.is_empty() {
            return codes;
        }
        let paused_until = self.paused_until_ms.load(Ordering::Relaxed);
        let probe = paused_until > 0;
        if probe {
            let now = self.now_ms();
            let won = now >= paused_until
                && self
                    .paused_until_ms
                    .compare_exchange(
                        paused_until,
                        now + BREAKER_PAUSE_MS,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok();
            if !won {
                DPP_HISTORY_SIDS
                    .with_label_values(&["skipped"])
                    .inc_by(misses.len() as u64);
                return codes;
            }
        }
        let request = LookupSidsRequest {
            post_ids: misses.iter().map(|&id| id as i64).collect(),
        };
        let mut client =
            self.clients[self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len()].clone();
        let start = Instant::now();
        let response = tokio::time::timeout(LOOKUP_TIMEOUT, client.lookup_sids(request)).await;
        let (outcome, results) = match response {
            Ok(Ok(response)) => {
                let results = response.into_inner().results;
                if results.len() == misses.len() {
                    ("ok", Some(results))
                } else {
                    ("error", None)
                }
            }
            Ok(Err(_)) => ("error", None),
            Err(_) => ("timeout", None),
        };
        SID_LOOKUP_DURATION
            .with_label_values(&[outcome])
            .observe(start.elapsed().as_secs_f64());
        let Some(results) = results else {
            DPP_HISTORY_SIDS
                .with_label_values(&[outcome])
                .inc_by(misses.len() as u64);
            if self.failures.fetch_add(1, Ordering::Relaxed) + 1 >= BREAKER_FAILURES {
                self.paused_until_ms
                    .store(self.now_ms() + BREAKER_PAUSE_MS, Ordering::Relaxed);
            }
            return codes;
        };
        self.failures.store(0, Ordering::Relaxed);
        if probe {
            self.paused_until_ms.store(0, Ordering::Relaxed);
        }
        let found: Vec<(u64, i32)> = misses
            .iter()
            .zip(results)
            .filter_map(|(&id, sids)| Some((id, *sids.codes.first()?)))
            .collect();
        DPP_HISTORY_SIDS
            .with_label_values(&["found"])
            .inc_by(found.len() as u64);
        DPP_HISTORY_SIDS
            .with_label_values(&["missing"])
            .inc_by((misses.len() - found.len()) as u64);
        for (id, l1) in found {
            self.cache.insert(id, l1);
            codes.insert(id, l1);
        }
        SID_CACHE_ENTRIES.set(self.cache.entry_count() as i64);
        codes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cached_sids_survive_failed_lookups_and_the_breaker_pauses() {
        let client = SidClient::connect_lazy("http://127.0.0.1:1").unwrap();
        client.cache.insert(7, 70);
        for _ in 0..BREAKER_FAILURES {
            assert_eq!(client.l1_codes(&[7, 8]).await, HashMap::from([(7, 70)]));
        }
        assert!(client.now_ms() < client.paused_until_ms.load(Ordering::Relaxed));
    }
}
