use std::fmt::Display;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{error, info, warn};
use xai_cache::discovery::{WilyDiscovery, XdsDiscovery};
use xai_cache::{
    CacheClient, ClientBuilder, ClientConfig, HashAlgorithm, KVCacheError, Key, Protocol, RingHash,
    TlsConfig,
};
use xai_manhattan::s2s::S2sConfig;
use xai_stats_receiver::StatsReceiverExt;

use crate::metrics::LabelMetrics;
use crate::mh_client::{ManhattanLabelFetcher, MhLabelClient};
use crate::remote::RemoteSource;
use crate::source::SafetyLabelSource;
use crate::warmer::{CacheWarmer, WarmFetcher};

const CACHE_PATH: &str = "/s/cache/safety_label_store:twemcaches";
const INIT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
const INIT_MAX_BACKOFF: Duration = Duration::from_secs(15);

pub struct LabelSourceConfig {
    pub datacenter: String,
    pub s2s: S2sPaths,
    pub deterministic_aperture: bool,
    pub twemcache_client_name: String,
    pub label_cache_eds: Option<String>,
    pub cache_capacity: Option<usize>,
    pub stats: Option<Arc<dyn StatsReceiverExt>>,
    pub init_deadline: tokio::time::Instant,
}

#[derive(Clone)]
pub struct S2sPaths {
    pub cert: String,
    pub key: String,
    pub ca_chain: String,
}

#[derive(Debug, thiserror::Error)]
#[error("{}", .0.join("; "))]
pub struct ConnectError(Vec<String>);

pub struct LabelClients {
    manhattan: Arc<MhLabelClient>,
    twemcache: Arc<CacheClient>,
    dragonfly: Option<Arc<CacheClient>>,
    cache_capacity: Option<usize>,
    metrics: LabelMetrics,
}

impl LabelClients {
    pub async fn connect(config: LabelSourceConfig) -> Result<Self, ConnectError> {
        let LabelSourceConfig {
            datacenter,
            s2s,
            deterministic_aperture,
            twemcache_client_name,
            label_cache_eds,
            cache_capacity,
            stats,
            init_deadline,
        } = config;
        let manhattan = async {
            let manhattan = retry("manhattan", init_deadline, || {
                let s2s = S2sConfig {
                    client_cert_path: s2s.cert.clone(),
                    client_key_path: s2s.key.clone(),
                    ca_cert_path: s2s.ca_chain.clone(),
                };
                MhLabelClient::new(&datacenter, s2s, deterministic_aperture)
            })
            .await?;
            warm_manhattan(&manhattan).await;
            Ok::<_, String>(manhattan)
        };
        let twemcache = async {
            let twemcache = retry("twemcache", init_deadline, || {
                twemcache_client(&datacenter, twemcache_client_name.clone(), s2s.clone())
            })
            .await?;
            let start = Instant::now();
            let server_count = twemcache.warm_up().await;
            info!(
                server_count,
                latency_ms = elapsed_ms(start),
                "Cache client connected to {CACHE_PATH}"
            );
            warm_cache(&twemcache).await;
            Ok::<_, String>(twemcache)
        };
        let dragonfly = async {
            let Some(eds) = &label_cache_eds else {
                return Ok(None);
            };
            let dragonfly = retry("dragonfly", init_deadline, || {
                dragonfly_client(eds, s2s.clone())
            })
            .await?;
            warm_cache(&dragonfly).await;
            Ok::<_, String>(Some(Arc::new(dragonfly)))
        };
        match tokio::join!(manhattan, twemcache, dragonfly) {
            (Ok(manhattan), Ok(twemcache), Ok(dragonfly)) => Ok(Self {
                manhattan: Arc::new(manhattan),
                twemcache: Arc::new(twemcache),
                dragonfly,
                cache_capacity,
                metrics: LabelMetrics::new(stats),
            }),
            (manhattan, twemcache, dragonfly) => Err(ConnectError(
                [manhattan.err(), twemcache.err(), dragonfly.err()]
                    .into_iter()
                    .flatten()
                    .collect(),
            )),
        }
    }

    pub fn into_source(self, warm_fetcher: Option<Arc<dyn WarmFetcher>>) -> SafetyLabelSource {
        let Self {
            manhattan,
            twemcache,
            dragonfly,
            cache_capacity,
            metrics,
        } = self;
        let mut remote = RemoteSource::new(twemcache, manhattan, metrics.clone());
        if let Some(fetcher) = warm_fetcher {
            info!("L2 cache warmer enabled");
            remote = remote.with_warmer(CacheWarmer::spawn(fetcher, metrics.clone()));
        }
        if let Some(dragonfly) = dragonfly {
            remote = remote.with_dragonfly(dragonfly);
        }
        SafetyLabelSource::new(Arc::new(remote), cache_capacity, metrics)
    }
}

async fn retry<T, E, Fut>(
    client: &str,
    deadline: tokio::time::Instant,
    mut build: impl FnMut() -> Fut,
) -> Result<T, String>
where
    Fut: Future<Output = Result<T, E>>,
    E: Display,
{
    let mut backoff = Duration::from_secs(1);
    let mut attempt = 1u32;
    loop {
        if tokio::time::Instant::now() >= deadline {
            error!(
                client,
                attempt, "Client init skipped; init deadline exhausted"
            );
            return Err(format!("{client}: init deadline exhausted before attempt"));
        }
        let error = match tokio::time::timeout(INIT_ATTEMPT_TIMEOUT, build()).await {
            Ok(Ok(t)) => {
                if attempt > 1 {
                    info!(client, attempt, "Client init succeeded after retry");
                }
                return Ok(t);
            }
            Ok(Err(e)) => e.to_string(),
            Err(_) => format!(
                "init attempt timed out after {}s",
                INIT_ATTEMPT_TIMEOUT.as_secs()
            ),
        };
        if tokio::time::Instant::now() + backoff >= deadline {
            error!(
                client,
                attempt,
                error = %error,
                "Client init failed; retry budget exhausted"
            );
            return Err(format!("{client}: {error}"));
        }
        warn!(
            client,
            attempt,
            backoff_secs = backoff.as_secs(),
            error = %error,
            "Client init failed; retrying"
        );
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(INIT_MAX_BACKOFF);
        attempt += 1;
    }
}

async fn twemcache_client(
    datacenter: &str,
    client_name: String,
    s2s: S2sPaths,
) -> Result<CacheClient, KVCacheError> {
    let discovery = Arc::new(WilyDiscovery::new(CACHE_PATH, client_name, datacenter).await?);
    let config = ClientConfig::builder()
        .request_timeout(Duration::from_millis(20))
        .connect_timeout(Duration::from_secs(5))
        .hash_algorithm(HashAlgorithm::FNV1)
        .connections_per_endpoint(2)
        .depth_cap(100)
        .failure_accrual_enabled(false)
        .build();
    ClientBuilder::new(Protocol::Memcached, discovery, config)
        .with_tls(TlsConfig {
            ca_cert_path: s2s.ca_chain,
            client_cert_path: s2s.cert,
            client_key_path: s2s.key,
        })
        .build()
        .await
}

async fn dragonfly_client(eds: &str, s2s: S2sPaths) -> Result<CacheClient, String> {
    let start = Instant::now();
    let client = dragonfly_ring_client(eds, s2s)
        .await
        .map_err(|e| e.to_string())?;
    let server_count = client.warm_up().await;
    if server_count == 0 {
        return Err(format!("xDS discovered no label-cache servers for {eds}"));
    }
    info!(
        server_count,
        latency_ms = elapsed_ms(start),
        "Label cache client connected to {eds}"
    );
    Ok(client)
}

async fn dragonfly_ring_client(eds: &str, s2s: S2sPaths) -> Result<CacheClient, KVCacheError> {
    let discovery = Arc::new(XdsDiscovery::new(eds).await?);
    let config = ClientConfig::builder()
        .request_timeout(Duration::from_millis(20))
        .connect_timeout(Duration::from_secs(5))
        .ring_hash(RingHash::XXHash64)
        .hash_algorithm(HashAlgorithm::MurmurHash2)
        .connections_per_endpoint(2)
        .failure_accrual_enabled(false)
        .build();
    ClientBuilder::new(Protocol::Redis, discovery, config)
        .with_tls(TlsConfig {
            ca_cert_path: s2s.ca_chain,
            client_cert_path: s2s.cert,
            client_key_path: s2s.key,
        })
        .build()
        .await
}

fn elapsed_ms(start: Instant) -> u64 {
    crate::dragonfly::millis(start.elapsed())
}

async fn warm_cache(twemcache: &CacheClient) {
    let start = Instant::now();
    let key = match Key::new(b"slm_warmup".to_vec()) {
        Ok(key) => key,
        Err(e) => {
            warn!(error = %e, "Cache warmup key construction failed (non-fatal)");
            return;
        }
    };
    match twemcache.multi_get(std::slice::from_ref(&key)).await {
        Ok(map) => match map.get(&key) {
            Some(Ok(_)) => info!(latency_ms = elapsed_ms(start), "Cache warmup succeeded"),
            Some(Err(e)) => warn!(
                latency_ms = elapsed_ms(start),
                error = %e,
                "Cache warmup failed (non-fatal)"
            ),
            None => warn!(
                latency_ms = elapsed_ms(start),
                "Cache warmup returned no response (non-fatal)"
            ),
        },
        Err(e) => warn!(
            latency_ms = elapsed_ms(start),
            error = %e,
            "Cache warmup failed (non-fatal)"
        ),
    }
}

async fn warm_manhattan(manhattan: &dyn ManhattanLabelFetcher) {
    let start = Instant::now();
    match manhattan.fetch_labels(&[0]).await {
        Ok(_) => info!(latency_ms = elapsed_ms(start), "Manhattan warmup succeeded"),
        Err(e) => warn!(
            latency_ms = elapsed_ms(start),
            error = %e,
            "Manhattan warmup failed (non-fatal)"
        ),
    }
}
