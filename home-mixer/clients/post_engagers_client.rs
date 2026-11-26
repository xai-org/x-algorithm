use std::collections::HashMap;
use std::time::{Duration, Instant};

use prost::Message;
use tonic::async_trait;
use tracing::warn;
use xai_proto::engagement_counter::PostEngagers;
use xai_redis_client::{NodeOpStat, XdsRedisClient, XdsRedisConfig};
use xai_stats_receiver::global_stats_receiver;

const CACHE_NAME: &str = "post-engagers-cache";
const CACHE_ENVIRONMENT: &str = "prod";
const CACHE_NAMESPACE: &str = "cache";
const CACHE_KEY_PREFIX: &str = "pe_";
const RESPONSE_TIMEOUT: Duration = Duration::from_millis(200);
const MGET_NODE_TIMEOUT: Duration = Duration::from_millis(100);

const LOOKUP_METRIC: &str = "PostEngagersClient.lookup";
const LATENCY_METRIC: &str = "PostEngagersClient.latency_seconds";

#[async_trait]
pub trait PostEngagersClient: Send + Sync {
    async fn get_post_engagers(&self, post_ids: &[u64]) -> HashMap<u64, Option<PostEngagers>>;
}

pub struct ProdPostEngagersClient {
    client: XdsRedisClient,
}

impl ProdPostEngagersClient {
    pub async fn new(datacenter: &str) -> anyhow::Result<Self> {
        let eds_resource_name = eds_resource_name(datacenter);
        let client = XdsRedisClient::new_with_response_timeout(
            XdsRedisConfig {
                eds_resource_name: eds_resource_name.clone(),
            },
            RESPONSE_TIMEOUT,
        )
        .await
        .map_err(|e| anyhow::anyhow!("ProdPostEngagersClient build failed: {e}"))?;
        if !client.is_ready() {
            warn!(
                "{eds_resource_name} has no endpoints yet; post engager lookups fail until it does"
            );
        }
        Ok(Self { client })
    }
}

#[async_trait]
impl PostEngagersClient for ProdPostEngagersClient {
    async fn get_post_engagers(&self, post_ids: &[u64]) -> HashMap<u64, Option<PostEngagers>> {
        if post_ids.is_empty() {
            return HashMap::new();
        }
        let keys: Vec<Vec<u8>> = post_ids.iter().map(|&id| cache_key(id)).collect();
        let start = Instant::now();
        let response = self
            .client
            .mget_raw_detailed(&keys, MGET_NODE_TIMEOUT)
            .await;
        observe_latency(start);

        let mut unanswered = vec![false; post_ids.len()];
        for &index in &response.failed {
            unanswered[index] = true;
        }
        let mut lookups = LookupCounts {
            unanswered: response.failed.len(),
            unanswered_timed_out: all_failed_nodes_timed_out(&response.stats),
            ..Default::default()
        };
        let mut records = HashMap::with_capacity(post_ids.len());
        for ((&post_id, value), unanswered) in post_ids.iter().zip(response.values).zip(unanswered)
        {
            if unanswered {
                continue;
            }
            match value.map(|bytes| PostEngagers::decode(bytes.as_slice())) {
                None => {
                    lookups.missing += 1;
                    records.insert(post_id, None);
                }
                Some(Ok(record)) => {
                    lookups.hit += 1;
                    records.insert(post_id, Some(record));
                }
                Some(Err(_)) => lookups.undecodable += 1,
            }
        }
        lookups.record();
        records
    }
}

#[derive(Default)]
struct LookupCounts {
    hit: usize,
    missing: usize,
    undecodable: usize,
    unanswered: usize,
    unanswered_timed_out: bool,
}

impl LookupCounts {
    fn record(&self) {
        let Some(receiver) = global_stats_receiver() else {
            return;
        };
        let (timeouts, errors) = if self.unanswered_timed_out {
            (self.unanswered, self.undecodable)
        } else {
            (0, self.undecodable + self.unanswered)
        };
        for (outcome, count) in [
            ("hit", self.hit),
            ("missing", self.missing),
            ("error", errors),
            ("timeout", timeouts),
        ] {
            if count > 0 {
                receiver.incr(LOOKUP_METRIC, &[("outcome", outcome)], count as u64);
            }
        }
    }
}

fn all_failed_nodes_timed_out(stats: &[NodeOpStat]) -> bool {
    let mut failed_nodes = stats.iter().filter(|stat| !stat.ok).peekable();
    failed_nodes.peek().is_some() && failed_nodes.all(|stat| stat.latency >= MGET_NODE_TIMEOUT)
}

fn observe_latency(start: Instant) {
    if let Some(receiver) = global_stats_receiver() {
        receiver.observe_expo(LATENCY_METRIC, &[], start.elapsed().as_secs_f64());
    }
}

fn eds_resource_name(datacenter: &str) -> String {
    format!(
        "xdstp://discovery-{datacenter}/envoy.config.endpoint.v3.ClusterLoadAssignment/redis-global.{CACHE_NAME}.{CACHE_ENVIRONMENT}.{CACHE_NAMESPACE}:redis"
    )
}

fn cache_key(post_id: u64) -> Vec<u8> {
    format!("{CACHE_KEY_PREFIX}{post_id}").into_bytes()
}

#[derive(Default)]
pub struct MockPostEngagersClient {
    pub records: HashMap<u64, PostEngagers>,
}

#[async_trait]
impl PostEngagersClient for MockPostEngagersClient {
    async fn get_post_engagers(&self, post_ids: &[u64]) -> HashMap<u64, Option<PostEngagers>> {
        post_ids
            .iter()
            .map(|&id| (id, self.records.get(&id).cloned()))
            .collect()
    }
}
