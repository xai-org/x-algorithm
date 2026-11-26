use std::sync::LazyLock;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::time::timeout;
use tonic::codec::CompressionEncoding;
use tonic::{async_trait, Status};
use tracing::{info, warn};
use xai_candidate_pipeline::component_library::utils::SuccessRateRouter;
use xai_recsys_proto::ad_index_service_client::AdIndexServiceClient;
use xai_recsys_proto::{AdIndexRequest, AdIndexResponse};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};
use xai_x_rpc::balanced_channel::{LbPolicy, LoadBalancedChannel};
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};
use xai_x_rpc::xds_endpoint_source::XdsEndpointSource;
use xai_xds_client::{ServiceState, StartFrom, XdsClient};

const TIMEOUT_MS: u64 = 1300;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
const NUM_ENDPOINTS: usize = 5;
const METRIC_NAME: &str = "AdIndexClient.get_eligible_ads";
const XDS_ENDPOINTS_METRIC_NAME: &str = "AdIndexClient.xds_endpoints";
const XDS_BUILD_METRIC_NAME: &str = "AdIndexClient.xds_build";
const ROUTE_METRIC_NAME: &str = "AdIndexClient.route";

pub const SR_ROUTER_DECIDER: &str = "enable_ad_index_sr_router_home_mixer";

type Client = AdIndexServiceClient<LoadBalancedChannel>;

fn app_env() -> String {
    std::env::var("APP_ENV").unwrap_or_else(|_| "prod".to_string())
}

fn ad_index_xds_listener(env: &str) -> String {
    format!("xai-ad-index.{env}.ads:grpc")
}

fn ad_index_xds_eds_resource(dc: &str, env: &str) -> String {
    format!(
        "xdstp://discovery-{dc}/envoy.config.endpoint.v3.ClusterLoadAssignment/xai-ad-index.{env}.ads:grpc"
    )
}

fn ad_index_sni(dc: &str, env: &str) -> String {
    format!("ads.ad-index.{env}.{dc}.s2s.twttr.net")
}

fn peer_dc(dc: &str) -> Option<&'static str> {
    match dc {
        "atla" => Some("pdxa"),
        "pdxa" => Some("atla"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdIndexCluster {
    Local,
    Remote,
}

impl AdIndexCluster {
    fn as_label(self) -> &'static str {
        match self {
            AdIndexCluster::Local => "local",
            AdIndexCluster::Remote => "remote",
        }
    }
}

#[async_trait]
pub trait AdIndexClient: Send + Sync {
    async fn get_eligible_ads(
        &self,
        request: AdIndexRequest,
        sr_router_enabled: bool,
    ) -> Result<AdIndexResponse, Status>;
}

static SR_ROUTER: LazyLock<SuccessRateRouter<AdIndexCluster>> =
    LazyLock::new(|| SuccessRateRouter::new(AdIndexCluster::Local, AdIndexCluster::Remote));

pub struct ProdAdIndexClient {
    local: Client,
    remote: Option<Client>,
}

impl ProdAdIndexClient {
    pub async fn new(dc: &str) -> Result<Self, Status> {
        let env = app_env();
        let (local, remote) = tokio::join!(
            async {
                let local = build_client(
                    dc,
                    &env,
                    StartFrom::Lds(ad_index_xds_listener(&env)),
                    AdIndexCluster::Local,
                )
                .await;
                record_build(AdIndexCluster::Local, local.is_ok());
                local
            },
            async {
                let remote_dc = peer_dc(dc)?;
                let resource = ad_index_xds_eds_resource(remote_dc, &env);
                match build_client(
                    remote_dc,
                    &env,
                    StartFrom::Eds(resource),
                    AdIndexCluster::Remote,
                )
                .await
                {
                    Ok(client) => {
                        record_build(AdIndexCluster::Remote, true);
                        Some(client)
                    }
                    Err(e) => {
                        record_build(AdIndexCluster::Remote, false);
                        warn!(remote_dc, error = %e, "AdIndexClient: remote DC client unavailable; SR router disabled");
                        None
                    }
                }
            }
        );
        Ok(Self {
            local: local?,
            remote,
        })
    }

    fn client_for(&self, cluster: AdIndexCluster) -> Client {
        match (cluster, &self.remote) {
            (AdIndexCluster::Remote, Some(remote)) => remote.clone(),
            _ => self.local.clone(),
        }
    }
}

async fn build_client(
    dc: &str,
    env: &str,
    start_from: StartFrom,
    cluster: AdIndexCluster,
) -> Result<Client, Status> {
    let resource = start_from.resource_name().to_string();
    info!(dc, %env, %resource, cluster = cluster.as_label(), "AdIndexClient: building xDS client");

    let xds_client = XdsClient::from_bootstrap()
        .map_err(|e| {
            Status::failed_precondition(format!(
                "ad-index xDS bootstrap unavailable ({resource}): {e}"
            ))
        })?
        .start_from(start_from)
        .build()
        .await
        .map_err(|e| {
            Status::unavailable(format!(
                "ad-index xDS client build failed ({resource}): {e}"
            ))
        })?;

    spawn_endpoint_gauge(cluster, xds_client.subscribe());

    let tls = TlsMode::mtls_from_env()
        .map_err(|e| Status::internal(format!("failed to load mTLS certs: {e}")))?
        .with_domain_override(ad_index_sni(dc, env));

    let channel = ChannelBuilder::new("ad-index")
        .request_timeout(Duration::from_millis(TIMEOUT_MS))
        .connect_timeout(CONNECT_TIMEOUT)
        .tls(tls)
        .endpoint_source(XdsEndpointSource::new(xds_client))
        .aperture(NUM_ENDPOINTS)
        .deterministic()
        .build_load_balanced(LbPolicy::penalized_peak_ewma())
        .await
        .map_err(|e| Status::unavailable(format!("failed to connect to ad-index ({dc}): {e}")))?;

    Ok(AdIndexServiceClient::new(channel)
        .send_compressed(CompressionEncoding::Zstd)
        .accept_compressed(CompressionEncoding::Zstd))
}

fn pick_cluster(
    router: &SuccessRateRouter<AdIndexCluster>,
    router_enabled: bool,
) -> AdIndexCluster {
    if router_enabled {
        router.route(AdIndexCluster::Local)
    } else {
        AdIndexCluster::Local
    }
}

fn record_build(cluster: AdIndexCluster, built: bool) {
    if let Some(receiver) = global_stats_receiver() {
        let result_label = if built { "built" } else { "failed" };
        receiver.incr(
            XDS_BUILD_METRIC_NAME,
            &[
                ("ad_index_cluster", cluster.as_label()),
                ("result", result_label),
            ],
            1,
        );
    }
}

fn record_route(cluster: AdIndexCluster, router_enabled: bool) {
    if let Some(receiver) = global_stats_receiver() {
        let router_label = if router_enabled { "on" } else { "off" };
        receiver.incr(
            ROUTE_METRIC_NAME,
            &[
                ("ad_index_cluster", cluster.as_label()),
                ("router", router_label),
            ],
            1,
        );
    }
}

fn record_stats(cluster: AdIndexCluster, success: bool, latency_ms: f64) {
    if let Some(receiver) = global_stats_receiver() {
        let cluster_label = cluster.as_label();
        let result_label = if success { "success" } else { "failure" };
        receiver.incr(
            METRIC_NAME,
            &[
                ("ad_index_cluster", cluster_label),
                ("result", result_label),
            ],
            1,
        );
        receiver.observe(
            METRIC_NAME,
            &[("ad_index_cluster", cluster_label)],
            latency_ms,
            HistogramBuckets::Bucket500To1000,
        );
    }
}

fn spawn_endpoint_gauge(cluster: AdIndexCluster, mut state_rx: watch::Receiver<ServiceState>) {
    tokio::spawn(async move {
        loop {
            let count = state_rx
                .borrow_and_update()
                .endpoint_data
                .as_ref()
                .map_or(0, |cla| cla.num_endpoints());
            if let Some(receiver) = global_stats_receiver() {
                receiver.gauge(
                    XDS_ENDPOINTS_METRIC_NAME,
                    &[("ad_index_cluster", cluster.as_label())],
                    count as f64,
                );
            }
            if state_rx.changed().await.is_err() {
                break;
            }
        }
    });
}

#[async_trait]
impl AdIndexClient for ProdAdIndexClient {
    async fn get_eligible_ads(
        &self,
        request: AdIndexRequest,
        sr_router_enabled: bool,
    ) -> Result<AdIndexResponse, Status> {
        let router_enabled = sr_router_enabled && self.remote.is_some();
        let cluster = pick_cluster(&SR_ROUTER, router_enabled);
        record_route(cluster, router_enabled);

        let mut grpc_request = tonic::Request::new(request);
        grpc_request.set_timeout(Duration::from_millis(TIMEOUT_MS));

        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(TIMEOUT_MS),
            self.client_for(cluster).get_eligible_ads(grpc_request),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("AdIndexClient: deadline exceeded"))
        .and_then(|r| r.map(|r| r.into_inner()));
        let latency_ms = start.elapsed().as_millis() as f64;

        if router_enabled {
            SR_ROUTER.record(cluster, result.is_ok());
        }
        record_stats(cluster, result.is_ok(), latency_ms);

        result
    }
}

pub struct MockAdIndexClient;

#[async_trait]
impl AdIndexClient for MockAdIndexClient {
    async fn get_eligible_ads(
        &self,
        _request: AdIndexRequest,
        _sr_router_enabled: bool,
    ) -> Result<AdIndexResponse, Status> {
        Ok(AdIndexResponse {
            ad_info: vec![],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xds_listener_follows_env() {
        assert_eq!(ad_index_xds_listener("prod"), "xai-ad-index.prod.ads:grpc");
        assert_eq!(
            ad_index_xds_listener("staging"),
            "xai-ad-index.staging.ads:grpc"
        );
    }

    #[test]
    fn remote_eds_resource_names_the_peer_authority() {
        assert_eq!(
            ad_index_xds_eds_resource("pdxa", "prod"),
            "xdstp://discovery-pdxa/envoy.config.endpoint.v3.ClusterLoadAssignment/xai-ad-index.prod.ads:grpc"
        );
        assert_eq!(
            ad_index_xds_eds_resource("atla", "staging"),
            "xdstp://discovery-atla/envoy.config.endpoint.v3.ClusterLoadAssignment/xai-ad-index.staging.ads:grpc"
        );
    }

    #[test]
    fn sni_matches_env() {
        assert_eq!(
            ad_index_sni("atla", "prod"),
            "ads.ad-index.prod.atla.s2s.twttr.net"
        );
        assert_eq!(
            ad_index_sni("pdxa", "staging"),
            "ads.ad-index.staging.pdxa.s2s.twttr.net"
        );
    }

    #[test]
    fn peer_dc_pairs_atla_and_pdxa() {
        assert_eq!(peer_dc("atla"), Some("pdxa"));
        assert_eq!(peer_dc("pdxa"), Some("atla"));
        assert_eq!(peer_dc("fou"), None);
    }

    fn failing_local_router() -> SuccessRateRouter<AdIndexCluster> {
        let router = SuccessRateRouter::new(AdIndexCluster::Local, AdIndexCluster::Remote);
        for _ in 0..100 {
            router.record(AdIndexCluster::Local, false);
            router.record(AdIndexCluster::Remote, true);
        }
        router
    }

    #[test]
    fn router_disabled_always_picks_local() {
        let router = failing_local_router();
        assert!((0..1000).all(|_| pick_cluster(&router, false) == AdIndexCluster::Local));
    }

    #[test]
    fn router_enabled_shifts_off_a_failing_local() {
        let router = failing_local_router();
        assert!((0..1000).any(|_| pick_cluster(&router, true) == AdIndexCluster::Remote));
    }
}
