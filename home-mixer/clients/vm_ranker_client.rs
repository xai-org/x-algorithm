use futures::future::join_all;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use strum::{EnumProperty, EnumString, VariantArray};
use tokio::sync::watch;
use tonic::codec::CompressionEncoding;
use tonic::{async_trait, Status};
use xai_candidate_pipeline::component_library::clients::XdsHealthProbe;
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};
use xai_vm_ranker_proto::vm_ranker_service_client::VmRankerServiceClient;
use xai_vm_ranker_proto::{GetCarouselModelRequest, RankRequest, RankResponse};
use xai_x_rpc::balanced_channel::{LbPolicy, LoadBalancedChannel};
use xai_x_rpc::grpc_client::{insecure_tls_config, Dscp};
use xai_x_rpc::service_probe::KeepAlive;
use xai_x_rpc::xds_endpoint_source::XdsEndpointSource;
use xai_xds_client::{ServiceState, StartFrom, XdsClient};

use super::rotating_channel::RotatingChannel;

pub const TIMEOUT_MS: u64 = 400;
const METRIC_NAME: &str = "VMRankerClient.rank";
const REFRESH_INTERVAL: Duration = Duration::from_secs(60);

const XDS_METRIC_NAME: &str = "VMRankerClient.xds";
const XDS_BUILD_METRIC_NAME: &str = "VMRankerClient.xds_build";
const XDS_ENDPOINTS_METRIC_NAME: &str = "VMRankerClient.xds_endpoints";
const TRANSPORT_SAFETY_NET_MS: u64 = 3000;
const XDS_BUILD_TIMEOUT: Duration = Duration::from_secs(10);
const XDS_SEND_TIMEOUT: Duration = Duration::from_millis(TIMEOUT_MS + 1300);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumString, EnumProperty, VariantArray)]
pub enum VMRankerCluster {
    #[strum(props(host = "experiment-1.prod.fou"))]
    Experiment1,
    #[strum(props(host = "experiment-2.prod.fou"))]
    Experiment2,
    #[strum(props(host = "experiment-3.prod.fou"))]
    Experiment3,
    #[strum(props(host = "experiment-4.prod.fou"))]
    Experiment4,
    #[strum(props(host = "experiment-5.prod.fou"))]
    Experiment5,
    #[strum(props(host = "experiment-6.prod.fou"))]
    Experiment6,
    #[strum(props(host = "experiment-7.prod.fou"))]
    Experiment7,
    #[strum(props(host = "experiment-8.prod.fou"))]
    Experiment8,
}

impl VMRankerCluster {
    pub fn endpoint(&self) -> String {
        let host = self
            .get_str("host")
            .expect("every variant must have a host");
        format!("https://xai-vm-ranker-{host}.twitter.biz:443")
    }

    pub fn xds_eds_resource(&self, authority: &str) -> String {
        let host = self
            .get_str("host")
            .expect("every variant must have a host");
        let parts: Vec<&str> = host.split('.').collect();
        let slot = parts.first().copied().unwrap_or("");
        let dc = parts.get(2).copied().unwrap_or("");
        format!(
            "xdstp://discovery-{authority}/envoy.config.endpoint.v3.ClusterLoadAssignment/{dc}-xai-vm-ranker-{slot}.prod.prod:grpc-tls"
        )
    }

    pub fn gate_name(&self) -> String {
        format!("{self:?}")
    }

    pub fn parse(s: &str) -> Self {
        s.parse().unwrap_or(Self::VARIANTS[0])
    }
}

#[async_trait]
pub trait VMRankerClient: Send + Sync {
    async fn rank(
        &self,
        cluster: VMRankerCluster,
        request: RankRequest,
    ) -> Result<RankResponse, Status>;

    async fn get_carousel_model(&self, _name: &str) -> Result<Vec<u8>, Status> {
        Err(Status::unimplemented("get_carousel_model"))
    }
}

pub struct ProdVMRankerClient {
    channels: HashMap<VMRankerCluster, Arc<RotatingChannel>>,
}

impl ProdVMRankerClient {
    pub async fn new() -> Result<Self, Status> {
        let mut channels = HashMap::new();
        for &cluster in VMRankerCluster::VARIANTS {
            let ch = RotatingChannel::new(
                cluster.endpoint(),
                Duration::from_millis(TIMEOUT_MS),
                REFRESH_INTERVAL,
            )
            .await?;
            channels.insert(cluster, ch);
        }
        Ok(Self { channels })
    }
}

#[async_trait]
impl VMRankerClient for ProdVMRankerClient {
    async fn rank(
        &self,
        cluster: VMRankerCluster,
        request: RankRequest,
    ) -> Result<RankResponse, Status> {
        let ch = self
            .channels
            .get(&cluster)
            .ok_or_else(|| Status::not_found(format!("No channel for cluster: {cluster:?}")))?;
        let channel = ch.next();

        let start = Instant::now();
        let result = VmRankerServiceClient::new((*channel).clone())
            .rank(request)
            .await
            .map(|r| r.into_inner());
        let latency_ms = start.elapsed().as_millis() as f64;

        if let Some(receiver) = global_stats_receiver() {
            let cluster_label = format!("{cluster:?}");
            let result_label = if result.is_ok() { "success" } else { "failure" };
            let scope = [
                ("vm_cluster", cluster_label.as_str()),
                ("result", result_label),
            ];
            receiver.incr(METRIC_NAME, &scope, 1);
            receiver.observe(
                METRIC_NAME,
                &[("vm_cluster", cluster_label.as_str())],
                latency_ms,
                HistogramBuckets::Bucket500To1000,
            );
        }

        result
    }

    async fn get_carousel_model(&self, name: &str) -> Result<Vec<u8>, Status> {
        let mut last_error = Status::unavailable("no vm-ranker channels");
        for cluster in VMRankerCluster::VARIANTS {
            let Some(ch) = self.channels.get(cluster) else {
                continue;
            };
            let request = GetCarouselModelRequest {
                name: name.to_string(),
            };
            match VmRankerServiceClient::new((*ch.next()).clone())
                .get_carousel_model(request)
                .await
            {
                Ok(response) => return Ok(response.into_inner().model_json.to_vec()),
                Err(status) => last_error = status,
            }
        }
        Err(last_error)
    }
}

pub struct XdsVMRankerClient {
    clients: HashMap<VMRankerCluster, VmRankerServiceClient<LoadBalancedChannel>>,
}

impl XdsVMRankerClient {
    pub async fn new(
        policy: LbPolicy,
        h2_initial_stream_window: u32,
        h2_initial_connection_window: u32,
        socket_buffer_bytes: u32,
        aperture_size: usize,
        health_probe: XdsHealthProbe,
        discovery_authority: &str,
    ) -> Result<Self, Status> {
        let builds = VMRankerCluster::VARIANTS.iter().map(|&cluster| {
            let eds_resource = cluster.xds_eds_resource(discovery_authority);
            let health_probe = health_probe.clone();
            async move {
                let build = async {
                    let xds_client = XdsClient::from_bootstrap()
                        .map_err(|e| {
                            Status::failed_precondition(format!(
                                "vm-ranker xDS bootstrap not available ({eds_resource}): {e}"
                            ))
                        })?
                        .start_from(StartFrom::Eds(eds_resource.clone()))
                        .build()
                        .await
                        .map_err(|e| {
                            Status::unavailable(format!(
                                "vm-ranker xDS client build failed ({eds_resource}): {e}"
                            ))
                        })?;
                    Self::spawn_endpoint_gauge(cluster, xds_client.subscribe());
                    let source = XdsEndpointSource::new(xds_client);
                    let mut builder = LoadBalancedChannel::builder(source)
                        .timeout(Duration::from_millis(TRANSPORT_SAFETY_NET_MS))
                        .connect_timeout(Duration::from_secs(1))
                        .with_insecure_tls(insecure_tls_config())
                        .http2_initial_windows(
                            Some(h2_initial_stream_window),
                            Some(h2_initial_connection_window),
                        )
                        .socket_buffer_bytes(Some(socket_buffer_bytes))
                        .dscp(Some(Dscp::InferenceCritical))
                        .keep_alive(KeepAlive {
                            interval: Some(Duration::from_secs(30)),
                            timeout: Some(Duration::from_secs(10)),
                            while_idle: true,
                        })
                        .aperture(aperture_size)
                        .deterministic()
                        .lb_policy(policy);
                    match &health_probe {
                        XdsHealthProbe::Readiness { port, path } => {
                            builder = builder.readiness_probe_with_path(*port, path.clone());
                        }
                        XdsHealthProbe::Tcp => {
                            builder = builder.tcp_health_probe();
                        }
                        XdsHealthProbe::Grpc { timeout, interval } => {
                            builder = builder
                                .grpc_health_probe()
                                .grpc_health_probe_timeout(*timeout)
                                .grpc_health_probe_interval(*interval);
                        }
                        XdsHealthProbe::None => {}
                    }
                    let channel = builder.channel().await.map_err(|e| {
                        Status::unavailable(format!(
                            "vm-ranker xDS channel build failed ({eds_resource}): {e}"
                        ))
                    })?;
                    Ok::<_, Status>(
                        VmRankerServiceClient::new(channel)
                            .send_compressed(CompressionEncoding::Zstd)
                            .accept_compressed(CompressionEncoding::Zstd),
                    )
                };
                let outcome = tokio::time::timeout(XDS_BUILD_TIMEOUT, build).await;
                (cluster, eds_resource, outcome)
            }
        });

        let mut clients = HashMap::new();
        for (cluster, eds_resource, outcome) in join_all(builds).await {
            match outcome {
                Ok(Ok(client)) => {
                    clients.insert(cluster, client);
                    Self::record_build(cluster, true);
                    tracing::info!(cluster = ?cluster, eds = %eds_resource, "XdsVMRankerClient: xDS P2C channel ready");
                }
                Ok(Err(e)) => {
                    Self::record_build(cluster, false);
                    tracing::warn!(
                        cluster = ?cluster, eds = %eds_resource, error = %e,
                        "XdsVMRankerClient: xDS P2C build failed; skipping cluster (DNS fallback)"
                    );
                }
                Err(_elapsed) => {
                    Self::record_build(cluster, false);
                    tracing::warn!(
                        cluster = ?cluster, eds = %eds_resource, timeout_s = XDS_BUILD_TIMEOUT.as_secs(),
                        "XdsVMRankerClient: xDS P2C build timed out; skipping cluster (DNS fallback)"
                    );
                }
            }
        }

        if clients.is_empty() {
            return Err(Status::unavailable(
                "vm-ranker xDS: no clusters built (all unavailable in discovery); using DNS",
            ));
        }
        tracing::info!(
            built = clients.len(),
            total = VMRankerCluster::VARIANTS.len(),
            "XdsVMRankerClient: xDS P2C build complete"
        );
        Ok(Self { clients })
    }

    fn record_build(cluster: VMRankerCluster, built: bool) {
        if let Some(receiver) = global_stats_receiver() {
            let cluster_label = format!("{cluster:?}");
            let result_label = if built { "built" } else { "skipped" };
            let scope = [
                ("vm_cluster", cluster_label.as_str()),
                ("result", result_label),
            ];
            receiver.incr(XDS_BUILD_METRIC_NAME, &scope, 1);
        }
    }

    fn spawn_endpoint_gauge(cluster: VMRankerCluster, mut state_rx: watch::Receiver<ServiceState>) {
        tokio::spawn(async move {
            let cluster_label = format!("{cluster:?}");
            loop {
                let count = {
                    let state = state_rx.borrow_and_update();
                    state
                        .endpoint_data
                        .as_ref()
                        .map(|cla| cla.num_endpoints())
                        .unwrap_or(0)
                };
                if let Some(receiver) = global_stats_receiver() {
                    receiver.gauge(
                        XDS_ENDPOINTS_METRIC_NAME,
                        &[("vm_cluster", cluster_label.as_str())],
                        count as f64,
                    );
                }
                if state_rx.changed().await.is_err() {
                    break;
                }
            }
        });
    }
}

#[async_trait]
impl VMRankerClient for XdsVMRankerClient {
    async fn rank(
        &self,
        cluster: VMRankerCluster,
        request: RankRequest,
    ) -> Result<RankResponse, Status> {
        let mut client =
            self.clients.get(&cluster).cloned().ok_or_else(|| {
                Status::not_found(format!("No xDS channel for cluster: {cluster:?}"))
            })?;

        let mut grpc_request = tonic::Request::new(request);
        grpc_request.set_timeout(Duration::from_millis(TIMEOUT_MS));

        let start = Instant::now();
        let result = match tokio::time::timeout(XDS_SEND_TIMEOUT, client.rank(grpc_request)).await {
            Ok(res) => res.map(|r| r.into_inner()),
            Err(_elapsed) => Err(Status::deadline_exceeded(
                "vm-ranker xDS rank exceeded XDS_SEND_TIMEOUT (no ready endpoint); DNS fallback",
            )),
        };
        let latency_ms = start.elapsed().as_millis() as f64;

        if let Some(receiver) = global_stats_receiver() {
            let cluster_label = format!("{cluster:?}");
            let result_label = if result.is_ok() { "success" } else { "failure" };
            let scope = [
                ("vm_cluster", cluster_label.as_str()),
                ("result", result_label),
            ];
            receiver.incr(XDS_METRIC_NAME, &scope, 1);
            receiver.observe(
                XDS_METRIC_NAME,
                &[("vm_cluster", cluster_label.as_str())],
                latency_ms,
                HistogramBuckets::Bucket500To1000,
            );
        }

        result
    }
}

pub struct MockVMRankerClient;

#[async_trait]
impl VMRankerClient for MockVMRankerClient {
    async fn rank(
        &self,
        _cluster: VMRankerCluster,
        _request: RankRequest,
    ) -> Result<RankResponse, Status> {
        Ok(RankResponse { candidates: vec![] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xds_eds_resource_matches_deployed_registration() {
        assert_eq!(
            VMRankerCluster::Experiment1.xds_eds_resource("atla"),
            "xdstp://discovery-atla/envoy.config.endpoint.v3.ClusterLoadAssignment/fou-xai-vm-ranker-experiment-1.prod.prod:grpc-tls"
        );
    }

    #[test]
    fn gate_name_matches_cluster_id_param_value() {
        let name = VMRankerCluster::Experiment1.gate_name();
        assert_eq!(name, "Experiment1");
        assert_eq!(VMRankerCluster::parse(&name), VMRankerCluster::Experiment1);
    }

    #[test]
    fn experiment_2_eds_and_gate_name() {
        assert_eq!(
            VMRankerCluster::Experiment2.xds_eds_resource("atla"),
            "xdstp://discovery-atla/envoy.config.endpoint.v3.ClusterLoadAssignment/fou-xai-vm-ranker-experiment-2.prod.prod:grpc-tls"
        );
        let name = VMRankerCluster::Experiment2.gate_name();
        assert_eq!(name, "Experiment2");
        assert_eq!(VMRankerCluster::parse(&name), VMRankerCluster::Experiment2);
    }

    #[test]
    fn experiment_3_eds_and_gate_name() {
        assert_eq!(
            VMRankerCluster::Experiment3.xds_eds_resource("atla"),
            "xdstp://discovery-atla/envoy.config.endpoint.v3.ClusterLoadAssignment/fou-xai-vm-ranker-experiment-3.prod.prod:grpc-tls"
        );
        let name = VMRankerCluster::Experiment3.gate_name();
        assert_eq!(name, "Experiment3");
        assert_eq!(VMRankerCluster::parse(&name), VMRankerCluster::Experiment3);
    }
}
