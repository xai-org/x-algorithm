use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use envoy_types::pb::envoy::config::core::v3::Node;
use envoy_types::pb::envoy::config::listener::v3::Listener;
use envoy_types::pb::envoy::service::discovery::v3::aggregated_discovery_service_client::AggregatedDiscoveryServiceClient;
use envoy_types::pb::envoy::service::discovery::v3::DiscoveryRequest;
use prost::Message;
use tower::util::Either;
use tracing::info;

use tonic::transport::Channel;
use xai_dark_traffic::{DarkTrafficLayer, ReloadableDarkTrafficConfigBuilder};
use xai_x_rpc::dynamic_channel_manager::{
    ChannelFactory, DynamicChannelManager, EndpointDiscovery, EndpointInfo,
};
use xai_x_rpc::grpc_client::TlsMode;
use xai_x_rpc::xds_channel_factory::XdsChannelFactory;

const CONFIG_PATH: &str = "/config/dark-traffic/dark_traffic.yaml";

pub const MIRROR_NAMESPACE: &str = "visibility";
pub const DEVEL_APP_ENV: &str = "devel";
pub const MIRROR_PORT_ID: &str = "grpc";
pub const MIRROR_WORKLOAD_PREFIX: &str = "xai-vf-service";

const LISTENER_TYPE_URL: &str = "type.googleapis.com/envoy.config.listener.v3.Listener";
const LDS_MAX_DECODING_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
const LDS_RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
const CHANNEL_CREATE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_DESTINATIONS: usize = 3;

pub fn devel_tls_domain(dc: &str) -> String {
    format!("visibility.xai-vf-service.devel.{dc}.s2s.twttr.net")
}

pub type DarkLayer = Either<DarkTrafficLayer, tower::layer::util::Identity>;

pub fn parse_mirror_listener(listener_name: &str) -> Option<EndpointInfo> {
    let dest = listener_name.rsplit('/').next().unwrap_or(listener_name);
    let dest = dest.split('?').next().unwrap_or(dest);
    let workload = dest.strip_suffix(&format!(
        ".{DEVEL_APP_ENV}.{MIRROR_NAMESPACE}:{MIRROR_PORT_ID}"
    ))?;
    if !workload.starts_with(MIRROR_WORKLOAD_PREFIX) || workload.contains('.') {
        return None;
    }
    Some(EndpointInfo {
        name: format!("{workload}.{DEVEL_APP_ENV}"),
        xds_dest: dest.to_string(),
    })
}

fn cap_destinations(mut endpoints: Vec<EndpointInfo>) -> (Vec<EndpointInfo>, Vec<EndpointInfo>) {
    endpoints.sort_by(|a, b| a.name.cmp(&b.name));
    let dropped = endpoints.split_off(endpoints.len().min(MAX_DESTINATIONS));
    (endpoints, dropped)
}

struct XdsMirrorDiscovery {
    server_uri: String,
}

#[async_trait::async_trait]
impl EndpointDiscovery for XdsMirrorDiscovery {
    async fn discover(&self) -> anyhow::Result<Vec<EndpointInfo>> {
        tokio::time::timeout(LDS_RESPONSE_TIMEOUT, self.fetch())
            .await
            .context("wildcard LDS exchange timed out")?
    }
}

impl XdsMirrorDiscovery {
    async fn fetch(&self) -> anyhow::Result<Vec<EndpointInfo>> {
        let channel = tonic::transport::Endpoint::from_shared(self.server_uri.clone())
            .context("invalid kube-discovery URI")?
            .connect()
            .await
            .context("failed to connect to kube-discovery")?;

        let mut client = AggregatedDiscoveryServiceClient::new(channel)
            .max_decoding_message_size(LDS_MAX_DECODING_MESSAGE_SIZE);

        let request = DiscoveryRequest {
            type_url: LISTENER_TYPE_URL.to_string(),
            resource_names: vec!["*".to_string()],
            node: Some(Node {
                id: format!("{MIRROR_WORKLOAD_PREFIX}-dark-traffic"),
                cluster: MIRROR_NAMESPACE.to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        use futures::StreamExt;
        let requests = futures::stream::iter([request]).chain(futures::stream::pending());
        let mut stream = client
            .stream_aggregated_resources(requests)
            .await
            .context("wildcard LDS stream failed to open")?
            .into_inner();

        let response = loop {
            let response = stream
                .message()
                .await
                .context("wildcard LDS stream errored")?
                .context("wildcard LDS stream ended without a listener-carrying response")?;
            if !response.resources.is_empty() {
                break response;
            }
        };

        let names: Vec<String> = response
            .resources
            .iter()
            .filter_map(|any| Some(Listener::decode(any.value.as_ref()).ok()?.name))
            .collect();
        anyhow::ensure!(
            !names.is_empty(),
            "none of {} LDS resources decoded as Listener",
            response.resources.len()
        );
        let (endpoints, dropped) = cap_destinations(
            names
                .iter()
                .filter_map(|name| parse_mirror_listener(name))
                .collect(),
        );

        if !dropped.is_empty() {
            tracing::warn!(
                dropped = %dropped.iter().map(|e| e.name.as_str()).collect::<Vec<_>>().join(", "),
                "dark_traffic: more than {MAX_DESTINATIONS} devel destinations; mirroring to the first by name"
            );
        }
        if endpoints.is_empty() {
            tracing::warn!(
                listeners = names.len(),
                "dark_traffic: no devel listeners matched"
            );
        } else {
            info!(
                names = %endpoints.iter().map(|e| e.name.as_str()).collect::<Vec<_>>().join(", "),
                "dark_traffic: discovery complete"
            );
        }

        Ok(endpoints)
    }
}

struct MirrorChannelFactory(XdsChannelFactory);

#[async_trait::async_trait]
impl ChannelFactory for MirrorChannelFactory {
    async fn create_channel(&self, ep: &EndpointInfo) -> anyhow::Result<Channel> {
        tokio::time::timeout(CHANNEL_CREATE_TIMEOUT, self.0.create_channel(ep))
            .await
            .with_context(|| format!("channel dial timed out for {}", ep.xds_dest))?
    }
}

pub fn resolve_layer() -> DarkLayer {
    if !std::env::var(crate::config::ENV_DARK_TRAFFIC_ENABLED)
        .ok()
        .as_deref()
        .is_some_and(dark_traffic_enabled)
    {
        info!("dark_traffic: disabled");
        return Either::Right(tower::layer::util::Identity::new());
    }

    let workload = std::env::var("WORKLOAD_NAME").ok();
    let max_ordinal: Option<u32> = std::env::var("DARK_TRAFFIC_MAX_ORDINAL")
        .ok()
        .and_then(|s| s.parse().ok());
    let ordinal: Option<u32> = std::env::var("ORDINAL_NUMBER")
        .ok()
        .and_then(|s| s.parse().ok());
    if !should_mirror(workload.as_deref(), ordinal, max_ordinal) {
        info!(
            ?workload,
            ?ordinal,
            ?max_ordinal,
            "dark_traffic: disabled (not a mirror host)"
        );
        return Either::Right(tower::layer::util::Identity::new());
    }

    let dc = std::env::var("DATACENTER").unwrap_or_else(|_| "atla".to_string());
    let discovery = XdsMirrorDiscovery {
        server_uri: format!("http://frontend.kube-discovery.prod.svc.{dc}.kube.int-x.ai:8082"),
    };
    let devel_domain = devel_tls_domain(&dc);
    info!(devel_domain, "dark_traffic: enabled");

    #[expect(clippy::expect_used, reason = "startup fail-fast: TLS is required")]
    let tls = TlsMode::mtls_from_env().expect("S2S TLS config required");
    let factory = MirrorChannelFactory(XdsChannelFactory::new(
        tls.with_domain_override(&devel_domain),
    ));

    let channels = DynamicChannelManager::new(Arc::new(factory), Arc::new(discovery));

    let config = ReloadableDarkTrafficConfigBuilder::new(CONFIG_PATH)
        .forwarders({
            let ch = Arc::clone(&channels);
            move || ch.channels()
        })
        .build();

    Either::Left(DarkTrafficLayer::new(config))
}

pub fn dark_traffic_enabled(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

fn should_mirror(workload: Option<&str>, ordinal: Option<u32>, max_ordinal: Option<u32>) -> bool {
    let Some(workload) = workload else {
        return false;
    };
    if workload.ends_with("-canary") {
        return false;
    }
    let Some(ord) = ordinal else {
        return false;
    };
    ord < max_ordinal.unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROD: Option<&str> = Some("xai-vf-service");
    const CANARY: Option<&str> = Some("xai-vf-service-canary");

    #[test]
    fn should_mirror_only_selected_prod_ordinals() {
        for (workload, ordinal, max_ordinal, expected) in [
            (PROD, Some(0), None, true),
            (PROD, Some(1), None, false),
            (PROD, None, None, false),
            (PROD, None, Some(3), false),
            (PROD, Some(0), Some(3), true),
            (PROD, Some(2), Some(3), true),
            (PROD, Some(3), Some(3), false),
            (CANARY, Some(0), Some(u32::MAX), false),
            (None, Some(0), Some(11), false),
        ] {
            assert_eq!(should_mirror(workload, ordinal, max_ordinal), expected);
        }
    }

    #[test]
    fn parse_accepts_only_vf_devel_listeners() {
        for (listener, workload, xds_dest) in [
            (
                "xai-vf-service.devel.visibility:grpc",
                "xai-vf-service.devel",
                "xai-vf-service.devel.visibility:grpc",
            ),
            (
                "xai-vf-service-pr-123.devel.visibility:grpc",
                "xai-vf-service-pr-123.devel",
                "xai-vf-service-pr-123.devel.visibility:grpc",
            ),
            (
                "xdstp://kube-discovery/envoy.config.listener.v3.Listener/xai-vf-service-pr-123.devel.visibility:grpc?key=val",
                "xai-vf-service-pr-123.devel",
                "xai-vf-service-pr-123.devel.visibility:grpc",
            ),
        ] {
            let ep = parse_mirror_listener(listener).expect(listener);
            assert_eq!(ep.name, workload);
            assert_eq!(ep.xds_dest, xds_dest);
        }
    }

    #[test]
    fn parse_rejects_out_of_scope_listeners() {
        for name in [
            "xai-vf-service.staging.visibility:grpc",
            "xai-vf-service-user1-foo.staging.visibility:grpc",
            "xdstp://kube-discovery/envoy.config.listener.v3.Listener/xai-vf-service.staging.visibility:grpc?key=val",
            "xai-vf-service.prod.visibility:grpc",
            "xai-vf-service.development.visibility:grpc",
            "xai-vf-service.staging-devel.visibility:grpc",
            "xai-vf-service.devel-staging.visibility:grpc",
            "xai-vf-service.devel.visibility:grpc-extra",
            "xai-vf-service.devel.visibility:grpc.other",
            "other-svc.devel.visibility:grpc",
            "xai-vf-service.devel.other:grpc",
            "xai-vf-service.devel.visibility:metrics",
            "evil.xai-vf-service.devel.visibility:grpc",
            "xai-vf-service.devel.visibility",
            "",
        ] {
            assert!(parse_mirror_listener(name).is_none(), "{name}");
        }
    }

    #[test]
    fn mirror_keeps_the_first_three_destinations_by_name() {
        let devel = |workload: &str| {
            parse_mirror_listener(&format!("{workload}.devel.visibility:grpc")).expect(workload)
        };
        let names = |endpoints: &[EndpointInfo]| {
            endpoints.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
        };
        let (kept, dropped) = cap_destinations(
            [
                "xai-vf-service-d",
                "xai-vf-service",
                "xai-vf-service-b",
                "xai-vf-service-a",
                "xai-vf-service-c",
            ]
            .map(devel)
            .into(),
        );
        assert_eq!(
            names(&kept),
            [
                "xai-vf-service-a.devel",
                "xai-vf-service-b.devel",
                "xai-vf-service-c.devel"
            ]
        );
        assert_eq!(
            names(&dropped),
            ["xai-vf-service-d.devel", "xai-vf-service.devel"]
        );

        let (kept, dropped) = cap_destinations(vec![devel("xai-vf-service")]);
        assert_eq!(names(&kept), ["xai-vf-service.devel"]);
        assert!(dropped.is_empty());
    }
}
