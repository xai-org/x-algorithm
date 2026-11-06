use std::time::Duration;

use anyhow::{Context, Result};
use tonic::transport::Channel;
use tracing::{info, warn};
use xai_stats_receiver::global_stats_receiver;
use xai_wily::WilyNs;
use xai_x_rpc::balanced_channel::{LbPolicy, LoadBalancedChannel};
use xai_x_rpc::endpoint_source::PollingEndpointSource;
use xai_x_rpc::endpoint_weights::EndpointWeights;
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};
use xai_x_rpc::lookup_service::LookupService;
use xai_x_rpc::timed_buffer::DEFAULT_BUFFER_MAX_WAIT;
use xai_x_rpc::total_timeout::DEFAULT_TOTAL_TIMEOUT;
use xai_x_rpc::wily_lookup_service::WilyNsLookup;
use xai_x_rpc::xds_channel_builder::XdsChannelBuildError;
use xai_xds_client::StartFrom;

const VF_WILY_PATH: &str = "/s/visibility/xai-vf-service:grpc";

const DEFAULT_VF_XDS_LISTENER: &str = "xai-vf-service.prod.visibility:grpc";

const VF_XDS_LISTENER_ENV: &str = "VF_CLIENT_XDS_LISTENER";

const VF_DNS_POLL_INTERVAL: Duration = Duration::from_millis(5_000);

const VF_XDS_BUILD_TIMEOUT: Duration = Duration::from_secs(10);

const VF_XDS_EAGER_RESOLUTION_TIMEOUT: Duration = Duration::from_secs(5);

const VF_READINESS_PROBE_PORT: u16 = 9090;

const VF_READINESS_PROBE_PATH: &str = "/readyz";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VfDiscovery {
    Wily,
    Xds,
}

impl VfDiscovery {
    pub(crate) fn parse(env_name: &str, value: Option<&str>, default: Self) -> Self {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("xds") => Self::Xds,
            Some("wily") => Self::Wily,
            None | Some("") => default,
            Some(other) => {
                warn!(
                    value = other,
                    env = env_name,
                    "unrecognized VF discovery mode; defaulting to {default:?}"
                );
                default
            }
        }
    }

    pub(crate) fn from_env(env_name: &str, default: Self) -> Self {
        Self::parse(env_name, std::env::var(env_name).ok().as_deref(), default)
    }
}

#[derive(Debug)]
pub(crate) enum VfChannelError {
    Config(anyhow::Error),
    Connect(anyhow::Error),
}

impl std::fmt::Display for VfChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (Self::Config(e) | Self::Connect(e)) = self;
        write!(f, "{e:#}")
    }
}

impl std::error::Error for VfChannelError {}

pub(crate) fn env_override(value: Option<&str>, default: String) -> String {
    match value.map(str::trim) {
        Some(v) if !v.is_empty() => v.to_string(),
        _ => default,
    }
}

fn vf_xds_listener(override_value: Option<&str>) -> String {
    env_override(override_value, DEFAULT_VF_XDS_LISTENER.to_string())
}

fn vf_xds_listener_from_env() -> String {
    vf_xds_listener(std::env::var(VF_XDS_LISTENER_ENV).ok().as_deref())
}

pub(crate) fn visibility_tls_domain(service: &str, dc: &str) -> String {
    format!("visibility.{service}.prod.{dc}.s2s.twttr.net")
}

fn vf_tls_domain(dc: &str) -> String {
    visibility_tls_domain("visibility-filtering-service", dc)
}

fn s2s_tls(tls_domain: &str) -> Result<TlsMode, VfChannelError> {
    Ok(TlsMode::mtls_from_env()
        .context("failed to load mTLS certs (S2S cert env vars required)")
        .map_err(VfChannelError::Config)?
        .with_domain_override(tls_domain.to_string()))
}

fn incr_build_metric(name: &str, transport: &str, result: &str) {
    if let Some(sr) = global_stats_receiver() {
        sr.incr(
            "vf_client_channel_build",
            &[
                ("channel", name),
                ("transport", transport),
                ("result", result),
            ],
            1,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XdsTarget {
    pub listener: String,
    pub tls_domain: String,
}

impl XdsTarget {
    pub(crate) fn vf(dc: &str) -> Self {
        Self {
            listener: vf_xds_listener_from_env(),
            tls_domain: vf_tls_domain(dc),
        }
    }
}

pub(crate) struct VfChannelParams<'a> {
    pub name: &'a str,
    pub dc: &'a str,
    pub discovery: VfDiscovery,
    pub xds: XdsTarget,
    pub aperture_size: usize,
    pub deterministic_aperture: bool,
    pub xds_lb_policy: LbPolicy,
}

pub(crate) enum VfChannel {
    Wily(Channel),
    Xds(LoadBalancedChannel),
}

pub(crate) async fn build_vf_channel(
    params: &VfChannelParams<'_>,
) -> Result<VfChannel, VfChannelError> {
    match params.discovery {
        VfDiscovery::Wily => Ok(VfChannel::Wily(build_vf_wily_channel(params).await?)),
        VfDiscovery::Xds => Ok(VfChannel::Xds(build_vf_xds_channel(params).await?)),
    }
}

pub(crate) async fn build_vf_xds_channel(
    params: &VfChannelParams<'_>,
) -> Result<LoadBalancedChannel, VfChannelError> {
    let name = params.name;
    let listener = &params.xds.listener;
    let result =
        match tokio::time::timeout(VF_XDS_BUILD_TIMEOUT, build_xds_channel(params, listener)).await
        {
            Ok(result) => result,
            Err(_) => Err(VfChannelError::Connect(anyhow::anyhow!(
                "timed out after {}s",
                VF_XDS_BUILD_TIMEOUT.as_secs()
            ))),
        };
    match result {
        Ok(channel) => {
            incr_build_metric(name, "xds", "built");
            Ok(channel)
        }
        Err(e) => {
            warn!(
                channel = name,
                listener = %listener,
                error = %e,
                "VF xDS channel build failed"
            );
            incr_build_metric(name, "xds", "failed");
            Err(e)
        }
    }
}

async fn build_vf_wily_channel(params: &VfChannelParams<'_>) -> Result<Channel, VfChannelError> {
    let channel = build_wily_channel(params).await?;
    incr_build_metric(params.name, "wily", "built");
    Ok(channel)
}

struct NamedWilyLookup {
    name: String,
    inner: WilyNsLookup,
}

#[tonic::async_trait]
impl LookupService for NamedWilyLookup {
    async fn resolve_service_endpoints(&self) -> Result<EndpointWeights, anyhow::Error> {
        self.inner
            .resolve_service_endpoints()
            .await
            .with_context(|| format!("{}: WilyNS lookup failed for {VF_WILY_PATH}", self.name))
    }
}

async fn build_wily_channel(params: &VfChannelParams<'_>) -> Result<Channel, VfChannelError> {
    let name = params.name;
    let dc = params.dc;
    let wilyns = WilyNs::new(xai_wily::WilyConfig::local_zone(dc))
        .with_context(|| format!("{name}: failed to create WilyNs"))
        .map_err(VfChannelError::Config)?;
    let lookup = NamedWilyLookup {
        name: name.to_string(),
        inner: WilyNsLookup {
            wilyns,
            wily_path: VF_WILY_PATH.to_string(),
            num_endpoints: None,
            shard_coordinate: None,
            filter_to_k8s_only: false,
        },
    };

    let source = PollingEndpointSource::new(lookup, VF_DNS_POLL_INTERVAL);

    let mut builder = ChannelBuilder::new(name)
        .tls(s2s_tls(&vf_tls_domain(dc))?)
        .endpoint_source(source)
        .aperture(params.aperture_size);
    if params.deterministic_aperture {
        builder = builder.deterministic();
    }
    builder
        .build()
        .await
        .with_context(|| format!("{name}: failed to connect to vf-service via WilyNS"))
        .map_err(VfChannelError::Connect)
}

async fn build_xds_channel(
    params: &VfChannelParams<'_>,
    listener: &str,
) -> Result<LoadBalancedChannel, VfChannelError> {
    let name = params.name;
    info!(
        channel = name,
        xds_listener = listener,
        aperture_size = params.aperture_size,
        deterministic_aperture = params.deterministic_aperture,
        lb_policy = ?params.xds_lb_policy,
        "building VF channel over xDS"
    );

    let mut builder = ChannelBuilder::new(name)
        .tls(s2s_tls(&params.xds.tls_domain)?)
        .xds(StartFrom::Lds(listener.to_string()))
        .await
        .map_err(|e| match e {
            e @ XdsChannelBuildError::Bootstrap { .. } => VfChannelError::Config(e.into()),
            e => VfChannelError::Connect(e.into()),
        })?
        .eager_resolution(VF_XDS_EAGER_RESOLUTION_TIMEOUT)
        .readiness_probe(VF_READINESS_PROBE_PORT, VF_READINESS_PROBE_PATH)
        .buffer_max_wait(DEFAULT_BUFFER_MAX_WAIT)
        .total_timeout(DEFAULT_TOTAL_TIMEOUT)
        .aperture(params.aperture_size);
    if params.deterministic_aperture {
        builder = builder.deterministic();
    }
    let channel = builder
        .build_load_balanced(params.xds_lb_policy)
        .await
        .with_context(|| format!("{name}: failed to build xDS channel for {listener}"))
        .map_err(VfChannelError::Connect)?;

    info!(
        channel = name,
        xds_listener = listener,
        "VF xDS channel ready"
    );
    Ok(channel)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENV: &str = "VF_TEST_DISCOVERY";

    #[test]
    fn discovery_parse_uses_default_unless_mode_is_explicit() {
        for default in [VfDiscovery::Wily, VfDiscovery::Xds] {
            assert_eq!(VfDiscovery::parse(ENV, None, default), default);
            assert_eq!(VfDiscovery::parse(ENV, Some(""), default), default);
            assert_eq!(VfDiscovery::parse(ENV, Some("bogus"), default), default);
        }
        assert_eq!(
            VfDiscovery::parse(ENV, Some("wily"), VfDiscovery::Xds),
            VfDiscovery::Wily
        );
        assert_eq!(
            VfDiscovery::parse(ENV, Some(" XDS "), VfDiscovery::Wily),
            VfDiscovery::Xds
        );
    }

    #[test]
    fn xds_listener_matches_live_registration() {
        assert_eq!(
            DEFAULT_VF_XDS_LISTENER,
            "xai-vf-service.prod.visibility:grpc"
        );
        assert_eq!(vf_xds_listener(None), DEFAULT_VF_XDS_LISTENER);
        assert_eq!(vf_xds_listener(Some("")), DEFAULT_VF_XDS_LISTENER);
        assert_eq!(vf_xds_listener(Some("  ")), DEFAULT_VF_XDS_LISTENER);
    }

    #[test]
    fn xds_listener_override() {
        assert_eq!(
            vf_xds_listener(Some("xai-vf-service.staging.visibility:grpc")),
            "xai-vf-service.staging.visibility:grpc"
        );
    }

    #[test]
    fn tls_domain_matches_wily_path_sni() {
        assert_eq!(
            vf_tls_domain("atla"),
            "visibility.visibility-filtering-service.prod.atla.s2s.twttr.net"
        );
    }
}
