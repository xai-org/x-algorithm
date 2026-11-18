pub mod following_candidate_pipeline;
pub mod for_you_candidate_pipeline;
pub mod phoenix_candidate_pipeline;
pub mod phoenix_scores_pipeline;
pub mod ranked_following_candidate_pipeline;
pub mod reverse_chron_posts_pipeline;

use std::sync::Arc;

use crate::clients::author_brand_safety_client::{
    AuthorBrandSafetyClient, ProdAuthorBrandSafetyClient,
};
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixPredictionClient, PhoenixProduct,
};
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalClient;
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::component_library::clients::{
    build_phoenix_xds_prediction_client, build_phoenix_xds_retrieval_client_boxed,
};
use xai_redis_client::{XdsRedisClient, XdsRedisConfig};

pub use xai_candidate_pipeline::component_library::clients::{PhoenixXdsConfig, XdsP2cConfig};

pub type VmRankerXdsConfig = XdsP2cConfig;

pub(crate) async fn build_phoenix_xds_client(
    config: &PhoenixXdsConfig,
) -> Result<Arc<dyn PhoenixPredictionClient + Send + Sync>, tonic::Status> {
    static PHOENIX_XDS_CLIENT: tokio::sync::OnceCell<
        Result<Arc<dyn PhoenixPredictionClient + Send + Sync>, tonic::Status>,
    > = tokio::sync::OnceCell::const_new();
    PHOENIX_XDS_CLIENT
        .get_or_init(|| build_phoenix_xds_prediction_client(config, PhoenixProduct::Home))
        .await
        .clone()
}

pub(crate) async fn build_phoenix_xds_retrieval_client(
    config: &PhoenixXdsConfig,
) -> Result<Arc<dyn PhoenixRetrievalClient + Send + Sync>, tonic::Status> {
    static PHOENIX_XDS_RETRIEVAL_CLIENT: tokio::sync::OnceCell<
        Result<Arc<dyn PhoenixRetrievalClient + Send + Sync>, tonic::Status>,
    > = tokio::sync::OnceCell::const_new();
    PHOENIX_XDS_RETRIEVAL_CLIENT
        .get_or_init(|| build_phoenix_xds_retrieval_client_boxed(config, PhoenixProduct::Home))
        .await
        .clone()
}

pub(crate) async fn build_vm_ranker_xds_client(
    config: &VmRankerXdsConfig,
) -> Option<Arc<dyn crate::clients::vm_ranker_client::VMRankerClient>> {
    static VM_RANKER_XDS_CLIENT: tokio::sync::OnceCell<
        Option<Arc<dyn crate::clients::vm_ranker_client::VMRankerClient>>,
    > = tokio::sync::OnceCell::const_new();
    VM_RANKER_XDS_CLIENT
        .get_or_init(|| build_vm_ranker_xds_client_uncached(config))
        .await
        .clone()
}

async fn build_vm_ranker_xds_client_uncached(
    config: &VmRankerXdsConfig,
) -> Option<Arc<dyn crate::clients::vm_ranker_client::VMRankerClient>> {
    use crate::clients::vm_ranker_client::{VMRankerClient, XdsVMRankerClient};
    let policy = config.parse_lb_policy();
    let health_probe = config.parse_health_probe();
    tracing::info!(
        vm_ranker_xds_built = policy.is_some(),
        aperture_size = config.aperture_size,
        health_probe = ?health_probe,
        discovery_authority = %config.discovery_authority,
        "VMRanker: resolved xDS ranking client (startup config)"
    );
    match policy {
        Some(policy) => {
            match XdsVMRankerClient::new(
                policy,
                config.h2_stream_window_bytes,
                config.h2_connection_window_bytes,
                config.socket_buffer_bytes,
                config.aperture_size as usize,
                health_probe,
                &config.discovery_authority,
            )
            .await
            {
                Ok(client) => Some(Arc::new(client) as Arc<dyn VMRankerClient>),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "VMRanker: failed to build xDS P2C client; falling back to DNS path"
                    );
                    None
                }
            }
        }
        None => None,
    }
}

const ATLA_PHOENIX_REQUEST_CACHE_EDS: &str = "xdstp://discovery-atla/envoy.config.endpoint.v3.ClusterLoadAssignment/redis-global.candidates-inference-requests.prod.twemcache:redis";
const PDXA_PHOENIX_REQUEST_CACHE_EDS: &str = "xdstp://discovery-pdxa/envoy.config.endpoint.v3.ClusterLoadAssignment/redis-global.candidates-inference-requests.prod.twemcache:redis";

type SharedRedisClient = Arc<dyn RedisClient + Send + Sync>;

const ATLA_TEACHER_CACHE_EDS: &str = "xdstp://discovery-atla/envoy.config.endpoint.v3.ClusterLoadAssignment/redis-global.home-teacher-scores-cache.prod.cache:redis";
const PDXA_TEACHER_CACHE_EDS: &str = "xdstp://discovery-pdxa/envoy.config.endpoint.v3.ClusterLoadAssignment/redis-global.home-teacher-scores-cache.prod.cache:redis";

pub(crate) async fn teacher_cache_redis_clients() -> (SharedRedisClient, SharedRedisClient) {
    let (atla, pdxa) = tokio::join!(
        XdsRedisClient::new(XdsRedisConfig {
            eds_resource_name: ATLA_TEACHER_CACHE_EDS.into(),
        }),
        XdsRedisClient::new(XdsRedisConfig {
            eds_resource_name: PDXA_TEACHER_CACHE_EDS.into(),
        }),
    );
    (
        Arc::new(atla.expect("Failed to create xDS Redis client for atla teacher cache")),
        Arc::new(pdxa.expect("Failed to create xDS Redis client for pdxa teacher cache")),
    )
}

async fn phoenix_request_cache_redis_client(
    eds_resource_name: &str,
    datacenter: &str,
) -> SharedRedisClient {
    let client = XdsRedisClient::new(XdsRedisConfig {
        eds_resource_name: eds_resource_name.into(),
    })
    .await
    .unwrap_or_else(|e| {
        panic!("Failed to create xDS Redis client for {datacenter} phoenix cache: {e:?}")
    });
    Arc::new(client)
}

pub(crate) async fn shared_phoenix_request_cache_redis_clients(
) -> (SharedRedisClient, SharedRedisClient) {
    static PHOENIX_REQUEST_CACHE_REDIS_CLIENTS: tokio::sync::OnceCell<(
        SharedRedisClient,
        SharedRedisClient,
    )> = tokio::sync::OnceCell::const_new();
    PHOENIX_REQUEST_CACHE_REDIS_CLIENTS
        .get_or_init(|| async {
            tokio::join!(
                phoenix_request_cache_redis_client(ATLA_PHOENIX_REQUEST_CACHE_EDS, "atla"),
                phoenix_request_cache_redis_client(PDXA_PHOENIX_REQUEST_CACHE_EDS, "pdxa"),
            )
        })
        .await
        .clone()
}

pub(crate) async fn shared_author_brand_safety_client(
    datacenter: &str,
) -> Arc<dyn AuthorBrandSafetyClient> {
    static AUTHOR_BRAND_SAFETY_CLIENT: tokio::sync::OnceCell<Arc<dyn AuthorBrandSafetyClient>> =
        tokio::sync::OnceCell::const_new();
    AUTHOR_BRAND_SAFETY_CLIENT
        .get_or_init(|| async {
            Arc::new(
                ProdAuthorBrandSafetyClient::new(datacenter)
                    .await
                    .expect("Failed to create AuthorBrandSafety client"),
            ) as Arc<dyn AuthorBrandSafetyClient>
        })
        .await
        .clone()
}
