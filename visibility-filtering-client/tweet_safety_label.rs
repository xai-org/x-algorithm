use crate::discovery::{
    build_vf_channel, env_override, visibility_tls_domain, VfChannel, VfChannelError,
    VfChannelParams, VfDiscovery, XdsTarget,
};
use crate::metrics::{ClientRpc, RequestMetrics};
use anyhow::anyhow;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tonic::codec::CompressionEncoding;
use tonic::{async_trait, Status};
use tracing::warn;
use xai_safety_label_store::types::SafetyLabelMap;
use xai_visibility_filtering_proto as vf_pb;
use xai_visibility_filtering_proto::safety_label_store_service_client::SafetyLabelStoreServiceClient;
use xai_visibility_filtering_proto::visibility_filtering_service_client::VisibilityFilteringServiceClient;
use xai_x_rpc::balanced_channel::{LbPolicy, LoadBalancedChannel};
use xai_x_thrift::tweet_safety_label::{
    AgentTool, BotMakerAction, GrokAnnotationAction, GrokAnnotationSource, SafetyLabel,
    SafetyLabelSource, SafetyLabelType, ToolAction,
};

const DEFAULT_TIMEOUT_MS: u64 = 200;
const DEFAULT_MAX_BATCH_SIZE: usize = 50;

pub const LATENCY_VM_METRIC: &str = "vf_client_get_safety_labels_latency_ms_vm";
pub const BATCH_LATENCY_VM_METRIC: &str = "vf_client_get_safety_labels_batch_latency_ms_vm";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SafetyLabelFailure {
    LookupFailed,
    BatchFailed,
}

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelsBatch {
    pub labels: HashMap<u64, SafetyLabelMap>,
    pub failures: HashMap<u64, SafetyLabelFailure>,
}

#[async_trait]
pub trait TweetSafetyLabelClient: Send + Sync {
    async fn get_safety_labels(&self, tweet_ids: Vec<u64>) -> Result<SafetyLabelsBatch, Status>;
}

pub struct ProdTweetSafetyLabelClient {
    client: ServiceClient,
    max_batch_size: usize,
    timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SafetyLabelsBackend {
    Vf,
    Store,
}

#[derive(Clone)]
enum SafetyLabelsRpc<T> {
    Vf(VisibilityFilteringServiceClient<T>),
    Store(SafetyLabelStoreServiceClient<T>),
}

impl<T> SafetyLabelsRpc<T>
where
    T: tonic::client::GrpcService<tonic::body::Body>,
    T::Error: Into<tonic::codegen::StdError>,
    T::ResponseBody: tonic::codegen::Body<Data = tonic::codegen::Bytes> + Send + 'static,
    <T::ResponseBody as tonic::codegen::Body>::Error: Into<tonic::codegen::StdError> + Send,
{
    fn new(backend: SafetyLabelsBackend, channel: T) -> Self {
        match backend {
            SafetyLabelsBackend::Vf => Self::Vf(
                VisibilityFilteringServiceClient::new(channel)
                    .send_compressed(CompressionEncoding::Zstd)
                    .accept_compressed(CompressionEncoding::Zstd),
            ),
            SafetyLabelsBackend::Store => Self::Store(
                SafetyLabelStoreServiceClient::new(channel)
                    .send_compressed(CompressionEncoding::Zstd)
                    .accept_compressed(CompressionEncoding::Zstd),
            ),
        }
    }

    async fn get_safety_labels(
        &mut self,
        request: tonic::Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<tonic::Response<vf_pb::GetSafetyLabelsResponse>, Status> {
        match self {
            Self::Vf(c) => c.get_safety_labels(request).await,
            Self::Store(c) => c.get_safety_labels(request).await,
        }
    }
}

#[derive(Clone)]
enum ServiceClient {
    Wily(SafetyLabelsRpc<tonic::transport::Channel>),
    Xds(SafetyLabelsRpc<LoadBalancedChannel>),
}

impl ServiceClient {
    async fn get_safety_labels(
        &mut self,
        request: tonic::Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<tonic::Response<vf_pb::GetSafetyLabelsResponse>, Status> {
        match self {
            Self::Wily(c) => c.get_safety_labels(request).await,
            Self::Xds(c) => c.get_safety_labels(request).await,
        }
    }
}

const VF_SAFETY_LABELS_BACKEND_ENV: &str = "VF_SAFETY_LABELS_BACKEND";
const VF_SAFETY_LABELS_DISCOVERY_ENV: &str = "VF_SAFETY_LABELS_DISCOVERY";
const VF_SAFETY_LABELS_APERTURE_SIZE: usize = 16;
const STORE_XDS_LISTENER_ENV: &str = "SAFETY_LABEL_STORE_CLIENT_XDS_LISTENER";
const STORE_TLS_NAME_ENV: &str = "SAFETY_LABEL_STORE_CLIENT_TLS_NAME";
const DEFAULT_STORE_XDS_LISTENER: &str = "xai-safety-label-store.prod.visibility:grpc";

fn channel_params(
    dc: &str,
    env: impl Fn(&str) -> Option<String>,
) -> Result<(SafetyLabelsBackend, VfChannelParams<'_>), VfChannelError> {
    let backend = match env(VF_SAFETY_LABELS_BACKEND_ENV)
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("" | "vf") => SafetyLabelsBackend::Vf,
        Some("store") => SafetyLabelsBackend::Store,
        Some(other) => {
            return Err(VfChannelError::Config(anyhow!(
                "unknown {VF_SAFETY_LABELS_BACKEND_ENV} value {other:?}; expected vf or store"
            )));
        }
    };
    let discovery = VfDiscovery::parse(
        VF_SAFETY_LABELS_DISCOVERY_ENV,
        env(VF_SAFETY_LABELS_DISCOVERY_ENV).as_deref(),
        match backend {
            SafetyLabelsBackend::Vf => VfDiscovery::Wily,
            SafetyLabelsBackend::Store => VfDiscovery::Xds,
        },
    );
    let (name, xds) = match backend {
        SafetyLabelsBackend::Vf => ("vf-safety-labels", XdsTarget::vf(dc)),
        SafetyLabelsBackend::Store => {
            if discovery == VfDiscovery::Wily {
                return Err(VfChannelError::Config(anyhow!(
                    "{VF_SAFETY_LABELS_BACKEND_ENV}=store is reached only over xDS; \
                     unset {VF_SAFETY_LABELS_DISCOVERY_ENV} or set it to xds"
                )));
            }
            let store = XdsTarget {
                listener: env_override(
                    env(STORE_XDS_LISTENER_ENV).as_deref(),
                    DEFAULT_STORE_XDS_LISTENER.to_string(),
                ),
                tls_domain: env_override(
                    env(STORE_TLS_NAME_ENV).as_deref(),
                    visibility_tls_domain("xai-safety-label-store", dc),
                ),
            };
            ("safety-label-store", store)
        }
    };
    let params = VfChannelParams {
        name,
        dc,
        discovery,
        xds,
        aperture_size: VF_SAFETY_LABELS_APERTURE_SIZE,
        deterministic_aperture: env("APP_ENV").as_deref() == Some("prod"),
        xds_lb_policy: LbPolicy::least_request(),
    };
    Ok((backend, params))
}

fn channel_error_to_status(e: VfChannelError) -> Status {
    match e {
        VfChannelError::Config(e) => Status::internal(format!("{e:#}")),
        VfChannelError::Connect(e) => Status::unavailable(format!("failed to connect: {e:#}")),
    }
}

impl ProdTweetSafetyLabelClient {
    pub async fn new(dc: &str) -> Result<Self, Status> {
        let (backend, params) =
            channel_params(dc, |name| std::env::var(name).ok()).map_err(channel_error_to_status)?;
        let client = match build_vf_channel(&params)
            .await
            .map_err(channel_error_to_status)?
        {
            VfChannel::Wily(ch) => ServiceClient::Wily(SafetyLabelsRpc::new(backend, ch)),
            VfChannel::Xds(ch) => ServiceClient::Xds(SafetyLabelsRpc::new(backend, ch)),
        };

        Ok(Self {
            client,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
        })
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout = if timeout_ms == 0 {
            Duration::from_millis(DEFAULT_TIMEOUT_MS)
        } else {
            Duration::from_millis(timeout_ms)
        };
        self
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = if size == 0 {
            DEFAULT_MAX_BATCH_SIZE
        } else {
            size
        };
        self
    }
}

fn proto_to_safety_label(label: &vf_pb::SafetyLabel) -> SafetyLabel {
    use xai_x_thrift::tweet_safety_label::PerspectivalUser;
    SafetyLabel {
        score: label.score.map(thrift::OrderedFloat::from),
        applicable_users: if label.applicable_users.is_empty() {
            None
        } else {
            Some(
                label
                    .applicable_users
                    .iter()
                    .map(|&uid| PerspectivalUser::new(uid))
                    .collect(),
            )
        },
        holdback_experiment: label.holdback_experiment,
        source: label.source.clone(),
        safety_label_source: label.safety_label_source.as_ref().map(|src| match src {
            vf_pb::safety_label::SafetyLabelSource::BotmakerAction(a) => {
                SafetyLabelSource::BotMakerAction(BotMakerAction::new(a.rule_id))
            }
            vf_pb::safety_label::SafetyLabelSource::ToolAction(a) => SafetyLabelSource::ToolAction(
                ToolAction::new(AgentTool::from(a.agent_tool), a.actor_ldap.clone()),
            ),
            vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(a) => {
                SafetyLabelSource::GrokAnnotationAction(GrokAnnotationAction::new(
                    GrokAnnotationSource::from(a.source),
                ))
            }
        }),
        created_at_msec: label.created_at_msec,
        expires_at_msec: label.expires_at_msec,
        applicable_countries: if label.applicable_countries.is_empty() {
            None
        } else {
            Some(label.applicable_countries.clone())
        },
    }
}

pub(crate) fn proto_to_safety_label_map(proto: &vf_pb::SafetyLabelMap) -> SafetyLabelMap {
    proto
        .labels
        .iter()
        .map(|(&type_id, label)| (SafetyLabelType::from(type_id), proto_to_safety_label(label)))
        .collect()
}

#[async_trait]
impl TweetSafetyLabelClient for ProdTweetSafetyLabelClient {
    async fn get_safety_labels(&self, tweet_ids: Vec<u64>) -> Result<SafetyLabelsBatch, Status> {
        if tweet_ids.is_empty() {
            return Ok(SafetyLabelsBatch {
                labels: HashMap::new(),
                failures: HashMap::new(),
            });
        }

        let timeout = self.timeout;
        let mut request_metrics = RequestMetrics::start(ClientRpc::GetSafetyLabels);

        let futures: Vec<_> = tweet_ids
            .chunks(self.max_batch_size)
            .map(|batch| {
                let batch_ids = batch.to_vec();
                let mut client = self.client.clone();
                async move {
                    let batch_start = Instant::now();
                    let mut request = tonic::Request::new(vf_pb::GetSafetyLabelsRequest {
                        tweet_ids: batch_ids.clone(),
                    });
                    request.set_timeout(timeout);
                    let result = client.get_safety_labels(request).await.map(|response| {
                        let response = response.into_inner();
                        let labels = response
                            .results
                            .iter()
                            .map(|(&id, proto)| (id, proto_to_safety_label_map(proto)))
                            .collect::<HashMap<u64, SafetyLabelMap>>();
                        let failures = response
                            .failed_ids
                            .into_iter()
                            .map(|id| (id, SafetyLabelFailure::LookupFailed))
                            .collect::<HashMap<u64, SafetyLabelFailure>>();

                        SafetyLabelsBatch { labels, failures }
                    });
                    (result, batch_start.elapsed())
                }
            })
            .collect();

        let responses = futures::future::join_all(futures).await;

        let mut labels = HashMap::with_capacity(tweet_ids.len());
        let mut failures = HashMap::new();

        for (i, (response, batch_elapsed)) in responses.into_iter().enumerate() {
            match response {
                Ok(batch) => {
                    labels.extend(batch.labels);
                    failures.extend(batch.failures);
                    request_metrics.batch(batch_elapsed, None);
                }
                Err(status) => {
                    warn!(batch = i, %status, "get_safety_labels batch failed");
                    request_metrics.batch(batch_elapsed, Some(status.code()));
                    let start = i * self.max_batch_size;
                    let end = usize::min(start + self.max_batch_size, tweet_ids.len());
                    for id in &tweet_ids[start..end] {
                        failures.insert(*id, SafetyLabelFailure::BatchFailed);
                    }
                }
            }
        }

        request_metrics.answered(failures.len());
        Ok(SafetyLabelsBatch { labels, failures })
    }
}

pub struct MockTweetSafetyLabelClient;

#[async_trait]
impl TweetSafetyLabelClient for MockTweetSafetyLabelClient {
    async fn get_safety_labels(&self, tweet_ids: Vec<u64>) -> Result<SafetyLabelsBatch, Status> {
        Ok(SafetyLabelsBatch {
            labels: tweet_ids
                .into_iter()
                .map(|id| (id, SafetyLabelMap::default()))
                .collect(),
            failures: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::service::Routes;
    use tonic::transport::Server;
    use tonic::{Request, Response};
    use xai_visibility_filtering_proto::safety_label_store_service_server::{
        SafetyLabelStoreService, SafetyLabelStoreServiceServer,
    };
    use xai_visibility_filtering_proto::visibility_filtering_service_server::{
        VisibilityFilteringService, VisibilityFilteringServiceServer,
    };

    #[derive(Clone)]
    struct TestVisibilityFilteringService {
        requests: Arc<Mutex<Vec<Vec<u64>>>>,
        responses: Arc<Mutex<Vec<Result<vf_pb::GetSafetyLabelsResponse, Status>>>>,
    }

    impl Default for TestVisibilityFilteringService {
        fn default() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                responses: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[tonic::async_trait]
    impl VisibilityFilteringService for TestVisibilityFilteringService {
        async fn evaluate_tweets(
            &self,
            _: Request<vf_pb::EvaluateTweetsRequest>,
        ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
            Err(Status::unimplemented("unused"))
        }

        async fn filter_tweets(
            &self,
            _request: Request<vf_pb::VisibilityFilterRequest>,
        ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
            unimplemented!()
        }

        async fn get_safety_labels(
            &self,
            request: Request<vf_pb::GetSafetyLabelsRequest>,
        ) -> Result<Response<vf_pb::GetSafetyLabelsResponse>, Status> {
            let tweet_ids = request.into_inner().tweet_ids;
            self.requests.lock().await.push(tweet_ids.clone());

            let mut responses = self.responses.lock().await;
            if !responses.is_empty() {
                return responses.remove(0).map(Response::new);
            }

            let results = tweet_ids
                .into_iter()
                .map(|id| (id, vf_pb::SafetyLabelMap::default()))
                .collect();
            Ok(Response::new(vf_pb::GetSafetyLabelsResponse {
                results,
                failed_ids: Vec::new(),
            }))
        }
    }

    #[derive(Clone, Default)]
    struct TestSafetyLabelStore {
        requests: Arc<Mutex<Vec<Vec<u64>>>>,
    }

    #[tonic::async_trait]
    impl SafetyLabelStoreService for TestSafetyLabelStore {
        async fn get_safety_labels(
            &self,
            request: Request<vf_pb::GetSafetyLabelsRequest>,
        ) -> Result<Response<vf_pb::GetSafetyLabelsResponse>, Status> {
            let tweet_ids = request.into_inner().tweet_ids;
            self.requests.lock().await.push(tweet_ids.clone());
            Ok(Response::new(vf_pb::GetSafetyLabelsResponse {
                results: tweet_ids
                    .into_iter()
                    .map(|id| (id, vf_pb::SafetyLabelMap::default()))
                    .collect(),
                failed_ids: Vec::new(),
            }))
        }
    }

    async fn serve(routes: Routes) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let incoming = TcpListenerStream::new(listener);
        let handle = tokio::spawn(async move {
            Server::builder()
                .add_routes(routes)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        (addr, handle)
    }

    async fn start_test_server(
        service: TestVisibilityFilteringService,
    ) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        serve(Routes::new(
            VisibilityFilteringServiceServer::new(service)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        ))
        .await
    }

    async fn test_client_for(
        backend: SafetyLabelsBackend,
        addr: SocketAddr,
    ) -> ProdTweetSafetyLabelClient {
        let endpoint = format!("http://{addr}");
        let channel = tonic::transport::Endpoint::from_shared(endpoint)
            .unwrap()
            .connect()
            .await
            .unwrap();
        ProdTweetSafetyLabelClient {
            client: ServiceClient::Wily(SafetyLabelsRpc::new(backend, channel)),
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
        }
    }

    async fn test_client(addr: SocketAddr) -> ProdTweetSafetyLabelClient {
        test_client_for(SafetyLabelsBackend::Vf, addr).await
    }

    fn env_of<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn channel_params_pick_the_backend_discovery_and_target_from_env() {
        let vf_target = XdsTarget {
            listener: "xai-vf-service.prod.visibility:grpc".to_string(),
            tls_domain: "visibility.visibility-filtering-service.prod.atla.s2s.twttr.net"
                .to_string(),
        };
        for vars in [&[][..], &[("VF_SAFETY_LABELS_BACKEND", " VF ")][..]] {
            let (backend, params) = channel_params("atla", env_of(vars)).unwrap();
            assert_eq!(backend, SafetyLabelsBackend::Vf);
            assert_eq!(params.name, "vf-safety-labels");
            assert_eq!(params.discovery, VfDiscovery::Wily);
            assert_eq!(params.xds, vf_target);
        }
        let (_, params) =
            channel_params("atla", env_of(&[("VF_SAFETY_LABELS_DISCOVERY", "xds")])).unwrap();
        assert_eq!(params.discovery, VfDiscovery::Xds);

        for vars in [
            &[("VF_SAFETY_LABELS_BACKEND", "store")][..],
            &[
                ("VF_SAFETY_LABELS_BACKEND", "store"),
                ("VF_SAFETY_LABELS_DISCOVERY", "xds"),
                (
                    "VF_CLIENT_XDS_LISTENER",
                    "xai-vf-service.staging.visibility:grpc",
                ),
            ][..],
        ] {
            let (backend, params) = channel_params("pdxa", env_of(vars)).unwrap();
            assert_eq!(backend, SafetyLabelsBackend::Store);
            assert_eq!(params.name, "safety-label-store");
            assert_eq!(params.discovery, VfDiscovery::Xds);
            assert_eq!(
                params.xds,
                XdsTarget {
                    listener: "xai-safety-label-store.prod.visibility:grpc".to_string(),
                    tls_domain: "visibility.xai-safety-label-store.prod.pdxa.s2s.twttr.net"
                        .to_string(),
                }
            );
        }
        let store_overrides = XdsTarget {
            listener: "xai-safety-label-store.staging.visibility:grpc".to_string(),
            tls_domain: "visibility.xai-safety-label-store.staging.atla.s2s.twttr.net".to_string(),
        };
        let (_, params) = channel_params(
            "atla",
            env_of(&[
                ("VF_SAFETY_LABELS_BACKEND", "store"),
                (
                    "SAFETY_LABEL_STORE_CLIENT_XDS_LISTENER",
                    store_overrides.listener.as_str(),
                ),
                (
                    "SAFETY_LABEL_STORE_CLIENT_TLS_NAME",
                    store_overrides.tls_domain.as_str(),
                ),
            ]),
        )
        .unwrap();
        assert_eq!(params.xds, store_overrides);

        let Err(VfChannelError::Config(e)) =
            channel_params("atla", env_of(&[("VF_SAFETY_LABELS_BACKEND", "stroe")]))
        else {
            panic!("an unknown backend is a config error");
        };
        let message = format!("{e:#}");
        assert!(message.contains("VF_SAFETY_LABELS_BACKEND"), "{message}");
        assert!(message.contains("stroe"), "{message}");

        let Err(VfChannelError::Config(e)) = channel_params(
            "atla",
            env_of(&[
                ("VF_SAFETY_LABELS_BACKEND", "store"),
                ("VF_SAFETY_LABELS_DISCOVERY", "wily"),
            ]),
        ) else {
            panic!("the store is reached only over xDS");
        };
        assert!(format!("{e:#}").contains("xDS"), "{e:#}");
    }

    #[tokio::test]
    async fn store_backend_calls_safety_label_store_service() {
        let store = TestSafetyLabelStore::default();
        let requests = store.requests.clone();
        let (addr, handle) = serve(Routes::new(
            SafetyLabelStoreServiceServer::new(store)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        ))
        .await;
        let client = test_client_for(SafetyLabelsBackend::Store, addr).await;

        let result = client.get_safety_labels(vec![1, 2]).await.unwrap();

        assert_eq!(*requests.lock().await, vec![vec![1, 2]]);
        assert_eq!(result.labels.len(), 2);
        assert!(result.failures.is_empty());
        handle.abort();
    }

    #[test]
    fn proto_botmaker_action_to_thrift() {
        let proto = vf_pb::SafetyLabel {
            safety_label_source: Some(vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                vf_pb::BotmakerAction { rule_id: 42 },
            )),
            ..Default::default()
        };
        let label = proto_to_safety_label(&proto);
        assert_eq!(
            label.safety_label_source,
            Some(SafetyLabelSource::BotMakerAction(BotMakerAction::new(42)))
        );
    }

    #[test]
    fn proto_tool_action_to_thrift() {
        let proto = vf_pb::SafetyLabel {
            safety_label_source: Some(vf_pb::safety_label::SafetyLabelSource::ToolAction(
                vf_pb::ToolAction {
                    agent_tool: 15,
                    actor_ldap: "someone".to_string(),
                },
            )),
            ..Default::default()
        };
        let label = proto_to_safety_label(&proto);
        assert_eq!(
            label.safety_label_source,
            Some(SafetyLabelSource::ToolAction(ToolAction::new(
                AgentTool::BME,
                "someone".to_string(),
            )))
        );
    }

    #[test]
    fn proto_grok_annotation_action_to_thrift() {
        let proto = vf_pb::SafetyLabel {
            safety_label_source: Some(
                vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(
                    vf_pb::GrokAnnotationAction { source: 1 },
                ),
            ),
            ..Default::default()
        };
        let label = proto_to_safety_label(&proto);
        assert_eq!(
            label.safety_label_source,
            Some(SafetyLabelSource::GrokAnnotationAction(
                GrokAnnotationAction::new(GrokAnnotationSource::GROX_PTOS)
            ))
        );
    }

    #[test]
    fn proto_no_source_to_thrift() {
        let proto = vf_pb::SafetyLabel::default();
        let label = proto_to_safety_label(&proto);
        assert_eq!(label.safety_label_source, None);
    }

    #[tokio::test]
    async fn get_safety_labels_returns_empty_map_for_empty_input() {
        let service = TestVisibilityFilteringService::default();
        let requests = service.requests.clone();
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await;

        let result = client.get_safety_labels(vec![]).await.unwrap();

        assert!(result.labels.is_empty());
        assert!(result.failures.is_empty());
        assert!(requests.lock().await.is_empty());
        handle.abort();
    }

    #[tokio::test]
    async fn get_safety_labels_uses_custom_batch_size() {
        let service = TestVisibilityFilteringService::default();
        let requests = service.requests.clone();
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await.with_max_batch_size(3);

        let tweet_ids: Vec<u64> = (1..=10).collect();
        let result = client.get_safety_labels(tweet_ids.clone()).await.unwrap();

        let requests = requests.lock().await.clone();
        let request_sets: HashSet<Vec<u64>> = requests.into_iter().collect();
        let expected_sets: HashSet<Vec<u64>> =
            [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9], vec![10]]
                .into_iter()
                .collect();

        assert_eq!(request_sets, expected_sets);
        assert_eq!(result.labels.len(), tweet_ids.len());
        assert!(result.failures.is_empty());
        for id in tweet_ids {
            assert!(result.labels.contains_key(&id));
        }
        handle.abort();
    }

    #[tokio::test]
    async fn with_max_batch_size_zero_falls_back_to_default_batch_size() {
        let service = TestVisibilityFilteringService::default();
        let requests = service.requests.clone();
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await.with_max_batch_size(0);

        let tweet_ids: Vec<u64> = (1..=(DEFAULT_MAX_BATCH_SIZE as u64 + 1)).collect();
        let result = client.get_safety_labels(tweet_ids.clone()).await.unwrap();

        let requests = requests.lock().await.clone();
        let request_sets: HashSet<Vec<u64>> = requests.into_iter().collect();
        let expected_sets: HashSet<Vec<u64>> = [
            (1..=DEFAULT_MAX_BATCH_SIZE as u64).collect(),
            vec![DEFAULT_MAX_BATCH_SIZE as u64 + 1],
        ]
        .into_iter()
        .collect();

        assert_eq!(request_sets, expected_sets);
        assert_eq!(result.labels.len(), tweet_ids.len());
        assert!(result.failures.is_empty());
        for id in tweet_ids {
            assert!(result.labels.contains_key(&id));
        }
        handle.abort();
    }

    #[tokio::test]
    async fn get_safety_labels_marks_transport_failed_batch_ids() {
        let service = TestVisibilityFilteringService {
            responses: Arc::new(Mutex::new(vec![Err(Status::unavailable("vf down"))])),
            ..Default::default()
        };
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await;

        let result = client.get_safety_labels(vec![1, 2]).await.unwrap();

        assert!(result.labels.is_empty());
        assert_eq!(
            result.failures,
            HashMap::from([
                (1, SafetyLabelFailure::BatchFailed),
                (2, SafetyLabelFailure::BatchFailed),
            ])
        );
        handle.abort();
    }

    #[tokio::test]
    async fn get_safety_labels_merges_batches() {
        let service = TestVisibilityFilteringService {
            responses: Arc::new(Mutex::new(vec![
                Ok(vf_pb::GetSafetyLabelsResponse {
                    results: HashMap::from([(1, vf_pb::SafetyLabelMap::default())]),
                    failed_ids: vec![2],
                }),
                Ok(vf_pb::GetSafetyLabelsResponse {
                    results: HashMap::from([(3, vf_pb::SafetyLabelMap::default())]),
                    failed_ids: vec![4],
                }),
            ])),
            ..Default::default()
        };
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await.with_max_batch_size(2);

        let result = client.get_safety_labels(vec![1, 2, 3, 4]).await.unwrap();

        assert_eq!(result.labels.len(), 2);
        assert!(result.labels.contains_key(&1));
        assert!(result.labels.contains_key(&3));
        assert_eq!(
            result.failures,
            HashMap::from([
                (2, SafetyLabelFailure::LookupFailed),
                (4, SafetyLabelFailure::LookupFailed),
            ])
        );
        handle.abort();
    }
}
