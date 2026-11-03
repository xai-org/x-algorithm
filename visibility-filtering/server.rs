use crate::dark_traffic_setup;
use crate::evaluate_tweets::EvaluateTweetsEndpoint;
use crate::filter_notifications::FilterNotificationsEndpoint;
use crate::filter_tweets::FilterTweetsEndpoint;
use crate::get_safety_labels::{GetSafetyLabelsCodec, GetSafetyLabelsEndpoint};
use crate::hydration::sources::test_users_metadata;
use crate::server_deps::{self, Comparators};
use std::convert::Infallible;
use std::sync::Arc;
use std::task::{Context, Poll};
use tonic::body::Body;
use tonic::codec::CompressionEncoding;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::metadata::MetadataMap;
use tonic::server::{Grpc, NamedService};
use tonic::{Request, Response, Status};
use xai_dark_traffic::RejectDarkTrafficLayer;
use xai_grpc_compression::GrpcZstdLayer;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_rpc::grpc_client::TlsMode;
use xai_x_service_builder::{XService, XServiceBuilder};

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    #[arg(long, default_value_t = 50051u16)]
    grpc_port: u16,
    #[arg(long, default_value_t = 9090u16)]
    metrics_port: u16,
    #[arg(long, default_value = "atla")]
    datacenter: String,
    #[arg(long, default_value = "")]
    otel_endpoint: String,
    #[arg(long)]
    test_users: bool,
}

pub async fn serve<S: XService<Config = Option<MetadataMap>>>(
    args: ServeArgs,
) -> anyhow::Result<()> {
    if args.test_users {
        crate::config::refuse_test_users_in_prod();
    }
    XServiceBuilder::new("visibility-filtering-service")
        .grpc_port(args.grpc_port)
        .metrics_port(args.metrics_port)
        .datacenter(args.datacenter)
        .otel_endpoint(args.otel_endpoint)
        .with_tls(TlsMode::server_mtls_from_env()?)
        .with_reflection(vf_pb::FILE_DESCRIPTOR_SET)
        .with_layer(dark_traffic_setup::resolve_layer())
        .with_layer(RejectDarkTrafficLayer::from_env())
        .with_layer(GrpcZstdLayer)
        .http_routes(xai_profiling::profiling_router())
        .run::<S>(args.test_users.then(test_users_metadata))
        .await
}

pub struct VFServer {
    pub(crate) evaluate_tweets: EvaluateTweetsEndpoint,
    pub(crate) filter_notifications: FilterNotificationsEndpoint,
    pub(crate) filter_tweets: FilterTweetsEndpoint,
    pub(crate) get_safety_labels: GetSafetyLabelsEndpoint,
}

#[tonic::async_trait]
impl XService for VFServer {
    type Config = Option<MetadataMap>;

    async fn build(ctx: xai_x_service_builder::ServiceContext<Option<MetadataMap>>) -> Self {
        VFServer::new(&ctx.datacenter, ctx.config.as_ref()).await
    }

    fn register(self: Arc<Self>, routes: &mut tonic::service::RoutesBuilder) {
        routes.add_service(VFRouter {
            generated_server: ACCEPTED_ENCODINGS.into_iter().fold(
                vf_pb::VisibilityFilteringServiceServer::from_arc(Arc::clone(&self)),
                |server, encoding| server.accept_compressed(encoding),
            ),
            vf_server: self,
        });
    }
}

const ACCEPTED_ENCODINGS: [CompressionEncoding; 2] =
    [CompressionEncoding::Zstd, CompressionEncoding::Gzip];

const GET_SAFETY_LABELS_PATH: &str =
    "/xai.visibility_filtering.v1.VisibilityFilteringService/GetSafetyLabels";

#[derive(Clone)]
struct VFRouter {
    generated_server: vf_pb::VisibilityFilteringServiceServer<VFServer>,
    vf_server: Arc<VFServer>,
}

impl NamedService for VFRouter {
    const NAME: &'static str = vf_pb::visibility_filtering_service_server::SERVICE_NAME;
}

impl Service<http::Request<Body>> for VFRouter {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        if request.uri().path() != GET_SAFETY_LABELS_PATH {
            return self.generated_server.call(request);
        }
        let vf_server = Arc::clone(&self.vf_server);
        Box::pin(async move {
            let method = tower::service_fn(move |request| {
                let vf_server = Arc::clone(&vf_server);
                async move { vf_server.get_safety_labels.handle(request).await }
            });
            let mut grpc = ACCEPTED_ENCODINGS
                .into_iter()
                .fold(Grpc::new(GetSafetyLabelsCodec), |grpc, encoding| {
                    grpc.accept_compressed(encoding)
                });
            Ok(grpc.unary(method, request).await)
        })
    }
}

impl VFServer {
    pub(crate) async fn new(datacenter: &str, metadata: Option<&MetadataMap>) -> Self {
        crate::config::refuse_reference();
        server_deps::build(datacenter, metadata)
            .await
            .into_server(Comparators::default())
    }
}

#[tonic::async_trait]
impl vf_pb::VisibilityFilteringService for VFServer {
    async fn evaluate_tweets(
        &self,
        request: Request<vf_pb::EvaluateTweetsRequest>,
    ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
        self.evaluate_tweets.handle(request).await
    }

    async fn filter_notifications(
        &self,
        request: Request<vf_pb::FilterNotificationsRequest>,
    ) -> Result<Response<vf_pb::FilterNotificationsResponse>, Status> {
        self.filter_notifications.handle(request).await
    }

    async fn filter_tweets(
        &self,
        request: Request<vf_pb::VisibilityFilterRequest>,
    ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
        self.filter_tweets.handle(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::sources::InMemorySources;
    use crate::metrics::Metrics;
    use crate::params::ClientSwitches;
    use crate::scenario::{tweet_data, Wire};
    use std::collections::HashMap;
    use xai_safety_label_source::testing::FakeLabels;
    use xai_visibility_filtering::vf_client::XaiVfClient;
    use xai_visibility_filtering_proto::visibility_filtering_service_client::VisibilityFilteringServiceClient;
    use xai_x_thrift::safety_result::FilteredReason;
    use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelType};
    use xai_x_thrift::tweet_service::{
        TweetFieldsResultFiltered, TweetFieldsResultFound, TweetFieldsResultNotFound,
        TweetFieldsResultState,
    };

    fn post_one_label() -> SafetyLabel {
        SafetyLabel {
            score: Some(0.5.into()),
            source: Some("botmaker".to_owned()),
            ..Default::default()
        }
    }

    fn post_one_labels() -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                3,
                vf_pb::SafetyLabel {
                    score: Some(0.5),
                    source: Some("botmaker".to_owned()),
                    ..Default::default()
                },
            )]),
        }
    }

    fn server(sources: InMemorySources) -> VFServer {
        let wire = Wire::new(Arc::new(sources));
        let post_evaluator = Arc::clone(wire.post_evaluator());
        VFServer {
            evaluate_tweets: wire.into_endpoint(),
            filter_notifications: FilterNotificationsEndpoint::new(Arc::clone(&post_evaluator)),
            filter_tweets: FilterTweetsEndpoint::new(
                post_evaluator,
                ClientSwitches::for_tests(),
                None,
            ),
            get_safety_labels: GetSafetyLabelsEndpoint::new(
                Arc::new(FakeLabels::default().into_source()),
                Metrics::none(),
            ),
        }
    }

    async fn serve(server: VFServer) -> (tonic::transport::Channel, tokio::task::JoinHandle<()>) {
        let mut routes = tonic::service::RoutesBuilder::default();
        Arc::new(server).register(&mut routes);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_routes(routes.routes())
                .serve_with_incoming(futures::stream::unfold(listener, |listener| async {
                    Some((listener.accept().await.map(|(socket, _)| socket), listener))
                }))
                .await
                .unwrap();
        });
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        (channel, handle)
    }

    #[tokio::test]
    async fn evaluate_tweets_loopback_overrides_only_a_tweetypie_found() {
        let (channel, handle) = serve(server(
            InMemorySources::default().tweet(1, 100).authors(&[100]),
        ))
        .await;
        let client = XaiVfClient::from_channel(channel);
        let found = || TweetFieldsResultState::Found(TweetFieldsResultFound::new(None));
        let protected = || {
            TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(
                FilteredReason::AuthorIsProtected(true),
            ))
        };
        let deleted =
            || TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(true, true, None));
        let not_found =
            TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(false, false, None));
        let cases = [
            (tweet_data(1, None), found(), Some(found())),
            (tweet_data(1, None), protected(), Some(protected())),
            (tweet_data(1, Some(2)), found(), None),
            (tweet_data(2, None), found(), Some(not_found)),
            (tweet_data(2, None), deleted(), Some(deleted())),
        ];
        let home = client
            .evaluate_tweets(
                vf_pb::EvaluateTweetsRequest {
                    safety_level: 8,
                    tweets: cases.iter().map(|(tweet, ..)| *tweet).collect(),
                    ..Default::default()
                },
                &Default::default(),
            )
            .await;
        handle.abort();
        assert!(handle.await.unwrap_err().is_cancelled());

        let served: Vec<_> = home
            .unwrap()
            .into_iter()
            .zip(&cases)
            .map(|(vf, (_, fetched, _))| vf.into_result_state(fetched.clone()))
            .collect();
        let expected: Vec<_> = cases.into_iter().map(|(.., expected)| expected).collect();
        assert_eq!(served, expected);
    }

    #[tokio::test]
    async fn get_safety_labels_loopback() {
        let (channel, handle) = serve(VFServer {
            get_safety_labels: GetSafetyLabelsEndpoint::new(
                Arc::new(
                    FakeLabels::default()
                        .label(1, SafetyLabelType::from(3), post_one_label())
                        .failing(2)
                        .into_source(),
                ),
                Metrics::none(),
            ),
            ..server(InMemorySources::default())
        })
        .await;
        let mut client = VisibilityFilteringServiceClient::new(channel)
            .send_compressed(CompressionEncoding::Zstd);
        let response = client
            .get_safety_labels(vf_pb::GetSafetyLabelsRequest {
                tweet_ids: vec![1, 2, 1],
            })
            .await;
        handle.abort();
        assert!(handle.await.unwrap_err().is_cancelled());

        let response = response.unwrap().into_inner();
        assert_eq!(response.results, HashMap::from([(1, post_one_labels())]));
        assert_eq!(response.failed_ids, vec![2]);
    }

    #[tokio::test]
    async fn filter_tweets_loopback() {
        let (channel, handle) = serve(server(
            InMemorySources::default().tweet(2, 20).authors(&[20]),
        ))
        .await;
        let mut client = VisibilityFilteringServiceClient::new(channel)
            .send_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Gzip);
        let tweet = |tweet_id, author_id| vf_pb::TweetInput {
            tweet_id,
            author_id,
        };
        let response = client
            .filter_tweets(vf_pb::VisibilityFilterRequest {
                safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                tweets: vec![tweet(2, Some(20)), tweet(1, None), tweet(2, Some(20))],
                viewer_id: None,
                country_code: None,
            })
            .await;
        handle.abort();
        assert!(handle.await.unwrap_err().is_cancelled());

        let results = response.unwrap().into_inner().results;
        let wire = |result: &vf_pb::TweetVisibilityResult| {
            (
                result.tweet_id,
                result.action.and_then(|a| a.kind),
                result.filtered_reason.clone().and_then(|r| r.reason),
            )
        };
        let allow = vf_pb::action::Kind::Allow(true);
        let drop = vf_pb::action::Kind::Drop(vf_pb::DropReason {});
        let unspecified = vf_pb::filtered_reason::Reason::UnspecifiedReason(true);
        assert_eq!(
            results.iter().map(wire).collect::<Vec<_>>(),
            vec![
                (2, Some(allow), None),
                (1, Some(drop), Some(unspecified)),
                (2, Some(allow), None),
            ]
        );
    }
}
