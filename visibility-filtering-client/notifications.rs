use std::num::NonZeroU64;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Result};
use xai_visibility_filtering_proto as vf_pb;
use xai_visibility_filtering_proto::visibility_filtering_service_client::VisibilityFilteringServiceClient;
use xai_x_rpc::balanced_channel::LoadBalancedChannel;
use xai_x_thrift::safety_level::SafetyLevel;

use crate::discovery::build_vf_xds_channel;
use crate::metrics::{ClientRpc, RequestMetrics};
use crate::vf_client::{
    channel_error_to_anyhow, make_vf_client, timeout_as_deadline_exceeded, within_deadline,
    xds_channel_params, XAI_VF_DEFAULT_TIMEOUT_MS,
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
#[strum(serialize_all = "snake_case")]
pub enum NotificationKind {
    Follow,
    FollowRequestAccepted,
    Mention,
    Reply,
    MediaTagged,
}

impl NotificationKind {
    pub const fn level(self) -> SafetyLevel {
        match self {
            Self::Follow => SafetyLevel::NOTIFICATIONS_FOLLOW,
            Self::FollowRequestAccepted | Self::Mention | Self::Reply | Self::MediaTagged => {
                SafetyLevel::NOTIFICATIONS_WRITER_V2
            }
        }
    }

    pub fn of_proto(kind: &vf_pb::notification::Kind) -> (Self, Option<u64>) {
        use vf_pb::notification::Kind;
        match kind {
            Kind::Follow(_) => (Self::Follow, None),
            Kind::FollowRequestAccepted(_) => (Self::FollowRequestAccepted, None),
            Kind::Mention(vf_pb::Mention { post_id }) => (Self::Mention, Some(*post_id)),
            Kind::Reply(vf_pb::Reply { post_id }) => (Self::Reply, Some(*post_id)),
            Kind::MediaTagged(vf_pb::MediaTagged { post_id }) => {
                (Self::MediaTagged, Some(*post_id))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Notification {
    Follow { actor_id: u64 },
    FollowRequestAccepted { actor_id: u64 },
}

impl Notification {
    fn into_proto(self) -> vf_pb::Notification {
        match self {
            Self::Follow { actor_id } => vf_pb::Notification {
                actor_id,
                kind: Some(vf_pb::notification::Kind::Follow(vf_pb::Follow {})),
            },
            Self::FollowRequestAccepted { actor_id } => vf_pb::Notification {
                actor_id,
                kind: Some(vf_pb::notification::Kind::FollowRequestAccepted(
                    vf_pb::FollowRequestAccepted {},
                )),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostNotification {
    pub kind: PostKind,
    pub actor_id: u64,
    pub post_id: NonZeroU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostKind {
    Mention,
    Reply,
    MediaTagged,
}

impl PostKind {
    const fn kind(self) -> NotificationKind {
        match self {
            Self::Mention => NotificationKind::Mention,
            Self::Reply => NotificationKind::Reply,
            Self::MediaTagged => NotificationKind::MediaTagged,
        }
    }
}

impl PostNotification {
    fn into_proto(self) -> vf_pb::Notification {
        use vf_pb::notification::Kind;
        let post_id = self.post_id.get();
        vf_pb::Notification {
            actor_id: self.actor_id,
            kind: Some(match self.kind {
                PostKind::Mention => Kind::Mention(vf_pb::Mention { post_id }),
                PostKind::Reply => Kind::Reply(vf_pb::Reply { post_id }),
                PostKind::MediaTagged => Kind::MediaTagged(vf_pb::MediaTagged { post_id }),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationOutcome {
    Allowed,
    Dropped {
        reason: Option<vf_pb::NotificationDropReason>,
    },
    Failed,
}

impl NotificationOutcome {
    fn decode(outcome: Option<vf_pb::notification_visibility_result::Outcome>) -> Result<Self> {
        use vf_pb::notification_visibility_result::Outcome;
        Ok(match outcome {
            Some(Outcome::Allowed(_)) => Self::Allowed,
            Some(Outcome::Dropped(dropped)) => Self::Dropped {
                reason: dropped.reason.map(|reason| {
                    vf_pb::NotificationDropReason::try_from(reason).unwrap_or_default()
                }),
            },
            Some(Outcome::Failed(_)) => Self::Failed,
            None => bail!("missing VF notification outcome"),
        })
    }
}

pub struct NotificationsClient {
    client: VisibilityFilteringServiceClient<LoadBalancedChannel>,
    timeout: Duration,
}

impl NotificationsClient {
    pub async fn connect(datacenter: &str) -> Result<Self> {
        let channel = build_vf_xds_channel(&xds_channel_params("vf-notifications", datacenter))
            .await
            .map_err(channel_error_to_anyhow)?;
        Ok(Self {
            client: make_vf_client(channel),
            timeout: Duration::from_millis(XAI_VF_DEFAULT_TIMEOUT_MS),
        })
    }

    pub async fn filter_notifications(
        &self,
        safety_level: SafetyLevel,
        recipient_id: u64,
        notifications: &[Notification],
    ) -> Result<Vec<NotificationOutcome>> {
        self.filter_notifications_recording(
            safety_level,
            recipient_id,
            notifications.iter().map(|n| n.into_proto()).collect(),
            RequestMetrics::start(ClientRpc::FilterNotifications),
        )
        .await
    }

    pub async fn filter_post_notifications(
        &self,
        recipient_id: u64,
        notifications: &[PostNotification],
    ) -> Result<Vec<NotificationOutcome>> {
        let Some(first) = notifications.first() else {
            return Ok(Vec::new());
        };
        self.filter_notifications_recording(
            first.kind.kind().level(),
            recipient_id,
            notifications.iter().map(|n| n.into_proto()).collect(),
            RequestMetrics::start(ClientRpc::FilterNotifications),
        )
        .await
    }

    async fn filter_notifications_recording(
        &self,
        safety_level: SafetyLevel,
        recipient_id: u64,
        notifications: Vec<vf_pb::Notification>,
        mut request_metrics: RequestMetrics,
    ) -> Result<Vec<NotificationOutcome>> {
        let sent = notifications.len();
        let mut rpc = tonic::Request::new(vf_pb::FilterNotificationsRequest {
            safety_level: safety_level.0,
            recipient_id,
            notifications,
        });
        rpc.set_timeout(self.timeout);
        let mut client = self.client.clone();
        let outcomes = within_deadline(self.timeout, async {
            let batch_start = Instant::now();
            let response = client
                .filter_notifications(rpc)
                .await
                .map_err(timeout_as_deadline_exceeded);
            request_metrics.batch(
                batch_start.elapsed(),
                response.as_ref().err().map(tonic::Status::code),
            );
            decode_response(sent, response?.into_inner())
        })
        .await;
        match &outcomes {
            Ok(_) => request_metrics.answered(0),
            Err(_) => request_metrics.failed(),
        }
        outcomes
    }
}

fn decode_response(
    sent: usize,
    response: vf_pb::FilterNotificationsResponse,
) -> Result<Vec<NotificationOutcome>> {
    ensure!(
        response.results.len() == sent,
        "VF notification count mismatch"
    );
    response
        .results
        .into_iter()
        .map(|result| NotificationOutcome::decode(result.outcome))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::RecordingReceiver;
    use std::sync::{Arc, Mutex};
    use tokio::sync::watch;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::codec::CompressionEncoding;
    use tonic::transport::Server;
    use tonic::{Request, Response, Status};
    use vf_pb::notification_visibility_result::Outcome;
    use vf_pb::visibility_filtering_service_server::{
        VisibilityFilteringService, VisibilityFilteringServiceServer,
    };
    use xai_x_rpc::balanced_channel::LbPolicy;
    use xai_x_rpc::endpoint_source::EndpointSource;
    use xai_x_rpc::endpoint_weights::{uniform_weights, EndpointWeights};

    const RECIPIENT: u64 = 7;

    type Answer =
        fn(&vf_pb::FilterNotificationsRequest) -> Vec<vf_pb::NotificationVisibilityResult>;

    struct AnswerStub {
        requests: Arc<Mutex<Vec<vf_pb::FilterNotificationsRequest>>>,
        answer: Answer,
    }

    #[tonic::async_trait]
    impl VisibilityFilteringService for AnswerStub {
        async fn filter_notifications(
            &self,
            request: Request<vf_pb::FilterNotificationsRequest>,
        ) -> Result<Response<vf_pb::FilterNotificationsResponse>, Status> {
            let request = request.into_inner();
            let results = (self.answer)(&request);
            self.requests.lock().unwrap().push(request);
            Ok(Response::new(vf_pb::FilterNotificationsResponse {
                results,
            }))
        }
    }

    struct HangStub;

    #[tonic::async_trait]
    impl VisibilityFilteringService for HangStub {
        async fn filter_notifications(
            &self,
            _: Request<vf_pb::FilterNotificationsRequest>,
        ) -> Result<Response<vf_pb::FilterNotificationsResponse>, Status> {
            std::future::pending().await
        }
    }

    struct UnimplementedStub;

    impl VisibilityFilteringService for UnimplementedStub {}

    struct FixedEndpoints(watch::Receiver<EndpointWeights>);

    impl EndpointSource for FixedEndpoints {
        fn watch(self) -> watch::Receiver<EndpointWeights> {
            self.0
        }
    }

    async fn serve<S: VisibilityFilteringService>(
        stub: S,
        timeout_ms: u64,
    ) -> (
        NotificationsClient,
        tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(
            Server::builder()
                .add_service(
                    VisibilityFilteringServiceServer::new(stub)
                        .accept_compressed(CompressionEncoding::Zstd)
                        .send_compressed(CompressionEncoding::Zstd),
                )
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        let (_endpoints, rx) = watch::channel(uniform_weights([addr]));
        let channel = LoadBalancedChannel::builder(FixedEndpoints(rx))
            .lb_policy(LbPolicy::least_request())
            .channel()
            .await
            .unwrap();
        let client = NotificationsClient {
            client: make_vf_client(channel),
            timeout: Duration::from_millis(timeout_ms),
        };
        (client, server)
    }

    async fn filter_against<S: VisibilityFilteringService>(
        stub: S,
        timeout_ms: u64,
        notifications: &[Notification],
    ) -> (Result<Vec<NotificationOutcome>>, Arc<RecordingReceiver>) {
        let (client, server) = serve(stub, timeout_ms).await;
        let sr = Arc::new(RecordingReceiver::default());
        let request_metrics =
            RequestMetrics::with_receiver(ClientRpc::FilterNotifications, Some(sr.clone()));
        let result = client
            .filter_notifications_recording(
                SafetyLevel::NOTIFICATIONS_FOLLOW,
                RECIPIENT,
                notifications.iter().map(|n| n.into_proto()).collect(),
                request_metrics,
            )
            .await;
        server.abort();
        (result, sr)
    }

    fn visibility_result(outcome: Option<Outcome>) -> vf_pb::NotificationVisibilityResult {
        vf_pb::NotificationVisibilityResult { outcome }
    }

    fn by_actor(
        request: &vf_pb::FilterNotificationsRequest,
    ) -> Vec<vf_pb::NotificationVisibilityResult> {
        request
            .notifications
            .iter()
            .map(|n| {
                visibility_result(Some(match n.actor_id {
                    10 => Outcome::Allowed(vf_pb::Allowed {}),
                    11 => Outcome::Dropped(vf_pb::Dropped {
                        reason: Some(vf_pb::NotificationDropReason::RecipientBlocksActor as i32),
                    }),
                    13 => Outcome::Dropped(vf_pb::Dropped { reason: Some(99) }),
                    _ => Outcome::Failed(vf_pb::Failed {}),
                }))
            })
            .collect()
    }

    #[test]
    fn follow_request_accepted_encodes_its_own_message() {
        assert_eq!(
            Notification::FollowRequestAccepted { actor_id: 10 }.into_proto(),
            vf_pb::Notification {
                actor_id: 10,
                kind: Some(vf_pb::notification::Kind::FollowRequestAccepted(
                    vf_pb::FollowRequestAccepted {}
                )),
            }
        );
    }

    #[tokio::test]
    async fn post_notifications_go_out_at_level_72_each_kind_as_its_own_message() {
        use vf_pb::notification::Kind;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (client, server) = serve(
            AnswerStub {
                requests: requests.clone(),
                answer: by_actor,
            },
            1_000,
        )
        .await;
        let post = |kind| PostNotification {
            kind,
            actor_id: 10,
            post_id: NonZeroU64::new(20).unwrap(),
        };
        let outcomes = client
            .filter_post_notifications(
                RECIPIENT,
                &[
                    post(PostKind::Mention),
                    post(PostKind::Reply),
                    post(PostKind::MediaTagged),
                ],
            )
            .await;
        server.abort();
        assert_eq!(outcomes.unwrap(), vec![NotificationOutcome::Allowed; 3]);
        let sent = |kind| vf_pb::Notification {
            actor_id: 10,
            kind: Some(kind),
        };
        assert_eq!(
            *requests.lock().unwrap(),
            vec![vf_pb::FilterNotificationsRequest {
                safety_level: 72,
                recipient_id: RECIPIENT,
                notifications: vec![
                    sent(Kind::Mention(vf_pb::Mention { post_id: 20 })),
                    sent(Kind::Reply(vf_pb::Reply { post_id: 20 })),
                    sent(Kind::MediaTagged(vf_pb::MediaTagged { post_id: 20 })),
                ],
            }]
        );
    }

    fn follows(actors: &[u64]) -> Vec<Notification> {
        actors
            .iter()
            .map(|&actor_id| Notification::Follow { actor_id })
            .collect()
    }

    #[tokio::test]
    async fn sends_the_level_number_and_follows_and_records_a_completed_call_with_outcomes_in_request_order(
    ) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (outcomes, sr) = filter_against(
            AnswerStub {
                requests: requests.clone(),
                answer: by_actor,
            },
            1_000,
            &follows(&[11, 10, 12, 11, 13]),
        )
        .await;
        assert_eq!(
            outcomes.unwrap(),
            vec![
                NotificationOutcome::Dropped {
                    reason: Some(vf_pb::NotificationDropReason::RecipientBlocksActor)
                },
                NotificationOutcome::Allowed,
                NotificationOutcome::Failed,
                NotificationOutcome::Dropped {
                    reason: Some(vf_pb::NotificationDropReason::RecipientBlocksActor)
                },
                NotificationOutcome::Dropped {
                    reason: Some(vf_pb::NotificationDropReason::Unspecified)
                },
            ]
        );
        let follow = |actor_id| vf_pb::Notification {
            actor_id,
            kind: Some(vf_pb::notification::Kind::Follow(vf_pb::Follow {})),
        };
        assert_eq!(
            *requests.lock().unwrap(),
            vec![vf_pb::FilterNotificationsRequest {
                safety_level: 246,
                recipient_id: RECIPIENT,
                notifications: vec![follow(11), follow(10), follow(12), follow(11), follow(13)],
            }]
        );
        assert_eq!(
            sr.series(),
            [
                "incr vf_client_filter_notifications|requests=completed",
                "observe_vm vf_client_filter_notifications_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_filter_notifications_latency_ms_vm",
            ]
        );
    }

    #[tokio::test]
    async fn a_short_response_a_missing_outcome_an_rpc_error_or_a_missed_deadline_fails_the_call() {
        let answer_stub = |answer| AnswerStub {
            requests: Arc::default(),
            answer,
        };
        let notifications = follows(&[10, 11]);
        let (short, sr) = filter_against(
            answer_stub(|request| {
                let mut results = by_actor(request);
                results.pop();
                results
            }),
            1_000,
            &notifications,
        )
        .await;
        assert_eq!(
            short.unwrap_err().to_string(),
            "VF notification count mismatch"
        );
        assert_eq!(
            sr.series(),
            [
                "incr vf_client_filter_notifications|requests=failed",
                "observe_vm vf_client_filter_notifications_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_filter_notifications_latency_ms_vm",
            ]
        );
        let (missing, sr) = filter_against(
            answer_stub(|request| {
                let mut results = by_actor(request);
                results[1].outcome = None;
                results
            }),
            1_000,
            &notifications,
        )
        .await;
        assert_eq!(
            missing.unwrap_err().to_string(),
            "missing VF notification outcome"
        );
        assert_eq!(
            sr.series(),
            [
                "incr vf_client_filter_notifications|requests=failed",
                "observe_vm vf_client_filter_notifications_batch_latency_ms_vm|result=success",
                "observe_vm vf_client_filter_notifications_latency_ms_vm",
            ]
        );

        let code = |result: Result<Vec<NotificationOutcome>>| {
            result
                .unwrap_err()
                .downcast_ref::<Status>()
                .map(Status::code)
        };
        let (unimplemented, sr) = filter_against(UnimplementedStub, 1_000, &notifications).await;
        assert_eq!(code(unimplemented), Some(tonic::Code::Unimplemented));
        assert_eq!(
            sr.series(),
            [
                "incr vf_client_filter_notifications_batch_error|code=other",
                "incr vf_client_filter_notifications|requests=failed",
                "observe_vm vf_client_filter_notifications_batch_latency_ms_vm|result=error",
                "observe_vm vf_client_filter_notifications_latency_ms_vm",
            ]
        );
        let (deadline, sr) = filter_against(HangStub, 50, &notifications).await;
        assert_eq!(code(deadline), Some(tonic::Code::DeadlineExceeded));
        assert_eq!(
            sr.counter("vf_client_filter_notifications|requests=failed"),
            1
        );
        assert!(sr
            .series()
            .contains(&"observe_vm vf_client_filter_notifications_latency_ms_vm".to_string()));
    }
}
