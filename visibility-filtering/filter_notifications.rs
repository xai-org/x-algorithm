use crate::ingress::{self, NotificationRequest};
use crate::metrics::{self, Endpoint, RequestLevel, RequestMetrics};
use crate::models::{Evaluation, NotificationKind};
use crate::params::ClientSwitches;
use crate::post_evaluator::PostEvaluator;
use crate::treatment;
use std::sync::Arc;
use strum::VariantArray;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterNotificationsEndpoint {
    post_evaluator: Arc<PostEvaluator>,
    client_switches: ClientSwitches,
}

impl FilterNotificationsEndpoint {
    pub(crate) fn new(post_evaluator: Arc<PostEvaluator>, client_switches: ClientSwitches) -> Self {
        Self {
            post_evaluator,
            client_switches,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::FilterNotificationsRequest>,
    ) -> Result<Response<vf_pb::FilterNotificationsResponse>, Status> {
        let level = ingress::notification_level(request.get_ref().safety_level);
        let request_level = RequestLevel::of(level.as_ref().ok().copied());
        metrics::observe_request(
            self.post_evaluator.metrics(),
            Endpoint::FilterNotifications,
            request_level,
            request,
            |request, context, request_metrics| async move {
                metrics::record_batch_size(&request_metrics, request.get_ref().notifications.len());
                let (level, notifications) = ingress::filter_notifications(
                    request.get_ref(),
                    level,
                    self.post_evaluator.rule_engine(),
                    &self.client_switches,
                )?;
                let response = match notifications {
                    NotificationRequest::Users(users) => {
                        let judged = context
                            .scope(self.post_evaluator.evaluate_user_notifications(
                                level,
                                &users.facts,
                                &users.rows,
                                &request_metrics,
                            ))
                            .await;
                        self.client_switches.log_exposures(&users.facts);
                        respond(
                            &request_metrics,
                            users
                                .rows
                                .iter()
                                .map(|&(kind, user)| (kind, judged.evaluation(user))),
                        )
                    }
                    NotificationRequest::Posts(posts) => {
                        let evaluations = context
                            .scope(self.post_evaluator.evaluate_post_notifications(
                                level,
                                &posts.facts,
                                &posts.rows,
                                &request_metrics,
                            ))
                            .await;
                        self.client_switches.log_exposures(&posts.facts);
                        respond(
                            &request_metrics,
                            posts
                                .rows
                                .iter()
                                .zip(&evaluations)
                                .map(|(&(kind, _), evaluation)| (kind, evaluation.as_ref())),
                        )
                    }
                };
                Ok(Response::new(response))
            },
        )
        .await
    }
}

fn respond<'a>(
    metrics: &RequestMetrics,
    rows: impl Iterator<Item = (NotificationKind, Option<&'a Evaluation>)>,
) -> vf_pb::FilterNotificationsResponse {
    let rows: Vec<_> = rows.collect();
    metrics::record_sent(
        metrics,
        rows.iter().filter_map(|&(_, evaluation)| evaluation),
        rows.iter()
            .filter(|(_, evaluation)| evaluation.is_none())
            .count(),
    );
    let results: Vec<_> = rows
        .into_iter()
        .map(|(kind, evaluation)| (kind, treatment::notification_result(evaluation)))
        .collect();
    for &kind in NotificationKind::VARIANTS {
        let mut labels = results
            .iter()
            .filter(|&&(row_kind, _)| row_kind == kind)
            .map(|&(_, (_, labels))| labels)
            .peekable();
        if labels.peek().is_some() {
            metrics::record_notification_results(metrics, kind.into(), labels);
        }
    }
    vf_pb::FilterNotificationsResponse {
        results: results.into_iter().map(|(_, (result, _))| result).collect(),
    }
}

#[cfg(test)]
mod tests;
