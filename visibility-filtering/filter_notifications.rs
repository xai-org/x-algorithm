use crate::ingress::{
    self, NotificationPost, NotificationRequest, NotificationUser, Notifications,
};
use crate::metrics::{self, Endpoint, RequestLevel, RequestMetrics};
use crate::models::{AuthorId, Evaluation, NotificationKind, TweetId};
use crate::params::ClientSwitches;
use crate::post_evaluator::PostEvaluator;
use crate::rules::SafetyLevel;
use crate::treatment;
use std::collections::HashMap;
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
                        let (by_actor, by_post) = context
                            .scope(async {
                                tokio::join!(
                                    self.users(level, &users, &request_metrics),
                                    self.recipient_posts(level, &users, &request_metrics),
                                )
                            })
                            .await;
                        self.client_switches.log_exposures(&users.facts);
                        respond(
                            &request_metrics,
                            users.rows.iter().map(|&(kind, user)| {
                                (kind, user_evaluation(user, &by_actor, &by_post))
                            }),
                        )
                    }
                    NotificationRequest::Posts(posts) => {
                        let by_post = context
                            .scope(self.posts(level, &posts, &request_metrics))
                            .await;
                        self.client_switches.log_exposures(&posts.facts);
                        respond(
                            &request_metrics,
                            posts
                                .rows
                                .iter()
                                .map(|&(kind, NotificationPost { post, actor })| {
                                    (
                                        kind,
                                        by_post.get(&(post, kind, actor)).and_then(Option::as_ref),
                                    )
                                }),
                        )
                    }
                };
                Ok(Response::new(response))
            },
        )
        .await
    }

    async fn users(
        &self,
        level: SafetyLevel,
        notifications: &Notifications<NotificationUser>,
        metrics: &RequestMetrics,
    ) -> HashMap<AuthorId, Option<Evaluation>> {
        let mut actors: Vec<AuthorId> = notifications
            .rows
            .iter()
            .map(|&(_, user)| user.actor)
            .collect();
        actors.sort_unstable_by_key(|actor| actor.0);
        actors.dedup();
        if actors.is_empty() {
            return HashMap::new();
        }
        self.post_evaluator
            .evaluate_users(level, notifications.facts.clone(), actors, metrics)
            .await
    }

    async fn recipient_posts(
        &self,
        level: SafetyLevel,
        notifications: &Notifications<NotificationUser>,
        metrics: &RequestMetrics,
    ) -> HashMap<TweetId, Result<(), Evaluation>> {
        let mut posts: Vec<TweetId> = notifications
            .rows
            .iter()
            .filter_map(|&(_, user)| user.recipient_post)
            .collect();
        posts.sort_unstable_by_key(|post| post.0);
        posts.dedup();
        if posts.is_empty() {
            return HashMap::new();
        }
        self.post_evaluator
            .resolve_recipient_posts(level, notifications.facts.clone(), posts, metrics)
            .await
    }

    async fn posts(
        &self,
        level: SafetyLevel,
        notifications: &Notifications<NotificationPost>,
        metrics: &RequestMetrics,
    ) -> HashMap<(TweetId, NotificationKind, AuthorId), Option<Evaluation>> {
        let mut posts: Vec<_> = notifications
            .rows
            .iter()
            .map(|&(kind, NotificationPost { post, actor })| (post, kind, actor))
            .collect();
        posts.sort_unstable_by_key(|&(post, kind, actor)| (post, kind, actor.0));
        posts.dedup();
        self.post_evaluator
            .evaluate_post_notifications(level, notifications.facts.clone(), &posts, metrics)
            .await
    }
}

fn user_evaluation<'a>(
    user: NotificationUser,
    by_actor: &'a HashMap<AuthorId, Option<Evaluation>>,
    by_post: &'a HashMap<TweetId, Result<(), Evaluation>>,
) -> Option<&'a Evaluation> {
    match user.recipient_post.map(|post| by_post.get(&post)) {
        None | Some(Some(Ok(()))) => by_actor.get(&user.actor).and_then(Option::as_ref),
        Some(Some(Err(evaluation))) => Some(evaluation),
        Some(None) => None,
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
