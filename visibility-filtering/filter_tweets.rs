use crate::ingress;
use crate::metrics::{self as ft_metrics, Endpoint, RequestLevel, RequestMetrics};
use crate::params::ClientSwitches;
use crate::post_evaluator::PostEvaluator;
use crate::rules::{NoTrace, SafetyLevel, WireLevel};
use crate::treatment;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterTweetsEndpoint {
    post_evaluator: Arc<PostEvaluator>,
    client_switches: ClientSwitches,
}

impl FilterTweetsEndpoint {
    pub(crate) fn new(post_evaluator: Arc<PostEvaluator>, client_switches: ClientSwitches) -> Self {
        Self {
            post_evaluator,
            client_switches,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::VisibilityFilterRequest>,
    ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
        let safety_level =
            ingress::level(WireLevel::FilterTweets(request.get_ref().safety_level()));
        ft_metrics::observe_request(
            self.post_evaluator.metrics(),
            Endpoint::FilterTweets,
            RequestLevel::of(safety_level.as_ref().ok().copied()),
            request,
            |request, context, metrics| self.handle_inner(request, context, metrics, safety_level),
        )
        .await
    }

    async fn handle_inner(
        &self,
        request: Request<vf_pb::VisibilityFilterRequest>,
        context: xai_x_rpc::CallContext,
        metrics: RequestMetrics,
        safety_level: Result<SafetyLevel, Status>,
    ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
        let req = request.into_inner();
        ft_metrics::record_batch_size(&metrics, req.tweets.len());
        let safety_level = safety_level?;
        let raw_viewer_id = req.viewer_id;
        let request = ingress::filter_tweets(req);
        ft_metrics::record_viewer_state(metrics.plain(), raw_viewer_id, request.viewer.id());

        let request = ingress::filter_tweets_batch(
            self.post_evaluator.rule_engine(),
            &self.client_switches,
            request.viewer,
            safety_level,
            request.rows,
        );
        let (_, outcomes) = context
            .scope(self.post_evaluator.run(request, &metrics, &mut NoTrace))
            .await;
        ft_metrics::record_sent(
            &metrics,
            outcomes.iter().map(|outcome| &outcome.evaluation),
            0,
        );

        let results = outcomes
            .into_iter()
            .map(treatment::filter_tweets_result)
            .collect();

        Ok(Response::new(vf_pb::VisibilityFilterResponse { results }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::sources::InMemorySources;
    use crate::hydration::sources::Source;
    use crate::metrics::Metrics;
    use crate::rules::RuleEngine;

    async fn gizmoduck_calls(viewer_id: Option<u64>) -> usize {
        let sources = Arc::new(InMemorySources::default());
        let endpoint = FilterTweetsEndpoint::new(
            Arc::new(PostEvaluator::new(
                Arc::<InMemorySources>::clone(&sources),
                RuleEngine::for_tests(),
                Arc::default(),
                Metrics::none(),
            )),
            ClientSwitches::for_tests(),
        );
        let response = endpoint
            .handle(Request::new(vf_pb::VisibilityFilterRequest {
                safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                tweets: vec![vf_pb::TweetInput {
                    tweet_id: 2,
                    author_id: Some(20),
                }],
                viewer_id,
                country_code: None,
            }))
            .await
            .unwrap();
        assert_eq!(response.into_inner().results.len(), 1);
        sources
            .calls()
            .into_iter()
            .filter(|source| matches!(source, Source::GizmoduckViewer | Source::GizmoduckAuthor))
            .count()
    }

    #[tokio::test]
    async fn endpoint_treats_zero_viewer_id_as_logged_out() {
        let logged_out = gizmoduck_calls(None).await;
        assert_eq!(gizmoduck_calls(Some(0)).await, logged_out);
        assert_eq!(gizmoduck_calls(Some(u64::MAX)).await, logged_out);
        assert_eq!(gizmoduck_calls(Some(42)).await, logged_out + 1);
    }
}
