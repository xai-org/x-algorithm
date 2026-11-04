use crate::ingress;
use crate::metrics::{self as ft_metrics, Endpoint, RequestLevel, RequestMetrics};
use crate::models::RequestViewer;
use crate::params::ClientSwitches;
use crate::post_evaluator::{PostEvaluation, PostEvaluator};
use crate::rules::{NoTrace, SafetyLevel, WireLevel};
use crate::treatment;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub(crate) type FinishComparison = Box<dyn FnOnce(&[PostEvaluation]) + Send>;

pub(crate) trait Comparator: Send + Sync {
    fn begin(&self, viewer: &RequestViewer, safety_level: SafetyLevel) -> Option<FinishComparison>;
}

pub struct FilterTweetsEndpoint {
    post_evaluator: Arc<PostEvaluator>,
    client_switches: ClientSwitches,
    comparator: Option<Box<dyn Comparator>>,
}

impl FilterTweetsEndpoint {
    pub(crate) fn new(
        post_evaluator: Arc<PostEvaluator>,
        client_switches: ClientSwitches,
        comparator: Option<Box<dyn Comparator>>,
    ) -> Self {
        Self {
            post_evaluator,
            client_switches,
            comparator,
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

        let finish_comparison = self
            .comparator
            .as_ref()
            .and_then(|comparator| comparator.begin(&request.viewer, safety_level));

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

        if let Some(finish_comparison) = finish_comparison {
            finish_comparison(&outcomes);
        }

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
    use std::sync::Mutex;

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
            None,
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

    #[derive(Default)]
    struct ViewerIds(Arc<Mutex<Vec<Option<u64>>>>);

    impl Comparator for ViewerIds {
        fn begin(
            &self,
            viewer: &RequestViewer,
            _safety_level: SafetyLevel,
        ) -> Option<FinishComparison> {
            self.0.lock().unwrap().push(viewer.id());
            None
        }
    }

    #[tokio::test]
    async fn comparator_sees_the_normalized_viewer_id() {
        let viewer_ids = ViewerIds::default();
        let seen = Arc::clone(&viewer_ids.0);
        let endpoint = FilterTweetsEndpoint::new(
            Arc::new(PostEvaluator::new(
                Arc::new(InMemorySources::default()),
                RuleEngine::for_tests(),
                Arc::default(),
                Metrics::none(),
            )),
            ClientSwitches::for_tests(),
            Some(Box::new(viewer_ids)),
        );
        for viewer_id in [Some(0), Some(42)] {
            endpoint
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
        }
        assert_eq!(*seen.lock().unwrap(), vec![None, Some(42)]);
    }

    #[derive(Default)]
    struct FinishedResults(Arc<Mutex<Vec<Vec<vf_pb::TweetVisibilityResult>>>>);

    impl Comparator for FinishedResults {
        fn begin(
            &self,
            _viewer: &RequestViewer,
            _safety_level: SafetyLevel,
        ) -> Option<FinishComparison> {
            let finished = Arc::clone(&self.0);
            Some(Box::new(move |outcomes: &[PostEvaluation]| {
                finished.lock().unwrap().push(
                    outcomes
                        .iter()
                        .cloned()
                        .map(treatment::filter_tweets_result)
                        .collect(),
                );
            }))
        }
    }

    #[tokio::test]
    async fn comparator_finishes_with_the_served_outcomes() {
        let comparator = FinishedResults::default();
        let finished = Arc::clone(&comparator.0);
        let endpoint = FilterTweetsEndpoint::new(
            Arc::new(PostEvaluator::new(
                Arc::new(InMemorySources::default()),
                RuleEngine::for_tests(),
                Arc::default(),
                Metrics::none(),
            )),
            ClientSwitches::for_tests(),
            Some(Box::new(comparator)),
        );
        let served = endpoint
            .handle(Request::new(vf_pb::VisibilityFilterRequest {
                safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                tweets: [(2, 20), (3, 30)]
                    .map(|(tweet_id, author_id)| vf_pb::TweetInput {
                        tweet_id,
                        author_id: Some(author_id),
                    })
                    .into(),
                viewer_id: Some(42),
                country_code: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .results;
        assert_eq!(*finished.lock().unwrap(), vec![served]);
    }
}
