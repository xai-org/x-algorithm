use crate::ingress::{self, Entry};
use crate::metrics::{self as ft_metrics, Endpoint, RequestLevel, RequestMetrics};
use crate::models::{Evaluation, RequestViewer};
use crate::params::ClientSwitches;
use crate::post_evaluator::{PostEvaluation, PostEvaluator};
use crate::quote::{Passes, QuotePassRunner};
use crate::rules::{SafetyLevel, WireLevel};
use crate::treatment;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use vf_pb::tweet_evaluation::Outcome;
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) trait EntryComparator: Send + Sync {
    fn compare(
        &self,
        viewer: &RequestViewer,
        safety_level: SafetyLevel,
        outcomes: &[PostEvaluation],
        quote_passes: &Passes,
    );
}

pub struct EvaluateTweetsEndpoint {
    post_evaluator: Arc<PostEvaluator>,
    client_switches: ClientSwitches,
    comparator: Option<Arc<dyn EntryComparator>>,
    quote_pass: QuotePassRunner,
}

impl EvaluateTweetsEndpoint {
    pub(crate) fn new(
        post_evaluator: Arc<PostEvaluator>,
        client_switches: ClientSwitches,
        comparator: Option<Box<dyn EntryComparator>>,
        quote_pass: QuotePassRunner,
    ) -> Self {
        Self {
            post_evaluator,
            client_switches,
            comparator: comparator.map(Arc::from),
            quote_pass,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::EvaluateTweetsRequest>,
    ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
        let safety_level =
            ingress::level(WireLevel::EvaluateTweets(request.get_ref().safety_level));
        ft_metrics::observe_request(
            self.post_evaluator.metrics(),
            Endpoint::EvaluateTweets,
            RequestLevel::of(safety_level.as_ref().ok().copied()),
            request,
            |request, context, metrics| async move {
                let twitter_context =
                    xai_twittercontext::extract_twitter_context(request.metadata());
                context
                    .scope(self.handle_inner(
                        request.into_inner(),
                        twitter_context,
                        &metrics,
                        safety_level,
                    ))
                    .await
                    .map(Response::new)
            },
        )
        .await
    }

    async fn handle_inner(
        &self,
        req: vf_pb::EvaluateTweetsRequest,
        twitter_context: Option<TwitterContextViewer>,
        metrics: &RequestMetrics,
        safety_level: Result<SafetyLevel, Status>,
    ) -> Result<vf_pb::EvaluateTweetsResponse, Status> {
        ft_metrics::record_batch_size(metrics, req.tweets.len());
        let safety_level = safety_level?;
        let ingress::Request {
            viewer,
            rows: entries,
        } = ingress::evaluate_tweets(&req, twitter_context, safety_level);
        let candidates = entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Judged(candidate) => Some(*candidate),
                Entry::NotJudged => None,
            })
            .collect();
        let request = ingress::evaluate_tweets_batch(
            &self.client_switches,
            viewer.clone(),
            safety_level,
            candidates,
        );
        let (outcomes, pending) = self
            .quote_pass
            .serve(&self.post_evaluator, request, metrics)
            .await;
        let mut evaluated = outcomes.iter().peekable();
        let answers: Vec<Answer> = entries
            .iter()
            .map(|entry| match entry {
                Entry::NotJudged => Answer::NotJudged,
                Entry::Judged(candidate) => {
                    match evaluated.next_if(|outcome| outcome.key() == candidate.key()) {
                        Some(outcome) => Answer::Evaluated(&outcome.evaluation),
                        None => Answer::Unpaired,
                    }
                }
            })
            .collect();
        let evaluations = || {
            answers.iter().filter_map(|answer| match answer {
                Answer::Evaluated(evaluation) => Some(*evaluation),
                Answer::NotJudged | Answer::Unpaired => None,
            })
        };
        ft_metrics::record_sent(
            metrics,
            evaluations(),
            answers
                .iter()
                .filter(|answer| matches!(answer, Answer::Unpaired))
                .count(),
        );
        let policies = self.client_switches.limited_actions_policies(
            &viewer,
            evaluations().flat_map(|evaluation| evaluation.verdict().limited_engagement_reasons()),
            metrics.plain(),
        );
        let results = answers
            .iter()
            .map(|answer| {
                let outcome = match answer {
                    Answer::NotJudged => Outcome::NotEvaluated(vf_pb::NotEvaluated {}),
                    Answer::Evaluated(evaluation) => treatment::evaluate_tweets_outcome(
                        Some(evaluation),
                        safety_level,
                        &policies,
                    ),
                    Answer::Unpaired => {
                        treatment::evaluate_tweets_outcome(None, safety_level, &policies)
                    }
                };
                vf_pb::TweetEvaluation {
                    outcome: Some(outcome),
                }
            })
            .collect();
        let comparator = self.comparator.clone();
        self.quote_pass.finish(
            &self.post_evaluator,
            pending,
            outcomes,
            move |outcomes, quote_passes| {
                if let Some(comparator) = comparator {
                    comparator.compare(&viewer, safety_level, outcomes, quote_passes);
                }
            },
        );
        Ok(vf_pb::EvaluateTweetsResponse { results })
    }
}

enum Answer<'a> {
    NotJudged,
    Unpaired,
    Evaluated(&'a Evaluation),
}

#[cfg(test)]
mod tests;
