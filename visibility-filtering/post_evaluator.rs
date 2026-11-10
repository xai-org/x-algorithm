use crate::hydration::fallback_cache::FallbackCaches;
use crate::hydration::sources::Sources;
use crate::hydration::{
    Cause, HydratedTweet, Hydration, HydrationRequest, Lookup, QuoteEntry, Relation, Unresolved,
};
use crate::metrics::{self as ft_metrics, Metrics, RequestMetrics, RetweetSources, Stage};
use crate::models::{
    Evaluation, EvaluationKey, PlacedCandidate, Placement, RequestFacts, RetweetMerge, TweetId,
};
use crate::rules::engine::Admitted;
use crate::rules::{NoTrace, PolicyTrace, RuleEngine, SafetyLevel};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use xai_visibility_filtering_proto as vf_pb;

mod author;
mod notification;
mod retweet;

pub(crate) use author::AuthorJudgement;
pub(crate) use retweet::hydrate_and_trace;
pub use retweet::MergeSelection;

pub struct PostBatch {
    pub(crate) facts: RequestFacts,
    pub safety_level: SafetyLevel,
    pub candidates: Vec<PlacedCandidate>,
    pub retweet_merge: RetweetMerge,
}

impl PostBatch {
    pub(crate) fn hydration_request<'a>(
        &self,
        fallback_caches: &'a FallbackCaches,
        metrics: &'a RequestMetrics,
    ) -> HydrationRequest<'a> {
        HydrationRequest::new(
            self.facts.clone(),
            self.candidates
                .iter()
                .map(|placed| placed.candidate)
                .collect(),
            self.retweet_merge,
            fallback_caches,
            metrics,
        )
        .with_quotes(
            self.candidates
                .iter()
                .filter_map(|placed| match placed.placement {
                    Placement::InnerQuoted { outer } => Some(QuoteEntry {
                        inner: placed.tweet_id(),
                        outer,
                    }),
                    Placement::Standalone | Placement::RetweetSource => None,
                })
                .collect(),
        )
    }
}

#[derive(Clone)]
pub struct PostEvaluation {
    pub tweet_id: TweetId,
    pub placement: Placement,
    pub source_tweet_id: Option<TweetId>,
    pub evaluation: Evaluation,
    pub safety_labels: Option<Arc<vf_pb::SafetyLabelMap>>,
}

impl PostEvaluation {
    pub fn key(&self) -> EvaluationKey {
        EvaluationKey {
            tweet_id: self.tweet_id,
            placement: self.placement,
        }
    }
}

pub struct PostEvaluator {
    sources: Arc<dyn Sources>,
    rule_engine: RuleEngine,
    fallback_caches: Arc<FallbackCaches>,
    metrics: Metrics,
}

impl PostEvaluator {
    pub(crate) fn new(
        sources: Arc<dyn Sources>,
        rule_engine: RuleEngine,
        fallback_caches: Arc<FallbackCaches>,
        metrics: Metrics,
    ) -> Self {
        Self {
            sources,
            rule_engine,
            fallback_caches,
            metrics,
        }
    }

    pub(crate) fn rule_engine(&self) -> &RuleEngine {
        &self.rule_engine
    }

    pub(crate) fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub(crate) async fn run<O: PostObserver>(
        &self,
        request: PostBatch,
        metrics: &RequestMetrics,
        observer: &mut O,
    ) -> (HydratedRequest, Vec<PostEvaluation>) {
        let retweet_merge = request.retweet_merge;
        let hydrated = self.hydrate(request, metrics).await;
        let outcomes = match retweet_merge {
            RetweetMerge::Unmerged => self.evaluate_in_request_order(&hydrated, observer),
            RetweetMerge::Merged => retweet::merge_sources(self, &hydrated, observer),
        };
        (hydrated, outcomes)
    }

    async fn hydrate(&self, request: PostBatch, metrics: &RequestMetrics) -> HydratedRequest {
        let started = Instant::now();
        let plan = self.rule_engine.plan(request.safety_level);
        let hydration = plan
            .hydrate(
                &*self.sources,
                request.hydration_request(&self.fallback_caches, metrics),
            )
            .await;
        let hydrated_at = Instant::now();
        let retweet_sources = if !hydration.is_on(Relation::RetweetSource) {
            RetweetSources::NoSource
        } else if hydration.has_fetched_retweet_sources() {
            RetweetSources::Fetched
        } else if request.candidates.iter().any(|candidate| {
            hydration
                .tweet(candidate.tweet_id())
                .is_some_and(|tweet| tweet.source_to_merge().is_some())
        }) {
            RetweetSources::InBatch
        } else {
            RetweetSources::NoSource
        };
        ft_metrics::record_phase(
            metrics,
            Stage::Hydration,
            retweet_sources,
            hydrated_at - started,
        );
        HydratedRequest {
            safety_level: request.safety_level,
            metrics: metrics.clone(),
            retweet_sources,
            candidates: request.candidates,
            hydration,
        }
    }

    pub(crate) fn evaluate_in_request_order<O: PostObserver>(
        &self,
        hydrated: &HydratedRequest,
        observer: &mut O,
    ) -> Vec<PostEvaluation> {
        let evaluating = Instant::now();
        let admitted = self
            .rule_engine
            .admit(hydrated.safety_level, &hydrated.hydration);
        let outcomes = hydrated
            .candidates()
            .map(|candidate| Self::outcome(hydrated, admitted.as_ref(), candidate.key(), observer))
            .collect();
        ft_metrics::record_phase(
            &hydrated.metrics,
            Stage::PostHydration,
            hydrated.retweet_sources,
            evaluating.elapsed(),
        );
        outcomes
    }

    pub(crate) fn evaluate<O: PostObserver>(
        &self,
        hydrated: &HydratedRequest,
        keys: impl IntoIterator<Item = EvaluationKey>,
        observer: &mut O,
    ) -> HashMap<EvaluationKey, PostEvaluation> {
        let evaluating = Instant::now();
        let admitted = self
            .rule_engine
            .admit(hydrated.safety_level, &hydrated.hydration);
        let mut outcomes = HashMap::new();
        for key in keys {
            outcomes
                .entry(key)
                .or_insert_with(|| Self::outcome(hydrated, admitted.as_ref(), key, observer));
        }
        ft_metrics::record_phase(
            &hydrated.metrics,
            Stage::PostHydration,
            hydrated.retweet_sources,
            evaluating.elapsed(),
        );
        outcomes
    }

    pub(crate) fn trace(
        &self,
        hydrated: &HydratedRequest,
        keys: impl IntoIterator<Item = EvaluationKey>,
        traces: &mut HashMap<EvaluationKey, PolicyTrace>,
    ) {
        let admitted = self
            .rule_engine
            .admit(hydrated.safety_level, &hydrated.hydration);
        for key in keys {
            if !traces.contains_key(&key) {
                Self::outcome(hydrated, admitted.as_ref(), key, traces);
            }
        }
    }

    fn outcome<O: PostObserver>(
        hydrated: &HydratedRequest,
        admitted: Option<&Admitted<'_>>,
        key: EvaluationKey,
        observer: &mut O,
    ) -> PostEvaluation {
        let EvaluationKey {
            tweet_id,
            placement,
        } = key;
        let copied = hydrated.hydration.is_on(Relation::RetweetSource)
            && placement != Placement::RetweetSource;
        let tweet = hydrated.hydration.tweet(tweet_id);
        let post = admitted.and_then(|admitted| admitted.post(tweet_id, copied, placement));
        let evaluation = match (tweet, post) {
            (Some(HydratedTweet::Resolved { .. }), Some(post)) => {
                if O::TOTAL {
                    let trace = post.trace();
                    let evaluation = trace.evaluation.clone();
                    observer.record(key, trace);
                    evaluation
                } else {
                    post.evaluate()
                }
            }
            (
                Some(HydratedTweet::Unresolved {
                    reason: Unresolved { lookup, cause },
                    deletion,
                    ..
                }),
                _,
            ) => match cause {
                Cause::NotFound => {
                    deletion.map_or(Evaluation::NotFound(*lookup), Evaluation::Deleted)
                }
                Cause::Failed => Evaluation::Failed(*lookup),
            },
            (Some(HydratedTweet::Resolved { .. }), None) | (None, _) => {
                Evaluation::Failed(Lookup::Tweet)
            }
        };
        PostEvaluation {
            tweet_id,
            placement,
            source_tweet_id: tweet.and_then(HydratedTweet::source_tweet_id),
            evaluation,
            safety_labels: tweet.and_then(HydratedTweet::safety_labels).cloned(),
        }
    }
}

pub(crate) trait PostObserver {
    const TOTAL: bool;

    fn record(&mut self, key: EvaluationKey, trace: PolicyTrace);

    #[inline]
    fn select(&mut self, _: EvaluationKey, _: MergeSelection) {}
}

impl PostObserver for NoTrace {
    const TOTAL: bool = false;

    #[inline]
    fn record(&mut self, _: EvaluationKey, _: PolicyTrace) {}
}

impl PostObserver for HashMap<EvaluationKey, PolicyTrace> {
    const TOTAL: bool = true;

    fn record(&mut self, key: EvaluationKey, trace: PolicyTrace) {
        self.insert(key, trace);
    }
}

pub(crate) struct HydratedRequest {
    safety_level: SafetyLevel,
    metrics: RequestMetrics,
    retweet_sources: RetweetSources,
    candidates: Vec<PlacedCandidate>,
    hydration: Hydration,
}

impl HydratedRequest {
    pub(crate) fn hydration(&self) -> &Hydration {
        &self.hydration
    }

    pub(crate) fn metrics(&self) -> &RequestMetrics {
        &self.metrics
    }

    #[cfg(test)]
    pub(crate) fn retweet_sources(&self) -> RetweetSources {
        self.retweet_sources
    }

    pub(crate) fn requested_ids(&self) -> impl ExactSizeIterator<Item = TweetId> + '_ {
        self.candidates().map(PlacedCandidate::tweet_id)
    }

    pub(crate) fn candidates(&self) -> impl ExactSizeIterator<Item = PlacedCandidate> + '_ {
        self.candidates.iter().copied()
    }
}

#[cfg(test)]
mod tests;
