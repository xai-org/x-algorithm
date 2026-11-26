use crate::hydration::fallback_cache::FallbackCaches;
use crate::hydration::sources::Sources;
use crate::hydration::{HydratedTweet, Hydration, HydrationRequest, Lookup, QuoteEntry};
use crate::metrics::{self as ft_metrics, Metrics, RequestMetrics, RetweetSources, Stage};
use crate::models::{
    Evaluation, EvaluationKey, PlacedCandidate, Placement, RequestFacts, RetweetMerge, TweetId,
};
use crate::rules::engine::{Admitted, AdmittedPost};
use crate::rules::{NoTrace, RuleEngine, SafetyLevel};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
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
        let hydrated = self.hydrate(request, metrics).await;
        let evaluations = match hydrated.retweet_merge {
            RetweetMerge::Unmerged => self.evaluate(&hydrated, [], false, observer).0,
            RetweetMerge::Merged => retweet::merge_sources(self, &hydrated, observer),
        };
        (hydrated, evaluations)
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
        let retweet_sources = if request.retweet_merge == RetweetMerge::Unmerged {
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
            retweet_merge: request.retweet_merge,
            candidates: request.candidates,
            hydration,
        }
    }

    pub(crate) fn evaluate<O: PostObserver>(
        &self,
        hydrated: &HydratedRequest,
        sources: impl IntoIterator<Item = TweetId>,
        repeats: bool,
        observer: &mut O,
    ) -> (Vec<PostEvaluation>, HashMap<TweetId, Evaluation>) {
        let evaluating = Instant::now();
        let admitted = self
            .rule_engine
            .admit(hydrated.safety_level, &hydrated.hydration);
        let mut firsts = repeats.then(HashMap::new);
        let mut evaluations = Vec::with_capacity(hydrated.candidates.len());
        for candidate in &hydrated.candidates {
            let key = candidate.key();
            push_once(&mut evaluations, firsts.as_mut(), key, || {
                Self::evaluate_post(hydrated, admitted.as_ref(), key, observer)
            });
        }
        let sources = sources
            .into_iter()
            .map(|source_id| {
                let key = EvaluationKey::retweet_source(source_id);
                let tweet = hydrated.hydration.tweet(source_id);
                let source = Self::evaluation(admitted.as_ref(), key, tweet, observer);
                (source_id, source)
            })
            .collect();
        ft_metrics::record_phase(
            &hydrated.metrics,
            Stage::PostHydration,
            hydrated.retweet_sources,
            evaluating.elapsed(),
        );
        (evaluations, sources)
    }

    fn evaluate_post<O: PostObserver>(
        hydrated: &HydratedRequest,
        admitted: Option<&Admitted<'_>>,
        key: EvaluationKey,
        observer: &mut O,
    ) -> PostEvaluation {
        let tweet = hydrated.hydration.tweet(key.tweet_id);
        PostEvaluation {
            tweet_id: key.tweet_id,
            placement: key.placement,
            source_tweet_id: tweet.and_then(HydratedTweet::source_tweet_id),
            evaluation: Self::evaluation(admitted, key, tweet, observer),
            safety_labels: tweet.and_then(HydratedTweet::safety_labels).cloned(),
        }
    }

    fn evaluation<O: PostObserver>(
        admitted: Option<&Admitted<'_>>,
        key: EvaluationKey,
        tweet: Option<&HydratedTweet>,
        observer: &mut O,
    ) -> Evaluation {
        let EvaluationKey {
            tweet_id,
            placement,
        } = key;
        match tweet {
            Some(HydratedTweet::Unresolved {
                reason, deletion, ..
            }) => reason.evaluation(*deletion),
            Some(HydratedTweet::Resolved { .. }) | None => admitted
                .and_then(|admitted| admitted.post(tweet_id, placement))
                .map_or(Evaluation::Failed(Lookup::Tweet), |post| {
                    observer.judge(key, &post)
                }),
        }
    }
}

fn push_once<K: Hash + Eq, V: Clone>(
    answers: &mut Vec<V>,
    firsts: Option<&mut HashMap<K, usize>>,
    key: K,
    answer: impl FnOnce() -> V,
) {
    let first = firsts.and_then(|firsts| match firsts.entry(key) {
        Entry::Occupied(first) => answers.get(*first.get()).cloned(),
        Entry::Vacant(slot) => {
            slot.insert(answers.len());
            None
        }
    });
    answers.push(first.unwrap_or_else(answer));
}

pub(crate) trait PostObserver {
    #[inline]
    fn judge(&mut self, _: EvaluationKey, post: &AdmittedPost<'_>) -> Evaluation {
        post.evaluate()
    }

    #[inline]
    fn select(&mut self, _: EvaluationKey, _: MergeSelection) {}
}

impl PostObserver for NoTrace {}

pub(crate) struct HydratedRequest {
    safety_level: SafetyLevel,
    metrics: RequestMetrics,
    retweet_sources: RetweetSources,
    retweet_merge: RetweetMerge,
    candidates: Vec<PlacedCandidate>,
    hydration: Hydration,
}

impl HydratedRequest {
    pub(crate) fn hydration(&self) -> &Hydration {
        &self.hydration
    }

    #[cfg(test)]
    pub(crate) fn retweet_sources(&self) -> RetweetSources {
        self.retweet_sources
    }

    pub(crate) fn requested_ids(&self) -> impl ExactSizeIterator<Item = TweetId> + '_ {
        self.candidates.iter().map(|placed| placed.tweet_id())
    }
}

#[cfg(test)]
mod tests;
