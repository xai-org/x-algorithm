use crate::evaluate_tweets::{EntryComparator, EvaluateTweetsEndpoint};
use crate::hydration::fallback_cache::FallbackCaches;
use crate::hydration::sources::Sources;
use crate::hydration::{HydratedTweet, Hydration, HydrationPlan, Hydrators, Unresolved};
use crate::metrics::{Endpoint, Metrics, RequestLevel, RequestMetrics, RetweetSources};
use crate::models::{
    ClientCapability, Evaluation, PlacedCandidate, PostCandidate, RawCandidate, ReactionContext,
    RequestFacts, RequestViewer, RetweetMerge, SafetyLabelType, TweetFeatures, TweetId,
    ViewerFeatures,
};
use crate::params::ClientSwitches;
use crate::post_evaluator::{PostBatch, PostEvaluation, PostEvaluator};
use crate::quote::{QuotePassMode, QuotePassRunner};
use crate::rules::{NoTrace, RuleEngine, SafetyLevel};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::{Request, Status};
use vf_pb::tweet_evaluation::Outcome;
use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, Safety, UserResponseState};
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) fn raw(tweet_id: u64, request_author_id: Option<u64>) -> RawCandidate {
    RawCandidate {
        tweet_id: TweetId(tweet_id),
        request_author_id,
    }
}

pub(crate) fn tweet_data(tweet_id: u64, outer_tweet_id: Option<u64>) -> vf_pb::TweetData {
    vf_pb::TweetData {
        tweet_id,
        quote_context: outer_tweet_id.map(|outer_tweet_id| vf_pb::QuoteContext { outer_tweet_id }),
    }
}

pub(crate) fn reaction_to(reacted_to_post_id: u64, reacted_to_user_id: u64) -> TweetFeatures {
    TweetFeatures {
        reaction_context: Some(ReactionContext {
            reacted_to_post_id,
            reacted_to_user_id,
        }),
        ..Default::default()
    }
}

pub(crate) fn label_map(label: SafetyLabelType) -> vf_pb::SafetyLabelMap {
    vf_pb::SafetyLabelMap {
        labels: HashMap::from([(label.0, vf_pb::SafetyLabel::default())]),
    }
}

pub(crate) fn protected_user(user_id: u64) -> GizmoduckUserResult {
    GizmoduckUserResult {
        user: Some(GizmoduckUser {
            user_id,
            safety: Safety {
                is_protected: true,
                ..Default::default()
            },
            ..Default::default()
        }),
        response_state: Some(UserResponseState::Found),
    }
}

pub(crate) struct Scenario<S> {
    sources: Arc<S>,
    level: SafetyLevel,
    viewer_id: Option<u64>,
    country_code: Option<String>,
    client_capability: ClientCapability,
    retweet_merge: RetweetMerge,
    fallback_caches: Arc<FallbackCaches>,
    metrics: Metrics,
}

pub(crate) struct InRequestOrder {
    pub(crate) viewer_features: ViewerFeatures,
    pub(crate) candidates: Vec<PostCandidate>,
    pub(crate) safety_labels: HashMap<TweetId, Arc<vf_pb::SafetyLabelMap>>,
    pub(crate) failed_ids: HashSet<TweetId>,
    pub(crate) unresolved: HashMap<TweetId, Unresolved>,
    pub(crate) failed_nodes: Vec<Result<Hydrators, Unresolved>>,
}

pub(crate) struct Evaluated {
    pub(crate) outcomes: Vec<PostEvaluation>,
    pub(crate) retweet_sources: RetweetSources,
}

impl Evaluated {
    pub(crate) fn evaluations(self) -> Vec<Evaluation> {
        self.outcomes
            .into_iter()
            .map(|outcome| outcome.evaluation)
            .collect()
    }
}

impl<S: Sources + 'static> Scenario<S> {
    pub(crate) fn new(sources: S, level: SafetyLevel) -> Self {
        Self {
            sources: Arc::new(sources),
            level,
            viewer_id: None,
            country_code: None,
            client_capability: ClientCapability::default(),
            retweet_merge: RetweetMerge::Unmerged,
            fallback_caches: Arc::default(),
            metrics: Metrics::none(),
        }
    }

    pub(crate) fn viewer(mut self, viewer_id: Option<u64>) -> Self {
        self.viewer_id = viewer_id;
        self
    }

    pub(crate) fn country(mut self, country_code: &str) -> Self {
        self.country_code = Some(country_code.to_owned());
        self
    }

    pub(crate) fn client(mut self, client_capability: ClientCapability) -> Self {
        self.client_capability = client_capability;
        self
    }

    pub(crate) fn retweet_merge(mut self, retweet_merge: RetweetMerge) -> Self {
        self.retweet_merge = retweet_merge;
        self
    }

    pub(crate) fn fallback_caches(mut self, fallback_caches: FallbackCaches) -> Self {
        self.fallback_caches = Arc::new(fallback_caches);
        self
    }

    pub(crate) fn metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = metrics;
        self
    }

    pub(crate) fn sources(&self) -> &S {
        &self.sources
    }

    pub(crate) async fn hydrate(&self, raw: &[RawCandidate]) -> InRequestOrder {
        self.hydrate_with_plan(RuleEngine::for_tests().plan(self.level), raw)
            .await
    }

    pub(crate) async fn hydrate_with_plan(
        &self,
        plan: &HydrationPlan,
        raw: &[RawCandidate],
    ) -> InRequestOrder {
        let metrics = self.request_metrics();
        let request = self
            .request(raw)
            .hydration_request(&self.fallback_caches, &metrics);
        let hydration = plan.hydrate(&*self.sources, request).await;
        InRequestOrder::of(&hydration, raw)
    }

    pub(crate) async fn evaluate(&self, raw: &[RawCandidate]) -> Evaluated {
        self.evaluate_with(RuleEngine::for_tests(), raw).await
    }

    pub(crate) async fn evaluate_with(
        &self,
        rule_engine: RuleEngine,
        raw: &[RawCandidate],
    ) -> Evaluated {
        let sources = Arc::clone(&self.sources) as Arc<dyn Sources>;
        let caches = Arc::clone(&self.fallback_caches);
        let post_evaluator = PostEvaluator::new(sources, rule_engine, caches, self.metrics.clone());
        let (hydrated, outcomes) = post_evaluator
            .run(self.request(raw), &self.request_metrics(), &mut NoTrace)
            .await;
        Evaluated {
            outcomes,
            retweet_sources: hydrated.retweet_sources(),
        }
    }

    fn request_metrics(&self) -> RequestMetrics {
        let endpoint = match self.retweet_merge {
            RetweetMerge::Merged => Endpoint::EvaluateTweets,
            RetweetMerge::Unmerged => Endpoint::FilterTweets,
        };
        self.metrics
            .request(endpoint, RequestLevel::Named(self.level))
    }

    fn request(&self, raw: &[RawCandidate]) -> PostBatch {
        PostBatch {
            facts: RequestFacts::new(
                RequestViewer::new(self.viewer_id, self.country_code.clone()),
                self.client_capability,
            ),
            safety_level: self.level,
            candidates: raw
                .iter()
                .copied()
                .map(PlacedCandidate::standalone)
                .collect(),
            retweet_merge: self.retweet_merge,
        }
    }
}

pub(crate) struct Wire {
    post_evaluator: Arc<PostEvaluator>,
    endpoint: EvaluateTweetsEndpoint,
}

impl Wire {
    pub(crate) fn new(sources: Arc<dyn Sources>) -> Self {
        Self::serving(
            sources,
            FallbackCaches::default(),
            None,
            QuotePassRunner::new(QuotePassMode::Off),
            Metrics::none(),
        )
    }

    pub(crate) fn serving(
        sources: Arc<dyn Sources>,
        fallback_caches: FallbackCaches,
        comparator: Option<Box<dyn EntryComparator>>,
        quote_pass: QuotePassRunner,
        metrics: Metrics,
    ) -> Self {
        let post_evaluator = Arc::new(PostEvaluator::new(
            sources,
            RuleEngine::for_tests(),
            Arc::new(fallback_caches),
            metrics,
        ));
        let endpoint = EvaluateTweetsEndpoint::new(
            Arc::clone(&post_evaluator),
            ClientSwitches::for_tests(),
            comparator,
            quote_pass,
        );
        Self {
            post_evaluator,
            endpoint,
        }
    }

    pub(crate) fn post_evaluator(&self) -> &Arc<PostEvaluator> {
        &self.post_evaluator
    }

    pub(crate) fn into_endpoint(self) -> EvaluateTweetsEndpoint {
        self.endpoint
    }

    pub(crate) async fn send(
        &self,
        request: vf_pb::EvaluateTweetsRequest,
    ) -> Result<Vec<Outcome>, Status> {
        self.handle(Request::new(request)).await
    }

    pub(crate) async fn send_with_context(
        &self,
        context: &TwitterContextViewer,
        request: vf_pb::EvaluateTweetsRequest,
    ) -> Result<Vec<Outcome>, Status> {
        let mut request = Request::new(request);
        request.metadata_mut().insert(
            "twittercontext",
            xai_twittercontext::hydrate_twitter_context(context).unwrap(),
        );
        self.handle(request).await
    }

    async fn handle(
        &self,
        request: Request<vf_pb::EvaluateTweetsRequest>,
    ) -> Result<Vec<Outcome>, Status> {
        let response = self.endpoint.handle(request).await?;
        Ok(response
            .into_inner()
            .results
            .into_iter()
            .map(|result| result.outcome.unwrap())
            .collect())
    }
}

impl InRequestOrder {
    fn of(hydration: &Hydration, raw: &[RawCandidate]) -> Self {
        let tweets = || {
            raw.iter()
                .filter_map(|c| Some((c.tweet_id, hydration.tweet(c.tweet_id)?)))
        };
        Self {
            viewer_features: hydration.viewer().clone(),
            candidates: tweets()
                .filter_map(|(_, tweet)| tweet.candidate().cloned())
                .collect(),
            safety_labels: tweets()
                .filter_map(|(id, tweet)| Some((id, Arc::clone(tweet.safety_labels()?))))
                .collect(),
            failed_ids: tweets()
                .filter(|(_, tweet)| {
                    tweet
                        .candidate()
                        .is_some_and(|candidate| !candidate.failed.is_empty())
                })
                .map(|(id, _)| id)
                .collect(),
            unresolved: tweets()
                .filter_map(|(id, tweet)| match tweet {
                    HydratedTweet::Resolved { .. } => None,
                    HydratedTweet::Unresolved { reason, .. } => Some((id, *reason)),
                })
                .collect(),
            failed_nodes: tweets()
                .map(|(_, tweet)| match tweet {
                    HydratedTweet::Resolved { candidate, .. } => Ok(candidate.failed),
                    HydratedTweet::Unresolved { reason, .. } => Err(*reason),
                })
                .collect(),
        }
    }
}
