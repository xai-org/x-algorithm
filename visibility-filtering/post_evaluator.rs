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
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeQuery, Graph};
    use crate::hydration::sources::Source;
    use crate::hydration::sources::{control, found_user, Fault, InMemorySources};
    use crate::hydration::{author_fallback_cache, tweet_fallback_cache, Hydrator, Hydrators};
    use crate::models::{LimitedEngagementReason, TweetFeatures, Verdict};
    use crate::rules::fixtures::{allow, dropped, limited, limited_for};
    use crate::scenario::{raw, Scenario};
    use std::time::Duration;
    use xai_core_entities::entities::{
        ConversationControlArm, GizmoduckUserResult, PureCoreData, UserResponseState,
    };
    use xai_visibility_filtering::models::FilteredReason;

    #[tokio::test(start_paused = true)]
    async fn pure_core_timeout_fails_every_candidate_at_hydration_timeout() {
        let sources = InMemorySources::default()
            .tweet_features(1, TweetFeatures::default())
            .tweet_features(2, TweetFeatures::default())
            .authors(&[20])
            .fault(Source::TesPureCoreData, Fault::Hangs);
        let scenario = Scenario::new(sources, SafetyLevel::TimelineHome).viewer(Some(50));
        let started = tokio::time::Instant::now();
        let evaluated = tokio::time::timeout(
            crate::hydration::HYDRATION_TIMEOUT * 2,
            scenario.evaluate(&[raw(1, None), raw(2, Some(20))]),
        )
        .await
        .unwrap();
        assert_eq!(started.elapsed(), crate::hydration::HYDRATION_TIMEOUT);
        assert_eq!(
            evaluated.evaluations(),
            vec![Evaluation::Failed(Lookup::Tweet); 2]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn home_hydration_limits_posts_whose_author_or_reply_root_blocks_the_viewer() {
        let reply =
            |author_id, conversation_id, in_reply_to_tweet_id, in_reply_to_user_id| PureCoreData {
                author_id,
                conversation_id: Some(conversation_id),
                in_reply_to_tweet_id: Some(in_reply_to_tweet_id),
                in_reply_to_user_id: Some(in_reply_to_user_id),
                ..Default::default()
            };
        let root_author_blocked = || {
            Evaluation::complete(limited(
                LimitedEngagementReason::RootAuthorBlockedViewer,
                "blocked_viewer/limited_engagement/root_author_blocked_viewer",
            ))
        };
        let raw = [1, 3, 100, 2, 201, 4, 7, 8].map(|id| raw(id, None));
        for delayed in [Source::TesPureCoreData, Source::TesConversationControl] {
            let home = |viewer_id| {
                let sources = InMemorySources::default()
                    .pure_core(1, reply(10, 100, 100, 30))
                    .tweet(3, 10)
                    .tweet(100, 30)
                    .pure_core(2, reply(20, 100, 101, 99))
                    .pure_core(201, reply(41, 200, 200, 60))
                    .pure_core(4, reply(21, 200, 201, 41))
                    .pure_core(7, reply(22, 500, 501, 99))
                    .pure_core(8, reply(23, 500, 7, 22))
                    .tweet(500, 30)
                    .authors(&[10, 20, 21, 22, 23, 30, 41])
                    .edge(Graph::Blocks, 30, 50)
                    .edge(Graph::Blocks, 60, 50)
                    .edge(Graph::Blocks, 99, 50)
                    .fault(delayed, Fault::Delays(Duration::from_millis(5)));
                Scenario::new(sources, SafetyLevel::TimelineHomeHydration).viewer(viewer_id)
            };
            let logged_in = home(Some(50));
            assert_eq!(
                logged_in.evaluate(&raw).await.evaluations(),
                vec![
                    root_author_blocked(),
                    Evaluation::complete(allow()),
                    Evaluation::complete(limited(
                        LimitedEngagementReason::BlockedViewer,
                        "blocked_viewer/limited_engagement",
                    )),
                    root_author_blocked(),
                    root_author_blocked(),
                    root_author_blocked(),
                    root_author_blocked(),
                    root_author_blocked(),
                ],
                "{delayed:?}"
            );
            let logged_out = home(None);
            assert_eq!(
                logged_out.evaluate(&raw).await.evaluations(),
                vec![Evaluation::complete(allow()); 8],
                "{delayed:?}"
            );
            assert!(logged_out.sources().selects().is_empty(), "{delayed:?}");
            let sources = logged_in.sources();
            assert_eq!(
                sources.selects(),
                [vec![
                    EdgeQuery::forward(Graph::Follows, vec![10, 20, 21, 22, 23, 30, 41]),
                    EdgeQuery::reverse(Graph::Blocks, vec![10, 20, 21, 22, 23, 30, 41, 60]),
                ]],
                "{delayed:?}"
            );
            let requested = [vec![1, 2, 3, 4, 7, 8, 100, 201]];
            assert_eq!(
                sources.keys(Source::TesPureCoreData),
                requested,
                "{delayed:?}"
            );
            assert_eq!(
                sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
                [vec![500]],
                "{delayed:?}"
            );
            assert_eq!(
                logged_out.sources().keys(Source::TesPureCoreData),
                requested,
                "{delayed:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_nested_reply_fails_a_node_only_when_its_request_tweet_roots_pure_core_failed() {
        let nested = |author_id, conversation_id| PureCoreData {
            author_id,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(conversation_id + 1),
            in_reply_to_user_id: Some(99),
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .pure_core(1, nested(10, 100))
            .pure_core(2, nested(11, 200))
            .tweet_features(200, TweetFeatures::default())
            .fail_key(Source::TesPureCoreData, 200)
            .authors(&[10, 11]);
        let scenario = Scenario::new(sources, SafetyLevel::TimelineHomeHydration).viewer(Some(50));
        let raw = [1, 100, 2, 200].map(|id| raw(id, None));
        assert_eq!(
            scenario.evaluate(&raw).await.evaluations(),
            [
                Evaluation::complete(allow()),
                Evaluation::NotFound(Lookup::Tweet),
                Evaluation::Complete {
                    verdict: allow(),
                    fail_open_defaults: Hydrators::of(Hydrator::BlockedByReplyRoot),
                },
                Evaluation::Failed(Lookup::Tweet),
            ]
        );
        assert_eq!(
            scenario.sources().keys(Source::TesPureCoreData),
            [vec![1, 2, 100, 200]]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn home_hydration_limits_a_nested_reply_whose_root_author_blocks_the_viewer() {
        use ConversationControlArm::{ByInvitation, Community};
        let nested = || PureCoreData {
            author_id: 20,
            conversation_id: Some(100),
            in_reply_to_tweet_id: Some(101),
            in_reply_to_user_id: Some(99),
            ..Default::default()
        };
        let world = |core: PureCoreData, arm: Option<ConversationControlArm>, has_root: bool| {
            let world = InMemorySources::default()
                .pure_core(2, core)
                .authors(&[20])
                .edge(Graph::Blocks, 30, 50)
                .edge(Graph::Blocks, 99, 50);
            let world = if has_root {
                world.tweet(100, 30)
            } else {
                world
            };
            match arm {
                Some(arm) => world.control(2, control(arm, 30, &[])),
                None => world,
            }
        };
        let author_select = vec![
            EdgeQuery::forward(Graph::Follows, vec![20]),
            EdgeQuery::reverse(Graph::Blocks, vec![20]),
        ];
        let root_select = |followed_by: Vec<u64>| {
            vec![
                EdgeQuery::reverse(Graph::Follows, followed_by),
                EdgeQuery::forward(Graph::SuperFollows, vec![]),
                EdgeQuery::forward(Graph::Follows, vec![]),
                EdgeQuery::reverse(Graph::Blocks, vec![30]),
            ]
        };
        let root_post_select = vec![EdgeQuery::reverse(Graph::Blocks, vec![30])];
        let root_blocked = limited(
            LimitedEngagementReason::RootAuthorBlockedViewer,
            "blocked_viewer/limited_engagement/root_author_blocked_viewer",
        );
        let root_blocked_in_control = limited_for(
            &[
                LimitedEngagementReason::RootAuthorBlockedViewer,
                LimitedEngagementReason::ConversationControl,
            ],
            "blocked_viewer/limited_engagement/root_author_blocked_viewer",
        );
        let no_root_call: (Vec<Vec<u64>>, Vec<Duration>) = (vec![], vec![]);
        let root_call_at_5_ms = (vec![vec![100]], vec![Duration::from_millis(5)]);
        let rows = [
            (
                "community, control first",
                world(nested(), Some(Community), true),
                Source::TesPureCoreData,
                root_blocked_in_control.clone(),
                vec![root_select(vec![30]), author_select.clone()],
                no_root_call.clone(),
            ),
            (
                "community, pure core first",
                world(nested(), Some(Community), true),
                Source::TesConversationControl,
                root_blocked_in_control.clone(),
                vec![author_select.clone(), root_select(vec![30])],
                no_root_call.clone(),
            ),
            (
                "by invitation, control first",
                world(nested(), Some(ByInvitation), true),
                Source::TesPureCoreData,
                root_blocked_in_control.clone(),
                vec![root_select(vec![]), author_select.clone()],
                no_root_call.clone(),
            ),
            (
                "by invitation, pure core first",
                world(nested(), Some(ByInvitation), true),
                Source::TesConversationControl,
                root_blocked_in_control,
                vec![author_select.clone(), root_select(vec![])],
                no_root_call.clone(),
            ),
            (
                "by invitation root post",
                world(
                    PureCoreData {
                        author_id: 20,
                        ..Default::default()
                    },
                    Some(ByInvitation),
                    true,
                ),
                Source::TesPureCoreData,
                limited(
                    LimitedEngagementReason::ConversationControl,
                    "limit_replies_by_invitation/limited_engagement/conversation_control",
                ),
                vec![author_select.clone()],
                no_root_call,
            ),
            (
                "uncontrolled, control first",
                world(nested(), None, true),
                Source::TesPureCoreData,
                root_blocked.clone(),
                vec![author_select.clone(), root_post_select.clone()],
                root_call_at_5_ms.clone(),
            ),
            (
                "uncontrolled, pure core first",
                world(nested(), None, true),
                Source::TesConversationControl,
                root_blocked,
                vec![author_select.clone(), root_post_select],
                root_call_at_5_ms.clone(),
            ),
            (
                "uncontrolled, root deleted",
                world(nested(), None, false),
                Source::TesPureCoreData,
                allow(),
                vec![author_select],
                root_call_at_5_ms,
            ),
        ];
        for (name, world, delayed, verdict, selects, (root_keys, root_starts)) in rows {
            let sources = world.fault(delayed, Fault::Delays(Duration::from_millis(5)));
            let scenario =
                Scenario::new(sources, SafetyLevel::TimelineHomeHydration).viewer(Some(50));
            let started = tokio::time::Instant::now();
            let evaluations = scenario.evaluate(&[raw(2, None)]).await.evaluations();
            let sources = scenario.sources();
            assert_eq!(evaluations, [Evaluation::complete(verdict)], "{name}");
            assert_eq!(sources.selects(), selects, "{name}");
            assert_eq!(sources.keys(Source::TesPureCoreData), [vec![2]], "{name}");
            assert_eq!(
                sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
                root_keys,
                "{name}"
            );
            let starts: Vec<Duration> = sources
                .starts((Relation::ConversationRoot, Source::TesPureCoreData))
                .into_iter()
                .map(|start| start - started)
                .collect();
            assert_eq!(starts, root_starts, "{name}");
            for source in [
                Source::TesTweetForVisibility,
                Source::TesConversationControl,
                Source::SafetyLabels,
            ] {
                assert_eq!(sources.keys(source), [vec![2]], "{name} {source:?}");
            }
            assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![20]], "{name}");
        }
    }

    #[tokio::test]
    async fn trusted_friends_posts_show_to_list_members_and_owners_and_a_failed_lookup_drops_them()
    {
        let trusted_friends = |list_id| TweetFeatures {
            trusted_friends_list_id: Some(list_id),
            ..Default::default()
        };
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, trusted_friends(7))
                .tweet(2, 11)
                .tweet_features(2, trusted_friends(7))
                .tweet(3, 12)
                .tweet_features(3, trusted_friends(9))
                .tweet(4, 13)
                .authors(&[10, 11, 12, 13])
                .trusted_friend(7, 50)
        };
        let shown = || Evaluation::complete(allow());
        let trusted_friends_drop = || {
            Evaluation::complete(dropped(
                FilteredReason::UnspecifiedReason,
                "trusted_friends_tweet/drop/unspecified",
            ))
        };

        let home = |sources: InMemorySources, viewer_id| {
            Scenario::new(sources, SafetyLevel::TimelineHomeHydration).viewer(viewer_id)
        };

        let healthy = home(world(), Some(50));
        let raw = [1, 2, 3, 4].map(|id| raw(id, None));
        assert_eq!(
            healthy.evaluate(&raw).await.evaluations(),
            [shown(), shown(), trusted_friends_drop(), shown()]
        );
        assert_eq!(healthy.sources().keys(Source::TrustedFriends), [vec![7, 9]]);

        let failed = home(
            world().fault(Source::TrustedFriends, Fault::Fails),
            Some(50),
        );
        assert_eq!(
            failed.evaluate(&[raw[0], raw[3]]).await.evaluations(),
            [trusted_friends_drop(), shown()]
        );

        for (viewer_id, raw) in [(None, &raw[..2]), (Some(50), &raw[3..])] {
            let scenario = home(world(), viewer_id);
            scenario.evaluate(raw).await;
            assert!(!scenario.sources().calls().contains(&Source::TrustedFriends));
        }
    }

    #[tokio::test]
    async fn local_posts_limit_viewers_outside_their_place_and_a_failed_lookup_limits_none() {
        const A: u64 = 0xa000_0000_0000_0001;
        const B: u64 = 0xb000_0000_0000_0002;
        let local = |place| TweetFeatures {
            narrowcast_place_id: Some(place),
            ..Default::default()
        };
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, local(A))
                .control(1, control(ConversationControlArm::Local, 10, &[]))
                .tweet(2, 11)
                .tweet_features(2, local(B))
                .control(2, control(ConversationControlArm::Local, 11, &[]))
                .tweet(3, 12)
                .authors(&[10, 11, 12])
        };
        let shown = || Evaluation::complete(allow());
        let local_limit = || {
            Evaluation::complete(limited(
                LimitedEngagementReason::LocalTweet,
                "local_tweet/limited_engagement",
            ))
        };

        let home = |sources: InMemorySources| {
            Scenario::new(sources, SafetyLevel::TimelineHomeHydration).viewer(Some(50))
        };
        let raw = [1, 2, 3].map(|id| raw(id, None));

        let located = home(world().located_in(B));
        assert_eq!(
            located.evaluate(&raw).await.evaluations(),
            [local_limit(), shown(), shown()]
        );
        assert_eq!(located.sources().keys(Source::UserLocation), [vec![A, B]]);

        assert_eq!(
            home(world()).evaluate(&raw[..2]).await.evaluations(),
            [local_limit(), local_limit()]
        );

        let failed = world()
            .located_in(B)
            .fault(Source::UserLocation, Fault::Fails);
        assert_eq!(
            home(failed).evaluate(&[raw[0], raw[2]]).await.evaluations(),
            [
                Evaluation::Complete {
                    verdict: allow(),
                    fail_open_defaults: Hydrators::of(Hydrator::OutsideNarrowcastPlace),
                },
                shown(),
            ]
        );
    }

    #[tokio::test]
    async fn my_network_posts_limit_viewers_outside_the_root_authors_network_or_its_failed_lookup()
    {
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .control(1, control(ConversationControlArm::MyNetwork, 10, &[]))
                .authors(&[10])
        };
        let shown = || Evaluation::complete(allow());
        let network_limit = || {
            Evaluation::complete(limited(
                LimitedEngagementReason::ConversationControl,
                "limit_replies_my_network/limited_engagement/conversation_control",
            ))
        };
        let home = |sources: InMemorySources| {
            Scenario::new(sources, SafetyLevel::TimelineHomeHydration).viewer(Some(50))
        };
        let raw = [raw(1, None)];

        let second_degree = home(world().second_degree_path(10, 50));
        assert_eq!(second_degree.evaluate(&raw).await.evaluations(), [shown()]);
        assert_eq!(second_degree.sources().keys(Source::Wingman), [vec![10]]);

        let followed = home(world().edge(Graph::Follows, 10, 50));
        assert_eq!(followed.evaluate(&raw).await.evaluations(), [shown()]);
        assert!(followed.sources().keys(Source::Wingman).is_empty());

        let failed = world()
            .second_degree_path(10, 50)
            .fault(Source::Wingman, Fault::Fails);
        for outside in [world(), failed] {
            assert_eq!(
                home(outside).evaluate(&raw).await.evaluations(),
                [network_limit()]
            );
        }
    }

    #[tokio::test]
    async fn a_circle_row_served_from_the_fallback_cache_still_asks_the_trusted_friends_list() {
        let trusted_friends = |list_id| TweetFeatures {
            trusted_friends_list_id: Some(list_id),
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet_features(1, trusted_friends(7))
            .tweet(2, 11)
            .tweet_features(2, trusted_friends(9))
            .authors(&[10, 11])
            .trusted_friend(7, 50);
        let scenario = Scenario::new(sources, SafetyLevel::TimelineHomeHydration)
            .viewer(Some(50))
            .fallback_caches(FallbackCaches {
                tweets: Some(tweet_fallback_cache(8)),
                authors: None,
            });
        let judged = || async {
            let raw = [raw(1, None), raw(2, None)];
            scenario.evaluate(&raw).await.evaluations()
        };
        let expected = [
            Evaluation::complete(allow()),
            Evaluation::complete(dropped(
                FilteredReason::UnspecifiedReason,
                "trusted_friends_tweet/drop/unspecified",
            )),
        ];

        assert_eq!(judged().await, expected);
        scenario
            .sources()
            .break_source(Source::TesTweetForVisibility, Fault::Fails);
        assert_eq!(judged().await, expected);
        assert_eq!(
            scenario.sources().keys(Source::TrustedFriends),
            [vec![7, 9], vec![7, 9]]
        );
    }

    #[tokio::test]
    async fn pure_core_alone_says_a_post_is_a_retweet() {
        let sources = InMemorySources::default()
            .retweet(1, 10, 5, Some(20))
            .tweet(2, 10)
            .authors(&[10])
            .edge(Graph::MuteRetweets, 50, 10);
        let evaluated = Scenario::new(sources, SafetyLevel::TimelineHome)
            .viewer(Some(50))
            .evaluate(&[raw(1, None), raw(2, None)])
            .await;
        assert_eq!(
            evaluated.evaluations(),
            vec![
                Evaluation::complete(dropped(
                    FilteredReason::UnspecifiedReason,
                    "viewer_mutes_retweets/drop/unspecified",
                )),
                Evaluation::complete(allow()),
            ]
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum PureCore {
        Found,
        NotFound,
        Failed,
        FailedCacheHit,
    }

    #[derive(Clone, Copy, Debug)]
    enum TweetRow {
        Found,
        NotFound,
        Failed,
        FailedCacheHit,
    }

    #[derive(Clone, Copy, Debug)]
    enum Author {
        Found,
        Partial,
        NotFound,
        Failed,
        FailedCacheHit,
        PartialCacheHit,
    }

    fn lookups(
        pure_core: PureCore,
        tweet_row: TweetRow,
        author: Author,
    ) -> (InMemorySources, Vec<(Source, Fault)>) {
        let mut broken = Vec::new();
        let sources = InMemorySources::default();
        let sources = match pure_core {
            PureCore::Found => sources.tweet(1, 10),
            PureCore::NotFound => sources,
            PureCore::Failed => sources.tweet(1, 10).fail_key(Source::TesPureCoreData, 1),
            PureCore::FailedCacheHit => {
                broken.push((Source::TesPureCoreData, Fault::Fails));
                sources.tweet(1, 10)
            }
        };
        let nullcast = TweetFeatures {
            is_nullcast: true,
            ..Default::default()
        };
        let sources = match tweet_row {
            TweetRow::Found => sources.tweet_features(1, nullcast),
            TweetRow::NotFound => sources.without_tweet_row(1),
            TweetRow::Failed => sources.fail_key(Source::TesTweetForVisibility, 1),
            TweetRow::FailedCacheHit => {
                broken.push((Source::TesTweetForVisibility, Fault::Fails));
                sources.tweet_features(1, nullcast)
            }
        };
        let sources = match author {
            Author::Found => sources.authors(&[10]),
            Author::Partial => sources.user(
                10,
                GizmoduckUserResult {
                    response_state: Some(UserResponseState::Partial),
                    ..found_user(10)
                },
            ),
            Author::NotFound => sources,
            Author::Failed => sources.fail_key(Source::GizmoduckAuthor, 10),
            Author::FailedCacheHit => {
                broken.push((Source::GizmoduckAuthor, Fault::Fails));
                sources.authors(&[10])
            }
            Author::PartialCacheHit => {
                broken.push((Source::GizmoduckAuthor, Fault::AnswersPartial));
                sources.authors(&[10])
            }
        };
        (sources, broken)
    }

    #[tokio::test]
    async fn a_post_is_judged_only_when_its_tweet_and_author_lookups_answer() {
        let nullcast_drop = dropped(FilteredReason::TweetIsNullcast, "nullcasted_tweet/drop");
        let judged = Evaluation::complete(nullcast_drop.clone());
        let not_found = Evaluation::NotFound;
        let failed = Evaluation::Failed;
        let rows = [
            (
                PureCore::Found,
                TweetRow::Found,
                Author::Found,
                judged.clone(),
            ),
            (
                PureCore::Found,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::Found,
                TweetRow::Failed,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::NotFound,
                TweetRow::Found,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::NotFound,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::NotFound,
                TweetRow::Failed,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::Failed,
                TweetRow::Found,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::Failed,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::Failed,
                TweetRow::Failed,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::Found,
                Author::Found,
                judged.clone(),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::Failed,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::Found,
                TweetRow::FailedCacheHit,
                Author::Found,
                judged.clone(),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::FailedCacheHit,
                Author::FailedCacheHit,
                judged.clone(),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::Partial,
                Evaluation::Partial {
                    verdict: nullcast_drop,
                    fail_open_defaults: Hydrators::of(Hydrator::AuthorSafety),
                },
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::NotFound,
                not_found(Lookup::Author),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::Failed,
                failed(Lookup::Author),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::FailedCacheHit,
                judged.clone(),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::PartialCacheHit,
                judged,
            ),
        ];
        for (pure_core, tweet_row, author, expected) in rows {
            for request_author_id in [None, Some(10)] {
                let (sources, broken) = lookups(pure_core, tweet_row, author);
                let scenario = Scenario::new(sources, SafetyLevel::TimelineHome)
                    .viewer(Some(50))
                    .fallback_caches(FallbackCaches {
                        tweets: Some(tweet_fallback_cache(8)),
                        authors: Some(author_fallback_cache(8)),
                    });
                let run = || async {
                    let raw = [raw(1, request_author_id)];
                    scenario.evaluate(&raw).await.evaluations().remove(0)
                };
                run().await;
                for (source, fault) in broken {
                    scenario.sources().break_source(source, fault);
                }
                assert_eq!(
                    run().await,
                    expected,
                    "{pure_core:?} {tweet_row:?} {author:?} {request_author_id:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_node_failed_for_one_request_author_of_a_repeated_tweet_fails_every_occurrence() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[11])
            .fail_key(Source::GizmoduckAuthor, 10);
        let evaluated = Scenario::new(sources, SafetyLevel::TimelineHome)
            .viewer(Some(50))
            .evaluate(&[raw(1, Some(10)), raw(1, Some(11))])
            .await;
        assert_eq!(
            evaluated.evaluations(),
            vec![Evaluation::Failed(Lookup::Author); 2]
        );
    }

    #[tokio::test]
    async fn run_preserves_order_duplicates_unresolved_tweets_and_labels() {
        let labels = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(999_999, vf_pb::SafetyLabel::default())]),
        };
        let sources = InMemorySources::default()
            .tweet(2, 20)
            .tweet_features(1, TweetFeatures::default())
            .fail_key(Source::TesPureCoreData, 1)
            .authors(&[20])
            .labels(1, Default::default())
            .labels(2, labels);
        let response = Scenario::new(sources, SafetyLevel::TimelineHome)
            .evaluate(&[raw(2, Some(20)), raw(1, None), raw(2, Some(20))])
            .await;

        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.tweet_id)
                .collect::<Vec<_>>(),
            vec![TweetId(2), TweetId(1), TweetId(2)]
        );
        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.evaluation.clone())
                .collect::<Vec<_>>(),
            vec![
                Evaluation::complete(allow()),
                Evaluation::Failed(Lookup::Tweet),
                Evaluation::complete(allow()),
            ]
        );
        assert_eq!(
            response.outcomes[1].evaluation.verdict(),
            &Verdict::lookup_failed()
        );
        assert!(response
            .outcomes
            .iter()
            .all(|outcome| outcome.safety_labels.is_some()));
        assert!(!response.outcomes[0]
            .safety_labels
            .as_ref()
            .unwrap()
            .labels
            .is_empty());
        assert!(Arc::ptr_eq(
            response.outcomes[0].safety_labels.as_ref().unwrap(),
            response.outcomes[2].safety_labels.as_ref().unwrap()
        ));
    }
}
