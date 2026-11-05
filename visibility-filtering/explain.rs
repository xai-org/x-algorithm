pub use crate::hydration::{Hydrator, Hydrators, Lookup};
pub use crate::models::viewer::Client;
pub use crate::models::{
    AuthorLabel, ClientCapability, ClientSwitch, ClientSwitchSet, Decided, DropReason, Evaluation,
    FosnrReason, LimitedActionType, LimitedEngagement, LimitedEngagementReason, MediaInterstitial,
    MediaRestriction, Notice, NsfwViewerDropReason, TombstoneReason, TweetId, Verdict,
    VerifyBlurSupport, ViewerAge, ViewerProfile, Withholding,
};
pub use crate::params::CountryList;
pub use crate::post_card::{Author, CardMedia, CardMediaKind, PostCard};
pub use crate::post_evaluator::MergeSelection;
pub use crate::rules::{
    ArticleStageLeaf, AuthorFactsLeaf, AuthorLabelsLeaf, AuthorSafetyLeaf, CommunityModeratorLeaf,
    CommunityViewerRemovedLeaf, CompoundLeaf, Contribution, EdgeLeaf, Effect, NodeKind, NodeTrace,
    PolicyTrace, PostFactsLeaf, PostModerationLeaf, Predicate, ReactedToAuthorSafetyLeaf,
    ReactedToPostSafetyLabelsLeaf, RelationshipLeaf, ReplyControlLeaf, RequestLeaf, RowTrace, Rpc,
    SafetyLevel, Skip, SourceTweetIdLeaf, Truth, TweetLeaf, TweetSafetyLabelsLeaf,
    ViewerAccountLeaf, Witness,
};

use crate::hydration::fallback_cache::FallbackCaches;
use crate::hydration::sources::{test_users_metadata, Sources};
use crate::hydration::{request_context, HydratedTweet, Hydration};
use crate::ingress;
use crate::metrics::{Endpoint, Metrics, RequestLevel};
use crate::models::{
    EvaluationKey, PlacedCandidate, PostCandidate, RawCandidate, RequestFacts, RequestViewer,
    Viewer,
};
use crate::params::{ClientSwitches, CountryLists};
use crate::post_card::PostCards;
use crate::post_evaluator::{self, HydratedRequest, PostEvaluator};
use crate::result_label::{label, state_label, Class, NONE};
use crate::rules::RuleEngine;
use crate::server_deps::{self, CacheWarming};
use crate::treatment;
use futures::future::{join, join_all};
use std::num::NonZeroU64;
use std::sync::Arc;
use tokio::time::Instant;
use xai_x_thrift::tweet_service::TweetFieldsResultState;

pub struct Explainer {
    cards: Arc<dyn PostCards>,
    post_evaluator: PostEvaluator,
    client_switches: ClientSwitches,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Route {
    rpc: Rpc,
    level: SafetyLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{rpc:?} cannot carry {level:?}")]
pub struct Unroutable {
    pub rpc: Rpc,
    pub level: SafetyLevel,
}

impl Route {
    pub fn new(rpc: Rpc, level: SafetyLevel) -> Result<Self, Unroutable> {
        if level.is_carried_by(rpc) {
            Ok(Self { rpc, level })
        } else {
            Err(Unroutable { rpc, level })
        }
    }

    pub fn rpc(self) -> Rpc {
        self.rpc
    }

    pub fn level(self) -> SafetyLevel {
        self.level
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ask<const N: usize> {
    pub viewer: Option<NonZeroU64>,
    pub post: u64,
    pub country: Option<String>,
    pub client: Client,
    pub routes: [Route; N],
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Explanation<const N: usize> {
    pub post: Option<PostCard>,
    pub viewer: ViewerFacts,
    pub in_nsfw_gating_country: FactObservation,
    pub levels: [LevelOutcome; N],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactObservation {
    NotRead,
    Recorded {
        route: Route,
        node: usize,
        truth: Truth,
    },
}

impl<const N: usize> Explanation<N> {
    pub fn new(
        post: Option<PostCard>,
        viewer: ViewerFacts,
        in_nsfw_gating_country: FactObservation,
        levels: [LevelOutcome; N],
    ) -> Self {
        Self {
            post,
            viewer,
            in_nsfw_gating_country,
            levels,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LevelOutcome {
    pub route: Route,
    pub capability: ClientCapability,
    pub judgement: Judgement,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Provenance {
    pub requested: Option<PolicyTrace>,
    pub source_tweet_id: Option<TweetId>,
    pub source: Option<PolicyTrace>,
    pub selection: MergeSelection,
    pub evaluation: Evaluation,
}

impl LevelOutcome {
    pub fn new(
        route: Route,
        capability: ClientCapability,
        judgement: Judgement,
        provenance: Provenance,
    ) -> Self {
        Self {
            route,
            capability,
            judgement,
            provenance,
        }
    }

    pub fn evaluate_tweets_sends(&self) -> Option<Sent<'_>> {
        match self.route.rpc {
            Rpc::FilterTweets => None,
            Rpc::EvaluateTweets => Some(match &self.judgement {
                Judgement::Judged {
                    sent: Some(state), ..
                } => Sent::ResultState(state),
                Judgement::Judged { sent: None, .. }
                | Judgement::NotFound(_)
                | Judgement::Failed(_) => Sent::Failed,
            }),
        }
    }

    pub fn label(&self) -> (String, String) {
        match (self.route.rpc, &self.judgement) {
            (Rpc::FilterTweets, Judgement::Judged { state, .. }) => state_label(state),
            (Rpc::FilterTweets | Rpc::EvaluateTweets, judgement) => judgement.label(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[expect(
    clippy::large_enum_variant,
    reason = "an explanation holds one per level, and nearly every level judges"
)]
pub enum Judgement {
    #[non_exhaustive]
    Judged {
        state: TweetFieldsResultState,
        decided_by: Option<&'static str>,
        fail_open: Hydrators,
        failed: Hydrators,
        sent: Option<TweetFieldsResultState>,
    },
    NotFound(Lookup),
    Failed(Lookup),
}

impl Judgement {
    pub fn label(&self) -> (String, String) {
        match self {
            Self::Judged {
                sent: Some(state), ..
            } => state_label(state),
            Self::Judged { sent: None, .. } | Self::Failed(_) => label(Class::Failed, NONE),
            Self::NotFound(_) => label(Class::NotFound, NONE),
        }
    }

    pub fn hidden(&self) -> bool {
        match self {
            Self::Judged { state, .. } => match state {
                TweetFieldsResultState::Found(_) => false,
                TweetFieldsResultState::NotFound(_)
                | TweetFieldsResultState::Failed(_)
                | TweetFieldsResultState::Filtered(_) => true,
            },
            Self::NotFound(_) | Self::Failed(_) => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sent<'a> {
    ResultState(&'a TweetFieldsResultState),
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewerFacts {
    LoggedOut,
    NotRead {
        id: NonZeroU64,
        relationship: Relationship,
    },
    ReadFailed {
        id: NonZeroU64,
        relationship: Relationship,
    },
    NoUser {
        id: NonZeroU64,
        relationship: Relationship,
    },
    LoggedIn {
        id: NonZeroU64,
        profile: ViewerProfile,
        age_verified_18: Seen,
        relationship: Relationship,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relationship {
    pub is_author: bool,
    pub follows: Seen,
    pub blocks: Seen,
    pub mutes: Seen,
    pub blocked_by: Seen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seen {
    Yes,
    No,
    NotHydrated,
    Failed,
}

impl Seen {
    fn read(reads: Hydrators, failed: Hydrators, hydrator: Hydrator, yes: bool) -> Self {
        if !reads.contains(hydrator) {
            Self::NotHydrated
        } else if failed.contains(hydrator) {
            Self::Failed
        } else if yes {
            Self::Yes
        } else {
            Self::No
        }
    }

    fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Yes, _) | (_, Self::Yes) => Self::Yes,
            (Self::No, _) | (_, Self::No) => Self::No,
            (Self::Failed, _) | (_, Self::Failed) => Self::Failed,
            (Self::NotHydrated, Self::NotHydrated) => Self::NotHydrated,
        }
    }
}

impl Explainer {
    pub async fn connect(datacenter: &str) -> Self {
        let metadata = test_users_metadata();
        let serving = server_deps::serving(
            datacenter,
            Some(&metadata),
            CacheWarming::Off,
            crate::params::Experiments::none(),
        )
        .await;
        Self::new(
            Arc::new(serving.sources),
            serving.fallback_caches,
            serving.country_lists,
            serving.client_switches,
            serving.metrics,
        )
    }

    pub async fn explain<const N: usize>(&self, ask: &Ask<N>) -> Explanation<N> {
        let post = request_context(Instant::now(), None).scope(self.cards.post_card(ask.post));
        let ((viewer, in_nsfw_gating_country, levels), post) = join(self.judge(ask), post).await;
        Explanation::new(post, viewer, in_nsfw_gating_country, levels)
    }
}

struct RouteRun {
    route: Route,
    facts: RequestFacts,
    hydrated: HydratedRequest,
    provenance: Provenance,
}

impl Explainer {
    fn new(
        sources: Arc<impl Sources + PostCards + 'static>,
        fallback_caches: FallbackCaches,
        country_lists: Arc<CountryLists>,
        client_switches: ClientSwitches,
        metrics: Metrics,
    ) -> Self {
        let rule_engine = RuleEngine::with_country_lists(country_lists);
        Self {
            cards: Arc::clone(&sources) as Arc<dyn PostCards>,
            post_evaluator: PostEvaluator::new(
                sources,
                rule_engine,
                Arc::new(fallback_caches),
                metrics,
            ),
            client_switches,
        }
    }

    async fn judge<const N: usize>(
        &self,
        ask: &Ask<N>,
    ) -> (ViewerFacts, FactObservation, [LevelOutcome; N]) {
        let evaluated = join_all(ask.routes.map(|route| self.evaluate(ask, route))).await;
        #[expect(clippy::unreachable, reason = "join_all answers each of the N routes")]
        let Ok(evaluated) = <[RouteRun; N]>::try_from(evaluated) else {
            unreachable!()
        };
        let viewer = Self::viewer_facts(ask, &evaluated);
        let in_nsfw_gating_country = Self::in_nsfw_gating_country(&evaluated);
        (
            viewer,
            in_nsfw_gating_country,
            evaluated.map(|level| self.outcome(ask, level)),
        )
    }

    async fn evaluate<const N: usize>(&self, ask: &Ask<N>, route: Route) -> RouteRun {
        let viewer_id = ask.viewer.map(NonZeroU64::get);
        let requested = PlacedCandidate::standalone(RawCandidate {
            tweet_id: TweetId(ask.post),
            request_author_id: None,
        });
        let candidates = vec![requested];
        let (request, endpoint) = match route.rpc {
            Rpc::FilterTweets => (
                ingress::filter_tweets_batch(
                    self.post_evaluator.rule_engine(),
                    &self.client_switches,
                    RequestViewer::new(viewer_id, ask.country.clone()),
                    route.level,
                    candidates,
                ),
                Endpoint::FilterTweets,
            ),
            Rpc::EvaluateTweets => (
                ingress::evaluate_tweets_batch(
                    &self.client_switches,
                    RequestViewer::with_context(
                        viewer_id,
                        Some(ask.client.context(viewer_id, ask.country.as_deref())),
                    ),
                    route.level,
                    candidates,
                ),
                Endpoint::EvaluateTweets,
            ),
        };
        let facts = request.facts.clone();
        let metrics = self
            .post_evaluator
            .metrics()
            .request(endpoint, RequestLevel::Named(route.level));
        request_context(Instant::now(), None)
            .scope(async {
                let (hydrated, outcomes, mut traced) =
                    post_evaluator::hydrate_and_trace(&self.post_evaluator, request, &metrics)
                        .await;
                let source_tweet_id = hydrated
                    .hydration()
                    .tweet(requested.tweet_id())
                    .and_then(HydratedTweet::source_tweet_id);
                let provenance = Provenance {
                    requested: traced.policies.remove(&requested.key()),
                    source_tweet_id,
                    source: source_tweet_id.and_then(|source| {
                        traced
                            .policies
                            .remove(&EvaluationKey::retweet_source(source))
                    }),
                    selection: traced
                        .selections
                        .get(&requested.key())
                        .copied()
                        .unwrap_or(MergeSelection::RequestedOnly),
                    evaluation: outcomes
                        .into_iter()
                        .next()
                        .map_or(Evaluation::Failed(Lookup::Tweet), |outcome| {
                            outcome.evaluation
                        }),
                };
                RouteRun {
                    route,
                    facts,
                    hydrated,
                    provenance,
                }
            })
            .await
    }

    fn outcome<const N: usize>(&self, ask: &Ask<N>, level: RouteRun) -> LevelOutcome {
        let evaluation = &level.provenance.evaluation;
        let judgement = match evaluation {
            Evaluation::Complete { verdict, .. } | Evaluation::Partial { verdict, .. } => {
                let viewer = level.facts.viewer();
                let policies = self.client_switches.limited_actions_policies(
                    viewer.context(),
                    viewer.id(),
                    viewer.country_code(),
                    verdict.limited_engagement_reasons(),
                    &Metrics::none(),
                );
                Judgement::Judged {
                    state: treatment::thrift_result_state(verdict, level.route.level, &policies),
                    decided_by: treatment::decided_by(verdict),
                    fail_open: evaluation.fail_open_defaults(),
                    failed: level.hydrated.hydration().failed_nodes(TweetId(ask.post)),
                    sent: treatment::evaluate_tweets_state(
                        evaluation,
                        level.route.level,
                        &policies,
                    ),
                }
            }
            Evaluation::NotFound(lookup) => Judgement::NotFound(*lookup),
            Evaluation::Deleted(_) => Judgement::NotFound(Lookup::Tweet),
            Evaluation::Failed(lookup) => Judgement::Failed(*lookup),
        };
        LevelOutcome::new(
            level.route,
            level.facts.client_capability(),
            judgement,
            level.provenance,
        )
    }

    fn viewer_facts<const N: usize>(ask: &Ask<N>, evaluated: &[RouteRun]) -> ViewerFacts {
        let Some(id) = ask.viewer else {
            return ViewerFacts::LoggedOut;
        };
        let relationship = Self::relationship(id, TweetId(ask.post), evaluated);
        let hydrations: Vec<&Hydration> = evaluated
            .iter()
            .map(|level| level.hydrated.hydration())
            .collect();
        let Some(hydration) = Self::profile_hydration(evaluated) else {
            let planned = hydrations
                .iter()
                .any(|hydration| hydration.planned().contains(Hydrator::ViewerProfile));
            return if planned {
                ViewerFacts::ReadFailed { id, relationship }
            } else {
                ViewerFacts::NotRead { id, relationship }
            };
        };
        let (Viewer::LoggedIn { profile, .. }, true) =
            (&hydration.viewer().viewer, hydration.viewer_found())
        else {
            return ViewerFacts::NoUser { id, relationship };
        };
        ViewerFacts::LoggedIn {
            id,
            profile: profile.clone(),
            age_verified_18: hydrations
                .iter()
                .map(|&hydration| {
                    let labeled = matches!(
                        hydration.viewer().viewer,
                        Viewer::LoggedIn {
                            has_age_verified_18_label: true,
                            ..
                        }
                    );
                    Seen::read(
                        hydration.planned(),
                        hydration.viewer_failed(),
                        Hydrator::ViewerLabels,
                        labeled,
                    )
                })
                .fold(Seen::NotHydrated, Seen::union),
            relationship,
        }
    }

    fn profile_hydration(evaluated: &[RouteRun]) -> Option<&Hydration> {
        evaluated
            .iter()
            .map(|level| level.hydrated.hydration())
            .find(|hydration| {
                hydration.planned().contains(Hydrator::ViewerProfile)
                    && !hydration.viewer_failed().contains(Hydrator::ViewerProfile)
            })
    }

    fn in_nsfw_gating_country(evaluated: &[RouteRun]) -> FactObservation {
        let leaf = NodeKind::Leaf(Predicate::ViewerAccount(
            ViewerAccountLeaf::ViewerAccountOrRequestCountryIn(CountryList::NsfwGating),
        ));
        evaluated
            .iter()
            .find_map(|level| {
                let trace = level.provenance.requested.as_ref()?;
                let node = trace.nodes.iter().find(|node| node.kind == leaf)?;
                Some(FactObservation::Recorded {
                    route: level.route,
                    node: node.node,
                    truth: node.truth,
                })
            })
            .unwrap_or(FactObservation::NotRead)
    }

    fn relationship(viewer: NonZeroU64, post: TweetId, evaluated: &[RouteRun]) -> Relationship {
        let posts: Vec<(&Hydration, &PostCandidate)> = evaluated
            .iter()
            .filter_map(|level| {
                let hydration = level.hydrated.hydration();
                Some((hydration, hydration.tweet(post)?.candidate()?))
            })
            .collect();
        let edge = |node| {
            posts
                .iter()
                .map(|&(hydration, candidate)| {
                    Seen::read(
                        hydration.planned(),
                        candidate.failed,
                        node,
                        candidate.edges.contains(node),
                    )
                })
                .fold(Seen::NotHydrated, Seen::union)
        };
        Relationship {
            is_author: posts
                .iter()
                .any(|(_, candidate)| candidate.author.id == viewer.get()),
            follows: edge(Hydrator::Follows),
            blocks: edge(Hydrator::Blocks),
            mutes: edge(Hydrator::Mutes),
            blocked_by: edge(Hydrator::BlockedByAuthor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::Graph;
    use crate::filter_tweets::FilterTweetsEndpoint;
    use crate::hydration::sources::{suspended, InMemorySources};
    use crate::rules::WireLevel;
    use crate::scenario::{tweet_data, Wire};
    use strum::VariantArray;
    use vf_pb::tweet_evaluation::Outcome;
    use xai_core_entities::entities::GizmoduckUser;
    use xai_visibility_filtering_proto as vf_pb;
    use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;

    const VIEWER: u64 = 50;

    #[tokio::test]
    async fn a_retweet_is_explained_through_every_route_as_its_handler_answers() {
        use MergeSelection::{RequestedOnly, SourceRestrictions, SourceWithheld};
        use Rpc::{EvaluateTweets, FilterTweets};
        use SafetyLevel::*;
        let sources = Arc::new(
            InMemorySources::default()
                .retweet(4, 40, 6, Some(60))
                .tweet(6, 60)
                .authors(&[40])
                .user(60, suspended())
                .viewer(VIEWER, GizmoduckUser::default())
                .edge(Graph::Follows, VIEWER, 40),
        );
        let routes: Vec<Route> = [FilterTweets, EvaluateTweets]
            .into_iter()
            .flat_map(|rpc| {
                SafetyLevel::VARIANTS
                    .iter()
                    .filter_map(move |&level| Route::new(rpc, level).ok())
            })
            .collect();
        let ask = Ask::<9> {
            viewer: NonZeroU64::new(VIEWER),
            post: 4,
            country: Some("us".to_owned()),
            client: Client::Web,
            routes: routes.try_into().unwrap(),
        };
        let explanation = Box::pin(
            Explainer::new(
                Arc::clone(&sources),
                FallbackCaches::default(),
                Arc::new(CountryLists::starting_at_default()),
                ClientSwitches::for_tests(),
                Metrics::none(),
            )
            .explain(&ask),
        )
        .await;

        assert_eq!(explanation.post.map(|card| card.id), Some(4));
        let ViewerFacts::LoggedIn { relationship, .. } = explanation.viewer else {
            panic!("{:?}", explanation.viewer);
        };
        assert_eq!(
            relationship,
            Relationship {
                is_author: false,
                follows: Seen::Yes,
                blocks: Seen::No,
                mutes: Seen::No,
                blocked_by: Seen::No,
            }
        );
        let first_reader = Route::new(FilterTweets, TimelineHome).unwrap();
        assert!(
            matches!(
                explanation.in_nsfw_gating_country,
                FactObservation::Recorded { route, truth: Truth::False, .. } if route == first_reader
            ),
            "{:?}",
            explanation.in_nsfw_gating_country
        );
        let judged: Vec<_> = explanation
            .levels
            .iter()
            .map(|level| {
                let route = level.route;
                (
                    route.rpc(),
                    route.level(),
                    level.label(),
                    level.provenance.selection,
                )
            })
            .collect();
        let allowed = ("allow", "none");
        let dropped = ("drop", "unspecified");
        let author_suspended = ("drop", "author_is_suspended");
        assert_eq!(
            judged,
            [
                (FilterTweets, FilterAll, dropped, RequestedOnly),
                (FilterTweets, TimelineHome, allowed, RequestedOnly),
                (
                    FilterTweets,
                    TimelineHomeRecommendations,
                    allowed,
                    RequestedOnly
                ),
                (
                    FilterTweets,
                    ImmersiveExpandedRecommendations,
                    allowed,
                    RequestedOnly
                ),
                (EvaluateTweets, FilterAll, dropped, SourceWithheld),
                (
                    EvaluateTweets,
                    TimelineHome,
                    author_suspended,
                    SourceWithheld
                ),
                (
                    EvaluateTweets,
                    TimelineHomeRecommendations,
                    author_suspended,
                    SourceWithheld
                ),
                (
                    EvaluateTweets,
                    TimelineHomeHydration,
                    author_suspended,
                    SourceWithheld
                ),
                (
                    EvaluateTweets,
                    TimelineHomePromotedHydration,
                    allowed,
                    SourceRestrictions
                ),
            ]
            .map(|(rpc, level, (class, reason), selection)| {
                (rpc, level, (class.to_owned(), reason.to_owned()), selection)
            })
        );

        let wire = Wire::new(sources);
        let filter_tweets = FilterTweetsEndpoint::new(
            Arc::clone(wire.post_evaluator()),
            ClientSwitches::for_tests(),
        );
        for level in &explanation.levels {
            let route = level.route;
            match route.rpc() {
                FilterTweets => {
                    let proto = (0..=255)
                        .filter_map(|number| vf_pb::SafetyLevel::try_from(number).ok())
                        .find(|&proto| {
                            SafetyLevel::parse(WireLevel::FilterTweets(proto)) == Ok(route.level())
                        })
                        .unwrap();
                    let response = filter_tweets
                        .handle(tonic::Request::new(vf_pb::VisibilityFilterRequest {
                            safety_level: proto.into(),
                            tweets: vec![vf_pb::TweetInput {
                                tweet_id: 4,
                                author_id: None,
                            }],
                            viewer_id: Some(VIEWER),
                            country_code: Some("us".to_owned()),
                        }))
                        .await
                        .unwrap()
                        .into_inner();
                    let [result] = &response.results[..] else {
                        panic!("{response:?}");
                    };
                    assert_eq!(
                        (result.action.unwrap(), result.filtered_reason.clone()),
                        treatment::proto_action(level.provenance.evaluation.verdict()),
                        "{route:?}"
                    );
                }
                EvaluateTweets => {
                    let thrift = ThriftLevel::ENUM_VALUES
                        .iter()
                        .copied()
                        .find(|&thrift| {
                            SafetyLevel::parse(WireLevel::EvaluateTweets(thrift.0))
                                == Ok(route.level())
                        })
                        .unwrap();
                    let sent = wire
                        .send_with_context(
                            &Client::Web.context(Some(VIEWER), Some("us")),
                            vf_pb::EvaluateTweetsRequest {
                                safety_level: thrift.0,
                                viewer_id: Some(VIEWER),
                                tweets: vec![tweet_data(4, None)],
                            },
                        )
                        .await
                        .unwrap();
                    let Some(Sent::ResultState(state)) = level.evaluate_tweets_sends() else {
                        panic!("{level:?}");
                    };
                    assert_eq!(
                        sent,
                        [Outcome::ResultStateThriftCompact(
                            xai_x_thrift::serialize_compact(state).unwrap().into()
                        )],
                        "{route:?}"
                    );
                }
            }
        }
    }
}
