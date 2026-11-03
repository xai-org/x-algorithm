use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::decode::author::{AuthorColumn, DecodedActor, DecodedAuthor};
use crate::hydration::decode::tweet::{PureCoreColumn, TweetRowColumn};
use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::fallback_cache::{Column, FallbackCache, FallbackCaches};
use crate::hydration::fetcher::Owner;
use crate::hydration::hydrator::Edge;
use crate::hydration::inventory::Source;
use crate::hydration::metrics::{
    record_flock_missing_keys, record_keyed_hydrator_request, record_request_country,
    record_tweet_states, record_viewer_country,
};
use crate::hydration::plan::Group;
use crate::hydration::sources::{Expired, Sources};
use crate::hydration::store::{CallRequest, Landing, Store};
use crate::hydration::tweet_source::TweetState;
use crate::hydration::{Hydration, HydrationPlan, HydrationRequest, Hydrator};
use crate::metrics::families::{
    HYDRATOR_EXPANDED_BATCHES, HYDRATOR_EXPANDED_KEYS, HYDRATOR_LATENCY_MS,
    VIEWER_NOT_AGE_VERIFIED, VIEWER_STATE,
};
use crate::metrics::{Metrics, RequestLevel, RequestMetrics};
use crate::models::{
    ArticleLifecycle, CommunityModeration, PureCore, TweetFeatures, Viewer, ViewerAge,
};
use crate::params::CountryList;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use std::convert::identity;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use xai_core_entities::entities::{ConversationControl, QuotedTweet};
use xai_visibility_filtering_proto as vf_pb;
use xai_x_rpc::WithBudget;

#[derive(Debug, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum ViewerCountry {
    NoAllowedList,
    Found,
    NoRow,
    Failed,
}

#[derive(Debug, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum NoViewerState {
    LoggedOut,
    NotFound,
    Failed,
}

#[derive(Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum UnverifiedAge {
    Adult,
    NotStated,
}

fn age_verification_prompted(
    viewer: &Viewer,
    request_country: Option<&str>,
) -> Option<(&'static str, UnverifiedAge)> {
    let Viewer::LoggedIn {
        profile,
        has_age_verified_18_label,
        ..
    } = viewer
    else {
        return None;
    };
    if *has_age_verified_18_label || profile.has_idv_premium {
        return None;
    }
    let age = match profile.viewer_age {
        ViewerAge::Known(_) if profile.is_underage() => return None,
        ViewerAge::Known(_) => UnverifiedAge::Adult,
        ViewerAge::NotStated => UnverifiedAge::NotStated,
        ViewerAge::Unknown => return None,
    };
    let code = request_country?;
    let country = [CountryList::AgeVerification, CountryList::Tombstone]
        .into_iter()
        .find_map(|list| list.names_by_default(code))?;
    Some((country, age))
}

fn record_age_verification_prompted(
    metrics: &Metrics,
    plan: &HydrationPlan,
    hydration: &Hydration,
) {
    let answered = !hydration.viewer_failed.contains(Hydrator::ViewerLabels)
        && !hydration.viewer_failed.contains(Hydrator::ViewerProfile);
    if plan.reads_viewer_labels()
        && answered
        && let Some((country, age)) = age_verification_prompted(
            &hydration.viewer.viewer,
            hydration.viewer.country_code.as_deref(),
        )
    {
        metrics.count(&VIEWER_NOT_AGE_VERIFIED, [country, age.into()], 1);
    }
}

fn record_viewer(metrics: &RequestMetrics, plan: &HydrationPlan, hydration: &Hydration) {
    record_age_verification_prompted(metrics.plain(), plan, hydration);
    if !plan.reads_viewer_profile() {
        return;
    }
    let state = match &hydration.viewer.viewer {
        Viewer::LoggedOut => NoViewerState::LoggedOut.into(),
        Viewer::LoggedIn { .. } if hydration.viewer_failed.contains(Hydrator::ViewerProfile) => {
            NoViewerState::Failed.into()
        }
        Viewer::LoggedIn { .. } => hydration
            .viewer_state
            .map_or(NoViewerState::NotFound.into(), <&str>::from),
    };
    metrics.count(&VIEWER_STATE, [state], 1);
    record_request_country(metrics.plain(), hydration.viewer.country_code.as_deref());
}

pub(super) enum Reply {
    PureCores(RawHydrationBatch<PureCore>),
    Tweets(RawHydrationBatch<TweetFeatures>),
    Controls(RawHydrationBatch<ConversationControl>),
    QuotedTweets(RawHydrationBatch<QuotedTweet>),
    TweetStates(RawHydrationBatch<TweetState>),
    Labels(RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>),
    Viewer(RawHydrationBatch<DecodedViewer>),
    Authors(RawHydrationBatch<DecodedAuthor>),
    Actors(RawHydrationBatch<DecodedActor>),
    ActorEmails(RawHydrationBatch<bool>),
    ActorPhones(RawHydrationBatch<bool>),
    Select(Vec<RawHydrationBatch<bool>>),
    Edge(Edge, RawHydrationBatch<bool>),
    ViewerCountry(RawHydrationBatch<Arc<str>>),
    CommunityModerations(RawHydrationBatch<CommunityModeration>),
    CommunityModerators(RawHydrationBatch<bool>),
    CommunityViewerRemovals(RawHydrationBatch<bool>),
    ArticleLifecycles(RawHydrationBatch<ArticleLifecycle>),
    ArticleMediaEntities(RawHydrationBatch<bool>),
    ArticleCoverMedia(RawHydrationBatch<bool>),
}

type InFlight<'p> = Pin<Box<dyn Future<Output = (CallRequest<'p>, Reply)> + Send + 'p>>;

struct Timed<'p> {
    metrics: &'p RequestMetrics,
    backend: &'static str,
    call: &'static str,
    call_deadline: Duration,
    candidate_count_by_claimed_key: FxHashMap<u64, usize>,
}

impl<'p> Timed<'p> {
    async fn run<V, A: AsRef<[RawHydrationBatch<V>]>>(
        &self,
        call: impl Future<Output = Result<A, Expired>>,
        failed: impl FnOnce(RawHydrationBatch<V>) -> A,
    ) -> A {
        let start = Instant::now();
        let answer = call
            .with_budget(self.call_deadline)
            .await
            .unwrap_or(Err(Expired));
        record_keyed_hydrator_request(
            self.metrics,
            self.backend,
            self.call,
            &self.candidate_count_by_claimed_key,
            answer.as_ref(),
            start.elapsed().as_secs_f64() * 1000.0,
        );
        answer.unwrap_or_else(|Expired| {
            let claimed = self.candidate_count_by_claimed_key.keys().copied();
            failed(HydrationBatch::failed(claimed))
        })
    }

    fn keyed<V, G>(
        self,
        call: CallRequest<'p>,
        get: G,
        after: impl FnOnce(RawHydrationBatch<V>) -> RawHydrationBatch<V> + Send + 'p,
        reply: fn(RawHydrationBatch<V>) -> Reply,
    ) -> InFlight<'p>
    where
        V: Send + 'p,
        G: for<'c> FnOnce(&'c CallRequest<'p>) -> BoxFuture<'c, RawHydrationBatch<V>> + Send + 'p,
    {
        Box::pin(async move {
            let get = async { Ok(get(&call).await) };
            let answer = self.run(get, identity).await;
            (call, reply(after(answer)))
        })
    }
}

impl HydrationPlan {
    pub(crate) async fn hydrate(
        &self,
        sources: &dyn Sources,
        request: HydrationRequest<'_>,
    ) -> Hydration {
        debug_assert!(
            matches!(request.metrics.level(), RequestLevel::Named(level) if level == self.level()),
            "{:?} hydrates under another request's metrics",
            self.level()
        );
        let started = Instant::now();
        let mut store = Store::new(
            self,
            request.viewer_id,
            request.client_capability,
            &request.raw_candidates,
            request.retweet_merge,
        )
        .with_quotes(&request.quotes)
        .with_users(&request.users);
        let mut running: FuturesUnordered<InFlight<'_>> = FuturesUnordered::new();
        let inputless = || self.groups().filter(|g| g.input.is_none());
        let mut ready: Vec<&Group> = inputless().collect();
        loop {
            while let Some(group) = ready.pop() {
                if let Some(call) = store.offer(group) {
                    let call_deadline = self.call_deadline(group, call.claimed_edges());
                    running.extend(Self::send(
                        call,
                        call_deadline,
                        sources,
                        request.fallback_caches,
                        request.metrics,
                    ));
                }
            }
            let Some((call, reply)) = running.next().await else {
                break;
            };
            let group = call.group;
            if store.land(&call, reply, started.elapsed()) == Landing::KeysJoined {
                ready.extend(inputless());
            }
            ready.extend(self.readers(group));
        }
        let metrics = request.metrics;
        if store.is_country_lookup_skipped() {
            record_viewer_country(metrics, ViewerCountry::NoAllowedList.into());
        }
        let tes_joined = store
            .tweets_elapsed
            .map_or(store.core_elapsed, |tweets| tweets.max(store.core_elapsed));
        metrics.plain().observe(
            &HYDRATOR_LATENCY_MS,
            ["tes", "join"],
            tes_joined.as_secs_f64() * 1000.0,
        );
        let hydration = store.assemble(request);
        record_viewer(metrics, self, &hydration);
        hydration
    }

    fn send<'p>(
        mut call: CallRequest<'p>,
        call_deadline: Duration,
        sources: &'p dyn Sources,
        fallback_caches: &'p FallbackCaches,
        metrics: &'p RequestMetrics,
    ) -> Option<InFlight<'p>> {
        let plain = metrics.plain();
        let group = call.group;
        let owner = group.fetcher.owner;
        let tweet_cache = fallback_caches.tweets.as_ref();
        let author_cache = fallback_caches
            .authors
            .as_ref()
            .filter(|_| owner == Owner::Request);
        let (backend, call_name) = group.label();
        if !call.is_first {
            let keys = call.key_count as u64;
            plain.count(&HYDRATOR_EXPANDED_BATCHES, [backend, call_name], 1);
            plain.count(&HYDRATOR_EXPANDED_KEYS, [backend, call_name], keys);
        }
        let timed = Timed {
            metrics,
            backend,
            call: call_name,
            call_deadline,
            candidate_count_by_claimed_key: mem::take(&mut call.candidate_count_by_claimed_key),
        };
        let in_flight: InFlight<'p> = match group.fetcher.source {
            Source::TesPureCoreData => timed.keyed(
                call,
                move |call| sources.pure_cores(owner, &call.keys),
                move |pure_cores| fall_back::<PureCoreColumn>(plain, tweet_cache, pure_cores),
                Reply::PureCores,
            ),
            Source::TesTweetForVisibility => timed.keyed(
                call,
                |call| sources.tweets(&call.keys),
                move |tweets| fall_back::<TweetRowColumn>(plain, tweet_cache, tweets),
                Reply::Tweets,
            ),
            Source::TesConversationControl => timed.keyed(
                call,
                |call| sources.conversation_controls(&call.keys),
                identity,
                Reply::Controls,
            ),
            Source::TesQuotedTweet => timed.keyed(
                call,
                |call| sources.quoted_tweets(&call.keys),
                identity,
                Reply::QuotedTweets,
            ),
            Source::TesTweetState => timed.keyed(
                call,
                |call| sources.tweet_states(&call.keys),
                move |states| record_tweet_states(metrics, states),
                Reply::TweetStates,
            ),
            Source::SafetyLabels => timed.keyed(
                call,
                move |call| sources.safety_labels(owner, &call.keys),
                identity,
                Reply::Labels,
            ),
            Source::GizmoduckViewer => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |_| sources.viewer(viewer_id, group.fields()),
                    identity,
                    Reply::Viewer,
                )
            }
            Source::GizmoduckAuthor => timed.keyed(
                call,
                move |call| sources.users(owner, &call.keys, group.fields()),
                move |authors| fall_back::<AuthorColumn>(plain, author_cache, authors),
                Reply::Authors,
            ),
            Source::GizmoduckActor => timed.keyed(
                call,
                |call| sources.actors(&call.keys, group.fields()),
                identity,
                Reply::Actors,
            ),
            Source::ActorEmails => timed.keyed(
                call,
                |call| sources.confirmed_emails(&call.keys),
                identity,
                Reply::ActorEmails,
            ),
            Source::ActorPhones => timed.keyed(
                call,
                |call| sources.verified_phones(&call.keys),
                identity,
                Reply::ActorPhones,
            ),
            Source::Flock => Box::pin(async move {
                let select = async {
                    let edges = sources
                        .select_edges(owner, call.viewer_id, call.queries())
                        .await?;
                    let (edges, missing) = missing_sets_read_no_edge(edges);
                    record_flock_missing_keys(metrics, timed.backend, timed.call, missing);
                    Ok(edges)
                };
                let per_query = |failed| vec![failed; call.queries().len()];
                let edges = timed.run(select, per_query).await;
                (call, Reply::Select(edges))
            }),
            Source::ViewerCountry => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |_| sources.viewer_country(viewer_id),
                    move |country| {
                        let result = match country.hydrated(&viewer_id) {
                            Some(Hydrated::Found(_)) => ViewerCountry::Found,
                            Some(Hydrated::NotFound) => ViewerCountry::NoRow,
                            _ => ViewerCountry::Failed,
                        };
                        record_viewer_country(metrics, result.into());
                        country
                    },
                    Reply::ViewerCountry,
                )
            }
            Source::Wingman => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.second_degree(viewer_id, &call.keys),
                    identity,
                    |answers| Reply::Edge(Edge::SecondDegree, answers),
                )
            }
            Source::TrustedFriends => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.trusted_friends(viewer_id, &call.keys),
                    identity,
                    |answers| Reply::Edge(Edge::TrustedFriends, answers),
                )
            }
            Source::UserLocation => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.outside_places(viewer_id, &call.keys),
                    identity,
                    |answers| Reply::Edge(Edge::OutsidePlace, answers),
                )
            }
            Source::CommunityModeration => timed.keyed(
                call,
                |call| sources.community_moderations(&call.community_posts),
                identity,
                Reply::CommunityModerations,
            ),
            Source::CommunityModerator => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.community_moderators(viewer_id, &call.keys),
                    identity,
                    Reply::CommunityModerators,
                )
            }
            Source::CommunityViewerRemoved => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.community_viewer_removals(viewer_id, &call.keys),
                    identity,
                    Reply::CommunityViewerRemovals,
                )
            }
            Source::ArticleMediaEntities => timed.keyed(
                call,
                |call| sources.article_media_entities(call.viewer_id, &call.keys),
                identity,
                Reply::ArticleMediaEntities,
            ),
            Source::ArticleCoverMedia => timed.keyed(
                call,
                |call| sources.article_cover_media(call.viewer_id, &call.keys),
                identity,
                Reply::ArticleCoverMedia,
            ),
            Source::ArticleLifecycle => timed.keyed(
                call,
                |call| sources.article_lifecycles(&call.keys),
                identity,
                Reply::ArticleLifecycles,
            ),
        };
        Some(in_flight)
    }
}

fn fall_back<C: Column>(
    metrics: &Metrics,
    cache: Option<&FallbackCache<C::Entry>>,
    batch: RawHydrationBatch<C::Value>,
) -> RawHydrationBatch<C::Value> {
    match cache {
        Some(cache) => cache.resolve_hydration_batch::<C>(metrics, batch),
        None => batch,
    }
}

fn missing_sets_read_no_edge(
    edges: Vec<RawHydrationBatch<bool>>,
) -> (Vec<RawHydrationBatch<bool>>, usize) {
    let mut missing = FxHashSet::default();
    let edges = edges
        .into_iter()
        .map(|edge| {
            let mut answers = edge.into_hydrated();
            for (destination, answer) in &mut answers {
                if let Hydrated::Partial(holds) = *answer {
                    *answer = Hydrated::Found(holds);
                    missing.insert(*destination);
                }
            }
            HydrationBatch::from_hydrated(answers)
        })
        .collect();
    (edges, missing.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeDirection, EdgeFrom, EdgeQuery, Graph};
    use crate::hydration::decode::author::fallback_cache;
    use crate::hydration::decode::tweet::tweet_fallback_cache;
    use crate::hydration::fetcher::FetcherId;
    use crate::hydration::sources::{Fault, InMemorySources, control, suspended};
    use crate::hydration::{
        BEST_EFFORT_HYDRATION_TIMEOUT, Cause, HYDRATION_TIMEOUT, Hydrator, Hydrators, Lookup,
        Relation, Unresolved,
    };
    use crate::metrics::families::HYDRATOR_TWEET_IDS;
    use crate::metrics::{Endpoint, RequestLevel};
    use crate::models::{
        ArticleLifecycle, AuthorId, ClientCapability, ClientSwitch, Evaluation, Placement,
        RetweetMerge, SafetyLabelType, TweetId, Viewer, ViewerProfile,
    };
    use crate::rules::{RuleEngine, SafetyLevel};
    use crate::scenario::{InRequestOrder, Scenario, label_map, protected_user, raw, reaction_to};
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::fmt::Write;
    use std::num::NonZeroU64;
    use xai_core_entities::entities::{
        ConversationControlArm, ExtendedProfile, GizmoduckUser, GizmoduckUserResult, PureCoreData,
        UserResponseState,
    };
    use xai_core_entities::gizmoduck_client::QueryFields;

    const VIEWER: u64 = 50;

    fn executor(
        sources: InMemorySources,
        level: SafetyLevel,
        viewer_id: Option<u64>,
    ) -> Scenario<InMemorySources> {
        let client = ClientCapability::DEFAULT.with(ClientSwitch::CommunityViewerRemovedLimits);
        Scenario::new(sources, level)
            .viewer(viewer_id)
            .country("US")
            .client(client)
    }

    fn tweet_failed() -> Unresolved {
        Unresolved {
            lookup: Lookup::Tweet,
            cause: Cause::Failed,
        }
    }

    fn ids(ids: &[u64]) -> HashSet<TweetId> {
        ids.iter().copied().map(TweetId).collect()
    }

    fn exclusive_tweet() -> TweetFeatures {
        TweetFeatures {
            exclusive_conversation_author_id: Some(30),
            ..Default::default()
        }
    }

    fn article_tweet(article_id: u64) -> TweetFeatures {
        TweetFeatures {
            article_id: NonZeroU64::new(article_id),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn failed_ids_reports_exactly_the_candidates_each_node_flags() {
        use ConversationControlArm::{Co, Subscribers};
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let rows = [
            ("healthy home", TimelineHome, world(), vec![], vec![]),
            (
                "healthy home hydration",
                TimelineHomeHydration,
                world(),
                vec![],
                vec![],
            ),
            (
                "failed pure core",
                TimelineHome,
                world().fault(Source::TesPureCoreData, Fault::Fails),
                vec![],
                vec![(TweetId(1), tweet_failed()), (TweetId(2), tweet_failed())],
            ),
            (
                "incomplete author",
                TimelineHome,
                world().user(
                    10,
                    GizmoduckUserResult {
                        response_state: Some(UserResponseState::Failed),
                        ..suspended()
                    },
                ),
                vec![],
                vec![(
                    TweetId(1),
                    Unresolved {
                        lookup: Lookup::Author,
                        cause: Cause::Failed,
                    },
                )],
            ),
            (
                "failed author-keyed select",
                TimelineHome,
                world().fail_graph(Graph::Mutes),
                vec![1, 2],
                vec![],
            ),
            (
                "failed blocked-by select",
                TimelineHomeHydration,
                world().fail_graph(Graph::Blocks),
                vec![1, 2],
                vec![],
            ),
            (
                "failed tweet row",
                TimelineHome,
                world().fail_key(Source::TesTweetForVisibility, 1),
                vec![],
                vec![(TweetId(1), tweet_failed())],
            ),
            (
                "failed label lookup",
                TimelineHome,
                world().fail_key(Source::SafetyLabels, 1),
                vec![1],
                vec![],
            ),
            (
                "failed exclusive select",
                TimelineHome,
                world()
                    .tweet_features(1, exclusive_tweet())
                    .fail_graph(Graph::SuperFollows),
                vec![1],
                vec![],
            ),
            (
                "failed root-edge select",
                TimelineHomeHydration,
                world()
                    .control(1, control(Subscribers, 30, &[]))
                    .fail_graph(Graph::SuperFollows),
                vec![1],
                vec![],
            ),
            (
                "failed conversation-control row",
                TimelineHomeHydration,
                world().fail_key(Source::TesConversationControl, 1),
                vec![1],
                vec![],
            ),
            (
                "failed country lookup",
                TimelineHomeHydration,
                world()
                    .control(1, control(Co, 30, &["us"]))
                    .control(2, control(Co, 30, &[]))
                    .fault(Source::ViewerCountry, Fault::Fails),
                vec![1],
                vec![],
            ),
            (
                "failed viewer",
                TimelineHome,
                world().fault(Source::GizmoduckViewer, Fault::Fails),
                vec![1, 2],
                vec![],
            ),
            (
                "failed article lifecycle",
                TimelineHomeHydration,
                world()
                    .tweet_features(1, article_tweet(70))
                    .fault(Source::ArticleLifecycle, Fault::Fails),
                vec![1],
                vec![],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, sources, failed, unresolved) in rows {
            let hydrated = executor(sources, level, Some(VIEWER)).hydrate(&raw).await;
            assert_eq!(
                (hydrated.failed_ids, hydrated.unresolved),
                (ids(&failed), unresolved.into_iter().collect()),
                "{name}"
            );
        }
    }

    #[tokio::test]
    async fn each_candidate_carries_the_nodes_that_failed_for_it() {
        use ConversationControlArm::MyNetwork;
        use Hydrator::{
            BlockedByAuthor, Blocks, Follows, MuteRetweets, Mutes, RootFollowsViewer,
            RootFollowsViewerSecondDegree,
        };
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let relationships = Hydrators::of(Follows).with(Blocks).with(Mutes);
        let rows = [
            (
                "failed tweet row",
                TimelineHome,
                Some(VIEWER),
                world()
                    .tweet_features(2, exclusive_tweet())
                    .fail_key(Source::TesTweetForVisibility, 1),
                [Err(tweet_failed()), Ok(Hydrators::empty())],
            ),
            (
                "failed root edge on a MyNetwork reply",
                TimelineHomeHydration,
                Some(VIEWER),
                world()
                    .control(1, control(MyNetwork, 30, &[]))
                    .fail_graph(Graph::Follows),
                [
                    Ok(Hydrators::of(RootFollowsViewer)
                        .with(RootFollowsViewerSecondDegree)
                        .with(Follows)
                        .with(BlockedByAuthor)),
                    Ok(Hydrators::of(Follows).with(BlockedByAuthor)),
                ],
            ),
            (
                "failed pure core, authors from the request",
                TimelineHome,
                Some(VIEWER),
                world().fault(Source::TesPureCoreData, Fault::Fails),
                [Err(tweet_failed()); 2],
            ),
            (
                "failed relationships select on a retweet and an original",
                TimelineHome,
                Some(VIEWER),
                world().retweet(1, 10, 5, Some(30)).fail_graph(Graph::Mutes),
                [Ok(relationships.with(MuteRetweets)), Ok(relationships)],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, viewer_id, sources, expected) in rows {
            let hydrated = executor(sources, level, viewer_id).hydrate(&raw).await;
            assert_eq!(hydrated.failed_nodes, expected, "{name}");
        }
    }

    #[tokio::test]
    async fn safety_labels_carry_only_the_tweets_whose_labels_were_found() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet(2, 20)
            .tweet(3, 30)
            .labels(2, Default::default())
            .fail_key(Source::SafetyLabels, 1);
        let raw = [raw(1, None), raw(2, None), raw(3, None)];
        let hydrated = executor(sources, SafetyLevel::TimelineHome, None)
            .hydrate(&raw)
            .await;
        assert_eq!(
            hydrated
                .safety_labels
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            ids(&[2])
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_best_effort_call_ends_at_its_ceiling_or_the_callers_budget_whichever_comes_first() {
        let ms = Duration::from_millis;
        for (remaining, ends_at) in [(ms(500), BEST_EFFORT_HYDRATION_TIMEOUT), (ms(200), ms(200))] {
            let sources = InMemorySources::default()
                .tweet(1, 10)
                .authors(&[10])
                .fault(Source::Flock, Fault::Hangs);
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
            let started = tokio::time::Instant::now();
            let hydrated = xai_x_rpc::CallContext {
                deadline: Some(started + remaining),
            }
            .scope(scenario.hydrate(&[raw(1, None)]))
            .await;
            assert_eq!(started.elapsed(), ends_at, "{remaining:?}");
            assert!(hydrated.candidates[0].failed.contains(Hydrator::Follows));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn author_calls_do_not_wait_for_tweets() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended())
            .edge(Graph::Follows, VIEWER, 10)
            .fault(Source::TesTweetForVisibility, Fault::Hangs);
        let raw = [raw(1, None)];
        let started = tokio::time::Instant::now();
        let scenario = executor(sources, SafetyLevel::TimelineHome, Some(VIEWER));
        let sources = scenario.sources();
        let hydration = scenario.hydrate(&raw);
        tokio::pin!(hydration);
        let early = tokio::time::timeout(HYDRATION_TIMEOUT / 2, &mut hydration).await;
        assert!(early.is_err());
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10]]);
        assert_eq!(sources.selects().len(), 1);

        let hydrated = hydration.await;
        assert_eq!(started.elapsed(), HYDRATION_TIMEOUT);
        assert_eq!(
            hydrated.unresolved,
            HashMap::from([(TweetId(1), tweet_failed())])
        );
    }

    fn sorted(mut calls: Vec<Source>) -> Vec<String> {
        let mut names: Vec<String> = calls.drain(..).map(|s| format!("{s:?}")).collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn empty_key_sets_and_logged_out_viewers_send_no_call() {
        let raw = [raw(1, None)];
        let logged_out = InMemorySources::default().tweet(1, 10);
        let scenario = executor(logged_out, SafetyLevel::TimelineHomeHydration, None);
        let logged_out = scenario.sources();
        scenario.hydrate(&raw).await;
        assert_eq!(
            sorted(logged_out.calls()),
            [
                "GizmoduckAuthor",
                "SafetyLabels",
                "TesConversationControl",
                "TesPureCoreData",
                "TesTweetForVisibility"
            ]
        );

        let logged_in = InMemorySources::default().tweet(1, 10);
        let scenario = executor(logged_in, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let logged_in = scenario.sources();
        scenario.hydrate(&raw).await;
        assert!(!logged_in.calls().contains(&Source::ViewerCountry));
        assert!(!logged_in.calls().contains(&Source::UserLocation));
        assert!(!logged_in.calls().contains(&Source::CommunityViewerRemoved));
        assert_eq!(
            logged_in.selects(),
            [vec![
                EdgeQuery {
                    graph: Graph::Follows,
                    direction: EdgeDirection::Forward,
                    from: EdgeFrom::Viewer,
                    destination_ids: vec![10],
                },
                EdgeQuery {
                    graph: Graph::Blocks,
                    direction: EdgeDirection::Reverse,
                    from: EdgeFrom::Viewer,
                    destination_ids: vec![10],
                },
            ]]
        );
    }

    #[tokio::test]
    async fn only_retweeters_are_asked_for_the_mute_retweets_edge() {
        let relationships = |authors: &[u64], retweeters: &[u64]| {
            vec![
                EdgeQuery::forward(Graph::Follows, authors.to_vec()),
                EdgeQuery::forward(Graph::Blocks, authors.to_vec()),
                EdgeQuery::forward(Graph::Mutes, authors.to_vec()),
                EdgeQuery::forward(Graph::MuteRetweets, retweeters.to_vec()),
            ]
        };
        let sources = InMemorySources::default()
            .retweet(1, 10, 5, Some(20))
            .tweet(2, 30)
            .tweet(5, 20);
        let scenario = executor(sources, SafetyLevel::TimelineHome, Some(VIEWER))
            .retweet_merge(RetweetMerge::Merged);
        scenario.hydrate(&[raw(1, None), raw(2, None)]).await;
        assert_eq!(
            scenario.sources().selects(),
            [relationships(&[10, 20, 30], &[10])]
        );

        let originals = InMemorySources::default().tweet(2, 30);
        let scenario = executor(originals, SafetyLevel::TimelineHome, Some(VIEWER));
        let originals = scenario.sources();
        scenario.hydrate(&[raw(2, None)]).await;
        assert_eq!(originals.selects(), [relationships(&[30], &[])]);
    }

    #[test]
    fn only_an_unverified_adult_or_no_stated_age_viewer_in_a_gated_country_is_prompted() {
        let viewer = |age, label, idv| Viewer::LoggedIn {
            id: VIEWER,
            profile: ViewerProfile {
                viewer_age: age,
                has_idv_premium: idv,
                ..ViewerProfile::default()
            },
            has_age_verified_18_label: label,
        };
        let adult = ViewerAge::Known(30);
        for (viewer, country, prompted) in [
            (
                viewer(adult, false, false),
                Some("gb"),
                Some(("gb", UnverifiedAge::Adult)),
            ),
            (
                viewer(ViewerAge::NotStated, false, false),
                Some("gb"),
                Some(("gb", UnverifiedAge::NotStated)),
            ),
            (viewer(ViewerAge::Known(15), false, false), Some("gb"), None),
            (viewer(ViewerAge::Unknown, false, false), Some("gb"), None),
            (Viewer::LoggedOut, Some("gb"), None),
            (viewer(adult, true, false), Some("gb"), None),
            (viewer(adult, false, true), Some("gb"), None),
            (viewer(adult, false, false), Some("ca"), None),
            (viewer(adult, false, false), None, None),
        ] {
            assert_eq!(
                age_verification_prompted(&viewer, country),
                prompted,
                "{viewer:?} {country:?}"
            );
        }
    }

    #[tokio::test]
    async fn filter_all_calls_pure_core_only_and_keeps_the_request_side_viewer() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet_features(1, exclusive_tweet());
        let scenario = executor(sources, SafetyLevel::FilterAll, Some(VIEWER));
        let sources = scenario.sources();
        let hydrated = scenario.hydrate(&[raw(1, None)]).await;
        assert_eq!(sources.calls(), [Source::TesPureCoreData]);
        assert_eq!(
            hydrated.viewer_features.viewer,
            Viewer::LoggedIn {
                id: VIEWER,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: false,
            }
        );
        assert_eq!(hydrated.viewer_features.country_code.as_deref(), Some("us"));
        assert_eq!(hydrated.candidates[0].author.id, 10);
        assert!(hydrated.safety_labels.is_empty());
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_failed_viewer_call_fails_both_viewer_nodes_unverified() {
        use crate::hydration::plan::HydrationPlan;
        use xai_core_entities::entities::{Label, Labels};
        use xai_x_thrift::user_labels::LabelValue;
        let viewer_nodes = Hydrators::of(Hydrator::ViewerProfile).with(Hydrator::ViewerLabels);
        let labeled = GizmoduckUser {
            labels: Labels {
                labels: vec![Label {
                    label_value: LabelValue::AGE_VERIFIED_18.0,
                    created_at_msec: 0,
                }],
            },
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .viewer(VIEWER, labeled)
            .fault(Source::GizmoduckViewer, Fault::Fails);
        let level = SafetyLevel::TimelineHomeHydration;
        let plan = HydrationPlan::new(level, viewer_nodes);
        let hydrated = executor(sources, level, Some(VIEWER))
            .hydrate_with_plan(&plan, &[raw(1, None)])
            .await;
        assert_eq!(hydrated.candidates[0].failed, viewer_nodes);
        assert_eq!(
            hydrated.viewer_features.viewer,
            Viewer::LoggedIn {
                id: VIEWER,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: false,
            }
        );
    }

    #[tokio::test]
    async fn a_level_that_plans_the_viewer_profile_decodes_it() {
        let sources = InMemorySources::default().viewer(
            VIEWER,
            GizmoduckUser {
                extended_profile: Some(ExtendedProfile {
                    age_in_years: Some(30),
                }),
                ..Default::default()
            },
        );
        let hydrated = executor(sources, SafetyLevel::TimelineHome, Some(VIEWER))
            .hydrate(&[])
            .await;
        let Viewer::LoggedIn { profile, .. } = hydrated.viewer_features.viewer else {
            panic!("a logged-in request stays logged in")
        };
        assert_ne!(profile, ViewerProfile::default());
    }

    #[tokio::test]
    async fn exclusive_edges_dedup_conversation_authors() {
        let sources = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .tweet(3, 40)
                .authors(&[10, 20, 40])
                .tweet_features(1, exclusive_tweet())
                .tweet_features(2, exclusive_tweet())
                .edge(Graph::SuperFollows, VIEWER, 30)
        };
        let raw = [raw(1, None), raw(2, None), raw(1, None), raw(3, Some(40))];
        for viewer_id in [Some(VIEWER), None] {
            let sources = sources();
            let scenario = executor(sources, SafetyLevel::TimelineHome, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            assert_eq!(sources.keys(Source::TesPureCoreData), [vec![1, 2, 3]]);
            assert_eq!(sources.keys(Source::TesTweetForVisibility), [vec![1, 2, 3]]);
            let exclusive = (Some(30), viewer_id.is_some());
            assert_eq!(
                hydrated
                    .candidates
                    .iter()
                    .map(|c| (
                        c.post.tweet_features.exclusive_conversation_author_id,
                        c.edges.contains(Hydrator::SuperFollowsExclusive)
                    ))
                    .collect::<Vec<_>>(),
                [exclusive, exclusive, exclusive, (None, false)]
            );
            let super_follows: Vec<Vec<u64>> = sources
                .selects()
                .into_iter()
                .flatten()
                .filter(|query| query.graph == Graph::SuperFollows)
                .map(|query| query.destination_ids)
                .collect();
            let expected: &[Vec<u64>] = if viewer_id.is_some() {
                &[vec![30]]
            } else {
                &[]
            };
            assert_eq!(super_follows, expected);
        }
    }

    #[tokio::test]
    async fn a_super_follows_key_two_groups_read_is_asked_once() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet(2, 20)
            .authors(&[10, 20])
            .tweet_features(1, exclusive_tweet())
            .control(2, control(ConversationControlArm::Subscribers, 30, &[]))
            .edge(Graph::SuperFollows, VIEWER, 30);
        let raw = [raw(1, None), raw(2, None)];
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let sources = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        let super_follows: Vec<Vec<u64>> = sources
            .selects()
            .into_iter()
            .flatten()
            .filter(|query| query.graph == Graph::SuperFollows)
            .map(|query| query.destination_ids)
            .collect();
        assert_eq!(super_follows, [vec![30]]);
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (
                    c.edges.contains(Hydrator::SuperFollowsExclusive),
                    c.edges.contains(Hydrator::SuperFollowsRoot)
                ))
                .collect::<Vec<_>>(),
            [(true, false), (false, true)]
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn a_root_select_claiming_the_exclusive_selects_key_runs_to_the_exclusive_selects_ceiling()
     {
        let ms = Duration::from_millis;
        for (latency, ends_at, answered) in [(ms(500), ms(500), true), (ms(1500), ms(1000), false)]
        {
            let world = InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .authors(&[10, 20])
                .tweet_features(1, exclusive_tweet())
                .control(2, control(ConversationControlArm::Subscribers, 30, &[]))
                .edge(Graph::SuperFollows, VIEWER, 30)
                .fault(Source::TesTweetForVisibility, Fault::Delays(ms(1)))
                .key_latency(Source::Flock, 30, latency);
            let scenario = executor(world, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
            let sources = scenario.sources();
            let started = tokio::time::Instant::now();
            let hydrated = scenario.hydrate(&[raw(1, None), raw(2, None)]).await;
            assert_eq!(started.elapsed(), ends_at, "{latency:?}");
            let super_follows: Vec<Vec<Graph>> = sources
                .selects()
                .into_iter()
                .filter(|queries| queries.iter().any(|q| q.graph == Graph::SuperFollows))
                .map(|queries| queries.iter().map(|query| query.graph).collect())
                .collect();
            assert_eq!(
                super_follows,
                [vec![
                    Graph::Follows,
                    Graph::SuperFollows,
                    Graph::Follows,
                    Graph::Blocks
                ]],
                "only the root select asks: {latency:?}"
            );
            let [exclusive, root] = &hydrated.candidates[..] else {
                panic!("two candidates");
            };
            assert_eq!(
                (
                    exclusive.edges.contains(Hydrator::SuperFollowsExclusive),
                    exclusive.failed.contains(Hydrator::SuperFollowsExclusive),
                    root.edges.contains(Hydrator::SuperFollowsRoot),
                    root.failed.contains(Hydrator::SuperFollowsRoot),
                ),
                (answered, !answered, answered, !answered),
                "{latency:?}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_root_select_claiming_no_super_follows_key_ends_at_its_own_ceiling() {
        let world = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .control(1, control(ConversationControlArm::Followers, 30, &[]))
            .key_latency(Source::Flock, 30, Duration::from_millis(500));
        let scenario = executor(world, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let started = tokio::time::Instant::now();
        let hydrated = scenario.hydrate(&[raw(1, None)]).await;
        assert_eq!(started.elapsed(), BEST_EFFORT_HYDRATION_TIMEOUT);
        assert!(
            hydrated.candidates[0]
                .failed
                .contains(Hydrator::ViewerFollowsRoot)
        );
    }

    fn root_edges(sources: &InMemorySources) -> Vec<Vec<EdgeQuery>> {
        sources
            .selects()
            .into_iter()
            .filter(|queries| {
                queries.iter().any(|query| {
                    query.graph == Graph::Follows && query.direction == EdgeDirection::Reverse
                })
            })
            .collect()
    }

    #[tokio::test]
    async fn one_select_carries_every_root_edge_and_a_failure_fails_every_tweet_it_keyed() {
        use ConversationControlArm::{ByInvitation, Community, Followers, MyNetwork, Subscribers};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .tweet(5, 10)
                .authors(&[10])
                .control(1, control(Community, 30, &[]))
                .control(2, control(MyNetwork, 30, &[]))
                .control(3, control(Subscribers, 40, &[]))
                .control(4, control(ByInvitation, 40, &[]))
                .control(5, control(Followers, 50, &[]))
                .edge(Graph::Follows, 30, VIEWER)
                .edge(Graph::SuperFollows, VIEWER, 40)
                .edge(Graph::Follows, VIEWER, 50)
        };
        let raw = [
            raw(1, None),
            raw(2, None),
            raw(3, None),
            raw(4, None),
            raw(5, None),
        ];
        let facts = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| {
                    (
                        c.edges.contains(Hydrator::RootFollowsViewer),
                        c.edges.contains(Hydrator::SuperFollowsRoot),
                        c.edges.contains(Hydrator::ViewerFollowsRoot),
                    )
                })
                .collect::<Vec<_>>()
        };

        let healthy = world();
        let scenario = executor(healthy, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let healthy = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(
            root_edges(healthy),
            [vec![
                EdgeQuery {
                    graph: Graph::Follows,
                    direction: EdgeDirection::Reverse,
                    from: EdgeFrom::Viewer,
                    destination_ids: vec![30],
                },
                EdgeQuery {
                    graph: Graph::SuperFollows,
                    direction: EdgeDirection::Forward,
                    from: EdgeFrom::Viewer,
                    destination_ids: vec![40],
                },
                EdgeQuery {
                    graph: Graph::Follows,
                    direction: EdgeDirection::Forward,
                    from: EdgeFrom::Viewer,
                    destination_ids: vec![50],
                },
                EdgeQuery {
                    graph: Graph::Blocks,
                    direction: EdgeDirection::Reverse,
                    from: EdgeFrom::Viewer,
                    destination_ids: vec![30, 40, 50],
                },
            ]]
        );
        assert_eq!(
            facts(&hydrated),
            [
                (true, false, false),
                (true, false, false),
                (false, true, false),
                (false, false, false),
                (false, false, true),
            ]
        );
        assert!(hydrated.failed_ids.is_empty());

        let failed = world().fail_graph(Graph::SuperFollows);
        let hydrated = executor(failed, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
            .hydrate(&raw)
            .await;
        assert_eq!(facts(&hydrated), [(false, false, false); 5]);
        assert_eq!(hydrated.failed_ids, ids(&[1, 2, 3, 5]));

        let logged_out = world();
        let scenario = executor(logged_out, SafetyLevel::TimelineHomeHydration, None);
        let logged_out = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert!(root_edges(logged_out).is_empty());
        assert_eq!(facts(&hydrated), [(false, false, false); 5]);
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_query_missing_from_the_select_answer_reads_no_edge_and_fails_no_candidate() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .edge(Graph::Follows, VIEWER, 10)
            .edge(Graph::Mutes, VIEWER, 10)
            .miss_graph(Graph::Mutes);
        let hydrated = executor(sources, SafetyLevel::TimelineHome, Some(VIEWER))
            .hydrate(&[raw(1, None)])
            .await;
        assert_eq!(
            hydrated.candidates[0].edges,
            crate::models::EdgeSet::default().with(Hydrator::Follows)
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn wingman_asks_once_for_my_network_roots_the_followed_by_edge_answered_no() {
        use ConversationControlArm::{Community, MyNetwork};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .tweet(5, 10)
                .tweet(6, 10)
                .authors(&[10])
                .control(1, control(Community, 30, &[]))
                .control(2, control(MyNetwork, 30, &[]))
                .control(3, control(MyNetwork, 60, &[]))
                .control(4, control(MyNetwork, 60, &[]))
                .control(5, control(MyNetwork, 70, &[]))
                .control(6, control(Community, 80, &[]))
                .edge(Graph::Follows, 30, VIEWER)
                .second_degree_path(60, VIEWER)
        };
        let raw = [1, 2, 3, 4, 5, 6].map(|id| raw(id, None));
        let second_degree = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| c.edges.contains(Hydrator::RootFollowsViewerSecondDegree))
                .collect::<Vec<_>>()
        };
        let level = SafetyLevel::TimelineHomeHydration;

        let healthy = world();
        let scenario = executor(healthy, level, Some(VIEWER));
        let healthy = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(healthy.keys(Source::Wingman), [vec![60, 70]]);
        assert_eq!(
            second_degree(&hydrated),
            [false, false, true, true, false, false]
        );
        assert!(hydrated.failed_ids.is_empty());

        let failed = world().fault(Source::Wingman, Fault::Fails);
        let hydrated = executor(failed, level, Some(VIEWER)).hydrate(&raw).await;
        assert_eq!(second_degree(&hydrated), [false; 6]);
        assert!(hydrated.failed_ids.is_empty());

        for (sources, viewer_id, raw) in [
            (world().fail_graph(Graph::Follows), Some(VIEWER), &raw[..]),
            (world(), None, &raw[..]),
            (world(), Some(VIEWER), &raw[..2]),
        ] {
            let scenario = executor(sources, level, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(raw).await;
            assert!(sources.keys(Source::Wingman).is_empty());
            assert_eq!(second_degree(&hydrated), vec![false; raw.len()]);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_timed_out_root_edge_call_fails_its_nodes_but_leaves_the_post_complete_whichever_input_lands_first()
     {
        let level = SafetyLevel::TimelineHomeHydration;
        for delayed in [Source::TesConversationControl, Source::TesPureCoreData] {
            let sources = InMemorySources::default()
                .tweet(1, 10)
                .authors(&[10])
                .control(1, control(ConversationControlArm::Subscribers, 40, &[]))
                .edge(Graph::SuperFollows, VIEWER, 40)
                .fault(delayed, Fault::Delays(BEST_EFFORT_HYDRATION_TIMEOUT / 2))
                .hang_graph(Graph::SuperFollows);
            let hydrated = executor(sources, level, Some(VIEWER))
                .hydrate(&[raw(1, None)])
                .await;
            assert_eq!(
                hydrated.candidates[0].failed,
                Hydrators::of(Hydrator::SuperFollowsRoot),
                "{delayed:?} lands last"
            );
            let engine = RuleEngine::for_tests();
            let evaluation = engine.evaluate(
                level,
                &hydrated.viewer_features,
                &hydrated.candidates[0],
                Placement::Standalone,
            );
            assert!(matches!(evaluation, Evaluation::Complete { .. }));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn only_the_blocked_by_node_a_post_reads_can_become_its_fail_open_default() {
        use ConversationControlArm::Subscribers;
        use Hydrator::BlockedByConversationRoot;
        let level = SafetyLevel::TimelineHomeHydration;
        let reply = |conversation_id, parent_id, parent_author_id| PureCoreData {
            author_id: 10,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(parent_id),
            in_reply_to_user_id: Some(parent_author_id),
            ..Default::default()
        };
        let control_fails =
            || InMemorySources::default().fault(Source::TesConversationControl, Fault::Fails);
        let rows = [
            (
                "a conversation root post whose conversation control call fails",
                control_fails().tweet(1, 10),
                &[1][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a conversation root post whose conversation root select claims its author and times out",
                InMemorySources::default()
                    .tweet(1, 10)
                    .control(1, control(Subscribers, 10, &[]))
                    .fault(
                        Source::TesPureCoreData,
                        Fault::Delays(BEST_EFFORT_HYDRATION_TIMEOUT / 2),
                    )
                    .hang_graph(Graph::SuperFollows),
                &[1][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a direct reply whose conversation control call fails",
                control_fails().pure_core(1, reply(100, 100, 30)),
                &[1][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a nested reply whose conversation root author is in the batch and whose conversation control call fails",
                control_fails().pure_core(1, reply(100, 2, 11)).pure_core(
                    2,
                    PureCoreData {
                        author_id: 11,
                        ..reply(100, 100, 30)
                    },
                ),
                &[1, 2][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a nested reply whose conversation root is outside the batch and whose conversation control call fails",
                control_fails().pure_core(1, reply(100, 200, 11)),
                &[1][..],
                BlockedByConversationRoot,
                true,
            ),
        ];
        let engine = RuleEngine::for_tests();
        for (name, sources, batch, node, reports) in rows {
            let hydrated = executor(sources.authors(&[10, 11, 30]), level, Some(VIEWER))
                .hydrate(&batch.iter().map(|&id| raw(id, None)).collect::<Vec<_>>())
                .await;
            let evaluation = engine.evaluate(
                level,
                &hydrated.viewer_features,
                &hydrated.candidates[0],
                Placement::Standalone,
            );
            assert_eq!(
                evaluation.fail_open_defaults().contains(node),
                reports,
                "{name}: {node:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_conversation_root_post_whose_author_blocks_the_viewer_reads_only_the_author_block() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .control(1, control(ConversationControlArm::Community, 10, &[]))
            .edge(Graph::Blocks, 10, VIEWER);
        let hydrated = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
            .hydrate(&[raw(1, None)])
            .await;
        let candidate = &hydrated.candidates[0];
        assert!(candidate.edges.contains(Hydrator::BlockedByAuthor));
        assert!(!candidate.post.conversation_root_author_blocks_viewer);
    }

    #[tokio::test]
    async fn a_reply_asks_whether_its_conversation_roots_author_blocks_the_viewer() {
        let reply = |author_id, conversation_id, parent_id, parent_author_id| PureCoreData {
            author_id,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(parent_id),
            in_reply_to_user_id: Some(parent_author_id),
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .pure_core(1, reply(10, 100, 100, 30))
            .pure_core(2, reply(11, 100, 1, 10))
            .pure_core(3, reply(12, 200, 201, 98))
            .tweet(200, 40)
            .authors(&[10, 11, 12]);
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        scenario
            .hydrate(&[raw(1, None), raw(2, None), raw(3, None)])
            .await;
        let sources = scenario.sources();
        assert_eq!(
            sources.selects(),
            [
                vec![
                    EdgeQuery::forward(Graph::Follows, vec![10, 11, 12]),
                    EdgeQuery::reverse(Graph::Blocks, vec![10, 11, 12, 30]),
                ],
                vec![EdgeQuery::reverse(Graph::Blocks, vec![40])],
            ]
        );
        assert_eq!(sources.keys(Source::TesPureCoreData), [vec![1, 2, 3]]);
        assert_eq!(
            sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
            [vec![200]]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_conversation_root_post_fails_only_the_nested_replies_under_it() {
        use Hydrator::{BlockedByConversationRootPost, ConversationRootPost};
        let nested = |author_id, conversation_id| PureCoreData {
            author_id,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(conversation_id + 1),
            in_reply_to_user_id: Some(99),
            ..Default::default()
        };
        let replies = || {
            InMemorySources::default()
                .pure_core(1, nested(10, 100))
                .tweet(2, 20)
                .pure_core(3, nested(11, 300))
                .pure_core(4, nested(12, 100))
                .authors(&[10, 11, 12, 20])
                .edge(Graph::Blocks, 30, VIEWER)
        };
        let world = || replies().tweet(100, 30).tweet(300, 31);
        let author_select = vec![
            EdgeQuery::forward(Graph::Follows, vec![10, 11, 12, 20]),
            EdgeQuery::reverse(Graph::Blocks, vec![10, 11, 12, 20]),
        ];
        let root_failed: Result<Hydrators, Unresolved> =
            Ok(Hydrators::of(ConversationRootPost).with(BlockedByConversationRootPost));
        let complete: Result<Hydrators, Unresolved> = Ok(Hydrators::empty());
        let rows = [
            (
                "root call times out",
                world().key_latency(
                    (Relation::ConversationRoot, Source::TesPureCoreData),
                    100,
                    BEST_EFFORT_HYDRATION_TIMEOUT * 2,
                ),
                [root_failed, complete, root_failed, root_failed],
                vec![author_select.clone()],
                vec![("conversation_root_pure_core_data", "timeout", 3)],
            ),
            (
                "one root key fails",
                world().fail_key((Relation::ConversationRoot, Source::TesPureCoreData), 100),
                [root_failed, complete, complete, root_failed],
                vec![
                    author_select.clone(),
                    vec![EdgeQuery::reverse(Graph::Blocks, vec![31])],
                ],
                vec![
                    ("conversation_root_author_blocked_by", "success", 1),
                    ("conversation_root_pure_core_data", "error", 2),
                    ("conversation_root_pure_core_data", "success", 1),
                ],
            ),
            (
                "no root is found",
                replies(),
                [complete; 4],
                vec![author_select],
                vec![("conversation_root_pure_core_data", "success", 3)],
            ),
        ];
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let level = SafetyLevel::TimelineHomeHydration;
        for (name, sources, failed, selects, tweet_ids) in rows {
            let (metrics, recording) = Metrics::recording();
            let scenario = executor(sources, level, Some(VIEWER)).metrics(metrics);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            let mut recorded: Vec<(String, String, usize)> = recording
                .take(&HYDRATOR_TWEET_IDS.family)
                .iter()
                .filter(|sample| {
                    matches!(
                        sample.label("call"),
                        Some(
                            "conversation_root_pure_core_data"
                                | "conversation_root_author_blocked_by"
                        )
                    )
                })
                .map(|sample| {
                    let label = |name| sample.label(name).unwrap_or_default().to_owned();
                    (label("call"), label("result"), sample.value as usize)
                })
                .collect();
            recorded.sort();
            let tweet_ids: Vec<(String, String, usize)> = tweet_ids
                .into_iter()
                .map(|(call, result, count)| (call.to_owned(), result.to_owned(), count))
                .collect();
            assert_eq!(recorded, tweet_ids, "{name}");
            assert_eq!(hydrated.failed_nodes, failed, "{name}");
            assert!(
                hydrated
                    .candidates
                    .iter()
                    .all(|c| !c.post.conversation_root_author_blocks_viewer),
                "{name}"
            );
            assert_eq!(
                sources.keys(Source::TesPureCoreData),
                [vec![1, 2, 3, 4]],
                "{name}"
            );
            assert_eq!(
                sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
                [vec![100, 300]],
                "{name}"
            );
            assert_eq!(sources.selects(), selects, "{name}");
        }
    }

    #[tokio::test]
    async fn each_article_column_asks_each_distinct_article_once() {
        use ArticleLifecycle::{Draft, Published};
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        for viewer_id in [Some(VIEWER), None] {
            let sources = InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .authors(&[10])
                .tweet_features(1, article_tweet(70))
                .tweet_features(2, article_tweet(70))
                .tweet_features(3, article_tweet(80))
                .lifecycle(70, Draft)
                .lifecycle(80, Published);
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            for source in [
                Source::ArticleLifecycle,
                Source::ArticleMediaEntities,
                Source::ArticleCoverMedia,
            ] {
                assert_eq!(
                    sources.keys(source),
                    [vec![70, 80]],
                    "{viewer_id:?} {source:?}"
                );
            }
            for source in [Source::ArticleMediaEntities, Source::ArticleCoverMedia] {
                assert_eq!(sources.article_viewers(source), [viewer_id], "{source:?}");
            }
            assert_eq!(
                hydrated
                    .candidates
                    .iter()
                    .map(|c| c.post.article_lifecycle)
                    .collect::<Vec<_>>(),
                [Some(Draft), Some(Draft), Some(Published), None],
                "{viewer_id:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_failed_article_column_reads_as_no_media_and_leaves_the_post_complete() {
        use crate::hydration::sources::found_user;
        use crate::models::NsfwViewerDropReason;
        use crate::rules::fixtures::{allow, nsfw_viewer_dropped};
        use xai_core_entities::entities::{ExtendedProfile, GizmoduckUser};
        let mut author = found_user(10);
        author.user.as_mut().unwrap().safety.nsfw_user = true;
        let dropped = nsfw_viewer_dropped(
            NsfwViewerDropReason::IsUnderage,
            "sensitive_viewer_underage/drop",
        );
        for (failing, verdict) in [
            (&[Source::ArticleMediaEntities][..], dropped.clone()),
            (&[Source::ArticleCoverMedia][..], dropped),
            (
                &[Source::ArticleMediaEntities, Source::ArticleCoverMedia][..],
                allow(),
            ),
        ] {
            let mut world = InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .user(10, author.clone())
                .viewer(
                    VIEWER,
                    GizmoduckUser {
                        user_id: VIEWER,
                        extended_profile: Some(ExtendedProfile {
                            age_in_years: Some(17),
                        }),
                        ..Default::default()
                    },
                )
                .tweet_features(1, article_tweet(70))
                .tweet_features(2, article_tweet(70))
                .tweet_features(3, article_tweet(80))
                .lifecycle(70, ArticleLifecycle::Published)
                .lifecycle(80, ArticleLifecycle::Published)
                .article_media_entities(70, true)
                .article_cover_media(70, true);
            for &source in failing {
                world = world.fail_key(source, 70);
            }
            let scenario = executor(world, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
            let hydrated = scenario
                .hydrate(&[raw(1, None), raw(2, None), raw(3, None), raw(4, None)])
                .await;
            assert_eq!(
                hydrated.failed_nodes,
                [
                    Ok(Hydrators::empty()),
                    Ok(Hydrators::empty()),
                    Ok(Hydrators::empty()),
                    Ok(Hydrators::empty())
                ],
                "{failing:?}"
            );
            let evaluations = scenario
                .evaluate(&[raw(1, None), raw(2, None)])
                .await
                .evaluations();
            let expected = Evaluation::complete(verdict);
            assert_eq!(evaluations, [expected.clone(), expected], "{failing:?}");
        }
    }

    #[tokio::test]
    async fn one_country_lookup_reaches_every_co_tweet_that_needs_it() {
        use ConversationControlArm::Co;
        let raw = [raw(1, None), raw(2, None)];
        let world = |countries: &[&str]| {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .authors(&[10])
                .control(1, control(Co, 30, countries))
                .control(2, control(Co, 30, countries))
                .country(VIEWER, "us")
        };
        let country = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| {
                    c.post
                        .conversation_control
                        .as_ref()
                        .unwrap()
                        .viewer_country
                        .as_deref()
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        };

        let listed = world(&["us"]);
        let scenario = executor(listed, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let listed = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(listed.keys(Source::ViewerCountry), [vec![VIEWER]]);
        assert_eq!(
            country(&hydrated),
            [Some("us".to_owned()), Some("us".to_owned())]
        );

        for (sources, viewer_id) in [(world(&[]), Some(VIEWER)), (world(&["us"]), None)] {
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            assert!(sources.keys(Source::ViewerCountry).is_empty());
            assert_eq!(country(&hydrated), [None, None]);
        }
    }

    #[tokio::test]
    async fn a_reaction_post_asks_its_own_calls_for_its_reacted_to_posts_labels_and_authors_safety()
    {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet_features(1, reaction_to(7, 70))
            .tweet(2, 20)
            .tweet_features(2, reaction_to(1, 20))
            .tweet(3, 30)
            .labels(7, label_map(SafetyLabelType::NSFW_HIGH_PRECISION))
            .authors(&[10, 20, 30])
            .user(70, protected_user(70));
        let requested = [raw(1, None), raw(2, None), raw(3, None)];
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let sources = scenario.sources();
        let hydrated = scenario.hydrate(&requested).await;
        for (fetcher, asked) in [
            (Source::SafetyLabels.into(), vec![1, 2, 3]),
            (
                FetcherId::from((Relation::ReactedTo, Source::SafetyLabels)),
                vec![1, 7],
            ),
            (Source::GizmoduckAuthor.into(), vec![10, 20, 30]),
            (
                (Relation::ReactedTo, Source::GizmoduckAuthor).into(),
                vec![20, 70],
            ),
        ] {
            let mut keys: Vec<u64> = sources.keys(fetcher).concat();
            keys.sort_unstable();
            assert_eq!(keys, asked, "{fetcher:?}");
        }
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (
                    c.post
                        .reacted_to_post_safety_labels
                        .has_label(SafetyLabelType::NSFW_HIGH_PRECISION),
                    c.post.reacted_to_author_is_protected,
                ))
                .collect::<Vec<_>>(),
            [(true, true), (false, false), (false, false)]
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_batch_without_a_reaction_post_sends_no_reacted_to_call() {
        let plain = executor(
            InMemorySources::default().tweet(3, 30).authors(&[30]),
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
        );
        plain.hydrate(&[raw(3, None)]).await;
        assert_eq!(plain.sources().keys(Source::SafetyLabels), [vec![3]]);
        assert_eq!(plain.sources().keys(Source::GizmoduckAuthor), [vec![30]]);
        for source in [
            (Relation::ReactedTo, Source::SafetyLabels),
            (Relation::ReactedTo, Source::GizmoduckAuthor),
        ] {
            assert!(plain.sources().keys(source).is_empty(), "{source:?}");
        }
    }

    #[tokio::test]
    async fn a_failed_or_missing_reacted_to_lookup_shows_its_reaction_post_unrestricted() {
        use crate::rules::fixtures::allow;
        use Hydrator::{ReactedToAuthorSafety, ReactedToPostSafetyLabels};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, reaction_to(7, 70))
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let shown = |fail_open_defaults| Evaluation::Complete {
            verdict: allow(),
            fail_open_defaults,
        };
        let rows = [
            (
                "reacted-to post's labels fail",
                world()
                    .labels(7, label_map(SafetyLabelType::NSFW_HIGH_PRECISION))
                    .fail_key((Relation::ReactedTo, Source::SafetyLabels), 7),
                Hydrators::of(ReactedToPostSafetyLabels),
            ),
            (
                "reacted-to author fails",
                world()
                    .user(70, protected_user(70))
                    .fail_key((Relation::ReactedTo, Source::GizmoduckAuthor), 70),
                Hydrators::of(ReactedToAuthorSafety),
            ),
            (
                "reacted-to post and author not found",
                world(),
                Hydrators::empty(),
            ),
        ];
        let raw = [raw(1, None), raw(2, None)];
        for (name, sources, failed) in rows {
            let evaluations = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
                .retweet_merge(RetweetMerge::Merged)
                .evaluate(&raw)
                .await
                .evaluations();
            assert_eq!(
                evaluations,
                [shown(failed), shown(Hydrators::empty())],
                "{name}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_reaction_posts_reacted_to_lookups_decide_it_when_its_pure_core_lands_last() {
        use crate::rules::fixtures::{blurred, dropped};
        use xai_visibility_filtering::models::FilteredReason;
        use xai_x_thrift::action::InterstitialReason;
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, reaction_to(7, 70))
                .key_latency(
                    Source::TesPureCoreData,
                    1,
                    BEST_EFFORT_HYDRATION_TIMEOUT / 10,
                )
                .authors(&[10, 70])
        };
        let rows = [
            (
                "reacted-to author protected",
                world().user(70, protected_user(70)),
                Evaluation::complete(dropped(
                    FilteredReason::AuthorIsProtected,
                    "protected_original_author/drop",
                )),
            ),
            (
                "reacted-to post labelled nsfw high precision",
                world().labels(7, label_map(SafetyLabelType::NSFW_HIGH_PRECISION)),
                Evaluation::complete(blurred(
                    InterstitialReason::Sensitive(true),
                    "original_tweet_adult_media_propagation/blur/sensitive",
                )),
            ),
            (
                "reacted-to post labelled gore high precision",
                world().labels(
                    7,
                    label_map(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
                ),
                Evaluation::complete(blurred(
                    InterstitialReason::Violence(true),
                    "original_tweet_gore_media_propagation/blur/violence",
                )),
            ),
        ];
        for (name, sources, expected) in rows {
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
                .retweet_merge(RetweetMerge::Merged);
            let evaluations = scenario.evaluate(&[raw(1, None)]).await.evaluations();
            assert_eq!(evaluations, [expected], "{name}");
            let sources = scenario.sources();
            assert_eq!(
                sources.keys((Relation::ReactedTo, Source::SafetyLabels)),
                [vec![7]],
                "{name}"
            );
            assert_eq!(
                sources.keys((Relation::ReactedTo, Source::GizmoduckAuthor)),
                [vec![70]],
                "{name}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_partial_slow_or_hung_reacted_to_lookup_leaves_the_requests_lookups_of_its_keys_complete()
     {
        use crate::rules::fixtures::allow;
        use Hydrator::{ReactedToAuthorSafety, ReactedToPostSafetyLabels};
        fn world() -> InMemorySources {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, reaction_to(7, 20))
                .key_latency(
                    Source::TesPureCoreData,
                    1,
                    BEST_EFFORT_HYDRATION_TIMEOUT / 10,
                )
                .authors(&[10, 20, 30])
        }
        type World = fn() -> InMemorySources;
        type Broken = fn(InMemorySources) -> InMemorySources;
        let worlds: [(&str, World); 3] = [
            ("reacted-to author also writes a requested post", || {
                world().tweet(2, 20)
            }),
            (
                "reacted-to author also writes a retweet's source the share names no author for",
                || world().retweet(2, 30, 5, None).tweet(5, 20),
            ),
            ("reacted-to post is also a retweet's source", || {
                world().retweet(2, 30, 7, Some(20)).tweet(7, 20)
            }),
        ];
        let both = Hydrators::of(ReactedToPostSafetyLabels).with(ReactedToAuthorSafety);
        let faults: [(&str, Broken, Hydrators); 4] = [
            (
                "fails",
                |sources| {
                    sources
                        .fail_key((Relation::ReactedTo, Source::SafetyLabels), 7)
                        .fail_key((Relation::ReactedTo, Source::GizmoduckAuthor), 20)
                },
                both,
            ),
            (
                "outlasts its ceiling",
                |sources| {
                    let slow = BEST_EFFORT_HYDRATION_TIMEOUT * 2;
                    sources
                        .key_latency((Relation::ReactedTo, Source::SafetyLabels), 7, slow)
                        .key_latency((Relation::ReactedTo, Source::GizmoduckAuthor), 20, slow)
                },
                both,
            ),
            (
                "hangs",
                |sources| {
                    sources
                        .fault((Relation::ReactedTo, Source::SafetyLabels), Fault::Hangs)
                        .fault((Relation::ReactedTo, Source::GizmoduckAuthor), Fault::Hangs)
                },
                both,
            ),
            (
                "answers partial",
                |sources| {
                    sources.fault(
                        (Relation::ReactedTo, Source::GizmoduckAuthor),
                        Fault::AnswersPartial,
                    )
                },
                Hydrators::of(ReactedToAuthorSafety),
            ),
        ];
        let requested = [raw(1, None), raw(2, None)];
        for (world, sources) in worlds {
            for (fault, broken, fail_open_defaults) in faults {
                let reaction = Evaluation::Complete {
                    verdict: allow(),
                    fail_open_defaults,
                };
                let scenario = executor(
                    broken(sources()),
                    SafetyLevel::TimelineHomeHydration,
                    Some(VIEWER),
                )
                .retweet_merge(RetweetMerge::Merged);
                let evaluations = scenario.evaluate(&requested).await.evaluations();
                assert_eq!(
                    evaluations,
                    [reaction, Evaluation::complete(allow())],
                    "{world}, the reacted-to lookup {fault}"
                );
                let hydrated = scenario.hydrate(&requested).await;
                assert_eq!(
                    hydrated.failed_nodes,
                    [Ok(fail_open_defaults), Ok(Hydrators::empty())],
                    "{world}, the reacted-to lookup {fault}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_reacted_to_author_lookup_neither_reads_nor_writes_the_author_fallback_cache() {
        let sources = InMemorySources::default()
            .tweet(1, 70)
            .tweet(2, 10)
            .tweet_features(2, reaction_to(7, 70))
            .tweet(3, 10)
            .tweet_features(3, reaction_to(8, 71))
            .tweet(4, 71)
            .authors(&[10])
            .user(70, protected_user(70))
            .user(71, protected_user(71));
        let caches = FallbackCaches {
            tweets: None,
            authors: Some(fallback_cache(8)),
        };
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
            .retweet_merge(RetweetMerge::Merged)
            .fallback_caches(caches);
        let first = scenario.hydrate(&[raw(1, None), raw(3, None)]).await;
        assert!(first.candidates[1].post.reacted_to_author_is_protected);

        let sources = scenario.sources();
        sources.break_source(Source::GizmoduckAuthor, Fault::Fails);
        sources.break_source((Relation::ReactedTo, Source::GizmoduckAuthor), Fault::Fails);
        let second = scenario
            .hydrate(&[raw(1, None), raw(2, None), raw(4, None)])
            .await;
        assert!(second.candidates[0].author.features.is_protected);
        assert!(!second.candidates[1].post.reacted_to_author_is_protected);
        assert_eq!(
            second.failed_nodes,
            [
                Ok(Hydrators::empty()),
                Ok(Hydrators::of(Hydrator::ReactedToAuthorSafety)),
                Err(Unresolved {
                    lookup: Lookup::Author,
                    cause: Cause::Failed,
                }),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_conversation_root_that_is_also_a_retweet_source_is_fetched_by_both_calls() {
        use crate::rules::fixtures::allow;
        let world_naming = |source_author| {
            InMemorySources::default()
                .pure_core(
                    1,
                    PureCoreData {
                        author_id: 10,
                        conversation_id: Some(100),
                        in_reply_to_tweet_id: Some(101),
                        in_reply_to_user_id: Some(99),
                        ..Default::default()
                    },
                )
                .retweet(2, 20, 100, source_author)
                .tweet(100, 30)
                .authors(&[10, 20, 30])
        };
        let world = || world_naming(Some(30));
        let root_failed = Evaluation::Complete {
            verdict: allow(),
            fail_open_defaults: Hydrators::of(Hydrator::BlockedByConversationRootPost),
        };
        let complete = Evaluation::complete(allow());
        let slow = BEST_EFFORT_HYDRATION_TIMEOUT * 2;
        let rows = [
            ("both answer", world(), [complete.clone(), complete.clone()]),
            (
                "the root call fails",
                world().fail_key((Relation::ConversationRoot, Source::TesPureCoreData), 100),
                [root_failed.clone(), complete.clone()],
            ),
            (
                "the root call times out",
                world().key_latency(
                    (Relation::ConversationRoot, Source::TesPureCoreData),
                    100,
                    slow,
                ),
                [root_failed.clone(), complete.clone()],
            ),
            (
                "the retweet-source call fails",
                world().fail_key(Source::TesPureCoreData, 100),
                [complete.clone(), Evaluation::Failed(Lookup::SharedTweet)],
            ),
            (
                "the root's blocked-by select claims an unauthored source's author and times out",
                world_naming(None)
                    .key_latency(
                        Source::TesPureCoreData,
                        100,
                        BEST_EFFORT_HYDRATION_TIMEOUT / 2,
                    )
                    .key_latency(Source::Flock, 30, HYDRATION_TIMEOUT / 2),
                [
                    root_failed.clone(),
                    Evaluation::Complete {
                        verdict: allow(),
                        fail_open_defaults: Hydrators::of(Hydrator::BlockedByAuthor),
                    },
                ],
            ),
        ];
        for (name, sources, expected) in rows {
            let (metrics, recording) = Metrics::recording();
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
                .retweet_merge(RetweetMerge::Merged)
                .metrics(metrics);
            let evaluations = scenario
                .evaluate(&[raw(1, None), raw(2, None)])
                .await
                .evaluations();
            let calls: BTreeSet<String> = recording
                .take(&HYDRATOR_TWEET_IDS.family)
                .iter()
                .filter(|sample| sample.label("backend") == Some("tes"))
                .filter_map(|sample| Some(sample.label("call")?.to_owned()))
                .filter(|call| call.ends_with("pure_core_data"))
                .collect();
            assert_eq!(
                calls,
                BTreeSet::from([
                    "conversation_root_pure_core_data".to_owned(),
                    "tweet_pure_core_data".to_owned()
                ]),
                "{name}"
            );
            let sources = scenario.sources();
            assert_eq!(
                sources.keys(Source::TesPureCoreData),
                [vec![1, 2], vec![100]],
                "{name}"
            );
            assert_eq!(
                sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
                [vec![100]],
                "{name}"
            );
            assert_eq!(evaluations, expected, "{name}");
        }
    }

    #[tokio::test]
    async fn a_conversation_root_lookup_reads_and_writes_the_tweet_fallback_cache() {
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
            .tweet(100, 30)
            .tweet(200, 40)
            .authors(&[10, 11, 30, 40]);
        let caches = FallbackCaches {
            tweets: Some(tweet_fallback_cache(8)),
            authors: None,
        };
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
            .fallback_caches(caches);
        let first = scenario
            .hydrate(&[raw(1, None), raw(2, None), raw(200, None)])
            .await;
        assert_eq!(first.failed_nodes, [Ok(Hydrators::empty()); 3]);

        let sources = scenario.sources();
        sources.break_source(Source::TesPureCoreData, Fault::Fails);
        sources.break_source(
            (Relation::ConversationRoot, Source::TesPureCoreData),
            Fault::Fails,
        );
        let second = scenario.hydrate(&[raw(100, None), raw(2, None)]).await;
        assert_eq!(second.failed_nodes, [Ok(Hydrators::empty()); 2]);
        assert_eq!(
            sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
            [vec![100], vec![200]]
        );
    }

    #[tokio::test]
    async fn authors_share_one_key_and_a_missing_user_is_complete() {
        let sources = InMemorySources::default().tweet(1, 10).tweet(2, 10);
        let raw = [raw(1, None), raw(2, None)];
        let scenario = executor(sources, SafetyLevel::TimelineHome, None);
        let sources = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10]]);
        assert!(
            hydrated
                .candidates
                .iter()
                .all(|c| !c.author.features.is_suspended)
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn repeated_ids_share_the_first_occurrences_resolution() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended());
        let raw = [raw(1, Some(10)), raw(1, Some(20)), raw(1, None)];
        let hydrated = executor(sources, SafetyLevel::TimelineHome, None)
            .hydrate(&raw)
            .await;
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (c.author.id, c.author.features.is_suspended))
                .collect::<Vec<_>>(),
            [(10, true); 3]
        );
        assert!(hydrated.failed_ids.is_empty());
        assert!(hydrated.unresolved.is_empty());
    }

    #[tokio::test]
    async fn user_rows_ask_the_actor_source_by_user_without_pure_core_and_ask_a_repeated_user_once()
    {
        use crate::hydration::{HydratedUser, RequestRows};
        let sources = InMemorySources::default()
            .user(10, suspended())
            .authors(&[20])
            .edge(Graph::Blocks, VIEWER, 20)
            .fail_key(Source::GizmoduckActor, 30);
        let plan = HydrationPlan::with_rows(
            SafetyLevel::QuotedTweetRules,
            RequestRows::Users,
            Hydrators::of(Hydrator::AuthorSafety).with(Hydrator::Blocks),
        );
        let caches = FallbackCaches {
            tweets: None,
            authors: None,
        };
        let users = [10, 20, 10, 30].map(AuthorId).to_vec();
        let hydration = plan
            .hydrate(
                &sources,
                HydrationRequest::for_users(
                    VIEWER,
                    users,
                    &caches,
                    &Metrics::none().request(
                        Endpoint::EvaluateTweets,
                        RequestLevel::Named(SafetyLevel::QuotedTweetRules),
                    ),
                ),
            )
            .await;
        assert_eq!(call_labels(&sources), ["Flock blocks", "GizmoduckActor"]);
        assert_eq!(sources.keys(Source::GizmoduckActor), [vec![10, 20, 30]]);
        assert!(matches!(
            hydration.user(AuthorId(10)),
            Some(HydratedUser::Resolved { user, edges, failed })
                if user.features.is_suspended && !edges.contains(Hydrator::Blocks) && failed.is_empty()
        ));
        assert!(matches!(
            hydration.user(AuthorId(20)),
            Some(HydratedUser::Resolved { edges, failed, .. })
                if edges.contains(Hydrator::Blocks) && failed.is_empty()
        ));
        assert!(matches!(
            hydration.user(AuthorId(30)),
            Some(HydratedUser::Unresolved(Cause::Failed))
        ));
    }

    #[tokio::test]
    async fn the_author_cache_serves_the_last_known_author_when_the_call_fails() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended());
        let raw = [raw(1, None)];
        let caches = FallbackCaches {
            tweets: None,
            authors: Some(fallback_cache(8)),
        };
        let scenario = executor(sources, SafetyLevel::TimelineHome, None).fallback_caches(caches);
        let sources = scenario.sources();
        let first = scenario.hydrate(&raw).await;
        assert!(first.candidates[0].author.features.is_suspended);

        sources.break_source(Source::GizmoduckAuthor, Fault::Fails);
        let second = scenario.hydrate(&raw).await;
        assert!(second.candidates[0].author.features.is_suspended);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn the_tweet_cache_serves_the_last_known_core_when_the_call_fails() {
        let sources = InMemorySources::default().tweet(1, 10).authors(&[10]);
        let raw = [raw(1, None)];
        let scenario =
            executor(sources, SafetyLevel::TimelineHome, None).fallback_caches(FallbackCaches {
                tweets: Some(tweet_fallback_cache(8)),
                authors: None,
            });
        let sources = scenario.sources();
        let first = scenario.hydrate(&raw).await;
        assert_eq!(first.candidates[0].author.id, 10);

        sources.break_source(Source::TesPureCoreData, Fault::Fails);
        let second = scenario.hydrate(&raw).await;
        assert_eq!(second.candidates[0].author.id, 10);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_pure_core_not_found_is_not_served_when_the_call_fails() {
        let sources = InMemorySources::default().tweet_features(1, TweetFeatures::default());
        let raw = [raw(1, Some(10))];
        let scenario =
            executor(sources, SafetyLevel::TimelineHome, None).fallback_caches(FallbackCaches {
                tweets: Some(tweet_fallback_cache(8)),
                authors: None,
            });
        let sources = scenario.sources();
        let first = scenario.hydrate(&raw).await;
        assert_eq!(
            first.unresolved,
            HashMap::from([(
                TweetId(1),
                Unresolved {
                    lookup: Lookup::Tweet,
                    cause: Cause::NotFound,
                }
            )])
        );

        sources.break_source(Source::TesPureCoreData, Fault::Fails);
        let second = scenario.hydrate(&raw).await;
        assert_eq!(
            second.unresolved,
            HashMap::from([(TweetId(1), tweet_failed())])
        );
    }

    #[tokio::test]
    async fn gizmoduck_calls_ask_for_every_field_any_level_reads() {
        use QueryFields::{ACCOUNT, EXTENDED_PROFILE, LABELS, SAFETY};
        for (level, viewer_fields) in [
            (
                SafetyLevel::TimelineHome,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY],
            ),
            (
                SafetyLevel::TimelineHomeHydration,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY, LABELS],
            ),
        ] {
            let sources = InMemorySources::default().tweet(1, 10);
            let scenario = executor(sources, level, Some(VIEWER));
            let sources = scenario.sources();
            scenario.hydrate(&[raw(1, None)]).await;
            assert_eq!(
                sources.fields(Source::GizmoduckViewer),
                [viewer_fields],
                "{level:?}"
            );
            assert_eq!(
                sources.fields(Source::GizmoduckAuthor),
                [vec![SAFETY, LABELS]],
                "{level:?}"
            );
        }
    }

    fn call_labels(sources: &InMemorySources) -> Vec<String> {
        let mut labels: Vec<String> = sources
            .fetcher_calls()
            .into_iter()
            .filter(|fetcher| fetcher.source != Source::Flock)
            .map(|fetcher| format!("{fetcher:?}"))
            .chain(sources.selects().into_iter().map(|queries| {
                let graphs: Vec<&str> = queries.iter().map(|q| <&str>::from(q.graph)).collect();
                format!("Flock {}", graphs.join(","))
            }))
            .collect();
        labels.sort();
        labels
    }

    fn every_source_world() -> InMemorySources {
        use ConversationControlArm::{Co, Community, MyNetwork};
        InMemorySources::default()
            .pure_core(
                1,
                PureCoreData {
                    author_id: 10,
                    conversation_id: Some(100),
                    in_reply_to_tweet_id: Some(100),
                    in_reply_to_user_id: Some(30),
                    ..Default::default()
                },
            )
            .tweet(2, 20)
            .tweet_features(1, exclusive_tweet())
            .control(1, control(Community, 30, &[]))
            .control(2, control(Co, 30, &["us"]))
            .tweet(3, 20)
            .tweet_features(
                3,
                TweetFeatures {
                    article_id: article_tweet(70).article_id,
                    trusted_friends_list_id: Some(7),
                    narrowcast_place_id: Some(0xa000_0000_0000_0001),
                    ..Default::default()
                },
            )
            .community_post(3, 500)
            .community_moderation(3, HIDDEN)
            .control(3, control(MyNetwork, 40, &[]))
            .lifecycle(70, ArticleLifecycle::Published)
            .pure_core(
                4,
                PureCoreData {
                    author_id: 20,
                    conversation_id: Some(400),
                    in_reply_to_tweet_id: Some(401),
                    ..Default::default()
                },
            )
            .tweet(400, 41)
            .tweet_features(2, reaction_to(200, 42))
            .authors(&[10, 20, 42])
    }

    #[tokio::test(start_paused = true)]
    async fn a_hung_source_delays_only_the_calls_waiting_on_it() {
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let raw = [
            raw(1, None),
            raw(2, None),
            raw(3, None),
            raw(4, None),
            raw(5, None),
        ];
        let engine = RuleEngine::for_tests();
        let mut rendered = String::new();
        let merged = |sources, level| {
            executor(sources, level, Some(VIEWER)).retweet_merge(RetweetMerge::Merged)
        };
        for level in [TimelineHome, TimelineHomeHydration] {
            let healthy = every_source_world();
            let scenario = merged(healthy, level);
            let healthy = scenario.sources();
            scenario.hydrate(&raw).await;
            let every_call = call_labels(healthy);
            let plan = engine.plan(level);
            assert_eq!(
                every_call.len(),
                plan.groups().count(),
                "{level:?}: {every_call:?}"
            );
            let planned = FetcherId::all()
                .filter(|&source| plan.groups().any(|group| group.fetcher == source));
            for hung in planned {
                let sources = every_source_world().fault(hung, Fault::Hangs);
                let started = tokio::time::Instant::now();
                let scenario = merged(sources, level);
                let sources = scenario.sources();
                let hydration = scenario.hydrate(&raw);
                tokio::pin!(hydration);
                let early =
                    tokio::time::timeout(BEST_EFFORT_HYDRATION_TIMEOUT / 2, &mut hydration).await;
                assert!(early.is_err(), "{level:?} {hung:?}");
                let mut waiting = every_call.clone();
                for call in call_labels(sources) {
                    let planned = waiting.iter().position(|planned| *planned == call);
                    let planned = planned
                        .unwrap_or_else(|| panic!("{level:?} {hung:?} sent unplanned {call}"));
                    waiting.remove(planned);
                }
                hydration.await;
                let ends = started.elapsed();
                let waiting = if waiting.is_empty() {
                    "nothing".to_owned()
                } else {
                    waiting.join("; ")
                };
                writeln!(
                    rendered,
                    "{level:?} {hung:?} ends: {ends:?} waits: {waiting}"
                )
                .unwrap();
            }
        }
        crate::pinned::assert_pinned("tests/fixtures/hung_source_waits.txt", &rendered);
    }

    const HIDDEN: CommunityModeration = CommunityModeration {
        is_hidden: true,
        is_author_removed: false,
    };

    #[tokio::test]
    async fn only_others_community_posts_are_looked_up_and_only_moderated_ones_ask_the_viewer() {
        use crate::hydration::community_source::CommunityPost;
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .community_post(1, 500)
                .community_moderation(1, HIDDEN)
                .tweet(2, 10)
                .community_post(2, 501)
                .tweet(3, 10)
                .tweet(4, VIEWER)
                .community_post(4, 500)
                .community_moderation(4, HIDDEN)
                .authors(&[10, VIEWER])
                .moderator_of(500)
        };
        let post = |tweet_id, author_id, community_id| CommunityPost {
            tweet_id,
            author_id,
            community_id,
        };
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let level = SafetyLevel::TimelineHomeHydration;

        let logged_in = world();
        let scenario = executor(logged_in, level, Some(VIEWER));
        let logged_in = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(
            logged_in.community_posts(),
            [post(1, 10, 500), post(2, 10, 501)]
        );
        assert_eq!(logged_in.keys(Source::CommunityModerator), [vec![500]]);
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (
                    c.post.community_moderation,
                    c.post.viewer_is_community_moderator
                ))
                .collect::<Vec<_>>(),
            [
                (HIDDEN, Some(true)),
                (CommunityModeration::default(), None),
                (CommunityModeration::default(), None),
                (CommunityModeration::default(), None),
            ]
        );

        let logged_out = world();
        let scenario = executor(logged_out, level, None);
        let logged_out = scenario.sources();
        scenario.hydrate(&raw).await;
        assert_eq!(
            logged_out.keys(Source::CommunityModeration),
            [vec![1, 2, 4]]
        );
        assert!(logged_out.keys(Source::CommunityModerator).is_empty());

        let failing = world().fault(Source::CommunityModerator, Fault::Fails);
        let hydrated = executor(failing, level, Some(VIEWER)).hydrate(&raw).await;
        assert_eq!(
            hydrated.candidates[0].post.viewer_is_community_moderator,
            None
        );
    }

    #[tokio::test]
    async fn every_community_post_asks_once_per_community_whether_the_viewer_was_removed() {
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .community_post(1, 500)
                .tweet(2, 10)
                .community_post(2, 501)
                .tweet(3, 10)
                .tweet(4, VIEWER)
                .community_post(4, 500)
                .authors(&[10, VIEWER])
                .removed_from(500)
        };
        let level = SafetyLevel::TimelineHomeHydration;
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let removed = |hydrated: &InRequestOrder| -> Vec<bool> {
            hydrated
                .candidates
                .iter()
                .map(|c| c.post.viewer_is_removed_from_community)
                .collect()
        };

        let logged_in = world();
        let scenario = executor(logged_in, level, Some(VIEWER));
        let logged_in = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(
            logged_in.keys(Source::CommunityViewerRemoved),
            [vec![500, 501]]
        );
        assert_eq!(removed(&hydrated), [true, false, false, true]);
        assert!(hydrated.failed_ids.is_empty());

        let logged_out = world();
        let scenario = executor(logged_out, level, None);
        let logged_out = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert!(logged_out.keys(Source::CommunityViewerRemoved).is_empty());
        assert_eq!(removed(&hydrated), [false; 4]);

        let failing = world().fault(Source::CommunityViewerRemoved, Fault::Fails);
        let hydrated = executor(failing, level, Some(VIEWER)).hydrate(&raw).await;
        assert_eq!(removed(&hydrated), [false; 4]);
        assert_eq!(
            hydrated.failed_nodes,
            [
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
                Ok(Hydrators::empty()),
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
            ]
        );

        let without_limits =
            executor(world(), level, Some(VIEWER)).client(ClientCapability::default());
        let hydrated = without_limits.hydrate(&raw).await;
        let removed_keys = without_limits
            .sources()
            .keys(Source::CommunityViewerRemoved);
        assert!(removed_keys.is_empty());
        assert_eq!(removed(&hydrated), [false; 4]);
    }
}
