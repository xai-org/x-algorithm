use crate::hydration::AuthorNotFound;
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::fallback_cache::{Column, FallbackCache, FallbackCaches};
use crate::hydration::fetcher::Owner;
use crate::hydration::metrics::{
    record_keyed_hydrator_request, record_request_country, record_viewer_country,
};
use crate::hydration::plan::Group;
use crate::hydration::sources::{
    Batches, CallContext, Expired, Fetches, Reply, SourceFetch, Sources,
};
use crate::hydration::store::{CallRequest, Landing, Store};
use crate::hydration::{Hydration, HydrationPlan, HydrationRequest, Hydrator};
use crate::metrics::families::{
    HYDRATOR_EXPANDED_BATCHES, HYDRATOR_EXPANDED_KEYS, HYDRATOR_LATENCY_MS,
    VIEWER_NOT_AGE_VERIFIED, VIEWER_STATE,
};
use crate::metrics::{Metrics, RequestLevel, RequestMetrics};
use crate::models::{Viewer, ViewerAge};
use crate::params::CountryList;
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::FxHashMap;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::time::{Duration, Instant};
use xai_x_rpc::WithBudget;

#[derive(Debug, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(super) enum ViewerCountry {
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
            hydration.viewer.facts.request_country(),
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
    record_request_country(metrics.plain(), hydration.viewer.facts.request_country());
}

pub(super) type InFlight<'p> = Pin<Box<dyn Future<Output = (CallRequest<'p>, Reply)> + Send + 'p>>;

pub(super) struct Timed<'p> {
    metrics: &'p RequestMetrics,
    backend: &'static str,
    call: &'static str,
    call_deadline: Duration,
    candidate_count_by_claimed_key: FxHashMap<u64, usize>,
}

impl Timed<'_> {
    async fn run<B: Batches>(
        &self,
        call: impl Future<Output = Result<B, Expired>>,
        queries: usize,
    ) -> B {
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
            B::timed_out(HydrationBatch::failed(claimed), queries)
        })
    }
}

pub(super) fn send<'p, F: SourceFetch, S: Fetches<F> + ?Sized>(
    call: CallRequest<'p>,
    sources: &'p S,
    context: CallContext<'p>,
    timed: Timed<'p>,
    reply: fn(F::Reply) -> Reply,
) -> Option<InFlight<'p>> {
    if F::NEEDS_VIEWER && call.viewer_id.is_none() {
        return None;
    }
    Some(Box::pin(async move {
        let input = call.input();
        let fetched = async { Ok(F::count_missing(&context, sources.fetch(&input).await?)) };
        let fetched = timed.run(fetched, call.queries().len()).await;
        let fetched = F::count(&context, &input, F::fall_back(&context, fetched));
        (call, reply(fetched))
    }))
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
            request.facts,
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
                    running.extend(self.send(
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
        let hydration = store.assemble(metrics);
        record_viewer(metrics, self, &hydration);
        hydration
    }

    fn send<'p>(
        &self,
        mut call: CallRequest<'p>,
        call_deadline: Duration,
        sources: &'p dyn Sources,
        fallback_caches: &'p FallbackCaches,
        metrics: &'p RequestMetrics,
    ) -> Option<InFlight<'p>> {
        let plain = metrics.plain();
        let group = call.group;
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
        let is_request = group.fetcher.owner == Owner::Request;
        let context = CallContext {
            metrics,
            label: (backend, call_name),
            tweet_cache: fallback_caches.tweets.as_ref(),
            author_cache: fallback_caches.authors.as_ref().filter(|_| is_request),
            author_not_found: if is_request {
                self.level().spec().author_not_found()
            } else {
                AuthorNotFound::Unresolves
            },
        };
        group.fetcher.source.send(call, sources, context, timed)
    }
}

pub(super) fn fall_back<C: Column>(
    metrics: &Metrics,
    cache: Option<&FallbackCache<C::Entry>>,
    batch: RawHydrationBatch<C::Value>,
) -> RawHydrationBatch<C::Value> {
    match cache {
        Some(cache) => cache.resolve_hydration_batch::<C>(metrics, batch),
        None => batch,
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeDirection, EdgeFrom, EdgeQuery, Graph};
    use crate::hydration::fetcher::FetcherId;
    use crate::hydration::sources::{Fault, InMemorySources, Source, control, suspended};
    use crate::hydration::{
        BEST_EFFORT_HYDRATION_TIMEOUT, Cause, HYDRATION_TIMEOUT, Hydrator, Hydrators, Lookup,
        Relation, Unresolved,
    };
    use crate::models::{
        ArticleLifecycle, ClientCapability, ClientSwitch, CommunityModeration, Evaluation,
        Placement, RetweetMerge, SafetyLabelType, TweetFeatures, TweetId, Viewer, ViewerProfile,
    };
    use crate::rules::{RuleEngine, SafetyLevel};
    use crate::scenario::{InRequestOrder, Scenario, label_map, protected_user, raw, reaction_to};
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::num::NonZeroU64;
    use xai_core_entities::entities::{ConversationControlArm, PureCoreData};

    pub(in crate::hydration) const VIEWER: u64 = 50;

    pub(in crate::hydration) fn executor(
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

    pub(in crate::hydration) fn tweet_failed() -> Unresolved {
        Unresolved {
            lookup: Lookup::Tweet,
            cause: Cause::Failed,
        }
    }

    pub(in crate::hydration) fn ids(ids: &[u64]) -> HashSet<TweetId> {
        ids.iter().copied().map(TweetId).collect()
    }

    pub(in crate::hydration) fn exclusive_tweet() -> TweetFeatures {
        TweetFeatures {
            exclusive_conversation_author_id: Some(30),
            ..Default::default()
        }
    }

    pub(in crate::hydration) fn article_tweet(article_id: u64) -> TweetFeatures {
        TweetFeatures {
            article_id: NonZeroU64::new(article_id),
            ..Default::default()
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_best_effort_call_ends_at_its_ceiling_or_the_callers_budget_whichever_comes_first() {
        let ceiling = BEST_EFFORT_HYDRATION_TIMEOUT;
        for (remaining, ends_at) in [(ceiling * 2, ceiling), (ceiling / 2, ceiling / 2)] {
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

    pub(in crate::hydration) fn sorted(mut calls: Vec<Source>) -> Vec<String> {
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
        assert_eq!(hydrated.viewer_features.facts.request_country(), Some("us"));
        assert_eq!(hydrated.candidates[0].author.id, 10);
        assert!(hydrated.safety_labels.is_empty());
        assert!(hydrated.failed_ids.is_empty());
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

    pub(in crate::hydration) fn root_edges(sources: &InMemorySources) -> Vec<Vec<EdgeQuery>> {
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

    pub(in crate::hydration) fn call_labels(sources: &InMemorySources) -> Vec<String> {
        let mut labels: Vec<String> = sources
            .fetcher_calls()
            .into_iter()
            .filter(|fetcher| fetcher.source != Source::Flock)
            .map(|fetcher| format!("{fetcher:?}"))
            .chain(
                sources
                    .selects()
                    .into_iter()
                    .map(|queries| flock_label(queries.iter().map(|q| q.graph))),
            )
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

    fn flock_label(graphs: impl Iterator<Item = Graph>) -> String {
        let graphs: Vec<&str> = graphs.map(<&str>::from).collect();
        format!("Flock {}", graphs.join(","))
    }

    fn call_label(group: &Group) -> String {
        if group.fetcher.source == Source::Flock {
            flock_label(group.edges().iter().map(|&(_, graph, ..)| graph))
        } else {
            format!("{:?}", group.fetcher)
        }
    }

    fn supplied_by<'p>(plan: &'p HydrationPlan, landed: &'p Group) -> Vec<&'p Group> {
        plan.groups()
            .filter(|group| {
                group
                    .input
                    .is_some_and(|input| landed.nodes.contains(input))
            })
            .collect()
    }

    fn closure_under<'p>(
        start: &[&'p Group],
        step: impl Fn(&'p Group) -> Vec<&'p Group>,
    ) -> BTreeSet<String> {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut reached = BTreeSet::new();
        let mut landed = start.to_vec();
        while let Some(group) = landed.pop() {
            for next in step(group) {
                if seen.insert(next.position) {
                    reached.insert(call_label(next));
                    landed.push(next);
                }
            }
        }
        reached
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
            let labels: BTreeSet<String> = plan.groups().map(call_label).collect();
            assert_eq!(labels.len(), plan.groups().count(), "{level:?}: {labels:?}");
            assert!(every_call.iter().eq(&labels), "{level:?}: {every_call:?}");
            let planned = FetcherId::all()
                .filter(|&source| plan.groups().any(|group| group.fetcher == source));
            for hung in planned {
                let hung_calls: Vec<&Group> = plan
                    .groups()
                    .filter(|group| group.fetcher == hung)
                    .collect();
                let ceiling = hung_calls
                    .iter()
                    .map(|group| {
                        plan.call_deadline(group, group.edges().iter().map(|&(edge, ..)| edge))
                    })
                    .max()
                    .unwrap();
                let after_it = closure_under(&hung_calls, |landed| supplied_by(plan, landed));
                let reading_it =
                    closure_under(&hung_calls, |landed| plan.readers(landed).collect());
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
                assert_eq!(started.elapsed(), ceiling, "{level:?} {hung:?} ends");
                let waiting: BTreeSet<String> = waiting.into_iter().collect();
                assert!(
                    after_it.is_subset(&waiting),
                    "{level:?} {hung:?}: a call after the hung source sent; waiting {waiting:?}"
                );
                assert!(
                    reading_it.is_superset(&waiting),
                    "{level:?} {hung:?}: a call that reads nothing from the hung source waited; waiting {waiting:?}"
                );
                for input in plan
                    .groups()
                    .filter(|group| waiting.contains(&call_label(group)))
                {
                    for next in supplied_by(plan, input) {
                        assert!(
                            waiting.contains(&call_label(next)),
                            "{level:?} {hung:?}: {} sent before {} landed its input",
                            call_label(next),
                            call_label(input)
                        );
                    }
                }
            }
        }
    }

    pub(in crate::hydration) const HIDDEN: CommunityModeration = CommunityModeration {
        is_hidden: true,
        is_author_removed: false,
    };
}
