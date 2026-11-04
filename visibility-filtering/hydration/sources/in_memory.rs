use super::socialgraph::FlockWorld;
use super::*;
use crate::clients::socialgraph_client::Graph;
use crate::hydration::hydrator::UserPair;
use crate::hydration::tweet_source::TweetState;
use crate::models::{ArticleLifecycle, CommunityModeration, TweetFeatures};
use rustc_hash::FxHashMap;
use std::num::NonZeroU64;
use std::sync::Mutex;
use tokio::time::{sleep, Instant};
use xai_core_entities::entities::{
    ConversationControl, ConversationControlArm, GizmoduckUser, GizmoduckUserResult, PureCoreData,
    QuotedTweet, Safety, UserResponseState,
};
use xai_visibility_filtering_proto as vf_pb;

pub(crate) fn found_user(user_id: u64) -> GizmoduckUserResult {
    GizmoduckUserResult {
        user: Some(GizmoduckUser {
            user_id,
            ..Default::default()
        }),
        response_state: Some(UserResponseState::Found),
    }
}

pub(crate) fn suspended() -> GizmoduckUserResult {
    GizmoduckUserResult {
        user: Some(GizmoduckUser {
            safety: Safety {
                suspended: true,
                ..Default::default()
            },
            ..Default::default()
        }),
        response_state: Some(UserResponseState::Found),
    }
}

pub(crate) fn control(
    arm: ConversationControlArm,
    root: u64,
    countries: &[&str],
) -> ConversationControl {
    ConversationControl {
        arm,
        conversation_tweet_author_id: root,
        invited_user_ids: vec![],
        invite_via_mention: None,
        allowed_country_codes: countries.iter().map(|c| (*c).to_owned()).collect(),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Fault {
    Fails,
    Hangs,
    Delays(std::time::Duration),
    AnswersPartial,
}

#[derive(Default)]
pub(crate) struct InMemorySources {
    worlds: Worlds,
    faults: Mutex<Vec<(FetcherId, Fault)>>,
    failed_keys: HashSet<(FetcherId, u64)>,
    latencies: HashMap<FetcherId, Duration>,
    key_latencies: HashMap<(FetcherId, u64), Duration>,
    calls: Mutex<Vec<(FetcherId, Vec<u64>)>>,
    viewers: Mutex<Vec<(Source, Option<u64>)>>,
    starts: Mutex<Vec<(FetcherId, Instant)>>,
    fields: Mutex<Vec<(FetcherId, Vec<QueryFields>)>>,
    selects: Mutex<Vec<Vec<EdgeQuery>>>,
    hung_graphs: HashSet<Graph>,
    community_posts: Mutex<Vec<CommunityPost>>,
}

pub(crate) struct InMemoryCall<'s> {
    fetcher: FetcherId,
    fails: bool,
    fault: Option<Fault>,
    failed_keys: &'s HashSet<(FetcherId, u64)>,
}

impl InMemoryCall<'_> {
    pub(crate) fn fails(&self) -> bool {
        self.fails
    }

    pub(crate) fn answers_partial(&self) -> bool {
        matches!(self.fault, Some(Fault::AnswersPartial))
    }

    pub(crate) fn keyed_with<V>(
        &self,
        keys: &[u64],
        held: impl Fn(u64) -> Option<V>,
    ) -> RawHydrationBatch<V> {
        let results: FxHashMap<_, _> = keys
            .iter()
            .map(|&key| {
                let result = if self.fails || self.failed_keys.contains(&(self.fetcher, key)) {
                    Err(())
                } else {
                    Ok(held(key))
                };
                (key, result)
            })
            .collect();
        HydrationBatch::from_results(keys.iter().copied(), results)
    }

    pub(crate) fn keyed<V: Clone>(
        &self,
        keys: &[u64],
        world: &HashMap<u64, V>,
    ) -> RawHydrationBatch<V> {
        self.keyed_with(keys, |key| world.get(&key).cloned())
    }

    pub(crate) fn held_in(&self, keys: &[u64], world: &HashSet<u64>) -> RawHydrationBatch<bool> {
        self.keyed_with(keys, |key| Some(world.contains(&key)))
    }

    pub(crate) fn viewer_edges(
        &self,
        input: &Input<'_>,
        world: &HashSet<UserPair>,
    ) -> RawHydrationBatch<bool> {
        self.keyed_with(input.keys, |destination| {
            Some(input.viewer_id.is_some_and(|source| {
                world.contains(&UserPair {
                    source,
                    destination,
                })
            }))
        })
    }
}

fn called_keys(input: &Input<'_>) -> Vec<u64> {
    if input.fetcher.source == Source::Flock {
        let ends = input.queries.iter().flat_map(|query| {
            query
                .from
                .id(None)
                .into_iter()
                .chain(query.destination_ids.iter().copied())
        });
        input.viewer_id.into_iter().chain(ends).collect()
    } else if input.fetcher.source == Source::CommunityModeration {
        input
            .community_posts
            .iter()
            .map(|post| post.tweet_id)
            .collect()
    } else {
        input.keys.to_vec()
    }
}

impl<F: Held> Fetches<F> for InMemorySources {
    fn fetch<'c>(&'c self, input: &'c Input<'c>) -> BoxFuture<'c, Result<F::Reply, Expired>> {
        Box::pin(async move {
            let fetcher = input.fetcher;
            self.fields
                .lock()
                .unwrap()
                .push((fetcher, input.fields.to_vec()));
            self.viewers
                .lock()
                .unwrap()
                .push((fetcher.source, input.viewer_id));
            if !input.queries.is_empty() {
                let mut recorded = input.queries.to_vec();
                for query in &mut recorded {
                    query.destination_ids.sort_unstable();
                }
                self.selects.lock().unwrap().push(recorded);
            }
            self.community_posts
                .lock()
                .unwrap()
                .extend_from_slice(input.community_posts);
            let fails = self.enter(fetcher, &called_keys(input)).await;
            let call = InMemoryCall {
                fetcher,
                fails,
                fault: self.fault_for(fetcher),
                failed_keys: &self.failed_keys,
            };
            let reply = F::in_memory(F::world(&self.worlds), input, &call);
            if input
                .queries
                .iter()
                .any(|query| self.hung_graphs.contains(&query.graph))
            {
                std::future::pending::<()>().await;
            }
            Ok(reply)
        })
    }
}

impl InMemorySources {
    pub(crate) fn hold<L: Held>(mut self, edit: impl FnOnce(&mut L::World)) -> Self {
        edit(L::world_mut(&mut self.worlds));
        self
    }

    fn world<L: Held>(&self) -> &L::World {
        L::world(&self.worlds)
    }

    pub(crate) fn tweet(self, tweet_id: u64, author_id: u64) -> Self {
        self.pure_core(
            tweet_id,
            PureCoreData {
                author_id,
                ..Default::default()
            },
        )
    }

    pub(crate) fn pure_core(self, tweet_id: u64, core: PureCoreData) -> Self {
        self.hold::<source::TesPureCoreData>(|cores| {
            cores.insert(tweet_id, core);
        })
        .hold::<source::TesTweetForVisibility>(|tweets| {
            tweets.entry(tweet_id).or_default();
        })
    }

    pub(crate) fn retweet(
        self,
        tweet_id: u64,
        author_id: u64,
        source_tweet_id: u64,
        source_user_id: Option<u64>,
    ) -> Self {
        let core = PureCoreData {
            author_id,
            source_tweet_id: Some(source_tweet_id),
            source_user_id,
            ..Default::default()
        };
        self.pure_core(tweet_id, core)
    }

    pub(crate) fn without_tweet_row(self, tweet_id: u64) -> Self {
        self.hold::<source::TesTweetForVisibility>(|tweets| {
            tweets.remove(&tweet_id);
        })
    }

    pub(crate) fn tweet_features(self, tweet_id: u64, tweet: TweetFeatures) -> Self {
        self.hold::<source::TesTweetForVisibility>(|tweets| {
            tweets.insert(tweet_id, tweet);
        })
    }

    pub(crate) fn community_post(self, tweet_id: u64, community_id: u64) -> Self {
        self.hold::<source::TesTweetForVisibility>(|tweets| {
            tweets.entry(tweet_id).or_default().community_id = NonZeroU64::new(community_id);
        })
    }

    pub(crate) fn control(self, tweet_id: u64, control: ConversationControl) -> Self {
        self.hold::<source::TesConversationControl>(|controls| {
            controls.insert(tweet_id, control);
        })
    }

    pub(crate) fn quote(self, outer: u64, tweet_id: u64, user_id: u64) -> Self {
        self.hold::<source::TesQuotedTweet>(|quoted| {
            quoted.insert(outer, QuotedTweet { tweet_id, user_id });
        })
    }

    pub(crate) fn tweet_state(self, tweet_id: u64, state: TweetState) -> Self {
        self.hold::<source::TesTweetState>(|states| {
            states.insert(tweet_id, state);
        })
    }

    pub(crate) fn labels(self, tweet_id: u64, labels: vf_pb::SafetyLabelMap) -> Self {
        self.hold::<source::SafetyLabels>(|held| {
            held.insert(tweet_id, Arc::new(labels));
        })
    }

    pub(crate) fn viewer(self, viewer_id: u64, user: GizmoduckUser) -> Self {
        self.hold::<source::GizmoduckViewer>(|viewers| {
            viewers.insert(viewer_id, user);
        })
    }

    pub(crate) fn user(self, user_id: u64, user: GizmoduckUserResult) -> Self {
        let actor = user.clone();
        self.hold::<source::GizmoduckAuthor>(|users| {
            users.insert(user_id, user);
        })
        .hold::<source::GizmoduckActor>(|users| {
            users.insert(user_id, actor);
        })
    }

    pub(crate) fn authors(self, user_ids: &[u64]) -> Self {
        user_ids.iter().fold(self, |sources, &user_id| {
            sources.user(user_id, found_user(user_id))
        })
    }

    pub(crate) fn edge(self, graph: Graph, source: u64, destination: u64) -> Self {
        self.flock(|flock| {
            flock.edges.insert((graph, source, destination));
        })
    }

    pub(crate) fn country(self, viewer_id: u64, country: &str) -> Self {
        self.hold::<source::ViewerCountry>(|countries| {
            countries.insert(viewer_id, Arc::from(country));
        })
    }

    pub(crate) fn second_degree_path(self, root_author: u64, viewer_id: u64) -> Self {
        self.hold::<source::Wingman>(|paths| {
            paths.insert(UserPair {
                source: viewer_id,
                destination: root_author,
            });
        })
    }

    pub(crate) fn community_moderation(
        self,
        tweet_id: u64,
        moderation: CommunityModeration,
    ) -> Self {
        self.hold::<source::CommunityModeration>(|world| {
            world.insert(tweet_id, moderation);
        })
    }

    pub(crate) fn moderator_of(self, community_id: u64) -> Self {
        self.hold::<source::CommunityModerator>(|communities| {
            communities.insert(community_id);
        })
    }

    pub(crate) fn removed_from(self, community_id: u64) -> Self {
        self.hold::<source::CommunityViewerRemoved>(|communities| {
            communities.insert(community_id);
        })
    }

    pub(crate) fn community_posts(&self) -> Vec<CommunityPost> {
        self.community_posts.lock().unwrap().clone()
    }

    pub(crate) fn article_media_entities(self, article_id: u64, present: bool) -> Self {
        self.hold::<source::ArticleMediaEntities>(|articles| {
            articles.insert(article_id, present);
        })
    }

    pub(crate) fn article_cover_media(self, article_id: u64, present: bool) -> Self {
        self.hold::<source::ArticleCoverMedia>(|articles| {
            articles.insert(article_id, present);
        })
    }

    pub(crate) fn lifecycle(self, article_id: u64, lifecycle: ArticleLifecycle) -> Self {
        self.hold::<source::ArticleLifecycle>(|lifecycles| {
            lifecycles.insert(article_id, lifecycle);
        })
    }

    pub(crate) fn trusted_friend(self, list_id: u64, viewer_id: u64) -> Self {
        self.hold::<source::TrustedFriends>(|memberships| {
            memberships.insert(UserPair {
                source: viewer_id,
                destination: list_id,
            });
        })
    }

    pub(crate) fn located_in(self, place_id: u64) -> Self {
        self.hold::<source::UserLocation>(|places| {
            places.insert(place_id);
        })
    }

    pub(crate) fn confirmed_email(self, user_id: u64) -> Self {
        self.hold::<source::ActorEmails>(|users| {
            users.insert(user_id);
        })
    }

    pub(crate) fn verified_phone(self, user_id: u64) -> Self {
        self.hold::<source::ActorPhones>(|users| {
            users.insert(user_id);
        })
    }

    fn flock(self, edit: impl FnOnce(&mut FlockWorld)) -> Self {
        self.hold::<source::Flock>(edit)
    }

    pub(crate) fn fail_graph(self, graph: Graph) -> Self {
        self.flock(|flock| {
            flock.failed_graphs.insert(graph);
        })
    }

    pub(crate) fn fail_edge(self, graph: Graph, destination: u64) -> Self {
        self.flock(|flock| {
            flock.failed_edges.insert((graph, destination));
        })
    }

    pub(crate) fn hang_graph(mut self, graph: Graph) -> Self {
        self.hung_graphs.insert(graph);
        self
    }

    pub(crate) fn miss_graph(self, graph: Graph) -> Self {
        self.flock(|flock| {
            flock.missing_graphs.insert(graph);
        })
    }

    pub(crate) fn selects(&self) -> Vec<Vec<EdgeQuery>> {
        self.selects.lock().unwrap().clone()
    }

    pub(crate) fn fault(self, fetcher: impl Into<FetcherId>, fault: Fault) -> Self {
        self.break_source(fetcher, fault);
        self
    }

    pub(crate) fn break_source(&self, fetcher: impl Into<FetcherId>, fault: Fault) {
        let fetcher = fetcher.into();
        self.faults.lock().unwrap().push((fetcher, fault));
    }

    pub(crate) fn fail_key(mut self, fetcher: impl Into<FetcherId>, key: u64) -> Self {
        let fetcher = fetcher.into();
        self.failed_keys.insert((fetcher, key));
        self
    }

    pub(crate) fn latency(mut self, fetcher: impl Into<FetcherId>, latency: Duration) -> Self {
        let fetcher = fetcher.into();
        self.latencies.insert(fetcher, latency);
        self
    }

    pub(crate) fn key_latency(
        mut self,
        fetcher: impl Into<FetcherId>,
        key: u64,
        latency: Duration,
    ) -> Self {
        let fetcher = fetcher.into();
        self.key_latencies.insert((fetcher, key), latency);
        self
    }

    pub(crate) fn calls(&self) -> Vec<Source> {
        self.fetcher_calls()
            .into_iter()
            .map(|fetcher| fetcher.source)
            .collect()
    }

    pub(crate) fn fetcher_calls(&self) -> Vec<FetcherId> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(fetcher, _)| *fetcher)
            .collect()
    }

    pub(crate) fn starts(&self, fetcher: impl Into<FetcherId>) -> Vec<Instant> {
        let fetcher = fetcher.into();
        self.starts
            .lock()
            .unwrap()
            .iter()
            .filter(|(called, _)| *called == fetcher)
            .map(|(_, started)| *started)
            .collect()
    }

    pub(crate) fn viewers(&self, source: Source) -> Vec<Option<u64>> {
        self.viewers
            .lock()
            .unwrap()
            .iter()
            .filter(|(called, _)| *called == source)
            .map(|&(_, viewer_id)| viewer_id)
            .collect()
    }

    pub(crate) fn keys(&self, fetcher: impl Into<FetcherId>) -> Vec<Vec<u64>> {
        let fetcher = fetcher.into();
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(called, _)| *called == fetcher)
            .map(|(_, keys)| keys.clone())
            .collect()
    }

    pub(crate) fn fields(&self, fetcher: impl Into<FetcherId>) -> Vec<Vec<QueryFields>> {
        let fetcher = fetcher.into();
        self.fields
            .lock()
            .unwrap()
            .iter()
            .filter(|(called, _)| *called == fetcher)
            .map(|(_, fields)| fields.clone())
            .collect()
    }

    async fn enter(&self, fetcher: FetcherId, keys: &[u64]) -> bool {
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        self.calls.lock().unwrap().push((fetcher, sorted));
        self.starts.lock().unwrap().push((fetcher, Instant::now()));
        let latency = keys
            .iter()
            .filter_map(|&key| self.key_latencies.get(&(fetcher, key)))
            .chain(self.latencies.get(&fetcher))
            .max();
        if let Some(latency) = latency {
            sleep(*latency).await;
        }
        match self.fault_for(fetcher) {
            Some(Fault::Hangs) => std::future::pending().await,
            Some(Fault::Delays(delay)) => {
                tokio::time::sleep(delay).await;
                false
            }
            Some(Fault::Fails) => true,
            Some(Fault::AnswersPartial) | None => false,
        }
    }

    fn fault_for(&self, fetcher: impl Into<FetcherId>) -> Option<Fault> {
        let fetcher = fetcher.into();
        self.faults
            .lock()
            .unwrap()
            .iter()
            .find(|(faulty, _)| *faulty == fetcher)
            .map(|(_, fault)| *fault)
    }
}

#[tonic::async_trait]
impl PostCards for InMemorySources {
    async fn post_card(&self, tweet_id: u64) -> Option<PostCard> {
        if matches!(self.fault_for(Source::TesPureCoreData), Some(Fault::Fails)) {
            return None;
        }
        let core = self
            .world::<source::TesPureCoreData>()
            .get(&tweet_id)?
            .clone();
        let author = self
            .world::<source::GizmoduckAuthor>()
            .get(&core.author_id)
            .and_then(|result| result.user.as_ref());
        Some(PostCard::new(
            tweet_id,
            core,
            author.map(|user| &user.profile),
            &[],
        ))
    }
}
