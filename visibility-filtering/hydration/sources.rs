use crate::clients::about_this_account_client::AboutThisAccountClient;
use crate::clients::article_client::ArticleClient;
use crate::clients::gizmoduck_client::GizmoduckLookup;
use crate::clients::socialgraph_client::{EdgeQuery, SocialgraphClient};
use crate::clients::trusted_friends_client::TrustedFriendsClient;
use crate::clients::user_devices_client::UserDevicesClient;
use crate::clients::user_location_client::UserLocationClient;
use crate::clients::wingman_client::WingmanClient;
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::community_source::{CommunityPost, CommunitySource};
use crate::hydration::decode::author::{AuthorFallbackCache, AuthorNotFound};
use crate::hydration::decode::tweet::TweetFallbackCache;
use crate::hydration::execute::{send, InFlight, Timed};
use crate::hydration::fetcher::{FetcherId, Fetchers};
use crate::hydration::hydrator::Edge;
use crate::hydration::store::{CallRequest, FetcherKey, FetcherMut, FetcherRef, Store};
use crate::hydration::tweet_source::TweetSource;
use crate::metrics::RequestMetrics;
use crate::post_card::{PostCard, PostCards};
use futures::future::{join, BoxFuture};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;
use strum::VariantArray;
use tonic::metadata::{MetadataMap, MetadataValue};
use xai_core_entities::gizmoduck_client::QueryFields;
use xai_core_entities::tweet_entity_service_client::TESClient;
use xai_safety_label_source::SafetyLabelSource;

mod about_this_account;
mod articles;
mod communities;
mod geoduck;
mod gizmoduck;
mod notifinfra;
mod safety_labels;
mod socialgraph;
mod tes;
mod trusted_friends;
mod wingman;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Expired;

pub(crate) trait SourceFetch: 'static {
    const BACKEND_LABEL: &'static str;
    const DECIDES_EXISTENCE: bool;
    const NEEDS_VIEWER: bool;
    const EDGE: Option<Edge>;

    type Key: Copy + Eq + Hash;
    type Reply: Batches;

    fn fetch<'c>(
        backends: &'c Backends,
        input: &'c Input<'c>,
    ) -> impl Future<Output = Result<Self::Reply, Expired>> + Send + 'c;

    fn count_missing(_: &CallContext<'_>, reply: Self::Reply) -> Self::Reply {
        reply
    }

    fn fall_back(_: &CallContext<'_>, reply: Self::Reply) -> Self::Reply {
        reply
    }

    fn count(_: &CallContext<'_>, _: &Input<'_>, reply: Self::Reply) -> Self::Reply {
        reply
    }

    #[cfg(test)]
    type World: Default + Send + Sync;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply;
}

pub(crate) trait Batches: AsRef<[RawHydrationBatch<Self::Value>]> + Send {
    type Value;

    fn timed_out(failed: RawHydrationBatch<Self::Value>, queries: usize) -> Self;
}

impl<V: Send> Batches for RawHydrationBatch<V> {
    type Value = V;

    fn timed_out(failed: Self, _: usize) -> Self {
        failed
    }
}

impl<V: Clone + Send> Batches for Vec<RawHydrationBatch<V>> {
    type Value = V;

    fn timed_out(failed: RawHydrationBatch<V>, queries: usize) -> Self {
        vec![failed; queries]
    }
}

pub(crate) struct Input<'c> {
    pub(crate) keys: &'c [u64],
    pub(crate) viewer_id: Option<u64>,
    pub(crate) fields: &'c [QueryFields],
    pub(crate) queries: &'c [EdgeQuery],
    pub(crate) community_posts: &'c [CommunityPost],
    #[cfg(test)]
    pub(crate) fetcher: FetcherId,
}

pub(crate) struct CallContext<'p> {
    pub(crate) metrics: &'p RequestMetrics,
    pub(crate) label: (&'static str, &'static str),
    pub(crate) tweet_cache: Option<&'p TweetFallbackCache>,
    pub(crate) author_cache: Option<&'p AuthorFallbackCache>,
    pub(crate) author_not_found: AuthorNotFound,
}

pub(crate) trait Fetches<F: SourceFetch>: Send + Sync {
    fn fetch<'c>(&'c self, input: &'c Input<'c>) -> BoxFuture<'c, Result<F::Reply, Expired>>;
}

pub(super) type Value<F> = <<F as SourceFetch>::Reply as Batches>::Value;

pub(super) trait Registered: SourceFetch {
    const SOURCE: Source;

    fn fetchers(all: &SourceFetchers) -> &Fetchers<Value<Self>, Self::Key>;
    fn fetchers_mut(all: &mut SourceFetchers) -> &mut Fetchers<Value<Self>, Self::Key>;
}

#[cfg(test)]
pub(crate) trait Held: SourceFetch {
    fn world(worlds: &Worlds) -> &Self::World;
    fn world_mut(worlds: &mut Worlds) -> &mut Self::World;
}

macro_rules! sources {
    ($($source:ident,)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, VariantArray)]
        pub(crate) enum Source {
            $($source,)*
        }

        pub(crate) mod source {
            $(pub(crate) struct $source;)*
        }

        impl Source {
            pub(super) const fn decides_existence(self) -> bool {
                match self {
                    $(Source::$source => <source::$source as SourceFetch>::DECIDES_EXISTENCE,)*
                }
            }

            #[cfg(test)]
            pub(super) const fn edge(self) -> Option<Edge> {
                match self {
                    $(Source::$source => <source::$source as SourceFetch>::EDGE,)*
                }
            }

            pub(super) const fn backend_label(self) -> &'static str {
                match self {
                    $(Source::$source => <source::$source as SourceFetch>::BACKEND_LABEL,)*
                }
            }

            pub(super) fn send<'p>(
                self,
                call: CallRequest<'p>,
                sources: &'p dyn Sources,
                context: CallContext<'p>,
                timed: Timed<'p>,
            ) -> Option<InFlight<'p>> {
                match self {
                    $(Source::$source => send::<source::$source, dyn Sources>(
                        call, sources, context, timed, Reply::$source,
                    ),)*
                }
            }
        }

        pub(crate) trait Sources: Send + Sync $(+ Fetches<source::$source>)* {}

        impl<T: Send + Sync $(+ Fetches<source::$source>)*> Sources for T {}

        #[derive(Default)]
        #[expect(non_snake_case, reason = "each field is named for its Source variant")]
        pub(super) struct SourceFetchers {
            $($source: Fetchers<Value<source::$source>, <source::$source as SourceFetch>::Key>,)*
        }

        impl SourceFetchers {
            pub(super) fn get(&self, id: FetcherId) -> FetcherRef<'_> {
                match id.source {
                    $(Source::$source => FetcherKey::view(self.$source.of(id.owner)),)*
                }
            }

            pub(super) fn get_mut(&mut self, id: FetcherId) -> FetcherMut<'_> {
                match id.source {
                    $(Source::$source => FetcherKey::view_mut(self.$source.of_mut(id.owner)),)*
                }
            }
        }

        pub(super) enum Reply {
            $($source(<source::$source as SourceFetch>::Reply),)*
        }

        impl Reply {
            pub(super) fn land(self, store: &mut Store, call: &CallRequest<'_>, elapsed: Duration) {
                match self {
                    $(Reply::$source(reply) => {
                        store.land_reply::<source::$source>(call, reply, elapsed);
                    })*
                }
            }
        }

        $(
            impl Registered for source::$source {
                const SOURCE: Source = Source::$source;

                fn fetchers(all: &SourceFetchers) -> &Fetchers<Value<Self>, Self::Key> {
                    &all.$source
                }

                fn fetchers_mut(all: &mut SourceFetchers) -> &mut Fetchers<Value<Self>, Self::Key> {
                    &mut all.$source
                }
            }
        )*

        #[cfg(test)]
        #[derive(Default)]
        #[expect(non_snake_case, reason = "each field is named for its Source variant")]
        pub(crate) struct Worlds {
            $($source: <source::$source as SourceFetch>::World,)*
        }

        $(
            #[cfg(test)]
            impl Held for source::$source {
                fn world(worlds: &Worlds) -> &Self::World {
                    &worlds.$source
                }

                fn world_mut(worlds: &mut Worlds) -> &mut Self::World {
                    &mut worlds.$source
                }
            }
        )*
    };
}

sources! {
    TesPureCoreData,
    TesTweetForVisibility,
    TesConversationControl,
    TesQuotedTweet,
    TesTweetState,
    SafetyLabels,
    GizmoduckViewer,
    GizmoduckAuthor,
    Flock,
    ViewerCountry,
    Wingman,
    CommunityModeration,
    CommunityModerator,
    CommunityViewerRemoved,
    ArticleLifecycle,
    ArticleMediaEntities,
    ArticleCoverMedia,
    TrustedFriends,
    UserLocation,
    GizmoduckActor,
    ActorEmails,
    ActorPhones,
}

pub(crate) struct Backends {
    pub(crate) tes: Arc<dyn TESClient + Send + Sync>,
    pub(crate) tweets: TweetSource,
    pub(crate) gizmoduck: GizmoduckLookup,
    pub(crate) socialgraph: Arc<dyn SocialgraphClient + Send + Sync>,
    pub(crate) about_this_account: AboutThisAccountClient,
    pub(crate) wingman: Arc<dyn WingmanClient>,
    pub(crate) articles: ArticleClient,
    pub(crate) trusted_friends: TrustedFriendsClient,
    pub(crate) user_location: UserLocationClient,
    pub(crate) safety_labels: Arc<SafetyLabelSource>,
    pub(crate) communities: CommunitySource,
    pub(crate) user_devices: UserDevicesClient,
}

pub(crate) struct ProdSources {
    backends: Backends,
}

impl ProdSources {
    pub(crate) fn new(backends: Backends) -> Self {
        Self { backends }
    }
}

impl<F: SourceFetch> Fetches<F> for ProdSources {
    fn fetch<'c>(&'c self, input: &'c Input<'c>) -> BoxFuture<'c, Result<F::Reply, Expired>> {
        Box::pin(F::fetch(&self.backends, input))
    }
}

const TEST_USERS_DTAB: &str = "/s/gizmoduck/test-users-permanent=>/s/gizmoduck/gizmoduck";

pub fn test_users_metadata() -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert("dtab-local", MetadataValue::from_static(TEST_USERS_DTAB));
    metadata
}

#[tonic::async_trait]
impl PostCards for ProdSources {
    async fn post_card(&self, tweet_id: u64) -> Option<PostCard> {
        let (core, media) = join(
            self.backends.tes.get_tweet_core_datas(vec![tweet_id]),
            self.backends.tes.get_tweet_media_entities(vec![tweet_id]),
        )
        .await;
        let core = only(core)?;
        let author = only(
            self.backends
                .gizmoduck
                .get_users(vec![core.author_id], &[QueryFields::PROFILE])
                .await,
        )
        .and_then(|result| result.user);
        Some(PostCard::new(
            tweet_id,
            core,
            author.as_ref().map(|user| &user.profile),
            &only(media).unwrap_or_default(),
        ))
    }
}

fn held_by(keys: &[u64], holders: Option<&HashSet<u64>>) -> RawHydrationBatch<bool> {
    let answers: HashMap<u64, Result<Option<bool>, ()>> = keys
        .iter()
        .map(|&key| {
            let held = holders
                .map(|holders| Some(holders.contains(&key)))
                .ok_or(());
            (key, held)
        })
        .collect();
    HydrationBatch::from_results(keys.iter().copied(), answers)
}

fn one_viewer_batch<V, E>(viewer_id: u64, answer: Result<Option<V>, E>) -> RawHydrationBatch<V> {
    HydrationBatch::from_results([viewer_id], HashMap::from([(viewer_id, answer)]))
}

fn only<V, E>(answers: HashMap<u64, Result<Option<V>, E>>) -> Option<V> {
    answers.into_values().next()?.ok().flatten()
}

#[cfg(test)]
pub(crate) use in_memory::{control, found_user, suspended, Fault, InMemoryCall, InMemorySources};

#[cfg(test)]
mod fake_backends;

#[cfg(test)]
mod in_memory;
