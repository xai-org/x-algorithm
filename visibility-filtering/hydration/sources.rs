use crate::clients::socialgraph_client::{EdgeQuery, SocialgraphClient};
use crate::clients::wingman_client::WingmanClient;
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::community_source::{CommunityPost, CommunitySource};
use crate::hydration::decode::author::{AuthorFallbackCache, AuthorNotFound};
use crate::hydration::decode::tweet::TweetFallbackCache;
use crate::hydration::execute::{send, InFlight, Timed};
use crate::hydration::fetcher::{Fetcher, FetcherId, Fetchers};
use crate::hydration::hydrator::{Edge, EdgeKey};
use crate::hydration::inventory::Answer;
use crate::hydration::store::{CallRequest, FetcherKey, Store};
use crate::hydration::tweet_source::TweetSource;
use crate::metrics::RequestMetrics;
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;
use strum::VariantArray;
use tonic::metadata::{MetadataMap, MetadataValue};
use xai_core_entities::gizmoduck_client::{GizmoduckClient, QueryFields};
use xai_core_entities::tweet_entity_service_client::TESClient;
use xai_safety_label_source::SafetyLabelSource;
use xai_strato::StratoGrpc;

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
    const DECIDES_EXISTENCE: bool;

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

    fn ran_out(&self) -> bool {
        false
    }
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
    const EDGE: Option<Edge> = Edge::answered_from_the_viewer_by(Self::SOURCE);

    fn fetchers(all: &SourceFetchers) -> &Fetchers<Value<Self>, Self::Key>;
    fn fetchers_mut(all: &mut SourceFetchers) -> &mut Fetchers<Value<Self>, Self::Key>;
}

#[cfg(test)]
pub(crate) trait Held: SourceFetch {
    fn world(worlds: &Worlds) -> &Self::World;
    fn world_mut(worlds: &mut Worlds) -> &mut Self::World;
}

macro_rules! sources {
    ($($source:ident => $backend:literal,)*) => {
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

            pub(super) const fn backend_label(self) -> &'static str {
                match self {
                    $(Source::$source => $backend,)*
                }
            }

            pub(super) fn send<'p>(
                self,
                call: CallRequest<'p>,
                sources: &'p dyn Sources,
                context: CallContext<'p>,
                timed: Timed<'p>,
            ) -> InFlight<'p> {
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
            pub(super) fn answer(&self, id: FetcherId, key: u64) -> Option<Answer> {
                match id.source {
                    $(Source::$source => FetcherKey::answer(self.$source.of(id.owner), key),)*
                }
            }

            pub(super) fn claim(&mut self, id: FetcherId, keys: Vec<u64>) -> Vec<u64> {
                match id.source {
                    $(Source::$source => FetcherKey::claim(self.$source.of_mut(id.owner), keys),)*
                }
            }

            pub(super) fn edges(&self, id: FetcherId) -> Option<&Fetcher<bool, EdgeKey>> {
                match id.source {
                    $(Source::$source => FetcherKey::edges(self.$source.of(id.owner)),)*
                }
            }

            pub(super) fn edges_mut(&mut self, id: FetcherId) -> Option<&mut Fetcher<bool, EdgeKey>> {
                match id.source {
                    $(Source::$source => FetcherKey::edges_mut(self.$source.of_mut(id.owner)),)*
                }
            }
        }

        pub(super) enum Reply {
            $($source(<source::$source as SourceFetch>::Reply),)*
        }

        impl Reply {
            pub(super) fn land(self, store: &mut Store, call: &CallRequest<'_>) {
                match self {
                    $(Reply::$source(reply) => store.land_reply::<source::$source>(call, reply),)*
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
    TesPureCoreData => "tes",
    TesTweetForVisibility => "tes",
    TesConversationControl => "tes",
    TesQuotedTweet => "tes",
    TesTweetState => "tes",
    SafetyLabels => "safety_labels",
    GizmoduckViewer => "gizmoduck",
    GizmoduckAuthor => "gizmoduck",
    Flock => "socialgraph",
    ViewerCountry => "about_this_account",
    Wingman => "wingman",
    CommunityModeration => "communities",
    CommunityModerator => "communities",
    CommunityViewerRemoved => "communities",
    ArticleLifecycle => "articles",
    ArticleMediaEntities => "articles",
    ArticleCoverMedia => "articles",
    TrustedFriends => "trusted_friends",
    UserLocation => "geoduck",
    GizmoduckActor => "gizmoduck",
    ActorEmails => "notifinfra",
    ActorPhones => "notifinfra",
}

pub(crate) struct Backends {
    pub(crate) tes: Arc<dyn TESClient + Send + Sync>,
    pub(crate) tweets: TweetSource,
    pub(crate) gizmoduck: Arc<dyn GizmoduckClient + Send + Sync>,
    pub(crate) socialgraph: Arc<dyn SocialgraphClient + Send + Sync>,
    pub(crate) strato: StratoGrpc,
    pub(crate) wingman: Arc<dyn WingmanClient>,
    pub(crate) safety_labels: Arc<SafetyLabelSource>,
    pub(crate) communities: CommunitySource,
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

fn held_by(keys: &[u64], holders: Option<&HashSet<u64>>) -> RawHydrationBatch<bool> {
    HydrationBatch::from_hydrated(
        keys.iter()
            .map(|&key| {
                let held = holders.map_or(Hydrated::Failed, |holders| {
                    Hydrated::Found(holders.contains(&key))
                });
                (key, held)
            })
            .collect(),
    )
}

fn one_viewer_batch<V, E>(viewer_id: u64, answer: Result<Option<V>, E>) -> RawHydrationBatch<V> {
    HydrationBatch::from_hydrated(FxHashMap::from_iter([(viewer_id, Hydrated::from(answer))]))
}

#[cfg(test)]
pub(crate) use in_memory::{control, found_user, suspended, Fault, InMemoryCall, InMemorySources};

#[cfg(test)]
mod fake_backends;

#[cfg(test)]
mod in_memory;
