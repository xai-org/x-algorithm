use super::conversation_root::ConversationRootAuthor;
use super::{RequestRow, RowSubject, Store};
use crate::hydration::fetcher::AnyFetcher;
use crate::hydration::hydrator::UserPair;
use crate::hydration::relation::Related;
use crate::hydration::sources::source;
use crate::hydration::{Cause, Hydrator, Hydrators, RequestRows};
use crate::models::{AuthorId, ClientSwitch, RetweetMerge, TweetId};
use std::fmt;
use std::num::NonZeroU64;
use strum::VariantArray;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::hydration) enum KeyOrigin {
    RequestTweets,
    RequestUsers,
    Viewer,
    PureCoreAuthor,
    PureCoreRetweeter,
    PureCoreReplyRoot,
    ExclusiveConversationAuthor,
    TweetArticle,
    ControlRoot(ControlArms),
    ConversationControlBlockAuthor,
    ConversationRootToFetch,
    RelatedTweet,
    RelatedAuthor,
    ConversationRootPostAuthor,
    ViewerForCoAllowedList,
    MyNetworkRootNotFollowingViewer,
    CommunityPost,
    ModeratedCommunity,
    TweetCommunity,
    TrustedFriendsList,
    NarrowcastPlace,
    QuotedAuthorToQuotingAuthor,
    QuotingAuthorToProtectedQuotedAuthor,
    OuterOfMissingQuote,
    MissingRequestTweet,
    UserIfViewerFiltersNoConfirmedEmail,
    UserIfViewerFiltersNoConfirmedPhone,
    AuthorIfViewerFiltersNoConfirmedEmail,
    AuthorIfViewerFiltersNoConfirmedPhone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::hydration) enum Names {
    Post,
    Author,
    Viewer,
    Related,
}

enum KeyedBy {
    Request(fn(&Store) -> Option<u64>),
    Row(fn(&Store, &RequestRow) -> Option<u64>),
    Related(fn(&Store, Related) -> Option<u64>),
    Quote(fn(&Store, &RequestRow, TweetId) -> PairKey),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PairKey {
    Asked(UserPair),
    Unasked,
    InputFailed,
}

struct Spec {
    label: &'static str,
    input: Option<Hydrator>,
    also_reads: Hydrators,
    names: Names,
    needs_viewer: bool,
    keyed_by: KeyedBy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, VariantArray)]
pub(in crate::hydration) enum ControlArms {
    CommunityOrMyNetwork,
    Subscribers,
    Followers,
}

impl ControlArms {
    fn contains(self, arm: ConversationControlArm) -> bool {
        let arms: &[ConversationControlArm] = match self {
            Self::CommunityOrMyNetwork => &[
                ConversationControlArm::Community,
                ConversationControlArm::MyNetwork,
            ],
            Self::Subscribers => &[ConversationControlArm::Subscribers],
            Self::Followers => &[ConversationControlArm::Followers],
        };
        arms.contains(&arm)
    }

    const fn label(self) -> &'static str {
        match self {
            Self::CommunityOrMyNetwork => "root:Community|MyNetwork",
            Self::Subscribers => "root:Subscribers",
            Self::Followers => "root:Followers",
        }
    }

    const fn key(self) -> fn(&Store, &RequestRow) -> Option<u64> {
        match self {
            Self::CommunityOrMyNetwork => {
                |store, row| store.control_root_author(row, Self::CommunityOrMyNetwork)
            }
            Self::Subscribers => |store, row| store.control_root_author(row, Self::Subscribers),
            Self::Followers => |store, row| store.control_root_author(row, Self::Followers),
        }
    }
}

impl KeyOrigin {
    const fn spec(self) -> Spec {
        use Hydrator as H;
        use KeyOrigin as K;
        const NOTHING: Hydrators = Hydrators::empty();
        match self {
            K::RequestTweets => Spec {
                label: "tweets",
                input: None,
                also_reads: NOTHING,
                names: Names::Post,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|_, row| row.tweet_id().map(|tweet_id| tweet_id.0)),
            },
            K::RequestUsers => Spec {
                label: "users",
                input: None,
                also_reads: NOTHING,
                names: Names::Author,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|_, row| match row.subject {
                    RowSubject::User => row.author.map(AuthorId::get),
                    RowSubject::Post(_) => None,
                }),
            },
            K::Viewer => Spec {
                label: "viewer",
                input: None,
                also_reads: NOTHING,
                names: Names::Viewer,
                needs_viewer: true,
                keyed_by: KeyedBy::Request(|store| store.facts.viewer_id()),
            },
            K::PureCoreAuthor => Spec {
                label: "author",
                input: Some(H::PureCore),
                also_reads: Hydrators::of(H::QuotedTweet),
                names: Names::Author,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|_, row| row.tweet_id().and(row.author).map(AuthorId::get)),
            },
            K::PureCoreRetweeter => Spec {
                label: "retweeter",
                input: Some(H::PureCore),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .source_tweet_id(row.tweet_id()?)
                        .and(row.author)
                        .map(AuthorId::get)
                }),
            },
            K::PureCoreReplyRoot => Spec {
                label: "reply_root",
                input: Some(H::PureCore),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| match store.conversation_root_author(row)? {
                    ConversationRootAuthor::PureCore(author) => Some(author.get()),
                    ConversationRootAuthor::ConversationControl(_)
                    | ConversationRootAuthor::RootPost(_)
                    | ConversationRootAuthor::RootPureCoreFailed
                    | ConversationRootAuthor::Unresolved => None,
                }),
            },
            K::ExclusiveConversationAuthor => Spec {
                label: "exclusive_author",
                input: Some(H::Tweet),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .request_fetcher::<source::TesTweetForVisibility>()
                        .get(row.tweet_id()?.0)?
                        .exclusive_conversation_author_id
                }),
            },
            K::TweetArticle => Spec {
                label: "article",
                input: Some(H::Tweet),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .request_fetcher::<source::TesTweetForVisibility>()
                        .get(row.tweet_id()?.0)?
                        .article_id
                        .map(NonZeroU64::get)
                }),
            },
            K::ControlRoot(arms) => Spec {
                label: arms.label(),
                input: Some(H::ConversationControl),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(arms.key()),
            },
            K::ConversationControlBlockAuthor => Spec {
                label: "control_root_author",
                input: Some(H::ConversationControl),
                also_reads: Hydrators::of(H::PureCore),
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    let control = store.control(row)?;
                    if arm_starts_conversation_root_select(control.arm) {
                        return Some(control.conversation_tweet_author_id);
                    }
                    match store.conversation_root_author(row)? {
                        ConversationRootAuthor::ConversationControl(author) => Some(author.get()),
                        ConversationRootAuthor::PureCore(_)
                        | ConversationRootAuthor::RootPost(_)
                        | ConversationRootAuthor::RootPureCoreFailed
                        | ConversationRootAuthor::Unresolved => None,
                    }
                }),
            },
            K::ConversationRootToFetch => Spec {
                label: "conversation_root_to_fetch",
                input: Some(H::PureCore),
                also_reads: Hydrators::of(H::ConversationControl),
                names: Names::Related,
                needs_viewer: true,
                keyed_by: KeyedBy::Row(|store, row| match store.conversation_root_author(row)? {
                    ConversationRootAuthor::RootPost(root) => Some(root.0),
                    ConversationRootAuthor::PureCore(_)
                    | ConversationRootAuthor::ConversationControl(_)
                    | ConversationRootAuthor::RootPureCoreFailed
                    | ConversationRootAuthor::Unresolved => None,
                }),
            },
            K::RelatedTweet => Spec {
                label: "related_tweet",
                input: None,
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Related(|_, related| Some(related.tweet_id.0)),
            },
            K::RelatedAuthor => Spec {
                label: "related_author",
                input: None,
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Related(|_, related| related.author.map(AuthorId::get)),
            },
            K::ConversationRootPostAuthor => Spec {
                label: "conversation_root_post_author",
                input: Some(H::ConversationRootPost),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Related(|store, root| {
                    store
                        .fetcher_of::<source::TesPureCoreData>(store.owner(H::ConversationRootPost))
                        .get(root.tweet_id.0)
                        .map(|core| core.author_id.get())
                }),
            },
            K::ViewerForCoAllowedList => Spec {
                label: "viewer:co_allowed_list",
                input: Some(H::ConversationControl),
                also_reads: NOTHING,
                names: Names::Viewer,
                needs_viewer: true,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .facts
                        .viewer_id()
                        .filter(|_| store.control(row).is_some_and(lists_countries))
                }),
            },
            K::MyNetworkRootNotFollowingViewer => Spec {
                label: "root:MyNetwork:not_followed",
                input: Some(H::RootFollowsViewer),
                also_reads: Hydrators::of(H::ConversationControl),
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .control(row)
                        .and_then(|control| store.root_not_following_viewer(control))
                }),
            },
            K::CommunityPost => Spec {
                label: "community_post",
                input: Some(H::Tweet),
                also_reads: Hydrators::of(H::PureCore),
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store.community_post(row).map(|post| post.tweet_id)
                }),
            },
            K::ModeratedCommunity => Spec {
                label: "moderated_community",
                input: Some(H::CommunityModeration),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: true,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .community_post(row)
                        .filter(|post| {
                            store
                                .request_fetcher::<source::CommunityModeration>()
                                .get(post.tweet_id)
                                .is_some_and(|moderation| moderation.is_moderated())
                        })
                        .map(|post| post.community_id)
                }),
            },
            K::TweetCommunity => Spec {
                label: "tweet_community",
                input: Some(H::Tweet),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: true,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .request_fetcher::<source::TesTweetForVisibility>()
                        .get(row.tweet_id()?.0)?
                        .community_id
                        .filter(|_| {
                            store
                                .facts
                                .client_capability()
                                .has(ClientSwitch::CommunityViewerRemovedLimits)
                        })
                        .map(NonZeroU64::get)
                }),
            },
            K::TrustedFriendsList => Spec {
                label: "trusted_friends_list",
                input: Some(H::Tweet),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .request_fetcher::<source::TesTweetForVisibility>()
                        .get(row.tweet_id()?.0)?
                        .trusted_friends_list_id
                }),
            },
            K::NarrowcastPlace => Spec {
                label: "narrowcast_place",
                input: Some(H::Tweet),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .request_fetcher::<source::TesTweetForVisibility>()
                        .get(row.tweet_id()?.0)?
                        .narrowcast_place_id
                }),
            },
            K::QuotedAuthorToQuotingAuthor => Spec {
                label: "quoted_author>quoting_author",
                input: Some(H::PureCore),
                also_reads: Hydrators::of(H::QuotedTweet),
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Quote(|store, parent, outer| {
                    store.quote_pair(parent, outer, |quoted, quoting| {
                        PairKey::Asked(UserPair {
                            source: quoted.get(),
                            destination: quoting.get(),
                        })
                    })
                }),
            },
            K::QuotingAuthorToProtectedQuotedAuthor => Spec {
                label: "quoting_author>protected_quoted_author",
                input: Some(H::AuthorSafety),
                also_reads: Hydrators::of(H::PureCore).with(H::QuotedTweet),
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Quote(|store, parent, outer| {
                    store.quote_pair(parent, outer, |quoted, quoting| {
                        let authors = store.request_fetcher::<source::GizmoduckAuthor>();
                        match authors
                            .get(quoted.get())
                            .filter(|_| !authors.is_incomplete(quoted.get()))
                        {
                            None => PairKey::InputFailed,
                            Some(author) if author.features.is_protected => {
                                PairKey::Asked(UserPair {
                                    source: quoting.get(),
                                    destination: quoted.get(),
                                })
                            }
                            Some(_) => PairKey::Unasked,
                        }
                    })
                }),
            },
            K::OuterOfMissingQuote => Spec {
                label: "outer_of_missing_quote",
                input: Some(H::PureCore),
                also_reads: NOTHING,
                names: Names::Related,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    let tweet_id = row.tweet_id()?;
                    store
                        .quotes
                        .outer_of(tweet_id)
                        .filter(|_| {
                            store.existence_miss(H::PureCore, tweet_id.0) == Some(Cause::NotFound)
                        })
                        .map(|outer| outer.0)
                }),
            },
            K::MissingRequestTweet => Spec {
                label: "missing_request_tweet",
                input: Some(H::Tweet),
                also_reads: NOTHING,
                names: Names::Post,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    let tweet_id = row.tweet_id()?;
                    let missing = store.retweet_merge == RetweetMerge::Merged
                        && store.requested_tweet_ids.contains(&tweet_id)
                        && store.existence_miss(H::Tweet, tweet_id.0) == Some(Cause::NotFound);
                    missing.then_some(tweet_id.0)
                }),
            },
            K::UserIfViewerFiltersNoConfirmedEmail => Spec {
                label: "users:viewer_filters_no_confirmed_email",
                input: Some(H::ViewerNotificationFilters),
                also_reads: NOTHING,
                names: Names::Author,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .key(K::RequestUsers, row)
                        .filter(|_| store.viewer_filters(|filters| filters.no_confirmed_email))
                }),
            },
            K::UserIfViewerFiltersNoConfirmedPhone => Spec {
                label: "users:viewer_filters_no_confirmed_phone",
                input: Some(H::ViewerNotificationFilters),
                also_reads: NOTHING,
                names: Names::Author,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .key(K::RequestUsers, row)
                        .filter(|_| store.viewer_filters(|filters| filters.no_confirmed_phone))
                }),
            },
            K::AuthorIfViewerFiltersNoConfirmedEmail => Spec {
                label: "author:viewer_filters_no_confirmed_email",
                input: Some(H::PureCore),
                also_reads: Hydrators::of(H::ViewerNotificationFilters),
                names: Names::Author,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .key(K::PureCoreAuthor, row)
                        .filter(|_| store.viewer_filters(|filters| filters.no_confirmed_email))
                }),
            },
            K::AuthorIfViewerFiltersNoConfirmedPhone => Spec {
                label: "author:viewer_filters_no_confirmed_phone",
                input: Some(H::PureCore),
                also_reads: Hydrators::of(H::ViewerNotificationFilters),
                names: Names::Author,
                needs_viewer: false,
                keyed_by: KeyedBy::Row(|store, row| {
                    store
                        .key(K::PureCoreAuthor, row)
                        .filter(|_| store.viewer_filters(|filters| filters.no_confirmed_phone))
                }),
            },
        }
    }

    pub(in crate::hydration) const fn input(self) -> Option<Hydrator> {
        self.spec().input
    }

    pub(in crate::hydration) const fn reads(self) -> Hydrators {
        let spec = self.spec();
        match spec.input {
            Some(input) => spec.also_reads.with(input),
            None => spec.also_reads,
        }
    }

    pub(in crate::hydration) const fn names(self) -> Names {
        self.spec().names
    }

    pub(in crate::hydration) const fn needs_viewer(self) -> bool {
        self.spec().needs_viewer
    }

    pub(in crate::hydration) const fn is_quote_pair(self) -> bool {
        matches!(self.spec().keyed_by, KeyedBy::Quote(_))
    }

    pub(in crate::hydration) const fn keys_user_rows(self) -> bool {
        let after_at_most_the_viewer = match self.input() {
            None => true,
            Some(input) => matches!(input.spec().key.names(), Names::Viewer),
        };
        after_at_most_the_viewer && matches!(self.names(), Names::Author | Names::Viewer)
    }

    pub(in crate::hydration) const fn keys_only_user_rows(self) -> bool {
        self.keys_user_rows() && matches!(self.names(), Names::Author)
    }

    pub(in crate::hydration) const fn at(self, rows: RequestRows) -> Self {
        match (rows, self) {
            (RequestRows::Users, KeyOrigin::PureCoreAuthor) => KeyOrigin::RequestUsers,
            (RequestRows::ActorPosts, KeyOrigin::RequestUsers) => KeyOrigin::PureCoreAuthor,
            (RequestRows::ActorPosts, KeyOrigin::UserIfViewerFiltersNoConfirmedEmail) => {
                KeyOrigin::AuthorIfViewerFiltersNoConfirmedEmail
            }
            (RequestRows::ActorPosts, KeyOrigin::UserIfViewerFiltersNoConfirmedPhone) => {
                KeyOrigin::AuthorIfViewerFiltersNoConfirmedPhone
            }
            _ => self,
        }
    }
}

impl fmt::Display for KeyOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.spec().label)
    }
}

fn arm_starts_conversation_root_select(arm: ConversationControlArm) -> bool {
    ControlArms::VARIANTS.iter().any(|arms| arms.contains(arm))
}

fn lists_countries(control: &ConversationControl) -> bool {
    control.arm == ConversationControlArm::Co && !control.allowed_country_codes.is_empty()
}

impl Store {
    pub(super) fn key(&self, origin: KeyOrigin, row: &RequestRow) -> Option<u64> {
        let spec = origin.spec();
        if spec.needs_viewer && self.facts.viewer_id().is_none() {
            return None;
        }
        match spec.keyed_by {
            KeyedBy::Request(key) => key(self),
            KeyedBy::Row(key) => key(self, row),
            KeyedBy::Related(_) | KeyedBy::Quote(_) => None,
        }
    }

    pub(super) fn quote_pair_key(
        &self,
        origin: KeyOrigin,
        parent: &RequestRow,
        outer: TweetId,
    ) -> PairKey {
        match origin.spec().keyed_by {
            KeyedBy::Quote(key) => key(self, parent, outer),
            KeyedBy::Request(_) | KeyedBy::Row(_) | KeyedBy::Related(_) => PairKey::Unasked,
        }
    }

    pub(super) fn related_key(&self, origin: KeyOrigin, related: Related) -> Option<u64> {
        let spec = origin.spec();
        if spec.needs_viewer && self.facts.viewer_id().is_none() {
            return None;
        }
        match spec.keyed_by {
            KeyedBy::Related(key) => key(self, related),
            KeyedBy::Request(_) | KeyedBy::Row(_) | KeyedBy::Quote(_) => None,
        }
    }

    pub(super) fn node_keys(&self, node: Hydrator) -> Vec<u64> {
        let spec = self.origin(node).spec();
        if spec.needs_viewer && self.facts.viewer_id().is_none() {
            return Vec::new();
        }
        match spec.keyed_by {
            KeyedBy::Request(key) => key(self).into_iter().collect(),
            KeyedBy::Row(key) => self.rows.iter().filter_map(|row| key(self, row)).collect(),
            KeyedBy::Related(key) => node.spec().relation.map_or_else(Vec::new, |relation| {
                self.related_posts(relation)
                    .filter_map(|related| key(self, related))
                    .collect()
            }),
            KeyedBy::Quote(_) => Vec::new(),
        }
    }

    fn control_root_author(&self, row: &RequestRow, arms: ControlArms) -> Option<u64> {
        self.control(row)
            .filter(|control| arms.contains(control.arm))
            .map(|control| control.conversation_tweet_author_id)
    }
}
