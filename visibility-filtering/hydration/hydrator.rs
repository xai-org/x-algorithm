use crate::clients::socialgraph_client::{EdgeDirection, Graph};
use crate::hydration::fetcher::{FetcherId, Owner};
use crate::hydration::inventory::{Stake, Tier};
use crate::hydration::relation::{Fetch, Relation};
use crate::hydration::sources::Source;
use crate::hydration::store::{ControlArms, KeyOrigin, Names};
use crate::rules::Expansion;
use strum::VariantArray;
use xai_core_entities::gizmoduck_client::QueryFields;

const ACTORS: &str = "actor_profile";

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
#[repr(u8)]
pub enum Hydrator {
    PureCore,
    Tweet,
    ConversationControl,
    TweetSafetyLabels,
    ViewerProfile,
    ViewerLabels,
    AuthorSafety,
    AuthorLabels,
    Follows,
    Blocks,
    Mutes,
    MuteRetweets,
    BlockedByAuthor,
    BlockedByReplyRoot,
    SuperFollowsExclusive,
    RootFollowsViewer,
    RootFollowsViewerSecondDegree,
    SuperFollowsRoot,
    ViewerFollowsRoot,
    BlockedByConversationRoot,
    ConversationRootPost,
    BlockedByConversationRootPost,
    ViewerCountry,
    CommunityModeration,
    CommunityModerator,
    CommunityViewerRemoved,
    ArticleLifecycle,
    ArticleMediaEntities,
    ArticleCoverMedia,
    TrustedFriends,
    OutsideNarrowcastPlace,
    QuotedTweet,
    AuthorBlocksOuterAuthor,
    OuterAuthorFollowsAuthor,
    ReactedToPostSafetyLabels,
    ReactedToAuthorSafety,
    ViewerNotificationFilters,
    AuthorProfile,
    AuthorFollowsViewer,
    AuthorConfirmedEmail,
    AuthorVerifiedPhone,
    TweetState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RequestRows {
    #[default]
    Posts,
    Users,
    ActorPosts,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Hydrators(u64);

impl Hydrators {
    pub const fn empty() -> Self {
        Self(0)
    }

    #[cfg(test)]
    pub fn all() -> Self {
        Self(u64::MAX >> (u64::BITS as usize - <Hydrator as strum::VariantArray>::VARIANTS.len()))
    }

    pub const fn of(hydrator: Hydrator) -> Self {
        Self(1 << hydrator as u8)
    }

    pub const fn with(self, hydrator: Hydrator) -> Self {
        self.union(Self::of(hydrator))
    }

    #[cfg(test)]
    pub const fn without(self, hydrator: Hydrator) -> Self {
        Self(self.0 & !Self::of(hydrator).0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, hydrator: Hydrator) -> bool {
        self.0 & Self::of(hydrator).0 != 0
    }

    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Part {
    Column,
    Fields(&'static [QueryFields]),
    Edge(Edge),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, VariantArray)]
#[repr(u8)]
pub(crate) enum Edge {
    Follows,
    Blocks,
    Mutes,
    MuteRetweets,
    BlockedBy,
    SuperFollows,
    FollowedBy,
    SecondDegree,
    TrustedFriends,
    OutsidePlace,
}

impl Edge {
    pub(super) const fn flock(self) -> Option<(Graph, EdgeDirection)> {
        use EdgeDirection::{Forward, Reverse};
        match self {
            Edge::Follows => Some((Graph::Follows, Forward)),
            Edge::Blocks => Some((Graph::Blocks, Forward)),
            Edge::Mutes => Some((Graph::Mutes, Forward)),
            Edge::MuteRetweets => Some((Graph::MuteRetweets, Forward)),
            Edge::BlockedBy => Some((Graph::Blocks, Reverse)),
            Edge::SuperFollows => Some((Graph::SuperFollows, Forward)),
            Edge::FollowedBy => Some((Graph::Follows, Reverse)),
            Edge::SecondDegree | Edge::TrustedFriends | Edge::OutsidePlace => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct UserPair {
    pub(crate) source: u64,
    pub(crate) destination: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EdgeKey {
    pub(super) edge: Edge,
    pub(super) pair: UserPair,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NodeSpec {
    pub(super) source: Source,
    pub(super) part: Part,
    pub(super) key: KeyOrigin,
    pub(super) label: (&'static str, &'static str),
    pub(super) relation: Option<Relation>,
    pub(super) declared: Stake,
}

impl NodeSpec {
    pub(super) const fn stake(self) -> Stake {
        let names_the_row = matches!(self.key.names(), Names::Post | Names::Author);
        if self.source.decides_existence() && names_the_row {
            Stake::Exists
        } else {
            self.declared
        }
    }

    pub(super) const fn fetcher(self) -> FetcherId {
        let owner = match self.relation {
            Some(relation) if relation.owns(self.source) => Owner::Related(relation),
            Some(_) | None => Owner::Request,
        };
        FetcherId {
            source: self.source,
            owner,
        }
    }
}

impl Hydrator {
    pub(super) const fn spec(self) -> NodeSpec {
        use Hydrator as H;
        use KeyOrigin as K;
        use Source as S;
        use Stake::{Absent, BestEffort, MustHave};
        const fn node(
            source: Source,
            part: Part,
            key: KeyOrigin,
            call: &'static str,
            declared: Stake,
        ) -> NodeSpec {
            NodeSpec {
                source,
                part,
                key,
                label: (source.backend_label(), call),
                relation: None,
                declared,
            }
        }
        const fn reads(relation: Relation, node: NodeSpec) -> NodeSpec {
            NodeSpec {
                relation: Some(relation),
                ..node
            }
        }
        const VIEWER_AUTHOR_EDGES: &str = "viewer_author_edges";
        const QUOTE_AUTHOR_EDGES: &str = "quote_author_edges";
        const CONVERSATION_ROOT_AUTHOR_EDGES: &str = "conversation_root_author_edges";
        const REACTED_TO: Stake = Expansion::ReactedTo.spec().on_failure;
        match self {
            H::PureCore => node(
                S::TesPureCoreData,
                Part::Column,
                K::RequestTweets,
                "tweet_pure_core_data",
                MustHave,
            ),
            H::Tweet => node(
                S::TesTweetForVisibility,
                Part::Column,
                K::RequestTweets,
                "tweet_for_visibility",
                MustHave,
            ),
            H::ConversationControl => node(
                S::TesConversationControl,
                Part::Column,
                K::RequestTweets,
                "tweet_conversation_control",
                BestEffort,
            ),
            H::TweetSafetyLabels => node(
                S::SafetyLabels,
                Part::Column,
                K::RequestTweets,
                "tweet_safety_labels",
                MustHave,
            ),
            H::ViewerProfile => node(
                S::GizmoduckViewer,
                Part::Fields(&[
                    QueryFields::ACCOUNT,
                    QueryFields::EXTENDED_PROFILE,
                    QueryFields::SAFETY,
                ]),
                K::Viewer,
                "viewer_profile",
                MustHave,
            ),
            H::ViewerLabels => node(
                S::GizmoduckViewer,
                Part::Fields(&[QueryFields::LABELS]),
                K::Viewer,
                "viewer_profile",
                MustHave,
            ),
            H::AuthorSafety => node(
                S::GizmoduckAuthor,
                Part::Fields(&[QueryFields::SAFETY]),
                K::PureCoreAuthor,
                "author_profile",
                MustHave,
            ),
            H::AuthorLabels => node(
                S::GizmoduckAuthor,
                Part::Fields(&[QueryFields::LABELS]),
                K::PureCoreAuthor,
                "author_profile",
                MustHave,
            ),
            H::Follows => node(
                S::Flock,
                Part::Edge(Edge::Follows),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                BestEffort,
            ),
            H::Blocks => node(
                S::Flock,
                Part::Edge(Edge::Blocks),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::Mutes => node(
                S::Flock,
                Part::Edge(Edge::Mutes),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::MuteRetweets => node(
                S::Flock,
                Part::Edge(Edge::MuteRetweets),
                K::PureCoreRetweeter,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::BlockedByAuthor => node(
                S::Flock,
                Part::Edge(Edge::BlockedBy),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                BestEffort,
            ),
            H::BlockedByReplyRoot => node(
                S::Flock,
                Part::Edge(Edge::BlockedBy),
                K::PureCoreReplyRoot,
                VIEWER_AUTHOR_EDGES,
                BestEffort,
            ),
            H::SuperFollowsExclusive => node(
                S::Flock,
                Part::Edge(Edge::SuperFollows),
                K::ExclusiveConversationAuthor,
                "exclusive_super_follow",
                MustHave,
            ),
            H::RootFollowsViewer => node(
                S::Flock,
                Part::Edge(Edge::FollowedBy),
                K::ControlRoot(ControlArms::CommunityOrMyNetwork),
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::RootFollowsViewerSecondDegree => node(
                S::Wingman,
                Part::Edge(Edge::SecondDegree),
                K::MyNetworkRootNotFollowingViewer,
                "conversation_root_author_second_degree",
                Absent(Tier::BestEffort),
            ),
            H::SuperFollowsRoot => node(
                S::Flock,
                Part::Edge(Edge::SuperFollows),
                K::ControlRoot(ControlArms::Subscribers),
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::ViewerFollowsRoot => node(
                S::Flock,
                Part::Edge(Edge::Follows),
                K::ControlRoot(ControlArms::Followers),
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::BlockedByConversationRoot => node(
                S::Flock,
                Part::Edge(Edge::BlockedBy),
                K::ConversationControlBlockAuthor,
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::ConversationRootPost => reads(
                Relation::ConversationRoot,
                node(
                    S::TesPureCoreData,
                    Part::Column,
                    K::RelatedTweet,
                    "conversation_root_pure_core_data",
                    BestEffort,
                ),
            ),
            H::BlockedByConversationRootPost => reads(
                Relation::ConversationRoot,
                node(
                    S::Flock,
                    Part::Edge(Edge::BlockedBy),
                    K::ConversationRootPostAuthor,
                    "conversation_root_author_blocked_by",
                    BestEffort,
                ),
            ),
            H::ViewerCountry => node(
                S::ViewerCountry,
                Part::Column,
                K::ViewerForCoAllowedList,
                "viewer_country",
                BestEffort,
            ),
            H::CommunityModeration => node(
                S::CommunityModeration,
                Part::Column,
                K::CommunityPost,
                "community_moderation",
                MustHave,
            ),
            H::CommunityModerator => node(
                S::CommunityModerator,
                Part::Column,
                K::ModeratedCommunity,
                "community_moderator",
                MustHave,
            ),
            H::CommunityViewerRemoved => node(
                S::CommunityViewerRemoved,
                Part::Column,
                K::TweetCommunity,
                "community_viewer_removed",
                BestEffort,
            ),
            H::ArticleLifecycle => node(
                S::ArticleLifecycle,
                Part::Column,
                K::TweetArticle,
                "article_lifecycle",
                MustHave,
            ),
            H::ArticleMediaEntities => node(
                S::ArticleMediaEntities,
                Part::Column,
                K::TweetArticle,
                "article_media_entities",
                Absent(Tier::MustHave),
            ),
            H::ArticleCoverMedia => node(
                S::ArticleCoverMedia,
                Part::Column,
                K::TweetArticle,
                "article_cover_media",
                Absent(Tier::MustHave),
            ),
            H::TrustedFriends => node(
                S::TrustedFriends,
                Part::Edge(Edge::TrustedFriends),
                K::TrustedFriendsList,
                "trusted_friends_membership",
                Absent(Tier::MustHave),
            ),
            H::OutsideNarrowcastPlace => node(
                S::UserLocation,
                Part::Edge(Edge::OutsidePlace),
                K::NarrowcastPlace,
                "viewer_location",
                BestEffort,
            ),
            H::AuthorBlocksOuterAuthor => reads(
                Relation::QuoteOuter,
                node(
                    S::Flock,
                    Part::Edge(Edge::Blocks),
                    K::QuotedAuthorToQuotingAuthor,
                    QUOTE_AUTHOR_EDGES,
                    MustHave,
                ),
            ),
            H::OuterAuthorFollowsAuthor => reads(
                Relation::QuoteOuter,
                node(
                    S::Flock,
                    Part::Edge(Edge::Follows),
                    K::QuotingAuthorToProtectedQuotedAuthor,
                    QUOTE_AUTHOR_EDGES,
                    BestEffort,
                ),
            ),
            H::QuotedTweet => node(
                S::TesQuotedTweet,
                Part::Column,
                K::OuterOfMissingQuote,
                "quoted_tweet",
                BestEffort,
            ),
            H::ReactedToPostSafetyLabels => reads(
                Relation::ReactedTo,
                node(
                    S::SafetyLabels,
                    Part::Column,
                    K::RelatedTweet,
                    "reacted_to_safety_labels",
                    REACTED_TO,
                ),
            ),
            H::ReactedToAuthorSafety => reads(
                Relation::ReactedTo,
                node(
                    S::GizmoduckAuthor,
                    Part::Fields(&[QueryFields::SAFETY]),
                    K::RelatedAuthor,
                    "reacted_to_author_profile",
                    REACTED_TO,
                ),
            ),
            H::ViewerNotificationFilters => node(
                S::GizmoduckViewer,
                Part::Fields(&[QueryFields::MUTE_SETTINGS]),
                K::Viewer,
                "viewer_profile",
                MustHave,
            ),
            H::AuthorProfile => node(
                S::GizmoduckActor,
                Part::Fields(&[QueryFields::PROFILE]),
                K::RequestUsers,
                ACTORS,
                MustHave,
            ),
            H::AuthorFollowsViewer => node(
                S::Flock,
                Part::Edge(Edge::FollowedBy),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::AuthorConfirmedEmail => node(
                S::ActorEmails,
                Part::Column,
                K::UserIfViewerFiltersNoConfirmedEmail,
                "actor_emails",
                MustHave,
            ),
            H::AuthorVerifiedPhone => node(
                S::ActorPhones,
                Part::Column,
                K::UserIfViewerFiltersNoConfirmedPhone,
                "actor_phones",
                MustHave,
            ),
            H::TweetState => node(
                S::TesTweetState,
                Part::Column,
                K::MissingRequestTweet,
                "tweet_state",
                Absent(Tier::BestEffort),
            ),
        }
    }

    pub(super) const fn spec_at(self, rows: RequestRows) -> NodeSpec {
        let spec = self.spec();
        match (rows, spec.fetcher()) {
            (
                RequestRows::Users,
                FetcherId {
                    source: Source::GizmoduckAuthor,
                    owner: Owner::Request,
                },
            ) => NodeSpec {
                source: Source::GizmoduckActor,
                key: KeyOrigin::RequestUsers,
                label: (Source::GizmoduckActor.backend_label(), ACTORS),
                ..spec
            },
            (
                RequestRows::ActorPosts,
                FetcherId {
                    source: Source::GizmoduckAuthor,
                    owner: Owner::Request,
                },
            ) => NodeSpec {
                source: Source::GizmoduckActor,
                key: KeyOrigin::PureCoreAuthor,
                label: (Source::GizmoduckActor.backend_label(), ACTORS),
                ..spec
            },
            (RequestRows::Posts | RequestRows::Users | RequestRows::ActorPosts, _) => NodeSpec {
                key: spec.key.at(rows),
                ..spec
            },
        }
    }

    #[cfg(test)]
    pub(crate) const fn declared_stake(self) -> Stake {
        self.spec().declared
    }

    pub(crate) const fn is_best_effort(self) -> bool {
        matches!(self.spec().declared.tier(), Tier::BestEffort)
    }

    pub(super) const fn input(self) -> Option<Hydrator> {
        self.input_at(RequestRows::Posts)
    }

    pub(crate) const fn relation(self) -> Option<Relation> {
        self.spec().relation
    }

    pub(super) const fn key_at(self, rows: RequestRows) -> KeyOrigin {
        self.spec_at(rows).key
    }

    pub(super) const fn input_at(self, rows: RequestRows) -> Option<Hydrator> {
        let input = self.key_at(rows).input();
        match (input, self.spec().relation) {
            (None, Some(relation)) => relation.spec().trigger.input(),
            _ => input,
        }
    }

    pub(super) const fn precedes_author(self, rows: RequestRows) -> bool {
        matches!(self.input_at(rows), None | Some(Hydrator::Tweet))
    }

    pub(super) const fn reads_at(self, rows: RequestRows) -> Hydrators {
        let reads = self.key_at(rows).reads();
        match self.spec().relation {
            Some(relation) => reads.union(relation.spec().trigger.reads()),
            None => reads,
        }
    }

    const fn closed_over_at(self, rows: RequestRows) -> Hydrators {
        let inputs = match self.input_at(rows) {
            Some(input) => Hydrators::of(input),
            None => Hydrators::empty(),
        };
        match self.spec().relation {
            Some(_) => inputs.union(self.reads_at(rows)),
            None => inputs,
        }
    }

    pub(super) const fn edge(self) -> Option<Edge> {
        match self.spec().part {
            Part::Edge(edge) => Some(edge),
            Part::Column | Part::Fields(_) => None,
        }
    }

    pub(super) const fn is_quote_pair_edge(self) -> bool {
        self.is_edge() && self.spec().key.is_quote_pair()
    }

    pub(crate) const fn is_viewer_edge(self) -> bool {
        self.is_edge() && !self.spec().key.is_quote_pair()
    }

    pub(super) const fn is_edge(self) -> bool {
        self.edge().is_some()
    }

    pub(crate) const fn needs_viewer(self) -> bool {
        let spec = self.spec();
        let joins_for_viewer = match spec.relation {
            Some(relation) => relation.spec().trigger.needs_viewer(),
            None => false,
        };
        self.is_viewer_edge() || spec.key.needs_viewer() || joins_for_viewer
    }
}

const MUST_HAVE: Hydrators = {
    let mut must_have = Hydrators::empty();
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        if !node.is_best_effort() {
            must_have = must_have.with(*node);
        }
        rest = tail;
    }
    must_have
};

impl Hydrators {
    pub(crate) const BLOCKED_BY_CONVERSATION_ROOT_AUTHOR: Self =
        Self::of(Hydrator::BlockedByReplyRoot)
            .with(Hydrator::BlockedByConversationRoot)
            .with(Hydrator::BlockedByConversationRootPost);

    pub(crate) const fn must_have(self) -> Self {
        self.intersection(MUST_HAVE)
    }

    pub(crate) const fn has_must_have(self) -> bool {
        !self.must_have().is_empty()
    }
}

const fn inputs_precede_nodes() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        let inputs = node
            .closed_over_at(RequestRows::Posts)
            .union(node.closed_over_at(RequestRows::Users))
            .union(node.closed_over_at(RequestRows::ActorPosts));
        let mut others = Hydrator::VARIANTS;
        while let [other, others_tail @ ..] = others {
            if inputs.contains(*other) && *other as u8 >= *node as u8 {
                return false;
            }
            others = others_tail;
        }
        rest = tail;
    }
    true
}

const fn best_effort_inputs_key_only_best_effort_nodes() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        if let Some(input) = node.input()
            && input.is_best_effort()
            && !node.is_best_effort()
        {
            return false;
        }
        rest = tail;
    }
    true
}

const fn owning_relations_read_owned_sources() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        let spec = node.spec();
        let fits = match spec.relation {
            Some(relation) if matches!(relation.spec().fetch, Fetch::Own(_)) => {
                relation.owns(spec.source) || matches!(spec.part, Part::Edge(_))
            }
            Some(_) | None => true,
        };
        if !fits {
            return false;
        }
        rest = tail;
    }
    true
}

const fn stakes_fit_their_nodes() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        let spec = node.spec();
        let derives_exists = matches!(spec.stake(), Stake::Exists)
            || matches!(node.spec_at(RequestRows::Users).stake(), Stake::Exists)
            || matches!(node.spec_at(RequestRows::ActorPosts).stake(), Stake::Exists);
        let fits = !matches!(spec.declared, Stake::Exists)
            && (!derives_exists || matches!(spec.declared, Stake::MustHave))
            && (spec.relation.is_none() || !derives_exists);
        if !fits {
            return false;
        }
        rest = tail;
    }
    true
}

const fn quote_pair_edges_are_the_quote_outers_edges() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        let quote_outer_owns = matches!(
            node.spec().fetcher().owner,
            Owner::Related(Relation::QuoteOuter)
        );
        if node.is_edge() && quote_outer_owns != node.is_quote_pair_edge() {
            return false;
        }
        rest = tail;
    }
    true
}

const _: () = assert!(inputs_precede_nodes());
const _: () = assert!(quote_pair_edges_are_the_quote_outers_edges());
const _: () = assert!(stakes_fit_their_nodes());
const _: () = assert!(best_effort_inputs_key_only_best_effort_nodes());
const _: () = assert!(owning_relations_read_owned_sources());
const _: () = assert!(Hydrator::VARIANTS.len() <= u64::BITS as usize);

impl Hydrators {
    pub(crate) const fn closed(self, rows: RequestRows) -> Self {
        let mut closed = match rows {
            RequestRows::Posts | RequestRows::ActorPosts => self.with(Hydrator::PureCore),
            RequestRows::Users => self,
        };
        let mut rest = Hydrator::VARIANTS;
        while let [head @ .., node] = rest {
            if closed.contains(*node) {
                closed = closed.union(node.closed_over_at(rows));
            }
            rest = head;
        }
        closed
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = Hydrator> {
        Hydrator::VARIANTS
            .iter()
            .copied()
            .filter(move |&node| self.contains(node))
    }
}
