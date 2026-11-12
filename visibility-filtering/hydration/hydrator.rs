use crate::clients::socialgraph_client::{EdgeDirection, Graph};
use crate::hydration::fetcher::{FetcherId, Owner};
use crate::hydration::inventory::{Stake, Tier};
use crate::hydration::relation::Relation;
use crate::hydration::sources::Source;
use crate::hydration::store::{ControlArms, KeyOrigin, Names};
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
    Column(Source),
    Fields(Source, &'static [QueryFields]),
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

#[derive(Clone, Copy)]
enum AnsweredBy {
    Flock(Graph, EdgeDirection),
    Viewer(Source),
}

impl Edge {
    const fn answered_by(self) -> AnsweredBy {
        use AnsweredBy::{Flock, Viewer};
        use EdgeDirection::{Forward, Reverse};
        match self {
            Edge::Follows => Flock(Graph::Follows, Forward),
            Edge::Blocks => Flock(Graph::Blocks, Forward),
            Edge::Mutes => Flock(Graph::Mutes, Forward),
            Edge::MuteRetweets => Flock(Graph::MuteRetweets, Forward),
            Edge::BlockedBy => Flock(Graph::Blocks, Reverse),
            Edge::SuperFollows => Flock(Graph::SuperFollows, Forward),
            Edge::FollowedBy => Flock(Graph::Follows, Reverse),
            Edge::SecondDegree => Viewer(Source::Wingman),
            Edge::TrustedFriends => Viewer(Source::TrustedFriends),
            Edge::OutsidePlace => Viewer(Source::UserLocation),
        }
    }

    pub(super) const fn flock(self) -> Option<(Graph, EdgeDirection)> {
        match self.answered_by() {
            AnsweredBy::Flock(graph, direction) => Some((graph, direction)),
            AnsweredBy::Viewer(_) => None,
        }
    }

    const fn source(self) -> Source {
        match self.answered_by() {
            AnsweredBy::Flock(..) => Source::Flock,
            AnsweredBy::Viewer(source) => source,
        }
    }

        pub(super) const fn answered_from_the_viewer_by(source: Source) -> Option<Edge> {
        let mut rest = Edge::VARIANTS;
        while let [edge, tail @ ..] = rest {
            if let AnsweredBy::Viewer(answering) = edge.answered_by()
                && answering as u8 == source as u8
            {
                return Some(*edge);
            }
            rest = tail;
        }
        None
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
    pub(super) part: Part,
    pub(super) key: KeyOrigin,
    pub(super) call: &'static str,
    pub(super) declared: Stake,
}

impl NodeSpec {
    pub(super) const fn source(self) -> Source {
        match self.part {
            Part::Column(source) | Part::Fields(source, _) => source,
            Part::Edge(edge) => edge.source(),
        }
    }

    pub(super) const fn label(self) -> (&'static str, &'static str) {
        (self.source().backend_label(), self.call)
    }

    pub(super) const fn stake(self) -> Stake {
        let names_the_row = matches!(self.key.names(), Names::Post | Names::Author);
        if self.source().decides_existence() && names_the_row {
            Stake::Exists
        } else {
            self.declared
        }
    }

    const fn is_viewer_edge(self) -> bool {
        matches!(self.part, Part::Edge(_)) && !self.key.is_quote_pair()
    }

    pub(super) const fn fetcher(self) -> FetcherId {
        let owner = match self.key.relation() {
            Some(relation) if !self.is_viewer_edge() => Owner::Related(relation),
            Some(_) | None => Owner::Request,
        };
        FetcherId {
            source: self.source(),
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
        const fn node(part: Part, key: KeyOrigin, call: &'static str, declared: Stake) -> NodeSpec {
            NodeSpec {
                part,
                key,
                call,
                declared,
            }
        }
        const VIEWER_AUTHOR_EDGES: &str = "viewer_author_edges";
        const QUOTE_AUTHOR_EDGES: &str = "quote_author_edges";
        const CONVERSATION_ROOT_AUTHOR_EDGES: &str = "conversation_root_author_edges";
        match self {
            H::PureCore => node(
                Part::Column(S::TesPureCoreData),
                K::RequestTweets,
                "tweet_pure_core_data",
                MustHave,
            ),
            H::Tweet => node(
                Part::Column(S::TesTweetForVisibility),
                K::RequestTweets,
                "tweet_for_visibility",
                MustHave,
            ),
            H::ConversationControl => node(
                Part::Column(S::TesConversationControl),
                K::RequestTweets,
                "tweet_conversation_control",
                BestEffort,
            ),
            H::TweetSafetyLabels => node(
                Part::Column(S::SafetyLabels),
                K::RequestTweets,
                "tweet_safety_labels",
                MustHave,
            ),
            H::ViewerProfile => node(
                Part::Fields(
                    S::GizmoduckViewer,
                    &[
                        QueryFields::ACCOUNT,
                        QueryFields::EXTENDED_PROFILE,
                        QueryFields::SAFETY,
                    ],
                ),
                K::Viewer,
                "viewer_profile",
                MustHave,
            ),
            H::ViewerLabels => node(
                Part::Fields(S::GizmoduckViewer, &[QueryFields::LABELS]),
                K::Viewer,
                "viewer_profile",
                MustHave,
            ),
            H::AuthorSafety => node(
                Part::Fields(S::GizmoduckAuthor, &[QueryFields::SAFETY]),
                K::PureCoreAuthor,
                "author_profile",
                MustHave,
            ),
            H::AuthorLabels => node(
                Part::Fields(S::GizmoduckAuthor, &[QueryFields::LABELS]),
                K::PureCoreAuthor,
                "author_profile",
                MustHave,
            ),
            H::Follows => node(
                Part::Edge(Edge::Follows),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                BestEffort,
            ),
            H::Blocks => node(
                Part::Edge(Edge::Blocks),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::Mutes => node(
                Part::Edge(Edge::Mutes),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::MuteRetweets => node(
                Part::Edge(Edge::MuteRetweets),
                K::PureCoreRetweeter,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::BlockedByAuthor => node(
                Part::Edge(Edge::BlockedBy),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                BestEffort,
            ),
            H::BlockedByReplyRoot => node(
                Part::Edge(Edge::BlockedBy),
                K::PureCoreReplyRoot,
                VIEWER_AUTHOR_EDGES,
                BestEffort,
            ),
            H::SuperFollowsExclusive => node(
                Part::Edge(Edge::SuperFollows),
                K::ExclusiveConversationAuthor,
                "exclusive_super_follow",
                MustHave,
            ),
            H::RootFollowsViewer => node(
                Part::Edge(Edge::FollowedBy),
                K::ControlRoot(ControlArms::CommunityOrMyNetwork),
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::RootFollowsViewerSecondDegree => node(
                Part::Edge(Edge::SecondDegree),
                K::MyNetworkRootNotFollowingViewer,
                "conversation_root_author_second_degree",
                Absent(Tier::BestEffort),
            ),
            H::SuperFollowsRoot => node(
                Part::Edge(Edge::SuperFollows),
                K::ControlRoot(ControlArms::Subscribers),
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::ViewerFollowsRoot => node(
                Part::Edge(Edge::Follows),
                K::ControlRoot(ControlArms::Followers),
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::BlockedByConversationRoot => node(
                Part::Edge(Edge::BlockedBy),
                K::ConversationControlBlockAuthor,
                CONVERSATION_ROOT_AUTHOR_EDGES,
                BestEffort,
            ),
            H::ConversationRootPost => node(
                Part::Column(S::TesPureCoreData),
                K::RelatedTweet(Relation::ConversationRoot),
                "conversation_root_pure_core_data",
                BestEffort,
            ),
            H::BlockedByConversationRootPost => node(
                Part::Edge(Edge::BlockedBy),
                K::ConversationRootPostAuthor,
                "conversation_root_author_blocked_by",
                BestEffort,
            ),
            H::ViewerCountry => node(
                Part::Column(S::ViewerCountry),
                K::ViewerForCoAllowedList,
                "viewer_country",
                BestEffort,
            ),
            H::CommunityModeration => node(
                Part::Column(S::CommunityModeration),
                K::CommunityPost,
                "community_moderation",
                MustHave,
            ),
            H::CommunityModerator => node(
                Part::Column(S::CommunityModerator),
                K::ModeratedCommunity,
                "community_moderator",
                MustHave,
            ),
            H::CommunityViewerRemoved => node(
                Part::Column(S::CommunityViewerRemoved),
                K::TweetCommunity,
                "community_viewer_removed",
                BestEffort,
            ),
            H::ArticleLifecycle => node(
                Part::Column(S::ArticleLifecycle),
                K::TweetArticle,
                "article_lifecycle",
                MustHave,
            ),
            H::ArticleMediaEntities => node(
                Part::Column(S::ArticleMediaEntities),
                K::TweetArticle,
                "article_media_entities",
                Absent(Tier::MustHave),
            ),
            H::ArticleCoverMedia => node(
                Part::Column(S::ArticleCoverMedia),
                K::TweetArticle,
                "article_cover_media",
                Absent(Tier::MustHave),
            ),
            H::TrustedFriends => node(
                Part::Edge(Edge::TrustedFriends),
                K::TrustedFriendsList,
                "trusted_friends_membership",
                Absent(Tier::MustHave),
            ),
            H::OutsideNarrowcastPlace => node(
                Part::Edge(Edge::OutsidePlace),
                K::NarrowcastPlace,
                "viewer_location",
                BestEffort,
            ),
            H::AuthorBlocksOuterAuthor => node(
                Part::Edge(Edge::Blocks),
                K::QuotedAuthorToQuotingAuthor,
                QUOTE_AUTHOR_EDGES,
                MustHave,
            ),
            H::OuterAuthorFollowsAuthor => node(
                Part::Edge(Edge::Follows),
                K::QuotingAuthorToProtectedQuotedAuthor,
                QUOTE_AUTHOR_EDGES,
                BestEffort,
            ),
            H::QuotedTweet => node(
                Part::Column(S::TesQuotedTweet),
                K::OuterOfMissingQuote,
                "quoted_tweet",
                BestEffort,
            ),
            H::ReactedToPostSafetyLabels => node(
                Part::Column(S::SafetyLabels),
                K::RelatedTweet(Relation::ReactedTo),
                "reacted_to_safety_labels",
                BestEffort,
            ),
            H::ReactedToAuthorSafety => node(
                Part::Fields(S::GizmoduckAuthor, &[QueryFields::SAFETY]),
                K::RelatedAuthor(Relation::ReactedTo),
                "reacted_to_author_profile",
                BestEffort,
            ),
            H::ViewerNotificationFilters => node(
                Part::Fields(S::GizmoduckViewer, &[QueryFields::MUTE_SETTINGS]),
                K::Viewer,
                "viewer_profile",
                MustHave,
            ),
            H::AuthorProfile => node(
                Part::Fields(S::GizmoduckActor, &[QueryFields::PROFILE]),
                K::RequestUsers,
                ACTORS,
                MustHave,
            ),
            H::AuthorFollowsViewer => node(
                Part::Edge(Edge::FollowedBy),
                K::PureCoreAuthor,
                VIEWER_AUTHOR_EDGES,
                MustHave,
            ),
            H::AuthorConfirmedEmail => node(
                Part::Column(S::ActorEmails),
                K::UserIfViewerFiltersNoConfirmedEmail,
                "actor_emails",
                MustHave,
            ),
            H::AuthorVerifiedPhone => node(
                Part::Column(S::ActorPhones),
                K::UserIfViewerFiltersNoConfirmedPhone,
                "actor_phones",
                MustHave,
            ),
            H::TweetState => node(
                Part::Column(S::TesTweetState),
                K::MissingRequestTweet,
                "tweet_state",
                Absent(Tier::BestEffort),
            ),
        }
    }

    pub(super) const fn spec_at(self, rows: RequestRows) -> NodeSpec {
        let spec = NodeSpec {
            key: self.spec().key.at(rows),
            ..self.spec()
        };
        match (rows, spec.part, spec.fetcher().owner) {
            (
                RequestRows::Users | RequestRows::ActorPosts,
                Part::Fields(Source::GizmoduckAuthor, fields),
                Owner::Request,
            ) => NodeSpec {
                part: Part::Fields(Source::GizmoduckActor, fields),
                call: ACTORS,
                ..spec
            },
            (RequestRows::Posts | RequestRows::Users | RequestRows::ActorPosts, _, _) => spec,
        }
    }

    pub(crate) const fn is_best_effort(self) -> bool {
        matches!(self.spec().declared.tier(), Tier::BestEffort)
    }

    pub(super) const fn input(self) -> Option<Hydrator> {
        self.input_at(RequestRows::Posts)
    }

    pub(crate) const fn relation(self) -> Option<Relation> {
        self.spec().key.relation()
    }

    pub(super) const fn key_at(self, rows: RequestRows) -> KeyOrigin {
        self.spec_at(rows).key
    }

    pub(super) const fn input_at(self, rows: RequestRows) -> Option<Hydrator> {
        let input = self.key_at(rows).input();
        match (input, self.relation()) {
            (None, Some(relation)) => relation.spec().input(),
            _ => input,
        }
    }

    pub(super) const fn precedes_author(self, rows: RequestRows) -> bool {
        matches!(self.input_at(rows), None | Some(Hydrator::Tweet))
    }

    pub(super) const fn reads_at(self, rows: RequestRows) -> Hydrators {
        let reads = self.key_at(rows).reads();
        match self.relation() {
            Some(relation) => reads.union(relation.spec().reads()),
            None => reads,
        }
    }

    const fn closed_over_at(self, rows: RequestRows) -> Hydrators {
        let inputs = match self.input_at(rows) {
            Some(input) => Hydrators::of(input),
            None => Hydrators::empty(),
        };
        match self.relation() {
            Some(_) => inputs.union(self.reads_at(rows)),
            None => inputs,
        }
    }

    pub(super) const fn edge(self) -> Option<Edge> {
        match self.spec().part {
            Part::Edge(edge) => Some(edge),
            Part::Column(_) | Part::Fields(..) => None,
        }
    }

    pub(super) const fn is_quote_pair_edge(self) -> bool {
        self.is_edge() && self.spec().key.is_quote_pair()
    }

    pub(crate) const fn is_viewer_edge(self) -> bool {
        self.spec().is_viewer_edge()
    }

    pub(super) const fn is_edge(self) -> bool {
        self.edge().is_some()
    }

    pub(crate) const fn needs_viewer(self) -> bool {
        let spec = self.spec();
        let joins_for_viewer = match self.relation() {
            Some(relation) => relation.spec().needs_viewer(),
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

const NEEDS_VIEWER: Hydrators = {
    let mut needs_viewer = Hydrators::empty();
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        if node.needs_viewer() {
            needs_viewer = needs_viewer.with(*node);
        }
        rest = tail;
    }
    needs_viewer
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

    pub(crate) const fn without_viewer(self) -> Self {
        self.difference(NEEDS_VIEWER)
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

const fn stakes_fit_their_nodes() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        let spec = node.spec();
        let derives_exists = matches!(spec.stake(), Stake::Exists)
            || matches!(node.spec_at(RequestRows::Users).stake(), Stake::Exists)
            || matches!(node.spec_at(RequestRows::ActorPosts).stake(), Stake::Exists);
        let fits = !matches!(spec.declared, Stake::Exists)
            && (!derives_exists || matches!(spec.declared, Stake::MustHave))
            && (node.relation().is_none() || !derives_exists);
        if !fits {
            return false;
        }
        rest = tail;
    }
    true
}

const _: () = assert!(inputs_precede_nodes());
const _: () = assert!(stakes_fit_their_nodes());
const _: () = assert!(best_effort_inputs_key_only_best_effort_nodes());
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
