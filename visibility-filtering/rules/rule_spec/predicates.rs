use super::features::Feature;
#[cfg(test)]
use super::features::Spelling;
use super::features::{
    truth, ArticleStageLeaf, AuthorEmailLeaf, AuthorFactsLeaf, AuthorLabelsLeaf, AuthorPhoneLeaf,
    AuthorProfileLeaf, AuthorSafetyLeaf, CommunityModeratorLeaf, CommunityViewerRemovedLeaf,
    CompoundLeaf, EvaluateFeature, Leaf, NotificationFilterLeaf, PairEdgeLeaf, PostFactsLeaf,
    PostModerationLeaf, ReactedToAuthorSafetyLeaf, ReactedToPostSafetyLabelsLeaf, Reads,
    RelationshipLeaf, ReplyControlLeaf, RequestLeaf, SourceTweetIdLeaf, TweetLeaf,
    TweetSafetyLabelsLeaf, ViewerAccountLeaf, ViewerLabelLeaf,
};
use super::*;
use crate::rules::rule_spec::engine::{Engine, Fold, PostEngine};

#[cfg(test)]
pub(in crate::rules) fn spelled_label(label: SafetyLabelType) -> String {
    label_meta(label).map_or_else(|| format!("{label:?}"), |meta| meta.ident.to_owned())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate<E: Engine = PostEngine> {
    Request(RequestLeaf),
    PostFacts(E::PostOnly<PostFactsLeaf>),
    SourceTweetId(E::PostOnly<SourceTweetIdLeaf>),
    Tweet(E::PostOnly<TweetLeaf>),
    TweetSafetyLabels(E::PostOnly<TweetSafetyLabelsLeaf>),
    ReplyControl(E::PostOnly<ReplyControlLeaf>),
    PostModeration(E::PostOnly<PostModerationLeaf>),
    ArticleStage(E::PostOnly<ArticleStageLeaf>),
    CommunityModerator(E::PostOnly<CommunityModeratorLeaf>),
    CommunityViewerRemoved(E::PostOnly<CommunityViewerRemovedLeaf>),
    ReactedToPostSafetyLabels(E::PostOnly<ReactedToPostSafetyLabelsLeaf>),
    ReactedToAuthorSafety(E::PostOnly<ReactedToAuthorSafetyLeaf>),
    ViewerAccount(ViewerAccountLeaf),
    ViewerLabel(ViewerLabelLeaf),
    NotificationFilter(NotificationFilterLeaf),
    AuthorFacts(AuthorFactsLeaf),
    AuthorSafety(AuthorSafetyLeaf),
    AuthorLabels(AuthorLabelsLeaf),
    AuthorProfile(AuthorProfileLeaf),
    AuthorEmail(AuthorEmailLeaf),
    AuthorPhone(AuthorPhoneLeaf),
    Relationship(RelationshipLeaf),
    PairEdge(PairEdgeLeaf),
    Compound(E::PostOnly<CompoundLeaf>),
}

trait PredicateVisitor<E: Engine> {
    type Out;
    fn leaf<L: Leaf>(self, leaf: L) -> Self::Out
    where
        L::Feature: Reads<E::Candidate>;
    fn post<L: Leaf>(self, leaf: E::PostOnly<L>) -> Self::Out;
    fn relationship(self, leaf: RelationshipLeaf) -> Self::Out;
    fn pair_edge(self, leaf: PairEdgeLeaf) -> Self::Out;
    fn compound(self, leaf: E::PostOnly<CompoundLeaf>) -> Self::Out;
}

impl<E: Engine> Predicate<E> {
    fn visit<V: PredicateVisitor<E>>(self, visitor: V) -> V::Out {
        match self {
            Self::Request(leaf) => visitor.leaf(leaf),
            Self::PostFacts(leaf) => visitor.post(leaf),
            Self::SourceTweetId(leaf) => visitor.post(leaf),
            Self::Tweet(leaf) => visitor.post(leaf),
            Self::TweetSafetyLabels(leaf) => visitor.post(leaf),
            Self::ReplyControl(leaf) => visitor.post(leaf),
            Self::PostModeration(leaf) => visitor.post(leaf),
            Self::ArticleStage(leaf) => visitor.post(leaf),
            Self::CommunityModerator(leaf) => visitor.post(leaf),
            Self::CommunityViewerRemoved(leaf) => visitor.post(leaf),
            Self::ReactedToPostSafetyLabels(leaf) => visitor.post(leaf),
            Self::ReactedToAuthorSafety(leaf) => visitor.post(leaf),
            Self::ViewerAccount(leaf) => visitor.leaf(leaf),
            Self::ViewerLabel(leaf) => visitor.leaf(leaf),
            Self::NotificationFilter(leaf) => visitor.leaf(leaf),
            Self::AuthorFacts(leaf) => visitor.leaf(leaf),
            Self::AuthorSafety(leaf) => visitor.leaf(leaf),
            Self::AuthorLabels(leaf) => visitor.leaf(leaf),
            Self::AuthorProfile(leaf) => visitor.leaf(leaf),
            Self::AuthorEmail(leaf) => visitor.leaf(leaf),
            Self::AuthorPhone(leaf) => visitor.leaf(leaf),
            Self::Relationship(leaf) => visitor.relationship(leaf),
            Self::PairEdge(leaf) => visitor.pair_edge(leaf),
            Self::Compound(leaf) => visitor.compound(leaf),
        }
    }

    pub(in crate::rules) fn requirements(self) -> Hydrators {
        self.visit(Requirements)
    }

    pub(in crate::rules) fn lift(self) -> Predicate {
        match self {
            Self::Request(leaf) => Predicate::Request(leaf),
            Self::PostFacts(leaf) => Predicate::PostFacts(E::open(leaf)),
            Self::SourceTweetId(leaf) => Predicate::SourceTweetId(E::open(leaf)),
            Self::Tweet(leaf) => Predicate::Tweet(E::open(leaf)),
            Self::TweetSafetyLabels(leaf) => Predicate::TweetSafetyLabels(E::open(leaf)),
            Self::ReplyControl(leaf) => Predicate::ReplyControl(E::open(leaf)),
            Self::PostModeration(leaf) => Predicate::PostModeration(E::open(leaf)),
            Self::ArticleStage(leaf) => Predicate::ArticleStage(E::open(leaf)),
            Self::CommunityModerator(leaf) => Predicate::CommunityModerator(E::open(leaf)),
            Self::CommunityViewerRemoved(leaf) => Predicate::CommunityViewerRemoved(E::open(leaf)),
            Self::ReactedToPostSafetyLabels(leaf) => {
                Predicate::ReactedToPostSafetyLabels(E::open(leaf))
            }
            Self::ReactedToAuthorSafety(leaf) => Predicate::ReactedToAuthorSafety(E::open(leaf)),
            Self::ViewerAccount(leaf) => Predicate::ViewerAccount(leaf),
            Self::ViewerLabel(leaf) => Predicate::ViewerLabel(leaf),
            Self::NotificationFilter(leaf) => Predicate::NotificationFilter(leaf),
            Self::AuthorFacts(leaf) => Predicate::AuthorFacts(leaf),
            Self::AuthorSafety(leaf) => Predicate::AuthorSafety(leaf),
            Self::AuthorLabels(leaf) => Predicate::AuthorLabels(leaf),
            Self::AuthorProfile(leaf) => Predicate::AuthorProfile(leaf),
            Self::AuthorEmail(leaf) => Predicate::AuthorEmail(leaf),
            Self::AuthorPhone(leaf) => Predicate::AuthorPhone(leaf),
            Self::Relationship(leaf) => Predicate::Relationship(leaf),
            Self::PairEdge(leaf) => Predicate::PairEdge(leaf),
            Self::Compound(leaf) => Predicate::Compound(E::open(leaf)),
        }
    }
}

pub(in crate::rules) fn truth_of<E: Fold>(
    predicate: Predicate<E>,
    context: &RuleContext<'_, E::Candidate>,
) -> Truth {
    predicate.visit(TruthOf(context))
}

impl Predicate {
    #[cfg(test)]
    pub(in crate::rules) fn spelled(self) -> String {
        self.visit(Spelled)
    }

    #[cfg(test)]
    pub(super) fn holds(self, context: &RuleContext<'_>) -> bool {
        self.truth(context).resolves_true()
    }

    #[cfg(test)]
    pub(in crate::rules) fn truth(self, context: &RuleContext<'_>) -> Truth {
        truth_of(self, context)
    }
}

#[cfg(test)]
struct Spelled;

#[cfg(test)]
impl PredicateVisitor<PostEngine> for Spelled {
    type Out = String;

    fn leaf<L: Leaf>(self, leaf: L) -> String {
        let name: &str = leaf.into();
        match leaf.meaning(Spelling) {
            Some(payload) => format!("{name}({payload})"),
            None => name.to_owned(),
        }
    }

    fn post<L: Leaf>(self, leaf: L) -> String {
        self.leaf(leaf)
    }

    fn relationship(self, leaf: RelationshipLeaf) -> String {
        <&str>::from(leaf).to_owned()
    }

    fn pair_edge(self, leaf: PairEdgeLeaf) -> String {
        <&str>::from(leaf).to_owned()
    }

    fn compound(self, leaf: CompoundLeaf) -> String {
        <&str>::from(leaf).to_owned()
    }
}

struct Requirements;

impl<E: Engine> PredicateVisitor<E> for Requirements {
    type Out = Hydrators;

    fn leaf<L: Leaf>(self, _: L) -> Hydrators {
        <L::Feature as Feature>::REQUIRED
    }

    fn post<L: Leaf>(self, _: E::PostOnly<L>) -> Hydrators {
        <L::Feature as Feature>::REQUIRED
    }

    fn relationship(self, leaf: RelationshipLeaf) -> Hydrators {
        Hydrators::of(leaf.node())
    }

    fn pair_edge(self, leaf: PairEdgeLeaf) -> Hydrators {
        Hydrators::of(leaf.node())
    }

    fn compound(self, leaf: E::PostOnly<CompoundLeaf>) -> Hydrators {
        E::open(leaf).interpret(&inputs::Required)
    }
}

struct TruthOf<'c, 'a, E: Engine>(&'c RuleContext<'a, E::Candidate>);

impl<E: Fold> PredicateVisitor<E> for TruthOf<'_, '_, E> {
    type Out = Truth;

    #[inline]
    fn leaf<L: Leaf>(self, leaf: L) -> Truth
    where
        L::Feature: Reads<E::Candidate>,
    {
        leaf.meaning(EvaluateFeature(self.0))
    }

    #[inline]
    fn post<L: Leaf>(self, leaf: E::PostOnly<L>) -> Truth {
        let (leaf, context) = E::post(self.0, leaf);
        leaf.meaning(EvaluateFeature(context))
    }

    #[inline]
    fn relationship(self, leaf: RelationshipLeaf) -> Truth {
        let node = leaf.node();
        truth(
            self.0.acquire_edge(node),
            self.0.failed().intersection(Hydrators::of(node)),
        )
    }

    #[inline]
    fn pair_edge(self, leaf: PairEdgeLeaf) -> Truth {
        let node = leaf.node();
        truth(
            self.0.acquire_pair_edge(node),
            self.0.failed().intersection(Hydrators::of(node)),
        )
    }

    #[inline]
    fn compound(self, leaf: E::PostOnly<CompoundLeaf>) -> Truth {
        let (leaf, context) = E::post(self.0, leaf);
        let inputs::Acquired { value, failed } = leaf.interpret(&inputs::EvaluateInputs(context));
        truth(value, failed)
    }
}
