use crate::hydration::Hydrator;
use crate::rules::rule_spec::engine::{Engine, UserEngine};
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum RelationshipLeaf {
    ViewerFollowsAuthor,
    ViewerBlocksAuthor,
    ViewerMutesAuthor,
    ViewerIsBlockedByAuthor,
    AuthorFollowsViewer,
}

impl RelationshipLeaf {
    #[inline]
    pub(in crate::rules) const fn node(self) -> Hydrator {
        match self {
            Self::ViewerFollowsAuthor => Hydrator::Follows,
            Self::ViewerBlocksAuthor => Hydrator::Blocks,
            Self::ViewerMutesAuthor => Hydrator::Mutes,
            Self::ViewerIsBlockedByAuthor => Hydrator::BlockedByAuthor,
            Self::AuthorFollowsViewer => Hydrator::AuthorFollowsViewer,
        }
    }

    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::Relationship(self))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum PairEdgeLeaf {
    AuthorBlocksOuterAuthor,
    OuterAuthorFollowsAuthor,
}

impl PairEdgeLeaf {
    #[inline]
    pub(in crate::rules) const fn node(self) -> Hydrator {
        match self {
            Self::AuthorBlocksOuterAuthor => Hydrator::AuthorBlocksOuterAuthor,
            Self::OuterAuthorFollowsAuthor => Hydrator::OuterAuthorFollowsAuthor,
        }
    }

    pub(in crate::rules) const fn leaf(self) -> RuleCond<UserEngine> {
        RuleCond::Leaf(Predicate::PairEdge(self))
    }
}
