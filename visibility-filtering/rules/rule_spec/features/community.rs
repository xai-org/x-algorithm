use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{CommunityModeration, PostCandidate};
use crate::rules::context::At;
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

pub(in crate::rules) enum PostModeration {}

impl Feature for PostModeration {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::CommunityModeration);
    type Value<'a> = CommunityModeration;
}

impl Reads<PostCandidate> for PostModeration {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> CommunityModeration {
        at.post().community_moderation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum PostModerationLeaf {
    TweetCommunityTweetIsHidden,
    TweetCommunityTweetAuthorIsRemoved,
}

impl Leaf for PostModerationLeaf {
    type Feature = PostModeration;

    #[inline]
    fn meaning<'a, M: Meaning<'a, PostModeration>>(self, m: M) -> M::Out {
        match self {
            Self::TweetCommunityTweetIsHidden => m.run(|moderation, _| moderation.is_hidden),
            Self::TweetCommunityTweetAuthorIsRemoved => {
                m.run(|moderation, _| moderation.is_author_removed)
            }
        }
    }
}

impl PostModerationLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::PostModeration(self))
    }
}

pub(in crate::rules) enum CommunityModerator {}

impl Feature for CommunityModerator {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::CommunityModerator);
    type Value<'a> = Option<bool>;
}

impl Reads<PostCandidate> for CommunityModerator {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> Option<bool> {
        at.post().viewer_is_community_moderator
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum CommunityModeratorLeaf {
    ViewerIsCommunityModerator,
}

impl Leaf for CommunityModeratorLeaf {
    type Feature = CommunityModerator;

    #[inline]
    fn meaning<'a, M: Meaning<'a, CommunityModerator>>(self, m: M) -> M::Out {
        match self {
            Self::ViewerIsCommunityModerator => m.run(|is_moderator, facts| {
                facts.viewer_id().is_some() && is_moderator.unwrap_or(true)
            }),
        }
    }
}

impl CommunityModeratorLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::CommunityModerator(self))
    }
}

pub(in crate::rules) enum CommunityViewerRemoved {}

impl Feature for CommunityViewerRemoved {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::CommunityViewerRemoved);
    type Value<'a> = bool;
}

impl Reads<PostCandidate> for CommunityViewerRemoved {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> bool {
        at.post().viewer_is_removed_from_community
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum CommunityViewerRemovedLeaf {
    ViewerIsRemovedFromCommunity,
}

impl Leaf for CommunityViewerRemovedLeaf {
    type Feature = CommunityViewerRemoved;

    #[inline]
    fn meaning<'a, M: Meaning<'a, CommunityViewerRemoved>>(self, m: M) -> M::Out {
        match self {
            Self::ViewerIsRemovedFromCommunity => m.run(|removed, _| removed),
        }
    }
}

impl CommunityViewerRemovedLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::CommunityViewerRemoved(self))
    }
}
