use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{PostCandidate, SafetyLabelMap, SafetyLabelType};
use crate::rules::context::{At, PostCoreFacts};
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};

pub(in crate::rules) enum PostFacts {}

impl Feature for PostFacts {
    const REQUIRED: Hydrators = Hydrators::empty();
    type Value<'a> = PostCoreFacts<'a>;
}

impl Reads<PostCandidate> for PostFacts {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> PostCoreFacts<'a> {
        at.post_facts()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum PostFactsLeaf {
    TweetCreatedAfter(u64),
}

impl Leaf for PostFactsLeaf {
    type Feature = PostFacts;

    #[inline]
    fn meaning<'a, M: Meaning<'a, PostFacts>>(self, m: M) -> M::Out {
        match self {
            Self::TweetCreatedAfter(unix_ms) => {
                m.run_with(unix_ms, |post, _| post.created_after(unix_ms))
            }
        }
    }
}

impl PostFactsLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::PostFacts(self))
    }
}

pub(in crate::rules) enum SourceTweetId {}

impl Feature for SourceTweetId {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::PureCore);
    type Value<'a> = Option<u64>;
}

impl Reads<PostCandidate> for SourceTweetId {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> Option<u64> {
        at.post().source_tweet_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum SourceTweetIdLeaf {
    TweetIsRetweet,
}

impl Leaf for SourceTweetIdLeaf {
    type Feature = SourceTweetId;

    #[inline]
    fn meaning<'a, M: Meaning<'a, SourceTweetId>>(self, m: M) -> M::Out {
        match self {
            Self::TweetIsRetweet => m.run(|source, _| source.is_some()),
        }
    }
}

impl SourceTweetIdLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::SourceTweetId(self))
    }
}

pub(in crate::rules) enum TweetSafetyLabels {}

impl Feature for TweetSafetyLabels {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::TweetSafetyLabels);
    type Value<'a> = &'a SafetyLabelMap;
}

impl Reads<PostCandidate> for TweetSafetyLabels {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> &'a SafetyLabelMap {
        &at.post().safety_labels
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum TweetSafetyLabelsLeaf {
    TweetHasLabel(SafetyLabelType),
}

impl Leaf for TweetSafetyLabelsLeaf {
    type Feature = TweetSafetyLabels;

    #[inline]
    fn meaning<'a, M: Meaning<'a, TweetSafetyLabels>>(self, m: M) -> M::Out {
        match self {
            Self::TweetHasLabel(label) => m.run_with(label, |labels, _| labels.has_label(label)),
        }
    }
}

impl TweetSafetyLabelsLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::TweetSafetyLabels(self))
    }
}

pub(in crate::rules) enum ReplyControl {}

impl Feature for ReplyControl {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ConversationControl);
    type Value<'a> = Option<&'a ConversationControl>;
}

impl Reads<PostCandidate> for ReplyControl {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> Option<&'a ConversationControl> {
        let features = at.post().conversation_control.as_ref();
        features.map(|features| &features.control)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ReplyControlLeaf {
    TweetHasConversationControl(ConversationControlArm),
    ViewerIsConversationRootAuthor,
    ViewerIsInvitedToConversation,
}

impl Leaf for ReplyControlLeaf {
    type Feature = ReplyControl;

    #[inline]
    fn meaning<'a, M: Meaning<'a, ReplyControl>>(self, m: M) -> M::Out {
        match self {
            Self::TweetHasConversationControl(arm) => m.run_with(arm, |control, _| {
                control.is_some_and(|control| control.arm == arm)
            }),
            Self::ViewerIsConversationRootAuthor => m.run(|control, facts| {
                control
                    .zip(facts.viewer_id())
                    .is_some_and(|(control, viewer_id)| {
                        viewer_id == control.conversation_tweet_author_id
                    })
            }),
            Self::ViewerIsInvitedToConversation => m.run(|control, facts| {
                control
                    .zip(facts.viewer_id())
                    .is_some_and(|(control, viewer_id)| {
                        control.invited_user_ids.contains(&viewer_id)
                    })
            }),
        }
    }
}

impl ReplyControlLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::ReplyControl(self))
    }
}
