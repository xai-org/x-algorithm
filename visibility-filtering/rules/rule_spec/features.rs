use super::inputs::Inputs;
#[cfg(test)]
use super::spelled_label;
use super::Truth;
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{AuthorLabel, ClientSwitch, PostCandidate, SafetyLabelType, VerifyBlurSupport};
use crate::params::CountryList;
use crate::rules::context::{At, CoreFacts};
use crate::rules::rule_spec::engine::Candidate;
use crate::rules::RuleContext;
use std::fmt::Debug;
use strum::VariantArray;
use xai_core_entities::entities::ConversationControlArm;

mod article;
mod author;
mod community;
mod compound;
mod edges;
mod post;
mod reaction;
mod request;
mod tweet;
mod viewer;

pub use article::ArticleStageLeaf;
pub use author::{
    AuthorEmailLeaf, AuthorFactsLeaf, AuthorLabelsLeaf, AuthorPhoneLeaf, AuthorProfileLeaf,
    AuthorSafetyLeaf,
};
pub use community::{CommunityModeratorLeaf, CommunityViewerRemovedLeaf, PostModerationLeaf};
pub use compound::CompoundLeaf;
pub use edges::{PairEdgeLeaf, RelationshipLeaf};
pub use post::{PostFactsLeaf, ReplyControlLeaf, SourceTweetIdLeaf, TweetSafetyLabelsLeaf};
pub use reaction::{ReactedToAuthorSafetyLeaf, ReactedToPostSafetyLabelsLeaf};
pub use request::RequestLeaf;
pub use tweet::TweetLeaf;
pub use viewer::{NotificationFilterLeaf, ViewerAccountLeaf, ViewerLabelLeaf};

pub(in crate::rules) use post::TweetSafetyLabels;

pub(in crate::rules) mod leaves {
    pub(in crate::rules) use super::ArticleStageLeaf::*;
    pub(in crate::rules) use super::AuthorEmailLeaf::*;
    pub(in crate::rules) use super::AuthorFactsLeaf::*;
    pub(in crate::rules) use super::AuthorLabelsLeaf::*;
    pub(in crate::rules) use super::AuthorPhoneLeaf::*;
    pub(in crate::rules) use super::AuthorProfileLeaf::*;
    pub(in crate::rules) use super::AuthorSafetyLeaf::*;
    pub(in crate::rules) use super::CommunityModeratorLeaf::*;
    pub(in crate::rules) use super::CommunityViewerRemovedLeaf::*;
    pub(in crate::rules) use super::CompoundLeaf::*;
    pub(in crate::rules) use super::NotificationFilterLeaf::*;
    pub(in crate::rules) use super::PairEdgeLeaf::*;
    pub(in crate::rules) use super::PostFactsLeaf::*;
    pub(in crate::rules) use super::PostModerationLeaf::*;
    pub(in crate::rules) use super::ReactedToAuthorSafetyLeaf::*;
    pub(in crate::rules) use super::ReactedToPostSafetyLabelsLeaf::*;
    pub(in crate::rules) use super::RelationshipLeaf::*;
    pub(in crate::rules) use super::ReplyControlLeaf::*;
    pub(in crate::rules) use super::RequestLeaf::*;
    pub(in crate::rules) use super::SourceTweetIdLeaf::*;
    pub(in crate::rules) use super::TweetLeaf::*;
    pub(in crate::rules) use super::TweetSafetyLabelsLeaf::*;
    pub(in crate::rules) use super::ViewerAccountLeaf::*;
    pub(in crate::rules) use super::ViewerLabelLeaf::*;
}

pub(in crate::rules) trait Feature {
    const REQUIRED: Hydrators;
    type Value<'a>: Copy;
}

pub(in crate::rules) trait Reads<C>: Feature {
    fn read<'a>(at: At<'_, 'a, C>) -> Self::Value<'a>;
}

pub(in crate::rules) trait Leaf:
    Copy + Debug + Eq + Into<&'static str> + 'static
{
    type Feature: Reads<PostCandidate>;
    fn meaning<'a, M: Meaning<'a, Self::Feature>>(self, meaning: M) -> M::Out;
}

pub(in crate::rules) trait Meaning<'a, F: Feature> {
    type Out;
    fn run(self, kernel: impl FnOnce(F::Value<'a>, CoreFacts<'a>) -> bool) -> Self::Out;
    fn run_with(
        self,
        payload: impl Payload,
        kernel: impl FnOnce(F::Value<'a>, CoreFacts<'a>) -> bool,
    ) -> Self::Out;
}

pub(in crate::rules) trait Payload {
    #[cfg(test)]
    fn spelled(self) -> String;
}

impl Payload for SafetyLabelType {
    #[cfg(test)]
    fn spelled(self) -> String {
        spelled_label(self)
    }
}

impl Payload for u64 {
    #[cfg(test)]
    fn spelled(self) -> String {
        self.to_string()
    }
}

impl Payload for &'static str {
    #[cfg(test)]
    fn spelled(self) -> String {
        self.to_owned()
    }
}

impl Payload for AuthorLabel {
    #[cfg(test)]
    fn spelled(self) -> String {
        format!("{self:?}")
    }
}

impl Payload for CountryList {
    #[cfg(test)]
    fn spelled(self) -> String {
        format!("{self:?}")
    }
}

impl Payload for ConversationControlArm {
    #[cfg(test)]
    fn spelled(self) -> String {
        format!("{self:?}")
    }
}

impl Payload for ClientSwitch {
    #[cfg(test)]
    fn spelled(self) -> String {
        <&str>::from(self).to_owned()
    }
}

impl Payload for VerifyBlurSupport {
    #[cfg(test)]
    fn spelled(self) -> String {
        format!("{self:?}")
    }
}

#[cfg(test)]
pub(super) struct Spelling;

#[cfg(test)]
impl<'a, F: Feature> Meaning<'a, F> for Spelling {
    type Out = Option<String>;

    fn run(self, _: impl FnOnce(F::Value<'a>, CoreFacts<'a>) -> bool) -> Option<String> {
        None
    }

    fn run_with(
        self,
        payload: impl Payload,
        _: impl FnOnce(F::Value<'a>, CoreFacts<'a>) -> bool,
    ) -> Option<String> {
        Some(payload.spelled())
    }
}

pub(super) struct EvaluateFeature<'c, 'a, C = PostCandidate>(pub(super) &'c RuleContext<'a, C>);

impl<'a, C: Candidate, F: Reads<C>> Meaning<'a, F> for EvaluateFeature<'_, 'a, C> {
    type Out = Truth;

    #[inline]
    fn run(self, kernel: impl FnOnce(F::Value<'a>, CoreFacts<'a>) -> bool) -> Truth {
        let context = self.0;
        truth(
            kernel(context.acquire::<F>(), context.facts()),
            context.failed().intersection(F::REQUIRED),
        )
    }

    #[inline]
    fn run_with(
        self,
        _: impl Payload,
        kernel: impl FnOnce(F::Value<'a>, CoreFacts<'a>) -> bool,
    ) -> Truth {
        Meaning::<F>::run(self, kernel)
    }
}

#[inline]
pub(super) fn truth(value: bool, failed: Hydrators) -> Truth {
    match (failed.is_empty(), value) {
        (false, default) => Truth::Unknown { default, failed },
        (true, true) => Truth::True,
        (true, false) => Truth::False,
    }
}

pub(in crate::rules) enum EdgeFeature<const N: u8> {}

impl<const N: u8> EdgeFeature<N> {
    #[expect(
        clippy::indexing_slicing,
        reason = "N is a Hydrator discriminant; an out-of-range N fails const evaluation"
    )]
    const NODE: Hydrator = <Hydrator as VariantArray>::VARIANTS[N as usize];
}

impl<const N: u8> Feature for EdgeFeature<N> {
    const REQUIRED: Hydrators = {
        assert!(
            Self::NODE.is_viewer_edge(),
            "EdgeFeature<N> names a node that is not a viewer edge"
        );
        Hydrators::of(Self::NODE)
    };
    type Value<'a> = bool;
}

impl<const N: u8> Reads<PostCandidate> for EdgeFeature<N> {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> bool {
        at.edge(Self::NODE)
    }
}

pub(in crate::rules) struct In<'a, F: Feature>(pub(in crate::rules) F::Value<'a>);

impl<'a, F: Reads<PostCandidate>> Inputs<'a> for In<'a, F> {
    const REQUIRED: Hydrators = F::REQUIRED;

    #[inline]
    fn acquire(context: &RuleContext<'a>) -> Self {
        Self(context.acquire::<F>())
    }
}
