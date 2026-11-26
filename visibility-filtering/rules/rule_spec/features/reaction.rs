use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{PostCandidate, SafetyLabelMap, SafetyLabelType};
use crate::rules::context::At;
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

pub(in crate::rules) enum ReactedToPostSafetyLabels {}

impl Feature for ReactedToPostSafetyLabels {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ReactedToPostSafetyLabels);
    type Value<'a> = &'a SafetyLabelMap;
}

impl Reads<PostCandidate> for ReactedToPostSafetyLabels {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> &'a SafetyLabelMap {
        &at.post().reacted_to_post_safety_labels
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ReactedToPostSafetyLabelsLeaf {
    ReactedToPostHasLabel(SafetyLabelType),
}

impl Leaf for ReactedToPostSafetyLabelsLeaf {
    type Feature = ReactedToPostSafetyLabels;

    #[inline]
    fn meaning<'a, M: Meaning<'a, ReactedToPostSafetyLabels>>(self, m: M) -> M::Out {
        match self {
            Self::ReactedToPostHasLabel(label) => {
                m.run_with(label, |labels, _| labels.has_label(label))
            }
        }
    }
}

impl ReactedToPostSafetyLabelsLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::ReactedToPostSafetyLabels(self))
    }
}

pub(in crate::rules) enum ReactedToAuthorSafety {}

impl Feature for ReactedToAuthorSafety {
    const REQUIRED: Hydrators = Hydrators::of(Hydrator::ReactedToAuthorSafety);
    type Value<'a> = bool;
}

impl Reads<PostCandidate> for ReactedToAuthorSafety {
    #[inline]
    fn read<'a>(at: At<'_, 'a>) -> bool {
        at.post().reacted_to_author_is_protected
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ReactedToAuthorSafetyLeaf {
    ReactedToAuthorIsProtected,
}

impl Leaf for ReactedToAuthorSafetyLeaf {
    type Feature = ReactedToAuthorSafety;

    #[inline]
    fn meaning<'a, M: Meaning<'a, ReactedToAuthorSafety>>(self, m: M) -> M::Out {
        match self {
            Self::ReactedToAuthorIsProtected => m.run(|is_protected, _| is_protected),
        }
    }
}

impl ReactedToAuthorSafetyLeaf {
    pub(in crate::rules) const fn leaf(self) -> RuleCond {
        RuleCond::Leaf(Predicate::ReactedToAuthorSafety(self))
    }
}
