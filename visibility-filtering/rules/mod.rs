mod action_kind;
mod author_rules;
#[cfg(test)]
mod catalog;
mod context;
mod defaults;
pub mod engine;
mod expansion;
#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
mod golden_corpus;
mod levels;
pub(crate) mod notification_kind;
mod rule_spec;
mod safety_level;
mod tweet_rules;

#[cfg(test)]
use crate::models::{Placement, PostCandidate, ViewerFeatures};
#[cfg(test)]
use crate::params::CountryLists;
pub(crate) use action_kind::ActionKind;
use context::RuleContext;
pub use defaults::Effect;
pub use engine::RuleEngine;
pub(crate) use expansion::Expansion;
pub(crate) use rule_spec::observed::NoTrace;
pub use rule_spec::observed::{NodeKind, NodeTrace, PolicyTrace, RowTrace, Skip, Witness};
pub(crate) use rule_spec::RuleId;
pub use rule_spec::{
    ArticleStageLeaf, AuthorFactsLeaf, AuthorLabelsLeaf, AuthorSafetyLeaf, CommunityModeratorLeaf,
    CommunityViewerRemovedLeaf, CompoundLeaf, Contribution, EdgeLeaf, PostFactsLeaf,
    PostModerationLeaf, Predicate, ReactedToAuthorSafetyLeaf, ReactedToPostSafetyLabelsLeaf,
    RelationshipLeaf, ReplyControlLeaf, RequestLeaf, SourceTweetIdLeaf, Truth, TweetLeaf,
    TweetSafetyLabelsLeaf, ViewerAccountLeaf,
};
pub(crate) use safety_level::{LevelError, LevelSpec, QuoteEntries, RerenderReason, WireLevel};
pub use safety_level::{Rpc, SafetyLevel};

#[cfg(test)]
pub(crate) fn test_context<'a>(
    viewer: &'a ViewerFeatures,
    candidate: &'a PostCandidate,
) -> RuleContext<'a> {
    placed_test_context(viewer, candidate, Placement::Standalone)
}

#[cfg(test)]
pub(crate) fn placed_test_context<'a>(
    viewer: &'a ViewerFeatures,
    candidate: &'a PostCandidate,
    placement: Placement,
) -> RuleContext<'a> {
    use std::sync::LazyLock;

    static COUNTRY_LISTS: LazyLock<CountryLists> = LazyLock::new(CountryLists::starting_at_default);
    RuleContext::new(viewer, candidate, placement, &COUNTRY_LISTS)
}
