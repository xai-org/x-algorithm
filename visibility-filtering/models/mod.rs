pub mod author;
pub mod conversation_control;
pub mod region;
pub mod safety_labels;
pub mod tweet;
pub mod verdict;
pub mod viewer;

pub use author::{AuthorFeatures, AuthorLabel, AuthorLabelSet, AuthorProfile};
pub use conversation_control::ConversationControlFeatures;
pub use safety_labels::{SafetyLabelMap, SafetyLabelType};
pub use tweet::{
    ArticleLifecycle, CommunityModeration, MediaFeature, NsfwFeature, ReactionContext,
    TweetFeatures,
};
pub(crate) use verdict::VerdictAction;
pub use verdict::{
    Decided, Deletion, DropReason, EmergencyCopy, Evaluation, FosnrReason, LabelInterstitial,
    LimitedActionType, LimitedEngagement, LimitedEngagementReason, MediaInterstitial,
    MediaRestriction, Notice, NsfwViewerDropReason, PublicInterest, TombstoneReason, Verdict,
    Withholding,
};
pub use viewer::{
    ClientCapability, ClientSwitch, ClientSwitchSet, NotificationFilters, VerifyBlurSupport,
    Viewer, ViewerAge, ViewerFeatures, ViewerProfile,
};
pub(crate) use viewer::{RequestFacts, RequestViewer};

use crate::hydration::{Hydrator, Hydrators};
pub(crate) use xai_visibility_filtering::notifications::NotificationKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TweetId(pub u64);

pub fn tweet_timestamp_ms(tweet_id: u64) -> u64 {
    const SNOWFLAKE_EPOCH_MS: u64 = 1288834974657;
    const SNOWFLAKE_TIMESTAMP_SHIFT: u32 = 22;
    (tweet_id >> SNOWFLAKE_TIMESTAMP_SHIFT) + SNOWFLAKE_EPOCH_MS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AuthorId(pub u64);

impl AuthorId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PureCore {
    pub author_id: AuthorId,
    pub source_tweet_id: Option<TweetId>,
    pub source_author_id: Option<AuthorId>,
    pub reply_to: Option<ReplyTo>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplyTo {
    Root {
        conversation_id: TweetId,
        root_author_id: AuthorId,
    },
    Nested {
        conversation_id: TweetId,
        parent_id: TweetId,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct RawCandidate {
    pub tweet_id: TweetId,
    pub request_author_id: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    Standalone,
    InnerQuoted { outer: TweetId },
    RetweetSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuoteLink {
    pub outer: TweetId,
    pub quoting_author: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quote {
    pub link: QuoteLink,
    pub pair_edges: EdgeSet,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RetweetMerge {
    Merged,
    #[default]
    Unmerged,
}

#[derive(Clone, Copy, Debug)]
pub struct PlacedCandidate {
    pub candidate: RawCandidate,
    pub placement: Placement,
}

impl PlacedCandidate {
    pub fn standalone(candidate: RawCandidate) -> Self {
        Self {
            candidate,
            placement: Placement::Standalone,
        }
    }

    pub fn tweet_id(self) -> TweetId {
        self.candidate.tweet_id
    }

    pub fn key(self) -> EvaluationKey {
        EvaluationKey {
            tweet_id: self.tweet_id(),
            placement: self.placement,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationKey {
    pub tweet_id: TweetId,
    pub placement: Placement,
}

impl EvaluationKey {
    pub fn retweet_source(tweet_id: TweetId) -> Self {
        Self {
            tweet_id,
            placement: Placement::RetweetSource,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PostCandidate {
    pub author: UserEntity,
    pub edges: EdgeSet,
    pub post: PostEntity,
    pub failed: Hydrators,
}

#[derive(Clone, Debug)]
pub struct UserCandidate {
    pub user: UserEntity,
    pub edges: EdgeSet,
    pub quote: Option<Quote>,
    pub failed: Hydrators,
}

#[cfg(test)]
impl UserCandidate {
    pub(crate) fn author_of(post: &PostCandidate) -> Self {
        Self {
            user: post.author.clone(),
            edges: post.edges,
            quote: None,
            failed: post.failed,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserEntity {
    pub id: u64,
    pub features: AuthorFeatures,
    pub labels: AuthorLabelSet,
    pub profile: AuthorProfile,
    pub lacks_confirmed_email: bool,
    pub lacks_verified_phone: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeSet(Hydrators);

impl EdgeSet {
    #[inline]
    pub fn contains(self, node: Hydrator) -> bool {
        self.0.contains(node)
    }

    #[inline]
    pub fn with(self, node: Hydrator) -> Self {
        Self(self.0.with(node))
    }
}

#[derive(Clone, Debug, Default)]
pub struct PostEntity {
    pub tweet_id: u64,
    pub source_tweet_id: Option<u64>,
    pub tweet_features: TweetFeatures,
    pub safety_labels: SafetyLabelMap,
    pub conversation_control: Option<ConversationControlFeatures>,
    pub community_moderation: CommunityModeration,
            pub viewer_is_community_moderator: Option<bool>,
    pub viewer_is_removed_from_community: bool,
        pub article_lifecycle: Option<ArticleLifecycle>,
    pub article_has_media_entities: bool,
    pub article_has_cover_media: bool,
    pub reacted_to_post_safety_labels: SafetyLabelMap,
    pub reacted_to_author_is_protected: bool,
    pub conversation_root_author_blocks_viewer: bool,
}
