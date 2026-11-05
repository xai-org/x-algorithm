use crate::hydration::{Hydrators, Lookup};
use crate::rules::{ActionKind, RuleId};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::{AppealablePolicy, InterstitialAction, InterstitialReason};
use xai_x_thrift::safety_result::SafetyResultReason;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum VerdictAction {
    Allow,
    Drop,
    Tombstone,
    Interstitial,
    LimitedEngagement,
    TweetInterstitial,
    SoftIntervention,
    Appealable,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Withheld(Decided<Withholding>),
    Fosnr(Decided<Notice>),
    Shown {
        label_interstitial: Option<Decided<LabelInterstitial>>,
        media: Option<Decided<MediaRestriction>>,
        engagement: Option<Decided<LimitedEngagement>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Notice {
    SoftIntervention(FosnrReason),
    Appealable {
        reason: FosnrReason,
        limited_actions: &'static [LimitedActionType],
    },
}

impl Notice {
    pub(crate) const fn kind(&self) -> ActionKind {
        match self {
            Self::SoftIntervention(_) => ActionKind::SoftIntervention,
            Self::Appealable { .. } => ActionKind::Appealable,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum LimitedActionType {
    Reply,
    Retweet,
    QuoteTweet,
    Like,
    React,
    SendViaDm,
    AddToBookmarks,
    AddToMoment,
    PinToProfile,
    ViewTweetActivity,
    ShareTweetVia,
    Follow,
    ListsAddRemove,
    MuteConversation,
    Embed,
    ViewHiddenReplies,
    HideCommunityTweet,
    CopyLink,
    VoteOnPoll,
    RemoveFromCommunity,
    ShowRetweetActionMenu,
    ReplyDownVote,
    Autoplay,
    EditTweet,
    Highlight,
    ViewPostEngagements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabelInterstitial {
    PublicInterest(PublicInterest),
    EmergencyDynamic(Box<EmergencyCopy>),
}

impl LabelInterstitial {
    pub fn limited_engagement_reason(&self) -> LimitedEngagementReason {
        match self {
            Self::PublicInterest(notice) => notice
                .limit
                .unwrap_or(LimitedEngagementReason::NonCompliant),
            Self::EmergencyDynamic(_) => LimitedEngagementReason::NonCompliant,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicInterest {
    pub reason: SafetyResultReason,
    pub limit: Option<LimitedEngagementReason>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmergencyCopy {
    pub copy: String,
    pub link: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FosnrReason {
    pub policy: AppealablePolicy,
    pub level: i8,
    pub proactive: bool,
    pub appeal_submitted: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decided<T> {
    pub value: T,
    pub(crate) id: RuleId,
    pub by: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Withholding {
    Drop(DropReason),
    Tombstone(TombstoneReason),
}

impl Withholding {
    pub(crate) const fn kind(&self) -> ActionKind {
        match self {
            Self::Drop(_) => ActionKind::Drop,
            Self::Tombstone(_) => ActionKind::Tombstone,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DropReason {
    Legacy(FilteredReason),
    NsfwViewer(NsfwViewerDropReason),
}

impl DropReason {
    pub fn legacy(&self) -> &FilteredReason {
        match self {
            Self::Legacy(reason) => reason,
            Self::NsfwViewer(_) => &FilteredReason::ContainNsfwMedia,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum NsfwViewerDropReason {
    IsUnderage,
    HasNoStatedAge,
    LoggedOut,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediaRestriction {
    MediaInterstitial(MediaInterstitial),
    NsfwInterstitial,
}

impl MediaRestriction {
    pub fn legacy(&self) -> &FilteredReason {
        match self {
            Self::MediaInterstitial(blur) => &blur.legacy,
            Self::NsfwInterstitial => &FilteredReason::ContainNsfwMedia,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MediaInterstitial {
    pub legacy: FilteredReason,
    pub reason: InterstitialReason,
    pub prompt: Option<InterstitialAction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitedEngagement {
    first: LimitedEngagementReason,
    rest: Vec<LimitedEngagementReason>,
}

impl LimitedEngagement {
    pub fn new(reason: LimitedEngagementReason) -> Self {
        Self {
            first: reason,
            rest: Vec::new(),
        }
    }

    pub fn add(&mut self, reason: LimitedEngagementReason) -> bool {
        let added = !self.reasons().any(|held| held == reason);
        if added {
            self.rest.push(reason);
        }
        added
    }

    pub fn reason(&self) -> LimitedEngagementReason {
        self.first
    }

    pub fn reasons(&self) -> impl Iterator<Item = LimitedEngagementReason> + '_ {
        std::iter::once(self.first).chain(self.rest.iter().copied())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum TombstoneReason {
    Bounced,
    SensitiveViewerAgeVerification,
    UpdateAppIos,
    UpdateAppAndroid,
    LocalRegulations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum LimitedEngagementReason {
    NonCompliant,
    ConversationControl,
    ReadonlyViewer,
    BlockedViewer,
    RootAuthorBlockedViewer,
    StaleTweet,
    CommunityTweetHidden,
    CommunityTweetMemberRemoved,
    CommunityTweetCommunityNotFound,
    CommunityTweetCommunityDeleted,
    CommunityTweetCommunitySuspended,
    CommunityTweetViewerRemoved,
    DynamicProductAd,
    LocalTweet,
}

impl LimitedEngagementReason {
    pub const fn limited_actions_string(self) -> &'static str {
        match self {
            Self::NonCompliant => "non_compliant",
            Self::ConversationControl => "limited_replies",
            Self::ReadonlyViewer => "readonly_viewer",
            Self::BlockedViewer => "blocked_viewer",
            Self::RootAuthorBlockedViewer => "root_author_blocked_viewer",
            Self::StaleTweet => "stale_tweet",
            Self::CommunityTweetHidden => "community_tweet_hidden",
            Self::CommunityTweetMemberRemoved => "community_tweet_member_removed",
            Self::CommunityTweetCommunityNotFound => "community_tweet_community_not_found",
            Self::CommunityTweetCommunityDeleted => "community_tweet_community_deleted",
            Self::CommunityTweetCommunitySuspended => "community_tweet_community_suspended",
            Self::CommunityTweetViewerRemoved => "community_tweet_viewer_removed",
            Self::DynamicProductAd => "dynamic_product_ad",
            Self::LocalTweet => "local_tweet",
        }
    }

    pub fn from_misinfo_source(source: &str) -> Option<Self> {
        [
            Self::NonCompliant,
            Self::ConversationControl,
            Self::CommunityTweetHidden,
            Self::CommunityTweetMemberRemoved,
            Self::CommunityTweetViewerRemoved,
            Self::StaleTweet,
            Self::LocalTweet,
            Self::ReadonlyViewer,
            Self::BlockedViewer,
            Self::RootAuthorBlockedViewer,
        ]
        .into_iter()
        .find(|reason| reason.limited_actions_string() == source)
    }
}

impl Verdict {
    pub(crate) fn limited_engagement_reasons(
        &self,
    ) -> impl Iterator<Item = LimitedEngagementReason> + '_ {
        let (interstitial, limit) = match self {
            Self::Shown {
                label_interstitial,
                engagement,
                ..
            } => (
                label_interstitial
                    .as_ref()
                    .map(|interstitial| interstitial.value.limited_engagement_reason()),
                engagement.as_ref(),
            ),
            Self::Withheld(_) | Self::Fosnr(_) => (None, None),
        };
        interstitial
            .into_iter()
            .chain(limit.into_iter().flat_map(|limit| limit.value.reasons()))
    }

    pub(crate) fn decided(&self) -> impl Iterator<Item = (ActionKind, &'static str)> + '_ {
        let (first, second, third) = match self {
            Self::Withheld(decided) => (Some((decided.value.kind(), decided.by)), None, None),
            Self::Fosnr(decided) => (Some((decided.value.kind(), decided.by)), None, None),
            Self::Shown {
                label_interstitial,
                media,
                engagement,
            } => (
                label_interstitial
                    .as_ref()
                    .map(|decided| (ActionKind::LabelInterstitial, decided.by)),
                media
                    .as_ref()
                    .map(|decided| (ActionKind::MediaRestriction, decided.by)),
                engagement
                    .as_ref()
                    .map(|decided| (ActionKind::LimitedEngagement, decided.by)),
            ),
        };
        first.into_iter().chain(second).chain(third)
    }

    pub const fn not_found() -> Self {
        Self::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::UnspecifiedReason)),
            id: RuleId::NotFound,
            by: "not_found",
        })
    }

    pub const fn lookup_failed() -> Self {
        Self::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::UnspecifiedReason)),
            id: RuleId::LookupFailed,
            by: "lookup_failed",
        })
    }
}

static NOT_FOUND: Verdict = Verdict::not_found();
static LOOKUP_FAILED: Verdict = Verdict::lookup_failed();

#[derive(Clone, Debug, PartialEq)]
pub enum Evaluation {
    Complete {
        verdict: Verdict,
        fail_open_defaults: Hydrators,
    },
    Partial {
        verdict: Verdict,
        fail_open_defaults: Hydrators,
    },
    NotFound(Lookup),
    Deleted(Deletion),
    Failed(Lookup),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deletion {
    Deleted,
    BounceDeleted,
}

impl Evaluation {
    pub fn verdict(&self) -> &Verdict {
        match self {
            Self::Complete { verdict, .. } | Self::Partial { verdict, .. } => verdict,
            Self::NotFound(_) | Self::Deleted(_) => &NOT_FOUND,
            Self::Failed(_) => &LOOKUP_FAILED,
        }
    }

    #[cfg(test)]
    pub fn into_verdict(self) -> Verdict {
        match self {
            Self::Complete { verdict, .. } | Self::Partial { verdict, .. } => verdict,
            Self::NotFound(_) | Self::Deleted(_) => Verdict::not_found(),
            Self::Failed(_) => Verdict::lookup_failed(),
        }
    }

    pub const fn complete(verdict: Verdict) -> Self {
        Self::Complete {
            verdict,
            fail_open_defaults: Hydrators::empty(),
        }
    }

    pub fn fail_open_defaults(&self) -> Hydrators {
        match self {
            Self::Complete {
                fail_open_defaults, ..
            }
            | Self::Partial {
                fail_open_defaults, ..
            } => *fail_open_defaults,
            Self::NotFound(_) | Self::Deleted(_) | Self::Failed(_) => Hydrators::empty(),
        }
    }
}
