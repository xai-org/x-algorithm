use crate::models::{
    ClientSwitch, LimitedActionType, LimitedEngagementReason, SafetyLabelType, TombstoneReason,
    VerifyBlurSupport,
};
use crate::params::CountryList;
use crate::rules::rule_spec::conditions::*;
use crate::rules::rule_spec::leaves::*;
use crate::rules::rule_spec::rule::Audience::{Everyone, NonAuthors, NonFollowers, OnlyAuthor};
use crate::rules::rule_spec::rule::{all, Rule, UserRule};
use crate::rules::rule_spec::*;
use xai_core_entities::entities::ConversationControlArm;
use xai_x_thrift::action::{AppealablePolicy, InterstitialReason};

pub(super) const FILTER_ALL_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::FilterAll,
    audience: Everyone,
    when: all![],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const ONLY_PEOPLE_I_FOLLOW_OUTSIDE_COMMUNITY_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::OnlyPeopleIFollowOutsideCommunity,
    audience: NonFollowers,
    when: all![ViewerMentionFilterIsFollowing, !TweetIsCommunityTweet],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const PDNA_DROP_SAFETY_RESULT: Rule = Rule {
    id: RuleId::Pdna,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::PDNA)],
    action: drop_post(DropAs::NsfwHighPrecisionSafetyResult),
};

pub(super) const BOUNCE_DROP_BOUNCED: Rule = Rule {
    id: RuleId::Bounce,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::BOUNCE)],
    action: drop_post(DropAs::TweetIsBounced),
};

pub(super) const BOUNCE_QUOTED_TOMBSTONE_BOUNCED: Rule = Rule {
    id: RuleId::BounceQuoted,
    audience: Everyone,
    when: all![IsQuotedInnerTweet, TweetHasLabel(SafetyLabelType::BOUNCE)],
    action: tombstone(TombstoneReason::Bounced),
};

pub(super) const SPAM_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::Spam,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::SPAM)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const FOR_EMERGENCY_USE_ONLY_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::ForEmergencyUseOnly,
    audience: Everyone,
    when: all![TweetHasLabel(SafetyLabelType::FOR_EMERGENCY_USE_ONLY)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const FOR_EMERGENCY_USE_ONLY_LABEL_INTERSTITIAL: Rule = Rule {
    id: RuleId::ForEmergencyUseOnly,
    audience: Everyone,
    when: all![TweetHasLabel(SafetyLabelType::FOR_EMERGENCY_USE_ONLY)],
    action: label_interstitial(LabelInterstitialSpec::EmergencyDynamic),
};

const PUBLIC_INTEREST_LABELS_START: u64 = 1554076800000;

pub(super) const PUBLIC_INTEREST_LABEL_INTERSTITIAL: Rule = Rule {
    id: RuleId::PublicInterest,
    audience: Everyone,
    when: all![
        TweetCreatedAfter(PUBLIC_INTEREST_LABELS_START),
        any![
            TweetHasLabel(SafetyLabelType::LOW_QUALITY),
            TweetHasLabel(SafetyLabelType::MISINFO_CIVIC),
            TweetHasLabel(SafetyLabelType::MISINFO_GENERIC),
            TweetHasLabel(SafetyLabelType::MISINFO_MEDICAL),
            TweetHasLabel(SafetyLabelType::MISINFO_CRISIS),
            TweetHasLabel(SafetyLabelType::IPI_DEVELOPMENT_ONLY)
        ]
    ],
    action: label_interstitial(LabelInterstitialSpec::PublicInterest),
};

const AUTHOR_LEVEL_3_LIMITED_ACTIONS: [LimitedActionType; 13] = {
    use LimitedActionType as A;
    [
        A::Like,
        A::Reply,
        A::Retweet,
        A::QuoteTweet,
        A::ShareTweetVia,
        A::AddToBookmarks,
        A::PinToProfile,
        A::CopyLink,
        A::SendViaDm,
        A::EditTweet,
        A::Highlight,
        A::Embed,
        A::ListsAddRemove,
    ]
};

pub(super) const FOSNR_AUTHOR_VIOLATIONS: [FosnrViolation; 5] = [
    FosnrViolation {
        label: SafetyLabelType::FOSNR_ABUSE,
        policy: AppealablePolicy::ABUSE,
        level: 3,
        limited_actions: &AUTHOR_LEVEL_3_LIMITED_ACTIONS,
    },
    FosnrViolation {
        label: SafetyLabelType::FOSNR_CIVIC_INTEGRITY,
        policy: AppealablePolicy::CIVIC_INTEGRITY,
        level: 3,
        limited_actions: &AUTHOR_LEVEL_3_LIMITED_ACTIONS,
    },
    FosnrViolation {
        label: SafetyLabelType::FOSNR_HATEFUL_CONDUCT,
        policy: AppealablePolicy::HATEFUL_CONDUCT,
        level: 3,
        limited_actions: &AUTHOR_LEVEL_3_LIMITED_ACTIONS,
    },
    FosnrViolation {
        label: SafetyLabelType::FOSNR_VIOLENT_SPEECH,
        policy: AppealablePolicy::VIOLENT_SPEECH,
        level: 3,
        limited_actions: &AUTHOR_LEVEL_3_LIMITED_ACTIONS,
    },
    FosnrViolation {
        label: SafetyLabelType::FOSNR_ABUSE_INSULTS,
        policy: AppealablePolicy::ABUSE,
        level: 1,
        limited_actions: &[LimitedActionType::EditTweet],
    },
];

pub(super) const FOSNR_HATEFUL_CONDUCT_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::FosnrHatefulConduct,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::FOSNR_HATEFUL_CONDUCT)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const FOSNR_VIOLENT_SPEECH_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::FosnrViolentSpeech,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::FOSNR_VIOLENT_SPEECH)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const FOSNR_ABUSE_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::FosnrAbuse,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::FOSNR_ABUSE)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const FOSNR_CIVIC_INTEGRITY_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::FosnrCivicIntegrity,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::FOSNR_CIVIC_INTEGRITY)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const FOSNR_ABUSE_INSULTS_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::FosnrAbuseInsults,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::FOSNR_ABUSE_INSULTS)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const FOSNR_ABUSE_INSULTS_NON_FOLLOWER_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::FosnrAbuseInsultsNonFollower,
    audience: NonFollowers,
    when: all![
        TweetHasLabel(SafetyLabelType::FOSNR_ABUSE_INSULTS),
        ClientHas(ClientSwitch::FosnrRules)
    ],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const FOSNR_ABUSE_INSULTS_FOLLOWER_SOFT_INTERVENTION_ABUSE: Rule = Rule {
    id: RuleId::FosnrAbuseInsultsFollower,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::FOSNR_ABUSE_INSULTS),
        ViewerFollowsAuthor,
        ClientHas(ClientSwitch::FosnrRules)
    ],
    action: soft_intervention(
        SafetyLabelType::FOSNR_ABUSE_INSULTS,
        AppealablePolicy::ABUSE,
        1,
    ),
};

pub(super) const FOSNR_AUTHOR_APPEALABLE: Rule = Rule {
    id: RuleId::FosnrAuthor,
    audience: OnlyAuthor,
    when: all![ClientHas(ClientSwitch::FosnrRules), HAS_FOSNR_LABEL],
    action: appealable(&FOSNR_AUTHOR_VIOLATIONS),
};

pub(super) const FOSNR_FALLBACK_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::FosnrFallback,
    audience: Everyone,
    when: all![ClientHas(ClientSwitch::FosnrFallbackDrops), HAS_FOSNR_LABEL],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const NULLCASTED_TWEET_DROP: Rule = Rule {
    id: RuleId::NullcastedTweet,
    audience: Everyone,
    when: all![TweetIsNullcast, !TweetIsRetweet, !TweetIsCommunityTweet],
    action: drop_post(DropAs::TweetIsNullcast),
};

pub(super) const STALE_TWEET_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::StaleTweet,
    audience: Everyone,
    when: all![
        TweetIsSupersededEdit,
        !IsQuotedInnerTweet,
        !TweetIsRetweet,
        !IsSourceTweet
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const STALE_TWEET_LIMITED_ENGAGEMENT: Rule = Rule {
    id: RuleId::StaleTweet,
    audience: Everyone,
    when: all![
        TweetIsSupersededEdit,
        ClientHas(ClientSwitch::StaleTweetLimits)
    ],
    action: limit(LimitedEngagementReason::StaleTweet),
};

pub(super) const LEGAL_TAKEDOWN_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::LegalTakedown,
    audience: NonAuthors,
    when: all![TweetLegalTakedownInRequestCountry],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const LOCAL_LAWS_TAKEDOWN_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::LocalLawsTakedown,
    audience: NonAuthors,
    when: all![TweetLocalLawsTakedownInRequestCountry],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const ARTICLE_TWEET_CONTENT_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::ArticleTweetContent,
    audience: Everyone,
    when: all![TweetHasArticle, !TweetArticleIsPublished],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const PROTECTED_COMMUNITY_TWEET_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::ProtectedCommunityTweet,
    audience: NonAuthors,
    when: all![TweetIsCommunityTweet, AuthorIsProtected],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const HIDDEN_COMMUNITY_TWEET_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::HiddenCommunityTweet,
    audience: NonAuthors,
    when: all![
        TweetIsCommunityTweet,
        TweetCommunityTweetIsHidden,
        !ViewerIsCommunityModerator
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const AUTHOR_REMOVED_COMMUNITY_TWEET_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::AuthorRemovedCommunityTweet,
    audience: NonAuthors,
    when: all![
        TweetIsCommunityTweet,
        TweetCommunityTweetAuthorIsRemoved,
        !ViewerIsCommunityModerator
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const SENSITIVE_VIEWER_LOGGED_OUT_DROP_MEDIA_LABEL: Rule = Rule {
    id: RuleId::SensitiveViewerLoggedOut,
    audience: NonAuthors,
    when: all![ViewerLoggedOut, TweetHasMedia, HAS_NSFW_MEDIA_LABEL],
    action: drop_post(DropAs::NsfwViewerLoggedOut),
};

pub(super) const SENSITIVE_VIEWER_LOGGED_OUT_DROP_NSFW_FLAG: Rule = Rule {
    id: RuleId::SensitiveViewerLoggedOut,
    audience: NonAuthors,
    when: all![
        ViewerLoggedOut,
        TweetHasMedia,
        !TweetIsRetweet,
        IS_NSFW_FLAGGED
    ],
    action: drop_post(DropAs::NsfwViewerLoggedOut),
};

pub(super) const SENSITIVE_VIEWER_LOGGED_OUT_DROP_TEXT_OR_CARD_LABEL: Rule = Rule {
    id: RuleId::SensitiveViewerLoggedOut,
    audience: NonAuthors,
    when: all![ViewerLoggedOut, HAS_NSFW_TEXT_OR_CARD_LABEL],
    action: drop_post(DropAs::NsfwViewerLoggedOut),
};

pub(super) const SENSITIVE_VIEWER_UNDERAGE_DROP_MEDIA_LABEL: Rule = Rule {
    id: RuleId::SensitiveViewerUnderage,
    audience: NonAuthors,
    when: all![ViewerUnderage, TweetHasMedia, HAS_NSFW_MEDIA_LABEL],
    action: drop_post(DropAs::NsfwViewerIsUnderage),
};

pub(super) const SENSITIVE_VIEWER_UNDERAGE_DROP_NSFW_FLAG: Rule = Rule {
    id: RuleId::SensitiveViewerUnderage,
    audience: NonAuthors,
    when: all![
        ViewerUnderage,
        TweetHasMedia,
        !TweetIsRetweet,
        IS_NSFW_FLAGGED
    ],
    action: drop_post(DropAs::NsfwViewerIsUnderage),
};

pub(super) const SENSITIVE_VIEWER_UNDERAGE_DROP_TEXT_OR_CARD_LABEL: Rule = Rule {
    id: RuleId::SensitiveViewerUnderage,
    audience: NonAuthors,
    when: all![ViewerUnderage, HAS_NSFW_TEXT_OR_CARD_LABEL],
    action: drop_post(DropAs::NsfwViewerIsUnderage),
};

pub(super) const SENSITIVE_VIEWER_NO_STATED_AGE_DROP_MEDIA_LABEL: Rule = Rule {
    id: RuleId::SensitiveViewerNoStatedAge,
    audience: NonAuthors,
    when: all![
        ViewerNoStatedAge,
        ViewerAccountOrRequestCountryIn(CountryList::NsfwGating),
        TweetHasMedia,
        HAS_NSFW_MEDIA_LABEL
    ],
    action: drop_post(DropAs::NsfwViewerHasNoStatedAge),
};

pub(super) const SENSITIVE_VIEWER_NO_STATED_AGE_DROP_NSFW_FLAG: Rule = Rule {
    id: RuleId::SensitiveViewerNoStatedAge,
    audience: NonAuthors,
    when: all![
        ViewerNoStatedAge,
        ViewerAccountOrRequestCountryIn(CountryList::NsfwGating),
        TweetHasMedia,
        !TweetIsRetweet,
        IS_NSFW_FLAGGED
    ],
    action: drop_post(DropAs::NsfwViewerHasNoStatedAge),
};

pub(super) const SENSITIVE_VIEWER_NO_STATED_AGE_DROP_TEXT_OR_CARD_LABEL: Rule = Rule {
    id: RuleId::SensitiveViewerNoStatedAge,
    audience: NonAuthors,
    when: all![
        ViewerNoStatedAge,
        ViewerAccountOrRequestCountryIn(CountryList::NsfwGating),
        HAS_NSFW_TEXT_OR_CARD_LABEL
    ],
    action: drop_post(DropAs::NsfwViewerHasNoStatedAge),
};

pub(super) const NSFW_SENSITIVE_VIEWER_TWEET_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::NsfwSensitiveViewerTweet,
    audience: NonAuthors,
    when: all![!ViewerAllowsSensitiveMedia, IS_NSFW_SENSITIVE_TWEET],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const NSFW_SENSITIVE_VIEWER_USER_DROP_NSFW_MEDIA: UserRule = Rule {
    id: RuleId::NsfwSensitiveViewerUser,
    audience: NonAuthors,
    when: all![!ViewerAllowsSensitiveMedia, IS_NSFW_SENSITIVE_AUTHOR],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const EXCLUSIVE_TWEET_DROP_LOGGED_OUT: Rule = Rule {
    id: RuleId::ExclusiveTweet,
    audience: Everyone,
    when: all![TweetHasExclusiveContent, ViewerLoggedOut],
    action: drop_post(DropAs::ExclusiveTweet),
};

pub(super) const EXCLUSIVE_TWEET_DROP_RETWEET: Rule = Rule {
    id: RuleId::ExclusiveTweet,
    audience: Everyone,
    when: all![
        TweetHasExclusiveContent,
        !ViewerIsConversationAuthor,
        !ViewerSuperFollowsAuthor,
        TweetIsRetweet
    ],
    action: drop_post(DropAs::ExclusiveTweet),
};

pub(super) const EXCLUSIVE_TWEET_DROP_NON_SUPER_FOLLOWER: Rule = Rule {
    id: RuleId::ExclusiveTweet,
    audience: NonAuthors,
    when: all![
        TweetHasExclusiveContent,
        !ViewerIsConversationAuthor,
        !ViewerSuperFollowsAuthor
    ],
    action: drop_post(DropAs::ExclusiveTweet),
};

pub(super) const AUTHOR_BLOCKS_VIEWER_EXCLUSIVE_CONTENT_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::AuthorBlocksViewerExclusiveContent,
    audience: NonAuthors,
    when: all![ViewerIsBlockedByAuthor, TweetHasExclusiveContent],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const CREATOR_TWEET_NSFW_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::CreatorTweetNsfw,
    audience: NonAuthors,
    when: all![
        TweetHasMedia,
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        TweetHasExclusiveContent
    ],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const TRUSTED_FRIENDS_TWEET_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::TrustedFriendsTweet,
    audience: NonAuthors,
    when: all![
        TweetIsTrustedFriendsTweet,
        !ViewerIsTrustedFriendsListMemberOrOwner
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

const NSFW_HIGH_PRECISION_CHANGED_AT: u64 = 1705536000000;

pub(super) const NSFW_HIGH_PRECISION_BLUR_NUDITY: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        TweetCreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT),
        !ViewerAllowsSensitiveMedia
    ],
    action: blur(InterstitialReason::Nudity(true)),
};

pub(super) const NSFW_HIGH_PRECISION_BLUR_SENSITIVE: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        !TweetCreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT),
        !ViewerAllowsSensitiveMedia
    ],
    action: blur(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_HIGH_PRECISION_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION)],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const NSFW_HIGH_PRECISION_TOMBSTONE_LOCAL_REGULATIONS: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        RequestCountryIn(CountryList::LocalRegulations)
    ],
    action: tombstone(TombstoneReason::LocalRegulations),
};

pub(super) const NSFW_HIGH_PRECISION_TOMBSTONE_AGE_VERIFICATION: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Unsupported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES
    ],
    action: tombstone(TombstoneReason::SensitiveViewerAgeVerification),
};

pub(super) const NSFW_HIGH_PRECISION_TOMBSTONE_UPDATE_APP_IOS: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::IosNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppIos),
};

pub(super) const NSFW_HIGH_PRECISION_TOMBSTONE_UPDATE_APP_ANDROID: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::AndroidNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppAndroid),
};

pub(super) const NSFW_HIGH_PRECISION_BLUR_SENSITIVE_AGE_PROMPT: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        !TweetCreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Supported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: blur_with_age_prompt(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_HIGH_PRECISION_BLUR_SENSITIVE_ALL_USERS: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: Everyone,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        !TweetCreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT),
        !ViewerAllowsSensitiveMedia,
        ClientHas(ClientSwitch::ModernBlur)
    ],
    action: blur(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_HIGH_PRECISION_BLUR_NUDITY_AGE_PROMPT: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        TweetCreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Supported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: blur_with_age_prompt(InterstitialReason::Nudity(true)),
};

pub(super) const NSFW_HIGH_PRECISION_BLUR_NUDITY_ALL_USERS: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: Everyone,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        TweetCreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT),
        !ViewerAllowsSensitiveMedia,
        ClientHas(ClientSwitch::ModernBlur)
    ],
    action: blur(InterstitialReason::Nudity(true)),
};

pub(super) const NSFW_HIGH_PRECISION_LEGACY_INTERSTITIAL: Rule = Rule {
    id: RuleId::NsfwHighPrecision,
    audience: Everyone,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION),
        !ClientHas(ClientSwitch::ModernBlur)
    ],
    action: legacy_nsfw_interstitial(),
};

pub(super) const GORE_AND_VIOLENCE_HIGH_PRECISION_BLUR: Rule = Rule {
    id: RuleId::GoreAndViolenceHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        !ViewerAllowsSensitiveMedia
    ],
    action: blur(InterstitialReason::Violence(true)),
};

pub(super) const GORE_AND_VIOLENCE_HIGH_PRECISION_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::GoreAndViolenceHighPrecision,
    audience: NonAuthors,
    when: all![TweetHasLabel(
        SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION
    )],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const GORE_AND_VIOLENCE_HIGH_PRECISION_TOMBSTONE_UPDATE_APP_IOS: Rule = Rule {
    id: RuleId::GoreAndViolenceHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::IosNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppIos),
};

pub(super) const GORE_AND_VIOLENCE_HIGH_PRECISION_TOMBSTONE_UPDATE_APP_ANDROID: Rule = Rule {
    id: RuleId::GoreAndViolenceHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::AndroidNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_TOMBSTONE_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppAndroid),
};

pub(super) const GORE_AND_VIOLENCE_HIGH_PRECISION_BLUR_ALL_USERS: Rule = Rule {
    id: RuleId::GoreAndViolenceHighPrecision,
    audience: Everyone,
    when: all![
        TweetHasLabel(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        !ViewerAllowsSensitiveMedia,
        ClientHas(ClientSwitch::ModernBlur)
    ],
    action: blur(InterstitialReason::Violence(true)),
};

pub(super) const GORE_AND_VIOLENCE_HIGH_PRECISION_LEGACY_INTERSTITIAL: Rule = Rule {
    id: RuleId::GoreAndViolenceHighPrecision,
    audience: Everyone,
    when: all![
        TweetHasLabel(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        !ClientHas(ClientSwitch::ModernBlur)
    ],
    action: legacy_nsfw_interstitial(),
};

pub(super) const GORE_AND_VIOLENCE_HIGH_PRECISION_BLUR_AGE_PROMPT: Rule = Rule {
    id: RuleId::GoreAndViolenceHighPrecision,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Supported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: blur_with_age_prompt(InterstitialReason::Violence(true)),
};

pub(super) const GORE_AND_VIOLENCE_IGNORING_SETTINGS_BLUR: Rule = Rule {
    id: RuleId::GoreAndViolenceIgnoringSettings,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        ClientHas(ClientSwitch::GoreBlurIgnoresSettings)
    ],
    action: blur(InterstitialReason::Violence(true)),
};

pub(super) const NSFW_CARD_IMAGE_BLUR_SENSITIVE: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE),
        !ViewerAllowsSensitiveMedia
    ],
    action: blur(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_CARD_IMAGE_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE)],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const NSFW_CARD_IMAGE_TOMBSTONE_AGE_VERIFICATION: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Unsupported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: tombstone(TombstoneReason::SensitiveViewerAgeVerification),
};

pub(super) const NSFW_CARD_IMAGE_TOMBSTONE_UPDATE_APP_IOS: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::IosNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppIos),
};

pub(super) const NSFW_CARD_IMAGE_TOMBSTONE_UPDATE_APP_ANDROID: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::AndroidNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppAndroid),
};

pub(super) const NSFW_CARD_IMAGE_BLUR_SENSITIVE_AGE_PROMPT: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE),
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Supported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: blur_with_age_prompt(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_CARD_IMAGE_BLUR_SENSITIVE_ALL_USERS: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: Everyone,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE),
        !ViewerAllowsSensitiveMedia,
        ClientHas(ClientSwitch::ModernBlur)
    ],
    action: blur(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_CARD_IMAGE_LEGACY_INTERSTITIAL: Rule = Rule {
    id: RuleId::NsfwCardImage,
    audience: Everyone,
    when: all![
        TweetHasLabel(SafetyLabelType::NSFW_CARD_IMAGE),
        !ClientHas(ClientSwitch::ModernBlur)
    ],
    action: legacy_nsfw_interstitial(),
};

pub(super) const NSFW_ADMIN_BLUR_SENSITIVE: Rule = Rule {
    id: RuleId::NsfwAdmin,
    audience: NonAuthors,
    when: all![IS_NSFW_ADMIN, TweetHasMedia, !ViewerAllowsSensitiveMedia],
    action: blur(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_ADMIN_BLUR_SENSITIVE_AGE_PROMPT: Rule = Rule {
    id: RuleId::NsfwAdmin,
    audience: NonAuthors,
    when: all![
        TweetNsfwAdminFlag,
        TweetHasMedia,
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Supported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: blur_with_age_prompt(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_ADMIN_BLUR_SENSITIVE_ALL_USERS: Rule = Rule {
    id: RuleId::NsfwAdmin,
    audience: Everyone,
    when: all![
        TweetNsfwAdminFlag,
        TweetHasMedia,
        !ViewerAllowsSensitiveMedia,
        ClientHas(ClientSwitch::ModernBlur)
    ],
    action: blur(InterstitialReason::Sensitive(true)),
};

pub(super) const NSFW_USER_BLUR_SENSITIVE_USER: Rule = Rule {
    id: RuleId::NsfwUser,
    audience: NonAuthors,
    when: all![IS_NSFW_USER, TweetHasMedia, !ViewerAllowsSensitiveMedia],
    action: blur(InterstitialReason::SensitiveUser(true)),
};

pub(super) const NSFW_USER_BLUR_SENSITIVE_USER_AGE_PROMPT: Rule = Rule {
    id: RuleId::NsfwUser,
    audience: NonAuthors,
    when: all![
        TweetNsfwUserFlag,
        TweetHasMedia,
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Supported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: blur_with_age_prompt(InterstitialReason::SensitiveUser(true)),
};

pub(super) const NSFW_USER_BLUR_SENSITIVE_USER_ALL_USERS: Rule = Rule {
    id: RuleId::NsfwUser,
    audience: Everyone,
    when: all![
        TweetNsfwUserFlag,
        TweetHasMedia,
        !ViewerAllowsSensitiveMedia,
        ClientHas(ClientSwitch::ModernBlur)
    ],
    action: blur(InterstitialReason::SensitiveUser(true)),
};

pub(super) const NSFW_ACCOUNT_TOMBSTONE_AGE_VERIFICATION: Rule = Rule {
    id: RuleId::NsfwAccount,
    audience: NonAuthors,
    when: all![
        TWEET_IS_NSFW_FLAGGED,
        TweetHasMedia,
        ClientVerifyBlurSupportIs(VerifyBlurSupport::Unsupported),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: tombstone(TombstoneReason::SensitiveViewerAgeVerification),
};

pub(super) const NSFW_ACCOUNT_TOMBSTONE_UPDATE_APP_IOS: Rule = Rule {
    id: RuleId::NsfwAccount,
    audience: NonAuthors,
    when: all![
        TWEET_IS_NSFW_FLAGGED,
        TweetHasMedia,
        ClientVerifyBlurSupportIs(VerifyBlurSupport::IosNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppIos),
};

pub(super) const NSFW_ACCOUNT_TOMBSTONE_UPDATE_APP_ANDROID: Rule = Rule {
    id: RuleId::NsfwAccount,
    audience: NonAuthors,
    when: all![
        TWEET_IS_NSFW_FLAGGED,
        TweetHasMedia,
        ClientVerifyBlurSupportIs(VerifyBlurSupport::AndroidNeedsUpdate),
        VIEWER_IS_NOT_AGE_VERIFIED_IN_AGE_VERIFICATION_COUNTRIES
    ],
    action: tombstone(TombstoneReason::UpdateAppAndroid),
};

pub(super) const NSFW_ACCOUNT_LEGACY_INTERSTITIAL: Rule = Rule {
    id: RuleId::NsfwAccount,
    audience: Everyone,
    when: all![
        TWEET_IS_NSFW_FLAGGED,
        TweetHasMedia,
        !ViewerAllowsSensitiveMedia,
        !ClientHas(ClientSwitch::ModernBlur)
    ],
    action: legacy_nsfw_interstitial(),
};

pub(super) const ORIGINAL_TWEET_ADULT_MEDIA_PROPAGATION_BLUR_SENSITIVE: Rule = Rule {
    id: RuleId::OriginalTweetAdultMediaPropagation,
    audience: Everyone,
    when: all![ReactedToPostHasLabel(SafetyLabelType::NSFW_HIGH_PRECISION)],
    action: blur(InterstitialReason::Sensitive(true)),
};

pub(super) const ORIGINAL_TWEET_GORE_MEDIA_PROPAGATION_BLUR_VIOLENCE: Rule = Rule {
    id: RuleId::OriginalTweetGoreMediaPropagation,
    audience: Everyone,
    when: all![ReactedToPostHasLabel(
        SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION
    )],
    action: blur(InterstitialReason::Violence(true)),
};

pub(super) const PROTECTED_ORIGINAL_AUTHOR_DROP: Rule = Rule {
    id: RuleId::ProtectedOriginalAuthor,
    audience: Everyone,
    when: all![ReactedToAuthorIsProtected],
    action: drop_post(DropAs::AuthorIsProtected),
};

pub(super) const DMCA_MEDIA_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::DmcaMedia,
    audience: Everyone,
    when: all![TweetHasDmcaMedia],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const GEO_RESTRICTED_MEDIA_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::GeoRestrictedMedia,
    audience: Everyone,
    when: all![TweetMediaGeoRestrictedInRequestCountry],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NSFW_USER_TWEET_FLAG_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::NsfwUserTweetFlag,
    audience: Everyone,
    when: all![TweetNsfwUserFlag],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const NSFW_ADMIN_TWEET_FLAG_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::NsfwAdminTweetFlag,
    audience: Everyone,
    when: all![TweetNsfwAdminFlag],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const NSFW_HIGH_RECALL_DROP_NSFW_MEDIA: Rule = Rule {
    id: RuleId::NsfwHighRecall,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::NSFW_HIGH_RECALL)],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const SOFT_NSFW_NEW_USER_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::SoftNsfwNewUser,
    audience: NonAuthors,
    when: all![
        RequestHasSoftNsfwNewUserDrop,
        ViewerIsNewAccount,
        TweetHasLabel(SafetyLabelType::SOFT_NSFW)
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const DO_NOT_AMPLIFY_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::DoNotAmplify,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::DO_NOT_AMPLIFY)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const MALICIOUS_URL_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::MaliciousUrl,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::MALICIOUS_URL)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const SPAM_HIGH_RECALL_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::SpamHighRecall,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::SPAM_HIGH_RECALL)],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const BRAZIL_ELECTION_LEGAL_DROP_UNDESIRABLE: Rule = Rule {
    id: RuleId::BrazilElectionLegal,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::BRAZIL_ELECTION_LEGAL),
        RequestCountryIs("br")
    ],
    action: drop_post(DropAs::PossiblyUndesirable),
};

pub(super) const LOW_QUALITY_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::LowQuality,
    audience: NonAuthors,
    when: all![
        TweetHasLabel(SafetyLabelType::LOW_QUALITY),
        !TweetCreatedAfter(PUBLIC_INTEREST_LABELS_START)
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const DUPLICATE_CONTENT_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::DuplicateContent,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::DUPLICATE_CONTENT)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const SMYTE_SPAM_TWEET_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::SmyteSpamTweet,
    audience: NonAuthors,
    when: all![TweetHasLabel(SafetyLabelType::SMYTE_SPAM_TWEET)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const DUPLICATE_MENTION_UQF_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::DuplicateMentionUqf,
    audience: NonFollowers,
    when: all![
        TweetHasLabel(SafetyLabelType::DUPLICATE_MENTION),
        ViewerHasUqfEnabled
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const ABUSIVE_UQF_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::AbusiveUqf,
    audience: NonFollowers,
    when: all![TweetHasLabel(SafetyLabelType::ABUSIVE), ViewerHasUqfEnabled],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const UNTRUSTED_URL_UQF_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::UntrustedUrlUqf,
    audience: NonFollowers,
    when: all![
        TweetHasLabel(SafetyLabelType::UNTRUSTED_URL),
        ViewerHasUqfEnabled
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const DOWNRANK_SPAM_REPLY_UQF_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::DownrankSpamReplyUqf,
    audience: NonFollowers,
    when: all![
        TweetHasLabel(SafetyLabelType::DOWNRANK_SPAM_REPLY),
        ViewerHasUqfEnabled
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const BLOCKED_VIEWER_LIMITED_ENGAGEMENT: UserRule = Rule {
    id: RuleId::BlockedViewer,
    audience: NonAuthors,
    when: all![ViewerIsBlockedByAuthor],
    action: limit(LimitedEngagementReason::BlockedViewer),
};

pub(super) const BLOCKED_VIEWER_LIMITED_ENGAGEMENT_ROOT_AUTHOR_BLOCKED_VIEWER: Rule = Rule {
    id: RuleId::BlockedViewer,
    audience: Everyone,
    when: all![ViewerIsBlockedByConversationRootAuthor],
    action: limit(LimitedEngagementReason::RootAuthorBlockedViewer),
};

pub(super) const LIMIT_REPLIES_BY_INVITATION_LIMITED_ENGAGEMENT_CONVERSATION_CONTROL: Rule = Rule {
    id: RuleId::LimitRepliesByInvitation,
    audience: Everyone,
    when: all![
        TweetHasConversationControl(ConversationControlArm::ByInvitation),
        !ViewerLoggedOut,
        !TweetIsRetweet,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED
    ],
    action: limit(LimitedEngagementReason::ConversationControl),
};

pub(super) const LIMIT_REPLIES_COMMUNITY_LIMITED_ENGAGEMENT_CONVERSATION_CONTROL: Rule = Rule {
    id: RuleId::LimitRepliesCommunity,
    audience: Everyone,
    when: all![
        TweetHasConversationControl(ConversationControlArm::Community),
        !ViewerLoggedOut,
        !TweetIsRetweet,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
        !ViewerIsFollowedByConversationRootAuthor
    ],
    action: limit(LimitedEngagementReason::ConversationControl),
};

pub(super) const LIMIT_REPLIES_FOLLOWERS_LIMITED_ENGAGEMENT_CONVERSATION_CONTROL: Rule = Rule {
    id: RuleId::LimitRepliesFollowers,
    audience: Everyone,
    when: all![
        TweetHasConversationControl(ConversationControlArm::Followers),
        !ViewerLoggedOut,
        !TweetIsRetweet,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
        !ViewerFollowsConversationRootAuthor
    ],
    action: limit(LimitedEngagementReason::ConversationControl),
};

pub(super) const LIMIT_REPLIES_SUBSCRIBERS_LIMITED_ENGAGEMENT_CONVERSATION_CONTROL: Rule = Rule {
    id: RuleId::LimitRepliesSubscribers,
    audience: Everyone,
    when: all![
        TweetHasConversationControl(ConversationControlArm::Subscribers),
        !ViewerLoggedOut,
        !TweetIsRetweet,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
        !ViewerSuperFollowsConversationRootAuthor
    ],
    action: limit(LimitedEngagementReason::ConversationControl),
};

pub(super) const LIMIT_REPLIES_VERIFIED_LIMITED_ENGAGEMENT_CONVERSATION_CONTROL: Rule = Rule {
    id: RuleId::LimitRepliesVerified,
    audience: Everyone,
    when: all![
        TweetHasConversationControl(ConversationControlArm::Verified),
        !ViewerLoggedOut,
        !TweetIsRetweet,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
        !ViewerHasVerifiedBadge
    ],
    action: limit(LimitedEngagementReason::ConversationControl),
};

pub(super) const LIMIT_REPLIES_MY_NETWORK_LIMITED_ENGAGEMENT_CONVERSATION_CONTROL: Rule = Rule {
    id: RuleId::LimitRepliesMyNetwork,
    audience: Everyone,
    when: all![
        TweetHasConversationControl(ConversationControlArm::MyNetwork),
        !ViewerLoggedOut,
        !TweetIsRetweet,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
        !ViewerIsInConversationRootAuthorNetwork
    ],
    action: limit(LimitedEngagementReason::ConversationControl),
};

pub(super) const LIMIT_REPLIES_CO_LIMITED_ENGAGEMENT_CONVERSATION_CONTROL: Rule = Rule {
    id: RuleId::LimitRepliesCo,
    audience: Everyone,
    when: all![
        TweetHasConversationControl(ConversationControlArm::Co),
        !ViewerLoggedOut,
        !TweetIsRetweet,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
        !ViewerIsInAllowedCountry
    ],
    action: limit(LimitedEngagementReason::ConversationControl),
};

pub(super) const COMMUNITY_TWEET_VIEWER_REMOVED_LIMITED_ENGAGEMENT: Rule = Rule {
    id: RuleId::CommunityTweetViewerRemoved,
    audience: Everyone,
    when: all![
        TweetIsCommunityTweet,
        ViewerIsRemovedFromCommunity,
        ClientHas(ClientSwitch::CommunityViewerRemovedLimits)
    ],
    action: limit(LimitedEngagementReason::CommunityTweetViewerRemoved),
};

pub(super) const DYNAMIC_PRODUCT_AD_LIMITED_ENGAGEMENT: Rule = Rule {
    id: RuleId::DynamicProductAd,
    audience: Everyone,
    when: all![TweetHasLabel(SafetyLabelType::DYNAMIC_PRODUCT_AD)],
    action: limit(LimitedEngagementReason::DynamicProductAd),
};

pub(super) const LOCAL_TWEET_LIMITED_ENGAGEMENT: Rule = Rule {
    id: RuleId::LocalTweet,
    audience: Everyone,
    when: all![
        TweetHasNarrowcastPlace,
        !ViewerLoggedOut,
        VIEWER_IS_NEITHER_ROOT_AUTHOR_NOR_INVITED,
        ViewerIsOutsideNarrowcastPlace
    ],
    action: limit(LimitedEngagementReason::LocalTweet),
};

pub(super) const READ_ONLY_VIEWER_LIMITED_ENGAGEMENT: Rule = Rule {
    id: RuleId::ReadOnlyViewer,
    audience: Everyone,
    when: all![ViewerReadOnly],
    action: limit(LimitedEngagementReason::ReadonlyViewer),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_moderated_community_drops_exempt_the_author_and_the_community_moderators() {
        use crate::models::{CommunityModeration, Decided, Placement, TweetFeatures, Verdict};
        use crate::rules::fixtures::{author_viewer, candidate, viewer};
        use crate::rules::{RuleEngine, SafetyLevel};
        use std::num::NonZeroU64;
        let engine = RuleEngine::for_tests();
        for (moderation, rule) in [
            (
                CommunityModeration {
                    is_hidden: true,
                    is_author_removed: false,
                },
                "hidden_community_tweet/drop/unspecified",
            ),
            (
                CommunityModeration {
                    is_hidden: false,
                    is_author_removed: true,
                },
                "author_removed_community_tweet/drop/unspecified",
            ),
        ] {
            for (viewer, is_moderator, drops) in [
                (viewer(1), Some(false), true),
                (viewer(1), Some(true), false),
                (author_viewer(), Some(false), false),
            ] {
                let mut post = candidate()
                    .with_tweet_features(TweetFeatures {
                        community_id: NonZeroU64::new(500),
                        ..Default::default()
                    })
                    .build();
                post.post.community_moderation = moderation;
                post.post.viewer_is_community_moderator = is_moderator;
                let verdict = engine
                    .evaluate(
                        SafetyLevel::TimelineHomeHydration,
                        &viewer,
                        &post,
                        Placement::Standalone,
                    )
                    .into_verdict();
                assert_eq!(
                    matches!(verdict, Verdict::Withheld(Decided { by, .. }) if by == rule),
                    drops,
                    "{rule}, moderator {is_moderator:?}: {verdict:?}"
                );
            }
        }
    }

    #[test]
    fn home_hydration_label_blurs_add_the_modern_blur_gate_to_the_timeline_home_conditions() {
        use crate::rules::levels::{TIMELINE_HOME, TIMELINE_HOME_HYDRATION};
        let label_blurs: Vec<&Rule> = TIMELINE_HOME
            .lists()
            .post
            .into_iter()
            .filter(|rule| {
                matches!(rule.action.spec(), ActionSpec::MediaRestriction(_))
                    && rule
                        .when
                        .conjuncts()
                        .iter()
                        .any(|condition| !condition.positive_tweet_labels().is_empty())
            })
            .collect();
        assert!(!label_blurs.is_empty());
        for home in label_blurs {
            let mut gated = home.when.conjuncts().to_vec();
            gated.push(ClientHas(ClientSwitch::ModernBlur).leaf());
            assert!(
                TIMELINE_HOME_HYDRATION
                    .lists()
                    .post
                    .iter()
                    .any(|rule| rule.id == home.id
                        && rule.action == home.action
                        && rule.when.conjuncts() == gated.as_slice()),
                "{} has no TimelineHomeHydration rule with its conditions and client_has(modern_blur)",
                home.name()
            );
        }
    }
}
