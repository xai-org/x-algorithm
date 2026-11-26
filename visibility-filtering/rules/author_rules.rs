use crate::models::AuthorLabel;
use crate::rules::rule_spec::leaves::*;
use crate::rules::rule_spec::rule::Audience::{Everyone, NonAuthors, NonFollowers};
use crate::rules::rule_spec::rule::{all, Rule, UserRule};
use crate::rules::rule_spec::*;

pub(super) const SUSPENDED_AUTHOR_DROP: UserRule = Rule {
    id: RuleId::SuspendedAuthor,
    audience: NonAuthors,
    when: all![AuthorIsSuspended],
    action: drop_post(DropAs::AuthorIsSuspended),
};

pub(super) const DEACTIVATED_AUTHOR_DROP: UserRule = Rule {
    id: RuleId::DeactivatedAuthor,
    audience: NonAuthors,
    when: all![AuthorIsDeactivated],
    action: drop_post(DropAs::AuthorIsDeactivated),
};

pub(super) const ERASED_AUTHOR_DROP_INACTIVE: UserRule = Rule {
    id: RuleId::ErasedAuthor,
    audience: NonAuthors,
    when: all![AuthorIsErased],
    action: drop_post(DropAs::AuthorAccountIsInactive),
};

pub(super) const OFFBOARDED_AUTHOR_DROP_INACTIVE: UserRule = Rule {
    id: RuleId::OffboardedAuthor,
    audience: NonAuthors,
    when: all![AuthorIsOffboarded],
    action: drop_post(DropAs::AuthorAccountIsInactive),
};

pub(super) const PROTECTED_AUTHOR_DROP: UserRule = Rule {
    id: RuleId::ProtectedAuthor,
    audience: NonFollowers,
    when: all![AuthorIsProtected],
    action: drop_post(DropAs::AuthorIsProtected),
};

pub(super) const VIEWER_BLOCKS_AUTHOR_DROP: UserRule = Rule {
    id: RuleId::ViewerBlocksAuthor,
    audience: Everyone,
    when: all![!ViewerLoggedOut, ViewerBlocksAuthor],
    action: drop_post(DropAs::ViewerBlocksAuthor),
};

pub(super) const VIEWER_MUTES_AUTHOR_DROP: UserRule = Rule {
    id: RuleId::ViewerMutesAuthor,
    audience: Everyone,
    when: all![!ViewerLoggedOut, ViewerMutesAuthor],
    action: drop_post(DropAs::ViewerMutesAuthor),
};

pub(super) const VIEWER_MUTES_RETWEETS_DROP_UNSPECIFIED: Rule = Rule {
    id: RuleId::ViewerMutesRetweets,
    audience: Everyone,
    when: all![
        !ViewerLoggedOut,
        TweetIsRetweet,
        ViewerMutesRetweetsFromAuthor
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const QUOTED_VIEWER_MUTES_AUTHOR_NOT_FOLLOWING_DROP: UserRule = Rule {
    id: RuleId::QuotedViewerMutesAuthorNotFollowing,
    audience: NonAuthors,
    when: all![!ViewerLoggedOut, ViewerMutesAuthor, !ViewerFollowsAuthor],
    action: drop_post(DropAs::ViewerMutesAuthor),
};

pub(super) const QUOTED_AUTHOR_BLOCKS_QUOTING_AUTHOR_DROP_AUTHOR_BLOCK_VIEWER: UserRule = Rule {
    id: RuleId::QuotedAuthorBlocksQuotingAuthor,
    audience: Everyone,
    when: all![!IsSelfQuote, AuthorBlocksOuterAuthor],
    action: drop_post(DropAs::AuthorBlockViewer),
};

pub(super) const PROTECTED_QUOTED_AUTHOR_DROP: UserRule = Rule {
    id: RuleId::ProtectedQuotedAuthor,
    audience: Everyone,
    when: all![!IsSelfQuote, !OuterAuthorFollowsAuthor, AuthorIsProtected],
    action: drop_post(DropAs::AuthorIsProtected),
};

pub(super) const DEACTIVATED_VIEWER_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::DeactivatedViewer,
    audience: NonAuthors,
    when: all![ViewerIsDeactivated],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const SUSPENDED_VIEWER_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::SuspendedViewer,
    audience: NonAuthors,
    when: all![ViewerIsSuspended],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const VIEWER_MUTES_AND_DOES_NOT_FOLLOW_AUTHOR_DROP_VIEWER_MUTES_AUTHOR: UserRule =
    Rule {
        id: RuleId::ViewerMutesAndDoesNotFollowAuthor,
        audience: NonAuthors,
        when: all![!ViewerLoggedOut, ViewerMutesAuthor, !ViewerFollowsAuthor],
        action: drop_post(DropAs::ViewerMutesAuthor),
    };

pub(super) const NO_CONFIRMED_EMAIL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NoConfirmedEmail,
    audience: NonFollowers,
    when: all![ViewerFiltersNoConfirmedEmail, !AuthorHasConfirmedEmail],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NO_CONFIRMED_PHONE_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NoConfirmedPhone,
    audience: NonFollowers,
    when: all![ViewerFiltersNoConfirmedPhone, !AuthorHasVerifiedPhone],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NO_DEFAULT_PROFILE_IMAGE_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NoDefaultProfileImage,
    audience: NonFollowers,
    when: all![
        ViewerFiltersDefaultProfileImage,
        AuthorHasDefaultProfileImage
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NO_NEW_USERS_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NoNewUsers,
    audience: NonFollowers,
    when: all![ViewerFiltersNewUsers, AuthorIsNewAccount],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NO_NOT_FOLLOWED_BY_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NoNotFollowedBy,
    audience: NonFollowers,
    when: all![ViewerFiltersNotFollowedBy, !AuthorFollowsViewer],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const ONLY_PEOPLE_I_FOLLOW_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::OnlyPeopleIFollow,
    audience: NonFollowers,
    when: all![ViewerMentionFilterIsFollowing],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NSFW_USER_AUTHOR_DROP_NSFW_MEDIA: UserRule = Rule {
    id: RuleId::NsfwUserAuthor,
    audience: NonAuthors,
    when: all![AuthorIsNsfwUser],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const NSFW_ADMIN_AUTHOR_DROP_NSFW_MEDIA: UserRule = Rule {
    id: RuleId::NsfwAdminAuthor,
    audience: NonAuthors,
    when: all![AuthorIsNsfwAdmin],
    action: drop_post(DropAs::ContainNsfwMedia),
};

pub(super) const NSFW_HIGH_RECALL_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NsfwHighRecallUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::NsfwHighRecall)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NSFW_HIGH_PRECISION_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NsfwHighPrecisionUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::NsfwHighPrecision)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NSFW_AVATAR_IMAGE_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NsfwAvatarImageUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::NsfwAvatarImage)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NSFW_BANNER_IMAGE_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NsfwBannerImageUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::NsfwBannerImage)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NSFW_NEAR_PERFECT_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NsfwNearPerfectUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::NsfwNearPerfect)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const SPAM_HIGH_RECALL_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::SpamHighRecallUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::SpamHighRecall)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const COMPROMISED_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::CompromisedUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::Compromised)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const READ_ONLY_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::ReadOnlyUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::ReadOnly)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const IMPERSONATION_HIGH_PRECISION_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::ImpersonationHighPrecisionUserLabel,
    audience: NonAuthors,
    when: all![AuthorHasLabel(AuthorLabel::ImpersonationHighPrecision)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const ABUSIVE_HIGH_RECALL_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::AbusiveHighRecallUserLabel,
    audience: NonFollowers,
    when: all![AuthorHasLabel(AuthorLabel::AbusiveHighRecall)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const DO_NOT_AMPLIFY_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::DoNotAmplifyUserLabel,
    audience: NonFollowers,
    when: all![AuthorHasLabel(AuthorLabel::DoNotAmplify)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const NOT_GRADUATED_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::NotGraduatedUserLabel,
    audience: NonFollowers,
    when: all![AuthorHasLabel(AuthorLabel::NotGraduated)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const AGATHA_SPAM_TOP_USER_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::AgathaSpamTopUserUserLabel,
    audience: NonFollowers,
    when: all![AuthorHasLabel(AuthorLabel::AgathaSpamTopUser)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const ENGAGEMENT_SPAMMER_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::EngagementSpammerUserLabel,
    audience: NonFollowers,
    when: all![AuthorHasLabel(AuthorLabel::EngagementSpammer)],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const BLINK_WORST_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::BlinkWorstUserLabel,
    audience: NonFollowers,
    when: all![
        AuthorHasLabel(AuthorLabel::BlinkWorst),
        ViewerHasUqfEnabled,
        !ViewerHasBlinkLabel
    ],
    action: drop_post(DropAs::UnspecifiedReason),
};

pub(super) const DOWNRANK_SPAM_REPLY_USER_LABEL_DROP_UNSPECIFIED: UserRule = Rule {
    id: RuleId::DownrankSpamReplyUserLabel,
    audience: NonFollowers,
    when: all![AuthorHasLabel(AuthorLabel::DownrankSpamReply)],
    action: drop_post(DropAs::UnspecifiedReason),
};
