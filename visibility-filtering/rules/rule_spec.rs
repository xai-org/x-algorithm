use crate::hydration::Hydrators;
use crate::models::{
    AuthorLabel, DropReason, EmergencyCopy, FosnrReason, LabelInterstitial, LimitedActionType,
    LimitedEngagementReason, MediaInterstitial, MediaRestriction, Notice, NsfwViewerDropReason,
    PublicInterest, SafetyLabelMap, SafetyLabelType, TombstoneReason, ViewerProfile, Withholding,
};
use crate::params::CountryList;
use crate::rules::action_kind::ActionKind;
use crate::rules::RuleContext;
use engine::{Engine, Fold, PostEngine};
use features::{In, TweetSafetyLabels};
use inputs::Interpreter;
use rule::{NamedCond, RuleCond};
use std::fmt::{self, Debug, Formatter};
use std::ops::Not;
use xai_safety_label_store::types::label_meta;
use xai_visibility_filtering::models::{FilteredReason, SafetyResult, SafetyResultReason};
use xai_x_thrift::action::{AppealablePolicy, InterstitialAction, InterstitialReason};
use xai_x_thrift::safety_result::SafetyResultReason as ThriftSafetyResultReason;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum RuleId {
    FilterAll,

    SuspendedAuthor,
    DeactivatedAuthor,
    ErasedAuthor,
    OffboardedAuthor,
    ProtectedAuthor,
    ViewerBlocksAuthor,
    ViewerMutesAuthor,
    ViewerMutesRetweets,
    QuotedViewerMutesAuthorNotFollowing,
    QuotedAuthorBlocksQuotingAuthor,
    ProtectedQuotedAuthor,
    DeactivatedViewer,
    SuspendedViewer,
    ViewerMutesAndDoesNotFollowAuthor,
    NoConfirmedEmail,
    NoConfirmedPhone,
    NoDefaultProfileImage,
    NoNewUsers,
    NoNotFollowedBy,
    OnlyPeopleIFollow,
    OnlyPeopleIFollowOutsideCommunity,

    Pdna,
    Bounce,
    BounceQuoted,
    Spam,
    ForEmergencyUseOnly,
    PublicInterest,
    FosnrHatefulConduct,
    FosnrViolentSpeech,
    FosnrAbuse,
    FosnrCivicIntegrity,
    FosnrAbuseInsults,
    FosnrAbuseInsultsNonFollower,
    FosnrAbuseInsultsFollower,
    FosnrAuthor,
    FosnrFallback,
    NullcastedTweet,
    StaleTweet,
    LegalTakedown,
    LocalLawsTakedown,
    ArticleTweetContent,
    ProtectedCommunityTweet,
    HiddenCommunityTweet,
    AuthorRemovedCommunityTweet,

    SensitiveViewerLoggedOut,
    SensitiveViewerUnderage,
    SensitiveViewerNoStatedAge,
    NsfwSensitiveViewerTweet,
    NsfwSensitiveViewerUser,

    ExclusiveTweet,
    AuthorBlocksViewerExclusiveContent,
    CreatorTweetNsfw,
    TrustedFriendsTweet,

    NsfwHighPrecision,
    GoreAndViolenceHighPrecision,
    GoreAndViolenceIgnoringSettings,
    NsfwCardImage,
    NsfwAdmin,
    NsfwUser,
    NsfwAccount,

    OriginalTweetAdultMediaPropagation,
    OriginalTweetGoreMediaPropagation,
    ProtectedOriginalAuthor,

    DmcaMedia,
    GeoRestrictedMedia,
    NsfwUserAuthor,
    NsfwAdminAuthor,
    NsfwUserTweetFlag,
    NsfwAdminTweetFlag,
    NsfwHighRecall,
    SoftNsfwNewUser,
    DoNotAmplify,
    MaliciousUrl,
    SpamHighRecall,
    BrazilElectionLegal,
    LowQuality,
    DuplicateContent,
    SmyteSpamTweet,
    DuplicateMentionUqf,
    AbusiveUqf,
    UntrustedUrlUqf,
    DownrankSpamReplyUqf,

    NsfwHighRecallUserLabel,
    NsfwHighPrecisionUserLabel,
    NsfwAvatarImageUserLabel,
    NsfwBannerImageUserLabel,
    NsfwNearPerfectUserLabel,
    SpamHighRecallUserLabel,
    CompromisedUserLabel,
    ReadOnlyUserLabel,
    ImpersonationHighPrecisionUserLabel,
    AbusiveHighRecallUserLabel,
    DoNotAmplifyUserLabel,
    NotGraduatedUserLabel,
    AgathaSpamTopUserUserLabel,
    EngagementSpammerUserLabel,
    BlinkWorstUserLabel,
    DownrankSpamReplyUserLabel,

    BlockedViewer,
    LimitRepliesByInvitation,
    LimitRepliesCommunity,
    LimitRepliesFollowers,
    LimitRepliesSubscribers,
    LimitRepliesVerified,
    LimitRepliesMyNetwork,
    LimitRepliesCo,
    CommunityTweetViewerRemoved,
    DynamicProductAd,
    LocalTweet,
    ReadOnlyViewer,

    NotFound,
    LookupFailed,
}

pub(super) mod conditions;
pub(super) mod engine;
pub(super) mod features;
pub(super) mod inputs;
#[cfg(test)]
pub(super) mod laws;
pub(super) mod observed;
mod predicates;
pub(super) mod rule;

pub(in crate::rules) use features::leaves;
pub use features::{
    ArticleStageLeaf, AuthorFactsLeaf, AuthorLabelsLeaf, AuthorSafetyLeaf, CommunityModeratorLeaf,
    CommunityViewerRemovedLeaf, CompoundLeaf, EdgeLeaf, PostFactsLeaf, PostModerationLeaf,
    ReactedToAuthorSafetyLeaf, ReactedToPostSafetyLabelsLeaf, RelationshipLeaf, ReplyControlLeaf,
    RequestLeaf, SourceTweetIdLeaf, TweetLeaf, TweetSafetyLabelsLeaf, ViewerAccountLeaf,
};
pub use predicates::Predicate;
pub(in crate::rules) use predicates::*;

#[cfg(test)]
impl RuleId {
    pub(crate) fn of_rule_name(name: &str) -> Option<Self> {
        name.split('/').next()?.parse().ok()
    }
}

pub(super) fn rule_name<E: Engine>(id: RuleId, action: &RuleAction<E>) -> String {
    let id: &str = id.into();
    let (kind, reason, prompt) = match action.spec() {
        ActionSpec::Drop(DropReason::Legacy(reason)) => {
            let reason = match reason {
                FilteredReason::UnspecifiedReason => "unspecified".to_owned(),
                FilteredReason::PossiblyUndesirable => "undesirable".to_owned(),
                FilteredReason::ContainNsfwMedia => "nsfw_media".to_owned(),
                FilteredReason::AuthorAccountIsInactive => "inactive".to_owned(),
                reason @ (FilteredReason::AuthorBlockViewer
                | FilteredReason::AuthorIsProtected
                | FilteredReason::AuthorIsUnsafe
                | FilteredReason::ReportedTweet
                | FilteredReason::TweetMatchesViewerMutedKeyword(_)
                | FilteredReason::TweetIsBounced
                | FilteredReason::SafetyResult(_)
                | FilteredReason::AuthorIsDeactivated
                | FilteredReason::AuthorIsSuspended
                | FilteredReason::ViewerMutesAuthor
                | FilteredReason::TweetIsNullcast
                | FilteredReason::ExclusiveTweet
                | FilteredReason::ViewerBlocksAuthor) => snake(reason),
            };
            ("drop", Some(reason), None)
        }
        ActionSpec::Drop(DropReason::NsfwViewer(reason)) => {
            ("drop", Some(<&str>::from(reason).to_owned()), None)
        }
        ActionSpec::Tombstone(TombstoneReason::SensitiveViewerAgeVerification) => {
            ("tombstone", Some("age_verification".to_owned()), None)
        }
        ActionSpec::Tombstone(reason) => ("tombstone", Some(<&str>::from(reason).to_owned()), None),
        ActionSpec::MediaRestriction(MediaRestriction::MediaInterstitial(blur)) => {
            #[expect(
                clippy::wildcard_enum_match_arm,
                reason = "InterstitialReason is a generated Thrift union; every variant but PossiblyUndesirable takes its snake-cased name, including any the IDL adds"
            )]
            let reason = match &blur.reason {
                InterstitialReason::PossiblyUndesirable(_) => "undesirable".to_owned(),
                reason => snake(reason),
            };
            let prompt = blur.prompt.map(|prompt| match prompt {
                InterstitialAction::AGE_VERIFICATION_PROMPT => "age_prompt".to_owned(),
                prompt => prompt.0.to_string(),
            });
            ("blur", Some(reason), prompt)
        }
        ActionSpec::MediaRestriction(MediaRestriction::NsfwInterstitial) => {
            ("legacy_interstitial", None, None)
        }
        ActionSpec::LimitedEngagement(reason) => (
            "limited_engagement",
            Some(<&str>::from(reason).to_owned()),
            None,
        ),
        ActionSpec::SoftIntervention(notice) => {
            let policy = match E::open(*notice).policy {
                AppealablePolicy::ABUSE => "abuse".to_owned(),
                policy => policy.0.to_string(),
            };
            ("soft_intervention", Some(policy), None)
        }
        ActionSpec::Appealable(_) => ("appealable", None, None),
        ActionSpec::LabelInterstitial(_) => ("label_interstitial", None, None),
    };
    let keeps_reason = matches!(action.spec(), ActionSpec::SoftIntervention(_));
    let squashed = |name: &str| name.replace('_', "");
    let reason = reason
        .map(|reason| {
            ["author_is_", "tweet_is_", "is_", "has_"]
                .iter()
                .find_map(|prefix| reason.strip_prefix(prefix))
                .unwrap_or(&reason)
                .to_owned()
        })
        .filter(|reason| keeps_reason || !squashed(id).contains(&squashed(reason)));
    let mut name = format!("{id}/{kind}");
    for part in [reason, prompt].into_iter().flatten() {
        name.push('/');
        name.push_str(&part);
    }
    name
}

fn snake(value: &impl std::fmt::Debug) -> String {
    snake_case(format!("{value:?}").split('(').next().unwrap_or_default())
}

fn snake_case(variant: &str) -> String {
    let mut snake = String::new();
    for (index, char) in variant.chars().enumerate() {
        if char.is_uppercase() && index > 0 {
            snake.push('_');
        }
        snake.push(char.to_ascii_lowercase());
    }
    snake
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FosnrViolation {
    pub(super) label: SafetyLabelType,
    pub(super) policy: AppealablePolicy,
    pub(super) level: i8,
    pub(super) limited_actions: &'static [LimitedActionType],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FosnrNotice {
    pub(super) label: SafetyLabelType,
    pub(super) policy: AppealablePolicy,
    pub(super) level: i8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LabelInterstitialSpec {
    PublicInterest,
    EmergencyDynamic,
}

impl LabelInterstitialSpec {
    pub(super) const fn labels(self) -> &'static [SafetyLabelType] {
        match self {
            Self::PublicInterest => &PUBLIC_INTEREST_LABELS,
            Self::EmergencyDynamic => &[SafetyLabelType::FOR_EMERGENCY_USE_ONLY],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ActionSpec<E: Engine = PostEngine> {
    Drop(DropReason),
    Tombstone(TombstoneReason),
    MediaRestriction(MediaRestriction),
    SoftIntervention(E::PostOnly<FosnrNotice>),
    LimitedEngagement(LimitedEngagementReason),
    Appealable(E::PostOnly<&'static [FosnrViolation]>),
    LabelInterstitial(E::PostOnly<LabelInterstitialSpec>),
}

#[derive(Clone, PartialEq)]
pub(super) struct RuleAction<E: Engine = PostEngine> {
    spec: ActionSpec<E>,
}

impl<E: Engine> RuleAction<E> {
    pub(super) const fn spec(&self) -> &ActionSpec<E> {
        &self.spec
    }

    pub(super) fn lift(&self) -> RuleAction {
        let spec = match &self.spec {
            ActionSpec::Drop(reason) => ActionSpec::Drop(reason.clone()),
            ActionSpec::Tombstone(reason) => ActionSpec::Tombstone(*reason),
            ActionSpec::MediaRestriction(restriction) => {
                ActionSpec::MediaRestriction(restriction.clone())
            }
            ActionSpec::SoftIntervention(notice) => ActionSpec::SoftIntervention(E::open(*notice)),
            ActionSpec::LimitedEngagement(reason) => ActionSpec::LimitedEngagement(*reason),
            ActionSpec::Appealable(violations) => ActionSpec::Appealable(E::open(*violations)),
            ActionSpec::LabelInterstitial(kind) => ActionSpec::LabelInterstitial(E::open(*kind)),
        };
        RuleAction { spec }
    }
}

impl<E: Engine> Debug for RuleAction<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.spec.fmt(f)
    }
}

impl<E: Engine> Eq for RuleAction<E> {}

#[cfg(test)]
impl<E: Engine> From<ActionSpec<E>> for RuleAction<E> {
    fn from(spec: ActionSpec<E>) -> Self {
        RuleAction { spec }
    }
}

#[derive(Clone, Copy)]
pub(super) enum DropAs {
    UnspecifiedReason,
    PossiblyUndesirable,
    ContainNsfwMedia,
    AuthorAccountIsInactive,
    AuthorIsSuspended,
    AuthorIsDeactivated,
    AuthorIsProtected,
    AuthorBlockViewer,
    ViewerBlocksAuthor,
    ViewerMutesAuthor,
    TweetIsBounced,
    TweetIsNullcast,
    ExclusiveTweet,
    NsfwHighPrecisionSafetyResult,
    NsfwViewerLoggedOut,
    NsfwViewerIsUnderage,
    NsfwViewerHasNoStatedAge,
}

impl DropAs {
    const fn reason(self) -> DropReason {
        let legacy = match self {
            DropAs::UnspecifiedReason => FilteredReason::UnspecifiedReason,
            DropAs::PossiblyUndesirable => FilteredReason::PossiblyUndesirable,
            DropAs::ContainNsfwMedia => FilteredReason::ContainNsfwMedia,
            DropAs::AuthorAccountIsInactive => FilteredReason::AuthorAccountIsInactive,
            DropAs::AuthorIsSuspended => FilteredReason::AuthorIsSuspended,
            DropAs::AuthorIsDeactivated => FilteredReason::AuthorIsDeactivated,
            DropAs::AuthorIsProtected => FilteredReason::AuthorIsProtected,
            DropAs::AuthorBlockViewer => FilteredReason::AuthorBlockViewer,
            DropAs::ViewerBlocksAuthor => FilteredReason::ViewerBlocksAuthor,
            DropAs::ViewerMutesAuthor => FilteredReason::ViewerMutesAuthor,
            DropAs::TweetIsBounced => FilteredReason::TweetIsBounced,
            DropAs::TweetIsNullcast => FilteredReason::TweetIsNullcast,
            DropAs::ExclusiveTweet => FilteredReason::ExclusiveTweet,
            DropAs::NsfwHighPrecisionSafetyResult => FilteredReason::SafetyResult(SafetyResult {
                reason: Some(SafetyResultReason::NsfwHighPrecision),
                action: xai_visibility_filtering::models::Action::Drop(
                    xai_visibility_filtering::models::DropReason {},
                ),
            }),
            DropAs::NsfwViewerLoggedOut => {
                return DropReason::NsfwViewer(NsfwViewerDropReason::LoggedOut);
            }
            DropAs::NsfwViewerIsUnderage => {
                return DropReason::NsfwViewer(NsfwViewerDropReason::IsUnderage);
            }
            DropAs::NsfwViewerHasNoStatedAge => {
                return DropReason::NsfwViewer(NsfwViewerDropReason::HasNoStatedAge);
            }
        };
        DropReason::Legacy(legacy)
    }
}

pub(super) const fn drop_post<E: Engine>(reason: DropAs) -> RuleAction<E> {
    RuleAction {
        spec: ActionSpec::Drop(reason.reason()),
    }
}

pub(super) const fn tombstone<E: Engine>(reason: TombstoneReason) -> RuleAction<E> {
    RuleAction {
        spec: ActionSpec::Tombstone(reason),
    }
}

pub(super) const fn limit<E: Engine>(reason: LimitedEngagementReason) -> RuleAction<E> {
    RuleAction {
        spec: ActionSpec::LimitedEngagement(reason),
    }
}

pub(super) const fn soft_intervention(
    label: SafetyLabelType,
    policy: AppealablePolicy,
    level: i8,
) -> RuleAction {
    RuleAction {
        spec: ActionSpec::SoftIntervention(FosnrNotice {
            label,
            policy,
            level,
        }),
    }
}

pub(super) const fn appealable(violations: &'static [FosnrViolation]) -> RuleAction {
    RuleAction {
        spec: ActionSpec::Appealable(violations),
    }
}

pub(super) const fn label_interstitial(spec: LabelInterstitialSpec) -> RuleAction {
    RuleAction {
        spec: ActionSpec::LabelInterstitial(spec),
    }
}

pub(super) const fn blur<E: Engine>(reason: InterstitialReason) -> RuleAction<E> {
    media_interstitial(reason, None)
}

pub(super) const fn blur_with_age_prompt<E: Engine>(reason: InterstitialReason) -> RuleAction<E> {
    media_interstitial(reason, Some(InterstitialAction::AGE_VERIFICATION_PROMPT))
}

const fn media_interstitial<E: Engine>(
    reason: InterstitialReason,
    prompt: Option<InterstitialAction>,
) -> RuleAction<E> {
    RuleAction {
        spec: ActionSpec::MediaRestriction(MediaRestriction::MediaInterstitial(
            MediaInterstitial {
                legacy: FilteredReason::ContainNsfwMedia,
                reason,
                prompt,
            },
        )),
    }
}

pub(super) const fn legacy_nsfw_interstitial<E: Engine>() -> RuleAction<E> {
    RuleAction {
        spec: ActionSpec::MediaRestriction(MediaRestriction::NsfwInterstitial),
    }
}

#[cfg(test)]
impl<E: Engine> ActionSpec<E> {
    pub(super) fn spelled(&self) -> String {
        match self {
            ActionSpec::Drop(reason) => format!("{reason:?}"),
            ActionSpec::Tombstone(reason) => format!("{reason:?}"),
            ActionSpec::MediaRestriction(restriction) => format!("{restriction:?}"),
            ActionSpec::SoftIntervention(notice) => {
                let FosnrNotice {
                    label,
                    policy,
                    level,
                } = E::open(*notice);
                format!("{} {policy:?} level {level}", spelled_label(label))
            }
            ActionSpec::LimitedEngagement(reason) => format!("{reason:?}"),
            ActionSpec::Appealable(violations) => E::open(*violations)
                .iter()
                .map(
                    |&FosnrViolation {
                         label,
                         policy,
                         level,
                         limited_actions,
                     }| {
                        format!(
                            "{} {policy:?} level {level} limits {limited_actions:?}",
                            spelled_label(label)
                        )
                    },
                )
                .collect::<Vec<_>>()
                .join(", "),
            ActionSpec::LabelInterstitial(kind) => format!("{:?}", E::open(*kind)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Contribution {
    Withholding(Withholding),
    Media(MediaRestriction),
    Engagement(LimitedEngagementReason),
    Notice(Notice),
    LabelInterstitial(LabelInterstitial),
}

impl<E: Fold> ActionSpec<E> {
    #[inline]
    pub(super) fn interpret<'a, I: Interpreter<'a, E::Candidate>>(
        &self,
        interpreter: &I,
    ) -> I::Output<Option<Contribution>> {
        match self {
            Self::Drop(reason) => interpreter
                .run(|()| Some(Contribution::Withholding(Withholding::Drop(reason.clone())))),
            Self::Tombstone(reason) => interpreter
                .run(|()| Some(Contribution::Withholding(Withholding::Tombstone(*reason)))),
            Self::MediaRestriction(restriction) => {
                interpreter.run(|()| Some(Contribution::Media(restriction.clone())))
            }
            Self::LimitedEngagement(reason) => {
                interpreter.run(|()| Some(Contribution::Engagement(*reason)))
            }
            Self::SoftIntervention(notice) => E::run_post(
                interpreter,
                *notice,
                |FosnrNotice {
                     label,
                     policy,
                     level,
                 },
                 In::<TweetSafetyLabels>(labels)| {
                    Some(Contribution::Notice(Notice::SoftIntervention(
                        fosnr_reason(label, policy, level, labels),
                    )))
                },
            ),
            Self::Appealable(violations) => E::run_post(
                interpreter,
                *violations,
                |violations, In::<TweetSafetyLabels>(labels)| {
                    appeal(violations, labels).map(Contribution::Notice)
                },
            ),
            Self::LabelInterstitial(kind) => E::run_post(
                interpreter,
                *kind,
                |kind, In::<TweetSafetyLabels>(labels)| {
                    match kind {
                        LabelInterstitialSpec::PublicInterest => public_interest(labels),
                        LabelInterstitialSpec::EmergencyDynamic => emergency_dynamic(labels),
                    }
                    .map(Contribution::LabelInterstitial)
                },
            ),
        }
    }

    pub(super) fn requirements(&self) -> Hydrators {
        self.interpret(&inputs::Required)
    }
}

impl<E: Engine> ActionSpec<E> {
    pub(super) const fn kind(&self) -> ActionKind {
        match self {
            Self::Appealable(_) => ActionKind::Appealable,
            Self::Drop(_) => ActionKind::Drop,
            Self::Tombstone(_) => ActionKind::Tombstone,
            Self::LabelInterstitial(_) => ActionKind::LabelInterstitial,
            Self::MediaRestriction(_) => ActionKind::MediaRestriction,
            Self::SoftIntervention(_) => ActionKind::SoftIntervention,
            Self::LimitedEngagement(_) => ActionKind::LimitedEngagement,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Truth {
    True,
    False,
    Unknown { default: bool, failed: Hydrators },
}

impl Truth {
    #[inline]
    pub(crate) fn resolves_true(self) -> bool {
        matches!(self, Truth::True | Truth::Unknown { default: true, .. })
    }

    #[inline]
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Truth::False, _) | (_, Truth::False) => Truth::False,
            (Truth::True, other) | (other, Truth::True) => other,
            (
                Truth::Unknown {
                    default: a,
                    failed: x,
                },
                Truth::Unknown {
                    default: b,
                    failed: y,
                },
            ) => Truth::Unknown {
                default: a && b,
                failed: x.union(y),
            },
        }
    }

    #[inline]
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Truth::True, _) | (_, Truth::True) => Truth::True,
            (Truth::False, other) | (other, Truth::False) => other,
            (
                Truth::Unknown {
                    default: a,
                    failed: x,
                },
                Truth::Unknown {
                    default: b,
                    failed: y,
                },
            ) => Truth::Unknown {
                default: a || b,
                failed: x.union(y),
            },
        }
    }
}

impl Not for Truth {
    type Output = Self;

    #[inline]
    fn not(self) -> Self {
        match self {
            Truth::True => Truth::False,
            Truth::False => Truth::True,
            Truth::Unknown { default, failed } => Truth::Unknown {
                default: !default,
                failed,
            },
        }
    }
}

fn appeal(violations: &[FosnrViolation], labels: &SafetyLabelMap) -> Option<Notice> {
    let violation = violations
        .iter()
        .find(|violation| labels.has_label(violation.label))?;
    Some(Notice::Appealable {
        reason: fosnr_reason(violation.label, violation.policy, violation.level, labels),
        limited_actions: violation.limited_actions,
    })
}

fn fosnr_reason(
    label: SafetyLabelType,
    policy: AppealablePolicy,
    level: i8,
    labels: &SafetyLabelMap,
) -> FosnrReason {
    FosnrReason {
        policy,
        level,
        proactive: !labels.is_by_agent(label),
        appeal_submitted: labels.has_label(SafetyLabelType::FOSNR_APPEAL_SUBMITTED),
    }
}

const PUBLIC_INTEREST_LABELS: [SafetyLabelType; 6] = [
    SafetyLabelType::LOW_QUALITY,
    SafetyLabelType::MISINFO_CIVIC,
    SafetyLabelType::MISINFO_GENERIC,
    SafetyLabelType::MISINFO_MEDICAL,
    SafetyLabelType::MISINFO_CRISIS,
    SafetyLabelType::IPI_DEVELOPMENT_ONLY,
];

macro_rules! named {
    ($($ident:ident),+ $(,)?) => {
        [$((stringify!($ident), ThriftSafetyResultReason::$ident)),+]
    };
}

const PUBLIC_INTEREST_REASONS: [(&str, ThriftSafetyResultReason); 30] = named![
    EPISODIC,
    ABUSE_EPISODIC_ENCOURAGE_SELF_HARM,
    ABUSE_EPISODIC_HATEFUL_CONDUCT,
    ABUSE_GLORIFICATION_OF_VIOLENCE,
    ABUSE_GRATUITOUS_GORE,
    ABUSE_MOB_HARASSMENT,
    ABUSE_MOMENT_OF_DEATH_OR_DECEASED_USER,
    ABUSE_PRIVATE_INFORMATION,
    ABUSE_RIGHT_TO_PRIVACY,
    ABUSE_THREAT_TO_EXPOSE,
    ABUSE_VIOLENT_SEXUAL_CONDUCT,
    ABUSE_VIOLENT_THREAT_HATEFUL_CONDUCT,
    ABUSE_VIOLENT_THREAT_OR_BOUNTY,
    ONE_OFF,
    VOTING_MISINFORMATION,
    HACKED_MATERIALS,
    SCAMS,
    PLATFORM_MANIPULATION,
    MISINFO_CIVIC,
    MISINFO_CRISIS,
    MISINFO_GENERIC,
    MISINFO_MEDICAL,
    DEVELOPMENT_ONLY_PUBLIC_INTEREST,
    NSFW_HIGH_PRECISION,
    GORE_AND_VIOLENCE_HIGH_PRECISION,
    FOSNR_SYNTHETIC_AND_MANIPULATED_MEDIA,
    FOSNR_CIVIC_INTEGRITY,
    FOSNR_ABUSE,
    FOSNR_HATEFUL_CONDUCT,
    FOSNR_VIOLENT_SPEECH,
];

fn spells(source: &str, ident: &str) -> bool {
    source
        .chars()
        .map(|letter| letter.to_ascii_uppercase())
        .eq(ident.chars().filter(|&letter| letter != '_'))
}

fn public_interest(labels: &SafetyLabelMap) -> Option<LabelInterstitial> {
    let label = *LabelInterstitialSpec::PublicInterest
        .labels()
        .iter()
        .find(|&&label| labels.has_label(label))?;
    let source = labels.source(label);
    let misinfo = |reason| PublicInterest {
        reason,
        limit: source.and_then(LimitedEngagementReason::from_misinfo_source),
    };
    let notice = match label {
        SafetyLabelType::LOW_QUALITY => PublicInterest {
            reason: source
                .and_then(|source| {
                    PUBLIC_INTEREST_REASONS
                        .into_iter()
                        .find(|(ident, _)| spells(source, ident))
                })
                .map_or(ThriftSafetyResultReason::ONE_OFF, |(_, reason)| reason),
            limit: Some(LimitedEngagementReason::NonCompliant),
        },
        SafetyLabelType::MISINFO_CIVIC => misinfo(ThriftSafetyResultReason::MISINFO_CIVIC),
        SafetyLabelType::MISINFO_GENERIC => misinfo(ThriftSafetyResultReason::MISINFO_GENERIC),
        SafetyLabelType::MISINFO_MEDICAL => misinfo(ThriftSafetyResultReason::MISINFO_MEDICAL),
        SafetyLabelType::MISINFO_CRISIS => misinfo(ThriftSafetyResultReason::MISINFO_CRISIS),
        _ => PublicInterest {
            reason: ThriftSafetyResultReason::DEVELOPMENT_ONLY_PUBLIC_INTEREST,
            limit: Some(LimitedEngagementReason::NonCompliant),
        },
    };
    Some(LabelInterstitial::PublicInterest(notice))
}

fn emergency_dynamic(labels: &SafetyLabelMap) -> Option<LabelInterstitial> {
    let source = labels.source(SafetyLabelType::FOR_EMERGENCY_USE_ONLY)?;
    let names_a_system = ["bot_id_", "Abuse", "hse"]
        .iter()
        .any(|prefix| source.starts_with(prefix))
        || [
            "A",
            "B",
            "AB",
            "OneOff",
            "VotingMisinformation",
            "HackedMaterials",
            "Scams",
            "PlatformManipulation",
        ]
        .contains(&source);
    if names_a_system {
        return None;
    }
    let (copy, link) = source
        .split_once('|')
        .map_or((source, None), |(copy, link)| {
            (copy, Some(link).filter(|link| !link.is_empty()))
        });
    Some(LabelInterstitial::EmergencyDynamic(Box::new(
        EmergencyCopy {
            copy: copy.to_owned(),
            link: link.map(str::to_owned),
        },
    )))
}

fn account_or_request_country_in(
    profile: Option<&ViewerProfile>,
    request_country: Option<&str>,
    in_list: impl FnOnce(&str) -> bool,
) -> bool {
    profile
        .and_then(|profile| profile.account_country_code.as_deref())
        .or(request_country)
        .is_some_and(in_list)
}

#[cfg(test)]
mod tests {
    use super::leaves::*;
    use super::*;
    use crate::hydration::Hydrator;
    use crate::models::{
        ClientCapability, ClientSwitch, PostCandidate, RequestFacts, VerifyBlurSupport, Viewer,
        ViewerFeatures,
    };
    use crate::rules::fixtures::{
        candidate, logged_out_viewer, viewer, viewer_with_profile, VIEWER_ID,
    };
    use crate::rules::test_context;

    impl Predicate {
        fn holds_for(self, viewer: &ViewerFeatures, candidate: &PostCandidate) -> bool {
            self.holds(&test_context(viewer, candidate).hydrated_by(self.requirements()))
        }
    }

    #[test]
    fn created_after_is_strict_at_the_unix_millisecond_boundary() {
        const CUTOFF_MS: u64 = 1705536000000;
        const AT_CUTOFF: u64 = (CUTOFF_MS - 1288834974657) << 22;
        let viewer = ViewerFeatures::default();
        for (tweet_id, expected) in [
            (AT_CUTOFF - 1, false),
            (AT_CUTOFF, false),
            (AT_CUTOFF + (1 << 22) - 1, false),
            (AT_CUTOFF + (1 << 22), true),
            (0, false),
        ] {
            let mut candidate = PostCandidate::default();
            candidate.post.tweet_id = tweet_id;
            let context = test_context(&viewer, &candidate);
            assert_eq!(
                Predicate::PostFacts(TweetCreatedAfter(CUTOFF_MS)).holds(&context),
                expected,
                "{tweet_id}"
            );
        }
    }

    #[test]
    fn age_verified_is_the_label_or_idv_premium_and_the_request_country_is_never_unknown() {
        let in_country = |country: Option<&str>, profile: ViewerProfile| ViewerFeatures {
            facts: country.map_or_else(RequestFacts::default, |country| {
                RequestFacts::default().with_country(country)
            }),
            ..viewer_with_profile(profile)
        };
        let labeled = ViewerFeatures {
            viewer: Viewer::LoggedIn {
                id: VIEWER_ID,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: true,
            },
            ..ViewerFeatures::default()
        };
        let idv_premium = ViewerProfile {
            has_idv_premium: true,
            ..ViewerProfile::default()
        };
        let account_in_fr = ViewerProfile {
            account_country_code: Some("fr".into()),
            ..ViewerProfile::default()
        };
        let request_country_in = Predicate::Request(RequestCountryIn(CountryList::AgeVerification));
        let age_verified = Predicate::Compound(ViewerAgeVerified);
        let candidate = candidate().build();
        for (predicate, viewer, expected) in [
            (age_verified, labeled, true),
            (age_verified, in_country(None, idv_premium), true),
            (
                age_verified,
                in_country(None, ViewerProfile::default()),
                false,
            ),
            (age_verified, logged_out_viewer(), false),
            (
                request_country_in,
                in_country(Some("fr"), ViewerProfile::default()),
                true,
            ),
            (
                request_country_in,
                in_country(Some("us"), account_in_fr),
                false,
            ),
            (
                request_country_in,
                in_country(None, ViewerProfile::default()),
                false,
            ),
        ] {
            assert_eq!(predicate.holds_for(&viewer, &candidate), expected);
        }

        let mut failed = candidate;
        failed.failed = Hydrators::all();
        let viewer = in_country(Some("fr"), ViewerProfile::default());
        let context = test_context(&viewer, &failed);
        assert_eq!(
            age_verified.truth(&context),
            Truth::Unknown {
                default: false,
                failed: Hydrators::of(Hydrator::ViewerProfile).with(Hydrator::ViewerLabels),
            }
        );
        assert_eq!(request_country_in.truth(&context), Truth::True);
    }

    #[test]
    fn the_client_checks_read_the_resolved_capability_and_are_never_unknown() {
        use strum::VariantArray;
        use VerifyBlurSupport::{IosNeedsUpdate, Supported};
        let viewer = ViewerFeatures {
            facts: RequestFacts::default().with_client(ClientCapability {
                verify_blur_support: Some(IosNeedsUpdate),
                switches_on: ClientSwitch::VARIANTS.iter().copied().collect(),
            }),
            ..viewer(VIEWER_ID)
        };
        let mut failed = candidate().build();
        failed.failed = Hydrators::all();
        let context = test_context(&viewer, &failed);
        for (support, expected) in [(IosNeedsUpdate, Truth::True), (Supported, Truth::False)] {
            let check = Predicate::Request(ClientVerifyBlurSupportIs(support));
            assert_eq!(check.truth(&context), expected, "{support:?}");
        }
        for switch in [
            ClientSwitch::ModernBlur,
            ClientSwitch::CommunityViewerRemovedLimits,
        ] {
            let check = Predicate::Request(ClientHas(switch));
            assert_eq!(check.truth(&context), Truth::True, "{switch:?}");
        }
    }

    #[tokio::test]
    async fn a_fosnr_label_an_agent_applied_reads_from_twemcache_as_not_proactive() {
        use crate::models::{Evaluation, Placement};
        use crate::rules::fixtures::{appealed, AUTHOR_ID};
        use crate::rules::{RuleEngine, SafetyLevel};
        use xai_safety_label_source::testing::FakeLabels;

        let captured = std::fs::read(crate::pinned::crate_path(
            "tests/fixtures/twemcache_agent_fosnr.mval",
        ))
        .unwrap();
        let mut post = candidate().build();
        let labels = FakeLabels::default()
            .cached(post.post.tweet_id, captured)
            .into_source()
            .get(&[post.post.tweet_id])
            .await
            .remove(&post.post.tweet_id)
            .and_then(Result::ok)
            .expect("twemcache holds the post's labels");
        post.post.safety_labels = SafetyLabelMap::from_proto_label_types(&labels);

        assert_eq!(
            RuleEngine::for_tests().evaluate(
                SafetyLevel::TimelineHomeHydration,
                &viewer(AUTHOR_ID),
                &post,
                Placement::Standalone,
            ),
            Evaluation::complete(appealed(
                AppealablePolicy::VIOLENT_SPEECH,
                3,
                false,
                false,
                "fosnr_author/appealable",
            ))
        );
    }
}
