use crate::hydration::Lookup;
use crate::models::{
    Decided, Deletion, DropReason, EmergencyCopy, Evaluation, FosnrReason, LabelInterstitial,
    LimitedActionType, LimitedEngagement, LimitedEngagementReason, MediaInterstitial,
    MediaRestriction, Notice, NsfwViewerDropReason, PublicInterest, TombstoneReason, Verdict,
    VerdictAction, Withholding,
};
use crate::params::limited_actions_copy::{Prompt, PromptKind, LEARN_MORE_PLACEHOLDER};
use crate::params::LimitedActionsPolicies;
use crate::post_evaluator::PostEvaluation;
use crate::quote::Rerendered;
use crate::result_label::{label, state_label, Class, Label, NONE};
use crate::rules::{RerenderReason, RuleId, SafetyLevel};
use std::sync::Arc;
use vf_pb::tweet_evaluation::Outcome as TweetEvaluationOutcome;
use vf_pb::NotificationDropReason;
use xai_visibility_filtering::graphql_results::translate;
use xai_visibility_filtering::links;
use xai_visibility_filtering::models::FilteredReason;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::action::{
    self, Action, AgeVerificationOption, AnyInterstitial, Appealable, AppealablePolicy,
    AppealableReason, Avoid, BasicLimitedActionPrompt, BlurredImageInterstitial,
    BrandSafetyRiskLevel, ComposedMediaVisibilityActions, CtaLimitedActionPrompt,
    EmergencyDynamicInterstitial, Interstitial, InterstitialAction, InterstitialLimitedEngagements,
    InterstitialReason, LimitedAction, LimitedActionCtaType, LimitedActionPrompt,
    LimitedActionsPolicy, LimitedEngagements, LocalizedMessage,
    LocalizedMessageLimitedActionPrompt, MediaAction, MediaInterstitial as ThriftMediaInterstitial,
    MessageLink, SoftIntervention as ThriftSoftIntervention, SoftInterventionDisplayType,
    SoftInterventionReason, Tombstone, TweetInterstitial,
};
use xai_x_thrift::safety_result::{
    FilteredReason as ThriftFilteredReason, SafetyResult as ThriftSafetyResult, SafetyResultReason,
};
use xai_x_thrift::tweet_service::{
    TweetFieldsResultFiltered, TweetFieldsResultFound, TweetFieldsResultNotFound,
    TweetFieldsResultState,
};

pub(crate) fn thrift_action(
    verdict: &Verdict,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> Action {
    match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(reason),
            ..
        }) => Action::Drop(action::Drop::new(drop_reason(reason, level), None)),
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(reason),
            ..
        }) => Action::Tombstone(Tombstone::new(Some(tombstone_reason(*reason)), None)),
        Verdict::Fosnr(Decided {
            value: Notice::SoftIntervention(reason),
            ..
        }) => Action::TweetInterstitial(TweetInterstitial {
            soft_intervention: Some(soft_intervention(reason)),
            avoid: Some(normal_avoid()),
            ..TweetInterstitial::default()
        }),
        Verdict::Fosnr(Decided {
            value:
                Notice::Appealable {
                    reason,
                    limited_actions,
                },
            ..
        }) => Action::TweetInterstitial(TweetInterstitial {
            limited_engagements: Some(fosnr_limited_engagements(reason, limited_actions)),
            avoid: Some(normal_avoid()),
            appealable: Some(Appealable::new(appealable_reason(reason), None)),
            ..TweetInterstitial::default()
        }),
        Verdict::Shown {
            label_interstitial:
                Some(Decided {
                    value: interstitial,
                    ..
                }),
            media,
            engagement: _,
        } => {
            let mut reasons = verdict.limited_engagement_reasons();
            let limit = reasons.next().map(|first| {
                let mut limit = LimitedEngagement::new(first);
                for reason in reasons {
                    limit.add(reason);
                }
                limit
            });
            let (media_action, all_media_visibility_results) = match media {
                Some(Decided {
                    value: MediaRestriction::MediaInterstitial(blur),
                    ..
                }) => (None, Some(blurred_media(blur))),
                Some(Decided {
                    value: MediaRestriction::NsfwInterstitial,
                    ..
                }) => (Some(MediaAction::Interstitial(nsfw_interstitial())), None),
                None => (None, None),
            };
            Action::TweetInterstitial(TweetInterstitial {
                interstitial: Some(any_interstitial(interstitial, policies.language())),
                limited_engagements: limit.map(|limit| limited_engagements(&limit, policies)),
                media_action,
                all_media_visibility_results,
                ..TweetInterstitial::default()
            })
        }
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: None,
        } => Action::Allow(action::Allow::new()),
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: Some(Decided { value, .. }),
        } => Action::LimitedEngagements(limited_engagements(value, policies)),
        Verdict::Shown {
            label_interstitial: None,
            media: Some(Decided { value, .. }),
            engagement: None,
        } => match value {
            MediaRestriction::MediaInterstitial(blur) => {
                Action::ComposedMediaVisibilityResults(blurred_media(blur))
            }
            MediaRestriction::NsfwInterstitial => Action::Interstitial(nsfw_interstitial()),
        },
        Verdict::Shown {
            label_interstitial: None,
            media: Some(media),
            engagement: Some(limit),
        } => {
            let (interstitial, all_media_visibility_results) = match &media.value {
                MediaRestriction::MediaInterstitial(blur) => (None, Some(blurred_media(blur))),
                MediaRestriction::NsfwInterstitial => (
                    Some(AnyInterstitial::Interstitial(nsfw_interstitial())),
                    None,
                ),
            };
            Action::TweetInterstitial(TweetInterstitial {
                interstitial,
                limited_engagements: Some(limited_engagements(&limit.value, policies)),
                all_media_visibility_results,
                ..TweetInterstitial::default()
            })
        }
    }
}

pub(crate) fn rerendered_state(rerendered: &Rerendered) -> TweetFieldsResultState {
    use ThriftFilteredReason as T;
    let reason = match rerendered.reason {
        RerenderReason::AuthorDeactivated => T::AuthorIsDeactivated(true),
        RerenderReason::AuthorInactive => T::AuthorAccountIsInactive(true),
        RerenderReason::AuthorSuspended => T::AuthorIsSuspended(true),
        RerenderReason::AuthorBlocksViewer => T::AuthorBlockViewer(true),
        RerenderReason::AuthorProtected => T::AuthorIsProtected(true),
        RerenderReason::Unspecified => T::UnspecifiedReason(true),
    };
    TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason))
}

pub(crate) fn thrift_result_state(
    verdict: &Verdict,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> TweetFieldsResultState {
    let reason = match verdict {
        Verdict::Fosnr(Decided {
            value: Notice::Appealable { reason, .. },
            ..
        }) => Some(fosnr_safety_result_reason(reason)),
        Verdict::Shown {
            label_interstitial:
                Some(Decided {
                    value: LabelInterstitial::PublicInterest(interstitial),
                    ..
                }),
            ..
        } => Some(interstitial.reason),
        Verdict::Withheld(_) | Verdict::Fosnr(_) | Verdict::Shown { .. } => None,
    };
    let safety_result = || {
        ThriftFilteredReason::SafetyResult(ThriftSafetyResult::new(
            reason,
            thrift_action(verdict, level, policies),
        ))
    };
    let found = |reason| TweetFieldsResultState::Found(TweetFieldsResultFound::new(reason));
    let filtered =
        |reason| TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason));
    match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(reason)),
            ..
        }) => filtered(legacy_drop_reason(reason).unwrap_or_else(safety_result)),
        Verdict::Withheld(_) => filtered(safety_result()),
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: None,
        } => found(None),
        Verdict::Shown {
            label_interstitial: None,
            media:
                Some(Decided {
                    value: MediaRestriction::NsfwInterstitial,
                    ..
                }),
            engagement: None,
        } => found(Some(ThriftFilteredReason::ContainNsfwMedia(true))),
        Verdict::Fosnr(_) | Verdict::Shown { .. } => found(Some(safety_result())),
    }
}

fn legacy_drop_reason(reason: &FilteredReason) -> Option<ThriftFilteredReason> {
    use ThriftFilteredReason as T;
    Some(match reason {
        FilteredReason::TweetIsBounced => T::TweetIsBounced(true),
        FilteredReason::AuthorBlockViewer => T::AuthorBlockViewer(true),
        FilteredReason::AuthorIsProtected => T::AuthorIsProtected(true),
        FilteredReason::AuthorIsSuspended => T::AuthorIsSuspended(true),
        FilteredReason::ContainNsfwMedia
        | FilteredReason::PossiblyUndesirable
        | FilteredReason::UnspecifiedReason
        | FilteredReason::AuthorAccountIsInactive
        | FilteredReason::AuthorIsUnsafe
        | FilteredReason::ReportedTweet
        | FilteredReason::TweetMatchesViewerMutedKeyword(_)
        | FilteredReason::SafetyResult(_)
        | FilteredReason::AuthorIsDeactivated
        | FilteredReason::ViewerMutesAuthor
        | FilteredReason::TweetIsNullcast
        | FilteredReason::ExclusiveTweet
        | FilteredReason::ViewerBlocksAuthor => return None,
    })
}

fn normal_avoid() -> Avoid {
    Avoid::new(None, Some(BrandSafetyRiskLevel::NORMAL), None)
}

fn appealable_reason(reason: &FosnrReason) -> AppealableReason {
    AppealableReason::new(
        reason.level,
        Box::new(reason.policy),
        reason.proactive,
        reason.appeal_submitted,
    )
}

fn fosnr_safety_result_reason(reason: &FosnrReason) -> SafetyResultReason {
    match reason.policy {
        AppealablePolicy::HATEFUL_CONDUCT => SafetyResultReason::FOSNR_HATEFUL_CONDUCT,
        AppealablePolicy::ABUSE => SafetyResultReason::FOSNR_ABUSE,
        AppealablePolicy::VIOLENT_SPEECH => SafetyResultReason::FOSNR_VIOLENT_SPEECH,
        AppealablePolicy::CIVIC_INTEGRITY => SafetyResultReason::FOSNR_CIVIC_INTEGRITY,
        AppealablePolicy::SYNTHETIC_AND_MANIPULATED_MEDIA => {
            SafetyResultReason::FOSNR_SYNTHETIC_AND_MANIPULATED_MEDIA
        }
        _ => SafetyResultReason::FOSNR_UNSPECIFIED,
    }
}

fn fosnr_limited_engagements(
    reason: &FosnrReason,
    limited_actions: &[LimitedActionType],
) -> LimitedEngagements {
    LimitedEngagements::new(
        action::LimitedEngagementReason::FosnrReason(appealable_reason(reason)),
        LimitedActionsPolicy::new(
            limited_actions
                .iter()
                .map(|&action_type| LimitedAction::new(limited_action_type(action_type), None))
                .collect(),
        ),
        (reason.level != 1).then(|| "freedom_of_speech_not_reach".to_string()),
    )
}

fn soft_intervention(reason: &FosnrReason) -> ThriftSoftIntervention {
    ThriftSoftIntervention {
        soft_intervention_reason: Some(SoftInterventionReason::Fosnr(Box::new(appealable_reason(
            reason,
        )))),
        engagement_nudge: Some(false),
        suppress_autoplay: Some(true),
        warning: None,
        details_url: None,
        display_type: Some(SoftInterventionDisplayType::FOSNR),
    }
}

fn any_interstitial(interstitial: &LabelInterstitial, language: &str) -> AnyInterstitial {
    match interstitial {
        LabelInterstitial::PublicInterest(PublicInterest { reason, limit }) => {
            let (resname, link) = public_interest_copy(*reason);
            AnyInterstitial::InterstitialLimitedEngagements(InterstitialLimitedEngagements::new(
                limit.map(limited_engagement_reason),
                localized_message(
                    &translate(resname, language)
                        .unwrap_or_default()
                        .replace("%{", "{"),
                    Some(link),
                    language,
                ),
                None,
            ))
        }
        LabelInterstitial::EmergencyDynamic(emergency) => {
            let EmergencyCopy { copy, link } = &**emergency;
            let message = match link {
                Some(_) => format!("{copy} {{{LEARN_MORE_PLACEHOLDER}}}"),
                None => copy.clone(),
            };
            AnyInterstitial::EmergencyDynamicInterstitial(EmergencyDynamicInterstitial::new(
                copy.clone(),
                link.clone(),
                Box::new(localized_message(&message, link.as_deref(), language)),
            ))
        }
    }
}

fn localized_message(text: &str, link: Option<&str>, language: &str) -> LocalizedMessage {
    let links: Vec<MessageLink> = link
        .into_iter()
        .map(|url| {
            MessageLink::new(
                LEARN_MORE_PLACEHOLDER.to_string(),
                translate("learn_more_link", language)
                    .unwrap_or("Learn more")
                    .to_string(),
                url.to_string(),
            )
        })
        .collect();
    LocalizedMessage::new(text.to_string(), language.to_string(), links)
}

fn public_interest_copy(reason: SafetyResultReason) -> (&'static str, &'static str) {
    use SafetyResultReason as R;
    let resname = match reason {
        R::EPISODIC => "public_interest.episodic",
        R::ABUSE_EPISODIC_ENCOURAGE_SELF_HARM => "public_interest.self_harm",
        R::ABUSE_EPISODIC_HATEFUL_CONDUCT => "public_interest.hateful_conduct",
        R::ABUSE_GRATUITOUS_GORE => "public_interest.gratuitous_gore",
        R::ABUSE_GLORIFICATION_OF_VIOLENCE => "public_interest.glorifying_violence",
        R::ABUSE_MOB_HARASSMENT => "public_interest.mob_harassment",
        R::ABUSE_MOMENT_OF_DEATH_OR_DECEASED_USER => "public_interest.moment_of_death",
        R::ABUSE_PRIVATE_INFORMATION => "public_interest.private_information",
        R::ABUSE_RIGHT_TO_PRIVACY => "public_interest.right_to_privacy",
        R::ABUSE_THREAT_TO_EXPOSE => "public_interest.threat_to_expose",
        R::ABUSE_VIOLENT_SEXUAL_CONDUCT => "public_interest.violent_sexual_conduct",
        R::ABUSE_VIOLENT_THREAT_HATEFUL_CONDUCT | R::ABUSE_VIOLENT_THREAT_OR_BOUNTY => {
            "public_interest.violent_threat"
        }
        R::VOTING_MISINFORMATION => "public_interest.voting_misinformation",
        R::HACKED_MATERIALS => "public_interest.hacked_materials",
        R::SCAMS => "public_interest.scams",
        R::PLATFORM_MANIPULATION => "public_interest.platform_manipulation",
        R::MISINFO_CIVIC => "misinfo.civic",
        R::MISINFO_CRISIS => "misinfo.crisis",
        R::MISINFO_GENERIC => "misinfo.generic",
        R::MISINFO_MEDICAL => "misinfo.medical",
        R::DEVELOPMENT_ONLY_PUBLIC_INTEREST => "public_interest.development_only",
        R::NSFW_HIGH_PRECISION => "public_interest.nsfw_high_precision",
        R::GORE_AND_VIOLENCE_HIGH_PRECISION => "public_interest.gore_and_violence_high_precision",
        R::FOSNR_SYNTHETIC_AND_MANIPULATED_MEDIA => "fosnr.synthetic_and_manipulated_media",
        R::FOSNR_CIVIC_INTEGRITY => "fosnr.civic_integrity",
        R::FOSNR_ABUSE => "fosnr.abusive_behavior",
        R::FOSNR_HATEFUL_CONDUCT => "fosnr.hateful_conduct",
        R::FOSNR_VIOLENT_SPEECH => "fosnr.violent_speech",
        _ => "public_interest.one_off",
    };
    let link = match reason {
        R::MISINFO_CIVIC => links::ELECTION_INTEGRITY,
        R::MISINFO_CRISIS => links::CRISIS_MISINFORMATION,
        R::MISINFO_GENERIC => links::TWITTER_RULES,
        _ => links::PUBLIC_INTEREST,
    };
    (resname, link)
}

fn nsfw_interstitial() -> Interstitial {
    Interstitial::new(InterstitialReason::ContainsNsfwMedia(true), None)
}

fn limited_engagements(
    limit: &LimitedEngagement,
    policies: &LimitedActionsPolicies,
) -> LimitedEngagements {
    let mut actions: Vec<LimitedAction> = Vec::new();
    for action in limit.reasons().flat_map(|reason| policies.actions(reason)) {
        let action_type = limited_action_type(action.action_type);
        if !actions
            .iter()
            .any(|kept| kept.limited_action_type == action_type)
        {
            let prompt = action.prompt.as_ref().map(limited_action_prompt);
            actions.push(LimitedAction::new(action_type, prompt));
        }
    }
    LimitedEngagements::new(
        limited_engagement_reason(limit.reason()),
        (!actions.is_empty()).then(|| LimitedActionsPolicy::new(actions)),
        limit.reason().limited_actions_string().to_string(),
    )
}

fn limited_action_prompt(prompt: &Prompt) -> LimitedActionPrompt {
    let headline = prompt.headline.clone();
    let subtext = prompt.subtext.clone();
    match &prompt.kind {
        PromptKind::Basic => LimitedActionPrompt::BasicLimitedActionPrompt(
            BasicLimitedActionPrompt::new(headline, subtext),
        ),
        PromptKind::SeeConversation => LimitedActionPrompt::CtaLimitedActionPrompt(
            CtaLimitedActionPrompt::new(headline, subtext, LimitedActionCtaType::SEE_CONVERSATION),
        ),
        PromptKind::LearnMore {
            language,
            link_text,
            url,
        } => LimitedActionPrompt::LocalizedMessageLimitedActionPrompt(
            LocalizedMessageLimitedActionPrompt::new(
                LocalizedMessage::new(headline, language.clone(), Vec::new()),
                LocalizedMessage::new(
                    subtext,
                    language.clone(),
                    vec![MessageLink::new(
                        LEARN_MORE_PLACEHOLDER.to_string(),
                        link_text.to_string(),
                        url.clone(),
                    )],
                ),
            ),
        ),
    }
}

fn limited_action_type(action_type: LimitedActionType) -> action::LimitedActionType {
    use action::LimitedActionType as T;
    use LimitedActionType as L;
    match action_type {
        L::Reply => T::REPLY,
        L::Retweet => T::RETWEET,
        L::QuoteTweet => T::QUOTE_TWEET,
        L::Like => T::LIKE,
        L::React => T::REACT,
        L::SendViaDm => T::SEND_VIA_DM,
        L::AddToBookmarks => T::ADD_TO_BOOKMARKS,
        L::AddToMoment => T::ADD_TO_MOMENT,
        L::PinToProfile => T::PIN_TO_PROFILE,
        L::ViewTweetActivity => T::VIEW_TWEET_ACTIVITY,
        L::ShareTweetVia => T::SHARE_TWEET_VIA,
        L::Follow => T::FOLLOW,
        L::ListsAddRemove => T::LISTS_ADD_REMOVE,
        L::MuteConversation => T::MUTE_CONVERSATION,
        L::Embed => T::EMBED,
        L::ViewHiddenReplies => T::VIEW_HIDDEN_REPLIES,
        L::HideCommunityTweet => T::HIDE_COMMUNITY_TWEET,
        L::CopyLink => T::COPY_LINK,
        L::VoteOnPoll => T::VOTE_ON_POLL,
        L::RemoveFromCommunity => T::REMOVE_FROM_COMMUNITY,
        L::ShowRetweetActionMenu => T::SHOW_RETWEET_ACTION_MENU,
        L::ReplyDownVote => T::REPLY_DOWN_VOTE,
        L::Autoplay => T::AUTOPLAY,
        L::EditTweet => T::EDIT_TWEET,
        L::Highlight => T::HIGHLIGHT,
        L::ViewPostEngagements => T::VIEW_POST_ENGAGEMENTS,
    }
}

const AGE_VERIFICATION_OPTIONS: [AgeVerificationOption; 2] = [
    AgeVerificationOption::SELFIE,
    AgeVerificationOption::PERSONA,
];

fn blurred_media(blur: &MediaInterstitial) -> ComposedMediaVisibilityActions {
    let verification_options = (blur.prompt == Some(InterstitialAction::AGE_VERIFICATION_PROMPT))
        .then(|| AGE_VERIFICATION_OPTIONS.to_vec());
    ComposedMediaVisibilityActions {
        media_interstitial: Some(Box::new(ThriftMediaInterstitial::BlurredImageInterstitial(
            BlurredImageInterstitial {
                reason: Some(blur.reason.clone()),
                opacity: Some(0.8.into()),
                interstitial_action: blur.prompt,
                available_verification_options: verification_options,
            },
        ))),
    }
}

fn drop_reason(reason: &DropReason, level: SafetyLevel) -> Option<action::DropReason> {
    let legacy = match reason {
        DropReason::Legacy(legacy) => legacy,
        DropReason::NsfwViewer(reason) => {
            return Some(match reason {
                NsfwViewerDropReason::IsUnderage => action::DropReason::NsfwViewerIsUnderage(true),
                NsfwViewerDropReason::HasNoStatedAge => {
                    action::DropReason::NsfwViewerHasNoStatedAge(true)
                }
                NsfwViewerDropReason::LoggedOut => action::DropReason::NsfwLoggedOut(true),
            });
        }
    };
    Some(match legacy {
        FilteredReason::AuthorIsProtected => action::DropReason::ProtectedAuthor(true),
        FilteredReason::AuthorIsSuspended => action::DropReason::SuspendedAuthor(true),
        FilteredReason::AuthorBlockViewer => action::DropReason::AuthorBlocksViewer(true),
        FilteredReason::ViewerBlocksAuthor => action::DropReason::ViewerBlocksAuthor(true),
        FilteredReason::ViewerMutesAuthor => action::DropReason::ViewerMutesAuthor(true),
        FilteredReason::ExclusiveTweet => action::DropReason::ExclusiveTweet(true),
        FilteredReason::UnspecifiedReason if level.spec().unspecified_drop_is_exact() => {
            action::DropReason::Unspecified(true)
        }
        FilteredReason::UnspecifiedReason
        | FilteredReason::ContainNsfwMedia
        | FilteredReason::PossiblyUndesirable
        | FilteredReason::AuthorAccountIsInactive
        | FilteredReason::AuthorIsUnsafe
        | FilteredReason::ReportedTweet
        | FilteredReason::TweetMatchesViewerMutedKeyword(_)
        | FilteredReason::TweetIsBounced
        | FilteredReason::SafetyResult(_)
        | FilteredReason::AuthorIsDeactivated
        | FilteredReason::TweetIsNullcast => return None,
    })
}

fn tombstone_reason(reason: TombstoneReason) -> action::TombstoneReason {
    match reason {
        TombstoneReason::Bounced => action::TombstoneReason::BOUNCED,
        TombstoneReason::SensitiveViewerAgeVerification => {
            action::TombstoneReason::SENSITIVE_VIEWER_AGE_VERIFICATION
        }
        TombstoneReason::UpdateAppIos => action::TombstoneReason::UPDATE_APP_IOS,
        TombstoneReason::UpdateAppAndroid => action::TombstoneReason::UPDATE_APP_ANDROID,
        TombstoneReason::LocalRegulations => action::TombstoneReason::LOCAL_REGULATIONS,
    }
}

fn limited_engagement_reason(reason: LimitedEngagementReason) -> action::LimitedEngagementReason {
    match reason {
        LimitedEngagementReason::NonCompliant => {
            action::LimitedEngagementReason::NonCompliant(action::NonCompliant::new())
        }
        LimitedEngagementReason::ConversationControl => {
            action::LimitedEngagementReason::ConversationControl(action::ConversationControl::new())
        }
        LimitedEngagementReason::ReadonlyViewer => {
            action::LimitedEngagementReason::ReadonlyViewer(action::ReadonlyViewer::new())
        }
        LimitedEngagementReason::BlockedViewer => {
            action::LimitedEngagementReason::BlockedViewer(action::BlockedViewer::new())
        }
        LimitedEngagementReason::RootAuthorBlockedViewer => {
            action::LimitedEngagementReason::RootAuthorBlockedViewer(
                action::RootAuthorBlockedViewer::new(),
            )
        }
        LimitedEngagementReason::StaleTweet => {
            action::LimitedEngagementReason::StaleTweet(action::StaleTweet::new())
        }
        LimitedEngagementReason::CommunityTweetHidden => {
            action::LimitedEngagementReason::CommunityTweetHidden(
                action::CommunityTweetHidden::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetMemberRemoved => {
            action::LimitedEngagementReason::CommunityTweetMemberRemoved(
                action::CommunityTweetMemberRemoved::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetCommunityNotFound => {
            action::LimitedEngagementReason::CommunityTweetCommunityNotFound(
                action::CommunityTweetCommunityNotFound::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetCommunityDeleted => {
            action::LimitedEngagementReason::CommunityTweetCommunityDeleted(
                action::CommunityTweetCommunityDeleted::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetCommunitySuspended => {
            action::LimitedEngagementReason::CommunityTweetCommunitySuspended(
                action::CommunityTweetCommunitySuspended::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetViewerRemoved => {
            action::LimitedEngagementReason::CommunityTweetViewerRemoved(
                action::CommunityTweetViewerRemoved::new(),
            )
        }
        LimitedEngagementReason::DynamicProductAd => {
            action::LimitedEngagementReason::DynamicProductAd(action::DynamicProductAd::new())
        }
        LimitedEngagementReason::LocalTweet => {
            action::LimitedEngagementReason::LocalTweet(action::LocalTweet::new())
        }
    }
}

pub(crate) fn proto_action(verdict: &Verdict) -> (vf_pb::Action, Option<vf_pb::FilteredReason>) {
    let (kind, filtered_reason) = match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(reason),
            ..
        }) => (
            vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            Some(reason.legacy().clone().into()),
        ),
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(_),
            ..
        }) => (
            vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            Some(FilteredReason::UnspecifiedReason.into()),
        ),
        Verdict::Shown {
            label_interstitial: None | Some(_),
            media: Some(Decided { value, .. }),
            engagement: None | Some(_),
        } => (
            vf_pb::action::Kind::Interstitial(true),
            Some(value.legacy().clone().into()),
        ),
        Verdict::Fosnr(_)
        | Verdict::Shown {
            label_interstitial: None | Some(_),
            media: None,
            engagement: None | Some(_),
        } => (vf_pb::action::Kind::Allow(true), None),
    };
    (vf_pb::Action { kind: Some(kind) }, filtered_reason)
}

#[cfg(test)]
const BY_RULE_ACTIONS: [VerdictAction; 6] = [
    VerdictAction::Drop,
    VerdictAction::Tombstone,
    VerdictAction::Interstitial,
    VerdictAction::LimitedEngagement,
    VerdictAction::SoftIntervention,
    VerdictAction::Appealable,
];

pub(crate) fn verdict_action(verdict: &Verdict) -> VerdictAction {
    match verdict {
        Verdict::Withheld(decided) => decided.value.kind().verdict_action(),
        Verdict::Fosnr(decided) => decided.value.kind().verdict_action(),
        Verdict::Shown {
            label_interstitial: Some(_),
            ..
        } => VerdictAction::TweetInterstitial,
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: None,
        } => VerdictAction::Allow,
        Verdict::Shown {
            label_interstitial: None,
            media: Some(_),
            engagement: None,
        } => VerdictAction::Interstitial,
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: Some(_),
        } => VerdictAction::LimitedEngagement,
        Verdict::Shown {
            label_interstitial: None,
            media: Some(_),
            engagement: Some(_),
        } => VerdictAction::TweetInterstitial,
    }
}

pub(crate) fn decided_rows(
    verdict: &Verdict,
) -> impl Iterator<Item = (&'static str, VerdictAction)> {
    verdict
        .decided()
        .map(|(kind, by)| (by, kind.verdict_action()))
}

pub(crate) fn decided_by(verdict: &Verdict) -> Option<&'static str> {
    decided_rows(verdict).next().map(|(rule, _)| rule)
}

pub(crate) fn filter_tweets_result(outcome: PostEvaluation) -> vf_pb::TweetVisibilityResult {
    let (action, filtered_reason) = proto_action(outcome.evaluation.verdict());
    vf_pb::TweetVisibilityResult {
        tweet_id: outcome.tweet_id.0,
        action: Some(action),
        filtered_reason,
        safety_labels: outcome.safety_labels.map(Arc::unwrap_or_clone),
    }
}

pub(crate) fn evaluate_tweets_outcome(
    evaluation: Option<&Evaluation>,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> TweetEvaluationOutcome {
    match evaluation
        .and_then(|evaluation| evaluate_tweets_state(evaluation, level, policies))
        .map(|state| xai_x_thrift::serialize_compact(&state))
    {
        Some(Ok(bytes)) => TweetEvaluationOutcome::ResultStateThriftCompact(bytes.into()),
        Some(Err(_)) | None => TweetEvaluationOutcome::Failed(vf_pb::Failed {}),
    }
}

pub(crate) fn evaluate_tweets_state(
    evaluation: &Evaluation,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> Option<TweetFieldsResultState> {
    match evaluation {
        Evaluation::Complete { verdict, .. } => Some(thrift_result_state(verdict, level, policies)),
        Evaluation::NotFound(_) => Some(TweetFieldsResultState::NotFound(
            TweetFieldsResultNotFound::new(false, false, None),
        )),
        Evaluation::Deleted(deletion) => Some(TweetFieldsResultState::NotFound(
            TweetFieldsResultNotFound::new(true, *deletion == Deletion::BounceDeleted, None),
        )),
        Evaluation::Partial { .. } | Evaluation::Failed(_) => None,
    }
}

pub(crate) fn evaluate_tweets_label(
    evaluation: &Evaluation,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> Label {
    evaluate_tweets_state(evaluation, level, policies)
        .map_or_else(|| label(Class::Failed, NONE), |state| state_label(&state))
}

pub(crate) fn notification_result(
    evaluation: Option<&Evaluation>,
) -> (
    vf_pb::NotificationVisibilityResult,
    (&'static str, &'static str),
) {
    use vf_pb::notification_visibility_result::Outcome;
    let dropped = |reason: NotificationDropReason| {
        (
            Outcome::Dropped(vf_pb::Dropped {
                reason: Some(reason.into()),
            }),
            ("dropped", notification_reason_label(reason)),
        )
    };
    let (outcome, labels) = match evaluation {
        Some(
            Evaluation::Complete {
                verdict,
                fail_open_defaults,
            }
            | Evaluation::Partial {
                verdict,
                fail_open_defaults,
            },
        ) if fail_open_defaults.is_empty() => match verdict {
            Verdict::Withheld(Decided { id, .. }) => dropped(notification_drop_reason(*id)),
            Verdict::Fosnr(_) | Verdict::Shown { .. } => {
                (Outcome::Allowed(vf_pb::Allowed {}), ("allowed", "none"))
            }
        },
        Some(Evaluation::NotFound(Lookup::Author | Lookup::SharedAuthor)) => {
            dropped(NotificationDropReason::ActorNotFound)
        }
        Some(
            Evaluation::NotFound(Lookup::Tweet | Lookup::SharedTweet) | Evaluation::Deleted(_),
        ) => dropped(NotificationDropReason::PostNotFound),
        Some(Evaluation::Complete { .. } | Evaluation::Partial { .. } | Evaluation::Failed(_))
        | None => (Outcome::Failed(vf_pb::Failed {}), ("failed", "none")),
    };
    (
        vf_pb::NotificationVisibilityResult {
            outcome: Some(outcome),
        },
        labels,
    )
}

const NOTIFICATION_DROP_REASONS: [(RuleId, NotificationDropReason); 4] = [
    (
        RuleId::SuspendedAuthor,
        NotificationDropReason::ActorSuspended,
    ),
    (
        RuleId::ProtectedAuthor,
        NotificationDropReason::ActorProtected,
    ),
    (
        RuleId::ViewerBlocksAuthor,
        NotificationDropReason::RecipientBlocksActor,
    ),
    (
        RuleId::ViewerMutesAndDoesNotFollowAuthor,
        NotificationDropReason::RecipientMutesActor,
    ),
];

fn notification_drop_reason(rule: RuleId) -> NotificationDropReason {
    NOTIFICATION_DROP_REASONS
        .iter()
        .find(|&&(id, _)| id == rule)
        .map_or(NotificationDropReason::Unspecified, |&(_, reason)| reason)
}

fn notification_reason_label(reason: NotificationDropReason) -> &'static str {
    match reason {
        NotificationDropReason::Unspecified => "unspecified",
        NotificationDropReason::ActorNotFound => "actor_not_found",
        NotificationDropReason::ActorSuspended => "actor_suspended",
        NotificationDropReason::ActorProtected => "actor_protected",
        NotificationDropReason::RecipientBlocksActor => "recipient_blocks_actor",
        NotificationDropReason::RecipientMutesActor => "recipient_mutes_actor",
        NotificationDropReason::PostNotFound => "post_not_found",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::fixtures::{
        allow, appealed, blurred, blurred_and_limited, dropped, fosnr, label_interstitial,
        legacy_interstitial, limited, limited_for, tombstoned, verify_blurred,
        with_label_interstitial,
    };
    use vf_pb::action::Kind;
    use xai_visibility_filtering::graphql_results::{
        is_dropped_media_blur, resolve_blurred_image_interstitial,
    };
    use xai_visibility_filtering::models::{KeywordMatch, SafetyResult};
    use xai_x_thrift::safety_result::SafetyResult as ThriftSafetyResult;
    use SafetyLevel::{FilterAll, TimelineHome};

    fn policies() -> LimitedActionsPolicies {
        LimitedActionsPolicies::for_tests(vec![(
            LimitedEngagementReason::ConversationControl,
            vec![LimitedActionType::Reply],
        )])
    }

    #[test]
    fn decided_by_names_the_deciding_rule_or_why_the_rules_never_ran() {
        use crate::hydration::{Hydrator, Hydrators};
        let suspended = dropped(FilteredReason::AuthorIsSuspended, "rule");
        for (evaluation, rule) in [
            (Evaluation::complete(allow()), None),
            (Evaluation::complete(suspended.clone()), Some("rule")),
            (
                Evaluation::Partial {
                    verdict: suspended,
                    fail_open_defaults: Hydrators::of(Hydrator::Follows),
                },
                Some("rule"),
            ),
            (Evaluation::NotFound(Lookup::Author), Some("not_found")),
            (Evaluation::Failed(Lookup::Tweet), Some("lookup_failed")),
        ] {
            assert_eq!(decided_by(evaluation.verdict()), rule, "{evaluation:?}");
        }
    }

    fn thrift_media(reason: InterstitialReason) -> ComposedMediaVisibilityActions {
        ComposedMediaVisibilityActions {
            media_interstitial: Some(Box::new(ThriftMediaInterstitial::BlurredImageInterstitial(
                BlurredImageInterstitial {
                    reason: Some(reason),
                    opacity: Some(0.8.into()),
                    interstitial_action: None,
                    available_verification_options: None,
                },
            ))),
        }
    }

    fn thrift_blur(reason: InterstitialReason) -> Action {
        Action::ComposedMediaVisibilityResults(thrift_media(reason))
    }

    fn thrift_limit_for(
        reason: action::LimitedEngagementReason,
        policy: Option<LimitedActionsPolicy>,
        limited_actions: &str,
    ) -> Action {
        Action::LimitedEngagements(LimitedEngagements::new(
            reason,
            policy,
            limited_actions.to_string(),
        ))
    }

    fn thrift_limit() -> LimitedEngagements {
        LimitedEngagements::new(
            Some(action::LimitedEngagementReason::ConversationControl(
                action::ConversationControl::new(),
            )),
            LimitedActionsPolicy::new(vec![LimitedAction::new(
                action::LimitedActionType::REPLY,
                None,
            )]),
            "limited_replies".to_string(),
        )
    }

    fn limit() -> Verdict {
        limited(LimitedEngagementReason::ConversationControl, "limit_rule")
    }

    fn blur(reason: InterstitialReason) -> Verdict {
        blurred(reason, "blur_rule")
    }

    fn thrift_drop(reason: Option<action::DropReason>) -> Action {
        Action::Drop(action::Drop::new(reason, None))
    }

    struct Projected {
        thrift: Action,
        proto: Kind,
        reason: Option<FilteredReason>,
        label: &'static str,
        rows: &'static [(&'static str, &'static str)],
    }

    fn verdict_cases() -> [(Verdict, Projected); 19] {
        use action::LimitedActionType as T;
        let proto_drop = Kind::Drop(vf_pb::DropReason {});
        let tombstone = |reason, code| {
            (
                tombstoned(reason, "rule"),
                Projected {
                    thrift: Action::Tombstone(Tombstone::new(
                        Some(action::TombstoneReason(code)),
                        None,
                    )),
                    proto: proto_drop,
                    reason: Some(FilteredReason::UnspecifiedReason),
                    label: "tombstone",
                    rows: &[("rule", "tombstone")],
                },
            )
        };
        let nsfw_viewer_drop = |reason, thrift| {
            (
                Verdict::Withheld(Decided {
                    value: Withholding::Drop(DropReason::NsfwViewer(reason)),
                    id: RuleId::FilterAll,
                    by: "rule",
                }),
                Projected {
                    thrift: thrift_drop(Some(thrift)),
                    proto: proto_drop,
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "drop",
                    rows: &[("rule", "drop")],
                },
            )
        };
        [
            (
                allow(),
                Projected {
                    thrift: Action::Allow(action::Allow::new()),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "allow",
                    rows: &[],
                },
            ),
            (
                dropped(FilteredReason::AuthorIsSuspended, "rule"),
                Projected {
                    thrift: thrift_drop(Some(action::DropReason::SuspendedAuthor(true))),
                    proto: proto_drop,
                    reason: Some(FilteredReason::AuthorIsSuspended),
                    label: "drop",
                    rows: &[("rule", "drop")],
                },
            ),
            nsfw_viewer_drop(
                NsfwViewerDropReason::IsUnderage,
                action::DropReason::NsfwViewerIsUnderage(true),
            ),
            nsfw_viewer_drop(
                NsfwViewerDropReason::HasNoStatedAge,
                action::DropReason::NsfwViewerHasNoStatedAge(true),
            ),
            nsfw_viewer_drop(
                NsfwViewerDropReason::LoggedOut,
                action::DropReason::NsfwLoggedOut(true),
            ),
            tombstone(TombstoneReason::Bounced, 2),
            tombstone(TombstoneReason::SensitiveViewerAgeVerification, 30),
            tombstone(TombstoneReason::UpdateAppIos, 31),
            tombstone(TombstoneReason::UpdateAppAndroid, 32),
            tombstone(TombstoneReason::LocalRegulations, 33),
            (
                blur(InterstitialReason::Sensitive(true)),
                Projected {
                    thrift: thrift_blur(InterstitialReason::Sensitive(true)),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "interstitial",
                    rows: &[("blur_rule", "interstitial")],
                },
            ),
            (
                limit(),
                Projected {
                    thrift: Action::LimitedEngagements(thrift_limit()),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "limited_engagement",
                    rows: &[("limit_rule", "limited_engagement")],
                },
            ),
            (
                limited_for(&[LimitedEngagementReason::StaleTweet], "limit_rule"),
                Projected {
                    thrift: thrift_limit_for(
                        action::LimitedEngagementReason::StaleTweet(action::StaleTweet::new()),
                        None,
                        "stale_tweet",
                    ),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "limited_engagement",
                    rows: &[("limit_rule", "limited_engagement")],
                },
            ),
            (
                blurred_and_limited(blur(InterstitialReason::Sensitive(true)), limit()),
                Projected {
                    thrift: Action::TweetInterstitial(TweetInterstitial {
                        limited_engagements: Some(thrift_limit()),
                        all_media_visibility_results: Some(thrift_media(
                            InterstitialReason::Sensitive(true),
                        )),
                        ..TweetInterstitial::default()
                    }),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "tweet_interstitial",
                    rows: &[
                        ("blur_rule", "interstitial"),
                        ("limit_rule", "limited_engagement"),
                    ],
                },
            ),
            (
                legacy_interstitial("fallback_rule"),
                Projected {
                    thrift: Action::Interstitial(Interstitial::new(
                        InterstitialReason::ContainsNsfwMedia(true),
                        None,
                    )),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "interstitial",
                    rows: &[("fallback_rule", "interstitial")],
                },
            ),
            (
                blurred_and_limited(legacy_interstitial("fallback_rule"), limit()),
                Projected {
                    thrift: Action::TweetInterstitial(TweetInterstitial {
                        interstitial: Some(AnyInterstitial::Interstitial(Interstitial::new(
                            InterstitialReason::ContainsNsfwMedia(true),
                            None,
                        ))),
                        limited_engagements: Some(thrift_limit()),
                        ..TweetInterstitial::default()
                    }),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "tweet_interstitial",
                    rows: &[
                        ("fallback_rule", "interstitial"),
                        ("limit_rule", "limited_engagement"),
                    ],
                },
            ),
            (
                fosnr(true, false, "fosnr_rule"),
                Projected {
                    thrift: Action::TweetInterstitial(TweetInterstitial {
                        soft_intervention: Some(ThriftSoftIntervention {
                            soft_intervention_reason: Some(SoftInterventionReason::Fosnr(
                                Box::new(AppealableReason::new(
                                    1,
                                    Box::new(action::AppealablePolicy::ABUSE),
                                    true,
                                    false,
                                )),
                            )),
                            engagement_nudge: Some(false),
                            suppress_autoplay: Some(true),
                            warning: None,
                            details_url: None,
                            display_type: Some(SoftInterventionDisplayType::FOSNR),
                        }),
                        avoid: Some(Avoid::new(None, Some(BrandSafetyRiskLevel::NORMAL), None)),
                        ..TweetInterstitial::default()
                    }),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "soft_intervention",
                    rows: &[("fosnr_rule", "soft_intervention")],
                },
            ),
            appeal(
                AppealablePolicy::VIOLENT_SPEECH,
                3,
                &[
                    T::LIKE,
                    T::REPLY,
                    T::RETWEET,
                    T::QUOTE_TWEET,
                    T::SHARE_TWEET_VIA,
                    T::ADD_TO_BOOKMARKS,
                    T::PIN_TO_PROFILE,
                    T::COPY_LINK,
                    T::SEND_VIA_DM,
                    T::EDIT_TWEET,
                    T::HIGHLIGHT,
                    T::EMBED,
                    T::LISTS_ADD_REMOVE,
                ],
                Some("freedom_of_speech_not_reach"),
            ),
            appeal(AppealablePolicy::ABUSE, 1, &[T::EDIT_TWEET], None),
        ]
    }

    fn appeal(
        policy: AppealablePolicy,
        level: i8,
        limited: &[action::LimitedActionType],
        legacy_limited_actions: Option<&str>,
    ) -> (Verdict, Projected) {
        let reason = AppealableReason::new(level, Box::new(policy), false, true);
        let limited = limited
            .iter()
            .map(|&action_type| LimitedAction::new(action_type, None))
            .collect();
        (
            appealed(policy, level, false, true, "fosnr_author"),
            Projected {
                thrift: Action::TweetInterstitial(TweetInterstitial {
                    limited_engagements: Some(LimitedEngagements::new(
                        action::LimitedEngagementReason::FosnrReason(reason.clone()),
                        LimitedActionsPolicy::new(limited),
                        legacy_limited_actions.map(str::to_string),
                    )),
                    avoid: Some(Avoid::new(None, Some(BrandSafetyRiskLevel::NORMAL), None)),
                    appealable: Some(Appealable::new(reason, None)),
                    ..TweetInterstitial::default()
                }),
                proto: Kind::Allow(true),
                reason: None,
                label: "appealable",
                rows: &[("fosnr_author", "appealable")],
            },
        )
    }

    #[test]
    fn every_verdict_case_projects_per_the_table() {
        for (verdict, expected) in verdict_cases() {
            let name = format!("{verdict:?}");
            assert_eq!(
                thrift_action(&verdict, TimelineHome, &policies()),
                expected.thrift,
                "{name}"
            );
            assert_eq!(
                <&str>::from(verdict_action(&verdict)),
                expected.label,
                "{name}"
            );
            assert_eq!(
                decided_rows(&verdict)
                    .map(|(rule, action)| (rule, <&str>::from(action)))
                    .collect::<Vec<_>>(),
                expected.rows,
                "{name}"
            );
            for (rule, action) in decided_rows(&verdict) {
                assert!(BY_RULE_ACTIONS.contains(&action), "{name}: {rule}");
            }
            let (proto, reason) = proto_action(&verdict);
            assert_eq!(proto.kind, Some(expected.proto), "{name}");
            assert_eq!(reason, expected.reason.map(Into::into), "{name}");
        }
    }

    #[test]
    fn a_limit_sends_its_first_reason_and_the_union_of_its_reasons_policies() {
        use LimitedActionType::{Like, Reply, Retweet};
        use LimitedEngagementReason::{BlockedViewer, ConversationControl, StaleTweet};
        let policies = LimitedActionsPolicies::for_tests(vec![
            (BlockedViewer, vec![Retweet, Reply, Like]),
            (ConversationControl, vec![Reply]),
        ]);
        let verdict = limited_for(
            &[StaleTweet, ConversationControl, BlockedViewer],
            "limit_rule",
        );
        let action = |action_type| LimitedAction::new(action_type, None);
        assert_eq!(
            thrift_action(&verdict, TimelineHome, &policies),
            thrift_limit_for(
                action::LimitedEngagementReason::StaleTweet(action::StaleTweet::new()),
                Some(LimitedActionsPolicy::new(vec![
                    action(action::LimitedActionType::REPLY),
                    action(action::LimitedActionType::RETWEET),
                    action(action::LimitedActionType::LIKE),
                ])),
                "stale_tweet",
            )
        );
        assert_eq!(<&str>::from(verdict_action(&verdict)), "limited_engagement");
        assert_eq!(
            decided_rows(&verdict)
                .map(|(rule, action)| (rule, <&str>::from(action)))
                .collect::<Vec<_>>(),
            [("limit_rule", "limited_engagement")]
        );
    }

    #[test]
    fn each_limited_engagement_reason_sends_its_thrift_arm_and_legacy_string() {
        use action::LimitedEngagementReason as T;
        use LimitedEngagementReason as R;
        for (reason, arm, legacy) in [
            (
                R::ConversationControl,
                T::ConversationControl(action::ConversationControl::new()),
                "limited_replies",
            ),
            (
                R::ReadonlyViewer,
                T::ReadonlyViewer(action::ReadonlyViewer::new()),
                "readonly_viewer",
            ),
            (
                R::BlockedViewer,
                T::BlockedViewer(action::BlockedViewer::new()),
                "blocked_viewer",
            ),
            (
                R::RootAuthorBlockedViewer,
                T::RootAuthorBlockedViewer(action::RootAuthorBlockedViewer::new()),
                "root_author_blocked_viewer",
            ),
            (
                R::StaleTweet,
                T::StaleTweet(action::StaleTweet::new()),
                "stale_tweet",
            ),
            (
                R::CommunityTweetHidden,
                T::CommunityTweetHidden(action::CommunityTweetHidden::new()),
                "community_tweet_hidden",
            ),
            (
                R::CommunityTweetMemberRemoved,
                T::CommunityTweetMemberRemoved(action::CommunityTweetMemberRemoved::new()),
                "community_tweet_member_removed",
            ),
            (
                R::CommunityTweetCommunityNotFound,
                T::CommunityTweetCommunityNotFound(action::CommunityTweetCommunityNotFound::new()),
                "community_tweet_community_not_found",
            ),
            (
                R::CommunityTweetCommunityDeleted,
                T::CommunityTweetCommunityDeleted(action::CommunityTweetCommunityDeleted::new()),
                "community_tweet_community_deleted",
            ),
            (
                R::CommunityTweetCommunitySuspended,
                T::CommunityTweetCommunitySuspended(action::CommunityTweetCommunitySuspended::new()),
                "community_tweet_community_suspended",
            ),
            (
                R::CommunityTweetViewerRemoved,
                T::CommunityTweetViewerRemoved(action::CommunityTweetViewerRemoved::new()),
                "community_tweet_viewer_removed",
            ),
            (
                R::DynamicProductAd,
                T::DynamicProductAd(action::DynamicProductAd::new()),
                "dynamic_product_ad",
            ),
            (
                R::LocalTweet,
                T::LocalTweet(action::LocalTweet::new()),
                "local_tweet",
            ),
        ] {
            assert_eq!(
                thrift_action(
                    &limited_for(&[reason], "limit_rule"),
                    TimelineHome,
                    &LimitedActionsPolicies::default(),
                ),
                thrift_limit_for(arm, None, legacy),
                "{reason:?}"
            );
        }
    }

    #[test]
    fn entity_mixer_resolves_the_blur_and_its_prompt_from_both_media_arms() {
        let reason = InterstitialReason::Nudity(true);
        let verify = verify_blurred(reason.clone(), "blur_rule");
        for verdict in [verify.clone(), blurred_and_limited(verify, limit())] {
            let rendered = resolve_blurred_image_interstitial(&ThriftSafetyResult::new(
                None,
                thrift_action(&verdict, TimelineHome, &policies()),
            ));
            assert_eq!(
                rendered,
                Some(BlurredImageInterstitial {
                    reason: Some(reason.clone()),
                    opacity: Some(0.8.into()),
                    interstitial_action: Some(InterstitialAction::AGE_VERIFICATION_PROMPT),
                    available_verification_options: Some(vec![
                        AgeVerificationOption::SELFIE,
                        AgeVerificationOption::PERSONA,
                    ]),
                }),
                "{verdict:?}"
            );
        }
    }

    #[test]
    fn drop_reasons_without_a_canonical_form_drop_without_a_reason() {
        assert_eq!(
            thrift_action(
                &dropped(FilteredReason::AuthorIsProtected, "rule"),
                TimelineHome,
                &policies()
            ),
            thrift_drop(Some(action::DropReason::ProtectedAuthor(true)))
        );
        assert_eq!(
            thrift_action(
                &dropped(FilteredReason::UnspecifiedReason, "rule"),
                FilterAll,
                &policies()
            ),
            thrift_drop(Some(action::DropReason::Unspecified(true)))
        );
        let lossy = [
            FilteredReason::UnspecifiedReason,
            FilteredReason::ContainNsfwMedia,
            FilteredReason::PossiblyUndesirable,
            FilteredReason::AuthorAccountIsInactive,
            FilteredReason::AuthorIsUnsafe,
            FilteredReason::ReportedTweet,
            FilteredReason::TweetMatchesViewerMutedKeyword(KeywordMatch {
                keyword: "kw".into(),
            }),
            FilteredReason::TweetIsBounced,
            FilteredReason::SafetyResult(SafetyResult::default()),
            FilteredReason::AuthorIsDeactivated,
            FilteredReason::TweetIsNullcast,
        ];
        for reason in lossy {
            let verdict = dropped(reason, "rule");
            assert_eq!(
                thrift_action(&verdict, TimelineHome, &policies()),
                thrift_drop(None),
                "{verdict:?}"
            );
        }
    }

    #[test]
    fn thrift_result_state_uses_a_legacy_arm_only_where_it_renders_differently() {
        use ThriftFilteredReason as T;
        let found = |reason| TweetFieldsResultState::Found(TweetFieldsResultFound::new(reason));
        let filtered =
            |reason| TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason));
        let safety_result_with = |reason, verdict: &Verdict| {
            T::SafetyResult(ThriftSafetyResult::new(
                reason,
                thrift_action(verdict, TimelineHome, &policies()),
            ))
        };
        let safety_result = |verdict: &Verdict| safety_result_with(None, verdict);
        let nsfw = legacy_interstitial("nsfw_rule");
        let mut cases = vec![
            (
                dropped(FilteredReason::TweetIsBounced, "rule"),
                filtered(T::TweetIsBounced(true)),
            ),
            (
                dropped(FilteredReason::AuthorBlockViewer, "rule"),
                filtered(T::AuthorBlockViewer(true)),
            ),
            (
                dropped(FilteredReason::AuthorIsProtected, "rule"),
                filtered(T::AuthorIsProtected(true)),
            ),
            (
                dropped(FilteredReason::AuthorIsSuspended, "rule"),
                filtered(T::AuthorIsSuspended(true)),
            ),
            (allow(), found(None)),
            (nsfw.clone(), found(Some(T::ContainNsfwMedia(true)))),
        ];
        for verdict in [
            dropped(FilteredReason::AuthorIsDeactivated, "rule"),
            dropped(FilteredReason::ExclusiveTweet, "rule"),
            Verdict::Withheld(Decided {
                value: Withholding::Drop(DropReason::NsfwViewer(NsfwViewerDropReason::IsUnderage)),
                id: RuleId::FilterAll,
                by: "rule",
            }),
            tombstoned(TombstoneReason::LocalRegulations, "rule"),
        ] {
            cases.push((verdict.clone(), filtered(safety_result(&verdict))));
        }
        for verdict in [
            limit(),
            blurred_and_limited(nsfw, limit()),
            fosnr(true, false, "fosnr_rule"),
        ] {
            cases.push((verdict.clone(), found(Some(safety_result(&verdict)))));
        }
        for (policy, reason) in [
            (
                AppealablePolicy::HATEFUL_CONDUCT,
                SafetyResultReason::FOSNR_HATEFUL_CONDUCT,
            ),
            (AppealablePolicy::ABUSE, SafetyResultReason::FOSNR_ABUSE),
            (
                AppealablePolicy::VIOLENT_SPEECH,
                SafetyResultReason::FOSNR_VIOLENT_SPEECH,
            ),
            (
                AppealablePolicy::CIVIC_INTEGRITY,
                SafetyResultReason::FOSNR_CIVIC_INTEGRITY,
            ),
        ] {
            let verdict = appealed(policy, 3, true, false, "fosnr_author");
            let expected = found(Some(safety_result_with(Some(reason), &verdict)));
            cases.push((verdict, expected));
        }
        for (verdict, expected) in cases {
            assert_eq!(
                thrift_result_state(&verdict, TimelineHome, &policies()),
                expected,
                "{verdict:?}"
            );
        }
    }

    #[test]
    fn dashboard_generator_pins_the_verdict_mix_labels() {
        use strum::VariantArray;
        let emitted: std::collections::BTreeSet<String> = VerdictAction::VARIANTS
            .iter()
            .map(|&action| <&str>::from(action).to_owned())
            .collect();
        let dashboard = crate::pinned::dashboard_tuple("FT_VERDICT_ACTIONS");
        assert_eq!(dashboard, emitted);
    }

    #[test]
    fn a_public_interest_post_is_sent_as_tweetypie_sends_it() {
        let verdict = label_interstitial(
            LabelInterstitial::PublicInterest(PublicInterest {
                reason: SafetyResultReason::ONE_OFF,
                limit: Some(LimitedEngagementReason::NonCompliant),
            }),
            "public_interest/label_interstitial",
            None,
        );
        let policies = LimitedActionsPolicies::for_tests(vec![(
            LimitedEngagementReason::NonCompliant,
            vec![LimitedActionType::Reply, LimitedActionType::Like],
        )]);
        let non_compliant =
            || action::LimitedEngagementReason::NonCompliant(action::NonCompliant::new());
        let sent = Action::TweetInterstitial(TweetInterstitial {
            interstitial: Some(AnyInterstitial::InterstitialLimitedEngagements(
                InterstitialLimitedEngagements::new(
                    non_compliant(),
                    LocalizedMessage::new(
                        "This Post violated the X Rules. However, X has determined that it may \
                         be in the public\u{2019}s interest for the Post to remain accessible. \
                         {learnmore}"
                            .to_string(),
                        "en".to_string(),
                        vec![MessageLink::new(
                            "learnmore".to_string(),
                            "Learn more".to_string(),
                            "https://help.x.com/rules-and-policies/public-interest".to_string(),
                        )],
                    ),
                    None,
                ),
            )),
            limited_engagements: Some(LimitedEngagements::new(
                non_compliant(),
                LimitedActionsPolicy::new(vec![
                    LimitedAction::new(action::LimitedActionType::REPLY, None),
                    LimitedAction::new(action::LimitedActionType::LIKE, None),
                ]),
                "non_compliant".to_string(),
            )),
            ..TweetInterstitial::default()
        });
        assert_eq!(
            thrift_result_state(&verdict, SafetyLevel::TimelineHomeHydration, &policies),
            TweetFieldsResultState::Found(TweetFieldsResultFound::new(
                ThriftFilteredReason::SafetyResult(ThriftSafetyResult::new(
                    SafetyResultReason::ONE_OFF,
                    sent,
                ))
            ))
        );
    }

    #[test]
    fn a_label_interstitial_over_the_legacy_nsfw_interstitial_drops_as_tweetypie_does() {
        let verdict = with_label_interstitial(
            crate::rules::fixtures::legacy_interstitial("nsfw_high_precision/legacy_interstitial"),
            LabelInterstitial::PublicInterest(PublicInterest {
                reason: SafetyResultReason::ONE_OFF,
                limit: Some(LimitedEngagementReason::NonCompliant),
            }),
            "public_interest/label_interstitial",
        );
        let TweetFieldsResultState::Found(found) = thrift_result_state(
            &verdict,
            SafetyLevel::TimelineHomeHydration,
            &LimitedActionsPolicies::default(),
        ) else {
            panic!("a label interstitial is found");
        };
        let reason = found
            .suppress_reason
            .expect("a label interstitial suppresses");
        assert!(is_dropped_media_blur(&reason), "{reason:?}");
    }

    #[test]
    fn an_emergency_post_carries_its_copy_and_a_later_limit() {
        let verdict = label_interstitial(
            LabelInterstitial::EmergencyDynamic(Box::new(EmergencyCopy {
                copy: "Stay safe.".to_owned(),
                link: Some("https://help.x.com/emergency".to_owned()),
            })),
            "for_emergency_use_only/label_interstitial",
            Some((
                LimitedEngagementReason::ReadonlyViewer,
                "read_only_viewer/limited_engagement",
            )),
        );
        let Action::TweetInterstitial(sent) = thrift_action(
            &verdict,
            SafetyLevel::TimelineHomeHydration,
            &LimitedActionsPolicies::default(),
        ) else {
            panic!("an emergency notice is sent as a TweetInterstitial");
        };
        assert_eq!(
            sent.interstitial,
            Some(AnyInterstitial::EmergencyDynamicInterstitial(
                EmergencyDynamicInterstitial::new(
                    "Stay safe.".to_string(),
                    "https://help.x.com/emergency".to_string(),
                    Box::new(LocalizedMessage::new(
                        "Stay safe. {learnmore}".to_string(),
                        "en".to_string(),
                        vec![MessageLink::new(
                            "learnmore".to_string(),
                            "Learn more".to_string(),
                            "https://help.x.com/emergency".to_string(),
                        )],
                    )),
                )
            ))
        );
        assert_eq!(
            sent.limited_engagements.map(|limit| limit.limited_actions),
            Some(Some("non_compliant".to_string()))
        );
        assert_eq!(<&str>::from(verdict_action(&verdict)), "tweet_interstitial");
        assert_eq!(
            decided_rows(&verdict)
                .map(|(rule, action)| (rule, <&str>::from(action)))
                .collect::<Vec<_>>(),
            [
                ("for_emergency_use_only/label_interstitial", "interstitial"),
                ("read_only_viewer/limited_engagement", "limited_engagement"),
            ]
        );
    }

    #[test]
    fn each_public_interest_reason_has_english_copy() {
        for &reason in SafetyResultReason::ENUM_VALUES {
            let (resname, _) = public_interest_copy(reason);
            assert!(translate(resname, "en").is_some(), "{reason:?} {resname}");
        }
    }

    #[test]
    fn each_evaluation_sends_and_labels_as_vf_counts_it() {
        use crate::hydration::{Hydrator, Hydrators};
        let shown = TweetFieldsResultState::Found(TweetFieldsResultFound::new(None));
        let suspended = TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(
            ThriftFilteredReason::AuthorIsSuspended(true),
        ));
        let not_found =
            TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(false, false, None));
        let cases = [
            (
                Evaluation::complete(allow()),
                Some(&shown),
                ("allow", "none"),
            ),
            (
                Evaluation::complete(dropped(FilteredReason::AuthorIsSuspended, "rule")),
                Some(&suspended),
                ("drop", "author_is_suspended"),
            ),
            (
                Evaluation::Partial {
                    verdict: dropped(FilteredReason::AuthorIsSuspended, "rule"),
                    fail_open_defaults: Hydrators::of(Hydrator::Follows),
                },
                None,
                ("failed", "none"),
            ),
            (
                Evaluation::NotFound(Lookup::Author),
                Some(&not_found),
                ("not_found", "none"),
            ),
            (Evaluation::Failed(Lookup::Tweet), None, ("failed", "none")),
        ];
        let policies = LimitedActionsPolicies::default();
        let level = SafetyLevel::TimelineHomeHydration;
        for (evaluation, sent, (class, reason)) in &cases {
            assert_eq!(
                evaluate_tweets_state(evaluation, level, &policies).as_ref(),
                *sent,
                "{evaluation:?}"
            );
            assert_eq!(
                evaluate_tweets_label(evaluation, level, &policies),
                ((*class).to_owned(), (*reason).to_owned()),
                "{evaluation:?}"
            );
        }
    }

    #[test]
    fn each_quote_rerender_sends_tweetypies_filtered_reason() {
        use crate::rules::QuoteEntries;
        use strum::VariantArray;
        use ThriftFilteredReason as T;
        let sent: Vec<(SafetyLevel, RuleId, TweetFieldsResultState)> = SafetyLevel::VARIANTS
            .iter()
            .flat_map(|&level| {
                let rerenders = match level.spec().quote_entries() {
                    QuoteEntries::JudgedWithQuotePass { rerenders } => rerenders,
                    QuoteEntries::NotJudged | QuoteEntries::Judged => &[],
                };
                rerenders.iter().map(move |&(by, reason)| {
                    (level, by, rerendered_state(&Rerendered { by, reason }))
                })
            })
            .collect();
        let expected: Vec<_> = [
            (RuleId::DeactivatedAuthor, T::AuthorIsDeactivated(true)),
            (RuleId::ErasedAuthor, T::AuthorIsDeactivated(true)),
            (RuleId::OffboardedAuthor, T::AuthorAccountIsInactive(true)),
            (RuleId::SuspendedAuthor, T::AuthorIsSuspended(true)),
            (
                RuleId::QuotedAuthorBlocksQuotingAuthor,
                T::AuthorBlockViewer(true),
            ),
            (RuleId::ViewerBlocksAuthor, T::UnspecifiedReason(true)),
            (
                RuleId::QuotedViewerMutesAuthorNotFollowing,
                T::UnspecifiedReason(true),
            ),
            (RuleId::ProtectedQuotedAuthor, T::AuthorIsProtected(true)),
        ]
        .into_iter()
        .map(|(rule, reason)| {
            (
                rule,
                TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason)),
            )
        })
        .collect();
        let expected: Vec<_> = SafetyLevel::VARIANTS
            .iter()
            .filter(|level| {
                matches!(
                    level.spec().quote_entries(),
                    QuoteEntries::JudgedWithQuotePass { .. }
                )
            })
            .flat_map(|&level| {
                expected
                    .iter()
                    .map(move |(rule, state)| (level, *rule, state.clone()))
            })
            .collect();
        assert_eq!(sent, expected);
    }
}
