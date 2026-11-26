use crate::models::{
    Decided, DropReason, EmergencyCopy, FosnrReason, LabelInterstitial, LimitedActionType,
    LimitedEngagement, LimitedEngagementReason, MediaInterstitial, MediaRestriction, Notice,
    NsfwViewerDropReason, PublicInterest, TombstoneReason, Verdict, Withholding,
};
use crate::params::limited_actions_copy::{Prompt, PromptKind, LEARN_MORE_PLACEHOLDER};
use crate::params::LimitedActionsPolicies;
use crate::rules::{RerenderReason, SafetyLevel};
use xai_visibility_filtering::graphql_results::translate;
use xai_visibility_filtering::links;
use xai_visibility_filtering::models::FilteredReason;
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
    TweetFieldsResultFiltered, TweetFieldsResultFound, TweetFieldsResultState,
};

pub(crate) fn thrift_result_state(
    verdict: &Verdict,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> TweetFieldsResultState {
    let safety_result = |reason: Option<SafetyResultReason>, action: Action| {
        ThriftFilteredReason::SafetyResult(ThriftSafetyResult::new(reason, action))
    };
    let found = |reason| TweetFieldsResultState::Found(TweetFieldsResultFound::new(reason));
    let shown = |reason, action| found(Some(safety_result(reason, action)));
    let filtered =
        |reason| TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason));
    match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(reason),
            ..
        }) => filtered(legacy_drop_reason(reason.legacy()).unwrap_or_else(|| {
            safety_result(
                None,
                Action::Drop(action::Drop::new(drop_reason(reason, level), None)),
            )
        })),
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(reason),
            ..
        }) => filtered(safety_result(
            None,
            Action::Tombstone(Tombstone::new(Some(tombstone_reason(*reason)), None)),
        )),
        Verdict::Fosnr(Decided {
            value: Notice::SoftIntervention(reason),
            ..
        }) => shown(
            None,
            Action::TweetInterstitial(TweetInterstitial {
                soft_intervention: Some(soft_intervention(reason)),
                avoid: Some(normal_avoid()),
                ..TweetInterstitial::default()
            }),
        ),
        Verdict::Fosnr(Decided {
            value:
                Notice::Appealable {
                    reason,
                    limited_actions,
                },
            ..
        }) => shown(
            Some(fosnr_safety_result_reason(reason)),
            Action::TweetInterstitial(TweetInterstitial {
                limited_engagements: Some(fosnr_limited_engagements(reason, limited_actions)),
                avoid: Some(normal_avoid()),
                appealable: Some(Appealable::new(appealable_reason(reason), None)),
                ..TweetInterstitial::default()
            }),
        ),
        Verdict::Shown {
            label_interstitial:
                Some(Decided {
                    value: interstitial,
                    ..
                }),
            media,
            engagement,
        } => {
            let mut limit = LimitedEngagement::new(interstitial.limited_engagement_reason());
            for reason in engagement.iter().flat_map(|held| held.value.reasons()) {
                limit.add(reason);
            }
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
            let reason = match interstitial {
                LabelInterstitial::PublicInterest(public_interest) => Some(public_interest.reason),
                LabelInterstitial::EmergencyDynamic(_) => None,
            };
            shown(
                reason,
                Action::TweetInterstitial(TweetInterstitial {
                    interstitial: Some(any_interstitial(interstitial, policies.language())),
                    limited_engagements: Some(limited_engagements(&limit, policies)),
                    media_action,
                    all_media_visibility_results,
                    ..TweetInterstitial::default()
                }),
            )
        }
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: None,
        } => found(None),
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: Some(Decided { value, .. }),
        } => shown(
            None,
            Action::LimitedEngagements(limited_engagements(value, policies)),
        ),
        Verdict::Shown {
            label_interstitial: None,
            media: Some(Decided { value, .. }),
            engagement: None,
        } => match value {
            MediaRestriction::MediaInterstitial(blur) => shown(
                None,
                Action::ComposedMediaVisibilityResults(blurred_media(blur)),
            ),
            MediaRestriction::NsfwInterstitial => {
                found(Some(ThriftFilteredReason::ContainNsfwMedia(true)))
            }
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
            shown(
                None,
                Action::TweetInterstitial(TweetInterstitial {
                    interstitial,
                    limited_engagements: Some(limited_engagements(&limit.value, policies)),
                    all_media_visibility_results,
                    ..TweetInterstitial::default()
                }),
            )
        }
    }
}

pub(crate) fn rerendered_state(reason: RerenderReason) -> TweetFieldsResultState {
    use ThriftFilteredReason as T;
    let reason = match reason {
        RerenderReason::AuthorDeactivated => T::AuthorIsDeactivated(true),
        RerenderReason::AuthorInactive => T::AuthorAccountIsInactive(true),
        RerenderReason::AuthorSuspended => T::AuthorIsSuspended(true),
        RerenderReason::AuthorBlocksViewer => T::AuthorBlockViewer(true),
        RerenderReason::AuthorProtected => T::AuthorIsProtected(true),
        RerenderReason::Unspecified => T::UnspecifiedReason(true),
    };
    TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason))
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
                    translate(resname, language)
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
                Box::new(localized_message(message, link.as_deref(), language)),
            ))
        }
    }
}

fn localized_message(text: String, link: Option<&str>, language: &str) -> LocalizedMessage {
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
    LocalizedMessage::new(text, language.to_string(), links)
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
            let prompt = action
                .prompt
                .as_ref()
                .map(|prompt| limited_action_prompt(prompt, policies.language()));
            actions.push(LimitedAction::new(action_type, prompt));
        }
    }
    LimitedEngagements::new(
        limited_engagement_reason(limit.reason()),
        (!actions.is_empty()).then(|| LimitedActionsPolicy::new(actions)),
        limit.reason().limited_actions_string().to_string(),
    )
}

fn limited_action_prompt(prompt: &Prompt, language: &str) -> LimitedActionPrompt {
    let headline = prompt.headline.to_string();
    let subtext = prompt.subtext.to_string();
    match &prompt.kind {
        PromptKind::Basic => LimitedActionPrompt::BasicLimitedActionPrompt(
            BasicLimitedActionPrompt::new(headline, subtext),
        ),
        PromptKind::SeeConversation => LimitedActionPrompt::CtaLimitedActionPrompt(
            CtaLimitedActionPrompt::new(headline, subtext, LimitedActionCtaType::SEE_CONVERSATION),
        ),
        PromptKind::LearnMore { link_text, url } => {
            LimitedActionPrompt::LocalizedMessageLimitedActionPrompt(
                LocalizedMessageLimitedActionPrompt::new(
                    LocalizedMessage::new(headline, language.to_string(), Vec::new()),
                    LocalizedMessage::new(
                        subtext,
                        language.to_string(),
                        vec![MessageLink::new(
                            LEARN_MORE_PLACEHOLDER.to_string(),
                            link_text.to_string(),
                            url.clone(),
                        )],
                    ),
                ),
            )
        }
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

#[cfg(test)]
mod tests;
