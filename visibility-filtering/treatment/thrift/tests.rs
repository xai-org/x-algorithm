use super::*;
use crate::rules::fixtures::{
    allow, appealed, blurred, blurred_and_limited, dropped, fosnr, label_interstitial,
    legacy_interstitial, limited, limited_for, tombstoned, verify_blurred, with_label_interstitial,
};
use crate::rules::RuleId;
use crate::treatment::proto::proto_action;
use crate::treatment::rpc::{decided_rows, verdict_action, BY_RULE_ACTIONS};
use vf_pb::action::Kind;
use xai_visibility_filtering::graphql_results::{
    is_dropped_media_blur, resolve_blurred_image_interstitial,
};
use xai_visibility_filtering::models::{KeywordMatch, SafetyResult};
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::safety_result::SafetyResult as ThriftSafetyResult;
use SafetyLevel::{FilterAll, TimelineHome};

fn policies() -> LimitedActionsPolicies {
    LimitedActionsPolicies::for_tests(vec![(
        LimitedEngagementReason::ConversationControl,
        vec![LimitedActionType::Reply],
    )])
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
                        soft_intervention_reason: Some(SoftInterventionReason::Fosnr(Box::new(
                            AppealableReason::new(
                                1,
                                Box::new(action::AppealablePolicy::ABUSE),
                                true,
                                false,
                            ),
                        ))),
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
            rerenders
                .iter()
                .map(move |&(by, reason)| (level, by, rerendered_state(&Rerendered { by, reason })))
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
