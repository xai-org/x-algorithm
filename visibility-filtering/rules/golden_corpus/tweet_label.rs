use super::{Role, Row, Subject};
use crate::models::{
    LimitedEngagementReason, RequestFacts, SafetyLabelType, TombstoneReason, ViewerFeatures,
    ViewerProfile,
};
use crate::rules::fixtures::{
    allow, appealed, candidate, dropped, labeled, limited, tombstoned, viewer_with_profile,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_visibility_filtering::models::{
    Action, DropReason, FilteredReason, SafetyResult, SafetyResultReason,
};
use xai_x_thrift::action::AppealablePolicy;

fn soft_nsfw_viewer(name: &'static str, is_new_account: bool, switch_on: bool) -> Role {
    let base = viewer_with_profile(ViewerProfile {
        is_new_account,
        ..ViewerProfile::default()
    });
    Role::As(
        name,
        ViewerFeatures {
            facts: RequestFacts::default().with_soft_nsfw_new_user_drop(switch_on),
            ..base
        },
    )
}

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "dynamic_product_ad_label",
            subject: Subject::Post(labeled(SafetyLabelType::DYNAMIC_PRODUCT_AD)),
            expect: vec![(
                Role::each([Role::NonFollower, Role::Author]),
                limited(
                    LimitedEngagementReason::DynamicProductAd,
                    "dynamic_product_ad/limited_engagement",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "soft_nsfw_label",
            subject: Subject::Post(labeled(SafetyLabelType::SOFT_NSFW)),
            expect: vec![
                (
                    soft_nsfw_viewer("new_account_switch_on", true, true),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "soft_nsfw_new_user/drop/unspecified",
                    ),
                ),
                (
                    Role::each([
                        soft_nsfw_viewer("new_account_switch_off", true, false),
                        soft_nsfw_viewer("old_account_switch_on", false, true),
                    ]),
                    allow(),
                ),
            ],
            at: vec![(
                TimelineHomeHydration,
                soft_nsfw_viewer("new_account_switch_on", true, true),
                allow(),
            )],
        },
        Row {
            name: "for_emergency_use_only_label",
            subject: Subject::Post(labeled(SafetyLabelType::FOR_EMERGENCY_USE_ONLY)),
            expect: vec![(
                Role::each([Role::NonFollower, Role::Follower, Role::Author]),
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "for_emergency_use_only/drop/unspecified",
                ),
            )],
            at: vec![(TimelineHomeHydration, Role::Author, allow())],
        },
        Row {
            name: "pdna_label",
            subject: Subject::Post(labeled(SafetyLabelType::PDNA)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(nsfw_high_precision_reason(), "pdna/drop/safety_result"),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "bounce_label",
            subject: Subject::Post(labeled(SafetyLabelType::BOUNCE)),
            expect: vec![
                (
                    Role::each([
                        Role::NonFollower,
                        Role::Follower,
                        Role::InnerQuoted(Box::new(Role::NonFollower)),
                        Role::InnerQuoted(Box::new(Role::LoggedOut)),
                    ]),
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
                (Role::Author, allow()),
                (
                    Role::InnerQuoted(Box::new(Role::Author)),
                    tombstoned(TombstoneReason::Bounced, "bounce_quoted/tombstone/bounced"),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "spam_label",
            subject: Subject::Post(labeled(SafetyLabelType::SPAM)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(FilteredReason::PossiblyUndesirable, "spam/drop/undesirable"),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "spam_and_bounce_labels",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::SPAM)
                    .with_label(SafetyLabelType::BOUNCE)
                    .build(),
            ),
            expect: vec![],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(FilteredReason::PossiblyUndesirable, "spam/drop/undesirable"),
                ),
            ],
        },
        Row {
            name: "fosnr_hateful_conduct_label",
            subject: Subject::Post(labeled(SafetyLabelType::FOSNR_HATEFUL_CONDUCT)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_hateful_conduct/drop/undesirable",
                    ),
                ),
                (
                    Role::Author,
                    appealed(
                        AppealablePolicy::HATEFUL_CONDUCT,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
            at: vec![(TimelineHome, Role::Author, allow())],
        },
        Row {
            name: "fosnr_violent_speech_label",
            subject: Subject::Post(labeled(SafetyLabelType::FOSNR_VIOLENT_SPEECH)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_violent_speech/drop/undesirable",
                    ),
                ),
                (
                    Role::Author,
                    appealed(
                        AppealablePolicy::VIOLENT_SPEECH,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
            at: vec![(TimelineHome, Role::Author, allow())],
        },
        Row {
            name: "fosnr_abuse_label",
            subject: Subject::Post(labeled(SafetyLabelType::FOSNR_ABUSE)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse/drop/undesirable",
                    ),
                ),
                (
                    Role::Author,
                    appealed(
                        AppealablePolicy::ABUSE,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
            at: vec![(TimelineHome, Role::Author, allow())],
        },
        Row {
            name: "fosnr_civic_integrity_label",
            subject: Subject::Post(labeled(SafetyLabelType::FOSNR_CIVIC_INTEGRITY)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_civic_integrity/drop/undesirable",
                    ),
                ),
                (
                    Role::Author,
                    appealed(
                        AppealablePolicy::CIVIC_INTEGRITY,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
            at: vec![(TimelineHome, Role::Author, allow())],
        },
        Row {
            name: "fosnr_level_3_and_level_1_labels",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::FOSNR_VIOLENT_SPEECH)
                    .with_label(SafetyLabelType::FOSNR_ABUSE_INSULTS)
                    .with_agent_label(SafetyLabelType::FOSNR_CIVIC_INTEGRITY)
                    .build(),
            ),
            expect: vec![
                (
                    Role::Author,
                    appealed(
                        AppealablePolicy::CIVIC_INTEGRITY,
                        3,
                        false,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_violent_speech/drop/undesirable",
                    ),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "fosnr_hateful_conduct_and_bounce_labels",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::FOSNR_HATEFUL_CONDUCT)
                    .with_label(SafetyLabelType::BOUNCE)
                    .build(),
            ),
            expect: vec![],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_hateful_conduct/drop/undesirable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_abuse_insults_and_bounce_labels",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::FOSNR_ABUSE_INSULTS)
                    .with_label(SafetyLabelType::BOUNCE)
                    .build(),
            ),
            expect: vec![
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults_non_follower/drop/undesirable",
                    ),
                ),
                (
                    Role::Follower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
            ],
            at: vec![],
        },
    ]
}

fn nsfw_high_precision_reason() -> FilteredReason {
    FilteredReason::SafetyResult(SafetyResult {
        reason: Some(SafetyResultReason::NsfwHighPrecision),
        action: Action::Drop(DropReason {}),
    })
}
