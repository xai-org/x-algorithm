use super::interstitial::AT_CUTOFF;
use super::{Role, Row, Subject};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::VerifyBlurSupport::{
    AndroidNeedsUpdate, IosNeedsUpdate, Supported, Unsupported,
};
use crate::models::{
    AuthorFeatures, ClientCapability, ClientSwitch, LimitedEngagementReason, NsfwFeature,
    NsfwViewerDropReason, PostCandidate, RequestFacts, SafetyLabelType, TombstoneReason,
    TweetFeatures, Verdict, VerifyBlurSupport, Viewer, ViewerFeatures, ViewerProfile,
};
use crate::rules::fixtures::{
    allow, author_viewer, blurred, blurred_and_limited, candidate, labeled, legacy_interstitial,
    limited, nsfw_high_precision_media, nsfw_viewer_dropped, on_client, sensitive_opt_in_viewer,
    tombstoned, verify_blurred, viewer, viewer_with_profile, AUTHOR_ID, CLIENT_CLASSES, VIEWER_ID,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_x_thrift::action::InterstitialReason;

fn client_classes() -> impl Iterator<Item = (VerifyBlurSupport, Role)> {
    CLIENT_CLASSES.into_iter().map(|class| {
        let support = class
            .capability
            .verify_blur_support
            .expect("every client class resolves a Verify blur support");
        let viewer = on_client(class.name, "fr", viewer(VIEWER_ID));
        (support, Role::As(class.name, viewer))
    })
}

fn per_client_class(verdict: impl Fn(VerifyBlurSupport) -> Verdict) -> Vec<(Role, Verdict)> {
    client_classes()
        .map(|(support, role)| (role, verdict(support)))
        .collect()
}

fn web_in_fr() -> Role {
    Role::As("web_in_fr", on_client("web", "fr", viewer(VIEWER_ID)))
}

fn web_in_us() -> Role {
    Role::As("web_in_us", on_client("web", "us", viewer(VIEWER_ID)))
}

fn author_web_in_fr() -> Role {
    Role::As("author_web_in_fr", on_client("web", "fr", author_viewer()))
}

fn author_web_in_us() -> Role {
    Role::As("author_web_in_us", on_client("web", "us", author_viewer()))
}

fn sensitive_opt_in_author(name: &'static str, country: &str) -> Role {
    Role::As(
        name,
        ViewerFeatures {
            viewer: Viewer::LoggedIn {
                id: AUTHOR_ID,
                profile: ViewerProfile {
                    allows_sensitive_media: true,
                    ..ViewerProfile::default()
                },
                has_age_verified_18_label: false,
            },
            ..on_client("web", country, author_viewer())
        },
    )
}

fn sensitive_opt_in_web_in_fr() -> Role {
    Role::As(
        "sensitive_opt_in_web_in_fr",
        on_client(
            "web",
            "fr",
            viewer_with_profile(ViewerProfile {
                allows_sensitive_media: true,
                ..ViewerProfile::default()
            }),
        ),
    )
}

fn sensitive_opt_in_web_in_us() -> Role {
    Role::As(
        "sensitive_opt_in_web_in_us",
        on_client("web", "us", sensitive_opt_in_viewer()),
    )
}

fn nsfw_flag(nsfw: NsfwFeature) -> TweetFeatures {
    TweetFeatures {
        nsfw,
        ..TweetFeatures::default()
    }
}

fn nsfw_admin_flag_media() -> PostCandidate {
    candidate()
        .with_tweet_features(nsfw_flag(NsfwFeature {
            admin: true,
            user: false,
        }))
        .with_media()
        .build()
}

pub(super) fn rows() -> Vec<Row> {
    let verify_high_precision = || {
        verify_blurred(
            InterstitialReason::Sensitive(true),
            "nsfw_high_precision/blur/sensitive/age_prompt",
        )
    };
    let plain_high_precision = || {
        blurred(
            InterstitialReason::Sensitive(true),
            "nsfw_high_precision/blur/sensitive",
        )
    };
    let high_precision_ios_tombstone = || {
        tombstoned(
            TombstoneReason::UpdateAppIos,
            "nsfw_high_precision/tombstone/update_app_ios",
        )
    };
    let high_precision_tombstone = |support: VerifyBlurSupport| match support {
        Unsupported => tombstoned(
            TombstoneReason::SensitiveViewerAgeVerification,
            "nsfw_high_precision/tombstone/age_verification",
        ),
        IosNeedsUpdate => high_precision_ios_tombstone(),
        AndroidNeedsUpdate => tombstoned(
            TombstoneReason::UpdateAppAndroid,
            "nsfw_high_precision/tombstone/update_app_android",
        ),
        Supported => verify_high_precision(),
    };
    let gore = |support| match support {
        Unsupported => legacy_interstitial("gore_and_violence_high_precision/legacy_interstitial"),
        IosNeedsUpdate => tombstoned(
            TombstoneReason::UpdateAppIos,
            "gore_and_violence_high_precision/tombstone/update_app_ios",
        ),
        AndroidNeedsUpdate => tombstoned(
            TombstoneReason::UpdateAppAndroid,
            "gore_and_violence_high_precision/tombstone/update_app_android",
        ),
        Supported => blurred(
            InterstitialReason::Violence(true),
            "gore_and_violence_high_precision/blur",
        ),
    };
    let (gore_blur_classes, gore_other_classes): (Vec<_>, Vec<_>) =
        client_classes().partition(|(support, _)| matches!(support, Supported));
    vec![
        Row {
            name: "nsfw_high_precision_media_per_client_class",
            subject: Subject::Post(nsfw_high_precision_media()),
            expect: per_client_class(high_precision_tombstone),
            at: vec![],
        },
        Row {
            name: "nsfw_high_precision_media_for_an_ios_outdated_viewer",
            subject: Subject::Post(nsfw_high_precision_media()),
            expect: vec![(
                Role::As(
                    "sensitive_opt_in_ios_outdated_in_fr",
                    on_client("ios_outdated", "fr", sensitive_opt_in_viewer()),
                ),
                high_precision_ios_tombstone(),
            )],
            at: vec![
                (
                    TimelineHome,
                    Role::As(
                        "ios_outdated_in_fr",
                        on_client("ios_outdated", "fr", viewer(VIEWER_ID)),
                    ),
                    plain_high_precision(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "age_verified_18_ios_outdated_in_fr",
                        on_client(
                            "ios_outdated",
                            "fr",
                            ViewerFeatures {
                                viewer: Viewer::LoggedIn {
                                    id: VIEWER_ID,
                                    profile: ViewerProfile::default(),
                                    has_age_verified_18_label: true,
                                },
                                ..ViewerFeatures::default()
                            },
                        ),
                    ),
                    plain_high_precision(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "author_ios_outdated_in_fr",
                        on_client("ios_outdated", "fr", author_viewer()),
                    ),
                    plain_high_precision(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "idv_premium_ios_outdated_in_fr",
                        on_client(
                            "ios_outdated",
                            "fr",
                            viewer_with_profile(ViewerProfile {
                                has_idv_premium: true,
                                ..ViewerProfile::default()
                            }),
                        ),
                    ),
                    plain_high_precision(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "ios_outdated_in_us",
                        on_client("ios_outdated", "us", viewer(VIEWER_ID)),
                    ),
                    plain_high_precision(),
                ),
            ],
        },
        Row {
            name: "nsfw_admin_flag_media_per_client_class",
            subject: Subject::Post(nsfw_admin_flag_media()),
            expect: per_client_class(|support| match support {
                Unsupported => tombstoned(
                    TombstoneReason::SensitiveViewerAgeVerification,
                    "nsfw_account/tombstone/age_verification",
                ),
                IosNeedsUpdate => tombstoned(
                    TombstoneReason::UpdateAppIos,
                    "nsfw_account/tombstone/update_app_ios",
                ),
                AndroidNeedsUpdate => tombstoned(
                    TombstoneReason::UpdateAppAndroid,
                    "nsfw_account/tombstone/update_app_android",
                ),
                Supported => verify_blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_admin/blur/sensitive/age_prompt",
                ),
            }),
            at: vec![],
        },
        Row {
            name: "nsfw_card_image_label_per_client_class",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_CARD_IMAGE)),
            expect: per_client_class(|support| match support {
                Unsupported => tombstoned(
                    TombstoneReason::SensitiveViewerAgeVerification,
                    "nsfw_card_image/tombstone/age_verification",
                ),
                IosNeedsUpdate => tombstoned(
                    TombstoneReason::UpdateAppIos,
                    "nsfw_card_image/tombstone/update_app_ios",
                ),
                AndroidNeedsUpdate => tombstoned(
                    TombstoneReason::UpdateAppAndroid,
                    "nsfw_card_image/tombstone/update_app_android",
                ),
                Supported => verify_blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_card_image/blur/sensitive/age_prompt",
                ),
            }),
            at: vec![],
        },
        Row {
            name: "gore_and_violence_label_per_client_class",
            subject: Subject::Post(labeled(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)),
            expect: gore_other_classes
                .into_iter()
                .map(|(support, role)| (role, gore(support)))
                .collect(),
            at: gore_blur_classes
                .into_iter()
                .map(|(support, role)| (TimelineHomeHydration, role, gore(support)))
                .collect(),
        },
        Row {
            name: "nsfw_high_precision_media_for_a_web_viewer",
            subject: Subject::Post(nsfw_high_precision_media()),
            expect: vec![
                (
                    Role::each([web_in_fr(), sensitive_opt_in_web_in_fr()]),
                    verify_high_precision(),
                ),
                (
                    sensitive_opt_in_author("sensitive_opt_in_author_web_in_fr", "fr"),
                    allow(),
                ),
                (
                    Role::As(
                        "logged_out_web_in_fr",
                        ViewerFeatures {
                            viewer: Viewer::LoggedOut,
                            ..on_client("web", "fr", viewer(VIEWER_ID))
                        },
                    ),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::LoggedOut,
                        "sensitive_viewer_logged_out/drop",
                    ),
                ),
            ],
            at: vec![
                (TimelineHome, web_in_fr(), plain_high_precision()),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "age_verified_18_web_in_fr",
                        ViewerFeatures {
                            viewer: Viewer::LoggedIn {
                                id: VIEWER_ID,
                                profile: ViewerProfile::default(),
                                has_age_verified_18_label: true,
                            },
                            ..on_client("web", "fr", viewer(VIEWER_ID))
                        },
                    ),
                    plain_high_precision(),
                ),
                (
                    TimelineHomeHydration,
                    author_web_in_fr(),
                    plain_high_precision(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "idv_premium_web_in_fr",
                        on_client(
                            "web",
                            "fr",
                            viewer_with_profile(ViewerProfile {
                                has_idv_premium: true,
                                ..ViewerProfile::default()
                            }),
                        ),
                    ),
                    plain_high_precision(),
                ),
                (TimelineHomeHydration, web_in_us(), plain_high_precision()),
            ],
        },
        Row {
            name: "nsfw_high_precision_media_with_a_failed_viewer_lookup",
            subject: Subject::Post({
                let mut candidate = nsfw_high_precision_media();
                candidate.failed =
                    Hydrators::of(Hydrator::ViewerProfile).with(Hydrator::ViewerLabels);
                candidate
            }),
            expect: vec![
                (web_in_fr(), verify_high_precision()),
                (
                    Role::As(
                        "ios_outdated_in_fr",
                        on_client("ios_outdated", "fr", viewer(VIEWER_ID)),
                    ),
                    high_precision_ios_tombstone(),
                ),
            ],
            at: vec![(TimelineHomeHydration, web_in_us(), plain_high_precision())],
        },
        Row {
            name: "nsfw_high_precision_media_from_a_blocking_author",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_media()
                    .with_edge(Hydrator::BlockedByAuthor)
                    .build(),
            ),
            expect: vec![(
                web_in_fr(),
                blurred_and_limited(
                    verify_high_precision(),
                    limited(
                        LimitedEngagementReason::BlockedViewer,
                        "blocked_viewer/limited_engagement",
                    ),
                ),
            )],
            at: vec![],
        },
        Row {
            name: "nsfw_high_precision_adult_media",
            subject: Subject::Post(
                candidate()
                    .tweet_id(AT_CUTOFF + (1 << 22))
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_media()
                    .build(),
            ),
            expect: vec![
                (
                    web_in_fr(),
                    verify_blurred(
                        InterstitialReason::Nudity(true),
                        "nsfw_high_precision/blur/nudity/age_prompt",
                    ),
                ),
                (sensitive_opt_in_web_in_us(), allow()),
            ],
            at: vec![
                (
                    TimelineHomeHydration,
                    web_in_us(),
                    blurred(
                        InterstitialReason::Nudity(true),
                        "nsfw_high_precision/blur/nudity",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::InnerQuoted(Box::new(web_in_us())),
                    blurred(
                        InterstitialReason::Nudity(true),
                        "nsfw_high_precision/blur/nudity",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    author_web_in_fr(),
                    blurred(
                        InterstitialReason::Nudity(true),
                        "nsfw_high_precision/blur/nudity",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_card_image_label_for_a_web_viewer",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_CARD_IMAGE)),
            expect: vec![
                (
                    web_in_fr(),
                    verify_blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_card_image/blur/sensitive/age_prompt",
                    ),
                ),
                (sensitive_opt_in_web_in_us(), allow()),
            ],
            at: vec![
                (
                    TimelineHomeHydration,
                    web_in_us(),
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_card_image/blur/sensitive",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    author_web_in_fr(),
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_card_image/blur/sensitive",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_admin_flag_media_for_a_web_viewer",
            subject: Subject::Post(nsfw_admin_flag_media()),
            expect: vec![
                (
                    web_in_fr(),
                    verify_blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_admin/blur/sensitive/age_prompt",
                    ),
                ),
                (sensitive_opt_in_web_in_us(), allow()),
            ],
            at: vec![
                (
                    TimelineHomeHydration,
                    web_in_us(),
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_admin/blur/sensitive",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    author_web_in_fr(),
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_admin/blur/sensitive",
                    ),
                ),
            ],
        },
        Row {
            name: "unflagged_media_from_an_nsfw_admin_and_user_author",
            subject: Subject::Post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_nsfw_admin: true,
                        is_nsfw_user: true,
                        ..Default::default()
                    })
                    .with_media()
                    .build(),
            ),
            expect: vec![],
            at: client_classes()
                .map(|(_, role)| (TimelineHomeHydration, role, allow()))
                .chain([
                    (TimelineHomeHydration, web_in_us(), allow()),
                    (
                        TimelineHomeHydration,
                        Role::As("no_client_context", viewer(VIEWER_ID)),
                        allow(),
                    ),
                    (
                        TimelineHomeHydration,
                        Role::LoggedOut,
                        nsfw_viewer_dropped(
                            NsfwViewerDropReason::LoggedOut,
                            "sensitive_viewer_logged_out/drop",
                        ),
                    ),
                    (
                        TimelineHome,
                        web_in_us(),
                        blurred(
                            InterstitialReason::Sensitive(true),
                            "nsfw_admin/blur/sensitive",
                        ),
                    ),
                ])
                .collect(),
        },
        Row {
            name: "nsfw_user_flag_media_for_a_web_viewer",
            subject: Subject::Post(
                candidate()
                    .with_tweet_features(nsfw_flag(NsfwFeature {
                        admin: false,
                        user: true,
                    }))
                    .with_media()
                    .build(),
            ),
            expect: vec![(
                web_in_fr(),
                verify_blurred(
                    InterstitialReason::SensitiveUser(true),
                    "nsfw_user/blur/sensitive_user/age_prompt",
                ),
            )],
            at: vec![
                (
                    TimelineHomeHydration,
                    web_in_us(),
                    blurred(
                        InterstitialReason::SensitiveUser(true),
                        "nsfw_user/blur/sensitive_user",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    author_web_in_us(),
                    blurred(
                        InterstitialReason::SensitiveUser(true),
                        "nsfw_user/blur/sensitive_user",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_admin_flag_gore_media_for_a_web_viewer",
            subject: Subject::Post(
                candidate()
                    .with_tweet_features(nsfw_flag(NsfwFeature {
                        admin: true,
                        user: false,
                    }))
                    .with_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)
                    .with_media()
                    .build(),
            ),
            expect: vec![(
                web_in_fr(),
                verify_blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_admin/blur/sensitive/age_prompt",
                ),
            )],
            at: vec![(
                TimelineHomeHydration,
                web_in_us(),
                blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_admin/blur/sensitive",
                ),
            )],
        },
        Row {
            name: "gore_and_violence_label_for_a_web_viewer",
            subject: Subject::Post(labeled(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)),
            expect: vec![
                (
                    Role::each([sensitive_opt_in_web_in_fr(), sensitive_opt_in_web_in_us()]),
                    blurred(
                        InterstitialReason::Violence(true),
                        "gore_and_violence_ignoring_settings/blur",
                    ),
                ),
                (
                    sensitive_opt_in_author("sensitive_opt_in_author_web_in_us", "us"),
                    allow(),
                ),
                (
                    Role::As(
                        "sensitive_opt_in_without_the_ignore_settings_switch_in_fr",
                        ViewerFeatures {
                            facts: RequestFacts::default().with_country("fr").with_client(
                                ClientCapability {
                                    verify_blur_support: Some(Supported),
                                    ..ClientCapability::DEFAULT
                                        .with(ClientSwitch::ModernBlur)
                                        .with(ClientSwitch::CommunityViewerRemovedLimits)
                                },
                            ),
                            ..sensitive_opt_in_viewer()
                        },
                    ),
                    verify_blurred(
                        InterstitialReason::Violence(true),
                        "gore_and_violence_high_precision/blur/age_prompt",
                    ),
                ),
            ],
            at: vec![
                (
                    TimelineHomeHydration,
                    web_in_fr(),
                    blurred(
                        InterstitialReason::Violence(true),
                        "gore_and_violence_high_precision/blur",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    author_web_in_us(),
                    blurred(
                        InterstitialReason::Violence(true),
                        "gore_and_violence_high_precision/blur",
                    ),
                ),
            ],
        },
    ]
}
