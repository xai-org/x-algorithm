use super::{Role, Row, Subject};
use crate::models::{AuthorFeatures, NsfwViewerDropReason, SafetyLabelType, ViewerAge};
use crate::rules::fixtures::{
    allow, blurred, candidate, dropped, labeled, no_stated_age_viewer, nsfw_viewer_dropped,
    sensitive_opt_in_viewer, tweet_candidate, viewer_with_age,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeRecommendations};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::InterstitialReason;

pub(super) const AT_CUTOFF: u64 = (1705536000000 - 1288834974657) << 22;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "nsfw_high_precision_label_at_cutoff",
            subject: Subject::Post(
                candidate()
                    .tweet_id(AT_CUTOFF)
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .build(),
            ),
            expect: vec![],
            at: vec![(
                TimelineHome,
                Role::NonFollower,
                blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_high_precision/blur/sensitive",
                ),
            )],
        },
        Row {
            name: "nsfw_admin_and_user_author_media",
            subject: Subject::Post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_nsfw_user: true,
                        is_nsfw_admin: true,
                        ..Default::default()
                    })
                    .with_media()
                    .build(),
            ),
            expect: vec![],
            at: vec![(
                TimelineHome,
                Role::NonFollower,
                blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_admin/blur/sensitive",
                ),
            )],
        },
        Row {
            name: "gore_and_nsfw_card_image_labels",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)
                    .with_label(SafetyLabelType::NSFW_CARD_IMAGE)
                    .build(),
            ),
            expect: vec![],
            at: vec![(
                TimelineHome,
                Role::NonFollower,
                blurred(
                    InterstitialReason::Violence(true),
                    "gore_and_violence_high_precision/blur",
                ),
            )],
        },
        Row {
            name: "nsfw_high_precision_adult_label",
            subject: Subject::Post(
                candidate()
                    .tweet_id(AT_CUTOFF + (1 << 22))
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .build(),
            ),
            expect: vec![],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    blurred(
                        InterstitialReason::Nudity(true),
                        "nsfw_high_precision/blur/nudity",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
                (TimelineHome, Role::Author, allow()),
            ],
        },
        Row {
            name: "nsfw_high_precision_label",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_HIGH_PRECISION)),
            expect: vec![
                (Role::Author, allow()),
                (
                    Role::each([
                        Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                        Role::NonFollower,
                        Role::Follower,
                    ]),
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_high_precision/drop/nsfw_media",
                    ),
                ),
            ],
            at: vec![
                (
                    TimelineHome,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_high_precision/blur/sensitive",
                    ),
                ),
                (
                    TimelineHome,
                    Role::LoggedOut,
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_high_precision/blur/sensitive",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHome,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
            ],
        },
        Row {
            name: "gore_and_violence_label",
            subject: Subject::Post(labeled(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)),
            expect: vec![
                (Role::Author, allow()),
                (
                    Role::Follower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "gore_and_violence_high_precision/drop/nsfw_media",
                    ),
                ),
            ],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    blurred(
                        InterstitialReason::Violence(true),
                        "gore_and_violence_high_precision/blur",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHome,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "gore_and_violence_high_precision/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_card_image_label",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_CARD_IMAGE)),
            expect: vec![
                (
                    Role::LoggedOut,
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::LoggedOut,
                        "sensitive_viewer_logged_out/drop",
                    ),
                ),
                (
                    Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::HasNoStatedAge,
                        "sensitive_viewer_no_stated_age/drop",
                    ),
                ),
                (
                    Role::Follower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_card_image/drop/nsfw_media",
                    ),
                ),
            ],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_card_image/blur/sensitive",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_card_image/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_admin_author_media",
            subject: Subject::Post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_nsfw_admin: true,
                        ..Default::default()
                    })
                    .with_media()
                    .build(),
            ),
            expect: vec![(
                Role::Follower,
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "nsfw_admin_author/drop/nsfw_media",
                ),
            )],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_admin/blur/sensitive",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::IsUnderage,
                        "sensitive_viewer_underage/drop",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_admin_author/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_user_author_media",
            subject: Subject::Post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_nsfw_user: true,
                        ..Default::default()
                    })
                    .with_media()
                    .build(),
            ),
            expect: vec![(
                Role::Follower,
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "nsfw_user_author/drop/nsfw_media",
                ),
            )],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    blurred(
                        InterstitialReason::SensitiveUser(true),
                        "nsfw_user/blur/sensitive_user",
                    ),
                ),
                (
                    TimelineHome,
                    Role::LoggedOut,
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::LoggedOut,
                        "sensitive_viewer_logged_out/drop",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_user_author/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_admin_flag_media",
            subject: Subject::Post(tweet_candidate(|t| {
                t.nsfw.admin = true;
                t.media.has_media = true;
            })),
            expect: vec![(
                Role::LoggedOut,
                nsfw_viewer_dropped(
                    NsfwViewerDropReason::LoggedOut,
                    "sensitive_viewer_logged_out/drop",
                ),
            )],
            at: vec![(
                TimelineHome,
                Role::NonFollower,
                blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_admin/blur/sensitive",
                ),
            )],
        },
        Row {
            name: "nsfw_user_flag_media",
            subject: Subject::Post(tweet_candidate(|t| {
                t.nsfw.user = true;
                t.media.has_media = true;
            })),
            expect: vec![(
                Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                nsfw_viewer_dropped(
                    NsfwViewerDropReason::HasNoStatedAge,
                    "sensitive_viewer_no_stated_age/drop",
                ),
            )],
            at: vec![(
                TimelineHome,
                Role::NonFollower,
                blurred(
                    InterstitialReason::SensitiveUser(true),
                    "nsfw_user/blur/sensitive_user",
                ),
            )],
        },
    ]
}
