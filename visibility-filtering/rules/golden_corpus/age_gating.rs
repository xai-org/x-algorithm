use super::{Role, Row, Subject};
use crate::models::{
    NsfwViewerDropReason, PostCandidate, RequestFacts, SafetyLabelType, Viewer, ViewerAge,
    ViewerFeatures, ViewerProfile,
};
use crate::rules::fixtures::{
    allow, blurred, candidate, dropped, labeled, legacy_interstitial, no_stated_age_viewer,
    nsfw_viewer_dropped, sensitive_opt_in_viewer, tweet_candidate, viewer_in_country,
    viewer_with_age, viewer_with_profile, AUTHOR_ID,
};
use crate::rules::SafetyLevel::{
    ImmersiveExpandedRecommendations, TimelineHome, TimelineHomeRecommendations,
};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::InterstitialReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "unflagged_media",
            subject: Subject::Post(candidate().with_media().build()),
            expect: vec![(
                Role::each([
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    Role::LoggedOut,
                    Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                ]),
                allow(),
            )],
            at: vec![],
        },
        Row {
            name: "nsfw_high_recall_media",
            subject: Subject::Post(labeled_media(SafetyLabelType::NSFW_HIGH_RECALL)),
            expect: vec![
                (
                    Role::LoggedOut,
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::LoggedOut,
                        "sensitive_viewer_logged_out/drop",
                    ),
                ),
                (
                    Role::each([
                        Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                        Role::As(
                            "underage_opted_in",
                            viewer_with_profile(ViewerProfile {
                                viewer_age: ViewerAge::Known(17),
                                allows_sensitive_media: true,
                                ..ViewerProfile::default()
                            }),
                        ),
                    ]),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::IsUnderage,
                        "sensitive_viewer_underage/drop",
                    ),
                ),
                (
                    Role::As(
                        "underage_author",
                        ViewerFeatures {
                            viewer: Viewer::LoggedIn {
                                id: AUTHOR_ID,
                                profile: ViewerProfile {
                                    viewer_age: ViewerAge::Known(17),
                                    ..ViewerProfile::default()
                                },
                                has_age_verified_18_label: false,
                            },
                            ..ViewerFeatures::default()
                        },
                    ),
                    allow(),
                ),
                (
                    Role::each([
                        Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                        Role::As(
                            "no_stated_age_in_de",
                            ViewerFeatures {
                                facts: RequestFacts::default().with_country("de"),
                                ..viewer_with_age(ViewerAge::NotStated)
                            },
                        ),
                        Role::As(
                            "no_stated_age_kr_account_in_us",
                            ViewerFeatures {
                                facts: RequestFacts::default().with_country("us"),
                                ..no_stated_age_viewer("kr")
                            },
                        ),
                    ]),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::HasNoStatedAge,
                        "sensitive_viewer_no_stated_age/drop",
                    ),
                ),
            ],
            at: vec![
                (
                    TimelineHome,
                    Role::As("adult", viewer_with_age(ViewerAge::Known(30))),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As("no_stated_age_in_us", no_stated_age_viewer("us")),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "no_stated_age_us_account_in_de",
                        ViewerFeatures {
                            facts: RequestFacts::default().with_country("de"),
                            ..no_stated_age_viewer("us")
                        },
                    ),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "no_stated_age_without_country",
                        viewer_with_age(ViewerAge::NotStated),
                    ),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As("unknown_age_in_gb", viewer_in_country("gb")),
                    allow(),
                ),
            ],
        },
        Row {
            name: "nsfw_high_precision_media",
            subject: Subject::Post(labeled_media(SafetyLabelType::NSFW_HIGH_PRECISION)),
            expect: vec![
                (
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::IsUnderage,
                        "sensitive_viewer_underage/drop",
                    ),
                ),
                (
                    Role::each([
                        Role::NonFollower,
                        Role::InnerQuoted(Box::new(Role::NonFollower)),
                    ]),
                    legacy_interstitial("nsfw_high_precision/legacy_interstitial"),
                ),
                (
                    Role::LoggedOut,
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::LoggedOut,
                        "sensitive_viewer_logged_out/drop",
                    ),
                ),
            ],
            at: vec![
                (ImmersiveExpandedRecommendations, Role::Author, allow()),
                (
                    ImmersiveExpandedRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_sensitive_viewer_tweet/drop/nsfw_media",
                    ),
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
            ],
        },
        Row {
            name: "nsfw_text_label",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_TEXT)),
            expect: vec![(
                Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                nsfw_viewer_dropped(
                    NsfwViewerDropReason::IsUnderage,
                    "sensitive_viewer_underage/drop",
                ),
            )],
            at: vec![(
                TimelineHomeRecommendations,
                Role::As("adult", viewer_with_age(ViewerAge::Known(30))),
                allow(),
            )],
        },
        Row {
            name: "gore_and_violence_media",
            subject: Subject::Post(labeled_media(
                SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION,
            )),
            expect: vec![
                (
                    Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::HasNoStatedAge,
                        "sensitive_viewer_no_stated_age/drop",
                    ),
                ),
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "gore_and_violence_high_precision/drop/nsfw_media",
                    ),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "nsfw_user_flag_media_retweet",
            subject: Subject::Post({
                let mut candidate = tweet_candidate(|t| {
                    t.nsfw.user = true;
                    t.media.has_media = true;
                });
                candidate.post.source_tweet_id = Some(2);
                candidate
            }),
            expect: vec![],
            at: vec![(
                TimelineHome,
                Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                blurred(
                    InterstitialReason::SensitiveUser(true),
                    "nsfw_user/blur/sensitive_user",
                ),
            )],
        },
    ]
}

fn labeled_media(label: SafetyLabelType) -> PostCandidate {
    candidate().with_label(label).with_media().build()
}
