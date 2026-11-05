use super::{Role, Row, Subject};
use crate::models::{RequestFacts, ViewerAge, ViewerFeatures};
use crate::rules::fixtures::{
    allow, author_candidate, author_viewer, dropped, tweet_candidate, viewer_in_country,
    viewer_with_age,
};
use crate::rules::SafetyLevel::{ImmersiveExpandedRecommendations, TimelineHome};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "dmca_media",
            subject: Subject::Post(tweet_candidate(|t| t.media.has_dmca_media = true)),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "dmca_media/drop/unspecified",
                ),
            )],
            at: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "nsfw_user_flag",
            subject: Subject::Post(tweet_candidate(|t| t.nsfw.user = true)),
            expect: vec![(
                Role::each([Role::NonFollower, Role::Author]),
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "nsfw_user_tweet_flag/drop/nsfw_media",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "nsfw_admin_flag",
            subject: Subject::Post(tweet_candidate(|t| t.nsfw.admin = true)),
            expect: vec![(
                Role::each([Role::NonFollower, Role::Author]),
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "nsfw_admin_tweet_flag/drop/nsfw_media",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "nsfw_user_author",
            subject: Subject::Post(author_candidate(|a| a.is_nsfw_user = true)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_user_author/drop/nsfw_media",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![
                (TimelineHome, Role::NonFollower, allow()),
                (
                    TimelineHome,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    allow(),
                ),
                (ImmersiveExpandedRecommendations, Role::Author, allow()),
                (
                    ImmersiveExpandedRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_sensitive_viewer_user/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_admin_author",
            subject: Subject::Post(author_candidate(|a| a.is_nsfw_admin = true)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_admin_author/drop/nsfw_media",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "geo_denied_media_de",
            subject: Subject::Post(tweet_candidate(|t| {
                t.media.geo_deny_list = vec!["de".to_string()]
            })),
            expect: vec![
                (
                    Role::each([
                        Role::As("in_de", viewer_in_country("de")),
                        Role::As(
                            "author_in_de",
                            ViewerFeatures {
                                facts: RequestFacts::default().with_country("de"),
                                ..author_viewer()
                            },
                        ),
                    ]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "geo_restricted_media/drop/unspecified",
                    ),
                ),
                (
                    Role::each([
                        Role::As("in_us", viewer_in_country("us")),
                        Role::NonFollower,
                    ]),
                    allow(),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "geo_allow_listed_media_us",
            subject: Subject::Post(tweet_candidate(|t| {
                t.media.geo_allow_list = vec!["us".to_string()]
            })),
            expect: vec![
                (
                    Role::each([
                        Role::NonFollower,
                        Role::As("in_de", viewer_in_country("de")),
                    ]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "geo_restricted_media/drop/unspecified",
                    ),
                ),
                (Role::As("in_us", viewer_in_country("us")), allow()),
            ],
            at: vec![],
        },
        Row {
            name: "geo_allow_listed_media_uppercase",
            subject: Subject::Post(tweet_candidate(|t| {
                t.media.geo_allow_list = vec!["US".to_string()]
            })),
            expect: vec![(Role::As("in_us", viewer_in_country("us")), allow())],
            at: vec![],
        },
        Row {
            name: "geo_denied_media_uppercase",
            subject: Subject::Post(tweet_candidate(|t| {
                t.media.geo_deny_list = vec!["DE".to_string()]
            })),
            expect: vec![(
                Role::As("in_de", viewer_in_country("de")),
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "geo_restricted_media/drop/unspecified",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "geo_denied_media_worldwide",
            subject: Subject::Post(tweet_candidate(|t| {
                t.media.geo_deny_list = vec!["xx".to_string()]
            })),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "geo_restricted_media/drop/unspecified",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "geo_denied_media_retweet",
            subject: Subject::Post({
                let mut candidate =
                    tweet_candidate(|t| t.media.geo_deny_list = vec!["de".to_string()]);
                candidate.post.source_tweet_id = Some(2);
                candidate
            }),
            expect: vec![(
                Role::As("in_de", viewer_in_country("de")),
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "geo_restricted_media/drop/unspecified",
                ),
            )],
            at: vec![],
        },
    ]
}
