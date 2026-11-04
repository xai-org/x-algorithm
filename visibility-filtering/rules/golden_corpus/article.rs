use super::{Role, Row, Subject};
use crate::hydration::Hydrator;
use crate::models::{
    ArticleLifecycle, NsfwViewerDropReason, PostCandidate, SafetyLabelType, Verdict, Viewer,
    ViewerAge,
};
use crate::rules::fixtures::{
    allow, author_viewer, blurred, candidate, dropped, nsfw_viewer_dropped, on_client,
    sensitive_opt_in_viewer, tweet_candidate, viewer_with_age, AUTHOR_ID,
};
use crate::rules::SafetyLevel::{
    FilterAll, ImmersiveExpandedRecommendations, NotificationsFollow, NotificationsWriterV2,
    QuotedTweetRules, TimelineHome, TimelineHomeHydration, TimelineHomeRecommendations,
};
use std::num::NonZeroU64;
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::InterstitialReason;

fn article_post(lifecycle: Option<ArticleLifecycle>) -> PostCandidate {
    let mut candidate = tweet_candidate(|t| t.article_id = NonZeroU64::new(1));
    candidate.post.article_lifecycle = lifecycle;
    candidate
}

fn every_role(verdict: fn() -> Verdict) -> Vec<(Role, Verdict)> {
    vec![
        (Role::NonFollower, verdict()),
        (Role::Follower, verdict()),
        (Role::LoggedOut, verdict()),
        (Role::Author, verdict()),
    ]
}

fn unpublished() -> Verdict {
    dropped(
        FilteredReason::UnspecifiedReason,
        "article_tweet_content/drop/unspecified",
    )
}

pub(super) fn rows() -> Vec<Row> {
    let mut author_sensitive_opt_in = sensitive_opt_in_viewer();
    if let Viewer::LoggedIn { id, .. } = &mut author_sensitive_opt_in.viewer {
        *id = AUTHOR_ID;
    }
    let mut rows = vec![
        Row {
            name: "article_draft",
            subject: Subject::Post(article_post(Some(ArticleLifecycle::Draft))),
            expect: every_role(unpublished),
            at: vec![],
        },
        Row {
            name: "article_soft_deleted",
            subject: Subject::Post(article_post(Some(ArticleLifecycle::SoftDeleted))),
            expect: every_role(unpublished),
            at: vec![],
        },
        Row {
            name: "article_without_lifecycle",
            subject: Subject::Post(article_post(None)),
            expect: every_role(unpublished),
            at: vec![],
        },
        Row {
            name: "article_published",
            subject: Subject::Post(article_post(Some(ArticleLifecycle::Published))),
            expect: every_role(allow),
            at: vec![],
        },
        Row {
            name: "no_article",
            subject: Subject::Post(tweet_candidate(|_| {})),
            expect: every_role(allow),
            at: vec![],
        },
    ];
    for (name, body, cover, flagged, restricts) in [
        ("nsfw_article_cover", false, true, true, true),
        ("nsfw_article_body", true, false, true, true),
        ("nsfw_article_without_media", false, false, true, false),
        ("unflagged_article_cover", false, true, false, false),
        ("unflagged_article_body", true, false, false, false),
    ] {
        let mut post = article_post(Some(ArticleLifecycle::Published));
        post.author.features.is_nsfw_user = flagged;
        post.post.tweet_features.nsfw.user = flagged;
        post.post.article_has_media_entities = body;
        post.post.article_has_cover_media = cover;
        let levels = if restricts {
            &[
                TimelineHomeHydration,
                TimelineHome,
                TimelineHomeRecommendations,
                ImmersiveExpandedRecommendations,
            ][..]
        } else {
            &[TimelineHomeHydration][..]
        };
        let mut expect = Vec::new();
        for &level in levels {
            let restricted = |reason, by| match level {
                TimelineHomeRecommendations => dropped(
                    FilteredReason::ContainNsfwMedia,
                    "nsfw_user_author/drop/nsfw_media",
                ),
                ImmersiveExpandedRecommendations => dropped(
                    FilteredReason::ContainNsfwMedia,
                    "nsfw_sensitive_viewer_user/drop/nsfw_media",
                ),
                FilterAll | TimelineHome | TimelineHomeHydration => nsfw_viewer_dropped(reason, by),
                QuotedTweetRules | NotificationsFollow | NotificationsWriterV2 => {
                    unreachable!("a user-only level judges no post")
                }
            };
            expect.extend([
                (
                    level,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    if restricts {
                        restricted(
                            NsfwViewerDropReason::IsUnderage,
                            "sensitive_viewer_underage/drop",
                        )
                    } else {
                        allow()
                    },
                ),
                (
                    level,
                    Role::As(
                        "no_stated_age",
                        on_client("web", "de", viewer_with_age(ViewerAge::NotStated)),
                    ),
                    if restricts {
                        restricted(
                            NsfwViewerDropReason::HasNoStatedAge,
                            "sensitive_viewer_no_stated_age/drop",
                        )
                    } else {
                        allow()
                    },
                ),
                (
                    level,
                    Role::LoggedOut,
                    if restricts {
                        restricted(
                            NsfwViewerDropReason::LoggedOut,
                            "sensitive_viewer_logged_out/drop",
                        )
                    } else {
                        allow()
                    },
                ),
            ]);
        }
        expect.push((
            TimelineHomeHydration,
            Role::As(
                "adult_sensitive_off",
                on_client("web", "us", viewer_with_age(ViewerAge::Known(18))),
            ),
            if restricts {
                blurred(
                    InterstitialReason::SensitiveUser(true),
                    "nsfw_user/blur/sensitive_user",
                )
            } else {
                allow()
            },
        ));
        if !body {
            expect.extend([
                (
                    TimelineHome,
                    Role::As("adult_sensitive_off", viewer_with_age(ViewerAge::Known(18))),
                    if restricts {
                        blurred(
                            InterstitialReason::SensitiveUser(true),
                            "nsfw_user/blur/sensitive_user",
                        )
                    } else {
                        allow()
                    },
                ),
                (
                    TimelineHome,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "author_sensitive_opt_in",
                        on_client("web", "us", author_sensitive_opt_in.clone()),
                    ),
                    allow(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "author_sensitive_off",
                        on_client("web", "us", author_viewer()),
                    ),
                    if restricts {
                        blurred(
                            InterstitialReason::SensitiveUser(true),
                            "nsfw_user/blur/sensitive_user",
                        )
                    } else {
                        allow()
                    },
                ),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    if flagged {
                        dropped(
                            FilteredReason::ContainNsfwMedia,
                            "nsfw_user_author/drop/nsfw_media",
                        )
                    } else {
                        allow()
                    },
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::NonFollower,
                    if flagged {
                        dropped(
                            FilteredReason::ContainNsfwMedia,
                            "nsfw_sensitive_viewer_user/drop/nsfw_media",
                        )
                    } else {
                        allow()
                    },
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
                (ImmersiveExpandedRecommendations, Role::Author, allow()),
            ]);
        }
        rows.push(Row {
            name,
            subject: Subject::Post(post),
            expect: vec![],
            at: expect,
        });
    }
    let mut creator = candidate()
        .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
        .with_edge(Hydrator::SuperFollowsExclusive)
        .build();
    creator.post.tweet_features.article_id = NonZeroU64::new(1);
    creator.post.tweet_features.exclusive_conversation_author_id = Some(42);
    creator.post.article_lifecycle = Some(ArticleLifecycle::Published);
    creator.post.article_has_media_entities = true;
    rows.push(Row {
        name: "exclusive_nsfw_article_body_super_followed",
        subject: Subject::Post(creator),
        expect: vec![
            (
                Role::As(
                    "sensitive_opt_in",
                    on_client("web", "us", sensitive_opt_in_viewer()),
                ),
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "creator_tweet_nsfw/drop/nsfw_media",
                ),
            ),
            (
                Role::As(
                    "author_sensitive_opt_in",
                    on_client("web", "us", author_sensitive_opt_in),
                ),
                allow(),
            ),
        ],
        at: vec![],
    });
    rows
}
