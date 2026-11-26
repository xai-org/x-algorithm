use super::{Role, Row, Subject};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    ClientSwitch, CommunityModeration, LimitedEngagementReason, PostCandidate, TweetFeatures,
    Verdict, ViewerFeatures,
};
use crate::rules::fixtures::{
    allow, author_viewer, candidate, dropped, limited, on_client, viewer, VIEWER_ID,
};
use std::num::NonZeroU64;
use xai_visibility_filtering::models::FilteredReason;

const HIDDEN: CommunityModeration = CommunityModeration {
    is_hidden: true,
    is_author_removed: false,
};

const AUTHOR_REMOVED: CommunityModeration = CommunityModeration {
    is_hidden: false,
    is_author_removed: true,
};

fn community_post(
    moderation: CommunityModeration,
    viewer_is_moderator: Option<bool>,
) -> PostCandidate {
    let mut post = candidate()
        .with_tweet_features(TweetFeatures {
            community_id: NonZeroU64::new(500),
            ..Default::default()
        })
        .build();
    post.post.community_moderation = moderation;
    post.post.viewer_is_community_moderator = viewer_is_moderator;
    post
}

fn hidden_drop() -> Verdict {
    dropped(
        FilteredReason::UnspecifiedReason,
        "hidden_community_tweet/drop/unspecified",
    )
}

fn author_removed_drop() -> Verdict {
    dropped(
        FilteredReason::UnspecifiedReason,
        "author_removed_community_tweet/drop/unspecified",
    )
}

fn viewer_removed_post(viewer_is_removed: bool) -> PostCandidate {
    {
        let mut candidate = community_post(CommunityModeration::default(), Some(false));
        candidate.post.viewer_is_removed_from_community = viewer_is_removed;
        candidate
    }
}

fn viewer_removed_limit() -> Verdict {
    limited(
        LimitedEngagementReason::CommunityTweetViewerRemoved,
        "community_tweet_viewer_removed/limited_engagement",
    )
}

fn ios_before_viewer_removed_limits() -> Role {
    let ios = on_client("ios_current", "us", viewer(VIEWER_ID));
    let client = ios
        .facts
        .client_capability()
        .without(ClientSwitch::CommunityViewerRemovedLimits);
    Role::As(
        "ios_10_15",
        ViewerFeatures {
            facts: ios.facts.with_client(client),
            ..ios
        },
    )
}

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "hidden_community_post",
            subject: Subject::Post(community_post(HIDDEN, Some(false))),
            expect: vec![(
                Role::each([Role::NonFollower, Role::Follower, Role::LoggedOut]),
                hidden_drop(),
            )],
            at: vec![],
        },
        Row {
            name: "hidden_community_post_moderator_lookup_failed",
            subject: Subject::Post(community_post(HIDDEN, None)),
            expect: vec![
                (Role::NonFollower, allow()),
                (Role::LoggedOut, hidden_drop()),
            ],
            at: vec![],
        },
        Row {
            name: "author_removed_community_post",
            subject: Subject::Post(community_post(AUTHOR_REMOVED, Some(false))),
            expect: vec![(
                Role::each([Role::NonFollower, Role::Follower, Role::LoggedOut]),
                author_removed_drop(),
            )],
            at: vec![],
        },
        Row {
            name: "hidden_author_removed_community_post",
            subject: Subject::Post(community_post(
                CommunityModeration {
                    is_hidden: true,
                    is_author_removed: true,
                },
                Some(false),
            )),
            expect: vec![(Role::NonFollower, hidden_drop())],
            at: vec![],
        },
        Row {
            name: "unmoderated_community_post",
            subject: Subject::Post(community_post(CommunityModeration::default(), None)),
            expect: vec![(Role::each([Role::NonFollower, Role::LoggedOut]), allow())],
            at: vec![],
        },
        Row {
            name: "community_post_viewer_removed",
            subject: Subject::Post(viewer_removed_post(true)),
            expect: vec![
                (
                    Role::each([
                        Role::As("web", on_client("web", "us", viewer(VIEWER_ID))),
                        Role::As(
                            "ios_current",
                            on_client("ios_current", "us", viewer(VIEWER_ID)),
                        ),
                        Role::As(
                            "android_current",
                            on_client("android_current", "us", viewer(VIEWER_ID)),
                        ),
                    ]),
                    viewer_removed_limit(),
                ),
                (
                    Role::each([
                        ios_before_viewer_removed_limits(),
                        Role::As("mac_app", on_client("mac_app", "us", viewer(VIEWER_ID))),
                    ]),
                    allow(),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "community_post_author_removed_from_its_community",
            subject: Subject::Post(viewer_removed_post(true)),
            expect: vec![(
                Role::As("author_on_web", on_client("web", "us", author_viewer())),
                viewer_removed_limit(),
            )],
            at: vec![],
        },
        Row {
            name: "community_post_viewer_not_removed",
            subject: Subject::Post(viewer_removed_post(false)),
            expect: vec![(
                Role::each([
                    Role::As("web", on_client("web", "us", viewer(VIEWER_ID))),
                    Role::As(
                        "ios_current",
                        on_client("ios_current", "us", viewer(VIEWER_ID)),
                    ),
                    Role::LoggedOut,
                ]),
                allow(),
            )],
            at: vec![],
        },
        Row {
            name: "community_post_viewer_removal_lookup_failed",
            subject: Subject::Post({
                let mut candidate = viewer_removed_post(false);
                candidate.failed = Hydrators::of(Hydrator::CommunityViewerRemoved);
                candidate
            }),
            expect: vec![(
                Role::As("web", on_client("web", "us", viewer(VIEWER_ID))),
                allow(),
            )],
            at: vec![],
        },
    ]
}
