use super::{Role, Row, Subject};
use crate::models::{
    AuthorFeatures, ClientCapability, ClientSwitch, LimitedEngagementReason, PostCandidate,
    RequestFacts, TweetFeatures, ViewerFeatures,
};
use crate::rules::fixtures::{
    allow, candidate, dropped, limited, tweet_candidate, viewer, VIEWER_ID,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use std::num::NonZeroU64;
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "nullcast",
            subject: Subject::Post(tweet_candidate(|t| t.is_nullcast = true)),
            expect: vec![(
                Role::each([Role::NonFollower, Role::Author]),
                dropped(FilteredReason::TweetIsNullcast, "nullcasted_tweet/drop"),
            )],
            at: vec![],
        },
        Row {
            name: "nullcast_retweet",
            subject: Subject::Post({
                let features = TweetFeatures {
                    is_nullcast: true,
                    ..Default::default()
                };
                candidate()
                    .with_tweet_features(features)
                    .retweet_of(2)
                    .build()
            }),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "nullcast_community",
            subject: Subject::Post(tweet_candidate(|t| {
                t.is_nullcast = true;
                t.community_id = NonZeroU64::new(500);
            })),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "protected_author_community_post",
            subject: Subject::Post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_protected: true,
                        ..Default::default()
                    })
                    .with_tweet_features(TweetFeatures {
                        community_id: NonZeroU64::new(500),
                        ..Default::default()
                    })
                    .build(),
            ),
            expect: vec![
                (
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "protected_community_tweet/drop/unspecified",
                    ),
                ),
                (
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "stale_edit",
            subject: Subject::Post(stale_candidate()),
            expect: vec![
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "stale_tweet/drop/unspecified",
                    ),
                ),
                (
                    Role::Author,
                    limited(
                        LimitedEngagementReason::StaleTweet,
                        "stale_tweet/limited_engagement",
                    ),
                ),
            ],
            at: vec![
                (
                    TimelineHome,
                    Role::RetweetSource(Box::new(Role::NonFollower)),
                    allow(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "client_without_stale_tweet_limits",
                        ViewerFeatures {
                            facts: RequestFacts::default().with_client(
                                ClientCapability::DEFAULT.without(ClientSwitch::StaleTweetLimits),
                            ),
                            ..viewer(VIEWER_ID)
                        },
                    ),
                    allow(),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    limited(
                        LimitedEngagementReason::StaleTweet,
                        "stale_tweet/limited_engagement",
                    ),
                ),
            ],
        },
        Row {
            name: "stale_edit_retweet",
            subject: Subject::Post({
                let mut retweet = stale_candidate();
                retweet.post.source_tweet_id = Some(2);
                retweet
            }),
            expect: vec![],
            at: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "current_edit",
            subject: Subject::Post(tweet_candidate(|t| t.latest_edit_tweet_id = Some(1))),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
    ]
}

fn stale_candidate() -> PostCandidate {
    tweet_candidate(|t| t.latest_edit_tweet_id = Some(2))
}
