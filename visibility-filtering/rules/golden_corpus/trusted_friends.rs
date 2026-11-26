use super::{Role, Row, Subject};
use crate::hydration::Hydrator::TrustedFriends;
use crate::models::TweetFeatures;
use crate::rules::fixtures::{allow, candidate, dropped, CandidateBuilder};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    let trusted_friends = || {
        dropped(
            FilteredReason::UnspecifiedReason,
            "trusted_friends_tweet/drop/unspecified",
        )
    };
    vec![
        Row {
            name: "trusted_friends",
            subject: Subject::Post(trusted_friends_post().build()),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower, Role::LoggedOut]),
                    trusted_friends(),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "trusted_friends_list_member_or_owner",
            subject: Subject::Post(trusted_friends_post().with_edge(TrustedFriends).build()),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
    ]
}

fn trusted_friends_post() -> CandidateBuilder {
    candidate().with_tweet_features(TweetFeatures {
        trusted_friends_list_id: Some(7),
        ..TweetFeatures::default()
    })
}
