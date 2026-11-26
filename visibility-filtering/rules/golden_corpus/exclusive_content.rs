use super::{Role, Row, Subject};
use crate::hydration::Hydrator;
use crate::models::{PostCandidate, SafetyLabelType};
use crate::rules::fixtures::{allow, candidate, dropped, CandidateBuilder};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "exclusive",
            subject: Subject::Post(exclusive_candidate(candidate())),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::LoggedOut]),
                    dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "exclusive_super_followed",
            subject: Subject::Post(exclusive_candidate(
                candidate().with_edge(Hydrator::SuperFollowsExclusive),
            )),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "exclusive_retweet",
            subject: Subject::Post(exclusive_candidate(candidate().retweet_of(2))),
            expect: vec![(
                Role::Author,
                dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
            )],
            at: vec![],
        },
        Row {
            name: "exclusive_nsfw_high_precision_media",
            subject: Subject::Post(exclusive_candidate(
                candidate()
                    .with_media()
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION),
            )),
            expect: vec![(
                Role::each([Role::NonFollower, Role::LoggedOut]),
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "creator_tweet_nsfw/drop/nsfw_media",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "exclusive_nsfw_high_precision_media_super_followed",
            subject: Subject::Post(exclusive_candidate(
                candidate()
                    .with_media()
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_edge(Hydrator::SuperFollowsExclusive),
            )),
            expect: vec![(
                Role::NonFollower,
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "creator_tweet_nsfw/drop/nsfw_media",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "exclusive_super_followed_author_blocks_viewer",
            subject: Subject::Post(exclusive_candidate(
                candidate()
                    .with_edge(Hydrator::SuperFollowsExclusive)
                    .with_edge(Hydrator::BlockedByAuthor),
            )),
            expect: vec![
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "author_blocks_viewer_exclusive_content/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "exclusive_author_blocks_viewer",
            subject: Subject::Post(exclusive_candidate(
                candidate().with_edge(Hydrator::BlockedByAuthor),
            )),
            expect: vec![(
                Role::NonFollower,
                dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
            )],
            at: vec![],
        },
    ]
}

fn exclusive_candidate(builder: CandidateBuilder) -> PostCandidate {
    let mut c = builder.build();
    c.post.tweet_features.exclusive_conversation_author_id = Some(42);
    c
}
