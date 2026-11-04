use super::{Role, Row, Subject};
use crate::models::{AuthorLabel, PostCandidate};
use crate::rules::fixtures::{allow, candidate, dropped};
use crate::rules::SafetyLevel::TimelineHome;
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "nsfw_high_recall_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::NsfwHighRecall)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "nsfw_high_precision_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::NsfwHighPrecision)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_high_precision_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "spam_high_recall_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::SpamHighRecall)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "spam_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "compromised_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::Compromised)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "compromised_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "read_only_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::ReadOnly)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "read_only_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "impersonation_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::ImpersonationHighPrecision)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "impersonation_high_precision_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "nsfw_avatar_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::NsfwAvatarImage)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_avatar_image_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "nsfw_banner_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::NsfwBannerImage)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_banner_image_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "abusive_high_recall_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::AbusiveHighRecall)),
            expect: vec![
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "abusive_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (Role::each([Role::Follower, Role::Author]), allow()),
            ],
            at: vec![],
        },
        Row {
            name: "nsfw_near_perfect_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::NsfwNearPerfect)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_near_perfect_user_label/drop/unspecified",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "do_not_amplify_user_label",
            subject: Subject::Post(user_labeled(AuthorLabel::DoNotAmplify)),
            expect: vec![
                (
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "do_not_amplify_user_label/drop/unspecified",
                    ),
                ),
                (Role::each([Role::Follower, Role::Author]), allow()),
            ],
            at: vec![],
        },
    ]
}

fn user_labeled(label: AuthorLabel) -> PostCandidate {
    candidate().with_author_user_label(label).build()
}
