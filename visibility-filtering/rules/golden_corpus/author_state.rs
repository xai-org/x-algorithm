use super::{Role, Row, Subject};
use crate::models::{AuthorFeatures, PostCandidate, SafetyLabelType, Verdict};
use crate::rules::fixtures::{allow, appealed, author_candidate, candidate, dropped};
use crate::rules::SafetyLevel::{
    TimelineHome, TimelineHomeHydration, TimelineHomePromotedHydration,
};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::AppealablePolicy;

fn fosnr_post(set: fn(&mut AuthorFeatures)) -> PostCandidate {
    let mut features = AuthorFeatures::default();
    set(&mut features);
    candidate()
        .with_author_features(features)
        .with_label(SafetyLabelType::FOSNR_HATEFUL_CONDUCT)
        .build()
}

fn author_appeals() -> Vec<(Role, Verdict)> {
    vec![(
        Role::Author,
        appealed(
            AppealablePolicy::HATEFUL_CONDUCT,
            3,
            true,
            false,
            "fosnr_author/appealable",
        ),
    )]
}

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "suspended_author",
            subject: Subject::Post(author_candidate(|a| a.is_suspended = true)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
                (Role::Author, allow()),
            ],
            at: vec![
                (TimelineHomePromotedHydration, Role::NonFollower, allow()),
                (
                    TimelineHomePromotedHydration,
                    Role::InnerQuoted(Box::new(Role::NonFollower)),
                    allow(),
                ),
            ],
        },
        Row {
            name: "suspended_author_nsfw_media",
            subject: Subject::Post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_suspended: true,
                        ..Default::default()
                    })
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_media()
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
            )],
            at: vec![],
        },
        Row {
            name: "suspended_author_bounced_post",
            subject: Subject::Post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_suspended: true,
                        ..Default::default()
                    })
                    .with_label(SafetyLabelType::BOUNCE)
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
            )],
            at: vec![],
        },
        Row {
            name: "deactivated_author",
            subject: Subject::Post(author_candidate(|a| a.is_deactivated = true)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::AuthorIsDeactivated,
                        "deactivated_author/drop",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![(TimelineHomePromotedHydration, Role::NonFollower, allow())],
        },
        Row {
            name: "suspended_and_deactivated_author",
            subject: Subject::Post(author_candidate(|a| {
                a.is_suspended = true;
                a.is_deactivated = true;
            })),
            expect: vec![],
            at: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::AuthorIsDeactivated,
                        "deactivated_author/drop",
                    ),
                ),
            ],
        },
        Row {
            name: "erased_author",
            subject: Subject::Post(author_candidate(|a| a.is_erased = true)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "erased_author/drop/inactive",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![(TimelineHomePromotedHydration, Role::NonFollower, allow())],
        },
        Row {
            name: "offboarded_author",
            subject: Subject::Post(author_candidate(|a| a.is_offboarded = true)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "offboarded_author/drop/inactive",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![(TimelineHomePromotedHydration, Role::NonFollower, allow())],
        },
        Row {
            name: "protected_author",
            subject: Subject::Post(author_candidate(|a| a.is_protected = true)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::LoggedOut]),
                    dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
                ),
                (Role::each([Role::Follower, Role::Author]), allow()),
            ],
            at: vec![
                (TimelineHomePromotedHydration, Role::NonFollower, allow()),
                (
                    TimelineHomePromotedHydration,
                    Role::InnerQuoted(Box::new(Role::NonFollower)),
                    allow(),
                ),
            ],
        },
        Row {
            name: "suspended_author_fosnr_label",
            subject: Subject::Post(fosnr_post(|a| a.is_suspended = true)),
            expect: author_appeals(),
            at: vec![],
        },
        Row {
            name: "deactivated_author_fosnr_label",
            subject: Subject::Post(fosnr_post(|a| a.is_deactivated = true)),
            expect: author_appeals(),
            at: vec![],
        },
        Row {
            name: "erased_author_fosnr_label",
            subject: Subject::Post(fosnr_post(|a| a.is_erased = true)),
            expect: author_appeals(),
            at: vec![],
        },
        Row {
            name: "offboarded_author_fosnr_label",
            subject: Subject::Post(fosnr_post(|a| a.is_offboarded = true)),
            expect: author_appeals(),
            at: vec![],
        },
        Row {
            name: "protected_author_fosnr_label",
            subject: Subject::Post(fosnr_post(|a| a.is_protected = true)),
            expect: author_appeals(),
            at: vec![],
        },
    ]
}
