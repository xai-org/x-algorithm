use super::{CorpusCase, Role, Row, Subject};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    AuthorLabel, AuthorProfile, EdgeSet, NotificationFilters, NotificationKind, Placement,
    PostCandidate, UserCandidate, UserEntity, Verdict, Viewer, ViewerFeatures, ViewerProfile,
};
use crate::rules::fixtures::{allow, candidate, dropped, AUTHOR_ID, VIEWER_ID};
use crate::rules::SafetyLevel::{self, NotificationsFollow, NotificationsWriterV2};
use std::iter::once;
use xai_visibility_filtering::models::FilteredReason;

struct Case {
    name: &'static str,
    actor: UserEntity,
    edges: EdgeSet,
    recipient: ViewerProfile,
    non_follower: Verdict,
    follower: Verdict,
    actor_self: Option<Verdict>,
}

fn actor(set: impl FnOnce(&mut UserEntity)) -> UserEntity {
    let mut actor = UserEntity {
        id: AUTHOR_ID,
        ..UserEntity::default()
    };
    set(&mut actor);
    actor
}

fn labeled(label: AuthorLabel) -> UserEntity {
    actor(|actor| actor.labels.insert(label))
}

fn filtering(set: fn(&mut NotificationFilters)) -> ViewerProfile {
    let mut notification_filters = NotificationFilters::default();
    set(&mut notification_filters);
    ViewerProfile {
        notification_filters,
        ..ViewerProfile::default()
    }
}

pub(super) fn recipient(id: u64, profile: ViewerProfile) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn {
            id,
            profile,
            has_age_verified_18_label: false,
        },
        ..ViewerFeatures::default()
    }
}

fn unspecified(by: &'static str) -> Verdict {
    dropped(FilteredReason::UnspecifiedReason, by)
}

fn deactivated() -> Verdict {
    dropped(
        FilteredReason::AuthorIsDeactivated,
        "deactivated_author/drop",
    )
}

fn suspended() -> Verdict {
    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop")
}

fn non_follower_drop(
    name: &'static str,
    actor: UserEntity,
    recipient: ViewerProfile,
    by: &'static str,
) -> Case {
    Case {
        name,
        actor,
        edges: EdgeSet::default(),
        recipient,
        non_follower: unspecified(by),
        follower: allow(),
        actor_self: Some(allow()),
    }
}

fn non_author_drop(
    name: &'static str,
    actor: UserEntity,
    recipient: ViewerProfile,
    verdict: fn() -> Verdict,
) -> Case {
    Case {
        name,
        actor,
        edges: EdgeSet::default(),
        recipient,
        non_follower: verdict(),
        follower: verdict(),
        actor_self: Some(allow()),
    }
}

fn cases(
    protected_non_follower: fn(Verdict) -> Verdict,
    only_people_i_follow: &'static str,
) -> Vec<Case> {
    let clean = ViewerProfile::default;
    let mut cases = vec![
        Case {
            name: "protected_actor",
            actor: actor(|actor| actor.features.is_protected = true),
            edges: EdgeSet::default(),
            recipient: clean(),
            non_follower: protected_non_follower(allow()),
            follower: allow(),
            actor_self: Some(allow()),
        },
        Case {
            name: "protected_and_suspended_actor",
            actor: actor(|actor| {
                actor.features.is_protected = true;
                actor.features.is_suspended = true;
            }),
            edges: EdgeSet::default(),
            recipient: clean(),
            non_follower: protected_non_follower(suspended()),
            follower: suspended(),
            actor_self: Some(allow()),
        },
        Case {
            name: "clean_actor",
            actor: actor(|_| {}),
            edges: EdgeSet::default(),
            recipient: clean(),
            non_follower: allow(),
            follower: allow(),
            actor_self: Some(allow()),
        },
        non_author_drop(
            "deactivated_actor",
            actor(|actor| actor.features.is_deactivated = true),
            clean(),
            deactivated,
        ),
        non_author_drop(
            "deactivated_and_suspended_actor",
            actor(|actor| {
                actor.features.is_deactivated = true;
                actor.features.is_suspended = true;
            }),
            clean(),
            deactivated,
        ),
        non_author_drop(
            "erased_actor",
            actor(|actor| actor.features.is_erased = true),
            clean(),
            || {
                dropped(
                    FilteredReason::AuthorAccountIsInactive,
                    "erased_author/drop/inactive",
                )
            },
        ),
        non_author_drop(
            "suspended_actor",
            actor(|actor| actor.features.is_suspended = true),
            clean(),
            suspended,
        ),
        non_author_drop(
            "deactivated_recipient",
            actor(|_| {}),
            ViewerProfile {
                is_deactivated: true,
                ..clean()
            },
            || unspecified("deactivated_viewer/drop/unspecified"),
        ),
        non_author_drop(
            "suspended_recipient",
            actor(|_| {}),
            ViewerProfile {
                is_suspended: true,
                ..clean()
            },
            || unspecified("suspended_viewer/drop/unspecified"),
        ),
        Case {
            name: "recipient_blocks_actor",
            actor: actor(|_| {}),
            edges: EdgeSet::default().with(Hydrator::Blocks),
            recipient: clean(),
            non_follower: dropped(
                FilteredReason::ViewerBlocksAuthor,
                "viewer_blocks_author/drop",
            ),
            follower: dropped(
                FilteredReason::ViewerBlocksAuthor,
                "viewer_blocks_author/drop",
            ),
            actor_self: None,
        },
        Case {
            name: "recipient_mutes_actor",
            actor: actor(|_| {}),
            edges: EdgeSet::default().with(Hydrator::Mutes),
            recipient: clean(),
            non_follower: dropped(
                FilteredReason::ViewerMutesAuthor,
                "viewer_mutes_and_does_not_follow_author/drop/viewer_mutes_author",
            ),
            follower: allow(),
            actor_self: Some(allow()),
        },
        non_follower_drop(
            "actor_without_confirmed_email",
            actor(|actor| actor.lacks_confirmed_email = true),
            filtering(|filters| filters.no_confirmed_email = true),
            "no_confirmed_email/drop/unspecified",
        ),
        non_follower_drop(
            "actor_without_verified_phone",
            actor(|actor| actor.lacks_verified_phone = true),
            filtering(|filters| filters.no_confirmed_phone = true),
            "no_confirmed_phone/drop/unspecified",
        ),
        non_follower_drop(
            "actor_with_default_profile_image",
            actor(|actor| {
                actor.profile = AuthorProfile {
                    has_default_profile_image: true,
                    ..AuthorProfile::default()
                };
            }),
            filtering(|filters| filters.default_profile_image = true),
            "no_default_profile_image/drop/unspecified",
        ),
        non_follower_drop(
            "new_actor",
            actor(|actor| {
                actor.profile = AuthorProfile {
                    is_new_account: true,
                    ..AuthorProfile::default()
                };
            }),
            filtering(|filters| filters.new_users = true),
            "no_new_users/drop/unspecified",
        ),
        non_follower_drop(
            "actor_not_following_recipient",
            actor(|_| {}),
            filtering(|filters| filters.not_followed_by = true),
            "no_not_followed_by/drop/unspecified",
        ),
        Case {
            name: "actor_following_recipient",
            actor: actor(|_| {}),
            edges: EdgeSet::default().with(Hydrator::AuthorFollowsViewer),
            recipient: filtering(|filters| filters.not_followed_by = true),
            non_follower: allow(),
            follower: allow(),
            actor_self: Some(allow()),
        },
        non_follower_drop(
            "recipient_hears_only_people_followed",
            actor(|_| {}),
            ViewerProfile {
                mention_filter_is_following: true,
                ..clean()
            },
            only_people_i_follow,
        ),
        Case {
            name: "unfiltered_actor",
            actor: actor(|actor| {
                actor.lacks_confirmed_email = true;
                actor.lacks_verified_phone = true;
                actor.profile = AuthorProfile {
                    has_default_profile_image: true,
                    is_new_account: true,
                };
            }),
            edges: EdgeSet::default(),
            recipient: clean(),
            non_follower: allow(),
            follower: allow(),
            actor_self: Some(allow()),
        },
    ];
    for (label, by) in [
        (
            AuthorLabel::ReadOnly,
            "read_only_user_label/drop/unspecified",
        ),
        (
            AuthorLabel::Compromised,
            "compromised_user_label/drop/unspecified",
        ),
        (
            AuthorLabel::SpamHighRecall,
            "spam_high_recall_user_label/drop/unspecified",
        ),
        (
            AuthorLabel::ImpersonationHighPrecision,
            "impersonation_high_precision_user_label/drop/unspecified",
        ),
    ] {
        cases.push(Case {
            name: by,
            actor: labeled(label),
            edges: EdgeSet::default(),
            recipient: clean(),
            non_follower: unspecified(by),
            follower: unspecified(by),
            actor_self: Some(allow()),
        });
    }
    for (label, by) in [
        (
            AuthorLabel::AbusiveHighRecall,
            "abusive_high_recall_user_label/drop/unspecified",
        ),
        (
            AuthorLabel::NotGraduated,
            "not_graduated_user_label/drop/unspecified",
        ),
        (
            AuthorLabel::AgathaSpamTopUser,
            "agatha_spam_top_user_user_label/drop/unspecified",
        ),
        (
            AuthorLabel::EngagementSpammer,
            "engagement_spammer_user_label/drop/unspecified",
        ),
        (
            AuthorLabel::DownrankSpamReply,
            "downrank_spam_reply_user_label/drop/unspecified",
        ),
    ] {
        cases.push(non_follower_drop(by, labeled(label), clean(), by));
    }
    let uqf = || ViewerProfile {
        has_uqf_enabled: true,
        ..clean()
    };
    cases.extend([
        non_follower_drop(
            "blink_worst_actor",
            labeled(AuthorLabel::BlinkWorst),
            uqf(),
            "blink_worst_user_label/drop/unspecified",
        ),
        Case {
            name: "blink_worst_actor_recipient_without_uqf",
            actor: labeled(AuthorLabel::BlinkWorst),
            edges: EdgeSet::default(),
            recipient: clean(),
            non_follower: allow(),
            follower: allow(),
            actor_self: Some(allow()),
        },
        Case {
            name: "blink_worst_actor_blink_labeled_recipient",
            actor: labeled(AuthorLabel::BlinkWorst),
            edges: EdgeSet::default(),
            recipient: ViewerProfile {
                has_blink_label: true,
                ..uqf()
            },
            non_follower: allow(),
            follower: allow(),
            actor_self: Some(allow()),
        },
    ]);
    cases
}

impl Case {
    fn rows(self, level: SafetyLevel) -> [Row; 2] {
        let user = |edges| UserCandidate {
            user: self.actor.clone(),
            edges,
            quote: None,
            failed: Hydrators::empty(),
        };
        let as_recipient = |name, id| Role::As(name, recipient(id, self.recipient.clone()));
        let follower = self.edges.with(Hydrator::Follows);
        [
            Row {
                name: self.name,
                subject: Subject::User(user(self.edges)),
                expect: vec![],
                at: once((
                    level,
                    as_recipient("non_follower", VIEWER_ID),
                    self.non_follower,
                ))
                .chain(
                    self.actor_self
                        .map(|verdict| (level, as_recipient("self", AUTHOR_ID), verdict)),
                )
                .collect(),
            },
            Row {
                name: self.name,
                subject: Subject::User(user(follower)),
                expect: vec![],
                at: vec![(level, as_recipient("follower", VIEWER_ID), self.follower)],
            },
        ]
    }
}

pub(super) fn rows() -> Vec<Row> {
    let defer_to_actor_state: fn(Verdict) -> Verdict = |otherwise| otherwise;
    let drop_as_protected: fn(Verdict) -> Verdict =
        |_| dropped(FilteredReason::AuthorIsProtected, "protected_author/drop");
    [
        (NotificationsFollow, defer_to_actor_state),
        (NotificationsWriterV2, drop_as_protected),
    ]
    .into_iter()
    .flat_map(|(level, protected_non_follower)| {
        cases(
            protected_non_follower,
            "only_people_i_follow/drop/unspecified",
        )
        .into_iter()
        .flat_map(move |case| case.rows(level))
    })
    .collect()
}

pub(super) fn post_kind_cases(level: SafetyLevel, kind: NotificationKind) -> Vec<CorpusCase> {
    cases(
        |_| dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
        "only_people_i_follow_outside_community/drop/unspecified",
    )
    .into_iter()
    .flat_map(|case| {
        let post = |edges| {
            Subject::Post(PostCandidate {
                author: case.actor.clone(),
                edges,
                ..candidate().build()
            })
        };
        let as_recipient = |role, id, edges, expected| CorpusCase {
            name: format!("{}/{level:?}/{kind:?}/{role}", case.name),
            level,
            kind: Some(kind),
            viewer: recipient(id, case.recipient.clone()),
            subject: post(edges),
            placement: Placement::Standalone,
            expected,
        };
        let follower = case.edges.with(Hydrator::Follows);
        let mut cases = vec![
            as_recipient(
                "non_follower",
                VIEWER_ID,
                case.edges,
                case.non_follower.clone(),
            ),
            as_recipient("follower", VIEWER_ID, follower, case.follower.clone()),
        ];
        cases.extend(
            case.actor_self
                .clone()
                .map(|verdict| as_recipient("self", AUTHOR_ID, case.edges, verdict)),
        );
        cases
    })
    .collect()
}
