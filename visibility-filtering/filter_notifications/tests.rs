use super::*;
use crate::clients::socialgraph_client::Graph;
use crate::hydration::sources::Source;
use crate::hydration::sources::{found_user, suspended, Fault, InMemorySources};
use crate::metrics::families::{
    EVALUATED_CANDIDATES, NOTIFICATION_RESULTS, REQUESTS, REQUESTS_BY_CALLER, UNRESOLVED_TWEETS,
    VERDICTS, VERDICTS_BY_RULE,
};
use crate::metrics::{Metrics, Recording};
use crate::params::ClientSwitches;
use crate::rules::RuleEngine;
use vf_pb::NotificationDropReason as DropReason;
use xai_core_entities::entities::{
    GizmoduckAdvancedFilters, GizmoduckMuteSettings, GizmoduckUser, GizmoduckUserResult, Label,
    Labels, MentionFilter,
};
use xai_x_thrift::user_labels::LabelValue;

const RECIPIENT: u64 = 50;
const FOLLOW: i32 = 246;
const FOLLOW_REQUEST_ACCEPTED: i32 = 72;
const ACTOR: u64 = 1;
const OTHER_ACTOR: u64 = 2;
const SUSPENDED_ACTOR: u64 = 3;
const MISSING_ACTOR: u64 = 4;
const POST: u64 = 100;
const OTHER_POST: u64 = 101;
const SUSPENDED_POST: u64 = 102;
const MISSING_POST: u64 = 999;
const RECIPIENT_POST: u64 = 104;

#[derive(Debug, PartialEq, Eq)]
enum Seen {
    Allowed,
    Dropped(DropReason),
    Failed,
}

fn follow(actor_id: u64) -> vf_pb::Notification {
    vf_pb::Notification {
        actor_id,
        kind: Some(vf_pb::notification::Kind::Follow(vf_pb::Follow {})),
    }
}

fn accepted(actor_id: u64) -> vf_pb::Notification {
    vf_pb::Notification {
        actor_id,
        kind: Some(vf_pb::notification::Kind::FollowRequestAccepted(
            vf_pb::FollowRequestAccepted {},
        )),
    }
}

fn like(actor_id: u64, post_id: u64) -> vf_pb::Notification {
    vf_pb::Notification {
        actor_id,
        kind: Some(vf_pb::notification::Kind::Like(vf_pb::Like { post_id })),
    }
}

fn repost(actor_id: u64, source_post_id: u64) -> vf_pb::Notification {
    vf_pb::Notification {
        actor_id,
        kind: Some(vf_pb::notification::Kind::Repost(vf_pb::Repost {
            source_post_id,
        })),
    }
}

fn mention(actor_id: u64, post_id: u64) -> vf_pb::Notification {
    vf_pb::Notification {
        actor_id,
        kind: Some(vf_pb::notification::Kind::Mention(vf_pb::Mention {
            post_id,
        })),
    }
}

fn reply(actor_id: u64, post_id: u64) -> vf_pb::Notification {
    vf_pb::Notification {
        actor_id,
        kind: Some(vf_pb::notification::Kind::Reply(vf_pb::Reply { post_id })),
    }
}

fn media_tagged(actor_id: u64, post_id: u64) -> vf_pb::Notification {
    vf_pb::Notification {
        actor_id,
        kind: Some(vf_pb::notification::Kind::MediaTagged(vf_pb::MediaTagged {
            post_id,
        })),
    }
}

fn posts_world() -> InMemorySources {
    world()
        .tweet(POST, ACTOR)
        .tweet(OTHER_POST, OTHER_ACTOR)
        .tweet(SUSPENDED_POST, SUSPENDED_ACTOR)
}

fn recipient(filters: GizmoduckAdvancedFilters) -> GizmoduckUser {
    GizmoduckUser {
        user_id: RECIPIENT,
        mute_settings: Some(GizmoduckMuteSettings {
            muted_keywords: vec![],
            advanced_notification_filters: filters,
        }),
        ..Default::default()
    }
}

fn filtering(filters: GizmoduckAdvancedFilters) -> InMemorySources {
    world().viewer(RECIPIENT, recipient(filters))
}

fn actor_with_image(user_id: u64) -> GizmoduckUserResult {
    let mut found = found_user(user_id);
    if let Some(user) = found.user.as_mut() {
        user.profile.has_profile_image = true;
    }
    found
}

fn labeled(user_id: u64, label: LabelValue) -> GizmoduckUserResult {
    let mut found = actor_with_image(user_id);
    if let Some(user) = found.user.as_mut() {
        user.labels = Labels {
            labels: vec![Label {
                label_value: label.0,
                created_at_msec: 0,
            }],
        };
    }
    found
}

fn world() -> InMemorySources {
    InMemorySources::default()
        .viewer(RECIPIENT, recipient(GizmoduckAdvancedFilters::default()))
        .user(ACTOR, actor_with_image(ACTOR))
        .user(OTHER_ACTOR, actor_with_image(OTHER_ACTOR))
        .user(SUSPENDED_ACTOR, suspended())
        .tweet(RECIPIENT_POST, RECIPIENT)
}

async fn filter(
    sources: InMemorySources,
    safety_level: i32,
    recipient_id: u64,
    notifications: Vec<vf_pb::Notification>,
) -> (Result<Vec<Seen>, tonic::Code>, Arc<InMemorySources>) {
    filter_with(
        Metrics::none(),
        sources,
        safety_level,
        recipient_id,
        notifications,
    )
    .await
}

async fn filter_with(
    metrics: Metrics,
    sources: InMemorySources,
    safety_level: i32,
    recipient_id: u64,
    notifications: Vec<vf_pb::Notification>,
) -> (Result<Vec<Seen>, tonic::Code>, Arc<InMemorySources>) {
    let sources = Arc::new(sources);
    let endpoint = FilterNotificationsEndpoint::new(
        Arc::new(PostEvaluator::new(
            Arc::clone(&sources) as _,
            RuleEngine::for_tests(),
            Arc::default(),
            metrics,
        )),
        ClientSwitches::for_tests(),
    );
    let response = endpoint
        .handle(Request::new(vf_pb::FilterNotificationsRequest {
            safety_level,
            recipient_id,
            notifications,
        }))
        .await;
    let seen = response
        .map(|response| {
            use vf_pb::notification_visibility_result::Outcome;
            response
                .into_inner()
                .results
                .into_iter()
                .map(|result| match result.outcome {
                    Some(Outcome::Allowed(_)) => Seen::Allowed,
                    Some(Outcome::Dropped(dropped)) => Seen::Dropped(
                        dropped
                            .reason
                            .and_then(|reason| DropReason::try_from(reason).ok())
                            .unwrap_or(DropReason::Unspecified),
                    ),
                    Some(Outcome::Failed(_)) | None => Seen::Failed,
                })
                .collect()
        })
        .map_err(|status| status.code());
    (seen, sources)
}

async fn follows(sources: InMemorySources, actors: &[u64]) -> Vec<Seen> {
    let notifications = actors.iter().copied().map(follow).collect();
    filter(sources, FOLLOW, RECIPIENT, notifications)
        .await
        .0
        .unwrap()
}

#[tokio::test]
async fn invalid_requests_are_refused_and_no_notifications_answer_empty() {
    let no_kind = vf_pb::Notification {
        actor_id: ACTOR,
        kind: None,
    };
    for (level, recipient_id, notifications) in [
        (FOLLOW, 0, vec![follow(ACTOR)]),
        (FOLLOW, RECIPIENT, vec![follow(ACTOR), no_kind]),
        (
            FOLLOW,
            RECIPIENT,
            vec![follow(ACTOR), accepted(OTHER_ACTOR)],
        ),
        (FOLLOW_REQUEST_ACCEPTED, RECIPIENT, vec![follow(ACTOR)]),
        (FOLLOW, RECIPIENT, vec![like(ACTOR, RECIPIENT_POST)]),
        (
            FOLLOW,
            RECIPIENT,
            vec![follow(ACTOR), repost(OTHER_ACTOR, RECIPIENT_POST)],
        ),
        (FOLLOW_REQUEST_ACCEPTED, RECIPIENT, vec![like(ACTOR, 0)]),
        (9_999, RECIPIENT, vec![follow(ACTOR)]),
        (FOLLOW, RECIPIENT, vec![mention(ACTOR, POST)]),
        (FOLLOW_REQUEST_ACCEPTED, RECIPIENT, vec![mention(ACTOR, 0)]),
        (FOLLOW_REQUEST_ACCEPTED, RECIPIENT, vec![reply(ACTOR, 0)]),
        (
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            vec![media_tagged(0, POST)],
        ),
        (
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            vec![mention(ACTOR, POST), accepted(OTHER_ACTOR)],
        ),
        (
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            vec![accepted(OTHER_ACTOR), media_tagged(ACTOR, POST)],
        ),
        (
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            vec![like(OTHER_ACTOR, RECIPIENT_POST), mention(ACTOR, POST)],
        ),
        (
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            vec![reply(ACTOR, POST), repost(OTHER_ACTOR, RECIPIENT_POST)],
        ),
    ] {
        let (seen, sources) = filter(world(), level, recipient_id, notifications).await;
        assert_eq!(seen, Err(tonic::Code::InvalidArgument), "level {level}");
        assert!(sources.calls().is_empty());
    }
    let (seen, sources) = filter(world(), 8, RECIPIENT, vec![follow(ACTOR)]).await;
    assert_eq!(seen, Err(tonic::Code::Unimplemented));
    assert!(sources.calls().is_empty());
    let (seen, sources) = filter(world(), FOLLOW, RECIPIENT, vec![]).await;
    assert_eq!(seen, Ok(vec![]));
    assert!(sources.calls().is_empty());
}

#[tokio::test]
async fn a_level_without_notifications_counts_as_an_unknown_level_failure() {
    let requests = |outcome| {
        [
            ("outcome", outcome),
            ("rpc", "filter_notifications"),
            ("safety_level", "unknown"),
        ]
        .to_vec()
    };
    for (family, expected) in [
        (
            &REQUESTS.family,
            vec![requests("started"), requests("failure")],
        ),
        (
            &REQUESTS_BY_CALLER.family,
            vec![vec![
                ("caller_identity", "unknown"),
                ("rpc", "filter_notifications"),
                ("outcome", "failure"),
            ]],
        ),
    ] {
        let (metrics, recording) = Metrics::recording();
        let (seen, _) = filter_with(metrics, world(), 8, RECIPIENT, vec![follow(ACTOR)]).await;
        assert_eq!(seen, Err(tonic::Code::Unimplemented));
        let recorded: Vec<Vec<(String, String)>> = recording
            .take(family)
            .into_iter()
            .map(|sample| sample.labels)
            .collect();
        let expected: Vec<Vec<(String, String)>> = expected
            .iter()
            .map(|labels| {
                labels
                    .iter()
                    .map(|&(label, value)| (label.to_owned(), value.to_owned()))
                    .collect()
            })
            .collect();
        assert_eq!(recorded, expected, "{}", family.name);
    }
}

#[tokio::test]
async fn each_notification_counts_in_the_verdict_families_and_unresolves_nothing() {
    let row = |value: &str, count: f64| {
        (
            value.to_owned(),
            "notifications_follow".to_owned(),
            "filter_notifications".to_owned(),
            count,
        )
    };
    for (family, label, judged, unevaluated) in [
        (
            &VERDICTS.family,
            "action",
            vec![row("allow", 2.0), row("drop", 2.0)],
            vec![row("drop", 2.0)],
        ),
        (
            &VERDICTS_BY_RULE.family,
            "rule",
            vec![row("not_found", 1.0), row("suspended_author/drop", 1.0)],
            vec![row("lookup_failed", 2.0)],
        ),
        (
            &EVALUATED_CANDIDATES.family,
            "fail_open_defaults",
            vec![row("none", 4.0)],
            vec![],
        ),
        (&UNRESOLVED_TWEETS.family, "lookup", vec![], vec![]),
    ] {
        for (sources, actors, expected) in [
            (
                world(),
                &[ACTOR, SUSPENDED_ACTOR, ACTOR, MISSING_ACTOR][..],
                judged,
            ),
            (
                world().fault(Source::GizmoduckViewer, Fault::Fails),
                &[ACTOR, ACTOR],
                unevaluated,
            ),
        ] {
            let (metrics, recording) = Metrics::recording();
            let notifications = actors.iter().copied().map(follow).collect();
            let (seen, _) = filter_with(metrics, sources, FOLLOW, RECIPIENT, notifications).await;
            assert!(seen.is_ok());
            let mut recorded: Vec<(String, String, String, f64)> = recording
                .take(family)
                .into_iter()
                .map(|sample| {
                    (
                        sample.label(label).unwrap().to_owned(),
                        sample.label("safety_level").unwrap().to_owned(),
                        sample.label("rpc").unwrap().to_owned(),
                        sample.value,
                    )
                })
                .collect();
            recorded.sort_by(|x, y| x.0.cmp(&y.0));
            assert_eq!(recorded, expected, "{} {actors:?}", family.name);
        }
    }
}

#[tokio::test]
async fn each_notification_answers_in_order_and_a_repeated_actor_costs_no_call() {
    let sources = world()
        .edge(Graph::Blocks, RECIPIENT, OTHER_ACTOR)
        .user(5, actor_with_image(5))
        .edge(Graph::Mutes, RECIPIENT, 5)
        .user(6, labeled(6, LabelValue::READ_ONLY));
    let (seen, sources) = filter(
        sources,
        FOLLOW,
        RECIPIENT,
        [
            ACTOR,
            SUSPENDED_ACTOR,
            OTHER_ACTOR,
            ACTOR,
            MISSING_ACTOR,
            5,
            6,
        ]
        .map(follow)
        .into(),
    )
    .await;
    assert_eq!(
        seen,
        Ok(vec![
            Seen::Allowed,
            Seen::Dropped(DropReason::ActorSuspended),
            Seen::Dropped(DropReason::RecipientBlocksActor),
            Seen::Allowed,
            Seen::Dropped(DropReason::ActorNotFound),
            Seen::Dropped(DropReason::RecipientMutesActor),
            Seen::Dropped(DropReason::Unspecified),
        ])
    );
    assert_eq!(
        sources.keys(Source::GizmoduckActor),
        [vec![
            ACTOR,
            OTHER_ACTOR,
            SUSPENDED_ACTOR,
            MISSING_ACTOR,
            5,
            6
        ]]
    );
    assert_eq!(sources.selects().len(), 1);
    assert_eq!(sources.keys(Source::GizmoduckViewer), [vec![RECIPIENT]]);
    for unasked in [
        Source::TesPureCoreData,
        Source::GizmoduckAuthor,
        Source::ActorEmails,
        Source::ActorPhones,
    ] {
        assert!(!sources.calls().contains(&unasked), "{unasked:?}");
    }
}

#[tokio::test]
async fn only_an_accepted_follow_request_drops_a_protected_actor() {
    const PROTECTED_ACTOR: u64 = 5;
    let mut protected = actor_with_image(PROTECTED_ACTOR);
    if let Some(user) = protected.user.as_mut() {
        user.safety.is_protected = true;
    }
    let world = || world().user(PROTECTED_ACTOR, protected.clone());
    let (accepted_results, sources) = filter(
        world(),
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        [PROTECTED_ACTOR, ACTOR, SUSPENDED_ACTOR]
            .map(accepted)
            .into(),
    )
    .await;
    assert_eq!(
        accepted_results,
        Ok(vec![
            Seen::Dropped(DropReason::ActorProtected),
            Seen::Allowed,
            Seen::Dropped(DropReason::ActorSuspended),
        ])
    );
    assert_eq!(sources.calls().len(), 3);
    assert_eq!(follows(world(), &[PROTECTED_ACTOR]).await, [Seen::Allowed]);
    let (followed, _) = filter(
        world().edge(Graph::Follows, RECIPIENT, PROTECTED_ACTOR),
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![accepted(PROTECTED_ACTOR)],
    )
    .await;
    assert_eq!(followed, Ok(vec![Seen::Allowed]));
}

#[tokio::test]
async fn a_like_and_a_repost_get_an_accepted_follow_request_s_verdict_on_each_actor() {
    const PROTECTED_ACTOR: u64 = 5;
    const MUTED_ACTOR: u64 = 6;
    const READ_ONLY_ACTOR: u64 = 7;
    const FAILED_ACTOR: u64 = 8;
    let mut protected = actor_with_image(PROTECTED_ACTOR);
    if let Some(user) = protected.user.as_mut() {
        user.safety.is_protected = true;
    }
    let world = || {
        world()
            .edge(Graph::Blocks, RECIPIENT, OTHER_ACTOR)
            .user(PROTECTED_ACTOR, protected.clone())
            .user(MUTED_ACTOR, actor_with_image(MUTED_ACTOR))
            .edge(Graph::Mutes, RECIPIENT, MUTED_ACTOR)
            .user(
                READ_ONLY_ACTOR,
                labeled(READ_ONLY_ACTOR, LabelValue::READ_ONLY),
            )
            .user(FAILED_ACTOR, actor_with_image(FAILED_ACTOR))
            .fail_key(Source::GizmoduckActor, FAILED_ACTOR)
    };
    let actors = [
        ACTOR,
        SUSPENDED_ACTOR,
        OTHER_ACTOR,
        MISSING_ACTOR,
        PROTECTED_ACTOR,
        MUTED_ACTOR,
        READ_ONLY_ACTOR,
        FAILED_ACTOR,
    ];
    for (name, kind) in [
        ("accepted", accepted as fn(u64) -> vf_pb::Notification),
        ("like", |actor| like(actor, RECIPIENT_POST)),
        ("repost", |actor| repost(actor, RECIPIENT_POST)),
    ] {
        let (seen, _) = filter(
            world(),
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            actors.map(kind).into(),
        )
        .await;
        assert_eq!(
            seen,
            Ok(vec![
                Seen::Allowed,
                Seen::Dropped(DropReason::ActorSuspended),
                Seen::Dropped(DropReason::RecipientBlocksActor),
                Seen::Dropped(DropReason::ActorNotFound),
                Seen::Dropped(DropReason::ActorProtected),
                Seen::Dropped(DropReason::RecipientMutesActor),
                Seen::Dropped(DropReason::Unspecified),
                Seen::Failed,
            ]),
            "{name}"
        );
    }
}

#[tokio::test]
async fn a_like_or_a_repost_drops_when_its_post_is_missing_or_not_the_recipient_s() {
    const FAILED_POST: u64 = 105;
    let sources = world()
        .tweet(POST, ACTOR)
        .tweet(FAILED_POST, RECIPIENT)
        .fail_key(Source::TesPureCoreData, FAILED_POST);
    let (seen, sources) = filter(
        sources,
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![
            like(ACTOR, RECIPIENT_POST),
            repost(ACTOR, MISSING_POST),
            like(OTHER_ACTOR, POST),
            repost(SUSPENDED_ACTOR, MISSING_POST),
            like(ACTOR, FAILED_POST),
            like(SUSPENDED_ACTOR, FAILED_POST),
            accepted(ACTOR),
            repost(OTHER_ACTOR, RECIPIENT_POST),
        ],
    )
    .await;
    assert_eq!(
        seen,
        Ok(vec![
            Seen::Allowed,
            Seen::Dropped(DropReason::PostNotFound),
            Seen::Dropped(DropReason::PostNotFound),
            Seen::Dropped(DropReason::PostNotFound),
            Seen::Failed,
            Seen::Failed,
            Seen::Allowed,
            Seen::Allowed,
        ])
    );
    assert_eq!(
        sources.keys(Source::TesPureCoreData),
        [vec![POST, RECIPIENT_POST, FAILED_POST, MISSING_POST]]
    );
    for unasked in [
        Source::TesTweetForVisibility,
        Source::SafetyLabels,
        Source::GizmoduckAuthor,
    ] {
        assert!(!sources.calls().contains(&unasked), "{unasked:?}");
    }
}

#[tokio::test]
async fn each_user_notification_counts_under_its_kind() {
    let (metrics, recording) = Metrics::recording();
    let (seen, _) = filter_with(
        metrics,
        world(),
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![
            like(ACTOR, RECIPIENT_POST),
            repost(SUSPENDED_ACTOR, RECIPIENT_POST),
            accepted(ACTOR),
            like(MISSING_ACTOR, RECIPIENT_POST),
        ],
    )
    .await;
    assert!(seen.is_ok());
    assert_eq!(
        notification_results(&recording),
        [
            result_row("follow_request_accepted", "allowed", "none"),
            result_row("like", "allowed", "none"),
            result_row("like", "dropped", "actor_not_found"),
            result_row("repost", "dropped", "actor_suspended"),
        ]
    );
}

fn notification_results(recording: &Recording) -> Vec<(String, String, String, f64)> {
    let mut recorded: Vec<_> = recording
        .take(&NOTIFICATION_RESULTS.family)
        .into_iter()
        .map(|sample| {
            let label = |name| sample.label(name).unwrap().to_owned();
            (
                label("notification_type"),
                label("outcome"),
                label("reason"),
                sample.value,
            )
        })
        .collect();
    recorded.sort_by(|x, y| x.partial_cmp(y).unwrap());
    recorded
}

fn result_row(kind: &str, outcome: &str, reason: &str) -> (String, String, String, f64) {
    (kind.to_owned(), outcome.to_owned(), reason.to_owned(), 1.0)
}

#[tokio::test]
async fn not_followed_by_drops_only_an_actor_neither_following_nor_followed() {
    let filters = GizmoduckAdvancedFilters {
        filter_not_followed_by: true,
        ..Default::default()
    };
    let sources = filtering(filters)
        .user(5, actor_with_image(5))
        .edge(Graph::Follows, OTHER_ACTOR, RECIPIENT)
        .edge(Graph::Follows, RECIPIENT, 5);
    assert_eq!(
        follows(sources, &[ACTOR, OTHER_ACTOR, 5]).await,
        [
            Seen::Dropped(DropReason::Unspecified),
            Seen::Allowed,
            Seen::Allowed
        ]
    );
}

#[tokio::test]
async fn email_and_phone_lookups_run_only_for_a_recipient_filtering_on_them() {
    for (filters, source) in [
        (
            GizmoduckAdvancedFilters {
                filter_no_confirmed_email: true,
                ..Default::default()
            },
            Source::ActorEmails,
        ),
        (
            GizmoduckAdvancedFilters {
                filter_no_confirmed_phone: true,
                ..Default::default()
            },
            Source::ActorPhones,
        ),
    ] {
        let sources = filtering(filters)
            .confirmed_email(ACTOR)
            .verified_phone(ACTOR)
            .edge(Graph::Follows, RECIPIENT, 5)
            .user(5, actor_with_image(5));
        let (seen, sources) = filter(
            sources,
            FOLLOW,
            RECIPIENT,
            [ACTOR, OTHER_ACTOR, OTHER_ACTOR, 5].map(follow).into(),
        )
        .await;
        assert_eq!(
            seen,
            Ok(vec![
                Seen::Allowed,
                Seen::Dropped(DropReason::Unspecified),
                Seen::Dropped(DropReason::Unspecified),
                Seen::Allowed,
            ]),
            "{source:?}"
        );
        assert_eq!(sources.keys(source), [vec![ACTOR, OTHER_ACTOR, 5]]);
        assert_eq!(sources.calls().len(), 4, "{source:?}");
    }
    let (_, unfiltered) = filter(world(), FOLLOW, RECIPIENT, vec![follow(ACTOR)]).await;
    assert_eq!(unfiltered.calls().len(), 3);
}

#[tokio::test]
async fn profile_age_and_mention_filters_drop_a_non_follower() {
    let fresh = || {
        let mut user = actor_with_image(5);
        if let Some(user) = user.user.as_mut() {
            user.created_at_msec = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            )
            .ok();
        }
        user
    };
    for (filters, user, mention_filter) in [
        (
            GizmoduckAdvancedFilters {
                filter_new_users: true,
                ..Default::default()
            },
            (5, fresh()),
            None,
        ),
        (
            GizmoduckAdvancedFilters {
                filter_default_profile_image: true,
                ..Default::default()
            },
            (6, found_user(6)),
            None,
        ),
        (
            GizmoduckAdvancedFilters::default(),
            (ACTOR, actor_with_image(ACTOR)),
            Some(MentionFilter::Following),
        ),
    ] {
        let mut viewer = recipient(filters);
        viewer.account.mention_filter = mention_filter;
        let (id, user) = user;
        let sources = world().viewer(RECIPIENT, viewer).user(id, user);
        assert_eq!(
            follows(sources, &[id]).await,
            [Seen::Dropped(DropReason::Unspecified)],
            "{id}"
        );
    }
}

#[tokio::test]
async fn recipient_state_and_labels_decode_as_scala_reads_them() {
    let mut suspended_recipient = recipient(GizmoduckAdvancedFilters::default());
    suspended_recipient.safety.suspended = true;
    let mut blink_recipient = recipient(GizmoduckAdvancedFilters::default());
    blink_recipient.labels = labeled(RECIPIENT, LabelValue::BLINK_BAD)
        .user
        .map(|user| user.labels)
        .unwrap_or_default();
    let blink_worst = || world().user(5, labeled(5, LabelValue::BLINK_WORST));
    for (name, sources, actor, expected) in [
        (
            "suspended recipient",
            world().viewer(RECIPIENT, suspended_recipient),
            ACTOR,
            Seen::Dropped(DropReason::Unspecified),
        ),
        (
            "blink worst actor, recipient with the omitted UQF setting",
            blink_worst(),
            5,
            Seen::Dropped(DropReason::Unspecified),
        ),
        (
            "blink worst actor, blink labeled recipient",
            blink_worst().viewer(RECIPIENT, blink_recipient),
            5,
            Seen::Allowed,
        ),
    ] {
        assert_eq!(follows(sources, &[actor]).await, [expected], "{name}");
    }
}

#[tokio::test]
async fn failed_lookups_fail_only_the_notifications_whose_verdict_read_them() {
    let email_filter = GizmoduckAdvancedFilters {
        filter_no_confirmed_email: true,
        ..Default::default()
    };
    for (name, sources, expected) in [
        (
            "actor",
            world().fail_key(Source::GizmoduckActor, ACTOR),
            [Seen::Failed, Seen::Allowed],
        ),
        (
            "partial actor",
            world().fault(Source::GizmoduckActor, Fault::AnswersPartial),
            [Seen::Failed, Seen::Failed],
        ),
        (
            "recipient",
            world().fault(Source::GizmoduckViewer, Fault::Fails),
            [Seen::Failed, Seen::Failed],
        ),
        (
            "email",
            filtering(email_filter.clone())
                .confirmed_email(OTHER_ACTOR)
                .fail_key(Source::ActorEmails, ACTOR),
            [Seen::Failed, Seen::Allowed],
        ),
        (
            "unread email",
            world().fault(Source::ActorEmails, Fault::Fails),
            [Seen::Allowed, Seen::Allowed],
        ),
        (
            "flock",
            world().fail_graph(Graph::Blocks),
            [Seen::Failed, Seen::Failed],
        ),
    ] {
        assert_eq!(
            follows(sources, &[ACTOR, OTHER_ACTOR]).await,
            expected,
            "{name}"
        );
    }
}

#[tokio::test]
async fn a_recipient_gizmoduck_does_not_find_evaluates_as_a_default_viewer() {
    let sources = InMemorySources::default()
        .user(ACTOR, actor_with_image(ACTOR))
        .user(SUSPENDED_ACTOR, suspended());
    assert_eq!(
        follows(sources, &[ACTOR, SUSPENDED_ACTOR]).await,
        [Seen::Allowed, Seen::Dropped(DropReason::ActorSuspended)]
    );
}

#[tokio::test]
async fn each_post_notification_judges_its_post_author_in_order_on_one_post_lookup() {
    const ACTORLESS_POST: u64 = 103;
    let sources = posts_world().tweet(ACTORLESS_POST, MISSING_ACTOR).edge(
        Graph::Blocks,
        RECIPIENT,
        OTHER_ACTOR,
    );
    let (seen, sources) = filter(
        sources,
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![
            mention(ACTOR, POST),
            mention(OTHER_ACTOR, POST),
            reply(SUSPENDED_ACTOR, SUSPENDED_POST),
            media_tagged(OTHER_ACTOR, OTHER_POST),
            mention(ACTOR, MISSING_POST),
            reply(OTHER_ACTOR, POST),
            media_tagged(ACTOR, POST),
            mention(MISSING_ACTOR, ACTORLESS_POST),
            mention(ACTOR, POST),
        ],
    )
    .await;
    assert_eq!(
        seen,
        Ok(vec![
            Seen::Allowed,
            Seen::Dropped(DropReason::PostNotFound),
            Seen::Dropped(DropReason::ActorSuspended),
            Seen::Dropped(DropReason::RecipientBlocksActor),
            Seen::Dropped(DropReason::PostNotFound),
            Seen::Dropped(DropReason::PostNotFound),
            Seen::Allowed,
            Seen::Dropped(DropReason::ActorNotFound),
            Seen::Allowed,
        ])
    );
    assert_eq!(
        sources.keys(Source::TesPureCoreData),
        [vec![
            POST,
            OTHER_POST,
            SUSPENDED_POST,
            ACTORLESS_POST,
            MISSING_POST
        ]]
    );
    assert_eq!(
        sources.keys(Source::GizmoduckActor),
        [vec![ACTOR, OTHER_ACTOR, SUSPENDED_ACTOR, MISSING_ACTOR]]
    );
    assert_eq!(sources.keys(Source::GizmoduckViewer), [vec![RECIPIENT]]);
    assert!(!sources.calls().contains(&Source::GizmoduckAuthor));
}

#[tokio::test]
async fn a_post_author_s_email_lookup_runs_only_for_a_recipient_filtering_on_it() {
    let filters = GizmoduckAdvancedFilters {
        filter_no_confirmed_email: true,
        ..Default::default()
    };
    for landing_last in [Source::GizmoduckViewer, Source::TesPureCoreData] {
        let sources = posts_world()
            .viewer(RECIPIENT, recipient(filters.clone()))
            .confirmed_email(ACTOR)
            .latency(landing_last, std::time::Duration::from_millis(20));
        let (seen, sources) = filter(
            sources,
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            vec![mention(ACTOR, POST), reply(OTHER_ACTOR, OTHER_POST)],
        )
        .await;
        assert_eq!(
            seen,
            Ok(vec![Seen::Allowed, Seen::Dropped(DropReason::Unspecified)]),
            "{landing_last:?}"
        );
        assert_eq!(
            sources.keys(Source::ActorEmails),
            [vec![ACTOR, OTHER_ACTOR]],
            "{landing_last:?}"
        );
    }
    let (_, unfiltered) = filter(
        posts_world(),
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![mention(ACTOR, POST)],
    )
    .await;
    assert!(!unfiltered.calls().contains(&Source::ActorEmails));
}

#[tokio::test]
async fn failed_lookups_fail_only_the_post_notifications_whose_verdict_read_them() {
    let email_filter = GizmoduckAdvancedFilters {
        filter_no_confirmed_email: true,
        ..Default::default()
    };
    for (name, sources, expected) in [
        (
            "post",
            posts_world().fail_key(Source::TesPureCoreData, POST),
            [Seen::Failed, Seen::Allowed],
        ),
        (
            "actor",
            posts_world().fail_key(Source::GizmoduckActor, ACTOR),
            [Seen::Failed, Seen::Allowed],
        ),
        (
            "recipient",
            posts_world().fault(Source::GizmoduckViewer, Fault::Fails),
            [Seen::Failed, Seen::Failed],
        ),
        (
            "email",
            posts_world()
                .viewer(RECIPIENT, recipient(email_filter.clone()))
                .confirmed_email(OTHER_ACTOR)
                .fail_key(Source::ActorEmails, ACTOR),
            [Seen::Failed, Seen::Allowed],
        ),
        (
            "flock",
            posts_world().fail_graph(Graph::Blocks),
            [Seen::Failed, Seen::Failed],
        ),
    ] {
        let (seen, _) = filter(
            sources,
            FOLLOW_REQUEST_ACCEPTED,
            RECIPIENT,
            vec![mention(ACTOR, POST), reply(OTHER_ACTOR, OTHER_POST)],
        )
        .await;
        assert_eq!(seen, Ok(expected.into()), "{name}");
    }
}

#[tokio::test]
async fn each_post_notification_counts_under_its_kind() {
    let (metrics, recording) = Metrics::recording();
    let (seen, _) = filter_with(
        metrics,
        posts_world(),
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![
            mention(ACTOR, POST),
            reply(OTHER_ACTOR, POST),
            media_tagged(ACTOR, MISSING_POST),
            mention(SUSPENDED_ACTOR, SUSPENDED_POST),
        ],
    )
    .await;
    assert!(seen.is_ok());
    assert_eq!(
        notification_results(&recording),
        [
            result_row("media_tagged", "dropped", "post_not_found"),
            result_row("mention", "allowed", "none"),
            result_row("mention", "dropped", "actor_suspended"),
            result_row("reply", "dropped", "post_not_found"),
        ]
    );
    let mut by_rule: Vec<(String, String, f64)> = recording
        .take(&VERDICTS_BY_RULE.family)
        .into_iter()
        .map(|sample| {
            let label = |name| sample.label(name).unwrap().to_owned();
            (label("rule"), label("safety_level"), sample.value)
        })
        .collect();
    by_rule.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let level = "notifications_writer_v2".to_owned();
    assert_eq!(
        by_rule,
        [
            ("not_found".to_owned(), level.clone(), 2.0),
            ("suspended_author/drop".to_owned(), level, 1.0),
        ]
    );
}

#[tokio::test]
async fn a_country_takedown_spares_a_post_notification() {
    use crate::models::TweetFeatures;
    use xai_core_entities::entities::TakedownReason;
    let taken_down = |reason| TweetFeatures {
        takedown_reasons: vec![reason],
        ..TweetFeatures::default()
    };
    let sources = posts_world()
        .country(RECIPIENT, "de")
        .tweet_features(
            POST,
            taken_down(TakedownReason::BystanderReport {
                country_code: "de".to_owned(),
            }),
        )
        .tweet_features(
            OTHER_POST,
            taken_down(TakedownReason::LegalRequest {
                country_code: "xx".to_owned(),
            }),
        );
    let (seen, _) = filter(
        sources,
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![mention(ACTOR, POST), mention(OTHER_ACTOR, OTHER_POST)],
    )
    .await;
    assert_eq!(
        seen,
        Ok(vec![Seen::Allowed, Seen::Dropped(DropReason::Unspecified)])
    );
}

#[tokio::test]
async fn a_failed_label_read_fails_a_mention_and_spares_a_media_tag_of_the_same_post() {
    let sources = posts_world().fail_key(Source::SafetyLabels, POST);
    let (seen, _) = filter(
        sources,
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![mention(ACTOR, POST), media_tagged(ACTOR, POST)],
    )
    .await;
    assert_eq!(seen, Ok(vec![Seen::Failed, Seen::Allowed]));
}

#[tokio::test]
async fn only_people_i_follow_exempts_a_post_notification_in_a_community() {
    const COMMUNITY_POST: u64 = 103;
    let mut viewer = recipient(GizmoduckAdvancedFilters::default());
    viewer.account.mention_filter = Some(MentionFilter::Following);
    let sources = posts_world()
        .viewer(RECIPIENT, viewer)
        .tweet(COMMUNITY_POST, ACTOR)
        .community_post(COMMUNITY_POST, 7);
    let (seen, _) = filter(
        sources,
        FOLLOW_REQUEST_ACCEPTED,
        RECIPIENT,
        vec![
            mention(ACTOR, POST),
            mention(ACTOR, COMMUNITY_POST),
            media_tagged(ACTOR, POST),
            media_tagged(ACTOR, COMMUNITY_POST),
        ],
    )
    .await;
    assert_eq!(
        seen,
        Ok(vec![
            Seen::Dropped(DropReason::Unspecified),
            Seen::Allowed,
            Seen::Dropped(DropReason::Unspecified),
            Seen::Allowed,
        ])
    );
}
