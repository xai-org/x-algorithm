use std::collections::BTreeMap;

use xai_abuse_proto::enforcement::ScoreResult;

use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{ActionControlRecord, ControlEntity, ControlMode, PageAction};
use crate::replay_tests::{
    message, on_config, switched_off, Account, BedConfig, TestBed, Trace, BAKED_POST_RULES,
    BAKED_USER_RULES, HEADS,
};

const BB1: (&str, &str) = HEADS[0];
const USER_SUSPEND_LABEL: &str = "anchor_campaign_suspend";
const POST_SUSPEND_AUTHOR_LABEL: &str = "cse_reports_embedding_v2";

const PARENT: i64 = 9_300_000_001;
const POSTER: i64 = 1_800_000_001;
const OTHER: i64 = 1_800_000_002;

async fn bed(records: Vec<ActionControlRecord>) -> TestBed {
    TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config: on_config(false, None),
        off_switches: vec![switched_off()],
        control_records: records,
    })
    .await
}

fn page_records() -> Vec<ActionControlRecord> {
    vec![ActionControlRecord {
        entity_type: ControlEntity::Post,
        ..record(
            BB1.0,
            None,
            None,
            ControlMode::Live,
            vec![PageAction::SUSPEND_AUTHOR],
        )
    }]
}

fn reply(account: Account) -> ScoreResult {
    ScoreResult {
        in_reply_to_post_id: PARENT,
        in_reply_to_user_id: POSTER,
        ..message(
            EntityType::Post,
            BB1.0,
            BB1.1,
            &[POST_SUSPEND_AUTHOR_LABEL],
            account,
        )
    }
}

fn parent(prefix: &[i64], typed: &[i64], reply_to: Option<i64>) -> serde_json::Value {
    let mut mentions = Vec::new();
    let mut at = 0;
    for u in prefix {
        mentions
            .push(serde_json::json!({"fromIndex": at, "toIndex": at + 10, "user": u.to_string()}));
        at += 11;
    }
    let visible = at;
    for u in typed {
        at += 4;
        mentions
            .push(serde_json::json!({"fromIndex": at, "toIndex": at + 10, "user": u.to_string()}));
        at += 11;
    }
    let mut core = serde_json::json!({"user": POSTER.to_string(), "text": "thanks"});
    if let Some(r) = reply_to {
        core["reply"] = serde_json::json!({"inReplyToStatus": "1", "inReplyToUser": r.to_string()});
    }
    serde_json::json!({
        "id": PARENT.to_string(),
        "coreData": core,
        "mentions": mentions,
        "visibleTextRange": {"fromIndex": visible, "toIndex": at + 30},
    })
}

fn info(t: &Trace) -> &BTreeMap<String, String> {
    &t.outcomes.last().expect("one outcome").info
}

fn assert_skipped(t: &Trace, reason: &str, rule: &str, answer: &str, what: &str) {
    assert_eq!(t.status, reason, "{what}");
    assert!(t.ais.is_empty(), "{what}: {:?}", t.ais);
    assert_eq!(t.tickets, 0, "{what}");
    assert_eq!(info(t)["matched_rule_id"], rule, "{what}");
    assert_eq!(info(t)["invited_reply"], answer, "{what}");
}

fn assert_acted(t: &Trace, what: &str) {
    assert_eq!(t.status, "success", "{what}");
    assert!(!t.ais.is_empty(), "{what}");
    assert!(!info(t).contains_key("invited_reply"), "{what}");
}

fn invited_parents() -> Vec<(&'static str, serde_json::Value, &'static str)> {
    let r = Account::Clean.user_id();
    vec![
        (
            "typed in a top-level post",
            parent(&[], &[r], None),
            "mentioned",
        ),
        (
            "typed after a prefix naming someone else",
            parent(&[OTHER], &[OTHER, r], Some(OTHER)),
            "mentioned",
        ),
        (
            "named in the prefix and typed again",
            parent(&[r], &[r], Some(r)),
            "mentioned",
        ),
    ]
}

fn not_invited_parents() -> Vec<(&'static str, serde_json::Value)> {
    let r = Account::Clean.user_id();
    vec![
        (
            "named only in the automatic prefix",
            parent(&[OTHER, r], &[], Some(OTHER)),
        ),
        (
            "named only in the prefix, others typed",
            parent(&[r, OTHER], &[OTHER], Some(OTHER)),
        ),
        ("mentions someone else", parent(&[], &[OTHER], None)),
        ("no mentions", parent(&[], &[], None)),
        (
            "a reply to someone else",
            parent(&[POSTER], &[], Some(POSTER)),
        ),
        (
            "the parent only replies to the replier",
            parent(&[r], &[], Some(r)),
        ),
        (
            "the parent replies to the replier and mentions someone else",
            parent(&[], &[OTHER], Some(r)),
        ),
    ]
}

#[tokio::test]
async fn both_paths_skip_every_invited_reply() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for records in [Vec::new(), page_records()] {
        for (what, record, answer) in invited_parents() {
            let b = bed(records.clone()).await;
            b.answer_reply_parent(PARENT, 200, record).await;
            let t = b.run_fresh(&reply(Account::Clean)).await;
            assert_skipped(&t, "invited_reply_skipped", "invited_reply", answer, what);
        }
    }
}

#[tokio::test]
async fn both_paths_still_act_on_a_reply_its_parent_did_not_invite() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for records in [Vec::new(), page_records()] {
        for (what, record) in not_invited_parents() {
            let b = bed(records.clone()).await;
            b.answer_reply_parent(PARENT, 200, record).await;
            assert_acted(&b.run_fresh(&reply(Account::Clean)).await, what);
        }
    }
}

#[tokio::test]
async fn an_unreadable_parent_skips_the_reply() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for records in [Vec::new(), page_records()] {
        for (what, status, record) in [
            ("missing", 200, serde_json::Value::Null),
            ("no core data", 200, serde_json::json!({"mentions": []})),
            ("column error", 500, serde_json::Value::Null),
        ] {
            let b = bed(records.clone()).await;
            b.answer_reply_parent(PARENT, status, record).await;
            let t = b.run_fresh(&reply(Account::Clean)).await;
            assert_skipped(
                &t,
                "invited_reply_unknown_skipped",
                "invited_reply_unknown",
                "unknown",
                what,
            );
        }
        let b = bed(records.clone()).await;
        let t = b.run_fresh(&reply(Account::Clean)).await;
        assert_eq!(
            t.status, "invited_reply_unknown_skipped",
            "no answer at all"
        );
    }
}

#[tokio::test]
async fn only_post_decisions_on_replies_read_the_parent() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(Vec::new()).await;
    let post = message(
        EntityType::Post,
        BB1.0,
        BB1.1,
        &[POST_SUSPEND_AUTHOR_LABEL],
        Account::Clean,
    );
    assert_acted(&b.run_fresh(&post).await, "not a reply");
    let user = ScoreResult {
        in_reply_to_post_id: PARENT,
        ..message(
            EntityType::User,
            BB1.0,
            BB1.1,
            &[USER_SUSPEND_LABEL],
            Account::Clean,
        )
    };
    assert_acted(&b.run_fresh(&user).await, "an account decision");
    assert_eq!(b.reply_parent_reads().await, 0);
}

#[tokio::test]
async fn a_parent_is_read_once_for_many_replies() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(page_records()).await;
    b.answer_reply_parent(PARENT, 200, parent(&[], &[OTHER], None))
        .await;
    for _ in 0..3 {
        assert_acted(&b.run_fresh(&reply(Account::Clean)).await, "not invited");
    }
    assert_eq!(b.reply_parent_reads().await, 1);
}

#[tokio::test]
async fn earlier_checks_keep_their_reason_on_an_invited_reply() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for records in [Vec::new(), page_records()] {
        let b = bed(records).await;
        let typed = parent(
            &[],
            &[
                Account::AboveTopLimit.user_id(),
                Account::BetweenLimits.user_id(),
            ],
            None,
        );
        b.answer_reply_parent(PARENT, 200, typed).await;
        assert_eq!(
            b.run_fresh(&reply(Account::AboveTopLimit)).await.status,
            "very_high_follower_count"
        );
        assert_eq!(
            b.run_fresh(&reply(Account::BetweenLimits)).await.status,
            "high_follower_count"
        );
    }
}
