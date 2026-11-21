use std::collections::BTreeMap;
use std::sync::Arc;

use xai_abuse_proto::enforcement::ScoreResult;

use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{ActionControlRecord, ControlEntity, ControlMode, PageAction};
use crate::gizmoduck::test_support::{AccountShape, FixedUsers};
use crate::gizmoduck::GizmoduckCoreClient;
use crate::replay_tests::{
    message, on_config, switched_off, Account, BedConfig, TestBed, Trace, BAKED_POST_RULES,
    BAKED_USER_RULES, HEADS,
};
use crate::safety_checks::TestAccountPolicy;

const BB1: (&str, &str) = HEADS[0];
const USER_SUSPEND_LABEL: &str = "anchor_campaign_suspend";
const POST_SUSPEND_AUTHOR_LABEL: &str = "cse_reports_embedding_v2";

const CONFIRMED: i32 = 3;
const UNCONFIRMED: i32 = 2;
const EXCUSED: i32 = 1;
const GRANDFATHERED: i32 = 0;

fn policy() -> TestAccountPolicy {
    TestAccountPolicy::new(
        &["test.example.com".into()],
        &["qa_test_role".into(), "test_role".into()],
        &["203.0.113.0/24".into()],
    )
    .unwrap()
}

async fn bed(
    records: Vec<ActionControlRecord>,
    shape: &AccountShape,
    accounts: &[Account],
) -> TestBed {
    let mut b = TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config: on_config(false, None),
        off_switches: vec![switched_off()],
        control_records: records,
    })
    .await;
    let users = Account::ALL
        .into_iter()
        .chain(Account::EXTRA)
        .filter(|a| !matches!(a, Account::NotFound))
        .map(|a| {
            let s = if accounts.iter().any(|x| x.user_id() == a.user_id()) {
                shape.clone()
            } else {
                AccountShape::default()
            };
            (a.user_id(), s.user(a.user_id()))
        })
        .collect();
    b.ctx.gd = GizmoduckCoreClient::from_lookup(Arc::new(FixedUsers {
        users,
        ..Default::default()
    }))
    .with_policy(policy());
    b
}

fn user_msg(account: Account) -> ScoreResult {
    message(
        EntityType::User,
        BB1.0,
        BB1.1,
        &[USER_SUSPEND_LABEL],
        account,
    )
}

fn post_msg(account: Account) -> ScoreResult {
    message(
        EntityType::Post,
        BB1.0,
        BB1.1,
        &[POST_SUSPEND_AUTHOR_LABEL],
        account,
    )
}

fn info(t: &Trace) -> &BTreeMap<String, String> {
    &t.outcomes.last().expect("one outcome").info
}

fn assert_skipped(t: &Trace, reason: &str, rule: &str, signal: &str, what: &str) {
    assert_eq!(t.status, reason, "{what}");
    assert!(t.ais.is_empty(), "{what}: {:?}", t.ais);
    assert_eq!(t.tickets, 0, "{what}");
    let i = info(t);
    assert_eq!(i["matched_rule_id"], rule, "{what}");
    assert_eq!(i["test_account_signal"], signal, "{what}");
}

fn assert_suspended(t: &Trace, what: &str) {
    assert_eq!(t.status, "success", "{what}");
    assert!(
        t.ais
            .iter()
            .any(|(_, r)| r.pointer("/action/suspendUser").is_some()),
        "{what}: {:?}",
        t.ais
    );
    assert!(!info(t).contains_key("test_account_signal"), "{what}");
}

fn test_account_shapes() -> Vec<(
    &'static str,
    AccountShape,
    &'static str,
    &'static str,
    &'static str,
)> {
    vec![
        (
            "temporary test account",
            AccountShape {
                test_user_status: Some(1),
                ..Default::default()
            },
            "test_user_skipped",
            "gizmoduck_test_user",
            "test_user_status",
        ),
        (
            "permanent test account",
            AccountShape {
                test_user_status: Some(2),
                ..Default::default()
            },
            "test_user_skipped",
            "gizmoduck_test_user",
            "test_user_status",
        ),
        (
            "test role",
            AccountShape {
                roles: vec!["other", "test_role"],
                ..Default::default()
            },
            "test_user_skipped",
            "test_account",
            "test_role",
        ),
        (
            "confirmed e-mail at the domain, any case",
            AccountShape {
                login_email: Some("Someone@TEST.example.com"),
                email_confirmed: Some(CONFIRMED),
                ..Default::default()
            },
            "test_user_skipped",
            "test_account",
            "test_email_domain",
        ),
        (
            "created in the network",
            AccountShape {
                creation_ip: Some("203.0.113.255"),
                ..Default::default()
            },
            "test_user_skipped",
            "test_account",
            "test_creation_cidr",
        ),
        (
            "created in the network, IPv4-mapped",
            AccountShape {
                creation_ip: Some("::ffff:203.0.113.7"),
                ..Default::default()
            },
            "test_user_skipped",
            "test_account",
            "test_creation_cidr",
        ),
    ]
}

fn near_misses() -> Vec<(&'static str, AccountShape)> {
    let email = |e: &'static str, state: Option<i32>| AccountShape {
        login_email: Some(e),
        email_confirmed: state,
        ..Default::default()
    };
    let ip = |ip: &'static str| AccountShape {
        creation_ip: Some(ip),
        ..Default::default()
    };
    vec![
        (
            "unconfirmed e-mail",
            email("a@test.example.com", Some(UNCONFIRMED)),
        ),
        ("excused e-mail", email("a@test.example.com", Some(EXCUSED))),
        (
            "grandfathered e-mail",
            email("a@test.example.com", Some(GRANDFATHERED)),
        ),
        ("no e-mail state", email("a@test.example.com", None)),
        ("redacted e-mail", email("<redacted>", Some(CONFIRMED))),
        (
            "suffix of another domain",
            email("a@eviltest.example.com", Some(CONFIRMED)),
        ),
        (
            "domain as a prefix",
            email("a@test.example.com.evil.test", Some(CONFIRMED)),
        ),
        (
            "domain as the local part",
            email("test.example.com@evil.test", Some(CONFIRMED)),
        ),
        (
            "two @",
            email("a@evil.test@test.example.com", Some(CONFIRMED)),
        ),
        (
            "empty local part",
            email("@test.example.com", Some(CONFIRMED)),
        ),
        (
            "subdomain",
            email("a@sub.test.example.com", Some(CONFIRMED)),
        ),
        (
            "look-alike letters",
            email("a@test.ex\u{430}mple.com", Some(CONFIRMED)),
        ),
        ("just below the network", ip("203.0.112.255")),
        ("just above the network", ip("203.0.114.0")),
        ("unparseable creation IP", ip("203.0.113")),
        (
            "role that only contains a test role name",
            AccountShape {
                roles: vec!["test_role_extra", "qa_test"],
                ..Default::default()
            },
        ),
        (
            "test status none",
            AccountShape {
                test_user_status: Some(0),
                ..Default::default()
            },
        ),
    ]
}

#[tokio::test]
async fn the_rules_path_skips_every_test_account() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for (what, shape, reason, rule, signal) in test_account_shapes() {
        let b = bed(Vec::new(), &shape, &[Account::Clean]).await;
        let t = b.run_fresh(&user_msg(Account::Clean)).await;
        assert_skipped(&t, reason, rule, signal, what);
    }
}

#[tokio::test]
async fn the_rules_path_still_acts_on_every_near_miss() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(Vec::new(), &AccountShape::default(), &[]).await;
    assert_suspended(
        &b.run_fresh(&user_msg(Account::Clean)).await,
        "plain account",
    );
    for (what, shape) in near_misses() {
        let b = bed(Vec::new(), &shape, &[Account::Clean]).await;
        assert_suspended(&b.run_fresh(&user_msg(Account::Clean)).await, what);
    }
}

#[tokio::test]
async fn the_page_path_skips_every_test_account() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let records = vec![record(
        BB1.0,
        None,
        None,
        ControlMode::Live,
        vec![PageAction::SUSPEND],
    )];
    let b = bed(records.clone(), &AccountShape::default(), &[]).await;
    let t = b.run_fresh(&user_msg(Account::Clean)).await;
    assert_suspended(&t, "plain account on the page");
    assert_eq!(info(&t)["matched_rule_id"], "page");
    for (what, shape, reason, rule, signal) in test_account_shapes() {
        let b = bed(records.clone(), &shape, &[Account::Clean]).await;
        let t = b.run_fresh(&user_msg(Account::Clean)).await;
        assert_skipped(&t, reason, rule, signal, what);
        assert_eq!(info(&t)["rules_would_rule_id"], rule, "{what}");
    }
}

#[tokio::test]
async fn a_post_decision_skips_a_test_account_author_on_both_paths() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let page = vec![ActionControlRecord {
        entity_type: ControlEntity::Post,
        ..record(
            BB1.0,
            None,
            None,
            ControlMode::Live,
            vec![PageAction::SUSPEND_AUTHOR],
        )
    }];
    for records in [Vec::new(), page] {
        let b = bed(records.clone(), &AccountShape::default(), &[]).await;
        assert_suspended(
            &b.run_fresh(&post_msg(Account::Clean)).await,
            "plain author",
        );
        for (what, shape, reason, rule, signal) in test_account_shapes() {
            let b = bed(records.clone(), &shape, &[Account::Clean]).await;
            let t = b.run_fresh(&post_msg(Account::Clean)).await;
            assert_skipped(&t, reason, rule, signal, what);
        }
    }
}

#[tokio::test]
async fn a_test_account_above_the_ceiling_reports_the_test_account() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let shape = AccountShape {
        roles: vec!["qa_test_role"],
        ..Default::default()
    };
    let b = bed(Vec::new(), &shape, &[Account::AboveTopLimit]).await;
    let t = b.run_fresh(&user_msg(Account::AboveTopLimit)).await;
    assert_skipped(
        &t,
        "test_user_skipped",
        "test_account",
        "test_role",
        "above the ceiling",
    );
    let b = bed(Vec::new(), &AccountShape::default(), &[]).await;
    let t = b.run_fresh(&user_msg(Account::AboveTopLimit)).await;
    assert_eq!(t.status, "very_high_follower_count");
}

#[tokio::test]
async fn a_test_account_skip_records_no_account_details() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let shape = AccountShape {
        roles: vec!["qa_test_role"],
        login_email: Some("someone@test.example.com"),
        email_confirmed: Some(CONFIRMED),
        creation_ip: Some("203.0.113.9"),
        ..Default::default()
    };
    let b = bed(Vec::new(), &shape, &[Account::Clean]).await;
    let t = b.run_fresh(&user_msg(Account::Clean)).await;
    let recorded = format!("{:?} {:?} {:?}", t.outcomes, t.dedup_calls, t.ais);
    for secret in ["someone", "test.example.com", "203.0.113", "qa_test_role"] {
        assert!(!recorded.contains(secret), "{secret} in {recorded}");
    }
}
