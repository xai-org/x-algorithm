use std::collections::BTreeMap;

use xai_abuse_proto::enforcement::{self as proto, ScoreResult};

use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{
    ActionControlRecord, ConfidenceEvidence, ControlEntity, ControlMode, EvidenceOverride,
    LevelRow, Lifetime, PageAction,
};
use crate::confidence::ConfidenceLevel;
use crate::replay_tests::{
    base_config, message, on_config, rule_labels, switched_off, Account, BedConfig, TestBed, Trace,
    ACTION_CONTROL_INFO_KEYS, BAKED_POST_RULES, BAKED_USER_RULES, HEADS, SWITCHED_OFF_HEAD,
};

const SUSPEND: PageAction = PageAction::SUSPEND;
const LIVENESS: PageAction = PageAction::LivenessCheck {};
const SHR7: PageAction = PageAction::SpamHighRecall {
    lifetime: Lifetime::Days7,
};
const DAY_MS: i64 = 86_400_000;

const BB1: (&str, &str) = HEADS[0];
const LABEL: &str = "anchor_campaign_suspend";

fn evidence(level: ConfidenceLevel) -> ConfidenceEvidence {
    let (lb, reads, humans) = match level {
        ConfidenceLevel::High => (0.951, 59, 0),
        ConfidenceLevel::Mid => (0.88, 50, 2),
        ConfidenceLevel::Low => (0.72, 12, 0),
    };
    ConfidenceEvidence {
        precision_lower_bound: lb,
        reads,
        humans,
        read_at_ms: now_millis() - DAY_MS,
        link: "documents/read.md".into(),
    }
}

fn proto_evidence(e: &ConfidenceEvidence) -> proto::ConfidenceEvidence {
    proto::ConfidenceEvidence {
        precision_lower_bound: e.precision_lower_bound,
        reads: e.reads,
        humans: e.humans,
        read_at_ms: e.read_at_ms,
        link: e.link.clone(),
    }
}

fn with_level(mut s: ScoreResult, level: i32, ev: Option<ConfidenceEvidence>) -> ScoreResult {
    s.confidence = level;
    s.confidence_evidence = ev.as_ref().map(proto_evidence);
    s
}

fn bb1(account: Account) -> ScoreResult {
    message(EntityType::User, BB1.0, BB1.1, &[LABEL], account)
}

fn high(account: Account) -> ScoreResult {
    with_level(
        bb1(account),
        proto::Confidence::High as i32,
        Some(evidence(ConfidenceLevel::High)),
    )
}

fn moved(head: &str, entity: ControlEntity) -> ActionControlRecord {
    ActionControlRecord {
        entity_type: entity,
        ..record(head, None, None, ControlMode::Rows, vec![])
    }
}

fn row(
    head: &str,
    row: LevelRow,
    mode: ControlMode,
    actions: Vec<PageAction>,
) -> ActionControlRecord {
    ActionControlRecord {
        confidence: Some(row),
        ..record(head, None, None, mode, actions)
    }
}

async fn bed_with(records: Vec<ActionControlRecord>, config: serde_json::Value) -> TestBed {
    TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config,
        off_switches: vec![switched_off()],
        control_records: records,
    })
    .await
}

async fn bed(records: Vec<ActionControlRecord>) -> TestBed {
    bed_with(records, on_config(false, None)).await
}

fn info(t: &Trace) -> &BTreeMap<String, String> {
    &t.outcomes.last().expect("one outcome").info
}

fn without_expiry(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(o) => o
            .iter()
            .map(|(k, v)| {
                let v = if k == "expiresAtMsec" {
                    serde_json::Value::Null
                } else {
                    without_expiry(v)
                };
                (k.clone(), v)
            })
            .collect(),
        serde_json::Value::Array(a) => a.iter().map(without_expiry).collect(),
        other => other.clone(),
    }
}

fn effect(t: &Trace) -> (String, Vec<(String, serde_json::Value)>, i64, Vec<String>) {
    (
        t.status.clone(),
        t.ais
            .iter()
            .map(|(k, v)| (k.clone(), without_expiry(v)))
            .collect(),
        t.tickets,
        t.dedup_calls.clone(),
    )
}

fn old_info(t: &Trace) -> Vec<BTreeMap<String, String>> {
    t.outcomes
        .iter()
        .map(|o| {
            o.info
                .iter()
                .filter(|(k, _)| {
                    !ACTION_CONTROL_INFO_KEYS.contains(&k.as_str())
                        && !crate::confidence::INFO_KEYS.contains(&k.as_str())
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .collect()
}

fn assert_logged(t: &Trace, because: &str) {
    let want = if info(t).contains_key("action_specs") {
        "dry_run"
    } else {
        action_control::STATUS_NO_ACTIONS
    };
    assert_eq!(t.status, want, "{t:?}");
    assert!(t.outcomes.last().unwrap().dry_run);
    assert_eq!(
        info(t)["matched_rule_id"],
        action_control::PAGE_RULE_ID,
        "never the rules"
    );
    assert!(t.ais.is_empty(), "no AIS call");
    assert_eq!(t.tickets, 0, "no daily-cap ticket");
    assert!(
        t.dedup_calls.iter().all(|c| c.starts_with("peek")),
        "a peek, never a claim or a write: {:?}",
        t.dedup_calls
    );
    let i = info(t);
    assert_eq!(i["action_control_mode"], "recording");
    assert_eq!(i["dry_run_source"], "page");
    assert_eq!(i["logged_because"], because, "{i:?}");
}

const LEVELS: [i32; 5] = [0, 1, 2, 3, 9];

#[tokio::test]
async fn a_scope_never_moved_carries_no_row_and_ignores_row_records() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let plain = bed(vec![]).await;
    let inert = bed(vec![
        row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
        row(BB1.0, LevelRow::Missing, ControlMode::Live, vec![SUSPEND]),
    ])
    .await;
    let head_live = bed(vec![record(
        BB1.0,
        None,
        None,
        ControlMode::Live,
        vec![LIVENESS],
    )])
    .await;
    let head_live_with_rows = bed(vec![
        record(BB1.0, None, None, ControlMode::Live, vec![LIVENESS]),
        row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
    ])
    .await;
    for account in Account::ALL {
        for level in LEVELS {
            let s = with_level(bb1(account), level, Some(evidence(ConfidenceLevel::High)));
            let want = plain.run_fresh(&s).await;
            let got = inert.run_fresh(&s).await;
            assert_eq!(effect(&got), effect(&want), "{account:?} {level}");
            assert_eq!(old_info(&got), old_info(&want), "{account:?} {level}");
            let want = head_live.run_fresh(&s).await;
            let got = head_live_with_rows.run_fresh(&s).await;
            assert_eq!(effect(&got), effect(&want), "{account:?} {level}");
            for t in [&want, &got] {
                assert!(!info(t).contains_key("level_row"), "{account:?} {level}");
                assert!(
                    !info(t).contains_key("logged_because"),
                    "{account:?} {level}"
                );
            }
            assert_eq!(got.outcomes[0].confidence, level, "the level as sent");
        }
    }
}

#[tokio::test]
async fn an_unset_row_logs_with_no_selection() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![moved(BB1.0, ControlEntity::User)]).await;
    for (level, want_row) in [(0, "missing"), (1, "low"), (2, "mid"), (3, "high")] {
        let t = b
            .run_fresh(&with_level(
                bb1(Account::Clean),
                level,
                Some(evidence(ConfidenceLevel::High)),
            ))
            .await;
        assert_logged(&t, "no_selection");
        assert_eq!(info(&t)["level_row"], want_row);
        assert!(
            info(&t).contains_key("rules_would_rule_id"),
            "what the rules would do is recorded"
        );
        assert_eq!(info(&t)["action_control_lkey"], format!("user/h/{}", BB1.0));
    }
    let b = bed(vec![
        moved(BB1.0, ControlEntity::User),
        row(BB1.0, LevelRow::High, ControlMode::Recording, vec![SUSPEND]),
    ])
    .await;
    let t = b.run_fresh(&high(Account::Clean)).await;
    assert_logged(&t, "no_selection");
    assert_eq!(t.status, "dry_run");
    assert_eq!(t.dedup_calls, vec!["peek(class=3)=None"]);
    assert_eq!(
        info(&t)["action_specs"],
        r#"[{"kind":"suspend","permanent":false,"policy":"PlatformManipulation"}]"#
    );
    assert_eq!(
        info(&t)["action_control_lkey"],
        format!("user/l/{}/high", BB1.0)
    );
}

#[tokio::test]
async fn the_built_in_table_never_acts() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![moved(BB1.0, ControlEntity::User)]).await;
    for account in Account::ALL {
        for level in LEVELS {
            let s = with_level(bb1(account), level, Some(evidence(ConfidenceLevel::High)));
            let t = b.run_fresh(&s).await;
            assert!(t.ais.is_empty(), "{account:?} {level}: {t:?}");
            assert_eq!(t.tickets, 0, "{account:?} {level}");
            assert!(t.outcomes[0].dry_run, "{account:?} {level}");
        }
    }
}

#[tokio::test]
async fn a_rules_row_and_a_move_with_every_row_rules_equal_today() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let today = bed(vec![]).await;
    let all_rules = bed(std::iter::once(moved(BB1.0, ControlEntity::User))
        .chain(LevelRow::ALL.map(|r| row(BB1.0, r, ControlMode::Rules, vec![])))
        .collect())
    .await;
    let mut labels = rule_labels(BAKED_USER_RULES);
    labels.push(String::new());
    for label in &labels {
        let ls: Vec<&str> = if label.is_empty() {
            vec![]
        } else {
            vec![label.as_str()]
        };
        for account in Account::ALL {
            for level in LEVELS {
                let s = with_level(
                    message(EntityType::User, BB1.0, BB1.1, &ls, account),
                    level,
                    Some(evidence(ConfidenceLevel::High)),
                );
                let want = today.run_fresh(&s).await;
                let got = all_rules.run_fresh(&s).await;
                let id = format!("{label} {account:?} {level}");
                assert_eq!(effect(&got), effect(&want), "{id}");
                assert_eq!(old_info(&got), old_info(&want), "{id}");
                assert_eq!(got.outcomes[0].dry_run, want.outcomes[0].dry_run, "{id}");
                assert_eq!(info(&got)["action_control_mode"], "rules", "{id}");
                assert!(info(&got).contains_key("level_row"), "{id}");
                assert!(!info(&got).contains_key("logged_because"), "{id}");
            }
        }
    }
}

#[tokio::test]
async fn an_unknown_level_lands_on_missing() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let unknown = with_level(
        bb1(Account::Clean),
        9,
        Some(evidence(ConfidenceLevel::High)),
    );

    let b = bed(vec![
        moved(BB1.0, ControlEntity::User),
        row(BB1.0, LevelRow::Low, ControlMode::Live, vec![LIVENESS]),
    ])
    .await;
    let t = b.run_fresh(&unknown).await;
    assert_logged(&t, "unknown_level");
    assert_eq!(info(&t)["level_row"], "missing");
    assert_eq!(t.outcomes[0].confidence, 9, "kept as sent");

    let b = bed(vec![
        moved(BB1.0, ControlEntity::User),
        row(BB1.0, LevelRow::Missing, ControlMode::Live, vec![LIVENESS]),
        row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
    ])
    .await;
    let liveness = bed(vec![record(
        BB1.0,
        None,
        None,
        ControlMode::Live,
        vec![LIVENESS],
    )])
    .await
    .run_fresh(&bb1(Account::Clean))
    .await;
    let mut no_read = unknown.clone();
    no_read.confidence_evidence = None;
    for s in [unknown, no_read, bb1(Account::Clean)] {
        let t = b.run_fresh(&s).await;
        assert_eq!(effect(&t), effect(&liveness), "{t:?}");
        assert_eq!(info(&t)["level_row"], "missing");
        assert!(!info(&t).contains_key("logged_because"));
    }
}

#[tokio::test]
async fn a_live_level_row_acts_whatever_its_evidence_with_its_own_actions() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![
        moved(BB1.0, ControlEntity::User),
        row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
        row(
            BB1.0,
            LevelRow::Mid,
            ControlMode::Live,
            vec![LIVENESS, SHR7],
        ),
        row(BB1.0, LevelRow::Low, ControlMode::Live, vec![LIVENESS]),
    ])
    .await;
    let good = b.run_fresh(&high(Account::Clean)).await;
    assert_eq!(good.status, "success", "{good:?}");
    assert_eq!(info(&good)["level_row"], "high");
    assert_eq!(info(&good)["confidence_evidence_supports"], "high");

    let mut old = evidence(ConfidenceLevel::High);
    old.read_at_ms = now_millis() - 31 * DAY_MS;
    let mut typed = evidence(ConfidenceLevel::High);
    typed.reads = 5;
    for (ev, supports) in [
        (None, "none"),
        (Some(evidence(ConfidenceLevel::Mid)), "mid"),
        (Some(evidence(ConfidenceLevel::Low)), "low"),
        (Some(old), "none"),
        (Some(typed), "none"),
    ] {
        let t = b
            .run_fresh(&with_level(
                bb1(Account::Clean),
                proto::Confidence::High as i32,
                ev.clone(),
            ))
            .await;
        assert_eq!(
            effect(&t),
            effect(&good),
            "the high row's own actions: {ev:?}"
        );
        let i = info(&t);
        assert_eq!(i["level_row"], "high");
        assert_eq!(i["action_control_mode"], "live", "{ev:?}");
        assert_eq!(i["confidence_evidence_supports"], supports, "{ev:?}");
        assert!(!i.contains_key("logged_because"), "{ev:?}");
        assert!(
            !i.contains_key(action_control::FORCED_RECORDING_KEY),
            "{ev:?}"
        );
    }
}

#[tokio::test]
async fn a_row_stored_with_an_override_reads_and_acts_and_the_override_is_inert() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let plain = bed(vec![
        moved(BB1.0, ControlEntity::User),
        row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
    ])
    .await;
    let no_read = with_level(bb1(Account::Clean), proto::Confidence::High as i32, None);
    let want = plain.run_fresh(&no_read).await;
    assert_eq!(want.status, "success", "{want:?}");
    for expires_at_ms in [now_millis() + DAY_MS, now_millis() - DAY_MS / 2] {
        let mut overridden = row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]);
        overridden.updated_at_ms = now_millis() - DAY_MS;
        overridden.evidence_override = Some(Box::new(EvidenceOverride {
            by: "operator@example.com".into(),
            reason: "read running".into(),
            expires_at_ms,
        }));
        let b = bed(vec![moved(BB1.0, ControlEntity::User), overridden]).await;
        let t = b.run_fresh(&no_read).await;
        assert_eq!(effect(&t), effect(&want), "{expires_at_ms}");
        assert_eq!(info(&t)["action_control_mode"], "live");
        assert!(!info(&t).contains_key("confidence_override_by"));
    }
}

#[tokio::test]
async fn a_live_row_passes_every_check_and_says_why_it_logged() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let live = |head: &str| {
        vec![
            moved(head, ControlEntity::User),
            row(head, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
        ]
    };
    let by_row = bed(live(BB1.0)).await;
    let by_record = bed(vec![record(
        BB1.0,
        None,
        None,
        ControlMode::Live,
        vec![SUSPEND],
    )])
    .await;
    for account in Account::ALL {
        let got = by_row.run_fresh(&high(account)).await;
        let want = by_record.run_fresh(&bb1(account)).await;
        assert_eq!(effect(&got), effect(&want), "{account:?}");
    }

    let off = bed_with(live(BB1.0), base_config(false, None)).await;
    assert_logged(
        &off.run_fresh(&high(Account::Clean)).await,
        "page_control_off",
    );

    let stale = bed(live(BB1.0)).await;
    stale
        .ctx
        .action_control
        .age_for_test(std::time::Duration::from_secs(3_600));
    assert_logged(&stale.run_fresh(&high(Account::Clean)).await, "stale_copy");

    let cse = with_level(
        message(
            EntityType::User,
            BB1.0,
            BB1.1,
            &["anchor_campaign_suspend_cse"],
            Account::Clean,
        ),
        proto::Confidence::High as i32,
        Some(evidence(ConfidenceLevel::High)),
    );
    let got = by_row.run_fresh(&cse).await;
    let want = by_record
        .run_fresh(&message(
            EntityType::User,
            BB1.0,
            BB1.1,
            &["anchor_campaign_suspend_cse"],
            Account::Clean,
        ))
        .await;
    assert_eq!(
        got.status, "success",
        "a live row acts like any live row: {got:?}"
    );
    assert_eq!(got.ais.len(), 1);
    assert!(!info(&got).contains_key("logged_because"));
    assert_eq!(
        want.status, "success",
        "a whole-head live record acts like any live record: {want:?}"
    );
    assert!(!info(&want).contains_key(action_control::FORCED_RECORDING_KEY));

    let switched = bed(live(SWITCHED_OFF_HEAD)).await;
    let s = with_level(
        message(
            EntityType::User,
            SWITCHED_OFF_HEAD,
            "clf_v1",
            &[LABEL],
            Account::Clean,
        ),
        proto::Confidence::High as i32,
        Some(evidence(ConfidenceLevel::High)),
    );
    let t = switched.run_fresh(&s).await;
    assert_eq!(t.status, off_switch::STATUS_SWITCHED_OFF);
    assert!(t.ais.is_empty());
    assert_eq!(info(&t)["logged_because"], "off_switch");

    let mut bad = row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]);
    bad.band = Some(LABEL.into());
    let broken = bed(vec![moved(BB1.0, ControlEntity::User), bad]).await;
    assert_logged(&broken.run_fresh(&high(Account::Clean)).await, "unreadable");

    let staging = bed_with(live(BB1.0), on_config(true, None)).await;
    let t = staging.run_fresh(&high(Account::Clean)).await;
    assert_eq!(t.status, "dry_run");
    assert!(t.ais.is_empty());
    assert_eq!(info(&t)["dry_run_source"], "global");
}

#[tokio::test]
async fn a_classifier_on_rows_wins_and_band_records_under_a_moved_head_do_not_count() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![
        moved(BB1.0, ControlEntity::User),
        row(BB1.0, LevelRow::High, ControlMode::Live, vec![LIVENESS]),
        record(BB1.0, None, Some(LABEL), ControlMode::Live, vec![SUSPEND]),
        ActionControlRecord {
            confidence: Some(LevelRow::High),
            ..record(
                BB1.0,
                Some(BB1.1),
                None,
                ControlMode::Recording,
                vec![SUSPEND],
            )
        },
        record(BB1.0, Some(BB1.1), None, ControlMode::Rows, vec![]),
    ])
    .await;
    let t = b.run_fresh(&high(Account::Clean)).await;
    assert_logged(&t, "no_selection");
    assert_eq!(
        info(&t)["action_control_lkey"],
        format!("user/cl/{}/{}/high", BB1.0, BB1.1)
    );

    let mut other = high(Account::Clean);
    other.model_version = "other_v1".into();
    let t = b.run_fresh(&other).await;
    assert_eq!(
        t.status, "success",
        "the head's high row, not the band record"
    );
    assert_eq!(
        info(&t)["action_control_lkey"],
        format!("user/l/{}/high", BB1.0)
    );
    assert_eq!(info(&t)["action_specs"], r#"[{"kind":"liveness_check"}]"#);
}

#[tokio::test]
async fn a_child_safety_head_on_rows_acts_like_any_other_head() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let (cse_head, cse_model) = HEADS[1];
    let msg = |account, level: i32| {
        with_level(
            message(EntityType::User, cse_head, cse_model, &[LABEL], account),
            level,
            Some(evidence(ConfidenceLevel::High)),
        )
    };
    let live = bed(vec![
        moved(cse_head, ControlEntity::User),
        row(cse_head, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
    ])
    .await;
    let other = bed(vec![
        moved(BB1.0, ControlEntity::User),
        row(BB1.0, LevelRow::High, ControlMode::Live, vec![SUSPEND]),
    ])
    .await;
    for account in Account::ALL {
        let got = live
            .run_fresh(&msg(account, proto::Confidence::High as i32))
            .await;
        let want = other.run_fresh(&high(account)).await;
        let shape = |t: &Trace| {
            let kinds: Vec<String> = t
                .ais
                .iter()
                .filter_map(|(_, r)| r.get("action").and_then(|a| a.as_object()))
                .flat_map(|a| a.keys().cloned())
                .collect();
            (t.status.clone(), t.tickets, t.dedup_calls.clone(), kinds)
        };
        assert_eq!(shape(&got), shape(&want), "{account:?}");
        assert_eq!(
            info(&got).get("action_specs"),
            info(&want).get("action_specs")
        );
        assert!(!info(&got).contains_key("logged_because"), "{account:?}");
    }

    let today = bed(vec![]).await;
    let all_rules = bed(std::iter::once(moved(cse_head, ControlEntity::User))
        .chain(LevelRow::ALL.map(|r| row(cse_head, r, ControlMode::Rules, vec![])))
        .collect())
    .await;
    for account in Account::ALL {
        for level in LEVELS {
            let s = msg(account, level);
            let want = today.run_fresh(&s).await;
            let got = all_rules.run_fresh(&s).await;
            assert_eq!(effect(&got), effect(&want), "{account:?} {level}");
            assert_eq!(old_info(&got), old_info(&want), "{account:?} {level}");
        }
    }
}
