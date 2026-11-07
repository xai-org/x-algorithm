use std::collections::BTreeMap;
use std::time::Duration;

use xai_abuse_proto::enforcement::ScoreResult;

use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{
    ActionControlRecord, ControlEntity, ControlMode, LevelRow, Lifetime, PageAction,
};
use crate::replay_tests::{
    Account, BAKED_POST_RULES, BAKED_USER_RULES, BedConfig, HEADS, TestBed, Trace, base_config,
    factory_message, message, on_config, scenarios, switched_off,
};

const SUSPEND: PageAction = PageAction::SUSPEND;
const LIVENESS: PageAction = PageAction::LivenessCheck {};
const SHR7: PageAction = PageAction::SpamHighRecall {
    lifetime: Lifetime::Days7,
};

const BB1: (&str, &str) = HEADS[0];

fn suspend_label() -> String {
    "anchor_campaign_suspend".into()
}

async fn bed(records: Vec<ActionControlRecord>, dry_run: bool) -> TestBed {
    bed_with(records, on_config(dry_run, None), BAKED_USER_RULES).await
}

async fn bed_with(
    records: Vec<ActionControlRecord>,
    config: serde_json::Value,
    user_rules: &str,
) -> TestBed {
    TestBed::new(BedConfig {
        user_rules,
        post_rules: BAKED_POST_RULES,
        config,
        off_switches: vec![switched_off()],
        control_records: records,
    })
    .await
}

fn bb1(labels: &[&str], account: Account) -> ScoreResult {
    message(EntityType::User, BB1.0, BB1.1, labels, account)
}

fn info(t: &Trace) -> &BTreeMap<String, String> {
    &t.outcomes.last().expect("one outcome").info
}

fn ais_names(t: &Trace) -> Vec<String> {
    t.ais
        .iter()
        .flat_map(|(_, r)| {
            let actions: Vec<&serde_json::Value> = match r.get("actions") {
                Some(serde_json::Value::Array(a)) => a.iter().collect(),
                _ => r.get("action").into_iter().collect(),
            };
            actions
                .into_iter()
                .filter_map(|a| a.as_object().and_then(|o| o.keys().next().cloned()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn live(head: &str, band: Option<&str>, actions: Vec<PageAction>) -> ActionControlRecord {
    record(head, None, band, ControlMode::Live, actions)
}

const FACTORY_HEAD: &str = "IsDelayedSuspendSpam";

fn factory_row(row: LevelRow, mode: ControlMode, actions: Vec<PageAction>) -> ActionControlRecord {
    ActionControlRecord {
        confidence: Some(row),
        ..record(FACTORY_HEAD, None, None, mode, actions)
    }
}

#[tokio::test]
async fn a_factory_row_without_actions_records_before_any_fetch() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for (records, logged) in [
        (vec![], "no_selection"),
        (
            vec![live(FACTORY_HEAD, None, vec![SUSPEND])],
            "no_selection",
        ),
        (
            vec![factory_row(LevelRow::Missing, ControlMode::Rules, vec![])],
            "factory_rules_row",
        ),
    ] {
        let b = bed(records, false).await;
        for account in Account::ALL {
            let t = b.run_fresh(&factory_message(account)).await;
            assert_eq!(t.status, action_control::STATUS_NO_ACTIONS, "{account:?}");
            assert!(t.ais.is_empty() && t.tickets == 0);
            assert_eq!(t.dedup_calls, vec!["peek(class=0)=None"], "{account:?}");
            assert!(t.outcomes[0].dry_run);
            let i = info(&t);
            assert_eq!(i["level_row"], "missing");
            assert_eq!(i["logged_because"], logged);
            assert_eq!(i["matched_rule_id"], action_control::PAGE_RULE_ID);
            assert!(!i.keys().any(|k| k.starts_with("rules_would")), "{i:?}");
        }
        assert_eq!(b.fetches().await, 0, "{logged}");
        b.run_fresh(&bb1(&[], Account::Clean)).await;
        assert!(b.fetches().await > 0, "a rules decision fetches");
    }
    let mut ignore = on_config(false, None);
    ignore["action_control"] = serde_json::json!({"enabled": true, "ignore_store": true});
    let b = bed_with(vec![], ignore, BAKED_USER_RULES).await;
    let t = b.run_fresh(&factory_message(Account::Clean)).await;
    assert_eq!(t.status, action_control::STATUS_NO_ACTIONS);
    assert_eq!(t.dedup_calls, vec!["peek(class=0)=None"]);
    assert_eq!(info(&t)["logged_because"], "ignore_store");
    assert!(!info(&t).keys().any(|k| k.starts_with("rules_would")));
    assert_eq!(b.fetches().await, 0);
}

#[tokio::test]
async fn a_factory_row_acts_whatever_its_evidence_without_the_rules() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    const DAY: i64 = 86_400_000;
    let b = bed(
        vec![factory_row(
            LevelRow::High,
            ControlMode::Live,
            vec![SUSPEND],
        )],
        false,
    )
    .await;
    let with_evidence = |days: i64| {
        let mut s = factory_message(Account::Clean);
        s.confidence = abuse_proto::Confidence::High as i32;
        s.confidence_evidence = Some(abuse_proto::ConfidenceEvidence {
            precision_lower_bound: 0.96,
            reads: 59,
            humans: 0,
            read_at_ms: now_millis() - days * DAY,
            link: "read:rd_1".into(),
        });
        s
    };
    let mut no_read = with_evidence(0);
    no_read.confidence_evidence = None;
    for (s, supports) in [
        (with_evidence(364), "high"),
        (with_evidence(366), "none"),
        (no_read, "none"),
    ] {
        let t = b.run_fresh(&s).await;
        assert_eq!(t.status, "success", "{:?}", info(&t));
        assert_eq!((t.ais.len(), t.tickets), (1, 1));
        let i = info(&t);
        assert_eq!(i["action_control_lkey"], "user/l/IsDelayedSuspendSpam/high");
        assert_eq!(i["matched_rule_id"], action_control::PAGE_RULE_ID);
        assert_eq!(i["confidence_evidence_supports"], supports);
        assert!(!i.contains_key("logged_because"), "{i:?}");
        assert!(!i.keys().any(|k| k.starts_with("rules_would")), "{i:?}");
    }
}

#[tokio::test]
async fn a_factory_row_that_acts_takes_dedup_the_off_switch_and_the_cap() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let row = || factory_row(LevelRow::Missing, ControlMode::Live, vec![SUSPEND]);
    let score = factory_message(Account::Clean);
    let b = bed(vec![row()], false).await;
    let t = b.run_fresh(&score).await;
    assert_eq!(t.status, "success", "{:?}", info(&t));
    assert_eq!(
        t.dedup_calls[0], "try_insert(class=3)=new",
        "claims at its suspend"
    );
    assert_eq!((t.ais.len(), t.tickets), (1, 1));
    let again = b.run(&score).await;
    assert_eq!(again.status, "dedup_skipped");
    assert!(again.ais.is_empty() && again.tickets == 0);

    let mut off = off_switch::test_support::record(FACTORY_HEAD, None, true, None);
    off.updated_at_ms = 1;
    let switched = TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config: on_config(false, None),
        off_switches: vec![off],
        control_records: vec![row()],
    })
    .await;
    let t = switched.run_fresh(&score).await;
    assert_eq!(t.status, off_switch::STATUS_SWITCHED_OFF);
    assert!(t.ais.is_empty() && t.tickets == 0);

    let capped = bed_with(vec![row()], on_config(false, Some(0)), BAKED_USER_RULES).await;
    let t = capped.run_fresh(&score).await;
    assert_eq!(t.status, "max_enforcement_reached");
    assert!(t.ais.is_empty());
}

#[tokio::test]
async fn a_factory_decision_for_a_test_user_is_skipped_first() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![], false).await;
    let mut score = factory_message(Account::Clean);
    score.user_id = 2_050_000_001;
    assert!(test_user::is_test_user_id(score.user_id));
    let t = b.run_fresh(&score).await;
    assert_eq!(t.status, "test_user_skipped");
    assert_eq!(b.fetches().await, 0);
}

#[tokio::test]
async fn factory_decisions_and_runs_mark_their_head_factory_in_the_seen_list() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut acting = message(
        EntityType::User,
        "FactoryActing",
        "factory_acting@1",
        &["model_factory"],
        Account::Clean,
    );
    acting.policy_version = "1".into();
    let acting_row = ActionControlRecord {
        confidence: Some(LevelRow::Missing),
        ..record(
            "FactoryActing",
            None,
            None,
            ControlMode::Live,
            vec![SUSPEND],
        )
    };
    let mut config = on_config(false, None);
    config["backfill_runs"] =
        serde_json::json!({"page_keys": ["page"], "max_user_decisions_per_day": 10});
    let b = bed_with(vec![acting_row], config, BAKED_USER_RULES).await;
    let (run_head, run_model) = HEADS[3];
    let mut run = crate::backfill::test_support::put_running(
        &b.backfill_store,
        "run-00000001",
        run_head,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    run.factory = true;
    b.backfill_store.rows.lock().unwrap().insert(
        (crate::backfill::RUN_PKEY.into(), run.run_id.clone()),
        serde_json::to_vec(&run).unwrap(),
    );
    let mut run_score = message(EntityType::User, run_head, run_model, &[], Account::Clean);
    run_score.backfill_run_id = run.run_id.clone();
    run_score.score_ts_ms = now_millis() - 60_000;

    assert_eq!(
        b.run_fresh(&factory_message(Account::Clean)).await.status,
        action_control::STATUS_NO_ACTIONS
    );
    assert_eq!(b.run_fresh(&acting).await.status, "success");
    b.run_fresh(&bb1(&[&suspend_label()], Account::Clean)).await;
    b.dedup.slots.lock().unwrap().clear();
    b.run_on(&b.backfill_processor(), &run_score).await;

    b.ctx.action_control.spawn_seen_writer().unwrap();
    let seen = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let seen = b.ctx.action_control.scan_seen().await.unwrap();
            if seen.len() >= 4 {
                return seen;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("four seen rows");
    let of = |head: &str| {
        seen.iter()
            .find(|s| s.head == head)
            .unwrap_or_else(|| panic!("{head} not seen: {seen:?}"))
    };
    for head in [FACTORY_HEAD, "FactoryActing", run_head] {
        let s = of(head);
        assert!(s.factory, "{head}");
        assert_eq!(s.rule_id, "", "{head}");
        assert_eq!(s.action_specs, serde_json::json!([]), "{head}");
    }
    let rules = of(BB1.0);
    assert!(!rules.factory);
    assert_eq!(rules.rule_id, suspend_label());
}

#[test]
fn only_a_policy_version_marks_a_factory_message() {
    let mut s = factory_message(Account::Clean);
    assert!(is_factory_message(&s));
    s.policy_version.clear();
    assert!(
        !is_factory_message(&s),
        "the routing label alone is not one"
    );
    assert!(!is_factory_message(&bb1(&[], Account::Clean)));
}

#[tokio::test]
async fn a_grox_impersonation_scam_message_is_suspended_by_its_rule() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let grox = |account| {
        message(
            EntityType::User,
            "ImpersonationScam",
            "impersonation_scam_classifier_v1",
            &[
                "impersonation_scam_classifier_v1_threshold_reached",
                "impersonation_scam_classifier_v1_suspend",
            ],
            account,
        )
    };
    let score = grox(Account::Clean);
    assert!(!is_factory_message(&score));
    let unset = bed(vec![], false).await;
    let t = unset.run_fresh(&score).await;
    assert_eq!(t.status, action_control::STATUS_NO_ACTIONS);
    assert!(t.ais.is_empty() && t.outcomes[0].dry_run);
    let b = bed(
        vec![record(
            "ImpersonationScam",
            None,
            None,
            ControlMode::Rules,
            vec![],
        )],
        false,
    )
    .await;
    let t = b.run_fresh(&score).await;
    assert_eq!(t.status, "success", "{:?}", info(&t));
    assert_eq!(ais_names(&t), vec!["suspendUser"]);
    let i = info(&t);
    assert_eq!(i["matched_rule_id"], "act_impersonation_scam_suspend");
    assert_eq!(i["action_control_mode"], "rules");
    assert_eq!(
        i["action_specs"],
        r#"[{"kind":"suspend","permanent":false,"policy":"PlatformManipulation"}]"#
    );
    for (account, want) in [
        (Account::Suspended, ALREADY_SUSPENDED),
        (Account::Deactivated, "already_deactivated"),
        (Account::BetweenLimits, "high_follower_count"),
        (Account::HighCred, "pagerank_skipped"),
    ] {
        let t = b.run_fresh(&grox(account)).await;
        assert_eq!(t.status, want, "{account:?}");
        assert!(t.ais.is_empty(), "{account:?}");
        assert_eq!(t.tickets, 0, "{account:?}");
    }
}

#[tokio::test]
async fn a_user_record_on_a_shared_head_never_touches_its_post_decisions() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let head = "SharedSpamBot";
    let model = "shared_spam_v1";
    let post_rules = ActionControlRecord {
        entity_type: ControlEntity::Post,
        ..record(head, None, None, ControlMode::Rules, vec![])
    };
    let none = bed(vec![post_rules.clone()], false).await;
    let user_live = bed(vec![post_rules, live(head, None, vec![SUSPEND])], false).await;
    let mut acted = 0;
    for label in crate::replay_tests::rule_labels(BAKED_POST_RULES) {
        let post = message(EntityType::Post, head, model, &[&label], Account::Clean);
        let was = none.run_fresh(&post).await;
        let now = user_live.run_fresh(&post).await;
        assert_eq!(now.status, was.status, "{label}");
        assert_eq!(ais_names(&now), ais_names(&was), "{label}");
        assert_eq!(
            (now.tickets, &now.dedup_calls),
            (was.tickets, &was.dedup_calls),
            "{label}"
        );
        if let Some(o) = now.outcomes.last() {
            assert_eq!(
                o.info.get("action_control_mode").map(String::as_str),
                Some("rules")
            );
            assert_eq!(
                o.info.get("action_control_lkey").map(String::as_str),
                Some("post/h/SharedSpamBot"),
                "{label}"
            );
        }
        acted += usize::from(!now.ais.is_empty());
    }
    assert!(acted > 0, "some post label acts through the rules");
    let user = message(
        EntityType::User,
        head,
        model,
        &[&suspend_label()],
        Account::Clean,
    );
    let t = user_live.run_fresh(&user).await;
    let i = info(&t);
    assert_eq!(i["action_control_mode"], "live");
    assert_eq!(i["action_control_lkey"], "user/h/SharedSpamBot");
    assert_eq!(i["matched_rule_id"], "page");
}

#[tokio::test]
async fn recording_takes_no_dedup_claim_no_ticket_and_no_ais_call() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    for rec in [
        record(BB1.0, None, None, ControlMode::Recording, vec![]),
        record(BB1.0, None, None, ControlMode::Recording, vec![LIVENESS]),
    ] {
        let page_actions = !rec.actions.is_empty();
        let b = bed(vec![rec], false).await;
        let t = b.run_fresh(&bb1(&[&label], Account::Clean)).await;
        assert_eq!(t.status, "dry_run");
        let want_peek = if page_actions {
            "peek(class=2)=None"
        } else {
            "peek(class=3)=None"
        };
        assert_eq!(
            t.dedup_calls,
            vec![want_peek],
            "a peek, never a claim or a write"
        );
        assert_eq!(t.tickets, 0, "no daily-cap ticket");
        assert!(t.ais.is_empty(), "no AIS call");
        let i = info(&t);
        assert_eq!(i["dry_run_source"], "page");
        assert_eq!(i["action_control_mode"], "recording");
        assert_eq!(i["would_dedup_skip"], "false");
        assert_eq!(i["would_rate_cap"], "false");
        assert_eq!(i["env"], "prod");
        let want_specs = if page_actions {
            r#"[{"kind":"liveness_check"}]"#
        } else {
            r#"[{"kind":"suspend","permanent":false,"policy":"PlatformManipulation"}]"#
        };
        assert_eq!(i["action_specs"], want_specs);
        assert!(!i.contains_key("acted_class"));

        b.dedup.hold(EntityType::User, Account::Clean.user_id(), 3);
        let t = b.run(&bb1(&[&label], Account::Clean)).await;
        assert_eq!(t.status, "dry_run");
        assert_eq!(info(&t)["would_dedup_skip"], "true");
        assert_eq!(info(&t)["would_dedup_held_class"], "3");
        assert!(
            t.dedup_calls.iter().all(|c| c.starts_with("peek")),
            "{:?}",
            t.dedup_calls
        );
    }
}

#[tokio::test]
async fn producer_dry_run_records_only() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let b = bed(vec![], false).await;
    let mut by_field = bb1(&[&label], Account::Clean);
    by_field.dry_run = true;
    let prefixed_label = format!("dry_run__{label}");
    let by_prefix = bb1(&[&prefixed_label], Account::Clean);
    for score in [by_field, by_prefix] {
        let t = b.run_fresh(&score).await;
        assert_eq!(t.status, "dry_run", "the rules still route on the label");
        assert_eq!(t.dedup_calls, vec!["peek(class=3)=None"]);
        assert_eq!(t.tickets, 0);
        assert!(t.ais.is_empty());
        let i = info(&t);
        assert_eq!(i["dry_run_source"], "producer");
        assert_eq!(i["matched_rule_id"], "anchor_campaign_suspend");
        assert_eq!(i["action_control_mode"], "rules");
    }
    let t = b.run_fresh(&bb1(&[&prefixed_label], Account::Clean)).await;
    assert_eq!(t.outcomes[0].labels, vec![prefixed_label.clone()]);
}

#[tokio::test]
async fn admission_strips_the_prefix_and_an_old_image_drops_it() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let b = bed(vec![], false).await;
    let mut p = b.processor();
    p.labels = HashSet::from([label.clone()]);
    let prefixed = format!("dry_run__{label}");
    let msg = |labels: &[&str]| xai_kafka::KafkaMessage {
        topic: p.topic.clone(),
        payload: Some(bb1(labels, Account::Clean).encode_to_vec()),
        ..Default::default()
    };
    p.process_batch(&[msg(&[&prefixed])]).await.unwrap();
    let outcomes = std::mem::take(&mut *b.outcomes.lock().unwrap());
    assert_eq!(outcomes.len(), 1, "admitted");
    assert_eq!(outcomes[0].status, "dry_run");
    assert_eq!(outcomes[0].info["dry_run_source"], "producer");
    assert!(b.ais.take().is_empty());
    assert!(!p.labels.contains(&prefixed));
    p.process_batch(&[msg(&["unsubscribed"])]).await.unwrap();
    assert!(
        b.outcomes.lock().unwrap().is_empty(),
        "unsubscribed labels are still dropped"
    );
}

#[tokio::test]
async fn live_page_actions_replace_the_rules_and_pass_every_gate() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let b = bed(
        vec![
            live(BB1.0, None, vec![SHR7, LIVENESS]),
            live(crate::replay_tests::SWITCHED_OFF_HEAD, None, vec![LIVENESS]),
        ],
        false,
    )
    .await;
    let t = b.run_fresh(&bb1(&[&label], Account::Clean)).await;
    assert_eq!(t.status, "success");
    assert_eq!(
        ais_names(&t),
        vec!["addLabelsV2", "bounceViaTemplate"],
        "label, then liveness check"
    );
    assert_eq!(t.tickets, 1);
    assert_eq!(
        t.dedup_calls,
        vec!["try_insert(class=2)=new", "update(class=2,own_claim=true)"]
    );
    let i = info(&t);
    assert_eq!(i["matched_rule_id"], "page");
    assert_eq!(i["rules_would_rule_id"], "anchor_campaign_suspend");
    assert_eq!(i["action_control_mode"], "live");
    assert_eq!(i["action_control_scope"], "head");
    assert!(!i.contains_key("dry_run_source"));

    for (account, reason) in [
        (Account::BetweenLimits, "high_follower_count"),
        (Account::AboveTopLimit, "very_high_follower_count"),
        (Account::HighCred, "pagerank_skipped"),
        (Account::NotFound, "user_not_found"),
    ] {
        let t = b.run_fresh(&bb1(&[&label], account)).await;
        assert_eq!(t.status, reason, "{account:?}");
        assert_eq!(info(&t)["matched_rule_id"], reason);
        assert!(t.ais.is_empty());
        assert_eq!(t.tickets, 0);
    }
    let off = message(
        EntityType::User,
        crate::replay_tests::SWITCHED_OFF_HEAD,
        "clf_v1",
        &[&label],
        Account::Clean,
    );
    let t = b.run_fresh(&off).await;
    assert_eq!(t.status, off_switch::STATUS_SWITCHED_OFF);
    assert_eq!(info(&t)["action_specs"], r#"[{"kind":"liveness_check"}]"#);
    assert!(t.ais.is_empty());

    let b = bed(vec![live(BB1.0, None, vec![SUSPEND])], false).await;
    let t = b.run_fresh(&bb1(&[&label], Account::Suspended)).await;
    assert_eq!(t.status, ALREADY_SUSPENDED);
    assert!(t.ais.is_empty());

    let b = bed_with(
        vec![live(BB1.0, None, vec![SUSPEND])],
        on_config(false, Some(0)),
        BAKED_USER_RULES,
    )
    .await;
    let t = b.run_fresh(&bb1(&[&label], Account::Clean)).await;
    assert_eq!(t.status, "max_enforcement_reached");
    assert!(t.ais.is_empty());
}

#[tokio::test]
async fn repeat_label_skip_is_label_only() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let b = bed(vec![live(BB1.0, None, vec![SHR7])], false).await;
    let t = b
        .run_fresh(&bb1(&[&label], Account::HasSpamHighRecall))
        .await;
    assert_eq!(t.status, "already_spam_high_recall_labeled");
    assert!(t.ais.is_empty());
    let b = bed(vec![live(BB1.0, None, vec![SHR7, LIVENESS])], false).await;
    let t = b
        .run_fresh(&bb1(&[&label], Account::HasSpamHighRecall))
        .await;
    assert_eq!(t.status, "success");
    assert_eq!(
        ais_names(&t),
        vec!["addLabelsV2", "bounceViaTemplate"],
        "a composite is never split"
    );
}

#[tokio::test]
async fn phoenix_skip_page_label_writes_label_118_once() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let phoenix = PageAction::PhoenixSkip {
        lifetime: Lifetime::Days30,
    };
    let b = bed(vec![live(BB1.0, None, vec![phoenix])], false).await;
    let t = b.run_fresh(&bb1(&[&label], Account::Clean)).await;
    assert_eq!(t.status, "success");
    assert_eq!(ais_names(&t), vec!["addLabelsV2"]);
    let label_id = t.ais[0].1.pointer(
        "/action/addLabelsV2/labels/0/nonPolicyBasedLabel/label/labelReferenceWithSafetyLabel/labelReference/labelIdType/nativeUserLabels",
    );
    assert_eq!(label_id, Some(&serde_json::json!("ClientEventPhoenixSkip")));

    let t = b.run_fresh(&bb1(&[&label], Account::HasPhoenixSkip)).await;
    assert_eq!(t.status, "already_phoenix_skip_labeled");
    assert!(t.ais.is_empty());
    let t = b
        .run_fresh(&bb1(&[&label], Account::HasSpamHighRecall))
        .await;
    assert_eq!(t.status, "success", "another label does not count");
}

#[tokio::test]
async fn band_records_split_a_two_band_head() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(
        vec![live(BB1.0, Some("anchor_campaign_user"), vec![LIVENESS])],
        false,
    )
    .await;
    let t = b
        .run_fresh(&bb1(&["anchor_campaign_user"], Account::Clean))
        .await;
    assert_eq!(ais_names(&t), vec!["bounceViaTemplate"]);
    assert_eq!(info(&t)["action_control_scope"], "band");
    let t = b
        .run_fresh(&bb1(&["anchor_campaign_suspend"], Account::Clean))
        .await;
    assert_eq!(
        ais_names(&t),
        vec!["suspendUser"],
        "the other band follows the rules"
    );
    assert_eq!(info(&t)["matched_rule_id"], "anchor_campaign_suspend");
    assert_eq!(info(&t)["action_control_mode"], "rules");
}

#[tokio::test]
async fn global_dry_run_records_only_like_the_other_two_sources() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    for (records, want_peek) in [
        (
            vec![live(BB1.0, None, vec![LIVENESS])],
            "peek(class=2)=None",
        ),
        (vec![], "peek(class=3)=None"),
    ] {
        let b = bed(records, true).await;
        let t = b.run_fresh(&bb1(&[&label], Account::Clean)).await;
        assert_eq!(t.status, "dry_run");
        assert!(t.ais.is_empty());
        let i = info(&t);
        assert_eq!(i["dry_run_source"], "global");
        assert_eq!(
            t.dedup_calls,
            vec![want_peek],
            "a peek, never a claim or a write"
        );
        assert_eq!(t.tickets, 0, "no daily-cap ticket");
        assert_eq!(i["would_dedup_skip"], "false");
        assert_eq!(i["would_rate_cap"], "false");

        b.dedup.hold(EntityType::User, Account::Clean.user_id(), 3);
        let t = b.run(&bb1(&[&label], Account::Clean)).await;
        assert_eq!(t.status, "dry_run");
        assert_eq!(t.dedup_calls, vec![want_peek.replace("None", "Some(3)")]);
        assert_eq!(t.tickets, 0);
        assert_eq!(info(&t)["would_dedup_skip"], "true");
        assert_eq!(info(&t)["would_dedup_held_class"], "3");
    }
}

fn rules_with_a_cse_suspend() -> String {
    let mut yaml = String::from("for_entity: user\nrules:\n");
    for c in safety_checks::USER_CHECKS {
        yaml.push_str(&format!(
            "  - id: {}\n    when: {:?}\n    then: {{ kind: skip, reason: {} }}\n",
            c.id, c.when, c.reason
        ));
    }
    yaml.push_str(
        r#"  - id: act_cse_label
    when: '"cse_label" in score.labels'
    then: { kind: act_suspend_user, perm: true, policy: "Cse" }
  - id: terminal
    when: "true"
    then: { kind: skip, reason: no_match }
"#,
    );
    yaml
}

#[tokio::test]
async fn a_cse_action_chosen_by_the_rules_leaves_a_live_page_setting_acting() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed_with(
        vec![live("SomeHead", None, vec![SUSPEND])],
        on_config(false, None),
        &rules_with_a_cse_suspend(),
    )
    .await;
    let t = b
        .run_fresh(&message(
            EntityType::User,
            "SomeHead",
            "clf",
            &["cse_label"],
            Account::Clean,
        ))
        .await;
    assert_eq!(t.status, "success");
    assert!(!t.ais.is_empty());
    let i = info(&t);
    assert_eq!(i["matched_rule_id"], "page");
    assert_eq!(i["rules_would_rule_id"], "act_cse_label");
    assert_eq!(i["action_control_mode"], "live");
    assert!(!i.contains_key("action_control_forced_recording"), "{i:?}");
}

#[tokio::test]
async fn a_rules_eval_error_leaves_a_live_page_setting_acting() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let rules = rules_with_a_cse_suspend().replace(
        r#"'"cse_label" in score.labels'"#,
        r#"'int("not a number") == 1'"#,
    );
    let seen_calls = || {
        ["queued", "recent", "dropped", "invalid"]
            .iter()
            .map(|l| {
                metrics::ACTION_CONTROL_SEEN_TOTAL
                    .with_label_values(&[l])
                    .get()
            })
            .sum::<u64>()
    };
    let b = bed_with(
        vec![live("SomeHead", None, vec![SUSPEND])],
        on_config(false, None),
        &rules,
    )
    .await;
    let before = seen_calls();
    let t = b
        .run_fresh(&message(
            EntityType::User,
            "SomeHead",
            "clf",
            &["cse_label"],
            Account::Clean,
        ))
        .await;
    assert_eq!(
        seen_calls(),
        before,
        "no seen row when the rules couldn't be evaluated"
    );
    assert_eq!(t.status, "success");
    assert!(!t.ais.is_empty());
    let i = info(&t);
    assert_eq!(i["matched_rule_id"], "page");
    assert!(!i.contains_key("rules_would_rule_id"));
    assert!(!i.contains_key("action_control_forced_recording"), "{i:?}");
}

#[tokio::test]
async fn a_post_page_decision_skips_a_suspended_or_deactivated_author() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let head = "SpamEmbeddingPtosDistilled";
    let label = crate::replay_tests::rule_labels(BAKED_POST_RULES)
        .into_iter()
        .next()
        .expect("a post routing label");
    let b = bed(
        vec![live(
            head,
            None,
            vec![
                PageAction::PostSpamHighRecall {
                    lifetime: Lifetime::Days30,
                },
                PageAction::SuspendAuthor {},
            ],
        )],
        false,
    )
    .await;
    for (account, want) in [
        (Account::Suspended, ALREADY_SUSPENDED),
        (Account::Deactivated, "already_deactivated"),
    ] {
        let t = b
            .run_fresh(&message(EntityType::Post, head, "clf", &[&label], account))
            .await;
        assert_eq!(t.status, want, "{account:?}");
        assert!(t.ais.is_empty(), "{account:?}");
        assert_eq!(t.tickets, 0, "{account:?}");
    }
}

fn live_everywhere(actions: Vec<PageAction>) -> Vec<ActionControlRecord> {
    HEADS
        .iter()
        .map(|(h, _)| *h)
        .chain(std::iter::once(crate::replay_tests::SWITCHED_OFF_HEAD))
        .map(|h| live(h, None, actions.clone()))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_rollback_plan_keeps_an_old_image_from_acting_on_page_managed_heads() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut records = live_everywhere(vec![SUSPEND]);
    records.extend(HEADS.iter().map(|(h, _)| {
        record(
            h,
            None,
            None,
            ControlMode::Recording,
            vec![PageAction::PostSpamHighRecall {
                lifetime: Lifetime::Days30,
            }],
        )
    }));
    let on = on_config(false, None);
    let current = bed_with(records.clone(), on.clone(), BAKED_USER_RULES).await;
    let rules = crate::rules::RulesCache::new();
    let safety = action_control::SafetyChecksByEntity {
        user: safety_checks::status(EntityType::User, &rules.resolve(EntityType::User, None)),
        post: safety_checks::status(EntityType::Post, &rules.resolve(EntityType::Post, None)),
    };
    let list = current
        .ctx
        .action_control
        .list(
            &action_control::ControlConfig::from_config(Some(&on)),
            safety,
        )
        .await;
    let planned = crate::action_control_rollback::plan(&list, "operator@example.com");
    assert!(!planned.is_empty());
    let switches: Vec<off_switch::OffSwitchRecord> = planned
        .iter()
        .map(|p| off_switch::test_support::record(&p.head, p.classifier.as_deref(), true, None))
        .collect();
    let managed: std::collections::HashSet<&str> =
        records.iter().map(|r| r.head.as_str()).collect();
    let mut old_image = base_config(false, None);
    old_image["action_control"] = serde_json::json!({"ignore_store": true});
    let old_bed = |off_switches| {
        TestBed::new(BedConfig {
            user_rules: BAKED_USER_RULES,
            post_rules: BAKED_POST_RULES,
            config: old_image.clone(),
            off_switches,
            control_records: records.clone(),
        })
    };
    let (with_plan, without_plan) = (old_bed(switches).await, old_bed(vec![]).await);
    let (mut ran, mut acted_without_plan) = (0, 0);
    for sc in scenarios(BAKED_USER_RULES, BAKED_POST_RULES)
        .into_iter()
        .filter(|s| !s.dry_run && s.cap.is_none() && s.held_class.is_none())
        .filter(|s| managed.contains(s.id.split('|').nth(1).unwrap_or("")))
    {
        let t = with_plan.run_fresh(&sc.score).await;
        assert!(t.ais.is_empty(), "{}: {} {:?}", sc.id, t.status, t.ais);
        assert_eq!(t.tickets, 0, "{}", sc.id);
        acted_without_plan += usize::from(!without_plan.run_fresh(&sc.score).await.ais.is_empty());
        ran += 1;
    }
    assert!(ran > 100, "ran {ran}");
    assert!(
        acted_without_plan > 0,
        "without the switches the old image acts, so the test bites"
    );
}

async fn assert_no_ais_anywhere(
    config: serde_json::Value,
    user_rules: &str,
    age: Option<Duration>,
) -> usize {
    let b = bed_with(live_everywhere(vec![SUSPEND]), config, user_rules).await;
    if let Some(age) = age {
        b.ctx.action_control.age_for_test(age);
    }
    let mut n = 0;
    for sc in scenarios(user_rules, BAKED_POST_RULES)
        .into_iter()
        .filter(|s| s.id.starts_with("user|") && !s.dry_run)
    {
        let t = b.run_fresh(&sc.score).await;
        assert!(t.ais.is_empty(), "{}: {:?}", sc.id, t.ais);
        assert_eq!(t.tickets, 0, "{}", sc.id);
        assert!(
            t.dedup_calls.iter().all(|c| c.starts_with("peek")),
            "{}: {:?}",
            sc.id,
            t.dedup_calls
        );
        n += 1;
    }
    n
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_failure_state_adds_an_ais_call() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut killed = on_config(false, None);
    killed["action_control"] = serde_json::json!({"enabled": false});
    let mut malformed = on_config(false, None);
    malformed["action_control"] = serde_json::json!({"enabled": "off"});
    let missing_ceiling = BAKED_USER_RULES.replacen(
        "cred.follower_count >= ",
        "false && cred.follower_count >= ",
        1,
    );
    assert_ne!(missing_ceiling, BAKED_USER_RULES);
    let cases: Vec<(&str, serde_json::Value, &str, Option<Duration>)> = vec![
        ("kill switch", killed, BAKED_USER_RULES, None),
        (
            "no action_control object in GrowthBook",
            base_config(false, None),
            BAKED_USER_RULES,
            None,
        ),
        (
            "malformed GrowthBook action_control",
            malformed,
            BAKED_USER_RULES,
            None,
        ),
        (
            "stale copy",
            on_config(false, None),
            BAKED_USER_RULES,
            Some(action_control::STALE_AFTER + Duration::from_secs(1)),
        ),
        (
            "a safety skip loosened in the rules",
            on_config(false, None),
            &missing_ceiling,
            None,
        ),
    ];
    for (name, config, rules, age) in cases {
        let n = assert_no_ais_anywhere(config, rules, age).await;
        assert!(n > 100, "{name}: ran {n}");
    }
}

#[tokio::test]
async fn damaged_rules_still_stop_page_decisions_with_the_built_in_checks() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let loosened = BAKED_USER_RULES.replacen(
        "cred.follower_count >= ",
        "false && cred.follower_count >= ",
        1,
    );
    let b = bed_with(
        vec![record(
            BB1.0,
            None,
            None,
            ControlMode::Recording,
            vec![SUSPEND],
        )],
        on_config(false, None),
        &loosened,
    )
    .await;
    let t = b
        .run(&bb1(&[&suspend_label()], Account::AboveTopLimit))
        .await;
    assert_eq!(t.status, "very_high_follower_count");
}


fn as_page_actions(specs: &serde_json::Value, entity: EntityType) -> Option<Vec<PageAction>> {
    let mut out = Vec::new();
    for s in specs.as_array()? {
        let kind = s.get("kind")?.as_str()?;
        let permanent = s
            .get("permanent")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let days = s.get("ttl_days").and_then(|v| v.as_i64());
        let lifetime = || match (permanent, days) {
            (true, _) => Some(Lifetime::Permanent),
            (false, Some(7)) => Some(Lifetime::Days7),
            (false, Some(30)) => Some(Lifetime::Days30),
            _ => None,
        };
        let label_ok = s.get("label").and_then(|v| v.as_str()) == Some("SpamHighRecall");
        out.push(match (kind, entity) {
            ("suspend", EntityType::User)
                if !permanent && s.get("policy")?.as_str()? == "PlatformManipulation" =>
            {
                PageAction::SUSPEND
            }
            ("suspend", EntityType::Post)
                if !permanent && s.get("policy")?.as_str()? == "PlatformManipulation" =>
            {
                PageAction::SuspendAuthor {}
            }
            ("liveness_check", EntityType::User) => PageAction::LivenessCheck {},
            ("label", EntityType::User) if label_ok => PageAction::SpamHighRecall {
                lifetime: lifetime()?,
            },
            ("post_label", EntityType::Post) if label_ok && lifetime()? != Lifetime::Days7 => {
                PageAction::PostSpamHighRecall {
                    lifetime: lifetime()?,
                }
            }
            _ => return None,
        });
    }
    (!out.is_empty()).then_some(out)
}

const SAFETY_REASONS: &[&str] = &[
    "user_in_allowlist",
    "post_in_allowlist",
    "user_not_found",
    "very_high_follower_count",
    "high_follower_count",
    "pagerank_skipped",
];

async fn move_to_page(
    user_rules: &str,
    post_rules: &str,
) -> BTreeMap<String, (usize, Vec<String>)> {
    let all = scenarios(user_rules, post_rules);
    let mut base: HashMap<String, Trace> = HashMap::new();
    let mut beds: HashMap<(bool, Option<u64>), TestBed> = HashMap::new();
    for sc in &all {
        let key = (sc.dry_run, sc.cap);
        if let std::collections::hash_map::Entry::Vacant(slot) = beds.entry(key) {
            slot.insert(
                TestBed::new(BedConfig {
                    user_rules,
                    post_rules,
                    config: on_config(sc.dry_run, sc.cap),
                    off_switches: vec![switched_off()],
                    control_records: vec![],
                })
                .await,
            );
        }
        let b = &beds[&key];
        let (et, id) = Facts::resolve_target(&sc.score);
        b.dedup.slots.lock().unwrap().clear();
        if let Some(c) = sc.held_class {
            b.dedup.hold(et, id, c);
        }
        base.insert(sc.id.clone(), b.run(&sc.score).await);
    }
    let key_of = |id: &str| id.split('|').take(3).collect::<Vec<_>>().join("|");
    let mut choice: HashMap<String, serde_json::Value> = HashMap::new();
    for sc in &all {
        if sc.id.contains("|Clean|dry=0|held=-")
            && let Some(t) = base.get(&sc.id)
            && t.status == "success"
            && let Some(specs) = t.outcomes[0].info.get("action_specs")
        {
            choice.insert(key_of(&sc.id), serde_json::from_str(specs).unwrap());
        }
    }
    let mut counts: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    let bump = |counts: &mut BTreeMap<String, (usize, Vec<String>)>, k: String, id: &str| {
        let e = counts.entry(k).or_default();
        e.0 += 1;
        if e.1.len() < 3 {
            e.1.push(id.to_owned());
        }
    };
    let mut unexplained = Vec::new();
    for sc in &all {
        let parts: Vec<&str> = sc.id.split('|').collect();
        let (entity, head, label) = (parts[0], parts[1], parts[2]);
        let entity_type = if entity == "post" {
            EntityType::Post
        } else {
            EntityType::User
        };
        let Some(specs) = choice.get(&key_of(&sc.id)) else {
            bump(
                &mut counts,
                "not moved: the rules do not act on this label".into(),
                &sc.id,
            );
            continue;
        };
        let Some(actions) = as_page_actions(specs, entity_type) else {
            bump(
                &mut counts,
                "not moved: the rules' actions are outside the page menu".into(),
                &sc.id,
            );
            continue;
        };
        let band = (!label.is_empty() && !label.starts_with("cap_full")).then_some(label);
        let rec = record(head, None, band, ControlMode::Live, actions);
        let b = &beds[&(sc.dry_run, sc.cap)];
        b.set_records(&[rec]).await;
        let (et, id) = Facts::resolve_target(&sc.score);
        b.dedup.slots.lock().unwrap().clear();
        if let Some(c) = sc.held_class {
            b.dedup.hold(et, id, c);
        }
        let moved = b.run(&sc.score).await;
        let was = &base[&sc.id];
        let same_ais = ais_names(was) == ais_names(&moved);
        let same_effect = was.status == moved.status && same_ais && was.tickets == moved.tickets;
        let category = if same_effect && was.dedup_calls == moved.dedup_calls {
            "identical".to_owned()
        } else if same_effect {
            "same status, actions and tickets; only the dedup calls differ".into()
        } else if entity == "post"
            && was.status == "high_follower_count"
            && moved.status == "very_high_follower_count"
            && moved.ais.is_empty()
            && moved.tickets == 0
        {
            "post author at the page's follower ceiling (the rules skip from a lower count)".into()
        } else if SAFETY_REASONS.contains(&was.status.as_str()) {
            if moved.status != was.status || !moved.ais.is_empty() || moved.tickets != 0 {
                unexplained.push(format!(
                    "{}: safety skip {} became {}",
                    sc.id, was.status, moved.status
                ));
                "UNEXPLAINED".into()
            } else {
                "same safety skip; only the dedup class or claim differs".into()
            }
        } else if (moved.status == ALREADY_SUSPENDED || moved.status == "already_deactivated")
            && moved.ais.is_empty()
        {
            "page decisions skip suspended or deactivated accounts (the rules acted or skipped differently)".into()
        } else if (was.status == "success" || was.status == "dry_run")
            && moved.status == "already_spam_high_recall_labeled"
        {
            "label-only page set skips an account that already has the label (the rule re-labels)"
                .into()
        } else if was.status == "dedup_skipped" && moved.status != "dedup_skipped" {
            "dedup class from the page's actions outranks the held entry (auto)".into()
        } else if was.status == "already_deactivated" {
            "deactivated account: `already_deactivated` is a routing rule, not a safety check"
                .into()
        } else if was.status == "already_spam_high_recall_labeled" {
            "already-labelled skip of a label-only rule vs the page's set".into()
        } else {
            unexplained.push(format!(
                "{}: {} {:?} cap={} {:?} → {} {:?} cap={} {:?}",
                sc.id,
                was.status,
                ais_names(was),
                was.tickets,
                was.dedup_calls,
                moved.status,
                ais_names(&moved),
                moved.tickets,
                moved.dedup_calls
            ));
            "UNEXPLAINED".into()
        };
        bump(&mut counts, category, &sc.id);
    }
    assert!(
        unexplained.is_empty(),
        "differences outside the approved list:\n{}",
        unexplained.join("\n")
    );
    counts
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn moving_every_head_to_the_page_with_its_current_actions() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let counts = move_to_page(BAKED_USER_RULES, BAKED_POST_RULES).await;
    let summary: String = counts
        .iter()
        .map(|(k, (n, ex))| format!("{n:>5}  {k}\n         e.g. {}\n", ex.join("  ")))
        .collect();
    eprintln!("move-to-page replay (baked rules):\n{summary}");
    if let Ok(path) = std::env::var("ACTION_CONTROL_MOVE_REPORT") {
        std::fs::write(path, &summary).unwrap();
    }
    let same = |k: &str| counts.get(k).map_or(0, |c| c.0);
    assert!(
        same("identical") + same("same status, actions and tickets; only the dedup calls differ")
            > 100,
        "{counts:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn moving_every_head_to_the_page_live_rules() {
    let Ok(dir) = std::env::var("ACTION_CONTROL_REPLAY_RULES_DIR") else {
        return;
    };
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let user = std::fs::read_to_string(format!("{dir}/rule_user.yaml")).unwrap();
    let post = std::fs::read_to_string(format!("{dir}/rule_post.yaml")).unwrap();
    let counts = move_to_page(&user, &post).await;
    let summary: String = counts
        .iter()
        .map(|(k, (n, ex))| format!("{n:>5}  {k}\n         e.g. {}\n", ex.join("  ")))
        .collect();
    eprintln!("move-to-page replay (live rules):\n{summary}");
    if let Ok(path) = std::env::var("ACTION_CONTROL_MOVE_REPORT_LIVE") {
        std::fs::write(path, &summary).unwrap();
    }
}

#[tokio::test]
async fn a_recording_decision_never_writes_or_releases_the_slot_on_retry() {
    let decision = DecisionContext {
        control: action_control::Resolved {
            mode: ControlMode::Recording,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(!decision.claimed());
    let producer = DecisionContext {
        control: action_control::Resolved {
            producer_dry: true,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(!producer.claimed());
    let global = DecisionContext {
        global_dry_run: true,
        ..Default::default()
    };
    assert!(!global.claimed());
    assert!(DecisionContext::default().claimed());
}

#[tokio::test]
async fn a_decision_pinned_to_global_dry_run_never_acts_after_the_flag_flips() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let b = bed(vec![], false).await;
    let ctx = &b.ctx;
    let score = bb1(&[&label], Account::Clean);
    let pinned = DecisionContext {
        global_dry_run: true,
        ..Default::default()
    };
    let tickets = b.tickets.load(std::sync::atomic::Ordering::Relaxed);
    let out = run_enforcement(ctx, &score, "t", &pinned).await.unwrap();
    assert_eq!(out.status, "dry_run");
    assert!(out.dry_run);
    assert_eq!(out.info["dry_run_source"], "global");
    assert_eq!(out.info["would_dedup_skip"], "false");
    assert!(b.ais.take().is_empty(), "no AIS call");
    assert_eq!(
        b.tickets.load(std::sync::atomic::Ordering::Relaxed),
        tickets,
        "no daily-cap ticket"
    );
    let live = run_enforcement(ctx, &score, "t", &DecisionContext::default())
        .await
        .unwrap();
    assert_eq!(
        live.status, "success",
        "the same message acts when not pinned"
    );
}

const RULES_FACT_GATED_SUSPEND: &str = r#"
for_entity: user
rules:
  - id: user_not_found
    when: "!user.present"
    then: { kind: skip, reason: user_not_found }
  - id: act_suspend_if_labeled
    when: '"fact_gated_suspend" in score.labels && "SpamHighRecall" in user.labels'
    then: { kind: act_suspend_user, perm: false, policy: "PlatformManipulation" }
  - id: act_liveness_fallback
    when: '"fact_gated_suspend" in score.labels'
    then: { kind: act_spam_liveness_check }
  - id: terminal
    when: "true"
    then: { kind: skip, reason: no_match }
"#;

async fn fact_gated_bed() -> TestBed {
    bed_with(vec![], base_config(false, None), RULES_FACT_GATED_SUSPEND).await
}

fn held(b: &TestBed, account: Account) -> Option<(u8, bool, Option<u64>)> {
    b.dedup
        .slots
        .lock()
        .unwrap()
        .get(&(EntityType::User.as_str().to_owned(), account.user_id()))
        .copied()
}

#[tokio::test]
async fn a_weaker_fallback_chosen_after_the_fetch_is_deduped_against_an_equal_held_entry() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = fact_gated_bed().await;
    b.dedup.slots.lock().unwrap().clear();
    b.dedup.hold(
        EntityType::User,
        Account::Clean.user_id(),
        dedup_cache::CLASS_CHALLENGE,
    );
    let t = b.run(&bb1(&["fact_gated_suspend"], Account::Clean)).await;
    assert_eq!(t.status, "dedup_skipped");
    assert_eq!(
        t.dedup_calls,
        vec![
            "try_insert(class=3)=bypass",
            "update_skip(class=0,own_claim=true)"
        ]
    );
    assert!(t.ais.is_empty(), "no AIS call: {:?}", t.ais);
    assert_eq!(t.tickets, 0, "no daily-cap ticket");
    assert_eq!(info(&t)["matched_rule_id"], "act_liveness_fallback");
    assert_eq!(info(&t)["dedup_held_class"], "2");
    assert_eq!(
        held(&b, Account::Clean),
        Some((dedup_cache::CLASS_CHALLENGE, false, None)),
        "the held liveness check is put back"
    );
}

#[tokio::test]
async fn a_fact_gated_suspend_that_fires_acts_over_a_held_liveness_check() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = fact_gated_bed().await;
    b.dedup.slots.lock().unwrap().clear();
    b.dedup.hold(
        EntityType::User,
        Account::HasSpamHighRecall.user_id(),
        dedup_cache::CLASS_CHALLENGE,
    );
    let t = b
        .run(&bb1(&["fact_gated_suspend"], Account::HasSpamHighRecall))
        .await;
    assert_eq!(t.status, "success");
    assert_eq!(ais_names(&t), vec!["suspendUser"]);
    assert_eq!(t.tickets, 1);
    assert_eq!(
        t.dedup_calls,
        vec![
            "try_insert(class=3)=bypass",
            "update(class=3,own_claim=true)"
        ]
    );
    assert_eq!(
        held(&b, Account::HasSpamHighRecall),
        Some((dedup_cache::CLASS_SUSPEND, false, None))
    );

    let t = b
        .run_fresh(&bb1(&["fact_gated_suspend"], Account::Clean))
        .await;
    assert_eq!(t.status, "success");
    assert_eq!(ais_names(&t), vec!["bounceViaTemplate"]);
    assert_eq!(t.tickets, 1);
}


const REPLY_HEAD: &str = "ReplyHead";
const REPLY_MODEL: &str = "reply_model_v1";
const REPLY_LABEL: &str = "reply_head_label";

fn listing(b: &TestBed, topic: &str, labels: &[&str], negative: &[&str]) -> ScoreResultProcessor {
    ScoreResultProcessor {
        topic: topic.into(),
        labels: labels.iter().map(|l| (*l).to_owned()).collect(),
        negative_labels: negative.iter().map(|l| (*l).to_owned()).collect(),
        ..b.processor()
    }
}

fn user_topic(b: &TestBed) -> ScoreResultProcessor {
    listing(
        b,
        "abuse.embeddings.user_decisions",
        &["anchor_campaign_suspend"],
        &[],
    )
}

fn post_topic(b: &TestBed) -> ScoreResultProcessor {
    listing(
        b,
        "abuse.embeddings.post_decisions",
        &["gibberish_post"],
        &[],
    )
}

fn reply_bot(entity: EntityType) -> ScoreResult {
    message(
        entity,
        REPLY_HEAD,
        REPLY_MODEL,
        &[REPLY_LABEL],
        Account::Clean,
    )
}

fn of_entity(entity: ControlEntity, r: ActionControlRecord) -> ActionControlRecord {
    ActionControlRecord {
        entity_type: entity,
        ..r
    }
}

fn on_rows(entity: ControlEntity) -> ActionControlRecord {
    admitting(of_entity(
        entity,
        record(REPLY_HEAD, None, None, ControlMode::Rows, vec![]),
    ))
}

fn admitting(r: ActionControlRecord) -> ActionControlRecord {
    ActionControlRecord {
        admit_unlisted_labels: true,
        ..r
    }
}

fn assert_dropped(t: &Trace, status: &str) {
    assert_eq!(t.status, status);
    assert!(t.outcomes.is_empty() && t.ais.is_empty() && t.dedup_calls.is_empty());
}

fn assert_logged_before_any_fetch(t: &Trace) {
    assert_eq!(t.status, action_control::STATUS_NO_ACTIONS);
    assert!(t.ais.is_empty() && t.tickets == 0);
    assert_eq!(t.dedup_calls, vec!["peek(class=0)=None"]);
    assert_eq!(t.outcomes.len(), 1);
    assert!(t.outcomes[0].dry_run);
    let i = info(t);
    assert_eq!(i["logged_because"], "no_selection");
    assert_eq!(i["level_row"], "missing");
    assert_eq!(i["dry_run_source"], "page");
    assert_eq!(i["matched_rule_id"], action_control::PAGE_RULE_ID);
    assert!(!i.keys().any(|k| k.starts_with("rules_would")), "{i:?}");
}

#[tokio::test]
async fn an_unlisted_head_without_a_deciding_record_is_dropped() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let rules = || admitting(record(REPLY_HEAD, None, None, ControlMode::Rules, vec![]));
    let rows_without_opt_in =
        |e| of_entity(e, record(REPLY_HEAD, None, None, ControlMode::Rows, vec![]));
    for records in [
        vec![
            of_entity(ControlEntity::User, rules()),
            of_entity(ControlEntity::Post, rules()),
        ],
        vec![
            rows_without_opt_in(ControlEntity::User),
            rows_without_opt_in(ControlEntity::Post),
        ],
    ] {
        let b = bed(records, false).await;
        let t = b
            .run_admitted(&user_topic(&b), &reply_bot(EntityType::User))
            .await;
        assert_dropped(&t, "no_match");
        let t = b
            .run_admitted(&post_topic(&b), &reply_bot(EntityType::Post))
            .await;
        assert_dropped(&t, "no_match");
        assert_eq!(b.fetches().await, 0);
    }
}

#[tokio::test]
async fn a_head_no_record_names_logs_only_on_both_levels() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![], false).await;
    let listed = listing(&b, "abuse.embeddings.user_decisions", &[REPLY_LABEL], &[]);
    for (processor, entity) in [
        (post_topic(&b), EntityType::Post),
        (user_topic(&b), EntityType::User),
        (listed, EntityType::User),
    ] {
        let t = b.run_admitted(&processor, &reply_bot(entity)).await;
        assert_logged_before_any_fetch(&t);
        assert_eq!(t.outcomes[0].head, REPLY_HEAD);
        assert_eq!(info(&t)["action_control_mode"], "recording");
    }
    let negative = listing(&b, "abuse.embeddings.post_decisions", &[], &[REPLY_LABEL]);
    let t = b
        .run_admitted(&negative, &reply_bot(EntityType::Post))
        .await;
    assert_dropped(&t, "negative_label");
    assert_eq!(b.fetches().await, 0);
}

#[tokio::test]
async fn a_first_page_save_keeps_an_unknown_heads_decisions_admitted() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let rules = crate::rules::RulesCache::new();
    let safety = action_control::SafetyChecksByEntity {
        user: safety_checks::status(EntityType::User, &rules.resolve(EntityType::User, None)),
        post: safety_checks::status(EntityType::Post, &rules.resolve(EntityType::Post, None)),
    };
    for (admit, admitted) in [(None, true), (Some(false), false)] {
        let b = bed(vec![], false).await;
        let t = b
            .run_admitted(&user_topic(&b), &reply_bot(EntityType::User))
            .await;
        assert_logged_before_any_fetch(&t);
        let control = b.ctx.dynamic_config.action_control_config();
        let out = b
            .ctx
            .action_control
            .set(
                action_control::ActionControlPutRequest {
                    entity_type: ControlEntity::User,
                    head: REPLY_HEAD.into(),
                    classifier: None,
                    band: None,
                    mode: ControlMode::Rows,
                    actions: vec![],
                    reason: "log only from the page".into(),
                    expected_version: None,
                    confirm: None,
                    updated_by: None,
                    confidence: None,
                    evidence: None,
                    evidence_override: None,
                    rows: Some(action_control::RowsRequest::default()),
                    admit_unlisted_labels: admit,
                    rollout_bp: None,
                    follower_threshold: None,
                    skip_high_credibility: None,
                },
                "operator@example.com",
                "page",
                now_millis(),
                &control,
                &safety,
            )
            .await
            .unwrap();
        assert!(out.harsher_because.is_empty(), "{:?}", out.harsher_because);
        assert_eq!(out.view.record.admit_unlisted_labels, admitted);
        let t = b
            .run_admitted(&user_topic(&b), &reply_bot(EntityType::User))
            .await;
        if admitted {
            assert_logged_before_any_fetch(&t);
            assert!(
                info(&t).contains_key("action_control_lkey"),
                "decided by the saved record"
            );
        } else {
            assert_dropped(&t, "no_match");
        }
    }
}

#[tokio::test]
async fn one_head_on_both_levels_logs_only_where_its_record_is_saved() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let user_rules = of_entity(
        ControlEntity::User,
        record(REPLY_HEAD, None, None, ControlMode::Rules, vec![]),
    );
    let b = bed(vec![on_rows(ControlEntity::Post), user_rules], false).await;
    let negative = listing(&b, "abuse.embeddings.post_decisions", &[], &[REPLY_LABEL]);
    for processor in [post_topic(&b), negative] {
        let t = b
            .run_admitted(&processor, &reply_bot(EntityType::Post))
            .await;
        assert_logged_before_any_fetch(&t);
        assert_eq!(t.outcomes[0].head, REPLY_HEAD);
    }
    let t = b
        .run_admitted(&user_topic(&b), &reply_bot(EntityType::User))
        .await;
    assert_dropped(&t, "no_match");
    assert_eq!(b.fetches().await, 0);

    b.set_records(&[on_rows(ControlEntity::Post), on_rows(ControlEntity::User)])
        .await;
    for (processor, entity) in [
        (post_topic(&b), EntityType::Post),
        (user_topic(&b), EntityType::User),
    ] {
        let t = b.run_admitted(&processor, &reply_bot(entity)).await;
        assert_logged_before_any_fetch(&t);
    }
    assert_eq!(b.fetches().await, 0);

    b.ctx.action_control.spawn_seen_writer().unwrap();
    let seen = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let seen = b.ctx.action_control.scan_seen().await.unwrap();
            if seen.iter().filter(|s| s.head == REPLY_HEAD).count() >= 2 {
                return seen;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the head seen on both levels");
    let mut levels: Vec<ControlEntity> = seen
        .iter()
        .filter(|s| s.head == REPLY_HEAD)
        .inspect(|s| assert!(!s.factory && s.rule_id.is_empty() && s.band == REPLY_LABEL))
        .map(|s| s.entity_type)
        .collect();
    levels.dedup();
    assert!(levels.contains(&ControlEntity::Post) && levels.contains(&ControlEntity::User));
}

#[tokio::test]
async fn the_saved_page_record_logs_only() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut records = vec![on_rows(ControlEntity::Post)];
    for row in [
        LevelRow::Low,
        LevelRow::Mid,
        LevelRow::High,
        LevelRow::Missing,
    ] {
        records.push(ActionControlRecord {
            confidence: Some(row),
            ..of_entity(
                ControlEntity::Post,
                record(REPLY_HEAD, None, None, ControlMode::Recording, vec![]),
            )
        });
    }
    let b = bed(records, false).await;
    let t = b
        .run_admitted(&post_topic(&b), &reply_bot(EntityType::Post))
        .await;
    assert_logged_before_any_fetch(&t);
    assert_eq!(b.fetches().await, 0);
}

#[tokio::test]
async fn an_unlisted_head_follows_its_record_and_factory_heads_are_unchanged() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(
        vec![
            admitting(live(REPLY_HEAD, None, vec![SHR7])),
            of_entity(
                ControlEntity::Post,
                record(REPLY_HEAD, None, None, ControlMode::Rules, vec![]),
            ),
            admitting(record(FACTORY_HEAD, None, None, ControlMode::Rows, vec![])),
        ],
        false,
    )
    .await;
    let t = b
        .run_admitted(&user_topic(&b), &reply_bot(EntityType::User))
        .await;
    assert_eq!(t.status, "success");
    assert_eq!(ais_names(&t), vec!["addLabelsV2"]);
    assert!(!info(&t).keys().any(|k| k.starts_with("rules_would")));
    let t = b
        .run_admitted(&post_topic(&b), &reply_bot(EntityType::Post))
        .await;
    assert_dropped(&t, "no_match");

    b.set_records(&[admitting(record(
        REPLY_HEAD,
        None,
        None,
        ControlMode::Recording,
        vec![],
    ))])
    .await;
    let t = b
        .run_admitted(&user_topic(&b), &reply_bot(EntityType::User))
        .await;
    assert_eq!(t.status, action_control::STATUS_NO_ACTIONS);
    assert!(t.ais.is_empty() && t.outcomes[0].dry_run);

    let t = b
        .run_admitted(&user_topic(&b), &factory_message(Account::Clean))
        .await;
    assert_dropped(&t, "no_match");
}

#[tokio::test]
async fn a_listed_head_with_a_record_is_unchanged() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    for records in [vec![], vec![live(BB1.0, None, vec![SHR7])]] {
        let b = bed(records, false).await;
        let msg = bb1(&[&label], Account::Clean);
        let via_admission = b.run_admitted(&user_topic(&b), &msg).await;
        let direct = b.run_fresh(&msg).await;
        assert_eq!(via_admission.status, direct.status);
        assert_eq!(ais_names(&via_admission), ais_names(&direct));
        assert_eq!(info(&via_admission), info(&direct));
    }
}

#[tokio::test]
async fn a_head_on_rows_without_the_opt_in_is_unchanged() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut records = vec![record(BB1.0, None, None, ControlMode::Rows, vec![])];
    for (row, mode) in [
        (LevelRow::Low, ControlMode::Rules),
        (LevelRow::Mid, ControlMode::Recording),
        (LevelRow::High, ControlMode::Rules),
        (LevelRow::Missing, ControlMode::Rules),
    ] {
        records.push(ActionControlRecord {
            confidence: Some(row),
            ..record(BB1.0, None, None, mode, vec![])
        });
    }
    let b = bed(records, false).await;
    for level in [
        abuse_proto::Confidence::High,
        abuse_proto::Confidence::Mid,
        abuse_proto::Confidence::Low,
        abuse_proto::Confidence::Unspecified,
    ] {
        let mut msg = bb1(&["not_listed"], Account::Clean);
        msg.confidence = level as i32;
        let t = b.run_admitted(&user_topic(&b), &msg).await;
        assert_dropped(&t, "no_match");
    }
    assert_eq!(b.fetches().await, 0);
}

const HIDE_REPLY: PageAction = PageAction::HideReply {
    lifetime: Lifetime::Permanent,
};

fn hiding(rollout_bp: Option<u32>) -> Vec<ActionControlRecord> {
    vec![
        on_rows(ControlEntity::Post),
        ActionControlRecord {
            confidence: Some(LevelRow::Missing),
            rollout_bp,
            ..of_entity(
                ControlEntity::Post,
                live(REPLY_HEAD, None, vec![HIDE_REPLY]),
            )
        },
    ]
}

fn reply_by(account: Account) -> ScoreResult {
    message(
        EntityType::Post,
        REPLY_HEAD,
        REPLY_MODEL,
        &[REPLY_LABEL],
        account,
    )
}

fn assert_hidden(t: &Trace) {
    assert_eq!(t.status, "success", "{:?}", t.outcomes);
    assert!(!t.ais.is_empty());
    assert!(!t.outcomes[0].dry_run);
}

fn assert_logged_with_hide(t: &Trace, because: &str) {
    assert_eq!(t.status, "dry_run", "{:?}", t.outcomes);
    assert!(t.ais.is_empty() && t.tickets == 0);
    assert!(t.outcomes[0].dry_run);
    let i = info(t);
    assert_eq!(i["logged_because"], because);
    assert_eq!(i["action_control_forced_recording"], because);
    assert_eq!(i["action_control_mode"], "recording");
    assert_eq!(i["dry_run_source"], "page");
    assert_eq!(i["level_row"], "missing");
    assert!(
        i["action_specs"].contains("RiskyHighVizReply"),
        "the row's action is recorded: {i:?}"
    );
}

#[tokio::test]
async fn a_row_rollout_hides_replies_inside_it_and_logs_the_rest() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let place = |a: Account| action_control::rollout_bucket("", REPLY_HEAD, a.user_id());
    let (mut low, mut high) = (Account::Clean, Account::HasPhoenixSkip);
    assert_ne!(place(low), place(high));
    if place(low) > place(high) {
        std::mem::swap(&mut low, &mut high);
    }
    let b = bed(hiding(Some(place(high))), false).await;
    let t = b.run_admitted(&post_topic(&b), &reply_by(low)).await;
    assert_hidden(&t);
    assert_eq!(info(&t)["rollout_bp"], place(high).to_string());
    assert_eq!(info(&t)["rollout_bucket"], place(low).to_string());
    let t = b.run_admitted(&post_topic(&b), &reply_by(high)).await;
    assert_logged_with_hide(&t, "outside_rollout");
    assert_eq!(info(&t)["rollout_bucket"], place(high).to_string());
    assert_eq!(t.dedup_calls.len(), 1, "{:?}", t.dedup_calls);
    assert!(t.dedup_calls[0].starts_with("peek"), "{:?}", t.dedup_calls);

    b.set_records(&hiding(Some(0))).await;
    for account in [low, high] {
        let t = b.run_admitted(&post_topic(&b), &reply_by(account)).await;
        assert_logged_with_hide(&t, "outside_rollout");
    }
    b.set_records(&hiding(None)).await;
    for account in [low, high] {
        let t = b.run_admitted(&post_topic(&b), &reply_by(account)).await;
        assert_hidden(&t);
        assert!(!info(&t).contains_key("rollout_bp"));
    }
}

#[tokio::test]
async fn a_row_this_image_cannot_read_logs_only() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![on_rows(ControlEntity::Post)], false).await;
    let mut newer = serde_json::to_value(&hiding(Some(10_000))[1]).unwrap();
    newer["v"] = (action_control::ACTION_CONTROL_RECORD_VERSION + 1).into();
    newer["rollout_next"] = 1.into();
    b.control_store.rows.lock().unwrap().insert(
        (
            action_control::MH_PKEY.into(),
            format!("post/l/{REPLY_HEAD}/missing"),
        ),
        serde_json::to_vec(&newer).unwrap(),
    );
    b.ctx.action_control.refresh().await.unwrap();
    let t = b
        .run_admitted(&post_topic(&b), &reply_by(Account::Clean))
        .await;
    assert_eq!(t.status, action_control::STATUS_NO_ACTIONS);
    assert!(t.ais.is_empty() && t.tickets == 0 && t.outcomes[0].dry_run);
    assert_eq!(info(&t)["logged_because"], "unreadable");
}

fn with_skips(
    r: ActionControlRecord,
    follower_threshold: Option<u32>,
    skip_high_credibility: Option<bool>,
) -> ActionControlRecord {
    ActionControlRecord {
        follower_threshold,
        skip_high_credibility,
        ..r
    }
}

#[tokio::test]
async fn an_account_head_skips_by_its_own_follower_threshold() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let at = Account::BetweenLimits.followers() as u32;
    let label = suspend_label();
    let head =
        |t: Option<u32>, c: Option<bool>| vec![with_skips(live(BB1.0, None, vec![SUSPEND]), t, c)];
    let b = bed(head(None, None), false).await;
    let run = |account: Account| bb1(&[&label], account);
    let cases: [(Option<u32>, Option<bool>, [&str; 4]); 6] = [
        (
            None,
            None,
            [
                "success",
                "high_follower_count",
                "very_high_follower_count",
                "pagerank_skipped",
            ],
        ),
        (
            Some(0),
            None,
            [
                "success",
                "success",
                "very_high_follower_count",
                "pagerank_skipped",
            ],
        ),
        (
            Some(at),
            None,
            [
                "success",
                "high_follower_count",
                "very_high_follower_count",
                "pagerank_skipped",
            ],
        ),
        (
            Some(at + 1),
            None,
            [
                "success",
                "success",
                "very_high_follower_count",
                "pagerank_skipped",
            ],
        ),
        (
            None,
            Some(false),
            [
                "success",
                "high_follower_count",
                "very_high_follower_count",
                "success",
            ],
        ),
        (
            Some(0),
            Some(false),
            ["success", "success", "very_high_follower_count", "success"],
        ),
    ];
    for (threshold, cred, want) in cases {
        b.set_records(&head(threshold, cred)).await;
        let accounts = [
            Account::Clean,
            Account::BetweenLimits,
            Account::AboveTopLimit,
            Account::HighCred,
        ];
        for (account, want) in accounts.into_iter().zip(want) {
            let t = b.run_fresh(&run(account)).await;
            assert_eq!(t.status, want, "{threshold:?} {cred:?} {account:?}");
            assert_eq!(
                t.ais.is_empty(),
                want != "success",
                "{threshold:?} {account:?}"
            );
            let i = info(&t);
            assert_eq!(
                i.get("action_control_follower_threshold"),
                threshold.map(|n| n.to_string()).as_ref(),
            );
            assert_eq!(
                i.get("action_control_skip_high_credibility"),
                cred.map(|c| c.to_string()).as_ref(),
            );
        }
        let t = b.run_fresh(&run(Account::NotFound)).await;
        assert_eq!(t.status, "user_not_found");
    }

    b.set_records(&head(Some(at), None)).await;
    let mut waived = run(Account::BetweenLimits);
    waived.skip_author_credibility_prechecks = true;
    assert_eq!(b.run_fresh(&waived).await.status, "high_follower_count");
}

#[tokio::test]
async fn a_post_head_skips_authors_by_its_own_threshold_and_keeps_the_waiver() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let at = Account::BetweenLimits.followers() as u32;
    let head = "SpamEmbeddingPtosDistilled";
    let label = crate::replay_tests::rule_labels(BAKED_POST_RULES)
        .into_iter()
        .next()
        .expect("a post routing label");
    let hide = vec![PageAction::PostSpamHighRecall {
        lifetime: Lifetime::Days30,
    }];
    let records = |t: Option<u32>, c: Option<bool>| {
        vec![with_skips(
            of_entity(ControlEntity::Post, live(head, None, hide.clone())),
            t,
            c,
        )]
    };
    let b = bed(records(None, None), false).await;
    let post = |account: Account, waived: bool| {
        let mut s = message(EntityType::Post, head, "clf", &[&label], account);
        s.skip_author_credibility_prechecks = waived;
        s
    };
    let cases = [
        (
            None,
            None,
            false,
            [
                "high_follower_count",
                "very_high_follower_count",
                "pagerank_skipped",
            ],
        ),
        (
            None,
            None,
            true,
            ["success", "very_high_follower_count", "success"],
        ),
        (
            Some(0),
            None,
            false,
            ["success", "very_high_follower_count", "pagerank_skipped"],
        ),
        (
            Some(at),
            None,
            false,
            [
                "high_follower_count",
                "very_high_follower_count",
                "pagerank_skipped",
            ],
        ),
        (
            Some(at),
            None,
            true,
            ["success", "very_high_follower_count", "success"],
        ),
        (
            None,
            Some(false),
            false,
            ["high_follower_count", "very_high_follower_count", "success"],
        ),
        (
            Some(0),
            Some(true),
            true,
            ["success", "very_high_follower_count", "success"],
        ),
    ];
    for (threshold, cred, waived, want) in cases {
        b.set_records(&records(threshold, cred)).await;
        let accounts = [
            Account::BetweenLimits,
            Account::AboveTopLimit,
            Account::HighCred,
        ];
        for (account, want) in accounts.into_iter().zip(want) {
            let t = b.run_fresh(&post(account, waived)).await;
            assert_eq!(
                t.status, want,
                "{threshold:?} {cred:?} waived={waived} {account:?}"
            );
            assert_eq!(t.ais.is_empty(), want != "success");
        }
        assert_eq!(
            b.run_fresh(&post(Account::Clean, waived)).await.status,
            "success"
        );
    }
}

#[tokio::test]
async fn every_head_takes_its_own_skips_alike() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    for (head, model) in HEADS.iter().copied().take(3) {
        let b = bed(
            vec![with_skips(
                live(head, None, vec![SUSPEND]),
                Some(0),
                Some(false),
            )],
            false,
        )
        .await;
        for account in [Account::BetweenLimits, Account::HighCred] {
            let t = b
                .run_fresh(&message(EntityType::User, head, model, &[&label], account))
                .await;
            assert_eq!(t.status, "success", "{head} {account:?}");
        }
        let t = b
            .run_fresh(&message(
                EntityType::User,
                head,
                model,
                &[&label],
                Account::AboveTopLimit,
            ))
            .await;
        assert_eq!(t.status, "very_high_follower_count", "{head}");
    }
}

#[tokio::test]
async fn a_factory_head_ignores_a_head_records_skips() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let rows = factory_row(LevelRow::Missing, ControlMode::Live, vec![SUSPEND]);
    let scope = with_skips(
        record(FACTORY_HEAD, None, None, ControlMode::Rows, vec![]),
        Some(0),
        Some(false),
    );
    let b = bed(vec![scope, rows], false).await;
    for (account, want) in [
        (Account::Clean, "success"),
        (Account::BetweenLimits, "high_follower_count"),
        (Account::HighCred, "pagerank_skipped"),
    ] {
        let t = b.run_fresh(&factory_message(account)).await;
        assert_eq!(t.status, want, "{account:?}");
        assert!(!info(&t).contains_key("action_control_follower_threshold"));
    }
}

fn as_written(mut r: ActionControlRecord) -> ActionControlRecord {
    r.v = r.schema_version();
    r
}

fn allowed_skips() -> Vec<(Option<u32>, Option<bool>)> {
    let view = safety_checks::follower_skip_view();
    let (today, ceiling) = (view.today_threshold as u32, view.ceiling as u32);
    vec![
        (Some(0), None),
        (Some(today), None),
        (Some(today * 5), None),
        (Some(ceiling - 1), Some(true)),
        (None, Some(false)),
        (Some(0), Some(false)),
    ]
}

fn assert_skipped_as_today(t: &Trace, today_reason: &str, what: &str) {
    assert!(
        t.ais.is_empty() && t.tickets == 0,
        "{what}: acted ({})",
        t.status
    );
    let recorded = t.outcomes.last().is_some_and(|o| o.dry_run);
    assert!(t.status == today_reason || recorded, "{what}: {}", t.status);
}

fn layouts(
    skips: (Option<u32>, Option<bool>),
) -> Vec<(
    &'static str,
    Vec<ActionControlRecord>,
    Vec<ActionControlRecord>,
)> {
    let (t, c) = skips;
    let missing_row = ActionControlRecord {
        confidence: Some(LevelRow::Missing),
        ..record(BB1.0, None, None, ControlMode::Live, vec![SUSPEND])
    };
    let classifier = record(BB1.0, Some(BB1.1), None, ControlMode::Live, vec![SUSPEND]);
    let head =
        |mode: ControlMode, actions: Vec<PageAction>| record(BB1.0, None, None, mode, actions);
    vec![
        (
            "live head",
            vec![as_written(head(ControlMode::Live, vec![SUSPEND]))],
            vec![as_written(with_skips(
                head(ControlMode::Live, vec![SUSPEND]),
                t,
                c,
            ))],
        ),
        (
            "live classifier under a rules head",
            vec![
                as_written(head(ControlMode::Rules, vec![])),
                as_written(classifier.clone()),
            ],
            vec![
                as_written(with_skips(head(ControlMode::Rules, vec![]), t, c)),
                as_written(classifier),
            ],
        ),
        (
            "confidence rows",
            vec![
                as_written(head(ControlMode::Rows, vec![])),
                as_written(missing_row.clone()),
            ],
            vec![
                as_written(with_skips(head(ControlMode::Rows, vec![]), t, c)),
                as_written(missing_row),
            ],
        ),
    ]
}

fn store_only(b: &TestBed, records: &[ActionControlRecord]) {
    b.control_store.rows.lock().unwrap().clear();
    for r in records {
        b.control_store.put_record(r);
    }
}

#[tokio::test]
async fn an_older_image_never_skips_fewer_accounts_than_today() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let older = crate::action_control::test_support::refresh_as_older_image;
    for skips in allowed_skips() {
        for (layout, before, after) in layouts(skips) {
            for read_before in [false, true] {
                let b = bed(if read_before { before.clone() } else { vec![] }, false).await;
                store_only(&b, &after);
                older(&b.ctx.action_control).await;
                for (account, today) in [
                    (Account::BetweenLimits, "high_follower_count"),
                    (Account::HighCred, "pagerank_skipped"),
                ] {
                    let t = b.run_fresh(&bb1(&[&label], account)).await;
                    let what =
                        format!("{layout}, {skips:?}, read before {read_before}, {account:?}");
                    assert_skipped_as_today(&t, today, &what);
                }
            }
        }
    }
}

#[tokio::test]
async fn this_image_acts_with_the_head_records_skips_in_every_layout() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    for (layout, _, after) in layouts((Some(0), Some(false))) {
        let b = bed(after, false).await;
        for account in [Account::BetweenLimits, Account::HighCred] {
            let t = b.run_fresh(&bb1(&[&label], account)).await;
            assert_eq!(t.status, "success", "{layout} {account:?}");
        }
    }
}

#[tokio::test]
async fn a_threshold_below_today_would_skip_fewer_accounts_on_an_older_image() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let label = suspend_label();
    let before = vec![as_written(live(BB1.0, None, vec![SUSPEND]))];
    let below = vec![as_written(with_skips(
        live(BB1.0, None, vec![SUSPEND]),
        Some(1),
        None,
    ))];
    let clean = bb1(&[&label], Account::Clean);
    let b = bed(below.clone(), false).await;
    assert_eq!(b.run_fresh(&clean).await.status, "high_follower_count");
    let b = bed(before, false).await;
    store_only(&b, &below);
    crate::action_control::test_support::refresh_as_older_image(&b.ctx.action_control).await;
    let t = b.run_fresh(&clean).await;
    assert_eq!(t.status, "success");
    assert!(!t.ais.is_empty());
}

#[tokio::test]
async fn the_follower_ceiling_stops_every_decision() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let post_head = "SpamEmbeddingPtosDistilled";
    let post_label = crate::replay_tests::rule_labels(BAKED_POST_RULES)
        .into_iter()
        .next()
        .expect("a post routing label");
    let user_label = suspend_label();
    let waived = |mut s: ScoreResult| {
        s.skip_author_credibility_prechecks = true;
        s
    };
    let post = |account| {
        waived(message(
            EntityType::Post,
            post_head,
            "clf",
            &[&post_label],
            account,
        ))
    };
    let user = |account| waived(bb1(&[&user_label], account));
    let hide = vec![PageAction::PostSpamHighRecall {
        lifetime: Lifetime::Days30,
    }];
    let page = vec![
        with_skips(
            of_entity(ControlEntity::Post, live(post_head, None, hide)),
            Some(0),
            None,
        ),
        with_skips(live(BB1.0, None, vec![SUSPEND]), Some(0), None),
    ];
    let rules = vec![of_entity(
        ControlEntity::Post,
        record(post_head, None, None, ControlMode::Rules, vec![]),
    )];
    for (path, records) in [("growthbook rules", rules), ("page", page)] {
        let b = bed(records, false).await;
        for (entity, msg) in [
            ("post", &post as &dyn Fn(Account) -> ScoreResult),
            ("account", &user),
        ] {
            let t = b.run_fresh(&msg(Account::AboveTopLimit)).await;
            assert_eq!(t.status, "very_high_follower_count", "{path} {entity}");
            assert!(t.ais.is_empty() && t.tickets == 0, "{path} {entity}");
            assert_eq!(info(&t)["matched_rule_id"], "very_high_follower_count");
            let below = b.run_fresh(&msg(Account::JustBelowTopLimit)).await;
            assert_ne!(below.status, "very_high_follower_count", "{path} {entity}");
            let acted = !(path == "growthbook rules" && entity == "account");
            assert_eq!(
                below.status == "success",
                acted,
                "{path} {entity}: {}",
                below.status
            );
        }
    }
}


const SUSPEND_PERMANENT_CSE: PageAction = PageAction::Suspend {
    permanent: true,
    policy: crate::action_control::SuspensionPolicy::Cse,
};

fn suspending(actions: Vec<PageAction>) -> Vec<ActionControlRecord> {
    vec![
        on_rows(ControlEntity::User),
        ActionControlRecord {
            confidence: Some(LevelRow::Missing),
            ..of_entity(ControlEntity::User, live(REPLY_HEAD, None, actions))
        },
    ]
}

fn suspend_body(t: &Trace) -> serde_json::Value {
    t.ais
        .iter()
        .find_map(|(_, r)| {
            let actions: Vec<&serde_json::Value> = match r.get("actions") {
                Some(serde_json::Value::Array(a)) => a.iter().collect(),
                _ => r.get("action").into_iter().collect(),
            };
            actions
                .into_iter()
                .find_map(|a| a.get("suspendUser").cloned())
        })
        .expect("a suspendUser call")
}

#[tokio::test]
async fn a_permanent_policy_suspend_reaches_ais_as_set_and_upgrades_a_suspended_account() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(suspending(vec![SUSPEND_PERMANENT_CSE]), false).await;
    for account in [Account::Clean, Account::Suspended] {
        let t = b
            .run_admitted(
                &user_topic(&b),
                &message(
                    EntityType::User,
                    REPLY_HEAD,
                    REPLY_MODEL,
                    &[REPLY_LABEL],
                    account,
                ),
            )
            .await;
        assert_eq!(t.status, "success", "{account:?}: {:?}", t.outcomes);
        let body = suspend_body(&t);
        assert_eq!(body["perm"], true, "{account:?}");
        assert_eq!(body["policy"], "Cse", "{account:?}");
        assert_eq!(
            info(&t)["action_specs"],
            r#"[{"kind":"suspend","permanent":true,"policy":"Cse"}]"#
        );
        assert!(
            t.dedup_calls.iter().any(|c| c.contains("class=4")),
            "{:?}",
            t.dedup_calls
        );
    }
    let t = b
        .run_admitted(
            &user_topic(&b),
            &message(
                EntityType::User,
                REPLY_HEAD,
                REPLY_MODEL,
                &[REPLY_LABEL],
                Account::Deactivated,
            ),
        )
        .await;
    assert_eq!(t.status, "already_deactivated");
    assert!(t.ais.is_empty());

    b.set_records(&suspending(vec![SUSPEND])).await;
    let t = b
        .run_admitted(
            &user_topic(&b),
            &message(
                EntityType::User,
                REPLY_HEAD,
                REPLY_MODEL,
                &[REPLY_LABEL],
                Account::Suspended,
            ),
        )
        .await;
    assert_eq!(t.status, ALREADY_SUSPENDED);
    assert!(t.ais.is_empty());
    let t = b
        .run_admitted(&user_topic(&b), &reply_bot(EntityType::User))
        .await;
    assert_eq!(t.status, "success");
    let body = suspend_body(&t);
    assert_eq!(
        (body["perm"].clone(), body["policy"].clone()),
        (
            serde_json::json!(false),
            serde_json::json!("PlatformManipulation")
        ),
        "a plain suspend is the temporary PlatformManipulation one, as before"
    );
    assert!(
        t.dedup_calls.iter().any(|c| c.contains("class=3")),
        "{:?}",
        t.dedup_calls
    );
}
