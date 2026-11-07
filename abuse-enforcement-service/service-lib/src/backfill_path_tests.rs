use std::collections::BTreeMap;
use std::time::Duration;

use xai_abuse_enforcement_types::action_control::LevelRow;
use xai_abuse_proto::enforcement::ScoreResult;

use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{ControlMode, ControlStore, Lifetime, PageAction};
use crate::backfill::test_support::put_running;
use crate::backfill::RunState;
use crate::backfill_claims::{ClaimKey, ClaimStore, ItemClaim, STALE_AFTER};
use crate::replay_tests::{
    message, on_config, scenarios, switched_off, Account, BedConfig, TestBed, Trace,
    BAKED_POST_RULES, BAKED_USER_RULES, HEADS, SWITCHED_OFF_HEAD,
};

const SUSPEND: PageAction = PageAction::SUSPEND;
const LIVENESS: PageAction = PageAction::LivenessCheck {};
const SHR7: PageAction = PageAction::SpamHighRecall {
    lifetime: Lifetime::Days7,
};
const BB1: (&str, &str) = HEADS[0];
const RUN: &str = "run-00000001";

#[derive(Clone)]
struct RunConfig {
    config: serde_json::Value,
    budget: u64,
}

fn run_config(dry_run: bool, budget: u64) -> RunConfig {
    let mut config = on_config(dry_run, None);
    config["overturn_hold_gate"] =
        serde_json::json!({"mode": "enforce", "on_probe_error": "allow"});
    RunConfig { config, budget }
}

async fn set_budget(
    store: &action_control::test_support::MemControlStore,
    bf: &backfill::Backfill,
    budget: u64,
) {
    let rec = backfill::BackfillSettingsRecord {
        v: backfill::BACKFILL_SETTINGS_RECORD_VERSION,
        max_user_decisions_per_day: budget,
        max_post_decisions_per_day: budget,
        max_rate_per_sec: 200,
        reason: "test".into(),
        updated_by: "operator@example.com".into(),
        updated_by_key: "page".into(),
        updated_at_ms: now_millis(),
        version: 1,
        write_nonce: 1,
        history: vec![],
    };
    store
        .put(
            backfill::SETTINGS_PKEY,
            backfill::SETTINGS_LKEY,
            serde_json::to_vec(&rec).unwrap(),
            0,
        )
        .await
        .unwrap();
    bf.refresh_settings().await.unwrap();
}

async fn pod_backfill(
    b: &TestBed,
    claims: Arc<dyn ClaimStore>,
    pod: &str,
) -> Arc<backfill::Backfill> {
    let bf = backfill::Backfill::new(b.backfill_store.clone(), Some(claims), pod.into());
    bf.refresh_settings().await.unwrap();
    Arc::new(bf)
}

async fn bed(run: RunConfig, records: Vec<action_control::ActionControlRecord>) -> TestBed {
    let b = TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config: run.config,
        off_switches: vec![switched_off()],
        control_records: records,
    })
    .await;
    set_budget(&b.backfill_store, &b.ctx.backfill, run.budget).await;
    b
}

fn with_run(mut s: ScoreResult, run_id: &str) -> ScoreResult {
    s.backfill_run_id = run_id.into();
    s.score_ts_ms = now_millis() - 60_000;
    s
}

fn bb1(account: Account) -> ScoreResult {
    with_run(message(EntityType::User, BB1.0, BB1.1, &[], account), RUN)
}

fn key(run_id: &str, entity: EntityType, id: i64) -> ClaimKey {
    ClaimKey {
        run_id: run_id.into(),
        entity,
        entity_id: id,
    }
}

fn finished(status: &str) -> ItemClaim {
    ItemClaim::Finished {
        status: status.into(),
    }
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

impl TestBed {
    fn run_tickets(&self) -> i64 {
        (self.claims.tickets(EntityType::User) + self.claims.tickets(EntityType::Post)) as i64
    }

    async fn run_backfill(&self, score: &ScoreResult) -> (Trace, i64) {
        let before = self.run_tickets();
        let t = self.run_on(&self.backfill_processor(), score).await;
        (t, self.run_tickets() - before)
    }

    fn clear_claims(&self) {
        self.claims.clear();
        self.backfill_store
            .rows
            .lock()
            .unwrap()
            .retain(|(p, _), _| p != backfill::HOLD_PKEY);
    }

    fn claim(&self, run_id: &str, entity: EntityType, id: i64) -> ItemClaim {
        self.claims.peek(&key(run_id, entity, id))
    }

    fn put_crashed_claim(&self, run_id: &str, entity: EntityType, id: i64) {
        self.claims.set_handed(
            key(run_id, entity, id),
            1,
            STALE_AFTER + Duration::from_secs(1),
        );
    }

    fn set_run_state(&self, run_id: &str, state: RunState) {
        let key = (backfill::RUN_PKEY.to_owned(), run_id.to_owned());
        let mut rows = self.backfill_store.rows.lock().unwrap();
        let mut rec: backfill::BackfillRunRecord =
            serde_json::from_slice(rows.get(&key).unwrap()).unwrap();
        rec.state = state;
        rows.insert(key, serde_json::to_vec(&rec).unwrap());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_id_on_a_live_topic_is_skipped_in_every_scenario() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 1_000), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let all = scenarios(BAKED_USER_RULES, BAKED_POST_RULES);
    assert!(all.len() > 400, "{}", all.len());
    for sc in all {
        let t = b.run(&with_run(sc.score.clone(), RUN)).await;
        assert_eq!(t.status, backfill::STATUS_ID_ON_LIVE_TOPIC, "{}", sc.id);
        assert!(t.dedup_calls.is_empty(), "{}", sc.id);
        assert_eq!(t.tickets, 0, "{}", sc.id);
        assert!(t.ais.is_empty(), "{}", sc.id);
        assert_eq!(t.outcomes.len(), 1, "{}", sc.id);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_message_without_a_run_on_the_backfill_topic_is_skipped_in_every_scenario() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 1_000), vec![]).await;
    for sc in scenarios(BAKED_USER_RULES, BAKED_POST_RULES) {
        let (t, run_tickets) = b.run_backfill(&sc.score).await;
        assert_eq!(t.status, backfill::STATUS_ID_MISSING, "{}", sc.id);
        assert!(t.dedup_calls.is_empty(), "{}", sc.id);
        assert_eq!((t.tickets, run_tickets), (0, 0), "{}", sc.id);
        assert!(t.ais.is_empty(), "{}", sc.id);
        assert_eq!(t.outcomes.len(), 1, "{}", sc.id);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_decision_takes_every_live_step_with_the_runs_actions() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for actions in [vec![SUSPEND], vec![SHR7, LIVENESS]] {
        let mut heads: Vec<&str> = HEADS.iter().map(|(h, _)| *h).collect();
        heads.push(SWITCHED_OFF_HEAD);
        heads.push("IsDelayedSuspendSpam");
        let mut compared = 0;
        let mut acted = 0;
        for dry_run in [false, true] {
            let user_records = heads
                .iter()
                .map(|h| record(h, None, None, ControlMode::Live, actions.clone()))
                .collect();
            let live = bed(run_config(dry_run, 1_000_000), user_records).await;
            let runs = bed(run_config(dry_run, 1_000_000), vec![]).await;
            for (i, h) in heads.iter().enumerate() {
                put_running(
                    &runs.backfill_store,
                    &format!("run-head-{i:04}"),
                    h,
                    actions.clone(),
                    vec![LevelRow::Missing],
                );
            }
            for sc in scenarios(BAKED_USER_RULES, BAKED_POST_RULES) {
                let (entity, entity_id) = Facts::resolve_target(&sc.score);
                if entity != EntityType::User || sc.dry_run != dry_run {
                    continue;
                }
                let head = crate::facts::head_label(&sc.score);
                let i = heads.iter().position(|h| *h == head).unwrap();
                let score = with_run(sc.score.clone(), &format!("run-head-{i:04}"));
                for b in [&live, &runs] {
                    b.dedup.slots.lock().unwrap().clear();
                    if let Some(c) = sc.held_class {
                        b.dedup.hold(entity, entity_id, c);
                    }
                }
                runs.clear_claims();
                let was = live.run(&sc.score).await;
                let (now, run_tickets) = runs.run_backfill(&score).await;
                if is_factory_message(&sc.score) {
                    assert_eq!(now.status, backfill::STATUS_BAD_MESSAGE, "{}", sc.id);
                    assert!(now.ais.is_empty() && run_tickets == 0, "{}", sc.id);
                    continue;
                }
                assert_eq!(now.status, was.status, "{}", sc.id);
                assert_eq!(ais_names(&now), ais_names(&was), "{}", sc.id);
                assert_eq!(now.dedup_calls, was.dedup_calls, "{}", sc.id);
                assert_eq!(now.tickets, 0, "{}: never a live-cap ticket", sc.id);
                assert_eq!(run_tickets, was.tickets, "{}", sc.id);
                let i = info(&now);
                assert_eq!(i["backfill_run_id"], score.backfill_run_id, "{}", sc.id);
                assert_eq!(i["backfill_run_by"], "operator@example.com", "{}", sc.id);
                assert_eq!(
                    i["matched_rule_id"],
                    info(&was)["matched_rule_id"],
                    "{}",
                    sc.id
                );
                let claim = runs.claim(&score.backfill_run_id, entity, entity_id);
                if dry_run {
                    assert_eq!(
                        claim,
                        ItemClaim::Free,
                        "{}: recording only writes no claim",
                        sc.id
                    );
                } else if !backfill::leaves_claim(&now.status) {
                    assert_eq!(claim, finished(&now.status), "{}", sc.id);
                }
                compared += 1;
                acted += usize::from(now.status == "success");
            }
        }
        assert!(
            compared > 100 && acted > 10,
            "{compared} compared, {acted} acted"
        );
    }
}

#[tokio::test]
async fn a_full_live_cap_does_not_stop_a_run() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut cfg = run_config(false, 10);
    cfg.config["max_user_enforcements_per_day"] = 0.into();
    let b = bed(cfg, vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let (t, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(t.status, "success");
    assert_eq!((t.tickets, run_tickets), (0, 1));
}

#[tokio::test]
async fn run_decisions_neither_count_toward_nor_trip_a_head_guard() {
    use crate::head_guard::test_support::{bucket_count, guard, with_counters};
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut b = bed(run_config(false, 10), vec![]).await;
    let (hg, _, buckets) = with_counters(&[guard(BB1.0, None, Some(1), None)]).await;
    b.ctx.head_guard = hg;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    for account in [Account::Clean, Account::HasSpamHighRecall] {
        let (t, run_tickets) = b.run_backfill(&bb1(account)).await;
        assert_eq!((t.status.as_str(), run_tickets), ("success", 1));
    }
    let lkey = off_switch::lkey_for(BB1.0, None);
    assert_eq!(bucket_count(&buckets, &lkey), 0);
    assert!(b.ctx.off_switch.get(&lkey).is_none());
}

#[tokio::test]
async fn an_empty_run_budget_stops_before_ais_and_releases_the_claim() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 0), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let (t, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(t.status, backfill::STATUS_BUDGET_REACHED);
    assert!(t.ais.is_empty());
    assert_eq!((t.tickets, run_tickets), (0, 0));
    assert!(
        b.claim(RUN, EntityType::User, Account::Clean.user_id()) == ItemClaim::Free,
        "a later send may still act"
    );
}

#[tokio::test]
async fn a_paused_stopped_or_unknown_run_records_without_any_fetch() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    for (state, want) in [
        (RunState::Paused, backfill::STATUS_RUN_PAUSED),
        (RunState::Stopped, backfill::STATUS_RUN_STOPPED),
        (RunState::Done, backfill::STATUS_RUN_DONE),
    ] {
        b.set_run_state(RUN, state);
        let (t, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
        assert_eq!(t.status, want);
        assert!(t.dedup_calls.is_empty() && t.ais.is_empty() && run_tickets == 0);
        assert_eq!(info(&t)["backfill_run_id"], RUN);
    }
    let (t, _) = b
        .run_backfill(&with_run(bb1(Account::Clean), "run-unknown1"))
        .await;
    assert_eq!(t.status, backfill::STATUS_RUN_UNKNOWN);
}

#[tokio::test]
async fn a_stop_saved_after_arrival_is_caught_before_ais() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    let rec = put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let score = bb1(Account::Clean);
    let decision = DecisionContext {
        control: resolve_run_control(&b.ctx, &score, EntityType::User, &rec),
        run: Some(Arc::new(rec)),
        ..Default::default()
    };
    b.set_run_state(RUN, RunState::Stopped);
    let before = b.run_tickets();
    let out = run_enforcement(&b.ctx, &score, backfill::BACKFILL_TOPIC, &decision)
        .await
        .unwrap();
    assert_eq!(out.status, backfill::STATUS_RUN_STOPPED);
    assert!(b.ais.take().is_empty());
    assert_eq!(b.run_tickets(), before);
    assert!(b.claim(RUN, EntityType::User, Account::Clean.user_id()) == ItemClaim::Free);
}

#[tokio::test]
async fn a_finished_claim_is_never_acted_on_again() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let (first, _) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(first.status, "success");
    assert_eq!(ais_names(&first), vec!["suspendUser"]);
    assert_eq!(
        b.claim(RUN, EntityType::User, Account::Clean.user_id()),
        finished("success")
    );
    b.dedup.slots.lock().unwrap().clear();
    let (again, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(again.status, backfill::STATUS_ALREADY_FINISHED);
    assert!(again.ais.is_empty() && again.dedup_calls.is_empty() && run_tickets == 0);
    assert_eq!(info(&again)["backfill_claim_status"], "success");
}

#[tokio::test]
async fn a_live_dedup_hold_skips_and_leaves_the_claim() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let uid = Account::Clean.user_id();
    b.dedup
        .hold(EntityType::User, uid, dedup_cache::CLASS_SUSPEND);
    let (t, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(t.status, "dedup_skipped");
    assert!(t.ais.is_empty() && run_tickets == 0);
    assert_eq!(info(&t)["backfill_run_id"], RUN);
    assert!(b.claim(RUN, EntityType::User, uid) == ItemClaim::Free);
}

#[tokio::test]
async fn a_decision_outside_the_window_is_not_in_the_run() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    for ts in [0, now_millis() - 3 * 86_400_000, now_millis() + 3_600_000] {
        let mut s = bb1(Account::Clean);
        s.score_ts_ms = ts;
        let (t, _) = b.run_backfill(&s).await;
        assert_eq!(t.status, backfill::STATUS_NOT_IN_RUN, "{ts}");
        assert!(t.ais.is_empty() && t.dedup_calls.is_empty());
    }
}

#[tokio::test]
async fn a_claim_another_pod_holds_is_never_taken_over() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let uid = Account::Clean.user_id();
    b.claims
        .set_handed(key(RUN, EntityType::User, uid), 7, Duration::ZERO);
    let (t, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(t.status, backfill::STATUS_CLAIM_TAKEN);
    assert!(t.ais.is_empty() && run_tickets == 0);
    assert!(matches!(
        b.claim(RUN, EntityType::User, uid),
        ItemClaim::Handed { nonce: 7, .. }
    ));
}

#[tokio::test]
async fn a_pair_finished_meanwhile_is_caught_before_ais() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    let rec = put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let score = bb1(Account::Clean);
    let uid = Account::Clean.user_id();
    b.claims
        .finish(&key(RUN, EntityType::User, uid), 0, "success")
        .await
        .unwrap();
    let decision = DecisionContext {
        control: resolve_run_control(&b.ctx, &score, EntityType::User, &rec),
        run: Some(Arc::new(rec)),
        ..Default::default()
    };
    let out = run_enforcement(&b.ctx, &score, backfill::BACKFILL_TOPIC, &decision)
        .await
        .unwrap();
    assert_eq!(out.status, backfill::STATUS_ALREADY_FINISHED);
    assert!(b.ais.take().is_empty());
    assert_eq!(b.run_tickets(), 0, "caught before the ticket");
    assert_eq!(b.claim(RUN, EntityType::User, uid), finished("success"));
}

#[tokio::test]
async fn a_claim_left_with_ais_reads_the_account_first() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let uid = Account::Suspended.user_id();
    b.put_crashed_claim(RUN, EntityType::User, uid);
    let (t, run_tickets) = b.run_backfill(&bb1(Account::Suspended)).await;
    assert_eq!(t.status, "already_suspended");
    assert!(t.ais.is_empty() && run_tickets == 0);
    assert_eq!(
        b.claim(RUN, EntityType::User, uid),
        finished("already_suspended")
    );

    let label_run = "run-00000002";
    put_running(
        &b.backfill_store,
        label_run,
        BB1.0,
        vec![SHR7],
        vec![LevelRow::Missing],
    );
    let uid = Account::HasSpamHighRecall.user_id();
    b.put_crashed_claim(label_run, EntityType::User, uid);
    let mut s = bb1(Account::HasSpamHighRecall);
    s.backfill_run_id = label_run.into();
    let (t, _) = b.run_backfill(&s).await;
    assert!(t.ais.is_empty(), "{}", t.status);

    let check_run = "run-00000003";
    put_running(
        &b.backfill_store,
        check_run,
        BB1.0,
        vec![LIVENESS],
        vec![LevelRow::Missing],
    );
    let uid = Account::Clean.user_id();
    b.put_crashed_claim(check_run, EntityType::User, uid);
    let mut s = bb1(Account::Clean);
    s.backfill_run_id = check_run.into();
    b.dedup.slots.lock().unwrap().clear();
    let (t, run_tickets) = b.run_backfill(&s).await;
    assert_eq!(t.status, "success");
    assert_eq!(ais_names(&t), vec!["bounceViaTemplate"]);
    assert_eq!(run_tickets, 1);
}

#[tokio::test]
async fn the_hold_lookup_fails_closed_for_run_decisions_only() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for mode in ["shadow", "off"] {
        let mut cfg = run_config(false, 10);
        cfg.config["overturn_hold_gate"] =
            serde_json::json!({"mode": mode, "on_probe_error": "allow"});
        let mut live = bed(
            cfg.clone(),
            vec![record(BB1.0, None, None, ControlMode::Live, vec![SUSPEND])],
        )
        .await;
        let mut runs = bed(cfg, vec![]).await;
        live.ctx.hold_gate = Arc::new(overturn_hold::HoldGate::new(None));
        runs.ctx.hold_gate = Arc::new(overturn_hold::HoldGate::new(None));
        put_running(
            &runs.backfill_store,
            RUN,
            BB1.0,
            vec![SUSPEND],
            vec![LevelRow::Missing],
        );
        let was = live
            .run(&message(
                EntityType::User,
                BB1.0,
                BB1.1,
                &[],
                Account::Clean,
            ))
            .await;
        assert_eq!(was.status, "success", "{mode}");
        let (now, run_tickets) = runs.run_backfill(&bb1(Account::Clean)).await;
        assert_eq!(
            now.status,
            overturn_hold::STATUS_HOLD_LOOKUP_FAILED,
            "{mode}"
        );
        assert!(now.ais.is_empty() && run_tickets == 0, "{mode}");
        assert!(runs.claim(RUN, EntityType::User, Account::Clean.user_id()) == ItemClaim::Free);
    }
    let mut cfg = run_config(false, 10);
    cfg.config["overturn_hold_gate"] = serde_json::json!({"mode": "off"});
    let runs = bed(cfg, vec![]).await;
    put_running(
        &runs.backfill_store,
        RUN,
        BB1.0,
        vec![LIVENESS],
        vec![LevelRow::Missing],
    );
    let (now, _) = runs.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(now.status, "success");
}

#[tokio::test]
async fn the_off_switch_pauses_the_run() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        SWITCHED_OFF_HEAD,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let s = |a: Account| {
        with_run(
            message(EntityType::User, SWITCHED_OFF_HEAD, "clf_v1", &[], a),
            RUN,
        )
    };
    let (t, run_tickets) = b.run_backfill(&s(Account::Clean)).await;
    assert_eq!(t.status, off_switch::STATUS_SWITCHED_OFF);
    assert!(t.ais.is_empty() && run_tickets == 0);
    let rec = b.ctx.backfill.run(RUN).await.unwrap().unwrap();
    assert_eq!(rec.state, RunState::Paused);
    assert_eq!(rec.paused_reason, Some(backfill::PausedReason::OffSwitch));
    let (t, _) = b.run_backfill(&s(Account::HasSpamHighRecall)).await;
    assert_eq!(t.status, backfill::STATUS_RUN_PAUSED);
    assert!(t.dedup_calls.is_empty());
}

#[tokio::test]
async fn an_unwritten_off_switch_hold_fails_the_decision() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    let rec = put_running(
        &b.backfill_store,
        RUN,
        SWITCHED_OFF_HEAD,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let score = with_run(
        message(
            EntityType::User,
            SWITCHED_OFF_HEAD,
            "clf_v1",
            &[],
            Account::Clean,
        ),
        RUN,
    );
    let decision = DecisionContext {
        control: resolve_run_control(&b.ctx, &score, EntityType::User, &rec),
        run: Some(Arc::new(rec)),
        ..Default::default()
    };
    b.backfill_store
        .fail_put
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let out = run_enforcement(&b.ctx, &score, backfill::BACKFILL_TOPIC, &decision).await;
    b.backfill_store
        .fail_put
        .store(false, std::sync::atomic::Ordering::Relaxed);
    assert!(out.is_err(), "{out:?}");
    assert!(b.ais.take().is_empty());
}

#[tokio::test]
async fn messages_the_run_does_not_cover_are_skipped() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let other_head = with_run(
        message(
            EntityType::User,
            HEADS[1].0,
            HEADS[1].1,
            &[],
            Account::Clean,
        ),
        RUN,
    );
    let mut mid = bb1(Account::Clean);
    mid.confidence = abuse_proto::Confidence::Mid as i32;
    let mut dry = bb1(Account::Clean);
    dry.dry_run = true;
    let post = with_run(
        message(EntityType::Post, BB1.0, BB1.1, &[], Account::Clean),
        RUN,
    );
    for (score, want) in [
        (other_head, backfill::STATUS_NOT_IN_RUN),
        (mid, backfill::STATUS_NOT_IN_RUN),
        (post, backfill::STATUS_NOT_IN_RUN),
        (dry, backfill::STATUS_BAD_MESSAGE),
    ] {
        let (t, _) = b.run_backfill(&score).await;
        assert_eq!(t.status, want);
        assert!(t.ais.is_empty() && t.dedup_calls.is_empty());
    }
}

#[tokio::test]
async fn a_run_records_only_where_a_live_record_would() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut off = run_config(false, 10);
    off.config["action_control"] = serde_json::json!({"enabled": false});
    let b = bed(off, vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let (t, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(t.status, "dry_run");
    assert_eq!(info(&t)["action_control_forced_recording"], "kill_switch");
    assert!(t.ais.is_empty() && run_tickets == 0);
    assert!(b.claim(RUN, EntityType::User, Account::Clean.user_id()) == ItemClaim::Free);

    let scope = record(BB1.0, None, None, ControlMode::Rows, vec![]);
    let mid_row = action_control::ActionControlRecord {
        confidence: Some(LevelRow::Mid),
        ..record(BB1.0, None, None, ControlMode::Recording, vec![SUSPEND])
    };
    let b = bed(run_config(false, 10), vec![scope, mid_row]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Mid],
    );
    let mut mid = bb1(Account::Clean);
    mid.confidence = abuse_proto::Confidence::Mid as i32;
    let (t, run_tickets) = b.run_backfill(&mid).await;
    assert_eq!(t.status, "success", "{:?}", info(&t));
    assert_eq!(run_tickets, 1);
    assert_eq!(info(&t)["level_row"], "mid");
    assert_eq!(info(&t)["confidence_evidence_supports"], "none");
    assert!(!info(&t).contains_key("action_control_forced_recording"));
}

#[tokio::test]
async fn a_factory_run_acts_whatever_the_messages_evidence() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    const DAY: i64 = 86_400_000;
    let evidence = |days: i64| abuse_proto::ConfidenceEvidence {
        precision_lower_bound: 0.96,
        reads: 59,
        humans: 0,
        read_at_ms: now_millis() - days * DAY,
        link: "read:rd_1".into(),
    };
    let mut high = bb1(Account::Clean);
    high.confidence = abuse_proto::Confidence::High as i32;
    async fn run_bed(factory: bool) -> TestBed {
        let b = bed(run_config(false, 10), vec![]).await;
        let mut rec = put_running(
            &b.backfill_store,
            RUN,
            BB1.0,
            vec![SUSPEND],
            vec![LevelRow::High],
        );
        rec.factory = factory;
        b.backfill_store.rows.lock().unwrap().insert(
            (backfill::RUN_PKEY.into(), RUN.into()),
            serde_json::to_vec(&rec).unwrap(),
        );
        b
    }
    let (t, run_tickets) = run_bed(false).await.run_backfill(&high).await;
    assert_eq!(t.status, "success", "{:?}", info(&t));
    assert_eq!(run_tickets, 1);
    assert!(!info(&t).contains_key("level_row"), "not on rows");

    for ev in [None, Some(evidence(366))] {
        let mut s = high.clone();
        s.confidence_evidence = ev.clone();
        let (t, run_tickets) = run_bed(true).await.run_backfill(&s).await;
        assert_eq!(t.status, "success", "{ev:?}: {:?}", info(&t));
        assert_eq!(run_tickets, 1);
        assert_eq!(info(&t)["level_row"], "high");
        assert_eq!(info(&t)["confidence_evidence_supports"], "none");
        assert!(!info(&t).contains_key("action_control_forced_recording"));
    }

    let b = run_bed(true).await;
    let mut s = high.clone();
    s.confidence_evidence = Some(evidence(364));
    let classifier_lkey = format!("user/c/{}/{}", BB1.0, BB1.1);
    b.control_store.put_raw(&classifier_lkey, b"{not a record");
    b.ctx.action_control.refresh().await.unwrap();
    let (t, run_tickets) = b.run_backfill(&s).await;
    assert_eq!(t.status, "dry_run", "{:?}", info(&t));
    assert_eq!(info(&t)["action_control_forced_recording"], "unreadable");
    assert!(t.ais.is_empty() && run_tickets == 0);
    b.control_store
        .rows
        .lock()
        .unwrap()
        .retain(|(_, k), _| *k != classifier_lkey);
    b.ctx.action_control.refresh().await.unwrap();
    let (t, run_tickets) = b.run_backfill(&s).await;
    assert_eq!(t.status, "success", "{:?}", info(&t));
    assert_eq!(run_tickets, 1);
    assert_eq!(info(&t)["confidence_evidence_supports"], "high");
    assert!(!info(&t).keys().any(|k| k.starts_with("rules_would")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_pods_racing_on_one_pair_act_once() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    let rec = put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let score = bb1(Account::Clean);
    let mut pod2 = b.ctx.clone();
    pod2.backfill = pod_backfill(&b, b.claims.clone(), "pod-2").await;
    *b.claims.state_barrier.lock().unwrap() = Some(Arc::new(tokio::sync::Barrier::new(2)));
    let decision = |nonce: i64| DecisionContext {
        control: resolve_run_control(&b.ctx, &score, EntityType::User, &rec),
        run: Some(Arc::new(rec.clone())),
        run_nonce: nonce,
        ..Default::default()
    };
    let (d1, d2) = (decision(1), decision(2));
    let (a, c) = tokio::join!(
        run_enforcement(&b.ctx, &score, backfill::BACKFILL_TOPIC, &d1),
        run_enforcement(&pod2, &score, backfill::BACKFILL_TOPIC, &d2),
    );
    *b.claims.state_barrier.lock().unwrap() = None;
    let mut statuses = vec![a.unwrap().status, c.unwrap().status];
    statuses.sort();
    assert_eq!(
        statuses,
        vec![
            backfill::STATUS_CLAIM_TAKEN.to_owned(),
            "success".to_owned()
        ]
    );
    assert_eq!(
        ais_names(&Trace {
            status: String::new(),
            outcomes: Vec::new(),
            dedup_calls: Vec::new(),
            tickets: 0,
            ais: b.ais.take(),
        }),
        vec!["suspendUser"]
    );
    assert_eq!(b.run_tickets(), 1);
    assert_eq!(
        b.claim(RUN, EntityType::User, Account::Clean.user_id()),
        finished("success")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_copies_in_one_pod_act_once() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    let rec = put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let score = bb1(Account::Clean);
    let decision = |nonce: i64| DecisionContext {
        control: resolve_run_control(&b.ctx, &score, EntityType::User, &rec),
        run: Some(Arc::new(rec.clone())),
        run_nonce: nonce,
        ..Default::default()
    };
    let (d1, d2) = (decision(1), decision(2));
    let (a, c) = tokio::join!(
        run_enforcement(&b.ctx, &score, backfill::BACKFILL_TOPIC, &d1),
        run_enforcement(&b.ctx, &score, backfill::BACKFILL_TOPIC, &d2),
    );
    let mut statuses = vec![a.unwrap().status, c.unwrap().status];
    statuses.sort();
    assert_eq!(
        statuses,
        vec![
            backfill::STATUS_ALREADY_FINISHED.to_owned(),
            "success".to_owned()
        ]
    );
    assert_eq!(b.ais.take().len(), 1);
    assert_eq!(b.run_tickets(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_pods_share_the_budget_exactly_from_a_cold_start() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 1), vec![]).await;
    let rec = put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![LIVENESS],
        vec![LevelRow::Missing],
    );
    let mut pod2 = b.ctx.clone();
    pod2.backfill = pod_backfill(&b, b.claims.clone(), "pod-2").await;
    let decide = |ctx: EnforcementCtx, account: Account, nonce: i64| {
        let score = bb1(account);
        let decision = DecisionContext {
            control: resolve_run_control(&ctx, &score, EntityType::User, &rec),
            run: Some(Arc::new(rec.clone())),
            run_nonce: nonce,
            ..Default::default()
        };
        tokio::spawn(async move {
            run_enforcement(&ctx, &score, backfill::BACKFILL_TOPIC, &decision)
                .await
                .unwrap()
                .status
        })
    };
    let a = decide(b.ctx.clone(), Account::Clean, 1);
    let c = decide(pod2.clone(), Account::HasSpamHighRecall, 2);
    let mut statuses = vec![a.await.unwrap(), c.await.unwrap()];
    statuses.sort();
    assert_eq!(
        statuses,
        vec![
            backfill::STATUS_BUDGET_REACHED.to_owned(),
            "success".to_owned()
        ]
    );
    assert_eq!(b.run_tickets(), 1);
}

#[tokio::test]
async fn a_pair_missing_from_the_frozen_list_is_not_in_the_run() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut b = bed(run_config(false, 10), vec![]).await;
    let strict = Arc::new(crate::backfill_claims::test_support::MemClaimStore::default());
    b.ctx.backfill = pod_backfill(&b, strict.clone(), "").await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let (t, _) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(t.status, backfill::STATUS_NOT_IN_RUN);
    assert!(t.ais.is_empty() && t.dedup_calls.is_empty());
    strict.add(key(RUN, EntityType::User, Account::Clean.user_id()));
    let (t, _) = b.run_backfill(&bb1(Account::Clean)).await;
    assert_eq!(t.status, "success");
}

#[tokio::test]
async fn an_unreachable_claim_store_fails_the_decision_for_a_retry() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(run_config(false, 10), vec![]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    b.claims
        .fail_reads
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let (t, run_tickets) = b.run_backfill(&bb1(Account::Clean)).await;
    assert!(t.status.starts_with("error:"), "{}", t.status);
    assert!(t.ais.is_empty() && run_tickets == 0);
}

#[tokio::test]
async fn a_run_skips_by_the_head_records_follower_threshold() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let at = Account::BetweenLimits.followers() as u32;
    let holder = |t: Option<u32>, c: Option<bool>| action_control::ActionControlRecord {
        follower_threshold: t,
        skip_high_credibility: c,
        ..record(BB1.0, None, None, ControlMode::Rules, vec![])
    };
    for (records, want) in [
        (vec![], ["high_follower_count", "pagerank_skipped"]),
        (
            vec![holder(None, None)],
            ["high_follower_count", "pagerank_skipped"],
        ),
        (vec![holder(Some(0), None)], ["success", "pagerank_skipped"]),
        (
            vec![holder(Some(at), None)],
            ["high_follower_count", "pagerank_skipped"],
        ),
        (
            vec![holder(None, Some(false))],
            ["high_follower_count", "success"],
        ),
        (vec![holder(Some(0), Some(false))], ["success", "success"]),
    ] {
        let b = bed(run_config(false, 10), records.clone()).await;
        put_running(
            &b.backfill_store,
            RUN,
            BB1.0,
            vec![SUSPEND],
            vec![LevelRow::Missing],
        );
        for (account, want) in [Account::BetweenLimits, Account::HighCred]
            .into_iter()
            .zip(want)
        {
            let (t, run_tickets) = b.run_backfill(&bb1(account)).await;
            assert_eq!(t.status, want, "{records:?} {account:?}");
            assert_eq!(run_tickets, i64::from(want == "success"), "{account:?}");
        }
        let (t, _) = b.run_backfill(&bb1(Account::AboveTopLimit)).await;
        assert_eq!(t.status, "very_high_follower_count");
    }
}

#[tokio::test]
async fn an_older_backfill_image_never_skips_fewer_accounts_than_today() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let view = safety_checks::follower_skip_view();
    let (today, ceiling) = (view.today_threshold as u32, view.ceiling as u32);
    let as_written = |mut r: action_control::ActionControlRecord| {
        r.v = r.schema_version();
        r
    };
    for (t, c) in [
        (Some(0), None),
        (Some(today), None),
        (Some(today * 5), None),
        (Some(ceiling - 1), Some(true)),
        (None, Some(false)),
        (Some(0), Some(false)),
    ] {
        let head = |mode: ControlMode| action_control::ActionControlRecord {
            follower_threshold: t,
            skip_high_credibility: c,
            ..record(BB1.0, None, None, mode, vec![])
        };
        let missing_row = action_control::ActionControlRecord {
            confidence: Some(LevelRow::Missing),
            ..record(BB1.0, None, None, ControlMode::Recording, vec![SUSPEND])
        };
        for records in [
            vec![as_written(head(ControlMode::Rules))],
            vec![as_written(head(ControlMode::Rows)), as_written(missing_row)],
        ] {
            let b = bed(run_config(false, 10), vec![]).await;
            for r in &records {
                b.control_store.put_record(r);
            }
            crate::action_control::test_support::refresh_as_older_image(&b.ctx.action_control)
                .await;
            put_running(
                &b.backfill_store,
                RUN,
                BB1.0,
                vec![SUSPEND],
                vec![LevelRow::Missing],
            );
            for (account, today) in [
                (Account::BetweenLimits, "high_follower_count"),
                (Account::HighCred, "pagerank_skipped"),
            ] {
                let (tr, run_tickets) = b.run_backfill(&bb1(account)).await;
                let what = format!("{t:?} {c:?} {:?} {account:?}", records[0].mode);
                assert!(
                    tr.ais.is_empty() && run_tickets == 0,
                    "{what}: {}",
                    tr.status
                );
                let recorded = tr.outcomes.last().is_some_and(|o| o.dry_run);
                assert!(tr.status == today || recorded, "{what}: {}", tr.status);
            }
        }
    }
}

#[tokio::test]
async fn lifting_a_skip_a_running_backfill_reads_needs_the_confirm() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    for renamed in [false, true] {
        let holder = if renamed { "OldEmbeddingHead" } else { BB1.0 };
        let mut cfg = run_config(false, 10);
        if renamed {
            cfg.config["action_control"]["aliases"] = serde_json::json!({ BB1.0: holder });
        }
        let b = bed(cfg, vec![]).await;
        if renamed {
            let new_name =
                action_control::lkey_for(action_control::ControlEntity::User, BB1.0, None, None);
            b.control_store
                .rows
                .lock()
                .unwrap()
                .remove(&(action_control::MH_PKEY.to_owned(), new_name));
            b.ctx.action_control.refresh().await.unwrap();
        }
        put_running(
            &b.backfill_store,
            RUN,
            BB1.0,
            vec![SUSPEND],
            vec![LevelRow::Missing],
        );
        let control = b.ctx.dynamic_config.action_control_config();
        let safety = action_control::SafetyChecksByEntity {
            user: safety_checks::status(
                EntityType::User,
                &b.ctx.rules_cache.resolve(EntityType::User, None),
            ),
            post: safety_checks::status(
                EntityType::Post,
                &b.ctx.rules_cache.resolve(EntityType::Post, None),
            ),
        };
        let lift = |confirm: Option<&str>| action_control::ActionControlPutRequest {
            entity_type: action_control::ControlEntity::User,
            head: holder.into(),
            classifier: None,
            band: None,
            mode: ControlMode::Rules,
            actions: vec![],
            reason: "lift the follower skip".into(),
            expected_version: (!renamed).then_some(1),
            confirm: confirm.map(Into::into),
            updated_by: None,
            confidence: None,
            evidence: None,
            evidence_override: None,
            rows: None,
            admit_unlisted_labels: None,
            rollout_bp: None,
            follower_threshold: Some(Some(0)),
            skip_high_credibility: None,
        };
        let save = |req| {
            b.ctx.action_control.set(
                req,
                "operator@example.com",
                "page",
                now_millis(),
                &control,
                &safety,
            )
        };
        match save(lift(None)).await {
            Err(action_control::SetError::Invalid("confirm_required", _)) => {}
            other => panic!("renamed={renamed}: {other:?}"),
        }
        save(lift(Some("ENFORCE"))).await.unwrap();
        let (t, run_tickets) = b.run_backfill(&bb1(Account::BetweenLimits)).await;
        assert_eq!(t.status, "success", "renamed={renamed}");
        assert_eq!(run_tickets, 1);
    }
}

#[tokio::test]
async fn a_run_stops_at_the_follower_ceiling() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let off = action_control::ActionControlRecord {
        follower_threshold: Some(0),
        ..record(BB1.0, None, None, ControlMode::Rules, vec![])
    };
    let b = bed(run_config(false, 10), vec![off]).await;
    put_running(
        &b.backfill_store,
        RUN,
        BB1.0,
        vec![SUSPEND],
        vec![LevelRow::Missing],
    );
    let (t, run_tickets) = b.run_backfill(&bb1(Account::AboveTopLimit)).await;
    assert_eq!(t.status, "very_high_follower_count");
    assert!(t.ais.is_empty() && run_tickets == 0);
    let (t, run_tickets) = b.run_backfill(&bb1(Account::JustBelowTopLimit)).await;
    assert_eq!(t.status, "success");
    assert_eq!(run_tickets, 1);
}
