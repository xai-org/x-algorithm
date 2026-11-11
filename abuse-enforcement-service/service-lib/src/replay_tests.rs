use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use xai_abuse_proto::enforcement::{
    self as abuse_proto, EntityType as ProtoEntityType, FiredHead, ScoreResult, SummaryCounters,
};
use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, Label, Labels, Safety};
use xai_core_entities::gizmoduck_client::MockGizmoduckClient;

use super::*;
use crate::ais_client::AisRecorder;
use crate::dedup_cache::test_support::RecordingDedup;

pub(crate) const BAKED_USER_RULES: &str = include_str!("../rules/enforcement_user.yaml");
pub(crate) const BAKED_POST_RULES: &str = include_str!("../rules/enforcement_post.yaml");

pub(crate) const ACTION_CONTROL_INFO_KEYS: &[&str] = &[
    "env",
    "dry_run_source",
    "action_specs",
    "action_control_mode",
    "action_control_scope",
    "action_control_updated_by",
    "action_control_updated_at_ms",
    "action_control_version",
    "action_control_lkey",
    "action_control_forced_recording",
    "action_control_follower_threshold",
    "action_control_skip_high_credibility",
    "matched_rule_id",
    "rules_would_rule_id",
    "rules_would_action_specs",
    "would_dedup_skip",
    "would_dedup_held_class",
    "would_rate_cap",
];

#[derive(Clone, Copy, Debug)]
pub(crate) enum Account {
    Clean,
    Suspended,
    Deactivated,
    HasSpamHighRecall,
    BetweenLimits,
    AboveTopLimit,
    HighCred,
    NotFound,
    HasPhoenixSkip,
    JustBelowTopLimit,
}

impl Account {
    pub(crate) const ALL: [Account; 8] = [
        Account::Clean,
        Account::Suspended,
        Account::Deactivated,
        Account::HasSpamHighRecall,
        Account::BetweenLimits,
        Account::AboveTopLimit,
        Account::HighCred,
        Account::NotFound,
    ];

        pub(crate) const EXTRA: [Account; 2] = [Account::HasPhoenixSkip, Account::JustBelowTopLimit];

    pub(crate) fn user_id(self) -> i64 {
        1_700_100_000 + self as i64
    }

            pub(crate) fn followers(self) -> i64 {
        let view = crate::safety_checks::follower_skip_view();
        match self {
            Account::BetweenLimits => view.today_threshold as i64,
            Account::AboveTopLimit => view.ceiling as i64 + 1,
            Account::JustBelowTopLimit => view.ceiling as i64 - 1,
            _ => 1,
        }
    }
}

pub(crate) struct TestBed {
    pub ctx: EnforcementCtx,
    pub dedup: Arc<RecordingDedup>,
    pub ais: Arc<AisRecorder>,
    pub tickets: Arc<AtomicI64>,
        pub claims: Arc<backfill_claims::test_support::MemClaimStore>,
    pub outcomes: Arc<std::sync::Mutex<Vec<abuse_proto::DecisionOutcome>>>,
    pub control_store: Arc<action_control::test_support::MemControlStore>,
    pub backfill_store: Arc<action_control::test_support::MemControlStore>,
        pub post_labels: Arc<crate::post_labels::test_support::FakePostLabels>,
    gizmoduck: Arc<MockGizmoduckClient>,
    strato: wiremock::MockServer,
}

pub(crate) struct BedConfig<'a> {
    pub user_rules: &'a str,
    pub post_rules: &'a str,
        pub config: serde_json::Value,
    pub off_switches: Vec<off_switch::OffSwitchRecord>,
    pub control_records: Vec<action_control::ActionControlRecord>,
}

fn gizmoduck_user(account: Account) -> Option<GizmoduckUserResult> {
    if matches!(account, Account::NotFound) {
        return None;
    }
    let labels = match account {
        Account::HasSpamHighRecall => vec![Label {
            label_value: 23,
            created_at_msec: 0,
        }],
        Account::HasPhoenixSkip => vec![Label {
            label_value: 118,
            created_at_msec: 0,
        }],
        _ => Vec::new(),
    };
    Some(GizmoduckUserResult {
        user: Some(GizmoduckUser {
            user_id: account.user_id() as u64,
            safety: Safety {
                suspended: matches!(account, Account::Suspended),
                deactivated: matches!(account, Account::Deactivated),
                ..Default::default()
            },
            labels: Labels { labels },
            ..Default::default()
        }),
        response_state: None,
    })
}

impl TestBed {
    pub(crate) async fn new(cfg: BedConfig<'_>) -> TestBed {
        let server = wiremock::MockServer::start().await;
        for account in Account::ALL.into_iter().chain(Account::EXTRA) {
            let body = serde_json::json!({"v": {
                "isHighPageRankUser": matches!(account, Account::HighCred),
                "userCredScore": 1.0,
                "followerCount": account.followers().to_string(),
            }});
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .and(wiremock::matchers::path_regex("/hpr$"))
                .and(wiremock::matchers::body_string_contains(
                    account.user_id().to_string(),
                ))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .with_priority(1)
                .mount(&server)
                .await;
        }
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({"v": null})),
            )
            .with_priority(5)
            .mount(&server)
            .await;
        #[allow(deprecated)]
        let column = |name: &str| {
            Arc::new(Strato::with_client(
                name,
                server.uri(),
                Strato::build_default_client(&Default::default()).unwrap(),
            ))
        };
        let gd = Arc::new(MockGizmoduckClient {
            users: Account::ALL
                .into_iter()
                .chain(Account::EXTRA)
                .map(|a| (a.user_id(), gizmoduck_user(a)))
                .collect::<HashMap<_, _>>(),
            ..Default::default()
        });
        let (user_cap, tickets) = sliding_window::counting_limiter_for_test();
        user_cap.seed_for_test();
        let dynamic_config = DynamicConfig::new(
            None,
            None,
            false,
            1_000_000,
            1_000_000,
            Some(user_cap),
            None,
        )
        .await
        .unwrap();
        dynamic_config.set_snapshot_for_test(growthbook::ConfigSnapshot {
            config: Some(cfg.config),
            rules_user: Some(Arc::from(cfg.user_rules)),
            rules_post: Some(Arc::from(cfg.post_rules)),
        });
        let rules_cache = rules::RulesCache::try_seed(cfg.user_rules, cfg.post_rules).unwrap();
        let switches = Arc::new(off_switch::test_support::MemStore::default());
        for r in &cfg.off_switches {
            switches.put_record(r);
        }
        let off_switch = Arc::new(off_switch::OffSwitch::new(
            switches,
            Arc::new(off_switch::test_support::MemStore::default()),
        ));
        off_switch.refresh().await.unwrap();
        let ais = Arc::new(AisRecorder::default());
        let outcomes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let control_store = Arc::new(action_control::test_support::MemControlStore::default());
        for r in rules_records().iter().chain(&cfg.control_records) {
            control_store.put_record(r);
        }
        let action_control = Arc::new(action_control::ActionControl::new(
            control_store.clone(),
            "prod".into(),
            String::new(),
        ));
        action_control.refresh().await.unwrap();
        let backfill_store = Arc::new(action_control::test_support::MemControlStore::default());
        let claims = Arc::new(backfill_claims::test_support::MemClaimStore::listing_all());
        let post_labels = Arc::new(crate::post_labels::test_support::FakePostLabels::default());
        let ctx = EnforcementCtx {
            ais_client: ais_client::AisClient::recording(ais.clone()),
            gd: gizmoduck::GizmoduckCoreClient::from_client(gd.clone()),
            cred_clients: strato::CredClients {
                high_page_rank: column("hpr"),
                grey_badge: column("grey"),
            },
            uas_strato: column("uas"),
            dynamic_config,
            rules_cache: Arc::new(rules_cache),
            allowlist: None,
            kafka_producer_decisions: None,
            kafka_producer_decisions_json: None,
            hold_gate: Arc::new(overturn_hold::HoldGate::new(Some(
                overturn_hold::FakeHoldStore::new(Vec::new()),
            ))),
            off_switch,
            head_guard: Arc::new(head_guard::HeadGuard::new(
                Arc::new(action_control::test_support::MemControlStore::default()),
                None,
            )),
            action_control,
            backfill: Arc::new(backfill::Backfill::new(
                backfill_store.clone(),
                Some(claims.clone()),
                String::new(),
            )),
            post_labels: Some(post_labels.clone()),
            outcome_log: Some(outcomes.clone()),
        };
        TestBed {
            ctx,
            dedup: Arc::new(RecordingDedup::default()),
            ais,
            tickets,
            claims,
            outcomes,
            control_store,
            backfill_store,
            post_labels,
            gizmoduck: gd,
            strato: server,
        }
    }

        pub(crate) async fn fetches(&self) -> usize {
        self.gizmoduck.call_count() + self.strato.received_requests().await.map_or(0, |r| r.len())
    }

    pub(crate) fn processor(&self) -> ScoreResultProcessor {
        ScoreResultProcessor {
            topic: "abuse.embeddings.user_decisions".into(),
            labels: HashSet::new(),
            negative_labels: HashSet::new(),
            ctx: self.ctx.clone(),
            dedup_cache: self.dedup.clone(),
            retry_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            last_progress: Arc::new(AtomicI64::new(0)),
            max_in_flight: 1,
            backfill_topic: false,
        }
    }

        pub(crate) fn backfill_processor(&self) -> ScoreResultProcessor {
        ScoreResultProcessor {
            topic: backfill::BACKFILL_TOPIC.into(),
            backfill_topic: true,
            ..self.processor()
        }
    }

            pub(crate) async fn set_records(&self, records: &[action_control::ActionControlRecord]) {
        self.control_store.rows.lock().unwrap().clear();
        for r in rules_records().iter().chain(records) {
            self.control_store.put_record(r);
        }
        self.ctx.action_control.refresh().await.unwrap();
    }

        pub(crate) async fn run_fresh(&self, score: &ScoreResult) -> Trace {
        self.dedup.slots.lock().unwrap().clear();
        self.run(score).await
    }

        pub(crate) async fn run(&self, score: &ScoreResult) -> Trace {
        self.run_on(&self.processor(), score).await
    }

        pub(crate) async fn run_on(
        &self,
        processor: &ScoreResultProcessor,
        score: &ScoreResult,
    ) -> Trace {
        self.dedup.take_calls();
        self.ais.take();
        self.outcomes.lock().unwrap().clear();
        let tickets_before = self.tickets.load(Ordering::Relaxed);
        let result = processor.process(score, None).await;
        self.trace(result, tickets_before)
    }

            pub(crate) async fn run_admitted(
        &self,
        processor: &ScoreResultProcessor,
        score: &ScoreResult,
    ) -> Trace {
        self.dedup.slots.lock().unwrap().clear();
        self.run_admitted_on_held(processor, score).await
    }

        pub(crate) async fn run_admitted_on_held(
        &self,
        processor: &ScoreResultProcessor,
        score: &ScoreResult,
    ) -> Trace {
        self.dedup.take_calls();
        self.ais.take();
        self.outcomes.lock().unwrap().clear();
        let tickets_before = self.tickets.load(Ordering::Relaxed);
        let result = match processor.admit(score).await {
            Admission::Admitted(control) => processor.process(score, control.map(|c| *c)).await,
            Admission::Dropped(status) => Ok(status.to_owned()),
        };
        self.trace(result, tickets_before)
    }

    fn trace(&self, result: Result<String, EnforcementFailed>, tickets_before: i64) -> Trace {
        let status = match &result {
            Ok(s) => s.clone(),
            Err(e) => format!("error:{}", e.source),
        };
        let outcomes = std::mem::take(&mut *self.outcomes.lock().unwrap());
        Trace {
            status,
            outcomes,
            dedup_calls: self.dedup.take_calls(),
            tickets: self.tickets.load(Ordering::Relaxed) - tickets_before,
            ais: self.ais.take(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Trace {
    pub status: String,
    pub outcomes: Vec<abuse_proto::DecisionOutcome>,
    pub dedup_calls: Vec<String>,
    pub tickets: i64,
    pub ais: Vec<(String, serde_json::Value)>,
}

fn scrub(v: &mut serde_json::Value) {
    const NOTE: &str = "Automated enforcement by xai-abuse-enforcement-service";
    match v {
        serde_json::Value::String(s) => {
            if let Some(i) = s.find(NOTE)
                && let Some(j) = s[i..].find(']')
            {
                s.replace_range(i..i + j + 1, "<audit note>");
            }
        }
        serde_json::Value::Object(m) => {
            for (k, x) in m.iter_mut() {
                if k.ends_with("Msec") || k.ends_with("Ms") {
                    *x = serde_json::Value::Null;
                } else {
                    scrub(x);
                }
            }
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(scrub),
        _ => {}
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn ais_action_names(ais: &[(String, serde_json::Value)]) -> Vec<String> {
    ais.iter()
        .flat_map(|(method, req)| {
            let actions: Vec<&serde_json::Value> = match req.get("actions") {
                Some(serde_json::Value::Array(a)) => a.iter().collect(),
                _ => req.get("action").into_iter().collect(),
            };
            actions
                .into_iter()
                .map(|a| {
                    let name = a
                        .as_object()
                        .and_then(|o| o.keys().next())
                        .map_or("?", String::as_str);
                    format!("{method}:{name}")
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

impl Trace {
        pub(crate) fn old_info(&self) -> Vec<BTreeMap<String, String>> {
        self.outcomes
            .iter()
            .map(|o| {
                o.info
                    .iter()
                    .filter(|(k, _)| !ACTION_CONTROL_INFO_KEYS.contains(&k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .collect()
    }

                    pub(crate) fn golden_line(&self, id: &str) -> String {
        let dry: Vec<bool> = self.outcomes.iter().map(|o| o.dry_run).collect();
        let mut ais = self.ais.clone();
        ais.iter_mut().for_each(|(_, r)| scrub(r));
        let mut info = self.old_info();
        for i in &mut info {
            i.remove("cred_follower_count");
        }
        let hashed = serde_json::json!({
            "info": info,
            "outcome_status": self.outcomes.iter().map(|o| o.status.clone()).collect::<Vec<_>>(),
            "ais": ais,
        })
        .to_string();
        if std::env::var("ACTION_CONTROL_REPLAY_DEBUG").is_ok_and(|p| id.contains(&p)) {
            eprintln!("DEBUG {id}: {hashed}");
        }
        format!(
            "{id}\t{}\t{dry:?}\t{}\tcap={}\t{}\t{:016x}",
            self.status,
            self.dedup_calls.join(","),
            self.tickets,
            ais_action_names(&self.ais).join(","),
            fnv(&hashed)
        )
    }
}

pub(crate) fn rule_labels(yaml: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = yaml;
    while let Some(i) = rest.find("in score.labels") {
        let before = &rest[..i];
        if let Some(end) = before.rfind('"')
            && let Some(start) = before[..end].rfind('"')
        {
            let label = &before[start + 1..end];
            if !label.is_empty()
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_')
                && !out.iter().any(|l| l == label)
            {
                out.push(label.to_owned());
            }
        }
        rest = &rest[i + "in score.labels".len()..];
    }
    out
}

pub(crate) fn message(
    entity: EntityType,
    head: &str,
    model: &str,
    labels: &[&str],
    account: Account,
) -> ScoreResult {
    let mut s = ScoreResult {
        user_id: account.user_id(),
        model_version: model.into(),
        summary: Some(SummaryCounters {
            fired_heads: vec![FiredHead {
                name: head.into(),
                score: 0.99,
                threshold: 0.9,
            }],
            labels: labels.iter().map(|l| (*l).to_owned()).collect(),
            ..Default::default()
        }),
        ..Default::default()
    };
    if entity == EntityType::Post {
        s.entity_type = ProtoEntityType::Post as i32;
        s.entity_id = 9_100_000_000 + account as i64;
    }
    s
}

pub(crate) fn factory_message(account: Account) -> ScoreResult {
    let mut s = message(
        EntityType::User,
        "IsDelayedSuspendSpam",
        "is_delayed_suspend_spam@3",
        &["model_factory"],
        account,
    );
    s.policy_version = "3".into();
    s
}

pub(crate) fn base_config(dry_run: bool, cap: Option<u64>) -> serde_json::Value {
    let mut c = serde_json::json!({ "dry_run": dry_run });
    if let Some(cap) = cap {
        c["max_user_enforcements_per_day"] = cap.into();
    }
    c
}

pub(crate) fn on_config(dry_run: bool, cap: Option<u64>) -> serde_json::Value {
    let mut c = base_config(dry_run, cap);
    c["action_control"] = serde_json::json!({"enabled": true});
    c
}

pub(crate) const HEADS: &[(&str, &str)] = &[
    (
        "UserEmbeddingPostSpamBot",
        "bb1_tweetcreate_abuse_any_v15_user_512_mlp",
    ),
    ("NearDupEmbeddingCseMatch", "cse_cluster_neardup_v1"),
    (
        "PandaReportsEmbeddingV10RoughUserAggGe4",
        "cse_reports_embedding_v10_rough_post_1024_std",
    ),
    ("GraphSuspensionTrainedFleet", "fleet_graph_v4_0"),
];

pub(crate) const SWITCHED_OFF_HEAD: &str = "SwitchedOffHead";

pub(crate) fn rules_records() -> Vec<action_control::ActionControlRecord> {
    HEADS
        .iter()
        .map(|(h, _)| *h)
        .chain([SWITCHED_OFF_HEAD])
        .flat_map(|h| {
            let user = action_control::test_support::record(
                h,
                None,
                None,
                action_control::ControlMode::Rules,
                Vec::new(),
            );
            let post = action_control::ActionControlRecord {
                entity_type: action_control::ControlEntity::Post,
                ..user.clone()
            };
            [user, post]
        })
        .collect()
}

pub(crate) struct Scenario {
    pub id: String,
    pub score: ScoreResult,
    pub dry_run: bool,
    pub cap: Option<u64>,
    pub held_class: Option<u8>,
}

pub(crate) fn scenarios(user_rules: &str, post_rules: &str) -> Vec<Scenario> {
    let mut out = Vec::new();
    let mut push = |id: String, score: ScoreResult| {
        for (dry_run, held_class) in [
            (false, None),
            (true, None),
            (false, Some(dedup_cache::CLASS_LABEL)),
        ] {
            {
                out.push(Scenario {
                    id: format!(
                        "{id}|dry={}|held={}",
                        dry_run as u8,
                        held_class.map_or("-".into(), |c| c.to_string())
                    ),
                    score: score.clone(),
                    dry_run,
                    cap: None,
                    held_class,
                });
            }
        }
    };
    for (entity, yaml) in [
        (EntityType::User, user_rules),
        (EntityType::Post, post_rules),
    ] {
        let mut labels = rule_labels(yaml);
        labels.push(String::new());
        for (i, label) in labels.iter().enumerate() {
            let (head, model) = HEADS[i % HEADS.len()];
            let ls: Vec<&str> = if label.is_empty() {
                vec![]
            } else {
                vec![label]
            };
            for account in Account::ALL {
                push(
                    format!("{}|{head}|{label}|{account:?}", entity.as_str()),
                    message(entity, head, model, &ls, account),
                );
            }
        }
    }
    for account in Account::ALL {
        push(
            format!("user|factory|{account:?}"),
            factory_message(account),
        );
    }
    let first_user_label = rule_labels(user_rules)
        .into_iter()
        .next()
        .unwrap_or_default();
    for account in [Account::Clean, Account::HasSpamHighRecall] {
        push(
            format!("user|{SWITCHED_OFF_HEAD}|{first_user_label}|{account:?}"),
            message(
                EntityType::User,
                SWITCHED_OFF_HEAD,
                "clf_v1",
                &[first_user_label.as_str()],
                account,
            ),
        );
    }
    for label in rule_labels(user_rules) {
        out.push(Scenario {
            id: format!("user|cap_full|{label}|Clean"),
            score: message(
                EntityType::User,
                HEADS[0].0,
                HEADS[0].1,
                &[label.as_str()],
                Account::Clean,
            ),
            dry_run: false,
            cap: Some(0),
            held_class: None,
        });
    }
    out
}

pub(crate) fn switched_off() -> off_switch::OffSwitchRecord {
    let mut r = off_switch::test_support::record(SWITCHED_OFF_HEAD, None, true, None);
    r.updated_at_ms = 1;
    r
}

pub(crate) async fn run_scenarios(
    user_rules: &str,
    post_rules: &str,
    configure: impl Fn(&mut BedConfig<'_>),
) -> Vec<(String, String)> {
    let all = scenarios(user_rules, post_rules);
    let mut beds: HashMap<(bool, Option<u64>), TestBed> = HashMap::new();
    let mut lines = Vec::with_capacity(all.len());
    for sc in all {
        let key = (sc.dry_run, sc.cap);
        if let std::collections::hash_map::Entry::Vacant(slot) = beds.entry(key) {
            let mut cfg = BedConfig {
                user_rules,
                post_rules,
                config: base_config(sc.dry_run, sc.cap),
                off_switches: vec![switched_off()],
                control_records: Vec::new(),
            };
            configure(&mut cfg);
            slot.insert(TestBed::new(cfg).await);
        }
        let bed = &beds[&key];
        let (entity_type, entity_id) = Facts::resolve_target(&sc.score);
        bed.dedup.slots.lock().unwrap().clear();
        if let Some(c) = sc.held_class {
            bed.dedup.hold(entity_type, entity_id, c);
        }
        let trace = bed.run(&sc.score).await;
        lines.push((sc.id.clone(), trace.golden_line(&sc.id)));
    }
    lines
}

const NO_RECORDS_GOLDEN: &str = include_str!("fixtures/replay_no_records.golden");

fn compare_or_write(
    lines: &[(String, String)],
    golden: Option<&str>,
    golden_path: &std::path::Path,
) {
    let text: String = lines.iter().map(|(_, l)| format!("{l}\n")).collect();
    if std::env::var("ACTION_CONTROL_REPLAY_WRITE").as_deref() == Ok("1") {
        std::fs::write(golden_path, &text).unwrap();
        eprintln!(
            "wrote {} scenarios to {}",
            lines.len(),
            golden_path.display()
        );
        return;
    }
    let golden = match golden {
        Some(g) => g.to_owned(),
        None => std::fs::read_to_string(golden_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", golden_path.display())),
    };
    let want: HashMap<&str, &str> = golden
        .lines()
        .filter_map(|l| Some((l.split('\t').next()?, l)))
        .collect();
    let mut diffs = Vec::new();
    for (id, line) in lines {
        match want.get(id.as_str()) {
            Some(w) if *w == line => {}
            Some(w) => diffs.push(format!("CHANGED\n  was: {w}\n  now: {line}")),
            None => diffs.push(format!("NEW SCENARIO (not in golden)\n  now: {line}")),
        }
    }
    assert_eq!(
        want.len(),
        lines.len(),
        "golden has {} scenarios, run has {}",
        want.len(),
        lines.len()
    );
    assert!(
        diffs.is_empty(),
        "{} of {} scenarios differ from the golden:\n{}",
        diffs.len(),
        lines.len(),
        diffs.join("\n")
    );
    eprintln!(
        "{} scenarios identical to {}",
        lines.len(),
        golden_path.display()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_no_records_matches_golden() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let lines = run_scenarios(BAKED_USER_RULES, BAKED_POST_RULES, |_| {}).await;
    let path = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default())
        .join("src/fixtures/replay_no_records.golden");
    compare_or_write(&lines, Some(NO_RECORDS_GOLDEN), &path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_no_records_live_rules_matches_golden() {
    let (Ok(dir), Ok(golden)) = (
        std::env::var("ACTION_CONTROL_REPLAY_RULES_DIR"),
        std::env::var("ACTION_CONTROL_REPLAY_GOLDEN"),
    ) else {
        eprintln!(
            "ACTION_CONTROL_REPLAY_RULES_DIR / _GOLDEN unset; skipping the live-rules replay"
        );
        return;
    };
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let user = std::fs::read_to_string(format!("{dir}/rule_user.yaml")).unwrap();
    let post = std::fs::read_to_string(format!("{dir}/rule_post.yaml")).unwrap();
    let lines = run_scenarios(&user, &post, |_| {}).await;
    compare_or_write(&lines, None, std::path::Path::new(&golden));
}
