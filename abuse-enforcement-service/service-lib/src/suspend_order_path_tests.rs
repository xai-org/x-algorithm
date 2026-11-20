use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use xai_abuse_proto::enforcement::ScoreResult;

use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{ActionControlRecord, ControlMode, PageAction, SuspensionPolicy};
use crate::dedup_cache::{SlotHolder, CLASS_SUSPEND, CLASS_SUSPEND_CSE, CLASS_SUSPEND_PERMANENT};
use crate::gizmoduck::test_support::{AccountShape, FixedUsers};
use crate::gizmoduck::GizmoduckCoreClient;
use crate::replay_tests::{
    message, on_config, rule_labels, switched_off, Account, BedConfig, TestBed, Trace,
    BAKED_POST_RULES, BAKED_USER_RULES, HEADS,
};

const ORDINARY: (&str, &str) = HEADS[0];
const CSE: (&str, &str) = HEADS[1];
const PERMANENT: (&str, &str) = HEADS[2];
const POST_HEAD: &str = "SpamEmbeddingPtosDistilled";
const LABEL: &str = "anchor_campaign_suspend";

const SUSPEND_PERMANENT_CSE: PageAction = PageAction::Suspend {
    permanent: true,
    policy: SuspensionPolicy::Cse,
};
const SUSPEND_PERMANENT: PageAction = PageAction::Suspend {
    permanent: true,
    policy: SuspensionPolicy::PlatformManipulation,
};

fn records() -> Vec<ActionControlRecord> {
    let live =
        |head: &str, action: PageAction| record(head, None, None, ControlMode::Live, vec![action]);
    vec![
        live(ORDINARY.0, PageAction::SUSPEND),
        live(CSE.0, SUSPEND_PERMANENT_CSE),
        live(PERMANENT.0, SUSPEND_PERMANENT),
        live(POST_HEAD, PageAction::SuspendAuthor {}),
    ]
}

async fn bed() -> TestBed {
    TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config: on_config(false, None),
        off_switches: vec![switched_off()],
        control_records: records(),
    })
    .await
}

async fn bed_suspended(account: Account, permanently: bool) -> (TestBed, Arc<FixedUsers>) {
    let mut b = bed().await;
    let mut user = AccountShape::default().user(account.user_id());
    user.safety = Some(xai_x_thrift::user::Safety {
        suspended: Some(true),
        suspension_details: Some(xai_x_thrift::user::SuspensionDetails {
            permanently_suspended: Some(permanently),
            ..Default::default()
        }),
        ..Default::default()
    });
    let users = Arc::new(FixedUsers {
        users: [(account.user_id(), user)].into(),
        ..Default::default()
    });
    b.ctx.gd = GizmoduckCoreClient::from_lookup(users.clone());
    (b, users)
}

fn user_message(head: (&str, &str), account: Account) -> ScoreResult {
    message(EntityType::User, head.0, head.1, &[LABEL], account)
}

fn ordinary(account: Account) -> ScoreResult {
    user_message(ORDINARY, account)
}

fn cse(account: Account) -> ScoreResult {
    user_message(CSE, account)
}

fn permanent(account: Account) -> ScoreResult {
    user_message(PERMANENT, account)
}

fn post_author(account: Account) -> ScoreResult {
    let label = rule_labels(BAKED_POST_RULES)
        .into_iter()
        .next()
        .expect("a post routing label");
    message(EntityType::Post, POST_HEAD, "clf", &[&label], account)
}

fn suspends(t: &Trace) -> Vec<(bool, String)> {
    t.ais
        .iter()
        .flat_map(|(_, r)| match r.get("actions") {
            Some(serde_json::Value::Array(a)) => a.clone(),
            _ => r.get("action").cloned().into_iter().collect(),
        })
        .filter_map(|a| {
            let s = a.get("suspendUser")?;
            Some((
                s.get("perm")?.as_bool()?,
                s.get("policy")?.as_str()?.to_owned(),
            ))
        })
        .collect()
}

fn sent_ordinary() -> Vec<(bool, String)> {
    vec![(false, "PlatformManipulation".into())]
}

fn sent_permanent() -> Vec<(bool, String)> {
    vec![(true, "PlatformManipulation".into())]
}

fn sent_cse() -> Vec<(bool, String)> {
    vec![(true, "Cse".into())]
}

fn info(t: &Trace) -> &BTreeMap<String, String> {
    &t.outcomes.last().expect("one outcome").info
}

fn nothing_sent(t: &Trace) -> bool {
    t.ais.is_empty() && t.tickets == 0
}

struct ScriptedSlot {
    claim: DedupCheck,
    reads: Mutex<VecDeque<Option<SlotHolder>>>,
    read_count: Mutex<usize>,
    read_keys: Mutex<Vec<DedupKey>>,
}

impl ScriptedSlot {
    fn new(claim: DedupCheck, reads: Vec<Option<SlotHolder>>) -> Arc<Self> {
        Arc::new(Self {
            claim,
            reads: Mutex::new(reads.into()),
            read_count: Mutex::new(0),
            read_keys: Mutex::new(Vec::new()),
        })
    }

    fn read_count(&self) -> usize {
        *self.read_count.lock().unwrap()
    }
}

#[async_trait::async_trait]
impl DedupSlot for ScriptedSlot {
    async fn try_insert(&self, _key: &DedupKey, _incoming: u8) -> DedupCheck {
        self.claim
    }
    async fn peek(&self, _key: &DedupKey, _incoming: u8) -> Option<u8> {
        None
    }
    async fn holder(&self, key: &DedupKey, _own: Option<u64>) -> Option<SlotHolder> {
        *self.read_count.lock().unwrap() += 1;
        self.read_keys.lock().unwrap().push(key.clone());
        let mut reads = self.reads.lock().unwrap();
        if reads.len() > 1 {
            reads.pop_front().flatten()
        } else {
            reads.front().copied().flatten()
        }
    }
    async fn update(&self, _key: &DedupKey, _v: &[u8], _class: u8, _n: Option<u64>) {}
    async fn update_skip(&self, _key: &DedupKey, _v: &[u8], _class: u8, _n: Option<u64>) {}
    async fn invalidate(&self, _key: &DedupKey, _n: Option<u64>) {}
}

const NEW: DedupCheck = DedupCheck::New {
    claim_nonce: Some(1),
};

fn displacing(held_class: u8) -> DedupCheck {
    DedupCheck::ClassBypass {
        held_class,
        held_in_flight: true,
        claim_nonce: Some(1),
    }
}

async fn run_on_slot(b: &TestBed, slot: Arc<ScriptedSlot>, score: &ScoreResult) -> Trace {
    let processor = ScoreResultProcessor {
        dedup_cache: slot,
        ..b.processor()
    };
    b.run_on(&processor, score).await
}

#[tokio::test]
async fn each_suspension_claims_at_its_class() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    for (score, class, sent) in [
        (ordinary(Account::Clean), CLASS_SUSPEND, sent_ordinary()),
        (
            permanent(Account::Clean),
            CLASS_SUSPEND_PERMANENT,
            sent_permanent(),
        ),
        (cse(Account::Clean), CLASS_SUSPEND_CSE, sent_cse()),
    ] {
        let t = b.run_fresh(&score).await;
        assert_eq!(t.status, "success", "{:?}", info(&t));
        assert_eq!(suspends(&t), sent);
        assert_eq!(t.dedup_calls[0], format!("try_insert(class={class})=new"));
        assert_eq!(info(&t)["acted_class"], class.to_string());
    }
}

#[tokio::test]
async fn no_other_policy_suspension_follows_a_cse_one() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    for (score, class) in [
        (ordinary(Account::Clean), CLASS_SUSPEND),
        (permanent(Account::Clean), CLASS_SUSPEND_PERMANENT),
    ] {
        let t = b.run_fresh(&cse(Account::Clean)).await;
        assert_eq!(suspends(&t), sent_cse(), "{:?}", info(&t));
        let t = b.run(&score).await;
        assert_eq!(t.status, "dedup_skipped");
        assert_eq!(
            t.dedup_calls[0],
            format!("try_insert(class={class})=duplicate")
        );
        assert!(nothing_sent(&t));
    }

    let (b, _) = bed_suspended(Account::Clean, true).await;
    for score in [ordinary(Account::Clean), permanent(Account::Clean)] {
        let t = b.run_fresh(&score).await;
        assert_eq!(t.status, ALREADY_PERMANENTLY_SUSPENDED, "{:?}", info(&t));
        assert!(nothing_sent(&t));
        assert_eq!(info(&t)[PERMANENT_SEEN_IN], PERMANENT_SEEN_IN_GIZMODUCK);
        assert_eq!(info(&t)["gizmoduck_permanently_suspended"], "true");
    }
}

#[tokio::test]
async fn a_cse_suspension_replaces_any_other_policy_suspension() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    for (score, sent) in [
        (ordinary(Account::Clean), sent_ordinary()),
        (permanent(Account::Clean), sent_permanent()),
    ] {
        let t = b.run_fresh(&score).await;
        assert_eq!(suspends(&t), sent, "{:?}", info(&t));
        let t = b.run(&cse(Account::Clean)).await;
        assert_eq!(t.status, "success", "{:?}", info(&t));
        assert_eq!(suspends(&t), sent_cse());
        assert_eq!(t.dedup_calls[0], "try_insert(class=5)=bypass");
    }

    for permanently in [false, true] {
        let (b, _) = bed_suspended(Account::Clean, permanently).await;
        let t = b.run_fresh(&cse(Account::Clean)).await;
        assert_eq!(t.status, "success", "{permanently}: {:?}", info(&t));
        assert_eq!(suspends(&t), sent_cse());
        assert!(!info(&t).contains_key(PERMANENT_SEEN_IN));
    }
}

#[tokio::test]
async fn a_permanent_suspension_still_outranks_an_ordinary_one() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    b.run_fresh(&permanent(Account::Clean)).await;
    let t = b.run(&ordinary(Account::Clean)).await;
    assert_eq!(t.status, "dedup_skipped");
    assert!(nothing_sent(&t));

    let t = b.run_fresh(&ordinary(Account::Clean)).await;
    assert_eq!(suspends(&t), sent_ordinary());
    let t = b.run(&permanent(Account::Clean)).await;
    assert_eq!(t.status, "success", "{:?}", info(&t));
    assert_eq!(suspends(&t), sent_permanent());
    assert_eq!(t.dedup_calls[0], "try_insert(class=4)=bypass");

    let (b, _) = bed_suspended(Account::Clean, false).await;
    let t = b.run_fresh(&permanent(Account::Clean)).await;
    assert_eq!(suspends(&t), sent_permanent(), "{:?}", info(&t));
}

#[tokio::test]
async fn a_permanent_entry_written_before_the_cse_class_still_holds() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    let id = Account::Clean.user_id();
    for score in [ordinary(Account::Clean), permanent(Account::Clean)] {
        b.dedup.slots.lock().unwrap().clear();
        b.dedup.hold(EntityType::User, id, CLASS_SUSPEND_PERMANENT);
        let t = b.run(&score).await;
        assert_eq!(t.status, "dedup_skipped");
        assert!(nothing_sent(&t));
    }
    b.dedup.slots.lock().unwrap().clear();
    b.dedup.hold(EntityType::User, id, CLASS_SUSPEND_PERMANENT);
    let t = b.run(&cse(Account::Clean)).await;
    assert_eq!(suspends(&t), sent_cse(), "{:?}", info(&t));
}

#[tokio::test]
async fn another_policy_suspension_racing_a_cse_one_waits_and_is_dropped() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    let in_flight = Some(SlotHolder::Claim(CLASS_SUSPEND_CSE));
    let sent = Some(SlotHolder::Outcome(CLASS_SUSPEND_CSE));
    for (score, mine) in [
        (ordinary(Account::Clean), sent_ordinary()),
        (permanent(Account::Clean), sent_permanent()),
    ] {
        let slot = ScriptedSlot::new(NEW, vec![in_flight, in_flight, sent]);
        let t = run_on_slot(&b, slot.clone(), &score).await;
        assert_eq!(t.status, ALREADY_CSE_SUSPENDED, "{:?}", info(&t));
        assert!(nothing_sent(&t));
        assert_eq!(info(&t)[PERMANENT_SEEN_IN], PERMANENT_SEEN_IN_DEDUP);
        assert_eq!(slot.read_count(), 3, "waited for the Cse one");

        let t = run_on_slot(&b, ScriptedSlot::new(NEW, vec![sent]), &score).await;
        assert_eq!(t.status, ALREADY_CSE_SUSPENDED);
        assert!(nothing_sent(&t));

        let slot = ScriptedSlot::new(NEW, vec![in_flight, Some(SlotHolder::Outcome(0))]);
        let t = run_on_slot(&b, slot, &score).await;
        assert_eq!(t.status, "success", "{:?}", info(&t));
        assert_eq!(suspends(&t), mine);
    }
}

#[tokio::test]
async fn a_cse_suspension_racing_another_policy_suspension_is_sent_last() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    for other in [CLASS_SUSPEND, CLASS_SUSPEND_PERMANENT] {
        let finished = Some(SlotHolder::Outcome(other));
        let slot = ScriptedSlot::new(
            displacing(other),
            vec![Some(SlotHolder::Own), Some(SlotHolder::Own), finished],
        );
        let t = run_on_slot(&b, slot.clone(), &cse(Account::Clean)).await;
        assert_eq!(t.status, "success", "{other}: {:?}", info(&t));
        assert_eq!(suspends(&t), sent_cse());
        assert_eq!(slot.read_count(), 3, "{other}: waited for the other one");

        let slot = ScriptedSlot::new(NEW, vec![Some(SlotHolder::Claim(other)), finished]);
        let t = run_on_slot(&b, slot.clone(), &cse(Account::Clean)).await;
        assert_eq!(suspends(&t), sent_cse(), "{other}");
        assert_eq!(slot.read_count(), 2, "{other}");
    }

    let slot = ScriptedSlot::new(
        displacing(CLASS_SUSPEND_PERMANENT),
        vec![Some(SlotHolder::Own)],
    );
    let started = std::time::Instant::now();
    let t = run_on_slot(&b, slot, &cse(Account::Clean)).await;
    assert_eq!(suspends(&t), sent_cse());
    assert!(started.elapsed() >= AWAIT_WEAKER);
}

#[tokio::test]
async fn a_permanent_suspension_racing_an_ordinary_one_is_sent_last() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    let in_flight = Some(SlotHolder::Claim(CLASS_SUSPEND_PERMANENT));
    let slot = ScriptedSlot::new(
        NEW,
        vec![
            in_flight,
            Some(SlotHolder::Outcome(CLASS_SUSPEND_PERMANENT)),
        ],
    );
    let t = run_on_slot(&b, slot.clone(), &ordinary(Account::Clean)).await;
    assert_eq!(t.status, ALREADY_PERMANENTLY_SUSPENDED, "{:?}", info(&t));
    assert!(nothing_sent(&t));
    assert_eq!(slot.read_count(), 2);

    let finished = Some(SlotHolder::Outcome(CLASS_SUSPEND));
    let slot = ScriptedSlot::new(
        displacing(CLASS_SUSPEND),
        vec![Some(SlotHolder::Own), finished],
    );
    let t = run_on_slot(&b, slot.clone(), &permanent(Account::Clean)).await;
    assert_eq!(suspends(&t), sent_permanent(), "{:?}", info(&t));
    assert_eq!(slot.read_count(), 2, "waited for the ordinary one");

    let over_outcome = DedupCheck::ClassBypass {
        held_class: CLASS_SUSPEND,
        held_in_flight: false,
        claim_nonce: Some(1),
    };
    let slot = ScriptedSlot::new(over_outcome, vec![Some(SlotHolder::Own)]);
    let t = run_on_slot(&b, slot.clone(), &permanent(Account::Clean)).await;
    assert_eq!(suspends(&t), sent_permanent());
    assert_eq!(slot.read_count(), 1);

    let slot = ScriptedSlot::new(
        displacing(CLASS_SUSPEND),
        vec![
            Some(SlotHolder::Claim(CLASS_SUSPEND_CSE)),
            Some(SlotHolder::Outcome(CLASS_SUSPEND_CSE)),
        ],
    );
    let t = run_on_slot(&b, slot, &permanent(Account::Clean)).await;
    assert_eq!(t.status, ALREADY_CSE_SUSPENDED, "{:?}", info(&t));
    assert!(nothing_sent(&t));
}

#[tokio::test]
async fn a_post_author_suspension_orders_against_the_author() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed().await;
    let in_flight = Some(SlotHolder::Claim(CLASS_SUSPEND_CSE));
    let sent = Some(SlotHolder::Outcome(CLASS_SUSPEND_CSE));
    let slot = ScriptedSlot::new(NEW, vec![in_flight, sent]);
    let t = run_on_slot(&b, slot.clone(), &post_author(Account::Clean)).await;
    assert_eq!(t.status, ALREADY_CSE_SUSPENDED, "{:?}", t.outcomes);
    assert!(nothing_sent(&t));
    assert_eq!(slot.read_count(), 2);
    assert!(
        slot.read_keys
            .lock()
            .unwrap()
            .iter()
            .all(|k| *k == DedupKey::entity(EntityType::User, Account::Clean.user_id())),
        "reads the author's slot"
    );

    let slot = ScriptedSlot::new(NEW, vec![Some(SlotHolder::Empty)]);
    let t = run_on_slot(&b, slot, &post_author(Account::Clean)).await;
    assert_eq!(t.status, "success", "{:?}", t.outcomes);
    assert_eq!(suspends(&t), sent_ordinary());
}

#[tokio::test]
async fn an_unreadable_store_never_lets_a_weaker_suspension_replace_a_stronger_one() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut b = bed().await;
    let down = Arc::new(FixedUsers {
        fail: true,
        ..Default::default()
    });
    b.ctx.gd = GizmoduckCoreClient::from_lookup(down.clone());
    for (held, score) in [
        (CLASS_SUSPEND_PERMANENT, ordinary(Account::Clean)),
        (CLASS_SUSPEND_CSE, permanent(Account::Clean)),
    ] {
        b.dedup.slots.lock().unwrap().clear();
        b.dedup
            .hold(EntityType::User, Account::Clean.user_id(), held);
        let t = b.run(&score).await;
        assert_eq!(t.status, "dedup_skipped");
        assert!(t.ais.is_empty());
        assert!(down.lookups.lock().unwrap().is_empty(), "no gizmoduck call");
    }

    let t = b.run_fresh(&ordinary(Account::Clean)).await;
    assert!(t.status.starts_with("error:"), "{}", t.status);
    assert!(t.ais.is_empty() && t.outcomes.is_empty());

    let b = bed().await;
    let t = run_on_slot(
        &b,
        ScriptedSlot::new(NEW, vec![None]),
        &ordinary(Account::Clean),
    )
    .await;
    assert_eq!(t.status, "success", "{:?}", info(&t));
    assert_eq!(suspends(&t), sent_ordinary());
}
