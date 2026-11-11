pub mod action_control;
pub mod action_control_rollback;
pub mod ais_client;
pub mod allowlist;
pub mod api_doc;
pub mod auth;
pub mod backfill;
pub mod backfill_claims;
pub mod confidence;
pub mod config;
pub mod decision;
pub mod dedup_cache;
pub mod entities;
pub mod facts;
pub mod gizmoduck;
pub mod gizmoduck_labels;
pub mod growthbook;
pub mod growthbook_writer;
pub mod head_guard;
#[cfg(test)]
mod ledger_contract_fixtures;
pub mod limiter;
pub mod manhattan;
pub mod metrics;
pub mod off_switch;
pub mod overturn_hold;
pub mod post_labels;
pub mod restart_on_config_change;
pub mod rules;
pub mod safety_checks;
pub mod service;
pub mod sliding_window;
pub mod strato;
pub mod test_user;

#[cfg(test)]
mod action_control_path_tests;
#[cfg(test)]
mod backfill_path_tests;
#[cfg(test)]
mod confidence_adversarial_tests;
#[cfg(test)]
mod confidence_path_tests;
#[cfg(test)]
mod replay_tests;
#[cfg(test)]
mod safety_audit_tests;

use xai_abuse_proto::enforcement as abuse_proto;

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;

use tokio::sync::Mutex;

use crate::dedup_cache::{DedupCheck, DedupKey, DedupSlot};
use anyhow::{Context, Result};
use axum::Router;

use futures::future::join_all;
use prost::Message;
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio::task::{JoinHandle, JoinSet};
use tracing::{Instrument, error, info, warn};
use xai_kafka::{
    BatchConsumerConfig, BatchResult, CancellationToken, KafkaBatchProcessor, KafkaConsumerConfig,
    KafkaConsumerConfigBuilder, KafkaMessage, KafkaProducer, KafkaProducerConfigBuilder,
    apply_auth_config, resolve_kafka_brokers, run_batch_consumer, self_delete_pod,
};

#[allow(deprecated)]
use xai_strato::{Strato, StratoClientConfig};

use crate::config::Config;
use crate::decision::Decision;
use crate::facts::{EntityFacts, EntityType, Facts, PostFacts, ScoreFacts, UserFacts};
use crate::growthbook::DynamicConfig;
use crate::growthbook_writer::GrowthBookWriter;
use crate::service::{
    AppState, EnforcementAction, PermanentAisFailure, enforce_actions, handle_allowlist_bulk,
    handle_allowlist_delete, handle_allowlist_delete_entity, handle_allowlist_get,
    handle_allowlist_get_entity, handle_allowlist_list, handle_allowlist_put,
    handle_allowlist_put_entity, handle_config_patch_field, handle_config_put, handle_dedup_get,
    handle_dedup_get_entity, handle_rate_limit, handle_rate_limit_increment,
    handle_rate_limit_reset,
};
use crate::strato::{
    fetch_cred, fetch_entity_allowlist, fetch_uas, fetch_user, fetch_user_allowlist,
};

const SERVICE_NAME: &str = env!("CARGO_PKG_NAME");


#[derive(Debug, Deserialize)]
struct TopicEntry {
    #[serde(default)]
    cluster: Option<String>,
    #[serde(default)]
    zone: Option<String>,
    #[serde(flatten)]
    processor: TopicConfig,
}

impl TopicEntry {
            fn cluster<'a>(&'a self, default: &'a str) -> &'a str {
        override_or_default(self.cluster.as_deref(), default)
    }

            fn zone<'a>(&'a self, default: &'a str) -> &'a str {
        override_or_default(self.zone.as_deref(), default)
    }
}

fn override_or_default<'a>(value: Option<&'a str>, default: &'a str) -> &'a str {
    match value.map(str::trim) {
        Some(v) if !v.is_empty() => v,
        _ => default,
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "processor", rename_all = "snake_case")]
enum TopicConfig {
            ScoreResult {
                                #[serde(default)]
        labels: HashSet<String>,
                        #[serde(default)]
        negative_labels: HashSet<String>,
    },
            StatsOnly {
        #[serde(default)]
        should_log_content: bool,
    },
}


const MAX_RETRIES: u32 = 5;

struct RetryEntry {
    score: abuse_proto::ScoreResult,
    attempt: u32,
    next_retry_at: tokio::time::Instant,
        topic: String,
        dedup_cache: Arc<dyn DedupSlot>,
                    claim_nonce: Option<u64>,
                decision: DecisionContext,
}

#[derive(Debug, Clone, Default)]
struct DecisionContext {
    control: action_control::Resolved,
            would_dedup_held: Option<u8>,
                held_class: u8,
                global_dry_run: bool,
            run: Option<Arc<backfill::BackfillRunRecord>>,
                run_nonce: i64,
}

impl DecisionContext {
                    fn records_only(&self) -> bool {
        self.global_dry_run || self.control.record_only()
    }

            fn dedup_key(&self, entity_type: EntityType, entity_id: i64) -> DedupKey {
        DedupKey::entity(entity_type, entity_id)
            .with_post_labels(self.control.dedup_labels().as_deref())
    }

        fn claimed(&self) -> bool {
        !self.records_only()
    }
}

#[derive(Debug)]
struct EnforcementFailed {
    permanent: bool,
    claim_nonce: Option<u64>,
    decision: DecisionContext,
    source: anyhow::Error,
}

impl PartialEq for RetryEntry {
    fn eq(&self, other: &Self) -> bool {
        self.next_retry_at == other.next_retry_at
    }
}
impl Eq for RetryEntry {}
impl PartialOrd for RetryEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RetryEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.next_retry_at.cmp(&self.next_retry_at)
    }
}

#[derive(Clone)]
#[allow(deprecated)]
struct EnforcementCtx {
    ais_client: ais_client::AisClient,
    gd: gizmoduck::GizmoduckCoreClient,
    cred_clients: strato::CredClients,
    uas_strato: Arc<Strato>,
    dynamic_config: DynamicConfig,
            rules_cache: Arc<rules::RulesCache>,
    allowlist: Option<allowlist::ManhattanAllowlist>,
            kafka_producer_decisions: Option<Arc<KafkaProducer>>,
                kafka_producer_decisions_json: Option<Arc<KafkaProducer>>,
                            hold_gate: Arc<overturn_hold::HoldGate>,
            off_switch: Arc<off_switch::OffSwitch>,
            head_guard: Arc<head_guard::HeadGuard>,
            action_control: Arc<action_control::ActionControl>,
            backfill: Arc<backfill::Backfill>,
            post_labels: Option<Arc<dyn post_labels::PostLabelReader>>,
        #[cfg(test)]
    outcome_log: Option<Arc<std::sync::Mutex<Vec<abuse_proto::DecisionOutcome>>>>,
}

fn gated_actions(
    specs: &[decision::ActionSpec],
    cfg: &overturn_hold::HoldGateConfig,
) -> Vec<overturn_hold::GatedAction> {
    use overturn_hold::GatedAction;
    let mut out = Vec::new();
    if cfg.gates_suspend() {
        let mut perm: Option<bool> = None;
        for spec in specs {
            if let decision::ActionSpec::SuspendUser { perm: p, .. } = spec {
                let all_perm = perm.get_or_insert(true);
                *all_perm &= *p;
            }
        }
        if let Some(perm) = perm {
            out.push(GatedAction::Suspend { perm });
        }
    }
    for spec in specs {
        if let decision::ActionSpec::AddLabelsV2 { labels, .. } = spec {
            for name in labels {
                let already = out
                    .iter()
                    .any(|a| matches!(a, GatedAction::Label { name: n } if n == name));
                if cfg.gates_label(name) && !already {
                    out.push(GatedAction::Label { name: name.clone() });
                }
            }
        }
    }
    out
}

fn strip_held_labels(specs: &mut Vec<decision::ActionSpec>, held: &[String]) {
    if held.is_empty() {
        return;
    }
    for spec in specs.iter_mut() {
        if let decision::ActionSpec::AddLabelsV2 { labels, .. } = spec {
            labels.retain(|l| !held.contains(l));
        }
    }
    specs.retain(
        |s| !matches!(s, decision::ActionSpec::AddLabelsV2 { labels, .. } if labels.is_empty()),
    );
}

struct ScoreResultProcessor {
    topic: String,
    labels: HashSet<String>,
    negative_labels: HashSet<String>,
    ctx: EnforcementCtx,
    dedup_cache: Arc<dyn DedupSlot>,
    retry_queue: Arc<Mutex<BinaryHeap<RetryEntry>>>,
        last_progress: Arc<AtomicI64>,
        max_in_flight: usize,
                backfill_topic: bool,
}


fn decision_outcome(
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
    status: String,
    dry_run: bool,
    info: BTreeMap<String, String>,
) -> abuse_proto::DecisionOutcome {
    let summary = score.summary.as_ref();
    let (entity_type, entity_id) = Facts::resolve_target(score);
    abuse_proto::DecisionOutcome {
        decided_at_ms: now_millis(),
        source_topic: source_topic.to_owned(),
        entity_type: entity_type.as_str().to_owned(),
        entity_id,
        model_version: score.model_version.clone(),
        status,
        dry_run,
        info,
        head: crate::facts::head_label(score).to_owned(),
        fired_heads: summary.map(|s| s.fired_heads.clone()).unwrap_or_default(),
        labels: summary.map(|s| s.labels.clone()).unwrap_or_default(),
        score_id: score.score_id.clone(),
        confidence: score.confidence,
        raw_score: score.raw_score,
        score: score.score,
        calibration_id: score.calibration_id.clone(),
        confidence_evidence: score.confidence_evidence.clone(),
        in_reply_to_user_id: score.in_reply_to_user_id,
        in_reply_to_post_id: score.in_reply_to_post_id,
    }
}

const ALREADY_SUSPENDED: &str = "already_suspended";

const ALREADY_SUSPENDED_STRIPPED: &str = "already_suspended_stripped";

const ALREADY_POST_LABELED: &str = "already_post_labeled";

const POST_LABELS_CARRIED: &str = "post_labels_carried";

const POST_LABELS_READ: &str = "post_labels_read";

async fn drop_carried_post_labels(
    reader: Option<&dyn post_labels::PostLabelReader>,
    facts: &mut Facts,
    specs: &mut Vec<decision::ActionSpec>,
    extra: &mut BTreeMap<String, String>,
) -> bool {
    let Some(reader) = reader else {
        metrics::POST_LABEL_READ_TOTAL
            .with_label_values(&["unavailable"])
            .inc();
        extra.insert(POST_LABELS_READ.into(), "unavailable".into());
        return false;
    };
    let now_ms = now_millis();
    let carried = match reader.labels(facts.entity_id, now_ms).await {
        Ok(Some(carried)) => {
            metrics::POST_LABEL_READ_TOTAL
                .with_label_values(&["ok"])
                .inc();
            carried
        }
        Ok(None) => {
            metrics::POST_LABEL_READ_TOTAL
                .with_label_values(&["not_found"])
                .inc();
            return false;
        }
        Err(e) => {
            warn!(
                post_id = facts.entity_id,
                "post label read failed: {e:#}; sending every label"
            );
            metrics::POST_LABEL_READ_TOTAL
                .with_label_values(&["failed"])
                .inc();
            extra.insert(POST_LABELS_READ.into(), "failed".into());
            return false;
        }
    };
    if let EntityFacts::Post(post) = &mut facts.entity {
        post.present = true;
        post.labels = carried.keys().cloned().collect();
    }
    let dropped = post_labels::drop_carried(specs, &carried, now_ms);
    if !dropped.is_empty() {
        extra.insert(
            POST_LABELS_CARRIED.into(),
            serde_json::to_string(&dropped).unwrap_or_else(|_| "[]".into()),
        );
    }
    specs.is_empty()
}

fn strip_redundant_suspends(
    specs: &mut Vec<decision::ActionSpec>,
    user: &crate::facts::GizmoduckFacts,
) -> Vec<decision::ActionSpec> {
    let is_suspend =
        |s: &decision::ActionSpec| matches!(s, decision::ActionSpec::SuspendUser { .. });
    let has_perm = specs
        .iter()
        .any(|s| matches!(s, decision::ActionSpec::SuspendUser { perm: true, .. }));
    if !user.suspended || has_perm || !specs.iter().any(is_suspend) {
        return Vec::new();
    }
    let (stripped, kept) = std::mem::take(specs).into_iter().partition(is_suspend);
    *specs = kept;
    stripped
}

fn incoming_dedup_class(
    dynamic_config: &DynamicConfig,
    rules_cache: &rules::RulesCache,
    score: &abuse_proto::ScoreResult,
) -> u8 {
    let (entity_type, _) = Facts::resolve_target(score);
    let rules_override = dynamic_config.enforcement_rules_yaml(entity_type);
    let compiled = rules_cache.resolve(entity_type, rules_override.as_ref());
    let score = ScoreFacts::from_score(score);
    let Ok(reachable) = rules::reachable_decisions(&compiled, &score) else {
        return dedup_cache::CLASS_NONE;
    };
    reachable
        .into_iter()
        .flat_map(|decision| match decision {
            Decision::Act(specs) => specs,
            Decision::Skip(_) => Vec::new(),
        })
        .map(|spec| spec.dedup_action_class())
        .max()
        .unwrap_or(dedup_cache::CLASS_NONE)
}

#[tracing::instrument(skip_all, fields(dry_run, uas))]
async fn run_enforcement_inner(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
    control: &mut action_control::Resolved,
    held_class: u8,
    pinned_global_dry_run: bool,
    extra: &mut BTreeMap<String, String>,
    run: Option<(&backfill::BackfillRunRecord, i64)>,
) -> Result<abuse_proto::DecisionOutcome> {
    let user_id = score.user_id;
    let (entity_type, entity_id) = Facts::resolve_target(score);
    let global_dry_run = pinned_global_dry_run || ctx.dynamic_config.is_dry_run();
    let mut dry_run = global_dry_run || control.record_only();
    if dry_run && let Some(source) = control.dry_run_source(global_dry_run) {
        extra.insert("dry_run_source".into(), source.into());
    }
    let span = tracing::Span::current();
    span.record("dry_run", dry_run);

    let gated_user_id = match entity_type {
        EntityType::User => entity_id,
        EntityType::Post => user_id,
    };
    let unfetched = || Facts {
        entity_type,
        entity_id,
        user_id: gated_user_id,
        topic: source_topic.to_owned(),
        score: ScoreFacts::from_score(score),
        entity: match entity_type {
            EntityType::User => EntityFacts::User(UserFacts::default()),
            EntityType::Post => EntityFacts::Post(PostFacts {
                author_id: user_id,
                ..Default::default()
            }),
        },
        uas: None,
    };
    if test_user::is_test_user_id(gated_user_id) {
        info!(
            user_id = gated_user_id,
            "test user (id bitmask/range); skipping enforcement"
        );
        return Ok(skip_outcome(
            "test_user_skipped".into(),
            dry_run,
            &unfetched(),
            score,
        ));
    }

    if control.rules_unread() && control.page_actions.is_empty() {
        record_seen_pair(&ctx.off_switch, &ctx.dynamic_config, score, source_topic);
        record_seen_bands(ctx, score, entity_type, None, control.factory);
        extra.insert(
            "matched_rule_id".into(),
            action_control::PAGE_RULE_ID.into(),
        );
        return Ok(skip_outcome(
            action_control::STATUS_NO_ACTIONS.into(),
            dry_run,
            &unfetched(),
            score,
        ));
    }

    let mut facts = match entity_type {
        EntityType::User => {
            let user_id = entity_id;
            let allowlist = fetch_user_allowlist(ctx.allowlist.as_ref(), user_id).await;
            if allowlist.is_allowlisted {
                let partial = Facts {
                    entity_type,
                    entity_id,
                    user_id,
                    topic: source_topic.to_owned(),
                    score: ScoreFacts::from_score(score),
                    entity: EntityFacts::User(UserFacts {
                        allowlist,
                        ..Default::default()
                    }),
                    uas: None,
                };
                return Ok(skip_outcome(
                    "user_in_allowlist".into(),
                    dry_run,
                    &partial,
                    score,
                ));
            }

            let (user_res, cred_res) = tokio::join!(
                fetch_user(&ctx.gd, user_id),
                fetch_cred(&ctx.cred_clients, user_id),
            );

            Facts {
                entity_type,
                entity_id,
                user_id,
                topic: source_topic.to_owned(),
                score: ScoreFacts::from_score(score),
                entity: EntityFacts::User(UserFacts {
                    allowlist,
                    user: user_res?,
                    cred: cred_res?,
                }),
                uas: None,
            }
        }
        EntityType::Post => {
            let author_id = user_id;
            let (post_allowlist, author_allowlist) = tokio::join!(
                fetch_entity_allowlist(ctx.allowlist.as_ref(), EntityType::Post, entity_id),
                fetch_user_allowlist(ctx.allowlist.as_ref(), author_id),
            );
            if post_allowlist.is_allowlisted || author_allowlist.is_allowlisted {
                let reason = if post_allowlist.is_allowlisted {
                    "post_in_allowlist"
                } else {
                    "user_in_allowlist"
                };
                let partial = Facts {
                    entity_type,
                    entity_id,
                    user_id: author_id,
                    topic: source_topic.to_owned(),
                    score: ScoreFacts::from_score(score),
                    entity: EntityFacts::Post(PostFacts {
                        allowlist: post_allowlist,
                        present: false,
                        author_id,
                        labels: Vec::new(),
                        author: UserFacts {
                            allowlist: author_allowlist,
                            ..Default::default()
                        },
                    }),
                    uas: None,
                };
                return Ok(skip_outcome(reason.into(), dry_run, &partial, score));
            }

            let (user_res, cred_res) = tokio::join!(
                fetch_user(&ctx.gd, author_id),
                fetch_cred(&ctx.cred_clients, author_id),
            );

            Facts {
                entity_type,
                entity_id,
                user_id: author_id,
                topic: source_topic.to_owned(),
                score: ScoreFacts::from_score(score),
                entity: EntityFacts::Post(PostFacts {
                    allowlist: post_allowlist,
                    present: false,
                    author_id,
                    labels: Vec::new(),
                    author: UserFacts {
                        allowlist: author_allowlist,
                        user: user_res?,
                        cred: cred_res?,
                    },
                }),
                uas: None,
            }
        }
    };

    record_seen_pair(&ctx.off_switch, &ctx.dynamic_config, score, &facts.topic);
    let compiled_rules = || {
        ctx.rules_cache.resolve(
            facts.entity_type,
            ctx.dynamic_config
                .enforcement_rules_yaml(facts.entity_type)
                .as_ref(),
        )
    };
    let rules_decision = if control.page_driven() && !control.rules_unread() {
        match crate::rules::decide_with_id(&compiled_rules(), &facts) {
            Ok(d) => Some(d),
            Err(e) => {
                warn!(
                    user_id,
                    topic = %facts.topic,
                    error = %e,
                    "action control: rules eval errored on a page decision",
                );
                None
            }
        }
    } else {
        None
    };
    let (decision, rule_id) = if control.page_driven() {
        if let Some((rules_would, id)) = &rules_decision {
            extra.insert("rules_would_rule_id".into(), id.clone());
            if let Decision::Act(specs) = rules_would {
                extra.insert(
                    "rules_would_action_specs".into(),
                    action_control::action_specs_json(specs),
                );
            }
        }
        if control.recording() && !dry_run {
            dry_run = true;
            extra.insert(
                "dry_run_source".into(),
                control
                    .dry_run_source(global_dry_run)
                    .unwrap_or("page")
                    .into(),
            );
        }
        match safety_checks::first_page_stop(facts.entity_type, &facts, control.author_skips) {
            Ok(Some(check)) => (Decision::Skip(check.reason.to_owned()), check.id.to_owned()),
            Ok(None) if control.page_actions.is_empty() => (
                Decision::Skip(action_control::STATUS_NO_ACTIONS.to_owned()),
                action_control::PAGE_RULE_ID.to_owned(),
            ),
            Ok(None) => (
                Decision::Act(action_control::page_specs(&control.page_actions)),
                action_control::PAGE_RULE_ID.to_owned(),
            ),
            Err(e) => {
                warn!(
                    user_id,
                    topic = %facts.topic,
                    error = %e,
                    "action control: safety check eval errored — skipping",
                );
                extra.insert("matched_rule_id".into(), "safety_checks".into());
                return Ok(skip_outcome(
                    "rule_eval_error".into(),
                    dry_run,
                    &facts,
                    score,
                ));
            }
        }
    } else {
        match crate::rules::decide_with_id(&compiled_rules(), &facts) {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    user_id,
                    topic = %facts.topic,
                    error = %e,
                    "rule pipeline eval errored — skipping; investigate the rule file",
                );
                return Ok(skip_outcome(
                    "rule_eval_error".into(),
                    dry_run,
                    &facts,
                    score,
                ));
            }
        }
    };
    let own = (decision.clone(), rule_id.clone());
    let seen_rules = match &rules_decision {
        Some(r) => Some(r),
        None if !control.page_driven() => Some(&own),
        None => None,
    };
    if control.rules_unread() {
        record_seen_bands(ctx, score, facts.entity_type, None, control.factory);
    } else if let Some(rules) = seen_rules {
        record_seen_bands(ctx, score, facts.entity_type, Some(rules), false);
    }
    extra.insert("matched_rule_id".into(), rule_id);

    match decision {
        Decision::Skip(reason) => Ok(skip_outcome(reason, dry_run, &facts, score)),
        Decision::Act(mut specs) => {
            if control.page_driven()
                && facts.entity_type == EntityType::User
                && let Some(label) = action_control::label_only(&control.page_actions)
                && facts.user().labels.iter().any(|l| l == label)
            {
                let reason = if label == gizmoduck_labels::CLIENT_EVENT_PHOENIX_SKIP {
                    "already_phoenix_skip_labeled"
                } else {
                    "already_spam_high_recall_labeled"
                };
                return Ok(skip_outcome(reason.into(), dry_run, &facts, score));
            }
            if control.page_driven()
                && facts.entity_type == EntityType::Post
                && post_labels::sent_labels(&specs).next().is_some()
                && drop_carried_post_labels(
                    ctx.post_labels.as_deref(),
                    &mut facts,
                    &mut specs,
                    extra,
                )
                .await
            {
                return Ok(skip_outcome(
                    ALREADY_POST_LABELED.into(),
                    dry_run,
                    &facts,
                    score,
                ));
            }
            if control.page_driven() {
                let account = facts.user();
                let upgrades = control.page_actions.iter().any(|a| {
                    matches!(
                        a,
                        action_control::PageAction::Suspend {
                            permanent: true,
                            ..
                        }
                    )
                });
                let skip = if account.suspended && !upgrades {
                    Some(ALREADY_SUSPENDED)
                } else if account.deactivated {
                    Some("already_deactivated")
                } else {
                    None
                };
                if let Some(reason) = skip {
                    return Ok(skip_outcome(reason.into(), dry_run, &facts, score));
                }
            }

            let stripped = strip_redundant_suspends(&mut specs, facts.user());
            let mut already_suspended_stripped = None;
            if !stripped.is_empty() {
                let names: Vec<&'static str> = stripped
                    .iter()
                    .map(|spec| {
                        EnforcementAction::from_spec(
                            spec,
                            facts.entity_id,
                            facts.user_id,
                            Vec::new(),
                        )
                        .name()
                    })
                    .collect();
                let names = serde_json::to_string(&names).unwrap_or_else(|_| "[]".into());
                if dry_run {
                    info!(
                        user_id = facts.user_id,
                        actions = %names,
                        "dry run; would call suspendUser, but the account is already suspended"
                    );
                }
                if specs.is_empty() {
                    info!(
                        user_id = facts.user_id,
                        "suspend target already suspended; skipping decision"
                    );
                    let mut outcome =
                        skip_outcome(ALREADY_SUSPENDED.into(), dry_run, &facts, score);
                    outcome
                        .info
                        .insert(ALREADY_SUSPENDED_STRIPPED.into(), names);
                    return Ok(outcome);
                }
                info!(
                    user_id = facts.user_id,
                    stripped = %names,
                    "suspend target already suspended; dropping the suspend, keeping the other actions"
                );
                already_suspended_stripped = Some(names);
            }

            // Overturn-hold gate: skip a suspend, or strip a label, that a
            // human reviewer overturned on appeal while that hold is still active.
            let mut hold_gate_info = BTreeMap::new();
            let live_gate_cfg = ctx.dynamic_config.overturn_hold_gate();
            let (gate_cfg, run_unchecked) =
                match run.map(|_| backfill::run_gate_cfg(&live_gate_cfg)) {
                    None => (live_gate_cfg, false),
                    Some(Some(cfg)) => (cfg, false),
                    Some(None) => (live_gate_cfg, true),
                };
            let gated = gated_actions(&specs, &gate_cfg);
            if run_unchecked && !gated.is_empty() {
                return Ok(hold_gate_skip_outcome(
                    overturn_hold::STATUS_HOLD_LOOKUP_FAILED,
                    BTreeMap::from([(
                        "overturn_hold_probe_error".to_owned(),
                        "gate_off".to_owned(),
                    )]),
                    dry_run,
                    &facts,
                    score,
                ));
            }
            if !gated.is_empty() {
                let mut gate = ctx
                    .hold_gate
                    .evaluate(&gate_cfg, facts.user_id, source_topic, &gated)
                    .await;
                if let Some(status) = gate.skip_status {
                    return Ok(hold_gate_skip_outcome(
                        status, gate.info, dry_run, &facts, score,
                    ));
                }
                if !gate.strip_labels.is_empty() {
                    strip_held_labels(&mut specs, &gate.strip_labels);
                    if specs.is_empty() {
                        gate.mark_label_collapse();
                        info!(
                            user_id = facts.user_id,
                            topic = source_topic,
                            labels_stripped = %gate.info
                                .get("overturn_hold_labels_stripped")
                                .map(String::as_str)
                                .unwrap_or(""),
                            "overturn-hold gate (enforce): every action was a held label; skipping decision"
                        );
                        return Ok(hold_gate_skip_outcome(
                            overturn_hold::STATUS_HOLD_OVERTURNED,
                            gate.info,
                            dry_run,
                            &facts,
                            score,
                        ));
                    }
                }
                hold_gate_info = gate.info;
            }

            let chosen_class = specs
                .iter()
                .map(decision::ActionSpec::dedup_action_class)
                .max()
                .unwrap_or(dedup_cache::CLASS_NONE);
            if !dry_run && chosen_class <= held_class {
                info!(
                    user_id = facts.user_id,
                    held_class, chosen_class, "dedup: chosen actions do not outrank the held entry"
                );
                let mut info = std::mem::take(&mut hold_gate_info);
                info.insert("dedup_held_class".into(), held_class.to_string());
                info.insert(
                    "action_specs".into(),
                    action_control::action_specs_json(&specs),
                );
                return Ok(hold_gate_skip_outcome(
                    "dedup_skipped",
                    info,
                    dry_run,
                    &facts,
                    score,
                ));
            }

            if let Some(outcome) = final_checks(
                &ctx.off_switch,
                &ctx.head_guard,
                &ctx.dynamic_config,
                &specs,
                &facts,
                score,
                source_topic,
                dry_run,
                &mut hold_gate_info,
                run.map(|(r, nonce)| (ctx.backfill.as_ref(), r, nonce)),
            )
            .await?
            {
                return Ok(outcome);
            }

            let user_id = facts.user_id;
            let entity_id = facts.entity_id;

            facts.uas = fetch_uas(&ctx.uas_strato, user_id).await;
            if let Some(uas) = &facts.uas {
                span.record("uas", tracing::field::display(uas));
            }

            info!(
                "processing score_result: model={}, status={}",
                score.model_version, score.score_status,
            );

            let mut additional_info_map = facts.to_additional_info_map();
            let ais_notes: Vec<String> = additional_info_map
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect();
            let acted_class = specs
                .iter()
                .map(decision::ActionSpec::dedup_action_class)
                .max()
                .unwrap_or(dedup_cache::CLASS_NONE);
            let actions: Vec<EnforcementAction> = specs
                .iter()
                .map(|spec| {
                    EnforcementAction::from_spec(spec, entity_id, user_id, ais_notes.clone())
                })
                .collect();
            let uas_action_dims: Vec<(&'static str, String)> = actions
                .iter()
                .map(|a| (a.name(), a.metric_label()))
                .collect();

            additional_info_map.insert(
                "action_kinds".into(),
                serde_json::to_string(&uas_action_dims.iter().map(|(a, _)| *a).collect::<Vec<_>>())
                    .unwrap_or_else(|_| "[]".into()),
            );
            additional_info_map.extend(hold_gate_info);
            if let Some(names) = already_suspended_stripped {
                additional_info_map.insert(ALREADY_SUSPENDED_STRIPPED.into(), names);
            }
            additional_info_map.insert(
                "action_specs".into(),
                action_control::action_specs_json(&specs),
            );

            enforce_actions(
                &ctx.ais_client,
                dry_run,
                Some(score),
                source_topic,
                facts.entity_type,
                actions,
            )
            .await?;
            info!("enforcement complete");
            let status = if dry_run { "dry_run" } else { "success" }.to_string();
            if !dry_run {
                additional_info_map.insert("acted_class".into(), acted_class.to_string());
            }
            if let Some(uas) = &facts.uas {
                let dims: Vec<(&str, &str)> = uas_action_dims
                    .iter()
                    .map(|(a, l)| (*a, l.as_str()))
                    .collect();
                metrics::observe_user_active_seconds(uas, facts.entity_type, &status, &dims);
            }
            Ok(decision_outcome(
                score,
                source_topic,
                status,
                dry_run,
                additional_info_map,
            ))
        }
    }
}

fn record_seen_pair(
    off_switch: &off_switch::OffSwitch,
    dynamic_config: &DynamicConfig,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
) -> Option<off_switch::SeenOutcome> {
    let head = crate::facts::head_label(score);
    if head.is_empty() {
        metrics::OFF_SWITCH_HEADLESS_TOTAL
            .with_label_values(&[source_topic, score.model_version.as_str()])
            .inc();
    }
    if !dynamic_config.off_switch_enabled() {
        return None;
    }
    Some(off_switch.record_seen(head, &score.model_version, now_millis()))
}

fn record_seen_bands(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    entity_type: EntityType,
    rules: Option<&(Decision, String)>,
    factory: bool,
) {
    let head = crate::facts::head_label(score);
    if head.is_empty() {
        return;
    }
    let mut bands: Vec<&str> = score
        .summary
        .as_ref()
        .map(|s| {
            s.labels
                .iter()
                .map(|l| action_control::strip_dry_run_prefix(l))
                .take(4)
                .collect()
        })
        .unwrap_or_default();
    if bands.is_empty() {
        bands.push("-");
    }
    let specs: &[decision::ActionSpec] = match rules {
        Some((Decision::Act(specs), _)) => specs,
        _ => &[],
    };
    let classifier = if score.model_version.is_empty() {
        "-"
    } else {
        score.model_version.as_str()
    };
    ctx.action_control
        .record_seen_row(action_control::ActionControlSeenRow {
            entity_type: action_control::entity_of(entity_type),
            head: head.to_owned(),
            classifier: classifier.to_owned(),
            row: confidence::row_of(score.confidence).0,
            last_seen_ms: now_millis(),
        });
    for band in bands {
        ctx.action_control
            .record_seen(action_control::ActionControlSeen {
                head: head.to_owned(),
                classifier: classifier.to_owned(),
                band: band.to_owned(),
                entity_type: action_control::entity_of(entity_type),
                factory,
                last_seen_ms: now_millis(),
                rule_id: rules.map(|r| r.1.clone()).unwrap_or_default(),
                action_specs: action_control::action_specs_value(specs),
            });
    }
}

#[allow(clippy::too_many_arguments)]
async fn final_checks(
    off_switch: &off_switch::OffSwitch,
    head_guard: &head_guard::HeadGuard,
    dynamic_config: &DynamicConfig,
    specs: &[decision::ActionSpec],
    facts: &Facts,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
    dry_run: bool,
    hold_gate_info: &mut BTreeMap<String, String>,
    run: Option<(&backfill::Backfill, &backfill::BackfillRunRecord, i64)>,
) -> Result<Option<abuse_proto::DecisionOutcome>> {
    let head = crate::facts::head_label(score);
    let classifier = score.model_version.as_str();
    let switched_off = |keys: Vec<(String, String)>, hold_gate_info: &mut BTreeMap<_, _>| {
        let dropped: Vec<&'static str> = specs
            .iter()
            .map(|spec| {
                EnforcementAction::from_spec(spec, facts.entity_id, facts.user_id, Vec::new())
                    .name()
            })
            .collect();
        let mut info = std::mem::take(hold_gate_info);
        info.extend(keys);
        info.insert(
            "off_switch_dropped_actions".into(),
            serde_json::to_string(&dropped).unwrap_or_else(|_| "[]".into()),
        );
        info.insert(
            "action_specs".into(),
            action_control::action_specs_json(specs),
        );
        hold_gate_skip_outcome(off_switch::STATUS_SWITCHED_OFF, info, dry_run, facts, score)
    };
    let guards_on = dynamic_config.off_switch_enabled();
    if guards_on && let Some(hit) = off_switch.lookup(head, classifier, now_millis()) {
        metrics::OFF_SWITCH_TOTAL
            .with_label_values(&[source_topic, head, hit.scope.as_str()])
            .inc();
        info!(
            user_id = facts.user_id,
            head,
            classifier,
            scope = hit.scope.as_str(),
            updated_by = %hit.record.updated_by,
            "enforcement off switch: skipping decision"
        );
        return Ok(Some(switched_off(hit.info_keys(), hold_gate_info)));
    }

    if dry_run {
        if let Some((used, cap, _)) = dynamic_config.rate_limit_stats(facts.entity_type) {
            hold_gate_info.insert("would_rate_cap".into(), (used >= cap).to_string());
        }
        return Ok(None);
    }
    if let Some((bf, run, nonce)) = run {
        return run_checks(bf, run, nonce, facts, score).await;
    }
    if guards_on
        && let Some(r) = head_guard
            .check(off_switch, head, classifier, now_millis())
            .await
    {
        info!(
            user_id = facts.user_id,
            head,
            classifier,
            window = r.window,
            limit = r.limit,
            "head guard: action ceiling reached; skipping decision"
        );
        return Ok(Some(switched_off(r.info_keys(), hold_gate_info)));
    }
    if !dynamic_config.try_enforce(facts.entity_type).await {
        warn!(
            entity_type = facts.entity_type.as_str(),
            "max enforcement has been reached; skipping"
        );
        return Ok(Some(skip_outcome(
            "max_enforcement_reached".into(),
            dry_run,
            facts,
            score,
        )));
    }
    if guards_on
        && let Some(r) = head_guard
            .count(off_switch, head, classifier, now_millis())
            .await
    {
        info!(
            user_id = facts.user_id,
            head,
            classifier,
            window = r.window,
            limit = r.limit,
            "head guard: past the action ceiling; skipping decision"
        );
        return Ok(Some(switched_off(r.info_keys(), hold_gate_info)));
    }
    Ok(None)
}

async fn run_checks(
    bf: &backfill::Backfill,
    run: &backfill::BackfillRunRecord,
    nonce: i64,
    facts: &Facts,
    score: &abuse_proto::ScoreResult,
) -> Result<Option<abuse_proto::DecisionOutcome>> {
    let skip = |status: &str| Ok(Some(skip_outcome(status.into(), false, facts, score)));
    let Some(now) = bf.run(&run.run_id).await? else {
        return skip(backfill::STATUS_RUN_UNKNOWN);
    };
    if let Err(status) = backfill::gate(&now, now_millis()) {
        return skip(status);
    }
    let claims = bf
        .claims()
        .ok_or_else(|| anyhow::anyhow!("no backfill claim store on this pod"))?;
    let key = backfill_claims::ClaimKey {
        run_id: run.run_id.clone(),
        entity: facts.entity_type,
        entity_id: facts.entity_id,
    };
    match claims.state(&key).await? {
        backfill_claims::ItemClaim::Finished { .. } => {
            return skip(backfill::STATUS_ALREADY_FINISHED);
        }
        backfill_claims::ItemClaim::Handed { nonce: holder, age }
            if holder != nonce
                && age >= backfill_claims::STALE_AFTER
                && backfill::already_applied(&run.actions, facts.user()) =>
        {
            return skip(backfill::STATUS_ALREADY_APPLIED);
        }
        _ => {}
    }
    let budget = bf.config().budget(facts.entity_type);
    match claims.take(&key, nonce, budget, bf.pod()).await? {
        backfill_claims::Take::Taken => Ok(None),
        backfill_claims::Take::Held => skip(backfill::STATUS_CLAIM_TAKEN),
        backfill_claims::Take::BudgetReached => skip(backfill::STATUS_BUDGET_REACHED),
        backfill_claims::Take::NotInList => skip(backfill::STATUS_NOT_IN_RUN),
    }
}

fn skip_outcome(
    reason: String,
    dry_run: bool,
    facts: &Facts,
    score: &abuse_proto::ScoreResult,
) -> abuse_proto::DecisionOutcome {
    decision_outcome(
        score,
        &facts.topic,
        reason,
        dry_run,
        facts.to_additional_info_map(),
    )
}

fn hold_gate_skip_outcome(
    status: &str,
    gate_info: BTreeMap<String, String>,
    dry_run: bool,
    facts: &Facts,
    score: &abuse_proto::ScoreResult,
) -> abuse_proto::DecisionOutcome {
    let mut outcome = skip_outcome(status.to_owned(), dry_run, facts, score);
    outcome.info.extend(gate_info);
    outcome
}

#[cfg(test)]
fn outcome_holds_full_dedup(status: &str) -> bool {
    dedup_retention_for(status) == DedupRetention::Full
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DedupRetention {
        Full,
            Skip,
                Release,
}

fn dedup_retention_for(status: &str) -> DedupRetention {
    if status == "success" {
        DedupRetention::Full
    } else if status == overturn_hold::STATUS_HOLD_LOOKUP_FAILED {
        DedupRetention::Release
    } else {
        DedupRetention::Skip
    }
}

async fn write_dedup_outcome(
    dedup_cache: &dyn DedupSlot,
    key: &DedupKey,
    score: &abuse_proto::ScoreResult,
    topic: &str,
    outcome: &abuse_proto::DecisionOutcome,
    claim_nonce: Option<u64>,
) {
    let retention = dedup_retention_for(&outcome.status);
    let forced_recording = outcome
        .info
        .contains_key(action_control::FORCED_RECORDING_KEY);
    if retention == DedupRetention::Release || forced_recording {
        dedup_cache.invalidate(key, claim_nonce).await;
        return;
    }
    let mut score_for_dedup = score.clone();
    score_for_dedup.decoded_actions.clear();
    let acted_class = outcome
        .info
        .get("acted_class")
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(dedup_cache::CLASS_NONE);
    let entry = entities::DedupEntry {
        status: &outcome.status,
        acted_class,
        topic,
        dry_run: outcome.dry_run,
        additional_info_map: &outcome.info,
        score_result: score_for_dedup,
    };
    let json_bytes = serde_json::to_vec(&entry).unwrap_or_default();
    let compressed = zstd::encode_all(json_bytes.as_slice(), 3).unwrap_or(json_bytes);
    if retention == DedupRetention::Full {
        dedup_cache
            .update(key, &compressed, acted_class, claim_nonce)
            .await;
    } else {
        dedup_cache
            .update_skip(key, &compressed, acted_class, claim_nonce)
            .await;
    }
}

const DECISIONS_PRODUCER: &str = "decisions";

const DECISIONS_JSON_PRODUCER: &str = "decisions_json";

pub(crate) const ADMIN_ACTIONS_PRODUCER: &str = "admin_actions";

#[derive(Clone, Default)]
pub struct KafkaProducers {
    by_name: HashMap<String, Arc<KafkaProducer>>,
}

impl KafkaProducers {
            pub fn get(&self, name: &str) -> Option<&Arc<KafkaProducer>> {
        self.by_name.get(name)
    }

            pub fn decisions(&self) -> Option<&Arc<KafkaProducer>> {
        self.get(DECISIONS_PRODUCER)
    }

            pub fn decisions_json(&self) -> Option<&Arc<KafkaProducer>> {
        self.get(DECISIONS_JSON_PRODUCER)
    }

            pub fn admin_actions(&self) -> Option<&Arc<KafkaProducer>> {
        self.get(ADMIN_ACTIONS_PRODUCER)
    }

                            pub async fn flush_all(&self, timeout: Duration) {
        for (name, producer) in &self.by_name {
            if let Err(e) = producer.flush(timeout).await {
                warn!("kafka producer '{name}' flush on shutdown failed: {e}");
            }
        }
    }
}

pub(crate) fn spawn_publish<M: prost::Message>(
    producer: Option<&Arc<KafkaProducer>>,
    sink: &'static str,
    key: impl FnOnce() -> Vec<u8>,
    record: &M,
) {
    let Some(producer) = producer else {
        return;
    };
    spawn_send(producer.clone(), sink, key(), record.encode_to_vec());
}

pub(crate) fn spawn_publish_json<M: serde::Serialize>(
    producer: Option<&Arc<KafkaProducer>>,
    sink: &'static str,
    key: impl FnOnce() -> Vec<u8>,
    record: &M,
) {
    let Some(producer) = producer else {
        return;
    };
    let bytes = match serde_json::to_vec(record) {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!("{sink} JSON encode failed, record dropped: {e}");
            metrics::KAFKA_PUBLISH_TOTAL
                .with_label_values(&[sink, "encode_error"])
                .inc();
            return;
        }
    };
    spawn_send(producer.clone(), sink, key(), bytes);
}

fn spawn_send(producer: Arc<KafkaProducer>, sink: &'static str, key: Vec<u8>, bytes: Vec<u8>) {
    tokio::spawn(async move {
        let result = match producer.send_with_key(Some(key.as_slice()), &bytes).await {
            Ok(_) => "ok",
            Err(e) => {
                tracing::debug!("{sink} produce failed (ignored): {e}");
                "error"
            }
        };
        metrics::KAFKA_PUBLISH_TOTAL
            .with_label_values(&[sink, result])
            .inc();
    });
}

fn publish_decision_outcome(ctx: &EnforcementCtx, outcome: &abuse_proto::DecisionOutcome) {
    #[cfg(test)]
    if let Some(log) = &ctx.outcome_log {
        log.lock().unwrap().push(outcome.clone());
    }
    spawn_publish(
        ctx.kafka_producer_decisions.as_ref(),
        DECISIONS_PRODUCER,
        || outcome.entity_id.to_string().into_bytes(),
        outcome,
    );
    spawn_publish_json(
        ctx.kafka_producer_decisions_json.as_ref(),
        DECISIONS_JSON_PRODUCER,
        || outcome.entity_id.to_string().into_bytes(),
        outcome,
    );
}

async fn run_enforcement(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
    decision: &DecisionContext,
) -> Result<abuse_proto::DecisionOutcome> {
    let mut control = decision.control.clone();
    let mut extra = BTreeMap::new();
    let run = decision.run.as_deref();
    let (entity_type, entity_id) = Facts::resolve_target(score);
    let key = run.map(|r| backfill_claims::ClaimKey {
        run_id: r.run_id.clone(),
        entity: entity_type,
        entity_id,
    });
    let _pair = match &key {
        Some(k) => Some(ctx.backfill.key_locks.lock(k).await),
        None => None,
    };
    let result = run_enforcement_inner(
        ctx,
        score,
        source_topic,
        &mut control,
        decision.held_class,
        decision.global_dry_run,
        &mut extra,
        run.map(|r| (r, decision.run_nonce)),
    )
    .await;
    let finish = |status: String| async move {
        if let (Some(k), Some(claims)) = (&key, ctx.backfill.claims())
            && let Err(e) = claims.finish(k, decision.run_nonce, &status).await
        {
            warn!(run_id = %k.run_id, entity_id = k.entity_id, "backfill claim finish failed: {e:#}");
        }
    };
    let mut outcome = match (result, run) {
        (Ok(outcome), _) => outcome,
        (Err(e), Some(_)) if e.downcast_ref::<PermanentAisFailure>().is_some() => {
            finish("ais_rejected".into()).await;
            return Err(e);
        }
        (Err(e), _) => return Err(e),
    };
    if let Some(run) = run {
        outcome.info.extend(backfill::info_keys(run, score));
        if outcome.status == off_switch::STATUS_SWITCHED_OFF {
            ctx.backfill.hold_for_off_switch(run, now_millis()).await?;
        }
        if !decision.records_only() && !backfill::leaves_claim(&outcome.status) {
            finish(outcome.status.clone()).await;
            outcome
                .info
                .insert("backfill_claim".into(), "finished".into());
        }
    }
    outcome.info.extend(env_info(ctx));
    outcome.info.extend(control.info_keys());
    if decision.records_only() {
        outcome.info.insert(
            "would_dedup_skip".into(),
            decision.would_dedup_held.is_some().to_string(),
        );
        if let Some(class) = decision.would_dedup_held {
            outcome
                .info
                .insert("would_dedup_held_class".into(), class.to_string());
        }
    }
    outcome.info.extend(extra);
    if control.row.is_some() && outcome.status == off_switch::STATUS_SWITCHED_OFF {
        outcome
            .info
            .entry("logged_because".into())
            .or_insert_with(|| "off_switch".into());
    }
    publish_decision_outcome(ctx, &outcome);
    Ok(outcome)
}

fn env_info(ctx: &EnforcementCtx) -> BTreeMap<String, String> {
    BTreeMap::from([("env".to_owned(), ctx.action_control.env.clone())])
}

// A decision that carries a policy version comes from an upstream decision
// producer that is not part of this release. Such decisions act only
// through their head's action-control rows.
fn is_factory_message(score: &abuse_proto::ScoreResult) -> bool {
    !score.policy_version.is_empty()
}

fn author_of(score: &abuse_proto::ScoreResult, entity_type: EntityType) -> i64 {
    match entity_type {
        EntityType::Post => score.user_id,
        EntityType::User => Facts::resolve_target(score).1,
    }
}

fn resolve_control(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    entity_type: EntityType,
) -> action_control::Resolved {
    let labels: &[String] = score.summary.as_ref().map_or(&[], |s| s.labels.as_slice());
    let producer_dry = score.dry_run || action_control::has_dry_run_prefix(labels);
    let factory = is_factory_message(score);
    let claim = Some(confidence::message_claim(score, now_millis()));
    let input = action_control::ResolveInput {
        head: crate::facts::head_label(score),
        classifier: &score.model_version,
        labels,
        entity: entity_type,
        factory,
        producer_dry,
        confidence: claim,
        author_id: author_of(score, entity_type),
    };
    ctx.action_control.resolve(
        &input,
        &ctx.dynamic_config.action_control_config(),
        |entity| safety_checks_ok(ctx, entity),
    )
}

fn resolve_run_control(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    entity_type: EntityType,
    run: &backfill::BackfillRunRecord,
) -> action_control::Resolved {
    let labels: &[String] = score.summary.as_ref().map_or(&[], |s| s.labels.as_slice());
    let input = action_control::ResolveInput {
        head: crate::facts::head_label(score),
        classifier: &score.model_version,
        labels,
        entity: entity_type,
        factory: run.factory,
        producer_dry: false,
        confidence: Some(confidence::message_claim(score, now_millis())),
        author_id: author_of(score, entity_type),
    };
    ctx.action_control.resolve_run(
        &input,
        &ctx.dynamic_config.action_control_config(),
        |entity| safety_checks_ok(ctx, entity),
        &run.actions,
    )
}

fn safety_checks_ok(ctx: &EnforcementCtx, entity: EntityType) -> bool {
    let rules = ctx.rules_cache.resolve(
        entity,
        ctx.dynamic_config.enforcement_rules_yaml(entity).as_ref(),
    );
    let ok = safety_checks::status(entity, &rules).ok;
    metrics::ACTION_CONTROL_SAFETY_CHECKS_OK
        .with_label_values(&[entity.as_str()])
        .set(i64::from(ok));
    ok
}

async fn build_kafka_producers(cfg: &Config, boot_config: &serde_json::Value) -> KafkaProducers {
    let mut producers = KafkaProducers::default();

    for (name, spec) in growthbook::kafka_producers_from_config(Some(boot_config)) {
        if !spec.enabled {
            warn!("kafka producer '{name}' configured but disabled (enabled=false)");
            continue;
        }
        let Some(topic) = spec.topic else {
            warn!("kafka producer '{name}' enabled but no topic set; skipping");
            continue;
        };
        let cluster =
            override_or_default(spec.cluster.as_deref(), &cfg.kafka_producer_mtls_cluster);
        let zone = override_or_default(spec.zone.as_deref(), &cfg.kafka_producer_mtls_zone);
        let producer_config = match KafkaProducerConfigBuilder::for_cluster_mtls_auto(
            cluster,
            topic.clone(),
            Some(zone),
        ) {
            Ok(builder) => builder.build(),
            Err(e) => {
                error!("kafka producer '{name}' mTLS config failed; skipping: {e}");
                continue;
            }
        };
        let mut producer = KafkaProducer::new(producer_config);
        match producer.start().await {
            Ok(()) => {
                info!("kafka producer '{name}' started -> topic {topic} on {cluster} ({zone})");
                producers.by_name.insert(name, Arc::new(producer));
            }
            Err(e) => {
                error!("kafka producer '{name}' failed to start; continuing without it: {e}")
            }
        }
    }
    producers
}

enum Admission {
        Admitted(Option<Box<action_control::Resolved>>),
        Dropped(&'static str),
}

impl ScoreResultProcessor {
                            async fn admit(&self, score: &abuse_proto::ScoreResult) -> Admission {
        let msg_labels: Vec<&str> = score
            .summary
            .as_ref()
            .map(|s| {
                s.labels
                    .iter()
                    .map(|l| action_control::strip_dry_run_prefix(l))
                    .collect()
            })
            .unwrap_or_default();
        let dropped = if !self.negative_labels.is_empty()
            && msg_labels.iter().any(|l| self.negative_labels.contains(*l))
        {
            "negative_label"
        } else if !self.labels.is_empty() && !msg_labels.iter().any(|l| self.labels.contains(*l)) {
            "no_match"
        } else {
            if !self.labels.is_empty() {
                let matches: Vec<&&str> = msg_labels
                    .iter()
                    .filter(|l| self.labels.contains(**l))
                    .collect();
                info!(?matches, "matching labels");
            }
            return Admission::Admitted(None);
        };
        if self.backfill_topic {
            return Admission::Dropped(dropped);
        }
        self.ctx
            .action_control
            .ensure_loaded(&self.ctx.dynamic_config.action_control_config())
            .await;
        let (entity_type, _) = Facts::resolve_target(score);
        let head = crate::facts::head_label(score);
        let control = resolve_control(&self.ctx, score, entity_type);
        let admits = if control.unknown {
            dropped == "no_match"
        } else {
            self.ctx.action_control.admits_unlisted(entity_type, head)
        };
        if !admits {
            return Admission::Dropped(dropped);
        }
        match control.admit_unlisted() {
            Some(control) => {
                info!(
                    list = dropped,
                    "admitted past the topic's label list by its head's record"
                );
                Admission::Admitted(Some(Box::new(control)))
            }
            None => Admission::Dropped(dropped),
        }
    }

                                                        #[tracing::instrument(skip_all)]
    async fn process(
        &self,
        score: &abuse_proto::ScoreResult,
        control: Option<action_control::Resolved>,
    ) -> Result<String, EnforcementFailed> {
        let source = "kafka";
        let topic = self.topic.as_str();
        let head = crate::facts::head_label(score);
        let model = score.model_version.as_str();
        let (entity_type, entity_id) = Facts::resolve_target(score);

        if entity_id == 0 {
            warn!(
                entity_type = entity_type.as_str(),
                "score has no entity id; skipping"
            );
            metrics::ENFORCEMENT_TOTAL
                .with_label_values(&[
                    source,
                    topic,
                    entity_type.as_str(),
                    "final",
                    "invalid_entity_id",
                    "",
                    "",
                    head,
                    model,
                ])
                .inc();
            publish_decision_outcome(
                &self.ctx,
                &decision_outcome(
                    score,
                    topic,
                    "invalid_entity_id".into(),
                    false,
                    env_info(&self.ctx),
                ),
            );
            return Ok("invalid_entity_id".into());
        }

        match (self.backfill_topic, score.backfill_run_id.is_empty()) {
            (false, false) => {
                return Ok(self.early_skip(
                    score,
                    backfill::STATUS_ID_ON_LIVE_TOPIC,
                    BTreeMap::new(),
                ));
            }
            (true, true) => {
                return Ok(self.early_skip(score, backfill::STATUS_ID_MISSING, BTreeMap::new()));
            }
            _ => {}
        }

        self.ctx
            .action_control
            .ensure_loaded(&self.ctx.dynamic_config.action_control_config())
            .await;
        let (control, run) = if self.backfill_topic {
            match self.admit_run(score, entity_type, entity_id).await {
                Ok(run) => (
                    resolve_run_control(&self.ctx, score, entity_type, &run),
                    Some(run),
                ),
                Err((status, info)) => return Ok(self.early_skip(score, status, info)),
            }
        } else {
            let control = control.unwrap_or_else(|| resolve_control(&self.ctx, score, entity_type));
            (control, None)
        };
        let mut decision = DecisionContext {
            control,
            would_dedup_held: None,
            held_class: dedup_cache::CLASS_NONE,
            global_dry_run: self.ctx.dynamic_config.is_dry_run(),
            run,
            run_nonce: backfill_claims::new_claim_nonce(),
        };

        let incoming_class = decision.control.incoming_dedup_class(|| {
            incoming_dedup_class(&self.ctx.dynamic_config, &self.ctx.rules_cache, score)
        });
        let dedup_key = decision.dedup_key(entity_type, entity_id);
        let claim_nonce = if decision.records_only() {
            decision.would_dedup_held = self.dedup_cache.peek(&dedup_key, incoming_class).await;
            None
        } else {
            match self
                .dedup_cache
                .try_insert(&dedup_key, incoming_class)
                .await
            {
                DedupCheck::New { claim_nonce } => claim_nonce,
                DedupCheck::ClassBypass {
                    held_class,
                    claim_nonce,
                } => {
                    info!(
                        incoming_class,
                        held_class, "dedup: class bypass (incoming outranks held entry)"
                    );
                    metrics::DEDUP_CLASS_BYPASS_TOTAL
                        .with_label_values(&[
                            topic,
                            head,
                            &held_class.to_string(),
                            &incoming_class.to_string(),
                        ])
                        .inc();
                    decision.held_class = held_class;
                    claim_nonce
                }
                DedupCheck::Duplicate => {
                    info!("dedup: skipping (already enforced recently)");
                    metrics::ENFORCEMENT_TOTAL
                        .with_label_values(&[
                            source,
                            topic,
                            entity_type.as_str(),
                            "final",
                            "dedup_skipped",
                            "",
                            "",
                            head,
                            model,
                        ])
                        .inc();
                    let mut info = env_info(&self.ctx);
                    if let Some(run) = decision.run.as_deref() {
                        info.extend(backfill::info_keys(run, score));
                    }
                    publish_decision_outcome(
                        &self.ctx,
                        &decision_outcome(score, topic, "dedup_skipped".into(), false, info),
                    );
                    return Ok("dedup_skipped".into());
                }
            }
        };

        let outcome = match run_enforcement(&self.ctx, score, topic, &decision).await {
            Ok(outcome) => outcome,
            Err(source) => {
                let permanent = source.downcast_ref::<PermanentAisFailure>().is_some();
                if permanent && decision.claimed() {
                    self.dedup_cache.invalidate(&dedup_key, claim_nonce).await;
                }
                return Err(EnforcementFailed {
                    permanent,
                    claim_nonce,
                    decision,
                    source,
                });
            }
        };

        if decision.claimed() {
            write_dedup_outcome(
                self.dedup_cache.as_ref(),
                &dedup_key,
                score,
                topic,
                &outcome,
                claim_nonce,
            )
            .await;
        }

        let status = outcome.status;

        if let Some(summary) = score.summary.as_ref() {
            for fh in &summary.fired_heads {
                metrics::FIRED_HEADS_TOTAL
                    .with_label_values(&[
                        topic,
                        entity_type.as_str(),
                        model,
                        fh.name.as_str(),
                        status.as_str(),
                    ])
                    .inc();
            }
        }

        if status != "success" {
            info!("enforcement outcome: {status}");
            metrics::ENFORCEMENT_TOTAL
                .with_label_values(&[
                    source,
                    topic,
                    entity_type.as_str(),
                    "final",
                    &status,
                    "",
                    "",
                    head,
                    model,
                ])
                .inc();
        }

        Ok(status)
    }

            fn early_skip(
        &self,
        score: &abuse_proto::ScoreResult,
        status: &str,
        info: BTreeMap<String, String>,
    ) -> String {
        let (entity_type, _) = Facts::resolve_target(score);
        metrics::ENFORCEMENT_TOTAL
            .with_label_values(&[
                "kafka",
                &self.topic,
                entity_type.as_str(),
                "final",
                status,
                "",
                "",
                crate::facts::head_label(score),
                &score.model_version,
            ])
            .inc();
        let mut info = info;
        info.extend(env_info(&self.ctx));
        publish_decision_outcome(
            &self.ctx,
            &decision_outcome(score, &self.topic, status.into(), false, info),
        );
        status.to_owned()
    }

                                async fn admit_run(
        &self,
        score: &abuse_proto::ScoreResult,
        entity_type: EntityType,
        entity_id: i64,
    ) -> Result<Arc<backfill::BackfillRunRecord>, (&'static str, BTreeMap<String, String>)> {
        let bf = &self.ctx.backfill;
        let run_id = score.backfill_run_id.as_str();
        let mut last_err = None;
        for backoff_ms in [0u64, 100, 500] {
            if backoff_ms > 0 {
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
            let run = match bf.run(run_id).await {
                Ok(run) => run,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };
            let Some(run) = run else {
                let info = BTreeMap::from([("backfill_run_id".to_owned(), run_id.to_owned())]);
                return Err((backfill::STATUS_RUN_UNKNOWN, info));
            };
            let info: BTreeMap<String, String> =
                backfill::info_keys(&run, score).into_iter().collect();
            if let Err(status) = backfill::gate(&run, now_millis()) {
                return Err((status, info));
            }
            if let Err(status) = backfill::check_message(&run, score) {
                return Err((status, info));
            }
            let Some(claims) = bf.claims() else {
                last_err = Some(anyhow::anyhow!("no backfill claim store on this pod"));
                break;
            };
            let key = backfill_claims::ClaimKey {
                run_id: run_id.to_owned(),
                entity: entity_type,
                entity_id,
            };
            match claims.state(&key).await {
                Ok(backfill_claims::ItemClaim::NotInList) => {
                    return Err((backfill::STATUS_NOT_IN_RUN, info));
                }
                Ok(backfill_claims::ItemClaim::Finished { status }) => {
                    let mut info = info;
                    info.insert("backfill_claim_status".into(), status);
                    return Err((backfill::STATUS_ALREADY_FINISHED, info));
                }
                Ok(_) => return Ok(Arc::new(run)),
                Err(e) => {
                    warn!(run_id, error = %format!("{e:#}"), "backfill claim read failed on arrival; deciding on");
                    return Ok(Arc::new(run));
                }
            }
        }
        warn!(
            run_id,
            error = %last_err.map(|e| format!("{e:#}")).unwrap_or_default(),
            "backfill run or claim unreadable; recording without acting"
        );
        let info = BTreeMap::from([("backfill_run_id".to_owned(), run_id.to_owned())]);
        Err((backfill::STATUS_RUN_UNREADABLE, info))
    }
}

#[async_trait::async_trait]
impl KafkaBatchProcessor for ScoreResultProcessor {
    async fn process_batch(&self, messages: &[KafkaMessage]) -> Result<BatchResult> {
        self.last_progress.store(now_millis(), Ordering::Relaxed);
        let max_in_flight = self.max_in_flight.max(1);
        let gate = Arc::new(Semaphore::new(max_in_flight));
        let failed_scores: Vec<Option<_>> = join_all(messages.iter().map(|message| {
            let start = std::time::Instant::now();
            let topic = self.topic.as_str();
            let gate = gate.clone();
            let span = tracing::info_span!(
                "message",
                topic = message.topic,
                entity_type = tracing::field::Empty,
                entity_id = tracing::field::Empty
            );
            async move {
                let _permit = gate
                    .acquire_owned()
                    .await
                    .expect("semaphore closed unexpectedly");
                let Some(payload) = &message.payload else {
                    metrics::KAFKA_MESSAGE_LATENCY
                        .with_label_values(&[topic, "", "no_payload", "false"])
                        .observe(start.elapsed().as_secs_f64());
                    return None;
                };

                let score = match abuse_proto::ScoreResult::decode(payload.as_slice()) {
                    Ok(s) => s,
                    Err(e) => {
                        warn!("failed to decode ScoreResult: {e}");
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, "", "decode_failed", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        return None;
                    }
                };

                let (entity_type, entity_id) = Facts::resolve_target(&score);
                let entity_type = entity_type.as_str();
                let span = tracing::Span::current();
                span.record("entity_type", entity_type);
                span.record("entity_id", entity_id);

                let control = match self.admit(&score).await {
                    Admission::Admitted(control) => control.map(|c| *c),
                    Admission::Dropped(status) => {
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, status, "false"])
                            .observe(start.elapsed().as_secs_f64());
                        return None;
                    }
                };

                match self.process(&score, control).await {
                    Ok(status) => {
                        let enforced = if status == "success" { "true" } else { "false" };
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, &status, enforced])
                            .observe(start.elapsed().as_secs_f64());
                        None
                    }
                    Err(f) if f.permanent => {
                        error!(
                            "enforcement failed permanently (not retried): {}",
                            f.source
                        );
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, "permanent_failure", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        None
                    }
                    Err(f) => {
                        warn!(error = %format!("{:#}", f.source), "enforcement failed (queued for retry)");
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, "error", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        Some((score, f.claim_nonce, f.decision))
                    }
                }
            }
            .instrument(span)
        }))
        .await;

        {
            let mut queue = self.retry_queue.lock().await;
            for (score, claim_nonce, decision) in failed_scores.into_iter().flatten() {
                queue.push(RetryEntry {
                    score,
                    attempt: 1,
                    next_retry_at: tokio::time::Instant::now() + Duration::from_secs(1),
                    topic: self.topic.clone(),
                    dedup_cache: self.dedup_cache.clone(),
                    claim_nonce,
                    decision,
                });
            }
            metrics::RETRY_QUEUE_SIZE.set(queue.len() as i64);
        }

        Ok(BatchResult::CommitAll)
    }
}


struct StatsOnlyProcessor {
    should_log_content: bool,
        last_progress: Arc<AtomicI64>,
}

#[async_trait::async_trait]
impl KafkaBatchProcessor for StatsOnlyProcessor {
    async fn process_batch(&self, messages: &[KafkaMessage]) -> Result<BatchResult> {
        self.last_progress.store(now_millis(), Ordering::Relaxed);
        if self.should_log_content {
            for message in messages {
                if let Some(payload) = &message.payload {
                    match abuse_proto::ScoreResult::decode(payload.as_slice()) {
                        Ok(score) => {
                            let json = serde_json::to_value(&score).unwrap_or_default();
                            info!(
                                topic = message.topic,
                                partition = message.partition,
                                offset = message.offset,
                                "message content: {}",
                                json
                            );
                        }
                        Err(e) => {
                            warn!(
                                topic = message.topic,
                                partition = message.partition,
                                offset = message.offset,
                                "failed to decode ScoreResult: {e}"
                            );
                        }
                    }
                }
            }
        }
        Ok(BatchResult::CommitAll)
    }
}


pub fn init_runtime_globals() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    xai_init_utils::init().rustls();
    metrics::init();
}

pub async fn build_state(cfg: Config) -> Result<(Arc<service::AppState>, Config)> {
    info!("Service starting (port={})", cfg.port);

    let ais_cert = cfg
        .strato_client_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.crt".into());
    let ais_key = cfg
        .strato_client_key_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.key".into());
    let ais_ca = cfg
        .strato_ca_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/ca/ca-bundle.crt".into());
    let ais_client = ais_client::AisClient::connect(&ais_client::AisClientConfig {
        wily_path: cfg.ais_wily_path.clone(),
        tls_server_name: cfg.ais_tls_server_name.clone(),
        zone: cfg.ais_zone.clone(),
        s2s_cert_path: ais_cert,
        s2s_key_path: ais_key,
        s2s_ca_path: ais_ca,
    })
    .context("failed to initialise AIS ThriftMux client")?;

    let limiter = match cfg.limiter_datacenter.as_deref() {
        Some(datacenter) if !datacenter.is_empty() => {
            let limiter_cfg = limiter::LimiterConfig {
                datacenter: datacenter.to_owned(),
                feature: cfg.limiter_feature.clone(),
                fail_open: cfg.limiter_fail_open,
                client_id: cfg.limiter_client_id.clone(),
                ca_cert_path: cfg.limiter_ca_cert_path.clone(),
                client_cert_path: cfg.limiter_client_cert_path.clone(),
                client_key_path: cfg.limiter_client_key_path.clone(),
            };
            match limiter::LimiterClient::connect(&limiter_cfg).await {
                Ok(client) => Some(client),
                Err(e) => {
                    warn!("Limiter connect failed: {e}; global enforcement cap disabled");
                    None
                }
            }
        }
        _ => {
            info!("LIMITER_DATACENTER not set; global enforcement cap disabled (all allowed)");
            None
        }
    };

    let mh_client = manhattan::build_client(&cfg).await?;
    let mh_tenant = manhattan::build_tenant(&cfg);
    info!(
        "Manhattan storage: cluster={} app={} dataset={} dc={} (dedup ttl={}s skip_ttl={}s)",
        cfg.mh_cluster,
        cfg.mh_app_id,
        cfg.mh_dataset,
        cfg.mh_datacenter,
        cfg.dedup_ttl_secs,
        cfg.dedup_skip_ttl_secs,
    );
    let dedup_cache = Some(dedup_cache::ManhattanDedupCache::new(
        mh_client.clone(),
        mh_tenant.clone(),
        cfg.dedup_ttl_secs,
        cfg.dedup_skip_ttl_secs,
    ));
    let allowlist = Some(allowlist::ManhattanAllowlist::new(
        mh_client.clone(),
        mh_tenant.clone(),
    ));

    let (sliding_user, sliding_post) = match limiter.as_ref() {
        Some(lim) => {
            let user_sw = sliding_window::SlidingWindowLimiter::new(
                lim.clone(),
                mh_client.clone(),
                mh_tenant.clone(),
                cfg.limiter_fail_open,
                sliding_window::USER_SNAP_PKEY,
                "user",
                sliding_window::DAILY_CAP,
            );
            let post_sw = sliding_window::SlidingWindowLimiter::new(
                lim.with_bucket(limiter::GLOBAL_BUCKET_POST_ID),
                mh_client.clone(),
                mh_tenant.clone(),
                cfg.limiter_fail_open,
                sliding_window::POST_SNAP_PKEY,
                "post",
                sliding_window::DAILY_CAP,
            );
            user_sw.spawn_background();
            post_sw.spawn_background();
            info!(
                "Sliding-window enforcement caps enabled for user + post (Limiter counters + Manhattan snapshots)"
            );
            (Some(user_sw), Some(post_sw))
        }
        None => (None, None),
    };

    let backfill_release = cfg
        .backfill_topic
        .as_deref()
        .is_some_and(|t| !t.trim().is_empty());
    let rules_cache = rules::RulesCache::new();
    let (dynamic_config, growthbook_writer) = if backfill_release {
        backfill_release_config(&cfg, sliding_user, sliding_post)?
    } else {
        live_config(&cfg, &rules_cache, sliding_user, sliding_post).await?
    };

    let boot_config = dynamic_config.config().unwrap_or(serde_json::Value::Null);

    let kafka_producers = build_kafka_producers(&cfg, &boot_config).await;
    if backfill_release && kafka_producers.decisions().is_none() {
        anyhow::bail!(
            "the backfill release publishes no decision outcomes: \
             BACKFILL_SERVICE_CONFIG_JSON needs an enabled kafka.producer.decisions"
        );
    }

    finish_state(
        cfg,
        ais_client,
        limiter,
        mh_client,
        mh_tenant,
        dedup_cache,
        allowlist,
        dynamic_config,
        rules_cache,
        growthbook_writer,
        boot_config,
        kafka_producers,
    )
    .await
}

fn backfill_release_config(
    cfg: &Config,
    sliding_user: Option<sliding_window::SlidingWindowLimiter>,
    sliding_post: Option<sliding_window::SlidingWindowLimiter>,
) -> Result<(DynamicConfig, Option<GrowthBookWriter>)> {
    let set = |v: &Option<String>| v.as_deref().is_some_and(|s| !s.trim().is_empty());
    if set(&cfg.growthbook_url) || set(&cfg.growthbook_key) || set(&cfg.growthbook_admin_api_key) {
        anyhow::bail!(
            "the backfill release reads no GrowthBook: unset GROWTHBOOK_URL, GROWTHBOOK_KEY \
             and GROWTHBOOK_ADMIN_API_KEY (its config is BACKFILL_SERVICE_CONFIG_JSON)"
        );
    }
    let raw = cfg
        .backfill_service_config_json
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("BACKFILL_TOPIC is set but BACKFILL_SERVICE_CONFIG_JSON is not")
        })?;
    let config: serde_json::Value =
        serde_json::from_str(raw).context("BACKFILL_SERVICE_CONFIG_JSON is not JSON")?;
    if !config.is_object() {
        anyhow::bail!("BACKFILL_SERVICE_CONFIG_JSON must be a JSON object");
    }
    info!(
        config = %config,
        "backfill release: config from BACKFILL_SERVICE_CONFIG_JSON; no GrowthBook client, rules are the baked-in mirror"
    );
    Ok((
        DynamicConfig::from_static(
            config,
            cfg.dry_run,
            cfg.max_enforcements_per_day,
            cfg.max_post_enforcements_per_day,
            sliding_user,
            sliding_post,
        ),
        None,
    ))
}

async fn live_config(
    cfg: &Config,
    rules_cache: &rules::RulesCache,
    sliding_user: Option<sliding_window::SlidingWindowLimiter>,
    sliding_post: Option<sliding_window::SlidingWindowLimiter>,
) -> Result<(DynamicConfig, Option<GrowthBookWriter>)> {
    if cfg
        .growthbook_url
        .as_deref()
        .filter(|s| !s.is_empty())
        .is_none()
        || cfg
            .growthbook_key
            .as_deref()
            .filter(|s| !s.is_empty())
            .is_none()
    {
        anyhow::bail!(
            "GrowthBook is required (GROWTHBOOK_URL + GROWTHBOOK_KEY): \
             needed for live config + rule features \
             xai_abuse_enforcement_service_rule_{{user,post}}"
        );
    }
    let dynamic_config = DynamicConfig::new(
        cfg.growthbook_url.as_deref(),
        cfg.growthbook_key.as_deref(),
        cfg.dry_run,
        cfg.max_enforcements_per_day,
        cfg.max_post_enforcements_per_day,
        sliding_user,
        sliding_post,
    )
    .await
    .map_err(|e| anyhow::anyhow!(e))?;

    let user_rules_yaml = dynamic_config
        .enforcement_rules_yaml(EntityType::User)
        .filter(|s| !s.trim().is_empty());
    let post_rules_yaml = dynamic_config
        .enforcement_rules_yaml(EntityType::Post)
        .filter(|s| !s.trim().is_empty());
    let (Some(user_yaml), Some(post_yaml)) = (user_rules_yaml, post_rules_yaml) else {
        anyhow::bail!(
            "GrowthBook rule features must both be set and non-empty at startup \
             (xai_abuse_enforcement_service_rule_user + \
             xai_abuse_enforcement_service_rule_post); refusing to boot on \
             baked-in defaults alone"
        );
    };
    let start = std::time::Instant::now();
    let user_status = rules_cache.status(EntityType::User, Some(&user_yaml));
    let post_status = rules_cache.status(EntityType::Post, Some(&post_yaml));
    if user_status.source != rules::RuleSource::Dynamic {
        anyhow::bail!(
            "failed to compile GrowthBook user rules at startup: {}",
            user_status
                .error
                .as_deref()
                .unwrap_or("override present but not dynamic")
        );
    }
    if post_status.source != rules::RuleSource::Dynamic {
        anyhow::bail!(
            "failed to compile GrowthBook post rules at startup: {}",
            post_status
                .error
                .as_deref()
                .unwrap_or("override present but not dynamic")
        );
    }
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        user_rules = user_status.rule_ids.len(),
        post_rules = post_status.rule_ids.len(),
        user_source = ?user_status.source,
        post_source = ?post_status.source,
        "GrowthBook rule pipelines compiled and live (baked-in defaults retained as fallback)"
    );

    let growthbook_writer = cfg.growthbook_admin_api_key.clone().map(|k| {
        info!(
            "GrowthBook writer enabled (host={}, feature={}, env={})",
            cfg.growthbook_admin_api_host, cfg.growthbook_feature_key, cfg.growthbook_environment,
        );
        GrowthBookWriter::new(&cfg.growthbook_admin_api_host, k)
    });
    if growthbook_writer.is_none() {
        info!("GROWTHBOOK_ADMIN_API_KEY not set — admin config-mutation endpoints will return 503");
    }
    Ok((dynamic_config, growthbook_writer))
}

#[allow(clippy::too_many_arguments)]
async fn finish_state(
    cfg: Config,
    ais_client: ais_client::AisClient,
    limiter: Option<limiter::LimiterClient>,
    mh_client: Arc<xai_manhattan::NativeManhattanClient>,
    mh_tenant: xai_manhattan::Tenant,
    dedup_cache: Option<dedup_cache::ManhattanDedupCache>,
    allowlist: Option<allowlist::ManhattanAllowlist>,
    dynamic_config: DynamicConfig,
    rules_cache: rules::RulesCache,
    growthbook_writer: Option<GrowthBookWriter>,
    boot_config: serde_json::Value,
    kafka_producers: KafkaProducers,
) -> Result<(Arc<service::AppState>, Config)> {
    let startup_probe =
        overturn_hold::startup_probe_requested(cfg.overturn_hold_startup_probe.as_deref());
    let hold_gate = Arc::new(overturn_hold::HoldGate::from_url(
        cfg.overturn_hold_ledger_url.as_deref(),
        startup_probe,
        cfg.overturn_hold_env.as_deref(),
    ));
    if startup_probe {
        let gate = hold_gate.clone();
        tokio::spawn(async move {
            gate.startup_probe().await;
        });
    }
    hold_gate.publish_config(&dynamic_config.overturn_hold_gate());

    let off_switch = Arc::new(off_switch::OffSwitch::new(
        Arc::new(off_switch::ManhattanOffSwitchStore::new(
            mh_client.clone(),
            mh_tenant.clone(),
            off_switch::MH_PKEY,
            off_switch::RECORD_TTL_SECS,
        )),
        Arc::new(off_switch::ManhattanOffSwitchStore::new(
            mh_client.clone(),
            mh_tenant.clone(),
            off_switch::SEEN_PKEY,
            off_switch::SEEN_TTL_SECS,
        )),
    ));
    let off_switch_enabled = dynamic_config.off_switch_enabled();
    metrics::OFF_SWITCH_ENABLED.set(i64::from(off_switch_enabled));
    metrics::OFF_SWITCH_FIRST_LOAD_PENDING.set(i64::from(off_switch_enabled));
    if !off_switch_enabled {
        info!(
            "off switch disabled in GrowthBook ({}); not reading the switch list",
            off_switch::CONFIG_KEY
        );
    }
    off_switch.spawn_refresh(dynamic_config.clone());
    off_switch.spawn_seen_writer();

    let make_counter: Option<head_guard::CounterFactory> = limiter.clone().map(|lim| {
        let (mh, tenant, fail_open) = (mh_client.clone(), mh_tenant.clone(), cfg.limiter_fail_open);
        Box::new(move |lkey: &str| {
            sliding_window::SlidingWindowLimiter::new(
                lim.with_bucket(head_guard::bucket_id(lkey)),
                mh.clone(),
                tenant.clone(),
                fail_open,
                format!("{}{lkey}", head_guard::COUNT_PKEY_PREFIX),
                "head_guard",
                head_guard::WINDOWS,
            )
        }) as head_guard::CounterFactory
    });
    if make_counter.is_none() {
        warn!("no Limiter connection; head-guard action ceilings are not enforced");
    }
    let head_guard = Arc::new(head_guard::HeadGuard::new(
        Arc::new(action_control::ManhattanControlStore::new(
            mh_client.clone(),
            mh_tenant.clone(),
        )),
        make_counter,
    ));
    head_guard.spawn_refresh(dynamic_config.clone());

    let rollout_salt = std::env::var(action_control::ROLLOUT_SALT_ENV).unwrap_or_default();
    info!(
        "action control: rollout buckets are {}",
        if rollout_salt.is_empty() {
            "unsalted"
        } else {
            "salted"
        }
    );
    let action_control = Arc::new(
        action_control::ActionControl::new(
            Arc::new(action_control::ManhattanControlStore::new(
                mh_client.clone(),
                mh_tenant.clone(),
            )),
            std::env::var("NAMESPACE").unwrap_or_else(|_| "local".to_owned()),
            std::env::var("HOSTNAME").unwrap_or_default(),
        )
        .with_rollout_salt(rollout_salt),
    );
    if dynamic_config.action_control_config().ignore_store {
        warn!(
            "action control: GrowthBook {}.ignore_store is set; page records are ignored and recording-only heads follow the GrowthBook rules",
            action_control::CONFIG_KEY
        );
    }
    action_control.spawn_refresh(dynamic_config.clone());
    action_control.spawn_seen_writer();

    let claims: Option<Arc<dyn backfill_claims::ClaimStore>> = match cfg
        .backfill_topic
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        None => None,
        Some(_) => {
            let dsn = cfg.backfill_claims_pg_dsn.as_deref().ok_or_else(|| {
                anyhow::anyhow!("BACKFILL_TOPIC is set but BACKFILL_CLAIMS_PG_DSN is not")
            })?;
            let env = std::env::var("NAMESPACE").unwrap_or_else(|_| "local".to_owned());
            Some(Arc::new(backfill_claims::PgClaimStore::new(dsn, &env)?))
        }
    };
    let release = claims.is_some();
    let backfill = Arc::new(
        backfill::Backfill::new(
            Arc::new(action_control::ManhattanControlStore::new(
                mh_client.clone(),
                mh_tenant.clone(),
            )),
            claims,
            std::env::var("HOSTNAME").unwrap_or_default(),
        )
        .with_access(backfill::BackfillAccess::new(
            &cfg.backfill_page_keys,
            &cfg.backfill_sender_keys,
            cfg.backfill_max_rate_per_sec,
        )),
    );
    if release {
        let access = backfill.access();
        info!(
            page_keys = ?access.page_keys,
            sender_keys = ?access.sender_keys,
            max_rate_ceiling = access.max_rate_ceiling,
            "backfill runs: keys and speed ceiling from the Helm values; budgets and speed limit from {}",
            backfill::SETTINGS_PKEY
        );
        backfill.spawn_settings_refresh();
    }

    let state = Arc::new(AppState {
        ais_client,
        dynamic_config,
        rules_cache: Arc::new(rules_cache),
        dedup_cache,
        limiter,
        allowlist,
        growthbook_writer,
        growthbook_feature_key: cfg.growthbook_feature_key.clone(),
        growthbook_environment: cfg.growthbook_environment.clone(),
        kafka_ready: Arc::new(AtomicBool::new(false)),
        kafka_producers,
        hold_gate,
        off_switch,
        head_guard,
        action_control,
        backfill,
        boot_config,
    });

    Ok((state, cfg))
}

pub fn build_api_router(state: Arc<service::AppState>, api_keys: auth::ApiKeys) -> Router {
    use axum::routing::{get, patch, post};

    let protected = Router::new()
        .route("/action", post(service::handle_action))
        .route(
            "/config",
            get(service::handle_config).put(handle_config_put),
        )
        .route("/config/{field}", patch(handle_config_patch_field))
        .route("/rules", get(service::handle_rules_get))
        .route("/allowlist", get(handle_allowlist_list))
        .route("/allowlist/bulk", post(handle_allowlist_bulk))
        .route(
            "/allowlist/{user_id}",
            get(handle_allowlist_get)
                .put(handle_allowlist_put)
                .delete(handle_allowlist_delete),
        )
        .route(
            "/allowlist/{entity_type}/{entity_id}",
            get(handle_allowlist_get_entity)
                .put(handle_allowlist_put_entity)
                .delete(handle_allowlist_delete_entity),
        )
        .route(
            "/rate_limit/{entity_type}",
            post(handle_rate_limit_increment),
        )
        .route(
            "/rate_limit/{entity_type}/reset",
            post(handle_rate_limit_reset),
        )
        .route(
            "/off_switches",
            get(service::handle_off_switches_get).put(service::handle_off_switches_put),
        )
        .route(
            "/head_guards",
            get(service::handle_head_guards_get).put(service::handle_head_guards_put),
        )
        .route(
            "/action_controls",
            get(service::handle_action_controls_get).put(service::handle_action_controls_put),
        )
        .route(
            "/action_controls/history",
            get(service::handle_action_controls_history),
        )
        .route(
            "/action_controls/pod",
            get(service::handle_action_controls_pod),
        )
        .route(
            "/backfill_runs",
            get(service::handle_backfill_runs_list).post(service::handle_backfill_run_start),
        )
        .route(
            "/backfill_runs/budget",
            get(service::handle_backfill_runs_budget),
        )
        .route(
            "/backfill_runs/{run_id}",
            get(service::handle_backfill_run_get),
        )
        .route(
            "/backfill_runs/{run_id}/{change}",
            post(service::handle_backfill_run_change),
        )
        .route(
            "/backfill_settings",
            get(service::handle_backfill_settings_get).put(service::handle_backfill_settings_put),
        )
        .with_state(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            api_keys,
            auth::require_api_key,
        ));

    let public = Router::new()
        .route("/dry_run", get(service::handle_dry_run))
        .route("/rate_limit/{entity_type}", get(handle_rate_limit))
        .route("/dedup/{user_id}", get(handle_dedup_get))
        .route(
            "/dedup/{entity_type}/{entity_id}",
            get(handle_dedup_get_entity),
        )
        .with_state(state);

    protected.merge(public)
}

pub fn build_docs_router() -> Router {
    use utoipa::OpenApi;
    use utoipa_swagger_ui::SwaggerUi;

    let openapi = api_doc::ApiDoc::openapi();

    Router::new().merge(SwaggerUi::new("/api/docs").url("/api/openapi.json", openapi))
}

pub(crate) fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn consumer_error_total(watched: &HashSet<String>) -> f64 {
    use prometheus::core::Collector;
    if watched.is_empty() {
        return 0.0;
    }
    xai_kafka::metrics::CONSUMER_ERROR_COUNT
        .collect()
        .iter()
        .flat_map(|mf| &mf.metric)
        .filter(|m| {
            m.label
                .iter()
                .any(|l| l.name() == "topic" && watched.contains(l.value()))
        })
        .filter_map(|m| m.counter.as_ref())
        .filter_map(|c| c.value)
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn health_decision(
    consumed_once: bool,
    stale: Duration,
    error_bad_for: Duration,
    stale_after: Duration,
    stale_self_delete_after: Duration,
    error_self_delete_after: Duration,
    error_bad_now: bool,
    quiet_topic: bool,
) -> (bool, bool) {
    let progress_ok = quiet_topic || (consumed_once && stale < stale_after);
    let ready = progress_ok && !error_bad_now;
    let stale_bad = !quiet_topic && stale >= stale_self_delete_after;
    let should_self_delete = stale_bad || error_bad_for >= error_self_delete_after;
    (ready, should_self_delete)
}

struct WatchdogConfig {
    interval: Duration,
    stale_after: Duration,
    stale_self_delete_after: Duration,
    error_rate_per_sec: f64,
    error_self_delete_after: Duration,
    self_delete_enabled: bool,
            error_topics: HashSet<String>,
        quiet_topic: bool,
}

async fn kafka_health_watchdog(
    last_progress_ms: Arc<AtomicI64>,
    kafka_ready: Arc<AtomicBool>,
    cfg: WatchdogConfig,
) {
    let started_ms = now_millis();
    let interval_secs = cfg.interval.as_secs_f64().max(1.0);
    let mut ticker = tokio::time::interval(cfg.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut errors_prev = consumer_error_total(&cfg.error_topics);
    let mut error_bad_since: Option<tokio::time::Instant> = None;
    let mut deleted = false;

    loop {
        ticker.tick().await;

        let last = last_progress_ms.load(Ordering::Relaxed);
        let consumed_once = last > 0;
        let reference = if consumed_once { last } else { started_ms };
        let stale = Duration::from_millis((now_millis() - reference).max(0) as u64);

        let errors_now = consumer_error_total(&cfg.error_topics);
        let err_rate = (errors_now - errors_prev).max(0.0) / interval_secs;
        errors_prev = errors_now;
        let error_bad_now = cfg.error_rate_per_sec > 0.0 && err_rate >= cfg.error_rate_per_sec;
        let now = tokio::time::Instant::now();
        let error_bad_for = if error_bad_now {
            now.duration_since(*error_bad_since.get_or_insert(now))
        } else {
            error_bad_since = None;
            Duration::ZERO
        };

        let (healthy, should_self_delete) = health_decision(
            consumed_once,
            stale,
            error_bad_for,
            cfg.stale_after,
            cfg.stale_self_delete_after,
            cfg.error_self_delete_after,
            error_bad_now,
            cfg.quiet_topic,
        );

        if kafka_ready.swap(healthy, Ordering::Relaxed) != healthy {
            info!(
                healthy,
                stale_secs = stale.as_secs(),
                err_rate,
                "kafka watchdog: readiness change"
            );
        }

        if should_self_delete && cfg.self_delete_enabled && !deleted {
            deleted = true;
            let reason = if error_bad_now && error_bad_for >= cfg.error_self_delete_after {
                "error_flood"
            } else {
                "stale"
            };
            error!(
                stale_secs = stale.as_secs(),
                err_rate,
                reason,
                "kafka watchdog: bad node (no progress or broker errors); self-deleting pod"
            );
            metrics::KAFKA_SELF_DELETE_TOTAL
                .with_label_values(&[reason])
                .inc();
            self_delete_pod().await;
        }
        if healthy {
            deleted = false;
        }
    }
}

const KAFKA_PREFLIGHT_ATTEMPTS: u32 = 3;
const KAFKA_PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(10);
const KAFKA_PREFLIGHT_BACKOFF: Duration = Duration::from_secs(2);

async fn probe_broker_reachability(config: KafkaConsumerConfig, timeout: Duration) -> Result<()> {
    use rdkafka::ClientConfig;
    use rdkafka::consumer::{BaseConsumer, Consumer};

    let topic = config.base_config.topic.clone();
    let brokers = resolve_kafka_brokers(&config.base_config).await?;

    let mut client_config = ClientConfig::new();
    client_config
        .set("group.id", &config.group_id)
        .set("bootstrap.servers", brokers.join(","))
        .set("enable.ssl.certificate.verification", "false")
        .set("ssl.endpoint.identification.algorithm", "none");

    apply_auth_config(
        &mut client_config,
        config.base_config.ssl.as_ref(),
        config.s2s_certs.as_ref(),
    );

    let consumer: BaseConsumer = client_config
        .create()
        .context("Failed to create BaseConsumer for reachability probe")?;

    tokio::task::spawn_blocking(move || -> Result<()> {
        consumer
            .fetch_metadata(Some(&topic), timeout)
            .context("Kafka broker reachability probe failed")?;
        Ok(())
    })
    .await
    .context("Reachability probe task panicked")?
}

const KAFKA_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);

pub struct KafkaConsumers {
    cancel: CancellationToken,
    supervisor: JoinHandle<()>,
}

impl KafkaConsumers {
                                                async fn shutdown(self, timeout: Duration) {
        self.cancel.cancel();
        let mut supervisor = self.supervisor;
        if tokio::time::timeout(timeout, &mut supervisor)
            .await
            .is_err()
        {
            warn!(
                "Kafka consumers did not stop within {timeout:?}; aborting the in-flight batch \
                 (redelivered to the next owner; its dedup claims may be stranded)"
            );
            supervisor.abort();
            let _ = supervisor.await;
        }
    }
}

pub async fn shutdown_kafka(state: &service::AppState, consumers: KafkaConsumers) {
    consumers.shutdown(KAFKA_SHUTDOWN_TIMEOUT).await;
    state
        .kafka_producers
        .flush_all(Duration::from_secs(5))
        .await;
}

pub fn service_ready(state: &service::AppState) -> bool {
    state.kafka_ready.load(Ordering::Relaxed)
        && switch_lists_loaded(
            &state.off_switch,
            &state.head_guard,
            &state.action_control,
            &state.dynamic_config,
        )
        && backfill_settings_loaded(&state.backfill)
}

fn backfill_settings_loaded(backfill: &backfill::Backfill) -> bool {
    backfill.claims().is_none() || backfill.settings_loaded()
}

pub fn switch_lists_loaded(
    off_switch: &off_switch::OffSwitch,
    head_guard: &head_guard::HeadGuard,
    action_control: &action_control::ActionControl,
    dynamic_config: &DynamicConfig,
) -> bool {
    let switches_on = dynamic_config.off_switch_enabled();
    off_switch.ready(switches_on)
        && head_guard.ready(switches_on)
        && action_control.ready(&dynamic_config.action_control_config())
}

const SWITCH_LISTS_POLL: Duration = Duration::from_millis(200);

async fn wait_for_switch_lists(
    off_switch: &off_switch::OffSwitch,
    head_guard: &head_guard::HeadGuard,
    action_control: &action_control::ActionControl,
    dynamic_config: &DynamicConfig,
) {
    let mut logged = false;
    while !switch_lists_loaded(off_switch, head_guard, action_control, dynamic_config) {
        if !logged {
            warn!(
                off_switch_loaded = off_switch.first_load_done(),
                head_guard_loaded = head_guard.first_load_done(),
                action_control_loaded = action_control.is_loaded(),
                "waiting for the first read of the off-switch, head-guard and action-control lists before taking Kafka work"
            );
            logged = true;
        }
        tokio::time::sleep(SWITCH_LISTS_POLL).await;
    }
    if logged {
        info!(
            "off-switch, head-guard and action-control lists loaded; starting the Kafka consumers"
        );
    }
}

pub struct KafkaStartup(JoinHandle<Result<KafkaConsumers>>);

pub fn spawn_kafka_consumers(
    state: Arc<service::AppState>,
    cfg: Config,
    shutdown: xai_service_runner::ShutdownSignal,
) -> KafkaStartup {
    KafkaStartup(tokio::spawn(async move {
        let started = start_kafka_consumers(state, &cfg).await;
        if let Err(e) = &started {
            error!("Kafka consumers failed to start; shutting down: {e:#}");
            shutdown.trigger();
        }
        started
    }))
}

pub async fn shutdown_kafka_startup(
    state: &service::AppState,
    startup: KafkaStartup,
) -> Result<()> {
    let task = startup.0;
    if !task.is_finished() {
        task.abort();
    }
    let result = match task.await {
        Ok(Ok(consumers)) => {
            shutdown_kafka(state, consumers).await;
            return Ok(());
        }
        Ok(Err(e)) => Err(e),
        Err(e) if e.is_cancelled() => Ok(()),
        Err(e) => Err(anyhow::anyhow!("Kafka consumer start task panicked: {e}")),
    };
    state
        .kafka_producers
        .flush_all(Duration::from_secs(5))
        .await;
    result
}

pub async fn start_kafka_consumers(
    state: Arc<service::AppState>,
    cfg: &Config,
) -> Result<KafkaConsumers> {
    wait_for_switch_lists(
        &state.off_switch,
        &state.head_guard,
        &state.action_control,
        &state.dynamic_config,
    )
    .await;
    if !backfill_settings_loaded(&state.backfill) {
        warn!("waiting for the first read of the backfill runs' settings before taking Kafka work");
        while !backfill_settings_loaded(&state.backfill) {
            tokio::time::sleep(SWITCH_LISTS_POLL).await;
        }
        info!("backfill runs' settings read; starting the Kafka consumers");
    }
    let cancel = CancellationToken::new();
    if !cfg.kafka_consumer_enabled {
        info!("KAFKA_CONSUMER_ENABLED=false — Kafka consumer disabled");
        state.kafka_ready.store(true, Ordering::Relaxed);
        return Ok(KafkaConsumers {
            cancel,
            supervisor: tokio::spawn(async {}),
        });
    }

    let backfill_topic = cfg
        .backfill_topic
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty());
    let topics = match backfill_topic {
        Some(topic) => backfill_topic_config(topic),
        None => live_topic_config(&state, cfg)?,
    };
    let backfill_topic = backfill_topic.is_some();
    let default_cluster = cfg.kafka_consumer_mtls_cluster.as_str();
    let default_zone = cfg.kafka_consumer_mtls_zone.as_str();
    info!("topic config: {} topic(s)", topics.len());
    for (topic, entry) in &topics {
        let cluster = entry.cluster(default_cluster);
        let zone = entry.zone(default_zone);
        match &entry.processor {
            TopicConfig::ScoreResult {
                labels,
                negative_labels,
            } => {
                info!(
                    "  topic={topic}, cluster={cluster}, zone={zone}, processor=score_result, labels={}, negative_labels={}",
                    labels.len(),
                    negative_labels.len()
                );
            }
            TopicConfig::StatsOnly { should_log_content } => {
                info!(
                    "  topic={topic}, cluster={cluster}, zone={zone}, processor=stats_only, should_log_content={should_log_content}"
                );
            }
        }
    }

    let s2s_certs = xai_kafka::load_s2s_certs()
        .context("failed to load S2S mTLS client certificate for Kafka consumers")?;

    let default_target = (default_cluster, default_zone);
    let probe_topic = if cfg.kafka_self_delete_enabled {
        preflight_probe_topic(
            topics
                .iter()
                .map(|(t, e)| (t.as_str(), e.cluster(default_cluster), e.zone(default_zone))),
            default_target,
        )
        .map(str::to_owned)
    } else {
        None
    };
    if cfg.kafka_self_delete_enabled && probe_topic.is_none() {
        info!(
            "Kafka broker preflight skipped: no topic on the default target \
             ({default_cluster}/{default_zone}); the runtime watchdog remains the only bad-node gate"
        );
    }
    if let Some(probe_topic) = probe_topic {
        let probe_config = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
            default_cluster,
            probe_topic.clone(),
            format!("{}-preflight", cfg.kafka_group_id()),
            s2s_certs.clone(),
            Some(default_zone),
        )
        .context("failed to configure mTLS consumer preflight")?
        .with_enable_auto_offset_store(false)
        .with_enable_auto_commit(false)
        .with_fetch_timeout_ms(10000)
        .build();

        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let start = std::time::Instant::now();
            match probe_broker_reachability(probe_config.clone(), KAFKA_PREFLIGHT_TIMEOUT).await {
                Ok(()) => {
                    info!(
                        attempt,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        topic = %probe_topic,
                        "Kafka broker preflight ok"
                    );
                    break;
                }
                Err(e) if attempt < KAFKA_PREFLIGHT_ATTEMPTS => {
                    warn!(
                        topic = %probe_topic,
                        "Kafka broker preflight attempt {attempt}/{KAFKA_PREFLIGHT_ATTEMPTS} failed \
                         (retrying in {KAFKA_PREFLIGHT_BACKOFF:?}): {e:#}"
                    );
                    tokio::time::sleep(KAFKA_PREFLIGHT_BACKOFF).await;
                }
                Err(e) => {
                    error!(
                        topic = %probe_topic,
                        "Kafka broker preflight failed after {KAFKA_PREFLIGHT_ATTEMPTS} attempts — \
                         likely a bad node; self-deleting to reschedule: {e:#}"
                    );
                    metrics::KAFKA_SELF_DELETE_TOTAL
                        .with_label_values(&["preflight"])
                        .inc();
                    self_delete_pod().await;
                    return Err(anyhow::anyhow!(
                        "Kafka broker preflight failed after {KAFKA_PREFLIGHT_ATTEMPTS} attempts: {e:#}"
                    ));
                }
            }
        }
    }

    let ca = cfg
        .strato_ca_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/ca/ca-bundle.crt".into());
    let crt = cfg
        .strato_client_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.crt".into());
    let key = cfg
        .strato_client_key_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.key".into());
    let gd = gizmoduck::GizmoduckCoreClient::connect(&gizmoduck::GizmoduckCoreClientConfig {
        zone: cfg.gizmoduck_zone.clone(),
        ca_cert_path: ca,
        client_cert_path: crt,
        client_key_path: key,
        client_id: cfg.gizmoduck_client_id.clone(),
    })
    .await
    .context("failed to create gizmoduck get-V2 fed-grpc client")?;

    #[allow(deprecated)]
    let cred_hpr = Arc::new(
        Strato::new(&cfg.high_page_rank_column, StratoClientConfig::default())
            .context("Failed to create high-page-rank Strato client")?,
    );
    #[allow(deprecated)]
    let cred_grey = Arc::new(
        Strato::new(&cfg.grey_badge_column, StratoClientConfig::default())
            .context("Failed to create grey-badge Strato client")?,
    );
    let cred_clients = strato::CredClients {
        high_page_rank: cred_hpr,
        grey_badge: cred_grey,
    };
    info!(
        high_page_rank_column = %cfg.high_page_rank_column,
        grey_badge_column = %cfg.grey_badge_column,
        "Cred Strato clients initialised"
    );

    #[allow(deprecated)]
    let uas_strato = Arc::new(
        Strato::new(&cfg.uas_column, StratoClientConfig::default())
            .context("Failed to create UAS Strato client")?,
    );
    info!(
        "UAS Strato client initialised for column {}",
        cfg.uas_column
    );

    let post_labels: Option<Arc<dyn post_labels::PostLabelReader>> =
        match post_labels::VfPostLabels::connect(&cfg.post_labels_zone).await {
            Ok(reader) => {
                info!(zone = %cfg.post_labels_zone, "post label reader initialised");
                Some(Arc::new(reader))
            }
            Err(e) => {
                warn!(
                    "post label reader unavailable: {e:#}; page post labels are sent \
                     without checking the labels a post already carries"
                );
                None
            }
        };

    let retry_queue = Arc::new(Mutex::new(BinaryHeap::<RetryEntry>::new()));

    let enforcement_ctx = EnforcementCtx {
        ais_client: state.ais_client.clone(),
        gd,
        cred_clients,
        uas_strato,
        dynamic_config: state.dynamic_config.clone(),
        rules_cache: state.rules_cache.clone(),
        allowlist: state.allowlist.clone(),
        kafka_producer_decisions: state.kafka_producers.decisions().cloned(),
        kafka_producer_decisions_json: state.kafka_producers.decisions_json().cloned(),
        hold_gate: state.hold_gate.clone(),
        off_switch: state.off_switch.clone(),
        head_guard: state.head_guard.clone(),
        action_control: state.action_control.clone(),
        backfill: state.backfill.clone(),
        post_labels,
        #[cfg(test)]
        outcome_log: None,
    };

    {
        let retry_ctx = enforcement_ctx.clone();
        let retry_queue = retry_queue.clone();
        tokio::spawn(async move {
            loop {
                let sleep_until = {
                    let queue = retry_queue.lock().await;
                    queue.peek().map(|e| e.next_retry_at)
                };
                match sleep_until {
                    Some(t) => tokio::time::sleep_until(t).await,
                    None => tokio::time::sleep(Duration::from_secs(5)).await,
                }

                let now = tokio::time::Instant::now();
                let mut ready: Vec<RetryEntry> = Vec::new();
                {
                    let mut queue = retry_queue.lock().await;
                    while queue.peek().is_some_and(|e| e.next_retry_at <= now) {
                        ready.push(queue.pop().unwrap());
                    }
                }

                for mut entry in ready {
                    let (entity_type, entity_id) = Facts::resolve_target(&entry.score);
                    let dedup_key = entry.decision.dedup_key(entity_type, entity_id);
                    let span = tracing::info_span!(
                        "retry",
                        entity_type = entity_type.as_str(),
                        entity_id,
                        attempt = entry.attempt
                    );

                    match run_enforcement(&retry_ctx, &entry.score, &entry.topic, &entry.decision)
                        .instrument(span.clone())
                        .await
                    {
                        Ok(outcome) => {
                            info!(parent: &span, "retry resolved: {}", outcome.status);
                            metrics::RETRY_ATTEMPTS
                                .with_label_values(&["success"])
                                .observe(entry.attempt as f64);
                            if entry.decision.claimed() {
                                write_dedup_outcome(
                                    entry.dedup_cache.as_ref(),
                                    &dedup_key,
                                    &entry.score,
                                    &entry.topic,
                                    &outcome,
                                    entry.claim_nonce,
                                )
                                .await;
                            }
                        }
                        Err(e) if e.downcast_ref::<PermanentAisFailure>().is_some() => {
                            error!(parent: &span, "retry abandoned (permanent failure): {e}");
                            metrics::RETRY_ATTEMPTS
                                .with_label_values(&["permanent"])
                                .observe(entry.attempt as f64);
                            if entry.decision.claimed() {
                                entry
                                    .dedup_cache
                                    .invalidate(&dedup_key, entry.claim_nonce)
                                    .await;
                            }
                        }
                        Err(e) => {
                            entry.attempt += 1;
                            if entry.attempt > MAX_RETRIES {
                                error!(parent: &span, "retry exhausted after {MAX_RETRIES} attempts: {e}");
                                metrics::RETRY_ATTEMPTS
                                    .with_label_values(&["exhausted"])
                                    .observe((entry.attempt - 1) as f64);
                                if entry.decision.claimed() {
                                    entry
                                        .dedup_cache
                                        .invalidate(&dedup_key, entry.claim_nonce)
                                        .await;
                                }
                            } else {
                                let backoff = Duration::from_secs(1 << (entry.attempt - 1));
                                warn!(parent: &span, "retry failed: {e}, next in {backoff:?}");
                                entry.next_retry_at = tokio::time::Instant::now() + backoff;
                                retry_queue.lock().await.push(entry);
                            }
                        }
                    }
                }

                metrics::RETRY_QUEUE_SIZE.set(retry_queue.lock().await.len() as i64);
            }
        });
    }

    let last_progress = Arc::new(AtomicI64::new(0));

    let mut consumers: JoinSet<()> = JoinSet::new();
    let mut error_topics: HashSet<String> = HashSet::new();
    let topics_len = topics.len();

    let dedup_cache: Arc<dyn DedupSlot> = Arc::new(
        state
            .dedup_cache
            .clone()
            .expect("Manhattan dedup cache required when Kafka is enabled"),
    );

    let kafka_group_id = cfg.kafka_group_id();

    let max_messages_per_poll = cfg.kafka_max_messages_per_poll.max(1);
    let max_in_flight = cfg.kafka_max_in_flight.max(1);
    info!(
        max_messages_per_poll,
        max_in_flight, "Kafka consumer batch limits"
    );

    let make_batch_config =
        |topic: &str, cluster: &str, zone: &str| -> Result<BatchConsumerConfig> {
            let kafka = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
                cluster,
                topic.to_owned(),
                topic_consumer_group_id(&kafka_group_id, topic),
                s2s_certs.clone(),
                Some(zone),
            )
            .with_context(|| {
                format!("failed to configure {cluster} mTLS consumer for {topic} (zone {zone})")
            })?
            .with_enable_auto_offset_store(false)
            .with_enable_auto_commit(false)
            .with_fetch_timeout_ms(10000)
            .build();

            Ok(BatchConsumerConfig::new(kafka, SERVICE_NAME)
                .with_max_messages_per_poll(max_messages_per_poll))
        };

    for (topic, entry) in topics {
        let cluster = entry.cluster(default_cluster).to_owned();
        let zone = entry.zone(default_zone).to_owned();

        let batch_config = match make_batch_config(&topic, &cluster, &zone) {
            Ok(config) => config,
            Err(e) => {
                error!("skipping Kafka consumer for topic {topic}: {e:#}");
                metrics::KAFKA_CONSUMER_START_TOTAL
                    .with_label_values(&[topic.as_str(), cluster.as_str(), "config_error"])
                    .inc();
                continue;
            }
        };
        metrics::KAFKA_CONSUMER_START_TOTAL
            .with_label_values(&[topic.as_str(), cluster.as_str(), "started"])
            .inc();
        if is_default_target((&cluster, &zone), default_target) {
            error_topics.insert(topic.clone());
        }
        let cancel = cancel.clone();

        match entry.processor {
            TopicConfig::ScoreResult {
                labels,
                negative_labels,
            } => {
                let processor = ScoreResultProcessor {
                    topic: topic.clone(),
                    labels,
                    negative_labels,
                    ctx: enforcement_ctx.clone(),
                    dedup_cache: dedup_cache.clone(),
                    retry_queue: retry_queue.clone(),
                    last_progress: last_progress.clone(),
                    max_in_flight,
                    backfill_topic,
                };

                consumers.spawn(async move {
                    info!("starting score_result Kafka consumer for topic {topic} on {cluster}");
                    if let Err(e) = run_batch_consumer(batch_config, processor, cancel).await {
                        error!("Kafka consumer for topic {topic} exited with error: {e}");
                    }
                });
            }
            TopicConfig::StatsOnly { should_log_content } => {
                let processor = StatsOnlyProcessor {
                    should_log_content,
                    last_progress: last_progress.clone(),
                };

                consumers.spawn(async move {
                    info!("starting stats_only Kafka consumer for topic {topic} on {cluster}");
                    if let Err(e) = run_batch_consumer(batch_config, processor, cancel).await {
                        error!(
                            "stats_only Kafka consumer for topic {topic} exited with error: {e}"
                        );
                    }
                });
            }
        }
    }

    if consumers.is_empty() {
        if topics_len > 0 {
            error!(
                "no Kafka consumers started ({} topic(s) configured, all skipped); \
                 serving without consumers — check \
                 abuse_enforcement_kafka_consumer_start_total{{result=\"config_error\"}}",
                topics_len
            );
        }
        state.kafka_ready.store(true, Ordering::Relaxed);
    } else {
        tokio::spawn(kafka_health_watchdog(
            last_progress.clone(),
            state.kafka_ready.clone(),
            WatchdogConfig {
                interval: Duration::from_secs(cfg.kafka_watchdog_interval_secs),
                stale_after: Duration::from_secs(cfg.kafka_watchdog_stale_secs),
                stale_self_delete_after: Duration::from_secs(cfg.kafka_watchdog_self_delete_secs),
                error_rate_per_sec: cfg.kafka_watchdog_error_rate_per_sec,
                error_self_delete_after: Duration::from_secs(
                    cfg.kafka_watchdog_error_self_delete_secs,
                ),
                self_delete_enabled: cfg.kafka_self_delete_enabled,
                error_topics,
                quiet_topic: backfill_topic,
            },
        ));
    }

    let rl_config = state.dynamic_config.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            for entity_type in [
                crate::facts::EntityType::User,
                crate::facts::EntityType::Post,
            ] {
                let remaining = rl_config
                    .rate_limit_stats(entity_type)
                    .map_or(-1, |(_used, _cap, remaining)| remaining as i64);
                metrics::RATE_LIMIT_REMAINING
                    .with_label_values(&[entity_type.as_str()])
                    .set(remaining);
            }
        }
    });

    Ok(KafkaConsumers {
        cancel,
        supervisor: supervise(consumers),
    })
}

fn live_topic_config(
    state: &service::AppState,
    cfg: &Config,
) -> Result<HashMap<String, TopicEntry>> {
    let growthbook_enabled = cfg.growthbook_url.is_some() && cfg.growthbook_key.is_some();
    let topic_labels_json: serde_json::Value =
        match std::fs::read_to_string(&cfg.topic_labels_config) {
            Ok(data) => serde_json::from_str(&data).with_context(|| {
                format!(
                    "failed to parse topic-labels config: {}",
                    cfg.topic_labels_config
                )
            })?,
            Err(e) if growthbook_enabled => {
                warn!(
                    "topic-labels config not found ({e}), falling back to GrowthBook / empty config"
                );
                serde_json::Value::Object(Default::default())
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "failed to read topic-labels config {}: {e}",
                    cfg.topic_labels_config
                ));
            }
        };

    let effective_topic_config =
        growthbook::kafka_consumer_topic_labels_from_config(Some(&state.boot_config))
            .unwrap_or(topic_labels_json);
    let mut topics = parse_topic_config(effective_topic_config)?;
    topics.retain(|topic, _| {
        let keep = !backfill::is_backfill_topic(topic);
        if !keep {
            error!(
                "topic map lists {topic}; the live pods never consume a backfill topic (the backfill deployment does)"
            );
        }
        keep
    });
    Ok(topics)
}

fn backfill_topic_config(topic: &str) -> HashMap<String, TopicEntry> {
    HashMap::from([(
        topic.to_owned(),
        TopicEntry {
            cluster: None,
            zone: None,
            processor: TopicConfig::ScoreResult {
                labels: HashSet::new(),
                negative_labels: HashSet::new(),
            },
        },
    )])
}

fn supervise(mut consumers: JoinSet<()>) -> JoinHandle<()> {
    tokio::spawn(async move { while consumers.join_next().await.is_some() {} })
}

fn topic_consumer_group_id(base_group_id: &str, topic: &str) -> String {
    format!("{base_group_id}-{topic}")
}

fn is_default_target(target: (&str, &str), default: (&str, &str)) -> bool {
    target == default
}

fn preflight_probe_topic<'a, I>(topics: I, default: (&str, &str)) -> Option<&'a str>
where
    I: IntoIterator<Item = (&'a str, &'a str, &'a str)>,
{
    topics
        .into_iter()
        .filter(|(_, cluster, zone)| is_default_target((cluster, zone), default))
        .map(|(topic, _, _)| topic)
        .min()
}

fn parse_topic_config(config: serde_json::Value) -> Result<HashMap<String, TopicEntry>> {
    serde_json::from_value(config)
        .context("topic config has wrong shape (expected {topic: {processor, ...}})")
}

pub(crate) fn validate_restart_config(config: &serde_json::Value) -> Result<()> {
    match growthbook::kafka_consumer_topic_labels_from_config(Some(config)) {
        Some(topics) => parse_topic_config(topics).map(drop),
        None => Ok(()),
    }
}

const CONFIG_REFRESH_INTERVAL: Duration = Duration::from_secs(15);

fn config_refresh_tick(
    dynamic_config: &DynamicConfig,
    hold_gate: &overturn_hold::HoldGate,
    boot_config: &serde_json::Value,
) -> restart_on_config_change::Check {
    dynamic_config.refresh();
    hold_gate.publish_config(&dynamic_config.overturn_hold_gate());
    let live = dynamic_config.config().unwrap_or(serde_json::Value::Null);
    let check = restart_on_config_change::check(boot_config, &live, validate_restart_config);
    restart_on_config_change::record(&check);
    check
}

pub fn spawn_config_refresh(
    state: &service::AppState,
    shutdown: xai_service_runner::ShutdownSignal,
) {
    spawn_config_refresh_with(
        state.dynamic_config.clone(),
        state.hold_gate.clone(),
        state.boot_config.clone(),
        shutdown,
        CONFIG_REFRESH_INTERVAL,
    );
}

fn spawn_config_refresh_with(
    dynamic_config: DynamicConfig,
    hold_gate: Arc<overturn_hold::HoldGate>,
    boot_config: serde_json::Value,
    shutdown: xai_service_runner::ShutdownSignal,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await;
        loop {
            tick.tick().await;
            if let restart_on_config_change::Check::Restart(paths) =
                config_refresh_tick(&dynamic_config, &hold_gate, &boot_config)
            {
                info!("restarting to apply config change: paths={paths:?}");
                shutdown.trigger();
                return;
            }
        }
    })
}

#[cfg(test)]
mod kafka_shutdown_tests {
    use super::*;

    #[tokio::test]
    async fn shutdown_waits_for_the_in_flight_batch() {
        let cancel = CancellationToken::new();
        let finished = Arc::new(AtomicBool::new(false));
        let supervisor = tokio::spawn({
            let (cancel, finished) = (cancel.clone(), finished.clone());
            async move {
                cancel.cancelled().await;
                tokio::time::sleep(Duration::from_millis(50)).await;
                finished.store(true, Ordering::SeqCst);
            }
        });

        KafkaConsumers { cancel, supervisor }
            .shutdown(Duration::from_secs(10))
            .await;

        assert!(finished.load(Ordering::SeqCst));
    }

            struct DropFlag(Arc<AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn shutdown_timeout_aborts_the_consumers() {
        let stopped = Arc::new(AtomicBool::new(false));
        let mut consumers = JoinSet::new();
        {
            let guard = DropFlag(stopped.clone());
            consumers.spawn(async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            });
        }
        let started = std::time::Instant::now();

        KafkaConsumers {
            cancel: CancellationToken::new(),
            supervisor: supervise(consumers),
        }
        .shutdown(Duration::from_millis(50))
        .await;
        assert!(started.elapsed() < Duration::from_secs(5));

        tokio::time::timeout(Duration::from_secs(5), async {
            while !stopped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the stuck consumer must be aborted, not detached");
    }
}

#[cfg(test)]
mod router_split_tests {
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware::Next;
    use axum::response::Response;
    use axum::routing::{get, post};

    async fn ok() -> &'static str {
        "ok"
    }

    async fn pass(req: Request<Body>, next: Next) -> Response {
        next.run(req).await
    }

                                            #[test]
    fn same_path_split_across_sub_routers_merges() {
        let protected = Router::new()
            .route("/rate_limit/{entity_type}", post(ok))
            .layer(axum::middleware::from_fn(pass));
        let public = Router::new().route("/rate_limit/{entity_type}", get(ok));
        let _app: Router = protected.merge(public);
    }
}

#[cfg(test)]
mod config_restart_tests {
    use super::validate_restart_config;
    use serde_json::json;

    fn valid_topics() -> serde_json::Value {
        json!({
            "abuse.v3.score_results": {
                "processor": "score_result",
                "labels": ["enforcement_threshold_reached"],
                "cluster": "mltraining",
            },
            "abuse.stats.v1": { "processor": "stats_only" },
        })
    }

    #[test]
    fn valid_topic_map_passes() {
        let config = json!({ "kafka": { "consumer": { "topic_labels": valid_topics() } } });
        validate_restart_config(&config).unwrap();
    }

    #[test]
    fn wrong_shape_topic_map_fails() {
        let array = json!({ "kafka": { "consumer": { "topic_labels": ["abuse.stats.v1"] } } });
        assert!(validate_restart_config(&array).is_err());

        let no_processor = json!({ "kafka": { "consumer": { "topic_labels": {
            "abuse.stats.v1": { "cluster": "phoenix" },
        } } } });
        assert!(validate_restart_config(&no_processor).is_err());

        let unknown_processor = json!({ "kafka": { "consumer": { "topic_labels": {
            "abuse.stats.v1": { "processor": "nope" },
        } } } });
        assert!(validate_restart_config(&unknown_processor).is_err());
    }

    #[test]
    fn absent_topic_map_passes() {
        validate_restart_config(&serde_json::Value::Null).unwrap();
        validate_restart_config(&json!({})).unwrap();
        validate_restart_config(&json!({ "kafka": { "producer": { "decisions": [1] } } })).unwrap();
    }

            #[tokio::test]
    async fn restart_trigger_stops_the_server() {
        use std::time::Duration;

        let server =
            xai_service_runner::ServerBuilder::new(0).drain_period(Duration::from_millis(10));
        let dc = test_dynamic_config(json!({ "topic_labels": {} })).await;
        let refresh = super::spawn_config_refresh_with(
            dc,
            test_hold_gate(),
            json!({}),
            server.shutdown_signal(),
            Duration::from_millis(5),
        );

        tokio::time::timeout(Duration::from_secs(10), server.run())
            .await
            .expect("server did not shut down after the restart trigger")
            .unwrap();
        tokio::time::timeout(Duration::from_secs(10), refresh)
            .await
            .expect("refresh task did not end")
            .unwrap();
    }

            async fn test_dynamic_config(config: serde_json::Value) -> crate::growthbook::DynamicConfig {
        let dc = crate::growthbook::DynamicConfig::new(None, None, false, 1, 1, None, None)
            .await
            .unwrap();
        dc.set_snapshot_for_test(crate::growthbook::ConfigSnapshot {
            config: Some(config),
            ..Default::default()
        });
        dc
    }

    fn test_hold_gate() -> std::sync::Arc<crate::overturn_hold::HoldGate> {
        std::sync::Arc::new(crate::overturn_hold::HoldGate::from_url(None, false, None))
    }

    #[tokio::test]
    async fn refresh_tick_restarts_only_on_a_valid_boot_only_change() {
        use crate::restart_on_config_change::Check;

        let boot =
            json!({ "dry_run": false, "topic_labels": { "a": { "processor": "stats_only" } } });
        let gate = test_hold_gate();

        let mut live = boot.clone();
        live["dry_run"] = json!(true);
        let dc = test_dynamic_config(live).await;
        assert_eq!(
            super::config_refresh_tick(&dc, &gate, &boot),
            Check::Unchanged
        );
        assert!(dc.is_dry_run(), "snapshot kept (no client to refresh from)");

        let mut live = boot.clone();
        live["topic_labels"]["b"] = json!({ "processor": "stats_only" });
        let dc = test_dynamic_config(live).await;
        assert_eq!(
            super::config_refresh_tick(&dc, &gate, &boot),
            Check::Restart(vec!["/topic_labels".to_string()])
        );

        let mut live = boot.clone();
        live["topic_labels"] = json!(["not a map"]);
        let dc = test_dynamic_config(live).await;
        assert_eq!(
            super::config_refresh_tick(&dc, &gate, &boot),
            Check::Invalid
        );
    }

    #[test]
    fn legacy_top_level_topic_map_is_validated() {
        validate_restart_config(&json!({ "topic_labels": valid_topics() })).unwrap();
        let bad = json!({ "topic_labels": { "abuse.stats.v1": "stats_only" } });
        assert!(validate_restart_config(&bad).is_err());
    }
}

#[cfg(test)]
mod kafka_topic_config_tests {
    use super::{
        KAFKA_PREFLIGHT_ATTEMPTS, KAFKA_PREFLIGHT_BACKOFF, KAFKA_PREFLIGHT_TIMEOUT, TopicConfig,
        TopicEntry, consumer_error_total, is_default_target, preflight_probe_topic,
        topic_consumer_group_id,
    };
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;
    use xai_kafka::{KafkaConsumerConfigBuilder, S2sCerts};

    #[test]
    fn dynamic_topics_keep_distinct_existing_group_ids() {
        let base_group_id = "xai-abuse-enforcement-service";

        assert_eq!(
            topic_consumer_group_id(base_group_id, "scores.primary"),
            "xai-abuse-enforcement-service-scores.primary"
        );
        assert_eq!(
            topic_consumer_group_id(base_group_id, "scores.secondary"),
            "xai-abuse-enforcement-service-scores.secondary"
        );
    }

    #[test]
    fn preflight_retry_budget_stays_bounded() {
        assert_eq!(KAFKA_PREFLIGHT_ATTEMPTS, 3);
        assert_eq!(KAFKA_PREFLIGHT_TIMEOUT, Duration::from_secs(10));
        assert_eq!(KAFKA_PREFLIGHT_BACKOFF, Duration::from_secs(2));
    }


    fn parse(json: serde_json::Value) -> HashMap<String, TopicEntry> {
        serde_json::from_value(json).expect("topic config should deserialize")
    }

                #[test]
    fn topic_entry_without_override_uses_deployment_default() {
        let topics = parse(json!({
            "abuse.v3.score_results": {
                "processor": "score_result",
                "labels": ["enforcement_threshold_reached"],
            },
            "abuse.stats.v1": { "processor": "stats_only" },
        }));

        for topic in ["abuse.v3.score_results", "abuse.stats.v1"] {
            let entry = &topics[topic];
            assert_eq!(entry.cluster("phoenix"), "phoenix");
            assert_eq!(entry.zone("atla"), "atla");
            assert!(entry.cluster.is_none());
        }
        assert!(matches!(
            topics["abuse.v3.score_results"].processor,
            TopicConfig::ScoreResult { .. }
        ));
        assert!(matches!(
            topics["abuse.stats.v1"].processor,
            TopicConfig::StatsOnly { .. }
        ));
    }

                    #[test]
    fn absent_null_and_blank_all_resolve_to_the_default() {
        let topics = parse(json!({
            "omitted": { "processor": "stats_only" },
            "explicit_null": { "processor": "stats_only", "cluster": null, "zone": null },
            "empty_string": { "processor": "stats_only", "cluster": "", "zone": "" },
            "whitespace": { "processor": "stats_only", "cluster": "   ", "zone": "\t" },
        }));

        for topic in ["omitted", "explicit_null", "empty_string", "whitespace"] {
            let entry = &topics[topic];
            assert_eq!(entry.cluster("phoenix"), "phoenix", "cluster for {topic}");
            assert_eq!(entry.zone("atla"), "atla", "zone for {topic}");
        }

        let padded = parse(json!({
            "t": { "processor": "stats_only", "cluster": " mltraining ", "zone": " atla " },
        }));
        assert_eq!(padded["t"].cluster("phoenix"), "mltraining");
        assert_eq!(padded["t"].zone("atla"), "atla");
    }

            #[test]
    fn topic_entry_override_wins_per_field() {
        let topics = parse(json!({
            "abuse.candidates.impersonation_scam": {
                "processor": "score_result",
                "labels": ["impersonation_scam_threshold_reached"],
                "cluster": "mltraining",
                "zone": "atla",
            },
            "abuse.zone_only.v1": { "processor": "stats_only", "zone": "pdxa" },
        }));

        let scam = &topics["abuse.candidates.impersonation_scam"];
        assert_eq!(scam.cluster("phoenix"), "mltraining");
        assert_eq!(scam.zone("atla"), "atla");
        let TopicConfig::ScoreResult { ref labels, .. } = scam.processor else {
            panic!("expected score_result processor alongside the cluster override");
        };
        assert!(labels.contains("impersonation_scam_threshold_reached"));

        let zone_only = &topics["abuse.zone_only.v1"];
        assert_eq!(zone_only.cluster("phoenix"), "phoenix");
        assert_eq!(zone_only.zone("atla"), "pdxa");
    }

                            #[test]
    fn preflight_probes_first_default_target_topic_only() {
        let topics = [
            ("abuse.candidates.impersonation_scam", "mltraining", "atla"),
            ("abuse.v3.score_results", "phoenix", "atla"),
            ("abuse.embeddings.user_decisions", "phoenix", "atla"),
        ];

        assert_eq!(
            preflight_probe_topic(topics, ("phoenix", "atla")),
            Some("abuse.embeddings.user_decisions")
        );
    }

                    #[test]
    fn zone_only_override_is_not_the_default_target() {
        assert!(!is_default_target(("phoenix", "pdxa"), ("phoenix", "atla")));
        assert!(!is_default_target(
            ("mltraining", "atla"),
            ("phoenix", "atla")
        ));
        assert!(is_default_target(("phoenix", "atla"), ("phoenix", "atla")));

        let topics = [
            ("abuse.a.zone_override", "phoenix", "pdxa"),
            ("abuse.b.default", "phoenix", "atla"),
        ];
        assert_eq!(
            preflight_probe_topic(topics, ("phoenix", "atla")),
            Some("abuse.b.default"),
            "a zone-only override must not be chosen as the boot probe"
        );
    }

            #[test]
    fn preflight_skipped_when_no_topic_on_default_target() {
        let topics = [("abuse.candidates.impersonation_scam", "mltraining", "atla")];

        assert_eq!(preflight_probe_topic(topics, ("phoenix", "atla")), None);
        assert_eq!(preflight_probe_topic([], ("phoenix", "atla")), None);
    }

                        #[test]
    fn error_flood_signal_counts_default_target_topics_only() {
        let default_topic = "test.error.scope.default";
        let override_topic = "test.error.scope.override";
        let bump = |topic: &str, times: u64| {
            for _ in 0..times {
                xai_kafka::metrics::CONSUMER_ERROR_COUNT
                    .with_label_values(&[topic, "group", "BrokerTransportFailure"])
                    .inc();
            }
        };

        let watched: HashSet<String> = [default_topic.to_owned()].into_iter().collect();
        let before = consumer_error_total(&watched);

        bump(override_topic, 50);
        assert_eq!(
            consumer_error_total(&watched),
            before,
            "override-cluster topic errors must not move the eviction signal"
        );

        bump(default_topic, 3);
        assert_eq!(
            consumer_error_total(&watched),
            before + 3.0,
            "default-cluster topic errors must still be counted"
        );

        assert_eq!(consumer_error_total(&HashSet::new()), 0.0);
    }

                            #[test]
    fn override_cluster_and_zone_reach_the_kafka_bootstrap() {
        let certs = S2sCerts::new("/ca.crt", "/tls.crt", "/tls.key");

        let config = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
            "mltraining",
            "abuse.candidates.impersonation_scam",
            "xai-abuse-enforcement-service-fou-staging-abuse.candidates.impersonation_scam",
            certs.clone(),
            Some("atla"),
        )
        .expect("mltraining/atla is a registered mTLS cluster+zone")
        .build();

        assert_eq!(
            config.base_config.dest,
            "mltraining-mtls-bootstrap.kafka.prod.atla-prod-messaging.kafka.kube.atla.twitter.com:9095"
        );
        assert_eq!(
            config.base_config.topic,
            "abuse.candidates.impersonation_scam"
        );

        let default_cluster = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
            "phoenix",
            "abuse.v3.score_results",
            "group",
            certs.clone(),
            Some("atla"),
        )
        .expect("phoenix/atla is a registered mTLS cluster+zone")
        .build();
        assert!(
            default_cluster
                .base_config
                .dest
                .starts_with("phoenix-mtls-bootstrap."),
            "unexpected default dest: {}",
            default_cluster.base_config.dest
        );
    }

                    #[test]
    fn unresolvable_override_is_an_error_not_a_panic() {
        let certs = S2sCerts::new("/ca.crt", "/tls.crt", "/tls.key");

        for (cluster, zone) in [
            ("not-a-cluster", "atla"),
            ("bluebird-1", "atla"),
            ("mltraining", "iad"),
        ] {
            assert!(
                KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
                    cluster,
                    "some.topic",
                    "group",
                    certs.clone(),
                    Some(zone),
                )
                .is_err(),
                "expected {cluster}/{zone} to be rejected"
            );
        }
    }
}

#[cfg(test)]
mod dedup_retention_tests {
    use super::{DedupRetention, dedup_retention_for, outcome_holds_full_dedup};

                            #[test]
    fn hold_lookup_failed_releases_the_claim_hold_overturned_is_a_skip() {
        assert_eq!(
            dedup_retention_for(crate::overturn_hold::STATUS_HOLD_LOOKUP_FAILED),
            DedupRetention::Release
        );
        assert_eq!(
            dedup_retention_for(crate::overturn_hold::STATUS_HOLD_OVERTURNED),
            DedupRetention::Skip
        );
        assert_eq!(dedup_retention_for("success"), DedupRetention::Full);
        assert_eq!(dedup_retention_for("dry_run"), DedupRetention::Skip);
        assert_eq!(dedup_retention_for("dedup_skipped"), DedupRetention::Skip);
    }

                        #[test]
    fn only_success_holds_full_dedup_window() {
        assert!(outcome_holds_full_dedup("success"));
        for skip in [
            "dry_run",
            "dedup_skipped",
            "very_high_follower_count",
            "high_follower_count",
            "pagerank_skipped",
            "gizmoduck_skipped",
            "user_not_found",
            "test_user_skipped",
            "user_in_allowlist",
            "post_in_allowlist",
            "rule_eval_error",
            "max_enforcement_reached",
            "invalid_entity_id",
            crate::action_control::STATUS_NO_ACTIONS,
            crate::overturn_hold::STATUS_HOLD_OVERTURNED,
            crate::overturn_hold::STATUS_HOLD_LOOKUP_FAILED,
        ] {
            assert!(!outcome_holds_full_dedup(skip), "{skip} must not hold 24h");
        }
    }
}

#[cfg(test)]
mod incoming_dedup_class_tests {
    use super::*;
    use crate::dedup_cache::{
        CLASS_CHALLENGE, CLASS_LABEL, CLASS_NONE, CLASS_SUSPEND, held_entry_meta, is_duplicate,
    };
    use serde_json::json;

                            const RULES: &str = r#"
for_entity: user
rules:
  - id: user_in_allowlist
    when: user_allowlist.is_allowlisted
    then: {kind: skip, reason: user_in_allowlist}
  - id: user_not_found
    when: "!user.present"
    then: {kind: skip, reason: user_not_found}
  - id: high_follower_count
    when: cred.follower_count >= 12.34
    then: {kind: skip, reason: high_follower_count}
  - id: act_bb1_post_spam_bot_suspend
    when: '"bb1_post_spam_bot_suspend" in score.labels'
    then: {kind: act_suspend_user, perm: false, policy: PlatformManipulation}
  - id: act_bb1_post_spam_bot_bounce
    when: '"bb1_post_spam_bot_bounce" in score.labels'
    then:
      kind: act_all
      actions:
        - {kind: act_add_labels_v2, labels: [SpamHighRecall], ttl_msec: 2592000000}
        - {kind: act_spam_liveness_check}
  - id: act_bb1_user_suspend
    when: '"bb1_user_suspend" in score.labels'
    then: {kind: act_suspend_user, perm: false, policy: PlatformManipulation}
  - id: act_liveness_check_and_spam_high_recall_7d
    when: '"liveness_check_and_spam_high_recall_7d" in score.labels'
    then:
      kind: act_all
      actions:
        - {kind: act_add_labels_v2, labels: [SpamHighRecall], ttl_msec: 604800000}
        - {kind: act_spam_liveness_check}
  - id: act_bb1_user_bounce
    when: '"bb1_user_bounce" in score.labels'
    then:
      kind: act_all
      actions:
        - {kind: act_add_labels_v2, labels: [SpamHighRecall], ttl_msec: 2592000000}
        - {kind: act_spam_liveness_check}
  - id: already_spam_high_recall_labeled_llm_slop
    when: '"llm_slop_user" in score.labels && "SpamHighRecall" in user.labels'
    then: {kind: skip, reason: already_labeled}
  - id: act_add_llm_slop_label
    when: '"llm_slop_user" in score.labels'
    then: {kind: act_add_labels_v2, labels: [SpamHighRecall], ttl_msec: 604800000}
  - id: already_suspended
    when: user.suspended
    then: {kind: skip, reason: already_suspended}
  - id: act_cusp_liveness
    when: '"enforcement_cusp_liveness" in score.labels && !("enforcement_threshold_reached" in score.labels)'
    then: {kind: act_spam_liveness_check}
  - id: hold_cluster_spam
    when: score.model_version.startsWith("cluster_spam")
    then: {kind: skip, reason: hold_cluster_spam}
  - id: act_suspend
    when: "true"
    then: {kind: act_suspend_user, perm: false, policy: PlatformManipulation}
"#;

    const UID: i64 = 1_700_000_042;

    async fn config(config: serde_json::Value) -> DynamicConfig {
        let dc = DynamicConfig::new(None, None, false, 1_000, 1_000, None, None)
            .await
            .unwrap();
        dc.set_snapshot_for_test(crate::growthbook::ConfigSnapshot {
            config: Some(config),
            rules_user: Some(Arc::from(RULES)),
            rules_post: None,
        });
        dc
    }

    fn row(head: &str, model: &str, labels: &[&str]) -> abuse_proto::ScoreResult {
        abuse_proto::ScoreResult {
            user_id: UID,
            model_version: model.into(),
            summary: Some(abuse_proto::SummaryCounters {
                labels: labels.iter().map(|l| (*l).to_owned()).collect(),
                fired_heads: vec![abuse_proto::FiredHead {
                    name: head.into(),
                    score: 0.99,
                    threshold: 0.9,
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn class_of(dc: &DynamicConfig, score: &abuse_proto::ScoreResult) -> u8 {
        incoming_dedup_class(dc, &rules::RulesCache::new(), score)
    }

        fn held(status: &str, acted_class: u8) -> Vec<u8> {
        let entry = json!({"status": status, "acted_class": acted_class, "topic": "t"});
        zstd::encode_all(entry.to_string().as_bytes(), 3).unwrap()
    }

        fn claim(class: u8) -> Vec<u8> {
        json!({"ts": 1, "acted_class": class})
            .to_string()
            .into_bytes()
    }

    fn acts_over(incoming: u8, held: &[u8]) -> bool {
        !is_duplicate(incoming, &held_entry_meta(held))
    }

    #[tokio::test]
    async fn class_is_the_strongest_action_of_the_decision() {
        let dc = config(json!({})).await;
        let class = |labels: &[&str]| class_of(&dc, &row("AnyHead", "m", labels));
        assert_eq!(class(&["bb1_post_spam_bot_suspend"]), CLASS_SUSPEND);
        assert_eq!(class(&["bb1_post_spam_bot_bounce"]), CLASS_CHALLENGE);
        assert_eq!(
            class(&["liveness_check_and_spam_high_recall_7d"]),
            CLASS_CHALLENGE
        );
        assert_eq!(class(&["enforcement_cusp_liveness"]), CLASS_CHALLENGE);
        assert_eq!(class(&["llm_slop_user"]), CLASS_LABEL);
        assert_eq!(class(&[]), CLASS_SUSPEND);
    }

            #[tokio::test]
    async fn a_decision_that_cannot_act_claims_no_class() {
        let dc = config(json!({})).await;
        let score = row("ClusterSpam", "cluster_spam_v3", &["cluster_spam_suspend"]);
        assert_eq!(class_of(&dc, &score), CLASS_NONE);
        assert!(!acts_over(
            CLASS_NONE,
            &held("high_follower_count", CLASS_NONE)
        ));
    }

                #[tokio::test]
    async fn heads_without_a_map_entry_get_their_action_class() {
        let cases = [
            ("GraphFleetM4", &["bb1_user_suspend"][..], CLASS_SUSPEND),
            (
                "GraphNeighbourNetwork",
                &["bb1_user_suspend"][..],
                CLASS_SUSPEND,
            ),
            ("IsCoordinatedSpammer", &[][..], CLASS_SUSPEND),
            (
                "UserEmbeddingOtherSystemBot",
                &["bb1_user_bounce"][..],
                CLASS_CHALLENGE,
            ),
            (
                "UserEmbeddingReplyBoostFarm",
                &["liveness_check_and_spam_high_recall_7d"][..],
                CLASS_CHALLENGE,
            ),
            (
                "UserEmbeddingLikeSpamBot",
                &["enforcement_cusp_liveness"][..],
                CLASS_CHALLENGE,
            ),
            ("llm_slop_user", &["llm_slop_user"][..], CLASS_LABEL),
        ];
        let empty = config(json!({})).await;
        let stale_map = config(json!({"dedup_head_action_class": {
            "GraphFleetM4": 0,
            "UserEmbeddingOtherSystemBot": "label",
            "llm_slop_user": "suspend",
            "UserEmbeddingPostSpamBot": "suspend",
        }}))
        .await;
        for (head, labels, want) in cases {
            let score = row(head, "m", labels);
            assert_eq!(class_of(&empty, &score), want, "{head}");
            assert_eq!(
                class_of(&stale_map, &score),
                want,
                "{head} with a stale map"
            );
        }
        let bounce = row(
            "UserEmbeddingPostSpamBot",
            "m",
            &["bb1_post_spam_bot_bounce"],
        );
        assert_eq!(class_of(&stale_map, &bounce), CLASS_CHALLENGE);
    }

                #[tokio::test]
    async fn a_suspend_acts_over_an_earlier_liveness_check_and_label_from_any_head() {
        let dc = config(json!({})).await;
        let liveness_and_label = held("success", CLASS_CHALLENGE);
        let label = held("success", CLASS_LABEL);
        for (head, labels) in [
            ("GraphFleetM4", &["bb1_user_suspend"][..]),
            ("ForgedLivenessCompleter", &["bb1_user_suspend"][..]),
            (
                "UserEmbeddingPostSpamBot",
                &["bb1_post_spam_bot_suspend"][..],
            ),
            ("ImpersonationScam", &[][..]),
            ("BrandNewHead", &["bb1_user_suspend"][..]),
        ] {
            let suspend = class_of(&dc, &row(head, "m", labels));
            assert!(
                acts_over(suspend, &liveness_and_label),
                "{head} over liveness"
            );
            assert!(acts_over(suspend, &label), "{head} over label");
            assert!(
                !acts_over(suspend, &held("success", CLASS_SUSPEND)),
                "{head} repeat"
            );
        }
        let bounce = class_of(
            &dc,
            &row("UserEmbeddingOtherSystemBot", "m", &["bb1_user_bounce"]),
        );
        assert!(acts_over(bounce, &label));
        assert!(!acts_over(bounce, &liveness_and_label));
        let slop = class_of(&dc, &row("llm_slop_user", "m", &["llm_slop_user"]));
        assert!(!acts_over(slop, &label));
        let suspend = class_of(&dc, &row("GraphFleetM4", "m", &["bb1_user_suspend"]));
        assert!(acts_over(suspend, &claim(bounce)));
        assert!(!acts_over(bounce, &claim(bounce)));
    }

            #[tokio::test]
    async fn dry_run_and_skip_entries_hold_no_strength() {
        let dc = config(json!({})).await;
        for labels in [&["llm_slop_user"][..], &["bb1_user_bounce"][..], &[][..]] {
            let incoming = class_of(&dc, &row("AnyHead", "m", labels));
            assert!(
                acts_over(incoming, &held("dry_run", CLASS_NONE)),
                "{labels:?}"
            );
            assert!(
                acts_over(incoming, &held("already_labeled", CLASS_NONE)),
                "{labels:?}"
            );
        }
    }
}

#[cfg(test)]
mod off_switch_hook_tests {
    use super::*;
    use crate::decision::ActionSpec;
    use crate::off_switch::test_support::{loaded, record};
    use serde_json::json;
    use xai_abuse_proto::enforcement::{FiredHead, ScoreResult, SummaryCounters};

    const TOPIC: &str = "abuse.embeddings.user_decisions";

    fn score(head: &str, classifier: &str) -> ScoreResult {
        ScoreResult {
            user_id: 42,
            model_version: classifier.into(),
            summary: Some(SummaryCounters {
                fired_heads: vec![FiredHead {
                    name: head.into(),
                    score: 0.99,
                    threshold: 0.9,
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn facts_for(score: &ScoreResult) -> Facts {
        Facts {
            entity_type: EntityType::User,
            entity_id: 42,
            user_id: 42,
            topic: TOPIC.into(),
            score: ScoreFacts::from_score(score),
            entity: EntityFacts::User(UserFacts::default()),
            uas: None,
        }
    }

        fn composite() -> Vec<ActionSpec> {
        vec![
            ActionSpec::AddLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: Some(86_400_000),
            },
            ActionSpec::SpamLivenessCheck,
        ]
    }

        async fn config(gb: serde_json::Value) -> (DynamicConfig, Arc<AtomicI64>) {
        let (user_cap, tickets) = sliding_window::counting_limiter_for_test();
        let dc = DynamicConfig::new(None, None, false, 1_000, 1_000, Some(user_cap), None)
            .await
            .unwrap();
        dc.set_snapshot_for_test(growthbook::ConfigSnapshot {
            config: Some(gb),
            ..Default::default()
        });
        (dc, tickets)
    }

    struct Run {
        outcome: Option<abuse_proto::DecisionOutcome>,
        hold_gate_info: BTreeMap<String, String>,
    }

    async fn run(
        sw: &off_switch::OffSwitch,
        dc: &DynamicConfig,
        score: &ScoreResult,
        specs: &[ActionSpec],
        dry_run: bool,
    ) -> Run {
        let guards = head_guard::HeadGuard::new(
            Arc::new(action_control::test_support::MemControlStore::default()),
            None,
        );
        run_guarded(sw, &guards, dc, score, specs, dry_run).await
    }

    async fn run_guarded(
        sw: &off_switch::OffSwitch,
        guards: &head_guard::HeadGuard,
        dc: &DynamicConfig,
        score: &ScoreResult,
        specs: &[ActionSpec],
        dry_run: bool,
    ) -> Run {
        let facts = facts_for(score);
        let mut hold_gate_info =
            BTreeMap::from([("overturn_hold_mode".to_owned(), "shadow".to_owned())]);
        let outcome = final_checks(
            sw,
            guards,
            dc,
            specs,
            &facts,
            score,
            TOPIC,
            dry_run,
            &mut hold_gate_info,
            None,
        )
        .await
        .unwrap();
        Run {
            outcome,
            hold_gate_info,
        }
    }

    #[tokio::test]
    async fn seen_pair_is_recorded_even_when_switched_off_and_not_when_killed() {
        use off_switch::SeenOutcome;
        let (sw, _) = loaded(&[record("SeenHookHead", None, true, None)]).await;
        let (on, _) = config(json!({})).await;
        let (killed, _) = config(json!({off_switch::CONFIG_KEY: {"enabled": false}})).await;

        assert_eq!(
            record_seen_pair(&sw, &killed, &score("SeenHookHead", "clf_v1"), TOPIC),
            None
        );
        assert_eq!(
            record_seen_pair(&sw, &on, &score("SeenHookHead", "clf_v1"), TOPIC),
            Some(SeenOutcome::Queued),
            "a switched-off head is still recorded"
        );
        assert_eq!(
            record_seen_pair(&sw, &on, &score("SeenHookHead", "clf_v1"), TOPIC),
            Some(SeenOutcome::Recent)
        );
    }

    #[tokio::test]
    async fn headless_decisions_are_counted_and_never_switched_off() {
        use off_switch::SeenOutcome;
        let topic = "headless.counter.test";
        let (sw, _) = loaded(&[record("x", None, true, None)]).await;
        let (on, tickets) = config(json!({})).await;
        let (killed, _) = config(json!({off_switch::CONFIG_KEY: {"enabled": false}})).await;
        let counter = metrics::OFF_SWITCH_HEADLESS_TOTAL.with_label_values(&[topic, "clf_v1"]);
        let before = counter.get();

        let mut headless = score("x", "clf_v1");
        headless.summary = None;
        assert_eq!(
            record_seen_pair(&sw, &on, &headless, topic),
            Some(SeenOutcome::Invalid),
            "not in the seen list"
        );
        assert_eq!(record_seen_pair(&sw, &killed, &headless, topic), None);
        assert_eq!(
            counter.get(),
            before + 2,
            "counted with the feature on or off"
        );

        let mut no_fired = score("x", "clf_v1");
        no_fired.summary.as_mut().unwrap().fired_heads.clear();
        record_seen_pair(&sw, &on, &no_fired, topic);
        assert_eq!(counter.get(), before + 3);

        assert!(
            run(&sw, &on, &headless, &composite(), false)
                .await
                .outcome
                .is_none()
        );
        assert_eq!(
            tickets.load(Ordering::Relaxed),
            1,
            "acts like any unswitched decision"
        );

        record_seen_pair(&sw, &on, &score("x", "clf_v1"), topic);
        assert_eq!(
            counter.get(),
            before + 3,
            "a decision with a head is not counted"
        );
    }

        struct Lists {
        sw: Arc<off_switch::OffSwitch>,
        sw_store: Arc<off_switch::test_support::MemStore>,
        hg: Arc<head_guard::HeadGuard>,
        hg_store: Arc<action_control::test_support::MemControlStore>,
        ac: Arc<action_control::ActionControl>,
        ac_store: Arc<action_control::test_support::MemControlStore>,
    }

    impl Lists {
        fn loaded(&self, dc: &DynamicConfig) -> bool {
            switch_lists_loaded(&self.sw, &self.hg, &self.ac, dc)
        }
    }

    fn lists_for_test() -> Lists {
        use crate::action_control::test_support::MemControlStore;
        use crate::off_switch::test_support::MemStore;
        let sw_store = Arc::new(MemStore::default());
        sw_store.fail_scan.store(true, Ordering::Relaxed);
        let hg_store = Arc::new(MemControlStore::default());
        hg_store.fail_scan.store(true, Ordering::Relaxed);
        let ac_store = Arc::new(MemControlStore::default());
        ac_store.fail_scan.store(true, Ordering::Relaxed);
        Lists {
            sw: Arc::new(off_switch::OffSwitch::new(
                sw_store.clone(),
                Arc::new(MemStore::default()),
            )),
            sw_store,
            hg: Arc::new(head_guard::HeadGuard::new(hg_store.clone(), None)),
            hg_store,
            ac: Arc::new(action_control::ActionControl::new(
                ac_store.clone(),
                "test".into(),
                String::new(),
            )),
            ac_store,
        }
    }

    const RELEASE_JSON: &str = r#"--backfill-service-config-json={"kafka":{"producer":{"decisions":{"enabled":true,"topic":"t"}}},"action_control":{"enabled":true}}"#;

    fn release_cfg(extra: &[&str]) -> Config {
        use clap::Parser;
        let mut args = vec![
            "xai-abuse-enforcement-service",
            "--backfill-topic=abuse.enforcement.backfill.v1",
        ];
        if !extra
            .iter()
            .any(|a| a.starts_with("--backfill-service-config-json"))
        {
            args.push(RELEASE_JSON);
        }
        args.extend_from_slice(extra);
        Config::parse_from(args)
    }

    #[test]
    fn the_backfill_release_reads_no_growthbook() {
        let (dc, writer) = backfill_release_config(&release_cfg(&[]), None, None).unwrap();
        assert!(!dc.reads_growthbook());
        assert!(writer.is_none());
        assert!(dc.action_control_config().enabled, "from the Helm JSON");
        assert_eq!(
            dc.enforcement_rules_yaml(EntityType::User),
            None,
            "baked-in rules"
        );
        assert!(growthbook::kafka_producers_from_config(dc.config().as_ref())["decisions"].enabled);
        dc.refresh();
        assert!(
            dc.action_control_config().enabled,
            "a refresh changes nothing"
        );

        for flag in [
            "--growthbook-url=https://cdn.growthbook.io",
            "--growthbook-key=sdk-key",
            "--growthbook-admin-api-key=admin-key",
        ] {
            let err = backfill_release_config(&release_cfg(&[flag]), None, None)
                .err()
                .unwrap_or_else(|| panic!("{flag} accepted"));
            assert!(
                err.to_string().contains("reads no GrowthBook"),
                "{flag}: {err}"
            );
        }
        for bad in [
            "--backfill-service-config-json=",
            "--backfill-service-config-json=not json",
            "--backfill-service-config-json=[1]",
        ] {
            assert!(
                backfill_release_config(&release_cfg(&[bad]), None, None).is_err(),
                "{bad} accepted"
            );
        }
    }

    #[test]
    fn the_backfill_keys_and_ceiling_come_from_the_helm_env() {
        use clap::Parser;
        let live = Config::parse_from(["xai-abuse-enforcement-service"]);
        let access = backfill::BackfillAccess::new(
            &live.backfill_page_keys,
            &live.backfill_sender_keys,
            live.backfill_max_rate_per_sec,
        );
        assert_eq!(
            access,
            backfill::BackfillAccess::default(),
            "live pods: nobody"
        );
        let cfg = release_cfg(&[
            "--backfill-page-keys=dev",
            "--backfill-sender-keys=backfill_sender",
            "--backfill-max-rate-per-sec=1000",
        ]);
        let access = backfill::BackfillAccess::new(
            &cfg.backfill_page_keys,
            &cfg.backfill_sender_keys,
            cfg.backfill_max_rate_per_sec,
        );
        assert!(access.page_keys.contains("dev") && access.page_keys.len() == 1);
        assert!(access.sender_keys.contains("backfill_sender"));
        assert_eq!(access.max_rate_ceiling, 1_000);
    }

    #[tokio::test]
    async fn backfill_readiness_waits_for_the_first_settings_read_only_on_the_backfill_pods() {
        let store = Arc::new(action_control::test_support::MemControlStore::default());
        let live = backfill::Backfill::new(store.clone(), None, String::new());
        assert!(
            backfill_settings_loaded(&live),
            "live pods never read the settings"
        );
        let claims: Arc<dyn backfill_claims::ClaimStore> =
            Arc::new(backfill_claims::test_support::MemClaimStore::default());
        let pod = backfill::Backfill::new(store.clone(), Some(claims), String::new());
        assert!(!backfill_settings_loaded(&pod));
        store.fail_get.store(true, Ordering::Relaxed);
        assert!(pod.refresh_settings().await.is_err());
        assert!(
            !backfill_settings_loaded(&pod),
            "a failed read is not a load"
        );
        store.fail_get.store(false, Ordering::Relaxed);
        pod.refresh_settings().await.unwrap();
        assert!(
            backfill_settings_loaded(&pod),
            "no record is a completed read (budget 0)"
        );
        assert_eq!(pod.config().budget(EntityType::User), 0);
    }

    #[tokio::test]
    async fn readiness_waits_for_the_first_read_of_every_list() {
        let l = lists_for_test();
        let (on, _) = config(json!({})).await;
        let (sw_off, _) = config(json!({off_switch::CONFIG_KEY: {"enabled": false}})).await;
        let (ac_ignored, _) =
            config(json!({action_control::CONFIG_KEY: {"ignore_store": true}})).await;
        let (both_off, _) = config(json!({
            off_switch::CONFIG_KEY: {"enabled": false},
            action_control::CONFIG_KEY: {"ignore_store": true},
        }))
        .await;

        assert!(!l.loaded(&on), "no read yet");
        assert!(l.loaded(&both_off), "both features off: no wait");
        assert!(!l.loaded(&sw_off), "action control still unread");
        assert!(!l.loaded(&ac_ignored), "off switch still unread");
        assert!(l.sw.refresh().await.is_err());
        assert!(l.hg.refresh().await.is_err());
        assert!(l.ac.refresh().await.is_err());
        assert!(!l.loaded(&on), "a failed read is not a load");

        l.sw_store.fail_scan.store(false, Ordering::Relaxed);
        l.sw.refresh().await.unwrap();
        assert!(!l.loaded(&ac_ignored), "the guard list still unread");
        l.hg_store.fail_scan.store(false, Ordering::Relaxed);
        l.hg.refresh().await.unwrap();
        assert!(!l.loaded(&on), "action control still unread");
        assert!(l.loaded(&ac_ignored));

        l.ac_store.fail_scan.store(false, Ordering::Relaxed);
        l.ac.refresh().await.unwrap();
        assert!(l.loaded(&on));
        l.sw_store.fail_scan.store(true, Ordering::Relaxed);
        l.hg_store.fail_scan.store(true, Ordering::Relaxed);
        l.ac_store.fail_scan.store(true, Ordering::Relaxed);
        assert!(l.sw.refresh().await.is_err());
        assert!(l.hg.refresh().await.is_err());
        assert!(l.ac.refresh().await.is_err());
        assert!(
            l.loaded(&on),
            "later failures keep the last good lists and stay ready"
        );
    }

                #[tokio::test]
    async fn an_unread_guard_list_alone_holds_the_pod() {
        let l = lists_for_test();
        let (on, _) = config(json!({})).await;
        let (sw_off, _) = config(json!({off_switch::CONFIG_KEY: {"enabled": false}})).await;
        l.sw_store.fail_scan.store(false, Ordering::Relaxed);
        l.sw.refresh().await.unwrap();
        l.ac_store.fail_scan.store(false, Ordering::Relaxed);
        l.ac.refresh().await.unwrap();
        assert!(!l.loaded(&on));
        assert!(l.loaded(&sw_off));
        l.hg_store.fail_scan.store(false, Ordering::Relaxed);
        l.hg.refresh().await.unwrap();
        assert!(l.loaded(&on));
    }

    fn wait_for(l: &Lists, dc: &DynamicConfig) -> tokio::task::JoinHandle<()> {
        let (sw, hg, ac, dc) = (l.sw.clone(), l.hg.clone(), l.ac.clone(), dc.clone());
        tokio::spawn(async move { wait_for_switch_lists(&sw, &hg, &ac, &dc).await })
    }

    #[tokio::test]
    async fn the_consumer_wait_holds_until_every_list_loads_or_their_features_go_off() {
        let l = lists_for_test();
        let (dc, _) = config(json!({})).await;
        let waiter = wait_for(&l, &dc);
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(!waiter.is_finished(), "no list read");

        l.sw_store.fail_scan.store(false, Ordering::Relaxed);
        l.sw.refresh().await.unwrap();
        l.ac_store.fail_scan.store(false, Ordering::Relaxed);
        l.ac.refresh().await.unwrap();
        tokio::time::sleep(Duration::from_millis(450)).await;
        assert!(!waiter.is_finished(), "the guard list still unread");

        l.hg_store.fail_scan.store(false, Ordering::Relaxed);
        l.hg.refresh().await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("released once every list is read")
            .unwrap();

        let l = lists_for_test();
        let waiter = wait_for(&l, &dc);
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(!waiter.is_finished());
        dc.set_snapshot_for_test(growthbook::ConfigSnapshot {
            config: Some(json!({
                off_switch::CONFIG_KEY: {"enabled": false},
                action_control::CONFIG_KEY: {"ignore_store": true},
            })),
            ..Default::default()
        });
        tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("released by the GrowthBook kill switches")
            .unwrap();
    }

    #[tokio::test]
    async fn switched_off_head_skips_whole_decision_without_a_rate_cap_ticket() {
        let (sw, _) = loaded(&[record("OffHeadA", None, true, None)]).await;
        let (dc, tickets) = config(json!({})).await;
        let hits = metrics::OFF_SWITCH_TOTAL.with_label_values(&[TOPIC, "OffHeadA", "head"]);
        let hits_before = hits.get();

        let r = run(&sw, &dc, &score("OffHeadA", "clf_v1"), &composite(), true).await;

        let outcome = r.outcome.expect("decision ends at the off switch");
        assert_eq!(outcome.status, off_switch::STATUS_SWITCHED_OFF);
        assert!(outcome.dry_run, "carries the global dry-run flag");
        assert_eq!(outcome.head, "OffHeadA");
        assert_eq!(outcome.model_version, "clf_v1");
        let info = &outcome.info;
        assert_eq!(info["off_switch_scope"], "head");
        assert_eq!(info["off_switch_reason"], "reason for OffHeadA");
        assert_eq!(info["off_switch_updated_by"], "operator@example.com");
        assert_eq!(
            info["off_switch_dropped_actions"], r#"["addLabelsV2","spamLivenessCheck"]"#,
            "every action of the composite is dropped together"
        );
        assert_eq!(info["overturn_hold_mode"], "shadow", "hold-gate keys kept");
        assert_eq!(tickets.load(Ordering::Relaxed), 0, "no rate-cap ticket");
        assert_eq!(hits.get(), hits_before + 1);
        assert_eq!(dedup_retention_for(&outcome.status), DedupRetention::Skip);
    }

    #[tokio::test]
    async fn pair_switch_blocks_only_its_classifier() {
        let (sw, _) = loaded(&[record("PairHead", Some("clf_v7"), true, None)]).await;
        let (dc, tickets) = config(json!({})).await;

        let r = run(&sw, &dc, &score("PairHead", "clf_v9"), &composite(), false).await;
        assert!(r.outcome.is_none(), "other classifier of the head acts");
        assert_eq!(tickets.load(Ordering::Relaxed), 1);

        let r = run(&sw, &dc, &score("PairHead", "clf_v7"), &composite(), false).await;
        let outcome = r.outcome.unwrap();
        assert_eq!(outcome.status, off_switch::STATUS_SWITCHED_OFF);
        assert!(!outcome.dry_run);
        assert_eq!(outcome.info["off_switch_scope"], "pair");
        assert_eq!(
            tickets.load(Ordering::Relaxed),
            1,
            "switched-off pair took no ticket"
        );
    }

    #[tokio::test]
    async fn other_heads_are_unaffected() {
        let (sw, _) = loaded(&[
            record("OffHeadB", None, true, None),
            record("OffHeadB2", Some("clf_v1"), true, None),
        ])
        .await;
        let (dc, tickets) = config(json!({})).await;

        let r = run(&sw, &dc, &score("OnHead", "clf_v1"), &composite(), false).await;
        assert!(r.outcome.is_none());
        assert_eq!(tickets.load(Ordering::Relaxed), 1);
        assert_eq!(r.hold_gate_info["overturn_hold_mode"], "shadow");

        let mut headless = score("x", "clf_v1");
        headless.summary = None;
        let r = run(&sw, &dc, &headless, &composite(), false).await;
        assert!(r.outcome.is_none());
        assert_eq!(tickets.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn turned_back_on_head_acts_again() {
        let (sw, _) = loaded(&[record("BackOn", None, true, None)]).await;
        let (dc, tickets) = config(json!({})).await;
        assert!(
            run(&sw, &dc, &score("BackOn", "c"), &composite(), false)
                .await
                .outcome
                .is_some()
        );
        sw.set(
            off_switch::OffSwitchPutRequest {
                head: "BackOn".into(),
                classifier: None,
                off: false,
                reason: String::new(),
                auto_on_at_ms: None,
                updated_by: None,
            },
            "pv2",
            now_millis(),
        )
        .await
        .unwrap();
        assert!(
            run(&sw, &dc, &score("BackOn", "c"), &composite(), false)
                .await
                .outcome
                .is_none()
        );
        assert_eq!(tickets.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn growthbook_kill_switch_turns_the_check_off() {
        let (sw, _) = loaded(&[record("OffHeadC", None, true, None)]).await;
        let (dc, tickets) = config(json!({"enforcement_off_switch": {"enabled": false}})).await;
        let r = run(&sw, &dc, &score("OffHeadC", "c"), &composite(), false).await;
        assert!(r.outcome.is_none());
        assert_eq!(tickets.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn store_outage_keeps_blocking_from_the_last_good_list() {
        let (sw, store) = loaded(&[record("OffHeadD", None, true, None)]).await;
        store
            .fail_scan
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(sw.refresh().await.is_err());
        let (dc, tickets) = config(json!({})).await;
        let r = run(&sw, &dc, &score("OffHeadD", "c"), &composite(), false).await;
        assert_eq!(r.outcome.unwrap().status, off_switch::STATUS_SWITCHED_OFF);
        assert_eq!(tickets.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn only_acting_decisions_count_toward_a_head_guard() {
        use head_guard::test_support::{bucket_count, guard, with_counters};
        let (hg, _, buckets) = with_counters(&[guard("GuardHead", None, Some(2), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        let (dc, tickets) = config(json!({})).await;
        let s = score("GuardHead", "clf_v1");

        for _ in 0..5 {
            let r = run_guarded(&sw, &hg, &dc, &s, &composite(), true).await;
            assert!(r.outcome.is_none());
        }
        assert_eq!(
            bucket_count(&buckets, "h/GuardHead"),
            0,
            "dry run never counts"
        );

        for _ in 0..2 {
            let r = run_guarded(&sw, &hg, &dc, &s, &composite(), false).await;
            assert!(r.outcome.is_none(), "acts up to the ceiling");
        }
        assert_eq!(bucket_count(&buckets, "h/GuardHead"), 2);
        assert_eq!(tickets.load(Ordering::Relaxed), 2);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 1);

        let r = run_guarded(&sw, &hg, &dc, &s, &composite(), false).await;
        let outcome = r.outcome.expect("switched off by the ceiling");
        assert_eq!(outcome.status, off_switch::STATUS_SWITCHED_OFF);
        assert_eq!(
            outcome.info["off_switch_updated_by"],
            head_guard::UPDATED_BY
        );
        assert_eq!(
            outcome.info["off_switch_reason"],
            "action ceiling: 2 in 1h (guard v1)"
        );
        assert_eq!(tickets.load(Ordering::Relaxed), 2, "no rate-cap ticket");
        assert_eq!(bucket_count(&buckets, "h/GuardHead"), 2);
    }

                #[tokio::test]
    async fn the_cap_runs_between_the_guard_check_and_the_count() {
        use head_guard::test_support::{bucket_count, guard, with_counters};
        let (hg, _, buckets) = with_counters(&[guard("CapCountHead", None, Some(3), None)]).await;
        let (sw, _) = loaded(&[]).await;
        let s = score("CapCountHead", "clf_v1");

        let (user_cap, _) = sliding_window::counting_limiter_for_test();
        user_cap.seed_for_test();
        let full = DynamicConfig::new(None, None, false, 1_000, 1_000, Some(user_cap), None)
            .await
            .unwrap();
        full.set_snapshot_for_test(growthbook::ConfigSnapshot {
            config: Some(json!({"max_user_enforcements_per_day": 0})),
            ..Default::default()
        });
        let r = run_guarded(&sw, &hg, &full, &s, &composite(), false).await;
        assert_eq!(r.outcome.unwrap().status, "max_enforcement_reached");
        assert_eq!(bucket_count(&buckets, "h/CapCountHead"), 0);

        let (dc, tickets) = config(json!({})).await;
        assert!(
            run_guarded(&sw, &hg, &dc, &s, &composite(), false)
                .await
                .outcome
                .is_none()
        );
        buckets.lock().unwrap()["h/CapCountHead"]
            .count
            .store(5, Ordering::SeqCst);
        let r = run_guarded(&sw, &hg, &dc, &s, &composite(), false).await;
        let outcome = r.outcome.expect("past the ceiling");
        assert_eq!(outcome.status, off_switch::STATUS_SWITCHED_OFF);
        assert_eq!(outcome.info["head_guard_count"], "6");
        assert_eq!(
            tickets.load(Ordering::Relaxed),
            2,
            "the skipped decision spent its ticket"
        );
    }

            #[tokio::test]
    async fn a_reached_ceiling_skips_before_the_cap() {
        use head_guard::test_support::{bucket_count, guard, with_counters};
        let (hg, _, buckets) = with_counters(&[guard("CapHead", None, Some(1), None)]).await;
        let (sw, sw_store) = loaded(&[]).await;
        sw_store.fail_put.store(true, Ordering::SeqCst);
        let (dc, tickets) = config(json!({})).await;
        let s = score("CapHead", "clf_v1");

        assert!(
            run_guarded(&sw, &hg, &dc, &s, &composite(), false)
                .await
                .outcome
                .is_none()
        );
        let r = run_guarded(&sw, &hg, &dc, &s, &composite(), false).await;
        let outcome = r.outcome.expect("at the ceiling");
        assert_eq!(outcome.status, off_switch::STATUS_SWITCHED_OFF);
        assert_eq!(outcome.info["head_guard_scope"], "head");
        assert_eq!(outcome.info["head_guard_window"], "1h");
        assert_eq!(outcome.info["head_guard_limit"], "1");
        assert_eq!(outcome.info["head_guard_count"], "1");
        assert_eq!(outcome.info["head_guard_version"], "1");
        assert_eq!(tickets.load(Ordering::Relaxed), 1);
        assert_eq!(sw_store.puts.load(Ordering::SeqCst), 0);

        let (killed, _) = config(json!({off_switch::CONFIG_KEY: {"enabled": false}})).await;
        assert!(
            run_guarded(&sw, &hg, &killed, &s, &composite(), false)
                .await
                .outcome
                .is_none()
        );
        assert_eq!(bucket_count(&buckets, "h/CapHead"), 1, "not counted");
    }
}

#[cfg(test)]
mod hold_gate_trigger_tests {
    use super::*;
    use crate::decision::ActionSpec;
    use crate::overturn_hold::{GateKind, GateMode, GatedAction, HoldGateConfig};

    fn suspend() -> ActionSpec {
        ActionSpec::SuspendUser {
            perm: false,
            policy: "PlatformManipulation".into(),
        }
    }

    fn label() -> ActionSpec {
        ActionSpec::AddLabelsV2 {
            labels: vec!["SpamHighRecall".into()],
            ttl_msec: None,
        }
    }

    fn perm_suspend() -> ActionSpec {
        ActionSpec::SuspendUser {
            perm: true,
            policy: "Cse".into(),
        }
    }

            fn suspend_only_cfg() -> HoldGateConfig {
        HoldGateConfig {
            mode: GateMode::Enforce,
            ..HoldGateConfig::default()
        }
    }

        fn label_cfg(labels: &[&str]) -> HoldGateConfig {
        HoldGateConfig {
            mode: GateMode::Enforce,
            kinds: vec![GateKind::Suspend, GateKind::Label],
            labels: labels.iter().map(|s| (*s).to_owned()).collect(),
            ..HoldGateConfig::default()
        }
    }

    fn labels_spec(labels: &[&str]) -> ActionSpec {
        ActionSpec::AddLabelsV2 {
            labels: labels.iter().map(|s| (*s).to_owned()).collect(),
            ttl_msec: Some(86_400_000),
        }
    }

    fn gated_label(name: &str) -> GatedAction {
        GatedAction::Label { name: name.into() }
    }

                        #[test]
    fn suspend_detection_covers_plain_composite_and_never_labels() {
        let c = suspend_only_cfg();
        assert_eq!(
            gated_actions(&[suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[perm_suspend()], &c),
            vec![GatedAction::PERMANENT_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[label(), suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(
                &[
                    ActionSpec::AddPostLabelsV2 {
                        labels: vec!["Cse".into()],
                        ttl_msec: None,
                    },
                    perm_suspend(),
                ],
                &c
            ),
            vec![GatedAction::PERMANENT_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[suspend(), perm_suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[perm_suspend(), suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[perm_suspend(), label(), perm_suspend()], &c),
            vec![GatedAction::PERMANENT_SUSPEND],
            "all-perm set stays permanent"
        );
        assert!(gated_actions(&[label()], &c).is_empty());
        assert!(gated_actions(&[label(), ActionSpec::Captcha], &c).is_empty());
        assert!(gated_actions(&[ActionSpec::Arkose, ActionSpec::SpamLivenessCheck], &c).is_empty());
        assert!(gated_actions(&[], &c).is_empty());
        assert_eq!(HoldGateConfig::default().kinds, vec![GateKind::Suspend]);
        assert!(HoldGateConfig::default().labels.is_empty());
    }

                        #[test]
    fn label_detection_follows_kinds_and_labels() {
        let c = label_cfg(&["SpamHighRecall"]);
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamHighRecall"])], &c),
            vec![gated_label("SpamHighRecall")]
        );
        assert_eq!(
            gated_actions(
                &[
                    labels_spec(&["SpamMediumRecall", "SpamHighRecall"]),
                    ActionSpec::Captcha,
                ],
                &c
            ),
            vec![gated_label("SpamHighRecall")],
            "labels not in the config are ignored"
        );
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamHighRecall"]), perm_suspend()], &c),
            vec![
                GatedAction::PERMANENT_SUSPEND,
                gated_label("SpamHighRecall")
            ]
        );
        assert_eq!(
            gated_actions(
                &[
                    labels_spec(&["SpamHighRecall"]),
                    labels_spec(&["SpamHighRecall", "Other"]),
                ],
                &c
            ),
            vec![gated_label("SpamHighRecall")]
        );
        let c2 = label_cfg(&["SpamHighRecall", "SpamMediumRecall"]);
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamMediumRecall", "SpamHighRecall"])], &c2),
            vec![
                gated_label("SpamMediumRecall"),
                gated_label("SpamHighRecall")
            ]
        );
        assert!(
            gated_actions(
                &[ActionSpec::AddPostLabelsV2 {
                    labels: vec!["SpamHighRecall".into()],
                    ttl_msec: None,
                }],
                &c
            )
            .is_empty()
        );
        let inert = label_cfg(&[]);
        assert!(gated_actions(&[labels_spec(&["SpamHighRecall"])], &inert).is_empty());
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamHighRecall"]), suspend()], &inert),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        let no_label_kind = HoldGateConfig {
            labels: vec!["SpamHighRecall".into()],
            ..suspend_only_cfg()
        };
        assert!(gated_actions(&[labels_spec(&["SpamHighRecall"])], &no_label_kind).is_empty());
        let label_only_kind = HoldGateConfig {
            kinds: vec![GateKind::Label],
            ..label_cfg(&["SpamHighRecall"])
        };
        assert_eq!(
            gated_actions(
                &[labels_spec(&["SpamHighRecall"]), suspend()],
                &label_only_kind
            ),
            vec![gated_label("SpamHighRecall")]
        );
    }

                #[test]
    fn strip_held_labels_removes_only_the_held_names() {
        let held = vec!["SpamHighRecall".to_owned()];
        let mut specs = vec![labels_spec(&["SpamHighRecall"])];
        strip_held_labels(&mut specs, &held);
        assert!(specs.is_empty());
        let mut specs = vec![labels_spec(&["SpamHighRecall", "SpamMediumRecall"])];
        strip_held_labels(&mut specs, &held);
        assert_eq!(specs, vec![labels_spec(&["SpamMediumRecall"])]);
        let mut specs = vec![labels_spec(&["SpamHighRecall"]), suspend()];
        strip_held_labels(&mut specs, &held);
        assert_eq!(specs, vec![suspend()]);
        let post = ActionSpec::AddPostLabelsV2 {
            labels: vec!["SpamHighRecall".into()],
            ttl_msec: None,
        };
        let mut specs = vec![
            ActionSpec::Captcha,
            labels_spec(&["SpamHighRecall"]),
            post.clone(),
        ];
        strip_held_labels(&mut specs, &held);
        assert_eq!(
            specs,
            vec![ActionSpec::Captcha, post],
            "post labels are never touched"
        );
        let before = vec![labels_spec(&["SpamHighRecall"]), suspend()];
        let mut specs = before.clone();
        strip_held_labels(&mut specs, &[]);
        assert_eq!(specs, before);
        let mut specs = vec![labels_spec(&["SpamMediumRecall"])];
        strip_held_labels(&mut specs, &held);
        assert_eq!(specs, vec![labels_spec(&["SpamMediumRecall"])]);
    }

                #[test]
    fn hold_skip_propagates_global_dry_run_and_audit_keys() {
        let score = abuse_proto::ScoreResult {
            user_id: 4242,
            model_version: "m@1".into(),
            ..Default::default()
        };
        let facts = Facts {
            entity_type: EntityType::User,
            entity_id: 4242,
            user_id: 4242,
            topic: "abuse.embeddings.user_decisions".into(),
            score: ScoreFacts::from_score(&score),
            entity: EntityFacts::User(UserFacts::default()),
            uas: None,
        };
        let mut gate_info = BTreeMap::new();
        gate_info.insert("overturn_hold_mode".to_owned(), "enforce".to_owned());
        gate_info.insert("overturn_hold_id".to_owned(), "77".to_owned());
        gate_info.insert("overturn_hold_head".to_owned(), "FollowBot".to_owned());
        gate_info.insert(
            "overturn_hold_expires_at".to_owned(),
            "2026-12-01T00:00:00+00:00".to_owned(),
        );
        gate_info.insert("overturn_hold_probe_ms".to_owned(), "2".to_owned());

        for global_dry_run in [true, false] {
            let out = hold_gate_skip_outcome(
                crate::overturn_hold::STATUS_HOLD_OVERTURNED,
                gate_info.clone(),
                global_dry_run,
                &facts,
                &score,
            );
            assert_eq!(out.status, "hold_overturned");
            assert_eq!(
                out.dry_run, global_dry_run,
                "hold skip must carry the service's global dry-run (G17)"
            );
            assert_eq!(out.source_topic, "abuse.embeddings.user_decisions");
            assert_eq!(out.entity_id, 4242);
            assert_eq!(out.info["overturn_hold_id"], "77");
            assert_eq!(out.info["overturn_hold_head"], "FollowBot");
            assert_eq!(out.info["overturn_hold_mode"], "enforce");
            assert_eq!(out.info["overturn_hold_probe_ms"], "2");
            assert!(out.info.contains_key("overturn_hold_expires_at"));
            assert!(
                out.info.contains_key("cred_is_high"),
                "{:?}",
                out.info.keys()
            );
            assert!(!outcome_holds_full_dedup(&out.status));
            assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
        }
        let mut label_info = BTreeMap::new();
        label_info.insert("overturn_hold_mode".to_owned(), "enforce".to_owned());
        label_info.insert(
            "overturn_hold_labels_stripped".to_owned(),
            "SpamHighRecall,SpamMediumRecall".to_owned(),
        );
        label_info.insert("overturn_hold_label_hold_id".to_owned(), "91,92".to_owned());
        let mut gate = crate::overturn_hold::GateOutcome {
            skip_status: None,
            info: label_info,
            strip_labels: vec!["SpamHighRecall".into(), "SpamMediumRecall".into()],
            strip_hold_ids: vec![91, 92],
        };
        gate.mark_label_collapse();
        let out = hold_gate_skip_outcome(
            crate::overturn_hold::STATUS_HOLD_OVERTURNED,
            gate.info,
            false,
            &facts,
            &score,
        );
        assert_eq!(out.status, "hold_overturned");
        assert!(!out.dry_run);
        assert_eq!(out.info["overturn_hold_kind"], "label");
        assert_eq!(
            out.info["overturn_hold_id"], "91",
            "first stripped label's hold"
        );
        assert_eq!(out.info["overturn_hold_label"], "SpamHighRecall");
        assert_eq!(
            out.info["overturn_hold_labels_stripped"],
            "SpamHighRecall,SpamMediumRecall"
        );
        assert_eq!(out.info["overturn_hold_label_hold_id"], "91,92");
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
        let mut suspend_info = gate_info.clone();
        suspend_info.insert("overturn_hold_kind".to_owned(), "suspend".to_owned());
        let out = hold_gate_skip_outcome(
            crate::overturn_hold::STATUS_HOLD_OVERTURNED,
            suspend_info,
            false,
            &facts,
            &score,
        );
        assert_eq!(out.info["overturn_hold_kind"], "suspend");
        assert_eq!(out.info["overturn_hold_id"], "77");
        assert!(!out.info.contains_key("overturn_hold_label"));

        let out = hold_gate_skip_outcome(
            crate::overturn_hold::STATUS_HOLD_LOOKUP_FAILED,
            BTreeMap::new(),
            true,
            &facts,
            &score,
        );
        assert_eq!(out.status, "hold_lookup_failed");
        assert!(out.dry_run);
        assert!(!outcome_holds_full_dedup(&out.status));
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Release);
    }
}

#[cfg(test)]
mod decision_outcome_json_tests {
    use super::*;
    use xai_abuse_proto::enforcement::{
        EntityType as ProtoEntityType, FiredHead, ScoreResult, SummaryCounters,
    };

                            #[test]
    fn decision_outcome_json_mirror_carries_funnel_fields() {
        let score = ScoreResult {
            user_id: 100,
            model_version: "my_model@1".into(),
            entity_type: ProtoEntityType::Post as i32,
            entity_id: 555,
            summary: Some(SummaryCounters {
                labels: vec!["my_model_threshold_reached".into()],
                fired_heads: vec![FiredHead {
                    name: "IsSpamPost".into(),
                    score: 0.99,
                    threshold: 0.9,
                }],
                ..Default::default()
            }),
            score_id: "run-abc-7".into(),
            ..Default::default()
        };
        let mut info = BTreeMap::new();
        info.insert(
            "action_kinds".to_owned(),
            r#"["addPostLabelsV2"]"#.to_owned(),
        );
        let outcome = decision_outcome(&score, "some.topic", "success".into(), true, info);
        assert_eq!(outcome.score_id, "run-abc-7");
        let v = serde_json::to_value(&outcome).expect("DecisionOutcome serializes to JSON");

        assert!(v["decided_at_ms"].as_i64().unwrap() > 0);
        assert_eq!(v["source_topic"], "some.topic");
        assert_eq!(v["entity_type"], "post");
        assert_eq!(v["entity_id"], 555);
        assert_eq!(v["model_version"], "my_model@1");
        assert_eq!(v["status"], "success");
        assert_eq!(v["dry_run"], true);
        assert_eq!(v["head"], "IsSpamPost");
        assert_eq!(v["fired_heads"][0]["name"], "IsSpamPost");
        assert_eq!(v["labels"][0], "my_model_threshold_reached");
        assert_eq!(v["info"]["action_kinds"], r#"["addPostLabelsV2"]"#);
        assert_eq!(v["score_id"], "run-abc-7");

        let legacy = ScoreResult {
            score_id: String::new(),
            ..score
        };
        let outcome = decision_outcome(
            &legacy,
            "some.topic",
            "success".into(),
            true,
            BTreeMap::new(),
        );
        assert_eq!(outcome.score_id, "");
        let v = serde_json::to_value(&outcome).expect("DecisionOutcome serializes to JSON");
        assert_eq!(v["score_id"], "");
    }

    #[test]
    fn decision_outcome_carries_the_reply_target_only_when_sent() {
        let reply = ScoreResult {
            user_id: 100,
            entity_type: ProtoEntityType::Post as i32,
            entity_id: 555,
            in_reply_to_user_id: 777,
            in_reply_to_post_id: 888,
            ..Default::default()
        };
        let outcome = decision_outcome(
            &reply,
            "some.topic",
            "success".into(),
            true,
            BTreeMap::new(),
        );
        assert_eq!(
            (outcome.in_reply_to_user_id, outcome.in_reply_to_post_id),
            (777, 888)
        );
        let v = serde_json::to_value(&outcome).expect("DecisionOutcome serializes to JSON");
        assert_eq!(
            (
                v["in_reply_to_user_id"].as_i64(),
                v["in_reply_to_post_id"].as_i64()
            ),
            (Some(777), Some(888))
        );
        let back = abuse_proto::DecisionOutcome::decode(outcome.encode_to_vec().as_slice())
            .expect("decodes");
        assert_eq!(
            (back.in_reply_to_user_id, back.in_reply_to_post_id),
            (777, 888)
        );

        let other = ScoreResult {
            in_reply_to_user_id: 0,
            in_reply_to_post_id: 0,
            ..reply
        };
        let outcome = decision_outcome(
            &other,
            "some.topic",
            "success".into(),
            true,
            BTreeMap::new(),
        );
        let v = serde_json::to_value(&outcome).expect("DecisionOutcome serializes to JSON");
        assert!(
            v.get("in_reply_to_user_id").is_none() && v.get("in_reply_to_post_id").is_none(),
            "{v}"
        );
        let sv = serde_json::to_value(&other).expect("ScoreResult serializes to JSON");
        assert!(
            sv.get("in_reply_to_user_id").is_none() && sv.get("in_reply_to_post_id").is_none(),
            "{sv}"
        );
    }
}

#[cfg(test)]
mod health_tests {
    use super::health_decision;
    use std::time::Duration;

    const STALE: Duration = Duration::from_secs(120);
    const STALE_DELETE: Duration = Duration::from_secs(240);
    const ERR_DELETE: Duration = Duration::from_secs(60);
    const Z: Duration = Duration::ZERO;
    const S5: Duration = Duration::from_secs(5);

    #[test]
    fn fresh_consumer_is_ready() {
        let (ready, del) =
            health_decision(true, S5, Z, STALE, STALE_DELETE, ERR_DELETE, false, false);
        assert!(ready);
        assert!(!del);
    }

    #[test]
    fn stale_under_threshold_is_notready_not_deleted() {
        let s = Duration::from_secs(130);
        let (ready, del) =
            health_decision(true, s, Z, STALE, STALE_DELETE, ERR_DELETE, false, false);
        assert!(!ready);
        assert!(!del);
    }

    #[test]
    fn fully_dead_no_progress_self_deletes_at_stale_window() {
        let s = Duration::from_secs(241);
        let (ready, del) =
            health_decision(false, s, Z, STALE, STALE_DELETE, ERR_DELETE, false, false);
        assert!(!ready);
        assert!(del);
    }

    #[test]
    fn flapping_node_self_deletes_fast_on_error_window() {
        let (ready, del) = health_decision(
            true,
            S5,
            Duration::from_secs(61),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
            false,
        );
        assert!(!ready);
        assert!(del);
        let (ready, del) = health_decision(
            true,
            S5,
            Duration::from_secs(45),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
            false,
        );
        assert!(!ready);
        assert!(!del);
    }

    #[test]
    fn brief_error_spike_does_not_evict() {
        let (ready, del) = health_decision(
            true,
            S5,
            Duration::from_secs(10),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
            false,
        );
        assert!(!ready);
        assert!(!del);
    }

    #[test]
    fn backfill_pod_on_an_empty_topic_is_ready_and_stays() {
        for s in [Z, Duration::from_secs(241), Duration::from_secs(86_400)] {
            let (ready, del) =
                health_decision(false, s, Z, STALE, STALE_DELETE, ERR_DELETE, false, true);
            assert!(ready, "not ready after {s:?} of silence");
            assert!(!del, "self-deleted after {s:?} of silence");
        }
        let s = Duration::from_secs(3_600);
        let (ready, del) =
            health_decision(true, s, Z, STALE, STALE_DELETE, ERR_DELETE, false, true);
        assert!(ready);
        assert!(!del);
    }

    #[test]
    fn backfill_pod_still_leaves_on_an_error_flood() {
        let s = Duration::from_secs(86_400);
        let (ready, del) = health_decision(
            false,
            s,
            Duration::from_secs(61),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
            true,
        );
        assert!(!ready);
        assert!(del);
        let (ready, del) = health_decision(
            false,
            s,
            Duration::from_secs(10),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
            true,
        );
        assert!(!ready);
        assert!(!del);
    }
}

#[cfg(test)]
mod already_suspended_gate_tests {
    use std::collections::HashMap;

    use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, PCFLabel, Safety};
    use xai_core_entities::gizmoduck_client::{
        GizmoduckClient, MockGizmoduckClient, QueryFields, UserFields, ViewerData,
    };

    use super::*;
    use crate::decision::ActionSpec;
    use crate::facts::GizmoduckFacts;
    use crate::overturn_hold::{FakeHoldStore, HOLD_KIND_SUSPEND, Hold, HoldGate};

    const UID: i64 = 1_700_000_042;

    fn suspend(perm: bool) -> ActionSpec {
        ActionSpec::SuspendUser {
            perm,
            policy: "PlatformManipulation".into(),
        }
    }

    fn label() -> ActionSpec {
        ActionSpec::AddLabelsV2 {
            labels: vec!["SpamHighRecall".into()],
            ttl_msec: Some(86_400_000),
        }
    }

    fn gizmoduck(suspended: bool) -> GizmoduckFacts {
        GizmoduckFacts {
            present: true,
            suspended,
            ..Default::default()
        }
    }

    fn strip(specs: &[ActionSpec], user: &GizmoduckFacts) -> (Vec<ActionSpec>, Vec<ActionSpec>) {
        let mut kept = specs.to_vec();
        let stripped = strip_redundant_suspends(&mut kept, user);
        (kept, stripped)
    }

    #[test]
    fn temporary_suspend_of_a_suspended_account_is_removed() {
        let suspended = gizmoduck(true);
        assert_eq!(
            strip(&[suspend(false)], &suspended),
            (vec![], vec![suspend(false)])
        );
        assert_eq!(
            strip(&[suspend(false), suspend(false)], &suspended),
            (vec![], vec![suspend(false), suspend(false)])
        );
    }

    #[test]
    fn other_actions_of_the_decision_are_kept_in_order() {
        let suspended = gizmoduck(true);
        assert_eq!(
            strip(
                &[label(), suspend(false), ActionSpec::SpamLivenessCheck],
                &suspended
            ),
            (
                vec![label(), ActionSpec::SpamLivenessCheck],
                vec![suspend(false)]
            )
        );
    }

    #[test]
    fn active_account_keeps_every_action() {
        let active = gizmoduck(false);
        for specs in [
            vec![suspend(false)],
            vec![suspend(true)],
            vec![label(), suspend(false)],
        ] {
            assert_eq!(strip(&specs, &active), (specs.clone(), vec![]));
        }
    }

    #[test]
    fn a_permanent_suspend_keeps_the_decision_whole() {
        let suspended = gizmoduck(true);
        for specs in [
            vec![suspend(true)],
            vec![suspend(false), suspend(true)],
            vec![suspend(true), label(), suspend(false)],
        ] {
            assert_eq!(strip(&specs, &suspended), (specs.clone(), vec![]));
        }
    }

    #[test]
    fn decisions_without_a_suspend_are_untouched() {
        let suspended = gizmoduck(true);
        for specs in [
            vec![label()],
            vec![ActionSpec::Arkose, ActionSpec::Captcha],
            vec![],
        ] {
            assert_eq!(strip(&specs, &suspended), (specs.clone(), vec![]));
        }
    }


            const RULES_YAML: &str = r#"
for_entity: user
rules:
  - id: act_head_suspend
    when: 'score.model_version == "head_v1"'
    then: { kind: act_suspend_user, perm: false, policy: "PlatformManipulation" }
  - id: act_head_suspend_and_label
    when: 'score.model_version == "head_v2"'
    then:
      kind: act_all
      actions:
        - { kind: act_suspend_user, perm: false, policy: "PlatformManipulation" }
        - { kind: act_add_labels_v2, labels: ["SpamHighRecall"], ttl_msec: 86400000 }
  - id: already_suspended
    when: user.suspended
    then: { kind: skip, reason: already_suspended }
  - id: terminal
    when: "true"
    then: { kind: skip, reason: no_match }
"#;

        fn suspend_hold() -> Hold {
        Hold {
            user_id: UID,
            hold_id: 7,
            head: Some("FollowBot".into()),
            expires_at: chrono::Utc::now() + chrono::Duration::days(30),
            case_group_id: Some(63),
            reason: "appeal_overturned".into(),
            action_kind: HOLD_KIND_SUSPEND.into(),
            label: None,
        }
    }

    fn gizmoduck_user(suspended: bool) -> Arc<MockGizmoduckClient> {
        let user = GizmoduckUser {
            user_id: UID as u64,
            safety: Safety {
                suspended,
                ..Default::default()
            },
            ..Default::default()
        };
        Arc::new(MockGizmoduckClient {
            users: HashMap::from([(
                UID,
                Some(GizmoduckUserResult {
                    user: Some(user),
                    response_state: None,
                }),
            )]),
            ..Default::default()
        })
    }

        struct FailingGizmoduck;

    #[async_trait::async_trait]
    impl GizmoduckClient for FailingGizmoduck {
        async fn get_users(
            &self,
            user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>> {
            user_ids
                .into_iter()
                .map(|id| (id, Err(anyhow::anyhow!("gizmoduck unavailable"))))
                .collect()
        }
        async fn get_users_with_perspective(
            &self,
            _viewer_id: i64,
            user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>> {
            self.get_users(user_ids).await
        }
        async fn get_viewer_roles(&self, _user_id: u64) -> anyhow::Result<Vec<String>> {
            unimplemented!()
        }
        async fn get_viewer_data(&self, _user_id: u64) -> anyhow::Result<ViewerData> {
            unimplemented!()
        }
        async fn get_viewer_data_with_fields(
            &self,
            _user_id: u64,
            _query_fields: &[QueryFields],
        ) -> anyhow::Result<ViewerData> {
            unimplemented!()
        }
        async fn get_pcf_labels(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<PCFLabel>> {
            unimplemented!()
        }
        async fn get_profile_description_languages(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<String>>> {
            unimplemented!()
        }
        async fn get_user_fields(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<UserFields>> {
            unimplemented!()
        }
        async fn get_by_screen_name(
            &self,
            _screen_name: &str,
        ) -> anyhow::Result<Option<GizmoduckUserResult>> {
            unimplemented!()
        }
    }

    struct Harness {
        ctx: EnforcementCtx,
        holds: Arc<FakeHoldStore>,
        gizmoduck_calls: Option<Arc<MockGizmoduckClient>>,
        _strato: wiremock::MockServer,
    }

                    async fn harness(gd: Arc<dyn GizmoduckClient + Send + Sync>, dry_run: bool) -> Harness {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({"v": null})),
            )
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
        let dynamic_config = DynamicConfig::new(None, None, dry_run, 1_000, 1_000, None, None)
            .await
            .unwrap();
        dynamic_config.set_snapshot_for_test(crate::growthbook::ConfigSnapshot {
            config: Some(serde_json::json!({
                "dry_run": dry_run,
                "overturn_hold_gate": {"mode": "enforce"},
            })),
            rules_user: Some(Arc::from(RULES_YAML)),
            rules_post: None,
        });
        let holds = FakeHoldStore::new(vec![suspend_hold()]);
        let ctx = EnforcementCtx {
            ais_client: ais_client::AisClient::unconnected(),
            gd: gizmoduck::GizmoduckCoreClient::from_client(gd),
            cred_clients: strato::CredClients {
                high_page_rank: column("hpr"),
                grey_badge: column("grey"),
            },
            uas_strato: column("uas"),
            dynamic_config,
            rules_cache: Arc::new(rules::RulesCache::new()),
            allowlist: None,
            kafka_producer_decisions: None,
            kafka_producer_decisions_json: None,
            hold_gate: Arc::new(HoldGate::new(Some(holds.clone()))),
            off_switch: Arc::new(off_switch::OffSwitch::new(
                Arc::new(off_switch::test_support::MemStore::default()),
                Arc::new(off_switch::test_support::MemStore::default()),
            )),
            head_guard: Arc::new(head_guard::HeadGuard::new(
                Arc::new(action_control::test_support::MemControlStore::default()),
                None,
            )),
            action_control: Arc::new(action_control::ActionControl::new(
                Arc::new(action_control::test_support::MemControlStore::default()),
                "test".into(),
                String::new(),
            )),
            backfill: Arc::new(backfill::Backfill::new(
                Arc::new(action_control::test_support::MemControlStore::default()),
                None,
                String::new(),
            )),
            post_labels: None,
            outcome_log: None,
        };
        Harness {
            ctx,
            holds,
            gizmoduck_calls: None,
            _strato: server,
        }
    }

    async fn harness_for(suspended: bool, dry_run: bool) -> Harness {
        let gd = gizmoduck_user(suspended);
        let mut h = harness(gd.clone(), dry_run).await;
        h.gizmoduck_calls = Some(gd);
        h
    }

    fn score(model_version: &str) -> abuse_proto::ScoreResult {
        assert!(!test_user::is_test_user_id(UID));
        abuse_proto::ScoreResult {
            user_id: UID,
            model_version: model_version.into(),
            ..Default::default()
        }
    }

            #[tokio::test]
    async fn control_active_account_reaches_the_hold_gate() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(false, false).await;
        let out = run_enforcement_inner(
            &h.ctx,
            &score("head_v1"),
            "t",
            &mut Default::default(),
            dedup_cache::CLASS_NONE,
            false,
            &mut BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.status, overturn_hold::STATUS_HOLD_OVERTURNED);
        assert_eq!(h.holds.calls(), 1);
    }

            #[tokio::test]
    async fn rules_stage_records_the_seen_pair() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(true, false).await;
        let mut s = score("head_v1");
        s.summary = Some(abuse_proto::SummaryCounters {
            fired_heads: vec![abuse_proto::FiredHead {
                name: "SeenPathHead".into(),
                score: 0.99,
                threshold: 0.9,
            }],
            ..Default::default()
        });
        let out = run_enforcement_inner(
            &h.ctx,
            &s,
            "t",
            &mut Default::default(),
            dedup_cache::CLASS_NONE,
            false,
            &mut BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.status, ALREADY_SUSPENDED);
        assert_eq!(
            h.ctx.off_switch.record_seen("SeenPathHead", "head_v1", 0),
            off_switch::SeenOutcome::Recent
        );
    }

                #[tokio::test]
    async fn suspended_account_is_skipped_before_the_hold_gate() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(true, false).await;
        let out = run_enforcement_inner(
            &h.ctx,
            &score("head_v1"),
            "t",
            &mut Default::default(),
            dedup_cache::CLASS_NONE,
            false,
            &mut BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.status, ALREADY_SUSPENDED);
        assert!(!out.dry_run);
        assert_eq!(out.info[ALREADY_SUSPENDED_STRIPPED], r#"["suspendUser"]"#);
        assert_eq!(out.info["gizmoduck_suspended"], "true");
        assert!(!out.info.contains_key("acted_class"));
        assert_eq!(h.holds.calls(), 0);
        assert_eq!(h.gizmoduck_calls.as_ref().unwrap().call_count(), 1);
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
    }

    #[tokio::test]
    async fn logging_only_mode_records_the_same_skip() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(true, true).await;
        let out = run_enforcement_inner(
            &h.ctx,
            &score("head_v1"),
            "t",
            &mut Default::default(),
            dedup_cache::CLASS_NONE,
            false,
            &mut BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.status, ALREADY_SUSPENDED);
        assert!(out.dry_run);
        assert_eq!(h.holds.calls(), 0);
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
    }

                    #[tokio::test]
    async fn suspend_plus_label_keeps_the_label() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(true, true).await;
        let out = run_enforcement_inner(
            &h.ctx,
            &score("head_v2"),
            "t",
            &mut Default::default(),
            dedup_cache::CLASS_NONE,
            false,
            &mut BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.status, "dry_run");
        assert_eq!(out.info["action_kinds"], r#"["addLabelsV2"]"#);
        assert_eq!(out.info[ALREADY_SUSPENDED_STRIPPED], r#"["suspendUser"]"#);
        assert!(
            !out.info.contains_key("acted_class"),
            "a dry run holds no dedup strength"
        );
        assert_eq!(h.holds.calls(), 0);
    }

            #[tokio::test]
    async fn gizmoduck_read_error_is_retried() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness(Arc::new(FailingGizmoduck), false).await;
        let err = run_enforcement_inner(
            &h.ctx,
            &score("head_v1"),
            "t",
            &mut Default::default(),
            dedup_cache::CLASS_NONE,
            false,
            &mut BTreeMap::new(),
            None,
        )
        .await
        .unwrap_err();
        assert!(format!("{err:#}").contains("gizmoduck"), "{err:#}");
        assert_eq!(h.holds.calls(), 0);
    }
}
