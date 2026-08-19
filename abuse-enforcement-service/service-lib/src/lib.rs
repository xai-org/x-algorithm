pub mod ais_client;
pub mod allowlist;
pub mod api_doc;
pub mod auth;
pub mod config;
pub mod decision;
pub mod dedup_cache;
pub mod entities;
pub mod facts;
pub mod gizmoduck;
pub mod gizmoduck_labels;
pub mod growthbook;
pub mod growthbook_writer;
pub mod limiter;
pub mod manhattan;
pub mod metrics;
pub mod rules;
pub mod service;
pub mod sliding_window;
pub mod strato;
pub mod test_user;

use xai_abuse_proto::enforcement as abuse_proto;

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;

use tokio::sync::Mutex;

use crate::dedup_cache::{DedupCheck, ManhattanDedupCache};
use anyhow::{Context, Result};
use axum::Router;

use clap::Parser;
use futures::future::join_all;
use prost::Message;
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tracing::{Instrument, error, info, warn};
use xai_kafka::{
    BatchConsumerConfig, BatchResult, CancellationToken, KafkaBatchProcessor, KafkaConsumerBuilder,
    KafkaConsumerConfig, KafkaMessage, KafkaProducer, KafkaProducerConfigBuilder, SslConfig,
    apply_auth_config, resolve_kafka_brokers, run_batch_consumer, self_delete_pod,
};
use xai_service_runner::{ServerBuilder, ServerInfo};
use xai_wily::WilyConfig;

use xai_strato::{Strato, StratoClientConfig};

use crate::config::{Config, KafkaConnConfig};
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
const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");


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
        dedup_cache: ManhattanDedupCache,
                    claim_nonce: Option<u64>,
}

#[derive(Debug)]
struct EnforcementFailed {
    permanent: bool,
    claim_nonce: Option<u64>,
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
struct EnforcementCtx {
    ais_client: ais_client::AisClient,
    gd: gizmoduck::GizmoduckCoreClient,
    cred_clients: strato::CredClients,
    uas_strato: Arc<Strato>,
    dynamic_config: DynamicConfig,
            rules_cache: Arc<rules::RulesCache>,
    allowlist: Option<allowlist::ManhattanAllowlist>,
            kafka_producer_decisions: Option<Arc<KafkaProducer>>,
}

struct ScoreResultProcessor {
    topic: String,
    labels: HashSet<String>,
    negative_labels: HashSet<String>,
    ctx: EnforcementCtx,
    dedup_cache: ManhattanDedupCache,
    retry_queue: Arc<Mutex<BinaryHeap<RetryEntry>>>,
        last_progress: Arc<AtomicI64>,
        max_in_flight: usize,
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
    }
}

#[tracing::instrument(skip_all, fields(dry_run, uas))]
async fn run_enforcement_inner(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
) -> Result<abuse_proto::DecisionOutcome> {
    let user_id = score.user_id;
    let (entity_type, entity_id) = Facts::resolve_target(score);
    let dry_run = ctx.dynamic_config.is_dry_run();
    let span = tracing::Span::current();
    span.record("dry_run", dry_run);

    let gated_user_id = match entity_type {
        EntityType::User => entity_id,
        EntityType::Post => user_id, 
    };
    if test_user::is_test_user_id(gated_user_id) {
        info!(
            user_id = gated_user_id,
            "test user (id bitmask/range); skipping enforcement"
        );
        let entity = match entity_type {
            EntityType::User => EntityFacts::User(UserFacts::default()),
            EntityType::Post => EntityFacts::Post(PostFacts {
                author_id: user_id,
                ..Default::default()
            }),
        };
        let partial = Facts {
            entity_type,
            entity_id,
            user_id: gated_user_id,
            topic: source_topic.to_owned(),
            score: ScoreFacts::from_score(score),
            entity,
            uas: None,
        };
        return Ok(skip_outcome(
            "test_user_skipped".into(),
            dry_run,
            &partial,
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

    let rules_override = ctx.dynamic_config.enforcement_rules_yaml(facts.entity_type);
    let compiled_rules = ctx
        .rules_cache
        .resolve(facts.entity_type, rules_override.as_deref());
    let decision = match crate::rules::decide_with(&compiled_rules, &facts) {
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
    };

    match decision {
        Decision::Skip(reason) => Ok(skip_outcome(reason, dry_run, &facts, score)),
        Decision::Act(specs) => {
            if !ctx.dynamic_config.try_enforce(facts.entity_type).await {
                warn!(
                    entity_type = facts.entity_type.as_str(),
                    "max enforcement has been reached; skipping"
                );
                return Ok(skip_outcome(
                    "max_enforcement_reached".into(),
                    dry_run,
                    &facts,
                    score,
                ));
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

fn outcome_holds_full_dedup(status: &str) -> bool {
    status == "success"
}

async fn write_dedup_outcome(
    dedup_cache: &ManhattanDedupCache,
    entity_type: EntityType,
    entity_id: i64,
    score: &abuse_proto::ScoreResult,
    topic: &str,
    outcome: &abuse_proto::DecisionOutcome,
    claim_nonce: Option<u64>,
) {
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
    if outcome_holds_full_dedup(&outcome.status) {
        dedup_cache
            .update(
                entity_type,
                entity_id,
                &compressed,
                acted_class,
                claim_nonce,
            )
            .await;
    } else {
        dedup_cache
            .update_skip(
                entity_type,
                entity_id,
                &compressed,
                acted_class,
                claim_nonce,
            )
            .await;
    }
}

const DECISIONS_PRODUCER: &str = "decisions";

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
    let key = key();
    let bytes = record.encode_to_vec();
    let producer = producer.clone();
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

fn publish_decision_outcome(
    producer: Option<&Arc<KafkaProducer>>,
    outcome: &abuse_proto::DecisionOutcome,
) {
    spawn_publish(
        producer,
        DECISIONS_PRODUCER,
        || outcome.entity_id.to_string().into_bytes(),
        outcome,
    );
}

async fn run_enforcement(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
) -> Result<abuse_proto::DecisionOutcome> {
    let outcome = run_enforcement_inner(ctx, score, source_topic).await?;
    publish_decision_outcome(ctx.kafka_producer_decisions.as_ref(), &outcome);
    Ok(outcome)
}

fn sasl_ssl_config(conn: &KafkaConnConfig, password: &str) -> SslConfig {
    SslConfig {
        security_protocol: "SASL_SSL".to_string(),
        sasl_mechanism: Some(conn.sasl_mechanism.clone()),
        sasl_username: Some(conn.sasl_username.clone()),
        sasl_password: Some(password.to_owned()),
    }
}

async fn build_kafka_producers(cfg: &Config, dynamic_config: &DynamicConfig) -> KafkaProducers {
    let mut producers = KafkaProducers::default();

    let conn = cfg.kafka_producer();
    let sasl_password = conn.sasl_password.clone();

    if !cfg.kafka_producer_mtls_enabled && sasl_password.is_none() {
        return producers;
    }

    for (name, spec) in dynamic_config.kafka_producers() {
        if !spec.enabled {
            warn!("kafka producer '{name}' configured but disabled (enabled=false)");
            continue;
        }
        let Some(topic) = spec.topic else {
            warn!("kafka producer '{name}' enabled but no topic set; skipping");
            continue;
        };
        let producer_config = if cfg.kafka_producer_mtls_enabled {
            match KafkaProducerConfigBuilder::for_cluster_mtls_auto(
                &cfg.kafka_producer_mtls_cluster,
                topic.clone(),
                Some(&cfg.kafka_producer_mtls_zone),
            ) {
                Ok(builder) => builder.build(),
                Err(e) => {
                    error!("kafka producer '{name}' mTLS config failed; skipping: {e}");
                    continue;
                }
            }
        } else {
            KafkaProducerConfigBuilder::new(conn.dest.clone(), topic.clone())
                .with_wily_config(WilyConfig::default())
                .with_ssl(sasl_ssl_config(
                    &conn,
                    sasl_password
                        .as_deref()
                        .expect("SASL password checked before producer construction"),
                ))
                .build()
        };
        let mut producer = KafkaProducer::new(producer_config);
        match producer.start().await {
            Ok(()) => {
                info!("kafka producer '{name}' started -> topic {topic}");
                producers.by_name.insert(name, Arc::new(producer));
            }
            Err(e) => {
                error!("kafka producer '{name}' failed to start; continuing without it: {e}")
            }
        }
    }
    producers
}

impl ScoreResultProcessor {
                                                #[tracing::instrument(skip_all)]
    async fn process(&self, score: &abuse_proto::ScoreResult) -> Result<String, EnforcementFailed> {
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
                self.ctx.kafka_producer_decisions.as_ref(),
                &decision_outcome(
                    score,
                    topic,
                    "invalid_entity_id".into(),
                    false,
                    BTreeMap::new(),
                ),
            );
            return Ok("invalid_entity_id".into());
        }

        let incoming_class = self.ctx.dynamic_config.dedup_action_class_for_head(head);
        let claim_nonce = match self
            .dedup_cache
            .try_insert(entity_type, entity_id, incoming_class)
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
                publish_decision_outcome(
                    self.ctx.kafka_producer_decisions.as_ref(),
                    &decision_outcome(score, topic, "dedup_skipped".into(), false, BTreeMap::new()),
                );
                return Ok("dedup_skipped".into());
            }
        };

        let outcome = match run_enforcement(&self.ctx, score, topic).await {
            Ok(outcome) => outcome,
            Err(source) => {
                let permanent = source.downcast_ref::<PermanentAisFailure>().is_some();
                if permanent {
                    self.dedup_cache
                        .invalidate(entity_type, entity_id, claim_nonce)
                        .await;
                }
                return Err(EnforcementFailed {
                    permanent,
                    claim_nonce,
                    source,
                });
            }
        };

        write_dedup_outcome(
            &self.dedup_cache,
            entity_type,
            entity_id,
            score,
            topic,
            &outcome,
            claim_nonce,
        )
        .await;

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

                let msg_labels: Vec<&String> = score
                    .summary
                    .as_ref()
                    .map(|s| s.labels.iter().collect())
                    .unwrap_or_default();

                if !self.negative_labels.is_empty()
                    && msg_labels.iter().any(|l| self.negative_labels.contains(*l))
                {
                    metrics::KAFKA_MESSAGE_LATENCY
                        .with_label_values(&[topic, entity_type, "negative_label", "false"])
                        .observe(start.elapsed().as_secs_f64());
                    return None;
                }

                if !self.labels.is_empty() {
                    let matches: Vec<&&String> = msg_labels
                        .iter()
                        .filter(|l| self.labels.contains(**l))
                        .collect();
                    if matches.is_empty() {
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, "no_match", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        return None;
                    }

                    info!(?matches, "matching labels");
                }

                match self.process(&score).await {
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
                        Some((score, f.claim_nonce))
                    }
                }
            }
            .instrument(span)
        }))
        .await;

        {
            let mut queue = self.retry_queue.lock().await;
            for (score, claim_nonce) in failed_scores.into_iter().flatten() {
                queue.push(RetryEntry {
                    score,
                    attempt: 1,
                    next_retry_at: tokio::time::Instant::now() + Duration::from_secs(1),
                    topic: self.topic.clone(),
                    dedup_cache: self.dedup_cache.clone(),
                    claim_nonce,
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
            );
            let post_sw = sliding_window::SlidingWindowLimiter::new(
                lim.with_bucket(limiter::GLOBAL_BUCKET_POST_ID),
                mh_client.clone(),
                mh_tenant.clone(),
                cfg.limiter_fail_open,
                sliding_window::POST_SNAP_PKEY,
                "post",
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

    let rules_cache = rules::RulesCache::new();
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

    let kafka_producers = build_kafka_producers(&cfg, &dynamic_config).await;

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

fn consumer_error_total() -> f64 {
    use prometheus::core::Collector;
    xai_kafka::metrics::CONSUMER_ERROR_COUNT
        .collect()
        .iter()
        .flat_map(|mf| &mf.metric)
        .filter_map(|m| m.counter.as_ref())
        .filter_map(|c| c.value)
        .sum()
}

fn health_decision(
    consumed_once: bool,
    stale: Duration,
    error_bad_for: Duration,
    stale_after: Duration,
    stale_self_delete_after: Duration,
    error_self_delete_after: Duration,
    error_bad_now: bool,
) -> (bool, bool) {
    let ready = consumed_once && stale < stale_after && !error_bad_now;
    let should_self_delete =
        stale >= stale_self_delete_after || error_bad_for >= error_self_delete_after;
    (ready, should_self_delete)
}

struct WatchdogConfig {
    interval: Duration,
    stale_after: Duration,
    stale_self_delete_after: Duration,
    error_rate_per_sec: f64,
    error_self_delete_after: Duration,
    self_delete_enabled: bool,
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
    let mut errors_prev = consumer_error_total();
    let mut error_bad_since: Option<tokio::time::Instant> = None;
    let mut deleted = false;

    loop {
        ticker.tick().await;

        let last = last_progress_ms.load(Ordering::Relaxed);
        let consumed_once = last > 0;
        let reference = if consumed_once { last } else { started_ms };
        let stale = Duration::from_millis((now_millis() - reference).max(0) as u64);

        let errors_now = consumer_error_total();
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

pub async fn start_kafka_consumers(
    state: Arc<service::AppState>,
    cfg: &Config,
) -> Result<JoinHandle<()>> {
    let consumer_conn = cfg.kafka_consumer();
    let Some(sasl_password) = consumer_conn.sasl_password.clone() else {
        info!(
            "no Kafka consumer credential (KAFKA_SASL_PASSWORD / KAFKA_CONSUMER_SASL_PASSWORD unset) — Kafka consumer disabled"
        );
        state.kafka_ready.store(true, Ordering::Relaxed);
        return Ok(tokio::spawn(async {}));
    };

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

    let effective_topic_config = state
        .dynamic_config
        .kafka_consumer_topic_labels()
        .unwrap_or(topic_labels_json);
    let topics: HashMap<String, TopicConfig> = serde_json::from_value(effective_topic_config)
        .context("topic config has wrong shape (expected {topic: {processor, ...}})")?;
    info!("topic config: {} topic(s)", topics.len());
    for (topic, tc) in &topics {
        match tc {
            TopicConfig::ScoreResult {
                labels,
                negative_labels,
            } => {
                info!(
                    "  topic={topic}, processor=score_result, labels={}, negative_labels={}",
                    labels.len(),
                    negative_labels.len()
                );
            }
            TopicConfig::StatsOnly { should_log_content } => {
                info!(
                    "  topic={topic}, processor=stats_only, should_log_content={should_log_content}"
                );
            }
        }
    }

    if cfg.kafka_self_delete_enabled && !topics.is_empty() {
        const PREFLIGHT_ATTEMPTS: u32 = 3;
        const PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(10);
        const PREFLIGHT_BACKOFF: Duration = Duration::from_secs(2);

        let probe_topic = topics.keys().min().expect("topics non-empty").clone();
        let probe_config: KafkaConsumerConfig = KafkaConsumerBuilder::new(
            consumer_conn.dest.clone(),
            probe_topic.clone(),
            format!("{}-preflight", cfg.kafka_group_id()),
        )
        .with_wily_config(WilyConfig::default())
        .with_ssl(sasl_ssl_config(&consumer_conn, &sasl_password))
        .into();

        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let start = std::time::Instant::now();
            match probe_broker_reachability(probe_config.clone(), PREFLIGHT_TIMEOUT).await {
                Ok(()) => {
                    info!(
                        attempt,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        topic = %probe_topic,
                        "Kafka broker preflight ok"
                    );
                    break;
                }
                Err(e) if attempt < PREFLIGHT_ATTEMPTS => {
                    warn!(
                        topic = %probe_topic,
                        "Kafka broker preflight attempt {attempt}/{PREFLIGHT_ATTEMPTS} failed \
                         (retrying in {PREFLIGHT_BACKOFF:?}): {e:#}"
                    );
                    tokio::time::sleep(PREFLIGHT_BACKOFF).await;
                }
                Err(e) => {
                    error!(
                        topic = %probe_topic,
                        "Kafka broker preflight failed after {PREFLIGHT_ATTEMPTS} attempts — \
                         likely a bad node; self-deleting to reschedule: {e:#}"
                    );
                    metrics::KAFKA_SELF_DELETE_TOTAL
                        .with_label_values(&["preflight"])
                        .inc();
                    self_delete_pod().await;
                    return Err(anyhow::anyhow!(
                        "Kafka broker preflight failed after {PREFLIGHT_ATTEMPTS} attempts: {e:#}"
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

    let cred_hpr = Arc::new(
        Strato::new(&cfg.high_page_rank_column, StratoClientConfig::default())
            .context("Failed to create high-page-rank Strato client")?,
    );
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

    let uas_strato = Arc::new(
        Strato::new(&cfg.uas_column, StratoClientConfig::default())
            .context("Failed to create UAS Strato client")?,
    );
    info!(
        "UAS Strato client initialised for column {}",
        cfg.uas_column
    );

    let cancel = CancellationToken::new();
    {
        let cancel = cancel.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                cancel.cancel();
            }
        });
    }

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
                    let span = tracing::info_span!(
                        "retry",
                        entity_type = entity_type.as_str(),
                        entity_id,
                        attempt = entry.attempt
                    );

                    match run_enforcement(&retry_ctx, &entry.score, &entry.topic)
                        .instrument(span.clone())
                        .await
                    {
                        Ok(outcome) => {
                            info!(parent: &span, "retry resolved: {}", outcome.status);
                            metrics::RETRY_ATTEMPTS
                                .with_label_values(&["success"])
                                .observe(entry.attempt as f64);
                            write_dedup_outcome(
                                &entry.dedup_cache,
                                entity_type,
                                entity_id,
                                &entry.score,
                                &entry.topic,
                                &outcome,
                                entry.claim_nonce,
                            )
                            .await;
                        }
                        Err(e) if e.downcast_ref::<PermanentAisFailure>().is_some() => {
                            error!(parent: &span, "retry abandoned (permanent failure): {e}");
                            metrics::RETRY_ATTEMPTS
                                .with_label_values(&["permanent"])
                                .observe(entry.attempt as f64);
                            entry
                                .dedup_cache
                                .invalidate(entity_type, entity_id, entry.claim_nonce)
                                .await;
                        }
                        Err(e) => {
                            entry.attempt += 1;
                            if entry.attempt > MAX_RETRIES {
                                error!(parent: &span, "retry exhausted after {MAX_RETRIES} attempts: {e}");
                                metrics::RETRY_ATTEMPTS
                                    .with_label_values(&["exhausted"])
                                    .observe((entry.attempt - 1) as f64);
                                entry
                                    .dedup_cache
                                    .invalidate(entity_type, entity_id, entry.claim_nonce)
                                    .await;
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
    if topics.is_empty() {
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
            },
        ));
    }

    let mut handles: Vec<JoinHandle<()>> = Vec::new();

    let dedup_cache = state
        .dedup_cache
        .clone()
        .expect("Manhattan dedup cache required when Kafka is enabled");

    let kafka_group_id = cfg.kafka_group_id();

    let max_messages_per_poll = cfg.kafka_max_messages_per_poll.max(1);
    let max_in_flight = cfg.kafka_max_in_flight.max(1);
    info!(
        max_messages_per_poll,
        max_in_flight, "Kafka consumer batch limits"
    );

    let make_batch_config = |topic: &str| {
        let kafka = KafkaConsumerBuilder::new(
            consumer_conn.dest.clone(),
            topic.to_owned(),
            format!("{}-{}", kafka_group_id, topic),
        )
        .with_wily_config(WilyConfig::default())
        .with_ssl(sasl_ssl_config(&consumer_conn, &sasl_password))
        .with_enable_auto_offset_store(false)
        .with_enable_auto_commit(false)
        .with_fetch_timeout_ms(10000);

        BatchConsumerConfig::new(kafka, SERVICE_NAME)
            .with_max_messages_per_poll(max_messages_per_poll)
    };

    for (topic, topic_cfg) in topics {
        let batch_config = make_batch_config(&topic);
        let cancel = cancel.clone();

        match topic_cfg {
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
                };

                handles.push(tokio::spawn(async move {
                    info!("starting score_result Kafka consumer for topic: {topic}");
                    if let Err(e) = run_batch_consumer(batch_config, processor, cancel).await {
                        error!("Kafka consumer for topic {topic} exited with error: {e}");
                    }
                }));
            }
            TopicConfig::StatsOnly { should_log_content } => {
                let processor = StatsOnlyProcessor {
                    should_log_content,
                    last_progress: last_progress.clone(),
                };

                handles.push(tokio::spawn(async move {
                    info!("starting stats_only Kafka consumer for topic: {topic}");
                    if let Err(e) = run_batch_consumer(batch_config, processor, cancel).await {
                        error!(
                            "stats_only Kafka consumer for topic {topic} exited with error: {e}"
                        );
                    }
                }));
            }
        }
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

    Ok(tokio::spawn(async move {
        for h in handles {
            let _ = h.await;
        }
    }))
}

pub async fn serve(router: Router, cfg: &Config) -> Result<()> {
    let server = ServerBuilder::new(cfg.port)
        .merge(router)
        .drain_period(Duration::from_secs(cfg.drain_period_secs))
        .info(ServerInfo::new(SERVICE_NAME, SERVICE_VERSION))
        .on_ready(|| async {
            info!("Server ready and accepting requests");
        });
    server.run().await
}

pub async fn run() -> Result<()> {
    init_runtime_globals();

    let cfg = Config::parse();
    let (state, cfg) = build_state(cfg).await?;
    let api_keys = auth::ApiKeys::from_config(&cfg)?;
    if api_keys.is_empty() {
        warn!(
            "API_KEYS_JSON is empty — every request to the protected admin \
             endpoints (/action, /config*, /allowlist*) will return 401"
        );
    } else {
        info!(
            key_ids = ?api_keys.configured_key_ids(),
            "api-key auth configured for protected admin endpoints"
        );
    }
    let router = Router::new()
        .nest("/api", build_api_router(state.clone(), api_keys))
        .merge(build_docs_router());
    let kafka_consumers_handle = start_kafka_consumers(state.clone(), &cfg).await?;

    serve(router, &cfg).await?;

    state
        .kafka_producers
        .flush_all(Duration::from_secs(5))
        .await;
    kafka_consumers_handle.abort();
    info!("Server shutdown complete");
    Ok(())
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
mod dedup_retention_tests {
    use super::outcome_holds_full_dedup;

                        #[test]
    fn only_success_holds_full_dedup_window() {
        assert!(outcome_holds_full_dedup("success"));
        for skip in [
            "dry_run", 
            "dedup_skipped",
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
        ] {
            assert!(!outcome_holds_full_dedup(skip), "{skip} must not hold 24h");
        }
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
        let (ready, del) = health_decision(true, S5, Z, STALE, STALE_DELETE, ERR_DELETE, false);
        assert!(ready);
        assert!(!del);
    }

    #[test]
    fn stale_under_threshold_is_notready_not_deleted() {
        let s = Duration::from_secs(130);
        let (ready, del) = health_decision(true, s, Z, STALE, STALE_DELETE, ERR_DELETE, false);
        assert!(!ready);
        assert!(!del);
    }

    #[test]
    fn fully_dead_no_progress_self_deletes_at_stale_window() {
        let s = Duration::from_secs(241);
        let (ready, del) = health_decision(false, s, Z, STALE, STALE_DELETE, ERR_DELETE, false);
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
        );
        assert!(!ready); 
        assert!(!del); 
    }
}
