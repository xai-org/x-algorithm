use crate::decision::{ActionSpec, Decision};
use crate::facts::{CredFacts, EntityFacts, EntityType, Facts, GizmoduckFacts, PostFacts};
use cel::{Context, Program, Value};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, RwLock};
use thiserror::Error;
use tracing::warn;

const USER_RULE_FILE_YAML: &str = include_str!("../rules/enforcement_user.yaml");
const POST_RULE_FILE_YAML: &str = include_str!("../rules/enforcement_post.yaml");
const MAX_RECOMMENDATION_RESTRICTION_TTL_MSEC: i64 = 30 * 24 * 60 * 60 * 1000;


#[derive(Debug, Deserialize)]
struct RuleFile {
                    #[serde(default)]
    for_entity: EntityType,
    rules: Vec<RuleDef>,
}

#[derive(Debug, Deserialize)]
struct RuleDef {
    id: String,
    when: String,
    then: Outcome,
}


#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ActionStep {
                                ActAddLabelsV2 {
        labels: Vec<String>,
        #[serde(default)]
        ttl_msec: Option<i64>,
    },
                                                ActAddPostLabelsV2 {
        labels: Vec<String>,
        #[serde(default)]
        ttl_msec: Option<i64>,
    },
            ActSuspendUser { perm: bool, policy: String },
                ActArkose,
                ActCaptcha,
                        ActSpamLivenessCheck,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SpecialOutcome {
                    Skip { reason: String },
                        ActAll { actions: Vec<ActionStep> },
                        ActRequestedActions,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum Outcome {
    Special(SpecialOutcome),
    Action(ActionStep),
}


#[derive(Debug)]
struct CompiledRule {
    id: String,
    when: Program,
    outcome: Outcome,
}

#[derive(Debug)]
pub struct CompiledRules {
            entity_type: EntityType,
    rules: Vec<CompiledRule>,
}


#[derive(Debug, Error)]
pub enum RuleCompileError {
    #[error("rule file is not valid YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("rule '{id}' has invalid CEL `when` expression: {source:?}")]
    Cel {
        id: String,
        source: cel::ParseErrors,
    },
    #[error(
        "rule '{id}' has an empty `act_all.actions` list; a composite must dispatch at least one action"
    )]
    EmptyActAll { id: String },
    #[error("rule '{id}' applies recommendation-limiting label '{label}' without an expiry")]
    MissingRecommendationRestrictionTtl { id: String, label: String },
    #[error(
        "rule '{id}' applies recommendation-limiting label '{label}' with invalid ttl_msec {ttl_msec}; expected 1..={max_ttl_msec}"
    )]
    InvalidRecommendationRestrictionTtl {
        id: String,
        label: String,
        ttl_msec: i64,
        max_ttl_msec: i64,
    },
}

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("failed to serialize Facts into the CEL context: {0}")]
    Bind(String),
    #[error("rule '{id}' raised execution error: {source}")]
    Exec {
        id: String,
        source: cel::ExecutionError,
    },
    #[error("rule '{id}' returned non-boolean value: {value:?}")]
    NonBoolResult { id: String, value: String },
    #[error(
        "no rule matched — the pipeline must have a terminal `when: true` rule \
         (the act:suspend default); check the rule file"
    )]
    NoMatch,
}

impl CompiledRules {
                        pub fn from_yaml(src: &str) -> Result<Self, RuleCompileError> {
        let file: RuleFile = serde_yaml::from_str(src)?;
        let rules = file
            .rules
            .into_iter()
            .map(|r| {
                let when = Program::compile(&r.when).map_err(|e| RuleCompileError::Cel {
                    id: r.id.clone(),
                    source: e,
                })?;
                if let Outcome::Special(SpecialOutcome::ActAll { actions }) = &r.then
                    && actions.is_empty()
                {
                    return Err(RuleCompileError::EmptyActAll { id: r.id.clone() });
                }
                validate_recommendation_restriction_ttls(&r.id, &r.then)?;
                Ok(CompiledRule {
                    id: r.id,
                    when,
                    outcome: r.then,
                })
            })
            .collect::<Result<Vec<_>, RuleCompileError>>()?;
        Ok(CompiledRules {
            entity_type: file.for_entity,
            rules,
        })
    }

    pub fn rule_ids(&self) -> impl Iterator<Item = &str> {
        self.rules.iter().map(|r| r.id.as_str())
    }

                pub fn entity_type(&self) -> EntityType {
        self.entity_type
    }
}

fn is_recommendation_limiting_label(label: &str) -> bool {
    let normalized: String = label
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    matches!(
        normalized.as_str(),
        "spamhighrecall" | "donotamplify" | "abusivehighrecall"
    )
}

fn validate_action_ttl(rule_id: &str, action: &ActionStep) -> Result<(), RuleCompileError> {
    let (labels, ttl_msec) = match action {
        ActionStep::ActAddLabelsV2 { labels, ttl_msec }
        | ActionStep::ActAddPostLabelsV2 { labels, ttl_msec } => (labels, ttl_msec),
        _ => return Ok(()),
    };

    for label in labels
        .iter()
        .filter(|label| is_recommendation_limiting_label(label))
    {
        let Some(ttl_msec) = ttl_msec else {
            return Err(RuleCompileError::MissingRecommendationRestrictionTtl {
                id: rule_id.to_string(),
                label: label.clone(),
            });
        };
        if *ttl_msec <= 0 || *ttl_msec > MAX_RECOMMENDATION_RESTRICTION_TTL_MSEC {
            return Err(RuleCompileError::InvalidRecommendationRestrictionTtl {
                id: rule_id.to_string(),
                label: label.clone(),
                ttl_msec: *ttl_msec,
                max_ttl_msec: MAX_RECOMMENDATION_RESTRICTION_TTL_MSEC,
            });
        }
    }
    Ok(())
}

fn validate_recommendation_restriction_ttls(
    rule_id: &str,
    outcome: &Outcome,
) -> Result<(), RuleCompileError> {
    match outcome {
        Outcome::Special(SpecialOutcome::ActAll { actions }) => {
            for action in actions {
                validate_action_ttl(rule_id, action)?;
            }
            Ok(())
        }
        Outcome::Action(action) => validate_action_ttl(rule_id, action),
        Outcome::Special(SpecialOutcome::Skip { .. }) => Ok(()),
    }
}

fn step_to_spec(step: &ActionStep) -> ActionSpec {
    match step {
        ActionStep::ActAddLabelsV2 { labels, ttl_msec } => ActionSpec::AddLabelsV2 {
            labels: labels.clone(),
            ttl_msec: *ttl_msec,
        },
        ActionStep::ActAddPostLabelsV2 { labels, ttl_msec } => ActionSpec::AddPostLabelsV2 {
            labels: labels.clone(),
            ttl_msec: *ttl_msec,
        },
        ActionStep::ActSuspendUser { perm, policy } => ActionSpec::SuspendUser {
            perm: *perm,
            policy: policy.clone(),
        },
        ActionStep::ActArkose => ActionSpec::Arkose,
        ActionStep::ActCaptcha => ActionSpec::Captcha,
        ActionStep::ActSpamLivenessCheck => ActionSpec::SpamLivenessCheck,
    }
}

fn outcome_to_decision(o: &Outcome) -> Decision {
    match o {
        Outcome::Special(SpecialOutcome::Skip { reason }) => Decision::Skip(reason.clone()),
        Outcome::Special(SpecialOutcome::ActAll { actions }) => {
            Decision::Act(actions.iter().map(step_to_spec).collect())
        }
        Outcome::Special(SpecialOutcome::ActRequestedActions) => Decision::ActRequestedActions,
        Outcome::Action(step) => Decision::Act(vec![step_to_spec(step)]),
    }
}


#[derive(Serialize)]
struct ScoreCel<'a> {
                labels: &'a [String],
    model_version: &'a str,
                        skip_author_credibility_prechecks: bool,
                        requested_actions: Vec<RequestedActionCel<'a>>,
            policy_version: &'a str,
}

#[derive(Serialize)]
struct RequestedActionCel<'a> {
    kind: &'a str,
    perm: bool,
    policy: &'a str,
    labels: &'a [String],
    ttl_msec: i64,
    head: &'a str,
}

#[derive(Serialize)]
struct AllowlistCel {
    is_allowlisted: bool,
}

#[derive(Serialize)]
struct UserCel<'a> {
    present: bool,
    suspended: bool,
    deactivated: bool,
            labels: &'a [String],
}

#[derive(Serialize)]
struct CredCel<'a> {
    is_high: bool,
    verified_type: &'a str,
    follower_count: i64,
    score: f64,
}

#[derive(Serialize)]
struct PostCel<'a> {
    present: bool,
    author_id: i64,
    labels: &'a [String],
}

fn project_score(facts: &Facts) -> ScoreCel<'_> {
    ScoreCel {
        labels: &facts.score.labels,
        model_version: &facts.score.model_version,
        skip_author_credibility_prechecks: facts.score.skip_author_credibility_prechecks,
        requested_actions: facts
            .score
            .requested_actions
            .iter()
            .map(|a| RequestedActionCel {
                kind: &a.kind,
                perm: a.perm,
                policy: &a.policy,
                labels: &a.labels,
                ttl_msec: a.ttl_msec,
                head: &a.head,
            })
            .collect(),
        policy_version: &facts.score.policy_version,
    }
}

fn project_allowlist(al: &crate::facts::AllowlistFacts) -> AllowlistCel {
    AllowlistCel {
        is_allowlisted: al.is_allowlisted,
    }
}

fn project_user(user: &GizmoduckFacts) -> UserCel<'_> {
    UserCel {
        present: user.present,
        suspended: user.suspended,
        deactivated: user.deactivated,
        labels: &user.labels,
    }
}

fn project_cred(cred: &CredFacts) -> CredCel<'_> {
    CredCel {
        is_high: cred.is_high.unwrap_or(false),
        verified_type: cred.verified_type.as_deref().unwrap_or(""),
        follower_count: cred.follower_count.unwrap_or(0),
        score: cred.score.unwrap_or(0.0),
    }
}

fn project_post(post: &PostFacts) -> PostCel<'_> {
    PostCel {
        present: post.present,
        author_id: post.author_id,
        labels: &post.labels,
    }
}


pub fn decide_with(rules: &CompiledRules, facts: &Facts) -> Result<Decision, EvalError> {
    let mut ctx = Context::default();
    bind_facts(&mut ctx, facts)?;

    for rule in &rules.rules {
        let value = rule.when.execute(&ctx).map_err(|e| EvalError::Exec {
            id: rule.id.clone(),
            source: e,
        })?;
        match value {
            Value::Bool(true) => return Ok(outcome_to_decision(&rule.outcome)),
            Value::Bool(false) => continue,
            other => {
                return Err(EvalError::NonBoolResult {
                    id: rule.id.clone(),
                    value: format!("{other:?}"),
                });
            }
        }
    }
    Err(EvalError::NoMatch)
}

type CredAliasConfig =
    std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>>;

static CRED_ALIASES: LazyLock<CredAliasConfig> =
    LazyLock::new(|| match std::env::var("RULES_YAML_ALIASES") {
        Ok(raw) => parse_cred_aliases(&raw),
        Err(_) => CredAliasConfig::default(),
    });

fn parse_cred_aliases(json: &str) -> CredAliasConfig {
    if json.is_empty() {
        return CredAliasConfig::default();
    }
    serde_json::from_str(json).unwrap_or_else(|e| {
        warn!("RULES_YAML_ALIASES ignored (parse error): {e}");
        CredAliasConfig::default()
    })
}

fn build_cred_alias(
    cred: &CredFacts,
    field_aliases: &std::collections::BTreeMap<String, String>,
) -> serde_json::Value {
    let mut value = serde_json::to_value(project_cred(cred)).unwrap_or(serde_json::Value::Null);
    if let Some(map) = value.as_object_mut() {
        for (old_field, current_field) in field_aliases {
            if let Some(v) = map.get(current_field).cloned() {
                map.insert(old_field.clone(), v);
            }
        }
    }
    value
}

fn bind_cred_aliases(ctx: &mut Context<'_>, cred: &CredFacts) -> Result<(), EvalError> {
    for (alias_var, field_aliases) in &*CRED_ALIASES {
        ctx.add_variable(alias_var.as_str(), build_cred_alias(cred, field_aliases))
            .map_err(|e| EvalError::Bind(format!("{alias_var}: {e}")))?;
    }
    Ok(())
}

fn bind_facts(ctx: &mut Context<'_>, facts: &Facts) -> Result<(), EvalError> {
    ctx.add_variable("score", project_score(facts))
        .map_err(|e| EvalError::Bind(format!("score: {e}")))?;
    ctx.add_variable("user_allowlist", project_allowlist(facts.user_allowlist()))
        .map_err(|e| EvalError::Bind(format!("user_allowlist: {e}")))?;
    match &facts.entity {
        EntityFacts::User(s) => {
            ctx.add_variable("user", project_user(&s.user))
                .map_err(|e| EvalError::Bind(format!("user: {e}")))?;
            ctx.add_variable("cred", project_cred(&s.cred))
                .map_err(|e| EvalError::Bind(format!("cred: {e}")))?;
            bind_cred_aliases(ctx, &s.cred)?;
        }
        EntityFacts::Post(p) => {
            ctx.add_variable("user", project_user(&p.author.user))
                .map_err(|e| EvalError::Bind(format!("user: {e}")))?;
            ctx.add_variable("cred", project_cred(&p.author.cred))
                .map_err(|e| EvalError::Bind(format!("cred: {e}")))?;
            bind_cred_aliases(ctx, &p.author.cred)?;
            ctx.add_variable("post", project_post(p))
                .map_err(|e| EvalError::Bind(format!("post: {e}")))?;
            ctx.add_variable("post_allowlist", project_allowlist(&p.allowlist))
                .map_err(|e| EvalError::Bind(format!("post_allowlist: {e}")))?;
        }
    }
    Ok(())
}


fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn nonempty(override_yaml: Option<&str>) -> Option<&str> {
    override_yaml.map(str::trim).filter(|s| !s.is_empty())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleSource {
            Static,
        Dynamic,
                    LastGood,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuleStatus {
    pub entity_type: String,
                    pub source: RuleSource,
        pub override_present: bool,
            pub override_hash: Option<String>,
                pub error: Option<String>,
        pub rule_ids: Vec<String>,
}

struct CacheEntry {
    hash: u64,
    compiled: Result<Arc<CompiledRules>, String>,
}

struct RuleSlot {
    entity_type: EntityType,
                baked_in: Arc<CompiledRules>,
            last_good: RwLock<Arc<CompiledRules>>,
    cached: RwLock<Option<CacheEntry>>,
}

impl RuleSlot {
    fn from_compiled(compiled: Arc<CompiledRules>) -> Self {
        Self {
            entity_type: compiled.entity_type,
            baked_in: compiled.clone(),
            last_good: RwLock::new(compiled),
            cached: RwLock::new(None),
        }
    }

    fn from_yaml(yaml: &str, expected: EntityType) -> Self {
        let compiled =
            CompiledRules::from_yaml(yaml).expect("baked-in / seeded rule file must compile");
        assert_eq!(
            compiled.entity_type, expected,
            "rule file `for_entity` ({:?}) does not match slot ({:?})",
            compiled.entity_type, expected,
        );
        Self::from_compiled(Arc::new(compiled))
    }

    fn last_good_arc(&self) -> Arc<CompiledRules> {
        self.last_good.read().unwrap().clone()
    }

            fn source_for_fallback(&self, last: &Arc<CompiledRules>) -> RuleSource {
        if Arc::ptr_eq(last, &self.baked_in) {
            RuleSource::Static
        } else {
            RuleSource::LastGood
        }
    }

            fn compile_and_cache(&self, src: &str, h: u64) -> Result<Arc<CompiledRules>, String> {
        let mut w = self.cached.write().unwrap();
        if let Some(e) = &*w
            && e.hash == h
        {
            return e.compiled.clone();
        }
        let et = self.entity_type.as_str();
        let compiled: Result<Arc<CompiledRules>, String> = match CompiledRules::from_yaml(src) {
            Ok(c) if c.entity_type == self.entity_type => Ok(Arc::new(c)),
            Ok(c) => Err(format!(
                "override `for_entity` ({:?}) does not match this pipeline ({et})",
                c.entity_type,
            )),
            Err(e) => Err(e.to_string()),
        };
        let label = if compiled.is_ok() { "success" } else { "fail" };
        crate::metrics::RULES_YAML_COMPILED
            .with_label_values(&[et, label])
            .inc();
        if let Ok(arc) = &compiled {
            *self.last_good.write().unwrap() = arc.clone();
        } else if let Err(msg) = &compiled {
            warn!(
                entity_type = et,
                error = %msg,
                "rules override rejected — keeping last-good pipeline"
            );
        }
        *w = Some(CacheEntry {
            hash: h,
            compiled: compiled.clone(),
        });
        compiled
    }

            fn resolve(&self, override_yaml: Option<&str>) -> Arc<CompiledRules> {
        if let Some(src) = nonempty(override_yaml) {
            let h = hash_str(src);
            if let Some(e) = &*self.cached.read().unwrap()
                && e.hash == h
            {
                return match &e.compiled {
                    Ok(arc) => arc.clone(),
                    Err(_) => self.last_good_arc(),
                };
            }
            return match self.compile_and_cache(src, h) {
                Ok(arc) => arc,
                Err(_) => self.last_good_arc(),
            };
        }
        self.last_good_arc()
    }

    fn status(&self, override_yaml: Option<&str>) -> RuleStatus {
        let entity = self.entity_type.as_str();
        let Some(src) = nonempty(override_yaml) else {
            let last = self.last_good_arc();
            return RuleStatus {
                entity_type: entity.to_string(),
                source: self.source_for_fallback(&last),
                override_present: false,
                override_hash: None,
                error: None,
                rule_ids: last.rule_ids().map(str::to_string).collect(),
            };
        };
        let h = hash_str(src);
        let cached = {
            let guard = self.cached.read().unwrap();
            match &*guard {
                Some(e) if e.hash == h => Some(e.compiled.clone()),
                _ => None,
            }
        };
        let compiled = match cached {
            Some(c) => c,
            None => self.compile_and_cache(src, h),
        };
        let override_hash = Some(format!("{h:016x}"));
        match compiled {
            Ok(arc) => RuleStatus {
                entity_type: entity.to_string(),
                source: RuleSource::Dynamic,
                override_present: true,
                override_hash,
                error: None,
                rule_ids: arc.rule_ids().map(str::to_string).collect(),
            },
            Err(msg) => {
                let last = self.last_good_arc();
                RuleStatus {
                    entity_type: entity.to_string(),
                    source: self.source_for_fallback(&last),
                    override_present: true,
                    override_hash,
                    error: Some(msg),
                    rule_ids: last.rule_ids().map(str::to_string).collect(),
                }
            }
        }
    }
}

pub struct RulesCache {
    user: RuleSlot,
    post: RuleSlot,
}

impl RulesCache {
                pub fn new() -> Self {
        let cache = Self {
            user: RuleSlot::from_yaml(USER_RULE_FILE_YAML, EntityType::User),
            post: RuleSlot::from_yaml(POST_RULE_FILE_YAML, EntityType::Post),
        };
        crate::metrics::RULES_YAML_COMPILED
            .with_label_values(&["user", "success"])
            .inc();
        crate::metrics::RULES_YAML_COMPILED
            .with_label_values(&["post", "success"])
            .inc();
        cache
    }

                    pub fn try_seed(user_yaml: &str, post_yaml: &str) -> Result<Self, String> {
        let user = CompiledRules::from_yaml(user_yaml).map_err(|e| format!("user rules: {e}"))?;
        if user.entity_type != EntityType::User {
            return Err(format!(
                "user rules `for_entity` is {:?}, expected User",
                user.entity_type
            ));
        }
        let post = CompiledRules::from_yaml(post_yaml).map_err(|e| format!("post rules: {e}"))?;
        if post.entity_type != EntityType::Post {
            return Err(format!(
                "post rules `for_entity` is {:?}, expected Post",
                post.entity_type
            ));
        }
        let cache = Self {
            user: RuleSlot::from_compiled(Arc::new(user)),
            post: RuleSlot::from_compiled(Arc::new(post)),
        };
        crate::metrics::RULES_YAML_COMPILED
            .with_label_values(&["user", "success"])
            .inc();
        crate::metrics::RULES_YAML_COMPILED
            .with_label_values(&["post", "success"])
            .inc();
        Ok(cache)
    }

    fn slot(&self, entity_type: EntityType) -> &RuleSlot {
        match entity_type {
            EntityType::User => &self.user,
            EntityType::Post => &self.post,
        }
    }

            pub fn resolve(
        &self,
        entity_type: EntityType,
        override_yaml: Option<&str>,
    ) -> Arc<CompiledRules> {
        self.slot(entity_type).resolve(override_yaml)
    }

        pub fn status(&self, entity_type: EntityType, override_yaml: Option<&str>) -> RuleStatus {
        self.slot(entity_type).status(override_yaml)
    }
}

impl Default for RulesCache {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{ActionSpec, Decision};
    use crate::facts::{AllowlistFacts, CredFacts, GizmoduckFacts, ScoreFacts, UserFacts};
    use xai_abuse_proto::enforcement::ScoreResult;

    const MOCK_USER_RULES_YAML: &str = include_str!("fixtures/mock_rules_user.yaml");
    const MOCK_POST_RULES_YAML: &str = include_str!("fixtures/mock_rules_post.yaml");

        fn mock_rules(entity_type: EntityType) -> CompiledRules {
        let yaml = match entity_type {
            EntityType::User => MOCK_USER_RULES_YAML,
            EntityType::Post => MOCK_POST_RULES_YAML,
        };
        let compiled = CompiledRules::from_yaml(yaml).expect("mock rules must compile");
        assert_eq!(compiled.entity_type, entity_type);
        compiled
    }


    const OVERRIDE_USER_YAML: &str = "for_entity: user\nrules:\n  - id: ovr_rule\n    when: \"true\"\n    then: {kind: skip, reason: t}\n";

    const SEED_USER: &str = "for_entity: user\nrules:\n  - id: seed_user\n    when: \"true\"\n    then: {kind: skip, reason: seed}\n";
    const SEED_POST: &str = "for_entity: post\nrules:\n  - id: seed_post\n    when: \"true\"\n    then: {kind: skip, reason: seed}\n";

    fn minimal_seeded() -> RulesCache {
        RulesCache::try_seed(SEED_USER, SEED_POST).expect("minimal seed")
    }

    #[test]
    fn baked_in_defaults_compile() {
        let cache = RulesCache::new();
        let user = cache.status(EntityType::User, None);
        let post = cache.status(EntityType::Post, None);
        assert_eq!(user.source, RuleSource::Static);
        assert_eq!(post.source, RuleSource::Static);
        assert!(!user.rule_ids.is_empty(), "user pipeline empty");
        assert!(!post.rule_ids.is_empty(), "post pipeline empty");
        for id in [
            "user_in_allowlist",
            "user_not_found",
            "very_high_follower_count",
            "high_follower_count",
            "pagerank_skipped",
        ] {
            assert!(
                user.rule_ids.iter().any(|r| r == id),
                "user missing guardrail {id:?}; got {:?}",
                user.rule_ids
            );
        }
        let pos = |id: &str| {
            user.rule_ids
                .iter()
                .position(|r| r == id)
                .unwrap_or_else(|| panic!("user pipeline missing {id:?}"))
        };
        assert!(
            pos("very_high_follower_count") < pos("high_follower_count"),
            "very_high_follower_count must precede high_follower_count; got {:?}",
            user.rule_ids
        );
        for id in [
            "post_in_allowlist",
            "user_in_allowlist",
            "user_not_found",
            "high_follower_count",
            "pagerank_skipped",
        ] {
            assert!(
                post.rule_ids.iter().any(|r| r == id),
                "post missing guardrail {id:?}; got {:?}",
                post.rule_ids
            );
        }
    }

    #[test]
    fn rules_cache_uses_valid_override() {
        let cache = minimal_seeded();
        let info = cache.status(EntityType::User, Some(OVERRIDE_USER_YAML));
        assert_eq!(info.source, RuleSource::Dynamic);
        assert!(info.override_present);
        assert_eq!(info.error, None);
        assert_eq!(info.rule_ids, vec!["ovr_rule".to_string()]);
        assert!(info.override_hash.is_some());
        assert_eq!(
            cache.status(EntityType::Post, None).source,
            RuleSource::Static
        );
        assert!(
            cache
                .status(EntityType::Post, None)
                .rule_ids
                .contains(&"seed_post".to_string())
        );
    }

    #[test]
    fn rules_cache_none_uses_static_seed() {
        let cache = minimal_seeded();
        let info = cache.status(EntityType::User, None);
        assert_eq!(info.source, RuleSource::Static);
        assert_eq!(info.override_hash, None);
        assert!(info.rule_ids.contains(&"seed_user".to_string()));
    }

    #[test]
    fn rules_cache_blank_override_uses_static_seed() {
        let cache = minimal_seeded();
        let s = cache.status(EntityType::User, Some("   \n "));
        assert_eq!(s.source, RuleSource::Static);
        assert!(!s.override_present);
    }

    #[test]
    fn rules_cache_falls_back_on_bad_yaml() {
        let cache = minimal_seeded();
        let info = cache.status(EntityType::User, Some("not: [valid"));
        assert_eq!(info.source, RuleSource::Static);
        assert!(info.override_present);
        assert!(info.error.is_some());
        assert!(info.rule_ids.contains(&"seed_user".to_string()));
    }

    #[test]
    fn rules_cache_bad_after_good_keeps_last_good() {
        let cache = minimal_seeded();
        let _ = cache.status(EntityType::User, Some(OVERRIDE_USER_YAML));
        assert_eq!(
            cache
                .status(EntityType::User, Some(OVERRIDE_USER_YAML))
                .source,
            RuleSource::Dynamic
        );
        let info = cache.status(EntityType::User, Some("not: [valid"));
        assert_eq!(info.source, RuleSource::LastGood);
        assert!(info.override_present);
        assert!(info.error.is_some());
        assert_eq!(info.rule_ids, vec!["ovr_rule".to_string()]);
    }

    #[test]
    fn rules_cache_falls_back_on_entity_mismatch() {
        let cache = minimal_seeded();
        let yaml = "for_entity: post\nrules:\n  - id: x\n    when: \"true\"\n    then: {kind: skip, reason: t}\n";
        let info = cache.status(EntityType::User, Some(yaml));
        assert_eq!(info.source, RuleSource::Static);
        assert!(info.override_present);
        assert!(info.error.as_deref().unwrap_or("").contains("for_entity"));
        assert!(info.rule_ids.contains(&"seed_user".to_string()));
    }

    #[test]
    fn rules_cache_resolve_returns_override_pipeline() {
        let cache = minimal_seeded();
        let compiled = cache.resolve(EntityType::User, Some(OVERRIDE_USER_YAML));
        let ids: Vec<&str> = compiled.rule_ids().collect();
        assert_eq!(ids, vec!["ovr_rule"]);
    }

    #[test]
    fn rules_cache_try_seed_rejects_entity_mismatch() {
        let err = match RulesCache::try_seed(SEED_POST, SEED_POST) {
            Ok(_) => panic!("expected entity mismatch error"),
            Err(e) => e,
        };
        assert!(err.contains("User") || err.contains("user"), "{err}");
    }

    #[test]
    fn baked_in_user_allowlist_skips() {
        let rules = RulesCache::new().resolve(EntityType::User, None);
        let mut f = base_facts();
        f.user_allowlist_mut().is_allowlisted = true;
        assert_eq!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("user_in_allowlist".into())
        );
    }


    fn base_facts() -> Facts {
        Facts {
            entity_type: crate::facts::EntityType::User,
            entity_id: 1,
            user_id: 1,
            topic: "test.topic".into(),
            score: ScoreFacts::from_score(&ScoreResult::default()),
            entity: EntityFacts::User(UserFacts {
                allowlist: AllowlistFacts::default(),
                user: GizmoduckFacts {
                    present: true,
                    ..Default::default()
                },
                cred: CredFacts::default(),
            }),
            uas: None,
        }
    }

                fn post_facts(author: UserFacts, score_labels: Vec<String>) -> Facts {
        let mut f = base_facts();
        f.entity_type = EntityType::Post;
        f.entity_id = 999; 
        f.user_id = 1; 
        f.score.labels = score_labels;
        f.entity = EntityFacts::Post(PostFacts {
            present: false,
            author_id: 1,
            labels: vec![],
            allowlist: AllowlistFacts::default(),
            author,
        });
        f
    }

        fn with_post_allowlisted(mut f: Facts) -> Facts {
        let EntityFacts::Post(ref mut p) = f.entity else {
            panic!("with_post_allowlisted requires a Post entity");
        };
        p.allowlist.is_allowlisted = true;
        f
    }

        fn with_post_labels(mut f: Facts, labels: Vec<String>) -> Facts {
        let EntityFacts::Post(ref mut p) = f.entity else {
            panic!("with_post_labels requires a Post entity");
        };
        p.labels = labels;
        f
    }

    fn safe_author() -> UserFacts {
        UserFacts {
            user: GizmoduckFacts {
                present: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

        fn d(f: &Facts) -> Decision {
        decide_with(&mock_rules(f.entity_type), f).expect("mock rule eval should succeed")
    }

                fn act_one(d: Decision) -> ActionSpec {
        match d {
            Decision::Act(mut v) if v.len() == 1 => v.pop().unwrap(),
            other => panic!("expected a single-action Act, got {other:?}"),
        }
    }


    #[test]
    fn malformed_or_unknown_then_block_is_rejected() {
        let cases: &[&str] = &[
            "rules:\n  - id: x\n    when: \"true\"\n    then: {reason: foo}\n",
            "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: lol}\n",
            "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: skip}\n",
            "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: act_add_labels_v2, ttl_msec: 100}\n",
            "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: act_suspend_user, perm: false}\n",
            "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: act_all}\n",
            "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: act_all, actions: [{kind: act_add_labels_v2, ttl_msec: 100}]}\n",
            "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: act_all, actions: [{kind: lol}]}\n",
        ];
        for bad in cases {
            let err = CompiledRules::from_yaml(bad).unwrap_err();
            assert!(
                matches!(err, RuleCompileError::Yaml(_)),
                "expected Yaml error for: {bad:?}, got {err:?}",
            );
        }
    }

    #[test]
    fn bad_cel_when_expression_is_rejected_at_compile_time() {
        let yaml = r#"
rules:
  - id: broken
    when: "this is not cel ((("
    then:
      kind: act_suspend_user
      perm: false
      policy: TestPolicy
"#;
        let err = CompiledRules::from_yaml(yaml).unwrap_err();
        assert!(matches!(err, RuleCompileError::Cel { .. }));
    }

    #[test]
    fn empty_act_all_is_rejected_at_compile_time() {
        let yaml = r#"
rules:
  - id: empty_composite
    when: "true"
    then:
      kind: act_all
      actions: []
"#;
        let err = CompiledRules::from_yaml(yaml).unwrap_err();
        assert!(matches!(err, RuleCompileError::EmptyActAll { .. }));
    }


    #[test]
    fn cred_alias_config_parses_and_ignores_malformed() {
        assert!(parse_cred_aliases("").is_empty());
        assert!(parse_cred_aliases("not json").is_empty());
        let cfg =
            parse_cred_aliases(r#"{"legacyvar":{"legacyflag":"is_high","legacyscore":"score"}}"#);
        assert_eq!(cfg["legacyvar"]["legacyflag"], "is_high");
        assert_eq!(cfg["legacyvar"]["legacyscore"], "score");
    }

    #[test]
    fn cred_alias_exposes_old_and_current_field_names() {
        let fields: std::collections::BTreeMap<String, String> = [
            ("legacyflag".to_string(), "is_high".to_string()),
            ("legacyscore".to_string(), "score".to_string()),
        ]
        .into_iter()
        .collect();
        let cred = CredFacts {
            is_high: Some(true),
            verified_type: Some("SomeType".into()),
            score: Some(60.0),
            follower_count: Some(1234),
        };
        let v = build_cred_alias(&cred, &fields);
        assert_eq!(v["legacyflag"], serde_json::json!(true));
        assert_eq!(v["legacyscore"], serde_json::json!(60.0));
        assert_eq!(v["is_high"], serde_json::json!(true));
        assert_eq!(v["verified_type"], serde_json::json!("SomeType"));
        assert_eq!(v["follower_count"], serde_json::json!(1234));
    }

    #[test]
    fn cred_alias_resolves_in_cel() {
        let fields: std::collections::BTreeMap<String, String> =
            [("legacyflag".to_string(), "is_high".to_string())]
                .into_iter()
                .collect();
        let cred = CredFacts {
            is_high: Some(true),
            ..Default::default()
        };
        let mut ctx = cel::Context::default();
        ctx.add_variable("legacyvar", build_cred_alias(&cred, &fields))
            .unwrap();
        let prog = cel::Program::compile("legacyvar.legacyflag && legacyvar.is_high").unwrap();
        assert!(matches!(
            prog.execute(&ctx).unwrap(),
            cel::Value::Bool(true)
        ));
    }


    fn one_rule_pipeline(yaml: &str) -> CompiledRules {
        CompiledRules::from_yaml(yaml).expect("one-rule yaml should compile")
    }

    #[test]
    fn skip_author_credibility_prechecks_flag_gates_an_opted_in_guard_rule() {
        let rules = one_rule_pipeline(
            r#"
rules:
  - id: high_follower_count
    when: "cred.follower_count >= 12.34 && !score.skip_author_credibility_prechecks"
    then: {kind: skip, reason: high_follower_count}
  - id: fallthrough_act
    when: "true"
    then: {kind: act_captcha}
"#,
        );
        let mut f = base_facts();
        let EntityFacts::User(ref mut u) = f.entity else {
            unreachable!("base_facts is a User entity");
        };
        u.cred.follower_count = Some(50_000);

        assert_eq!(
            decide_with(&rules, &f).expect("eval should succeed"),
            Decision::Skip("high_follower_count".into())
        );

        f.score.skip_author_credibility_prechecks = true;
        assert_eq!(
            decide_with(&rules, &f).expect("eval should succeed"),
            Decision::Act(vec![ActionSpec::Captcha])
        );
    }

    #[test]
    fn act_arkose_outcome_parses_and_maps_to_arkose_decision() {
        let rules = one_rule_pipeline(
            r#"
rules:
  - id: arkose_test
    when: "true"
    then:
      kind: act_arkose
"#,
        );
        let f = base_facts();
        let decision = decide_with(&rules, &f).expect("eval should succeed");
        assert_eq!(decision, Decision::Act(vec![ActionSpec::Arkose]));
    }

    #[test]
    fn act_captcha_outcome_parses_and_maps_to_captcha_decision() {
        let rules = one_rule_pipeline(
            r#"
rules:
  - id: captcha_test
    when: "true"
    then:
      kind: act_captcha
"#,
        );
        let f = base_facts();
        let decision = decide_with(&rules, &f).expect("eval should succeed");
        assert_eq!(decision, Decision::Act(vec![ActionSpec::Captcha]));
    }

    #[test]
    fn act_add_labels_v2_without_ttl_parses_as_no_expiry() {
        let rules = one_rule_pipeline(
            r#"
rules:
  - id: label_no_ttl
    when: "true"
    then:
      kind: act_add_labels_v2
      labels: ["MockLabel"]
"#,
        );
        let f = base_facts();
        let decision = decide_with(&rules, &f).expect("eval should succeed");
        assert_eq!(
            decision,
            Decision::Act(vec![ActionSpec::AddLabelsV2 {
                labels: vec!["MockLabel".into()],
                ttl_msec: None,
            }]),
        );
    }

    #[test]
    fn recommendation_restriction_requires_ttl() {
        let yaml = r#"
rules:
  - id: no_expiry
    when: "true"
    then: { kind: act_add_labels_v2, labels: ["SpamHighRecall"] }
"#;
        assert!(matches!(
            CompiledRules::from_yaml(yaml),
            Err(RuleCompileError::MissingRecommendationRestrictionTtl { id, label })
                if id == "no_expiry" && label == "SpamHighRecall"
        ));
    }

    #[test]
    fn recommendation_restriction_rejects_invalid_ttl() {
        for ttl_msec in [-1, 0, MAX_RECOMMENDATION_RESTRICTION_TTL_MSEC + 1] {
            let yaml = format!(
                "rules:\n  - id: invalid_expiry\n    when: \"true\"\n    then: {{ kind: act_add_post_labels_v2, labels: [\"Do_Not_Amplify\"], ttl_msec: {ttl_msec} }}\n"
            );
            assert!(matches!(
                CompiledRules::from_yaml(&yaml),
                Err(RuleCompileError::InvalidRecommendationRestrictionTtl {
                    id,
                    label,
                    ttl_msec: actual,
                    max_ttl_msec: MAX_RECOMMENDATION_RESTRICTION_TTL_MSEC,
                }) if id == "invalid_expiry" && label == "Do_Not_Amplify" && actual == ttl_msec
            ));
        }
    }

    #[test]
    fn recommendation_restriction_accepts_max_ttl_in_nested_action() {
        let yaml = format!(
            "rules:\n  - id: reviewed_expiry\n    when: \"true\"\n    then:\n      kind: act_all\n      actions:\n        - {{ kind: act_add_labels_v2, labels: [\"AbusiveHighRecall\"], ttl_msec: {} }}\n        - {{ kind: act_captcha }}\n",
            MAX_RECOMMENDATION_RESTRICTION_TTL_MSEC
        );
        CompiledRules::from_yaml(&yaml).expect("bounded recommendation restriction should compile");
    }

    #[test]
    fn unrelated_label_may_keep_existing_no_expiry_behavior() {
        let yaml = r#"
rules:
  - id: unrelated
    when: "true"
    then: { kind: act_add_labels_v2, labels: ["MockLabel"] }
"#;
        CompiledRules::from_yaml(yaml).expect("unrelated label should remain backward compatible");
    }

    #[test]
    fn act_all_outcome_parses_into_ordered_multi_action_decision() {
        let rules = one_rule_pipeline(
            r#"
rules:
  - id: composite_test
    when: "true"
    then:
      kind: act_all
      actions:
        - kind: act_add_labels_v2
          labels: ["MockLabel"]
          ttl_msec: 100
        - kind: act_captcha
"#,
        );
        let f = base_facts();
        let decision = decide_with(&rules, &f).expect("eval should succeed");
        assert_eq!(
            decision,
            Decision::Act(vec![
                ActionSpec::AddLabelsV2 {
                    labels: vec!["MockLabel".into()],
                    ttl_msec: Some(100),
                },
                ActionSpec::Captcha,
            ]),
        );
    }


            fn facts_with_requested_action() -> Facts {
        let mut f = base_facts();
        f.score.requested_actions = vec![crate::facts::RequestedActionFacts {
            kind: "suspend".into(),
            perm: false,
            policy: "PlatformManipulation".into(),
            labels: vec!["SpamHighRecall".into()],
            ttl_msec: 1000,
            head: "IsSpammer".into(),
        }];
        f.score.policy_version = "3".into();
        f
    }

    #[test]
    fn act_requested_actions_outcome_parses_and_maps_to_decision() {
        let rules = one_rule_pipeline(
            r#"
rules:
  - id: generic_test
    when: "size(score.requested_actions) > 0"
    then:
      kind: act_requested_actions
  - id: fallthrough
    when: "true"
    then: {kind: skip, reason: no_requested_actions}
"#,
        );
        assert_eq!(
            decide_with(&rules, &base_facts()).expect("eval should succeed"),
            Decision::Skip("no_requested_actions".into())
        );
        assert_eq!(
            decide_with(&rules, &facts_with_requested_action()).expect("eval should succeed"),
            Decision::ActRequestedActions
        );
    }

    #[test]
    fn requested_actions_entries_and_policy_version_are_cel_readable() {
        let rules = one_rule_pipeline(
            r#"
rules:
  - id: entry_fields
    when: >
      score.policy_version == "3"
      && score.requested_actions.exists(a,
           a.kind == "suspend"
           && a.policy == "PlatformManipulation"
           && !a.perm
           && a.ttl_msec == 1000
           && a.head == "IsSpammer"
           && "SpamHighRecall" in a.labels)
    then: {kind: skip, reason: matched}
  - id: fallthrough
    when: "true"
    then: {kind: skip, reason: unmatched}
"#,
        );
        assert_eq!(
            decide_with(&rules, &facts_with_requested_action()).expect("eval should succeed"),
            Decision::Skip("matched".into())
        );
        assert_eq!(
            decide_with(&rules, &base_facts()).expect("eval should succeed"),
            Decision::Skip("unmatched".into())
        );
    }

    #[test]
    fn act_requested_actions_is_rejected_inside_act_all() {
        let yaml = "rules:\n  - id: x\n    when: \"true\"\n    then: {kind: act_all, actions: [{kind: act_requested_actions}]}\n";
        let err = CompiledRules::from_yaml(yaml).unwrap_err();
        assert!(matches!(err, RuleCompileError::Yaml(_)), "{err:?}");
    }


            fn rule_index(ids: &[String], id: &str) -> usize {
        ids.iter()
            .position(|r| r == id)
            .unwrap_or_else(|| panic!("rule {id:?} missing; got {ids:?}"))
    }

    #[test]
    fn baked_in_generic_rule_sits_directly_above_each_terminal_rule() {
        let cache = RulesCache::new();
        let user_ids = cache.status(EntityType::User, None).rule_ids;
        let generic = rule_index(&user_ids, "act_requested_actions");
        let guard = rule_index(&user_ids, "platform_row_without_requested_actions");
        let terminal = rule_index(&user_ids, "act_suspend");
        assert_eq!(terminal, user_ids.len() - 1, "act_suspend must be last");
        assert_eq!(
            generic,
            terminal - 1,
            "generic rule must sit directly above act_suspend"
        );
        assert_eq!(
            guard,
            generic - 1,
            "empty-list guard must sit directly above the generic rule"
        );
        assert!(generic > rule_index(&user_ids, "pagerank_skipped"));

        let post_ids = cache.status(EntityType::Post, None).rule_ids;
        let generic = rule_index(&post_ids, "act_requested_actions");
        let terminal = rule_index(&post_ids, "post_no_actionable_label");
        assert_eq!(
            terminal,
            post_ids.len() - 1,
            "post terminal skip must be last"
        );
        assert_eq!(
            generic,
            terminal - 1,
            "generic rule must sit directly above the post terminal skip"
        );
        assert!(generic > rule_index(&post_ids, "pagerank_skipped"));
    }

    #[test]
    fn baked_in_user_pipeline_routes_requested_actions_to_generic_dispatch() {
        let rules = RulesCache::new().resolve(EntityType::User, None);
        assert_eq!(
            decide_with(&rules, &facts_with_requested_action()).expect("eval should succeed"),
            Decision::ActRequestedActions
        );
        assert_eq!(
            decide_with(&rules, &base_facts()).expect("eval should succeed"),
            Decision::Act(vec![ActionSpec::SuspendUser {
                perm: false,
                policy: "PlatformManipulation".into(),
            }])
        );
    }

    #[test]
    fn baked_in_user_platform_label_with_empty_list_skips_not_suspends() {
        let rules = RulesCache::new().resolve(EntityType::User, None);
        let mut f = base_facts();
        f.score
            .labels
            .push("abuse_platform_requested_actions".into());
        assert_eq!(
            decide_with(&rules, &f).expect("eval should succeed"),
            Decision::Skip("platform_row_without_requested_actions".into())
        );
        let mut f = facts_with_requested_action();
        f.score
            .labels
            .push("abuse_platform_requested_actions".into());
        assert_eq!(
            decide_with(&rules, &f).expect("eval should succeed"),
            Decision::ActRequestedActions
        );
    }

    #[test]
    fn baked_in_user_guardrails_still_win_over_generic_dispatch() {
        let rules = RulesCache::new().resolve(EntityType::User, None);
        let mut f = facts_with_requested_action();
        f.user_allowlist_mut().is_allowlisted = true;
        assert_eq!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("user_in_allowlist".into())
        );
        let mut f = facts_with_requested_action();
        f.cred_mut().follower_count = Some(5_000);
        assert_eq!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("high_follower_count".into())
        );
    }

    #[test]
    fn baked_in_user_pipeline_splits_follower_bands_at_the_ceiling() {
        let rules = RulesCache::new().resolve(EntityType::User, None);
        let mut f = facts_with_requested_action();
        f.cred_mut().follower_count = Some(250_000);
        assert_eq!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("very_high_follower_count".into())
        );
        f.cred_mut().follower_count = Some(50_000);
        assert_eq!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("high_follower_count".into())
        );
        f.cred_mut().follower_count = Some(500);
        assert_ne!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("high_follower_count".into())
        );
        assert_ne!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("very_high_follower_count".into())
        );
    }

    #[test]
    fn baked_in_user_ceiling_ignores_skip_author_credibility_prechecks() {
        let rules = RulesCache::new().resolve(EntityType::User, None);
        let mut f = facts_with_requested_action();
        f.cred_mut().follower_count = Some(250_000);
        f.score.skip_author_credibility_prechecks = true;
        assert_eq!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("very_high_follower_count".into())
        );
    }

    #[test]
    fn baked_in_post_pipeline_routes_requested_actions_to_generic_dispatch() {
        let rules = RulesCache::new().resolve(EntityType::Post, None);
        let mut f = post_facts(safe_author(), vec![]);
        f.score.requested_actions = vec![crate::facts::RequestedActionFacts {
            kind: "post_label".into(),
            labels: vec!["SpamHighRecall".into()],
            head: "IsSpamPost".into(),
            ..Default::default()
        }];
        assert_eq!(
            decide_with(&rules, &f).expect("eval should succeed"),
            Decision::ActRequestedActions
        );
        assert_eq!(
            decide_with(&rules, &post_facts(safe_author(), vec![])).unwrap(),
            Decision::Skip("post_no_actionable_label".into())
        );
        let mut author = safe_author();
        author.cred.is_high = Some(true);
        let mut f = post_facts(author, vec![]);
        f.score.requested_actions = vec![crate::facts::RequestedActionFacts {
            kind: "post_label".into(),
            labels: vec!["SpamHighRecall".into()],
            ..Default::default()
        }];
        assert_eq!(
            decide_with(&rules, &f).unwrap(),
            Decision::Skip("pagerank_skipped".into())
        );
    }


    #[test]
    fn mock_user_allowlisted_skips() {
        let mut f = base_facts();
        f.user_allowlist_mut().is_allowlisted = true;
        assert_eq!(d(&f), Decision::Skip("allowlisted".into()));
    }

    #[test]
    fn mock_user_allowlist_wins_over_later_rules() {
        let mut f = base_facts();
        f.user_allowlist_mut().is_allowlisted = true;
        f.cred_mut().is_high = Some(true);
        assert_eq!(d(&f), Decision::Skip("allowlisted".into()));
    }

    #[test]
    fn mock_user_not_present_skips() {
        let mut f = base_facts();
        f.user_mut().present = false;
        assert_eq!(d(&f), Decision::Skip("not_present".into()));
    }

    #[test]
    fn mock_cred_is_high_skips() {
        let mut f = base_facts();
        f.cred_mut().is_high = Some(true);
        assert_eq!(d(&f), Decision::Skip("cred_skip".into()));
    }

    #[test]
    fn mock_cred_follower_count_skips() {
        let mut f = base_facts();
        f.cred_mut().follower_count = Some(1234); 
        assert_eq!(d(&f), Decision::Skip("cred_skip".into()));
    }

    #[test]
    fn mock_cred_score_skips() {
        let mut f = base_facts();
        f.cred_mut().score = Some(7.5); 
        assert_eq!(d(&f), Decision::Skip("cred_skip".into()));
    }

    #[test]
    fn rule_referencing_unbound_variable_compiles_but_eval_errors() {
        let yaml = "for_entity: user\nrules:\n  - id: stray_ads_rule\n    when: 'ads.accounts.exists(a, a.dormant_status == \"Active\")'\n    then: {kind: skip, reason: ads_skip}\n  - id: fallthrough\n    when: \"true\"\n    then: {kind: skip, reason: t}\n";
        let compiled = CompiledRules::from_yaml(yaml)
            .expect("unbound variables must NOT be caught at compile time");
        let err = decide_with(&compiled, &base_facts())
            .expect_err("evaluating an unbound variable must error");
        assert!(matches!(err, EvalError::Exec { ref id, .. } if id == "stray_ads_rule"));
    }

    #[test]
    fn mock_composite_label_yields_label_then_captcha() {
        let mut f = base_facts();
        f.score.labels = vec!["composite_label".into()];
        assert_eq!(
            d(&f),
            Decision::Act(vec![
                ActionSpec::AddLabelsV2 {
                    labels: vec!["MockLabel".into()],
                    ttl_msec: Some(1000),
                },
                ActionSpec::Captcha,
            ]),
        );
    }

    #[test]
    fn mock_label_only_yields_single_add_label() {
        let mut f = base_facts();
        f.score.labels = vec!["label_only".into()];
        assert_eq!(
            d(&f),
            Decision::Act(vec![ActionSpec::AddLabelsV2 {
                labels: vec!["MockLabel".into()],
                ttl_msec: Some(1000),
            }]),
        );
    }

    #[test]
    fn mock_catch_all_suspends() {
        let f = base_facts();
        assert_eq!(
            act_one(d(&f)),
            ActionSpec::SuspendUser {
                perm: false,
                policy: "TestPolicy".into(),
            }
        );
    }


    #[test]
    fn mock_post_allowlisted_skips() {
        let post = with_post_allowlisted(post_facts(safe_author(), vec![]));
        assert_eq!(d(&post), Decision::Skip("post_allowlisted".into()));

        let mut author = post_facts(safe_author(), vec![]);
        author.user_allowlist_mut().is_allowlisted = true;
        assert_eq!(d(&author), Decision::Skip("post_allowlisted".into()));
    }

    #[test]
    fn mock_post_author_cred_skips() {
        let mut author = safe_author();
        author.cred.is_high = Some(true);
        let f = post_facts(author, vec![]);
        assert_eq!(d(&f), Decision::Skip("author_cred_skip".into()));
    }

    #[test]
    fn mock_post_label_acts_on_post() {
        let f = with_post_labels(post_facts(safe_author(), vec![]), vec!["post_label".into()]);
        assert_eq!(
            d(&f),
            Decision::Act(vec![ActionSpec::AddPostLabelsV2 {
                labels: vec!["SomeLabel".into()],
                ttl_msec: None,
            }]),
        );
    }

    #[test]
    fn mock_post_catch_all_skips() {
        let f = post_facts(safe_author(), vec![]);
        assert_eq!(d(&f), Decision::Skip("post_no_action".into()));
    }
}
